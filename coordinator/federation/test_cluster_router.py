"""Unit tests for the federation cluster router (Plan C Phase 1, item C4).

These tests run with no external dependencies: HTTP and the kubernetes
client are mocked out, the registry is fed from in-memory data, and the
Prometheus surface is exercised through ``CollectorRegistry`` when
``prometheus_client`` is available.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from federation.cluster_router import (
    ClusterEntry,
    ClusterRouter,
    ClusterRoutingDecision,
    _resolve_custom_callable,
    build_router_from_env,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
def _entry(
    name: str,
    *,
    region: str = "",
    priority: int = 10,
    tags: tuple[str, ...] = (),
    management_uri: str | None = None,
) -> ClusterEntry:
    # Default management_uri encodes the peer name into the host so the
    # load-aware tests can map URIs back to peer names without extra wiring.
    if management_uri is None:
        management_uri = f"https://{name}:15671"
    return ClusterEntry(
        name=name,
        region=region,
        rabbitmq_uri=f"amqps://user@{name}/%2F",
        management_uri=management_uri,
        priority=priority,
        tags=tags,
        tls_enabled=False,
    )


class _FakeRegistry:
    """Minimal stand-in for ClusterRegistry that satisfies ClusterRouter."""

    def __init__(
        self,
        entries: list[ClusterEntry],
        *,
        unhealthy: set[str] | None = None,
    ) -> None:
        self._entries = entries
        self._unhealthy = unhealthy or set()

    def clusters(self) -> list[ClusterEntry]:
        return list(self._entries)

    def is_peer_healthy(self, name: str) -> bool:
        return name not in self._unhealthy

    def self_name(self) -> str:
        return "local"

    def force_reload(self) -> None:
        return None


@pytest.fixture()
def three_peer_registry() -> _FakeRegistry:
    return _FakeRegistry(
        [
            _entry("us-east", region="us-east-1", priority=20),
            _entry("eu-central", region="eu-central-1", priority=10),
            _entry("airgap-gov", region="on-prem-gov", priority=5),
        ]
    )


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------
def test_routing_decision_is_frozen():
    d = ClusterRoutingDecision(
        cluster_name=None, queue_name="ocr_gpu", reason_code="local_default"
    )
    with pytest.raises(Exception):  # FrozenInstanceError is a dataclass
        d.cluster_name = "us-east"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Empty registry / disabled feature
# ---------------------------------------------------------------------------
def test_empty_registry_routes_local_regardless_of_strategy():
    empty = _FakeRegistry([])
    for strategy in (
        "local",
        "priority",
        "round_robin",
        "load_aware",
        "region_affinity",
        "custom",
    ):
        router = ClusterRouter(
            empty, strategy=strategy, local_cluster_name="local"
        )
        decision = router.select_cluster(
            task_name="jobs.tasks.process_document",
            base_queue="ocr_gpu",
            kwargs={},
        )
        assert decision.cluster_name is None
        assert decision.queue_name == "ocr_gpu"
        assert decision.reason_code == "local_default"


def test_unknown_strategy_falls_back_to_local(three_peer_registry):
    router = ClusterRouter(
        three_peer_registry,
        strategy="bogus",  # type: ignore[arg-type]
        local_cluster_name="local",
    )
    assert router.strategy == "local"
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name is None


# ---------------------------------------------------------------------------
# local strategy
# ---------------------------------------------------------------------------
def test_local_strategy_keeps_base_queue(three_peer_registry):
    router = ClusterRouter(
        three_peer_registry, strategy="local", local_cluster_name="local"
    )
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name is None
    assert decision.queue_name == "ocr_gpu"
    assert decision.reason_code == "local_default"


# ---------------------------------------------------------------------------
# priority strategy
# ---------------------------------------------------------------------------
def test_priority_strategy_picks_highest(three_peer_registry):
    router = ClusterRouter(
        three_peer_registry, strategy="priority", local_cluster_name="local"
    )
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name == "us-east"
    assert decision.queue_name == "ocr_gpu.us-east"
    assert decision.reason_code == "priority"
    assert decision.metadata["peer"] == "us-east"


def test_priority_strategy_breaks_ties_by_name():
    reg = _FakeRegistry(
        [
            _entry("zzz-east", priority=10),
            _entry("aaa-west", priority=10),
        ]
    )
    router = ClusterRouter(reg, strategy="priority", local_cluster_name="local")
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name == "aaa-west"


def test_priority_strategy_skips_unhealthy_peers():
    reg = _FakeRegistry(
        [
            _entry("us-east", priority=20),
            _entry("eu-central", priority=10),
        ],
        unhealthy={"us-east"},
    )
    router = ClusterRouter(reg, strategy="priority", local_cluster_name="local")
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name == "eu-central"


def test_no_healthy_peers_returns_local():
    reg = _FakeRegistry(
        [_entry("us-east"), _entry("eu-central")],
        unhealthy={"us-east", "eu-central"},
    )
    router = ClusterRouter(reg, strategy="priority", local_cluster_name="local")
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name is None
    assert decision.reason_code == "no_healthy_peers"


# ---------------------------------------------------------------------------
# round_robin strategy
# ---------------------------------------------------------------------------
def test_round_robin_rotates_across_peers(three_peer_registry):
    counter = itertools.count()

    def _next() -> int:
        return next(counter)

    router = ClusterRouter(
        three_peer_registry,
        strategy="round_robin",
        local_cluster_name="local",
        round_robin_state=_next,
    )
    seen = []
    for _ in range(6):
        d = router.select_cluster(
            task_name="t", base_queue="ocr_gpu", kwargs={}
        )
        seen.append(d.cluster_name)
    # Sorted-by-name peers are: airgap-gov, eu-central, us-east -> repeated
    assert seen == [
        "airgap-gov",
        "eu-central",
        "us-east",
        "airgap-gov",
        "eu-central",
        "us-east",
    ]


# ---------------------------------------------------------------------------
# load_aware strategy
# ---------------------------------------------------------------------------
def test_load_aware_picks_lowest_unacked():
    reg = _FakeRegistry(
        [_entry("a"), _entry("b"), _entry("c")]
    )
    loads = {"a": 50, "b": 5, "c": 30}

    def _probe(mgmt: str, queue: str) -> int | None:
        # Map management_uri back to peer name -- our fake URIs include the
        # peer name as the host portion.
        name = mgmt.split("//")[-1].split(":")[0]
        return loads.get(name)

    router = ClusterRouter(
        reg,
        strategy="load_aware",
        local_cluster_name="local",
        load_probe=_probe,
    )
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name == "b"
    assert decision.metadata["load"] == 5


def test_load_aware_caches_within_window():
    reg = _FakeRegistry([_entry("a"), _entry("b")])
    call_counter = {"a": 0, "b": 0}

    def _probe(mgmt: str, queue: str) -> int | None:
        name = mgmt.split("//")[-1].split(":")[0]
        call_counter[name] += 1
        return 100 if name == "a" else 1

    fake_now = [1000.0]
    router = ClusterRouter(
        reg,
        strategy="load_aware",
        local_cluster_name="local",
        load_probe=_probe,
        clock=lambda: fake_now[0],
        load_aware_cache_seconds=5.0,
    )
    router.select_cluster(task_name="t", base_queue="ocr_gpu", kwargs={})
    router.select_cluster(task_name="t", base_queue="ocr_gpu", kwargs={})
    # Within the cache window, only one probe per peer.
    assert call_counter == {"a": 1, "b": 1}
    # Advance past the window: probes happen again.
    fake_now[0] += 6.0
    router.select_cluster(task_name="t", base_queue="ocr_gpu", kwargs={})
    assert call_counter == {"a": 2, "b": 2}


def test_load_aware_unknown_loads_sort_last():
    reg = _FakeRegistry([_entry("a"), _entry("b")])

    def _probe(mgmt: str, queue: str) -> int | None:
        name = mgmt.split("//")[-1].split(":")[0]
        return 7 if name == "b" else None

    router = ClusterRouter(
        reg,
        strategy="load_aware",
        local_cluster_name="local",
        load_probe=_probe,
    )
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name == "b"


# ---------------------------------------------------------------------------
# region_affinity strategy
# ---------------------------------------------------------------------------
def test_region_affinity_prefers_matching_region(three_peer_registry):
    router = ClusterRouter(
        three_peer_registry,
        strategy="region_affinity",
        local_cluster_name="local",
    )
    decision = router.select_cluster(
        task_name="t",
        base_queue="ocr_gpu",
        kwargs={"origin_region": "eu-central-1"},
    )
    assert decision.cluster_name == "eu-central"


def test_region_affinity_falls_back_to_priority_when_no_match(three_peer_registry):
    router = ClusterRouter(
        three_peer_registry,
        strategy="region_affinity",
        local_cluster_name="local",
    )
    decision = router.select_cluster(
        task_name="t",
        base_queue="ocr_gpu",
        kwargs={"origin_region": "ap-southeast-2"},
    )
    # No match -> highest priority.
    assert decision.cluster_name == "us-east"


def test_region_affinity_with_no_origin_region_falls_back(three_peer_registry):
    router = ClusterRouter(
        three_peer_registry,
        strategy="region_affinity",
        local_cluster_name="local",
    )
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name == "us-east"


# ---------------------------------------------------------------------------
# custom strategy
# ---------------------------------------------------------------------------
def test_custom_callable_resolution_via_dotted_path():
    # pin to a known callable in the stdlib
    target = _resolve_custom_callable("os.path:join")
    assert callable(target)
    target = _resolve_custom_callable("os.path.join")
    assert callable(target)


def test_custom_callable_resolution_invalid_returns_none():
    assert _resolve_custom_callable("") is None
    assert _resolve_custom_callable("nonexistent") is None
    assert _resolve_custom_callable("nonexistent_module_xyz.attr") is None
    assert _resolve_custom_callable("os.path:not_a_real_attr") is None
    # symbol exists but is not callable
    assert _resolve_custom_callable("os:sep") is None


def test_custom_strategy_with_unresolvable_path_returns_local(three_peer_registry):
    router = ClusterRouter(
        three_peer_registry,
        strategy="custom",
        local_cluster_name="local",
        custom_callable="nonexistent.module:fn",
    )
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name is None
    assert decision.reason_code == "custom_unresolved"


def test_custom_strategy_string_return_is_promoted(three_peer_registry, monkeypatch):
    # Define a callable on a real importable module.
    import federation.cluster_router as cr

    def _picker(**kwargs: Any) -> str:
        return "eu-central"

    monkeypatch.setattr(cr, "_test_picker", _picker, raising=False)
    router = ClusterRouter(
        three_peer_registry,
        strategy="custom",
        local_cluster_name="local",
        custom_callable="federation.cluster_router:_test_picker",
    )
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name == "eu-central"
    assert decision.queue_name == "ocr_gpu.eu-central"
    assert decision.reason_code == "custom_picked"


def test_custom_strategy_returns_decision_directly(three_peer_registry, monkeypatch):
    import federation.cluster_router as cr

    def _picker(**kwargs: Any) -> ClusterRoutingDecision:
        return ClusterRoutingDecision(
            cluster_name="airgap-gov",
            queue_name="ocr_gpu.airgap-gov",
            reason_code="custom_explicit",
            metadata={"by": "test"},
        )

    monkeypatch.setattr(cr, "_test_picker_dec", _picker, raising=False)
    router = ClusterRouter(
        three_peer_registry,
        strategy="custom",
        local_cluster_name="local",
        custom_callable="federation.cluster_router:_test_picker_dec",
    )
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name == "airgap-gov"
    assert decision.reason_code == "custom_explicit"


def test_custom_strategy_none_return_routes_local(three_peer_registry, monkeypatch):
    import federation.cluster_router as cr

    def _picker(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(cr, "_test_picker_none", _picker, raising=False)
    router = ClusterRouter(
        three_peer_registry,
        strategy="custom",
        local_cluster_name="local",
        custom_callable="federation.cluster_router:_test_picker_none",
    )
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name is None
    assert decision.reason_code == "custom_local"


# ---------------------------------------------------------------------------
# Per-GPU + federation queue composition
# ---------------------------------------------------------------------------
def test_per_gpu_and_federation_compose(three_peer_registry):
    """``base_queue=ocr_gpu_2`` + peer ``eu-central`` -> ``ocr_gpu_2.eu-central``."""
    router = ClusterRouter(
        three_peer_registry,
        strategy="region_affinity",
        local_cluster_name="local",
    )
    decision = router.select_cluster(
        task_name="jobs.tasks.process_document",
        base_queue="ocr_gpu_2",
        kwargs={"origin_region": "eu-central-1"},
    )
    assert decision.cluster_name == "eu-central"
    assert decision.queue_name == "ocr_gpu_2.eu-central"


# ---------------------------------------------------------------------------
# Prometheus counter increments
# ---------------------------------------------------------------------------
def test_prometheus_counter_increments_for_each_decision(three_peer_registry):
    router_local = ClusterRouter(
        three_peer_registry, strategy="local", local_cluster_name="local"
    )
    router_priority = ClusterRouter(
        three_peer_registry, strategy="priority", local_cluster_name="local"
    )
    router_local.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    router_priority.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )

    # Read the counter via the global registry.
    try:
        from prometheus_client import REGISTRY  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("prometheus_client not installed")

    samples = []
    for metric in REGISTRY.collect():
        if metric.name in (
            "ocr_cluster_router_decisions",
            "ocr_cluster_router_decisions_total",
        ):
            samples.extend(metric.samples)
    # We should see at least one local + one peer label combination.
    by_label = {
        (s.labels.get("strategy"), s.labels.get("decision")): s.value
        for s in samples
        if s.name.endswith("_total")
    }
    assert by_label.get(("local", "local"), 0) >= 1
    assert by_label.get(("priority", "peer"), 0) >= 1


# ---------------------------------------------------------------------------
# build_router_from_env
# ---------------------------------------------------------------------------
def test_build_router_from_env_disabled(monkeypatch):
    monkeypatch.delenv("OCR_FEDERATION_ROUTING_ENABLED", raising=False)
    assert build_router_from_env() is None
    monkeypatch.setenv("OCR_FEDERATION_ROUTING_ENABLED", "false")
    assert build_router_from_env() is None


def test_build_router_from_env_enabled_constructs_router(monkeypatch, tmp_path):
    registry_file = tmp_path / "clusters.yaml"
    registry_file.write_text(
        "self:\n  name: local-test\nclusters:\n  - name: a\n    region: r1\n    priority: 5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OCR_FEDERATION_ROUTING_ENABLED", "true")
    monkeypatch.setenv("OCR_FEDERATION_ROUTING_STRATEGY", "priority")
    monkeypatch.setenv("OCR_FEDERATION_REGISTRY_PATH", str(registry_file))
    monkeypatch.setenv("OCR_CLUSTER_NAME", "local-test")

    router = build_router_from_env()
    assert router is not None
    assert router.strategy == "priority"
    decision = router.select_cluster(
        task_name="t", base_queue="ocr_gpu", kwargs={}
    )
    assert decision.cluster_name == "a"
    assert decision.queue_name == "ocr_gpu.a"


def test_build_router_from_env_invalid_strategy_falls_back(
    monkeypatch, tmp_path
):
    registry_file = tmp_path / "clusters.yaml"
    registry_file.write_text("self:\n  name: lt\nclusters: []\n", encoding="utf-8")
    monkeypatch.setenv("OCR_FEDERATION_ROUTING_ENABLED", "1")
    monkeypatch.setenv("OCR_FEDERATION_ROUTING_STRATEGY", "wrong-choice")
    monkeypatch.setenv("OCR_FEDERATION_REGISTRY_PATH", str(registry_file))

    router = build_router_from_env()
    assert router is not None
    assert router.strategy == "local"
