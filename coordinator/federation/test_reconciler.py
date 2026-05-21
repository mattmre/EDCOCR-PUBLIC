"""Unit tests for the federation reconciler (Plan C Phase 1, item C2).

These tests run with no external dependencies: HTTP and the kubernetes
client are both mocked. The Prometheus metrics surface uses a fresh
``CollectorRegistry`` per test (when prometheus_client is installed) and a
no-op shim otherwise so the suite passes on minimal Python interpreters.
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from federation.reconciler import (
    Metrics,
    PeerCluster,
    PeerHealth,
    Reconciler,
    Registry,
    _policy_body,
    _policy_name,
    _probe_management_api,
    _probe_queue_metrics,
    load_registry_from_path,
)

try:
    from prometheus_client import CollectorRegistry  # type: ignore[import-not-found]

    _PROMETHEUS = True
except ImportError:  # pragma: no cover - environment-dependent
    CollectorRegistry = None  # type: ignore[assignment,misc]
    _PROMETHEUS = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_REGISTRY_JSON = {
    "version": 1,
    "self": {"name": "prod-us-east-1"},
    "queue_classes": ["ocr_gpu", "ocr_cpu", "nlp", "translation_batch"],
    "healthcheck_interval_seconds": 30,
    "unhealthy_threshold_seconds": 60,
    "clusters": [
        {
            "name": "prod-eu-west-1",
            "region": "eu-west-1",
            "rabbitmq_uri": "amqps://user@rabbit.eu/%2F",
            "management_uri": "https://rabbit.eu:15671",
            "priority": 10,
            "tags": ["eu-residency"],
            "tls": {"enabled": True, "ca_secret_ref": "rabbit-ca"},
        },
        {
            "name": "airgap-gov-1",
            "region": "on-prem",
            "rabbitmq_uri": "amqps://user@rabbit.gov/%2F",
            "management_uri": "https://rabbit.gov:15671",
            "priority": 5,
            "tags": ["privileged"],
            "tls": {"enabled": True, "ca_secret_ref": "rabbit-gov-ca"},
        },
    ],
}


@pytest.fixture()
def metrics_registry() -> Any:
    if _PROMETHEUS:
        return CollectorRegistry()
    return None


@pytest.fixture()
def metrics(metrics_registry: Any) -> Metrics:
    return Metrics.build(registry=metrics_registry)


@pytest.fixture()
def sample_registry() -> Registry:
    return Registry.from_json(json.dumps(SAMPLE_REGISTRY_JSON))


# ---------------------------------------------------------------------------
# Registry parsing
# ---------------------------------------------------------------------------
def test_registry_parses_full_document(sample_registry: Registry) -> None:
    assert sample_registry.self_name == "prod-us-east-1"
    assert sample_registry.queue_classes == (
        "ocr_gpu",
        "ocr_cpu",
        "nlp",
        "translation_batch",
    )
    assert sample_registry.healthcheck_interval_seconds == 30
    assert sample_registry.unhealthy_threshold_seconds == 60
    assert [c.name for c in sample_registry.clusters] == [
        "prod-eu-west-1",
        "airgap-gov-1",
    ]
    eu = sample_registry.clusters[0]
    assert eu.priority == 10
    assert eu.tls_enabled is True
    assert eu.tls_ca_secret_ref == "rabbit-ca"
    assert eu.tags == ("eu-residency",)


def test_registry_handles_missing_optional_fields() -> None:
    registry = Registry.from_json(
        json.dumps(
            {
                "self": {"name": "self-1"},
                "queue_classes": [],
                "clusters": [{"name": "p1", "rabbitmq_uri": "amqps://x"}],
            }
        )
    )
    peer = registry.clusters[0]
    assert peer.region == ""
    assert peer.priority == 10
    assert peer.tls_enabled is False
    assert peer.tags == ()
    assert registry.healthcheck_interval_seconds == 30
    assert registry.unhealthy_threshold_seconds == 120


def test_load_registry_from_path(tmp_path: Any) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps(SAMPLE_REGISTRY_JSON), encoding="utf-8")
    registry = load_registry_from_path(str(path))
    assert len(registry.clusters) == 2


# ---------------------------------------------------------------------------
# Health probing
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status: int, body: dict[str, Any] | bytes = b"") -> None:
        self.status = status
        if isinstance(body, dict):
            body = json.dumps(body).encode("utf-8")
        self._buf = io.BytesIO(body)

    def read(self) -> bytes:
        return self._buf.read()

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def test_probe_management_api_success() -> None:
    def opener(_req: Any, timeout: float | None = None) -> _FakeResponse:
        del timeout
        return _FakeResponse(200, {"status": "ok"})

    healthy, payload = _probe_management_api(
        "https://rabbit.eu:15671", opener=opener
    )
    assert healthy is True
    assert payload == {"status": "ok"}


def test_probe_management_api_failure_status() -> None:
    def opener(_req: Any, timeout: float | None = None) -> _FakeResponse:
        del timeout
        return _FakeResponse(503)

    healthy, payload = _probe_management_api(
        "https://rabbit.eu:15671", opener=opener
    )
    assert healthy is False
    assert payload == {}


def test_probe_management_api_network_error() -> None:
    def opener(_req: Any, timeout: float | None = None) -> _FakeResponse:
        del timeout
        raise OSError("connection refused")

    healthy, _ = _probe_management_api(
        "https://rabbit.eu:15671", opener=opener
    )
    assert healthy is False


def test_probe_management_api_empty_uri_returns_unhealthy() -> None:
    healthy, _ = _probe_management_api("", opener=lambda *_a, **_k: _FakeResponse(200))
    assert healthy is False


def test_probe_queue_metrics_returns_payload() -> None:
    captured = {}

    def opener(req: Any, timeout: float | None = None) -> _FakeResponse:
        del timeout
        captured["url"] = req.full_url
        return _FakeResponse(200, {"messages_unacknowledged": 17})

    payload = _probe_queue_metrics(
        "https://rabbit.eu:15671/", "ocr_gpu", opener=opener
    )
    assert payload == {"messages_unacknowledged": 17}
    assert captured["url"].endswith("/api/queues/%2F/ocr_gpu")


# ---------------------------------------------------------------------------
# Policy body helpers
# ---------------------------------------------------------------------------
def test_policy_name_replaces_underscores() -> None:
    assert (
        _policy_name("ocr-local", "prod-eu-west-1", "translation_batch")
        == "ocr-local-fed-policy-prod-eu-west-1-translation-batch"
    )


def test_policy_body_shape() -> None:
    body = _policy_body(
        release="ocr-local",
        peer="prod-eu-west-1",
        queue_class="ocr_gpu",
        priority=10,
        namespace="ocr",
        max_overflow=5000,
    )
    assert body["apiVersion"] == "rabbitmq.com/v1beta1"
    assert body["kind"] == "Policy"
    assert body["metadata"]["name"].endswith("-prod-eu-west-1-ocr-gpu")
    assert body["metadata"]["namespace"] == "ocr"
    assert body["metadata"]["labels"]["federation.ocr-local.io/peer"] == "prod-eu-west-1"
    spec = body["spec"]
    assert spec["pattern"] == "^ocr_gpu.*$"
    assert spec["applyTo"] == "queues"
    assert spec["priority"] == 10
    assert spec["definition"]["federation-upstream"] == "prod-eu-west-1-upstream"
    assert spec["definition"]["max-length"] == 5000


# ---------------------------------------------------------------------------
# Reconciler integration
# ---------------------------------------------------------------------------
def _build_opener(
    *, alive_peers: set[str], queue_payload: dict[str, Any] | None = None
) -> Any:
    payload = queue_payload or {
        "messages_unacknowledged": 3,
        "message_stats": {"publish_in": 10, "publish_out": 7},
    }

    def opener(req: Any, timeout: float | None = None) -> _FakeResponse:
        del timeout
        url = req.full_url
        for peer in alive_peers:
            host = peer.replace("prod-", "").replace("airgap-", "")
            if host in url or peer in url:
                if "/aliveness-test/" in url:
                    return _FakeResponse(200, {"status": "ok"})
                if "/api/queues/" in url:
                    return _FakeResponse(200, payload)
        if "/aliveness-test/" in url:
            return _FakeResponse(503)
        return _FakeResponse(404)

    return opener


def test_reconcile_marks_healthy_peer_and_installs_policies(
    sample_registry: Registry, metrics: Metrics
) -> None:
    api = MagicMock()

    def opener(req: Any, timeout: float | None = None) -> _FakeResponse:
        del timeout
        if "/aliveness-test/" in req.full_url:
            return _FakeResponse(200, {"status": "ok"})
        if "/api/queues/" in req.full_url:
            return _FakeResponse(
                200,
                {
                    "messages_unacknowledged": 4,
                    "message_stats": {"publish_in": 10, "publish_out": 5},
                },
            )
        return _FakeResponse(404)

    reconciler = Reconciler(
        sample_registry,
        release_name="ocr-local",
        namespace="ocr",
        opener=opener,
        kubernetes_api=api,
        metrics=metrics,
    )
    health = reconciler.reconcile_once(now=1000.0)

    assert health["prod-eu-west-1"].healthy is True
    assert health["airgap-gov-1"].healthy is True
    # 2 peers x 4 queue classes => 8 policy creations on the first cycle.
    assert api.create_namespaced_custom_object.call_count == 8
    # All policy names follow the helper convention.
    created_names = {
        call.kwargs["body"]["metadata"]["name"]
        for call in api.create_namespaced_custom_object.call_args_list
    }
    assert "ocr-local-fed-policy-prod-eu-west-1-ocr-gpu" in created_names
    assert "ocr-local-fed-policy-airgap-gov-1-translation-batch" in created_names


def test_reconcile_unhealthy_peer_removes_policies_after_threshold(
    sample_registry: Registry, metrics: Metrics
) -> None:
    api = MagicMock()
    healthy_opener = MagicMock(
        side_effect=lambda req, timeout=None: _FakeResponse(200, {"status": "ok"})
        if "/aliveness-test/" in req.full_url
        else _FakeResponse(200, {})
    )
    reconciler = Reconciler(
        sample_registry,
        release_name="ocr-local",
        namespace="ocr",
        opener=healthy_opener,
        kubernetes_api=api,
        metrics=metrics,
    )
    reconciler.reconcile_once(now=0.0)
    assert api.create_namespaced_custom_object.call_count == 8

    # Switch to a fully-unhealthy opener and step time past the threshold.
    reconciler._opener = MagicMock(
        side_effect=lambda *_a, **_k: _FakeResponse(503)
    )
    reconciler.reconcile_once(now=10.0)
    # Within threshold (60s) policies remain installed.
    assert api.delete_namespaced_custom_object.call_count == 0

    reconciler.reconcile_once(now=200.0)
    # Past threshold, policies for both peers are removed (8 deletions).
    assert api.delete_namespaced_custom_object.call_count == 8
    deleted_names = {
        call.kwargs["name"]
        for call in api.delete_namespaced_custom_object.call_args_list
    }
    assert "ocr-local-fed-policy-prod-eu-west-1-nlp" in deleted_names


def test_reconcile_recovers_unhealthy_peer(
    sample_registry: Registry, metrics: Metrics
) -> None:
    api = MagicMock()
    state = {"healthy": True}

    def opener(req: Any, timeout: float | None = None) -> _FakeResponse:
        del timeout
        if "/aliveness-test/" in req.full_url:
            return _FakeResponse(200, {"status": "ok"}) if state["healthy"] else _FakeResponse(503)
        return _FakeResponse(200, {})

    reconciler = Reconciler(
        sample_registry,
        release_name="ocr-local",
        namespace="ocr",
        opener=opener,
        kubernetes_api=api,
        metrics=metrics,
    )
    reconciler.reconcile_once(now=0.0)
    initial_creates = api.create_namespaced_custom_object.call_count
    state["healthy"] = False
    reconciler.reconcile_once(now=500.0)
    assert api.delete_namespaced_custom_object.call_count == 8
    state["healthy"] = True
    reconciler.reconcile_once(now=600.0)
    # All 8 policies recreated after recovery.
    assert api.create_namespaced_custom_object.call_count == initial_creates + 8


def test_reconcile_without_kubernetes_api_is_observe_only(
    sample_registry: Registry, metrics: Metrics
) -> None:
    def opener(req: Any, timeout: float | None = None) -> _FakeResponse:
        del timeout
        if "/aliveness-test/" in req.full_url:
            return _FakeResponse(200, {"status": "ok"})
        return _FakeResponse(200, {})

    reconciler = Reconciler(
        sample_registry,
        release_name="ocr-local",
        namespace="ocr",
        opener=opener,
        kubernetes_api=None,
        metrics=metrics,
    )
    # Should run without raising, even though there is no API to apply to.
    reconciler.reconcile_once(now=10.0)
    assert reconciler._installed_policies  # tracked locally, just not applied


# ---------------------------------------------------------------------------
# Metrics smoke tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _PROMETHEUS, reason="prometheus_client not installed")
def test_metrics_register_distinct_collectors(metrics_registry: Any) -> None:
    metrics = Metrics.build(registry=metrics_registry)
    metrics.peer_up.labels(cluster="a").set(1)
    metrics.lag_seconds.labels(cluster="a", queue="ocr_gpu").set(2.5)
    # If registration was buggy the next call would raise "duplicated timeseries".
    metrics.peer_up.labels(cluster="a").set(0)


def test_peer_health_default() -> None:
    h = PeerHealth()
    assert h.healthy is True
    assert h.consecutive_failures == 0


def test_peer_cluster_from_dict_minimal() -> None:
    p = PeerCluster.from_dict({"name": "x", "rabbitmq_uri": "amqps://x"})
    assert p.name == "x"
    assert p.region == ""
    assert p.tls_enabled is False
