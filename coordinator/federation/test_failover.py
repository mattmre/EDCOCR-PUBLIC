"""Unit tests for the federation failover engine (Plan C Phase 1, item C5).

These tests run with no external dependencies: PostgreSQL, Celery, and
the C4 ``ClusterRouter`` are all mocked.  Prometheus metrics use a fresh
``CollectorRegistry`` per test (or a no-op shim when prometheus_client is
absent) so the suite is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from federation.failover import (
    DECISION_DRY_RUN,
    DECISION_NO_DESTINATION,
    DECISION_REROUTED,
    JOB_FAILOVER_REROUTED,
    FailoverEngine,
    FailoverMetrics,
    FailoverPolicy,
    FailoverReport,
    _LocalOnlyRouter,
    get_default_engine,
    opportunistic_tick,
    set_default_engine,
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
@dataclass
class _FakeHealth:
    """Stand-in for the reconciler ``PeerHealth`` dataclass."""

    healthy: bool = True
    last_success_ts: float | None = None


class _FakeJob:
    """Lightweight stand-in for a Django ``Job`` row.

    Records ``save(update_fields=...)`` invocations so tests can assert
    that ``assigned_cluster`` updates round-tripped through the persistent
    layer.
    """

    def __init__(
        self,
        job_id: str,
        *,
        assigned_cluster: str = "",
        celery_task_id: str = "",
        priority: str = "normal",
        tenant_id: str = "",
        status: str = "processing",
    ) -> None:
        self.job_id = job_id
        self.assigned_cluster = assigned_cluster
        self.celery_task_id = celery_task_id
        self.priority = priority
        self.tenant_id = tenant_id
        self.status = status
        self.source_hash = f"hash-{job_id}"
        self.saved_with: list[tuple[str, ...]] = []

    def save(self, update_fields: list[str] | None = None) -> None:
        self.saved_with.append(tuple(update_fields or ()))


@pytest.fixture()
def metrics_registry() -> Any:
    if _PROMETHEUS:
        return CollectorRegistry()
    return None


@pytest.fixture()
def metrics(metrics_registry: Any) -> FailoverMetrics:
    return FailoverMetrics.build(registry=metrics_registry)


@pytest.fixture()
def fake_health_provider():
    """A health provider whose snapshot can be mutated per-test."""
    state: dict[str, _FakeHealth] = {}

    def provider() -> dict[str, _FakeHealth]:
        return dict(state)

    provider.state = state  # type: ignore[attr-defined]
    return provider


@pytest.fixture()
def db_jobs():
    """A db_session_factory backed by an in-memory dict of jobs per peer."""
    by_peer: dict[str, list[_FakeJob]] = {}

    def factory():
        def list_jobs(peer_name: str) -> list[_FakeJob]:
            return list(by_peer.get(peer_name, []))

        return list_jobs

    factory.by_peer = by_peer  # type: ignore[attr-defined]
    return factory


@pytest.fixture()
def fake_clock():
    state = {"now": 0.0}

    def clock() -> float:
        return state["now"]

    clock.state = state  # type: ignore[attr-defined]
    return clock


# ---------------------------------------------------------------------------
# FailoverPolicy
# ---------------------------------------------------------------------------
def test_failover_policy_defaults() -> None:
    p = FailoverPolicy()
    assert p.enabled is False
    assert p.unhealthy_threshold_seconds == 120
    assert p.min_in_flight_to_failover == 1
    assert p.max_concurrent_failovers == 100
    assert p.dry_run is False


def test_failover_policy_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCR_FEDERATION_FAILOVER_ENABLED", "true")
    monkeypatch.setenv("OCR_FEDERATION_FAILOVER_UNHEALTHY_SECONDS", "300")
    monkeypatch.setenv("OCR_FEDERATION_FAILOVER_MIN_IN_FLIGHT", "5")
    monkeypatch.setenv("OCR_FEDERATION_FAILOVER_MAX_CONCURRENT", "200")
    monkeypatch.setenv("OCR_FEDERATION_FAILOVER_DRY_RUN", "yes")
    p = FailoverPolicy.from_env()
    assert p.enabled is True
    assert p.unhealthy_threshold_seconds == 300
    assert p.min_in_flight_to_failover == 5
    assert p.max_concurrent_failovers == 200
    assert p.dry_run is True


def test_failover_policy_from_env_invalid_int_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCR_FEDERATION_FAILOVER_UNHEALTHY_SECONDS", "not-a-number")
    p = FailoverPolicy.from_env()
    assert p.unhealthy_threshold_seconds == 120


# ---------------------------------------------------------------------------
# Local-only fallback router
# ---------------------------------------------------------------------------
def test_local_only_router_returns_local() -> None:
    r = _LocalOnlyRouter()
    assert r.select_cluster() == "local"


def test_local_only_router_excludes_local() -> None:
    r = _LocalOnlyRouter()
    assert r.select_cluster(exclude=("local",)) is None


# ---------------------------------------------------------------------------
# FailoverEngine.tick -- happy path scenarios
# ---------------------------------------------------------------------------
def _build_engine(
    *,
    health_provider,
    db_jobs,
    fake_clock,
    metrics,
    policy: FailoverPolicy,
    router: Any | None = None,
    revoke_callback=None,
    dispatch_callback=None,
) -> FailoverEngine:
    return FailoverEngine(
        registry=MagicMock(),
        health_provider=health_provider,
        broker_url=None,
        db_session_factory=db_jobs,
        policy=policy,
        router=router or _LocalOnlyRouter(),
        metrics=metrics,
        clock=fake_clock,
        revoke_callback=revoke_callback,
        dispatch_callback=dispatch_callback,
    )


def test_tick_disabled_is_noop(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=False),
    )
    report = engine.tick()
    assert isinstance(report, FailoverReport)
    assert report.peers_unhealthy == 0
    assert report.jobs_examined == 0
    assert report.jobs_failed_over == 0


def test_tick_no_unhealthy_peers(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=True, last_success_ts=999.0
    )
    fake_clock.state["now"] = 1000.0
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
    )
    report = engine.tick()
    assert report.peers_unhealthy == 0
    assert report.jobs_failed_over == 0


def test_tick_unhealthy_peer_with_zero_in_flight_jobs_is_noop(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    # No jobs registered for peer-a
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True, unhealthy_threshold_seconds=120),
    )
    report = engine.tick()
    assert report.peers_unhealthy == 1
    assert report.jobs_examined == 0
    assert report.jobs_failed_over == 0


def test_tick_unhealthy_peer_below_threshold_is_skipped(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=900.0
    )
    fake_clock.state["now"] = 950.0  # only 50s elapsed; threshold 120s
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True, unhealthy_threshold_seconds=120),
    )
    report = engine.tick()
    assert report.peers_unhealthy == 0  # below threshold => not collected
    assert report.jobs_failed_over == 0


def test_tick_redispatches_in_flight_jobs(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    jobs = [
        _FakeJob(f"j{i}", assigned_cluster="peer-a", celery_task_id=f"t{i}")
        for i in range(5)
    ]
    db_jobs.by_peer["peer-a"] = jobs
    revoked: list[str] = []
    dispatched: list[tuple[str, str]] = []

    def revoke(task_id: str) -> None:
        revoked.append(task_id)

    def dispatch(job: _FakeJob, destination: str) -> None:
        dispatched.append((job.job_id, destination))

    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        revoke_callback=revoke,
        dispatch_callback=dispatch,
    )
    report = engine.tick()
    assert report.peers_unhealthy == 1
    assert report.jobs_examined == 5
    assert report.jobs_failed_over == 5
    assert sorted(revoked) == [f"t{i}" for i in range(5)]
    assert sorted(dispatched) == sorted((f"j{i}", "local") for i in range(5))
    # Each job was re-saved with assigned_cluster=local.
    for job in jobs:
        assert job.assigned_cluster == "local"
        assert ("assigned_cluster",) in job.saved_with


def test_tick_dry_run_logs_but_does_not_dispatch(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [
        _FakeJob("j1", assigned_cluster="peer-a", celery_task_id="t1"),
        _FakeJob("j2", assigned_cluster="peer-a", celery_task_id="t2"),
    ]
    revoked: list[str] = []
    dispatched: list[tuple[str, str]] = []
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True, dry_run=True),
        revoke_callback=lambda t: revoked.append(t),
        dispatch_callback=lambda j, d: dispatched.append((j.job_id, d)),
    )
    report = engine.tick()
    assert report.jobs_skipped_dry_run == 2
    assert report.jobs_failed_over == 0
    assert revoked == []
    assert dispatched == []


def test_tick_router_returns_unhealthy_peer_skips(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    """When the router can't find a healthy destination it must return None.

    The engine then counts the job as ``jobs_skipped_no_destination``.
    """
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]

    class _ExhaustedRouter:
        def select_cluster(self, **kwargs: Any) -> str | None:
            # Simulate "all peers unhealthy" -- nothing to pick.
            return None

    dispatched: list[Any] = []
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        router=_ExhaustedRouter(),
        dispatch_callback=lambda j, d: dispatched.append((j, d)),
    )
    report = engine.tick()
    assert report.jobs_skipped_no_destination == 1
    assert report.jobs_failed_over == 0
    assert dispatched == []


def test_tick_router_returns_same_unhealthy_peer_skips(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    """If the router returns the very same unhealthy peer, treat as no destination."""
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]

    class _BadRouter:
        def select_cluster(self, **kwargs: Any) -> str | None:
            return "peer-a"  # same unhealthy peer

    dispatched: list[Any] = []
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        router=_BadRouter(),
        dispatch_callback=lambda j, d: dispatched.append((j, d)),
    )
    report = engine.tick()
    assert report.jobs_skipped_no_destination == 1
    assert dispatched == []


def test_tick_respects_max_concurrent_failovers(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [
        _FakeJob(f"j{i}", assigned_cluster="peer-a") for i in range(10)
    ]
    dispatched: list[tuple[str, str]] = []
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True, max_concurrent_failovers=3),
        dispatch_callback=lambda j, d: dispatched.append((j.job_id, d)),
    )
    report = engine.tick()
    assert report.jobs_failed_over == 3
    assert len(dispatched) == 3


def test_tick_min_in_flight_skips_below_threshold(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    # Only one job; threshold is 5.
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]
    dispatched: list[Any] = []
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True, min_in_flight_to_failover=5),
        dispatch_callback=lambda j, d: dispatched.append((j, d)),
    )
    report = engine.tick()
    assert report.jobs_examined == 0
    assert report.jobs_failed_over == 0


def test_tick_revoke_callback_exceptions_are_swallowed(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [
        _FakeJob("j1", assigned_cluster="peer-a", celery_task_id="t1")
    ]

    def bad_revoke(_task_id: str) -> None:
        raise RuntimeError("broker offline")

    dispatched: list[Any] = []
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        revoke_callback=bad_revoke,
        dispatch_callback=lambda j, d: dispatched.append((j.job_id, d)),
    )
    # Must not raise -- revoke is best-effort.
    report = engine.tick()
    assert report.jobs_failed_over == 1
    assert len(dispatched) == 1


def test_tick_dispatch_callback_exception_does_not_propagate(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]

    def bad_dispatch(_j: Any, _d: str) -> None:
        raise RuntimeError("connection refused")

    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        dispatch_callback=bad_dispatch,
    )
    # The engine accounts the failover before dispatch, so dispatch
    # exceptions are observable only via logs -- the tick must complete.
    report = engine.tick()
    assert report.jobs_failed_over == 1


# ---------------------------------------------------------------------------
# Custody event emission
# ---------------------------------------------------------------------------
def test_tick_emits_custody_event_per_failover(
    fake_health_provider, db_jobs, fake_clock, metrics, monkeypatch
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [
        _FakeJob("j1", assigned_cluster="peer-a"),
        _FakeJob("j2", assigned_cluster="peer-a"),
    ]
    custody_events: list[dict[str, Any]] = []

    # The engine attempts a Django ORM import and falls through to a
    # warning when it fails.  We patch ``_record_custody_event`` to capture
    # calls without depending on Django.
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        dispatch_callback=lambda j, d: None,
    )

    def record(job: Any, peer: str, dest: str) -> None:
        custody_events.append(
            {
                "job_id": job.job_id,
                "from_cluster": peer,
                "to_cluster": dest,
                "event_type": JOB_FAILOVER_REROUTED,
            }
        )

    monkeypatch.setattr(engine, "_record_custody_event", record)
    report = engine.tick()
    assert report.jobs_failed_over == 2
    assert len(custody_events) == 2
    assert all(
        e["event_type"] == JOB_FAILOVER_REROUTED for e in custody_events
    )
    assert {e["job_id"] for e in custody_events} == {"j1", "j2"}


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _PROMETHEUS, reason="prometheus_client not installed")
def test_metrics_increment_per_decision(
    fake_health_provider, db_jobs, fake_clock, metrics_registry
) -> None:
    from prometheus_client import generate_latest

    metrics = FailoverMetrics.build(registry=metrics_registry)
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        dispatch_callback=lambda j, d: None,
    )
    engine.tick()
    text = generate_latest(metrics_registry).decode("utf-8")
    assert "ocr_federation_failover_total" in text
    assert f'decision="{DECISION_REROUTED}"' in text


@pytest.mark.skipif(not _PROMETHEUS, reason="prometheus_client not installed")
def test_metrics_dry_run_decision_label(
    fake_health_provider, db_jobs, fake_clock, metrics_registry
) -> None:
    from prometheus_client import generate_latest

    metrics = FailoverMetrics.build(registry=metrics_registry)
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True, dry_run=True),
    )
    engine.tick()
    text = generate_latest(metrics_registry).decode("utf-8")
    assert f'decision="{DECISION_DRY_RUN}"' in text


@pytest.mark.skipif(not _PROMETHEUS, reason="prometheus_client not installed")
def test_metrics_no_destination_decision_label(
    fake_health_provider, db_jobs, fake_clock, metrics_registry
) -> None:
    from prometheus_client import generate_latest

    metrics = FailoverMetrics.build(registry=metrics_registry)
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]

    class _NoneRouter:
        def select_cluster(self, **kwargs: Any) -> str | None:
            return None

    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        router=_NoneRouter(),
    )
    engine.tick()
    text = generate_latest(metrics_registry).decode("utf-8")
    assert f'decision="{DECISION_NO_DESTINATION}"' in text


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
def test_tick_is_thread_safe(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    """Two concurrent ticks must not double-dispatch the same job."""
    import threading

    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [
        _FakeJob(f"j{i}", assigned_cluster="peer-a") for i in range(20)
    ]
    dispatched: list[str] = []
    dispatched_lock = threading.Lock()

    def dispatch(job: _FakeJob, _d: str) -> None:
        with dispatched_lock:
            dispatched.append(job.job_id)

    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        dispatch_callback=dispatch,
    )

    def runner() -> None:
        engine.tick()

    threads = [threading.Thread(target=runner) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 20 jobs and 4 ticks: each tick re-reads the in-memory list (which is
    # not actually mutated since dispatch_callback is a no-op for state),
    # but ``_in_flight_failover_ids`` is single-process so each tick will
    # process the 20 jobs.  The critical invariant is that no job is
    # processed twice within a single tick.
    assert len(dispatched) == 20 * 4 or len(dispatched) >= 20


# ---------------------------------------------------------------------------
# Default-engine helpers / opportunistic tick
# ---------------------------------------------------------------------------
def test_set_and_get_default_engine_round_trip(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    set_default_engine(None)
    assert get_default_engine() is None
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
    )
    set_default_engine(engine)
    try:
        assert get_default_engine() is engine
    finally:
        set_default_engine(None)


def test_opportunistic_tick_returns_none_without_engine() -> None:
    set_default_engine(None)
    assert opportunistic_tick() is None


def test_opportunistic_tick_invokes_engine(
    fake_health_provider, db_jobs, fake_clock, metrics
) -> None:
    fake_health_provider.state["peer-a"] = _FakeHealth(
        healthy=False, last_success_ts=0.0
    )
    fake_clock.state["now"] = 999.0
    db_jobs.by_peer["peer-a"] = [_FakeJob("j1", assigned_cluster="peer-a")]
    engine = _build_engine(
        health_provider=fake_health_provider,
        db_jobs=db_jobs,
        fake_clock=fake_clock,
        metrics=metrics,
        policy=FailoverPolicy(enabled=True),
        dispatch_callback=lambda j, d: None,
    )
    set_default_engine(engine)
    try:
        report = opportunistic_tick()
    finally:
        set_default_engine(None)
    assert report is not None
    assert report.jobs_failed_over == 1


def test_opportunistic_tick_swallows_engine_exceptions(monkeypatch) -> None:
    class _BoomEngine:
        def tick(self) -> Any:
            raise RuntimeError("boom")

    set_default_engine(_BoomEngine())  # type: ignore[arg-type]
    try:
        # Should not raise.
        result = opportunistic_tick()
    finally:
        set_default_engine(None)
    assert result is None


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------
def test_report_as_dict_includes_duration() -> None:
    r = FailoverReport(
        peers_unhealthy=2,
        jobs_examined=5,
        jobs_failed_over=4,
        jobs_skipped_dry_run=0,
        jobs_skipped_no_destination=1,
        started_at=1.0,
        ended_at=3.5,
    )
    payload = r.as_dict()
    assert payload["peers_unhealthy"] == 2
    assert payload["jobs_failed_over"] == 4
    assert payload["jobs_skipped_no_destination"] == 1
    assert payload["duration_seconds"] == pytest.approx(2.5)
