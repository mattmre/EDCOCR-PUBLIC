"""Integration tests for the reconciler -> failover-engine wiring (Plan C C5).

The C2 reconciler accepts an optional ``failover_engine`` parameter; when
set it must call ``failover_engine.tick()`` exactly once per
``reconcile_once`` cycle.  When absent (the default), it must not invoke
anything new.  These tests cover both branches without touching the real
failover-engine implementation.
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from federation.reconciler import (
    Metrics,
    Reconciler,
    Registry,
)

SAMPLE_REGISTRY_JSON = {
    "version": 1,
    "self": {"name": "prod-us-east-1"},
    "queue_classes": ["ocr_gpu"],
    "healthcheck_interval_seconds": 30,
    "unhealthy_threshold_seconds": 60,
    "clusters": [
        {
            "name": "prod-eu-west-1",
            "region": "eu-west-1",
            "rabbitmq_uri": "amqps://user@rabbit.eu/%2F",
            "management_uri": "https://rabbit.eu:15671",
            "priority": 10,
            "tls": {"enabled": True, "ca_secret_ref": "rabbit-ca"},
        },
    ],
}


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


@pytest.fixture()
def sample_registry() -> Registry:
    return Registry.from_json(json.dumps(SAMPLE_REGISTRY_JSON))


@pytest.fixture()
def metrics() -> Metrics:
    try:
        from prometheus_client import (
            CollectorRegistry,  # type: ignore[import-not-found]
        )
        return Metrics.build(registry=CollectorRegistry())
    except ImportError:  # pragma: no cover - no prom in env
        return Metrics.build()


def _healthy_opener(req: Any, timeout: float | None = None) -> _FakeResponse:
    del timeout
    if "/aliveness-test/" in req.full_url:
        return _FakeResponse(200, {"status": "ok"})
    return _FakeResponse(200, {})


def test_reconcile_without_failover_engine_does_not_invoke_anything(
    sample_registry: Registry, metrics: Metrics
) -> None:
    """The default constructor must omit ``failover_engine``; behaviour is unchanged."""
    api = MagicMock()
    reconciler = Reconciler(
        sample_registry,
        release_name="ocr-local",
        namespace="ocr",
        opener=_healthy_opener,
        kubernetes_api=api,
        metrics=metrics,
    )
    assert reconciler.failover_engine is None
    # Run a cycle -- nothing failover-related should happen, and existing
    # policy creation must still work.
    reconciler.reconcile_once(now=10.0)
    assert api.create_namespaced_custom_object.call_count >= 1


def test_reconcile_with_failover_engine_invokes_tick_exactly_once(
    sample_registry: Registry, metrics: Metrics
) -> None:
    api = MagicMock()
    failover_engine = MagicMock()
    failover_engine.tick.return_value = MagicMock(as_dict=lambda: {"ok": True})

    reconciler = Reconciler(
        sample_registry,
        release_name="ocr-local",
        namespace="ocr",
        opener=_healthy_opener,
        kubernetes_api=api,
        metrics=metrics,
        failover_engine=failover_engine,
    )
    reconciler.reconcile_once(now=10.0)
    assert failover_engine.tick.call_count == 1
    reconciler.reconcile_once(now=20.0)
    assert failover_engine.tick.call_count == 2


def test_reconcile_failover_engine_exception_does_not_break_cycle(
    sample_registry: Registry, metrics: Metrics
) -> None:
    api = MagicMock()
    failover_engine = MagicMock()
    failover_engine.tick.side_effect = RuntimeError("engine boom")

    reconciler = Reconciler(
        sample_registry,
        release_name="ocr-local",
        namespace="ocr",
        opener=_healthy_opener,
        kubernetes_api=api,
        metrics=metrics,
        failover_engine=failover_engine,
    )
    # Must not raise -- the cycle isolates the engine error.
    reconciler.reconcile_once(now=10.0)
    assert failover_engine.tick.call_count == 1


def test_reconcile_calls_failover_after_policy_logic(
    sample_registry: Registry, metrics: Metrics
) -> None:
    """Failover must run AFTER policy add/remove so the engine sees fresh
    health state for the cycle.
    """
    call_order: list[str] = []

    api = MagicMock()
    api.create_namespaced_custom_object.side_effect = (
        lambda **_kw: call_order.append("policy_create")
    )

    class _RecordingEngine:
        def tick(self) -> Any:
            call_order.append("failover_tick")
            return MagicMock(as_dict=lambda: {})

    reconciler = Reconciler(
        sample_registry,
        release_name="ocr-local",
        namespace="ocr",
        opener=_healthy_opener,
        kubernetes_api=api,
        metrics=metrics,
        failover_engine=_RecordingEngine(),
    )
    reconciler.reconcile_once(now=10.0)
    assert "failover_tick" in call_order
    assert "policy_create" in call_order
    # Policy create must come before failover tick on the same cycle.
    assert call_order.index("policy_create") < call_order.index("failover_tick")
