"""FastAPI tests for ``api.routers.translation_batch`` (Plan B Wave M2 -- B17).

Exercises the REST endpoints in isolation against an app that mounts only
the ``translation_batch`` router.  Tests patch ``submit_batch``,
``fan_out``, ``get_status``, ``collect_results``, and ``cancel_batch`` to
keep the API contract under test independent of Django.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.limits import limiter  # noqa: E402


@pytest.fixture
def client() -> Iterator[TestClient]:
    from api.identity import AuthIdentity
    from api.routers.translation_batch import router

    app = FastAPI()

    @app.middleware("http")
    async def _operator_identity(request, call_next):
        request.state.identity = AuthIdentity(
            subject="translation-batch-test",
            role="operator",
            auth_method="test",
        )
        return await call_next(request)

    app.include_router(router)
    previous_enabled = limiter.enabled
    limiter.enabled = False
    try:
        with (
            patch(
                "api.routers.translation_batch._run_rls_bound",
                side_effect=lambda _tenant, func, *args, **kwargs: func(
                    *args, **kwargs
                ),
            ),
            TestClient(app) as c,
        ):
            yield c
    finally:
        limiter.enabled = previous_enabled


def _payload(**overrides):
    p = {
        "tenant_id": "tenant-a",
        "source_lang": "en",
        "target_lang": "fr",
        "inputs": [
            {"client_ref": "r1", "text": "hello"},
        ],
    }
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# Helper for snapshots
# ---------------------------------------------------------------------------


def _make_snap(batch_id, status, total=1, **kw):
    from ocr_local.translation.batch import BatchStatusSnapshot

    return BatchStatusSnapshot(
        batch_id=batch_id,
        tenant_id=kw.get("tenant_id", "tenant-a"),
        source_lang=kw.get("source_lang", "en"),
        target_lang=kw.get("target_lang", "fr"),
        status=status,
        total_inputs=total,
        completed_inputs=kw.get("completed_inputs", 0),
        failed_inputs=kw.get("failed_inputs", 0),
        pending_inputs=kw.get("pending_inputs", total),
        running_inputs=kw.get("running_inputs", 0),
        submitted_at=kw.get("submitted_at"),
        completed_at=kw.get("completed_at"),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/translation/batches
# ---------------------------------------------------------------------------


def test_submit_returns_202_with_batch_id(client):
    review = SimpleNamespace(review_id="rev_0123456789ab")
    with patch("api.routers.translation_batch.submit_batch", return_value="abc123"):
        with patch(
            "api.routers.translation_batch._enqueue_certification_review",
            return_value=review,
        ) as enqueue:
            with patch("api.routers.translation_batch.fan_out", return_value=1):
                resp = client.post("/api/v1/translation/batches", json=_payload())
    assert resp.status_code == 202
    body = resp.json()
    assert body["batch_id"] == "abc123"
    assert body["review_id"] == "rev_0123456789ab"
    assert body["status"] == "pending"
    enqueue.assert_called_once_with("abc123", "tenant-a")


def test_submit_review_queue_failure_is_visible_to_caller(client):
    with patch("api.routers.translation_batch.submit_batch", return_value="abc123"):
        with patch(
            "api.routers.translation_batch.cancel_batch", return_value=1
        ) as cancel:
            with patch(
                "api.routers.translation_batch._enqueue_certification_review",
                side_effect=RuntimeError("review store down"),
            ):
                resp = client.post("/api/v1/translation/batches", json=_payload())
    assert resp.status_code == 503
    assert resp.json()["detail"]["batch_id"] == "abc123"
    cancel.assert_called_once_with("abc123")


def test_submit_validation_error_returns_400(client):
    from ocr_local.translation.batch import BatchValidationError

    with patch(
        "api.routers.translation_batch.submit_batch",
        side_effect=BatchValidationError("inputs[0].text exceeds size cap"),
    ):
        resp = client.post("/api/v1/translation/batches", json=_payload())
    assert resp.status_code == 400
    body = resp.json()
    assert "size cap" in body["detail"]


def test_submit_certified_true_returns_400(client):
    from ocr_local.translation.custody_adapter import ReasonCode
    from ocr_local.translation.policy import PolicyDenied

    payload = _payload(requested_certified=True)
    with patch(
        "api.routers.translation_batch.submit_batch",
        side_effect=PolicyDenied(
            ReasonCode.BATCH_REJECTED_CERTIFIED,
            "requested_certified=True rejected at submit time",
        ),
    ):
        resp = client.post("/api/v1/translation/batches", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert "BATCH_REJECTED_CERTIFIED" in body["detail"] or "certified" in body["detail"]


def test_submit_runtime_error_returns_503(client):
    """Django not configured -> RuntimeError -> 503."""
    with patch(
        "api.routers.translation_batch.submit_batch",
        side_effect=RuntimeError("Django not configured"),
    ):
        resp = client.post("/api/v1/translation/batches", json=_payload())
    assert resp.status_code == 503


def test_submit_fan_out_failure_is_visible_to_caller(client):
    """A broker outage must not be reported as a successful 202 submission."""
    with patch("api.routers.translation_batch.submit_batch", return_value="abc123"):
        with patch(
            "api.routers.translation_batch._enqueue_certification_review",
            return_value=SimpleNamespace(review_id="rev_0123456789ab"),
        ):
            with patch(
                "api.routers.translation_batch.fan_out",
                side_effect=RuntimeError("broker down"),
            ):
                resp = client.post("/api/v1/translation/batches", json=_payload())
    assert resp.status_code == 503
    assert resp.json()["detail"]["batch_id"] == "abc123"


def test_submit_invalid_payload_returns_422(client):
    """Pydantic validation: missing required field."""
    bad = {"source_lang": "en", "target_lang": "fr", "inputs": []}
    resp = client.post("/api/v1/translation/batches", json=bad)
    assert resp.status_code == 422


def test_reconcile_repairs_missing_review_and_pending_dispatch():
    from api.routers.translation_batch import reconcile_translation_batch_handoffs

    BatchTranslationJob = SimpleNamespace(
        STATUS_PENDING="pending",
        STATUS_RUNNING="running",
        STATUS_COMPLETED="completed",
    )
    row = SimpleNamespace(
        batch_id="batch-recover",
        tenant_id="tenant-a",
        status=BatchTranslationJob.STATUS_PENDING,
    )
    certified_review = SimpleNamespace(
        review_id="review-certified",
        metadata={
            "batch_id": "batch-certified",
            "tenant_id": "tenant-a",
            "certified": True,
            "certification": {"custody_event_id": "c" * 64},
        },
    )
    models_module = SimpleNamespace(BatchTranslationJob=BatchTranslationJob)
    rls_module = SimpleNamespace(
        control_plane_rls_context=MagicMock(return_value=MagicMock()),
        tenant_rls_context=MagicMock(return_value=MagicMock()),
    )
    review_queue = MagicMock()
    review_queue.list_pending_translation_promotions.return_value = [certified_review]
    review_queue.confirm_translation_promotion.return_value = True

    with (
        patch.dict(
            sys.modules,
            {"jobs.models": models_module, "jobs.rls": rls_module},
        ),
        patch("api.coordinator_bridge.ensure_coordinator_django"),
        patch(
            "api.routers.translation_batch._translation_input_lease_seconds",
            return_value=1800,
        ),
        patch(
            "api.routers.translation_batch._translation_recovery_candidates",
            return_value=[row],
        ),
        patch(
            "api.routers.translation_batch._enqueue_certification_review",
            return_value=SimpleNamespace(metadata={}),
        ) as enqueue,
        patch(
            "api.routers.translation_batch.reclaim_stale_batch_inputs",
            return_value=0,
        ) as reclaim,
        patch("api.routers.translation_batch.fan_out", return_value=1) as dispatch,
        patch("api.routers.translation_batch.promote_batch_certified") as promote,
    ):
        assert reconcile_translation_batch_handoffs(queue=review_queue) == 2

    enqueue.assert_called_once_with("batch-recover", "tenant-a")
    reclaim.assert_called_once_with("batch-recover", lease_seconds=1800)
    dispatch.assert_called_once_with("batch-recover")
    promote.assert_called_once_with(
        "batch-certified",
        custody_event_id="c" * 64,
        tenant_id="tenant-a",
    )
    review_queue.confirm_translation_promotion.assert_called_once_with(
        "review-certified", "c" * 64
    )


def test_reconcile_contains_one_batch_dispatch_failure_and_continues():
    from api.routers.translation_batch import reconcile_translation_batch_handoffs

    BatchTranslationJob = SimpleNamespace(
        STATUS_PENDING="pending",
        STATUS_RUNNING="running",
        STATUS_COMPLETED="completed",
    )
    failed_row = SimpleNamespace(
        batch_id="batch-a-fails",
        tenant_id="tenant-a",
        status=BatchTranslationJob.STATUS_PENDING,
    )
    next_row = SimpleNamespace(
        batch_id="batch-b-continues",
        tenant_id="tenant-a",
        status=BatchTranslationJob.STATUS_PENDING,
    )
    review_queue = MagicMock()
    review_queue.list_pending_translation_promotions.return_value = []

    with (
        patch.dict(
            sys.modules,
            {
                "jobs.models": SimpleNamespace(BatchTranslationJob=BatchTranslationJob),
                "jobs.rls": SimpleNamespace(
                    control_plane_rls_context=MagicMock(return_value=MagicMock()),
                    tenant_rls_context=MagicMock(return_value=MagicMock()),
                ),
            },
        ),
        patch("api.coordinator_bridge.ensure_coordinator_django"),
        patch(
            "api.routers.translation_batch._translation_input_lease_seconds",
            return_value=1800,
        ),
        patch(
            "api.routers.translation_batch._translation_recovery_candidates",
            return_value=[failed_row, next_row],
        ),
        patch(
            "api.routers.translation_batch._enqueue_certification_review",
            return_value=SimpleNamespace(metadata={}),
        ),
        patch(
            "api.routers.translation_batch.reclaim_stale_batch_inputs",
            return_value=0,
        ),
        patch(
            "api.routers.translation_batch.fan_out",
            side_effect=[RuntimeError("broker down"), 1],
        ) as dispatch,
    ):
        assert reconcile_translation_batch_handoffs(queue=review_queue) == 1

    assert dispatch.call_args_list == [
        call("batch-a-fails"),
        call("batch-b-continues"),
    ]


def test_reconcile_quarantines_irrecoverable_promotions_and_continues():
    from api.routers.translation_batch import reconcile_translation_batch_handoffs
    from ocr_local.translation.batch import BatchNotFoundError

    BatchTranslationJob = SimpleNamespace(
        STATUS_PENDING="pending",
        STATUS_RUNNING="running",
    )
    invalid = SimpleNamespace(
        review_id="review-invalid",
        metadata={"batch_id": "", "tenant_id": "tenant-a"},
    )
    missing = SimpleNamespace(
        review_id="review-missing",
        metadata={
            "batch_id": "batch-missing",
            "tenant_id": "tenant-a",
            "certification": {"custody_event_id": "a" * 64},
        },
    )
    valid = SimpleNamespace(
        review_id="review-valid",
        metadata={
            "batch_id": "batch-valid",
            "tenant_id": "tenant-a",
            "certification": {"custody_event_id": "b" * 64},
        },
    )
    queue = MagicMock()
    queue.list_pending_translation_promotions.return_value = [
        invalid,
        missing,
        valid,
    ]
    queue.abandon_translation_promotion.return_value = True
    queue.confirm_translation_promotion.return_value = True

    with (
        patch.dict(
            sys.modules,
            {
                "jobs.models": SimpleNamespace(BatchTranslationJob=BatchTranslationJob),
                "jobs.rls": SimpleNamespace(
                    control_plane_rls_context=MagicMock(return_value=MagicMock()),
                    tenant_rls_context=MagicMock(return_value=MagicMock()),
                ),
            },
        ),
        patch("api.coordinator_bridge.ensure_coordinator_django"),
        patch(
            "api.routers.translation_batch._translation_input_lease_seconds",
            return_value=1800,
        ),
        patch(
            "api.routers.translation_batch._translation_recovery_candidates",
            return_value=[],
        ),
        patch(
            "api.routers.translation_batch.promote_batch_certified",
            side_effect=[BatchNotFoundError("batch-missing"), None],
        ) as promote,
    ):
        assert reconcile_translation_batch_handoffs(queue=queue) == 3

    assert queue.abandon_translation_promotion.call_args_list == [
        call("review-invalid", "invalid durable promotion binding"),
        call("review-missing", "coordinator batch no longer exists"),
    ]
    assert [args.args[0] for args in promote.call_args_list] == [
        "batch-missing",
        "batch-valid",
    ]
    queue.confirm_translation_promotion.assert_called_once_with(
        "review-valid", "b" * 64
    )


def test_reconcile_defers_retryable_promotion_and_continues():
    from api.routers.translation_batch import reconcile_translation_batch_handoffs

    BatchTranslationJob = SimpleNamespace(
        STATUS_PENDING="pending",
        STATUS_RUNNING="running",
    )
    retryable = SimpleNamespace(
        review_id="review-retryable",
        metadata={
            "batch_id": "batch-not-complete",
            "tenant_id": "tenant-a",
            "certification": {"custody_event_id": "a" * 64},
        },
    )
    valid = SimpleNamespace(
        review_id="review-valid-after-retry",
        metadata={
            "batch_id": "batch-complete",
            "tenant_id": "tenant-a",
            "certification": {"custody_event_id": "b" * 64},
        },
    )
    queue = MagicMock()
    queue.list_pending_translation_promotions.return_value = [retryable, valid]
    queue.defer_translation_promotion.return_value = True
    queue.confirm_translation_promotion.return_value = True

    with (
        patch.dict(
            sys.modules,
            {
                "jobs.models": SimpleNamespace(BatchTranslationJob=BatchTranslationJob),
                "jobs.rls": SimpleNamespace(
                    control_plane_rls_context=MagicMock(return_value=MagicMock()),
                    tenant_rls_context=MagicMock(return_value=MagicMock()),
                ),
            },
        ),
        patch("api.coordinator_bridge.ensure_coordinator_django"),
        patch(
            "api.routers.translation_batch._translation_input_lease_seconds",
            return_value=1800,
        ),
        patch(
            "api.routers.translation_batch._translation_recovery_candidates",
            return_value=[],
        ),
        patch(
            "api.routers.translation_batch.promote_batch_certified",
            side_effect=[RuntimeError("batch not complete"), None],
        ),
    ):
        assert reconcile_translation_batch_handoffs(queue=queue) == 1

    queue.defer_translation_promotion.assert_called_once_with(
        "review-retryable", "RuntimeError: batch not complete"
    )
    queue.confirm_translation_promotion.assert_called_once_with(
        "review-valid-after-retry", "b" * 64
    )


def test_reconcile_loop_retries_transient_failure_without_restart(monkeypatch):
    from api.main import _translation_handoff_reconcile_loop

    monkeypatch.setenv("TRANSLATION_RECONCILE_INTERVAL_SECONDS", "1")
    reconcile = MagicMock(side_effect=[RuntimeError("database unavailable"), 1])
    waits = 0
    stop_event = MagicMock(spec=threading.Event)
    stop_event.is_set.return_value = False

    def bounded_wait(_interval):
        nonlocal waits
        waits += 1
        return waits >= 2

    stop_event.wait.side_effect = bounded_wait

    with (
        patch(
            "api.routers.translation_batch.reconcile_translation_batch_handoffs",
            reconcile,
        ),
    ):
        _translation_handoff_reconcile_loop(stop_event)

    assert reconcile.call_count == 2


def test_blocked_reconcile_worker_is_daemon_and_shutdown_is_bounded():
    from api.main import _translation_handoff_reconcile_loop

    entered = threading.Event()
    release = threading.Event()
    stop = threading.Event()

    def blocked_reconcile():
        entered.set()
        release.wait(timeout=5)

    with patch(
        "api.routers.translation_batch.reconcile_translation_batch_handoffs",
        side_effect=blocked_reconcile,
    ):
        worker = threading.Thread(
            target=_translation_handoff_reconcile_loop,
            args=(stop,),
            daemon=True,
        )
        worker.start()
        assert entered.wait(timeout=2)
        stop.set()
        worker.join(timeout=0.05)
        assert worker.daemon is True
        assert worker.is_alive() is True
        release.set()
        worker.join(timeout=2)
        assert worker.is_alive() is False


# ---------------------------------------------------------------------------
# GET /api/v1/translation/batches/{batch_id}
# ---------------------------------------------------------------------------


def test_get_status_returns_snapshot(client):
    snap = _make_snap(
        "abc123", "running", total=3, completed_inputs=1, pending_inputs=2
    )
    with patch("api.routers.translation_batch.get_status", return_value=snap):
        resp = client.get("/api/v1/translation/batches/abc123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["batch_id"] == "abc123"
    assert body["status"] == "running"
    assert body["total_inputs"] == 3
    assert body["completed_inputs"] == 1
    assert body["pending_inputs"] == 2


def test_get_status_unknown_batch_returns_404(client):
    from ocr_local.translation.batch import BatchNotFoundError

    with patch(
        "api.routers.translation_batch.get_status",
        side_effect=BatchNotFoundError("abc123"),
    ):
        resp = client.get("/api/v1/translation/batches/abc123")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/translation/batches/{batch_id}/results
# ---------------------------------------------------------------------------


def test_get_results_409_when_not_terminal(client):
    snap = _make_snap("abc123", "running", total=2)
    with patch("api.routers.translation_batch.get_status", return_value=snap):
        resp = client.get("/api/v1/translation/batches/abc123/results")
    assert resp.status_code == 409


def test_get_results_200_when_completed(client):
    from ocr_local.translation.batch import BatchTranslationResult

    snap = _make_snap(
        "abc123",
        "completed",
        total=2,
        completed_inputs=2,
        pending_inputs=0,
    )
    results = [
        BatchTranslationResult(
            client_ref="r1",
            target_text="bonjour",
            engine_id="opus-en-fr",
            confidence=0.9,
            glossary_hits=[],
            error=None,
        ),
        BatchTranslationResult(
            client_ref="r2",
            target_text="monde",
            engine_id="opus-en-fr",
            confidence=0.85,
            glossary_hits=[],
            error=None,
        ),
    ]
    with patch("api.routers.translation_batch.get_status", return_value=snap):
        with patch(
            "api.routers.translation_batch.collect_results",
            return_value=results,
        ):
            resp = client.get("/api/v1/translation/batches/abc123/results")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2
    assert body[0]["client_ref"] == "r1"
    assert body[0]["target_text"] == "bonjour"
    assert body[1]["client_ref"] == "r2"


def test_get_results_unknown_batch_returns_404(client):
    from ocr_local.translation.batch import BatchNotFoundError

    with patch(
        "api.routers.translation_batch.get_status",
        side_effect=BatchNotFoundError("abc123"),
    ):
        resp = client.get("/api/v1/translation/batches/abc123/results")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v1/translation/batches/{batch_id}/cancel
# ---------------------------------------------------------------------------


def test_cancel_returns_revoked_count(client):
    running = _make_snap("abc123", "running")
    cancelled = _make_snap("abc123", "cancelled")
    with patch("api.routers.translation_batch.cancel_batch", return_value=3):
        with patch(
            "api.routers.translation_batch.get_status",
            side_effect=[running, cancelled],
        ):
            resp = client.post("/api/v1/translation/batches/abc123/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["batch_id"] == "abc123"
    assert body["revoked"] == 3
    assert body["status"] == "cancelled"


def test_cancel_unknown_batch_returns_404(client):
    from ocr_local.translation.batch import BatchNotFoundError

    with patch(
        "api.routers.translation_batch.get_status",
        side_effect=BatchNotFoundError("abc123"),
    ):
        resp = client.post("/api/v1/translation/batches/abc123/cancel")
    assert resp.status_code == 404


def test_cancel_idempotent_terminal(client):
    """Cancel on a terminal batch returns 200 with revoked=0."""
    snap = _make_snap("abc123", "completed", pending_inputs=0)
    with patch("api.routers.translation_batch.cancel_batch", return_value=0):
        with patch("api.routers.translation_batch.get_status", return_value=snap):
            resp = client.post("/api/v1/translation/batches/abc123/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["revoked"] == 0


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------


def test_router_prefix_and_tags():
    from api.routers.translation_batch import router

    assert router.prefix == "/api/v1/translation"
    assert "translation" in router.tags


def test_router_has_expected_routes():
    from api.routers.translation_batch import router

    paths = {(r.path, tuple(sorted(r.methods))) for r in router.routes}
    assert ("/api/v1/translation/batches", ("POST",)) in paths
    assert ("/api/v1/translation/batches/{batch_id}", ("GET",)) in paths
    assert (
        "/api/v1/translation/batches/{batch_id}/results",
        ("GET",),
    ) in paths
    assert (
        "/api/v1/translation/batches/{batch_id}/cancel",
        ("POST",),
    ) in paths
