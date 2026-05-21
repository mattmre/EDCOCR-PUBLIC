"""FastAPI tests for ``api.routers.translation_batch`` (Plan B Wave M2 -- B17).

Exercises the REST endpoints in isolation against an app that mounts only
the ``translation_batch`` router.  Tests patch ``submit_batch``,
``fan_out``, ``get_status``, ``collect_results``, and ``cancel_batch`` to
keep the API contract under test independent of Django.
"""
from __future__ import annotations

from typing import Iterator
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client() -> Iterator[TestClient]:
    from api.routers.translation_batch import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


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
    with patch("api.routers.translation_batch.submit_batch", return_value="abc123"):
        with patch("api.routers.translation_batch.fan_out", return_value=1):
            resp = client.post("/api/v1/translation/batches", json=_payload())
    assert resp.status_code == 202
    body = resp.json()
    assert body["batch_id"] == "abc123"
    assert body["status"] == "pending"


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


def test_submit_fan_out_failure_does_not_block_response(client):
    """A broker outage during fan_out must not fail the submit response."""
    with patch("api.routers.translation_batch.submit_batch", return_value="abc123"):
        with patch(
            "api.routers.translation_batch.fan_out",
            side_effect=RuntimeError("broker down"),
        ):
            resp = client.post("/api/v1/translation/batches", json=_payload())
    assert resp.status_code == 202
    assert resp.json()["batch_id"] == "abc123"


def test_submit_invalid_payload_returns_422(client):
    """Pydantic validation: missing required field."""
    bad = {"source_lang": "en", "target_lang": "fr", "inputs": []}
    resp = client.post("/api/v1/translation/batches", json=bad)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/translation/batches/{batch_id}
# ---------------------------------------------------------------------------


def test_get_status_returns_snapshot(client):
    snap = _make_snap("abc123", "running", total=3, completed_inputs=1, pending_inputs=2)
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
        "abc123", "completed", total=2,
        completed_inputs=2, pending_inputs=0,
    )
    results = [
        BatchTranslationResult(
            client_ref="r1", target_text="bonjour",
            engine_id="opus-en-fr", confidence=0.9,
            glossary_hits=[], error=None,
        ),
        BatchTranslationResult(
            client_ref="r2", target_text="monde",
            engine_id="opus-en-fr", confidence=0.85,
            glossary_hits=[], error=None,
        ),
    ]
    with patch("api.routers.translation_batch.get_status", return_value=snap):
        with patch(
            "api.routers.translation_batch.collect_results", return_value=results,
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
    with patch("api.routers.translation_batch.cancel_batch", return_value=3):
        resp = client.post("/api/v1/translation/batches/abc123/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["batch_id"] == "abc123"
    assert body["revoked"] == 3
    assert body["status"] == "cancelled"


def test_cancel_unknown_batch_returns_404(client):
    from ocr_local.translation.batch import BatchNotFoundError

    with patch(
        "api.routers.translation_batch.cancel_batch",
        side_effect=BatchNotFoundError("abc123"),
    ):
        resp = client.post("/api/v1/translation/batches/abc123/cancel")
    assert resp.status_code == 404


def test_cancel_idempotent_terminal(client):
    """Cancel on a terminal batch returns 200 with revoked=0."""
    with patch("api.routers.translation_batch.cancel_batch", return_value=0):
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
        "/api/v1/translation/batches/{batch_id}/results", ("GET",),
    ) in paths
    assert (
        "/api/v1/translation/batches/{batch_id}/cancel", ("POST",),
    ) in paths
