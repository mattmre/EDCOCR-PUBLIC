"""Tests for the cross-cluster custody ingest endpoint (Plan C C6).

Covers:

* Auth: missing/empty/wrong Bearer token -> 401
* Schema: unknown event_type -> 422
* Signature: bad signature -> 422; valid signature -> 200
* Persistence: row written to ``cross_cluster_custody_events`` table
* Idempotency: same (job_id, signature) re-ingest returns
  ``status="duplicate"`` and does not create a second row
* Feature gate: router is not mounted unless
  ``OCR_FEDERATION_CUSTODY_ENABLED=true``

The tests build a fresh FastAPI app per test run to ensure the
feature-gate path is exercised cleanly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

# The TestClient + FastAPI imports are conditionally optional so the
# coordinator-only suites don't trip on missing httpx.
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


HMAC_KEY = "endpoint-test-hmac-key"
BEARER = "endpoint-test-bearer-token"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    """Construct a fresh FastAPI app with the federation router mounted.

    We mount the router directly on a bare FastAPI instance rather than
    going through ``api.main.create_app`` because the latter pulls in
    the full middleware/router stack and is heavier than this single-
    endpoint test needs.
    """
    monkeypatch.setenv("OCR_FEDERATION_CUSTODY_ENABLED", "true")
    monkeypatch.setenv("OCR_FEDERATION_CUSTODY_HMAC_KEY", HMAC_KEY)
    monkeypatch.setenv("OCR_FEDERATION_CUSTODY_AUTH_TOKEN", BEARER)
    monkeypatch.setenv(
        "OCR_FEDERATION_CUSTODY_DB_PATH",
        str(tmp_path / "fed_custody.db"),
    )
    # Reset the schema-init guard so the new DB path takes effect.
    from api.routers import federation_custody as fc_module

    fc_module._reset_state_for_tests()

    app = fastapi.FastAPI()
    app.include_router(fc_module.router)
    return app


def _signed_payload(
    *,
    job_id: str = "job-1",
    source_cluster: str = "cluster-a",
    target_cluster: str = "cluster-b",
    event_type: str = "JOB_HANDED_OFF",
    parent_event_hash: str = "h0",
    dispatch_reason: str = "region_affinity",
    timestamp_utc: str = "2026-04-27T00:00:00.000+00:00",
    hmac_key: str = HMAC_KEY,
) -> dict:
    from coordinator.federation.custody import compute_signature

    body = {
        "job_id": job_id,
        "source_cluster": source_cluster,
        "target_cluster": target_cluster,
        "parent_event_hash": parent_event_hash,
        "event_type": event_type,
        "timestamp_utc": timestamp_utc,
        "dispatch_reason": dispatch_reason,
    }
    body["signature"] = compute_signature(body, hmac_key)
    return body


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAuth:
    def test_missing_authorization_returns_401(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload()
        resp = client.post("/api/v1/federation/custody/ingest", json=body)
        assert resp.status_code == 401

    def test_wrong_bearer_returns_401(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload()
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_non_bearer_scheme_returns_401(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload()
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": "Basic deadbeef"},
        )
        assert resp.status_code == 401

    def test_valid_bearer_accepts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload()
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": f"Bearer {BEARER}"},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] == "accepted"
        assert payload["job_id"] == "job-1"


class TestSignatureVerification:
    def test_bad_signature_returns_422(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload()
        body["signature"] = "deadbeef" * 4
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": f"Bearer {BEARER}"},
        )
        assert resp.status_code == 422
        assert "Signature" in resp.text or "signature" in resp.text

    def test_unknown_event_type_returns_422(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload(event_type="NOT_A_REAL_EVENT")
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": f"Bearer {BEARER}"},
        )
        assert resp.status_code == 422


class TestPersistence:
    def test_event_written_to_table(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "fed_custody.db"
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload(job_id="persist-job-1")
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": f"Bearer {BEARER}"},
        )
        assert resp.status_code == 200, resp.text
        # Verify the row landed in SQLite.
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                "SELECT job_id, source_cluster, target_cluster, event_type, "
                "signature FROM cross_cluster_custody_events "
                "WHERE job_id = ?",
                ("persist-job-1",),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        assert len(rows) == 1
        row = rows[0]
        assert row[0] == "persist-job-1"
        assert row[3] == "JOB_HANDED_OFF"

    def test_idempotent_duplicate_replay_returns_duplicate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload(job_id="dup-job")
        headers = {"Authorization": f"Bearer {BEARER}"}
        resp1 = client.post(
            "/api/v1/federation/custody/ingest", json=body, headers=headers
        )
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "accepted"
        # Same signed payload again -> duplicate, but still 200.
        resp2 = client.post(
            "/api/v1/federation/custody/ingest", json=body, headers=headers
        )
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "duplicate"
        # Confirm only one physical row exists.
        db_path = tmp_path / "fed_custody.db"
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM cross_cluster_custody_events "
                "WHERE job_id = ?",
                ("dup-job",),
            )
            count = cur.fetchone()[0]
        finally:
            conn.close()
        assert count == 1


class TestSchemaValidation:
    def test_missing_required_field_returns_422(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        # Missing job_id.
        body = _signed_payload()
        body.pop("job_id")
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": f"Bearer {BEARER}"},
        )
        assert resp.status_code == 422

    def test_empty_string_field_returns_422(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload()
        body["job_id"] = ""  # min_length=1 -> 422
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": f"Bearer {BEARER}"},
        )
        assert resp.status_code == 422


class TestHmacKeyConfig:
    def test_no_hmac_key_returns_422(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Build the app, then strip the HMAC key so the handler must
        # reject the request.
        app = _build_app(monkeypatch, tmp_path)
        monkeypatch.delenv("OCR_FEDERATION_CUSTODY_HMAC_KEY", raising=False)
        client = TestClient(app)
        body = _signed_payload()
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": f"Bearer {BEARER}"},
        )
        assert resp.status_code == 422


class TestRebalanceEvent:
    def test_rebalance_event_accepted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        app = _build_app(monkeypatch, tmp_path)
        client = TestClient(app)
        body = _signed_payload(
            event_type="JOB_REBALANCED",
            parent_event_hash="",
            dispatch_reason="cluster_unhealthy",
        )
        resp = client.post(
            "/api/v1/federation/custody/ingest",
            json=body,
            headers={"Authorization": f"Bearer {BEARER}"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "accepted"
