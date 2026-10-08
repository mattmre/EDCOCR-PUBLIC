"""Tests for api/routers/events.py — event replay and DLQ endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import get_engine, reset_engine
from api.event_store import EventStore, reset_event_store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Give each test a fresh SQLite database and event store."""
    reset_engine()
    reset_event_store()
    db_file = str(tmp_path / "test_jobs.db")
    with (
        patch("api.config.DB_PATH", db_file),
        patch("api.database.DB_PATH", db_file),
        patch("api.config.OCR_API_KEY", ""),
        patch("api.config.ALLOW_UNAUTHENTICATED", True),
        # These tests exercise the operator-only DLQ surface.  Keep the
        # suite-wide anonymous default at viewer and opt this legacy
        # single-tenant fixture into its required role explicitly.
        patch("api.config.ANONYMOUS_ROLE", "operator"),
        patch("api.auth.ANONYMOUS_ROLE", "operator"),
        patch("api.config.ENABLE_MULTITENANCY", False),
        patch("api.auth.ENABLE_MULTITENANCY", False),
        patch("api.auth.get_api_key", return_value=""),
        patch("api.auth.allow_unauthenticated", return_value=True),
        patch("api.config.API_AUDIT_LOG_PATH", str(tmp_path / "api-audit.jsonl")),
        patch("api.config.EVENT_STORE_ENABLED", True),
        patch("api.config.EVENT_STORE_PATH", str(tmp_path / "events.db")),
        patch("api.config.WEBHOOK_DLQ_ENABLED", True),
        patch("api.config.WEBHOOK_DLQ_PATH", str(tmp_path / "dlq.jsonl")),
    ):
        reset_engine()
        get_engine(db_file)
        yield
        reset_event_store()
        reset_engine()


@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with isolated config."""
    from api.main import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture()
def store(tmp_path):
    """Create a fresh EventStore for testing."""
    db_path = str(tmp_path / "events.db")
    return EventStore(db_path=db_path)


# ---------------------------------------------------------------------------
# Event replay endpoint
# ---------------------------------------------------------------------------


class TestGetJobEvents:
    """Tests for GET /api/v1/jobs/{job_id}/events."""

    def test_returns_events_for_job(self, client, tmp_path):
        """Returns stored events for a valid job ID."""
        store = EventStore(db_path=str(tmp_path / "events.db"))
        with (
            patch("api.event_store._store", store),
            patch("api.event_store._store_lock"),
        ):
            store.store_event(
                "job.submitted", "job_aaaaaaaaaaaa", {"status": "submitted"}
            )
            store.store_event(
                "job.completed", "job_aaaaaaaaaaaa", {"status": "completed"}
            )

            resp = client.get("/api/v1/jobs/job_aaaaaaaaaaaa/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job_aaaaaaaaaaaa"
        assert data["count"] == 2
        assert len(data["events"]) == 2

    def test_since_id_filters_events(self, client, tmp_path):
        """since_id query parameter filters to events after the given ID."""
        store = EventStore(db_path=str(tmp_path / "events.db"))
        with (
            patch("api.event_store._store", store),
            patch("api.event_store._store_lock"),
        ):
            id1 = store.store_event("job.submitted", "job_aaaaaaaaaaaa", {"seq": 1})
            store.store_event("job.completed", "job_aaaaaaaaaaaa", {"seq": 2})

            resp = client.get(f"/api/v1/jobs/job_aaaaaaaaaaaa/events?since_id={id1}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["events"][0]["payload"]["seq"] == 2

    def test_empty_events_for_unknown_job(self, client, tmp_path):
        """Returns empty events list for a job with no events."""
        store = EventStore(db_path=str(tmp_path / "events.db"))
        with (
            patch("api.event_store._store", store),
            patch("api.event_store._store_lock"),
        ):
            resp = client.get("/api/v1/jobs/job_aaaaaaaaaaaa/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["events"] == []

    def test_invalid_job_id_returns_400(self, client):
        """Malformed job_id returns 400."""
        resp = client.get("/api/v1/jobs/invalid-id/events")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DLQ list endpoint
# ---------------------------------------------------------------------------


class TestListDlqEntries:
    """Tests for GET /api/v1/webhooks/dlq."""

    def test_returns_empty_when_no_entries(self, client, tmp_path):
        """Returns empty list when DLQ file does not exist."""
        with patch("api.webhook_dlq._dlq_path", return_value=tmp_path / "dlq.jsonl"):
            resp = client.get("/api/v1/webhooks/dlq")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["entries"] == []

    def test_returns_dlq_entries(self, client, tmp_path):
        """Returns DLQ entries in reverse chronological order."""
        from api.webhook_dlq import add_to_dlq

        dlq_file = tmp_path / "dlq.jsonl"
        add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com/hook",
            event_type="job.completed",
            payload={"event": "job.completed"},
            last_error="HTTP 500",
            attempts=4,
            dlq_file=dlq_file,
        )

        with patch("api.webhook_dlq._dlq_path", return_value=dlq_file):
            resp = client.get("/api/v1/webhooks/dlq")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["entries"][0]["job_id"] == "job_aaaaaaaaaaaa"


# ---------------------------------------------------------------------------
# DLQ get single entry
# ---------------------------------------------------------------------------


class TestGetDlqEntry:
    """Tests for GET /api/v1/webhooks/dlq/{entry_id}."""

    def test_returns_entry_by_id(self, client, tmp_path):
        """Returns the DLQ entry matching the given ID."""
        from api.webhook_dlq import add_to_dlq

        dlq_file = tmp_path / "dlq.jsonl"
        entry_id = add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com/hook",
            event_type="job.completed",
            payload={"event": "job.completed"},
            last_error="HTTP 500",
            attempts=4,
            dlq_file=dlq_file,
        )

        with patch("api.webhook_dlq._dlq_path", return_value=dlq_file):
            resp = client.get(f"/api/v1/webhooks/dlq/{entry_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == entry_id

    def test_returns_404_for_nonexistent(self, client, tmp_path):
        """Returns 404 when entry ID is not found."""
        dlq_file = tmp_path / "dlq.jsonl"
        with patch("api.webhook_dlq._dlq_path", return_value=dlq_file):
            resp = client.get("/api/v1/webhooks/dlq/dlq_0000000000000000")

        assert resp.status_code == 404

    def test_invalid_dlq_id_returns_400(self, client):
        """Malformed DLQ entry_id returns 400."""
        resp = client.get("/api/v1/webhooks/dlq/invalid-id")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DLQ retry endpoint
# ---------------------------------------------------------------------------


class TestRetryDlqEntry:
    """Tests for POST /api/v1/webhooks/dlq/{entry_id}/retry."""

    def test_returns_404_for_nonexistent(self, client, tmp_path):
        """Returns 404 when entry ID is not found."""
        dlq_file = tmp_path / "dlq.jsonl"
        with patch("api.webhook_dlq._dlq_path", return_value=dlq_file):
            resp = client.post("/api/v1/webhooks/dlq/dlq_0000000000000000/retry")

        assert resp.status_code == 404

    def test_returns_409_for_already_retried(self, client, tmp_path):
        """Returns 409 when entry has already been retried."""
        from api.webhook_dlq import add_to_dlq, mark_dlq_retried

        dlq_file = tmp_path / "dlq.jsonl"
        entry_id = add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com/hook",
            event_type="job.completed",
            payload={},
            last_error="error",
            attempts=4,
            dlq_file=dlq_file,
        )
        mark_dlq_retried(entry_id, dlq_file=dlq_file)

        with patch("api.webhook_dlq._dlq_path", return_value=dlq_file):
            resp = client.post(f"/api/v1/webhooks/dlq/{entry_id}/retry")

        assert resp.status_code == 409

    def test_invalid_dlq_id_returns_400(self, client):
        """Malformed DLQ entry_id returns 400."""
        resp = client.post("/api/v1/webhooks/dlq/bad-id/retry")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Multi-tenant DLQ isolation (AEP BE003)
# ---------------------------------------------------------------------------


def _make_tenant_request(*, tenant_id=None, platform_admin=False):
    """Build a Starlette Request with tenant state for unit-level authz checks."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/v1/webhooks/dlq",
        "raw_path": b"/api/v1/webhooks/dlq",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    req = Request(scope)
    req.state.tenant_id = tenant_id
    req.state.tenant_permissions = ["platform_admin"] if platform_admin else []
    req.state.identity = None
    return req


class TestDlqTenantScopeHelpers:
    """Unit tests for _dlq_scope_tenant / _require_dlq_entry_access."""

    def test_single_tenant_unscoped(self):
        from api.routers.events import _dlq_scope_tenant

        assert _dlq_scope_tenant(_make_tenant_request(tenant_id=None)) is None

    def test_tenant_key_scoped(self):
        from api.routers.events import _dlq_scope_tenant

        assert (
            _dlq_scope_tenant(_make_tenant_request(tenant_id="tenant_aaaaaaaaaaaa"))
            == "tenant_aaaaaaaaaaaa"
        )

    def test_platform_admin_unscoped(self):
        from api.routers.events import _dlq_scope_tenant

        assert (
            _dlq_scope_tenant(
                _make_tenant_request(
                    tenant_id="tenant_aaaaaaaaaaaa", platform_admin=True
                )
            )
            is None
        )

    def test_require_access_foreign_raises_404(self):
        from fastapi import HTTPException

        from api.routers.events import _require_dlq_entry_access

        entry = {
            "id": "dlq_aaaaaaaaaaaaaaaa",
            "job_id": "job_bbbbbbbbbbbb",
            "tenant_id": "tenant_bbbbbbbbbbbb",
        }
        req = _make_tenant_request(tenant_id="tenant_aaaaaaaaaaaa")
        with pytest.raises(HTTPException) as ei:
            _require_dlq_entry_access(entry, req)
        assert ei.value.status_code == 404

    def test_require_access_owned_ok(self):
        from api.routers.events import _require_dlq_entry_access

        entry = {
            "id": "dlq_aaaaaaaaaaaaaaaa",
            "job_id": "job_aaaaaaaaaaaa",
            "tenant_id": "tenant_aaaaaaaaaaaa",
        }
        req = _make_tenant_request(tenant_id="tenant_aaaaaaaaaaaa")
        _require_dlq_entry_access(entry, req)  # no raise

    def test_require_access_admin_sees_foreign(self):
        from api.routers.events import _require_dlq_entry_access

        entry = {
            "id": "dlq_aaaaaaaaaaaaaaaa",
            "job_id": "job_bbbbbbbbbbbb",
            "tenant_id": "tenant_bbbbbbbbbbbb",
        }
        req = _make_tenant_request(tenant_id="tenant_aaaaaaaaaaaa", platform_admin=True)
        _require_dlq_entry_access(entry, req)  # no raise


class TestDlqMultiTenantRouter:
    """Integration-style router tests: tenant A cannot list/get/retry B."""

    def _seed_entries(self, dlq_file):
        from api.webhook_dlq import add_to_dlq

        id_a = add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://a.example.com/hook",
            event_type="job.completed",
            payload={"tenant": "a", "secret": "a-payload"},
            last_error="HTTP 500",
            attempts=4,
            tenant_id="tenant_aaaaaaaaaaaa",
            dlq_file=dlq_file,
        )
        id_b = add_to_dlq(
            job_id="job_bbbbbbbbbbbb",
            webhook_url="https://b.example.com/hook",
            event_type="job.failed",
            payload={"tenant": "b", "secret": "b-payload"},
            last_error="HTTP 502",
            attempts=4,
            tenant_id="tenant_bbbbbbbbbbbb",
            dlq_file=dlq_file,
        )
        return id_a, id_b

    def test_list_isolates_tenant_entries(self, client, tmp_path):
        """Tenant A list must not include tenant B entries."""
        dlq_file = tmp_path / "dlq.jsonl"
        self._seed_entries(dlq_file)

        with (
            patch("api.webhook_dlq._dlq_path", return_value=dlq_file),
            patch(
                "api.routers.events._request_tenant_id",
                return_value="tenant_aaaaaaaaaaaa",
            ),
            patch("api.routers.events._is_platform_admin", return_value=False),
        ):
            resp = client.get("/api/v1/webhooks/dlq")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["entries"][0]["tenant_id"] == "tenant_aaaaaaaaaaaa"
        assert data["entries"][0]["payload"]["secret"] == "a-payload"
        job_ids = {e["job_id"] for e in data["entries"]}
        assert "job_bbbbbbbbbbbb" not in job_ids

    def test_list_single_tenant_sees_all(self, client, tmp_path):
        """Single-tenant mode (tenant_id None) still returns the full queue."""
        dlq_file = tmp_path / "dlq.jsonl"
        self._seed_entries(dlq_file)

        with patch("api.webhook_dlq._dlq_path", return_value=dlq_file):
            resp = client.get("/api/v1/webhooks/dlq")

        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_list_platform_admin_sees_all(self, client, tmp_path):
        """platform_admin may list every tenant's DLQ entries."""
        dlq_file = tmp_path / "dlq.jsonl"
        self._seed_entries(dlq_file)

        with (
            patch("api.webhook_dlq._dlq_path", return_value=dlq_file),
            patch(
                "api.routers.events._request_tenant_id",
                return_value="tenant_aaaaaaaaaaaa",
            ),
            patch("api.routers.events._is_platform_admin", return_value=True),
        ):
            resp = client.get("/api/v1/webhooks/dlq")

        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_get_foreign_returns_404(self, client, tmp_path):
        """Tenant A cannot GET tenant B's DLQ entry (404, not 403)."""
        dlq_file = tmp_path / "dlq.jsonl"
        _id_a, id_b = self._seed_entries(dlq_file)

        with (
            patch("api.webhook_dlq._dlq_path", return_value=dlq_file),
            patch(
                "api.routers.events._request_tenant_id",
                return_value="tenant_aaaaaaaaaaaa",
            ),
            patch("api.routers.events._is_platform_admin", return_value=False),
        ):
            resp = client.get(f"/api/v1/webhooks/dlq/{id_b}")

        assert resp.status_code == 404

    def test_get_owned_returns_200(self, client, tmp_path):
        """Tenant A can GET its own DLQ entry."""
        dlq_file = tmp_path / "dlq.jsonl"
        id_a, _id_b = self._seed_entries(dlq_file)

        with (
            patch("api.webhook_dlq._dlq_path", return_value=dlq_file),
            patch(
                "api.routers.events._request_tenant_id",
                return_value="tenant_aaaaaaaaaaaa",
            ),
            patch("api.routers.events._is_platform_admin", return_value=False),
        ):
            resp = client.get(f"/api/v1/webhooks/dlq/{id_a}")

        assert resp.status_code == 200
        assert resp.json()["id"] == id_a

    def test_retry_foreign_returns_404_without_delivery(self, client, tmp_path):
        """Tenant A cannot retry tenant B's entry; delivery is not attempted."""
        dlq_file = tmp_path / "dlq.jsonl"
        _id_a, id_b = self._seed_entries(dlq_file)

        with (
            patch("api.webhook_dlq._dlq_path", return_value=dlq_file),
            patch(
                "api.routers.events._request_tenant_id",
                return_value="tenant_aaaaaaaaaaaa",
            ),
            patch("api.routers.events._is_platform_admin", return_value=False),
            patch("ocr_distributed.ssrf.safe_opener") as opener,
        ):
            resp = client.post(f"/api/v1/webhooks/dlq/{id_b}/retry")

        assert resp.status_code == 404
        opener.open.assert_not_called()

    def test_legacy_unresolvable_hidden_from_tenant_list(self, client, tmp_path):
        """Legacy unstamped unresolvable entries are admin-only."""
        import json

        dlq_file = tmp_path / "dlq.jsonl"
        legacy = {
            "id": "dlq_abc0000000000099",
            "job_id": "job_deadbeefdead",
            "webhook_url": "https://example.com/hook",
            "event_type": "job.completed",
            "payload": {"secret": "legacy"},
            "last_error": "gone",
            "attempts": 4,
            "created_at": "2026-01-01T00:00:00+00:00",
            "retried_at": None,
        }
        dlq_file.write_text(
            json.dumps(legacy, separators=(",", ":")) + "\n", encoding="utf-8"
        )

        with (
            patch("api.webhook_dlq._dlq_path", return_value=dlq_file),
            patch(
                "api.webhook_dlq._lookup_resource_tenant_id",
                return_value=None,
            ),
            patch(
                "api.routers.events._request_tenant_id",
                return_value="tenant_aaaaaaaaaaaa",
            ),
            patch("api.routers.events._is_platform_admin", return_value=False),
        ):
            resp = client.get("/api/v1/webhooks/dlq")

        assert resp.status_code == 200
        assert resp.json()["count"] == 0
        assert resp.json()["entries"] == []

    def test_legacy_unresolvable_get_404_for_tenant(self, client, tmp_path):
        """Tenant key cannot GET an unstamped unresolvable entry."""
        import json

        dlq_file = tmp_path / "dlq.jsonl"
        entry_id = "dlq_abc0000000000098"
        legacy = {
            "id": entry_id,
            "job_id": "job_deadbeefdead",
            "webhook_url": "https://example.com/hook",
            "event_type": "job.completed",
            "payload": {"secret": "legacy"},
            "last_error": "gone",
            "attempts": 4,
            "created_at": "2026-01-01T00:00:00+00:00",
            "retried_at": None,
        }
        dlq_file.write_text(
            json.dumps(legacy, separators=(",", ":")) + "\n", encoding="utf-8"
        )

        with (
            patch("api.webhook_dlq._dlq_path", return_value=dlq_file),
            patch(
                "api.webhook_dlq._lookup_resource_tenant_id",
                return_value=None,
            ),
            patch(
                "api.routers.events._request_tenant_id",
                return_value="tenant_aaaaaaaaaaaa",
            ),
            patch("api.routers.events._is_platform_admin", return_value=False),
        ):
            resp = client.get(f"/api/v1/webhooks/dlq/{entry_id}")

        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dlq_secret_lookup_failure_never_sends_unsigned_retry():
    """A stored secret failure must fail closed before the network call."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # Import the webhook module before mocking its source dependencies.  The
    # module uses imported aliases; importing it while these patches are
    # active would bind MagicMocks globally and contaminate later tests.
    import api.webhooks as _webhook_module  # noqa: F401
    from api.routers.events import retry_dlq_entry

    request = _make_tenant_request()
    entry = {
        "id": "dlq_aaaaaaaaaaaaaaaa",
        "job_id": "job_aaaaaaaaaaaa",
        "webhook_url": "https://example.com/hook",
        "event_type": "job.completed",
        "payload": {"event": "job.completed"},
        "retried_at": None,
    }
    session = MagicMock()
    session.get.return_value = SimpleNamespace(webhook_secret="encrypted-secret")
    factory = MagicMock(return_value=session)
    opener = MagicMock()

    with (
        patch("api.webhook_dlq.get_dlq_entry", return_value=entry),
        patch("api.webhook_dlq.claim_dlq_retry", return_value="claim-token"),
        patch("api.config.decrypt_webhook_secret", side_effect=ValueError("bad key")),
        patch("api.database.get_session_factory", return_value=factory),
        patch("ocr_distributed.ssrf.validate_webhook_url"),
        patch("ocr_distributed.ssrf.safe_opener", opener),
    ):
        result = await retry_dlq_entry(request, entry["id"], _auth=None)

    assert result["status"] == "failed"
    assert "delivery not attempted" in result["error"]
    opener.open.assert_not_called()
