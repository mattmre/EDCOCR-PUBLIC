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
        patch("api.config.ENABLE_MULTITENANCY", False),
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
        with patch("api.event_store._store", store), \
             patch("api.event_store._store_lock"):
            store.store_event("job.submitted", "job_aaaaaaaaaaaa", {"status": "submitted"})
            store.store_event("job.completed", "job_aaaaaaaaaaaa", {"status": "completed"})

            resp = client.get("/api/v1/jobs/job_aaaaaaaaaaaa/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job_aaaaaaaaaaaa"
        assert data["count"] == 2
        assert len(data["events"]) == 2

    def test_since_id_filters_events(self, client, tmp_path):
        """since_id query parameter filters to events after the given ID."""
        store = EventStore(db_path=str(tmp_path / "events.db"))
        with patch("api.event_store._store", store), \
             patch("api.event_store._store_lock"):
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
        with patch("api.event_store._store", store), \
             patch("api.event_store._store_lock"):
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
