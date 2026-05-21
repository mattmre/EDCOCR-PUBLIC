"""Integration tests for durable event streaming.

Verifies that:
- publish_job_event stores events in the SQLite event store
- Failed webhook deliveries are added to the dead-letter queue
- Event store is disabled when EVENT_STORE_ENABLED is False
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from api import events
from api.event_store import EventStore, reset_event_store


@pytest.fixture(autouse=True)
def _reset_store():
    """Ensure module singleton is clean between tests."""
    reset_event_store()
    yield
    reset_event_store()


def _job(**overrides):
    """Build a lightweight job-like object for event publication tests."""
    defaults = {
        "job_id": "job_aaaaaaaaaaaa",
        "status": "processing",
        "priority": "normal",
        "tenant_id": None,
        "batch_id": None,
        "source_file": "sample.pdf",
        "pages_completed": 3,
        "total_pages": 10,
        "current_stage": "processing",
        "result_path": "",
        "error_message": "",
        "processing_time": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _batch(**overrides):
    """Build a lightweight batch-like object."""
    payload = {
        "batch_id": "batch_aaaaaaaaaaaa",
        "status": "submitted",
        "priority": "normal",
        "total_jobs": 2,
        "jobs_completed": 0,
        "jobs_failed": 0,
        "jobs_cancelled": 0,
        "processing_time": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


# ---------------------------------------------------------------------------
# Event store integration with publish_job_event
# ---------------------------------------------------------------------------


class TestPublishJobEventStoresEvent:
    """Verify publish_job_event persists to the event store."""

    def test_stores_event_when_enabled(self, tmp_path):
        """publish_job_event stores event in SQLite when EVENT_STORE_ENABLED."""
        store = EventStore(db_path=str(tmp_path / "events.db"))

        with (
            patch.object(events.config, "API_EVENT_STREAM_ENABLED", False, create=True),
            patch.object(events.config, "EVENT_STORE_ENABLED", True, create=True),
            patch("api.event_store._store", store),
            patch("api.event_store._store_lock"),
            patch("api.routers.ws.notify_job_update_sync"),
        ):
            events.publish_job_event(_job(), "job.progress")

        stored = store.get_events_since("job_aaaaaaaaaaaa")
        assert len(stored) == 1
        assert stored[0]["event_type"] == "job.progress"
        assert stored[0]["payload"]["status"] == "processing"

    def test_skips_store_when_disabled(self, tmp_path):
        """publish_job_event does not store events when EVENT_STORE_ENABLED is False."""
        store = EventStore(db_path=str(tmp_path / "events.db"))

        with (
            patch.object(events.config, "API_EVENT_STREAM_ENABLED", False, create=True),
            patch.object(events.config, "EVENT_STORE_ENABLED", False, create=True),
            patch("api.event_store._store", store),
            patch("api.event_store._store_lock"),
            patch("api.routers.ws.notify_job_update_sync"),
        ):
            events.publish_job_event(_job(), "job.progress")

        stored = store.get_events_since("job_aaaaaaaaaaaa")
        assert len(stored) == 0

    def test_stores_terminal_events(self, tmp_path):
        """Terminal events (completed/failed/cancelled) are stored."""
        store = EventStore(db_path=str(tmp_path / "events.db"))

        with (
            patch.object(events.config, "API_EVENT_STREAM_ENABLED", False, create=True),
            patch.object(events.config, "EVENT_STORE_ENABLED", True, create=True),
            patch("api.event_store._store", store),
            patch("api.event_store._store_lock"),
            patch("api.routers.ws.notify_job_update_sync"),
        ):
            events.publish_job_event(
                _job(status="completed", result_path="/output"),
                "job.completed",
            )
            events.publish_job_event(
                _job(status="failed", error_message="boom"),
                "job.failed",
            )

        stored = store.get_events_since("job_aaaaaaaaaaaa")
        assert len(stored) == 2
        types = [e["event_type"] for e in stored]
        assert "job.completed" in types
        assert "job.failed" in types

    def test_event_store_failure_does_not_break_publish(self, tmp_path):
        """An exception from the event store does not prevent publication."""
        with (
            patch.object(events.config, "API_EVENT_STREAM_ENABLED", False, create=True),
            patch.object(events.config, "EVENT_STORE_ENABLED", True, create=True),
            patch("api.event_store.get_event_store", side_effect=RuntimeError("db error")),
            patch("api.routers.ws.notify_job_update_sync") as mock_ws,
        ):
            record = events.publish_job_event(_job(), "job.progress")

        # Publication still returns the record
        assert record["event_type"] == "job.progress"
        # WebSocket notification still fires
        mock_ws.assert_called_once()


# ---------------------------------------------------------------------------
# Batch event store integration
# ---------------------------------------------------------------------------


class TestPublishBatchEventStoresEvent:
    """Verify publish_batch_event persists to the event store."""

    def test_stores_batch_event(self, tmp_path):
        """publish_batch_event stores batch events in SQLite."""
        store = EventStore(db_path=str(tmp_path / "events.db"))

        with (
            patch.object(events.config, "API_EVENT_STREAM_ENABLED", False, create=True),
            patch.object(events.config, "EVENT_STORE_ENABLED", True, create=True),
            patch("api.event_store._store", store),
            patch("api.event_store._store_lock"),
        ):
            events.publish_batch_event(
                _batch(status="completed", jobs_completed=2),
                "batch.completed",
            )

        stored = store.get_events_since("batch_aaaaaaaaaaaa")
        assert len(stored) == 1
        assert stored[0]["event_type"] == "batch.completed"


# ---------------------------------------------------------------------------
# Webhook DLQ integration
# ---------------------------------------------------------------------------


class TestWebhookDlqIntegration:
    """Verify webhook delivery adds to DLQ on exhausted retries."""

    def test_dlq_entry_created_on_delivery_failure(self, tmp_path):
        """Failed webhook delivery creates a DLQ entry."""
        from api.database import Job, get_engine, get_session_factory, reset_engine

        reset_engine()
        db_file = str(tmp_path / "test.db")
        with (
            patch("api.config.DB_PATH", db_file),
            patch("api.database.DB_PATH", db_file),
        ):
            reset_engine()
            get_engine(db_file)
            factory = get_session_factory(db_file)

            # Create a job with webhook
            session = factory()
            job = Job(
                job_id="job_dlqtest00001",
                status="completed",
                source_file="test.pdf",
                webhook_url="https://example.com/failing-hook",
                webhook_secret="secret123",
            )
            session.add(job)
            session.commit()
            session.close()

            dlq_file = tmp_path / "dlq.jsonl"

            import urllib.error

            with (
                patch("api.config.WEBHOOK_ALLOW_HTTP", True),
                patch("api.config.WEBHOOK_ALLOW_PRIVATE", True),
                patch("api.config.WEBHOOK_DLQ_ENABLED", True),
                patch("api.config.WEBHOOK_DLQ_PATH", str(dlq_file)),
                patch("api.webhook_dlq._dlq_path", return_value=dlq_file),
                patch(
                    "api.webhooks._safe_opener.open",
                    side_effect=urllib.error.URLError("Connection refused"),
                ),
            ):
                from api.webhooks import deliver_webhook

                deliver_webhook(
                    "job_dlqtest00001",
                    factory,
                    webhook_timeout=5,
                    webhook_max_retries=0,  # 0 retries = 1 attempt only
                )

            # Verify DLQ entry was created
            assert dlq_file.exists()
            lines = dlq_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1

            entry = json.loads(lines[0])
            assert entry["job_id"] == "job_dlqtest00001"
            assert entry["webhook_url"] == "https://example.com/failing-hook"
            assert entry["event_type"] == "job.completed"
            assert "Connection refused" in entry["last_error"]

            reset_engine()
