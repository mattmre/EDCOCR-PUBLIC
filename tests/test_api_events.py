"""Tests for api/events.py."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from api import events


def _job(**overrides):
    """Build a lightweight job-like object for event publication tests."""
    return SimpleNamespace(
        job_id="job_aaaaaaaaaaaa",
        status="processing",
        priority="normal",
        tenant_id=None,
        batch_id=None,
        source_file="sample.pdf",
        pages_completed=3,
        total_pages=10,
        current_stage="processing",
        result_path="",
        error_message="",
        processing_time=None,
        **overrides,
    )


def _batch(**overrides):
    """Build a lightweight batch-like object for event publication tests."""
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


def test_publish_job_event_appends_jsonl_and_notifies_websocket(tmp_path):
    """Job publication writes a durable JSONL line and bridges to websocket sync."""
    log_path = tmp_path / "logs" / "api-events.jsonl"

    with (
        patch.object(events.config, "API_EVENT_STREAM_ENABLED", True, create=True),
        patch.object(events.config, "API_EVENT_STREAM_PATH", str(log_path), create=True),
        patch("api.routers.ws.notify_job_update_sync") as mock_notify,
    ):
        record = events.publish_job_event(_job(), "job.progress")

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    stored = json.loads(lines[0])
    assert stored["event_type"] == "job.progress"
    assert stored["stream"] == "job"
    assert stored["job_id"] == "job_aaaaaaaaaaaa"
    assert record["pages_completed"] == 3

    mock_notify.assert_called_once()
    payload = mock_notify.call_args.args[1]
    assert payload["type"] == "progress"
    assert payload["status"] == "processing"
    assert payload["pages_completed"] == 3


def test_publish_batch_event_respects_disabled_stream(tmp_path):
    """Disabled streaming should skip JSONL creation."""
    log_path = tmp_path / "logs" / "api-events.jsonl"

    with (
        patch.object(events.config, "API_EVENT_STREAM_ENABLED", False, create=True),
        patch.object(events.config, "API_EVENT_STREAM_PATH", str(log_path), create=True),
    ):
        record = events.publish_batch_event(_batch(status="completed"), "batch.completed")

    assert record["event_type"] == "batch.completed"
    assert not log_path.exists()
