"""Durable local event publication for job and batch lifecycle updates."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api import config

logger = logging.getLogger(__name__)

_EVENT_STREAM_LOCK = threading.Lock()


def _now_utc_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _event_stream_path() -> Path:
    """Return the configured event-stream path."""
    configured = getattr(config, "API_EVENT_STREAM_PATH", "")
    if configured:
        return Path(configured)
    return Path(config.OUTPUT_FOLDER) / "logs" / "api-events.jsonl"


def _append_event_record(record: dict[str, Any]) -> None:
    """Append a JSONL event record when event streaming is enabled."""
    if not getattr(config, "API_EVENT_STREAM_ENABLED", False):
        return

    path = _event_stream_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _EVENT_STREAM_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _job_event_record(job, event_type: str) -> dict[str, Any]:
    """Build a durable event record for a job lifecycle update."""
    return {
        "timestamp": _now_utc_iso(),
        "stream": "job",
        "event_type": event_type,
        "job_id": job.job_id,
        "status": job.status,
        "priority": job.priority,
        "tenant_id": job.tenant_id,
        "batch_id": job.batch_id,
        "source_file": job.source_file,
        "pages_completed": int(job.pages_completed or 0),
        "total_pages": int(job.total_pages or 0),
        "current_stage": job.current_stage or job.status,
        "result_path": job.result_path or "",
        "error_message": job.error_message or "",
        "processing_time": job.processing_time,
    }


def _job_ws_payload(job, event_type: str) -> dict[str, Any]:
    """Build a websocket-compatible payload for a job lifecycle update."""
    if event_type == "job.completed" or job.status == "completed":
        return {
            "type": "completed",
            "job_id": job.job_id,
            "status": "completed",
            "output_path": job.result_path or "",
            "pages_completed": int(job.pages_completed or 0),
        }

    if event_type == "job.failed" or job.status == "failed":
        return {
            "type": "failed",
            "job_id": job.job_id,
            "status": "failed",
            "error": job.error_message or "",
            "pages_completed": int(job.pages_completed or 0),
        }

    if event_type == "job.cancelled" or job.status == "cancelled":
        return {
            "type": "cancelled",
            "job_id": job.job_id,
            "status": "cancelled",
        }

    return {
        "type": "progress",
        "job_id": job.job_id,
        "status": job.status,
        "pages_completed": int(job.pages_completed or 0),
        "current_stage": job.current_stage or job.status,
    }


def _store_event_durable(job_id: str, event_type: str, record: dict[str, Any]) -> None:
    """Persist an event to the SQLite event store for replay support."""
    if not getattr(config, "EVENT_STORE_ENABLED", False):
        return
    try:
        from api.event_store import get_event_store

        store = get_event_store()
        store.store_event(event_type, job_id, record)
    except Exception:
        logger.exception(
            "Failed to store durable event %s for %s in event store",
            event_type,
            job_id,
        )


def publish_job_event(job, event_type: str) -> dict[str, Any]:
    """Persist and broadcast a job lifecycle event."""
    record = _job_event_record(job, event_type)
    try:
        _append_event_record(record)
    except Exception:
        logger.exception("Failed to append durable job event %s for %s", event_type, job.job_id)

    # Store in SQLite event store for replay
    _store_event_durable(job.job_id, event_type, record)

    try:
        from api.routers.ws import notify_job_update_sync

        notify_job_update_sync(job.job_id, _job_ws_payload(job, event_type))
    except Exception:
        logger.exception("Failed to forward job event %s for %s to websocket clients", event_type, job.job_id)

    return record


def _batch_event_record(batch, event_type: str) -> dict[str, Any]:
    """Build a durable event record for a batch lifecycle update."""
    return {
        "timestamp": _now_utc_iso(),
        "stream": "batch",
        "event_type": event_type,
        "batch_id": batch.batch_id,
        "status": batch.status,
        "priority": batch.priority,
        "total_jobs": int(batch.total_jobs or 0),
        "jobs_completed": int(batch.jobs_completed or 0),
        "jobs_failed": int(batch.jobs_failed or 0),
        "jobs_cancelled": int(batch.jobs_cancelled or 0),
        "processing_time": batch.processing_time,
    }


def publish_batch_event(batch, event_type: str) -> dict[str, Any]:
    """Persist a batch lifecycle event."""
    record = _batch_event_record(batch, event_type)
    try:
        _append_event_record(record)
    except Exception:
        logger.exception("Failed to append durable batch event %s for %s", event_type, batch.batch_id)

    # Store in SQLite event store for replay
    _store_event_durable(batch.batch_id, event_type, record)

    return record
