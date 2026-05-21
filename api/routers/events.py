"""Event replay and webhook DLQ management endpoints.

Provides:
- ``GET /api/v1/jobs/{job_id}/events`` -- replay stored events for a job
- ``GET /api/v1/webhooks/dlq`` -- list dead-letter queue entries
- ``GET /api/v1/webhooks/dlq/{entry_id}`` -- get a single DLQ entry
- ``POST /api/v1/webhooks/dlq/{entry_id}/retry`` -- retry a DLQ entry
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.database import Job, get_session_factory
from api.identity import require_role

router = APIRouter(tags=["events"])

_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12}$")
_DLQ_ID_RE = re.compile(r"^dlq_[0-9a-f]{16}$")


def _validate_job_id(job_id: str) -> None:
    """Raise 400 if job_id is malformed."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id format")


def _validate_dlq_id(entry_id: str) -> None:
    """Raise 400 if entry_id is malformed."""
    if not _DLQ_ID_RE.match(entry_id):
        raise HTTPException(status_code=400, detail="Invalid DLQ entry_id format")


def _request_tenant_id(request: Request):
    """Return tenant scope for authenticated tenant keys."""
    return getattr(request.state, "tenant_id", None)


def _verify_job_tenant(job_id: str, tenant_id) -> None:
    """Verify the job exists and belongs to the tenant; raise 404 otherwise."""
    session = get_session_factory()()
    try:
        query = session.query(Job.job_id).filter(Job.job_id == job_id)
        if tenant_id is not None:
            query = query.filter(Job.tenant_id == tenant_id)
        if not query.first():
            raise HTTPException(
                status_code=404,
                detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
            )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Event replay
# ---------------------------------------------------------------------------


@router.get("/api/v1/jobs/{job_id}/events")
async def get_job_events(
    request: Request,
    job_id: str,
    since_id: Optional[str] = Query(default=None, description="Return events after this event ID"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum events to return"),
    _auth: None = Depends(require_role("viewer", "operator", "admin")),
):
    """Replay stored events for a job.

    Returns a JSON array of events in chronological order.
    If ``since_id`` is provided, only events occurring after that
    event are returned (useful for reconnecting clients).
    """
    _validate_job_id(job_id)

    # Verify the job belongs to the requesting tenant (multi-tenant mode only).
    # In single-tenant mode (tenant_id is None), skip the ownership check so
    # that event replay works even when no Job row exists in the database.
    tenant_id = _request_tenant_id(request)
    if tenant_id is not None:
        _verify_job_tenant(job_id, tenant_id)

    from api.event_store import get_event_store

    store = get_event_store()
    events = store.get_events_since(job_id, since_id=since_id, limit=limit)

    return {
        "job_id": job_id,
        "events": events,
        "count": len(events),
    }


# ---------------------------------------------------------------------------
# Webhook DLQ
# ---------------------------------------------------------------------------


@router.get("/api/v1/webhooks/dlq")
async def list_dlq_entries(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum entries to return"),
    _auth: None = Depends(require_role("operator", "admin")),
):
    """List webhook dead-letter queue entries.

    Returns entries in reverse chronological order (most recent first).
    """
    from api.webhook_dlq import list_dlq

    entries = list_dlq(limit=limit)
    return {
        "entries": entries,
        "count": len(entries),
    }


@router.get("/api/v1/webhooks/dlq/{entry_id}")
async def get_dlq_entry_endpoint(
    request: Request,
    entry_id: str,
    _auth: None = Depends(require_role("operator", "admin")),
):
    """Get a single DLQ entry by ID."""
    _validate_dlq_id(entry_id)

    from api.webhook_dlq import get_dlq_entry

    entry = get_dlq_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="DLQ entry not found")

    return entry


@router.post("/api/v1/webhooks/dlq/{entry_id}/retry")
async def retry_dlq_entry(
    request: Request,
    entry_id: str,
    _auth: None = Depends(require_role("operator", "admin")),
):
    """Retry a failed webhook delivery from the dead-letter queue.

    Re-delivers the original payload to the webhook URL and marks
    the DLQ entry as retried.
    """
    _validate_dlq_id(entry_id)

    from api.webhook_dlq import get_dlq_entry, mark_dlq_retried

    entry = get_dlq_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="DLQ entry not found")

    if entry.get("retried_at"):
        raise HTTPException(
            status_code=409,
            detail="DLQ entry has already been retried",
        )

    # Attempt redelivery using the stored payload
    import json
    import urllib.error
    import urllib.request

    payload = entry["payload"]
    payload_json = json.dumps(payload, separators=(",", ":"))

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "OCR-Pipeline-Webhook-DLQ-Retry/1.0",
        "X-Webhook-Event": entry.get("event_type", ""),
        "X-Webhook-Job-ID": entry.get("job_id", ""),
        "X-DLQ-Retry": "true",
        "X-DLQ-Entry-ID": entry_id,
    }

    try:
        from ocr_distributed.ssrf import validate_webhook_url

        validate_webhook_url(
            entry["webhook_url"],
            allow_http=True,
            allow_private=True,
        )
    except (ValueError, ImportError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Webhook URL validation failed: {exc}",
        )

    try:
        from ocr_distributed.ssrf import safe_opener

        req = urllib.request.Request(
            entry["webhook_url"],
            data=payload_json.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with safe_opener.open(req, timeout=30) as resp:
            status_code = resp.getcode()

        if 200 <= status_code < 300:
            mark_dlq_retried(entry_id)
            return {
                "status": "delivered",
                "entry_id": entry_id,
                "http_status": status_code,
            }
        else:
            return {
                "status": "failed",
                "entry_id": entry_id,
                "http_status": status_code,
                "error": f"HTTP {status_code}",
            }

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return {
            "status": "failed",
            "entry_id": entry_id,
            "error": str(exc)[:500],
        }
