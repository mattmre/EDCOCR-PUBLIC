"""Event replay and webhook DLQ management endpoints.

Provides:
- ``GET /api/v1/jobs/{job_id}/events`` -- replay stored events for a job
- ``GET /api/v1/webhooks/dlq`` -- list dead-letter queue entries
- ``GET /api/v1/webhooks/dlq/{entry_id}`` -- get a single DLQ entry
- ``POST /api/v1/webhooks/dlq/{entry_id}/retry`` -- retry a DLQ entry
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.database import Job, get_session_factory
from api.identity import require_role
from api.limits import get_default_rate, limiter

router = APIRouter(tags=["events"])
logger = logging.getLogger(__name__)

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
    """Return the request's authorized tenant scope."""
    from api.authz import require_tenant_scope_or_platform

    return require_tenant_scope_or_platform(request)


def _is_platform_admin(request: Request) -> bool:
    """True when server-attached tenant_permissions includes platform_admin.

    Trusts only ``request.state.tenant_permissions`` (API key path), never free-form
    JWT ``identity.claims["permissions"]`` (R4-RC-11 / RC-04 amp).
    """
    from api.authz import is_platform_admin

    return is_platform_admin(request)


def _dlq_scope_tenant(request: Request):
    """Tenant filter for DLQ list/get/retry.

    - Platform admin → None (global view).
    - Tenant-scoped key → forced to own tenant_id.
    - Multitenancy off + no tenant → None (single-tenant unscoped).
    - Multitenancy on without tenant/platform → 403 fail-closed (RC-01).

    Uses local ``_request_tenant_id`` / ``_is_platform_admin`` so unit tests can
    patch those symbols while sharing the same fail-closed policy as
    ``api.authz.require_tenant_scope_or_platform``.
    """
    if _is_platform_admin(request):
        return None
    tenant_id = _request_tenant_id(request)
    if tenant_id is not None:
        return tenant_id
    from api.config import ENABLE_MULTITENANCY

    if ENABLE_MULTITENANCY:
        from api.authz import require_tenant_scope_or_platform

        # Reuse shared error shape / policy for MT fail-closed.
        return require_tenant_scope_or_platform(request)
    return None


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
                detail={
                    "error": "job_not_found",
                    "message": f"Job {job_id} not found.",
                },
            )
    finally:
        session.close()


def _require_dlq_entry_access(entry: dict, request: Request) -> None:
    """Raise 404 when a tenant-scoped caller does not own the DLQ entry.

    Foreign / unstamped-unresolvable entries return 404 (not 403) to avoid
    leaking existence of other tenants' webhook payloads (AEP BE003).
    """
    scope = _dlq_scope_tenant(request)
    if scope is None:
        return
    from api.webhook_dlq import entry_owned_by_tenant

    if not entry_owned_by_tenant(entry, scope):
        raise HTTPException(status_code=404, detail="DLQ entry not found")


# ---------------------------------------------------------------------------
# Event replay
# ---------------------------------------------------------------------------


@router.get("/api/v1/jobs/{job_id}/events")
@limiter.limit(get_default_rate())
async def get_job_events(
    request: Request,
    job_id: str,
    since_id: Optional[str] = Query(
        default=None,
        max_length=128,
        description="Return events after this event ID",
    ),
    limit: int = Query(
        default=100, ge=1, le=1000, description="Maximum events to return"
    ),
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
@limiter.limit(get_default_rate())
async def list_dlq_entries(
    request: Request,
    limit: int = Query(
        default=100, ge=1, le=1000, description="Maximum entries to return"
    ),
    _auth: None = Depends(require_role("operator", "admin")),
):
    """List webhook dead-letter queue entries.

    Returns entries in reverse chronological order (most recent first).
    Tenant-scoped keys only see their own entries (AEP BE003); platform
    admins and single-tenant mode see the full queue.
    """
    from api.webhook_dlq import list_dlq

    scope = _dlq_scope_tenant(request)
    entries = list_dlq(limit=limit, tenant_id=scope)
    return {
        "entries": entries,
        "count": len(entries),
    }


@router.get("/api/v1/webhooks/dlq/{entry_id}")
@limiter.limit(get_default_rate())
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

    _require_dlq_entry_access(entry, request)
    return entry


@router.post("/api/v1/webhooks/dlq/{entry_id}/retry")
@limiter.limit(get_default_rate())
async def retry_dlq_entry(
    request: Request,
    entry_id: str,
    _auth: None = Depends(require_role("operator", "admin")),
):
    """Retry a failed webhook delivery from the dead-letter queue.

    Re-delivers the original payload to the webhook URL and marks
    the DLQ entry as retried. Ownership is checked before delivery
    so tenant keys cannot retry foreign payloads (AEP BE003).
    """
    _validate_dlq_id(entry_id)

    from api.webhook_dlq import (
        claim_dlq_retry,
        get_dlq_entry,
        mark_dlq_retried,
        release_dlq_retry,
    )

    entry = get_dlq_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="DLQ entry not found")

    # Ownership before any side-effect (retry delivery or mark).
    _require_dlq_entry_access(entry, request)

    if entry.get("retried_at"):
        raise HTTPException(
            status_code=409,
            detail="DLQ entry has already been retried",
        )

    # Attempt redelivery using the stored payload (S2-13 / bb4 CRITICAL).
    import json
    import os
    import time
    import urllib.error
    import urllib.request

    payload = entry["payload"]
    payload_json = json.dumps(payload, separators=(",", ":"))

    # SSRF fail-closed: never allow private/link-local by default; HTTP only
    # when WEBHOOK_ALLOW_HTTP=true (lab). Production must use HTTPS public URLs.
    allow_http = os.environ.get("WEBHOOK_ALLOW_HTTP", "").lower() in (
        "1",
        "true",
        "yes",
    )
    allow_private = os.environ.get("WEBHOOK_ALLOW_PRIVATE_RETRY", "").lower() in (
        "1",
        "true",
        "yes",
    )
    # LT-16: production hard-refuses private DLQ retry even if env set.
    _dep_dlq = (os.environ.get("DEPLOYMENT_ENV") or "").strip().lower()
    if _dep_dlq in ("production", "prod") and allow_private:
        raise HTTPException(
            status_code=400,
            detail=(
                "LT-16: WEBHOOK_ALLOW_PRIVATE_RETRY refused under production "
                "(private SSRF residual)"
            ),
        )
    try:
        from ocr_distributed.ssrf import (
            pinned_url_for_request,
            validate_webhook_url,
        )

        validate_webhook_url(
            entry["webhook_url"],
            allow_http=allow_http,
            allow_private=allow_private,
        )
    except (ValueError, ImportError):
        raise HTTPException(
            status_code=400,
            detail="Webhook URL validation failed.",
        )

    claim_token = claim_dlq_retry(entry_id)
    if claim_token is None:
        raise HTTPException(
            status_code=409,
            detail="DLQ entry is already retried or currently being retried",
        )

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "OCR-Pipeline-Webhook-DLQ-Retry/1.0",
        "X-Webhook-Event": entry.get("event_type", ""),
        "X-Webhook-Job-ID": entry.get("job_id", ""),
        "X-DLQ-Retry": "true",
        "X-DLQ-Entry-ID": entry_id,
    }

    # Re-sign like primary delivery path (api/webhooks.compute_signature).
    # A secret lookup/decryption failure is a delivery failure, not permission
    # to downgrade a previously signed callback to an unsigned one.
    secret = None
    try:
        from api import config as api_config
        from api.config import decrypt_webhook_secret
        from api.database import Batch, Job, get_session_factory
        from api.webhooks import compute_signature

        session = get_session_factory()()
        try:
            resource_id = entry.get("job_id")
            resource = session.get(Job, resource_id)
            if resource is None:
                resource = session.get(Batch, resource_id)
            stored_secret = getattr(resource, "webhook_secret", None)
            if stored_secret:
                secret = decrypt_webhook_secret(stored_secret)
            if not secret:
                secret = getattr(api_config, "WEBHOOK_SECRET", "") or os.environ.get(
                    "WEBHOOK_SECRET", ""
                )
        finally:
            session.close()
        if secret:
            ts = int(time.time())
            headers["X-Webhook-Signature"] = compute_signature(payload_json, secret, ts)
            headers["X-Webhook-Timestamp"] = str(ts)
    except Exception as exc:
        logger.error(
            "DLQ retry %s refused: webhook secret lookup/signing failed: %s",
            entry_id,
            exc,
        )
        release_dlq_retry(entry_id, claim_token)
        return {
            "status": "failed",
            "entry_id": entry_id,
            "error": "Webhook secret lookup/signing failed; delivery not attempted",
        }

    try:
        from ocr_distributed.ssrf import safe_opener

        with pinned_url_for_request(
            entry["webhook_url"],
            allow_http=allow_http,
            allow_private=allow_private,
        ) as delivery_url:
            req = urllib.request.Request(
                delivery_url,
                data=payload_json.encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with safe_opener.open(req, timeout=30) as resp:
                status_code = resp.getcode()

        if 200 <= status_code < 300:
            if not mark_dlq_retried(entry_id, claim_token=claim_token):
                logger.error(
                    "DLQ retry %s delivered but could not commit claim",
                    entry_id,
                )
            return {
                "status": "delivered",
                "entry_id": entry_id,
                "http_status": status_code,
            }
        else:
            release_dlq_retry(entry_id, claim_token)
            return {
                "status": "failed",
                "entry_id": entry_id,
                "http_status": status_code,
                "error": f"HTTP {status_code}",
            }

    except (ValueError, urllib.error.URLError, urllib.error.HTTPError, OSError):
        release_dlq_retry(entry_id, claim_token)
        return {
            "status": "failed",
            "entry_id": entry_id,
            "error": "Webhook delivery failed.",
        }
