"""Dashboard metrics endpoints."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.dashboard import DashboardSnapshot, MetricWindow, get_collector
from api.identity import require_role
from api.limits import get_default_rate, limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

_TENANT_ID_RE = re.compile(r"^tenant_[0-9a-f]{12}$")
_MAX_TENANT_SNAPSHOT_JOBS = 100_000


def _validate_tenant_id(tenant_id: Optional[str]) -> str:
    """Validate and normalise an optional tenant_id query parameter.

    Returns the validated tenant_id string, or empty string when *None*
    is passed (i.e. parameter was omitted).  Raises 400 when the format
    is invalid.
    """
    if tenant_id is None:
        return ""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    return tenant_id


def _parse_window(window: str) -> "MetricWindow":
    """Parse a ``window`` query parameter, rejecting unknown values.

    The ``MINUTE_5`` default applies only when the caller omits the
    parameter (FastAPI ``Query`` default). An explicitly-passed unknown
    value is a caller bug and returns 422 instead of silently serving
    data for a window the caller did not ask for.
    """
    try:
        return MetricWindow[window]
    except KeyError:
        valid = ", ".join(sorted(m.name for m in MetricWindow))
        raise HTTPException(
            status_code=422,
            detail=f"Invalid window '{window}'. Valid values: {valid}",
        )


def _request_tenant_id(request: Request) -> Optional[str]:
    """Return authenticated tenant scope, if any."""
    return getattr(request.state, "tenant_id", None)


def _is_platform_admin(request: Request) -> bool:
    """True when server-attached tenant_permissions includes platform_admin.

    Trusts only ``request.state.tenant_permissions`` (API key path), never free-form
    JWT ``identity.claims["permissions"]`` (R4-RC-11 / RC-04 amp).
    """
    from api.authz import is_platform_admin

    return is_platform_admin(request)


def _effective_dashboard_tenant(request: Request, requested: Optional[str]) -> str:
    """Resolve tenant filter for dashboard metrics (authz-bound).

    - Tenant-scoped keys are forced to their own tenant; mismatch → 403.
    - Platform admins may pass any valid ``tenant_id`` query/path value.
    - Single-tenant / anonymous auth (no ``request.state.tenant_id``) keeps
      optional filter semantics for backward compatibility.
    """
    from api.authz import require_tenant_scope_or_platform

    # Fail-closed under multitenancy when no tenant binding (RC-01).
    scoped = require_tenant_scope_or_platform(request)
    req = _validate_tenant_id(requested) if requested is not None else ""
    if scoped is not None:
        if req and req != scoped:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "tenant_mismatch",
                    "message": "Cannot query dashboard metrics for another tenant.",
                },
            )
        return scoped
    # platform admin or single-tenant: honour explicit request or empty (global)
    return req


@router.get("", name="dashboard_snapshot")
@limiter.limit(get_default_rate())
async def get_snapshot(
    request: Request,
    window: str = Query("MINUTE_5", max_length=32, description="Time window"),
    tenant_id: Optional[str] = Query(
        None, max_length=64, description="Filter by tenant"
    ),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _effective_dashboard_tenant(request, tenant_id)
    collector = get_collector()
    w = _parse_window(window)
    result = collector.get_snapshot(w, tenant_id=tid).to_dict()
    if tid:
        result["tenant_id"] = tid
    return result


@router.get("/throughput", name="dashboard_throughput")
@limiter.limit(get_default_rate())
async def get_throughput(
    request: Request,
    window: str = Query("MINUTE_5", max_length=32),
    bucket_seconds: int = Query(60, ge=1, le=86_400),
    tenant_id: Optional[str] = Query(
        None, max_length=64, description="Filter by tenant"
    ),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _effective_dashboard_tenant(request, tenant_id)
    collector = get_collector()
    w = _parse_window(window)
    return collector.get_throughput_series(w, bucket_seconds, tenant_id=tid)


@router.get("/latency", name="dashboard_latency")
@limiter.limit(get_default_rate())
async def get_latency(
    request: Request,
    window: str = Query("MINUTE_5", max_length=32),
    bucket_seconds: int = Query(60, ge=1, le=86_400),
    tenant_id: Optional[str] = Query(
        None, max_length=64, description="Filter by tenant"
    ),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _effective_dashboard_tenant(request, tenant_id)
    collector = get_collector()
    w = _parse_window(window)
    return collector.get_latency_series(w, bucket_seconds, tenant_id=tid)


@router.get("/tenant/{tenant_id}", name="dashboard_tenant_snapshot")
@limiter.limit(get_default_rate())
async def get_tenant_snapshot(
    request: Request,
    tenant_id: str,
    window: str = Query("MINUTE_5", max_length=32, description="Time window"),
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Return a DashboardSnapshot filtered by tenant_id.

    Queries the job database for the given tenant within the requested
    time window and computes throughput and latency metrics.
    Requires both ENABLE_DASHBOARD and ENABLE_MULTITENANCY to be active.
    """
    enable_mt = os.environ.get("ENABLE_MULTITENANCY", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if not enable_mt:
        raise HTTPException(
            status_code=404,
            detail="Multi-tenancy is not enabled",
        )

    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")

    # Authz: tenant keys may only query their own path tenant_id, and an
    # unscoped principal must fail closed under multitenancy before this
    # explicit tenant path becomes an IDOR surface.
    from api.authz import require_tenant_scope_or_platform

    auth_tid = require_tenant_scope_or_platform(request)
    if (
        auth_tid is not None
        and not _is_platform_admin(request)
        and tenant_id != auth_tid
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tenant_mismatch",
                "message": "Cannot query dashboard metrics for another tenant.",
            },
        )

    w = _parse_window(window)

    # Query the job database for tenant-scoped metrics
    from datetime import datetime, timedelta, timezone

    from api.database import Job, get_session_factory

    session = get_session_factory()()
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = now - timedelta(seconds=w.value)

        jobs_query = session.query(Job).filter(
            Job.tenant_id == tenant_id, Job.created_at >= cutoff
        )
        jobs = jobs_query.limit(_MAX_TENANT_SNAPSHOT_JOBS + 1).all()
        if len(jobs) > _MAX_TENANT_SNAPSHOT_JOBS:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "dashboard_scope_too_large",
                    "message": (
                        "Tenant dashboard snapshot exceeds the configured "
                        "row safety limit. Narrow the time window or use the "
                        "aggregated dashboard endpoints."
                    ),
                },
            )

        completed = [j for j in jobs if j.status == "completed"]
        failed = [j for j in jobs if j.status == "failed"]
        active = [j for j in jobs if j.status in ("submitted", "processing")]
        queued = [j for j in jobs if j.status == "submitted"]

        # Throughput
        total_pages = sum(int(j.pages_completed or 0) for j in completed)
        total_docs = len(completed)
        elapsed = w.value

        pages_per_minute = (total_pages / elapsed) * 60 if elapsed else 0.0
        docs_per_hour = (total_docs / elapsed) * 3600 if elapsed else 0.0

        # Latency
        processing_times = sorted(
            float(j.processing_time) for j in completed if j.processing_time is not None
        )

        from api.dashboard import _percentile

        avg_lat = (
            sum(processing_times) / len(processing_times) * 1000
            if processing_times
            else 0.0
        )
        p50 = _percentile(processing_times, 50) * 1000 if processing_times else 0.0
        p95 = _percentile(processing_times, 95) * 1000 if processing_times else 0.0
        p99 = _percentile(processing_times, 99) * 1000 if processing_times else 0.0

        snapshot = DashboardSnapshot(
            timestamp=time.time(),
            pages_per_minute=pages_per_minute,
            docs_per_hour=docs_per_hour,
            bytes_per_second=0.0,
            avg_latency_ms=avg_lat,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            total_jobs=len(jobs),
            active_jobs=len(active),
            completed_jobs=len(completed),
            failed_jobs=len(failed),
            queued_jobs=len(queued),
        )
        result = snapshot.to_dict()
        result["tenant_id"] = tenant_id
        return result
    finally:
        session.close()
