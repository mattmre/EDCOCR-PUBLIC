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


@router.get("", name="dashboard_snapshot")
@limiter.limit(get_default_rate())
async def get_snapshot(
    request: Request,
    window: str = Query("MINUTE_5", description="Time window"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _validate_tenant_id(tenant_id)
    collector = get_collector()
    try:
        w = MetricWindow[window]
    except KeyError:
        w = MetricWindow.MINUTE_5
    result = collector.get_snapshot(w, tenant_id=tid).to_dict()
    if tid:
        result["tenant_id"] = tid
    return result


@router.get("/throughput", name="dashboard_throughput")
@limiter.limit(get_default_rate())
async def get_throughput(
    request: Request,
    window: str = Query("MINUTE_5"),
    bucket_seconds: int = Query(60),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _validate_tenant_id(tenant_id)
    collector = get_collector()
    try:
        w = MetricWindow[window]
    except KeyError:
        w = MetricWindow.MINUTE_5
    return collector.get_throughput_series(w, bucket_seconds, tenant_id=tid)


@router.get("/latency", name="dashboard_latency")
@limiter.limit(get_default_rate())
async def get_latency(
    request: Request,
    window: str = Query("MINUTE_5"),
    bucket_seconds: int = Query(60),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _validate_tenant_id(tenant_id)
    collector = get_collector()
    try:
        w = MetricWindow[window]
    except KeyError:
        w = MetricWindow.MINUTE_5
    return collector.get_latency_series(w, bucket_seconds, tenant_id=tid)


@router.get("/tenant/{tenant_id}", name="dashboard_tenant_snapshot")
@limiter.limit(get_default_rate())
async def get_tenant_snapshot(
    request: Request,
    tenant_id: str,
    window: str = Query("MINUTE_5", description="Time window"),
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

    try:
        w = MetricWindow[window]
    except KeyError:
        w = MetricWindow.MINUTE_5

    # Query the job database for tenant-scoped metrics
    from datetime import datetime, timedelta, timezone

    from api.database import Job, get_session_factory

    session = get_session_factory()()
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = now - timedelta(seconds=w.value)

        jobs = (
            session.query(Job)
            .filter(Job.tenant_id == tenant_id, Job.created_at >= cutoff)
            .all()
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
            float(j.processing_time)
            for j in completed
            if j.processing_time is not None
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
