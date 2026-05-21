"""Consolidated usage, cost, and SLA reporting API (M-24).

Provides a single entry point that merges data from three overlapping systems:

1. ``cost_tracking.py`` -- standalone pipeline-level per-tenant cost tracker
   (in-memory + optional JSON persistence, used by pipeline threads).
2. ``sla_monitoring.py`` -- standalone SLA/SLO monitoring engine
   (sliding metric windows, used by pipeline and coordinator).
3. ``api/usage.py`` + ``api/slo.py`` -- database-backed per-tenant usage
   tracking and SLO snapshots (used by the REST API layer).

This module does **not** replace any of the three sources -- they continue to
be used directly by their respective subsystems.  Instead it wraps them into
a unified read-only query API so that API endpoints can serve a single
combined report rather than requiring callers to hit three different
endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class PipelineCostBreakdown(BaseModel):
    """Cost breakdown from the standalone cost_tracking module."""

    page_cost: float = 0.0
    gpu_cost: float = 0.0
    storage_cost: float = 0.0
    api_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"


class PipelineUsageSummary(BaseModel):
    """Usage counters from the standalone cost_tracking module."""

    tenant_id: str
    pages_processed: int = 0
    gpu_seconds: float = 0.0
    storage_bytes: int = 0
    api_calls: int = 0
    jobs_submitted: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    first_activity: str = ""
    last_activity: str = ""


class TenantCostSummary(BaseModel):
    """Unified cost summary for a tenant.

    Merges pipeline-level tracking (cost_tracking.py) with
    database-backed usage accounting (api/usage.py).
    """

    tenant_id: str
    pipeline_usage: Optional[PipelineUsageSummary] = None
    pipeline_cost: Optional[PipelineCostBreakdown] = None
    db_cost: Optional[dict[str, Any]] = None
    source: str = "none"  # "pipeline", "database", "both", "none"


class SloStatusItem(BaseModel):
    """A single SLO evaluation result."""

    name: str
    metric: str
    target: float
    unit: str
    current_value: float
    compliant: bool
    margin: float
    sample_count: int


class TenantSlaSummary(BaseModel):
    """Unified SLA summary for a tenant.

    Merges standalone SLA monitoring (sla_monitoring.py) with
    database-backed SLO snapshots (api/slo.py).
    """

    tenant_id: str
    pipeline_sla: Optional[dict[str, Any]] = None
    db_slo: Optional[dict[str, Any]] = None
    source: str = "none"  # "pipeline", "database", "both", "none"


class CombinedUsageReport(BaseModel):
    """Full combined report merging cost + SLA + usage data for a tenant."""

    tenant_id: str
    report_time: str
    cost: TenantCostSummary
    sla: TenantSlaSummary
    sources_available: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Delegation helpers (import-safe -- standalone modules may not be available)
# ---------------------------------------------------------------------------


def _get_pipeline_cost_data(tenant_id: str) -> Optional[dict[str, Any]]:
    """Fetch cost report from the standalone cost_tracking module.

    Returns None if the module is unavailable or the tenant has no data.
    """
    try:
        from cost_tracking import get_tracker
    except ImportError:
        logger.debug("cost_tracking module not available")
        return None

    tracker = get_tracker()
    return tracker.get_cost_report(tenant_id)


def _get_pipeline_sla_data(tenant_id: str) -> Optional[dict[str, Any]]:
    """Fetch SLA report from the standalone sla_monitoring module.

    Returns None if the module is unavailable or the tenant has no data.
    """
    try:
        from sla_monitoring import get_monitor
    except ImportError:
        logger.debug("sla_monitoring module not available")
        return None

    monitor = get_monitor()
    report = monitor.evaluate_tenant(tenant_id)
    return asdict(report)


def _get_db_cost_data(
    tenant_id: str, period: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Fetch cost data from the database-backed api.usage module.

    Returns None if the database is unavailable or no record exists.
    """
    try:
        from api.usage import build_cost_summary, get_usage
    except ImportError:
        logger.debug("api.usage module not available")
        return None

    try:
        usage = get_usage(tenant_id, period=period)
        if usage is None:
            return None
        cost = build_cost_summary(usage)
        return {
            "period": getattr(usage, "period", period),
            "jobs_submitted": int(getattr(usage, "jobs_submitted", 0) or 0),
            "pages_processed": int(getattr(usage, "pages_processed", 0) or 0),
            "storage_bytes_used": int(
                getattr(usage, "storage_bytes_used", 0) or 0
            ),
            "api_calls": int(getattr(usage, "api_calls", 0) or 0),
            "processing_seconds": float(
                getattr(usage, "processing_seconds", 0.0) or 0.0
            ),
            "estimated_costs": cost,
        }
    except Exception:
        logger.debug("Failed to fetch DB cost data for %s", tenant_id, exc_info=True)
        return None


def _get_db_slo_data(
    tenant_id: str, window_hours: Optional[int] = None
) -> Optional[dict[str, Any]]:
    """Fetch SLO snapshot from the database-backed api.slo module.

    Returns None if the database is unavailable or an error occurs.
    """
    try:
        from api.slo import build_tenant_slo_snapshot
    except ImportError:
        logger.debug("api.slo module not available")
        return None

    try:
        snapshot = build_tenant_slo_snapshot(
            tenant_id, window_hours=window_hours
        )
        # Convert datetime objects to ISO strings for JSON serialization
        for key in ("window_start", "window_end"):
            val = snapshot.get(key)
            if isinstance(val, datetime):
                snapshot[key] = val.isoformat()
        return snapshot
    except Exception:
        logger.debug("Failed to fetch DB SLO data for %s", tenant_id, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tenant_cost_summary(
    tenant_id: str,
    *,
    period: Optional[str] = None,
) -> TenantCostSummary:
    """Build a unified cost summary for *tenant_id*.

    Queries both the standalone ``cost_tracking`` module (pipeline-level)
    and the database-backed ``api.usage`` module (API-level).  The ``source``
    field indicates which data sources contributed to the result.
    """
    pipeline_data = _get_pipeline_cost_data(tenant_id)
    db_data = _get_db_cost_data(tenant_id, period=period)

    pipeline_usage = None
    pipeline_cost = None
    if pipeline_data:
        usage_raw = pipeline_data.get("usage", {})
        pipeline_usage = PipelineUsageSummary(
            tenant_id=usage_raw.get("tenant_id", tenant_id),
            pages_processed=usage_raw.get("pages_processed", 0),
            gpu_seconds=usage_raw.get("gpu_seconds", 0.0),
            storage_bytes=usage_raw.get("storage_bytes", 0),
            api_calls=usage_raw.get("api_calls", 0),
            jobs_submitted=usage_raw.get("jobs_submitted", 0),
            jobs_completed=usage_raw.get("jobs_completed", 0),
            jobs_failed=usage_raw.get("jobs_failed", 0),
            first_activity=usage_raw.get("first_activity", ""),
            last_activity=usage_raw.get("last_activity", ""),
        )
        cost_raw = pipeline_data.get("cost", {})
        pipeline_cost = PipelineCostBreakdown(
            page_cost=cost_raw.get("page_cost", 0.0),
            gpu_cost=cost_raw.get("gpu_cost", 0.0),
            storage_cost=cost_raw.get("storage_cost", 0.0),
            api_cost=cost_raw.get("api_cost", 0.0),
            total_cost=cost_raw.get("total_cost", 0.0),
            currency=cost_raw.get("currency", "USD"),
        )

    has_pipeline = pipeline_data is not None
    has_db = db_data is not None
    if has_pipeline and has_db:
        source = "both"
    elif has_pipeline:
        source = "pipeline"
    elif has_db:
        source = "database"
    else:
        source = "none"

    return TenantCostSummary(
        tenant_id=tenant_id,
        pipeline_usage=pipeline_usage,
        pipeline_cost=pipeline_cost,
        db_cost=db_data,
        source=source,
    )


def get_tenant_sla_summary(
    tenant_id: str,
    *,
    window_hours: Optional[int] = None,
) -> TenantSlaSummary:
    """Build a unified SLA summary for *tenant_id*.

    Queries both the standalone ``sla_monitoring`` module (pipeline-level
    sliding windows) and the database-backed ``api.slo`` module (job-level
    rolling snapshots).  The ``source`` field indicates which data sources
    contributed to the result.
    """
    pipeline_sla = _get_pipeline_sla_data(tenant_id)
    db_slo = _get_db_slo_data(tenant_id, window_hours=window_hours)

    has_pipeline = pipeline_sla is not None
    has_db = db_slo is not None
    if has_pipeline and has_db:
        source = "both"
    elif has_pipeline:
        source = "pipeline"
    elif has_db:
        source = "database"
    else:
        source = "none"

    return TenantSlaSummary(
        tenant_id=tenant_id,
        pipeline_sla=pipeline_sla,
        db_slo=db_slo,
        source=source,
    )


def get_combined_usage_report(
    tenant_id: str,
    *,
    period: Optional[str] = None,
    sla_window_hours: Optional[int] = None,
) -> CombinedUsageReport:
    """Build a combined usage report merging cost, SLA, and usage data.

    This is the primary entry point for API endpoints that want a single
    comprehensive view of a tenant's operational state.
    """
    cost = get_tenant_cost_summary(tenant_id, period=period)
    sla = get_tenant_sla_summary(tenant_id, window_hours=sla_window_hours)

    sources: list[str] = []
    if cost.pipeline_usage is not None or cost.pipeline_cost is not None:
        sources.append("cost_tracking")
    if cost.db_cost is not None:
        sources.append("api_usage_db")
    if sla.pipeline_sla is not None:
        sources.append("sla_monitoring")
    if sla.db_slo is not None:
        sources.append("api_slo_db")

    return CombinedUsageReport(
        tenant_id=tenant_id,
        report_time=datetime.now(timezone.utc).isoformat(),
        cost=cost,
        sla=sla,
        sources_available=sources,
    )
