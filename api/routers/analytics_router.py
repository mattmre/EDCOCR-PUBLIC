"""Historical analytics endpoints."""

from __future__ import annotations

import re
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.analytics import TimeGranularity, get_analytics_store
from api.identity import require_role
from api.limits import get_default_rate, limiter

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

_TENANT_ID_RE = re.compile(r"^tenant_[0-9a-f]{12}$")
_MAX_ANALYTICS_LOOKBACK_HOURS = 24 * 7
_MAX_ANALYTICS_BREAKDOWN = 100


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


def _effective_analytics_tenant(request: Request, requested: Optional[str]) -> str:
    """Resolve tenant filter for analytics aggregates (authz-bound).

    Mirrors dashboard ``_effective_dashboard_tenant``:

    - Tenant-scoped keys are forced to their own tenant; mismatch → 403.
    - Platform admins may pass any valid ``tenant_id`` query value.
    - Single-tenant / anonymous auth (no ``request.state.tenant_id``) keeps
      optional filter semantics for backward compatibility.
    """
    from api.authz import require_tenant_scope_or_platform

    scoped = require_tenant_scope_or_platform(request)
    req = _validate_tenant_id(requested) if requested is not None else ""
    if scoped is not None:
        if req and req != scoped:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "tenant_mismatch",
                    "message": "Cannot query analytics for another tenant.",
                },
            )
        return scoped
    # platform admin or single-tenant: honour explicit request or empty (global)
    return req


@router.get("", name="analytics_summary")
@limiter.limit(get_default_rate())
async def get_summary(
    request: Request,
    hours: int = Query(
        24,
        ge=1,
        le=_MAX_ANALYTICS_LOOKBACK_HOURS,
        description="Lookback window in hours (maximum 7 days).",
    ),
    tenant_id: Optional[str] = Query(
        None, max_length=64, description="Filter by tenant"
    ),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _effective_analytics_tenant(request, tenant_id)
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    result = store.get_period_stats(start, end, tenant_id=tid).to_dict()
    if tid:
        result["tenant_id"] = tid
    return result


@router.get("/trends", name="analytics_trends")
@limiter.limit(get_default_rate())
async def get_trends(
    request: Request,
    period_hours: int = Query(
        24,
        ge=1,
        le=_MAX_ANALYTICS_LOOKBACK_HOURS,
    ),
    tenant_id: Optional[str] = Query(
        None, max_length=64, description="Filter by tenant"
    ),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _effective_analytics_tenant(request, tenant_id)
    store = get_analytics_store()
    result = store.get_trend(period_hours * 3600, tenant_id=tid).to_dict()
    if tid:
        result["tenant_id"] = tid
    return result


@router.get("/series", name="analytics_series")
@limiter.limit(get_default_rate())
async def get_series(
    request: Request,
    hours: int = Query(24, ge=1, le=_MAX_ANALYTICS_LOOKBACK_HOURS),
    granularity: str = Query("HOURLY", max_length=32),
    tenant_id: Optional[str] = Query(
        None, max_length=64, description="Filter by tenant"
    ),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _effective_analytics_tenant(request, tenant_id)
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    try:
        g = TimeGranularity[granularity]
    except KeyError:
        g = TimeGranularity.HOURLY
    return [s.to_dict() for s in store.get_time_series(start, end, g, tenant_id=tid)]


@router.get("/engines", name="analytics_top_engines")
@limiter.limit(get_default_rate())
async def get_top_engines(
    request: Request,
    hours: int = Query(24, ge=1, le=_MAX_ANALYTICS_LOOKBACK_HOURS),
    limit: int = Query(10, ge=1, le=_MAX_ANALYTICS_BREAKDOWN),
    tenant_id: Optional[str] = Query(
        None, max_length=64, description="Filter by tenant"
    ),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _effective_analytics_tenant(request, tenant_id)
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    return [
        {"engine": engine, "count": count}
        for engine, count in store.get_top_engines(start, end, limit, tenant_id=tid)
    ]


@router.get("/languages", name="analytics_top_languages")
@limiter.limit(get_default_rate())
async def get_top_languages(
    request: Request,
    hours: int = Query(24, ge=1, le=_MAX_ANALYTICS_LOOKBACK_HOURS),
    limit: int = Query(10, ge=1, le=_MAX_ANALYTICS_BREAKDOWN),
    tenant_id: Optional[str] = Query(
        None, max_length=64, description="Filter by tenant"
    ),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _effective_analytics_tenant(request, tenant_id)
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    return [
        {"language": lang, "count": count}
        for lang, count in store.get_top_languages(start, end, limit, tenant_id=tid)
    ]


@router.get("/workers", name="analytics_worker_stats")
@limiter.limit(get_default_rate())
async def get_worker_stats(
    request: Request,
    hours: int = Query(24, ge=1, le=_MAX_ANALYTICS_LOOKBACK_HOURS),
    tenant_id: Optional[str] = Query(
        None, max_length=64, description="Filter by tenant"
    ),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    tid = _effective_analytics_tenant(request, tenant_id)
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    return store.get_worker_stats(start, end, tenant_id=tid)
