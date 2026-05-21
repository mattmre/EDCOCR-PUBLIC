"""Historical analytics endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Query, Request

from api.analytics import TimeGranularity, get_analytics_store
from api.identity import require_role
from api.limits import get_default_rate, limiter

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("", name="analytics_summary")
@limiter.limit(get_default_rate())
async def get_summary(
    request: Request,
    hours: int = Query(24, description="Lookback window in hours"),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    return store.get_period_stats(start, end).to_dict()


@router.get("/trends", name="analytics_trends")
@limiter.limit(get_default_rate())
async def get_trends(
    request: Request,
    period_hours: int = Query(24),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    store = get_analytics_store()
    return store.get_trend(period_hours * 3600).to_dict()


@router.get("/series", name="analytics_series")
@limiter.limit(get_default_rate())
async def get_series(
    request: Request,
    hours: int = Query(24),
    granularity: str = Query("HOURLY"),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    try:
        g = TimeGranularity[granularity]
    except KeyError:
        g = TimeGranularity.HOURLY
    return [s.to_dict() for s in store.get_time_series(start, end, g)]


@router.get("/engines", name="analytics_top_engines")
@limiter.limit(get_default_rate())
async def get_top_engines(
    request: Request,
    hours: int = Query(24),
    limit: int = Query(10),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    return [
        {"engine": engine, "count": count}
        for engine, count in store.get_top_engines(start, end, limit)
    ]


@router.get("/languages", name="analytics_top_languages")
@limiter.limit(get_default_rate())
async def get_top_languages(
    request: Request,
    hours: int = Query(24),
    limit: int = Query(10),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    return [
        {"language": lang, "count": count}
        for lang, count in store.get_top_languages(start, end, limit)
    ]


@router.get("/workers", name="analytics_worker_stats")
@limiter.limit(get_default_rate())
async def get_worker_stats(
    request: Request,
    hours: int = Query(24),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    store = get_analytics_store()
    end = time.time()
    start = end - (hours * 3600)
    return store.get_worker_stats(start, end)
