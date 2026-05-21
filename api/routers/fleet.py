"""Worker fleet status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from api.fleet_status import get_fleet_tracker
from api.identity import require_role
from api.limits import get_default_rate, limiter

router = APIRouter(prefix="/api/v1/fleet", tags=["fleet"])


@router.get("", name="fleet_snapshot")
@limiter.limit(get_default_rate())
async def get_fleet(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator")),
):
    tracker = get_fleet_tracker()
    return tracker.get_snapshot().to_dict()


@router.get("/{worker_id}", name="fleet_worker_detail")
@limiter.limit(get_default_rate())
async def get_worker(
    request: Request,
    worker_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    tracker = get_fleet_tracker()
    worker = tracker.get_worker(worker_id)
    if worker is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Worker {worker_id} not found"},
        )
    return worker.to_dict()
