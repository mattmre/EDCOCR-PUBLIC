"""Queue alerting endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from api.identity import require_role
from api.limits import get_default_rate, limiter
from api.queue_alerting import AlertState, get_queue_monitor

router = APIRouter(tags=["alerts"])


class QueueThresholdRequest(BaseModel):
    """Operator-tunable queue alert thresholds."""

    warning_depth: int = Field(ge=0)
    critical_depth: int = Field(ge=1)
    warning_wait_seconds: float = Field(ge=0.0)
    critical_wait_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_ordering(self) -> "QueueThresholdRequest":
        if self.critical_depth < self.warning_depth:
            raise ValueError("critical_depth must be greater than or equal to warning_depth")
        if self.critical_wait_seconds < self.warning_wait_seconds:
            raise ValueError(
                "critical_wait_seconds must be greater than or equal to warning_wait_seconds"
            )
        return self


@router.get("/api/v1/alerts", name="alerts_snapshot")
@limiter.limit(get_default_rate())
async def get_alerts_snapshot(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator")),
):
    monitor = get_queue_monitor()
    return monitor.get_snapshot().to_dict()


@router.get("/api/v1/alerts/active", name="alerts_active")
@limiter.limit(get_default_rate())
async def get_active_alerts(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator")),
):
    monitor = get_queue_monitor()
    return [a.to_dict() for a in monitor.get_active_alerts()]


@router.get("/api/v1/queues/thresholds", name="queue_thresholds_list")
@limiter.limit(get_default_rate())
async def list_queue_thresholds(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator")),
):
    monitor = get_queue_monitor()
    return [threshold.to_dict() for threshold in monitor.list_thresholds()]


@router.get("/api/v1/queues/{queue_name}/threshold", name="queue_threshold_get")
@limiter.limit(get_default_rate())
async def get_queue_threshold(
    request: Request,
    queue_name: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    monitor = get_queue_monitor()
    threshold = monitor.get_threshold(queue_name)
    if threshold is None:
        raise HTTPException(status_code=404, detail=f"Queue threshold {queue_name} not found")
    return threshold.to_dict()


@router.put("/api/v1/queues/{queue_name}/threshold", name="queue_threshold_update")
@limiter.limit(get_default_rate())
async def update_queue_threshold(
    request: Request,
    queue_name: str,
    body: QueueThresholdRequest,
    _auth: None = Depends(require_role("admin")),
):
    monitor = get_queue_monitor()
    monitor.set_threshold(
        queue_name=queue_name,
        warning_depth=body.warning_depth,
        critical_depth=body.critical_depth,
        warning_wait_seconds=body.warning_wait_seconds,
        critical_wait_seconds=body.critical_wait_seconds,
    )
    threshold = monitor.get_threshold(queue_name)
    if threshold is None:
        raise HTTPException(status_code=500, detail="Queue threshold update failed")
    return threshold.to_dict()


@router.post("/api/v1/alerts/{alert_id}/acknowledge", name="alerts_acknowledge")
@limiter.limit(get_default_rate())
async def acknowledge_alert(
    request: Request,
    alert_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    monitor = get_queue_monitor()
    # acknowledge_alert returns None; check alert existence separately
    alert = monitor._alerts.get(alert_id)
    if alert is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Alert {alert_id} not found"},
        )
    if alert.state == AlertState.ACTIVE:
        monitor.acknowledge_alert(alert_id)
    return {"status": "acknowledged", "alert_id": alert_id}


@router.post("/api/v1/alerts/{alert_id}/resolve", name="alerts_resolve")
@limiter.limit(get_default_rate())
async def resolve_alert(
    request: Request,
    alert_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    monitor = get_queue_monitor()
    alert = monitor._alerts.get(alert_id)
    if alert is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Alert {alert_id} not found"},
        )
    if alert.state != AlertState.RESOLVED:
        monitor.resolve_alert(alert_id)
    return {"status": "resolved", "alert_id": alert_id}


@router.get("/api/v1/queues/{queue_name}/history", name="queue_history")
@limiter.limit(get_default_rate())
async def get_queue_history(
    request: Request,
    queue_name: str,
    window_seconds: int = Query(3600),
    _auth: None = Depends(require_role("admin", "operator")),
):
    monitor = get_queue_monitor()
    # get_queue_history returns a list of dicts already
    return monitor.get_queue_history(queue_name, window_seconds)
