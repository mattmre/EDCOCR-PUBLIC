"""API-side Prometheus metrics endpoint.

Exposes pipeline dashboard, fleet status, queue monitoring, SLA compliance,
and cost tracking metrics from the FastAPI process in Prometheus text format.
This is separate from the coordinator's ORM-backed Prometheus collector --
it surfaces real-time in-process metrics collected by the API server's
dashboard, fleet, queue, SLA, and cost singletons.

Auth: Requires X-API-Key header (same as other API endpoints).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from api.dashboard import MetricWindow, get_collector
from api.fleet_status import WorkerState, get_fleet_tracker
from api.identity import require_role
from api.queue_alerting import get_queue_monitor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prometheus"])

# Isolated registry to avoid collisions with the coordinator's default
# registry or any other prometheus_client users in the same process.
_REGISTRY = CollectorRegistry()

# --- Throughput gauges ---
_THROUGHPUT_PPM = Gauge(
    "ocr_api_throughput_pages_per_minute",
    "Pages processed per minute (5-min window)",
    registry=_REGISTRY,
)
_THROUGHPUT_DPH = Gauge(
    "ocr_api_throughput_docs_per_hour",
    "Documents processed per hour (5-min window)",
    registry=_REGISTRY,
)

# --- Latency gauges (labelled by percentile) ---
_LATENCY = Gauge(
    "ocr_api_latency_ms",
    "Processing latency in milliseconds",
    labelnames=["percentile"],
    registry=_REGISTRY,
)

# --- Pipeline stage gauges ---
_STAGE_QUEUE_DEPTH = Gauge(
    "ocr_api_stage_queue_depth",
    "Queue depth per pipeline stage",
    labelnames=["stage"],
    registry=_REGISTRY,
)
_STAGE_ACTIVE_WORKERS = Gauge(
    "ocr_api_stage_active_workers",
    "Active workers per pipeline stage",
    labelnames=["stage"],
    registry=_REGISTRY,
)

# --- Fleet gauges ---
_FLEET_WORKERS = Gauge(
    "ocr_api_fleet_workers",
    "Number of workers by state",
    labelnames=["state"],
    registry=_REGISTRY,
)
_FLEET_GPU_UTIL = Gauge(
    "ocr_api_fleet_gpu_utilization_pct",
    "Average GPU utilization across the fleet (percent)",
    registry=_REGISTRY,
)

# --- Queue monitor gauges ---
_QUEUE_DEPTH = Gauge(
    "ocr_api_queue_depth",
    "Queue depth per monitored queue",
    labelnames=["queue_name"],
    registry=_REGISTRY,
)
_QUEUE_ACTIVE_ALERTS = Gauge(
    "ocr_api_queue_active_alerts",
    "Number of active queue alerts",
    registry=_REGISTRY,
)

# --- SLA gauges (per-tenant) ---
_SLA_COMPLIANCE_PCT = Gauge(
    "ocr_sla_compliance_pct",
    "Overall SLA compliance percentage",
    labelnames=["tenant_id"],
    registry=_REGISTRY,
)
_SLA_VIOLATION_COUNT = Gauge(
    "ocr_sla_violation_count",
    "Current SLA violation count",
    labelnames=["tenant_id"],
    registry=_REGISTRY,
)
_SLA_AVAILABILITY_PCT = Gauge(
    "ocr_sla_availability_pct",
    "Current SLA availability percentage",
    labelnames=["tenant_id"],
    registry=_REGISTRY,
)
_SLA_P95_LATENCY_SECONDS = Gauge(
    "ocr_sla_p95_latency_seconds",
    "Current P95 latency in seconds",
    labelnames=["tenant_id"],
    registry=_REGISTRY,
)

# --- Cost gauges (per-tenant) ---
_COST_ESTIMATE_TOTAL = Gauge(
    "ocr_cost_estimate_total",
    "Total estimated cost in USD",
    labelnames=["tenant_id"],
    registry=_REGISTRY,
)
_TENANT_GPU_SECONDS = Gauge(
    "ocr_tenant_gpu_seconds",
    "GPU seconds consumed per tenant",
    labelnames=["tenant_id"],
    registry=_REGISTRY,
)
_TENANT_STORAGE_BYTES = Gauge(
    "ocr_tenant_storage_bytes",
    "Storage bytes consumed per tenant",
    labelnames=["tenant_id"],
    registry=_REGISTRY,
)

# --- Language detection metrics (Plan A -- PR A4) ---
# Counter of language-detection events, labelled by detected language
# (PaddleOCR code, e.g. "en", "fr", "ch"), the tier the language belongs to
# ("core" or "extended" per ``language_config``), and the detection level
# ("page" or "document").  Drives Grafana panels and alerting on the
# distribution of detected languages across the pipeline.
ocr_language_detected_total = Counter(
    "ocr_language_detected_total",
    "Language detection events by language, tier, and level",
    ["lang", "tier", "level"],
    registry=_REGISTRY,
)

# Histogram of language-detection confidence scores, bucketed for easy
# p50/p95/p99 computation in Grafana.  Labelled by PaddleOCR language code
# and detection level.
ocr_language_confidence = Histogram(
    "ocr_language_confidence",
    "Language detection confidence distribution",
    ["lang", "level"],
    buckets=[0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
    registry=_REGISTRY,
)

# Counter of pages where more than one Unicode script family was detected,
# labelled by the dominant script.  Used to surface multilingual corpora
# and to validate script-heuristic behaviour.
ocr_language_mixed_script_pages_total = Counter(
    "ocr_language_mixed_script_pages_total",
    "Pages with mixed Unicode script detected",
    ["primary_script"],
    registry=_REGISTRY,
)


def _refresh_metrics() -> None:
    """Pull latest values from the in-process singletons into the
    prometheus_client Gauge objects so that ``generate_latest`` returns
    fresh data.
    """
    # --- Dashboard metrics ---
    dashboard = get_collector()
    snap = dashboard.get_snapshot(MetricWindow.MINUTE_5)

    _THROUGHPUT_PPM.set(snap.pages_per_minute)
    _THROUGHPUT_DPH.set(snap.docs_per_hour)

    _LATENCY.labels(percentile="p50").set(snap.p50_latency_ms)
    _LATENCY.labels(percentile="p95").set(snap.p95_latency_ms)
    _LATENCY.labels(percentile="p99").set(snap.p99_latency_ms)

    # Stage-level metrics
    for stage in snap.stages:
        _STAGE_QUEUE_DEPTH.labels(stage=stage.stage).set(stage.queue_depth)
        _STAGE_ACTIVE_WORKERS.labels(stage=stage.stage).set(stage.active_workers)

    # --- Fleet metrics ---
    fleet = get_fleet_tracker()
    fleet_snap = fleet.get_snapshot()

    for ws in WorkerState:
        count = 0
        if ws == WorkerState.ONLINE:
            count = fleet_snap.online_workers
        elif ws == WorkerState.BUSY:
            count = fleet_snap.busy_workers
        elif ws == WorkerState.IDLE:
            count = fleet_snap.idle_workers
        elif ws == WorkerState.OFFLINE:
            count = fleet_snap.offline_workers
        elif ws == WorkerState.ERROR:
            count = fleet_snap.error_workers
        elif ws == WorkerState.DRAINING:
            count = fleet_snap.draining_workers
        _FLEET_WORKERS.labels(state=ws.value).set(count)

    _FLEET_GPU_UTIL.set(fleet_snap.avg_gpu_utilization_pct)

    # --- Queue monitor metrics ---
    qm = get_queue_monitor()
    q_snap = qm.get_snapshot()
    for q_info in q_snap.queues:
        _QUEUE_DEPTH.labels(queue_name=q_info["queue_name"]).set(q_info["depth"])
    _QUEUE_ACTIVE_ALERTS.set(len(q_snap.active_alerts))

    # --- SLA monitoring bridge ---
    try:
        from sla_monitoring import get_monitor

        monitor = get_monitor()
        # Enumerate tracked tenants from internal window keys
        with monitor._lock:
            tenant_ids = list(monitor._windows.keys())
        for tenant_id in tenant_ids:
            report = monitor.evaluate_tenant(tenant_id)
            _SLA_COMPLIANCE_PCT.labels(tenant_id=tenant_id).set(
                report.compliance_percentage
            )
            _SLA_VIOLATION_COUNT.labels(tenant_id=tenant_id).set(
                report.violation_count
            )
            # Extract availability and p95 latency from individual SLO statuses
            for slo_status in report.slo_statuses:
                if slo_status.definition.metric == "availability":
                    _SLA_AVAILABILITY_PCT.labels(tenant_id=tenant_id).set(
                        slo_status.current_value
                    )
                elif slo_status.definition.metric == "p95_latency":
                    _SLA_P95_LATENCY_SECONDS.labels(tenant_id=tenant_id).set(
                        slo_status.current_value
                    )
    except Exception:
        logger.debug("SLA monitoring bridge unavailable", exc_info=True)

    # --- Cost tracking bridge ---
    try:
        from cost_tracking import get_tracker

        tracker = get_tracker()
        for tenant_id, usage in tracker.get_all_usage().items():
            cost = usage.estimated_cost()
            _COST_ESTIMATE_TOTAL.labels(tenant_id=tenant_id).set(
                cost.get("total_cost", 0)
            )
            _TENANT_GPU_SECONDS.labels(tenant_id=tenant_id).set(
                usage.gpu_seconds
            )
            _TENANT_STORAGE_BYTES.labels(tenant_id=tenant_id).set(
                usage.storage_bytes
            )
    except Exception:
        logger.debug("Cost tracking bridge unavailable", exc_info=True)


@router.get("/api/v1/prometheus/", name="prometheus_metrics")
async def prometheus_metrics(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator")),
) -> Response:
    """Return Prometheus text-format metrics from the API process."""
    _refresh_metrics()
    body = generate_latest(_REGISTRY)
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
