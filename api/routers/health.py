"""Health check endpoint."""

from __future__ import annotations

import logging
import os
import shutil
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from api.database import get_session_factory
from api.identity import require_role
from api.models import DetailedHealthResponse, HealthResponse, SubsystemCheck

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

_start_time = time.time()

try:
    from ocr_local.config.version import __version__ as _version
except ImportError:
    _version = "unknown"


@router.get("/api/v1/health", name="health_check", response_model=HealthResponse)
async def health_check():
    """Return liveness/database health without exposing tenant workload counts."""
    try:
        factory = get_session_factory()
        session = factory()
        try:
            session.execute(text("SELECT 1"))
        finally:
            session.close()
    except Exception as e:
        logger.warning(
            "Could not retrieve job counts for health check: %s", e, exc_info=True
        )
        return HealthResponse(
            status="unhealthy",
            version=_version,
            uptime_seconds=round(time.time() - _start_time, 1),
        )

    return HealthResponse(
        status="healthy",
        version=_version,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ---------------------------------------------------------------------------
# Subsystem check helpers
# ---------------------------------------------------------------------------


def _check_database() -> SubsystemCheck:
    """Probe database connectivity with a lightweight query."""
    try:
        t0 = time.monotonic()
        session_factory = get_session_factory()
        with session_factory() as session:
            session.execute(text("SELECT 1"))
        ms = (time.monotonic() - t0) * 1000
        return SubsystemCheck(
            status="healthy", message="SQLite OK", latency_ms=round(ms, 1)
        )
    except Exception:
        logger.warning("Detailed health database probe failed", exc_info=True)
        return SubsystemCheck(status="unhealthy", message="Database unavailable")


def _check_disk(path: str, label: str, min_free_gb: float = 1.0) -> SubsystemCheck:
    """Check directory existence and free disk space."""
    try:
        if not os.path.isdir(path):
            return SubsystemCheck(
                status="unhealthy",
                message=f"{label} directory not found",
            )
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        if free_gb < min_free_gb:
            return SubsystemCheck(
                status="degraded",
                message=f"{label}: {free_gb:.1f} GB free (threshold: {min_free_gb} GB)",
            )
        return SubsystemCheck(
            status="healthy",
            message=f"{label}: {free_gb:.1f} GB free",
        )
    except Exception:
        logger.warning("Detailed health %s disk probe failed", label, exc_info=True)
        return SubsystemCheck(status="unhealthy", message=f"{label} check failed")


def _check_models() -> SubsystemCheck:
    """Verify that the FastText language-detection model is available."""
    model_path = "/app/lid.176.bin"
    for p in [
        model_path,
        "/app/models/lid.176.bin",
        "lid.176.bin",
        os.path.join(os.path.dirname(__file__), "..", "lid.176.bin"),
    ]:
        if os.path.isfile(p):
            return SubsystemCheck(status="healthy", message="FastText model available")
    return SubsystemCheck(
        status="degraded",
        message="FastText model not found (language detection will use fallback)",
    )


def _check_heartbeat(active_jobs: int | None = None) -> SubsystemCheck:
    """Check the pipeline monitor heartbeat file age."""
    heartbeat_path = os.environ.get("HEALTHCHECK_FILE", "/app/ocr_healthcheck")
    if not os.path.isfile(heartbeat_path):
        if active_jobs is not None and active_jobs <= 0:
            return SubsystemCheck(
                status="healthy",
                message="No active pipeline jobs",
            )
        return SubsystemCheck(
            status="degraded",
            message="Pipeline heartbeat file not found (pipeline may not be running)",
        )
    try:
        age = time.time() - os.path.getmtime(heartbeat_path)
        if active_jobs is not None and active_jobs <= 0:
            return SubsystemCheck(
                status="healthy",
                message=f"No active pipeline jobs; last heartbeat {age:.0f}s ago",
            )
        if age > 120:
            return SubsystemCheck(
                status="unhealthy",
                message=f"Heartbeat stale: {age:.0f}s old",
            )
        if age > 60:
            return SubsystemCheck(
                status="degraded",
                message=f"Heartbeat aging: {age:.0f}s old",
            )
        return SubsystemCheck(status="healthy", message=f"Heartbeat: {age:.0f}s ago")
    except Exception:
        logger.warning("Detailed health heartbeat probe failed", exc_info=True)
        return SubsystemCheck(status="unhealthy", message="Heartbeat check failed")


def _check_external_translation() -> SubsystemCheck:
    """Report OCR-side EDC_TRANSLATION readiness without failing when disabled."""

    try:
        from ocr_local.translation.readiness import external_translation_readiness
        from pipeline_config import create_pipeline_config

        status = external_translation_readiness(create_pipeline_config())
    except Exception:
        logger.warning(
            "Detailed health translation readiness probe failed", exc_info=True
        )
        return SubsystemCheck(
            status="unhealthy",
            message="EDC_TRANSLATION readiness check failed",
        )
    if status.status == "disabled":
        return SubsystemCheck(status="healthy", message=status.message)
    if status.ready:
        return SubsystemCheck(
            status="healthy",
            message=status.message,
            latency_ms=status.latency_ms,
        )
    return SubsystemCheck(
        status="degraded",
        message=status.message,
        latency_ms=status.latency_ms,
    )


@router.get("/api/v1/ready", name="ready_check", response_model=HealthResponse)
@router.get("/api/v1/readiness", name="readiness_check", response_model=HealthResponse)
async def readiness_check():
    """Kubernetes/operator-friendly readiness alias for the basic health probe."""
    result = await health_check()
    if result.status != "healthy":
        return JSONResponse(status_code=503, content=result.model_dump())
    return result


# ---------------------------------------------------------------------------
# Detailed health endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/features",
    name="pipeline_features",
    tags=["health"],
)
async def pipeline_features(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Return runtime availability of optional pipeline features (maintenance review)."""
    del request
    from feature_flags import get_pipeline_features

    return get_pipeline_features()


@router.get(
    "/api/v1/translation/readiness",
    name="external_translation_readiness_check",
    response_model=SubsystemCheck,
    tags=["health"],
)
async def external_translation_readiness_check():
    """Return OCR-side readiness for the optional external EDC_TRANSLATION service."""
    return _check_external_translation()


@router.get(
    "/api/v1/health/detailed",
    name="detailed_health_check",
    response_model=DetailedHealthResponse,
    tags=["health"],
)
async def detailed_health_check(request: Request):
    """Consolidated health check with subsystem probing."""
    # Lab unauthenticated probes stay open (middleware exemption). An
    # authenticated submit-only tenant key must not browse job counts or
    # host recon — same capability contract as dashboard GETs.
    if getattr(request.state, "identity", None) is not None:
        from api.authz import enforce_tenant_capability

        enforce_tenant_capability(request)

    checks: dict[str, SubsystemCheck] = {}
    job_counts: dict[str, int] = {}
    drain = None
    job_counts_check: SubsystemCheck | None = None
    # In a multi-tenant deployment an unauthenticated monitor must not receive
    # a global workload oracle, and a tenant monitor may only see its own
    # counts.  Platform admins retain the global view for operations.
    from api.authz import is_platform_admin
    from api.config import ENABLE_MULTITENANCY

    tenant_id = getattr(request.state, "tenant_id", None)
    is_platform = is_platform_admin(request)
    if ENABLE_MULTITENANCY and not is_platform and not tenant_id:
        count_sql = None
    elif ENABLE_MULTITENANCY and tenant_id:
        count_sql = text(
            "SELECT status, COUNT(*), SUM(CASE WHEN pid IS NOT NULL THEN 1 ELSE 0 END) FROM jobs "
            "WHERE tenant_id = :tenant_id GROUP BY status"
        )
    else:
        count_sql = text(
            "SELECT status, COUNT(*), SUM(CASE WHEN pid IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM jobs GROUP BY status"
        )
    try:
        if count_sql is not None:
            session_factory = get_session_factory()
            with session_factory() as session:
                params = {"tenant_id": tenant_id} if tenant_id else {}
                rows = session.execute(count_sql, params).fetchall()
                job_counts = {row[0]: row[1] for row in rows}
                drain = {
                    "scope": "tenant" if ENABLE_MULTITENANCY and tenant_id else "global",
                    "nonterminal_jobs": sum(
                        count for status, count in job_counts.items()
                        if status not in {"completed", "failed", "cancelled"}
                    ),
                    "pipeline_writers": sum(int(row[2] or 0) for row in rows),
                }
                if drain["scope"] == "global":
                    from api import job_manager
                    from api.background_work import active_count

                    manager = job_manager._manager_instance
                    drain["manager_tasks"] = (
                        manager._priority_queue.pending + manager._priority_queue.active
                        if manager is not None else 0
                    )
                    # Sample producers before consumers: a finalizing manager
                    # task can hand work to the webhook pool before releasing
                    # its slot. Sampling the reverse order can miss both.
                    drain["background_writers"] = active_count()
    except Exception:
        logger.warning("Detailed health job-count query failed", exc_info=True)
        # An empty count is not equivalent to zero jobs. Mark the result
        # unhealthy so monitoring cannot report a misleading green state.
        job_counts_check = SubsystemCheck(
            status="unhealthy", message="Job counts unavailable"
        )

    # Database
    if job_counts_check is not None:
        checks["job_counts"] = job_counts_check
    checks["database"] = _check_database()

    # Disk space
    # Use the same resolved paths as the application, rather than consulting
    # environment variables a second time.  This keeps health truthful when
    # the app is configured through a settings object or an embedding process
    # (and makes the probe test the actual ingest/output roots).
    from api import config

    output_dir = getattr(config, "OUTPUT_FOLDER", "/app/ocr_output")
    checks["disk_output"] = _check_disk(output_dir, "Output")

    source_dir = getattr(config, "SOURCE_FOLDER", "/app/ocr_source")
    checks["disk_source"] = _check_disk(source_dir, "Source")

    # FastText model
    checks["models"] = _check_models()

    # Pipeline heartbeat
    active_jobs = int(job_counts.get("submitted", 0) or 0) + int(
        job_counts.get("processing", 0) or 0
    )
    checks["pipeline"] = _check_heartbeat(active_jobs)

    # Optional EDC_TRANSLATION preflight
    checks["external_translation"] = _check_external_translation()

    # Derive overall status from worst individual check
    statuses = [c.status for c in checks.values()]
    if "unhealthy" in statuses:
        overall = "unhealthy"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    result = DetailedHealthResponse(
        status=overall,
        version=_version,
        uptime_seconds=round(time.time() - _start_time, 1),
        jobs=job_counts,
        checks=checks,
        drain=drain,
    )
    if overall == "unhealthy":
        return JSONResponse(status_code=503, content=result.model_dump())
    return result
