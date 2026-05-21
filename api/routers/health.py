"""Health check endpoint."""

from __future__ import annotations

import logging
import os
import shutil
import time

from fastapi import APIRouter, Request
from sqlalchemy import func, text

from api.database import Job, get_session_factory
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
    """Return pipeline health status."""
    # Job counts from database (single GROUP BY query)
    job_counts: dict[str, int] = {}
    try:
        factory = get_session_factory()
        session = factory()
        try:
            all_statuses = ("submitted", "processing", "completed", "failed", "cancelled")
            counts = dict(
                session.query(Job.status, func.count(Job.status))
                .filter(Job.status.in_(all_statuses))
                .group_by(Job.status)
                .all()
            )
            job_counts = {status: counts.get(status, 0) for status in all_statuses}
        finally:
            session.close()
    except Exception as e:
        logger.warning("Could not retrieve job counts for health check: %s", e, exc_info=True)

    return HealthResponse(
        status="healthy",
        version=_version,
        uptime_seconds=round(time.time() - _start_time, 1),
        jobs=job_counts,
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
        return SubsystemCheck(status="healthy", message="SQLite OK", latency_ms=round(ms, 1))
    except Exception as e:
        return SubsystemCheck(status="unhealthy", message=f"Database error: {e}")


def _check_disk(path: str, label: str, min_free_gb: float = 1.0) -> SubsystemCheck:
    """Check directory existence and free disk space."""
    try:
        if not os.path.isdir(path):
            return SubsystemCheck(
                status="unhealthy",
                message=f"{label} directory not found: {path}",
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
    except Exception as e:
        return SubsystemCheck(status="unhealthy", message=f"{label} check failed: {e}")


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
            return SubsystemCheck(status="healthy", message=f"FastText model found: {p}")
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
    except Exception as e:
        return SubsystemCheck(status="unhealthy", message=f"Heartbeat check failed: {e}")


def _check_external_translation() -> SubsystemCheck:
    """Report OCR-side EDC_TRANSLATION readiness without failing when disabled."""

    try:
        from ocr_local.translation.readiness import external_translation_readiness
        from pipeline_config import create_pipeline_config

        status = external_translation_readiness(create_pipeline_config())
    except Exception as e:
        return SubsystemCheck(
            status="unhealthy",
            message=f"EDC_TRANSLATION readiness check failed: {e}",
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
    return await health_check()


# ---------------------------------------------------------------------------
# Detailed health endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/features",
    name="pipeline_features",
    tags=["health"],
)
async def pipeline_features():
    """Return runtime availability of optional pipeline features."""
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
    checks: dict[str, SubsystemCheck] = {}
    job_counts: dict[str, int] = {}
    try:
        session_factory = get_session_factory()
        with session_factory() as session:
            rows = session.execute(
                text("SELECT status, COUNT(*) FROM jobs GROUP BY status")
            ).fetchall()
            job_counts = {row[0]: row[1] for row in rows}
    except Exception:
        pass

    # Database
    checks["database"] = _check_database()

    # Disk space
    output_dir = os.environ.get("OCR_OUTPUT_DIR", "/app/ocr_output")
    checks["disk_output"] = _check_disk(output_dir, "Output")

    source_dir = os.environ.get("OCR_SOURCE_DIR", "/app/ocr_source")
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

    return DetailedHealthResponse(
        status=overall,
        version=_version,
        uptime_seconds=round(time.time() - _start_time, 1),
        jobs=job_counts,
        checks=checks,
    )
