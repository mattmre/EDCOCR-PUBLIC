"""Tenant-facing SLA/SLO snapshot helpers."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import api.config as config
from api.database import Job, get_session_factory


def get_slo_targets() -> dict[str, float | int]:
    """Return the configured tenant SLO target values."""
    return {
        "window_hours": int(config.TENANT_SLO_WINDOW_HOURS),
        "success_rate_min": float(config.TENANT_SLO_TARGET_SUCCESS_RATE),
        "p95_processing_seconds_max": float(
            config.TENANT_SLO_TARGET_P95_PROCESSING_SECONDS
        ),
    }


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    """Return a deterministic nearest-rank percentile for a non-empty list."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def build_tenant_slo_snapshot(
    tenant_id: str,
    *,
    window_hours: Optional[int] = None,
    session=None,
    now: Optional[datetime] = None,
) -> dict[str, object]:
    """Build a rolling SLO snapshot for one tenant."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        targets = get_slo_targets()
        effective_window_hours = int(window_hours or targets["window_hours"])
        if now is None:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
        window_start = now - timedelta(hours=effective_window_hours)

        jobs = (
            session.query(Job)
            .filter(
                Job.tenant_id == tenant_id,
                Job.created_at >= window_start,
            )
            .all()
        )

        completed_jobs = [job for job in jobs if job.status == "completed"]
        failed_jobs = [job for job in jobs if job.status == "failed"]
        cancelled_jobs = [job for job in jobs if job.status == "cancelled"]
        active_jobs = [
            job for job in jobs if job.status in {"submitted", "processing"}
        ]
        terminal_jobs = completed_jobs + failed_jobs + cancelled_jobs

        processing_times = [
            float(job.processing_time)
            for job in completed_jobs
            if job.processing_time is not None
        ]
        pages_processed = sum(int(job.pages_completed or 0) for job in completed_jobs)

        terminal_count = len(terminal_jobs)
        completed_count = len(completed_jobs)
        failed_count = len(failed_jobs)

        success_rate = (
            completed_count / terminal_count if terminal_count > 0 else 1.0
        )
        failure_rate = failed_count / terminal_count if terminal_count > 0 else 0.0
        avg_processing = (
            sum(processing_times) / len(processing_times)
            if processing_times
            else None
        )
        p95_processing = _percentile(processing_times, 0.95)

        throughput_jobs = completed_count / effective_window_hours
        throughput_pages = pages_processed / effective_window_hours

        success_rate_met = success_rate >= float(targets["success_rate_min"])
        p95_processing_met = (
            True
            if p95_processing is None
            else p95_processing <= float(targets["p95_processing_seconds_max"])
        )

        return {
            "tenant_id": tenant_id,
            "window_hours": effective_window_hours,
            "window_start": window_start,
            "window_end": now,
            "jobs_total": len(jobs),
            "terminal_jobs": terminal_count,
            "completed_jobs": completed_count,
            "failed_jobs": failed_count,
            "cancelled_jobs": len(cancelled_jobs),
            "active_jobs": len(active_jobs),
            "pages_processed": pages_processed,
            "success_rate": round(success_rate, 6),
            "failure_rate": round(failure_rate, 6),
            "avg_processing_seconds": (
                round(avg_processing, 6) if avg_processing is not None else None
            ),
            "p95_processing_seconds": (
                round(p95_processing, 6) if p95_processing is not None else None
            ),
            "throughput_jobs_per_hour": round(throughput_jobs, 6),
            "throughput_pages_per_hour": round(throughput_pages, 6),
            "targets": {
                "success_rate_min": float(targets["success_rate_min"]),
                "p95_processing_seconds_max": float(
                    targets["p95_processing_seconds_max"]
                ),
            },
            "status": {
                "success_rate_met": success_rate_met,
                "p95_processing_met": p95_processing_met,
                "overall_met": success_rate_met and p95_processing_met,
            },
        }
    finally:
        if own_session:
            session.close()
