"""API views for the coordinator.

Provides a lightweight JSON metrics endpoint for monitoring the
distributed pipeline health without requiring Prometheus.

R4-RC-07/08 residual: JSON metrics job/page queries must not fleet-scan
without tenant scope when multi-tenancy is enabled (or when a tenant is
presented). Worker fleet counts remain global (no tenant_id on Worker).
"""

import os

from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Avg, Count
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_GET

from .metrics_auth import extract_tenant_scope, has_valid_metrics_key
from .models import Job, PageResult, TaskDispatchOutbox, Worker
from .prometheus_metrics import get_metrics_text

PAGE_PROCESSED_STATUSES = ("ok", "fallback", "image_only", "completed")


@require_GET
def health_probe(request):
    """Unauthenticated coordinator liveness probe with a real DB check."""
    try:
        # A trivial ORM query proves the Django process can reach its DB.
        Job.objects.only("job_id").first()
    except Exception:
        return JsonResponse({"status": "unhealthy"}, status=503)
    return JsonResponse({"status": "healthy"})


def _multitenancy_enabled() -> bool:
    return os.environ.get("ENABLE_MULTITENANCY", "").lower() in ("1", "true", "yes")


def _require_tenant_for_job_metrics(request):
    """Return (tenant_id|None, error_response|None).

    When multi-tenancy is on, tenant_id is required for job/page aggregates
    (unscoped ORM fleet residual). Workers stay global.
    """
    tenant_id = extract_tenant_scope(request)
    if _multitenancy_enabled() and not tenant_id:
        return None, JsonResponse(
            {
                "error": "tenant_id is required",
                "message": (
                    "Multi-tenant metrics require tenant_id "
                    "(query param or X-Tenant-Id header)."
                ),
            },
            status=400,
        )
    return tenant_id, None


def _jobs_qs(tenant_id: str | None):
    qs = Job.objects.all()
    if tenant_id:
        qs = qs.filter(tenant_id=tenant_id)
    return qs


def _pages_qs(tenant_id: str | None):
    qs = PageResult.objects.filter(status__in=PAGE_PROCESSED_STATUSES)
    if tenant_id:
        qs = qs.filter(job__tenant_id=tenant_id)
    return qs


@require_GET
@staff_member_required
def dashboard(request):
    """Operator Dashboard view (staff session; tenant-pinned under multi-tenancy).

    RC-01 / C01: under ``ENABLE_MULTITENANCY``, unscoped staff fleet-scan of
    ``Job.objects.all()`` is refused — same rule as JSON metrics.
    Worker online counts remain global (no tenant_id on Worker).
    """
    tenant_id = extract_tenant_scope(request)
    if _multitenancy_enabled() and not tenant_id:
        return HttpResponse(
            "tenant_id is required under multi-tenancy "
            "(query param tenant_id or X-Tenant-Id header; RC-01/C01).",
            status=400,
            content_type="text/plain; charset=utf-8",
        )
    jobs_qs = _jobs_qs(tenant_id)
    recent_jobs = jobs_qs.order_by("-created_at")[:20]
    stats = {
        "jobs_total": jobs_qs.count(),
        "workers_online": Worker.objects.filter(status=Worker.Status.ONLINE).count(),
        "pages_processed": _pages_qs(tenant_id).count(),
        "recent_jobs": recent_jobs,
        "tenant_id": tenant_id,
    }
    return render(request, "dashboard.html", stats)


@require_GET
def metrics(request):
    """Return pipeline health metrics as JSON.

    Endpoint: GET /api/v1/metrics/

    Authentication: If METRICS_API_KEY is set, requires X-API-Key header.

    Under multi-tenancy, job/page aggregates require ``tenant_id`` (R4-RC-07/08).
    """
    # Authenticate if METRICS_API_KEY is configured
    if not has_valid_metrics_key(request):
        return JsonResponse({"error": "Unauthorized"}, status=401)

    tenant_id, err = _require_tenant_for_job_metrics(request)
    if err is not None:
        return err

    from .rls import tenant_rls_context

    with tenant_rls_context(tenant_id):
        return _metrics_json_body(tenant_id)


def _metrics_json_body(tenant_id: str | None):
    """Build metrics JSON under an active RLS tenant GUC."""
    now = timezone.now()
    one_hour_ago = now - timezone.timedelta(hours=1)
    jobs_qs = _jobs_qs(tenant_id)

    # Job metrics — single aggregated query instead of N+1
    job_counts = dict(
        jobs_qs.values_list("status").annotate(c=Count("pk")).values_list("status", "c")
    )
    # Ensure all statuses are present in the dict
    for status_choice in Job.Status:
        job_counts.setdefault(status_choice.value, 0)

    recent_jobs = jobs_qs.filter(created_at__gte=one_hour_ago)
    recent_failed = recent_jobs.filter(status=Job.Status.FAILED).count()
    recent_total = recent_jobs.count()
    error_rate_1h = (recent_failed / recent_total) if recent_total > 0 else 0.0

    # Worker metrics — single aggregated query instead of N+1 (fleet-global)
    worker_counts = dict(
        Worker.objects.values_list("status")
        .annotate(c=Count("pk"))
        .values_list("status", "c")
    )
    for status_choice in Worker.Status:
        worker_counts.setdefault(status_choice.value, 0)

    total_gpu_workers = Worker.objects.filter(
        gpu_available=True,
        status__in=[Worker.Status.ONLINE, Worker.Status.BUSY],
    ).count()

    # Page metrics (tenant-pinned when tenant_id set)
    page_stats = _pages_qs(tenant_id).aggregate(
        total=Count("id"),
        avg_time_ms=Avg("processing_time_ms"),
    )
    outbox_qs = TaskDispatchOutbox.objects.all()
    if tenant_id:
        outbox_qs = outbox_qs.filter(job__tenant_id=tenant_id)
    dead_outbox = outbox_qs.filter(status=TaskDispatchOutbox.Status.DEAD).count()

    return JsonResponse(
        {
            "jobs": {
                "by_status": job_counts,
                "total": sum(job_counts.values()),
                "error_rate_1h": round(error_rate_1h, 4),
            },
            "workers": {
                "by_status": worker_counts,
                "total": sum(worker_counts.values()),
                "gpu_available": total_gpu_workers,
            },
            "pages": {
                "total_processed": page_stats["total"] or 0,
                "avg_processing_time_ms": round(page_stats["avg_time_ms"] or 0, 1),
            },
            "dispatch_outbox": {"dead": dead_outbox},
            "tenant_id": tenant_id,
            "timestamp": now.isoformat(),
        }
    )


@require_GET
def prometheus_metrics(request):
    """Return pipeline health metrics in Prometheus text format.

    Endpoint: GET /api/v1/prometheus/

    Authentication: Same as /api/v1/metrics/ -- requires METRICS_API_KEY if set.
    Under multi-tenancy, tenant_id is required so fleet-wide ORM scrapes cannot
    dump all tenants' job aggregates (R4-RC-07/08 residual).
    """
    if not has_valid_metrics_key(request):
        return JsonResponse({"error": "Unauthorized"}, status=401)

    tenant_id, err = _require_tenant_for_job_metrics(request)
    if err is not None:
        return err

    from django.http import HttpResponse

    body = get_metrics_text(tenant_id=tenant_id)
    return HttpResponse(body, content_type="text/plain; version=0.0.4; charset=utf-8")
