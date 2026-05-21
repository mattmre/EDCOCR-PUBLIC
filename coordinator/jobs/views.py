"""API views for the coordinator.

Provides a lightweight JSON metrics endpoint for monitoring the
distributed pipeline health without requiring Prometheus.
"""

from django.db.models import Avg, Count
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_GET

from .metrics_auth import has_valid_metrics_key
from .models import Job, PageResult, Worker
from .prometheus_metrics import get_metrics_text

PAGE_PROCESSED_STATUSES = ("ok", "fallback", "image_only", "completed")


@require_GET
def dashboard(request):
    """Operator Dashboard view."""
    recent_jobs = Job.objects.order_by("-created_at")[:20]
    stats = {
        "jobs_total": Job.objects.count(),
        "workers_online": Worker.objects.filter(status=Worker.Status.ONLINE).count(),
        "pages_processed": PageResult.objects.filter(status__in=PAGE_PROCESSED_STATUSES).count(),
        "recent_jobs": recent_jobs,
    }
    return render(request, "dashboard.html", stats)

@require_GET
def metrics(request):
    """Return pipeline health metrics as JSON.

    Endpoint: GET /api/v1/metrics/

    Authentication: If METRICS_API_KEY is set, requires X-API-Key header.

    Returns a JSON object with:
    - jobs: counts by status, recent error rate
    - workers: counts by status, total fleet capacity
    - pages: total processed, average processing time
    """
    # Authenticate if METRICS_API_KEY is configured
    if not has_valid_metrics_key(request):
        return JsonResponse({"error": "Unauthorized"}, status=401)

    now = timezone.now()
    one_hour_ago = now - timezone.timedelta(hours=1)

    # Job metrics — single aggregated query instead of N+1
    job_counts = dict(
        Job.objects.values_list("status")
        .annotate(c=Count("pk"))
        .values_list("status", "c")
    )
    # Ensure all statuses are present in the dict
    for status_choice in Job.Status:
        job_counts.setdefault(status_choice.value, 0)

    recent_jobs = Job.objects.filter(created_at__gte=one_hour_ago)
    recent_failed = recent_jobs.filter(status=Job.Status.FAILED).count()
    recent_total = recent_jobs.count()
    error_rate_1h = (recent_failed / recent_total) if recent_total > 0 else 0.0

    # Worker metrics — single aggregated query instead of N+1
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

    # Page metrics
    page_stats = PageResult.objects.filter(status__in=PAGE_PROCESSED_STATUSES).aggregate(
        total=Count("id"),
        avg_time_ms=Avg("processing_time_ms"),
    )

    return JsonResponse({
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
        "timestamp": now.isoformat(),
    })


@require_GET
def prometheus_metrics(request):
    """Return pipeline health metrics in Prometheus text format.

    Endpoint: GET /api/v1/prometheus/

    Authentication: Same as /api/v1/metrics/ -- requires METRICS_API_KEY if set.
    """
    if not has_valid_metrics_key(request):
        return JsonResponse({"error": "Unauthorized"}, status=401)

    from django.http import HttpResponse

    body = get_metrics_text()
    return HttpResponse(body, content_type="text/plain; version=0.0.4; charset=utf-8")
