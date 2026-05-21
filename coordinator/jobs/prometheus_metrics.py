"""Prometheus metrics collector for the OCR coordinator.

Exposes pipeline health metrics in Prometheus text format via a custom
collector that queries Django ORM on each scrape.  Results are cached
with tier-specific TTLs to reduce database load under frequent scrapes.
"""

import hashlib
import json
import math

from django.utils import timezone
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
)

from .metrics_cache import _cache

# Registry isolated from default to avoid double-registration in tests.
# auto_describe is False because PipelineCollector provides its own describe().
REGISTRY = CollectorRegistry(auto_describe=False)

# Known page statuses to bound cardinality of ocr_pages_by_status metric.
_KNOWN_PAGE_STATUSES = ("ok", "fallback", "image_only", "completed", "failed", "pending")

# TTL tiers (seconds) for scrape caching.
_TTL_HOT = 15   # Fast-changing: error rates, completion rate, stuck jobs
_TTL_WARM = 30  # Moderate: job counts, worker counts, page stats
_TTL_COLD = 60  # Slow-changing: custody chain verification

# Histogram bucket boundaries for processing duration (in seconds).
_DURATION_BUCKETS = (0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600)


def _percentile(values: list, pct: float) -> float:
    """Compute the *pct*-th percentile from an unsorted list of numbers.

    Uses linear interpolation between surrounding ranks (matching NumPy
    default ``method='linear'``).  Returns 0.0 for empty input.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (n - 1)
    lower = int(math.floor(rank))
    upper = min(lower + 1, n - 1)
    frac = rank - lower
    result = sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])
    return round(result, 1)


class PipelineCollector:
    """Custom Prometheus collector that queries Django ORM on each scrape.

    This avoids stale counters -- metrics are always fresh from the database.
    ORM queries are grouped into three tiers with different cache TTLs:
    - hot (15s): error rates, completion rate, stuck count
    - warm (30s): job/worker counts, page stats, backend counts
    - cold (60s): custody chain violation checks
    """

    def describe(self):
        """Return metric descriptors without querying the database.

        This prevents the registry from calling collect() at registration
        time, which would fail before Django migrations have run.
        """
        yield GaugeMetricFamily("ocr_jobs_total", "Total jobs by status")
        yield GaugeMetricFamily("ocr_job_error_rate_1h", "Job failure rate over the last hour")
        yield GaugeMetricFamily("ocr_workers_total", "Total workers by status")
        yield GaugeMetricFamily("ocr_gpu_workers_available", "GPU workers online or busy")
        yield CounterMetricFamily("ocr_pages_processed_total", "Total pages processed by engine and status")
        yield GaugeMetricFamily("ocr_page_processing_time_avg_ms", "Average page processing time")
        yield GaugeMetricFamily("ocr_pages_by_status", "Pages by processing status")
        yield GaugeMetricFamily("ocr_custody_violations_total", "Custody chain violations in the last hour")
        yield GaugeMetricFamily("ocr_s3_job_error_rate_1h", "S3 job failure rate last hour")
        yield GaugeMetricFamily("ocr_job_completion_rate_1h", "Job completion rate last hour")
        yield GaugeMetricFamily("ocr_jobs_by_storage_backend", "Jobs by storage backend")
        yield GaugeMetricFamily("ocr_jobs_stuck_total", "Jobs stuck in processing/ingesting over 1 hour")
        yield GaugeMetricFamily("ocr_page_processing_time_p95_ms", "95th percentile page processing time")
        yield GaugeMetricFamily("ocr_page_processing_time_p99_ms", "99th percentile page processing time")
        yield GaugeMetricFamily("ocr_queue_depth", "Depth per pipeline stage queue")
        yield GaugeMetricFamily("ocr_dpi_escalation_total", "Pages that needed DPI escalation retry")
        yield GaugeMetricFamily("ocr_pages_by_engine", "Pages processed per OCR engine")
        # Tenant-scoped metrics
        yield GaugeMetricFamily("ocr_tenant_jobs_total", "Jobs by tenant and status")
        yield GaugeMetricFamily("ocr_tenant_pages_processed", "Pages processed per tenant")
        yield GaugeMetricFamily("ocr_tenant_error_rate", "Per-tenant error rate (1h window)")
        yield GaugeMetricFamily("ocr_tenant_processing_time_avg_ms", "Average processing time per tenant")
        yield HistogramMetricFamily(
            "ocr_processing_duration_seconds",
            "Processing duration per page in seconds",
            labels=["status", "engine", "tenant_id"],
        )

    def _collect_warm(self):
        """Collect moderate-frequency metrics: job/worker counts, page stats."""
        from django.db.models import Avg, Count

        from .models import Job, PageResult, Worker

        job_counts = dict(
            Job.objects.values_list("status")
            .annotate(c=Count("pk"))
            .values_list("status", "c")
        )

        worker_counts = dict(
            Worker.objects.values_list("status")
            .annotate(c=Count("pk"))
            .values_list("status", "c")
        )

        gpu_count = Worker.objects.filter(
            gpu_available=True,
            status__in=[Worker.Status.ONLINE, Worker.Status.BUSY],
        ).count()

        page_statuses = ("ok", "fallback", "image_only", "completed")
        page_stats = PageResult.objects.filter(status__in=page_statuses).aggregate(
            total=Count("id"),
            avg_time_ms=Avg("processing_time_ms"),
        )

        page_status_counts = dict(
            PageResult.objects.filter(status__in=_KNOWN_PAGE_STATUSES)
            .values_list("status")
            .annotate(c=Count("pk"))
            .values_list("status", "c")
        )

        backend_counts = dict(
            Job.objects.values_list("storage_backend_used")
            .annotate(c=Count("pk"))
            .values_list("storage_backend_used", "c")
        )

        # --- Percentile computation (p95, p99) ---
        # Fetch processing times from completed pages for percentile calculation.
        # Limit to recent 10k pages for bounded memory usage.
        processing_times = list(
            PageResult.objects.filter(
                status__in=page_statuses,
                processing_time_ms__gt=0,
            )
            .order_by("-pk")
            .values_list("processing_time_ms", flat=True)[:10000]
        )
        p95_ms = _percentile(processing_times, 95)
        p99_ms = _percentile(processing_times, 99)

        # --- Queue depth (M-11) ---
        # Approximate queue depth per pipeline stage from Job/PageResult status.
        # The 'queue' label uses the 6 logical pipeline stage names.
        processing_count = Job.objects.filter(status=Job.Status.PROCESSING).count()
        # Split GPU vs CPU processing by checking assigned_worker against
        # Worker capabilities.  Workers with gpu_available=True are GPU workers.
        gpu_hostnames = set(
            Worker.objects.filter(gpu_available=True)
            .values_list("hostname", flat=True)
        )
        gpu_processing = Job.objects.filter(
            status=Job.Status.PROCESSING,
            assigned_worker__in=gpu_hostnames,
        ).count() if gpu_hostnames else 0
        cpu_processing = processing_count - gpu_processing

        queue_depths = {
            "extraction": Job.objects.filter(status=Job.Status.INGESTING).count(),
            "ocr_gpu": gpu_processing,
            "ocr_cpu": cpu_processing,
            "compression": Job.objects.filter(status="compressing").count()
            if hasattr(Job.Status, "COMPRESSING")
            else 0,
            "nlp": PageResult.objects.filter(status="nlp_pending").count(),
            "assembly": Job.objects.filter(status=Job.Status.ASSEMBLING).count(),
        }

        # --- DPI escalation count ---
        # Pages with status="fallback" are those that needed DPI escalation
        # (Tesseract fallback after PaddleOCR low confidence).
        dpi_escalation_count = PageResult.objects.filter(
            status="fallback"
        ).count()

        # --- Pages by engine (ocr_method) ---
        engine_counts = dict(
            PageResult.objects.exclude(ocr_method="")
            .values_list("ocr_method")
            .annotate(c=Count("pk"))
            .values_list("ocr_method", "c")
        )

        # --- Page throughput by engine and status (M-13) ---
        # Normalize ocr_method to canonical engine labels (paddle, tesseract, onnx)
        # and page status to success/failed for the counter metric.
        _success_statuses = {"ok", "completed", "fallback", "image_only"}
        _failed_statuses = {"failed"}
        _engine_map = {
            "paddleocr": "paddle",
            "paddle": "paddle",
            "tesseract": "tesseract",
            "onnx": "onnx",
            "onnxruntime": "onnx",
            "imageonly": "onnx",  # image-only pages lack a true engine
        }
        throughput_rows = list(
            PageResult.objects.exclude(ocr_method="")
            .filter(status__in=list(_success_statuses | _failed_statuses))
            .values("ocr_method", "status")
            .annotate(c=Count("pk"))
        )
        page_throughput: dict[tuple[str, str], int] = {}
        for row in throughput_rows:
            raw_engine = (row["ocr_method"] or "").lower().replace(" ", "").replace("-", "").replace("_", "")
            engine_label = _engine_map.get(raw_engine, raw_engine)
            status_label = "success" if row["status"] in _success_statuses else "failed"
            key = (engine_label, status_label)
            page_throughput[key] = page_throughput.get(key, 0) + row["c"]
        # Also count failed pages that have no ocr_method set
        failed_no_engine = PageResult.objects.filter(
            status__in=list(_failed_statuses),
            ocr_method="",
        ).count()
        if failed_no_engine > 0:
            key = ("unknown", "failed")
            page_throughput[key] = page_throughput.get(key, 0) + failed_no_engine

        # --- Tenant-scoped metrics ---
        # Job counts grouped by tenant_id and status (exclude empty tenant_id)
        tenant_job_rows = list(
            Job.objects.exclude(tenant_id="")
            .values("tenant_id", "status")
            .annotate(c=Count("pk"))
        )

        # Pages processed per tenant
        tenant_pages_rows = list(
            PageResult.objects.filter(
                status__in=page_statuses,
                job__tenant_id__gt="",
            )
            .values("job__tenant_id")
            .annotate(total=Count("pk"))
        )

        # Avg processing time per tenant
        tenant_avg_time_rows = list(
            PageResult.objects.filter(
                status__in=page_statuses,
                processing_time_ms__gt=0,
                job__tenant_id__gt="",
            )
            .values("job__tenant_id")
            .annotate(avg_ms=Avg("processing_time_ms"))
        )

        # --- Processing duration histogram (status x engine x tenant_id) ---
        # Fetch recent page-level processing times grouped by status/engine/tenant.
        # Limit to last 24h to keep query bounded.
        now = timezone.now()
        one_day_ago = now - timezone.timedelta(hours=24)
        hist_rows = list(
            PageResult.objects.filter(
                processing_time_ms__gt=0,
                job__created_at__gte=one_day_ago,
            )
            .values_list("status", "ocr_method", "processing_time_ms", "job__tenant_id")
        )

        # Build histogram data: group by (status, engine, tenant_id) and compute buckets.
        hist_groups: dict[tuple[str, str, str], list[float]] = {}
        for row_status, row_method, row_ms, row_tenant in hist_rows:
            key = (
                row_status or "unknown",
                row_method or "unknown",
                row_tenant or "default",
            )
            hist_groups.setdefault(key, []).append(row_ms / 1000.0)

        # Pre-compute bucket counts, sum, and total count for each group.
        histogram_data: list[dict] = []
        for (hist_status, hist_engine, hist_tenant), values_sec in hist_groups.items():
            bucket_counts = []
            for bound in _DURATION_BUCKETS:
                bucket_counts.append(sum(1 for v in values_sec if v <= bound))
            total_count = len(values_sec)
            total_sum = sum(values_sec)
            histogram_data.append({
                "status": hist_status,
                "engine": hist_engine,
                "tenant_id": hist_tenant,
                "bucket_counts": bucket_counts,
                "count": total_count,
                "sum": round(total_sum, 3),
            })


        return {
            "job_counts": job_counts,
            "worker_counts": worker_counts,
            "gpu_count": gpu_count,
            "pages_total": page_stats["total"] or 0,
            "avg_time_ms": round(page_stats["avg_time_ms"] or 0, 1),
            "page_status_counts": page_status_counts,
            "backend_counts": backend_counts,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "queue_depths": queue_depths,
            "dpi_escalation_count": dpi_escalation_count,
            "engine_counts": engine_counts,
            "page_throughput": page_throughput,
            "tenant_job_rows": tenant_job_rows,
            "tenant_pages_rows": tenant_pages_rows,
            "tenant_avg_time_rows": tenant_avg_time_rows,
            "histogram_data": histogram_data,
        }

    def _collect_hot(self):
        """Collect high-frequency metrics: error rates, completion rate, stuck count."""
        from django.db.models import Count

        from .models import Job

        now = timezone.now()
        one_hour_ago = now - timezone.timedelta(hours=1)

        recent_jobs = Job.objects.filter(created_at__gte=one_hour_ago)
        recent_failed = recent_jobs.filter(status=Job.Status.FAILED).count()
        recent_total = recent_jobs.count()
        error_rate = round(recent_failed / recent_total, 4) if recent_total > 0 else 0.0

        # S3 error rate
        s3_recent = Job.objects.filter(
            created_at__gte=one_hour_ago,
            storage_backend_used="s3",
        )
        s3_failed = s3_recent.filter(status=Job.Status.FAILED).count()
        s3_total = s3_recent.count()
        s3_error_rate = round(s3_failed / s3_total, 4) if s3_total > 0 else 0.0

        # Completion rate
        recent_completed = recent_jobs.filter(status=Job.Status.COMPLETED).count()
        terminal_total = recent_completed + recent_failed
        completion_rate = round(recent_completed / terminal_total, 4) if terminal_total > 0 else 1.0

        # Stuck jobs
        stuck_count = Job.objects.filter(
            status__in=[Job.Status.PROCESSING, Job.Status.INGESTING],
            started_at__isnull=False,
            started_at__lt=one_hour_ago,
        ).count()

        # Per-tenant error rates (1h window)
        tenant_error_rows = []
        tenant_recent = recent_jobs.exclude(tenant_id="")
        tenant_totals = dict(
            tenant_recent.values_list("tenant_id")
            .annotate(c=Count("pk"))
            .values_list("tenant_id", "c")
        )
        tenant_failures = dict(
            tenant_recent.filter(status=Job.Status.FAILED)
            .values_list("tenant_id")
            .annotate(c=Count("pk"))
            .values_list("tenant_id", "c")
        )
        for tid, total in tenant_totals.items():
            failed = tenant_failures.get(tid, 0)
            rate = round(failed / total, 4) if total > 0 else 0.0
            tenant_error_rows.append({"tenant_id": tid, "error_rate": rate})

        return {
            "error_rate": error_rate,
            "s3_error_rate": s3_error_rate,
            "completion_rate": completion_rate,
            "stuck_count": stuck_count,
            "tenant_error_rows": tenant_error_rows,
        }

    def _collect_cold(self):
        """Collect low-frequency metrics: custody chain violations."""
        from .models import CustodyEvent

        now = timezone.now()
        one_hour_ago = now - timezone.timedelta(hours=1)

        violations = 0
        recent_doc_ids = list(
            CustodyEvent.objects.filter(
                chain_finalized=True,
                timestamp__gte=one_hour_ago,
            )
            .values_list("document_id", flat=True)
            .distinct()[:100]
        )
        for doc_id in recent_doc_ids:
            events = CustodyEvent.objects.filter(
                document_id=doc_id, chain_finalized=True
            ).order_by("timestamp")
            prev_hash = None
            for event in events:
                event_dict = {
                    "document_id": event.document_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(timespec="milliseconds"),
                    "data": event.data,
                    "prev_hash": prev_hash,
                }
                expected_hash = hashlib.sha256(
                    json.dumps(event_dict, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
                if event.event_hash != expected_hash:
                    violations += 1
                prev_hash = event.event_hash
        return violations

    def collect(self):
        # Fetch all three tiers through the cache
        warm = _cache.get_or_compute("warm", _TTL_WARM, self._collect_warm)
        hot = _cache.get_or_compute("hot", _TTL_HOT, self._collect_hot)
        violations = _cache.get_or_compute("cold", _TTL_COLD, self._collect_cold)

        # Import models lazily for status enums
        from .models import Job, Worker

        # ----- Job metrics -----
        job_status = GaugeMetricFamily(
            "ocr_jobs_total",
            "Total jobs by status",
            labels=["status"],
        )
        job_counts = warm["job_counts"]
        for status_choice in Job.Status:
            job_status.add_metric(
                [status_choice.value],
                job_counts.get(status_choice.value, 0),
            )
        yield job_status

        # Error rate (1h window)
        error_rate = GaugeMetricFamily(
            "ocr_job_error_rate_1h",
            "Job failure rate over the last hour (0.0 to 1.0)",
        )
        error_rate.add_metric([], hot["error_rate"])
        yield error_rate

        # ----- Worker metrics -----
        worker_status = GaugeMetricFamily(
            "ocr_workers_total",
            "Total workers by status",
            labels=["status"],
        )
        worker_counts = warm["worker_counts"]
        for status_choice in Worker.Status:
            worker_status.add_metric(
                [status_choice.value],
                worker_counts.get(status_choice.value, 0),
            )
        yield worker_status

        gpu_workers = GaugeMetricFamily(
            "ocr_gpu_workers_available",
            "Number of GPU workers currently online or busy",
        )
        gpu_workers.add_metric([], warm["gpu_count"])
        yield gpu_workers

        # ----- Page throughput counter (M-13: engine x status) -----
        pages_processed = CounterMetricFamily(
            "ocr_pages_processed_total",
            "Total pages processed by engine and status",
            labels=["engine", "status"],
        )
        for (engine_label, status_label), count in warm["page_throughput"].items():
            pages_processed.add_metric([engine_label, status_label], count)
        yield pages_processed

        avg_time = GaugeMetricFamily(
            "ocr_page_processing_time_avg_ms",
            "Average page processing time in milliseconds",
        )
        avg_time.add_metric([], warm["avg_time_ms"])
        yield avg_time

        # Pages by status (bounded to known statuses to prevent cardinality explosion)
        page_by_status = GaugeMetricFamily(
            "ocr_pages_by_status",
            "Pages by processing status",
            labels=["status"],
        )
        for s, c in warm["page_status_counts"].items():
            if s:
                page_by_status.add_metric([s], c)
        yield page_by_status

        # ----- Custody chain violations (last 1h) -----
        custody_violations_metric = GaugeMetricFamily(
            "ocr_custody_violations_total",
            "Custody chain hash violations detected in the last hour",
        )
        custody_violations_metric.add_metric([], violations)
        yield custody_violations_metric

        # ----- S3 job error rate (1h) -----
        s3_error_rate = GaugeMetricFamily(
            "ocr_s3_job_error_rate_1h",
            "S3-backed job failure rate over the last hour (0.0 to 1.0)",
        )
        s3_error_rate.add_metric([], hot["s3_error_rate"])
        yield s3_error_rate

        # ----- Job completion rate (1h) -----
        completion_rate = GaugeMetricFamily(
            "ocr_job_completion_rate_1h",
            "Job completion rate over the last hour (0.0 to 1.0)",
        )
        completion_rate.add_metric([], hot["completion_rate"])
        yield completion_rate

        # ----- Jobs by storage backend -----
        jobs_by_backend = GaugeMetricFamily(
            "ocr_jobs_by_storage_backend",
            "Total jobs by storage backend",
            labels=["backend"],
        )
        backend_counts = warm["backend_counts"]
        for backend in ("nfs", "s3", ""):
            label = backend if backend else "unset"
            jobs_by_backend.add_metric([label], backend_counts.get(backend, 0))
        yield jobs_by_backend

        # ----- Jobs stuck (processing/ingesting > 1 hour) -----
        jobs_stuck = GaugeMetricFamily(
            "ocr_jobs_stuck_total",
            "Jobs in processing/ingesting state for over 1 hour",
        )
        jobs_stuck.add_metric([], hot["stuck_count"])
        yield jobs_stuck

        # ----- Processing time percentiles -----
        p95_metric = GaugeMetricFamily(
            "ocr_page_processing_time_p95_ms",
            "95th percentile page processing time in milliseconds",
        )
        p95_metric.add_metric([], warm["p95_ms"])
        yield p95_metric

        p99_metric = GaugeMetricFamily(
            "ocr_page_processing_time_p99_ms",
            "99th percentile page processing time in milliseconds",
        )
        p99_metric.add_metric([], warm["p99_ms"])
        yield p99_metric

        # ----- Queue depth by stage (M-11) -----
        queue_depth = GaugeMetricFamily(
            "ocr_queue_depth",
            "Depth per pipeline stage queue",
            labels=["queue"],
        )
        for queue_label, depth in warm["queue_depths"].items():
            queue_depth.add_metric([queue_label], depth)
        yield queue_depth

        # ----- DPI escalation total -----
        dpi_escalation = GaugeMetricFamily(
            "ocr_dpi_escalation_total",
            "Pages that needed DPI escalation retry (fallback status)",
        )
        dpi_escalation.add_metric([], warm["dpi_escalation_count"])
        yield dpi_escalation

        # ----- Pages by engine -----
        pages_by_engine = GaugeMetricFamily(
            "ocr_pages_by_engine",
            "Pages processed per OCR engine",
            labels=["engine"],
        )
        for engine, count in warm["engine_counts"].items():
            pages_by_engine.add_metric([engine], count)
        yield pages_by_engine

        # ----- Tenant-scoped metrics -----
        tenant_jobs = GaugeMetricFamily(
            "ocr_tenant_jobs_total",
            "Total jobs by tenant and status",
            labels=["tenant_id", "status"],
        )
        for row in warm["tenant_job_rows"]:
            tenant_jobs.add_metric(
                [row["tenant_id"], row["status"]], row["c"],
            )
        yield tenant_jobs

        tenant_pages = GaugeMetricFamily(
            "ocr_tenant_pages_processed",
            "Total pages processed per tenant",
            labels=["tenant_id"],
        )
        for row in warm["tenant_pages_rows"]:
            tenant_pages.add_metric(
                [row["job__tenant_id"]], row["total"],
            )
        yield tenant_pages

        tenant_error_rate = GaugeMetricFamily(
            "ocr_tenant_error_rate",
            "Per-tenant job error rate over the last hour (0.0 to 1.0)",
            labels=["tenant_id"],
        )
        for row in hot["tenant_error_rows"]:
            tenant_error_rate.add_metric(
                [row["tenant_id"]], row["error_rate"],
            )
        yield tenant_error_rate

        tenant_avg_time = GaugeMetricFamily(
            "ocr_tenant_processing_time_avg_ms",
            "Average page processing time per tenant in milliseconds",
            labels=["tenant_id"],
        )
        for row in warm["tenant_avg_time_rows"]:
            tenant_avg_time.add_metric(
                [row["job__tenant_id"]],
                round(row["avg_ms"] or 0, 1),
            )
        yield tenant_avg_time

        # ----- Processing duration histogram (status x engine x tenant_id) -----
        duration_hist = HistogramMetricFamily(
            "ocr_processing_duration_seconds",
            "Processing duration per page in seconds",
            labels=["status", "engine", "tenant_id"],
        )
        for entry in warm["histogram_data"]:
            buckets = [
                (str(bound), entry["bucket_counts"][i])
                for i, bound in enumerate(_DURATION_BUCKETS)
            ]
            buckets.append(("+Inf", entry["count"]))
            duration_hist.add_metric(
                [entry["status"], entry["engine"], entry["tenant_id"]],
                buckets=buckets,
                sum_value=entry["sum"],
            )
        yield duration_hist


# Register the custom collector
REGISTRY.register(PipelineCollector())


def get_metrics_text():
    """Generate Prometheus text format metrics."""
    return generate_latest(REGISTRY)
