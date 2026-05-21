"""Pipeline dashboard metrics collection and aggregation.

Collects real-time throughput, latency, and processing metrics for
the operations dashboard. Thread-safe with sliding window support.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MetricWindow(Enum):
    MINUTE_1 = 60
    MINUTE_5 = 300
    MINUTE_15 = 900
    HOUR_1 = 3600
    HOUR_24 = 86400


@dataclass
class ThroughputPoint:
    timestamp: float
    pages: int = 0
    documents: int = 0
    bytes_processed: int = 0
    tenant_id: str = ""


@dataclass
class LatencyPoint:
    timestamp: float
    job_id: str = ""
    total_ms: float = 0.0
    ocr_ms: float = 0.0
    compression_ms: float = 0.0
    queue_wait_ms: float = 0.0
    tenant_id: str = ""


@dataclass
class PipelineStageMetrics:
    """Metrics for a single pipeline stage."""
    stage: str
    active_workers: int = 0
    completed: int = 0
    failed: int = 0
    avg_latency_ms: float = 0.0
    queue_depth: int = 0


@dataclass
class DashboardSnapshot:
    """Point-in-time dashboard state."""
    timestamp: float = 0.0
    # Throughput
    pages_per_minute: float = 0.0
    docs_per_hour: float = 0.0
    bytes_per_second: float = 0.0
    # Latency
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    # Counts
    total_jobs: int = 0
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    queued_jobs: int = 0
    # Stages
    stages: list = field(default_factory=list)

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "throughput": {
                "pages_per_minute": round(self.pages_per_minute, 2),
                "docs_per_hour": round(self.docs_per_hour, 2),
                "bytes_per_second": round(self.bytes_per_second, 2),
            },
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 2),
                "p50_ms": round(self.p50_latency_ms, 2),
                "p95_ms": round(self.p95_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
            },
            "jobs": {
                "total": self.total_jobs,
                "active": self.active_jobs,
                "completed": self.completed_jobs,
                "failed": self.failed_jobs,
                "queued": self.queued_jobs,
            },
            "stages": [
                {
                    "stage": s.stage,
                    "active_workers": s.active_workers,
                    "completed": s.completed,
                    "failed": s.failed,
                    "avg_latency_ms": round(s.avg_latency_ms, 2),
                    "queue_depth": s.queue_depth,
                }
                for s in self.stages
            ],
        }


class MetricsCollector:
    """Thread-safe metrics collector with sliding window aggregation."""

    def __init__(self, max_window: int = 86400):
        self._lock = threading.Lock()
        self._max_window = max_window
        self._throughput: deque = deque()
        self._latency: deque = deque()
        self._job_counts = {
            "total": 0,
            "active": 0,
            "completed": 0,
            "failed": 0,
            "queued": 0,
        }
        self._tenant_job_counts: dict = {}  # tenant_id -> {total, active, ...}
        self._stage_metrics: dict = {}
        self._tenant_stage_metrics: dict = {}  # tenant_id -> {stage -> metrics}

    def record_throughput(self, pages: int = 0, documents: int = 0,
                         bytes_processed: int = 0, tenant_id: str = ""):
        """Record a throughput data point."""
        point = ThroughputPoint(
            timestamp=time.time(),
            pages=pages,
            documents=documents,
            bytes_processed=bytes_processed,
            tenant_id=tenant_id,
        )
        with self._lock:
            self._throughput.append(point)
            self._prune_old(self._throughput)

    def record_latency(self, job_id: str = "", total_ms: float = 0.0,
                       ocr_ms: float = 0.0, compression_ms: float = 0.0,
                       queue_wait_ms: float = 0.0, tenant_id: str = ""):
        """Record a latency data point."""
        point = LatencyPoint(
            timestamp=time.time(),
            job_id=job_id,
            total_ms=total_ms,
            ocr_ms=ocr_ms,
            compression_ms=compression_ms,
            queue_wait_ms=queue_wait_ms,
            tenant_id=tenant_id,
        )
        with self._lock:
            self._latency.append(point)
            self._prune_old(self._latency)

    def update_job_counts(self, total: int = 0, active: int = 0,
                          completed: int = 0, failed: int = 0, queued: int = 0,
                          tenant_id: str = ""):
        """Update current job counts.

        When *tenant_id* is provided the counts are stored in a per-tenant
        bucket **and** added to the global aggregate.  When omitted the
        global aggregate is set directly (backward-compatible behaviour).
        """
        counts = {
            "total": total,
            "active": active,
            "completed": completed,
            "failed": failed,
            "queued": queued,
        }
        with self._lock:
            if tenant_id:
                self._tenant_job_counts[tenant_id] = counts
            else:
                self._job_counts["total"] = total
                self._job_counts["active"] = active
                self._job_counts["completed"] = completed
                self._job_counts["failed"] = failed
                self._job_counts["queued"] = queued

    def update_stage(self, stage: str, active_workers: int = 0,
                     completed: int = 0, failed: int = 0,
                     avg_latency_ms: float = 0.0, queue_depth: int = 0,
                     tenant_id: str = ""):
        """Update metrics for a pipeline stage.

        When *tenant_id* is provided the stage metrics are stored in a
        per-tenant bucket.  When omitted the global stage metrics are set
        directly (backward-compatible behaviour).
        """
        metrics = PipelineStageMetrics(
            stage=stage,
            active_workers=active_workers,
            completed=completed,
            failed=failed,
            avg_latency_ms=avg_latency_ms,
            queue_depth=queue_depth,
        )
        with self._lock:
            if tenant_id:
                if tenant_id not in self._tenant_stage_metrics:
                    self._tenant_stage_metrics[tenant_id] = {}
                self._tenant_stage_metrics[tenant_id][stage] = metrics
            else:
                self._stage_metrics[stage] = metrics

    def get_snapshot(self, window: MetricWindow = MetricWindow.MINUTE_5,
                     tenant_id: str = "") -> DashboardSnapshot:
        """Get current dashboard snapshot for the given time window.

        When *tenant_id* is provided, only data points recorded for that
        tenant are included.  When omitted, all data is aggregated
        (backward-compatible behaviour).
        """
        now = time.time()
        cutoff = now - window.value

        with self._lock:
            # Filter to window (and optionally by tenant)
            if tenant_id:
                throughput_window = [
                    p for p in self._throughput
                    if p.timestamp >= cutoff and p.tenant_id == tenant_id
                ]
                latency_window = [
                    p for p in self._latency
                    if p.timestamp >= cutoff and p.tenant_id == tenant_id
                ]
            else:
                throughput_window = [p for p in self._throughput if p.timestamp >= cutoff]
                latency_window = [p for p in self._latency if p.timestamp >= cutoff]

            snapshot = DashboardSnapshot(timestamp=now)

            # Throughput
            if throughput_window:
                total_pages = sum(p.pages for p in throughput_window)
                total_docs = sum(p.documents for p in throughput_window)
                total_bytes = sum(p.bytes_processed for p in throughput_window)
                elapsed = window.value

                snapshot.pages_per_minute = (total_pages / elapsed) * 60
                snapshot.docs_per_hour = (total_docs / elapsed) * 3600
                snapshot.bytes_per_second = total_bytes / elapsed

            # Latency
            if latency_window:
                latencies = sorted(p.total_ms for p in latency_window)
                snapshot.avg_latency_ms = sum(latencies) / len(latencies)
                snapshot.p50_latency_ms = _percentile(latencies, 50)
                snapshot.p95_latency_ms = _percentile(latencies, 95)
                snapshot.p99_latency_ms = _percentile(latencies, 99)

            # Jobs
            if tenant_id and tenant_id in self._tenant_job_counts:
                jc = self._tenant_job_counts[tenant_id]
                snapshot.total_jobs = jc["total"]
                snapshot.active_jobs = jc["active"]
                snapshot.completed_jobs = jc["completed"]
                snapshot.failed_jobs = jc["failed"]
                snapshot.queued_jobs = jc["queued"]
            elif not tenant_id:
                snapshot.total_jobs = self._job_counts["total"]
                snapshot.active_jobs = self._job_counts["active"]
                snapshot.completed_jobs = self._job_counts["completed"]
                snapshot.failed_jobs = self._job_counts["failed"]
                snapshot.queued_jobs = self._job_counts["queued"]
            # else: tenant_id given but no counts stored → defaults (zeros)

            # Stages
            if tenant_id:
                tenant_stages = self._tenant_stage_metrics.get(tenant_id, {})
                snapshot.stages = list(tenant_stages.values())
            else:
                snapshot.stages = list(self._stage_metrics.values())

            return snapshot

    def get_throughput_series(self, window: MetricWindow = MetricWindow.HOUR_1,
                              bucket_seconds: int = 60,
                              tenant_id: str = "") -> list:
        """Get time-series throughput data bucketed by interval.

        When *tenant_id* is provided, only data points for that tenant
        are included.
        """
        now = time.time()
        cutoff = now - window.value

        with self._lock:
            if tenant_id:
                points = [
                    p for p in self._throughput
                    if p.timestamp >= cutoff and p.tenant_id == tenant_id
                ]
            else:
                points = [p for p in self._throughput if p.timestamp >= cutoff]

        if not points:
            return []

        # Bucket
        buckets = {}
        for p in points:
            bucket_key = int(p.timestamp / bucket_seconds) * bucket_seconds
            if bucket_key not in buckets:
                buckets[bucket_key] = {"timestamp": bucket_key, "pages": 0, "documents": 0, "bytes": 0}
            buckets[bucket_key]["pages"] += p.pages
            buckets[bucket_key]["documents"] += p.documents
            buckets[bucket_key]["bytes"] += p.bytes_processed

        return sorted(buckets.values(), key=lambda b: b["timestamp"])

    def get_latency_series(self, window: MetricWindow = MetricWindow.HOUR_1,
                            bucket_seconds: int = 60,
                            tenant_id: str = "") -> list:
        """Get time-series latency data bucketed by interval.

        When *tenant_id* is provided, only data points for that tenant
        are included.
        """
        now = time.time()
        cutoff = now - window.value

        with self._lock:
            if tenant_id:
                points = [
                    p for p in self._latency
                    if p.timestamp >= cutoff and p.tenant_id == tenant_id
                ]
            else:
                points = [p for p in self._latency if p.timestamp >= cutoff]

        if not points:
            return []

        buckets = {}
        for p in points:
            bucket_key = int(p.timestamp / bucket_seconds) * bucket_seconds
            if bucket_key not in buckets:
                buckets[bucket_key] = {"timestamp": bucket_key, "latencies": []}
            buckets[bucket_key]["latencies"].append(p.total_ms)

        result = []
        for bk in sorted(buckets.keys()):
            lats = buckets[bk]["latencies"]
            result.append({
                "timestamp": bk,
                "avg_ms": sum(lats) / len(lats),
                "p50_ms": _percentile(sorted(lats), 50),
                "p95_ms": _percentile(sorted(lats), 95),
                "count": len(lats),
            })

        return result

    def reset(self):
        """Clear all collected metrics."""
        with self._lock:
            self._throughput.clear()
            self._latency.clear()
            self._job_counts = {k: 0 for k in self._job_counts}
            self._tenant_job_counts.clear()
            self._stage_metrics.clear()
            self._tenant_stage_metrics.clear()

    def _prune_old(self, dq: deque):
        """Remove data points older than max_window."""
        cutoff = time.time() - self._max_window
        while dq and dq[0].timestamp < cutoff:
            dq.popleft()


def _percentile(sorted_data: list, pct: float) -> float:
    """Calculate percentile from pre-sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# Global singleton
_collector = None
_collector_lock = threading.Lock()


def get_collector() -> MetricsCollector:
    """Get or create the global MetricsCollector singleton."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector
