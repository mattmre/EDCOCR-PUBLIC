"""Historical processing analytics and trend analysis.

Aggregates job processing history for trend analysis, reporting,
and capacity planning. Supports daily/hourly rollups.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TimeGranularity(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class JobRecord:
    """A completed job record for analytics."""
    job_id: str
    timestamp: float  # completion time
    pages: int = 0
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    success: bool = True
    engine: str = ""  # paddle, tesseract, etc.
    language: str = ""
    doc_type: str = ""  # classification
    worker_id: str = ""


@dataclass
class PeriodStats:
    """Aggregated stats for a time period."""
    period_start: float = 0.0
    period_end: float = 0.0
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    total_pages: int = 0
    total_bytes: int = 0
    avg_duration_seconds: float = 0.0
    p50_duration_seconds: float = 0.0
    p95_duration_seconds: float = 0.0
    avg_pages_per_job: float = 0.0
    pages_per_minute: float = 0.0
    success_rate: float = 0.0
    engine_breakdown: dict = field(default_factory=dict)  # engine -> count
    language_breakdown: dict = field(default_factory=dict)  # lang -> count
    doc_type_breakdown: dict = field(default_factory=dict)  # type -> count

    def to_dict(self) -> dict:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_jobs": self.total_jobs,
            "successful_jobs": self.successful_jobs,
            "failed_jobs": self.failed_jobs,
            "total_pages": self.total_pages,
            "total_bytes": self.total_bytes,
            "avg_duration_seconds": round(self.avg_duration_seconds, 2),
            "p50_duration_seconds": round(self.p50_duration_seconds, 2),
            "p95_duration_seconds": round(self.p95_duration_seconds, 2),
            "avg_pages_per_job": round(self.avg_pages_per_job, 2),
            "pages_per_minute": round(self.pages_per_minute, 2),
            "success_rate": round(self.success_rate, 4),
            "engine_breakdown": dict(self.engine_breakdown),
            "language_breakdown": dict(self.language_breakdown),
            "doc_type_breakdown": dict(self.doc_type_breakdown),
        }


@dataclass
class TrendAnalysis:
    """Trend analysis comparing two periods."""
    current: PeriodStats = field(default_factory=PeriodStats)
    previous: PeriodStats = field(default_factory=PeriodStats)
    throughput_change_pct: float = 0.0
    latency_change_pct: float = 0.0
    success_rate_change_pct: float = 0.0
    volume_change_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "current": self.current.to_dict(),
            "previous": self.previous.to_dict(),
            "changes": {
                "throughput_pct": round(self.throughput_change_pct, 2),
                "latency_pct": round(self.latency_change_pct, 2),
                "success_rate_pct": round(self.success_rate_change_pct, 2),
                "volume_pct": round(self.volume_change_pct, 2),
            },
        }


class AnalyticsStore:
    """Thread-safe store for historical job analytics."""

    def __init__(self, max_records: int = 100000):
        self._lock = threading.Lock()
        self._records: list = []  # List of JobRecord, sorted by timestamp
        self._max_records = max_records

    def record_job(self, job_id: str, pages: int = 0, duration_seconds: float = 0.0,
                   file_size_bytes: int = 0, success: bool = True,
                   engine: str = "", language: str = "", doc_type: str = "",
                   worker_id: str = "", timestamp: float = None):
        """Record a completed job."""
        rec = JobRecord(
            job_id=job_id,
            timestamp=timestamp if timestamp is not None else time.time(),
            pages=pages,
            duration_seconds=duration_seconds,
            file_size_bytes=file_size_bytes,
            success=success,
            engine=engine,
            language=language,
            doc_type=doc_type,
            worker_id=worker_id,
        )
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

    def get_period_stats(self, start: float, end: float) -> PeriodStats:
        """Compute aggregated stats for a time period."""
        with self._lock:
            records = [r for r in self._records if start <= r.timestamp < end]

        stats = PeriodStats(period_start=start, period_end=end)
        if not records:
            return stats

        stats.total_jobs = len(records)
        stats.successful_jobs = sum(1 for r in records if r.success)
        stats.failed_jobs = stats.total_jobs - stats.successful_jobs
        stats.total_pages = sum(r.pages for r in records)
        stats.total_bytes = sum(r.file_size_bytes for r in records)

        durations = [r.duration_seconds for r in records if r.duration_seconds > 0]
        if durations:
            stats.avg_duration_seconds = sum(durations) / len(durations)
            sorted_d = sorted(durations)
            stats.p50_duration_seconds = _percentile(sorted_d, 50)
            stats.p95_duration_seconds = _percentile(sorted_d, 95)

        stats.avg_pages_per_job = stats.total_pages / stats.total_jobs

        elapsed_minutes = (end - start) / 60
        if elapsed_minutes > 0:
            stats.pages_per_minute = stats.total_pages / elapsed_minutes

        stats.success_rate = stats.successful_jobs / stats.total_jobs if stats.total_jobs > 0 else 0.0

        # Breakdowns
        engine_counts = defaultdict(int)
        lang_counts = defaultdict(int)
        type_counts = defaultdict(int)
        for r in records:
            if r.engine:
                engine_counts[r.engine] += 1
            if r.language:
                lang_counts[r.language] += 1
            if r.doc_type:
                type_counts[r.doc_type] += 1
        stats.engine_breakdown = dict(engine_counts)
        stats.language_breakdown = dict(lang_counts)
        stats.doc_type_breakdown = dict(type_counts)

        return stats

    def get_time_series(self, start: float, end: float,
                        granularity: TimeGranularity = TimeGranularity.HOURLY) -> list:
        """Get time-series stats bucketed by granularity."""
        bucket_size = {
            TimeGranularity.HOURLY: 3600,
            TimeGranularity.DAILY: 86400,
            TimeGranularity.WEEKLY: 604800,
        }[granularity]

        result = []
        current = start
        while current < end:
            bucket_end = min(current + bucket_size, end)
            stats = self.get_period_stats(current, bucket_end)
            result.append(stats)
            current = bucket_end

        return result

    def get_trend(self, period_seconds: float = 86400) -> TrendAnalysis:
        """Compare current period to previous period of same length."""
        now = time.time()
        current_start = now - period_seconds
        previous_start = current_start - period_seconds

        current = self.get_period_stats(current_start, now)
        previous = self.get_period_stats(previous_start, current_start)

        trend = TrendAnalysis(current=current, previous=previous)

        # Calculate changes
        if previous.pages_per_minute > 0:
            trend.throughput_change_pct = (
                (current.pages_per_minute - previous.pages_per_minute)
                / previous.pages_per_minute * 100
            )

        if previous.avg_duration_seconds > 0:
            trend.latency_change_pct = (
                (current.avg_duration_seconds - previous.avg_duration_seconds)
                / previous.avg_duration_seconds * 100
            )

        if previous.success_rate > 0:
            trend.success_rate_change_pct = (
                (current.success_rate - previous.success_rate)
                / previous.success_rate * 100
            )

        if previous.total_jobs > 0:
            trend.volume_change_pct = (
                (current.total_jobs - previous.total_jobs)
                / previous.total_jobs * 100
            )

        return trend

    def get_top_engines(self, start: float, end: float, limit: int = 10) -> list:
        """Get engines ranked by usage count."""
        stats = self.get_period_stats(start, end)
        sorted_engines = sorted(stats.engine_breakdown.items(), key=lambda x: x[1], reverse=True)
        return sorted_engines[:limit]

    def get_top_languages(self, start: float, end: float, limit: int = 10) -> list:
        """Get languages ranked by usage count."""
        stats = self.get_period_stats(start, end)
        sorted_langs = sorted(stats.language_breakdown.items(), key=lambda x: x[1], reverse=True)
        return sorted_langs[:limit]

    def get_worker_stats(self, start: float, end: float) -> dict:
        """Get per-worker statistics."""
        with self._lock:
            records = [r for r in self._records if start <= r.timestamp < end and r.worker_id]

        worker_stats = defaultdict(lambda: {"jobs": 0, "pages": 0, "successes": 0, "failures": 0, "total_duration": 0.0})
        for r in records:
            ws = worker_stats[r.worker_id]
            ws["jobs"] += 1
            ws["pages"] += r.pages
            ws["total_duration"] += r.duration_seconds
            if r.success:
                ws["successes"] += 1
            else:
                ws["failures"] += 1

        result = {}
        for wid, ws in worker_stats.items():
            result[wid] = {
                "worker_id": wid,
                "total_jobs": ws["jobs"],
                "total_pages": ws["pages"],
                "successes": ws["successes"],
                "failures": ws["failures"],
                "avg_duration": round(ws["total_duration"] / ws["jobs"], 2) if ws["jobs"] > 0 else 0,
                "success_rate": round(ws["successes"] / ws["jobs"], 4) if ws["jobs"] > 0 else 0,
            }

        return result

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    def reset(self):
        """Clear all records."""
        with self._lock:
            self._records.clear()


def _percentile(sorted_data: list, pct: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# Global singleton
_store = None
_store_lock = threading.Lock()


def get_analytics_store() -> AnalyticsStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = AnalyticsStore()
    return _store
