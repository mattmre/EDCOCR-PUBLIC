"""
Unit tests for pipeline dashboard metrics (api/dashboard.py).

Tests cover:
- MetricWindow enum values and counts
- ThroughputPoint creation and fields
- LatencyPoint creation and fields
- PipelineStageMetrics creation
- DashboardSnapshot defaults and to_dict
- MetricsCollector construction and recording
- Snapshot throughput/latency calculations
- Time-series bucketing (throughput + latency)
- Reset, pruning, percentile helpers
- Global singleton (get_collector)
- Thread safety (concurrent record calls)

Run with: python -m pytest tests/test_dashboard.py -v
"""

import threading
import time

import pytest

# Add project root to path
from api.dashboard import (
    DashboardSnapshot,
    LatencyPoint,
    MetricsCollector,
    MetricWindow,
    PipelineStageMetrics,
    ThroughputPoint,
    _percentile,
    get_collector,
)

# ---------------------------------------------------------------------------
# Tests: MetricWindow
# ---------------------------------------------------------------------------


class TestMetricWindow:
    def test_enum_values(self):
        assert MetricWindow.MINUTE_1.value == 60
        assert MetricWindow.MINUTE_5.value == 300
        assert MetricWindow.MINUTE_15.value == 900
        assert MetricWindow.HOUR_1.value == 3600
        assert MetricWindow.HOUR_24.value == 86400

    def test_enum_count(self):
        assert len(MetricWindow) == 5

    def test_enum_members(self):
        names = {m.name for m in MetricWindow}
        assert names == {"MINUTE_1", "MINUTE_5", "MINUTE_15", "HOUR_1", "HOUR_24"}


# ---------------------------------------------------------------------------
# Tests: ThroughputPoint
# ---------------------------------------------------------------------------


class TestThroughputPoint:
    def test_creation_defaults(self):
        tp = ThroughputPoint(timestamp=1000.0)
        assert tp.timestamp == 1000.0
        assert tp.pages == 0
        assert tp.documents == 0
        assert tp.bytes_processed == 0

    def test_creation_with_values(self):
        tp = ThroughputPoint(timestamp=2000.0, pages=10, documents=2, bytes_processed=4096)
        assert tp.pages == 10
        assert tp.documents == 2
        assert tp.bytes_processed == 4096

    def test_fields_mutable(self):
        tp = ThroughputPoint(timestamp=1000.0)
        tp.pages = 5
        assert tp.pages == 5


# ---------------------------------------------------------------------------
# Tests: LatencyPoint
# ---------------------------------------------------------------------------


class TestLatencyPoint:
    def test_creation_defaults(self):
        lp = LatencyPoint(timestamp=1000.0)
        assert lp.timestamp == 1000.0
        assert lp.job_id == ""
        assert lp.total_ms == 0.0
        assert lp.ocr_ms == 0.0
        assert lp.compression_ms == 0.0
        assert lp.queue_wait_ms == 0.0

    def test_creation_with_values(self):
        lp = LatencyPoint(
            timestamp=2000.0,
            job_id="job-123",
            total_ms=150.5,
            ocr_ms=100.0,
            compression_ms=30.0,
            queue_wait_ms=20.5,
        )
        assert lp.job_id == "job-123"
        assert lp.total_ms == 150.5
        assert lp.ocr_ms == 100.0
        assert lp.compression_ms == 30.0
        assert lp.queue_wait_ms == 20.5

    def test_fields_mutable(self):
        lp = LatencyPoint(timestamp=1000.0)
        lp.total_ms = 99.9
        assert lp.total_ms == 99.9


# ---------------------------------------------------------------------------
# Tests: PipelineStageMetrics
# ---------------------------------------------------------------------------


class TestPipelineStageMetrics:
    def test_creation_defaults(self):
        psm = PipelineStageMetrics(stage="ocr")
        assert psm.stage == "ocr"
        assert psm.active_workers == 0
        assert psm.completed == 0
        assert psm.failed == 0
        assert psm.avg_latency_ms == 0.0
        assert psm.queue_depth == 0

    def test_creation_with_values(self):
        psm = PipelineStageMetrics(
            stage="compression",
            active_workers=4,
            completed=100,
            failed=3,
            avg_latency_ms=45.2,
            queue_depth=12,
        )
        assert psm.stage == "compression"
        assert psm.active_workers == 4
        assert psm.completed == 100
        assert psm.failed == 3
        assert psm.avg_latency_ms == 45.2
        assert psm.queue_depth == 12


# ---------------------------------------------------------------------------
# Tests: DashboardSnapshot
# ---------------------------------------------------------------------------


class TestDashboardSnapshot:
    def test_defaults(self):
        snap = DashboardSnapshot()
        assert snap.timestamp == 0.0
        assert snap.pages_per_minute == 0.0
        assert snap.docs_per_hour == 0.0
        assert snap.bytes_per_second == 0.0
        assert snap.avg_latency_ms == 0.0
        assert snap.p50_latency_ms == 0.0
        assert snap.p95_latency_ms == 0.0
        assert snap.p99_latency_ms == 0.0
        assert snap.total_jobs == 0
        assert snap.active_jobs == 0
        assert snap.completed_jobs == 0
        assert snap.failed_jobs == 0
        assert snap.queued_jobs == 0
        assert snap.stages == []

    def test_to_dict_structure(self):
        snap = DashboardSnapshot(timestamp=1000.0)
        d = snap.to_dict()
        assert "timestamp" in d
        assert "throughput" in d
        assert "latency" in d
        assert "jobs" in d
        assert "stages" in d

    def test_to_dict_throughput_keys(self):
        snap = DashboardSnapshot(pages_per_minute=10.123, docs_per_hour=5.678, bytes_per_second=1024.999)
        d = snap.to_dict()
        assert d["throughput"]["pages_per_minute"] == 10.12
        assert d["throughput"]["docs_per_hour"] == 5.68
        assert d["throughput"]["bytes_per_second"] == 1025.0

    def test_to_dict_latency_keys(self):
        snap = DashboardSnapshot(avg_latency_ms=100.555, p50_latency_ms=80.111, p95_latency_ms=200.999, p99_latency_ms=500.006)
        d = snap.to_dict()
        assert d["latency"]["avg_ms"] == 100.56
        assert d["latency"]["p50_ms"] == 80.11
        assert d["latency"]["p95_ms"] == 201.0
        assert d["latency"]["p99_ms"] == 500.01

    def test_to_dict_jobs_keys(self):
        snap = DashboardSnapshot(total_jobs=10, active_jobs=3, completed_jobs=5, failed_jobs=1, queued_jobs=1)
        d = snap.to_dict()
        assert d["jobs"]["total"] == 10
        assert d["jobs"]["active"] == 3
        assert d["jobs"]["completed"] == 5
        assert d["jobs"]["failed"] == 1
        assert d["jobs"]["queued"] == 1

    def test_to_dict_stages(self):
        stage = PipelineStageMetrics(stage="ocr", active_workers=2, completed=50, failed=1, avg_latency_ms=33.333, queue_depth=5)
        snap = DashboardSnapshot(stages=[stage])
        d = snap.to_dict()
        assert len(d["stages"]) == 1
        assert d["stages"][0]["stage"] == "ocr"
        assert d["stages"][0]["avg_latency_ms"] == 33.33
        assert d["stages"][0]["queue_depth"] == 5

    def test_to_dict_empty_stages(self):
        snap = DashboardSnapshot()
        d = snap.to_dict()
        assert d["stages"] == []


# ---------------------------------------------------------------------------
# Tests: MetricsCollector
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_construction(self):
        mc = MetricsCollector()
        assert mc._max_window == 86400

    def test_construction_custom_window(self):
        mc = MetricsCollector(max_window=3600)
        assert mc._max_window == 3600

    def test_record_throughput_adds_point(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=5, documents=1, bytes_processed=2048)
        assert len(mc._throughput) == 1
        assert mc._throughput[0].pages == 5
        assert mc._throughput[0].documents == 1
        assert mc._throughput[0].bytes_processed == 2048

    def test_record_throughput_multiple(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=3)
        mc.record_throughput(pages=7)
        assert len(mc._throughput) == 2

    def test_record_latency_adds_point(self):
        mc = MetricsCollector()
        mc.record_latency(job_id="j1", total_ms=100.0, ocr_ms=80.0)
        assert len(mc._latency) == 1
        assert mc._latency[0].job_id == "j1"
        assert mc._latency[0].total_ms == 100.0
        assert mc._latency[0].ocr_ms == 80.0

    def test_record_latency_multiple(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=50.0)
        mc.record_latency(total_ms=150.0)
        assert len(mc._latency) == 2

    def test_update_job_counts(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=10, active=3, completed=5, failed=1, queued=1)
        assert mc._job_counts["total"] == 10
        assert mc._job_counts["active"] == 3
        assert mc._job_counts["completed"] == 5
        assert mc._job_counts["failed"] == 1
        assert mc._job_counts["queued"] == 1

    def test_update_job_counts_overwrites(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=5)
        mc.update_job_counts(total=10, active=2)
        assert mc._job_counts["total"] == 10
        assert mc._job_counts["active"] == 2

    def test_update_stage(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=4, completed=100, failed=2, avg_latency_ms=50.0, queue_depth=8)
        assert "ocr" in mc._stage_metrics
        assert mc._stage_metrics["ocr"].active_workers == 4
        assert mc._stage_metrics["ocr"].completed == 100

    def test_update_stage_overwrites(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=2)
        mc.update_stage("ocr", active_workers=6, completed=200)
        assert mc._stage_metrics["ocr"].active_workers == 6
        assert mc._stage_metrics["ocr"].completed == 200

    def test_update_stage_multiple_stages(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=2)
        mc.update_stage("compression", active_workers=1)
        assert len(mc._stage_metrics) == 2
        assert "ocr" in mc._stage_metrics
        assert "compression" in mc._stage_metrics


# ---------------------------------------------------------------------------
# Tests: get_snapshot
# ---------------------------------------------------------------------------


class TestGetSnapshot:
    def test_empty_collector(self):
        mc = MetricsCollector()
        snap = mc.get_snapshot()
        assert snap.pages_per_minute == 0.0
        assert snap.docs_per_hour == 0.0
        assert snap.bytes_per_second == 0.0
        assert snap.avg_latency_ms == 0.0
        assert snap.p50_latency_ms == 0.0
        assert snap.p95_latency_ms == 0.0
        assert snap.p99_latency_ms == 0.0
        assert snap.total_jobs == 0
        assert snap.stages == []

    def test_snapshot_has_timestamp(self):
        mc = MetricsCollector()
        before = time.time()
        snap = mc.get_snapshot()
        after = time.time()
        assert before <= snap.timestamp <= after

    def test_snapshot_throughput_calculation(self):
        mc = MetricsCollector()
        # Record 300 pages in the current window
        mc.record_throughput(pages=300, documents=10, bytes_processed=1_000_000)
        snap = mc.get_snapshot(window=MetricWindow.MINUTE_5)
        # 300 pages / 300 seconds * 60 = 60 pages/min
        assert snap.pages_per_minute == pytest.approx(60.0, rel=1e-2)
        # 10 docs / 300s * 3600 = 120 docs/hour
        assert snap.docs_per_hour == pytest.approx(120.0, rel=1e-2)
        # 1_000_000 bytes / 300s ≈ 3333.33 bytes/sec
        assert snap.bytes_per_second == pytest.approx(3333.33, rel=1e-2)

    def test_snapshot_latency_percentiles(self):
        mc = MetricsCollector()
        # Record a range of latencies
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            mc.record_latency(total_ms=float(ms))
        snap = mc.get_snapshot(window=MetricWindow.MINUTE_5)
        assert snap.avg_latency_ms == pytest.approx(55.0, rel=1e-2)
        assert snap.p50_latency_ms == pytest.approx(55.0, rel=1e-1)
        assert snap.p95_latency_ms > snap.p50_latency_ms
        assert snap.p99_latency_ms >= snap.p95_latency_ms

    def test_snapshot_includes_stages(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=2, completed=50)
        mc.update_stage("postprocess", active_workers=1, completed=30)
        snap = mc.get_snapshot()
        assert len(snap.stages) == 2
        stage_names = {s.stage for s in snap.stages}
        assert stage_names == {"ocr", "postprocess"}

    def test_snapshot_includes_job_counts(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=20, active=5, completed=12, failed=2, queued=1)
        snap = mc.get_snapshot()
        assert snap.total_jobs == 20
        assert snap.active_jobs == 5
        assert snap.completed_jobs == 12
        assert snap.failed_jobs == 2
        assert snap.queued_jobs == 1

    def test_snapshot_different_windows(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=100)
        snap_1m = mc.get_snapshot(window=MetricWindow.MINUTE_1)
        snap_1h = mc.get_snapshot(window=MetricWindow.HOUR_1)
        # Same pages, but divided by different elapsed => different rates
        assert snap_1m.pages_per_minute > snap_1h.pages_per_minute


# ---------------------------------------------------------------------------
# Tests: get_throughput_series
# ---------------------------------------------------------------------------


class TestGetThroughputSeries:
    def test_empty(self):
        mc = MetricsCollector()
        series = mc.get_throughput_series()
        assert series == []

    def test_bucketing(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=5)
        mc.record_throughput(pages=3)
        series = mc.get_throughput_series(window=MetricWindow.MINUTE_5, bucket_seconds=60)
        assert len(series) >= 1
        # Both points are in same bucket (recorded within the same second)
        total_pages = sum(b["pages"] for b in series)
        assert total_pages == 8

    def test_series_sorted_by_timestamp(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=1)
        mc.record_throughput(pages=2)
        series = mc.get_throughput_series(window=MetricWindow.HOUR_1)
        timestamps = [b["timestamp"] for b in series]
        assert timestamps == sorted(timestamps)

    def test_series_bucket_keys(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=1, documents=1, bytes_processed=100)
        series = mc.get_throughput_series(window=MetricWindow.MINUTE_5)
        assert len(series) >= 1
        bucket = series[0]
        assert "timestamp" in bucket
        assert "pages" in bucket
        assert "documents" in bucket
        assert "bytes" in bucket


# ---------------------------------------------------------------------------
# Tests: get_latency_series
# ---------------------------------------------------------------------------


class TestGetLatencySeries:
    def test_empty(self):
        mc = MetricsCollector()
        series = mc.get_latency_series()
        assert series == []

    def test_bucketing(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=100.0)
        mc.record_latency(total_ms=200.0)
        series = mc.get_latency_series(window=MetricWindow.MINUTE_5, bucket_seconds=60)
        assert len(series) >= 1
        # Both latencies in the same bucket
        assert series[0]["count"] == 2
        assert series[0]["avg_ms"] == pytest.approx(150.0, rel=1e-2)

    def test_series_sorted_by_timestamp(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=50.0)
        mc.record_latency(total_ms=75.0)
        series = mc.get_latency_series(window=MetricWindow.HOUR_1)
        timestamps = [b["timestamp"] for b in series]
        assert timestamps == sorted(timestamps)

    def test_series_bucket_keys(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=42.0)
        series = mc.get_latency_series(window=MetricWindow.MINUTE_5)
        assert len(series) >= 1
        bucket = series[0]
        assert "timestamp" in bucket
        assert "avg_ms" in bucket
        assert "p50_ms" in bucket
        assert "p95_ms" in bucket
        assert "count" in bucket


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_throughput(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=10)
        mc.reset()
        assert len(mc._throughput) == 0

    def test_reset_clears_latency(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=100.0)
        mc.reset()
        assert len(mc._latency) == 0

    def test_reset_clears_job_counts(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=50, active=10)
        mc.reset()
        assert mc._job_counts["total"] == 0
        assert mc._job_counts["active"] == 0

    def test_reset_clears_stages(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=4)
        mc.reset()
        assert len(mc._stage_metrics) == 0


# ---------------------------------------------------------------------------
# Tests: _prune_old
# ---------------------------------------------------------------------------


class TestPruneOld:
    def test_prune_removes_old_data(self):
        mc = MetricsCollector(max_window=10)
        # Manually insert an old point
        old_point = ThroughputPoint(timestamp=time.time() - 20, pages=1)
        mc._throughput.append(old_point)
        mc._throughput.append(ThroughputPoint(timestamp=time.time(), pages=2))
        mc._prune_old(mc._throughput)
        assert len(mc._throughput) == 1
        assert mc._throughput[0].pages == 2

    def test_prune_keeps_recent_data(self):
        mc = MetricsCollector(max_window=3600)
        mc._throughput.append(ThroughputPoint(timestamp=time.time(), pages=1))
        mc._prune_old(mc._throughput)
        assert len(mc._throughput) == 1

    def test_prune_empty_deque(self):
        mc = MetricsCollector()
        mc._prune_old(mc._throughput)  # Should not raise
        assert len(mc._throughput) == 0


# ---------------------------------------------------------------------------
# Tests: _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_two_elements(self):
        result = _percentile([10.0, 20.0], 50)
        assert 10.0 <= result <= 20.0

    def test_p50_median(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = _percentile(data, 50)
        assert result == 30.0

    def test_p0_returns_first(self):
        data = [10.0, 20.0, 30.0]
        assert _percentile(data, 0) == 10.0

    def test_p100_returns_last(self):
        data = [10.0, 20.0, 30.0]
        assert _percentile(data, 100) == 30.0

    def test_p95_high_value(self):
        data = list(range(1, 101))  # 1 to 100
        result = _percentile(data, 95)
        assert result >= 94.0


# ---------------------------------------------------------------------------
# Tests: get_collector singleton
# ---------------------------------------------------------------------------


class TestGetCollector:
    def test_returns_instance(self):
        collector = get_collector()
        assert isinstance(collector, MetricsCollector)

    def test_returns_same_instance(self):
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2


# ---------------------------------------------------------------------------
# Tests: Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_throughput_recording(self):
        mc = MetricsCollector()
        errors = []

        def record_batch(n):
            try:
                for i in range(n):
                    mc.record_throughput(pages=1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record_batch, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(mc._throughput) == 200

    def test_concurrent_latency_recording(self):
        mc = MetricsCollector()
        errors = []

        def record_batch(n):
            try:
                for i in range(n):
                    mc.record_latency(total_ms=float(i))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record_batch, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(mc._latency) == 200

    def test_concurrent_mixed_operations(self):
        mc = MetricsCollector()
        errors = []

        def writer():
            try:
                for i in range(20):
                    mc.record_throughput(pages=1)
                    mc.record_latency(total_ms=float(i))
                    mc.update_job_counts(total=i)
                    mc.update_stage("ocr", completed=i)
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(20):
                    mc.get_snapshot()
                    mc.get_throughput_series()
                    mc.get_latency_series()
            except Exception as exc:
                errors.append(exc)

        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
