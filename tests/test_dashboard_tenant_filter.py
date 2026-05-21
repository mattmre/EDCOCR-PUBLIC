"""Tests for tenant-scoped dashboard filtering (M-26).

Verifies that:
- Dashboard endpoints work without tenant_id (backward compatible)
- Dashboard endpoints filter by tenant_id when provided
- Invalid/nonexistent tenant_id returns empty or appropriate response
- Per-tenant job counts and stage metrics are isolated
- Thread safety is preserved with tenant filtering

Run with: python -m pytest tests/test_dashboard_tenant_filter.py -v
"""

import threading

import pytest

# Add project root to path
from api.dashboard import (
    LatencyPoint,
    MetricsCollector,
    MetricWindow,
    ThroughputPoint,
)

# ---------------------------------------------------------------------------
# Tests: ThroughputPoint / LatencyPoint tenant_id field
# ---------------------------------------------------------------------------


class TestDataPointTenantField:
    def test_throughput_point_default_tenant(self):
        tp = ThroughputPoint(timestamp=1000.0)
        assert tp.tenant_id == ""

    def test_throughput_point_with_tenant(self):
        tp = ThroughputPoint(timestamp=1000.0, pages=5, tenant_id="tenant_aabbccddeeff")
        assert tp.tenant_id == "tenant_aabbccddeeff"
        assert tp.pages == 5

    def test_latency_point_default_tenant(self):
        lp = LatencyPoint(timestamp=1000.0)
        assert lp.tenant_id == ""

    def test_latency_point_with_tenant(self):
        lp = LatencyPoint(timestamp=1000.0, total_ms=50.0, tenant_id="tenant_112233445566")
        assert lp.tenant_id == "tenant_112233445566"
        assert lp.total_ms == 50.0


# ---------------------------------------------------------------------------
# Tests: record_throughput / record_latency with tenant_id
# ---------------------------------------------------------------------------


class TestRecordWithTenant:
    def test_record_throughput_with_tenant(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=10, tenant_id="tenant_aabbccddeeff")
        assert len(mc._throughput) == 1
        assert mc._throughput[0].tenant_id == "tenant_aabbccddeeff"
        assert mc._throughput[0].pages == 10

    def test_record_throughput_without_tenant(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=5)
        assert len(mc._throughput) == 1
        assert mc._throughput[0].tenant_id == ""

    def test_record_latency_with_tenant(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=100.0, tenant_id="tenant_aabbccddeeff")
        assert len(mc._latency) == 1
        assert mc._latency[0].tenant_id == "tenant_aabbccddeeff"
        assert mc._latency[0].total_ms == 100.0

    def test_record_latency_without_tenant(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=50.0)
        assert len(mc._latency) == 1
        assert mc._latency[0].tenant_id == ""


# ---------------------------------------------------------------------------
# Tests: update_job_counts with tenant_id
# ---------------------------------------------------------------------------


class TestUpdateJobCountsWithTenant:
    def test_global_counts_unchanged(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=10, active=3)
        assert mc._job_counts["total"] == 10
        assert mc._job_counts["active"] == 3

    def test_tenant_counts_stored_separately(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=10, active=3)
        mc.update_job_counts(total=5, active=2, tenant_id="tenant_aabbccddeeff")
        # Global unchanged
        assert mc._job_counts["total"] == 10
        # Tenant-specific stored
        assert mc._tenant_job_counts["tenant_aabbccddeeff"]["total"] == 5
        assert mc._tenant_job_counts["tenant_aabbccddeeff"]["active"] == 2

    def test_multiple_tenants(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=5, tenant_id="tenant_aaaaaaaaaaaa")
        mc.update_job_counts(total=3, tenant_id="tenant_bbbbbbbbbbbb")
        assert mc._tenant_job_counts["tenant_aaaaaaaaaaaa"]["total"] == 5
        assert mc._tenant_job_counts["tenant_bbbbbbbbbbbb"]["total"] == 3

    def test_tenant_counts_overwrite(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=5, tenant_id="tenant_aaaaaaaaaaaa")
        mc.update_job_counts(total=10, active=2, tenant_id="tenant_aaaaaaaaaaaa")
        assert mc._tenant_job_counts["tenant_aaaaaaaaaaaa"]["total"] == 10
        assert mc._tenant_job_counts["tenant_aaaaaaaaaaaa"]["active"] == 2


# ---------------------------------------------------------------------------
# Tests: update_stage with tenant_id
# ---------------------------------------------------------------------------


class TestUpdateStageWithTenant:
    def test_global_stage_unchanged(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=4)
        assert "ocr" in mc._stage_metrics
        assert mc._stage_metrics["ocr"].active_workers == 4

    def test_tenant_stage_stored_separately(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=4)
        mc.update_stage("ocr", active_workers=2, tenant_id="tenant_aabbccddeeff")
        # Global unchanged
        assert mc._stage_metrics["ocr"].active_workers == 4
        # Tenant-specific stored
        assert mc._tenant_stage_metrics["tenant_aabbccddeeff"]["ocr"].active_workers == 2

    def test_multiple_tenant_stages(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", completed=10, tenant_id="tenant_aaaaaaaaaaaa")
        mc.update_stage("ocr", completed=20, tenant_id="tenant_bbbbbbbbbbbb")
        assert mc._tenant_stage_metrics["tenant_aaaaaaaaaaaa"]["ocr"].completed == 10
        assert mc._tenant_stage_metrics["tenant_bbbbbbbbbbbb"]["ocr"].completed == 20


# ---------------------------------------------------------------------------
# Tests: get_snapshot with tenant_id filtering
# ---------------------------------------------------------------------------


class TestGetSnapshotTenantFilter:
    def test_no_filter_returns_all_data(self):
        """Without tenant_id, snapshot aggregates all data (backward compat)."""
        mc = MetricsCollector()
        mc.record_throughput(pages=100, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_throughput(pages=200, tenant_id="tenant_bbbbbbbbbbbb")
        mc.record_throughput(pages=50)  # no tenant

        snap = mc.get_snapshot(window=MetricWindow.MINUTE_5)
        # All 350 pages should be counted
        expected_ppm = (350 / 300) * 60
        assert snap.pages_per_minute == pytest.approx(expected_ppm, rel=1e-2)

    def test_filter_by_tenant(self):
        """With tenant_id, snapshot only includes that tenant's data."""
        mc = MetricsCollector()
        mc.record_throughput(pages=100, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_throughput(pages=200, tenant_id="tenant_bbbbbbbbbbbb")

        snap = mc.get_snapshot(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        expected_ppm = (100 / 300) * 60
        assert snap.pages_per_minute == pytest.approx(expected_ppm, rel=1e-2)

    def test_filter_nonexistent_tenant_returns_zeros(self):
        """Filtering by a tenant with no data returns zero metrics."""
        mc = MetricsCollector()
        mc.record_throughput(pages=100, tenant_id="tenant_aaaaaaaaaaaa")

        snap = mc.get_snapshot(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_cccccccccccc",
        )
        assert snap.pages_per_minute == 0.0
        assert snap.docs_per_hour == 0.0
        assert snap.avg_latency_ms == 0.0
        assert snap.total_jobs == 0

    def test_filter_latency_by_tenant(self):
        """Latency metrics are filtered by tenant."""
        mc = MetricsCollector()
        mc.record_latency(total_ms=100.0, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_latency(total_ms=200.0, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_latency(total_ms=500.0, tenant_id="tenant_bbbbbbbbbbbb")

        snap_a = mc.get_snapshot(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        assert snap_a.avg_latency_ms == pytest.approx(150.0, rel=1e-2)

        snap_b = mc.get_snapshot(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_bbbbbbbbbbbb",
        )
        assert snap_b.avg_latency_ms == pytest.approx(500.0, rel=1e-2)

    def test_filter_job_counts_by_tenant(self):
        """Job counts are tenant-scoped when tenant_id is provided."""
        mc = MetricsCollector()
        mc.update_job_counts(total=100, active=10)
        mc.update_job_counts(total=5, active=2, tenant_id="tenant_aaaaaaaaaaaa")

        # Global
        snap_all = mc.get_snapshot(window=MetricWindow.MINUTE_5)
        assert snap_all.total_jobs == 100
        assert snap_all.active_jobs == 10

        # Tenant
        snap_t = mc.get_snapshot(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        assert snap_t.total_jobs == 5
        assert snap_t.active_jobs == 2

    def test_filter_stages_by_tenant(self):
        """Stage metrics are tenant-scoped when tenant_id is provided."""
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=8)
        mc.update_stage("ocr", active_workers=2, tenant_id="tenant_aaaaaaaaaaaa")

        snap_all = mc.get_snapshot(window=MetricWindow.MINUTE_5)
        assert len(snap_all.stages) == 1
        assert snap_all.stages[0].active_workers == 8

        snap_t = mc.get_snapshot(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        assert len(snap_t.stages) == 1
        assert snap_t.stages[0].active_workers == 2

    def test_filter_empty_tenant_stages(self):
        """Tenant with no stages returns empty stages list."""
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=4)

        snap = mc.get_snapshot(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_cccccccccccc",
        )
        assert snap.stages == []

    def test_to_dict_includes_tenant_id_field_not_added_by_collector(self):
        """The collector snapshot does NOT add tenant_id — that's the router's job."""
        mc = MetricsCollector()
        mc.record_throughput(pages=10, tenant_id="tenant_aaaaaaaaaaaa")
        snap = mc.get_snapshot(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        d = snap.to_dict()
        # to_dict itself should not have tenant_id; the router adds it
        assert "tenant_id" not in d


# ---------------------------------------------------------------------------
# Tests: get_throughput_series with tenant_id filtering
# ---------------------------------------------------------------------------


class TestGetThroughputSeriesTenantFilter:
    def test_no_filter_includes_all(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=5, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_throughput(pages=3, tenant_id="tenant_bbbbbbbbbbbb")
        series = mc.get_throughput_series(window=MetricWindow.MINUTE_5)
        total_pages = sum(b["pages"] for b in series)
        assert total_pages == 8

    def test_filter_by_tenant(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=5, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_throughput(pages=3, tenant_id="tenant_bbbbbbbbbbbb")
        series = mc.get_throughput_series(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        total_pages = sum(b["pages"] for b in series)
        assert total_pages == 5

    def test_nonexistent_tenant_returns_empty(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=5, tenant_id="tenant_aaaaaaaaaaaa")
        series = mc.get_throughput_series(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_cccccccccccc",
        )
        assert series == []


# ---------------------------------------------------------------------------
# Tests: get_latency_series with tenant_id filtering
# ---------------------------------------------------------------------------


class TestGetLatencySeriesTenantFilter:
    def test_no_filter_includes_all(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=100.0, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_latency(total_ms=200.0, tenant_id="tenant_bbbbbbbbbbbb")
        series = mc.get_latency_series(window=MetricWindow.MINUTE_5)
        total_count = sum(b["count"] for b in series)
        assert total_count == 2

    def test_filter_by_tenant(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=100.0, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_latency(total_ms=200.0, tenant_id="tenant_bbbbbbbbbbbb")
        series = mc.get_latency_series(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        total_count = sum(b["count"] for b in series)
        assert total_count == 1
        assert series[0]["avg_ms"] == pytest.approx(100.0, rel=1e-2)

    def test_nonexistent_tenant_returns_empty(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=100.0, tenant_id="tenant_aaaaaaaaaaaa")
        series = mc.get_latency_series(
            window=MetricWindow.MINUTE_5,
            tenant_id="tenant_cccccccccccc",
        )
        assert series == []


# ---------------------------------------------------------------------------
# Tests: reset clears tenant-specific data
# ---------------------------------------------------------------------------


class TestResetWithTenantData:
    def test_reset_clears_tenant_job_counts(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=5, tenant_id="tenant_aaaaaaaaaaaa")
        mc.reset()
        assert len(mc._tenant_job_counts) == 0

    def test_reset_clears_tenant_stage_metrics(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=2, tenant_id="tenant_aaaaaaaaaaaa")
        mc.reset()
        assert len(mc._tenant_stage_metrics) == 0

    def test_reset_clears_tenant_data_points(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=10, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_latency(total_ms=100.0, tenant_id="tenant_aaaaaaaaaaaa")
        mc.reset()
        assert len(mc._throughput) == 0
        assert len(mc._latency) == 0


# ---------------------------------------------------------------------------
# Tests: backward compatibility (existing tests still pass)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure all existing calling patterns still work unchanged."""

    def test_record_throughput_no_tenant(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=5, documents=1, bytes_processed=2048)
        assert len(mc._throughput) == 1
        assert mc._throughput[0].pages == 5

    def test_record_latency_no_tenant(self):
        mc = MetricsCollector()
        mc.record_latency(job_id="j1", total_ms=100.0)
        assert len(mc._latency) == 1
        assert mc._latency[0].total_ms == 100.0

    def test_update_job_counts_no_tenant(self):
        mc = MetricsCollector()
        mc.update_job_counts(total=10, active=3, completed=5, failed=1, queued=1)
        assert mc._job_counts["total"] == 10

    def test_update_stage_no_tenant(self):
        mc = MetricsCollector()
        mc.update_stage("ocr", active_workers=4, completed=100)
        assert mc._stage_metrics["ocr"].active_workers == 4

    def test_get_snapshot_no_tenant(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=300, documents=10, bytes_processed=1_000_000)
        snap = mc.get_snapshot(window=MetricWindow.MINUTE_5)
        assert snap.pages_per_minute == pytest.approx(60.0, rel=1e-2)

    def test_get_throughput_series_no_tenant(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=5)
        mc.record_throughput(pages=3)
        series = mc.get_throughput_series(window=MetricWindow.MINUTE_5, bucket_seconds=60)
        total_pages = sum(b["pages"] for b in series)
        assert total_pages == 8

    def test_get_latency_series_no_tenant(self):
        mc = MetricsCollector()
        mc.record_latency(total_ms=100.0)
        mc.record_latency(total_ms=200.0)
        series = mc.get_latency_series(window=MetricWindow.MINUTE_5, bucket_seconds=60)
        assert series[0]["count"] == 2
        assert series[0]["avg_ms"] == pytest.approx(150.0, rel=1e-2)


# ---------------------------------------------------------------------------
# Tests: thread safety with tenant filtering
# ---------------------------------------------------------------------------


class TestThreadSafetyWithTenants:
    def test_concurrent_tenant_recording(self):
        mc = MetricsCollector()
        errors = []

        def record_for_tenant(tid, n):
            try:
                for i in range(n):
                    mc.record_throughput(pages=1, tenant_id=tid)
                    mc.record_latency(total_ms=float(i), tenant_id=tid)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(
                target=record_for_tenant,
                args=(f"tenant_{'a' * 12}", 50),
            ),
            threading.Thread(
                target=record_for_tenant,
                args=(f"tenant_{'b' * 12}", 50),
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(mc._throughput) == 100
        assert len(mc._latency) == 100

    def test_concurrent_snapshot_with_tenant_filter(self):
        mc = MetricsCollector()
        mc.record_throughput(pages=10, tenant_id="tenant_aaaaaaaaaaaa")
        mc.record_throughput(pages=20, tenant_id="tenant_bbbbbbbbbbbb")
        errors = []

        def read_snapshot(tid):
            try:
                for _ in range(20):
                    mc.get_snapshot(tenant_id=tid)
                    mc.get_throughput_series(tenant_id=tid)
                    mc.get_latency_series(tenant_id=tid)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=read_snapshot, args=("tenant_aaaaaaaaaaaa",)),
            threading.Thread(target=read_snapshot, args=("tenant_bbbbbbbbbbbb",)),
            threading.Thread(target=read_snapshot, args=("",)),  # global
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# Tests: router-level _validate_tenant_id helper
# ---------------------------------------------------------------------------


class TestValidateTenantId:
    """Test the router's _validate_tenant_id helper directly."""

    def test_none_returns_empty(self):
        from api.routers.dashboard import _validate_tenant_id

        assert _validate_tenant_id(None) == ""

    def test_valid_id_returned(self):
        from api.routers.dashboard import _validate_tenant_id

        assert _validate_tenant_id("tenant_aabbccddeeff") == "tenant_aabbccddeeff"

    def test_invalid_format_raises_400(self):
        from fastapi import HTTPException

        from api.routers.dashboard import _validate_tenant_id

        with pytest.raises(HTTPException) as exc_info:
            _validate_tenant_id("bad_id")
        assert exc_info.value.status_code == 400

    def test_invalid_too_short_raises_400(self):
        from fastapi import HTTPException

        from api.routers.dashboard import _validate_tenant_id

        with pytest.raises(HTTPException) as exc_info:
            _validate_tenant_id("tenant_abc")
        assert exc_info.value.status_code == 400

    def test_invalid_uppercase_raises_400(self):
        from fastapi import HTTPException

        from api.routers.dashboard import _validate_tenant_id

        with pytest.raises(HTTPException) as exc_info:
            _validate_tenant_id("tenant_AABBCCDDEEFF")
        assert exc_info.value.status_code == 400
