"""Tests for per-tenant cost tracking module."""

import json
import threading

import pytest

from cost_tracking import (
    COST_PER_API_CALL,
    COST_PER_GB_STORED,
    COST_PER_GPU_SECOND,
    COST_PER_PAGE,
    ENABLE_COST_TRACKING,
    CostTracker,
    TenantUsage,
    get_tracker,
    reset_global_tracker,
)

# ---------------------------------------------------------------------------
# TestConstants — verify default env-based constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Validate module-level constants have expected defaults."""

    def test_cost_tracking_disabled_by_default(self):
        assert ENABLE_COST_TRACKING is False

    def test_default_cost_per_page(self):
        assert COST_PER_PAGE == 0.01

    def test_default_cost_per_gpu_second(self):
        assert COST_PER_GPU_SECOND == 0.001

    def test_default_cost_per_gb_stored(self):
        assert COST_PER_GB_STORED == 0.05

    def test_default_cost_per_api_call(self):
        assert COST_PER_API_CALL == 0.0001


# ---------------------------------------------------------------------------
# TestTenantUsageDefaults
# ---------------------------------------------------------------------------


class TestTenantUsageDefaults:
    """Test TenantUsage dataclass defaults."""

    def test_default_values(self):
        usage = TenantUsage(tenant_id="t1")
        assert usage.tenant_id == "t1"
        assert usage.pages_processed == 0
        assert usage.gpu_seconds == 0.0
        assert usage.storage_bytes == 0
        assert usage.api_calls == 0
        assert usage.jobs_submitted == 0
        assert usage.jobs_completed == 0
        assert usage.jobs_failed == 0
        assert usage.first_activity == ""
        assert usage.last_activity == ""

    def test_storage_gb_zero(self):
        usage = TenantUsage(tenant_id="t1")
        assert usage.storage_gb == 0.0

    def test_storage_gb_conversion(self):
        usage = TenantUsage(tenant_id="t1", storage_bytes=1024**3)
        assert usage.storage_gb == 1.0

    def test_storage_gb_fractional(self):
        usage = TenantUsage(tenant_id="t1", storage_bytes=512 * 1024 * 1024)
        assert usage.storage_gb == pytest.approx(0.5, rel=1e-6)

    def test_custom_initial_values(self):
        usage = TenantUsage(
            tenant_id="t2",
            pages_processed=100,
            gpu_seconds=50.5,
            storage_bytes=2048,
            api_calls=10,
        )
        assert usage.pages_processed == 100
        assert usage.gpu_seconds == 50.5
        assert usage.storage_bytes == 2048
        assert usage.api_calls == 10


# ---------------------------------------------------------------------------
# TestEstimatedCost
# ---------------------------------------------------------------------------


class TestEstimatedCost:
    """Test estimated_cost calculation."""

    def test_zero_usage(self):
        usage = TenantUsage(tenant_id="t1")
        cost = usage.estimated_cost()
        assert cost["page_cost"] == 0.0
        assert cost["gpu_cost"] == 0.0
        assert cost["storage_cost"] == 0.0
        assert cost["api_cost"] == 0.0
        assert cost["total_cost"] == 0.0
        assert cost["currency"] == "USD"

    def test_page_cost_calculation(self):
        usage = TenantUsage(tenant_id="t1", pages_processed=100)
        cost = usage.estimated_cost()
        assert cost["page_cost"] == round(100 * COST_PER_PAGE, 4)

    def test_gpu_cost_calculation(self):
        usage = TenantUsage(tenant_id="t1", gpu_seconds=3600.0)
        cost = usage.estimated_cost()
        assert cost["gpu_cost"] == round(3600.0 * COST_PER_GPU_SECOND, 4)

    def test_storage_cost_calculation(self):
        usage = TenantUsage(tenant_id="t1", storage_bytes=1024**3)
        cost = usage.estimated_cost()
        assert cost["storage_cost"] == round(1.0 * COST_PER_GB_STORED, 4)

    def test_api_cost_calculation(self):
        usage = TenantUsage(tenant_id="t1", api_calls=1000)
        cost = usage.estimated_cost()
        assert cost["api_cost"] == round(1000 * COST_PER_API_CALL, 4)

    def test_total_is_sum_of_parts(self):
        usage = TenantUsage(
            tenant_id="t1",
            pages_processed=50,
            gpu_seconds=120.0,
            storage_bytes=2 * 1024**3,
            api_calls=200,
        )
        cost = usage.estimated_cost()
        expected_total = (
            cost["page_cost"]
            + cost["gpu_cost"]
            + cost["storage_cost"]
            + cost["api_cost"]
        )
        assert cost["total_cost"] == round(expected_total, 4)

    def test_currency_is_usd(self):
        usage = TenantUsage(tenant_id="t1")
        assert usage.estimated_cost()["currency"] == "USD"


# ---------------------------------------------------------------------------
# TestCostTrackerInit
# ---------------------------------------------------------------------------


class TestCostTrackerInit:
    """Test CostTracker initialization."""

    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.get_all_usage() == {}

    def test_no_persist_path(self):
        tracker = CostTracker()
        assert tracker._persist_path is None

    def test_persist_path_set(self, tmp_path):
        p = tmp_path / "cost.json"
        tracker = CostTracker(persist_path=str(p))
        assert tracker._persist_path == p

    def test_loads_existing_data(self, tmp_path):
        p = tmp_path / "cost.json"
        data = {
            "tenant_a": {
                "tenant_id": "tenant_a",
                "pages_processed": 42,
                "gpu_seconds": 10.0,
                "storage_bytes": 0,
                "api_calls": 5,
                "jobs_submitted": 3,
                "jobs_completed": 2,
                "jobs_failed": 1,
                "first_activity": "2026-01-01T00:00:00+00:00",
                "last_activity": "2026-01-02T00:00:00+00:00",
            }
        }
        p.write_text(json.dumps(data), encoding="utf-8")
        tracker = CostTracker(persist_path=str(p))
        usage = tracker.get_usage("tenant_a")
        assert usage is not None
        assert usage.pages_processed == 42
        assert usage.api_calls == 5


# ---------------------------------------------------------------------------
# TestRecordPages
# ---------------------------------------------------------------------------


class TestRecordPages:
    """Test record_pages method."""

    def test_single_record(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 10)
        usage = tracker.get_usage("t1")
        assert usage.pages_processed == 10

    def test_multiple_records_accumulate(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 10)
        tracker.record_pages("t1", 5)
        assert tracker.get_usage("t1").pages_processed == 15

    def test_sets_first_activity(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 1)
        usage = tracker.get_usage("t1")
        assert usage.first_activity != ""
        assert usage.last_activity != ""

    def test_first_activity_not_overwritten(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 1)
        first = tracker.get_usage("t1").first_activity
        tracker.record_pages("t1", 1)
        assert tracker.get_usage("t1").first_activity == first

    def test_last_activity_updated(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 1)
        first_last = tracker.get_usage("t1").last_activity
        # Record again -- last_activity should update (or stay same if too fast)
        tracker.record_pages("t1", 1)
        assert tracker.get_usage("t1").last_activity >= first_last


# ---------------------------------------------------------------------------
# TestRecordGpuTime
# ---------------------------------------------------------------------------


class TestRecordGpuTime:
    """Test record_gpu_time method."""

    def test_single_record(self):
        tracker = CostTracker()
        tracker.record_gpu_time("t1", 5.5)
        assert tracker.get_usage("t1").gpu_seconds == 5.5

    def test_accumulation(self):
        tracker = CostTracker()
        tracker.record_gpu_time("t1", 2.0)
        tracker.record_gpu_time("t1", 3.5)
        assert tracker.get_usage("t1").gpu_seconds == pytest.approx(5.5)

    def test_sets_timestamps(self):
        tracker = CostTracker()
        tracker.record_gpu_time("t1", 1.0)
        usage = tracker.get_usage("t1")
        assert usage.first_activity != ""


# ---------------------------------------------------------------------------
# TestRecordStorage
# ---------------------------------------------------------------------------


class TestRecordStorage:
    """Test record_storage method."""

    def test_single_record(self):
        tracker = CostTracker()
        tracker.record_storage("t1", 1024)
        assert tracker.get_usage("t1").storage_bytes == 1024

    def test_accumulation(self):
        tracker = CostTracker()
        tracker.record_storage("t1", 1024)
        tracker.record_storage("t1", 2048)
        assert tracker.get_usage("t1").storage_bytes == 3072

    def test_sets_timestamps(self):
        tracker = CostTracker()
        tracker.record_storage("t1", 100)
        assert tracker.get_usage("t1").first_activity != ""


# ---------------------------------------------------------------------------
# TestRecordApiCall
# ---------------------------------------------------------------------------


class TestRecordApiCall:
    """Test record_api_call method."""

    def test_single_call(self):
        tracker = CostTracker()
        tracker.record_api_call("t1")
        assert tracker.get_usage("t1").api_calls == 1

    def test_multiple_calls(self):
        tracker = CostTracker()
        for _ in range(5):
            tracker.record_api_call("t1")
        assert tracker.get_usage("t1").api_calls == 5

    def test_sets_timestamps(self):
        tracker = CostTracker()
        tracker.record_api_call("t1")
        assert tracker.get_usage("t1").last_activity != ""


# ---------------------------------------------------------------------------
# TestRecordJobs
# ---------------------------------------------------------------------------


class TestRecordJobs:
    """Test job recording methods."""

    def test_job_submitted(self):
        tracker = CostTracker()
        tracker.record_job_submitted("t1")
        assert tracker.get_usage("t1").jobs_submitted == 1

    def test_job_completed(self):
        tracker = CostTracker()
        tracker.record_job_completed("t1")
        assert tracker.get_usage("t1").jobs_completed == 1

    def test_job_failed(self):
        tracker = CostTracker()
        tracker.record_job_failed("t1")
        assert tracker.get_usage("t1").jobs_failed == 1

    def test_mixed_job_lifecycle(self):
        tracker = CostTracker()
        tracker.record_job_submitted("t1")
        tracker.record_job_submitted("t1")
        tracker.record_job_submitted("t1")
        tracker.record_job_completed("t1")
        tracker.record_job_completed("t1")
        tracker.record_job_failed("t1")
        usage = tracker.get_usage("t1")
        assert usage.jobs_submitted == 3
        assert usage.jobs_completed == 2
        assert usage.jobs_failed == 1

    def test_submitted_sets_timestamps(self):
        tracker = CostTracker()
        tracker.record_job_submitted("t1")
        assert tracker.get_usage("t1").first_activity != ""

    def test_completed_updates_last_activity(self):
        tracker = CostTracker()
        tracker.record_job_submitted("t1")
        first_last = tracker.get_usage("t1").last_activity
        tracker.record_job_completed("t1")
        assert tracker.get_usage("t1").last_activity >= first_last

    def test_failed_updates_last_activity(self):
        tracker = CostTracker()
        tracker.record_job_submitted("t1")
        tracker.record_job_failed("t1")
        assert tracker.get_usage("t1").last_activity != ""


# ---------------------------------------------------------------------------
# TestGetUsage
# ---------------------------------------------------------------------------


class TestGetUsage:
    """Test get_usage method."""

    def test_existing_tenant(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 5)
        usage = tracker.get_usage("t1")
        assert usage is not None
        assert usage.tenant_id == "t1"

    def test_missing_tenant(self):
        tracker = CostTracker()
        assert tracker.get_usage("nonexistent") is None

    def test_returns_correct_tenant(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 10)
        tracker.record_pages("t2", 20)
        assert tracker.get_usage("t1").pages_processed == 10
        assert tracker.get_usage("t2").pages_processed == 20


# ---------------------------------------------------------------------------
# TestGetAllUsage
# ---------------------------------------------------------------------------


class TestGetAllUsage:
    """Test get_all_usage method."""

    def test_empty(self):
        tracker = CostTracker()
        assert tracker.get_all_usage() == {}

    def test_multiple_tenants(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 5)
        tracker.record_pages("t2", 10)
        all_usage = tracker.get_all_usage()
        assert len(all_usage) == 2
        assert "t1" in all_usage
        assert "t2" in all_usage

    def test_returns_copy(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 1)
        copy = tracker.get_all_usage()
        copy["t_new"] = TenantUsage(tenant_id="t_new")
        # Original should not be affected
        assert tracker.get_usage("t_new") is None


# ---------------------------------------------------------------------------
# TestGetCostReport
# ---------------------------------------------------------------------------


class TestGetCostReport:
    """Test get_cost_report method."""

    def test_missing_tenant_returns_none(self):
        tracker = CostTracker()
        assert tracker.get_cost_report("nonexistent") is None

    def test_report_structure(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 10)
        report = tracker.get_cost_report("t1")
        assert report is not None
        assert report["tenant_id"] == "t1"
        assert "usage" in report
        assert "cost" in report

    def test_report_usage_fields(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 25)
        tracker.record_gpu_time("t1", 10.0)
        report = tracker.get_cost_report("t1")
        assert report["usage"]["pages_processed"] == 25
        assert report["usage"]["gpu_seconds"] == 10.0

    def test_report_cost_fields(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 100)
        report = tracker.get_cost_report("t1")
        assert "total_cost" in report["cost"]
        assert report["cost"]["page_cost"] == round(100 * COST_PER_PAGE, 4)


# ---------------------------------------------------------------------------
# TestGetAllCostReports
# ---------------------------------------------------------------------------


class TestGetAllCostReports:
    """Test get_all_cost_reports method."""

    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.get_all_cost_reports() == []

    def test_multiple_tenants(self):
        tracker = CostTracker()
        tracker.record_pages("t2", 20)
        tracker.record_pages("t1", 10)
        reports = tracker.get_all_cost_reports()
        assert len(reports) == 2

    def test_sorted_by_tenant_id(self):
        tracker = CostTracker()
        tracker.record_pages("charlie", 1)
        tracker.record_pages("alpha", 1)
        tracker.record_pages("bravo", 1)
        reports = tracker.get_all_cost_reports()
        assert [r["tenant_id"] for r in reports] == ["alpha", "bravo", "charlie"]

    def test_each_report_has_cost(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 5)
        reports = tracker.get_all_cost_reports()
        assert len(reports) == 1
        assert "cost" in reports[0]
        assert "usage" in reports[0]


# ---------------------------------------------------------------------------
# TestResetTenant
# ---------------------------------------------------------------------------


class TestResetTenant:
    """Test reset_tenant method."""

    def test_removes_existing_tenant(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 10)
        tracker.reset_tenant("t1")
        assert tracker.get_usage("t1") is None

    def test_reset_nonexistent_is_noop(self):
        tracker = CostTracker()
        tracker.reset_tenant("nonexistent")  # should not raise
        assert tracker.get_usage("nonexistent") is None

    def test_does_not_affect_other_tenants(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 5)
        tracker.record_pages("t2", 10)
        tracker.reset_tenant("t1")
        assert tracker.get_usage("t1") is None
        assert tracker.get_usage("t2").pages_processed == 10


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """Test save/load persistence."""

    def test_persist_and_load(self, tmp_path):
        p = tmp_path / "cost.json"
        tracker1 = CostTracker(persist_path=str(p))
        tracker1.record_pages("t1", 42)
        tracker1.record_gpu_time("t1", 12.5)
        tracker1.record_storage("t1", 1024)
        tracker1.record_api_call("t1")
        tracker1.record_job_submitted("t1")
        tracker1.record_job_completed("t1")
        tracker1.persist()

        tracker2 = CostTracker(persist_path=str(p))
        usage = tracker2.get_usage("t1")
        assert usage is not None
        assert usage.pages_processed == 42
        assert usage.gpu_seconds == 12.5
        assert usage.storage_bytes == 1024
        assert usage.api_calls == 1
        assert usage.jobs_submitted == 1
        assert usage.jobs_completed == 1

    def test_persist_multiple_tenants(self, tmp_path):
        p = tmp_path / "cost.json"
        tracker1 = CostTracker(persist_path=str(p))
        tracker1.record_pages("t1", 10)
        tracker1.record_pages("t2", 20)
        tracker1.persist()

        tracker2 = CostTracker(persist_path=str(p))
        assert tracker2.get_usage("t1").pages_processed == 10
        assert tracker2.get_usage("t2").pages_processed == 20

    def test_corrupt_file_handled(self, tmp_path):
        p = tmp_path / "cost.json"
        p.write_text("NOT VALID JSON{{{{", encoding="utf-8")
        tracker = CostTracker(persist_path=str(p))
        assert tracker.get_all_usage() == {}

    def test_missing_file_ok(self, tmp_path):
        p = tmp_path / "nonexistent.json"
        tracker = CostTracker(persist_path=str(p))
        assert tracker.get_all_usage() == {}

    def test_persist_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "cost.json"
        tracker = CostTracker(persist_path=str(p))
        tracker.record_pages("t1", 1)
        tracker.persist()
        assert p.exists()

    def test_persist_without_path_is_noop(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 1)
        tracker.persist()  # should not raise

    def test_atomic_write_does_not_leave_tmp(self, tmp_path):
        p = tmp_path / "cost.json"
        tracker = CostTracker(persist_path=str(p))
        tracker.record_pages("t1", 1)
        tracker.persist()
        tmp_file = p.with_suffix(".tmp")
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test concurrent access to CostTracker."""

    def test_concurrent_record_pages(self):
        tracker = CostTracker()
        num_threads = 10
        records_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(records_per_thread):
                tracker.record_pages("t1", 1)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.get_usage("t1").pages_processed == num_threads * records_per_thread

    def test_concurrent_multiple_tenants(self):
        tracker = CostTracker()
        num_tenants = 5
        records_per_tenant = 50
        barrier = threading.Barrier(num_tenants)

        def worker(tid):
            barrier.wait()
            for _ in range(records_per_tenant):
                tracker.record_pages(tid, 1)
                tracker.record_api_call(tid)

        threads = [
            threading.Thread(target=worker, args=(f"t{i}",))
            for i in range(num_tenants)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(num_tenants):
            usage = tracker.get_usage(f"t{i}")
            assert usage.pages_processed == records_per_tenant
            assert usage.api_calls == records_per_tenant

    def test_concurrent_mixed_operations(self):
        tracker = CostTracker()
        iterations = 100
        barrier = threading.Barrier(4)

        def record_pages():
            barrier.wait()
            for _ in range(iterations):
                tracker.record_pages("t1", 1)

        def record_gpu():
            barrier.wait()
            for _ in range(iterations):
                tracker.record_gpu_time("t1", 0.1)

        def record_api():
            barrier.wait()
            for _ in range(iterations):
                tracker.record_api_call("t1")

        def read_usage():
            barrier.wait()
            for _ in range(iterations):
                tracker.get_usage("t1")
                tracker.get_all_usage()

        threads = [
            threading.Thread(target=record_pages),
            threading.Thread(target=record_gpu),
            threading.Thread(target=record_api),
            threading.Thread(target=read_usage),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        usage = tracker.get_usage("t1")
        assert usage.pages_processed == iterations
        assert usage.gpu_seconds == pytest.approx(iterations * 0.1)
        assert usage.api_calls == iterations


# ---------------------------------------------------------------------------
# TestGlobalTracker
# ---------------------------------------------------------------------------


class TestGlobalTracker:
    """Test module-level global tracker functions."""

    def setup_method(self):
        reset_global_tracker()

    def teardown_method(self):
        reset_global_tracker()

    def test_get_tracker_returns_instance(self):
        tracker = get_tracker()
        assert isinstance(tracker, CostTracker)

    def test_get_tracker_is_singleton(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_reset_clears_singleton(self):
        t1 = get_tracker()
        reset_global_tracker()
        t2 = get_tracker()
        assert t1 is not t2

    def test_get_tracker_with_persist_path(self, tmp_path):
        p = tmp_path / "global.json"
        tracker = get_tracker(persist_path=str(p))
        assert tracker._persist_path == p

    def test_persist_path_ignored_on_subsequent_calls(self, tmp_path):
        p1 = tmp_path / "first.json"
        p2 = tmp_path / "second.json"
        t1 = get_tracker(persist_path=str(p1))
        t2 = get_tracker(persist_path=str(p2))
        # Second call should return same tracker, ignoring p2
        assert t1 is t2
        assert t2._persist_path == p1


# ---------------------------------------------------------------------------
# TestMultipleTenants
# ---------------------------------------------------------------------------


class TestMultipleTenants:
    """Test isolation between tenants."""

    def test_separate_page_counts(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 10)
        tracker.record_pages("t2", 20)
        assert tracker.get_usage("t1").pages_processed == 10
        assert tracker.get_usage("t2").pages_processed == 20

    def test_separate_gpu_time(self):
        tracker = CostTracker()
        tracker.record_gpu_time("t1", 5.0)
        tracker.record_gpu_time("t2", 15.0)
        assert tracker.get_usage("t1").gpu_seconds == 5.0
        assert tracker.get_usage("t2").gpu_seconds == 15.0

    def test_separate_costs(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 100)
        tracker.record_pages("t2", 200)
        r1 = tracker.get_cost_report("t1")
        r2 = tracker.get_cost_report("t2")
        assert r1["cost"]["page_cost"] < r2["cost"]["page_cost"]

    def test_reset_one_does_not_affect_other(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 10)
        tracker.record_pages("t2", 20)
        tracker.reset_tenant("t1")
        assert tracker.get_usage("t1") is None
        assert tracker.get_usage("t2").pages_processed == 20


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_pages(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 0)
        assert tracker.get_usage("t1").pages_processed == 0

    def test_zero_gpu_seconds(self):
        tracker = CostTracker()
        tracker.record_gpu_time("t1", 0.0)
        assert tracker.get_usage("t1").gpu_seconds == 0.0

    def test_large_page_count(self):
        tracker = CostTracker()
        tracker.record_pages("t1", 1_000_000)
        assert tracker.get_usage("t1").pages_processed == 1_000_000

    def test_large_storage_bytes(self):
        tracker = CostTracker()
        one_tb = 1024**4
        tracker.record_storage("t1", one_tb)
        assert tracker.get_usage("t1").storage_bytes == one_tb
        assert tracker.get_usage("t1").storage_gb == pytest.approx(1024.0)

    def test_tenant_id_with_special_chars(self):
        tracker = CostTracker()
        tid = "org-123/tenant@example.com"
        tracker.record_pages(tid, 5)
        assert tracker.get_usage(tid).pages_processed == 5

    def test_empty_string_tenant_id(self):
        tracker = CostTracker()
        tracker.record_pages("", 1)
        assert tracker.get_usage("").pages_processed == 1

    def test_estimated_cost_rounding(self):
        usage = TenantUsage(
            tenant_id="t1",
            pages_processed=3,
            gpu_seconds=0.001,
            storage_bytes=1,
            api_calls=1,
        )
        cost = usage.estimated_cost()
        # All values should be rounded to 4 decimal places
        for key in ("page_cost", "gpu_cost", "storage_cost", "api_cost", "total_cost"):
            value_str = str(cost[key])
            if "." in value_str:
                decimals = len(value_str.split(".")[1])
                assert decimals <= 4
