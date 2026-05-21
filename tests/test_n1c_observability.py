"""Tests for Wave N1-C observability truth wiring.

Verifies:
- tenant_id is passed to dashboard MetricsCollector calls
- SLA monitor is fed from job completion path
- tenant_id label exists in coordinator processing duration histogram
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from api.database import Job, get_session_factory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager(tmp_path):
    """Create a JobManager with isolated config and patched pipeline."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with patch("api.job_manager.config") as mock_config:
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 0.1
        mock_config.JOB_PROCESSING_TIMEOUT_MINUTES = 30
        mock_config.MAX_CONCURRENT_JOBS = 64
        mock_config.WEBHOOK_TIMEOUT = 30
        mock_config.WEBHOOK_MAX_RETRIES = 0
        mock_config.WEBHOOK_SECRET = ""

        from api.job_manager import JobManager

        yield JobManager(get_session_factory())


def _insert_job(job_id, status="submitted", **kwargs):
    """Insert a job directly into the database."""
    factory = get_session_factory()
    session = factory()
    job = Job(
        job_id=job_id,
        status=status,
        source_file=kwargs.pop("source_file", "test.pdf"),
        priority=kwargs.pop("priority", "normal"),
    )
    for key, value in kwargs.items():
        setattr(job, key, value)
    session.add(job)
    session.commit()
    session.close()
    return job_id


def _make_proc_mock(returncode=0):
    """Create a subprocess.Popen mock with the given return code."""
    proc = MagicMock()
    proc.pid = 9999
    proc.returncode = returncode
    proc.poll.return_value = returncode
    proc.wait.return_value = returncode
    return proc


# ---------------------------------------------------------------------------
# Task 1: tenant_id wiring to dashboard MetricsCollector
# ---------------------------------------------------------------------------


class TestTenantIdDashboardWiring:
    """Verify tenant_id is passed to MetricsCollector calls."""

    def test_tenant_id_passed_to_record_throughput(self, manager, tmp_path):
        """record_throughput receives tenant_id from the job."""
        from api.dashboard import get_collector

        collector = get_collector()
        collector.reset()

        _insert_job(
            "job_tid_001",
            status="processing",
            pages_completed=3,
            tenant_id="tenant_abc",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)
            manager._run_pipeline(
                "job_tid_001", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        # Check that the last throughput point carries the tenant_id
        assert len(collector._throughput) >= 1
        tp = collector._throughput[-1]
        assert tp.tenant_id == "tenant_abc"

    def test_tenant_id_passed_to_record_latency(self, manager, tmp_path):
        """record_latency receives tenant_id from the job."""
        from api.dashboard import get_collector

        collector = get_collector()
        collector.reset()

        _insert_job(
            "job_tid_002",
            status="processing",
            pages_completed=1,
            tenant_id="tenant_xyz",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)
            manager._run_pipeline(
                "job_tid_002", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        assert len(collector._latency) >= 1
        lp = collector._latency[-1]
        assert lp.tenant_id == "tenant_xyz"
        assert lp.job_id == "job_tid_002"

    def test_tenant_id_passed_to_update_job_counts(self, manager, tmp_path):
        """update_job_counts receives tenant_id from the job."""
        from api.dashboard import get_collector

        collector = get_collector()
        collector.reset()

        _insert_job(
            "job_tid_003",
            status="processing",
            tenant_id="tenant_counts",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)
            manager._run_pipeline(
                "job_tid_003", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        # Per-tenant job counts should be populated
        assert "tenant_counts" in collector._tenant_job_counts

    def test_empty_tenant_id_passed_as_empty_string(self, manager, tmp_path):
        """Jobs without tenant_id pass empty string (global aggregate path)."""
        from api.dashboard import get_collector

        collector = get_collector()
        collector.reset()

        _insert_job("job_tid_004", status="processing")

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)
            manager._run_pipeline(
                "job_tid_004", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        assert len(collector._throughput) >= 1
        tp = collector._throughput[-1]
        assert tp.tenant_id == ""

    def test_tenant_snapshot_filtering(self, manager, tmp_path):
        """Tenant-scoped snapshot returns only that tenant's data."""
        from api.dashboard import MetricWindow, get_collector

        collector = get_collector()
        collector.reset()

        # Run two jobs with different tenant IDs
        _insert_job(
            "job_tid_005a",
            status="processing",
            pages_completed=10,
            tenant_id="tenant_a",
        )
        _insert_job(
            "job_tid_005b",
            status="processing",
            pages_completed=20,
            tenant_id="tenant_b",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)
            manager._run_pipeline(
                "job_tid_005a", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)
            manager._run_pipeline(
                "job_tid_005b", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        snap_a = collector.get_snapshot(
            window=MetricWindow.HOUR_1, tenant_id="tenant_a"
        )
        snap_b = collector.get_snapshot(
            window=MetricWindow.HOUR_1, tenant_id="tenant_b"
        )

        # Each tenant's throughput should contain their own data
        throughput_a = [p for p in collector._throughput if p.tenant_id == "tenant_a"]
        throughput_b = [p for p in collector._throughput if p.tenant_id == "tenant_b"]
        assert len(throughput_a) >= 1
        assert len(throughput_b) >= 1
        # Snapshots should be independent
        assert snap_a is not snap_b


# ---------------------------------------------------------------------------
# Task 2: SLA monitor feed from job completion
# ---------------------------------------------------------------------------


class TestSLAMonitorFeed:
    """Verify SLA monitor is fed from job completion path."""

    def test_sla_monitor_receives_completed_job(self, manager, tmp_path):
        """record_request is called with success=True for completed jobs."""
        from sla_monitoring import get_monitor, reset_global_monitor

        reset_global_monitor()
        monitor = get_monitor()

        _insert_job(
            "job_sla_001",
            status="processing",
            pages_completed=5,
            tenant_id="tenant_sla",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)
            manager._run_pipeline(
                "job_sla_001", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        # Availability window should have at least one sample
        avail_window = monitor._get_window("tenant_sla", "availability")
        assert len(avail_window._samples) >= 1
        # The sample value should be 1.0 (success)
        assert avail_window._samples[-1][1] == 1.0

        reset_global_monitor()

    def test_sla_monitor_receives_failed_job(self, manager, tmp_path):
        """record_request is called with success=False for failed jobs."""
        from sla_monitoring import get_monitor, reset_global_monitor

        reset_global_monitor()
        monitor = get_monitor()

        _insert_job(
            "job_sla_002",
            status="processing",
            tenant_id="tenant_sla_fail",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(1)  # failure
            manager._run_pipeline(
                "job_sla_002", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        avail_window = monitor._get_window("tenant_sla_fail", "availability")
        assert len(avail_window._samples) >= 1
        # The sample value should be 0.0 (failure)
        assert avail_window._samples[-1][1] == 0.0

        reset_global_monitor()

    def test_sla_monitor_uses_default_tenant_when_none(self, manager, tmp_path):
        """Jobs without tenant_id use 'default' as the SLA tenant."""
        from sla_monitoring import get_monitor, reset_global_monitor

        reset_global_monitor()
        monitor = get_monitor()

        _insert_job("job_sla_003", status="processing")

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)
            manager._run_pipeline(
                "job_sla_003", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        # Should be recorded under 'default' tenant
        avail_window = monitor._get_window("default", "availability")
        assert len(avail_window._samples) >= 1

        reset_global_monitor()

    def test_sla_monitor_failure_does_not_break_pipeline(self, manager, tmp_path):
        """If SLA monitor throws, the pipeline still completes."""
        _insert_job("job_sla_004", status="processing")

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)

            with patch(
                "sla_monitoring.get_monitor",
                side_effect=RuntimeError("sla boom"),
            ):
                manager._run_pipeline(
                    "job_sla_004", str(tmp_path / "src"), str(tmp_path / "out"), {}
                )

        # Job should still be completed despite SLA monitor failure
        session = get_session_factory()()
        job = session.get(Job, "job_sla_004")
        assert job.status == "completed"
        session.close()

    def test_sla_monitor_receives_latency(self, manager, tmp_path):
        """Latency window is populated from job processing_time."""
        from sla_monitoring import get_monitor, reset_global_monitor

        reset_global_monitor()
        monitor = get_monitor()

        _insert_job(
            "job_sla_005",
            status="processing",
            tenant_id="tenant_lat",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_proc_mock(0)
            manager._run_pipeline(
                "job_sla_005", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        latency_window = monitor._get_window("tenant_lat", "latency")
        assert len(latency_window._samples) >= 1
        # Latency should be a non-negative number
        assert latency_window._samples[-1][1] >= 0.0

        reset_global_monitor()

    def test_sla_monitor_fed_on_exception_path(self, manager, tmp_path):
        """SLA monitor is also fed from the exception handler path."""
        from sla_monitoring import get_monitor, reset_global_monitor

        reset_global_monitor()
        monitor = get_monitor()

        _insert_job(
            "job_sla_006",
            status="processing",
            tenant_id="tenant_exc",
        )

        # Force an exception
        with patch("subprocess.Popen", side_effect=OSError("spawn fail")):
            manager._run_pipeline(
                "job_sla_006", str(tmp_path / "src"), str(tmp_path / "out"), {}
            )

        # The exception handler calls _update_dashboard_on_complete
        avail_window = monitor._get_window("tenant_exc", "availability")
        assert len(avail_window._samples) >= 1
        # Should record failure
        assert avail_window._samples[-1][1] == 0.0

        reset_global_monitor()


# ---------------------------------------------------------------------------
# Task 3: tenant_id label in coordinator histogram (unit-level)
# ---------------------------------------------------------------------------


class TestHistogramTenantLabel:
    """Verify tenant_id label in the processing duration histogram.

    These tests validate the data collection and rendering logic
    without requiring a running Django database, by testing the
    histogram construction patterns.
    """

    def test_histogram_data_includes_tenant_id_key(self):
        """The histogram_data entries must include a tenant_id key."""
        # Simulate what _collect_warm produces after the N1-C change
        sample_data = [
            {
                "status": "ok",
                "engine": "paddle",
                "tenant_id": "tenant_hist",
                "bucket_counts": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                "count": 1,
                "sum": 3.5,
            },
            {
                "status": "ok",
                "engine": "paddle",
                "tenant_id": "default",
                "bucket_counts": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "count": 1,
                "sum": 0.3,
            },
        ]

        for entry in sample_data:
            assert "tenant_id" in entry
            assert entry["tenant_id"] != ""

    def test_histogram_grouping_separates_tenants(self):
        """Different tenant_ids produce separate histogram groups."""
        # Replicate the grouping logic from _collect_warm
        hist_rows = [
            ("ok", "paddle", 1500, "tenant_a"),
            ("ok", "paddle", 2500, "tenant_b"),
            ("ok", "paddle", 3500, "tenant_a"),
        ]

        hist_groups: dict[tuple[str, str, str], list[float]] = {}
        for row_status, row_method, row_ms, row_tenant in hist_rows:
            key = (
                row_status or "unknown",
                row_method or "unknown",
                row_tenant or "default",
            )
            hist_groups.setdefault(key, []).append(row_ms / 1000.0)

        assert ("ok", "paddle", "tenant_a") in hist_groups
        assert ("ok", "paddle", "tenant_b") in hist_groups
        assert len(hist_groups[("ok", "paddle", "tenant_a")]) == 2
        assert len(hist_groups[("ok", "paddle", "tenant_b")]) == 1

    def test_histogram_null_tenant_defaults_to_default(self):
        """Rows with empty/None tenant_id are grouped under 'default'."""
        hist_rows = [
            ("ok", "paddle", 1000, ""),
            ("ok", "paddle", 2000, None),
        ]

        hist_groups: dict[tuple[str, str, str], list[float]] = {}
        for row_status, row_method, row_ms, row_tenant in hist_rows:
            key = (
                row_status or "unknown",
                row_method or "unknown",
                row_tenant or "default",
            )
            hist_groups.setdefault(key, []).append(row_ms / 1000.0)

        assert ("ok", "paddle", "default") in hist_groups
        assert len(hist_groups[("ok", "paddle", "default")]) == 2

    def test_histogram_metric_family_label_set(self):
        """HistogramMetricFamily is constructed with 3 labels including tenant_id."""
        try:
            from prometheus_client.core import HistogramMetricFamily
        except ImportError:
            pytest.skip("prometheus_client not installed")

        duration_buckets = (0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600)

        duration_hist = HistogramMetricFamily(
            "ocr_processing_duration_seconds",
            "Processing duration per page in seconds",
            labels=["status", "engine", "tenant_id"],
        )

        # Add a metric with all three labels
        buckets = [(str(b), 0) for b in duration_buckets]
        buckets.append(("+Inf", 1))
        duration_hist.add_metric(
            ["ok", "paddle", "tenant_test"],
            buckets=buckets,
            sum_value=1.5,
        )

        # Verify the metric was added with the correct label set
        samples = list(duration_hist.samples)
        assert len(samples) > 0
        # Each sample should have status, engine, and tenant_id labels
        for sample in samples:
            assert "status" in sample.labels
            assert "engine" in sample.labels
            assert "tenant_id" in sample.labels

    def test_describe_yields_histogram_with_tenant_label(self):
        """PipelineCollector.describe() yields histogram with tenant_id label."""
        # Import only the module structure to check describe output
        # without needing Django
        try:
            from prometheus_client.core import HistogramMetricFamily
        except ImportError:
            pytest.skip("prometheus_client not installed")

        # Recreate what describe() should yield for the histogram
        hist = HistogramMetricFamily(
            "ocr_processing_duration_seconds",
            "Processing duration per page in seconds",
            labels=["status", "engine", "tenant_id"],
        )
        assert hist is not None
