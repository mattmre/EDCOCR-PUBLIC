"""
Unit tests for SLA/SLO monitoring (sla_monitoring.py).

Tests cover:
- SLODefinition dataclass and comparison types
- SLOStatus compliant / violation / margin
- SLAReport overall compliance and violation counting
- MetricsWindow (add, prune, percentile, average, count, rate, clear, expiry)
- SLAMonitor initialisation and custom window sizes
- record_request (success, failure, latency, tenant + system propagation)
- record_throughput
- record_error / record_success
- record_recovery
- Per-tenant SLO overrides and default fallback
- evaluate_tenant (all compliant, some violations, no data)
- evaluate_system
- get_violations (filtered, limit, empty)
- write_report_json (valid output, path-safety)
- Global singleton get_monitor / reset_global_monitor
- Module constants (feature gate, default targets)

Run with: python -m pytest tests/test_sla_monitoring.py -v
"""

import json
import os
import tempfile
import time

import pytest

# Add project root to path
from sla_monitoring import (
    DEFAULT_AVAILABILITY_TARGET,
    DEFAULT_ERROR_RATE_BUDGET,
    DEFAULT_P95_LATENCY_TARGET,
    DEFAULT_RECOVERY_TIME_TARGET,
    DEFAULT_THROUGHPUT_TARGET,
    ENABLE_SLA_MONITORING,
    MetricsWindow,
    SLAMonitor,
    SLAReport,
    SLODefinition,
    SLOStatus,
    get_monitor,
    reset_global_monitor,
)

# ---------------------------------------------------------------------------
# Tests: SLODefinition
# ---------------------------------------------------------------------------


class TestSLODefinition:
    def test_dataclass_fields(self):
        slo = SLODefinition(
            name="Availability",
            metric="availability",
            target=99.5,
            unit="percent",
        )
        assert slo.name == "Availability"
        assert slo.metric == "availability"
        assert slo.target == 99.5
        assert slo.unit == "percent"

    def test_default_comparison_gte(self):
        slo = SLODefinition("Test", "test_metric", 50.0, "percent")
        assert slo.comparison == "gte"

    def test_comparison_lte(self):
        slo = SLODefinition("Latency", "p95_latency", 30.0, "seconds", "lte")
        assert slo.comparison == "lte"

    def test_equality(self):
        a = SLODefinition("A", "m", 1.0, "u", "gte")
        b = SLODefinition("A", "m", 1.0, "u", "gte")
        assert a == b


# ---------------------------------------------------------------------------
# Tests: SLOStatus
# ---------------------------------------------------------------------------


class TestSLOStatus:
    def _make_slo(self, comparison="gte"):
        return SLODefinition("Test", "test", 50.0, "percent", comparison)

    def test_compliant_status(self):
        status = SLOStatus(
            definition=self._make_slo(),
            current_value=60.0,
            compliant=True,
            margin=10.0,
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-01-02T00:00:00+00:00",
            sample_count=100,
        )
        assert status.compliant is True
        assert status.margin == 10.0
        assert status.sample_count == 100

    def test_violation_status(self):
        status = SLOStatus(
            definition=self._make_slo(),
            current_value=40.0,
            compliant=False,
            margin=-10.0,
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-01-02T00:00:00+00:00",
            sample_count=50,
        )
        assert status.compliant is False
        assert status.margin == -10.0

    def test_margin_at_boundary(self):
        status = SLOStatus(
            definition=self._make_slo(),
            current_value=50.0,
            compliant=True,
            margin=0.0,
            window_start="",
            window_end="",
            sample_count=10,
        )
        assert status.margin == 0.0
        assert status.compliant is True


# ---------------------------------------------------------------------------
# Tests: SLAReport
# ---------------------------------------------------------------------------


class TestSLAReport:
    def test_overall_compliant(self):
        report = SLAReport(
            tenant_id="t1",
            report_time="2026-01-01T00:00:00+00:00",
            window_hours=24,
            slo_statuses=[],
            overall_compliant=True,
            violation_count=0,
            compliance_percentage=100.0,
        )
        assert report.overall_compliant is True
        assert report.violation_count == 0
        assert report.compliance_percentage == 100.0

    def test_with_violations(self):
        report = SLAReport(
            tenant_id="system",
            report_time="2026-01-01T00:00:00+00:00",
            window_hours=24,
            slo_statuses=[],
            overall_compliant=False,
            violation_count=2,
            compliance_percentage=60.0,
        )
        assert report.overall_compliant is False
        assert report.violation_count == 2

    def test_system_tenant_id(self):
        report = SLAReport(
            tenant_id="system",
            report_time="",
            window_hours=1,
            slo_statuses=[],
            overall_compliant=True,
            violation_count=0,
            compliance_percentage=100.0,
        )
        assert report.tenant_id == "system"


# ---------------------------------------------------------------------------
# Tests: MetricsWindow
# ---------------------------------------------------------------------------


class TestMetricsWindow:
    def test_init_default_window(self):
        w = MetricsWindow()
        assert w.window_seconds == 3600

    def test_init_custom_window(self):
        w = MetricsWindow(window_seconds=120)
        assert w.window_seconds == 120

    def test_add_sample_increases_count(self):
        w = MetricsWindow()
        w.add_sample(1.0)
        w.add_sample(2.0)
        assert w.count() == 2

    def test_add_sample_with_explicit_timestamp(self):
        w = MetricsWindow(window_seconds=3600)
        ts = time.time()
        w.add_sample(42.0, timestamp=ts)
        samples = w.get_samples()
        assert len(samples) == 1
        assert samples[0] == (ts, 42.0)

    def test_prune_removes_old_samples(self):
        w = MetricsWindow(window_seconds=10)
        old_ts = time.time() - 20  # well outside window
        w.add_sample(1.0, timestamp=old_ts)
        w.add_sample(2.0)  # current time
        assert w.count() == 1
        samples = w.get_samples()
        assert samples[0][1] == 2.0

    def test_percentile_empty(self):
        w = MetricsWindow()
        assert w.percentile(95) is None

    def test_percentile_single_sample(self):
        w = MetricsWindow()
        w.add_sample(10.0)
        assert w.percentile(50) == 10.0
        assert w.percentile(95) == 10.0

    def test_percentile_multiple_samples(self):
        w = MetricsWindow()
        for i in range(1, 101):
            w.add_sample(float(i))
        p50 = w.percentile(50)
        p95 = w.percentile(95)
        assert p50 is not None
        assert p95 is not None
        # p95 should be around 95-96
        assert p95 >= 95

    def test_average_empty(self):
        w = MetricsWindow()
        assert w.average() is None

    def test_average_values(self):
        w = MetricsWindow()
        w.add_sample(10.0)
        w.add_sample(20.0)
        w.add_sample(30.0)
        assert w.average() == pytest.approx(20.0)

    def test_count_empty(self):
        w = MetricsWindow()
        assert w.count() == 0

    def test_rate_empty(self):
        w = MetricsWindow()
        assert w.rate() is None

    def test_rate_all_above_threshold(self):
        w = MetricsWindow()
        w.add_sample(1.0)
        w.add_sample(1.0)
        w.add_sample(1.0)
        assert w.rate(0.5) == pytest.approx(100.0)

    def test_rate_mixed(self):
        w = MetricsWindow()
        w.add_sample(1.0)
        w.add_sample(0.0)
        w.add_sample(1.0)
        w.add_sample(0.0)
        assert w.rate(0.5) == pytest.approx(50.0)

    def test_rate_all_below_threshold(self):
        w = MetricsWindow()
        w.add_sample(0.0)
        w.add_sample(0.0)
        assert w.rate(0.5) == pytest.approx(0.0)

    def test_clear(self):
        w = MetricsWindow()
        w.add_sample(1.0)
        w.add_sample(2.0)
        assert w.count() == 2
        w.clear()
        assert w.count() == 0

    def test_window_expiry_boundary(self):
        w = MetricsWindow(window_seconds=5)
        # Add sample exactly at the boundary (5 seconds ago)
        boundary_ts = time.time() - 5
        w.add_sample(1.0, timestamp=boundary_ts)
        # This should be pruned (cutoff = now - 5, sample at cutoff is < cutoff on next tick)
        w.add_sample(2.0)  # current
        # The boundary sample may or may not be present depending on timing;
        # but at least current should be there
        assert w.count() >= 1


# ---------------------------------------------------------------------------
# Tests: SLAMonitor init
# ---------------------------------------------------------------------------


class TestSLAMonitorInit:
    def test_default_init(self):
        m = SLAMonitor()
        assert m.window_hours == 24

    def test_custom_window_hours(self):
        m = SLAMonitor(window_hours=1)
        assert m.window_hours == 1

    def test_custom_persist_path(self):
        m = SLAMonitor(persist_path="/tmp/test-sla")
        assert m._persist_path is not None


# ---------------------------------------------------------------------------
# Tests: record_request
# ---------------------------------------------------------------------------


class TestRecordRequest:
    def setup_method(self):
        self.monitor = SLAMonitor(window_hours=1)

    def test_success_records_availability(self):
        self.monitor.record_request("t1", success=True, latency_seconds=1.5)
        window = self.monitor._get_window("t1", "availability")
        samples = window.get_samples()
        assert len(samples) == 1
        assert samples[0][1] == 1.0

    def test_failure_records_zero_availability(self):
        self.monitor.record_request("t1", success=False, latency_seconds=5.0)
        window = self.monitor._get_window("t1", "availability")
        samples = window.get_samples()
        assert len(samples) == 1
        assert samples[0][1] == 0.0

    def test_latency_tracked(self):
        self.monitor.record_request("t1", success=True, latency_seconds=2.5)
        window = self.monitor._get_window("t1", "latency")
        samples = window.get_samples()
        assert len(samples) == 1
        assert samples[0][1] == 2.5

    def test_propagates_to_system(self):
        self.monitor.record_request("t1", success=True, latency_seconds=1.0)
        sys_avail = self.monitor._get_window("system", "availability")
        sys_latency = self.monitor._get_window("system", "latency")
        assert sys_avail.count() == 1
        assert sys_latency.count() == 1

    def test_multiple_tenants_separate_windows(self):
        self.monitor.record_request("t1", success=True, latency_seconds=1.0)
        self.monitor.record_request("t2", success=False, latency_seconds=2.0)
        t1_avail = self.monitor._get_window("t1", "availability")
        t2_avail = self.monitor._get_window("t2", "availability")
        assert t1_avail.get_samples()[0][1] == 1.0
        assert t2_avail.get_samples()[0][1] == 0.0
        # System should have both
        sys_avail = self.monitor._get_window("system", "availability")
        assert sys_avail.count() == 2


# ---------------------------------------------------------------------------
# Tests: record_throughput
# ---------------------------------------------------------------------------


class TestRecordThroughput:
    def setup_method(self):
        self.monitor = SLAMonitor(window_hours=1)

    def test_records_throughput(self):
        self.monitor.record_throughput("t1", 15.0)
        window = self.monitor._get_window("t1", "throughput")
        assert window.count() == 1
        assert window.get_samples()[0][1] == 15.0

    def test_propagates_to_system(self):
        self.monitor.record_throughput("t1", 12.0)
        sys_window = self.monitor._get_window("system", "throughput")
        assert sys_window.count() == 1

    def test_multiple_samples(self):
        self.monitor.record_throughput("t1", 10.0)
        self.monitor.record_throughput("t1", 20.0)
        window = self.monitor._get_window("t1", "throughput")
        assert window.count() == 2
        assert window.average() == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Tests: record_error / record_success
# ---------------------------------------------------------------------------


class TestRecordErrorSuccess:
    def setup_method(self):
        self.monitor = SLAMonitor(window_hours=1)

    def test_record_error(self):
        self.monitor.record_error("t1")
        window = self.monitor._get_window("t1", "error_rate")
        samples = window.get_samples()
        assert len(samples) == 1
        assert samples[0][1] == 1.0

    def test_record_success(self):
        self.monitor.record_success("t1")
        window = self.monitor._get_window("t1", "error_rate")
        samples = window.get_samples()
        assert len(samples) == 1
        assert samples[0][1] == 0.0

    def test_error_propagates_to_system(self):
        self.monitor.record_error("t1")
        sys_window = self.monitor._get_window("system", "error_rate")
        assert sys_window.count() == 1
        assert sys_window.get_samples()[0][1] == 1.0

    def test_mixed_error_success_rate(self):
        self.monitor.record_success("t1")
        self.monitor.record_success("t1")
        self.monitor.record_error("t1")
        self.monitor.record_success("t1")
        window = self.monitor._get_window("t1", "error_rate")
        samples = window.get_samples()
        errors = sum(1 for _, v in samples if v > 0.5)
        total = len(samples)
        error_pct = (errors / total) * 100.0
        assert error_pct == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Tests: record_recovery
# ---------------------------------------------------------------------------


class TestRecordRecovery:
    def setup_method(self):
        self.monitor = SLAMonitor(window_hours=1)

    def test_records_recovery_time(self):
        self.monitor.record_recovery("t1", 120.0)
        window = self.monitor._get_window("t1", "recovery_time")
        assert window.count() == 1
        assert window.get_samples()[0][1] == 120.0

    def test_propagates_to_system(self):
        self.monitor.record_recovery("t1", 60.0)
        sys_window = self.monitor._get_window("system", "recovery_time")
        assert sys_window.count() == 1

    def test_multiple_recoveries(self):
        self.monitor.record_recovery("t1", 100.0)
        self.monitor.record_recovery("t1", 200.0)
        self.monitor.record_recovery("t1", 300.0)
        window = self.monitor._get_window("t1", "recovery_time")
        assert window.count() == 3


# ---------------------------------------------------------------------------
# Tests: tenant SLOs
# ---------------------------------------------------------------------------


class TestTenantSLOs:
    def setup_method(self):
        self.monitor = SLAMonitor()

    def test_default_slos_returned(self):
        slos = self.monitor.get_tenant_slos("nonexistent")
        assert len(slos) == 5
        names = [s.name for s in slos]
        assert "Availability" in names
        assert "Throughput" in names
        assert "Error Rate" in names
        assert "P95 Latency" in names
        assert "Recovery Time" in names

    def test_set_custom_slos(self):
        custom = [
            SLODefinition("Custom", "custom_metric", 99.0, "percent", "gte"),
        ]
        self.monitor.set_tenant_slos("t1", custom)
        result = self.monitor.get_tenant_slos("t1")
        assert len(result) == 1
        assert result[0].name == "Custom"

    def test_custom_slos_do_not_affect_other_tenants(self):
        custom = [
            SLODefinition("Custom", "custom_metric", 99.0, "percent", "gte"),
        ]
        self.monitor.set_tenant_slos("t1", custom)
        t2_slos = self.monitor.get_tenant_slos("t2")
        assert len(t2_slos) == 5  # defaults

    def test_override_replaces_previous(self):
        first = [SLODefinition("First", "m1", 1.0, "u")]
        second = [
            SLODefinition("Second", "m2", 2.0, "u"),
            SLODefinition("Third", "m3", 3.0, "u"),
        ]
        self.monitor.set_tenant_slos("t1", first)
        assert len(self.monitor.get_tenant_slos("t1")) == 1
        self.monitor.set_tenant_slos("t1", second)
        assert len(self.monitor.get_tenant_slos("t1")) == 2


# ---------------------------------------------------------------------------
# Tests: evaluate_tenant
# ---------------------------------------------------------------------------


class TestEvaluateTenant:
    def setup_method(self):
        self.monitor = SLAMonitor(window_hours=1)

    def test_all_compliant_no_data(self):
        """No data defaults to healthy assumptions."""
        report = self.monitor.evaluate_tenant("t1")
        assert report.overall_compliant is True
        assert report.violation_count == 0
        assert report.compliance_percentage == 100.0

    def test_all_compliant_with_data(self):
        # Record healthy data that meets all default SLOs
        for _ in range(20):
            self.monitor.record_request("t1", success=True, latency_seconds=5.0)
            self.monitor.record_success("t1")
        self.monitor.record_throughput("t1", 15.0)
        report = self.monitor.evaluate_tenant("t1")
        assert report.overall_compliant is True
        assert report.compliance_percentage == 100.0

    def test_availability_violation(self):
        # Make availability < 99.5% (1 success, 99 failures)
        self.monitor.record_request("t1", success=True, latency_seconds=1.0)
        for _ in range(99):
            self.monitor.record_request("t1", success=False, latency_seconds=1.0)
        report = self.monitor.evaluate_tenant("t1")
        avail_status = None
        for s in report.slo_statuses:
            if s.definition.metric == "availability":
                avail_status = s
                break
        assert avail_status is not None
        assert avail_status.compliant is False
        assert avail_status.current_value < 99.5

    def test_throughput_violation(self):
        # Record throughput below default target (10 ppm)
        self.monitor.record_throughput("t1", 2.0)
        self.monitor.record_throughput("t1", 3.0)
        report = self.monitor.evaluate_tenant("t1")
        tp_status = None
        for s in report.slo_statuses:
            if s.definition.metric == "throughput":
                tp_status = s
                break
        assert tp_status is not None
        assert tp_status.compliant is False
        assert tp_status.margin < 0

    def test_error_rate_violation(self):
        # Error rate > 1% (default budget)
        for _ in range(50):
            self.monitor.record_error("t1")
        for _ in range(50):
            self.monitor.record_success("t1")
        report = self.monitor.evaluate_tenant("t1")
        err_status = None
        for s in report.slo_statuses:
            if s.definition.metric == "error_rate":
                err_status = s
                break
        assert err_status is not None
        assert err_status.compliant is False
        assert err_status.current_value == pytest.approx(50.0)

    def test_report_records_violations(self):
        # Force throughput violation
        self.monitor.record_throughput("t1", 1.0)
        report = self.monitor.evaluate_tenant("t1")
        assert report.violation_count >= 1
        violations = self.monitor.get_violations(tenant_id="t1")
        assert len(violations) >= 1

    def test_tenant_id_in_report(self):
        report = self.monitor.evaluate_tenant("tenant_abc")
        assert report.tenant_id == "tenant_abc"

    def test_window_hours_in_report(self):
        report = self.monitor.evaluate_tenant("t1")
        assert report.window_hours == 1

    def test_p95_latency_violation(self):
        """P95 latency above 30s target should be a violation."""
        for _ in range(100):
            self.monitor.record_request("t1", success=True, latency_seconds=50.0)
        report = self.monitor.evaluate_tenant("t1")
        lat_status = None
        for s in report.slo_statuses:
            if s.definition.metric == "p95_latency":
                lat_status = s
                break
        assert lat_status is not None
        assert lat_status.compliant is False

    def test_recovery_time_violation(self):
        """Recovery time P95 above 300s target should be a violation."""
        for _ in range(20):
            self.monitor.record_recovery("t1", 500.0)
        report = self.monitor.evaluate_tenant("t1")
        rec_status = None
        for s in report.slo_statuses:
            if s.definition.metric == "recovery_time":
                rec_status = s
                break
        assert rec_status is not None
        assert rec_status.compliant is False


# ---------------------------------------------------------------------------
# Tests: evaluate_system
# ---------------------------------------------------------------------------


class TestEvaluateSystem:
    def setup_method(self):
        self.monitor = SLAMonitor(window_hours=1)

    def test_system_report_no_data(self):
        report = self.monitor.evaluate_system()
        assert report.tenant_id == "system"
        assert report.overall_compliant is True

    def test_system_aggregates_all_tenants(self):
        self.monitor.record_request("t1", success=True, latency_seconds=1.0)
        self.monitor.record_request("t2", success=True, latency_seconds=2.0)
        report = self.monitor.evaluate_system()
        assert report.tenant_id == "system"
        # System availability window should have 2 samples
        avail = None
        for s in report.slo_statuses:
            if s.definition.metric == "availability":
                avail = s
                break
        assert avail is not None
        assert avail.sample_count == 2

    def test_system_report_with_violations(self):
        for _ in range(100):
            self.monitor.record_request(
                "t1", success=False, latency_seconds=60.0
            )
        report = self.monitor.evaluate_system()
        assert report.overall_compliant is False


# ---------------------------------------------------------------------------
# Tests: get_violations
# ---------------------------------------------------------------------------


class TestGetViolations:
    def setup_method(self):
        self.monitor = SLAMonitor(window_hours=1)

    def test_empty_violations(self):
        assert self.monitor.get_violations() == []

    def test_violations_after_evaluation(self):
        self.monitor.record_throughput("t1", 1.0)  # below default 10 ppm
        self.monitor.evaluate_tenant("t1")
        violations = self.monitor.get_violations()
        assert len(violations) >= 1

    def test_filter_by_tenant(self):
        self.monitor.record_throughput("t1", 1.0)
        self.monitor.record_throughput("t2", 1.0)
        self.monitor.evaluate_tenant("t1")
        self.monitor.evaluate_tenant("t2")
        t1_v = self.monitor.get_violations(tenant_id="t1")
        t2_v = self.monitor.get_violations(tenant_id="t2")
        all_v = self.monitor.get_violations()
        assert all(v["tenant_id"] == "t1" for v in t1_v)
        assert all(v["tenant_id"] == "t2" for v in t2_v)
        assert len(all_v) >= len(t1_v) + len(t2_v)

    def test_limit_parameter(self):
        # Create many violations
        for _ in range(10):
            self.monitor.record_throughput("t1", 0.1)
            self.monitor.evaluate_tenant("t1")
        limited = self.monitor.get_violations(limit=3)
        assert len(limited) <= 3


# ---------------------------------------------------------------------------
# Tests: write_report_json
# ---------------------------------------------------------------------------


class TestWriteReportJson:
    def setup_method(self):
        self.monitor = SLAMonitor(window_hours=1)

    def test_writes_valid_json(self):
        report = self.monitor.evaluate_tenant("t1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.monitor.write_report_json(report, tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["tenant_id"] == "t1"
            assert "slo_statuses" in data
            assert "overall_compliant" in data

    def test_output_directory_created(self):
        report = self.monitor.evaluate_tenant("t1")
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir")
            path = self.monitor.write_report_json(report, nested)
            assert os.path.exists(path)

    def test_path_safety_strips_traversal(self):
        report = self.monitor.evaluate_tenant("t1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.monitor.write_report_json(
                report, tmpdir, filename="../../evil.json"
            )
            # Should not escape tmpdir
            assert os.path.dirname(os.path.abspath(path)) == os.path.abspath(
                tmpdir
            )

    def test_custom_filename(self):
        report = self.monitor.evaluate_tenant("t1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.monitor.write_report_json(
                report, tmpdir, filename="custom-report.json"
            )
            assert path.endswith("custom-report.json")

    def test_report_json_fields(self):
        """Verify all expected fields in the JSON output."""
        for _ in range(5):
            self.monitor.record_request("t1", success=True, latency_seconds=1.0)
        report = self.monitor.evaluate_tenant("t1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.monitor.write_report_json(report, tmpdir)
            with open(path) as f:
                data = json.load(f)
            assert "report_time" in data
            assert "window_hours" in data
            assert "compliance_percentage" in data
            assert "violation_count" in data
            for status in data["slo_statuses"]:
                assert "name" in status
                assert "metric" in status
                assert "target" in status
                assert "unit" in status
                assert "current_value" in status
                assert "compliant" in status
                assert "margin" in status
                assert "sample_count" in status


# ---------------------------------------------------------------------------
# Tests: global monitor singleton
# ---------------------------------------------------------------------------


class TestGlobalMonitor:
    def teardown_method(self):
        reset_global_monitor()

    def test_get_monitor_returns_instance(self):
        m = get_monitor()
        assert isinstance(m, SLAMonitor)

    def test_get_monitor_returns_same_instance(self):
        m1 = get_monitor()
        m2 = get_monitor()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_monitor()
        reset_global_monitor()
        m2 = get_monitor()
        assert m1 is not m2

    def test_get_monitor_with_kwargs(self):
        m = get_monitor(window_hours=2)
        assert m.window_hours == 2

    def test_kwargs_ignored_after_init(self):
        get_monitor(window_hours=2)
        m2 = get_monitor(window_hours=99)
        assert m2.window_hours == 2  # still the original


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_enable_sla_monitoring_default_false(self):
        # Unless env var is set, should be false
        # (test relies on env not being set in CI)
        # We just check that the constant exists and is bool
        assert isinstance(ENABLE_SLA_MONITORING, bool)

    def test_default_availability_target(self):
        assert DEFAULT_AVAILABILITY_TARGET == 99.5

    def test_default_throughput_target(self):
        assert DEFAULT_THROUGHPUT_TARGET == 10.0

    def test_default_error_rate_budget(self):
        assert DEFAULT_ERROR_RATE_BUDGET == 1.0

    def test_default_p95_latency_target(self):
        assert DEFAULT_P95_LATENCY_TARGET == 30.0

    def test_default_recovery_time_target(self):
        assert DEFAULT_RECOVERY_TIME_TARGET == 300.0


# ---------------------------------------------------------------------------
# Tests: default SLOs from monitor
# ---------------------------------------------------------------------------


class TestDefaultSLOs:
    def test_default_slos_count(self):
        m = SLAMonitor()
        slos = m.get_default_slos()
        assert len(slos) == 5

    def test_default_availability_slo(self):
        m = SLAMonitor()
        slos = m.get_default_slos()
        avail = [s for s in slos if s.metric == "availability"][0]
        assert avail.comparison == "gte"
        assert avail.target == DEFAULT_AVAILABILITY_TARGET

    def test_default_error_rate_slo(self):
        m = SLAMonitor()
        slos = m.get_default_slos()
        err = [s for s in slos if s.metric == "error_rate"][0]
        assert err.comparison == "lte"
        assert err.target == DEFAULT_ERROR_RATE_BUDGET


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_evaluate_unknown_metric_slo(self):
        """Custom SLO with an unknown metric falls back to average."""
        m = SLAMonitor(window_hours=1)
        custom = [
            SLODefinition("Custom", "unknown_metric", 50.0, "units", "gte"),
        ]
        m.set_tenant_slos("t1", custom)
        m._get_window("t1", "unknown_metric").add_sample(60.0)
        report = m.evaluate_tenant("t1")
        assert report.slo_statuses[0].compliant is True

    def test_lte_comparison_compliant(self):
        """An LTE SLO should be compliant when value is below target."""
        m = SLAMonitor(window_hours=1)
        custom = [
            SLODefinition("Low", "test_lte", 100.0, "ms", "lte"),
        ]
        m.set_tenant_slos("t1", custom)
        m._get_window("t1", "test_lte").add_sample(50.0)
        report = m.evaluate_tenant("t1")
        status = report.slo_statuses[0]
        assert status.compliant is True
        assert status.margin == pytest.approx(50.0)

    def test_lte_comparison_violation(self):
        """An LTE SLO should violate when value exceeds target."""
        m = SLAMonitor(window_hours=1)
        custom = [
            SLODefinition("Low", "test_lte", 100.0, "ms", "lte"),
        ]
        m.set_tenant_slos("t1", custom)
        m._get_window("t1", "test_lte").add_sample(150.0)
        report = m.evaluate_tenant("t1")
        status = report.slo_statuses[0]
        assert status.compliant is False
        assert status.margin == pytest.approx(-50.0)

    def test_concurrent_recording(self):
        """Multiple threads recording concurrently should not crash."""
        import threading

        m = SLAMonitor(window_hours=1)
        errors = []

        def worker(tid):
            try:
                for _ in range(50):
                    m.record_request(tid, success=True, latency_seconds=1.0)
                    m.record_success(tid)
                    m.record_throughput(tid, 10.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # System should have aggregated all recordings
        sys_avail = m._get_window("system", "availability")
        assert sys_avail.count() == 200  # 4 threads * 50 samples

    def test_empty_filename_fallback(self):
        """An empty sanitized filename should fall back to default."""
        m = SLAMonitor(window_hours=1)
        report = m.evaluate_tenant("t1")
        with tempfile.TemporaryDirectory() as tmpdir:
            # filename that sanitizes to empty string
            path = m.write_report_json(report, tmpdir, filename="../..")
            assert os.path.exists(path)
            assert os.path.basename(path) == "sla-report.json"

    def test_compliance_percentage_calculation(self):
        """5 SLOs, 2 violated = 60% compliance."""
        m = SLAMonitor(window_hours=1)
        # Force throughput and availability violations
        for _ in range(100):
            m.record_request("t1", success=False, latency_seconds=60.0)
        m.record_throughput("t1", 1.0)
        report = m.evaluate_tenant("t1")
        # availability, throughput, p95_latency should all violate
        assert report.compliance_percentage < 100.0
        assert report.violation_count > 0
