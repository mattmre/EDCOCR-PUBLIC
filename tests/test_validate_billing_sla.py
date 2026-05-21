"""Tests for scripts/validate_billing_sla.py — billing & SLA policy validation."""

from __future__ import annotations

import json
import textwrap

import pytest

from scripts.validate_billing_sla import (
    CheckResult,
    ValidationReport,
    check_billing_formula_locked,
    check_billing_formula_version,
    check_cost_defaults,
    check_cost_persistence,
    check_cost_tracking_exists,
    check_grafana_panels,
    check_policy_document,
    check_prometheus_metrics,
    check_sla_defaults,
    check_sla_formula_version,
    check_sla_monitoring_exists,
    check_sla_report_export,
    check_sla_window_default,
    check_tenant_override_mechanism,
    main,
    run_validation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COST_TRACKING_SOURCE = textwrap.dedent("""\
    COST_PER_PAGE = float(os.environ.get("COST_PER_PAGE", "0.01"))
    COST_PER_GPU_SECOND = float(os.environ.get("COST_PER_GPU_SECOND", "0.001"))
    COST_PER_GB_STORED = float(os.environ.get("COST_PER_GB_STORED", "0.05"))
    COST_PER_API_CALL = float(os.environ.get("COST_PER_API_CALL", "0.0001"))
    BILLING_FORMULA_VERSION = "1.0.0"
    locked: bool = True
    def persist(self):
        pass
    persist_path = None
""")

SLA_MONITORING_SOURCE = textwrap.dedent("""\
    DEFAULT_AVAILABILITY_TARGET = float(os.environ.get("SLO_AVAILABILITY_TARGET", "99.5"))
    DEFAULT_THROUGHPUT_TARGET = float(os.environ.get("SLO_THROUGHPUT_TARGET", "10.0"))
    DEFAULT_ERROR_RATE_BUDGET = float(os.environ.get("SLO_ERROR_RATE_BUDGET", "1.0"))
    DEFAULT_P95_LATENCY_TARGET = float(os.environ.get("SLO_P95_LATENCY_TARGET", "30.0"))
    DEFAULT_RECOVERY_TIME_TARGET = float(os.environ.get("SLO_RECOVERY_TIME_TARGET", "300.0"))
    SLA_FORMULA_VERSION = "1.0.0"
    def __init__(self, window_seconds: int = 3600):
        pass
    def set_tenant_slos(self, tenant_id, slos):
        pass
    def write_report_json(self, report, output_dir, filename="sla-report.json"):
        pass
""")

PROMETHEUS_SOURCE = textwrap.dedent("""\
    _SLA_COMPLIANCE_PCT = Gauge("ocr_sla_compliance_pct", "help", labelnames=["tenant_id"])
    _SLA_VIOLATION_COUNT = Gauge("ocr_sla_violation_count", "help", labelnames=["tenant_id"])
    _SLA_AVAILABILITY_PCT = Gauge("ocr_sla_availability_pct", "help", labelnames=["tenant_id"])
    _SLA_P95_LATENCY_SECONDS = Gauge("ocr_sla_p95_latency_seconds", "help", labelnames=["tenant_id"])
    _COST_ESTIMATE_TOTAL = Gauge("ocr_cost_estimate_total", "help", labelnames=["tenant_id"])
    _TENANT_GPU_SECONDS = Gauge("ocr_tenant_gpu_seconds", "help", labelnames=["tenant_id"])
    _TENANT_STORAGE_BYTES = Gauge("ocr_tenant_storage_bytes", "help", labelnames=["tenant_id"])
""")

GRAFANA_SOURCE = textwrap.dedent("""\
    "title": "Cost per Tenant"
    "title": "Tenant Storage Consumption"
    "title": "SLA Compliance Rate"
    "title": "SLA Breach History"
""")


@pytest.fixture()
def mock_project(tmp_path):
    """Create a minimal project tree matching the expected layout."""
    (tmp_path / "cost_tracking.py").write_text(COST_TRACKING_SOURCE, encoding="utf-8")
    (tmp_path / "sla_monitoring.py").write_text(SLA_MONITORING_SOURCE, encoding="utf-8")

    api_dir = tmp_path / "api"
    api_dir.mkdir()
    (api_dir / "prometheus.py").write_text(PROMETHEUS_SOURCE, encoding="utf-8")

    helm_dir = tmp_path / "helm" / "ocr-local" / "templates"
    helm_dir.mkdir(parents=True)
    (helm_dir / "grafana-dashboard-configmap.yaml").write_text(
        GRAFANA_SOURCE, encoding="utf-8"
    )

    docs_dir = tmp_path / "docs" / "operations"
    docs_dir.mkdir(parents=True)
    (docs_dir / "billing-sla-policy.md").write_text("# Policy", encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# CheckResult / ValidationReport unit tests
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_pass_result(self):
        r = CheckResult("test", True, "ok")
        assert r.passed is True
        assert r.name == "test"

    def test_fail_result(self):
        r = CheckResult("test", False, "bad")
        assert r.passed is False


class TestValidationReport:
    def test_add_pass(self):
        report = ValidationReport()
        report.add(CheckResult("a", True))
        assert report.passed == 1
        assert report.failed == 0
        assert report.all_passed is True

    def test_add_fail(self):
        report = ValidationReport()
        report.add(CheckResult("a", False))
        assert report.passed == 0
        assert report.failed == 1
        assert report.all_passed is False

    def test_to_dict(self):
        report = ValidationReport(project_root="/tmp")
        report.add(CheckResult("a", True, "ok"))
        d = report.to_dict()
        assert d["total_checks"] == 1
        assert d["passed"] == 1
        assert d["all_passed"] is True

    def test_to_json(self):
        report = ValidationReport()
        report.add(CheckResult("a", True))
        j = report.to_json()
        data = json.loads(j)
        assert data["passed"] == 1

    def test_to_text_pass(self):
        report = ValidationReport(project_root="/tmp")
        report.add(CheckResult("check1", True, "details"))
        text = report.to_text()
        assert "PASS" in text
        assert "ALL CHECKS PASSED" in text

    def test_to_text_fail(self):
        report = ValidationReport(project_root="/tmp")
        report.add(CheckResult("check1", False, "bad"))
        text = report.to_text()
        assert "FAIL" in text
        assert "SOME CHECKS FAILED" in text


# ---------------------------------------------------------------------------
# Cost tracking validation
# ---------------------------------------------------------------------------


class TestCostTrackingChecks:
    def test_cost_tracking_exists(self, mock_project):
        r = check_cost_tracking_exists(mock_project)
        assert r.passed is True

    def test_cost_tracking_missing(self, tmp_path):
        r = check_cost_tracking_exists(tmp_path)
        assert r.passed is False

    def test_cost_defaults_match(self, mock_project):
        results = check_cost_defaults(mock_project)
        assert len(results) == 4
        assert all(r.passed for r in results), [r.detail for r in results]

    def test_cost_defaults_wrong_value(self, tmp_path):
        source = 'COST_PER_PAGE = float(os.environ.get("COST_PER_PAGE", "0.99"))\n'
        (tmp_path / "cost_tracking.py").write_text(source, encoding="utf-8")
        results = check_cost_defaults(tmp_path)
        page_result = [r for r in results if "COST_PER_PAGE" in r.name][0]
        assert page_result.passed is False
        assert "0.99" in page_result.detail

    def test_cost_defaults_unreadable(self, tmp_path):
        results = check_cost_defaults(tmp_path)
        assert len(results) == 1
        assert results[0].passed is False

    def test_billing_formula_version(self, mock_project):
        r = check_billing_formula_version(mock_project)
        assert r.passed is True

    def test_billing_formula_version_mismatch(self, tmp_path):
        (tmp_path / "cost_tracking.py").write_text(
            'BILLING_FORMULA_VERSION = "2.0.0"\n', encoding="utf-8"
        )
        r = check_billing_formula_version(tmp_path)
        assert r.passed is False

    def test_billing_formula_locked(self, mock_project):
        r = check_billing_formula_locked(mock_project)
        assert r.passed is True

    def test_billing_formula_not_locked(self, tmp_path):
        (tmp_path / "cost_tracking.py").write_text(
            "locked: bool = False\n", encoding="utf-8"
        )
        r = check_billing_formula_locked(tmp_path)
        assert r.passed is False

    def test_cost_persistence(self, mock_project):
        r = check_cost_persistence(mock_project)
        assert r.passed is True


# ---------------------------------------------------------------------------
# SLA monitoring validation
# ---------------------------------------------------------------------------


class TestSLAMonitoringChecks:
    def test_sla_monitoring_exists(self, mock_project):
        r = check_sla_monitoring_exists(mock_project)
        assert r.passed is True

    def test_sla_monitoring_missing(self, tmp_path):
        r = check_sla_monitoring_exists(tmp_path)
        assert r.passed is False

    def test_sla_defaults_match(self, mock_project):
        results = check_sla_defaults(mock_project)
        assert len(results) == 5
        assert all(r.passed for r in results), [r.detail for r in results]

    def test_sla_defaults_wrong_value(self, tmp_path):
        source = 'DEFAULT_AVAILABILITY_TARGET = float(os.environ.get("SLO_AVAILABILITY_TARGET", "90.0"))\n'
        (tmp_path / "sla_monitoring.py").write_text(source, encoding="utf-8")
        results = check_sla_defaults(tmp_path)
        avail = [r for r in results if "AVAILABILITY" in r.name][0]
        assert avail.passed is False

    def test_sla_formula_version(self, mock_project):
        r = check_sla_formula_version(mock_project)
        assert r.passed is True

    def test_sla_window_default(self, mock_project):
        r = check_sla_window_default(mock_project)
        assert r.passed is True

    def test_tenant_override_mechanism(self, mock_project):
        r = check_tenant_override_mechanism(mock_project)
        assert r.passed is True

    def test_sla_report_export(self, mock_project):
        r = check_sla_report_export(mock_project)
        assert r.passed is True


# ---------------------------------------------------------------------------
# Prometheus bridge validation
# ---------------------------------------------------------------------------


class TestPrometheusChecks:
    def test_all_metrics_present(self, mock_project):
        results = check_prometheus_metrics(mock_project)
        assert len(results) == 7
        assert all(r.passed for r in results), [r.detail for r in results]

    def test_missing_metric(self, tmp_path):
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "prometheus.py").write_text(
            '"ocr_cost_estimate_total"', encoding="utf-8"
        )
        results = check_prometheus_metrics(tmp_path)
        passed_names = {r.name for r in results if r.passed}
        failed_names = {r.name for r in results if not r.passed}
        assert "Prometheus metric ocr_cost_estimate_total" in passed_names
        assert len(failed_names) > 0

    def test_unreadable(self, tmp_path):
        results = check_prometheus_metrics(tmp_path)
        assert len(results) == 1
        assert results[0].passed is False


# ---------------------------------------------------------------------------
# Grafana panel validation
# ---------------------------------------------------------------------------


class TestGrafanaPanelChecks:
    def test_all_panels_present(self, mock_project):
        results = check_grafana_panels(mock_project)
        assert len(results) == 4
        assert all(r.passed for r in results), [r.detail for r in results]

    def test_missing_panel(self, tmp_path):
        helm_dir = tmp_path / "helm" / "ocr-local" / "templates"
        helm_dir.mkdir(parents=True)
        (helm_dir / "grafana-dashboard-configmap.yaml").write_text(
            '"title": "Cost per Tenant"', encoding="utf-8"
        )
        results = check_grafana_panels(tmp_path)
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        assert len(passed) == 1
        assert len(failed) == 3

    def test_unreadable(self, tmp_path):
        results = check_grafana_panels(tmp_path)
        assert len(results) == 1
        assert results[0].passed is False


# ---------------------------------------------------------------------------
# Policy document check
# ---------------------------------------------------------------------------


class TestPolicyDocumentCheck:
    def test_policy_exists(self, mock_project):
        r = check_policy_document(mock_project)
        assert r.passed is True

    def test_policy_missing(self, tmp_path):
        r = check_policy_document(tmp_path)
        assert r.passed is False


# ---------------------------------------------------------------------------
# Full validation run
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_all_pass_on_valid_project(self, mock_project):
        report = run_validation(mock_project)
        assert report.all_passed, report.to_text()
        assert report.passed > 0
        assert report.failed == 0

    def test_failures_on_empty_dir(self, tmp_path):
        report = run_validation(tmp_path)
        assert not report.all_passed
        assert report.failed > 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_pass(self, mock_project):
        rc = main(["--project-root", str(mock_project)])
        assert rc == 0

    def test_cli_fail(self, tmp_path):
        rc = main(["--project-root", str(tmp_path)])
        assert rc == 1

    def test_cli_json_output(self, mock_project, capsys):
        rc = main(["--project-root", str(mock_project), "--json"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["all_passed"] is True

    def test_cli_report_file(self, mock_project, tmp_path):
        report_file = tmp_path / "report.txt"
        rc = main([
            "--project-root", str(mock_project),
            "--report", str(report_file),
        ])
        assert rc == 0
        assert report_file.is_file()
        content = report_file.read_text(encoding="utf-8")
        assert "ALL CHECKS PASSED" in content

    def test_cli_json_report_file(self, mock_project, tmp_path):
        report_file = tmp_path / "report.json"
        rc = main([
            "--project-root", str(mock_project),
            "--json",
            "--report", str(report_file),
        ])
        assert rc == 0
        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert data["all_passed"] is True
