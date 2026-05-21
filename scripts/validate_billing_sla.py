#!/usr/bin/env python3
"""Validate billing and SLA policy alignment.

Reads the production source files (cost_tracking.py, sla_monitoring.py,
api/prometheus.py, Grafana dashboard) and verifies that their constants
and configuration match the locked billing-sla-policy.md document.

Usage:
    python scripts/validate_billing_sla.py [--project-root PATH] [--json] [--report PATH]

Exit codes:
    0 — all checks pass
    1 — one or more checks failed
    2 — argument / IO error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Policy constants — these are the locked values from billing-sla-policy.md
# ---------------------------------------------------------------------------

EXPECTED_COST_PER_PAGE = 0.01
EXPECTED_COST_PER_GPU_SECOND = 0.001
EXPECTED_COST_PER_GB_STORED = 0.05
EXPECTED_COST_PER_API_CALL = 0.0001
EXPECTED_BILLING_FORMULA_VERSION = "1.0.0"

EXPECTED_AVAILABILITY_TARGET = 99.5
EXPECTED_THROUGHPUT_TARGET = 10.0
EXPECTED_ERROR_RATE_BUDGET = 1.0
EXPECTED_P95_LATENCY_TARGET = 30.0
EXPECTED_RECOVERY_TIME_TARGET = 300.0
EXPECTED_SLA_FORMULA_VERSION = "1.0.0"

EXPECTED_PROMETHEUS_METRICS = [
    "ocr_cost_estimate_total",
    "ocr_tenant_gpu_seconds",
    "ocr_tenant_storage_bytes",
    "ocr_sla_compliance_pct",
    "ocr_sla_violation_count",
    "ocr_sla_availability_pct",
    "ocr_sla_p95_latency_seconds",
]

EXPECTED_GRAFANA_PANELS = [
    "Cost per Tenant",
    "Tenant Storage Consumption",
    "SLA Compliance Rate",
    "SLA Breach History",
]


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    """Aggregated validation report."""

    project_root: str = ""
    checks: list[CheckResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict:
        return {
            "project_root": self.project_root,
            "total_checks": self.passed + self.failed,
            "passed": self.passed,
            "failed": self.failed,
            "all_passed": self.all_passed,
            "checks": [asdict(c) for c in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_text(self) -> str:
        lines = [
            "=" * 60,
            "Billing & SLA Policy Validation Report",
            "=" * 60,
            f"Project root: {self.project_root}",
            f"Total checks: {self.passed + self.failed}",
            f"Passed: {self.passed}",
            f"Failed: {self.failed}",
            "",
        ]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.name}")
            if c.detail:
                lines.append(f"         {c.detail}")
        lines.append("")
        lines.append(
            "RESULT: ALL CHECKS PASSED"
            if self.all_passed
            else "RESULT: SOME CHECKS FAILED"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_file(path: Path) -> str | None:
    """Read a file and return its contents, or None if it does not exist."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _extract_float_constant(source: str, name: str) -> float | None:
    """Extract a float constant from Python source of the form:
        NAME = float(os.environ.get("...", "DEFAULT"))
    or:
        NAME = <literal>

    Handles both single-line and multi-line variants (e.g. when the
    ``float(`` call wraps across lines).
    """
    # Pattern 1: float(os.environ.get("...", "DEFAULT")) — possibly multi-line
    pattern = (
        rf'{name}\s*=\s*float\(\s*os\.environ\.get\([^,]+,\s*"([^"]+)"\)\s*\)'
    )
    m = re.search(pattern, source, re.DOTALL)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Pattern 2: direct float literal
    pattern2 = rf"{name}\s*=\s*([0-9]+\.?[0-9]*)"
    m2 = re.search(pattern2, source)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            pass

    return None


def _extract_string_constant(source: str, name: str) -> str | None:
    """Extract a string constant of the form NAME = "VALUE"."""
    pattern = rf'{name}\s*=\s*["\']([^"\']+)["\']'
    m = re.search(pattern, source)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def check_cost_tracking_exists(root: Path) -> CheckResult:
    """Verify cost_tracking.py exists."""
    path = root / "cost_tracking.py"
    if path.is_file():
        return CheckResult("cost_tracking.py exists", True)
    return CheckResult("cost_tracking.py exists", False, f"Not found at {path}")


def check_cost_defaults(root: Path) -> list[CheckResult]:
    """Verify default cost rates in cost_tracking.py match policy."""
    results = []
    source = _read_file(root / "cost_tracking.py")
    if source is None:
        results.append(
            CheckResult(
                "cost_tracking.py readable", False, "Cannot read file"
            )
        )
        return results

    checks = [
        ("COST_PER_PAGE", EXPECTED_COST_PER_PAGE),
        ("COST_PER_GPU_SECOND", EXPECTED_COST_PER_GPU_SECOND),
        ("COST_PER_GB_STORED", EXPECTED_COST_PER_GB_STORED),
        ("COST_PER_API_CALL", EXPECTED_COST_PER_API_CALL),
    ]
    for name, expected in checks:
        actual = _extract_float_constant(source, name)
        if actual is None:
            results.append(
                CheckResult(
                    f"cost default {name}",
                    False,
                    f"Could not extract {name} from source",
                )
            )
        elif abs(actual - expected) < 1e-9:
            results.append(
                CheckResult(f"cost default {name}", True, f"{actual}")
            )
        else:
            results.append(
                CheckResult(
                    f"cost default {name}",
                    False,
                    f"Expected {expected}, found {actual}",
                )
            )
    return results


def check_billing_formula_version(root: Path) -> CheckResult:
    """Verify BILLING_FORMULA_VERSION matches policy."""
    source = _read_file(root / "cost_tracking.py")
    if source is None:
        return CheckResult(
            "billing formula version", False, "Cannot read cost_tracking.py"
        )
    version = _extract_string_constant(source, "BILLING_FORMULA_VERSION")
    if version == EXPECTED_BILLING_FORMULA_VERSION:
        return CheckResult("billing formula version", True, version)
    return CheckResult(
        "billing formula version",
        False,
        f"Expected {EXPECTED_BILLING_FORMULA_VERSION!r}, found {version!r}",
    )


def check_billing_formula_locked(root: Path) -> CheckResult:
    """Verify BillingFormula has locked=True default."""
    source = _read_file(root / "cost_tracking.py")
    if source is None:
        return CheckResult(
            "billing formula locked", False, "Cannot read cost_tracking.py"
        )
    if re.search(r"locked:\s*bool\s*=\s*True", source):
        return CheckResult("billing formula locked", True, "locked=True")
    return CheckResult(
        "billing formula locked",
        False,
        "Could not confirm locked=True in BillingFormula",
    )


def check_sla_monitoring_exists(root: Path) -> CheckResult:
    """Verify sla_monitoring.py exists."""
    path = root / "sla_monitoring.py"
    if path.is_file():
        return CheckResult("sla_monitoring.py exists", True)
    return CheckResult("sla_monitoring.py exists", False, f"Not found at {path}")


def check_sla_defaults(root: Path) -> list[CheckResult]:
    """Verify default SLA targets in sla_monitoring.py match policy."""
    results = []
    source = _read_file(root / "sla_monitoring.py")
    if source is None:
        results.append(
            CheckResult(
                "sla_monitoring.py readable", False, "Cannot read file"
            )
        )
        return results

    checks = [
        ("DEFAULT_AVAILABILITY_TARGET", EXPECTED_AVAILABILITY_TARGET),
        ("DEFAULT_THROUGHPUT_TARGET", EXPECTED_THROUGHPUT_TARGET),
        ("DEFAULT_ERROR_RATE_BUDGET", EXPECTED_ERROR_RATE_BUDGET),
        ("DEFAULT_P95_LATENCY_TARGET", EXPECTED_P95_LATENCY_TARGET),
        ("DEFAULT_RECOVERY_TIME_TARGET", EXPECTED_RECOVERY_TIME_TARGET),
    ]
    for name, expected in checks:
        actual = _extract_float_constant(source, name)
        if actual is None:
            results.append(
                CheckResult(
                    f"SLA default {name}",
                    False,
                    f"Could not extract {name} from source",
                )
            )
        elif abs(actual - expected) < 1e-9:
            results.append(
                CheckResult(f"SLA default {name}", True, f"{actual}")
            )
        else:
            results.append(
                CheckResult(
                    f"SLA default {name}",
                    False,
                    f"Expected {expected}, found {actual}",
                )
            )
    return results


def check_sla_formula_version(root: Path) -> CheckResult:
    """Verify SLA_FORMULA_VERSION matches policy."""
    source = _read_file(root / "sla_monitoring.py")
    if source is None:
        return CheckResult(
            "SLA formula version", False, "Cannot read sla_monitoring.py"
        )
    version = _extract_string_constant(source, "SLA_FORMULA_VERSION")
    if version == EXPECTED_SLA_FORMULA_VERSION:
        return CheckResult("SLA formula version", True, version)
    return CheckResult(
        "SLA formula version",
        False,
        f"Expected {EXPECTED_SLA_FORMULA_VERSION!r}, found {version!r}",
    )


def check_sla_window_default(root: Path) -> CheckResult:
    """Verify MetricsWindow default window is 3600 seconds (1 hour)."""
    source = _read_file(root / "sla_monitoring.py")
    if source is None:
        return CheckResult(
            "SLA window default", False, "Cannot read sla_monitoring.py"
        )
    # Look for MetricsWindow.__init__ default: window_seconds: int = 3600
    m = re.search(r"window_seconds[:\s]*int\s*=\s*(\d+)", source)
    if m and int(m.group(1)) == 3600:
        return CheckResult("SLA window default", True, "3600 seconds (1 hour)")
    # Also accept window_seconds=3600 in the constructor
    m2 = re.search(r"MetricsWindow\((\d+)\)", source)
    if m2 and int(m2.group(1)) == 3600:
        return CheckResult("SLA window default", True, "3600 seconds (1 hour)")
    return CheckResult(
        "SLA window default",
        False,
        "Could not confirm default window_seconds=3600",
    )


def check_prometheus_metrics(root: Path) -> list[CheckResult]:
    """Verify expected Prometheus metrics exist in api/prometheus.py."""
    results = []
    source = _read_file(root / "api" / "prometheus.py")
    if source is None:
        results.append(
            CheckResult(
                "api/prometheus.py readable", False, "Cannot read file"
            )
        )
        return results

    for metric_name in EXPECTED_PROMETHEUS_METRICS:
        # Look for the metric name in Gauge() or similar constructor
        if metric_name in source:
            results.append(
                CheckResult(f"Prometheus metric {metric_name}", True)
            )
        else:
            results.append(
                CheckResult(
                    f"Prometheus metric {metric_name}",
                    False,
                    f"{metric_name!r} not found in api/prometheus.py",
                )
            )
    return results


def check_grafana_panels(root: Path) -> list[CheckResult]:
    """Verify expected cost/SLA panels exist in the Grafana dashboard."""
    results = []
    dashboard_path = (
        root
        / "helm"
        / "ocr-local"
        / "templates"
        / "grafana-dashboard-configmap.yaml"
    )
    source = _read_file(dashboard_path)
    if source is None:
        results.append(
            CheckResult(
                "Grafana dashboard readable",
                False,
                f"Cannot read {dashboard_path}",
            )
        )
        return results

    for panel_title in EXPECTED_GRAFANA_PANELS:
        if panel_title in source:
            results.append(
                CheckResult(f"Grafana panel '{panel_title}'", True)
            )
        else:
            results.append(
                CheckResult(
                    f"Grafana panel '{panel_title}'",
                    False,
                    "Panel title not found in dashboard",
                )
            )
    return results


def check_policy_document(root: Path) -> CheckResult:
    """Verify the policy document exists."""
    path = root / "docs" / "operations" / "billing-sla-policy.md"
    if path.is_file():
        return CheckResult("billing-sla-policy.md exists", True)
    return CheckResult(
        "billing-sla-policy.md exists", False, f"Not found at {path}"
    )


def check_tenant_override_mechanism(root: Path) -> CheckResult:
    """Verify set_tenant_slos exists in sla_monitoring.py for per-tenant overrides."""
    source = _read_file(root / "sla_monitoring.py")
    if source is None:
        return CheckResult(
            "tenant SLO override mechanism",
            False,
            "Cannot read sla_monitoring.py",
        )
    if "def set_tenant_slos" in source:
        return CheckResult(
            "tenant SLO override mechanism",
            True,
            "set_tenant_slos() found",
        )
    return CheckResult(
        "tenant SLO override mechanism",
        False,
        "set_tenant_slos() not found",
    )


def check_cost_persistence(root: Path) -> CheckResult:
    """Verify CostTracker supports persistence (persist_path parameter)."""
    source = _read_file(root / "cost_tracking.py")
    if source is None:
        return CheckResult(
            "cost persistence mechanism",
            False,
            "Cannot read cost_tracking.py",
        )
    if "persist_path" in source and "def persist" in source:
        return CheckResult(
            "cost persistence mechanism",
            True,
            "persist_path + persist() found",
        )
    return CheckResult(
        "cost persistence mechanism",
        False,
        "persist_path or persist() not found",
    )


def check_sla_report_export(root: Path) -> CheckResult:
    """Verify SLAMonitor has write_report_json for report export."""
    source = _read_file(root / "sla_monitoring.py")
    if source is None:
        return CheckResult(
            "SLA report export",
            False,
            "Cannot read sla_monitoring.py",
        )
    if "def write_report_json" in source:
        return CheckResult(
            "SLA report export", True, "write_report_json() found"
        )
    return CheckResult(
        "SLA report export",
        False,
        "write_report_json() not found",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_validation(root: Path) -> ValidationReport:
    """Execute all validation checks and return a report."""
    report = ValidationReport(project_root=str(root))

    # Cost tracking
    report.add(check_cost_tracking_exists(root))
    for r in check_cost_defaults(root):
        report.add(r)
    report.add(check_billing_formula_version(root))
    report.add(check_billing_formula_locked(root))
    report.add(check_cost_persistence(root))

    # SLA monitoring
    report.add(check_sla_monitoring_exists(root))
    for r in check_sla_defaults(root):
        report.add(r)
    report.add(check_sla_formula_version(root))
    report.add(check_sla_window_default(root))
    report.add(check_tenant_override_mechanism(root))
    report.add(check_sla_report_export(root))

    # Prometheus bridge
    for r in check_prometheus_metrics(root):
        report.add(r)

    # Grafana dashboard
    for r in check_grafana_panels(root):
        report.add(r)

    # Policy document
    report.add(check_policy_document(root))

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_project_root(arg: str | None) -> Path:
    """Resolve project root, defaulting to the repo root."""
    if arg:
        return Path(arg).resolve()
    # Walk up from script location to find the repo root
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "cost_tracking.py").is_file():
        return candidate
    # Fallback to cwd
    return Path.cwd()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate billing and SLA policy alignment"
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Path to project root (default: auto-detect)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write report to file (text or JSON based on --json flag)",
    )
    args = parser.parse_args(argv)

    root = _resolve_project_root(args.project_root)
    report = run_validation(root)

    output = report.to_json() if args.json else report.to_text()

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(output, encoding="utf-8")
        print(f"Report written to {report_path}")
    else:
        print(output)

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
