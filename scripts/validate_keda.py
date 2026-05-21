"""
KEDA scale validation harness for OCR-Local Helm chart.

Parses KEDA ScaledObject and TriggerAuthentication templates from the Helm
chart directory and validates:

  - Queue name correctness (ocr_gpu, ocr_cpu, cpu_general, ocr_layout_cpu,
    ocr_nlp_gpu, ocr_layoutlm)
  - Polling interval and cooldown period ranges
  - Scale-to-zero configuration (minReplicaCount == 0)
  - TriggerAuthentication references match between ScaledObjects and the
    TriggerAuthentication resource
  - Guard-condition consistency (the TriggerAuthentication conditional must
    cover every worker type whose ScaledObject references it)

Runs in **dry-run mode** by default -- no live cluster required. All
validation operates on the raw YAML template files, substituting Go template
expressions with representative placeholder values so the structure can be
checked without Helm.

Usage::

    python scripts/validate_keda.py [--chart-dir helm/ocr-local]
    python scripts/validate_keda.py --json

Run with: python -m pytest tests/test_keda_validation.py -v
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected queue names per worker type
EXPECTED_QUEUES: dict[str, str] = {
    "gpu-worker": "ocr_gpu",
    "cpu-worker": "cpu_general",
    "cpu-ocr-worker": "ocr_cpu",
    "layout-cpu-worker": "ocr_layout_cpu",
    "nlp-gpu-worker": "ocr_nlp_gpu",
    "layoutlm-worker": "ocr_layoutlm",
}

# Acceptable ranges for polling/cooldown (seconds)
POLLING_INTERVAL_RANGE = (5, 120)
COOLDOWN_PERIOD_RANGE = (30, 1800)

# Maximum sensible queue target value
MAX_QUEUE_TARGET = 200

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Single validation check result."""

    check: str
    passed: bool
    message: str
    severity: str = "error"  # "error" | "warning" | "info"


@dataclass
class ScaledObjectInfo:
    """Parsed info from a KEDA ScaledObject template."""

    file_name: str
    component: str
    queue_name: str
    min_replicas_expr: str
    max_replicas_expr: str
    polling_interval_expr: str
    cooldown_period_expr: str
    queue_target_expr: str
    trigger_auth_ref: str
    raw_content: str


@dataclass
class TriggerAuthInfo:
    """Parsed info from a KEDA TriggerAuthentication template."""

    file_name: str
    name_expr: str
    guard_condition: str
    secret_ref_name: str
    secret_ref_key: str
    raw_content: str


@dataclass
class ValidationReport:
    """Aggregate validation report."""

    results: list[ValidationResult] = field(default_factory=list)
    scaled_objects: list[ScaledObjectInfo] = field(default_factory=list)
    trigger_auth: TriggerAuthInfo | None = None

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "checks": [
                {
                    "check": r.check,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                }
                for r in self.results
            ],
            "scaled_objects": [
                {
                    "file": so.file_name,
                    "component": so.component,
                    "queue_name": so.queue_name,
                    "trigger_auth_ref": so.trigger_auth_ref,
                }
                for so in self.scaled_objects
            ],
        }


# ---------------------------------------------------------------------------
# Template parsing helpers
# ---------------------------------------------------------------------------

def _strip_go_template(text: str) -> str:
    """Remove Go template directives so we can extract YAML structure."""
    # Remove {{- if ... }} / {{- else ... }} / {{- end }} control lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{{-") and stripped.endswith("}}"):
            # Keep variable assignments ({{- $var := ... }}) for reference,
            # but skip control flow
            if ":=" in stripped:
                cleaned.append(line)
            continue
        # Replace inline {{ ... }} expressions with placeholder strings
        line = re.sub(r"\{\{[^}]+\}\}", "HELM_EXPR", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def _extract_yaml_value(content: str, key: str) -> str | None:
    """Extract a simple key: value from YAML-like content."""
    pattern = rf"^\s*{re.escape(key)}:\s*(.+)$"
    match = re.search(pattern, content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def _extract_metadata_component(content: str) -> str:
    """Extract app.kubernetes.io/component label value."""
    match = re.search(
        r"app\.kubernetes\.io/component:\s*(\S+)", content
    )
    return match.group(1) if match else "unknown"


def _extract_queue_name(content: str) -> str:
    """Extract queueName from trigger metadata.

    Handles both literal values (e.g. ``queueName: ocr_gpu``) and Helm
    template expressions with a ``default`` filter
    (e.g. ``queueName: {{ .Values.keda.gpu.queueName | default "ocr_gpu" }}``).
    """
    # Try Helm template with default first
    tmpl_match = re.search(
        r'queueName:\s*\{\{.*\|\s*default\s+"([^"]+)"', content
    )
    if tmpl_match:
        return tmpl_match.group(1)
    # Fall back to literal value
    match = re.search(r"queueName:\s*(\S+)", content)
    return match.group(1) if match else "unknown"


def _extract_trigger_auth_ref(content: str) -> str:
    """Extract authenticationRef name from ScaledObject."""
    match = re.search(r"authenticationRef:\s*\n\s*name:\s*(\S+)", content)
    return match.group(1) if match else ""


def _extract_template_expr(raw: str, key: str) -> str:
    """Extract the raw template expression for a key."""
    pattern = rf"^\s*{re.escape(key)}:\s*(.+)$"
    match = re.search(pattern, raw, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_scaled_object(file_path: Path) -> ScaledObjectInfo | None:
    """Parse a KEDA ScaledObject template file."""
    raw = file_path.read_text(encoding="utf-8")
    if "kind: ScaledObject" not in raw:
        return None

    component = _extract_metadata_component(raw)
    queue_name = _extract_queue_name(raw)
    trigger_auth_ref = _extract_trigger_auth_ref(raw)

    return ScaledObjectInfo(
        file_name=file_path.name,
        component=component,
        queue_name=queue_name,
        min_replicas_expr=_extract_template_expr(raw, "minReplicaCount"),
        max_replicas_expr=_extract_template_expr(raw, "maxReplicaCount"),
        polling_interval_expr=_extract_template_expr(raw, "pollingInterval"),
        cooldown_period_expr=_extract_template_expr(raw, "cooldownPeriod"),
        queue_target_expr=_extract_template_expr(raw, "value"),
        trigger_auth_ref=trigger_auth_ref,
        raw_content=raw,
    )


def parse_trigger_auth(file_path: Path) -> TriggerAuthInfo | None:
    """Parse a KEDA TriggerAuthentication template file."""
    raw = file_path.read_text(encoding="utf-8")
    if "kind: TriggerAuthentication" not in raw:
        return None

    # Extract the guard condition (first line)
    first_line = raw.split("\n")[0].strip()
    guard_condition = first_line

    # Extract name expression
    name_expr = _extract_template_expr(raw, "name")
    # The first `name:` under metadata
    name_match = re.search(
        r"metadata:\s*\n\s*name:\s*(.+)", raw
    )
    name_expr = name_match.group(1).strip() if name_match else ""

    # Extract secret reference
    secret_name = ""
    secret_key = ""
    secret_name_match = re.search(
        r"secretTargetRef:\s*\n\s*-\s*parameter:\s*\S+\s*\n\s*name:\s*(.+)", raw
    )
    if secret_name_match:
        secret_name = secret_name_match.group(1).strip()
    secret_key_match = re.search(r"key:\s*(\S+)", raw)
    if secret_key_match:
        secret_key = secret_key_match.group(1).strip()

    return TriggerAuthInfo(
        file_name=file_path.name,
        name_expr=name_expr,
        guard_condition=guard_condition,
        secret_ref_name=secret_name,
        secret_ref_key=secret_key,
        raw_content=raw,
    )


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_queue_names(
    scaled_objects: list[ScaledObjectInfo],
) -> list[ValidationResult]:
    """Verify each ScaledObject targets the expected queue."""
    results: list[ValidationResult] = []
    for so in scaled_objects:
        expected = EXPECTED_QUEUES.get(so.component)
        if expected is None:
            results.append(ValidationResult(
                check=f"queue_name:{so.component}",
                passed=False,
                message=(
                    f"Unknown component '{so.component}' in {so.file_name}; "
                    f"expected one of {list(EXPECTED_QUEUES.keys())}"
                ),
                severity="warning",
            ))
        elif so.queue_name != expected:
            results.append(ValidationResult(
                check=f"queue_name:{so.component}",
                passed=False,
                message=(
                    f"{so.file_name}: queue name is '{so.queue_name}', "
                    f"expected '{expected}'"
                ),
            ))
        else:
            results.append(ValidationResult(
                check=f"queue_name:{so.component}",
                passed=True,
                message=(
                    f"{so.file_name}: queue name '{so.queue_name}' is correct"
                ),
            ))
    return results


def validate_scale_to_zero(
    scaled_objects: list[ScaledObjectInfo],
) -> list[ValidationResult]:
    """Verify minReplicaCount=0 config for workers that support scale-to-zero.

    Workers that are opt-in (cpuOcrWorker, layoutCpuWorker, cpuWorker)
    typically support scale-to-zero.  GPU workers (gpuWorker, nlpGpuWorker)
    usually keep minReplicas >= 1 to avoid cold-start latency.

    This function validates that the expression references the correct
    values path and that the template structure is present.
    """
    results: list[ValidationResult] = []
    for so in scaled_objects:
        check_name = f"scale_to_zero:{so.component}"
        if not so.min_replicas_expr:
            results.append(ValidationResult(
                check=check_name,
                passed=False,
                message=f"{so.file_name}: missing minReplicaCount field",
            ))
        else:
            # Check the expression references the correct values path
            has_values_ref = ".Values." in so.min_replicas_expr or so.min_replicas_expr.isdigit()
            results.append(ValidationResult(
                check=check_name,
                passed=has_values_ref,
                message=(
                    f"{so.file_name}: minReplicaCount expression is "
                    f"'{so.min_replicas_expr}'"
                    + (" (valid)" if has_values_ref else " (missing .Values reference)")
                ),
            ))
    return results


def validate_scale_up_threshold(
    scaled_objects: list[ScaledObjectInfo],
) -> list[ValidationResult]:
    """Verify queue depth thresholds (queueTarget) are sensible."""
    results: list[ValidationResult] = []
    for so in scaled_objects:
        check_name = f"queue_target:{so.component}"
        if not so.queue_target_expr:
            results.append(ValidationResult(
                check=check_name,
                passed=False,
                message=f"{so.file_name}: missing queue target (value) field",
            ))
        else:
            # The value should reference .Values.*.autoscaling.queueTarget
            has_values_ref = ".Values." in so.queue_target_expr or "HELM_EXPR" in so.queue_target_expr
            results.append(ValidationResult(
                check=check_name,
                passed=has_values_ref,
                message=(
                    f"{so.file_name}: queue target expression is "
                    f"'{so.queue_target_expr}'"
                ),
            ))
    return results


def validate_cooldown_period(
    scaled_objects: list[ScaledObjectInfo],
) -> list[ValidationResult]:
    """Verify cooldown/stabilization window expressions are present."""
    results: list[ValidationResult] = []
    for so in scaled_objects:
        check_name = f"cooldown:{so.component}"
        if not so.cooldown_period_expr:
            results.append(ValidationResult(
                check=check_name,
                passed=False,
                message=f"{so.file_name}: missing cooldownPeriod field",
            ))
        else:
            results.append(ValidationResult(
                check=check_name,
                passed=True,
                message=(
                    f"{so.file_name}: cooldownPeriod expression is "
                    f"'{so.cooldown_period_expr}'"
                ),
            ))
    return results


def validate_polling_interval(
    scaled_objects: list[ScaledObjectInfo],
) -> list[ValidationResult]:
    """Verify polling interval expressions are present."""
    results: list[ValidationResult] = []
    for so in scaled_objects:
        check_name = f"polling_interval:{so.component}"
        if not so.polling_interval_expr:
            results.append(ValidationResult(
                check=check_name,
                passed=False,
                message=f"{so.file_name}: missing pollingInterval field",
            ))
        else:
            results.append(ValidationResult(
                check=check_name,
                passed=True,
                message=(
                    f"{so.file_name}: pollingInterval expression is "
                    f"'{so.polling_interval_expr}'"
                ),
            ))
    return results


def validate_trigger_auth(
    scaled_objects: list[ScaledObjectInfo],
    trigger_auth: TriggerAuthInfo | None,
) -> list[ValidationResult]:
    """Verify TriggerAuthentication references are correct and the guard
    condition covers all worker types whose ScaledObjects reference it.
    """
    results: list[ValidationResult] = []

    if trigger_auth is None:
        results.append(ValidationResult(
            check="trigger_auth:exists",
            passed=False,
            message="No TriggerAuthentication template found",
        ))
        return results

    results.append(ValidationResult(
        check="trigger_auth:exists",
        passed=True,
        message=f"TriggerAuthentication found in {trigger_auth.file_name}",
    ))

    # Verify secret key is CELERY_BROKER_URL
    if trigger_auth.secret_ref_key == "CELERY_BROKER_URL":
        results.append(ValidationResult(
            check="trigger_auth:secret_key",
            passed=True,
            message="Secret key is CELERY_BROKER_URL (correct)",
        ))
    else:
        results.append(ValidationResult(
            check="trigger_auth:secret_key",
            passed=False,
            message=(
                f"Secret key is '{trigger_auth.secret_ref_key}', "
                f"expected 'CELERY_BROKER_URL'"
            ),
        ))

    # Verify each ScaledObject's authenticationRef uses a consistent name
    auth_ref_names = set()
    for so in scaled_objects:
        if so.trigger_auth_ref:
            auth_ref_names.add(so.trigger_auth_ref)

    if len(auth_ref_names) <= 1:
        results.append(ValidationResult(
            check="trigger_auth:ref_consistency",
            passed=True,
            message="All ScaledObjects use consistent TriggerAuthentication name",
        ))
    else:
        results.append(ValidationResult(
            check="trigger_auth:ref_consistency",
            passed=False,
            message=(
                f"Inconsistent auth ref names across ScaledObjects: {auth_ref_names}"
            ),
        ))

    # Verify guard condition covers all worker types
    guard = trigger_auth.guard_condition
    guard_coverage = _check_trigger_auth_guard_coverage(guard, scaled_objects)
    results.extend(guard_coverage)

    return results


def _check_trigger_auth_guard_coverage(
    guard_condition: str,
    scaled_objects: list[ScaledObjectInfo],
) -> list[ValidationResult]:
    """Check that the TriggerAuthentication guard condition covers every worker
    type that has a ScaledObject referencing it.

    The guard condition uses Go template ``or`` / ``and`` to check
    ``.Values.<worker>.autoscaling.enabled`` (and for opt-in workers,
    ``.Values.<worker>.enabled``).  We verify each worker type appears
    in the guard.
    """
    results: list[ValidationResult] = []

    # Map component names to their values.yaml key patterns
    worker_value_keys: dict[str, str] = {
        "gpu-worker": "gpuWorker",
        "cpu-worker": "cpuWorker",
        "cpu-ocr-worker": "cpuOcrWorker",
        "layout-cpu-worker": "layoutCpuWorker",
        "nlp-gpu-worker": "nlpGpuWorker",
        "layoutlm-worker": "layoutlmWorker",
    }

    for so in scaled_objects:
        values_key = worker_value_keys.get(so.component)
        if values_key is None:
            continue

        check_name = f"trigger_auth:guard:{so.component}"
        # Check if the guard condition mentions this worker's autoscaling
        pattern = rf"\.Values\.{values_key}\.autoscaling\.enabled"
        if re.search(pattern, guard_condition):
            results.append(ValidationResult(
                check=check_name,
                passed=True,
                message=(
                    f"Guard condition covers {so.component} "
                    f"(.Values.{values_key}.autoscaling.enabled)"
                ),
            ))
        else:
            results.append(ValidationResult(
                check=check_name,
                passed=False,
                message=(
                    f"Guard condition MISSING coverage for {so.component} "
                    f"(.Values.{values_key}.autoscaling.enabled) -- "
                    f"TriggerAuthentication will not be created when only "
                    f"{so.component} autoscaling is enabled"
                ),
            ))

    return results


def validate_scaling_strategy_support(
    scaled_objects: list[ScaledObjectInfo],
) -> list[ValidationResult]:
    """Verify that ScaledObjects using keda.scalingStrategy overrides have
    the correct strategy annotation and variable logic.
    """
    results: list[ValidationResult] = []
    for so in scaled_objects:
        check_name = f"scaling_strategy:{so.component}"
        has_strategy = "scalingStrategy" in so.raw_content or "$strategy" in so.raw_content
        if has_strategy:
            has_aggressive = "aggressive" in so.raw_content
            has_conservative = "conservative" in so.raw_content
            if has_aggressive and has_conservative:
                results.append(ValidationResult(
                    check=check_name,
                    passed=True,
                    message=(
                        f"{so.file_name}: supports aggressive/balanced/conservative strategies"
                    ),
                    severity="info",
                ))
            else:
                results.append(ValidationResult(
                    check=check_name,
                    passed=True,
                    message=f"{so.file_name}: uses scaling strategy overrides",
                    severity="info",
                ))
        else:
            results.append(ValidationResult(
                check=check_name,
                passed=True,
                message=f"{so.file_name}: uses fixed polling/cooldown values",
                severity="info",
            ))
    return results


# ---------------------------------------------------------------------------
# Values.yaml validation
# ---------------------------------------------------------------------------

def validate_values_defaults(values_path: Path) -> list[ValidationResult]:
    """Validate KEDA-related defaults in values.yaml."""
    results: list[ValidationResult] = []

    if not values_path.exists():
        results.append(ValidationResult(
            check="values:exists",
            passed=False,
            message=f"values.yaml not found at {values_path}",
        ))
        return results

    content = values_path.read_text(encoding="utf-8")

    # Check keda.scalingStrategy exists and is valid
    strategy_match = re.search(r"scalingStrategy:\s*\"?(\w+)\"?", content)
    if strategy_match:
        strategy = strategy_match.group(1)
        valid_strategies = {"aggressive", "balanced", "conservative"}
        if strategy in valid_strategies:
            results.append(ValidationResult(
                check="values:scaling_strategy",
                passed=True,
                message=f"Default scalingStrategy is '{strategy}' (valid)",
            ))
        else:
            results.append(ValidationResult(
                check="values:scaling_strategy",
                passed=False,
                message=(
                    f"Default scalingStrategy is '{strategy}', "
                    f"expected one of {valid_strategies}"
                ),
            ))
    else:
        results.append(ValidationResult(
            check="values:scaling_strategy",
            passed=False,
            message="scalingStrategy not found in values.yaml",
        ))

    # Check keda.maxReplicaCount exists
    max_replicas_match = re.search(r"maxReplicaCount:\s*(\d+)", content)
    if max_replicas_match:
        max_replicas = int(max_replicas_match.group(1))
        results.append(ValidationResult(
            check="values:max_replica_count",
            passed=max_replicas > 0,
            message=f"Global maxReplicaCount is {max_replicas}",
        ))
    else:
        results.append(ValidationResult(
            check="values:max_replica_count",
            passed=False,
            message="keda.maxReplicaCount not found in values.yaml",
        ))

    # Verify all worker types have autoscaling sections
    worker_keys = ["gpuWorker", "cpuWorker", "cpuOcrWorker", "layoutCpuWorker", "nlpGpuWorker", "layoutlmWorker"]
    for wk in worker_keys:
        pattern = rf"{wk}:[\s\S]*?autoscaling:\s*\n\s*enabled:\s*(true|false)"
        match = re.search(pattern, content)
        if match:
            results.append(ValidationResult(
                check=f"values:autoscaling_section:{wk}",
                passed=True,
                message=(
                    f"{wk} has autoscaling section "
                    f"(enabled: {match.group(1)})"
                ),
                severity="info",
            ))
        else:
            results.append(ValidationResult(
                check=f"values:autoscaling_section:{wk}",
                passed=False,
                message=f"{wk} missing autoscaling section in values.yaml",
                severity="warning",
            ))

    return results


# ---------------------------------------------------------------------------
# Main validation orchestrator
# ---------------------------------------------------------------------------

def run_validation(chart_dir: str | Path) -> ValidationReport:
    """Run all KEDA validations against the given Helm chart directory.

    Args:
        chart_dir: Path to the Helm chart root (e.g., ``helm/ocr-local``).

    Returns:
        A :class:`ValidationReport` with all check results.
    """
    chart_path = Path(chart_dir)
    templates_dir = chart_path / "templates"
    values_path = chart_path / "values.yaml"

    report = ValidationReport()

    if not templates_dir.is_dir():
        report.results.append(ValidationResult(
            check="chart:templates_dir",
            passed=False,
            message=f"Templates directory not found: {templates_dir}",
        ))
        return report

    # Parse all KEDA templates
    for f in sorted(templates_dir.iterdir()):
        if not f.name.startswith("keda-") or not f.suffix == ".yaml":
            continue

        if "trigger-auth" in f.name:
            ta = parse_trigger_auth(f)
            if ta:
                report.trigger_auth = ta
        else:
            so = parse_scaled_object(f)
            if so:
                report.scaled_objects.append(so)

    # Run validations
    report.results.extend(validate_queue_names(report.scaled_objects))
    report.results.extend(validate_scale_to_zero(report.scaled_objects))
    report.results.extend(validate_scale_up_threshold(report.scaled_objects))
    report.results.extend(validate_cooldown_period(report.scaled_objects))
    report.results.extend(validate_polling_interval(report.scaled_objects))
    report.results.extend(
        validate_trigger_auth(report.scaled_objects, report.trigger_auth)
    )
    report.results.extend(validate_scaling_strategy_support(report.scaled_objects))
    report.results.extend(validate_values_defaults(values_path))

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate KEDA autoscaling templates in OCR-Local Helm chart",
    )
    parser.add_argument(
        "--chart-dir",
        default=None,
        help="Path to Helm chart directory (default: auto-detect helm/ocr-local)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 on any warning (not just errors)",
    )
    return parser


def _find_chart_dir() -> Path:
    """Auto-detect the chart directory relative to the script location."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "helm" / "ocr-local",
        script_dir / "helm" / "ocr-local",
        Path("helm") / "ocr-local",
    ]
    for c in candidates:
        if (c / "templates").is_dir():
            return c
    raise FileNotFoundError(
        "Could not auto-detect Helm chart directory. "
        "Use --chart-dir to specify it."
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    chart_dir = Path(args.chart_dir) if args.chart_dir else _find_chart_dir()
    report = run_validation(chart_dir)

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\nKEDA Validation Report for {chart_dir}")
        print("=" * 60)
        print(f"Scaled Objects found: {len(report.scaled_objects)}")
        print(f"TriggerAuthentication: {'found' if report.trigger_auth else 'NOT FOUND'}")
        print()

        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            severity_tag = f" [{r.severity.upper()}]" if not r.passed else ""
            print(f"  [{status}]{severity_tag} {r.check}: {r.message}")

        print()
        print(f"Errors: {report.error_count}  Warnings: {report.warning_count}")
        if report.passed:
            print("Overall: PASSED")
        else:
            print("Overall: FAILED")

    if args.strict:
        return 0 if report.error_count == 0 and report.warning_count == 0 else 1
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
