"""Prometheus recording rule generation for production monitoring.

Generates recording rule YAML for rate smoothing, worker utilization,
and SLA burn-rate calculations. Output is compatible with the Helm
PrometheusRule CRD template.

Usage:
    from coordinator.jobs.prometheus_recording_rules import generate_recording_rules_yaml
    yaml_text = generate_recording_rules_yaml()
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Recording rule definitions
# ---------------------------------------------------------------------------

# Each rule is a dict with:
#   record: the new metric name
#   expr: the PromQL expression
#   labels: optional extra labels dict


def _rate_smoothing_rules() -> list[dict]:
    """Rate smoothing rules for key error and completion metrics.

    Smooths gauge-based rates over multiple windows for alerting stability.
    """
    return [
        # Error rate smoothing (5m, 15m, 1h)
        {
            "record": "ocr:error_rate:avg5m",
            "expr": "avg_over_time(ocr_job_error_rate_1h[5m])",
        },
        {
            "record": "ocr:error_rate:avg15m",
            "expr": "avg_over_time(ocr_job_error_rate_1h[15m])",
        },
        {
            "record": "ocr:error_rate:avg1h",
            "expr": "avg_over_time(ocr_job_error_rate_1h[1h])",
        },
        # Completion rate smoothing
        {
            "record": "ocr:completion_rate:avg5m",
            "expr": "avg_over_time(ocr_job_completion_rate_1h[5m])",
        },
        {
            "record": "ocr:completion_rate:avg15m",
            "expr": "avg_over_time(ocr_job_completion_rate_1h[15m])",
        },
        {
            "record": "ocr:completion_rate:avg1h",
            "expr": "avg_over_time(ocr_job_completion_rate_1h[1h])",
        },
        # S3 error rate smoothing
        {
            "record": "ocr:s3_error_rate:avg5m",
            "expr": "avg_over_time(ocr_s3_job_error_rate_1h[5m])",
        },
        {
            "record": "ocr:s3_error_rate:avg15m",
            "expr": "avg_over_time(ocr_s3_job_error_rate_1h[15m])",
        },
    ]


def _worker_utilization_rules() -> list[dict]:
    """Worker utilization ratio rules."""
    return [
        # Busy workers / total workers
        {
            "record": "ocr:worker_utilization_ratio",
            "expr": (
                'sum(ocr_workers_total{status="busy"}) '
                "/ clamp_min(sum(ocr_workers_total), 1)"
            ),
        },
        # GPU worker availability ratio
        {
            "record": "ocr:gpu_worker_availability_ratio",
            "expr": (
                "ocr_gpu_workers_available "
                "/ clamp_min("
                'sum(ocr_workers_total{status="online"}) '
                '+ sum(ocr_workers_total{status="busy"}), 1)'
            ),
        },
        # Average processing time (smoothed)
        {
            "record": "ocr:processing_time_avg:avg5m",
            "expr": "avg_over_time(ocr_page_processing_time_avg_ms[5m])",
        },
        {
            "record": "ocr:processing_time_avg:avg15m",
            "expr": "avg_over_time(ocr_page_processing_time_avg_ms[15m])",
        },
    ]


def _sla_burn_rate_rules(slo_target: float = 0.99) -> list[dict]:
    """SLA error-budget burn-rate rules.

    Computes how fast the error budget is being consumed relative to
    a target SLO (default 99%).

    burn_rate = actual_error_rate / allowed_error_rate
    where allowed_error_rate = 1 - slo_target

    A burn_rate of 1.0 means consuming budget at exactly the allowed rate.
    A burn_rate of 14.4 over 1h corresponds to exhausting a 30-day budget
    in approximately 2 days (Google SRE multi-window approach).
    """
    allowed_error = round(1 - slo_target, 6)

    return [
        # 1h burn rate
        {
            "record": "ocr:error_budget_burn_rate_1h",
            "expr": (
                f"avg_over_time(ocr_job_error_rate_1h[1h]) / {allowed_error}"
            ),
        },
        # 6h burn rate
        {
            "record": "ocr:error_budget_burn_rate_6h",
            "expr": (
                f"avg_over_time(ocr_job_error_rate_1h[6h]) / {allowed_error}"
            ),
        },
        # 24h burn rate
        {
            "record": "ocr:error_budget_burn_rate_24h",
            "expr": (
                f"avg_over_time(ocr_job_error_rate_1h[24h]) / {allowed_error}"
            ),
        },
        # Error budget consumed (fraction: 0.0 = full budget, 1.0 = exhausted)
        {
            "record": "ocr:error_budget_consumed_1h",
            "expr": (
                f"1 - ((1 - avg_over_time(ocr_job_error_rate_1h[1h])) / (1 - {1 - allowed_error}))"
            ),
        },
    ]


def get_all_recording_rules(slo_target: float = 0.99) -> list[dict]:
    """Return all recording rules as a flat list of dicts."""
    rules = []
    rules.extend(_rate_smoothing_rules())
    rules.extend(_worker_utilization_rules())
    rules.extend(_sla_burn_rate_rules(slo_target))
    return rules


# ---------------------------------------------------------------------------
# YAML generation (compatible with Helm PrometheusRule CRD)
# ---------------------------------------------------------------------------

def _indent(text: str, spaces: int) -> str:
    """Indent each line of text by the given number of spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.split("\n"))


def generate_recording_rules_yaml(
    group_name: str = "ocr-recording-rules",
    interval: str = "30s",
    slo_target: float = 0.99,
) -> str:
    """Generate Prometheus recording rules in YAML format.

    The output is a rules group suitable for embedding in a PrometheusRule
    CRD spec.groups[] array.

    Args:
        group_name: Name for the recording rules group.
        interval: Rule evaluation interval.
        slo_target: SLO target for error budget calculations.

    Returns:
        YAML string for the recording rules group.
    """
    rules = get_all_recording_rules(slo_target)

    lines = []
    lines.append(f"- name: {group_name}")
    lines.append(f"  interval: {interval}")
    lines.append("  rules:")

    for rule in rules:
        lines.append(f"    - record: {rule['record']}")
        # Quote the expression to handle special YAML characters
        expr = rule["expr"]
        lines.append(f'      expr: {expr}')
        if "labels" in rule:
            lines.append("      labels:")
            for k, v in rule["labels"].items():
                lines.append(f"        {k}: {v}")

    return "\n".join(lines) + "\n"


def generate_full_prometheusrule_yaml(
    release_name: str = "ocr-local",
    slo_target: float = 0.99,
) -> str:
    """Generate a complete PrometheusRule CRD YAML document.

    This is useful for standalone deployment outside the Helm chart.
    """
    rules_group = generate_recording_rules_yaml(slo_target=slo_target)

    doc = f"""apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: {release_name}-recording-rules
  labels:
    app.kubernetes.io/name: ocr-local
    app.kubernetes.io/component: monitoring
spec:
  groups:
{_indent(rules_group, 4)}
"""
    return doc
