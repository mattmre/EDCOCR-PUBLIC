#!/usr/bin/env python3
"""Evaluate accumulated baseline data and produce health assessment.

Reads JSONL output from production_baseline.py, computes statistical
baselines, identifies anomalies, and outputs a production-readiness report.

Usage:
    python scripts/evaluate_baseline.py --input baseline_data/baseline.jsonl
    python scripts/evaluate_baseline.py --input data.jsonl --output-dir reports/
    python scripts/evaluate_baseline.py --input data.jsonl --format both
    python scripts/evaluate_baseline.py --input data.jsonl --anomaly-threshold 3.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse helpers from production_baseline if available; fall back to inline.
try:
    from scripts.production_baseline import (
        compute_series_stats,
        extract_time_series,
        read_snapshots,
    )
except ImportError:
    # Standalone execution -- define locally.
    def read_snapshots(filepath: Path) -> list[dict]:
        snapshots = []
        if not filepath.exists():
            return snapshots
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        snapshots.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return snapshots

    def _percentile(sorted_values: list[float], pct: float) -> float:
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * (pct / 100)
        f = int(k)
        c = f + 1
        if c >= len(sorted_values):
            return sorted_values[-1]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def _stddev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    def compute_series_stats(values: list[float]) -> dict:
        if not values:
            return {
                "count": 0, "mean": 0.0, "stddev": 0.0,
                "min": 0.0, "max": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0,
            }
        sorted_vals = sorted(values)
        return {
            "count": len(values),
            "mean": round(sum(values) / len(values), 4),
            "stddev": round(_stddev(values), 4),
            "min": round(sorted_vals[0], 4),
            "max": round(sorted_vals[-1], 4),
            "p50": round(_percentile(sorted_vals, 50), 4),
            "p95": round(_percentile(sorted_vals, 95), 4),
            "p99": round(_percentile(sorted_vals, 99), 4),
        }

    def _append_if_numeric(lst, value):
        if value is not None and isinstance(value, (int, float)):
            lst.append(float(value))

    def extract_time_series(snapshots: list[dict]) -> dict[str, list[float]]:
        series = {
            "error_rate": [], "completion_rate": [], "s3_error_rate": [],
            "pages_processed": [], "avg_processing_time_ms": [],
            "gpu_workers_available": [], "workers_online": [],
            "workers_busy": [], "stuck_jobs": [],
            "pages_per_minute": [], "docs_per_hour": [],
        }
        for snap in snapshots:
            coord = snap.get("sources", {}).get("coordinator", {})
            dash = snap.get("sources", {}).get("api_dashboard", {})
            jobs = coord.get("jobs", {})
            workers = coord.get("workers", {})
            pages = coord.get("pages", {})
            _append_if_numeric(series["error_rate"], jobs.get("error_rate_1h"))
            _append_if_numeric(series["completion_rate"], jobs.get("completion_rate_1h"))
            _append_if_numeric(series["s3_error_rate"], jobs.get("s3_error_rate_1h"))
            _append_if_numeric(series["pages_processed"], pages.get("total_processed"))
            _append_if_numeric(series["avg_processing_time_ms"], pages.get("avg_processing_time_ms"))
            _append_if_numeric(series["gpu_workers_available"], workers.get("gpu_available"))
            _append_if_numeric(series["stuck_jobs"], jobs.get("stuck_total"))
            by_status = workers.get("by_status", {})
            _append_if_numeric(series["workers_online"], by_status.get("online", 0))
            _append_if_numeric(series["workers_busy"], by_status.get("busy", 0))
            throughput = dash.get("throughput", {})
            _append_if_numeric(series["pages_per_minute"], throughput.get("pages_per_minute"))
            _append_if_numeric(series["docs_per_hour"], throughput.get("docs_per_hour"))
        return series


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    values: list[float],
    threshold_sigma: float = 2.0,
) -> list[dict]:
    """Detect values that exceed threshold_sigma standard deviations from mean.

    Returns list of anomaly dicts with index, value, z_score, and direction.
    """
    if len(values) < 3:
        return []

    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))

    if std == 0:
        return []

    anomalies = []
    for i, v in enumerate(values):
        z = (v - mean) / std
        if abs(z) > threshold_sigma:
            anomalies.append({
                "index": i,
                "value": round(v, 4),
                "z_score": round(z, 4),
                "direction": "high" if z > 0 else "low",
            })

    return anomalies


# ---------------------------------------------------------------------------
# Health assessment
# ---------------------------------------------------------------------------

# Baseline health thresholds (production defaults)
DEFAULT_THRESHOLDS = {
    "max_avg_error_rate": 0.05,       # 5% average error rate
    "max_peak_error_rate": 0.15,      # 15% peak error rate
    "min_avg_completion_rate": 0.90,   # 90% average completion rate
    "max_avg_stuck_jobs": 1.0,        # Average stuck jobs
    "min_avg_gpu_workers": 1.0,       # At least 1 GPU worker on average
    "max_avg_processing_time_ms": 30000,  # 30s average processing time
}


def assess_health(
    series: dict[str, list[float]],
    thresholds: dict | None = None,
) -> dict:
    """Assess production health against thresholds.

    Returns dict with overall status, per-metric results, and anomalies.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    checks = []
    overall_pass = True

    # Error rate check
    error_stats = compute_series_stats(series.get("error_rate", []))
    if error_stats["count"] > 0:
        avg_pass = error_stats["mean"] <= thresholds["max_avg_error_rate"]
        peak_pass = error_stats["max"] <= thresholds["max_peak_error_rate"]
        passed = avg_pass and peak_pass
        checks.append({
            "metric": "error_rate",
            "passed": passed,
            "details": {
                "avg": error_stats["mean"],
                "max": error_stats["max"],
                "threshold_avg": thresholds["max_avg_error_rate"],
                "threshold_peak": thresholds["max_peak_error_rate"],
            },
        })
        if not passed:
            overall_pass = False

    # Completion rate check
    comp_stats = compute_series_stats(series.get("completion_rate", []))
    if comp_stats["count"] > 0:
        passed = comp_stats["mean"] >= thresholds["min_avg_completion_rate"]
        checks.append({
            "metric": "completion_rate",
            "passed": passed,
            "details": {
                "avg": comp_stats["mean"],
                "threshold": thresholds["min_avg_completion_rate"],
            },
        })
        if not passed:
            overall_pass = False

    # Stuck jobs check
    stuck_stats = compute_series_stats(series.get("stuck_jobs", []))
    if stuck_stats["count"] > 0:
        passed = stuck_stats["mean"] <= thresholds["max_avg_stuck_jobs"]
        checks.append({
            "metric": "stuck_jobs",
            "passed": passed,
            "details": {
                "avg": stuck_stats["mean"],
                "max": stuck_stats["max"],
                "threshold": thresholds["max_avg_stuck_jobs"],
            },
        })
        if not passed:
            overall_pass = False

    # GPU workers check
    gpu_stats = compute_series_stats(series.get("gpu_workers_available", []))
    if gpu_stats["count"] > 0:
        passed = gpu_stats["mean"] >= thresholds["min_avg_gpu_workers"]
        checks.append({
            "metric": "gpu_workers_available",
            "passed": passed,
            "details": {
                "avg": gpu_stats["mean"],
                "min": gpu_stats["min"],
                "threshold": thresholds["min_avg_gpu_workers"],
            },
        })
        if not passed:
            overall_pass = False

    # Processing time check
    time_stats = compute_series_stats(series.get("avg_processing_time_ms", []))
    if time_stats["count"] > 0:
        passed = time_stats["mean"] <= thresholds["max_avg_processing_time_ms"]
        checks.append({
            "metric": "avg_processing_time_ms",
            "passed": passed,
            "details": {
                "avg": time_stats["mean"],
                "p95": time_stats["p95"],
                "threshold": thresholds["max_avg_processing_time_ms"],
            },
        })
        if not passed:
            overall_pass = False

    return {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "checks_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "failed_count": sum(1 for c in checks if not c["passed"]),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(
    snapshots: list[dict],
    anomaly_threshold: float = 2.0,
) -> dict:
    """Full evaluation: stats, anomalies, and health assessment.

    Returns comprehensive evaluation report dict.
    """
    series = extract_time_series(snapshots)

    # Compute stats for each metric
    stats = {}
    for name, values in series.items():
        stats[name] = compute_series_stats(values)

    # Detect anomalies per metric
    anomalies = {}
    for name, values in series.items():
        anom = detect_anomalies(values, anomaly_threshold)
        if anom:
            anomalies[name] = anom

    # Health assessment
    health = assess_health(series)

    timestamps = [s.get("timestamp", 0) for s in snapshots]
    start_ts = min(timestamps) if timestamps else 0
    end_ts = max(timestamps) if timestamps else 0

    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection_period": {
            "start": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat() if start_ts else None,
            "end": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat() if end_ts else None,
            "duration_hours": round((end_ts - start_ts) / 3600, 2) if end_ts > start_ts else 0,
            "snapshot_count": len(snapshots),
        },
        "statistics": stats,
        "anomalies": anomalies,
        "anomaly_threshold_sigma": anomaly_threshold,
        "health_assessment": health,
    }


def generate_evaluation_markdown(evaluation: dict) -> str:
    """Convert evaluation report to markdown."""
    lines = []
    lines.append("# Baseline Evaluation Report")
    lines.append("")

    period = evaluation.get("collection_period", {})
    health = evaluation.get("health_assessment", {})

    lines.append(f"- **Generated**: {evaluation.get('generated_at', 'unknown')}")
    lines.append(f"- **Period**: {period.get('duration_hours', 0)} hours, {period.get('snapshot_count', 0)} snapshots")
    lines.append(f"- **Overall Status**: **{health.get('overall_status', 'UNKNOWN')}**")
    lines.append(f"- **Checks**: {health.get('passed_count', 0)}/{health.get('checks_count', 0)} passed")
    lines.append("")

    # Health checks
    lines.append("## Health Checks")
    lines.append("")
    lines.append("| Metric | Status | Details |")
    lines.append("|--------|--------|---------|")
    for check in health.get("checks", []):
        status = "PASS" if check["passed"] else "FAIL"
        details_str = ", ".join(f"{k}={v}" for k, v in check.get("details", {}).items())
        lines.append(f"| {check['metric']} | {status} | {details_str} |")
    lines.append("")

    # Statistics summary
    lines.append("## Statistics Summary")
    lines.append("")
    stats = evaluation.get("statistics", {})
    lines.append("| Metric | Count | Mean | P50 | P95 | Stddev |")
    lines.append("|--------|-------|------|-----|-----|--------|")
    for name, st in stats.items():
        lines.append(
            f"| {name} | {st.get('count', 0)} | {st.get('mean', 0)} | "
            f"{st.get('p50', 0)} | {st.get('p95', 0)} | {st.get('stddev', 0)} |"
        )
    lines.append("")

    # Anomalies
    anomalies = evaluation.get("anomalies", {})
    if anomalies:
        threshold = evaluation.get("anomaly_threshold_sigma", 2.0)
        lines.append(f"## Anomalies (>{threshold} sigma)")
        lines.append("")
        for name, anom_list in anomalies.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Index | Value | Z-Score | Direction |")
            lines.append("|-------|-------|---------|-----------|")
            for a in anom_list[:20]:  # Limit display
                lines.append(
                    f"| {a['index']} | {a['value']} | {a['z_score']} | {a['direction']} |"
                )
            if len(anom_list) > 20:
                lines.append(f"| ... | ({len(anom_list) - 20} more) | | |")
            lines.append("")
    else:
        lines.append("## Anomalies")
        lines.append("")
        lines.append("No anomalies detected.")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate accumulated baseline data and produce health assessment.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file from production_baseline.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseline_data"),
        help="Output directory for reports (default: baseline_data)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=2.0,
        help="Standard deviation threshold for anomaly detection (default: 2.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    snapshots = read_snapshots(args.input)
    if not snapshots:
        print("ERROR: No valid snapshots in input file", file=sys.stderr)
        return 1

    evaluation = evaluate_baseline(snapshots, args.anomaly_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("json", "both"):
        json_path = args.output_dir / "evaluation_report.json"
        json_path.write_text(
            json.dumps(evaluation, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {json_path}")

    if args.format in ("markdown", "both"):
        md_path = args.output_dir / "evaluation_report.md"
        md_path.write_text(
            generate_evaluation_markdown(evaluation),
            encoding="utf-8",
        )
        print(f"Markdown report: {md_path}")

    # Print summary
    health = evaluation.get("health_assessment", {})
    status = health.get("overall_status", "UNKNOWN")
    passed = health.get("passed_count", 0)
    total = health.get("checks_count", 0)
    anomaly_count = sum(len(v) for v in evaluation.get("anomalies", {}).values())
    print(f"\nHealth: {status} ({passed}/{total} checks passed, {anomaly_count} anomalies)")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
