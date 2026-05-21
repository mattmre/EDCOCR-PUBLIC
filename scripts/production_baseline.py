#!/usr/bin/env python3
"""Production traffic monitoring baseline capture tool.

Captures periodic snapshots from the coordinator metrics endpoint and/or
the API dashboard/analytics endpoints, persists time-series data to JSONL,
and generates baseline summary reports (JSON + markdown) for multi-day
production traffic analysis.

Usage:
    # Continuous capture at 5-minute intervals for 24 hours
    python scripts/production_baseline.py --interval 300 --duration 24

    # Single snapshot
    python scripts/production_baseline.py --snapshot

    # Custom endpoints and output
    python scripts/production_baseline.py \\
        --coordinator-url http://coord:8000/api/v1/metrics/ \\
        --api-url http://api:8080 \\
        --api-key <key> \\
        --output-dir ./baseline_data \\
        --interval 60 --duration 1

    # Generate report from existing JSONL data
    python scripts/production_baseline.py --report-only --input baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COORDINATOR_URL = "http://localhost:8000/api/v1/metrics/"
DEFAULT_API_URL = "http://localhost:8080"
DEFAULT_INTERVAL_SECONDS = 300  # 5 minutes
DEFAULT_DURATION_HOURS = 24
DEFAULT_OUTPUT_DIR = "baseline_data"

SCHEMA_VERSION = "1.0.0"

# Shutdown flag for graceful termination
_shutdown = False


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only -- no new dependencies)
# ---------------------------------------------------------------------------

def _fetch_json(url: str, api_key: str | None = None,
                timeout: int = 15) -> dict | None:
    """Fetch JSON from a URL with optional API key auth.

    Returns parsed dict on success, None on any error.
    """
    req = Request(url)
    if api_key:
        req.add_header("X-Api-Key", api_key)

    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (HTTPError, URLError, json.JSONDecodeError, OSError):
        return None


def _fetch_coordinator_metrics(url: str, api_key: str | None = None) -> dict | None:
    """Fetch coordinator metrics JSON endpoint."""
    return _fetch_json(url, api_key)


def _fetch_api_dashboard(base_url: str, api_key: str | None = None) -> dict | None:
    """Fetch API dashboard snapshot endpoint."""
    url = base_url.rstrip("/") + "/api/v1/dashboard/snapshot"
    return _fetch_json(url, api_key)


def _fetch_api_fleet(base_url: str, api_key: str | None = None) -> dict | None:
    """Fetch API fleet status endpoint."""
    url = base_url.rstrip("/") + "/api/v1/fleet/status"
    return _fetch_json(url, api_key)


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------

def capture_snapshot(
    coordinator_url: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Capture a single metrics snapshot from all available sources.

    Returns a dict with schema_version, timestamp, and data from each source.
    """
    now = time.time()
    iso_ts = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": now,
        "timestamp_iso": iso_ts,
        "sources": {},
    }

    # Coordinator metrics
    if coordinator_url:
        data = _fetch_coordinator_metrics(coordinator_url, api_key)
        if data is not None:
            snapshot["sources"]["coordinator"] = data

    # API dashboard
    if api_url:
        dashboard_data = _fetch_api_dashboard(api_url, api_key)
        if dashboard_data is not None:
            snapshot["sources"]["api_dashboard"] = dashboard_data

        fleet_data = _fetch_api_fleet(api_url, api_key)
        if fleet_data is not None:
            snapshot["sources"]["api_fleet"] = fleet_data

    return snapshot


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------

def append_snapshot(filepath: Path, snapshot: dict) -> None:
    """Append a snapshot as a single JSONL line."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")


def read_snapshots(filepath: Path) -> list[dict]:
    """Read all snapshots from a JSONL file."""
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


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], pct: float) -> float:
    """Calculate percentile from pre-sorted data."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _stddev(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division with zero guard."""
    return a / b if b != 0 else default


def compute_series_stats(values: list[float]) -> dict:
    """Compute comprehensive statistics for a numeric series."""
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


# ---------------------------------------------------------------------------
# Metric extraction from snapshots
# ---------------------------------------------------------------------------

def _extract_metric(snapshot: dict, path: list[str], default=None):
    """Extract a nested value from a snapshot dict using a key path."""
    current = snapshot
    for key in path:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def extract_time_series(snapshots: list[dict]) -> dict[str, list[float]]:
    """Extract named time series from a list of snapshots.

    Returns dict mapping metric name to list of float values.
    """
    series: dict[str, list[float]] = {
        "error_rate": [],
        "completion_rate": [],
        "s3_error_rate": [],
        "pages_processed": [],
        "avg_processing_time_ms": [],
        "gpu_workers_available": [],
        "workers_online": [],
        "workers_busy": [],
        "stuck_jobs": [],
        "pages_per_minute": [],
        "docs_per_hour": [],
    }

    for snap in snapshots:
        coord = snap.get("sources", {}).get("coordinator", {})
        dash = snap.get("sources", {}).get("api_dashboard", {})

        # Coordinator metrics
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

        # Worker status breakdown
        by_status = workers.get("by_status", {})
        online = by_status.get("online", 0)
        busy = by_status.get("busy", 0)
        _append_if_numeric(series["workers_online"], online)
        _append_if_numeric(series["workers_busy"], busy)

        # Dashboard metrics
        throughput = dash.get("throughput", {})
        _append_if_numeric(series["pages_per_minute"], throughput.get("pages_per_minute"))
        _append_if_numeric(series["docs_per_hour"], throughput.get("docs_per_hour"))

    return series


def _append_if_numeric(lst: list, value) -> None:
    """Append value to list only if it is a valid number."""
    if value is not None and isinstance(value, (int, float)):
        lst.append(float(value))


# ---------------------------------------------------------------------------
# Hourly pattern analysis
# ---------------------------------------------------------------------------

def compute_hourly_patterns(snapshots: list[dict]) -> dict[int, dict]:
    """Compute per-hour-of-day statistics.

    Returns dict mapping hour (0-23) to stats dict with
    avg_pages_per_minute and avg_error_rate.
    """
    hourly: dict[int, dict[str, list[float]]] = {
        h: {"ppm": [], "error_rate": []} for h in range(24)
    }

    for snap in snapshots:
        ts = snap.get("timestamp")
        if ts is None:
            continue
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour

        coord = snap.get("sources", {}).get("coordinator", {})
        dash = snap.get("sources", {}).get("api_dashboard", {})

        er = coord.get("jobs", {}).get("error_rate_1h")
        if er is not None and isinstance(er, (int, float)):
            hourly[hour]["error_rate"].append(float(er))

        ppm = dash.get("throughput", {}).get("pages_per_minute")
        if ppm is not None and isinstance(ppm, (int, float)):
            hourly[hour]["ppm"].append(float(ppm))

    result = {}
    for h in range(24):
        ppm_vals = hourly[h]["ppm"]
        er_vals = hourly[h]["error_rate"]
        result[h] = {
            "avg_pages_per_minute": round(sum(ppm_vals) / len(ppm_vals), 4) if ppm_vals else 0.0,
            "avg_error_rate": round(sum(er_vals) / len(er_vals), 4) if er_vals else 0.0,
            "sample_count": max(len(ppm_vals), len(er_vals)),
        }
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_baseline_report(snapshots: list[dict]) -> dict:
    """Generate a comprehensive baseline summary report from snapshots.

    Returns a JSON-serializable dict.
    """
    if not snapshots:
        return {
            "schema_version": SCHEMA_VERSION,
            "error": "No snapshots available for report generation",
            "snapshot_count": 0,
        }

    timestamps = [s.get("timestamp", 0) for s in snapshots]
    start_ts = min(timestamps)
    end_ts = max(timestamps)
    duration_hours = (end_ts - start_ts) / 3600

    series = extract_time_series(snapshots)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection_period": {
            "start": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
            "duration_hours": round(duration_hours, 2),
            "snapshot_count": len(snapshots),
        },
        "throughput": {
            "pages_per_minute": compute_series_stats(series["pages_per_minute"]),
            "docs_per_hour": compute_series_stats(series["docs_per_hour"]),
        },
        "latency": {
            "avg_processing_time_ms": compute_series_stats(series["avg_processing_time_ms"]),
        },
        "error_rates": {
            "job_error_rate": compute_series_stats(series["error_rate"]),
            "s3_error_rate": compute_series_stats(series["s3_error_rate"]),
            "completion_rate": compute_series_stats(series["completion_rate"]),
        },
        "workers": {
            "gpu_available": compute_series_stats(series["gpu_workers_available"]),
            "online": compute_series_stats(series["workers_online"]),
            "busy": compute_series_stats(series["workers_busy"]),
        },
        "queue_health": {
            "stuck_jobs": compute_series_stats(series["stuck_jobs"]),
        },
        "volume": {
            "pages_processed_series": compute_series_stats(series["pages_processed"]),
        },
        "hourly_patterns": compute_hourly_patterns(snapshots),
    }

    return report


def generate_markdown_report(report: dict) -> str:
    """Convert a baseline report dict to markdown format."""
    lines = []
    lines.append("# Production Baseline Report")
    lines.append("")

    period = report.get("collection_period", {})
    lines.append(f"- **Generated**: {report.get('generated_at', 'unknown')}")
    lines.append(f"- **Period**: {period.get('start', '?')} to {period.get('end', '?')}")
    lines.append(f"- **Duration**: {period.get('duration_hours', 0)} hours")
    lines.append(f"- **Snapshots**: {period.get('snapshot_count', 0)}")
    lines.append("")

    # Throughput
    lines.append("## Throughput")
    lines.append("")
    ppm = report.get("throughput", {}).get("pages_per_minute", {})
    dph = report.get("throughput", {}).get("docs_per_hour", {})
    lines.append("| Metric | Mean | P50 | P95 | P99 | Stddev |")
    lines.append("|--------|------|-----|-----|-----|--------|")
    lines.append(
        f"| Pages/min | {ppm.get('mean', 0)} | {ppm.get('p50', 0)} | "
        f"{ppm.get('p95', 0)} | {ppm.get('p99', 0)} | {ppm.get('stddev', 0)} |"
    )
    lines.append(
        f"| Docs/hour | {dph.get('mean', 0)} | {dph.get('p50', 0)} | "
        f"{dph.get('p95', 0)} | {dph.get('p99', 0)} | {dph.get('stddev', 0)} |"
    )
    lines.append("")

    # Latency
    lines.append("## Latency")
    lines.append("")
    lat = report.get("latency", {}).get("avg_processing_time_ms", {})
    lines.append("| Metric | Mean | P50 | P95 | P99 | Min | Max |")
    lines.append("|--------|------|-----|-----|-----|-----|-----|")
    lines.append(
        f"| Avg Processing (ms) | {lat.get('mean', 0)} | {lat.get('p50', 0)} | "
        f"{lat.get('p95', 0)} | {lat.get('p99', 0)} | {lat.get('min', 0)} | {lat.get('max', 0)} |"
    )
    lines.append("")

    # Error Rates
    lines.append("## Error Rates")
    lines.append("")
    er = report.get("error_rates", {}).get("job_error_rate", {})
    cr = report.get("error_rates", {}).get("completion_rate", {})
    lines.append("| Metric | Mean | Max | Stddev |")
    lines.append("|--------|------|-----|--------|")
    lines.append(f"| Job Error Rate | {er.get('mean', 0)} | {er.get('max', 0)} | {er.get('stddev', 0)} |")
    lines.append(f"| Completion Rate | {cr.get('mean', 0)} | {cr.get('min', 0)} (min) | {cr.get('stddev', 0)} |")
    lines.append("")

    # Workers
    lines.append("## Worker Utilization")
    lines.append("")
    gpu = report.get("workers", {}).get("gpu_available", {})
    busy = report.get("workers", {}).get("busy", {})
    lines.append("| Metric | Mean | Peak | Min |")
    lines.append("|--------|------|------|-----|")
    lines.append(f"| GPU Workers Available | {gpu.get('mean', 0)} | {gpu.get('max', 0)} | {gpu.get('min', 0)} |")
    lines.append(f"| Busy Workers | {busy.get('mean', 0)} | {busy.get('max', 0)} | {busy.get('min', 0)} |")
    lines.append("")

    # Queue Health
    lines.append("## Queue Health")
    lines.append("")
    stuck = report.get("queue_health", {}).get("stuck_jobs", {})
    lines.append(f"- **Stuck Jobs**: mean={stuck.get('mean', 0)}, max={stuck.get('max', 0)}")
    lines.append("")

    # Hourly Patterns
    lines.append("## Hourly Patterns (UTC)")
    lines.append("")
    hourly = report.get("hourly_patterns", {})
    if hourly:
        lines.append("| Hour | Avg Pages/min | Avg Error Rate | Samples |")
        lines.append("|------|--------------|----------------|---------|")
        for h in range(24):
            data = hourly.get(str(h), hourly.get(h, {}))
            lines.append(
                f"| {h:02d}:00 | {data.get('avg_pages_per_minute', 0)} | "
                f"{data.get('avg_error_rate', 0)} | {data.get('sample_count', 0)} |"
            )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Continuous capture loop
# ---------------------------------------------------------------------------

def run_capture_loop(
    coordinator_url: str | None,
    api_url: str | None,
    api_key: str | None,
    interval_seconds: int,
    duration_hours: float,
    output_path: Path,
) -> int:
    """Run continuous capture for the specified duration.

    Returns count of snapshots captured.
    """
    end_time = time.time() + (duration_hours * 3600)
    count = 0

    while not _shutdown and time.time() < end_time:
        snapshot = capture_snapshot(coordinator_url, api_url, api_key)
        append_snapshot(output_path, snapshot)
        count += 1

        # Wait for next interval (check shutdown every second)
        next_capture = time.time() + interval_seconds
        while not _shutdown and time.time() < next_capture and time.time() < end_time:
            time.sleep(min(1.0, next_capture - time.time()))

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Production traffic monitoring baseline capture tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Sources
    parser.add_argument(
        "--coordinator-url",
        default=DEFAULT_COORDINATOR_URL,
        help=f"Coordinator metrics endpoint (default: {DEFAULT_COORDINATOR_URL})",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="API base URL for dashboard/fleet endpoints",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication (or set METRICS_API_KEY env var)",
    )

    # Capture mode
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Take a single snapshot and exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"Capture interval in seconds (default: {DEFAULT_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_HOURS,
        help=f"Capture duration in hours (default: {DEFAULT_DURATION_HOURS})",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Override JSONL output filename (default: baseline_YYYYMMDD_HHMMSS.jsonl)",
    )

    # Report mode
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing JSONL without capturing new data",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input JSONL file for --report-only mode",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    # Resolve API key
    api_key = args.api_key or os.environ.get("METRICS_API_KEY")

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    # Report-only mode
    if args.report_only:
        if not args.input:
            print("ERROR: --input required with --report-only", file=sys.stderr)
            return 1
        if not args.input.exists():
            print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
            return 1

        snapshots = read_snapshots(args.input)
        if not snapshots:
            print("ERROR: No valid snapshots in input file", file=sys.stderr)
            return 1

        report = generate_baseline_report(snapshots)
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "baseline_report.json"
        json_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report written to: {json_path}")

        md_path = output_dir / "baseline_report.md"
        md_path.write_text(generate_markdown_report(report), encoding="utf-8")
        print(f"Markdown report written to: {md_path}")

        return 0

    # Resolve output path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_file:
        output_path = output_dir / args.output_file
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"baseline_{ts}.jsonl"

    # Single snapshot mode
    if args.snapshot:
        snapshot = capture_snapshot(args.coordinator_url, args.api_url, api_key)
        append_snapshot(output_path, snapshot)
        source_count = len(snapshot.get("sources", {}))
        print(f"Snapshot captured ({source_count} sources) -> {output_path}")

        # Also print summary
        print(json.dumps(snapshot, indent=2, default=str))
        return 0

    # Continuous capture mode
    print("Starting baseline capture:")
    print(f"  Coordinator: {args.coordinator_url}")
    if args.api_url:
        print(f"  API: {args.api_url}")
    print(f"  Interval: {args.interval}s")
    print(f"  Duration: {args.duration}h")
    print(f"  Output: {output_path}")
    print("")

    count = run_capture_loop(
        coordinator_url=args.coordinator_url,
        api_url=args.api_url,
        api_key=api_key,
        interval_seconds=args.interval,
        duration_hours=args.duration,
        output_path=output_path,
    )

    print(f"\nCapture complete: {count} snapshots -> {output_path}")

    # Generate report if we have data
    if count > 0:
        snapshots = read_snapshots(output_path)
        report = generate_baseline_report(snapshots)

        json_path = output_dir / "baseline_report.json"
        json_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        md_path = output_dir / "baseline_report.md"
        md_path.write_text(generate_markdown_report(report), encoding="utf-8")

        print(f"Report: {json_path}")
        print(f"Report: {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
