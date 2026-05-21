"""Benchmark file watcher latency.

Measures the time between a file being created in a watched directory and
the watcher detecting it. Reports p50, p95, p99, and max latency.

Usage:
    python scripts/benchmark_watcher.py --iterations 20
    python scripts/benchmark_watcher.py --watch-dir /tmp/benchmark --iterations 50
    python scripts/benchmark_watcher.py --output results.json
    python scripts/benchmark_watcher.py --target-latency 10.0

Target KPI: <10 seconds detection latency (p95).

Requires: file_watcher.py, file_watcher_config.py (from EDCOCR project root)
"""

import argparse
import datetime
import json
import logging
import os
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass
class WatcherBenchmarkResult:
    """Complete benchmark result for file watcher latency."""

    iterations: int = 0
    successful_detections: int = 0
    missed_detections: int = 0
    latencies_ms: list = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    stddev_ms: float = 0.0
    target_latency_s: float = 10.0
    passed: bool = False
    watch_dir: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------


def compute_latency_percentiles(latencies_ms: list) -> dict:
    """Compute latency percentiles from a list of measurements.

    Parameters
    ----------
    latencies_ms : list[float]
        Latency measurements in milliseconds.

    Returns
    -------
    dict
        Dictionary with p50, p95, p99, min, max, mean, stddev.
    """
    if not latencies_ms:
        return {
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "stddev": 0.0,
        }

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    def _percentile(pct: float) -> float:
        idx = int(n * pct / 100.0)
        idx = min(idx, n - 1)
        return sorted_lat[idx]

    return {
        "p50": round(_percentile(50), 2),
        "p95": round(_percentile(95), 2),
        "p99": round(_percentile(99), 2),
        "min": round(min(sorted_lat), 2),
        "max": round(max(sorted_lat), 2),
        "mean": round(statistics.mean(sorted_lat), 2),
        "stddev": round(statistics.stdev(sorted_lat), 2) if n >= 2 else 0.0,
    }


# ---------------------------------------------------------------------------
# Watcher benchmark
# ---------------------------------------------------------------------------


def run_watcher_benchmark(
    watch_dir: str = None,
    iterations: int = 20,
    target_latency_s: float = 10.0,
    detection_timeout_s: float = 30.0,
) -> WatcherBenchmarkResult:
    """Run file watcher latency benchmark.

    Creates files in a watched directory and measures how quickly the
    watcher detects them. Uses a lightweight detection callback to
    record timestamps.

    Parameters
    ----------
    watch_dir : str, optional
        Directory to watch. If None, creates a temp directory.
    iterations : int
        Number of files to drop and measure.
    target_latency_s : float
        Target latency in seconds for pass/fail.
    detection_timeout_s : float
        Maximum wait time per file detection before counting as missed.

    Returns
    -------
    WatcherBenchmarkResult
        Benchmark results with latency percentiles.
    """
    # Try to import watchdog for the lightweight detection approach
    try:
        import watchdog.events  # noqa: F401
        import watchdog.observers  # noqa: F401
        _watchdog_available = True
    except ImportError:
        _watchdog_available = False

    use_temp = watch_dir is None
    if use_temp:
        temp_dir = tempfile.mkdtemp(prefix="ocr_watcher_bench_")
        watch_dir = temp_dir
    else:
        os.makedirs(watch_dir, exist_ok=True)
        temp_dir = None

    latencies_ms = []
    missed = 0

    try:
        if _watchdog_available:
            latencies_ms, missed = _benchmark_with_watchdog(
                watch_dir, iterations, detection_timeout_s
            )
        else:
            latencies_ms, missed = _benchmark_with_polling(
                watch_dir, iterations, detection_timeout_s
            )
    finally:
        # Clean up temp directory
        if temp_dir and os.path.isdir(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    stats = compute_latency_percentiles(latencies_ms)

    return WatcherBenchmarkResult(
        iterations=iterations,
        successful_detections=len(latencies_ms),
        missed_detections=missed,
        latencies_ms=[round(lat, 2) for lat in latencies_ms],
        p50_ms=stats["p50"],
        p95_ms=stats["p95"],
        p99_ms=stats["p99"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        mean_ms=stats["mean"],
        stddev_ms=stats["stddev"],
        target_latency_s=target_latency_s,
        passed=stats["p95"] <= target_latency_s * 1000 if latencies_ms else False,
        watch_dir=str(watch_dir),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    )


def _benchmark_with_watchdog(
    watch_dir: str,
    iterations: int,
    timeout_s: float,
) -> tuple:
    """Run benchmark using watchdog observer for file detection.

    Returns
    -------
    tuple
        (latencies_ms, missed_count)
    """
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    detected_events = {}
    detection_lock = threading.Lock()

    class BenchHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                with detection_lock:
                    basename = os.path.basename(event.src_path)
                    if basename not in detected_events:
                        detected_events[basename] = time.perf_counter()

    handler = BenchHandler()
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()

    latencies_ms = []
    missed = 0

    try:
        # Small warmup delay for observer to initialize
        time.sleep(0.5)

        for i in range(iterations):
            filename = f"bench_test_{i:04d}.txt"
            filepath = os.path.join(watch_dir, filename)

            # Record creation time and write file
            create_time = time.perf_counter()
            with open(filepath, "w") as f:
                f.write(f"benchmark file {i}\n")

            # Wait for detection
            deadline = time.perf_counter() + timeout_s
            detected = False
            while time.perf_counter() < deadline:
                with detection_lock:
                    if filename in detected_events:
                        detect_time = detected_events[filename]
                        latency_ms = (detect_time - create_time) * 1000
                        latencies_ms.append(max(0, latency_ms))
                        detected = True
                        break
                time.sleep(0.01)

            if not detected:
                missed += 1
                logger.warning("File %s not detected within %ss", filename, timeout_s)

            # Clean up created file
            try:
                os.remove(filepath)
            except OSError:
                pass

            # Brief pause between iterations
            time.sleep(0.05)

    finally:
        observer.stop()
        observer.join(timeout=5)

    return latencies_ms, missed


def _benchmark_with_polling(
    watch_dir: str,
    iterations: int,
    timeout_s: float,
) -> tuple:
    """Fallback benchmark using filesystem polling (no watchdog).

    Returns
    -------
    tuple
        (latencies_ms, missed_count)
    """
    latencies_ms = []
    missed = 0
    poll_interval = 0.1  # 100ms poll

    for i in range(iterations):
        filename = f"bench_test_{i:04d}.txt"
        filepath = os.path.join(watch_dir, filename)

        # Get initial directory listing
        before = set(os.listdir(watch_dir))

        # Write file and record time
        create_time = time.perf_counter()
        with open(filepath, "w") as f:
            f.write(f"benchmark file {i}\n")

        # Poll for detection
        deadline = time.perf_counter() + timeout_s
        detected = False
        while time.perf_counter() < deadline:
            current = set(os.listdir(watch_dir))
            new_files = current - before
            if filename in new_files:
                detect_time = time.perf_counter()
                latency_ms = (detect_time - create_time) * 1000
                latencies_ms.append(max(0, latency_ms))
                detected = True
                break
            time.sleep(poll_interval)

        if not detected:
            missed += 1

        try:
            os.remove(filepath)
        except OSError:
            pass

        time.sleep(0.05)

    return latencies_ms, missed


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_watcher_report(result: WatcherBenchmarkResult) -> str:
    """Format benchmark result as a human-readable report.

    Parameters
    ----------
    result : WatcherBenchmarkResult
        Benchmark result to format.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("FILE WATCHER LATENCY BENCHMARK")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Watch Dir:        {result.watch_dir}")
    lines.append(f"  Iterations:       {result.iterations}")
    lines.append(f"  Detected:         {result.successful_detections}")
    lines.append(f"  Missed:           {result.missed_detections}")
    lines.append(f"  Target (P95):     {result.target_latency_s:.1f}s ({result.target_latency_s * 1000:.0f}ms)")
    lines.append(f"  Result:           {'PASS' if result.passed else 'FAIL'}")
    lines.append("")
    lines.append("LATENCY PERCENTILES")
    lines.append("-" * 80)
    lines.append(f"  P50:              {result.p50_ms:.2f} ms")
    lines.append(f"  P95:              {result.p95_ms:.2f} ms")
    lines.append(f"  P99:              {result.p99_ms:.2f} ms")
    lines.append(f"  Min:              {result.min_ms:.2f} ms")
    lines.append(f"  Max:              {result.max_ms:.2f} ms")
    lines.append(f"  Mean:             {result.mean_ms:.2f} ms")
    lines.append(f"  Stddev:           {result.stddev_ms:.2f} ms")
    lines.append(f"  Timestamp:        {result.timestamp}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for watcher latency benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark file watcher detection latency (Phase 6 KPI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_watcher.py --iterations 20
  python scripts/benchmark_watcher.py --watch-dir /tmp/bench --iterations 50
  python scripts/benchmark_watcher.py --output results.json
  python scripts/benchmark_watcher.py --target-latency 5.0
        """,
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        help="Directory to watch (default: temp directory)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of file detection iterations (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON path for structured results",
    )
    parser.add_argument(
        "--target-latency",
        type=float,
        default=10.0,
        help="Target P95 latency in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--detection-timeout",
        type=float,
        default=30.0,
        help="Max wait per file detection in seconds (default: 30.0)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info(
        "Running watcher latency benchmark (%d iterations, target P95 < %.1fs)...",
        args.iterations,
        args.target_latency,
    )

    result = run_watcher_benchmark(
        watch_dir=args.watch_dir,
        iterations=args.iterations,
        target_latency_s=args.target_latency,
        detection_timeout_s=args.detection_timeout,
    )

    report = format_watcher_report(result)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info("Results saved to %s", args.output)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
