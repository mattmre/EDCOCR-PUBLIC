"""Benchmark suite for adaptive batch sizing (ENABLE_ADAPTIVE_BATCH).

Generates synthetic workloads of varying complexity and tests the
AdaptiveBatchSizer from adaptive_batch.py with different batch sizes
and strategies.  Measures throughput (pages/sec), latency, and memory
usage, then emits a tuning recommendation report in JSON and markdown.

Usage:
    python scripts/benchmark_adaptive_batch.py
    python scripts/benchmark_adaptive_batch.py --workload-size 200 --iterations 5
    python scripts/benchmark_adaptive_batch.py --output-dir ./reports
    python scripts/benchmark_adaptive_batch.py --strategies adaptive,fixed
"""

import argparse
import datetime
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ocr_local.infra.adaptive_batch import (
    AdaptiveBatchSizer,
    BatchConfig,
    BatchResult,
    BatchStrategy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Workload profiles
# ---------------------------------------------------------------------------

# Predefined workload mixes
WORKLOAD_PROFILES = {
    "small": {
        "description": "Small documents (invoices, receipts)",
        "width_range": (800, 1200),
        "height_range": (1000, 1600),
        "file_size_range": (50_000, 200_000),
        "dpi": 200,
        "table_probability": 0.3,
        "image_probability": 0.1,
    },
    "medium": {
        "description": "Standard A4/letter documents",
        "width_range": (2400, 2600),
        "height_range": (3300, 3600),
        "file_size_range": (500_000, 2_000_000),
        "dpi": 300,
        "table_probability": 0.2,
        "image_probability": 0.2,
    },
    "large": {
        "description": "Large engineering drawings, maps",
        "width_range": (4000, 6000),
        "height_range": (5000, 8000),
        "file_size_range": (3_000_000, 10_000_000),
        "dpi": 400,
        "table_probability": 0.1,
        "image_probability": 0.5,
    },
    "mixed": {
        "description": "Mixed complexity (real-world distribution)",
        "width_range": (800, 5000),
        "height_range": (1000, 7000),
        "file_size_range": (50_000, 8_000_000),
        "dpi": 300,
        "table_probability": 0.25,
        "image_probability": 0.3,
    },
}


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass
class StrategyBenchmarkResult:
    """Benchmark results for a single strategy run."""

    strategy: str = ""
    max_batch_size: int = 0
    workload_profile: str = ""
    pages_total: int = 0
    batches_processed: int = 0
    total_duration_s: float = 0.0
    throughput_pages_per_sec: float = 0.0
    avg_batch_size: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    memory_peak_mb: float = 0.0
    final_batch_size: int = 0
    adaptation_count: int = 0


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all strategies and workloads."""

    timestamp: str = ""
    workload_size: int = 0
    iterations: int = 0
    results: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic workload generator
# ---------------------------------------------------------------------------


def _get_rng(seed=None):
    """Return a random.Random instance for deterministic generation."""
    import random
    return random.Random(seed)


def generate_workload(
    sizer: AdaptiveBatchSizer,
    profile_name: str,
    page_count: int,
    seed: int = 42,
) -> list:
    """Generate a list of PageComplexity objects for a workload profile.

    Parameters
    ----------
    sizer : AdaptiveBatchSizer
        Sizer instance used to compute complexity scores.
    profile_name : str
        Name of a profile from WORKLOAD_PROFILES.
    page_count : int
        Number of pages to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[PageComplexity]
        Generated page complexity profiles.
    """
    profile = WORKLOAD_PROFILES.get(profile_name, WORKLOAD_PROFILES["mixed"])
    rng = _get_rng(seed)
    pages = []

    for i in range(page_count):
        width = rng.randint(*profile["width_range"])
        height = rng.randint(*profile["height_range"])
        file_size = rng.randint(*profile["file_size_range"])
        has_tables = rng.random() < profile["table_probability"]
        has_images = rng.random() < profile["image_probability"]

        pc = sizer.compute_complexity(
            width=width,
            height=height,
            file_size=file_size,
            dpi=profile["dpi"],
            has_tables=has_tables,
            has_images=has_images,
        )
        pc.page_number = i + 1
        pages.append(pc)

    return pages


# ---------------------------------------------------------------------------
# Simulated processing
# ---------------------------------------------------------------------------

_PROCESS_MEMORY_MB = 0.0


def _get_process_memory_mb() -> float:
    """Get current process memory usage in MB (best-effort)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def simulate_batch_processing(
    pages: list,
    batch_size: int,
    base_latency_ms: float = 10.0,
) -> BatchResult:
    """Simulate processing a batch of pages with complexity-scaled latency.

    Parameters
    ----------
    pages : list[PageComplexity]
        Pages in this batch.
    batch_size : int
        Nominal batch size.
    base_latency_ms : float
        Base latency per page in milliseconds.

    Returns
    -------
    BatchResult
        Simulated batch outcome.
    """
    if not pages:
        return BatchResult()

    start = time.perf_counter()
    # Simulate work proportional to complexity
    avg_complexity = sum(p.complexity_score for p in pages) / len(pages)
    # Scale latency: simple pages are fast, complex pages are slower
    per_page_ms = base_latency_ms * (0.5 + avg_complexity)
    total_sim_ms = per_page_ms * len(pages)

    # Sleep a fraction of the simulated time (capped to avoid long benchmarks)
    sleep_s = min(total_sim_ms / 1000.0, 0.05)
    time.sleep(sleep_s)

    duration = time.perf_counter() - start
    mem = _get_process_memory_mb()

    return BatchResult(
        batch_size=batch_size,
        pages_processed=len(pages),
        duration_seconds=duration,
        memory_peak_mb=mem if mem > 0 else avg_complexity * 100,
        avg_page_complexity=round(avg_complexity, 4),
        success_count=len(pages),
        failure_count=0,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_strategy_benchmark(
    strategy: BatchStrategy,
    max_batch_size: int,
    workload_profile: str,
    page_count: int,
    seed: int = 42,
) -> StrategyBenchmarkResult:
    """Benchmark a single batch strategy on a given workload.

    Parameters
    ----------
    strategy : BatchStrategy
        The batch sizing strategy to test.
    max_batch_size : int
        Maximum batch size for the config.
    workload_profile : str
        Name of the workload profile.
    page_count : int
        Total pages to process.
    seed : int
        Random seed.

    Returns
    -------
    StrategyBenchmarkResult
        Measured performance metrics.
    """
    config = BatchConfig(
        strategy=strategy,
        min_batch_size=1,
        max_batch_size=max_batch_size,
        target_memory_pct=75.0,
        warmup_batches=3,
        adjustment_factor=0.1,
    )
    sizer = AdaptiveBatchSizer(config)

    pages = generate_workload(sizer, workload_profile, page_count, seed=seed)
    if not pages:
        return StrategyBenchmarkResult(
            strategy=strategy.value,
            max_batch_size=max_batch_size,
            workload_profile=workload_profile,
        )

    batch_latencies_ms = []
    batches = 0
    total_pages = 0
    overall_start = time.perf_counter()
    idx = 0

    while idx < len(pages):
        remaining = pages[idx:]
        recommended = sizer.recommend_batch_size(remaining[:max_batch_size])
        batch_pages = remaining[:recommended]

        result = simulate_batch_processing(batch_pages, recommended)
        sizer.record_result(result)

        batch_latencies_ms.append(result.duration_seconds * 1000)
        batches += 1
        total_pages += result.pages_processed
        idx += result.pages_processed

    overall_duration = time.perf_counter() - overall_start
    final_batch_size = sizer.get_current_batch_size()

    # Compute adaptation count (how many times batch size changed)
    history = sizer.get_history()
    adaptation_count = 0
    if len(history) >= 2:
        for i in range(1, len(history)):
            if history[i].batch_size != history[i - 1].batch_size:
                adaptation_count += 1

    throughput = total_pages / overall_duration if overall_duration > 0 else 0

    avg_lat = statistics.mean(batch_latencies_ms) if batch_latencies_ms else 0
    sorted_lat = sorted(batch_latencies_ms) if batch_latencies_ms else [0]
    p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
    p95_lat = sorted_lat[p95_idx]

    avg_bs = total_pages / batches if batches > 0 else 0

    return StrategyBenchmarkResult(
        strategy=strategy.value,
        max_batch_size=max_batch_size,
        workload_profile=workload_profile,
        pages_total=total_pages,
        batches_processed=batches,
        total_duration_s=round(overall_duration, 4),
        throughput_pages_per_sec=round(throughput, 2),
        avg_batch_size=round(avg_bs, 2),
        avg_latency_ms=round(avg_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        memory_peak_mb=round(_get_process_memory_mb(), 1),
        final_batch_size=final_batch_size,
        adaptation_count=adaptation_count,
    )


def run_full_benchmark(
    workload_size: int = 100,
    iterations: int = 3,
    strategies: list = None,
    batch_sizes: list = None,
    profiles: list = None,
) -> BenchmarkReport:
    """Run the full adaptive batch benchmark suite.

    Parameters
    ----------
    workload_size : int
        Number of pages per workload.
    iterations : int
        Number of iterations per configuration.
    strategies : list[str], optional
        Strategy names to test. Default: all.
    batch_sizes : list[int], optional
        Max batch sizes to test. Default: [4, 8, 16, 32].
    profiles : list[str], optional
        Workload profiles to test. Default: all.

    Returns
    -------
    BenchmarkReport
        Complete benchmark report.
    """
    if strategies is None:
        strategies = ["fixed", "adaptive", "memory_aware", "throughput_optimal"]
    if batch_sizes is None:
        batch_sizes = [4, 8, 16, 32]
    if profiles is None:
        profiles = list(WORKLOAD_PROFILES.keys())

    strategy_map = {s.value: s for s in BatchStrategy}
    results = []

    for profile in profiles:
        for strat_name in strategies:
            strat = strategy_map.get(strat_name)
            if strat is None:
                logger.warning("Unknown strategy: %s", strat_name)
                continue

            for bs in batch_sizes:
                iteration_results = []
                for it in range(iterations):
                    r = run_strategy_benchmark(
                        strategy=strat,
                        max_batch_size=bs,
                        workload_profile=profile,
                        page_count=workload_size,
                        seed=42 + it,
                    )
                    iteration_results.append(r)

                # Average across iterations
                avg = StrategyBenchmarkResult(
                    strategy=strat_name,
                    max_batch_size=bs,
                    workload_profile=profile,
                    pages_total=workload_size,
                    batches_processed=round(
                        statistics.mean([r.batches_processed for r in iteration_results])
                    ),
                    total_duration_s=round(
                        statistics.mean([r.total_duration_s for r in iteration_results]), 4
                    ),
                    throughput_pages_per_sec=round(
                        statistics.mean([r.throughput_pages_per_sec for r in iteration_results]), 2
                    ),
                    avg_batch_size=round(
                        statistics.mean([r.avg_batch_size for r in iteration_results]), 2
                    ),
                    avg_latency_ms=round(
                        statistics.mean([r.avg_latency_ms for r in iteration_results]), 2
                    ),
                    p95_latency_ms=round(
                        statistics.mean([r.p95_latency_ms for r in iteration_results]), 2
                    ),
                    memory_peak_mb=round(
                        max(r.memory_peak_mb for r in iteration_results), 1
                    ),
                    final_batch_size=round(
                        statistics.mean([r.final_batch_size for r in iteration_results])
                    ),
                    adaptation_count=round(
                        statistics.mean([r.adaptation_count for r in iteration_results])
                    ),
                )
                results.append(avg)

    # Generate recommendations
    recommendations = _generate_recommendations(results)

    return BenchmarkReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        workload_size=workload_size,
        iterations=iterations,
        results=[asdict(r) for r in results],
        recommendations=recommendations,
    )


def _generate_recommendations(results: list) -> list:
    """Generate tuning recommendations from benchmark results.

    Parameters
    ----------
    results : list[StrategyBenchmarkResult]
        Benchmark results across all configurations.

    Returns
    -------
    list[dict]
        Recommendations per workload profile.
    """
    recs = []
    profiles = set(r.workload_profile for r in results)

    for profile in sorted(profiles):
        profile_results = [r for r in results if r.workload_profile == profile]
        if not profile_results:
            continue

        best = max(profile_results, key=lambda r: r.throughput_pages_per_sec)
        lowest_latency = min(profile_results, key=lambda r: r.p95_latency_ms)

        recs.append({
            "workload_profile": profile,
            "recommended_strategy": best.strategy,
            "recommended_max_batch_size": best.max_batch_size,
            "best_throughput_pps": best.throughput_pages_per_sec,
            "lowest_p95_latency_strategy": lowest_latency.strategy,
            "lowest_p95_latency_ms": lowest_latency.p95_latency_ms,
            "lowest_latency_batch_size": lowest_latency.max_batch_size,
        })

    return recs


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_markdown_report(report: BenchmarkReport) -> str:
    """Format the benchmark report as markdown.

    Parameters
    ----------
    report : BenchmarkReport
        Complete benchmark report.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# Adaptive Batch Benchmark Report",
        "",
        f"**Timestamp**: {report.timestamp}",
        f"**Workload size**: {report.workload_size} pages",
        f"**Iterations**: {report.iterations}",
        "",
        "## Results",
        "",
        "| Profile | Strategy | Max Batch | Throughput (p/s) | Avg Latency (ms) | P95 Latency (ms) | Avg Batch Size |",
        "|---------|----------|-----------|------------------|-------------------|-------------------|----------------|",
    ]

    for r in report.results:
        lines.append(
            f"| {r['workload_profile']} | {r['strategy']} | {r['max_batch_size']} "
            f"| {r['throughput_pages_per_sec']:.2f} "
            f"| {r['avg_latency_ms']:.2f} "
            f"| {r['p95_latency_ms']:.2f} "
            f"| {r['avg_batch_size']:.1f} |"
        )

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    for rec in report.recommendations:
        lines.append(
            f"- **{rec['workload_profile']}**: Use `{rec['recommended_strategy']}` "
            f"with max_batch_size={rec['recommended_max_batch_size']} "
            f"({rec['best_throughput_pps']:.2f} pages/sec)"
        )

    lines.append("")
    return "\n".join(lines)


def format_console_report(report: BenchmarkReport) -> str:
    """Format the benchmark report for console output.

    Parameters
    ----------
    report : BenchmarkReport
        Complete benchmark report.

    Returns
    -------
    str
        Console-formatted report.
    """
    lines = [
        "",
        "=" * 90,
        "ADAPTIVE BATCH BENCHMARK REPORT",
        "=" * 90,
        "",
        f"  Timestamp:      {report.timestamp}",
        f"  Workload size:  {report.workload_size} pages",
        f"  Iterations:     {report.iterations}",
        "",
        "-" * 90,
        f"{'Profile':<12} {'Strategy':<20} {'MaxBS':>5} {'Throughput':>12} "
        f"{'AvgLat(ms)':>12} {'P95Lat(ms)':>12} {'AvgBS':>6}",
        "-" * 90,
    ]

    for r in report.results:
        lines.append(
            f"{r['workload_profile']:<12} {r['strategy']:<20} {r['max_batch_size']:>5} "
            f"{r['throughput_pages_per_sec']:>12.2f} "
            f"{r['avg_latency_ms']:>12.2f} "
            f"{r['p95_latency_ms']:>12.2f} "
            f"{r['avg_batch_size']:>6.1f}"
        )

    lines.append("-" * 90)
    lines.append("")
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 90)

    for rec in report.recommendations:
        lines.append(
            f"  {rec['workload_profile']:<12}: "
            f"strategy={rec['recommended_strategy']}, "
            f"max_batch={rec['recommended_max_batch_size']}, "
            f"throughput={rec['best_throughput_pps']:.2f} p/s"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for adaptive batch benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive batch sizing strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_adaptive_batch.py
  python scripts/benchmark_adaptive_batch.py --workload-size 200 --iterations 5
  python scripts/benchmark_adaptive_batch.py --strategies adaptive,fixed
  python scripts/benchmark_adaptive_batch.py --output-dir ./reports
        """,
    )
    parser.add_argument(
        "--workload-size",
        type=int,
        default=100,
        help="Number of pages per workload (default: 100)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Iterations per configuration (default: 3)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategies (default: all)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated max batch sizes (default: 4,8,16,32)",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=None,
        help="Comma-separated workload profiles (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output reports (JSON + markdown)",
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

    strategies = args.strategies.split(",") if args.strategies else None
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else None
    profiles = args.profiles.split(",") if args.profiles else None

    logger.info("Running adaptive batch benchmark...")

    report = run_full_benchmark(
        workload_size=args.workload_size,
        iterations=args.iterations,
        strategies=strategies,
        batch_sizes=batch_sizes,
        profiles=profiles,
    )

    # Console output
    print(format_console_report(report))

    # Save reports
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "adaptive_batch_benchmark.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("JSON report saved to %s", json_path)

        md_path = out_dir / "adaptive_batch_benchmark.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(report))
        logger.info("Markdown report saved to %s", md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
