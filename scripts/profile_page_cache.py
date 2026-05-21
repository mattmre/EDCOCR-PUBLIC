"""Memory profiling tool for page cache (ENABLE_PAGE_CACHE).

Tests the PageCache from page_cache.py under varying document loads.
Measures memory usage, LRU eviction rates, cache hit/miss ratios, and
profiles memory growth over time with different cache sizes.  Emits a
profiling report in JSON and markdown.

Usage:
    python scripts/profile_page_cache.py
    python scripts/profile_page_cache.py --cache-size 256 --page-count 500
    python scripts/profile_page_cache.py --output-dir ./reports
    python scripts/profile_page_cache.py --strategies lru --ttl 30
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

from ocr_local.infra.page_cache import CacheStrategy, PageCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Simulated page sizes (bytes)
SMALL_PAGE_SIZE = 50_000       # ~50 KB
MEDIUM_PAGE_SIZE = 500_000     # ~500 KB
LARGE_PAGE_SIZE = 2_000_000    # ~2 MB

# Default cache sizes to test (in MiB)
DEFAULT_CACHE_SIZES_MIB = [64, 128, 256, 512]


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class CacheProfileResult:
    """Profiling results for a single cache configuration."""

    cache_size_mib: int = 0
    max_entries: int = 0
    strategy: str = "lru"
    ttl_seconds: float = 0.0
    page_count: int = 0
    page_size_label: str = ""
    page_size_bytes: int = 0
    # Performance metrics
    total_puts: int = 0
    total_gets: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    # Timing
    avg_put_us: float = 0.0
    avg_get_us: float = 0.0
    p95_put_us: float = 0.0
    p95_get_us: float = 0.0
    total_duration_s: float = 0.0
    # Memory
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_delta_mb: float = 0.0
    peak_cache_size_bytes: int = 0
    peak_cache_entries: int = 0
    # Growth snapshots (entries over time)
    growth_snapshots: list = field(default_factory=list)


@dataclass
class CacheProfileReport:
    """Complete profiling report across all configurations."""

    timestamp: str = ""
    page_count: int = 0
    results: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_process_memory_mb() -> float:
    """Get current process memory usage in MB (best-effort)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _generate_page_data(size_bytes: int, seed: int = 0) -> bytes:
    """Generate deterministic synthetic page data.

    Parameters
    ----------
    size_bytes : int
        Target data size.
    seed : int
        Seed for data variation.

    Returns
    -------
    bytes
        Synthetic page bytes.
    """
    # Use a repeating pattern with seed variation for fast deterministic generation
    pattern = bytes([(seed + i) & 0xFF for i in range(min(1024, size_bytes))])
    repeats = (size_bytes // len(pattern)) + 1
    return (pattern * repeats)[:size_bytes]


def _percentile(values: list, pct: float) -> float:
    """Compute a percentile from a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * pct / 100.0), len(s) - 1)
    return s[idx]


# ---------------------------------------------------------------------------
# Profiling engine
# ---------------------------------------------------------------------------


def profile_cache_config(
    cache_size_mib: int,
    max_entries: int,
    page_count: int,
    page_size_bytes: int,
    page_size_label: str,
    strategy: CacheStrategy = CacheStrategy.LRU,
    ttl_seconds: float = 0.0,
    access_pattern: str = "sequential",
    snapshot_interval: int = 50,
) -> CacheProfileResult:
    """Profile a single cache configuration.

    Parameters
    ----------
    cache_size_mib : int
        Cache size in MiB.
    max_entries : int
        Maximum cache entries.
    page_count : int
        Total pages to process.
    page_size_bytes : int
        Size of each page.
    page_size_label : str
        Human label for the page size.
    strategy : CacheStrategy
        Cache eviction strategy.
    ttl_seconds : float
        TTL for cache entries (0 = no expiry).
    access_pattern : str
        "sequential", "random", or "zipf" (skewed access).
    snapshot_interval : int
        Take a memory/stats snapshot every N operations.

    Returns
    -------
    CacheProfileResult
        Profiling metrics.
    """
    import random
    rng = random.Random(42)

    cache = PageCache(
        max_size_bytes=cache_size_mib * 1024 * 1024,
        max_entries=max_entries,
        default_ttl=ttl_seconds,
        strategy=strategy,
    )

    mem_before = _get_process_memory_mb()
    put_times_us = []
    get_times_us = []
    growth_snapshots = []
    peak_bytes = 0
    peak_entries = 0

    overall_start = time.perf_counter()

    # Phase 1: populate cache (puts)
    for i in range(page_count):
        key = f"page_{i:06d}"
        data = _generate_page_data(page_size_bytes, seed=i)

        t0 = time.perf_counter()
        cache.put(key, data, metadata={"page": i, "size": page_size_bytes})
        t1 = time.perf_counter()
        put_times_us.append((t1 - t0) * 1_000_000)

        # Snapshots
        if (i + 1) % snapshot_interval == 0:
            stats = cache.get_stats()
            growth_snapshots.append({
                "operation": i + 1,
                "current_size_bytes": stats.current_size_bytes,
                "current_entries": stats.current_entries,
                "evictions": stats.evictions,
                "hit_rate": round(stats.hit_rate, 4),
            })
            peak_bytes = max(peak_bytes, stats.current_size_bytes)
            peak_entries = max(peak_entries, stats.current_entries)

    # Phase 2: access pages (gets) with specified pattern
    access_indices = list(range(page_count))
    if access_pattern == "random":
        rng.shuffle(access_indices)
    elif access_pattern == "zipf":
        # Skewed: frequently access early pages
        access_indices = []
        for _ in range(page_count):
            # Simple Zipf-like: square-root distribution favoring lower indices
            idx = int(rng.paretovariate(1.5)) % page_count
            access_indices.append(idx)

    for i in access_indices:
        key = f"page_{i:06d}"
        t0 = time.perf_counter()
        cache.get(key)
        t1 = time.perf_counter()
        get_times_us.append((t1 - t0) * 1_000_000)

    overall_duration = time.perf_counter() - overall_start
    mem_after = _get_process_memory_mb()

    final_stats = cache.get_stats()
    peak_bytes = max(peak_bytes, final_stats.current_size_bytes)
    peak_entries = max(peak_entries, final_stats.current_entries)

    # Clean up
    cache.clear()

    return CacheProfileResult(
        cache_size_mib=cache_size_mib,
        max_entries=max_entries,
        strategy=strategy.value,
        ttl_seconds=ttl_seconds,
        page_count=page_count,
        page_size_label=page_size_label,
        page_size_bytes=page_size_bytes,
        total_puts=len(put_times_us),
        total_gets=len(get_times_us),
        hits=final_stats.hits,
        misses=final_stats.misses,
        evictions=final_stats.evictions,
        hit_rate=round(final_stats.hit_rate, 4),
        avg_put_us=round(statistics.mean(put_times_us), 2) if put_times_us else 0,
        avg_get_us=round(statistics.mean(get_times_us), 2) if get_times_us else 0,
        p95_put_us=round(_percentile(put_times_us, 95), 2),
        p95_get_us=round(_percentile(get_times_us, 95), 2),
        total_duration_s=round(overall_duration, 4),
        memory_before_mb=round(mem_before, 1),
        memory_after_mb=round(mem_after, 1),
        memory_delta_mb=round(mem_after - mem_before, 1),
        peak_cache_size_bytes=peak_bytes,
        peak_cache_entries=peak_entries,
        growth_snapshots=growth_snapshots,
    )


def run_full_profile(
    cache_sizes_mib: list = None,
    page_count: int = 200,
    ttl_seconds: float = 0.0,
) -> CacheProfileReport:
    """Run the full page cache profiling suite.

    Parameters
    ----------
    cache_sizes_mib : list[int], optional
        Cache sizes to test in MiB. Default: [64, 128, 256, 512].
    page_count : int
        Pages to process per configuration.
    ttl_seconds : float
        TTL for cache entries.

    Returns
    -------
    CacheProfileReport
        Complete profiling report.
    """
    if cache_sizes_mib is None:
        cache_sizes_mib = list(DEFAULT_CACHE_SIZES_MIB)

    page_sizes = [
        ("small", SMALL_PAGE_SIZE),
        ("medium", MEDIUM_PAGE_SIZE),
        ("large", LARGE_PAGE_SIZE),
    ]

    results = []

    for cache_mib in cache_sizes_mib:
        for label, psize in page_sizes:
            # Estimate max_entries from cache size and page size
            max_entries = max(16, (cache_mib * 1024 * 1024) // max(1, psize))

            logger.info(
                "Profiling: cache=%dMiB, pages=%d, page_size=%s (%d bytes), max_entries=%d",
                cache_mib, page_count, label, psize, max_entries,
            )

            for pattern in ["sequential", "random", "zipf"]:
                result = profile_cache_config(
                    cache_size_mib=cache_mib,
                    max_entries=max_entries,
                    page_count=page_count,
                    page_size_bytes=psize,
                    page_size_label=f"{label}_{pattern}",
                    strategy=CacheStrategy.LRU,
                    ttl_seconds=ttl_seconds,
                    access_pattern=pattern,
                )
                results.append(result)

    recommendations = _generate_recommendations(results)

    return CacheProfileReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        page_count=page_count,
        results=[asdict(r) for r in results],
        recommendations=recommendations,
    )


def _generate_recommendations(results: list) -> list:
    """Generate sizing recommendations from profiling results.

    Parameters
    ----------
    results : list[CacheProfileResult]
        Profiling results.

    Returns
    -------
    list[dict]
        Recommendations.
    """
    recs = []

    # Group by page size base label
    size_groups = {}
    for r in results:
        base_label = r.page_size_label.split("_")[0]
        size_groups.setdefault(base_label, []).append(r)

    for label, group in sorted(size_groups.items()):
        # Find best hit rate config
        best_hit = max(group, key=lambda r: r.hit_rate)
        # Find most memory-efficient (best hit rate per MiB)
        efficiency = [(r, r.hit_rate / max(1, r.cache_size_mib)) for r in group]
        best_efficient = max(efficiency, key=lambda x: x[1])[0]

        recs.append({
            "page_size": label,
            "best_hit_rate_config": {
                "cache_mib": best_hit.cache_size_mib,
                "hit_rate": best_hit.hit_rate,
                "evictions": best_hit.evictions,
            },
            "most_efficient_config": {
                "cache_mib": best_efficient.cache_size_mib,
                "hit_rate": best_efficient.hit_rate,
                "evictions": best_efficient.evictions,
            },
        })

    return recs


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_markdown_report(report: CacheProfileReport) -> str:
    """Format the profile report as markdown.

    Parameters
    ----------
    report : CacheProfileReport
        Complete profiling report.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# Page Cache Memory Profile Report",
        "",
        f"**Timestamp**: {report.timestamp}",
        f"**Pages per config**: {report.page_count}",
        "",
        "## Results",
        "",
        "| Cache (MiB) | Page Size | Hits | Misses | Evictions | Hit Rate | Avg Put (us) | Avg Get (us) |",
        "|-------------|-----------|------|--------|-----------|----------|-------------|-------------|",
    ]

    for r in report.results:
        lines.append(
            f"| {r['cache_size_mib']} | {r['page_size_label']} "
            f"| {r['hits']} | {r['misses']} | {r['evictions']} "
            f"| {r['hit_rate']:.4f} "
            f"| {r['avg_put_us']:.2f} | {r['avg_get_us']:.2f} |"
        )

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    for rec in report.recommendations:
        bc = rec["best_hit_rate_config"]
        lines.append(
            f"- **{rec['page_size']}** pages: "
            f"Use {bc['cache_mib']} MiB cache "
            f"(hit rate: {bc['hit_rate']:.4f}, evictions: {bc['evictions']})"
        )

    lines.append("")
    return "\n".join(lines)


def format_console_report(report: CacheProfileReport) -> str:
    """Format the profile report for console output.

    Parameters
    ----------
    report : CacheProfileReport
        Complete profiling report.

    Returns
    -------
    str
        Console-formatted report.
    """
    lines = [
        "",
        "=" * 100,
        "PAGE CACHE MEMORY PROFILE REPORT",
        "=" * 100,
        "",
        f"  Timestamp:      {report.timestamp}",
        f"  Pages/config:   {report.page_count}",
        "",
        "-" * 100,
        f"{'Cache(MiB)':>10} {'Page Size':<18} {'Hits':>6} {'Miss':>6} {'Evict':>6} "
        f"{'HitRate':>8} {'Put(us)':>10} {'Get(us)':>10} {'Mem(MB)':>8}",
        "-" * 100,
    ]

    for r in report.results:
        lines.append(
            f"{r['cache_size_mib']:>10} {r['page_size_label']:<18} "
            f"{r['hits']:>6} {r['misses']:>6} {r['evictions']:>6} "
            f"{r['hit_rate']:>8.4f} {r['avg_put_us']:>10.2f} "
            f"{r['avg_get_us']:>10.2f} {r['memory_delta_mb']:>8.1f}"
        )

    lines.append("-" * 100)
    lines.append("")
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 100)

    for rec in report.recommendations:
        bc = rec["best_hit_rate_config"]
        lines.append(
            f"  {rec['page_size']:<12}: cache={bc['cache_mib']}MiB, "
            f"hit_rate={bc['hit_rate']:.4f}, evictions={bc['evictions']}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for page cache profiling."""
    parser = argparse.ArgumentParser(
        description="Profile page cache memory usage and performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/profile_page_cache.py
  python scripts/profile_page_cache.py --cache-size 256 --page-count 500
  python scripts/profile_page_cache.py --output-dir ./reports
  python scripts/profile_page_cache.py --ttl 30
        """,
    )
    parser.add_argument(
        "--cache-size",
        type=str,
        default=None,
        help="Comma-separated cache sizes in MiB (default: 64,128,256,512)",
    )
    parser.add_argument(
        "--page-count",
        type=int,
        default=200,
        help="Number of pages per configuration (default: 200)",
    )
    parser.add_argument(
        "--ttl",
        type=float,
        default=0.0,
        help="Cache entry TTL in seconds (default: 0 = no expiry)",
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

    cache_sizes = (
        [int(x) for x in args.cache_size.split(",")]
        if args.cache_size
        else None
    )

    logger.info("Running page cache profiling...")

    report = run_full_profile(
        cache_sizes_mib=cache_sizes,
        page_count=args.page_count,
        ttl_seconds=args.ttl,
    )

    # Console output
    print(format_console_report(report))

    # Save reports
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "page_cache_profile.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("JSON report saved to %s", json_path)

        md_path = out_dir / "page_cache_profile.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(report))
        logger.info("Markdown report saved to %s", md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
