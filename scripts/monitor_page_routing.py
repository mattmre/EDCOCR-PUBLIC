"""Quality monitoring tool for page routing (ENABLE_PAGE_ROUTING).

Compares OCR quality with and without smart routing enabled.  Tracks
which processing backend handles which pages and why, measures
per-backend accuracy and latency, and generates a quality comparison
report in JSON and markdown.

Usage:
    python scripts/monitor_page_routing.py
    python scripts/monitor_page_routing.py --sample-count 200
    python scripts/monitor_page_routing.py --output-dir ./reports
    python scripts/monitor_page_routing.py --default-target cpu_tesseract
"""

import argparse
import datetime
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ocr_local.infra.page_routing import (
    PageFeatures,
    PageRouter,
    RoutingDecision,
    RoutingTarget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Simulated per-backend quality scores (higher = better)
_BACKEND_QUALITY = {
    RoutingTarget.GPU_PADDLE: 0.95,
    RoutingTarget.GPU_TESSERACT: 0.85,
    RoutingTarget.CPU_PADDLE: 0.92,
    RoutingTarget.CPU_TESSERACT: 0.82,
    RoutingTarget.CPU_ONNX: 0.90,
    RoutingTarget.SKIP: 0.0,
}

# Complexity-adjusted quality penalty (higher complexity -> lower quality)
_COMPLEXITY_PENALTY = 0.15


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class BackendStats:
    """Per-backend aggregated statistics."""

    backend: str = ""
    pages_routed: int = 0
    avg_estimated_duration_ms: float = 0.0
    avg_quality_score: float = 0.0
    avg_complexity: float = 0.0
    reasons: list = field(default_factory=list)


@dataclass
class RoutingComparisonResult:
    """Comparison between smart routing and default routing."""

    mode: str = ""  # "smart" or "default"
    total_pages: int = 0
    avg_quality_score: float = 0.0
    avg_estimated_duration_ms: float = 0.0
    total_estimated_duration_ms: float = 0.0
    backend_distribution: dict = field(default_factory=dict)
    backend_stats: list = field(default_factory=list)
    pages_skipped: int = 0


@dataclass
class RoutingMonitorReport:
    """Complete routing quality monitor report."""

    timestamp: str = ""
    sample_count: int = 0
    smart_routing: dict = field(default_factory=dict)
    default_routing: dict = field(default_factory=dict)
    quality_improvement_pct: float = 0.0
    latency_reduction_pct: float = 0.0
    routing_decisions: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic page generation
# ---------------------------------------------------------------------------


def generate_sample_pages(count: int, seed: int = 42) -> list:
    """Generate a diverse set of sample PageFeatures.

    Creates pages with varying complexity, languages, table/image/handwriting
    presence to exercise all routing rules.

    Parameters
    ----------
    count : int
        Number of pages to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[PageFeatures]
        Generated page feature sets.
    """
    import random
    rng = random.Random(seed)

    languages = ["en", "fr", "de", "ja", "zh", "ar", "hi", "ko"]
    pages = []

    for i in range(count):
        # Create diverse pages
        roll = rng.random()
        if roll < 0.05:
            # Tiny page (should be skipped)
            width = rng.randint(20, 80)
            height = rng.randint(20, 80)
            complexity = 0.0
        elif roll < 0.15:
            # Handwritten page
            width = rng.randint(2000, 3000)
            height = rng.randint(2500, 3600)
            complexity = rng.uniform(0.5, 0.9)
        elif roll < 0.30:
            # Table-heavy page
            width = rng.randint(2400, 2600)
            height = rng.randint(3300, 3600)
            complexity = rng.uniform(0.4, 0.8)
        elif roll < 0.45:
            # Very simple page (low complexity)
            width = rng.randint(800, 1200)
            height = rng.randint(1000, 1400)
            complexity = rng.uniform(0.0, 0.15)
        elif roll < 0.60:
            # Complex page
            width = rng.randint(3000, 5000)
            height = rng.randint(4000, 7000)
            complexity = rng.uniform(0.8, 1.0)
        else:
            # Normal page
            width = rng.randint(2000, 3000)
            height = rng.randint(2800, 3600)
            complexity = rng.uniform(0.2, 0.7)

        features = PageFeatures(
            page_number=i + 1,
            width=width,
            height=height,
            dpi=rng.choice([200, 300, 400]),
            file_size_bytes=rng.randint(50_000, 5_000_000),
            estimated_text_density=rng.uniform(0.1, 0.9),
            has_tables=(roll >= 0.15 and roll < 0.30),
            has_images=rng.random() < 0.3,
            is_handwritten=(roll >= 0.05 and roll < 0.15),
            language=rng.choice(languages),
            complexity_score=round(complexity, 4),
        )
        pages.append(features)

    return pages


# ---------------------------------------------------------------------------
# Quality simulation
# ---------------------------------------------------------------------------


def simulate_quality(
    decision: RoutingDecision,
    features: PageFeatures,
    seed: int = 0,
) -> float:
    """Simulate an OCR quality score for a routing decision.

    The simulation assigns a base quality per backend, then adjusts
    for page complexity.  This provides a deterministic but realistic
    approximation for comparing routing strategies.

    Parameters
    ----------
    decision : RoutingDecision
        Routing decision.
    features : PageFeatures
        Page features.
    seed : int
        Noise seed.

    Returns
    -------
    float
        Simulated quality score in [0.0, 1.0].
    """
    import random
    rng = random.Random(seed + features.page_number)

    base_quality = _BACKEND_QUALITY.get(decision.target, 0.85)

    # Penalty for complex pages
    penalty = features.complexity_score * _COMPLEXITY_PENALTY

    # Small random noise
    noise = rng.gauss(0, 0.02)

    # Handwritten pages get extra quality boost on GPU_PADDLE
    bonus = 0.0
    if features.is_handwritten and decision.target == RoutingTarget.GPU_PADDLE:
        bonus = 0.05

    quality = max(0.0, min(1.0, base_quality - penalty + noise + bonus))
    return round(quality, 4)


# ---------------------------------------------------------------------------
# Routing evaluation
# ---------------------------------------------------------------------------


def evaluate_routing(
    pages: list,
    router: PageRouter,
    mode_label: str,
) -> RoutingComparisonResult:
    """Route pages through a router and collect statistics.

    Parameters
    ----------
    pages : list[PageFeatures]
        Pages to route.
    router : PageRouter
        Router instance.
    mode_label : str
        Label for this routing mode ("smart" or "default").

    Returns
    -------
    RoutingComparisonResult
        Aggregated routing statistics.
    """
    decisions = router.route_batch(pages)
    backend_groups = {}
    quality_scores = []
    duration_estimates = []
    skipped = 0

    for features, decision in zip(pages, decisions):
        if decision.target == RoutingTarget.SKIP:
            skipped += 1
            continue

        quality = simulate_quality(decision, features)
        quality_scores.append(quality)
        duration_estimates.append(decision.estimated_duration_ms)

        key = decision.target.value
        if key not in backend_groups:
            backend_groups[key] = {
                "pages": [],
                "qualities": [],
                "durations": [],
                "complexities": [],
                "reasons": [],
            }
        backend_groups[key]["pages"].append(features.page_number)
        backend_groups[key]["qualities"].append(quality)
        backend_groups[key]["durations"].append(decision.estimated_duration_ms)
        backend_groups[key]["complexities"].append(features.complexity_score)
        if decision.reason and decision.reason not in backend_groups[key]["reasons"]:
            backend_groups[key]["reasons"].append(decision.reason)

    # Build per-backend stats
    backend_stats = []
    distribution = {}
    for backend, data in sorted(backend_groups.items()):
        count = len(data["pages"])
        distribution[backend] = count
        backend_stats.append(BackendStats(
            backend=backend,
            pages_routed=count,
            avg_estimated_duration_ms=round(statistics.mean(data["durations"]), 2),
            avg_quality_score=round(statistics.mean(data["qualities"]), 4),
            avg_complexity=round(statistics.mean(data["complexities"]), 4),
            reasons=data["reasons"][:5],  # Cap reasons list
        ))

    total_dur = sum(duration_estimates) if duration_estimates else 0.0

    return RoutingComparisonResult(
        mode=mode_label,
        total_pages=len(pages),
        avg_quality_score=round(statistics.mean(quality_scores), 4) if quality_scores else 0,
        avg_estimated_duration_ms=round(statistics.mean(duration_estimates), 2) if duration_estimates else 0,
        total_estimated_duration_ms=round(total_dur, 2),
        backend_distribution=distribution,
        backend_stats=[asdict(s) for s in backend_stats],
        pages_skipped=skipped,
    )


def run_routing_monitor(
    sample_count: int = 100,
    default_target: str = "gpu_paddle",
    seed: int = 42,
) -> RoutingMonitorReport:
    """Run the full routing quality monitor.

    Parameters
    ----------
    sample_count : int
        Number of pages to evaluate.
    default_target : str
        Default routing target for baseline comparison.
    seed : int
        Random seed.

    Returns
    -------
    RoutingMonitorReport
        Complete comparison report.
    """
    target_map = {t.value: t for t in RoutingTarget}
    default = target_map.get(default_target, RoutingTarget.GPU_PADDLE)

    pages = generate_sample_pages(sample_count, seed=seed)

    # Smart routing (uses built-in rules)
    smart_router = PageRouter()
    smart_result = evaluate_routing(pages, smart_router, "smart")

    # Default routing (no rules, always routes to default)
    default_router = PageRouter(rules=[], default_target=default)
    default_result = evaluate_routing(pages, default_router, "default")

    # Quality improvement
    quality_improvement = 0.0
    if default_result.avg_quality_score > 0:
        quality_improvement = (
            (smart_result.avg_quality_score - default_result.avg_quality_score)
            / default_result.avg_quality_score * 100
        )

    # Latency reduction
    latency_reduction = 0.0
    if default_result.total_estimated_duration_ms > 0:
        latency_reduction = (
            (default_result.total_estimated_duration_ms - smart_result.total_estimated_duration_ms)
            / default_result.total_estimated_duration_ms * 100
        )

    # Collect per-page decisions for the detail report
    decisions_detail = []
    smart_decisions = smart_router.route_batch(pages)
    for features, decision in zip(pages, smart_decisions):
        decisions_detail.append({
            "page": features.page_number,
            "complexity": features.complexity_score,
            "is_handwritten": features.is_handwritten,
            "has_tables": features.has_tables,
            "target": decision.target.value,
            "reason": decision.reason,
            "estimated_duration_ms": decision.estimated_duration_ms,
        })

    # Recommendations
    recommendations = _generate_recommendations(
        smart_result, default_result, quality_improvement, latency_reduction
    )

    return RoutingMonitorReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        sample_count=sample_count,
        smart_routing=asdict(smart_result),
        default_routing=asdict(default_result),
        quality_improvement_pct=round(quality_improvement, 2),
        latency_reduction_pct=round(latency_reduction, 2),
        routing_decisions=decisions_detail,
        recommendations=recommendations,
    )


def _generate_recommendations(
    smart: RoutingComparisonResult,
    default: RoutingComparisonResult,
    quality_pct: float,
    latency_pct: float,
) -> list:
    """Generate routing recommendations from comparison results.

    Parameters
    ----------
    smart : RoutingComparisonResult
        Smart routing results.
    default : RoutingComparisonResult
        Default routing results.
    quality_pct : float
        Quality improvement percentage.
    latency_pct : float
        Latency reduction percentage.

    Returns
    -------
    list[str]
        Recommendations.
    """
    recs = []

    if quality_pct > 1:
        recs.append(
            f"Smart routing improves quality by {quality_pct:.1f}%% -- "
            "recommended for production."
        )
    elif quality_pct > 0:
        recs.append(
            f"Smart routing provides marginal quality improvement ({quality_pct:.1f}%%). "
            "Consider for mixed-complexity workloads."
        )
    else:
        recs.append(
            "Smart routing does not improve quality for this workload. "
            "Default routing may be sufficient."
        )

    if latency_pct > 10:
        recs.append(
            f"Smart routing reduces estimated latency by {latency_pct:.1f}%% "
            "via offloading simple pages to faster backends."
        )

    if smart.pages_skipped > 0:
        recs.append(
            f"{smart.pages_skipped} tiny pages were skipped by smart routing, "
            "saving processing time."
        )

    # Check backend distribution balance
    dist = smart.backend_distribution
    total = sum(dist.values()) or 1
    for backend, count in dist.items():
        pct = count / total * 100
        if pct > 80:
            recs.append(
                f"Backend '{backend}' handles {pct:.0f}%% of pages. "
                "Consider adding more routing rules for better distribution."
            )

    return recs


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_markdown_report(report: RoutingMonitorReport) -> str:
    """Format the routing monitor report as markdown.

    Parameters
    ----------
    report : RoutingMonitorReport
        Complete report.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# Page Routing Quality Monitor Report",
        "",
        f"**Timestamp**: {report.timestamp}",
        f"**Sample count**: {report.sample_count} pages",
        f"**Quality improvement**: {report.quality_improvement_pct:+.2f}%",
        f"**Latency reduction**: {report.latency_reduction_pct:+.2f}%",
        "",
        "## Smart Routing",
        "",
        f"- Avg quality: {report.smart_routing.get('avg_quality_score', 0):.4f}",
        f"- Avg latency: {report.smart_routing.get('avg_estimated_duration_ms', 0):.2f} ms",
        f"- Pages skipped: {report.smart_routing.get('pages_skipped', 0)}",
        "",
        "### Backend Distribution (Smart)",
        "",
        "| Backend | Pages |",
        "|---------|-------|",
    ]

    for backend, count in report.smart_routing.get("backend_distribution", {}).items():
        lines.append(f"| {backend} | {count} |")

    lines.extend([
        "",
        "## Default Routing",
        "",
        f"- Avg quality: {report.default_routing.get('avg_quality_score', 0):.4f}",
        f"- Avg latency: {report.default_routing.get('avg_estimated_duration_ms', 0):.2f} ms",
        "",
        "## Recommendations",
        "",
    ])

    for rec in report.recommendations:
        lines.append(f"- {rec}")

    lines.append("")
    return "\n".join(lines)


def format_console_report(report: RoutingMonitorReport) -> str:
    """Format the routing monitor report for console output.

    Parameters
    ----------
    report : RoutingMonitorReport
        Complete report.

    Returns
    -------
    str
        Console-formatted report.
    """
    lines = [
        "",
        "=" * 80,
        "PAGE ROUTING QUALITY MONITOR",
        "=" * 80,
        "",
        f"  Sample count:         {report.sample_count} pages",
        f"  Quality improvement:  {report.quality_improvement_pct:+.2f}%",
        f"  Latency reduction:    {report.latency_reduction_pct:+.2f}%",
        "",
        "-" * 80,
        "SMART ROUTING",
        "-" * 80,
        f"  Avg quality:    {report.smart_routing.get('avg_quality_score', 0):.4f}",
        f"  Avg latency:    {report.smart_routing.get('avg_estimated_duration_ms', 0):.2f} ms",
        f"  Total latency:  {report.smart_routing.get('total_estimated_duration_ms', 0):.0f} ms",
        f"  Skipped:        {report.smart_routing.get('pages_skipped', 0)}",
        "",
        "  Backend Distribution:",
    ]

    for backend, count in report.smart_routing.get("backend_distribution", {}).items():
        lines.append(f"    {backend:<20}: {count}")

    lines.extend([
        "",
        "-" * 80,
        "DEFAULT ROUTING",
        "-" * 80,
        f"  Avg quality:    {report.default_routing.get('avg_quality_score', 0):.4f}",
        f"  Avg latency:    {report.default_routing.get('avg_estimated_duration_ms', 0):.2f} ms",
        f"  Total latency:  {report.default_routing.get('total_estimated_duration_ms', 0):.0f} ms",
        "",
        "-" * 80,
        "PER-BACKEND STATS (Smart Routing)",
        "-" * 80,
    ])

    for bs in report.smart_routing.get("backend_stats", []):
        lines.append(
            f"  {bs['backend']:<20}: {bs['pages_routed']:>4} pages, "
            f"quality={bs['avg_quality_score']:.4f}, "
            f"latency={bs['avg_estimated_duration_ms']:.1f}ms, "
            f"complexity={bs['avg_complexity']:.4f}"
        )

    lines.extend([
        "",
        "-" * 80,
        "RECOMMENDATIONS",
        "-" * 80,
    ])

    for rec in report.recommendations:
        lines.append(f"  - {rec}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for page routing quality monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor page routing quality and performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/monitor_page_routing.py
  python scripts/monitor_page_routing.py --sample-count 200
  python scripts/monitor_page_routing.py --default-target cpu_tesseract
  python scripts/monitor_page_routing.py --output-dir ./reports
        """,
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Number of pages to evaluate (default: 100)",
    )
    parser.add_argument(
        "--default-target",
        type=str,
        default="gpu_paddle",
        choices=[t.value for t in RoutingTarget],
        help="Default routing target for baseline (default: gpu_paddle)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
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

    logger.info("Running page routing quality monitor...")

    report = run_routing_monitor(
        sample_count=args.sample_count,
        default_target=args.default_target,
        seed=args.seed,
    )

    # Console output
    print(format_console_report(report))

    # Save reports
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "page_routing_monitor.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("JSON report saved to %s", json_path)

        md_path = out_dir / "page_routing_monitor.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(report))
        logger.info("Markdown report saved to %s", md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
