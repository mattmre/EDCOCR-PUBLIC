"""GPU kernel fusion benchmarks for gpu_optimization.py.

Benchmarks the BatchPreprocessor and GpuOptimizer with and without
fusion enabled.  Measures latency, memory bandwidth, and throughput
on synthetic image batches.  Falls back gracefully when GPU or heavy
dependencies (torch, numpy) are unavailable.

Usage:
    python scripts/benchmark_gpu_fusion.py
    python scripts/benchmark_gpu_fusion.py --batch-size 16 --iterations 10
    python scripts/benchmark_gpu_fusion.py --output-dir ./reports
    python scripts/benchmark_gpu_fusion.py --image-size 1024x1024
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

from ocr_local.infra.gpu_optimization import (
    BatchPreprocessor,
    FusionConfig,
    FusionStrategy,
    GpuOptimizer,
    OptimizationLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 8
DEFAULT_ITERATIONS = 5
DEFAULT_IMAGE_SIZE = (640, 640)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class FusionBenchmarkResult:
    """Benchmark results for a single fusion configuration."""

    config_label: str = ""
    fusion_strategy: str = "none"
    optimization_level: str = "none"
    fp16_enabled: bool = False
    preprocessing_on_gpu: bool = False
    batch_size: int = 0
    image_size: str = ""
    iterations: int = 0
    # Timing
    avg_batch_latency_ms: float = 0.0
    p95_batch_latency_ms: float = 0.0
    min_batch_latency_ms: float = 0.0
    max_batch_latency_ms: float = 0.0
    avg_per_image_ms: float = 0.0
    throughput_images_per_sec: float = 0.0
    # Memory
    estimated_memory_mb: float = 0.0
    memory_bandwidth_mbps: float = 0.0
    # Optimal batch
    optimal_batch_for_1gb: int = 0
    optimal_batch_for_4gb: int = 0
    optimal_batch_for_8gb: int = 0


@dataclass
class GpuFusionReport:
    """Complete GPU fusion benchmark report."""

    timestamp: str = ""
    gpu_available: bool = False
    numpy_available: bool = False
    torch_available: bool = False
    gpu_capabilities: list = field(default_factory=list)
    recommended_config: dict = field(default_factory=dict)
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


def _check_numpy() -> bool:
    """Check if numpy is available."""
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torch() -> bool:
    """Check if torch with CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _generate_test_images(count: int, size: tuple) -> list:
    """Generate synthetic PIL images for benchmarking.

    Parameters
    ----------
    count : int
        Number of images to generate.
    size : tuple[int, int]
        (width, height) of each image.

    Returns
    -------
    list[PIL.Image.Image]
        Generated test images.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not available; cannot generate test images")
        return []

    images = []
    try:
        import numpy as np
        for i in range(count):
            rng = np.random.RandomState(seed=i)
            arr = rng.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            images.append(Image.fromarray(arr, "RGB"))
    except ImportError:
        # Fallback: solid-color images
        for i in range(count):
            color = ((i * 37) % 256, (i * 73) % 256, (i * 113) % 256)
            images.append(Image.new("RGB", size, color))

    return images


def _percentile(values: list, pct: float) -> float:
    """Compute a percentile from a list."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * pct / 100.0), len(s) - 1)
    return s[idx]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def benchmark_fusion_config(
    config: FusionConfig,
    label: str,
    batch_size: int,
    image_size: tuple,
    iterations: int,
) -> FusionBenchmarkResult:
    """Benchmark a single fusion configuration.

    Parameters
    ----------
    config : FusionConfig
        Fusion configuration to test.
    label : str
        Human-readable label.
    batch_size : int
        Number of images per batch.
    image_size : tuple[int, int]
        (width, height) of test images.
    iterations : int
        Number of benchmark iterations.

    Returns
    -------
    FusionBenchmarkResult
        Benchmark metrics.
    """
    preprocessor = BatchPreprocessor(config)

    images = _generate_test_images(batch_size, image_size)
    if not images:
        logger.warning("No test images generated; skipping %s", label)
        return FusionBenchmarkResult(config_label=label)

    # Warmup
    try:
        preprocessor.preprocess_batch(images, target_size=image_size)
    except Exception as e:
        logger.warning("Warmup failed for %s: %s", label, e)
        return FusionBenchmarkResult(config_label=label)

    # Benchmark iterations
    batch_latencies_ms = []

    for _ in range(iterations):
        start = time.perf_counter()
        preprocessor.preprocess_batch(images, target_size=image_size)
        elapsed_ms = (time.perf_counter() - start) * 1000
        batch_latencies_ms.append(elapsed_ms)

    if not batch_latencies_ms:
        return FusionBenchmarkResult(config_label=label)

    avg_lat = statistics.mean(batch_latencies_ms)
    avg_per_img = avg_lat / max(1, batch_size)
    total_images = batch_size * iterations
    total_time_s = sum(batch_latencies_ms) / 1000
    throughput = total_images / total_time_s if total_time_s > 0 else 0

    # Memory estimation
    estimated_mem = preprocessor.estimate_memory_mb(
        batch_size, image_size[0], image_size[1]
    )

    # Memory bandwidth estimate (bytes processed per second)
    bytes_per_image = image_size[0] * image_size[1] * 3 * 4  # float32
    total_bytes = bytes_per_image * batch_size
    bandwidth_mbps = (total_bytes / (avg_lat / 1000)) / (1024 * 1024) if avg_lat > 0 else 0

    # Optimal batch sizes for various memory budgets
    opt_1gb = preprocessor.get_optimal_batch_size(1024, image_size)
    opt_4gb = preprocessor.get_optimal_batch_size(4096, image_size)
    opt_8gb = preprocessor.get_optimal_batch_size(8192, image_size)

    return FusionBenchmarkResult(
        config_label=label,
        fusion_strategy=config.strategy.value,
        optimization_level=config.level.value,
        fp16_enabled=config.enable_fp16,
        preprocessing_on_gpu=config.preprocessing_on_gpu,
        batch_size=batch_size,
        image_size=f"{image_size[0]}x{image_size[1]}",
        iterations=iterations,
        avg_batch_latency_ms=round(avg_lat, 2),
        p95_batch_latency_ms=round(_percentile(batch_latencies_ms, 95), 2),
        min_batch_latency_ms=round(min(batch_latencies_ms), 2),
        max_batch_latency_ms=round(max(batch_latencies_ms), 2),
        avg_per_image_ms=round(avg_per_img, 2),
        throughput_images_per_sec=round(throughput, 2),
        estimated_memory_mb=round(estimated_mem, 2),
        memory_bandwidth_mbps=round(bandwidth_mbps, 2),
        optimal_batch_for_1gb=opt_1gb,
        optimal_batch_for_4gb=opt_4gb,
        optimal_batch_for_8gb=opt_8gb,
    )


def run_gpu_fusion_benchmark(
    batch_size: int = DEFAULT_BATCH_SIZE,
    iterations: int = DEFAULT_ITERATIONS,
    image_size: tuple = DEFAULT_IMAGE_SIZE,
) -> GpuFusionReport:
    """Run the full GPU fusion benchmark suite.

    Parameters
    ----------
    batch_size : int
        Images per batch.
    iterations : int
        Iterations per configuration.
    image_size : tuple[int, int]
        (width, height) of test images.

    Returns
    -------
    GpuFusionReport
        Complete benchmark report.
    """
    numpy_avail = _check_numpy()
    torch_avail = _check_torch()

    # Detect GPU capabilities
    optimizer = GpuOptimizer()
    caps = optimizer.detect_capabilities()
    recommended = optimizer.recommend_config(caps)

    configs = [
        (
            "Baseline (no fusion)",
            FusionConfig(
                level=OptimizationLevel.NONE,
                strategy=FusionStrategy.NONE,
                max_batch_images=batch_size,
                enable_fp16=False,
                preprocessing_on_gpu=False,
            ),
        ),
        (
            "Preprocess batch (CPU)",
            FusionConfig(
                level=OptimizationLevel.BASIC,
                strategy=FusionStrategy.PREPROCESS_BATCH,
                max_batch_images=batch_size,
                enable_fp16=False,
                preprocessing_on_gpu=False,
            ),
        ),
        (
            "Inference batch (CPU)",
            FusionConfig(
                level=OptimizationLevel.BASIC,
                strategy=FusionStrategy.INFERENCE_BATCH,
                max_batch_images=batch_size,
                enable_fp16=False,
                preprocessing_on_gpu=False,
            ),
        ),
        (
            "Full pipeline (CPU)",
            FusionConfig(
                level=OptimizationLevel.AGGRESSIVE,
                strategy=FusionStrategy.FULL_PIPELINE,
                max_batch_images=batch_size,
                enable_fp16=False,
                preprocessing_on_gpu=False,
            ),
        ),
    ]

    # Add GPU configs if available
    if torch_avail:
        configs.extend([
            (
                "Preprocess batch (GPU)",
                FusionConfig(
                    level=OptimizationLevel.BASIC,
                    strategy=FusionStrategy.PREPROCESS_BATCH,
                    max_batch_images=batch_size,
                    enable_fp16=False,
                    preprocessing_on_gpu=True,
                ),
            ),
            (
                "Full pipeline (GPU, FP16)",
                FusionConfig(
                    level=OptimizationLevel.AGGRESSIVE,
                    strategy=FusionStrategy.FULL_PIPELINE,
                    max_batch_images=batch_size,
                    enable_fp16=True,
                    preprocessing_on_gpu=True,
                    pin_memory=True,
                ),
            ),
        ])

    results = []
    for label, config in configs:
        logger.info("Benchmarking: %s", label)
        result = benchmark_fusion_config(
            config=config,
            label=label,
            batch_size=batch_size,
            image_size=image_size,
            iterations=iterations,
        )
        results.append(result)

    # Recommendations
    recommendations = _generate_recommendations(results, caps, recommended)

    return GpuFusionReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        gpu_available=torch_avail,
        numpy_available=numpy_avail,
        torch_available=torch_avail,
        gpu_capabilities=[asdict(c) for c in caps] if caps else [],
        recommended_config={
            "level": recommended.level.value,
            "strategy": recommended.strategy.value,
            "max_batch_images": recommended.max_batch_images,
            "enable_fp16": recommended.enable_fp16,
            "preprocessing_on_gpu": recommended.preprocessing_on_gpu,
        },
        results=[asdict(r) for r in results],
        recommendations=recommendations,
    )


def _generate_recommendations(
    results: list,
    caps: list,
    recommended: FusionConfig,
) -> list:
    """Generate fusion optimization recommendations.

    Parameters
    ----------
    results : list[FusionBenchmarkResult]
        Benchmark results.
    caps : list[GpuCapability]
        Detected GPU capabilities.
    recommended : FusionConfig
        Auto-recommended configuration.

    Returns
    -------
    list[str]
        Recommendations.
    """
    recs = []

    # Find fastest config
    valid = [r for r in results if r.throughput_images_per_sec > 0]
    if not valid:
        recs.append("No valid benchmark results. Ensure Pillow is installed.")
        return recs

    fastest = max(valid, key=lambda r: r.throughput_images_per_sec)
    baseline = next((r for r in valid if "Baseline" in r.config_label), valid[0])

    speedup = (
        fastest.throughput_images_per_sec / baseline.throughput_images_per_sec
        if baseline.throughput_images_per_sec > 0 else 1.0
    )

    recs.append(
        f"Fastest config: '{fastest.config_label}' "
        f"at {fastest.throughput_images_per_sec:.1f} images/sec "
        f"({speedup:.2f}x vs baseline)."
    )

    if caps:
        best_gpu = max(caps, key=lambda c: c.memory_total_mb)
        recs.append(
            f"GPU detected: {best_gpu.name} ({best_gpu.memory_total_mb}MB). "
            f"Recommended strategy: {recommended.strategy.value}."
        )
        if best_gpu.supports_fp16:
            recs.append(
                "FP16 supported. Enable GPU_ENABLE_FP16=true for "
                "potential 1.5-2x throughput improvement."
            )
    else:
        recs.append(
            "No GPU detected. Using CPU-only preprocessing. "
            "Install torch with CUDA for GPU acceleration."
        )

    # Memory guidance
    recs.append(
        f"Optimal batch sizes: "
        f"1GB VRAM -> {fastest.optimal_batch_for_1gb}, "
        f"4GB VRAM -> {fastest.optimal_batch_for_4gb}, "
        f"8GB VRAM -> {fastest.optimal_batch_for_8gb}."
    )

    return recs


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_markdown_report(report: GpuFusionReport) -> str:
    """Format the GPU fusion report as markdown.

    Parameters
    ----------
    report : GpuFusionReport
        Complete report.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# GPU Fusion Benchmark Report",
        "",
        f"**Timestamp**: {report.timestamp}",
        f"**GPU available**: {report.gpu_available}",
        f"**Numpy available**: {report.numpy_available}",
        "",
        "## Results",
        "",
        "| Config | Strategy | Batch | Avg Latency (ms) | Throughput (img/s) | Est. Memory (MB) |",
        "|--------|----------|-------|-------------------|--------------------|-------------------|",
    ]

    for r in report.results:
        lines.append(
            f"| {r['config_label']} | {r['fusion_strategy']} | {r['batch_size']} "
            f"| {r['avg_batch_latency_ms']:.2f} "
            f"| {r['throughput_images_per_sec']:.2f} "
            f"| {r['estimated_memory_mb']:.2f} |"
        )

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    for rec in report.recommendations:
        lines.append(f"- {rec}")

    lines.append("")
    return "\n".join(lines)


def format_console_report(report: GpuFusionReport) -> str:
    """Format the GPU fusion report for console output.

    Parameters
    ----------
    report : GpuFusionReport
        Complete report.

    Returns
    -------
    str
        Console-formatted report.
    """
    lines = [
        "",
        "=" * 100,
        "GPU FUSION BENCHMARK REPORT",
        "=" * 100,
        "",
        f"  GPU available:    {report.gpu_available}",
        f"  Numpy available:  {report.numpy_available}",
        f"  Torch+CUDA:       {report.torch_available}",
    ]

    if report.gpu_capabilities:
        for cap in report.gpu_capabilities:
            lines.append(
                f"  GPU: {cap['name']} (CC {cap['compute_capability']}, "
                f"{cap['memory_total_mb']}MB, FP16={cap['supports_fp16']})"
            )

    lines.extend([
        "",
        "-" * 100,
        f"{'Config':<30} {'Strategy':<18} {'Batch':>5} {'AvgLat(ms)':>12} "
        f"{'Throughput':>12} {'Memory(MB)':>12} {'BW(MB/s)':>10}",
        "-" * 100,
    ])

    for r in report.results:
        lines.append(
            f"{r['config_label']:<30} {r['fusion_strategy']:<18} "
            f"{r['batch_size']:>5} {r['avg_batch_latency_ms']:>12.2f} "
            f"{r['throughput_images_per_sec']:>12.2f} "
            f"{r['estimated_memory_mb']:>12.2f} "
            f"{r['memory_bandwidth_mbps']:>10.2f}"
        )

    lines.append("-" * 100)
    lines.append("")
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 100)

    for rec in report.recommendations:
        lines.append(f"  - {rec}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for GPU fusion benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark GPU kernel fusion for OCR preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_gpu_fusion.py
  python scripts/benchmark_gpu_fusion.py --batch-size 16 --iterations 10
  python scripts/benchmark_gpu_fusion.py --image-size 1024x1024
  python scripts/benchmark_gpu_fusion.py --output-dir ./reports
        """,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Images per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Benchmark iterations per config (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default=f"{DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]}",
        help=f"Image dimensions WxH (default: {DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]})",
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

    # Parse image size
    try:
        w, h = args.image_size.split("x")
        image_size = (int(w), int(h))
    except (ValueError, AttributeError):
        logger.error("Invalid image size format. Use WxH (e.g., 640x640)")
        return 1

    logger.info("Running GPU fusion benchmarks...")

    report = run_gpu_fusion_benchmark(
        batch_size=args.batch_size,
        iterations=args.iterations,
        image_size=image_size,
    )

    # Console output
    print(format_console_report(report))

    # Save reports
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "gpu_fusion_benchmark.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("JSON report saved to %s", json_path)

        md_path = out_dir / "gpu_fusion_benchmark.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(report))
        logger.info("Markdown report saved to %s", md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
