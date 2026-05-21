"""Benchmark signature detection precision, recall, and F1.

Generates a synthetic test corpus of images with and without signatures,
runs the signature_verification module against them, and reports binary
detection metrics (presence_detected vs ground truth).

Usage:
    python scripts/benchmark_signature.py generate --output-dir benchmark_data/signatures --count 100
    python scripts/benchmark_signature.py run --corpus-dir benchmark_data/signatures
    python scripts/benchmark_signature.py report --results-file results.json

The ``generate`` subcommand creates labeled PNG images using Pillow:
  - images WITH drawn signature strokes (curves, loops)
  - images WITHOUT signatures (clean typed text blocks)
  - images with noise/stamps (false-positive traps)

The ``run`` subcommand invokes ``signature_verification.analyze_signature_page``
on each image and compares ``presence_detected`` to the ground-truth label.

The ``report`` subcommand reads a JSON results file and prints a Markdown
report with a confusion matrix.

Requires: signature_verification.py, Pillow (from EDCOCR project root)
"""

import argparse
import datetime
import json
import logging
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class SignatureBenchmarkResult:
    """Complete benchmark result for signature detection."""

    total_samples: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    avg_inference_time_ms: float = 0.0
    min_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    p95_inference_time_ms: float = 0.0
    dataset: str = ""
    model: str = "signature_verification.analyze_signature_page"
    target_f1: float = 60.0
    passed: bool = False
    timestamp: str = ""
    per_category: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthetic corpus generation
# ---------------------------------------------------------------------------

# Category labels used in the ground-truth manifest
CATEGORY_SIGNATURE = "signature"
CATEGORY_NO_SIGNATURE = "no_signature"
CATEGORY_NOISE_STAMP = "noise_stamp"

_GROUND_TRUTH_FILE = "ground_truth.json"


def _draw_signature_stroke(draw: ImageDraw.ImageDraw, rng: random.Random,
                           region: tuple[int, int, int, int]) -> None:
    """Draw a hand-drawn-looking signature stroke inside *region*."""
    x1, y1, x2, y2 = region
    width = x2 - x1
    height = y2 - y1

    # Generate a series of points that form a cursive-like path
    num_points = rng.randint(12, 25)
    points = []
    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        x = x1 + int(t * width)
        # Create wavy vertical offset (simulates cursive loops)
        y_mid = (y1 + y2) // 2
        wave = int(height * 0.35 * math.sin(t * math.pi * rng.uniform(2.5, 5.0)))
        jitter_y = rng.randint(-int(height * 0.1), int(height * 0.1))
        y = max(y1, min(y2, y_mid + wave + jitter_y))
        points.append((x, y))

    # Draw the stroke path
    pen_width = rng.randint(2, 4)
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="black", width=pen_width)

    # Add a small loop or flourish at start/end
    if rng.random() < 0.6 and len(points) >= 2:
        cx, cy = points[-1]
        r = rng.randint(4, 10)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=1)


def _draw_typed_text_block(draw: ImageDraw.ImageDraw, rng: random.Random,
                           region: tuple[int, int, int, int]) -> None:
    """Draw clean horizontal text lines (printed-looking) in *region*."""
    x1, y1, x2, y2 = region
    line_height = 14
    y = y1 + 4
    words = [
        "Invoice", "Date:", "2026-01-15", "Amount:", "$1,234.56",
        "Payment", "Terms:", "Net", "30", "Days",
        "Bill To:", "Acme Corporation", "123 Main St",
        "Reference:", "PO-98765", "Thank you",
    ]
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    while y + line_height < y2:
        num_words = rng.randint(2, 5)
        line_text = " ".join(rng.choices(words, k=num_words))
        draw.text((x1 + 4, y), line_text, fill="black", font=font)
        y += line_height


def _draw_noise_stamp(draw: ImageDraw.ImageDraw, rng: random.Random,
                      region: tuple[int, int, int, int]) -> None:
    """Draw stamp-like circular marks and noise — a false-positive trap."""
    x1, y1, x2, y2 = region
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = min(x2 - x1, y2 - y1) // 3

    # Outer circle (stamp border)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=3)
    # Inner circle
    inner_r = int(r * 0.65)
    draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
                 outline="black", width=1)
    # Stamp text
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((cx - 20, cy - 6), "APPROVED", fill="black", font=font)

    # Random noise dots
    for _ in range(rng.randint(30, 80)):
        nx = rng.randint(x1, x2)
        ny = rng.randint(y1, y2)
        draw.point((nx, ny), fill="black")


def generate_corpus(output_dir: str, count: int = 100, seed: int = 42) -> dict:
    """Generate a synthetic signature detection corpus.

    Creates *count* labeled PNG images and a ``ground_truth.json`` manifest.

    Parameters
    ----------
    output_dir : str
        Directory to write images and manifest into.
    count : int
        Total number of images to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        The ground-truth manifest mapping filename to category label.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    manifest: dict[str, str] = {}

    # Distribute roughly equally across three categories
    sig_count = count // 3
    no_sig_count = count // 3
    noise_count = count - sig_count - no_sig_count

    plan: list[tuple[int, str]] = []
    for i in range(sig_count):
        plan.append((i, CATEGORY_SIGNATURE))
    for i in range(no_sig_count):
        plan.append((sig_count + i, CATEGORY_NO_SIGNATURE))
    for i in range(noise_count):
        plan.append((sig_count + no_sig_count + i, CATEGORY_NOISE_STAMP))

    rng.shuffle(plan)

    for idx, (sample_id, category) in enumerate(plan):
        img_w = rng.randint(400, 600)
        img_h = rng.randint(300, 500)
        img = Image.new("L", (img_w, img_h), color=255)
        draw = ImageDraw.Draw(img)

        # All images get some typed text in the upper portion
        _draw_typed_text_block(draw, rng, (10, 10, img_w - 10, img_h // 2))

        # Signature region in lower half
        sig_region = (
            img_w // 6,
            img_h // 2 + 20,
            img_w - img_w // 6,
            img_h - 20,
        )

        if category == CATEGORY_SIGNATURE:
            _draw_signature_stroke(draw, rng, sig_region)
        elif category == CATEGORY_NOISE_STAMP:
            _draw_noise_stamp(draw, rng, sig_region)
        # CATEGORY_NO_SIGNATURE: leave lower half blank (already white)

        fname = f"sample_{idx:04d}.png"
        img.save(str(out / fname))
        manifest[fname] = category

    # Write manifest
    manifest_path = out / _GROUND_TRUTH_FILE
    with open(str(manifest_path), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info("Generated %d images in %s", len(manifest), output_dir)
    return manifest


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(corpus_dir: str) -> tuple[SignatureBenchmarkResult, list[dict]]:
    """Run signature detection against a corpus directory.

    Reads ``ground_truth.json`` from *corpus_dir*, loads each image, and
    calls ``analyze_signature_page`` with a synthetic OCR keyword line so
    the module has a candidate region to evaluate.

    Parameters
    ----------
    corpus_dir : str
        Directory containing PNG images and ``ground_truth.json``.

    Returns
    -------
    tuple[SignatureBenchmarkResult, list[dict]]
        Benchmark result and per-sample detail records.
    """
    try:
        from signature_verification import analyze_signature_page
    except ImportError:
        logger.error(
            "Cannot import signature_verification module. "
            "Ensure signature_verification.py is on sys.path."
        )
        return SignatureBenchmarkResult(
            dataset=str(corpus_dir),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ), []

    corpus = Path(corpus_dir)
    manifest_path = corpus / _GROUND_TRUTH_FILE
    if not manifest_path.exists():
        logger.error("Ground-truth manifest not found: %s", manifest_path)
        return SignatureBenchmarkResult(
            dataset=str(corpus_dir),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ), []

    with open(str(manifest_path), encoding="utf-8") as fh:
        manifest: dict[str, str] = json.load(fh)

    predictions: list[bool] = []
    ground_truth: list[bool] = []
    categories: list[str] = []
    timings: list[float] = []
    details: list[dict] = []

    for fname, category in sorted(manifest.items()):
        img_path = corpus / fname
        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            continue

        img = Image.open(str(img_path)).convert("L")
        img_array = np.array(img)
        img_h, img_w = img_array.shape[:2]

        # Provide a synthetic OCR keyword line so the module creates a
        # candidate region covering the lower half (where we drew signatures)
        sig_region_bbox = [
            [img_w // 6, img_h // 2 + 10],
            [img_w - img_w // 6, img_h // 2 + 10],
            [img_w - img_w // 6, img_h // 2 + 30],
            [img_w // 6, img_h // 2 + 30],
        ]
        paddle_lines = [
            ("Signature: _______________", sig_region_bbox, 0.85),
        ]

        gt_positive = category == CATEGORY_SIGNATURE

        start = time.perf_counter()
        page_result = analyze_signature_page(
            image=img,
            page_num=1,
            structure_data=None,
            paddle_lines=paddle_lines,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        predicted = page_result.presence_detected
        predictions.append(predicted)
        ground_truth.append(gt_positive)
        categories.append(category)
        timings.append(elapsed_ms)

        details.append({
            "file": fname,
            "category": category,
            "ground_truth": gt_positive,
            "predicted": predicted,
            "correct": predicted == gt_positive,
            "inference_ms": round(elapsed_ms, 2),
        })

    metrics = compute_metrics(predictions, ground_truth)

    # Per-category breakdown
    per_cat: dict[str, dict] = {}
    unique_cats = sorted(set(categories))
    for cat in unique_cats:
        cat_preds = [p for p, c in zip(predictions, categories) if c == cat]
        cat_gt = [g for g, c in zip(ground_truth, categories) if c == cat]
        cat_m = compute_metrics(cat_preds, cat_gt)
        per_cat[cat] = cat_m

    result = SignatureBenchmarkResult(
        total_samples=len(predictions),
        true_positives=metrics["tp"],
        false_positives=metrics["fp"],
        true_negatives=metrics["tn"],
        false_negatives=metrics["fn"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        accuracy=metrics["accuracy"],
        avg_inference_time_ms=round(statistics.mean(timings), 2) if timings else 0.0,
        min_inference_time_ms=round(min(timings), 2) if timings else 0.0,
        max_inference_time_ms=round(max(timings), 2) if timings else 0.0,
        p95_inference_time_ms=round(
            sorted(timings)[min(int(len(timings) * 0.95), len(timings) - 1)], 2
        ) if len(timings) >= 5 else (round(max(timings), 2) if timings else 0.0),
        dataset=str(corpus_dir),
        target_f1=60.0,
        passed=metrics["f1"] * 100 >= 60.0,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        per_category=per_cat,
    )

    return result, details


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(predictions: list[bool], ground_truth: list[bool]) -> dict:
    """Compute binary classification metrics.

    Parameters
    ----------
    predictions : list[bool]
        Predicted presence flags.
    ground_truth : list[bool]
        True presence flags.

    Returns
    -------
    dict
        Dictionary with tp, fp, tn, fn, precision, recall, f1, accuracy.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truth "
            f"({len(ground_truth)}) must have the same length"
        )

    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(metrics: dict, result: SignatureBenchmarkResult | None = None) -> str:
    """Generate a Markdown report with confusion matrix.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from ``compute_metrics``.
    result : SignatureBenchmarkResult, optional
        Full benchmark result for additional context.

    Returns
    -------
    str
        Markdown-formatted report string.
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("SIGNATURE DETECTION BENCHMARK")
    lines.append("=" * 80)
    lines.append("")

    if result:
        lines.append(f"  Dataset:          {result.dataset}")
        lines.append(f"  Samples:          {result.total_samples}")
        lines.append(f"  Target F1:        {result.target_f1:.1f}%")
        lines.append(f"  Result:           {'PASS' if result.passed else 'FAIL'}")
        lines.append("")

    lines.append("DETECTION METRICS")
    lines.append("-" * 80)
    lines.append(f"  Precision:        {metrics['precision']:.4f}")
    lines.append(f"  Recall:           {metrics['recall']:.4f}")
    lines.append(f"  F1:               {metrics['f1']:.4f}")
    lines.append(f"  Accuracy:         {metrics['accuracy']:.4f}")
    lines.append("")

    lines.append("CONFUSION MATRIX")
    lines.append("-" * 80)
    lines.append("                    Predicted Positive   Predicted Negative")
    lines.append(f"  Actual Positive   TP = {metrics['tp']:<16d} FN = {metrics['fn']}")
    lines.append(f"  Actual Negative   FP = {metrics['fp']:<16d} TN = {metrics['tn']}")
    lines.append("")

    if result and result.per_category:
        lines.append("PER-CATEGORY BREAKDOWN")
        lines.append("-" * 80)
        lines.append(
            f"{'Category':<20} {'Precision':>10} {'Recall':>10} "
            f"{'F1':>10} {'Accuracy':>10}"
        )
        lines.append("-" * 80)
        for cat in sorted(result.per_category.keys()):
            m = result.per_category[cat]
            lines.append(
                f"{cat:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1']:>10.4f} {m['accuracy']:>10.4f}"
            )
        lines.append("")

    if result:
        lines.append("INFERENCE TIMING")
        lines.append("-" * 80)
        lines.append(f"  Avg:              {result.avg_inference_time_ms:.2f} ms")
        lines.append(f"  Min:              {result.min_inference_time_ms:.2f} ms")
        lines.append(f"  Max:              {result.max_inference_time_ms:.2f} ms")
        lines.append(f"  P95:              {result.p95_inference_time_ms:.2f} ms")
        lines.append(f"  Timestamp:        {result.timestamp}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point for signature benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark signature detection precision/recall/F1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_signature.py generate --output-dir benchmark_data/signatures --count 100
  python scripts/benchmark_signature.py run --corpus-dir benchmark_data/signatures
  python scripts/benchmark_signature.py report --results-file results.json
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # -- generate --
    gen_parser = subparsers.add_parser(
        "generate", help="Generate synthetic test corpus"
    )
    gen_parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write generated images and manifest",
    )
    gen_parser.add_argument(
        "--count", type=int, default=100,
        help="Number of test images to generate (default: 100)",
    )
    gen_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # -- run --
    run_parser = subparsers.add_parser(
        "run", help="Run benchmark against a corpus"
    )
    run_parser.add_argument(
        "--corpus-dir", required=True,
        help="Directory containing images and ground_truth.json",
    )
    run_parser.add_argument(
        "--output", type=str,
        help="Output JSON path for structured results",
    )
    run_parser.add_argument(
        "--target-f1", type=float, default=60.0,
        help="Target F1 percentage for pass/fail (default: 60.0)",
    )

    # -- report --
    report_parser = subparsers.add_parser(
        "report", help="Generate report from results file"
    )
    report_parser.add_argument(
        "--results-file", required=True,
        help="Path to JSON results file from 'run' subcommand",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "generate":
        manifest = generate_corpus(
            output_dir=args.output_dir,
            count=args.count,
            seed=args.seed,
        )
        print(f"Generated {len(manifest)} images in {args.output_dir}")
        return 0

    elif args.command == "run":
        result, details = run_benchmark(corpus_dir=args.corpus_dir)
        result.target_f1 = args.target_f1
        result.passed = result.f1 * 100 >= args.target_f1

        metrics = {
            "tp": result.true_positives,
            "fp": result.false_positives,
            "tn": result.true_negatives,
            "fn": result.false_negatives,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "accuracy": result.accuracy,
        }
        report = generate_report(metrics, result)
        print(report)

        if args.output:
            output_data = asdict(result)
            output_data["details"] = details
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            logger.info("Results saved to %s", args.output)

        return 0 if result.passed else 1

    elif args.command == "report":
        results_path = Path(args.results_file)
        if not results_path.exists():
            logger.error("Results file not found: %s", args.results_file)
            return 1

        with open(str(results_path), encoding="utf-8") as fh:
            data = json.load(fh)

        metrics = {
            "tp": data.get("true_positives", 0),
            "fp": data.get("false_positives", 0),
            "tn": data.get("true_negatives", 0),
            "fn": data.get("false_negatives", 0),
            "precision": data.get("precision", 0.0),
            "recall": data.get("recall", 0.0),
            "f1": data.get("f1", 0.0),
            "accuracy": data.get("accuracy", 0.0),
        }

        result = SignatureBenchmarkResult(
            total_samples=data.get("total_samples", 0),
            true_positives=metrics["tp"],
            false_positives=metrics["fp"],
            true_negatives=metrics["tn"],
            false_negatives=metrics["fn"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            accuracy=metrics["accuracy"],
            avg_inference_time_ms=data.get("avg_inference_time_ms", 0.0),
            min_inference_time_ms=data.get("min_inference_time_ms", 0.0),
            max_inference_time_ms=data.get("max_inference_time_ms", 0.0),
            p95_inference_time_ms=data.get("p95_inference_time_ms", 0.0),
            dataset=data.get("dataset", ""),
            target_f1=data.get("target_f1", 60.0),
            passed=data.get("passed", False),
            timestamp=data.get("timestamp", ""),
            per_category=data.get("per_category", {}),
        )

        report = generate_report(metrics, result)
        print(report)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
