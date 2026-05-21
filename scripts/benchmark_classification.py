"""Benchmark document classification accuracy.

Measures classification accuracy against labeled datasets or synthetic
documents. Reports per-class precision, recall, F1, and inference timing.

Usage:
    python scripts/benchmark_classification.py --synthetic
    python scripts/benchmark_classification.py --synthetic --sample-size 200
    python scripts/benchmark_classification.py --dataset-dir /path/to/rvl-cdip --sample-size 100
    python scripts/benchmark_classification.py --synthetic --output results.json
    python scripts/benchmark_classification.py --synthetic --target-accuracy 95.0

The --synthetic flag generates documents with known classifications for CI
testing without external datasets. The --dataset-dir flag loads real labeled
documents from a directory structured as: <dataset-dir>/<class_name>/*.pdf|*.png

Requires: classification.py (from EDCOCR project root)
"""

import argparse
import datetime
import json
import logging
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure project root is on sys.path so classification can be imported
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class ClassMetrics:
    """Per-class precision, recall, and F1 score."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


@dataclass
class ClassificationBenchmarkResult:
    """Complete benchmark result for document classification."""

    total_documents: int = 0
    correct: int = 0
    incorrect: int = 0
    accuracy: float = 0.0
    per_class_metrics: dict = field(default_factory=dict)
    avg_inference_time_ms: float = 0.0
    min_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    p95_inference_time_ms: float = 0.0
    dataset: str = ""
    model: str = ""
    target_accuracy: float = 95.0
    passed: bool = False
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_per_class_metrics(
    predictions: list,
    ground_truth: list,
    classes: list,
) -> dict:
    """Compute precision, recall, F1 per class.

    Parameters
    ----------
    predictions : list[str]
        Predicted class labels.
    ground_truth : list[str]
        True class labels.
    classes : list[str]
        All possible class names.

    Returns
    -------
    dict[str, dict]
        Mapping of class name to {precision, recall, f1, support}.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truth "
            f"({len(ground_truth)}) must have the same length"
        )

    results = {}
    for cls in classes:
        tp = sum(
            1 for p, g in zip(predictions, ground_truth)
            if p == cls and g == cls
        )
        fp = sum(
            1 for p, g in zip(predictions, ground_truth)
            if p == cls and g != cls
        )
        fn = sum(
            1 for p, g in zip(predictions, ground_truth)
            if p != cls and g == cls
        )
        support = sum(1 for g in ground_truth if g == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[cls] = asdict(ClassMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            support=support,
        ))

    return results


def compute_accuracy(predictions: list, ground_truth: list) -> float:
    """Compute overall accuracy.

    Parameters
    ----------
    predictions : list[str]
        Predicted class labels.
    ground_truth : list[str]
        True class labels.

    Returns
    -------
    float
        Accuracy as a value between 0.0 and 1.0.
    """
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


# ---------------------------------------------------------------------------
# Synthetic document generation
# ---------------------------------------------------------------------------

# Representative text patterns per document class (used for synthetic generation)
_SYNTHETIC_TEXT_TEMPLATES = {
    "invoice": (
        "INVOICE #INV-{num}\nBill To: {company}\n"
        "Date: 2026-01-{day}\nAmount Due: ${amount}\n"
        "Payment Terms: Net 30\nSubtotal: ${subtotal}\nTax: ${tax}\nTotal: ${amount}"
    ),
    "contract": (
        "AGREEMENT\nThis contract is entered into between {party_a} and {party_b}.\n"
        "WHEREAS the parties agree to the following terms and conditions:\n"
        "Section 1. Term and Termination\nSection 2. Confidentiality\n"
        "Signature: _______________\nDate: 2026-01-{day}"
    ),
    "letter": (
        "Dear {recipient},\n\nThank you for your recent correspondence regarding "
        "{subject}. We are writing to inform you that your request has been reviewed.\n\n"
        "Please do not hesitate to contact us if you have any questions.\n\n"
        "Sincerely,\n{sender}"
    ),
    "form": (
        "APPLICATION FORM\nFull Name: _______________\nDate of Birth: ___/___/______\n"
        "Address: _______________\nPhone: _______________\n"
        "Email: _______________\nSignature: _______________\nDate: ___/___/______"
    ),
    "report": (
        "ANNUAL REPORT {year}\nExecutive Summary\n\n"
        "This report presents the findings from our comprehensive analysis "
        "of operations during fiscal year {year}.\n\n"
        "Key Findings:\n- Revenue increased by {pct}%\n- Costs reduced by {cost_pct}%\n"
        "Conclusion and Recommendations"
    ),
    "memo": (
        "MEMORANDUM\nTO: {recipient}\nFROM: {sender}\n"
        "DATE: 2026-01-{day}\nRE: {subject}\n\n"
        "This memo is to inform you of the following updates regarding the project."
    ),
    "receipt": (
        "RECEIPT\nStore: {company}\nDate: 2026-01-{day}\nTransaction #{num}\n"
        "Item 1: Widget x{qty} - ${price}\nItem 2: Gadget x1 - ${price2}\n"
        "Subtotal: ${subtotal}\nTax: ${tax}\nTotal: ${amount}\n"
        "Payment Method: Credit Card"
    ),
    "handwritten": (
        "Dear diary,\nToday was an interesting day at the office. "
        "I met with the team to discuss the project timeline. "
        "Notes: review budget proposal, call vendor, schedule meeting."
    ),
    "photograph": (
        "IMG_{num}.jpg\nCamera: Canon EOS R5\nDate: 2026-01-{day}\n"
        "Resolution: 8192x5464\nExposure: 1/250s f/4.0 ISO 400"
    ),
    "other": (
        "Document {num}\nMiscellaneous content that does not fit neatly "
        "into any specific category. Contains mixed text and references."
    ),
}

_NAMES = ["Acme Corp", "Global Industries", "Smith & Associates", "TechVentures"]
_PEOPLE = ["John Smith", "Jane Doe", "Alex Johnson", "Maria Garcia"]
_SUBJECTS = ["Q1 Budget", "Project Alpha", "Policy Update", "Vendor Selection"]


def _generate_synthetic_text(doc_type: str, seed: int) -> str:
    """Generate synthetic text for a given document type.

    Parameters
    ----------
    doc_type : str
        Document type from DOCUMENT_TYPES.
    seed : int
        Random seed for reproducible generation.

    Returns
    -------
    str
        Generated text matching the document type patterns.
    """
    rng = random.Random(seed)
    template = _SYNTHETIC_TEXT_TEMPLATES.get(doc_type, _SYNTHETIC_TEXT_TEMPLATES["other"])

    return template.format(
        num=rng.randint(1000, 9999),
        company=rng.choice(_NAMES),
        party_a=rng.choice(_NAMES),
        party_b=rng.choice(_NAMES),
        recipient=rng.choice(_PEOPLE),
        sender=rng.choice(_PEOPLE),
        subject=rng.choice(_SUBJECTS),
        day=rng.randint(1, 28),
        amount=rng.randint(100, 9999),
        subtotal=rng.randint(80, 8000),
        tax=rng.randint(5, 500),
        price=rng.randint(10, 200),
        price2=rng.randint(10, 200),
        qty=rng.randint(1, 10),
        year=rng.choice([2024, 2025, 2026]),
        pct=rng.randint(1, 30),
        cost_pct=rng.randint(1, 15),
    )


def run_synthetic_benchmark(
    sample_size: int = 100,
    target_accuracy: float = 95.0,
) -> ClassificationBenchmarkResult:
    """Run benchmark with synthetic documents (for CI/testing).

    Generates documents with known classifications, runs the classification
    module against them, and compares predictions to ground truth.

    Parameters
    ----------
    sample_size : int
        Number of synthetic documents to generate.
    target_accuracy : float
        Target accuracy percentage for pass/fail determination.

    Returns
    -------
    ClassificationBenchmarkResult
        Benchmark results including accuracy and per-class metrics.
    """
    try:
        from classification import DOCUMENT_TYPES, classify_page_by_text
    except ImportError:
        logger.error(
            "Cannot import classification module. "
            "Ensure classification.py is on sys.path."
        )
        return ClassificationBenchmarkResult(
            dataset="synthetic",
            model="classification.classify_page_by_text",
            target_accuracy=target_accuracy,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    doc_types = [t for t in DOCUMENT_TYPES if t != "other"]
    if not doc_types:
        doc_types = ["invoice", "contract", "letter", "form", "report"]

    ground_truth = []
    predictions = []
    timings = []

    docs_per_type = max(1, sample_size // len(doc_types))
    remainder = sample_size - docs_per_type * len(doc_types)

    sample_idx = 0
    for dtype in doc_types:
        count = docs_per_type + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1

        for i in range(count):
            text = _generate_synthetic_text(dtype, seed=sample_idx * 1000 + i)
            ground_truth.append(dtype)

            start = time.perf_counter()
            result = classify_page_by_text(text, page_num=0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)

            # Extract predicted type from classification result
            if isinstance(result, dict):
                predicted = result.get("predicted_type", "other")
            elif isinstance(result, str):
                predicted = result
            elif hasattr(result, "predicted_type"):
                predicted = result.predicted_type
            else:
                predicted = "other"

            predictions.append(predicted)
            sample_idx += 1

    # Compute metrics
    accuracy = compute_accuracy(predictions, ground_truth)
    all_classes = sorted(set(ground_truth) | set(predictions))
    per_class = compute_per_class_metrics(predictions, ground_truth, all_classes)

    return ClassificationBenchmarkResult(
        total_documents=len(predictions),
        correct=sum(1 for p, g in zip(predictions, ground_truth) if p == g),
        incorrect=sum(1 for p, g in zip(predictions, ground_truth) if p != g),
        accuracy=round(accuracy * 100, 2),
        per_class_metrics=per_class,
        avg_inference_time_ms=round(statistics.mean(timings), 2) if timings else 0.0,
        min_inference_time_ms=round(min(timings), 2) if timings else 0.0,
        max_inference_time_ms=round(max(timings), 2) if timings else 0.0,
        p95_inference_time_ms=round(
            sorted(timings)[min(int(len(timings) * 0.95), len(timings) - 1)], 2
        ) if len(timings) >= 5 else (round(max(timings), 2) if timings else 0.0),
        dataset="synthetic",
        model="classification.classify_page_by_text",
        target_accuracy=target_accuracy,
        passed=accuracy * 100 >= target_accuracy,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    )


def run_dataset_benchmark(
    dataset_dir: str,
    sample_size: int = None,
    target_accuracy: float = 95.0,
) -> ClassificationBenchmarkResult:
    """Run benchmark against a labeled dataset directory.

    Expects directory structure: <dataset_dir>/<class_name>/<files...>
    where class_name matches one of the DOCUMENT_TYPES.

    Parameters
    ----------
    dataset_dir : str
        Path to labeled dataset root directory.
    sample_size : int, optional
        Maximum documents to sample per class. None = use all.
    target_accuracy : float
        Target accuracy percentage.

    Returns
    -------
    ClassificationBenchmarkResult
        Benchmark results.
    """
    try:
        from classification import classify_page_by_text
    except ImportError:
        logger.error("Cannot import classification module.")
        return ClassificationBenchmarkResult(
            dataset=str(dataset_dir),
            model="classification.classify_page_by_text",
            target_accuracy=target_accuracy,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        logger.error("Dataset directory does not exist: %s", dataset_dir)
        return ClassificationBenchmarkResult(
            dataset=str(dataset_dir),
            model="classification.classify_page_by_text",
            target_accuracy=target_accuracy,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    ground_truth = []
    predictions = []
    timings = []

    supported_exts = {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower()

        files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_exts
        ]

        if sample_size is not None:
            files = files[:sample_size]

        for fpath in files:
            try:
                # For .txt files, read text directly
                if fpath.suffix.lower() == ".txt":
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                else:
                    # For non-text files, use filename as minimal text
                    # (real benchmark would run OCR first)
                    logger.warning(
                        "Non-text file %s: using filename for classification "
                        "(run OCR first for accurate results)", fpath.name
                    )
                    text = fpath.stem.replace("_", " ").replace("-", " ")

                ground_truth.append(class_name)

                start = time.perf_counter()
                result = classify_page_by_text(text, page_num=0)
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings.append(elapsed_ms)

                if isinstance(result, dict):
                    predicted = result.get("document_type", "other")
                elif isinstance(result, str):
                    predicted = result
                elif hasattr(result, "document_type"):
                    predicted = result.document_type
                else:
                    predicted = "other"

                predictions.append(predicted)

            except Exception as exc:
                logger.warning("Failed to classify %s: %s", fpath, exc)

    if not predictions:
        logger.error("No documents were successfully classified from %s", dataset_dir)
        return ClassificationBenchmarkResult(
            dataset=str(dataset_dir),
            model="classification.classify_page_by_text",
            target_accuracy=target_accuracy,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    accuracy = compute_accuracy(predictions, ground_truth)
    all_classes = sorted(set(ground_truth) | set(predictions))
    per_class = compute_per_class_metrics(predictions, ground_truth, all_classes)

    return ClassificationBenchmarkResult(
        total_documents=len(predictions),
        correct=sum(1 for p, g in zip(predictions, ground_truth) if p == g),
        incorrect=sum(1 for p, g in zip(predictions, ground_truth) if p != g),
        accuracy=round(accuracy * 100, 2),
        per_class_metrics=per_class,
        avg_inference_time_ms=round(statistics.mean(timings), 2) if timings else 0.0,
        min_inference_time_ms=round(min(timings), 2) if timings else 0.0,
        max_inference_time_ms=round(max(timings), 2) if timings else 0.0,
        p95_inference_time_ms=round(
            sorted(timings)[min(int(len(timings) * 0.95), len(timings) - 1)], 2
        ) if len(timings) >= 5 else (round(max(timings), 2) if timings else 0.0),
        dataset=str(dataset_dir),
        model="classification.classify_page_by_text",
        target_accuracy=target_accuracy,
        passed=accuracy * 100 >= target_accuracy,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_classification_report(result: ClassificationBenchmarkResult) -> str:
    """Format benchmark result as a human-readable report.

    Parameters
    ----------
    result : ClassificationBenchmarkResult
        Benchmark result to format.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("DOCUMENT CLASSIFICATION BENCHMARK")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Dataset:          {result.dataset}")
    lines.append(f"  Model:            {result.model}")
    lines.append(f"  Documents:        {result.total_documents}")
    lines.append(f"  Correct:          {result.correct}")
    lines.append(f"  Incorrect:        {result.incorrect}")
    lines.append(f"  Accuracy:         {result.accuracy:.2f}%")
    lines.append(f"  Target:           {result.target_accuracy:.1f}%")
    lines.append(f"  Result:           {'PASS' if result.passed else 'FAIL'}")
    lines.append(f"  Avg Inference:    {result.avg_inference_time_ms:.2f} ms")
    lines.append(f"  P95 Inference:    {result.p95_inference_time_ms:.2f} ms")
    lines.append(f"  Timestamp:        {result.timestamp}")
    lines.append("")

    if result.per_class_metrics:
        lines.append("PER-CLASS METRICS")
        lines.append("-" * 80)
        lines.append(
            f"{'Class':<20} {'Precision':>10} {'Recall':>10} "
            f"{'F1':>10} {'Support':>10}"
        )
        lines.append("-" * 80)

        for cls in sorted(result.per_class_metrics.keys()):
            m = result.per_class_metrics[cls]
            lines.append(
                f"{cls:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1']:>10.4f} {m['support']:>10d}"
            )
        lines.append("-" * 80)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for classification benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark document classification accuracy (Phase 6 KPI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_classification.py --synthetic
  python scripts/benchmark_classification.py --synthetic --sample-size 200
  python scripts/benchmark_classification.py --dataset-dir /data/rvl-cdip
  python scripts/benchmark_classification.py --synthetic --output results.json
  python scripts/benchmark_classification.py --synthetic --target-accuracy 90.0
        """,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to labeled dataset (subdir per class)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic documents (no external dataset needed)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of documents to benchmark (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON path for structured results",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=95.0,
        help="Target accuracy percentage (default: 95.0)",
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

    if not args.synthetic and not args.dataset_dir:
        parser.error("Either --synthetic or --dataset-dir is required")

    # Run benchmark
    if args.synthetic:
        logger.info(
            "Running synthetic classification benchmark (%d documents)...",
            args.sample_size,
        )
        result = run_synthetic_benchmark(
            sample_size=args.sample_size,
            target_accuracy=args.target_accuracy,
        )
    else:
        logger.info(
            "Running dataset classification benchmark from %s...",
            args.dataset_dir,
        )
        result = run_dataset_benchmark(
            dataset_dir=args.dataset_dir,
            sample_size=args.sample_size,
            target_accuracy=args.target_accuracy,
        )

    # Display report
    report = format_classification_report(result)
    print(report)

    # Save JSON if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info("Results saved to %s", args.output)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
