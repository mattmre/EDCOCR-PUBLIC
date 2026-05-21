"""Confidence calibration pipeline for production prediction data.

Ingests production prediction data (JSON files with predictions and
ground truth), computes reliability diagrams and Expected Calibration
Error (ECE), applies calibration methods (temperature scaling, Platt
scaling, isotonic regression), and generates before/after calibration
reports.

Wraps ``layoutlm_calibration.py`` for production use.

All heavy ML imports are lazy -- the module is importable and testable
without GPU dependencies.

Usage::

    python scripts/calibrate_confidence.py \\
        --data-dir ./calibration_data \\
        --method all \\
        --output-dir ./calibration_results

Data format:
    Each JSON file in ``--data-dir`` should contain::

        {
          "predictions": [
            {"label": "B-DATE", "confidence": 0.95, "logit": 2.5},
            ...
          ],
          "ground_truth": ["B-DATE", "O", "B-AMOUNT", ...]
        }
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CalibrationData:
    """Container for calibration input data."""

    predictions: List[Dict[str, Any]] = field(default_factory=list)
    ground_truth: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    logits: List[float] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)


@dataclass
class CalibrationReport:
    """Result report from a calibration run."""

    method: str = ""
    num_samples: int = 0
    ece_before: float = 0.0
    ece_after: float = 0.0
    ece_improvement: float = 0.0
    reliability_before: Dict[str, Any] = field(default_factory=dict)
    reliability_after: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_calibration_data(data_dir: str) -> CalibrationData:
    """Load prediction data from a directory of JSON files.

    Each JSON file should have ``predictions`` (list of dicts with
    ``label``, ``confidence``, optionally ``logit``) and
    ``ground_truth`` (list of label strings).

    Args:
        data_dir: Path to directory of calibration data files.

    Returns:
        Aggregated CalibrationData.
    """
    path = Path(data_dir)
    if not path.is_dir():
        logger.error("Data directory does not exist: %s", data_dir)
        return CalibrationData()

    all_preds: List[Dict[str, Any]] = []
    all_gt: List[str] = []

    for json_file in sorted(path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            preds = data.get("predictions", [])
            gt = data.get("ground_truth", [])

            n = min(len(preds), len(gt))
            all_preds.extend(preds[:n])
            all_gt.extend(gt[:n])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load %s: %s", json_file, exc)

    if not all_preds:
        logger.warning("No prediction data found in %s", data_dir)
        return CalibrationData()

    # Extract parallel arrays
    confidences: List[float] = []
    logits: List[float] = []
    labels: List[int] = []

    for pred, true_label in zip(all_preds, all_gt):
        conf = float(pred.get("confidence", 0.0))
        logit = float(pred.get("logit", conf))
        pred_label = pred.get("label", "")
        correct = 1 if pred_label == true_label else 0

        confidences.append(conf)
        logits.append(logit)
        labels.append(correct)

    return CalibrationData(
        predictions=all_preds,
        ground_truth=all_gt,
        confidences=confidences,
        logits=logits,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# ECE computation (standalone, no dependency on calibration module)
# ---------------------------------------------------------------------------


def compute_ece(
    predictions: List[float],
    labels: List[int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    ECE measures the weighted average gap between predicted confidence
    and actual accuracy across n_bins equal-width bins in [0, 1].

    Args:
        predictions: Predicted confidence scores (0-1).
        labels: Binary correctness labels (0 or 1).
        n_bins: Number of calibration bins.

    Returns:
        ECE value (0.0 = perfectly calibrated).
    """
    if not predictions or not labels:
        return 0.0

    n = min(len(predictions), len(labels))
    predictions = predictions[:n]
    labels = labels[:n]

    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for idx in range(n):
        p = max(0.0, min(1.0, predictions[idx]))
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append(idx)

    ece = 0.0
    for bin_indices in bins:
        if not bin_indices:
            continue
        bin_confs = [predictions[i] for i in bin_indices]
        bin_labels = [labels[i] for i in bin_indices]
        avg_conf = sum(bin_confs) / len(bin_confs)
        avg_acc = sum(bin_labels) / len(bin_labels)
        ece += (len(bin_indices) / n) * abs(avg_acc - avg_conf)

    return ece


def compute_reliability_bins(
    predictions: List[float],
    labels: List[int],
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    """Compute reliability diagram bin data.

    Args:
        predictions: Confidence scores.
        labels: Binary correctness labels.
        n_bins: Number of bins.

    Returns:
        List of bin dicts with avg_confidence, avg_accuracy, count.
    """
    if not predictions or not labels:
        return []

    n = min(len(predictions), len(labels))
    predictions = predictions[:n]
    labels = labels[:n]

    bin_width = 1.0 / n_bins
    all_bins: List[List[int]] = [[] for _ in range(n_bins)]

    for idx in range(n):
        p = max(0.0, min(1.0, predictions[idx]))
        bin_idx = min(int(p * n_bins), n_bins - 1)
        all_bins[bin_idx].append(idx)

    result: List[Dict[str, Any]] = []
    for b in range(n_bins):
        bin_start = b * bin_width
        bin_end = (b + 1) * bin_width
        indices = all_bins[b]

        if indices:
            avg_conf = sum(predictions[i] for i in indices) / len(indices)
            avg_acc = sum(labels[i] for i in indices) / len(indices)
        else:
            avg_conf = 0.0
            avg_acc = 0.0

        result.append({
            "bin_start": round(bin_start, 4),
            "bin_end": round(bin_end, 4),
            "avg_confidence": round(avg_conf, 6),
            "avg_accuracy": round(avg_acc, 6),
            "count": len(indices),
        })

    return result


# ---------------------------------------------------------------------------
# Calibration methods (standalone implementations)
# ---------------------------------------------------------------------------


def apply_temperature_scaling(
    confidences: List[float],
    labels: List[int],
) -> Tuple[List[float], float]:
    """Apply temperature scaling calibration.

    Finds the optimal temperature T that minimizes NLL on the
    provided data via grid search.

    Args:
        confidences: Raw confidence scores.
        labels: Binary correctness labels.

    Returns:
        Tuple of (calibrated_confidences, optimal_temperature).
    """
    import math

    if not confidences or not labels:
        return confidences, 1.0

    # Grid search for optimal temperature
    best_t = 1.0
    best_nll = float("inf")

    for t_candidate in [v / 10.0 for v in range(1, 101)]:
        nll = 0.0
        for conf, label in zip(confidences, labels):
            # Convert confidence to logit, scale, convert back
            conf_clipped = max(1e-7, min(1 - 1e-7, conf))
            logit = math.log(conf_clipped / (1 - conf_clipped))
            p = 1.0 / (1.0 + math.exp(-logit / t_candidate))
            p_target = p if label == 1 else (1.0 - p)
            p_target = max(p_target, 1e-15)
            nll -= math.log(p_target)

        if nll < best_nll:
            best_nll = nll
            best_t = t_candidate

    # Apply optimal temperature
    calibrated = []
    for conf in confidences:
        conf_clipped = max(1e-7, min(1 - 1e-7, conf))
        logit = math.log(conf_clipped / (1 - conf_clipped))
        p = 1.0 / (1.0 + math.exp(-logit / best_t))
        calibrated.append(p)

    return calibrated, best_t


def apply_platt_scaling(
    confidences: List[float],
    labels: List[int],
) -> Tuple[List[float], float, float]:
    """Apply Platt scaling calibration.

    Fits sigmoid parameters a, b via gradient descent.

    Args:
        confidences: Raw confidence scores.
        labels: Binary correctness labels.

    Returns:
        Tuple of (calibrated_confidences, a, b).
    """
    import math

    if not confidences or not labels:
        return confidences, 1.0, 0.0

    # Convert to logits
    logits = []
    for c in confidences:
        c_clipped = max(1e-7, min(1 - 1e-7, c))
        logits.append(math.log(c_clipped / (1 - c_clipped)))

    # Gradient descent for Platt parameters
    a = 1.0
    b = 0.0
    lr = 0.01
    n = len(logits)

    for _ in range(1000):
        grad_a = 0.0
        grad_b = 0.0
        for logit, label in zip(logits, labels):
            z = a * logit + b
            if z >= 0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                exp_z = math.exp(z)
                p = exp_z / (1.0 + exp_z)
            err = p - label
            grad_a += err * logit
            grad_b += err
        a -= lr * grad_a / n
        b -= lr * grad_b / n

    # Apply calibration
    calibrated = []
    for logit in logits:
        z = a * logit + b
        if z >= 0:
            p = 1.0 / (1.0 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            p = exp_z / (1.0 + exp_z)
        calibrated.append(p)

    return calibrated, a, b


def apply_isotonic_regression(
    confidences: List[float],
    labels: List[int],
) -> Tuple[List[float], bool]:
    """Apply isotonic regression calibration.

    Requires sklearn.  Falls back to identity if unavailable.

    Args:
        confidences: Raw confidence scores.
        labels: Binary correctness labels.

    Returns:
        Tuple of (calibrated_confidences, success_flag).
    """
    if not confidences or not labels:
        return confidences, False

    try:
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(confidences, labels)
        calibrated = ir.predict(confidences).tolist()
        return calibrated, True
    except ImportError:
        logger.warning(
            "sklearn not available -- isotonic regression skipped."
        )
        return list(confidences), False


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_calibration_pipeline(
    data_dir: str,
    method: str = "all",
    output_dir: str = "./calibration_results",
    n_bins: int = 10,
) -> List[CalibrationReport]:
    """Run the calibration pipeline.

    Args:
        data_dir: Directory with prediction JSON files.
        method: Calibration method (``"temperature"``, ``"platt"``,
                ``"isotonic"``, or ``"all"``).
        output_dir: Output directory for reports.
        n_bins: Number of bins for reliability diagrams.

    Returns:
        List of CalibrationReport instances (one per method).
    """
    data = load_calibration_data(data_dir)
    if not data.confidences:
        logger.error("No calibration data loaded.")
        return []

    logger.info(
        "Loaded %d predictions for calibration.", len(data.confidences)
    )

    # Compute before-calibration metrics
    ece_before = compute_ece(data.confidences, data.labels, n_bins)
    reliability_before = compute_reliability_bins(
        data.confidences, data.labels, n_bins,
    )

    methods_to_run = []
    if method == "all":
        methods_to_run = ["temperature", "platt", "isotonic"]
    else:
        methods_to_run = [method]

    reports: List[CalibrationReport] = []

    for m in methods_to_run:
        logger.info("Running %s calibration...", m)

        calibrated: List[float] = []
        params: Dict[str, Any] = {}

        if m == "temperature":
            calibrated, temp = apply_temperature_scaling(
                data.confidences, data.labels,
            )
            params = {"temperature": round(temp, 4)}

        elif m == "platt":
            calibrated, a, b = apply_platt_scaling(
                data.confidences, data.labels,
            )
            params = {"platt_a": round(a, 6), "platt_b": round(b, 6)}

        elif m == "isotonic":
            calibrated, success = apply_isotonic_regression(
                data.confidences, data.labels,
            )
            params = {"fitted": success}

        else:
            logger.warning("Unknown method %r, skipping.", m)
            continue

        # Compute after-calibration metrics
        ece_after = compute_ece(calibrated, data.labels, n_bins)
        reliability_after = compute_reliability_bins(
            calibrated, data.labels, n_bins,
        )

        report = CalibrationReport(
            method=m,
            num_samples=len(data.confidences),
            ece_before=round(ece_before, 6),
            ece_after=round(ece_after, 6),
            ece_improvement=round(ece_before - ece_after, 6),
            reliability_before={
                "bins": reliability_before,
                "ece": round(ece_before, 6),
            },
            reliability_after={
                "bins": reliability_after,
                "ece": round(ece_after, 6),
            },
            parameters=params,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        reports.append(report)

    # Save reports
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for report in reports:
        json_path = out / f"calibration_{report.method}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2)
        logger.info("Calibration report saved to %s", json_path)

    # Summary report
    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "num_samples": len(data.confidences),
        "ece_before": round(ece_before, 6),
        "methods": [
            {
                "method": r.method,
                "ece_after": r.ece_after,
                "ece_improvement": r.ece_improvement,
                "parameters": r.parameters,
            }
            for r in reports
        ],
        "best_method": (
            min(reports, key=lambda r: r.ece_after).method
            if reports
            else "none"
        ),
    }

    summary_path = out / "calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Calibrate confidence scores from production predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory of prediction JSON files.",
    )
    parser.add_argument(
        "--method", default="all",
        choices=["temperature", "platt", "isotonic", "all"],
        help="Calibration method to apply.",
    )
    parser.add_argument(
        "--output-dir", default="./calibration_results",
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of bins for reliability diagrams.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for confidence calibration."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    reports = run_calibration_pipeline(
        data_dir=args.data_dir,
        method=args.method,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
    )

    if not reports:
        print("\nNo calibration results generated.")
        return 1

    print("\n=== Confidence Calibration Complete ===")
    for r in reports:
        print(f"\n  Method: {r.method}")
        print(f"    ECE before: {r.ece_before:.6f}")
        print(f"    ECE after:  {r.ece_after:.6f}")
        print(f"    Improvement: {r.ece_improvement:.6f}")
        print(f"    Params: {r.parameters}")

    best = min(reports, key=lambda r: r.ece_after)
    print(f"\n  Best method: {best.method} (ECE={best.ece_after:.6f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
