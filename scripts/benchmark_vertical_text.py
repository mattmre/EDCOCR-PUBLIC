"""Benchmark CJK vertical text detection and reading order accuracy.

Measures vertical text detection accuracy, reading order correctness
(Kendall's tau), column grouping F1, and rotation accuracy against a
labeled corpus. Supports real corpus evaluation, dry-run validation,
synthetic benchmark, and synthetic image generation.

Usage:
    python scripts/benchmark_vertical_text.py --corpus-dir /path/to/corpus
    python scripts/benchmark_vertical_text.py --corpus-dir /path/to/corpus --dry-run
    python scripts/benchmark_vertical_text.py --synthetic
    python scripts/benchmark_vertical_text.py --synthetic --sample-size 50
    python scripts/benchmark_vertical_text.py --synthetic --output results.json
    python scripts/benchmark_vertical_text.py --synthetic --output-md report.md
    python scripts/benchmark_vertical_text.py --generate-synthetic 20 --output-dir /tmp/corpus

Expected corpus structure:
    corpus/
      vertical_only/     # Pages with pure vertical CJK text
      mixed_layout/      # Pages with mixed vertical + horizontal text
      horizontal_only/   # Pages with only horizontal text (negative cases)

Each test case: <name>.png|.jpg|.tiff + <name>.expected.json sidecar.

Sidecar JSON schema:
    {
      "direction": "vertical" | "horizontal" | "mixed",
      "expected_order": ["text_1", "text_2", ...],
      "expected_columns": 3,
      "vertical_regions": [
        {"text": "...", "reading_order": 0, "bbox": [x1, y1, x2, y2]}
      ],
      "ocr_lines": [
        {"text": "...", "box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "confidence": 0.95}
      ],
      "page_width": 1000,
      "expected_rotation": "ccw"
    }

Requires: vertical_text.py (from EDCOCR project root)
"""

import argparse
import datetime
import json
import logging
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure project root is on sys.path so vertical_text can be imported
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import of vertical_text module
# ---------------------------------------------------------------------------

try:
    from vertical_text import (
        classify_page_text_direction,
        group_vertical_columns,
        is_vertical_text_box,
        sort_mixed_reading_order,
    )

    _VERTICAL_TEXT_AVAILABLE = True
except ImportError:
    logger.warning(
        "vertical_text module not available. "
        "Only dry-run and synthetic modes will work."
    )
    _VERTICAL_TEXT_AVAILABLE = False

# Guarded PIL import for synthetic image generation
try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
CORPUS_SUBDIRS = ("vertical_only", "mixed_layout", "horizontal_only")
SIDECAR_SUFFIX = ".expected.json"


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class CategoryMetrics:
    """Metrics for a single corpus category (e.g., vertical_only)."""

    category: str = ""
    total_cases: int = 0
    direction_correct: int = 0
    direction_accuracy: float = 0.0
    avg_kendall_tau: float = 0.0
    avg_column_precision: float = 0.0
    avg_column_f1: float = 0.0
    rotation_accuracy: float = 0.0
    avg_inference_time_ms: float = 0.0
    errors: list = field(default_factory=list)


@dataclass
class VerticalTextBenchmarkResult:
    """Complete benchmark result for CJK vertical text validation."""

    total_cases: int = 0
    direction_accuracy: float = 0.0
    avg_kendall_tau: float = 0.0
    avg_column_precision: float = 0.0
    avg_column_f1: float = 0.0
    rotation_accuracy: float = 0.0
    avg_inference_time_ms: float = 0.0
    min_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    p95_inference_time_ms: float = 0.0
    per_category: dict = field(default_factory=dict)
    dataset: str = ""
    target_direction_accuracy: float = 90.0
    target_ordering_tau: float = 0.80
    passed: bool = False
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_kendall_tau(predicted_order: list, expected_order: list) -> float:
    """Compute Kendall's tau rank correlation between two orderings.

    Returns a value between -1.0 and 1.0 where 1.0 indicates perfect
    agreement. Returns 0.0 if the inputs are empty or have length < 2.

    Parameters
    ----------
    predicted_order : list[str]
        Text items in predicted reading order.
    expected_order : list[str]
        Text items in expected reading order.

    Returns
    -------
    float
        Kendall's tau correlation coefficient.
    """
    if len(predicted_order) < 2 or len(expected_order) < 2:
        return 0.0

    # Build rank mapping from expected order
    expected_ranks = {text: i for i, text in enumerate(expected_order)}

    # Filter predicted to only items present in expected
    common_predicted = [t for t in predicted_order if t in expected_ranks]
    if len(common_predicted) < 2:
        return 0.0

    # Build rank pairs
    n = len(common_predicted)
    predicted_ranks = list(range(n))
    expected_rank_values = [expected_ranks[t] for t in common_predicted]

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = predicted_ranks[i] - predicted_ranks[j]
            exp_diff = expected_rank_values[i] - expected_rank_values[j]
            product = pred_diff * exp_diff
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            # Ties are ignored

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.0

    return (concordant - discordant) / total_pairs


def compute_column_precision(predicted_columns: int, expected_columns: int) -> float:
    """Compute column count precision.

    Returns 1.0 if column counts match exactly, otherwise a ratio showing
    how close the prediction is to the expected count.

    Parameters
    ----------
    predicted_columns : int
        Number of columns detected.
    expected_columns : int
        Number of expected columns.

    Returns
    -------
    float
        Precision value between 0.0 and 1.0.
    """
    if expected_columns <= 0:
        return 1.0 if predicted_columns == 0 else 0.0
    if predicted_columns <= 0:
        return 0.0

    max_val = max(predicted_columns, expected_columns)
    return 1.0 - abs(predicted_columns - expected_columns) / max_val


def compute_column_f1(predicted_columns: int, expected_columns: int) -> float:
    """Compute column detection F1 score.

    Treats column detection as a counting problem where precision measures
    how many predicted columns correspond to real columns, and recall
    measures how many real columns were detected.

    Parameters
    ----------
    predicted_columns : int
        Number of columns detected.
    expected_columns : int
        Number of expected columns.

    Returns
    -------
    float
        F1 score between 0.0 and 1.0.
    """
    if expected_columns <= 0 and predicted_columns <= 0:
        return 1.0
    if expected_columns <= 0 or predicted_columns <= 0:
        return 0.0

    tp = min(predicted_columns, expected_columns)
    fp = max(0, predicted_columns - expected_columns)
    fn = max(0, expected_columns - predicted_columns)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) == 0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def compute_rotation_accuracy(
    predicted_rotations: list,
    expected_rotations: list,
) -> float:
    """Compute rotation direction accuracy.

    Compares predicted rotation directions against ground truth.

    Parameters
    ----------
    predicted_rotations : list[str]
        Predicted rotation directions (e.g., "ccw", "cw", "none").
    expected_rotations : list[str]
        Expected rotation directions.

    Returns
    -------
    float
        Accuracy as a value between 0.0 and 1.0.
    """
    if not predicted_rotations or not expected_rotations:
        return 0.0
    if len(predicted_rotations) != len(expected_rotations):
        return 0.0

    correct = sum(
        1 for p, e in zip(predicted_rotations, expected_rotations) if p == e
    )
    return correct / len(predicted_rotations)


# ---------------------------------------------------------------------------
# VerticalTextValidator class
# ---------------------------------------------------------------------------


class VerticalTextValidator:
    """High-level validator for CJK vertical text detection and analysis.

    Loads documents from a corpus directory or synthetic data, runs the
    vertical_text module's detection pipeline, and validates reading order,
    column detection, and rotation accuracy against ground truth.

    Parameters
    ----------
    corpus_dir : str or None
        Path to labeled corpus directory (optional).
    verbose : bool
        If True, log detailed per-case results.
    """

    def __init__(self, corpus_dir: str = None, verbose: bool = False):
        self.corpus_dir = corpus_dir
        self.verbose = verbose
        self._cases = []

    def load_corpus(self, corpus_dir: str = None) -> int:
        """Load test cases from a corpus directory.

        Parameters
        ----------
        corpus_dir : str or None
            Path to corpus root. Falls back to self.corpus_dir.

        Returns
        -------
        int
            Number of valid test cases loaded.
        """
        target = corpus_dir or self.corpus_dir
        if not target:
            logger.error("No corpus directory specified.")
            return 0

        corpus_path = Path(target)
        if not corpus_path.is_dir():
            logger.error("Corpus directory does not exist: %s", target)
            return 0

        self._cases = []
        for subdir_name in CORPUS_SUBDIRS:
            subdir = corpus_path / subdir_name
            if not subdir.is_dir():
                continue

            sidecars = {
                f.name.replace(SIDECAR_SUFFIX, ""): f
                for f in subdir.iterdir()
                if f.is_file() and f.name.endswith(SIDECAR_SUFFIX)
            }

            for stem, sidecar_path in sorted(sidecars.items()):
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    self._cases.append({
                        "stem": stem,
                        "category": subdir_name,
                        "sidecar": data,
                    })
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to load %s: %s", sidecar_path, exc)

        return len(self._cases)

    def load_ground_truth(self, ground_truth: list) -> int:
        """Load test cases from a list of ground truth dicts.

        Each dict should follow the sidecar JSON schema with optional
        ``vertical_regions`` field:

            {
              "direction": "vertical",
              "expected_order": ["text_1", "text_2"],
              "vertical_regions": [
                {"text": "...", "reading_order": 0, "bbox": [x1, y1, x2, y2]}
              ],
              "ocr_lines": [...],
              "page_width": 1000
            }

        Parameters
        ----------
        ground_truth : list[dict]
            List of ground truth test case dicts.

        Returns
        -------
        int
            Number of test cases loaded.
        """
        self._cases = []
        for i, gt in enumerate(ground_truth):
            category = "vertical_only"
            direction = gt.get("direction", "horizontal")
            if direction == "horizontal":
                category = "horizontal_only"
            elif direction == "mixed":
                category = "mixed_layout"

            self._cases.append({
                "stem": gt.get("id", f"case_{i}"),
                "category": category,
                "sidecar": gt,
            })
        return len(self._cases)

    def validate(
        self,
        target_direction_accuracy: float = 90.0,
        target_ordering_tau: float = 0.80,
    ) -> VerticalTextBenchmarkResult:
        """Run validation against loaded test cases.

        Parameters
        ----------
        target_direction_accuracy : float
            Target direction detection accuracy percentage.
        target_ordering_tau : float
            Target Kendall's tau for reading order.

        Returns
        -------
        VerticalTextBenchmarkResult
            Validation results.
        """
        if not _VERTICAL_TEXT_AVAILABLE:
            logger.error("vertical_text module not available.")
            return VerticalTextBenchmarkResult(
                dataset=self.corpus_dir or "in-memory",
                target_direction_accuracy=target_direction_accuracy,
                target_ordering_tau=target_ordering_tau,
                timestamp=datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="milliseconds"),
            )

        all_dir_correct = 0
        all_taus = []
        all_col_f1s = []
        all_col_precs = []
        all_rot_pred = []
        all_rot_exp = []
        all_timings = []
        per_category = {n: CategoryMetrics(category=n) for n in CORPUS_SUBDIRS}

        for case in self._cases:
            data = case["sidecar"]
            cat_name = case["category"]
            cat = per_category.get(cat_name)
            if cat is None:
                continue

            ocr_lines = _parse_ocr_lines(data)
            expected_direction = data.get("direction", "horizontal")
            expected_order = data.get("expected_order", [])
            expected_columns = data.get("expected_columns", 0)
            page_width = data.get("page_width", 1000)
            expected_rotation = data.get("expected_rotation")

            # Also extract expected_order from vertical_regions if present
            vertical_regions = data.get("vertical_regions", [])
            if not expected_order and vertical_regions:
                sorted_regions = sorted(
                    vertical_regions, key=lambda r: r.get("reading_order", 0)
                )
                expected_order = [r["text"] for r in sorted_regions]
            if not expected_columns and vertical_regions:
                expected_columns = len(
                    set(r.get("bbox", [0, 0, 0, 0])[0] for r in vertical_regions)
                )

            cat.total_cases += 1

            # Direction detection
            start = time.perf_counter()
            direction_result = classify_page_text_direction(ocr_lines)
            elapsed_ms = (time.perf_counter() - start) * 1000
            all_timings.append(elapsed_ms)

            predicted_direction = direction_result.get("direction", "horizontal")
            if predicted_direction == expected_direction:
                cat.direction_correct += 1
                all_dir_correct += 1

            # Reading order
            if expected_order and ocr_lines:
                sorted_result = sort_mixed_reading_order(ocr_lines, page_width)
                predicted_order = [item[0] for item in sorted_result]
                tau = compute_kendall_tau(predicted_order, expected_order)
                all_taus.append(tau)

            # Column grouping
            if expected_columns > 0 and predicted_direction in ("vertical", "mixed"):
                vertical_boxes = []
                for text, box, conf in ocr_lines:
                    if is_vertical_text_box(box, text=str(text)):
                        vertical_boxes.append((box, text, conf))

                if vertical_boxes:
                    columns = group_vertical_columns(vertical_boxes, page_width)
                    pred_cols = len(columns)
                else:
                    pred_cols = 0

                all_col_precs.append(compute_column_precision(pred_cols, expected_columns))
                all_col_f1s.append(compute_column_f1(pred_cols, expected_columns))

            # Rotation accuracy
            if expected_rotation is not None:
                # Predict: vertical => ccw, horizontal => none
                if predicted_direction in ("vertical", "mixed"):
                    pred_rot = "ccw"
                else:
                    pred_rot = "none"
                all_rot_pred.append(pred_rot)
                all_rot_exp.append(expected_rotation)

            if self.verbose:
                logger.info(
                    "Case %s: direction=%s (expected=%s)",
                    case["stem"],
                    predicted_direction,
                    expected_direction,
                )

        # Aggregate
        total_cases = sum(c.total_cases for c in per_category.values())
        for cat in per_category.values():
            if cat.total_cases > 0:
                cat.direction_accuracy = round(
                    cat.direction_correct / cat.total_cases * 100, 2
                )

        direction_accuracy = (
            round(all_dir_correct / total_cases * 100, 2)
            if total_cases > 0
            else 0.0
        )
        avg_tau = round(statistics.mean(all_taus), 4) if all_taus else 0.0
        avg_col_prec = (
            round(statistics.mean(all_col_precs), 4) if all_col_precs else 0.0
        )
        avg_col_f1 = (
            round(statistics.mean(all_col_f1s), 4) if all_col_f1s else 0.0
        )
        rot_acc = (
            round(compute_rotation_accuracy(all_rot_pred, all_rot_exp), 4)
            if all_rot_pred
            else 0.0
        )

        passed = (
            direction_accuracy >= target_direction_accuracy
            and (avg_tau >= target_ordering_tau if all_taus else True)
        )

        return VerticalTextBenchmarkResult(
            total_cases=total_cases,
            direction_accuracy=direction_accuracy,
            avg_kendall_tau=avg_tau,
            avg_column_precision=avg_col_prec,
            avg_column_f1=avg_col_f1,
            rotation_accuracy=rot_acc,
            avg_inference_time_ms=(
                round(statistics.mean(all_timings), 2) if all_timings else 0.0
            ),
            min_inference_time_ms=(
                round(min(all_timings), 2) if all_timings else 0.0
            ),
            max_inference_time_ms=(
                round(max(all_timings), 2) if all_timings else 0.0
            ),
            p95_inference_time_ms=(
                round(
                    sorted(all_timings)[
                        min(int(len(all_timings) * 0.95), len(all_timings) - 1)
                    ],
                    2,
                )
                if len(all_timings) >= 5
                else (round(max(all_timings), 2) if all_timings else 0.0)
            ),
            per_category={k: asdict(v) for k, v in per_category.items()},
            dataset=self.corpus_dir or "in-memory",
            target_direction_accuracy=target_direction_accuracy,
            target_ordering_tau=target_ordering_tau,
            passed=passed,
            timestamp=datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="milliseconds"),
        )


# ---------------------------------------------------------------------------
# Corpus validation (dry-run)
# ---------------------------------------------------------------------------


def validate_corpus_structure(corpus_dir: str) -> dict:
    """Validate corpus directory structure and sidecar files.

    Parameters
    ----------
    corpus_dir : str
        Path to the corpus root directory.

    Returns
    -------
    dict
        Validation result with keys: valid (bool), categories (dict),
        errors (list), total_cases (int).
    """
    corpus_path = Path(corpus_dir)
    result = {
        "valid": True,
        "categories": {},
        "errors": [],
        "total_cases": 0,
    }

    if not corpus_path.is_dir():
        result["valid"] = False
        result["errors"].append(f"Corpus directory does not exist: {corpus_dir}")
        return result

    found_any_subdir = False

    for subdir_name in CORPUS_SUBDIRS:
        subdir = corpus_path / subdir_name
        cat_info = {
            "exists": subdir.is_dir(),
            "image_count": 0,
            "sidecar_count": 0,
            "valid_pairs": 0,
            "missing_sidecars": [],
            "missing_images": [],
            "invalid_sidecars": [],
        }

        if not subdir.is_dir():
            result["categories"][subdir_name] = cat_info
            continue

        found_any_subdir = True

        # Find images and sidecars
        images = {
            f.stem: f
            for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
        }
        sidecars = {
            f.name.replace(SIDECAR_SUFFIX, ""): f
            for f in subdir.iterdir()
            if f.is_file() and f.name.endswith(SIDECAR_SUFFIX)
        }

        cat_info["image_count"] = len(images)
        cat_info["sidecar_count"] = len(sidecars)

        # Check for matched pairs
        for stem, img_path in images.items():
            if stem in sidecars:
                # Validate sidecar JSON
                sidecar_path = sidecars[stem]
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Validate required fields
                    missing_fields = []
                    for req_field in ("direction", "expected_order", "ocr_lines", "page_width"):
                        if req_field not in data:
                            missing_fields.append(req_field)

                    if missing_fields:
                        cat_info["invalid_sidecars"].append(
                            f"{sidecar_path.name}: missing fields {missing_fields}"
                        )
                    else:
                        cat_info["valid_pairs"] += 1
                        result["total_cases"] += 1

                except json.JSONDecodeError as e:
                    cat_info["invalid_sidecars"].append(
                        f"{sidecar_path.name}: invalid JSON ({e})"
                    )
            else:
                cat_info["missing_sidecars"].append(img_path.name)

        # Check for orphan sidecars
        for stem in sidecars:
            if stem not in images:
                cat_info["missing_images"].append(f"{stem}{SIDECAR_SUFFIX}")

        if cat_info["invalid_sidecars"]:
            result["valid"] = False
            result["errors"].extend(cat_info["invalid_sidecars"])

        result["categories"][subdir_name] = cat_info

    if not found_any_subdir:
        result["valid"] = False
        result["errors"].append(
            f"No recognized subdirectories found. "
            f"Expected at least one of: {', '.join(CORPUS_SUBDIRS)}"
        )

    return result


# ---------------------------------------------------------------------------
# Corpus benchmark runner
# ---------------------------------------------------------------------------


def _parse_ocr_lines(sidecar_data: dict) -> list:
    """Parse OCR lines from sidecar JSON into (text, box, confidence) tuples.

    Parameters
    ----------
    sidecar_data : dict
        Parsed sidecar JSON data.

    Returns
    -------
    list[tuple]
        List of (text, box_points, confidence) tuples.
    """
    lines = []
    for entry in sidecar_data.get("ocr_lines", []):
        text = entry.get("text", "")
        box = entry.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]])
        confidence = entry.get("confidence", 0.9)
        lines.append((text, box, confidence))
    return lines


def run_corpus_benchmark(
    corpus_dir: str,
    target_direction_accuracy: float = 90.0,
    target_ordering_tau: float = 0.80,
) -> VerticalTextBenchmarkResult:
    """Run benchmark against a labeled corpus directory.

    Parameters
    ----------
    corpus_dir : str
        Path to corpus root directory.
    target_direction_accuracy : float
        Target direction detection accuracy percentage.
    target_ordering_tau : float
        Target Kendall's tau for reading order accuracy.

    Returns
    -------
    VerticalTextBenchmarkResult
        Benchmark results.
    """
    if not _VERTICAL_TEXT_AVAILABLE:
        logger.error("vertical_text module not available for corpus benchmark.")
        return VerticalTextBenchmarkResult(
            dataset=str(corpus_dir),
            target_direction_accuracy=target_direction_accuracy,
            target_ordering_tau=target_ordering_tau,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="milliseconds"
            ),
        )

    corpus_path = Path(corpus_dir)
    all_direction_correct = 0
    all_direction_total = 0
    all_taus = []
    all_col_precisions = []
    all_col_f1s = []
    all_rot_pred = []
    all_rot_exp = []
    all_timings = []
    per_category = {}

    for subdir_name in CORPUS_SUBDIRS:
        subdir = corpus_path / subdir_name
        if not subdir.is_dir():
            continue

        cat = CategoryMetrics(category=subdir_name)
        cat_taus = []
        cat_col_precisions = []
        cat_col_f1s = []
        cat_timings = []

        # Find valid pairs
        sidecars = {
            f.name.replace(SIDECAR_SUFFIX, ""): f
            for f in subdir.iterdir()
            if f.is_file() and f.name.endswith(SIDECAR_SUFFIX)
        }

        for stem, sidecar_path in sorted(sidecars.items()):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                ocr_lines = _parse_ocr_lines(data)
                expected_direction = data.get("direction", "horizontal")
                expected_order = data.get("expected_order", [])
                expected_columns = data.get("expected_columns", 0)
                page_width = data.get("page_width", 1000)
                expected_rotation = data.get("expected_rotation")

                # Derive expected_order from vertical_regions if absent
                vertical_regions = data.get("vertical_regions", [])
                if not expected_order and vertical_regions:
                    sorted_regions = sorted(
                        vertical_regions, key=lambda r: r.get("reading_order", 0)
                    )
                    expected_order = [r["text"] for r in sorted_regions]

                cat.total_cases += 1

                # -- Direction detection --
                start = time.perf_counter()
                direction_result = classify_page_text_direction(ocr_lines)
                elapsed_ms = (time.perf_counter() - start) * 1000
                cat_timings.append(elapsed_ms)

                predicted_direction = direction_result.get("direction", "horizontal")
                if predicted_direction == expected_direction:
                    cat.direction_correct += 1

                # -- Reading order (Kendall's tau) --
                if expected_order and ocr_lines:
                    if predicted_direction in ("vertical", "mixed"):
                        sorted_result = sort_mixed_reading_order(
                            ocr_lines, page_width
                        )
                        predicted_order = [item[0] for item in sorted_result]
                    else:
                        predicted_order = [item[0] for item in ocr_lines]

                    tau = compute_kendall_tau(predicted_order, expected_order)
                    cat_taus.append(tau)
                    all_taus.append(tau)

                # -- Column grouping precision + F1 --
                if expected_columns > 0 and predicted_direction in ("vertical", "mixed"):
                    vertical_boxes = []
                    for text, box, conf in ocr_lines:
                        if is_vertical_text_box(box, text=str(text)):
                            vertical_boxes.append((box, text, conf))

                    if vertical_boxes:
                        columns = group_vertical_columns(vertical_boxes, page_width)
                        pred_cols = len(columns)
                    else:
                        pred_cols = 0

                    col_prec = compute_column_precision(pred_cols, expected_columns)
                    col_f1 = compute_column_f1(pred_cols, expected_columns)
                    cat_col_precisions.append(col_prec)
                    cat_col_f1s.append(col_f1)
                    all_col_precisions.append(col_prec)
                    all_col_f1s.append(col_f1)

                # -- Rotation accuracy --
                if expected_rotation is not None:
                    if predicted_direction in ("vertical", "mixed"):
                        pred_rot = "ccw"
                    else:
                        pred_rot = "none"
                    all_rot_pred.append(pred_rot)
                    all_rot_exp.append(expected_rotation)

            except Exception as exc:
                cat.errors.append(f"{stem}: {exc}")
                logger.warning("Error processing %s: %s", stem, exc)

        # Compute category-level metrics
        if cat.total_cases > 0:
            cat.direction_accuracy = round(
                cat.direction_correct / cat.total_cases * 100, 2
            )
        if cat_taus:
            cat.avg_kendall_tau = round(statistics.mean(cat_taus), 4)
        if cat_col_precisions:
            cat.avg_column_precision = round(
                statistics.mean(cat_col_precisions), 4
            )
        if cat_col_f1s:
            cat.avg_column_f1 = round(statistics.mean(cat_col_f1s), 4)
        if cat_timings:
            cat.avg_inference_time_ms = round(statistics.mean(cat_timings), 2)
            all_timings.extend(cat_timings)

        all_direction_correct += cat.direction_correct
        all_direction_total += cat.total_cases

        per_category[subdir_name] = asdict(cat)

    # Compute overall metrics
    total_cases = all_direction_total
    direction_accuracy = (
        round(all_direction_correct / total_cases * 100, 2)
        if total_cases > 0
        else 0.0
    )
    avg_tau = round(statistics.mean(all_taus), 4) if all_taus else 0.0
    avg_col = (
        round(statistics.mean(all_col_precisions), 4)
        if all_col_precisions
        else 0.0
    )
    avg_col_f1 = (
        round(statistics.mean(all_col_f1s), 4) if all_col_f1s else 0.0
    )
    rot_acc = (
        round(compute_rotation_accuracy(all_rot_pred, all_rot_exp), 4)
        if all_rot_pred
        else 0.0
    )

    passed = (
        direction_accuracy >= target_direction_accuracy
        and (avg_tau >= target_ordering_tau if all_taus else True)
    )

    return VerticalTextBenchmarkResult(
        total_cases=total_cases,
        direction_accuracy=direction_accuracy,
        avg_kendall_tau=avg_tau,
        avg_column_precision=avg_col,
        avg_column_f1=avg_col_f1,
        rotation_accuracy=rot_acc,
        avg_inference_time_ms=(
            round(statistics.mean(all_timings), 2) if all_timings else 0.0
        ),
        min_inference_time_ms=(
            round(min(all_timings), 2) if all_timings else 0.0
        ),
        max_inference_time_ms=(
            round(max(all_timings), 2) if all_timings else 0.0
        ),
        p95_inference_time_ms=(
            round(
                sorted(all_timings)[
                    min(int(len(all_timings) * 0.95), len(all_timings) - 1)
                ],
                2,
            )
            if len(all_timings) >= 5
            else (round(max(all_timings), 2) if all_timings else 0.0)
        ),
        per_category=per_category,
        dataset=str(corpus_dir),
        target_direction_accuracy=target_direction_accuracy,
        target_ordering_tau=target_ordering_tau,
        passed=passed,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------


def _make_box(x1, y1, x2, y2):
    """Create a 4-point polygon from corner coordinates."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _make_vertical_line(text, x_center, y_top, width=30, height=200, conf=0.90):
    """Create a vertical OCR line tuple."""
    x1 = x_center - width // 2
    box = _make_box(x1, y_top, x1 + width, y_top + height)
    return (text, box, conf)


def _make_horizontal_line(text, x1, y1, width=400, height=30, conf=0.90):
    """Create a horizontal OCR line tuple."""
    box = _make_box(x1, y1, x1 + width, y1 + height)
    return (text, box, conf)


def _generate_vertical_page(rng, num_columns=3, lines_per_col=4, page_width=1000):
    """Generate a synthetic vertical-only page with known ordering.

    Returns (ocr_lines, expected_order, expected_columns, page_width).
    """
    ocr_lines = []
    expected_order = []
    col_spacing = page_width // (num_columns + 1)

    # Columns are right-to-left
    for col_idx in range(num_columns):
        x_center = page_width - (col_idx + 1) * col_spacing
        for line_idx in range(lines_per_col):
            text = f"\u4e2d\u6587{col_idx}_{line_idx}"
            y_top = 50 + line_idx * 220
            line = _make_vertical_line(text, x_center, y_top)
            ocr_lines.append(line)
            expected_order.append(text)

    # Shuffle OCR lines to simulate unordered OCR output
    rng.shuffle(ocr_lines)

    return ocr_lines, expected_order, num_columns, page_width


def _generate_horizontal_page(rng, num_lines=5, page_width=1000):
    """Generate a synthetic horizontal-only page (negative case).

    Returns (ocr_lines, expected_order, expected_columns, page_width).
    """
    ocr_lines = []
    expected_order = []

    for i in range(num_lines):
        text = f"Line {i + 1}: English text content"
        y_top = 50 + i * 60
        line = _make_horizontal_line(text, 50, y_top)
        ocr_lines.append(line)
        expected_order.append(text)

    rng.shuffle(ocr_lines)
    return ocr_lines, expected_order, 0, page_width


def _generate_mixed_page(rng, num_v_columns=2, v_lines_per_col=3,
                         num_h_lines=3, page_width=1000):
    """Generate a synthetic mixed-layout page.

    Returns (ocr_lines, expected_order, expected_columns, page_width).
    """
    ocr_lines = []
    expected_order = []

    # Horizontal lines at the top
    for i in range(num_h_lines):
        text = f"Header line {i + 1}"
        y_top = 10 + i * 40
        line = _make_horizontal_line(text, 50, y_top)
        ocr_lines.append(line)
        expected_order.append(text)

    # Vertical columns below
    col_spacing = page_width // (num_v_columns + 1)
    for col_idx in range(num_v_columns):
        x_center = page_width - (col_idx + 1) * col_spacing
        for line_idx in range(v_lines_per_col):
            text = f"\u65e5\u672c\u8a9e{col_idx}_{line_idx}"
            y_top = 200 + line_idx * 220
            line = _make_vertical_line(text, x_center, y_top)
            ocr_lines.append(line)
            expected_order.append(text)

    rng.shuffle(ocr_lines)
    return ocr_lines, expected_order, num_v_columns, page_width


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------


# CJK characters for synthetic images
_CJK_CHARS = list(
    "\u4e2d\u6587\u6d4b\u8bd5\u6570\u636e"  # Chinese
    "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8"  # Japanese
    "\ud55c\uad6d\uc5b4\ud14c\uc2a4\ud2b8"  # Korean
    "\u5c71\u6c34\u98a8\u6708\u82b1\u9ce5"  # Nature kanji
    "\u5929\u5730\u4eba\u751f\u6642\u9593"  # Common kanji
)


def generate_synthetic_vertical_image(
    rng,
    page_width=800,
    page_height=1200,
    num_columns=3,
    chars_per_column=6,
):
    """Generate a synthetic vertical text image with ground truth.

    Creates a PNG image with vertical CJK text columns rendered using PIL,
    along with a matching sidecar ground truth dict.

    Parameters
    ----------
    rng : random.Random
        Random number generator for reproducibility.
    page_width : int
        Image width in pixels.
    page_height : int
        Image height in pixels.
    num_columns : int
        Number of vertical text columns.
    chars_per_column : int
        Number of characters per column.

    Returns
    -------
    tuple[Image, dict]
        (PIL Image, sidecar ground truth dict). If PIL is not available,
        returns (None, sidecar dict with ocr_lines only).
    """
    ocr_lines = []
    expected_order = []
    vertical_regions = []
    col_spacing = page_width // (num_columns + 1)
    char_height = 40
    char_width = 30

    for col_idx in range(num_columns):
        x_center = page_width - (col_idx + 1) * col_spacing
        for char_idx in range(chars_per_column):
            char = rng.choice(_CJK_CHARS)
            y_top = 50 + char_idx * (char_height + 10)
            x1 = x_center - char_width // 2
            box = _make_box(x1, y_top, x1 + char_width, y_top + char_height)
            text = char
            ocr_lines.append((text, box, 0.92))
            expected_order.append(text)
            vertical_regions.append({
                "text": text,
                "reading_order": len(expected_order) - 1,
                "bbox": [x1, y_top, x1 + char_width, y_top + char_height],
            })

    sidecar = {
        "direction": "vertical",
        "expected_order": expected_order,
        "expected_columns": num_columns,
        "vertical_regions": vertical_regions,
        "ocr_lines": [
            {"text": t, "box": b, "confidence": c} for t, b, c in ocr_lines
        ],
        "page_width": page_width,
        "expected_rotation": "ccw",
    }

    img = None
    if _PIL_AVAILABLE:
        img = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(img)

        # Try system font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for region in vertical_regions:
            x1, y1, x2, y2 = region["bbox"]
            draw.text((x1 + 2, y1 + 2), region["text"], fill="black", font=font)
            draw.rectangle([x1, y1, x2, y2], outline="gray")

    return img, sidecar


def generate_synthetic_corpus(
    output_dir: str,
    num_samples: int = 20,
    seed: int = 42,
) -> dict:
    """Generate a complete synthetic corpus with images and sidecars.

    Creates the corpus directory structure with vertical_only, mixed_layout,
    and horizontal_only subdirectories, each containing synthetic test images
    and their ground truth sidecar JSON files.

    Parameters
    ----------
    output_dir : str
        Root directory for the generated corpus.
    num_samples : int
        Total number of test cases to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Summary with keys: total_generated, per_category counts, output_dir.
    """
    rng = random.Random(seed)
    output_path = Path(output_dir)

    n_vertical = max(1, int(num_samples * 0.4))
    n_mixed = max(1, int(num_samples * 0.3))
    n_horizontal = max(1, num_samples - n_vertical - n_mixed)

    summary = {
        "total_generated": 0,
        "per_category": {},
        "output_dir": str(output_dir),
        "pil_available": _PIL_AVAILABLE,
    }

    # Generate vertical_only
    vert_dir = output_path / "vertical_only"
    vert_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_vertical):
        num_cols = rng.randint(2, 5)
        chars_per = rng.randint(3, 8)
        img, sidecar = generate_synthetic_vertical_image(
            rng, num_columns=num_cols, chars_per_column=chars_per
        )
        stem = f"vertical_{i:04d}"
        if img is not None:
            img.save(str(vert_dir / f"{stem}.png"))
        with open(vert_dir / f"{stem}{SIDECAR_SUFFIX}", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
    summary["per_category"]["vertical_only"] = n_vertical

    # Generate horizontal_only
    horiz_dir = output_path / "horizontal_only"
    horiz_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_horizontal):
        num_lines = rng.randint(3, 8)
        ocr_lines, expected_order, _, page_width = _generate_horizontal_page(
            rng, num_lines=num_lines
        )
        sidecar = {
            "direction": "horizontal",
            "expected_order": expected_order,
            "expected_columns": 0,
            "vertical_regions": [],
            "ocr_lines": [
                {"text": t, "box": b, "confidence": c} for t, b, c in ocr_lines
            ],
            "page_width": page_width,
            "expected_rotation": "none",
        }
        stem = f"horizontal_{i:04d}"
        if _PIL_AVAILABLE:
            img = Image.new("RGB", (1000, 600), "white")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except (OSError, IOError):
                font = ImageFont.load_default()
            for entry in sidecar["ocr_lines"]:
                box = entry["box"]
                draw.text(
                    (box[0][0] + 2, box[0][1] + 2),
                    entry["text"],
                    fill="black",
                    font=font,
                )
            img.save(str(horiz_dir / f"{stem}.png"))
        with open(horiz_dir / f"{stem}{SIDECAR_SUFFIX}", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
    summary["per_category"]["horizontal_only"] = n_horizontal

    # Generate mixed_layout
    mixed_dir = output_path / "mixed_layout"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_mixed):
        num_v = rng.randint(1, 3)
        vl_per = rng.randint(2, 4)
        num_h = rng.randint(2, 4)
        ocr_lines, expected_order, expected_columns, page_width = _generate_mixed_page(
            rng, num_v_columns=num_v, v_lines_per_col=vl_per, num_h_lines=num_h
        )
        vertical_regions = []
        order_idx = 0
        for text, box, conf in ocr_lines:
            if is_vertical_text_box(box, text=str(text)) if _VERTICAL_TEXT_AVAILABLE else False:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                vertical_regions.append({
                    "text": text,
                    "reading_order": order_idx,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                })
                order_idx += 1
        sidecar = {
            "direction": "mixed",
            "expected_order": expected_order,
            "expected_columns": expected_columns,
            "vertical_regions": vertical_regions,
            "ocr_lines": [
                {"text": t, "box": b, "confidence": c} for t, b, c in ocr_lines
            ],
            "page_width": page_width,
            "expected_rotation": "ccw",
        }
        stem = f"mixed_{i:04d}"
        if _PIL_AVAILABLE:
            img = Image.new("RGB", (1000, 1000), "white")
            img.save(str(mixed_dir / f"{stem}.png"))
        with open(mixed_dir / f"{stem}{SIDECAR_SUFFIX}", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
    summary["per_category"]["mixed_layout"] = n_mixed

    summary["total_generated"] = n_vertical + n_horizontal + n_mixed
    return summary


def run_synthetic_benchmark(
    sample_size: int = 30,
    target_direction_accuracy: float = 90.0,
    target_ordering_tau: float = 0.80,
) -> VerticalTextBenchmarkResult:
    """Run benchmark with synthetic data (no corpus required).

    Generates pages with known layouts and expected orderings, then
    measures how well the vertical text module detects and reorders them.

    Parameters
    ----------
    sample_size : int
        Total number of synthetic test cases to generate.
    target_direction_accuracy : float
        Target direction detection accuracy percentage.
    target_ordering_tau : float
        Target Kendall's tau for reading order accuracy.

    Returns
    -------
    VerticalTextBenchmarkResult
        Benchmark results.
    """
    if not _VERTICAL_TEXT_AVAILABLE:
        logger.error(
            "vertical_text module not available. "
            "Cannot run synthetic benchmark."
        )
        return VerticalTextBenchmarkResult(
            dataset="synthetic",
            target_direction_accuracy=target_direction_accuracy,
            target_ordering_tau=target_ordering_tau,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="milliseconds"
            ),
        )

    rng = random.Random(42)

    # Split samples across categories: 40% vertical, 30% mixed, 30% horizontal
    n_vertical = max(1, int(sample_size * 0.4))
    n_mixed = max(1, int(sample_size * 0.3))
    n_horizontal = max(1, sample_size - n_vertical - n_mixed)

    generators = (
        [("vertical_only", "vertical", _generate_vertical_page)] * n_vertical
        + [("mixed_layout", "mixed", _generate_mixed_page)] * n_mixed
        + [("horizontal_only", "horizontal", _generate_horizontal_page)] * n_horizontal
    )

    all_direction_correct = 0
    all_taus = []
    all_col_precisions = []
    all_col_f1s = []
    all_rot_pred = []
    all_rot_exp = []
    all_timings = []
    per_category = {}

    for cat_name in CORPUS_SUBDIRS:
        per_category[cat_name] = CategoryMetrics(category=cat_name)

    for cat_name, expected_direction, generator in generators:
        cat = per_category[cat_name]

        if generator == _generate_vertical_page:
            num_cols = rng.randint(2, 5)
            lines_per = rng.randint(2, 5)
            ocr_lines, expected_order, expected_columns, page_width = generator(
                rng, num_columns=num_cols, lines_per_col=lines_per
            )
        elif generator == _generate_mixed_page:
            num_v = rng.randint(1, 3)
            vl_per = rng.randint(2, 4)
            num_h = rng.randint(2, 4)
            ocr_lines, expected_order, expected_columns, page_width = generator(
                rng, num_v_columns=num_v, v_lines_per_col=vl_per,
                num_h_lines=num_h
            )
        else:
            num_lines = rng.randint(3, 8)
            ocr_lines, expected_order, expected_columns, page_width = generator(
                rng, num_lines=num_lines
            )

        cat.total_cases += 1

        # -- Direction detection --
        start = time.perf_counter()
        direction_result = classify_page_text_direction(ocr_lines)
        elapsed_ms = (time.perf_counter() - start) * 1000
        all_timings.append(elapsed_ms)

        predicted_direction = direction_result.get("direction", "horizontal")
        if predicted_direction == expected_direction:
            cat.direction_correct += 1
            all_direction_correct += 1

        # -- Reading order --
        if expected_order and ocr_lines:
            sorted_result = sort_mixed_reading_order(
                ocr_lines, page_width
            )
            predicted_order = [item[0] for item in sorted_result]
            tau = compute_kendall_tau(predicted_order, expected_order)
            all_taus.append(tau)

        # -- Column grouping --
        if expected_columns > 0 and predicted_direction in ("vertical", "mixed"):
            vertical_boxes = []
            for text, box, conf in ocr_lines:
                if is_vertical_text_box(box, text=str(text)):
                    vertical_boxes.append((box, text, conf))

            if vertical_boxes:
                columns = group_vertical_columns(vertical_boxes, page_width)
                predicted_columns = len(columns)
            else:
                predicted_columns = 0

            all_col_precisions.append(
                compute_column_precision(predicted_columns, expected_columns)
            )
            all_col_f1s.append(
                compute_column_f1(predicted_columns, expected_columns)
            )

        # -- Rotation accuracy --
        if expected_direction in ("vertical", "mixed"):
            expected_rot = "ccw"
        else:
            expected_rot = "none"
        if predicted_direction in ("vertical", "mixed"):
            pred_rot = "ccw"
        else:
            pred_rot = "none"
        all_rot_pred.append(pred_rot)
        all_rot_exp.append(expected_rot)

    # Finalize per-category metrics
    total_cases = sum(c.total_cases for c in per_category.values())
    for cat in per_category.values():
        if cat.total_cases > 0:
            cat.direction_accuracy = round(
                cat.direction_correct / cat.total_cases * 100, 2
            )

    per_category_dict = {k: asdict(v) for k, v in per_category.items()}

    direction_accuracy = (
        round(all_direction_correct / total_cases * 100, 2)
        if total_cases > 0
        else 0.0
    )
    avg_tau = round(statistics.mean(all_taus), 4) if all_taus else 0.0
    avg_col = (
        round(statistics.mean(all_col_precisions), 4)
        if all_col_precisions
        else 0.0
    )
    avg_col_f1 = (
        round(statistics.mean(all_col_f1s), 4) if all_col_f1s else 0.0
    )
    rot_acc = (
        round(compute_rotation_accuracy(all_rot_pred, all_rot_exp), 4)
        if all_rot_pred
        else 0.0
    )

    passed = (
        direction_accuracy >= target_direction_accuracy
        and (avg_tau >= target_ordering_tau if all_taus else True)
    )

    return VerticalTextBenchmarkResult(
        total_cases=total_cases,
        direction_accuracy=direction_accuracy,
        avg_kendall_tau=avg_tau,
        avg_column_precision=avg_col,
        avg_column_f1=avg_col_f1,
        rotation_accuracy=rot_acc,
        avg_inference_time_ms=(
            round(statistics.mean(all_timings), 2) if all_timings else 0.0
        ),
        min_inference_time_ms=(
            round(min(all_timings), 2) if all_timings else 0.0
        ),
        max_inference_time_ms=(
            round(max(all_timings), 2) if all_timings else 0.0
        ),
        p95_inference_time_ms=(
            round(
                sorted(all_timings)[
                    min(int(len(all_timings) * 0.95), len(all_timings) - 1)
                ],
                2,
            )
            if len(all_timings) >= 5
            else (round(max(all_timings), 2) if all_timings else 0.0)
        ),
        per_category=per_category_dict,
        dataset="synthetic",
        target_direction_accuracy=target_direction_accuracy,
        target_ordering_tau=target_ordering_tau,
        passed=passed,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_benchmark_report(result: VerticalTextBenchmarkResult) -> str:
    """Format benchmark result as a human-readable report.

    Parameters
    ----------
    result : VerticalTextBenchmarkResult
        Benchmark result to format.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("CJK VERTICAL TEXT BENCHMARK")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Dataset:              {result.dataset}")
    lines.append(f"  Total Cases:          {result.total_cases}")
    lines.append(f"  Direction Accuracy:   {result.direction_accuracy:.2f}%")
    lines.append(f"  Avg Kendall's Tau:    {result.avg_kendall_tau:.4f}")
    lines.append(f"  Avg Column Precision: {result.avg_column_precision:.4f}")
    lines.append(f"  Avg Column F1:        {result.avg_column_f1:.4f}")
    lines.append(f"  Rotation Accuracy:    {result.rotation_accuracy:.4f}")
    lines.append(f"  Target Dir. Acc:      {result.target_direction_accuracy:.1f}%")
    lines.append(f"  Target Ordering Tau:  {result.target_ordering_tau:.2f}")
    lines.append(f"  Result:               {'PASS' if result.passed else 'FAIL'}")
    lines.append(f"  Avg Inference:        {result.avg_inference_time_ms:.2f} ms")
    lines.append(f"  P95 Inference:        {result.p95_inference_time_ms:.2f} ms")
    lines.append(f"  Timestamp:            {result.timestamp}")
    lines.append("")

    if result.per_category:
        lines.append("PER-CATEGORY METRICS")
        lines.append("-" * 80)
        lines.append(
            f"{'Category':<20} {'Cases':>8} {'Dir.Acc':>10} "
            f"{'Tau':>10} {'ColPrec':>10} {'Errors':>8}"
        )
        lines.append("-" * 80)

        for cat_name in CORPUS_SUBDIRS:
            if cat_name in result.per_category:
                m = result.per_category[cat_name]
                lines.append(
                    f"{cat_name:<20} {m.get('total_cases', 0):>8d} "
                    f"{m.get('direction_accuracy', 0.0):>10.2f} "
                    f"{m.get('avg_kendall_tau', 0.0):>10.4f} "
                    f"{m.get('avg_column_precision', 0.0):>10.4f} "
                    f"{len(m.get('errors', [])):>8d}"
                )
        lines.append("-" * 80)

    lines.append("")
    return "\n".join(lines)


def format_markdown_report(result: VerticalTextBenchmarkResult) -> str:
    """Format benchmark result as a markdown report.

    Parameters
    ----------
    result : VerticalTextBenchmarkResult
        Benchmark result to format.

    Returns
    -------
    str
        Markdown-formatted report string.
    """
    lines = []
    lines.append("# CJK Vertical Text Benchmark Report")
    lines.append("")
    lines.append(f"**Date**: {result.timestamp}")
    lines.append(f"**Dataset**: {result.dataset}")
    lines.append(f"**Result**: {'PASS' if result.passed else 'FAIL'}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value | Target |")
    lines.append("|--------|-------|--------|")
    lines.append(
        f"| Direction Accuracy | {result.direction_accuracy:.2f}% "
        f"| {result.target_direction_accuracy:.1f}% |"
    )
    lines.append(
        f"| Avg Kendall's Tau | {result.avg_kendall_tau:.4f} "
        f"| {result.target_ordering_tau:.2f} |"
    )
    lines.append(
        f"| Avg Column Precision | {result.avg_column_precision:.4f} | -- |"
    )
    lines.append(
        f"| Avg Column F1 | {result.avg_column_f1:.4f} | -- |"
    )
    lines.append(
        f"| Rotation Accuracy | {result.rotation_accuracy:.4f} | -- |"
    )
    lines.append(f"| Total Cases | {result.total_cases} | -- |")
    lines.append(
        f"| Avg Inference | {result.avg_inference_time_ms:.2f} ms | -- |"
    )
    lines.append(
        f"| P95 Inference | {result.p95_inference_time_ms:.2f} ms | -- |"
    )
    lines.append("")

    if result.per_category:
        lines.append("## Per-Category Breakdown")
        lines.append("")
        lines.append(
            "| Category | Cases | Dir. Accuracy | Kendall Tau "
            "| Col. Precision | Errors |"
        )
        lines.append(
            "|----------|-------|---------------|-------------|"
            "----------------|--------|"
        )
        for cat_name in CORPUS_SUBDIRS:
            if cat_name in result.per_category:
                m = result.per_category[cat_name]
                lines.append(
                    f"| {cat_name} | {m.get('total_cases', 0)} "
                    f"| {m.get('direction_accuracy', 0.0):.2f}% "
                    f"| {m.get('avg_kendall_tau', 0.0):.4f} "
                    f"| {m.get('avg_column_precision', 0.0):.4f} "
                    f"| {len(m.get('errors', []))} |"
                )
        lines.append("")

    lines.append("## Known Limitations")
    lines.append("")
    lines.append(
        "- Mixed-layout pages with overlapping vertical and horizontal regions "
        "may have lower ordering accuracy due to Y-position interleaving."
    )
    lines.append(
        "- Column grouping tolerance is a fixed fraction of page width "
        "(`COLUMN_GROUPING_TOLERANCE=0.05`), which may under- or over-segment "
        "pages with unusual column spacing."
    )
    lines.append(
        "- Single-character vertical boxes are excluded from detection "
        "by default (`min_chars=2`)."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CJK vertical text detection and reading order "
            "accuracy (Phase 6D validation)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_vertical_text.py --corpus-dir /data/cjk-corpus
  python scripts/benchmark_vertical_text.py --corpus-dir /data/cjk-corpus --dry-run
  python scripts/benchmark_vertical_text.py --synthetic
  python scripts/benchmark_vertical_text.py --synthetic --sample-size 50
  python scripts/benchmark_vertical_text.py --synthetic --output results.json
  python scripts/benchmark_vertical_text.py --synthetic --output-md report.md
        """,
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        help="Path to labeled corpus directory",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic test data (no corpus needed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate corpus structure without running detection",
    )
    parser.add_argument(
        "--generate-synthetic",
        type=int,
        metavar="N",
        help="Generate N synthetic test images with ground truth sidecars",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for --generate-synthetic (default: ./synthetic_corpus)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Number of synthetic test cases (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON path for structured results",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        help="Output markdown path for human-readable report",
    )
    parser.add_argument(
        "--target-direction-accuracy",
        type=float,
        default=90.0,
        help="Target direction detection accuracy %% (default: 90.0)",
    )
    parser.add_argument(
        "--target-ordering-tau",
        type=float,
        default=0.80,
        help="Target Kendall's tau for ordering (default: 0.80)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def main(argv=None):
    """CLI entry point for vertical text benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Generate synthetic corpus mode
    if args.generate_synthetic is not None:
        out_dir = args.output_dir or os.path.join(os.getcwd(), "synthetic_corpus")
        logger.info(
            "Generating %d synthetic test cases in %s...",
            args.generate_synthetic,
            out_dir,
        )
        summary = generate_synthetic_corpus(
            output_dir=out_dir,
            num_samples=args.generate_synthetic,
        )
        print(f"\nGenerated {summary['total_generated']} test cases in {out_dir}")
        for cat, count in summary.get("per_category", {}).items():
            print(f"  {cat}: {count}")
        if not summary.get("pil_available"):
            print("  (PIL not available -- image files were not generated)")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return 0

    if not args.synthetic and not args.corpus_dir:
        parser.error("Either --synthetic or --corpus-dir is required")

    # Dry-run mode: validate corpus structure only
    if args.dry_run:
        if not args.corpus_dir:
            parser.error("--dry-run requires --corpus-dir")

        logger.info("Validating corpus structure at %s...", args.corpus_dir)
        validation = validate_corpus_structure(args.corpus_dir)

        print("")
        print("=" * 60)
        print("CORPUS STRUCTURE VALIDATION")
        print("=" * 60)
        print(f"  Valid:       {validation['valid']}")
        print(f"  Total Cases: {validation['total_cases']}")
        print("")

        for cat_name, cat_info in validation.get("categories", {}).items():
            print(f"  {cat_name}/")
            print(f"    Exists:          {cat_info['exists']}")
            print(f"    Images:          {cat_info['image_count']}")
            print(f"    Sidecars:        {cat_info['sidecar_count']}")
            print(f"    Valid Pairs:     {cat_info['valid_pairs']}")
            if cat_info.get("missing_sidecars"):
                print(f"    Missing Sidecars: {cat_info['missing_sidecars'][:5]}")
            if cat_info.get("invalid_sidecars"):
                print(f"    Invalid Sidecars: {cat_info['invalid_sidecars'][:5]}")
            print("")

        if validation["errors"]:
            print("ERRORS:")
            for err in validation["errors"]:
                print(f"  - {err}")
            print("")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(validation, f, indent=2)
            logger.info("Validation results saved to %s", args.output)

        return 0 if validation["valid"] else 1

    # Run benchmark
    if args.synthetic:
        logger.info(
            "Running synthetic vertical text benchmark (%d cases)...",
            args.sample_size,
        )
        result = run_synthetic_benchmark(
            sample_size=args.sample_size,
            target_direction_accuracy=args.target_direction_accuracy,
            target_ordering_tau=args.target_ordering_tau,
        )
    else:
        logger.info(
            "Running corpus vertical text benchmark from %s...",
            args.corpus_dir,
        )
        result = run_corpus_benchmark(
            corpus_dir=args.corpus_dir,
            target_direction_accuracy=args.target_direction_accuracy,
            target_ordering_tau=args.target_ordering_tau,
        )

    # Display report
    report = format_benchmark_report(result)
    print(report)

    # Save JSON if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info("Results saved to %s", args.output)

    # Save markdown if requested
    if args.output_md:
        md_report = format_markdown_report(result)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md_report)
        logger.info("Markdown report saved to %s", args.output_md)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
