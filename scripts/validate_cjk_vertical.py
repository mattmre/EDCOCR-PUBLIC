"""CJK vertical text real-world validation suite with synthetic corpus.

Generates synthetic test cases using PIL (vertical columns, mixed layouts,
horizontal baselines, multi-column vertical, small text blocks) and validates
the vertical_text module's detection, column grouping, and reading order
accuracy against known ground truth.

Metrics:
  - Detection accuracy (vertical vs horizontal classification)
  - Column grouping precision
  - Reading order accuracy (Kendall's tau rank correlation)

Usage:
    python scripts/validate_cjk_vertical.py --generate --output-dir validation_data/cjk
    python scripts/validate_cjk_vertical.py --validate --corpus-dir validation_data/cjk
    python scripts/validate_cjk_vertical.py --validate --synthetic
    python scripts/validate_cjk_vertical.py --validate --synthetic --output results.json

Requires: vertical_text.py (from EDCOCR project root)
"""

from __future__ import annotations

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
# Graceful imports
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
    logger.warning("vertical_text module not available.")
    _VERTICAL_TEXT_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIDECAR_SUFFIX = ".expected.json"
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# CJK characters for synthetic generation
_CJK_CHARS_ZH = list("\u4e2d\u6587\u6d4b\u8bd5\u5c71\u6c34\u98ce\u6708\u5929\u5730\u4eba\u751f")
_CJK_CHARS_JA = list("\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8\u6771\u4eac\u5927\u962a\u6625\u590f")
_CJK_CHARS_KO = list("\ud55c\uad6d\uc5b4\ud14c\uc2a4\ud2b8\uc11c\uc6b8\ubd80\uc0b0\ub300\uad6c")
_CJK_CHARS_ALL = _CJK_CHARS_ZH + _CJK_CHARS_JA + _CJK_CHARS_KO

CORPUS_CATEGORIES = (
    "vertical_only",
    "mixed_layout",
    "horizontal_only",
    "multi_column",
    "small_text",
)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class ValidationMetrics:
    """Validation metrics for the CJK vertical text suite."""

    total_cases: int = 0
    direction_correct: int = 0
    direction_accuracy: float = 0.0
    avg_kendall_tau: float = 0.0
    column_grouping_precision: float = 0.0
    avg_inference_time_ms: float = 0.0
    per_category: dict = field(default_factory=dict)
    passed: bool = False
    target_direction_accuracy: float = 90.0
    target_ordering_tau: float = 0.80
    dataset: str = ""
    timestamp: str = ""


@dataclass
class CategoryResult:
    """Per-category validation result."""

    category: str = ""
    total_cases: int = 0
    direction_correct: int = 0
    direction_accuracy: float = 0.0
    avg_kendall_tau: float = 0.0
    column_precision: float = 0.0
    avg_inference_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------


def _make_box(x1, y1, x2, y2):
    """Create a 4-point polygon from corner coordinates."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _make_vertical_line(text, x_center, y_top, width=30, height=200, conf=0.92):
    """Create a vertical OCR line tuple (height >> width)."""
    x1 = x_center - width // 2
    box = _make_box(x1, y_top, x1 + width, y_top + height)
    return (text, box, conf)


def _make_horizontal_line(text, x1, y1, width=400, height=30, conf=0.92):
    """Create a horizontal OCR line tuple (width >> height)."""
    box = _make_box(x1, y1, x1 + width, y1 + height)
    return (text, box, conf)


# ---------------------------------------------------------------------------
# Kendall's tau computation
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

    expected_ranks = {text: i for i, text in enumerate(expected_order)}
    common_predicted = [t for t in predicted_order if t in expected_ranks]
    if len(common_predicted) < 2:
        return 0.0

    n = len(common_predicted)
    expected_rank_values = [expected_ranks[t] for t in common_predicted]

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = i - j
            exp_diff = expected_rank_values[i] - expected_rank_values[j]
            product = pred_diff * exp_diff
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.0

    return (concordant - discordant) / total_pairs


def compute_column_precision(predicted_columns: int, expected_columns: int) -> float:
    """Compute column count precision.

    Returns 1.0 if column counts match exactly, otherwise a ratio.

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


# ---------------------------------------------------------------------------
# Synthetic test case generators
# ---------------------------------------------------------------------------


def generate_vertical_only_case(rng, case_id, num_columns=3, lines_per_col=4,
                                page_width=1000, page_height=1200):
    """Generate a pure vertical text test case.

    All text boxes are arranged as tall, narrow columns with CJK characters
    read top-to-bottom within each column, columns read right-to-left.

    Returns
    -------
    dict
        Test case sidecar dict with ocr_lines, expected_order, etc.
    """
    ocr_lines = []
    expected_order = []
    col_spacing = page_width // (num_columns + 1)

    # Columns right-to-left
    for col_idx in range(num_columns):
        x_center = page_width - (col_idx + 1) * col_spacing
        for line_idx in range(lines_per_col):
            chars = "".join(rng.choices(_CJK_CHARS_ALL, k=rng.randint(2, 4)))
            text = f"{chars}_{col_idx}_{line_idx}"
            y_top = 50 + line_idx * 220
            line = _make_vertical_line(text, x_center, y_top)
            ocr_lines.append(line)
            expected_order.append(text)

    shuffled_lines = list(ocr_lines)
    rng.shuffle(shuffled_lines)

    return {
        "id": case_id,
        "direction": "vertical",
        "expected_order": expected_order,
        "expected_columns": num_columns,
        "ocr_lines": [
            {"text": t, "box": b, "confidence": c} for t, b, c in shuffled_lines
        ],
        "page_width": page_width,
    }


def generate_horizontal_only_case(rng, case_id, num_lines=5, page_width=1000):
    """Generate a horizontal-only test case (negative case).

    All text boxes are wide and short -- should NOT be classified as vertical.

    Returns
    -------
    dict
        Test case sidecar dict.
    """
    ocr_lines = []
    expected_order = []

    for i in range(num_lines):
        text = f"Line {i + 1}: English text sample content"
        y_top = 50 + i * 60
        line = _make_horizontal_line(text, 50, y_top)
        ocr_lines.append(line)
        expected_order.append(text)

    shuffled_lines = list(ocr_lines)
    rng.shuffle(shuffled_lines)

    return {
        "id": case_id,
        "direction": "horizontal",
        "expected_order": expected_order,
        "expected_columns": 0,
        "ocr_lines": [
            {"text": t, "box": b, "confidence": c} for t, b, c in shuffled_lines
        ],
        "page_width": page_width,
    }


def generate_mixed_layout_case(rng, case_id, num_v_columns=2, v_lines_per_col=3,
                               num_h_lines=3, page_width=1000):
    """Generate a mixed-layout test case.

    Horizontal headers at the top of the page, vertical CJK columns below.

    Returns
    -------
    dict
        Test case sidecar dict.
    """
    ocr_lines = []
    expected_order = []

    # Horizontal headers at the top
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
            chars = "".join(rng.choices(_CJK_CHARS_JA, k=rng.randint(2, 4)))
            text = f"{chars}_{col_idx}_{line_idx}"
            y_top = 200 + line_idx * 220
            line = _make_vertical_line(text, x_center, y_top)
            ocr_lines.append(line)
            expected_order.append(text)

    shuffled_lines = list(ocr_lines)
    rng.shuffle(shuffled_lines)

    return {
        "id": case_id,
        "direction": "mixed",
        "expected_order": expected_order,
        "expected_columns": num_v_columns,
        "ocr_lines": [
            {"text": t, "box": b, "confidence": c} for t, b, c in shuffled_lines
        ],
        "page_width": page_width,
    }


def generate_multi_column_case(rng, case_id, num_columns=5, lines_per_col=6,
                               page_width=1200):
    """Generate a multi-column vertical layout (5+ columns, dense).

    Stress-tests column grouping with many closely-spaced vertical columns.

    Returns
    -------
    dict
        Test case sidecar dict.
    """
    ocr_lines = []
    expected_order = []
    col_spacing = page_width // (num_columns + 1)

    for col_idx in range(num_columns):
        x_center = page_width - (col_idx + 1) * col_spacing
        for line_idx in range(lines_per_col):
            chars = "".join(rng.choices(_CJK_CHARS_ZH, k=rng.randint(3, 5)))
            text = f"{chars}_mc{col_idx}_{line_idx}"
            y_top = 30 + line_idx * 180
            line = _make_vertical_line(text, x_center, y_top)
            ocr_lines.append(line)
            expected_order.append(text)

    shuffled_lines = list(ocr_lines)
    rng.shuffle(shuffled_lines)

    return {
        "id": case_id,
        "direction": "vertical",
        "expected_order": expected_order,
        "expected_columns": num_columns,
        "ocr_lines": [
            {"text": t, "box": b, "confidence": c} for t, b, c in shuffled_lines
        ],
        "page_width": page_width,
    }


def generate_small_text_case(rng, case_id, page_width=600):
    """Generate a small text block edge case.

    Few vertical boxes with minimal character counts, testing the
    min_chars boundary behavior and edge-case geometry.

    Returns
    -------
    dict
        Test case sidecar dict.
    """
    ocr_lines = []
    expected_order = []

    # 2 small vertical boxes at distinct x positions
    for col_idx in range(2):
        x_center = page_width - (col_idx + 1) * 200
        for line_idx in range(2):
            chars = "".join(rng.choices(_CJK_CHARS_KO, k=2))
            text = f"{chars}_sm{col_idx}_{line_idx}"
            y_top = 50 + line_idx * 120
            # Smaller height, still above 2.0 aspect ratio
            line = _make_vertical_line(text, x_center, y_top, width=25, height=80)
            ocr_lines.append(line)
            expected_order.append(text)

    shuffled_lines = list(ocr_lines)
    rng.shuffle(shuffled_lines)

    return {
        "id": case_id,
        "direction": "vertical",
        "expected_order": expected_order,
        "expected_columns": 2,
        "ocr_lines": [
            {"text": t, "box": b, "confidence": c} for t, b, c in shuffled_lines
        ],
        "page_width": page_width,
    }


# ---------------------------------------------------------------------------
# Synthetic corpus generation with images
# ---------------------------------------------------------------------------


def _draw_sidecar_to_image(sidecar: dict, page_width: int, page_height: int):
    """Render a sidecar's OCR lines to a PIL Image if PIL is available.

    Parameters
    ----------
    sidecar : dict
        Test case sidecar with ocr_lines.
    page_width : int
        Image width in pixels.
    page_height : int
        Image height in pixels.

    Returns
    -------
    Image or None
        PIL Image if available, else None.
    """
    if not _PIL_AVAILABLE:
        return None

    img = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for entry in sidecar.get("ocr_lines", []):
        box = entry["box"]
        x1 = min(pt[0] for pt in box)
        y1 = min(pt[1] for pt in box)
        x2 = max(pt[0] for pt in box)
        y2 = max(pt[1] for pt in box)
        draw.rectangle([x1, y1, x2, y2], outline="gray")
        draw.text((x1 + 2, y1 + 2), entry["text"], fill="black", font=font)

    return img


def generate_corpus(output_dir: str, num_samples: int = 20, seed: int = 42) -> dict:
    """Generate a synthetic validation corpus with images and sidecar JSON files.

    Creates the corpus directory structure with five categories:
    vertical_only, mixed_layout, horizontal_only, multi_column, small_text.

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
        Summary with total_generated, per_category counts, output_dir.
    """
    rng = random.Random(seed)
    output_path = Path(output_dir)

    n_vert = max(1, int(num_samples * 0.25))
    n_horiz = max(1, int(num_samples * 0.20))
    n_mixed = max(1, int(num_samples * 0.25))
    n_multi = max(1, int(num_samples * 0.15))
    n_small = max(1, num_samples - n_vert - n_horiz - n_mixed - n_multi)

    allocations = {
        "vertical_only": (n_vert, generate_vertical_only_case),
        "horizontal_only": (n_horiz, generate_horizontal_only_case),
        "mixed_layout": (n_mixed, generate_mixed_layout_case),
        "multi_column": (n_multi, generate_multi_column_case),
        "small_text": (n_small, generate_small_text_case),
    }

    summary = {
        "total_generated": 0,
        "per_category": {},
        "output_dir": str(output_dir),
        "pil_available": _PIL_AVAILABLE,
    }

    for cat_name, (count, generator) in allocations.items():
        cat_dir = output_path / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            case_id = f"{cat_name}_{i:04d}"

            if generator == generate_vertical_only_case:
                sidecar = generator(
                    rng, case_id,
                    num_columns=rng.randint(2, 4),
                    lines_per_col=rng.randint(3, 5),
                )
            elif generator == generate_horizontal_only_case:
                sidecar = generator(
                    rng, case_id,
                    num_lines=rng.randint(3, 8),
                )
            elif generator == generate_mixed_layout_case:
                sidecar = generator(
                    rng, case_id,
                    num_v_columns=rng.randint(1, 3),
                    v_lines_per_col=rng.randint(2, 4),
                    num_h_lines=rng.randint(2, 4),
                )
            elif generator == generate_multi_column_case:
                sidecar = generator(
                    rng, case_id,
                    num_columns=rng.randint(4, 6),
                    lines_per_col=rng.randint(4, 7),
                )
            else:
                sidecar = generator(rng, case_id)

            # Write sidecar JSON
            sidecar_path = cat_dir / f"{case_id}{SIDECAR_SUFFIX}"
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2, ensure_ascii=False)

            # Write image if PIL available
            page_width = sidecar.get("page_width", 1000)
            page_height = 1200
            img = _draw_sidecar_to_image(sidecar, page_width, page_height)
            if img is not None:
                img.save(str(cat_dir / f"{case_id}.png"))

        summary["per_category"][cat_name] = count
        summary["total_generated"] += count

    return summary


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------


def _parse_ocr_lines(sidecar_data: dict) -> list:
    """Parse OCR lines from sidecar JSON into (text, box, confidence) tuples."""
    lines = []
    for entry in sidecar_data.get("ocr_lines", []):
        text = entry.get("text", "")
        box = entry.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]])
        confidence = entry.get("confidence", 0.9)
        lines.append((text, box, confidence))
    return lines


def validate_case(sidecar: dict) -> dict:
    """Validate a single test case against the vertical_text module.

    Parameters
    ----------
    sidecar : dict
        Test case sidecar data.

    Returns
    -------
    dict
        Per-case result with direction_correct, kendall_tau,
        column_precision, inference_time_ms.
    """
    if not _VERTICAL_TEXT_AVAILABLE:
        return {"error": "vertical_text module not available"}

    ocr_lines = _parse_ocr_lines(sidecar)
    expected_direction = sidecar.get("direction", "horizontal")
    expected_order = sidecar.get("expected_order", [])
    expected_columns = sidecar.get("expected_columns", 0)
    page_width = sidecar.get("page_width", 1000)

    # Direction detection
    start = time.perf_counter()
    direction_result = classify_page_text_direction(ocr_lines)
    elapsed_ms = (time.perf_counter() - start) * 1000

    predicted_direction = direction_result.get("direction", "horizontal")
    direction_correct = predicted_direction == expected_direction

    # Reading order
    tau = 0.0
    if expected_order and ocr_lines:
        sorted_result = sort_mixed_reading_order(ocr_lines, page_width)
        predicted_order = [item[0] for item in sorted_result]
        tau = compute_kendall_tau(predicted_order, expected_order)

    # Column grouping
    col_prec = 0.0
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
    elif expected_columns == 0:
        col_prec = 1.0  # Correctly no columns expected

    return {
        "direction_correct": direction_correct,
        "predicted_direction": predicted_direction,
        "expected_direction": expected_direction,
        "kendall_tau": round(tau, 4),
        "column_precision": round(col_prec, 4),
        "inference_time_ms": round(elapsed_ms, 3),
    }


def run_validation(
    cases: list,
    target_direction_accuracy: float = 90.0,
    target_ordering_tau: float = 0.80,
    dataset_label: str = "synthetic",
) -> ValidationMetrics:
    """Run validation against a list of test cases.

    Parameters
    ----------
    cases : list[dict]
        List of dicts with keys: id, category, sidecar.
    target_direction_accuracy : float
        Target direction detection accuracy percentage.
    target_ordering_tau : float
        Target Kendall's tau for reading order.
    dataset_label : str
        Label for the dataset.

    Returns
    -------
    ValidationMetrics
        Aggregated validation metrics.
    """
    all_dir_correct = 0
    all_taus = []
    all_col_precs = []
    all_timings = []
    per_cat = {}

    for case in cases:
        cat_name = case.get("category", "unknown")
        sidecar = case.get("sidecar", case)

        if cat_name not in per_cat:
            per_cat[cat_name] = CategoryResult(category=cat_name)

        cat = per_cat[cat_name]
        result = validate_case(sidecar)

        if "error" in result:
            continue

        cat.total_cases += 1
        all_timings.append(result["inference_time_ms"])

        if result["direction_correct"]:
            cat.direction_correct += 1
            all_dir_correct += 1

        if result["kendall_tau"] != 0.0 or sidecar.get("expected_order"):
            all_taus.append(result["kendall_tau"])

        all_col_precs.append(result["column_precision"])

    # Aggregate per-category
    for cat in per_cat.values():
        if cat.total_cases > 0:
            cat.direction_accuracy = round(
                cat.direction_correct / cat.total_cases * 100, 2
            )

    total = sum(c.total_cases for c in per_cat.values())
    direction_accuracy = (
        round(all_dir_correct / total * 100, 2) if total > 0 else 0.0
    )
    avg_tau = round(statistics.mean(all_taus), 4) if all_taus else 0.0
    avg_col_prec = round(statistics.mean(all_col_precs), 4) if all_col_precs else 0.0
    avg_timing = round(statistics.mean(all_timings), 2) if all_timings else 0.0

    passed = (
        direction_accuracy >= target_direction_accuracy
        and (avg_tau >= target_ordering_tau if all_taus else True)
    )

    return ValidationMetrics(
        total_cases=total,
        direction_correct=all_dir_correct,
        direction_accuracy=direction_accuracy,
        avg_kendall_tau=avg_tau,
        column_grouping_precision=avg_col_prec,
        avg_inference_time_ms=avg_timing,
        per_category={k: asdict(v) for k, v in per_cat.items()},
        passed=passed,
        target_direction_accuracy=target_direction_accuracy,
        target_ordering_tau=target_ordering_tau,
        dataset=dataset_label,
        timestamp=datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="milliseconds"),
    )


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_corpus_cases(corpus_dir: str) -> list:
    """Load test cases from a corpus directory.

    Parameters
    ----------
    corpus_dir : str
        Path to corpus root directory.

    Returns
    -------
    list[dict]
        List of test case dicts with keys: id, category, sidecar.
    """
    cases = []
    corpus_path = Path(corpus_dir)

    if not corpus_path.is_dir():
        logger.error("Corpus directory does not exist: %s", corpus_dir)
        return cases

    for subdir in corpus_path.iterdir():
        if not subdir.is_dir():
            continue

        cat_name = subdir.name

        for f in sorted(subdir.iterdir()):
            if not f.name.endswith(SIDECAR_SUFFIX):
                continue

            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)

                stem = f.name.replace(SIDECAR_SUFFIX, "")
                cases.append({
                    "id": stem,
                    "category": cat_name,
                    "sidecar": data,
                })
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", f, exc)

    return cases


def generate_synthetic_cases(num_samples: int = 25, seed: int = 42) -> list:
    """Generate synthetic test cases in-memory (no disk I/O).

    Parameters
    ----------
    num_samples : int
        Total number of test cases.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        List of test case dicts with keys: id, category, sidecar.
    """
    rng = random.Random(seed)
    cases = []

    n_vert = max(1, int(num_samples * 0.25))
    n_horiz = max(1, int(num_samples * 0.20))
    n_mixed = max(1, int(num_samples * 0.25))
    n_multi = max(1, int(num_samples * 0.15))
    n_small = max(1, num_samples - n_vert - n_horiz - n_mixed - n_multi)

    specs = []
    for i in range(n_vert):
        specs.append(("vertical_only", generate_vertical_only_case, {
            "num_columns": rng.randint(2, 4),
            "lines_per_col": rng.randint(3, 5),
        }))
    for i in range(n_horiz):
        specs.append(("horizontal_only", generate_horizontal_only_case, {
            "num_lines": rng.randint(3, 8),
        }))
    for i in range(n_mixed):
        specs.append(("mixed_layout", generate_mixed_layout_case, {
            "num_v_columns": rng.randint(1, 3),
            "v_lines_per_col": rng.randint(2, 4),
            "num_h_lines": rng.randint(2, 4),
        }))
    for i in range(n_multi):
        specs.append(("multi_column", generate_multi_column_case, {
            "num_columns": rng.randint(4, 6),
            "lines_per_col": rng.randint(4, 7),
        }))
    for i in range(n_small):
        specs.append(("small_text", generate_small_text_case, {}))

    for idx, (cat, gen, kwargs) in enumerate(specs):
        case_id = f"{cat}_{idx:04d}"
        sidecar = gen(rng, case_id, **kwargs)
        cases.append({
            "id": case_id,
            "category": cat,
            "sidecar": sidecar,
        })

    return cases


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(metrics: ValidationMetrics) -> str:
    """Format validation metrics as a human-readable report.

    Parameters
    ----------
    metrics : ValidationMetrics
        Validation result.

    Returns
    -------
    str
        Formatted report.
    """
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("CJK VERTICAL TEXT VALIDATION REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Dataset:               {metrics.dataset}")
    lines.append(f"  Total Cases:           {metrics.total_cases}")
    lines.append(f"  Direction Accuracy:    {metrics.direction_accuracy:.2f}%"
                 f"  (target: {metrics.target_direction_accuracy:.1f}%)")
    lines.append(f"  Avg Kendall's Tau:     {metrics.avg_kendall_tau:.4f}"
                 f"  (target: {metrics.target_ordering_tau:.2f})")
    lines.append(f"  Column Precision:      {metrics.column_grouping_precision:.4f}")
    lines.append(f"  Avg Inference Time:    {metrics.avg_inference_time_ms:.2f} ms")
    lines.append(f"  Result:                {'PASS' if metrics.passed else 'FAIL'}")
    lines.append(f"  Timestamp:             {metrics.timestamp}")
    lines.append("")

    if metrics.per_category:
        lines.append("PER-CATEGORY BREAKDOWN")
        lines.append("-" * 72)
        lines.append(
            f"{'Category':<20} {'Cases':>8} {'Dir.Acc':>10} "
            f"{'Tau':>10} {'ColPrec':>10}"
        )
        lines.append("-" * 72)

        for cat_name in sorted(metrics.per_category.keys()):
            m = metrics.per_category[cat_name]
            lines.append(
                f"{cat_name:<20} {m.get('total_cases', 0):>8d} "
                f"{m.get('direction_accuracy', 0.0):>10.2f} "
                f"{m.get('avg_kendall_tau', 0.0):>10.4f} "
                f"{m.get('column_precision', 0.0):>10.4f}"
            )
        lines.append("-" * 72)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
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
            "CJK vertical text real-world validation suite with "
            "synthetic corpus generation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_cjk_vertical.py --generate --output-dir validation_data/cjk
  python scripts/validate_cjk_vertical.py --validate --corpus-dir validation_data/cjk
  python scripts/validate_cjk_vertical.py --validate --synthetic
  python scripts/validate_cjk_vertical.py --validate --synthetic --output results.json
        """,
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic corpus with images and sidecar files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation against corpus or synthetic data",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use in-memory synthetic test data (no corpus directory required)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        help="Path to corpus directory (for --generate output or --validate input)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Alias for --corpus-dir when used with --generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON path for structured results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        help="Number of synthetic test cases to generate (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.generate and not args.validate:
        parser.error("Either --generate or --validate is required")

    # Generate mode
    if args.generate:
        out_dir = args.output_dir or args.corpus_dir
        if not out_dir:
            out_dir = os.path.join(os.getcwd(), "validation_data", "cjk")

        logger.info(
            "Generating %d synthetic test cases in %s...",
            args.num_samples,
            out_dir,
        )
        summary = generate_corpus(
            output_dir=out_dir,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        print(f"\nGenerated {summary['total_generated']} test cases in {out_dir}")
        for cat, count in sorted(summary.get("per_category", {}).items()):
            print(f"  {cat}: {count}")
        if not summary.get("pil_available"):
            print("  (PIL not available -- image files were not generated)")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return 0

    # Validate mode
    if args.validate:
        if args.synthetic:
            logger.info(
                "Running synthetic validation (%d cases)...", args.num_samples,
            )
            cases = generate_synthetic_cases(
                num_samples=args.num_samples, seed=args.seed,
            )
            dataset_label = "synthetic"
        elif args.corpus_dir:
            logger.info("Loading corpus from %s...", args.corpus_dir)
            cases = load_corpus_cases(args.corpus_dir)
            dataset_label = args.corpus_dir
            if not cases:
                print("No valid test cases found in corpus directory.")
                return 1
        else:
            parser.error("--validate requires either --synthetic or --corpus-dir")
            return 1

        metrics = run_validation(
            cases,
            target_direction_accuracy=args.target_direction_accuracy,
            target_ordering_tau=args.target_ordering_tau,
            dataset_label=dataset_label,
        )

        report = format_report(metrics)
        print(report)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(asdict(metrics), f, indent=2)
            logger.info("Results saved to %s", args.output)

        return 0 if metrics.passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
