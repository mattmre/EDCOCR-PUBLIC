"""CJK vertical text detection, reading order, and rotation for forensic OCR pipeline.

Detects vertical text layouts common in Chinese, Japanese, and Korean documents
using bounding box geometry analysis. Provides:

- Text direction classification (horizontal, vertical, mixed)
- Column grouping for vertical text regions
- Reading order re-sorting (right-to-left columns, top-to-bottom within)
- Vertical text crop rotation for improved OCR recognition
- Analysis sidecar output: EXPORT/VERTICAL/<subfolder>/<doc>.vertical.json

Opt-in via ENABLE_VERTICAL_TEXT env var (default: disabled).

Graceful degradation: all functions return safe defaults on error.
No new external dependencies -- only stdlib, Pillow (already required).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    sanitize_path_segment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_VERTICAL_TEXT = os.environ.get("ENABLE_VERTICAL_TEXT", "false").lower() in ("1", "true", "yes")

VERTICAL_ASPECT_RATIO_THRESHOLD = 2.0  # height/width ratio to classify as vertical
COLUMN_GROUPING_TOLERANCE = 0.05  # fraction of page width for column grouping
CJK_LANGUAGES = {"ch", "chinese_cht", "japan", "korean"}

CJK_UNICODE_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0xFF00, 0xFFEF),   # Fullwidth Forms
]

# ---------------------------------------------------------------------------
# Guarded PIL import
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage  # noqa: F401 — used to test availability

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------


def is_cjk_char(char: str) -> bool:
    """Check if a character is in CJK Unicode ranges.

    Args:
        char: A single character string.

    Returns:
        True if the character falls within any defined CJK Unicode range.
    """
    if not char or len(char) != 1:
        return False
    codepoint = ord(char)
    for start, end in CJK_UNICODE_RANGES:
        if start <= codepoint <= end:
            return True
    return False


def contains_cjk(text: str) -> bool:
    """Check if text contains any CJK characters.

    Args:
        text: Input text string.

    Returns:
        True if at least one CJK character is found.
    """
    if not text:
        return False
    for char in text:
        if is_cjk_char(char):
            return True
    return False


# ---------------------------------------------------------------------------
# Geometry analysis
# ---------------------------------------------------------------------------


def _box_dimensions(box_points) -> tuple[float, float]:
    """Compute width and height from a 4-point polygon.

    Args:
        box_points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] from PaddleOCR.

    Returns:
        (width, height) tuple.
    """
    if not box_points or len(box_points) < 4:
        return (0.0, 0.0)

    try:
        xs = [float(pt[0]) for pt in box_points]
        ys = [float(pt[1]) for pt in box_points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return (max(width, 0.0), max(height, 0.0))
    except (TypeError, IndexError, ValueError):
        return (0.0, 0.0)


def _box_center_x(box_points) -> float:
    """Get the horizontal center of a bounding box.

    Args:
        box_points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].

    Returns:
        X coordinate of the box center.
    """
    if not box_points or len(box_points) < 4:
        return 0.0
    try:
        xs = [float(pt[0]) for pt in box_points]
        return (min(xs) + max(xs)) / 2.0
    except (TypeError, IndexError, ValueError):
        return 0.0


def _box_top_y(box_points) -> float:
    """Get the top (minimum) Y coordinate of a bounding box.

    Args:
        box_points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].

    Returns:
        Minimum Y coordinate.
    """
    if not box_points or len(box_points) < 4:
        return 0.0
    try:
        ys = [float(pt[1]) for pt in box_points]
        return min(ys)
    except (TypeError, IndexError, ValueError):
        return 0.0


def is_vertical_text_box(box_points, text: str = "", min_chars: int = 2) -> bool:
    """Determine if a detected text box represents vertical text.

    A text box is classified as vertical when its height-to-width aspect ratio
    exceeds the threshold. Single characters are ambiguous and are excluded by
    default.

    Args:
        box_points: 4-point polygon from PaddleOCR [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        text: OCR text content (used for min_chars check).
        min_chars: Minimum character count to consider (default 2).

    Returns:
        True if the bounding box geometry indicates vertical text.
    """
    if not box_points or len(box_points) < 4:
        return False

    # Single characters are ambiguous for direction detection
    if text and len(text.strip()) < min_chars:
        return False

    width, height = _box_dimensions(box_points)
    if width <= 0:
        return False

    aspect_ratio = height / width
    return aspect_ratio >= VERTICAL_ASPECT_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Page-level direction classification
# ---------------------------------------------------------------------------


def classify_page_text_direction(ocr_results: list, lang: str = None) -> dict:
    """Classify the predominant text direction of a page.

    Args:
        ocr_results: List of (text, box_points, confidence) tuples from PaddleOCR.
        lang: Detected language code (optional hint).

    Returns:
        Dict with keys:
            direction: "horizontal", "vertical", or "mixed"
            vertical_ratio: float 0-1
            vertical_count: int
            horizontal_count: int
    """
    default = {
        "direction": "horizontal",
        "vertical_ratio": 0.0,
        "vertical_count": 0,
        "horizontal_count": 0,
    }

    if not ocr_results:
        return default

    vertical_count = 0
    horizontal_count = 0

    for item in ocr_results:
        try:
            text, box, _conf = item
            if is_vertical_text_box(box, text=str(text)):
                vertical_count += 1
            else:
                horizontal_count += 1
        except (ValueError, TypeError):
            horizontal_count += 1

    total = vertical_count + horizontal_count
    if total == 0:
        return default

    vertical_ratio = vertical_count / total

    if vertical_ratio >= 0.6:
        direction = "vertical"
    elif vertical_ratio >= 0.2:
        direction = "mixed"
    else:
        direction = "horizontal"

    return {
        "direction": direction,
        "vertical_ratio": round(vertical_ratio, 4),
        "vertical_count": vertical_count,
        "horizontal_count": horizontal_count,
    }


# ---------------------------------------------------------------------------
# Column grouping
# ---------------------------------------------------------------------------


def group_vertical_columns(
    vertical_boxes: list,
    page_width: float,
    tolerance: float = None,
) -> list:
    """Group vertical text boxes into columns by x-coordinate proximity.

    Args:
        vertical_boxes: List of (box, text, confidence) tuples for vertical lines.
        page_width: Width of the page in pixels.
        tolerance: X-coordinate grouping tolerance as fraction of page_width.
            Defaults to COLUMN_GROUPING_TOLERANCE.

    Returns:
        List of columns, each column is a list of (box, text, confidence) sorted
        top-to-bottom by Y coordinate.
    """
    if not vertical_boxes:
        return []

    if tolerance is None:
        tolerance = COLUMN_GROUPING_TOLERANCE

    tol_px = max(page_width * tolerance, 1.0) if page_width > 0 else 10.0

    # Sort by x-center descending (right-to-left) for natural CJK order
    sorted_boxes = sorted(
        vertical_boxes,
        key=lambda item: _box_center_x(item[0]),
        reverse=True,
    )

    columns = []  # Each column: {"x_center": float, "items": [(box, text, conf)]}

    for box, text, conf in sorted_boxes:
        center_x = _box_center_x(box)
        placed = False
        for col in columns:
            if abs(center_x - col["x_center"]) <= tol_px:
                col["items"].append((box, text, conf))
                # Update running average for column center
                col["x_values"].append(center_x)
                col["x_center"] = sum(col["x_values"]) / len(col["x_values"])
                placed = True
                break
        if not placed:
            columns.append({
                "x_center": center_x,
                "x_values": [center_x],
                "items": [(box, text, conf)],
            })

    # Sort items within each column by top-Y (top-to-bottom)
    result = []
    for col in columns:
        sorted_items = sorted(col["items"], key=lambda item: _box_top_y(item[0]))
        result.append(sorted_items)

    return result


# ---------------------------------------------------------------------------
# Reading order sorting
# ---------------------------------------------------------------------------


def sort_vertical_reading_order(columns: list) -> list:
    """Sort grouped columns in traditional CJK right-to-left reading order.

    Columns are sorted right-to-left (highest x first).
    Within each column, lines are sorted top-to-bottom (lowest y first).

    Args:
        columns: List of columns as returned by group_vertical_columns.

    Returns:
        Flat list of (box, text, confidence) in reading order.
    """
    if not columns:
        return []

    # Columns should already be sorted right-to-left from group_vertical_columns,
    # but enforce it by sorting on the average x-center of the first item.
    def _col_x(col):
        if col:
            return _box_center_x(col[0][0])
        return 0.0

    sorted_cols = sorted(columns, key=_col_x, reverse=True)

    result = []
    for col in sorted_cols:
        # Items within column already sorted top-to-bottom
        result.extend(col)

    return result


def sort_mixed_reading_order(
    ocr_lines: list,
    page_width: float,
    lang: str = None,
) -> list:
    """Sort a mixed page with both horizontal and vertical text.

    Strategy:
    1. Separate horizontal and vertical lines
    2. Sort horizontal lines normally (top-to-bottom, left-to-right)
    3. Group and sort vertical lines (right-to-left columns, top-to-bottom within)
    4. Interleave based on vertical position (top content first)

    Args:
        ocr_lines: List of (text, box_points, confidence) tuples.
        page_width: Width of the page in pixels.
        lang: Detected language code (optional).

    Returns:
        List of (text, box_points, confidence) in reading order.
    """
    if not ocr_lines:
        return []

    horizontal = []
    vertical = []

    for item in ocr_lines:
        try:
            text, box, conf = item
            if is_vertical_text_box(box, text=str(text)):
                vertical.append((box, text, conf))
            else:
                horizontal.append(item)
        except (ValueError, TypeError):
            horizontal.append(item)

    # Sort horizontal lines: top-to-bottom, then left-to-right
    horizontal_sorted = sorted(
        horizontal,
        key=lambda item: (_box_top_y(item[1]) if len(item) >= 2 else 0.0,
                          _box_center_x(item[1]) if len(item) >= 2 else 0.0),
    )

    if not vertical:
        return horizontal_sorted

    # Group and sort vertical columns
    columns = group_vertical_columns(vertical, page_width)
    vertical_sorted_raw = sort_vertical_reading_order(columns)

    # Convert back to (text, box, conf) format for consistency
    vertical_sorted = [(text, box, conf) for box, text, conf in vertical_sorted_raw]

    # Interleave: merge both lists by top-Y position
    merged = []
    h_idx = 0
    v_idx = 0

    while h_idx < len(horizontal_sorted) and v_idx < len(vertical_sorted):
        h_y = _box_top_y(horizontal_sorted[h_idx][1]) if len(horizontal_sorted[h_idx]) >= 2 else 0.0
        v_y = _box_top_y(vertical_sorted[v_idx][1]) if len(vertical_sorted[v_idx]) >= 2 else 0.0

        if h_y <= v_y:
            merged.append(horizontal_sorted[h_idx])
            h_idx += 1
        else:
            merged.append(vertical_sorted[v_idx])
            v_idx += 1

    # Append remaining items
    merged.extend(horizontal_sorted[h_idx:])
    merged.extend(vertical_sorted[v_idx:])

    return merged


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------


def rotate_vertical_crop(image, direction: str = "ccw"):
    """Rotate a vertical text crop to horizontal orientation for OCR.

    Vertical CJK text can benefit from rotation to horizontal before OCR
    to improve recognition accuracy with engines trained on horizontal text.

    Args:
        image: PIL Image of the cropped text region.
        direction: "ccw" (counter-clockwise 90) or "cw" (clockwise 90).

    Returns:
        Rotated PIL Image, or the original image if rotation is not possible.
    """
    if not _PIL_AVAILABLE:
        logger.debug("PIL not available; vertical crop rotation skipped.")
        return image

    if image is None:
        return image

    try:
        if direction == "ccw":
            return image.rotate(90, expand=True)
        elif direction == "cw":
            return image.rotate(-90, expand=True)
        else:
            logger.warning("Unknown rotation direction '%s'; returning original.", direction)
            return image
    except Exception as e:
        logger.warning("Vertical crop rotation failed: %s", e)
        return image


# ---------------------------------------------------------------------------
# Text extraction with vertical awareness
# ---------------------------------------------------------------------------


def extract_text_vertical_aware(
    ocr_results: list,
    page_width: float,
    lang: str = None,
) -> str:
    """Extract text from OCR results with vertical text awareness.

    Reorders text based on detected text direction and CJK reading order
    conventions. Horizontal pages are sorted normally; vertical pages use
    right-to-left column ordering; mixed pages interleave both.

    Args:
        ocr_results: List of (text, box_points, confidence) tuples.
        page_width: Width of the page in pixels.
        lang: Detected language code (optional).

    Returns:
        Extracted text in correct reading order, joined by newlines.
    """
    if not ocr_results:
        return ""

    direction_info = classify_page_text_direction(ocr_results, lang)
    direction = direction_info.get("direction", "horizontal")

    if direction == "horizontal":
        # Standard left-to-right, top-to-bottom
        sorted_lines = sorted(
            ocr_results,
            key=lambda item: (_box_top_y(item[1]) if len(item) >= 2 else 0.0,
                              _box_center_x(item[1]) if len(item) >= 2 else 0.0),
        )
    elif direction == "vertical":
        # All-vertical: group into columns, right-to-left
        vertical_boxes = [(box, text, conf) for text, box, conf in ocr_results]
        columns = group_vertical_columns(vertical_boxes, page_width)
        ordered_raw = sort_vertical_reading_order(columns)
        sorted_lines = [(text, box, conf) for box, text, conf in ordered_raw]
    else:
        # Mixed: interleave horizontal and vertical
        sorted_lines = sort_mixed_reading_order(ocr_results, page_width, lang)

    texts = []
    for item in sorted_lines:
        try:
            text = str(item[0]).strip()
            if text:
                texts.append(text)
        except (IndexError, TypeError):
            continue

    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VerticalTextAnalysis:
    """Analysis of vertical text on a page."""

    page_number: int
    direction: str = "horizontal"  # "horizontal", "vertical", "mixed"
    vertical_ratio: float = 0.0
    vertical_line_count: int = 0
    horizontal_line_count: int = 0
    column_count: int = 0
    lang_detected: str = ""
    reading_order_applied: bool = False


@dataclass
class DocumentVerticalText:
    """Vertical text analysis for an entire document."""

    document_id: str
    source_file: str
    pages: list = field(default_factory=list)  # List[VerticalTextAnalysis as dict]
    total_vertical_pages: int = 0
    total_mixed_pages: int = 0
    has_vertical_content: bool = False


# ---------------------------------------------------------------------------
# Page-level analysis
# ---------------------------------------------------------------------------


def analyze_page_vertical_text(
    ocr_results: list,
    page_number: int,
    page_width: float,
    lang: str = None,
) -> VerticalTextAnalysis:
    """Full analysis of vertical text on a single page.

    Args:
        ocr_results: List of (text, box_points, confidence) tuples.
        page_number: Page number (1-based).
        page_width: Width of the page in pixels.
        lang: Detected language code (optional).

    Returns:
        VerticalTextAnalysis dataclass with detection results.
    """
    analysis = VerticalTextAnalysis(page_number=page_number)

    if not ocr_results:
        return analysis

    try:
        direction_info = classify_page_text_direction(ocr_results, lang)
        analysis.direction = direction_info["direction"]
        analysis.vertical_ratio = direction_info["vertical_ratio"]
        analysis.vertical_line_count = direction_info["vertical_count"]
        analysis.horizontal_line_count = direction_info["horizontal_count"]
        analysis.lang_detected = lang or ""

        # Count columns for vertical/mixed pages
        if analysis.direction in ("vertical", "mixed") and page_width > 0:
            vertical_boxes = []
            for item in ocr_results:
                try:
                    text, box, conf = item
                    if is_vertical_text_box(box, text=str(text)):
                        vertical_boxes.append((box, text, conf))
                except (ValueError, TypeError):
                    continue

            if vertical_boxes:
                columns = group_vertical_columns(vertical_boxes, page_width)
                analysis.column_count = len(columns)
                analysis.reading_order_applied = True

    except Exception as e:
        logger.warning(
            "Vertical text analysis failed on page %d: %s", page_number, e
        )

    return analysis


# ---------------------------------------------------------------------------
# Document-level finalization
# ---------------------------------------------------------------------------


def finalize_vertical_text(doc_vt: DocumentVerticalText) -> DocumentVerticalText:
    """Compute document-level summary from page-level vertical text results.

    Args:
        doc_vt: DocumentVerticalText with pages already populated.

    Returns:
        The same DocumentVerticalText with summary fields computed.
    """
    if not doc_vt.pages:
        return doc_vt

    vertical_pages = 0
    mixed_pages = 0

    for page_data in doc_vt.pages:
        if isinstance(page_data, VerticalTextAnalysis):
            direction = page_data.direction
        elif isinstance(page_data, dict):
            direction = page_data.get("direction", "horizontal")
        else:
            continue

        if direction == "vertical":
            vertical_pages += 1
        elif direction == "mixed":
            mixed_pages += 1

    doc_vt.total_vertical_pages = vertical_pages
    doc_vt.total_mixed_pages = mixed_pages
    doc_vt.has_vertical_content = (vertical_pages + mixed_pages) > 0

    return doc_vt


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_vertical_analysis_json(
    doc_vt: DocumentVerticalText,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write .vertical.json sidecar file.

    Output to EXPORT/VERTICAL/<subfolder>/<name>.vertical.json

    Args:
        doc_vt: Finalized DocumentVerticalText dataclass.
        output_folder: Root output directory (e.g. /app/ocr_output).
        subfolder: Relative subfolder path mirroring source structure.
        pipeline_version: Pipeline version string for metadata.

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        vertical_dir = os.path.join(output_folder, "EXPORT", "VERTICAL")

        if subfolder and subfolder != ".":
            safe_parts = [
                s for s in (
                    sanitize_path_segment(p)
                    for p in subfolder.replace("\\", "/").split("/")
                    if p
                )
                if s
            ]
            target_dir = os.path.join(vertical_dir, *safe_parts) if safe_parts else vertical_dir
        else:
            target_dir = vertical_dir

        # Path traversal protection
        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(vertical_dir)):
            logger.error("Path traversal blocked in vertical text output: %s", subfolder)
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc_vt.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.vertical.json")

        # Build pages output
        pages_output = []
        for page_data in doc_vt.pages:
            if isinstance(page_data, VerticalTextAnalysis):
                pages_output.append(asdict(page_data))
            elif isinstance(page_data, dict):
                pages_output.append(page_data)

        report = {
            "schema_version": "1.0",
            "document_id": doc_vt.document_id,
            "source_file": doc_vt.source_file,
            "processing": {
                "analysis_engine": "geometry_aspect_ratio",
                "aspect_ratio_threshold": VERTICAL_ASPECT_RATIO_THRESHOLD,
                "column_grouping_tolerance": COLUMN_GROUPING_TOLERANCE,
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            },
            "document_summary": {
                "total_vertical_pages": doc_vt.total_vertical_pages,
                "total_mixed_pages": doc_vt.total_mixed_pages,
                "has_vertical_content": doc_vt.has_vertical_content,
            },
            "pages": pages_output,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as e:
        logger.error("Failed to write vertical text JSON for %s: %s", doc_vt.document_id, e)
        return None
