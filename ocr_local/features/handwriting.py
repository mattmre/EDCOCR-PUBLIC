"""Handwriting detection for forensic OCR pipeline.

Detects handwritten vs printed text regions using confidence heuristics,
geometry analysis, and optional image variance from existing PaddleOCR output.

Output: EXPORT/HANDWRITING/<subfolder>/<document_name>.handwriting.json

Graceful degradation: if OpenCV (cv2) is not installed, image variance
detection is skipped while confidence and geometry analysis continue to work.
"""

import datetime
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    sanitize_path_segment,
)

__all__ = [
    "HANDWRITING_CONFIDENCE_THRESHOLD",
    "HANDWRITING_PAGE_THRESHOLD",
    "GEOMETRY_SPACING_CV_THRESHOLD",
    "GEOMETRY_HEIGHT_CV_THRESHOLD",
    "IMAGE_VARIANCE_THRESHOLD",
    "MIN_SAMPLES_FOR_VARIANCE",
    "HandwritingRegion",
    "PageHandwriting",
    "DocumentHandwriting",
    "detect_handwriting_by_confidence",
    "detect_handwriting_by_geometry",
    "detect_handwriting_by_image",
    "merge_handwriting_signals",
    "is_handwritten_page",
    "finalize_handwriting",
    "write_handwriting_json",
]

logger = logging.getLogger(__name__)

# --- Guarded OpenCV import ---
try:
    import cv2
    import numpy as np

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

HANDWRITING_CONFIDENCE_THRESHOLD = 0.65  # Below this = potentially handwritten
HANDWRITING_PAGE_THRESHOLD = 0.40  # Min fraction of lines to flag page
GEOMETRY_SPACING_CV_THRESHOLD = 0.45  # Coefficient of variation for irregular spacing
GEOMETRY_HEIGHT_CV_THRESHOLD = 0.35  # CV for variable character heights
IMAGE_VARIANCE_THRESHOLD = 0.30  # Stroke width variance threshold
MIN_SAMPLES_FOR_VARIANCE = 5  # Minimum contours/widths for meaningful stroke variance

# Weighted voting weights for signal merging
_WEIGHT_CONFIDENCE = 0.5
_WEIGHT_GEOMETRY = 0.3
_WEIGHT_IMAGE = 0.2
_MERGED_HANDWRITING_THRESHOLD = 0.5  # Decision threshold for weighted voting score


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HandwritingRegion:
    """A detected handwriting region on a page."""

    bbox: list = field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float = 0.0  # Detection confidence
    text: str = ""  # OCR text from this region
    ocr_confidence: float = 0.0  # Original OCR confidence
    detection_method: str = ""  # "confidence_heuristic", "geometry", "image_variance"


@dataclass
class PageHandwriting:
    """Handwriting analysis for a single page."""

    page_num: int
    has_handwriting: bool = False
    handwriting_coverage: float = 0.0  # 0.0-1.0 fraction of page area
    handwriting_regions: list = field(default_factory=list)  # List[HandwritingRegion as dict]
    printed_line_count: int = 0
    handwritten_line_count: int = 0
    detection_methods_used: list = field(default_factory=list)


@dataclass
class DocumentHandwriting:
    """Handwriting analysis for an entire document."""

    document_id: str
    source_file: str
    pages: list = field(default_factory=list)  # List[PageHandwriting as dict]
    total_handwritten_pages: int = 0
    total_pages_with_mixed: int = 0
    overall_handwriting_coverage: float = 0.0
    is_primarily_handwritten: bool = False


# ---------------------------------------------------------------------------
# Helper: coefficient of variation
# ---------------------------------------------------------------------------


def _coefficient_of_variation(values: list) -> float:
    """Compute coefficient of variation (std / mean). Returns 0.0 if invalid."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


# ---------------------------------------------------------------------------
# Detection: confidence heuristic
# ---------------------------------------------------------------------------


def detect_handwriting_by_confidence(
    paddle_lines: list,
    page_num: int,
    image_size: tuple = (1, 1),
) -> PageHandwriting:
    """Classify lines as handwritten or printed based on OCR confidence.

    Args:
        paddle_lines: List of (text, confidence, [x1, y1, x2, y2]) tuples.
        page_num: Page number (1-based).
        image_size: (width, height) of the page image for coverage calculation.

    Returns:
        PageHandwriting with regions for handwritten lines.
    """
    result = PageHandwriting(page_num=page_num)
    if not paddle_lines:
        return result

    try:
        page_area = max(image_size[0] * image_size[1], 1)
        hw_area = 0.0
        regions = []

        for text, confidence, bbox in paddle_lines:
            if confidence < HANDWRITING_CONFIDENCE_THRESHOLD:
                result.handwritten_line_count += 1
                region_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0)
                hw_area += region_area
                regions.append(asdict(HandwritingRegion(
                    bbox=list(bbox),
                    confidence=round(1.0 - confidence, 4),
                    text=str(text),
                    ocr_confidence=round(confidence, 4),
                    detection_method="confidence_heuristic",
                )))
            else:
                result.printed_line_count += 1

        total_lines = result.handwritten_line_count + result.printed_line_count
        hw_fraction = result.handwritten_line_count / total_lines if total_lines > 0 else 0.0

        result.has_handwriting = hw_fraction >= HANDWRITING_PAGE_THRESHOLD
        result.handwriting_coverage = round(hw_area / page_area, 4)
        result.handwriting_regions = regions
        if regions:
            result.detection_methods_used = ["confidence_heuristic"]
    except Exception as e:
        logger.warning("Confidence-based handwriting detection failed on page %d: %s", page_num, e)

    return result


# ---------------------------------------------------------------------------
# Detection: geometry analysis
# ---------------------------------------------------------------------------


def detect_handwriting_by_geometry(paddle_lines: list, page_num: int) -> dict:
    """Analyze bounding box geometry for handwriting indicators.

    Handwriting exhibits irregular inter-line spacing, variable character
    heights, and non-horizontal baselines compared to printed text.

    Args:
        paddle_lines: List of (text, confidence, [x1, y1, x2, y2]) tuples.
        page_num: Page number (1-based).

    Returns:
        Dict with is_handwritten, spacing_cv, height_cv, irregular_lines.
    """
    default = {"is_handwritten": False, "spacing_cv": 0.0, "height_cv": 0.0, "irregular_lines": 0}

    if len(paddle_lines) < 3:
        return default

    try:
        # Sort lines by vertical position (y1)
        sorted_lines = sorted(paddle_lines, key=lambda ln: ln[2][1])

        # Compute heights and inter-line spacings
        heights = []
        spacings = []
        for i, (_, _, bbox) in enumerate(sorted_lines):
            h = bbox[3] - bbox[1]
            if h > 0:
                heights.append(h)
            if i > 0:
                prev_bbox = sorted_lines[i - 1][2]
                spacing = bbox[1] - prev_bbox[3]
                spacings.append(spacing)

        spacing_cv = _coefficient_of_variation(spacings) if len(spacings) >= 2 else 0.0
        height_cv = _coefficient_of_variation(heights) if len(heights) >= 2 else 0.0

        is_handwritten = (
            spacing_cv >= GEOMETRY_SPACING_CV_THRESHOLD
            or height_cv >= GEOMETRY_HEIGHT_CV_THRESHOLD
        )

        return {
            "is_handwritten": is_handwritten,
            "spacing_cv": round(spacing_cv, 4),
            "height_cv": round(height_cv, 4),
        }
    except Exception as e:
        logger.warning("Geometry-based handwriting detection failed on page %d: %s", page_num, e)
        return default


# ---------------------------------------------------------------------------
# Detection: image variance (OpenCV)
# ---------------------------------------------------------------------------


def detect_handwriting_by_image(image, page_num: int) -> dict:
    """Estimate stroke width variance using edge detection and contour analysis.

    Handwriting produces higher variance in stroke widths compared to
    uniform printed text (fonts have consistent stroke widths).

    Args:
        image: PIL Image or numpy array.
        page_num: Page number (1-based).

    Returns:
        Dict with is_handwritten (bool) and stroke_variance (float).
    """
    default = {"is_handwritten": False, "stroke_variance": 0.0}

    if not _CV2_AVAILABLE:
        return default

    try:
        # Convert PIL Image to numpy array if needed
        if hasattr(image, "convert"):
            img_array = np.array(image.convert("L"))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                img_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img_array = image
        else:
            return default

        # Canny edge detection
        edges = cv2.Canny(img_array, 50, 150)

        # Find contours and measure bounding box widths as stroke proxy
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < MIN_SAMPLES_FOR_VARIANCE:
            return default

        # Use minimum bounding rect width as stroke width proxy
        widths = []
        for cnt in contours:
            if len(cnt) >= 5:
                _, (w, h), _ = cv2.minAreaRect(cnt)
                stroke_w = min(w, h)
                if stroke_w > 0:
                    widths.append(stroke_w)

        if len(widths) < MIN_SAMPLES_FOR_VARIANCE:
            return default

        stroke_cv = _coefficient_of_variation(widths)
        is_handwritten = stroke_cv >= IMAGE_VARIANCE_THRESHOLD

        return {
            "is_handwritten": is_handwritten,
            "stroke_variance": round(stroke_cv, 4),
        }
    except Exception as e:
        logger.warning("Image-based handwriting detection failed on page %d: %s", page_num, e)
        return default


# ---------------------------------------------------------------------------
# Signal merging
# ---------------------------------------------------------------------------


def merge_handwriting_signals(
    confidence_result: PageHandwriting,
    geometry_result: Optional[dict],
    image_result: Optional[dict],
    page_num: int,
) -> PageHandwriting:
    """Merge multiple detection signals using weighted voting.

    Weights: confidence=0.5, geometry=0.3, image=0.2.

    Args:
        confidence_result: PageHandwriting from confidence detection.
        geometry_result: Dict from geometry detection (or None).
        image_result: Dict from image detection (or None).
        page_num: Page number (1-based).

    Returns:
        Merged PageHandwriting with combined detection results.
    """
    merged = PageHandwriting(page_num=page_num)
    merged.handwriting_regions = list(confidence_result.handwriting_regions)
    merged.printed_line_count = confidence_result.printed_line_count
    merged.handwritten_line_count = confidence_result.handwritten_line_count
    merged.handwriting_coverage = confidence_result.handwriting_coverage
    merged.detection_methods_used = list(confidence_result.detection_methods_used)

    # Weighted score
    score = 0.0
    if confidence_result.has_handwriting:
        score += _WEIGHT_CONFIDENCE
    if geometry_result and geometry_result.get("is_handwritten", False):
        score += _WEIGHT_GEOMETRY
        if "geometry" not in merged.detection_methods_used:
            merged.detection_methods_used.append("geometry")
    if image_result and image_result.get("is_handwritten", False):
        score += _WEIGHT_IMAGE
        if "image_variance" not in merged.detection_methods_used:
            merged.detection_methods_used.append("image_variance")

    # Threshold: if weighted score exceeds the decision threshold, flag as handwritten
    merged.has_handwriting = score >= _MERGED_HANDWRITING_THRESHOLD

    return merged


# ---------------------------------------------------------------------------
# Convenience: page-level handwriting predicate
# ---------------------------------------------------------------------------


def is_handwritten_page(handwriting_result) -> bool:
    """Check whether a page's handwriting detection indicates handwriting.

    This is a convenience predicate for use by engine_selection and
    other routing logic that needs a simple boolean answer.

    Parameters
    ----------
    handwriting_result : PageHandwriting or dict or None
        The handwriting analysis result for a single page, as
        returned by :func:`merge_handwriting_signals` or
        :func:`detect_handwriting_by_confidence`.

    Returns
    -------
    bool
        ``True`` if the page has been flagged as containing handwriting.
    """
    if handwriting_result is None:
        return False
    if isinstance(handwriting_result, PageHandwriting):
        return handwriting_result.has_handwriting
    if isinstance(handwriting_result, dict):
        return bool(handwriting_result.get("has_handwriting", False))
    return False


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def finalize_handwriting(doc_hw: DocumentHandwriting) -> DocumentHandwriting:
    """Compute document-level summary from page-level handwriting results.

    Args:
        doc_hw: DocumentHandwriting with pages already populated.

    Returns:
        The same DocumentHandwriting with summary fields computed.
    """
    if not doc_hw.pages:
        return doc_hw

    hw_pages = 0
    mixed_pages = 0
    total_coverage = 0.0

    for page_data in doc_hw.pages:
        if isinstance(page_data, PageHandwriting):
            has_hw = page_data.has_handwriting
            coverage = page_data.handwriting_coverage
            printed = page_data.printed_line_count
            handwritten = page_data.handwritten_line_count
        elif isinstance(page_data, dict):
            has_hw = page_data.get("has_handwriting", False)
            coverage = page_data.get("handwriting_coverage", 0.0)
            printed = page_data.get("printed_line_count", 0)
            handwritten = page_data.get("handwritten_line_count", 0)
        else:
            continue

        if has_hw:
            hw_pages += 1
        if handwritten > 0 and printed > 0:
            mixed_pages += 1
        total_coverage += coverage

    total_pages = len(doc_hw.pages)
    doc_hw.total_handwritten_pages = hw_pages
    doc_hw.total_pages_with_mixed = mixed_pages
    doc_hw.overall_handwriting_coverage = round(total_coverage / total_pages, 4) if total_pages > 0 else 0.0
    doc_hw.is_primarily_handwritten = hw_pages > (total_pages / 2)

    return doc_hw


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_handwriting_json(
    doc_hw: DocumentHandwriting,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write .handwriting.json sidecar file.

    Output to EXPORT/HANDWRITING/<subfolder>/<name>.handwriting.json

    Args:
        doc_hw: Finalized DocumentHandwriting dataclass.
        output_folder: Root output directory (e.g. /app/ocr_output).
        subfolder: Relative subfolder path mirroring source structure.
        pipeline_version: Pipeline version string for metadata.

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        handwriting_dir = os.path.join(output_folder, "EXPORT", "HANDWRITING")

        if subfolder and subfolder != ".":
            safe_parts = [
                s for s in (
                    sanitize_path_segment(p)
                    for p in subfolder.replace("\\", "/").split("/")
                    if p
                )
                if s
            ]
            target_dir = os.path.join(handwriting_dir, *safe_parts) if safe_parts else handwriting_dir
        else:
            target_dir = handwriting_dir

        # Path traversal protection
        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(handwriting_dir)):
            logger.error("Path traversal blocked in handwriting output: %s", subfolder)
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc_hw.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.handwriting.json")

        # Build detection engine description
        engines = ["confidence_heuristic", "geometry"]
        if _CV2_AVAILABLE:
            engines.append("image_variance")

        # Build pages output
        pages_output = []
        for page_data in doc_hw.pages:
            if isinstance(page_data, PageHandwriting):
                pages_output.append(asdict(page_data))
            elif isinstance(page_data, dict):
                pages_output.append(page_data)

        report = {
            "schema_version": "1.0",
            "document_id": doc_hw.document_id,
            "source_file": doc_hw.source_file,
            "processing": {
                "detection_engine": "+".join(engines),
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            },
            "document_summary": {
                "total_handwritten_pages": doc_hw.total_handwritten_pages,
                "total_pages_with_mixed": doc_hw.total_pages_with_mixed,
                "overall_handwriting_coverage": doc_hw.overall_handwriting_coverage,
                "is_primarily_handwritten": doc_hw.is_primarily_handwritten,
            },
            "pages": pages_output,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as e:
        logger.error("Failed to write handwriting JSON for %s: %s", doc_hw.document_id, e)
        return None
