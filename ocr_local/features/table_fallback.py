"""Per-region table OCR fallback retry for forensic OCR pipeline (Phase 4D).

When table-region OCR produces low-confidence results, this module provides
logic to crop the table region from the page image, apply targeted
preprocessing, and build a retry strategy using an alternative engine.

The module defines retry **strategy and evaluation** only.  Actual OCR engine
invocation remains in the pipeline caller (ocr_gpu_async.py) to avoid
circular dependencies and GPU-resource ownership issues.

Output: EXPORT/TABLE_FALLBACK/<subfolder>/<document_name>.table_fallback.json

Configuration (env vars, all opt-in):
    ENABLE_TABLE_FALLBACK          = "false"  (master toggle)
    TABLE_FALLBACK_CONFIDENCE_THRESHOLD = "0.6"
    TABLE_FALLBACK_MAX_RETRIES     = "2"

Graceful degradation: if OpenCV (cv2) is not installed, preprocessing
strategies degrade to PIL-only contrast enhancement.  The module never
crashes the pipeline -- every public function wraps its logic in try/except
and returns safe defaults on failure.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter

from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    sanitize_path_segment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenCV availability guard
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants (env-configurable, opt-in)
# ---------------------------------------------------------------------------

ENABLE_TABLE_FALLBACK = (
    os.environ.get("ENABLE_TABLE_FALLBACK", "false").lower() == "true"
)
TABLE_FALLBACK_CONFIDENCE_THRESHOLD = float(
    os.environ.get("TABLE_FALLBACK_CONFIDENCE_THRESHOLD", "0.6")
)
TABLE_FALLBACK_MAX_RETRIES = int(
    os.environ.get("TABLE_FALLBACK_MAX_RETRIES", "2")
)

# Confidence gap below which two results are considered "close"
_CONFIDENCE_CLOSE_DELTA = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TableRegion:
    """A detected table region on a page."""

    bbox: tuple  # (x, y, w, h)
    page_number: int = 0
    original_confidence: float = 0.0
    original_text: str = ""
    original_engine: str = ""  # "paddle", "tesseract"


@dataclass
class TableFallbackResult:
    """Result of fallback retry for a single table region."""

    region: TableRegion = field(default_factory=lambda: TableRegion(bbox=(0, 0, 0, 0)))
    final_text: str = ""
    final_confidence: float = 0.0
    final_engine: str = ""
    attempts: list = field(default_factory=list)  # list of dicts
    improved: bool = False
    fallback_applied: bool = False


@dataclass
class PageTableFallbackAnalysis:
    """Fallback analysis for all table regions on a single page."""

    page_number: int = 0
    table_count: int = 0
    fallback_triggered: int = 0
    fallback_improved: int = 0
    results: list = field(default_factory=list)  # list of TableFallbackResult as dict


@dataclass
class DocumentTableFallbackSummary:
    """Document-level summary of table fallback results."""

    document_id: str = ""
    source_file: str = ""
    total_pages: int = 0
    total_tables: int = 0
    total_fallback_triggered: int = 0
    total_fallback_improved: int = 0
    pages: list = field(default_factory=list)  # list of PageTableFallbackAnalysis as dict


# ---------------------------------------------------------------------------
# Region cropping
# ---------------------------------------------------------------------------


def crop_table_region(
    page_image: Optional[Image.Image],
    bbox: tuple,
    padding: int = 10,
) -> Optional[Image.Image]:
    """Crop a table region from the page image with optional padding.

    Args:
        page_image: PIL Image of the full page.
        bbox: (x, y, w, h) bounding box of the table.
        padding: pixels of padding around the region (clamped to image bounds).

    Returns:
        PIL Image of the cropped table region, or None on failure.
    """
    if page_image is None:
        return None

    try:
        x, y, w, h = bbox
        img_w, img_h = page_image.size

        if w <= 0 or h <= 0:
            logger.debug("Table bbox has zero/negative size: %s", bbox)
            return None

        # Apply padding, clamped to image bounds
        left = max(0, x - padding)
        upper = max(0, y - padding)
        right = min(img_w, x + w + padding)
        lower = min(img_h, y + h + padding)

        if right <= left or lower <= upper:
            logger.debug("Clamped bbox collapsed to zero area: %s", bbox)
            return None

        return page_image.crop((left, upper, right, lower))
    except Exception as exc:
        logger.warning("Failed to crop table region %s: %s", bbox, exc)
        return None


# ---------------------------------------------------------------------------
# Confidence assessment
# ---------------------------------------------------------------------------


def assess_table_confidence(
    ocr_lines: list,
    bbox: tuple,
) -> float:
    """Assess OCR confidence for text within a table bounding box.

    Args:
        ocr_lines: list of (box_coords, text, confidence) from OCR output.
            box_coords can be [x1, y1, x2, y2] or [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        bbox: (x, y, w, h) of the table region.

    Returns:
        Average confidence for text lines whose centers fall within the
        table region, or 0.0 if no lines match.
    """
    if not ocr_lines or not bbox:
        return 0.0

    try:
        tx, ty, tw, th = bbox
        region_right = tx + tw
        region_bottom = ty + th

        confidences = []
        for line in ocr_lines:
            if len(line) < 3:
                continue
            _text, line_box, confidence = line[0], line[1], line[2]

            # Compute center of the line bounding box
            if isinstance(line_box, (list, tuple)) and len(line_box) == 4:
                if isinstance(line_box[0], (list, tuple)):
                    # Polygon: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    xs = [p[0] for p in line_box]
                    ys = [p[1] for p in line_box]
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                else:
                    # Flat: [x1, y1, x2, y2]
                    cx = (line_box[0] + line_box[2]) / 2
                    cy = (line_box[1] + line_box[3]) / 2
            else:
                continue

            # Check if center falls within table region
            if tx <= cx <= region_right and ty <= cy <= region_bottom:
                confidences.append(float(confidence))

        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
    except Exception as exc:
        logger.warning("Failed to assess table confidence for bbox %s: %s", bbox, exc)
        return 0.0


# ---------------------------------------------------------------------------
# Preprocessing strategies
# ---------------------------------------------------------------------------


def preprocess_table_region(
    image: Optional[Image.Image],
    strategy: str = "enhance",
) -> Optional[Image.Image]:
    """Apply preprocessing to a table region for retry.

    Strategies:
    - "enhance": contrast enhancement + sharpening (PIL-only)
    - "binarize": adaptive thresholding (requires OpenCV, degrades to PIL)
    - "denoise": denoising + binarization (requires OpenCV, degrades to PIL)

    Args:
        image: PIL Image of the cropped table region.
        strategy: one of "enhance", "binarize", "denoise".

    Returns:
        Preprocessed PIL Image, or original on failure.
    """
    if image is None:
        return None

    try:
        if strategy == "enhance":
            return _preprocess_enhance(image)
        if strategy == "binarize":
            return _preprocess_binarize(image)
        if strategy == "denoise":
            return _preprocess_denoise(image)

        logger.warning("Unknown preprocessing strategy %r, using 'enhance'", strategy)
        return _preprocess_enhance(image)
    except Exception as exc:
        logger.warning("Preprocessing strategy %r failed: %s", strategy, exc)
        return image


def _preprocess_enhance(image: Image.Image) -> Image.Image:
    """Contrast enhancement + sharpening using PIL (no OpenCV needed)."""
    enhanced = ImageEnhance.Contrast(image).enhance(1.5)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    return sharpened


def _preprocess_binarize(image: Image.Image) -> Image.Image:
    """Adaptive thresholding using OpenCV, with PIL fallback."""
    if not _CV2_AVAILABLE:
        logger.debug("OpenCV unavailable for binarize, falling back to enhance")
        return _preprocess_enhance(image)

    gray = np.array(image.convert("L"))
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    return Image.fromarray(binary, mode="L")


def _preprocess_denoise(image: Image.Image) -> Image.Image:
    """Denoising + binarization using OpenCV, with PIL fallback."""
    if not _CV2_AVAILABLE:
        logger.debug("OpenCV unavailable for denoise, falling back to enhance")
        return _preprocess_enhance(image)

    gray = np.array(image.convert("L"))
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    return Image.fromarray(binary, mode="L")


# ---------------------------------------------------------------------------
# Retry strategy
# ---------------------------------------------------------------------------


def build_retry_strategy(
    original_engine: str,
    original_confidence: float,
) -> list:
    """Determine the retry strategy based on what was tried.

    Returns a list of retry step dicts, each with "engine" and "preprocessing"
    keys.  The strategy alternates engines and escalates preprocessing
    aggressiveness with each retry.

    Args:
        original_engine: engine used on the first attempt ("paddle" or "tesseract").
        original_confidence: confidence from the first attempt.

    Returns:
        List of dicts, e.g.:
        [
            {"engine": "tesseract", "preprocessing": "enhance"},
            {"engine": "paddle", "preprocessing": "binarize"},
        ]
    """
    steps = []
    alt_engine = "tesseract" if original_engine == "paddle" else "paddle"

    # Step 1: try the alternate engine with light preprocessing
    steps.append({"engine": alt_engine, "preprocessing": "enhance"})

    # Step 2: try the original engine with heavier preprocessing
    preprocessing = "binarize" if original_confidence < 0.4 else "denoise"
    steps.append({"engine": original_engine, "preprocessing": preprocessing})

    # Trim to max retries
    return steps[:TABLE_FALLBACK_MAX_RETRIES]


# ---------------------------------------------------------------------------
# Fallback trigger check
# ---------------------------------------------------------------------------


def should_trigger_fallback(
    confidence: float,
    threshold: Optional[float] = None,
) -> bool:
    """Check if a table region should trigger fallback retry.

    Args:
        confidence: OCR confidence for the table region (0.0-1.0).
        threshold: custom threshold (defaults to TABLE_FALLBACK_CONFIDENCE_THRESHOLD).

    Returns:
        True if the confidence is below the threshold.
    """
    effective_threshold = (
        threshold if threshold is not None else TABLE_FALLBACK_CONFIDENCE_THRESHOLD
    )
    return confidence < effective_threshold


# ---------------------------------------------------------------------------
# Result evaluation
# ---------------------------------------------------------------------------


def evaluate_fallback_result(
    original_confidence: float,
    original_text: str,
    new_confidence: float,
    new_text: str,
) -> dict:
    """Compare original and fallback results to pick the better one.

    Uses confidence as primary metric.  If confidences are within
    _CONFIDENCE_CLOSE_DELTA of each other, prefers the result with more
    text content (longer string after stripping whitespace).

    Args:
        original_confidence: confidence from original OCR.
        original_text: text from original OCR.
        new_confidence: confidence from fallback OCR.
        new_text: text from fallback OCR.

    Returns:
        Dict with:
            "use_fallback": bool,
            "reason": str,
            "confidence_delta": float,
    """
    delta = new_confidence - original_confidence
    orig_len = len((original_text or "").strip())
    new_len = len((new_text or "").strip())

    # If confidences are very close, prefer more text
    if abs(delta) <= _CONFIDENCE_CLOSE_DELTA:
        if new_len > orig_len:
            return {
                "use_fallback": True,
                "reason": "similar confidence, more text",
                "confidence_delta": round(delta, 4),
            }
        return {
            "use_fallback": False,
            "reason": "similar confidence, original has enough text",
            "confidence_delta": round(delta, 4),
        }

    if new_confidence > original_confidence:
        return {
            "use_fallback": True,
            "reason": "higher confidence",
            "confidence_delta": round(delta, 4),
        }

    return {
        "use_fallback": False,
        "reason": "fallback did not improve confidence",
        "confidence_delta": round(delta, 4),
    }


# ---------------------------------------------------------------------------
# Page-level analysis
# ---------------------------------------------------------------------------


def analyze_page_tables(
    page_number: int,
    table_regions: list,
    ocr_lines: list,
) -> PageTableFallbackAnalysis:
    """Analyze all table regions on a page for potential fallback.

    This function checks each table region's confidence against the
    threshold and builds the retry strategy where needed.  It does NOT
    execute the retries (that is the pipeline caller's job).

    Args:
        page_number: 1-based page number.
        table_regions: list of TableRegion instances.
        ocr_lines: list of (box, text, confidence) from the page OCR.

    Returns:
        PageTableFallbackAnalysis with per-region results.
    """
    analysis = PageTableFallbackAnalysis(page_number=page_number)

    if not table_regions:
        return analysis

    analysis.table_count = len(table_regions)

    for region in table_regions:
        try:
            confidence = assess_table_confidence(ocr_lines, region.bbox)
            if confidence == 0.0:
                confidence = region.original_confidence

            trigger = should_trigger_fallback(confidence)

            result = TableFallbackResult(
                region=region,
                final_text=region.original_text,
                final_confidence=confidence,
                final_engine=region.original_engine,
                fallback_applied=False,
                improved=False,
            )

            if trigger:
                analysis.fallback_triggered += 1
                result.fallback_applied = True
                result.attempts = build_retry_strategy(
                    region.original_engine, confidence
                )

            analysis.results.append(asdict(result))
        except Exception as exc:
            logger.warning(
                "Error analyzing table region on page %d: %s", page_number, exc
            )

    return analysis


# ---------------------------------------------------------------------------
# Document-level finalization
# ---------------------------------------------------------------------------


def finalize_table_fallback(
    page_analyses: list,
    document_id: str = "",
    source_file: str = "",
) -> DocumentTableFallbackSummary:
    """Create document-level summary of table fallback results.

    Args:
        page_analyses: list of PageTableFallbackAnalysis instances or dicts.
        document_id: unique document identifier.
        source_file: source file path.

    Returns:
        DocumentTableFallbackSummary with aggregated counts.
    """
    summary = DocumentTableFallbackSummary(
        document_id=document_id,
        source_file=source_file,
    )

    if not page_analyses:
        return summary

    for page_data in page_analyses:
        if isinstance(page_data, PageTableFallbackAnalysis):
            summary.total_tables += page_data.table_count
            summary.total_fallback_triggered += page_data.fallback_triggered
            summary.total_fallback_improved += page_data.fallback_improved
            summary.pages.append(asdict(page_data))
        elif isinstance(page_data, dict):
            summary.total_tables += page_data.get("table_count", 0)
            summary.total_fallback_triggered += page_data.get("fallback_triggered", 0)
            summary.total_fallback_improved += page_data.get("fallback_improved", 0)
            summary.pages.append(page_data)

    summary.total_pages = len(page_analyses)
    return summary


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_table_fallback_json(
    summary: DocumentTableFallbackSummary,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write table fallback analysis to JSON sidecar.

    Output to EXPORT/TABLE_FALLBACK/<subfolder>/<name>.table_fallback.json

    Args:
        summary: finalized DocumentTableFallbackSummary.
        output_folder: root output directory (e.g. /app/ocr_output).
        subfolder: relative subfolder path mirroring source structure.
        pipeline_version: pipeline version string for metadata.

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        fallback_dir = os.path.join(output_folder, "EXPORT", "TABLE_FALLBACK")

        if subfolder and subfolder != ".":
            safe_parts = [
                s
                for s in (
                    sanitize_path_segment(p)
                    for p in subfolder.replace("\\", "/").split("/")
                    if p
                )
                if s
            ]
            target_dir = (
                os.path.join(fallback_dir, *safe_parts) if safe_parts else fallback_dir
            )
        else:
            target_dir = fallback_dir

        # Path traversal protection
        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(fallback_dir)):
            logger.error(
                "Path traversal blocked in table fallback output: %s", subfolder
            )
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(summary.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.table_fallback.json")

        report = {
            "schema_version": "1.0",
            "document_id": summary.document_id,
            "source_file": summary.source_file,
            "processing": {
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="milliseconds"),
                "table_fallback_enabled": True,
                "confidence_threshold": TABLE_FALLBACK_CONFIDENCE_THRESHOLD,
                "max_retries": TABLE_FALLBACK_MAX_RETRIES,
            },
            "document_summary": {
                "total_pages": summary.total_pages,
                "total_tables": summary.total_tables,
                "total_fallback_triggered": summary.total_fallback_triggered,
                "total_fallback_improved": summary.total_fallback_improved,
            },
            "pages": summary.pages,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as exc:
        logger.error(
            "Failed to write table fallback JSON for %s: %s",
            summary.document_id,
            exc,
        )
        return None
