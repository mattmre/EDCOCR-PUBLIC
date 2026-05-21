"""Adaptive DPI escalation for low-confidence OCR pages.

When OCR confidence falls below a threshold, re-extracts the page image
at progressively higher DPI and re-runs OCR. Caps at MAX_ESCALATION_RETRIES
to prevent infinite loops.

DPI schedule matches reprocess/renderer.py: 300 -> 450 -> 600.

Usage (standalone):
    from dpi_escalation import should_escalate, get_next_dpi, re_extract_page_at_dpi

Usage (pipeline integration):
    Imported by ocr_gpu_async.py when ENABLE_DPI_ESCALATION is True.
    Low-confidence pages are re-queued on the image_queue at higher DPI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

try:
    from pdf2image import convert_from_path as _convert_from_path
except ImportError:
    _convert_from_path = None  # type: ignore[assignment]

try:
    from ocr_metrics import record_dpi_escalation as _record_dpi_escalation
except ImportError:
    _record_dpi_escalation = None  # type: ignore[assignment]

__all__ = [
    "DPI_SCHEDULE",
    "MAX_ESCALATION_RETRIES",
    "CONFIDENCE_THRESHOLD_RETRY",
    "POPPLER_TIMEOUT",
    "EscalationResult",
    "get_next_dpi",
    "should_escalate",
    "re_extract_page_at_dpi",
]

logger = logging.getLogger(__name__)

# DPI schedule matches reprocess/renderer.py
DPI_SCHEDULE = [300, 450, 600]
MAX_ESCALATION_RETRIES = 2
CONFIDENCE_THRESHOLD_RETRY = 0.60  # Below this triggers re-extraction
POPPLER_TIMEOUT = 300  # seconds; prevents hung threads on malformed PDFs


@dataclass
class EscalationResult:
    """Result of a DPI escalation attempt."""

    escalated: bool = False
    original_dpi: int = 300
    final_dpi: int = 300
    original_confidence: float = 0.0
    final_confidence: float = 0.0
    retries_used: int = 0


def get_next_dpi(current_dpi: int) -> int | None:
    """Get next DPI step from escalation schedule.

    Args:
        current_dpi: Current DPI value.

    Returns:
        Next DPI value, or None if already at maximum or value not in schedule.
    """
    try:
        idx = DPI_SCHEDULE.index(current_dpi)
        if idx < len(DPI_SCHEDULE) - 1:
            return DPI_SCHEDULE[idx + 1]
    except ValueError:
        pass
    return None


def should_escalate(
    confidence: float,
    retries: int,
    threshold: float = CONFIDENCE_THRESHOLD_RETRY,
) -> bool:
    """Check if a page should be re-extracted at higher DPI.

    Args:
        confidence: OCR confidence score (0.0 - 1.0).
        retries: Number of escalation retries already attempted.
        threshold: Confidence threshold below which escalation triggers.

    Returns:
        True if the page should be re-extracted at higher DPI.
    """
    if confidence >= threshold:
        return False
    if retries >= MAX_ESCALATION_RETRIES:
        return False
    return True


def re_extract_page_at_dpi(
    pdf_path: str,
    page_num: int,
    dpi: int,
    *,
    from_dpi: int | None = None,
) -> Image.Image | None:
    """Re-extract a single page from a PDF at the specified DPI.

    Uses pdf2image (Poppler) to render the page at higher resolution.
    Only works for PDF source files -- image sources cannot be re-extracted.

    When *from_dpi* is provided the Prometheus ``ocr_dpi_escalation_total``
    counter is incremented (requires ``ocr_metrics`` + ``prometheus_client``).

    Args:
        pdf_path: Path to the source PDF file.
        page_num: 1-based page number to extract.
        dpi: Target DPI for extraction.
        from_dpi: Original DPI before escalation (used for metrics).

    Returns:
        PIL Image (RGB) or None on failure.
    """
    if _convert_from_path is None:
        logger.warning("pdf2image not available; cannot re-extract at higher DPI")
        return None

    # Record DPI escalation metric
    if from_dpi is not None and _record_dpi_escalation is not None:
        _record_dpi_escalation(from_dpi, dpi)

    try:
        images = _convert_from_path(
            pdf_path, first_page=page_num, last_page=page_num, dpi=dpi,
            timeout=POPPLER_TIMEOUT,
        )
        if images:
            return images[0].convert("RGB")
    except Exception as e:
        logger.warning(
            "DPI escalation re-extraction failed for page %d at %d DPI: %s",
            page_num,
            dpi,
            e,
        )
    return None
