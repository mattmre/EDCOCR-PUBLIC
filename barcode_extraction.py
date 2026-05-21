"""Barcode and QR code extraction for forensic OCR pipeline.

Extracts barcodes (Code 128, Code 39, EAN-13, UPC-A, etc.) and QR codes from
page images using pyzbar with graceful degradation if pyzbar is not installed.

Output: EXPORT/SYMBOLOGY/<subfolder>/<document_name>.symbology.json

Graceful degradation: if pyzbar is not installed, barcode extraction is skipped
and an empty result set is returned. The OMR detection module operates
independently and does not require pyzbar.

Configuration (env vars):
- ``ENABLE_SYMBOLOGY_EXTRACTION``: master toggle (default: false)
- ``SYMBOLOGY_BARCODE_ENABLED``: enable barcode/QR extraction (default: true)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Guarded pyzbar import ---
try:
    from pyzbar import pyzbar as _pyzbar_module

    _PYZBAR_AVAILABLE = True
except ImportError:
    _pyzbar_module = None
    _PYZBAR_AVAILABLE = False

# --- Guarded PIL import ---
try:
    from PIL import Image as _PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PILImage = None
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DetectedBarcode:
    """A single detected barcode or QR code."""

    barcode_type: str = ""  # e.g. "QR_CODE", "CODE128", "EAN13", "CODE39"
    data: str = ""  # Decoded data string
    bbox: list = field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float = 1.0  # pyzbar always returns exact decode (1.0)
    page_num: int = 0
    raw_type: str = ""  # Raw pyzbar type string


@dataclass
class PageBarcodes:
    """Barcode extraction results for a single page."""

    page_num: int
    barcodes: list = field(default_factory=list)  # List[DetectedBarcode as dict]
    total_barcodes: int = 0
    barcode_types_found: list = field(default_factory=list)  # unique types


# ---------------------------------------------------------------------------
# Type normalization
# ---------------------------------------------------------------------------

# pyzbar returns type strings like "QRCODE", "CODE128", "EAN13", etc.
# Normalize to a consistent format.
_TYPE_MAP = {
    "QRCODE": "QR_CODE",
    "CODE128": "CODE128",
    "CODE39": "CODE39",
    "EAN13": "EAN13",
    "EAN8": "EAN8",
    "UPCA": "UPC_A",
    "UPCE": "UPC_E",
    "I25": "INTERLEAVED_2_OF_5",
    "DATABAR": "DATABAR",
    "DATABAR-EXP": "DATABAR_EXPANDED",
    "CODABAR": "CODABAR",
    "PDF417": "PDF417",
    "DATAMATRIX": "DATA_MATRIX",
    "AZTEC": "AZTEC",
}


def _normalize_barcode_type(raw_type: str) -> str:
    """Normalize a pyzbar barcode type string to a standard label."""
    return _TYPE_MAP.get(raw_type, raw_type)


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------


def _to_pil_image(image) -> Optional[object]:
    """Convert various image types to PIL Image for pyzbar.

    Args:
        image: PIL Image, numpy array, or similar.

    Returns:
        PIL Image or None if conversion fails.
    """
    if not _PIL_AVAILABLE:
        return None

    if isinstance(image, _PILImage.Image):
        return image

    if isinstance(image, np.ndarray):
        try:
            return _PILImage.fromarray(image)
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# Barcode extraction
# ---------------------------------------------------------------------------


class BarcodeExtractor:
    """Extracts barcodes and QR codes from page images using pyzbar.

    Graceful degradation: returns empty results when pyzbar is not installed.
    """

    def __init__(self):
        self._available = _PYZBAR_AVAILABLE
        if not self._available:
            logger.info(
                "pyzbar not available; barcode extraction will be skipped. "
                "Install pyzbar (pip install pyzbar) and libzbar0 for barcode support."
            )

    @property
    def is_available(self) -> bool:
        """Whether pyzbar is installed and functional."""
        return self._available

    def extract(self, image, page_num: int = 0) -> list:
        """Extract barcodes and QR codes from a single image.

        Args:
            image: PIL Image or numpy array.
            page_num: Page number for result attribution (1-based).

        Returns:
            List of DetectedBarcode dataclass instances.
        """
        if not self._available:
            return []

        pil_image = _to_pil_image(image)
        if pil_image is None:
            logger.debug(
                "Could not convert image to PIL format for barcode extraction "
                "(page %d)",
                page_num,
            )
            return []

        results = []
        try:
            decoded = _pyzbar_module.decode(pil_image)
            for symbol in decoded:
                # Extract bounding box as [x1, y1, x2, y2]
                rect = symbol.rect
                bbox = [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height]

                # Decode data (pyzbar returns bytes)
                try:
                    data_str = symbol.data.decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    data_str = str(symbol.data)

                raw_type = symbol.type
                barcode = DetectedBarcode(
                    barcode_type=_normalize_barcode_type(raw_type),
                    data=data_str,
                    bbox=bbox,
                    confidence=1.0,  # pyzbar is deterministic; decode = 100%
                    page_num=page_num,
                    raw_type=raw_type,
                )
                results.append(barcode)
        except Exception as exc:
            logger.warning(
                "Barcode extraction failed on page %d: %s", page_num, exc
            )

        return results

    def extract_page(self, image, page_num: int = 0) -> PageBarcodes:
        """Extract barcodes from a page image and return structured results.

        Args:
            image: PIL Image or numpy array.
            page_num: Page number for result attribution (1-based).

        Returns:
            PageBarcodes dataclass with all detected barcodes.
        """
        barcodes = self.extract(image, page_num)

        barcode_dicts = []
        types_found = set()
        for bc in barcodes:
            types_found.add(bc.barcode_type)
            barcode_dicts.append({
                "barcode_type": bc.barcode_type,
                "data": bc.data,
                "bbox": bc.bbox,
                "confidence": bc.confidence,
                "page_num": bc.page_num,
                "raw_type": bc.raw_type,
            })

        return PageBarcodes(
            page_num=page_num,
            barcodes=barcode_dicts,
            total_barcodes=len(barcode_dicts),
            barcode_types_found=sorted(types_found),
        )
