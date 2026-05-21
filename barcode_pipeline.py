"""QR/Barcode live decoder integration for the OCR pipeline.

Phase 10 — Persistent Intelligence Platform: provides a unified barcode
decoding interface that wraps pyzbar (primary) and python-zxing (fallback)
with graceful degradation when neither is installed.

This module builds on the existing ``barcode_extraction.py`` data model
(``DetectedBarcode``, ``PageBarcodes``) and adds multi-backend decoding
with a single pipeline-friendly entry point.

Configuration:
- ``ENABLE_BARCODE_DECODE``: env var to enable/disable (default: false)

Usage from GPU worker::

    from barcode_pipeline import BarcodePipeline

    pipeline = BarcodePipeline()
    if pipeline.is_available:
        results = pipeline.decode_page(pil_image, page_num=1)
        # results: list of dicts with barcode_type, data, bbox, confidence
"""

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_BARCODE_DECODE = os.environ.get(
    'ENABLE_BARCODE_DECODE', 'false'
).lower() in ('true', '1', 'yes')

# ---------------------------------------------------------------------------
# Backend imports (graceful degradation)
# ---------------------------------------------------------------------------

_PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar  # noqa: F401

    _PYZBAR_AVAILABLE = True
except ImportError:
    pass

_ZXING_AVAILABLE = False
_zxing_module = None
try:
    import zxing as _zxing_module

    _ZXING_AVAILABLE = True
except ImportError:
    pass

# Import data model from existing barcode_extraction module
try:
    from barcode_extraction import (
        BarcodeExtractor,
        DetectedBarcode,
        PageBarcodes,
        _normalize_barcode_type,
        _to_pil_image,
    )

    _EXTRACTOR_AVAILABLE = True
except ImportError:
    _EXTRACTOR_AVAILABLE = False
    BarcodeExtractor = None
    DetectedBarcode = None
    PageBarcodes = None
    _normalize_barcode_type = None
    _to_pil_image = None

# ---------------------------------------------------------------------------
# PIL import
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PILImage = None
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# ZXing decoder backend
# ---------------------------------------------------------------------------


class _ZxingDecoder:
    """Decode barcodes using python-zxing (java-based, requires JRE)."""

    def __init__(self):
        self._reader = None
        if _ZXING_AVAILABLE:
            try:
                self._reader = _zxing_module.BarCodeReader()
            except Exception as exc:
                logger.debug("ZXing reader initialization failed: %s", exc)

    @property
    def is_available(self):
        return self._reader is not None

    def decode(self, image, page_num=0):
        """Decode barcodes from a PIL Image.

        Returns list of dicts with barcode_type, data, bbox, confidence, page_num.
        """
        if not self.is_available or not _PIL_AVAILABLE:
            return []

        results = []
        try:
            # zxing requires a file path; write to temp
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                if isinstance(image, _PILImage.Image):
                    image.save(tmp_path)
                else:
                    return []

            try:
                decoded = self._reader.decode(tmp_path)
                if decoded and decoded.raw:
                    results.append({
                        'barcode_type': decoded.format or 'UNKNOWN',
                        'data': decoded.raw,
                        'bbox': [],  # zxing does not always provide bbox
                        'confidence': 1.0,
                        'page_num': page_num,
                    })
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as exc:
            logger.debug("ZXing decode failed on page %d: %s", page_num, exc)

        return results


# ---------------------------------------------------------------------------
# Unified Barcode Pipeline
# ---------------------------------------------------------------------------


class BarcodePipeline:
    """Unified barcode/QR decoding pipeline with multi-backend support.

    Tries pyzbar first (fastest, no JRE dependency), then falls back to
    python-zxing. Returns empty results when no backend is available.
    """

    def __init__(self):
        self._pyzbar_extractor = None
        self._zxing_decoder = None

        if _EXTRACTOR_AVAILABLE and _PYZBAR_AVAILABLE:
            self._pyzbar_extractor = BarcodeExtractor()

        if _ZXING_AVAILABLE:
            self._zxing_decoder = _ZxingDecoder()

        if not self.is_available:
            logger.info(
                "Barcode pipeline: no decoding backend available. "
                "Install pyzbar or python-zxing for barcode support."
            )

    @property
    def is_available(self):
        """Whether at least one decoding backend is functional."""
        pyzbar_ok = (
            self._pyzbar_extractor is not None
            and self._pyzbar_extractor.is_available
        )
        zxing_ok = (
            self._zxing_decoder is not None
            and self._zxing_decoder.is_available
        )
        return pyzbar_ok or zxing_ok

    @property
    def active_backend(self):
        """Name of the primary backend that will be used for decoding."""
        if (
            self._pyzbar_extractor is not None
            and self._pyzbar_extractor.is_available
        ):
            return 'pyzbar'
        if (
            self._zxing_decoder is not None
            and self._zxing_decoder.is_available
        ):
            return 'zxing'
        return 'none'

    def decode_page(self, image, page_num=0):
        """Decode all barcodes/QR codes from a page image.

        Args:
            image: PIL Image or numpy array.
            page_num: Page number for result attribution (1-based).

        Returns:
            List of dicts, each containing:
            - barcode_type: normalized type string
            - data: decoded data string
            - bbox: [x1, y1, x2, y2] bounding box (may be empty)
            - confidence: decode confidence (1.0 for exact match)
            - page_num: source page number
        """
        if not ENABLE_BARCODE_DECODE:
            return []

        if not self.is_available:
            return []

        # Try pyzbar first
        if (
            self._pyzbar_extractor is not None
            and self._pyzbar_extractor.is_available
        ):
            barcodes = self._pyzbar_extractor.extract(image, page_num)
            if barcodes:
                return [
                    {
                        'barcode_type': bc.barcode_type,
                        'data': bc.data,
                        'bbox': bc.bbox,
                        'confidence': bc.confidence,
                        'page_num': bc.page_num,
                    }
                    for bc in barcodes
                ]

        # Fallback to zxing
        if (
            self._zxing_decoder is not None
            and self._zxing_decoder.is_available
        ):
            # Convert to PIL for zxing
            if _PIL_AVAILABLE and _to_pil_image is not None:
                pil_img = _to_pil_image(image)
            elif _PIL_AVAILABLE and isinstance(image, _PILImage.Image):
                pil_img = image
            else:
                pil_img = None

            if pil_img is not None:
                return self._zxing_decoder.decode(pil_img, page_num)

        return []

    def decode_page_structured(self, image, page_num=0):
        """Decode barcodes and return a PageBarcodes-compatible dict.

        Returns a dict matching the PageBarcodes dataclass structure for
        compatibility with the existing symbology extraction output.
        """
        items = self.decode_page(image, page_num)
        types_found = sorted({item['barcode_type'] for item in items})
        return {
            'page_num': page_num,
            'barcodes': items,
            'total_barcodes': len(items),
            'barcode_types_found': types_found,
            'backend': self.active_backend,
        }
