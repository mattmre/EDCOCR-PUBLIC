"""Symbology extraction orchestrator for forensic OCR pipeline.

Coordinates barcode/QR extraction (via pyzbar) and OMR checkbox detection
(via OpenCV contour analysis) across document pages. Both sub-modules are
CPU-only and do not require GPU resources.

Output: EXPORT/SYMBOLOGY/<subfolder>/<document_name>.symbology.json

Configuration (env vars):
- ``ENABLE_SYMBOLOGY_EXTRACTION``: master toggle (default: false, opt-in)
- ``SYMBOLOGY_BARCODE_ENABLED``: enable barcode/QR extraction (default: true)
- ``SYMBOLOGY_OMR_ENABLED``: enable OMR checkbox detection (default: true)
- ``OMR_MIN_MARK_SIZE``: minimum mark size in pixels (default: 15)
- ``OMR_MAX_MARK_SIZE``: maximum mark size in pixels (default: 50)

Graceful degradation:
- pyzbar not installed -> barcode extraction skipped, OMR still works
- OpenCV not installed -> OMR detection skipped, barcode extraction still works
- Neither installed -> empty results returned, no errors
"""

import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from barcode_extraction import BarcodeExtractor
from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    sanitize_path_segment,
)
from omr_detection import OMRDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_SYMBOLOGY_EXTRACTION = os.environ.get(
    "ENABLE_SYMBOLOGY_EXTRACTION", ""
).lower() in ("1", "true", "yes")

SYMBOLOGY_BARCODE_ENABLED = os.environ.get(
    "SYMBOLOGY_BARCODE_ENABLED", "true"
).lower() in ("1", "true", "yes")

SYMBOLOGY_OMR_ENABLED = os.environ.get(
    "SYMBOLOGY_OMR_ENABLED", "true"
).lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PageSymbology:
    """Symbology results for a single page."""

    page_num: int
    barcodes: list = field(default_factory=list)  # List[barcode dicts]
    marks: list = field(default_factory=list)  # List[mark dicts]
    total_barcodes: int = 0
    total_marks: int = 0
    checked_marks: int = 0


@dataclass
class DocumentSymbology:
    """Symbology results for an entire document."""

    document_id: str
    source_file: str
    pages: list = field(default_factory=list)  # List[PageSymbology as dict]
    total_barcodes: int = 0
    total_marks: int = 0
    checked_marks: int = 0
    unchecked_marks: int = 0
    barcode_types_found: list = field(default_factory=list)  # unique types


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SymbologyExtractor:
    """Orchestrates barcode and OMR extraction across document pages."""

    def __init__(self):
        self._barcode_extractor = BarcodeExtractor()
        self._omr_detector = OMRDetector()

    @property
    def barcode_available(self) -> bool:
        """Whether barcode extraction is available."""
        return self._barcode_extractor.is_available

    @property
    def omr_available(self) -> bool:
        """Whether OMR detection is available."""
        return self._omr_detector.is_available

    def extract_page(
        self,
        image,
        page_num: int = 0,
        barcode_enabled: bool = True,
        omr_enabled: bool = True,
    ) -> PageSymbology:
        """Run barcode and OMR extraction on a single page image.

        Args:
            image: PIL Image or numpy array.
            page_num: Page number (1-based).
            barcode_enabled: Whether to run barcode extraction.
            omr_enabled: Whether to run OMR detection.

        Returns:
            PageSymbology with combined results.
        """
        barcodes = []
        marks = []

        if barcode_enabled and SYMBOLOGY_BARCODE_ENABLED:
            page_barcodes = self._barcode_extractor.extract_page(image, page_num)
            barcodes = page_barcodes.barcodes

        if omr_enabled and SYMBOLOGY_OMR_ENABLED:
            page_marks = self._omr_detector.detect_page(image, page_num)
            marks = page_marks.marks

        checked = sum(1 for m in marks if m.get("checked", False))

        return PageSymbology(
            page_num=page_num,
            barcodes=barcodes,
            marks=marks,
            total_barcodes=len(barcodes),
            total_marks=len(marks),
            checked_marks=checked,
        )

    def extract_document(
        self,
        pages: list,
        document_id: str,
        source_file: str,
    ) -> DocumentSymbology:
        """Run extraction across all pages of a document.

        Args:
            pages: List of dicts with 'image' (PIL/np) and 'page_num' (int).
            document_id: Unique document identifier.
            source_file: Source file path.

        Returns:
            DocumentSymbology with aggregated results.
        """
        doc = DocumentSymbology(
            document_id=document_id,
            source_file=source_file,
        )

        for page_info in pages:
            image = page_info.get("image")
            page_num = page_info.get("page_num", 0)

            if image is None:
                continue

            page_result = self.extract_page(image, page_num)
            doc.pages.append({
                "page_num": page_result.page_num,
                "barcodes": page_result.barcodes,
                "marks": page_result.marks,
                "total_barcodes": page_result.total_barcodes,
                "total_marks": page_result.total_marks,
                "checked_marks": page_result.checked_marks,
            })

        return finalize_symbology(doc)


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def finalize_symbology(doc: DocumentSymbology) -> DocumentSymbology:
    """Compute document-level summary from page-level symbology results.

    Args:
        doc: DocumentSymbology with pages already populated.

    Returns:
        The same DocumentSymbology with summary fields computed.
    """
    total_barcodes = 0
    total_marks = 0
    checked = 0
    barcode_types = set()

    for page_data in doc.pages:
        if isinstance(page_data, PageSymbology):
            total_barcodes += page_data.total_barcodes
            total_marks += page_data.total_marks
            checked += page_data.checked_marks
            for bc in page_data.barcodes:
                if isinstance(bc, dict):
                    barcode_types.add(bc.get("barcode_type", ""))
        elif isinstance(page_data, dict):
            total_barcodes += page_data.get("total_barcodes", 0)
            total_marks += page_data.get("total_marks", 0)
            checked += page_data.get("checked_marks", 0)
            for bc in page_data.get("barcodes", []):
                if isinstance(bc, dict):
                    barcode_types.add(bc.get("barcode_type", ""))

    doc.total_barcodes = total_barcodes
    doc.total_marks = total_marks
    doc.checked_marks = checked
    doc.unchecked_marks = total_marks - checked
    doc.barcode_types_found = sorted(barcode_types - {""})

    return doc


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_symbology_json(
    doc: DocumentSymbology,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write .symbology.json sidecar file.

    Output to EXPORT/SYMBOLOGY/<subfolder>/<name>.symbology.json

    Args:
        doc: Finalized DocumentSymbology dataclass.
        output_folder: Root output directory (e.g. /app/ocr_output).
        subfolder: Relative subfolder path mirroring source structure.
        pipeline_version: Pipeline version string for metadata.

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        symbology_dir = os.path.join(output_folder, "EXPORT", "SYMBOLOGY")

        if subfolder and subfolder != ".":
            safe_parts = [
                s for s in (
                    sanitize_path_segment(p)
                    for p in subfolder.replace("\\", "/").split("/")
                    if p
                )
                if s
            ]
            target_dir = (
                os.path.join(symbology_dir, *safe_parts)
                if safe_parts
                else symbology_dir
            )
        else:
            target_dir = symbology_dir

        # Path traversal protection
        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(symbology_dir)):
            logger.error(
                "Path traversal blocked in symbology output: %s", subfolder
            )
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.symbology.json")

        # Build engines description
        engines = []
        extractor = SymbologyExtractor()
        if SYMBOLOGY_BARCODE_ENABLED and extractor.barcode_available:
            engines.append("pyzbar")
        if SYMBOLOGY_OMR_ENABLED and extractor.omr_available:
            engines.append("opencv_omr")

        # Build pages output
        pages_output = []
        for page_data in doc.pages:
            if isinstance(page_data, PageSymbology):
                pages_output.append({
                    "page_num": page_data.page_num,
                    "barcodes": page_data.barcodes,
                    "marks": page_data.marks,
                    "total_barcodes": page_data.total_barcodes,
                    "total_marks": page_data.total_marks,
                    "checked_marks": page_data.checked_marks,
                })
            elif isinstance(page_data, dict):
                pages_output.append(page_data)

        report = {
            "schema_version": "1.0",
            "document_id": doc.document_id,
            "source_file": doc.source_file,
            "processing": {
                "detection_engines": "+".join(engines) if engines else "none",
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            },
            "summary": {
                "total_barcodes": doc.total_barcodes,
                "total_marks": doc.total_marks,
                "checked_marks": doc.checked_marks,
                "unchecked_marks": doc.unchecked_marks,
                "barcode_types_found": doc.barcode_types_found,
            },
            "pages": pages_output,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as exc:
        logger.error(
            "Failed to write symbology JSON for %s: %s", doc.document_id, exc
        )
        return None
