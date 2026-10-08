"""Processing integrity validation for forensic OCR pipeline.

Generates per-document validation reports (.validation.json sidecar files)
with page-level OCR confidence, method tracking, and quality classification.

Output: EXPORT/VALIDATION/<subfolder>/<document_name>.validation.json
"""

import datetime
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    resolve_safe_output_subfolder,
    write_json_atomic,
)

__all__ = [
    "PageValidation",
    "DocumentValidation",
    "classify_quality",
    "compute_file_hash",
    "finalize_validation",
    "write_validation_json",
]

logger = logging.getLogger(__name__)


@dataclass
class PageValidation:
    """Per-page validation data collected during OCR."""

    page_num: int
    ocr_method: str = ""  # "PaddleOCR", "Tesseract", "ImageOnly", "EXTRACT_FAILED"
    ocr_language: str = ""
    ocr_confidence: float = 0.0  # Mean confidence (0.0-1.0)
    text_length: int = 0
    has_text: bool = False
    status: Literal["pending", "ok", "fallback", "image_only", "failed", "unknown"] = "pending"
    detected_language: str = ""
    model_language: str = ""
    needs_review: bool = False
    review_reasons: list = field(default_factory=list)
    geometry_issues: list = field(default_factory=list)
    candidate_reviews: list = field(default_factory=list)
    independent_coverage: dict = field(default_factory=dict)


@dataclass
class DocumentValidation:
    """Per-document validation report."""

    document_id: str
    source_file: str
    source_page_count: int = 0
    output_page_count: int = 0
    page_count_match: bool = True
    pages: list = field(default_factory=list)  # List[PageValidation as dict]

    # Quality summary (computed after all pages collected)
    classification: str = ""  # high_quality, acceptable, degraded, review_required
    overall_confidence: float = 0.0
    pages_with_text: int = 0
    pages_image_only: int = 0
    pages_failed: int = 0
    pages_needing_review: int = 0
    text_extraction_rate: float = 0.0
    total_text_length: int = 0
    ocr_methods_used: list = field(default_factory=list)
    output_hash: str = ""  # SHA-256 hash of the output PDF


def classify_quality(doc_val: DocumentValidation) -> str:
    """Classify document quality based on metrics.

    - high_quality: All pages have text, mean confidence >= 0.85
    - acceptable: text_extraction_rate >= 0.80, mean confidence >= 0.60
    - degraded: text_extraction_rate >= 0.50 or mean confidence >= 0.40
    - review_required: below all thresholds
    """
    rate = doc_val.text_extraction_rate
    conf = doc_val.overall_confidence

    # Mean model confidence is neither completeness nor accuracy proof.
    # Explicit page defects veto a positive document disposition.
    if (
        not doc_val.page_count_match
        or doc_val.pages_failed
        or doc_val.pages_image_only
        or doc_val.pages_needing_review
    ):
        return "review_required"

    if rate >= 1.0 and conf >= 0.85:
        return "high_quality"
    if rate >= 0.80 and conf >= 0.60:
        return "acceptable"
    if rate >= 0.50 or conf >= 0.40:
        return "degraded"
    return "review_required"


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file contents. Returns hex digest."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def finalize_validation(doc_val: DocumentValidation) -> DocumentValidation:
    """Compute summary statistics from page-level data."""
    if not doc_val.pages:
        doc_val.classification = "review_required"
        return doc_val

    total_conf = 0.0
    conf_count = 0
    methods = set()
    doc_val.pages_with_text = 0
    doc_val.pages_image_only = 0
    doc_val.pages_failed = 0
    doc_val.pages_needing_review = 0
    doc_val.total_text_length = 0

    for page_dict in doc_val.pages:
        # Accept both dict and PageValidation objects
        if isinstance(page_dict, PageValidation):
            page = page_dict
        else:
            page = PageValidation(**{
                k: v for k, v in page_dict.items()
                if k in PageValidation.__dataclass_fields__
            })

        if page.has_text:
            doc_val.pages_with_text += 1
        if page.ocr_method == "ImageOnly":
            doc_val.pages_image_only += 1
        if page.status == "failed":
            doc_val.pages_failed += 1
        if page.needs_review or page.status in {"unknown", "pending"}:
            doc_val.pages_needing_review += 1
        if 0.0 <= page.ocr_confidence <= 1.0:
            total_conf += page.ocr_confidence
            conf_count += 1
        doc_val.total_text_length += page.text_length
        if page.ocr_method:
            methods.add(page.ocr_method)

    doc_val.overall_confidence = (total_conf / conf_count) if conf_count > 0 else 0.0
    total_pages = len(doc_val.pages)
    doc_val.text_extraction_rate = (
        doc_val.pages_with_text / total_pages if total_pages > 0 else 0.0
    )
    doc_val.ocr_methods_used = sorted(methods)
    doc_val.page_count_match = doc_val.source_page_count == doc_val.output_page_count
    doc_val.classification = classify_quality(doc_val)

    return doc_val


def write_validation_json(
    doc_val: DocumentValidation,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write .validation.json sidecar file.

    Output to EXPORT/VALIDATION/<subfolder>/<name>.validation.json

    Args:
        doc_val: Finalized DocumentValidation dataclass.
        output_folder: Root output directory (e.g. /app/ocr_output).
        subfolder: Relative subfolder path mirroring source structure.
        pipeline_version: Pipeline version string for metadata.

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        validation_dir = os.path.join(output_folder, "EXPORT", "VALIDATION")
        target_dir = resolve_safe_output_subfolder(validation_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc_val.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.validation.json")

        report = {
            "schema_version": "1.0",
            "document_id": doc_val.document_id,
            "source_file": doc_val.source_file,
            "processing": {
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            },
            "page_count": {
                "source": doc_val.source_page_count,
                "output": doc_val.output_page_count,
                "match": doc_val.page_count_match,
            },
            "quality": {
                "classification": doc_val.classification,
                "overall_confidence": round(doc_val.overall_confidence, 4),
                "text_extraction_rate": round(doc_val.text_extraction_rate, 4),
                "total_text_length": doc_val.total_text_length,
                "pages_with_text": doc_val.pages_with_text,
                "pages_image_only": doc_val.pages_image_only,
                "pages_failed": doc_val.pages_failed,
                "pages_needing_review": doc_val.pages_needing_review,
                "ocr_methods_used": doc_val.ocr_methods_used,
            },
            "pages": doc_val.pages,
            "output_hash": doc_val.output_hash,
        }

        write_json_atomic(json_path, report, allowed_root=validation_dir)

        return json_path
    except Exception as e:
        logger.error("Failed to write validation JSON for %s: %s", doc_val.document_id, e)
        return None
