"""Experimental signature verification heuristics for the OCR pipeline.

This module is intentionally conservative. It separates:

- signature presence detection: "is there ink in a signature-designated area?"
- authenticity review signals: "does this look suspicious enough for review?"

It never asserts that a signature is authentic. The strongest outcome is
"inconclusive" or "review_required".
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    sanitize_path_segment,
)

__all__ = [
    "SignatureCandidate",
    "PageSignatureVerification",
    "DocumentSignatureVerification",
    "analyze_signature_page",
    "finalize_signature_verification",
    "write_signature_verification_json",
]

logger = logging.getLogger(__name__)

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


_SIGNATURE_KEYWORDS = ("signature", "sign here", "signed by", "authorized signature")
_SIGNATURE_LINE_MARKERS = ("___", "---", "...", "===")
_PRESENCE_MIN_INK_RATIO = 0.008
_TYPED_TEXT_CONFIDENCE = 0.9
_SUSPICIOUS_TEXT_CONFIDENCE = 0.96
_MIN_STROKE_COMPLEXITY = 0.12
_WIDE_SIGNATURE_RATIO = 1.8


@dataclass
class SignatureCandidate:
    page_num: int
    source: str
    bbox: list = field(default_factory=list)
    label: str = ""
    ocr_confidence: float = 0.0
    presence_detected: bool = False
    presence_confidence: float = 0.0
    authenticity_signal: str = "not_applicable"
    review_required: bool = False
    reason_codes: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass
class PageSignatureVerification:
    page_num: int
    has_signature_candidate: bool = False
    presence_detected: bool = False
    review_required: bool = False
    authenticity_signal: str = "not_applicable"
    candidates: list = field(default_factory=list)


@dataclass
class DocumentSignatureVerification:
    document_id: str
    source_file: str
    pages: list = field(default_factory=list)
    total_candidate_pages: int = 0
    total_presence_pages: int = 0
    total_review_pages: int = 0
    experimental: bool = True


def _bbox_to_ints(bbox: list, image_shape: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    height, width = image_shape[:2]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _crop_with_padding(image: np.ndarray, bbox: list, pad: int = 8) -> Optional[np.ndarray]:
    resolved = _bbox_to_ints(bbox, image.shape)
    if resolved is None:
        return None
    x1, y1, x2, y2 = resolved
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.shape[1], x2 + pad)
    y2 = min(image.shape[0], y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _to_grayscale_array(image) -> Optional[np.ndarray]:
    try:
        if hasattr(image, "convert"):
            return np.array(image.convert("L"))
        arr = np.array(image)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if _CV2_AVAILABLE:
                return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            return np.mean(arr[:, :, :3], axis=2).astype(np.uint8)
    except Exception:
        logger.warning("Failed to normalize page image for signature verification", exc_info=True)
    return None


def _stroke_complexity(binary: np.ndarray) -> float:
    if binary.size == 0:
        return 0.0
    row_changes = np.abs(np.diff(binary.astype(np.int16), axis=1)).sum()
    col_changes = np.abs(np.diff(binary.astype(np.int16), axis=0)).sum()
    denom = max(binary.shape[0] * binary.shape[1], 1)
    return round(float((row_changes + col_changes) / denom), 4)


def _estimate_presence_metrics(crop: np.ndarray) -> dict:
    if crop.size == 0:
        return {
            "ink_ratio": 0.0,
            "stroke_complexity": 0.0,
            "dark_span_ratio": 0.0,
            "aspect_ratio": 0.0,
        }

    threshold = np.percentile(crop, 70)
    binary = crop < max(threshold, 180)
    ink_ratio = round(float(binary.mean()), 4)
    stroke_complexity = _stroke_complexity(binary)

    dark_cols = np.where(binary.any(axis=0))[0]
    dark_span_ratio = 0.0
    if dark_cols.size:
        dark_span_ratio = round(float((dark_cols[-1] - dark_cols[0] + 1) / max(binary.shape[1], 1)), 4)

    aspect_ratio = round(float(binary.shape[1] / max(binary.shape[0], 1)), 4)
    return {
        "ink_ratio": ink_ratio,
        "stroke_complexity": stroke_complexity,
        "dark_span_ratio": dark_span_ratio,
        "aspect_ratio": aspect_ratio,
    }


def _presence_confidence(metrics: dict) -> float:
    raw = (
        min(metrics["ink_ratio"] / max(_PRESENCE_MIN_INK_RATIO * 4, 0.0001), 1.0) * 0.45
        + min(metrics["stroke_complexity"] / max(_MIN_STROKE_COMPLEXITY * 2, 0.0001), 1.0) * 0.35
        + min(metrics["dark_span_ratio"] / 0.65, 1.0) * 0.20
    )
    return round(min(max(raw, 0.0), 1.0), 4)


def _typed_text_signal(label: str, ocr_confidence: float) -> bool:
    clean = (label or "").strip()
    if not clean:
        return False
    alpha_chars = sum(ch.isalpha() for ch in clean)
    if alpha_chars < 4:
        return False
    return ocr_confidence >= _SUSPICIOUS_TEXT_CONFIDENCE


def _candidate_from_form_field(page_num: int, field: dict) -> SignatureCandidate:
    return SignatureCandidate(
        page_num=page_num,
        source="form_field",
        bbox=list(field.get("bbox", []) or []),
        label=str(field.get("label", "") or ""),
        ocr_confidence=round(float(field.get("confidence", 0.0) or 0.0), 4),
    )


def _label_has_signature_line_markers(label: str) -> bool:
    text = str(label or "")
    if not text:
        return False
    return any(marker in text for marker in _SIGNATURE_LINE_MARKERS) or sum(
        text.count(char) for char in "_.-"
    ) >= 3


def _project_ocr_keyword_bbox(bbox: list, image_shape: tuple[int, int], label: str) -> list:
    resolved = _bbox_to_ints(list(bbox or []), image_shape)
    if resolved is None:
        return []

    x1, y1, x2, y2 = resolved
    page_height, page_width = image_shape[:2]
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    projected_y1 = max(0, y1 - max(int(round(height * 0.35)), 4))
    projected_y2 = min(page_height, y2 + max(int(round(height * 1.2)), 12))

    if _label_has_signature_line_markers(label):
        projected_x1 = min(
            page_width - 1,
            x1 + max(int(round(width * 0.45)), max(int(round(height * 1.5)), 12)),
        )
        projected_x2 = min(
            page_width,
            max(
                projected_x1 + max(int(round(width * 0.75)), 32),
                x2 + max(int(round(width * 0.5)), 24),
            ),
        )
    else:
        projected_x1 = min(page_width - 1, x2 + max(int(round(height * 0.25)), 2))
        projected_x2 = min(
            page_width,
            projected_x1 + max(int(round(width * 1.75)), max(int(round(height * 6)), 48)),
        )

    if projected_x2 - projected_x1 < max(int(round(height * 1.5)), 16):
        projected_x1 = max(0, x1)
        projected_x2 = min(page_width, x2)
        projected_y1 = min(page_height - 1, y2 + 2)
        projected_y2 = min(
            page_height,
            projected_y1 + max(int(round(height * 2.0)), 24),
        )

    if projected_x2 <= projected_x1 or projected_y2 <= projected_y1:
        return []

    return [projected_x1, projected_y1, projected_x2, projected_y2]


def _fallback_candidates_from_lines(
    page_num: int,
    paddle_lines: list,
    image_shape: tuple[int, int],
) -> list[SignatureCandidate]:
    candidates: list[SignatureCandidate] = []
    for text, box, conf in paddle_lines or []:
        text_lower = str(text).lower()
        if not any(keyword in text_lower for keyword in _SIGNATURE_KEYWORDS):
            continue
        xs = [pt[0] for pt in box] if box and isinstance(box[0], (list, tuple)) else []
        ys = [pt[1] for pt in box] if box and isinstance(box[0], (list, tuple)) else []
        bbox = [min(xs), min(ys), max(xs), max(ys)] if xs and ys else []
        projected_bbox = _project_ocr_keyword_bbox(bbox, image_shape, str(text))
        candidates.append(
            SignatureCandidate(
                page_num=page_num,
                source="ocr_keyword",
                bbox=projected_bbox,
                label=str(text),
                ocr_confidence=round(float(conf or 0.0), 4),
            )
        )
    return candidates


def analyze_signature_page(
    image,
    page_num: int,
    structure_data: Optional[dict],
    paddle_lines: Optional[list] = None,
) -> PageSignatureVerification:
    """Analyze a page for experimental signature presence/authenticity signals."""
    result = PageSignatureVerification(page_num=page_num)
    gray = _to_grayscale_array(image)
    if gray is None:
        return result

    candidates: list[SignatureCandidate] = []
    for form_field in (structure_data or {}).get("form_fields", []):
        if form_field.get("field_type") == "signature":
            candidates.append(_candidate_from_form_field(page_num, form_field))

    if not candidates:
        candidates.extend(_fallback_candidates_from_lines(page_num, paddle_lines or [], gray.shape))

    if not candidates:
        return result

    for candidate in candidates:
        crop = _crop_with_padding(gray, candidate.bbox)
        if crop is None:
            candidate.reason_codes.append("invalid_bbox")
            candidate.authenticity_signal = "review_required"
            candidate.review_required = True
            result.candidates.append(asdict(candidate))
            continue

        metrics = _estimate_presence_metrics(crop)
        candidate.metrics = metrics
        candidate.presence_confidence = _presence_confidence(metrics)
        candidate.presence_detected = bool(
            metrics["ink_ratio"] >= _PRESENCE_MIN_INK_RATIO
            and metrics["dark_span_ratio"] >= 0.15
        )

        if candidate.presence_detected:
            candidate.reason_codes.append("ink_present")
            if metrics["aspect_ratio"] >= _WIDE_SIGNATURE_RATIO:
                candidate.reason_codes.append("wide_signature_span")
            if metrics["stroke_complexity"] >= _MIN_STROKE_COMPLEXITY:
                candidate.reason_codes.append("stroke_complexity_present")
        else:
            candidate.reason_codes.append("no_signature_ink")

        if not candidate.presence_detected:
            candidate.authenticity_signal = "not_applicable"
        else:
            candidate.authenticity_signal = "inconclusive"
            if _typed_text_signal(candidate.label, candidate.ocr_confidence):
                candidate.reason_codes.append("typed_text_suspected")
                candidate.authenticity_signal = "review_required"
                candidate.review_required = True
            elif metrics["stroke_complexity"] < _MIN_STROKE_COMPLEXITY:
                candidate.reason_codes.append("low_stroke_complexity")
                candidate.authenticity_signal = "review_required"
                candidate.review_required = True

        result.candidates.append(asdict(candidate))

    result.has_signature_candidate = bool(result.candidates)
    result.presence_detected = any(c["presence_detected"] for c in result.candidates)
    result.review_required = any(c["review_required"] for c in result.candidates)
    if result.review_required:
        result.authenticity_signal = "review_required"
    elif result.presence_detected:
        result.authenticity_signal = "inconclusive"
    return result


def finalize_signature_verification(
    document: DocumentSignatureVerification,
) -> DocumentSignatureVerification:
    if not document.pages:
        return document

    document.total_candidate_pages = 0
    document.total_presence_pages = 0
    document.total_review_pages = 0

    for page in document.pages:
        if isinstance(page, PageSignatureVerification):
            page_data = asdict(page)
        else:
            page_data = page
        if page_data.get("has_signature_candidate"):
            document.total_candidate_pages += 1
        if page_data.get("presence_detected"):
            document.total_presence_pages += 1
        if page_data.get("review_required"):
            document.total_review_pages += 1
    return document


def write_signature_verification_json(
    document: DocumentSignatureVerification,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    try:
        base_dir = os.path.join(output_folder, "EXPORT", "SIGNATURE")
        if subfolder and subfolder != ".":
            safe_parts = [
                s for s in (
                    sanitize_path_segment(p)
                    for p in subfolder.replace("\\", "/").split("/")
                    if p
                )
                if s
            ]
            target_dir = os.path.join(base_dir, *safe_parts) if safe_parts else base_dir
        else:
            target_dir = base_dir

        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(base_dir)):
            logger.error("Path traversal blocked in signature output: %s", subfolder)
            return None

        os.makedirs(target_dir, exist_ok=True)
        base_name = build_sidecar_base_name(document.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.signature.json")

        pages = [
            asdict(page) if isinstance(page, PageSignatureVerification) else page
            for page in document.pages
        ]
        report = {
            "schema_version": "1.0",
            "document_id": document.document_id,
            "source_file": document.source_file,
            "processing": {
                "experimental": True,
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
                "notes": [
                    "Signature presence and authenticity review are separate signals.",
                    "This feature never certifies a signature as authentic.",
                    "review_required and inconclusive must be treated as analyst-review outcomes.",
                ],
            },
            "document_summary": {
                "total_candidate_pages": document.total_candidate_pages,
                "total_presence_pages": document.total_presence_pages,
                "total_review_pages": document.total_review_pages,
                "experimental": document.experimental,
            },
            "pages": pages,
        }

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        return json_path
    except Exception as exc:
        logger.error(
            "Failed to write signature verification JSON for %s: %s",
            document.document_id,
            exc,
        )
        return None
