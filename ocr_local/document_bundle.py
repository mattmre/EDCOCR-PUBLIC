"""OCR-side DocumentBundle v1 exporter.

This module intentionally does not import ``ocr_local.translation``.  It
builds the versioned OCR-to-translation contract from OCR-shaped spans so
the OCR side can be tested before translation code is physically split out.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ocr_local.contracts import sha256_hex, validate_contract_payload

DOCUMENT_BUNDLE_SCHEMA_VERSION = "document-bundle-v1"


def build_document_bundle(
    *,
    document_id: str,
    source_file_name: str,
    spans: Iterable[Mapping[str, Any]],
    source_file_bytes: bytes | None = None,
    source_file_sha256: str | None = None,
    source_ocr_sha256: str | None = None,
    language_metadata: Mapping[str, Any] | None = None,
    ocr_engine_metadata: Mapping[str, Any] | None = None,
    custody_chain_head: str = "n/a",
    artifact_manifest: Mapping[str, Any] | None = None,
    privilege_flags: Mapping[str, bool] | None = None,
    tenant_policy_hash: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Build a schema-valid ``DocumentBundle v1`` from OCR-like spans."""

    normalized_spans = _normalize_spans(spans)
    if not normalized_spans:
        raise ValueError("DocumentBundle requires at least one OCR span")

    file_hash = source_file_sha256
    if file_hash is None:
        file_hash = sha256_hex(source_file_bytes or source_file_name.encode("utf-8"))

    ocr_hash = source_ocr_sha256 or _compute_ocr_text_sha256(normalized_spans)
    lang_meta = dict(language_metadata or _default_language_metadata(normalized_spans))
    engine_meta = dict(
        ocr_engine_metadata
        or {
            "engine_id": "ocr_local.fixture",
            "engine_version": "unknown",
        }
    )

    bundle: dict[str, Any] = {
        "schema_version": DOCUMENT_BUNDLE_SCHEMA_VERSION,
        "document_id": document_id,
        "source_file_name": source_file_name,
        "source_file_sha256": file_hash,
        "source_ocr_sha256": ocr_hash,
        "pages": _pages_from_spans(normalized_spans),
        "spans": normalized_spans,
        "language_metadata": lang_meta,
        "ocr_engine_metadata": engine_meta,
        "custody_chain_head": custody_chain_head,
        "artifact_manifest": dict(
            artifact_manifest
            or _default_artifact_manifest(source_file_name, file_hash, ocr_hash)
        ),
    }

    if privilege_flags is not None:
        bundle["privilege_flags"] = dict(privilege_flags)
    if tenant_policy_hash is not None:
        bundle["tenant_policy_hash"] = tenant_policy_hash

    if validate:
        validate_document_bundle(bundle)
    return bundle


def validate_document_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``DocumentBundle v1`` payload."""

    return validate_contract_payload(bundle, DOCUMENT_BUNDLE_SCHEMA_VERSION)


def write_document_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    """Validate and write a ``DocumentBundle v1`` JSON file."""

    validate_document_bundle(bundle)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _normalize_spans(spans: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        page_number = int(span.get("page_number", span.get("page_num", 1)))
        bbox = _normalize_bbox(span.get("bbox", span.get("source_bbox")))
        raw_bboxes = span.get("bboxes", span.get("source_bboxes"))
        bboxes = [_normalize_bbox(item) for item in raw_bboxes] if raw_bboxes else [bbox]

        item: dict[str, Any] = {
            "span_id": str(span.get("span_id", f"span-{index + 1}")),
            "page_number": page_number,
            "text": str(span.get("text", span.get("source_text", ""))),
            "bbox": bbox,
            "bboxes": bboxes,
        }
        if "language" in span:
            item["language"] = str(span["language"])
        if "confidence" in span:
            item["confidence"] = float(span["confidence"])
        if "metadata" in span and isinstance(span["metadata"], Mapping):
            item["metadata"] = dict(span["metadata"])
        normalized.append(item)
    return normalized


def _normalize_bbox(value: Any) -> list[float]:
    if value is None:
        raise ValueError("OCR span is missing a bbox")
    bbox = [float(part) for part in value]
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain exactly 4 numbers, got {len(bbox)}")
    return bbox


def _compute_ocr_text_sha256(spans: list[dict[str, Any]]) -> str:
    ordered = sorted(spans, key=lambda item: (item["page_number"], item["span_id"]))
    text_layer = "\n".join(item["text"] for item in ordered)
    return sha256_hex(text_layer.encode("utf-8"))


def _default_language_metadata(spans: list[dict[str, Any]]) -> dict[str, Any]:
    languages = [span["language"] for span in spans if span.get("language")]
    primary = languages[0] if languages else "und"
    return {
        "primary_language": primary,
        "detected_languages": sorted(set(languages or [primary])),
        "source": "span_fixture",
    }


def _pages_from_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        grouped[span["page_number"]].append(span)

    pages: list[dict[str, Any]] = []
    for page_number in sorted(grouped):
        page_spans = grouped[page_number]
        pages.append(
            {
                "page_number": page_number,
                "text": "\n".join(span["text"] for span in page_spans),
                "span_ids": [span["span_id"] for span in page_spans],
            }
        )
    return pages


def _default_artifact_manifest(
    source_file_name: str,
    source_file_sha256: str,
    source_ocr_sha256: str,
) -> dict[str, Any]:
    return {
        "artifacts": [
            {
                "artifact_id": "source_file",
                "artifact_type": "source_file",
                "path": source_file_name,
                "sha256": source_file_sha256,
            },
            {
                "artifact_id": "ocr_text",
                "artifact_type": "ocr_text",
                "sha256": source_ocr_sha256,
                "mime_type": "text/plain",
            },
        ]
    }
