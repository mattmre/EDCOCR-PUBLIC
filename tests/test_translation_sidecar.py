"""Tests for ocr_local.translation.sidecar -- Plan B Wave M1 PR2."""

from __future__ import annotations

import json

import pytest

from ocr_local.translation.models import (
    DocumentTranslation,
    PageTranslation,
    SpanTranslation,
)
from ocr_local.translation.sidecar import (
    SchemaValidationError,
    _validate_doc,
    write_translation_json,
    write_translation_md,
)


def make_doc(
    *,
    document_id: str = "doc_42",
    target_language: str = "fr",
    certified: bool = False,
    pages: list[PageTranslation] | None = None,
) -> DocumentTranslation:
    """Build a valid DocumentTranslation for tests."""
    if pages is None:
        pages = [
            PageTranslation(
                page_num=1,
                spans=[
                    SpanTranslation(
                        span_id="p1_s1",
                        source_text="hello",
                        target_text="bonjour",
                        source_bbox=[0.0, 0.0, 100.0, 12.0],
                        source_bboxes=[[0.0, 0.0, 100.0, 12.0]],
                        source_language="en",
                        target_language=target_language,
                        confidence=0.95,
                        quality_score=0.88,
                        engine_id="passthrough",
                        glossary_hits=[],
                    )
                ],
            )
        ]
    return DocumentTranslation(
        schema_version="1.0",
        document_id=document_id,
        source_file=f"{document_id}.pdf",
        source_language="en",
        target_language=target_language,
        certified=certified,
        engine={
            "id": "passthrough",
            "is_local": True,
            "license": "Apache-2.0",
            "provider_retention_class": "local_only",
            "weights_sha256": "",
        },
        glossary=None,
        quality={
            "mean_score": 0.88,
            "below_threshold_count": 0,
            "quality_class": "standard",
        },
        pages=pages,
        stats={"chars": 5, "tokens": 1},
        custody={
            "chain_head": "deadbeef",
            "clock_source": "monotonic_ntp",
            "source_ocr_sha256": "cafebabe",
        },
        processing={
            "pipeline_version": "1.2.1",
            "timestamp": "2026-04-24T00:00:00Z",
            "enable_translation": True,
        },
    )


def test_write_json_creates_file(tmp_path):
    doc = make_doc()
    path = write_translation_json(doc, str(tmp_path))
    import os

    assert os.path.exists(path)


def test_write_json_path_contains_target_language(tmp_path):
    doc = make_doc(target_language="fr")
    path = write_translation_json(doc, str(tmp_path))
    assert path.endswith(".fr.translation.json")


def test_write_json_certified_false_by_default(tmp_path):
    doc = make_doc()
    path = write_translation_json(doc, str(tmp_path))
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["certified"] is False


def test_write_json_certified_true_raises(tmp_path):
    doc = make_doc(certified=True)
    with pytest.raises(SchemaValidationError):
        write_translation_json(doc, str(tmp_path))


def test_write_json_schema_version(tmp_path):
    doc = make_doc()
    path = write_translation_json(doc, str(tmp_path))
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["schema_version"] == "1.0"


def test_write_json_export_path_structure(tmp_path):
    doc = make_doc()
    path = write_translation_json(doc, str(tmp_path))
    norm = path.replace("\\", "/")
    assert "EXPORT/TRANSLATION/" in norm


def test_write_json_subfolder(tmp_path):
    doc = make_doc()
    path = write_translation_json(doc, str(tmp_path), subfolder="batch_a/sub")
    norm = path.replace("\\", "/")
    assert "EXPORT/TRANSLATION/batch_a/sub/" in norm


def test_write_json_round_trip(tmp_path):
    doc = make_doc(document_id="abc", target_language="es")
    path = write_translation_json(doc, str(tmp_path))
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["document_id"] == "abc"
    assert payload["target_language"] == "es"
    assert payload["source_language"] == "en"
    assert payload["pages"][0]["page_num"] == 1
    assert payload["pages"][0]["spans"][0]["target_text"] == "bonjour"
    assert payload["engine"]["id"] == "passthrough"


def test_write_md_creates_file(tmp_path):
    doc = make_doc()
    path = write_translation_md(doc, str(tmp_path))
    import os

    assert os.path.exists(path)
    assert path.endswith(".md")


def test_write_md_contains_source_language(tmp_path):
    doc = make_doc()
    path = write_translation_md(doc, str(tmp_path))
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "source_language: en" in content


def test_write_md_certified_false_shown(tmp_path):
    doc = make_doc()
    path = write_translation_md(doc, str(tmp_path))
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "certified: false" in content


def test_validate_doc_passes_valid():
    doc = make_doc()
    # Must not raise
    _validate_doc(doc)


def test_validate_doc_rejects_certified_true():
    doc = make_doc(certified=True)
    with pytest.raises(SchemaValidationError):
        _validate_doc(doc)


def test_write_json_empty_pages(tmp_path):
    doc = make_doc(pages=[])
    path = write_translation_json(doc, str(tmp_path))
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["pages"] == []
