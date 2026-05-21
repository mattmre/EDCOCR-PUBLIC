"""Contract-first tests for EDC DocumentBundle and TranslationBundle v1."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from ocr_local.contracts import canonical_json_sha256, load_contract_schema
from ocr_local.document_bundle import (
    build_document_bundle,
    validate_document_bundle,
    write_document_bundle,
)
from ocr_local.translation.bundle_adapter import (
    translate_document_bundle,
    validate_translation_bundle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "edc_contracts"


def _load_fixture(name: str) -> dict:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize(
    ("schema_name", "fixture_name"),
    [
        ("document-bundle-v1", "document-bundle-v1.valid.json"),
        ("translation-bundle-v1", "translation-bundle-v1.valid.json"),
    ],
)
def test_bundle_schema_fixtures_validate(schema_name: str, fixture_name: str):
    schema = load_contract_schema(schema_name)
    payload = _load_fixture(fixture_name)

    jsonschema.Draft7Validator.check_schema(schema)
    jsonschema.validate(payload, schema)


@pytest.mark.parametrize(
    ("schema_name", "fixture_name", "field_name"),
    [
        ("document-bundle-v1", "document-bundle-v1.valid.json", "source_ocr_sha256"),
        ("translation-bundle-v1", "translation-bundle-v1.valid.json", "source_bundle_sha256"),
    ],
)
def test_bundle_schema_rejects_missing_required_field(
    schema_name: str,
    fixture_name: str,
    field_name: str,
):
    payload = _load_fixture(fixture_name)
    payload.pop(field_name)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, load_contract_schema(schema_name))


def test_contract_schema_loader_reads_bundle_contracts():
    assert load_contract_schema("document-bundle-v1")["title"] == "EDC DocumentBundle v1"
    assert (
        load_contract_schema("translation-bundle-v1")["title"]
        == "EDC TranslationBundle v1"
    )


def test_document_bundle_exporter_preserves_span_geometry():
    bundle = build_document_bundle(
        document_id="doc-123",
        source_file_name="sample.pdf",
        source_file_bytes=b"%PDF fixture bytes",
        spans=_ocr_like_spans(),
        custody_chain_head="custody-head-123",
        ocr_engine_metadata={
            "engine_id": "ocr_local.test",
            "engine_version": "fixture",
        },
    )

    validate_document_bundle(bundle)
    assert bundle["schema_version"] == "document-bundle-v1"
    assert bundle["source_file_sha256"] != bundle["source_ocr_sha256"]
    assert bundle["pages"][0]["span_ids"] == ["p1-s1", "p1-s2"]
    assert bundle["spans"][1]["bbox"] == [20.0, 50.0, 210.0, 72.0]
    assert bundle["spans"][1]["bboxes"] == [[20.0, 50.0, 210.0, 72.0]]


def test_write_document_bundle_round_trip(tmp_path: Path):
    bundle = build_document_bundle(
        document_id="doc-write",
        source_file_name="write.pdf",
        source_file_bytes=b"write fixture",
        spans=_ocr_like_spans()[:1],
    )

    out_path = write_document_bundle(bundle, tmp_path / "bundle.json")
    loaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert loaded == bundle
    validate_document_bundle(loaded)


def test_translate_document_bundle_e2e_passthrough():
    document_bundle = build_document_bundle(
        document_id="doc-e2e",
        source_file_name="e2e.pdf",
        source_file_bytes=b"e2e fixture",
        spans=_ocr_like_spans(),
        custody_chain_head="custody-head-e2e",
    )
    frozen_input = copy.deepcopy(document_bundle)

    translation_bundle = translate_document_bundle(
        document_bundle,
        target_language="fr",
        engine_id="passthrough",
    )

    validate_translation_bundle(translation_bundle)
    assert document_bundle == frozen_input
    assert translation_bundle["schema_version"] == "translation-bundle-v1"
    assert translation_bundle["document_id"] == "doc-e2e"
    assert translation_bundle["source_bundle_sha256"] == canonical_json_sha256(
        document_bundle
    )
    assert translation_bundle["target_language"] == "fr"
    assert translation_bundle["engine_provider"]["id"] == "passthrough"
    assert translation_bundle["certified"] is False
    assert [span["span_id"] for span in translation_bundle["translated_spans"]] == [
        "p1-s1",
        "p1-s2",
    ]
    assert translation_bundle["translated_spans"][0]["source_bbox"] == [
        12.0,
        20.0,
        120.0,
        42.0,
    ]


def test_translate_document_bundle_rejects_invalid_input():
    document_bundle = build_document_bundle(
        document_id="doc-invalid",
        source_file_name="invalid.pdf",
        source_file_bytes=b"invalid fixture",
        spans=_ocr_like_spans()[:1],
    )
    document_bundle.pop("source_ocr_sha256")

    with pytest.raises(jsonschema.ValidationError):
        translate_document_bundle(
            document_bundle,
            target_language="fr",
            engine_id="passthrough",
        )


def _ocr_like_spans() -> list[dict]:
    return [
        {
            "span_id": "p1-s1",
            "page_number": 1,
            "text": "Hello world.",
            "bbox": [12.0, 20.0, 120.0, 42.0],
            "language": "en",
            "confidence": 0.99,
        },
        {
            "span_id": "p1-s2",
            "page_number": 1,
            "text": "This span carries OCR geometry.",
            "bbox": [20.0, 50.0, 210.0, 72.0],
            "language": "en",
            "confidence": 0.98,
        },
    ]
