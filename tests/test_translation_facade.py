"""Direct tests for ``ocr_local.translation.api`` -- Plan B M1-PR5.

These tests mock heavy collaborators (engines, metrics) to exercise the
facade's data-flow + invariants in isolation.
"""
from __future__ import annotations

from unittest.mock import patch


def test_translate_document_empty_snap():
    """page_data_snap=None -> still returns one doc per target with 0 pages."""
    from ocr_local.translation.api import translate_document

    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr"],
        tenant_id="default",
        page_data_snap=None,
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].pages == []


def test_translate_document_with_snap():
    """page_data_snap with text_by_page -> returns docs with pages populated."""
    from ocr_local.translation.api import translate_document

    snap = {"text_by_page": {1: "Hello world.", 2: "Second page."}}
    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr"],
        tenant_id="default",
        page_data_snap=snap,
    )
    assert len(result) == 1
    assert len(result[0].pages) == 2


def test_facade_certified_false():
    """certified is never True for raw facade output."""
    from ocr_local.translation.api import translate_document

    snap = {"text_by_page": {1: "ok"}}
    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr", "es", "de"],
        tenant_id="default",
        page_data_snap=snap,
    )
    assert all(d.certified is False for d in result)


def test_facade_tenant_policy_hash_in_custody():
    from ocr_local.translation.api import translate_document

    snap = {"text_by_page": {1: "x"}}
    with patch(
        "ocr_local.translation.api.compute_policy_hash", return_value="HASH123"
    ):
        result = translate_document(
            doc_path="/tmp/doc.pdf",
            target_languages=["fr"],
            tenant_id="default",
            page_data_snap=snap,
        )
    assert result[0].custody["tenant_policy_hash"] == "HASH123"


def test_facade_passthrough_for_same_language():
    """When src==tgt, the router selects passthrough -> target_text==source_text."""
    from ocr_local.translation.api import translate_document

    # Mark src=fr via document_level so router gets fr->fr.
    snap = {"text_by_page": {1: "Bonjour."}, "detected_language": "fr"}
    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr"],
        tenant_id="default",
        page_data_snap=snap,
    )
    assert len(result) == 1
    doc = result[0]
    assert doc.engine["id"] == "passthrough"
    # Each span's target_text equals source_text for passthrough.
    for page in doc.pages:
        for span in page.spans:
            assert span.target_text == span.source_text


def test_facade_fails_open_on_policy_denied():
    """PolicyDenied from router -> empty list, no exception."""
    from ocr_local.translation.api import translate_document
    from ocr_local.translation.custody_adapter import ReasonCode
    from ocr_local.translation.policy import PolicyDenied

    with patch(
        "ocr_local.translation.api._instantiate_engine",
        side_effect=PolicyDenied(ReasonCode.TENANT_POLICY, "blocked"),
    ):
        result = translate_document(
            doc_path="/tmp/doc.pdf",
            target_languages=["fr"],
            tenant_id="default",
            page_data_snap={"text_by_page": {1: "hello"}},
        )
    assert result == []


def test_facade_emits_chars_metric():
    """record_translation_chars is called with the per-engine total."""
    from ocr_local.translation.api import translate_document

    snap = {"text_by_page": {1: "hello world"}}
    with patch(
        "ocr_local.translation.api.record_translation_chars"
    ) as mock_record:
        translate_document(
            doc_path="/tmp/doc.pdf",
            target_languages=["fr"],
            tenant_id="default",
            page_data_snap=snap,
        )
    assert mock_record.called
    # First positional: tenant_id; we just confirm a call occurred with our tenant.
    args, _ = mock_record.call_args
    assert args[0] == "default"


def test_facade_multiple_languages():
    from ocr_local.translation.api import translate_document

    snap = {"text_by_page": {1: "hi"}}
    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr", "de"],
        tenant_id="default",
        page_data_snap=snap,
    )
    assert {d.target_language for d in result} == {"fr", "de"}


def test_facade_processing_timestamp_set():
    """processing.timestamp is an ISO-8601 string ending with Z."""
    from ocr_local.translation.api import translate_document

    snap = {"text_by_page": {1: "hi"}}
    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr"],
        tenant_id="default",
        page_data_snap=snap,
    )
    ts = result[0].processing["timestamp"]
    assert isinstance(ts, str)
    assert ts.endswith("Z")
    # Crude ISO check: contains 'T' between date + time.
    assert "T" in ts


def test_facade_schema_version_1_0():
    from ocr_local.translation.api import translate_document

    snap = {"text_by_page": {1: "hi"}}
    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr"],
        tenant_id="default",
        page_data_snap=snap,
    )
    assert result[0].schema_version == "1.0"
