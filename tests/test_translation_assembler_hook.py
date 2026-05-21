"""Tests for the translation assembler facade -- Plan B M1-PR5.

These tests exercise ``ocr_local.translation.api.translate_document`` in
isolation: they do NOT import PaddleOCR, do NOT start the real pipeline,
and use mocks throughout for the engine registry and tenant policy
hooks.  The goal is to verify the fail-open semantics + the data flow
the assembler hook in ``ocr_gpu_async.py`` depends on.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_translate_document_returns_list():
    """Empty page_data_snap -> empty list (no spans to translate)."""
    from ocr_local.translation.api import translate_document

    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr"],
        tenant_id="default",
        page_data_snap=None,
    )
    assert isinstance(result, list)
    # One DocumentTranslation is still produced (with zero pages); but
    # the contract is "returns a list" -- not "empty list when snap None".
    assert len(result) == 1
    assert result[0].pages == []


def test_translate_document_fails_open():
    """If the engine raises, returns [] (no exception escapes)."""
    from ocr_local.translation.api import translate_document

    with patch(
        "ocr_local.translation.api._instantiate_engine",
        side_effect=RuntimeError("engine boom"),
    ):
        result = translate_document(
            doc_path="/tmp/doc.pdf",
            target_languages=["fr"],
            tenant_id="default",
            page_data_snap={"text_by_page": {1: "hello"}},
        )
    assert result == []


def test_translate_document_certified_false():
    """All returned docs have certified=False (NEVER True from raw output)."""
    from ocr_local.translation.api import translate_document

    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr", "es"],
        tenant_id="default",
        page_data_snap={"text_by_page": {1: "hello world"}},
    )
    assert len(result) == 2
    assert all(doc.certified is False for doc in result)


def test_translate_document_multiple_targets():
    """Two targets -> two DocumentTranslation instances."""
    from ocr_local.translation.api import translate_document

    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr", "de"],
        tenant_id="default",
        page_data_snap={"text_by_page": {1: "Hello."}},
    )
    assert len(result) == 2
    targets = {d.target_language for d in result}
    assert targets == {"fr", "de"}


def test_resolve_source_language_from_plan_a():
    """Plan A languages dict -> source='plan_a'."""
    from ocr_local.translation.api import _resolve_source_language

    snap = {
        "languages": {
            1: [{"language": "fr"}, {"language": "fr"}, {"language": "en"}],
            2: [{"language": "fr"}],
        }
    }
    src, source = _resolve_source_language(snap)
    assert src == "fr"
    assert source == "plan_a"


def test_resolve_source_language_document_level():
    """detected_language without per-span data -> source='document_level'."""
    from ocr_local.translation.api import _resolve_source_language

    snap = {"detected_language": "es"}
    src, source = _resolve_source_language(snap)
    assert src == "es"
    assert source == "document_level"


def test_resolve_source_language_from_page_language_snap():
    """Assembler PageLanguage snapshots under language feed source resolution."""
    from ocr_local.translation.api import _resolve_source_language

    snap = {
        "language": {
            1: SimpleNamespace(primary_language="de", span_count=2),
            2: {"primary_language": "fr", "span_count": 1},
        }
    }
    src, source = _resolve_source_language(snap)
    assert src == "de"
    assert source == "plan_a"


def test_resolve_source_language_default_fallback():
    """None snap -> ('en', 'default_fallback')."""
    from ocr_local.translation.api import _resolve_source_language

    src, source = _resolve_source_language(None)
    assert src == "en"
    assert source == "default_fallback"


def test_noop_chain_has_log_event():
    """_make_noop_chain returns an object whose log_event does not raise."""
    from ocr_local.translation.api import _make_noop_chain

    chain = _make_noop_chain()
    # Should not raise -- both methods are no-ops.
    chain.log_event("ANYTHING", {"k": "v"})
    chain.append_event("ANYTHING_ELSE", {"k": "v"})


def test_translate_document_uses_policy():
    """Tenant policy is loaded once and passed through router."""
    from ocr_local.translation.api import translate_document

    with patch(
        "ocr_local.translation.api.load_tenant_policy"
    ) as mock_load, patch(
        "ocr_local.translation.api.compute_policy_hash",
        return_value="hash_xyz",
    ):
        mock_load.return_value = MagicMock()
        result = translate_document(
            doc_path="/tmp/doc.pdf",
            target_languages=["fr"],
            tenant_id="acme",
            page_data_snap={"text_by_page": {}},
        )
    mock_load.assert_called_once_with("acme")
    assert len(result) == 1
    assert result[0].custody["tenant_policy_hash"] == "hash_xyz"


def test_translate_to_language_assembles_document():
    """Returned DocumentTranslation has schema_version='1.0' + source_file set."""
    from ocr_local.translation.api import translate_document

    result = translate_document(
        doc_path="/data/inputs/foo.pdf",
        target_languages=["fr"],
        tenant_id="default",
        page_data_snap={"text_by_page": {1: "Hello world."}},
    )
    assert len(result) == 1
    doc = result[0]
    assert doc.schema_version == "1.0"
    assert doc.source_file == "foo.pdf"


def test_translate_document_tenant_id_forwarded():
    """tenant_id is recorded in custody block of every returned doc."""
    from ocr_local.translation.api import translate_document

    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr", "es"],
        tenant_id="tenant-42",
        page_data_snap={"text_by_page": {1: "ok"}},
    )
    assert len(result) == 2
    for doc in result:
        assert doc.custody.get("tenant_id") == "tenant-42"


def test_translate_document_empty_target_languages():
    """No targets -> empty list, no error."""
    from ocr_local.translation.api import translate_document

    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=[],
        tenant_id="default",
        page_data_snap={"text_by_page": {1: "hello"}},
    )
    assert result == []


def test_translate_document_external_preferred_by_env(monkeypatch):
    """Opt-in external service path maps TranslationBundle to legacy dataclass."""
    from ocr_local.contracts import canonical_json_sha256
    from ocr_local.translation.api import translate_document
    from ocr_local.translation.external_client import TranslationReadinessStatus

    captured = {}

    class _Client:
        def __init__(self, base_url, *, api_key=None, timeout=30.0):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["timeout"] = timeout

        def translate_bundle(self, document_bundle, *, target_language, provider_id):
            captured["document_bundle"] = document_bundle
            captured["target_language"] = target_language
            captured["provider_id"] = provider_id
            return {
                "schema_version": "translation-bundle-v1",
                "document_id": document_bundle["document_id"],
                "source_ocr_sha256": document_bundle["source_ocr_sha256"],
                "source_bundle_sha256": canonical_json_sha256(document_bundle),
                "target_language": target_language,
                "translated_spans": [
                    {
                        "span_id": document_bundle["spans"][0]["span_id"],
                        "page_number": 1,
                        "source_text": "Hello.",
                        "translated_text": "Bonjour.",
                        "source_bbox": [0.0, 0.0, 100.0, 12.0],
                        "source_bboxes": [[0.0, 0.0, 100.0, 12.0]],
                        "source_language": "en",
                        "target_language": "fr",
                        "confidence": 0.99,
                        "quality_score": 0.91,
                        "engine_id": "stub",
                        "glossary_hits": [],
                    }
                ],
                "engine_provider": {
                    "id": "stub",
                    "family": "passthrough",
                    "is_local": True,
                    "is_cloud": False,
                    "license": "Apache-2.0",
                    "provider_retention_class": "local_only",
                },
                "model_provenance": {"weights_sha256": "n/a"},
                "quality_scores": {
                    "mean_score": 0.91,
                    "below_threshold_count": 0,
                    "quality_class": "draft",
                },
                "certified": False,
                "custody_chain_head": "external-head",
                "artifact_manifest": {"artifacts": []},
            }

    monkeypatch.setenv("EDC_TRANSLATION_PREFER_EXTERNAL", "true")
    monkeypatch.setenv("EDC_TRANSLATION_URL", "http://translation.local")
    monkeypatch.setenv("EDC_TRANSLATION_PROVIDER_ID", "stub")
    monkeypatch.setattr("ocr_local.translation.api.TranslationServiceClient", _Client)
    monkeypatch.setattr(
        "ocr_local.translation.api.external_translation_readiness",
        lambda *_args, **_kwargs: TranslationReadinessStatus(
            status="ready",
            ready=True,
            enabled=True,
            url="http://translation.local/health",
            message="ready",
        ),
    )

    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr"],
        tenant_id="tenant-a",
        page_data_snap={"texts": {1: "Hello."}, "detected_language": "en"},
    )

    assert len(result) == 1
    assert result[0].pages[0].spans[0].target_text == "Bonjour."
    assert result[0].engine["id"] == "stub"
    assert result[0].custody["source_ocr_sha256"] == captured["document_bundle"][
        "source_ocr_sha256"
    ]
    assert result[0].custody["tenant_policy_hash"] == captured["document_bundle"][
        "tenant_policy_hash"
    ]
    assert captured["base_url"] == "http://translation.local"
    assert captured["provider_id"] == "stub"


def test_translate_document_external_failure_falls_back(monkeypatch):
    """External service errors do not remove the existing local fail-open path."""
    from ocr_local.translation.api import translate_document

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def translate_bundle(self, *_args, **_kwargs):
            raise RuntimeError("service down")

    monkeypatch.setenv("EDC_TRANSLATION_PREFER_EXTERNAL", "true")
    monkeypatch.setattr("ocr_local.translation.api.TranslationServiceClient", _Client)

    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr"],
        tenant_id="default",
        page_data_snap={"text_by_page": {1: "Hello."}, "detected_language": "en"},
    )

    assert len(result) == 1
    assert result[0].processing["enable_translation"] is True
    assert result[0].processing.get("translation_service") != "external"


def test_translate_document_external_readiness_blocks_dispatch(monkeypatch):
    """Preference alone is not enough; readiness must be green."""
    from ocr_local.translation.api import translate_document
    from ocr_local.translation.external_client import TranslationReadinessStatus

    class _Client:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("external client should not be constructed")

    monkeypatch.setenv("EDC_TRANSLATION_PREFER_EXTERNAL", "true")
    monkeypatch.setenv("EDC_TRANSLATION_URL", "http://translation.local")
    monkeypatch.setattr("ocr_local.translation.api.TranslationServiceClient", _Client)
    monkeypatch.setattr(
        "ocr_local.translation.api.external_translation_readiness",
        lambda *_args, **_kwargs: TranslationReadinessStatus(
            status="unreachable",
            ready=False,
            enabled=True,
            url="http://translation.local/health",
            message="connection refused",
        ),
    )

    result = translate_document(
        doc_path="/tmp/doc.pdf",
        target_languages=["fr"],
        tenant_id="default",
        page_data_snap={"text_by_page": {1: "Hello."}, "detected_language": "en"},
    )

    assert len(result) == 1
    assert result[0].processing.get("translation_service") != "external"
