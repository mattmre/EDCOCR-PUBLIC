"""Tests for ``ocr_local.translation.models`` dataclasses."""

from __future__ import annotations

import dataclasses
import json

import pytest

from ocr_local.translation.models import (
    DocumentTranslation,
    EngineCapability,
    PageTranslation,
    SpanTranslation,
    TranslationRequest,
)


def _make_span(span_id: str = "s0", text: str = "hello") -> SpanTranslation:
    return SpanTranslation(
        span_id=span_id,
        source_text=text,
        target_text=text,
        source_bbox=[0.0, 0.0, 100.0, 12.0],
        source_bboxes=[[0.0, 0.0, 100.0, 12.0]],
        source_language="en",
        target_language="en",
        confidence=1.0,
        quality_score=None,
        engine_id="passthrough",
    )


def test_span_translation_construction_and_field_access():
    span = _make_span(span_id="abc", text="hello world")
    assert span.span_id == "abc"
    assert span.source_text == "hello world"
    assert span.target_text == "hello world"
    assert span.source_bbox == [0.0, 0.0, 100.0, 12.0]
    assert span.source_language == "en"
    assert span.target_language == "en"
    assert span.confidence == 1.0
    assert span.quality_score is None
    assert span.engine_id == "passthrough"


def test_document_translation_certified_default_false():
    """Critical invariant -- certified must NEVER default to True."""

    doc = DocumentTranslation(
        schema_version="1.0",
        document_id="doc-123",
        source_file="/tmp/x.pdf",
        source_language="en",
        target_language="fr",
    )
    assert doc.certified is False


def test_engine_capability_is_frozen():
    cap = EngineCapability(
        id="x",
        is_local=True,
        is_cloud=False,
        supports_pairs="any",
        quality_class="draft",
        latency_class="realtime",
        license="Apache-2.0",
        provider_retention_class="local_only",
        deployment_envs=["local"],
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.id = "y"  # type: ignore[misc]


def test_translation_request_defaults():
    req = TranslationRequest(src_lang="en", tgt_lang="fr")
    assert req.quality == "standard"
    assert req.latency == "standard"
    assert req.privilege_flag is False
    assert req.tenant_id == "default"


def test_span_translation_empty_glossary_hits_default():
    span = _make_span()
    assert span.glossary_hits == []


def test_page_translation_with_multiple_spans():
    spans = [_make_span(span_id=f"s{i}", text=f"t{i}") for i in range(5)]
    page = PageTranslation(page_num=1, spans=spans)
    assert page.page_num == 1
    assert len(page.spans) == 5
    assert page.spans[2].span_id == "s2"


def test_document_translation_with_pages():
    page1 = PageTranslation(page_num=1, spans=[_make_span("s0")])
    page2 = PageTranslation(page_num=2, spans=[_make_span("s1")])
    doc = DocumentTranslation(
        schema_version="1.0",
        document_id="doc-1",
        source_file="/tmp/x.pdf",
        source_language="en",
        target_language="fr",
        pages=[page1, page2],
    )
    assert len(doc.pages) == 2
    assert doc.pages[0].page_num == 1
    assert doc.pages[1].page_num == 2


def test_engine_capability_supports_pairs_any():
    cap = EngineCapability(
        id="x",
        is_local=True,
        is_cloud=False,
        supports_pairs="any",
        quality_class="draft",
        latency_class="realtime",
        license="Apache-2.0",
        provider_retention_class="local_only",
        deployment_envs=["local"],
    )
    assert cap.supports_pairs == "any"


def test_engine_capability_supports_pairs_list_of_tuples():
    pairs = [("en", "fr"), ("en", "de"), ("fr", "en")]
    cap = EngineCapability(
        id="x",
        is_local=True,
        is_cloud=False,
        supports_pairs=pairs,
        quality_class="standard",
        latency_class="standard",
        license="MIT",
        provider_retention_class="local_only",
        deployment_envs=["local", "air_gapped"],
    )
    assert cap.supports_pairs == pairs
    assert ("en", "fr") in cap.supports_pairs


def test_span_translation_source_bboxes_polygon_list():
    """Multi-line spans must support a polygon list of bboxes."""

    polygon = [
        [0.0, 0.0, 100.0, 12.0],
        [0.0, 12.0, 100.0, 24.0],
        [0.0, 24.0, 80.0, 36.0],
    ]
    span = SpanTranslation(
        span_id="s0",
        source_text="multi-line",
        target_text="multi-line",
        source_bbox=polygon[0],
        source_bboxes=polygon,
        source_language="en",
        target_language="en",
        confidence=0.9,
        quality_score=0.8,
        engine_id="passthrough",
    )
    assert span.source_bboxes == polygon
    assert len(span.source_bboxes) == 3


def test_document_translation_stats_default_empty_dict():
    doc = DocumentTranslation(
        schema_version="1.0",
        document_id="doc-1",
        source_file="/tmp/x.pdf",
        source_language="en",
        target_language="fr",
    )
    assert doc.stats == {}
    assert doc.engine == {}
    assert doc.quality == {}
    assert doc.custody == {}
    assert doc.processing == {}
    assert doc.glossary is None


def test_document_translation_asdict_is_json_safe():
    """``dataclasses.asdict`` round-trip must produce a JSON-safe dict."""

    page = PageTranslation(page_num=1, spans=[_make_span("s0", "hello")])
    doc = DocumentTranslation(
        schema_version="1.0",
        document_id="doc-1",
        source_file="/tmp/x.pdf",
        source_language="en",
        target_language="en",
        pages=[page],
        stats={"page_count": 1},
    )
    payload = dataclasses.asdict(doc)
    # Must be JSON-serialisable round-trip.
    blob = json.dumps(payload)
    parsed = json.loads(blob)
    assert parsed["document_id"] == "doc-1"
    assert parsed["certified"] is False
    assert parsed["pages"][0]["page_num"] == 1
    assert parsed["pages"][0]["spans"][0]["span_id"] == "s0"
