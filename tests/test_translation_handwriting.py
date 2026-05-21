"""Tests for ocr_local.translation.handwriting opt-in MT path."""

from __future__ import annotations

import importlib

from ocr_local.translation.engines.passthrough import PassthroughEngine
from ocr_local.translation.models import EngineCapability, SpanTranslation


class _NativeHandwritingEngine(PassthroughEngine):
    """Test fixture -- engine that claims native handwriting support."""

    capability = EngineCapability(
        id="test_native_hw",
        is_local=True,
        is_cloud=False,
        supports_pairs="any",
        quality_class="draft",
        latency_class="realtime",
        license="Apache-2.0",
        provider_retention_class="local_only",
        deployment_envs=["local"],
        cost_per_1m_chars_usd=0.0,
        cost_per_1m_tokens_usd=0.0,
        handles_handwriting_natively=True,
    )


def _reload_handwriting(monkeypatch, value: str | None):
    """Helper -- toggle ENABLE_HANDWRITING_MT and reload module."""
    if value is None:
        monkeypatch.delenv("ENABLE_HANDWRITING_MT", raising=False)
    else:
        monkeypatch.setenv("ENABLE_HANDWRITING_MT", value)
    import ocr_local.translation.handwriting as hw

    return importlib.reload(hw)


def test_should_translate_handwriting_default_off(monkeypatch):
    hw = _reload_handwriting(monkeypatch, None)
    assert hw.ENABLE_HANDWRITING_MT is False
    assert hw.should_translate_handwriting({"is_handwriting": True}) is False


def test_should_translate_handwriting_when_enabled(monkeypatch):
    hw = _reload_handwriting(monkeypatch, "true")
    assert hw.ENABLE_HANDWRITING_MT is True
    assert hw.should_translate_handwriting({"is_handwriting": True}) is True


def test_should_translate_handwriting_engine_disabled(monkeypatch):
    hw = _reload_handwriting(monkeypatch, "true")
    assert hw.should_translate_handwriting({"is_handwriting": False}) is False


def test_should_translate_when_flag_off_and_handwriting(monkeypatch):
    hw = _reload_handwriting(monkeypatch, "false")
    assert hw.should_translate_handwriting({"is_handwriting": True}) is False


def test_translate_handwriting_uses_ocr_fallback(monkeypatch):
    hw = _reload_handwriting(monkeypatch, "true")
    engine = PassthroughEngine()
    span = {
        "text": "raw_handwriting_glyphs",
        "is_handwriting": True,
        "span_id": "hw_a",
        "bbox": [0.0, 0.0, 50.0, 10.0],
    }
    result = hw.translate_handwriting_span(
        span,
        engine,
        ocr_text_fallback="cleaned ocr text",
        src="en",
        tgt="fr",
    )
    assert result is not None
    # Engine does not handle handwriting natively => OCR fallback used.
    assert result.source_text == "cleaned ocr text"


def test_translate_handwriting_uses_native_when_capable(monkeypatch):
    hw = _reload_handwriting(monkeypatch, "true")
    engine = _NativeHandwritingEngine()
    span = {
        "text": "raw native handwriting",
        "is_handwriting": True,
        "span_id": "hw_b",
    }
    result = hw.translate_handwriting_span(
        span,
        engine,
        ocr_text_fallback="ocr fallback should NOT be used",
        src="en",
        tgt="fr",
    )
    assert result is not None
    assert result.source_text == "raw native handwriting"


def test_translate_handwriting_returns_span_translation(monkeypatch):
    hw = _reload_handwriting(monkeypatch, "true")
    engine = PassthroughEngine()
    span = {"text": "ignored", "is_handwriting": True, "span_id": "hw_c"}
    result = hw.translate_handwriting_span(
        span, engine, ocr_text_fallback="fallback", src="en", tgt="fr"
    )
    assert isinstance(result, SpanTranslation)


def test_translate_handwriting_engine_id_set(monkeypatch):
    hw = _reload_handwriting(monkeypatch, "true")
    engine = PassthroughEngine()
    span = {"text": "ignored", "is_handwriting": True, "span_id": "hw_d"}
    result = hw.translate_handwriting_span(
        span, engine, ocr_text_fallback="fallback", src="en", tgt="fr"
    )
    assert result.engine_id == engine.capability.id
