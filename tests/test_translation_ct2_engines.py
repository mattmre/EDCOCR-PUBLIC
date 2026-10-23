"""Consumer fail-closed contracts for missing CT2 runtimes and models."""

from __future__ import annotations

import importlib.util
import sys
from unittest import mock

import pytest

from ocr_local.translation.engines import ENGINE_REGISTRY
from ocr_local.translation.engines.local_ct2_madlad import MADLADEngine
from ocr_local.translation.engines.local_ct2_nllb import NLLBEngine
from ocr_local.translation.engines.local_ct2_opus import OpusMTEngine

_HAS_CT2 = importlib.util.find_spec("ctranslate2") is not None

# Importing the CT2 engine modules above triggers ``@register_engine``
# at import time.  When ``ctranslate2`` is missing, downstream tests
# (notably ``tests/test_translation_policy.py``) exercise the router
# and would try to instantiate these engines, tripping the init guard.
# Roll the registrations back when the optional runtime is unavailable so
# unrelated policy tests cannot select adapters they cannot instantiate.
_CT2_ENGINE_IDS = ("local_ct2_opus", "local_ct2_nllb", "local_ct2_madlad")
if not _HAS_CT2:
    for _engine_id in _CT2_ENGINE_IDS:
        ENGINE_REGISTRY.pop(_engine_id, None)



# ---------------------------------------------------------------------------
# Init guards -- real model dirs raise RuntimeError when ctranslate2 is missing
# ---------------------------------------------------------------------------


def test_opus_init_raises_without_ct2():
    with mock.patch.dict(sys.modules, {"ctranslate2": None}):
        with pytest.raises(RuntimeError, match="ctranslate2"):
            OpusMTEngine(model_dir="models/opus")


def test_opus_init_reports_missing_sentencepiece(monkeypatch, tmp_path):
    model_dir = tmp_path / "opus"
    model_dir.mkdir()
    (model_dir / "source.spm").write_text("fixture", encoding="utf-8")
    (model_dir / "target.spm").write_text("fixture", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "ctranslate2", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "sentencepiece", None)

    with pytest.raises(RuntimeError, match="sentencepiece"):
        OpusMTEngine(model_dir=str(model_dir))


def test_nllb_init_raises_without_ct2():
    with mock.patch.dict(sys.modules, {"ctranslate2": None}):
        with pytest.raises(RuntimeError, match="ctranslate2"):
            NLLBEngine(model_dir="models/nllb")


def test_madlad_init_raises_without_ct2():
    with mock.patch.dict(sys.modules, {"ctranslate2": None}):
        with pytest.raises(RuntimeError, match="ctranslate2"):
            MADLADEngine(model_dir="models/madlad")


# ---------------------------------------------------------------------------
# Fail-closed translation path (no synthetic translations)
# ---------------------------------------------------------------------------


def test_opus_without_model_fails_closed():
    engine = OpusMTEngine()
    spans = [{"text": "hello", "span_id": "s0", "bbox": [0, 0, 10, 10]}]
    with pytest.raises(RuntimeError, match="model_dir is required"):
        engine.translate_spans(spans, "en", "fr")


def test_nllb_adapter_fails_closed():
    engine = NLLBEngine()
    spans = [{"text": "world", "span_id": "s0"}]
    with pytest.raises(RuntimeError, match="model_dir is required"):
        engine.translate_spans(spans, "en", "fr")


def test_madlad_adapter_fails_closed():
    engine = MADLADEngine()
    spans = [{"text": "foo", "span_id": "s0"}]
    with pytest.raises(RuntimeError, match="model_dir is required"):
        engine.translate_spans(spans, "en", "fr")


def test_real_nllb_adapter_requires_explicit_nc_policy(monkeypatch):
    from ocr_local.translation.models import TranslationRequest
    from ocr_local.translation.policy import TenantPolicy
    from ocr_local.translation.router import _rank_candidates

    engine_id = NLLBEngine.capability.id
    monkeypatch.setitem(ENGINE_REGISTRY, engine_id, NLLBEngine)
    req = TranslationRequest(src_lang="en", tgt_lang="fr", tenant_id="commercial")
    denied = TenantPolicy(
        tenant_id="commercial", allowed_engine_ids=[engine_id],
        allow_nllb_commercial=False,
    )
    permitted = TenantPolicy(
        tenant_id="commercial", allowed_engine_ids=[engine_id],
        allow_nllb_commercial=True,
    )
    assert _rank_candidates(req, denied) == []
    assert _rank_candidates(req, permitted) == [NLLBEngine]


