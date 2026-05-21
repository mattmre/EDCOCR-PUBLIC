"""Tests for the CTranslate2-backed engine adapters.

These tests are designed to run on the SDK CI lane *without*
``ctranslate2`` installed.  Init-time guards are exercised via
``sys.modules`` patching; capability/registry checks read the engine
class attributes directly without instantiating, so they don't require
the runtime.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from unittest import mock

import pytest

from ocr_local.translation.engines import ENGINE_REGISTRY
from ocr_local.translation.engines.local_ct2_madlad import MADLADEngine
from ocr_local.translation.engines.local_ct2_nllb import NLLBEngine
from ocr_local.translation.engines.local_ct2_opus import OpusMTEngine
from ocr_local.translation.models import SpanTranslation

_HAS_CT2 = importlib.util.find_spec("ctranslate2") is not None

# Importing the CT2 engine modules above triggers ``@register_engine``
# at import time.  When ``ctranslate2`` is missing, downstream tests
# (notably ``tests/test_translation_policy.py``) exercise the router
# and would try to instantiate these engines, tripping the init guard.
# Roll the registrations back immediately so the registry only contains
# engines we register on purpose -- the ``ensure_registered`` fixture
# below adds them back for the few tests that care.
_CT2_ENGINE_IDS = ("local_ct2_opus", "local_ct2_nllb", "local_ct2_madlad")
if not _HAS_CT2:
    for _engine_id in _CT2_ENGINE_IDS:
        ENGINE_REGISTRY.pop(_engine_id, None)


@pytest.fixture
def ensure_registered():
    """Make sure the three CT2 engines are present in the registry.

    The package's lazy registration only registers them when
    ``ctranslate2`` is installed, so on CI lanes without the runtime we
    register them by hand for the registry-membership tests, then
    unregister so we don't pollute downstream tests (the router would
    otherwise try to instantiate them and trip the init-time
    ``ctranslate2`` guard).
    """

    from ocr_local.translation.engines import register_engine

    added: list[str] = []
    for cls in (OpusMTEngine, NLLBEngine, MADLADEngine):
        if cls.capability.id not in ENGINE_REGISTRY:
            register_engine(cls)
            added.append(cls.capability.id)
    try:
        yield
    finally:
        for engine_id in added:
            ENGINE_REGISTRY.pop(engine_id, None)


# ---------------------------------------------------------------------------
# Registry membership
# ---------------------------------------------------------------------------


def test_opus_mt_in_registry_when_ct2_available(ensure_registered):
    assert "local_ct2_opus" in ENGINE_REGISTRY
    assert ENGINE_REGISTRY["local_ct2_opus"] is OpusMTEngine


def test_nllb_in_registry_when_ct2_available(ensure_registered):
    assert "local_ct2_nllb" in ENGINE_REGISTRY
    assert ENGINE_REGISTRY["local_ct2_nllb"] is NLLBEngine


def test_madlad_in_registry_when_ct2_available(ensure_registered):
    assert "local_ct2_madlad" in ENGINE_REGISTRY
    assert ENGINE_REGISTRY["local_ct2_madlad"] is MADLADEngine


# ---------------------------------------------------------------------------
# Init guards -- real model dirs raise RuntimeError when ctranslate2 is missing
# ---------------------------------------------------------------------------


def test_opus_init_raises_without_ct2():
    with mock.patch.dict(sys.modules, {"ctranslate2": None}):
        with pytest.raises(RuntimeError, match="ctranslate2"):
            OpusMTEngine(model_dir="models/opus")


def test_nllb_init_raises_without_ct2():
    with mock.patch.dict(sys.modules, {"ctranslate2": None}):
        with pytest.raises(RuntimeError, match="ctranslate2"):
            NLLBEngine(model_dir="models/nllb")


def test_madlad_init_raises_without_ct2():
    with mock.patch.dict(sys.modules, {"ctranslate2": None}):
        with pytest.raises(RuntimeError, match="ctranslate2"):
            MADLADEngine(model_dir="models/madlad")


# ---------------------------------------------------------------------------
# Capability/license checks (no instantiation needed)
# ---------------------------------------------------------------------------


def test_opus_license_is_cc_by():
    assert OpusMTEngine.capability.license == "CC-BY-4.0"


def test_nllb_license_is_nc():
    # Must contain "NC" (case-insensitive) so the router commercial
    # filter excludes this engine for commercial tenants.
    assert "NC" in NLLBEngine.capability.license.upper()
    assert NLLBEngine.capability.license == "CC-BY-NC-4.0"


def test_madlad_license_is_apache():
    assert MADLADEngine.capability.license == "Apache-2.0"


def test_all_ct2_engines_are_local():
    for cls in (OpusMTEngine, NLLBEngine, MADLADEngine):
        assert cls.capability.is_local is True
        assert cls.capability.is_cloud is False


def test_all_ct2_engines_retention_local_only():
    for cls in (OpusMTEngine, NLLBEngine, MADLADEngine):
        assert cls.capability.provider_retention_class == "local_only"


# ---------------------------------------------------------------------------
# Stub translation path (runs without ctranslate2 or model weights)
# ---------------------------------------------------------------------------


def test_opus_stub_translate_returns_spans():
    engine = OpusMTEngine()
    spans = [{"text": "hello", "span_id": "s0", "bbox": [0, 0, 10, 10]}]
    out = engine.translate_spans(spans, "en", "fr")
    assert isinstance(out, list)
    assert all(isinstance(o, SpanTranslation) for o in out)
    assert out[0].target_text == "[OPUS-MT stub] hello"


def test_nllb_stub_translate_returns_spans():
    engine = NLLBEngine()
    spans = [{"text": "world", "span_id": "s0"}]
    out = engine.translate_spans(spans, "en", "fr")
    assert out[0].target_text == "[NLLB-200 stub] world"


def test_madlad_stub_translate_returns_spans():
    engine = MADLADEngine()
    spans = [{"text": "foo", "span_id": "s0"}]
    out = engine.translate_spans(spans, "en", "fr")
    assert out[0].target_text == "[MADLAD-400 stub] foo"


def test_opus_stub_span_count_matches():
    engine = OpusMTEngine()
    spans = [{"text": f"line {i}", "span_id": f"s{i}"} for i in range(5)]
    out = engine.translate_spans(spans, "en", "fr")
    assert len(out) == len(spans)


def test_model_provenance_has_required_keys():
    engine = OpusMTEngine()
    prov = engine.model_provenance()
    for key in ("weights_sha256", "license", "runtime_version"):
        assert key in prov


def test_provenance_cached():
    engine = OpusMTEngine()
    prov1 = engine.model_provenance()
    prov2 = engine.model_provenance()
    assert prov1 is prov2


def test_opus_model_provenance_loads_provenance_json(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    payload = {
        "weights_sha256": "c" * 64,
        "license": "CC-BY-4.0",
        "runtime_version": "4.7.1",
        "slsa_provenance_uri": "https://models.example/slsa",
        "intoto_attestation_sha256": "a" * 64,
        "sbom_sha256": "b" * 64,
    }
    (model_dir / "provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    engine = OpusMTEngine()
    engine._model_dir = str(model_dir)

    assert engine.model_provenance() == payload


def test_stub_deterministic():
    engine = OpusMTEngine()
    spans = [{"text": "hello world", "span_id": "s0"}]
    out1 = engine.translate_spans(spans, "en", "fr", seed=42)
    out2 = engine.translate_spans(spans, "en", "fr", seed=42)
    assert out1[0].target_text == out2[0].target_text


def test_handles_handwriting_natively_false():
    for cls in (OpusMTEngine, NLLBEngine, MADLADEngine):
        assert cls.capability.handles_handwriting_natively is False


def test_ct2_engines_capability_id_matches_registry_key(ensure_registered):
    for cls in (OpusMTEngine, NLLBEngine, MADLADEngine):
        registered = ENGINE_REGISTRY[cls.capability.id]
        assert registered is cls


def test_opus_real_ct2_path_uses_translator_and_sentencepiece(monkeypatch, tmp_path):
    model_dir = tmp_path / "opus"
    model_dir.mkdir()
    (model_dir / "source.spm").write_text("fixture", encoding="utf-8")
    (model_dir / "target.spm").write_text("fixture", encoding="utf-8")

    class _Result:
        hypotheses = [["Bonjour", "▁monde"]]

    class _Translator:
        def __init__(self, model_path, device):
            assert model_path == str(model_dir)
            assert device == "cpu"

        def translate_batch(self, batch, **kwargs):
            assert batch == [["Hello", "▁world"]]
            assert kwargs["beam_size"] == 2
            return [_Result()]

    class _SentencePiece:
        def __init__(self, model_file):
            self.model_file = model_file

        def encode(self, text, out_type=str):
            assert text == "Hello world"
            return ["Hello", "▁world"]

        def decode(self, pieces):
            assert pieces == ["Bonjour", "▁monde"]
            return "Bonjour monde"

    fake_ct2 = mock.MagicMock(Translator=_Translator)
    fake_spm = mock.MagicMock(SentencePieceProcessor=_SentencePiece)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    monkeypatch.setitem(sys.modules, "sentencepiece", fake_spm)

    engine = OpusMTEngine(model_dir=str(model_dir))
    out = engine.translate_spans(
        [{"text": "Hello world", "span_id": "s0", "bbox": [0, 0, 1, 1]}],
        "en",
        "fr",
        beam_size=2,
    )

    assert out[0].target_text == "Bonjour monde"
    assert out[0].engine_id == "local_ct2_opus"
