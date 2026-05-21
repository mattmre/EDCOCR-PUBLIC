"""Contract tests for ``ocr_local.translation`` engine adapters.

These tests deliberately use only the stdlib + pytest so they can run on
the SDK CI lane without ``ctranslate2`` installed.  Every concrete
engine in :data:`ENGINE_REGISTRY` is exercised against the same set of
contract assertions, so adding a new engine in a future PR
automatically inherits this coverage.
"""

from __future__ import annotations

import pytest

from ocr_local.translation import (
    ENGINE_REGISTRY,
    EngineCapability,
    SpanTranslation,
    TranslationEngine,
    get_engine,
    iter_engines,
    register_engine,
)
from ocr_local.translation.engines.passthrough import PassthroughEngine

VALID_QUALITY_CLASSES = {"draft", "standard", "legal"}
VALID_LATENCY_CLASSES = {"realtime", "standard", "bulk"}
VALID_RETENTION_CLASSES = {
    "local_only",
    "zero_retention_with_baa",
    "retention_enabled",
    "unknown",
}


def _all_engine_classes() -> list[tuple[str, type[TranslationEngine]]]:
    """Return ``(engine_id, engine_class)`` pairs from the live registry."""

    return list(iter_engines())


def test_passthrough_in_registry():
    """PassthroughEngine must always be present -- it has no optional deps."""

    assert "passthrough" in ENGINE_REGISTRY
    assert ENGINE_REGISTRY["passthrough"] is PassthroughEngine


def test_all_engines_have_capability():
    """Every registered engine class must declare a ``capability`` attr."""

    for engine_id, cls in _all_engine_classes():
        assert hasattr(cls, "capability"), f"{engine_id} missing capability"
        assert isinstance(cls.capability, EngineCapability)


def test_capability_id_matches_registry_key():
    """Registry key must match ``capability.id`` for every engine."""

    for engine_id, cls in _all_engine_classes():
        assert cls.capability.id == engine_id


def test_capability_has_license_spdx():
    """Every engine must declare a non-empty SPDX license string."""

    for engine_id, cls in _all_engine_classes():
        assert isinstance(cls.capability.license, str)
        assert cls.capability.license, f"{engine_id} has empty license"


def test_capability_provider_retention_class_valid():
    for _engine_id, cls in _all_engine_classes():
        assert cls.capability.provider_retention_class in VALID_RETENTION_CLASSES


def test_capability_quality_class_valid():
    for _engine_id, cls in _all_engine_classes():
        assert cls.capability.quality_class in VALID_QUALITY_CLASSES


def test_capability_latency_class_valid():
    for _engine_id, cls in _all_engine_classes():
        assert cls.capability.latency_class in VALID_LATENCY_CLASSES


def test_capability_is_local_xor_is_cloud_or_both():
    """At least one of ``is_local`` / ``is_cloud`` must be True."""

    for engine_id, cls in _all_engine_classes():
        assert cls.capability.is_local or cls.capability.is_cloud, (
            f"{engine_id} is neither local nor cloud"
        )


def test_translate_spans_returns_list():
    """Passthrough must return a list of ``SpanTranslation`` instances."""

    engine = PassthroughEngine()
    spans = [{"span_id": "s0", "text": "hello", "bbox": [0.0, 0.0, 10.0, 12.0]}]
    out = engine.translate_spans(spans, src="en", tgt="en")
    assert isinstance(out, list)
    assert all(isinstance(item, SpanTranslation) for item in out)


def test_translate_spans_preserves_span_count():
    engine = PassthroughEngine()
    spans = [
        {"span_id": "s0", "text": "alpha", "bbox": [0.0, 0.0, 10.0, 12.0]},
        {"span_id": "s1", "text": "beta", "bbox": [0.0, 12.0, 10.0, 24.0]},
        {"span_id": "s2", "text": "gamma", "bbox": [0.0, 24.0, 10.0, 36.0]},
    ]
    out = engine.translate_spans(spans, src="en", tgt="en")
    assert len(out) == len(spans)


def test_translate_spans_span_id_preserved():
    engine = PassthroughEngine()
    spans = [
        {"span_id": "alpha-1", "text": "a", "bbox": [0.0, 0.0, 1.0, 1.0]},
        {"span_id": "beta-2", "text": "b", "bbox": [0.0, 0.0, 1.0, 1.0]},
    ]
    out = engine.translate_spans(spans, src="en", tgt="en")
    assert [s.span_id for s in out] == ["alpha-1", "beta-2"]


def test_translate_spans_engine_id_set():
    engine = PassthroughEngine()
    spans = [{"span_id": "s0", "text": "x", "bbox": [0.0, 0.0, 1.0, 1.0]}]
    out = engine.translate_spans(spans, src="en", tgt="en")
    assert out[0].engine_id == engine.capability.id


def test_model_provenance_has_weights_sha256():
    engine = PassthroughEngine()
    prov = engine.model_provenance()
    assert "weights_sha256" in prov


def test_model_provenance_has_license():
    engine = PassthroughEngine()
    prov = engine.model_provenance()
    assert "license" in prov
    assert prov["license"]


def test_runtime_info_returns_dict():
    engine = PassthroughEngine()
    info = engine.runtime_info()
    assert isinstance(info, dict)
    assert info  # non-empty


def test_get_engine_returns_correct_class():
    assert get_engine("passthrough") is PassthroughEngine


def test_get_engine_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_engine("nonexistent-engine-id-zzz")


def test_register_engine_decorator():
    """Registering a fresh engine class must make it visible via the registry."""

    sentinel_id = "test-contract-sentinel"

    # Make sure the test is hermetic across reruns.
    ENGINE_REGISTRY.pop(sentinel_id, None)
    try:

        @register_engine
        class _SentinelEngine(TranslationEngine):
            capability = EngineCapability(
                id=sentinel_id,
                is_local=True,
                is_cloud=False,
                supports_pairs="any",
                quality_class="draft",
                latency_class="realtime",
                license="Apache-2.0",
                provider_retention_class="local_only",
                deployment_envs=["local"],
            )

            def translate_spans(self, spans, src, tgt, glossary=None, seed=42, beam_size=4):
                return []

            def model_provenance(self) -> dict:
                return {
                    "weights_sha256": "n/a",
                    "license": self.capability.license,
                    "runtime_version": "test",
                }

        assert sentinel_id in ENGINE_REGISTRY
        assert get_engine(sentinel_id) is _SentinelEngine
    finally:
        ENGINE_REGISTRY.pop(sentinel_id, None)
