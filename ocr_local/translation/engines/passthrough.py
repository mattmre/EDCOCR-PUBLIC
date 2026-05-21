"""PassthroughEngine -- same-language no-op reference adapter.

Useful for two purposes:

* It gives the contract test a concrete, dependency-free engine to
  exercise without requiring ``ctranslate2`` or any LLM weights.
* It serves as the canonical "do nothing" adapter for cases where the
  caller wants the translation enrichment shape without actually
  changing the text (for example when src == tgt).
"""

from __future__ import annotations

from typing import Any

from ocr_local.translation.engines import register_engine
from ocr_local.translation.engines.base import TranslationEngine
from ocr_local.translation.models import EngineCapability, SpanTranslation


@register_engine
class PassthroughEngine(TranslationEngine):
    """Same-language no-op adapter -- emits ``SpanTranslation`` records unchanged."""

    capability = EngineCapability(
        id="passthrough",
        is_local=True,
        is_cloud=False,
        supports_pairs="any",
        quality_class="draft",
        latency_class="realtime",
        license="Apache-2.0",
        provider_retention_class="local_only",
        deployment_envs=["local", "air_gapped", "cloud"],
        cost_per_1m_chars_usd=0.0,
        cost_per_1m_tokens_usd=0.0,
    )

    def translate_spans(
        self,
        spans: list[dict],
        src: str,
        tgt: str,
        glossary: Any | None = None,
        seed: int = 42,
        beam_size: int = 4,
    ) -> list[SpanTranslation]:
        results: list[SpanTranslation] = []
        for i, span in enumerate(spans):
            bbox = span.get("bbox", [0.0, 0.0, 100.0, 12.0])
            results.append(
                SpanTranslation(
                    span_id=span.get("span_id", f"s{i}"),
                    source_text=span["text"],
                    target_text=span["text"],  # passthrough -- no change
                    source_bbox=bbox,
                    source_bboxes=[bbox],
                    source_language=src,
                    target_language=tgt,
                    confidence=1.0,
                    quality_score=None,
                    engine_id=self.capability.id,
                )
            )
        return results

    def model_provenance(self) -> dict:
        return {
            "weights_sha256": "n/a",
            "license": self.capability.license,
            "runtime_version": "builtin",
        }
