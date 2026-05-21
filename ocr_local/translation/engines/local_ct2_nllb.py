"""CTranslate2-backed NLLB-200 engine adapter.

NLLB-200 (No Language Left Behind) is Meta's 200-language multilingual
translation model released under CC-BY-NC-4.0.  The ``NC`` (non-
commercial) clause is critical: the router relies on the SPDX string
``CC-BY-NC-4.0`` to filter this engine out of commercial-tenant routing
decisions.  Do not change this license string without coordinating an
update to the router's commercial filter.
"""

from __future__ import annotations

import os
from typing import Any

from ocr_local.translation.engines import register_engine
from ocr_local.translation.engines.base import TranslationEngine
from ocr_local.translation.models import EngineCapability, SpanTranslation


@register_engine
class NLLBEngine(TranslationEngine):
    """NLLB-200 adapter -- CC-BY-NC-4.0, local-only, standard quality."""

    capability = EngineCapability(
        id="local_ct2_nllb",
        is_local=True,
        is_cloud=False,
        supports_pairs="any",
        quality_class="standard",
        latency_class="standard",
        license="CC-BY-NC-4.0",
        provider_retention_class="local_only",
        deployment_envs=["local", "air_gapped"],
        cost_per_1m_chars_usd=0.0,
        cost_per_1m_tokens_usd=0.0,
        handles_handwriting_natively=False,
    )

    def __init__(
        self,
        model_dir: str = "",
        *,
        model_id: str = "",
        tenant_policy: Any | None = None,
    ) -> None:
        if model_id and not model_dir:
            from ocr_local.translation.cache import get_translation_model_path

            try:
                from pipeline_config import get_config  # type: ignore[import-not-found]

                cfg = get_config()
                airgapped = bool(getattr(cfg, "translation_airgapped", False))
            except (ImportError, AttributeError, RuntimeError):
                airgapped = False
            resolved = get_translation_model_path(
                self.capability.id,
                model_id,
                allow_download=not airgapped,
                tenant_policy=tenant_policy,
            )
            model_dir = str(resolved)
        if model_dir:
            try:
                import ctranslate2  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "ctranslate2 not installed. "
                    "Install with: pip install -r requirements-translation.txt"
                ) from exc
        self._model_dir = model_dir
        self._model_id = model_id
        self._provenance: dict | None = None

    def translate_spans(
        self,
        spans: list[dict],
        src: str,
        tgt: str,
        glossary: Any | None = None,
        seed: int = 42,
        beam_size: int = 4,
    ) -> list[SpanTranslation]:
        if not self._model_dir:
            return [
                SpanTranslation(
                    span_id=s.get("span_id", f"s{i}"),
                    source_text=s["text"],
                    target_text=f"[NLLB-200 stub] {s['text']}",
                    source_bbox=s.get("bbox", [0.0, 0.0, 100.0, 12.0]),
                    source_bboxes=[s.get("bbox", [0.0, 0.0, 100.0, 12.0])],
                    source_language=src,
                    target_language=tgt,
                    confidence=0.85,
                    quality_score=None,
                    engine_id=self.capability.id,
                )
                for i, s in enumerate(spans)
            ]
        raise NotImplementedError(
            "Real CT2 model inference requires model_dir to be set"
        )

    def model_provenance(self) -> dict:
        if self._provenance is not None:
            return self._provenance
        sha = "not_loaded"
        if self._model_dir:
            sha_file = os.path.join(self._model_dir, "MODEL_SHA256")
            if os.path.exists(sha_file):
                with open(sha_file) as f:
                    sha = f.read().strip()
        self._provenance = {
            "weights_sha256": sha,
            "license": self.capability.license,
            "runtime_version": self.runtime_info().get("version", "unknown"),
        }
        return self._provenance
