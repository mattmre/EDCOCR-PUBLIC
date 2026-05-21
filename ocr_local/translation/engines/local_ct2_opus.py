"""CTranslate2-backed OPUS-MT engine adapter.

OPUS-MT is the Helsinki-NLP family of bilingual translation models released
under CC-BY-4.0.  Empty ``model_dir`` keeps the deterministic local stub used
by SDK lanes.  A non-empty ``model_dir`` now enables real CTranslate2 inference
with SentencePiece tokenizers when the optional runtime and model files are
available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ocr_local.translation.engines import register_engine
from ocr_local.translation.engines.base import TranslationEngine
from ocr_local.translation.models import EngineCapability, SpanTranslation


@register_engine
class OpusMTEngine(TranslationEngine):
    """OPUS-MT adapter -- CC-BY-4.0, local-only, standard quality."""

    capability = EngineCapability(
        id="local_ct2_opus",
        is_local=True,
        is_cloud=False,
        supports_pairs="any",
        quality_class="standard",
        latency_class="standard",
        license="CC-BY-4.0",
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
        # When ``model_id`` is supplied we resolve through the Wave M2
        # cache module so atomic download / integrity / NC-license
        # filtering all fire automatically.  Callers that already have
        # a resolved path (legacy / tests) continue to pass
        # ``model_dir`` directly.
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
                import ctranslate2
                import sentencepiece as spm
            except ImportError as exc:
                raise RuntimeError(
                    "ctranslate2 not installed. "
                    "Install with: pip install -r requirements-translation.txt"
                ) from exc
            model_path = Path(model_dir)
            source_spm = model_path / "source.spm"
            target_spm = model_path / "target.spm"
            if not source_spm.is_file() or not target_spm.is_file():
                raise RuntimeError(
                    "OPUS-MT CT2 model_dir must contain source.spm and target.spm"
                )
            self._translator = ctranslate2.Translator(str(model_path), device="cpu")
            self._source_tokenizer = spm.SentencePieceProcessor(model_file=str(source_spm))
            self._target_tokenizer = spm.SentencePieceProcessor(model_file=str(target_spm))
        else:
            self._translator = None
            self._source_tokenizer = None
            self._target_tokenizer = None
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
                    target_text=f"[OPUS-MT stub] {s['text']}",
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

        assert self._translator is not None
        assert self._source_tokenizer is not None
        assert self._target_tokenizer is not None
        tokenized = [
            self._source_tokenizer.encode(s["text"], out_type=str)
            for s in spans
        ]
        results = self._translator.translate_batch(
            tokenized,
            beam_size=beam_size,
            max_decoding_length=256,
            replace_unknowns=True,
        )
        translated_texts = [
            self._target_tokenizer.decode(result.hypotheses[0])
            for result in results
        ]
        return [
            SpanTranslation(
                span_id=s.get("span_id", f"s{i}"),
                source_text=s["text"],
                target_text=translated_texts[i],
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

    def model_provenance(self) -> dict:
        if self._provenance is not None:
            return self._provenance
        sha = "not_loaded"
        if self._model_dir:
            provenance_file = os.path.join(self._model_dir, "provenance.json")
            if os.path.exists(provenance_file):
                import json

                with open(provenance_file, encoding="utf-8") as f:
                    self._provenance = json.load(f)
                return self._provenance
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
