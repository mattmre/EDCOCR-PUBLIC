"""CTranslate2-backed NLLB-200 engine adapter.

NLLB-200 (No Language Left Behind) is Meta's 200-language multilingual
translation model released under CC-BY-NC-4.0.  The ``NC`` (non-
commercial) clause is critical: the router relies on the SPDX string
``CC-BY-NC-4.0`` to filter this engine out of commercial-tenant routing
decisions.  Do not change this license string without coordinating an
update to the router's commercial filter.

Real CT2 inference requires a converted model directory with SentencePiece
tokenizers.  Empty ``model_dir`` fails closed (no synthetic translations).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ocr_local.translation.engines import register_engine
from ocr_local.translation.engines.base import (
    TranslationEngine,
    ct2_hypothesis_confidence,
)
from ocr_local.translation.models import EngineCapability, SpanTranslation

# Minimal ISO-639-1 / common aliases → Flores-200 style NLLB codes.
_NLLB_LANG: dict[str, str] = {
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "fr": "fra_Latn",
    "fra": "fra_Latn",
    "es": "spa_Latn",
    "spa": "spa_Latn",
    "de": "deu_Latn",
    "deu": "deu_Latn",
    "pt": "por_Latn",
    "por": "por_Latn",
    "it": "ita_Latn",
    "ita": "ita_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
}


def _nllb_code(lang: str) -> str:
    key = (lang or "").strip().lower().replace("-", "_")
    if "_" in key and len(key) > 3:
        # already looks like eng_Latn
        return lang if lang[0].islower() else lang
    return _NLLB_LANG.get(key, lang if lang else "eng_Latn")


def _load_ct2_runtime():
    try:
        import ctranslate2
    except ImportError as exc:
        raise RuntimeError(
            "ctranslate2 not installed. "
            "Install with: pip install -r requirements-translation.txt"
        ) from exc
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "sentencepiece not installed. "
            "Install with: pip install -r requirements-translation.txt"
        ) from exc
    return ctranslate2, spm


def _resolve_spm(model_path: Path) -> Path:
    """Locate a SentencePiece model file in an NLLB CT2 export."""
    candidates = [
        model_path / "sentencepiece.bpe.model",
        model_path / "sentencepiece.model",
        model_path / "source.spm",
        model_path / "spm.model",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise RuntimeError(
        "NLLB CT2 model_dir must contain a SentencePiece model "
        "(sentencepiece.bpe.model / sentencepiece.model / source.spm)"
    )


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
    # Real CT2 path implemented; still requires model_dir at translate time.
    production_ready = True

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
            ctranslate2, spm = _load_ct2_runtime()
            model_path = Path(model_dir)
            spm_path = _resolve_spm(model_path)
            self._translator = ctranslate2.Translator(str(model_path), device="cpu")
            self._tokenizer = spm.SentencePieceProcessor(model_file=str(spm_path))
        else:
            self._translator = None
            self._tokenizer = None
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
            raise RuntimeError(
                "NLLB model_dir is required; synthetic translation is disabled"
            )

        assert self._translator is not None
        assert self._tokenizer is not None
        src_code = _nllb_code(src)
        tgt_code = _nllb_code(tgt)
        tokenized = []
        for s in spans:
            pieces = self._tokenizer.encode(s["text"], out_type=str)
            # NLLB expects source language token then body; target via target_prefix.
            tokenized.append([src_code, *pieces])
        target_prefix = [[tgt_code] for _ in spans]
        results = self._translator.translate_batch(
            tokenized,
            target_prefix=target_prefix,
            beam_size=beam_size,
            max_decoding_length=256,
            replace_unknowns=True,
        )
        out: list[SpanTranslation] = []
        for i, (s, result) in enumerate(zip(spans, results)):
            hyp = list(result.hypotheses[0])
            # Strip leading target language token if present.
            if hyp and hyp[0] == tgt_code:
                hyp = hyp[1:]
            conf = ct2_hypothesis_confidence(result)
            out.append(
                SpanTranslation(
                    span_id=s.get("span_id", f"s{i}"),
                    source_text=s["text"],
                    target_text=self._tokenizer.decode(hyp),
                    source_bbox=s.get("bbox", [0.0, 0.0, 100.0, 12.0]),
                    source_bboxes=[s.get("bbox", [0.0, 0.0, 100.0, 12.0])],
                    source_language=src,
                    target_language=tgt,
                    confidence=conf,
                    quality_score=conf,
                    engine_id=self.capability.id,
                )
            )
        return out

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
