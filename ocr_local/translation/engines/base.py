"""Abstract base class for translation engine adapters.

Every concrete engine declares an immutable ``capability`` ClassVar of
type :class:`ocr_local.translation.models.EngineCapability` and
implements :meth:`TranslationEngine.translate_spans` plus
:meth:`TranslationEngine.model_provenance`.  LLM-backed engines should
mix in :class:`LLMEngineMixin` to pick up the deterministic decode
parameters and prompt-template hash helper.

Per Plan B Wave M2 PR B19 (E-B-008 / RED-07) every concrete engine's
``model_provenance()`` is validated through
:func:`ocr_local.translation.provenance.validate_engine_provenance`
when a registry-aware caller binds the engine.  The validator gates on
``pipeline_config.translation_enforce_provenance`` -- when True any
engine missing the SLSA / in-toto / CycloneDX SBOM triple is rejected
before it can be returned to the router.
"""

from __future__ import annotations

import importlib.metadata
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ocr_local.translation.models import EngineCapability, SpanTranslation
    from ocr_local.translation.provenance import ModelProvenance


def ct2_hypothesis_confidence(result: Any) -> float:
    """Map a CTranslate2 ``TranslationResult`` to confidence in ``[0.0, 1.0]``.

    CT2 exposes sequence log-probabilities on ``result.scores``. Confidence is
    derived from those scores only — never a hardcoded stub (e.g. 0.85).

    Missing scores → ``0.0`` (unknown / untrusted). Length-normalizes by
    hypothesis token count so longer sequences are not systematically lower.
    """
    scores = getattr(result, "scores", None)
    if not scores:
        return 0.0
    try:
        logp = float(scores[0])
    except (TypeError, ValueError, IndexError):
        return 0.0
    hyp = getattr(result, "hypotheses", None)
    try:
        n_tokens = max(len(hyp[0]) if hyp else 1, 1)
    except (TypeError, IndexError):
        n_tokens = 1
    mean_logp = logp / n_tokens
    # Geometric-mean token probability ≈ exp(mean logp); clamp for stability.
    conf = math.exp(max(min(mean_logp, 0.0), -20.0))
    return max(0.0, min(1.0, conf))


class TranslationEngine(ABC):
    """Abstract base for all translation engine adapters."""

    capability: ClassVar["EngineCapability"]
    # Adapters that cannot produce real inference must opt out so routers do
    # not select a scaffold and mislabel synthetic text as translation.
    production_ready: ClassVar[bool] = True

    @abstractmethod
    def translate_spans(
        self,
        spans: list[dict],
        src: str,
        tgt: str,
        glossary: Any | None = None,
        seed: int = 42,
        beam_size: int = 4,
    ) -> list["SpanTranslation"]:
        """Translate a list of span dicts.

        Each span has ``text``, ``bbox``, and ``span_id`` keys.  The
        adapter is responsible for preserving ``span_id`` on the
        returned :class:`SpanTranslation` instances so downstream
        merging by span identity continues to work.
        """

    @abstractmethod
    def model_provenance(self) -> dict:
        """Return provenance dict with ``weights_sha256``, ``license``, ``runtime_version``."""

    def runtime_info(self) -> dict:
        """Return runtime info -- default impl reports the ``ctranslate2`` version."""

        try:
            ver = importlib.metadata.version("ctranslate2")
        except importlib.metadata.PackageNotFoundError:
            ver = "not_installed"
        return {"runtime": "ctranslate2", "version": ver}

    def validated_provenance(
        self,
        *,
        enforce: bool | None = None,
    ) -> "ModelProvenance":
        """Return the engine's provenance after running B19 validation.

        Convenience wrapper -- delegates to
        :func:`ocr_local.translation.provenance.validate_engine_provenance`
        so engine consumers don't have to import the validator
        separately.  Pass ``enforce=True`` to force enforcement
        regardless of pipeline config (useful for tests).
        """

        from ocr_local.translation.provenance import validate_engine_provenance

        return validate_engine_provenance(self, enforce=enforce)


class LLMEngineMixin:
    """Mixin for LLM-backed engines -- adds decode params + prompt hash."""

    def llm_decode_params(self) -> dict:
        """Deterministic decode parameters used for reproducible translations."""

        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_new_tokens": 512,
        }

    def prompt_template_hash(self) -> str:
        """SHA-256 of the prompt template used for this engine."""

        import hashlib

        template = getattr(self, "_PROMPT_TEMPLATE", "")
        return hashlib.sha256(template.encode()).hexdigest()
