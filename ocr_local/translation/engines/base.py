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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ocr_local.translation.models import EngineCapability, SpanTranslation
    from ocr_local.translation.provenance import ModelProvenance


class TranslationEngine(ABC):
    """Abstract base for all translation engine adapters."""

    capability: ClassVar["EngineCapability"]

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
