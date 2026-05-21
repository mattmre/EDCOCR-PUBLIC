"""Translation enrichment package -- Plan B Wave M1.

Public surface for the translation pipeline foundation: the abstract
``TranslationEngine`` adapter contract, the JSON-friendly dataclasses
that describe span/page/document translations, and a lightweight engine
registry.  No engine implementations beyond the reference
``PassthroughEngine`` are wired here -- the registry guard in
``engines/__init__.py`` lazily imports the CTranslate2-backed engines
only when ``ctranslate2`` is installed.
"""

from __future__ import annotations

from ocr_local.translation.engines import (
    ENGINE_REGISTRY,
    get_engine,
    iter_engines,
    register_engine,
)
from ocr_local.translation.engines.base import LLMEngineMixin, TranslationEngine
from ocr_local.translation.metrics import (
    record_translation_chars,
    record_translation_duration,
    record_translation_tokens,
)
from ocr_local.translation.models import (
    DocumentTranslation,
    EngineCapability,
    PageTranslation,
    SpanTranslation,
    TranslationRequest,
)
from ocr_local.translation.sidecar import (
    SchemaValidationError,
    write_translation_json,
    write_translation_md,
)

__all__ = [
    "SpanTranslation",
    "PageTranslation",
    "DocumentTranslation",
    "TranslationRequest",
    "EngineCapability",
    "TranslationEngine",
    "LLMEngineMixin",
    "ENGINE_REGISTRY",
    "register_engine",
    "get_engine",
    "iter_engines",
    "SchemaValidationError",
    "write_translation_json",
    "write_translation_md",
    "record_translation_chars",
    "record_translation_tokens",
    "record_translation_duration",
]
