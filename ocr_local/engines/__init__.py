"""Pluggable OCR engine framework (Phase 0 foundation).

Additive only -- nothing in this package is wired into the production pipeline
yet (see ``docs/architecture/pluggable-engines-enrichment-program-20260619.md``).
The forensic CTC cascade in ``ocr_gpu_async.py`` is unchanged; an unconfigured
pipeline behaves byte-identically to today.

Public surface:
- :class:`OCREngine` / :data:`OcrLine` -- the engine adapter ABC + line contract
- :class:`EngineCapability` -- per-engine license/lane/routing metadata
- :class:`EnginePolicy` + :func:`filter_engines` -- governance / lane filtering
- :class:`EngineRegistry` / :data:`DEFAULT_REGISTRY` -- registration + discovery
"""
from __future__ import annotations

from ocr_local.engines.base import OCREngine, OcrLine
from ocr_local.engines.capability import CostClass, EngineCapability, Lane, LicenseClass
from ocr_local.engines.governance import (
    ADMITTED,
    EngineDecision,
    EnginePolicy,
    compute_policy_hash,
    evaluate_engine,
    filter_engines,
    rank_engines,
)
from ocr_local.engines.registry import (
    DEFAULT_REGISTRY,
    EngineFactory,
    EngineRegistry,
    get_default_registry,
)

__all__ = [
    "OCREngine",
    "OcrLine",
    "EngineCapability",
    "Lane",
    "LicenseClass",
    "CostClass",
    "EnginePolicy",
    "EngineDecision",
    "evaluate_engine",
    "filter_engines",
    "rank_engines",
    "compute_policy_hash",
    "ADMITTED",
    "EngineRegistry",
    "EngineFactory",
    "DEFAULT_REGISTRY",
    "get_default_registry",
]
