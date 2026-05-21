"""Dataclasses describing the translation enrichment contract.

These types form the JSON-friendly data model emitted by the translation
pipeline (see Plan B Wave M1 in
``docs/planning/2026-04-24-translation-swarm/``).  They are deliberately
stdlib-only so the contract test in
``tests/test_translation_engine_contract.py`` can run on the SDK CI lane
without optional dependencies.

Field invariants worth highlighting:

* ``EngineCapability`` is ``frozen=True`` -- engines must declare a
  single immutable capability descriptor at class scope.
* ``DocumentTranslation.certified`` defaults to ``False`` and must
  **never** be defaulted to ``True``.  Certification can only be set by
  an explicit downstream attestation step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class EngineCapability:
    """Immutable descriptor of a translation engine's capabilities."""

    id: str
    is_local: bool
    is_cloud: bool
    # Either an explicit list of (src, tgt) BCP-47 pairs, or the literal
    # string "any" for engines that accept arbitrary language pairs.
    supports_pairs: list[tuple[str, str]] | str
    quality_class: Literal["draft", "standard", "legal"]
    latency_class: Literal["realtime", "standard", "bulk"]
    license: str  # SPDX identifier, e.g. "Apache-2.0"
    provider_retention_class: Literal[
        "local_only",
        "zero_retention_with_baa",
        "retention_enabled",
        "unknown",
    ]
    deployment_envs: list[str]  # e.g. ["local", "air_gapped", "cloud"]
    cost_per_1m_chars_usd: float | None = None
    cost_per_1m_tokens_usd: float | None = None
    # Plan B Q11 -- whether the engine can translate handwriting OCR
    # output natively without a separate cleanup pass.
    handles_handwriting_natively: bool = False


@dataclass
class SpanTranslation:
    """Translation for a single OCR line/span."""

    span_id: str
    source_text: str
    target_text: str
    source_bbox: list[float]          # primary bbox [x0, y0, x1, y1]
    source_bboxes: list[list[float]]  # polygon list for multi-line spans
    source_language: str               # BCP-47
    target_language: str               # BCP-47
    confidence: float
    quality_score: float | None
    engine_id: str
    glossary_hits: list[str] = field(default_factory=list)


@dataclass
class PageTranslation:
    """All span translations for a single page."""

    page_num: int
    spans: list[SpanTranslation] = field(default_factory=list)


@dataclass
class DocumentTranslation:
    """Top-level translation sidecar for a single document."""

    schema_version: str
    document_id: str
    source_file: str
    source_language: str
    target_language: str
    # NEVER default to True -- certification is an explicit downstream
    # attestation, not a property of the raw translation output.
    certified: bool = False
    engine: dict = field(default_factory=dict)
    glossary: dict | None = None
    quality: dict = field(default_factory=dict)
    pages: list[PageTranslation] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    custody: dict = field(default_factory=dict)
    processing: dict = field(default_factory=dict)


@dataclass
class TranslationRequest:
    """Caller-side request describing the desired translation behaviour."""

    src_lang: str
    tgt_lang: str
    quality: str = "standard"
    latency: str = "standard"
    privilege_flag: bool = False
    tenant_id: str = "default"
