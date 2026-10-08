"""Per-tenant translation glossary (Plan B Wave M2).

Glossary entries are tenant-scoped term overrides applied to translation
output before sidecar serialization.  Literal entries do exact substring
replacement (case-sensitive or insensitive); regex entries use the bounded
third-party ``regex`` engine with compile-time validation, a match timeout,
and a runtime char-budget guard.

Pure module: no Django imports at top level.  ``load_tenant_glossary``
performs a lazy Django import inside the function body so this module
remains importable from non-Django contexts (the standalone OCR
pipeline, tests without DJANGO_SETTINGS_MODULE).

Custody emission: callers wrap span-level translation and emit
``GLOSSARY_APPLIED`` once per span when ``apply_glossary`` returns at
least one hit.  See ``ocr_local.translation.custody_adapter.emit_glossary_applied``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Iterable

try:
    import regex as _regex  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on the runtime image
    _regex = None

if TYPE_CHECKING:
    from ocr_local.features.custody import CustodyChain

__all__ = [
    "GlossaryEntry",
    "GlossaryHit",
    "load_tenant_glossary",
    "apply_glossary",
    "emit_glossary_applied_for_span",
    "compute_glossary_hash",
    "to_ct2_prefix_bias",
    "to_ct2_forbid_tokens",
    "to_llm_prompt_block",
]

logger = logging.getLogger(__name__)

# Cap input length per regex apply call to prevent ReDoS-style abuse.
# 200K chars is well above any realistic OCR span (~1-2K chars).
_REGEX_INPUT_CHAR_LIMIT = 200_000
# ``regex`` enforces this timeout inside its matching engine, including
# catastrophic-backtracking paths that cannot be interrupted by stdlib re.
_REGEX_TIMEOUT_SECONDS = 0.1
# Bound the complete regex glossary pass as well as each individual engine
# call.  Otherwise 500 entries could each consume the per-entry timeout.
_REGEX_APPLY_BUDGET_SECONDS = 1.0
_MAX_GLOSSARY_ENTRIES = 10_000
_MAX_GLOSSARY_ENTRIES_PER_APPLY = 500
_MAX_GLOSSARY_HITS_PER_ENTRY = 10_000
_MAX_GLOSSARY_HITS_PER_APPLY = 20_000
_GLOSSARY_OUTPUT_CHAR_LIMIT = 1_000_000


@dataclass(frozen=True)
class GlossaryEntry:
    """In-memory representation of a glossary entry.

    Mirrors the Django ``jobs.models.GlossaryEntry`` row but does not
    require Django to construct.  ``term_id`` is the row primary key
    when loaded from the DB; pure-test callers may use any string.
    """

    term_id: str
    tenant_id: str
    source_term: str
    target_term: str
    source_lang: str
    target_lang: str
    case_sensitive: bool = False
    is_regex: bool = False
    priority: int = 100


@dataclass(frozen=True)
class GlossaryHit:
    """Record of a single glossary application within a span."""

    term_id: str
    source_term: str
    target_term: str
    position_start: int
    position_end: int
    span_index: int = 0


def load_tenant_glossary(
    tenant_id: str | None,
    source_lang: str,
    target_lang: str,
) -> list[GlossaryEntry]:
    """Load tenant glossary entries for a language pair.

    Returns entries ordered by ``priority`` ascending, then by id.  When
    ``tenant_id`` is ``None`` (single-tenant or anonymous mode) the
    function returns an empty list -- glossary scoping is always
    explicit.

    Performs a lazy Django import so the module is importable in
    non-Django contexts.  When Django (or the model row) is unavailable
    the function returns an empty list and logs at DEBUG.
    """
    if tenant_id is None:
        return []

    try:
        # Lazy import: must succeed only when caller has Django configured.
        from jobs.models import (
            GlossaryEntry as DjangoGlossaryEntry,  # type: ignore[import-not-found]
        )
    except Exception as exc:  # pragma: no cover - non-Django fallback
        logger.debug(
            "load_tenant_glossary: Django/jobs.models not importable (%s)",
            exc,
        )
        return []

    queryset = DjangoGlossaryEntry.objects.filter(
        tenant_id=tenant_id,
        source_lang=source_lang,
        target_lang=target_lang,
    ).order_by("priority", "id")

    rows = list(queryset[: _MAX_GLOSSARY_ENTRIES + 1])
    if len(rows) > _MAX_GLOSSARY_ENTRIES:
        raise RuntimeError(
            "tenant glossary exceeds the configured safety limit; "
            "reduce entries before translation"
        )

    return [
        GlossaryEntry(
            term_id=str(row.pk),
            tenant_id=row.tenant_id,
            source_term=row.source_term,
            target_term=row.target_term,
            source_lang=row.source_lang,
            target_lang=row.target_lang,
            case_sensitive=bool(row.case_sensitive),
            is_regex=bool(row.is_regex),
            priority=int(row.priority),
        )
        for row in rows
    ]


def _literal_apply(
    text: str,
    entry: GlossaryEntry,
    span_index: int,
) -> tuple[str, list[GlossaryHit]]:
    """Apply a literal glossary entry; return (new_text, hits).

    Hits' ``position_start``/``position_end`` are recorded against the
    pre-mutation indices (before any earlier hits in the same call were
    spliced in).  Callers that need stable post-mutation offsets should
    not assume they are absolute.
    """
    hits: list[GlossaryHit] = []
    flags = 0 if entry.case_sensitive else re.IGNORECASE
    needle = re.escape(entry.source_term)

    if not needle:
        return text, hits

    out_parts: list[str] = []
    cursor = 0
    output_chars = 0
    for match in re.finditer(needle, text, flags=flags):
        start, end = match.start(), match.end()
        out_parts.append(text[cursor:start])
        out_parts.append(entry.target_term)
        output_chars += (start - cursor) + len(entry.target_term)
        if (
            len(hits) >= _MAX_GLOSSARY_HITS_PER_ENTRY
            or output_chars + (len(text) - end) > _GLOSSARY_OUTPUT_CHAR_LIMIT
        ):
            logger.warning(
                "Skipping glossary entry %s: hit/output safety limit exceeded",
                entry.term_id,
            )
            return text, []
        hits.append(
            GlossaryHit(
                term_id=entry.term_id,
                source_term=entry.source_term,
                target_term=entry.target_term,
                position_start=start,
                position_end=end,
                span_index=span_index,
            )
        )
        cursor = end
    out_parts.append(text[cursor:])
    output_chars += len(text) - cursor
    if output_chars > _GLOSSARY_OUTPUT_CHAR_LIMIT:
        logger.warning(
            "Skipping glossary entry %s: output exceeds %d chars",
            entry.term_id,
            _GLOSSARY_OUTPUT_CHAR_LIMIT,
        )
        return text, []
    return "".join(out_parts), hits


def _regex_apply(
    text: str,
    entry: GlossaryEntry,
    span_index: int,
    *,
    deadline: float | None = None,
) -> tuple[str, list[GlossaryHit]]:
    """Apply a regex glossary entry; return (new_text, hits).

    Regex matching is delegated to the third-party ``regex`` engine, which
    supports a hard timeout for each matching operation.  The entry is
    fail-closed when the bounded engine is unavailable, the pattern fails to
    compile, the input exceeds ``_REGEX_INPUT_CHAR_LIMIT``, or matching times
    out.  In all of those cases the original text and no hits are returned;
    a timeout can therefore never expose a partial replacement.
    """
    if _regex is None:
        logger.warning(
            "Skipping regex glossary entry %s: bounded regex engine unavailable",
            entry.term_id,
        )
        return text, []

    if len(text) > _REGEX_INPUT_CHAR_LIMIT:
        logger.warning(
            "Skipping regex glossary entry %s: input exceeds %d chars",
            entry.term_id,
            _REGEX_INPUT_CHAR_LIMIT,
        )
        return text, []

    flags = 0 if entry.case_sensitive else _regex.IGNORECASE
    try:
        compiled = _regex.compile(entry.source_term, flags=flags)
    except _regex.error as exc:
        logger.warning(
            "Skipping invalid regex glossary entry %s: %s",
            entry.term_id,
            exc,
        )
        return text, []

    hits: list[GlossaryHit] = []
    out_parts: list[str] = []
    cursor = 0
    output_chars = 0
    try:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "Skipping regex glossary entry %s: aggregate match budget exhausted",
                    entry.term_id,
                )
                return text, []
            match_timeout = min(_REGEX_TIMEOUT_SECONDS, remaining)
        else:
            match_timeout = _REGEX_TIMEOUT_SECONDS
        for match in compiled.finditer(text, timeout=match_timeout):
            start, end = match.start(), match.end()
            # Allow back-references via match.expand for parity with re.sub.
            try:
                replaced = match.expand(entry.target_term)
            except (_regex.error, IndexError):
                replaced = entry.target_term
            out_parts.append(text[cursor:start])
            out_parts.append(replaced)
            output_chars += (start - cursor) + len(replaced)
            if (
                len(hits) >= _MAX_GLOSSARY_HITS_PER_ENTRY
                or output_chars + (len(text) - end) > _GLOSSARY_OUTPUT_CHAR_LIMIT
            ):
                logger.warning(
                    "Skipping regex glossary entry %s: hit/output safety limit exceeded",
                    entry.term_id,
                )
                return text, []
            hits.append(
                GlossaryHit(
                    term_id=entry.term_id,
                    source_term=entry.source_term,
                    target_term=replaced,
                    position_start=start,
                    position_end=end,
                    span_index=span_index,
                )
            )
            cursor = end
    except TimeoutError:
        logger.warning(
            "Skipping regex glossary entry %s: match exceeded %.3f seconds",
            entry.term_id,
            _REGEX_TIMEOUT_SECONDS,
        )
        return text, []
    out_parts.append(text[cursor:])
    output_chars += len(text) - cursor
    if output_chars > _GLOSSARY_OUTPUT_CHAR_LIMIT:
        logger.warning(
            "Skipping regex glossary entry %s: output exceeds %d chars",
            entry.term_id,
            _GLOSSARY_OUTPUT_CHAR_LIMIT,
        )
        return text, []
    return "".join(out_parts), hits


def apply_glossary(
    text: str,
    entries: list[GlossaryEntry],
    *,
    span_index: int = 0,
) -> tuple[str, list[GlossaryHit]]:
    """Apply ``entries`` to ``text`` in priority order.

    Returns a tuple of ``(modified_text, hits)``.  When no entries match,
    ``text`` is returned unchanged and ``hits`` is an empty list.

    Entries are applied in input order (callers should pre-sort by
    priority via ``load_tenant_glossary``).  Each entry runs against the
    *current* text -- chained replacements within a single span are
    supported.

    Regex entries that fail to compile are skipped with a warning; they
    do not raise.
    """
    if not entries or not text:
        return text, []

    if len(entries) > _MAX_GLOSSARY_ENTRIES_PER_APPLY:
        raise RuntimeError(
            "glossary application exceeds the per-span safety limit; "
            "narrow the language-pair glossary"
        )

    original_text = text
    current = text
    all_hits: list[GlossaryHit] = []
    regex_deadline = time.monotonic() + _REGEX_APPLY_BUDGET_SECONDS
    for entry in entries:
        if entry.is_regex:
            current, hits = _regex_apply(
                current,
                entry,
                span_index,
                deadline=regex_deadline,
            )
        else:
            current, hits = _literal_apply(current, entry, span_index)
        if len(all_hits) + len(hits) > _MAX_GLOSSARY_HITS_PER_APPLY:
            logger.warning(
                "Reverting glossary application: aggregate hit limit exceeded"
            )
            return original_text, []
        all_hits.extend(hits)
    return current, all_hits


def emit_glossary_applied_for_span(
    chain: "CustodyChain",
    *,
    glossary_id: str,
    glossary_hash: str,
    hits: list[GlossaryHit],
    span_id: str | None = None,
    **extra,
) -> None:
    """Emit a single ``GLOSSARY_APPLIED`` event for a span with hits.

    Helper around ``custody_adapter.emit_glossary_applied`` that bundles
    the per-span hit payload (term ids, positions) so the audit trail
    captures exactly which terms were applied to a single span.

    No-op when ``hits`` is empty -- callers should not invoke this for
    spans that produced no hits.
    """
    if not hits:
        return

    from ocr_local.translation.custody_adapter import emit_glossary_applied

    emit_glossary_applied(
        chain,
        glossary_id=glossary_id,
        glossary_hash=glossary_hash,
        hit_count=len(hits),
        span_id=span_id,
        glossary_hits=[asdict(h) for h in hits],
        **extra,
    )


# Re-export at module level for convenience -- some callers prefer to
# bind the dataclass-default-factory at field-init time.
_ = field  # silence unused-import lint when we add fields later


# ---------------------------------------------------------------------------
# Constrained-decode hooks (Plan B Wave M2 PR B16)
# ---------------------------------------------------------------------------
#
# These helpers convert a glossary entry list into the per-engine input
# shapes the constrained-decoding pipeline expects.  Each helper is pure
# (no Django, no I/O) so it stays callable from any layer -- the
# router, a worker thread, or a unit test.
#
# Three constrained-decode modes are supported per the
# ``TRANSLATION_APPLIED`` custody event payload (E-B-013):
#
#   * ``prefer``         -- CT2 prefix-bias dict (soft preference)
#   * ``force``          -- CT2 forbid-tokens list (any miss is rejected)
#   * ``reject-on-miss`` -- same as force, plus router checks hit count
#   * ``llm-prompt-block`` -- text injected into the LLM-MT system prompt


def compute_glossary_hash(entries: list[GlossaryEntry]) -> str:
    """Return a deterministic SHA-256 hex digest over ``entries``.

    The digest is pinned into ``TRANSLATION_APPLIED`` custody events as
    ``glossary_hash`` so the audit trail can prove which glossary was
    applied to a given translation.  The hash input is a canonical JSON
    encoding of every entry's content fields (term identity, source/
    target term, case sensitivity, regex flag, priority) sorted by
    ``priority`` then ``term_id`` so two glossaries with identical
    content but different load order produce the same digest.

    The ``tenant_id`` and language fields are intentionally excluded so
    the same lexicon imported into two tenants gets the same hash --
    matching how external glossary registries identify content.
    """

    canonical = [
        {
            "term_id": e.term_id,
            "source_term": e.source_term,
            "target_term": e.target_term,
            "case_sensitive": bool(e.case_sensitive),
            "is_regex": bool(e.is_regex),
            "priority": int(e.priority),
        }
        for e in sorted(entries, key=lambda x: (x.priority, x.term_id))
    ]
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_ct2_prefix_bias(
    entries: Iterable[GlossaryEntry],
    *,
    bias_strength: float = 2.0,
) -> dict[str, float]:
    """Translate ``entries`` into a CT2 prefix-bias dict.

    CTranslate2's translator accepts a per-string score offset which
    biases beam search toward emitting that string.  This helper takes
    each non-regex entry's ``target_term`` and emits a mapping
    ``{target_term: bias_strength}`` -- callers feed this into
    ``Translator.translate_batch(prefix_bias=...)``.

    Regex entries are skipped (they don't have a fixed target string a
    decoder can bias toward); callers that need regex-driven biasing
    must precompute the post-substitution target text via
    :func:`apply_glossary` first.

    Empty target terms are skipped so we never bias the decoder toward
    emitting nothing.  Duplicates are merged with the maximum bias.
    """

    result: dict[str, float] = {}
    for entry in entries:
        if entry.is_regex:
            continue
        target = entry.target_term
        if not target:
            continue
        existing = result.get(target)
        if existing is None or bias_strength > existing:
            result[target] = bias_strength
    return result


def to_ct2_forbid_tokens(
    entries: Iterable[GlossaryEntry],
    *,
    block_source_terms: bool = True,
) -> list[str]:
    """Translate ``entries`` into a CT2 forbid-tokens list.

    Used by the ``reject-on-miss`` constrained-decode mode: when the
    router determines that every entry's target_term *must* appear, it
    forbids the engine from emitting the *source* terms (so a model
    that fails to translate the term forces a beam re-roll instead of
    leaking the source-language word into the output).

    When ``block_source_terms=False`` only target-language stop tokens
    that are intentionally pinned via the (optional) glossary
    ``notes`` field starting with ``forbid:`` are returned.  This
    matches the LLM-MT path which usually wants to forbid translation
    artefacts ("[UNK]", "<unk>", etc.) rather than source-language
    fragments.

    Regex entries are skipped (no fixed token to forbid).  Returned
    list is deduplicated, sorted for determinism.
    """

    blocked: set[str] = set()
    for entry in entries:
        if entry.is_regex:
            continue
        if block_source_terms and entry.source_term:
            blocked.add(entry.source_term)
        # Optional inline forbid directive in entry-style API:
        notes = getattr(entry, "notes", None)
        if isinstance(notes, str) and notes.startswith("forbid:"):
            for tok in notes[len("forbid:") :].split(","):
                tok = tok.strip()
                if tok:
                    blocked.add(tok)
    return sorted(blocked)


def to_llm_prompt_block(
    entries: Iterable[GlossaryEntry],
    *,
    max_entries: int = 200,
    header: str = "Use the following terminology when translating:",
) -> str:
    """Render glossary as a prompt-injection block for LLM-MT engines.

    Returns a plaintext block of the form::

        Use the following terminology when translating:
        - "Party" -> "Partie"
        - "Defendant" -> "Defendeur"

    Caller injects the result into the system prompt of a Qwen / Gemini
    / Claude MT call.  Regex entries are emitted verbatim with a
    ``(regex)`` annotation so the model treats them as patterns.

    Long glossaries are truncated at ``max_entries`` to bound prompt
    size; truncation is signalled by appending ``... (N more)`` so the
    audit trail shows we deliberately dropped entries instead of
    silently swallowing them.

    Returns an empty string if there are no entries -- callers should
    skip prompt injection entirely in that case.
    """

    sortable = sorted(
        (e for e in entries if e.source_term),
        key=lambda x: (x.priority, x.term_id),
    )
    if not sortable:
        return ""

    truncated = sortable[:max_entries]
    overflow = len(sortable) - len(truncated)

    lines: list[str] = [header]
    for entry in truncated:
        if entry.is_regex:
            lines.append(f'- (regex) "{entry.source_term}" -> "{entry.target_term}"')
        else:
            lines.append(f'- "{entry.source_term}" -> "{entry.target_term}"')
    if overflow > 0:
        lines.append(f"... ({overflow} more)")
    return "\n".join(lines)
