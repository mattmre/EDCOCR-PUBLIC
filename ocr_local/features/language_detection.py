"""Per-span language detection dataclasses and helpers (Plan A -- PR A1).

Pure additive scaffold for the span-level language detection pipeline.  This
module defines the data contract for the forthcoming ``.language.json``
sidecar and provides two stdlib-only helpers used by later PRs:

* :func:`get_script_family` -- classify a single codepoint into one of a
  small set of Unicode script families used across the pipeline.
* :func:`redact_text_sample` -- privilege-aware text-sample redaction so
  that short or privilege-flagged spans never leak source text into audit
  output.

No pipeline wiring is performed in this PR; full detection logic and the
FastText integration land in PR A2/A3.  See also
``schemas/language.schema.json`` for the JSON sidecar contract and
``docs/architecture/mp-061-ph3-pipeline-config-wiring.md`` for the
surrounding strategic context.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

__all__ = [
    "SpanLanguage",
    "PageLanguage",
    "DocumentLanguage",
    "get_script_family",
    "redact_text_sample",
    "SCRIPT_FAMILIES",
    "FASTTEXT_TO_PADDLE",
    "aggregate_page_from_full_text",
    "aggregate_page_from_spans",
    "detect_span_language",
    "finalize_document_language",
    "write_language_json",
    "redetect_document",
    "LANGUAGE_SCHEMA_VERSION",
]

logger = logging.getLogger(__name__)

# Schema version emitted into the .language.json sidecar.  Matches the
# ``schema_version`` required property in ``schemas/language.schema.json``.
LANGUAGE_SCHEMA_VERSION = "1.0"

# Ordered list of script family labels emitted by :func:`get_script_family`.
# ``other`` is the catch-all bucket for unclassified codepoints (digits,
# punctuation, symbols, and scripts we do not currently track explicitly).
SCRIPT_FAMILIES: tuple[str, ...] = (
    "latin",
    "cyrillic",
    "cjk",
    "arabic",
    "devanagari",
    "georgian",
    "greek",
    "hangul",
    "other",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpanLanguage:
    """Language label for a single OCR line (span)."""

    bbox: list[float]            # [x1, y1, x2, y2] pixel coords (page DPI space)
    text_sample: str             # first 60 chars, for audit only (may be redacted)
    language: str                # PaddleOCR code from LANG_MAPPING, or "und"
    confidence: float            # 0.0-1.0
    script: str                  # Unicode script family (see SCRIPT_FAMILIES)
    detection_method: str        # "fasttext" | "script_heuristic" | "inherited_page"
    char_count: int              # len(stripped text) -- short-span gate audit


@dataclass
class PageLanguage:
    """Aggregate language stats for one page."""

    page_num: int
    primary_language: str        # dominant by char_count weighting, or "und"
    primary_confidence: float    # weighted avg confidence of primary-language spans
    languages_detected: list[str]          # deduped, sorted by descending char share
    language_char_shares: dict[str, float] # code -> fraction of chars on page
    scripts_detected: list[str]            # deduped Unicode script families
    mixed_script: bool           # True iff len(scripts_detected) >= 2
    span_count: int
    spans_labeled: int           # how many spans got a non-"und" label
    spans: list[SpanLanguage] = field(default_factory=list)


@dataclass
class DocumentLanguage:
    """Document-level language aggregate."""

    document_id: str
    source_file: str
    primary_language: str
    primary_confidence: float
    languages_detected: list[str]
    language_char_shares: dict[str, float]
    page_count: int
    pages_with_mixed_script: int
    pages: list[PageLanguage]
    processing: dict             # engine, pipeline_version, detector_model_sha256,
                                 # tokenizer_sha256, fasttext_model_sha256, timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_script_family(char: str) -> str:
    """Return the Unicode script family for a single character.

    The classification uses hand-rolled block ranges -- no ``unicodedata``
    dependency -- so the function is deterministic and stable across
    Python versions.  Input validation: empty string or strings longer
    than one codepoint return ``"other"``.

    Recognized families (keep in sync with :data:`SCRIPT_FAMILIES`):

    * ``latin``      -- U+0041..U+007A basic Latin letters plus the
                         extended range U+00C0..U+024F.
    * ``cyrillic``   -- U+0400..U+04FF.
    * ``cjk``        -- CJK Unified Ideographs (U+4E00..U+9FFF),
                         Hiragana/Katakana (U+3040..U+30FF), and Hangul
                         (U+AC00..U+D7A3).
    * ``arabic``     -- U+0600..U+06FF and U+0750..U+077F.
    * ``devanagari`` -- U+0900..U+097F.
    * ``georgian``   -- U+10A0..U+10FF.
    * ``greek``      -- U+0370..U+03FF.
    * ``hangul``     -- returned separately only when we later re-partition
                         the CJK bucket; currently CJK subsumes Hangul for
                         parity with the rest of the pipeline.
    * ``other``      -- everything else (digits, punctuation, unknown scripts).
    """
    if not char or len(char) != 1:
        return "other"

    cp = ord(char)

    # Latin: basic A-Z / a-z plus Latin-1 Supplement through Latin Extended-B.
    if 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A:
        return "latin"
    if 0x00C0 <= cp <= 0x024F:
        return "latin"

    # Greek and Coptic.
    if 0x0370 <= cp <= 0x03FF:
        return "greek"

    # Cyrillic.
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"

    # Arabic (main block + Arabic Supplement).
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return "arabic"

    # Devanagari.
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"

    # Georgian.
    if 0x10A0 <= cp <= 0x10FF:
        return "georgian"

    # CJK unified ideographs + Hiragana/Katakana + Hangul syllables.
    if 0x4E00 <= cp <= 0x9FFF:
        return "cjk"
    if 0x3040 <= cp <= 0x30FF:
        return "cjk"
    if 0xAC00 <= cp <= 0xD7A3:
        return "cjk"

    return "other"


def redact_text_sample(
    text: str,
    privilege_flagged: bool,
    token_count: int,
    threshold: int = 500,
) -> str:
    """Return an audit-safe text sample for a span.

    Privileged spans and very short documents never expose source text;
    all other spans return the first 60 characters of ``text`` so that
    downstream reviewers can eyeball the span without reading full
    content.

    Parameters
    ----------
    text:
        The raw OCR text for the span.
    privilege_flagged:
        ``True`` if the span has been flagged by the privilege detector
        (attorney-client, work product, etc.).
    token_count:
        The total document token count.  Short documents below
        ``threshold`` tokens also redact their samples to avoid leaking
        identifying content from single-page exhibits.
    threshold:
        Minimum document token count required to include a text sample.
        Defaults to ``500``.

    Returns
    -------
    str
        The empty string when redacted, otherwise ``text[:60]``.
    """
    if privilege_flagged:
        return ""
    if token_count < threshold:
        return ""
    if text is None:
        return ""
    return text[:60]


# ---------------------------------------------------------------------------
# FastText -> PaddleOCR mapping
# ---------------------------------------------------------------------------


def _build_fasttext_to_paddle() -> dict[str, str]:
    """Build a FastText-label -> PaddleOCR-code mapping.

    ``LANG_MAPPING`` in :mod:`ocr_local.config.language_config` already keys
    on FastText language codes (the 2/3-letter form emitted in
    ``__label__xx`` predictions), so no reversal is required -- we simply
    return a shallow copy so callers cannot mutate the registry table.

    Gracefully returns an empty mapping when the language_config module is
    unavailable (e.g. during minimal-import test collection or when the
    package has been aggressively trimmed for an air-gapped bundle).
    """
    try:
        from ocr_local.config.language_config import LANG_MAPPING
    except Exception:  # pragma: no cover - defensive
        return {}
    try:
        return dict(LANG_MAPPING)
    except Exception:  # pragma: no cover - defensive
        return {}


# Eagerly built module-level mapping -- matches the other language registry
# constants in :mod:`ocr_local.config.language_config`.
FASTTEXT_TO_PADDLE: dict[str, str] = _build_fasttext_to_paddle()


# ---------------------------------------------------------------------------
# Pipeline functions (Plan A -- PR A2)
# ---------------------------------------------------------------------------


def _normalize_text_sample_for_fasttext(text: str) -> str:
    """Strip embedded newlines -- FastText rejects ``\\n`` inside predict input."""
    if not text:
        return ""
    return text.replace("\n", " ").replace("\r", " ").strip()


def _primary_script_for_text(text: str) -> str:
    """Return the dominant Unicode script family for ``text``.

    Falls back to ``"other"`` when the text is empty or every codepoint
    falls into the catch-all bucket.  Matches the classification used by
    :func:`get_script_family` so callers can pass the result straight into
    a :class:`SpanLanguage` without additional post-processing.
    """
    if not text:
        return "other"
    counts: dict[str, int] = {}
    for ch in text:
        fam = get_script_family(ch)
        counts[fam] = counts.get(fam, 0) + 1
    if not counts:
        return "other"
    non_other = {k: v for k, v in counts.items() if k != "other"}
    if non_other:
        return max(non_other.items(), key=lambda kv: kv[1])[0]
    return "other"


def aggregate_page_from_full_text(
    page_num: int,
    full_text: str,
    fasttext_model: Any,
    fasttext_model_sha256: str,
    confidence_threshold: float = 0.4,
) -> PageLanguage:
    """Produce a :class:`PageLanguage` from a page's full OCR text.

    Used in PR A2 before per-span detection is available.  The page ends
    up with a single :class:`SpanLanguage` representing the aggregate of
    all OCR lines, which means char-count-weighted aggregation degenerates
    to a trivial 1:1 mapping.  PR A3 will replace this path with a real
    per-line detection loop that builds a multi-span PageLanguage.

    Parameters
    ----------
    page_num:
        1-indexed page number.
    full_text:
        Raw OCR text for the page.  Passed through
        :func:`_normalize_text_sample_for_fasttext` before being fed to
        the detector.
    fasttext_model:
        Loaded FastText model exposing a ``.predict(text, k=N)`` method.
        May be ``None`` when the pipeline ran without FastText installed.
    fasttext_model_sha256:
        SHA-256 of the loaded lid.176.bin artifact -- recorded on the
        resulting :class:`SpanLanguage` for audit trails.  Unused at the
        span layer today; retained for API symmetry with later PRs.
    confidence_threshold:
        Minimum FastText confidence required for a non-``"und"`` label.
    """
    # ``_`` to quiet linters -- kept in the public signature for symmetry
    # with the document-level finalizer that records the same SHA-256.
    del fasttext_model_sha256

    normalized = _normalize_text_sample_for_fasttext(full_text)
    char_count = len(normalized)

    und_page = PageLanguage(
        page_num=int(page_num),
        primary_language="und",
        primary_confidence=0.0,
        languages_detected=[],
        language_char_shares={},
        scripts_detected=[],
        mixed_script=False,
        span_count=0,
        spans_labeled=0,
        spans=[],
    )

    if not normalized:
        return und_page

    # Detect language via FastText.  Any failure path falls back to the
    # ``"und"`` page so that a misconfigured detector never blocks the
    # OCR sidecar from being written.
    language = "und"
    confidence = 0.0
    detection_method = "fasttext"
    if fasttext_model is None:
        detection_method = "inherited_page"
    else:
        try:
            labels, scores = fasttext_model.predict(normalized, k=3)
        except Exception as exc:
            logger.warning(
                "FastText predict failed for page %s: %s", page_num, exc
            )
            labels, scores = (), ()

        if labels:
            raw_label = labels[0] if isinstance(labels[0], str) else str(labels[0])
            ft_code = raw_label.replace("__label__", "").strip().lower()
            try:
                raw_conf = float(scores[0]) if scores else 0.0
            except (TypeError, ValueError):
                raw_conf = 0.0

            if raw_conf >= float(confidence_threshold):
                mapped = FASTTEXT_TO_PADDLE.get(ft_code)
                if mapped:
                    language = mapped
                    confidence = max(0.0, min(1.0, raw_conf))

    script = _primary_script_for_text(normalized)
    scripts_detected = [script] if script != "other" else []

    span = SpanLanguage(
        bbox=[0.0, 0.0, 0.0, 0.0],
        text_sample=normalized[:60],
        language=language,
        confidence=confidence,
        script=script,
        detection_method=detection_method,
        char_count=char_count,
    )

    spans_labeled = 0 if language == "und" else 1
    if language == "und":
        lang_shares: dict[str, float] = {}
        detected: list[str] = []
    else:
        lang_shares = {language: 1.0}
        detected = [language]

    return PageLanguage(
        page_num=int(page_num),
        primary_language=language,
        primary_confidence=confidence,
        languages_detected=detected,
        language_char_shares=lang_shares,
        scripts_detected=scripts_detected,
        mixed_script=False,
        span_count=1,
        spans_labeled=spans_labeled,
        spans=[span],
    )


# ---------------------------------------------------------------------------
# Per-span detection (Plan A -- PR A3)
# ---------------------------------------------------------------------------


# Script-family -> default PaddleOCR language code used when a span is too
# short for FastText to disambiguate reliably.  Latin short spans remain
# ``"und"`` because without surrounding context we cannot choose between
# English, French, Spanish, etc.  Plan B will re-route the CJK bucket per
# tenant; for now we default to Simplified Chinese.
_SHORT_SPAN_SCRIPT_LANG: dict[str, str] = {
    "cjk": "ch",
    "arabic": "ar",
    "devanagari": "hi",
    "cyrillic": "ru",
    "greek": "el",
    "georgian": "ka",
    "hangul": "korean",
    "latin": "und",
    "other": "und",
}


def detect_span_language(
    text: str,
    bbox: list[float],
    fasttext_model: Any,
    short_span_threshold: int = 20,
    confidence_threshold: float = 0.4,
) -> SpanLanguage:
    """Detect the language for a single OCR span (line).

    Short spans (fewer non-whitespace characters than
    ``short_span_threshold``) fall back to a Unicode-script heuristic
    because FastText is unreliable on very short strings.  Longer spans
    invoke ``fasttext_model.predict`` and map the emitted ``__label__xx``
    code to a PaddleOCR code via :data:`FASTTEXT_TO_PADDLE`.  Any exception
    from the detector -- or a confidence below ``confidence_threshold`` --
    degrades the span to ``language="und"`` without raising.

    The returned :class:`SpanLanguage` always carries a 60-character
    ``text_sample`` (via :func:`redact_text_sample` with
    ``privilege_flagged=False`` and ``token_count=len(text.split())``) so
    downstream reviewers can still eyeball span provenance in the audit
    sidecar.

    Parameters
    ----------
    text:
        Raw OCR text for the span.
    bbox:
        Pixel-space bounding box ``[x1, y1, x2, y2]`` at page DPI.
    fasttext_model:
        Loaded FastText model with a ``.predict(text, k=N)`` method.
        ``None`` is tolerated and produces an ``"und"`` span with
        ``detection_method="fasttext"`` and confidence ``0.0``.
    short_span_threshold:
        Minimum non-whitespace character count required to call FastText.
    confidence_threshold:
        Minimum FastText confidence required for a non-``"und"`` label.
    """
    stripped = (text or "").strip()
    bbox_out = list(bbox) if bbox is not None else [0.0, 0.0, 0.0, 0.0]

    # char_count reflects the *stripped* length so downstream aggregators
    # weight spans by visible content rather than whitespace padding.
    char_count = len(stripped)
    non_ws_count = sum(1 for ch in stripped if not ch.isspace())

    # Build a redaction-safe text sample irrespective of outcome so that
    # even "und" spans carry their first 60 characters of context.
    token_count = len((text or "").split())
    text_sample = redact_text_sample(
        (text or "")[:60],
        privilege_flagged=False,
        token_count=token_count,
    )

    # Primary script family is useful for both branches.
    script = _primary_script_for_text(stripped)

    # Case 1: empty text -> "und".
    if not stripped:
        return SpanLanguage(
            bbox=bbox_out,
            text_sample=text_sample,
            language="und",
            confidence=0.0,
            script=script,
            detection_method="fasttext",
            char_count=char_count,
        )

    # Case 2: short span -> script heuristic.
    if non_ws_count < int(short_span_threshold):
        short_lang = _SHORT_SPAN_SCRIPT_LANG.get(script, "und")
        return SpanLanguage(
            bbox=bbox_out,
            text_sample=text_sample,
            language=short_lang,
            # Script heuristic has no calibrated probability; record 0.0
            # so downstream aggregators fall back to the ``"und"`` default
            # rather than over-weighting short spans.
            confidence=0.0,
            script=script,
            detection_method="script_heuristic",
            char_count=char_count,
        )

    # Case 3: long span -> FastText.
    language = "und"
    confidence = 0.0
    if fasttext_model is not None:
        try:
            labels, scores = fasttext_model.predict(
                stripped.replace("\n", " "), k=1
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("FastText predict failed for span: %s", exc)
            labels, scores = (), ()

        if labels:
            raw_label = labels[0] if isinstance(labels[0], str) else str(labels[0])
            ft_code = raw_label.replace("__label__", "").strip().lower()
            try:
                raw_conf = float(scores[0]) if scores else 0.0
            except (TypeError, ValueError):
                raw_conf = 0.0

            if raw_conf >= float(confidence_threshold):
                mapped = FASTTEXT_TO_PADDLE.get(ft_code)
                if mapped:
                    language = mapped
                    confidence = max(0.0, min(1.0, raw_conf))

    return SpanLanguage(
        bbox=bbox_out,
        text_sample=text_sample,
        language=language,
        confidence=confidence,
        script=script,
        detection_method="fasttext",
        char_count=char_count,
    )


def aggregate_page_from_spans(
    page_num: int,
    spans: list[SpanLanguage],
) -> PageLanguage:
    """Roll up a list of :class:`SpanLanguage` into a :class:`PageLanguage`.

    Aggregation rules:

    * ``primary_language`` is the language with the largest total
      ``char_count`` share, excluding ``"und"`` spans.  If every span is
      ``"und"`` the page collapses to ``"und"`` with zero confidence.
    * ``primary_confidence`` is the character-count-weighted mean of the
      per-span confidences for the primary language.
    * ``language_char_shares`` normalises per-language character counts
      against the total of non-``"und"`` characters so the emitted shares
      sum to ``1.0`` (within float tolerance).
    * ``scripts_detected`` is the deduplicated set of ``script`` fields,
      excluding ``"other"``.  ``mixed_script`` is set when two or more
      real scripts co-occur on the page.
    * ``spans_labeled`` counts the spans whose language is not ``"und"``.

    The returned object carries the full span list so callers (e.g. the
    GPU worker hook in ``ocr_gpu_async``) can decide whether to serialise
    it based on ``language_include_spans`` rather than mutating the
    aggregate after the fact.
    """
    span_list = list(spans or [])

    if not span_list:
        return PageLanguage(
            page_num=int(page_num),
            primary_language="und",
            primary_confidence=0.0,
            languages_detected=[],
            language_char_shares={},
            scripts_detected=[],
            mixed_script=False,
            span_count=0,
            spans_labeled=0,
            spans=[],
        )

    per_lang_chars: dict[str, int] = {}
    per_lang_conf_num: dict[str, float] = {}
    scripts_seen: set[str] = set()
    spans_labeled = 0

    for span in span_list:
        if span is None:
            continue
        chars = int(getattr(span, "char_count", 0) or 0)
        lang = getattr(span, "language", "und") or "und"
        script = getattr(span, "script", "other") or "other"

        if script and script != "other":
            scripts_seen.add(script)

        if lang == "und" or chars <= 0:
            continue

        spans_labeled += 1
        per_lang_chars[lang] = per_lang_chars.get(lang, 0) + chars
        per_lang_conf_num[lang] = (
            per_lang_conf_num.get(lang, 0.0)
            + float(getattr(span, "confidence", 0.0) or 0.0) * chars
        )

    total_labeled_chars = sum(per_lang_chars.values())

    if per_lang_chars and total_labeled_chars > 0:
        # Stable ordering: descending share, ascending code as tiebreaker.
        primary_language = max(
            per_lang_chars.items(), key=lambda kv: (kv[1], -ord(kv[0][0]) if kv[0] else 0)
        )[0]
        primary_chars = per_lang_chars[primary_language]
        primary_confidence = (
            per_lang_conf_num.get(primary_language, 0.0) / primary_chars
            if primary_chars > 0
            else 0.0
        )
        shares = {
            code: chars / total_labeled_chars
            for code, chars in per_lang_chars.items()
        }
        languages_detected = sorted(
            shares.keys(), key=lambda c: (-shares[c], c)
        )
    else:
        primary_language = "und"
        primary_confidence = 0.0
        shares = {}
        languages_detected = []

    scripts_detected = sorted(scripts_seen)
    mixed_script = len(scripts_detected) >= 2

    return PageLanguage(
        page_num=int(page_num),
        primary_language=primary_language,
        primary_confidence=round(max(0.0, min(1.0, primary_confidence)), 6),
        languages_detected=languages_detected,
        language_char_shares={
            code: round(share, 6) for code, share in shares.items()
        },
        scripts_detected=scripts_detected,
        mixed_script=mixed_script,
        span_count=len(span_list),
        spans_labeled=spans_labeled,
        spans=span_list,
    )


def finalize_document_language(
    document_id: str,
    source_file: str,
    pages: Iterable[PageLanguage],
    fasttext_model_sha256: str,
    tokenizer_sha256: str,
    pipeline_version: str,
) -> DocumentLanguage:
    """Aggregate a page-level result list into a :class:`DocumentLanguage`.

    - ``primary_language`` is chosen by total character share across every
      page.  Ties fall through to the first language encountered.
    - ``language_char_shares`` is normalised across the entire document.
    - ``pages_with_mixed_script`` counts pages whose
      :attr:`PageLanguage.mixed_script` flag is ``True``.
    - The ``processing`` dict is assembled to satisfy the ``required``
      keys in ``schemas/language.schema.json``.
    """
    page_list = list(pages)

    total_chars = 0
    per_lang_chars: dict[str, int] = {}
    per_lang_conf_num: dict[str, float] = {}

    mixed_pages = 0
    for page in page_list:
        if page is None:
            continue
        if page.mixed_script:
            mixed_pages += 1
        share_map = dict(page.language_char_shares or {})
        # Determine per-page char total from spans so multi-span pages
        # still get proportional weighting once PR A3 lands.
        page_total = sum(
            getattr(s, "char_count", 0) for s in (page.spans or [])
        ) or sum(
            int(round(v * 1000)) for v in share_map.values()
        )
        if page_total <= 0:
            continue
        total_chars += page_total
        for code, share in share_map.items():
            contrib = int(round(share * page_total))
            per_lang_chars[code] = per_lang_chars.get(code, 0) + contrib
            per_lang_conf_num[code] = (
                per_lang_conf_num.get(code, 0.0)
                + float(page.primary_confidence) * contrib
            )

    if total_chars > 0 and per_lang_chars:
        lang_shares = {
            code: round(chars / total_chars, 6)
            for code, chars in per_lang_chars.items()
        }
        primary_language = max(
            per_lang_chars.items(), key=lambda kv: (kv[1], kv[0])
        )[0]
        primary_chars = per_lang_chars[primary_language]
        primary_confidence = (
            per_lang_conf_num.get(primary_language, 0.0) / primary_chars
            if primary_chars > 0
            else 0.0
        )
        languages_detected = sorted(
            lang_shares.keys(), key=lambda c: (-lang_shares[c], c)
        )
    else:
        lang_shares = {}
        primary_language = "und"
        primary_confidence = 0.0
        languages_detected = []

    processing = {
        "detection_engine": "fasttext",
        "detector_model_sha256": fasttext_model_sha256 or "",
        "tokenizer_sha256": tokenizer_sha256 or "",
        "fasttext_model_sha256": fasttext_model_sha256 or "",
        "pipeline_version": pipeline_version,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }

    return DocumentLanguage(
        document_id=document_id,
        source_file=source_file,
        primary_language=primary_language,
        primary_confidence=round(max(0.0, min(1.0, primary_confidence)), 6),
        languages_detected=languages_detected,
        language_char_shares=lang_shares,
        page_count=len(page_list),
        pages_with_mixed_script=mixed_pages,
        pages=page_list,
        processing=processing,
    )


def _sanitize_subfolder(subfolder: str) -> list[str]:
    """Split ``subfolder`` into safe path segments (no traversal components)."""
    if not subfolder or subfolder == ".":
        return []
    safe: list[str] = []
    for raw in subfolder.replace("\\", "/").split("/"):
        if not raw or raw in (".", ".."):
            continue
        # Strip nulls and path separators defensively.
        cleaned = raw.replace("\x00", "").strip()
        if not cleaned or cleaned in (".", ".."):
            continue
        safe.append(cleaned)
    return safe


def write_language_json(
    doc_lang: DocumentLanguage,
    output_base_dir: str,
    source_file: str,
    include_spans: bool = False,
) -> Optional[str]:
    """Write ``doc_lang`` to ``EXPORT/LANGUAGE/<subfolder>/<name>.language.json``.

    Parameters
    ----------
    doc_lang:
        Finalised document-level language aggregate.
    output_base_dir:
        Root output directory (e.g. ``/app/ocr_output``).  The sidecar is
        written under ``<base>/EXPORT/LANGUAGE/...``.
    source_file:
        Either a full path to the source document or a bare basename.
        The directory component (if any) is mirrored beneath
        ``EXPORT/LANGUAGE`` to match other sidecars.
    include_spans:
        When ``False`` (the default), ``pages[].spans`` are stripped from
        the serialised output while ``span_count``/``spans_labeled`` are
        preserved for audit.  PR A4 flips this on for verbose mode.
    """
    language_dir = os.path.join(output_base_dir, "EXPORT", "LANGUAGE")

    source_name = os.path.basename(source_file) or doc_lang.document_id or "document"
    base_name = os.path.splitext(source_name)[0]
    subfolder_raw = os.path.dirname(source_file)
    safe_parts = _sanitize_subfolder(subfolder_raw)

    target_dir = (
        os.path.join(language_dir, *safe_parts) if safe_parts else language_dir
    )

    resolved_target = os.path.realpath(target_dir)
    resolved_root = os.path.realpath(language_dir)
    if not resolved_target.startswith(resolved_root):
        logger.error(
            "Path traversal blocked in language output: %r", subfolder_raw
        )
        return None

    os.makedirs(target_dir, exist_ok=True)
    json_path = os.path.join(target_dir, f"{base_name}.language.json")

    # Build per-page payload, optionally stripping spans.
    pages_out: list[dict[str, Any]] = []
    for page in doc_lang.pages or []:
        if page is None:
            continue
        page_dict = asdict(page)
        if not include_spans:
            page_dict["spans"] = []
        pages_out.append(page_dict)

    payload: dict[str, Any] = {
        "schema_version": LANGUAGE_SCHEMA_VERSION,
        "document_id": doc_lang.document_id,
        "source_file": doc_lang.source_file,
        "processing": dict(doc_lang.processing or {}),
        "document_summary": {
            "primary_language": doc_lang.primary_language,
            "primary_confidence": doc_lang.primary_confidence,
            "languages_detected": list(doc_lang.languages_detected or []),
            "language_char_shares": dict(doc_lang.language_char_shares or {}),
            "page_count": doc_lang.page_count,
            "pages_with_mixed_script": doc_lang.pages_with_mixed_script,
        },
        "pages": pages_out,
    }

    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    except OSError as exc:
        logger.error(
            "Failed to write language JSON for %s: %s",
            doc_lang.document_id,
            exc,
        )
        return None

    return json_path


# ---------------------------------------------------------------------------
# Re-detection CLI + API (Plan A -- PR A5)
# ---------------------------------------------------------------------------


def _load_fasttext_model(fasttext_model_path: str) -> Any:
    """Load a FastText language-id model from ``fasttext_model_path``.

    Returns the loaded model object or ``None`` when FastText is unavailable
    (either the package is not installed, the model file is missing, or load
    raises).  Mirrors the gentle-degradation behaviour of the full pipeline:
    re-detection never blows up because the detector is absent.
    """
    if not fasttext_model_path or not os.path.exists(fasttext_model_path):
        return None
    try:
        import fasttext  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("fasttext import failed during redetect: %s", exc)
        return None
    try:
        model = fasttext.load_model(fasttext_model_path)
    except Exception as exc:
        logger.warning(
            "fasttext.load_model failed for %r: %s", fasttext_model_path, exc,
        )
        return None
    # Attach a best-effort SHA-256 so downstream custody/audit paths can
    # reference the exact on-disk detector artifact.
    try:
        import hashlib
        with open(fasttext_model_path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        setattr(model, "_sha256", sha)
    except Exception:  # pragma: no cover - defensive
        setattr(model, "_sha256", "")
    return model


def _emit_language_redetected_event(
    document_id: str,
    source_file: str,
    output_base_dir: str,
    doc_lang: DocumentLanguage,
) -> None:
    """Emit a ``LANGUAGE_REDETECTED`` custody event for ``document_id``.

    This is best-effort: any failure to write a custody event (missing
    custody module, disk error, permission error, ...) is logged at DEBUG
    and swallowed.  Re-detection must never fail because the audit log
    cannot be written.
    """
    try:
        from ocr_local.features.custody import CustodyChain
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Custody module unavailable for redetect: %s", exc)
        return
    try:
        custody_dir = os.path.join(output_base_dir, "custody")
        chain = CustodyChain(
            document_id=document_id,
            source_path=source_file,
            custody_dir=custody_dir,
        )
        chain.append_event(
            "LANGUAGE_REDETECTED",
            {
                "document_id": document_id,
                "source_file": source_file,
                "primary_language": doc_lang.primary_language,
                "primary_confidence": round(doc_lang.primary_confidence, 4),
                "languages_detected": list(doc_lang.languages_detected or []),
                "page_count": doc_lang.page_count,
                "pages_with_mixed_script": doc_lang.pages_with_mixed_script,
                "detection_engine": doc_lang.processing.get(
                    "detection_engine", "fasttext",
                ),
                "fasttext_model_sha256": doc_lang.processing.get(
                    "fasttext_model_sha256", "",
                ),
            },
        )
    except Exception as exc:
        logger.debug(
            "Failed to emit LANGUAGE_REDETECTED custody event: %s", exc,
        )


def redetect_document(
    pdf_path: str,
    output_json_path: str,
    output_base_dir: str,
    fasttext_model_path: str = "/app/lid.176.bin",
    config: Any = None,
) -> dict:
    """Re-run language detection against an already-OCR'd PDF text layer.

    This does NOT re-run OCR.  It extracts the embedded text layer from
    ``pdf_path`` (page by page), feeds each line through the existing
    per-span detection helpers, and writes a fresh ``.language.json``
    sidecar under ``output_base_dir/EXPORT/LANGUAGE/``.  A best-effort
    ``LANGUAGE_REDETECTED`` custody event is emitted when the custody
    module is importable.

    Parameters
    ----------
    pdf_path:
        Path to an OCR'd PDF (text layer must already be embedded).
    output_json_path:
        Optional explicit path for the ``.language.json`` sidecar.  When
        empty or falsy, the path is auto-derived as
        ``<output_base_dir>/EXPORT/LANGUAGE/<basename>.language.json``.
    output_base_dir:
        Root output directory (e.g. ``ocr_output``).  Used both for the
        auto-derived sidecar path and for the custody subfolder.
    fasttext_model_path:
        Filesystem path to a FastText ``lid.176.bin`` artifact.  When the
        file is missing or the package is not importable, redetect falls
        through to the script-heuristic path and returns an ``error``
        status for missing FastText (graceful, never crashes).
    config:
        Optional :class:`PipelineConfig` (or compatible object exposing
        ``language_short_span_threshold`` / ``language_confidence_threshold``).
        When ``None``, defaults from PR A3 are used.

    Returns
    -------
    dict
        ``{"status": "ok", "output": <path>, "primary_language": <code>,
        "page_count": <int>}`` on success, or
        ``{"status": "error", "error": <message>}`` on failure.
    """
    # Input validation -- guard against empty/missing input before we
    # touch fitz or FastText.
    if not pdf_path or not isinstance(pdf_path, str):
        return {"status": "error", "error": "pdf_path is required"}
    if not os.path.exists(pdf_path):
        return {"status": "error", "error": f"pdf not found: {pdf_path}"}

    # Pull thresholds from config when available.
    short_span_threshold = 20
    confidence_threshold = 0.4
    include_spans = False
    if config is not None:
        short_span_threshold = int(
            getattr(config, "language_short_span_threshold", 20) or 20,
        )
        confidence_threshold = float(
            getattr(config, "language_confidence_threshold", 0.4) or 0.4,
        )
        include_spans = bool(
            getattr(config, "language_include_spans", False),
        )

    # Load FastText (graceful when unavailable).
    fasttext_model = _load_fasttext_model(fasttext_model_path)
    if fasttext_model is None:
        # Graceful error: caller asked for re-detection but we cannot
        # produce a meaningful FastText label.  The CLI and REST endpoint
        # surface this as a non-zero exit / error response rather than a
        # crash so operators can investigate the missing model.
        return {
            "status": "error",
            "error": f"FastText model unavailable at {fasttext_model_path}",
        }

    # Extract text from the already-OCR'd PDF.
    try:
        import fitz  # type: ignore
    except Exception as exc:
        return {
            "status": "error",
            "error": f"PyMuPDF (fitz) unavailable: {exc}",
        }

    source_name = os.path.basename(pdf_path) or "document"
    document_id = os.path.splitext(source_name)[0] or "document"

    pages: list[PageLanguage] = []
    try:
        with fitz.open(pdf_path) as pdf:
            total_pages = pdf.page_count
            for page_index in range(total_pages):
                try:
                    page = pdf.load_page(page_index)
                    raw_text = page.get_text("text") or ""
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "redetect: failed to extract p%d of %s: %s",
                        page_index + 1, source_name, exc,
                    )
                    raw_text = ""

                spans: list[SpanLanguage] = []
                for raw_line in raw_text.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    # Text-layer extraction has no glyph geometry, so the
                    # bbox is a placeholder.  Consumers that key off bbox
                    # (e.g. XMP embedding) skip spans with zero-area boxes.
                    spans.append(
                        detect_span_language(
                            text=line,
                            bbox=[0.0, 0.0, 100.0, 12.0],
                            fasttext_model=fasttext_model,
                            short_span_threshold=short_span_threshold,
                            confidence_threshold=confidence_threshold,
                        ),
                    )

                pages.append(
                    aggregate_page_from_spans(
                        page_num=page_index + 1, spans=spans,
                    ),
                )
    except Exception as exc:
        return {"status": "error", "error": f"pdf open failed: {exc}"}

    fasttext_sha = getattr(fasttext_model, "_sha256", "") or ""
    try:
        from ocr_local.config.version import __version__ as pipeline_version
    except Exception:  # pragma: no cover - defensive
        pipeline_version = "unknown"

    doc_lang = finalize_document_language(
        document_id=document_id,
        source_file=source_name,
        pages=pages,
        fasttext_model_sha256=fasttext_sha,
        tokenizer_sha256=fasttext_sha,
        pipeline_version=pipeline_version,
    )

    # Compute output path.  Explicit path wins; otherwise derive under
    # EXPORT/LANGUAGE.
    if output_json_path:
        try:
            target_dir = os.path.dirname(output_json_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            # Build payload using the same shape as write_language_json so
            # the two write paths agree on the wire.
            pages_out: list[dict[str, Any]] = []
            for page in doc_lang.pages or []:
                if page is None:
                    continue
                page_dict = asdict(page)
                if not include_spans:
                    page_dict["spans"] = []
                pages_out.append(page_dict)
            payload: dict[str, Any] = {
                "schema_version": LANGUAGE_SCHEMA_VERSION,
                "document_id": doc_lang.document_id,
                "source_file": doc_lang.source_file,
                "processing": dict(doc_lang.processing or {}),
                "document_summary": {
                    "primary_language": doc_lang.primary_language,
                    "primary_confidence": doc_lang.primary_confidence,
                    "languages_detected": list(doc_lang.languages_detected or []),
                    "language_char_shares": dict(
                        doc_lang.language_char_shares or {},
                    ),
                    "page_count": doc_lang.page_count,
                    "pages_with_mixed_script": doc_lang.pages_with_mixed_script,
                },
                "pages": pages_out,
            }
            with open(output_json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            out_path: Optional[str] = output_json_path
        except OSError as exc:
            return {
                "status": "error",
                "error": f"failed to write sidecar: {exc}",
            }
    else:
        out_path = write_language_json(
            doc_lang=doc_lang,
            output_base_dir=output_base_dir,
            source_file=source_name,
            include_spans=include_spans,
        )
        if out_path is None:
            return {
                "status": "error",
                "error": "failed to write language sidecar",
            }

    # Best-effort custody event.
    _emit_language_redetected_event(
        document_id=document_id,
        source_file=source_name,
        output_base_dir=output_base_dir,
        doc_lang=doc_lang,
    )

    return {
        "status": "ok",
        "output": out_path,
        "primary_language": doc_lang.primary_language,
        "page_count": doc_lang.page_count,
    }


def _cli_main(argv: Optional[list[str]] = None) -> int:
    """Argparse entry point shared by ``__main__`` and tests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-run language detection on an OCR'd PDF",
    )
    parser.add_argument("--doc", required=True, help="Path to OCR'd PDF")
    parser.add_argument(
        "--output",
        required=False,
        default="",
        help="Output .language.json path (auto-derived if omitted)",
    )
    parser.add_argument(
        "--output-base-dir",
        default="ocr_output",
        help="Base output dir (default: ocr_output)",
    )
    parser.add_argument(
        "--fasttext-model",
        default="/app/lid.176.bin",
        help="Path to the FastText lid.176.bin artifact",
    )
    args = parser.parse_args(argv)

    result = redetect_document(
        pdf_path=args.doc,
        output_json_path=args.output or "",
        output_base_dir=args.output_base_dir,
        fasttext_model_path=args.fasttext_model,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    import sys

    sys.exit(_cli_main())
