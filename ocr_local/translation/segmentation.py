"""Sentence segmentation for the translation pipeline.

Primary segmenter is ``pysbd`` -- when not available we fall back to a
deterministic regex-based splitter so the translation pipeline keeps
working in CI lanes that don't install the optional translation
dependencies (see ``requirements-translation.txt``).

The segmenter returns span-shaped dicts so downstream code can feed
them directly into a :class:`~ocr_local.translation.engines.base.TranslationEngine`
adapter.  The ``bbox`` field is a placeholder ``[0, 0, 100, 12]``: this
module operates on raw text and has no glyph geometry.  Callers that
need real bboxes must compute them upstream.
"""

from __future__ import annotations

import re


def segment_to_sentences(text: str, lang: str) -> list[dict]:
    """Segment ``text`` into sentence span dicts.

    Returns a list of ``{"text": str, "bbox": list[float], "span_id": str}``
    dicts.  Empty or whitespace-only sentences are filtered out.
    """

    try:
        import pysbd

        segmenter = pysbd.Segmenter(language=_pysbd_lang(lang), clean=False)
        sentences = segmenter.segment(text)
    except (ImportError, ValueError):
        sentences = _regex_segment(text)

    return [
        {
            "text": s.strip(),
            "bbox": [0.0, 0.0, 100.0, 12.0],
            "span_id": f"seg_{i}",
        }
        for i, s in enumerate(sentences)
        if s and s.strip()
    ]


def _pysbd_lang(lang: str) -> str:
    """Map a BCP-47 language tag to a pysbd-supported language code.

    Unknown languages fall back to English -- pysbd raises ``ValueError``
    on unsupported codes, and the fallback keeps the segmenter usable
    for languages without dedicated rules.
    """

    mapping = {
        "en": "en",
        "fr": "fr",
        "de": "de",
        "es": "es",
        "it": "it",
        "pt": "pt",
        "nl": "nl",
        "ru": "ru",
        "pl": "pl",
        "cs": "cs",
        "zh": "zh",
        "ja": "ja",
        "ko": "ko",
        "ar": "ar",
    }
    base = lang.split("-")[0].lower()
    return mapping.get(base, "en")


def _regex_segment(text: str) -> list[str]:
    """Deterministic regex fallback splitter.

    Splits on ``.`` / ``!`` / ``?`` followed by whitespace.  Good enough
    for the pipeline contract test and for languages that pysbd doesn't
    cover; production deployments install pysbd.
    """

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]
