"""Handwriting MT opt-in flag and translation routing.

Plan B Q11 establishes the handwriting-aware translation path:
detect handwriting via the existing pipeline -> if the engine handles
handwriting natively, pass the raw span text -> otherwise, use the
OCR-extracted text as the translation input -> reassemble.

The ``ENABLE_HANDWRITING_MT`` flag defaults to ``False`` and must be
left off in production until a 48h bake under the standard rollout
process.  Use the ``"1" / "true" / "yes"`` env-var pattern (consistent
with other opt-in features) -- not ``== "true"``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ocr_local.translation.engines.base import TranslationEngine
    from ocr_local.translation.models import SpanTranslation


# Default OFF -- never flip without 48h bake + explicit post-bake PR.
ENABLE_HANDWRITING_MT: bool = os.environ.get(
    "ENABLE_HANDWRITING_MT", "false"
).lower() in ("1", "true", "yes")


def should_translate_handwriting(span: dict) -> bool:
    """Return ``True`` only when both gates pass.

    Both ``ENABLE_HANDWRITING_MT`` (env flag, default False) and the
    span-level ``is_handwriting`` flag must be true for handwriting
    spans to enter the translation path.
    """

    if not ENABLE_HANDWRITING_MT:
        return False
    return bool(span.get("is_handwriting", False))


def translate_handwriting_span(
    span: dict,
    engine: "TranslationEngine",
    *,
    ocr_text_fallback: str,
    src: str,
    tgt: str,
) -> Optional["SpanTranslation"]:
    """Translate a handwriting span via the supplied engine.

    If ``engine.capability.handles_handwriting_natively`` is ``True``,
    the span's own ``text`` is used as the input; otherwise the
    ``ocr_text_fallback`` is used (this is the cleaned text emitted by
    the existing OCR-based handwriting analyzer in
    ``ocr_local.features.handwriting``).
    """

    if engine.capability.handles_handwriting_natively:
        input_text = span.get("text", ocr_text_fallback)
    else:
        input_text = ocr_text_fallback

    spans_to_translate = [
        {
            "text": input_text,
            "span_id": span.get("span_id", "hw_0"),
            "bbox": span.get("bbox", [0.0, 0.0, 100.0, 12.0]),
        }
    ]
    results = engine.translate_spans(spans_to_translate, src, tgt)
    return results[0] if results else None
