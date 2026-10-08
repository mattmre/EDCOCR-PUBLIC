"""Tesseract adapter -- forensic fallback engine.

The production pipeline runs Tesseract inline and writes text directly at
`(left, top)` with a fixed font size. This adapter instead emits the
pipeline-standard `(text, box, confidence)` tuple so Tesseract flows through the
uniform `insert_text_line` writer like every other engine.

NOTE (behavior change, fallback-only): routing Tesseract output through
`insert_text_line` changes the rendered text-layer geometry for Tesseract pages
(adaptive bbox sizing vs the old fixed-fontsize anchor). This does NOT affect the
forensic PaddleOCR CTC path. The byte-identical guarantee in Phase 1 covers the
Paddle default; Tesseract is an explicitly-documented improvement.

`tesseract_data_to_lines` is a pure function (no pytesseract dependency) so the
tuple-conversion logic is unit-testable without the binary installed.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from ocr_local.config.language_config import get_tesseract_code
from ocr_local.engines import catalog
from ocr_local.engines.base import OCREngine, OcrLine
from ocr_local.engines.capability import EngineCapability

# Mirrors the GS/Tesseract/Poppler subprocess timeout convention (gotcha #83).
TESSERACT_TIMEOUT = int(os.environ.get("TESSERACT_TIMEOUT", "120"))


def tesseract_data_to_lines(data: dict) -> list[OcrLine]:
    """Convert a `pytesseract.image_to_data` DICT into `(text, box, confidence)`.

    Mirrors the production branch's filtering: keep only entries with
    ``conf > 0`` and non-empty text. ``box`` is the axis-aligned
    ``[left, top, left+width, top+height]``; ``confidence`` is ``conf / 100``.
    """
    lines: list[OcrLine] = []
    texts = data.get("text") or []
    for i in range(len(texts)):
        txt = (texts[i] or "").strip()
        if not txt:
            continue
        try:
            conf_val = int(float(data["conf"][i]))
        except (KeyError, IndexError, ValueError, TypeError):
            conf_val = -1
        if conf_val <= 0:
            continue
        try:
            left = float(data["left"][i])
            top = float(data["top"][i])
            width = float(data["width"][i])
            height = float(data["height"][i])
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        box = [left, top, left + width, top + height]
        lines.append((txt, box, conf_val / 100.0))
    return lines


class TesseractEngine(OCREngine):
    """Adapter over pytesseract (forensic fallback)."""

    @property
    def capability(self) -> EngineCapability:
        return catalog.TESSERACT

    def is_available(self) -> bool:
        try:
            import pytesseract  # noqa: F401

            return True
        except Exception:
            return False

    def get_handle(self, lang_code: str, device: str = "gpu:0") -> tuple[Any, Any]:
        # Tesseract is a stateless subprocess: no persistent engine, no GPU lock.
        return (None, contextlib.nullcontext())

    def run(self, handle: Any, image: Any, lang_code: str) -> list[OcrLine]:
        import pytesseract

        data = pytesseract.image_to_data(
            image,
            lang=get_tesseract_code(lang_code),
            output_type=pytesseract.Output.DICT,
            timeout=TESSERACT_TIMEOUT,
        )
        return tesseract_data_to_lines(data)
