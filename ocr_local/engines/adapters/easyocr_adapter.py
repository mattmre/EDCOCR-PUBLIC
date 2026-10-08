"""EasyOCR adapter -- optional forensic-lane alternative (handwriting-friendly).

Wraps the existing `EasyOCREngine` (`easyocr_engine.py`), which already returns
the `(text, box, confidence)` tuple. Unlike the legacy `get_easyocr_engine`
singleton, this adapter does NOT gate on the `ENABLE_EASYOCR` env flag --
engine availability/selection is governed by the registry + EnginePolicy, not
the legacy flag. Heavy deps are imported lazily.
"""
from __future__ import annotations

import contextlib
from typing import Any

from ocr_local.engines import catalog
from ocr_local.engines.base import OCREngine, OcrLine
from ocr_local.engines.capability import EngineCapability


class EasyOCRAdapter(OCREngine):
    """Adapter over the EasyOCR CRNN+CTC engine."""

    def __init__(self) -> None:
        self._engines: dict[str, Any] = {}

    @property
    def capability(self) -> EngineCapability:
        return catalog.EASYOCR

    def is_available(self) -> bool:
        try:
            from easyocr_engine import _EASYOCR_AVAILABLE

            return bool(_EASYOCR_AVAILABLE)
        except Exception:
            return False

    def get_handle(self, lang_code: str, device: str = "cpu") -> tuple[Any, Any]:
        from easyocr_engine import EasyOCREngine, map_paddle_to_easyocr

        code = map_paddle_to_easyocr(lang_code) or "en"
        engine = self._engines.get(code)
        if engine is None:
            engine = EasyOCREngine(languages=[code], gpu=False)
            self._engines[code] = engine
        # EasyOCR is CPU-only by design; no GPU inference lock needed.
        return (engine, contextlib.nullcontext())

    def run(self, handle: Any, image: Any, lang_code: str) -> list[OcrLine]:
        engine, _lock = handle if isinstance(handle, tuple) else (handle, None)
        if engine is None:
            return []
        # EasyOCREngine.ocr already returns the (text, box, confidence) contract.
        return engine.ocr(image)
