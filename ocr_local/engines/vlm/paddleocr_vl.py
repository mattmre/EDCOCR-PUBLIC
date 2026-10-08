"""PaddleOCR-VL adapter (AI-enhanced lane, generative).

PaddleOCR-VL (0.9B, Apache-2.0) ships in paddleocr >= 3.3. This repo pins
paddleocr 2.9.1, so the model is NOT available here: ``is_available()`` returns
False and ``analyze_pages`` raises ``NotImplementedError`` rather than shipping
inference code that cannot be exercised. The real pipeline lands when the VLM
worker image (paddleocr >= 3.3) is built -- see the Phase 3b infra.
"""
from __future__ import annotations

from typing import Any

from ocr_local.engines import catalog
from ocr_local.engines.capability import EngineCapability
from ocr_local.engines.vlm.base import AiTextResult, VLMEngine


class PaddleOCRVLEngine(VLMEngine):
    @property
    def capability(self) -> EngineCapability:
        return catalog.PADDLEOCR_VL

    def is_available(self) -> bool:
        # Phase 3a stub: the AI-text contract + adapter shell are in place, but
        # PaddleOCR-VL inference is not wired yet (Phase 3b builds the VLM worker
        # image with baked model weights). Report unavailable so the registry
        # never routes to an engine that cannot actually run end-to-end.
        return False

    def analyze_pages(self, images: list[Any], lang_code: str = "en") -> AiTextResult:
        raise NotImplementedError(
            "PaddleOCR-VL inference is wired in the VLM worker image (Phase 3b); "
            "the AI-text sidecar contract (ocr_local.engines.vlm) is ready now."
        )
