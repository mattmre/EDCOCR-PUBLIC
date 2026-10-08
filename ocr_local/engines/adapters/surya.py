"""Surya adapter (AI-enhanced lane, GATED license, det+rec heads -- not CTC).

Surya OCR 2 uses dedicated detection + recognition heads (not a free-running LLM
decoder), so it is the most transcription-faithful AI-lane engine -- but its
weights are commercial-gated (modified OpenRAIL-M, <$5M tier) and it is NOT
CTC-classified, so it stays in the AI-enhanced lane and is admitted only via
``allow_gated_licensed`` + ack. Phase 4 shell: ``is_available()`` returns False
and ``run`` raises ``NotImplementedError`` -- the Surya runtime + gated weights
are wired in the Surya worker image (Phase 4b).
"""
from __future__ import annotations

import contextlib
from typing import Any

from ocr_local.engines import catalog
from ocr_local.engines.base import OCREngine, OcrLine
from ocr_local.engines.capability import EngineCapability


class SuryaEngine(OCREngine):
    @property
    def capability(self) -> EngineCapability:
        return catalog.SURYA

    def is_available(self) -> bool:
        # Phase 4 shell: Surya runtime + gated OpenRAIL weights are wired in the
        # Surya worker image (Phase 4b). Report unavailable until then.
        return False

    def get_handle(self, lang_code: str, device: str = "gpu:0") -> tuple[Any, Any]:
        return (None, contextlib.nullcontext())

    def run(self, handle: Any, image: Any, lang_code: str) -> list[OcrLine]:
        raise NotImplementedError(
            "Surya inference is wired in the Surya worker image (Phase 4b); "
            "weights are commercial-gated (OpenRAIL-M, <$5M tier)."
        )
