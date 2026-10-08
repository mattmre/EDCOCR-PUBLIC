"""Granite-Docling adapter (AI-enhanced lane, generative VLM).

Granite-Docling 258M (Apache-2.0, IBM) is a compact, CPU-feasible document VLM
emitting DocTags. Like the PaddleOCR-VL adapter this is a Phase 4 shell:
``is_available()`` returns False and ``analyze_pages`` raises
``NotImplementedError`` -- real inference is wired in the Docling worker image
(Phase 3b/4b) with the model present. No speculative inference code.
"""
from __future__ import annotations

from typing import Any

from ocr_local.engines import catalog
from ocr_local.engines.capability import EngineCapability
from ocr_local.engines.vlm.base import AiTextResult, VLMEngine


class GraniteDoclingEngine(VLMEngine):
    @property
    def capability(self) -> EngineCapability:
        return catalog.GRANITE_DOCLING

    def is_available(self) -> bool:
        # Phase 4 shell: inference wired in the Docling worker image (Phase 4b).
        return False

    def analyze_pages(self, images: list[Any], lang_code: str = "en") -> AiTextResult:
        raise NotImplementedError(
            "Granite-Docling inference is wired in the Docling worker image "
            "(Phase 4b); the AI-text sidecar contract is ready now."
        )
