"""AI-enhanced (generative VLM) OCR engines -- the AI lane.

These engines produce ADDITIVE, advisory sidecars (`EXPORT/AI_TEXT/*.aitext.json`)
and never the forensic PDF text-of-record. They are governed exactly like any
other pluggable engine (default-blocked by `forensic_mode`).
"""
from __future__ import annotations

from ocr_local.engines.vlm.base import (
    AiTextBlock,
    AiTextPage,
    AiTextResult,
    VLMEngine,
    build_ai_text_payload,
    write_ai_text_json,
)
from ocr_local.engines.vlm.granite_docling import GraniteDoclingEngine
from ocr_local.engines.vlm.paddleocr_vl import PaddleOCRVLEngine

__all__ = [
    "AiTextBlock",
    "AiTextPage",
    "AiTextResult",
    "VLMEngine",
    "build_ai_text_payload",
    "write_ai_text_json",
    "PaddleOCRVLEngine",
    "GraniteDoclingEngine",
]
