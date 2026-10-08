"""VLM engine ABC + the AI-text sidecar data model and writer.

The data model and ``write_ai_text_json`` are pure and fully testable. Concrete
VLM adapters (e.g. PaddleOCR-VL) implement :meth:`VLMEngine.analyze_pages`;
actual inference is operator-runtime-gated (the model + its runtime must be
installed), so adapters that cannot run in a given environment report
``is_available() == False`` and raise ``NotImplementedError`` from
``analyze_pages`` rather than shipping speculative inference code.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from ocr_local.engines.capability import EngineCapability

SCHEMA_VERSION = "1.0"
EXPORT_SUBDIR = "AI_TEXT"
SIDECAR_SUFFIX = ".aitext.json"


@dataclass
class AiTextBlock:
    text: str
    type: str = "text"
    bbox: list[float] | None = None
    confidence: float | None = None


@dataclass
class AiTextPage:
    page_num: int
    markdown: str = ""
    text: str = ""
    blocks: list[AiTextBlock] = field(default_factory=list)


@dataclass
class AiTextResult:
    engine_id: str
    license_spdx: str
    pages: list[AiTextPage] = field(default_factory=list)
    engine_version: str | None = None


class VLMEngine(ABC):
    """Adapter over a generative document VLM (AI-enhanced lane)."""

    @property
    @abstractmethod
    def capability(self) -> EngineCapability:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """True only when the model + its runtime are actually installed."""

    @abstractmethod
    def analyze_pages(self, images: list[Any], lang_code: str = "en") -> AiTextResult:
        """Run the VLM over page images and return advisory AI text.

        Implementations MUST raise ``NotImplementedError`` (not fabricate output)
        when the runtime is unavailable.
        """


def build_ai_text_payload(
    result: AiTextResult,
    *,
    source_file: str,
    document_id: str,
    pipeline_version: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build the schema-conformant AI-text sidecar payload (pure)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "source_file": source_file,
        "advisory": True,
        "deterministic": False,
        "engine": {
            "engine_id": result.engine_id,
            "engine_version": result.engine_version,
            "lane": "ai_enhanced",
            "is_generative": True,
            "license_spdx": result.license_spdx,
        },
        "processing": {
            "pipeline_version": pipeline_version,
            "timestamp": timestamp,
        },
        "pages": [
            {
                "page_num": page.page_num,
                "markdown": page.markdown,
                "text": page.text,
                "blocks": [asdict(block) for block in page.blocks],
            }
            for page in result.pages
        ],
    }


def write_ai_text_json(
    result: AiTextResult,
    output_folder: str,
    *,
    source_file: str,
    document_id: str,
    pipeline_version: str,
    timestamp: str,
    subfolder: str = "",
) -> str:
    """Write the AI-text sidecar to ``EXPORT/AI_TEXT/<subfolder>/<base>.aitext.json``.

    Returns the written path. Includes a realpath traversal guard, matching the
    existing enrichment sidecar writers.
    """
    payload = build_ai_text_payload(
        result,
        source_file=source_file,
        document_id=document_id,
        pipeline_version=pipeline_version,
        timestamp=timestamp,
    )

    from ocr_distributed.ocr_utils import (
        build_sidecar_base_name,
        resolve_safe_output_subfolder,
        write_json_atomic,
    )

    base = build_sidecar_base_name(source_file)
    root_dir = os.path.join(output_folder, "EXPORT", EXPORT_SUBDIR)
    out_dir = resolve_safe_output_subfolder(root_dir, subfolder)
    out_path = os.path.join(out_dir, f"{base}{SIDECAR_SUFFIX}")

    os.makedirs(out_dir, exist_ok=True)
    write_json_atomic(
        out_path,
        payload,
        allowed_root=root_dir,
        default=None,
    )
    return out_path
