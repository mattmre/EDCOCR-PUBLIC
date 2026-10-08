"""Built-in OCREngine adapters + registration helpers (Phase 1a).

Additive: registering these adapters does not change the production cascade.
Phase 1b wires the worker to resolve its per-page engine through the registry
via `engine_selection.select_engine_for_page` -> `engine_for_selector_name`.
"""
from __future__ import annotations

from ocr_local.engines.adapters.easyocr_adapter import EasyOCRAdapter
from ocr_local.engines.adapters.paddle_ctc import PaddleCTCEngine
from ocr_local.engines.adapters.surya import SuryaEngine
from ocr_local.engines.adapters.tesseract import (
    TesseractEngine,
    tesseract_data_to_lines,
)
from ocr_local.engines.registry import DEFAULT_REGISTRY, EngineRegistry

# Maps the `engine_selection` selector output ("paddle"/"tesseract"/"easyocr")
# to the catalog engine_ids the adapters register under.
SELECTOR_NAME_TO_ENGINE_ID: dict[str, str] = {
    "paddle": "paddle-ctc",
    "tesseract": "tesseract",
    "easyocr": "easyocr",
}

_BUILTIN_FACTORIES = {
    "paddle-ctc": PaddleCTCEngine,
    "tesseract": TesseractEngine,
    "easyocr": EasyOCRAdapter,
}


def register_builtin_adapters(
    registry: EngineRegistry | None = None, *, replace: bool = True
) -> EngineRegistry:
    """Register the three forensic-lane adapters into ``registry``.

    Idempotent with ``replace=True``. Returns the registry for chaining.
    """
    reg = registry or DEFAULT_REGISTRY
    for engine_id, factory in _BUILTIN_FACTORIES.items():
        reg.register(engine_id, factory, replace=replace)
    return reg


def engine_for_selector_name(name: str, registry: EngineRegistry | None = None):
    """Resolve a selector name ("paddle"/...) to a registered OCREngine.

    Raises KeyError if the name is unknown or the engine is not registered.
    """
    reg = registry or DEFAULT_REGISTRY
    engine_id = SELECTOR_NAME_TO_ENGINE_ID.get(name, name)
    return reg.get(engine_id)


__all__ = [
    "PaddleCTCEngine",
    "TesseractEngine",
    "EasyOCRAdapter",
    "SuryaEngine",
    "tesseract_data_to_lines",
    "SELECTOR_NAME_TO_ENGINE_ID",
    "register_builtin_adapters",
    "engine_for_selector_name",
]
