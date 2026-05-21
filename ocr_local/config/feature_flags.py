"""Runtime feature registry for the OCR pipeline.

Queries which optional modules are importable at startup without
modifying any existing _FEATURE_AVAILABLE flag patterns.

Usage:
    from feature_flags import get_pipeline_features, is_feature_available
    features = get_pipeline_features()  # dict[str, bool]
    if is_feature_available("ner"):
        ...
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# Map feature names to the module/attribute they require.
# Format: feature_name -> (module_name, attribute_name_or_None)
_FEATURE_MAP: dict[str, tuple[str, str | None]] = {
    "fasttext": ("fasttext", None),
    "paddleocr": ("paddleocr", "PaddleOCR"),
    "pdf2image": ("pdf2image", "convert_from_path"),
    "custody": ("custody", "CustodyChain"),
    "validation": ("validation", "ProcessingValidator"),
    "dpi_escalation": ("dpi_escalation", "DPIEscalationManager"),
    "ner": ("ner", "NERProcessor"),
    "handwriting": ("handwriting", "HandwritingDetector"),
    "signature_verification": ("signature_verification", "SignatureVerifier"),
    "vertical_text": ("vertical_text", "VerticalTextDetector"),
    "table_fallback": ("table_fallback", "TableFallbackAnalyzer"),
    "classification": ("classification", "DocumentClassifier"),
    "extraction": ("extraction", "StructuredExtractor"),
    "semantic_extraction": ("semantic_extraction", "SemanticExtractor"),
    "entity_consolidator": ("entity_consolidator", "EntityConsolidator"),
    "relationship_extraction": ("relationship_extraction", "RelationshipExtractor"),
    "routing": ("routing", None),
    "output_assembler": ("output_assembler", "OutputAssembler"),
    "exception_router": ("exception_router", "ExceptionRouter"),
    "unicode_utils": ("unicode_utils", None),
    "font_selector": ("font_selector", "get_font_path"),
    "gpu_optimization": ("gpu_optimization", None),
    "page_cache": ("page_cache", None),
    "page_routing": ("page_routing", None),
    "onnx_inference": ("ocr_inference_backend", None),
    "engine_selection": ("engine_selection", None),
    "noise_profiling": ("noise_profiling", None),
}

_cache: dict[str, bool] | None = None


def get_pipeline_features(force_refresh: bool = False) -> dict[str, bool]:
    """Return dict of feature_name -> is_available for all known features.

    Results are cached after the first call because feature availability is
    determined by which packages are installed and does not change at runtime.
    Pass ``force_refresh=True`` to re-probe (useful in tests).
    """
    global _cache  # noqa: PLW0603
    if _cache is not None and not force_refresh:
        return dict(_cache)

    result: dict[str, bool] = {}
    for feature, (module_name, attr) in _FEATURE_MAP.items():
        try:
            mod = importlib.import_module(module_name)
            if attr:
                available = hasattr(mod, attr)
            else:
                available = True
        except ImportError:
            available = False
        except Exception as exc:
            logger.debug("Feature probe %s failed: %s", feature, exc)
            available = False
        result[feature] = available

    _cache = result
    return dict(result)


def is_feature_available(feature: str) -> bool:
    """Check if a named feature is available. Returns False for unknown features."""
    return get_pipeline_features().get(feature, False)
