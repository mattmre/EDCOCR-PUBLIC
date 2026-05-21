"""Lightweight Prometheus counters for OCR pipeline instrumentation.

Provides opt-in counters that can be imported from any pipeline module
(e.g. ``ocr_gpu_async.py``, ``dpi_escalation.py``) without requiring
Django or the full coordinator stack.

Counters:
    ``ocr_engine_usage_total``  – Per-engine OCR operation counter (M-14)
    ``ocr_dpi_escalation_total`` – DPI escalation event counter (M-15)

If ``prometheus_client`` is not installed, all public functions are
safe no-ops so callers never need to guard imports.

Usage::

    from ocr_metrics import record_engine_usage, record_dpi_escalation

    record_engine_usage("paddle")
    record_dpi_escalation(300, 450)
"""

from __future__ import annotations

import logging

__all__ = [
    "ENGINE_USAGE_COUNTER",
    "DPI_ESCALATION_COUNTER",
    "record_engine_usage",
    "record_dpi_escalation",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional prometheus_client import (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter

    _HAS_PROM = True
except ImportError:  # pragma: no cover
    _HAS_PROM = False
    Counter = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# M-14: Per-engine OCR breakdown counter
# ---------------------------------------------------------------------------
_VALID_ENGINES = frozenset({"paddle", "tesseract", "onnx", "image_only"})

if _HAS_PROM:
    ENGINE_USAGE_COUNTER: Counter | None = Counter(
        "ocr_engine_usage_total",
        "Total OCR operations by engine type",
        ["engine"],
    )
else:
    ENGINE_USAGE_COUNTER = None


def record_engine_usage(engine: str) -> None:
    """Increment the per-engine OCR usage counter.

    Args:
        engine: One of ``"paddle"``, ``"tesseract"``, ``"onnx"``,
                ``"image_only"``.  Unknown values are normalised to
                the closest known label or logged as a warning.
    """
    label = _normalise_engine(engine)
    if ENGINE_USAGE_COUNTER is not None:
        ENGINE_USAGE_COUNTER.labels(engine=label).inc()


def _normalise_engine(raw: str) -> str:
    """Map raw engine strings to canonical labels."""
    key = raw.lower().strip().replace(" ", "").replace("-", "").replace("_", "")
    _alias_map = {
        "paddle": "paddle",
        "paddleocr": "paddle",
        "tesseract": "tesseract",
        "onnx": "onnx",
        "onnxruntime": "onnx",
        "imageonly": "image_only",
        "image_only": "image_only",
    }
    mapped = _alias_map.get(key)
    if mapped is not None:
        return mapped
    if raw in _VALID_ENGINES:
        return raw
    logger.debug("Unknown engine label %r, using as-is", raw)
    return raw


# ---------------------------------------------------------------------------
# M-15: DPI escalation rate counter
# ---------------------------------------------------------------------------
if _HAS_PROM:
    DPI_ESCALATION_COUNTER: Counter | None = Counter(
        "ocr_dpi_escalation_total",
        "Total DPI escalation events",
        ["from_dpi", "to_dpi"],
    )
else:
    DPI_ESCALATION_COUNTER = None


def record_dpi_escalation(from_dpi: int, to_dpi: int) -> None:
    """Increment the DPI escalation counter.

    Args:
        from_dpi: Original DPI before escalation (e.g. 300).
        to_dpi:   Target DPI after escalation (e.g. 450, 600).
    """
    if DPI_ESCALATION_COUNTER is not None:
        DPI_ESCALATION_COUNTER.labels(
            from_dpi=str(from_dpi),
            to_dpi=str(to_dpi),
        ).inc()
