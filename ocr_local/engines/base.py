"""OCREngine ABC + the normalized ``(text, box, confidence)`` line contract.

Phase 0 foundation for the pluggable-engine framework
(see ``docs/architecture/pluggable-engines-enrichment-program-20260619.md``).
Additive only -- nothing here is wired into ``ocr_gpu_async.py`` yet; the
forensic CTC cascade is unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ocr_local.engines.capability import EngineCapability

# Pipeline-standard OCR line: ``(text, box, confidence)``.  ``box`` is a
# 4-point polygon ``[[x, y] x 4]`` (PaddleOCR detector space) or an
# axis-aligned ``[x1, y1, x2, y2]``; ``confidence`` is a float in ``[0, 1]``.
# This matches ``extract_paddle_lines()`` output (docs/release-candidate.md) so a
# registry-selected engine flows through the existing ``insert_text_line`` PDF
# writer without touching the assembler, chunk-PDF format, or custody chain.
OcrLine = tuple[str, Any, float]


class OCREngine(ABC):
    """Adapter over a concrete OCR engine.

    Implementations MUST be cheap to construct -- defer model/weight loading to
    :meth:`get_handle` (mirrors the existing lazy ``get_or_create_engine``
    pattern in ``ocr_gpu_async.py``).  A forensic (CTC) engine declares
    ``capability.is_deterministic_ctc=True`` and ``is_generative=False``; the
    governance layer (:mod:`ocr_local.engines.governance`) then guarantees a
    generative engine can never become the silent forensic default.
    """

    @property
    @abstractmethod
    def capability(self) -> EngineCapability:
        """Static license/lane metadata for this engine."""

    @property
    def name(self) -> str:
        return self.capability.engine_id

    @property
    def is_ctc(self) -> bool:
        """True only for a deterministic engine with no generative surface."""
        return self.capability.is_deterministic_ctc and not self.capability.is_generative

    @abstractmethod
    def is_available(self) -> bool:
        """True when the engine's runtime deps/models are importable/loadable."""

    @abstractmethod
    def get_handle(self, lang_code: str, device: str = "gpu:0") -> tuple[Any, Any]:
        """Return ``(engine_handle, inference_lock)`` for ``lang_code``.

        Mirrors ``get_or_create_engine`` -- the returned lock serializes GPU
        inference across worker threads.
        """

    @abstractmethod
    def run(self, handle: Any, image: Any, lang_code: str) -> list[OcrLine]:
        """Run OCR on ``image`` and return normalized ``(text, box, confidence)``.

        The returned list MUST honor the :data:`OcrLine` contract so downstream
        consumers (``insert_text_line``, handwriting, table-fallback) are
        engine-agnostic.
        """
