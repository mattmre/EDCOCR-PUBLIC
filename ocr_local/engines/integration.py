"""Phase 1b pipeline integration: selector-driven engine routing.

`resolve_engine_lines` is the single, fully-tested entry point the production
worker calls (behind the default-off `OCR_ENGINE_REGISTRY` flag) to pick an OCR
engine per page and return the pipeline-standard `(text, box, confidence)` lines.

Forensic-default safety: when the selector returns "paddle" (the default for
`OCR_ENGINE_SELECTION=paddle`) -- or on ANY error -- this delegates to
``paddle_runner``, the caller's production paddle path (shared `_ENGINE_CACHE`
+ two-pass behavior). So the forensic CTC output is byte-identical to today
unless an operator explicitly selects a different engine AND it is available.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from ocr_local.engines.adapters import engine_for_selector_name
from ocr_local.engines.base import OcrLine
from ocr_local.engines.catalog import PADDLE_CTC
from ocr_local.engines.governance import (
    BLOCKED_FORENSIC_MODE,
    EnginePolicy,
    evaluate_engine,
)
from ocr_local.engines.provenance import (
    build_engine_provenance_event,
    build_lane_boundary_event,
    build_license_blocked_event,
)
from ocr_local.engines.registry import EngineRegistry

logger = logging.getLogger(__name__)


class EngineAdmissionDenied(RuntimeError):
    """Raised when policy leaves no admissible OCR engine to run."""

# Provenance labels per selector engine name: (ocr_method, PageStatus value).
# Single source of truth shared with the worker so a registry-routed page is
# never mislabeled -- e.g. a Tesseract/EasyOCR page must NOT be recorded as
# "PaddleOCR" in custody / validation / PDF metadata.
ENGINE_PROVENANCE: dict[str, tuple[str, str]] = {
    "paddle": ("PaddleOCR", "Paddle"),
    "tesseract": ("Tesseract", "Tesseract"),
    "easyocr": ("EasyOCR", "EasyOCR"),
}


def engine_provenance(engine_name: str) -> tuple[str, str]:
    """Return ``(ocr_method, PageStatus_value)`` for a selector engine name.

    Unknown names fall back to the forensic paddle label.
    """
    return ENGINE_PROVENANCE.get(engine_name, ENGINE_PROVENANCE["paddle"])


def resolve_engine_lines(
    *,
    image: Any,
    image_np: Any,
    lang_code: str,
    registry: EngineRegistry,
    paddle_runner: Callable[[], list[OcrLine]],
    force: str | None = None,
    is_handwritten: bool = False,
    policy: EnginePolicy | None = None,
    page_id: str | None = None,
    custody_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, list[OcrLine]]:
    """Select an engine and return ``(engine_name, lines)``.

    ``engine_name`` is the selector name actually used ("paddle"/"tesseract"/
    "easyocr"), so the caller can label provenance (ocr_method / PageStatus)
    from the real engine. The paddle default path -- and ANY error/unavailability
    -- returns ``("paddle", paddle_runner())``.

    ``lines`` are the pipeline-standard `(text, box, confidence)` tuples.

    Parameters
    ----------
    image : PIL.Image.Image
        Page image used only for quality analysis in ``auto`` selection mode.
    image_np : numpy.ndarray
        The image array actually fed to the engine.
    lang_code : str
        PaddleOCR-style language hint.
    registry : EngineRegistry
        Registry the non-paddle adapters are registered in.
    paddle_runner : callable
        Zero-arg callable returning the production paddle path's lines. Invoked
        for the default/forensic path and as the fallback on any divergence.
    force : str, optional
        Force a selector name ("paddle"/"tesseract"/"easyocr").
    is_handwritten : bool
        Handwriting hint for ``auto`` routing.
    policy : EnginePolicy, optional
        Governance policy for admitting non-paddle engines. Defaults to the
        forensic-safe policy.
    page_id : str, optional
        Stable page identifier embedded in custody events.
    custody_event_sink : callable, optional
        Receives ENGINE_PROVENANCE / LICENSE_BLOCKED /
        LANE_BOUNDARY_ENFORCED event dicts for the caller to append to its
        custody chain.
    """
    active_policy = policy or EnginePolicy()
    event_page_id = page_id or ""

    def _emit(event: dict[str, Any]) -> None:
        if custody_event_sink is not None:
            custody_event_sink(event)

    def _fallback_to_paddle(fell_back_from: str | None = None) -> tuple[str, list[OcrLine]]:
        decision = evaluate_engine(PADDLE_CTC, active_policy)
        if not decision.admitted:
            _emit(
                build_license_blocked_event(
                    capability=PADDLE_CTC,
                    policy=active_policy,
                    reason=decision.reason,
                    page_id=event_page_id,
                )
            )
            raise EngineAdmissionDenied(
                f"paddle fallback denied by policy: {decision.reason}"
            )
        lines = paddle_runner()
        _emit(
            build_engine_provenance_event(
                page_id=event_page_id,
                capability=PADDLE_CTC,
                policy=active_policy,
                fell_back_from=fell_back_from,
            )
        )
        return "paddle", lines

    # Import here so this module stays importable without the (optional)
    # engine_selection dependency present.
    try:
        from engine_selection import select_engine_for_page
    except Exception:  # pragma: no cover - selector always present in repo
        return _fallback_to_paddle()

    try:
        name = select_engine_for_page(
            image=image, force=force, is_handwritten=is_handwritten
        )
    except Exception as exc:
        logger.warning("engine selection failed (%s); using paddle default", exc)
        return _fallback_to_paddle()

    if name == "paddle":
        return _fallback_to_paddle()

    try:
        engine = engine_for_selector_name(name, registry)
        capability = engine.capability
    except Exception as exc:
        logger.warning(
            "pluggable engine %r failed (%s); using paddle default", name, exc
        )
        return _fallback_to_paddle(name)

    decision = evaluate_engine(capability, active_policy)
    if not decision.admitted:
        if decision.reason == BLOCKED_FORENSIC_MODE:
            _emit(
                build_lane_boundary_event(
                    capability=capability,
                    policy=active_policy,
                    page_id=event_page_id,
                )
            )
        else:
            _emit(
                build_license_blocked_event(
                    capability=capability,
                    policy=active_policy,
                    reason=decision.reason,
                    page_id=event_page_id,
                )
            )
        logger.warning(
            "engine %r denied by policy (%s); using paddle default",
            capability.engine_id,
            decision.reason,
        )
        return _fallback_to_paddle(capability.engine_id)

    try:
        available = engine.is_available()
    except Exception as exc:
        logger.warning(
            "pluggable engine %r availability failed (%s); using paddle default",
            name,
            exc,
        )
        return _fallback_to_paddle(capability.engine_id)

    if not available:
        logger.info("engine %r unavailable; using paddle default", name)
        return _fallback_to_paddle(capability.engine_id)

    try:
        handle = engine.get_handle(lang_code)
        lines = engine.run(handle, image_np, lang_code)
    except Exception as exc:
        logger.warning(
            "pluggable engine %r failed (%s); using paddle default", name, exc
        )
        return _fallback_to_paddle(capability.engine_id)

    _emit(
        build_engine_provenance_event(
            page_id=event_page_id,
            capability=capability,
            policy=active_policy,
        )
    )
    logger.info("page OCR via pluggable engine %r (%d lines)", name, len(lines))
    return name, lines
