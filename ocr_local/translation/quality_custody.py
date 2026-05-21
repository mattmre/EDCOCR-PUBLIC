"""Quality-estimation custody helpers -- Plan B Wave 3 (B15).

This module emits custody events for the COMETKiwi quality-estimation
pipeline.  It deliberately does NOT modify
``ocr_local.translation.custody_adapter`` because B17 owns adding new
:class:`ReasonCode` values during Wave 3.  Instead, this helper emits via
the existing :class:`ReasonCode.QUALITY_ESTIMATED` value when present;
when not present (Wave 3 ordering hasn't landed yet), it falls back to
``ReasonCode.TRANSLATION_COMPLETED`` and stuffs the QE payload into the
event ``data`` dict.

# TODO: Wave 4: promote QUALITY_ESTIMATED to a dedicated ReasonCode
# after B17 lands.  At that point, the fallback branch below can be
# removed and the event_type can be tightened to a single string.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ocr_local.translation.custody_adapter import ReasonCode

if TYPE_CHECKING:
    from ocr_local.features.custody import CustodyChain
    from ocr_local.translation.quality_estimation import (
        DocumentQualityReport,
    )

logger = logging.getLogger(__name__)

__all__ = ["emit_quality_estimated"]


def _resolve_quality_reason() -> tuple[str, bool]:
    """Return ``(reason_code_string, is_dedicated)``.

    ``is_dedicated`` is True when the dedicated
    ``QUALITY_ESTIMATED`` reason code exists in the current
    ``ReasonCode`` enum (i.e. B17 has landed).  When False, callers
    should treat the result as a generic completion event and place the
    QE payload under ``data``.
    """
    dedicated = getattr(ReasonCode, "QUALITY_ESTIMATED", None)
    if dedicated is not None:
        return str(dedicated), True
    # Fallback: B17 hasn't added the new ReasonCode yet -- piggyback on
    # the closest existing event so the QE outcome still lands in the
    # audit log.
    fallback = getattr(ReasonCode, "TRANSLATION_COMPLETED", None)
    if fallback is not None:
        return str(fallback), False
    # Last-resort: synthetic string so we never crash a translation run.
    return "TRANSLATION_QUALITY_ESTIMATED", False


def emit_quality_estimated(
    chain: "CustodyChain",
    report: "DocumentQualityReport",
    *,
    document_id: str,
    tenant_id: str = "unknown",
    target_language: str = "",
    quality_reason: str | None = None,
    **extra: object,
) -> None:
    """Emit the QE custody event.

    Parameters
    ----------
    chain:
        Hash-chained custody log instance.  When ``None`` the call is a
        no-op (the caller is expected to guard, but we re-check defensively).
    report:
        The aggregated :class:`DocumentQualityReport`.
    document_id:
        Document identifier so the event can be correlated against the
        upstream OCR job.
    tenant_id:
        Tenant identifier.  Defaults to ``"unknown"`` to mirror the
        existing custody helpers.
    target_language:
        BCP-47 target language for the translation that was scored.
    quality_reason:
        Optional human-readable note (e.g. ``"below_warn_threshold"``).
    extra:
        Forwarded to the chain as additional event payload keys.

    Notes
    -----
    Custody emission MUST never raise; failures are logged at DEBUG
    level only.  This mirrors the fail-open contract the rest of the
    translation pipeline follows (cf. ).
    """
    if chain is None:
        return

    reason_code, is_dedicated = _resolve_quality_reason()

    payload = {
        "reason_code": reason_code,
        "document_id": document_id,
        "tenant_id": tenant_id,
        "target_language": target_language,
        "model_id": report.model_id,
        "score_mean": report.score_mean,
        "score_min": report.score_min,
        "threshold_warn": report.threshold_warn,
        "threshold_reject": report.threshold_reject,
        "threshold_warn_count": report.threshold_warn_count,
        "threshold_reject_count": report.threshold_reject_count,
        "scored_count": report.scored_count,
        "span_count": report.span_count,
        **extra,
    }
    if quality_reason is not None:
        payload["quality_reason"] = quality_reason
    if not is_dedicated:
        # When piggybacking on TRANSLATION_COMPLETED, surface the actual
        # event class via ``data`` so downstream consumers can filter.
        payload["event_subtype"] = "QUALITY_ESTIMATED"

    event_name = (
        "TRANSLATION_QUALITY_ESTIMATED" if is_dedicated else "TRANSLATION_COMPLETED"
    )
    try:
        chain.log_event(event_name, payload)
    except Exception:
        logger.debug(
            "emit_quality_estimated: chain.log_event failed (non-fatal)",
            exc_info=True,
        )
