"""Custody event helpers for translation pipeline."""
from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_local.features.custody import CustodyChain


class ReasonCode(StrEnum):
    PRIVILEGE_BLOCKED = "PRIVILEGE_BLOCKED"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    COST_CEILING_EXCEEDED = "COST_CEILING_EXCEEDED"
    TENANT_POLICY = "TENANT_POLICY"
    RESIDENCY_VIOLATION = "RESIDENCY_VIOLATION"
    MISSING_MFA = "MISSING_MFA"
    # Model-cache lifecycle (Plan B Wave M2).
    MODEL_DOWNLOADED = "MODEL_DOWNLOADED"
    MODEL_INTEGRITY_VERIFIED = "MODEL_INTEGRITY_VERIFIED"
    MODEL_INTEGRITY_FAILED = "MODEL_INTEGRITY_FAILED"
    MODEL_EVICTED = "MODEL_EVICTED"
    MODEL_LOAD_BLOCKED_NC_LICENSE = "MODEL_LOAD_BLOCKED_NC_LICENSE"
    MODEL_PINNED = "MODEL_PINNED"
    MODEL_UNPINNED = "MODEL_UNPINNED"
    # Tenant-aware routing (Plan B Wave M2 PR B14 -- router_v2).
    TENANT_ENGINE_SELECTED = "TENANT_ENGINE_SELECTED"
    TENANT_ENGINE_NO_CANDIDATES = "TENANT_ENGINE_NO_CANDIDATES"
    TENANT_POLICY_HYDRATED = "TENANT_POLICY_HYDRATED"
    # Batch translation scheduling (Plan B Wave M2 -- B17).
    BATCH_SUBMITTED = "BATCH_SUBMITTED"
    BATCH_FAN_OUT = "BATCH_FAN_OUT"
    BATCH_INPUT_COMPLETED = "BATCH_INPUT_COMPLETED"
    BATCH_INPUT_FAILED = "BATCH_INPUT_FAILED"
    BATCH_CANCELLED = "BATCH_CANCELLED"
    BATCH_COMPLETED = "BATCH_COMPLETED"
    BATCH_REJECTED_CERTIFIED = "BATCH_REJECTED_CERTIFIED"
    # Quality estimation (Plan B Wave 3 -- B15/B17).
    # Promoted to a dedicated reason code so quality_custody can drop the
    # TRANSLATION_COMPLETED fallback branch (see ocr_local/translation/
    # quality_custody.py: _resolve_quality_reason).
    QUALITY_ESTIMATED = "QUALITY_ESTIMATED"


def emit_translation_applied(
    chain: "CustodyChain",
    *,
    engine_id: str,
    src: str,
    tgt: str,
    span_count: int,
    char_count: int,
    tenant_id: str,
    model_id: str,
    weights_sha256: str,
    **extra,
) -> None:
    """Emit TRANSLATION_APPLIED custody event."""
    chain.log_event(
        "TRANSLATION_APPLIED",
        {
            "engine_id": engine_id,
            "src_lang": src,
            "tgt_lang": tgt,
            "span_count": span_count,
            "char_count": char_count,
            "tenant_id": tenant_id,
            "model_id": model_id,
            "weights_sha256": weights_sha256,
            **extra,
        },
    )


def emit_translation_rejected(
    chain: "CustodyChain",
    *,
    reason_code: ReasonCode,
    engine_id: str | None = None,
    message: str = "",
    tenant_id: str = "unknown",
    **extra,
) -> None:
    """Emit TRANSLATION_REJECTED custody event.

    MUST be called BEFORE raising :class:`PolicyDenied` so the audit trail
    captures the rejection even when callers swallow or re-raise the
    exception.
    """
    chain.log_event(
        "TRANSLATION_REJECTED",
        {
            "reason_code": str(reason_code),
            "engine_id": engine_id,
            "message": message,
            "tenant_id": tenant_id,
            **extra,
        },
    )


def emit_translation_fallback(
    chain: "CustodyChain",
    *,
    from_engine: str,
    to_engine: str,
    reason: str,
    **extra,
) -> None:
    chain.log_event(
        "TRANSLATION_FALLBACK",
        {
            "from_engine": from_engine,
            "to_engine": to_engine,
            "reason": reason,
            **extra,
        },
    )


def emit_quality_below_threshold(
    chain: "CustodyChain",
    *,
    engine_id: str,
    score: float,
    threshold: float,
    span_id: str,
    **extra,
) -> None:
    chain.log_event(
        "QUALITY_BELOW_THRESHOLD",
        {
            "engine_id": engine_id,
            "score": score,
            "threshold": threshold,
            "span_id": span_id,
            **extra,
        },
    )


def emit_glossary_applied(
    chain: "CustodyChain",
    *,
    glossary_id: str,
    glossary_hash: str,
    hit_count: int,
    **extra,
) -> None:
    chain.log_event(
        "GLOSSARY_APPLIED",
        {
            "glossary_id": glossary_id,
            "glossary_hash": glossary_hash,
            "hit_count": hit_count,
            **extra,
        },
    )


_ALLOWED_REVIEW_AUTH = frozenset({"piv_cac", "oidc_mfa", "hardware_token"})


def emit_translation_reviewed(
    chain: "CustodyChain",
    *,
    reviewer_id: str,
    auth_method: str,
    decision: str,
    job_id: str,
    **extra,
) -> None:
    """Emit TRANSLATION_REVIEWED.

    ``decision`` is typically ``"approve"`` or ``"reject"``.  ``auth_method``
    must be one of the strong-auth methods (PIV/CAC, OIDC+MFA, hardware
    token); password-based auth is rejected with :class:`ValueError`.
    """
    certified = bool(extra.pop("certified", False))
    tsa_client = extra.pop("tsa_client", None)
    tsa_store_dir = extra.pop("tsa_store_dir", None)
    if auth_method not in _ALLOWED_REVIEW_AUTH:
        raise ValueError(
            f"auth_method must be one of {sorted(_ALLOWED_REVIEW_AUTH)}, "
            f"got {auth_method!r}"
        )
    payload = {
        "reviewer_id": reviewer_id,
        "auth_method": auth_method,
        "decision": decision,
        "job_id": job_id,
        "certified": certified,
        **extra,
    }
    if certified:
        if tsa_client is None or tsa_store_dir is None:
            from ocr_local.translation.tsa import TSATimestampUnavailable

            raise TSATimestampUnavailable(
                "certified translation review requires tsa_client and tsa_store_dir"
            )
        try:
            from ocr_local.translation.tsa import anchor_certification_payload

            anchor = anchor_certification_payload(
                payload,
                tsa_client=tsa_client,
                tsa_store_dir=tsa_store_dir,
            )
        except Exception as exc:
            chain.log_event(
                "CUSTODY_TSA_WARNING",
                {
                    "job_id": job_id,
                    "reviewer_id": reviewer_id,
                    "decision": decision,
                    "certification_refused": True,
                    "error_class": exc.__class__.__name__,
                    "error_msg_truncated": str(exc)[:200],
                },
            )
            raise
        payload.update(
            {
                "clock_source": anchor.clock_source,
                "rfc3161_tsr_sha256": anchor.rfc3161_tsr_sha256,
                "tsa_request_sha256": anchor.tsa_request_sha256,
            }
        )
    chain.log_event(
        "TRANSLATION_REVIEWED",
        payload,
    )


def emit_translation_skipped(
    chain: "CustodyChain",
    *,
    reason: str,
    tenant_id: str = "unknown",
    **extra,
) -> None:
    chain.log_event(
        "TRANSLATION_SKIPPED",
        {"reason": reason, "tenant_id": tenant_id, **extra},
    )


def emit_tenant_engine_selected(
    chain: "CustodyChain",
    *,
    tenant_id: str,
    engine_id: str,
    model_id: str,
    source_lang: str,
    target_lang: str,
    **extra,
) -> None:
    """Emit TENANT_ENGINE_SELECTED -- tenant-aware router successfully selected an engine."""
    chain.log_event(
        "TENANT_ENGINE_SELECTED",
        {
            "reason_code": str(ReasonCode.TENANT_ENGINE_SELECTED),
            "tenant_id": tenant_id,
            "engine_id": engine_id,
            "model_id": model_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            **extra,
        },
    )


def emit_tenant_engine_no_candidates(
    chain: "CustodyChain",
    *,
    tenant_id: str,
    source_lang: str,
    target_lang: str,
    filter_reasons: list[str],
    **extra,
) -> None:
    """Emit TENANT_ENGINE_NO_CANDIDATES -- no eligible engine after filters.

    MUST be called BEFORE raising :class:`NoEligibleEngineError` (gotcha #87
    pattern: custody event always lands in the audit log, even when callers
    swallow the exception).
    """
    chain.log_event(
        "TENANT_ENGINE_NO_CANDIDATES",
        {
            "reason_code": str(ReasonCode.TENANT_ENGINE_NO_CANDIDATES),
            "tenant_id": tenant_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "filter_reasons": list(filter_reasons),
            **extra,
        },
    )


def emit_tenant_policy_hydrated(
    chain: "CustodyChain",
    *,
    tenant_id: str,
    policy_hash: str,
    **extra,
) -> None:
    """Emit TENANT_POLICY_HYDRATED once per tenant_id resolution.

    The ``policy_hash`` is the SHA-256 of the canonical policy JSON
    (see :func:`ocr_local.translation.policy.compute_policy_hash`) so
    reviewers can prove which policy version drove the routing decision.
    """
    chain.log_event(
        "TENANT_POLICY_HYDRATED",
        {
            "reason_code": str(ReasonCode.TENANT_POLICY_HYDRATED),
            "tenant_id": tenant_id,
            "policy_hash": policy_hash,
            **extra,
        },
    )


# ---------------------------------------------------------------------------
# Batch translation events (Plan B Wave M2 -- B17)
# ---------------------------------------------------------------------------


def emit_batch_submitted(
    chain: "CustodyChain",
    *,
    batch_id: str,
    tenant_id: str,
    source_lang: str,
    target_lang: str,
    input_count: int,
    **extra,
) -> None:
    """Emit BATCH_SUBMITTED -- batch accepted and persisted."""
    chain.log_event(
        "BATCH_SUBMITTED",
        {
            "reason_code": str(ReasonCode.BATCH_SUBMITTED),
            "batch_id": batch_id,
            "tenant_id": tenant_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "input_count": input_count,
            **extra,
        },
    )


def emit_batch_fan_out(
    chain: "CustodyChain",
    *,
    batch_id: str,
    dispatched_count: int,
    fan_out_size: int,
    **extra,
) -> None:
    """Emit BATCH_FAN_OUT -- subtasks dispatched to translation_batch queue."""
    chain.log_event(
        "BATCH_FAN_OUT",
        {
            "reason_code": str(ReasonCode.BATCH_FAN_OUT),
            "batch_id": batch_id,
            "dispatched_count": dispatched_count,
            "fan_out_size": fan_out_size,
            **extra,
        },
    )


def emit_batch_input_completed(
    chain: "CustodyChain",
    *,
    batch_id: str,
    client_ref: str,
    engine_id: str,
    char_count: int,
    **extra,
) -> None:
    """Emit BATCH_INPUT_COMPLETED -- a single input finished translation."""
    chain.log_event(
        "BATCH_INPUT_COMPLETED",
        {
            "reason_code": str(ReasonCode.BATCH_INPUT_COMPLETED),
            "batch_id": batch_id,
            "client_ref": client_ref,
            "engine_id": engine_id,
            "char_count": char_count,
            **extra,
        },
    )


def emit_batch_input_failed(
    chain: "CustodyChain",
    *,
    batch_id: str,
    client_ref: str,
    error: str,
    **extra,
) -> None:
    """Emit BATCH_INPUT_FAILED -- a single input failed after retries."""
    chain.log_event(
        "BATCH_INPUT_FAILED",
        {
            "reason_code": str(ReasonCode.BATCH_INPUT_FAILED),
            "batch_id": batch_id,
            "client_ref": client_ref,
            "error": error,
            **extra,
        },
    )


def emit_batch_cancelled(
    chain: "CustodyChain",
    *,
    batch_id: str,
    revoked_count: int,
    **extra,
) -> None:
    """Emit BATCH_CANCELLED -- batch cancelled by tenant request."""
    chain.log_event(
        "BATCH_CANCELLED",
        {
            "reason_code": str(ReasonCode.BATCH_CANCELLED),
            "batch_id": batch_id,
            "revoked_count": revoked_count,
            **extra,
        },
    )


def emit_batch_completed(
    chain: "CustodyChain",
    *,
    batch_id: str,
    completed_count: int,
    failed_count: int,
    **extra,
) -> None:
    """Emit BATCH_COMPLETED -- all inputs reached a terminal state."""
    chain.log_event(
        "BATCH_COMPLETED",
        {
            "reason_code": str(ReasonCode.BATCH_COMPLETED),
            "batch_id": batch_id,
            "completed_count": completed_count,
            "failed_count": failed_count,
            **extra,
        },
    )


def emit_batch_rejected_certified(
    chain: "CustodyChain",
    *,
    tenant_id: str,
    source_lang: str,
    target_lang: str,
    **extra,
) -> None:
    """Emit BATCH_REJECTED_CERTIFIED -- submission requested certified=True.

    MUST be called BEFORE raising :class:`PolicyDenied` (gotcha #87).
    Certification can only be set via the review queue after a strong-auth
    custody event; the submit path always rejects ``certified=True``.
    """
    chain.log_event(
        "BATCH_REJECTED_CERTIFIED",
        {
            "reason_code": str(ReasonCode.BATCH_REJECTED_CERTIFIED),
            "tenant_id": tenant_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            **extra,
        },
    )
