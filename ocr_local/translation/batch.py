"""Batch translation scheduling --   (B17).

Provides the public API for tenants to submit large batches of short
text inputs (or already-OCR'd documents) for asynchronous translation,
fan-out across CT2 engines via Celery, and collect results.

Design notes
------------

* Persistence lives in the coordinator's PostgreSQL via the
  ``BatchTranslationJob`` and ``BatchTranslationInput`` Django models
  (migration ``0012_batch_translation``).  Django imports are lazy so
  that ``ocr_local.translation.batch`` remains importable in non-Django
  contexts (the standalone OCR pipeline, the SDK CI lane).
* Glossary preprocessing runs **per-input** before engine selection in
  the Celery task body, mirroring the existing pattern.
* ``requested_certified=True`` is rejected at submit time -- the
  ``BATCH_REJECTED_CERTIFIED`` custody event is emitted BEFORE the
  ``PolicyDenied`` raise (gotcha #87).
* All public dataclasses are stdlib-only so the unit tests can run
  without Django, Celery, or the API stack.
* Cancellation issues a Celery ``revoke`` to each pending task and
  flips the batch status to ``cancelled``.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import timezone
from typing import TYPE_CHECKING, Optional

from ocr_local.translation.custody_adapter import (
    ReasonCode,
    emit_batch_cancelled,
    emit_batch_fan_out,
    emit_batch_rejected_certified,
    emit_batch_submitted,
)
from ocr_local.translation.policy import PolicyDenied

if TYPE_CHECKING:
    from ocr_local.features.custody import CustodyChain

logger = logging.getLogger(__name__)

__all__ = [
    "BatchInput",
    "BatchTranslationRequest",
    "BatchTranslationResult",
    "BatchStatusSnapshot",
    "submit_batch",
    "fan_out",
    "collect_results",
    "get_status",
    "cancel_batch",
    "BatchValidationError",
    "BatchNotFoundError",
]


# Hard cap that protects the call site even if pipeline_config is not
# yet wired -- callers that need a stricter cap pass it explicitly.
DEFAULT_MAX_INPUTS = 1000
DEFAULT_INPUT_MAX_BYTES = 8 * 1024
DEFAULT_FAN_OUT_SIZE = 32


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


class BatchValidationError(ValueError):
    """Raised when a :class:`BatchTranslationRequest` violates its invariants."""


class BatchNotFoundError(LookupError):
    """Raised when a batch_id has no row in the coordinator DB."""


@dataclass(frozen=True)
class BatchInput:
    """One input within a batch translation request.

    ``client_ref`` is a tenant-supplied opaque identifier that must be
    unique within the request -- :func:`submit_batch` enforces dedup
    and raises :class:`BatchValidationError` if duplicates are seen.
    """

    client_ref: str
    text: str
    optional_metadata: Optional[dict] = None


@dataclass(frozen=True)
class BatchTranslationRequest:
    """Tenant-submitted batch translation request.

    ``requested_certified`` MUST default to False; setting it to True
    raises :class:`PolicyDenied` immediately because certification can
    only be assigned by the review queue after a strong-auth event
    (gotcha #88).
    """

    tenant_id: str
    source_lang: str
    target_lang: str
    inputs: list[BatchInput]
    priority: int = 0
    glossary_enabled: bool = True
    requested_certified: bool = False


@dataclass(frozen=True)
class BatchTranslationResult:
    """Result of translating a single :class:`BatchInput`."""

    client_ref: str
    target_text: str
    engine_id: str
    confidence: Optional[float]
    glossary_hits: list
    error: Optional[str]


@dataclass(frozen=True)
class BatchStatusSnapshot:
    """Lightweight status payload returned by :func:`get_status`."""

    batch_id: str
    tenant_id: str
    source_lang: str
    target_lang: str
    status: str
    total_inputs: int
    completed_inputs: int
    failed_inputs: int
    pending_inputs: int
    running_inputs: int
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Lazy Django import helpers
# ---------------------------------------------------------------------------


def _load_django_models:
    """Lazy import of the coordinator Django models.

    Returns ``(BatchTranslationJob, BatchTranslationInput)`` or raises
    ``RuntimeError`` when Django settings are not configured.  Callers
    in non-Django contexts (the SDK CI lane, unit tests without
    DJANGO_SETTINGS_MODULE) should patch :func:`_load_django_models`
    or call helpers that accept an injected backend.
    """
    try:
        from jobs.models import (  # type: ignore[import-not-found]
            BatchTranslationInput,
            BatchTranslationJob,
        )
    except Exception as exc:  # pragma: no cover - non-Django fallback
        raise RuntimeError(
            "ocr_local.translation.batch requires Django to be configured "
            "(jobs.models.BatchTranslationJob unavailable: %s)" % exc
        ) from exc
    return BatchTranslationJob, BatchTranslationInput


# ---------------------------------------------------------------------------
# submit_batch / validation
# ---------------------------------------------------------------------------


def _validate_request(
    req: BatchTranslationRequest,
    *,
    max_inputs: int,
    input_max_bytes: int,
) -> None:
    """Validate ``req`` invariants -- raise on violation, return None on OK."""
    if not req.tenant_id:
        raise BatchValidationError("tenant_id is required")
    if not req.source_lang or not req.target_lang:
        raise BatchValidationError(
            "source_lang and target_lang are required"
        )
    if not isinstance(req.inputs, list) or not req.inputs:
        raise BatchValidationError("inputs must be a non-empty list")
    if len(req.inputs) > max_inputs:
        raise BatchValidationError(
            f"inputs ({len(req.inputs)}) exceeds max_inputs ({max_inputs})"
        )
    if not isinstance(req.priority, int) or req.priority < 0 or req.priority > 9:
        raise BatchValidationError(
            f"priority must be 0..9, got {req.priority!r}"
        )

    seen_refs: set[str] = set
    for idx, inp in enumerate(req.inputs):
        if not isinstance(inp, BatchInput):
            raise BatchValidationError(
                f"inputs[{idx}] must be BatchInput, got "
                f"{type(inp).__name__}"
            )
        if not inp.client_ref:
            raise BatchValidationError(
                f"inputs[{idx}].client_ref is required"
            )
        if inp.client_ref in seen_refs:
            raise BatchValidationError(
                f"inputs[{idx}].client_ref={inp.client_ref!r} "
                "duplicates an earlier entry"
            )
        seen_refs.add(inp.client_ref)
        if not isinstance(inp.text, str):
            raise BatchValidationError(
                f"inputs[{idx}].text must be str, got "
                f"{type(inp.text).__name__}"
            )
        text_bytes = len(inp.text.encode("utf-8"))
        if text_bytes > input_max_bytes:
            raise BatchValidationError(
                f"inputs[{idx}].text is {text_bytes} bytes, exceeds "
                f"input_max_bytes ({input_max_bytes})"
            )


def submit_batch(
    req: BatchTranslationRequest,
    *,
    max_inputs: int = DEFAULT_MAX_INPUTS,
    input_max_bytes: int = DEFAULT_INPUT_MAX_BYTES,
    custody_chain: "CustodyChain | None" = None,
) -> str:
    """Persist a new batch and return ``batch_id``.

    Validation order (matters for tests / audit trail):

    1. Reject ``requested_certified=True`` -- emit
       ``BATCH_REJECTED_CERTIFIED`` BEFORE raising :class:`PolicyDenied`.
    2. Run input invariants (max inputs, dedup, size cap).
    3. Persist ``BatchTranslationJob`` + ``BatchTranslationInput`` rows.
    4. Emit ``BATCH_SUBMITTED`` custody event.
    """
    if req.requested_certified:
        if custody_chain is not None:
            try:
                emit_batch_rejected_certified(
                    custody_chain,
                    tenant_id=req.tenant_id,
                    source_lang=req.source_lang,
                    target_lang=req.target_lang,
                )
            except Exception:
                logger.debug(
                    "submit_batch: custody emit failed (non-fatal)",
                    exc_info=True,
                )
        raise PolicyDenied(
            ReasonCode.BATCH_REJECTED_CERTIFIED,
            "requested_certified=True is rejected at submit time; "
            "certification requires the review queue strong-auth path",
        )

    _validate_request(
        req, max_inputs=max_inputs, input_max_bytes=input_max_bytes,
    )

    BatchTranslationJob, BatchTranslationInput = _load_django_models
    batch_id = uuid.uuid4.hex

    # Persist atomically -- the coordinator default DB is PostgreSQL,
    # but tests may run on SQLite via Django settings overrides.
    from django.db import transaction  # type: ignore[import-not-found]

    with transaction.atomic:
        job_row = BatchTranslationJob.objects.create(
            batch_id=batch_id,
            tenant_id=req.tenant_id,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            status=BatchTranslationJob.STATUS_PENDING,
            priority=req.priority,
            glossary_enabled=req.glossary_enabled,
            total_inputs=len(req.inputs),
            completed_inputs=0,
            failed_inputs=0,
        )
        for idx, inp in enumerate(req.inputs):
            BatchTranslationInput.objects.create(
                batch=job_row,
                client_ref=inp.client_ref,
                input_index=idx,
                text=inp.text,
                status='pending',
                optional_metadata_json=dict(inp.optional_metadata or {}),
            )

    if custody_chain is not None:
        try:
            emit_batch_submitted(
                custody_chain,
                batch_id=batch_id,
                tenant_id=req.tenant_id,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                input_count=len(req.inputs),
            )
        except Exception:
            logger.debug(
                "submit_batch: BATCH_SUBMITTED custody emit failed (non-fatal)",
                exc_info=True,
            )

    return batch_id


# ---------------------------------------------------------------------------
# fan_out
# ---------------------------------------------------------------------------


def fan_out(
    batch_id: str,
    *,
    fan_out_size: int = DEFAULT_FAN_OUT_SIZE,
    custody_chain: "CustodyChain | None" = None,
) -> int:
    """Dispatch Celery subtasks for every pending input in ``batch_id``.

    Returns the count of subtasks dispatched.  Flips the batch status
    from ``pending`` to ``running``.  Emits a single ``BATCH_FAN_OUT``
    custody event with ``dispatched_count`` and ``fan_out_size``.
    """
    if fan_out_size <= 0:
        raise ValueError(f"fan_out_size must be >= 1, got {fan_out_size}")

    BatchTranslationJob, BatchTranslationInput = _load_django_models

    try:
        job_row = BatchTranslationJob.objects.get(batch_id=batch_id)
    except BatchTranslationJob.DoesNotExist as exc:
        raise BatchNotFoundError(batch_id) from exc

    if job_row.status in BatchTranslationJob.TERMINAL_STATUSES:
        # Idempotent no-op for terminal states.
        return 0

    # Lazy Celery import so ``ocr_local.translation.batch`` is importable
    # in environments without Celery (the SDK lane, pure unit tests).
    try:
        from coordinator.jobs.tasks_translation_batch import (  # type: ignore[import-not-found]
            translate_batch_input,
        )
    except Exception as coordinator_exc:
        try:
            from jobs.tasks_translation_batch import translate_batch_input  # type: ignore[import-not-found]
        except Exception as jobs_exc:
            raise RuntimeError(
                "Celery task translate_batch_input unavailable: "
                f"coordinator.jobs import failed: {coordinator_exc}; jobs import failed: {jobs_exc}"
            ) from jobs_exc

    pending_inputs = list(
        BatchTranslationInput.objects.filter(
            batch=job_row, status='pending'
        ).order_by('input_index')
    )

    dispatched = 0
    for chunk_start in range(0, len(pending_inputs), fan_out_size):
        chunk = pending_inputs[chunk_start:chunk_start + fan_out_size]
        for inp_row in chunk:
            try:
                async_result = translate_batch_input.apply_async(
                    kwargs={
                        'batch_id': batch_id,
                        'client_ref': inp_row.client_ref,
                        'text': inp_row.text,
                        'source_lang': job_row.source_lang,
                        'target_lang': job_row.target_lang,
                        'tenant_id': job_row.tenant_id,
                        'glossary_enabled': job_row.glossary_enabled,
                    },
                    queue='translation_batch',
                    priority=job_row.priority,
                )
                inp_row.celery_task_id = async_result.id
                inp_row.status = 'running'
                inp_row.save(update_fields=['celery_task_id', 'status'])
                dispatched += 1
            except Exception as exc:
                logger.warning(
                    "fan_out: failed to dispatch task for batch=%s ref=%s: %s",
                    batch_id, inp_row.client_ref, exc,
                )

    if dispatched > 0:
        job_row.status = BatchTranslationJob.STATUS_RUNNING
        job_row.save(update_fields=['status'])

    if custody_chain is not None:
        try:
            emit_batch_fan_out(
                custody_chain,
                batch_id=batch_id,
                dispatched_count=dispatched,
                fan_out_size=fan_out_size,
            )
        except Exception:
            logger.debug(
                "fan_out: custody emit failed (non-fatal)", exc_info=True,
            )

    return dispatched


# ---------------------------------------------------------------------------
# get_status / collect_results
# ---------------------------------------------------------------------------


def _isoformat(dt) -> Optional[str]:
    if dt is None:
        return None
    try:
        return dt.astimezone(timezone.utc).isoformat
    except Exception:
        try:
            return dt.isoformat
        except Exception:
            return None


def get_status(batch_id: str) -> BatchStatusSnapshot:
    """Return a :class:`BatchStatusSnapshot` for ``batch_id``."""
    BatchTranslationJob, BatchTranslationInput = _load_django_models
    try:
        job_row = BatchTranslationJob.objects.get(batch_id=batch_id)
    except BatchTranslationJob.DoesNotExist as exc:
        raise BatchNotFoundError(batch_id) from exc

    counts = (
        BatchTranslationInput.objects
        .filter(batch=job_row)
        .values_list('status', flat=True)
    )
    pending = running = completed = failed = 0
    for s in counts:
        if s == 'pending':
            pending += 1
        elif s == 'running':
            running += 1
        elif s == 'completed':
            completed += 1
        elif s == 'failed':
            failed += 1

    return BatchStatusSnapshot(
        batch_id=batch_id,
        tenant_id=job_row.tenant_id,
        source_lang=job_row.source_lang,
        target_lang=job_row.target_lang,
        status=job_row.status,
        total_inputs=job_row.total_inputs,
        completed_inputs=completed,
        failed_inputs=failed,
        pending_inputs=pending,
        running_inputs=running,
        submitted_at=_isoformat(job_row.submitted_at),
        completed_at=_isoformat(job_row.completed_at),
    )


def collect_results(batch_id: str) -> list[BatchTranslationResult]:
    """Return per-input results in original input order.

    Returns results regardless of batch status -- callers needing
    "only when terminal" semantics should consult :func:`get_status`
    first and gate on ``status in {completed, failed, cancelled}``.
    """
    BatchTranslationJob, BatchTranslationInput = _load_django_models
    try:
        job_row = BatchTranslationJob.objects.get(batch_id=batch_id)
    except BatchTranslationJob.DoesNotExist as exc:
        raise BatchNotFoundError(batch_id) from exc

    rows = (
        BatchTranslationInput.objects
        .filter(batch=job_row)
        .order_by('input_index')
    )
    return [
        BatchTranslationResult(
            client_ref=r.client_ref,
            target_text=r.target_text or "",
            engine_id=r.engine_id or "",
            confidence=r.confidence,
            glossary_hits=list(r.glossary_hits_json or []),
            error=(r.error or None) if r.error else None,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# cancel_batch
# ---------------------------------------------------------------------------


def cancel_batch(
    batch_id: str,
    *,
    custody_chain: "CustodyChain | None" = None,
) -> int:
    """Cancel a batch.

    For each pending or running input, issues a Celery ``revoke`` and
    marks the row as ``cancelled``.  Terminal-state batches are no-ops
    that return 0 without raising.

    Returns the number of inputs that were revoked.
    """
    BatchTranslationJob, BatchTranslationInput = _load_django_models
    try:
        job_row = BatchTranslationJob.objects.get(batch_id=batch_id)
    except BatchTranslationJob.DoesNotExist as exc:
        raise BatchNotFoundError(batch_id) from exc

    if job_row.status in BatchTranslationJob.TERMINAL_STATUSES:
        return 0

    # Lazy Celery import -- when the broker is unreachable we still
    # update DB rows to cancelled so ``get_status`` reflects the cancel.
    revoke = None
    try:
        from coordinator.coordinator.celery import app as _celery_app
        revoke = _celery_app.control.revoke
    except Exception as exc:
        logger.warning(
            "cancel_batch: Celery app unavailable, skipping revoke: %s", exc,
        )

    revoked = 0
    pending_or_running = list(
        BatchTranslationInput.objects.filter(
            batch=job_row, status__in=['pending', 'running'],
        )
    )
    for inp_row in pending_or_running:
        if revoke is not None and inp_row.celery_task_id:
            try:
                revoke(inp_row.celery_task_id, terminate=False)
            except Exception as exc:
                logger.debug(
                    "cancel_batch: revoke failed for task=%s: %s",
                    inp_row.celery_task_id, exc,
                )
        inp_row.status = 'cancelled'
        inp_row.error = inp_row.error or 'cancelled'
        inp_row.save(update_fields=['status', 'error'])
        revoked += 1

    job_row.status = BatchTranslationJob.STATUS_CANCELLED
    job_row.save(update_fields=['status'])

    if custody_chain is not None:
        try:
            emit_batch_cancelled(
                custody_chain,
                batch_id=batch_id,
                revoked_count=revoked,
            )
        except Exception:
            logger.debug(
                "cancel_batch: custody emit failed (non-fatal)", exc_info=True,
            )

    return revoked


# ---------------------------------------------------------------------------
# Convenience helpers exposed for the Celery task body
# ---------------------------------------------------------------------------


def to_dict(req: BatchTranslationRequest) -> dict:
    """Return a JSON-friendly dict of ``req`` (used by the API router)."""
    return {
        "tenant_id": req.tenant_id,
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
        "priority": req.priority,
        "glossary_enabled": req.glossary_enabled,
        "requested_certified": req.requested_certified,
        "inputs": [
            {
                "client_ref": inp.client_ref,
                "text": inp.text,
                "optional_metadata": dict(inp.optional_metadata or {}),
            }
            for inp in req.inputs
        ],
    }


def from_dict(data: dict) -> BatchTranslationRequest:
    """Build a :class:`BatchTranslationRequest` from a JSON-friendly dict."""
    inputs = [
        BatchInput(
            client_ref=str(item["client_ref"]),
            text=str(item.get("text", "")),
            optional_metadata=item.get("optional_metadata"),
        )
        for item in (data.get("inputs") or [])
    ]
    return BatchTranslationRequest(
        tenant_id=str(data["tenant_id"]),
        source_lang=str(data["source_lang"]),
        target_lang=str(data["target_lang"]),
        inputs=inputs,
        priority=int(data.get("priority", 0)),
        glossary_enabled=bool(data.get("glossary_enabled", True)),
        requested_certified=bool(data.get("requested_certified", False)),
    )
