"""Batch translation scheduling -- Plan B Wave M2 (B17).

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
  the Celery task body, mirroring the existing pattern from PR #794.
* ``requested_certified=True`` is rejected at submit time -- the
  ``BATCH_REJECTED_CERTIFIED`` custody event is emitted BEFORE the
  ``PolicyDenied`` raise (gotcha #87).
* All public dataclasses are stdlib-only so the unit tests can run
  without Django, Celery, or the API stack.
* Cancellation issues a Celery ``revoke`` to each pending task and
  flips the batch status to ``cancelled``.
"""

from __future__ import annotations

import json
import logging
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    "reclaim_stale_batch_inputs",
    "collect_results",
    "promote_batch_certified",
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
DEFAULT_MAX_RESULT_BYTES = 16 * 1024 * 1024

# Certification is a control-plane operation, so its evidence lookup must be
# bounded even when the optional JSONL fallback is pointed at a shared output
# directory.  These limits are deliberately independent of the normal
# custody-loader defaults: this path scans a directory rather than loading a
# caller-selected single chain.
_CUSTODY_SCAN_MAX_FILES = 10_000
_CUSTODY_SCAN_MAX_FILE_BYTES = 16 * 1024 * 1024
_CUSTODY_SCAN_MAX_LINE_BYTES = 256 * 1024


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


class BatchValidationError(ValueError):
    """Raised when a :class:`BatchTranslationRequest` violates its invariants."""


class BatchNotFoundError(LookupError):
    """Raised when a batch_id has no row in the coordinator DB."""


class _AllDispatchesFailed(RuntimeError):
    """Signal a broker outage only after durable failure state commits."""


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
    # S2-04: True only after review certify; delivery gate uses this.
    certified: bool = False


# ---------------------------------------------------------------------------
# Lazy Django import helpers
# ---------------------------------------------------------------------------


def _load_django_models():
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
        raise BatchValidationError("source_lang and target_lang are required")
    if not isinstance(req.inputs, list) or not req.inputs:
        raise BatchValidationError("inputs must be a non-empty list")
    if len(req.inputs) > max_inputs:
        raise BatchValidationError(
            f"inputs ({len(req.inputs)}) exceeds max_inputs ({max_inputs})"
        )
    if not isinstance(req.priority, int) or req.priority < 0 or req.priority > 9:
        raise BatchValidationError(f"priority must be 0..9, got {req.priority!r}")

    seen_refs: set[str] = set()
    for idx, inp in enumerate(req.inputs):
        if not isinstance(inp, BatchInput):
            raise BatchValidationError(
                f"inputs[{idx}] must be BatchInput, got {type(inp).__name__}"
            )
        if not inp.client_ref:
            raise BatchValidationError(f"inputs[{idx}].client_ref is required")
        if inp.client_ref in seen_refs:
            raise BatchValidationError(
                f"inputs[{idx}].client_ref={inp.client_ref!r} "
                "duplicates an earlier entry"
            )
        seen_refs.add(inp.client_ref)
        if not isinstance(inp.text, str):
            raise BatchValidationError(
                f"inputs[{idx}].text must be str, got {type(inp.text).__name__}"
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
        req,
        max_inputs=max_inputs,
        input_max_bytes=input_max_bytes,
    )

    BatchTranslationJob, BatchTranslationInput = _load_django_models()
    batch_id = uuid.uuid4().hex

    # Persist atomically -- the coordinator default DB is PostgreSQL,
    # but tests may run on SQLite via Django settings overrides.
    from django.db import transaction  # type: ignore[import-not-found]

    with transaction.atomic():
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
                status="pending",
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


def _mark_pending_dispatch_failed(
    *,
    job_row,
    batch_model,
    input_model,
    reason: str,
    terminal_batch: bool = True,
) -> int:
    """Terminally account for inputs that could not be put on the broker.

    A persisted ``pending`` row with no Celery task has no automatic recovery
    path. Marking it failed makes the outage observable and prevents clients
    from polling a batch forever while preserving the input/error audit row.
    """
    now = datetime.now(timezone.utc)
    error = f"dispatch_failed: {reason}"[:8000]
    pending = list(input_model.objects.filter(batch=job_row, status="pending"))
    for inp_row in pending:
        inp_row.status = "failed"
        inp_row.error = error
        inp_row.completed_at = now
        inp_row.save(update_fields=["status", "error", "completed_at"])
    if pending:
        from django.db.models import F  # type: ignore[import-not-found]

        batch_model.objects.filter(pk=job_row.pk).update(
            failed_inputs=F("failed_inputs") + len(pending)
        )
    if pending and terminal_batch:
        job_row.status = batch_model.STATUS_FAILED
        job_row.completed_at = now
        job_row.save(update_fields=["status", "completed_at"])
    return len(pending)


def fan_out(
    batch_id: str,
    *,
    fan_out_size: int = DEFAULT_FAN_OUT_SIZE,
    custody_chain: "CustodyChain | None" = None,
) -> int:
    """Claim one pending batch row, then dispatch it exactly once at a time."""
    if fan_out_size <= 0:
        raise ValueError(f"fan_out_size must be >= 1, got {fan_out_size}")
    BatchTranslationJob, _BatchTranslationInput = _load_django_models()
    from django.db import transaction  # type: ignore[import-not-found]

    dispatch_failure: _AllDispatchesFailed | None = None
    with transaction.atomic():
        try:
            job_row = BatchTranslationJob.objects.select_for_update().get(
                batch_id=batch_id
            )
        except BatchTranslationJob.DoesNotExist as exc:
            raise BatchNotFoundError(batch_id) from exc
        if job_row.status != BatchTranslationJob.STATUS_PENDING:
            return 0
        try:
            dispatched = _fan_out_claimed(
                batch_id,
                fan_out_size=fan_out_size,
                custody_chain=custody_chain,
            )
        except _AllDispatchesFailed as exc:
            # The exception describes terminal state written inside this
            # transaction.  Let the transaction commit before surfacing it.
            dispatch_failure = exc
            dispatched = 0
    if dispatch_failure is not None:
        raise dispatch_failure
    return dispatched


def _fan_out_claimed(
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

    BatchTranslationJob, BatchTranslationInput = _load_django_models()

    try:
        job_row = BatchTranslationJob.objects.get(batch_id=batch_id)
    except BatchTranslationJob.DoesNotExist as exc:
        raise BatchNotFoundError(batch_id) from exc

    if job_row.status in BatchTranslationJob.TERMINAL_STATUSES:
        # Idempotent no-op for terminal states.
        return 0

    pending_inputs = list(
        BatchTranslationInput.objects.filter(batch=job_row, status="pending").order_by(
            "input_index"
        )
    )

    # Lazy Celery import so ``ocr_local.translation.batch`` is importable
    # in environments without Celery (the SDK lane, pure unit tests).
    try:
        from coordinator.jobs.tasks_translation_batch import (  # type: ignore[import-not-found]
            translate_batch_input,
        )
    except Exception as coordinator_exc:
        try:
            from jobs.tasks_translation_batch import (
                translate_batch_input,  # type: ignore[import-not-found]
            )
        except Exception as jobs_exc:
            reason = (
                "Celery task translate_batch_input unavailable: "
                f"coordinator.jobs import failed: {coordinator_exc}; "
                f"jobs import failed: {jobs_exc}"
            )
            _mark_pending_dispatch_failed(
                job_row=job_row,
                batch_model=BatchTranslationJob,
                input_model=BatchTranslationInput,
                reason=reason,
            )
            raise _AllDispatchesFailed(reason) from jobs_exc

    dispatched = 0
    dispatch_errors: list[str] = []
    for chunk_start in range(0, len(pending_inputs), fan_out_size):
        chunk = pending_inputs[chunk_start : chunk_start + fan_out_size]
        for inp_row in chunk:
            try:
                current_status = BatchTranslationJob.objects.get(
                    batch_id=batch_id
                ).status
                if current_status in BatchTranslationJob.TERMINAL_STATUSES:
                    break
                async_result = translate_batch_input.apply_async(
                    kwargs={
                        "batch_id": batch_id,
                        "client_ref": inp_row.client_ref,
                        "text": inp_row.text,
                        "source_lang": job_row.source_lang,
                        "target_lang": job_row.target_lang,
                        "tenant_id": job_row.tenant_id,
                        "glossary_enabled": job_row.glossary_enabled,
                    },
                    queue="translation_batch",
                    priority=job_row.priority,
                )
                accepted = BatchTranslationInput.objects.filter(
                    batch=job_row,
                    client_ref=inp_row.client_ref,
                    status="pending",
                ).update(
                    celery_task_id=async_result.id,
                    status="running",
                    started_at=datetime.now(timezone.utc),
                )
                if accepted:
                    dispatched += 1
                else:
                    try:
                        async_result.revoke(terminate=False)
                    except Exception:
                        logger.warning(
                            "fan_out: could not revoke dispatch rejected by cancel fence; "
                            "batch=%s ref=%s",
                            batch_id,
                            inp_row.client_ref,
                            exc_info=True,
                        )
            except Exception as exc:
                dispatch_errors.append(str(exc))
                logger.warning(
                    "fan_out: failed to dispatch task for batch=%s ref=%s: %s",
                    batch_id,
                    inp_row.client_ref,
                    exc,
                )

    if dispatch_errors:
        _mark_pending_dispatch_failed(
            job_row=job_row,
            batch_model=BatchTranslationJob,
            input_model=BatchTranslationInput,
            reason=dispatch_errors[0],
            terminal_batch=dispatched == 0,
        )

    if dispatched > 0:
        BatchTranslationJob.objects.filter(
            batch_id=batch_id,
            status__in=[
                BatchTranslationJob.STATUS_PENDING,
                BatchTranslationJob.STATUS_RUNNING,
            ],
        ).update(status=BatchTranslationJob.STATUS_RUNNING)
        # An accepted task may finish before the dispatch loop has marked a
        # later broker failure. Its worker then observes a still-pending row
        # and cannot finalize. Close that race after all pending failures are
        # terminally accounted; the worker wins the same status CAS if later.
        outstanding = BatchTranslationInput.objects.filter(
            batch=job_row,
            status__in=["pending", "running"],
        ).count()
        if outstanding == 0:
            BatchTranslationJob.objects.filter(
                batch_id=batch_id,
                status__in=[
                    BatchTranslationJob.STATUS_PENDING,
                    BatchTranslationJob.STATUS_RUNNING,
                ],
            ).update(
                status=BatchTranslationJob.STATUS_COMPLETED,
                completed_at=datetime.now(timezone.utc),
            )

    if custody_chain is not None:
        try:
            emit_batch_fan_out(
                custody_chain,
                batch_id=batch_id,
                tenant_id=job_row.tenant_id,
                dispatched_count=dispatched,
                fan_out_size=fan_out_size,
                failed_dispatch_count=len(dispatch_errors),
            )
        except Exception:
            logger.debug(
                "fan_out: custody emit failed (non-fatal)",
                exc_info=True,
            )

    # No work was accepted by the broker.  The durable rows are already
    # marked failed above, so surface the outage to the HTTP caller instead
    # of returning a misleading 202/pending response.  Partial dispatch is
    # intentionally returned as a usable batch with per-input failures.
    if dispatch_errors and dispatched == 0:
        raise _AllDispatchesFailed(
            f"Translation batch dispatch failed for every input: {dispatch_errors[0]}"
        )

    return dispatched


def reclaim_stale_batch_inputs(batch_id: str, *, lease_seconds: int) -> int:
    """Return expired dispatched/running inputs to pending for safe fan-out."""
    if not isinstance(lease_seconds, int) or isinstance(lease_seconds, bool):
        raise ValueError("lease_seconds must be an integer")
    if not 1 <= lease_seconds <= 24 * 60 * 60:
        raise ValueError("lease_seconds must be between 1 and 86400")

    BatchTranslationJob, BatchTranslationInput = _load_django_models()
    from django.db import transaction  # type: ignore[import-not-found]
    from django.db.models import Q  # type: ignore[import-not-found]

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=lease_seconds)
    with transaction.atomic():
        try:
            job_row = BatchTranslationJob.objects.select_for_update().get(
                batch_id=batch_id
            )
        except BatchTranslationJob.DoesNotExist as exc:
            raise BatchNotFoundError(batch_id) from exc
        if job_row.status not in {
            BatchTranslationJob.STATUS_PENDING,
            BatchTranslationJob.STATUS_RUNNING,
        }:
            return 0
        reclaimed = BatchTranslationInput.objects.filter(
            Q(started_at__lte=cutoff) | Q(started_at__isnull=True),
            batch=job_row,
            status="running",
        ).update(
            status="pending",
            celery_task_id="",
            started_at=None,
        )
        if reclaimed:
            BatchTranslationJob.objects.filter(
                batch_id=batch_id,
                status=BatchTranslationJob.STATUS_RUNNING,
            ).update(status=BatchTranslationJob.STATUS_PENDING)
        return reclaimed


# ---------------------------------------------------------------------------
# get_status / collect_results
# ---------------------------------------------------------------------------


def _isoformat(dt) -> Optional[str]:
    if dt is None:
        return None
    try:
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        try:
            return dt.isoformat()
        except Exception:
            return None


def get_status(batch_id: str) -> BatchStatusSnapshot:
    """Return a :class:`BatchStatusSnapshot` for ``batch_id``."""
    BatchTranslationJob, BatchTranslationInput = _load_django_models()
    try:
        job_row = BatchTranslationJob.objects.get(batch_id=batch_id)
    except BatchTranslationJob.DoesNotExist as exc:
        raise BatchNotFoundError(batch_id) from exc

    counts = BatchTranslationInput.objects.filter(batch=job_row).values_list(
        "status", flat=True
    )
    pending = running = completed = failed = 0
    for s in counts:
        if s == "pending":
            pending += 1
        elif s == "running":
            running += 1
        elif s == "completed":
            completed += 1
        elif s == "failed":
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
        certified=bool(getattr(job_row, "certified", False)),
    )


def collect_results(
    batch_id: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    max_bytes: int = DEFAULT_MAX_RESULT_BYTES,
) -> list[BatchTranslationResult]:
    """Return per-input results in original input order.

    Returns results regardless of batch status -- callers needing
    "only when terminal" semantics should consult :func:`get_status`
    first and gate on ``status in {completed, failed, cancelled}``.

    S2-04: when the tenant's ``require_certified`` policy is True and the
    batch row is not yet ``certified=True``, raises :class:`PolicyDenied`
    so uncertified results cannot be delivered.
    """
    from ocr_local.translation.policy import (
        assert_certified_delivery_allowed,
        get_tenant_policy,
    )

    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise BatchValidationError("result offset must be a non-negative integer")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise BatchValidationError("result limit must be a positive integer")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise BatchValidationError("result byte cap must be a positive integer")

    BatchTranslationJob, BatchTranslationInput = _load_django_models()
    try:
        job_row = BatchTranslationJob.objects.get(batch_id=batch_id)
    except BatchTranslationJob.DoesNotExist as exc:
        raise BatchNotFoundError(batch_id) from exc

    policy = get_tenant_policy(job_row.tenant_id)
    assert_certified_delivery_allowed(
        policy,
        certified=bool(getattr(job_row, "certified", False)),
    )

    rows = BatchTranslationInput.objects.filter(batch=job_row).order_by("input_index")
    results: list[BatchTranslationResult] = []
    skipped = 0
    total_bytes = 0
    for row in rows:
        if skipped < offset:
            skipped += 1
            continue
        if limit is not None and len(results) >= limit:
            break
        result = BatchTranslationResult(
            client_ref=row.client_ref,
            target_text=row.target_text or "",
            engine_id=row.engine_id or "",
            confidence=row.confidence,
            glossary_hits=list(row.glossary_hits_json or []),
            error=(row.error or None) if row.error else None,
        )
        try:
            result_bytes = len(
                json.dumps(
                    {
                        "client_ref": result.client_ref,
                        "target_text": result.target_text,
                        "engine_id": result.engine_id,
                        "confidence": result.confidence,
                        "glossary_hits": result.glossary_hits,
                        "error": result.error,
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError) as exc:
            raise BatchValidationError(
                "stored translation result is not JSON-safe"
            ) from exc
        if total_bytes + result_bytes > max_bytes:
            raise BatchValidationError(
                "translation results exceed the configured response size limit"
            )
        total_bytes += result_bytes
        results.append(result)
    return results


def _verify_custody_event_store_bound(
    event_id: str,
    *,
    batch_id: str,
    batch_tenant_id: str,
) -> None:
    """PH1-15 / PH2-19: bind custody evidence to the exact batch and tenant.

    Accepts a row in ``jobs.CustodyEvent`` whose ``event_hash`` equals the
    id (or ``id`` pk as string) only when its document/data envelope names the
    exact batch and tenant. When Django is unavailable, falls back to bounded
    scanning of custody JSONL under ``OCR_CUSTODY_DIR`` / ``CUSTODY_DIR``;
    hash equality alone is intentionally insufficient.

    Raises :class:`PolicyDenied` when unbound.
    """
    # 1) Django ORM custody table (preferred durable store).
    try:
        from jobs.models import CustodyEvent  # type: ignore[import-not-found]

        qs = CustodyEvent.objects.filter(
            event_hash=event_id,
            event_type="translation_review_authorized",
        )
        if qs.exists():
            evt = qs.select_related("job").first()
            if evt is None:
                raise PolicyDenied(
                    ReasonCode.TENANT_POLICY,
                    f"custody event {event_id!r} lookup returned no row",
                )
            expected_document_id = f"batch:{batch_id}"
            if getattr(evt, "document_id", "") != expected_document_id:
                raise PolicyDenied(
                    ReasonCode.TENANT_POLICY,
                    (
                        f"custody event {event_id!r} is not bound to batch "
                        f"{batch_id!r} (PH1-15 store-bind)"
                    ),
                )
            data = getattr(evt, "data", None)
            if (
                not isinstance(data, dict)
                or str(data.get("batch_id") or "") != batch_id
                or str(data.get("review_id") or "").strip() == ""
                or str(data.get("reviewer_id") or "").strip() == ""
                or str(data.get("auth_method") or "")
                not in {"piv_cac", "oidc_mfa", "hardware_token"}
            ):
                raise PolicyDenied(
                    ReasonCode.TENANT_POLICY,
                    (
                        f"custody event {event_id!r} has no exact batch binding "
                        "(PH1-15 store-bind)"
                    ),
                )
            event_tid = str(data.get("tenant_id") or "").strip()
            if event_tid != batch_tenant_id:
                raise PolicyDenied(
                    ReasonCode.TENANT_POLICY,
                    (
                        f"custody event {event_id!r} tenant {event_tid!r} does not "
                        f"match batch tenant {batch_tenant_id!r} (PH1-15 store-bind)"
                    ),
                )
            job = getattr(evt, "job", None)
            if job is not None:
                job_tid = (getattr(job, "tenant_id", None) or "").strip()
                if job_tid != batch_tenant_id:
                    raise PolicyDenied(
                        ReasonCode.TENANT_POLICY,
                        (
                            f"custody event {event_id!r} tenant {job_tid!r} "
                            f"does not match batch tenant {batch_tenant_id!r} "
                            "(PH1-15 store-bind)"
                        ),
                    )
            return
    except PolicyDenied:
        raise
    except Exception as exc:
        logger.debug(
            "promote store-bind: CustodyEvent lookup unavailable (%s); trying JSONL",
            exc,
        )

    # 2) JSONL custody files (pipeline forensic path).
    import json
    import os
    from pathlib import Path

    roots: list[Path] = []
    for key in ("OCR_CUSTODY_DIR", "CUSTODY_DIR", "OCR_OUTPUT_DIR"):
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        p = Path(raw)
        if key == "OCR_OUTPUT_DIR":
            p = p / "custody"
        roots.append(p)
    files_scanned = 0

    def _matches_bound_event(obj: object) -> bool:
        if not isinstance(obj, dict):
            return False
        if str(obj.get("hash") or "") != event_id:
            return False
        if str(obj.get("document_id") or "") != f"batch:{batch_id}":
            return False
        if str(obj.get("event_type") or "") != "translation_review_authorized":
            return False
        data = obj.get("data")
        return (
            isinstance(data, dict)
            and str(data.get("batch_id") or "") == batch_id
            and str(data.get("tenant_id") or "").strip() == batch_tenant_id
            and str(data.get("review_id") or "").strip() != ""
            and str(data.get("reviewer_id") or "").strip() != ""
            and str(data.get("auth_method") or "")
            in {"piv_cac", "oidc_mfa", "hardware_token"}
        )

    for root in roots:
        try:
            root_stat = root.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            continue
        for path in root.glob("*.custody.jsonl"):
            files_scanned += 1
            if files_scanned > _CUSTODY_SCAN_MAX_FILES:
                break
            try:
                path_stat = path.lstat()
                if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(
                    path_stat.st_mode
                ):
                    continue
                fd = os.open(
                    os.fspath(path),
                    os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                )
                with os.fdopen(fd, "rb") as fh:
                    scanned_bytes = 0
                    while scanned_bytes < _CUSTODY_SCAN_MAX_FILE_BYTES:
                        raw_line = fh.readline(_CUSTODY_SCAN_MAX_LINE_BYTES + 1)
                        if not raw_line:
                            break
                        scanned_bytes += len(raw_line)
                        if len(raw_line) > _CUSTODY_SCAN_MAX_LINE_BYTES:
                            continue
                        try:
                            obj = json.loads(raw_line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if _matches_bound_event(obj):
                            return
            except (OSError, ValueError):
                continue
        if files_scanned > _CUSTODY_SCAN_MAX_FILES:
            break

    raise PolicyDenied(
        ReasonCode.TENANT_POLICY,
        (
            f"promote_batch_certified custody_event_id {event_id!r} not found "
            "in CustodyEvent store or custody JSONL (PH1-15 store-bind)"
        ),
    )


def promote_batch_certified(
    batch_id: str,
    *,
    custody_event_id: str,
    tenant_id: str | None = None,
) -> bool:
    """Promote ``BatchTranslationJob.certified`` after real custody linkage.

    S2-04 / S3-01 residual close: delivery gate requires ``certified=True``
    for ``require_certified`` tenants. This is the only product path that
    may set the flag. Refuses:

    * missing/blank/synthetic ``custody_event_id`` (``waiver:*`` prefix)
    * unknown batch
    * tenant mismatch when ``tenant_id`` is provided
    * custody id not bound in durable store (PH1-15 / PH2-19)

    Returns True when the row was flipped False→True, False if already
    certified (idempotent). Raises :class:`BatchNotFoundError` or
    :class:`PolicyDenied`.
    """
    event_id = (custody_event_id or "").strip()
    _synthetic_prefixes = ("waiver:", "fake:", "synthetic:", "test:")
    if not event_id or any(event_id.startswith(p) for p in _synthetic_prefixes):
        raise PolicyDenied(
            ReasonCode.TENANT_POLICY,
            (
                "promote_batch_certified requires a real custody_event_id "
                f"(got {custody_event_id!r}); synthetic IDs refused (S3-01)"
            ),
        )
    if len(event_id) != 64 or any(ch not in "0123456789abcdef" for ch in event_id):
        raise PolicyDenied(
            ReasonCode.TENANT_POLICY,
            "promote_batch_certified requires a sealed SHA-256 custody event hash",
        )

    BatchTranslationJob, _BatchTranslationInput = _load_django_models()
    try:
        job_row = BatchTranslationJob.objects.get(batch_id=batch_id)
    except BatchTranslationJob.DoesNotExist as exc:
        raise BatchNotFoundError(batch_id) from exc

    if tenant_id is not None and job_row.tenant_id != tenant_id:
        raise PolicyDenied(
            ReasonCode.TENANT_POLICY,
            (
                f"batch {batch_id} tenant {job_row.tenant_id!r} does not match "
                f"authorizing tenant {tenant_id!r}"
            ),
        )

    terminal_status = getattr(BatchTranslationJob, "STATUS_COMPLETED", "completed")
    if getattr(job_row, "status", None) != terminal_status:
        raise PolicyDenied(
            ReasonCode.TENANT_POLICY,
            f"batch {batch_id} must be completed before certification",
        )

    batch_tid = (getattr(job_row, "tenant_id", None) or tenant_id or "").strip()
    _verify_custody_event_store_bound(
        event_id,
        batch_id=batch_id,
        batch_tenant_id=batch_tid,
    )

    if bool(getattr(job_row, "certified", False)):
        return False

    updated = BatchTranslationJob.objects.filter(
        batch_id=batch_id, certified=False
    ).update(certified=True)
    if updated:
        logger.info(
            "batch %s promoted certified=True custody_event_id=%s tenant=%s",
            batch_id,
            event_id,
            job_row.tenant_id,
        )
        return True
    return False


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
    BatchTranslationJob, BatchTranslationInput = _load_django_models()
    from django.db import transaction  # type: ignore[import-not-found]

    # Lazy Celery import -- when the broker is unreachable we still
    # update DB rows to cancelled so ``get_status`` reflects the cancel.
    revoke = None
    try:
        from coordinator.coordinator.celery import app as _celery_app

        revoke = _celery_app.control.revoke
    except Exception as exc:
        logger.warning(
            "cancel_batch: Celery app unavailable, skipping revoke: %s",
            exc,
        )

    # The database fence is authoritative; broker revoke is best-effort after
    # commit. Workers reconcile under the same input row lock and therefore
    # cannot resurrect a cancelled input with a late success/failure write.
    with transaction.atomic():
        try:
            job_row = BatchTranslationJob.objects.select_for_update().get(
                batch_id=batch_id
            )
        except BatchTranslationJob.DoesNotExist as exc:
            raise BatchNotFoundError(batch_id) from exc

        if job_row.status in BatchTranslationJob.TERMINAL_STATUSES:
            return 0

        pending_or_running = list(
            BatchTranslationInput.objects.select_for_update()
            .filter(
                batch=job_row,
                status__in=["pending", "running"],
            )
            .order_by("input_index")
        )
        task_ids = [
            str(inp_row.celery_task_id)
            for inp_row in pending_or_running
            if inp_row.celery_task_id
        ]
        for inp_row in pending_or_running:
            inp_row.status = "cancelled"
            inp_row.error = inp_row.error or "cancelled"
            inp_row.completed_at = datetime.now(timezone.utc)
            inp_row.save(update_fields=["status", "error", "completed_at"])

        job_row.status = BatchTranslationJob.STATUS_CANCELLED
        job_row.completed_at = datetime.now(timezone.utc)
        job_row.save(update_fields=["status", "completed_at"])

    revoked = len(pending_or_running)
    if revoke is not None:
        for task_id in task_ids:
            try:
                revoke(task_id, terminate=False)
            except Exception as exc:
                logger.debug(
                    "cancel_batch: revoke failed for task=%s: %s",
                    task_id,
                    exc,
                )

    if custody_chain is not None:
        try:
            emit_batch_cancelled(
                custody_chain,
                batch_id=batch_id,
                tenant_id=job_row.tenant_id,
                revoked_count=revoked,
            )
        except Exception:
            logger.debug(
                "cancel_batch: custody emit failed (non-fatal)",
                exc_info=True,
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
