"""Batch translation endpoints -- Plan B Wave M2 (B17).

Feature-gated: only registered with the FastAPI application when
``ENABLE_TRANSLATION_API`` is truthy in the environment (same gate as
the existing translation router).  The global ``api_key_middleware``
establishes identity; explicit route dependencies then enforce
operator/read roles even when the deliberate development-only anonymous
mode is enabled.

Rate limiting: ``POST /api/v1/translation/batches`` is capped at
5 req/min per API key (slowapi).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from asgiref.sync import sync_to_async
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from api.identity import require_role
from api.language_tags import canonicalize_language_tag
from api.limits import get_default_rate, limiter
from api.routers.translation_scope import (
    bind_requested_tenant,
    has_tenant_scope,
    require_owned_tenant,
)
from ocr_local.translation.batch import (
    BatchInput,
    BatchNotFoundError,
    BatchStatusSnapshot,
    BatchTranslationRequest,
    BatchTranslationResult,
    BatchValidationError,
    cancel_batch,
    collect_results,
    fan_out,
    get_status,
    promote_batch_certified,
    reclaim_stale_batch_inputs,
    submit_batch,
)
from ocr_local.translation.policy import PolicyDenied

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/translation", tags=["translation"])


def _request_rls_tenant(request: Request) -> str | None:
    """Use control-plane RLS for platform admins, tenant RLS otherwise."""
    if not has_tenant_scope(request):
        return None
    return getattr(request.state, "tenant_id", None)


def _run_rls_bound(tenant_id: str | None, func, *args, **kwargs):
    """Execute one Django ORM operation with an explicit RLS identity."""
    from api.coordinator_bridge import ensure_coordinator_django

    ensure_coordinator_django()
    from jobs.rls import (  # type: ignore[import-not-found]
        control_plane_rls_context,
        tenant_rls_context,
    )

    if tenant_id:
        with tenant_rls_context(tenant_id):
            return func(*args, **kwargs)
    with control_plane_rls_context():
        return func(*args, **kwargs)


def _enqueue_certification_review(batch_id: str, tenant_id: str):
    """Create the durable API-side review handle for a submitted batch."""
    from api.review_queue import ReviewReason
    from api.routers.review import _get_queue

    return _get_queue().add_unique(
        job_id=f"translation-batch:{batch_id}",
        reason=ReviewReason.TRANSLATION_CERTIFICATION.value,
        metadata={"batch_id": batch_id, "tenant_id": tenant_id},
    )


def _translation_input_lease_seconds() -> int:
    from jobs.tasks_translation_batch import (  # type: ignore[import-not-found]
        _input_lease_seconds,
    )

    return _input_lease_seconds()


def _translation_recovery_candidates(batch_model, *, cutoff, limit: int):
    """Select only new dispatches and expired leases, never lifetime history."""
    from django.db.models import Q  # type: ignore[import-not-found]

    return list(
        batch_model.objects.filter(
            Q(status=batch_model.STATUS_PENDING)
            | Q(
                status=batch_model.STATUS_RUNNING,
                inputs__status="running",
                inputs__started_at__lte=cutoff,
            )
            | Q(
                status=batch_model.STATUS_RUNNING,
                inputs__status="running",
                inputs__started_at__isnull=True,
                submitted_at__lte=cutoff,
            )
        )
        .distinct()
        .order_by("submitted_at", "batch_id")[:limit]
    )


def reconcile_translation_batch_handoffs(*, queue=None) -> int:
    """Repair a bounded set of dispatch leases and pending promotions."""
    from api.coordinator_bridge import ensure_coordinator_django

    ensure_coordinator_django()
    from jobs.models import BatchTranslationJob  # type: ignore[import-not-found]
    from jobs.rls import (  # type: ignore[import-not-found]
        control_plane_rls_context,
        tenant_rls_context,
    )

    try:
        limit = int(os.environ.get("TRANSLATION_RECONCILE_BATCH_LIMIT", "100"))
    except (TypeError, ValueError):
        limit = 100
    limit = min(max(limit, 1), 1000)
    lease_seconds = _translation_input_lease_seconds()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=lease_seconds)

    with control_plane_rls_context():
        rows = _translation_recovery_candidates(
            BatchTranslationJob,
            cutoff=cutoff,
            limit=limit,
        )

    repaired = 0
    for row in rows:
        try:
            _enqueue_certification_review(row.batch_id, row.tenant_id)
            with tenant_rls_context(row.tenant_id):
                reclaim_stale_batch_inputs(row.batch_id, lease_seconds=lease_seconds)
                fan_out(row.batch_id)
            repaired += 1
        except Exception:
            logger.exception(
                "Translation handoff recovery failed for batch %s; continuing",
                row.batch_id,
            )

    if queue is None:
        from api.routers.review import _get_queue

        queue = _get_queue()
    for review_item in queue.list_pending_translation_promotions(limit=limit):
        metadata = review_item.metadata or {}
        certification = metadata.get("certification") or {}
        batch_id = str(metadata.get("batch_id") or "").strip()
        tenant_id = str(metadata.get("tenant_id") or "").strip()
        custody_event_id = str(certification.get("custody_event_id") or "").strip()
        if not batch_id or not tenant_id or not custody_event_id:
            if queue.abandon_translation_promotion(
                review_item.review_id,
                "invalid durable promotion binding",
            ):
                repaired += 1
            else:
                logger.warning(
                    "Translation certification recovery could not quarantine "
                    "invalid review %s",
                    review_item.review_id,
                )
            continue
        try:
            with tenant_rls_context(tenant_id):
                promote_batch_certified(
                    batch_id,
                    custody_event_id=custody_event_id,
                    tenant_id=tenant_id,
                )
            if not queue.confirm_translation_promotion(
                review_item.review_id, custody_event_id
            ):
                raise RuntimeError("translation promotion confirmation CAS lost")
            repaired += 1
        except BatchNotFoundError:
            if queue.abandon_translation_promotion(
                review_item.review_id,
                "coordinator batch no longer exists",
            ):
                repaired += 1
            else:
                logger.warning(
                    "Translation certification recovery could not quarantine "
                    "missing batch %s review %s",
                    batch_id,
                    review_item.review_id,
                )
        except Exception as exc:
            if not queue.defer_translation_promotion(
                review_item.review_id,
                f"{type(exc).__name__}: {exc}",
            ):
                logger.warning(
                    "Translation certification recovery could not defer "
                    "review %s after a retryable failure",
                    review_item.review_id,
                )
            logger.exception(
                "Translation certification recovery failed for batch %s; continuing",
                batch_id,
            )
    return repaired


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class BatchInputModel(BaseModel):
    client_ref: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=8_192)
    optional_metadata: Optional[dict] = None

    @field_validator("optional_metadata")
    @classmethod
    def _metadata_budget(cls, value: Optional[dict]) -> Optional[dict]:
        if value is None:
            return None
        # The outer request body is bounded by the API middleware, but each
        # input's opaque metadata also needs a local budget so 1,000 inputs
        # cannot each carry an unbounded nested object into the coordinator.
        try:
            encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("optional_metadata must be JSON-serializable") from exc
        if len(encoded.encode("utf-8")) > 16_384:
            raise ValueError("optional_metadata exceeds the 16 KiB per-input limit")
        return value


class BatchSubmitRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=128)
    source_lang: str = Field(..., min_length=1, max_length=10)
    target_lang: str = Field(..., min_length=1, max_length=10)
    inputs: list[BatchInputModel] = Field(default_factory=list, max_length=1000)
    priority: int = Field(default=0, ge=0, le=9)
    glossary_enabled: bool = True
    requested_certified: bool = False

    @field_validator("source_lang", "target_lang")
    @classmethod
    def _language_tag(cls, value: str) -> str:
        return canonicalize_language_tag(value, max_length=10)


class BatchSubmitResponse(BaseModel):
    batch_id: str
    review_id: str
    status: str = "pending"


class BatchStatusModel(BaseModel):
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


class BatchResultModel(BaseModel):
    client_ref: str
    target_text: str
    engine_id: str
    confidence: Optional[float]
    glossary_hits: list
    error: Optional[str]


class BatchCancelResponse(BaseModel):
    batch_id: str
    revoked: int
    status: str = "cancelled"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _to_request(req: BatchSubmitRequest) -> BatchTranslationRequest:
    return BatchTranslationRequest(
        tenant_id=req.tenant_id,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        inputs=[
            BatchInput(
                client_ref=item.client_ref,
                text=item.text,
                optional_metadata=item.optional_metadata,
            )
            for item in req.inputs
        ],
        priority=req.priority,
        glossary_enabled=req.glossary_enabled,
        requested_certified=req.requested_certified,
    )


def _to_status_model(snap: BatchStatusSnapshot) -> BatchStatusModel:
    return BatchStatusModel(
        batch_id=snap.batch_id,
        tenant_id=snap.tenant_id,
        source_lang=snap.source_lang,
        target_lang=snap.target_lang,
        status=snap.status,
        total_inputs=snap.total_inputs,
        completed_inputs=snap.completed_inputs,
        failed_inputs=snap.failed_inputs,
        pending_inputs=snap.pending_inputs,
        running_inputs=snap.running_inputs,
        submitted_at=snap.submitted_at,
        completed_at=snap.completed_at,
    )


def _to_result_model(r: BatchTranslationResult) -> BatchResultModel:
    return BatchResultModel(
        client_ref=r.client_ref,
        target_text=r.target_text,
        engine_id=r.engine_id,
        confidence=r.confidence,
        glossary_hits=list(r.glossary_hits or []),
        error=r.error,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/batches",
    status_code=202,
    response_model=BatchSubmitResponse,
    name="submit_translation_batch",
)
@limiter.limit("5/minute")
async def submit_translation_batch(
    request: Request,
    req: BatchSubmitRequest,
    _auth: None = Depends(require_role("admin", "operator")),
) -> BatchSubmitResponse:
    """Submit a new batch translation request.

    Returns 202 with the new ``batch_id`` on success.  Returns 400 on
    validation errors (oversize input, duplicate ``client_ref``, etc.)
    or when ``requested_certified=True`` is rejected.
    """
    effective_tenant_id = bind_requested_tenant(request, req.tenant_id)
    if effective_tenant_id != req.tenant_id:
        req = req.model_copy(update={"tenant_id": effective_tenant_id})
    try:
        domain_req = _to_request(req)
        batch_id = await sync_to_async(_run_rls_bound, thread_sensitive=True)(
            effective_tenant_id, submit_batch, domain_req
        )
    except BatchValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PolicyDenied as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # Django not configured -- 503 so callers retry against a
        # properly configured deployment.
        logger.error("Translation batch persistence is unavailable", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "translation_batch_unavailable",
                "message": "Translation batch persistence is temporarily unavailable.",
            },
        ) from exc

    try:
        review_item = _enqueue_certification_review(
            batch_id,
            effective_tenant_id,
        )
    except Exception as exc:
        logger.exception(
            "Translation certification review could not be created for %s", batch_id
        )
        try:
            await sync_to_async(_run_rls_bound, thread_sensitive=True)(
                effective_tenant_id, cancel_batch, batch_id
            )
        except Exception:
            logger.critical(
                "Could not compensate batch %s after review handoff failure",
                batch_id,
                exc_info=True,
            )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "translation_review_unavailable",
                "message": "Batch persisted but its certification review was unavailable.",
                "batch_id": batch_id,
            },
        ) from exc

    # A batch with no broker task would otherwise remain pending forever.
    # ``fan_out`` records dispatch failures against the durable rows and this
    # endpoint surfaces the outage instead of returning a false 202 success.
    try:
        await sync_to_async(_run_rls_bound, thread_sensitive=True)(
            effective_tenant_id, fan_out, batch_id
        )
    except Exception as exc:
        logger.warning(
            "submit_translation_batch: fan_out failed for batch_id=%s: %s",
            batch_id,
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Translation batch was persisted but dispatch failed",
                "batch_id": batch_id,
            },
        ) from exc

    return BatchSubmitResponse(
        batch_id=batch_id,
        review_id=review_item.review_id,
        status="pending",
    )


@router.get(
    "/batches/{batch_id}",
    response_model=BatchStatusModel,
    name="get_translation_batch_status",
)
@limiter.limit(get_default_rate())
async def get_translation_batch_status(
    request: Request,
    batch_id: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
) -> BatchStatusModel:
    """Return the current status snapshot for ``batch_id``."""
    try:
        snap = await sync_to_async(_run_rls_bound, thread_sensitive=True)(
            _request_rls_tenant(request), get_status, batch_id
        )
    except BatchNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    except RuntimeError as exc:
        logger.error("Translation batch status lookup is unavailable", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "translation_batch_unavailable",
                "message": "Translation batch status is temporarily unavailable.",
            },
        ) from exc
    require_owned_tenant(
        request,
        snap.tenant_id,
        not_found_detail=f"batch {batch_id} not found",
    )
    return _to_status_model(snap)


@router.get(
    "/batches/{batch_id}/results",
    response_model=list[BatchResultModel],
    name="get_translation_batch_results",
)
@limiter.limit(get_default_rate())
async def get_translation_batch_results(
    request: Request,
    batch_id: str,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=1_000_000),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
) -> list[BatchResultModel]:
    """Return per-input results for ``batch_id``.

    Returns 409 when the batch is not yet in a terminal status -- callers
    should poll ``GET /batches/{id}`` and only fetch results once the
    status is one of ``completed``, ``failed``, ``cancelled``.
    """
    try:
        rls_tenant = _request_rls_tenant(request)
        snap = await sync_to_async(_run_rls_bound, thread_sensitive=True)(
            rls_tenant, get_status, batch_id
        )
    except BatchNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    except RuntimeError as exc:
        logger.error("Translation batch result lookup is unavailable", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "translation_batch_unavailable",
                "message": "Translation batch results are temporarily unavailable.",
            },
        ) from exc

    require_owned_tenant(
        request,
        snap.tenant_id,
        not_found_detail=f"batch {batch_id} not found",
    )

    if snap.status not in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"batch {batch_id} is in status={snap.status!r}; results "
                "are only available once the batch reaches a terminal "
                "status (completed/failed/cancelled)"
            ),
        )

    try:
        results = await sync_to_async(_run_rls_bound, thread_sensitive=True)(
            rls_tenant,
            collect_results,
            batch_id,
            limit=limit,
            offset=offset,
        )
    except BatchNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    except PolicyDenied as exc:
        # S2-04: require_certified delivery gate
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except BatchValidationError as exc:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "translation_results_too_large",
                "message": str(exc),
            },
        ) from exc
    except RuntimeError as exc:
        logger.error("Translation batch result lookup is unavailable", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "translation_batch_unavailable",
                "message": "Translation batch cancellation is temporarily unavailable.",
            },
        ) from exc

    return [_to_result_model(r) for r in results]


@router.post(
    "/batches/{batch_id}/cancel",
    response_model=BatchCancelResponse,
    name="cancel_translation_batch",
)
@limiter.limit(get_default_rate())
async def cancel_translation_batch(
    request: Request,
    batch_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
) -> BatchCancelResponse:
    """Cancel a pending or running batch.

    Returns 200 with ``revoked`` count on success.  Returns 404 when
    the batch is unknown.  Idempotent: calling cancel on a terminal
    batch returns ``revoked=0`` without error.
    """
    try:
        rls_tenant = _request_rls_tenant(request)
        snap = await sync_to_async(_run_rls_bound, thread_sensitive=True)(
            rls_tenant, get_status, batch_id
        )
        require_owned_tenant(
            request,
            snap.tenant_id,
            not_found_detail=f"batch {batch_id} not found",
        )
        revoked = await sync_to_async(_run_rls_bound, thread_sensitive=True)(
            rls_tenant, cancel_batch, batch_id
        )
        final_snap = await sync_to_async(_run_rls_bound, thread_sensitive=True)(
            rls_tenant, get_status, batch_id
        )
    except BatchNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    except RuntimeError as exc:
        logger.error("Translation batch cancellation is unavailable", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "translation_batch_unavailable",
                "message": "Translation batch cancellation is temporarily unavailable.",
            },
        ) from exc
    return BatchCancelResponse(
        batch_id=batch_id,
        revoked=revoked,
        status=final_snap.status,
    )
