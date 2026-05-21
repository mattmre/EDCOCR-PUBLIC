"""Batch translation endpoints -- Plan B Wave M2 (B17).

Feature-gated: only registered with the FastAPI application when
``ENABLE_TRANSLATION_API`` is truthy in the environment (same gate as
the existing translation router).  Auth is handled by the global
``api_key_middleware`` -- endpoints here do not need an explicit
``Depends(require_api_key)``.

Rate limiting: ``POST /api/v1/translation/batches`` is capped at
5 req/min per API key (slowapi).
"""
from __future__ import annotations

import logging
from typing import Optional

from asgiref.sync import sync_to_async
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.limits import limiter
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
    submit_batch,
)
from ocr_local.translation.policy import PolicyDenied

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/translation", tags=["translation"])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class BatchInputModel(BaseModel):
    client_ref: str
    text: str
    optional_metadata: Optional[dict] = None


class BatchSubmitRequest(BaseModel):
    tenant_id: str
    source_lang: str
    target_lang: str
    inputs: list[BatchInputModel] = Field(default_factory=list)
    priority: int = 0
    glossary_enabled: bool = True
    requested_certified: bool = False


class BatchSubmitResponse(BaseModel):
    batch_id: str
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
) -> BatchSubmitResponse:
    """Submit a new batch translation request.

    Returns 202 with the new ``batch_id`` on success.  Returns 400 on
    validation errors (oversize input, duplicate ``client_ref``, etc.)
    or when ``requested_certified=True`` is rejected.
    """
    try:
        domain_req = _to_request(req)
        batch_id = await sync_to_async(submit_batch, thread_sensitive=True)(domain_req)
    except BatchValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PolicyDenied as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # Django not configured -- 503 so callers retry against a
        # properly configured deployment.
        raise HTTPException(status_code=503, detail=str(exc))

    # Best-effort fan-out -- if the broker is unavailable we still
    # return the batch_id so the caller can poll status and retry
    # via a side channel later.
    try:
        await sync_to_async(fan_out, thread_sensitive=True)(batch_id)
    except Exception as exc:
        logger.warning(
            "submit_translation_batch: fan_out failed for batch_id=%s: %s",
            batch_id, exc,
        )

    return BatchSubmitResponse(batch_id=batch_id, status="pending")


@router.get(
    "/batches/{batch_id}",
    response_model=BatchStatusModel,
    name="get_translation_batch_status",
)
async def get_translation_batch_status(batch_id: str) -> BatchStatusModel:
    """Return the current status snapshot for ``batch_id``."""
    try:
        snap = await sync_to_async(get_status, thread_sensitive=True)(batch_id)
    except BatchNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return _to_status_model(snap)


@router.get(
    "/batches/{batch_id}/results",
    response_model=list[BatchResultModel],
    name="get_translation_batch_results",
)
async def get_translation_batch_results(
    batch_id: str,
) -> list[BatchResultModel]:
    """Return per-input results for ``batch_id``.

    Returns 409 when the batch is not yet in a terminal status -- callers
    should poll ``GET /batches/{id}`` and only fetch results once the
    status is one of ``completed``, ``failed``, ``cancelled``.
    """
    try:
        snap = await sync_to_async(get_status, thread_sensitive=True)(batch_id)
    except BatchNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

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
        results = await sync_to_async(collect_results, thread_sensitive=True)(batch_id)
    except BatchNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return [_to_result_model(r) for r in results]


@router.post(
    "/batches/{batch_id}/cancel",
    response_model=BatchCancelResponse,
    name="cancel_translation_batch",
)
async def cancel_translation_batch(batch_id: str) -> BatchCancelResponse:
    """Cancel a pending or running batch.

    Returns 200 with ``revoked`` count on success.  Returns 404 when
    the batch is unknown.  Idempotent: calling cancel on a terminal
    batch returns ``revoked=0`` without error.
    """
    try:
        revoked = await sync_to_async(cancel_batch, thread_sensitive=True)(batch_id)
    except BatchNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return BatchCancelResponse(
        batch_id=batch_id, revoked=revoked, status="cancelled",
    )
