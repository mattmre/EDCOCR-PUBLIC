"""Review queue endpoints for human review of low-confidence documents."""

from __future__ import annotations

import logging
import re
import threading

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.database import Job, get_session_factory
from api.deps import pagination_params
from api.identity import require_role
from api.limits import get_default_rate, limiter
from api.models import (
    ErrorResponse,
    ReviewCertifyRequest,
    ReviewDecisionRequest,
    ReviewItemResponse,
    ReviewQueueResponse,
    ReviewStatsResponse,
)
from api.review_queue import ReviewQueue, ReviewReason, ReviewStatus
from exception_router import ExceptionRouter

logger = logging.getLogger(__name__)

_REVIEW_ID_RE = re.compile(r"^rev_[0-9a-f]{12}$")

router = APIRouter(prefix="/api/v1/review", tags=["review"])

# Module-level singleton; lazily initialized to avoid import-time DB access.
_review_queue: ReviewQueue | None = None
_review_queue_lock = threading.Lock()


def _get_queue() -> ReviewQueue:
    """Return the module-level ReviewQueue singleton (thread-safe)."""
    global _review_queue
    if _review_queue is None:
        with _review_queue_lock:
            if _review_queue is None:
                _review_queue = ReviewQueue()
    return _review_queue


def _request_tenant_id(request: Request) -> str | None:
    """Return tenant scope for authenticated tenant keys."""
    return getattr(request.state, "tenant_id", None)


def _tenant_job_ids(tenant_id: str) -> list[str]:
    """Return all job IDs belonging to a tenant."""
    session = get_session_factory()()
    try:
        rows = session.query(Job.job_id).filter(Job.tenant_id == tenant_id).all()
        return [r[0] for r in rows]
    finally:
        session.close()


def _verify_review_item_tenant(
    item, tenant_id: str | None, allowed_job_ids: list[str] | None
) -> bool:
    """Return True if the review item belongs to the tenant."""
    if tenant_id is None:
        return True
    if allowed_job_ids is not None:
        return item.job_id in allowed_job_ids
    # Fallback: check the job table directly
    session = get_session_factory()()
    try:
        return bool(
            session.query(Job.job_id)
            .filter(Job.job_id == item.job_id, Job.tenant_id == tenant_id)
            .first()
        )
    finally:
        session.close()


def _validate_review_id(review_id: str) -> None:
    """Raise 400 if the review_id format is invalid."""
    if not _REVIEW_ID_RE.match(review_id):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_review_id",
                "message": "Invalid review ID format. Expected: rev_<12 hex chars>",
            },
        )


# ------------------------------------------------------------------
# GET /api/v1/review/queue --- List pending review items
# ------------------------------------------------------------------


@router.get(
    "/queue",
    name="list_review_queue",
    response_model=ReviewQueueResponse,
    responses={403: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def list_review_queue(
    request: Request,
    reason: str | None = Query(default=None, description="Filter by review reason"),
    status: str | None = Query(default=None, description="Filter by status (default: pending)"),
    pagination: tuple[int, int] = Depends(pagination_params),
    _auth: None = Depends(require_role("admin", "operator")),
):
    """List review queue items with optional filtering.

    By default returns pending items. Use the ``status`` parameter to
    filter by a specific status (pending, approved, rejected, reprocess).
    """
    limit, offset = pagination

    # Validate status parameter against ReviewStatus enum values
    _valid_statuses = {s.value for s in ReviewStatus}
    if status is not None and status not in _valid_statuses:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_status",
                "message": (
                    f"Invalid status filter: '{status}'. "
                    f"Must be one of: {sorted(_valid_statuses)}"
                ),
            },
        )

    # Validate reason parameter against ReviewReason enum values
    _valid_reasons = {r.value for r in ReviewReason}
    if reason is not None and reason not in _valid_reasons:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_reason",
                "message": (
                    f"Invalid reason filter: '{reason}'. "
                    f"Must be one of: {sorted(_valid_reasons)}"
                ),
            },
        )

    queue = _get_queue()
    tenant_id = _request_tenant_id(request)

    # Tenant isolation: restrict results to jobs owned by this tenant
    allowed_job_ids = None
    if tenant_id is not None:
        allowed_job_ids = _tenant_job_ids(tenant_id)

    # Default to pending if no status specified
    effective_status = status if status is not None else "pending"

    items = queue.list_all(
        limit=limit, offset=offset, status=effective_status, reason=reason,
        allowed_job_ids=allowed_job_ids,
    )
    total = queue.count(
        status=effective_status, reason=reason,
        allowed_job_ids=allowed_job_ids,
    )

    return ReviewQueueResponse(
        items=[ReviewItemResponse(**item.to_dict()) for item in items],
        total=total,
    )


# ------------------------------------------------------------------
# GET /api/v1/review/stats --- Review queue statistics
# ------------------------------------------------------------------


@router.get(
    "/stats",
    name="review_stats",
    response_model=ReviewStatsResponse,
    responses={403: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def review_stats(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Return aggregate review queue statistics."""
    queue = _get_queue()
    return ReviewStatsResponse(**queue.stats())


# ------------------------------------------------------------------
# GET /api/v1/review/rules --- List configured routing rules
# ------------------------------------------------------------------

# Module-level router singleton; lazily initialized.
_exception_router: ExceptionRouter | None = None
_exception_router_lock = threading.Lock()


def _get_router() -> ExceptionRouter:
    """Return the module-level ExceptionRouter singleton (thread-safe)."""
    global _exception_router
    if _exception_router is None:
        with _exception_router_lock:
            if _exception_router is None:
                _exception_router = ExceptionRouter()
    return _exception_router


@router.get(
    "/rules",
    name="list_review_rules",
    responses={403: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def list_review_rules(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """List all configured exception routing rules and their status."""
    er = _get_router()
    return {"rules": er.get_rules()}


# ------------------------------------------------------------------
# GET /api/v1/review/{review_id} --- Get review item details
# ------------------------------------------------------------------


@router.get(
    "/{review_id}",
    name="get_review_item",
    response_model=ReviewItemResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def get_review_item(
    request: Request,
    review_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Get details for a specific review item."""
    _validate_review_id(review_id)

    queue = _get_queue()
    item = queue.get(review_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "review_not_found",
                "message": f"Review item {review_id} not found.",
            },
        )

    # Tenant isolation: verify the review item's job belongs to this tenant
    tenant_id = _request_tenant_id(request)
    if not _verify_review_item_tenant(item, tenant_id, None):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "review_not_found",
                "message": f"Review item {review_id} not found.",
            },
        )

    return ReviewItemResponse(**item.to_dict())


# ------------------------------------------------------------------
# POST /api/v1/review/{review_id}/decision --- Submit decision
# ------------------------------------------------------------------


@router.post(
    "/{review_id}/decision",
    name="review_decision",
    response_model=ReviewItemResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def review_decision(
    request: Request,
    review_id: str,
    body: ReviewDecisionRequest,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Submit a review decision (approve, reject, or request reprocessing)."""
    _validate_review_id(review_id)

    # Tenant isolation: verify the review item's job belongs to this tenant
    # before allowing any decision to be recorded.
    tenant_id = _request_tenant_id(request)
    if tenant_id is not None:
        queue_check = _get_queue()
        existing = queue_check.get(review_id)
        if existing is None or not _verify_review_item_tenant(existing, tenant_id, None):
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "review_not_found",
                    "message": f"Review item {review_id} not found.",
                },
            )

    queue = _get_queue()

    try:
        item = queue.decide(
            review_id=review_id,
            status=body.status,
            reviewer=body.reviewer,
            notes=body.notes,
        )
    except ValueError as exc:
        # Distinguish between "already decided" and "invalid status"
        if "already" in str(exc).lower():
            raise HTTPException(
                status_code=409,
                detail={"error": "already_decided", "message": str(exc)},
            )
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": str(exc)},
        )

    if item is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "review_not_found",
                "message": f"Review item {review_id} not found.",
            },
        )

    logger.info(
        "Review decision: %s -> %s by %s",
        review_id,
        body.status,
        body.reviewer or "(anonymous)",
    )
    return ReviewItemResponse(**item.to_dict())


# ------------------------------------------------------------------
# POST /api/v1/review/{review_id}/certify --- Strong-auth certification
# ------------------------------------------------------------------


@router.post(
    "/{review_id}/certify",
    name="review_certify",
    response_model=ReviewItemResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def review_certify(
    request: Request,
    review_id: str,
    body: ReviewCertifyRequest,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Certify a review item after explicit operator strong-auth proof.

    Certification is represented as an approved review decision with the
    selected auth method and a non-secret token fingerprint recorded in the
    decision notes. This endpoint does not claim legal certified-translation
    promotion by itself; it records the operator certification event required
    by the management UI and custody/audit flow.
    """
    _validate_review_id(review_id)

    queue = _get_queue()
    existing = queue.get(review_id)
    tenant_id = _request_tenant_id(request)
    if existing is None or not _verify_review_item_tenant(existing, tenant_id, None):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "review_not_found",
                "message": f"Review item {review_id} not found.",
            },
        )

    if existing.status != ReviewStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "already_decided",
                "message": f"Review item {review_id} is already '{existing.status}'.",
            },
        )

    token_fingerprint = f"{len(body.auth_token)}-chars"
    notes = (
        f"certified via {body.auth_method}; token_fingerprint={token_fingerprint}"
    )
    if body.notes:
        notes = f"{notes}; notes={body.notes}"

    identity = getattr(request.state, "identity", None)
    reviewer = getattr(identity, "subject", None) or "operator"

    item = queue.certify(
        review_id=review_id,
        reviewer=reviewer,
        notes=notes,
        auth_method=body.auth_method,
    )
    if item is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "review_not_found",
                "message": f"Review item {review_id} not found.",
            },
        )

    logger.info("Review certified: %s via %s", review_id, body.auth_method)
    return ReviewItemResponse(**item.to_dict())
