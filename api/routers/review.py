"""Review queue endpoints for human review of low-confidence documents."""

from __future__ import annotations

import asyncio
import logging
import os
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
_MAX_EVALUATE_TEXT_LENGTH = 102400
_MAX_TENANT_SCOPED_JOB_IDS = 100_000
_CERTIFICATION_CUSTODY_EVENT_TYPES = frozenset(
    {"review_certification_authorized", "translation_review_authorized"}
)

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


def _is_platform_admin(request: Request) -> bool:
    """True when server-attached tenant_permissions includes platform_admin.

    Trusts only ``request.state.tenant_permissions`` (API key path), never free-form
    JWT ``identity.claims["permissions"]`` (R4-RC-11 / RC-04 amp).
    """
    from api.authz import is_platform_admin

    return is_platform_admin(request)


def _tenant_job_ids_sync(tenant_id: str) -> list[str]:
    """Return a bounded tenant job allowlist, failing closed when too large."""
    session = get_session_factory()()
    try:
        rows = (
            session.query(Job.job_id)
            .filter(Job.tenant_id == tenant_id)
            .limit(_MAX_TENANT_SCOPED_JOB_IDS + 1)
            .all()
        )
        if len(rows) > _MAX_TENANT_SCOPED_JOB_IDS:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "tenant_scope_too_large",
                    "message": "Tenant search scope exceeds the configured safety limit.",
                },
            )
        job_ids = [r[0] for r in rows]
    finally:
        session.close()
    if os.environ.get("ENABLE_TRANSLATION_API", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return job_ids
    try:
        from api.coordinator_bridge import ensure_coordinator_django

        ensure_coordinator_django()
        from jobs.models import BatchTranslationJob  # type: ignore[import-not-found]
        from jobs.rls import tenant_rls_context  # type: ignore[import-not-found]

        remaining = _MAX_TENANT_SCOPED_JOB_IDS + 1 - len(job_ids)
        with tenant_rls_context(tenant_id):
            batch_ids = list(
                BatchTranslationJob.objects.filter(tenant_id=tenant_id)
                .order_by("batch_id")
                .values_list("batch_id", flat=True)[:remaining]
            )
        job_ids.extend(f"translation-batch:{batch_id}" for batch_id in batch_ids)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Translation review tenant scope is unavailable")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "translation_review_scope_unavailable",
                "message": "Translation review tenant scope could not be verified.",
            },
        )
    if len(job_ids) > _MAX_TENANT_SCOPED_JOB_IDS:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "tenant_scope_too_large",
                "message": "Tenant search scope exceeds the configured safety limit.",
            },
        )
    return job_ids


async def _tenant_job_ids(tenant_id: str) -> list[str]:
    """Resolve the cross-store tenant allowlist off the async request thread."""
    return await asyncio.to_thread(_tenant_job_ids_sync, tenant_id)


def _verify_review_item_tenant(
    item, tenant_id: str | None, allowed_job_ids: list[str] | None
) -> bool:
    """Return True if the review item belongs to the tenant."""
    if tenant_id is None:
        return True
    metadata = getattr(item, "metadata", None) or {}
    batch_id = str(metadata.get("batch_id") or "").strip()
    if batch_id:
        return bool(
            item.job_id == f"translation-batch:{batch_id}"
            and str(metadata.get("tenant_id") or "").strip() == tenant_id
        )
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


def _translation_review_has_authoritative_tenant(item, tenant_id: str) -> bool:
    """Verify a translation review against the coordinator RLS store."""
    metadata = getattr(item, "metadata", None) or {}
    batch_id = str(metadata.get("batch_id") or "").strip()
    if not (
        batch_id
        and item.job_id == f"translation-batch:{batch_id}"
        and str(metadata.get("tenant_id") or "").strip() == tenant_id
    ):
        return False
    try:
        from api.coordinator_bridge import ensure_coordinator_django

        ensure_coordinator_django()
        from jobs.models import BatchTranslationJob  # type: ignore[import-not-found]
        from jobs.rls import tenant_rls_context  # type: ignore[import-not-found]

        with tenant_rls_context(tenant_id):
            return BatchTranslationJob.objects.filter(
                batch_id=batch_id,
                tenant_id=tenant_id,
            ).exists()
    except Exception:
        logger.exception("Translation review tenant binding is unavailable")
        return False


async def _verify_review_item_tenant_authoritative(
    item, tenant_id: str | None, allowed_job_ids: list[str] | None
) -> bool:
    """Verify direct review access, including coordinator-owned batches."""
    if tenant_id is None:
        return True
    metadata = getattr(item, "metadata", None) or {}
    if str(metadata.get("batch_id") or "").strip():
        return await asyncio.to_thread(
            _translation_review_has_authoritative_tenant,
            item,
            tenant_id,
        )
    return _verify_review_item_tenant(item, tenant_id, allowed_job_ids)


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


def _load_review_custody_event(custody_event_id: str):
    """Resolve a custody id in the coordinator's durable store.

    Review certification is an API control-plane operation, so the Django
    import remains lazy.  If the coordinator store is unavailable, returning
    ``None`` is intentional: a string that merely looks non-synthetic is not
    certification evidence.
    """
    try:
        from api.coordinator_bridge import ensure_coordinator_django

        ensure_coordinator_django()
        from jobs.models import CustodyEvent  # type: ignore[import-not-found]
    except Exception:
        logger.warning(
            "Review custody binding unavailable: coordinator Django models not configured"
        )
        return None

    try:
        from jobs.rls import control_plane_rls_context  # type: ignore[import-not-found]

        with control_plane_rls_context():
            event = (
                CustodyEvent.objects.filter(event_hash=custody_event_id)
                .select_related("job")
                .first()
            )
        return event
    except Exception:
        logger.warning(
            "Review custody binding lookup failed for custody_event_id=%s",
            custody_event_id,
            exc_info=True,
        )
        return None


def _custody_event_binds_review(
    item, custody_event_id: str, tenant_id: str | None
) -> bool:
    """Require durable custody evidence bound to the exact review job.

    The API review queue stores the external/API job id.  When the job has a
    coordinator cross-reference, that UUID is the only acceptable custody
    parent; otherwise the queue job id is used for legacy deployments.  A
    platform-admin request still gets job binding, while a tenant-scoped
    request additionally gets an exact tenant match.
    """
    item_metadata = getattr(item, "metadata", None) or {}
    batch_id = str(item_metadata.get("batch_id") or "").strip()
    if batch_id:
        if item.job_id != f"translation-batch:{batch_id}":
            return False
        expected_tenant = str(item_metadata.get("tenant_id") or "").strip()
        request_tenant = (tenant_id or "").strip()
        if request_tenant and expected_tenant != request_tenant:
            return False
        coordinator_job_id = ""
    else:
        session = get_session_factory()()
        try:
            api_job = session.query(Job).filter(Job.job_id == item.job_id).first()
            if api_job is None:
                return False
            expected_tenant = str(getattr(api_job, "tenant_id", None) or "").strip()
            request_tenant = (tenant_id or "").strip()
            if request_tenant and expected_tenant != request_tenant:
                return False
            coordinator_job_id = str(
                getattr(api_job, "coordinator_job_id", None) or ""
            ).strip()
        finally:
            session.close()

    event = _load_review_custody_event(custody_event_id)
    if event is None or not getattr(event, "event_hash", ""):
        return False
    bound_job = getattr(event, "job", None)
    if bound_job is None:
        if coordinator_job_id:
            return False
        if str(getattr(event, "tenant_id", "") or "").strip() != (
            expected_tenant or "__single_tenant__"
        ):
            return False
    else:
        if str(getattr(bound_job, "job_id", "")) != coordinator_job_id:
            return False
        bound_tenant = str(getattr(bound_job, "tenant_id", None) or "").strip()
        if expected_tenant and bound_tenant != expected_tenant:
            return False
    if (
        str(getattr(event, "event_type", "") or "").strip()
        not in _CERTIFICATION_CUSTODY_EVENT_TYPES
    ):
        return False
    event_data = getattr(event, "data", None)
    if not isinstance(event_data, dict):
        return False
    if str(event_data.get("review_id") or "").strip() != str(item.review_id):
        return False
    if bound_job is None and str(event_data.get("api_job_id") or "") != item.job_id:
        return False
    event_tenant = str(event_data.get("tenant_id") or "").strip()
    if expected_tenant and event_tenant != expected_tenant:
        return False
    return True


def _emit_review_certification_event(
    item,
    *,
    tenant_id: str | None,
    reviewer: str,
    auth_method: str,
    proof_id: str,
):
    """Seal server-verified authorization against the exact coordinator job."""
    from api.coordinator_bridge import ensure_coordinator_django

    ensure_coordinator_django()
    item_metadata = getattr(item, "metadata", None) or {}
    batch_id = str(item_metadata.get("batch_id") or "").strip()
    if batch_id:
        expected_tenant = str(item_metadata.get("tenant_id") or "").strip()
        if item.job_id != f"translation-batch:{batch_id}":
            raise ValueError("translation review batch binding changed")
        if tenant_id and expected_tenant != tenant_id:
            raise ValueError("review tenant binding changed")
        coordinator_job_id = ""
    else:
        session = get_session_factory()()
        try:
            api_job = session.query(Job).filter(Job.job_id == item.job_id).first()
            if api_job is None:
                raise ValueError("review job no longer exists")
            expected_tenant = str(getattr(api_job, "tenant_id", None) or "").strip()
            if tenant_id and expected_tenant != tenant_id:
                raise ValueError("review tenant binding changed")
            coordinator_job_id = str(
                getattr(api_job, "coordinator_job_id", None) or ""
            ).strip()
        finally:
            session.close()

    event_data = {
        "review_id": item.review_id,
        "api_job_id": item.job_id,
        "tenant_id": expected_tenant,
        "reviewer_id": reviewer,
        "auth_method": auth_method,
        "proof_id": proof_id,
    }
    event_type = (
        "translation_review_authorized"
        if batch_id
        else "review_certification_authorized"
    )
    if batch_id:
        event_data["batch_id"] = batch_id
    if coordinator_job_id:
        from jobs.models import Job as CoordinatorJob  # type: ignore[import-not-found]
        from jobs.rls import worker_tenant_rls_context  # type: ignore[import-not-found]
        from jobs.tasks import _record_custody_event  # type: ignore[import-not-found]

        with worker_tenant_rls_context(expected_tenant):
            coordinator_job = CoordinatorJob.objects.get(job_id=coordinator_job_id)
            event = _record_custody_event(
                coordinator_job,
                coordinator_job.source_hash[:16]
                if coordinator_job.source_hash
                else str(coordinator_job.job_id)[:16],
                event_type,
                data=event_data,
                worker_hostname="api-review",
            )
    else:
        from jobs.rls import control_plane_rls_context  # type: ignore[import-not-found]
        from jobs.tasks import (  # type: ignore[import-not-found]
            _record_control_plane_custody_event,
        )

        with control_plane_rls_context():
            event = _record_control_plane_custody_event(
                tenant_id=expected_tenant or "__single_tenant__",
                document_id=f"batch:{batch_id}"
                if batch_id
                else f"review-{item.review_id}",
                event_type=event_type,
                data=event_data,
                worker_hostname="api-review",
            )
    return event, expected_tenant


def _promote_translation_review(
    item, custody_event_id: str, tenant_id: str | None
) -> None:
    """Idempotently persist the coordinator half of certification."""
    batch_id = str((getattr(item, "metadata", None) or {}).get("batch_id") or "")
    if not batch_id:
        return
    from jobs.rls import (  # type: ignore[import-not-found]
        control_plane_rls_context,
        tenant_rls_context,
    )

    from ocr_local.translation.batch import promote_batch_certified

    context = (
        tenant_rls_context(tenant_id) if tenant_id else control_plane_rls_context()
    )
    with context:
        promote_batch_certified(
            batch_id,
            custody_event_id=custody_event_id,
            tenant_id=tenant_id or None,
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
    reason: str | None = Query(
        default=None, max_length=64, description="Filter by review reason"
    ),
    status: str | None = Query(
        default=None,
        max_length=32,
        description="Filter by status (default: pending)",
    ),
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
    from api.authz import require_tenant_scope_or_platform

    # Fail-closed under multitenancy (RC-01 / R4-RC-13).
    scoped = require_tenant_scope_or_platform(request)
    allowed_job_ids = None if scoped is None else await _tenant_job_ids(scoped)

    # Default to pending if no status specified
    effective_status = status if status is not None else "pending"

    items = queue.list_all(
        limit=limit,
        offset=offset,
        status=effective_status,
        reason=reason,
        allowed_job_ids=allowed_job_ids,
    )
    total = queue.count(
        status=effective_status,
        reason=reason,
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
    """Return aggregate review queue statistics.

    Tenant-scoped keys only see aggregates for their own jobs (maintenance review).
    Platform admins and single-tenant mode retain global stats.
    """
    queue = _get_queue()
    from api.authz import require_tenant_scope_or_platform

    scoped = require_tenant_scope_or_platform(request)
    allowed_job_ids = None if scoped is None else await _tenant_job_ids(scoped)
    return ReviewStatsResponse(**queue.stats(allowed_job_ids=allowed_job_ids))


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


@router.post(
    "/evaluate",
    name="evaluate_and_enqueue_review",
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def evaluate_and_enqueue_review(
    request: Request,
    body: dict,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Evaluate exception-routing rules and enqueue HITL review items.

    Post-sweep HIGH: wires ExceptionRouter → ReviewQueue (previously evaluate-only).
    Body fields: ``job_id`` (required), optional ``validation``, ``classification``,
    ``handwriting``, ``extracted_text``.
    """
    from api.authz import require_tenant_scope_or_platform
    from api.job_manager import get_manager

    if not isinstance(body, dict):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_request",
                "message": "evaluate body must be a JSON object",
            },
        )

    for field_name in ("validation", "classification", "handwriting"):
        value = body.get(field_name)
        if value is not None and not isinstance(value, dict):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_request",
                    "message": f"{field_name} must be a JSON object when provided",
                },
            )

    raw_extracted_text = body.get("extracted_text", "")
    if raw_extracted_text is None:
        extracted_text = ""
    elif not isinstance(raw_extracted_text, str):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_request",
                "message": "extracted_text must be a string when provided",
            },
        )
    else:
        extracted_text = raw_extracted_text
    if len(extracted_text) > _MAX_EVALUATE_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "request_too_large",
                "message": (
                    "extracted_text exceeds the 102400-character evaluation limit"
                ),
            },
        )

    tenant_id = require_tenant_scope_or_platform(request)
    job_id = str((body or {}).get("job_id") or "").strip()
    if not job_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": "job_id is required for evaluate_and_enqueue",
            },
        )
    # Tenant pin: refuse foreign job_id (same isolation class as other review routes).
    if tenant_id is not None:
        job = get_manager().get_job(job_id, tenant_id=tenant_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "job_not_found",
                    "message": f"Job {job_id} not found for tenant scope.",
                },
            )
    er = _get_router()
    queue = _get_queue()
    decision, review_ids = er.evaluate_and_enqueue(
        job_id=job_id,
        review_queue=queue,
        validation_data=(body or {}).get("validation"),
        classification_data=(body or {}).get("classification"),
        handwriting_data=(body or {}).get("handwriting"),
        extracted_text=extracted_text,
    )
    return {
        "should_route": decision.should_route,
        "triggered_rules": list(decision.triggered_rules or []),
        "reasons": list(decision.reasons or []),
        "confidence": decision.confidence,
        "review_ids": review_ids,
    }


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
    from api.authz import require_tenant_scope_or_platform

    tenant_id = require_tenant_scope_or_platform(request)
    if not await _verify_review_item_tenant_authoritative(item, tenant_id, None):
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
    from api.authz import require_tenant_scope_or_platform

    tenant_id = require_tenant_scope_or_platform(request)
    queue = _get_queue()
    existing = queue.get(review_id)
    if existing is None or not await _verify_review_item_tenant_authoritative(
        existing, tenant_id, None
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "review_not_found",
                "message": f"Review item {review_id} not found.",
            },
        )
    if (
        existing.reason == ReviewReason.TRANSLATION_CERTIFICATION.value
        and body.status == ReviewStatus.APPROVED
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "strong_auth_certification_required",
                "message": "Translation certification must use the strong-auth certify endpoint.",
            },
        )

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
        501: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
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

    Certification records the selected auth method and a non-secret proof
    identifier, seals a real custody event, and promotes a bound translation
    batch when the review reason is translation certification. It does not
    make a broader legal or production-attestation claim beyond that durable
    product workflow.
    """
    _validate_review_id(review_id)

    queue = _get_queue()
    existing = queue.get(review_id)
    from api.authz import require_tenant_scope_or_platform

    tenant_id = require_tenant_scope_or_platform(request)
    if existing is None or not await _verify_review_item_tenant_authoritative(
        existing, tenant_id, None
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "review_not_found",
                "message": f"Review item {review_id} not found.",
            },
        )

    if existing.status != ReviewStatus.PENDING:
        certification = (existing.metadata or {}).get("certification") or {}
        custody_event_id = str(certification.get("custody_event_id") or "").strip()
        if (
            existing.reason == ReviewReason.TRANSLATION_CERTIFICATION.value
            and existing.status == ReviewStatus.APPROVED
            and bool((existing.metadata or {}).get("certified"))
            and custody_event_id
        ):
            try:
                await asyncio.to_thread(
                    _promote_translation_review,
                    existing,
                    custody_event_id,
                    tenant_id,
                )
                if not queue.confirm_translation_promotion(
                    existing.review_id, custody_event_id
                ):
                    raise RuntimeError("translation promotion confirmation CAS lost")
            except Exception as exc:
                logger.exception("Could not resume translation certification")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "translation_certification_unavailable",
                        "message": "Translation certification recovery is pending.",
                    },
                ) from exc
            return ReviewItemResponse(**existing.to_dict())
        raise HTTPException(
            status_code=409,
            detail={
                "error": "already_decided",
                "message": f"Review item {review_id} is already '{existing.status}'.",
            },
        )

    if not os.environ.get("REVIEW_STRONG_AUTH_VERIFIER_URL", "").strip():
        raise HTTPException(
            status_code=501,
            detail={
                "error": "certify_verifier_not_configured",
                "message": "Strong-auth certification verifier is not configured.",
            },
        )

    identity = getattr(request.state, "identity", None)
    reviewer = str(getattr(identity, "subject", None) or "").strip()
    if not reviewer:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "certify_identity_required",
                "message": "Strong-auth certification requires a bound requester identity.",
            },
        )
    from api.strong_auth import StrongAuthUnavailable, verify_review_strong_auth

    try:
        proof = await asyncio.to_thread(
            verify_review_strong_auth,
            token=body.auth_token,
            auth_method=body.auth_method,
            review_id=review_id,
            job_id=existing.job_id,
            tenant_id=tenant_id or "",
            expected_subject=reviewer,
        )
    except StrongAuthUnavailable as exc:
        raise HTTPException(
            status_code=501,
            detail={
                "error": "certify_verifier_not_configured",
                "message": (
                    "Strong-auth certification could not be verified; "
                    "the review remains pending."
                ),
            },
        ) from exc

    notes = f"certified via {proof.auth_method}; proof_id={proof.proof_id}"
    if body.notes:
        notes = f"{notes}; notes={body.notes}"
    try:
        event, event_tenant = _emit_review_certification_event(
            existing,
            tenant_id=tenant_id,
            reviewer=proof.subject,
            auth_method=proof.auth_method,
            proof_id=proof.proof_id,
        )
        custody_event_id = str(event.event_hash)
    except Exception as exc:
        logger.exception("Could not seal review certification custody")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "custody_event_unavailable",
                "message": "Certification custody could not be durably recorded.",
            },
        ) from exc
    if not _custody_event_binds_review(existing, custody_event_id, event_tenant):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "custody_event_unbound",
                "message": "Server-emitted certification custody failed binding verification.",
            },
        )
    try:
        item = queue.certify(
            review_id=review_id,
            reviewer=reviewer,
            notes=notes,
            auth_method=body.auth_method,
            require_custody_event=True,
            custody_event_id=custody_event_id,
            refuse_synthetic_custody=True,
        )
    except ValueError as exc:
        if "already decided" in str(exc).lower() or "CAS" in str(exc):
            raise HTTPException(
                status_code=409,
                detail={"error": "already_decided", "message": str(exc)},
            ) from exc
        raise HTTPException(
            status_code=400,
            detail={"error": "certify_rejected", "message": str(exc)},
        ) from exc
    if item is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "review_not_found",
                "message": f"Review item {review_id} not found.",
            },
        )
    batch_id = str((existing.metadata or {}).get("batch_id") or "").strip()
    if batch_id:
        try:
            await asyncio.to_thread(
                _promote_translation_review,
                item,
                custody_event_id,
                event_tenant,
            )
            if not queue.confirm_translation_promotion(
                item.review_id, custody_event_id
            ):
                raise RuntimeError("translation promotion confirmation CAS lost")
        except Exception as exc:
            # Reconcile forward only. Coordinator promotion may have committed
            # even when the caller observed an error; rolling the review back
            # would permit rejection of an already-certified batch.
            logger.exception(
                "Could not confirm certified translation promotion; recovery pending"
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "translation_certification_unavailable",
                    "message": "Translation certification recovery is pending.",
                },
            ) from exc

    logger.info("Review certified: %s via %s", review_id, body.auth_method)
    return ReviewItemResponse(**item.to_dict())
