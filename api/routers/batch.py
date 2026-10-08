"""Batch job management endpoints."""

import json
import re
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from api import config
from api.batch_manager import BatchManager
from api.batch_models import (
    BatchJobSummary,
    BatchLinks,
    BatchListResponse,
    BatchProgressInfo,
    BatchStatusResponse,
    BatchSubmitResponse,
)
from api.database import get_session_factory
from api.deps import pagination_params
from api.identity import require_role
from api.job_manager import get_manager
from api.limits import get_default_rate, get_submit_rate, limiter
from api.models import ErrorResponse
from api.path_safety import validate_source_path_input

_BATCH_ID_RE = re.compile(r"^batch_[0-9a-f]{12}$")
_MAX_SOURCE_PATHS_JSON_BYTES = 64 * 1024

router = APIRouter(prefix="/api/v1/jobs/batch", tags=["batch"])


def _get_batch_manager() -> BatchManager:
    return BatchManager(get_session_factory(), job_manager=get_manager())


def _validate_batch_source_paths(paths: list[str]) -> list[str]:
    """Resolve batch source paths against the configured ingest root."""
    return [
        str(
            validate_source_path_input(
                path_value=path,
                field_name=f"source_paths[{index}]",
                allowed_roots=(config.SOURCE_FOLDER,),
            )
        )
        for index, path in enumerate(paths)
    ]


def _batch_status_response(batch, jobs) -> BatchStatusResponse:
    """Build a BatchStatusResponse from a Batch and its child jobs."""
    job_summaries = [
        BatchJobSummary(
            job_id=j.job_id,
            source_file=j.source_file,
            status=j.status,
        )
        for j in jobs
    ]

    statuses = [j.status for j in jobs]
    total = len(statuses)
    progress = BatchProgressInfo(
        submitted=statuses.count("submitted"),
        processing=statuses.count("processing"),
        completed=statuses.count("completed"),
        failed=statuses.count("failed"),
        cancelled=statuses.count("cancelled"),
        percent_complete=round(
            sum(1 for s in statuses if s in ("completed", "failed", "cancelled"))
            / max(total, 1)
            * 100,
            1,
        ),
    )
    settings = batch.settings
    rejected_inputs = settings.get("rejected_inputs", [])
    if not isinstance(rejected_inputs, list):
        rejected_inputs = []

    return BatchStatusResponse(
        batch_id=batch.batch_id,
        status=batch.status,
        created_at=batch.created_at,
        completed_at=batch.completed_at,
        processing_time=batch.processing_time,
        total_jobs=batch.total_jobs,
        progress=progress,
        jobs=job_summaries,
        rejected_inputs=[str(item)[:255] for item in rejected_inputs],
        settings=settings,
        webhook_status=batch.webhook_status,
    )


# ------------------------------------------------------------------
# POST /api/v1/jobs/batch --- Submit batch
# ------------------------------------------------------------------


@router.post(
    "",
    name="submit_batch",
    response_model=BatchSubmitResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
)
@limiter.limit(get_submit_rate())
async def submit_batch(
    request: Request,
    files: list[UploadFile] = File(
        default=[], description="List of files to upload and process."
    ),
    source_paths: Optional[str] = Form(
        None,
        max_length=_MAX_SOURCE_PATHS_JSON_BYTES,
        description="JSON array string of server-side file paths.",
    ),
    priority: str = Form(
        "normal",
        max_length=16,
        description="Processing priority: 'low', 'normal', or 'urgent'.",
    ),
    enable_docintel: bool = Form(
        False,
        description="Enable structured Document Intelligence extraction (requires GPU).",
    ),
    docintel_mode: str = Form(
        "full",
        max_length=32,
        description="DocIntel extraction mode: 'layout_only', 'tables_only', or 'full'.",
    ),
    skip_ocr: bool = Form(
        False,
        description="Skip OCR and perform NLP/DocIntel only. Assumes textual input.",
    ),
    processing_timeout_minutes: Optional[int] = Form(
        None,
        ge=1,
        le=config.MAX_PROCESSING_TIMEOUT_MINUTES,
        description=(
            "Custom timeout in minutes for these jobs before they are marked "
            f"failed (1-{config.MAX_PROCESSING_TIMEOUT_MINUTES})."
        ),
    ),
    webhook_url: Optional[str] = Form(
        None,
        max_length=2_048,
        description="URL to receive a POST request when the batch completes or fails.",
    ),
    webhook_secret: Optional[str] = Form(
        None,
        max_length=256,
        description="Secret used to sign the webhook payload (HMAC-SHA256).",
    ),
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Submit multiple documents for OCR processing as a batch."""
    manager = _get_batch_manager()

    if skip_ocr:
        raise HTTPException(
            status_code=501,
            detail={
                "error": "unsupported_feature",
                "message": (
                    "skip_ocr is not supported by the standalone API pipeline; "
                    "use the coordinator text-only path"
                ),
            },
        )

    # Validate priority
    if priority not in ("urgent", "normal", "low"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_priority",
                "message": f"Invalid priority: {priority}. Must be urgent, normal, or low.",
            },
        )

    # S2-04 / post-sweep HIGH: urgent is gated (same as single-job submit).
    import os

    from api.authz import is_platform_admin
    from api.job_manager import normalize_job_priority

    perms = set(getattr(request.state, "tenant_permissions", None) or [])
    allow_urgent = (
        is_platform_admin(request)
        or "priority_urgent" in perms
        or os.environ.get("OCR_ALLOW_CLIENT_URGENT", "").lower() in ("1", "true", "yes")
    )
    requested = (priority or "normal").strip().lower()
    if requested == "urgent" and not allow_urgent:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "priority_forbidden",
                "message": (
                    "priority=urgent requires platform_admin, "
                    "priority_urgent permission, or OCR_ALLOW_CLIENT_URGENT "
                    "(S2-04 QoS gate)"
                ),
            },
        )
    priority = normalize_job_priority(priority, allow_urgent=allow_urgent)

    # Validate docintel_mode
    if docintel_mode not in ("layout_only", "tables_only", "full"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_docintel_mode",
                "message": f"Invalid docintel_mode: {docintel_mode}.",
            },
        )

    if processing_timeout_minutes is not None and processing_timeout_minutes < 1:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_processing_timeout_minutes",
                "message": "processing_timeout_minutes must be >= 1.",
            },
        )
    if (
        processing_timeout_minutes is not None
        and processing_timeout_minutes > config.MAX_PROCESSING_TIMEOUT_MINUTES
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_processing_timeout_minutes",
                "message": (
                    "processing_timeout_minutes must be <= "
                    f"{config.MAX_PROCESSING_TIMEOUT_MINUTES}."
                ),
            },
        )

    # Build the bounded upload view before resolving any server-side paths.
    # Cardinality is a cheap request-level guard and must run before path
    # validation, which can perform filesystem work for every supplied item.
    file_tuples: list[tuple[str, object]] = []
    for f in files:
        if f.filename:
            file_tuples.append((f.filename, f.file))

    max_batch_size = getattr(config, "MAX_BATCH_SIZE", 50)
    try:
        max_batch_size = int(max_batch_size)
    except (TypeError, ValueError):
        max_batch_size = 50
    if max_batch_size < 1:
        max_batch_size = 1

    # Parse source_paths JSON
    parsed_paths: list[str] = []
    if source_paths:
        from api.authz import require_platform_admin_for_shared_paths

        require_platform_admin_for_shared_paths(request)
        if len(source_paths.encode("utf-8")) > _MAX_SOURCE_PATHS_JSON_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "source_paths_too_large",
                    "message": "source_paths exceeds the request safety limit",
                },
            )
        try:
            parsed_paths = json.loads(source_paths)
            if not isinstance(parsed_paths, list):
                raise ValueError("source_paths must be a JSON array")
            for p in parsed_paths:
                if not isinstance(p, str) or not p.strip():
                    raise ValueError("Each source_path must be a non-empty string")
            if len(file_tuples) + len(parsed_paths) > max_batch_size:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "batch_too_large",
                        "message": (f"Batch size exceeds maximum of {max_batch_size}."),
                    },
                )
            parsed_paths = _validate_batch_source_paths(parsed_paths)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_source_paths",
                    "message": f"Invalid source_paths: {exc}",
                },
            )

    # Also enforce the count when source_paths was omitted (the source-path
    # branch performs the same check before its expensive resolver).
    if len(file_tuples) + len(parsed_paths) > max_batch_size:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "batch_too_large",
                "message": f"Batch size exceeds maximum of {max_batch_size}.",
            },
        )

    if not file_tuples and not parsed_paths:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": "Provide at least one file upload or source_path.",
            },
        )

    # Validate webhook URL
    if webhook_url and isinstance(webhook_url, str):
        from api.config import WEBHOOK_ALLOW_HTTP, WEBHOOK_ALLOW_PRIVATE
        from api.webhooks import validate_webhook_url

        try:
            webhook_url = validate_webhook_url(
                webhook_url,
                allow_http=WEBHOOK_ALLOW_HTTP,
                allow_private=WEBHOOK_ALLOW_PRIVATE,
            )
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_webhook_url",
                    "message": "Webhook URL failed validation.",
                },
            )

    try:
        batch, child_jobs = manager.submit_batch(
            files=file_tuples,
            source_paths=parsed_paths if parsed_paths else None,
            priority=priority,
            enable_docintel=enable_docintel,
            docintel_mode=docintel_mode,
            skip_ocr=skip_ocr,
            processing_timeout_minutes=processing_timeout_minutes,
            webhook_url=webhook_url if isinstance(webhook_url, str) else None,
            webhook_secret=webhook_secret if isinstance(webhook_secret, str) else None,
            tenant_id=_request_tenant_id(request),
        )
    except ValueError as exc:
        msg = str(exc)
        if "exceeds maximum" in msg.lower() or "upload exceeds" in msg.lower():
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "file_too_large"
                    if "upload exceeds" in msg.lower()
                    else "batch_too_large",
                    "message": msg,
                },
            )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": "The batch submission could not be accepted.",
            },
        )

    job_summaries = [
        BatchJobSummary(
            job_id=j.job_id,
            source_file=j.source_file,
            status=j.status,
        )
        for j in child_jobs
    ]

    return BatchSubmitResponse(
        batch_id=batch.batch_id,
        status=batch.status,
        created_at=batch.created_at,
        total_jobs=batch.total_jobs,
        priority=batch.priority,
        jobs=job_summaries,
        rejected_inputs=[
            str(item)[:255]
            for item in (batch.settings.get("rejected_inputs", []) or [])
            if isinstance(item, (str, int, float))
        ],
        links=BatchLinks(
            **{
                "self": str(
                    request.url_for("get_batch_status", batch_id=batch.batch_id)
                ),
                "jobs": str(request.url_for("list_jobs"))
                + f"?batch_id={batch.batch_id}",
            }
        ),
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs/batch --- List batches (must be before /{batch_id})
# ------------------------------------------------------------------


def _request_tenant_id(request: Request):
    """Return the request's authorized tenant scope for batch submission."""
    from api.authz import require_tenant_scope_or_platform

    return require_tenant_scope_or_platform(request)


def _is_platform_admin(request: Request) -> bool:
    """True when server-attached tenant_permissions includes platform_admin.

    Trusts only ``request.state.tenant_permissions`` (API key path), never free-form
    JWT ``identity.claims["permissions"]`` (R4-RC-11 / RC-04 amp).
    """
    from api.authz import is_platform_admin

    return is_platform_admin(request)


@router.get(
    "",
    name="list_batches",
    response_model=BatchListResponse,
)
@limiter.limit(get_default_rate())
async def list_batches(
    request: Request,
    status: Optional[str] = None,
    pagination: tuple[int, int] = Depends(pagination_params),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """List batches with optional status filter and pagination.

    Supports ``limit``/``offset`` query parameters (canonical) as well as
    the deprecated ``page``/``per_page`` pair for backward compatibility.
    Tenant-scoped keys only see their own batches (AEP r2-BE001).
    """
    limit, offset = pagination
    manager = _get_batch_manager()
    from api.authz import require_tenant_scope_or_platform

    # Fail-closed under multitenancy: no tenant binding ⇒ 403 (not fleet list).
    scoped = require_tenant_scope_or_platform(request)
    batches, total = manager.list_batches(
        status=status, limit=limit, offset=offset, tenant_id=scoped
    )

    batch_responses = []
    for batch in batches:
        batch_jobs = manager.get_batch_jobs(batch.batch_id, tenant_id=scoped)
        batch_responses.append(_batch_status_response(batch, batch_jobs))

    return BatchListResponse(
        batches=batch_responses,
        total=total,
        limit=limit,
        offset=offset,
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs/batch/{batch_id} --- Batch status
# ------------------------------------------------------------------


@router.get(
    "/{batch_id}",
    name="get_batch_status",
    response_model=BatchStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def get_batch_status(
    request: Request,
    batch_id: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Get status and progress of a batch."""
    if not _BATCH_ID_RE.match(batch_id):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_batch_id",
                "message": "Invalid batch ID format.",
            },
        )

    manager = _get_batch_manager()
    from api.authz import require_tenant_scope_or_platform

    scope = require_tenant_scope_or_platform(request)
    batch = manager.get_batch(batch_id, tenant_id=scope)
    if not batch:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "batch_not_found",
                "message": f"Batch {batch_id} not found.",
            },
        )

    jobs = manager.get_batch_jobs(batch_id, tenant_id=scope)
    return _batch_status_response(batch, jobs)


# ------------------------------------------------------------------
# DELETE /api/v1/jobs/batch/{batch_id} --- Cancel batch
# ------------------------------------------------------------------


@router.delete(
    "/{batch_id}",
    name="cancel_batch",
    response_model=BatchStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def cancel_batch(
    request: Request,
    batch_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Cancel all running jobs in a batch."""
    if not _BATCH_ID_RE.match(batch_id):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_batch_id",
                "message": "Invalid batch ID format.",
            },
        )

    manager = _get_batch_manager()
    from api.authz import require_tenant_scope_or_platform

    scope = require_tenant_scope_or_platform(request)
    batch = manager.cancel_batch(batch_id, tenant_id=scope)
    if not batch:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "batch_not_found",
                "message": f"Batch {batch_id} not found.",
            },
        )

    jobs = manager.get_batch_jobs(batch_id, tenant_id=scope)
    return _batch_status_response(batch, jobs)


# ------------------------------------------------------------------
# POST /api/v1/jobs/batch/{batch_id}/retry --- Retry failed jobs
# ------------------------------------------------------------------


@router.post(
    "/{batch_id}/retry",
    name="retry_batch",
    response_model=BatchStatusResponse,
    status_code=200,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
@limiter.limit(get_submit_rate())
async def retry_batch(
    request: Request,
    batch_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Retry all failed or cancelled jobs in a batch."""
    if not _BATCH_ID_RE.match(batch_id):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_batch_id",
                "message": "Invalid batch ID format.",
            },
        )

    manager = _get_batch_manager()
    from api.authz import require_tenant_scope_or_platform

    scope = require_tenant_scope_or_platform(request)
    try:
        result = manager.retry_batch(batch_id, tenant_id=scope)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "no_retryable_jobs", "message": str(exc)},
        )

    if not result:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "batch_not_found",
                "message": f"Batch {batch_id} not found.",
            },
        )

    batch, _new_jobs = result
    jobs = manager.get_batch_jobs(batch_id, tenant_id=scope)
    return _batch_status_response(batch, jobs)
