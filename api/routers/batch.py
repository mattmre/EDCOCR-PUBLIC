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
            sum(
                1 for s in statuses if s in ("completed", "failed", "cancelled")
            )
            / max(total, 1)
            * 100,
            1,
        ),
    )

    return BatchStatusResponse(
        batch_id=batch.batch_id,
        status=batch.status,
        created_at=batch.created_at,
        completed_at=batch.completed_at,
        processing_time=batch.processing_time,
        total_jobs=batch.total_jobs,
        progress=progress,
        jobs=job_summaries,
        settings=batch.settings,
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
    },
)
@limiter.limit(get_submit_rate())
async def submit_batch(
    request: Request,
    files: list[UploadFile] = File(default=[], description="List of files to upload and process."),
    source_paths: Optional[str] = Form(None, description="JSON array string of server-side file paths."),
    priority: str = Form("normal", description="Processing priority: 'low', 'normal', or 'urgent'."),
    enable_docintel: bool = Form(False, description="Enable structured Document Intelligence extraction (requires GPU)."),
    docintel_mode: str = Form("full", description="DocIntel extraction mode: 'layout_only', 'tables_only', or 'full'."),
    skip_ocr: bool = Form(False, description="Skip OCR and perform NLP/DocIntel only. Assumes textual input."),
    processing_timeout_minutes: Optional[int] = Form(None, description="Custom timeout in minutes for these jobs before they are marked failed."),
    webhook_url: Optional[str] = Form(None, description="URL to receive a POST request when the batch completes or fails."),
    webhook_secret: Optional[str] = Form(None, description="Secret used to sign the webhook payload (HMAC-SHA256)."),
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Submit multiple documents for OCR processing as a batch."""
    manager = _get_batch_manager()

    # Validate priority
    if priority not in ("urgent", "normal", "low"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_priority",
                "message": f"Invalid priority: {priority}. Must be urgent, normal, or low.",
            },
        )

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

    # Parse source_paths JSON
    parsed_paths: list[str] = []
    if source_paths:
        try:
            parsed_paths = json.loads(source_paths)
            if not isinstance(parsed_paths, list):
                raise ValueError("source_paths must be a JSON array")
            for p in parsed_paths:
                if not isinstance(p, str) or not p.strip():
                    raise ValueError("Each source_path must be a non-empty string")
            parsed_paths = _validate_batch_source_paths(parsed_paths)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_source_paths",
                    "message": f"Invalid source_paths: {exc}",
                },
            )

    # Read uploaded files
    file_tuples: list[tuple[str, bytes]] = []
    for f in files:
        if f.filename:
            content = await f.read()
            if len(content) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "file_too_large",
                        "message": f"File {f.filename} exceeds {config.MAX_UPLOAD_SIZE_MB} MB limit.",
                    },
                )
            file_tuples.append((f.filename, content))

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
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_webhook_url", "message": str(exc)},
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
        )
    except ValueError as exc:
        msg = str(exc)
        if "exceeds maximum" in msg.lower():
            raise HTTPException(
                status_code=413,
                detail={"error": "batch_too_large", "message": msg},
            )
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": msg},
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
):
    """List batches with optional status filter and pagination.

    Supports ``limit``/``offset`` query parameters (canonical) as well as
    the deprecated ``page``/``per_page`` pair for backward compatibility.
    """
    limit, offset = pagination
    manager = _get_batch_manager()
    batches, total = manager.list_batches(status=status, limit=limit, offset=offset)

    batch_responses = []
    for batch in batches:
        batch_jobs = manager.get_batch_jobs(batch.batch_id)
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
async def get_batch_status(request: Request, batch_id: str):
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
    batch = manager.get_batch(batch_id)
    if not batch:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "batch_not_found",
                "message": f"Batch {batch_id} not found.",
            },
        )

    jobs = manager.get_batch_jobs(batch_id)
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
    batch = manager.cancel_batch(batch_id)
    if not batch:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "batch_not_found",
                "message": f"Batch {batch_id} not found.",
            },
        )

    jobs = manager.get_batch_jobs(batch_id)
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
    try:
        result = manager.retry_batch(batch_id)
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
    jobs = manager.get_batch_jobs(batch_id)
    return _batch_status_response(batch, jobs)
