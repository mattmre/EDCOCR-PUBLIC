"""Job management endpoints."""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Literal, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from api import config
from api.database import Job, get_session_factory
from api.deps import pagination_params, parse_job_submit_form
from api.identity import require_role
from api.job_manager import JobManager, get_manager
from api.limits import get_default_rate, get_submit_rate, limiter
from api.models import (
    ErrorResponse,
    JobLinks,
    JobListResponse,
    JobProgress,
    JobResultResponse,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
)
from api.path_safety import ensure_path_within_roots
from api.quota import QuotaExceededError

logger = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12}$")

# Extended filter (D6) -- recognised values; unknown values are rejected.
_VALID_STATUS_FILTERS = {
    "submitted",
    "queued",
    "pending",
    "processing",
    "running",
    "completed",
    "failed",
    "cancelled",
}
_VALID_SORTS = {
    "submitted_at_desc",
    "submitted_at_asc",
    "duration_desc",
    "status",
}

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def _get_manager() -> JobManager:
    return get_manager()


def _request_tenant_id(request: Request) -> Optional[str]:
    """Return tenant scope for authenticated tenant keys."""
    return getattr(request.state, "tenant_id", None)


def _is_platform_admin(request: Request) -> bool:
    """Return True when the caller authenticated as a platform admin.

    Platform admins (``permissions`` includes ``platform_admin``) can pass an
    explicit ``tenant_id`` query param to scope a list to any tenant; regular
    callers are always pinned to their own tenant id.
    """
    permissions = getattr(request.state, "tenant_permissions", None) or []
    if "platform_admin" in permissions:
        return True
    identity = getattr(request.state, "identity", None)
    if identity is not None:
        claims = getattr(identity, "claims", {}) or {}
        if "platform_admin" in (claims.get("permissions") or []):
            return True
    return False


def _job_status_response(job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        priority=job.priority,
        source_file=job.source_file,
        progress=JobProgress(
            total_pages=job.total_pages or 0,
            pages_completed=job.pages_completed or 0,
            percent_complete=job.percent_complete(),
            current_stage=job.current_stage or "submitted",
        ),
        settings=job.settings,
        webhook_status=job.webhook_status,
    )


# ------------------------------------------------------------------
# POST /api/v1/jobs --- Submit job
# ------------------------------------------------------------------

@router.post(
    "",
    name="submit_job",
    response_model=JobSubmitResponse,
    status_code=201,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
@limiter.limit(get_submit_rate())
async def submit_job(
    request: Request,
    form_data: JobSubmitRequest = Depends(parse_job_submit_form),
    file: Optional[UploadFile] = File(None),
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Submit a document for OCR processing."""
    manager = _get_manager()
    tenant_id = _request_tenant_id(request)

    # Check queue capacity
    if not manager.check_queue_capacity():
        raise HTTPException(
            status_code=429,
            detail={"error": "queue_full", "message": "Job queue is at capacity. Please try again later."},
        )

    if not file and not form_data.source_path:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": "Provide file upload or source_path."},
        )

    try:
        upload_content = None
        upload_filename = None
        if file:
            upload_content = await file.read()
            if len(upload_content) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail={"error": "file_too_large",
                            "message": f"Upload exceeds {config.MAX_UPLOAD_SIZE_MB} MB limit."},
                )
            upload_filename = file.filename

        job = manager.submit(
            source_path=form_data.source_path,
            upload_filename=upload_filename,
            upload_content=upload_content,
            tenant_id=tenant_id,
            priority=form_data.priority,
            enable_docintel=form_data.enable_docintel,
            docintel_mode=form_data.docintel_mode,
            skip_ocr=form_data.skip_ocr,
            processing_timeout_minutes=form_data.processing_timeout_minutes,
            webhook_url=form_data.webhook_url,
            webhook_secret=form_data.webhook_secret,
        )
    except QuotaExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "message": str(exc),
                "details": {
                    "tenant_id": exc.tenant_id,
                    "limit_type": exc.limit_type,
                    "current": exc.current,
                    "maximum": exc.maximum,
                },
            },
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": str(exc)})
    except ValueError as exc:
        if "capacity" in str(exc).lower():
            raise HTTPException(
                status_code=429,
                detail={"error": "queue_full", "message": str(exc)},
            )
        raise HTTPException(status_code=400, detail={"error": "invalid_request", "message": str(exc)})

    return JobSubmitResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        priority=job.priority,
        source_file=job.source_file,
        estimated_pages=job.total_pages,
        links=JobLinks(**{
            "self": str(request.url_for("get_job_status", job_id=job.job_id)),
            "result": str(request.url_for("get_job_result", job_id=job.job_id)),
        }),
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs --- List jobs
# ------------------------------------------------------------------

@router.get("", name="list_jobs", response_model=JobListResponse)
@limiter.limit(get_default_rate())
async def list_jobs(
    request: Request,
    status: Optional[list[str]] = Query(
        default=None,
        description=(
            "Filter by one or more statuses (repeatable: ?status=running&"
            "status=completed). A single value also works (?status=failed)."
        ),
    ),
    batch_id: Optional[str] = Query(default=None, description="Filter by batch ID"),
    tenant_id: Optional[str] = Query(
        default=None,
        description="Filter by tenant ID (platform-admin only; non-admin callers are always scoped to their own tenant).",
    ),
    submitted_after: Optional[datetime] = Query(
        default=None,
        description="Return only jobs created at or after this ISO 8601 timestamp.",
    ),
    submitted_before: Optional[datetime] = Query(
        default=None,
        description="Return only jobs created at or before this ISO 8601 timestamp.",
    ),
    q: Optional[str] = Query(
        default=None,
        max_length=256,
        description="Substring search on source_filename and job_id (case-insensitive).",
    ),
    sort: Literal[
        "submitted_at_desc",
        "submitted_at_asc",
        "duration_desc",
        "status",
    ] = Query(
        default="submitted_at_desc",
        description="Sort order applied after filtering.",
    ),
    pagination: tuple[int, int] = Depends(pagination_params),
):
    """List jobs with optional filters, sorting, and pagination.

    Supports ``limit``/``offset`` query parameters (canonical) as well as
    the deprecated ``page``/``per_page`` pair for backward compatibility.

    Filter parameters (D6):

    - ``status``: single status (legacy) OR repeated multi-status filter.
      Both ``?status=failed`` and ``?status=running&status=completed`` are
      accepted; the repeated form takes precedence.
    - ``tenant_id``: only honoured for platform-admin callers; otherwise
      the request is silently scoped to the caller's own tenant id.
    - ``submitted_after`` / ``submitted_before``: ISO 8601 datetimes.
    - ``q``: case-insensitive substring match on ``source_file`` and
      ``job_id`` with ``%``/``_`` escaped.
    - ``sort``: one of ``submitted_at_desc`` (default), ``submitted_at_asc``,
      ``duration_desc``, ``status``.
    """
    limit, offset = pagination
    manager = _get_manager()
    caller_tenant_id = _request_tenant_id(request)

    # Normalise multi-status filter.  FastAPI maps repeated ?status=foo&
    # status=bar into a list; a single ?status=foo also arrives as a one-
    # element list because the parameter type is ``Optional[list[str]]``.
    if status is None:
        status_list: Optional[list[str]] = None
    else:
        cleaned = [s for s in status if s]
        status_list = cleaned or None

    if status_list:
        for value in status_list:
            if value not in _VALID_STATUS_FILTERS:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "invalid_status",
                        "message": f"Unknown status filter value: {value!r}",
                    },
                )

    if sort not in _VALID_SORTS:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_sort", "message": f"Unknown sort value: {sort!r}"},
        )

    if (
        submitted_after is not None
        and submitted_before is not None
        and submitted_after > submitted_before
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_date_range",
                "message": "submitted_after must be <= submitted_before",
            },
        )

    # Tenant isolation (gotcha #80): platform-admin callers may target a
    # specific tenant via ?tenant_id=; everyone else is forced to their
    # caller-tenant scope.  When tenant_id is None on both sides the call
    # falls back to single-tenant / anonymous behaviour.
    if caller_tenant_id is not None:
        if tenant_id is not None and tenant_id != caller_tenant_id:
            if not _is_platform_admin(request):
                # Silently rewrite to the caller's tenant.  Returning 403
                # would leak that the requested tenant exists.
                tenant_id = caller_tenant_id
        else:
            tenant_id = caller_tenant_id
    # caller_tenant_id is None: respect explicit tenant_id only when
    # platform-admin; otherwise drop it so we don't accidentally let an
    # unauthenticated caller filter by a foreign tenant.
    elif tenant_id is not None and not _is_platform_admin(request):
        tenant_id = None

    jobs, total = manager.list_jobs(
        batch_id=batch_id,
        tenant_id=tenant_id,
        limit=limit,
        offset=offset,
        status_in=status_list,
        submitted_after=submitted_after,
        submitted_before=submitted_before,
        q=q,
        sort=sort,
    )
    return JobListResponse(
        jobs=[_job_status_response(j) for j in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id} --- Job status
# ------------------------------------------------------------------

@router.get(
    "/{job_id}",
    name="get_job_status",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def get_job_status(request: Request, job_id: str):
    """Get status and progress of a specific job."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail={"error": "invalid_job_id", "message": "Invalid job ID format."})
    manager = _get_manager()
    job = manager.get_job(job_id, tenant_id=_request_tenant_id(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    return _job_status_response(job)


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/result --- Result metadata
# ------------------------------------------------------------------

@router.get(
    "/{job_id}/result",
    name="get_job_result",
    response_model=JobResultResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def get_job_result(request: Request, job_id: str):
    """Get result metadata and artifact links for a completed job."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail={"error": "invalid_job_id", "message": "Invalid job ID format."})
    manager = _get_manager()
    job = manager.get_job(job_id, tenant_id=_request_tenant_id(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    if job.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "job_not_complete",
                "message": f"Job is {job.status}. Progress: {job.percent_complete()}%",
            },
        )

    artifacts = manager.get_result_artifacts(job_id)
    download_links = {
        k: str(request.url_for("download_artifact", job_id=job_id)) + "?" + urlencode({"type": k})
        for k in artifacts
    }

    return JobResultResponse(
        job_id=job.job_id,
        status=job.status,
        completed_at=job.completed_at,
        processing_time_seconds=job.processing_time,
        artifacts=download_links,
        metadata={"pages_processed": job.pages_completed or 0},
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/result/download --- Download artifact
# ------------------------------------------------------------------

@router.get(
    "/{job_id}/result/download",
    name="download_artifact",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def download_artifact(request: Request, job_id: str, type: str = "pdf"):
    """Download a specific result artifact (pdf, text, structure)."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail={"error": "invalid_job_id", "message": "Invalid job ID format."})
    manager = _get_manager()
    job = manager.get_job(job_id, tenant_id=_request_tenant_id(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail={"error": "job_not_complete", "message": f"Job is {job.status}."},
        )

    artifacts = manager.get_result_artifacts(job_id)
    if type not in artifacts:
        raise HTTPException(
            status_code=404,
            detail={"error": "artifact_not_found", "message": f"No '{type}' artifact available."},
        )

    artifact_path = artifacts[type]
    if job.result_path:
        artifact_path = str(
            ensure_path_within_roots(
                path_value=artifact_path,
                field_name="artifact_path",
                allowed_roots=[job.result_path],
            ),
        )

    media_types = {"pdf": "application/pdf", "text": "text/plain", "structure": "application/json"}
    return FileResponse(
        path=artifact_path,
        media_type=media_types.get(type, "application/octet-stream"),
        filename=f"{job_id}.{type}",
    )


# ------------------------------------------------------------------
# POST /api/v1/jobs/{job_id}/retry --- Retry failed job
# ------------------------------------------------------------------

@router.post(
    "/{job_id}/retry",
    name="retry_job",
    response_model=JobSubmitResponse,
    status_code=201,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
@limiter.limit(get_submit_rate())
async def retry_job(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Retry a failed or cancelled job by re-submitting with the same source."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail={"error": "invalid_job_id", "message": "Invalid job ID format."})
    manager = _get_manager()

    if not manager.check_queue_capacity():
        raise HTTPException(
            status_code=429,
            detail={"error": "queue_full", "message": "Job queue is at capacity. Please try again later."},
        )

    try:
        new_job = manager.retry_job(job_id, tenant_id=_request_tenant_id(request))
    except QuotaExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "message": str(exc),
                "details": {
                    "tenant_id": exc.tenant_id,
                    "limit_type": exc.limit_type,
                    "current": exc.current,
                    "maximum": exc.maximum,
                },
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "invalid_state", "message": str(exc)},
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "source_missing", "message": str(exc)},
        )

    if not new_job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )

    return JobSubmitResponse(
        job_id=new_job.job_id,
        status=new_job.status,
        created_at=new_job.created_at,
        priority=new_job.priority,
        source_file=new_job.source_file,
        estimated_pages=new_job.total_pages,
        links=JobLinks(**{
            "self": str(request.url_for("get_job_status", job_id=new_job.job_id)),
            "result": str(request.url_for("get_job_result", job_id=new_job.job_id)),
        }),
    )


# ------------------------------------------------------------------
# POST /api/v1/jobs/{job_id}/redetect-language --- Re-run language detection
# ------------------------------------------------------------------


@router.post(
    "/{job_id}/redetect-language",
    name="redetect_language",
    status_code=202,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def redetect_language(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Re-run language detection on an already-completed job's output PDF.

    Enqueues a background thread that re-runs per-span language detection
    against the embedded text layer of the job's output PDF.  OCR is NOT
    re-executed; this is a cheap, detector-only pass that writes a new
    ``.language.json`` sidecar and emits a ``LANGUAGE_REDETECTED`` custody
    event.

    Responses
    ---------
    - ``202`` -- redetect queued; returns ``{"status": "queued", "job_id": ...}``
    - ``404`` -- job not found (including when tenant scoping hides it)
    - ``409`` -- job is not in ``completed`` state
    """
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )

    manager = _get_manager()
    job = manager.get_job(job_id, tenant_id=_request_tenant_id(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "job_not_complete",
                "message": (
                    "Language re-detection requires a completed job; "
                    f"current status is {job.status}."
                ),
            },
        )

    artifacts = manager.get_result_artifacts(job_id)
    pdf_path = artifacts.get("pdf")
    output_base_dir = job.result_path or config.OUTPUT_FOLDER

    if not pdf_path:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "pdf_missing",
                "message": "Completed job has no PDF artifact to re-detect.",
            },
        )

    import threading

    def _run() -> None:
        try:
            from ocr_local.features.language_detection import redetect_document

            redetect_document(
                pdf_path=pdf_path,
                output_json_path="",
                output_base_dir=str(output_base_dir),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Background redetect failed for %s: %s", job_id, exc)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "queued", "job_id": job_id}


# ------------------------------------------------------------------
# DELETE /api/v1/jobs/{job_id} --- Cancel job
# ------------------------------------------------------------------

@router.delete(
    "/{job_id}",
    name="cancel_job",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def cancel_job(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Cancel a running job."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail={"error": "invalid_job_id", "message": "Invalid job ID format."})
    manager = _get_manager()
    job = manager.cancel_job(job_id, tenant_id=_request_tenant_id(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    return _job_status_response(job)


# ------------------------------------------------------------------
# SSE helpers
# ------------------------------------------------------------------

# Terminal job states that end the SSE stream.
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


def _sse_event(event_type: str, data: dict) -> str:
    """Format a single Server-Sent Event frame.

    Returns a string in the SSE wire format:
        event: <type>\\ndata: <json>\\n\\n
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _sse_generator(
    job_id: str,
    request: Request,
    session_factory,
    tenant_id: Optional[str] = None,
):
    """Yield SSE frames tracking *job_id* until it reaches a terminal state.

    Uses an asyncio.Event notification hub (``api.sse_notifier``) so that
    job-status changes wake the generator immediately instead of requiring
    a fixed-interval poll.  A 30-second fallback timeout ensures the stream
    still advances even when a signal is missed (process restart, etc.).
    """
    from api.sse_notifier import create_job_event, remove_job_event

    last_status: Optional[str] = None
    last_pages: Optional[int] = None
    timeout = float(os.environ.get("SSE_STREAM_TIMEOUT", "1800"))  # 30 min
    # default raised from 2s to 30s -- events fire instantly on change;
    # this interval is now only a safety-net fallback.
    fallback_interval = float(os.environ.get("SSE_POLL_INTERVAL", "30.0"))
    start = time.monotonic()

    _evt = create_job_event(job_id)
    try:
        while time.monotonic() - start < timeout:
            # Respect client disconnects
            if await request.is_disconnected():
                return

            try:
                session = session_factory()
                try:
                    query = session.query(Job).filter(Job.job_id == job_id)
                    if tenant_id is not None:
                        query = query.filter(Job.tenant_id == tenant_id)
                    job = query.first()
                finally:
                    session.close()

                if not job:
                    yield _sse_event("error", {"job_id": job_id, "error": "Job not found"})
                    yield _sse_event("done", {"job_id": job_id})
                    return

                current_status = job.status

                # Emit a status event whenever the status changes.
                if current_status != last_status:
                    yield _sse_event("status", {
                        "job_id": job_id,
                        "status": current_status,
                        "previous_status": last_status,
                    })
                    last_status = current_status

                # Emit progress when in a non-terminal state and page counts changed.
                if current_status not in _TERMINAL_STATES:
                    pages_completed = job.pages_completed or 0
                    if pages_completed != last_pages:
                        progress_data: dict = {
                            "job_id": job_id,
                            "status": current_status,
                            "pages_completed": pages_completed,
                        }
                        if job.total_pages:
                            progress_data["total_pages"] = job.total_pages
                            progress_data["percent"] = round(
                                pages_completed / job.total_pages * 100, 1,
                            )
                        if job.current_stage:
                            progress_data["current_stage"] = job.current_stage
                        yield _sse_event("progress", progress_data)
                        last_pages = pages_completed

                # Terminal state handling
                if current_status in _TERMINAL_STATES:
                    if current_status == "completed":
                        result_data: dict = {
                            "job_id": job_id,
                            "status": "completed",
                        }
                        if job.result_path:
                            result_data["result_path"] = job.result_path
                        if job.processing_time is not None:
                            result_data["processing_time_seconds"] = job.processing_time
                        if job.pages_completed is not None:
                            result_data["pages_processed"] = job.pages_completed
                        yield _sse_event("result", result_data)
                    elif current_status == "failed":
                        yield _sse_event("error", {
                            "job_id": job_id,
                            "status": "failed",
                            "error": job.error_message or "Unknown error",
                        })
                    # cancelled falls through to done

                    yield _sse_event("done", {"job_id": job_id})
                    return

            except Exception as exc:
                logger.warning("SSE generator error for %s: %s", job_id, exc)
                yield _sse_event("error", {"job_id": job_id, "error": str(exc)})

            # wait for event-driven notification or fallback timeout
            try:
                await asyncio.wait_for(
                    asyncio.shield(_evt.wait()), timeout=fallback_interval,
                )
                _evt.clear()  # reset for next signal
            except asyncio.TimeoutError:
                pass  # fallback poll — query DB on next iteration

        # Stream timeout
        yield _sse_event("error", {"job_id": job_id, "error": "Stream timeout"})
        yield _sse_event("done", {"job_id": job_id})
    finally:
        remove_job_event(job_id, _evt)


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/stream --- SSE streaming
# ------------------------------------------------------------------

@router.get(
    "/{job_id}/stream",
    name="stream_job_progress",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def stream_job_progress(request: Request, job_id: str):
    """Stream job progress via Server-Sent Events (SSE).

    Sends events for status changes, page completion, and final result.

    Event types:
    - **status** -- emitted when job status changes (submitted, processing, etc.)
    - **progress** -- periodic progress with pages completed / total pages
    - **result** -- emitted when job completes successfully
    - **error** -- emitted on job failure or stream error
    - **done** -- final event, stream closes after this
    """
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )

    session_factory = get_session_factory()
    tenant_id = _request_tenant_id(request)

    # Verify job exists (and belongs to this tenant) before opening the stream.
    session = session_factory()
    try:
        query = session.query(Job).filter(Job.job_id == job_id)
        if tenant_id is not None:
            query = query.filter(Job.tenant_id == tenant_id)
        job = query.first()
    finally:
        session.close()

    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )

    return StreamingResponse(
        _sse_generator(job_id, request, session_factory, tenant_id=tenant_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
