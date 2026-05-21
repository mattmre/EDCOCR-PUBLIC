"""FastAPI dependencies for request parsing."""

from typing import Optional

from fastapi import Form, HTTPException, Query
from pydantic import ValidationError

from api import config
from api.models import JobSubmitRequest
from api.path_safety import validate_source_path_input

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

# Maximum number of results a single request may return.
MAX_PAGE_LIMIT = 200


def normalize_pagination(
    limit: int,
    offset: int,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> tuple[int, int]:
    """Pure normalization of ``(limit, offset, page, per_page)`` to ``(limit, offset)``.

    Safe to call outside FastAPI DI -- no ``Query(...)`` sentinels.  Use
    ``pagination_params`` (below) as the FastAPI dependency; use this helper
    in unit tests or backend code.
    """
    # Guard against Query/Form sentinels leaking through direct calls
    _page = page if isinstance(page, int) else None
    _per_page = per_page if isinstance(per_page, int) else None

    if _page is not None and _per_page is not None:
        limit = _per_page
        offset = (_page - 1) * _per_page
    elif _page is not None:
        offset = (_page - 1) * limit
    elif _per_page is not None:
        limit = _per_page
    return limit, offset


def pagination_params(
    limit: int = Query(50, ge=1, le=MAX_PAGE_LIMIT, description="Maximum number of results to return (1-200)."),
    offset: int = Query(0, ge=0, description="Number of results to skip."),
    # Backward-compatible aliases (deprecated, prefer limit/offset)
    page: Optional[int] = Query(None, ge=1, description="Page number (deprecated -- use offset)."),
    per_page: Optional[int] = Query(None, ge=1, le=MAX_PAGE_LIMIT, description="Page size (deprecated -- use limit)."),
) -> tuple[int, int]:
    """FastAPI dependency that normalizes pagination query parameters.

    Accepts the canonical ``limit``/``offset`` query parameters as well as
    the legacy ``page``/``per_page`` pair.  When legacy parameters are
    provided they are converted internally so that downstream code always
    receives a simple ``(limit, offset)`` tuple.
    """
    return normalize_pagination(limit, offset, page, per_page)


def parse_job_submit_form(
    source_path: Optional[str] = Form(None, description="Path to a file already on the server's NFS/S3 storage. Provide this OR a file upload."),
    priority: str = Form("normal", description="Processing priority: 'low', 'normal', or 'urgent'."),
    enable_docintel: bool = Form(False, description="Enable structured Document Intelligence extraction (requires GPU)."),
    docintel_mode: str = Form("full", description="DocIntel extraction mode: 'layout_only', 'tables_only', or 'full'."),
    skip_ocr: bool = Form(False, description="Skip OCR processing and only perform NLP/DocIntel if enabled. Assumes input is already textual."),
    processing_timeout_minutes: Optional[int] = Form(None, description="Custom timeout in minutes for this job before it is marked failed."),
    webhook_url: Optional[str] = Form(None, description="URL to receive a POST request when the job completes or fails."),
    webhook_secret: Optional[str] = Form(None, description="Secret used to sign the webhook payload (HMAC-SHA256)."),
) -> JobSubmitRequest:
    """Parse multipart form fields into a validated Pydantic model."""
    if source_path:
        source_path = str(
            validate_source_path_input(
                path_value=source_path,
                field_name="source_path",
                allowed_roots=(config.SOURCE_FOLDER,),
            )
        )
    # Validate webhook URL if provided (isinstance check guards against direct
    # calls in tests where Form(None) defaults are not resolved by FastAPI DI)
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
                detail={
                    "error": "invalid_webhook_url",
                    "message": str(exc),
                },
            )
    # Normalize Form defaults to None/bool for direct calls outside FastAPI DI
    wh_url = webhook_url if isinstance(webhook_url, str) else None
    wh_secret = webhook_secret if isinstance(webhook_secret, str) else None
    timeout_override = (
        processing_timeout_minutes
        if isinstance(processing_timeout_minutes, int)
        else None
    )
    skip_ocr_val = skip_ocr if isinstance(skip_ocr, bool) else False
    enable_docintel_val = enable_docintel if isinstance(enable_docintel, bool) else False
    try:
        return JobSubmitRequest(
            source_path=source_path,
            priority=priority,
            enable_docintel=enable_docintel_val,
            docintel_mode=docintel_mode,
            skip_ocr=skip_ocr_val,
            processing_timeout_minutes=timeout_override,
            webhook_url=wh_url,
            webhook_secret=wh_secret,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())
