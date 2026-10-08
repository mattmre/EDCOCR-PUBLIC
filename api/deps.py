"""FastAPI dependencies for request parsing."""

from datetime import datetime, timezone
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
MAX_PAGE_OFFSET = 1_000_000


def utc_naive_datetime(value: Optional[datetime]) -> Optional[datetime]:
    """Match persisted UTC timestamps; timezone-free client values mean UTC."""
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


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
    # Guard against Query/Form sentinels leaking through direct calls.  ``bool``
    # is an ``int`` subclass, but accepting it here would turn malformed
    # internal calls into surprising page/offset values.
    _page = page if isinstance(page, int) and not isinstance(page, bool) else None
    _per_page = (
        per_page
        if isinstance(per_page, int) and not isinstance(per_page, bool)
        else None
    )

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")
    if limit < 1 or limit > MAX_PAGE_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_LIMIT}")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise ValueError("offset must be an integer")

    if _page is not None and _page > MAX_PAGE_OFFSET:
        raise ValueError("page exceeds the maximum supported pagination depth")
    if _page is not None and _page < 1:
        raise ValueError("page must be at least 1")
    if _per_page is not None and (_per_page < 1 or _per_page > MAX_PAGE_LIMIT):
        raise ValueError(f"per_page must be between 1 and {MAX_PAGE_LIMIT}")
    if offset < 0 or offset > MAX_PAGE_OFFSET:
        raise ValueError("offset exceeds the maximum supported pagination depth")

    if _page is not None and _per_page is not None:
        limit = _per_page
        offset = (_page - 1) * _per_page
    elif _page is not None:
        offset = (_page - 1) * limit
    elif _per_page is not None:
        limit = _per_page
    if offset > MAX_PAGE_OFFSET:
        raise ValueError("pagination offset exceeds the maximum supported depth")
    return limit, offset


def pagination_params(
    limit: int = Query(
        50,
        ge=1,
        le=MAX_PAGE_LIMIT,
        description="Maximum number of results to return (1-200).",
    ),
    offset: int = Query(
        0,
        ge=0,
        le=MAX_PAGE_OFFSET,
        description="Number of results to skip (maximum 1,000,000).",
    ),
    # Backward-compatible aliases (deprecated, prefer limit/offset)
    page: Optional[int] = Query(
        None,
        ge=1,
        le=MAX_PAGE_OFFSET,
        description="Page number (deprecated -- use offset).",
    ),
    per_page: Optional[int] = Query(
        None,
        ge=1,
        le=MAX_PAGE_LIMIT,
        description="Page size (deprecated -- use limit).",
    ),
) -> tuple[int, int]:
    """FastAPI dependency that normalizes pagination query parameters.

    Accepts the canonical ``limit``/``offset`` query parameters as well as
    the legacy ``page``/``per_page`` pair.  When legacy parameters are
    provided they are converted internally so that downstream code always
    receives a simple ``(limit, offset)`` tuple.
    """
    try:
        return normalize_pagination(limit, offset, page, per_page)
    except ValueError as exc:
        # Direct callers should receive ValueError; an HTTP dependency must
        # never turn malformed legacy aliases into a 500 response.
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_pagination", "message": str(exc)},
        ) from exc


def parse_job_submit_form(
    source_path: Optional[str] = Form(
        None,
        description="Path to a file already on the server's NFS/S3 storage. Provide this OR a file upload.",
    ),
    priority: str = Form(
        "normal", description="Processing priority: 'low', 'normal', or 'urgent'."
    ),
    enable_docintel: bool = Form(
        False,
        description="Enable structured Document Intelligence extraction (requires GPU).",
    ),
    docintel_mode: str = Form(
        "full",
        description="DocIntel extraction mode: 'layout_only', 'tables_only', or 'full'.",
    ),
    skip_ocr: bool = Form(
        False,
        description="Skip OCR processing and only perform NLP/DocIntel if enabled. Assumes input is already textual.",
    ),
    processing_timeout_minutes: Optional[int] = Form(
        None,
        ge=1,
        le=config.MAX_PROCESSING_TIMEOUT_MINUTES,
        description=(
            "Custom timeout in minutes for this job before it is marked failed "
            f"(1-{config.MAX_PROCESSING_TIMEOUT_MINUTES})."
        ),
    ),
    webhook_url: Optional[str] = Form(
        None,
        description="URL to receive a POST request when the job completes or fails.",
    ),
    webhook_secret: Optional[str] = Form(
        None,
        max_length=256,
        description="Secret used to sign the webhook payload (HMAC-SHA256).",
    ),
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
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_webhook_url",
                    "message": "Webhook URL failed validation.",
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
    enable_docintel_val = (
        enable_docintel if isinstance(enable_docintel, bool) else False
    )
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
