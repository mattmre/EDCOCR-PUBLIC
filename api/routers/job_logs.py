"""Per-job NDJSON log streaming endpoint (D7).

Exposes ``GET /api/v1/jobs/{job_id}/logs`` which streams the on-disk
NDJSON log records produced by :mod:`api.job_log_writer` during pipeline
execution.

The endpoint enforces tenant isolation: a caller scoped to ``tenant_a``
asking for a job owned by ``tenant_b`` receives ``404`` (not ``403``) so
that job-id enumeration cannot leak cross-tenant existence.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from api.database import Job, get_session_factory
from api.identity import require_role
from api.job_log_writer import (
    _LEVEL_PRIORITY,
    filter_records,
    job_log_path,
    parse_log_line,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["job-logs"])

_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12}$")
_DEFAULT_LIMIT = 500
_MAX_LIMIT = 5000

# 404 body shape used for both "no logs" and cross-tenant cases.
_NOT_FOUND_BODY = {"detail": "no job logs available", "code": "NO_PER_JOB_LOGS"}


def _validate_job_id(job_id: str) -> None:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id format")


def _request_tenant_id(request: Request):
    return getattr(request.state, "tenant_id", None)


def _job_belongs_to_tenant(job_id: str, tenant_id) -> bool:
    """Return True iff the job exists and is owned by ``tenant_id``.

    In single-tenant / unauthenticated mode (``tenant_id is None``) we
    skip the DB ownership check entirely so logs can stream without a
    backing ``Job`` row (mirrors the events endpoint behaviour).
    """
    if tenant_id is None:
        return True
    session = get_session_factory()()
    try:
        query = session.query(Job.job_id).filter(
            Job.job_id == job_id,
            Job.tenant_id == tenant_id,
        )
        return query.first() is not None
    finally:
        session.close()


def _parse_since(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        # Accept "Z" suffix as UTC (datetime.fromisoformat doesn't until 3.11+).
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid 'since' value: {raw!r} (expected ISO-8601)",
        )


def _validate_level(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    upper = raw.upper()
    if upper not in _LEVEL_PRIORITY:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid 'level' value: {raw!r}",
        )
    return upper


def _stream_records(
    path,
    *,
    since: Optional[datetime],
    level: Optional[str],
    limit: int,
) -> Iterator[bytes]:
    """Generator yielding filtered NDJSON bytes one record at a time.

    The file is read line-by-line so we never buffer the entire log in
    memory (large jobs may emit tens of thousands of records).  Filtering
    is applied per-record so the response budget (``limit``) is honoured
    on the *post-filter* count.
    """
    import json as _json

    emitted = 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                if emitted >= limit:
                    break
                obj = parse_log_line(raw)
                if obj is None:
                    continue
                # Reuse filter_records on a single-element list for parity.
                kept = filter_records([obj], since=since, level=level)
                if not kept:
                    continue
                yield (_json.dumps(kept[0], ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                emitted += 1
    except FileNotFoundError:
        # Race: file vanished between existence check and open -- emit nothing.
        return
    except OSError as exc:  # pragma: no cover - defensive
        logger.debug("job log read failed: %s", exc)
        return


@router.get("/api/v1/jobs/{job_id}/logs")
async def get_job_logs(
    request: Request,
    job_id: str,
    since: Optional[str] = Query(
        default=None,
        description="ISO-8601 timestamp; only records strictly after this are returned.",
    ),
    limit: int = Query(
        default=_DEFAULT_LIMIT,
        ge=1,
        le=_MAX_LIMIT,
        description="Max records to stream (1..5000).",
    ),
    level: Optional[str] = Query(
        default=None,
        description="Minimum level (DEBUG|INFO|WARN|ERROR).",
    ),
    _auth: None = Depends(require_role("viewer", "operator", "admin")),
):
    """Stream the per-job NDJSON log file.

    Returns ``404 NO_PER_JOB_LOGS`` if the file is missing or the job is
    owned by a different tenant.  Successful responses use
    ``application/x-ndjson`` with chunked transfer encoding.
    """
    _validate_job_id(job_id)

    since_dt = _parse_since(since)
    level_norm = _validate_level(level)

    tenant_id = _request_tenant_id(request)
    if not _job_belongs_to_tenant(job_id, tenant_id):
        # Tenant isolation: do NOT leak existence -- return 404, not 403.
        raise HTTPException(status_code=404, detail=_NOT_FOUND_BODY)

    path = job_log_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=_NOT_FOUND_BODY)

    return StreamingResponse(
        _stream_records(path, since=since_dt, level=level_norm, limit=limit),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
