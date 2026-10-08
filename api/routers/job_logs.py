"""Per-job NDJSON log streaming endpoint (D7).

Exposes ``GET /api/v1/jobs/{job_id}/logs`` which streams the on-disk
NDJSON log records produced by :mod:`api.job_log_writer` during pipeline
execution.

The endpoint enforces tenant isolation: a caller scoped to ``tenant_a``
asking for a job owned by ``tenant_b`` receives ``404`` (not ``403``) so
that job-id enumeration cannot leak cross-tenant existence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from api.database import Job, get_session_factory
from api.identity import require_role
from api.job_log_writer import (
    _LEVEL_PRIORITY,
    _MAX_LOG_RECORD_BYTES,
    filter_records,
    job_log_path,
    parse_log_line,
)
from api.limits import get_default_rate, limiter
from api.path_safety import _open_regular_file_under_root

logger = logging.getLogger(__name__)

router = APIRouter(tags=["job-logs"])

_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12}$")
_DEFAULT_LIMIT = 500
_MAX_LIMIT = 5000
_MAX_SCAN_BYTES = 4 * 1024 * 1024
_CURSOR_RE = re.compile(r"^v1\.([0-9a-f]+)\.([0-9a-f]+)\.([0-9a-f]+)\.([0-9a-f]{64})\.([01])$")

# 404 body shape used for both "no logs" and cross-tenant cases.
_NOT_FOUND_BODY = {"detail": "no job logs available", "code": "NO_PER_JOB_LOGS"}


def _validate_job_id(job_id: str) -> None:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id format")


def _request_tenant_id(request: Request):
    """Return the request's authorized tenant scope.

    A missing tenant is only a valid global scope for legacy single-tenant
    mode or an explicitly server-authorized platform admin.  OAuth callers
    without a tenant mapping must not turn this log endpoint into a fleet-wide
    read surface when multitenancy is enabled.
    """
    from api.authz import require_tenant_scope_or_platform

    return require_tenant_scope_or_platform(request)


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


def _open_log_descriptor(path) -> int:
    """Open a log by descriptor so a path swap cannot redirect the stream."""
    fd = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise OSError("job log is not a single-link regular file")
        return fd
    except Exception:
        os.close(fd)
        raise


def _fingerprint(fh, offset: int) -> str:
    """Bind a position to the file contents without trusting a client path."""
    position = fh.tell()
    try:
        digest = hashlib.sha256()
        fh.seek(0)
        digest.update(fh.read(min(64, offset)))
        fh.seek(max(0, offset - 64))
        digest.update(fh.read(min(64, offset)))
        return digest.hexdigest()
    finally:
        fh.seek(position)


def _open_job_log_descriptor(job_id: str) -> int:
    """Prefer the pipeline's job directory, retaining legacy flat-log reads.

    Open every component relative to the configured output root so a symlink
    in a job/log directory cannot redirect an authorized read into another job.
    """
    legacy_path = job_log_path(job_id)
    output_root = legacy_path.parents[2]
    relative_path = legacy_path.relative_to(output_root)
    try:
        fd, _ = _open_regular_file_under_root(output_root, f"{job_id}/{relative_path.as_posix()}")
        return fd
    except FileNotFoundError:
        fd, _ = _open_regular_file_under_root(output_root, relative_path)
        return fd


def _read_records(
    file_descriptor: int,
    *,
    since: Optional[datetime],
    level: Optional[str],
    limit: int,
    cursor: Optional[str],
) -> tuple[bytes, str, bool]:
    """Read a bounded file slice and return its safe continuation position."""
    emitted = 0
    output: list[bytes] = []
    reset = False
    discarding = False
    with os.fdopen(file_descriptor, "rb") as fh:
        file_stat = os.fstat(fh.fileno())
        offset = 0
        if cursor:
            match = _CURSOR_RE.fullmatch(cursor)
            if match is None:
                raise HTTPException(status_code=422, detail="Invalid log cursor")
            device, inode, candidate = (int(value, 16) for value in match.groups()[:3])
            if (
                device == file_stat.st_dev
                and inode == file_stat.st_ino
                and candidate <= file_stat.st_size
                and match.group(4) == _fingerprint(fh, candidate)
            ):
                offset = candidate
                discarding = match.group(5) == "1"
            else:
                reset = True
        fh.seek(offset)
        scanned = 0
        while emitted < limit and scanned < _MAX_SCAN_BYTES:
            start = fh.tell()
            raw = fh.readline(_MAX_LOG_RECORD_BYTES + 1)
            if not raw:
                break
            scanned += len(raw)
            if discarding:
                if raw.endswith(b"\n"):
                    discarding = False
                continue
            if not raw.endswith(b"\n"):
                if len(raw) <= _MAX_LOG_RECORD_BYTES and fh.tell() >= file_stat.st_size:
                    fh.seek(start)  # an unfinished writer line is retried next time
                    break
                # Resume after the end of this oversized record, even across pages.
                discarding = True
                continue
            if len(raw) > _MAX_LOG_RECORD_BYTES:
                logger.warning("Skipping oversized job log record from %s", file_descriptor)
                continue
            try:
                obj = parse_log_line(raw.decode("utf-8"))
            except UnicodeDecodeError:
                continue
            if obj is None:
                continue
            kept = filter_records([obj], since=since, level=level)
            if kept:
                output.append((json.dumps(kept[0], ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
                emitted += 1
        offset = fh.tell()
        next_cursor = f"v1.{file_stat.st_dev:x}.{file_stat.st_ino:x}.{offset:x}.{_fingerprint(fh, offset)}.{int(discarding)}"
    return b"".join(output), next_cursor, reset


@router.get("/api/v1/jobs/{job_id}/logs")
@limiter.limit(get_default_rate())
async def get_job_logs(
    request: Request,
    job_id: str,
    since: Optional[str] = Query(
        default=None,
        max_length=64,
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
        max_length=16,
        description="Minimum level (DEBUG|INFO|WARN|ERROR).",
    ),
    cursor: Optional[str] = Query(default=None, max_length=256, description="Opaque file continuation from X-Next-Cursor."),
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

    try:
        file_descriptor = _open_job_log_descriptor(job_id)
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail=_NOT_FOUND_BODY)

    body, next_cursor, reset = _read_records(
        file_descriptor, since=since_dt, level=level_norm, limit=limit, cursor=cursor
    )
    return Response(
        body,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Next-Cursor": next_cursor,
            "X-Log-Reset": "true" if reset else "false",
            "Access-Control-Expose-Headers": "X-Next-Cursor, X-Log-Reset",
        },
    )
