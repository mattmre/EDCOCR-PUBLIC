"""Structured request-level API audit logging with tamper-evident hash chains.

Each audit entry is cryptographically linked to the previous entry via
SHA-256 hashing, following the same pattern as ``custody.py`` to create a
verifiable, forensic-grade API request audit trail.

Output: JSONL file (one JSON object per line) for append-only safety.
Default location: ``ocr_output/logs/api-audit.jsonl``
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import api.config as config

logger = logging.getLogger(__name__)

_AUDIT_LOCK = threading.Lock()
_AUDIT_MAX_LINE_BYTES = 64 * 1024
_AUDIT_MAX_REQUEST_ID_LENGTH = 128
_AUDIT_MAX_FIELD_LENGTH = 512

# Tracks the SHA-256 hash of the most recently written audit entry so the
# next entry can reference it via ``prev_hash``, forming a hash chain.
_prev_hash: str | None = None
_chain_path: str | None = None

# Regex for extracting job IDs from URL paths.
_JOB_ID_PATH_RE = re.compile(r"/api/v1/jobs/(job_[0-9a-f]{12})")

# Paths that are excluded from audit logging when health exclusion is on.
_HEALTH_PATHS = {
    "/api/v1/health",
    "/api/v1/health/detailed",
    "/api/v1/ready",
    "/api/v1/readiness",
    "/api/v1/translation/readiness",
}


def get_api_audit_log_path() -> str:
    """Return the configured audit log path or the default output path."""
    if config.API_AUDIT_LOG_PATH:
        return config.API_AUDIT_LOG_PATH
    return os.path.join(config.OUTPUT_FOLDER, "logs", "api-audit.jsonl")


def _ensure_request_id(request: Request) -> str:
    """Attach or reuse a stable request id for audit correlation."""
    request_id = getattr(request.state, "request_id", "")
    if request_id:
        return request_id

    request_id = request.headers.get("X-Request-ID", "").strip()
    if not request_id or len(request_id) > _AUDIT_MAX_REQUEST_ID_LENGTH:
        request_id = f"req_{uuid.uuid4().hex[:24]}"

    request.state.request_id = request_id
    return request_id


def _derive_auth_outcome(request: Request, status_code: int) -> str:
    """Resolve the best available auth outcome label for audit logs."""
    explicit_outcome = getattr(request.state, "audit_auth_outcome", "")
    if explicit_outcome:
        return explicit_outcome

    identity = getattr(request.state, "identity", None)
    if identity is not None:
        return "authorized"
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    return "unknown"


def _extract_job_id(path: str) -> str | None:
    """Extract a job ID from the request URL path, if present."""
    match = _JOB_ID_PATH_RE.search(path)
    return match.group(1) if match else None


def _bounded_text(value: Any, *, limit: int = _AUDIT_MAX_FIELD_LENGTH) -> str:
    """Keep attacker-controlled request metadata bounded in audit records."""
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 14] + "...[truncated]"


def mask_api_key(request: Request) -> str | None:
    """Return a masked representation of the API key used in the request.

    Shows the first 8 characters followed by ``***`` so operators can
    identify *which* key was used without exposing the full secret.
    Returns ``None`` when no API key header was provided.
    """
    raw_key = request.headers.get("X-API-Key", "")
    if not raw_key:
        return None
    if len(raw_key) <= 8:
        return raw_key[:4] + "***"
    return raw_key[:8] + "***"


def _compute_event_hash(event: dict[str, Any]) -> str:
    """Compute a SHA-256 hash over all event fields except ``hash`` itself."""
    hashable = {k: v for k, v in event.items() if k != "hash"}
    event_bytes = json.dumps(hashable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(event_bytes).hexdigest()


def _ensure_event_size(event: dict[str, Any]) -> None:
    """Reject caller-supplied audit metadata before hashing or writing it."""
    try:
        event_bytes = json.dumps(event, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("API audit event is not JSON serializable") from exc
    if len(event_bytes) > _AUDIT_MAX_LINE_BYTES:
        raise ValueError("API audit event exceeds the safety line limit")


def build_api_audit_event(
    request: Request,
    status_code: int,
    duration_ms: float,
) -> dict:
    """Build a safe structured audit record for one API request.

    The returned dict includes a ``prev_hash`` field linking to the previous
    audit entry and a ``hash`` field computed over all other fields.  These
    two fields form a tamper-evident hash chain compatible with the forensic
    chain-of-custody pattern in ``custody.py``.
    """
    identity = getattr(request.state, "identity", None)
    client_host = request.client.host if request.client else ""
    path = request.url.path

    duration = float(duration_ms)
    if not math.isfinite(duration) or abs(duration) > 1_000_000_000:
        raise ValueError("duration_ms is outside the supported range")

    event: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "request_id": _ensure_request_id(request),
        "method": request.method,
        "path": _bounded_text(path),
        "status_code": int(status_code),
        "duration_ms": round(duration, 3),
        "client_ip": _bounded_text(client_host),
        "user_agent": _bounded_text(request.headers.get("user-agent", "")),
        "query_present": bool(request.url.query),
        "api_key_masked": mask_api_key(request),
        "auth_method": _bounded_text(getattr(identity, "auth_method", "")),
        "auth_outcome": _bounded_text(_derive_auth_outcome(request, status_code)),
        "subject": _bounded_text(getattr(identity, "subject", "")),
        "role": _bounded_text(getattr(identity, "role", ""), limit=128),
        "tenant_id": _bounded_text(getattr(request.state, "tenant_id", None)) or None,
        "key_id": _bounded_text(getattr(request.state, "key_id", None)) or None,
        "job_id": _extract_job_id(path),
        "prev_hash": _prev_hash,
    }

    event_hash = _compute_event_hash(event)
    event["hash"] = event_hash

    return event


def _load_last_hash(audit_path: str) -> str | None:
    """Load the last committed hash from an existing audit log.

    A process restart must continue the existing chain rather than silently
    creating a second genesis entry.  Verify the existing log before extending
    it so a damaged chain is never treated as valid state.
    """
    if not os.path.exists(audit_path) or os.path.getsize(audit_path) == 0:
        return None

    valid, message = verify_audit_chain(audit_path)
    if not valid:
        raise RuntimeError(f"refusing to extend invalid audit chain: {message}")

    last_hash: str | None = None
    with open(audit_path, "rb") as handle:
        while True:
            raw_line = handle.readline(_AUDIT_MAX_LINE_BYTES + 1)
            if not raw_line:
                break
            if len(raw_line) > _AUDIT_MAX_LINE_BYTES:
                raise RuntimeError("audit chain entry exceeds the safety line limit")
            line = raw_line.strip()
            if line:
                try:
                    event = json.loads(line.decode("utf-8"))
                    last_hash = event["hash"]
                except (UnicodeDecodeError, TypeError, ValueError, KeyError) as exc:
                    raise RuntimeError("audit chain tail is not valid JSON") from exc
    return last_hash


def record_api_audit_event(event: dict) -> None:
    """Append a single audit record to the JSONL audit log."""
    if not config.API_AUDIT_LOG_ENABLED:
        return

    audit_path = get_api_audit_log_path()
    audit_dir = os.path.dirname(audit_path)
    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)

    global _prev_hash, _chain_path  # noqa: PLW0603

    with _AUDIT_LOCK:
        # Build and record may be separated by middleware scheduling, and
        # multiple requests may build concurrently.  The record lock is the
        # serialization point: bind the event to the current committed
        # predecessor and only advance the in-memory head after the write
        # succeeds.
        if _chain_path != audit_path:
            _prev_hash = _load_last_hash(audit_path)
            _chain_path = audit_path

        event["prev_hash"] = _prev_hash
        event.pop("hash", None)
        _ensure_event_size(event)
        event["hash"] = _compute_event_hash(event)
        line = json.dumps(event, ensure_ascii=False, default=str)
        if len(line.encode("utf-8")) > _AUDIT_MAX_LINE_BYTES:
            raise ValueError("API audit event exceeds the safety line limit")
        with open(audit_path, "a", encoding="utf-8") as handle:
            offset = handle.tell()
            try:
                handle.write(line + "\n")
                handle.flush()
            except Exception:
                # Preserve a valid JSONL boundary when a stream reports a
                # failure after a partial write.
                try:
                    handle.seek(offset)
                    handle.truncate()
                except Exception:
                    logger.exception("Failed to roll back partial audit write")
                raise

        _prev_hash = event["hash"]


def _should_skip_audit(path: str) -> bool:
    """Return True when the request path should be excluded from audit."""
    if not config.API_AUDIT_EXCLUDE_HEALTH:
        return False
    return path in _HEALTH_PATHS


def verify_audit_chain(filepath: str) -> tuple[bool, str]:
    """Verify the integrity of an API audit JSONL hash chain.

    Follows the same verification pattern as ``custody.verify_custody_file``
    to detect tampered or reordered entries.

    Args:
        filepath: Path to the ``.jsonl`` audit log file.

    Returns:
        ``(is_valid, message)`` tuple.
    """
    # Stream verification so a long-lived audit log cannot force the API to
    # materialize every historical event during restart or operator checks.
    previous_hash: str | None = None
    event_count = 0
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            while True:
                raw_line = handle.readline(_AUDIT_MAX_LINE_BYTES + 1)
                if not raw_line:
                    break
                if len(raw_line.encode("utf-8")) > _AUDIT_MAX_LINE_BYTES:
                    return False, (
                        f"Failed to load audit file: entry {event_count} "
                        "exceeds the safety line limit"
                    )
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    return (
                        False,
                        f"Failed to load audit file: entry {event_count} is not an object",
                    )
                if "prev_hash" not in event or "hash" not in event:
                    return False, (
                        f"Failed to load audit file: entry {event_count} "
                        "is missing chain fields"
                    )

                if event.get("prev_hash") != previous_hash:
                    return False, (
                        f"Broken chain at entry {event_count}: "
                        f"expected prev_hash={previous_hash}, got {event.get('prev_hash')}"
                    )

                hashable = {k: v for k, v in event.items() if k != "hash"}
                event_bytes = json.dumps(hashable, sort_keys=True, default=str).encode(
                    "utf-8"
                )
                computed = hashlib.sha256(event_bytes).hexdigest()

                if computed != event.get("hash"):
                    return False, (
                        f"Tampered entry {event_count}: computed hash {computed} != "
                        f"stored {event.get('hash')}"
                    )

                previous_hash = event.get("hash")
                event_count += 1
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        return False, f"Failed to load audit file: {exc}"

    if not event_count:
        return True, "Empty audit chain"
    return True, f"Audit chain verified: {event_count} entries"


def reset_chain_state() -> None:
    """Reset the in-memory hash chain state.

    Intended for test isolation only -- must not be called in production.
    """
    global _prev_hash, _chain_path  # noqa: PLW0603
    with _AUDIT_LOCK:
        _prev_hash = None
        _chain_path = None


class ApiAuditMiddleware(BaseHTTPMiddleware):
    """Outermost middleware that persists request-level audit metadata.

    Features:
    - SHA-256 hash-chained JSONL entries for tamper detection
    - Job ID extraction from URL path parameters
    - Masked API key logging (first 8 chars + ``***``)
    - Configurable health endpoint exclusion (``API_AUDIT_EXCLUDE_HEALTH``)
    - Thread-safe file writes
    - Never logs raw API keys or request bodies
    """

    async def dispatch(self, request: Request, call_next):
        if not config.API_AUDIT_LOG_ENABLED:
            return await call_next(request)

        if _should_skip_audit(request.url.path):
            return await call_next(request)

        request_id = _ensure_request_id(request)
        start = time.monotonic()
        status_code = 500
        response = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers.setdefault("X-Request-ID", request_id)
        finally:
            duration_ms = (time.monotonic() - start) * 1000.0
            try:
                record_api_audit_event(
                    build_api_audit_event(request, status_code, duration_ms)
                )
            except Exception:
                logger.exception("Failed to persist API audit event")

                # A successful response without its request-level forensic
                # record is not a truthful success.  Replace an uncommitted
                # response with a stable operator-facing failure; the
                # original exception, if any, still propagates normally.
                if response is not None:
                    response = JSONResponse(
                        status_code=503,
                        content={
                            "error": "audit_unavailable",
                            "message": "Request audit persistence is unavailable",
                            "request_id": request_id,
                        },
                        headers={"X-Request-ID": request_id},
                    )
        return response
