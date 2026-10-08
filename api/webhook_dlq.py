"""Webhook dead-letter queue (DLQ) for failed webhook deliveries.

When all retry attempts are exhausted for a webhook delivery, the
full payload and metadata are written to a JSONL file so they can
be inspected and retried later via the ``/api/v1/webhooks/dlq`` endpoints.

Thread-safe: uses a threading lock for file I/O.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # POSIX deployments use an OS lock to coordinate API processes.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


def _default_output_folder() -> str:
    """Return the default output root when api.config is unavailable."""
    return (
        os.environ.get("OUTPUT_FOLDER", "").strip()
        or os.environ.get("OCR_OUTPUT_DIR", "").strip()
        or "/app/ocr_output"
    )


_CONFIG_FALLBACK_LOGGED = False
_MISSING = object()


def _note_config_fallback(reason: str, attr: str) -> None:
    """Emit one log line the first time a DLQ config fallback is used."""
    global _CONFIG_FALLBACK_LOGGED
    if not _CONFIG_FALLBACK_LOGGED:
        _CONFIG_FALLBACK_LOGGED = True
        logger.warning(
            "Webhook DLQ using fallback defaults (%s for %r); "
            "api.config unavailable or attribute missing",
            reason,
            attr,
        )


def _config_value(attr: str, default):
    """Read a config value from api.config, with fallback.

    Only ``ImportError`` (config module genuinely absent, e.g. standalone
    use/tests) falls back to ``default``. Any other exception from the
    ``api.config`` import — notably the production fail-closed
    ``RuntimeError`` guard — propagates so misconfiguration is never
    masked by silent defaults. Missing attributes also fall back; the
    first fallback of either kind logs one warning line.
    """
    try:
        from api import config
    except ImportError:
        _note_config_fallback("import failed", attr)
        return default
    value = getattr(config, attr, _MISSING)
    if value is _MISSING:
        _note_config_fallback("attribute missing", attr)
        return default
    return value


# ---------------------------------------------------------------------------
# DLQ file operations
# ---------------------------------------------------------------------------

_dlq_lock = threading.Lock()
_DLQ_RETRY_LEASE_SECONDS = 300
_DLQ_MAX_FILE_BYTES = 256 * 1024 * 1024
_DLQ_MAX_LINE_BYTES = 16 * 1024 * 1024
_DLQ_DELETE = object()
_DLQ_ABORT = object()


@contextmanager
def _dlq_file_lock(path: Path):
    """Serialize DLQ operations across threads and API processes.

    The JSONL file is shared by multiple workers in a normal deployment.
    ``threading.Lock`` protects only one interpreter, so use a deterministic
    sidecar lock file and ``flock`` where available.  Opening the lock with
    ``O_NOFOLLOW`` prevents a symlinked lock path from redirecting a worker to
    an attacker-selected file.
    """
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    handle = None
    with _dlq_lock:
        try:
            flags = os.O_CREAT | os.O_RDWR
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(lock_path, flags | nofollow, 0o600)
            handle = os.fdopen(fd, "a+b")
            fd = None
            try:
                os.fchmod(handle.fileno(), 0o600)
            except OSError:
                pass
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if handle is not None:
                try:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
            if fd is not None:
                os.close(fd)


def _dlq_file_within_limit(path: Path) -> bool:
    """Fail closed before an operator endpoint reads an oversized DLQ."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        logger.warning("Unable to stat webhook DLQ %s: %s", path, exc)
        return False
    if size > _DLQ_MAX_FILE_BYTES:
        logger.error(
            "Refusing webhook DLQ operation for %s: file is %d bytes (limit %d)",
            path,
            size,
            _DLQ_MAX_FILE_BYTES,
        )
        return False
    return True


def _iter_dlq_entries(path: Path):
    """Yield bounded JSON objects without materializing the whole DLQ."""
    with path.open("rb") as handle:
        while True:
            raw = handle.readline(_DLQ_MAX_LINE_BYTES + 1)
            if not raw:
                return
            if len(raw) > _DLQ_MAX_LINE_BYTES:
                # Drain the remainder of an oversized JSONL record before
                # continuing, so a following valid entry remains visible.
                while raw and not raw.endswith(b"\n"):
                    raw = handle.readline(_DLQ_MAX_LINE_BYTES + 1)
                logger.warning(
                    "Skipping oversized webhook DLQ line (limit %d bytes)",
                    _DLQ_MAX_LINE_BYTES,
                )
                continue
            try:
                entry = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                logger.warning("Skipping malformed webhook DLQ line")
                continue
            if isinstance(entry, dict):
                yield entry


def _atomic_rewrite(path: Path, lines: list[str]) -> None:
    """Rewrite a DLQ file without a truncate-on-crash window."""
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
            handle.flush()
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _atomic_transform(path: Path, transform) -> bool:
    """Stream a bounded DLQ rewrite while preserving malformed records.

    The previous mutation paths used unbounded text iteration and built a list
    containing the whole file.  This helper caps both each raw line and total
    input bytes, writes to a same-directory temporary file, and only replaces
    the source after a successful complete transform.  Returning ``None``
    from *transform* preserves a valid record; ``_DLQ_DELETE`` removes it;
    ``_DLQ_ABORT`` leaves the original file untouched.
    """
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    changed = False
    aborted = False
    total_read = 0
    try:
        with path.open("rb") as source, temp.open("wb") as destination:
            while True:
                raw = source.readline(_DLQ_MAX_LINE_BYTES + 1)
                if not raw:
                    break
                total_read += len(raw)
                if len(raw) > _DLQ_MAX_LINE_BYTES or total_read > _DLQ_MAX_FILE_BYTES:
                    logger.error(
                        "Refusing webhook DLQ mutation: bounded input limit exceeded"
                    )
                    aborted = True
                    break

                if not raw.strip():
                    destination.write(raw)
                    continue
                try:
                    entry = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # Forensic preservation: malformed data is not silently
                    # discarded just because an adjacent record changed.
                    destination.write(raw)
                    continue
                if not isinstance(entry, dict):
                    destination.write(raw)
                    continue

                action = transform(entry)
                if action is _DLQ_ABORT:
                    aborted = True
                    break
                if action is _DLQ_DELETE:
                    changed = True
                    continue
                if action is None:
                    destination.write(raw)
                    continue

                serialized = (
                    json.dumps(
                        action, separators=(",", ":"), ensure_ascii=False
                    ).encode("utf-8")
                    + b"\n"
                )
                if len(serialized) > _DLQ_MAX_LINE_BYTES:
                    logger.error(
                        "Refusing webhook DLQ mutation: transformed line exceeds limit"
                    )
                    aborted = True
                    break
                destination.write(serialized)
                changed = True

            if not aborted and changed:
                destination.flush()
                try:
                    os.fsync(destination.fileno())
                except OSError:
                    pass

        if aborted or not changed:
            return False
        try:
            os.chmod(temp, 0o600)
        except OSError as exc:
            logger.error("Refusing webhook DLQ mutation: chmod failed: %s", exc)
            return False
        os.replace(temp, path)
        return True
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _dlq_path() -> Path:
    """Return the configured DLQ file path."""
    configured = _config_value(
        "WEBHOOK_DLQ_PATH",
        os.path.join(
            _default_output_folder(),
            "logs",
            "webhook_dlq.jsonl",
        ),
    )
    return Path(configured)


def _lookup_resource_tenant_id(resource_id: str) -> Optional[str]:
    """Resolve tenant ownership for a job_id or batch_id stored on a DLQ entry.

    Used for legacy unstamped entries only. Returns None when the resource
    is missing, has no tenant stamp, or the DB is unavailable.
    """
    if not resource_id:
        return None
    try:
        from api.database import Batch, Job, get_session_factory

        session = get_session_factory()()
        try:
            job = session.query(Job).filter(Job.job_id == resource_id).first()
            if job is not None:
                tid = getattr(job, "tenant_id", None)
                return tid if isinstance(tid, str) and tid else None

            batch = session.query(Batch).filter(Batch.batch_id == resource_id).first()
            if batch is not None:
                durable_tid = getattr(batch, "tenant_id", None)
                durable_tid = (
                    durable_tid.strip() if isinstance(durable_tid, str) else ""
                )
                try:
                    settings = batch.settings or {}
                except Exception:
                    settings = {}
                legacy_tid = (
                    settings.get("tenant_id") if isinstance(settings, dict) else None
                )
                legacy_tid = legacy_tid.strip() if isinstance(legacy_tid, str) else ""
                if durable_tid and legacy_tid and durable_tid != legacy_tid:
                    logger.error(
                        "Batch %s has conflicting tenant bindings; "
                        "DLQ lookup uses durable tenant",
                        resource_id,
                    )
                if durable_tid:
                    return durable_tid
                if isinstance(settings, dict):
                    return legacy_tid or None
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.warning(
            "DLQ tenant lookup failed for resource %s: %s",
            resource_id,
            exc,
        )
        return None


def resolve_entry_tenant_id(
    entry: dict[str, Any],
    *,
    lookup: Optional[Any] = None,
) -> Optional[str]:
    """Return the tenant that owns a DLQ entry.

    Prefer the stamped ``tenant_id`` field (new entries). For legacy
    unstamped rows, attempt job/batch resolution via ``lookup`` (or the
    default ORM helper). Returns ``None`` when ownership cannot be
    determined — those entries are admin-only under multi-tenant keys.
    """
    stamped = entry.get("tenant_id")
    if isinstance(stamped, str) and stamped:
        return stamped
    # Explicit null / missing stamp → legacy path
    resource_id = entry.get("job_id")
    if not isinstance(resource_id, str) or not resource_id:
        return None
    resolver = lookup if lookup is not None else _lookup_resource_tenant_id
    try:
        resolved = resolver(resource_id)
    except Exception as exc:
        logger.warning("DLQ legacy tenant resolve failed for %s: %s", resource_id, exc)
        return None
    return resolved if isinstance(resolved, str) and resolved else None


def entry_owned_by_tenant(
    entry: dict[str, Any],
    tenant_id: str,
    *,
    lookup: Optional[Any] = None,
) -> bool:
    """True when the entry is owned by ``tenant_id``.

    Unstamped unresolvable entries return False (fail-closed for tenant
    keys; platform admins / single-tenant callers skip this check).
    """
    if not tenant_id:
        return True
    resolved = resolve_entry_tenant_id(entry, lookup=lookup)
    if resolved is None:
        return False
    return resolved == tenant_id


def add_to_dlq(
    *,
    job_id: str,
    webhook_url: str,
    event_type: str,
    payload: dict[str, Any],
    last_error: str,
    attempts: int,
    tenant_id: Optional[str] = None,
    dlq_file: Optional[Path] = None,
) -> str:
    """Add a failed webhook delivery to the dead-letter queue.

    Parameters
    ----------
    job_id : str
        The job (or batch) ID the webhook was for.
    webhook_url : str
        The target URL that failed.
    event_type : str
        The event type (e.g. ``job.completed``).
    payload : dict
        The full webhook payload that was to be delivered.
    last_error : str
        The last error message from the delivery attempt.
    attempts : int
        Total number of delivery attempts made.
    tenant_id : str, optional
        Owning tenant for multi-tenant isolation (AEP BE003). Stamped on
        the JSONL entry when known; ``None`` for single-tenant / legacy.
    dlq_file : Path, optional
        Override for the DLQ file path (used in tests).

    Returns
    -------
    str
        The generated DLQ entry ID.
    """
    if not _config_value("WEBHOOK_DLQ_ENABLED", True):
        return ""

    entry_id = f"dlq_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()

    stamped_tenant = tenant_id if isinstance(tenant_id, str) and tenant_id else None

    entry = {
        "id": entry_id,
        "job_id": job_id,
        "webhook_url": webhook_url,
        "event_type": event_type,
        "payload": payload,
        "last_error": last_error,
        "attempts": attempts,
        "tenant_id": stamped_tenant,
        "created_at": now,
        "retried_at": None,
    }

    serialized = json.dumps(entry, separators=(",", ":"), ensure_ascii=False)
    line = (serialized + "\n").encode("utf-8")
    if len(line) > _DLQ_MAX_LINE_BYTES:
        raise ValueError("Webhook DLQ entry exceeds the configured line size limit")

    path = dlq_file or _dlq_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _dlq_file_lock(path):
        if path.is_symlink():
            raise ValueError("Webhook DLQ path must not be a symlink")
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            current_size = 0
        if current_size + len(line) > _DLQ_MAX_FILE_BYTES:
            raise ValueError("Webhook DLQ has reached the configured file size limit")
        with path.open("ab") as f:
            f.write(line)
            f.flush()
        try:
            os.chmod(path, 0o600)
        except OSError as exc:
            raise RuntimeError("Webhook DLQ permissions could not be hardened") from exc

    logger.info(
        "Added webhook DLQ entry %s for job %s (tenant=%s, url=%s, error=%s)",
        entry_id,
        job_id,
        stamped_tenant,
        webhook_url,
        last_error[:100],
    )

    return entry_id


def purge_dlq_entries_for_tenant(
    tenant_id: str,
    *,
    resource_ids: Optional[set[str]] = None,
    dlq_file: Optional[Path] = None,
) -> int:
    """Remove DLQ records owned by a tenant and atomically rewrite the file.

    New entries carry a tenant stamp.  ``resource_ids`` covers legacy entries
    created before that stamp existed; callers should provide the tenant's job
    and batch IDs while those records are still available in the database.
    """
    if not isinstance(tenant_id, str) or not tenant_id:
        return 0
    path = dlq_file or _dlq_path()
    if not path.exists():
        return 0
    if not _dlq_file_within_limit(path):
        raise ValueError("Webhook DLQ exceeds the configured purge size limit")

    legacy_ids = {str(value) for value in (resource_ids or set()) if value}
    removed = 0
    with _dlq_file_lock(path):

        def _remove_owned(entry):
            nonlocal removed
            stamped = entry.get("tenant_id")
            resource_id = entry.get("job_id")
            owned = stamped == tenant_id or (
                not stamped
                and isinstance(resource_id, str)
                and resource_id in legacy_ids
            )
            if owned:
                removed += 1
                return _DLQ_DELETE
            return None

        rewritten = _atomic_transform(path, _remove_owned)
    return removed if rewritten else 0


def list_dlq(
    *,
    limit: int = 100,
    tenant_id: Optional[str] = None,
    dlq_file: Optional[Path] = None,
    lookup: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Read DLQ entries from the JSONL file.

    Parameters
    ----------
    limit : int
        Maximum number of entries to return (most recent first).
    tenant_id : str, optional
        When set, only return entries owned by this tenant. Legacy
        unstamped entries are resolved via job/batch lookup; unresolvable
        rows are hidden (admin-only). When ``None``, return all entries
        (single-tenant mode or platform_admin global view).
    dlq_file : Path, optional
        Override for the DLQ file path (used in tests).
    lookup : callable, optional
        Optional ``(resource_id) -> tenant_id | None`` override for tests.

    Returns
    -------
    list of dict
        DLQ entries, most recent first.
    """
    path = dlq_file or _dlq_path()
    if not path.exists():
        return []
    if not _dlq_file_within_limit(path):
        return []

    try:
        limit = min(int(limit), 1000)
    except (TypeError, ValueError):
        return []
    if limit <= 0:
        return []

    # The file is append-only, so a bounded deque retains exactly the newest
    # candidates while preserving the existing reverse-chronological API.
    entries: deque[dict[str, Any]] = deque(maxlen=limit)
    with _dlq_file_lock(path):
        for entry in _iter_dlq_entries(path):
            if tenant_id is None or entry_owned_by_tenant(
                entry, tenant_id, lookup=lookup
            ):
                entries.append(entry)

    return list(reversed(entries))


def get_dlq_entry(
    entry_id: str,
    *,
    dlq_file: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Look up a single DLQ entry by ID.

    Parameters
    ----------
    entry_id : str
        The DLQ entry ID.
    dlq_file : Path, optional
        Override for the DLQ file path (used in tests).

    Returns
    -------
    dict or None
        The DLQ entry, or None if not found.
    """
    path = dlq_file or _dlq_path()
    if not path.exists():
        return None
    if not _dlq_file_within_limit(path):
        return None

    with _dlq_file_lock(path):
        for entry in _iter_dlq_entries(path):
            if entry.get("id") == entry_id:
                return entry

    return None


def mark_dlq_retried(
    entry_id: str,
    *,
    dlq_file: Optional[Path] = None,
    claim_token: Optional[str] = None,
) -> bool:
    """Mark a DLQ entry as retried by updating its ``retried_at`` field.

    The JSONL file is rewritten with the updated entry.

    Parameters
    ----------
    entry_id : str
        The DLQ entry ID.
    dlq_file : Path, optional
        Override for the DLQ file path (used in tests).

    Returns
    -------
    bool
        True if the entry was found and updated.
    """
    path = dlq_file or _dlq_path()
    if not path.exists():
        return False
    if not _dlq_file_within_limit(path):
        return False

    now = datetime.now(timezone.utc).isoformat()
    with _dlq_file_lock(path):

        def _mark(entry):
            if entry.get("id") != entry_id:
                return None
            if entry.get("retried_at"):
                return _DLQ_ABORT
            if claim_token is not None and entry.get("retry_token") != claim_token:
                return _DLQ_ABORT
            entry["retried_at"] = now
            entry.pop("retrying_at", None)
            entry.pop("retry_token", None)
            return entry

        return _atomic_transform(path, _mark)


def claim_dlq_retry(
    entry_id: str,
    *,
    dlq_file: Optional[Path] = None,
    lease_seconds: int = _DLQ_RETRY_LEASE_SECONDS,
) -> Optional[str]:
    """Atomically claim a DLQ entry for one redelivery attempt.

    The claim closes the check-then-deliver race between concurrent operators.
    A crashed claimant becomes retryable after the lease expires, preserving
    at-least-once delivery without allowing two live requests to send the same
    entry concurrently.
    """
    path = dlq_file or _dlq_path()
    if not path.exists():
        return None
    if not _dlq_file_within_limit(path):
        return None

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    token = uuid.uuid4().hex
    with _dlq_file_lock(path):

        def _claim(entry):
            if entry.get("id") != entry_id:
                return None
            if entry.get("retried_at"):
                return _DLQ_ABORT
            retrying_at = entry.get("retrying_at")
            if retrying_at:
                try:
                    claimed_at = datetime.fromisoformat(retrying_at)
                    if claimed_at.tzinfo is None:
                        claimed_at = claimed_at.replace(tzinfo=timezone.utc)
                    if (now - claimed_at).total_seconds() < lease_seconds:
                        return _DLQ_ABORT
                except (TypeError, ValueError):
                    # Malformed claim state fails closed until an operator
                    # repairs the entry explicitly.
                    return _DLQ_ABORT
            entry["retrying_at"] = now_iso
            entry["retry_token"] = token
            return entry

        return token if _atomic_transform(path, _claim) else None


def release_dlq_retry(
    entry_id: str,
    claim_token: str,
    *,
    dlq_file: Optional[Path] = None,
) -> bool:
    """Release a failed DLQ retry claim without changing delivery history."""
    path = dlq_file or _dlq_path()
    if not path.exists():
        return False
    if not _dlq_file_within_limit(path):
        return False

    with _dlq_file_lock(path):

        def _release(entry):
            if (
                entry.get("id") == entry_id
                and entry.get("retry_token") == claim_token
                and not entry.get("retried_at")
            ):
                entry.pop("retrying_at", None)
                entry.pop("retry_token", None)
                return entry
            return None

        return _atomic_transform(path, _release)
