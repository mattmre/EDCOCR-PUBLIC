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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _default_output_folder() -> str:
    """Return the default output root when api.config is unavailable."""
    return (
        os.environ.get("OUTPUT_FOLDER", "").strip()
        or os.environ.get("OCR_OUTPUT_DIR", "").strip()
        or "/app/ocr_output"
    )


def _config_value(attr: str, default):
    """Read a config value from api.config, with fallback."""
    try:
        from api import config

        return getattr(config, attr, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# DLQ file operations
# ---------------------------------------------------------------------------

_dlq_lock = threading.Lock()


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


def add_to_dlq(
    *,
    job_id: str,
    webhook_url: str,
    event_type: str,
    payload: dict[str, Any],
    last_error: str,
    attempts: int,
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

    entry = {
        "id": entry_id,
        "job_id": job_id,
        "webhook_url": webhook_url,
        "event_type": event_type,
        "payload": payload,
        "last_error": last_error,
        "attempts": attempts,
        "created_at": now,
        "retried_at": None,
    }

    path = dlq_file or _dlq_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _dlq_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    logger.info(
        "Added webhook DLQ entry %s for job %s (url=%s, error=%s)",
        entry_id,
        job_id,
        webhook_url,
        last_error[:100],
    )

    return entry_id


def list_dlq(
    *,
    limit: int = 100,
    dlq_file: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Read all DLQ entries from the JSONL file.

    Parameters
    ----------
    limit : int
        Maximum number of entries to return (most recent first).
    dlq_file : Path, optional
        Override for the DLQ file path (used in tests).

    Returns
    -------
    list of dict
        DLQ entries, most recent first.
    """
    path = dlq_file or _dlq_path()
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with _dlq_lock:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed DLQ line: %s", line[:80])

    # Most recent first, limited
    entries.reverse()
    return entries[:limit]


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

    with _dlq_lock:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("id") == entry_id:
                        return entry
                except json.JSONDecodeError:
                    continue

    return None


def mark_dlq_retried(
    entry_id: str,
    *,
    dlq_file: Optional[Path] = None,
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

    now = datetime.now(timezone.utc).isoformat()
    found = False

    with _dlq_lock:
        lines: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                    if entry.get("id") == entry_id:
                        entry["retried_at"] = now
                        found = True
                    lines.append(json.dumps(entry, separators=(",", ":")))
                except json.JSONDecodeError:
                    lines.append(stripped)

        if found:
            with path.open("w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")

    return found
