"""Per-job NDJSON log writer (D7).

Appends structured log records to ``OCR_OUTPUT_DIR/logs/jobs/{job_id}.jsonl``
so the management UI Logs tab can stream them via the per-job logs endpoint.

The writer is best-effort: every I/O error is swallowed so a logging failure
never bubbles up into the pipeline.  Files are opened in append mode with a
single ``write()`` per record so concurrent writers from different threads
do not corrupt records (POSIX append-mode writes < PIPE_BUF bytes are atomic
and JSONL records are typically far below that threshold).

Each record has the shape::

    {
        "ts": "2026-04-25T10:00:00.123456+00:00",
        "level": "INFO",
        "code": "JOB_STARTED",
        "job_id": "job_abcdef012345",
        "message": "...",
        "data": {...}
    }
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Allowed level strings (uppercase) -- ordering used by the read endpoint.
_LEVEL_ORDER = ("DEBUG", "INFO", "WARN", "WARNING", "ERROR")
_LEVEL_PRIORITY = {"DEBUG": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3}

# Per-job lock -- guards file open + write + flush so two threads writing for
# the same job_id can't interleave a partial line on Windows where append-mode
# atomicity is not guaranteed.
_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()
_write_counter: dict[str, int] = {}
_FSYNC_EVERY_N_WRITES = 25


def _output_dir() -> str:
    """Resolve the OCR output directory at write time (env may change in tests)."""
    return os.environ.get("OCR_OUTPUT_DIR", "").strip() or os.environ.get(
        "OUTPUT_FOLDER", ""
    ).strip() or "/app/ocr_output"


def job_log_path(job_id: str, *, base_dir: Optional[str] = None) -> Path:
    """Return the on-disk path for a job's NDJSON log file.

    Public so the read endpoint can resolve the same path.
    """
    root = Path(base_dir) if base_dir else Path(_output_dir())
    return root / "logs" / "jobs" / f"{job_id}.jsonl"


def _get_lock(job_id: str) -> threading.Lock:
    with _locks_lock:
        lock = _locks.get(job_id)
        if lock is None:
            lock = threading.Lock()
            _locks[job_id] = lock
        return lock


def _coerce_record(
    record: dict[str, Any],
    *,
    job_id: str,
) -> dict[str, Any]:
    """Normalise a record before serialisation.

    Adds defaults for ``ts``, ``level``, ``job_id`` if missing.
    """
    out = dict(record)
    out.setdefault("ts", datetime.now(timezone.utc).isoformat())
    level = str(out.get("level", "INFO")).upper()
    if level not in _LEVEL_PRIORITY:
        level = "INFO"
    out["level"] = level
    out.setdefault("job_id", job_id)
    out.setdefault("code", "EVENT")
    out.setdefault("message", "")
    if "data" in out and out["data"] is None:
        del out["data"]
    return out


def write_job_log(
    job_id: str,
    record: dict[str, Any],
    *,
    base_dir: Optional[str] = None,
) -> None:
    """Append ``record`` to the per-job NDJSON log.

    Best-effort: any error is logged at debug level and swallowed.
    """
    try:
        path = job_log_path(job_id, base_dir=base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        coerced = _coerce_record(record, job_id=job_id)
        line = json.dumps(coerced, ensure_ascii=False, separators=(",", ":")) + "\n"
        lock = _get_lock(job_id)
        with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
                # Periodic fsync so the read endpoint sees recent writes
                # promptly without paying the syscall cost on every event.
                count = _write_counter.get(job_id, 0) + 1
                _write_counter[job_id] = count
                if count % _FSYNC_EVERY_N_WRITES == 0:
                    try:
                        fh.flush()
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
    except Exception:  # pragma: no cover - defensive
        logger.debug("write_job_log failed for %s", job_id, exc_info=True)


def parse_log_line(raw: str) -> Optional[dict[str, Any]]:
    """Parse a single NDJSON log line, returning ``None`` on malformed input."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def filter_records(
    records: Iterable[dict[str, Any]],
    *,
    since: Optional[datetime] = None,
    level: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Filter parsed log records by ``since`` and minimum ``level``."""
    threshold = _LEVEL_PRIORITY.get((level or "").upper()) if level else None
    out: list[dict[str, Any]] = []
    for record in records:
        if since is not None:
            ts_raw = record.get("ts")
            if isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except ValueError:
                    ts = None
                if ts is not None:
                    if ts.tzinfo is None and since.tzinfo is not None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if since.tzinfo is None and ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                    if ts <= since:
                        continue
        if threshold is not None:
            rec_level = str(record.get("level", "INFO")).upper()
            if _LEVEL_PRIORITY.get(rec_level, 1) < threshold:
                continue
        out.append(record)
    return out


__all__ = [
    "write_job_log",
    "job_log_path",
    "parse_log_line",
    "filter_records",
]
