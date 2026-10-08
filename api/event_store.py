"""SQLite-backed durable event store for job lifecycle events.

Persists events so that WebSocket clients can replay missed events
on reconnect and webhook dead-letter entries can be inspected/retried.

Thread-safe: uses a dedicated SQLite connection with a threading lock.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


def _decode_event_payload(row: sqlite3.Row) -> dict[str, Any] | None:
    """Convert a row to an event, hiding corrupt payloads from replay APIs."""
    raw_payload = row["payload_json"]
    if not isinstance(raw_payload, str):
        logger.error("Skipping non-text durable event %s", row["id"])
        return None
    if len(raw_payload.encode("utf-8")) > MAX_EVENT_PAYLOAD_BYTES:
        logger.error("Skipping oversized durable event %s", row["id"])
        return None
    try:
        payload = json.loads(raw_payload)
    except (TypeError, json.JSONDecodeError) as exc:
        logger.error("Skipping corrupt durable event %s: %s", row["id"], exc)
        return None
    if not isinstance(payload, dict):
        logger.error("Skipping non-object durable event payload %s", row["id"])
        return None
    return {
        "id": row["id"],
        "event_type": row["event_type"],
        "job_id": row["job_id"],
        "payload": payload,
        "created_at": row["created_at"],
        "delivered_at": row["delivered_at"],
    }


def _default_output_folder() -> str:
    """Return the default output root when api.config is unavailable."""
    return (
        os.environ.get("OUTPUT_FOLDER", "").strip()
        or os.environ.get("OCR_OUTPUT_DIR", "").strip()
        or "/app/ocr_output"
    )


# ---------------------------------------------------------------------------
# Configuration defaults (overridden via api.config at runtime)
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = os.path.join(
    _default_output_folder(),
    "event_store.db",
)
_DEFAULT_RETENTION_HOURS = 72
_DEFAULT_DLQ_PATH = os.path.join(
    _default_output_folder(),
    "logs",
    "webhook_dlq.jsonl",
)

# Durable events can contain OCR-derived error text and paths.  Keep a single
# malformed/oversized event from turning the SQLite or JSONL sinks into an
# unbounded memory sink.
MAX_EVENT_PAYLOAD_BYTES = 1024 * 1024
MAX_EVENT_QUERY_LIMIT = 1000
MAX_UNDELIVERED_EVENTS = 10000


def _config_value(attr: str, default):
    """Read a config value from api.config, with fallback."""
    try:
        from api import config

        return getattr(config, attr, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Event Store
# ---------------------------------------------------------------------------


class EventStore:
    """SQLite-backed store for durable job events.

    Each event is immutable once stored. The ``delivered_at`` field
    tracks whether the event has been successfully broadcast to at
    least one consumer (WebSocket or webhook).
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _config_value(
            "EVENT_STORE_PATH",
            _DEFAULT_DB_PATH,
        )
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Return (or lazily create) the SQLite connection."""
        if self._conn is None:
            db_dir = os.path.dirname(self._db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create the events table if it does not exist."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    seq INTEGER
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_job_id ON events (job_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_created ON events (created_at)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_seq ON events (seq)")
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_event(
        self,
        event_type: str,
        job_id: str,
        payload: dict[str, Any],
    ) -> str:
        """Persist an event and return its unique ID.

        Parameters
        ----------
        event_type : str
            Event type (e.g. ``job.submitted``, ``job.completed``).
        job_id : str
            The job this event belongs to.
        payload : dict
            Full event payload (will be JSON-serialized).

        Returns
        -------
        str
            The generated event ID.
        """
        if not isinstance(event_type, str) or not event_type or len(event_type) > 128:
            raise ValueError(
                "event_type must be a non-empty string of at most 128 characters"
            )
        if not isinstance(job_id, str) or not job_id or len(job_id) > 256:
            raise ValueError(
                "job_id must be a non-empty string of at most 256 characters"
            )
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")

        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, separators=(",", ":"))
        if len(payload_json.encode("utf-8")) > MAX_EVENT_PAYLOAD_BYTES:
            raise ValueError(f"event payload exceeds {MAX_EVENT_PAYLOAD_BYTES} bytes")

        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO events (id, event_type, job_id, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, event_type, job_id, payload_json, now),
            )
            # Assign a monotonic sequence number for ordering
            seq = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("UPDATE events SET seq = ? WHERE id = ?", (seq, event_id))
            conn.commit()

        return event_id

    def get_events_since(
        self,
        job_id: str,
        since_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return events for a job, optionally after a given event ID.

        Parameters
        ----------
        job_id : str
            The job to retrieve events for.
        since_id : str, optional
            If provided, only events with a sequence number greater
            than this event's sequence number are returned.
        limit : int
            Maximum number of events to return.

        Returns
        -------
        list of dict
            Each dict contains ``id``, ``event_type``, ``job_id``,
            ``payload``, ``created_at``, and ``delivered_at``.
        """
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        limit = min(limit, MAX_EVENT_QUERY_LIMIT)

        with self._lock:
            conn = self._get_conn()

            if since_id:
                # Find the seq of the since_id event
                row = conn.execute(
                    "SELECT seq FROM events WHERE id = ? AND job_id = ?",
                    (since_id, job_id),
                ).fetchone()
                if row is None:
                    since_seq = 0
                else:
                    since_seq = row["seq"] or 0

                rows = conn.execute(
                    """
                    SELECT id, event_type, job_id, payload_json, created_at, delivered_at
                    FROM events
                    WHERE job_id = ? AND seq > ?
                    ORDER BY seq ASC
                    LIMIT ?
                    """,
                    (job_id, since_seq, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, event_type, job_id, payload_json, created_at, delivered_at
                    FROM events
                    WHERE job_id = ?
                    ORDER BY seq ASC
                    LIMIT ?
                    """,
                    (job_id, limit),
                ).fetchall()

        return [
            event for row in rows if (event := _decode_event_payload(row)) is not None
        ]

    def mark_delivered(self, event_id: str) -> bool:
        """Mark an event as delivered. Returns True if the event existed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "UPDATE events SET delivered_at = ? WHERE id = ? AND delivered_at IS NULL",
                (now, event_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_undelivered(self, max_age_hours: int = 24) -> list[dict[str, Any]]:
        """Return events that have not been delivered within the age window.

        Parameters
        ----------
        max_age_hours : int
            Only events created within this many hours are returned.

        Returns
        -------
        list of dict
            Undelivered events ordered by creation time.
        """
        if not isinstance(max_age_hours, int) or isinstance(max_age_hours, bool):
            raise ValueError("max_age_hours must be an integer")
        if not 0 <= max_age_hours <= 24 * 365:
            raise ValueError("max_age_hours is outside the supported range")

        cutoff = datetime.fromtimestamp(
            time.time() - max_age_hours * 3600,
            tz=timezone.utc,
        ).isoformat()

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT id, event_type, job_id, payload_json, created_at, delivered_at
                FROM events
                WHERE delivered_at IS NULL AND created_at >= ?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (cutoff, MAX_UNDELIVERED_EVENTS),
            ).fetchall()

        return [
            event for row in rows if (event := _decode_event_payload(row)) is not None
        ]

    def cleanup(self, older_than_hours: Optional[int] = None) -> int:
        """Purge events older than the given threshold.

        Parameters
        ----------
        older_than_hours : int, optional
            Events older than this many hours are deleted.
            Defaults to ``EVENT_RETENTION_HOURS`` (72).

        Returns
        -------
        int
            Number of events deleted.
        """
        if older_than_hours is None:
            older_than_hours = _config_value(
                "EVENT_RETENTION_HOURS",
                _DEFAULT_RETENTION_HOURS,
            )
        if not isinstance(older_than_hours, int) or isinstance(older_than_hours, bool):
            raise ValueError("older_than_hours must be an integer")
        if not 0 <= older_than_hours <= 24 * 365 * 10:
            raise ValueError("older_than_hours is outside the supported range")

        cutoff = datetime.fromtimestamp(
            time.time() - older_than_hours * 3600,
            tz=timezone.utc,
        ).isoformat()

        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM events WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount

    def delete_for_resources(self, resource_ids: Iterable[str]) -> int:
        """Delete durable events for the supplied job or batch IDs.

        The event table deliberately stores a generic ``job_id`` resource
        column because both job and batch lifecycle events use the same replay
        store.  Tenant purge therefore supplies both ID sets captured before
        the parent rows are deleted.  IDs are chunked to stay below SQLite's
        bind-variable limit.
        """
        ids = sorted({str(value) for value in resource_ids if value})
        if not ids:
            return 0

        deleted = 0
        with self._lock:
            conn = self._get_conn()
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                cursor = conn.execute(
                    f"DELETE FROM events WHERE job_id IN ({placeholders})",
                    chunk,
                )
                deleted += max(cursor.rowcount, 0)
            conn.commit()
        return deleted

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Optional[EventStore] = None
_store_lock = threading.Lock()


def get_event_store(db_path: Optional[str] = None) -> EventStore:
    """Return (or lazily create) the module-level EventStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = EventStore(db_path=db_path)
    return _store


def reset_event_store() -> None:
    """Close and discard the module-level EventStore (used in tests)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None
