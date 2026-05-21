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
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_seq ON events (seq)"
            )
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
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, separators=(",", ":"))

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
        with self._lock:
            conn = self._get_conn()

            if since_id:
                # Find the seq of the since_id event
                row = conn.execute(
                    "SELECT seq FROM events WHERE id = ?", (since_id,)
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
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "job_id": r["job_id"],
                "payload": json.loads(r["payload_json"]),
                "created_at": r["created_at"],
                "delivered_at": r["delivered_at"],
            }
            for r in rows
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
                """,
                (cutoff,),
            ).fetchall()

        return [
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "job_id": r["job_id"],
                "payload": json.loads(r["payload_json"]),
                "created_at": r["created_at"],
                "delivered_at": r["delivered_at"],
            }
            for r in rows
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
