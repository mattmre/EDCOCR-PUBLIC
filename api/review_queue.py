"""Human-review queue for low-confidence and exception documents.

Documents classified as 'review_required' or 'degraded' by the validation
module are routed here for human review. Reviewers can approve, reject,
or request reprocessing.

Storage: SQLite table alongside existing job database.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from api.config import DB_PATH
from api.db_security import set_db_permissions

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REPROCESS = "reprocess"


class ReviewReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    DEGRADED_QUALITY = "degraded_quality"
    HANDWRITING_DETECTED = "handwriting_detected"
    DPI_ESCALATION_FAILED = "dpi_escalation_failed"
    IMAGE_ONLY_PAGES = "image_only_pages"
    CLASSIFICATION_UNCERTAIN = "classification_uncertain"
    MANUAL_FLAG = "manual_flag"
    TRANSLATION_CERTIFICATION = "translation_certification"


@dataclass
class ReviewItem:
    """A single item in the human-review queue."""

    review_id: str
    job_id: str
    reason: str
    confidence: float = 0.0
    quality_classification: str = ""
    status: str = ReviewStatus.PENDING
    reviewer: str = ""
    decision_notes: str = ""
    created_at: str = ""
    reviewed_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON responses."""
        return {
            "review_id": self.review_id,
            "job_id": self.job_id,
            "reason": self.reason,
            "confidence": self.confidence,
            "quality_classification": self.quality_classification,
            "status": self.status,
            "reviewer": self.reviewer,
            "decision_notes": self.decision_notes,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ReviewItem":
        """Construct from a sqlite3.Row."""
        meta_raw = row["metadata_json"]
        meta = _decode_review_metadata(meta_raw)
        return cls(
            review_id=row["review_id"],
            job_id=row["job_id"],
            reason=row["reason"],
            confidence=row["confidence"],
            quality_classification=row["quality_classification"],
            status=row["status"],
            reviewer=row["reviewer"] or "",
            decision_notes=row["decision_notes"] or "",
            created_at=row["created_at"] or "",
            reviewed_at=row["reviewed_at"] or "",
            metadata=meta,
        )


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS review_items (
    review_id        TEXT PRIMARY KEY,
    job_id           TEXT NOT NULL,
    reason           TEXT NOT NULL,
    confidence       REAL NOT NULL DEFAULT 0.0,
    quality_classification TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'pending',
    reviewer         TEXT DEFAULT '',
    decision_notes   TEXT DEFAULT '',
    created_at       TEXT NOT NULL,
    reviewed_at      TEXT DEFAULT '',
    metadata_json    TEXT DEFAULT '{}'
);
"""

_CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_review_status ON review_items (status);",
    "CREATE INDEX IF NOT EXISTS idx_review_job_id ON review_items (job_id);",
    "CREATE INDEX IF NOT EXISTS idx_review_reason ON review_items (reason);",
    "CREATE INDEX IF NOT EXISTS idx_review_created ON review_items (created_at);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_review_translation_job_unique "
    "ON review_items (job_id, reason) "
    "WHERE reason = 'translation_certification';",
]

_ALLOWED_JOB_IDS_TABLE = "_ocr_allowed_job_ids"


def _set_allowed_job_ids(conn: sqlite3.Connection, job_ids: list[str]) -> None:
    """Load a tenant allowlist without exceeding SQLite bind-variable limits."""
    conn.execute(
        f"CREATE TEMP TABLE IF NOT EXISTS {_ALLOWED_JOB_IDS_TABLE} "
        "(job_id TEXT PRIMARY KEY)"
    )
    conn.execute(f"DELETE FROM {_ALLOWED_JOB_IDS_TABLE}")
    conn.executemany(
        f"INSERT OR IGNORE INTO {_ALLOWED_JOB_IDS_TABLE} (job_id) VALUES (?)",
        ((str(job_id),) for job_id in job_ids),
    )


MAX_REVIEW_METADATA_JSON_BYTES = 256 * 1024
MAX_REVIEW_TEXT_LENGTH = 4_096
MAX_REVIEW_PAGE_SIZE = 1_000


def _decode_review_metadata(raw: object) -> dict:
    """Decode DB metadata without allowing a tampered row to expand memory."""
    if not isinstance(raw, str) or not raw:
        return {}
    if len(raw.encode("utf-8")) > MAX_REVIEW_METADATA_JSON_BYTES:
        logger.error("Skipping oversized review metadata")
        return {}
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError, UnicodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _bounded_review_text(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_REVIEW_TEXT_LENGTH:
        raise ValueError(
            f"{field} must be a string of at most {MAX_REVIEW_TEXT_LENGTH} characters"
        )
    return value


class ReviewQueue:
    """SQLite-backed review queue.

    Uses threading.local() for per-thread connection management and an
    RLock to serialize all database operations for thread safety.

    SQLite WAL mode: enables concurrent reads during writes.
    See docs/architecture/adr-sqlite-to-postgresql-migration.md (maintenance review/maintenance review).
    """

    def __init__(self, db_path: str = ""):
        self._db_path = db_path or DB_PATH
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._lock = threading.RLock()
        self._initialized = False
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA wal_autocheckpoint=1000")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        """Create the review_items table if it does not exist."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn = self._get_conn()
            conn.execute(_CREATE_TABLE_SQL)
            for idx_sql in _CREATE_INDEX_SQL:
                conn.execute(idx_sql)
            conn.commit()
            # maintenance review: restrict DB file to owner read/write only. Must run
            # after the first connect so the file exists on disk.
            set_db_permissions(self._db_path)
            self._initialized = True

    def add(
        self,
        job_id: str,
        reason: str,
        confidence: float = 0.0,
        quality_classification: str = "",
        metadata: Optional[dict] = None,
    ) -> ReviewItem:
        """Add a document to the review queue.

        Args:
            job_id: The OCR job identifier.
            reason: Why the item needs review (should match ReviewReason values).
            confidence: Overall OCR confidence score.
            quality_classification: Quality classification from validation module.
            metadata: Optional extra metadata (page counts, methods, etc.).

        Returns:
            The newly created ReviewItem.
        """
        job_id = _bounded_review_text(job_id, "job_id")
        reason = _bounded_review_text(reason, "reason")
        quality_classification = _bounded_review_text(
            quality_classification, "quality_classification"
        )
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError("confidence must be numeric")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        metadata = metadata or {}
        review_id = f"rev_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        )
        meta_json = json.dumps(metadata)
        if len(meta_json.encode("utf-8")) > MAX_REVIEW_METADATA_JSON_BYTES:
            raise ValueError("review metadata exceeds the configured 256 KiB limit")

        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO review_items
                   (review_id, job_id, reason, confidence, quality_classification,
                    status, reviewer, decision_notes, created_at, reviewed_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    review_id,
                    job_id,
                    reason,
                    confidence,
                    quality_classification,
                    ReviewStatus.PENDING,
                    "",
                    "",
                    now,
                    "",
                    meta_json,
                ),
            )
            conn.commit()

        return ReviewItem(
            review_id=review_id,
            job_id=job_id,
            reason=reason,
            confidence=confidence,
            quality_classification=quality_classification,
            status=ReviewStatus.PENDING,
            reviewer="",
            decision_notes="",
            created_at=now,
            reviewed_at="",
            metadata=metadata,
        )

    def add_unique(
        self,
        job_id: str,
        reason: str,
        *,
        metadata: Optional[dict] = None,
    ) -> ReviewItem:
        """Return the existing job/reason item or create it atomically.

        This is reserved for durable workflow handles whose schema has a
        matching unique index (currently translation certification).
        """
        if reason != ReviewReason.TRANSLATION_CERTIFICATION.value:
            raise ValueError("add_unique is only supported for durable review reasons")
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM review_items WHERE job_id = ? AND reason = ?",
                (job_id, reason),
            ).fetchone()
            if row is not None:
                item = ReviewItem.from_row(row)
                requested_metadata = metadata or {}
                if any(
                    item.metadata.get(key) != value
                    for key, value in requested_metadata.items()
                ):
                    raise ValueError(
                        "durable review job/reason binding metadata changed"
                    )
                return item
            try:
                return self.add(job_id=job_id, reason=reason, metadata=metadata)
            except sqlite3.IntegrityError:
                # A different API process may win between SELECT and INSERT.
                # The database unique index is the cross-process authority.
                # Clear SQLite's failed transaction before rereading the
                # winner; otherwise this thread-local connection retains a
                # stale snapshot and can pin the WAL indefinitely.
                conn.rollback()
                row = conn.execute(
                    "SELECT * FROM review_items WHERE job_id = ? AND reason = ?",
                    (job_id, reason),
                ).fetchone()
                if row is None:
                    raise
                item = ReviewItem.from_row(row)
                requested_metadata = metadata or {}
                if any(
                    item.metadata.get(key) != value
                    for key, value in requested_metadata.items()
                ):
                    raise ValueError(
                        "durable review job/reason binding metadata changed"
                    )
                return item

    def get(self, review_id: str) -> Optional[ReviewItem]:
        """Get a review item by ID.

        Returns:
            The ReviewItem, or None if not found.
        """
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM review_items WHERE review_id = ?", (review_id,)
            ).fetchone()
        if row is None:
            return None
        return ReviewItem.from_row(row)

    def list_pending_translation_promotions(
        self, *, limit: int = 100
    ) -> list[ReviewItem]:
        """Return a bounded set of certified reviews not confirmed promoted."""
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 1000
        ):
            raise ValueError("limit must be between 1 and 1000")
        with self._lock:
            rows = (
                self._get_conn()
                .execute(
                    """SELECT * FROM review_items
                   WHERE reason = ? AND status = ?
                     AND json_valid(metadata_json)
                     AND json_extract(metadata_json, '$.certified') = 1
                     AND COALESCE(
                         json_extract(metadata_json, '$.promotion_confirmed'), 0
                     ) = 0
                     AND COALESCE(
                         json_extract(metadata_json, '$.promotion_abandoned'), 0
                     ) = 0
                   ORDER BY
                     COALESCE(
                       json_extract(metadata_json, '$.promotion_last_attempt_at'),
                       ''
                     ) ASC,
                     reviewed_at ASC
                   LIMIT ?""",
                    (
                        ReviewReason.TRANSLATION_CERTIFICATION,
                        ReviewStatus.APPROVED,
                        limit,
                    ),
                )
                .fetchall()
            )
        return [ReviewItem.from_row(row) for row in rows]

    def confirm_translation_promotion(
        self, review_id: str, custody_event_id: str
    ) -> bool:
        """CAS-mark one certified review after coordinator promotion succeeds."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM review_items WHERE review_id = ?", (review_id,)
            ).fetchone()
            if row is None:
                return False
            item = ReviewItem.from_row(row)
            certification = (item.metadata or {}).get("certification") or {}
            if (
                item.reason != ReviewReason.TRANSLATION_CERTIFICATION
                or item.status != ReviewStatus.APPROVED
                or not bool((item.metadata or {}).get("certified"))
                or str(certification.get("custody_event_id") or "")
                != str(custody_event_id)
            ):
                return False
            if bool((item.metadata or {}).get("promotion_confirmed")):
                return True
            metadata = dict(item.metadata)
            metadata["promotion_confirmed"] = True
            updated = json.dumps(metadata)
            cur = conn.execute(
                """UPDATE review_items SET metadata_json = ?
                   WHERE review_id = ? AND metadata_json = ?""",
                (updated, review_id, row["metadata_json"] or "{}"),
            )
            conn.commit()
            return cur.rowcount == 1

    def abandon_translation_promotion(self, review_id: str, reason: str) -> bool:
        """Quarantine an irrecoverable promotion so newer work cannot starve."""
        reason = _bounded_review_text(reason, "reason")
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM review_items WHERE review_id = ?", (review_id,)
            ).fetchone()
            if row is None:
                return False
            item = ReviewItem.from_row(row)
            if (
                item.reason != ReviewReason.TRANSLATION_CERTIFICATION
                or item.status != ReviewStatus.APPROVED
                or not bool((item.metadata or {}).get("certified"))
            ):
                return False
            metadata = dict(item.metadata)
            if bool(metadata.get("promotion_confirmed")):
                return False
            metadata["promotion_abandoned"] = True
            metadata["promotion_abandon_reason"] = reason
            updated = json.dumps(metadata)
            cur = conn.execute(
                """UPDATE review_items SET metadata_json = ?
                   WHERE review_id = ? AND metadata_json = ?""",
                (updated, review_id, row["metadata_json"] or "{}"),
            )
            conn.commit()
            return cur.rowcount == 1

    def defer_translation_promotion(self, review_id: str, reason: str) -> bool:
        """Durably rotate a retryable promotion behind unattempted work."""
        # Exception text comes from coordinator/database drivers and is not a
        # trusted bounded API field. Preserve a useful prefix without letting
        # an oversized driver message abort the entire recovery sweep.
        reason = _bounded_review_text(str(reason)[:MAX_REVIEW_TEXT_LENGTH], "reason")
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM review_items WHERE review_id = ?", (review_id,)
            ).fetchone()
            if row is None:
                return False
            item = ReviewItem.from_row(row)
            metadata = dict(item.metadata or {})
            if (
                item.reason != ReviewReason.TRANSLATION_CERTIFICATION
                or item.status != ReviewStatus.APPROVED
                or not bool(metadata.get("certified"))
                or bool(metadata.get("promotion_confirmed"))
                or bool(metadata.get("promotion_abandoned"))
            ):
                return False
            try:
                attempts = int(metadata.get("promotion_attempt_count") or 0)
            except (TypeError, ValueError):
                attempts = 0
            metadata["promotion_attempt_count"] = max(attempts, 0) + 1
            metadata["promotion_last_attempt_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="milliseconds")
            metadata["promotion_last_error"] = reason
            updated = json.dumps(metadata)
            cur = conn.execute(
                """UPDATE review_items SET metadata_json = ?
                   WHERE review_id = ? AND metadata_json = ?""",
                (updated, review_id, row["metadata_json"] or "{}"),
            )
            conn.commit()
            return cur.rowcount == 1

    def list_pending(
        self,
        limit: int = 50,
        offset: int = 0,
        reason: Optional[str] = None,
    ) -> list[ReviewItem]:
        """List pending review items with optional filtering.

        Args:
            limit: Maximum number of items to return.
            offset: Number of items to skip.
            reason: Optional reason filter.

        Returns:
            List of ReviewItem objects with status='pending'.
        """
        return self.list_all(
            limit=limit, offset=offset, status=ReviewStatus.PENDING, reason=reason
        )

    def decide(
        self,
        review_id: str,
        status: str,
        reviewer: str = "",
        notes: str = "",
    ) -> Optional[ReviewItem]:
        """Record a review decision.

        Args:
            review_id: The review item identifier.
            status: Decision status (approved, rejected, reprocess).
            reviewer: Name or identifier of the reviewer.
            notes: Optional decision notes.

        Returns:
            The updated ReviewItem, or None if not found.

        Raises:
            ValueError: If the item is not in 'pending' status or the
                        decision status is invalid.
        """
        reviewer = _bounded_review_text(reviewer, "reviewer")
        notes = _bounded_review_text(notes, "notes")
        valid_decisions = {
            ReviewStatus.APPROVED,
            ReviewStatus.REJECTED,
            ReviewStatus.REPROCESS,
        }
        if status not in valid_decisions:
            raise ValueError(
                f"Invalid decision status: {status}. "
                f"Must be one of: {', '.join(sorted(valid_decisions))}"
            )

        with self._lock:
            item = self.get(review_id)
            if item is None:
                return None

            if item.status != ReviewStatus.PENDING:
                raise ValueError(
                    f"Review item {review_id} is already '{item.status}', "
                    f"cannot change to '{status}'."
                )

            now = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="milliseconds"
            )

            conn = self._get_conn()
            # R2-RC-04: CAS on status=pending so concurrent decides cannot both win.
            cur = conn.execute(
                """UPDATE review_items
                   SET status = ?, reviewer = ?, decision_notes = ?, reviewed_at = ?
                   WHERE review_id = ? AND status = ?""",
                (status, reviewer, notes, now, review_id, ReviewStatus.PENDING),
            )
            conn.commit()
            if cur.rowcount != 1:
                raise ValueError(
                    f"Review item {review_id} is already decided "
                    "(CAS lost concurrent race)."
                )

        item.status = status
        item.reviewer = reviewer
        item.decision_notes = notes
        item.reviewed_at = now
        return item

    def certify(
        self,
        review_id: str,
        reviewer: str,
        notes: str,
        auth_method: str,
        *,
        require_custody_event: bool = True,
        custody_event_id: Optional[str] = None,
        refuse_synthetic_custody: bool = True,
    ) -> Optional[ReviewItem]:
        """Approve and mark a pending or already-approved item as certified.

        RC-08: certification metadata records custody linkage. Callers that
        cannot supply a custody event must fail closed unless explicitly
        waiving forensic mode. Synthetic ``waiver:*`` ids are refused when
        ``refuse_synthetic_custody`` is True (default).
        """
        reviewer = _bounded_review_text(reviewer, "reviewer")
        notes = _bounded_review_text(notes, "notes")
        auth_method = _bounded_review_text(auth_method, "auth_method")
        if custody_event_id is not None:
            custody_event_id = _bounded_review_text(
                custody_event_id, "custody_event_id"
            )
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM review_items WHERE review_id = ?", (review_id,)
            ).fetchone()
            if row is None:
                return None
            item = ReviewItem.from_row(row)
            original_metadata_json = row["metadata_json"] or "{}"

            already_certified = bool((item.metadata or {}).get("certified"))
            if item.status not in {ReviewStatus.PENDING, ReviewStatus.APPROVED} or (
                item.status == ReviewStatus.APPROVED and already_certified
            ):
                raise ValueError(
                    f"Review item {review_id} is already '{item.status}', "
                    "cannot certify."
                )

            if require_custody_event and not custody_event_id:
                raise ValueError(
                    "certify requires custody_event_id (RC-08). "
                    "Emit a custody certification event before certifying."
                )
            if refuse_synthetic_custody and custody_event_id:
                cid = str(custody_event_id).strip().lower()
                if cid.startswith("waiver:") or cid.startswith("fake:"):
                    raise ValueError(
                        "synthetic custody_event_id refused (RC-08); "
                        "emit a real custody event first"
                    )

            now = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="milliseconds"
            )
            metadata = dict(item.metadata)
            metadata["certified"] = True
            metadata["certification"] = {
                "auth_method": auth_method,
                "reviewer": reviewer,
                "certified_at": now,
                "custody_event_id": custody_event_id,
            }
            metadata_json = json.dumps(metadata)
            if len(metadata_json.encode("utf-8")) > MAX_REVIEW_METADATA_JSON_BYTES:
                raise ValueError("review metadata exceeds the configured 256 KiB limit")

            cur = conn.execute(
                """UPDATE review_items
                   SET status = ?, reviewer = ?, decision_notes = ?, reviewed_at = ?,
                       metadata_json = ?
                   WHERE review_id = ? AND status = ? AND reviewer = ?
                     AND decision_notes = ? AND reviewed_at = ?
                     AND metadata_json = ?""",
                (
                    ReviewStatus.APPROVED,
                    reviewer,
                    notes,
                    now,
                    metadata_json,
                    review_id,
                    item.status,
                    item.reviewer,
                    item.decision_notes,
                    item.reviewed_at,
                    original_metadata_json,
                ),
            )
            conn.commit()
            if cur.rowcount != 1:
                raise ValueError(
                    f"Review item {review_id} is already decided "
                    "(CAS lost concurrent race)."
                )

        item.status = ReviewStatus.APPROVED
        item.reviewer = reviewer
        item.decision_notes = notes
        item.reviewed_at = now
        item.metadata = metadata
        return item

    def stats(self, allowed_job_ids: Optional[list[str]] = None) -> dict:
        """Return review queue statistics.

        Args:
            allowed_job_ids: When provided, restrict aggregates to these job IDs
                only (multi-tenant isolation). ``None`` = global (single-tenant /
                platform-admin). Empty list = zero stats (fail-closed).

        Returns:
            Dictionary with pending, approved, rejected, reprocess counts,
            total count, average review time, and oldest pending timestamp.
        """
        if allowed_job_ids is not None and not allowed_job_ids:
            return {
                "pending": 0,
                "approved": 0,
                "rejected": 0,
                "reprocess": 0,
                "total": 0,
                "avg_review_seconds": 0.0,
                "oldest_pending": "",
            }

        job_filter_sql = ""
        job_filter_params: list = []
        if allowed_job_ids is not None:
            job_filter_sql = (
                f" AND job_id IN (SELECT job_id FROM {_ALLOWED_JOB_IDS_TABLE})"
            )

        with self._lock:
            conn = self._get_conn()
            if allowed_job_ids is not None:
                _set_allowed_job_ids(conn, allowed_job_ids)

            # Count by status
            count_where = f"WHERE 1=1{job_filter_sql}" if job_filter_sql else ""
            rows = conn.execute(
                f"SELECT status, COUNT(*) as cnt FROM review_items {count_where} GROUP BY status",
                job_filter_params,
            ).fetchall()
            counts = {row["status"]: row["cnt"] for row in rows}

            pending = counts.get(ReviewStatus.PENDING, 0)
            approved = counts.get(ReviewStatus.APPROVED, 0)
            rejected = counts.get(ReviewStatus.REJECTED, 0)
            reprocess = counts.get(ReviewStatus.REPROCESS, 0)
            total = pending + approved + rejected + reprocess

            # Oldest pending
            oldest_row = conn.execute(
                "SELECT MIN(created_at) as oldest FROM review_items "
                f"WHERE status = ?{job_filter_sql}",
                [ReviewStatus.PENDING, *job_filter_params],
            ).fetchone()
            oldest_pending = (
                oldest_row["oldest"] if oldest_row and oldest_row["oldest"] else ""
            )

            # Average review time (for decided items with both timestamps)
            avg_row = conn.execute(
                """SELECT AVG(
                       (julianday(reviewed_at) - julianday(created_at)) * 86400
                   ) as avg_seconds
                   FROM review_items
                   WHERE status != ? AND reviewed_at != ''"""
                + job_filter_sql,
                [ReviewStatus.PENDING, *job_filter_params],
            ).fetchone()
            avg_review_seconds = (
                round(avg_row["avg_seconds"], 1)
                if avg_row and avg_row["avg_seconds"] is not None
                else 0.0
            )

        return {
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "reprocess": reprocess,
            "total": total,
            "avg_review_seconds": avg_review_seconds,
            "oldest_pending": oldest_pending,
        }

    def list_all(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        allowed_job_ids: Optional[list[str]] = None,
    ) -> list[ReviewItem]:
        """List review items with optional filtering.

        Args:
            limit: Maximum number of items to return.
            offset: Number of items to skip.
            status: Optional status filter.
            reason: Optional reason filter.
            allowed_job_ids: When provided, restrict results to these job IDs
                only (used for multi-tenant isolation).

        Returns:
            List of ReviewItem objects ordered by created_at descending.
        """
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_REVIEW_PAGE_SIZE
        ):
            raise ValueError(f"limit must be between 1 and {MAX_REVIEW_PAGE_SIZE}")
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or not 0 <= offset <= 10_000_000
        ):
            raise ValueError("offset is outside the supported range")
        if allowed_job_ids is not None and not allowed_job_ids:
            return []

        clauses: list[str] = []
        params: list = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if reason is not None:
            clauses.append("reason = ?")
            params.append(reason)
        if allowed_job_ids is not None:
            clauses.append(f"job_id IN (SELECT job_id FROM {_ALLOWED_JOB_IDS_TABLE})")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = (
            f"SELECT * FROM review_items{where} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        with self._lock:
            conn = self._get_conn()
            if allowed_job_ids is not None:
                _set_allowed_job_ids(conn, allowed_job_ids)
            rows = conn.execute(query, params).fetchall()
        return [ReviewItem.from_row(r) for r in rows]

    def count(
        self,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        allowed_job_ids: Optional[list[str]] = None,
    ) -> int:
        """Count review items matching the given filters.

        Args:
            status: Optional status filter.
            reason: Optional reason filter.
            allowed_job_ids: When provided, restrict count to these job IDs
                only (used for multi-tenant isolation).

        Returns:
            Total count of matching items.
        """
        if allowed_job_ids is not None and not allowed_job_ids:
            return 0

        clauses: list[str] = []
        params: list = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if reason is not None:
            clauses.append("reason = ?")
            params.append(reason)
        if allowed_job_ids is not None:
            clauses.append(f"job_id IN (SELECT job_id FROM {_ALLOWED_JOB_IDS_TABLE})")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"SELECT COUNT(*) as cnt FROM review_items{where}"

        with self._lock:
            conn = self._get_conn()
            if allowed_job_ids is not None:
                _set_allowed_job_ids(conn, allowed_job_ids)
            row = conn.execute(query, params).fetchone()
        return row["cnt"] if row else 0

    def remove_jobs(self, job_ids: Iterable[str]) -> int:
        """Remove review records for many jobs in one bounded transaction."""
        ids = sorted({str(value) for value in job_ids if value})
        if not ids:
            return 0

        with self._lock:
            conn = self._get_conn()
            _set_allowed_job_ids(conn, ids)
            cursor = conn.execute(
                "DELETE FROM review_items WHERE job_id IN "
                f"(SELECT job_id FROM {_ALLOWED_JOB_IDS_TABLE})"
            )
            conn.commit()
            return max(cursor.rowcount, 0)

    def remove_tenant_metadata(self, tenant_id: str) -> int:
        """Remove review rows whose metadata tenant_id matches *tenant_id*."""
        tid = str(tenant_id or "").strip()
        if not tid:
            return 0
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM review_items "
                "WHERE json_extract(metadata_json, '$.tenant_id') = ?",
                (tid,),
            )
            conn.commit()
            return max(cursor.rowcount, 0)

    def close(self) -> None:
        """Close this instance's current-thread SQLite connection."""
        with self._lock:
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                conn.close()
                self._local.conn = None
