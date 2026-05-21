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
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

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
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
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
]


class ReviewQueue:
    """SQLite-backed review queue.

    Uses threading.local() for per-thread connection management and an
    RLock to serialize all database operations for thread safety.

    SQLite WAL mode: enables concurrent reads during writes.
    See docs/architecture/adr-sqlite-to-postgresql-migration.md (/).
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
            # restrict DB file to owner read/write only. Must run
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
        review_id = f"rev_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        )
        meta_json = json.dumps(metadata or {})

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
            metadata=metadata or {},
        )

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
            conn.execute(
                """UPDATE review_items
                   SET status = ?, reviewer = ?, decision_notes = ?, reviewed_at = ?
                   WHERE review_id = ?""",
                (status, reviewer, notes, now, review_id),
            )
            conn.commit()

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
    ) -> Optional[ReviewItem]:
        """Approve and mark a pending review item as certified.

        Certification is deliberately tied to an explicit review approval. The
        auth token itself must not be stored; callers pass only the selected
        method and non-secret notes/fingerprint.
        """
        with self._lock:
            item = self.get(review_id)
            if item is None:
                return None

            if item.status != ReviewStatus.PENDING:
                raise ValueError(
                    f"Review item {review_id} is already '{item.status}', "
                    "cannot certify."
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
            }

            conn = self._get_conn()
            conn.execute(
                """UPDATE review_items
                   SET status = ?, reviewer = ?, decision_notes = ?, reviewed_at = ?,
                       metadata_json = ?
                   WHERE review_id = ?""",
                (
                    ReviewStatus.APPROVED,
                    reviewer,
                    notes,
                    now,
                    json.dumps(metadata),
                    review_id,
                ),
            )
            conn.commit()

        item.status = ReviewStatus.APPROVED
        item.reviewer = reviewer
        item.decision_notes = notes
        item.reviewed_at = now
        item.metadata = metadata
        return item

    def stats(self) -> dict:
        """Return review queue statistics.

        Returns:
            Dictionary with pending, approved, rejected, reprocess counts,
            total count, average review time, and oldest pending timestamp.
        """
        with self._lock:
            conn = self._get_conn()

            # Count by status
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM review_items GROUP BY status"
            ).fetchall()
            counts = {row["status"]: row["cnt"] for row in rows}

            pending = counts.get(ReviewStatus.PENDING, 0)
            approved = counts.get(ReviewStatus.APPROVED, 0)
            rejected = counts.get(ReviewStatus.REJECTED, 0)
            reprocess = counts.get(ReviewStatus.REPROCESS, 0)
            total = pending + approved + rejected + reprocess

            # Oldest pending
            oldest_row = conn.execute(
                "SELECT MIN(created_at) as oldest FROM review_items WHERE status = ?",
                (ReviewStatus.PENDING,),
            ).fetchone()
            oldest_pending = oldest_row["oldest"] if oldest_row and oldest_row["oldest"] else ""

            # Average review time (for decided items with both timestamps)
            avg_row = conn.execute(
                """SELECT AVG(
                       (julianday(reviewed_at) - julianday(created_at)) * 86400
                   ) as avg_seconds
                   FROM review_items
                   WHERE status != ? AND reviewed_at != ''""",
                (ReviewStatus.PENDING,),
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
            placeholders = ",".join("?" for _ in allowed_job_ids)
            clauses.append(f"job_id IN ({placeholders})")
            params.extend(allowed_job_ids)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = (
            f"SELECT * FROM review_items{where} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        with self._lock:
            conn = self._get_conn()
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
            placeholders = ",".join("?" for _ in allowed_job_ids)
            clauses.append(f"job_id IN ({placeholders})")
            params.extend(allowed_job_ids)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"SELECT COUNT(*) as cnt FROM review_items{where}"

        with self._lock:
            conn = self._get_conn()
            row = conn.execute(query, params).fetchone()
        return row["cnt"] if row else 0
