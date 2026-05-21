"""Entity search index for cross-document entity and extraction recall.

Maintains a SQLite index of entities and extracted key-value pairs from
processed documents. Populated from entity consolidator output during
or after job processing.

Design: append-only index -- entities are added when jobs complete,
never updated. Deleted when the parent job is cleaned up.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from api.config import OUTPUT_FOLDER
from api.db_security import set_db_permissions

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = os.path.join(OUTPUT_FOLDER, "entity_index.db")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IndexedEntity:
    """A single indexed entity record."""

    entity_id: str
    job_id: str
    entity_type: str
    text: str
    confidence: float = 0.0
    source: str = ""  # "ner", "extraction", "classification"
    page: int = 0
    document_name: str = ""
    indexed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "job_id": self.job_id,
            "entity_type": self.entity_type,
            "text": self.text,
            "confidence": self.confidence,
            "source": self.source,
            "page": self.page,
            "document_name": self.document_name,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> IndexedEntity:
        return cls(
            entity_id=row["entity_id"],
            job_id=row["job_id"],
            entity_type=row["entity_type"],
            text=row["text"],
            confidence=row["confidence"],
            source=row["source"],
            page=row["page"],
            document_name=row["document_name"],
            indexed_at=row["indexed_at"],
        )


@dataclass
class IndexedExtraction:
    """A single indexed key-value extraction record."""

    extraction_id: str
    job_id: str
    field_name: str  # "invoice_number", "date", "amount", etc.
    field_value: str
    confidence: float = 0.0
    page: int = 0
    document_name: str = ""
    indexed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "extraction_id": self.extraction_id,
            "job_id": self.job_id,
            "field_name": self.field_name,
            "field_value": self.field_value,
            "confidence": self.confidence,
            "page": self.page,
            "document_name": self.document_name,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> IndexedExtraction:
        return cls(
            extraction_id=row["extraction_id"],
            job_id=row["job_id"],
            field_name=row["field_name"],
            field_value=row["field_value"],
            confidence=row["confidence"],
            page=row["page"],
            document_name=row["document_name"],
            indexed_at=row["indexed_at"],
        )


# ---------------------------------------------------------------------------
# Entity index
# ---------------------------------------------------------------------------


def _generate_id(prefix: str) -> str:
    """Generate a short unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _escape_like(value: str) -> str:
    r"""Escape LIKE wildcard characters (``%``, ``_``, ``\``) in *value*.

    The escaped value is intended for use with ``ESCAPE '\'`` in a
    LIKE clause so that literal ``%`` and ``_`` characters in user input
    are not treated as wildcards.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class EntityIndex:
    """SQLite-backed entity and extraction search index.

    Thread-safe: uses threading.local() for per-thread connections
    and an RLock to serialize all database operations.

    SQLite WAL mode: enables concurrent reads during writes.
    See docs/architecture/adr-sqlite-to-postgresql-migration.md (/).
    """

    def __init__(self, db_path: str = ""):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._lock = threading.RLock()
        self._initialized = False

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA wal_autocheckpoint=1000")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        if not self._initialized:
            self._ensure_schema(conn)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create tables and indexes if they do not exist."""
        with self._init_lock:
            if self._initialized:
                return
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS indexed_entities (
                    entity_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    source TEXT DEFAULT '',
                    page INTEGER DEFAULT 0,
                    document_name TEXT DEFAULT '',
                    indexed_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_entities_job
                    ON indexed_entities (job_id);
                CREATE INDEX IF NOT EXISTS idx_entities_type
                    ON indexed_entities (entity_type);
                CREATE INDEX IF NOT EXISTS idx_entities_confidence
                    ON indexed_entities (confidence);

                CREATE TABLE IF NOT EXISTS indexed_extractions (
                    extraction_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    field_value TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    page INTEGER DEFAULT 0,
                    document_name TEXT DEFAULT '',
                    indexed_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_extractions_job
                    ON indexed_extractions (job_id);
                CREATE INDEX IF NOT EXISTS idx_extractions_field
                    ON indexed_extractions (field_name);
                CREATE INDEX IF NOT EXISTS idx_extractions_confidence
                    ON indexed_extractions (confidence);
            """)
            # restrict DB file to owner read/write only. Must run
            # after the first connect so the file exists on disk.
            set_db_permissions(self._db_path)
            self._initialized = True

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_entities(
        self, job_id: str, document_name: str, entities: list[dict]
    ) -> int:
        """Index entities from a completed job. Returns count indexed."""
        if not entities:
            return 0

        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        rows = []
        for ent in entities:
            entity_id = _generate_id("ent")
            rows.append((
                entity_id,
                job_id,
                ent.get("type", ent.get("entity_type", "")),
                ent.get("text", ""),
                float(ent.get("confidence", 0.0)),
                ent.get("source", ""),
                int(ent.get("page", 0)),
                document_name,
                now,
            ))

        with self._lock:
            conn = self._get_conn()
            conn.executemany(
                """INSERT INTO indexed_entities
                   (entity_id, job_id, entity_type, text, confidence, source,
                    page, document_name, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
        logger.debug(
            "Indexed %d entities for job %s document %s",
            len(rows), job_id, document_name,
        )
        return len(rows)

    def index_extractions(
        self, job_id: str, document_name: str, extractions: list[dict]
    ) -> int:
        """Index key-value extractions from a completed job. Returns count indexed."""
        if not extractions:
            return 0

        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        rows = []
        for ext in extractions:
            extraction_id = _generate_id("ext")
            rows.append((
                extraction_id,
                job_id,
                ext.get("key", ext.get("field_name", "")),
                ext.get("value", ext.get("field_value", "")),
                float(ext.get("confidence", 0.0)),
                int(ext.get("page", 0)),
                document_name,
                now,
            ))

        with self._lock:
            conn = self._get_conn()
            conn.executemany(
                """INSERT INTO indexed_extractions
                   (extraction_id, job_id, field_name, field_value, confidence,
                    page, document_name, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
        logger.debug(
            "Indexed %d extractions for job %s document %s",
            len(rows), job_id, document_name,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_entities(
        self,
        entity_type: Optional[str] = None,
        text_query: Optional[str] = None,
        job_id: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
        offset: int = 0,
        allowed_job_ids: Optional[list[str]] = None,
    ) -> tuple[list[IndexedEntity], int]:
        """Search indexed entities. Returns (results, total_count).

        Parameters
        ----------
        allowed_job_ids : list[str], optional
            When provided, restrict results to these job IDs only
            (used for multi-tenant isolation).
        """
        conditions: list[str] = []
        params: list = []

        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        if text_query:
            conditions.append(r"text LIKE ? ESCAPE '\'")
            params.append(f"%{_escape_like(text_query)}%")
        if job_id:
            conditions.append("job_id = ?")
            params.append(job_id)
        if allowed_job_ids is not None and not job_id:
            if not allowed_job_ids:
                # No jobs for this tenant -- return empty
                return [], 0
            placeholders = ",".join("?" for _ in allowed_job_ids)
            conditions.append(f"job_id IN ({placeholders})")
            params.extend(allowed_job_ids)
        if min_confidence > 0.0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._lock:
            conn = self._get_conn()

            # Total count
            count_row = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM indexed_entities{where}",
                params,
            ).fetchone()
            total = count_row["cnt"] if count_row else 0

            # Paginated results
            results_rows = conn.execute(
                f"SELECT * FROM indexed_entities{where}"
                " ORDER BY indexed_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

        results = [IndexedEntity.from_row(r) for r in results_rows]
        return results, total

    def search_extractions(
        self,
        field_name: Optional[str] = None,
        value_query: Optional[str] = None,
        job_id: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
        offset: int = 0,
        allowed_job_ids: Optional[list[str]] = None,
    ) -> tuple[list[IndexedExtraction], int]:
        """Search indexed extractions. Returns (results, total_count).

        Parameters
        ----------
        allowed_job_ids : list[str], optional
            When provided, restrict results to these job IDs only
            (used for multi-tenant isolation).
        """
        conditions: list[str] = []
        params: list = []

        if field_name:
            conditions.append("field_name = ?")
            params.append(field_name)
        if value_query:
            conditions.append(r"field_value LIKE ? ESCAPE '\'")
            params.append(f"%{_escape_like(value_query)}%")
        if job_id:
            conditions.append("job_id = ?")
            params.append(job_id)
        if allowed_job_ids is not None and not job_id:
            if not allowed_job_ids:
                return [], 0
            placeholders = ",".join("?" for _ in allowed_job_ids)
            conditions.append(f"job_id IN ({placeholders})")
            params.extend(allowed_job_ids)
        if min_confidence > 0.0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._lock:
            conn = self._get_conn()

            count_row = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM indexed_extractions{where}",
                params,
            ).fetchone()
            total = count_row["cnt"] if count_row else 0

            results_rows = conn.execute(
                f"SELECT * FROM indexed_extractions{where}"
                " ORDER BY indexed_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

        results = [IndexedExtraction.from_row(r) for r in results_rows]
        return results, total

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_job(self, job_id: str) -> int:
        """Remove all index entries for a job. Returns count removed."""
        with self._lock:
            conn = self._get_conn()
            cursor_ent = conn.execute(
                "DELETE FROM indexed_entities WHERE job_id = ?", (job_id,)
            )
            cursor_ext = conn.execute(
                "DELETE FROM indexed_extractions WHERE job_id = ?", (job_id,)
            )
            conn.commit()
            total = cursor_ent.rowcount + cursor_ext.rowcount
        if total > 0:
            logger.debug("Removed %d index entries for job %s", total, job_id)
        return total

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Index statistics: total entities, total extractions, unique types, etc."""
        with self._lock:
            conn = self._get_conn()

            ent_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM indexed_entities"
            ).fetchone()["cnt"]

            ext_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM indexed_extractions"
            ).fetchone()["cnt"]

            unique_types = conn.execute(
                "SELECT COUNT(DISTINCT entity_type) AS cnt FROM indexed_entities"
            ).fetchone()["cnt"]

            unique_fields = conn.execute(
                "SELECT COUNT(DISTINCT field_name) AS cnt FROM indexed_extractions"
            ).fetchone()["cnt"]

            # Union of distinct job_ids from both tables
            jobs_indexed = conn.execute(
                "SELECT COUNT(*) AS cnt FROM ("
                "  SELECT DISTINCT job_id FROM indexed_entities"
                "  UNION"
                "  SELECT DISTINCT job_id FROM indexed_extractions"
                ")"
            ).fetchone()["cnt"]

        return {
            "total_entities": ent_count,
            "total_extractions": ext_count,
            "unique_entity_types": unique_types,
            "unique_field_names": unique_fields,
            "jobs_indexed": jobs_indexed,
        }
