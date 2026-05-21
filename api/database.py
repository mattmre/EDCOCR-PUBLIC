"""SQLAlchemy models and session management (SQLite for Phase 4A)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import DB_PATH
from api.db_security import set_db_permissions

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    """Multi-tenancy: organization record."""

    __tablename__ = "tenants"

    tenant_id = Column(String(64), primary_key=True)  # tenant_{hex12}
    name = Column(String(256), nullable=False)
    display_name = Column(String(256), nullable=True)
    status = Column(String(20), nullable=False, default="active")  # active, suspended, deleted
    tier = Column(String(20), nullable=False, default="standard")  # free, standard, enterprise
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=True)
    max_concurrent_jobs = Column(Integer, default=4)
    max_pages_per_month = Column(Integer, default=10000)
    max_storage_bytes = Column(Integer, default=10 * 1024**3)  # 10 GiB
    allowed_features = Column(Text, default="[]")  # JSON list
    admin_email = Column(String(256), nullable=True)


class TenantApiKey(Base):
    """Multi-tenancy: API key record (stores SHA-256 hash only)."""

    __tablename__ = "tenant_api_keys"

    key_id = Column(String(64), primary_key=True)  # key_{hex12}
    tenant_id = Column(String(64), ForeignKey("tenants.tenant_id"), nullable=False)
    api_key_hash = Column(String(128), nullable=False)  # SHA-256 hex digest
    name = Column(String(256), nullable=True)
    status = Column(String(20), nullable=False, default="active")  # active, revoked
    permissions = Column(Text, default='["submit","read"]')  # JSON list
    created_at = Column(DateTime, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)


class TenantRateLimit(Base):
    """Multi-tenancy: per-tenant rate limit override."""

    __tablename__ = "tenant_rate_limits"

    tenant_id = Column(
        String(64), ForeignKey("tenants.tenant_id"), primary_key=True
    )
    rate_limit = Column(
        String(64), nullable=False
    )  # e.g. "100/minute", "500/hour"


class UsageRecord(Base):
    """Multi-tenancy: monthly usage tracking per tenant."""

    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), ForeignKey("tenants.tenant_id"), nullable=False)
    period = Column(String(7), nullable=False)  # "2026-03"
    jobs_submitted = Column(Integer, default=0)
    pages_processed = Column(Integer, default=0)
    storage_bytes_used = Column(Integer, default=0)
    api_calls = Column(Integer, default=0)
    processing_seconds = Column(Float, default=0.0)

    __table_args__ = (
        Index("uq_usage_records_tenant_period", "tenant_id", "period", unique=True),
    )


class Job(Base):
    """Persistent job record."""

    __tablename__ = "jobs"

    job_id = Column(String(64), primary_key=True)
    status = Column(String(20), nullable=False, default="submitted")
    priority = Column(String(10), nullable=False, default="normal")
    source_file = Column(String(512), nullable=False)
    source_hash = Column(String(64), nullable=True)
    total_pages = Column(Integer, nullable=True)
    pages_completed = Column(Integer, default=0)
    current_stage = Column(String(30), default="submitted")
    settings_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    result_path = Column(String(512), nullable=True)
    pid = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)
    batch_id = Column(String(64), nullable=True)

    # Webhook notification fields
    webhook_url = Column(String(2048), nullable=True)
    webhook_secret = Column(String(256), nullable=True)
    webhook_status = Column(String(20), nullable=True)  # pending, delivered, failed
    webhook_attempts = Column(Integer, default=0)
    webhook_last_error = Column(Text, nullable=True)

    # Multi-tenancy (nullable for backward compat with existing jobs)
    tenant_id = Column(String(64), nullable=True)

    # Cross-reference to coordinator (nullable — only set when job is delegated)
    coordinator_job_id = Column(String(36), nullable=True, index=True)
    # Fields present in coordinator but missing from API model
    source_type = Column(String(10), nullable=True)  # 'pdf', 'image', etc.
    pages_failed = Column(Integer, default=0)

    __table_args__ = (
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_created", "created_at"),
        Index("idx_jobs_batch_id", "batch_id"),
        Index("idx_jobs_tenant", "tenant_id"),
        Index("idx_jobs_priority", "priority"),
        Index("idx_jobs_coordinator_id", "coordinator_job_id"),
    )

    @property
    def settings(self) -> dict:
        return json.loads(self.settings_json) if self.settings_json else {}

    @settings.setter
    def settings(self, value: dict) -> None:
        self.settings_json = json.dumps(value)

    def percent_complete(self) -> float:
        if not self.total_pages or self.total_pages == 0:
            return 0.0
        return round(self.pages_completed / self.total_pages * 100, 1)


class Batch(Base):
    """Persistent batch record grouping multiple jobs."""

    __tablename__ = "batches"

    batch_id = Column(String(64), primary_key=True)
    status = Column(String(20), nullable=False, default="submitted")
    total_jobs = Column(Integer, default=0)
    jobs_completed = Column(Integer, default=0)
    jobs_failed = Column(Integer, default=0)
    jobs_cancelled = Column(Integer, default=0)
    priority = Column(String(10), nullable=False, default="normal")
    settings_json = Column(Text, default="{}")
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )
    completed_at = Column(DateTime, nullable=True)
    processing_time = Column(Float, nullable=True)

    # Webhook notification fields
    webhook_url = Column(String(2048), nullable=True)
    webhook_secret = Column(String(256), nullable=True)
    webhook_status = Column(String(20), nullable=True)
    webhook_attempts = Column(Integer, default=0)
    webhook_last_error = Column(Text, nullable=True)

    @property
    def settings(self) -> dict:
        return json.loads(self.settings_json) if self.settings_json else {}

    @settings.setter
    def settings(self, value: dict) -> None:
        self.settings_json = json.dumps(value)


def _ensure_batch_schema(engine) -> None:
    """Add batch schema elements required for older databases."""
    insp = inspect(engine)
    if "jobs" not in insp.get_table_names():
        return

    columns = {col["name"] for col in insp.get_columns("jobs")}
    if "batch_id" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN batch_id VARCHAR(64)"))
            conn.commit()
        logger.info("Added batch_id column to jobs table")

    indexes = {idx["name"] for idx in insp.get_indexes("jobs") if idx.get("name")}
    if "idx_jobs_batch_id" not in indexes:
        with engine.connect() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_jobs_batch_id ON jobs (batch_id)"))
            conn.commit()
        logger.info("Created idx_jobs_batch_id index")


def _ensure_tenant_schema(engine) -> None:
    """Migrate existing database: add multi-tenancy tables and columns if missing."""
    insp = inspect(engine)
    existing_tables = insp.get_table_names()

    for table_name in ("tenants", "tenant_api_keys", "usage_records", "tenant_rate_limits"):
        if table_name not in existing_tables:
            Base.metadata.tables[table_name].create(engine)
            logger.info("Created table: %s", table_name)

    if "jobs" in existing_tables:
        columns = {col["name"] for col in insp.get_columns("jobs")}
        if "tenant_id" not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN tenant_id VARCHAR(64)"))
                conn.commit()
            logger.info("Added tenant_id column to jobs table")

        indexes = {idx["name"] for idx in insp.get_indexes("jobs") if idx.get("name")}
        if "idx_jobs_tenant" not in indexes:
            with engine.connect() as conn:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_jobs_tenant ON jobs (tenant_id)"))
                conn.commit()
            logger.info("Created idx_jobs_tenant index")

    if "usage_records" in insp.get_table_names():
        columns = {col["name"] for col in insp.get_columns("usage_records")}
        if "processing_seconds" not in columns:
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE usage_records "
                        "ADD COLUMN processing_seconds FLOAT DEFAULT 0.0"
                    )
                )
                conn.commit()
            logger.info("Added processing_seconds column to usage_records table")

        indexes = {
            idx["name"] for idx in insp.get_indexes("usage_records") if idx.get("name")
        }
        if "uq_usage_records_tenant_period" not in indexes:
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "uq_usage_records_tenant_period ON usage_records (tenant_id, period)"
                    )
                )
                conn.commit()
            logger.info("Created uq_usage_records_tenant_period index")


def _ensure_coordinator_xref_schema(engine) -> None:
    """Add coordinator cross-reference columns to existing jobs table."""
    insp = inspect(engine)
    if "jobs" not in insp.get_table_names():
        return

    columns = {col["name"] for col in insp.get_columns("jobs")}

    if "coordinator_job_id" not in columns:
        with engine.connect() as conn:
            conn.execute(
                text("ALTER TABLE jobs ADD COLUMN coordinator_job_id VARCHAR(36)")
            )
            conn.commit()
        logger.info("Added coordinator_job_id column to jobs table")

    if "source_type" not in columns:
        with engine.connect() as conn:
            conn.execute(
                text("ALTER TABLE jobs ADD COLUMN source_type VARCHAR(10)")
            )
            conn.commit()
        logger.info("Added source_type column to jobs table")

    if "pages_failed" not in columns:
        with engine.connect() as conn:
            conn.execute(
                text("ALTER TABLE jobs ADD COLUMN pages_failed INTEGER DEFAULT 0")
            )
            conn.commit()
        logger.info("Added pages_failed column to jobs table")

    indexes = {idx["name"] for idx in insp.get_indexes("jobs") if idx.get("name")}
    if "idx_jobs_coordinator_id" not in indexes:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_coordinator_id "
                    "ON jobs (coordinator_job_id)"
                )
            )
            conn.commit()
        logger.info("Created idx_jobs_coordinator_id index")


# ---------------------------------------------------------------------------
# Engine / session factory
# ---------------------------------------------------------------------------
# SQLite WAL mode: enables concurrent reads during writes.
# See docs/architecture/adr-sqlite-to-postgresql-migration.md (/).

_engine = None
_SessionLocal = None
_db_init_lock = threading.Lock()


def _configure_sqlite_wal(dbapi_conn, connection_record):
    """Set WAL journal mode and tuning PRAGMAs on every new SQLite connection.

    WAL (Write-Ahead Logging) allows concurrent readers while one writer
    is active, which is a significant improvement over the default
    rollback-journal mode that blocks all access during writes.

    PRAGMA synchronous=NORMAL is safe with WAL and avoids the fsync
    overhead of FULL mode on every commit.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA wal_autocheckpoint=1000")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def get_engine(db_path: Optional[str] = None):
    global _engine
    if _engine is None:
        with _db_init_lock:
            if _engine is None:
                path = db_path or DB_PATH
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                _engine = create_engine(
                    f"sqlite:///{path}",
                    echo=False,
                    connect_args={"check_same_thread": False},
                )
                event.listen(_engine, "connect", _configure_sqlite_wal)
                _ensure_batch_schema(_engine)
                Base.metadata.create_all(_engine)
                _ensure_tenant_schema(_engine)
                _ensure_coordinator_xref_schema(_engine)
                # restrict DB file to owner read/write only. Must
                # run after create_all so the file exists on disk.
                set_db_permissions(path)
    return _engine


def get_session_factory(db_path: Optional[str] = None) -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        with _db_init_lock:
            if _SessionLocal is None:
                _SessionLocal = sessionmaker(bind=get_engine(db_path))
    return _SessionLocal


def get_db(db_path: Optional[str] = None) -> Session:
    """Yield a session for dependency injection."""
    factory = get_session_factory(db_path)
    session = factory()
    try:
        yield session
    finally:
        session.close()


def reset_engine():
    """Reset cached engine/session (used in tests)."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
