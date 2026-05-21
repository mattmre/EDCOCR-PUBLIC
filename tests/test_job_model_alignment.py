"""Tests for Job model alignment between API and coordinator.

Verifies cross-reference fields and aligned shared fields exist in the
API Job model (SQLAlchemy / SQLite).
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

# Ensure API config picks up a temp DB path before import
os.environ.setdefault("OCR_OUTPUT_DIR", tempfile.mkdtemp())

from api.database import Base, Job  # noqa: E402


@pytest.fixture()
def db_session(tmp_path):
    """Create a fresh in-memory SQLite database with the Job table."""
    db_file = tmp_path / "test_alignment.db"
    engine = create_engine(f"sqlite:///{db_file}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


# -- Schema presence tests --------------------------------------------------


class TestApiJobCrossReferenceFields:
    """Verify that cross-reference and aligned fields exist on the API Job."""

    def test_coordinator_job_id_column_exists(self):
        col_names = [c.key for c in Job.__table__.columns]
        assert "coordinator_job_id" in col_names

    def test_source_type_column_exists(self):
        col_names = [c.key for c in Job.__table__.columns]
        assert "source_type" in col_names

    def test_pages_failed_column_exists(self):
        col_names = [c.key for c in Job.__table__.columns]
        assert "pages_failed" in col_names


class TestApiJobFieldProperties:
    """Verify column properties match the design spec."""

    def test_coordinator_job_id_nullable(self, db_session):
        """coordinator_job_id must accept None (not delegated to coordinator)."""
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            source_file="/tmp/test.pdf",
            coordinator_job_id=None,
        )
        db_session.add(job)
        db_session.commit()
        fetched = db_session.query(Job).filter_by(job_id=job.job_id).one()
        assert fetched.coordinator_job_id is None

    def test_coordinator_job_id_accepts_uuid_string(self, db_session):
        """coordinator_job_id stores a UUID string when set."""
        coord_id = str(uuid.uuid4())
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            source_file="/tmp/test.pdf",
            coordinator_job_id=coord_id,
        )
        db_session.add(job)
        db_session.commit()
        fetched = db_session.query(Job).filter_by(job_id=job.job_id).one()
        assert fetched.coordinator_job_id == coord_id

    def test_source_type_nullable(self, db_session):
        """source_type is nullable (backward compat with existing rows)."""
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            source_file="/tmp/test.pdf",
        )
        db_session.add(job)
        db_session.commit()
        fetched = db_session.query(Job).filter_by(job_id=job.job_id).one()
        assert fetched.source_type is None

    def test_source_type_stores_value(self, db_session):
        """source_type stores short string like 'pdf' or 'image'."""
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            source_file="/tmp/test.pdf",
            source_type="pdf",
        )
        db_session.add(job)
        db_session.commit()
        fetched = db_session.query(Job).filter_by(job_id=job.job_id).one()
        assert fetched.source_type == "pdf"

    def test_pages_failed_default_zero(self, db_session):
        """pages_failed defaults to 0."""
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            source_file="/tmp/test.pdf",
        )
        db_session.add(job)
        db_session.commit()
        fetched = db_session.query(Job).filter_by(job_id=job.job_id).one()
        assert fetched.pages_failed == 0

    def test_pages_failed_stores_count(self, db_session):
        """pages_failed stores an integer count."""
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            source_file="/tmp/test.pdf",
            pages_failed=3,
        )
        db_session.add(job)
        db_session.commit()
        fetched = db_session.query(Job).filter_by(job_id=job.job_id).one()
        assert fetched.pages_failed == 3


class TestApiJobIdFormat:
    """Verify API job_id format is distinct from coordinator UUID format."""

    def test_api_job_id_format(self):
        """API job_id uses 'job_<hex12>' format (16 chars total)."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        assert job_id.startswith("job_")
        assert len(job_id) == 16

    def test_coordinator_uuid_format_different(self):
        """Coordinator UUID is 36 chars with dashes — distinct from API format."""
        coord_id = str(uuid.uuid4())
        assert len(coord_id) == 36
        assert "-" in coord_id

    def test_formats_never_collide(self):
        """API job_id (16 chars, no dashes) never matches UUID (36 chars, dashes)."""
        api_id = f"job_{uuid.uuid4().hex[:12]}"
        coord_id = str(uuid.uuid4())
        assert len(api_id) != len(coord_id)
        assert not re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            api_id,
        )


class TestCoordinatorMigrationExists:
    """Verify the Django migration file for api_job_id exists."""

    def test_migration_0010_exists(self):
        migration_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "coordinator",
            "jobs",
            "migrations",
            "0010_job_api_job_id.py",
        )
        assert os.path.isfile(migration_path), (
            f"Migration file not found: {migration_path}"
        )

    def test_migration_references_api_job_id(self):
        migration_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "coordinator",
            "jobs",
            "migrations",
            "0010_job_api_job_id.py",
        )
        with open(migration_path) as f:
            content = f.read()
        assert "api_job_id" in content
        assert "max_length=64" in content


class TestCoordinatorXrefSchemaMigration:
    """Verify _ensure_coordinator_xref_schema adds columns to existing DBs."""

    def test_adds_columns_to_existing_table(self, tmp_path):
        """Simulate an old DB without new columns and verify migration."""
        from api.database import _ensure_coordinator_xref_schema

        db_file = tmp_path / "legacy.db"
        engine = create_engine(f"sqlite:///{db_file}", echo=False)
        # Create a minimal jobs table without new columns
        with engine.connect() as conn:
            conn.execute(
                __import__("sqlalchemy").text(
                    "CREATE TABLE jobs (job_id VARCHAR(64) PRIMARY KEY, "
                    "status VARCHAR(20), source_file VARCHAR(512))"
                )
            )
            conn.commit()

        _ensure_coordinator_xref_schema(engine)

        insp = inspect(engine)
        columns = {col["name"] for col in insp.get_columns("jobs")}
        assert "coordinator_job_id" in columns
        assert "source_type" in columns
        assert "pages_failed" in columns
        engine.dispose()

    def test_index_created(self, tmp_path):
        """Verify idx_jobs_coordinator_id index is created."""
        from api.database import _ensure_coordinator_xref_schema

        db_file = tmp_path / "legacy_idx.db"
        engine = create_engine(f"sqlite:///{db_file}", echo=False)
        with engine.connect() as conn:
            conn.execute(
                __import__("sqlalchemy").text(
                    "CREATE TABLE jobs (job_id VARCHAR(64) PRIMARY KEY)"
                )
            )
            conn.commit()

        _ensure_coordinator_xref_schema(engine)

        insp = inspect(engine)
        indexes = {idx["name"] for idx in insp.get_indexes("jobs") if idx.get("name")}
        assert "idx_jobs_coordinator_id" in indexes
        engine.dispose()
