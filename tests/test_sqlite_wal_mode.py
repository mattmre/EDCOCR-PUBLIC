"""Tests verifying WAL journal mode is enabled on all SQLite databases.

Each of the three API-layer SQLite databases must use WAL mode to enable
concurrent reads during writes. This test module verifies that PRAGMA
settings are correctly applied at connection creation time.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from api.entity_index import EntityIndex
from api.review_queue import ReviewQueue

# ---------------------------------------------------------------------------
# api/database.py — SQLAlchemy engine with event listener
# ---------------------------------------------------------------------------


class TestDatabaseWalMode:
    """Verify WAL mode on the main SQLAlchemy-managed jobs database."""

    def test_engine_uses_wal_mode(self, tmp_path):
        """SQLAlchemy engine connections should have WAL journal mode."""
        from api.database import reset_engine

        db_file = str(tmp_path / "test_jobs.db")

        # Reset any cached engine from prior tests
        reset_engine()

        try:
            with patch("api.database.DB_PATH", db_file):
                from api.database import get_engine

                engine = get_engine(db_path=db_file)

                with engine.connect() as conn:
                    result = conn.execute(
                        __import__("sqlalchemy").text("PRAGMA journal_mode")
                    ).fetchone()
                    assert result[0] == "wal"

                    result = conn.execute(
                        __import__("sqlalchemy").text("PRAGMA synchronous")
                    ).fetchone()
                    # synchronous=NORMAL is value 1
                    assert result[0] == 1

                    result = conn.execute(
                        __import__("sqlalchemy").text("PRAGMA busy_timeout")
                    ).fetchone()
                    assert result[0] == 5000

                    result = conn.execute(
                        __import__("sqlalchemy").text("PRAGMA wal_autocheckpoint")
                    ).fetchone()
                    assert result[0] == 1000
        finally:
            reset_engine()

    def test_wal_sidecar_files_created(self, tmp_path):
        """WAL mode creates -wal and -shm sidecar files on first write."""
        from api.database import reset_engine

        db_file = str(tmp_path / "test_sidecar.db")
        reset_engine()

        try:
            with patch("api.database.DB_PATH", db_file):
                from api.database import get_engine

                engine = get_engine(db_path=db_file)

                # Force a write to create sidecar files
                with engine.connect() as conn:
                    conn.execute(
                        __import__("sqlalchemy").text(
                            "CREATE TABLE IF NOT EXISTS _wal_test (id INTEGER)"
                        )
                    )
                    conn.commit()

                wal_path = tmp_path / "test_sidecar.db-wal"
                shm_path = tmp_path / "test_sidecar.db-shm"
                assert wal_path.exists() or shm_path.exists(), (
                    "WAL mode should create -wal and/or -shm sidecar files"
                )
        finally:
            reset_engine()


# ---------------------------------------------------------------------------
# api/review_queue.py — raw sqlite3 connections
# ---------------------------------------------------------------------------


class TestReviewQueueWalMode:
    """Verify WAL mode on the review queue database."""

    def test_review_queue_uses_wal_mode(self, tmp_path):
        db_file = str(tmp_path / "test_review.db")
        with patch("api.review_queue.DB_PATH", db_file):
            queue = ReviewQueue(db_path=db_file)
            conn = queue._get_conn()

            result = conn.execute("PRAGMA journal_mode").fetchone()
            assert result[0] == "wal"

    def test_review_queue_synchronous_normal(self, tmp_path):
        db_file = str(tmp_path / "test_review_sync.db")
        with patch("api.review_queue.DB_PATH", db_file):
            queue = ReviewQueue(db_path=db_file)
            conn = queue._get_conn()

            result = conn.execute("PRAGMA synchronous").fetchone()
            # synchronous=NORMAL is value 1
            assert result[0] == 1

    def test_review_queue_busy_timeout(self, tmp_path):
        db_file = str(tmp_path / "test_review_bt.db")
        with patch("api.review_queue.DB_PATH", db_file):
            queue = ReviewQueue(db_path=db_file)
            conn = queue._get_conn()

            result = conn.execute("PRAGMA busy_timeout").fetchone()
            assert result[0] == 5000

    def test_review_queue_wal_autocheckpoint(self, tmp_path):
        db_file = str(tmp_path / "test_review_cp.db")
        with patch("api.review_queue.DB_PATH", db_file):
            queue = ReviewQueue(db_path=db_file)
            conn = queue._get_conn()

            result = conn.execute("PRAGMA wal_autocheckpoint").fetchone()
            assert result[0] == 1000


# ---------------------------------------------------------------------------
# api/entity_index.py — raw sqlite3 connections
# ---------------------------------------------------------------------------


class TestEntityIndexWalMode:
    """Verify WAL mode on the entity index database."""

    def test_entity_index_uses_wal_mode(self, tmp_path):
        db_file = str(tmp_path / "test_entity.db")
        index = EntityIndex(db_path=db_file)
        conn = index._get_conn()

        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"

    def test_entity_index_synchronous_normal(self, tmp_path):
        db_file = str(tmp_path / "test_entity_sync.db")
        index = EntityIndex(db_path=db_file)
        conn = index._get_conn()

        result = conn.execute("PRAGMA synchronous").fetchone()
        assert result[0] == 1

    def test_entity_index_busy_timeout(self, tmp_path):
        db_file = str(tmp_path / "test_entity_bt.db")
        index = EntityIndex(db_path=db_file)
        conn = index._get_conn()

        result = conn.execute("PRAGMA busy_timeout").fetchone()
        assert result[0] == 5000

    def test_entity_index_wal_autocheckpoint(self, tmp_path):
        db_file = str(tmp_path / "test_entity_cp.db")
        index = EntityIndex(db_path=db_file)
        conn = index._get_conn()

        result = conn.execute("PRAGMA wal_autocheckpoint").fetchone()
        assert result[0] == 1000


# ---------------------------------------------------------------------------
# Cross-database consistency
# ---------------------------------------------------------------------------


class TestWalConsistency:
    """All three databases use the same PRAGMA configuration."""

    @pytest.mark.parametrize(
        "pragma,expected",
        [
            ("journal_mode", "wal"),
            ("synchronous", 1),
            ("busy_timeout", 5000),
            ("wal_autocheckpoint", 1000),
        ],
    )
    def test_review_queue_pragmas(self, tmp_path, pragma, expected):
        db_file = str(tmp_path / f"test_rq_{pragma}.db")
        with patch("api.review_queue.DB_PATH", db_file):
            queue = ReviewQueue(db_path=db_file)
            conn = queue._get_conn()
            result = conn.execute(f"PRAGMA {pragma}").fetchone()
            assert result[0] == expected

    @pytest.mark.parametrize(
        "pragma,expected",
        [
            ("journal_mode", "wal"),
            ("synchronous", 1),
            ("busy_timeout", 5000),
            ("wal_autocheckpoint", 1000),
        ],
    )
    def test_entity_index_pragmas(self, tmp_path, pragma, expected):
        db_file = str(tmp_path / f"test_ei_{pragma}.db")
        index = EntityIndex(db_path=db_file)
        conn = index._get_conn()
        result = conn.execute(f"PRAGMA {pragma}").fetchone()
        assert result[0] == expected
