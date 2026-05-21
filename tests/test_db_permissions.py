"""Tests for SQLite database files must not be world-readable.

The ``api.db_security.set_db_permissions`` helper restricts SQLite
database files to owner read/write only (mode 0o600). These tests
verify that the helper is invoked by each module that creates a
SQLite database and that the resulting file has the expected mode.

All mode-sensitive assertions are skipped on Windows because
``os.chmod`` only toggles the read-only flag there.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from api.db_security import set_db_permissions

WINDOWS = sys.platform == "win32"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _mode_bits(path: str) -> int:
    """Return the permission bits (low 9) of *path*."""
    return stat.S_IMODE(os.stat(path).st_mode)


# ---------------------------------------------------------------------------
# set_db_permissions helper
# ---------------------------------------------------------------------------


class TestSetDbPermissions:
    @pytest.mark.skipif(WINDOWS, reason="Windows chmod is limited to read-only flag")
    def test_sets_mode_0600(self, tmp_path):
        db = tmp_path / "example.db"
        db.write_bytes(b"")
        # Start with a permissive mode so the assertion is meaningful.
        os.chmod(db, 0o644)
        assert _mode_bits(str(db)) == 0o644

        set_db_permissions(str(db))

        assert _mode_bits(str(db)) == 0o600

    def test_missing_file_is_non_fatal(self, tmp_path):
        """Helper must never raise on missing files."""
        missing = str(tmp_path / "does_not_exist.db")
        # Must not raise.
        set_db_permissions(missing)

    def test_oserror_is_swallowed(self, tmp_path):
        """Any OSError from os.chmod is logged and swallowed."""
        db = tmp_path / "example.db"
        db.write_bytes(b"")

        with patch("api.db_security.os.chmod", side_effect=OSError("boom")):
            # Must not raise.
            set_db_permissions(str(db))


# ---------------------------------------------------------------------------
# api.database -- main jobs.db
# ---------------------------------------------------------------------------


class TestDatabaseJobsDbPermissions:
    @pytest.mark.skipif(WINDOWS, reason="Windows chmod is limited to read-only flag")
    def test_jobs_db_is_created_with_mode_0600(self, tmp_path):
        from api import database

        db_path = tmp_path / "jobs.db"

        # Reset cached engine so we start from a clean slate, then build
        # a fresh engine pointed at the tmp path.
        database.reset_engine()
        try:
            engine = database.get_engine(db_path=str(db_path))
            assert engine is not None
            assert db_path.exists()
            assert _mode_bits(str(db_path)) == 0o600
        finally:
            database.reset_engine()

    def test_get_engine_invokes_set_db_permissions(self, tmp_path):
        """Engine creation must call set_db_permissions exactly once."""
        from api import database

        db_path = tmp_path / "jobs.db"
        database.reset_engine()
        try:
            with patch("api.database.set_db_permissions") as mock_chmod:
                database.get_engine(db_path=str(db_path))
                mock_chmod.assert_called_once_with(str(db_path))
        finally:
            database.reset_engine()


# ---------------------------------------------------------------------------
# api.review_queue -- review queue SQLite
# ---------------------------------------------------------------------------


class TestReviewQueuePermissions:
    @pytest.mark.skipif(WINDOWS, reason="Windows chmod is limited to read-only flag")
    def test_review_queue_db_is_created_with_mode_0600(self, tmp_path):
        from api.review_queue import ReviewQueue

        db_path = tmp_path / "review.db"
        ReviewQueue(db_path=str(db_path))

        assert db_path.exists()
        assert _mode_bits(str(db_path)) == 0o600

    def test_review_queue_invokes_set_db_permissions(self, tmp_path):
        from api import review_queue

        db_path = tmp_path / "review.db"
        with patch("api.review_queue.set_db_permissions") as mock_chmod:
            review_queue.ReviewQueue(db_path=str(db_path))
            mock_chmod.assert_called_once_with(str(db_path))


# ---------------------------------------------------------------------------
# api.entity_index -- entity/extraction index SQLite
# ---------------------------------------------------------------------------


class TestEntityIndexPermissions:
    @pytest.mark.skipif(WINDOWS, reason="Windows chmod is limited to read-only flag")
    def test_entity_index_db_is_created_with_mode_0600(self, tmp_path):
        from api.entity_index import EntityIndex

        db_path = tmp_path / "entity_index.db"
        index = EntityIndex(db_path=str(db_path))
        # Schema init happens lazily on first query; force it via a
        # real insert (index_entities short-circuits on empty list).
        index.index_entities(
            "job_test000000",
            "doc.pdf",
            [{"type": "PERSON", "text": "A", "confidence": 0.9, "source": "ner", "page": 1}],
        )

        assert db_path.exists()
        assert _mode_bits(str(db_path)) == 0o600

    def test_entity_index_invokes_set_db_permissions(self, tmp_path):
        from api import entity_index

        db_path = tmp_path / "entity_index.db"
        with patch("api.entity_index.set_db_permissions") as mock_chmod:
            idx = entity_index.EntityIndex(db_path=str(db_path))
            # Trigger schema init (and therefore the permission call).
            idx.index_entities(
                "job_test000000",
                "doc.pdf",
                [{"type": "PERSON", "text": "A", "confidence": 0.9, "source": "ner", "page": 1}],
            )
            mock_chmod.assert_called_once_with(str(db_path))


# ---------------------------------------------------------------------------
# Negative test: default permissions would be world-readable
# ---------------------------------------------------------------------------


@pytest.mark.skipif(WINDOWS, reason="POSIX permission semantics only")
def test_default_sqlite_file_would_be_world_readable(tmp_path):
    """Sanity check: without the fix, SQLite files get 0o644 by default.

    This test documents why the fix is necessary -- a fresh SQLite
    database file created on POSIX inherits the process umask, which is
    typically 0o022 and yields 0o644 (world-readable).  tightens
    that to 0o600.
    """
    import sqlite3

    # Force a permissive umask so this test is deterministic across CI.
    old_umask = os.umask(0o022)
    try:
        db = tmp_path / "baseline.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
        conn.close()

        # World- or group-readable bit should be set (demonstrates the
        # pre-fix state). Don't assert the exact mode because CI umask
        # varies; just assert that "other" has *some* access.
        mode = _mode_bits(str(db))
        others_readable = bool(mode & stat.S_IROTH) or bool(mode & stat.S_IRGRP)
        assert others_readable, f"Expected non-0o600 default mode, got {oct(mode)}"

        # Now apply the fix and verify it tightens to 0o600.
        set_db_permissions(str(db))
        assert _mode_bits(str(db)) == 0o600
    finally:
        os.umask(old_umask)


# ---------------------------------------------------------------------------
# Path reference: ensure the helper accepts common path inputs
# ---------------------------------------------------------------------------


class TestAcceptsPathInput:
    """set_db_permissions should accept string paths for SQLite files."""

    @pytest.mark.skipif(WINDOWS, reason="Windows chmod is limited to read-only flag")
    def test_accepts_str_path(self, tmp_path):
        db = tmp_path / "strpath.db"
        db.write_bytes(b"")
        os.chmod(db, 0o644)
        set_db_permissions(str(db))
        assert _mode_bits(str(db)) == 0o600

    @pytest.mark.skipif(WINDOWS, reason="Windows chmod is limited to read-only flag")
    def test_accepts_stringified_pathlib_path(self, tmp_path):
        db: Path = tmp_path / "pathlib.db"
        db.write_bytes(b"")
        os.chmod(db, 0o644)
        set_db_permissions(str(db))
        assert _mode_bits(str(db)) == 0o600
