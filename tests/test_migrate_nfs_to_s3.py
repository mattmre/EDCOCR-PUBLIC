"""Tests for scripts/migrate_nfs_to_s3.py.

Validates the NFS-to-S3 migration script including job discovery,
per-job migration with checksum verification, dry-run mode, resume
mode, and CLI argument handling.
"""

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the script under test via importlib (it lives in scripts/)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, _SCRIPTS_DIR)

_mod = importlib.import_module("migrate_nfs_to_s3")
sha256_file = _mod.sha256_file
discover_jobs = _mod.discover_jobs
migrate_job = _mod.migrate_job
run_migration = _mod.run_migration
main = _mod.main

# Import NFSBackend from the same module path used by the script
NFSBackend = _mod.NFSBackend

# C-18 / C-19 helpers
_load_state_file = _mod._load_state_file
_save_state_file = _mod._save_state_file
_update_job_db = _mod._update_job_db


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def nfs_tree(tmp_path):
    """Create a mock NFS directory structure with two jobs."""
    nfs_root = tmp_path / "nfs"
    jobs_dir = nfs_root / "jobs"

    # Job 1: two files
    job1_dir = jobs_dir / "job-001" / "output" / "EXPORT" / "PDF"
    job1_dir.mkdir(parents=True)
    (job1_dir / "doc.pdf").write_bytes(b"fake-pdf-content-job1")

    job1_text = jobs_dir / "job-001" / "output" / "EXPORT" / "TEXT"
    job1_text.mkdir(parents=True)
    (job1_text / "doc.txt").write_text("extracted text job1")

    # Job 2: one file
    job2_dir = jobs_dir / "job-002" / "output" / "EXPORT" / "PDF"
    job2_dir.mkdir(parents=True)
    (job2_dir / "doc2.pdf").write_bytes(b"fake-pdf-content-job2")

    return str(nfs_root)


@pytest.fixture
def mock_s3():
    """Create a mock S3Backend with in-memory storage."""
    s3 = MagicMock()
    s3.exists.return_value = False

    uploaded_files = {}

    def fake_upload(local_path, key):
        with open(local_path, "rb") as f:
            uploaded_files[key] = f.read()
        return f"s3://bucket/{key}"

    def fake_download(key, local_path):
        if key in uploaded_files:
            with open(local_path, "wb") as f:
                f.write(uploaded_files[key])
        return local_path

    s3.upload_file.side_effect = fake_upload
    s3.download_file.side_effect = fake_download
    s3._uploaded = uploaded_files
    return s3


# ===========================================================================
# sha256_file tests
# ===========================================================================


class TestSha256File:
    """Validate SHA-256 file hashing."""

    def test_sha256_known_content(self, tmp_path):
        """SHA-256 of known content should match expected digest."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")
        digest = sha256_file(str(f))
        assert len(digest) == 64
        assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_sha256_empty_file(self, tmp_path):
        """SHA-256 of empty file should match known empty hash."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        digest = sha256_file(str(f))
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ===========================================================================
# discover_jobs tests
# ===========================================================================


class TestDiscoverJobs:
    """Validate NFS job discovery."""

    def test_discover_jobs_finds_directories(self, nfs_tree):
        """Should find both job directories."""
        jobs = discover_jobs(nfs_tree)
        assert jobs == ["job-001", "job-002"]

    def test_discover_jobs_empty(self, tmp_path):
        """Empty NFS root should return empty list."""
        jobs = discover_jobs(str(tmp_path))
        assert jobs == []

    def test_discover_jobs_missing_dir(self, tmp_path):
        """Non-existent NFS root should return empty list."""
        jobs = discover_jobs(str(tmp_path / "nonexistent"))
        assert jobs == []

    def test_discover_jobs_ignores_files(self, tmp_path):
        """Files in jobs dir should be ignored (only directories)."""
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        (jobs_dir / "not-a-job.txt").write_text("ignore me")
        (jobs_dir / "real-job").mkdir()
        jobs = discover_jobs(str(tmp_path))
        assert jobs == ["real-job"]


# ===========================================================================
# migrate_job tests
# ===========================================================================


class TestMigrateJob:
    """Validate per-job migration logic."""

    def test_dry_run_counts_files(self, nfs_tree, mock_s3):
        """Dry run should count files without uploading (uses _to_upload keys)."""
        nfs = NFSBackend(root=nfs_tree)
        stats = migrate_job("job-001", nfs, mock_s3, execute=False)
        assert stats["files_found"] == 2
        assert stats["files_to_upload"] == 2  # counted in dry run
        assert stats["files_to_verify"] == 2  # auto-verified in dry run
        assert stats["dry_run"] is True
        assert stats["errors"] == []
        mock_s3.upload_file.assert_not_called()

    def test_execute_uploads_and_verifies(self, nfs_tree, mock_s3):
        """Execute mode should upload and verify each file."""
        nfs = NFSBackend(root=nfs_tree)
        stats = migrate_job("job-001", nfs, mock_s3, execute=True)
        assert stats["files_found"] == 2
        assert stats["files_uploaded"] == 2
        assert stats["files_verified"] == 2
        assert stats["bytes_uploaded"] > 0
        assert stats["errors"] == []

    def test_resume_skips_existing(self, nfs_tree, mock_s3):
        """Resume mode should skip files already in S3."""
        nfs = NFSBackend(root=nfs_tree)
        mock_s3.exists.return_value = True
        stats = migrate_job("job-001", nfs, mock_s3, execute=True, resume=True)
        assert stats["files_skipped"] == 2
        assert stats["files_uploaded"] == 0
        mock_s3.upload_file.assert_not_called()

    def test_empty_job_returns_zero_counts(self, tmp_path, mock_s3):
        """Job with no files should return zero counts."""
        nfs_root = tmp_path / "nfs"
        (nfs_root / "jobs" / "empty-job").mkdir(parents=True)
        nfs = NFSBackend(root=str(nfs_root))
        stats = migrate_job("empty-job", nfs, mock_s3, execute=True)
        assert stats["files_found"] == 0
        assert stats["files_uploaded"] == 0

    def test_delete_nfs_after_verified_upload(self, nfs_tree, mock_s3):
        """NFS cleanup should remove job directory after verified upload."""
        nfs = NFSBackend(root=nfs_tree)
        job_path = os.path.join(nfs_tree, "jobs", "job-001")
        assert os.path.isdir(job_path)

        stats = migrate_job(
            "job-001", nfs, mock_s3,
            execute=True, delete_nfs=True,
        )
        assert stats["files_deleted"] == 2
        assert not os.path.exists(job_path)

    def test_no_delete_on_verification_failure(self, nfs_tree, mock_s3):
        """NFS files should NOT be deleted if verification fails."""
        nfs = NFSBackend(root=nfs_tree)

        # Make download return different content (checksum mismatch)
        def bad_download(key, local_path):
            with open(local_path, "wb") as f:
                f.write(b"corrupted-data")
            return local_path

        mock_s3.download_file.side_effect = bad_download

        stats = migrate_job(
            "job-001", nfs, mock_s3,
            execute=True, delete_nfs=True,
        )
        assert stats["files_verified"] == 0
        assert stats["files_deleted"] == 0
        assert len(stats["errors"]) == 2  # Two checksum mismatches

        job_path = os.path.join(nfs_tree, "jobs", "job-001")
        assert os.path.isdir(job_path)  # Still exists


# ===========================================================================
# run_migration tests
# ===========================================================================


class TestRunMigration:
    """Validate the full migration orchestrator."""

    def test_dry_run_all_jobs(self, nfs_tree):
        """Dry run across all jobs should produce correct totals."""
        with patch("migrate_nfs_to_s3.S3Backend") as MockS3:
            mock_s3 = MagicMock()
            mock_s3.exists.return_value = False
            MockS3.return_value = mock_s3

            summary = run_migration(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
                execute=False,
            )
            t = summary["totals"]
            assert t["jobs_total"] == 2
            assert t["jobs_migrated"] == 2
            assert t["files_total"] == 3  # 2 in job-001, 1 in job-002
            assert t["jobs_with_errors"] == 0

    def test_specific_job_ids(self, nfs_tree):
        """Specifying job_ids should only migrate those jobs."""
        with patch("migrate_nfs_to_s3.S3Backend") as MockS3:
            mock_s3 = MagicMock()
            mock_s3.exists.return_value = False
            MockS3.return_value = mock_s3

            summary = run_migration(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
                execute=False,
                job_ids=["job-002"],
            )
            t = summary["totals"]
            assert t["jobs_total"] == 1
            assert len(summary["jobs"]) == 1
            assert summary["jobs"][0]["job_id"] == "job-002"


# ===========================================================================
# CLI tests
# ===========================================================================


class TestMainCli:
    """Validate the main() CLI entry point."""

    def test_dry_run_returns_zero(self, nfs_tree, capsys):
        """Dry run with valid args should return 0."""
        with patch("migrate_nfs_to_s3.S3Backend") as MockS3, \
             patch("migrate_nfs_to_s3.check_connectivity", return_value=(True, [])):
            mock_s3 = MagicMock()
            mock_s3.exists.return_value = False
            MockS3.return_value = mock_s3

            result = main([
                "--nfs-root", nfs_tree,
                "--s3-endpoint", "http://localhost:9000",
                "--s3-bucket", "test",
                "--s3-access-key", "key",
                "--s3-secret-key", "secret",
            ])
            assert result == 0
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out

    def test_delete_nfs_without_execute_fails(self, nfs_tree, capsys):
        """--delete-nfs without --execute should return 1."""
        result = main([
            "--nfs-root", nfs_tree,
            "--s3-endpoint", "http://localhost:9000",
            "--s3-bucket", "test",
            "--s3-access-key", "key",
            "--s3-secret-key", "secret",
            "--delete-nfs",
        ])
        assert result == 1

    def test_output_writes_json_report(self, nfs_tree, tmp_path):
        """--output should write JSON report file."""
        report_path = str(tmp_path / "report.json")
        with patch("migrate_nfs_to_s3.S3Backend") as MockS3, \
             patch("migrate_nfs_to_s3.check_connectivity", return_value=(True, [])):
            mock_s3 = MagicMock()
            mock_s3.exists.return_value = False
            MockS3.return_value = mock_s3

            result = main([
                "--nfs-root", nfs_tree,
                "--s3-endpoint", "http://localhost:9000",
                "--s3-bucket", "test",
                "--s3-access-key", "key",
                "--s3-secret-key", "secret",
                "--output", report_path,
            ])
            assert result == 0
            assert os.path.exists(report_path)
            with open(report_path) as f:
                report = json.load(f)
            assert "totals" in report
            assert "jobs" in report


# ===========================================================================
# C-16: Dry-run output clarity tests
# ===========================================================================


class TestDryRunClarity:
    """Validate dry-run output labels and summary fields."""

    def test_dry_run_projected_labels(self, nfs_tree, capsys):
        """Dry-run print output should contain '(projected)' labels."""
        with patch("migrate_nfs_to_s3.S3Backend") as MockS3, \
             patch("migrate_nfs_to_s3.check_connectivity", return_value=(True, [])):
            mock_s3 = MagicMock()
            mock_s3.exists.return_value = False
            MockS3.return_value = mock_s3

            result = main([
                "--nfs-root", nfs_tree,
                "--s3-endpoint", "http://localhost:9000",
                "--s3-bucket", "test",
                "--s3-access-key", "key",
                "--s3-secret-key", "secret",
            ])
            assert result == 0
            captured = capsys.readouterr()
            assert "(projected)" in captured.out
            assert "DRY RUN" in captured.out

    def test_run_migration_dry_run_field_false(self, nfs_tree):
        """run_migration with execute=False should have dry_run=True."""
        with patch("migrate_nfs_to_s3.S3Backend") as MockS3:
            mock_s3 = MagicMock()
            mock_s3.exists.return_value = False
            MockS3.return_value = mock_s3

            summary = run_migration(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
                execute=False,
            )
            assert summary["dry_run"] is True

    def test_run_migration_execute_dry_run_false(self, nfs_tree, mock_s3):
        """run_migration with execute=True should have dry_run=False."""
        with patch("migrate_nfs_to_s3.S3Backend") as MockS3:
            MockS3.return_value = mock_s3

            summary = run_migration(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
                execute=True,
            )
            assert summary["dry_run"] is False

    def test_migrate_job_dry_run_field(self, nfs_tree, mock_s3):
        """migrate_job stats should include dry_run field."""
        nfs = NFSBackend(root=nfs_tree)
        stats_dry = migrate_job("job-001", nfs, mock_s3, execute=False)
        assert stats_dry["dry_run"] is True

        stats_exec = migrate_job("job-001", nfs, mock_s3, execute=True)
        assert stats_exec["dry_run"] is False


# ===========================================================================
# C-17: Parallel upload tests
# ===========================================================================


class TestParallelUpload:
    """Validate concurrent file upload support."""

    def test_parallel_upload_produces_correct_stats(self, nfs_tree, mock_s3):
        """parallel=2 should upload all files and produce same stats."""
        nfs = NFSBackend(root=nfs_tree)
        stats = migrate_job("job-001", nfs, mock_s3, execute=True, parallel=2)
        assert stats["files_found"] == 2
        assert stats["files_uploaded"] == 2
        assert stats["files_verified"] == 2
        assert stats["bytes_uploaded"] > 0
        assert stats["errors"] == []

    def test_parallel_dry_run(self, nfs_tree, mock_s3):
        """parallel=2 dry run should count files without uploading."""
        nfs = NFSBackend(root=nfs_tree)
        stats = migrate_job("job-001", nfs, mock_s3, execute=False, parallel=2)
        assert stats["files_to_upload"] == 2
        assert stats["files_to_verify"] == 2
        assert stats["dry_run"] is True
        mock_s3.upload_file.assert_not_called()

    def test_parallel_captures_errors(self, nfs_tree, mock_s3):
        """Errors in parallel threads should be captured in stats."""
        nfs = NFSBackend(root=nfs_tree)
        mock_s3.upload_file.side_effect = RuntimeError("upload boom")
        stats = migrate_job("job-001", nfs, mock_s3, execute=True, parallel=2)
        assert len(stats["errors"]) == 2
        assert stats["files_uploaded"] == 0

    def test_parallel_cli_flag_zero(self, nfs_tree, capsys):
        """--parallel 0 should return error."""
        result = main([
            "--nfs-root", nfs_tree,
            "--s3-endpoint", "http://localhost:9000",
            "--s3-bucket", "test",
            "--s3-access-key", "key",
            "--s3-secret-key", "secret",
            "--parallel", "0",
        ])
        assert result == 1

    def test_parallel_cli_flag_max(self, nfs_tree, capsys):
        """--parallel 33 should return error."""
        result = main([
            "--nfs-root", nfs_tree,
            "--s3-endpoint", "http://localhost:9000",
            "--s3-bucket", "test",
            "--s3-access-key", "key",
            "--s3-secret-key", "secret",
            "--parallel", "33",
        ])
        assert result == 1

    def test_parallel_default_is_one(self, nfs_tree, mock_s3):
        """Default parallel=1 should work (sequential)."""
        nfs = NFSBackend(root=nfs_tree)
        stats = migrate_job("job-001", nfs, mock_s3, execute=True, parallel=1)
        assert stats["files_uploaded"] == 2
        assert stats["errors"] == []


# ===========================================================================
# C-18: Progress persistence tests
# ===========================================================================


class TestProgressPersistence:
    """Validate state-file based progress persistence."""

    def test_state_file_created_on_execute(self, nfs_tree, mock_s3, tmp_path):
        """State file should be created during execute migration."""
        state_path = str(tmp_path / "state.json")
        nfs = NFSBackend(root=nfs_tree)
        migrate_job(
            "job-001", nfs, mock_s3,
            execute=True, state_file=state_path,
        )
        assert os.path.exists(state_path)
        with open(state_path) as f:
            state = json.load(f)
        assert "jobs" in state
        assert "job-001" in state["jobs"]
        assert len(state["jobs"]["job-001"]["verified_files"]) == 2
        assert state["jobs"]["job-001"]["status"] == "completed"

    def test_state_file_resume_skips_verified(self, nfs_tree, mock_s3, tmp_path):
        """Resume with state file should skip already-verified files."""
        state_path = str(tmp_path / "state.json")
        nfs = NFSBackend(root=nfs_tree)

        # First run: migrate job
        stats1 = migrate_job(
            "job-001", nfs, mock_s3,
            execute=True, state_file=state_path,
        )
        assert stats1["files_verified"] == 2

        # Reset mock
        mock_s3.upload_file.reset_mock()

        # Second run with resume: should skip all files via state
        stats2 = migrate_job(
            "job-001", nfs, mock_s3,
            execute=True, resume=True, state_file=state_path,
        )
        assert stats2["files_skipped"] == 2
        assert stats2["files_uploaded"] == 0
        mock_s3.upload_file.assert_not_called()

    def test_state_file_run_migration(self, nfs_tree, tmp_path):
        """run_migration with state_file should create state file."""
        state_path = str(tmp_path / "migration_state.json")
        with patch("migrate_nfs_to_s3.S3Backend") as MockS3:
            mock_s3 = MagicMock()
            mock_s3.exists.return_value = False
            MockS3.return_value = mock_s3

            run_migration(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
                execute=False,
                state_file=state_path,
            )
        assert os.path.exists(state_path)
        with open(state_path) as f:
            state = json.load(f)
        assert state["nfs_root"] == nfs_tree
        assert state["s3_bucket"] == "test"
        assert "started_at" in state

    def test_state_file_atomic_write(self, tmp_path):
        """Atomic write should not leave .tmp file on success."""
        state_path = str(tmp_path / "state.json")
        _save_state_file(state_path, {"test": True})
        assert os.path.exists(state_path)
        assert not os.path.exists(state_path + ".tmp")
        with open(state_path) as f:
            data = json.load(f)
        assert data == {"test": True}

    def test_load_state_file_missing(self, tmp_path):
        """Loading non-existent state file should return None."""
        result = _load_state_file(str(tmp_path / "missing.json"))
        assert result is None

    def test_load_state_file_valid(self, tmp_path):
        """Loading valid state file should return dict."""
        state_path = str(tmp_path / "state.json")
        _save_state_file(state_path, {"jobs": {"j1": {"status": "completed"}}})
        result = _load_state_file(state_path)
        assert result is not None
        assert result["jobs"]["j1"]["status"] == "completed"

    def test_state_file_records_failed_status(self, nfs_tree, tmp_path):
        """State file should record failed status when errors occur."""
        state_path = str(tmp_path / "state.json")
        nfs = NFSBackend(root=nfs_tree)

        # Mock bad download (checksum mismatch)
        bad_s3 = MagicMock()
        bad_s3.exists.return_value = False

        def bad_upload(local_path, key):
            return f"s3://bucket/{key}"

        def bad_download(key, local_path):
            with open(local_path, "wb") as f:
                f.write(b"corrupted-data")
            return local_path

        bad_s3.upload_file.side_effect = bad_upload
        bad_s3.download_file.side_effect = bad_download

        migrate_job(
            "job-001", nfs, bad_s3,
            execute=True, state_file=state_path,
        )
        with open(state_path) as f:
            state = json.load(f)
        assert state["jobs"]["job-001"]["status"] == "failed"


# ===========================================================================
# C-19: Database backend update tests
# ===========================================================================


class TestDatabaseUpdate:
    """Validate --update-db database backend updates."""

    def test_update_db_requires_execute(self, nfs_tree):
        """--update-db without --execute should return error."""
        result = main([
            "--nfs-root", nfs_tree,
            "--s3-endpoint", "http://localhost:9000",
            "--s3-bucket", "test",
            "--s3-access-key", "key",
            "--s3-secret-key", "secret",
            "--update-db",
        ])
        assert result == 1

    def test_update_job_db_no_psycopg2(self):
        """_update_job_db should return False when psycopg2 is not available."""
        with patch.object(_mod, "psycopg2", None):
            result = _update_job_db("job-001")
            assert result is False

    def test_update_job_db_success(self):
        """_update_job_db should return True on successful DB update."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn

        with patch.object(_mod, "psycopg2", mock_psycopg2):
            result = _update_job_db(
                "job-001", db_url="postgresql://test:test@localhost/test",
            )
        assert result is True
        mock_psycopg2.connect.assert_called_once_with(
            "postgresql://test:test@localhost/test",
        )
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_update_job_db_connection_failure(self):
        """_update_job_db should return False on connection failure."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.side_effect = Exception("Connection refused")

        with patch.object(_mod, "psycopg2", mock_psycopg2), \
             patch.dict(os.environ, {"DATABASE_URL": "postgresql://bad"}):
            result = _update_job_db("job-001")
            assert result is False

    def test_update_job_db_env_vars(self):
        """_update_job_db should build URL from individual env vars."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn

        env = {
            "DATABASE_URL": "",
            "POSTGRES_HOST": "dbhost",
            "POSTGRES_PORT": "5433",
            "POSTGRES_USER": "myuser",
            "POSTGRES_PASSWORD": "mypass",
            "POSTGRES_DB": "mydb",
        }

        with patch.object(_mod, "psycopg2", mock_psycopg2), \
             patch.dict(os.environ, env):
            result = _update_job_db("job-001")
        assert result is True
        mock_psycopg2.connect.assert_called_once_with(
            "postgresql://myuser:mypass@dbhost:5433/mydb",
        )

    def test_update_job_db_url_override(self):
        """db_url parameter should override env vars."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn

        with patch.object(_mod, "psycopg2", mock_psycopg2):
            result = _update_job_db(
                "job-001", db_url="postgresql://override:5432/db",
            )
        assert result is True
        mock_psycopg2.connect.assert_called_once_with(
            "postgresql://override:5432/db",
        )
