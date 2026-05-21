"""Tests for NFS-to-S3 migration dry-run clarity, sample mode, and connectivity pre-check.

Complements the existing ``test_migrate_nfs_to_s3.py`` suite with targeted
coverage for:

- Dry-run output labels (``files_to_upload`` instead of ``files_uploaded``)
- ``--sample`` flag limiting file processing per job
- ``check_connectivity()`` pre-flight checks
- Exit codes for failure modes
"""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the script under test via importlib (it lives in scripts/)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_mod = importlib.import_module("migrate_nfs_to_s3")
sha256_file = _mod.sha256_file
discover_jobs = _mod.discover_jobs
migrate_job = _mod.migrate_job
run_migration = _mod.run_migration
run_validation = _mod.run_validation
check_connectivity = _mod.check_connectivity
main = _mod.main
NFSBackend = _mod.NFSBackend


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def nfs_tree(tmp_path):
    """Create a mock NFS directory structure with two jobs."""
    nfs_root = tmp_path / "nfs"
    jobs_dir = nfs_root / "jobs"

    # Job 1: three files
    job1_pdf = jobs_dir / "job-001" / "output" / "EXPORT" / "PDF"
    job1_pdf.mkdir(parents=True)
    (job1_pdf / "doc.pdf").write_bytes(b"fake-pdf-content-job1")

    job1_text = jobs_dir / "job-001" / "output" / "EXPORT" / "TEXT"
    job1_text.mkdir(parents=True)
    (job1_text / "doc.txt").write_text("extracted text job1")

    job1_ner = jobs_dir / "job-001" / "output" / "EXPORT" / "NER"
    job1_ner.mkdir(parents=True)
    (job1_ner / "doc.ner.json").write_text('{"entities": []}')

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
# Dry-run output label tests
# ===========================================================================


class TestDryRunLabels:
    """Verify dry-run mode uses 'to_upload' / 'to_verify' counter names."""

    def test_migrate_job_dry_run_uses_to_upload_keys(self, nfs_tree, mock_s3):
        """migrate_job() in dry-run must use files_to_upload, not files_uploaded."""
        nfs = NFSBackend(root=nfs_tree)
        stats = migrate_job("job-001", nfs, mock_s3, execute=False)

        # Dry-run keys present
        assert "files_to_upload" in stats
        assert "files_to_verify" in stats
        assert "bytes_to_upload" in stats
        assert stats["dry_run"] is True

        # Live-mode keys absent
        assert "files_uploaded" not in stats
        assert "files_verified" not in stats
        assert "bytes_uploaded" not in stats

        # Values are correct
        assert stats["files_to_upload"] == 3
        assert stats["files_to_verify"] == 3
        assert stats["bytes_to_upload"] > 0
        assert stats["errors"] == []

        # No actual upload happened
        mock_s3.upload_file.assert_not_called()

    def test_migrate_job_execute_uses_uploaded_keys(self, nfs_tree, mock_s3):
        """migrate_job() in execute mode must use files_uploaded, not files_to_upload."""
        nfs = NFSBackend(root=nfs_tree)
        stats = migrate_job("job-001", nfs, mock_s3, execute=True)

        assert "files_uploaded" in stats
        assert "files_verified" in stats
        assert "bytes_uploaded" in stats
        assert stats["dry_run"] is False

        assert "files_to_upload" not in stats
        assert "files_to_verify" not in stats

    def test_run_migration_dry_run_totals_use_to_upload(self, nfs_tree):
        """run_migration() in dry-run produces totals with files_to_upload."""
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

            assert t["mode"] == "DRY RUN"
            assert "files_to_upload" in t
            assert "files_to_verify" in t
            assert "bytes_to_upload" in t
            assert "files_uploaded" not in t
            assert "files_verified" not in t
            assert t["files_to_upload"] == 4  # 3 in job-001, 1 in job-002

    def test_cli_dry_run_output_no_misleading_uploaded(self, nfs_tree, capsys):
        """CLI dry-run output must say 'to upload' not 'uploaded'."""
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
            assert "Files to upload (projected):" in captured.out
            assert "Files to verify (projected):" in captured.out
            assert "Bytes to transfer (projected):" in captured.out
            assert "No files will be transferred" in captured.out
            assert "No files were transferred" in captured.out

            # Ensure misleading labels are NOT present
            assert "Files uploaded:" not in captured.out
            assert "Files verified:" not in captured.out
            assert "Bytes transferred:" not in captured.out

    def test_cli_dry_run_banner_present(self, nfs_tree, capsys):
        """CLI dry-run must show start and end banners."""
        with patch("migrate_nfs_to_s3.S3Backend") as MockS3, \
             patch("migrate_nfs_to_s3.check_connectivity", return_value=(True, [])):
            mock_s3 = MagicMock()
            mock_s3.exists.return_value = False
            MockS3.return_value = mock_s3

            main([
                "--nfs-root", nfs_tree,
                "--s3-endpoint", "http://localhost:9000",
                "--s3-bucket", "test",
                "--s3-access-key", "key",
                "--s3-secret-key", "secret",
            ])
            captured = capsys.readouterr()
            assert "DRY RUN MODE" in captured.out
            assert "DRY RUN COMPLETE" in captured.out


# ===========================================================================
# Sample mode tests
# ===========================================================================


class TestSampleMode:
    """Verify --sample limits file processing per job in validate mode."""

    def test_sample_limits_files_per_job(self, nfs_tree):
        """run_validation(sample=1) should inspect at most 1 file per job."""
        with patch("migrate_nfs_to_s3._check_s3_connectivity", return_value="ok"):
            report = run_validation(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
                sample=1,
            )

        # job-001 has 3 files, job-002 has 1 file -> sampled 1+1 = 2
        assert report.total_files == 2
        assert len(report.file_manifest) == 2

        # Should have a warning about sampling
        sampling_warnings = [w for w in report.warnings if "Sampling active" in w]
        assert len(sampling_warnings) == 1
        assert "skipped 2" in sampling_warnings[0]  # 3-1 = 2 skipped from job-001

    def test_sample_none_processes_all_files(self, nfs_tree):
        """run_validation(sample=None) should process all files."""
        with patch("migrate_nfs_to_s3._check_s3_connectivity", return_value="ok"):
            report = run_validation(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
                sample=None,
            )

        assert report.total_files == 4  # 3 + 1
        sampling_warnings = [w for w in report.warnings if "Sampling active" in w]
        assert len(sampling_warnings) == 0

    def test_sample_larger_than_files_processes_all(self, nfs_tree):
        """sample=100 when job has 3 files should process all 3."""
        with patch("migrate_nfs_to_s3._check_s3_connectivity", return_value="ok"):
            report = run_validation(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
                sample=100,
            )

        assert report.total_files == 4
        sampling_warnings = [w for w in report.warnings if "Sampling active" in w]
        assert len(sampling_warnings) == 0

    def test_sample_without_validate_fails(self, nfs_tree):
        """--sample without --validate should return error exit code."""
        result = main([
            "--nfs-root", nfs_tree,
            "--s3-endpoint", "http://localhost:9000",
            "--s3-bucket", "test",
            "--s3-access-key", "key",
            "--s3-secret-key", "secret",
            "--sample", "5",
        ])
        assert result == 1


# ===========================================================================
# Connectivity pre-check tests
# ===========================================================================


class TestCheckConnectivity:
    """Verify check_connectivity() pre-flight checks."""

    def test_both_ok(self, nfs_tree):
        """Both NFS and S3 reachable should return (True, [])."""
        with patch("migrate_nfs_to_s3._check_s3_connectivity", return_value="ok"):
            ok, errors = check_connectivity(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
            )
        assert ok is True
        assert errors == []

    def test_nfs_missing(self, tmp_path):
        """Non-existent NFS root should report error."""
        with patch("migrate_nfs_to_s3._check_s3_connectivity", return_value="ok"):
            ok, errors = check_connectivity(
                nfs_root=str(tmp_path / "nonexistent"),
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
            )
        assert ok is False
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_s3_unreachable(self, nfs_tree):
        """S3 connectivity failure should report error."""
        with patch("migrate_nfs_to_s3._check_s3_connectivity", return_value="unreachable"):
            ok, errors = check_connectivity(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
            )
        assert ok is False
        assert len(errors) == 1
        assert "unreachable" in errors[0]

    def test_s3_auth_failed(self, nfs_tree):
        """S3 auth failure should report error."""
        with patch("migrate_nfs_to_s3._check_s3_connectivity", return_value="auth_failed"):
            ok, errors = check_connectivity(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="bad",
                s3_secret_key="bad",
            )
        assert ok is False
        assert len(errors) == 1
        assert "auth_failed" in errors[0]

    def test_s3_bucket_not_found(self, nfs_tree):
        """S3 bucket not found should report error."""
        with patch("migrate_nfs_to_s3._check_s3_connectivity", return_value="bucket_not_found"):
            ok, errors = check_connectivity(
                nfs_root=nfs_tree,
                s3_endpoint="http://localhost:9000",
                s3_bucket="no-such-bucket",
                s3_access_key="key",
                s3_secret_key="secret",
            )
        assert ok is False
        assert "bucket_not_found" in errors[0]

    def test_both_fail(self, tmp_path):
        """Both NFS and S3 failures should report two errors."""
        with patch("migrate_nfs_to_s3._check_s3_connectivity", return_value="unreachable"):
            ok, errors = check_connectivity(
                nfs_root=str(tmp_path / "nonexistent"),
                s3_endpoint="http://localhost:9000",
                s3_bucket="test",
                s3_access_key="key",
                s3_secret_key="secret",
            )
        assert ok is False
        assert len(errors) == 2


# ===========================================================================
# Exit code tests
# ===========================================================================


class TestExitCodes:
    """Verify exit codes for various failure modes."""

    def test_missing_credentials_returns_1(self, nfs_tree):
        """Missing S3 credentials should return exit code 1."""
        # Clear env vars so credentials are empty
        with patch.dict(os.environ, {}, clear=True):
            result = main([
                "--nfs-root", nfs_tree,
                "--s3-endpoint", "http://localhost:9000",
                "--s3-bucket", "test",
            ])
        assert result == 1

    def test_nonexistent_nfs_root_returns_1(self, tmp_path):
        """Non-existent NFS root should return exit code 1."""
        result = main([
            "--nfs-root", str(tmp_path / "nonexistent"),
            "--s3-endpoint", "http://localhost:9000",
            "--s3-bucket", "test",
            "--s3-access-key", "key",
            "--s3-secret-key", "secret",
        ])
        assert result == 1

    def test_invalid_job_id_returns_1(self, nfs_tree):
        """Job ID with path traversal should return exit code 1."""
        result = main([
            "--nfs-root", nfs_tree,
            "--s3-endpoint", "http://localhost:9000",
            "--s3-bucket", "test",
            "--s3-access-key", "key",
            "--s3-secret-key", "secret",
            "--job-ids", "../escape",
        ])
        assert result == 1

    def test_connectivity_failure_returns_1(self, nfs_tree):
        """Connectivity pre-check failure should return exit code 1."""
        with patch("migrate_nfs_to_s3.check_connectivity",
                    return_value=(False, ["S3 connectivity failed: unreachable"])):
            result = main([
                "--nfs-root", nfs_tree,
                "--s3-endpoint", "http://localhost:9000",
                "--s3-bucket", "test",
                "--s3-access-key", "key",
                "--s3-secret-key", "secret",
            ])
        assert result == 1

    def test_output_report_without_validate_returns_1(self, nfs_tree):
        """--output-report without --validate should return exit code 1."""
        result = main([
            "--nfs-root", nfs_tree,
            "--s3-endpoint", "http://localhost:9000",
            "--s3-bucket", "test",
            "--s3-access-key", "key",
            "--s3-secret-key", "secret",
            "--output-report", "report.json",
        ])
        assert result == 1

    def test_delete_nfs_without_execute_returns_1(self, nfs_tree):
        """--delete-nfs without --execute should return exit code 1."""
        result = main([
            "--nfs-root", nfs_tree,
            "--s3-endpoint", "http://localhost:9000",
            "--s3-bucket", "test",
            "--s3-access-key", "key",
            "--s3-secret-key", "secret",
            "--delete-nfs",
        ])
        assert result == 1
