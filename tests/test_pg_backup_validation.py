"""Tests for PostgreSQL backup validation framework.

Run with: python -m pytest tests/test_pg_backup_validation.py -v
"""

import datetime
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from pg_backup_validation import (
    BACKUP_FILE_PATTERN,
    DEFAULT_BACKUP_DIR,
    DEFAULT_MAX_AGE_HOURS,
    DEFAULT_MIN_SIZE_BYTES,
    EXPECTED_TABLES,
    BackupFileInfo,
    CheckResult,
    ValidationReport,
    build_parser,
    check_config,
    find_backup_files,
    generate_report,
    main,
    mask_database_url,
    parse_backup_timestamp,
    parse_database_url,
    run_command,
    trigger_backup,
    verify_backup_file,
    verify_backups,
    write_report_json,
    write_report_markdown,
)

# ===========================================================================
# Test helpers
# ===========================================================================

SAMPLE_DB_URL = "postgres://ocr:secret123@localhost:5432/ocr_coordinator"
SAMPLE_DB_URL_NO_PASS = "postgres://ocr@localhost:5432/ocr_coordinator"


def _create_fake_backup(tmpdir: str, name: str, size: int = 2048) -> str:
    """Create a fake backup file with given name and size."""
    path = os.path.join(tmpdir, name)
    with open(path, "wb") as f:
        f.write(b"\x00" * size)
    return path


# ===========================================================================
# Tests: mask_database_url
# ===========================================================================


class TestMaskDatabaseUrl:
    """Tests for password masking in database URLs."""

    def test_masks_password(self):
        result = mask_database_url(SAMPLE_DB_URL)
        assert "secret123" not in result
        assert "****" in result

    def test_no_password(self):
        result = mask_database_url(SAMPLE_DB_URL_NO_PASS)
        assert result == SAMPLE_DB_URL_NO_PASS

    def test_empty_string(self):
        assert mask_database_url("") == ""

    def test_none_like_input(self):
        # Should not crash on empty input
        assert mask_database_url("") == ""


# ===========================================================================
# Tests: parse_database_url
# ===========================================================================


class TestParseDatabaseUrl:
    """Tests for database URL parsing."""

    def test_full_url(self):
        parts = parse_database_url(SAMPLE_DB_URL)
        assert parts["host"] == "localhost"
        assert parts["port"] == "5432"
        assert parts["user"] == "ocr"
        assert parts["password"] == "secret123"
        assert parts["dbname"] == "ocr_coordinator"

    def test_custom_port(self):
        parts = parse_database_url("postgres://ocr:pass@db.host:5433/mydb")
        assert parts["host"] == "db.host"
        assert parts["port"] == "5433"
        assert parts["dbname"] == "mydb"

    def test_default_port(self):
        parts = parse_database_url("postgres://ocr:pass@host/db")
        assert parts["port"] == "5432"

    def test_no_password(self):
        parts = parse_database_url(SAMPLE_DB_URL_NO_PASS)
        assert parts["password"] == ""


# ===========================================================================
# Tests: parse_backup_timestamp
# ===========================================================================


class TestParseBackupTimestamp:
    """Tests for extracting timestamps from backup filenames."""

    def test_valid_filename(self):
        ts = parse_backup_timestamp("ocr-coordinator-20260315-020000.sql.gz")
        assert ts is not None
        assert ts.year == 2026
        assert ts.month == 3
        assert ts.day == 15
        assert ts.hour == 2

    def test_invalid_filename(self):
        assert parse_backup_timestamp("random-file.sql.gz") is None

    def test_partial_match(self):
        assert parse_backup_timestamp("ocr-coordinator-bad-date.sql.gz") is None

    def test_wrong_extension(self):
        assert parse_backup_timestamp("ocr-coordinator-20260315-020000.tar.gz") is None

    def test_path_with_directory(self):
        ts = parse_backup_timestamp("/backups/ocr-coordinator-20260101-120000.sql.gz")
        assert ts is not None
        assert ts.month == 1


# ===========================================================================
# Tests: BACKUP_FILE_PATTERN
# ===========================================================================


class TestBackupFilePattern:
    """Tests for the backup file regex pattern."""

    def test_valid_pattern_matches(self):
        assert BACKUP_FILE_PATTERN.search("ocr-coordinator-20260315-020000.sql.gz")

    def test_extracts_timestamp(self):
        match = BACKUP_FILE_PATTERN.search("ocr-coordinator-20260315-143025.sql.gz")
        assert match is not None
        assert match.group(1) == "20260315-143025"

    def test_no_match_wrong_prefix(self):
        assert BACKUP_FILE_PATTERN.search("my-backup-20260315-020000.sql.gz") is None


# ===========================================================================
# Tests: find_backup_files
# ===========================================================================


class TestFindBackupFiles:
    """Tests for finding backup files in a directory."""

    def test_finds_matching_files(self, tmp_path):
        _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz")
        _create_fake_backup(str(tmp_path), "ocr-coordinator-20260314-020000.sql.gz")
        files = find_backup_files(str(tmp_path))
        assert len(files) == 2

    def test_ignores_non_matching_files(self, tmp_path):
        _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz")
        _create_fake_backup(str(tmp_path), "other-file.txt")
        files = find_backup_files(str(tmp_path))
        assert len(files) == 1

    def test_empty_directory(self, tmp_path):
        files = find_backup_files(str(tmp_path))
        assert len(files) == 0

    def test_nonexistent_directory(self):
        files = find_backup_files("/nonexistent/path/backup")
        assert len(files) == 0

    def test_sorted_newest_first(self, tmp_path):
        import time
        _create_fake_backup(str(tmp_path), "ocr-coordinator-20260314-020000.sql.gz")
        time.sleep(0.05)
        _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz")
        files = find_backup_files(str(tmp_path))
        # Newest (most recently modified) should be first
        assert files[0].name == "ocr-coordinator-20260315-020000.sql.gz"


# ===========================================================================
# Tests: run_command
# ===========================================================================


class TestRunCommand:
    """Tests for subprocess command execution."""

    def test_successful_command(self):
        rc, stdout, stderr = run_command(["python", "--version"])
        assert rc == 0
        # Python version goes to stdout or stderr depending on version
        assert "Python" in stdout or "Python" in stderr

    def test_failed_command(self):
        rc, stdout, stderr = run_command(["python", "-c", "import sys; sys.exit(1)"])
        assert rc == 1

    def test_nonexistent_command(self):
        rc, stdout, stderr = run_command(["nonexistent_binary_xyz"])
        assert rc == -1
        assert "not found" in stderr.lower() or "nonexistent" in stderr.lower()

    def test_timeout(self):
        # Use a very short timeout
        rc, stdout, stderr = run_command(
            ["python", "-c", "import time; time.sleep(10)"],
            timeout=1,
        )
        assert rc == -1
        assert "timed out" in stderr.lower()

    def test_custom_env(self):
        rc, stdout, stderr = run_command(
            ["python", "-c", "import os; print(os.environ.get('TEST_VAR', ''))"],
            env={"TEST_VAR": "hello"},
        )
        assert rc == 0
        assert "hello" in stdout


# ===========================================================================
# Tests: check_config
# ===========================================================================


class TestCheckConfig:
    """Tests for configuration checking."""

    @patch("pg_backup_validation.run_command")
    def test_check_with_valid_url(self, mock_run):
        mock_run.return_value = (0, "accepting connections", "")
        results = check_config(database_url=SAMPLE_DB_URL, backup_dir="/tmp")
        url_check = next(r for r in results if r.name == "database_url_format")
        assert url_check.passed is True

    def test_check_without_url(self):
        results = check_config(database_url="")
        url_check = next(r for r in results if r.name == "database_url_format")
        assert url_check.passed is False

    @patch("pg_backup_validation.run_command")
    def test_check_database_connectivity_pass(self, mock_run):
        mock_run.return_value = (0, "localhost:5432 - accepting connections", "")
        results = check_config(database_url=SAMPLE_DB_URL, backup_dir="/tmp")
        conn_check = next(r for r in results if r.name == "database_connectivity")
        assert conn_check.passed is True

    @patch("pg_backup_validation.run_command")
    def test_check_database_connectivity_fail(self, mock_run):
        mock_run.return_value = (1, "", "connection refused")
        results = check_config(database_url=SAMPLE_DB_URL, backup_dir="/tmp")
        conn_check = next(r for r in results if r.name == "database_connectivity")
        assert conn_check.passed is False

    def test_check_backup_dir_exists(self, tmp_path):
        with patch("pg_backup_validation.run_command", return_value=(0, "ok", "")):
            results = check_config(database_url=SAMPLE_DB_URL, backup_dir=str(tmp_path))
        dir_check = next(r for r in results if r.name == "backup_dir_exists")
        assert dir_check.passed is True

    def test_check_backup_dir_missing(self):
        with patch("pg_backup_validation.run_command", return_value=(0, "ok", "")):
            results = check_config(
                database_url=SAMPLE_DB_URL,
                backup_dir="/nonexistent/pg_backups",
            )
        dir_check = next(r for r in results if r.name == "backup_dir_exists")
        assert dir_check.passed is False

    @patch("pg_backup_validation.run_command")
    def test_check_pg_dump_available(self, mock_run):
        def side_effect(cmd, **kwargs):
            if cmd[0] == "pg_dump":
                return (0, "pg_dump (PostgreSQL) 16.1", "")
            return (0, "ok", "")
        mock_run.side_effect = side_effect
        results = check_config(database_url=SAMPLE_DB_URL, backup_dir="/tmp")
        dump_check = next(r for r in results if r.name == "pg_dump_available")
        assert dump_check.passed is True

    @patch("pg_backup_validation.run_command")
    def test_check_pg_dump_missing(self, mock_run):
        def side_effect(cmd, **kwargs):
            if cmd[0] == "pg_dump":
                return (-1, "", "Command not found: pg_dump")
            return (0, "ok", "")
        mock_run.side_effect = side_effect
        results = check_config(database_url=SAMPLE_DB_URL, backup_dir="/tmp")
        dump_check = next(r for r in results if r.name == "pg_dump_available")
        assert dump_check.passed is False

    def test_check_helm_values_enabled(self):
        helm = {
            "postgresql": {
                "backup": {
                    "enabled": True,
                    "schedule": "0 2 * * *",
                    "retentionCount": 7,
                }
            }
        }
        with patch("pg_backup_validation.run_command", return_value=(0, "ok", "")):
            results = check_config(
                database_url=SAMPLE_DB_URL,
                backup_dir="/tmp",
                helm_values=helm,
            )
        helm_check = next(r for r in results if r.name == "helm_backup_enabled")
        assert helm_check.passed is True
        schedule_check = next(r for r in results if r.name == "helm_backup_schedule")
        assert schedule_check.passed is True
        retention_check = next(r for r in results if r.name == "helm_backup_retention")
        assert retention_check.passed is True

    def test_check_helm_values_disabled(self):
        helm = {
            "postgresql": {
                "backup": {
                    "enabled": False,
                }
            }
        }
        with patch("pg_backup_validation.run_command", return_value=(0, "ok", "")):
            results = check_config(
                database_url=SAMPLE_DB_URL,
                backup_dir="/tmp",
                helm_values=helm,
            )
        helm_check = next(r for r in results if r.name == "helm_backup_enabled")
        assert helm_check.passed is False

    def test_check_helm_invalid_schedule(self):
        helm = {
            "postgresql": {
                "backup": {
                    "enabled": True,
                    "schedule": "bad schedule",
                    "retentionCount": 7,
                }
            }
        }
        with patch("pg_backup_validation.run_command", return_value=(0, "ok", "")):
            results = check_config(
                database_url=SAMPLE_DB_URL,
                backup_dir="/tmp",
                helm_values=helm,
            )
        schedule_check = next(r for r in results if r.name == "helm_backup_schedule")
        assert schedule_check.passed is False


# ===========================================================================
# Tests: trigger_backup
# ===========================================================================


class TestTriggerBackup:
    """Tests for manual backup triggering."""

    @patch("pg_backup_validation.run_command")
    def test_dry_run(self, mock_run):
        success, info, errors = trigger_backup(
            database_url=SAMPLE_DB_URL,
            backup_dir="/tmp/backups",
            dry_run=True,
        )
        assert success is True
        assert info is None
        assert errors == []
        mock_run.assert_not_called()

    @patch("pg_backup_validation.run_command")
    def test_backup_pg_dump_fails(self, mock_run, tmp_path):
        mock_run.return_value = (1, "", "pg_dump: error: connection refused")
        success, info, errors = trigger_backup(
            database_url=SAMPLE_DB_URL,
            backup_dir=str(tmp_path),
        )
        assert success is False
        assert len(errors) > 0
        assert "pg_dump failed" in errors[0]

    @patch("pg_backup_validation.run_command")
    def test_backup_succeeds(self, mock_run, tmp_path):
        backup_dir = str(tmp_path)

        def side_effect(cmd, **kwargs):
            if cmd[0] == "pg_dump":
                # Simulate pg_dump creating a file
                for i, arg in enumerate(cmd):
                    if arg == "-f" and i + 1 < len(cmd):
                        # Create a fake backup file
                        with open(cmd[i + 1], "wb") as f:
                            f.write(b"\x00" * 4096)
                        break
                return (0, "", "")
            if cmd[0] == "pg_restore":
                return (0, "TABLE public.jobs_job\nTABLE public.jobs_worker\n", "")
            return (0, "", "")

        mock_run.side_effect = side_effect
        success, info, errors = trigger_backup(
            database_url=SAMPLE_DB_URL,
            backup_dir=backup_dir,
        )
        assert success is True
        assert info is not None
        assert info.size_bytes >= DEFAULT_MIN_SIZE_BYTES
        assert info.pg_restore_list_ok is True

    @patch("pg_backup_validation.run_command")
    def test_backup_too_small(self, mock_run, tmp_path):
        backup_dir = str(tmp_path)

        def side_effect(cmd, **kwargs):
            if cmd[0] == "pg_dump":
                for i, arg in enumerate(cmd):
                    if arg == "-f" and i + 1 < len(cmd):
                        # Create a tiny file (truncated)
                        with open(cmd[i + 1], "wb") as f:
                            f.write(b"\x00" * 10)
                        break
                return (0, "", "")
            return (0, "", "")

        mock_run.side_effect = side_effect
        success, info, errors = trigger_backup(
            database_url=SAMPLE_DB_URL,
            backup_dir=backup_dir,
        )
        assert success is False
        assert any("too small" in e for e in errors)


# ===========================================================================
# Tests: verify_backup_file
# ===========================================================================


class TestVerifyBackupFile:
    """Tests for individual backup file verification."""

    @patch("pg_backup_validation.run_command")
    def test_verify_valid_file(self, mock_run, tmp_path):
        mock_run.return_value = (0, "TABLE public.jobs_job\nTABLE public.jobs_worker\n", "")
        fp = _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz", 4096)
        info = verify_backup_file(fp, max_age_hours=9999)
        assert info.pg_restore_list_ok is True
        assert info.size_bytes == 4096
        assert info.is_valid_format is True
        assert info.table_count == 2
        assert len(info.errors) == 0

    def test_verify_nonexistent_file(self):
        info = verify_backup_file("/nonexistent/backup.sql.gz")
        assert info.size_bytes == 0
        assert len(info.errors) > 0
        assert "not found" in info.errors[0].lower()

    @patch("pg_backup_validation.run_command")
    def test_verify_too_small(self, mock_run, tmp_path):
        mock_run.return_value = (0, "", "")
        fp = _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz", 100)
        info = verify_backup_file(fp, min_size_bytes=1024)
        assert any("too small" in e.lower() for e in info.errors)

    @patch("pg_backup_validation.run_command")
    def test_verify_too_old(self, mock_run, tmp_path):
        mock_run.return_value = (0, "", "")
        fp = _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz", 4096)
        # Set file mtime to 48 hours ago
        old_time = datetime.datetime.now().timestamp() - (48 * 3600)
        os.utime(fp, (old_time, old_time))
        info = verify_backup_file(fp, max_age_hours=24)
        assert any("too old" in e.lower() for e in info.errors)

    @patch("pg_backup_validation.run_command")
    def test_verify_pg_restore_fails(self, mock_run, tmp_path):
        mock_run.return_value = (1, "", "pg_restore: error: invalid file")
        fp = _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz", 4096)
        info = verify_backup_file(fp, max_age_hours=9999)
        assert info.pg_restore_list_ok is False
        assert any("pg_restore" in e for e in info.errors)

    @patch("pg_backup_validation.run_command")
    def test_verify_invalid_filename_format(self, mock_run, tmp_path):
        mock_run.return_value = (0, "", "")
        fp = _create_fake_backup(str(tmp_path), "random-backup.sql.gz", 4096)
        # This file won't match the pattern in find_backup_files but we
        # can still pass it directly to verify
        info = verify_backup_file(fp, max_age_hours=9999)
        assert info.is_valid_format is False


# ===========================================================================
# Tests: verify_backups
# ===========================================================================


class TestVerifyBackups:
    """Tests for verifying all backups in a directory."""

    @patch("pg_backup_validation.run_command")
    def test_verify_multiple_files(self, mock_run, tmp_path):
        mock_run.return_value = (0, "TABLE public.jobs_job\n", "")
        _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz", 4096)
        _create_fake_backup(str(tmp_path), "ocr-coordinator-20260314-020000.sql.gz", 4096)
        results = verify_backups(str(tmp_path), max_age_hours=9999)
        assert len(results) == 2

    def test_verify_empty_directory(self, tmp_path):
        results = verify_backups(str(tmp_path))
        assert len(results) == 0


# ===========================================================================
# Tests: generate_report
# ===========================================================================


class TestGenerateReport:
    """Tests for report generation."""

    @patch("pg_backup_validation.run_command")
    def test_report_structure(self, mock_run, tmp_path):
        mock_run.return_value = (0, "ok", "")
        _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz", 4096)
        report = generate_report(
            database_url=SAMPLE_DB_URL,
            backup_dir=str(tmp_path),
            max_age_hours=9999,
        )
        assert report.timestamp != ""
        assert report.mode == "report"
        assert "****" in report.database_url_masked or "secret" not in report.database_url_masked
        assert "overall_health" in report.summary
        assert isinstance(report.checks, list)
        assert isinstance(report.backup_files, list)

    @patch("pg_backup_validation.run_command")
    def test_report_critical_no_backups(self, mock_run, tmp_path):
        mock_run.return_value = (0, "ok", "")
        report = generate_report(
            database_url=SAMPLE_DB_URL,
            backup_dir=str(tmp_path),
            max_age_hours=24,
        )
        assert report.summary["overall_health"] == "CRITICAL"
        assert report.summary["total_backup_files"] == 0

    @patch("pg_backup_validation.run_command")
    def test_report_healthy(self, mock_run, tmp_path):
        # All checks pass + recent valid backups = HEALTHY
        def side_effect(cmd, **kwargs):
            if cmd[0] == "pg_restore":
                return (0, "TABLE public.jobs_job\n", "")
            return (0, "ok", "")
        mock_run.side_effect = side_effect
        _create_fake_backup(str(tmp_path), "ocr-coordinator-20260315-020000.sql.gz", 4096)
        report = generate_report(
            database_url=SAMPLE_DB_URL,
            backup_dir=str(tmp_path),
            max_age_hours=9999,
        )
        assert report.summary["valid_backup_files"] >= 1


# ===========================================================================
# Tests: write_report_json
# ===========================================================================


class TestWriteReportJson:
    """Tests for JSON report output."""

    def test_write_creates_file(self, tmp_path):
        report = ValidationReport(
            timestamp="2026-03-15T02:00:00",
            mode="report",
            summary={"overall_health": "HEALTHY"},
        )
        filepath = write_report_json(report, str(tmp_path))
        assert Path(filepath).exists()
        data = json.loads(Path(filepath).read_text())
        assert data["mode"] == "report"
        assert data["summary"]["overall_health"] == "HEALTHY"

    def test_write_creates_output_dir(self, tmp_path):
        out = str(tmp_path / "nested" / "reports")
        report = ValidationReport(timestamp="2026-03-15T02:00:00")
        filepath = write_report_json(report, out)
        assert Path(filepath).exists()


# ===========================================================================
# Tests: write_report_markdown
# ===========================================================================


class TestWriteReportMarkdown:
    """Tests for markdown report output."""

    def test_write_creates_file(self, tmp_path):
        report = ValidationReport(
            timestamp="2026-03-15T02:00:00",
            mode="report",
            database_url_masked="postgres://ocr:****@localhost/db",
            backup_dir="/backups",
            summary={
                "overall_health": "HEALTHY",
                "total_backup_files": 3,
                "valid_backup_files": 3,
                "recent_backup_files": 2,
                "max_age_hours": 24,
                "checks_passed": 5,
                "checks_total": 5,
            },
            checks=[
                {"name": "test_check", "passed": True, "message": "all good"},
            ],
            backup_files=[
                {
                    "filename": "ocr-coordinator-20260315-020000.sql.gz",
                    "size_bytes": 4096,
                    "age_hours": 1.5,
                    "is_valid_format": True,
                    "pg_restore_list_ok": True,
                    "table_count": 10,
                    "errors": [],
                },
            ],
        )
        filepath = write_report_markdown(report, str(tmp_path))
        content = Path(filepath).read_text()
        assert "# PostgreSQL Backup Validation Report" in content
        assert "HEALTHY" in content
        assert "test_check" in content
        assert "ocr-coordinator-20260315" in content

    def test_write_with_errors(self, tmp_path):
        report = ValidationReport(
            timestamp="2026-03-15T02:00:00",
            summary={"overall_health": "CRITICAL"},
            errors=["Backup file corrupted", "pg_restore failed"],
        )
        filepath = write_report_markdown(report, str(tmp_path))
        content = Path(filepath).read_text()
        assert "## Errors" in content
        assert "Backup file corrupted" in content


# ===========================================================================
# Tests: CheckResult dataclass
# ===========================================================================


class TestCheckResult:
    """Tests for CheckResult data structure."""

    def test_basic_construction(self):
        r = CheckResult(name="test", passed=True, message="ok")
        assert r.name == "test"
        assert r.passed is True
        assert r.details is None

    def test_with_details(self):
        r = CheckResult(name="row_count", passed=False, message="mismatch", details={"src": 10, "dst": 5})
        assert r.details["src"] == 10


# ===========================================================================
# Tests: BackupFileInfo dataclass
# ===========================================================================


class TestBackupFileInfo:
    """Tests for BackupFileInfo data structure."""

    def test_default_values(self):
        info = BackupFileInfo(
            path="/backups/test.sql.gz",
            filename="test.sql.gz",
            size_bytes=1024,
            timestamp_str="20260315-020000",
        )
        assert info.pg_restore_list_ok is False
        assert info.table_count == 0
        assert info.tables_found == []
        assert info.errors == []
        assert info.age_hours == 0.0


# ===========================================================================
# Tests: build_parser
# ===========================================================================


class TestBuildParser:
    """Tests for CLI argument parser."""

    def test_parser_creates(self):
        parser = build_parser()
        assert parser is not None

    def test_check_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--check", "--database-url", SAMPLE_DB_URL])
        assert args.check is True
        assert args.database_url == SAMPLE_DB_URL

    def test_verify_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--verify", "--backup-dir", "/my/backups"])
        assert args.verify is True
        assert args.backup_dir == "/my/backups"

    def test_backup_mode_with_dry_run(self):
        parser = build_parser()
        args = parser.parse_args(["--backup", "--dry-run"])
        assert args.backup is True
        assert args.dry_run is True

    def test_report_mode_with_output(self):
        parser = build_parser()
        args = parser.parse_args(["--report", "--output-dir", "/tmp/reports"])
        assert args.report is True
        assert args.output_dir == "/tmp/reports"

    def test_max_age_hours(self):
        parser = build_parser()
        args = parser.parse_args(["--verify", "--max-age-hours", "48"])
        assert args.max_age_hours == 48.0

    def test_default_values(self):
        parser = build_parser()
        args = parser.parse_args(["--check"])
        assert args.backup_dir == DEFAULT_BACKUP_DIR
        assert args.max_age_hours == DEFAULT_MAX_AGE_HOURS
        assert args.dry_run is False
        assert args.verbose is False


# ===========================================================================
# Tests: main (CLI integration)
# ===========================================================================


class TestMain:
    """Tests for main CLI entry point."""

    def test_no_mode_fails(self):
        with pytest.raises(SystemExit):
            main([])

    @patch("pg_backup_validation.check_config")
    def test_check_mode_passes(self, mock_check):
        mock_check.return_value = [
            CheckResult(name="test", passed=True, message="ok"),
        ]
        rc = main(["--check", "--database-url", SAMPLE_DB_URL])
        assert rc == 0

    @patch("pg_backup_validation.check_config")
    def test_check_mode_fails(self, mock_check):
        mock_check.return_value = [
            CheckResult(name="test", passed=False, message="fail"),
        ]
        rc = main(["--check", "--database-url", SAMPLE_DB_URL])
        assert rc == 1

    @patch("pg_backup_validation.verify_backups")
    def test_verify_mode_no_files(self, mock_verify, tmp_path):
        mock_verify.return_value = []
        rc = main(["--verify", "--backup-dir", str(tmp_path)])
        assert rc == 1

    @patch("pg_backup_validation.verify_backups")
    def test_verify_mode_with_errors(self, mock_verify, tmp_path):
        mock_verify.return_value = [
            BackupFileInfo(
                path="/backups/test.sql.gz",
                filename="test.sql.gz",
                size_bytes=4096,
                timestamp_str="20260315-020000",
                pg_restore_list_ok=True,
                errors=["too old"],
            ),
        ]
        rc = main(["--verify", "--backup-dir", str(tmp_path)])
        assert rc == 1

    @patch("pg_backup_validation.trigger_backup")
    def test_backup_mode_dry_run(self, mock_backup):
        mock_backup.return_value = (True, None, [])
        rc = main(["--backup", "--database-url", SAMPLE_DB_URL, "--dry-run"])
        assert rc == 0

    def test_backup_mode_no_url(self):
        # Ensure DATABASE_URL env is not set for this test
        with patch.dict(os.environ, {}, clear=False):
            if "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]
            rc = main(["--backup", "--database-url", ""])
        assert rc == 1

    @patch("pg_backup_validation.generate_report")
    @patch("pg_backup_validation.write_report_json")
    @patch("pg_backup_validation.write_report_markdown")
    def test_report_mode(self, mock_md, mock_json, mock_report, tmp_path):
        mock_report.return_value = ValidationReport(
            timestamp="2026-03-15T02:00:00",
            summary={"overall_health": "HEALTHY"},
        )
        mock_json.return_value = str(tmp_path / "report.json")
        mock_md.return_value = str(tmp_path / "report.md")
        rc = main(["--report", "--output-dir", str(tmp_path)])
        assert rc == 0

    @patch("pg_backup_validation.generate_report")
    @patch("pg_backup_validation.write_report_json")
    @patch("pg_backup_validation.write_report_markdown")
    def test_report_critical_returns_1(self, mock_md, mock_json, mock_report, tmp_path):
        mock_report.return_value = ValidationReport(
            timestamp="2026-03-15T02:00:00",
            summary={"overall_health": "CRITICAL"},
        )
        mock_json.return_value = str(tmp_path / "report.json")
        mock_md.return_value = str(tmp_path / "report.md")
        rc = main(["--report", "--output-dir", str(tmp_path)])
        assert rc == 1

    def test_restore_test_no_url(self):
        with patch.dict(os.environ, {}, clear=False):
            if "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]
            rc = main(["--restore-test", "--database-url", ""])
        assert rc == 1


# ===========================================================================
# Tests: constants
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_default_max_age(self):
        assert DEFAULT_MAX_AGE_HOURS == 24

    def test_default_min_size(self):
        assert DEFAULT_MIN_SIZE_BYTES == 1024

    def test_default_backup_dir(self):
        assert DEFAULT_BACKUP_DIR == "/backups"

    def test_expected_tables(self):
        assert "jobs_job" in EXPECTED_TABLES
        assert "jobs_worker" in EXPECTED_TABLES
        assert "jobs_pageresult" in EXPECTED_TABLES
        assert "jobs_custodyevent" in EXPECTED_TABLES
