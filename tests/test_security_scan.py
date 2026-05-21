"""Tests for scripts/security_scan.py.

Covers Finding/ScanReport dataclasses, Severity enum, each individual
check function, run_scan integration, and report output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from security_scan import (
    Finding,
    ScanReport,
    Severity,
    _find_project_root,
    check_auth_bypass,
    check_cors_config,
    check_dockerfile_user,
    check_env_files,
    check_gitignore_env,
    check_hardcoded_secrets,
    check_rate_limit_auth,
    check_tls_config,
    print_report,
    run_scan,
)

# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class TestSeverity:
    """Tests for the Severity enum."""

    def test_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

    def test_member_count(self):
        assert len(Severity) == 5


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


class TestFinding:
    """Tests for the Finding dataclass."""

    def test_creation(self):
        f = Finding(
            check_id="TEST-001",
            title="Test finding",
            severity=Severity.LOW,
            description="A test",
            recommendation="Fix it",
            passed=True,
        )
        assert f.check_id == "TEST-001"
        assert f.passed is True
        assert f.details == ""

    def test_with_details(self):
        f = Finding(
            check_id="TEST-002",
            title="Detailed finding",
            severity=Severity.HIGH,
            description="Has details",
            recommendation="Read them",
            passed=False,
            details="line1\nline2",
        )
        assert f.details == "line1\nline2"
        assert f.passed is False


# ---------------------------------------------------------------------------
# ScanReport dataclass
# ---------------------------------------------------------------------------


class TestScanReport:
    """Tests for the ScanReport dataclass."""

    def test_empty_report(self):
        r = ScanReport()
        assert r.passed == 0
        assert r.failed == 0
        assert r.findings == []

    def test_add_passed(self):
        r = ScanReport()
        r.add(Finding("T-1", "pass", Severity.LOW, "", "", True))
        assert r.passed == 1
        assert r.failed == 0
        assert len(r.findings) == 1

    def test_add_failed(self):
        r = ScanReport()
        r.add(Finding("T-2", "fail", Severity.HIGH, "", "", False))
        assert r.passed == 0
        assert r.failed == 1

    def test_add_mixed(self):
        r = ScanReport()
        r.add(Finding("T-1", "pass", Severity.LOW, "", "", True))
        r.add(Finding("T-2", "fail", Severity.HIGH, "", "", False))
        r.add(Finding("T-3", "pass2", Severity.INFO, "", "", True))
        assert r.passed == 2
        assert r.failed == 1

    def test_to_dict_structure(self):
        r = ScanReport()
        r.add(Finding("T-1", "pass", Severity.LOW, "d", "r", True, "det"))
        d = r.to_dict()
        assert d["summary"]["passed"] == 1
        assert d["summary"]["failed"] == 0
        assert d["summary"]["total"] == 1
        assert len(d["findings"]) == 1
        f = d["findings"][0]
        assert f["check_id"] == "T-1"
        assert f["severity"] == "low"
        assert f["passed"] is True
        assert f["details"] == "det"

    def test_to_dict_json_serializable(self):
        r = ScanReport()
        r.add(Finding("T-1", "t", Severity.CRITICAL, "d", "r", False))
        # Should not raise
        output = json.dumps(r.to_dict())
        assert "T-1" in output


# ---------------------------------------------------------------------------
# check_env_files
# ---------------------------------------------------------------------------


class TestCheckEnvFiles:
    """Tests for check_env_files."""

    def test_clean_directory(self, tmp_path):
        report = ScanReport()
        check_env_files(tmp_path, report)
        assert report.passed == 1
        assert report.findings[0].check_id == "SEC-001"
        assert report.findings[0].passed is True

    @patch("security_scan._is_git_tracked", return_value=True)
    def test_env_with_secret(self, _mock_tracked, tmp_path):
        env_file = tmp_path / "coordinator" / ".env"
        env_file.parent.mkdir()
        env_file.write_text("DB_PASSWORD=supersecretvalue123\n")
        report = ScanReport()
        check_env_files(tmp_path, report)
        assert report.failed == 1
        assert report.findings[0].passed is False
        assert "DB_PASSWORD" in report.findings[0].details

    def test_env_untracked_skipped(self, tmp_path):
        """Untracked .env files are not flagged (SEC-001 false positive fix)."""
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET_KEY=realproductionvalue999\n")
        # tmp_path is not a git repo, so _is_git_tracked returns False
        report = ScanReport()
        check_env_files(tmp_path, report)
        assert report.passed == 1
        assert report.findings[0].passed is True

    def test_env_with_placeholder_passes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET_KEY=changeme\n")
        report = ScanReport()
        check_env_files(tmp_path, report)
        assert report.passed == 1

    def test_env_comment_ignored(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# SECRET_KEY=realvalue123456\n")
        report = ScanReport()
        check_env_files(tmp_path, report)
        assert report.passed == 1


# ---------------------------------------------------------------------------
# check_gitignore_env
# ---------------------------------------------------------------------------


class TestCheckGitignoreEnv:
    """Tests for check_gitignore_env."""

    def test_with_env_in_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.pyc\n.env\n__pycache__/\n")
        report = ScanReport()
        check_gitignore_env(tmp_path, report)
        assert report.passed == 1
        assert report.findings[0].check_id == "SEC-002"

    def test_without_env_in_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")
        report = ScanReport()
        check_gitignore_env(tmp_path, report)
        assert report.failed == 1

    def test_no_gitignore(self, tmp_path):
        report = ScanReport()
        check_gitignore_env(tmp_path, report)
        assert report.failed == 1


# ---------------------------------------------------------------------------
# check_dockerfile_user
# ---------------------------------------------------------------------------


class TestCheckDockerfileUser:
    """Tests for check_dockerfile_user."""

    def test_with_user_directive(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11-slim\nRUN pip install app\nUSER ocr\nCMD [\"python\", \"app.py\"]\n")
        report = ScanReport()
        check_dockerfile_user(tmp_path, report)
        assert report.passed == 1
        assert report.findings[0].check_id == "SEC-003"

    def test_without_user_directive(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11-slim\nRUN pip install app\nCMD [\"python\", \"app.py\"]\n")
        report = ScanReport()
        check_dockerfile_user(tmp_path, report)
        assert report.failed == 1
        assert "No USER directive" in report.findings[0].details

    def test_no_dockerfiles(self, tmp_path):
        report = ScanReport()
        check_dockerfile_user(tmp_path, report)
        # No Dockerfiles means nothing to fail
        assert report.passed == 1


# ---------------------------------------------------------------------------
# check_cors_config
# ---------------------------------------------------------------------------


class TestCheckCorsConfig:
    """Tests for check_cors_config."""

    def test_with_cors_middleware(self, tmp_path):
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "main.py").write_text(
            "from starlette.middleware.cors import CORSMiddleware\napp.add_middleware(CORSMiddleware)\n"
        )
        report = ScanReport()
        check_cors_config(tmp_path, report)
        assert report.passed == 1
        assert report.findings[0].check_id == "SEC-004"

    def test_without_cors_middleware(self, tmp_path):
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
        report = ScanReport()
        check_cors_config(tmp_path, report)
        assert report.failed == 1

    def test_no_main_py(self, tmp_path):
        report = ScanReport()
        check_cors_config(tmp_path, report)
        assert report.failed == 1


# ---------------------------------------------------------------------------
# check_tls_config
# ---------------------------------------------------------------------------


class TestCheckTlsConfig:
    """Tests for check_tls_config."""

    def test_with_tls(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text(
            "services:\n  proxy:\n    ports:\n      - '443:443'\n"
        )
        report = ScanReport()
        check_tls_config(tmp_path, report)
        assert report.passed == 1

    def test_without_tls(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text(
            "services:\n  api:\n    ports:\n      - '8080:8080'\n"
        )
        report = ScanReport()
        check_tls_config(tmp_path, report)
        assert report.failed == 1


# ---------------------------------------------------------------------------
# check_hardcoded_secrets
# ---------------------------------------------------------------------------


class TestCheckHardcodedSecrets:
    """Tests for check_hardcoded_secrets."""

    def test_clean_files(self, tmp_path):
        (tmp_path / "app.py").write_text("import os\nname = 'hello'\n")
        report = ScanReport()
        check_hardcoded_secrets(tmp_path, report)
        assert report.passed == 1
        assert report.findings[0].check_id == "SEC-008"

    def test_hardcoded_password(self, tmp_path):
        (tmp_path / "app.py").write_text(
            'db_password = "my_super_secret_password_123"\n'
        )
        report = ScanReport()
        check_hardcoded_secrets(tmp_path, report)
        assert report.failed == 1

    def test_env_lookup_ignored(self, tmp_path):
        (tmp_path / "app.py").write_text(
            'password = os.environ.get("DB_PASSWORD", "default_value_here")\n'
        )
        report = ScanReport()
        check_hardcoded_secrets(tmp_path, report)
        assert report.passed == 1


# ---------------------------------------------------------------------------
# check_auth_bypass
# ---------------------------------------------------------------------------


class TestCheckAuthBypass:
    """Tests for check_auth_bypass."""

    def test_no_auth_file(self, tmp_path):
        report = ScanReport()
        check_auth_bypass(tmp_path, report)
        assert report.passed == 1

    def test_with_bypass_flag(self, tmp_path):
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "auth.py").write_text(
            'allow = os.environ.get("ALLOW_UNAUTHENTICATED", "false")\n'
        )
        report = ScanReport()
        check_auth_bypass(tmp_path, report)
        assert report.failed == 1


# ---------------------------------------------------------------------------
# check_rate_limit_auth
# ---------------------------------------------------------------------------


class TestCheckRateLimitAuth:
    """Tests for check_rate_limit_auth."""

    def test_no_limits_file(self, tmp_path):
        report = ScanReport()
        check_rate_limit_auth(tmp_path, report)
        assert report.failed == 1

    def test_with_auth_limit(self, tmp_path):
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "limits.py").write_text(
            'AUTH_RATE_LIMIT = "5/minute"\n'
        )
        report = ScanReport()
        check_rate_limit_auth(tmp_path, report)
        assert report.passed == 1


# ---------------------------------------------------------------------------
# run_scan integration
# ---------------------------------------------------------------------------


class TestRunScan:
    """Tests for the run_scan integration function."""

    def test_returns_scan_report(self, tmp_path):
        report = run_scan(tmp_path)
        assert isinstance(report, ScanReport)
        assert report.passed + report.failed == len(report.findings)

    def test_all_checks_run(self, tmp_path):
        report = run_scan(tmp_path)
        check_ids = {f.check_id for f in report.findings}
        expected = {f"SEC-{i:03d}" for i in range(1, 10)}
        assert check_ids == expected


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------


class TestPrintReport:
    """Tests for print_report output."""

    def test_does_not_crash_empty(self, capsys):
        report = ScanReport()
        print_report(report)
        captured = capsys.readouterr()
        assert "Security Scan Report" in captured.out

    def test_shows_failed_findings(self, capsys):
        report = ScanReport()
        report.add(
            Finding("T-1", "Failing check", Severity.HIGH, "desc", "fix", False, "detail line")
        )
        print_report(report)
        captured = capsys.readouterr()
        assert "[FAIL]" in captured.out
        assert "T-1" in captured.out
        assert "detail line" in captured.out

    def test_shows_passed_findings(self, capsys):
        report = ScanReport()
        report.add(Finding("T-2", "Passing check", Severity.LOW, "d", "r", True))
        print_report(report)
        captured = capsys.readouterr()
        assert "[PASS]" in captured.out
        assert "T-2" in captured.out


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    """Tests for _find_project_root."""

    def test_returns_path(self):
        result = _find_project_root()
        assert isinstance(result, Path)
