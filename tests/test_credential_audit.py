"""Tests for scripts/credential_audit.py.

Covers CredentialType/CredentialStatus enums, CredentialFinding dataclass,
CredentialAuditor construction and all methods, scan_env_files, scan_source_code,
check_credential_strength, audit_all aggregation, generate_report, and CLI parsing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from credential_audit import (
    KNOWN_DEFAULTS,
    CredentialAuditor,
    CredentialFinding,
    CredentialStatus,
    CredentialType,
    _find_project_root,
    main,
)

# ---------------------------------------------------------------------------
# CredentialType enum
# ---------------------------------------------------------------------------


class TestCredentialType:
    """Tests for the CredentialType enum."""

    def test_values(self):
        assert CredentialType.DATABASE.value == "database"
        assert CredentialType.API_KEY.value == "api_key"
        assert CredentialType.S3_ACCESS.value == "s3_access"
        assert CredentialType.S3_SECRET.value == "s3_secret"
        assert CredentialType.DJANGO_SECRET.value == "django_secret"
        assert CredentialType.RABBITMQ.value == "rabbitmq"
        assert CredentialType.REDIS.value == "redis"
        assert CredentialType.SMTP.value == "smtp"
        assert CredentialType.OAUTH.value == "oauth"
        assert CredentialType.CUSTOM.value == "custom"

    def test_member_count(self):
        assert len(CredentialType) == 10


# ---------------------------------------------------------------------------
# CredentialStatus enum
# ---------------------------------------------------------------------------


class TestCredentialStatus:
    """Tests for the CredentialStatus enum."""

    def test_values(self):
        assert CredentialStatus.SECURE.value == "secure"
        assert CredentialStatus.WEAK.value == "weak"
        assert CredentialStatus.DEFAULT.value == "default"
        assert CredentialStatus.EXPOSED.value == "exposed"
        assert CredentialStatus.EXPIRED.value == "expired"
        assert CredentialStatus.MISSING.value == "missing"

    def test_member_count(self):
        assert len(CredentialStatus) == 6


# ---------------------------------------------------------------------------
# CredentialFinding dataclass
# ---------------------------------------------------------------------------


class TestCredentialFinding:
    """Tests for the CredentialFinding dataclass."""

    def test_creation(self):
        f = CredentialFinding(
            credential_type=CredentialType.DATABASE,
            status=CredentialStatus.WEAK,
            location=".env:3",
            message="DB_PASSWORD is weak",
            severity="high",
            recommendation="Use a stronger password.",
        )
        assert f.credential_type == CredentialType.DATABASE
        assert f.status == CredentialStatus.WEAK
        assert f.location == ".env:3"
        assert f.message == "DB_PASSWORD is weak"
        assert f.severity == "high"
        assert f.recommendation == "Use a stronger password."

    def test_equality(self):
        a = CredentialFinding(
            CredentialType.API_KEY, CredentialStatus.EXPOSED,
            "app.py:10", "key exposed", "critical", "rotate",
        )
        b = CredentialFinding(
            CredentialType.API_KEY, CredentialStatus.EXPOSED,
            "app.py:10", "key exposed", "critical", "rotate",
        )
        assert a == b

    def test_different_severity(self):
        a = CredentialFinding(
            CredentialType.REDIS, CredentialStatus.DEFAULT,
            ".env:1", "msg", "critical", "fix",
        )
        b = CredentialFinding(
            CredentialType.REDIS, CredentialStatus.DEFAULT,
            ".env:1", "msg", "low", "fix",
        )
        assert a != b


# ---------------------------------------------------------------------------
# CredentialAuditor – construction
# ---------------------------------------------------------------------------


class TestAuditorConstruction:
    """Tests for CredentialAuditor __init__."""

    def test_stores_project_root(self, tmp_path):
        auditor = CredentialAuditor(str(tmp_path))
        assert auditor.project_root == tmp_path.resolve()

    def test_resolves_relative_path(self):
        auditor = CredentialAuditor(".")
        assert auditor.project_root.is_absolute()


# ---------------------------------------------------------------------------
# scan_env_files
# ---------------------------------------------------------------------------


class TestScanEnvFiles:
    """Tests for CredentialAuditor.scan_env_files."""

    def test_empty_directory(self, tmp_path):
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert findings == []

    def test_default_password_detected(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("DB_PASSWORD=changeme\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert len(findings) == 1
        assert findings[0].status == CredentialStatus.DEFAULT
        assert findings[0].severity == "critical"
        assert "DB_PASSWORD" in findings[0].message

    def test_minioadmin_detected(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("S3_SECRET_KEY=minioadmin\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert len(findings) == 1
        assert findings[0].status == CredentialStatus.DEFAULT

    def test_weak_password_detected(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("REDIS_PASSWORD=abc\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert len(findings) == 1
        assert findings[0].status == CredentialStatus.WEAK

    def test_strong_password_no_finding(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "DB_PASSWORD="
            + "Fj7$Nq2L"
            + "r9!Zp6Xm"
            + "\n"
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert findings == []

    def test_comments_skipped(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("# DB_PASSWORD=changeme\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert findings == []

    def test_non_secret_keys_skipped(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("APP_NAME=changeme\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert findings == []

    def test_quoted_values_handled(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("DB_PASSWORD='changeme'\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert len(findings) == 1
        assert findings[0].status == CredentialStatus.DEFAULT

    def test_env_example_excluded(self, tmp_path):
        env = tmp_path / ".env.example"
        env.write_text("DB_PASSWORD=changeme\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert findings == []

    def test_multiple_findings(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "DB_PASSWORD=changeme\n"
            "REDIS_PASSWORD=admin\n"
            "APP_MODE=production\n"
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert len(findings) == 2

    def test_coordinator_env_scanned(self, tmp_path):
        coord = tmp_path / "coordinator"
        coord.mkdir()
        env = coord / ".env"
        env.write_text("DJANGO_SECRET_KEY=changeme\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert len(findings) >= 1
        assert any("DJANGO_SECRET_KEY" in f.message for f in findings)

    def test_empty_value_skipped(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("DB_PASSWORD=\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_env_files()
        assert findings == []


# ---------------------------------------------------------------------------
# scan_source_code
# ---------------------------------------------------------------------------


class TestScanSourceCode:
    """Tests for CredentialAuditor.scan_source_code."""

    def test_clean_source(self, tmp_path):
        (tmp_path / "app.py").write_text("import os\nname = 'hello'\n")
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_source_code()
        assert findings == []

    def test_hardcoded_password_detected(self, tmp_path):
        (tmp_path / "config.py").write_text(
            'DB_PASSWORD = "'
            + "fixture_exposed_"
            + 'token_value"\n'
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_source_code()
        assert len(findings) == 1
        assert findings[0].status == CredentialStatus.EXPOSED
        assert findings[0].severity == "high"

    def test_hardcoded_secret_key_detected(self, tmp_path):
        (tmp_path / "settings.py").write_text(
            'DJANGO_SECRET = "my-django-key-goes-here"\n'
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_source_code()
        assert len(findings) == 1

    def test_env_lookup_ignored(self, tmp_path):
        (tmp_path / "app.py").write_text(
            'DB_PASSWORD = os.environ.get("DB_PASSWORD", "fallback")\n'
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_source_code()
        assert findings == []

    def test_comment_line_ignored(self, tmp_path):
        (tmp_path / "app.py").write_text(
            '# DB_PASSWORD = "do_not_use_this"\n'
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_source_code()
        assert findings == []

    def test_test_files_excluded(self, tmp_path):
        (tmp_path / "test_config.py").write_text(
            'DB_PASSWORD = "'
            + "fixture_test_"
            + 'token_value"\n'
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_source_code()
        assert findings == []

    def test_yaml_file_scanned(self, tmp_path):
        (tmp_path / "deploy.yml").write_text(
            'REDIS_PASSWORD = "hardcoded_redis_pw"\n'
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_source_code()
        assert len(findings) == 1

    def test_example_files_excluded(self, tmp_path):
        (tmp_path / "config.example.py").write_text(
            'DB_PASSWORD = "placeholder_value"\n'
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.scan_source_code()
        assert findings == []


# ---------------------------------------------------------------------------
# check_credential_strength
# ---------------------------------------------------------------------------


class TestCheckCredentialStrength:
    """Tests for CredentialAuditor.check_credential_strength."""

    def setup_method(self):
        self.auditor = CredentialAuditor(".")

    def test_empty_string_is_missing(self):
        status, msg = self.auditor.check_credential_strength("")
        assert status == CredentialStatus.MISSING

    def test_whitespace_only_is_missing(self):
        status, _ = self.auditor.check_credential_strength("   ")
        assert status == CredentialStatus.MISSING

    @pytest.mark.parametrize("value", sorted(KNOWN_DEFAULTS))
    def test_known_defaults(self, value):
        status, _ = self.auditor.check_credential_strength(value)
        assert status == CredentialStatus.DEFAULT

    def test_default_case_insensitive(self):
        status, _ = self.auditor.check_credential_strength("MiniOAdmin")
        assert status == CredentialStatus.DEFAULT

    def test_placeholder_substring_changeme(self):
        status, _ = self.auditor.check_credential_strength("please-changeme-now")
        assert status == CredentialStatus.DEFAULT

    def test_placeholder_substring_django_insecure(self):
        status, _ = self.auditor.check_credential_strength(
            "django-insecure-some-key-here"
        )
        assert status == CredentialStatus.DEFAULT

    def test_short_password_weak(self):
        status, _ = self.auditor.check_credential_strength("Xy3!")
        assert status == CredentialStatus.WEAK

    def test_all_lowercase_weak(self):
        status, _ = self.auditor.check_credential_strength("abcdefghijklmn")
        assert status == CredentialStatus.WEAK

    def test_all_digits_weak(self):
        status, _ = self.auditor.check_credential_strength("123456789012")
        assert status == CredentialStatus.WEAK

    def test_strong_password_secure(self):
        status, _ = self.auditor.check_credential_strength("x9kF!mQ2pL@w3nR7")
        assert status == CredentialStatus.SECURE

    def test_secure_returns_recommendation(self):
        _, msg = self.auditor.check_credential_strength("x9kF!mQ2pL@w3nR7")
        assert "meets" in msg.lower() or "strength" in msg.lower()

    def test_weak_returns_recommendation(self):
        _, msg = self.auditor.check_credential_strength("short")
        assert msg  # non-empty recommendation

    def test_mixed_case_digits_long_is_secure(self):
        status, _ = self.auditor.check_credential_strength("AbCdEf123456")
        assert status == CredentialStatus.SECURE


# ---------------------------------------------------------------------------
# audit_all
# ---------------------------------------------------------------------------


class TestAuditAll:
    """Tests for CredentialAuditor.audit_all."""

    def test_empty_project(self, tmp_path):
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.audit_all()
        assert findings == []

    def test_aggregates_env_and_source(self, tmp_path):
        # Create an env finding
        env = tmp_path / ".env"
        env.write_text("DB_PASSWORD=changeme\n")
        # Create a source finding
        (tmp_path / "config.py").write_text(
            'API_SECRET = "hardcoded_api_secret_value"\n'
        )
        auditor = CredentialAuditor(str(tmp_path))
        findings = auditor.audit_all()
        assert len(findings) >= 2
        statuses = {f.status for f in findings}
        assert CredentialStatus.DEFAULT in statuses
        assert CredentialStatus.EXPOSED in statuses

    def test_returns_list(self, tmp_path):
        auditor = CredentialAuditor(str(tmp_path))
        result = auditor.audit_all()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for CredentialAuditor.generate_report."""

    def setup_method(self):
        self.auditor = CredentialAuditor(".")

    def test_empty_findings(self):
        report = self.auditor.generate_report([])
        assert report["total_findings"] == 0
        assert report["by_severity"]["critical"] == 0
        assert report["by_severity"]["high"] == 0
        assert report["by_severity"]["medium"] == 0
        assert report["by_severity"]["low"] == 0
        assert report["findings"] == []

    def test_report_structure(self):
        findings = [
            CredentialFinding(
                CredentialType.DATABASE, CredentialStatus.DEFAULT,
                ".env:1", "weak", "critical", "fix",
            ),
            CredentialFinding(
                CredentialType.API_KEY, CredentialStatus.EXPOSED,
                "app.py:5", "exposed", "high", "rotate",
            ),
        ]
        report = self.auditor.generate_report(findings)
        assert report["total_findings"] == 2
        assert report["by_severity"]["critical"] == 1
        assert report["by_severity"]["high"] == 1
        assert len(report["findings"]) == 2

    def test_by_status_populated(self):
        findings = [
            CredentialFinding(
                CredentialType.REDIS, CredentialStatus.WEAK,
                ".env:1", "weak", "high", "fix",
            ),
        ]
        report = self.auditor.generate_report(findings)
        assert report["by_status"]["weak"] == 1

    def test_report_json_serializable(self):
        findings = [
            CredentialFinding(
                CredentialType.S3_ACCESS, CredentialStatus.DEFAULT,
                ".env:2", "msg", "critical", "fix",
            ),
        ]
        report = self.auditor.generate_report(findings)
        output = json.dumps(report)
        assert "critical" in output

    def test_finding_dict_fields(self):
        findings = [
            CredentialFinding(
                CredentialType.SMTP, CredentialStatus.EXPOSED,
                "mail.py:10", "hardcoded", "medium", "rotate",
            ),
        ]
        report = self.auditor.generate_report(findings)
        f = report["findings"][0]
        assert f["credential_type"] == "smtp"
        assert f["status"] == "exposed"
        assert f["location"] == "mail.py:10"
        assert f["message"] == "hardcoded"
        assert f["severity"] == "medium"
        assert f["recommendation"] == "rotate"


# ---------------------------------------------------------------------------
# CLI / argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the CLI entry-point and argument parsing."""

    def test_json_output(self, tmp_path, capsys):
        main(["--project-root", str(tmp_path), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "total_findings" in data
        assert "by_severity" in data

    def test_text_output(self, tmp_path, capsys):
        main(["--project-root", str(tmp_path)])
        captured = capsys.readouterr()
        assert "Credential Audit Report" in captured.out

    def test_strict_mode_clean_exits_zero(self, tmp_path):
        # Clean project should exit normally (no exception)
        main(["--project-root", str(tmp_path), "--strict"])

    def test_strict_mode_findings_exits_one(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("DB_PASSWORD=changeme\n")
        with pytest.raises(SystemExit) as exc_info:
            main(["--project-root", str(tmp_path), "--strict"])
        assert exc_info.value.code == 1

    def test_default_project_root(self, capsys):
        # Calling with no --project-root should work (auto-detect)
        main(["--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data["total_findings"], int)


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    """Tests for _find_project_root helper."""

    def test_returns_path(self):
        root = _find_project_root()
        assert isinstance(root, Path)
        assert root.is_absolute()
