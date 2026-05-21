"""Tests for credential cutover tooling.

Tests the enhanced --production flag in validate_phase7c_env.py and the
credential_cutover_checklist.py script for correct detection of insecure
defaults, placeholder values, and JSON output format.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the scripts as modules
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import credential_cutover_checklist as checklist_script
import validate_phase7c_env as env_script

# ---------------------------------------------------------------------------
# Helper to write a .env file
# ---------------------------------------------------------------------------

def _write_env(tmp_path: Path, content: str) -> Path:
    """Write content to a .env file in tmp_path and return the path."""
    env_file = tmp_path / ".env"
    env_file.write_text(content, encoding="utf-8")
    return env_file


# ===========================================================================
# Tests for validate_phase7c_env.py enhancements
# ===========================================================================


class TestEnvValidationProductionFlag:
    """Tests for the --production flag in validate_phase7c_env.py."""

    def test_production_required_keys_list_exists(self):
        """PRODUCTION_REQUIRED_KEYS constant is defined and has expected keys."""
        assert hasattr(env_script, "PRODUCTION_REQUIRED_KEYS")
        expected = {"FLOWER_PASSWORD", "METRICS_API_KEY", "OCR_API_KEY"}
        assert set(env_script.PRODUCTION_REQUIRED_KEYS) == expected

    def test_known_insecure_defaults_expanded(self):
        """KNOWN_INSECURE_DEFAULTS includes the expanded set of insecure values."""
        assert hasattr(env_script, "KNOWN_INSECURE_DEFAULTS")
        expected_subset = {"change-me", "change-me-in-production", "password", "secret", "admin", "test", "minioadmin"}
        assert expected_subset.issubset(env_script.KNOWN_INSECURE_DEFAULTS)

    def test_validate_production_all_present_and_secure(self):
        """Production validation passes when all keys are present with secure values."""
        env_vars = {
            "FLOWER_PASSWORD": "fL0w3r-S3cur3-P4ss!",
            "METRICS_API_KEY": "mEtR1cS-k3y-4bCd-EfGh-1j2k",
            "OCR_API_KEY": "0cR-4p1-k3y-sEcUrE-v4Lu3",
        }
        results = env_script._validate_production(env_vars)
        assert len(results) == 3
        assert all(r.status == "ok" for r in results)

    def test_validate_production_missing_keys(self):
        """Production validation catches missing keys."""
        env_vars = {}
        results = env_script._validate_production(env_vars)
        assert len(results) == 3
        assert all(r.status == "missing" for r in results)

    def test_validate_production_empty_values(self):
        """Production validation catches empty values."""
        env_vars = {
            "FLOWER_PASSWORD": "",
            "METRICS_API_KEY": "",
            "OCR_API_KEY": "",
        }
        results = env_script._validate_production(env_vars)
        assert len(results) == 3
        assert all(r.status == "empty" for r in results)

    def test_validate_production_insecure_defaults(self):
        """Production validation catches known insecure defaults."""
        env_vars = {
            "FLOWER_PASSWORD": "password",
            "METRICS_API_KEY": "admin",
            "OCR_API_KEY": "test",
        }
        results = env_script._validate_production(env_vars)
        assert len(results) == 3
        assert all(r.status == "insecure_default" for r in results)

    def test_validate_production_placeholder_values(self):
        """Production validation catches placeholder values that are not exact insecure defaults."""
        env_vars = {
            "FLOWER_PASSWORD": "placeholder_password_123",
            "METRICS_API_KEY": "example-api-key-for-testing",
            "OCR_API_KEY": "change-me-to-real-key",
        }
        results = env_script._validate_production(env_vars)
        assert len(results) == 3
        # These contain placeholder patterns but are not exact insecure defaults
        # "placeholder_password_123" -> placeholder (contains "placeholder")
        # "example-api-key-for-testing" -> placeholder (contains "example")
        # "change-me-to-real-key" -> insecure_default (contains "change-me")
        for r in results:
            assert r.status in ("placeholder", "insecure_default"), (
                f"Key {r.key} had unexpected status: {r.status}"
            )

    def test_run_without_production_flag_ignores_production_keys(self, tmp_path):
        """Default run() (no --production) does not check production keys."""
        env_file = _write_env(tmp_path, (
            "DJANGO_SECRET_KEY=secure-key-abc123\n"
            "POSTGRES_PASSWORD=secure-pg-pass-xyz\n"
            "RABBITMQ_PASSWORD=secure-rabbit-pass\n"
            "RABBITMQ_ERLANG_COOKIE=randomcookievalue123\n"
            "DATABASE_URL=postgres://ocr:realpass@db:5432/ocr\n"
            "CELERY_BROKER_URL=amqp://ocr:realpass@broker:5672//\n"
        ))
        # No FLOWER_PASSWORD, METRICS_API_KEY, OCR_API_KEY
        result = env_script.run(env_path=env_file, report=None, production=False)
        assert result == 0  # passes without production keys

    def test_run_with_production_flag_fails_on_missing(self, tmp_path):
        """run() with production=True fails when production keys are missing."""
        env_file = _write_env(tmp_path, (
            "DJANGO_SECRET_KEY=secure-key-abc123\n"
            "POSTGRES_PASSWORD=secure-pg-pass-xyz\n"
            "RABBITMQ_PASSWORD=secure-rabbit-pass\n"
            "DATABASE_URL=postgres://ocr:realpass@db:5432/ocr\n"
            "CELERY_BROKER_URL=amqp://ocr:realpass@broker:5672//\n"
        ))
        result = env_script.run(env_path=env_file, report=None, production=True)
        assert result == 1  # fails because production keys are missing

    def test_run_with_production_flag_passes_all_secure(self, tmp_path):
        """run() with production=True passes when all keys are secure."""
        env_file = _write_env(tmp_path, (
            "DJANGO_SECRET_KEY=secure-key-abc123\n"
            "POSTGRES_PASSWORD=secure-pg-pass-xyz\n"
            "RABBITMQ_PASSWORD=secure-rabbit-pass\n"
            "DATABASE_URL=postgres://ocr:realpass@db:5432/ocr\n"
            "CELERY_BROKER_URL=amqp://ocr:realpass@broker:5672//\n"
            "RABBITMQ_ERLANG_COOKIE=rK9x2mNpQwT7vYhL3sZjFbCe\n"
            "FLOWER_PASSWORD=fL0w3r-S3cur3-P4ss\n"
            "METRICS_API_KEY=mEtR1cS-k3y-4bCd\n"
            "OCR_API_KEY=0cR-4p1-k3y-sEcUrE\n"
        ))
        result = env_script.run(env_path=env_file, report=None, production=True)
        assert result == 0

    def test_run_with_production_flag_fails_on_insecure(self, tmp_path):
        """run() with production=True fails when production keys have insecure values."""
        env_file = _write_env(tmp_path, (
            "DJANGO_SECRET_KEY=secure-key-abc123\n"
            "POSTGRES_PASSWORD=secure-pg-pass-xyz\n"
            "RABBITMQ_PASSWORD=secure-rabbit-pass\n"
            "DATABASE_URL=postgres://ocr:realpass@db:5432/ocr\n"
            "CELERY_BROKER_URL=amqp://ocr:realpass@broker:5672//\n"
            "RABBITMQ_ERLANG_COOKIE=change-me-in-production\n"
            "FLOWER_PASSWORD=password\n"
            "METRICS_API_KEY=admin\n"
            "OCR_API_KEY=test\n"
        ))
        result = env_script.run(env_path=env_file, report=None, production=True)
        assert result == 1

    def test_is_insecure_default_helper(self):
        """_is_insecure_default correctly identifies insecure values."""
        assert env_script._is_insecure_default("password") is True
        assert env_script._is_insecure_default("Password") is True  # case-insensitive
        assert env_script._is_insecure_default("change-me-in-production") is True
        assert env_script._is_insecure_default("admin") is True
        assert env_script._is_insecure_default("test") is True
        assert env_script._is_insecure_default("minioadmin") is True
        assert env_script._is_insecure_default("xK9z-real-secret-2024") is False

    def test_parse_args_production_flag(self):
        """--production flag is correctly parsed."""
        args = env_script.parse_args(["--production"])
        assert args.production is True

        args_default = env_script.parse_args([])
        assert args_default.production is False


# ===========================================================================
# Tests for credential_cutover_checklist.py
# ===========================================================================


class TestCredentialCutoverChecklist:
    """Tests for the credential_cutover_checklist.py script."""

    def test_security_relevant_keys_constant(self):
        """SECURITY_RELEVANT_KEYS has all expected keys."""
        expected = {
            "DJANGO_SECRET_KEY", "POSTGRES_PASSWORD", "RABBITMQ_PASSWORD",
            "RABBITMQ_ERLANG_COOKIE", "REDIS_PASSWORD", "FLOWER_PASSWORD",
            "METRICS_API_KEY", "OCR_API_KEY", "S3_ACCESS_KEY", "S3_SECRET_KEY",
            "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD",
        }
        assert set(checklist_script.SECURITY_RELEVANT_KEYS) == expected

    def test_assess_env_all_secure(self):
        """assess_env returns all-secure results for strong credentials."""
        env_vars = {
            "DJANGO_SECRET_KEY": "xK9z-real-secret-2024-abcdef",
            "POSTGRES_PASSWORD": "pG-str0ng-p4ss-w0rd-xyz789",
            "RABBITMQ_PASSWORD": "rAbb1t-s3cur3-p4ss-123",
            "RABBITMQ_ERLANG_COOKIE": "eRl4ng-c00k13-r4nd0m-val",
            "REDIS_PASSWORD": "r3d1s-s3cur3-p4ss-456",
            "FLOWER_PASSWORD": "fL0w3r-s3cur3-p4ss-789",
            "METRICS_API_KEY": "m3tr1cs-k3y-s3cur3-abc",
            "OCR_API_KEY": "0cr-k3y-s3cur3-def",
            "S3_ACCESS_KEY": "s3-prod-access-key-xK9z42Qm",
            "S3_SECRET_KEY": "s3-prod-secret-key-7bN3mPqR-longvalue",
            "MINIO_ROOT_USER": "minio-prod-admin-user",
            "MINIO_ROOT_PASSWORD": "m1n10-pr0d-s3cr3t-p4ss",
        }
        reports = checklist_script.assess_env(env_vars)
        assert len(reports) == 12
        assert all(r.has_value for r in reports)
        assert all(not r.is_insecure for r in reports)

    def test_assess_env_detects_insecure_values(self):
        """assess_env correctly flags insecure default values."""
        env_vars = {
            "DJANGO_SECRET_KEY": "change-me-to-a-random-secret-key",
            "POSTGRES_PASSWORD": "password",
            "RABBITMQ_PASSWORD": "change-me-to-a-strong-password",
            "RABBITMQ_ERLANG_COOKIE": "change-me-in-production",
            "REDIS_PASSWORD": "secret",
            "FLOWER_PASSWORD": "admin",
            "METRICS_API_KEY": "test",
            "OCR_API_KEY": "change-me-to-a-real-key",
            "S3_ACCESS_KEY": "minioadmin",
            "S3_SECRET_KEY": "minioadmin",
            "MINIO_ROOT_USER": "admin",
            "MINIO_ROOT_PASSWORD": "password",
        }
        reports = checklist_script.assess_env(env_vars)
        assert len(reports) == 12
        assert all(r.has_value for r in reports)
        assert all(r.is_insecure for r in reports), (
            f"Expected all insecure, got: "
            f"{[(r.key, r.is_insecure) for r in reports if not r.is_insecure]}"
        )

    def test_assess_env_detects_missing_keys(self):
        """assess_env flags missing keys correctly."""
        env_vars = {}
        reports = checklist_script.assess_env(env_vars)
        assert len(reports) == 12
        assert all(not r.has_value for r in reports)
        assert all(not r.is_insecure for r in reports)
        assert all("MISSING" in r.recommendation for r in reports)

    def test_json_report_format(self):
        """build_json_report produces valid JSON with expected schema."""
        env_vars = {
            "DJANGO_SECRET_KEY": "real-secure-key-abc",
            "POSTGRES_PASSWORD": "password",  # insecure
            # Everything else missing
        }
        reports = checklist_script.assess_env(env_vars)
        report = checklist_script.build_json_report(reports, "coordinator/.env")

        assert "env_file" in report
        assert report["env_file"] == "coordinator/.env"
        assert "all_secure" in report
        assert report["all_secure"] is False  # has insecure + missing
        assert "summary" in report
        assert report["summary"]["total"] == 12
        assert report["summary"]["secure"] == 1  # only DJANGO_SECRET_KEY
        assert report["summary"]["insecure"] == 1  # POSTGRES_PASSWORD
        assert report["summary"]["missing"] == 10  # everything else
        assert "keys" in report
        assert len(report["keys"]) == 12

        # Validate each key entry has required fields
        for key_entry in report["keys"]:
            assert "key" in key_entry
            assert "has_value" in key_entry
            assert "is_insecure" in key_entry
            assert "recommendation" in key_entry

    def test_json_report_all_secure(self):
        """build_json_report sets all_secure=True when everything is secure."""
        env_vars = {k: f"secure-value-{i}-xyz" for i, k in enumerate(checklist_script.SECURITY_RELEVANT_KEYS)}
        reports = checklist_script.assess_env(env_vars)
        report = checklist_script.build_json_report(reports, "test.env")
        assert report["all_secure"] is True
        assert report["summary"]["insecure"] == 0
        assert report["summary"]["missing"] == 0

    def test_json_report_serializable(self):
        """build_json_report output is JSON-serializable."""
        env_vars = {"DJANGO_SECRET_KEY": "real-key"}
        reports = checklist_script.assess_env(env_vars)
        report = checklist_script.build_json_report(reports, "test.env")
        # Should not raise
        json_str = json.dumps(report, indent=2)
        parsed = json.loads(json_str)
        assert parsed["env_file"] == "test.env"

    def test_run_returns_zero_for_all_secure(self, tmp_path):
        """run() returns 0 when all security-relevant keys are secure."""
        content_lines = []
        for i, key in enumerate(checklist_script.SECURITY_RELEVANT_KEYS):
            content_lines.append(f"{key}=secure-value-{i}-xYz789")
        env_file = _write_env(tmp_path, "\n".join(content_lines) + "\n")
        result = checklist_script.run(env_path=env_file)
        assert result == 0

    def test_run_returns_one_for_insecure(self, tmp_path):
        """run() returns 1 when insecure values are found."""
        env_file = _write_env(tmp_path, (
            "DJANGO_SECRET_KEY=change-me\n"
            "POSTGRES_PASSWORD=password\n"
        ))
        result = checklist_script.run(env_path=env_file)
        assert result == 1

    def test_run_returns_one_for_missing(self, tmp_path):
        """run() returns 1 when security-relevant keys are missing."""
        env_file = _write_env(tmp_path, "UNRELATED_KEY=value\n")
        result = checklist_script.run(env_path=env_file)
        assert result == 1

    def test_run_returns_two_for_missing_file(self, tmp_path):
        """run() returns 2 when env file does not exist."""
        result = checklist_script.run(env_path=tmp_path / "nonexistent.env")
        assert result == 2

    def test_run_writes_json_report(self, tmp_path):
        """run() writes a JSON report file when --json-report is given."""
        content_lines = []
        for i, key in enumerate(checklist_script.SECURITY_RELEVANT_KEYS):
            content_lines.append(f"{key}=secure-value-{i}-xYz789")
        env_file = _write_env(tmp_path, "\n".join(content_lines) + "\n")
        report_path = tmp_path / "report.json"

        result = checklist_script.run(env_path=env_file, json_report=report_path)
        assert result == 0
        assert report_path.exists()

        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["all_secure"] is True
        assert len(data["keys"]) == 12

    def test_no_credential_values_in_output(self, tmp_path, capsys):
        """Verify that actual credential values never appear in stdout output."""
        secret_value = "SUPER_SECRET_DO_NOT_LEAK_xK9z42Qm"
        env_file = _write_env(tmp_path, f"DJANGO_SECRET_KEY={secret_value}\n")

        checklist_script.run(env_path=env_file)
        captured = capsys.readouterr()
        assert secret_value not in captured.out
        assert secret_value not in captured.err

    def test_is_insecure_value_helper(self):
        """_is_insecure_value correctly identifies insecure values."""
        assert checklist_script._is_insecure_value("password") is True
        assert checklist_script._is_insecure_value("Password") is True
        assert checklist_script._is_insecure_value("change-me-in-production") is True
        assert checklist_script._is_insecure_value("change-me-to-something") is True
        assert checklist_script._is_insecure_value("minioadmin") is True
        assert checklist_script._is_insecure_value("admin") is True
        assert checklist_script._is_insecure_value("test") is True
        assert checklist_script._is_insecure_value("secret") is True
        assert checklist_script._is_insecure_value("your_api_key") is True
        assert checklist_script._is_insecure_value("placeholder_value") is True
        assert checklist_script._is_insecure_value("example-key") is True
        # Secure values
        assert checklist_script._is_insecure_value("xK9z-real-secret-2024") is False
        assert checklist_script._is_insecure_value("s3-prod-access-key-xK9z42Qm") is False

    def test_parse_args_defaults(self):
        """Default arguments are correctly set."""
        args = checklist_script.parse_args([])
        assert args.env_file == Path("coordinator/.env")
        assert args.json_report is None

    def test_parse_args_custom(self):
        """Custom arguments are correctly parsed."""
        args = checklist_script.parse_args([
            "--env-file", "custom/.env",
            "--json-report", "output/report.json",
        ])
        assert args.env_file == Path("custom/.env")
        assert args.json_report == Path("output/report.json")
