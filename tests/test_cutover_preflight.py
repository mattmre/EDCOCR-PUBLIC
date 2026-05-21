"""Tests for the production cutover pre-flight checker.

Covers credential strength validation, connectivity probes, deployment
state warnings, drain checks, Docker image verification, output formats,
and the overall orchestration logic.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure scripts/ is importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cutover_preflight import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    _extract_host_port,
    _has_repeated_chars,
    _has_sequential_pattern,
    _parse_env_file,
    _shannon_entropy,
    check_backup_recency,
    check_credential_posture,
    check_credential_strength,
    check_deployment_state,
    check_docker_images,
    check_env_completeness,
    check_pipeline_drain,
    check_service_connectivity,
    compute_verdict,
    main,
    render_json,
    render_markdown,
    render_text,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_env(tmp_path: Path, content: str) -> Path:
    """Write a .env file and return its path."""
    env_file = tmp_path / ".env"
    env_file.write_text(content, encoding="utf-8")
    return env_file


def _strong_env_vars() -> dict[str, str]:
    """Return env vars with strong credentials (all checks should pass)."""
    return {
        "DJANGO_SECRET_KEY": "Xk9v3rP7mQ2wL5nT8jF1bR4hG6yC0aD9eI3kM7oS2uN4xZ8vB1lW5qE6tJ0pA3nH7dU",
        "POSTGRES_PASSWORD": "Z4rT9wQ2xL7mP3nK8jF5hG0bR6yC1aD2eI",
        "RABBITMQ_PASSWORD": "W7xK3mQ9vL5nT2jF8bR4hG0yC6aD1eI3p",
        "RABBITMQ_ERLANG_COOKIE": "K9v3rP7mQ2wL5nT8jF1bR4hG6yC0aD9eI3kM7o",
        "REDIS_PASSWORD": "T8jF1bR4hG6yC0aD9eI3kM7oS2uN4xZ8",
        "FLOWER_PASSWORD": "G6yC0aD9eI3kM7oS2uN4xZ8vB1lW5qE6",
        "METRICS_API_KEY": "C0aD9eI3kM7oS2uN4xZ8vB1lW5qE6tJ0pA3nH7dU",
        "OCR_API_KEY": "D9eI3kM7oS2uN4xZ8vB1lW5qE6tJ0pA3nH7dU9v3r",
        "S3_ACCESS_KEY": "ocrlocal-707a3eeb4c68",
        "S3_SECRET_KEY": "VouJ9g4in3SvgFaua8BklDc3xQ_1xTzbxRT4Zy1wRmg",
        "MINIO_ROOT_USER": "ocrlocal-707a3eeb4c68",
        "MINIO_ROOT_PASSWORD": "VouJ9g4in3SvgFaua8BklDc3xQ_1xTzbxRT4Zy1wRmg",
        "DEPLOYMENT_ENV": "production",
        "PRODUCTION_READINESS_ACK": "true",
        "DJANGO_DEBUG": "False",
        "DATABASE_URL": "postgres://ocr:pass@localhost:5432/ocr_coordinator",
        "CELERY_BROKER_URL": "amqp://user:pass@localhost:5672//",
        "S3_ENDPOINT": "http://localhost:9000",
    }


# ---------------------------------------------------------------------------
# Test: Shannon entropy
# ---------------------------------------------------------------------------

class TestShannonEntropy:
    def test_empty_string(self):
        assert _shannon_entropy("") == 0.0

    def test_single_char_repeated(self):
        # All same char => 0 entropy
        assert _shannon_entropy("aaaaaaa") == 0.0

    def test_uniform_distribution(self):
        # "ab" repeated -> 2 equally-likely chars -> entropy = 1.0
        ent = _shannon_entropy("abababab")
        assert abs(ent - 1.0) < 0.01

    def test_high_entropy(self):
        # Random-looking string should have high entropy
        ent = _shannon_entropy("Xk9v3rP7mQ2wL5nT8jF1bR4hG6yC0aD9eI3")
        assert ent > 3.0


# ---------------------------------------------------------------------------
# Test: Sequential pattern detection
# ---------------------------------------------------------------------------

class TestSequentialPattern:
    def test_no_pattern(self):
        assert not _has_sequential_pattern("Xk9v3rP7mQ2w")

    def test_ascending_chars(self):
        assert _has_sequential_pattern("xxabcdexx")

    def test_ascending_digits(self):
        assert _has_sequential_pattern("xx12345xx")

    def test_short_string(self):
        assert not _has_sequential_pattern("ab")

    def test_exact_threshold(self):
        assert _has_sequential_pattern("abcde", run_length=5)
        assert not _has_sequential_pattern("abcd", run_length=5)


# ---------------------------------------------------------------------------
# Test: Repeated character detection
# ---------------------------------------------------------------------------

class TestRepeatedChars:
    def test_no_repetition(self):
        assert not _has_repeated_chars("abcdefgh")

    def test_high_repetition(self):
        assert _has_repeated_chars("aaaaaab")  # 'a' is 6/7 = 85%

    def test_empty(self):
        assert not _has_repeated_chars("")

    def test_threshold_boundary(self):
        # 6/10 = 60% -- at boundary, not above
        assert not _has_repeated_chars("aaaaaabbbb", threshold=0.6)
        # 7/10 = 70% -- above
        assert _has_repeated_chars("aaaaaaabbb", threshold=0.6)


# ---------------------------------------------------------------------------
# Test: URL host/port extraction
# ---------------------------------------------------------------------------

class TestExtractHostPort:
    def test_postgres_url(self):
        result = _extract_host_port("postgres://user:pass@dbhost:5432/mydb")
        assert result == ("dbhost", 5432)

    def test_amqp_url(self):
        result = _extract_host_port("amqp://user:pass@broker:5672//")
        assert result == ("broker", 5672)

    def test_redis_url(self):
        result = _extract_host_port("redis://:password@redis-host:6379/0")
        assert result == ("redis-host", 6379)

    def test_http_url(self):
        result = _extract_host_port("http://minio:9000")
        assert result == ("minio", 9000)

    def test_https_default_port(self):
        result = _extract_host_port("https://s3.example.com/bucket")
        assert result == ("s3.example.com", 443)

    def test_postgres_default_port(self):
        result = _extract_host_port("postgres://user:pass@dbhost/mydb")
        assert result == ("dbhost", 5432)

    def test_db_postgresql_scheme(self):
        result = _extract_host_port("db+postgresql://user:pass@dbhost:5432/mydb")
        assert result == ("dbhost", 5432)

    def test_db_postgresql_default_port(self):
        result = _extract_host_port("db+postgresql://user:pass@dbhost/mydb")
        assert result == ("dbhost", 5432)

    def test_invalid_url(self):
        assert _extract_host_port("not-a-url") is None

    def test_empty_string(self):
        assert _extract_host_port("") is None


# ---------------------------------------------------------------------------
# Test: Credential Strength Validation
# ---------------------------------------------------------------------------

class TestCredentialStrength:
    def test_strong_credentials(self):
        env = _strong_env_vars()
        results = check_credential_strength(env)
        for r in results:
            assert r["status"] == PASS, f"{r['name']}: {r['details']}"

    def test_short_django_key(self):
        env = _strong_env_vars()
        env["DJANGO_SECRET_KEY"] = "short-key-123"  # < 50 chars
        results = check_credential_strength(env)
        django_result = next(r for r in results if "DJANGO_SECRET_KEY" in r["name"])
        assert django_result["status"] == FAIL
        assert "length" in django_result["details"]

    def test_short_password(self):
        env = _strong_env_vars()
        env["POSTGRES_PASSWORD"] = "abc123"  # < 16 chars
        results = check_credential_strength(env)
        pg_result = next(r for r in results if "POSTGRES_PASSWORD" in r["name"])
        assert pg_result["status"] == FAIL
        assert "length" in pg_result["details"]

    def test_low_entropy(self):
        env = _strong_env_vars()
        env["REDIS_PASSWORD"] = "aaaaaaaaaaaaaaaa"  # 16 chars but 0 entropy
        results = check_credential_strength(env)
        redis_result = next(r for r in results if "REDIS_PASSWORD" in r["name"])
        assert redis_result["status"] == FAIL
        assert "entropy" in redis_result["details"]

    def test_sequential_pattern(self):
        env = _strong_env_vars()
        env["FLOWER_PASSWORD"] = "prefix_abcdefg_suffix1234"
        results = check_credential_strength(env)
        flower_result = next(r for r in results if "FLOWER_PASSWORD" in r["name"])
        assert flower_result["status"] == FAIL
        assert "sequential" in flower_result["details"]

    def test_missing_credential_skipped(self):
        # Missing keys are not checked by strength validator (posture handles that)
        results = check_credential_strength({})
        assert len(results) == 0

    def test_repeated_chars(self):
        env = _strong_env_vars()
        # 21 chars, above min length, but one char dominates
        env["POSTGRES_PASSWORD"] = "aaaaaaaaaaaaaaaaabbbb"
        results = check_credential_strength(env)
        pg_result = next(r for r in results if "POSTGRES_PASSWORD" in r["name"])
        assert pg_result["status"] == FAIL
        assert "dominates" in pg_result["details"] or "entropy" in pg_result["details"]

    def test_identifier_keys_skip_entropy(self):
        """S3_ACCESS_KEY and MINIO_ROOT_USER are identifiers, not secrets.

        They should pass with low entropy as long as they meet length requirements.
        """
        env = _strong_env_vars()
        # Use a low-entropy but sufficiently long identifier
        env["S3_ACCESS_KEY"] = "ocrlocal-prod-user"  # 18 chars, low entropy
        env["MINIO_ROOT_USER"] = "ocrlocal-prod-user"  # 18 chars, low entropy
        results = check_credential_strength(env)
        s3_result = next(r for r in results if "S3_ACCESS_KEY" in r["name"])
        minio_result = next(r for r in results if "MINIO_ROOT_USER" in r["name"])
        assert s3_result["status"] == PASS, f"S3_ACCESS_KEY: {s3_result['details']}"
        assert minio_result["status"] == PASS, f"MINIO_ROOT_USER: {minio_result['details']}"


# ---------------------------------------------------------------------------
# Test: Service Connectivity Probes
# ---------------------------------------------------------------------------

class TestServiceConnectivity:
    def test_skip_when_url_not_set(self):
        results = check_service_connectivity({})
        assert all(r["status"] == SKIP for r in results)

    def test_pass_on_successful_connect(self):
        env = {"DATABASE_URL": "postgres://user:pass@db:5432/mydb"}
        with patch("cutover_preflight.socket.socket") as mock_sock:
            instance = MagicMock()
            mock_sock.return_value = instance
            results = check_service_connectivity(env)
        db_results = [r for r in results if "DATABASE_URL" in r["name"]]
        assert len(db_results) == 1
        assert db_results[0]["status"] == PASS
        instance.connect.assert_called_once_with(("db", 5432))
        instance.close.assert_called_once()

    def test_fail_on_timeout(self):
        env = {"DATABASE_URL": "postgres://user:pass@db:5432/mydb"}
        with patch("cutover_preflight.socket.socket") as mock_sock:
            instance = MagicMock()
            instance.connect.side_effect = socket.timeout("timed out")
            mock_sock.return_value = instance
            results = check_service_connectivity(env)
        db_results = [r for r in results if "DATABASE_URL" in r["name"]]
        assert db_results[0]["status"] == FAIL
        assert "timed out" in db_results[0]["details"]

    def test_socket_closed_on_connect_failure(self):
        """Verify socket.close() is called even when connect raises."""
        env = {"DATABASE_URL": "postgres://user:pass@db:5432/mydb"}
        with patch("cutover_preflight.socket.socket") as mock_sock:
            instance = MagicMock()
            instance.connect.side_effect = ConnectionRefusedError("refused")
            mock_sock.return_value = instance
            check_service_connectivity(env)
        instance.close.assert_called_once()

    def test_fail_on_connection_refused(self):
        env = {"S3_ENDPOINT": "http://minio:9000"}
        with patch("cutover_preflight.socket.socket") as mock_sock:
            instance = MagicMock()
            instance.connect.side_effect = ConnectionRefusedError("refused")
            mock_sock.return_value = instance
            results = check_service_connectivity(env)
        s3_results = [r for r in results if "S3_ENDPOINT" in r["name"]]
        assert s3_results[0]["status"] == FAIL

    def test_warn_on_unparseable_url(self):
        env = {"DATABASE_URL": "not-a-url"}
        results = check_service_connectivity(env)
        db_results = [r for r in results if "DATABASE_URL" in r["name"]]
        assert db_results[0]["status"] == WARN

    def test_multiple_services(self):
        env = {
            "DATABASE_URL": "postgres://u:p@db:5432/x",
            "CELERY_BROKER_URL": "amqp://u:p@broker:5672//",
            "S3_ENDPOINT": "http://minio:9000",
        }
        with patch("cutover_preflight.socket.socket") as mock_sock:
            instance = MagicMock()
            mock_sock.return_value = instance
            results = check_service_connectivity(env)
        passed = [r for r in results if r["status"] == PASS]
        assert len(passed) == 3


# ---------------------------------------------------------------------------
# Test: Deployment State
# ---------------------------------------------------------------------------

class TestDeploymentState:
    def test_production_ready(self):
        env = {"DEPLOYMENT_ENV": "production", "PRODUCTION_READINESS_ACK": "true", "DJANGO_DEBUG": "False"}
        results = check_deployment_state(env)
        state_r = next(r for r in results if r["name"] == "deployment_state")
        assert state_r["status"] == PASS
        debug_r = next(r for r in results if r["name"] == "deployment_debug")
        assert debug_r["status"] == PASS

    def test_production_no_ack(self):
        env = {"DEPLOYMENT_ENV": "production", "PRODUCTION_READINESS_ACK": "false", "DJANGO_DEBUG": "False"}
        results = check_deployment_state(env)
        state_r = next(r for r in results if r["name"] == "deployment_state")
        assert state_r["status"] == WARN

    def test_staging(self):
        env = {"DEPLOYMENT_ENV": "staging", "DJANGO_DEBUG": "False"}
        results = check_deployment_state(env)
        state_r = next(r for r in results if r["name"] == "deployment_state")
        assert state_r["status"] == WARN
        assert "staging" in state_r["details"]

    def test_debug_enabled(self):
        env = {"DEPLOYMENT_ENV": "production", "PRODUCTION_READINESS_ACK": "true", "DJANGO_DEBUG": "true"}
        results = check_deployment_state(env)
        debug_r = next(r for r in results if r["name"] == "deployment_debug")
        assert debug_r["status"] == FAIL

    def test_no_deployment_env(self):
        env = {"DJANGO_DEBUG": "False"}
        results = check_deployment_state(env)
        state_r = next(r for r in results if r["name"] == "deployment_state")
        assert state_r["status"] == WARN


# ---------------------------------------------------------------------------
# Test: Pipeline Drain
# ---------------------------------------------------------------------------

class TestPipelineDrain:
    def test_skip_when_no_coordinator_url(self):
        results = check_pipeline_drain({})
        assert results[0]["status"] == SKIP

    def test_skip_when_coordinator_unreachable(self):
        env = {"COORDINATOR_URL": "http://unreachable:8000"}
        # urlopen will fail since host doesn't exist
        results = check_pipeline_drain(env)
        assert results[0]["status"] == SKIP

    def test_pass_when_no_active_jobs(self):
        env = {"COORDINATOR_URL": "http://localhost:8000"}
        response_data = json.dumps({"active_jobs": 0}).encode()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = response_data
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            results = check_pipeline_drain(env)
        assert results[0]["status"] == PASS

    def test_warn_when_active_jobs(self):
        env = {"COORDINATOR_URL": "http://localhost:8000"}
        response_data = json.dumps({"active_jobs": 5}).encode()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = response_data
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            results = check_pipeline_drain(env)
        assert results[0]["status"] == WARN
        assert "5" in results[0]["details"]

    def test_fail_when_require_drain(self):
        env = {"COORDINATOR_URL": "http://localhost:8000"}
        response_data = json.dumps({"active_jobs": 3}).encode()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = response_data
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            results = check_pipeline_drain(env, require_drain=True)
        assert results[0]["status"] == FAIL

    def test_api_host_fallback(self):
        env = {"API_HOST": "localhost", "API_PORT": "9000"}
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("unreachable")
            results = check_pipeline_drain(env)
        assert results[0]["status"] == SKIP


# ---------------------------------------------------------------------------
# Test: Docker Image Check
# ---------------------------------------------------------------------------

class TestDockerImages:
    def test_docker_not_found(self):
        with patch("cutover_preflight.subprocess.run", side_effect=FileNotFoundError):
            results = check_docker_images()
        assert results[0]["status"] == SKIP
        assert "not found" in results[0]["details"]

    def test_docker_command_fails(self):
        proc = MagicMock()
        proc.returncode = 1
        with patch("cutover_preflight.subprocess.run", return_value=proc):
            results = check_docker_images()
        assert results[0]["status"] == SKIP

    def test_no_ocr_images(self):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "nginx:latest\nredis:7\n"
        with patch("cutover_preflight.subprocess.run", return_value=proc):
            results = check_docker_images()
        assert results[0]["status"] == WARN

    def test_ocr_images_found(self):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "ocr-local:latest\nocr-worker:1.2.0\nredis:7\n"
        with patch("cutover_preflight.subprocess.run", return_value=proc):
            results = check_docker_images()
        assert results[0]["status"] == PASS
        assert "2" in results[0]["details"]

    def test_version_check_matches(self):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "ocr-local:1.2.0\nocr-worker:1.2.0\n"
        with patch("cutover_preflight.subprocess.run", return_value=proc):
            with patch.dict("sys.modules", {}):
                # version.py should be importable from PROJECT_ROOT
                results = check_docker_images(check_version=True)
        # Should have at least the base image check
        assert any(r["status"] == PASS for r in results)

    def test_docker_timeout(self):
        with patch("cutover_preflight.subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 15)):
            results = check_docker_images()
        assert results[0]["status"] == SKIP
        assert "timed out" in results[0]["details"]


# ---------------------------------------------------------------------------
# Test: Credential Posture (delegates to existing module)
# ---------------------------------------------------------------------------

class TestCredentialPosture:
    def test_delegation_success(self):
        env = _strong_env_vars()
        results = check_credential_posture(env)
        # Should have one result per SECURITY_RELEVANT_KEYS
        assert len(results) > 0
        # All strong creds should pass
        for r in results:
            if r["status"] != SKIP:
                assert r["status"] == PASS, f"{r['name']}: {r['details']}"

    def test_delegation_with_missing_key(self):
        env = {"DJANGO_SECRET_KEY": ""}  # present but empty
        results = check_credential_posture(env)
        # Should detect missing/empty credentials
        fail_results = [r for r in results if r["status"] == FAIL]
        assert len(fail_results) > 0

    def test_import_failure_skips(self):
        env = _strong_env_vars()
        with patch.dict("sys.modules", {"credential_cutover_checklist": None}):
            with patch("builtins.__import__", side_effect=ImportError("mocked")):
                # This won't fully work because the function uses from-import
                # Instead, mock at the function level
                pass

        # Direct test: if credential_cutover_checklist is not importable
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        def mock_import(name, *args, **kwargs):
            if name == "credential_cutover_checklist":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            results = check_credential_posture(env)
        assert any(r["status"] == SKIP for r in results)


# ---------------------------------------------------------------------------
# Test: Env Completeness
# ---------------------------------------------------------------------------

class TestEnvCompleteness:
    def test_delegation_with_valid_env(self, tmp_path):
        env_file = _write_env(tmp_path, "\n".join([
            "DJANGO_SECRET_KEY=Xk9v3rP7mQ2wL5nT8jF1bR4hG6yC0aD9eI3kM7oS2uN4xZ8vB1lW5qE6tJ0pA3nH7dU",
            "POSTGRES_PASSWORD=Z4rT9wQ2xL7mP3nK8jF5hG0bR6yC1aD2eI",
            "RABBITMQ_PASSWORD=W7xK3mQ9vL5nT2jF8bR4hG0yC6aD1eI3p",
            "RABBITMQ_ERLANG_COOKIE=K9v3rP7mQ2wL5nT8jF1bR4hG6yC0aD9eI3kM7o",
            "DATABASE_URL=postgres://ocr:Z4rT9wQ2xL7mP3nK8jF5hG0bR6yC1aD2eI@postgres:5432/ocr_coordinator",
            "CELERY_BROKER_URL=amqp://ocr_user:W7xK3mQ9vL5nT2jF8bR4hG0yC6aD1eI3p@rabbitmq:5672//",
            "FLOWER_PASSWORD=73bQZmA2uXh64hLRBziGsSCAs4x0pkbeXDSr0UCuyNg",
            "METRICS_API_KEY=ivd1y967XRBmsYcGwOpDKLpqxLrAm5desALWU9Mvrm8",
            "OCR_API_KEY=C0aD9eI3kM7oS2uN4xZ8vB1lW5qE6tJ0pA3nH7dU9v3r",
        ]))
        results = check_env_completeness(env_file)
        assert len(results) > 0

    def test_delegation_with_missing_file(self, tmp_path):
        env_file = tmp_path / "nonexistent.env"
        results = check_env_completeness(env_file)
        assert len(results) > 0
        # Should fail or skip
        assert any(r["status"] in (FAIL, SKIP) for r in results)


# ---------------------------------------------------------------------------
# Test: Backup Recency
# ---------------------------------------------------------------------------

class TestBackupRecency:
    def test_no_backup_dir(self):
        results = check_backup_recency({})
        assert results[0]["status"] == SKIP

    def test_recent_backup(self, tmp_path):
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        backup_file = backup_dir / "db_backup.sql"
        backup_file.write_text("-- backup data")
        results = check_backup_recency({"BACKUP_DIR": str(backup_dir)})
        assert results[0]["status"] == PASS

    def test_old_backup(self, tmp_path):
        import time
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        backup_file = backup_dir / "db_backup.sql"
        backup_file.write_text("-- old backup")
        # Set modification time to 48 hours ago
        old_time = time.time() - (48 * 3600)
        import os
        os.utime(str(backup_file), (old_time, old_time))
        results = check_backup_recency({"BACKUP_DIR": str(backup_dir)})
        assert results[0]["status"] == WARN
        assert ">24h" in results[0]["details"]


# ---------------------------------------------------------------------------
# Test: Verdict computation
# ---------------------------------------------------------------------------

class TestComputeVerdict:
    def test_all_pass(self):
        results = [
            {"status": PASS, "name": "a", "details": "ok", "remediation": None},
            {"status": PASS, "name": "b", "details": "ok", "remediation": None},
        ]
        assert compute_verdict(results) == "READY"

    def test_all_pass_with_skip(self):
        results = [
            {"status": PASS, "name": "a", "details": "ok", "remediation": None},
            {"status": SKIP, "name": "b", "details": "skip", "remediation": None},
        ]
        assert compute_verdict(results) == "READY"

    def test_any_fail(self):
        results = [
            {"status": PASS, "name": "a", "details": "ok", "remediation": None},
            {"status": FAIL, "name": "b", "details": "bad", "remediation": "fix"},
        ]
        assert compute_verdict(results) == "NOT READY"

    def test_fail_overrides_warn(self):
        results = [
            {"status": WARN, "name": "a", "details": "meh", "remediation": None},
            {"status": FAIL, "name": "b", "details": "bad", "remediation": "fix"},
        ]
        assert compute_verdict(results) == "NOT READY"

    def test_warn_only(self):
        results = [
            {"status": PASS, "name": "a", "details": "ok", "remediation": None},
            {"status": WARN, "name": "b", "details": "meh", "remediation": None},
        ]
        assert compute_verdict(results) == "PARTIAL"

    def test_empty_results(self):
        assert compute_verdict([]) == "READY"


# ---------------------------------------------------------------------------
# Test: JSON output
# ---------------------------------------------------------------------------

class TestJsonOutput:
    def test_json_structure(self):
        results = [
            {"name": "check_a", "status": PASS, "details": "good", "remediation": None},
            {"name": "check_b", "status": FAIL, "details": "bad", "remediation": "fix it"},
        ]
        data = render_json(results, "NOT READY", "test.env")
        assert data["verdict"] == "NOT READY"
        assert data["env_file"] == "test.env"
        assert data["summary"]["pass"] == 1
        assert data["summary"]["fail"] == 1
        assert data["summary"]["total"] == 2
        assert len(data["checks"]) == 2
        assert "timestamp" in data

    def test_json_serializable(self):
        results = [{"name": "x", "status": PASS, "details": "ok", "remediation": None}]
        data = render_json(results, "READY", "test.env")
        # Must not raise
        json.dumps(data)


# ---------------------------------------------------------------------------
# Test: Markdown output
# ---------------------------------------------------------------------------

class TestMarkdownOutput:
    def test_markdown_structure(self):
        results = [
            {"name": "check_a", "status": PASS, "details": "good", "remediation": None},
            {"name": "check_b", "status": FAIL, "details": "bad", "remediation": "fix it"},
        ]
        md = render_markdown(results, "NOT READY", "test.env")
        assert "# Production Cutover Pre-Flight Report" in md
        assert "NOT READY" in md
        assert "| `check_a`" in md
        assert "## Remediation Required" in md
        assert "fix it" in md

    def test_markdown_no_remediation_section(self):
        results = [
            {"name": "check_a", "status": PASS, "details": "good", "remediation": None},
        ]
        md = render_markdown(results, "READY", "test.env")
        assert "Remediation Required" not in md


# ---------------------------------------------------------------------------
# Test: Text output
# ---------------------------------------------------------------------------

class TestTextOutput:
    def test_text_structure(self):
        results = [
            {"name": "check_a", "status": PASS, "details": "good", "remediation": None},
            {"name": "check_b", "status": FAIL, "details": "bad", "remediation": "fix it"},
        ]
        text = render_text(results, "NOT READY")
        assert "Pre-Flight Report" in text
        assert "[PASS]" in text
        assert "[FAIL]" in text
        assert "NOT READY" in text
        assert "-> fix it" in text


# ---------------------------------------------------------------------------
# Test: Overall orchestration
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_missing_env_file(self, tmp_path):
        env_file = tmp_path / "nonexistent.env"
        results = run_all_checks(env_file)
        assert results[0]["status"] == FAIL
        assert "not found" in results[0]["details"]

    def test_valid_env_with_skipped_connectivity(self, tmp_path):
        env_file = _write_env(tmp_path, "\n".join([
            f"DJANGO_SECRET_KEY={'A' * 60}",
            f"POSTGRES_PASSWORD={'B' * 20}",
            "DEPLOYMENT_ENV=development",
            "DJANGO_DEBUG=False",
        ]))
        results = run_all_checks(env_file, skip_connectivity=True)
        # Should have env_file PASS and various checks
        assert results[0]["status"] == PASS
        assert results[0]["name"] == "env_file"
        # Connectivity should be skipped
        conn_results = [r for r in results if r["name"] == "connectivity"]
        assert len(conn_results) == 1
        assert conn_results[0]["status"] == SKIP


# ---------------------------------------------------------------------------
# Test: parse_env_file
# ---------------------------------------------------------------------------

class TestParseEnvFile:
    def test_basic_parsing(self, tmp_path):
        env_file = _write_env(tmp_path, "KEY1=value1\nKEY2=value2\n")
        result = _parse_env_file(env_file)
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_comments_and_blanks(self, tmp_path):
        env_file = _write_env(tmp_path, "# comment\n\nKEY=val\n")
        result = _parse_env_file(env_file)
        assert result == {"KEY": "val"}

    def test_nonexistent_file(self, tmp_path):
        result = _parse_env_file(tmp_path / "nope.env")
        assert result == {}

    def test_double_quoted_value(self, tmp_path):
        env_file = _write_env(tmp_path, 'KEY="my value"\n')
        result = _parse_env_file(env_file)
        assert result == {"KEY": "my value"}

    def test_single_quoted_value(self, tmp_path):
        env_file = _write_env(tmp_path, "KEY='my value'\n")
        result = _parse_env_file(env_file)
        assert result == {"KEY": "my value"}

    def test_unquoted_value_unchanged(self, tmp_path):
        env_file = _write_env(tmp_path, "KEY=no_quotes\n")
        result = _parse_env_file(env_file)
        assert result == {"KEY": "no_quotes"}

    def test_mismatched_quotes_unchanged(self, tmp_path):
        env_file = _write_env(tmp_path, "KEY=\"mixed'\n")
        result = _parse_env_file(env_file)
        # Mismatched quotes should not be stripped
        assert result == {"KEY": "\"mixed'"}


# ---------------------------------------------------------------------------
# Test: CLI main
# ---------------------------------------------------------------------------

class TestCLIMain:
    def test_missing_env_file_exit_code(self, tmp_path):
        rc = main(["--env-file", str(tmp_path / "nope.env"), "--skip-connectivity"])
        assert rc == 1  # NOT READY (FAIL on missing file)

    def test_json_output_flag(self, tmp_path, capsys):
        env_file = _write_env(tmp_path, "DEPLOYMENT_ENV=production\nDJANGO_DEBUG=False\n")
        main(["--env-file", str(env_file), "--json", "--skip-connectivity"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "verdict" in data
        assert "checks" in data

    def test_report_flag(self, tmp_path):
        env_file = _write_env(tmp_path, "DEPLOYMENT_ENV=production\nDJANGO_DEBUG=False\n")
        report_file = tmp_path / "report.md"
        main(["--env-file", str(env_file), "--report", str(report_file), "--skip-connectivity"])
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "Pre-Flight Report" in content
