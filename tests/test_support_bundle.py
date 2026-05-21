"""Unit tests for scripts/support_bundle.py.

Covers system info collection, config snapshot with secret redaction,
log tail with mock log files, health check delegation via mocked
subprocesses, manifest generation, and JSON summary output.

Run with: python -m pytest tests/test_support_bundle.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.support_bundle import (
    _is_secret_key,
    _mask_value,
    _run_command,
    _safe_disk_usage,
    collect_config_snapshot,
    collect_dependency_versions,
    collect_git_status,
    collect_health_status,
    collect_log_tail,
    collect_system_info,
    create_bundle,
    main,
)

# ---------------------------------------------------------------------------
# Secret masking tests
# ---------------------------------------------------------------------------


class TestSecretMasking:
    """Tests for _is_secret_key and _mask_value."""

    def test_secret_key_password(self):
        assert _is_secret_key("POSTGRES_PASSWORD") is True

    def test_secret_key_token(self):
        assert _is_secret_key("OCR_API_TOKEN") is True

    def test_secret_key_secret(self):
        assert _is_secret_key("DJANGO_SECRET_KEY") is True

    def test_secret_key_api_key(self):
        assert _is_secret_key("METRICS_API_KEY") is True

    def test_non_secret_key(self):
        assert _is_secret_key("OCR_DPI") is False

    def test_non_secret_enable(self):
        assert _is_secret_key("ENABLE_DOCINTEL") is False

    def test_case_insensitive(self):
        assert _is_secret_key("my_password_field") is True
        assert _is_secret_key("some_token_here") is True

    def test_mask_value_redacts_secret(self):
        assert _mask_value("OCR_API_KEY", "super-secret-123") == "***REDACTED***"

    def test_mask_value_preserves_non_secret(self):
        assert _mask_value("OCR_DPI", "300") == "300"


# ---------------------------------------------------------------------------
# _run_command tests
# ---------------------------------------------------------------------------


class TestRunCommand:
    """Tests for _run_command subprocess wrapper."""

    def test_successful_command(self):
        rc, out = _run_command([sys.executable, "-c", "print('hello')"])
        assert rc == 0
        assert "hello" in out

    def test_command_not_found(self):
        rc, out = _run_command(["nonexistent_binary_xyz_123"])
        assert rc == -1
        assert "not found" in out.lower() or "error" in out.lower()

    def test_command_timeout(self):
        rc, out = _run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
        )
        assert rc == -1
        assert "timed out" in out.lower() or "timeout" in out.lower()

    def test_command_with_stderr(self):
        rc, out = _run_command(
            [sys.executable, "-c", "import sys; sys.stderr.write('warn\\n')"]
        )
        assert rc == 0
        assert "warn" in out


# ---------------------------------------------------------------------------
# System info tests
# ---------------------------------------------------------------------------


class TestCollectSystemInfo:
    """Tests for collect_system_info."""

    def test_contains_python_version(self):
        info = collect_system_info()
        assert "python_version" in info
        assert sys.version in info["python_version"]

    def test_contains_platform(self):
        info = collect_system_info()
        assert "platform" in info
        assert info["platform"]  # non-empty

    def test_contains_architecture(self):
        info = collect_system_info()
        assert "architecture" in info

    def test_contains_gpu_section(self):
        info = collect_system_info()
        assert "gpu" in info

    @patch("scripts.support_bundle._run_command")
    def test_gpu_info_captured(self, mock_run):
        mock_run.return_value = (0, "NVIDIA GeForce RTX 3090, 24576 MiB, 535.86")
        info = collect_system_info()
        assert "nvidia_smi" in info["gpu"]


# ---------------------------------------------------------------------------
# Config snapshot tests
# ---------------------------------------------------------------------------


class TestCollectConfigSnapshot:
    """Tests for collect_config_snapshot."""

    def test_env_vars_captured(self):
        with patch.dict(os.environ, {"OCR_DPI": "300", "ENABLE_DOCINTEL": "true"}, clear=False):
            config = collect_config_snapshot()
            assert "OCR_DPI" in config["env_vars"]
            assert config["env_vars"]["OCR_DPI"] == "300"

    def test_secret_env_vars_redacted(self):
        with patch.dict(os.environ, {"OCR_API_KEY": "super-secret"}, clear=False):
            config = collect_config_snapshot()
            assert config["env_vars"]["OCR_API_KEY"] == "***REDACTED***"

    def test_coordinator_env_keys_only(self, tmp_path):
        env_file = tmp_path / "coordinator" / ".env"
        env_file.parent.mkdir(parents=True)
        env_file.write_text("DJANGO_SECRET_KEY=my-secret\nDATABASE_URL=postgres://...\n# comment\n")
        with patch("scripts.support_bundle.PROJECT_ROOT", tmp_path):
            config = collect_config_snapshot()
            # Keys are listed but values are not
            assert "DJANGO_SECRET_KEY" in config["coordinator_env_keys"]
            assert "DATABASE_URL" in config["coordinator_env_keys"]
            # Values must not appear in the keys list
            assert "my-secret" not in str(config["coordinator_env_keys"])

    def test_docker_compose_files_listed(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text("services: {}")
        with patch("scripts.support_bundle.PROJECT_ROOT", tmp_path):
            config = collect_config_snapshot()
            assert "docker-compose.yml" in config["docker_compose_files"]


# ---------------------------------------------------------------------------
# Dependency versions tests
# ---------------------------------------------------------------------------


class TestCollectDependencyVersions:
    """Tests for collect_dependency_versions."""

    def test_contains_pip_freeze_section(self):
        deps = collect_dependency_versions()
        assert "pip freeze" in deps

    @patch("scripts.support_bundle._run_command")
    def test_docker_version_section(self, mock_run):
        mock_run.return_value = (0, "Docker version 24.0.7")
        deps = collect_dependency_versions()
        # Should contain the Docker section header
        assert "Docker" in deps


# ---------------------------------------------------------------------------
# Log tail tests
# ---------------------------------------------------------------------------


class TestCollectLogTail:
    """Tests for collect_log_tail."""

    def test_log_tail_with_mock_log(self, tmp_path):
        log_dir = tmp_path / "ocr_output" / "logs"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "ocr_pipeline_20260326.log"
        lines = [f"Line {i}" for i in range(200)]
        log_file.write_text("\n".join(lines))
        with patch("scripts.support_bundle.PROJECT_ROOT", tmp_path):
            tail = collect_log_tail()
            # Should contain last 100 lines
            assert "Line 199" in tail
            assert "Line 100" in tail
            # Should NOT contain line 0 (only last 100)
            assert "Line 0\n" not in tail

    def test_log_tail_no_log_dir(self, tmp_path):
        with patch("scripts.support_bundle.PROJECT_ROOT", tmp_path):
            tail = collect_log_tail()
            assert "not found" in tail.lower()

    def test_log_tail_empty_log_dir(self, tmp_path):
        log_dir = tmp_path / "ocr_output" / "logs"
        log_dir.mkdir(parents=True)
        with patch("scripts.support_bundle.PROJECT_ROOT", tmp_path):
            tail = collect_log_tail()
            assert "No pipeline logs found" in tail


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------


class TestCollectHealthStatus:
    """Tests for collect_health_status."""

    @patch("scripts.support_bundle._run_command")
    def test_health_captures_json_output(self, mock_run):
        mock_json = json.dumps({"status": "pass", "checks": []})
        mock_run.return_value = (0, mock_json)
        health = collect_health_status()
        assert "verify_release_state" in health
        assert health["verify_release_state"]["status"] == "pass"

    @patch("scripts.support_bundle._run_command")
    def test_health_handles_failure(self, mock_run):
        mock_run.return_value = (-1, "Script not found")
        health = collect_health_status()
        assert "verify_release_state" in health
        assert "error" in health["verify_release_state"]

    @patch("scripts.support_bundle._run_command")
    def test_health_handles_non_json(self, mock_run):
        mock_run.return_value = (0, "Some preamble text\nNot JSON at all")
        health = collect_health_status()
        assert "verify_release_state" in health
        assert "raw_output" in health["verify_release_state"]


# ---------------------------------------------------------------------------
# Git status tests
# ---------------------------------------------------------------------------


class TestCollectGitStatus:
    """Tests for collect_git_status."""

    def test_git_status_contains_branch(self):
        git_info = collect_git_status()
        assert "Branch:" in git_info

    def test_git_status_contains_commit(self):
        git_info = collect_git_status()
        assert "Commit:" in git_info

    def test_git_status_contains_recent_commits(self):
        git_info = collect_git_status()
        assert "Recent commits" in git_info


# ---------------------------------------------------------------------------
# Bundle creation tests
# ---------------------------------------------------------------------------


class TestCreateBundle:
    """Tests for create_bundle."""

    def test_bundle_creates_directory(self, tmp_path):
        bundle_dir = create_bundle(output_dir=tmp_path)
        assert bundle_dir.exists()
        assert bundle_dir.is_dir()

    def test_bundle_contains_required_files(self, tmp_path):
        bundle_dir = create_bundle(output_dir=tmp_path)
        expected_files = ["system.json", "config.json", "deps.txt", "git.txt", "manifest.json"]
        for f in expected_files:
            assert (bundle_dir / f).exists(), f"Missing {f}"

    def test_bundle_without_optional_files(self, tmp_path):
        bundle_dir = create_bundle(output_dir=tmp_path)
        # Logs and health are opt-in
        assert not (bundle_dir / "logs.txt").exists()
        assert not (bundle_dir / "health.json").exists()

    def test_bundle_with_logs(self, tmp_path):
        bundle_dir = create_bundle(output_dir=tmp_path, include_logs=True)
        assert (bundle_dir / "logs.txt").exists()

    def test_bundle_with_health(self, tmp_path):
        bundle_dir = create_bundle(output_dir=tmp_path, include_health=True)
        assert (bundle_dir / "health.json").exists()

    def test_manifest_has_correct_structure(self, tmp_path):
        bundle_dir = create_bundle(output_dir=tmp_path)
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        assert "bundle_timestamp" in manifest
        assert "files" in manifest
        assert "total_files" in manifest
        assert manifest["total_files"] >= 5  # system, config, deps, git, manifest

    def test_manifest_file_sizes(self, tmp_path):
        bundle_dir = create_bundle(output_dir=tmp_path)
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        for entry in manifest["files"]:
            assert "file" in entry
            assert "size_bytes" in entry
            assert entry["size_bytes"] > 0

    def test_json_summary_output(self, tmp_path, capsys):
        create_bundle(output_dir=tmp_path, json_summary=True)
        captured = capsys.readouterr()
        summary = json.loads(captured.out)
        assert "bundle_dir" in summary
        assert "timestamp" in summary
        assert "files" in summary
        assert "total_size_bytes" in summary

    def test_system_json_is_valid(self, tmp_path):
        bundle_dir = create_bundle(output_dir=tmp_path)
        sys_data = json.loads((bundle_dir / "system.json").read_text())
        assert "python_version" in sys_data
        assert "platform" in sys_data

    def test_config_json_redacts_secrets(self, tmp_path):
        with patch.dict(os.environ, {"OCR_SECRET_VALUE": "hunter2"}, clear=False):
            bundle_dir = create_bundle(output_dir=tmp_path)
            config = json.loads((bundle_dir / "config.json").read_text())
            assert config["env_vars"].get("OCR_SECRET_VALUE") == "***REDACTED***"


# ---------------------------------------------------------------------------
# CLI / main tests
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() CLI entry point."""

    def test_main_default(self, tmp_path, capsys):
        rc = main(["--output-dir", str(tmp_path)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Support bundle created" in captured.out

    def test_main_json_summary(self, tmp_path, capsys):
        rc = main(["--output-dir", str(tmp_path), "--json-summary"])
        assert rc == 0
        captured = capsys.readouterr()
        summary = json.loads(captured.out)
        assert "bundle_dir" in summary

    def test_main_include_logs(self, tmp_path, capsys):
        rc = main(["--output-dir", str(tmp_path), "--include-logs"])
        assert rc == 0
        # Find the bundle dir
        bundles = list(tmp_path.glob("support-bundle-*"))
        assert len(bundles) == 1
        assert (bundles[0] / "logs.txt").exists()


# ---------------------------------------------------------------------------
# Disk usage tests
# ---------------------------------------------------------------------------


class TestDiskUsage:
    """Tests for _safe_disk_usage."""

    def test_existing_path(self, tmp_path):
        usage = _safe_disk_usage(tmp_path)
        assert usage is not None
        assert "total_gb" in usage
        assert "free_gb" in usage
        assert "used_gb" in usage

    def test_nonexistent_path(self):
        usage = _safe_disk_usage(Path("/nonexistent/path/xyz"))
        assert usage is None
