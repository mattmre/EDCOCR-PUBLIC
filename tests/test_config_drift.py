"""Tests for the validate_runtime_config management command (C-14).

Validates drift detection logic, snapshot/baseline I/O, monitored-variable
filtering, and both JSON and human-readable output formats.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent / "coordinator" / "jobs" / "management" / "commands"))

import validate_runtime_config as vrc

# ---------------------------------------------------------------------------
# _is_monitored tests
# ---------------------------------------------------------------------------


class TestIsMonitored:
    """Tests for the _is_monitored filter function."""

    @pytest.mark.parametrize("name", [
        "DJANGO_SECRET_KEY",
        "DJANGO_SETTINGS_MODULE",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
        "RABBITMQ_PASSWORD",
        "S3_ENDPOINT",
        "S3_BUCKET",
        "API_HOST",
        "API_PORT",
        "MAX_UPLOAD_SIZE_MB",
        "WEBHOOK_SECRET",
        "WEBHOOK_TIMEOUT",
        "ENABLE_TRANSFORMS",
        "ENABLE_STAMPING",
        "NUM_WORKERS",
        "NUM_EXTRACTORS",
        "CHUNK_QUEUE_SIZE",
        "REDIS_URL",
        "REDIS_SENTINEL_HOSTS",
        "JOB_PROCESSING_TIMEOUT_MINUTES",
        "JOB_RETENTION_DAYS",
        "METRICS_API_KEY",
        "TENANT_ID",
    ])
    def test_monitored_prefixes(self, name):
        assert vrc._is_monitored(name) is True

    @pytest.mark.parametrize("name", [
        "DATABASE_URL",
        "DEPLOYMENT_ENV",
        "PRODUCTION_READINESS_ACK",
        "STORAGE_BACKEND",
        "NFS_ROOT",
        "OCR_API_KEY",
        "DPI",
        "TEMP_FOLDER",
        "LOG_DIR",
    ])
    def test_monitored_exact_names(self, name):
        assert vrc._is_monitored(name) is True

    @pytest.mark.parametrize("name", [
        "HOME",
        "PATH",
        "USER",
        "SHELL",
        "TERM",
        "LANG",
        "XDG_SESSION_TYPE",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ])
    def test_unmonitored_vars(self, name):
        assert vrc._is_monitored(name) is False


# ---------------------------------------------------------------------------
# capture_monitored_env tests
# ---------------------------------------------------------------------------


class TestCaptureMonitoredEnv:
    """Tests for capture_monitored_env."""

    def test_filters_only_monitored_vars(self):
        fake_env = {
            "DJANGO_SECRET_KEY": "secret",
            "DATABASE_URL": "postgres://...",
            "HOME": "/home/user",
            "PATH": "/usr/bin",
            "CELERY_BROKER_URL": "amqp://...",
            "RANDOM_UNRELATED_VAR": "ignored",
        }
        with mock.patch.dict("os.environ", fake_env, clear=True):
            result = vrc.capture_monitored_env()

        assert "DJANGO_SECRET_KEY" in result
        assert "DATABASE_URL" in result
        assert "CELERY_BROKER_URL" in result
        assert "HOME" not in result
        assert "PATH" not in result
        assert "RANDOM_UNRELATED_VAR" not in result

    def test_returns_sorted_keys(self):
        fake_env = {
            "S3_BUCKET": "bucket",
            "API_HOST": "localhost",
            "DJANGO_DEBUG": "false",
        }
        with mock.patch.dict("os.environ", fake_env, clear=True):
            result = vrc.capture_monitored_env()

        assert list(result.keys()) == sorted(result.keys())

    def test_empty_env(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            result = vrc.capture_monitored_env()
        assert result == {}


# ---------------------------------------------------------------------------
# save_snapshot / load_baseline tests
# ---------------------------------------------------------------------------


class TestSnapshotIO:
    """Tests for save_snapshot and load_baseline."""

    def test_save_and_load_roundtrip(self, tmp_path):
        filepath = tmp_path / "baseline.json"
        env = {"DJANGO_SECRET_KEY": "abc", "S3_BUCKET": "my-bucket"}
        vrc.save_snapshot(filepath, env=env)

        assert filepath.exists()
        loaded = vrc.load_baseline(filepath)
        assert loaded == env

    def test_snapshot_contains_timestamp(self, tmp_path):
        filepath = tmp_path / "baseline.json"
        vrc.save_snapshot(filepath, env={"DPI": "300"})
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert "timestamp" in data
        assert "variables" in data

    def test_snapshot_creates_parent_dirs(self, tmp_path):
        filepath = tmp_path / "sub" / "dir" / "baseline.json"
        vrc.save_snapshot(filepath, env={"DPI": "300"})
        assert filepath.exists()

    def test_load_baseline_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vrc.load_baseline(tmp_path / "nonexistent.json")

    def test_load_baseline_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            vrc.load_baseline(bad_file)

    def test_save_snapshot_uses_current_env_when_none(self, tmp_path):
        filepath = tmp_path / "snap.json"
        fake_env = {"DJANGO_SECRET_KEY": "val1", "HOME": "/home"}
        with mock.patch.dict("os.environ", fake_env, clear=True):
            returned = vrc.save_snapshot(filepath)
        assert "DJANGO_SECRET_KEY" in returned
        assert "HOME" not in returned


# ---------------------------------------------------------------------------
# detect_drift tests
# ---------------------------------------------------------------------------


class TestDetectDrift:
    """Tests for detect_drift."""

    def test_no_drift(self):
        baseline = {"DJANGO_SECRET_KEY": "abc", "S3_BUCKET": "bucket"}
        current = {"DJANGO_SECRET_KEY": "abc", "S3_BUCKET": "bucket"}
        drift = vrc.detect_drift(baseline, current)
        assert drift["missing"] == []
        assert drift["changed"] == []
        assert drift["added"] == []

    def test_missing_var(self):
        baseline = {"DJANGO_SECRET_KEY": "abc", "S3_BUCKET": "bucket"}
        current = {"DJANGO_SECRET_KEY": "abc"}
        drift = vrc.detect_drift(baseline, current)
        assert len(drift["missing"]) == 1
        assert drift["missing"][0]["key"] == "S3_BUCKET"
        assert drift["missing"][0]["baseline_value"] == "bucket"
        assert drift["changed"] == []
        assert drift["added"] == []

    def test_changed_var(self):
        baseline = {"DJANGO_SECRET_KEY": "abc"}
        current = {"DJANGO_SECRET_KEY": "xyz"}
        drift = vrc.detect_drift(baseline, current)
        assert len(drift["changed"]) == 1
        assert drift["changed"][0]["key"] == "DJANGO_SECRET_KEY"
        assert drift["changed"][0]["baseline_value"] == "abc"
        assert drift["changed"][0]["current_value"] == "xyz"

    def test_added_var(self):
        baseline = {"DJANGO_SECRET_KEY": "abc"}
        current = {"DJANGO_SECRET_KEY": "abc", "NEW_API_VAR": "new"}
        drift = vrc.detect_drift(baseline, current)
        assert len(drift["added"]) == 1
        assert drift["added"][0]["key"] == "NEW_API_VAR"
        assert drift["added"][0]["current_value"] == "new"

    def test_mixed_drift(self):
        baseline = {"A": "1", "B": "2", "C": "3"}
        current = {"A": "1", "B": "changed", "D": "4"}
        drift = vrc.detect_drift(baseline, current)
        assert len(drift["missing"]) == 1   # C removed
        assert len(drift["changed"]) == 1   # B changed
        assert len(drift["added"]) == 1     # D added
        assert drift["missing"][0]["key"] == "C"
        assert drift["changed"][0]["key"] == "B"
        assert drift["added"][0]["key"] == "D"

    def test_empty_baseline_and_current(self):
        drift = vrc.detect_drift({}, {})
        assert drift == {"missing": [], "changed": [], "added": []}

    def test_all_missing(self):
        baseline = {"X": "1", "Y": "2"}
        drift = vrc.detect_drift(baseline, {})
        assert len(drift["missing"]) == 2
        assert drift["changed"] == []
        assert drift["added"] == []

    def test_all_added(self):
        current = {"X": "1", "Y": "2"}
        drift = vrc.detect_drift({}, current)
        assert len(drift["added"]) == 2
        assert drift["missing"] == []
        assert drift["changed"] == []


# ---------------------------------------------------------------------------
# Output formatting tests
# ---------------------------------------------------------------------------


class TestFormatTable:
    """Tests for format_table output."""

    def test_no_drift(self):
        drift = {"missing": [], "changed": [], "added": []}
        output = vrc.format_table(drift)
        assert "No configuration drift detected" in output

    def test_with_missing(self):
        drift = {
            "missing": [{"key": "S3_BUCKET", "baseline_value": "bucket"}],
            "changed": [],
            "added": [],
        }
        output = vrc.format_table(drift)
        assert "MISSING" in output
        assert "S3_BUCKET" in output
        assert "bucket" in output

    def test_with_changed(self):
        drift = {
            "missing": [],
            "changed": [{"key": "DPI", "baseline_value": "300", "current_value": "600"}],
            "added": [],
        }
        output = vrc.format_table(drift)
        assert "CHANGED" in output
        assert "DPI" in output
        assert "300" in output
        assert "600" in output

    def test_with_added(self):
        drift = {
            "missing": [],
            "changed": [],
            "added": [{"key": "NEW_VAR", "current_value": "val"}],
        }
        output = vrc.format_table(drift)
        assert "ADDED" in output
        assert "NEW_VAR" in output

    def test_summary_line(self):
        drift = {
            "missing": [{"key": "A", "baseline_value": "1"}],
            "changed": [{"key": "B", "baseline_value": "2", "current_value": "3"}],
            "added": [{"key": "C", "current_value": "4"}],
        }
        output = vrc.format_table(drift)
        assert "1 missing" in output
        assert "1 changed" in output
        assert "1 added" in output
        assert "3 total" in output


class TestFormatJson:
    """Tests for format_json output."""

    def test_no_drift(self):
        drift = {"missing": [], "changed": [], "added": []}
        output = json.loads(vrc.format_json(drift))
        assert output["drift_detected"] is False
        assert output["summary"]["missing"] == 0
        assert output["summary"]["changed"] == 0
        assert output["summary"]["added"] == 0

    def test_with_drift(self):
        drift = {
            "missing": [{"key": "A", "baseline_value": "1"}],
            "changed": [],
            "added": [{"key": "B", "current_value": "2"}],
        }
        output = json.loads(vrc.format_json(drift))
        assert output["drift_detected"] is True
        assert output["summary"]["missing"] == 1
        assert output["summary"]["added"] == 1
        assert len(output["details"]["missing"]) == 1
        assert len(output["details"]["added"]) == 1


# ---------------------------------------------------------------------------
# CLI entry-point tests
# ---------------------------------------------------------------------------


class TestCliMain:
    """Tests for the standalone CLI entry-point."""

    def test_snapshot_mode(self, tmp_path):
        filepath = str(tmp_path / "snap.json")
        fake_env = {"DJANGO_SECRET_KEY": "key", "HOME": "/home"}
        with mock.patch.dict("os.environ", fake_env, clear=True):
            rc = vrc.cli_main(["--snapshot", filepath])
        assert rc == 0
        assert Path(filepath).exists()
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        assert "DJANGO_SECRET_KEY" in data["variables"]
        assert "HOME" not in data["variables"]

    def test_baseline_no_drift(self, tmp_path):
        filepath = str(tmp_path / "baseline.json")
        fake_env = {"DJANGO_SECRET_KEY": "key"}
        with mock.patch.dict("os.environ", fake_env, clear=True):
            vrc.save_snapshot(filepath)
            rc = vrc.cli_main(["--baseline", filepath])
        assert rc == 0

    def test_baseline_with_drift(self, tmp_path):
        filepath = str(tmp_path / "baseline.json")
        vrc.save_snapshot(filepath, env={"DJANGO_SECRET_KEY": "old"})
        fake_env = {"DJANGO_SECRET_KEY": "new"}
        with mock.patch.dict("os.environ", fake_env, clear=True):
            rc = vrc.cli_main(["--baseline", filepath])
        assert rc == 1

    def test_baseline_file_not_found(self, tmp_path):
        filepath = str(tmp_path / "nonexistent.json")
        rc = vrc.cli_main(["--baseline", filepath])
        assert rc == 1

    def test_baseline_invalid_json(self, tmp_path):
        filepath = str(tmp_path / "bad.json")
        Path(filepath).write_text("bad{json", encoding="utf-8")
        rc = vrc.cli_main(["--baseline", filepath])
        assert rc == 1

    def test_json_output_mode(self, tmp_path, capsys):
        filepath = str(tmp_path / "baseline.json")
        vrc.save_snapshot(filepath, env={"DJANGO_SECRET_KEY": "old"})
        fake_env = {"DJANGO_SECRET_KEY": "new"}
        with mock.patch.dict("os.environ", fake_env, clear=True):
            rc = vrc.cli_main(["--baseline", filepath, "--json"])
        assert rc == 1
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["drift_detected"] is True

    def test_no_drift_json_output(self, tmp_path, capsys):
        filepath = str(tmp_path / "baseline.json")
        fake_env = {"DJANGO_SECRET_KEY": "same"}
        with mock.patch.dict("os.environ", fake_env, clear=True):
            vrc.save_snapshot(filepath)
            rc = vrc.cli_main(["--baseline", filepath, "--json"])
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["drift_detected"] is False


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests simulating the full snapshot-then-compare workflow."""

    def test_full_workflow_no_drift(self, tmp_path):
        filepath = tmp_path / "config.json"
        env = {"DJANGO_SECRET_KEY": "k", "S3_BUCKET": "b", "DPI": "300"}
        with mock.patch.dict("os.environ", env, clear=True):
            vrc.save_snapshot(filepath)
            baseline = vrc.load_baseline(filepath)
            current = vrc.capture_monitored_env()
        drift = vrc.detect_drift(baseline, current)
        assert not any(drift[c] for c in ("missing", "changed", "added"))

    def test_full_workflow_with_drift(self, tmp_path):
        filepath = tmp_path / "config.json"
        env_before = {"DJANGO_SECRET_KEY": "k1", "S3_BUCKET": "b", "DPI": "300"}
        with mock.patch.dict("os.environ", env_before, clear=True):
            vrc.save_snapshot(filepath)
        baseline = vrc.load_baseline(filepath)

        env_after = {"DJANGO_SECRET_KEY": "k2", "ENABLE_TRANSFORMS": "true"}
        with mock.patch.dict("os.environ", env_after, clear=True):
            current = vrc.capture_monitored_env()

        drift = vrc.detect_drift(baseline, current)
        # S3_BUCKET and DPI missing
        assert len(drift["missing"]) == 2
        # DJANGO_SECRET_KEY changed
        assert len(drift["changed"]) == 1
        # ENABLE_TRANSFORMS added
        assert len(drift["added"]) == 1
