"""Tests for standalone SFTP ingest module (Task 8)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from sftp_ingest import (
    DEFAULT_EXTENSIONS,
    DEFAULT_LOCAL_STAGING,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PORT,
    DEFAULT_REMOTE_PATH,
    DEFAULT_TIMEOUT,
    SFTPIngestConfig,
    SFTPIngestService,
    _extensions_to_patterns,
    _parse_bool,
    _parse_extensions,
    load_config_from_env,
)

# ---------------------------------------------------------------------------
# SFTPIngestConfig validation
# ---------------------------------------------------------------------------


class TestSFTPIngestConfig:
    """Test the SFTPIngestConfig dataclass and its validation."""

    def test_default_values(self):
        cfg = SFTPIngestConfig()
        assert cfg.enabled is False
        assert cfg.host == ""
        assert cfg.port == DEFAULT_PORT
        assert cfg.username == ""
        assert cfg.password == ""
        assert cfg.private_key_path == ""
        assert cfg.remote_path == DEFAULT_REMOTE_PATH
        assert cfg.local_staging_path == DEFAULT_LOCAL_STAGING
        assert cfg.poll_interval_seconds == DEFAULT_POLL_INTERVAL
        assert cfg.delete_after_download is False
        assert cfg.file_extensions == DEFAULT_EXTENSIONS
        assert cfg.max_file_size_mb == DEFAULT_MAX_FILE_SIZE_MB
        assert cfg.timeout_seconds == DEFAULT_TIMEOUT

    def test_valid_config_with_password(self):
        cfg = SFTPIngestConfig(
            host="sftp.example.com",
            username="user",
            password="secret",
        )
        errors = cfg.validate()
        assert errors == []

    def test_valid_config_with_key(self):
        cfg = SFTPIngestConfig(
            host="sftp.example.com",
            username="user",
            private_key_path="/path/to/key",
        )
        errors = cfg.validate()
        assert errors == []

    def test_missing_host(self):
        cfg = SFTPIngestConfig(username="user", password="secret")
        errors = cfg.validate()
        assert any("host" in e.lower() for e in errors)

    def test_blank_host(self):
        cfg = SFTPIngestConfig(host="  ", username="user", password="secret")
        errors = cfg.validate()
        assert any("host" in e.lower() for e in errors)

    def test_missing_username(self):
        cfg = SFTPIngestConfig(host="sftp.example.com", password="secret")
        errors = cfg.validate()
        assert any("username" in e.lower() for e in errors)

    def test_missing_credentials(self):
        cfg = SFTPIngestConfig(host="sftp.example.com", username="user")
        errors = cfg.validate()
        assert any("password" in e.lower() or "key" in e.lower() for e in errors)

    def test_invalid_port_zero(self):
        cfg = SFTPIngestConfig(
            host="sftp.example.com", username="user", password="s", port=0
        )
        errors = cfg.validate()
        assert any("port" in e.lower() for e in errors)

    def test_invalid_port_too_high(self):
        cfg = SFTPIngestConfig(
            host="sftp.example.com", username="user", password="s", port=70000
        )
        errors = cfg.validate()
        assert any("port" in e.lower() for e in errors)

    def test_negative_poll_interval(self):
        cfg = SFTPIngestConfig(
            host="h", username="u", password="p", poll_interval_seconds=-1
        )
        errors = cfg.validate()
        assert any("poll" in e.lower() for e in errors)

    def test_zero_poll_interval(self):
        cfg = SFTPIngestConfig(
            host="h", username="u", password="p", poll_interval_seconds=0
        )
        errors = cfg.validate()
        assert any("poll" in e.lower() for e in errors)

    def test_negative_timeout(self):
        cfg = SFTPIngestConfig(
            host="h", username="u", password="p", timeout_seconds=-5
        )
        errors = cfg.validate()
        assert any("timeout" in e.lower() for e in errors)

    def test_zero_max_file_size(self):
        cfg = SFTPIngestConfig(
            host="h", username="u", password="p", max_file_size_mb=0
        )
        errors = cfg.validate()
        assert any("file size" in e.lower() for e in errors)

    def test_empty_extensions(self):
        cfg = SFTPIngestConfig(
            host="h", username="u", password="p", file_extensions=[]
        )
        errors = cfg.validate()
        assert any("extension" in e.lower() for e in errors)

    def test_multiple_errors_returned(self):
        cfg = SFTPIngestConfig()
        errors = cfg.validate()
        assert len(errors) >= 3  # host, username, credentials


# ---------------------------------------------------------------------------
# Environment variable parsing
# ---------------------------------------------------------------------------


class TestLoadConfigFromEnv:
    """Test loading config from environment variables."""

    def test_defaults_when_no_env_vars(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = load_config_from_env()
        assert cfg.enabled is False
        assert cfg.host == ""
        assert cfg.port == DEFAULT_PORT
        assert cfg.poll_interval_seconds == DEFAULT_POLL_INTERVAL

    def test_all_env_vars_set(self):
        env = {
            "SFTP_INGEST_ENABLED": "true",
            "SFTP_INGEST_HOST": "sftp.corp.internal",
            "SFTP_INGEST_PORT": "2222",
            "SFTP_INGEST_USERNAME": "ocr-svc",
            "SFTP_INGEST_PASSWORD": "hunter2",
            "SFTP_INGEST_KEY_PATH": "/etc/ssh/ocr_key",
            "SFTP_INGEST_KNOWN_HOSTS": "/etc/ssh/known_hosts",
            "SFTP_INGEST_ALLOW_UNKNOWN": "false",
            "SFTP_INGEST_REMOTE_PATH": "/drop/incoming",
            "SFTP_INGEST_LOCAL_STAGING": "/tmp/sftp-staging",
            "SFTP_INGEST_DESTINATION": "/app/ocr_source",
            "SFTP_INGEST_POLL_INTERVAL": "60",
            "SFTP_INGEST_DELETE_AFTER": "true",
            "SFTP_INGEST_EXTENSIONS": ".pdf,.docx,.tiff",
            "SFTP_INGEST_MAX_FILE_SIZE_MB": "1000",
            "SFTP_INGEST_TIMEOUT": "45",
            "SFTP_INGEST_PROCESSED_DB": "/data/processed.db",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_config_from_env()

        assert cfg.enabled is True
        assert cfg.host == "sftp.corp.internal"
        assert cfg.port == 2222
        assert cfg.username == "ocr-svc"
        assert cfg.password == "hunter2"
        assert cfg.private_key_path == "/etc/ssh/ocr_key"
        assert cfg.known_hosts_path == "/etc/ssh/known_hosts"
        assert cfg.allow_unknown_host is False
        assert cfg.remote_path == "/drop/incoming"
        assert cfg.local_staging_path == "/tmp/sftp-staging"
        assert cfg.destination_path == "/app/ocr_source"
        assert cfg.poll_interval_seconds == 60.0
        assert cfg.delete_after_download is True
        assert cfg.file_extensions == [".pdf", ".docx", ".tiff"]
        assert cfg.max_file_size_mb == 1000
        assert cfg.timeout_seconds == 45.0
        assert cfg.processed_db == "/data/processed.db"

    def test_enabled_truthy_values(self):
        for val in ("1", "true", "yes", "True", "YES"):
            with mock.patch.dict(os.environ, {"SFTP_INGEST_ENABLED": val}, clear=True):
                cfg = load_config_from_env()
            assert cfg.enabled is True, f"Expected true for '{val}'"

    def test_enabled_falsy_values(self):
        for val in ("0", "false", "no", "", "anything"):
            with mock.patch.dict(os.environ, {"SFTP_INGEST_ENABLED": val}, clear=True):
                cfg = load_config_from_env()
            assert cfg.enabled is False, f"Expected false for '{val}'"

    def test_invalid_port_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"SFTP_INGEST_PORT": "abc"}, clear=True):
            cfg = load_config_from_env()
        assert cfg.port == DEFAULT_PORT

    def test_invalid_poll_interval_falls_back_to_default(self):
        with mock.patch.dict(
            os.environ, {"SFTP_INGEST_POLL_INTERVAL": "not-a-number"}, clear=True
        ):
            cfg = load_config_from_env()
        assert cfg.poll_interval_seconds == DEFAULT_POLL_INTERVAL

    def test_invalid_timeout_falls_back_to_default(self):
        with mock.patch.dict(
            os.environ, {"SFTP_INGEST_TIMEOUT": "bad"}, clear=True
        ):
            cfg = load_config_from_env()
        assert cfg.timeout_seconds == DEFAULT_TIMEOUT

    def test_invalid_max_size_falls_back_to_default(self):
        with mock.patch.dict(
            os.environ, {"SFTP_INGEST_MAX_FILE_SIZE_MB": "xyz"}, clear=True
        ):
            cfg = load_config_from_env()
        assert cfg.max_file_size_mb == DEFAULT_MAX_FILE_SIZE_MB

    def test_host_whitespace_stripped(self):
        with mock.patch.dict(
            os.environ, {"SFTP_INGEST_HOST": "  sftp.example.com  "}, clear=True
        ):
            cfg = load_config_from_env()
        assert cfg.host == "sftp.example.com"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Test extension parsing and boolean parsing helpers."""

    def test_parse_bool_true_values(self):
        assert _parse_bool("1") is True
        assert _parse_bool("true") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("  True  ") is True

    def test_parse_bool_false_values(self):
        assert _parse_bool("0") is False
        assert _parse_bool("false") is False
        assert _parse_bool("no") is False
        assert _parse_bool("") is False
        assert _parse_bool("anything") is False

    def test_parse_extensions_with_dots(self):
        result = _parse_extensions(".pdf,.tif,.png")
        assert result == [".pdf", ".tif", ".png"]

    def test_parse_extensions_without_dots(self):
        result = _parse_extensions("pdf,tif,png")
        assert result == [".pdf", ".tif", ".png"]

    def test_parse_extensions_mixed(self):
        result = _parse_extensions(".pdf,tif,.PNG")
        assert result == [".pdf", ".tif", ".png"]

    def test_parse_extensions_with_spaces(self):
        result = _parse_extensions(" .pdf , .tif , .png ")
        assert result == [".pdf", ".tif", ".png"]

    def test_parse_extensions_empty_string(self):
        result = _parse_extensions("")
        assert result == []

    def test_parse_extensions_skips_empty_entries(self):
        result = _parse_extensions(".pdf,,,.tif,")
        assert result == [".pdf", ".tif"]

    def test_extensions_to_patterns(self):
        result = _extensions_to_patterns([".pdf", ".tif", ".png"])
        assert result == ["*.pdf", "*.tif", "*.png"]

    def test_extensions_to_patterns_without_dots(self):
        result = _extensions_to_patterns(["pdf", "tif"])
        assert result == ["*.pdf", "*.tif"]


# ---------------------------------------------------------------------------
# SFTPIngestService
# ---------------------------------------------------------------------------


class TestSFTPIngestService:
    """Test the standalone SFTP ingest service."""

    def _valid_config(self, tmp_path: Path) -> SFTPIngestConfig:
        return SFTPIngestConfig(
            enabled=True,
            host="sftp.example.com",
            port=22,
            username="ocr-user",
            password="secret",
            remote_path="/incoming",
            local_staging_path=str(tmp_path / "staging"),
            destination_path=str(tmp_path / "destination"),
            poll_interval_seconds=5.0,
        )

    def test_is_running_false_before_start(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)
        assert service.is_running is False

    def test_config_property(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)
        assert service.config is cfg

    @mock.patch("sftp_ingest.PARAMIKO_AVAILABLE", False)
    def test_start_fails_without_paramiko(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)
        with pytest.raises(RuntimeError, match="paramiko"):
            service.start()

    @mock.patch("sftp_ingest.PARAMIKO_AVAILABLE", False)
    def test_poll_once_fails_without_paramiko(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)
        with pytest.raises(RuntimeError, match="paramiko"):
            service.poll_once()

    def test_start_fails_with_invalid_config(self, tmp_path):
        cfg = SFTPIngestConfig(enabled=True)  # missing host, username, creds
        service = SFTPIngestService(cfg)
        with pytest.raises(RuntimeError, match="invalid"):
            service.start()

    def test_poll_once_fails_with_invalid_config(self, tmp_path):
        cfg = SFTPIngestConfig(enabled=True)
        service = SFTPIngestService(cfg)
        with pytest.raises(RuntimeError, match="invalid"):
            service.poll_once()

    def test_stage_file_copies_to_destination(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)

        # Create a fake processed store
        from file_watcher import ProcessedFileStore

        service._processed_store = ProcessedFileStore()

        # Create a staged file
        staging = tmp_path / "staging"
        staging.mkdir(parents=True)
        staged_file = staging / "document.pdf"
        staged_file.write_bytes(b"%PDF-1.4 fake content")

        result = service._stage_file(
            str(staged_file), None, "sftp://example:22/incoming/document.pdf|123|456"
        )
        assert result is True

        dest = tmp_path / "destination"
        assert (dest / "document.pdf").exists()
        assert (dest / "document.pdf").read_bytes() == b"%PDF-1.4 fake content"

    def test_stage_file_handles_duplicate_names(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)

        from file_watcher import ProcessedFileStore

        service._processed_store = ProcessedFileStore()

        dest = tmp_path / "destination"
        dest.mkdir(parents=True)
        (dest / "document.pdf").write_bytes(b"existing")

        staging = tmp_path / "staging"
        staging.mkdir(parents=True)
        staged_file = staging / "document.pdf"
        staged_file.write_bytes(b"new content")

        result = service._stage_file(str(staged_file), None, "key1")
        assert result is True

        # Should create document_1.pdf since document.pdf already exists
        assert (dest / "document_1.pdf").exists()
        assert (dest / "document_1.pdf").read_bytes() == b"new content"

    def test_stage_file_enforces_max_size(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        cfg.max_file_size_mb = 1  # 1 MB limit
        service = SFTPIngestService(cfg)

        from file_watcher import ProcessedFileStore

        service._processed_store = ProcessedFileStore()

        staging = tmp_path / "staging"
        staging.mkdir(parents=True)
        staged_file = staging / "huge.pdf"
        # Write 2 MB of data (exceeds 1 MB limit)
        staged_file.write_bytes(b"x" * (2 * 1024 * 1024))

        result = service._stage_file(str(staged_file), None, "key-big")
        assert result is False

    def test_stage_file_rejects_missing_file(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)

        from file_watcher import ProcessedFileStore

        service._processed_store = ProcessedFileStore()

        result = service._stage_file(
            str(tmp_path / "nonexistent.pdf"), None, "key-missing"
        )
        assert result is False

    def test_stage_file_skips_already_processed(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)

        from file_watcher import ProcessedFileStore

        store = ProcessedFileStore()
        store.add("key-already")
        service._processed_store = store

        staging = tmp_path / "staging"
        staging.mkdir(parents=True)
        staged_file = staging / "doc.pdf"
        staged_file.write_bytes(b"content")

        result = service._stage_file(str(staged_file), None, "key-already")
        assert result is False

    def test_stage_file_marks_processed_on_success(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)

        from file_watcher import ProcessedFileStore

        store = ProcessedFileStore()
        service._processed_store = store

        staging = tmp_path / "staging"
        staging.mkdir(parents=True)
        staged_file = staging / "doc.pdf"
        staged_file.write_bytes(b"content")

        key = "sftp://example:22/incoming/doc.pdf|7|999"
        service._stage_file(str(staged_file), None, key)
        assert store.contains(key)

    def test_stop_without_start_is_safe(self, tmp_path):
        cfg = self._valid_config(tmp_path)
        service = SFTPIngestService(cfg)
        service.stop()  # Should not raise


# ---------------------------------------------------------------------------
# File extension filtering
# ---------------------------------------------------------------------------


class TestFileExtensionFiltering:
    """Test that file extension filtering works correctly."""

    def test_default_extensions_include_common_formats(self):
        assert ".pdf" in DEFAULT_EXTENSIONS
        assert ".tif" in DEFAULT_EXTENSIONS
        assert ".tiff" in DEFAULT_EXTENSIONS
        assert ".jpg" in DEFAULT_EXTENSIONS
        assert ".jpeg" in DEFAULT_EXTENSIONS
        assert ".png" in DEFAULT_EXTENSIONS

    def test_extensions_to_patterns_preserves_all(self):
        patterns = _extensions_to_patterns(DEFAULT_EXTENSIONS)
        assert len(patterns) == len(DEFAULT_EXTENSIONS)
        for ext in DEFAULT_EXTENSIONS:
            assert f"*{ext}" in patterns


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestMainCLI:
    """Test the CLI entry point."""

    @mock.patch("sftp_ingest.PARAMIKO_AVAILABLE", False)
    def test_main_exits_when_paramiko_missing(self):
        from sftp_ingest import main

        with mock.patch.dict(
            os.environ, {"SFTP_INGEST_ENABLED": "true"}, clear=True
        ):
            result = main([])
        assert result == 1

    def test_main_exits_when_disabled(self):
        from sftp_ingest import main

        with mock.patch.dict(os.environ, {}, clear=True):
            result = main([])
        assert result == 0

    def test_main_exits_on_config_validation_error(self):
        from sftp_ingest import main

        env = {
            "SFTP_INGEST_ENABLED": "true",
            "SFTP_INGEST_HOST": "",  # invalid
        }
        with mock.patch.dict(os.environ, env, clear=True):
            result = main([])
        assert result == 1

    def test_main_once_mode_calls_poll_once(self):
        from sftp_ingest import main

        env = {
            "SFTP_INGEST_ENABLED": "true",
            "SFTP_INGEST_HOST": "sftp.example.com",
            "SFTP_INGEST_USERNAME": "user",
            "SFTP_INGEST_PASSWORD": "pass",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch(
                "sftp_ingest.SFTPIngestService.poll_once"
            ) as mock_poll,
        ):
            result = main(["--once"])
        mock_poll.assert_called_once()
        assert result == 0

    def test_main_once_mode_returns_1_on_error(self):
        from sftp_ingest import main

        env = {
            "SFTP_INGEST_ENABLED": "true",
            "SFTP_INGEST_HOST": "sftp.example.com",
            "SFTP_INGEST_USERNAME": "user",
            "SFTP_INGEST_PASSWORD": "pass",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch(
                "sftp_ingest.SFTPIngestService.poll_once",
                side_effect=RuntimeError("connection failed"),
            ),
        ):
            result = main(["--once"])
        assert result == 1


# ---------------------------------------------------------------------------
# Integration with RemoteIngestConfig
# ---------------------------------------------------------------------------


class TestRemoteConfigBridge:
    """Test bridging from SFTPIngestConfig to RemoteIngestConfig."""

    def test_build_remote_config_sets_sftp_protocol(self, tmp_path):
        cfg = SFTPIngestConfig(
            host="sftp.example.com",
            port=2222,
            username="user",
            password="pass",
            private_key_path="/key",
            known_hosts_path="/known",
            allow_unknown_host=True,
            remote_path="/drop",
            poll_interval_seconds=10.0,
            timeout_seconds=45.0,
            delete_after_download=True,
            file_extensions=[".pdf", ".tif"],
            local_staging_path=str(tmp_path / "staging"),
        )
        service = SFTPIngestService(cfg)
        remote_cfg = service._build_remote_config()

        assert remote_cfg.protocol == "sftp"
        assert remote_cfg.host == "sftp.example.com"
        assert remote_cfg.port == 2222
        assert remote_cfg.username == "user"
        assert remote_cfg.password == "pass"
        assert remote_cfg.key_filename == "/key"
        assert remote_cfg.known_hosts_path == "/known"
        assert remote_cfg.allow_unknown_host is True
        assert remote_cfg.remote_path == "/drop"
        assert remote_cfg.poll_interval_seconds == 10.0
        assert remote_cfg.timeout_seconds == 45.0
        assert remote_cfg.delete_after_fetch is True
        assert remote_cfg.patterns == ["*.pdf", "*.tif"]

    def test_build_watcher_config_sets_staging_dir(self, tmp_path):
        cfg = SFTPIngestConfig(
            host="h",
            username="u",
            password="p",
            local_staging_path="/my/staging",
        )
        service = SFTPIngestService(cfg)
        watcher_cfg = service._build_watcher_config()
        assert watcher_cfg.remote_staging_dir == "/my/staging"


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Test behaviour when paramiko is not available."""

    @mock.patch("sftp_ingest.PARAMIKO_AVAILABLE", False)
    def test_service_start_raises_without_paramiko(self, tmp_path):
        cfg = SFTPIngestConfig(
            host="sftp.example.com", username="u", password="p"
        )
        service = SFTPIngestService(cfg)
        with pytest.raises(RuntimeError, match="paramiko"):
            service.start()

    @mock.patch("sftp_ingest.PARAMIKO_AVAILABLE", False)
    def test_service_poll_once_raises_without_paramiko(self, tmp_path):
        cfg = SFTPIngestConfig(
            host="sftp.example.com", username="u", password="p"
        )
        service = SFTPIngestService(cfg)
        with pytest.raises(RuntimeError, match="paramiko"):
            service.poll_once()

    @mock.patch("sftp_ingest.PARAMIKO_AVAILABLE", True)
    def test_service_start_succeeds_with_paramiko(self, tmp_path):
        """Verify that with paramiko available, start proceeds to the poller."""
        cfg = SFTPIngestConfig(
            host="sftp.example.com",
            username="u",
            password="p",
            local_staging_path=str(tmp_path / "staging"),
            destination_path=str(tmp_path / "dest"),
            poll_interval_seconds=1.0,
        )
        service = SFTPIngestService(cfg)

        # Mock the RemoteIngestPoller at its source module since sftp_ingest
        # imports it lazily inside start()
        with mock.patch(
            "file_watcher_remote.RemoteIngestPoller"
        ) as mock_poller_cls:
            mock_poller = mock.MagicMock()
            mock_poller_cls.return_value = mock_poller

            # Make run() set the shutdown event so the thread exits
            def _fake_run():
                service._shutdown_event.set()

            mock_poller.run.side_effect = _fake_run

            service.start()
            # Wait for thread to complete
            if service._thread:
                service._thread.join(timeout=5)

            mock_poller_cls.assert_called_once()
            mock_poller.run.assert_called_once()

        service.stop()
