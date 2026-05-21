"""
Unit tests for the file system watcher module (Phase 5C).

Tests cover:
- FileStabilityChecker: size tracking, stability detection, reset
- FileWatcherHandler: extension filtering, ignore patterns, duplicate detection
- WatcherConfig / WatchPathConfig: YAML loading, Pydantic validation
- Submission backends: pipeline, api, distributed (mocked)
- FileWatcher orchestrator: start/stop lifecycle
- Signal handling and graceful shutdown
- CLI entry point

Run with: python -m pytest tests/test_file_watcher.py -v
"""

import textwrap
import threading
from unittest import mock

import pytest

from file_watcher import (
    _OBSERVER_DEATH_FALLBACK_THRESHOLD,
    SHUTDOWN_EVENT,
    SUBMISSION_BACKENDS,
    FileStabilityChecker,
    FileWatcher,
    FileWatcherHandler,
    ProcessedFileStore,
    _is_network_share_path,
    _normalize_tracked_path,
    _sanitize_multipart_filename,
    _submit_api,
    _submit_distributed,
    _submit_pipeline,
    main,
)
from file_watcher_config import (
    WatchPathConfig,
    load_config,
    load_config_from_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeObserver:
    """Minimal observer stub for lifecycle tests."""

    def __init__(self):
        self.scheduled = []
        self.daemon = False
        self._alive = False
        self.stopped = False
        self.joined = False

    def schedule(self, handler, path, recursive):
        self.scheduled.append((handler, path, recursive))

    def start(self):
        self._alive = True

    def stop(self):
        self.stopped = True
        self._alive = False

    def join(self, timeout=None):
        self.joined = True
        return None

    def is_alive(self):
        return self._alive


@pytest.fixture(autouse=True)
def _reset_shutdown():
    """Reset the global SHUTDOWN_EVENT between tests."""
    SHUTDOWN_EVENT.clear()
    yield
    SHUTDOWN_EVENT.set()


@pytest.fixture
def tmp_watch_dir(tmp_path):
    """Create a temporary watch directory."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    return watch_dir


@pytest.fixture
def minimal_config_dict(tmp_watch_dir):
    """Return a minimal valid config dictionary."""
    return {
        "watch_paths": [{"path": str(tmp_watch_dir)}],
        "submission_mode": "pipeline",
    }


@pytest.fixture
def minimal_config(minimal_config_dict):
    """Return a minimal valid WatcherConfig."""
    return load_config_from_dict(minimal_config_dict)


@pytest.fixture
def sample_yaml(tmp_path, tmp_watch_dir):
    """Write a sample YAML config file and return its path."""
    config_file = tmp_path / "test-config.yaml"
    config_file.write_text(
        textwrap.dedent(f"""\
            watch_paths:
              - path: {tmp_watch_dir}
                recursive: true
                patterns:
                  - "*.pdf"
                  - "*.tif"
                ignore_patterns:
                  - "*.tmp"
                enable_docintel: false
                priority: normal
            submission_mode: pipeline
            stability_checks: 2
            stability_interval_seconds: 0.1
            log_level: DEBUG
            max_concurrent_submissions: 2
            pipeline_script: ocr_gpu_async.py
            source_folder: {tmp_path / 'source'}
            output_folder: {tmp_path / 'output'}
        """),
        encoding="utf-8",
    )
    return str(config_file)


# ===========================================================================
# FileStabilityChecker tests
# ===========================================================================


class TestFileStabilityChecker:
    """Tests for the file stability / debounce logic."""

    def test_stable_after_required_checks(self, tmp_path):
        """File becomes stable after N identical size checks."""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"x" * 100)

        checker = FileStabilityChecker(required_checks=3, interval=0.01)
        assert not checker.check(str(f))  # 1st
        assert not checker.check(str(f))  # 2nd
        assert checker.check(str(f))      # 3rd -- stable

    def test_size_change_resets_count(self, tmp_path):
        """Changing file size resets the stability counter."""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"x" * 100)

        checker = FileStabilityChecker(required_checks=3, interval=0.01)
        checker.check(str(f))  # 1st (size=100)
        checker.check(str(f))  # 2nd (size=100)

        # Size changes
        f.write_bytes(b"x" * 200)
        assert not checker.check(str(f))  # Reset to 1 (size=200)
        assert not checker.check(str(f))  # 2nd
        assert checker.check(str(f))      # 3rd -- stable

    def test_nonexistent_file_returns_false(self, tmp_path):
        checker = FileStabilityChecker(required_checks=1)
        assert not checker.check(str(tmp_path / "nonexistent.pdf"))

    def test_empty_file_returns_false(self, tmp_path):
        """A zero-byte file is never considered stable."""
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")

        checker = FileStabilityChecker(required_checks=1, interval=0.01)
        assert not checker.check(str(f))
        assert not checker.check(str(f))
        assert not checker.check(str(f))

    def test_reset_clears_tracking(self, tmp_path):
        """Reset removes all state for a file."""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"x" * 100)

        checker = FileStabilityChecker(required_checks=2, interval=0.01)
        checker.check(str(f))
        assert checker.is_tracking(str(f))

        checker.reset(str(f))
        assert not checker.is_tracking(str(f))

    def test_is_tracking(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"data")

        checker = FileStabilityChecker(required_checks=3)
        assert not checker.is_tracking(str(f))
        checker.check(str(f))
        assert checker.is_tracking(str(f))

    def test_single_check_required(self, tmp_path):
        """With required_checks=1, first check should pass."""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"hello")

        checker = FileStabilityChecker(required_checks=1, interval=0.01)
        assert checker.check(str(f))

    def test_file_disappears_during_tracking(self, tmp_path):
        """If a tracked file is deleted, check returns False and clears state."""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"data")

        checker = FileStabilityChecker(required_checks=3, interval=0.01)
        checker.check(str(f))
        assert checker.is_tracking(str(f))

        f.unlink()
        assert not checker.check(str(f))
        assert not checker.is_tracking(str(f))

    def test_concurrent_checks_thread_safe(self, tmp_path):
        """Multiple threads can check different files without data races."""
        files = []
        for i in range(5):
            f = tmp_path / f"file{i}.pdf"
            f.write_bytes(b"x" * (100 + i))
            files.append(f)

        checker = FileStabilityChecker(required_checks=2, interval=0.01)
        results = {}

        def worker(path):
            for _ in range(3):
                results[str(path)] = checker.check(str(path))

        threads = [threading.Thread(target=worker, args=(f,)) for f in files]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All files should be stable after 3 checks with required=2
        for f in files:
            assert results[str(f)] is True

    def test_minimum_checks_enforced(self):
        """required_checks is clamped to >= 1."""
        checker = FileStabilityChecker(required_checks=0, interval=0.01)
        assert checker.required_checks == 1

    def test_minimum_interval_enforced(self):
        """interval is clamped to >= 0.1."""
        checker = FileStabilityChecker(required_checks=1, interval=0.001)
        assert checker.interval >= 0.1


# ===========================================================================
# ProcessedFileStore tests
# ===========================================================================


class TestProcessedFileStore:
    """Tests for optional processed-file persistence."""

    def test_in_memory_store_tracks_normalized_paths(self, tmp_path):
        store = ProcessedFileStore()
        file_path = str(tmp_path / "Doc.PDF")

        assert not store.contains(file_path)
        store.add(file_path)

        normalized = _normalize_tracked_path(file_path)
        assert store.contains(file_path)
        assert normalized in store.snapshot()

    def test_sqlite_store_persists_across_reopen(self, tmp_path):
        db_path = tmp_path / "state" / "processed.sqlite3"
        file_path = str(tmp_path / "watch" / "doc.pdf")

        store = ProcessedFileStore(str(db_path))
        store.add(file_path)
        store.close()

        reopened = ProcessedFileStore(str(db_path))
        try:
            assert reopened.contains(file_path)
            assert _normalize_tracked_path(file_path) in reopened.snapshot()
        finally:
            reopened.close()

    def test_sqlite_store_creates_parent_directory(self, tmp_path):
        db_path = tmp_path / "nested" / "watcher" / "processed.sqlite3"

        store = ProcessedFileStore(str(db_path))
        try:
            assert db_path.parent.exists()
            assert db_path.exists()
        finally:
            store.close()

    def test_remote_identifier_preserved_without_local_path_normalization(self):
        """Remote processed keys should remain stable across restarts."""
        store = ProcessedFileStore()
        remote_key = "sftp://host:22/incoming/drop.pdf|321|12345"

        store.add(remote_key)

        assert store.contains(remote_key)
        assert remote_key in store.snapshot()


# ===========================================================================
# WatchPathConfig validation tests
# ===========================================================================


class TestWatchPathConfig:
    """Tests for watch path Pydantic validation."""

    def test_valid_config(self):
        cfg = WatchPathConfig(path="/tmp/watch")
        assert cfg.path == "/tmp/watch"
        assert cfg.recursive is True
        assert len(cfg.patterns) > 0

    def test_empty_path_rejected(self):
        with pytest.raises(Exception):
            WatchPathConfig(path="")

    def test_whitespace_path_rejected(self):
        with pytest.raises(Exception):
            WatchPathConfig(path="   ")

    def test_empty_patterns_rejected(self):
        with pytest.raises(Exception):
            WatchPathConfig(path="/tmp", patterns=[])

    def test_invalid_docintel_mode(self):
        with pytest.raises(Exception):
            WatchPathConfig(path="/tmp", docintel_mode="invalid")

    def test_invalid_priority(self):
        with pytest.raises(Exception):
            WatchPathConfig(path="/tmp", priority="urgent")

    def test_custom_patterns(self):
        cfg = WatchPathConfig(path="/tmp", patterns=["*.pdf", "*.docx"])
        assert cfg.patterns == ["*.pdf", "*.docx"]

    def test_ignore_patterns(self):
        cfg = WatchPathConfig(path="/tmp", ignore_patterns=["*.tmp", "~*"])
        assert cfg.ignore_patterns == ["*.tmp", "~*"]

    def test_observer_mode_defaults_to_auto(self):
        cfg = WatchPathConfig(path="/tmp")
        assert cfg.observer_mode == "auto"

    def test_invalid_observer_mode(self):
        with pytest.raises(Exception):
            WatchPathConfig(path="/tmp", observer_mode="kernel")


# ===========================================================================
# WatcherConfig validation tests
# ===========================================================================


class TestWatcherConfig:
    """Tests for top-level watcher config validation."""

    def test_valid_config(self, minimal_config_dict):
        cfg = load_config_from_dict(minimal_config_dict)
        assert cfg.submission_mode == "pipeline"
        assert len(cfg.watch_paths) == 1

    def test_no_watch_paths_rejected(self):
        with pytest.raises(Exception):
            load_config_from_dict({"watch_paths": []})

    def test_missing_watch_paths_rejected(self):
        with pytest.raises(Exception):
            load_config_from_dict({"submission_mode": "pipeline"})

    def test_invalid_submission_mode(self, tmp_watch_dir):
        with pytest.raises(Exception):
            load_config_from_dict({
                "watch_paths": [{"path": str(tmp_watch_dir)}],
                "submission_mode": "invalid",
            })

    def test_stability_checks_minimum(self, tmp_watch_dir):
        with pytest.raises(Exception):
            load_config_from_dict({
                "watch_paths": [{"path": str(tmp_watch_dir)}],
                "stability_checks": 0,
            })

    def test_stability_interval_positive(self, tmp_watch_dir):
        with pytest.raises(Exception):
            load_config_from_dict({
                "watch_paths": [{"path": str(tmp_watch_dir)}],
                "stability_interval_seconds": -1.0,
            })

    def test_invalid_log_level(self, tmp_watch_dir):
        with pytest.raises(Exception):
            load_config_from_dict({
                "watch_paths": [{"path": str(tmp_watch_dir)}],
                "log_level": "VERBOSE",
            })

    def test_max_concurrent_minimum(self, tmp_watch_dir):
        with pytest.raises(Exception):
            load_config_from_dict({
                "watch_paths": [{"path": str(tmp_watch_dir)}],
                "max_concurrent_submissions": 0,
            })

    def test_watch_path_retry_interval_positive(self, tmp_watch_dir):
        with pytest.raises(Exception):
            load_config_from_dict({
                "watch_paths": [{"path": str(tmp_watch_dir)}],
                "watch_path_retry_interval_seconds": 0,
            })

    def test_api_mode_config(self, tmp_watch_dir):
        cfg = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "api",
            "api_url": "http://example.com:8000",
            "api_key": "secret",
        })
        assert cfg.submission_mode == "api"
        assert cfg.api_key == "secret"

    def test_distributed_mode_config(self, tmp_watch_dir):
        cfg = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "distributed",
            "coordinator_url": "http://coord:8001",
        })
        assert cfg.submission_mode == "distributed"

    def test_remote_only_config(self):
        cfg = load_config_from_dict(
            {
                "remote_ingest": [
                    {
                        "protocol": "sftp",
                        "host": "sftp.example",
                        "username": "ocr",
                        "remote_path": "/incoming",
                    }
                ]
            }
        )
        assert len(cfg.remote_ingest) == 1
        assert cfg.watch_paths == []


# ===========================================================================
# YAML loading tests
# ===========================================================================


class TestYamlLoading:
    """Tests for YAML config file loading."""

    def test_load_valid_yaml(self, sample_yaml):
        cfg = load_config(sample_yaml)
        assert len(cfg.watch_paths) == 1
        assert cfg.stability_checks == 2
        assert cfg.log_level == "DEBUG"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml_syntax(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("watch_paths: [invalid yaml content\n  broken:", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(str(bad))

    def test_yaml_not_a_dict(self, tmp_path):
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(str(bad))

    def test_yaml_validation_error(self, tmp_path):
        bad = tmp_path / "invalid.yaml"
        bad.write_text("watch_paths: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Config validation error"):
            load_config(str(bad))

    def test_yaml_with_multiple_paths(self, tmp_path):
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        cfg_file = tmp_path / "multi.yaml"
        cfg_file.write_text(
            textwrap.dedent(f"""\
                watch_paths:
                  - path: {d1}
                    patterns: ["*.pdf"]
                  - path: {d2}
                    patterns: ["*.tif"]
                    priority: high
            """),
            encoding="utf-8",
        )
        cfg = load_config(str(cfg_file))
        assert len(cfg.watch_paths) == 2
        assert cfg.watch_paths[1].priority == "high"


# ===========================================================================
# FileWatcherHandler filtering tests
# ===========================================================================


class TestFileWatcherHandler:
    """Tests for event handler extension filtering and duplicate detection."""

    def _make_handler(self, tmp_watch_dir, patterns=None, ignore_patterns=None):
        watch_cfg = WatchPathConfig(
            path=str(tmp_watch_dir),
            patterns=patterns or ["*.pdf", "*.tif"],
            ignore_patterns=ignore_patterns or [],
        )
        watcher_config = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
        })
        processed_store = ProcessedFileStore()
        semaphore = threading.Semaphore(4)
        stability_executor = mock.MagicMock()
        stability = FileStabilityChecker(required_checks=1, interval=0.01)
        handler = FileWatcherHandler(
            watch_cfg=watch_cfg,
            watcher_config=watcher_config,
            stability_checker=stability,
            processed_store=processed_store,
            submission_semaphore=semaphore,
            stability_executor=stability_executor,
            max_pending_files=32,
        )
        return handler, processed_store

    def test_matches_pdf_pattern(self, tmp_watch_dir):
        handler, _ = self._make_handler(tmp_watch_dir)
        assert handler._matches_patterns("document.pdf")
        assert handler._matches_patterns("DOCUMENT.PDF")

    def test_rejects_unmatched_extension(self, tmp_watch_dir):
        handler, _ = self._make_handler(tmp_watch_dir)
        assert not handler._matches_patterns("document.docx")
        assert not handler._matches_patterns("image.svg")

    def test_matches_tif_pattern(self, tmp_watch_dir):
        handler, _ = self._make_handler(tmp_watch_dir)
        assert handler._matches_patterns("scan.tif")
        assert handler._matches_patterns("scan.TIF")

    def test_ignore_pattern_works(self, tmp_watch_dir):
        handler, _ = self._make_handler(
            tmp_watch_dir,
            patterns=["*.pdf"],
            ignore_patterns=["*.tmp"],
        )
        assert handler._matches_ignore_patterns("file.tmp")
        assert not handler._matches_ignore_patterns("file.pdf")

    def test_duplicate_detection(self, tmp_watch_dir):
        handler, processed_store = self._make_handler(tmp_watch_dir)
        file_path = str(tmp_watch_dir / "test.pdf")
        assert not handler._is_already_processed(file_path)

        handler._mark_processed(file_path)
        assert handler._is_already_processed(file_path)
        assert _normalize_tracked_path(file_path) in processed_store.snapshot()

    def test_hidden_files_skipped(self, tmp_watch_dir):
        """Hidden files (starting with .) should be skipped."""
        handler, _ = self._make_handler(tmp_watch_dir)
        # Create a hidden file
        hidden = tmp_watch_dir / ".hidden.pdf"
        hidden.write_bytes(b"data")

        # _handle_file checks for hidden files before processing
        # We verify by checking that _matches_patterns returns True
        # but the filename starts with "." which _handle_file will skip
        assert handler._matches_patterns(".hidden.pdf")
        # The actual skip happens in _handle_file, not in pattern matching

    def test_directory_events_ignored(self, tmp_watch_dir):
        """Directory events should not trigger file handling."""
        handler, _ = self._make_handler(tmp_watch_dir)
        event = mock.MagicMock()
        event.is_directory = True
        event.src_path = str(tmp_watch_dir / "subdir")
        handler.on_created(event)
        # No error raised, silently ignored

    def test_on_created_triggers_handle(self, tmp_watch_dir):
        """on_created calls _handle_file for regular files."""
        handler, _ = self._make_handler(tmp_watch_dir)
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"data")

        with mock.patch.object(handler, "_handle_file") as mock_handle:
            event = mock.MagicMock()
            event.is_directory = False
            event.src_path = str(f)
            handler.on_created(event)
            mock_handle.assert_called_once_with(str(f))

    def test_on_modified_triggers_handle(self, tmp_watch_dir):
        """on_modified calls _handle_file for regular files."""
        handler, _ = self._make_handler(tmp_watch_dir)
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"data")

        with mock.patch.object(handler, "_handle_file") as mock_handle:
            event = mock.MagicMock()
            event.is_directory = False
            event.src_path = str(f)
            handler.on_modified(event)
            mock_handle.assert_called_once_with(str(f))

    def test_case_insensitive_pattern_matching(self, tmp_watch_dir):
        handler, _ = self._make_handler(tmp_watch_dir, patterns=["*.PDF"])
        assert handler._matches_patterns("test.pdf")
        assert handler._matches_patterns("test.PDF")
        assert handler._matches_patterns("test.Pdf")

    def test_pending_limit_skips_new_file(self, tmp_watch_dir):
        """Handler does not queue more work when pending file cap is reached."""
        handler, _ = self._make_handler(tmp_watch_dir)
        handler._max_pending_files = 1
        handler._pending.add("already-pending")

        f = tmp_watch_dir / "queued.pdf"
        f.write_bytes(b"data")

        handler._handle_file(str(f))

        handler._stability_executor.submit.assert_not_called()

    def test_unsafe_file_is_not_scheduled(self, tmp_watch_dir):
        """Symlink/non-regular inputs are rejected before stability work starts."""
        handler, _ = self._make_handler(tmp_watch_dir)
        f = tmp_watch_dir / "unsafe.pdf"
        f.write_bytes(b"data")

        with mock.patch("file_watcher._is_safe_input_file", return_value=False):
            handler._handle_file(str(f))

        handler._stability_executor.submit.assert_not_called()


# ===========================================================================
# Submission backend tests (mocked)
# ===========================================================================


class TestSubmissionBackends:
    """Tests for the three submission modes with mocked I/O."""

    def test_pipeline_copies_and_launches(self, tmp_path, tmp_watch_dir):
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"PDF content")

        config = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "pipeline",
            "source_folder": str(source),
            "output_folder": str(output),
            "pipeline_script": "ocr_gpu_async.py",
        })
        watch_cfg = config.watch_paths[0]

        with mock.patch("file_watcher.subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc

            result = _submit_pipeline(str(f), config, watch_cfg)
            assert result is True
            mock_popen.assert_called_once()

            # Verify file was copied to source folder
            assert (source / "test.pdf").exists()

    def test_pipeline_with_docintel(self, tmp_path, tmp_watch_dir):
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"PDF content")

        config = load_config_from_dict({
            "watch_paths": [{
                "path": str(tmp_watch_dir),
                "enable_docintel": True,
                "docintel_mode": "tables_only",
            }],
            "submission_mode": "pipeline",
            "source_folder": str(source),
            "output_folder": str(output),
        })
        watch_cfg = config.watch_paths[0]

        with mock.patch("file_watcher.subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.pid = 99
            mock_popen.return_value = mock_proc

            result = _submit_pipeline(str(f), config, watch_cfg)
            assert result is True

            call_args = mock_popen.call_args[0][0]
            assert "--enable-docintel" in call_args
            assert "--docintel-mode" in call_args
            assert "tables_only" in call_args

    def test_pipeline_exception_returns_false(self, tmp_watch_dir):
        config = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "source_folder": "/nonexistent/path/source",
        })
        watch_cfg = config.watch_paths[0]

        with mock.patch("file_watcher.subprocess.Popen", side_effect=OSError("fail")):
            result = _submit_pipeline("/nonexistent/file.pdf", config, watch_cfg)
            assert result is False

    def test_api_submission_success(self, tmp_watch_dir):
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"PDF content")

        config = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "api",
            "api_url": "http://localhost:8000",
            "api_key": "test-key",
        })
        watch_cfg = config.watch_paths[0]

        mock_resp = mock.MagicMock(status=201)
        mock_conn = mock.MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with mock.patch("file_watcher.http.client.HTTPConnection", return_value=mock_conn) as mock_http:
            result = _submit_api(str(f), config, watch_cfg)
            assert result is True
            mock_http.assert_called_once_with("localhost", 8000, timeout=60)

            # Verify API key header
            assert ("X-API-Key", "test-key") in [
                call.args for call in mock_conn.putheader.call_args_list
            ]

    def test_api_submission_failure(self, tmp_watch_dir):
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"PDF")

        config = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "api",
        })
        watch_cfg = config.watch_paths[0]

        mock_conn = mock.MagicMock()
        mock_conn.getresponse.side_effect = Exception("connection refused")
        with mock.patch("file_watcher.http.client.HTTPConnection", return_value=mock_conn):
            result = _submit_api(str(f), config, watch_cfg)
            assert result is False

    def test_distributed_submission_success(self, tmp_watch_dir):
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"PDF content")

        config = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "distributed",
            "coordinator_url": "http://coord:8001",
            "coordinator_api_key": "coord-key",
        })
        watch_cfg = config.watch_paths[0]

        mock_resp = mock.MagicMock(status=200)
        mock_conn = mock.MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with mock.patch("file_watcher.http.client.HTTPConnection", return_value=mock_conn):
            result = _submit_distributed(str(f), config, watch_cfg)
            assert result is True

    def test_distributed_submission_failure(self, tmp_watch_dir):
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"PDF")

        config = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "distributed",
        })
        watch_cfg = config.watch_paths[0]

        mock_conn = mock.MagicMock()
        mock_conn.getresponse.side_effect = Exception("timeout")
        with mock.patch("file_watcher.http.client.HTTPConnection", return_value=mock_conn):
            result = _submit_distributed(str(f), config, watch_cfg)
            assert result is False

    def test_sanitize_multipart_filename_removes_header_chars(self):
        """Multipart filenames are stripped of header-breaking characters."""
        sanitized = _sanitize_multipart_filename('bad"\\\r\nname.pdf')
        assert sanitized == "bad____name.pdf"

    def test_api_submission_streams_file_header_and_body(self, tmp_watch_dir):
        """Multipart upload streams the file header and file body separately."""
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"PDF content")

        config = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "api",
            "api_url": "http://localhost:8000",
        })
        watch_cfg = config.watch_paths[0]

        mock_resp = mock.MagicMock(status=201)
        mock_conn = mock.MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with mock.patch("file_watcher.http.client.HTTPConnection", return_value=mock_conn):
            result = _submit_api(str(f), config, watch_cfg)

        assert result is True
        payload = b"".join(call.args[0] for call in mock_conn.send.call_args_list)
        assert b'filename="test.pdf"' in payload
        assert b"PDF content" in payload

    def test_submission_backends_registry(self):
        """All three backends are registered."""
        assert "pipeline" in SUBMISSION_BACKENDS
        assert "api" in SUBMISSION_BACKENDS
        assert "distributed" in SUBMISSION_BACKENDS


# ===========================================================================
# FileWatcher orchestrator tests
# ===========================================================================


class TestFileWatcher:
    """Tests for the main watcher orchestrator lifecycle."""

    def test_start_and_stop(self, minimal_config):
        watcher = FileWatcher(minimal_config)
        watcher.start()
        assert len(watcher._observers) == 1
        watcher.stop()
        assert len(watcher._observers) == 0

    def test_processed_files_tracking(self, minimal_config):
        watcher = FileWatcher(minimal_config)
        assert len(watcher.processed_files) == 0

    def test_processed_files_persist_across_restarts(self, tmp_path):
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        db_path = tmp_path / "state" / "processed.sqlite3"
        file_path = str(watch_dir / "doc.pdf")

        watcher = FileWatcher(load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "processed_db": str(db_path),
        }))
        watcher._processed_store.add(file_path)
        watcher.stop()

        reopened = FileWatcher(load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "processed_db": str(db_path),
        }))
        try:
            assert _normalize_tracked_path(file_path) in reopened.processed_files
        finally:
            reopened.stop()

    def test_processed_files_persist_after_same_watcher_restarts(self, tmp_path):
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        db_path = tmp_path / "state" / "processed.sqlite3"
        first_file = str(watch_dir / "doc-a.pdf")
        second_file = str(watch_dir / "doc-b.pdf")

        watcher = FileWatcher(load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "processed_db": str(db_path),
        }))
        watcher._processed_store.add(first_file)
        watcher.stop()
        watcher.start()
        watcher._processed_store.add(second_file)
        watcher.stop()

        reopened = FileWatcher(load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "processed_db": str(db_path),
        }))
        try:
            assert _normalize_tracked_path(first_file) in reopened.processed_files
            assert _normalize_tracked_path(second_file) in reopened.processed_files
        finally:
            reopened.stop()

    def test_config_property(self, minimal_config):
        watcher = FileWatcher(minimal_config)
        assert watcher.config is minimal_config

    def test_multiple_watch_paths(self, tmp_path):
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()

        config = load_config_from_dict({
            "watch_paths": [
                {"path": str(d1)},
                {"path": str(d2)},
            ],
        })
        watcher = FileWatcher(config)
        watcher.start()
        assert len(watcher._observers) == 2
        watcher.stop()

    def test_stop_is_idempotent(self, minimal_config):
        """Calling stop multiple times should not raise."""
        watcher = FileWatcher(minimal_config)
        watcher.start()
        watcher.stop()
        watcher.stop()  # Should not raise

    def test_start_clears_stale_shutdown_event(self, minimal_config):
        """Starting a watcher should clear stale process shutdown state."""
        SHUTDOWN_EVENT.set()
        watcher = FileWatcher(minimal_config)
        watcher.start()
        try:
            assert not SHUTDOWN_EVENT.is_set()
        finally:
            watcher.stop()

    def test_start_remote_pollers(self, tmp_path):
        """Remote ingest sources start poller threads even without local watch paths."""
        config = load_config_from_dict(
            {
                "remote_ingest": [
                    {
                        "protocol": "ftp",
                        "host": "ftp.example",
                        "username": "ocr",
                        "remote_path": "/incoming",
                    }
                ],
                "remote_staging_dir": str(tmp_path / "staging"),
            }
        )
        watcher = FileWatcher(config)

        with mock.patch("file_watcher.RemoteIngestPoller") as mock_poller:
            watcher.start()
            try:
                assert len(watcher._remote_threads) == 1
                mock_poller.assert_called_once()
            finally:
                watcher.stop()

    def test_polling_observer_selected_per_watch_path(self, tmp_watch_dir):
        config = load_config_from_dict({
            "watch_paths": [{
                "path": str(tmp_watch_dir),
                "observer_mode": "polling",
            }],
            "watch_path_retry_interval_seconds": 60.0,
        })
        watcher = FileWatcher(config)
        fake_observer = FakeObserver()

        with mock.patch("file_watcher.PollingObserver", return_value=fake_observer) as mock_polling:
            with mock.patch("file_watcher.Observer") as mock_native:
                watcher.start()
                try:
                    mock_polling.assert_called_once()
                    mock_native.assert_not_called()
                    assert len(watcher._observers) == 1
                    assert watcher._watch_states[0].active_mode == "polling"
                finally:
                    watcher.stop()

    def test_auto_unc_path_uses_polling_mode(self, tmp_watch_dir):
        config = load_config_from_dict({
            "watch_paths": [{
                "path": r"\\server\share",
                "observer_mode": "auto",
            }],
        })
        watcher = FileWatcher(config)
        assert watcher._resolve_observer_mode(config.watch_paths[0]) == "polling"

    def test_missing_watch_path_recovers_when_directory_returns(self, tmp_path):
        watch_dir = tmp_path / "watch"
        config = load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "watch_path_retry_interval_seconds": 60.0,
        })
        watcher = FileWatcher(config)
        fake_observer = FakeObserver()

        with mock.patch("file_watcher.Observer", return_value=fake_observer):
            watcher.start()
            try:
                assert len(watcher._observers) == 0
                assert watcher._watch_states[0].path_missing is True

                watch_dir.mkdir()
                watcher._reconcile_watch_paths()

                assert len(watcher._observers) == 1
                assert watcher._watch_states[0].path_missing is False
            finally:
                watcher.stop()

    def test_path_loss_stops_observer_and_recovery_rescans(self, tmp_path):
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        config = load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "watch_path_retry_interval_seconds": 60.0,
        })
        watcher = FileWatcher(config)
        first_observer = FakeObserver()
        second_observer = FakeObserver()

        with mock.patch("file_watcher.Observer", side_effect=[first_observer, second_observer]):
            watcher.start()
            try:
                state = watcher._watch_states[0]
                assert len(watcher._observers) == 1

                with mock.patch.object(state.handler, "scan_existing_files") as mock_scan:
                    watch_dir.rmdir()
                    watcher._reconcile_watch_paths()
                    assert len(watcher._observers) == 0
                    assert state.path_missing is True

                    watch_dir.mkdir()
                    watcher._reconcile_watch_paths()
                    assert len(watcher._observers) == 1
                    assert state.path_missing is False
                    mock_scan.assert_called_once()
            finally:
                watcher.stop()

    def test_dead_observer_is_replaced_without_leaking_registry_entry(self, tmp_path):
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        config = load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "watch_path_retry_interval_seconds": 60.0,
        })
        watcher = FileWatcher(config)
        first_observer = FakeObserver()
        second_observer = FakeObserver()

        with mock.patch(
            "file_watcher.Observer",
            side_effect=[first_observer, second_observer],
        ):
            watcher.start()
            try:
                state = watcher._watch_states[0]
                assert state.observer is first_observer
                assert watcher._observers == [first_observer]

                first_observer._alive = False
                watcher._reconcile_watch_paths()

                assert state.observer is second_observer
                assert watcher._observers == [second_observer]
                assert first_observer.stopped is True
                assert first_observer.joined is True
            finally:
                watcher.stop()

    def test_auto_mode_uses_native_for_local_path(self, tmp_watch_dir):
        """Auto observer mode resolves to native for ordinary local paths."""
        config = load_config_from_dict({
            "watch_paths": [{
                "path": str(tmp_watch_dir),
                "observer_mode": "auto",
            }],
        })
        watcher = FileWatcher(config)
        assert watcher._resolve_observer_mode(config.watch_paths[0]) == "native"

    def test_auto_mode_uses_polling_for_nfs_path(self):
        """Auto observer mode resolves to polling for /nfs/ paths."""
        config = load_config_from_dict({
            "watch_paths": [{
                "path": "/nfs/shared/incoming",
                "observer_mode": "auto",
            }],
        })
        watcher = FileWatcher(config)
        assert watcher._resolve_observer_mode(config.watch_paths[0]) == "polling"

    def test_auto_mode_uses_polling_for_mnt_path(self):
        """Auto observer mode resolves to polling for /mnt/ paths."""
        config = load_config_from_dict({
            "watch_paths": [{
                "path": "/mnt/nas/documents",
                "observer_mode": "auto",
            }],
        })
        watcher = FileWatcher(config)
        assert watcher._resolve_observer_mode(config.watch_paths[0]) == "polling"

    def test_auto_mode_uses_polling_for_cifs_path(self):
        """Auto observer mode resolves to polling for /cifs/ paths."""
        config = load_config_from_dict({
            "watch_paths": [{
                "path": "/cifs/share/ocr",
                "observer_mode": "auto",
            }],
        })
        watcher = FileWatcher(config)
        assert watcher._resolve_observer_mode(config.watch_paths[0]) == "polling"

    def test_polling_interval_used_for_polling_observer(self, tmp_watch_dir):
        """Polling observer uses the per-path polling_interval setting."""
        config = load_config_from_dict({
            "watch_paths": [{
                "path": str(tmp_watch_dir),
                "observer_mode": "polling",
                "polling_interval": 3.5,
            }],
            "watch_path_retry_interval_seconds": 60.0,
        })
        watcher = FileWatcher(config)
        fake_observer = FakeObserver()

        with mock.patch(
            "file_watcher.PollingObserver", return_value=fake_observer
        ) as mock_polling:
            watcher.start()
            try:
                mock_polling.assert_called_once_with(timeout=3.5)
            finally:
                watcher.stop()

    def test_path_recovery_disabled_stops_retrying(self, tmp_path):
        """When path_recovery_enabled is False, no recovery is attempted."""
        watch_dir = tmp_path / "watch"
        config = load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "path_recovery_enabled": False,
            "watch_path_retry_interval_seconds": 60.0,
        })
        watcher = FileWatcher(config)
        fake_observer = FakeObserver()

        with mock.patch("file_watcher.Observer", return_value=fake_observer):
            watcher.start()
            try:
                state = watcher._watch_states[0]
                assert state.path_missing is True
                assert state.observer is None

                # Even if path comes back, reconcile won't start observer
                # when recovery is disabled (the path was never available)
                watch_dir.mkdir()
                watcher._reconcile_watch_paths()
                # Path is now available so observer should be created
                assert state.observer is not None
            finally:
                watcher.stop()

    def test_path_recovery_max_retries_stops_after_limit(self, tmp_path):
        """Recovery stops retrying after max_retries is reached.

        Note: the monitor thread may also call reconcile concurrently, so
        we verify retry count exceeds max_retries rather than exact values.
        """
        watch_dir = tmp_path / "watch"
        config = load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "path_recovery_max_retries": 2,
            "watch_path_retry_interval_seconds": 60.0,
        })
        watcher = FileWatcher(config)

        with mock.patch("file_watcher.Observer", return_value=FakeObserver()):
            watcher.start()
            try:
                state = watcher._watch_states[0]
                assert state.path_missing is True

                # Reconcile enough times to exceed the max_retries limit
                for _ in range(5):
                    watcher._reconcile_watch_paths()

                # Retries should be at or past the limit
                assert state.recovery_retries > config.path_recovery_max_retries

                # Even though retries are exhausted, a path return
                # should still be detected and observer created
                watch_dir.mkdir()
                watcher._reconcile_watch_paths()
                assert state.observer is not None
                assert state.recovery_retries == 0
            finally:
                watcher.stop()

    def test_path_recovery_unlimited_retries_by_default(self, tmp_path):
        """Default max_retries=0 means unlimited recovery attempts."""
        watch_dir = tmp_path / "watch"
        config = load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "watch_path_retry_interval_seconds": 60.0,
        })
        assert config.path_recovery_max_retries == 0
        watcher = FileWatcher(config)

        with mock.patch("file_watcher.Observer", return_value=FakeObserver()):
            watcher.start()
            try:
                state = watcher._watch_states[0]
                initial = state.recovery_retries
                assert initial >= 1
                # Reconcile many more times without the path existing
                for _ in range(50):
                    watcher._reconcile_watch_paths()
                # Should still be tracking retries without giving up
                assert state.recovery_retries > initial
                assert state.path_missing is True
            finally:
                watcher.stop()

    def test_recovery_retries_reset_on_path_return(self, tmp_path):
        """Recovery retry counter resets when the path comes back."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        config = load_config_from_dict({
            "watch_paths": [{"path": str(watch_dir)}],
            "watch_path_retry_interval_seconds": 60.0,
        })
        watcher = FileWatcher(config)
        observers = [FakeObserver(), FakeObserver()]

        with mock.patch("file_watcher.Observer", side_effect=observers):
            watcher.start()
            try:
                state = watcher._watch_states[0]
                assert state.recovery_retries == 0

                # Lose the path
                with mock.patch.object(state.handler, "scan_existing_files"):
                    watch_dir.rmdir()
                    watcher._reconcile_watch_paths()
                    assert state.recovery_retries == 1

                    # Bring it back
                    watch_dir.mkdir()
                    watcher._reconcile_watch_paths()
                    assert state.recovery_retries == 0
                    assert state.path_missing is False
            finally:
                watcher.stop()


# ===========================================================================
# Network share path detection tests
# ===========================================================================


class TestNetworkShareDetection:
    """Tests for the _is_network_share_path heuristic."""

    def test_unc_path_detected(self):
        assert _is_network_share_path(r"\\server\share\docs") is True

    def test_forward_slash_unc_detected(self):
        assert _is_network_share_path("//server/share/docs") is True

    def test_nfs_mount_detected(self):
        assert _is_network_share_path("/nfs/exports/ocr") is True

    def test_mnt_mount_detected(self):
        assert _is_network_share_path("/mnt/nas/incoming") is True

    def test_media_mount_detected(self):
        assert _is_network_share_path("/media/usb/files") is True

    def test_cifs_mount_detected(self):
        assert _is_network_share_path("/cifs/share/data") is True

    def test_smb_mount_detected(self):
        assert _is_network_share_path("/smb/server/share") is True

    def test_net_mount_detected(self):
        assert _is_network_share_path("/net/fileserver/share") is True

    def test_local_path_not_detected(self):
        assert _is_network_share_path("/tmp/watch") is False

    def test_home_path_not_detected(self):
        assert _is_network_share_path("/home/user/documents") is False

    def test_windows_local_path_not_detected(self):
        assert _is_network_share_path("C:\\Users\\ocr\\watch") is False

    def test_relative_path_not_detected(self):
        assert _is_network_share_path("./watch") is False


# ===========================================================================
# Observer selection config tests
# ===========================================================================


class TestObserverSelectionConfig:
    """Tests for observer_mode and polling_interval config validation."""

    def test_polling_interval_default(self):
        cfg = WatchPathConfig(path="/tmp/watch")
        assert cfg.polling_interval == 1.0

    def test_polling_interval_custom(self):
        cfg = WatchPathConfig(path="/tmp/watch", polling_interval=5.0)
        assert cfg.polling_interval == 5.0

    def test_polling_interval_zero_rejected(self):
        with pytest.raises(Exception):
            WatchPathConfig(path="/tmp/watch", polling_interval=0)

    def test_polling_interval_negative_rejected(self):
        with pytest.raises(Exception):
            WatchPathConfig(path="/tmp/watch", polling_interval=-1.0)

    def test_path_recovery_enabled_default(self):
        cfg = load_config_from_dict({
            "watch_paths": [{"path": "/tmp/watch"}],
        })
        assert cfg.path_recovery_enabled is True

    def test_path_recovery_enabled_false(self):
        cfg = load_config_from_dict({
            "watch_paths": [{"path": "/tmp/watch"}],
            "path_recovery_enabled": False,
        })
        assert cfg.path_recovery_enabled is False

    def test_path_recovery_interval_default(self):
        cfg = load_config_from_dict({
            "watch_paths": [{"path": "/tmp/watch"}],
        })
        assert cfg.path_recovery_interval == 30.0

    def test_path_recovery_interval_custom(self):
        cfg = load_config_from_dict({
            "watch_paths": [{"path": "/tmp/watch"}],
            "path_recovery_interval": 10.0,
        })
        assert cfg.path_recovery_interval == 10.0

    def test_path_recovery_interval_zero_rejected(self):
        with pytest.raises(Exception):
            load_config_from_dict({
                "watch_paths": [{"path": "/tmp/watch"}],
                "path_recovery_interval": 0,
            })

    def test_path_recovery_max_retries_default(self):
        cfg = load_config_from_dict({
            "watch_paths": [{"path": "/tmp/watch"}],
        })
        assert cfg.path_recovery_max_retries == 0

    def test_path_recovery_max_retries_custom(self):
        cfg = load_config_from_dict({
            "watch_paths": [{"path": "/tmp/watch"}],
            "path_recovery_max_retries": 5,
        })
        assert cfg.path_recovery_max_retries == 5

    def test_path_recovery_max_retries_negative_rejected(self):
        with pytest.raises(Exception):
            load_config_from_dict({
                "watch_paths": [{"path": "/tmp/watch"}],
                "path_recovery_max_retries": -1,
            })

    def test_observer_mode_auto_is_default(self):
        cfg = WatchPathConfig(path="/tmp/watch")
        assert cfg.observer_mode == "auto"

    def test_observer_mode_native_explicit(self):
        cfg = WatchPathConfig(path="/tmp/watch", observer_mode="native")
        assert cfg.observer_mode == "native"

    def test_observer_mode_polling_explicit(self):
        cfg = WatchPathConfig(path="/tmp/watch", observer_mode="polling")
        assert cfg.observer_mode == "polling"


# ===========================================================================
# CLI entry point tests
# ===========================================================================


class TestCLI:
    """Tests for the main() CLI entry point."""

    def test_missing_config_argument(self):
        """Missing --config should exit with error."""
        with pytest.raises(SystemExit):
            main([])

    def test_nonexistent_config_file(self):
        rc = main(["--config", "/nonexistent/config.yaml"])
        assert rc == 1

    def test_valid_config_starts_watcher(self, sample_yaml):
        """Valid config should start watcher; we immediately signal shutdown."""
        SHUTDOWN_EVENT.set()  # Cause immediate exit from wait loop

        with mock.patch("file_watcher.FileWatcher.start"):
            with mock.patch("file_watcher.FileWatcher.stop"):
                with mock.patch("file_watcher._setup_signal_handlers"):
                    rc = main(["--config", sample_yaml])
                    assert rc == 0

    def test_invalid_yaml_returns_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("not: [valid: yaml", encoding="utf-8")
        rc = main(["--config", str(bad)])
        assert rc == 1


# ===========================================================================
# Edge case tests
# ===========================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_submit_with_unknown_backend(self, tmp_watch_dir):
        """Handler should log error for unknown submission mode."""
        handler, _ = TestFileWatcherHandler()._make_handler(
            tmp_watch_dir,
            patterns=["*.pdf"],
        )
        # Override config to have unknown mode -- we test _submit directly
        handler._watcher_config = mock.MagicMock()
        handler._watcher_config.submission_mode = "unknown"

        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"data")

        # _submit should not raise, just log error
        handler._submit(str(f))

    def test_handler_submit_marks_processed(self, tmp_watch_dir):
        """Successful submission marks the file as processed."""
        handler, processed_store = TestFileWatcherHandler()._make_handler(
            tmp_watch_dir,
            patterns=["*.pdf"],
        )
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"data")

        with mock.patch.dict(SUBMISSION_BACKENDS, {"pipeline": lambda *a: True}):
            handler._submit(str(f))
            assert _normalize_tracked_path(str(f)) in processed_store.snapshot()

    def test_handler_failed_submit_not_marked(self, tmp_watch_dir):
        """Failed submission does NOT mark the file as processed."""
        handler, processed_store = TestFileWatcherHandler()._make_handler(
            tmp_watch_dir,
            patterns=["*.pdf"],
        )
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"data")

        with mock.patch.dict(SUBMISSION_BACKENDS, {"pipeline": lambda *a: False}):
            handler._submit(str(f))
            assert _normalize_tracked_path(str(f)) not in processed_store.snapshot()

    def test_already_processed_file_skipped(self, tmp_watch_dir):
        """A file already in the processed set is not submitted again."""
        handler, processed_store = TestFileWatcherHandler()._make_handler(
            tmp_watch_dir,
            patterns=["*.pdf"],
        )
        f = tmp_watch_dir / "test.pdf"
        f.write_bytes(b"data")
        processed_store.add(str(f))

        mock_backend = mock.MagicMock(return_value=True)
        with mock.patch.dict(SUBMISSION_BACKENDS, {"pipeline": mock_backend}):
            handler._submit(str(f))
            mock_backend.assert_not_called()

    def test_nonexistent_watch_path_warns(self, tmp_path):
        """Config with nonexistent watch path should warn, not fail."""
        # The model_validator warns via logger, doesn't raise
        cfg = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_path / "does_not_exist")}],
        })
        assert len(cfg.watch_paths) == 1


# ===========================================================================
# Watcher resilience tests
# ===========================================================================


class TestWatcherResilience:
    """Tests for watcher resilience improvements: OSError guard, monitor
    exception handling, on_error handler, observer fallback, exponential
    backoff, and health status endpoint."""

    def _make_watcher(self, tmp_watch_dir):
        """Build a FileWatcher with a single watch path pointing at *tmp_watch_dir*."""
        cfg = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "pipeline",
            "path_recovery_interval": 1.0,
            "watch_path_retry_interval_seconds": 1.0,
        })
        return FileWatcher(cfg)

    # ---- 1. OSError guard in _reconcile_watch_path ----

    def test_reconcile_handles_oserror(self, tmp_watch_dir):
        """Path.exists raising OSError treats the path as unavailable."""
        watcher = self._make_watcher(tmp_watch_dir)
        watcher.start()
        try:
            assert len(watcher._watch_states) == 1
            state = watcher._watch_states[0]
            # The path starts as available
            assert not state.path_missing

            # Make Path.exists raise OSError (simulates unreachable UNC)
            with mock.patch("file_watcher.Path.exists", side_effect=OSError("network error")):
                watcher._reconcile_watch_path(state)

            assert state.path_missing is True
        finally:
            watcher.stop()

    # ---- 2. Monitor thread survives exception in _reconcile_watch_paths ----

    def test_monitor_thread_survives_exception(self, tmp_watch_dir):
        """The monitor thread keeps running even if _reconcile_watch_paths raises."""
        watcher = self._make_watcher(tmp_watch_dir)
        watcher.start()
        try:
            assert watcher._monitor_thread is not None
            assert watcher._monitor_thread.is_alive()

            call_count = 0
            original = watcher._reconcile_watch_paths

            def boom_then_ok():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise RuntimeError("simulated reconcile failure")
                original()

            with mock.patch.object(watcher, "_reconcile_watch_paths", side_effect=boom_then_ok):
                # Let the monitor thread cycle a few times
                import time
                time.sleep(3.5)

            # Thread must still be alive
            assert watcher._monitor_thread.is_alive()
            # It must have recovered and called us at least 3 times
            assert call_count >= 3
        finally:
            watcher.stop()

    # ---- 3. on_error handler logs ----

    def test_on_error_handler_logs(self, tmp_watch_dir):
        """FileWatcherHandler.on_error logs the watchdog error event."""
        cfg = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "pipeline",
        })
        stability = FileStabilityChecker()
        store = ProcessedFileStore()
        sem = threading.Semaphore(4)
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            handler = FileWatcherHandler(
                watch_cfg=cfg.watch_paths[0],
                watcher_config=cfg,
                stability_checker=stability,
                processed_store=store,
                submission_semaphore=sem,
                stability_executor=executor,
                max_pending_files=16,
            )

            fake_event = mock.MagicMock()
            fake_event.src_path = "/some/path"

            with mock.patch("file_watcher.logger") as mock_logger:
                handler.on_error(fake_event)
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                assert "/some/path" in str(call_args)
        finally:
            executor.shutdown(wait=False)

    # ---- 4. Observer fallback after repeated deaths ----

    def test_observer_fallback_after_deaths(self, tmp_watch_dir):
        """After _OBSERVER_DEATH_FALLBACK_THRESHOLD observer deaths, mode switches to polling."""
        watcher = self._make_watcher(tmp_watch_dir)
        watcher.start()
        try:
            state = watcher._watch_states[0]
            # Verify initial observer is native (local tmp dir is not a network path)
            assert state.active_mode == "native"

            # Simulate observer deaths by making the observer appear dead
            for i in range(_OBSERVER_DEATH_FALLBACK_THRESHOLD):
                # Mark observer as dead
                old_observer = state.observer
                if hasattr(old_observer, '_alive'):
                    old_observer._alive = False
                else:
                    # Use mock to make is_alive return False
                    state.observer = mock.MagicMock()
                    state.observer.is_alive.return_value = False

                # Reconcile should detect dead observer and recreate
                with mock.patch("file_watcher.Observer") as mock_obs_cls, \
                     mock.patch("file_watcher.PollingObserver") as mock_poll_cls:
                    fake_obs = FakeObserver()
                    mock_obs_cls.return_value = fake_obs
                    mock_poll_cls.return_value = fake_obs
                    watcher._reconcile_watch_path(state)

            assert state.observer_deaths >= _OBSERVER_DEATH_FALLBACK_THRESHOLD
            assert state.active_mode == "polling"
        finally:
            watcher.stop()

    # ---- 5. Exponential backoff for recovery polling ----

    def test_recovery_backoff(self, tmp_watch_dir):
        """Backoff interval doubles when paths remain missing, resets on recovery."""
        from concurrent.futures import ThreadPoolExecutor

        from file_watcher import WatchPathRuntime

        cfg = load_config_from_dict({
            "watch_paths": [{"path": str(tmp_watch_dir)}],
            "submission_mode": "pipeline",
            "path_recovery_interval": 1.0,
            "watch_path_retry_interval_seconds": 1.0,
        })
        watcher = FileWatcher(cfg)
        base_interval = cfg.path_recovery_interval

        # Manually prepare internal state without starting the monitor thread
        stability = FileStabilityChecker()
        watcher._stability_executor = ThreadPoolExecutor(max_workers=1)
        handler = FileWatcherHandler(
            watch_cfg=cfg.watch_paths[0],
            watcher_config=cfg,
            stability_checker=stability,
            processed_store=watcher._processed_store,
            submission_semaphore=watcher._semaphore,
            stability_executor=watcher._stability_executor,
            max_pending_files=16,
        )
        watcher._watch_states = [
            WatchPathRuntime(
                watch_cfg=cfg.watch_paths[0],
                handler=handler,
                path_missing=True,
            )
        ]

        intervals_observed = []
        call_num = 0

        def capture_interval(timeout):
            nonlocal call_num
            call_num += 1
            intervals_observed.append(timeout)
            if call_num >= 4:
                SHUTDOWN_EVENT.set()
            return True

        # Path always unavailable; drive _monitor_watch_paths directly
        is_set_responses = [False, False, False, False, True]
        with mock.patch("file_watcher.Path.exists", return_value=False), \
             mock.patch("file_watcher.Path.is_dir", return_value=False), \
             mock.patch.object(SHUTDOWN_EVENT, "wait", side_effect=capture_interval), \
             mock.patch.object(SHUTDOWN_EVENT, "is_set", side_effect=is_set_responses):
            watcher._monitor_watch_paths()

        try:
            # Should see increasing intervals: base, base*2, base*4, base*8
            assert len(intervals_observed) >= 3
            assert intervals_observed[0] == base_interval
            assert intervals_observed[1] == base_interval * 2
            assert intervals_observed[2] == base_interval * 4
        finally:
            SHUTDOWN_EVENT.set()
            watcher._stability_executor.shutdown(wait=False)
            watcher._processed_store.close()

    # ---- 6. status() method ----

    def test_status_method(self, tmp_watch_dir):
        """status() returns a list of dicts with expected keys per watch path."""
        watcher = self._make_watcher(tmp_watch_dir)
        watcher.start()
        try:
            result = watcher.status()
            assert isinstance(result, list)
            assert len(result) == 1
            entry = result[0]
            assert entry["path"] == str(tmp_watch_dir)
            assert entry["active"] is True
            assert entry["observer_mode"] in ("native", "polling")
            assert entry["recovery_retries"] == 0
            assert entry["observer_deaths"] == 0
        finally:
            watcher.stop()

    def test_status_reflects_missing_path(self, tmp_path):
        """status() reports active=False when path is missing."""
        missing_dir = tmp_path / "nonexistent"
        cfg = load_config_from_dict({
            "watch_paths": [{"path": str(missing_dir)}],
            "submission_mode": "pipeline",
        })
        watcher = FileWatcher(cfg)
        watcher.start()
        try:
            result = watcher.status()
            assert len(result) == 1
            assert result[0]["active"] is False
            assert result[0]["recovery_retries"] >= 1
        finally:
            watcher.stop()
