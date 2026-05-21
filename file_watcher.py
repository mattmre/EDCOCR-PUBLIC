"""File system watcher for hot-folder OCR submission (Phase 5C).

Monitors configured directories for new files and automatically submits
them to the OCR pipeline via one of three backends:
  - pipeline: subprocess launch of ocr_gpu_async.py
  - api: HTTP POST to the REST API
  - distributed: HTTP POST to the Celery coordinator

Debounce logic ensures partially-written files are not submitted until
their size has stabilized.

Usage:
    python file_watcher.py --config file-watcher.yaml
"""

from __future__ import annotations

import argparse
import fnmatch
import http.client
import logging
import os
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlsplit

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from file_watcher_config import WatcherConfig, WatchPathConfig, load_config
from file_watcher_remote import RemoteIngestPoller

logger = logging.getLogger("file_watcher")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHUTDOWN_EVENT = threading.Event()
HTTP_CHUNK_SIZE = 1024 * 1024
_OBSERVER_DEATH_FALLBACK_THRESHOLD = 3


# ---------------------------------------------------------------------------
# File stability checker
# ---------------------------------------------------------------------------


class FileStabilityChecker:
    """Track file sizes over multiple checks to detect write completion.

    A file is considered stable when its size remains unchanged for
    ``required_checks`` consecutive polls spaced ``interval`` seconds apart.
    """

    def __init__(self, required_checks: int = 3, interval: float = 2.0):
        self._required_checks = max(1, required_checks)
        self._interval = max(0.1, interval)
        # file_path -> (last_size, consecutive_stable_count)
        self._tracking: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    @property
    def required_checks(self) -> int:
        return self._required_checks

    @property
    def interval(self) -> float:
        return self._interval

    def check(self, file_path: str) -> bool:
        """Return True if the file size has been stable for enough checks.

        Returns False if the file does not exist or is still changing.
        """
        try:
            current_size = os.path.getsize(file_path)
        except OSError:
            with self._lock:
                self._tracking.pop(file_path, None)
            return False

        if current_size == 0:
            return False

        with self._lock:
            prev_size, count = self._tracking.get(file_path, (-1, 0))
            if current_size == prev_size:
                count += 1
            else:
                count = 1
            self._tracking[file_path] = (current_size, count)
            return count >= self._required_checks

    def reset(self, file_path: str) -> None:
        """Remove tracking state for a file."""
        with self._lock:
            self._tracking.pop(file_path, None)

    def is_tracking(self, file_path: str) -> bool:
        """Return True if the file is currently being tracked."""
        with self._lock:
            return file_path in self._tracking


def _normalize_tracked_path(file_path: str) -> str:
    """Return a stable normalized path key for processed-file tracking."""
    if "://" in file_path:
        return file_path.strip()
    return os.path.normcase(os.path.abspath(file_path))


class ProcessedFileStore:
    """Thread-safe processed-file tracker with optional SQLite persistence."""

    def __init__(self, db_path: str = ""):
        self._lock = threading.Lock()
        self._paths: set[str] = set()
        self._conn: Optional[sqlite3.Connection] = None
        self._db_path = db_path.strip()
        if self._db_path:
            self._open()

    def _open(self) -> None:
        """Open the SQLite store and preload known processed files."""
        if self._db_path == ":memory:":
            db_location = self._db_path
        else:
            db_file = Path(self._db_path).expanduser()
            db_file.parent.mkdir(parents=True, exist_ok=True)
            db_location = str(db_file)

        conn = sqlite3.connect(db_location, check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_files (
                path TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        rows = conn.execute("SELECT path FROM processed_files").fetchall()
        self._paths = {row[0] for row in rows}
        self._conn = conn

    def reopen(self) -> None:
        """Reopen the persistence connection after a prior close()."""
        if not self._db_path:
            return
        with self._lock:
            if self._conn is not None:
                return
            self._open()

    def contains(self, file_path: str) -> bool:
        """Return True if the file path is already marked processed."""
        normalized = _normalize_tracked_path(file_path)
        with self._lock:
            return normalized in self._paths

    def add(self, file_path: str) -> None:
        """Mark a file path processed and persist it when configured."""
        normalized = _normalize_tracked_path(file_path)
        with self._lock:
            self._paths.add(normalized)
            if self._conn is not None:
                self._conn.execute(
                    "INSERT OR IGNORE INTO processed_files(path) VALUES (?)",
                    (normalized,),
                )
                self._conn.commit()

    def snapshot(self) -> set[str]:
        """Return a copy of the processed-file set."""
        with self._lock:
            return set(self._paths)

    def close(self) -> None:
        """Close the persistence connection if one is open."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# ---------------------------------------------------------------------------
# Submission backends
# ---------------------------------------------------------------------------


def _is_safe_input_file(file_path: str) -> bool:
    """Reject symlinks and non-regular files before submission."""
    path = Path(file_path)
    try:
        if path.is_symlink():
            logger.warning("Rejecting symlink input: %s", file_path)
            return False
        if not path.exists() or not path.is_file():
            logger.warning("Rejecting non-regular input: %s", file_path)
            return False
    except OSError:
        logger.exception("Failed to inspect input file: %s", file_path)
        return False
    return True


def _sanitize_multipart_filename(filename: str) -> str:
    """Remove header-breaking characters from multipart filenames."""
    sanitized = filename.replace("\r", "_").replace("\n", "_")
    sanitized = sanitized.replace("\\", "_").replace('"', "_")
    return sanitized or "upload.bin"


def _submit_http_multipart(
    *,
    file_path: str,
    url: str,
    api_key: str,
    extra_fields: list[tuple[str, str]],
    label: str,
) -> bool:
    """Stream a multipart file upload without loading the whole body into memory."""
    if not _is_safe_input_file(file_path):
        return False

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Unsupported submission URL: {url}")

    boundary = f"----FileWatcherBoundary{secrets.token_hex(16)}"
    original_filename = Path(file_path).name
    safe_filename = _sanitize_multipart_filename(original_filename)
    encoded_filename = quote(original_filename, safe="")
    file_size = Path(file_path).stat().st_size

    file_part_header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"; '
        f"filename*=UTF-8''{encoded_filename}\r\n"
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    extra_parts = b"".join(
        (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}"
        ).encode("utf-8")
        for name, value in extra_fields
    )
    closing = f"\r\n--{boundary}--\r\n".encode("utf-8")
    content_length = len(file_part_header) + file_size + len(extra_parts) + len(closing)

    path_with_query = parsed.path or "/"
    if parsed.query:
        path_with_query = f"{path_with_query}?{parsed.query}"

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_cls(parsed.hostname, port, timeout=60)
    try:
        conn.putrequest("POST", path_with_query)
        conn.putheader("Host", parsed.netloc)
        conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        conn.putheader("Content-Length", str(content_length))
        if api_key:
            conn.putheader("X-API-Key", api_key)
        conn.endheaders()

        conn.send(file_part_header)
        with open(file_path, "rb") as handle:
            while chunk := handle.read(HTTP_CHUNK_SIZE):
                conn.send(chunk)
        conn.send(extra_parts)
        conn.send(closing)

        response = conn.getresponse()
        status = response.status
        response.read()
        logger.info("%s submission for %s returned status %d", label, file_path, status)
        return 200 <= status < 300
    finally:
        conn.close()


def _submit_pipeline(
    file_path: str,
    config: WatcherConfig,
    watch_cfg: WatchPathConfig,
) -> bool:
    """Submit a file by copying to source folder and launching subprocess."""
    try:
        if not _is_safe_input_file(file_path):
            return False

        source_dir = Path(config.source_folder)
        source_dir.mkdir(parents=True, exist_ok=True)
        dest = source_dir / Path(file_path).name
        shutil.copy2(file_path, dest)

        cmd = [
            sys.executable,
            config.pipeline_script,
            "--source", str(source_dir),
            "--output", config.output_folder,
        ]
        if watch_cfg.enable_docintel:
            cmd.append("--enable-docintel")
            if watch_cfg.docintel_mode != "full":
                cmd.extend(["--docintel-mode", watch_cfg.docintel_mode])

        logger.info("Launching pipeline for %s: %s", file_path, cmd)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        logger.info("Pipeline started (PID %d) for %s", proc.pid, file_path)
        return True
    except Exception:
        logger.exception("Pipeline submission failed for %s", file_path)
        return False


def _submit_api(
    file_path: str,
    config: WatcherConfig,
    watch_cfg: WatchPathConfig,
) -> bool:
    """Submit a file to the REST API via HTTP POST."""
    try:
        extra_fields: list[tuple[str, str]] = []
        if watch_cfg.priority != "normal":
            extra_fields.append(("priority", watch_cfg.priority))
        if watch_cfg.enable_docintel:
            extra_fields.append(("enable_docintel", "true"))

        return _submit_http_multipart(
            file_path=file_path,
            url=f"{config.api_url.rstrip('/')}/api/jobs",
            api_key=config.api_key,
            extra_fields=extra_fields,
            label="API",
        )
    except Exception:
        logger.exception("API submission failed for %s", file_path)
        return False


def _submit_distributed(
    file_path: str,
    config: WatcherConfig,
    watch_cfg: WatchPathConfig,
) -> bool:
    """Submit a file to the distributed coordinator via HTTP POST."""
    try:
        extra_fields: list[tuple[str, str]] = []
        if watch_cfg.priority != "normal":
            extra_fields.append(("priority", watch_cfg.priority))

        return _submit_http_multipart(
            file_path=file_path,
            url=f"{config.coordinator_url.rstrip('/')}/api/v1/jobs/",
            api_key=config.coordinator_api_key,
            extra_fields=extra_fields,
            label="Coordinator",
        )
    except Exception:
        logger.exception("Distributed submission failed for %s", file_path)
        return False


SUBMISSION_BACKENDS = {
    "pipeline": _submit_pipeline,
    "api": _submit_api,
    "distributed": _submit_distributed,
}


def _submit_with_config(
    *,
    file_path: str,
    watcher_config: WatcherConfig,
    watch_cfg,
    processed_store: ProcessedFileStore,
    submission_semaphore: threading.Semaphore,
    processed_key: Optional[str] = None,
) -> bool:
    """Submit a file with the configured backend and mark success in the store."""
    tracked_key = processed_key or file_path
    if processed_store.contains(tracked_key):
        return False
    if not _is_safe_input_file(file_path):
        return False

    backend = SUBMISSION_BACKENDS.get(watcher_config.submission_mode)
    if not backend:
        logger.error(
            "Unknown submission mode: %s", watcher_config.submission_mode
        )
        return False

    acquired = submission_semaphore.acquire(timeout=30)
    if not acquired:
        logger.warning("Submission throttled for %s, retrying later", file_path)
        return False

    try:
        logger.info(
            "Submitting file: %s (mode=%s)",
            file_path,
            watcher_config.submission_mode,
        )
        success = backend(file_path, watcher_config, watch_cfg)
        if success:
            processed_store.add(tracked_key)
            logger.info("Successfully submitted: %s", file_path)
        else:
            logger.warning("Submission returned failure for %s", file_path)
        return success
    except Exception:
        logger.exception("Submission error for %s", file_path)
        return False
    finally:
        submission_semaphore.release()


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------


class FileWatcherHandler(FileSystemEventHandler):
    """Handles filesystem created/modified events with debounce and filtering."""

    def __init__(
        self,
        watch_cfg: WatchPathConfig,
        watcher_config: WatcherConfig,
        stability_checker: FileStabilityChecker,
        processed_store: ProcessedFileStore,
        submission_semaphore: threading.Semaphore,
        stability_executor: ThreadPoolExecutor,
        max_pending_files: int,
    ):
        super().__init__()
        self._watch_cfg = watch_cfg
        self._watcher_config = watcher_config
        self._stability = stability_checker
        self._processed_store = processed_store
        self._semaphore = submission_semaphore
        self._stability_executor = stability_executor
        self._max_pending_files = max_pending_files
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()

    def _matches_patterns(self, filename: str) -> bool:
        """Return True if filename matches any configured glob pattern."""
        lower = filename.lower()
        for pattern in self._watch_cfg.patterns:
            if fnmatch.fnmatch(lower, pattern.lower()):
                return True
        return False

    def _matches_ignore_patterns(self, filename: str) -> bool:
        """Return True if filename matches any ignore pattern."""
        lower = filename.lower()
        for pattern in self._watch_cfg.ignore_patterns:
            if fnmatch.fnmatch(lower, pattern.lower()):
                return True
        return False

    def _is_already_processed(self, file_path: str) -> bool:
        """Return True if the file has already been submitted."""
        return self._processed_store.contains(file_path)

    def _mark_processed(self, file_path: str) -> None:
        """Mark a file as successfully submitted."""
        self._processed_store.add(file_path)

    def _handle_file(self, file_path: str) -> None:
        """Process a detected file: filter, debounce, submit."""
        filename = os.path.basename(file_path)

        # Extension filter
        if not self._matches_patterns(filename):
            return

        # Ignore filter
        if self._matches_ignore_patterns(filename):
            return

        # Skip hidden files
        if filename.startswith("."):
            return

        if not _is_safe_input_file(file_path):
            return

        # Duplicate check
        if self._is_already_processed(file_path):
            return

        # Avoid scheduling duplicate pending checks
        with self._pending_lock:
            if file_path in self._pending:
                return
            if len(self._pending) >= self._max_pending_files:
                logger.warning("Pending watcher queue is full, skipping %s", file_path)
                return
            self._pending.add(file_path)

        # Start bounded stability check work in the shared executor
        try:
            self._stability_executor.submit(self._stability_check_loop, file_path)
        except RuntimeError:
            with self._pending_lock:
                self._pending.discard(file_path)
            logger.warning("Stability executor is unavailable for %s", file_path)

    def _stability_check_loop(self, file_path: str) -> None:
        """Poll file size until stable, then submit."""
        try:
            while not SHUTDOWN_EVENT.is_set():
                if self._stability.check(file_path):
                    # File is stable -- submit it
                    self._submit(file_path)
                    return
                # Wait between checks
                SHUTDOWN_EVENT.wait(self._stability.interval)
                if SHUTDOWN_EVENT.is_set():
                    return
                # If file disappeared, bail out
                if not os.path.exists(file_path):
                    logger.debug("File disappeared during stability check: %s", file_path)
                    return
        except Exception:
            logger.exception("Stability check error for %s", file_path)
        finally:
            with self._pending_lock:
                self._pending.discard(file_path)
            self._stability.reset(file_path)

    def _submit(self, file_path: str) -> None:
        """Submit the file to the configured backend."""
        _submit_with_config(
            file_path=file_path,
            watcher_config=self._watcher_config,
            watch_cfg=self._watch_cfg,
            processed_store=self._processed_store,
            submission_semaphore=self._semaphore,
        )

    def scan_existing_files(self) -> None:
        """Queue existing files after a watched path recovers."""
        root = Path(self._watch_cfg.path)
        if not root.exists() or not root.is_dir():
            return

        try:
            candidates = root.rglob("*") if self._watch_cfg.recursive else root.iterdir()
        except OSError:
            logger.exception("Failed to enumerate watch path for rescan: %s", root)
            return

        for candidate in candidates:
            if SHUTDOWN_EVENT.is_set():
                return
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                logger.debug("Skipping unreadable path during rescan: %s", candidate)
                continue
            self._handle_file(str(candidate))

    # --- watchdog event callbacks ---

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_file(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_file(event.src_path)

    def on_error(self, event):
        """Log watchdog observer errors instead of silently swallowing them."""
        logger.error(
            "Watchdog observer error on %s: %s",
            getattr(event, "src_path", "unknown"),
            event,
        )


# ---------------------------------------------------------------------------
# Main watcher orchestrator
# ---------------------------------------------------------------------------


def _is_network_share_path(path: str) -> bool:
    """Detect whether a path is likely a network share.

    Checks for UNC paths (``\\\\server\\share``) and common Linux NFS/CIFS
    mount prefixes.  This heuristic intentionally casts a wide net so that
    ``auto`` observer mode falls back to polling for paths where native
    filesystem events are unreliable.
    """
    normalized = path.replace("/", "\\")
    if normalized.startswith("\\\\"):
        return True

    posix = path.replace("\\", "/")
    network_prefixes = ("/mnt/", "/media/", "/net/", "/nfs/", "/cifs/", "/smb/")
    for prefix in network_prefixes:
        if posix.startswith(prefix):
            return True

    return False


@dataclass
class WatchPathRuntime:
    """Runtime state for a configured watch path."""

    watch_cfg: WatchPathConfig
    handler: FileWatcherHandler
    observer: Optional[object] = None
    active_mode: str = ""
    path_missing: bool = False
    needs_rescan: bool = False
    recovery_retries: int = 0
    observer_deaths: int = 0


class FileWatcher:
    """Orchestrates filesystem observers for all configured watch paths."""

    def __init__(self, config: WatcherConfig):
        self._config = config
        self._observers: list[Observer] = []
        self._remote_threads: list[threading.Thread] = []
        self._watch_states: list[WatchPathRuntime] = []
        self._processed_store = ProcessedFileStore(config.processed_db)
        self._semaphore = threading.Semaphore(config.max_concurrent_submissions)
        self._stability_executor: Optional[ThreadPoolExecutor] = None
        self._max_pending_files = max(16, config.max_concurrent_submissions * 8)
        self._observer_lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None

    @property
    def config(self) -> WatcherConfig:
        return self._config

    @property
    def processed_files(self) -> set[str]:
        return self._processed_store.snapshot()

    def _resolve_observer_mode(self, watch_cfg: WatchPathConfig) -> str:
        """Resolve ``auto`` observer mode to a concrete backend.

        When set to ``auto``, uses polling for UNC paths and common
        network-share mount prefixes (``/mnt/``, ``/nfs/``, etc.),
        and native filesystem events for everything else.
        """
        if watch_cfg.observer_mode != "auto":
            return watch_cfg.observer_mode

        if _is_network_share_path(watch_cfg.path):
            return "polling"
        return "native"

    def _create_observer(
        self,
        watch_cfg: WatchPathConfig,
        *,
        force_polling: bool = False,
    ) -> tuple[object, str]:
        """Create the concrete observer for a watch path."""
        mode = "polling" if force_polling else self._resolve_observer_mode(watch_cfg)
        if mode == "polling":
            timeout = max(0.2, watch_cfg.polling_interval)
            return PollingObserver(timeout=timeout), mode
        return Observer(), mode

    def _stop_runtime_observer(self, state: WatchPathRuntime) -> None:
        """Stop and detach an active observer for a watch path."""
        observer = state.observer
        if observer is None:
            return

        try:
            observer.stop()
        except Exception:
            logger.exception("Error stopping observer for %s", state.watch_cfg.path)

        try:
            observer.join(timeout=10)
        except Exception:
            logger.exception("Error joining observer for %s", state.watch_cfg.path)

        with self._observer_lock:
            if observer in self._observers:
                self._observers.remove(observer)

        state.observer = None
        state.active_mode = ""

    def _reconcile_watch_path(self, state: WatchPathRuntime) -> None:
        """Ensure a watch path has the correct observer lifecycle.

        When the watched directory disappears, the observer is stopped and
        recovery retries are tracked.  If ``path_recovery_enabled`` is
        ``False`` or ``path_recovery_max_retries`` has been exceeded, no
        further recovery attempts are made until the watcher is restarted.
        """
        path_obj = Path(state.watch_cfg.path)
        try:
            path_available = path_obj.exists() and path_obj.is_dir()
        except OSError:
            logger.warning(
                "OSError checking path %s — treating as unavailable",
                state.watch_cfg.path,
            )
            path_available = False
        max_retries = self._config.path_recovery_max_retries

        if not path_available:
            if state.observer is not None:
                logger.warning(
                    "Watch path unavailable, stopping observer until it returns: %s",
                    state.watch_cfg.path,
                )
                self._stop_runtime_observer(state)
                state.needs_rescan = True
            elif not state.path_missing:
                logger.warning(
                    "Watch path unavailable, waiting for recovery: %s",
                    state.watch_cfg.path,
                )
            state.path_missing = True

            if not self._config.path_recovery_enabled:
                return

            if max_retries > 0 and state.recovery_retries >= max_retries:
                if state.recovery_retries == max_retries:
                    logger.error(
                        "Path recovery exhausted after %d retries: %s",
                        max_retries,
                        state.watch_cfg.path,
                    )
                    state.recovery_retries += 1
                return

            state.recovery_retries += 1
            return

        if state.observer is not None and getattr(
            state.observer, "is_alive", lambda: False
        )():
            state.path_missing = False
            return

        if state.observer is not None:
            state.observer_deaths += 1
            logger.warning(
                "Watch observer is no longer alive (death #%d), recreating for %s",
                state.observer_deaths,
                state.watch_cfg.path,
            )
            self._stop_runtime_observer(state)

            if state.observer_deaths >= _OBSERVER_DEATH_FALLBACK_THRESHOLD:
                logger.warning(
                    "Observer died %d times for %s — forcing polling mode",
                    state.observer_deaths,
                    state.watch_cfg.path,
                )

        force_polling = state.observer_deaths >= _OBSERVER_DEATH_FALLBACK_THRESHOLD
        observer, mode = self._create_observer(
            state.watch_cfg, force_polling=force_polling
        )
        observer.schedule(
            state.handler,
            path=state.watch_cfg.path,
            recursive=state.watch_cfg.recursive,
        )
        observer.daemon = True
        observer.start()

        with self._observer_lock:
            self._observers.append(observer)

        state.observer = observer
        state.active_mode = mode
        was_recovered = state.path_missing or state.needs_rescan
        state.path_missing = False
        state.recovery_retries = 0
        logger.info(
            "Watching: %s (recursive=%s, patterns=%s, observer_mode=%s)",
            state.watch_cfg.path,
            state.watch_cfg.recursive,
            state.watch_cfg.patterns,
            mode,
        )

        if was_recovered:
            logger.info("Rescanning recovered watch path: %s", state.watch_cfg.path)
            state.handler.scan_existing_files()
            state.needs_rescan = False

    def _reconcile_watch_paths(self) -> None:
        """Reconcile all watch paths against current availability."""
        for state in self._watch_states:
            self._reconcile_watch_path(state)

    def _monitor_watch_paths(self) -> None:
        """Monitor watched paths for loss and recovery.

        Uses exponential backoff (base ``path_recovery_interval``, capped at
        300 s) when any watch path is missing, and resets to the base interval
        whenever a previously-missing path recovers.  Falls back to the
        baseline ``watch_path_retry_interval_seconds`` when all paths are
        healthy.
        """
        _BACKOFF_CAP = 300.0
        backoff_interval = self._config.path_recovery_interval

        while not SHUTDOWN_EVENT.is_set():
            previously_missing = {
                id(s) for s in self._watch_states if s.path_missing
            }
            try:
                self._reconcile_watch_paths()
            except Exception:
                logger.exception(
                    "Unexpected error in _reconcile_watch_paths — "
                    "monitor thread will retry next cycle"
                )

            any_missing = any(s.path_missing for s in self._watch_states)
            any_recovered = any(
                id(s) in previously_missing and not s.path_missing
                for s in self._watch_states
            )

            if any_recovered:
                backoff_interval = self._config.path_recovery_interval

            if any_missing:
                interval = backoff_interval
                backoff_interval = min(backoff_interval * 2, _BACKOFF_CAP)
            else:
                interval = self._config.watch_path_retry_interval_seconds
                backoff_interval = self._config.path_recovery_interval

            SHUTDOWN_EVENT.wait(interval)

    def start(self) -> None:
        """Create and start observers for each configured watch path."""
        # Allow a stopped watcher to be started again in the same process.
        SHUTDOWN_EVENT.clear()
        self._processed_store.reopen()
        stability = FileStabilityChecker(
            required_checks=self._config.stability_checks,
            interval=self._config.stability_interval_seconds,
        )
        self._stability_executor = ThreadPoolExecutor(
            max_workers=self._config.max_concurrent_submissions,
            thread_name_prefix="watcher-stability",
        )

        self._watch_states = []
        for watch_cfg in self._config.watch_paths:
            handler = FileWatcherHandler(
                watch_cfg=watch_cfg,
                watcher_config=self._config,
                stability_checker=stability,
                processed_store=self._processed_store,
                submission_semaphore=self._semaphore,
                stability_executor=self._stability_executor,
                max_pending_files=self._max_pending_files,
            )
            self._watch_states.append(
                WatchPathRuntime(watch_cfg=watch_cfg, handler=handler)
            )

        self._reconcile_watch_paths()
        self._monitor_thread = threading.Thread(
            target=self._monitor_watch_paths,
            name="watcher-path-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

        for index, remote_cfg in enumerate(self._config.remote_ingest):
            poller = RemoteIngestPoller(
                remote_cfg=remote_cfg,
                watcher_config=self._config,
                processed_store=self._processed_store,
                submit_file=self._submit_remote_file,
                shutdown_event=SHUTDOWN_EVENT,
            )
            thread = threading.Thread(
                target=poller.run,
                daemon=True,
                name=f"watcher-remote-{index}",
            )
            thread.start()
            self._remote_threads.append(thread)
            logger.info(
                "Remote ingest polling: %s://%s:%s%s",
                remote_cfg.protocol,
                remote_cfg.host,
                remote_cfg.port,
                remote_cfg.remote_path,
            )

        logger.info(
            "File watcher started: %d path(s), %d remote source(s), mode=%s",
            len(self._watch_states),
            len(self._remote_threads),
            self._config.submission_mode,
        )

    def _submit_remote_file(self, file_path: str, remote_cfg, processed_key: str) -> bool:
        """Submit a staged remote file through the configured backend."""
        return _submit_with_config(
            file_path=file_path,
            watcher_config=self._config,
            watch_cfg=remote_cfg,
            processed_store=self._processed_store,
            submission_semaphore=self._semaphore,
            processed_key=processed_key,
        )

    def stop(self) -> None:
        """Stop all observers gracefully."""
        logger.info("Stopping file watcher...")
        SHUTDOWN_EVENT.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=10)
            self._monitor_thread = None

        for state in self._watch_states:
            self._stop_runtime_observer(state)

        self._observers.clear()
        for thread in self._remote_threads:
            try:
                thread.join(timeout=10)
            except Exception:
                logger.exception("Error joining remote poller thread")
        self._remote_threads.clear()
        self._watch_states.clear()
        self._watch_states.clear()
        if self._stability_executor is not None:
            self._stability_executor.shutdown(wait=False, cancel_futures=True)
            self._stability_executor = None
        self._processed_store.close()
        logger.info("File watcher stopped.")

    def status(self) -> list[dict]:
        """Return per-path health status for monitoring."""
        result = []
        for wp_rt in self._watch_states:
            result.append({
                "path": wp_rt.watch_cfg.path,
                "active": not wp_rt.path_missing,
                "observer_mode": wp_rt.active_mode,
                "recovery_retries": wp_rt.recovery_retries,
                "observer_deaths": wp_rt.observer_deaths,
            })
        return result

    def wait(self) -> None:
        """Block until SHUTDOWN_EVENT is set."""
        try:
            while not SHUTDOWN_EVENT.is_set():
                SHUTDOWN_EVENT.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _setup_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers for graceful shutdown."""
    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.info("Received %s, initiating shutdown", sig_name)
        SHUTDOWN_EVENT.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    """Configure stdlib logging for the watcher."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments, load config, and run the file watcher.

    Returns
    -------
    int
        Exit code (0 on clean shutdown, 1 on error).
    """
    parser = argparse.ArgumentParser(
        description="File system watcher for OCR pipeline hot-folder submission.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    _setup_logging(config.log_level)
    _setup_signal_handlers()

    watcher: Optional[FileWatcher] = None
    try:
        watcher = FileWatcher(config)
        watcher.start()
        watcher.wait()
    except Exception:
        logger.exception("Fatal error in file watcher")
        return 1
    finally:
        if watcher is not None:
            watcher.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
