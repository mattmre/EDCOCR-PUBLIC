"""Standalone SFTP ingest module for OCR pipeline (Task 8).

Polls a remote SFTP server for new files and stages them into the local
``ocr_source/`` directory where the existing pipeline or file watcher picks
them up automatically.

This module is a thin convenience wrapper around the existing
``file_watcher_remote.RemoteIngestPoller`` transport and
``file_watcher.ProcessedFileStore`` dedup store.  It provides:

- Environment-variable-only configuration (no YAML needed)
- Graceful degradation when ``paramiko`` is not installed
- Standalone CLI entry point

Configuration is controlled via environment variables:

    SFTP_INGEST_ENABLED           Enable the poller (default: false)
    SFTP_INGEST_HOST              SFTP server hostname (required)
    SFTP_INGEST_PORT              SFTP server port (default: 22)
    SFTP_INGEST_USERNAME          Login username (required)
    SFTP_INGEST_PASSWORD          Login password (prefer key auth)
    SFTP_INGEST_KEY_PATH          Path to SSH private key
    SFTP_INGEST_KNOWN_HOSTS       Path to known_hosts file
    SFTP_INGEST_ALLOW_UNKNOWN     Accept unknown host keys (default: false)
    SFTP_INGEST_REMOTE_PATH       Remote directory to poll (default: /ingest)
    SFTP_INGEST_LOCAL_STAGING     Local staging directory (default: ocr_source/sftp_staging)
    SFTP_INGEST_POLL_INTERVAL     Seconds between polls (default: 30.0)
    SFTP_INGEST_DELETE_AFTER      Delete remote file after download (default: false)
    SFTP_INGEST_EXTENSIONS        Comma-separated file extensions (default: .pdf,.tif,.tiff,.jpg,.jpeg,.png)
    SFTP_INGEST_MAX_FILE_SIZE_MB  Maximum file size in MB (default: 500)
    SFTP_INGEST_TIMEOUT           Connection timeout in seconds (default: 30.0)
    SFTP_INGEST_PROCESSED_DB      SQLite path for processed-file persistence

Usage:
    python sftp_ingest.py
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("sftp_ingest")

# ---------------------------------------------------------------------------
# Paramiko availability check
# ---------------------------------------------------------------------------

try:
    import paramiko  # noqa: F401

    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

DEFAULT_PORT = 22
DEFAULT_REMOTE_PATH = "/ingest"
DEFAULT_LOCAL_STAGING = "ocr_source/sftp_staging"
DEFAULT_POLL_INTERVAL = 30.0
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_FILE_SIZE_MB = 500
DEFAULT_EXTENSIONS = [".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png"]
DEFAULT_DESTINATION = "ocr_source"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SFTPIngestConfig:
    """Configuration for standalone SFTP ingest polling."""

    enabled: bool = False
    host: str = ""
    port: int = DEFAULT_PORT
    username: str = ""
    password: str = ""
    private_key_path: str = ""
    known_hosts_path: str = ""
    allow_unknown_host: bool = False
    remote_path: str = DEFAULT_REMOTE_PATH
    local_staging_path: str = DEFAULT_LOCAL_STAGING
    destination_path: str = DEFAULT_DESTINATION
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL
    delete_after_download: bool = False
    file_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    timeout_seconds: float = DEFAULT_TIMEOUT
    processed_db: str = ""

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty if valid)."""
        errors: list[str] = []
        if not self.host or not self.host.strip():
            errors.append("SFTP host is required (SFTP_INGEST_HOST)")
        if not self.username or not self.username.strip():
            errors.append("SFTP username is required (SFTP_INGEST_USERNAME)")
        if not self.password and not self.private_key_path:
            errors.append(
                "Either password (SFTP_INGEST_PASSWORD) or private key "
                "(SFTP_INGEST_KEY_PATH) is required"
            )
        if self.port < 1 or self.port > 65535:
            errors.append(f"Port must be 1-65535, got {self.port}")
        if self.poll_interval_seconds <= 0:
            errors.append("Poll interval must be > 0")
        if self.timeout_seconds <= 0:
            errors.append("Timeout must be > 0")
        if self.max_file_size_mb < 1:
            errors.append("Max file size must be >= 1 MB")
        if not self.file_extensions:
            errors.append("At least one file extension is required")
        return errors


def _parse_bool(value: str) -> bool:
    """Parse a boolean from an environment variable string."""
    return value.strip().lower() in ("1", "true", "yes")


def _parse_extensions(raw: str) -> list[str]:
    """Parse comma-separated extension list into glob patterns.

    Accepts both ``.pdf`` and ``pdf`` forms and normalises them to
    ``*.pdf`` glob patterns for compatibility with the underlying poller.
    """
    extensions = []
    for ext in raw.split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        extensions.append(ext)
    return extensions


def load_config_from_env() -> SFTPIngestConfig:
    """Build an ``SFTPIngestConfig`` from environment variables."""
    raw_extensions = os.environ.get("SFTP_INGEST_EXTENSIONS", "")
    extensions = (
        _parse_extensions(raw_extensions) if raw_extensions.strip() else list(DEFAULT_EXTENSIONS)
    )

    port_raw = os.environ.get("SFTP_INGEST_PORT", str(DEFAULT_PORT))
    try:
        port = int(port_raw)
    except (ValueError, TypeError):
        port = DEFAULT_PORT

    poll_raw = os.environ.get("SFTP_INGEST_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL))
    try:
        poll_interval = float(poll_raw)
    except (ValueError, TypeError):
        poll_interval = DEFAULT_POLL_INTERVAL

    timeout_raw = os.environ.get("SFTP_INGEST_TIMEOUT", str(DEFAULT_TIMEOUT))
    try:
        timeout = float(timeout_raw)
    except (ValueError, TypeError):
        timeout = DEFAULT_TIMEOUT

    max_size_raw = os.environ.get("SFTP_INGEST_MAX_FILE_SIZE_MB", str(DEFAULT_MAX_FILE_SIZE_MB))
    try:
        max_size = int(max_size_raw)
    except (ValueError, TypeError):
        max_size = DEFAULT_MAX_FILE_SIZE_MB

    return SFTPIngestConfig(
        enabled=_parse_bool(os.environ.get("SFTP_INGEST_ENABLED", "")),
        host=os.environ.get("SFTP_INGEST_HOST", "").strip(),
        port=port,
        username=os.environ.get("SFTP_INGEST_USERNAME", "").strip(),
        password=os.environ.get("SFTP_INGEST_PASSWORD", ""),
        private_key_path=os.environ.get("SFTP_INGEST_KEY_PATH", "").strip(),
        known_hosts_path=os.environ.get("SFTP_INGEST_KNOWN_HOSTS", "").strip(),
        allow_unknown_host=_parse_bool(os.environ.get("SFTP_INGEST_ALLOW_UNKNOWN", "")),
        remote_path=os.environ.get("SFTP_INGEST_REMOTE_PATH", DEFAULT_REMOTE_PATH).strip(),
        local_staging_path=os.environ.get(
            "SFTP_INGEST_LOCAL_STAGING", DEFAULT_LOCAL_STAGING
        ).strip(),
        destination_path=os.environ.get(
            "SFTP_INGEST_DESTINATION", DEFAULT_DESTINATION
        ).strip(),
        poll_interval_seconds=poll_interval,
        delete_after_download=_parse_bool(os.environ.get("SFTP_INGEST_DELETE_AFTER", "")),
        file_extensions=extensions,
        max_file_size_mb=max_size,
        timeout_seconds=timeout,
        processed_db=os.environ.get("SFTP_INGEST_PROCESSED_DB", "").strip(),
    )


# ---------------------------------------------------------------------------
# Extension-to-glob conversion helper
# ---------------------------------------------------------------------------


def _extensions_to_patterns(extensions: list[str]) -> list[str]:
    """Convert file extensions to glob-style patterns for the poller.

    ``[".pdf", ".tif"]`` becomes ``["*.pdf", "*.tif"]``.
    """
    patterns = []
    for ext in extensions:
        ext = ext.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        patterns.append(f"*{ext}")
    return patterns


# ---------------------------------------------------------------------------
# Standalone SFTP Ingest Service
# ---------------------------------------------------------------------------


class SFTPIngestService:
    """Standalone SFTP ingest that polls a remote server and stages files locally.

    Downloaded files are placed into ``destination_path`` (default ``ocr_source/``)
    for the existing pipeline or file watcher to pick up.

    Parameters
    ----------
    config : SFTPIngestConfig
        Configuration for the SFTP connection and polling behaviour.
    """

    def __init__(self, config: SFTPIngestConfig):
        self._config = config
        self._shutdown_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processed_store: Optional[object] = None
        self._poller: Optional[object] = None

    @property
    def config(self) -> SFTPIngestConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the SFTP polling thread.

        Raises
        ------
        RuntimeError
            If paramiko is not installed or configuration is invalid.
        """
        if not PARAMIKO_AVAILABLE:
            raise RuntimeError(
                "paramiko is required for SFTP ingest but is not installed. "
                "Install it with: pip install paramiko>=3.4"
            )

        errors = self._config.validate()
        if errors:
            raise RuntimeError(
                "SFTP ingest configuration is invalid:\n  - "
                + "\n  - ".join(errors)
            )

        # Allow restart after stop in the same process
        self._shutdown_event.clear()

        # Create processed file store
        from file_watcher import ProcessedFileStore

        self._processed_store = ProcessedFileStore(self._config.processed_db)

        # Build the remote config bridge to the existing poller
        remote_cfg = self._build_remote_config()

        # Build a minimal watcher config for staging directory
        watcher_cfg = self._build_watcher_config()

        from file_watcher_remote import RemoteIngestPoller

        self._poller = RemoteIngestPoller(
            remote_cfg=remote_cfg,
            watcher_config=watcher_cfg,
            processed_store=self._processed_store,
            submit_file=self._stage_file,
            shutdown_event=self._shutdown_event,
        )

        self._thread = threading.Thread(
            target=self._poller.run,
            daemon=True,
            name="sftp-ingest-poller",
        )
        self._thread.start()
        logger.info(
            "SFTP ingest started: sftp://%s:%d%s -> %s (poll every %.0fs)",
            self._config.host,
            self._config.port,
            self._config.remote_path,
            self._config.destination_path,
            self._config.poll_interval_seconds,
        )

    def stop(self) -> None:
        """Stop the polling thread gracefully."""
        logger.info("Stopping SFTP ingest...")
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        if self._processed_store is not None:
            self._processed_store.close()
            self._processed_store = None
        logger.info("SFTP ingest stopped.")

    def wait(self) -> None:
        """Block until shutdown is requested."""
        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass

    def poll_once(self) -> None:
        """Execute a single poll cycle (useful for testing and scripting).

        Raises
        ------
        RuntimeError
            If paramiko is not installed or configuration is invalid.
        """
        if not PARAMIKO_AVAILABLE:
            raise RuntimeError(
                "paramiko is required for SFTP ingest but is not installed."
            )

        errors = self._config.validate()
        if errors:
            raise RuntimeError(
                "SFTP ingest configuration is invalid:\n  - "
                + "\n  - ".join(errors)
            )

        from file_watcher import ProcessedFileStore
        from file_watcher_remote import RemoteIngestPoller

        store = ProcessedFileStore(self._config.processed_db)
        try:
            remote_cfg = self._build_remote_config()
            watcher_cfg = self._build_watcher_config()
            poller = RemoteIngestPoller(
                remote_cfg=remote_cfg,
                watcher_config=watcher_cfg,
                processed_store=store,
                submit_file=self._stage_file,
                shutdown_event=threading.Event(),
            )
            poller.poll_once()
        finally:
            store.close()

    def _build_remote_config(self) -> object:
        """Build a RemoteIngestConfig-compatible namespace for the poller."""
        from file_watcher_config import RemoteIngestConfig

        patterns = _extensions_to_patterns(self._config.file_extensions)
        return RemoteIngestConfig(
            protocol="sftp",
            host=self._config.host,
            port=self._config.port,
            username=self._config.username,
            password=self._config.password,
            key_filename=self._config.private_key_path,
            known_hosts_path=self._config.known_hosts_path,
            allow_unknown_host=self._config.allow_unknown_host,
            remote_path=self._config.remote_path,
            patterns=patterns,
            ignore_patterns=[],
            enable_docintel=False,
            docintel_mode="full",
            priority="normal",
            poll_interval_seconds=self._config.poll_interval_seconds,
            timeout_seconds=self._config.timeout_seconds,
            delete_after_fetch=self._config.delete_after_download,
        )

    def _build_watcher_config(self) -> object:
        """Build a minimal WatcherConfig-compatible namespace for staging dir."""
        from types import SimpleNamespace

        return SimpleNamespace(
            remote_staging_dir=self._config.local_staging_path,
        )

    def _stage_file(
        self, staged_path: str, _remote_cfg: object, processed_key: str
    ) -> bool:
        """Move a downloaded file from staging to the destination directory.

        This method replaces the normal watcher submission backends.  Instead
        of launching a subprocess or POSTing to the API, it simply copies the
        file into ``destination_path`` where the existing pipeline or file
        watcher can pick it up.

        Returns True on success so the remote dedup key is recorded.
        """
        try:
            src = Path(staged_path)
            if not src.exists() or not src.is_file():
                logger.warning("Staged file does not exist: %s", staged_path)
                return False

            # Enforce max file size
            file_size_mb = src.stat().st_size / (1024 * 1024)
            if file_size_mb > self._config.max_file_size_mb:
                logger.warning(
                    "File exceeds max size (%d MB > %d MB): %s",
                    int(file_size_mb),
                    self._config.max_file_size_mb,
                    staged_path,
                )
                return False

            # Check already processed
            if self._processed_store is not None and self._processed_store.contains(
                processed_key
            ):
                logger.debug("Already processed: %s", processed_key)
                return False

            dest_dir = Path(self._config.destination_path)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            # Avoid overwriting existing files by adding a suffix
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                counter = 1
                while dest.exists():
                    dest = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.copy2(str(src), str(dest))
            logger.info(
                "Staged file to destination: %s -> %s",
                src.name,
                dest,
            )

            # Mark as processed
            if self._processed_store is not None:
                self._processed_store.add(processed_key)

            return True
        except Exception:
            logger.exception("Failed to stage file to destination: %s", staged_path)
            return False


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_SERVICE_INSTANCE: Optional[SFTPIngestService] = None


def _setup_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers for graceful shutdown."""

    def _handler(signum, _frame):
        sig_name = (
            signal.Signals(signum).name
            if hasattr(signal, "Signals")
            else str(signum)
        )
        logger.info("Received %s, initiating shutdown", sig_name)
        if _SERVICE_INSTANCE is not None:
            _SERVICE_INSTANCE.stop()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    """Configure stdlib logging for SFTP ingest."""
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
    """Run the standalone SFTP ingest service.

    Returns
    -------
    int
        Exit code (0 on clean shutdown, 1 on error).
    """
    global _SERVICE_INSTANCE

    parser = argparse.ArgumentParser(
        description="Standalone SFTP ingest for OCR pipeline.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("SFTP_INGEST_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log verbosity level (default: INFO)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll cycle and exit",
    )
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)

    if not PARAMIKO_AVAILABLE:
        logger.error(
            "paramiko is not installed. Install with: pip install paramiko>=3.4"
        )
        return 1

    config = load_config_from_env()

    if not config.enabled:
        logger.info(
            "SFTP ingest is disabled. Set SFTP_INGEST_ENABLED=true to enable."
        )
        return 0

    errors = config.validate()
    if errors:
        for err in errors:
            logger.error("Config error: %s", err)
        return 1

    # Mask credentials in startup log
    logger.info(
        "SFTP ingest config: host=%s port=%d user=%s remote=%s destination=%s",
        config.host,
        config.port,
        config.username,
        config.remote_path,
        config.destination_path,
    )

    service = SFTPIngestService(config)

    if args.once:
        try:
            service.poll_once()
            return 0
        except Exception:
            logger.exception("Single poll cycle failed")
            return 1

    _SERVICE_INSTANCE = service
    _setup_signal_handlers()

    try:
        service.start()
        service.wait()
    except RuntimeError as exc:
        logger.error("Failed to start SFTP ingest: %s", exc)
        return 1
    except Exception:
        logger.exception("Fatal error in SFTP ingest")
        return 1
    finally:
        service.stop()
        _SERVICE_INSTANCE = None

    return 0


if __name__ == "__main__":
    sys.exit(main())
