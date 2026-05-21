"""Configuration loader for the file system watcher (Phase 5C).

Loads and validates YAML configuration for hot-folder monitoring.
Uses Pydantic for schema validation and PyYAML for file parsing.

Run with: python file_watcher.py --config file-watcher.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)

DEFAULT_FILE_PATTERNS = [
    "*.pdf", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png",
    "*.bmp", "*.gif", "*.webp", "*.jp2",
]


class WatchPathConfig(BaseModel):
    """Configuration for a single watched directory."""

    path: str
    recursive: bool = True
    observer_mode: Literal["auto", "native", "polling"] = "auto"
    polling_interval: float = 1.0
    patterns: list[str] = list(DEFAULT_FILE_PATTERNS)
    ignore_patterns: list[str] = []
    enable_docintel: bool = False
    docintel_mode: str = "full"
    priority: str = "normal"

    @field_validator("path")
    @classmethod
    def path_must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Watch path must not be empty")
        return v.strip()

    @field_validator("polling_interval")
    @classmethod
    def polling_interval_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("polling_interval must be > 0")
        return v

    @field_validator("patterns")
    @classmethod
    def patterns_must_be_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one file pattern is required")
        for pattern in v:
            if not pattern or not pattern.strip():
                raise ValueError("Pattern must not be empty")
        return v

    @field_validator("docintel_mode")
    @classmethod
    def docintel_mode_valid(cls, v: str) -> str:
        allowed = {"full", "layout_only", "tables_only"}
        if v not in allowed:
            raise ValueError(f"docintel_mode must be one of {allowed}")
        return v

    @field_validator("priority")
    @classmethod
    def priority_valid(cls, v: str) -> str:
        allowed = {"low", "normal", "high"}
        if v not in allowed:
            raise ValueError(f"priority must be one of {allowed}")
        return v


class RemoteIngestConfig(BaseModel):
    """Configuration for a single remote FTP/FTPS/SFTP source."""

    protocol: Literal["ftp", "ftps", "sftp"]
    host: str
    port: int | None = None
    username: str = ""
    password: str = ""
    key_filename: str = ""
    known_hosts_path: str = ""
    allow_unknown_host: bool = False
    passive_mode: bool = True
    remote_path: str = "/"
    patterns: list[str] = list(DEFAULT_FILE_PATTERNS)
    ignore_patterns: list[str] = []
    enable_docintel: bool = False
    docintel_mode: str = "full"
    priority: str = "normal"
    poll_interval_seconds: float = 15.0
    timeout_seconds: float = 30.0
    delete_after_fetch: bool = False
    max_file_size_mb: float = 0  # 0 = unlimited

    @field_validator("host")
    @classmethod
    def host_must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Remote host must not be empty")
        return v.strip()

    @field_validator("remote_path")
    @classmethod
    def remote_path_must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("remote_path must not be empty")
        return v.strip()

    @field_validator("patterns")
    @classmethod
    def remote_patterns_must_be_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one remote file pattern is required")
        for pattern in v:
            if not pattern or not pattern.strip():
                raise ValueError("Pattern must not be empty")
        return v

    @field_validator("docintel_mode")
    @classmethod
    def remote_docintel_mode_valid(cls, v: str) -> str:
        allowed = {"full", "layout_only", "tables_only"}
        if v not in allowed:
            raise ValueError(f"docintel_mode must be one of {allowed}")
        return v

    @field_validator("priority")
    @classmethod
    def remote_priority_valid(cls, v: str) -> str:
        allowed = {"low", "normal", "high"}
        if v not in allowed:
            raise ValueError(f"priority must be one of {allowed}")
        return v

    @field_validator("poll_interval_seconds")
    @classmethod
    def poll_interval_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def timeout_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timeout_seconds must be > 0")
        return v

    @model_validator(mode="after")
    def default_port(self) -> "RemoteIngestConfig":
        if self.port is not None:
            return self
        if self.protocol in {"ftp", "ftps"}:
            self.port = 21
        else:
            self.port = 22
        return self


class WatcherConfig(BaseModel):
    """Top-level watcher configuration."""

    watch_paths: list[WatchPathConfig] = []
    remote_ingest: list[RemoteIngestConfig] = []
    submission_mode: Literal["pipeline", "api", "distributed"] = "pipeline"

    stability_checks: int = 3
    stability_interval_seconds: float = 2.0

    api_url: str = "http://localhost:8000"
    api_key: str = ""

    coordinator_url: str = "http://localhost:8001"
    coordinator_api_key: str = ""

    pipeline_script: str = "ocr_gpu_async.py"
    source_folder: str = "/app/ocr_source"
    output_folder: str = "/app/ocr_output"
    remote_staging_dir: str = "/app/ocr_output/remote-ingest"

    log_level: str = "INFO"
    processed_db: str = ""
    max_concurrent_submissions: int = 4
    watch_path_retry_interval_seconds: float = 5.0
    path_recovery_enabled: bool = True
    path_recovery_interval: float = 30.0
    path_recovery_max_retries: int = 0

    @field_validator("remote_staging_dir")
    @classmethod
    def remote_staging_dir_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("remote_staging_dir must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def at_least_one_input(self) -> "WatcherConfig":
        if not self.watch_paths and not self.remote_ingest:
            raise ValueError(
                "At least one watch_path or remote_ingest source is required"
            )
        return self

    @field_validator("stability_checks")
    @classmethod
    def stability_checks_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("stability_checks must be >= 1")
        return v

    @field_validator("stability_interval_seconds")
    @classmethod
    def stability_interval_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("stability_interval_seconds must be > 0")
        return v

    @field_validator("max_concurrent_submissions")
    @classmethod
    def max_concurrent_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_concurrent_submissions must be >= 1")
        return v

    @field_validator("watch_path_retry_interval_seconds")
    @classmethod
    def watch_path_retry_interval_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("watch_path_retry_interval_seconds must be > 0")
        return v

    @field_validator("path_recovery_interval")
    @classmethod
    def path_recovery_interval_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("path_recovery_interval must be > 0")
        return v

    @field_validator("path_recovery_max_retries")
    @classmethod
    def path_recovery_max_retries_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("path_recovery_max_retries must be >= 0")
        return v

    @field_validator("log_level")
    @classmethod
    def log_level_valid(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @model_validator(mode="after")
    def validate_paths_exist(self) -> "WatcherConfig":
        """Warn (but do not fail) if watch paths do not exist yet.

        On Windows, ``Path.exists()`` raises ``OSError`` for unreachable UNC
        paths (e.g. ``\\\\server\\share``) instead of returning ``False``.
        We catch that so config validation never crashes on network paths.
        """
        for watch_path in self.watch_paths:
            path = Path(watch_path.path)
            try:
                exists = path.exists()
            except OSError:
                exists = False
            if not exists:
                logger.warning("Watch path does not exist: %s", watch_path.path)
        return self


def load_config(config_path: str) -> WatcherConfig:
    """Load and validate watcher configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping, got {type(data).__name__}"
        )

    try:
        return WatcherConfig(**data)
    except Exception as exc:
        raise ValueError(f"Config validation error: {exc}") from exc


def load_config_from_dict(data: dict) -> WatcherConfig:
    """Create a WatcherConfig from an already-parsed dictionary."""
    return WatcherConfig(**data)
