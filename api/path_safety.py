"""Path containment helpers for server-side file operations."""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath
from typing import Iterable

from fastapi import HTTPException

_WINDOWS_DEVICE_PREFIXES = ("\\\\.\\", "\\\\?\\")
_SAFE_SOURCE_PATH_RE = re.compile(r"^[a-zA-Z0-9_\-/\\.:\s]+$")


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return True when ``path`` is inside ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_relative_to_windows(path: PureWindowsPath, root: PureWindowsPath) -> bool:
    """Return True when a Windows path is inside a Windows root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _looks_like_windows_absolute(path_value: str) -> bool:
    """Detect Windows absolute paths independent of the current OS."""
    normalized = path_value.replace("/", "\\")
    return bool(re.match(r"^[a-zA-Z]:\\", normalized) or normalized.startswith("\\\\"))


def _resolve_allowed_roots(roots: Iterable[str]) -> tuple[Path, ...]:
    """Resolve configured root directories into canonical absolute paths."""
    resolved_roots: list[Path] = []
    for root in roots:
        if not root:
            continue
        resolved_roots.append(Path(root).expanduser().resolve(strict=False))
    return tuple(resolved_roots)


def _validate_raw_path(path_value: str, field_name: str) -> None:
    """Reject empty and device-style path values before path resolution."""
    if not path_value or not path_value.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_path",
                "message": f"{field_name} must be a non-empty absolute path.",
            },
        )

    if os.name == "nt":
        normalized = path_value.replace("/", "\\")
        if normalized.startswith(_WINDOWS_DEVICE_PREFIXES):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_path",
                    "message": f"{field_name} device paths are not allowed.",
                },
            )


def ensure_path_within_roots(
    *,
    path_value: str,
    field_name: str,
    allowed_roots: Iterable[str],
) -> Path:
    """Resolve and validate a path is contained within one of the allowed roots."""
    _validate_raw_path(path_value, field_name)

    root_values = tuple(root for root in allowed_roots if root)
    if _looks_like_windows_absolute(path_value) or any(
        _looks_like_windows_absolute(root) for root in root_values
    ):
        candidate_windows = PureWindowsPath(path_value)
        if not candidate_windows.is_absolute():
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_path",
                    "message": f"{field_name} must be an absolute path.",
                },
            )

        resolved_windows_roots = tuple(PureWindowsPath(root) for root in root_values)
        if not resolved_windows_roots:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "server_misconfigured",
                    "message": "Path allowlist is not configured.",
                },
            )

        if not any(
            _is_relative_to_windows(candidate_windows, root)
            for root in resolved_windows_roots
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "path_not_allowed",
                    "message": f"{field_name} is outside allowed roots.",
                    "details": {
                        "field": field_name,
                        "path": str(candidate_windows),
                        "allowed_roots": [str(root) for root in resolved_windows_roots],
                    },
                },
            )

        return Path(str(candidate_windows))

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_path",
                "message": f"{field_name} must be an absolute path.",
            },
        )

    resolved_path = candidate.resolve(strict=False)
    resolved_roots = _resolve_allowed_roots(root_values)
    if not resolved_roots:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_misconfigured",
                "message": "Path allowlist is not configured.",
            },
        )

    if not any(_is_relative_to(resolved_path, root) for root in resolved_roots):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "path_not_allowed",
                "message": f"{field_name} is outside allowed roots.",
                "details": {
                    "field": field_name,
                    "path": str(resolved_path),
                    "allowed_roots": [str(root) for root in resolved_roots],
                },
            },
        )

    return resolved_path


def validate_source_path_input(
    *,
    path_value: str,
    field_name: str,
    allowed_roots: Iterable[str],
) -> Path:
    """Validate a user-provided source path before server-side ingestion."""
    if ".." in path_value or not _SAFE_SOURCE_PATH_RE.match(path_value):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_source_path",
                "message": f"{field_name} contains disallowed characters.",
            },
        )

    return ensure_path_within_roots(
        path_value=path_value,
        field_name=field_name,
        allowed_roots=allowed_roots,
    )
