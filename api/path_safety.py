"""Path containment helpers for server-side file operations."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

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
    """Reject empty, parent-segment, and device-style path values."""
    if not path_value or not path_value.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_path",
                "message": f"{field_name} must be a non-empty absolute path.",
            },
        )

    # RC-09 / C01-A12: reject residual ".." on all OS branches (Windows pure-path
    # previously accepted PureWindowsPath.relative_to with leftover .. segments).
    posix_norm = str(path_value).replace("\\", "/")
    parts = Path(posix_norm).parts
    if ".." in parts or "/../" in f"/{posix_norm}/":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_path",
                "message": f"{field_name} must not contain parent-directory segments.",
            },
        )

    # Device-namespace prefixes: reject whenever present (not only on Windows).
    win_norm = str(path_value).replace("/", "\\")
    if win_norm.startswith(_WINDOWS_DEVICE_PREFIXES):
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
                    # Do not disclose host filesystem layout to API callers.
                    "details": {"field": field_name},
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
                # Do not disclose host filesystem layout to API callers.
                "details": {"field": field_name},
            },
        )

    return resolved_path


def resolve_job_output_path(
    job_id: str,
    result_path: str | Path | None,
    output_root: str | Path,
) -> Path:
    """Resolve a job output directory only when it is job-bound.

    ``result_path`` is persisted state, not an authorization boundary.  A
    poisoned row must not redirect a worker or artifact route to another
    job's directory merely because the path is inside the global output root.
    Empty legacy values resolve to the canonical ``OUTPUT_ROOT/job_id`` path;
    non-empty values must resolve to that exact path.
    """
    job_id_s = str(job_id or "")
    if not job_id_s or Path(job_id_s).name != job_id_s:
        raise ValueError("job output identity is invalid")

    try:
        # Resolve the configured root, but keep the job component lexical.
        # Resolving ``root/job_id`` would turn a symlinked job directory into
        # an apparently canonical foreign output tree.
        resolved_root = Path(output_root).expanduser().resolve(strict=False)
        expected = resolved_root / job_id_s
        candidate = (
            expected
            if not result_path
            else Path(result_path).expanduser().resolve(strict=False)
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("job output path is not canonical") from exc

    if candidate != expected or expected.is_symlink():
        raise ValueError("job output path is not canonical")
    return expected


def assert_safe_download_inode(
    *,
    path: Path,
    field_name: str = "artifact_path",
    reject_hardlinks: bool = True,
) -> Path:
    """Refuse symlink and multi-hardlink download targets (A14 / Sweep A).

    Path jail alone does not stop a same-FS hardlink of a foreign inode into
    the job tree. ``FileResponse`` would then serve foreign content under a
    jailed path string.
    """
    try:
        st = path.lstat()
    except OSError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "artifact_not_found",
                "message": f"{field_name} is not readable.",
            },
        ) from exc

    import stat as stat_mod

    if stat_mod.S_ISLNK(st.st_mode) or path.is_symlink():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "artifact_symlink_refused",
                "message": f"{field_name} must not be a symbolic link.",
            },
        )
    if reject_hardlinks and getattr(st, "st_nlink", 1) > 1:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "artifact_hardlink_refused",
                "message": (
                    f"{field_name} has multiple hard links; refusing download "
                    "(A14 TOCTOU / foreign-inode residual)."
                ),
            },
        )
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "artifact_not_found",
                "message": f"{field_name} is not a regular file.",
            },
        )
    # LT-12: open without following a symlink race after lstat (O_NOFOLLOW when available).
    import os as _os_nf

    flags = _os_nf.O_RDONLY | getattr(_os_nf, "O_NONBLOCK", 0)
    nofollow = getattr(_os_nf, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    try:
        fd = _os_nf.open(path, flags)
        try:
            st2 = _os_nf.fstat(fd)
        finally:
            _os_nf.close(fd)
        if reject_hardlinks and getattr(st2, "st_nlink", 1) > 1:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "artifact_hardlink_refused",
                    "message": (
                        f"{field_name} has multiple hard links after open; "
                        "refusing download (LT-12 O_NOFOLLOW recheck)."
                    ),
                },
            )
    except OSError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "artifact_open_refused",
                "message": f"{field_name} cannot be opened without following links.",
            },
        ) from exc
    return path


def secure_file_response(path: Path, *, media_type: str, filename: str):
    """Stream an O_NOFOLLOW-bound inode instead of reopening a path later.

    Path-based response classes open after the route returns. Binding the
    descriptor here closes the check/open TOCTOU window for download routes.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "artifact_not_found",
                "message": "Artifact is not readable.",
            },
        ) from exc

    try:
        inode = os.fstat(fd)
        if not stat.S_ISREG(inode.st_mode) or getattr(inode, "st_nlink", 1) > 1:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "artifact_inode_refused",
                    "message": "Artifact inode is not eligible for download.",
                },
            )
        response = _stream_snapshot_response(
            fd,
            inode,
            media_type=media_type,
            filename=filename,
        )
        return response
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "artifact_changed",
                "message": "Artifact changed during snapshot capture.",
            },
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)


_ROOT_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_ROOT_FILE_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_SNAPSHOT_CREATE_FLAGS = (
    os.O_RDWR
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


@dataclass
class _BoundJobDirectory:
    """Configured storage root and one job child bound to stable inodes."""

    storage_root: Path
    job_root: Path
    storage_fd: int
    job_fd: int

    def assert_unchanged(self) -> None:
        _assert_directory_fd_matches_path(self.storage_root, self.storage_fd)
        _assert_directory_fd_matches_path(self.job_root, self.job_fd)

    def close(self) -> None:
        for fd in (self.job_fd, self.storage_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.job_fd = -1
        self.storage_fd = -1


def _bind_job_directory_under_root(
    storage_root: str | os.PathLike[str],
    job_id: str,
    *,
    persisted_result_path: str | os.PathLike[str] | None = None,
) -> _BoundJobDirectory:
    """Bind a configured storage root before authorizing its job child.

    The configured root name, not a separately resolved absolute job path, is
    the namespace authority. The child is opened relative to that held parent
    descriptor, and persisted state must lexically identify that exact child.
    """
    job_id_s = str(job_id or "")
    if not job_id_s or Path(job_id_s).name != job_id_s:
        raise ValueError("job output identity is invalid")
    storage_path = Path(os.path.abspath(os.path.expanduser(os.fspath(storage_root))))
    storage_fd = -1
    job_fd = -1
    try:
        storage_fd, _metadata = _open_directory_bound(storage_path)
        job_root = storage_path / job_id_s
        if persisted_result_path:
            raw_result = os.fspath(persisted_result_path)
            if not isinstance(raw_result, str) or ".." in Path(raw_result).parts:
                raise ValueError("job output path is not canonical")
            persisted = Path(os.path.abspath(os.path.expanduser(raw_result)))
            if persisted != job_root:
                raise ValueError("job output path is not canonical")
        job_fd = _open_directory_under_root_fd(storage_fd, job_id_s)
        binding = _BoundJobDirectory(
            storage_root=storage_path,
            job_root=job_root,
            storage_fd=storage_fd,
            job_fd=job_fd,
        )
        binding.assert_unchanged()
        return binding
    except Exception:
        if job_fd >= 0:
            os.close(job_fd)
        if storage_fd >= 0:
            os.close(storage_fd)
        raise


def _open_directory_bound(root: Path) -> tuple[int, os.stat_result]:
    """Open *root* without following links and reject an open-time name swap."""
    before = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError("authorized root must be a real directory")
    fd = os.open(root, _ROOT_DIRECTORY_OPEN_FLAGS)
    try:
        opened = os.fstat(fd)
        after = os.stat(root, follow_symlinks=False)
        expected = (before.st_dev, before.st_ino)
        if expected != (opened.st_dev, opened.st_ino) or expected != (
            after.st_dev,
            after.st_ino,
        ):
            raise ValueError("authorized root changed while it was opened")
        return fd, opened
    except Exception:
        os.close(fd)
        raise


def _assert_directory_fd_matches_path(root: Path, directory_fd: int) -> None:
    """Fail when the lexical authorized-root name no longer names *directory_fd*."""
    opened = os.fstat(directory_fd)
    current = os.stat(root, follow_symlinks=False)
    if (
        not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ValueError("authorized root changed after it was opened")


def _directory_fd_alias(directory_fd: int) -> Path:
    """Return a pathname that remains bound to an open directory descriptor.

    Existing atomic JSON/custody writers are pathname based. Linux ``/proc``
    and POSIX ``/dev/fd`` aliases let those writers stay attached to the
    already-authorized inode if the original job-root name is replaced. A
    platform without a verifiable descriptor alias fails closed.
    """
    opened = os.fstat(directory_fd)
    if not stat.S_ISDIR(opened.st_mode):
        raise ValueError("directory_fd must reference a directory")
    for base in (Path("/proc/self/fd"), Path("/dev/fd")):
        candidate = base / str(directory_fd)
        try:
            current = os.stat(candidate)
        except OSError:
            continue
        if (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino):
            return candidate
    raise OSError("no verifiable directory-descriptor path is available")


def _normalize_relative_access_path(relative_path: str | os.PathLike[str]) -> str:
    try:
        normalized = os.fspath(relative_path)
    except TypeError as exc:
        raise ValueError("relative_path must be a non-empty string") from exc
    if not isinstance(normalized, str) or not normalized:
        raise ValueError("relative_path must be a non-empty string")
    return normalized.replace("\\", "/")


def _open_directory_under_root_fd(
    root_fd: int,
    relative_path: str | os.PathLike[str],
) -> int:
    """Open a descendant directory from an already-bound authorized root."""
    normalized = _normalize_relative_access_path(relative_path)
    parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or "\x00" in normalized
    ):
        raise ValueError("relative_path must stay below the authorized root")
    current_fd = os.dup(root_fd)
    try:
        for part in parts:
            child_fd = os.open(part, _ROOT_DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_or_create_directory_under_root_fd(
    root_fd: int,
    relative_path: str | os.PathLike[str],
    *,
    mode: int = 0o750,
) -> int:
    """Open/create a descendant directory without following any component."""
    normalized = _normalize_relative_access_path(relative_path)
    parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or "\x00" in normalized
    ):
        raise ValueError("relative_path must stay below the authorized root")
    current_fd = os.dup(root_fd)
    try:
        for part in parts:
            try:
                child_fd = os.open(
                    part,
                    _ROOT_DIRECTORY_OPEN_FLAGS,
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=mode, dir_fd=current_fd)
                except FileExistsError:
                    # Another writer won the create race; the O_NOFOLLOW open
                    # below determines whether it created a real directory.
                    pass
                child_fd = os.open(
                    part,
                    _ROOT_DIRECTORY_OPEN_FLAGS,
                    dir_fd=current_fd,
                )
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_regular_file_under_root_fd(
    root_fd: int,
    relative_path: str | os.PathLike[str],
    *,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    expected_size: int | None = None,
    expected_mtime_ns: int | None = None,
    expected_ctime_ns: int | None = None,
) -> tuple[int, os.stat_result]:
    """Open a regular descendant relative to an already-bound root fd."""
    normalized = _normalize_relative_access_path(relative_path)
    parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or "\x00" in normalized
    ):
        raise ValueError("relative_path must stay below the authorized root")

    current_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child_fd = os.open(
                part,
                _ROOT_DIRECTORY_OPEN_FLAGS,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = child_fd
        file_fd = os.open(
            parts[-1],
            _ROOT_FILE_OPEN_FLAGS,
            dir_fd=current_fd,
        )
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode) or getattr(metadata, "st_nlink", 1) > 1:
            os.close(file_fd)
            raise ValueError("artifact inode is not eligible for access")
        if (
            expected_device is not None
            and expected_inode is not None
            and (metadata.st_dev, metadata.st_ino) != (expected_device, expected_inode)
        ):
            os.close(file_fd)
            raise ValueError("artifact inode changed after discovery")
        if expected_size is not None and metadata.st_size != expected_size:
            os.close(file_fd)
            raise ValueError("artifact size changed after discovery")
        if expected_mtime_ns is not None and metadata.st_mtime_ns != expected_mtime_ns:
            os.close(file_fd)
            raise ValueError("artifact modification time changed after discovery")
        if expected_ctime_ns is not None and metadata.st_ctime_ns != expected_ctime_ns:
            os.close(file_fd)
            raise ValueError("artifact change time changed after discovery")
        os.close(current_fd)
        current_fd = -1
        return file_fd, metadata
    except Exception:
        if current_fd >= 0:
            os.close(current_fd)
        raise


def _inode_snapshot_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _snapshot_open_regular_file(
    fd: int,
    metadata: os.stat_result,
):
    """Capture one stable inode version before constructing a response."""
    snapshot = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
    try:
        before = os.fstat(fd)
        if before.st_nlink != 1 or metadata.st_nlink != 1:
            raise ValueError("artifact inode is not a single-link regular file")
        if _inode_snapshot_signature(before) != _inode_snapshot_signature(metadata):
            raise ValueError("artifact changed before snapshot capture")
        os.lseek(fd, 0, os.SEEK_SET)
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                raise ValueError("artifact ended before its authorized size")
            snapshot.write(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError("artifact grew during snapshot capture")
        after = os.fstat(fd)
        if _inode_snapshot_signature(after) != _inode_snapshot_signature(before):
            raise ValueError("artifact changed during snapshot capture")
        snapshot.flush()
        snapshot.seek(0)
        return snapshot
    except Exception:
        snapshot.close()
        raise


def _read_open_regular_file_bounded(
    fd: int,
    metadata: os.stat_result,
    *,
    max_bytes: int,
) -> bytes:
    """Return stable exact-size bytes with pre/post inode metadata checks."""
    before = os.fstat(fd)
    if _inode_snapshot_signature(before) != _inode_snapshot_signature(metadata):
        raise ValueError("artifact changed before bounded read")
    if before.st_size > max_bytes:
        raise ValueError("artifact exceeds the bounded read limit")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = before.st_size
    while remaining:
        chunk = os.read(fd, min(remaining, 64 * 1024))
        if not chunk:
            raise ValueError("artifact ended before its authorized size")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        raise ValueError("artifact grew during bounded read")
    after = os.fstat(fd)
    if _inode_snapshot_signature(after) != _inode_snapshot_signature(before):
        raise ValueError("artifact changed during bounded read")
    return b"".join(chunks)


def _stream_snapshot_response(
    fd: int,
    metadata: os.stat_result,
    *,
    media_type: str,
    filename: str,
):
    snapshot = _snapshot_open_regular_file(fd, metadata)

    def _chunks():
        while True:
            chunk = snapshot.read(64 * 1024)
            if not chunk:
                break
            yield chunk

    headers = {
        "Content-Length": str(metadata.st_size),
        "Content-Disposition": (
            "attachment; filename*=UTF-8''" + quote(str(filename), safe="")
        ),
    }
    return StreamingResponse(
        _chunks(),
        media_type=media_type,
        headers=headers,
        background=BackgroundTask(snapshot.close),
    )


def _open_regular_file_under_root(
    root: Path,
    relative_path: str | os.PathLike[str],
) -> tuple[int, os.stat_result]:
    """Open a regular file relative to one already-authorized job root.

    The root and every descendant directory are opened with ``O_NOFOLLOW``;
    the returned descriptor therefore remains bound to the inode selected by
    the authorization decision even if a shared filesystem swaps path names
    while the response is being prepared.
    """
    root_fd, _root_metadata = _open_directory_bound(root)
    try:
        return _open_regular_file_under_root_fd(root_fd, relative_path)
    finally:
        os.close(root_fd)


def secure_file_response_under_root(
    root: Path,
    relative_path: str | os.PathLike[str],
    *,
    media_type: str,
    filename: str,
):
    """Stream a file opened relative to an authorized job root."""
    try:
        fd, inode = _open_regular_file_under_root(root, relative_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "artifact_not_found",
                "message": "Artifact is not readable.",
            },
        ) from exc

    try:
        return _stream_snapshot_response(
            fd,
            inode,
            media_type=media_type,
            filename=filename,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "artifact_changed",
                "message": "Artifact changed during snapshot capture.",
            },
        ) from exc
    finally:
        os.close(fd)


def secure_file_response_under_root_fd(
    root_fd: int,
    relative_path: str | os.PathLike[str],
    *,
    media_type: str,
    filename: str,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    expected_size: int | None = None,
    expected_mtime_ns: int | None = None,
    expected_ctime_ns: int | None = None,
):
    """Stream a descendant opened from an already-bound job-root descriptor."""
    try:
        fd, inode = _open_regular_file_under_root_fd(
            root_fd,
            relative_path,
            expected_device=expected_device,
            expected_inode=expected_inode,
            expected_size=expected_size,
            expected_mtime_ns=expected_mtime_ns,
            expected_ctime_ns=expected_ctime_ns,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "artifact_not_found",
                "message": "Artifact is not readable.",
            },
        ) from exc

    try:
        return _stream_snapshot_response(
            fd,
            inode,
            media_type=media_type,
            filename=filename,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "artifact_changed",
                "message": "Artifact changed during snapshot capture.",
            },
        ) from exc
    finally:
        os.close(fd)


def read_regular_file_bounded(path: Path, *, max_bytes: int) -> bytes:
    """Read one regular inode through an already-open descriptor.

    This is for JSON/bundle assembly, where ``FileResponse`` is not involved.
    Opening with ``O_NOFOLLOW`` and retaining the descriptor until the bounded
    read completes prevents a shared-volume writer from swapping the path
    between validation and use.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError:
        raise
    try:
        inode = os.fstat(fd)
        if not stat.S_ISREG(inode.st_mode) or getattr(inode, "st_nlink", 1) > 1:
            raise ValueError("artifact inode is not eligible for bounded reading")
        return _read_open_regular_file_bounded(fd, inode, max_bytes=max_bytes)
    finally:
        if fd >= 0:
            os.close(fd)


def read_regular_file_bounded_under_root(
    root: Path,
    relative_path: str | os.PathLike[str],
    *,
    max_bytes: int,
) -> bytes:
    """Read a bounded artifact through a descriptor-relative open."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    fd, inode = _open_regular_file_under_root(root, relative_path)
    try:
        return _read_open_regular_file_bounded(fd, inode, max_bytes=max_bytes)
    finally:
        if fd >= 0:
            os.close(fd)


def read_regular_file_bounded_under_root_fd(
    root_fd: int,
    relative_path: str | os.PathLike[str],
    *,
    max_bytes: int,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    expected_size: int | None = None,
    expected_mtime_ns: int | None = None,
    expected_ctime_ns: int | None = None,
) -> bytes:
    """Read a bounded descendant from an already-bound job-root descriptor."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    fd, inode = _open_regular_file_under_root_fd(
        root_fd,
        relative_path,
        expected_device=expected_device,
        expected_inode=expected_inode,
        expected_size=expected_size,
        expected_mtime_ns=expected_mtime_ns,
        expected_ctime_ns=expected_ctime_ns,
    )
    try:
        return _read_open_regular_file_bounded(fd, inode, max_bytes=max_bytes)
    finally:
        if fd >= 0:
            os.close(fd)


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


# Known path-bearing transform/stamp param keys (RC-09 / C03-A03).
_PATH_PARAM_KEYS_SCALAR = frozenset(
    {
        "output_dir",
        "source_pdf",
        "source_path",
        "input_path",
        "output_path",
        "mask_path",
        "overlay_path",
        "watermark_path",
        "font_path",
    }
)
_PATH_PARAM_KEYS_LIST = frozenset(
    {
        "input_files",
        "source_files",
        "paths",
        "files",
    }
)
_PATH_PARAM_KEYS_FILE_SCALAR = frozenset(
    {
        "source_pdf",
        "source_path",
        "mask_path",
        "overlay_path",
        "watermark_path",
        "font_path",
    }
)


def sandbox_params_paths(
    params: dict | None,
    *,
    read_roots: Iterable[str],
    write_roots: Iterable[str],
) -> dict:
    """Re-root every path-bearing transform param (RC-09).

    Top-level ``input_path``/``output_path`` alone are security theater when ops
    open ``params.output_dir`` / ``params.input_files`` / ``params.source_pdf``
    without containment. This walks known keys and absolute-looking strings and
    rewrites them to resolved, rooted paths (fail closed on escape).
    """
    if not params:
        return {}
    out: dict = dict(params)
    read_roots_t = tuple(read_roots)
    write_roots_t = tuple(write_roots)

    def _root_scalar(key: str, value: object, roots: Iterable[str]) -> str:
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_path",
                    "message": f"params.{key} must be a non-empty path string.",
                },
            )
        return str(
            ensure_path_within_roots(
                path_value=value,
                field_name=f"params.{key}",
                allowed_roots=roots,
            )
        )

    for key in list(out.keys()):
        val = out[key]
        if key in _PATH_PARAM_KEYS_SCALAR:
            roots = (
                write_roots_t if key in ("output_dir", "output_path") else read_roots_t
            )
            out[key] = _root_scalar(key, val, roots)
        elif key in _PATH_PARAM_KEYS_LIST:
            if not isinstance(val, list):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "invalid_path",
                        "message": f"params.{key} must be a list of paths.",
                    },
                )
            rooted: list[str] = []
            for i, item in enumerate(val):
                if not isinstance(item, str) or not item.strip():
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "invalid_path",
                            "message": f"params.{key}[{i}] must be a path string.",
                        },
                    )
                rooted.append(
                    str(
                        ensure_path_within_roots(
                            path_value=item,
                            field_name=f"params.{key}[{i}]",
                            allowed_roots=read_roots_t,
                        )
                    )
                )
            out[key] = rooted
        elif (
            isinstance(val, str)
            and val.strip()
            and (val.startswith("/") or _looks_like_windows_absolute(val))
        ):
            # Fail-closed on absolute path-looking unknown keys
            out[key] = str(
                ensure_path_within_roots(
                    path_value=val,
                    field_name=f"params.{key}",
                    allowed_roots=read_roots_t,
                )
            )
    return out


def _sha256_fd(file_fd: int) -> str:
    """Hash the already-open regular inode without reopening its pathname."""
    hasher = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(file_fd, 1024 * 1024, offset)
        if not chunk:
            break
        hasher.update(chunk)
        offset += len(chunk)
    return hasher.hexdigest()


def _copy_fd_to_fd(src_fd: int, dst_fd: int) -> None:
    offset = 0
    while True:
        chunk = os.pread(src_fd, 1024 * 1024, offset)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            written = os.write(dst_fd, view)
            if written <= 0:
                raise OSError("snapshot copy made no progress")
            view = view[written:]
        offset += len(chunk)
    os.fsync(dst_fd)


class AuthorizedOperationIO:
    """Descriptor-stable input snapshot and output parent for transform/stamp."""

    def __init__(
        self,
        *,
        lexical_input: str,
        lexical_output: str,
        input_hash: str,
        input_nlink: int,
        input_size: int,
        document_id: str,
        input_exec_path: str,
        output_exec_path: str,
        snapshot_dir_fd: int,
        snapshot_name: str,
        snapshot_fd: int,
        output_parent_fd: int,
        output_name: str,
        output_root_fd: int,
        owned_fds: list[int],
    ) -> None:
        self.lexical_input = lexical_input
        self.lexical_output = lexical_output
        self.input_hash = input_hash
        self.input_nlink = input_nlink
        self.input_size = input_size
        self.document_id = document_id
        self.input_exec_path = input_exec_path
        self.output_exec_path = output_exec_path
        self._snapshot_dir_fd = snapshot_dir_fd
        self._snapshot_name = snapshot_name
        self._snapshot_fd = snapshot_fd
        self._output_parent_fd = output_parent_fd
        self._output_name = output_name
        self._output_root_fd = output_root_fd
        self._owned_fds = owned_fds
        self._extra_snapshot_names: list[str] = []
        self._extra_snapshots: list[tuple[str, int]] = []

    def assert_snapshot_unchanged(self) -> None:
        """Require snapshot names to still identify the copied inodes."""
        opened = os.fstat(self._snapshot_fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError("input snapshot inode changed during the operation")
        current = os.stat(
            self._snapshot_name,
            dir_fd=self._snapshot_dir_fd,
            follow_symlinks=False,
        )
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("input snapshot name no longer matches the copied inode")
        for name, snap_fd in self._extra_snapshots:
            extra_opened = os.fstat(snap_fd)
            if not stat.S_ISREG(extra_opened.st_mode) or extra_opened.st_nlink != 1:
                raise ValueError(
                    "extra input snapshot inode changed during the operation"
                )
            extra_current = os.stat(
                name,
                dir_fd=self._snapshot_dir_fd,
                follow_symlinks=False,
            )
            if (extra_current.st_dev, extra_current.st_ino) != (
                extra_opened.st_dev,
                extra_opened.st_ino,
            ):
                raise ValueError(
                    "extra input snapshot name no longer matches the copied inode"
                )

    def hash_output(self) -> str:
        """Hash the output inode created under the held parent directory."""
        self.assert_snapshot_unchanged()
        file_fd, metadata = _open_regular_file_under_root_fd(
            self._output_parent_fd,
            self._output_name,
        )
        try:
            if metadata.st_nlink != 1:
                raise ValueError("output inode is not a single-link regular file")
            return _sha256_fd(file_fd)
        finally:
            os.close(file_fd)

    def bind_path_params(
        self,
        params: dict | None,
        *,
        read_roots: Iterable[str],
        write_root: str,
    ) -> dict:
        """Rewrite extra path params onto descriptor-bound snapshots.

        Primary ``input_path``/``output_path`` are already fd-aliased by
        ``bind_authorized_operation_io``. Extra keys (``source_pdf``,
        ``input_files``, ``overlay_path``, ``output_dir``, ...) stay lexical
        until this method copies regular files into the same ``.ops-scratch``
        staging directory and rewrites directories onto held write-root
        descriptors. After it returns, engines cannot observe a later lexical
        pathname swap of those extra paths.
        """
        if not params:
            return {}
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")

        write_root_path = Path(write_root).expanduser().resolve(strict=False)
        opened_root = os.fstat(self._output_root_fd)
        try:
            current_root = os.stat(write_root_path, follow_symlinks=False)
        except OSError as exc:
            raise ValueError("authorized write root is not available") from exc
        if (
            not stat.S_ISDIR(current_root.st_mode)
            or stat.S_ISLNK(current_root.st_mode)
            or (current_root.st_dev, current_root.st_ino)
            != (opened_root.st_dev, opened_root.st_ino)
        ):
            raise ValueError("write_root does not match the bound output root")

        read_root_fds: dict[str, int] = {}
        out: dict = dict(params)

        def _read_root_fd(root_path: Path) -> int:
            key = str(root_path)
            cached = read_root_fds.get(key)
            if cached is not None:
                return cached
            fd, _meta = _open_directory_bound(root_path)
            self._owned_fds.append(fd)
            read_root_fds[key] = fd
            return fd

        def _relative_to_roots(
            path_value: str, roots: Iterable[str]
        ) -> tuple[Path, str]:
            abs_path = Path(path_value).expanduser().resolve(strict=False)
            for raw_root in roots:
                if not raw_root:
                    continue
                root_path = Path(raw_root).expanduser().resolve(strict=False)
                try:
                    relative = abs_path.relative_to(root_path)
                except ValueError:
                    continue
                if not relative.parts:
                    raise ValueError(
                        "path must name an object under the authorized root"
                    )
                return root_path, relative.as_posix()
            raise ValueError("path is not inside an authorized root")

        def _same_path(left: str, right: str) -> bool:
            try:
                return Path(left).expanduser().resolve(strict=False) == Path(
                    right
                ).expanduser().resolve(strict=False)
            except OSError:
                return os.path.normpath(left) == os.path.normpath(right)

        def _snapshot_file(path_value: object) -> str:
            if not isinstance(path_value, str) or not path_value.strip():
                raise ValueError("path param must be a non-empty string")
            root_path, relative = _relative_to_roots(path_value, read_roots)
            root_fd = _read_root_fd(root_path)
            source_fd, source_meta = _open_regular_file_under_root_fd(
                root_fd,
                relative,
            )
            self._owned_fds.append(source_fd)
            if not stat.S_ISREG(source_meta.st_mode) or source_meta.st_nlink != 1:
                raise ValueError("extra input inode is not a single-link regular file")
            # Engines call Path(path).suffix / get_file_extension; keep it.
            suffix = Path(path_value).suffix if Path(path_value).suffix else ".bin"
            snapshot_name = f"snap-{uuid.uuid4().hex}{suffix}"
            snapshot_fd = os.open(
                snapshot_name,
                _SNAPSHOT_CREATE_FLAGS,
                0o600,
                dir_fd=self._snapshot_dir_fd,
            )
            self._owned_fds.append(snapshot_fd)
            self._extra_snapshot_names.append(snapshot_name)
            _copy_fd_to_fd(source_fd, snapshot_fd)
            snapshot_meta = os.fstat(snapshot_fd)
            if not stat.S_ISREG(snapshot_meta.st_mode) or snapshot_meta.st_nlink != 1:
                raise ValueError(
                    "extra input snapshot is not a single-link regular file"
                )
            self._extra_snapshots.append((snapshot_name, snapshot_fd))
            return str(_directory_fd_alias(self._snapshot_dir_fd) / snapshot_name)

        def _bind_output_dir(path_value: object) -> str:
            if not isinstance(path_value, str) or not path_value.strip():
                raise ValueError("output_dir must be a non-empty string")
            abs_path = Path(path_value).expanduser().resolve(strict=False)
            try:
                relative = abs_path.relative_to(write_root_path)
            except ValueError as exc:
                raise ValueError(
                    "output_dir is not inside the authorized write root"
                ) from exc
            if not relative.parts:
                return str(_directory_fd_alias(self._output_root_fd))
            dir_fd = _open_or_create_directory_under_root_fd(
                self._output_root_fd,
                relative.as_posix(),
            )
            self._owned_fds.append(dir_fd)
            opened = os.fstat(dir_fd)
            if not stat.S_ISDIR(opened.st_mode):
                raise ValueError("output_dir is not a directory")
            return str(_directory_fd_alias(dir_fd))

        def _bind_output_file(path_value: object) -> str:
            if not isinstance(path_value, str) or not path_value.strip():
                raise ValueError("output_path must be a non-empty string")
            abs_path = Path(path_value).expanduser().resolve(strict=False)
            try:
                relative = abs_path.relative_to(write_root_path)
            except ValueError as exc:
                raise ValueError(
                    "output_path is not inside the authorized write root"
                ) from exc
            if not relative.parts:
                raise ValueError("output_path must name a file under the write root")
            parent_rel = relative.parent.as_posix() if len(relative.parts) > 1 else "."
            if parent_rel in {"", "."}:
                parent_fd = os.dup(self._output_root_fd)
            else:
                parent_fd = _open_or_create_directory_under_root_fd(
                    self._output_root_fd,
                    parent_rel,
                )
            self._owned_fds.append(parent_fd)
            name = relative.name
            if name in {"", ".", ".."}:
                raise ValueError("output_path must name a file")
            return str(_directory_fd_alias(parent_fd) / name)

        for key, val in list(out.items()):
            if key == "output_dir":
                out[key] = _bind_output_dir(val)
            elif key == "input_path":
                if isinstance(val, str) and _same_path(val, self.lexical_input):
                    out[key] = self.input_exec_path
                else:
                    out[key] = _snapshot_file(val)
            elif key == "output_path":
                if isinstance(val, str) and _same_path(val, self.lexical_output):
                    out[key] = self.output_exec_path
                else:
                    out[key] = _bind_output_file(val)
            elif key in _PATH_PARAM_KEYS_FILE_SCALAR:
                out[key] = _snapshot_file(val)
            elif key in _PATH_PARAM_KEYS_LIST:
                if not isinstance(val, list):
                    raise ValueError(f"params.{key} must be a list of paths")
                out[key] = [_snapshot_file(item) for item in val]
            elif (
                isinstance(val, str)
                and val.strip()
                and (val.startswith("/") or _looks_like_windows_absolute(val))
            ):
                out[key] = _snapshot_file(val)
        return out

    def close(self) -> None:
        names = [self._snapshot_name, *self._extra_snapshot_names]
        seen: set[str] = set()
        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            try:
                os.unlink(name, dir_fd=self._snapshot_dir_fd)
            except OSError:
                pass
        while self._owned_fds:
            fd = self._owned_fds.pop()
            try:
                os.close(fd)
            except OSError:
                pass


def bind_authorized_operation_io(
    *,
    input_path: str,
    output_path: str,
    read_roots: Iterable[str],
    write_root: str,
) -> AuthorizedOperationIO:
    """Bind a stable input snapshot and output parent for one operation.

    Transform/stamp engines receive descriptor aliases for the primary
    input file and output parent, so a later lexical pathname swap cannot
    change those bytes or the directory the primary output is written to.
    Extra path-bearing params (``source_pdf``, ``input_files``,
    ``overlay_path``, ``output_dir``, ...) remain lexical until
    ``AuthorizedOperationIO.bind_path_params`` rewrites them onto the same
    ``.ops-scratch`` staging directory or a held write-root descriptor.
    Custody hashes the primary inodes.
    """
    input_abs = Path(input_path)
    output_abs = Path(output_path)
    owned: list[int] = []
    try:
        input_root_path = None
        input_relative = None
        for raw_root in read_roots:
            if not raw_root:
                continue
            root_path = Path(raw_root).expanduser().resolve(strict=False)
            try:
                input_relative = input_abs.resolve(strict=False).relative_to(root_path)
            except ValueError:
                continue
            input_root_path = root_path
            break
        if input_root_path is None or input_relative is None:
            raise ValueError("input path is not inside an authorized read root")
        output_root_path = Path(write_root).expanduser().resolve(strict=False)
        try:
            output_relative = output_abs.resolve(strict=False).relative_to(
                output_root_path
            )
        except ValueError as exc:
            raise ValueError(
                "output path is not inside the authorized write root"
            ) from exc
        if not output_relative.parts:
            raise ValueError("output path must name a file under the write root")

        input_root_fd, _input_root_meta = _open_directory_bound(input_root_path)
        owned.append(input_root_fd)
        output_root_fd, _output_root_meta = _open_directory_bound(output_root_path)
        owned.append(output_root_fd)

        source_fd, source_meta = _open_regular_file_under_root_fd(
            input_root_fd,
            input_relative.as_posix(),
        )
        owned.append(source_fd)
        if source_meta.st_nlink != 1:
            raise ValueError("input inode is not a single-link regular file")

        staging_fd = _open_or_create_directory_under_root_fd(
            output_root_fd,
            ".ops-scratch",
            mode=0o700,
        )
        owned.append(staging_fd)
        suffix = input_abs.suffix if input_abs.suffix else ".bin"
        snapshot_name = f"snap-{uuid.uuid4().hex}{suffix}"
        snapshot_fd = os.open(
            snapshot_name,
            _SNAPSHOT_CREATE_FLAGS,
            0o600,
            dir_fd=staging_fd,
        )
        owned.append(snapshot_fd)
        _copy_fd_to_fd(source_fd, snapshot_fd)
        snapshot_meta = os.fstat(snapshot_fd)
        if not stat.S_ISREG(snapshot_meta.st_mode) or snapshot_meta.st_nlink != 1:
            raise ValueError("input snapshot is not a single-link regular file")
        input_hash = _sha256_fd(snapshot_fd)

        output_parent_rel = (
            output_relative.parent.as_posix() if len(output_relative.parts) > 1 else "."
        )
        if output_parent_rel in {"", "."}:
            output_parent_fd = os.dup(output_root_fd)
        else:
            output_parent_fd = _open_or_create_directory_under_root_fd(
                output_root_fd,
                output_parent_rel,
            )
        owned.append(output_parent_fd)
        output_name = output_relative.name
        if output_name in {"", ".", ".."}:
            raise ValueError("output path must name a file")

        document_id = hashlib.sha256(
            f"{input_path}|{input_hash}|{snapshot_meta.st_size}|{snapshot_meta.st_nlink}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        return AuthorizedOperationIO(
            lexical_input=input_path,
            lexical_output=output_path,
            input_hash=input_hash,
            input_nlink=snapshot_meta.st_nlink,
            input_size=snapshot_meta.st_size,
            document_id=document_id,
            input_exec_path=str(_directory_fd_alias(staging_fd) / snapshot_name),
            output_exec_path=str(_directory_fd_alias(output_parent_fd) / output_name),
            snapshot_dir_fd=staging_fd,
            snapshot_name=snapshot_name,
            snapshot_fd=snapshot_fd,
            output_parent_fd=output_parent_fd,
            output_name=output_name,
            output_root_fd=output_root_fd,
            owned_fds=owned,
        )
    except Exception:
        while owned:
            fd = owned.pop()
            try:
                os.close(fd)
            except OSError:
                pass
        raise
