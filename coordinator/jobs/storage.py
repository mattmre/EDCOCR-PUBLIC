"""Storage backend abstractions for coordinator task I/O."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import stat
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

_MAX_STORAGE_LIST_OBJECTS = 100_000
_DEFAULT_MAX_STORAGE_TRANSFER_BYTES = 5 * 1024 * 1024 * 1024
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)


def _max_storage_transfer_bytes(max_bytes: int | None = None) -> int:
    """Return the effective cap for one S3 transfer or caller-bounded read.

    Presigned I/O already has a transfer ceiling. Credentialed S3
    ``download_fileobj`` calls need the same protection because the SDK will
    otherwise stream an arbitrarily large or never-ending object to disk. The
    per-call limit can tighten, but never raise, the configured storage cap.
    """
    raw = os.environ.get("S3_STORAGE_MAX_TRANSFER_BYTES", "").strip()
    if not raw:
        raw = os.environ.get("S3_PRESIGNED_MAX_TRANSFER_BYTES", "").strip()
    if not raw:
        configured = _DEFAULT_MAX_STORAGE_TRANSFER_BYTES
    else:
        try:
            configured = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "S3_STORAGE_MAX_TRANSFER_BYTES must be a positive integer"
            ) from exc
        if configured <= 0:
            raise ValueError("S3_STORAGE_MAX_TRANSFER_BYTES must be positive")
    if max_bytes is None:
        return configured
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    return min(configured, max_bytes)


class _BoundedWriter:
    """File-like adapter that stops a storage SDK before disk exhaustion."""

    def __init__(self, destination, max_bytes: int):
        self._destination = destination
        self._max_bytes = max_bytes
        self._written = 0

    def write(self, data):
        next_size = self._written + len(data)
        if next_size > self._max_bytes:
            raise ValueError(
                f"S3 object exceeds S3_STORAGE_MAX_TRANSFER_BYTES ({self._max_bytes})"
            )
        self._written = next_size
        return self._destination.write(data)


class _BoundedReader:
    """File-like adapter that stops a storage SDK on a growing source file."""

    def __init__(self, source, max_bytes: int):
        self._source = source
        self._max_bytes = max_bytes
        self._read = 0

    def read(self, size=-1):
        allowance = self._max_bytes - self._read + 1
        if size < 0:
            bounded_size = allowance
        else:
            bounded_size = min(size, allowance)
        data = self._source.read(bounded_size)
        self._read += len(data)
        if self._read > self._max_bytes:
            raise ValueError(
                f"S3 object exceeds S3_STORAGE_MAX_TRANSFER_BYTES ({self._max_bytes})"
            )
        return data

    def __getattr__(self, name):
        return getattr(self._source, name)


def _lstat_regular_single_link(path: str, *, label: str) -> os.stat_result:
    """Open a local path for metadata checks without following symlinks."""
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} is not accessible") from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise RuntimeError(f"{label} must not be a symlink")
    if not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"{label} must be a regular file")
    if path_stat.st_nlink != 1:
        raise RuntimeError(f"{label} must not have multiple hard links")
    return path_stat


def _assert_regular_single_link(metadata: os.stat_result, *, label: str) -> None:
    """Validate an already-collected stat result without reopening its path."""
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{label} must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular file")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"{label} must not have multiple hard links")


def _reject_symlink_ancestors(path: str, *, stop_at: str | None = None) -> None:
    """Reject symlinked directory components in a local storage path."""
    absolute = os.path.abspath(path)
    # Walk from the filesystem root by default.  Checking only the immediate
    # parent leaves a race/escape window when a higher-level job or tenant
    # directory is replaced with a symlink between validation and I/O.
    stop = os.path.abspath(stop_at) if stop_at else os.path.abspath(os.sep)
    try:
        relative = os.path.relpath(absolute, stop)
    except ValueError as exc:
        raise RuntimeError("storage path is on a different filesystem root") from exc
    cursor = stop
    if relative == os.curdir:
        return
    for component in relative.split(os.sep):
        if component in ("", os.curdir):
            continue
        if component == os.pardir:
            raise RuntimeError("storage path escapes its configured root")
        cursor = os.path.join(cursor, component)
        try:
            if os.path.islink(cursor):
                raise RuntimeError("storage path contains a symlinked directory")
        except OSError as exc:
            raise RuntimeError("storage path could not be inspected") from exc


def _open_local_parent(path: str, *, create: bool = True) -> tuple[int, str, str]:
    """Open a local target's parent without following any path component."""
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute) or os.path.abspath(os.curdir)
    basename = os.path.basename(absolute)
    if not basename or basename in {os.curdir, os.pardir}:
        raise RuntimeError("local storage target must be a file path")
    relative = os.path.relpath(parent, os.path.abspath(os.sep))
    current_fd = os.open(os.path.abspath(os.sep), _DIRECTORY_OPEN_FLAGS)
    try:
        if relative != os.curdir:
            for component in relative.split(os.sep):
                if component in {"", os.curdir}:
                    continue
                if component == os.pardir:
                    raise RuntimeError("local storage target escapes filesystem root")
                try:
                    child_fd = os.open(
                        component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    child_fd = os.open(
                        component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd
                    )
                os.close(current_fd)
                current_fd = child_fd
        return current_fd, basename, absolute
    except Exception:
        os.close(current_fd)
        raise


def _ensure_local_directory(path: str) -> None:
    """Create one local directory tree without following ancestors."""
    parent_fd, basename, _ = _open_local_parent(path)
    try:
        try:
            child_fd = os.open(basename, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            os.mkdir(basename, mode=0o700, dir_fd=parent_fd)
            child_fd = os.open(basename, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
        try:
            metadata = os.fstat(child_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError("local storage directory is not a directory")
        finally:
            os.close(child_fd)
    finally:
        os.close(parent_fd)


def _copy_fd_atomic(
    source_fd: int,
    target: str,
    *,
    prefix: str = ".ocr-copy-",
    max_bytes: int | None = None,
) -> None:
    """Copy a stable regular-file snapshot through a private temporary file."""
    before = os.fstat(source_fd)
    _assert_regular_single_link(before, label="copy source")
    if max_bytes is not None:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if before.st_size > max_bytes:
            raise ValueError("download exceeds requested max_bytes")
    parent_fd = -1
    temporary_name = None
    temporary_fd = -1
    try:
        parent_fd, basename, _ = _open_local_parent(target)
        for _ in range(5):
            candidate = f"{prefix}{uuid.uuid4().hex}"
            try:
                temporary_fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
                temporary_name = candidate
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError("could not allocate a private local copy file")

        with (
            os.fdopen(temporary_fd, "wb") as destination,
            os.fdopen(os.dup(source_fd), "rb") as source,
        ):
            temporary_fd = -1
            copied = 0
            limit = before.st_size if max_bytes is None else max_bytes
            while True:
                chunk = source.read(min(1024 * 1024, limit - copied + 1))
                if not chunk:
                    break
                if copied + len(chunk) > limit:
                    raise ValueError(
                        "download exceeds requested max_bytes or source size"
                    )
                destination.write(chunk)
                copied += len(chunk)
            after = os.fstat(source_fd)
            _assert_regular_single_link(after, label="copy source")
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_nlink",
            )
            if copied != before.st_size or any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                raise RuntimeError("copy source changed during read")
            destination.flush()
            os.fsync(destination.fileno())

        try:
            existing = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _assert_regular_single_link(existing, label="copy target")
        os.replace(
            temporary_name,
            basename,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name is not None and parent_fd >= 0:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Could not remove temporary local copy %s", target)
        if parent_fd >= 0:
            os.close(parent_fd)


def _copy_file_atomic(
    source: str, target: str, *, max_bytes: int | None = None
) -> None:
    """Copy one regular file through a private temporary file and replace."""
    source_fd = -1
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise RuntimeError("copy source must be a regular file")
        if source_stat.st_nlink != 1:
            raise RuntimeError("copy source must not have multiple hard links")
        _copy_fd_atomic(source_fd, target, max_bytes=max_bytes)
    finally:
        if source_fd >= 0:
            os.close(source_fd)


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier."""

    @abstractmethod
    def upload_file(self, local_path: str, key: str) -> str:
        """Upload local file to backend and return remote locator."""

    @abstractmethod
    def download_file(
        self, key: str, local_path: str, *, max_bytes: int | None = None
    ) -> str:
        """Download backend object to local path, optionally with a byte cap."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete backend object, if present."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether backend object exists."""

    @abstractmethod
    def list_objects(self, prefix: str) -> list[str]:
        """List all object keys matching prefix."""

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        """Return temporary download URL, if backend supports it."""
        raise NotImplementedError("Presigned URLs are not supported by this backend")

    def presigned_upload_url(self, key: str, expires: int = 3600) -> str:
        """Return temporary upload URL, if backend supports it."""
        raise NotImplementedError(
            "Presigned upload URLs are not supported by this backend"
        )

    def presigned_delete_url(self, key: str, expires: int = 3600) -> str:
        """Return a temporary delete URL, if backend supports it."""
        raise NotImplementedError(
            "Presigned delete URLs are not supported by this backend"
        )

    def delete_many(self, keys: list[str]) -> int:
        """Delete multiple objects. Returns count of successfully deleted objects.

        Default implementation falls back to sequential delete().
        Subclasses may override for batch optimization.
        """
        deleted = 0
        for key in keys:
            try:
                self.delete(key)
                deleted += 1
            except Exception as exc:
                logger.warning("Failed to delete key %s: %s", key, exc)
        return deleted


@dataclass
class NFSBackend(StorageBackend):
    """NFS-backed storage backend."""

    root: str

    @property
    def backend_name(self) -> str:
        return "nfs"

    def to_absolute_path(self, key: str) -> str:
        root = os.path.realpath(os.path.abspath(os.path.expanduser(self.root)))
        parts = self._key_parts(key)
        resolved = os.path.abspath(os.path.join(root, *parts))
        try:
            contained = os.path.commonpath((root, resolved)) == root
        except ValueError as exc:
            raise ValueError(f"Path traversal detected in key: {key}") from exc
        if not contained:
            raise ValueError(f"Path traversal detected in key: {key}")
        # A path that is lexically inside the root is still unsafe when an
        # attacker can replace an ancestor with a symlink to another job.
        relative_parts = os.path.relpath(resolved, root).split(os.sep)
        cursor = root
        for part in relative_parts:
            if part in ("", os.curdir):
                continue
            cursor = os.path.join(cursor, part)
            if os.path.islink(cursor):
                raise ValueError(f"Symlinked storage path is not allowed: {key}")
        return resolved

    @staticmethod
    def _key_parts(key: str) -> list[str]:
        """Return safe path components for a storage key.

        The descriptor-bound operations below deliberately do not use a
        resolved path after this validation.  A key containing ``..`` or a
        platform separator is rejected before any directory is opened.
        """
        if not isinstance(key, str) or "\x00" in key:
            raise ValueError("storage key must be a non-NUL string")
        normalized = key.replace("\\", "/").lstrip("/")
        parts = [part for part in normalized.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"invalid storage key: {key!r}")
        return parts

    def _open_root(self) -> tuple[int, str]:
        """Open the configured root once, without following its final link."""
        root = os.path.realpath(os.path.abspath(os.path.expanduser(self.root)))
        return os.open(root, _DIRECTORY_OPEN_FLAGS), root

    def _open_parent(self, key: str, *, create: bool) -> tuple[int, str]:
        """Open the parent of *key* using directory descriptors only."""
        parts = self._key_parts(key)
        root_fd, _ = self._open_root()
        current_fd = root_fd
        try:
            for component in parts[:-1]:
                try:
                    child_fd = os.open(
                        component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    child_fd = os.open(
                        component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd
                    )
                os.close(current_fd)
                current_fd = child_fd
            return current_fd, parts[-1]
        except Exception:
            os.close(current_fd)
            raise

    @staticmethod
    def _assert_file_stat(metadata: os.stat_result, *, label: str) -> None:
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} must not be a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{label} must be a regular file")
        if metadata.st_nlink != 1:
            raise RuntimeError(f"{label} must not have multiple hard links")

    def _open_source_fd(self, local_path: str) -> tuple[int, os.stat_result]:
        fd = os.open(local_path, _READ_OPEN_FLAGS)
        try:
            metadata = os.fstat(fd)
            self._assert_file_stat(metadata, label="upload source")
            return fd, metadata
        except Exception:
            os.close(fd)
            raise

    def _copy_to_rooted_key(self, local_path: str, key: str) -> None:
        """Publish a local file under a directory-descriptor-bound key."""
        source_fd, _ = self._open_source_fd(local_path)
        parent_fd = -1
        temporary_name = None
        try:
            parent_fd, basename = self._open_parent(key, create=True)
            for _ in range(5):
                candidate = f".ocr-upload-{uuid.uuid4().hex}"
                try:
                    temporary_fd = os.open(
                        candidate,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_CLOEXEC", 0),
                        0o600,
                        dir_fd=parent_fd,
                    )
                    temporary_name = candidate
                    break
                except FileExistsError:
                    continue
            else:
                raise RuntimeError("could not allocate a private NFS upload file")

            with (
                os.fdopen(temporary_fd, "wb") as destination,
                os.fdopen(source_fd, "rb") as source,
            ):
                source_fd = -1
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())

            try:
                existing = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                self._assert_file_stat(existing, label="NFS upload target")
            os.replace(
                temporary_name,
                basename,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temporary_name = None
            os.fsync(parent_fd)
        finally:
            if source_fd >= 0:
                os.close(source_fd)
            if temporary_name is not None and parent_fd >= 0:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Could not remove partial NFS upload %s", key)
            if parent_fd >= 0:
                os.close(parent_fd)

    def _download_rooted_key(
        self, key: str, local_path: str, *, max_bytes: int | None = None
    ) -> None:
        """Copy a key opened through a directory descriptor to a local path."""
        with self.open_key(key) as source:
            _copy_fd_atomic(
                source.fileno(),
                local_path,
                prefix=".ocr-download-",
                max_bytes=max_bytes,
            )

    @contextmanager
    def open_key(self, key: str):
        """Open and revalidate one NFS key through its current parent FD.

        Callers consume the returned descriptor while the parent directory is
        held open. A path validated earlier is never trusted after a
        concurrent rename or replacement.
        """
        parent_fd = -1
        source_fd = -1
        try:
            parent_fd, basename = self._open_parent(key, create=False)
            try:
                source_fd = os.open(basename, _READ_OPEN_FLAGS, dir_fd=parent_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RuntimeError(f"NFS key {key} is symlinked") from exc
                raise
            metadata = os.fstat(source_fd)
            self._assert_file_stat(metadata, label="NFS reopened source")
            with os.fdopen(source_fd, "rb") as source:
                source_fd = -1
                yield source
        finally:
            if source_fd >= 0:
                os.close(source_fd)
            if parent_fd >= 0:
                os.close(parent_fd)

    def upload_file(self, local_path: str, key: str) -> str:
        target = self.to_absolute_path(key)
        self._copy_to_rooted_key(local_path, key)
        return target

    def download_file(
        self, key: str, local_path: str, *, max_bytes: int | None = None
    ) -> str:
        self._download_rooted_key(key, local_path, max_bytes=max_bytes)
        return local_path

    def delete(self, key: str) -> None:
        parent_fd = -1
        try:
            parent_fd, basename = self._open_parent(key, create=False)
            try:
                target_stat = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            self._assert_file_stat(target_stat, label="NFS delete target")
            os.unlink(basename, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileNotFoundError:
            return
        finally:
            if parent_fd >= 0:
                os.close(parent_fd)

    def exists(self, key: str) -> bool:
        try:
            with self.open_key(key):
                return True
        except FileNotFoundError:
            return False
        except (NotADirectoryError, RuntimeError):
            return False

    def list_objects(self, prefix: str) -> list[str]:
        """List files under a descriptor-bound prefix.

        ``os.walk(prefix_path)`` validates a path and then reopens each
        component by name.  On a shared NFS volume, replacing the job
        directory after validation could make a migration or purge enumerate
        another tenant's tree.  Keep the root and every traversed directory
        open with ``O_NOFOLLOW`` instead.
        """
        prefix_parts = self._key_parts(prefix)
        try:
            root_fd, _root = self._open_root()
        except OSError:
            return []

        current_fd = root_fd
        try:
            for component in prefix_parts:
                try:
                    child_fd = os.open(
                        component,
                        _DIRECTORY_OPEN_FLAGS,
                        dir_fd=current_fd,
                    )
                except (FileNotFoundError, NotADirectoryError):
                    os.close(current_fd)
                    return []
                os.close(current_fd)
                current_fd = child_fd
        except Exception:
            os.close(current_fd)
            raise

        keys: list[str] = []
        pending: list[tuple[int, list[str]]] = [(current_fd, prefix_parts)]
        while pending:
            directory_fd, directory_parts = pending.pop()
            try:
                with os.scandir(directory_fd) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
                for entry in entries:
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISLNK(entry_stat.st_mode):
                        continue
                    entry_parts = directory_parts + [entry.name]
                    if stat.S_ISDIR(entry_stat.st_mode):
                        try:
                            child_fd = os.open(
                                entry.name,
                                _DIRECTORY_OPEN_FLAGS,
                                dir_fd=directory_fd,
                            )
                        except OSError:
                            continue
                        pending.append((child_fd, entry_parts))
                        continue
                    if not stat.S_ISREG(entry_stat.st_mode):
                        continue
                    try:
                        file_fd = os.open(
                            entry.name,
                            _READ_OPEN_FLAGS,
                            dir_fd=directory_fd,
                        )
                        try:
                            bound_stat = os.fstat(file_fd)
                        finally:
                            os.close(file_fd)
                    except OSError:
                        continue
                    if not stat.S_ISREG(bound_stat.st_mode) or bound_stat.st_nlink != 1:
                        continue
                    keys.append("/".join(entry_parts))
                    if len(keys) >= _MAX_STORAGE_LIST_OBJECTS:
                        raise RuntimeError(
                            "NFS object listing exceeds the configured safety bound"
                        )
            finally:
                os.close(directory_fd)
        return keys


@dataclass
class S3Backend(StorageBackend):
    """S3-compatible storage backend."""

    endpoint: str
    bucket: str
    access_key: str
    secret_key: str
    region: str = ""
    client: object | None = None
    auth_mode: str = "static"
    session_token: str = ""

    def __post_init__(self) -> None:
        if self.auth_mode not in {"static", "iam"}:
            raise ValueError("S3_AUTH_MODE must be static or iam")
        if self.auth_mode == "iam" and (self.access_key or self.secret_key or self.session_token):
            raise ValueError("S3_AUTH_MODE=iam must not receive explicit S3 credentials")
        if self.client is not None:
            return
        credentials = {}
        if self.auth_mode == "static":
            if not self.access_key or not self.secret_key:
                raise ValueError("static S3 authentication requires both access and secret keys")
            credentials = {
                "aws_access_key_id": self.access_key,
                "aws_secret_access_key": self.secret_key,
            }
            if self.session_token:
                credentials["aws_session_token"] = self.session_token
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for S3 backend. Install it in coordinator dependencies."
            ) from exc
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint or None,
            # Omit credential arguments for IAM so botocore uses its provider
            # chain. Ambient AWS keys are EnvProvider and win over role, SSO,
            # container, and IMDS providers; do not replace that chain.
            **credentials,
            region_name=self.region or None,
            # Keep SDK and presigned clients on the same modern signing
            # contract, including endpoints with an unspecified region.
            config=Config(signature_version="s3v4"),
        )

    @property
    def backend_name(self) -> str:
        return "s3"

    def object_revision(self, key: str) -> tuple[str, str, int]:
        """Read a remote identity suitable for validating cached object bytes.

        HEAD failures and incomplete responses must never authorize a cache hit.
        VersionId distinguishes writes even when their content has the same ETag.
        """
        assert self.client is not None
        response = self.client.head_object(Bucket=self.bucket, Key=self._validate_key(key))
        etag = response.get("ETag")
        length = response.get("ContentLength")
        version = response.get("VersionId", "")
        if (
            not isinstance(etag, str)
            or not etag
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 0
            or not isinstance(version, str)
        ):
            raise RuntimeError("S3 HEAD omitted usable object revision metadata")
        return version, etag, length

    @staticmethod
    def _validate_key(key: str, *, allow_empty: bool = False) -> str:
        """Require canonical object keys before crossing the S3 boundary."""
        if not isinstance(key, str) or "\x00" in key:
            raise ValueError("S3 object key must be a non-NUL string")
        if not key and allow_empty:
            return key
        parts = key.replace("\\", "/").split("/")
        if (
            not key
            or key.startswith("/")
            or "\\" in key
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("S3 object key is not canonical")
        return key

    @staticmethod
    def _validate_prefix(prefix: str) -> str:
        """Validate an S3 listing prefix without treating its final slash as a key.

        Retention and purge callers intentionally use directory-like prefixes
        such as ``jobs/<id>/``.  Object keys themselves remain strict, while a
        single trailing slash is valid S3 prefix syntax.  Interior empty
        components, traversal components, backslashes, and leading slashes
        are still rejected so a caller cannot widen a listing accidentally.
        """
        if not isinstance(prefix, str) or "\x00" in prefix:
            raise ValueError("S3 object prefix must be a non-NUL string")
        if not prefix:
            return prefix
        if prefix.startswith("/") or "\\" in prefix:
            raise ValueError("S3 object prefix is not canonical")
        parts = prefix.split("/")
        if prefix.endswith("/"):
            parts = parts[:-1]
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("S3 object prefix is not canonical")
        return prefix

    def upload_file(self, local_path: str, key: str) -> str:
        assert self.client is not None
        key = self._validate_key(key)
        max_transfer_bytes = _max_storage_transfer_bytes()
        source_fd = os.open(local_path, _READ_OPEN_FLAGS)
        try:
            metadata = os.fstat(source_fd)
            _assert_regular_single_link(metadata, label="S3 upload source")
            if metadata.st_size > max_transfer_bytes:
                raise ValueError(
                    "S3 object exceeds S3_STORAGE_MAX_TRANSFER_BYTES "
                    f"({max_transfer_bytes})"
                )
            with os.fdopen(source_fd, "rb") as source:
                source_fd = -1
                self.client.upload_fileobj(
                    _BoundedReader(source, max_transfer_bytes),
                    self.bucket,
                    key,
                )
        finally:
            if source_fd >= 0:
                os.close(source_fd)
        return f"s3://{self.bucket}/{key}"

    def download_file(
        self, key: str, local_path: str, *, max_bytes: int | None = None
    ) -> str:
        assert self.client is not None
        key = self._validate_key(key)
        max_transfer_bytes = _max_storage_transfer_bytes(max_bytes)
        parent_fd, basename, _ = _open_local_parent(local_path)
        temporary_name = None
        temporary_fd = -1
        try:
            try:
                existing = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                _assert_regular_single_link(existing, label="S3 download target")
            for _ in range(5):
                candidate = f".ocr-s3-download-{uuid.uuid4().hex}"
                try:
                    temporary_fd = os.open(
                        candidate,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_CLOEXEC", 0),
                        0o600,
                        dir_fd=parent_fd,
                    )
                    temporary_name = candidate
                    break
                except FileExistsError:
                    continue
            else:
                raise RuntimeError("could not allocate a private S3 download file")
            with os.fdopen(temporary_fd, "wb") as destination:
                temporary_fd = -1
                self.client.download_fileobj(
                    self.bucket,
                    key,
                    _BoundedWriter(destination, max_transfer_bytes),
                )
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(
                temporary_name,
                basename,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temporary_name = None
            os.fsync(parent_fd)
        finally:
            if temporary_fd >= 0:
                os.close(temporary_fd)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning(
                        "Could not remove partial S3 download %s", local_path
                    )
            os.close(parent_fd)
        return local_path

    def delete(self, key: str) -> None:
        assert self.client is not None
        key = self._validate_key(key)
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        assert self.client is not None
        key = self._validate_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            # Only treat 404/NoSuchKey as "not found"; re-raise unexpected errors
            error_code = getattr(getattr(exc, "response", None), "get", lambda *a: None)
            if callable(error_code):
                resp = getattr(exc, "response", None) or {}
                code = (
                    resp.get("Error", {}).get("Code", "")
                    if isinstance(resp, dict)
                    else ""
                )
                http_status = (
                    resp.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
                    if isinstance(resp, dict)
                    else 0
                )
                if code in ("404", "NoSuchKey") or http_status == 404:
                    return False
            raise

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        assert self.client is not None
        key = self._validate_key(key)
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def presigned_upload_url(self, key: str, expires: int = 3600) -> str:
        """Generate a presigned PUT URL for uploading to S3."""
        assert self.client is not None
        key = self._validate_key(key)
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def presigned_delete_url(self, key: str, expires: int = 3600) -> str:
        """Generate a presigned DELETE URL for cancellation cleanup."""
        assert self.client is not None
        key = self._validate_key(key)
        return self.client.generate_presigned_url(
            "delete_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def list_objects(self, prefix: str) -> list[str]:
        """List all object keys matching prefix using paginator."""
        assert self.client is not None
        prefix = self._validate_prefix(prefix)
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
                if len(keys) >= _MAX_STORAGE_LIST_OBJECTS:
                    raise RuntimeError(
                        "S3 object listing exceeds the configured safety bound"
                    )
        return keys

    def delete_many(self, keys: list[str]) -> int:
        """Batch delete using S3 DeleteObjects API (max 1000 per request)."""
        assert self.client is not None
        if not keys:
            return 0
        keys = [self._validate_key(key) for key in keys]

        deleted = 0
        batch_size = 1000
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            try:
                response = self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={
                        "Objects": [{"Key": k} for k in batch],
                        "Quiet": True,
                    },
                )
            except Exception as exc:
                logger.error("S3 batch delete failed for %d keys: %s", len(batch), exc)
                # A transport/API failure has no reliable per-key result.  A
                # sequential retry is safe, but any retry failure must surface
                # instead of returning a falsely complete deletion count.
                retry_failures = []
                for key in batch:
                    try:
                        self.delete(key)
                    except Exception as retry_exc:
                        retry_failures.append((key, retry_exc))
                if retry_failures:
                    key, retry_exc = retry_failures[0]
                    raise RuntimeError(
                        f"S3 deletion failed for key {key!r} after batch retry"
                    ) from retry_exc
                deleted += len(batch)
                continue

            errors = response.get("Errors", []) or []
            if errors:
                for error in errors:
                    logger.warning(
                        "S3 batch delete error: Key=%s Code=%s Message=%s",
                        error.get("Key"),
                        error.get("Code"),
                        error.get("Message"),
                    )
                # Some objects may already have been deleted, but the caller
                # cannot safely treat the batch as complete.  Refuse to report
                # success and let retention/purge preserve its DB/audit state.
                raise RuntimeError(
                    f"S3 batch deletion returned {len(errors)} per-key errors"
                )
            deleted += len(batch)
        return deleted


def _serialize_cache_access(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._cache_access():
            return method(self, *args, **kwargs)

    return guarded


@dataclass
class CachedS3Backend(StorageBackend):
    """S3 backend with worker-local LRU cache to avoid redundant downloads."""

    inner: S3Backend
    cache_dir: str = "/tmp/ocr-cache"
    max_size_bytes: int = 1 * 1024**3
    max_object_bytes: int = 512 * 1024**2
    _eviction_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _download_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    _access_times: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    _cache_thread_lock: object = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _cache_lock_state: object = field(
        default_factory=threading.local, init=False, repr=False
    )
    _source_cache: object = field(default=None, init=False, repr=False)

    @contextmanager
    def _cache_access(self):
        """Serialize publication/read/eviction across shared POSIX workers.

        Reentrant in one instance for download -> eviction; separate instances
        and processes synchronize on the same never-evicted lock inode.
        """
        import fcntl

        with self._cache_thread_lock:
            if getattr(self._cache_lock_state, "held", False):
                yield
                return
            parent_fd, basename, _ = _open_local_parent(
                os.path.join(self.cache_dir, ".cache-access.lock")
            )
            descriptor = -1
            try:
                descriptor = os.open(
                    basename,
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
                _assert_regular_single_link(os.fstat(descriptor), label="cache lock")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                self._cache_lock_state.held = True
                try:
                    yield
                finally:
                    self._cache_lock_state.held = False
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                os.close(parent_fd)

    def __post_init__(self) -> None:
        if self.max_size_bytes < 2:
            raise ValueError("S3 cache budget must allow both source and artifact caches")
        # One advertised byte budget covers both immutable source leases and
        # the legacy mutable artifact cache. Each receives half of the cap.
        self._source_budget_bytes = self.max_size_bytes // 2
        self._artifact_budget_bytes = self.max_size_bytes - self._source_budget_bytes
        try:
            _ensure_local_directory(self.cache_dir)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "CachedS3Backend cache_dir must be a real, usable directory"
            ) from exc
        try:
            os.chmod(self.cache_dir, 0o700)
        except OSError as exc:
            raise RuntimeError(
                f"CachedS3Backend cache_dir permissions could not be hardened: {exc}"
            ) from exc
        from .source_cache import ImmutableSourceCache

        self._source_cache = ImmutableSourceCache(
            os.path.join(self.cache_dir, "immutable_sources"),
            max_size_bytes=self._source_budget_bytes,
            max_object_bytes=min(
                _max_storage_transfer_bytes(),
                self._source_budget_bytes,
                self.max_object_bytes,
            ),
        )

    def source_revision(self, key: str) -> str:
        """Return an authority-bound remote revision token for source leases."""
        return self.source_revision_info(key)[0]

    def source_revision_info(self, key: str) -> tuple[str, int]:
        """Return remote revision and byte size from one HEAD request."""
        revision = self.inner.object_revision(key)
        token = json.dumps(
            [self.inner.endpoint, self.inner.bucket, self.inner.region, *revision],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return token, revision[2]

    @contextmanager
    def lease_source(self, identity, *, authorize):
        """Yield a shared source path while its revision and reader lock hold.

        ``authorize`` checks the caller's tenant/job/capability on every lease;
        it must raise on a revoked or expired task.  No per-page file copy is
        made.  The caller must keep this context open through source reading.
        """
        from .source_cache import SourceIdentity

        if not isinstance(identity, SourceIdentity):
            raise TypeError("identity must be a SourceIdentity")
        if self._source_cache is None:
            raise RuntimeError("source cache is unavailable")

        def checked_authority():
            authorize()
            if self.source_revision(identity.object_key) != identity.revision:
                raise RuntimeError("S3 source revision changed")

        def fetch(temporary):
            self.inner.download_file(
                identity.object_key,
                temporary,
                max_bytes=identity.size_bytes,
            )
            if self.source_revision(identity.object_key) != identity.revision:
                raise RuntimeError("S3 source revision changed during download")

        with self._source_cache.lease(identity, fetch, checked_authority) as path:
            yield path

    @staticmethod
    def _harden_cache_file(path: str) -> None:
        """Reject cache symlinks and restrict cached OCR bytes to the owner."""
        descriptor = os.open(path, _READ_OPEN_FLAGS)
        try:
            metadata = os.fstat(descriptor)
            _assert_regular_single_link(metadata, label="CachedS3Backend cache entry")
            # chmod on an already-private inode still changes ctime and can
            # falsely invalidate another process's immutable payload read.
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)

    def _remember_access(self, path: str) -> None:
        # Recency belongs to worker-local metadata, not the immutable bytes'
        # mtime/ctime used by the integrity fence during concurrent reads.
        with self._eviction_lock:
            self._access_times[path] = time.time()

    @property
    def backend_name(self) -> str:
        return self.inner.backend_name

    @_serialize_cache_access
    def upload_file(self, local_path: str, key: str) -> str:
        locator = self.inner.upload_file(local_path, key)
        # An overwrite of a fixed object key invalidates any worker-local
        # bytes previously cached under that key.  Do this only after the
        # remote upload succeeds so a failed upload does not discard a usable
        # prior value.
        self._remove_cache_entries(key)
        return locator

    @_serialize_cache_access
    def download_file(
        self, key: str, local_path: str, *, max_bytes: int | None = None
    ) -> str:
        revision = self.inner.object_revision(key)
        cache_path = self._cache_path(key, revision)
        effective_max_bytes = _max_storage_transfer_bytes(max_bytes)
        if revision[2] > effective_max_bytes:
            raise ValueError("download exceeds requested max_bytes")
        if os.path.isfile(cache_path):
            # Cache hit: enforce the same cap as a fresh SDK download.
            self._harden_cache_file(cache_path)
            if os.lstat(cache_path).st_size != revision[2]:
                raise RuntimeError("Cached S3 object size differs from remote revision")
            _copy_file_atomic(cache_path, local_path, max_bytes=effective_max_bytes)
            self._remember_access(cache_path)
            return local_path
        # Cache miss: serialize same-process misses so callers cannot share a
        # single ``<hash>.tmp`` file and overwrite one another's downloads.
        with self._download_lock:
            if not os.path.isfile(cache_path):
                # The cache lock is process-local; Celery prefork workers may
                # share this directory without sharing the lock. A unique
                # temporary name prevents one process from replacing or
                # cleaning another process's in-flight download.
                cache_tmp = f"{cache_path}.tmp-{uuid.uuid4().hex}"
                _ensure_local_directory(os.path.dirname(cache_path))
                try:
                    if max_bytes is None:
                        self.inner.download_file(key, cache_tmp)
                    else:
                        self.inner.download_file(key, cache_tmp, max_bytes=max_bytes)
                    self._harden_cache_file(cache_tmp)
                    if os.lstat(cache_tmp).st_size > effective_max_bytes:
                        raise ValueError("download exceeds requested max_bytes")
                    if os.lstat(cache_tmp).st_size != revision[2]:
                        raise RuntimeError("S3 object size changed during cached download")
                    if self.inner.object_revision(key) != revision:
                        raise RuntimeError("S3 object revision changed during cached download")
                    # Atomic rename (same filesystem) to prevent partial cache entries
                    os.replace(cache_tmp, cache_path)
                except Exception:
                    # Clean up partial download
                    if os.path.exists(cache_tmp):
                        try:
                            os.remove(cache_tmp)
                        except OSError:
                            pass
                    raise
            self._harden_cache_file(cache_path)
        _copy_file_atomic(cache_path, local_path, max_bytes=effective_max_bytes)
        self._remember_access(cache_path)
        self._maybe_evict()
        return local_path

    @_serialize_cache_access
    def delete(self, key: str) -> None:
        with self._download_lock:
            self._remove_cache_entries(key)
        self.inner.delete(key)

    @_serialize_cache_access
    def delete_many(self, keys: list[str]) -> int:
        with self._download_lock:
            for key in keys:
                self._remove_cache_entries(key)
        return self.inner.delete_many(keys)

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)

    def list_objects(self, prefix: str) -> list[str]:
        return self.inner.list_objects(prefix)

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        return self.inner.presigned_url(key, expires)

    def presigned_upload_url(self, key: str, expires: int = 3600) -> str:
        return self.inner.presigned_upload_url(key, expires)

    def presigned_delete_url(self, key: str, expires: int = 3600) -> str:
        return self.inner.presigned_delete_url(key, expires)

    def _cache_path(self, key: str, revision: tuple[str, str, int] | None = None) -> str:
        namespace = [self.inner.endpoint, self.inner.bucket, self.inner.region, key]
        prefix = hashlib.sha256(json.dumps(namespace, ensure_ascii=False).encode()).hexdigest()
        base = os.path.join(self.cache_dir, prefix)
        if revision is None:
            return base
        suffix = hashlib.sha256(json.dumps(revision, ensure_ascii=False).encode()).hexdigest()
        return f"{base}-{suffix}"

    def _remove_cache_entries(self, key: str) -> None:
        prefix = os.path.basename(self._cache_path(key)) + "-"
        for name in os.listdir(self.cache_dir):
            if name.startswith(prefix) and ".tmp-" not in name:
                self._remove_cache_entry(os.path.join(self.cache_dir, name))

    @staticmethod
    def _remove_cache_entry(cache_path: str) -> None:
        """Remove one cached object, refusing to hide a stale-byte failure."""
        if not os.path.lexists(cache_path):
            return
        if os.path.isdir(cache_path) and not os.path.islink(cache_path):
            raise RuntimeError(
                f"CachedS3Backend cache entry is unexpectedly a directory: {cache_path}"
            )
        try:
            os.remove(cache_path)
        except OSError as exc:
            raise RuntimeError(
                f"CachedS3Backend could not remove cache entry: {cache_path}"
            ) from exc
        if os.path.lexists(cache_path):
            raise RuntimeError(
                f"CachedS3Backend cache entry remains after removal: {cache_path}"
            )

    @_serialize_cache_access
    def invalidate(self, key: str) -> None:
        """Drop any local cache entry for *key* (fail closed on unexpected errors)."""
        with self._download_lock:
            self._remove_cache_entries(key)

    @_serialize_cache_access
    def _maybe_evict(self) -> None:
        if not self._eviction_lock.acquire(blocking=False):
            return  # Another thread is already evicting
        try:
            entries: list[tuple[str, float, int]] = []
            total = 0
            for fname in os.listdir(self.cache_dir):
                if fname == ".cache-access.lock":
                    continue
                fpath = os.path.join(self.cache_dir, fname)
                if (
                    os.path.isfile(fpath)
                    and not fpath.endswith(".tmp")
                    and ".tmp-" not in fname
                ):
                    stat = os.stat(fpath)
                    entries.append(
                        (
                            fpath,
                            self._access_times.get(fpath, stat.st_mtime),
                            stat.st_size,
                        )
                    )
                    total += stat.st_size
            current_paths = {entry[0] for entry in entries}
            self._access_times = {
                path: stamp
                for path, stamp in self._access_times.items()
                if path in current_paths
            }
            if total <= self._artifact_budget_bytes:
                return
            # Evict oldest first until below 80% of max
            target = int(self._artifact_budget_bytes * 0.8)
            entries.sort(key=lambda e: e[1])  # oldest mtime first
            for fpath, _, size in entries:
                if total <= target:
                    break
                try:
                    os.remove(fpath)
                    self._access_times.pop(fpath, None)
                    total -= size
                except OSError:
                    pass
        finally:
            self._eviction_lock.release()


class _PresignedOnlyDenyClient:
    """Fail closed if credential-free OCR code accidentally enters an SDK path."""

    def __getattr__(self, name: str):
        raise RuntimeError(f"S3 SDK operation {name} is forbidden in presigned-only worker")


def create_storage_backend(
    *,
    backend_name: str,
    nfs_root: str,
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
    s3_region: str = "",
    s3_client: object | None = None,
    presigned_only_worker: bool = False,
    s3_auth_mode: str = "static",
    s3_session_token: str = "",
) -> StorageBackend:
    """Create storage backend from normalized config values."""
    normalized = backend_name.strip().lower()
    if normalized == "nfs":
        return NFSBackend(root=nfs_root)
    if normalized == "s3":
        if presigned_only_worker:
            if s3_access_key or s3_secret_key or s3_session_token or s3_client is not None:
                raise ValueError("presigned-only worker must not receive S3 credentials or SDK client")
            s3_client = _PresignedOnlyDenyClient()
        return S3Backend(
            endpoint=s3_endpoint,
            bucket=s3_bucket,
            access_key=s3_access_key,
            secret_key=s3_secret_key,
            region=s3_region,
            client=s3_client,
            auth_mode=s3_auth_mode,
            session_token=s3_session_token,
        )
    raise ValueError(f"Unsupported storage backend '{backend_name}'")
