"""Shared tenant job filesystem purge (R4-RC-16 dual-plane).

Used by:
- FastAPI ``api.tenant_manager.purge_tenant_data`` (SQLite API plane)
- Django ``manage.py purge_tenant`` (coordinator / NFS plane)

Removes structurally job-bound trees under SOURCE / OUTPUT / TEMP / EXPORT.
Opaque shared resume/custody stores are preserved and matching ownership hints
require manual review; their mutable contents never grant deletion authority.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


@dataclass
class _BoundDeleteRoot:
    path: Path
    fd: int
    device: int
    inode: int


@dataclass(frozen=True)
class _DeleteToken:
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class _TreeEntryToken:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    link_count: int


@dataclass
class _TreeSnapshot:
    entries: dict[tuple[str, ...], _TreeEntryToken]

    @property
    def root(self) -> _TreeEntryToken:
        return self.entries[()]


@dataclass
class _SnapshotBudget:
    entries: int = 0
    path_bytes: int = 0
    inodes: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class _PreflightCandidate:
    display_path: Path
    binding: _BoundDeleteRoot
    relative_path: Path
    snapshot: _TreeSnapshot | None


@dataclass(frozen=True)
class _NamespaceDiscovery:
    records: tuple[tuple[object, ...], ...]
    candidates: tuple[Path, ...] = ()


_MAX_PURGE_TREE_ENTRIES = 100_000
_MAX_PURGE_TREE_DEPTH = 64
_MAX_PURGE_TREE_PATH_BYTES = 16 * 1024 * 1024
_MAX_PURGE_NAMESPACE_ENTRIES = 4096
_MAX_SHARED_AUTHORITY_ENTRIES = 4096
_MAX_SHARED_AUTHORITY_BYTES = 64 * 1024 * 1024


def _bind_delete_root(root: Path) -> _BoundDeleteRoot:
    """Bind one configured storage root and reject an open-time name swap."""
    path = Path(os.path.abspath(root))
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise RuntimeError(f"configured purge root is not a real directory: {path}")
    fd = os.open(path, _DIRECTORY_OPEN_FLAGS)
    try:
        opened = os.fstat(fd)
        after = os.stat(path, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino)
        if identity != (opened.st_dev, opened.st_ino) or identity != (
            after.st_dev,
            after.st_ino,
        ):
            raise RuntimeError(f"configured purge root changed while opening: {path}")
        return _BoundDeleteRoot(path, fd, opened.st_dev, opened.st_ino)
    except Exception:
        os.close(fd)
        raise


def _assert_bound_delete_root(binding: _BoundDeleteRoot) -> None:
    """Fail if the configured lexical root no longer names the held inode."""
    opened = os.fstat(binding.fd)
    current = os.stat(binding.path, follow_symlinks=False)
    expected = (binding.device, binding.inode)
    if (
        not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (opened.st_dev, opened.st_ino) != expected
        or (current.st_dev, current.st_ino) != expected
    ):
        raise RuntimeError(f"configured purge root changed: {binding.path}")


def _binding_for_root(
    bindings: Sequence[_BoundDeleteRoot],
    root: Path,
) -> _BoundDeleteRoot | None:
    key = _path_identity(root)
    return next(
        (binding for binding in bindings if _path_identity(binding.path) == key),
        None,
    )


def _read_regular_fd_bounded(fd: int, *, max_bytes: int) -> bytes:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink > 1:
        raise RuntimeError("purge authority file is not a safe regular inode")
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    os.lseek(fd, 0, os.SEEK_SET)
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > max_bytes:
        raise RuntimeError("purge authority file exceeds bounded read limit")
    after = os.fstat(fd)
    if _regular_stat_identity(before) != _regular_stat_identity(after):
        raise RuntimeError("purge authority file changed during bounded read")
    if len(raw) != before.st_size:
        raise RuntimeError("purge authority file changed during bounded read")
    return raw


def _regular_stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _tree_token_from_open_fd(fd: int, metadata: os.stat_result) -> _TreeEntryToken:
    """Snapshot one already-open structurally job-bound regular file."""
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError("purge tree contains an unsafe regular inode")
    after = os.fstat(fd)
    if _regular_stat_identity(metadata) != _regular_stat_identity(after):
        raise RuntimeError("purge tree file changed during preflight")
    return _TreeEntryToken(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
        link_count=metadata.st_nlink,
    )


def _directory_tree_token(metadata: os.stat_result) -> _TreeEntryToken:
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("purge tree directory token is not a directory")
    return _TreeEntryToken(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
        link_count=metadata.st_nlink,
    )


def _same_tree_entry(
    expected: _TreeEntryToken,
    actual: _TreeEntryToken,
    *,
    allow_rename_ctime: bool = False,
) -> bool:
    if (
        expected.device,
        expected.inode,
        stat.S_IFMT(expected.mode),
        expected.size if stat.S_ISREG(expected.mode) else None,
        expected.mtime_ns if stat.S_ISREG(expected.mode) else None,
        expected.link_count if stat.S_ISREG(expected.mode) else None,
    ) != (
        actual.device,
        actual.inode,
        stat.S_IFMT(actual.mode),
        actual.size if stat.S_ISREG(actual.mode) else None,
        actual.mtime_ns if stat.S_ISREG(actual.mode) else None,
        actual.link_count if stat.S_ISREG(actual.mode) else None,
    ):
        return False
    return allow_rename_ctime or expected.ctime_ns == actual.ctime_ns


def _same_tree_snapshot(
    expected: _TreeSnapshot,
    actual: _TreeSnapshot,
    *,
    allow_root_rename_ctime: bool = False,
) -> bool:
    if expected.entries.keys() != actual.entries.keys():
        return False
    return all(
        _same_tree_entry(
            token,
            actual.entries[relative],
            allow_rename_ctime=allow_root_rename_ctime and relative == (),
        )
        for relative, token in expected.entries.items()
    )


def _open_regular_relative_fd(
    root_fd: int,
    relative_path: Path,
    *,
    display_path: Path,
) -> tuple[int, os.stat_result]:
    """Open a regular descendant through no-follow directory descriptors."""
    parts = relative_path.parts
    if not parts or any(part in {"", os.curdir, os.pardir} for part in parts):
        raise RuntimeError(f"invalid root-relative authority path: {display_path}")
    parent_fd = os.dup(root_fd)
    file_fd = -1
    try:
        for component in parts[:-1]:
            child_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        before = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise RuntimeError(f"authority path is not a regular file: {display_path}")
        file_fd = os.open(parts[-1], _FILE_OPEN_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(file_fd)
        _assert_same_inode(before, opened, display_path)
        return file_fd, opened
    except Exception:
        if file_fd >= 0:
            os.close(file_fd)
        raise
    finally:
        os.close(parent_fd)


def _open_directory_relative_fd(root_fd: int, relative_path: Path) -> int:
    parts = relative_path.parts
    if not parts or any(part in {"", os.curdir, os.pardir} for part in parts):
        raise RuntimeError("invalid root-relative directory authority path")
    current_fd = os.dup(root_fd)
    try:
        for component in parts:
            child_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _bounded_directory_names(
    directory_fd: int,
    *,
    max_entries: int,
    label: str,
) -> tuple[str, ...]:
    """Enumerate one attacker-influenced namespace before sorting it."""
    names: list[str] = []
    with os.scandir(directory_fd) as iterator:
        for entry in iterator:
            if len(names) >= max_entries:
                raise RuntimeError(f"{label} exceeds entry bound")
            names.append(entry.name)
    return tuple(sorted(names))


def _namespace_stat_record(metadata: os.stat_result) -> tuple[int, ...]:
    """Identity record that stays stable when a retained directory loses a child."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_nlink if stat.S_ISREG(metadata.st_mode) else 0,
    )


def _snapshot_export_discovery(
    *,
    root_bindings: Sequence[_BoundDeleteRoot],
    bases: Sequence[Path],
    job_id: str,
    tenant_id: str,
) -> _NamespaceDiscovery:
    """Snapshot direct EXPORT namespaces and derive exact structural candidates."""
    records: list[tuple[object, ...]] = []
    candidates: list[Path] = []
    for base in bases:
        binding = _binding_for_root(root_bindings, base)
        if binding is None:
            records.append((_path_identity(base), "root-unbound"))
            continue
        _assert_bound_delete_root(binding)
        export_fd = -1
        try:
            export_fd = _open_directory_relative_fd(binding.fd, Path("EXPORT"))
        except FileNotFoundError:
            records.append((_path_identity(binding.path), "EXPORT", "missing"))
            continue
        try:
            export_metadata = os.fstat(export_fd)
            names = _bounded_directory_names(
                export_fd,
                max_entries=_MAX_PURGE_NAMESPACE_ENTRIES,
                label="EXPORT namespace",
            )
            records.append(
                (
                    _path_identity(binding.path),
                    "EXPORT",
                    "present",
                    *_namespace_stat_record(export_metadata),
                )
            )
            for name in names:
                metadata = os.stat(
                    name,
                    dir_fd=export_fd,
                    follow_symlinks=False,
                )
                records.append(
                    (
                        _path_identity(binding.path),
                        "EXPORT-entry",
                        name,
                        *_namespace_stat_record(metadata),
                    )
                )
                if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(
                    metadata.st_mode
                ):
                    sub = binding.path / "EXPORT" / name
                    candidates.append(sub / job_id)
                    candidates.append(sub / tenant_id / job_id)
            if (
                _bounded_directory_names(
                    export_fd,
                    max_entries=_MAX_PURGE_NAMESPACE_ENTRIES,
                    label="EXPORT namespace",
                )
                != names
            ):
                raise RuntimeError(f"EXPORT namespace changed during scan: {base}")
        finally:
            os.close(export_fd)
    return _NamespaceDiscovery(
        records=tuple(records),
        candidates=tuple(candidates),
    )


def _snapshot_shared_authority(
    *,
    root_bindings: Sequence[_BoundDeleteRoot],
    temp_bases: Sequence[Path],
    output_bases: Sequence[Path],
    job_id: str,
    tenant_id: str,
) -> _NamespaceDiscovery:
    """Snapshot mutable shared TEMP/custody authority for one purge barrier."""
    records: list[tuple[object, ...]] = []
    entry_count = 0
    byte_count = 0

    def reserve_entry(path: Path) -> None:
        nonlocal entry_count
        entry_count += 1
        if entry_count > _MAX_SHARED_AUTHORITY_ENTRIES:
            raise RuntimeError(
                f"shared purge authority scan exceeds entry bound: {path}"
            )

    def reserve_bytes(raw: bytes, path: Path) -> None:
        nonlocal byte_count
        byte_count += len(raw)
        if byte_count > _MAX_SHARED_AUTHORITY_BYTES:
            raise RuntimeError(
                f"shared purge authority scan exceeds byte bound: {path}"
            )

    for base in temp_bases:
        binding = _binding_for_root(root_bindings, base)
        if binding is None:
            records.append((_path_identity(base), "TEMP-root-unbound"))
            continue
        _assert_bound_delete_root(binding)
        names = _bounded_directory_names(
            binding.fd,
            max_entries=_MAX_SHARED_AUTHORITY_ENTRIES,
            label="shared purge authority scan",
        )
        for name in names:
            # These are structural namespaces represented by negative or
            # positive forest candidates, not opaque shared authority.
            if name in {job_id, "jobs", "logs"}:
                continue
            path = binding.path / name
            reserve_entry(path)
            metadata = os.stat(name, dir_fd=binding.fd, follow_symlinks=False)
            records.append(
                (
                    _path_identity(binding.path),
                    "TEMP-entry",
                    name,
                    *_namespace_stat_record(metadata),
                )
            )
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                continue
            child_fd = -1
            manifest_fd = -1
            try:
                child_fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=binding.fd)
                opened = os.fstat(child_fd)
                _assert_same_inode(metadata, opened, path)
                try:
                    manifest_fd = os.open(
                        "resume_manifest.json",
                        _FILE_OPEN_FLAGS,
                        dir_fd=child_fd,
                    )
                except FileNotFoundError:
                    records.append(
                        (
                            _path_identity(binding.path),
                            "TEMP-manifest",
                            name,
                            "missing",
                        )
                    )
                else:
                    raw = _read_regular_fd_bounded(
                        manifest_fd,
                        max_bytes=1024 * 1024,
                    )
                    reserve_bytes(raw, path / "resume_manifest.json")
                    manifest_metadata = os.fstat(manifest_fd)
                    records.append(
                        (
                            _path_identity(binding.path),
                            "TEMP-manifest",
                            name,
                            *_regular_stat_identity(manifest_metadata),
                            hashlib.sha256(raw).hexdigest(),
                        )
                    )
                _assert_name_inode(binding.fd, name, opened, path)
            finally:
                if manifest_fd >= 0:
                    os.close(manifest_fd)
                if child_fd >= 0:
                    os.close(child_fd)
        if (
            _bounded_directory_names(
                binding.fd,
                max_entries=_MAX_SHARED_AUTHORITY_ENTRIES,
                label="shared purge authority scan",
            )
            != names
        ):
            raise RuntimeError(f"shared TEMP namespace changed during scan: {base}")

    for base in output_bases:
        binding = _binding_for_root(root_bindings, base)
        if binding is None:
            records.append((_path_identity(base), "CUSTODY-root-unbound"))
            continue
        _assert_bound_delete_root(binding)
        for store_relative in (Path("EXPORT/CUSTODY"), Path("EXPORT/CUSTODY/ops")):
            store_fd = -1
            store_key = store_relative.as_posix()
            try:
                store_fd = _open_directory_relative_fd(binding.fd, store_relative)
            except FileNotFoundError:
                records.append((_path_identity(binding.path), store_key, "missing"))
                continue
            try:
                names = _bounded_directory_names(
                    store_fd,
                    max_entries=_MAX_SHARED_AUTHORITY_ENTRIES,
                    label="shared purge authority scan",
                )
                records.append((_path_identity(binding.path), store_key, "present"))
                for name in names:
                    if store_relative == Path("EXPORT/CUSTODY") and name in {
                        job_id,
                        tenant_id,
                        "ops",
                    }:
                        continue
                    path = binding.path / store_relative / name
                    reserve_entry(path)
                    metadata = os.stat(
                        name,
                        dir_fd=store_fd,
                        follow_symlinks=False,
                    )
                    records.append(
                        (
                            _path_identity(binding.path),
                            store_key,
                            "entry",
                            name,
                            *_namespace_stat_record(metadata),
                        )
                    )
                    if not (
                        name.endswith(".custody.jsonl")
                        and stat.S_ISREG(metadata.st_mode)
                        and not stat.S_ISLNK(metadata.st_mode)
                    ):
                        continue
                    file_fd = os.open(name, _FILE_OPEN_FLAGS, dir_fd=store_fd)
                    try:
                        opened = os.fstat(file_fd)
                        _assert_same_inode(metadata, opened, path)
                        raw = _read_regular_fd_bounded(
                            file_fd,
                            max_bytes=256 * 1024,
                        )
                        reserve_bytes(raw, path)
                        records.append(
                            (
                                _path_identity(binding.path),
                                store_key,
                                "bytes",
                                name,
                                *_regular_stat_identity(opened),
                                hashlib.sha256(raw).hexdigest(),
                            )
                        )
                        _assert_name_inode(store_fd, name, opened, path)
                    finally:
                        os.close(file_fd)
                if (
                    _bounded_directory_names(
                        store_fd,
                        max_entries=_MAX_SHARED_AUTHORITY_ENTRIES,
                        label="shared purge authority scan",
                    )
                    != names
                ):
                    raise RuntimeError(
                        f"shared custody namespace changed during scan: "
                        f"{base / store_relative}"
                    )
            finally:
                os.close(store_fd)
    return _NamespaceDiscovery(records=tuple(records))


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def path_based_doc_id(path: str) -> str:
    """Legacy path-only id (older code paths / tests)."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def compute_source_fingerprint_fast(path: str) -> dict:
    """Match ocr_gpu_async.compute_source_fingerprint_fast (stat only)."""
    stat_info = os.stat(path)
    mtime_ns = int(
        getattr(stat_info, "st_mtime_ns", int(stat_info.st_mtime * 1_000_000_000))
    )
    size_bytes = int(stat_info.st_size)
    stat_key = f"{path}:{mtime_ns}:{size_bytes}"
    stat_hash = hashlib.sha256(stat_key.encode("utf-8")).hexdigest()
    return {
        "content_sha256": stat_hash,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_resume_doc_id(path: str, source_fingerprint: dict) -> str:
    """Match ocr_gpu_async.build_resume_doc_id (production TEMP dir name)."""
    content_sha256 = str(source_fingerprint.get("content_sha256", ""))
    size_bytes = int(source_fingerprint.get("size_bytes", 0) or 0)
    mtime_ns = int(source_fingerprint.get("mtime_ns", 0) or 0)
    payload = f"{path}|{content_sha256}|{size_bytes}|{mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _as_paths(raw: Optional[Sequence[str | Path]]) -> list[Path]:
    out: list[Path] = []
    for item in raw or ():
        if item:
            out.append(Path(item))
    return out


def _path_is_under_roots(path: Path, roots: Sequence[Path]) -> bool:
    """Return true only when a direct deletion candidate is root-contained."""
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root.resolve(strict=False))
            return True
        except (OSError, ValueError):
            continue
    return False


def _path_identity(path: str | Path) -> str:
    """Return a lexical path identity without following the current leaf."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _validate_identity_component(value: str, *, field_name: str) -> str:
    """Validate a persisted identifier before using it as a path component."""
    component = str(value or "")
    if (
        not component
        or len(component.encode("utf-8")) > 255
        or Path(component).name != component
        or component in {os.curdir, os.pardir}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", component)
    ):
        raise RuntimeError(f"invalid {field_name} for tenant purge")
    return component


def _is_exact_job_bound_path(
    path: str | Path,
    roots: Sequence[Path],
    job_id: str,
) -> bool:
    """Accept only canonical per-job layouts for untrusted direct paths."""
    candidate = _path_identity(path)
    return any(
        candidate
        in {
            _path_identity(root / job_id),
            _path_identity(root / "jobs" / job_id),
        }
        for root in roots
    )


def safe_nfs_job_path(
    raw_path: str | Path | None,
    nfs_root: str | Path | None,
    *,
    job_id: str | None = None,
) -> Path | None:
    """Return a safe, root-contained NFS job path.

    Database paths are untrusted deletion inputs.  When ``job_id`` is given,
    the path must resolve to the job's canonical directory (either
    ``NFS_ROOT/jobs/{job_id}`` or the legacy ``NFS_ROOT/{job_id}`` layout).
    This prevents a poisoned row for one job from deleting another job's
    artifacts while still allowing the historical lab layout.
    """
    if not raw_path or not nfs_root:
        return None
    try:
        root = Path(nfs_root).resolve(strict=False)
        candidate = Path(raw_path).resolve(strict=False)
        candidate.relative_to(root)
    except (OSError, ValueError, TypeError):
        return None
    if candidate == root:
        return None

    if job_id is None:
        return candidate

    job_id_s = str(job_id)
    for expected in (root / "jobs" / job_id_s, root / job_id_s):
        try:
            expected_resolved = expected.resolve(strict=False)
        except OSError:
            continue
        # Do not accept a canonical directory that is itself a symlink out of
        # the configured root.  A resolved target outside root is not a safe
        # interpretation of the deployment's NFS namespace.
        if expected_resolved == expected and candidate == expected_resolved:
            return candidate
    return None


def remove_path_no_follow(path: str | Path) -> bool:
    """Remove one candidate while keeping the parent/root inode bound.

    A path-only ``lstat`` followed by ``rmtree`` still permits a shared-volume
    rename/symlink swap between validation and deletion.  This implementation
    opens every directory relative to a no-follow descriptor and rechecks the
    selected inode before removing each name.  It deliberately refuses
    ambiguous filesystem races instead of guessing which tree an operator
    intended to purge.
    """
    candidate = Path(path)
    absolute = os.path.abspath(candidate)
    parent = os.path.dirname(absolute) or os.path.abspath(os.sep)
    basename = os.path.basename(absolute)
    if not basename or basename in {os.curdir, os.pardir}:
        raise RuntimeError(f"invalid purge path: {candidate}")

    parent_fd = os.open(os.path.abspath(os.sep), _DIRECTORY_OPEN_FLAGS)
    try:
        relative_parent = os.path.relpath(parent, os.path.abspath(os.sep))
        if relative_parent != os.curdir:
            for component in relative_parent.split(os.sep):
                if component in {"", os.curdir}:
                    continue
                if component == os.pardir:
                    raise RuntimeError(
                        f"purge path escapes filesystem root: {candidate}"
                    )
                child_fd = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=parent_fd,
                )
                os.close(parent_fd)
                parent_fd = child_fd

        return _remove_name_no_follow(parent_fd, basename, candidate)
    finally:
        os.close(parent_fd)


def _remove_name_no_follow(parent_fd: int, basename: str, candidate: Path) -> bool:
    """Remove one name below an already-bound parent descriptor."""
    try:
        metadata = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"refusing to remove symlinked path: {candidate}")

    if stat.S_ISDIR(metadata.st_mode):
        target_fd = os.open(
            basename,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=parent_fd,
        )
        try:
            target_metadata = os.fstat(target_fd)
            _assert_same_inode(metadata, target_metadata, candidate)
            _remove_directory_contents(target_fd, candidate)
            _assert_name_inode(parent_fd, basename, metadata, candidate)
            os.rmdir(basename, dir_fd=parent_fd)
        finally:
            os.close(target_fd)
    elif stat.S_ISREG(metadata.st_mode):
        target_fd = os.open(basename, _FILE_OPEN_FLAGS, dir_fd=parent_fd)
        try:
            target_metadata = os.fstat(target_fd)
            _assert_same_inode(metadata, target_metadata, candidate)
            _assert_name_inode(parent_fd, basename, metadata, candidate)
            os.unlink(basename, dir_fd=parent_fd)
        finally:
            os.close(target_fd)
    else:
        raise RuntimeError(f"unsupported purge filesystem object: {candidate}")
    try:
        os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    return False


def _remove_relative_path_no_follow(
    root_fd: int,
    relative_path: Path,
    *,
    display_path: Path,
    expected_token: _DeleteToken | None = None,
) -> bool:
    """Remove a candidate by walking only below a held configured-root fd."""
    parts = relative_path.parts
    if not parts or any(part in {"", os.curdir, os.pardir} for part in parts):
        raise RuntimeError(f"invalid root-relative purge path: {display_path}")
    parent_fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            try:
                child_fd = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                return False
            os.close(parent_fd)
            parent_fd = child_fd
        if expected_token is not None:
            try:
                current = os.stat(
                    parts[-1],
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"purge target disappeared after preflight: {display_path}"
                ) from exc
            if (
                current.st_dev,
                current.st_ino,
                stat.S_IFMT(current.st_mode),
            ) != (
                expected_token.device,
                expected_token.inode,
                stat.S_IFMT(expected_token.mode),
            ):
                raise RuntimeError(
                    f"purge target changed after preflight: {display_path}"
                )
        return _remove_name_no_follow(parent_fd, parts[-1], display_path)
    finally:
        os.close(parent_fd)


def _token_relative_path_no_follow(
    root_fd: int,
    relative_path: Path,
    *,
    display_path: Path,
) -> _DeleteToken | None:
    """Bind a deletion candidate below a held root without following names."""
    parts = relative_path.parts
    if not parts or any(part in {"", os.curdir, os.pardir} for part in parts):
        raise RuntimeError(f"invalid root-relative purge path: {display_path}")
    parent_fd = os.dup(root_fd)
    target_fd = -1
    try:
        for component in parts[:-1]:
            try:
                child_fd = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                return None
            os.close(parent_fd)
            parent_fd = child_fd
        try:
            metadata = os.stat(
                parts[-1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"refusing symlinked purge candidate: {display_path}")
        flags = (
            _DIRECTORY_OPEN_FLAGS
            if stat.S_ISDIR(metadata.st_mode)
            else _FILE_OPEN_FLAGS
        )
        if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise RuntimeError(f"unsupported purge candidate: {display_path}")
        target_fd = os.open(parts[-1], flags, dir_fd=parent_fd)
        opened = os.fstat(target_fd)
        _assert_same_inode(metadata, opened, display_path)
        return _DeleteToken(opened.st_dev, opened.st_ino, opened.st_mode)
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        os.close(parent_fd)


def _snapshot_open_tree(
    target_fd: int,
    metadata: os.stat_result,
    *,
    display_path: Path,
    expected_root_device: int,
    budget: _SnapshotBudget | None = None,
) -> _TreeSnapshot:
    """Recursively preflight one opened job-bound tree with hard bounds."""
    if metadata.st_dev != expected_root_device:
        raise RuntimeError(
            f"purge candidate crosses configured-root device at {display_path}"
        )
    entries: dict[tuple[str, ...], _TreeEntryToken] = {}
    budget = budget or _SnapshotBudget()
    root_device = expected_root_device

    def _reserve_entry(
        current: os.stat_result,
        child_path: Path,
        relative: tuple[str, ...],
    ) -> None:
        budget.entries += 1
        budget.path_bytes += len(os.fsencode("/".join(relative)))
        if budget.entries > _MAX_PURGE_TREE_ENTRIES:
            raise RuntimeError(f"purge tree exceeds entry bound at {child_path}")
        if budget.path_bytes > _MAX_PURGE_TREE_PATH_BYTES:
            raise RuntimeError(f"purge tree exceeds path-byte bound at {child_path}")
        if current.st_dev != root_device:
            raise RuntimeError(f"purge tree crosses filesystem devices at {child_path}")
        inode_key = (current.st_dev, current.st_ino)
        if inode_key in budget.inodes:
            raise RuntimeError(
                f"purge tree contains a duplicate inode alias: {child_path}"
            )
        budget.inodes.add(inode_key)

    def _visit(
        current_fd: int,
        current: os.stat_result,
        relative: tuple[str, ...],
        current_path: Path,
        depth: int,
    ) -> None:
        _reserve_entry(current, current_path, relative)
        if stat.S_ISREG(current.st_mode):
            entries[relative] = _tree_token_from_open_fd(current_fd, current)
            return
        if not stat.S_ISDIR(current.st_mode):
            raise RuntimeError(f"unsupported purge filesystem object: {current_path}")
        if depth > _MAX_PURGE_TREE_DEPTH:
            raise RuntimeError(f"purge tree exceeds depth bound at {current_path}")
        entries[relative] = _directory_tree_token(current)
        names: list[str] = []
        with os.scandir(current_fd) as iterator:
            for entry in iterator:
                # Bound the iterator before sorting or retaining the complete
                # contents of an attacker-controlled directory.
                if budget.entries + len(names) + 1 > _MAX_PURGE_TREE_ENTRIES:
                    raise RuntimeError(
                        f"purge tree exceeds entry bound at {current_path}"
                    )
                names.append(entry.name)
        expected_names = set(names)
        for name in sorted(names):
            child_path = current_path / name
            child_relative = relative + (name,)
            child_fd = -1
            try:
                child_metadata = os.stat(
                    name,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(child_metadata.st_mode):
                    raise RuntimeError(f"refusing symlinked purge entry: {child_path}")
                if stat.S_ISDIR(child_metadata.st_mode):
                    flags = _DIRECTORY_OPEN_FLAGS
                elif stat.S_ISREG(child_metadata.st_mode):
                    flags = _FILE_OPEN_FLAGS
                else:
                    raise RuntimeError(
                        f"unsupported purge filesystem object: {child_path}"
                    )
                child_fd = os.open(name, flags, dir_fd=current_fd)
                opened = os.fstat(child_fd)
                _assert_same_inode(child_metadata, opened, child_path)
                _visit(
                    child_fd,
                    opened,
                    child_relative,
                    child_path,
                    depth + 1,
                )
                _assert_name_inode(current_fd, name, opened, child_path)
            finally:
                if child_fd >= 0:
                    os.close(child_fd)
        with os.scandir(current_fd) as iterator:
            final_names = {entry.name for entry in iterator}
        if final_names != expected_names:
            raise RuntimeError(f"purge tree changed during preflight: {current_path}")

    _visit(target_fd, metadata, (), display_path, 0)
    return _TreeSnapshot(entries)


def _snapshot_relative_tree_no_follow(
    root_fd: int,
    relative_path: Path,
    *,
    display_path: Path,
    expected_root_device: int,
    budget: _SnapshotBudget | None = None,
) -> _TreeSnapshot | None:
    """Snapshot one candidate below a held configured-root descriptor."""
    parts = relative_path.parts
    if not parts or any(part in {"", os.curdir, os.pardir} for part in parts):
        raise RuntimeError(f"invalid root-relative purge path: {display_path}")
    parent_fd = os.dup(root_fd)
    target_fd = -1
    try:
        for component in parts[:-1]:
            try:
                child_fd = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                return None
            child_metadata = os.fstat(child_fd)
            if child_metadata.st_dev != expected_root_device:
                os.close(child_fd)
                raise RuntimeError(
                    f"purge candidate crosses configured-root device at {display_path}"
                )
            os.close(parent_fd)
            parent_fd = child_fd
        try:
            before = os.stat(
                parts[-1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(before.st_mode):
            raise RuntimeError(f"refusing symlinked purge candidate: {display_path}")
        if stat.S_ISDIR(before.st_mode):
            flags = _DIRECTORY_OPEN_FLAGS
        elif stat.S_ISREG(before.st_mode):
            flags = _FILE_OPEN_FLAGS
        else:
            raise RuntimeError(f"unsupported purge candidate: {display_path}")
        target_fd = os.open(parts[-1], flags, dir_fd=parent_fd)
        opened = os.fstat(target_fd)
        _assert_same_inode(before, opened, display_path)
        if opened.st_dev != expected_root_device:
            raise RuntimeError(
                f"purge candidate crosses configured-root device at {display_path}"
            )
        snapshot = _snapshot_open_tree(
            target_fd,
            opened,
            display_path=display_path,
            expected_root_device=expected_root_device,
            budget=budget,
        )
        _assert_name_inode(parent_fd, parts[-1], opened, display_path)
        return snapshot
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        os.close(parent_fd)


def _remove_preflighted_directory_contents(
    directory_fd: int,
    directory: Path,
    snapshot: _TreeSnapshot,
    commit_guard: Callable[[], None],
    relative: tuple[str, ...] = (),
) -> None:
    """Delete only descendants present in the complete preflight snapshot."""
    expected_names = {
        child_relative[len(relative)]
        for child_relative in snapshot.entries
        if len(child_relative) == len(relative) + 1
        and child_relative[: len(relative)] == relative
    }
    with os.scandir(directory_fd) as iterator:
        actual_names = {entry.name for entry in iterator}
    if actual_names != expected_names:
        raise RuntimeError(f"purge tree changed after preflight: {directory}")

    for name in sorted(expected_names):
        child_path = directory / name
        child_relative = relative + (name,)
        expected = snapshot.entries[child_relative]
        child_fd = -1
        try:
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode):
                raise RuntimeError(
                    f"refusing symlinked purge entry after preflight: {child_path}"
                )
            flags = (
                _DIRECTORY_OPEN_FLAGS
                if stat.S_ISDIR(before.st_mode)
                else _FILE_OPEN_FLAGS
            )
            if not (stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode)):
                raise RuntimeError(f"unsupported purge filesystem object: {child_path}")
            child_fd = os.open(name, flags, dir_fd=directory_fd)
            opened = os.fstat(child_fd)
            _assert_same_inode(before, opened, child_path)
            actual = (
                _directory_tree_token(opened)
                if stat.S_ISDIR(opened.st_mode)
                else _tree_token_from_open_fd(child_fd, opened)
            )
            if not _same_tree_entry(expected, actual):
                raise RuntimeError(f"purge entry changed after preflight: {child_path}")
            if stat.S_ISDIR(opened.st_mode):
                _remove_preflighted_directory_contents(
                    child_fd,
                    child_path,
                    snapshot,
                    commit_guard,
                    child_relative,
                )
                commit_guard()
                _assert_name_inode(directory_fd, name, opened, child_path)
                os.rmdir(name, dir_fd=directory_fd)
                removed = os.fstat(child_fd)
                if (
                    (removed.st_dev, removed.st_ino) != (opened.st_dev, opened.st_ino)
                    or not stat.S_ISDIR(removed.st_mode)
                    or removed.st_nlink != 0
                ):
                    raise RuntimeError(
                        f"purge directory remained linked after removal: {child_path}"
                    )
            else:
                commit_guard()
                current_opened = os.fstat(child_fd)
                current_token = _tree_token_from_open_fd(
                    child_fd,
                    current_opened,
                )
                if not _same_tree_entry(expected, current_token):
                    raise RuntimeError(f"purge entry changed at commit: {child_path}")
                _assert_name_inode(directory_fd, name, opened, child_path)
                os.unlink(name, dir_fd=directory_fd)
                removed = os.fstat(child_fd)
                if (
                    (removed.st_dev, removed.st_ino) != (opened.st_dev, opened.st_ino)
                    or not stat.S_ISREG(removed.st_mode)
                    or removed.st_nlink != 0
                ):
                    raise RuntimeError(
                        f"purge file remained linked after removal: {child_path}"
                    )
        finally:
            if child_fd >= 0:
                os.close(child_fd)


def _remove_relative_tree_preflighted(
    root_fd: int,
    relative_path: Path,
    *,
    display_path: Path,
    snapshot: _TreeSnapshot,
    commit_guard: Callable[[], None],
) -> bool:
    """Remove one exact preflighted tree below a held configured root."""
    parts = relative_path.parts
    parent_fd = os.dup(root_fd)
    target_fd = -1
    try:
        for component in parts[:-1]:
            child_fd = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = child_fd
        before = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        flags = (
            _DIRECTORY_OPEN_FLAGS if stat.S_ISDIR(before.st_mode) else _FILE_OPEN_FLAGS
        )
        if stat.S_ISLNK(before.st_mode) or not (
            stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode)
        ):
            raise RuntimeError(
                f"unsafe purge candidate after preflight: {display_path}"
            )
        target_fd = os.open(parts[-1], flags, dir_fd=parent_fd)
        opened = os.fstat(target_fd)
        _assert_same_inode(before, opened, display_path)
        actual_root = (
            _directory_tree_token(opened)
            if stat.S_ISDIR(opened.st_mode)
            else _tree_token_from_open_fd(target_fd, opened)
        )
        if not _same_tree_entry(snapshot.root, actual_root):
            raise RuntimeError(f"purge target changed after preflight: {display_path}")
        if stat.S_ISDIR(opened.st_mode):
            _remove_preflighted_directory_contents(
                target_fd,
                display_path,
                snapshot,
                commit_guard,
            )
            commit_guard()
            _assert_name_inode(parent_fd, parts[-1], opened, display_path)
            os.rmdir(parts[-1], dir_fd=parent_fd)
            removed = os.fstat(target_fd)
            if (
                (removed.st_dev, removed.st_ino) != (opened.st_dev, opened.st_ino)
                or not stat.S_ISDIR(removed.st_mode)
                or removed.st_nlink != 0
            ):
                raise RuntimeError(
                    f"purge directory remained linked after removal: {display_path}"
                )
        else:
            commit_guard()
            current_opened = os.fstat(target_fd)
            current_token = _tree_token_from_open_fd(target_fd, current_opened)
            if not _same_tree_entry(snapshot.root, current_token):
                raise RuntimeError(f"purge target changed at commit: {display_path}")
            _assert_name_inode(parent_fd, parts[-1], opened, display_path)
            os.unlink(parts[-1], dir_fd=parent_fd)
            removed = os.fstat(target_fd)
            if (
                (removed.st_dev, removed.st_ino) != (opened.st_dev, opened.st_ino)
                or not stat.S_ISREG(removed.st_mode)
                or removed.st_nlink != 0
            ):
                raise RuntimeError(
                    f"purge file remained linked after removal: {display_path}"
                )
        return True
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        os.close(parent_fd)


def _assert_same_inode(
    expected: os.stat_result,
    actual: os.stat_result,
    path: Path,
) -> None:
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise RuntimeError(f"purge target changed during inspection: {path}")


def _assert_name_inode(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    path: Path,
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"purge target disappeared during inspection: {path}"
        ) from exc
    _assert_same_inode(expected, current, path)


def _remove_directory_contents(directory_fd: int, directory: Path) -> None:
    with os.scandir(directory_fd) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        name = entry.name
        child_path = directory / name
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                f"purge entry could not be inspected: {child_path}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(
                f"refusing to remove symlinked purge entry: {child_path}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
            try:
                _assert_same_inode(metadata, os.fstat(child_fd), child_path)
                _remove_directory_contents(child_fd, child_path)
                _assert_name_inode(directory_fd, name, metadata, child_path)
                os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"unsupported purge filesystem object: {child_path}")
        child_fd = os.open(name, _FILE_OPEN_FLAGS, dir_fd=directory_fd)
        try:
            _assert_same_inode(metadata, os.fstat(child_fd), child_path)
            _assert_name_inode(directory_fd, name, metadata, child_path)
            os.unlink(name, dir_fd=directory_fd)
        finally:
            os.close(child_fd)


def default_temp_bases(
    *,
    output_bases: Sequence[Path] | None = None,
) -> list[Path]:
    """Guess TEMP roots when env/config unset (compose / Docker defaults)."""
    bases: list[Path] = []
    for env_key in ("OCR_TEMP_DIR", "TEMP_FOLDER"):
        raw = os.environ.get(env_key, "")
        if raw:
            bases.append(Path(raw))
    if bases:
        return bases
    guesses = [
        Path("/app/ocr_temp"),
        Path(os.environ.get("OCR_OUTPUT_DIR", "/app/ocr_output")).parent / "ocr_temp",
    ]
    for ob in output_bases or ():
        guesses.append(Path(ob).parent / "ocr_temp")
        guesses.append(Path(ob) / "ocr_temp")
    return guesses


def resolve_fs_bases(
    *,
    source_bases: Optional[Sequence[str | Path]] = None,
    output_bases: Optional[Sequence[str | Path]] = None,
    temp_bases: Optional[Sequence[str | Path]] = None,
    config_mod=None,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Resolve SOURCE/OUTPUT/TEMP roots from args, config module, or env."""
    sources = _as_paths(source_bases)
    outputs = _as_paths(output_bases)
    temps = _as_paths(temp_bases)

    if config_mod is not None:
        for key, env_fallback, bucket in (
            ("SOURCE_FOLDER", "OCR_SOURCE_DIR", sources),
            ("OUTPUT_FOLDER", "OCR_OUTPUT_DIR", outputs),
            ("TEMP_FOLDER", "OCR_TEMP_DIR", temps),
        ):
            raw = getattr(config_mod, key, None) or os.environ.get(env_fallback, "")
            if raw and Path(raw) not in bucket:
                bucket.append(Path(raw))
    else:
        for env_key, bucket in (
            ("OCR_SOURCE_DIR", sources),
            ("SOURCE_FOLDER", sources),
            ("OCR_OUTPUT_DIR", outputs),
            ("OUTPUT_FOLDER", outputs),
            ("OCR_TEMP_DIR", temps),
            ("TEMP_FOLDER", temps),
        ):
            raw = os.environ.get(env_key, "")
            if raw and Path(raw) not in bucket:
                bucket.append(Path(raw))

    if not temps:
        temps = default_temp_bases(output_bases=outputs)
    return sources, outputs, temps


def purge_fs_job_artifacts(
    *,
    job_id: str,
    tenant_id: str,
    source_file: Optional[str] = None,
    source_hash: Optional[str] = None,
    result_path: Optional[str] = None,
    nfs_job_path: Optional[str] = None,
    source_bases: Optional[Sequence[str | Path]] = None,
    output_bases: Optional[Sequence[str | Path]] = None,
    temp_bases: Optional[Sequence[str | Path]] = None,
    config_mod=None,
    extra_dirs: Optional[Iterable[str | Path]] = None,
    strict: bool = False,
) -> int:
    """Remove job FS trees under SOURCE/OUTPUT/TEMP/EXPORT (R4-RC-16).

    Returns the number of paths successfully removed.  With ``strict=True``,
    every deletion candidate must be root-contained and verifiably removed;
    any untrusted or failed candidate raises ``RuntimeError`` before the
    caller is allowed to delete database rows.
    """
    source_bases_p, output_bases_p, temp_bases_p = resolve_fs_bases(
        source_bases=source_bases,
        output_bases=output_bases,
        temp_bases=temp_bases,
        config_mod=config_mod,
    )

    root_bindings: list[_BoundDeleteRoot] = []
    absent_roots: list[Path] = []
    try:
        seen_roots: set[str] = set()
        for root in source_bases_p + output_bases_p + temp_bases_p:
            root_key = _path_identity(root)
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)
            try:
                root_bindings.append(_bind_delete_root(root))
            except FileNotFoundError:
                # A root absent at authorization time cannot gain deletion
                # authority merely by appearing later in this purge.
                absent_roots.append(Path(os.path.abspath(root)))
                continue
            except (OSError, RuntimeError) as exc:
                if strict:
                    raise RuntimeError(
                        f"unable to bind configured purge root {root}: {exc}"
                    ) from exc
                logger.warning(
                    "Skipping unsafe configured purge root %s: %s", root, exc
                )

        return _purge_fs_job_artifacts_resolved(
            job_id=job_id,
            tenant_id=tenant_id,
            source_file=source_file,
            source_hash=source_hash,
            result_path=result_path,
            nfs_job_path=nfs_job_path,
            source_bases_p=source_bases_p,
            output_bases_p=output_bases_p,
            temp_bases_p=temp_bases_p,
            root_bindings=root_bindings,
            absent_roots=absent_roots,
            extra_dirs=extra_dirs,
            strict=strict,
        )
    finally:
        for binding in root_bindings:
            try:
                os.close(binding.fd)
            except OSError:
                pass


def _purge_fs_job_artifacts_resolved(
    *,
    job_id: str,
    tenant_id: str,
    source_file: Optional[str],
    source_hash: Optional[str],
    result_path: Optional[str],
    nfs_job_path: Optional[str],
    source_bases_p: Sequence[Path],
    output_bases_p: Sequence[Path],
    temp_bases_p: Sequence[Path],
    root_bindings: Sequence[_BoundDeleteRoot],
    absent_roots: Sequence[Path],
    extra_dirs: Optional[Iterable[str | Path]],
    strict: bool,
) -> int:
    """Perform one purge using configured roots bound by the public wrapper."""

    job_id_s = _validate_identity_component(str(job_id), field_name="job_id")
    tenant_id_s = _validate_identity_component(
        str(tenant_id),
        field_name="tenant_id",
    )
    candidates: list[Path] = []
    persisted_source_hash = str(source_hash or "").strip().lower()
    if persisted_source_hash and not re.fullmatch(
        r"[0-9a-f]{64}", persisted_source_hash
    ):
        if strict:
            raise RuntimeError("persisted source_hash is malformed")
        persisted_source_hash = ""
    manual_review_reasons: set[str] = set()
    shared_authority_entries = 0
    shared_authority_bytes = 0

    def _account_shared_authority_entry(path: Path) -> None:
        nonlocal shared_authority_entries
        shared_authority_entries += 1
        if shared_authority_entries > _MAX_SHARED_AUTHORITY_ENTRIES:
            raise RuntimeError(
                f"shared purge authority scan exceeds entry bound: {path}"
            )

    def _account_shared_authority_bytes(size: int, path: Path) -> None:
        nonlocal shared_authority_bytes
        shared_authority_bytes += size
        if shared_authority_bytes > _MAX_SHARED_AUTHORITY_BYTES:
            raise RuntimeError(
                f"shared purge authority scan exceeds byte bound: {path}"
            )

    def _ownership_hint_state(
        jobs: set[str],
        tenants: set[str],
    ) -> str:
        """Classify mutable shared-store owner hints without trusting them."""
        job_match = job_id_s in jobs
        tenant_match = tenant_id_s in tenants
        job_foreign = any(value != job_id_s for value in jobs)
        tenant_foreign = any(value != tenant_id_s for value in tenants)
        has_target = job_match or tenant_match
        has_foreign = job_foreign or tenant_foreign
        if has_target:
            return "mixed_hint" if has_foreign else "target_hint"
        if has_foreign:
            return "foreign"
        return "ambiguous"

    delete_roots = source_bases_p + output_bases_p + temp_bases_p

    # These values originate in database rows / optional caller arguments.
    # Never treat them as an unrestricted deletion path: only accept a
    # candidate already contained by a configured storage root. Derived
    # ``base / job_id`` candidates below are root-contained by construction.
    if nfs_job_path:
        nfs_candidate = Path(nfs_job_path)
        if _is_exact_job_bound_path(nfs_candidate, delete_roots, job_id_s):
            candidates.append(nfs_candidate)
        else:
            if strict:
                raise RuntimeError(
                    f"uncontained nfs_job_path during tenant purge: {nfs_candidate}"
                )
            logger.warning(
                "Skipping uncontained nfs_job_path during tenant purge: %s",
                nfs_candidate,
            )

    all_bases = source_bases_p + output_bases_p + temp_bases_p
    export_discovery = _snapshot_export_discovery(
        root_bindings=root_bindings,
        bases=all_bases,
        job_id=job_id_s,
        tenant_id=tenant_id_s,
    )
    for base in all_bases:
        candidates.append(base / job_id_s)
        candidates.append(base / "jobs" / job_id_s)
        # Per-job API logs are a separate auxiliary output tree rather than a
        # child of OUTPUT/{job_id}.  Keep them within the same root-contained
        # deletion policy so a hard tenant purge cannot leave OCR/error text
        # behind in logs/jobs/{job_id}.jsonl.
        candidates.append(base / "logs" / "jobs" / f"{job_id_s}.jsonl")
    candidates.extend(export_discovery.candidates)

    shared_authority_snapshot = (
        _snapshot_shared_authority(
            root_bindings=root_bindings,
            temp_bases=temp_bases_p,
            output_bases=output_bases_p,
            job_id=job_id_s,
            tenant_id=tenant_id_s,
        )
        if strict
        else None
    )

    # Only a path that is structurally bound below SOURCE/{job_id} may be
    # inspected to derive resume/custody identities.  SOURCE/<basename> is a
    # shared ingest name: by purge time it may contain a different tenant's
    # live upload and must never become authority for the old job.
    path_candidates: list[str] = []
    shared_source_paths: list[str] = []
    if source_file:
        sf = str(source_file)
        sf_path = Path(sf)
        for sb in source_bases_p:
            path_candidates.append(str(sb / job_id_s / Path(sf).name))
            if not sf_path.is_absolute() and ".." not in sf_path.parts:
                path_candidates.append(str(sb / job_id_s / sf_path))
            # An absolute persisted source may participate only when its
            # lexical path is already inside this job's dedicated SOURCE dir.
            try:
                absolute_source = Path(_path_identity(sf_path))
                absolute_source.relative_to(Path(_path_identity(sb / job_id_s)))
            except ValueError:
                pass
            else:
                path_candidates.append(str(absolute_source))
            source_name = sf_path.name
            if source_name:
                shared_source_paths.append(str(sb / source_name))

    # ``source_file`` can be a stored path rather than a path derived from a
    # configured base. Filter every direct/path-composed candidate before it
    # is considered for unlinking or fingerprint probing.
    path_candidates = list(dict.fromkeys(path_candidates))

    source_path_set = {_path_identity(path) for path in path_candidates}
    shared_source_path_set = {_path_identity(path) for path in shared_source_paths}
    live_resume_ids: set[str] = set()
    legacy_resume_ids: set[str] = set()
    # Path-hash and resume ids are derived only from job-bound copies. They
    # are hints for manual review, never deletion authority in a shared TEMP
    # namespace.
    for pth in path_candidates:
        legacy_resume_ids.add(path_based_doc_id(pth))
    for pth in path_candidates:
        for binding in root_bindings:
            try:
                relative_source = Path(_path_identity(pth)).relative_to(binding.path)
            except ValueError:
                continue
            try:
                file_fd, source_stat = _open_regular_relative_fd(
                    binding.fd,
                    relative_source,
                    display_path=Path(pth),
                )
            except FileNotFoundError:
                continue
            except (OSError, RuntimeError):
                if strict:
                    raise RuntimeError(f"unable to bind job source authority: {pth}")
                continue
            try:
                mtime_ns = int(
                    getattr(
                        source_stat,
                        "st_mtime_ns",
                        int(source_stat.st_mtime * 1_000_000_000),
                    )
                )
                size_bytes = int(source_stat.st_size)
                stat_hash = hashlib.sha256(
                    f"{pth}:{mtime_ns}:{size_bytes}".encode("utf-8")
                ).hexdigest()
                resume_id = build_resume_doc_id(
                    pth,
                    {
                        "content_sha256": stat_hash,
                        "size_bytes": size_bytes,
                        "mtime_ns": mtime_ns,
                    },
                )
                live_resume_ids.add(resume_id)
            finally:
                os.close(file_fd)

    for tb in temp_bases_p:
        binding = _binding_for_root(root_bindings, tb)
        if binding is None:
            continue
        try:
            names: list[str] = []
            with os.scandir(binding.fd) as iterator:
                for entry in iterator:
                    _account_shared_authority_entry(binding.path / entry.name)
                    names.append(entry.name)
        except (OSError, RuntimeError) as exc:
            if strict:
                raise RuntimeError(f"unable to scan TEMP root {tb}: {exc}") from exc
            logger.warning("Skipping bounded TEMP authority scan %s: %s", tb, exc)
            continue
        for name in sorted(names):
            # TEMP/{job_id} is a structural job namespace and is preflighted
            # below. Every other directory is a shared opaque resume store;
            # its mutable manifest may signal manual review but can never
            # authorize automatic deletion.
            if name == job_id_s:
                continue
            child_fd = -1
            manifest_fd = -1
            child_path = binding.path / name
            try:
                entry_stat = os.stat(
                    name,
                    dir_fd=binding.fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(entry_stat.st_mode) or stat.S_ISLNK(
                    entry_stat.st_mode
                ):
                    if name in live_resume_ids or name in legacy_resume_ids:
                        manual_review_reasons.add(
                            f"target-correlated nonregular shared TEMP entry "
                            f"{child_path} requires manual review"
                        )
                    continue
                child_fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=binding.fd)
                child_stat = os.fstat(child_fd)
                _assert_same_inode(entry_stat, child_stat, child_path)
                try:
                    manifest_fd = os.open(
                        "resume_manifest.json",
                        _FILE_OPEN_FLAGS,
                        dir_fd=child_fd,
                    )
                except FileNotFoundError:
                    if name in live_resume_ids or name in legacy_resume_ids:
                        manual_review_reasons.add(
                            f"shared opaque TEMP directory {child_path} matches "
                            "a job-bound resume hint"
                        )
                    continue
                manifest_raw = _read_regular_fd_bounded(
                    manifest_fd,
                    max_bytes=1024 * 1024,
                )
                _account_shared_authority_bytes(
                    len(manifest_raw),
                    child_path / "resume_manifest.json",
                )
                payload = json.loads(manifest_raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("resume manifest must be an object")
                payload_job = str(payload.get("job_id") or "")
                payload_tenant = str(payload.get("tenant_id") or "")
                payload_fingerprint = payload.get("source_fingerprint") or {}
                if not isinstance(payload_fingerprint, dict):
                    raise ValueError("resume source_fingerprint must be an object")
                owner_state = _ownership_hint_state(
                    {payload_job} if payload_job else set(),
                    {payload_tenant} if payload_tenant else set(),
                )
                if owner_state == "foreign":
                    continue
                mpath = str(payload.get("source_path") or "")
                mpath_identity = _path_identity(mpath) if mpath else ""
                manifest_hashes = {
                    str(value).lower()
                    for value in (
                        payload_fingerprint.get("content_sha256"),
                        payload.get("source_hash"),
                    )
                    if value
                }
                if any(
                    not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in manifest_hashes
                ):
                    raise ValueError("resume manifest contains a malformed hash")
                has_target_hint = bool(
                    owner_state in {"target_hint", "mixed_hint"}
                    or mpath_identity in source_path_set
                    or mpath_identity in shared_source_path_set
                    or name in live_resume_ids
                    or name in legacy_resume_ids
                    or persisted_source_hash
                    and persisted_source_hash in manifest_hashes
                )
                if has_target_hint:
                    manual_review_reasons.add(
                        f"shared opaque TEMP manifest "
                        f"{child_path / 'resume_manifest.json'} requires manual review"
                    )
            except (
                OSError,
                RuntimeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                if strict:
                    raise RuntimeError(
                        f"unable to inspect TEMP purge authority {child_path}: {exc}"
                    ) from exc
            finally:
                if manifest_fd >= 0:
                    os.close(manifest_fd)
                if child_fd >= 0:
                    os.close(child_fd)
    for ob in output_bases_p:
        binding = _binding_for_root(root_bindings, ob)
        if binding is None:
            continue
        for store_relative in (
            Path("EXPORT/CUSTODY"),
            Path("EXPORT/CUSTODY/ops"),
        ):
            store_fd = -1
            try:
                store_fd = _open_directory_relative_fd(binding.fd, store_relative)
                custody_names: list[str] = []
                with os.scandir(store_fd) as iterator:
                    for entry in iterator:
                        entry_path = binding.path / store_relative / entry.name
                        _account_shared_authority_entry(entry_path)
                        custody_names.append(entry.name)
            except FileNotFoundError:
                if store_fd >= 0:
                    os.close(store_fd)
                continue
            except (OSError, RuntimeError) as exc:
                if store_fd >= 0:
                    os.close(store_fd)
                if strict:
                    raise RuntimeError(
                        f"unable to scan custody authority under {ob}: {exc}"
                    ) from exc
                continue
            try:
                for name in sorted(custody_names):
                    if not name.endswith(".custody.jsonl"):
                        continue
                    jsonl_fd = -1
                    jsonl_path = binding.path / store_relative / name
                    try:
                        entry_stat = os.stat(
                            name,
                            dir_fd=store_fd,
                            follow_symlinks=False,
                        )
                        if not stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(
                            entry_stat.st_mode
                        ):
                            stem = name[: -len(".custody.jsonl")]
                            if stem in {
                                job_id_s,
                                tenant_id_s,
                                *live_resume_ids,
                                *legacy_resume_ids,
                            }:
                                manual_review_reasons.add(
                                    "target-correlated nonregular shared custody "
                                    f"entry {jsonl_path} requires manual review"
                                )
                            continue
                        jsonl_fd = os.open(name, _FILE_OPEN_FLAGS, dir_fd=store_fd)
                        opened = os.fstat(jsonl_fd)
                        _assert_same_inode(entry_stat, opened, jsonl_path)
                        raw = _read_regular_fd_bounded(
                            jsonl_fd,
                            max_bytes=256 * 1024,
                        )
                        _account_shared_authority_bytes(len(raw), jsonl_path)
                        raw = raw.decode("utf-8")

                        event_jobs: set[str] = set()
                        event_tenants: set[str] = set()
                        source_identities: set[str] = set()
                        declared_hashes: set[str] = set()
                        for line in raw.splitlines():
                            if not line.strip():
                                continue
                            ev = json.loads(line)
                            if not isinstance(ev, dict):
                                raise ValueError("custody event must be an object")
                            data = ev.get("data") or {}
                            if not isinstance(data, dict):
                                data = {}
                            event_job = str(
                                data.get("job_id") or ev.get("job_id") or ""
                            )
                            event_tenant = str(
                                data.get("tenant_id") or ev.get("tenant_id") or ""
                            )
                            if event_job:
                                event_jobs.add(event_job)
                            if event_tenant:
                                event_tenants.add(event_tenant)
                            src = str(
                                data.get("source_path")
                                or data.get("input_path")
                                or data.get("source_file")
                                or ev.get("source_path")
                                or ""
                            )
                            if src:
                                source_identities.add(_path_identity(src))
                            fingerprint = data.get("source_fingerprint") or {}
                            if not isinstance(fingerprint, dict):
                                fingerprint = {}
                            for value in (
                                data.get("file_hash"),
                                data.get("source_hash"),
                                data.get("input_hash"),
                                fingerprint.get("content_sha256"),
                            ):
                                if not value:
                                    continue
                                declared = str(value).lower()
                                if not re.fullmatch(r"[0-9a-f]{64}", declared):
                                    raise ValueError(
                                        "custody authority contains a malformed hash"
                                    )
                                declared_hashes.add(declared)

                        owner_state = _ownership_hint_state(
                            event_jobs,
                            event_tenants,
                        )
                        if owner_state == "foreign":
                            # Explicit foreign ownership always wins over a
                            # coincidental filename, source path, or digest.
                            continue
                        has_legacy_hint = bool(
                            owner_state in {"target_hint", "mixed_hint"}
                            or source_identities & source_path_set
                            or source_identities & shared_source_path_set
                            or (
                                persisted_source_hash
                                and persisted_source_hash in declared_hashes
                            )
                            or name[: -len(".custody.jsonl")] in live_resume_ids
                            or name[: -len(".custody.jsonl")] in legacy_resume_ids
                        )
                        if has_legacy_hint:
                            manual_review_reasons.add(
                                f"global shared custody file {jsonl_path} "
                                "requires manual review"
                            )
                    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
                        if strict:
                            raise RuntimeError(
                                f"unable to inspect custody authority {jsonl_path}: {exc}"
                            ) from exc
                    finally:
                        if jsonl_fd >= 0:
                            os.close(jsonl_fd)
            finally:
                os.close(store_fd)

    if result_path:
        rp = Path(result_path)
        result_is_job_bound = _is_exact_job_bound_path(
            rp,
            output_bases_p,
            job_id_s,
        )
        if strict and not result_is_job_bound:
            raise RuntimeError(f"unbound result_path during tenant purge: {rp}")
        if result_is_job_bound:
            candidates.append(rp)
        else:
            if strict:
                raise RuntimeError(f"uncontained result_path during tenant purge: {rp}")
            logger.warning(
                "Skipping uncontained result_path during tenant purge: %s", rp
            )

    for extra in extra_dirs or ():
        if extra:
            extra_path = Path(extra)
            if _is_exact_job_bound_path(extra_path, delete_roots, job_id_s):
                candidates.append(extra_path)
            else:
                if strict:
                    raise RuntimeError(
                        f"uncontained extra purge path during tenant purge: {extra_path}"
                    )
                logger.warning("Skipping uncontained extra purge path: %s", extra_path)

    if strict and shared_authority_snapshot is not None:
        after_authority_scan = _snapshot_shared_authority(
            root_bindings=root_bindings,
            temp_bases=temp_bases_p,
            output_bases=output_bases_p,
            job_id=job_id_s,
            tenant_id=tenant_id_s,
        )
        if after_authority_scan.records != shared_authority_snapshot.records:
            raise RuntimeError(
                "shared purge authority changed during ownership inspection"
            )

    if manual_review_reasons:
        details = "; ".join(sorted(manual_review_reasons))
        message = f"manual review required before shared artifact purge: {details}"
        if strict:
            raise RuntimeError(message)
        logger.warning(message)

    preflight: list[_PreflightCandidate] = []
    preflight_budget = _SnapshotBudget()
    unique_candidates: dict[str, Path] = {}
    for candidate in candidates:
        unique_candidates.setdefault(_path_identity(candidate), candidate)

    for key, candidate in sorted(
        unique_candidates.items(),
        key=lambda item: (len(Path(item[0]).parts), item[0]),
    ):
        candidate_path = Path(key)
        matches: list[tuple[int, _BoundDeleteRoot, Path]] = []
        for binding in root_bindings:
            try:
                relative = candidate_path.relative_to(binding.path)
            except ValueError:
                continue
            if relative == Path(os.curdir):
                continue
            matches.append((len(binding.path.parts), binding, relative))
        if not matches:
            under_configured_root = False
            for root in delete_roots:
                try:
                    candidate_path.relative_to(Path(_path_identity(root)))
                    under_configured_root = True
                    break
                except ValueError:
                    continue
            if strict and not under_configured_root:
                raise RuntimeError(
                    f"uncontained deletion candidate during tenant purge: {candidate}"
                )
            # A configured root absent when authorization began cannot gain
            # authority if it or a candidate appears during the purge.
            continue
        _depth, binding, relative = max(matches, key=lambda item: item[0])
        if any(
            existing.binding is binding
            and (
                relative == existing.relative_path
                or relative.is_relative_to(existing.relative_path)
            )
            for existing in preflight
        ):
            continue
        try:
            _assert_bound_delete_root(binding)
            snapshot = _snapshot_relative_tree_no_follow(
                binding.fd,
                relative,
                display_path=candidate,
                expected_root_device=binding.device,
                budget=preflight_budget,
            )
            preflight.append(
                _PreflightCandidate(
                    display_path=candidate,
                    binding=binding,
                    relative_path=relative,
                    snapshot=snapshot,
                )
            )
        except OSError as exc:
            if strict:
                raise RuntimeError(
                    f"failed to preflight purge path {candidate}: {exc}"
                ) from exc
            logger.debug("Purge preflight failed %s: %s", candidate, exc)
        except RuntimeError:
            if strict:
                raise
            logger.warning("Skipping unsafe purge candidate: %s", candidate)

    # Strict mode performs a complete recursive pass before the first unlink.
    for binding in root_bindings:
        _assert_bound_delete_root(binding)
    for root in absent_roots:
        if os.path.lexists(root):
            raise RuntimeError(
                f"configured purge root appeared after authorization: {root}"
            )

    # Rewalk every structural forest as one bounded invocation immediately
    # before commit, then delete only the recorded bottom-up forest.
    verification_budget = _SnapshotBudget()
    verified: list[_PreflightCandidate] = []
    for item in preflight:
        try:
            _assert_bound_delete_root(item.binding)
            current = _snapshot_relative_tree_no_follow(
                item.binding.fd,
                item.relative_path,
                display_path=item.display_path,
                expected_root_device=item.binding.device,
                budget=verification_budget,
            )
            if (item.snapshot is None) != (current is None) or (
                item.snapshot is not None
                and current is not None
                and not _same_tree_snapshot(item.snapshot, current)
            ):
                raise RuntimeError(
                    f"purge tree changed after preflight: {item.display_path}"
                )
            verified.append(item)
        except (OSError, RuntimeError) as exc:
            if strict:
                raise RuntimeError(
                    f"failed to verify purge path {item.display_path}: {exc}"
                ) from exc
            logger.warning("Skipping changed purge candidate: %s", item.display_path)

    for binding in root_bindings:
        _assert_bound_delete_root(binding)
    for root in absent_roots:
        if os.path.lexists(root):
            raise RuntimeError(
                f"configured purge root appeared after authorization: {root}"
            )

    commit_guard_ran = False

    def _validate_commit_plan() -> None:
        nonlocal commit_guard_ran
        if commit_guard_ran:
            return
        for bound_root in root_bindings:
            _assert_bound_delete_root(bound_root)
        for absent_root in absent_roots:
            if os.path.lexists(absent_root):
                raise RuntimeError(
                    f"configured purge root appeared after authorization: {absent_root}"
                )
        current_export = _snapshot_export_discovery(
            root_bindings=root_bindings,
            bases=all_bases,
            job_id=job_id_s,
            tenant_id=tenant_id_s,
        )
        if current_export.records != export_discovery.records or tuple(
            map(_path_identity, current_export.candidates)
        ) != tuple(map(_path_identity, export_discovery.candidates)):
            raise RuntimeError("EXPORT purge namespace changed at commit")
        if shared_authority_snapshot is not None:
            current_authority = _snapshot_shared_authority(
                root_bindings=root_bindings,
                temp_bases=temp_bases_p,
                output_bases=output_bases_p,
                job_id=job_id_s,
                tenant_id=tenant_id_s,
            )
            if current_authority.records != shared_authority_snapshot.records:
                raise RuntimeError("shared purge authority changed at commit")
        commit_budget = _SnapshotBudget()
        for planned in preflight:
            current = _snapshot_relative_tree_no_follow(
                planned.binding.fd,
                planned.relative_path,
                display_path=planned.display_path,
                expected_root_device=planned.binding.device,
                budget=commit_budget,
            )
            if (planned.snapshot is None) != (current is None) or (
                planned.snapshot is not None
                and current is not None
                and not _same_tree_snapshot(planned.snapshot, current)
            ):
                raise RuntimeError(
                    f"purge tree changed at commit: {planned.display_path}"
                )
        commit_guard_ran = True

    commit_guard = _validate_commit_plan if strict else (lambda: None)
    removable = [item for item in verified if item.snapshot is not None]
    if strict and not removable:
        _validate_commit_plan()

    removed = 0
    for item in removable:
        assert item.snapshot is not None
        try:
            _assert_bound_delete_root(item.binding)
            if _remove_relative_tree_preflighted(
                item.binding.fd,
                item.relative_path,
                display_path=item.display_path,
                snapshot=item.snapshot,
                commit_guard=commit_guard,
            ):
                removed += 1
            _assert_bound_delete_root(item.binding)
        except OSError as exc:
            if strict:
                raise RuntimeError(
                    f"failed to remove purge path {item.display_path}: {exc}"
                ) from exc
            logger.debug("Purge removal failed %s: %s", item.display_path, exc)
        except RuntimeError:
            if strict:
                raise
            logger.warning("Skipping unsafe purge candidate: %s", item.display_path)

    # A successful strict purge reports only after the structural namespace
    # is still absent through the same bound roots. Concurrent writers are a
    # retryable saga failure, never a false successful purge.
    if strict:
        audit_budget = _SnapshotBudget()
        for item in preflight:
            current = _snapshot_relative_tree_no_follow(
                item.binding.fd,
                item.relative_path,
                display_path=item.display_path,
                expected_root_device=item.binding.device,
                budget=audit_budget,
            )
            if current is not None:
                raise RuntimeError(
                    f"purge target appeared or survived commit: {item.display_path}"
                )
        for binding in root_bindings:
            _assert_bound_delete_root(binding)
        for root in absent_roots:
            if os.path.lexists(root):
                raise RuntimeError(
                    f"configured purge root appeared after authorization: {root}"
                )
        final_export = _snapshot_export_discovery(
            root_bindings=root_bindings,
            bases=all_bases,
            job_id=job_id_s,
            tenant_id=tenant_id_s,
        )
        if final_export.records != export_discovery.records or tuple(
            map(_path_identity, final_export.candidates)
        ) != tuple(map(_path_identity, export_discovery.candidates)):
            raise RuntimeError("EXPORT purge namespace changed during commit")
        if shared_authority_snapshot is not None:
            final_authority = _snapshot_shared_authority(
                root_bindings=root_bindings,
                temp_bases=temp_bases_p,
                output_bases=output_bases_p,
                job_id=job_id_s,
                tenant_id=tenant_id_s,
            )
            if final_authority.records != shared_authority_snapshot.records:
                raise RuntimeError("shared purge authority changed during commit")
    return removed
