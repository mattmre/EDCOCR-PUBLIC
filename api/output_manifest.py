"""Output manifest builder for OCR job results.

Scans a job's output directory structure and builds a manifest of all
produced artifacts with their types, paths, sizes, and schema versions.
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from schemas import SCHEMA_VERSION
except ImportError:
    SCHEMA_VERSION = "1.0"

# Map EXPORT/ subdirectory names to output types
OUTPUT_DIR_MAP: dict[str, str] = {
    "PDF": "searchable_pdf",
    "TEXT": "ocr_text",
    "STRUCTURE": "structure",
    "ENTITIES": "entities",
    "NER": "ner",
    "EXTRACTION": "extraction",
    "CLASSIFICATION": "classification",
    "VALIDATION": "validation",
    "HANDWRITING": "handwriting",
    "SIGNATURE": "signature",
    "VERTICAL": "vertical",
    "RETRIEVAL": "retrieval",
    "FORM": "form",
    "AI_TEXT": "ai_text",
    "LANGUAGE": "language",
    "SYMBOLOGY": "symbology",
    "CUSTODY": "custody",
}

# File extension patterns for non-EXPORT artifacts
EXTENSION_MAP: dict[str, str] = {
    ".custody.jsonl": "custody",
}

# MIME types for known output types
_MIME_TYPES: dict[str, str] = {
    "searchable_pdf": "application/pdf",
    "ocr_text": "text/plain",
    "structure": "application/json",
    "entities": "application/json",
    "ner": "application/json",
    "extraction": "application/json",
    "classification": "application/json",
    "validation": "application/json",
    "handwriting": "application/json",
    "signature": "application/json",
    "vertical": "application/json",
    "retrieval": "application/json",
    "form": "application/json",
    "ai_text": "application/json",
    "language": "application/json",
    "symbology": "application/json",
    "custody": "application/jsonl",
}

# Valid output types for the retrieval endpoint
VALID_OUTPUT_TYPES = frozenset(OUTPUT_DIR_MAP.values()) | frozenset(
    EXTENSION_MAP.values()
)

_MAX_MANIFEST_SCAN_ENTRIES = 20_000
_MAX_MANIFEST_ARTIFACTS = 10_000
_MAX_MANIFEST_SCAN_DEPTH = 32

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
class OutputArtifact:
    """A single output artifact produced by the pipeline."""

    output_type: str
    filename: str
    relative_path: str
    size_bytes: int
    mime_type: str = "application/octet-stream"
    schema_version: str = SCHEMA_VERSION
    # Descriptor-discovery metadata is deliberately not serialized. Callers
    # use it to prove a later bounded read/stream opened the exact inode that
    # the manifest authorized, even if an EXPORT child directory was renamed.
    device: int | None = None
    inode: int | None = None
    mtime_ns: int | None = None
    ctime_ns: int | None = None
    nlink: int | None = None


@dataclass
class OutputManifest:
    """Manifest of all output artifacts for a job."""

    job_id: str
    output_dir: str
    artifacts: list[OutputArtifact] = field(default_factory=list)
    schema_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "job_id": self.job_id,
            "artifacts": [
                {
                    "output_type": a.output_type,
                    "filename": a.filename,
                    "relative_path": a.relative_path,
                    "size_bytes": a.size_bytes,
                    "mime_type": a.mime_type,
                    "schema_version": a.schema_version,
                }
                for a in self.artifacts
            ],
            "schema_versions": dict(self.schema_versions),
        }


def build_manifest_from_root_fd(
    job_id: str,
    output_dir: str,
    root_fd: int,
) -> OutputManifest:
    """Build a manifest below an already-authorized output-root descriptor.

    Scans ``output_dir/EXPORT/<subdir>/`` for each known subdirectory
    in :data:`OUTPUT_DIR_MAP`, and also checks for custody files via
    :data:`EXTENSION_MAP`.

    Args:
        job_id: The job identifier.
        output_dir: Absolute path to the job's output directory.

    Returns:
        An :class:`OutputManifest` populated with discovered artifacts.
    """
    manifest = OutputManifest(job_id=job_id, output_dir=output_dir)
    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("output manifest root fd must reference a directory")

    scanned = 0

    def _count_entry() -> None:
        nonlocal scanned
        scanned += 1
        if scanned > _MAX_MANIFEST_SCAN_ENTRIES:
            raise ValueError("output manifest scan exceeds configured bound")

    def _open_relative_directory(parts: list[str]) -> int | None:
        current_fd = os.dup(root_fd)
        try:
            for part in parts:
                try:
                    child_fd = os.open(
                        part,
                        _DIRECTORY_OPEN_FLAGS,
                        dir_fd=current_fd,
                    )
                except (FileNotFoundError, NotADirectoryError, OSError):
                    os.close(current_fd)
                    return None
                os.close(current_fd)
                current_fd = child_fd
            return current_fd
        except Exception:
            os.close(current_fd)
            raise

    def _bounded_regular_files(parts: list[str]):
        """Yield descriptor-bound regular files under a relative directory."""
        start_fd = _open_relative_directory(parts)
        if start_fd is None:
            return

        def _walk_directory(
            current_fd: int,
            current_parts: list[str],
            depth: int = 0,
        ):
            """Walk one directory at a time without retaining descendant fds."""
            names: list[str] = []
            with os.scandir(current_fd) as iterator:
                for entry in iterator:
                    # Enforce the global cap while consuming the iterator,
                    # before sorting or retaining every entry in a large
                    # attacker-controlled directory.
                    _count_entry()
                    names.append(entry.name)
            for name in sorted(names):
                entry_parts = current_parts + [name]
                try:
                    entry_stat = os.stat(
                        name,
                        dir_fd=current_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    continue
                if stat.S_ISLNK(entry_stat.st_mode):
                    continue
                if stat.S_ISDIR(entry_stat.st_mode):
                    if depth >= _MAX_MANIFEST_SCAN_DEPTH:
                        raise ValueError(
                            "output manifest scan exceeds configured depth"
                        )
                    try:
                        child_fd = os.open(
                            name,
                            _DIRECTORY_OPEN_FLAGS,
                            dir_fd=current_fd,
                        )
                    except OSError:
                        continue
                    try:
                        yield from _walk_directory(
                            child_fd,
                            entry_parts,
                            depth + 1,
                        )
                    finally:
                        os.close(child_fd)
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    continue
                if name.endswith(".hitl-hold.json"):
                    continue
                try:
                    file_fd = os.open(
                        name,
                        _FILE_OPEN_FLAGS,
                        dir_fd=current_fd,
                    )
                    try:
                        bound_stat = os.fstat(file_fd)
                    finally:
                        os.close(file_fd)
                except OSError:
                    continue
                if stat.S_ISREG(bound_stat.st_mode) and bound_stat.st_nlink == 1:
                    yield "/".join(entry_parts), bound_stat

        try:
            yield from _walk_directory(start_fd, list(parts))
        finally:
            os.close(start_fd)

    def _direct_regular_files():
        """Yield safe regular files directly below the opened job root."""
        try:
            names: list[str] = []
            with os.scandir(root_fd) as iterator:
                for entry in iterator:
                    _count_entry()
                    names.append(entry.name)
            for name in sorted(names):
                if not any(name.endswith(pattern) for pattern in EXTENSION_MAP):
                    continue
                try:
                    file_fd = os.open(
                        name,
                        _FILE_OPEN_FLAGS,
                        dir_fd=root_fd,
                    )
                    try:
                        bound_stat = os.fstat(file_fd)
                    finally:
                        os.close(file_fd)
                except OSError:
                    continue
                if stat.S_ISREG(bound_stat.st_mode) and bound_stat.st_nlink == 1:
                    yield name, bound_stat
        except OSError:
            return

    def append_artifact(
        relative_path: str,
        metadata: os.stat_result,
        output_type: str,
    ) -> None:
        if len(manifest.artifacts) >= _MAX_MANIFEST_ARTIFACTS:
            raise ValueError("output manifest contains too many artifacts")
        artifact = OutputArtifact(
            output_type=output_type,
            filename=relative_path.rsplit("/", 1)[-1],
            relative_path=relative_path,
            size_bytes=metadata.st_size,
            mime_type=_MIME_TYPES.get(output_type, "application/octet-stream"),
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
            nlink=metadata.st_nlink,
        )
        manifest.artifacts.append(artifact)
        manifest.schema_versions.setdefault(output_type, SCHEMA_VERSION)

    # Scan EXPORT subdirectories.
    for dir_name, output_type in OUTPUT_DIR_MAP.items():
        for relative_path, metadata in _bounded_regular_files(["EXPORT", dir_name]):
            if output_type == "custody" and not relative_path.endswith(
                ".custody.jsonl"
            ):
                continue
            append_artifact(relative_path, metadata, output_type)

    # Scan for custody files directly below the opened output root.
    for relative_path, metadata in _direct_regular_files():
        output_type = next(
            output_type
            for pattern, output_type in EXTENSION_MAP.items()
            if relative_path.endswith(pattern)
        )
        append_artifact(relative_path, metadata, output_type)
    return manifest


def build_manifest(job_id: str, output_dir: str) -> OutputManifest:
    """Build output manifest by binding and scanning one output directory."""
    empty = OutputManifest(job_id=job_id, output_dir=output_dir)
    # Do not resolve the lexical root first: resolution could follow a swapped
    # name into another job's tree before O_NOFOLLOW gets a chance to apply.
    try:
        root_fd = os.open(Path(output_dir), _DIRECTORY_OPEN_FLAGS)
    except OSError:
        return empty
    try:
        return build_manifest_from_root_fd(job_id, output_dir, root_fd)
    finally:
        os.close(root_fd)
