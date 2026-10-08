"""Versioned active-output generation pointers for coordinator jobs.

PDF and TEXT bytes are written to immutable generation-scoped locations.  A
single ``settings_json`` pointer update is the visibility commit.  Fixed legacy
paths are resolved only when the pointer key is entirely absent.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

ACTIVE_OUTPUT_POINTER_KEY = "_active_output_generation"
POINTER_VERSION = 1
_GENERATION_RE = re.compile(r"^[a-f0-9]{32}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class ArtifactPointerError(RuntimeError):
    """Raised when a present active-output pointer is unsafe or malformed."""


@dataclass(frozen=True)
class ArtifactLocation:
    kind: str
    generation_id: str | None
    relative_path: str
    locator: str
    sha256: str | None


@dataclass(frozen=True)
class ActiveOutputArtifacts:
    revision: str | None
    backend: str
    legacy: bool
    pdf: ArtifactLocation
    text: ArtifactLocation
    pointer: dict | None


def make_generation_id(seed: str) -> str:
    """Derive a path-safe, collision-resistant generation ID from a claim."""
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:32]


def _validate_generation_id(generation_id: object) -> str:
    if not isinstance(generation_id, str) or not _GENERATION_RE.fullmatch(
        generation_id
    ):
        raise ArtifactPointerError("invalid artifact generation ID")
    return generation_id


def _base_name(source_file: str) -> str:
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    if not base_name or base_name in (".", ".."):
        raise ArtifactPointerError("invalid source filename for artifact pointer")
    return base_name


def generation_relative_path(
    generation_id: str,
    kind: str,
    source_file: str,
) -> str:
    generation_id = _validate_generation_id(generation_id)
    base_name = _base_name(source_file)
    if kind == "pdf":
        directory, extension = "PDF", ".pdf"
    elif kind == "text":
        directory, extension = "TEXT", ".txt"
    else:
        raise ArtifactPointerError(f"unsupported artifact kind: {kind}")
    return (
        f"output/generations/{generation_id}/EXPORT/{directory}/{base_name}{extension}"
    )


def legacy_relative_path(kind: str, source_file: str) -> str:
    base_name = _base_name(source_file)
    if kind == "pdf":
        return f"output/EXPORT/PDF/{base_name}.pdf"
    if kind == "text":
        return f"output/EXPORT/TEXT/{base_name}.txt"
    raise ArtifactPointerError(f"unsupported artifact kind: {kind}")


def _relative_posix_locator(relative_path: str, *, error: str) -> str:
    """Return a canonical relative POSIX locator; never a joined jail path."""
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\x00" in relative_path
    ):
        raise ArtifactPointerError(error)
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts or "." in posix_path.parts:
        raise ArtifactPointerError(error)
    normalized = str(posix_path)
    if normalized != relative_path or "\\" in relative_path:
        raise ArtifactPointerError(error)
    if not posix_path.parts:
        raise ArtifactPointerError(error)
    return normalized


def _validate_nfs_relative_path(relative_path: str) -> str:
    """Validate a canonical NFS artifact path relative to its job root."""
    return _relative_posix_locator(
        relative_path, error="artifact path escapes the NFS job root"
    )


def _safe_nfs_locator(job_root: str, relative_path: str) -> str:
    """Return a contained absolute NFS path for existing worker consumers.

    Readers that need race-resistant authority must open the validated
    ``relative_path`` with a held directory descriptor. ``locator`` remains
    the historical absolute path contract for worker and storage callers.
    """
    if not isinstance(job_root, str) or not job_root:
        raise ArtifactPointerError("NFS job root is missing")
    relative = _validate_nfs_relative_path(relative_path)
    root = os.path.realpath(job_root)
    candidate = os.path.realpath(os.path.join(root, *PurePosixPath(relative).parts))
    try:
        contained = os.path.commonpath((root, candidate)) == root
    except ValueError as exc:
        raise ArtifactPointerError("artifact path crosses filesystem roots") from exc
    if not contained:
        raise ArtifactPointerError("artifact path escapes the NFS job root")
    return candidate


def _safe_s3_locator(job_id: object, relative_path: str) -> str:
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ArtifactPointerError("artifact key escapes the job prefix")
    normalized = str(posix_path)
    if normalized != relative_path or "\\" in relative_path:
        raise ArtifactPointerError("artifact key is not canonical POSIX form")
    prefix = f"jobs/{job_id}/"
    locator = prefix + normalized
    if not locator.startswith(prefix):
        raise ArtifactPointerError("artifact key escapes the job prefix")
    return locator


def _locator(
    *,
    backend: str,
    job_id: object,
    job_root: str,
    relative_path: str,
) -> str:
    if backend == "nfs":
        return _safe_nfs_locator(job_root, relative_path)
    if backend == "s3":
        return _safe_s3_locator(job_id, relative_path)
    raise ArtifactPointerError(f"unsupported artifact backend: {backend}")


def generation_locations(
    *,
    job_id: object,
    source_file: str,
    backend: str,
    generation_id: str,
    job_root: str = "",
) -> tuple[ArtifactLocation, ArtifactLocation]:
    locations = []
    for kind in ("pdf", "text"):
        relative_path = generation_relative_path(generation_id, kind, source_file)
        locations.append(
            ArtifactLocation(
                kind=kind,
                generation_id=generation_id,
                relative_path=relative_path,
                locator=_locator(
                    backend=backend,
                    job_id=job_id,
                    job_root=job_root,
                    relative_path=relative_path,
                ),
                sha256=None,
            )
        )
    return locations[0], locations[1]


def build_active_pointer(
    *,
    job_id: object,
    source_file: str,
    backend: str,
    revision: str,
    pdf_generation_id: str,
    pdf_sha256: str,
    text_generation_id: str,
    text_sha256: str,
) -> dict:
    revision = _validate_generation_id(revision)
    pdf_generation_id = _validate_generation_id(pdf_generation_id)
    text_generation_id = _validate_generation_id(text_generation_id)
    if not _SHA256_RE.fullmatch(pdf_sha256 or ""):
        raise ArtifactPointerError("invalid PDF SHA-256 in artifact pointer")
    if not _SHA256_RE.fullmatch(text_sha256 or ""):
        raise ArtifactPointerError("invalid TEXT SHA-256 in artifact pointer")
    return {
        "version": POINTER_VERSION,
        "job_id": str(job_id),
        "backend": backend,
        "revision": revision,
        "artifacts": {
            "pdf": {
                "generation_id": pdf_generation_id,
                "relative_path": generation_relative_path(
                    pdf_generation_id,
                    "pdf",
                    source_file,
                ),
                "sha256": pdf_sha256,
            },
            "text": {
                "generation_id": text_generation_id,
                "relative_path": generation_relative_path(
                    text_generation_id,
                    "text",
                    source_file,
                ),
                "sha256": text_sha256,
            },
        },
    }


def replace_pointer_pdf(
    pointer: dict,
    *,
    job_id: object,
    source_file: str,
    revision: str,
    pdf_generation_id: str,
    pdf_sha256: str,
) -> dict:
    """Return a copied pointer with a new immutable PDF generation."""
    if not isinstance(pointer, dict) or "artifacts" not in pointer:
        raise ArtifactPointerError("invalid active artifact pointer structure")
    copied = copy.deepcopy(pointer)
    copied["revision"] = _validate_generation_id(revision)
    copied["job_id"] = str(job_id)
    if not isinstance(copied.get("artifacts"), dict):
        raise ArtifactPointerError("invalid active artifact pointer structure")
    copied["artifacts"]["pdf"] = {
        "generation_id": _validate_generation_id(pdf_generation_id),
        "relative_path": generation_relative_path(
            pdf_generation_id,
            "pdf",
            source_file,
        ),
        "sha256": pdf_sha256,
    }
    if not _SHA256_RE.fullmatch(pdf_sha256 or ""):
        raise ArtifactPointerError("invalid PDF SHA-256 in artifact pointer")
    return copied


def _validate_pointer(
    pointer: object,
    *,
    job_id: object,
    source_file: str,
    backend: str,
) -> dict:
    if not isinstance(pointer, dict):
        raise ArtifactPointerError("active artifact pointer must be an object")
    if pointer.get("version") != POINTER_VERSION:
        raise ArtifactPointerError("unsupported active artifact pointer version")
    if pointer.get("job_id") != str(job_id):
        raise ArtifactPointerError("active artifact pointer belongs to another job")
    if pointer.get("backend") != backend:
        raise ArtifactPointerError("active artifact pointer backend mismatch")
    _validate_generation_id(pointer.get("revision"))
    artifacts = pointer.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"pdf", "text"}:
        raise ArtifactPointerError("active artifact pointer set is incomplete")
    for kind in ("pdf", "text"):
        entry = artifacts.get(kind)
        if not isinstance(entry, dict):
            raise ArtifactPointerError(f"active {kind} artifact pointer is invalid")
        generation_id = _validate_generation_id(entry.get("generation_id"))
        expected_path = generation_relative_path(generation_id, kind, source_file)
        if entry.get("relative_path") != expected_path:
            raise ArtifactPointerError(
                f"active {kind} artifact path is unsafe or non-canonical"
            )
        if not _SHA256_RE.fullmatch(entry.get("sha256") or ""):
            raise ArtifactPointerError(f"active {kind} artifact SHA-256 is invalid")
    return pointer


def resolve_active_artifacts(
    *,
    job_id: object,
    source_file: str,
    settings_json: dict | None,
    backend: str,
    job_root: str = "",
) -> ActiveOutputArtifacts:
    settings_json = settings_json or {}
    if ACTIVE_OUTPUT_POINTER_KEY not in settings_json:
        locations = []
        for kind in ("pdf", "text"):
            relative_path = legacy_relative_path(kind, source_file)
            locations.append(
                ArtifactLocation(
                    kind=kind,
                    generation_id=None,
                    relative_path=relative_path,
                    locator=_locator(
                        backend=backend,
                        job_id=job_id,
                        job_root=job_root,
                        relative_path=relative_path,
                    ),
                    sha256=None,
                )
            )
        return ActiveOutputArtifacts(
            revision=None,
            backend=backend,
            legacy=True,
            pdf=locations[0],
            text=locations[1],
            pointer=None,
        )

    pointer = _validate_pointer(
        settings_json[ACTIVE_OUTPUT_POINTER_KEY],
        job_id=job_id,
        source_file=source_file,
        backend=backend,
    )
    resolved = {}
    for kind in ("pdf", "text"):
        entry = pointer["artifacts"][kind]
        resolved[kind] = ArtifactLocation(
            kind=kind,
            generation_id=entry["generation_id"],
            relative_path=entry["relative_path"],
            locator=_locator(
                backend=backend,
                job_id=job_id,
                job_root=job_root,
                relative_path=entry["relative_path"],
            ),
            sha256=entry["sha256"],
        )
    return ActiveOutputArtifacts(
        revision=pointer["revision"],
        backend=backend,
        legacy=False,
        pdf=resolved["pdf"],
        text=resolved["text"],
        pointer=copy.deepcopy(pointer),
    )


def active_generation_ids(pointer: dict) -> frozenset[str]:
    artifacts = pointer.get("artifacts", {}) if isinstance(pointer, dict) else {}
    return frozenset(
        entry.get("generation_id")
        for entry in artifacts.values()
        if isinstance(entry, dict) and isinstance(entry.get("generation_id"), str)
    )


def list_inactive_generation_ids(
    *,
    known_generation_ids: set[str] | frozenset[str],
    active_pointer: dict | None,
) -> frozenset[str]:
    """Return generation IDs safe to delete (present but not active).

    Cleanup must never remove an active generation. Unknown / malformed active
    pointers fail closed by returning an empty set (no deletions).
    """
    if not isinstance(active_pointer, dict):
        return frozenset()
    try:
        keep = active_generation_ids(active_pointer)
    except Exception:
        return frozenset()
    if not keep:
        return frozenset()
    known = {
        value
        for value in known_generation_ids
        if isinstance(value, str) and _GENERATION_RE.fullmatch(value)
    }
    return frozenset(known - keep)
