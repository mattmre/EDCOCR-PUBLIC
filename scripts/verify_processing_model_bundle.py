#!/usr/bin/env python3
"""Generate or verify an exact-file lock for the baked PaddleOCR cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

_SCHEMA_VERSION = 1
_BUFFER_SIZE = 1024 * 1024


class ModelBundleError(ValueError):
    """The model bundle or its lock does not satisfy the admission contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, Any]]:
    root = root.resolve(strict=True)
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            candidate = current / name
            if candidate.is_symlink():
                raise ModelBundleError(
                    f"model bundle contains a symbolic link: {candidate.relative_to(root)}"
                )
        for name in file_names:
            candidate = current / name
            if candidate.is_symlink():
                raise ModelBundleError(
                    f"model bundle contains a symbolic link: {candidate.relative_to(root)}"
                )
            if not candidate.is_file():
                raise ModelBundleError(
                    f"model bundle contains a non-regular file: {candidate.relative_to(root)}"
                )
            files.append(candidate)

    if not files:
        raise ModelBundleError("model bundle contains no regular files")

    inventory: list[dict[str, Any]] = []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative_path = path.relative_to(root).as_posix()
        inventory.append(
            {
                "path": relative_path,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def _bundle_digest(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in files:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_lock(root: Path) -> dict[str, Any]:
    """Build a deterministic exact-file inventory for *root*."""
    files = _inventory(root)
    return {
        "schemaVersion": _SCHEMA_VERSION,
        "algorithm": "sha256",
        "fileCount": len(files),
        "byteCount": sum(entry["size"] for entry in files),
        "bundleSha256": _bundle_digest(files),
        "files": files,
    }


def _validate_lock(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if lock.get("schemaVersion") != _SCHEMA_VERSION:
        raise ModelBundleError(
            f"unsupported schemaVersion: {lock.get('schemaVersion')!r}"
        )
    if lock.get("algorithm") != "sha256":
        raise ModelBundleError("model lock algorithm must be sha256")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise ModelBundleError("model lock files must be a non-empty list")

    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ModelBundleError("model lock file entry must be an object")
        path = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(path, str) or not path:
            raise ModelBundleError("model lock file path must be non-empty")
        pure_path = PurePosixPath(path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or path != pure_path.as_posix()
        ):
            raise ModelBundleError(f"unsafe model lock path: {path!r}")
        if path in seen:
            raise ModelBundleError(f"duplicate model lock path: {path}")
        seen.add(path)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ModelBundleError(f"invalid model lock size for {path}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ModelBundleError(f"invalid sha256 for {path}")

    if lock.get("fileCount") != len(files):
        raise ModelBundleError("model lock fileCount does not match files")
    if lock.get("byteCount") != sum(entry["size"] for entry in files):
        raise ModelBundleError("model lock byteCount does not match files")
    if lock.get("bundleSha256") != _bundle_digest(files):
        raise ModelBundleError("model lock bundleSha256 does not match files")
    return files


def verify_lock(root: Path, lock: dict[str, Any]) -> None:
    """Fail unless *root* exactly matches every file in *lock*."""
    expected = _validate_lock(lock)
    actual = build_lock(root)["files"]
    expected_paths = [entry["path"] for entry in expected]
    actual_paths = [entry["path"] for entry in actual]
    if expected_paths != actual_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected_paths))
        raise ModelBundleError(
            f"model file inventory mismatch: missing={missing}, extra={extra}"
        )

    for expected_entry, actual_entry in zip(expected, actual, strict=True):
        path = expected_entry["path"]
        if expected_entry["size"] != actual_entry["size"]:
            raise ModelBundleError(f"size mismatch for model file: {path}")
        if expected_entry["sha256"] != actual_entry["sha256"]:
            raise ModelBundleError(f"digest mismatch for model file: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "verify"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--root", type=Path, required=True)
        command_parser.add_argument("--lock", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "generate":
        payload = build_lock(args.root)
        args.lock.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "OCR_PROCESSING_MODEL_LOCK_GENERATE=PASS "
            f"files={payload['fileCount']} bytes={payload['byteCount']} "
            f"digest=sha256:{payload['bundleSha256']}"
        )
        return 0

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise ModelBundleError("model lock root must be an object")
    verify_lock(args.root, lock)
    print(
        "OCR_PROCESSING_MODEL_LOCK_VERIFY=PASS "
        f"files={lock['fileCount']} bytes={lock['byteCount']} "
        f"digest=sha256:{lock['bundleSha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
