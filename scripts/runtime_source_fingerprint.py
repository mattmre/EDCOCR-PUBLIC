#!/usr/bin/env python3
"""Fingerprint source/build inputs copied into an OCR_LOCAL runtime image.

The digest is intended for an OCI image label. It binds a built image to the
source snapshot used for the local runtime qualification, including dirty and
untracked source files while excluding docs, evidence, and tests that are not
part of the runtime source payload.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FINGERPRINT_VERSION = b"ocr-local-runtime-source-v5\0"
NON_RUNTIME_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "tests",
        "failover_drill_results",
        "_test_nfs_root",
        "node_modules",
        ".next",
        ".git",
        ".github",
        ".venv",
        "venv",
        "docs",
        "legacy",
        "helm",
        "terraform",
        "playwright",
        "playwright-artifacts",
        "benchmark_results",
        "ocr_output",
        "ocr_temp",
        "ocr_source",
        "models",
        "sdk",
        "dashboard",
        "out",
        "tmp",
        ".hypothesis",
    }
)

PROFILE_INPUTS = {
    "root-cpu": {
        "files": (
            "Dockerfile",
            ".dockerignore",
            "healthcheck.sh",
            "requirements.txt",
            "requirements-language-id.txt",
            "scripts/harden_imgaug_runtime.py",
            "coordinator/requirements.txt",
        ),
        "root_glob": "*.py",
        "directories": (
            "api",
            "coordinator",
            "ocr_distributed",
            "ocr_local",
            "reprocess",
            "schemas",
        ),
    },
    "coordinator": {
        "files": (
            ".dockerignore",
            "coordinator/Dockerfile.coordinator",
            "coordinator/requirements.txt",
            "optimize_pdfs.py",
        ),
        "root_glob": None,
        "directories": ("coordinator", "ocr_distributed", "ocr_local", "schemas"),
    },
    "worker": {
        # Current worker Dockerfiles use COPY . /app. Hash a safe superset of
        # the root Docker context; false invalidation is acceptable, missing
        # copied runtime inputs are not.
        "files": (),
        "root_glob": None,
        "directories": (".",),
    },
    "frontend": {
        "files": (
            ".dockerignore",
            "Dockerfile.frontend",
            "Dockerfile.frontend.dockerignore",
            "frontend/package.json",
            "frontend/package-lock.json",
            "frontend/tsconfig.json",
            "frontend/next.config.mjs",
            "frontend/postcss.config.mjs",
            "frontend/tailwind.config.ts",
            "frontend/.eslintrc.json",
        ),
        "root_glob": None,
        "directories": ("frontend/src", "frontend/public"),
    },
}


def _included_files(root: Path, profile: str) -> list[Path]:
    try:
        inputs = PROFILE_INPUTS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown runtime profile: {profile}") from exc

    files: set[Path] = set()
    for relative in inputs["files"]:
        path = root / relative
        if path.is_file() or path.is_symlink():
            files.add(path)

    root_glob = inputs["root_glob"]
    if root_glob:
        files.update(path for path in root.glob(root_glob) if path.is_file())

    for relative in inputs["directories"]:
        directory = root / relative
        if not directory.is_dir():
            continue
        files.add(directory)
        for path in directory.rglob("*"):
            if not (path.is_file() or path.is_symlink() or path.is_dir()):
                continue
            if (
                any(
                    part in NON_RUNTIME_DIRECTORY_NAMES
                    for part in path.relative_to(root).parts
                )
                or path.name.startswith(".env")
                or path.suffix in {".log", ".pyc"}
            ):
                continue
            files.add(path)

    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def fingerprint_for_profile(root: Path, profile: str) -> str:
    """Return a deterministic SHA-256 over the selected build source inputs."""
    root = root.resolve(strict=True)
    digest = hashlib.sha256()
    digest.update(FINGERPRINT_VERSION)
    digest.update(profile.encode("ascii"))
    digest.update(b"\0")

    for path in _included_files(root, profile):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        mode = path.lstat().st_mode
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(stat.S_IMODE(mode).to_bytes(4, "big"))
        if stat.S_ISLNK(mode):
            target = os.readlink(path).encode("utf-8")
            digest.update(b"L")
            digest.update(len(target).to_bytes(4, "big"))
            digest.update(target)
        elif stat.S_ISDIR(mode):
            digest.update(b"D")
        else:
            content = path.read_bytes()
            digest.update(b"F")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_INPUTS), required=True)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(fingerprint_for_profile(args.repo, args.profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
