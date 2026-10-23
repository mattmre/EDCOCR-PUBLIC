#!/usr/bin/env python3
"""Automate mechanical release preparation steps.

Capabilities:
  A) Version Bump  -- update tracked version sources with per-file atomic writes
  B) CHANGELOG     -- insert a new section header with placeholder subsections
  C) Validation    -- run check_version_consistency + verify_release_state
  D) Dry-Run       -- preview changes without writing files

Usage:
    python scripts/prepare_release.py --bump patch [--dry-run] [--skip-validation]
    python scripts/prepare_release.py --version X.Y.Z[-prerelease][+build] [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# SemVer 2.0.0. Groups 1-3 are the numeric core so parse_version keeps its
# tuple[int, int, int] contract. Optional prerelease (group 4) and build
# (group 5) are admitted by the same expression: an empty suffix does not
# match, and a numeric prerelease identifier cannot have a leading zero.
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$",
    re.ASCII,
)
_VERSION_ASSIGN_RE = re.compile(
    r"""^__version__\s*=\s*["']([^"']+)["']""",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Version source definitions
#
# Each entry maps a human label to:
#   - rel_path:  file path relative to project root
#   - pattern:   regex with exactly one capture group for the version string
#   - template:  replacement template using \1 / {version} for substitution
# ---------------------------------------------------------------------------

_VERSION_SOURCES: list[dict] = [
    {
        "label": "ocr_local/config/version.py",
        "rel_path": "ocr_local/config/version.py",
        "pattern": r"""(^__version__\s*=\s*["'])[^"']+(["'])""",
        "template": r"\g<1>{version}\2",
    },
    {
        "label": "sdk/python/pyproject.toml",
        "rel_path": "sdk/python/pyproject.toml",
        "pattern": r'(^version\s*=\s*")[^"]+(")',
        "template": r"\g<1>{version}\2",
    },
    {
        "label": "sdk/python/src/edcocr_sdk/__init__.py",
        "rel_path": "sdk/python/src/edcocr_sdk/__init__.py",
        "pattern": r'(^__version__\s*=\s*["\'])[^"\']+(["\'])',
        "template": r"\g<1>{version}\2",
    },
    {
        "label": "sdk/python/src/edcocr_sdk/client.py",
        "rel_path": "sdk/python/src/edcocr_sdk/client.py",
        "pattern": r'(^SDK_VERSION\s*=\s*["\'])[^"\']+(["\'])',
        "template": r"\g<1>{version}\2",
    },
    {
        "label": "sdk/typescript/package.json",
        "rel_path": "sdk/typescript/package.json",
        # JSON: "version": "X.Y.Z"
        "pattern": r'("version"\s*:\s*")[^"]+(")',
        "template": r"\g<1>{version}\2",
    },
    {
        "label": "sdk/typescript/src/client.ts",
        "rel_path": "sdk/typescript/src/client.ts",
        "pattern": r"""(export\s+const\s+SDK_VERSION\s*=\s*['"])[^'"]+(['"])""",
        "template": r"\g<1>{version}\2",
    },
    {
        "label": "helm/ocr-local/Chart.yaml",
        "rel_path": "helm/ocr-local/Chart.yaml",
        "pattern": r'(^appVersion:\s*)["\']?[^"\'"\n]+["\']?',
        "template": r'\g<1>"{version}"',
    },
    {
        "label": "api/tracing.py",
        "rel_path": "api/tracing.py",
        "pattern": r'(^_SERVICE_VERSION_DEFAULT\s*=\s*["\'])[^"\']+(["\'])',
        "template": r"\g<1>{version}\2",
    },
    {
        "label": "sdk/typescript/ocr_client.ts",
        "rel_path": "sdk/typescript/ocr_client.ts",
        "pattern": r"('User-Agent'\s*:\s*'ocr-local-typescript-sdk/)[^']+(')",
        "template": r"\g<1>{version}\2",
    },
]


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse a SemVer string into ``(major, minor, patch)``.

    Accepts a stable core, a prerelease (``X.Y.Z-rc.1``), and build
    metadata (``X.Y.Z+build`` or both). The tuple is only the numeric
    core. ``bump_version`` increments that core and returns a stable
    version; an explicit prerelease or build is set with ``--version``.

    Raises ValueError when the string is not strict SemVer 2.0.0,
    including an empty suffix or a leading-zero numeric prerelease.
    """
    match = _SEMVER_RE.match(version_str.strip())
    if not match:
        raise ValueError(
            f"Invalid version format: {version_str!r} "
            "(expected SemVer X.Y.Z with optional prerelease/build)"
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump_version(current: str, bump_type: str) -> str:
    """Compute the next stable version for a major, minor, or patch bump.

    Prerelease and build metadata on ``current`` are not preserved.
    """
    major, minor, patch = parse_version(current)
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type!r} (expected major/minor/patch)")


def read_current_version(root: Path) -> str:
    """Read the product ``__version__`` from the canonical version module."""
    path = root / "ocr_local" / "config" / "version.py"
    text = path.read_text(encoding="utf-8")
    match = _VERSION_ASSIGN_RE.search(text)
    if not match:
        raise ValueError(f"Cannot find __version__ in {path}")
    return match.group(1)


# ---------------------------------------------------------------------------
# File update logic
# ---------------------------------------------------------------------------


def compute_file_update(
    root: Path,
    source: dict,
    new_version: str,
) -> dict | None:
    """Compute the updated content for a single version source file.

    Returns a dict with keys:
        label, rel_path, full_path, old_content, new_content, changed

    Returns None if the file does not exist. Product-version updates name
    the canonical file, not a forwarding shim.
    """
    full_path = root / source["rel_path"]
    if not full_path.is_file():
        return None

    old_content = full_path.read_text(encoding="utf-8")
    template = source["template"].replace("{version}", new_version)
    new_content = re.sub(
        source["pattern"],
        template,
        old_content,
        count=1,
        flags=re.MULTILINE,
    )
    rel_path = full_path.relative_to(root).as_posix()
    return {
        "label": source["label"],
        "rel_path": rel_path,
        "full_path": full_path,
        "old_content": old_content,
        "new_content": new_content,
        "changed": new_content != old_content,
    }


def compute_all_updates(
    root: Path,
    new_version: str,
) -> list[dict]:
    """Compute updates for all version source files.

    Returns list of update dicts (only includes existing files).
    """
    updates = []
    for source in _VERSION_SOURCES:
        result = compute_file_update(root, source, new_version)
        if result is not None:
            updates.append(result)
    return updates


def apply_updates_atomic(updates: list[dict]) -> list[str]:
    """Write all file updates atomically using temp-then-rename.

    If any write fails, reports which files were already modified.
    Returns list of modified file relative paths.

    Raises RuntimeError if a write fails partway through.
    """
    modified: list[str] = []
    for update in updates:
        if not update["changed"]:
            continue
        full_path: Path = update["full_path"]
        try:
            # Write to a temp file in the same directory, then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(full_path.parent),
                prefix=f".{full_path.name}.",
                suffix=".tmp",
            )
            try:
                os.write(fd, update["new_content"].encode("utf-8"))
            finally:
                os.close(fd)
            # Atomic rename (on same filesystem)
            os.replace(tmp_path, str(full_path))
            modified.append(update["rel_path"])
        except OSError as exc:
            # Clean up temp file if it exists
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise RuntimeError(
                f"Failed to write {update['rel_path']}: {exc}\n"
                f"Already modified: {modified}"
            ) from exc
    return modified


# ---------------------------------------------------------------------------
# CHANGELOG update
# ---------------------------------------------------------------------------


def compute_changelog_update(
    root: Path,
    new_version: str,
    date_str: str | None = None,
) -> dict | None:
    """Compute CHANGELOG.md update with a new version section.

    Inserts a new section header after the preamble (first non-## section).
    Returns update dict or None if CHANGELOG.md does not exist.
    """
    changelog_path = root / "CHANGELOG.md"
    if not changelog_path.exists():
        return None

    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    old_content = changelog_path.read_text(encoding="utf-8")

    new_section = (
        f"## [{new_version}] - {date_str}\n"
        "\n"
        "### Added\n"
        "\n"
        "### Changed\n"
        "\n"
        "### Fixed\n"
        "\n"
    )

    # Find the first existing ## section header
    first_section = re.search(r"^## ", old_content, re.MULTILINE)
    existing_version = re.search(
        rf"^##[ \t]+\[{re.escape(new_version)}\](?=[ \t\r\n]|$)",
        old_content,
        re.MULTILINE,
    )
    if existing_version:
        new_content = old_content
    elif first_section:
        insert_pos = first_section.start()
        new_content = (
            old_content[:insert_pos] + new_section + old_content[insert_pos:]
        )
    else:
        # No existing sections -- append at end
        new_content = old_content.rstrip("\n") + "\n\n" + new_section

    return {
        "label": "CHANGELOG.md",
        "rel_path": "CHANGELOG.md",
        "full_path": changelog_path,
        "old_content": old_content,
        "new_content": new_content,
        "changed": new_content != old_content,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def run_validation(root: Path) -> tuple[bool, str]:
    """Run check_version_consistency and verify_release_state.

    Returns (passed, summary_message).
    """
    messages = []
    all_ok = True

    # 1) check_version_consistency
    cvc_path = root / "scripts" / "check_version_consistency.py"
    if not cvc_path.exists():
        all_ok = False
        messages.append("[FAIL] Version consistency: check_version_consistency.py not found")
    else:
        try:
            spec = importlib.util.spec_from_file_location(
                "check_version_consistency", str(cvc_path)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            versions = mod.collect_versions(root)
            ok, report = mod.check_consistency(versions)
            if not ok:
                all_ok = False
                messages.append(f"[FAIL] Version consistency:\n{report}")
            else:
                messages.append(f"[PASS] Version consistency: {report.splitlines()[0]}")
        except Exception as exc:
            all_ok = False
            messages.append(f"[FAIL] Version consistency check error: {exc}")

    # 2) verify_release_state (only if importable)
    vrs_path = root / "scripts" / "verify_release_state.py"
    if not vrs_path.exists():
        all_ok = False
        messages.append("[FAIL] Release state: verify_release_state.py not found")
    else:
        try:
            spec = importlib.util.spec_from_file_location(
                "verify_release_state", str(vrs_path)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            results = mod.run_all_checks(root)
            failed = [k for k, v in results.items() if not v["passed"]]
            if failed:
                all_ok = False
                messages.append(
                    f"[FAIL] Release state: {len(failed)} checks failed: {', '.join(failed)}"
                )
            else:
                messages.append(
                    f"[PASS] Release state: all {len(results)} checks passed"
                )
        except Exception as exc:
            all_ok = False
            messages.append(f"[FAIL] Release state check error: {exc}")

    return all_ok, "\n".join(messages)


# ---------------------------------------------------------------------------
# Dry-run reporting
# ---------------------------------------------------------------------------


def format_dry_run(updates: list[dict], changelog_update: dict | None) -> str:
    """Format a diff-style summary of planned changes."""
    lines = []
    all_updates = list(updates)
    if changelog_update:
        all_updates.append(changelog_update)

    for update in all_updates:
        if not update["changed"]:
            continue
        lines.append(f"--- {update['rel_path']}")
        # Show a simplified diff: lines that differ
        old_lines = update["old_content"].splitlines()
        new_lines = update["new_content"].splitlines()

        # Simple line-by-line comparison for changed lines
        max_lines = max(len(old_lines), len(new_lines))
        shown = False
        for i in range(max_lines):
            old = old_lines[i] if i < len(old_lines) else ""
            new = new_lines[i] if i < len(new_lines) else ""
            if old != new:
                if old:
                    lines.append(f"  - {old}")
                if new:
                    lines.append(f"  + {new}")
                shown = True
        if not shown:
            lines.append("  (no changes)")
        lines.append("")

    if not lines:
        lines.append("No changes would be made.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def prepare_release(
    root: Path,
    *,
    bump: str | None = None,
    version: str | None = None,
    dry_run: bool = False,
    skip_validation: bool = False,
) -> int:
    """Run release preparation.

    Args:
        root: Project root directory.
        bump: Bump type ('major', 'minor', 'patch').
        version: Explicit version string (mutually exclusive with bump).
        dry_run: If True, show what would change without modifying files.
        skip_validation: If True, skip post-update validation.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    # Read current version
    try:
        current_version = read_current_version(root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: Cannot read current version: {exc}", file=sys.stderr)
        return 1

    # Compute new version
    if version:
        # Validate explicit version
        try:
            parse_version(version)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        new_version = version.strip()
    elif bump:
        try:
            new_version = bump_version(current_version, bump)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        print("ERROR: Must specify --bump or --version", file=sys.stderr)
        return 1

    print(f"Current version: {current_version}")
    print(f"New version:     {new_version}")
    print()

    if new_version == current_version:
        print("WARNING: New version is the same as current version.")

    # Compute all updates
    updates = compute_all_updates(root, new_version)
    changelog_update = compute_changelog_update(root, new_version)

    # Count changes
    changed_count = sum(1 for u in updates if u["changed"])
    if changelog_update and changelog_update["changed"]:
        changed_count += 1

    if dry_run:
        print("=== DRY RUN (no files will be modified) ===\n")
        print(format_dry_run(updates, changelog_update))
        print(f"Files that would be modified: {changed_count}")
        return 0

    # Apply version source updates
    try:
        modified_files = apply_updates_atomic(updates)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Apply CHANGELOG update
    if changelog_update and changelog_update["changed"]:
        try:
            changelog_modified = apply_updates_atomic([changelog_update])
            modified_files.extend(changelog_modified)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    # Report
    print("Modified files:")
    for f in modified_files:
        print(f"  {f}")

    skipped = [u["rel_path"] for u in updates if not u["changed"]]
    if skipped:
        print(f"\nSkipped (already at {new_version}):")
        for f in skipped:
            print(f"  {f}")

    missing = set(s["rel_path"] for s in _VERSION_SOURCES) - set(
        u["rel_path"] for u in updates
    )
    if missing:
        print("\nNot found (skipped):")
        for f in sorted(missing):
            print(f"  {f}")

    # Validation
    if not skip_validation:
        print("\n--- Validation ---")
        passed, summary = run_validation(root)
        print(summary)
        if not passed:
            print("\nWARNING: Post-update validation has failures.")
            return 1

    print(f"\nRelease preparation complete: {current_version} -> {new_version}")
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare a release: bump versions, update CHANGELOG, validate."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Bump type (major, minor, patch)",
    )
    group.add_argument(
        "--version",
        metavar="SEMVER",
        help="Explicit SemVer to set, including prerelease or build",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-update validation checks",
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        help="Project root directory (default: auto-detect)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT

    return prepare_release(
        root,
        bump=args.bump,
        version=args.version,
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    sys.exit(main())
