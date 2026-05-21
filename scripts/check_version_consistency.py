#!/usr/bin/env python3
"""Check version consistency across all project version sources.

Reads versions from:
  - version.py                           (__version__)
  - sdk/python/pyproject.toml            (project.version)
  - sdk/python/src/edcocr_sdk/__init__.py  (__version__)
  - sdk/python/src/edcocr_sdk/client.py    (SDK_VERSION)
  - sdk/typescript/package.json          (version)
  - sdk/typescript/src/client.ts         (SDK_VERSION)
  - helm/ocr-local/Chart.yaml           (appVersion)

Exits 0 if all match, exits 1 if any mismatch.
"""

import json
import re
import sys
from pathlib import Path

# Resolve project root relative to this script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Version source file paths (relative to project root)
VERSION_SOURCES = {
    "version.py": "version.py",
    "sdk/python/pyproject.toml": "sdk/python/pyproject.toml",
    "sdk/python/src/edcocr_sdk/__init__.py": "sdk/python/src/edcocr_sdk/__init__.py",
    "sdk/python/src/edcocr_sdk/client.py": "sdk/python/src/edcocr_sdk/client.py",
    "sdk/typescript/package.json": "sdk/typescript/package.json",
    "sdk/typescript/src/client.ts": "sdk/typescript/src/client.ts",
    "helm/ocr-local/Chart.yaml": "helm/ocr-local/Chart.yaml",
}


def _resolve_forwarded_version_path(path: Path, text: str) -> Path | None:
    """Resolve an ``import_module("pkg.version")`` style forwarding shim."""
    match = re.search(r'import_module\(["\']([^"\']+)["\']\)', text)
    if not match:
        return None

    module_parts = match.group(1).split(".")
    search_roots = [path.parent, *path.parent.parents]
    for root in search_roots:
        candidate = root.joinpath(*module_parts).with_suffix(".py")
        if candidate.exists():
            return candidate

    return path.parent.joinpath(*module_parts).with_suffix(".py")


def read_version_py(path: Path, *, _seen: set[Path] | None = None) -> str:
    """Extract ``__version__`` from a Python file or forwarding shim."""
    if _seen is None:
        _seen = set()

    resolved = path.resolve()
    if resolved in _seen:
        raise ValueError(f"Circular version forwarding detected at {path}")
    _seen.add(resolved)

    text = path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if match:
        return match.group(1)

    forwarded_path = _resolve_forwarded_version_path(path, text)
    if forwarded_path is not None and forwarded_path.exists():
        return read_version_py(forwarded_path, _seen=_seen)

    raise ValueError(f"No __version__ found in {path}")


def read_pyproject_version(path: Path) -> str:
    """Extract version from pyproject.toml [project] section."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise ValueError(f"No version found in {path}")
    return match.group(1)


def read_package_json_version(path: Path) -> str:
    """Extract version from package.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if not version:
        raise ValueError(f"No version found in {path}")
    return version


def read_chart_app_version(path: Path) -> str:
    """Extract appVersion from Chart.yaml."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^appVersion:\s*["\']?([^"\'"\n]+)["\']?', text, re.MULTILINE)
    if not match:
        raise ValueError(f"No appVersion found in {path}")
    return match.group(1).strip()


def read_sdk_version_py(path: Path) -> str:
    """Extract SDK_VERSION from a Python client file."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^SDK_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise ValueError(f"No SDK_VERSION found in {path}")
    return match.group(1)


def read_sdk_version_ts(path: Path) -> str:
    """Extract SDK_VERSION from a TypeScript client file."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"""^export\s+const\s+SDK_VERSION\s*=\s*['"]([^'"]+)['"]""", text, re.MULTILINE)
    if not match:
        raise ValueError(f"No SDK_VERSION found in {path}")
    return match.group(1)


def collect_versions(root: Path = None) -> dict[str, str]:
    """Collect versions from all sources.

    Args:
        root: Project root directory. Defaults to auto-detected root.

    Returns:
        Dict mapping source label to version string.
    """
    if root is None:
        root = PROJECT_ROOT

    versions = {}

    extractors = {
        "version.py": read_version_py,
        "sdk/python/pyproject.toml": read_pyproject_version,
        "sdk/python/src/edcocr_sdk/__init__.py": read_version_py,
        "sdk/python/src/edcocr_sdk/client.py": read_sdk_version_py,
        "sdk/typescript/package.json": read_package_json_version,
        "sdk/typescript/src/client.ts": read_sdk_version_ts,
        "helm/ocr-local/Chart.yaml": read_chart_app_version,
    }

    for label, rel_path in VERSION_SOURCES.items():
        full_path = root / rel_path
        if not full_path.exists():
            versions[label] = f"FILE_NOT_FOUND ({rel_path})"
            continue
        try:
            versions[label] = extractors[label](full_path)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            versions[label] = f"PARSE_ERROR ({exc})"

    return versions


def check_consistency(versions: dict[str, str]) -> tuple[bool, str]:
    """Check whether all versions are identical.

    Returns:
        Tuple of (all_match: bool, report: str).
    """
    unique = set(versions.values())
    lines = []

    for label, ver in sorted(versions.items()):
        lines.append(f"  {label}: {ver}")

    if len(unique) == 1:
        ver = next(iter(unique))
        report = f"All {len(versions)} version sources match: {ver}\n" + "\n".join(lines)
        return True, report
    else:
        report = (
            f"VERSION MISMATCH: found {len(unique)} distinct values "
            f"across {len(versions)} sources\n" + "\n".join(lines)
        )
        return False, report


def main() -> int:
    """Run version consistency check. Returns 0 on success, 1 on mismatch."""
    versions = collect_versions()
    ok, report = check_consistency(versions)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
