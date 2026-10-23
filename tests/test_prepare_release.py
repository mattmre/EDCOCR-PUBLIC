"""Consumer tests for scripts/prepare_release.py.

Owned fixtures use a root ``version.py`` forwarding shim and the canonical
``ocr_local/config/version.py`` module. The public SDK package is
``edcocr_sdk``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.prepare_release import (
    apply_updates_atomic,
    bump_version,
    compute_all_updates,
    parse_version,
    prepare_release,
    read_current_version,
)

SDK_PACKAGE = "edcocr_sdk"
_SHIM = (
    '"""forwarding shim."""\n'
    "import importlib as _importlib\n"
    "import sys as _sys\n"
    '_sys.modules[__name__] = _importlib.import_module("ocr_local.config.version")\n'
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _setup_project(tmp_path: Path, version: str = "1.2.0") -> Path:
    """Create a shim plus the canonical module and the public SDK sources."""
    root = tmp_path / "project"
    root.mkdir()
    _write(root / "version.py", _SHIM)
    _write(
        root / "ocr_local" / "config" / "version.py",
        f'__version__ = "{version}"\n',
    )
    _write(
        root / "sdk" / "python" / "pyproject.toml",
        f'[project]\nname = "edcocr-sdk"\nversion = "{version}"\n',
    )
    _write(
        root / "sdk" / "python" / "src" / SDK_PACKAGE / "__init__.py",
        f'__version__ = "{version}"\n',
    )
    _write(
        root / "sdk" / "python" / "src" / SDK_PACKAGE / "client.py",
        f'SDK_VERSION = "{version}"\n',
    )
    _write(
        root / "CHANGELOG.md",
        "# Changelog\n\n"
        f"## [{version}] - 2026-03-26\n\n"
        "### Added\n- Initial release\n",
    )
    return root


def _file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return hashes


def _canonical(root: Path) -> Path:
    return root / "ocr_local" / "config" / "version.py"


class TestParseVersion:
    @pytest.mark.parametrize(
        "version",
        [
            "1.2.3",
            "0.0.0",
            "10.20.30",
            "4.2.0-rc.1",
            "1.2.3-alpha.0",
            "1.2.3+build",
            "1.2.3+build.01",
            "1.2.3-rc.1+build.5",
            "  4.2.0-rc.1  ",
        ],
    )
    def test_accepts_stable_prerelease_and_build(self, version):
        assert parse_version(version) == tuple(
            int(part) for part in version.strip().split("+", 1)[0].split("-", 1)[0].split(".")
        )

    @pytest.mark.parametrize(
        "version",
        [
            "",
            "1.2",
            "1.2.3.4",
            "1.2.3-",
            "1.2.3+",
            "1.2.3-rc.",
            "1.2.3-.1",
            "1.2.3-rc.1+",
            "1.2.3-+meta",
            "1.2.3-01",
            "1.2.3-rc.01",
            "01.2.3",
            "1.02.3",
            "1.2.03",
            "bad",
        ],
    )
    def test_rejects_malformed_semver(self, version):
        with pytest.raises(ValueError):
            parse_version(version)


class TestBumpVersion:
    def test_patch_minor_and_major_stay_stable(self):
        assert bump_version("1.2.0", "patch") == "1.2.1"
        assert bump_version("1.2.3", "minor") == "1.3.0"
        assert bump_version("1.2.3", "major") == "2.0.0"
        assert bump_version("0.0.0", "patch") == "0.0.1"

    def test_prerelease_and_build_increment_core_only(self):
        assert bump_version("4.2.0-rc.1", "patch") == "4.2.1"
        assert bump_version("4.2.0-rc.1", "minor") == "4.3.0"
        assert bump_version("1.2.3+build.7", "major") == "2.0.0"

    def test_invalid_bump_type(self):
        with pytest.raises(ValueError):
            bump_version("1.0.0", "prerelease")


class TestReadCurrentVersion:
    def test_reads_canonical_prerelease_without_rewriting_shim(self, tmp_path):
        root = _setup_project(tmp_path, "4.2.0-rc.1")
        assert read_current_version(root) == "4.2.0-rc.1"
        assert '__version__ = "4.2.0-rc.1"' in _canonical(root).read_text(encoding="utf-8")

    def test_reads_stable_canonical_version(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        _write(_canonical(root), "__version__ = '1.3.0'\n")
        assert read_current_version(root) == "1.3.0"

    def test_missing_entry_raises(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        with pytest.raises(FileNotFoundError):
            read_current_version(root)

    def test_shim_without_target_raises(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        _write(root / "version.py", _SHIM)
        with pytest.raises(FileNotFoundError):
            read_current_version(root)


class TestPrepareRelease:
    def test_dry_run_does_not_mutate(self, tmp_path):
        root = _setup_project(tmp_path, "4.2.0-rc.1")
        before = _file_hashes(root)
        result = prepare_release(root, version="4.2.0-rc.1", dry_run=True)
        assert result == 0
        assert _file_hashes(root) == before
        assert (root / "version.py").read_text(encoding="utf-8") == _SHIM

    def test_explicit_prerelease_writes_canonical_not_shim(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        result = prepare_release(root, version="3.0.0-rc.1", skip_validation=True)
        assert result == 0
        assert read_current_version(root) == "3.0.0-rc.1"
        assert '__version__ = "3.0.0-rc.1"' in _canonical(root).read_text(encoding="utf-8")
        assert (root / "version.py").read_text(encoding="utf-8") == _SHIM
        init_py = root / "sdk" / "python" / "src" / SDK_PACKAGE / "__init__.py"
        assert '__version__ = "3.0.0-rc.1"' in init_py.read_text(encoding="utf-8")
        client_py = root / "sdk" / "python" / "src" / SDK_PACKAGE / "client.py"
        assert 'SDK_VERSION = "3.0.0-rc.1"' in client_py.read_text(encoding="utf-8")

    def test_explicit_build_writes_canonical(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        result = prepare_release(root, version="1.2.3+build.7", skip_validation=True)
        assert result == 0
        assert read_current_version(root) == "1.2.3+build.7"
        assert (root / "version.py").read_text(encoding="utf-8") == _SHIM

    def test_same_version_preserves_existing_candidate_section(self, tmp_path):
        root = _setup_project(tmp_path, "4.2.0-rc.1")
        _write(
            root / "CHANGELOG.md",
            "# Changelog\n\n## [4.2.0-rc.1] - Unpublished candidate\n\n"
            "- Existing candidate evidence.\n",
        )
        before = _file_hashes(root)
        assert prepare_release(root, version="4.2.0-rc.1", skip_validation=True) == 0
        assert _file_hashes(root) == before

    def test_malformed_version_does_not_write(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        before = _file_hashes(root)
        for bad in ("1.2.3-", "1.2.3-01", "1.2.3+", "bad", "1١.2.3", "1.2.3-rc.١"):
            assert prepare_release(root, version=bad, skip_validation=True) == 1
            assert _file_hashes(root) == before

    def test_stable_bump_updates_canonical_and_changelog(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        assert prepare_release(root, bump="patch", skip_validation=True) == 0
        assert read_current_version(root) == "1.2.1"
        assert (root / "version.py").read_text(encoding="utf-8") == _SHIM
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## [1.2.1]" in changelog
        assert "## [1.2.0]" in changelog

    def test_bump_from_prerelease_promotes_stable_core(self, tmp_path):
        root = _setup_project(tmp_path, "4.2.0-rc.1")
        assert prepare_release(root, bump="patch", skip_validation=True) == 0
        assert read_current_version(root) == "4.2.1"
        assert (root / "version.py").read_text(encoding="utf-8") == _SHIM

    def test_missing_version_entry_fails(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        assert prepare_release(root, bump="patch", skip_validation=True) == 1


class TestApplyUpdatesAtomic:
    def test_writes_canonical_and_leaves_no_temp_file(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        updates = compute_all_updates(root, "9.9.9-rc.2")
        modified = apply_updates_atomic(updates)
        assert "ocr_local/config/version.py" in modified
        assert "version.py" not in modified
        assert read_current_version(root) == "9.9.9-rc.2"
        assert (root / "version.py").read_text(encoding="utf-8") == _SHIM
        assert list(root.rglob("*.tmp")) == []

    def test_failed_replace_keeps_completed_file_intact(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        updates = [u for u in compute_all_updates(root, "2.0.0") if u["changed"]]
        assert len(updates) >= 2
        calls = {"n": 0}
        original = os.replace

        def failing_replace(src, dst):
            calls["n"] += 1
            if calls["n"] > 1:
                raise OSError("simulated write failure")
            return original(src, dst)

        with patch("scripts.prepare_release.os.replace", side_effect=failing_replace):
            with pytest.raises(RuntimeError):
                apply_updates_atomic(updates)

        first = updates[0]["full_path"]
        assert first.read_text(encoding="utf-8") == updates[0]["new_content"]
        assert (root / "version.py").read_text(encoding="utf-8") == _SHIM
        assert list(root.rglob("*.tmp")) == []
