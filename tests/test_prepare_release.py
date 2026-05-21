"""Tests for scripts/prepare_release.py."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.prepare_release import (
    _VERSION_SOURCES,
    apply_updates_atomic,
    bump_version,
    compute_all_updates,
    compute_changelog_update,
    compute_file_update,
    format_dry_run,
    parse_version,
    prepare_release,
    read_current_version,
    run_validation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    """Write content to path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _setup_project(
    tmp_path: Path,
    version: str = "1.2.0",
) -> Path:
    """Create a minimal project root with all version source files."""
    root = tmp_path / "project"
    root.mkdir()

    # version.py
    _write(root / "version.py", f'__version__ = "{version}"\n')

    # sdk/python/pyproject.toml
    _write(
        root / "sdk" / "python" / "pyproject.toml",
        f'[project]\nname = "edcocr-sdk"\nversion = "{version}"\n',
    )

    # sdk/python/src/edcocr_sdk/__init__.py
    _write(
        root / "sdk" / "python" / "src" / "edcocr_sdk" / "__init__.py",
        f'__version__ = "{version}"\n',
    )

    # sdk/python/src/edcocr_sdk/client.py
    _write(
        root / "sdk" / "python" / "src" / "edcocr_sdk" / "client.py",
        f'SDK_VERSION = "{version}"\n'
        f'USER_AGENT = f"edcocr-sdk-python/{{SDK_VERSION}}"\n',
    )

    # sdk/typescript/package.json
    pkg_json = json.dumps(
        {"name": "@edcocr/sdk", "version": version}, indent=2
    )
    _write(root / "sdk" / "typescript" / "package.json", pkg_json)

    # sdk/typescript/src/client.ts
    _write(
        root / "sdk" / "typescript" / "src" / "client.ts",
        f"export const SDK_VERSION = '{version}';\n",
    )

    # helm/ocr-local/Chart.yaml
    _write(
        root / "helm" / "ocr-local" / "Chart.yaml",
        f'apiVersion: v2\nname: ocr-local\nversion: 0.4.0\nappVersion: "{version}"\n',
    )

    # api/tracing.py
    _write(
        root / "api" / "tracing.py",
        f'_SERVICE_VERSION_DEFAULT = "{version}"\n',
    )

    # sdk/typescript/ocr_client.ts
    _write(
        root / "sdk" / "typescript" / "ocr_client.ts",
        f"const headers = {{\n"
        f"  'User-Agent': 'ocr-local-typescript-sdk/{version}',\n"
        f"}};\n",
    )

    # CHANGELOG.md
    _write(
        root / "CHANGELOG.md",
        "# Changelog\n\n"
        f"## [{version}] - 2026-03-26\n\n"
        "### Added\n- Initial release\n",
    )

    return root


# ---------------------------------------------------------------------------
# parse_version tests
# ---------------------------------------------------------------------------


class TestParseVersion:
    def test_valid_version(self):
        assert parse_version("1.2.3") == (1, 2, 3)

    def test_zero_version(self):
        assert parse_version("0.0.0") == (0, 0, 0)

    def test_large_numbers(self):
        assert parse_version("10.20.30") == (10, 20, 30)

    def test_whitespace_stripped(self):
        assert parse_version("  1.2.3  ") == (1, 2, 3)

    def test_invalid_format_alpha(self):
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2.abc")

    def test_invalid_format_extra_dot(self):
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2.3.4")

    def test_invalid_format_empty(self):
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("")

    def test_invalid_format_partial(self):
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2")


# ---------------------------------------------------------------------------
# bump_version tests
# ---------------------------------------------------------------------------


class TestBumpVersion:
    def test_bump_patch(self):
        assert bump_version("1.2.0", "patch") == "1.2.1"

    def test_bump_minor(self):
        assert bump_version("1.2.3", "minor") == "1.3.0"

    def test_bump_major(self):
        assert bump_version("1.2.3", "major") == "2.0.0"

    def test_bump_patch_from_zero(self):
        assert bump_version("0.0.0", "patch") == "0.0.1"

    def test_bump_minor_resets_patch(self):
        assert bump_version("1.2.5", "minor") == "1.3.0"

    def test_bump_major_resets_minor_and_patch(self):
        assert bump_version("1.5.9", "major") == "2.0.0"

    def test_invalid_bump_type(self):
        with pytest.raises(ValueError, match="Invalid bump type"):
            bump_version("1.0.0", "prerelease")


# ---------------------------------------------------------------------------
# read_current_version tests
# ---------------------------------------------------------------------------


class TestReadCurrentVersion:
    def test_reads_version(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        _write(root / "version.py", '__version__ = "2.5.1"\n')
        assert read_current_version(root) == "2.5.1"

    def test_single_quotes(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        _write(root / "version.py", "__version__ = '3.0.0'\n")
        assert read_current_version(root) == "3.0.0"

    def test_missing_file(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        with pytest.raises(FileNotFoundError):
            read_current_version(root)

    def test_no_version_string(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        _write(root / "version.py", "# no version here\n")
        with pytest.raises(ValueError, match="Cannot find __version__"):
            read_current_version(root)


# ---------------------------------------------------------------------------
# compute_file_update tests
# ---------------------------------------------------------------------------


class TestComputeFileUpdate:
    def test_updates_version_py(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        source = _VERSION_SOURCES[0]  # version.py
        result = compute_file_update(root, source, "1.1.0")
        assert result is not None
        assert result["changed"]
        assert '"1.1.0"' in result["new_content"]

    def test_no_change_when_same_version(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        source = _VERSION_SOURCES[0]  # version.py
        result = compute_file_update(root, source, "1.0.0")
        assert result is not None
        assert not result["changed"]

    def test_missing_file_returns_none(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        source = _VERSION_SOURCES[0]
        result = compute_file_update(root, source, "1.0.0")
        assert result is None

    def test_updates_pyproject_toml(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        source = _VERSION_SOURCES[1]  # sdk/python/pyproject.toml
        result = compute_file_update(root, source, "2.0.0")
        assert result is not None
        assert result["changed"]
        assert 'version = "2.0.0"' in result["new_content"]

    def test_updates_package_json(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        source = _VERSION_SOURCES[4]  # sdk/typescript/package.json
        result = compute_file_update(root, source, "3.0.0")
        assert result is not None
        assert result["changed"]
        assert '"version": "3.0.0"' in result["new_content"]

    def test_updates_chart_yaml(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        source = _VERSION_SOURCES[6]  # helm/ocr-local/Chart.yaml
        result = compute_file_update(root, source, "2.0.0")
        assert result is not None
        assert result["changed"]
        assert '"2.0.0"' in result["new_content"]


# ---------------------------------------------------------------------------
# compute_all_updates tests
# ---------------------------------------------------------------------------


class TestComputeAllUpdates:
    def test_all_sources_found(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        updates = compute_all_updates(root, "1.1.0")
        assert len(updates) == len(_VERSION_SOURCES)
        assert all(u["changed"] for u in updates)

    def test_partial_project(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        _write(root / "version.py", '__version__ = "1.0.0"\n')
        updates = compute_all_updates(root, "1.1.0")
        assert len(updates) == 1
        assert updates[0]["label"] == "version.py"


# ---------------------------------------------------------------------------
# apply_updates_atomic tests
# ---------------------------------------------------------------------------


class TestApplyUpdatesAtomic:
    def test_writes_changed_files(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        updates = compute_all_updates(root, "2.0.0")
        modified = apply_updates_atomic(updates)
        assert len(modified) == len(_VERSION_SOURCES)

        # Verify content was written
        assert read_current_version(root) == "2.0.0"

    def test_skips_unchanged_files(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        updates = compute_all_updates(root, "1.0.0")
        modified = apply_updates_atomic(updates)
        assert len(modified) == 0

    def test_error_reports_already_modified(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        updates = compute_all_updates(root, "2.0.0")
        # Make the second file's directory read-only to force error
        # (This is platform-dependent, so we mock os.replace instead)
        call_count = 0

        import os as _os

        orig_replace = _os.replace

        def failing_replace(src, dst):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise OSError("Simulated write failure")
            return orig_replace(src, dst)

        with patch("scripts.prepare_release.os.replace", side_effect=failing_replace):
            with pytest.raises(RuntimeError, match="Failed to write"):
                apply_updates_atomic(updates)


# ---------------------------------------------------------------------------
# compute_changelog_update tests
# ---------------------------------------------------------------------------


class TestComputeChangelogUpdate:
    def test_inserts_new_section(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        result = compute_changelog_update(root, "1.1.0", date_str="2026-04-01")
        assert result is not None
        assert result["changed"]
        assert "## [1.1.0] - 2026-04-01" in result["new_content"]
        assert "### Added" in result["new_content"]
        assert "### Changed" in result["new_content"]
        assert "### Fixed" in result["new_content"]

    def test_preserves_existing_content(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        result = compute_changelog_update(root, "1.1.0")
        assert result is not None
        # Original content should still be there
        assert "## [1.0.0]" in result["new_content"]
        assert "Initial release" in result["new_content"]

    def test_new_section_before_existing(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        result = compute_changelog_update(root, "1.1.0")
        assert result is not None
        # New section should come before old section
        new_pos = result["new_content"].index("## [1.1.0]")
        old_pos = result["new_content"].index("## [1.0.0]")
        assert new_pos < old_pos

    def test_missing_changelog(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        result = compute_changelog_update(root, "1.0.0")
        assert result is None

    def test_changelog_without_existing_sections(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        _write(root / "CHANGELOG.md", "# Changelog\n\nSome preamble text.\n")
        result = compute_changelog_update(root, "1.0.0", date_str="2026-01-01")
        assert result is not None
        assert result["changed"]
        assert "## [1.0.0] - 2026-01-01" in result["new_content"]

    def test_uses_current_date_by_default(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        result = compute_changelog_update(root, "1.1.0")
        assert result is not None
        # Should contain today's date
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert f"## [1.1.0] - {today}" in result["new_content"]


# ---------------------------------------------------------------------------
# format_dry_run tests
# ---------------------------------------------------------------------------


class TestFormatDryRun:
    def test_shows_changes(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        updates = compute_all_updates(root, "2.0.0")
        output = format_dry_run(updates, None)
        assert "version.py" in output
        assert "2.0.0" in output

    def test_no_changes_message(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        updates = compute_all_updates(root, "1.0.0")
        output = format_dry_run(updates, None)
        assert "No changes" in output

    def test_includes_changelog(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        updates = compute_all_updates(root, "1.0.0")
        changelog = compute_changelog_update(root, "1.1.0")
        output = format_dry_run(updates, changelog)
        assert "CHANGELOG.md" in output


# ---------------------------------------------------------------------------
# run_validation tests
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_passes_with_consistent_versions(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        # Create scripts dir with check_version_consistency
        scripts_dir = root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        _write(
            scripts_dir / "check_version_consistency.py",
            textwrap.dedent("""\
                def collect_versions(root):
                    return {"version.py": "1.0.0"}

                def check_consistency(versions):
                    return True, "All 1 version sources match: 1.0.0"
            """),
        )
        _write(
            scripts_dir / "verify_release_state.py",
            textwrap.dedent("""\
                def run_all_checks(root):
                    return {"version": {"passed": True}}
            """),
        )

        passed, summary = run_validation(root)
        assert passed
        assert "[PASS]" in summary


# ---------------------------------------------------------------------------
# prepare_release integration tests
# ---------------------------------------------------------------------------


class TestPrepareRelease:
    def test_bump_patch(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        result = prepare_release(root, bump="patch", skip_validation=True)
        assert result == 0
        assert read_current_version(root) == "1.2.1"

    def test_bump_minor(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        result = prepare_release(root, bump="minor", skip_validation=True)
        assert result == 0
        assert read_current_version(root) == "1.3.0"

    def test_bump_major(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        result = prepare_release(root, bump="major", skip_validation=True)
        assert result == 0
        assert read_current_version(root) == "2.0.0"

    def test_explicit_version(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        result = prepare_release(root, version="3.0.0", skip_validation=True)
        assert result == 0
        assert read_current_version(root) == "3.0.0"

    def test_dry_run_no_modifications(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        result = prepare_release(root, bump="patch", dry_run=True)
        assert result == 0
        # Version should NOT have changed
        assert read_current_version(root) == "1.2.0"

    def test_invalid_explicit_version(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        result = prepare_release(root, version="bad", skip_validation=True)
        assert result == 1

    def test_no_bump_or_version(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        result = prepare_release(root, skip_validation=True)
        assert result == 1

    def test_missing_version_py(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        result = prepare_release(root, bump="patch", skip_validation=True)
        assert result == 1

    def test_changelog_updated(self, tmp_path):
        root = _setup_project(tmp_path, "1.2.0")
        prepare_release(root, bump="patch", skip_validation=True)
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## [1.2.1]" in changelog
        # Original entry preserved
        assert "## [1.2.0]" in changelog

    def test_all_files_updated(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        prepare_release(root, version="2.0.0", skip_validation=True)

        # Check each file was updated
        assert read_current_version(root) == "2.0.0"

        pyproject = (root / "sdk" / "python" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        assert 'version = "2.0.0"' in pyproject

        init_py = (
            root / "sdk" / "python" / "src" / "edcocr_sdk" / "__init__.py"
        ).read_text(encoding="utf-8")
        assert '__version__ = "2.0.0"' in init_py

        client_py = (
            root / "sdk" / "python" / "src" / "edcocr_sdk" / "client.py"
        ).read_text(encoding="utf-8")
        assert 'SDK_VERSION = "2.0.0"' in client_py

        pkg_json = json.loads(
            (root / "sdk" / "typescript" / "package.json").read_text(
                encoding="utf-8"
            )
        )
        assert pkg_json["version"] == "2.0.0"

        client_ts = (root / "sdk" / "typescript" / "src" / "client.ts").read_text(
            encoding="utf-8"
        )
        assert "SDK_VERSION = '2.0.0'" in client_ts

        chart = (root / "helm" / "ocr-local" / "Chart.yaml").read_text(
            encoding="utf-8"
        )
        assert '"2.0.0"' in chart

        tracing = (root / "api" / "tracing.py").read_text(encoding="utf-8")
        assert '_SERVICE_VERSION_DEFAULT = "2.0.0"' in tracing

        ocr_client = (root / "sdk" / "typescript" / "ocr_client.ts").read_text(
            encoding="utf-8"
        )
        assert "ocr-local-typescript-sdk/2.0.0" in ocr_client
