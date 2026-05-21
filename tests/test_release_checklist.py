"""Tests for scripts/release_checklist.py."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.release_checklist import (
    check_changelog,
    check_ci_status,
    check_docker_build,
    check_git_state,
    check_release_evidence,
    check_sdk_packages,
    check_tag_readiness,
    check_version_consistency,
    format_markdown_report,
    format_text_table,
    run_checklist,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    """Write content to path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _setup_project(tmp_path: Path, version: str = "1.2.0") -> Path:
    """Create minimal project root for checklist testing."""
    root = tmp_path / "project"
    root.mkdir()
    _write(root / "version.py", f'__version__ = "{version}"\n')
    return root


# ---------------------------------------------------------------------------
# A) check_version_consistency
# ---------------------------------------------------------------------------


class TestCheckVersionConsistency:
    def test_passes_with_consistent_versions(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        scripts_dir = root / "scripts"
        scripts_dir.mkdir()
        _write(
            scripts_dir / "check_version_consistency.py",
            textwrap.dedent("""\
                def collect_versions(root):
                    return {"version.py": "1.0.0"}

                def check_consistency(versions):
                    return True, "All 1 version sources match: 1.0.0"
            """),
        )
        result = check_version_consistency(root)
        assert result["passed"]
        assert "match" in result["detail"]

    def test_fails_with_mismatch(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        scripts_dir = root / "scripts"
        scripts_dir.mkdir()
        _write(
            scripts_dir / "check_version_consistency.py",
            textwrap.dedent("""\
                def collect_versions(root):
                    return {"a": "1.0.0", "b": "2.0.0"}

                def check_consistency(versions):
                    return False, "VERSION MISMATCH: found 2 distinct values"
            """),
        )
        result = check_version_consistency(root)
        assert not result["passed"]

    def test_handles_import_error(self, tmp_path):
        root = _setup_project(tmp_path)
        # No scripts dir -- import will fail
        result = check_version_consistency(root)
        assert not result["passed"]
        assert "Error" in result["detail"]


# ---------------------------------------------------------------------------
# B) check_changelog
# ---------------------------------------------------------------------------


class TestCheckChangelog:
    def test_passes_with_valid_entry(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        _write(
            root / "CHANGELOG.md",
            "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n- Feature\n",
        )
        result = check_changelog(root)
        assert result["passed"]
        assert "content" in result["detail"]

    def test_fails_when_missing(self, tmp_path):
        root = _setup_project(tmp_path)
        result = check_changelog(root)
        assert not result["passed"]
        assert "not found" in result["detail"]

    def test_fails_when_version_not_in_changelog(self, tmp_path):
        root = _setup_project(tmp_path, "2.0.0")
        _write(
            root / "CHANGELOG.md",
            "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n- Old\n",
        )
        result = check_changelog(root)
        assert not result["passed"]
        assert "No entry" in result["detail"]

    def test_fails_when_entry_has_no_content(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        _write(
            root / "CHANGELOG.md",
            "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n## [0.9.0] - 2025-12-01\n\n### Added\n- Old\n",
        )
        result = check_changelog(root)
        assert not result["passed"]
        assert "no content" in result["detail"]

    def test_fails_when_version_py_missing(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        _write(root / "CHANGELOG.md", "# Changelog\n")
        result = check_changelog(root)
        assert not result["passed"]


# ---------------------------------------------------------------------------
# C) check_git_state
# ---------------------------------------------------------------------------


class TestCheckGitState:
    def test_passes_clean_main(self, tmp_path):
        root = _setup_project(tmp_path)
        mock_results = [
            MagicMock(stdout="", returncode=0),       # status --porcelain
            MagicMock(stdout="main\n", returncode=0),  # rev-parse
            MagicMock(returncode=0),                    # fetch
            MagicMock(stdout="0\n", returncode=0),     # rev-list
        ]
        with patch("scripts.release_checklist.subprocess.run", side_effect=mock_results):
            result = check_git_state(root)
        assert result["passed"]

    def test_fails_uncommitted_changes(self, tmp_path):
        root = _setup_project(tmp_path)
        mock_results = [
            MagicMock(stdout="M file.py\n", returncode=0),  # status --porcelain
            MagicMock(stdout="main\n", returncode=0),         # rev-parse
            MagicMock(returncode=0),                           # fetch
            MagicMock(stdout="0\n", returncode=0),            # rev-list
        ]
        with patch("scripts.release_checklist.subprocess.run", side_effect=mock_results):
            result = check_git_state(root)
        assert not result["passed"]
        assert "uncommitted" in result["detail"]

    def test_fails_not_on_main(self, tmp_path):
        root = _setup_project(tmp_path)
        mock_results = [
            MagicMock(stdout="", returncode=0),             # status
            MagicMock(stdout="feat/test\n", returncode=0),  # rev-parse
            MagicMock(returncode=0),                         # fetch
            MagicMock(stdout="0\n", returncode=0),          # rev-list
        ]
        with patch("scripts.release_checklist.subprocess.run", side_effect=mock_results):
            result = check_git_state(root)
        assert not result["passed"]
        assert "feat/test" in result["detail"]

    def test_fails_behind_remote(self, tmp_path):
        root = _setup_project(tmp_path)
        mock_results = [
            MagicMock(stdout="", returncode=0),            # status
            MagicMock(stdout="main\n", returncode=0),      # rev-parse
            MagicMock(returncode=0),                        # fetch
            MagicMock(stdout="3\n", returncode=0),         # rev-list
        ]
        with patch("scripts.release_checklist.subprocess.run", side_effect=mock_results):
            result = check_git_state(root)
        assert not result["passed"]
        assert "behind" in result["detail"]

    def test_handles_timeout(self, tmp_path):
        root = _setup_project(tmp_path)
        import subprocess

        with patch(
            "scripts.release_checklist.subprocess.run",
            side_effect=subprocess.TimeoutExpired("git", 30),
        ):
            result = check_git_state(root)
        assert not result["passed"]
        assert "git error" in result["detail"]


# ---------------------------------------------------------------------------
# D) check_ci_status
# ---------------------------------------------------------------------------


class TestCheckCIStatus:
    def test_passes_successful_run(self, tmp_path):
        root = _setup_project(tmp_path)
        runs = [{"status": "completed", "conclusion": "success",
                 "name": "CI", "headSha": "abc12345"}]
        mock_result = MagicMock(
            returncode=0, stdout=json.dumps(runs)
        )
        with patch("scripts.release_checklist.subprocess.run", return_value=mock_result):
            result = check_ci_status(root)
        assert result["passed"]
        assert "passed" in result["detail"]

    def test_fails_failed_run(self, tmp_path):
        root = _setup_project(tmp_path)
        runs = [{"status": "completed", "conclusion": "failure",
                 "name": "CI", "headSha": "abc12345"}]
        mock_result = MagicMock(
            returncode=0, stdout=json.dumps(runs)
        )
        with patch("scripts.release_checklist.subprocess.run", return_value=mock_result):
            result = check_ci_status(root)
        assert not result["passed"]
        assert "failure" in result["detail"]

    def test_fails_in_progress(self, tmp_path):
        root = _setup_project(tmp_path)
        runs = [{"status": "in_progress", "conclusion": "",
                 "name": "CI", "headSha": "abc12345"}]
        mock_result = MagicMock(
            returncode=0, stdout=json.dumps(runs)
        )
        with patch("scripts.release_checklist.subprocess.run", return_value=mock_result):
            result = check_ci_status(root)
        assert not result["passed"]
        assert "in_progress" in result["detail"]

    def test_fails_no_runs(self, tmp_path):
        root = _setup_project(tmp_path)
        mock_result = MagicMock(returncode=0, stdout="[]")
        with patch("scripts.release_checklist.subprocess.run", return_value=mock_result):
            result = check_ci_status(root)
        assert not result["passed"]
        assert "No CI runs" in result["detail"]

    def test_handles_gh_not_found(self, tmp_path):
        root = _setup_project(tmp_path)
        with patch(
            "scripts.release_checklist.subprocess.run",
            side_effect=FileNotFoundError("gh"),
        ):
            result = check_ci_status(root)
        assert not result["passed"]
        assert "not found" in result["detail"]

    def test_handles_gh_error(self, tmp_path):
        root = _setup_project(tmp_path)
        mock_result = MagicMock(returncode=1, stderr="auth required")
        with patch("scripts.release_checklist.subprocess.run", return_value=mock_result):
            result = check_ci_status(root)
        assert not result["passed"]
        assert "auth" in result["detail"]


# ---------------------------------------------------------------------------
# E) check_docker_build
# ---------------------------------------------------------------------------


class TestCheckDockerBuild:
    def test_passes_on_success(self, tmp_path):
        root = _setup_project(tmp_path)
        mock_result = MagicMock(returncode=0)
        with patch("scripts.release_checklist.subprocess.run", return_value=mock_result):
            result = check_docker_build(root)
        assert result["passed"]

    def test_fails_on_build_error(self, tmp_path):
        root = _setup_project(tmp_path)
        _write(root / "Dockerfile", "FROM scratch\n")
        mock_result = MagicMock(returncode=1, stderr="build error")
        with patch("scripts.release_checklist.subprocess.run", return_value=mock_result):
            result = check_docker_build(root)
        assert not result["passed"]


# ---------------------------------------------------------------------------
# F) check_sdk_packages
# ---------------------------------------------------------------------------


class TestCheckSdkPackages:
    def test_fails_when_pyproject_missing(self, tmp_path):
        root = _setup_project(tmp_path)
        result = check_sdk_packages(root)
        assert not result["passed"]
        assert "not found" in result["detail"]


# ---------------------------------------------------------------------------
# G) check_release_evidence
# ---------------------------------------------------------------------------


class TestCheckReleaseEvidence:
    def test_passes_when_evidence_exists(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        evidence_dir = root / "docs" / "release-evidence" / "v1.0.0"
        evidence_dir.mkdir(parents=True)
        _write(evidence_dir / "evidence.md", "# Evidence\n")
        result = check_release_evidence(root)
        assert result["passed"]

    def test_fails_when_no_evidence(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        result = check_release_evidence(root)
        assert not result["passed"]
        assert "No evidence" in result["detail"]

    def test_fails_when_version_unreadable(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        result = check_release_evidence(root)
        assert not result["passed"]


# ---------------------------------------------------------------------------
# H) check_tag_readiness
# ---------------------------------------------------------------------------


class TestCheckTagReadiness:
    def test_passes_when_tag_absent(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        mock_result = MagicMock(stdout="", returncode=0)
        with patch("scripts.release_checklist.subprocess.run", return_value=mock_result):
            result = check_tag_readiness(root)
        assert result["passed"]
        assert "does not exist" in result["detail"]

    def test_fails_when_tag_exists(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        mock_result = MagicMock(stdout="v1.0.0\n", returncode=0)
        with patch("scripts.release_checklist.subprocess.run", return_value=mock_result):
            result = check_tag_readiness(root)
        assert not result["passed"]
        assert "already exists" in result["detail"]

    def test_handles_git_error(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        import subprocess

        with patch(
            "scripts.release_checklist.subprocess.run",
            side_effect=subprocess.TimeoutExpired("git", 10),
        ):
            result = check_tag_readiness(root)
        assert not result["passed"]


# ---------------------------------------------------------------------------
# run_checklist tests
# ---------------------------------------------------------------------------


class TestRunChecklist:
    def test_core_checks_always_run(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        scripts_dir = root / "scripts"
        scripts_dir.mkdir()
        _write(
            scripts_dir / "check_version_consistency.py",
            textwrap.dedent("""\
                def collect_versions(root):
                    return {"version.py": "1.0.0"}

                def check_consistency(versions):
                    return True, "All match: 1.0.0"
            """),
        )
        _write(root / "CHANGELOG.md", "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n- Feat\n")

        with patch("scripts.release_checklist.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            results = run_checklist(root)

        # Should have: version, changelog, git, ci, evidence, tag = 6
        assert len(results) == 6
        names = [r["name"] for r in results]
        assert "Version consistency" in names
        assert "CHANGELOG entry" in names
        assert "Git state" in names
        assert "CI status" in names
        assert "Release evidence" in names
        assert "Tag readiness" in names

    def test_optional_checks_included(self, tmp_path):
        root = _setup_project(tmp_path, "1.0.0")
        scripts_dir = root / "scripts"
        scripts_dir.mkdir()
        _write(
            scripts_dir / "check_version_consistency.py",
            textwrap.dedent("""\
                def collect_versions(root):
                    return {"version.py": "1.0.0"}

                def check_consistency(versions):
                    return True, "All match: 1.0.0"
            """),
        )
        _write(root / "CHANGELOG.md", "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n- Feat\n")

        with patch("scripts.release_checklist.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            results = run_checklist(root, check_docker=True, check_sdks=True)

        # Should have 8 checks now
        assert len(results) == 8
        names = [r["name"] for r in results]
        assert "Docker build" in names
        assert "SDK packages" in names


# ---------------------------------------------------------------------------
# Output formatting tests
# ---------------------------------------------------------------------------


class TestOutputFormatting:
    def test_text_table_all_pass(self):
        results = [
            {"name": "Test A", "passed": True, "detail": "OK"},
            {"name": "Test B", "passed": True, "detail": "Good"},
        ]
        output = format_text_table(results)
        assert "READY" in output
        assert "[PASS]" in output
        assert "2/2" in output

    def test_text_table_some_fail(self):
        results = [
            {"name": "Test A", "passed": True, "detail": "OK"},
            {"name": "Test B", "passed": False, "detail": "Bad"},
        ]
        output = format_text_table(results)
        assert "NOT READY" in output
        assert "[FAIL]" in output
        assert "1/2" in output

    def test_markdown_report(self):
        results = [
            {"name": "Test A", "passed": True, "detail": "OK"},
            {"name": "Test B", "passed": False, "detail": "Bad"},
        ]
        output = format_markdown_report(results)
        assert "# Pre-Release Checklist Report" in output
        assert "NOT READY" in output
        assert "| Test A | PASS |" in output
        assert "| Test B | FAIL |" in output
