"""Tests for scripts/upgrade_readiness.py — upgrade readiness and config drift validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure scripts/ is importable before any local imports
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pytest  # noqa: E402

import upgrade_readiness as ur  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_project(tmp_path):
    """Create a minimal project skeleton for tests."""
    # version.py
    (tmp_path / "version.py").write_text(
        '__version__ = "1.2.0"\n', encoding="utf-8"
    )
    # requirements.txt
    (tmp_path / "requirements.txt").write_text(
        "paddlepaddle==2.6.2\n"
        "paddleocr==2.9.1\n"
        "numpy==1.26.4\n"
        "opencv-python-headless==4.11.0.86\n"
        "django>=5.2,<5.3\n"
        "fastapi==0.135.2\n",
        encoding="utf-8",
    )
    # CHANGELOG.md
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [1.3.0] - 2026-06-01\n\n"
        "### Added\n"
        "- New feature\n\n"
        "### Changed\n"
        "- **Breaking**: Removed legacy sync pipeline\n\n"
        "## [1.2.0] - 2026-03-26\n\n"
        "### Added\n"
        "- Noise profiling\n\n"
        "## [1.1.0] - 2026-03-25\n\n"
        "### Added\n"
        "- Security audit\n\n"
        "## [1.0.0] - 2026-03-15\n\n"
        "### Added\n"
        "- Migration Guide v0.9 to v1.0: Breaking changes\n\n",
        encoding="utf-8",
    )
    # scripts/ dir
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "migrate_nfs_to_s3.py").write_text("# migration\n", encoding="utf-8")
    return tmp_path


# ===========================================================================
# A) Version compatibility tests
# ===========================================================================


class TestParseVersion:
    def test_standard_semver(self):
        assert ur.parse_semver("1.2.3") == (1, 2, 3)

    def test_prerelease_stripped(self):
        assert ur.parse_semver("1.2.3-rc1") == (1, 2, 3)

    def test_build_metadata_stripped(self):
        assert ur.parse_semver("2.0.0+build.123") == (2, 0, 0)

    def test_two_part(self):
        assert ur.parse_semver("1.2") == (1, 2, 0)

    def test_single_part(self):
        assert ur.parse_semver("3") == (3, 0, 0)


class TestIsValidUpgradePath:
    def test_same_version_invalid(self):
        valid, _ = ur.is_valid_upgrade_path("1.2.0", "1.2.0")
        assert valid is False

    def test_downgrade_invalid(self):
        valid, reason = ur.is_valid_upgrade_path("1.2.0", "1.1.0")
        assert valid is False
        assert "Downgrade" in reason

    def test_patch_upgrade_valid(self):
        valid, _ = ur.is_valid_upgrade_path("1.2.0", "1.2.1")
        assert valid is True

    def test_minor_upgrade_valid(self):
        valid, _ = ur.is_valid_upgrade_path("1.2.0", "1.3.0")
        assert valid is True

    def test_major_upgrade_valid_with_warning(self):
        valid, reason = ur.is_valid_upgrade_path("1.2.0", "2.0.0")
        assert valid is True
        assert "Major version bump" in reason


class TestCheckVersionCompatibility:
    def test_no_target_version_skips(self):
        result = ur.check_version_compatibility("1.2.0", None)
        assert result.status == "skip"

    def test_valid_minor_upgrade_passes(self):
        result = ur.check_version_compatibility("1.2.0", "1.3.0")
        assert result.status == "pass"

    def test_downgrade_fails(self):
        result = ur.check_version_compatibility("1.2.0", "1.1.0")
        assert result.status == "fail"

    def test_major_upgrade_warns(self):
        result = ur.check_version_compatibility("1.2.0", "2.0.0")
        assert result.status == "warn"


# ===========================================================================
# B) Config drift detection tests
# ===========================================================================


class TestCaptureEnv:
    def test_filters_monitored_prefixes(self):
        env = {
            "OCR_API_KEY": "secret",
            "ENABLE_NER": "true",
            "HOME": "/home/user",
            "PATH": "/usr/bin",
            "DJANGO_SECRET_KEY": "abc",
        }
        captured = ur.capture_env(env)
        assert "OCR_API_KEY" in captured
        assert "ENABLE_NER" in captured
        assert "DJANGO_SECRET_KEY" in captured
        assert "HOME" not in captured
        assert "PATH" not in captured

    def test_exact_names_captured(self):
        env = {"DATABASE_URL": "postgres://...", "DPI": "300", "UNRELATED": "x"}
        captured = ur.capture_env(env)
        assert "DATABASE_URL" in captured
        assert "DPI" in captured
        assert "UNRELATED" not in captured


class TestDetectDrift:
    def test_no_drift(self):
        baseline = {"OCR_API_KEY": "abc", "DPI": "300"}
        current = {"OCR_API_KEY": "abc", "DPI": "300"}
        drift = ur.detect_drift(baseline, current)
        assert drift["added"] == []
        assert drift["removed"] == []
        assert drift["changed"] == []

    def test_added_detected(self):
        baseline = {"OCR_API_KEY": "abc"}
        current = {"OCR_API_KEY": "abc", "ENABLE_NER": "true"}
        drift = ur.detect_drift(baseline, current)
        assert len(drift["added"]) == 1
        assert drift["added"][0]["key"] == "ENABLE_NER"

    def test_removed_detected(self):
        baseline = {"OCR_API_KEY": "abc", "ENABLE_NER": "true"}
        current = {"OCR_API_KEY": "abc"}
        drift = ur.detect_drift(baseline, current)
        assert len(drift["removed"]) == 1
        assert drift["removed"][0]["key"] == "ENABLE_NER"

    def test_changed_detected(self):
        baseline = {"DPI": "300"}
        current = {"DPI": "450"}
        drift = ur.detect_drift(baseline, current)
        assert len(drift["changed"]) == 1
        assert drift["changed"][0]["baseline_value"] == "300"
        assert drift["changed"][0]["current_value"] == "450"


class TestBaselineSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        baseline_file = tmp_path / "baseline.json"
        env = {"OCR_API_KEY": "secret", "DPI": "300"}
        ur.save_baseline(env, baseline_file)

        loaded = ur.load_baseline(baseline_file)
        assert loaded == env

    def test_save_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "baseline.json"
        ur.save_baseline({"X": "1"}, deep_path)
        assert deep_path.exists()

    def test_saved_file_has_timestamp(self, tmp_path):
        baseline_file = tmp_path / "baseline.json"
        ur.save_baseline({"X": "1"}, baseline_file)
        data = json.loads(baseline_file.read_text(encoding="utf-8"))
        assert "timestamp" in data
        assert "variables" in data


class TestCheckConfigDrift:
    def test_no_baseline_path_skips(self):
        result = ur.check_config_drift(None, {})
        assert result.status == "skip"

    def test_missing_baseline_file_skips(self, tmp_path):
        result = ur.check_config_drift(tmp_path / "missing.json", {})
        assert result.status == "skip"

    def test_no_drift_passes(self, tmp_path):
        baseline_file = tmp_path / "baseline.json"
        env = {"OCR_API_KEY": "abc"}
        ur.save_baseline(env, baseline_file)
        result = ur.check_config_drift(baseline_file, env)
        assert result.status == "pass"

    def test_drift_detected_warns(self, tmp_path):
        baseline_file = tmp_path / "baseline.json"
        ur.save_baseline({"OCR_API_KEY": "old"}, baseline_file)
        result = ur.check_config_drift(baseline_file, {"OCR_API_KEY": "new"})
        assert result.status == "warn"
        assert "1 changed" in result.summary


# ===========================================================================
# C) Deprecated settings tests
# ===========================================================================


class TestCheckDeprecatedSettings:
    def test_no_deprecated_passes(self):
        env = {"OCR_API_KEY": "abc", "ENABLE_NER": "true"}
        result = ur.check_deprecated_settings(env)
        assert result.status == "pass"

    def test_deprecated_setting_detected(self):
        env = {"ENABLE_HPI": "true"}
        result = ur.check_deprecated_settings(env)
        assert result.status == "warn"
        assert "1 deprecated" in result.summary
        assert any("ENABLE_HPI" in d for d in result.details)

    def test_removed_setting_is_blocker(self):
        env = {"PADDLEX_MODEL": "ppocr"}
        result = ur.check_deprecated_settings(env)
        assert result.status == "fail"
        assert any("removed" in d.lower() for d in result.details)

    def test_custom_deprecated_registry(self):
        custom = {
            "OLD_VAR": {
                "replacement": "NEW_VAR",
                "since": "1.0.0",
                "removed": None,
                "guidance": "Use NEW_VAR instead.",
            }
        }
        env = {"OLD_VAR": "value"}
        result = ur.check_deprecated_settings(env, deprecated=custom)
        assert result.status == "warn"
        assert any("OLD_VAR" in d for d in result.details)


# ===========================================================================
# D) Required migrations tests
# ===========================================================================


class TestParseChangelogVersions:
    def test_parses_versions(self, tmp_project):
        sections = ur.parse_changelog_versions(tmp_project / "CHANGELOG.md")
        versions = [v for v, _ in sections]
        assert "1.3.0" in versions
        assert "1.2.0" in versions
        assert "1.0.0" in versions

    def test_missing_changelog_returns_empty(self, tmp_path):
        sections = ur.parse_changelog_versions(tmp_path / "CHANGELOG.md")
        assert sections == []


class TestCheckRequiredMigrations:
    def test_no_target_skips(self, tmp_project):
        result = ur.check_required_migrations("1.2.0", None, tmp_project)
        assert result.status == "skip"

    def test_no_changelog_skips(self, tmp_path):
        (tmp_path / "version.py").write_text('__version__="1.0.0"\n', encoding="utf-8")
        result = ur.check_required_migrations("1.0.0", "1.1.0", tmp_path)
        assert result.status == "skip"

    def test_breaking_changes_detected(self, tmp_project):
        result = ur.check_required_migrations("1.2.0", "1.3.0", tmp_project)
        assert result.status == "warn"
        assert any("Breaking" in d or "breaking" in d for d in result.details)

    def test_no_breaking_changes_passes(self, tmp_project):
        # Upgrade from 1.1.0 to 1.2.0 -- the 1.2.0 section has no breaking
        result = ur.check_required_migrations("1.1.0", "1.2.0", tmp_project)
        assert result.status == "pass"

    def test_migration_scripts_listed(self, tmp_project):
        result = ur.check_required_migrations("1.2.0", "1.3.0", tmp_project)
        assert any("migrate_nfs_to_s3" in d for d in result.details)


# ===========================================================================
# E) Dependency compatibility tests
# ===========================================================================


class TestParseRequirements:
    def test_parses_pinned_versions(self, tmp_project):
        pkgs = ur.parse_requirements(tmp_project / "requirements.txt")
        assert pkgs["paddlepaddle"] == "==2.6.2"
        assert pkgs["numpy"] == "==1.26.4"

    def test_handles_range_pins(self, tmp_project):
        pkgs = ur.parse_requirements(tmp_project / "requirements.txt")
        assert "django" in pkgs
        assert ">=" in pkgs["django"]

    def test_missing_file_returns_empty(self, tmp_path):
        pkgs = ur.parse_requirements(tmp_path / "nonexistent.txt")
        assert pkgs == {}

    def test_comments_ignored(self, tmp_path):
        (tmp_path / "req.txt").write_text(
            "# comment\nnumpy==1.26.4\n", encoding="utf-8"
        )
        pkgs = ur.parse_requirements(tmp_path / "req.txt")
        assert "numpy" in pkgs
        assert len(pkgs) == 1


class TestVersionMatchesCondition:
    def test_gte(self):
        assert ur._version_matches_condition("2.0.0", ">=2.0.0") is True
        assert ur._version_matches_condition("1.9.0", ">=2.0.0") is False

    def test_lt(self):
        assert ur._version_matches_condition("2.6.2", "<3.0.0") is True
        assert ur._version_matches_condition("3.0.0", "<3.0.0") is False

    def test_eq(self):
        assert ur._version_matches_condition("1.26.4", "==1.26.4") is True
        assert ur._version_matches_condition("1.26.5", "==1.26.4") is False

    def test_gt(self):
        assert ur._version_matches_condition("2.0.1", ">2.0.0") is True
        assert ur._version_matches_condition("2.0.0", ">2.0.0") is False

    def test_lte(self):
        assert ur._version_matches_condition("2.0.0", "<=2.0.0") is True
        assert ur._version_matches_condition("2.0.1", "<=2.0.0") is False


class TestCheckDependencyCompatibility:
    def test_current_deps_pass(self, tmp_project):
        """Current production deps (numpy 1.26, paddle 2.6, opencv 4.11) are clean."""
        result = ur.check_dependency_compatibility(tmp_project)
        assert result.status == "pass"

    def test_numpy2_paddle_conflict_detected(self, tmp_path):
        (tmp_path / "requirements.txt").write_text(
            "numpy==2.1.0\npaddlepaddle==2.6.2\n", encoding="utf-8"
        )
        result = ur.check_dependency_compatibility(tmp_path)
        assert result.status == "fail"
        assert any("numpy" in d.lower() for d in result.details)

    def test_opencv_numpy_conflict_detected(self, tmp_path):
        (tmp_path / "requirements.txt").write_text(
            "opencv-python-headless==4.12.0.88\nnumpy==1.26.4\n",
            encoding="utf-8",
        )
        result = ur.check_dependency_compatibility(tmp_path)
        assert result.status == "fail"
        assert any("opencv" in d.lower() for d in result.details)

    def test_custom_incompatibilities(self, tmp_path):
        (tmp_path / "requirements.txt").write_text(
            "foo==2.0.0\nbar==1.0.0\n", encoding="utf-8"
        )
        custom_rules = [
            {
                "package_a": "foo",
                "condition_a": ">=2.0.0",
                "package_b": "bar",
                "condition_b": "<2.0.0",
                "severity": "blocker",
                "explanation": "foo 2+ needs bar 2+",
            },
        ]
        result = ur.check_dependency_compatibility(tmp_path, custom_rules)
        assert result.status == "fail"

    def test_empty_requirements_skips(self, tmp_path):
        result = ur.check_dependency_compatibility(tmp_path)
        assert result.status == "skip"


# ===========================================================================
# Report formatting tests
# ===========================================================================


class TestFormatting:
    def _make_report(self) -> ur.ReadinessReport:
        return ur.ReadinessReport(
            version_current="1.2.0",
            version_target="1.3.0",
            timestamp="2026-03-26T00:00:00+00:00",
            checks=[
                ur.CheckResult("Version Compatibility", "pass", "Valid upgrade"),
                ur.CheckResult("Config Drift", "warn", "1 added", ["ADDED: X=1"]),
                ur.CheckResult("Deprecated Settings", "pass", "None found"),
                ur.CheckResult("Required Migrations", "skip", "Skipped"),
                ur.CheckResult("Dependency Compatibility", "pass", "Clean"),
            ],
        )

    def test_text_report_contains_version(self):
        report = self._make_report()
        text = ur.format_text_report(report)
        assert "1.2.0" in text
        assert "1.3.0" in text
        assert "UPGRADE READINESS REPORT" in text

    def test_text_report_shows_warnings(self):
        report = self._make_report()
        text = ur.format_text_report(report)
        assert "[WARN]" in text
        assert "WARNINGS" in text

    def test_json_report_parses(self):
        report = self._make_report()
        raw = ur.format_json_report(report)
        data = json.loads(raw)
        assert data["ready"] is True  # no blockers, only warnings
        assert data["exit_code"] == 2  # warnings
        assert len(data["checks"]) == 5

    def test_markdown_report_has_table(self):
        report = self._make_report()
        md = ur.format_markdown_report(report)
        assert "| Category | Status | Summary |" in md
        assert "# Upgrade Readiness Report" in md

    def test_blockers_report_not_ready(self):
        report = ur.ReadinessReport(
            version_current="1.2.0",
            version_target="1.1.0",
            timestamp="2026-03-26T00:00:00+00:00",
            checks=[
                ur.CheckResult("Version Compatibility", "fail", "Downgrade"),
            ],
        )
        text = ur.format_text_report(report)
        assert "NOT READY" in text
        assert report.exit_code == 1

    def test_all_pass_shows_ready(self):
        report = ur.ReadinessReport(
            version_current="1.2.0",
            version_target="1.3.0",
            timestamp="2026-03-26T00:00:00+00:00",
            checks=[
                ur.CheckResult("Test", "pass", "OK"),
            ],
        )
        text = ur.format_text_report(report)
        assert "READY" in text
        assert report.exit_code == 0


# ===========================================================================
# Integration: run_readiness_checks
# ===========================================================================


class TestRunReadinessChecks:
    def test_full_run_with_target(self, tmp_project):
        report = ur.run_readiness_checks(
            target_version="1.3.0",
            project_root=tmp_project,
            env_override={},
        )
        assert report.version_current == "1.2.0"
        assert report.version_target == "1.3.0"
        assert len(report.checks) == 5

    def test_full_run_no_target(self, tmp_project):
        report = ur.run_readiness_checks(
            project_root=tmp_project,
            env_override={},
        )
        # Version compat and migrations should be skipped
        version_check = report.checks[0]
        assert version_check.status == "skip"

    def test_deprecated_env_flagged(self, tmp_project):
        report = ur.run_readiness_checks(
            project_root=tmp_project,
            env_override={"ENABLE_HPI": "true"},
        )
        deprecated_check = next(
            c for c in report.checks if c.category == "Deprecated Settings"
        )
        assert deprecated_check.status == "warn"

    def test_env_file_loading(self, tmp_project):
        env_file = tmp_project / ".env"
        env_file.write_text(
            'OCR_API_KEY="test-key"\nENABLE_NER=true\n',
            encoding="utf-8",
        )
        # Saving a baseline then comparing
        baseline_file = tmp_project / "baseline.json"
        ur.save_baseline({"OCR_API_KEY": "old-key"}, baseline_file)

        report = ur.run_readiness_checks(
            env_file=str(env_file),
            baseline_path=str(baseline_file),
            compare_baseline=True,
            project_root=tmp_project,
        )
        drift_check = next(
            c for c in report.checks if c.category == "Config Drift"
        )
        # Should detect drift (changed OCR_API_KEY)
        assert drift_check.status == "warn"


# ===========================================================================
# CLI tests
# ===========================================================================


class TestCLI:
    def test_save_baseline_mode(self, tmp_path, monkeypatch):
        baseline_file = tmp_path / "bl.json"
        monkeypatch.setenv("OCR_TEST_VAR", "hello")
        exit_code = ur.main([
            "--save-baseline",
            "--baseline-path", str(baseline_file),
        ])
        assert exit_code == 0
        assert baseline_file.exists()

    def test_json_output_mode(self, tmp_project, capsys):
        ur.main([
            "--target-version", "1.3.0",
            "--json",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "checks" in data

    def test_report_file_written(self, tmp_project, tmp_path):
        report_file = tmp_path / "report.md"
        ur.main([
            "--target-version", "1.3.0",
            "--report", str(report_file),
        ])
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "# Upgrade Readiness Report" in content


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_load_env_file_with_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            'OCR_API_KEY="my-key"\n'
            "ENABLE_NER='true'\n"
            "DPI=300\n"
            "# comment\n"
            "\n"
            "INVALID_LINE_NO_EQUALS\n",
            encoding="utf-8",
        )
        result = ur.load_env_file(env_file)
        assert result["OCR_API_KEY"] == "my-key"
        assert result["ENABLE_NER"] == "true"
        assert result["DPI"] == "300"
        assert "INVALID_LINE_NO_EQUALS" not in result

    def test_read_version_missing_file(self, tmp_path):
        version = ur.read_current_version(tmp_path)
        assert version == "0.0.0"

    def test_extract_pinned_version_exact(self):
        assert ur._extract_pinned_version("==2.6.2") == "2.6.2"

    def test_extract_pinned_version_range(self):
        assert ur._extract_pinned_version(">=5.2,<5.3") == "5.2"

    def test_extract_pinned_version_none(self):
        assert ur._extract_pinned_version("") is None

    def test_invalid_json_baseline(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json!", encoding="utf-8")
        result = ur.check_config_drift(bad_file, {})
        assert result.status == "fail"
        assert "Failed to load" in result.summary
