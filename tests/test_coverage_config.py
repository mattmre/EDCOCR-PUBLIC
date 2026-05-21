"""Tests for coverage reporting configuration (CI-07).

Validates that pytest-cov is configured correctly across the CI workflow,
requirements, and .coveragerc configuration file.
"""

import configparser
import os

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCoveragerc:
    """Tests for .coveragerc configuration file."""

    def test_coveragerc_exists(self):
        """Verify .coveragerc file exists at project root."""
        coveragerc_path = os.path.join(ROOT_DIR, ".coveragerc")
        assert os.path.isfile(coveragerc_path), ".coveragerc file should exist at project root"

    def test_coveragerc_is_parseable(self):
        """Verify .coveragerc is a valid INI file that configparser can read."""
        coveragerc_path = os.path.join(ROOT_DIR, ".coveragerc")
        config = configparser.ConfigParser()
        parsed = config.read(coveragerc_path)
        assert len(parsed) == 1, ".coveragerc should be parseable by configparser"

    def test_coveragerc_has_run_section(self):
        """Verify .coveragerc has a [run] section with source config."""
        coveragerc_path = os.path.join(ROOT_DIR, ".coveragerc")
        config = configparser.ConfigParser()
        config.read(coveragerc_path)
        assert config.has_section("run"), ".coveragerc must have a [run] section"
        assert config.has_option("run", "source"), "[run] section must have 'source' option"

    def test_coveragerc_has_report_section(self):
        """Verify .coveragerc has a [report] section."""
        coveragerc_path = os.path.join(ROOT_DIR, ".coveragerc")
        config = configparser.ConfigParser()
        config.read(coveragerc_path)
        assert config.has_section("report"), ".coveragerc must have a [report] section"

    def test_coveragerc_omits_test_directories(self):
        """Verify .coveragerc omits tests, legacy, and docs from coverage."""
        coveragerc_path = os.path.join(ROOT_DIR, ".coveragerc")
        config = configparser.ConfigParser()
        config.read(coveragerc_path)
        omit_value = config.get("run", "omit", fallback="")
        assert "tests/*" in omit_value, "tests/* must be omitted from coverage"
        assert "legacy/*" in omit_value, "legacy/* must be omitted from coverage"
        assert "docs/*" in omit_value, "docs/* must be omitted from coverage"

    def test_coveragerc_fail_under_is_zero(self):
        """Verify fail_under is 0 (informational, not gating)."""
        coveragerc_path = os.path.join(ROOT_DIR, ".coveragerc")
        config = configparser.ConfigParser()
        config.read(coveragerc_path)
        fail_under = config.get("report", "fail_under", fallback=None)
        assert fail_under == "0", "fail_under must be 0 for initial rollout"

    def test_coveragerc_excludes_pragma_no_cover(self):
        """Verify pragma: no cover is in the exclude_lines list."""
        coveragerc_path = os.path.join(ROOT_DIR, ".coveragerc")
        config = configparser.ConfigParser()
        config.read(coveragerc_path)
        exclude_lines = config.get("report", "exclude_lines", fallback="")
        assert "pragma: no cover" in exclude_lines, "exclude_lines must include 'pragma: no cover'"


class TestRequirements:
    """Tests for pytest-cov in requirements.txt."""

    def test_pytest_cov_in_requirements(self):
        """Verify pytest-cov is listed in requirements.txt."""
        req_path = os.path.join(ROOT_DIR, "requirements.txt")
        with open(req_path, encoding="utf-8") as f:
            content = f.read()
        assert "pytest-cov" in content, "pytest-cov must be present in requirements.txt"


class TestCIWorkflow:
    """Tests for coverage flags in CI workflow."""

    @pytest.fixture()
    def ci_content(self):
        """Read CI workflow content."""
        ci_path = os.path.join(ROOT_DIR, ".github", "workflows", "ci.yml")
        with open(ci_path, encoding="utf-8") as f:
            return f.read()

    def test_root_tests_have_cov_flag(self, ci_content):
        """Verify root test step includes --cov flag."""
        assert "--cov=" in ci_content, "CI workflow must use --cov flag for test coverage"

    def test_root_tests_have_xml_report(self, ci_content):
        """Verify root test step generates XML coverage report."""
        assert "--cov-report=xml:" in ci_content, "CI workflow must generate XML coverage report"

    def test_root_tests_have_term_missing(self, ci_content):
        """Verify root test step includes terminal missing-lines report."""
        assert "--cov-report=term-missing" in ci_content, (
            "CI workflow must include term-missing coverage report"
        )

    def test_coverage_artifact_upload_exists(self, ci_content):
        """Verify coverage artifact upload step exists in CI."""
        assert "Upload coverage report" in ci_content, (
            "CI workflow must have coverage artifact upload step"
        )

    def test_coordinator_tests_have_cov_flag(self, ci_content):
        """Verify coordinator test step includes --cov flag."""
        # Find the coordinator test line specifically
        assert "--cov=coordinator" in ci_content, (
            "Coordinator tests must use --cov=coordinator flag"
        )

    def test_coordinator_coverage_xml_report(self, ci_content):
        """Verify coordinator test step generates its own coverage XML."""
        assert "coordinator-coverage.xml" in ci_content, (
            "Coordinator tests must generate coordinator-coverage.xml"
        )

    def test_coordinator_coverage_upload_exists(self, ci_content):
        """Verify coordinator coverage artifact upload step exists."""
        assert "Upload coordinator coverage report" in ci_content, (
            "CI workflow must have coordinator coverage artifact upload step"
        )

    def test_pytest_cov_installed_in_root_job(self, ci_content):
        """Verify pytest-cov is installed in root-lint-and-tests job."""
        assert "pytest-cov" in ci_content, "pytest-cov must be installed in CI"

    def test_coverage_retention_days(self, ci_content):
        """Verify coverage artifacts have retention-days configured."""
        assert "retention-days:" in ci_content, (
            "Coverage artifact upload must have retention-days configured"
        )
