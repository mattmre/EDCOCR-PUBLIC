"""Tests for scripts/verify_release_state.py."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.verify_release_state import (
    DOCS_CANON_SPINE,
    check_changelog,
    check_ci_workflows,
    check_docs_canon,
    check_helm_chart,
    check_version_consistency,
    extract_changelog_version,
    extract_openapi_version,
    extract_otel_config_version,
    extract_tracing_version,
    extract_ts_sdk_useragent_version,
    generate_markdown_report,
    generate_text_report,
    read_canonical_version,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _setup_minimal_root(tmp_path: Path, version: str = "1.2.0") -> Path:
    """Create a minimal project root with version.py."""
    root = tmp_path / "project"
    root.mkdir()
    _write(root / "version.py", f'__version__ = "{version}"\n')
    return root


# ---------------------------------------------------------------------------
# A) Version extraction tests
# ---------------------------------------------------------------------------


class TestVersionExtraction:
    """Test individual version extraction functions."""

    def test_read_canonical_version(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "2.0.0")
        assert read_canonical_version(root) == "2.0.0"

    def test_read_canonical_version_forwarding_shim(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        _write(
            root / "version.py",
            """\
            import importlib as _importlib
            import sys as _sys

            _sys.modules[__name__] = _importlib.import_module("ocr_local.config.version")
            """,
        )
        _write(root / "ocr_local" / "config" / "version.py", '__version__ = "2.0.0"\n')

        assert read_canonical_version(root) == "2.0.0"

    def test_read_canonical_version_missing(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        with pytest.raises(FileNotFoundError):
            read_canonical_version(root)

    def test_read_canonical_version_no_match(self, tmp_path):
        root = _setup_minimal_root(tmp_path)
        _write(root / "version.py", "# no version here\n")
        with pytest.raises(ValueError, match="No __version__"):
            read_canonical_version(root)

    def test_extract_openapi_version(self, tmp_path):
        f = tmp_path / "openapi.json"
        data = {"openapi": "3.1.0", "info": {"title": "Test", "version": "1.5.0"}}
        f.write_text(json.dumps(data), encoding="utf-8")
        assert extract_openapi_version(f) == "1.5.0"

    def test_extract_openapi_version_missing_field(self, tmp_path):
        f = tmp_path / "openapi.json"
        f.write_text('{"info": {}}', encoding="utf-8")
        with pytest.raises(ValueError, match="No info.version"):
            extract_openapi_version(f)

    def test_extract_tracing_version(self, tmp_path):
        f = tmp_path / "tracing.py"
        _write(f, '_SERVICE_VERSION_DEFAULT = "3.1.4"\n')
        assert extract_tracing_version(f) == "3.1.4"

    def test_extract_tracing_version_no_match(self, tmp_path):
        f = tmp_path / "tracing.py"
        _write(f, "# nothing here\n")
        with pytest.raises(ValueError, match="No _SERVICE_VERSION_DEFAULT"):
            extract_tracing_version(f)

    def test_extract_otel_config_version(self, tmp_path):
        f = tmp_path / "otel.yaml"
        content = """\
        resource:
          attributes:
            - key: service.version
              value: "1.2.0"
              action: upsert
        """
        _write(f, content)
        assert extract_otel_config_version(f) == "1.2.0"

    def test_extract_otel_config_version_unquoted(self, tmp_path):
        f = tmp_path / "otel.yaml"
        content = """\
        resource:
          attributes:
            - key: service.version
              value: 2.0.0
              action: upsert
        """
        _write(f, content)
        assert extract_otel_config_version(f) == "2.0.0"

    def test_extract_otel_config_version_no_match(self, tmp_path):
        f = tmp_path / "otel.yaml"
        _write(f, "receivers:\n  otlp:\n")
        with pytest.raises(ValueError, match="No service.version"):
            extract_otel_config_version(f)

    def test_extract_ts_sdk_useragent_version(self, tmp_path):
        f = tmp_path / "client.ts"
        _write(f, "'User-Agent': 'ocr-local-typescript-sdk/1.2.0',\n")
        assert extract_ts_sdk_useragent_version(f) == "1.2.0"

    def test_extract_ts_sdk_useragent_version_no_match(self, tmp_path):
        f = tmp_path / "client.ts"
        _write(f, "const x = 1;\n")
        with pytest.raises(ValueError, match="No User-Agent version"):
            extract_ts_sdk_useragent_version(f)

    def test_extract_changelog_version(self, tmp_path):
        f = tmp_path / "CHANGELOG.md"
        _write(f, "# Changelog\n\n## [1.2.0] - 2026-03-26\n\n### Added\n- stuff\n")
        assert extract_changelog_version(f) == "1.2.0"

    def test_extract_changelog_version_skips_unreleased(self, tmp_path):
        f = tmp_path / "CHANGELOG.md"
        _write(
            f,
            "# Changelog\n\n"
            "## [Unreleased]\n\n"
            "- pending note\n\n"
            "## [1.2.1] - 2026-03-29\n\n"
            "### Fixed\n- release-state sync\n",
        )
        assert extract_changelog_version(f) == "1.2.1"

    def test_extract_changelog_version_no_match(self, tmp_path):
        f = tmp_path / "CHANGELOG.md"
        _write(f, "# Changelog\n\nNothing here\n")
        with pytest.raises(ValueError, match="No numbered version section header"):
            extract_changelog_version(f)


# ---------------------------------------------------------------------------
# B) Docs canon validation
# ---------------------------------------------------------------------------


class TestDocsCanon:
    """Test docs canon validation."""

    def test_all_docs_present(self, tmp_path):
        root = _setup_minimal_root(tmp_path)
        for rel in DOCS_CANON_SPINE:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("Some content", encoding="utf-8")

        result = check_docs_canon(root)
        assert result["passed"] is True
        assert result["present"] == len(DOCS_CANON_SPINE)
        assert result["missing"] == []
        assert result["empty"] == []

    def test_missing_docs(self, tmp_path):
        root = _setup_minimal_root(tmp_path)
        # Create only first 5 docs
        for rel in DOCS_CANON_SPINE[:5]:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("Content", encoding="utf-8")

        result = check_docs_canon(root)
        assert result["passed"] is False
        assert len(result["missing"]) == len(DOCS_CANON_SPINE) - 5

    def test_empty_doc(self, tmp_path):
        root = _setup_minimal_root(tmp_path)
        for rel in DOCS_CANON_SPINE:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("Content", encoding="utf-8")

        # Make one empty
        (root / DOCS_CANON_SPINE[0]).write_text("", encoding="utf-8")

        result = check_docs_canon(root)
        assert result["passed"] is False
        assert len(result["empty"]) == 1
        assert DOCS_CANON_SPINE[0] in result["empty"]


# ---------------------------------------------------------------------------
# C) Helm chart consistency
# ---------------------------------------------------------------------------


class TestHelmChart:
    """Test Helm chart consistency check."""

    def test_helm_chart_matches(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "1.2.0")
        chart_dir = root / "helm" / "ocr-local"
        chart_dir.mkdir(parents=True)
        _write(
            chart_dir / "Chart.yaml",
            'apiVersion: v2\nversion: 0.4.0\nappVersion: "1.2.0"\n',
        )

        result = check_helm_chart(root)
        assert result["passed"] is True
        assert result["app_version"] == "1.2.0"
        assert result["chart_version"] == "0.4.0"
        assert result["app_version_matches"] is True

    def test_helm_chart_mismatch(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "1.2.0")
        chart_dir = root / "helm" / "ocr-local"
        chart_dir.mkdir(parents=True)
        _write(
            chart_dir / "Chart.yaml",
            'apiVersion: v2\nversion: 0.3.0\nappVersion: "1.1.0"\n',
        )

        result = check_helm_chart(root)
        assert result["passed"] is False
        assert result["app_version_matches"] is False

    def test_helm_chart_missing(self, tmp_path):
        root = _setup_minimal_root(tmp_path)
        result = check_helm_chart(root)
        assert result["passed"] is False
        assert "not found" in result.get("error", "")


# ---------------------------------------------------------------------------
# D) CHANGELOG validation
# ---------------------------------------------------------------------------


class TestChangelog:
    """Test CHANGELOG validation."""

    def test_valid_changelog(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "1.2.0")
        _write(
            root / "CHANGELOG.md",
            "# Changelog\n\n## [1.2.0] - 2026-03-26\n\n### Added\n- Feature X\n",
        )

        result = check_changelog(root)
        assert result["passed"] is True
        assert result["has_current_version_entry"] is True
        assert result["entry_has_content"] is True

    def test_changelog_missing_version(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "2.0.0")
        _write(
            root / "CHANGELOG.md",
            "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n- Old stuff\n",
        )

        result = check_changelog(root)
        assert result["passed"] is False
        assert result["has_current_version_entry"] is False

    def test_changelog_empty_entry(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "1.2.0")
        _write(
            root / "CHANGELOG.md",
            "# Changelog\n\n## [1.2.0] - 2026-03-26\n\n## [1.1.0] - 2026-03-25\n\n### Added\n",
        )

        result = check_changelog(root)
        assert result["passed"] is False
        assert result["has_current_version_entry"] is True
        assert result["entry_has_content"] is False

    def test_changelog_not_found(self, tmp_path):
        root = _setup_minimal_root(tmp_path)
        result = check_changelog(root)
        assert result["passed"] is False
        assert result["changelog_exists"] is False


# ---------------------------------------------------------------------------
# E) CI workflow integrity
# ---------------------------------------------------------------------------


class TestCIWorkflows:
    """Test CI workflow integrity check."""

    def test_all_workflows_present(self, tmp_path):
        root = _setup_minimal_root(tmp_path)
        ci_dir = root / ".github" / "workflows"
        ci_dir.mkdir(parents=True)

        ci_content = 'python-version: ["3.10", "3.11"]\n'
        for rel_path in [
            "ci.yml",
            "release.yml",
            "docker-publish.yml",
            "sdk-publish.yml",
        ]:
            (ci_dir / rel_path).write_text(ci_content, encoding="utf-8")

        result = check_ci_workflows(root)
        assert result["passed"] is True
        assert len(result["missing"]) == 0

    def test_missing_workflow(self, tmp_path):
        root = _setup_minimal_root(tmp_path)
        ci_dir = root / ".github" / "workflows"
        ci_dir.mkdir(parents=True)

        # Only create ci.yml
        (ci_dir / "ci.yml").write_text(
            'python-version: ["3.10", "3.11"]\n', encoding="utf-8"
        )

        result = check_ci_workflows(root)
        assert result["passed"] is False
        assert len(result["missing"]) == 3

    def test_missing_python_version(self, tmp_path):
        root = _setup_minimal_root(tmp_path)
        ci_dir = root / ".github" / "workflows"
        ci_dir.mkdir(parents=True)

        # ci.yml with only 3.11
        (ci_dir / "ci.yml").write_text(
            'python-version: ["3.11"]\n', encoding="utf-8"
        )
        for name in ["release.yml", "docker-publish.yml", "sdk-publish.yml"]:
            (ci_dir / name).write_text("name: test\n", encoding="utf-8")

        result = check_ci_workflows(root)
        assert result["passed"] is False
        assert any("3.10" in issue for issue in result["python_version_issues"])


# ---------------------------------------------------------------------------
# Version consistency (integration of base + extended)
# ---------------------------------------------------------------------------


class TestVersionConsistency:
    """Test extended version consistency check."""

    def test_all_match(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "1.2.0")

        # Create extended sources
        _write(
            root / "docs" / "openapi.json",
            json.dumps({"info": {"version": "1.2.0"}}),
        )
        _write(root / "api" / "tracing.py", '_SERVICE_VERSION_DEFAULT = "1.2.0"\n')
        _write(
            root / "otel" / "otel-collector-config.yaml",
            "resource:\n  attributes:\n    - key: service.version\n      value: \"1.2.0\"\n",
        )
        _write(
            root / "sdk" / "typescript" / "ocr_client.ts",
            "'User-Agent': 'ocr-local-typescript-sdk/1.2.0',\n",
        )
        _write(
            root / "CHANGELOG.md",
            "# Changelog\n\n## [1.2.0] - 2026-03-26\n\n### Added\n- X\n",
        )

        # Mock the base collector to avoid needing all SDK files
        def mock_collect(root=None):
            return {"version.py": "1.2.0"}

        with patch.dict(
            "sys.modules",
            {"check_version_consistency": type("m", (), {"collect_versions": mock_collect})()},
        ):
            result = check_version_consistency(root)

        assert result["canonical"] == "1.2.0"
        # The extended sources should match
        for label in [
            "docs/openapi.json",
            "api/tracing.py",
            "otel/otel-collector-config.yaml",
            "sdk/typescript/ocr_client.ts",
            "CHANGELOG.md",
        ]:
            assert label in result["sources"]
            assert result["sources"][label] == "1.2.0"

    def test_mismatch_detected(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "1.2.0")

        # Create one mismatched source
        _write(
            root / "docs" / "openapi.json",
            json.dumps({"info": {"version": "1.1.0"}}),
        )

        def mock_collect(root=None):
            return {"version.py": "1.2.0"}

        with patch.dict(
            "sys.modules",
            {"check_version_consistency": type("m", (), {"collect_versions": mock_collect})()},
        ):
            result = check_version_consistency(root)

        assert result["passed"] is False
        assert "docs/openapi.json" in result["mismatches"]
        assert result["mismatches"]["docs/openapi.json"] == "1.1.0"

    def test_missing_extended_source(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "1.2.0")

        def mock_collect(root=None):
            return {"version.py": "1.2.0"}

        with patch.dict(
            "sys.modules",
            {"check_version_consistency": type("m", (), {"collect_versions": mock_collect})()},
        ):
            result = check_version_consistency(root)

        # All extended sources are missing, so they should be FILE_NOT_FOUND
        for label in [
            "docs/openapi.json",
            "api/tracing.py",
            "otel/otel-collector-config.yaml",
            "sdk/typescript/ocr_client.ts",
            "CHANGELOG.md",
        ]:
            assert "FILE_NOT_FOUND" in result["sources"][label]
            assert label in result["mismatches"]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Test text and markdown report generation."""

    @pytest.fixture()
    def sample_results(self):
        return {
            "version_consistency": {
                "passed": True,
                "canonical": "1.2.0",
                "sources": {"version.py": "1.2.0"},
                "mismatches": {},
                "errors": [],
                "total_sources": 12,
            },
            "docs_canon": {
                "passed": True,
                "total_files": 13,
                "present": 13,
                "missing": [],
                "empty": [],
            },
            "helm_chart": {
                "passed": True,
                "app_version": "1.2.0",
                "chart_version": "0.4.0",
                "app_version_matches": True,
                "chart_version_set": True,
            },
            "changelog": {
                "passed": True,
                "changelog_exists": True,
                "has_current_version_entry": True,
                "entry_has_content": True,
                "latest_version": "1.2.0",
                "canonical_version": "1.2.0",
            },
            "ci_workflows": {
                "passed": True,
                "total_expected": 4,
                "present": [
                    ".github/workflows/ci.yml",
                    ".github/workflows/release.yml",
                    ".github/workflows/docker-publish.yml",
                    ".github/workflows/sdk-publish.yml",
                ],
                "missing": [],
                "python_version_issues": [],
            },
        }

    def test_text_report_all_pass(self, sample_results):
        report = generate_text_report(sample_results)
        assert "ALL CHECKS PASSED" in report
        assert "[PASS] A)" in report
        assert "[PASS] B)" in report
        assert "[PASS] C)" in report
        assert "[PASS] D)" in report
        assert "[PASS] E)" in report

    def test_text_report_with_failures(self, sample_results):
        sample_results["docs_canon"]["passed"] = False
        sample_results["docs_canon"]["missing"] = ["docs/README.md"]
        report = generate_text_report(sample_results)
        assert "SOME CHECKS FAILED" in report
        assert "[FAIL] B)" in report
        assert "MISSING: docs/README.md" in report

    def test_markdown_report_structure(self, sample_results):
        report = generate_markdown_report(sample_results)
        assert "# Release State Verification Report" in report
        assert "| Check | Status |" in report
        assert "## A) Version Consistency" in report
        assert "## B) Docs Canon Validation" in report
        assert "## C) Helm Chart Consistency" in report
        assert "## D) CHANGELOG Validation" in report
        assert "## E) CI Workflow Integrity" in report

    def test_markdown_report_with_mismatches(self, sample_results):
        sample_results["version_consistency"]["passed"] = False
        sample_results["version_consistency"]["mismatches"] = {
            "docs/openapi.json": "1.1.0"
        }
        report = generate_markdown_report(sample_results)
        assert "FAIL" in report
        assert "`docs/openapi.json`" in report
        assert "`1.1.0`" in report


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestJSONOutput:
    """Test that results are JSON-serializable."""

    def test_results_json_serializable(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "1.2.0")

        def mock_collect(root=None):
            return {"version.py": "1.2.0"}

        with patch.dict(
            "sys.modules",
            {"check_version_consistency": type("m", (), {"collect_versions": mock_collect})()},
        ):
            results = run_all_checks(root)

        # Must not raise
        output = json.dumps(results, indent=2)
        parsed = json.loads(output)
        assert "version_consistency" in parsed
        assert "docs_canon" in parsed
        assert "helm_chart" in parsed
        assert "changelog" in parsed
        assert "ci_workflows" in parsed


# ---------------------------------------------------------------------------
# run_all_checks integration
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    """Test the top-level orchestrator."""

    def test_run_all_checks_returns_five_categories(self, tmp_path):
        root = _setup_minimal_root(tmp_path, "1.2.0")

        def mock_collect(root=None):
            return {"version.py": "1.2.0"}

        with patch.dict(
            "sys.modules",
            {"check_version_consistency": type("m", (), {"collect_versions": mock_collect})()},
        ):
            results = run_all_checks(root)

        assert set(results.keys()) == {
            "version_consistency",
            "docs_canon",
            "helm_chart",
            "changelog",
            "ci_workflows",
        }
        # Each result has a 'passed' key
        for key, val in results.items():
            assert "passed" in val, f"{key} missing 'passed' key"
