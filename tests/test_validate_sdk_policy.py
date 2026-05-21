"""Tests for scripts/validate_sdk_policy.py — SDK versioning policy validation."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from validate_sdk_policy import (
    CheckResult,
    ValidationReport,
    check_api_stability,
    check_ci_publish_workflow,
    check_deprecation_markers,
    check_policy_document,
    check_python_sdk_structure,
    check_typescript_sdk_structure,
    check_version_alignment,
    format_json,
    format_markdown,
    format_text,
    main,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    """Create a minimal project tree that passes all checks."""
    # version.py
    (tmp_path / "version.py").write_text('__version__ = "1.2.0"\n', encoding="utf-8")

    # Python SDK
    sdk_py = tmp_path / "sdk" / "python"
    sdk_py.mkdir(parents=True)
    (sdk_py / "pyproject.toml").write_text(
        textwrap.dedent("""\
            [project]
            name = "edcocr-sdk"
            version = "1.2.0"
            license = "MIT"
            requires-python = ">=3.10"
        """),
        encoding="utf-8",
    )
    pkg = sdk_py / "src" / "edcocr_sdk"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "1.2.0"\n', encoding="utf-8")
    (pkg / "client.py").write_text('SDK_VERSION = "1.2.0"\n', encoding="utf-8")
    (pkg / "py.typed").write_text("", encoding="utf-8")

    # TypeScript SDK
    sdk_ts = tmp_path / "sdk" / "typescript"
    sdk_ts.mkdir(parents=True)
    (sdk_ts / "package.json").write_text(
        json.dumps({
            "name": "@edcocr/sdk",
            "version": "1.2.0",
            "main": "dist/index.js",
            "types": "dist/index.d.ts",
            "engines": {"node": ">=18.0.0"},
        }),
        encoding="utf-8",
    )
    src_ts = sdk_ts / "src"
    src_ts.mkdir()
    (src_ts / "client.ts").write_text("export const SDK_VERSION = '1.2.0';\n", encoding="utf-8")

    # Helm chart
    helm = tmp_path / "helm" / "ocr-local"
    helm.mkdir(parents=True)
    (helm / "Chart.yaml").write_text('appVersion: "1.2.0"\n', encoding="utf-8")

    # api/versioning.py
    api = tmp_path / "api"
    api.mkdir()
    (api / "versioning.py").write_text(
        textwrap.dedent("""\
            class StabilityTier:
                STABLE = "stable"
            API_SURFACE = ()
            def check_backward_compatibility(): pass
        """),
        encoding="utf-8",
    )

    # Policy document
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SDK-VERSIONING-POLICY.md").write_text(
        textwrap.dedent("""\
            # SDK Versioning Policy
            ## 1. Versioning Strategy
            ## 2. Compatibility Matrix
            ## 3. Deprecation Lifecycle
            ## 4. Breaking Change Policy
            ## 5. Release Process
            ## 6. Support Lifecycle
        """),
        encoding="utf-8",
    )

    # CI workflow
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "sdk-publish.yml").write_text(
        textwrap.dedent("""\
            on:
              push:
                tags:
                  - 'sdk-python-v*'
                  - 'sdk-ts-v*'
            jobs:
              version-check:
                run: python scripts/check_version_consistency.py
              publish-pypi:
                run: twine upload
              publish-npm:
                run: npm publish
        """),
        encoding="utf-8",
    )

    return tmp_path


# ---------------------------------------------------------------------------
# Check A: Version alignment
# ---------------------------------------------------------------------------


class TestVersionAlignment:
    def test_all_versions_match(self, fake_project: Path) -> None:
        result = check_version_alignment(fake_project)
        assert result.passed
        assert "1.2.0" in result.message

    def test_version_py_missing(self, tmp_path: Path) -> None:
        result = check_version_alignment(tmp_path)
        assert not result.passed
        assert "not found" in result.message

    def test_sdk_version_mismatch(self, fake_project: Path) -> None:
        # Introduce mismatch in Python SDK __init__.py
        init = fake_project / "sdk" / "python" / "src" / "edcocr_sdk" / "__init__.py"
        init.write_text('__version__ = "9.9.9"\n', encoding="utf-8")
        result = check_version_alignment(fake_project)
        assert not result.passed
        assert "mismatch" in result.message.lower()

    def test_missing_source_file(self, fake_project: Path) -> None:
        # Remove TypeScript package.json
        (fake_project / "sdk" / "typescript" / "package.json").unlink()
        result = check_version_alignment(fake_project)
        assert not result.passed
        assert "not found" in result.message.lower()

    def test_unparseable_version_py(self, fake_project: Path) -> None:
        (fake_project / "version.py").write_text("# no version here\n", encoding="utf-8")
        result = check_version_alignment(fake_project)
        assert not result.passed
        assert "parse" in result.message.lower()

    def test_version_py_shim_uses_canonical_package_version(self, fake_project: Path) -> None:
        canonical = fake_project / "ocr_local" / "config"
        canonical.mkdir(parents=True)
        (canonical / "version.py").write_text('__version__ = "1.2.0"\n', encoding="utf-8")
        (fake_project / "version.py").write_text(
            'import importlib\nimport sys\nsys.modules[__name__] = importlib.import_module("ocr_local.config.version")\n',
            encoding="utf-8",
        )

        result = check_version_alignment(fake_project)
        assert result.passed
        assert "1.2.0" in result.message


# ---------------------------------------------------------------------------
# Check B: Python SDK structure
# ---------------------------------------------------------------------------


class TestPythonSdkStructure:
    def test_valid_structure(self, fake_project: Path) -> None:
        result = check_python_sdk_structure(fake_project)
        assert result.passed

    def test_missing_pyproject(self, tmp_path: Path) -> None:
        result = check_python_sdk_structure(tmp_path)
        assert not result.passed
        assert "not found" in result.message

    def test_missing_required_field(self, fake_project: Path) -> None:
        pyproject = fake_project / "sdk" / "python" / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
                [project]
                name = "edcocr-sdk"
                version = "1.2.0"
            """),
            encoding="utf-8",
        )
        result = check_python_sdk_structure(fake_project)
        assert not result.passed
        # Should flag missing license and requires-python
        assert any("license" in d.lower() or "requires-python" in d.lower() for d in result.details)

    def test_missing_py_typed(self, fake_project: Path) -> None:
        (fake_project / "sdk" / "python" / "src" / "edcocr_sdk" / "py.typed").unlink()
        result = check_python_sdk_structure(fake_project)
        assert not result.passed
        assert any("py.typed" in d for d in result.details)


# ---------------------------------------------------------------------------
# Check C: TypeScript SDK structure
# ---------------------------------------------------------------------------


class TestTypescriptSdkStructure:
    def test_valid_structure(self, fake_project: Path) -> None:
        result = check_typescript_sdk_structure(fake_project)
        assert result.passed

    def test_missing_package_json(self, tmp_path: Path) -> None:
        result = check_typescript_sdk_structure(tmp_path)
        assert not result.passed
        assert "not found" in result.message

    def test_missing_required_field(self, fake_project: Path) -> None:
        pkg = fake_project / "sdk" / "typescript" / "package.json"
        pkg.write_text(
            json.dumps({"name": "@edcocr/sdk", "version": "1.2.0"}),
            encoding="utf-8",
        )
        result = check_typescript_sdk_structure(fake_project)
        assert not result.passed
        assert any("main" in d.lower() or "types" in d.lower() for d in result.details)

    def test_missing_engines_node(self, fake_project: Path) -> None:
        pkg = fake_project / "sdk" / "typescript" / "package.json"
        pkg.write_text(
            json.dumps({
                "name": "@edcocr/sdk",
                "version": "1.2.0",
                "main": "dist/index.js",
                "types": "dist/index.d.ts",
            }),
            encoding="utf-8",
        )
        result = check_typescript_sdk_structure(fake_project)
        assert not result.passed
        assert any("engines" in d.lower() for d in result.details)

    def test_invalid_json(self, fake_project: Path) -> None:
        pkg = fake_project / "sdk" / "typescript" / "package.json"
        pkg.write_text("{ invalid json }", encoding="utf-8")
        result = check_typescript_sdk_structure(fake_project)
        assert not result.passed
        assert "invalid json" in result.message.lower()


# ---------------------------------------------------------------------------
# Check D: API stability
# ---------------------------------------------------------------------------


class TestApiStability:
    def test_valid_versioning(self, fake_project: Path) -> None:
        result = check_api_stability(fake_project)
        assert result.passed

    def test_missing_versioning_file(self, tmp_path: Path) -> None:
        result = check_api_stability(tmp_path)
        assert not result.passed
        assert "not found" in result.message

    def test_missing_api_surface(self, fake_project: Path) -> None:
        (fake_project / "api" / "versioning.py").write_text(
            "class StabilityTier: pass\ndef check_backward_compatibility(): pass\n",
            encoding="utf-8",
        )
        result = check_api_stability(fake_project)
        assert not result.passed
        assert any("API_SURFACE" in d for d in result.details)


# ---------------------------------------------------------------------------
# Check E: Deprecation markers
# ---------------------------------------------------------------------------


class TestDeprecationMarkers:
    def test_no_deprecated_endpoints(self, fake_project: Path) -> None:
        result = check_deprecation_markers(fake_project)
        assert result.passed
        assert "no deprecated" in result.message.lower()

    def test_deprecated_without_warnings(self, fake_project: Path) -> None:
        (fake_project / "api" / "versioning.py").write_text(
            textwrap.dedent("""\
                class StabilityTier:
                    STABLE = "stable"
                API_SURFACE = ()
                deprecated=True
                def check_backward_compatibility(): pass
            """),
            encoding="utf-8",
        )
        result = check_deprecation_markers(fake_project)
        assert not result.passed

    def test_deprecated_with_python_warnings(self, fake_project: Path) -> None:
        (fake_project / "api" / "versioning.py").write_text(
            "deprecated=True\nAPI_SURFACE=()\n",
            encoding="utf-8",
        )
        (fake_project / "sdk" / "python" / "src" / "edcocr_sdk" / "client.py").write_text(
            'SDK_VERSION = "1.2.0"\nimport warnings\nwarnings.warn("deprecated")\n',
            encoding="utf-8",
        )
        (fake_project / "sdk" / "typescript" / "src" / "client.ts").write_text(
            "export const SDK_VERSION = '1.2.0';\n/** @deprecated */\nconsole.warn('deprecated');\n",
            encoding="utf-8",
        )
        result = check_deprecation_markers(fake_project)
        assert result.passed

    def test_missing_versioning_skips(self, tmp_path: Path) -> None:
        result = check_deprecation_markers(tmp_path)
        assert result.passed
        assert "skipping" in result.message.lower()


# ---------------------------------------------------------------------------
# Check F: Policy document
# ---------------------------------------------------------------------------


class TestPolicyDocument:
    def test_valid_document(self, fake_project: Path) -> None:
        result = check_policy_document(fake_project)
        assert result.passed

    def test_missing_document(self, tmp_path: Path) -> None:
        result = check_policy_document(tmp_path)
        assert not result.passed
        assert "not found" in result.message

    def test_empty_document(self, fake_project: Path) -> None:
        (fake_project / "docs" / "SDK-VERSIONING-POLICY.md").write_text("", encoding="utf-8")
        result = check_policy_document(fake_project)
        assert not result.passed
        assert "empty" in result.message

    def test_missing_sections(self, fake_project: Path) -> None:
        (fake_project / "docs" / "SDK-VERSIONING-POLICY.md").write_text(
            "# SDK Versioning Policy\n## Versioning Strategy\n",
            encoding="utf-8",
        )
        result = check_policy_document(fake_project)
        assert not result.passed
        assert "section" in result.message.lower()


# ---------------------------------------------------------------------------
# Check G: CI publish workflow
# ---------------------------------------------------------------------------


class TestCiPublishWorkflow:
    def test_valid_workflow(self, fake_project: Path) -> None:
        result = check_ci_publish_workflow(fake_project)
        assert result.passed

    def test_missing_workflow(self, tmp_path: Path) -> None:
        result = check_ci_publish_workflow(tmp_path)
        assert not result.passed
        assert "not found" in result.message

    def test_missing_python_tag(self, fake_project: Path) -> None:
        wf = fake_project / ".github" / "workflows" / "sdk-publish.yml"
        wf.write_text(
            "tags:\n  - 'sdk-ts-v*'\nnpm publish\ncheck_version_consistency\n",
            encoding="utf-8",
        )
        result = check_ci_publish_workflow(fake_project)
        assert not result.passed
        assert any("python" in d.lower() for d in result.details)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


class TestOutputFormatters:
    def test_format_text(self, fake_project: Path) -> None:
        report = run_all_checks(fake_project)
        text = format_text(report)
        assert "SDK Policy Validation Report" in text
        assert "PASS" in text

    def test_format_json(self, fake_project: Path) -> None:
        report = run_all_checks(fake_project)
        output = format_json(report)
        data = json.loads(output)
        assert "passed" in data
        assert "checks" in data
        assert "summary" in data
        assert data["summary"]["total"] == 7

    def test_format_markdown(self, fake_project: Path) -> None:
        report = run_all_checks(fake_project)
        md = format_markdown(report)
        assert "# SDK Policy Validation Report" in md
        assert "| Check |" in md


# ---------------------------------------------------------------------------
# Run all checks / CLI
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    def test_all_pass(self, fake_project: Path) -> None:
        report = run_all_checks(fake_project)
        assert report.passed
        assert len(report.checks) == 7
        assert all(c.passed for c in report.checks)

    def test_failure_propagates(self, fake_project: Path) -> None:
        # Break version alignment
        (fake_project / "version.py").write_text('__version__ = "0.0.0"\n', encoding="utf-8")
        report = run_all_checks(fake_project)
        assert not report.passed
        version_check = next(c for c in report.checks if c.name == "version_alignment")
        assert not version_check.passed


class TestMainCli:
    def test_exit_0_on_pass(self, fake_project: Path) -> None:
        code = main(["--project-root", str(fake_project)])
        assert code == 0

    def test_exit_1_on_fail(self, fake_project: Path) -> None:
        (fake_project / "version.py").write_text('__version__ = "0.0.0"\n', encoding="utf-8")
        code = main(["--project-root", str(fake_project)])
        assert code == 1

    def test_json_output(self, fake_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["--project-root", str(fake_project), "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["passed"] is True

    def test_report_file(self, fake_project: Path, tmp_path: Path) -> None:
        report_path = tmp_path / "report.md"
        code = main(["--project-root", str(fake_project), "--report", str(report_path)])
        assert code == 0
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "# SDK Policy Validation Report" in content

    def test_exit_2_on_bad_root(self, tmp_path: Path) -> None:
        bad = tmp_path / "nonexistent"
        code = main(["--project-root", str(bad)])
        assert code == 2


class TestValidationReport:
    def test_add_passing_check(self) -> None:
        report = ValidationReport()
        report.add(CheckResult(name="test", passed=True, message="ok"))
        assert report.passed
        assert len(report.checks) == 1

    def test_add_failing_check_sets_overall_fail(self) -> None:
        report = ValidationReport()
        report.add(CheckResult(name="test", passed=False, message="bad"))
        assert not report.passed
