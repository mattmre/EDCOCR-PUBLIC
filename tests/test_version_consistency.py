"""Tests for scripts/check_version_consistency.py."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import check_version_consistency as _cvc  # noqa: E402

check_consistency = _cvc.check_consistency
collect_versions = _cvc.collect_versions
read_chart_app_version = _cvc.read_chart_app_version
read_package_json_version = _cvc.read_package_json_version
read_pyproject_version = _cvc.read_pyproject_version
read_sdk_version_py = _cvc.read_sdk_version_py
read_sdk_version_ts = _cvc.read_sdk_version_ts
read_version_py = _cvc.read_version_py


# -------------------------------------------------------------------
# Unit tests for individual extractors
# -------------------------------------------------------------------


class TestReadVersionPy:
    def test_extracts_version(self, tmp_path):
        f = tmp_path / "version.py"
        f.write_text('__version__ = "2.3.4"\n')
        assert read_version_py(f) == "2.3.4"

    def test_single_quotes(self, tmp_path):
        f = tmp_path / "v.py"
        f.write_text("__version__ = '1.0.0-rc1'\n")
        assert read_version_py(f) == "1.0.0-rc1"

    def test_missing_raises(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("# no version here\n")
        with pytest.raises(ValueError, match="No __version__"):
            read_version_py(f)

    def test_forwarding_shim(self, tmp_path):
        forwarded = tmp_path / "ocr_local" / "config" / "version.py"
        forwarded.parent.mkdir(parents=True)
        forwarded.write_text('__version__ = "1.2.1"\n', encoding="utf-8")

        shim = tmp_path / "version.py"
        shim.write_text(
            textwrap.dedent(
                """\
                import importlib as _importlib
                import sys as _sys

                _sys.modules[__name__] = _importlib.import_module("ocr_local.config.version")
                """
            ),
            encoding="utf-8",
        )

        assert read_version_py(shim) == "1.2.1"

    def test_forwarding_shim_from_subdirectory(self, tmp_path):
        forwarded = tmp_path / "ocr_local" / "config" / "version.py"
        forwarded.parent.mkdir(parents=True)
        forwarded.write_text('__version__ = "1.2.1"\n', encoding="utf-8")

        shim = tmp_path / "sdk" / "python" / "src" / "edcocr_sdk" / "__init__.py"
        shim.parent.mkdir(parents=True)
        shim.write_text(
            textwrap.dedent(
                """\
                import importlib as _importlib
                import sys as _sys

                _sys.modules[__name__] = _importlib.import_module("ocr_local.config.version")
                """
            ),
            encoding="utf-8",
        )

        assert read_version_py(shim) == "1.2.1"


class TestReadPyprojectVersion:
    def test_extracts_version(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text('[project]\nname = "pkg"\nversion = "1.2.3"\n')
        assert read_pyproject_version(f) == "1.2.3"

    def test_missing_raises(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text("[project]\nname = \"pkg\"\n")
        with pytest.raises(ValueError, match="No version"):
            read_pyproject_version(f)


class TestReadPackageJsonVersion:
    def test_extracts_version(self, tmp_path):
        f = tmp_path / "package.json"
        f.write_text(json.dumps({"name": "pkg", "version": "3.0.0"}))
        assert read_package_json_version(f) == "3.0.0"

    def test_missing_raises(self, tmp_path):
        f = tmp_path / "package.json"
        f.write_text(json.dumps({"name": "pkg"}))
        with pytest.raises(ValueError, match="No version"):
            read_package_json_version(f)


class TestReadChartAppVersion:
    def test_quoted(self, tmp_path):
        f = tmp_path / "Chart.yaml"
        f.write_text('apiVersion: v2\nname: test\nappVersion: "1.0.0"\n')
        assert read_chart_app_version(f) == "1.0.0"

    def test_unquoted(self, tmp_path):
        f = tmp_path / "Chart.yaml"
        f.write_text("apiVersion: v2\nname: test\nappVersion: 2.1.0\n")
        assert read_chart_app_version(f) == "2.1.0"

    def test_missing_raises(self, tmp_path):
        f = tmp_path / "Chart.yaml"
        f.write_text("apiVersion: v2\nname: test\n")
        with pytest.raises(ValueError, match="No appVersion"):
            read_chart_app_version(f)


class TestReadSdkVersionPy:
    """Tests for extracting SDK_VERSION from Python client.py."""

    def test_extracts_version(self, tmp_path):
        f = tmp_path / "client.py"
        f.write_text('SDK_VERSION = "1.1.1"\nUSER_AGENT = f"sdk/{SDK_VERSION}"\n')
        assert read_sdk_version_py(f) == "1.1.1"

    def test_single_quotes(self, tmp_path):
        f = tmp_path / "client.py"
        f.write_text("SDK_VERSION = '2.0.0-beta'\n")
        assert read_sdk_version_py(f) == "2.0.0-beta"

    def test_missing_raises(self, tmp_path):
        f = tmp_path / "client.py"
        f.write_text("# no SDK_VERSION here\n")
        with pytest.raises(ValueError, match="No SDK_VERSION"):
            read_sdk_version_py(f)


class TestReadSdkVersionTs:
    """Tests for extracting SDK_VERSION from TypeScript client.ts."""

    def test_extracts_version(self, tmp_path):
        f = tmp_path / "client.ts"
        f.write_text("export const SDK_VERSION = '1.1.1';\n")
        assert read_sdk_version_ts(f) == "1.1.1"

    def test_double_quotes(self, tmp_path):
        f = tmp_path / "client.ts"
        f.write_text('export const SDK_VERSION = "3.0.0";\n')
        assert read_sdk_version_ts(f) == "3.0.0"

    def test_missing_raises(self, tmp_path):
        f = tmp_path / "client.ts"
        f.write_text("const VERSION = '1.0.0';\n")
        with pytest.raises(ValueError, match="No SDK_VERSION"):
            read_sdk_version_ts(f)


# -------------------------------------------------------------------
# Integration: check that real project files all match
# -------------------------------------------------------------------


class TestProjectVersionsMatch:
    def test_all_versions_match(self):
        """All version sources in the real project must agree."""
        versions = collect_versions(PROJECT_ROOT)
        ok, report = check_consistency(versions)
        assert ok, f"Version mismatch detected:\n{report}"

    def test_collect_versions_returns_all_sources(self):
        versions = collect_versions(PROJECT_ROOT)
        expected_keys = {
            "version.py",
            "sdk/python/pyproject.toml",
            "sdk/python/src/edcocr_sdk/__init__.py",
            "sdk/python/src/edcocr_sdk/client.py",
            "sdk/typescript/package.json",
            "sdk/typescript/src/client.ts",
            "helm/ocr-local/Chart.yaml",
        }
        assert set(versions.keys()) == expected_keys


# -------------------------------------------------------------------
# Mismatch detection
# -------------------------------------------------------------------


class TestMismatchDetection:
    def _make_project(self, tmp_path, version="1.0.0", overrides=None):
        """Create a minimal project tree with consistent versions."""
        overrides = overrides or {}

        (tmp_path / "version.py").write_text(
            f'__version__ = "{overrides.get("version.py", version)}"\n'
        )

        (tmp_path / "sdk" / "python" / "src" / "edcocr_sdk").mkdir(
            parents=True, exist_ok=True
        )
        (tmp_path / "sdk" / "python" / "pyproject.toml").write_text(
            textwrap.dedent(f"""\
                [project]
                name = "edcocr-sdk"
                version = "{overrides.get("sdk/python/pyproject.toml", version)}"
            """)
        )
        (
            tmp_path / "sdk" / "python" / "src" / "edcocr_sdk" / "__init__.py"
        ).write_text(
            f'__version__ = "{overrides.get("sdk/python/src/edcocr_sdk/__init__.py", version)}"\n'
        )
        (
            tmp_path / "sdk" / "python" / "src" / "edcocr_sdk" / "client.py"
        ).write_text(
            f'SDK_VERSION = "{overrides.get("sdk/python/src/edcocr_sdk/client.py", version)}"\n'
        )

        (tmp_path / "sdk" / "typescript").mkdir(parents=True, exist_ok=True)
        (tmp_path / "sdk" / "typescript" / "package.json").write_text(
            json.dumps(
                {
                    "name": "@edcocr/sdk",
                    "version": overrides.get(
                        "sdk/typescript/package.json", version
                    ),
                }
            )
        )
        (tmp_path / "sdk" / "typescript" / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "sdk" / "typescript" / "src" / "client.ts").write_text(
            f"export const SDK_VERSION = '{overrides.get('sdk/typescript/src/client.ts', version)}';\n"
        )

        (tmp_path / "helm" / "ocr-local").mkdir(parents=True, exist_ok=True)
        (tmp_path / "helm" / "ocr-local" / "Chart.yaml").write_text(
            textwrap.dedent(f"""\
                apiVersion: v2
                name: ocr-local
                version: 0.4.0
                appVersion: "{overrides.get("helm/ocr-local/Chart.yaml", version)}"
            """)
        )

    def test_all_match(self, tmp_path):
        self._make_project(tmp_path, "2.0.0")
        versions = collect_versions(tmp_path)
        ok, report = check_consistency(versions)
        assert ok
        assert "2.0.0" in report

    def test_one_mismatch_detected(self, tmp_path):
        self._make_project(
            tmp_path,
            "1.0.0",
            overrides={"sdk/typescript/package.json": "1.0.1"},
        )
        versions = collect_versions(tmp_path)
        ok, report = check_consistency(versions)
        assert not ok
        assert "MISMATCH" in report
        assert "1.0.0" in report
        assert "1.0.1" in report

    def test_missing_file_detected(self, tmp_path):
        self._make_project(tmp_path, "1.0.0")
        # Remove one file
        (tmp_path / "sdk" / "typescript" / "package.json").unlink()
        versions = collect_versions(tmp_path)
        ok, report = check_consistency(versions)
        assert not ok
        assert "FILE_NOT_FOUND" in report


# -------------------------------------------------------------------
# Script exit code
# -------------------------------------------------------------------


class TestScriptExitCode:
    def test_exit_0_when_consistent(self):
        """The script should exit 0 when run against the real project."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "check_version_consistency.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"Script exited {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_exit_1_on_mismatch(self, tmp_path):
        """The script should exit 1 when versions disagree."""
        # Create a minimal project with a mismatch
        (tmp_path / "version.py").write_text('__version__ = "1.0.0"\n')
        (tmp_path / "sdk" / "python" / "src" / "edcocr_sdk").mkdir(
            parents=True
        )
        (tmp_path / "sdk" / "python" / "pyproject.toml").write_text(
            '[project]\nname = "pkg"\nversion = "1.0.0"\n'
        )
        (
            tmp_path / "sdk" / "python" / "src" / "edcocr_sdk" / "__init__.py"
        ).write_text('__version__ = "9.9.9"\n')
        (
            tmp_path / "sdk" / "python" / "src" / "edcocr_sdk" / "client.py"
        ).write_text('SDK_VERSION = "1.0.0"\n')
        (tmp_path / "sdk" / "typescript").mkdir(parents=True)
        (tmp_path / "sdk" / "typescript" / "package.json").write_text(
            json.dumps({"name": "t", "version": "1.0.0"})
        )
        (tmp_path / "sdk" / "typescript" / "src").mkdir(parents=True)
        (tmp_path / "sdk" / "typescript" / "src" / "client.ts").write_text(
            "export const SDK_VERSION = '1.0.0';\n"
        )
        (tmp_path / "helm" / "ocr-local").mkdir(parents=True)
        (tmp_path / "helm" / "ocr-local" / "Chart.yaml").write_text(
            'apiVersion: v2\nname: x\nappVersion: "1.0.0"\n'
        )

        # Copy the script into the tmp project so it auto-detects root
        script_src = PROJECT_ROOT / "scripts" / "check_version_consistency.py"
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "check_version_consistency.py").write_text(
            script_src.read_text(encoding="utf-8")
        )

        result = subprocess.run(
            [sys.executable, str(tmp_path / "scripts" / "check_version_consistency.py")],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 1, (
            f"Expected exit 1 but got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "MISMATCH" in result.stdout
