"""Unit tests for scripts/env_preflight.py.

Covers Python version check, system tool detection with mocked subprocess,
directory checks with tmp dirs, GPU detection, port checking with mocked
sockets, CPU-only mode, JSON output, and the full CLI entry point.

Run with: python -m pytest tests/test_env_preflight.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.env_preflight import (
    DEFAULT_PORTS,
    FAIL,
    MIN_PYTHON,
    PASS,
    REQUIRED_PACKAGES,
    SKIP,
    SYSTEM_TOOLS,
    CheckResult,
    PreflightReport,
    build_parser,
    check_directories,
    check_fasttext_model,
    check_gpu_readiness,
    check_port_available,
    check_ports,
    check_python_version,
    check_required_packages,
    check_system_tools,
    format_json_report,
    format_text_report,
    main,
    run_preflight,
)

# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------


class TestPythonVersion:
    """Tests for check_python_version."""

    def test_current_python_passes(self):
        result = check_python_version()
        # We are running on Python 3.10+, so it should pass
        assert result.status == PASS
        assert result.name == "Python version"

    @patch("scripts.env_preflight.sys")
    def test_old_python_fails(self, mock_sys):
        mock_sys.version_info = (3, 9, 0, "final", 0)
        # Re-import to apply mock? No, just call with mock in scope
        # The function reads sys.version_info directly, so we patch it
        with patch("scripts.env_preflight.sys.version_info", new=(3, 9)):
            result = check_python_version()
            assert result.status == FAIL
            assert "3.9" in result.detail

    def test_detail_mentions_minimum(self):
        result = check_python_version()
        assert f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}" in result.detail


# ---------------------------------------------------------------------------
# Package import checks
# ---------------------------------------------------------------------------


class TestRequiredPackages:
    """Tests for check_required_packages."""

    def test_returns_list(self):
        results = check_required_packages()
        assert isinstance(results, list)
        assert len(results) == len(REQUIRED_PACKAGES)

    def test_each_result_has_status(self):
        results = check_required_packages()
        for r in results:
            assert r.status in (PASS, FAIL)
            assert r.name.startswith("Package:")

    @patch("scripts.env_preflight.importlib.import_module")
    def test_missing_package_fails(self, mock_import):
        mock_import.side_effect = ImportError("No module named 'paddleocr'")
        results = check_required_packages()
        # All packages will fail because we mocked import_module globally
        for r in results:
            assert r.status == FAIL

    @patch("scripts.env_preflight.importlib.import_module")
    def test_successful_import_passes(self, mock_import):
        mock_import.return_value = MagicMock()
        results = check_required_packages()
        for r in results:
            assert r.status == PASS


# ---------------------------------------------------------------------------
# System tool checks
# ---------------------------------------------------------------------------


class TestSystemTools:
    """Tests for check_system_tools."""

    @patch("scripts.env_preflight._run_command")
    def test_all_tools_found(self, mock_run):
        mock_run.return_value = (0, "some version output")
        results = check_system_tools()
        assert all(r.status == PASS for r in results)

    @patch("scripts.env_preflight._run_command")
    def test_tool_not_found(self, mock_run):
        mock_run.return_value = (-1, "Command not found: docker")
        results = check_system_tools()
        assert all(r.status == FAIL for r in results)

    @patch("scripts.env_preflight._run_command")
    def test_tool_with_version_in_stderr(self, mock_run):
        # Some tools report version info via stderr with non-zero exit
        mock_run.return_value = (1, "pdftoppm version 22.02.0")
        results = check_system_tools()
        # Should pass because output contains "version"
        for r in results:
            assert r.status == PASS

    def test_returns_one_result_per_tool(self):
        with patch("scripts.env_preflight._run_command", return_value=(0, "ok")):
            results = check_system_tools()
            assert len(results) == len(SYSTEM_TOOLS)


# ---------------------------------------------------------------------------
# Directory checks
# ---------------------------------------------------------------------------


class TestDirectories:
    """Tests for check_directories."""

    def test_existing_dirs_pass(self, tmp_path):
        (tmp_path / "ocr_source").mkdir()
        (tmp_path / "ocr_output").mkdir()
        (tmp_path / "ocr_temp").mkdir()
        results = check_directories(project_root=tmp_path)
        assert all(r.status == PASS for r in results)

    def test_missing_dir_fails(self, tmp_path):
        # Don't create any directories
        results = check_directories(project_root=tmp_path)
        assert all(r.status == FAIL for r in results)

    def test_partial_dirs(self, tmp_path):
        (tmp_path / "ocr_source").mkdir()
        # ocr_output and ocr_temp are missing
        results = check_directories(project_root=tmp_path)
        statuses = {r.name: r.status for r in results}
        assert statuses["Directory: ocr_source"] == PASS
        assert statuses["Directory: ocr_output"] == FAIL
        assert statuses["Directory: ocr_temp"] == FAIL

    def test_writable_check(self, tmp_path):
        (tmp_path / "ocr_source").mkdir()
        (tmp_path / "ocr_output").mkdir()
        (tmp_path / "ocr_temp").mkdir()
        results = check_directories(project_root=tmp_path)
        for r in results:
            if "ocr_output" in r.name or "ocr_temp" in r.name:
                assert "writable" in r.detail


# ---------------------------------------------------------------------------
# FastText model check
# ---------------------------------------------------------------------------


class TestFastTextModel:
    """Tests for check_fasttext_model."""

    def test_model_found(self, tmp_path):
        model = tmp_path / "lid.176.bin"
        model.write_bytes(b"x" * 1024)
        result = check_fasttext_model(project_root=tmp_path)
        assert result.status == PASS
        assert "MB" in result.detail

    def test_model_not_found(self, tmp_path):
        result = check_fasttext_model(project_root=tmp_path)
        assert result.status == FAIL
        assert "not found" in result.detail

    def test_model_in_models_subdir(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        (models / "lid.176.bin").write_bytes(b"x" * 2048)
        result = check_fasttext_model(project_root=tmp_path)
        assert result.status == PASS


# ---------------------------------------------------------------------------
# GPU readiness check
# ---------------------------------------------------------------------------


class TestGpuReadiness:
    """Tests for check_gpu_readiness."""

    def test_cpu_only_skips(self):
        result = check_gpu_readiness(cpu_only=True)
        assert result.status == SKIP
        assert "cpu-only" in result.detail.lower()

    @patch("scripts.env_preflight._run_command")
    def test_gpu_available(self, mock_run):
        mock_run.return_value = (0, "NVIDIA GeForce RTX 3090\nNVIDIA GeForce RTX 3090")
        result = check_gpu_readiness(cpu_only=False)
        assert result.status == PASS
        assert "2 GPU(s)" in result.detail

    @patch("scripts.env_preflight._run_command")
    def test_gpu_not_available(self, mock_run):
        mock_run.return_value = (-1, "Command not found: nvidia-smi")
        result = check_gpu_readiness(cpu_only=False)
        assert result.status == FAIL

    @patch("scripts.env_preflight._run_command")
    def test_single_gpu(self, mock_run):
        mock_run.return_value = (0, "NVIDIA A100")
        result = check_gpu_readiness(cpu_only=False)
        assert result.status == PASS
        assert "1 GPU(s)" in result.detail


# ---------------------------------------------------------------------------
# Port checks
# ---------------------------------------------------------------------------


class TestPortChecks:
    """Tests for check_port_available and check_ports."""

    @patch("scripts.env_preflight.socket.socket")
    def test_port_in_use(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
        mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_sock.connect_ex.return_value = 0
        result = check_port_available("API", 8000)
        assert result.status == PASS
        assert "in use" in result.detail

    @patch("scripts.env_preflight.socket.socket")
    def test_port_available(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
        mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_sock.connect_ex.return_value = 1
        result = check_port_available("API", 8000)
        assert result.status == PASS
        assert "available" in result.detail

    @patch("scripts.env_preflight.socket.socket")
    def test_port_check_error(self, mock_socket_cls):
        mock_socket_cls.return_value.__enter__ = MagicMock(
            side_effect=OSError("Network error")
        )
        mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)
        result = check_port_available("API", 8000)
        assert result.status == FAIL

    def test_check_ports_returns_all(self):
        with patch("scripts.env_preflight.check_port_available") as mock_check:
            mock_check.return_value = CheckResult("port", PASS, "ok")
            results = check_ports()
            assert len(results) == len(DEFAULT_PORTS)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


class TestReportFormatting:
    """Tests for format_text_report and format_json_report."""

    def _make_report(self) -> PreflightReport:
        report = PreflightReport()
        report.checks = [
            CheckResult("Python version", PASS, "3.10"),
            CheckResult("Tool: docker", FAIL, "not found"),
            CheckResult("GPU readiness", SKIP, "cpu-only"),
        ]
        report.compute_verdict()
        return report

    def test_text_report_contains_checks(self):
        report = self._make_report()
        text = format_text_report(report)
        assert "Python version" in text
        assert "Tool: docker" in text
        assert "PASS" in text
        assert "FAIL" in text
        assert "NOT-READY" in text

    def test_text_report_summary_counts(self):
        report = self._make_report()
        text = format_text_report(report)
        assert "PASS=1" in text
        assert "FAIL=1" in text
        assert "SKIP=1" in text

    def test_json_report_valid_json(self):
        report = self._make_report()
        json_str = format_json_report(report)
        data = json.loads(json_str)
        assert data["verdict"] == "NOT-READY"
        assert data["summary"]["pass"] == 1
        assert data["summary"]["fail"] == 1
        assert data["summary"]["skip"] == 1
        assert len(data["checks"]) == 3


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------


class TestVerdict:
    """Tests for PreflightReport.compute_verdict."""

    def test_all_pass_is_ready(self):
        report = PreflightReport()
        report.checks = [CheckResult("A", PASS, ""), CheckResult("B", PASS, "")]
        assert report.compute_verdict() == "READY"

    def test_any_fail_is_not_ready(self):
        report = PreflightReport()
        report.checks = [CheckResult("A", PASS, ""), CheckResult("B", FAIL, "")]
        assert report.compute_verdict() == "NOT-READY"

    def test_skip_only_is_ready(self):
        report = PreflightReport()
        report.checks = [CheckResult("A", PASS, ""), CheckResult("B", SKIP, "")]
        assert report.compute_verdict() == "READY"

    def test_empty_is_ready(self):
        report = PreflightReport()
        assert report.compute_verdict() == "READY"


# ---------------------------------------------------------------------------
# Full run_preflight
# ---------------------------------------------------------------------------


class TestRunPreflight:
    """Tests for run_preflight orchestration."""

    @patch("scripts.env_preflight.check_gpu_readiness")
    @patch("scripts.env_preflight.check_fasttext_model")
    @patch("scripts.env_preflight.check_directories")
    @patch("scripts.env_preflight.check_system_tools")
    @patch("scripts.env_preflight.check_required_packages")
    @patch("scripts.env_preflight.check_python_version")
    def test_all_checks_run(self, mock_py, mock_pkgs, mock_tools, mock_dirs, mock_ft, mock_gpu):
        mock_py.return_value = CheckResult("py", PASS, "ok")
        mock_pkgs.return_value = [CheckResult("pkg", PASS, "ok")]
        mock_tools.return_value = [CheckResult("tool", PASS, "ok")]
        mock_dirs.return_value = [CheckResult("dir", PASS, "ok")]
        mock_ft.return_value = CheckResult("ft", PASS, "ok")
        mock_gpu.return_value = CheckResult("gpu", PASS, "ok")

        report = run_preflight()
        assert report.verdict == "READY"
        assert len(report.checks) == 6

    def test_cpu_only_skips_gpu(self):
        with patch("scripts.env_preflight.check_gpu_readiness") as mock_gpu:
            mock_gpu.return_value = CheckResult("GPU", SKIP, "skipped")
            run_preflight(cpu_only=True)
            mock_gpu.assert_called_once_with(cpu_only=True)

    def test_ports_only_when_requested(self):
        with patch("scripts.env_preflight.check_ports") as mock_ports:
            mock_ports.return_value = [CheckResult("port", PASS, "ok")]
            run_preflight(check_ports_flag=False)
            mock_ports.assert_not_called()

            run_preflight(check_ports_flag=True)
            mock_ports.assert_called_once()


# ---------------------------------------------------------------------------
# CLI / main tests
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() CLI entry point."""

    @patch("scripts.env_preflight.run_preflight")
    def test_main_text_output(self, mock_run, capsys):
        report = PreflightReport()
        report.checks = [CheckResult("A", PASS, "ok")]
        report.compute_verdict()
        mock_run.return_value = report

        rc = main([])
        assert rc == 0
        captured = capsys.readouterr()
        assert "READY" in captured.out

    @patch("scripts.env_preflight.run_preflight")
    def test_main_json_output(self, mock_run, capsys):
        report = PreflightReport()
        report.checks = [CheckResult("A", PASS, "ok")]
        report.compute_verdict()
        mock_run.return_value = report

        rc = main(["--json"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["verdict"] == "READY"

    @patch("scripts.env_preflight.run_preflight")
    def test_main_returns_1_on_fail(self, mock_run):
        report = PreflightReport()
        report.checks = [CheckResult("A", FAIL, "bad")]
        report.compute_verdict()
        mock_run.return_value = report

        rc = main([])
        assert rc == 1

    @patch("scripts.env_preflight.run_preflight")
    def test_main_report_file(self, mock_run, tmp_path):
        report = PreflightReport()
        report.checks = [CheckResult("A", PASS, "ok")]
        report.compute_verdict()
        mock_run.return_value = report

        report_path = tmp_path / "report.txt"
        rc = main(["--report", str(report_path)])
        assert rc == 0
        assert report_path.exists()
        content = report_path.read_text()
        assert "READY" in content

    @patch("scripts.env_preflight.run_preflight")
    def test_main_json_report_file(self, mock_run, tmp_path):
        report = PreflightReport()
        report.checks = [CheckResult("A", PASS, "ok")]
        report.compute_verdict()
        mock_run.return_value = report

        report_path = tmp_path / "report.json"
        rc = main(["--report", str(report_path)])
        assert rc == 0
        data = json.loads(report_path.read_text())
        assert data["verdict"] == "READY"

    def test_build_parser(self):
        parser = build_parser()
        args = parser.parse_args(["--check-ports", "--cpu-only", "--json"])
        assert args.check_ports is True
        assert args.cpu_only is True
        assert args.json_output is True
