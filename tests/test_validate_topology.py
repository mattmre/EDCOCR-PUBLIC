"""Unit tests for scripts/validate_topology.py.

Covers topology detection, per-topology validation, GPU detection mocking,
file existence checks, env value validation, port checking, JSON output,
and the full CLI entry point.

Run with: python -m pytest tests/test_validate_topology.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.validate_topology import (
    ALL_TOPOLOGIES,
    REQUIRED_ENV,
    REQUIRED_FILES,
    TOPOLOGY_AIRGAPPED,
    TOPOLOGY_DISTRIBUTED_GPU,
    TOPOLOGY_DISTRIBUTED_MIXED,
    TOPOLOGY_KUBERNETES,
    TOPOLOGY_LABELS,
    TOPOLOGY_MULTI_GPU,
    TOPOLOGY_SINGLE_CPU,
    TOPOLOGY_SINGLE_GPU,
    CheckResult,
    TopologyReport,
    build_parser,
    check_command_available,
    check_port_available,
    detect_gpu_count,
    detect_topology,
    format_markdown_report,
    format_text_report,
    get_env,
    load_env_file,
    main,
    run_validation,
    validate_env_values,
    validate_env_vars,
    validate_files,
    validate_gpu,
    validate_ports,
    validate_tools,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project structure for file validation."""
    (tmp_path / "docker-compose.yml").write_text("services: {}")
    (tmp_path / "Dockerfile").write_text("FROM python:3.10")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "generate_multi_gpu_compose.py").write_text("# gen")
    (scripts / "airgap-bundle.sh").write_text("#!/bin/bash")
    (scripts / "airgap-deploy.sh").write_text("#!/bin/bash")
    coordinator = tmp_path / "coordinator"
    coordinator.mkdir()
    (coordinator / "docker-compose.coordinator.yml").write_text("services: {}")
    (coordinator / "docker-compose.worker.yml").write_text("services: {}")
    (coordinator / "docker-compose.cpu-only.yml").write_text("services: {}")
    (coordinator / "Dockerfile.coordinator").write_text("FROM python:3.10")
    (coordinator / "Dockerfile.worker").write_text("FROM python:3.10")
    helm = tmp_path / "helm" / "ocr-local"
    helm.mkdir(parents=True)
    (helm / "Chart.yaml").write_text("apiVersion: v2")
    (helm / "values.yaml").write_text("# values")
    return tmp_path


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    """Create a sample .env file."""
    p = tmp_path / ".env"
    p.write_text(textwrap.dedent("""\
        DJANGO_SECRET_KEY=test-secret-key-12345
        POSTGRES_PASSWORD=supersecret
        RABBITMQ_PASSWORD=rabbitpass
        DATABASE_URL=postgres://ocr:pass@localhost:5432/ocr_coordinator
        CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
        OCR_TASK_ROUTING=cpu
        ENABLE_PER_GPU_QUEUES=true
        GPU_COUNT=4
    """))
    return p


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestDetectTopology:
    """Tests for topology auto-detection logic."""

    def test_explicit_override(self) -> None:
        env = {"DEPLOYMENT_TOPOLOGY": "kubernetes"}
        assert detect_topology(env) == TOPOLOGY_KUBERNETES

    def test_explicit_override_invalid_falls_through(self) -> None:
        env = {"DEPLOYMENT_TOPOLOGY": "not-a-topology"}
        # Falls through to default detection
        result = detect_topology(env)
        assert result in ALL_TOPOLOGIES

    def test_kubernetes_detection(self) -> None:
        env = {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}
        assert detect_topology(env) == TOPOLOGY_KUBERNETES

    def test_distributed_gpu_detection(self) -> None:
        env = {
            "DATABASE_URL": "postgres://localhost/db",
            "CELERY_BROKER_URL": "amqp://localhost//",
        }
        assert detect_topology(env) == TOPOLOGY_DISTRIBUTED_GPU

    def test_distributed_mixed_detection(self) -> None:
        env = {
            "DATABASE_URL": "postgres://localhost/db",
            "CELERY_BROKER_URL": "amqp://localhost//",
            "OCR_TASK_ROUTING": "auto",
        }
        assert detect_topology(env) == TOPOLOGY_DISTRIBUTED_MIXED

    def test_distributed_mixed_cpu_routing(self) -> None:
        env = {
            "DATABASE_URL": "postgres://localhost/db",
            "CELERY_BROKER_URL": "amqp://localhost//",
            "OCR_TASK_ROUTING": "cpu",
        }
        assert detect_topology(env) == TOPOLOGY_DISTRIBUTED_MIXED

    def test_multi_gpu_from_env(self) -> None:
        env = {"ENABLE_PER_GPU_QUEUES": "true"}
        assert detect_topology(env) == TOPOLOGY_MULTI_GPU

    def test_multi_gpu_from_gpu_count(self) -> None:
        env = {"GPU_COUNT": "4"}
        with patch("scripts.validate_topology.detect_gpu_count", return_value=4):
            assert detect_topology(env) == TOPOLOGY_MULTI_GPU

    def test_single_cpu_from_routing(self) -> None:
        env = {"OCR_TASK_ROUTING": "cpu"}
        assert detect_topology(env) == TOPOLOGY_SINGLE_CPU

    @patch("scripts.validate_topology.detect_gpu_count", return_value=0)
    def test_single_cpu_no_gpu(self, mock_gpu: MagicMock) -> None:
        assert detect_topology({}) == TOPOLOGY_SINGLE_CPU

    @patch("scripts.validate_topology.detect_gpu_count", return_value=1)
    def test_default_single_gpu(self, mock_gpu: MagicMock) -> None:
        assert detect_topology({}) == TOPOLOGY_SINGLE_GPU


# ---------------------------------------------------------------------------
# GPU detection tests
# ---------------------------------------------------------------------------


class TestDetectGpuCount:
    """Tests for GPU count detection with mocked subprocess."""

    def test_gpu_count_from_env(self) -> None:
        assert detect_gpu_count({"GPU_COUNT": "3"}) == 3

    def test_gpu_count_from_env_invalid(self) -> None:
        # Invalid GPU_COUNT falls through to nvidia-smi
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert detect_gpu_count({"GPU_COUNT": "abc"}) == 0

    @patch("subprocess.run")
    def test_gpu_count_from_nvidia_smi(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3090\nNVIDIA GeForce RTX 3090\n",
        )
        assert detect_gpu_count({}) == 2

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_gpu_count_nvidia_smi_not_found(self, mock_run: MagicMock) -> None:
        assert detect_gpu_count({}) == 0

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10))
    def test_gpu_count_nvidia_smi_timeout(self, mock_run: MagicMock) -> None:
        assert detect_gpu_count({}) == 0

    @patch("subprocess.run")
    def test_gpu_count_nvidia_smi_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert detect_gpu_count({}) == 0


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


class TestEnvHelpers:
    """Tests for load_env_file and get_env."""

    def test_load_env_file(self, env_file: Path) -> None:
        env = load_env_file(env_file)
        assert env["DJANGO_SECRET_KEY"] == "test-secret-key-12345"
        assert env["POSTGRES_PASSWORD"] == "supersecret"
        assert env["GPU_COUNT"] == "4"

    def test_load_env_file_missing(self, tmp_path: Path) -> None:
        env = load_env_file(tmp_path / "nonexistent.env")
        assert env == {}

    def test_load_env_file_comments_and_blanks(self, tmp_path: Path) -> None:
        p = tmp_path / ".env"
        p.write_text("# comment\n\nKEY=value\n")
        env = load_env_file(p)
        assert env == {"KEY": "value"}

    def test_load_env_file_quoted_values(self, tmp_path: Path) -> None:
        p = tmp_path / ".env"
        p.write_text('KEY1="double quoted"\nKEY2=\'single quoted\'\n')
        env = load_env_file(p)
        assert env["KEY1"] == "double quoted"
        assert env["KEY2"] == "single quoted"

    def test_get_env_override_takes_precedence(self) -> None:
        with patch.dict(os.environ, {"KEY": "from_os"}, clear=False):
            assert get_env("KEY", {"KEY": "from_override"}) == "from_override"

    def test_get_env_falls_back_to_os(self) -> None:
        with patch.dict(os.environ, {"KEY": "from_os"}, clear=False):
            assert get_env("KEY", {}) == "from_os"

    def test_get_env_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert get_env("NONEXISTENT_KEY_12345", {}) is None


# ---------------------------------------------------------------------------
# Validation: env vars
# ---------------------------------------------------------------------------


class TestValidateEnvVars:
    """Tests for required environment variable validation."""

    def test_single_gpu_no_required_vars(self) -> None:
        results = validate_env_vars(TOPOLOGY_SINGLE_GPU, {})
        assert len(results) == 1
        assert results[0].status == "skip"

    def test_distributed_gpu_all_set(self) -> None:
        env = {
            "DJANGO_SECRET_KEY": "s",
            "POSTGRES_PASSWORD": "p",
            "RABBITMQ_PASSWORD": "r",
            "DATABASE_URL": "postgres://localhost/db",
            "CELERY_BROKER_URL": "amqp://localhost//",
        }
        results = validate_env_vars(TOPOLOGY_DISTRIBUTED_GPU, env)
        assert all(r.status == "pass" for r in results)

    def test_distributed_gpu_missing_vars(self) -> None:
        results = validate_env_vars(TOPOLOGY_DISTRIBUTED_GPU, {})
        failed = [r for r in results if r.status == "fail"]
        assert len(failed) == len(REQUIRED_ENV[TOPOLOGY_DISTRIBUTED_GPU])

    def test_single_cpu_routing_set(self) -> None:
        env = {"OCR_TASK_ROUTING": "cpu"}
        results = validate_env_vars(TOPOLOGY_SINGLE_CPU, env)
        assert any(r.status == "pass" and "OCR_TASK_ROUTING" in r.message for r in results)


# ---------------------------------------------------------------------------
# Validation: files
# ---------------------------------------------------------------------------


class TestValidateFiles:
    """Tests for required file existence validation."""

    def test_single_gpu_files_exist(self, project_root: Path) -> None:
        results = validate_files(TOPOLOGY_SINGLE_GPU, project_root)
        assert all(r.status == "pass" for r in results)

    def test_missing_compose_file(self, tmp_path: Path) -> None:
        results = validate_files(TOPOLOGY_SINGLE_GPU, tmp_path)
        failed = [r for r in results if r.status == "fail"]
        assert len(failed) >= 1
        assert any("docker-compose.yml" in r.message for r in failed)

    def test_kubernetes_files_exist(self, project_root: Path) -> None:
        results = validate_files(TOPOLOGY_KUBERNETES, project_root)
        assert all(r.status == "pass" for r in results)

    def test_distributed_files_exist(self, project_root: Path) -> None:
        results = validate_files(TOPOLOGY_DISTRIBUTED_GPU, project_root)
        assert all(r.status == "pass" for r in results)

    def test_airgapped_files_exist(self, project_root: Path) -> None:
        results = validate_files(TOPOLOGY_AIRGAPPED, project_root)
        assert all(r.status == "pass" for r in results)


# ---------------------------------------------------------------------------
# Validation: env values
# ---------------------------------------------------------------------------


class TestValidateEnvValues:
    """Tests for env var value correctness validation."""

    def test_cpu_routing_correct(self) -> None:
        env = {"OCR_TASK_ROUTING": "cpu"}
        results = validate_env_values(TOPOLOGY_SINGLE_CPU, env)
        assert any(r.status == "pass" and "routing" in r.name for r in results)

    def test_cpu_routing_wrong(self) -> None:
        env = {"OCR_TASK_ROUTING": "gpu"}
        results = validate_env_values(TOPOLOGY_SINGLE_CPU, env)
        assert any(r.status == "fail" and "routing" in r.name for r in results)

    def test_multi_gpu_per_gpu_queues_enabled(self) -> None:
        env = {"ENABLE_PER_GPU_QUEUES": "true", "GPU_COUNT": "4"}
        results = validate_env_values(TOPOLOGY_MULTI_GPU, env)
        passed = [r for r in results if r.status == "pass"]
        assert len(passed) >= 2

    def test_multi_gpu_per_gpu_queues_disabled(self) -> None:
        env = {"ENABLE_PER_GPU_QUEUES": "false", "GPU_COUNT": "4"}
        results = validate_env_values(TOPOLOGY_MULTI_GPU, env)
        assert any(r.status == "fail" and "per_gpu" in r.name for r in results)

    def test_multi_gpu_count_too_low(self) -> None:
        env = {"ENABLE_PER_GPU_QUEUES": "true", "GPU_COUNT": "1"}
        results = validate_env_values(TOPOLOGY_MULTI_GPU, env)
        assert any(r.status == "fail" and "gpu_count" in r.name for r in results)

    def test_distributed_db_url_valid(self) -> None:
        env = {
            "DATABASE_URL": "postgres://ocr:pass@localhost:5432/db",
            "CELERY_BROKER_URL": "amqp://guest:guest@localhost:5672//",
        }
        results = validate_env_values(TOPOLOGY_DISTRIBUTED_GPU, env)
        passed = [r for r in results if r.status == "pass"]
        assert len(passed) >= 2


# ---------------------------------------------------------------------------
# Validation: tools
# ---------------------------------------------------------------------------


class TestValidateTools:
    """Tests for CLI tool availability checks."""

    @patch("shutil.which", return_value="/usr/bin/docker")
    def test_docker_available(self, mock_which: MagicMock) -> None:
        results = validate_tools(TOPOLOGY_SINGLE_GPU)
        assert any(r.status == "pass" and "docker" in r.name for r in results)

    @patch("shutil.which", return_value=None)
    def test_docker_missing(self, mock_which: MagicMock) -> None:
        results = validate_tools(TOPOLOGY_SINGLE_GPU)
        assert any(r.status == "fail" and r.name == "docker" for r in results)

    @patch("shutil.which", side_effect=lambda cmd: "/usr/bin/kubectl" if cmd == "kubectl" else "/usr/bin/helm" if cmd == "helm" else None)
    def test_kubernetes_tools(self, mock_which: MagicMock) -> None:
        results = validate_tools(TOPOLOGY_KUBERNETES)
        assert any(r.status == "pass" and "kubectl" in r.name for r in results)
        assert any(r.status == "pass" and "helm" in r.name for r in results)


# ---------------------------------------------------------------------------
# Validation: GPU
# ---------------------------------------------------------------------------


class TestValidateGpu:
    """Tests for GPU availability validation."""

    @patch("scripts.validate_topology.detect_gpu_count", return_value=1)
    def test_single_gpu_has_gpu(self, mock_gpu: MagicMock) -> None:
        results = validate_gpu(TOPOLOGY_SINGLE_GPU)
        assert any(r.status == "pass" for r in results)

    @patch("scripts.validate_topology.detect_gpu_count", return_value=0)
    def test_single_gpu_no_gpu(self, mock_gpu: MagicMock) -> None:
        results = validate_gpu(TOPOLOGY_SINGLE_GPU)
        assert any(r.status == "fail" for r in results)

    @patch("scripts.validate_topology.detect_gpu_count", return_value=0)
    def test_single_cpu_no_gpu_ok(self, mock_gpu: MagicMock) -> None:
        results = validate_gpu(TOPOLOGY_SINGLE_CPU)
        assert any(r.status == "pass" for r in results)

    @patch("scripts.validate_topology.detect_gpu_count", return_value=2)
    def test_single_cpu_has_gpu_warns(self, mock_gpu: MagicMock) -> None:
        results = validate_gpu(TOPOLOGY_SINGLE_CPU)
        assert any(r.status == "warn" for r in results)

    @patch("scripts.validate_topology.detect_gpu_count", return_value=4)
    def test_multi_gpu_sufficient(self, mock_gpu: MagicMock) -> None:
        env = {"GPU_COUNT": "4"}
        results = validate_gpu(TOPOLOGY_MULTI_GPU, env)
        assert any(r.status == "pass" for r in results)

    @patch("scripts.validate_topology.detect_gpu_count", return_value=1)
    def test_multi_gpu_insufficient(self, mock_gpu: MagicMock) -> None:
        results = validate_gpu(TOPOLOGY_MULTI_GPU)
        assert any(r.status == "fail" for r in results)

    def test_kubernetes_gpu_skipped(self) -> None:
        results = validate_gpu(TOPOLOGY_KUBERNETES)
        assert any(r.status == "skip" for r in results)


# ---------------------------------------------------------------------------
# Validation: ports
# ---------------------------------------------------------------------------


class TestValidatePorts:
    """Tests for port availability checking."""

    def test_ports_skipped_by_default(self) -> None:
        results = validate_ports(TOPOLOGY_DISTRIBUTED_GPU, check_ports=False)
        assert all(r.status == "skip" for r in results)

    @patch("scripts.validate_topology.check_port_available", return_value=True)
    def test_ports_all_available(self, mock_port: MagicMock) -> None:
        results = validate_ports(TOPOLOGY_DISTRIBUTED_GPU, check_ports=True)
        passed = [r for r in results if r.status == "pass"]
        assert len(passed) >= 3

    @patch("scripts.validate_topology.check_port_available", return_value=False)
    def test_ports_in_use(self, mock_port: MagicMock) -> None:
        results = validate_ports(TOPOLOGY_DISTRIBUTED_GPU, check_ports=True)
        warned = [r for r in results if r.status == "warn"]
        assert len(warned) >= 3

    def test_no_ports_for_single_gpu(self) -> None:
        results = validate_ports(TOPOLOGY_SINGLE_GPU, check_ports=True)
        assert all(r.status == "skip" for r in results)


# ---------------------------------------------------------------------------
# TopologyReport
# ---------------------------------------------------------------------------


class TestTopologyReport:
    """Tests for TopologyReport data class."""

    def test_verdict_pass(self) -> None:
        report = TopologyReport(
            topology=TOPOLOGY_SINGLE_GPU,
            topology_label="Test",
            detected=True,
            checks=[CheckResult("env", "test", "pass", "ok")],
        )
        assert report.verdict == "PASS"

    def test_verdict_fail(self) -> None:
        report = TopologyReport(
            topology=TOPOLOGY_SINGLE_GPU,
            topology_label="Test",
            detected=True,
            checks=[
                CheckResult("env", "test1", "pass", "ok"),
                CheckResult("env", "test2", "fail", "missing"),
            ],
        )
        assert report.verdict == "FAIL"

    def test_verdict_warn(self) -> None:
        report = TopologyReport(
            topology=TOPOLOGY_SINGLE_GPU,
            topology_label="Test",
            detected=True,
            checks=[
                CheckResult("env", "test1", "pass", "ok"),
                CheckResult("env", "test2", "warn", "advisory"),
            ],
        )
        assert report.verdict == "WARN"

    def test_to_dict(self) -> None:
        report = TopologyReport(
            topology=TOPOLOGY_SINGLE_GPU,
            topology_label="Test",
            detected=True,
            checks=[CheckResult("env", "test", "pass", "ok")],
        )
        d = report.to_dict()
        assert d["topology"] == TOPOLOGY_SINGLE_GPU
        assert d["verdict"] == "PASS"
        assert d["summary"]["passed"] == 1
        assert len(d["checks"]) == 1

    def test_counts(self) -> None:
        report = TopologyReport(
            topology=TOPOLOGY_SINGLE_GPU,
            topology_label="Test",
            detected=False,
            checks=[
                CheckResult("a", "1", "pass", "ok"),
                CheckResult("a", "2", "fail", "bad"),
                CheckResult("a", "3", "warn", "meh"),
                CheckResult("a", "4", "skip", "na"),
            ],
        )
        assert report.passed == 1
        assert report.failed == 1
        assert report.warnings == 1
        assert report.skipped == 1


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    """Tests for text and markdown formatting."""

    def test_text_report_contains_topology(self) -> None:
        report = TopologyReport(
            topology=TOPOLOGY_SINGLE_GPU,
            topology_label="Single-node GPU",
            detected=True,
            checks=[CheckResult("env", "test", "pass", "ok")],
        )
        text = format_text_report(report)
        assert "Single-node GPU" in text
        assert "PASS" in text

    def test_markdown_report_has_table(self) -> None:
        report = TopologyReport(
            topology=TOPOLOGY_KUBERNETES,
            topology_label="Kubernetes",
            detected=False,
            checks=[CheckResult("tools", "kubectl", "pass", "kubectl available")],
        )
        md = format_markdown_report(report)
        assert "| Category |" in md
        assert "kubectl available" in md

    def test_text_report_fail_verdict(self) -> None:
        report = TopologyReport(
            topology=TOPOLOGY_SINGLE_GPU,
            topology_label="Test",
            detected=True,
            checks=[CheckResult("env", "test", "fail", "missing KEY")],
        )
        text = format_text_report(report)
        assert "Verdict: FAIL" in text


# ---------------------------------------------------------------------------
# Full validation (integration)
# ---------------------------------------------------------------------------


class TestRunValidation:
    """Tests for run_validation integration."""

    @patch("scripts.validate_topology.detect_gpu_count", return_value=0)
    def test_auto_detect_cpu(self, mock_gpu: MagicMock, project_root: Path) -> None:
        report = run_validation(project_root=project_root)
        # With no GPU, should detect as single-cpu
        assert report.topology == TOPOLOGY_SINGLE_CPU
        assert report.detected is True

    def test_explicit_topology(self, project_root: Path) -> None:
        report = run_validation(topology=TOPOLOGY_KUBERNETES, project_root=project_root)
        assert report.topology == TOPOLOGY_KUBERNETES
        assert report.detected is False

    def test_with_env_file(self, project_root: Path, env_file: Path) -> None:
        report = run_validation(
            topology=TOPOLOGY_DISTRIBUTED_GPU,
            env_file=env_file,
            project_root=project_root,
        )
        # env file has all required vars for distributed-gpu
        env_checks = [c for c in report.checks if c.category == "env"]
        assert all(c.status == "pass" for c in env_checks)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the main() CLI entry point."""

    @patch("scripts.validate_topology.detect_gpu_count", return_value=0)
    def test_main_json_output(self, mock_gpu: MagicMock, capsys: pytest.CaptureFixture, project_root: Path) -> None:
        exit_code = main(["--topology", "single-cpu", "--json", "--project-root", str(project_root)])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["topology"] == "single-cpu"
        assert "verdict" in data
        # single-cpu without OCR_TASK_ROUTING set will fail
        assert exit_code in (0, 1)

    @patch("scripts.validate_topology.detect_gpu_count", return_value=0)
    def test_main_text_output(self, mock_gpu: MagicMock, capsys: pytest.CaptureFixture, project_root: Path) -> None:
        main(["--topology", "kubernetes", "--project-root", str(project_root)])
        captured = capsys.readouterr()
        assert "Kubernetes" in captured.out

    @patch("scripts.validate_topology.detect_gpu_count", return_value=0)
    def test_main_report_file(self, mock_gpu: MagicMock, tmp_path: Path, project_root: Path) -> None:
        report_path = tmp_path / "report.md"
        main(["--topology", "kubernetes", "--report", str(report_path), "--project-root", str(project_root)])
        assert report_path.is_file()
        content = report_path.read_text(encoding="utf-8")
        assert "# Topology Validation Report" in content

    @patch("scripts.validate_topology.detect_gpu_count", return_value=1)
    def test_main_pass_exit_code(self, mock_gpu: MagicMock, project_root: Path) -> None:
        exit_code = main(["--topology", "single-gpu", "--project-root", str(project_root)])
        # May pass or fail depending on docker availability, but should not crash
        assert exit_code in (0, 1)

    def test_build_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--topology", "kubernetes", "--json"])
        assert args.topology == "kubernetes"
        assert args.json_output is True
        assert args.check_ports is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_topologies_have_labels(self) -> None:
        for t in ALL_TOPOLOGIES:
            assert t in TOPOLOGY_LABELS

    def test_all_topologies_have_required_env(self) -> None:
        for t in ALL_TOPOLOGIES:
            assert t in REQUIRED_ENV

    def test_all_topologies_have_required_files(self) -> None:
        for t in ALL_TOPOLOGIES:
            assert t in REQUIRED_FILES

    def test_check_port_available_function(self) -> None:
        # Port 1 is rarely in use, so should be available
        result = check_port_available(1)
        assert isinstance(result, bool)

    def test_check_command_not_found(self) -> None:
        assert check_command_available(["nonexistent_cmd_12345"]) is False

    def test_detect_topology_empty_env(self) -> None:
        """With clean env and mocked no-GPU, should get single-cpu."""
        with patch("scripts.validate_topology.detect_gpu_count", return_value=0):
            with patch.dict(os.environ, {}, clear=True):
                result = detect_topology({})
                assert result == TOPOLOGY_SINGLE_CPU

    def test_env_file_no_equals(self, tmp_path: Path) -> None:
        """Lines without = sign are skipped."""
        p = tmp_path / ".env"
        p.write_text("INVALID_LINE\nKEY=value\n")
        env = load_env_file(p)
        assert env == {"KEY": "value"}

    def test_multi_gpu_gpu_count_non_numeric(self) -> None:
        env = {"ENABLE_PER_GPU_QUEUES": "true", "GPU_COUNT": "not-a-number"}
        results = validate_env_values(TOPOLOGY_MULTI_GPU, env)
        assert any(r.status == "fail" and "not a valid integer" in r.message for r in results)
