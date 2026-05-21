#!/usr/bin/env python3
"""Deployment topology detection and validation for EDCOCR.

Detects the intended deployment topology (single-node GPU, single-node CPU,
multi-GPU, distributed, Kubernetes, air-gapped) and validates that the host
satisfies all prerequisites for the detected mode.

Usage:
    python scripts/validate_topology.py
    python scripts/validate_topology.py --topology single-gpu
    python scripts/validate_topology.py --env-file coordinator/.env --check-ports
    python scripts/validate_topology.py --json
    python scripts/validate_topology.py --report topology-report.md
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOPOLOGY_SINGLE_GPU = "single-gpu"
TOPOLOGY_SINGLE_CPU = "single-cpu"
TOPOLOGY_MULTI_GPU = "multi-gpu"
TOPOLOGY_DISTRIBUTED_GPU = "distributed-gpu"
TOPOLOGY_DISTRIBUTED_MIXED = "distributed-mixed"
TOPOLOGY_KUBERNETES = "kubernetes"
TOPOLOGY_AIRGAPPED = "airgapped"

ALL_TOPOLOGIES = [
    TOPOLOGY_SINGLE_GPU,
    TOPOLOGY_SINGLE_CPU,
    TOPOLOGY_MULTI_GPU,
    TOPOLOGY_DISTRIBUTED_GPU,
    TOPOLOGY_DISTRIBUTED_MIXED,
    TOPOLOGY_KUBERNETES,
    TOPOLOGY_AIRGAPPED,
]

# Human-readable names
TOPOLOGY_LABELS: dict[str, str] = {
    TOPOLOGY_SINGLE_GPU: "Single-node GPU",
    TOPOLOGY_SINGLE_CPU: "Single-node CPU (ONNX)",
    TOPOLOGY_MULTI_GPU: "Multi-GPU",
    TOPOLOGY_DISTRIBUTED_GPU: "Distributed (GPU Workers)",
    TOPOLOGY_DISTRIBUTED_MIXED: "Distributed (CPU + GPU Mixed)",
    TOPOLOGY_KUBERNETES: "Kubernetes (Helm)",
    TOPOLOGY_AIRGAPPED: "Air-gapped",
}

# Required environment variables per topology
REQUIRED_ENV: dict[str, list[str]] = {
    TOPOLOGY_SINGLE_GPU: [],
    TOPOLOGY_SINGLE_CPU: ["OCR_TASK_ROUTING"],
    TOPOLOGY_MULTI_GPU: ["ENABLE_PER_GPU_QUEUES", "GPU_COUNT"],
    TOPOLOGY_DISTRIBUTED_GPU: [
        "DJANGO_SECRET_KEY",
        "POSTGRES_PASSWORD",
        "RABBITMQ_PASSWORD",
        "DATABASE_URL",
        "CELERY_BROKER_URL",
    ],
    TOPOLOGY_DISTRIBUTED_MIXED: [
        "DJANGO_SECRET_KEY",
        "POSTGRES_PASSWORD",
        "RABBITMQ_PASSWORD",
        "DATABASE_URL",
        "CELERY_BROKER_URL",
        "OCR_TASK_ROUTING",
    ],
    TOPOLOGY_KUBERNETES: [],
    TOPOLOGY_AIRGAPPED: [],
}

# Required files per topology (relative to project root)
REQUIRED_FILES: dict[str, list[str]] = {
    TOPOLOGY_SINGLE_GPU: ["docker-compose.yml", "Dockerfile"],
    TOPOLOGY_SINGLE_CPU: ["docker-compose.yml", "Dockerfile"],
    TOPOLOGY_MULTI_GPU: ["docker-compose.yml", "Dockerfile", "scripts/generate_multi_gpu_compose.py"],
    TOPOLOGY_DISTRIBUTED_GPU: [
        "coordinator/docker-compose.coordinator.yml",
        "coordinator/docker-compose.worker.yml",
        "coordinator/Dockerfile.coordinator",
        "coordinator/Dockerfile.worker",
    ],
    TOPOLOGY_DISTRIBUTED_MIXED: [
        "coordinator/docker-compose.coordinator.yml",
        "coordinator/docker-compose.worker.yml",
        "coordinator/docker-compose.cpu-only.yml",
        "coordinator/Dockerfile.coordinator",
        "coordinator/Dockerfile.worker",
    ],
    TOPOLOGY_KUBERNETES: ["helm/ocr-local/Chart.yaml", "helm/ocr-local/values.yaml"],
    TOPOLOGY_AIRGAPPED: ["scripts/airgap-bundle.sh", "scripts/airgap-deploy.sh"],
}

# Default ports to check per topology
DEFAULT_PORTS: dict[str, list[tuple[int, str]]] = {
    TOPOLOGY_SINGLE_GPU: [],
    TOPOLOGY_SINGLE_CPU: [],
    TOPOLOGY_MULTI_GPU: [],
    TOPOLOGY_DISTRIBUTED_GPU: [
        (5432, "PostgreSQL"),
        (5672, "RabbitMQ AMQP"),
        (15672, "RabbitMQ Management"),
        (6379, "Redis"),
        (8000, "Coordinator API"),
    ],
    TOPOLOGY_DISTRIBUTED_MIXED: [
        (5432, "PostgreSQL"),
        (5672, "RabbitMQ AMQP"),
        (15672, "RabbitMQ Management"),
        (6379, "Redis"),
        (8000, "Coordinator API"),
    ],
    TOPOLOGY_KUBERNETES: [],
    TOPOLOGY_AIRGAPPED: [],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Single validation check result."""

    category: str
    name: str
    status: str  # "pass", "fail", "skip", "warn"
    message: str


@dataclass
class TopologyReport:
    """Full topology validation report."""

    topology: str
    topology_label: str
    detected: bool  # True if auto-detected, False if explicitly specified
    checks: list[CheckResult] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == "skip")

    @property
    def verdict(self) -> str:
        if self.failed > 0:
            return "FAIL"
        if self.warnings > 0:
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict:
        return {
            "topology": self.topology,
            "topology_label": self.topology_label,
            "detected": self.detected,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "summary": {
                "passed": self.passed,
                "failed": self.failed,
                "warnings": self.warnings,
                "skipped": self.skipped,
                "total": len(self.checks),
            },
            "checks": [asdict(c) for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Safe I/O
# ---------------------------------------------------------------------------


def _safe_print(*args: object, **kwargs: object) -> None:
    """Print with fallback for Windows console encoding."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        text = text.replace("\u2713", "OK").replace("\u2717", "FAIL").replace("\u26a0", "WARN")
        end = kwargs.get("end", "\n")
        stream = kwargs.get("file", sys.stdout)
        if isinstance(stream, io.TextIOWrapper):
            stream.buffer.write((text + str(end)).encode("utf-8", errors="replace"))
        else:
            print(text, **kwargs)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse a .env file into a dict (no shell expansion, simple KEY=VALUE)."""
    env: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return env
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Strip surrounding quotes
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env[key] = value
    return env


def get_env(key: str, env_override: dict[str, str] | None = None) -> str | None:
    """Get env var from override dict first, then os.environ."""
    if env_override and key in env_override:
        return env_override[key]
    return os.environ.get(key)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_gpu_count(env_override: dict[str, str] | None = None) -> int:
    """Detect number of NVIDIA GPUs available.

    Checks GPU_COUNT env var first, then runs nvidia-smi.
    Returns 0 if no GPUs detected or on error.
    """
    gpu_count_str = get_env("GPU_COUNT", env_override)
    if gpu_count_str:
        try:
            return int(gpu_count_str)
        except ValueError:
            pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return 0


def check_command_available(cmd: list[str], timeout: int = 10) -> bool:
    """Check if a command is available on PATH and executes successfully."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a TCP port is available (not in use)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            # connect_ex returns 0 if port is in use (connection succeeded)
            return result != 0
    except OSError:
        return True  # If we can't check, assume available


def detect_topology(
    env_override: dict[str, str] | None = None,
    project_root: Path | None = None,
) -> str:
    """Auto-detect the deployment topology based on environment and resources.

    Detection priority:
    1. DEPLOYMENT_TOPOLOGY env var (explicit override)
    2. KUBERNETES_SERVICE_HOST env var -> kubernetes
    3. DATABASE_URL + CELERY_BROKER_URL + OCR_TASK_ROUTING in (auto, cpu) -> distributed-mixed
    4. DATABASE_URL + CELERY_BROKER_URL -> distributed-gpu
    5. ENABLE_PER_GPU_QUEUES=true or GPU count > 1 -> multi-gpu
    6. OCR_TASK_ROUTING=cpu or no GPU -> single-cpu
    7. Default -> single-gpu
    """
    # 1. Explicit override
    explicit = get_env("DEPLOYMENT_TOPOLOGY", env_override)
    if explicit and explicit in ALL_TOPOLOGIES:
        return explicit

    # 2. Kubernetes
    if get_env("KUBERNETES_SERVICE_HOST", env_override):
        return TOPOLOGY_KUBERNETES

    # 3/4. Distributed (check for coordinator env vars)
    has_db = bool(get_env("DATABASE_URL", env_override))
    has_broker = bool(get_env("CELERY_BROKER_URL", env_override))
    if has_db and has_broker:
        routing = get_env("OCR_TASK_ROUTING", env_override)
        if routing in ("auto", "cpu"):
            return TOPOLOGY_DISTRIBUTED_MIXED
        return TOPOLOGY_DISTRIBUTED_GPU

    # 5. Multi-GPU
    per_gpu_queues = get_env("ENABLE_PER_GPU_QUEUES", env_override)
    if per_gpu_queues and per_gpu_queues.lower() in ("1", "true", "yes"):
        return TOPOLOGY_MULTI_GPU
    gpu_count = detect_gpu_count(env_override)
    if gpu_count > 1:
        return TOPOLOGY_MULTI_GPU

    # 6. CPU-only
    routing = get_env("OCR_TASK_ROUTING", env_override)
    if routing == "cpu":
        return TOPOLOGY_SINGLE_CPU
    if gpu_count == 0:
        return TOPOLOGY_SINGLE_CPU

    # 7. Default
    return TOPOLOGY_SINGLE_GPU


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_env_vars(
    topology: str,
    env_override: dict[str, str] | None = None,
) -> list[CheckResult]:
    """Validate required environment variables for a topology."""
    results: list[CheckResult] = []
    required = REQUIRED_ENV.get(topology, [])

    if not required:
        results.append(CheckResult(
            category="env",
            name="env_vars",
            status="skip",
            message=f"No required env vars for {topology}",
        ))
        return results

    for key in required:
        value = get_env(key, env_override)
        if value:
            results.append(CheckResult(
                category="env",
                name=f"env_{key}",
                status="pass",
                message=f"{key} is set",
            ))
        else:
            results.append(CheckResult(
                category="env",
                name=f"env_{key}",
                status="fail",
                message=f"{key} is not set (required for {topology})",
            ))

    return results


def validate_files(
    topology: str,
    project_root: Path,
) -> list[CheckResult]:
    """Validate required files exist for a topology."""
    results: list[CheckResult] = []
    required = REQUIRED_FILES.get(topology, [])

    if not required:
        results.append(CheckResult(
            category="files",
            name="required_files",
            status="skip",
            message=f"No required files for {topology}",
        ))
        return results

    for rel_path in required:
        full_path = project_root / rel_path
        if full_path.is_file():
            results.append(CheckResult(
                category="files",
                name=f"file_{rel_path.replace('/', '_')}",
                status="pass",
                message=f"{rel_path} exists",
            ))
        else:
            results.append(CheckResult(
                category="files",
                name=f"file_{rel_path.replace('/', '_')}",
                status="fail",
                message=f"{rel_path} not found",
            ))

    return results


def validate_tools(topology: str) -> list[CheckResult]:
    """Validate required CLI tools are available."""
    results: list[CheckResult] = []

    # Docker is needed for most topologies
    if topology not in (TOPOLOGY_KUBERNETES,):
        docker_available = shutil.which("docker") is not None
        if docker_available:
            results.append(CheckResult(
                category="tools",
                name="docker",
                status="pass",
                message="docker is available on PATH",
            ))
        else:
            results.append(CheckResult(
                category="tools",
                name="docker",
                status="fail",
                message="docker not found on PATH",
            ))

        compose_available = shutil.which("docker-compose") is not None or docker_available
        if compose_available:
            results.append(CheckResult(
                category="tools",
                name="docker_compose",
                status="pass",
                message="docker compose is available",
            ))
        else:
            results.append(CheckResult(
                category="tools",
                name="docker_compose",
                status="warn",
                message="docker-compose not found (docker compose plugin may work)",
            ))

    # kubectl and helm for Kubernetes
    if topology == TOPOLOGY_KUBERNETES:
        kubectl_available = shutil.which("kubectl") is not None
        if kubectl_available:
            results.append(CheckResult(
                category="tools",
                name="kubectl",
                status="pass",
                message="kubectl is available on PATH",
            ))
        else:
            results.append(CheckResult(
                category="tools",
                name="kubectl",
                status="fail",
                message="kubectl not found on PATH",
            ))

        helm_available = shutil.which("helm") is not None
        if helm_available:
            results.append(CheckResult(
                category="tools",
                name="helm",
                status="pass",
                message="helm is available on PATH",
            ))
        else:
            results.append(CheckResult(
                category="tools",
                name="helm",
                status="fail",
                message="helm not found on PATH",
            ))

    # GPU tools for GPU topologies
    if topology in (TOPOLOGY_SINGLE_GPU, TOPOLOGY_MULTI_GPU, TOPOLOGY_DISTRIBUTED_GPU):
        nvidia_smi = shutil.which("nvidia-smi") is not None
        if nvidia_smi:
            results.append(CheckResult(
                category="tools",
                name="nvidia_smi",
                status="pass",
                message="nvidia-smi is available on PATH",
            ))
        else:
            results.append(CheckResult(
                category="tools",
                name="nvidia_smi",
                status="warn",
                message="nvidia-smi not found (GPU may not be accessible from host)",
            ))

    return results


def validate_gpu(
    topology: str,
    env_override: dict[str, str] | None = None,
) -> list[CheckResult]:
    """Validate GPU availability matches topology requirements."""
    results: list[CheckResult] = []

    gpu_count = detect_gpu_count(env_override)

    if topology == TOPOLOGY_SINGLE_GPU:
        if gpu_count >= 1:
            results.append(CheckResult(
                category="gpu",
                name="gpu_available",
                status="pass",
                message=f"{gpu_count} GPU(s) detected",
            ))
        else:
            results.append(CheckResult(
                category="gpu",
                name="gpu_available",
                status="fail",
                message="No GPU detected (single-gpu topology requires at least 1 GPU)",
            ))

    elif topology == TOPOLOGY_SINGLE_CPU:
        if gpu_count == 0:
            results.append(CheckResult(
                category="gpu",
                name="gpu_not_needed",
                status="pass",
                message="No GPU detected (expected for CPU topology)",
            ))
        else:
            results.append(CheckResult(
                category="gpu",
                name="gpu_not_needed",
                status="warn",
                message=f"{gpu_count} GPU(s) detected but CPU topology selected (GPU will be unused)",
            ))

    elif topology == TOPOLOGY_MULTI_GPU:
        expected_str = get_env("GPU_COUNT", env_override)
        expected = int(expected_str) if expected_str else 2
        if gpu_count >= expected:
            results.append(CheckResult(
                category="gpu",
                name="multi_gpu",
                status="pass",
                message=f"{gpu_count} GPUs detected (need {expected})",
            ))
        elif gpu_count >= 2:
            results.append(CheckResult(
                category="gpu",
                name="multi_gpu",
                status="warn",
                message=f"{gpu_count} GPUs detected but GPU_COUNT={expected}",
            ))
        else:
            results.append(CheckResult(
                category="gpu",
                name="multi_gpu",
                status="fail",
                message=f"Multi-GPU requires >= 2 GPUs, found {gpu_count}",
            ))

    elif topology in (TOPOLOGY_DISTRIBUTED_GPU, TOPOLOGY_DISTRIBUTED_MIXED):
        # For distributed, GPU check is informational (workers may be remote)
        results.append(CheckResult(
            category="gpu",
            name="gpu_info",
            status="pass" if gpu_count > 0 else "skip",
            message=f"{gpu_count} local GPU(s) detected (workers may be remote)",
        ))

    elif topology == TOPOLOGY_KUBERNETES:
        results.append(CheckResult(
            category="gpu",
            name="gpu_info",
            status="skip",
            message="GPU availability is cluster-dependent",
        ))

    elif topology == TOPOLOGY_AIRGAPPED:
        results.append(CheckResult(
            category="gpu",
            name="gpu_info",
            status="skip",
            message="GPU check depends on base topology",
        ))

    return results


def validate_ports(
    topology: str,
    check_ports: bool = False,
) -> list[CheckResult]:
    """Validate that required ports are available."""
    results: list[CheckResult] = []
    ports = DEFAULT_PORTS.get(topology, [])

    if not ports or not check_ports:
        results.append(CheckResult(
            category="ports",
            name="port_check",
            status="skip",
            message="Port check skipped" if ports else f"No ports to check for {topology}",
        ))
        return results

    for port, service in ports:
        available = check_port_available(port)
        if available:
            results.append(CheckResult(
                category="ports",
                name=f"port_{port}",
                status="pass",
                message=f"Port {port} ({service}) is available",
            ))
        else:
            results.append(CheckResult(
                category="ports",
                name=f"port_{port}",
                status="warn",
                message=f"Port {port} ({service}) is already in use",
            ))

    return results


def validate_env_values(
    topology: str,
    env_override: dict[str, str] | None = None,
) -> list[CheckResult]:
    """Validate specific env var values beyond mere presence."""
    results: list[CheckResult] = []

    if topology == TOPOLOGY_SINGLE_CPU:
        routing = get_env("OCR_TASK_ROUTING", env_override)
        if routing == "cpu":
            results.append(CheckResult(
                category="env_values",
                name="routing_cpu",
                status="pass",
                message="OCR_TASK_ROUTING=cpu (correct for CPU topology)",
            ))
        else:
            results.append(CheckResult(
                category="env_values",
                name="routing_cpu",
                status="fail",
                message=f"OCR_TASK_ROUTING={routing!r}, expected 'cpu' for CPU topology",
            ))

    if topology == TOPOLOGY_MULTI_GPU:
        per_gpu = get_env("ENABLE_PER_GPU_QUEUES", env_override)
        if per_gpu and per_gpu.lower() in ("1", "true", "yes"):
            results.append(CheckResult(
                category="env_values",
                name="per_gpu_queues",
                status="pass",
                message="ENABLE_PER_GPU_QUEUES is enabled",
            ))
        else:
            results.append(CheckResult(
                category="env_values",
                name="per_gpu_queues",
                status="fail",
                message=f"ENABLE_PER_GPU_QUEUES={per_gpu!r}, expected true for multi-gpu",
            ))

        gpu_count_str = get_env("GPU_COUNT", env_override)
        if gpu_count_str:
            try:
                count = int(gpu_count_str)
                if count >= 2:
                    results.append(CheckResult(
                        category="env_values",
                        name="gpu_count_value",
                        status="pass",
                        message=f"GPU_COUNT={count}",
                    ))
                else:
                    results.append(CheckResult(
                        category="env_values",
                        name="gpu_count_value",
                        status="fail",
                        message=f"GPU_COUNT={count}, need >= 2 for multi-gpu",
                    ))
            except ValueError:
                results.append(CheckResult(
                    category="env_values",
                    name="gpu_count_value",
                    status="fail",
                    message=f"GPU_COUNT={gpu_count_str!r} is not a valid integer",
                ))

    if topology in (TOPOLOGY_DISTRIBUTED_GPU, TOPOLOGY_DISTRIBUTED_MIXED):
        db_url = get_env("DATABASE_URL", env_override)
        if db_url and db_url.startswith(("postgres://", "postgresql://")):
            results.append(CheckResult(
                category="env_values",
                name="database_url_format",
                status="pass",
                message="DATABASE_URL has valid postgres:// scheme",
            ))
        elif db_url:
            results.append(CheckResult(
                category="env_values",
                name="database_url_format",
                status="warn",
                message=f"DATABASE_URL scheme is unusual: {db_url[:20]}...",
            ))

        broker_url = get_env("CELERY_BROKER_URL", env_override)
        if broker_url and broker_url.startswith(("amqp://", "redis://")):
            results.append(CheckResult(
                category="env_values",
                name="broker_url_format",
                status="pass",
                message="CELERY_BROKER_URL has valid scheme",
            ))
        elif broker_url:
            results.append(CheckResult(
                category="env_values",
                name="broker_url_format",
                status="warn",
                message=f"CELERY_BROKER_URL scheme is unusual: {broker_url[:20]}...",
            ))

    return results


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------


def run_validation(
    topology: str | None = None,
    env_file: str | Path | None = None,
    check_ports: bool = False,
    project_root: Path | None = None,
) -> TopologyReport:
    """Run full topology validation and return a report."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    # Load env file if provided
    env_override: dict[str, str] | None = None
    if env_file:
        env_override = load_env_file(env_file)

    # Detect or use explicit topology
    detected = topology is None
    if detected:
        resolved_topology = detect_topology(env_override, project_root)
    else:
        resolved_topology = topology  # type: ignore[assignment]

    label = TOPOLOGY_LABELS.get(resolved_topology, resolved_topology)
    report = TopologyReport(
        topology=resolved_topology,
        topology_label=label,
        detected=detected,
    )

    # Run all validation checks
    report.checks.extend(validate_env_vars(resolved_topology, env_override))
    report.checks.extend(validate_env_values(resolved_topology, env_override))
    report.checks.extend(validate_files(resolved_topology, project_root))
    report.checks.extend(validate_tools(resolved_topology))
    report.checks.extend(validate_gpu(resolved_topology, env_override))
    report.checks.extend(validate_ports(resolved_topology, check_ports))

    return report


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_text_report(report: TopologyReport) -> str:
    """Format report as a human-readable text table."""
    lines: list[str] = []
    detection = "auto-detected" if report.detected else "explicitly set"
    lines.append(f"Topology: {report.topology_label} ({report.topology}) [{detection}]")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append("")

    # Status symbols
    symbols = {"pass": "[PASS]", "fail": "[FAIL]", "skip": "[SKIP]", "warn": "[WARN]"}

    # Group by category
    categories: dict[str, list[CheckResult]] = {}
    for check in report.checks:
        categories.setdefault(check.category, []).append(check)

    for category, checks in categories.items():
        lines.append(f"--- {category.upper()} ---")
        for check in checks:
            sym = symbols.get(check.status, "[????]")
            lines.append(f"  {sym} {check.message}")
        lines.append("")

    lines.append(f"Summary: {report.passed} passed, {report.failed} failed, "
                 f"{report.warnings} warnings, {report.skipped} skipped")
    lines.append(f"Verdict: {report.verdict}")

    return "\n".join(lines)


def format_markdown_report(report: TopologyReport) -> str:
    """Format report as Markdown for file output."""
    lines: list[str] = []
    detection = "auto-detected" if report.detected else "explicitly set"
    lines.append("# Topology Validation Report")
    lines.append("")
    lines.append(f"- **Topology**: {report.topology_label} (`{report.topology}`) [{detection}]")
    lines.append(f"- **Timestamp**: {report.timestamp}")
    lines.append(f"- **Verdict**: **{report.verdict}**")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Category | Status | Message |")
    lines.append("|----------|--------|---------|")
    for check in report.checks:
        lines.append(f"| {check.category} | {check.status.upper()} | {check.message} |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Passed: {report.passed}")
    lines.append(f"- Failed: {report.failed}")
    lines.append(f"- Warnings: {report.warnings}")
    lines.append(f"- Skipped: {report.skipped}")
    lines.append(f"- Total: {len(report.checks)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and validate EDCOCR deployment topology.",
    )
    parser.add_argument(
        "--topology",
        choices=ALL_TOPOLOGIES,
        default=None,
        help="Explicit topology to validate (default: auto-detect)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file to load (supplements os.environ)",
    )
    parser.add_argument(
        "--check-ports",
        action="store_true",
        default=False,
        help="Check if required ports are available",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write Markdown report to file",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Override project root directory (default: auto-detect)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on PASS/WARN, 1 on FAIL."""
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(args.project_root) if args.project_root else None

    report = run_validation(
        topology=args.topology,
        env_file=args.env_file,
        check_ports=args.check_ports,
        project_root=project_root,
    )

    if args.json_output:
        _safe_print(json.dumps(report.to_dict(), indent=2))
    else:
        _safe_print(format_text_report(report))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(format_markdown_report(report), encoding="utf-8")
        _safe_print(f"\nReport written to {report_path}")

    return 0 if report.verdict in ("PASS", "WARN") else 1


if __name__ == "__main__":
    sys.exit(main())
