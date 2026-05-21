#!/usr/bin/env python3
"""Environment preflight checker for EDCOCR.

Quick pre-deployment check that validates all prerequisites are met.
Focuses on binary availability, file structure, and runtime readiness
rather than credentials and connectivity (use cutover_preflight.py
for those).

Usage:
    python scripts/env_preflight.py
    python scripts/env_preflight.py --check-ports
    python scripts/env_preflight.py --cpu-only
    python scripts/env_preflight.py --json
    python scripts/env_preflight.py --report preflight-report.md
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: locate project root
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SUBPROCESS_TIMEOUT = 10  # seconds

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# Minimum Python version
MIN_PYTHON = (3, 10)

# Required importable packages and their import names
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("Pillow", "PIL"),
    ("PyMuPDF", "fitz"),
    ("pytesseract", "pytesseract"),
    ("paddleocr", "paddleocr"),
]

# System tools to check: (display_name, command_variants)
# Each variant is tried in order; first success wins.
SYSTEM_TOOLS: list[tuple[str, list[list[str]]]] = [
    ("docker", [["docker", "--version"]]),
    ("docker-compose", [["docker", "compose", "version"], ["docker-compose", "--version"]]),
    ("ghostscript", [["gs", "--version"], ["gswin64c", "--version"], ["gswin32c", "--version"]]),
    ("poppler", [["pdftoppm", "-v"]]),
]

# Key directories relative to project root
KEY_DIRECTORIES: list[tuple[str, str, bool]] = [
    # (name, relative_path, needs_writable)
    ("ocr_source", "ocr_source", False),
    ("ocr_output", "ocr_output", True),
    ("ocr_temp", "ocr_temp", True),
]

# FastText model locations to check
FASTTEXT_MODEL_PATHS = [
    "/app/lid.176.bin",
    "lid.176.bin",
    "models/lid.176.bin",
]

# Ports to check (name, port)
DEFAULT_PORTS: list[tuple[str, int]] = [
    ("API", 8000),
    ("PostgreSQL", 5432),
    ("RabbitMQ", 5672),
    ("Redis", 6379),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single preflight check."""

    name: str
    status: str  # PASS, FAIL, SKIP
    detail: str = ""


@dataclass
class PreflightReport:
    """Overall preflight report."""

    checks: list[CheckResult] = field(default_factory=list)
    timestamp: str = ""
    verdict: str = ""  # READY, NOT-READY

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def compute_verdict(self) -> str:
        """Compute overall verdict from check results."""
        has_fail = any(c.status == FAIL for c in self.checks)
        self.verdict = "NOT-READY" if has_fail else "READY"
        return self.verdict


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def _run_command(cmd: list[str], *, timeout: int = SUBPROCESS_TIMEOUT) -> tuple[int, str]:
    """Run a subprocess safely. Returns (returncode, output). Never raises."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.stderr:
            output = output + " " + result.stderr.strip() if output else result.stderr.strip()
        return result.returncode, output
    except FileNotFoundError:
        return -1, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, f"Timed out after {timeout}s"
    except Exception as exc:
        return -1, str(exc)


def check_python_version() -> CheckResult:
    """Check Python version is >= MIN_PYTHON."""
    current = sys.version_info[:2]
    if current >= MIN_PYTHON:
        return CheckResult(
            name="Python version",
            status=PASS,
            detail=f"{current[0]}.{current[1]} (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
        )
    return CheckResult(
        name="Python version",
        status=FAIL,
        detail=f"{current[0]}.{current[1]} (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
    )


def check_required_packages() -> list[CheckResult]:
    """Check that required Python packages are importable."""
    results: list[CheckResult] = []
    for display_name, import_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            results.append(CheckResult(
                name=f"Package: {display_name}",
                status=PASS,
                detail=f"import {import_name} OK",
            ))
        except ImportError as exc:
            results.append(CheckResult(
                name=f"Package: {display_name}",
                status=FAIL,
                detail=f"import {import_name} failed: {exc}",
            ))
        except Exception as exc:
            results.append(CheckResult(
                name=f"Package: {display_name}",
                status=FAIL,
                detail=f"import {import_name} error: {exc}",
            ))
    return results


def check_system_tools() -> list[CheckResult]:
    """Check availability of required system tools."""
    results: list[CheckResult] = []
    for tool_name, variants in SYSTEM_TOOLS:
        found = False
        for cmd in variants:
            rc, out = _run_command(cmd)
            if rc == 0 or (rc != -1 and "version" in out.lower()):
                results.append(CheckResult(
                    name=f"Tool: {tool_name}",
                    status=PASS,
                    detail=out[:200],
                ))
                found = True
                break
        if not found:
            results.append(CheckResult(
                name=f"Tool: {tool_name}",
                status=FAIL,
                detail=f"Not found (tried: {', '.join(v[0] for v in variants)})",
            ))
    return results


def check_directories(project_root: Path | None = None) -> list[CheckResult]:
    """Check existence and writability of key directories."""
    root = project_root or PROJECT_ROOT
    results: list[CheckResult] = []
    for name, rel_path, needs_writable in KEY_DIRECTORIES:
        path = root / rel_path
        if not path.exists():
            results.append(CheckResult(
                name=f"Directory: {name}",
                status=FAIL,
                detail=f"{path} does not exist",
            ))
        elif needs_writable and not os.access(str(path), os.W_OK):
            results.append(CheckResult(
                name=f"Directory: {name}",
                status=FAIL,
                detail=f"{path} is not writable",
            ))
        else:
            detail = f"{path} exists"
            if needs_writable:
                detail += " (writable)"
            results.append(CheckResult(
                name=f"Directory: {name}",
                status=PASS,
                detail=detail,
            ))
    return results


def check_fasttext_model(project_root: Path | None = None) -> CheckResult:
    """Check if FastText language detection model exists."""
    root = project_root or PROJECT_ROOT
    for model_path in FASTTEXT_MODEL_PATHS:
        p = Path(model_path)
        if not p.is_absolute():
            p = root / model_path
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            return CheckResult(
                name="FastText model",
                status=PASS,
                detail=f"Found at {p} ({size_mb:.1f} MB)",
            )
    return CheckResult(
        name="FastText model",
        status=FAIL,
        detail=f"lid.176.bin not found (checked: {', '.join(FASTTEXT_MODEL_PATHS)})",
    )


def check_gpu_readiness(*, cpu_only: bool = False) -> CheckResult:
    """Check GPU availability via nvidia-smi."""
    if cpu_only:
        return CheckResult(
            name="GPU readiness",
            status=SKIP,
            detail="Skipped (--cpu-only mode)",
        )

    rc, out = _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if rc == 0 and out.strip():
        gpu_count = len(out.strip().splitlines())
        return CheckResult(
            name="GPU readiness",
            status=PASS,
            detail=f"{gpu_count} GPU(s) detected: {out.strip().splitlines()[0]}",
        )
    return CheckResult(
        name="GPU readiness",
        status=FAIL,
        detail=f"nvidia-smi not available or no GPUs: {out}",
    )


def check_port_available(name: str, port: int) -> CheckResult:
    """Check if a port is available (not currently in use)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            result = sock.connect_ex(("127.0.0.1", port))
            if result == 0:
                # Port is in use (something is listening)
                return CheckResult(
                    name=f"Port {port} ({name})",
                    status=PASS,
                    detail=f"Port {port} is in use (service likely running)",
                )
            else:
                return CheckResult(
                    name=f"Port {port} ({name})",
                    status=PASS,
                    detail=f"Port {port} is available (not in use)",
                )
    except Exception as exc:
        return CheckResult(
            name=f"Port {port} ({name})",
            status=FAIL,
            detail=f"Error checking port {port}: {exc}",
        )


def check_ports(ports: list[tuple[str, int]] | None = None) -> list[CheckResult]:
    """Check all configured ports."""
    if ports is None:
        ports = DEFAULT_PORTS
    return [check_port_available(name, port) for name, port in ports]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_text_report(report: PreflightReport) -> str:
    """Format the report as a text table."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EDCOCR Environment Preflight Check")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append("=" * 72)
    lines.append("")

    # Determine column widths
    max_name = max((len(c.name) for c in report.checks), default=20)
    max_status = 6

    lines.append(f"{'Check':<{max_name}}  {'Status':<{max_status}}  Detail")
    lines.append(f"{'-' * max_name}  {'-' * max_status}  {'-' * 40}")

    for check in report.checks:
        lines.append(f"{check.name:<{max_name}}  {check.status:<{max_status}}  {check.detail}")

    lines.append("")
    lines.append(f"{'=' * 72}")

    pass_count = sum(1 for c in report.checks if c.status == PASS)
    fail_count = sum(1 for c in report.checks if c.status == FAIL)
    skip_count = sum(1 for c in report.checks if c.status == SKIP)

    lines.append(
        f"Result: {report.verdict}  "
        f"(PASS={pass_count}, FAIL={fail_count}, SKIP={skip_count})"
    )
    lines.append(f"{'=' * 72}")

    return "\n".join(lines)


def format_json_report(report: PreflightReport) -> str:
    """Format the report as JSON."""
    data = {
        "timestamp": report.timestamp,
        "verdict": report.verdict,
        "checks": [asdict(c) for c in report.checks],
        "summary": {
            "pass": sum(1 for c in report.checks if c.status == PASS),
            "fail": sum(1 for c in report.checks if c.status == FAIL),
            "skip": sum(1 for c in report.checks if c.status == SKIP),
            "total": len(report.checks),
        },
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_preflight(
    *,
    check_ports_flag: bool = False,
    cpu_only: bool = False,
    project_root: Path | None = None,
) -> PreflightReport:
    """Run all preflight checks and return the report."""
    report = PreflightReport()

    # A) Python environment
    report.checks.append(check_python_version())
    report.checks.extend(check_required_packages())

    # B) System tools
    report.checks.extend(check_system_tools())

    # C) Directory structure
    report.checks.extend(check_directories(project_root))

    # D) FastText model
    report.checks.append(check_fasttext_model(project_root))

    # E) GPU readiness
    report.checks.append(check_gpu_readiness(cpu_only=cpu_only))

    # F) Port availability (optional)
    if check_ports_flag:
        report.checks.extend(check_ports())

    report.compute_verdict()
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="EDCOCR environment preflight checker.",
    )
    parser.add_argument(
        "--check-ports",
        action="store_true",
        default=False,
        help="Check availability of ports (8000, 5432, 5672, 6379)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Skip GPU readiness check (CPU-only deployment)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write report to file (markdown or text based on extension)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    report = run_preflight(
        check_ports_flag=args.check_ports,
        cpu_only=args.cpu_only,
    )

    if args.json_output:
        print(format_json_report(report))
    else:
        print(format_text_report(report))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        content = format_json_report(report) if args.report.endswith(".json") else format_text_report(report)
        report_path.write_text(content, encoding="utf-8")
        if not args.json_output:
            print(f"\nReport written to: {report_path}")

    return 0 if report.verdict == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
