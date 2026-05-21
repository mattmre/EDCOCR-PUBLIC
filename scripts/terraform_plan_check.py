#!/usr/bin/env python3
"""Terraform plan-mode dry-run validator for OCR-Local.

Wraps `terraform init -backend=false`, `terraform validate`, and
`terraform plan -input=false` to perform structural validation of
Terraform configurations without requiring cloud credentials.

Usage:
    python scripts/terraform_plan_check.py --environment staging
    python scripts/terraform_plan_check.py --environment production --skip-plan
    python scripts/terraform_plan_check.py --environment staging --output-dir reports/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERRAFORM_ROOT = Path("terraform")

ENVIRONMENT_PATHS = {
    "staging": "environments/staging",
    "production": "environments/production",
}

# Known plan output patterns
PLAN_RESOURCE_PATTERN = re.compile(
    r"^\s*#\s+([\w.]+)\s+will be (created|destroyed|updated|replaced)"
)
PLAN_SUMMARY_PATTERN = re.compile(
    r"Plan:\s+(\d+)\s+to add,\s+(\d+)\s+to change,\s+(\d+)\s+to destroy"
)
NO_CHANGES_PATTERN = re.compile(r"No changes\.\s+Your infrastructure matches")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """Result of a subprocess execution."""

    command: str
    returncode: int
    stdout: str
    stderr: str
    success: bool = False

    def __post_init__(self):
        self.success = self.returncode == 0


@dataclass
class PlanAnalysis:
    """Parsed analysis of a terraform plan."""

    to_add: int = 0
    to_change: int = 0
    to_destroy: int = 0
    resources: list[dict] = field(default_factory=list)
    no_changes: bool = False
    has_destroys: bool = False
    raw_output: str = ""

    def to_dict(self) -> dict:
        return {
            "to_add": self.to_add,
            "to_change": self.to_change,
            "to_destroy": self.to_destroy,
            "no_changes": self.no_changes,
            "has_destroys": self.has_destroys,
            "resource_count": len(self.resources),
            "resources": self.resources,
        }


@dataclass
class PlanCheckReport:
    """Full plan check report."""

    environment: str
    timestamp: str = ""
    terraform_available: bool = False
    terraform_version: str = ""
    init_result: Optional[CommandResult] = None
    validate_result: Optional[CommandResult] = None
    plan_result: Optional[CommandResult] = None
    plan_analysis: Optional[PlanAnalysis] = None
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        result = {
            "environment": self.environment,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "terraform_available": self.terraform_available,
            "terraform_version": self.terraform_version,
            "errors": self.errors,
            "warnings": self.warnings,
        }
        if self.init_result:
            result["init"] = {
                "success": self.init_result.success,
                "returncode": self.init_result.returncode,
            }
        if self.validate_result:
            result["validate"] = {
                "success": self.validate_result.success,
                "returncode": self.validate_result.returncode,
                "output": self.validate_result.stdout[:2000],
            }
        if self.plan_analysis:
            result["plan_analysis"] = self.plan_analysis.to_dict()
        return result

    def to_markdown(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"# Terraform Plan Check: {self.environment}",
            "",
            f"**Status**: {status}",
            f"**Timestamp**: {self.timestamp}",
            f"**Terraform**: {'Available' if self.terraform_available else 'NOT FOUND'}",
        ]
        if self.terraform_version:
            lines.append(f"**Version**: {self.terraform_version}")
        lines.append("")

        # Steps
        lines.append("## Steps")
        lines.append("")

        if self.init_result:
            init_icon = "OK" if self.init_result.success else "FAIL"
            lines.append(f"1. **Init** ({init_icon}): `terraform init -backend=false`")
        if self.validate_result:
            val_icon = "OK" if self.validate_result.success else "FAIL"
            lines.append(f"2. **Validate** ({val_icon}): `terraform validate`")
        if self.plan_result:
            plan_icon = "OK" if self.plan_result.success else "FAIL"
            lines.append(f"3. **Plan** ({plan_icon}): `terraform plan -input=false`")
        lines.append("")

        # Plan analysis
        if self.plan_analysis:
            lines.append("## Plan Analysis")
            lines.append("")
            if self.plan_analysis.no_changes:
                lines.append("No changes detected.")
            else:
                lines.append(
                    f"- **Add**: {self.plan_analysis.to_add}"
                )
                lines.append(
                    f"- **Change**: {self.plan_analysis.to_change}"
                )
                lines.append(
                    f"- **Destroy**: {self.plan_analysis.to_destroy}"
                )
            lines.append("")

            if self.plan_analysis.has_destroys:
                lines.append(
                    "**WARNING**: Plan includes destroy operations!"
                )
                lines.append("")

        # Errors and warnings
        if self.errors:
            lines.append("## Errors")
            lines.append("")
            for e in self.errors:
                lines.append(f"- {e}")
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run_command(
    cmd: list[str],
    cwd: Optional[Path] = None,
    timeout: int = 300,
    env: Optional[dict] = None,
) -> CommandResult:
    """Run a subprocess command and capture output."""
    cmd_str = " ".join(cmd)
    logger.info("Running: %s", cmd_str)

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
        )
        return CommandResult(
            command=cmd_str,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            command=cmd_str,
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
        )
    except FileNotFoundError:
        return CommandResult(
            command=cmd_str,
            returncode=-2,
            stdout="",
            stderr="Command not found",
        )


def find_terraform_binary() -> Optional[str]:
    """Locate the terraform binary on PATH."""
    return shutil.which("terraform")


def get_terraform_version(terraform_bin: str) -> str:
    """Get terraform version string."""
    result = run_command([terraform_bin, "version", "-json"], timeout=30)
    if result.success:
        try:
            data = json.loads(result.stdout)
            return data.get("terraform_version", "unknown")
        except (json.JSONDecodeError, KeyError):
            # Fall back to parsing text output
            result2 = run_command([terraform_bin, "version"], timeout=30)
            if result2.success:
                m = re.search(r"Terraform v([\d.]+)", result2.stdout)
                if m:
                    return m.group(1)
    return "unknown"


# ---------------------------------------------------------------------------
# Plan analysis
# ---------------------------------------------------------------------------


def analyze_plan_output(stdout: str) -> PlanAnalysis:
    """Parse terraform plan output into structured analysis."""
    analysis = PlanAnalysis(raw_output=stdout)

    # Check for no-changes
    if NO_CHANGES_PATTERN.search(stdout):
        analysis.no_changes = True
        return analysis

    # Parse individual resource actions
    for line in stdout.splitlines():
        m = PLAN_RESOURCE_PATTERN.match(line)
        if m:
            analysis.resources.append({
                "address": m.group(1),
                "action": m.group(2),
            })

    # Parse summary line
    m = PLAN_SUMMARY_PATTERN.search(stdout)
    if m:
        analysis.to_add = int(m.group(1))
        analysis.to_change = int(m.group(2))
        analysis.to_destroy = int(m.group(3))
    else:
        # Count from individual resources
        analysis.to_add = sum(
            1 for r in analysis.resources if r["action"] == "created"
        )
        analysis.to_change = sum(
            1 for r in analysis.resources if r["action"] == "updated"
        )
        analysis.to_destroy = sum(
            1 for r in analysis.resources if r["action"] == "destroyed"
        )

    analysis.has_destroys = analysis.to_destroy > 0

    return analysis


# ---------------------------------------------------------------------------
# Plan check orchestrator
# ---------------------------------------------------------------------------


def run_plan_check(
    terraform_root: Path,
    environment: str,
    skip_plan: bool = False,
) -> PlanCheckReport:
    """Run terraform init + validate + plan for an environment."""
    report = PlanCheckReport(environment=environment)

    # Resolve environment path
    if environment not in ENVIRONMENT_PATHS:
        report.add_error(
            f"Unknown environment: {environment}. "
            f"Valid: {', '.join(ENVIRONMENT_PATHS.keys())}"
        )
        return report

    env_path = terraform_root / ENVIRONMENT_PATHS[environment]
    if not env_path.is_dir():
        report.add_error(f"Environment directory not found: {env_path}")
        return report

    # Check terraform binary
    terraform_bin = find_terraform_binary()
    if not terraform_bin:
        report.terraform_available = False
        report.add_error(
            "Terraform binary not found on PATH. "
            "Install from https://www.terraform.io/downloads "
            "or use `validate_terraform.py` for offline validation."
        )
        return report

    report.terraform_available = True
    report.terraform_version = get_terraform_version(terraform_bin)

    # Step 1: terraform init -backend=false
    init_result = run_command(
        [terraform_bin, "init", "-backend=false", "-no-color"],
        cwd=env_path,
        timeout=120,
    )
    report.init_result = init_result

    if not init_result.success:
        report.add_error(
            f"terraform init failed (exit {init_result.returncode}): "
            f"{init_result.stderr[:500]}"
        )
        return report

    # Step 2: terraform validate
    validate_result = run_command(
        [terraform_bin, "validate", "-no-color"],
        cwd=env_path,
        timeout=60,
    )
    report.validate_result = validate_result

    if not validate_result.success:
        report.add_error(
            f"terraform validate failed (exit {validate_result.returncode}): "
            f"{validate_result.stdout[:500]} {validate_result.stderr[:500]}"
        )
        return report

    # Step 3: terraform plan (optional)
    if not skip_plan:
        # Plan without credentials will likely fail for cloud providers,
        # but can still validate variable references and module structure
        plan_result = run_command(
            [terraform_bin, "plan", "-input=false", "-no-color"],
            cwd=env_path,
            timeout=300,
        )
        report.plan_result = plan_result

        if plan_result.success:
            report.plan_analysis = analyze_plan_output(plan_result.stdout)
            if report.plan_analysis.has_destroys:
                report.add_warning(
                    f"Plan includes {report.plan_analysis.to_destroy} destroy "
                    f"operation(s) -- review carefully"
                )
        else:
            # Plan failure without credentials is expected -- record but
            # do not fail the check if it is a credential/provider error
            stderr_lower = plan_result.stderr.lower() + plan_result.stdout.lower()
            credential_errors = [
                "no valid credential",
                "authentication",
                "unauthorized",
                "access denied",
                "no credentials",
                "could not load plugin",
                "provider registry",
            ]
            is_credential_error = any(e in stderr_lower for e in credential_errors)

            if is_credential_error:
                report.add_warning(
                    "terraform plan failed due to missing credentials "
                    "(expected for offline validation)"
                )
            else:
                report.add_error(
                    f"terraform plan failed (exit {plan_result.returncode}): "
                    f"{plan_result.stderr[:500]}"
                )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Terraform plan-mode dry-run validator for OCR-Local",
    )
    parser.add_argument(
        "--environment",
        default="staging",
        choices=list(ENVIRONMENT_PATHS.keys()),
        help="Target environment (default: staging)",
    )
    parser.add_argument(
        "--skip-plan",
        action="store_true",
        help="Skip terraform plan (only run init + validate)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON + Markdown report output",
    )
    parser.add_argument(
        "--terraform-root",
        default=None,
        help="Path to terraform/ directory (default: auto-detect)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print JSON report to stdout",
    )

    args = parser.parse_args(argv)

    # Resolve terraform root
    if args.terraform_root:
        tf_root = Path(args.terraform_root)
    else:
        script_dir = Path(__file__).resolve().parent
        tf_root = script_dir.parent / "terraform"
        if not tf_root.is_dir():
            tf_root = Path.cwd() / "terraform"

    if not tf_root.is_dir():
        print(f"ERROR: Terraform root not found at {tf_root}", file=sys.stderr)
        return 2

    report = run_plan_check(tf_root, args.environment, skip_plan=args.skip_plan)

    # Output
    if args.json_only:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_markdown())
        print(f"\n{'PASSED' if report.passed else 'FAILED'}: "
              f"{len(report.errors)} errors, {len(report.warnings)} warnings")

    # Write reports
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        json_path = out_dir / f"terraform-plan-check-{timestamp}.json"
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )

        md_path = out_dir / f"terraform-plan-check-{timestamp}.md"
        md_path.write_text(report.to_markdown(), encoding="utf-8")

        if not args.json_only:
            print(f"\nReports written to:\n  {json_path}\n  {md_path}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
