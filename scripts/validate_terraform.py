#!/usr/bin/env python3
"""Unified Terraform validation tool for OCR-Local cloud modules.

Validates Terraform module structure, variable definitions, security patterns,
naming conventions, and tagging compliance across EKS, GKE, and OKE modules.

Usage:
    python scripts/validate_terraform.py --environment eks
    python scripts/validate_terraform.py --environment all --strict
    python scripts/validate_terraform.py --environment gke --output-dir reports/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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

ENVIRONMENTS = {
    "eks": "modules/eks",
    "gke": "modules/gke",
    "oke": "modules/oke",
    "shared": "modules/shared",
    "staging": "environments/staging",
    "production": "environments/production",
}

# Credential patterns that should never appear in .tf files
CREDENTIAL_PATTERNS = [
    (r"(?i)access_key\s*=\s*\"[A-Z0-9]{16,}\"", "Hardcoded AWS access key"),
    (r"(?i)secret_key\s*=\s*\"[A-Za-z0-9/+=]{20,}\"", "Hardcoded AWS secret key"),
    (r"(?i)password\s*=\s*\"(?!changeme\"|\"$)[^\"]{8,}\"", "Hardcoded password"),
    (r"(?i)token\s*=\s*\"[A-Za-z0-9_\-]{20,}\"", "Hardcoded token"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID literal"),
    (r"(?i)private_key\s*=\s*\"-----BEGIN", "Embedded private key"),
]

# Required resource naming prefix for OCR-Local
NAMING_PREFIX = "ocr-local"

# Required tags/labels for all resources
REQUIRED_TAG_KEYS_AWS = {"Project", "ManagedBy", "Environment"}
REQUIRED_TAG_KEYS_GCP = {"project", "managed-by"}
REQUIRED_TAG_KEYS_OCI = {"Project", "ManagedBy"}

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single validation finding."""

    rule: str
    severity: str
    message: str
    file: str
    line: Optional[int] = None

    def to_dict(self) -> dict:
        result = {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "file": self.file,
        }
        if self.line is not None:
            result["line"] = self.line
        return result


@dataclass
class ValidationReport:
    """Aggregate validation report."""

    environment: str
    timestamp: str = ""
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    variables_checked: int = 0
    passed: bool = True

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_INFO)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)
        if finding.severity == SEVERITY_ERROR:
            self.passed = False

    def to_dict(self) -> dict:
        return {
            "environment": self.environment,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "summary": {
                "files_scanned": self.files_scanned,
                "variables_checked": self.variables_checked,
                "errors": self.error_count,
                "warnings": self.warning_count,
                "info": self.info_count,
            },
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_markdown(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"# Terraform Validation Report: {self.environment}",
            "",
            f"**Status**: {status}",
            f"**Timestamp**: {self.timestamp}",
            f"**Files scanned**: {self.files_scanned}",
            f"**Variables checked**: {self.variables_checked}",
            "",
            "## Summary",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| Errors | {self.error_count} |",
            f"| Warnings | {self.warning_count} |",
            f"| Info | {self.info_count} |",
            "",
        ]

        if self.findings:
            lines.append("## Findings")
            lines.append("")
            lines.append("| Severity | Rule | File | Message |")
            lines.append("|----------|------|------|---------|")
            for f in self.findings:
                line_ref = f" (L{f.line})" if f.line else ""
                lines.append(
                    f"| {f.severity} | {f.rule} | `{f.file}{line_ref}` | {f.message} |"
                )
            lines.append("")
        else:
            lines.append("No findings. All checks passed.")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_tf_files(module_path: Path) -> list[Path]:
    """Return all .tf files in a module directory."""
    if not module_path.is_dir():
        return []
    return sorted(module_path.glob("*.tf"))


def read_file_lines(path: Path) -> list[str]:
    """Read file and return lines."""
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def parse_variables(lines: list[str]) -> list[dict]:
    """Parse variable blocks from Terraform source lines.

    Returns list of dicts with keys: name, line, has_description, has_type,
    has_default, is_sensitive.
    """
    variables = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^variable\s+"([^"]+)"\s*\{', line)
        if m:
            var = {
                "name": m.group(1),
                "line": i + 1,
                "has_description": False,
                "has_type": False,
                "has_default": False,
                "is_sensitive": False,
            }
            brace_depth = 1
            j = i + 1
            while j < len(lines) and brace_depth > 0:
                inner = lines[j].strip()
                brace_depth += inner.count("{") - inner.count("}")
                if inner.startswith("description"):
                    var["has_description"] = True
                if inner.startswith("type"):
                    var["has_type"] = True
                if inner.startswith("default"):
                    var["has_default"] = True
                if inner.startswith("sensitive") and "true" in inner:
                    var["is_sensitive"] = True
                j += 1
            variables.append(var)
            i = j
        else:
            i += 1
    return variables


def parse_resources(lines: list[str]) -> list[dict]:
    """Parse resource blocks from Terraform source lines.

    Returns list of dicts with keys: type, name, line, content_lines.
    """
    resources = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^resource\s+"([^"]+)"\s+"([^"]+)"\s*\{', line)
        if m:
            res = {
                "type": m.group(1),
                "name": m.group(2),
                "line": i + 1,
                "content_lines": [],
            }
            brace_depth = 1
            j = i + 1
            while j < len(lines) and brace_depth > 0:
                inner = lines[j]
                brace_depth += inner.count("{") - inner.count("}")
                res["content_lines"].append(inner)
                j += 1
            resources.append(res)
            i = j
        else:
            i += 1
    return resources


def parse_terraform_block(lines: list[str]) -> dict:
    """Parse the terraform {} block for version constraints and backend config."""
    result = {
        "has_required_version": False,
        "has_required_providers": False,
        "has_backend": False,
        "backend_commented": False,
        "required_version_line": None,
    }
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("terraform") and "{" in line:
            brace_depth = 1
            j = i + 1
            while j < len(lines) and brace_depth > 0:
                inner = lines[j].strip()
                brace_depth += inner.count("{") - inner.count("}")
                if "required_version" in inner:
                    result["has_required_version"] = True
                    result["required_version_line"] = j + 1
                if "required_providers" in inner:
                    result["has_required_providers"] = True
                if re.match(r'^\s*backend\s+"', inner):
                    result["has_backend"] = True
                if re.match(r'^\s*#\s*backend\s+"', inner):
                    result["backend_commented"] = True
                j += 1
            break
        i += 1
    return result


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def check_variable_descriptions(
    variables: list[dict], tf_file: str, report: ValidationReport
) -> None:
    """Every variable must have a description."""
    for var in variables:
        report.variables_checked += 1
        if not var["has_description"]:
            report.add(
                Finding(
                    rule="VAR-DESC",
                    severity=SEVERITY_ERROR,
                    message=f"Variable '{var['name']}' is missing a description",
                    file=tf_file,
                    line=var["line"],
                )
            )


def check_variable_types(
    variables: list[dict], tf_file: str, report: ValidationReport
) -> None:
    """Every variable must have an explicit type constraint."""
    for var in variables:
        if not var["has_type"]:
            report.add(
                Finding(
                    rule="VAR-TYPE",
                    severity=SEVERITY_ERROR,
                    message=f"Variable '{var['name']}' is missing a type constraint",
                    file=tf_file,
                    line=var["line"],
                )
            )


def check_required_variables(
    variables: list[dict], tf_file: str, report: ValidationReport
) -> None:
    """Flag variables without defaults (required at plan time)."""
    for var in variables:
        if not var["has_default"]:
            report.add(
                Finding(
                    rule="VAR-REQUIRED",
                    severity=SEVERITY_INFO,
                    message=f"Variable '{var['name']}' has no default (required at apply time)",
                    file=tf_file,
                    line=var["line"],
                )
            )


def check_credential_patterns(
    lines: list[str], tf_file: str, report: ValidationReport
) -> None:
    """Scan for hardcoded credentials in .tf files."""
    for i, line in enumerate(lines):
        for pattern, desc in CREDENTIAL_PATTERNS:
            if re.search(pattern, line):
                report.add(
                    Finding(
                        rule="CRED-HARDCODED",
                        severity=SEVERITY_ERROR,
                        message=f"{desc} detected",
                        file=tf_file,
                        line=i + 1,
                    )
                )


def check_provider_version_constraints(
    tf_block: dict, tf_file: str, report: ValidationReport
) -> None:
    """Terraform block must declare required_version and required_providers."""
    if not tf_block["has_required_version"]:
        report.add(
            Finding(
                rule="TF-VERSION",
                severity=SEVERITY_ERROR,
                message="Missing required_version constraint in terraform block",
                file=tf_file,
            )
        )
    if not tf_block["has_required_providers"]:
        # Environments may not have required_providers if they rely on modules
        report.add(
            Finding(
                rule="TF-PROVIDERS",
                severity=SEVERITY_WARNING,
                message="No required_providers block found (expected in modules)",
                file=tf_file,
            )
        )


def check_backend_configuration(
    tf_block: dict, tf_file: str, report: ValidationReport
) -> None:
    """Backend config should exist (even if commented for template modules)."""
    if not tf_block["has_backend"] and not tf_block["backend_commented"]:
        # Only applies to environment-level configs
        if "environments" in tf_file:
            report.add(
                Finding(
                    rule="BACKEND-CFG",
                    severity=SEVERITY_WARNING,
                    message="No backend configuration found (remote state recommended for environments)",
                    file=tf_file,
                )
            )


def check_resource_naming(
    resources: list[dict], tf_file: str, report: ValidationReport
) -> None:
    """Resource names should follow a consistent convention."""
    for res in resources:
        content_text = "\n".join(res["content_lines"])
        # Check that display_name or Name tag contains cluster_name reference
        has_name_ref = (
            "var.cluster_name" in content_text
            or "${var.cluster_name}" in content_text
            or "display_name" in content_text
        )
        # Skip data sources and simple resources
        if res["type"].startswith("aws_iam_role_policy_attachment"):
            continue
        if res["type"].startswith("google_project_iam_member"):
            continue
        if not has_name_ref and "name" not in res["type"]:
            report.add(
                Finding(
                    rule="RES-NAMING",
                    severity=SEVERITY_WARNING,
                    message=(
                        f"Resource '{res['type']}.{res['name']}' may not reference "
                        f"var.cluster_name in its naming"
                    ),
                    file=tf_file,
                    line=res["line"],
                )
            )


def check_resource_tags(
    resources: list[dict],
    tf_file: str,
    report: ValidationReport,
    provider: str,
) -> None:
    """Resources should have tags/labels/freeform_tags."""
    tag_keyword = {
        "eks": "tags",
        "gke": "labels",
        "oke": "freeform_tags",
        "shared": "labels",
        "staging": "tags",
        "production": "tags",
    }.get(provider, "tags")

    # Resource types that typically support tags
    taggable_prefixes = [
        "aws_vpc",
        "aws_subnet",
        "aws_eks_cluster",
        "aws_eks_node_group",
        "aws_ecr_repository",
        "aws_internet_gateway",
        "aws_nat_gateway",
        "aws_eip",
        "aws_route_table",
        "google_compute_network",
        "google_compute_subnetwork",
        "google_container_cluster",
        "google_container_node_pool",
        "google_artifact_registry_repository",
        "oci_core_vcn",
        "oci_core_internet_gateway",
        "oci_core_nat_gateway",
        "oci_core_service_gateway",
        "oci_core_route_table",
        "oci_core_subnet",
        "oci_containerengine_cluster",
        "oci_containerengine_node_pool",
    ]

    for res in resources:
        if not any(res["type"].startswith(p) for p in taggable_prefixes):
            continue
        content_text = "\n".join(res["content_lines"])
        if tag_keyword not in content_text:
            report.add(
                Finding(
                    rule="RES-TAGS",
                    severity=SEVERITY_WARNING,
                    message=(
                        f"Resource '{res['type']}.{res['name']}' is missing "
                        f"{tag_keyword} (tagging/labeling recommended)"
                    ),
                    file=tf_file,
                    line=res["line"],
                )
            )


def check_sensitive_variables(
    variables: list[dict], tf_file: str, report: ValidationReport
) -> None:
    """Variables that look like secrets should be marked sensitive."""
    secret_patterns = ["password", "secret", "token", "private_key", "api_key"]
    for var in variables:
        name_lower = var["name"].lower()
        if any(p in name_lower for p in secret_patterns):
            if not var["is_sensitive"]:
                report.add(
                    Finding(
                        rule="VAR-SENSITIVE",
                        severity=SEVERITY_WARNING,
                        message=(
                            f"Variable '{var['name']}' looks like a secret "
                            f"but is not marked sensitive"
                        ),
                        file=tf_file,
                        line=var["line"],
                    )
                )


def check_placeholder_values(
    lines: list[str], tf_file: str, report: ValidationReport
) -> None:
    """Flag obvious placeholder values that must be replaced."""
    placeholder_patterns = [
        (r"placeholder", "Placeholder value found"),
        (r"REPLACE_ME", "REPLACE_ME placeholder found"),
        (r"TODO", "TODO marker found"),
        (r"FIXME", "FIXME marker found"),
    ]
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip comments for TODO/FIXME (those are legitimate)
        if stripped.startswith("#"):
            continue
        for pattern, desc in placeholder_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                report.add(
                    Finding(
                        rule="PLACEHOLDER",
                        severity=SEVERITY_WARNING,
                        message=desc,
                        file=tf_file,
                        line=i + 1,
                    )
                )


# ---------------------------------------------------------------------------
# Module validation orchestrator
# ---------------------------------------------------------------------------


def validate_module(
    terraform_root: Path,
    module_rel_path: str,
    provider: str,
    report: ValidationReport,
) -> None:
    """Run all validation checks on a single module directory."""
    module_path = terraform_root / module_rel_path
    tf_files = parse_tf_files(module_path)

    if not tf_files:
        report.add(
            Finding(
                rule="MODULE-MISSING",
                severity=SEVERITY_ERROR,
                message=f"No .tf files found in {module_rel_path}",
                file=module_rel_path,
            )
        )
        return

    for tf_file in tf_files:
        report.files_scanned += 1
        rel_name = str(tf_file.relative_to(terraform_root))
        lines = read_file_lines(tf_file)

        # Parse structures
        variables = parse_variables(lines)
        resources = parse_resources(lines)
        tf_block = parse_terraform_block(lines)

        # Run checks
        check_variable_descriptions(variables, rel_name, report)
        check_variable_types(variables, rel_name, report)
        check_required_variables(variables, rel_name, report)
        check_sensitive_variables(variables, rel_name, report)
        check_credential_patterns(lines, rel_name, report)
        check_placeholder_values(lines, rel_name, report)
        check_resource_naming(resources, rel_name, report)
        check_resource_tags(resources, rel_name, report, provider)

        # Terraform block checks (only on main.tf)
        if tf_file.name == "main.tf" and tf_block["has_required_version"]:
            check_provider_version_constraints(tf_block, rel_name, report)
            check_backend_configuration(tf_block, rel_name, report)
        elif tf_file.name == "main.tf":
            # No terraform block at all
            report.add(
                Finding(
                    rule="TF-BLOCK",
                    severity=SEVERITY_WARNING,
                    message="No terraform {} block found in main.tf",
                    file=rel_name,
                )
            )


def resolve_environments(environment: str) -> list[tuple[str, str]]:
    """Map --environment flag to list of (provider, rel_path) tuples."""
    if environment == "all":
        return list(ENVIRONMENTS.items())
    if environment in ENVIRONMENTS:
        return [(environment, ENVIRONMENTS[environment])]
    # Support comma-separated
    result = []
    for env in environment.split(","):
        env = env.strip()
        if env in ENVIRONMENTS:
            result.append((env, ENVIRONMENTS[env]))
    return result


def run_validation(
    terraform_root: Path,
    environment: str,
    strict: bool = False,
) -> ValidationReport:
    """Run full validation and return a report."""
    report = ValidationReport(environment=environment)

    targets = resolve_environments(environment)
    if not targets:
        report.add(
            Finding(
                rule="ENV-UNKNOWN",
                severity=SEVERITY_ERROR,
                message=f"Unknown environment: {environment}. "
                f"Valid: {', '.join(ENVIRONMENTS.keys())}, all",
                file="(argument)",
            )
        )
        return report

    for provider, rel_path in targets:
        validate_module(terraform_root, rel_path, provider, report)

    # In strict mode, warnings also cause failure
    if strict:
        if report.warning_count > 0:
            report.passed = False

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Terraform modules for OCR-Local cloud deployment",
    )
    parser.add_argument(
        "--environment",
        default="all",
        help="Target environment: eks, gke, oke, shared, staging, production, all (default: all)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit code 1)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON + Markdown report output",
    )
    parser.add_argument(
        "--terraform-root",
        default=None,
        help="Path to terraform/ directory (default: auto-detect from repo root)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print JSON report to stdout instead of human-readable output",
    )

    args = parser.parse_args(argv)

    # Resolve terraform root
    if args.terraform_root:
        tf_root = Path(args.terraform_root)
    else:
        # Auto-detect: look for terraform/ relative to script location
        script_dir = Path(__file__).resolve().parent
        tf_root = script_dir.parent / "terraform"
        if not tf_root.is_dir():
            tf_root = Path.cwd() / "terraform"

    if not tf_root.is_dir():
        print(f"ERROR: Terraform root not found at {tf_root}", file=sys.stderr)
        return 2

    report = run_validation(tf_root, args.environment, strict=args.strict)

    # Output
    if args.json_only:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        md = report.to_markdown()
        print(md)
        print(f"\n{'PASSED' if report.passed else 'FAILED'}: "
              f"{report.error_count} errors, {report.warning_count} warnings, "
              f"{report.info_count} info")

    # Write to output directory if requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        json_path = out_dir / f"terraform-validation-{timestamp}.json"
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )

        md_path = out_dir / f"terraform-validation-{timestamp}.md"
        md_path.write_text(report.to_markdown(), encoding="utf-8")

        if not args.json_only:
            print(f"\nReports written to:\n  {json_path}\n  {md_path}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
