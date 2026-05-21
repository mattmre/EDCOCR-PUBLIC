#!/usr/bin/env python3
"""Validate OKE (Oracle Kubernetes Engine) Terraform configurations.

Parses OKE-specific Terraform HCL files and validates Oracle Cloud
Infrastructure patterns including compartment IDs, GPU shapes, flex
shape configurations, network security rules, and node pool settings.

Usage:
    python scripts/validate_terraform_oke.py --terraform-dir terraform/modules/oke
    python scripts/validate_terraform_oke.py --terraform-dir terraform/ --strict
    python scripts/validate_terraform_oke.py --terraform-dir terraform/ --output-dir results/
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("validate_terraform_oke")


# ---------------------------------------------------------------------------
# Known OCI GPU shapes and configurations
# ---------------------------------------------------------------------------

# GPU shapes available on OCI
OCI_GPU_SHAPES = {
    "VM.GPU2.1": {"gpu_count": 1, "gpu_type": "P100", "vram_gb": 16},
    "VM.GPU3.1": {"gpu_count": 1, "gpu_type": "V100", "vram_gb": 16},
    "VM.GPU3.2": {"gpu_count": 2, "gpu_type": "V100", "vram_gb": 32},
    "VM.GPU3.4": {"gpu_count": 4, "gpu_type": "V100", "vram_gb": 64},
    "BM.GPU2.2": {"gpu_count": 2, "gpu_type": "P100", "vram_gb": 32},
    "BM.GPU3.8": {"gpu_count": 8, "gpu_type": "V100", "vram_gb": 128},
    "BM.GPU4.8": {"gpu_count": 8, "gpu_type": "A100", "vram_gb": 320},
    "BM.GPU.A10.4": {"gpu_count": 4, "gpu_type": "A10", "vram_gb": 96},
    "BM.GPU.A100-v2.8": {"gpu_count": 8, "gpu_type": "A100-80G", "vram_gb": 640},
    "VM.GPU.A10.1": {"gpu_count": 1, "gpu_type": "A10", "vram_gb": 24},
    "VM.GPU.A10.2": {"gpu_count": 2, "gpu_type": "A10", "vram_gb": 48},
}

# Flex shapes that require shape_config
OCI_FLEX_SHAPES = {
    "VM.Standard.E3.Flex",
    "VM.Standard.E4.Flex",
    "VM.Standard.E5.Flex",
    "VM.Standard3.Flex",
    "VM.Standard.A1.Flex",
    "VM.Optimized3.Flex",
}

# OCID patterns
OCID_PATTERN = re.compile(
    r"ocid1\.[a-z]+\.[a-z0-9]+\.[a-z0-9-]*\.[a-z0-9]+"
)
COMPARTMENT_OCID_PATTERN = re.compile(
    r"ocid1\.compartment\.[a-z0-9]+\.[a-z0-9-]*\.[a-z0-9]+"
)
PLACEHOLDER_OCID_PATTERN = re.compile(
    r"ocid1\.[a-z]+\.[a-z0-9]+\.\.placeholder"
)

# Minimum Kubernetes version for OKE
MIN_K8S_VERSION = "v1.26.0"

# Valid CIDR pattern
CIDR_PATTERN = re.compile(
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}"
)


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationFinding:
    """A single validation finding."""

    rule: str
    severity: str  # "error", "warning", "info"
    message: str
    file: str = ""
    line: int = 0
    resource: str = ""


@dataclass
class ValidationReport:
    """Complete validation report."""

    terraform_dir: str = ""
    timestamp: str = ""
    files_scanned: int = 0
    findings: list = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    passed: bool = True
    strict_mode: bool = False


# ---------------------------------------------------------------------------
# HCL parsing helpers (lightweight, no external dependency)
# ---------------------------------------------------------------------------


def find_tf_files(terraform_dir: str) -> list[Path]:
    """Find all .tf files in the given directory tree.

    Parameters
    ----------
    terraform_dir : str
        Root directory to search.

    Returns
    -------
    list[Path]
        List of .tf file paths.
    """
    tf_dir = Path(terraform_dir)
    if not tf_dir.is_dir():
        return []
    return sorted(tf_dir.rglob("*.tf"))


def extract_resource_blocks(content: str) -> list[dict]:
    """Extract resource block metadata from Terraform HCL content.

    This is a lightweight regex-based parser for common patterns. It does
    not implement full HCL parsing but catches the patterns we need to
    validate for OKE configurations.

    Parameters
    ----------
    content : str
        Terraform HCL file content.

    Returns
    -------
    list[dict]
        Extracted resource block information.
    """
    blocks = []
    # Match resource "type" "name" { ... }
    resource_pattern = re.compile(
        r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{', re.MULTILINE
    )
    for match in resource_pattern.finditer(content):
        resource_type = match.group(1)
        resource_name = match.group(2)
        start = match.start()
        line_num = content[:start].count("\n") + 1

        # Find the block content (simple brace matching)
        brace_count = 0
        block_start = content.index("{", match.start())
        pos = block_start
        for i, ch in enumerate(content[block_start:], block_start):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    pos = i
                    break

        block_content = content[block_start:pos + 1]

        blocks.append({
            "type": resource_type,
            "name": resource_name,
            "line": line_num,
            "content": block_content,
        })

    return blocks


def extract_variable_blocks(content: str) -> list[dict]:
    """Extract variable definitions from Terraform HCL content.

    Parameters
    ----------
    content : str
        Terraform HCL file content.

    Returns
    -------
    list[dict]
        Variable definitions with name, type, default, etc.
    """
    variables = []
    var_pattern = re.compile(
        r'variable\s+"([^"]+)"\s*\{', re.MULTILINE
    )
    for match in var_pattern.finditer(content):
        var_name = match.group(1)
        start = match.start()
        line_num = content[:start].count("\n") + 1

        # Find block content
        brace_count = 0
        block_start = content.index("{", start)
        pos = block_start
        for i, ch in enumerate(content[block_start:], block_start):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    pos = i
                    break

        block_content = content[block_start:pos + 1]

        # Extract default value
        default_match = re.search(
            r'default\s*=\s*"?([^"\n]+)"?', block_content
        )
        default_val = default_match.group(1).strip() if default_match else None

        # Extract type
        type_match = re.search(r'type\s*=\s*(\S+)', block_content)
        type_val = type_match.group(1).strip() if type_match else None

        variables.append({
            "name": var_name,
            "line": line_num,
            "default": default_val,
            "type": type_val,
            "content": block_content,
        })

    return variables


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def validate_compartment_id(
    content: str,
    file_path: str,
    strict: bool = False,
) -> list[ValidationFinding]:
    """Check compartment_id references for valid OCID format.

    Parameters
    ----------
    content : str
        Terraform file content.
    file_path : str
        Path to the file being validated.
    strict : bool
        If True, flag placeholder OCIDs as errors.

    Returns
    -------
    list[ValidationFinding]
        Findings from compartment ID validation.
    """
    findings = []

    # Check for hardcoded compartment IDs
    for i, line in enumerate(content.split("\n"), 1):
        if "compartment_id" in line and "var." not in line:
            if COMPARTMENT_OCID_PATTERN.search(line):
                findings.append(ValidationFinding(
                    rule="OKE-CO",
                    severity="warning",
                    message="Hardcoded compartment OCID found; use variable reference",
                    file=file_path,
                    line=i,
                ))

    return findings


def validate_gpu_shapes(
    resources: list[dict],
    file_path: str,
    strict: bool = False,
) -> list[ValidationFinding]:
    """Validate GPU node pool shape configurations.

    Parameters
    ----------
    resources : list[dict]
        Extracted resource blocks.
    file_path : str
        Path to the file being validated.
    strict : bool
        If True, flag unknown shapes as errors.

    Returns
    -------
    list[ValidationFinding]
        Findings from GPU shape validation.
    """
    findings = []

    for resource in resources:
        if resource["type"] != "oci_containerengine_node_pool":
            continue

        content = resource["content"]

        # Check node_shape
        shape_match = re.search(r'node_shape\s*=\s*(?:var\.(\w+)|"([^"]+)")', content)
        if shape_match:
            _shape_var = shape_match.group(1)  # noqa: F841
            shape_val = shape_match.group(2)

            if shape_val and "GPU" in shape_val.upper():
                if shape_val not in OCI_GPU_SHAPES:
                    findings.append(ValidationFinding(
                        rule="OKE-GPU-001",
                        severity="error" if strict else "warning",
                        message=f"Unknown GPU shape '{shape_val}'; "
                                f"known shapes: {', '.join(sorted(OCI_GPU_SHAPES.keys()))}",
                        file=file_path,
                        line=resource["line"],
                        resource=f"{resource['type']}.{resource['name']}",
                    ))

        # Check for node_shape_config on GPU shapes (should NOT be present for
        # bare metal GPU shapes)
        if "node_shape_config" in content:
            if shape_val and shape_val.startswith("BM."):
                findings.append(ValidationFinding(
                    rule="OKE-GPU-002",
                    severity="warning",
                    message="node_shape_config is not applicable for bare metal shapes",
                    file=file_path,
                    line=resource["line"],
                    resource=f"{resource['type']}.{resource['name']}",
                ))

    return findings


def validate_flex_shapes(
    resources: list[dict],
    variables: list[dict],
    file_path: str,
) -> list[ValidationFinding]:
    """Validate flex shape configurations have required shape_config.

    Parameters
    ----------
    resources : list[dict]
        Extracted resource blocks.
    variables : list[dict]
        Extracted variable definitions.
    file_path : str
        Path to the file being validated.

    Returns
    -------
    list[ValidationFinding]
        Findings from flex shape validation.
    """
    findings = []

    # Build map of variable defaults
    var_defaults = {v["name"]: v["default"] for v in variables}

    for resource in resources:
        if resource["type"] != "oci_containerengine_node_pool":
            continue

        content = resource["content"]

        shape_match = re.search(r'node_shape\s*=\s*(?:var\.(\w+)|"([^"]+)")', content)
        if not shape_match:
            continue

        shape_var = shape_match.group(1)
        shape_val = shape_match.group(2)

        # Resolve variable reference
        if shape_var and shape_var in var_defaults:
            shape_val = var_defaults[shape_var]

        if shape_val and shape_val in OCI_FLEX_SHAPES:
            if "node_shape_config" not in content:
                findings.append(ValidationFinding(
                    rule="OKE-FLEX-001",
                    severity="error",
                    message=f"Flex shape '{shape_val}' requires node_shape_config "
                            f"with ocpus and memory_in_gbs",
                    file=file_path,
                    line=resource["line"],
                    resource=f"{resource['type']}.{resource['name']}",
                ))
            else:
                # Validate shape_config has required fields
                if "ocpus" not in content:
                    findings.append(ValidationFinding(
                        rule="OKE-FLEX-002",
                        severity="error",
                        message="Flex shape node_shape_config missing 'ocpus'",
                        file=file_path,
                        line=resource["line"],
                        resource=f"{resource['type']}.{resource['name']}",
                    ))
                if "memory_in_gbs" not in content:
                    findings.append(ValidationFinding(
                        rule="OKE-FLEX-003",
                        severity="error",
                        message="Flex shape node_shape_config missing 'memory_in_gbs'",
                        file=file_path,
                        line=resource["line"],
                        resource=f"{resource['type']}.{resource['name']}",
                    ))

    return findings


def validate_network_security(
    resources: list[dict],
    file_path: str,
    strict: bool = False,
) -> list[ValidationFinding]:
    """Validate network security list configurations.

    Parameters
    ----------
    resources : list[dict]
        Extracted resource blocks.
    file_path : str
        Path to the file being validated.
    strict : bool
        If True, flag open ingress rules as errors.

    Returns
    -------
    list[ValidationFinding]
        Findings from network security validation.
    """
    findings = []

    for resource in resources:
        if resource["type"] != "oci_core_security_list":
            continue

        content = resource["content"]

        # Check for overly permissive ingress (0.0.0.0/0 on all protocols)
        ingress_blocks = re.findall(
            r'ingress_security_rules\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}',
            content,
            re.DOTALL,
        )

        for block in ingress_blocks:
            if '0.0.0.0/0' in block and 'protocol  = "all"' in block:
                findings.append(ValidationFinding(
                    rule="OKE-NET-001",
                    severity="error" if strict else "warning",
                    message="Ingress rule allows all protocols from 0.0.0.0/0; "
                            "restrict to specific ports/protocols",
                    file=file_path,
                    line=resource["line"],
                    resource=f"{resource['type']}.{resource['name']}",
                ))

        # Check for Kubernetes API access restrictions
        if "api" in resource["name"].lower():
            if '0.0.0.0/0' in content and "6443" in content:
                findings.append(ValidationFinding(
                    rule="OKE-NET-002",
                    severity="warning",
                    message="Kubernetes API (6443) is open to 0.0.0.0/0; "
                            "consider restricting to known CIDRs",
                    file=file_path,
                    line=resource["line"],
                    resource=f"{resource['type']}.{resource['name']}",
                ))

    return findings


def validate_placeholder_ocids(
    content: str,
    file_path: str,
    strict: bool = False,
) -> list[ValidationFinding]:
    """Check for placeholder OCIDs that need replacement.

    Parameters
    ----------
    content : str
        Terraform file content.
    file_path : str
        Path to the file being validated.
    strict : bool
        If True, flag placeholders as errors.

    Returns
    -------
    list[ValidationFinding]
        Findings from placeholder OCID check.
    """
    findings = []

    for i, line in enumerate(content.split("\n"), 1):
        if PLACEHOLDER_OCID_PATTERN.search(line):
            findings.append(ValidationFinding(
                rule="OKE-OCID-001",
                severity="error" if strict else "warning",
                message="Placeholder OCID found; replace with actual "
                        "resource OCID before deployment",
                file=file_path,
                line=i,
            ))

    return findings


def validate_kubernetes_version(
    content: str,
    variables: list[dict],
    file_path: str,
) -> list[ValidationFinding]:
    """Validate Kubernetes version meets minimum requirements.

    Parameters
    ----------
    content : str
        Terraform file content.
    variables : list[dict]
        Extracted variable definitions.
    file_path : str
        Path to the file being validated.

    Returns
    -------
    list[ValidationFinding]
        Findings from version validation.
    """
    findings = []

    for var in variables:
        if var["name"] == "kubernetes_version" and var["default"]:
            version = var["default"].strip('"').lstrip("v")
            min_version = MIN_K8S_VERSION.lstrip("v")

            try:
                v_parts = [int(x) for x in version.split(".")]
                min_parts = [int(x) for x in min_version.split(".")]

                if v_parts < min_parts:
                    findings.append(ValidationFinding(
                        rule="OKE-VER-001",
                        severity="error",
                        message=f"Kubernetes version {version} is below minimum "
                                f"{MIN_K8S_VERSION} for OKE",
                        file=file_path,
                        line=var["line"],
                    ))
            except ValueError:
                findings.append(ValidationFinding(
                    rule="OKE-VER-002",
                    severity="warning",
                    message=f"Cannot parse Kubernetes version: {version}",
                    file=file_path,
                    line=var["line"],
                ))

    return findings


def validate_boot_volume(
    resources: list[dict],
    file_path: str,
) -> list[ValidationFinding]:
    """Validate boot volume sizes for GPU nodes.

    Parameters
    ----------
    resources : list[dict]
        Extracted resource blocks.
    file_path : str
        Path to the file being validated.

    Returns
    -------
    list[ValidationFinding]
        Findings from boot volume validation.
    """
    findings = []

    for resource in resources:
        if resource["type"] != "oci_containerengine_node_pool":
            continue

        content = resource["content"]

        # Check if GPU node pool
        if "gpu" not in resource["name"].lower():
            continue

        boot_vol_match = re.search(
            r'boot_volume_size_in_gbs\s*=\s*(?:var\.(\w+)|(\d+))',
            content,
        )

        if boot_vol_match:
            size_str = boot_vol_match.group(2)
            if size_str:
                size = int(size_str)
                if size < 100:
                    findings.append(ValidationFinding(
                        rule="OKE-BOOT-001",
                        severity="warning",
                        message=f"GPU node boot volume ({size} GB) may be too small; "
                                f"recommend >= 100 GB for container images",
                        file=file_path,
                        line=resource["line"],
                        resource=f"{resource['type']}.{resource['name']}",
                    ))
        else:
            findings.append(ValidationFinding(
                rule="OKE-BOOT-002",
                severity="info",
                message="GPU node pool has no explicit boot_volume_size_in_gbs; "
                        "default (50 GB) may be insufficient",
                file=file_path,
                line=resource["line"],
                resource=f"{resource['type']}.{resource['name']}",
            ))

    return findings


def validate_node_labels(
    resources: list[dict],
    file_path: str,
) -> list[ValidationFinding]:
    """Validate node pools have appropriate labels for scheduling.

    Parameters
    ----------
    resources : list[dict]
        Extracted resource blocks.
    file_path : str
        Path to the file being validated.

    Returns
    -------
    list[ValidationFinding]
        Findings from node label validation.
    """
    findings = []

    for resource in resources:
        if resource["type"] != "oci_containerengine_node_pool":
            continue

        content = resource["content"]

        if "initial_node_labels" not in content:
            findings.append(ValidationFinding(
                rule="OKE-LABEL-001",
                severity="warning",
                message="Node pool has no initial_node_labels; "
                        "labels are required for pod scheduling",
                file=file_path,
                line=resource["line"],
                resource=f"{resource['type']}.{resource['name']}",
            ))

        # GPU pools should have nvidia.com/gpu label
        if "gpu" in resource["name"].lower():
            if "nvidia.com/gpu" not in content:
                findings.append(ValidationFinding(
                    rule="OKE-LABEL-002",
                    severity="warning",
                    message="GPU node pool missing 'nvidia.com/gpu' label; "
                            "GPU scheduling may not work correctly",
                    file=file_path,
                    line=resource["line"],
                    resource=f"{resource['type']}.{resource['name']}",
                ))

    return findings


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------


def validate_oke_terraform(
    terraform_dir: str,
    strict: bool = False,
) -> ValidationReport:
    """Run all OKE Terraform validations.

    Parameters
    ----------
    terraform_dir : str
        Path to Terraform configuration directory.
    strict : bool
        If True, treat warnings as errors.

    Returns
    -------
    ValidationReport
        Complete validation report.
    """
    report = ValidationReport(
        terraform_dir=terraform_dir,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        strict_mode=strict,
    )

    tf_files = find_tf_files(terraform_dir)
    report.files_scanned = len(tf_files)

    if not tf_files:
        report.findings.append(asdict(ValidationFinding(
            rule="OKE-FILE-001",
            severity="error",
            message=f"No .tf files found in {terraform_dir}",
        )))
        report.error_count = 1
        report.passed = False
        return report

    all_findings: list[ValidationFinding] = []

    for tf_file in tf_files:
        file_path = str(tf_file)

        try:
            content = tf_file.read_text(encoding="utf-8")
        except Exception as exc:
            all_findings.append(ValidationFinding(
                rule="OKE-FILE-002",
                severity="error",
                message=f"Cannot read file: {exc}",
                file=file_path,
            ))
            continue

        # Skip non-OKE files (but still check placeholders)
        is_oke_file = any(
            keyword in content
            for keyword in [
                "oci_containerengine",
                "oci_core_vcn",
                "oci_core_subnet",
                "oci_core_security_list",
                "compartment_id",
            ]
        )

        resources = extract_resource_blocks(content)
        variables = extract_variable_blocks(content)

        # Always check placeholders
        all_findings.extend(
            validate_placeholder_ocids(content, file_path, strict)
        )

        if not is_oke_file:
            continue

        # OKE-specific validations
        all_findings.extend(
            validate_compartment_id(content, file_path, strict)
        )
        all_findings.extend(
            validate_gpu_shapes(resources, file_path, strict)
        )
        all_findings.extend(
            validate_flex_shapes(resources, variables, file_path)
        )
        all_findings.extend(
            validate_network_security(resources, file_path, strict)
        )
        all_findings.extend(
            validate_kubernetes_version(content, variables, file_path)
        )
        all_findings.extend(
            validate_boot_volume(resources, file_path)
        )
        all_findings.extend(
            validate_node_labels(resources, file_path)
        )

    # Compile report
    report.findings = [asdict(f) for f in all_findings]
    report.error_count = sum(1 for f in all_findings if f.severity == "error")
    report.warning_count = sum(1 for f in all_findings if f.severity == "warning")
    report.info_count = sum(1 for f in all_findings if f.severity == "info")

    if strict:
        report.passed = report.error_count == 0 and report.warning_count == 0
    else:
        report.passed = report.error_count == 0

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report_markdown(report: ValidationReport) -> str:
    """Format validation report as markdown.

    Parameters
    ----------
    report : ValidationReport
        Validation results.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    status = "PASS" if report.passed else "FAIL"
    lines = [
        "# OKE Terraform Validation Report",
        "",
        f"**Directory**: {report.terraform_dir}",
        f"**Files Scanned**: {report.files_scanned}",
        f"**Strict Mode**: {report.strict_mode}",
        f"**Status**: {status}",
        f"**Timestamp**: {report.timestamp}",
        "",
        "## Summary",
        "",
        f"- Errors: {report.error_count}",
        f"- Warnings: {report.warning_count}",
        f"- Info: {report.info_count}",
        "",
    ]

    if report.findings:
        lines.extend([
            "## Findings",
            "",
            "| Severity | Rule | File | Line | Message |",
            "|----------|------|------|------|---------|",
        ])

        for f in report.findings:
            file_short = Path(f.get("file", "")).name or "-"
            lines.append(
                f"| {f.get('severity', '')} "
                f"| {f.get('rule', '')} "
                f"| {file_short} "
                f"| {f.get('line', '')} "
                f"| {f.get('message', '')} |"
            )
        lines.append("")
    else:
        lines.extend([
            "## Findings",
            "",
            "No findings. All checks passed.",
            "",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for OKE Terraform validation."""
    parser = argparse.ArgumentParser(
        description="Validate OKE Terraform configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_terraform_oke.py --terraform-dir terraform/modules/oke
  python scripts/validate_terraform_oke.py --terraform-dir terraform/ --strict
  python scripts/validate_terraform_oke.py --terraform-dir terraform/ --output-dir results/
        """,
    )
    parser.add_argument(
        "--terraform-dir",
        type=str,
        required=True,
        help="Path to Terraform configuration directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for output reports (JSON + markdown)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    report = validate_oke_terraform(
        terraform_dir=args.terraform_dir,
        strict=args.strict,
    )

    # Print markdown report
    md_report = format_report_markdown(report)
    print(md_report)

    # Save outputs
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "oke-terraform-validation.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("JSON report saved to %s", json_path)

        md_path = out_dir / "oke-terraform-validation.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        logger.info("Markdown report saved to %s", md_path)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
