#!/usr/bin/env python3
"""GKE-specific Terraform validation for OCR-Local cloud deployment.

Validates Google Kubernetes Engine Terraform configurations against
best-practice rules for security, cost, and OCR workload suitability.

Categories:
  - Node pool configuration (machine types, GPU pools, disk sizing)
  - GKE best practices (workload identity, VPC-native, private cluster)
  - GKE autopilot vs standard mode detection
  - IAM bindings for least-privilege
  - Firewall rules and network policies
  - Recommended labels (team, environment, cost-center)

Usage:
  python scripts/validate_terraform_gke.py --terraform-dir terraform/
  python scripts/validate_terraform_gke.py --terraform-dir terraform/ --strict
  python scripts/validate_terraform_gke.py --terraform-dir terraform/ --output-dir reports/

Run with: python scripts/validate_terraform_gke.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recommended GKE GPU accelerator types for OCR workloads
RECOMMENDED_GPU_TYPES = {
    "nvidia-tesla-t4",
    "nvidia-l4",
    "nvidia-tesla-a100",
    "nvidia-a100-80gb",
    "nvidia-tesla-v100",
}

# Machine types suitable for GPU workloads (N1 and A2 families)
GPU_COMPATIBLE_MACHINE_FAMILIES = {"n1", "n2", "a2", "g2"}

# Machine types suitable for CPU workloads
CPU_MACHINE_FAMILIES = {"e2", "n2", "n2d", "c2", "c2d", "t2d", "t2a"}

# Minimum recommended disk size for GPU nodes (GB) -- models need space
MIN_GPU_DISK_SIZE_GB = 100

# Required labels for cost tracking and operations
REQUIRED_LABELS = {"project", "managed-by"}
RECOMMENDED_LABELS = {"environment", "team", "cost-center"}

# GKE release channels in order of stability
VALID_RELEASE_CHANNELS = {"RAPID", "REGULAR", "STABLE"}

# IAM roles that should be present for least-privilege node SA
REQUIRED_NODE_IAM_ROLES = {
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/artifactregistry.reader",
}

# IAM roles that are overly permissive and should be avoided
OVERLY_PERMISSIVE_ROLES = {
    "roles/editor",
    "roles/owner",
    "roles/compute.admin",
    "roles/container.admin",
}

# Regex to extract HCL blocks
RE_RESOURCE_BLOCK = re.compile(
    r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{', re.MULTILINE
)
RE_VARIABLE_BLOCK = re.compile(
    r'variable\s+"([^"]+)"\s*\{', re.MULTILINE
)
RE_STRING_VALUE = re.compile(
    r'([\w-]+)\s*=\s*"([^"]*)"', re.MULTILINE
)
RE_NUMBER_VALUE = re.compile(
    r'(\w+)\s*=\s*(\d+)', re.MULTILINE
)
RE_BOOL_VALUE = re.compile(
    r'(\w+)\s*=\s*(true|false)', re.MULTILINE
)
RE_BLOCK_START = re.compile(
    r'(\w+)\s*\{', re.MULTILINE
)
RE_ROLE_VALUE = re.compile(
    r'role\s*=\s*"([^"]*)"', re.MULTILINE
)
RE_DEFAULT_VALUE = re.compile(
    r'default\s*=\s*"([^"]*)"', re.MULTILINE
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Validation finding severity level."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    """A single validation finding."""

    rule_id: str
    severity: Severity
    message: str
    file: str = ""
    line: int = 0
    recommendation: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class ValidationReport:
    """Aggregated validation report."""

    terraform_dir: str
    timestamp: str = ""
    findings: list[Finding] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.INFO)

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict:
        return {
            "terraform_dir": self.terraform_dir,
            "timestamp": self.timestamp,
            "summary": {
                "total_findings": len(self.findings),
                "errors": self.error_count,
                "warnings": self.warning_count,
                "info": self.info_count,
                "passed": self.passed,
            },
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_markdown(self) -> str:
        lines = [
            "# GKE Terraform Validation Report",
            "",
            f"**Generated**: {self.timestamp}",
            f"**Terraform directory**: `{self.terraform_dir}`",
            f"**Status**: {'PASSED' if self.passed else 'FAILED'}",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total findings | {len(self.findings)} |",
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
            for f in sorted(self.findings, key=lambda x: (
                {"error": 0, "warning": 1, "info": 2}[x.severity.value],
                x.rule_id,
            )):
                sev = f.severity.value.upper()
                file_ref = f"`{f.file}`" if f.file else "-"
                lines.append(f"| {sev} | {f.rule_id} | {file_ref} | {f.message} |")

            # Recommendations section
            recs = [f for f in self.findings if f.recommendation]
            if recs:
                lines.append("")
                lines.append("## Recommendations")
                lines.append("")
                for f in recs:
                    lines.append(f"- **{f.rule_id}**: {f.recommendation}")

        else:
            lines.append("No findings -- all GKE validation rules passed.")

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HCL parsing helpers
# ---------------------------------------------------------------------------


def read_tf_files(terraform_dir: str) -> dict[str, str]:
    """Read all .tf files from the given directory tree.

    Returns a dict mapping relative file paths to their contents.
    """
    tf_files: dict[str, str] = {}
    root = Path(terraform_dir)
    if not root.is_dir():
        return tf_files
    for tf_path in root.rglob("*.tf"):
        rel = str(tf_path.relative_to(root)).replace("\\", "/")
        try:
            tf_files[rel] = tf_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Failed to read %s", tf_path)
    return tf_files


def find_resources(content: str, resource_type: str) -> list[tuple[str, str]]:
    """Find all resource blocks of a given type in HCL content.

    Returns list of (resource_name, block_content) tuples.
    """
    results = []
    pattern = re.compile(
        rf'resource\s+"{re.escape(resource_type)}"\s+"(\w+)"\s*\{{',
        re.MULTILINE,
    )
    for match in pattern.finditer(content):
        name = match.group(1)
        start = match.end()
        block = _extract_block(content, start)
        results.append((name, block))
    return results


def find_blocks(content: str, block_name: str) -> list[str]:
    """Find all named blocks (e.g. 'workload_identity_config') in HCL content.

    Handles both ``block_name {`` and ``block_name = {`` syntax.
    """
    results = []
    pattern = re.compile(
        rf'{re.escape(block_name)}\s*=?\s*\{{',
        re.MULTILINE,
    )
    for match in pattern.finditer(content):
        start = match.end()
        block = _extract_block(content, start)
        results.append(block)
    return results


def _extract_block(content: str, start_after_brace: int) -> str:
    """Extract content of a block starting after the opening brace."""
    depth = 1
    i = start_after_brace
    while i < len(content) and depth > 0:
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
        i += 1
    return content[start_after_brace:i - 1] if depth == 0 else content[start_after_brace:]


def extract_string_values(content: str, key: str) -> list[str]:
    """Extract all string values for a given key from HCL content.

    Handles both direct assignment (key = "value") and list assignment
    (key = ["value1", "value2"]).
    """
    results: list[str] = []
    # Direct assignment: key = "value"
    direct_pattern = re.compile(
        rf'{re.escape(key)}\s*=\s*"([^"]*)"', re.MULTILINE
    )
    results.extend(m.group(1) for m in direct_pattern.finditer(content))
    # List assignment: key = ["value1", "value2"]
    list_pattern = re.compile(
        rf'{re.escape(key)}\s*=\s*\[([^\]]*)\]', re.MULTILINE
    )
    for m in list_pattern.finditer(content):
        list_content = m.group(1)
        results.extend(
            sm.group(1)
            for sm in re.finditer(r'"([^"]*)"', list_content)
        )
    return results


def extract_number_values(content: str, key: str) -> list[int]:
    """Extract all numeric values for a given key from HCL content."""
    pattern = re.compile(rf'{re.escape(key)}\s*=\s*(\d+)', re.MULTILINE)
    return [int(m.group(1)) for m in pattern.finditer(content)]


def extract_bool_values(content: str, key: str) -> list[bool]:
    """Extract all boolean values for a given key from HCL content."""
    pattern = re.compile(rf'{re.escape(key)}\s*=\s*(true|false)', re.MULTILINE)
    return [m.group(1) == "true" for m in pattern.finditer(content)]


def has_block(content: str, block_name: str) -> bool:
    """Check if a named block exists in the HCL content."""
    pattern = re.compile(rf'{re.escape(block_name)}\s*\{{', re.MULTILINE)
    return bool(pattern.search(content))


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def validate_gke_cluster(
    tf_files: dict[str, str], strict: bool = False
) -> list[Finding]:
    """Validate GKE cluster configuration."""
    findings: list[Finding] = []

    # Look for GKE cluster resources
    cluster_found = False
    for file_path, content in tf_files.items():
        clusters = find_resources(content, "google_container_cluster")
        for name, block in clusters:
            cluster_found = True
            findings.extend(_validate_cluster_block(file_path, name, block, strict))

    if not cluster_found:
        # Check if there is a module reference to GKE
        has_gke_module = any(
            "modules/gke" in content or 'source.*gke' in content
            for content in tf_files.values()
        )
        if not has_gke_module:
            findings.append(Finding(
                rule_id="GKE-001",
                severity=Severity.INFO,
                message="No google_container_cluster resource found in terraform files",
                recommendation="Add a GKE cluster resource or module reference",
            ))

    return findings


def _validate_cluster_block(
    file_path: str, name: str, block: str, strict: bool
) -> list[Finding]:
    """Validate a single GKE cluster resource block."""
    findings: list[Finding] = []

    # GKE-010: Workload Identity
    if not has_block(block, "workload_identity_config"):
        findings.append(Finding(
            rule_id="GKE-010",
            severity=Severity.ERROR,
            message=f"Cluster '{name}' missing workload_identity_config (Workload Identity)",
            file=file_path,
            recommendation="Add workload_identity_config with workload_pool to enable GKE Workload Identity",
        ))

    # GKE-011: VPC-native mode (ip_allocation_policy)
    if not has_block(block, "ip_allocation_policy"):
        findings.append(Finding(
            rule_id="GKE-011",
            severity=Severity.ERROR,
            message=f"Cluster '{name}' missing ip_allocation_policy (VPC-native mode)",
            file=file_path,
            recommendation="Add ip_allocation_policy with secondary ranges for pods and services",
        ))

    # GKE-012: Private cluster
    if not has_block(block, "private_cluster_config"):
        findings.append(Finding(
            rule_id="GKE-012",
            severity=Severity.WARNING if not strict else Severity.ERROR,
            message=f"Cluster '{name}' missing private_cluster_config (private cluster)",
            file=file_path,
            recommendation="Enable private_cluster_config for production deployments",
        ))
    else:
        pc_blocks = find_blocks(block, "private_cluster_config")
        for pc_block in pc_blocks:
            private_nodes = extract_bool_values(pc_block, "enable_private_nodes")
            if private_nodes and not private_nodes[0]:
                findings.append(Finding(
                    rule_id="GKE-012",
                    severity=Severity.WARNING,
                    message=f"Cluster '{name}' has private_cluster_config but enable_private_nodes is false",
                    file=file_path,
                    recommendation="Set enable_private_nodes = true for production",
                ))

    # GKE-013: Release channel
    if has_block(block, "release_channel"):
        rc_blocks = find_blocks(block, "release_channel")
        for rc_block in rc_blocks:
            channels = extract_string_values(rc_block, "channel")
            for ch in channels:
                if ch not in VALID_RELEASE_CHANNELS:
                    findings.append(Finding(
                        rule_id="GKE-013",
                        severity=Severity.WARNING,
                        message=f"Cluster '{name}' uses unknown release channel '{ch}'",
                        file=file_path,
                        recommendation=f"Use one of: {', '.join(sorted(VALID_RELEASE_CHANNELS))}",
                    ))
                if strict and ch == "RAPID":
                    findings.append(Finding(
                        rule_id="GKE-013",
                        severity=Severity.WARNING,
                        message=f"Cluster '{name}' uses RAPID release channel (less stable)",
                        file=file_path,
                        recommendation="Use REGULAR or STABLE for production workloads",
                    ))

    # GKE-014: Default node pool removal
    remove_defaults = extract_bool_values(block, "remove_default_node_pool")
    if not remove_defaults or not remove_defaults[0]:
        findings.append(Finding(
            rule_id="GKE-014",
            severity=Severity.WARNING,
            message=f"Cluster '{name}' does not remove default node pool",
            file=file_path,
            recommendation="Set remove_default_node_pool = true and use separately managed node pools",
        ))

    # GKE-015: Resource labels
    labels_blocks = find_blocks(block, "resource_labels")
    if not labels_blocks:
        # Check for inline labels via var reference
        label_refs = re.findall(r'resource_labels\s*=\s*(\S+)', block)
        if not label_refs:
            findings.append(Finding(
                rule_id="GKE-015",
                severity=Severity.WARNING,
                message=f"Cluster '{name}' missing resource_labels",
                file=file_path,
                recommendation="Add resource_labels for cost tracking and operations",
            ))

    return findings


def validate_node_pools(
    tf_files: dict[str, str], strict: bool = False
) -> list[Finding]:
    """Validate GKE node pool configurations."""
    findings: list[Finding] = []
    gpu_pool_found = False

    for file_path, content in tf_files.items():
        pools = find_resources(content, "google_container_node_pool")
        for name, block in pools:
            findings.extend(_validate_node_pool(file_path, name, block, strict))
            if has_block(block, "guest_accelerator"):
                gpu_pool_found = True

    # Check for GPU node pool existence (required for OCR GPU workers)
    if not gpu_pool_found:
        # Check variables for gpu_type presence (might be in a module)
        has_gpu_var = any(
            "gpu_type" in content or "guest_accelerator" in content
            for content in tf_files.values()
        )
        if not has_gpu_var:
            findings.append(Finding(
                rule_id="GKE-020",
                severity=Severity.WARNING,
                message="No GPU node pool found -- OCR pipeline requires GPU workers",
                file="",
                recommendation="Add a GPU node pool with NVIDIA accelerators for OCR processing",
            ))

    return findings


def _validate_node_pool(
    file_path: str, name: str, block: str, strict: bool
) -> list[Finding]:
    """Validate a single node pool resource."""
    findings: list[Finding] = []
    is_gpu = has_block(block, "guest_accelerator")

    # GKE-021: Machine type validation
    machine_types = extract_string_values(block, "machine_type")
    for mt in machine_types:
        family = mt.split("-")[0] if "-" in mt else mt
        if is_gpu and family not in GPU_COMPATIBLE_MACHINE_FAMILIES:
            findings.append(Finding(
                rule_id="GKE-021",
                severity=Severity.WARNING,
                message=f"Node pool '{name}' uses machine type '{mt}' which may not support GPUs",
                file=file_path,
                recommendation=f"Use a GPU-compatible machine family: {', '.join(sorted(GPU_COMPATIBLE_MACHINE_FAMILIES))}",
            ))

    # GKE-022: GPU type validation
    if is_gpu:
        gpu_blocks = find_blocks(block, "guest_accelerator")
        for gpu_block in gpu_blocks:
            gpu_types = extract_string_values(gpu_block, "type")
            for gt in gpu_types:
                if gt not in RECOMMENDED_GPU_TYPES:
                    findings.append(Finding(
                        rule_id="GKE-022",
                        severity=Severity.INFO,
                        message=f"Node pool '{name}' uses GPU type '{gt}' which is not in recommended set",
                        file=file_path,
                        recommendation=f"Recommended GPU types: {', '.join(sorted(RECOMMENDED_GPU_TYPES))}",
                    ))

            # GKE-023: GPU driver auto-install
            if not has_block(gpu_block, "gpu_driver_installation_config"):
                findings.append(Finding(
                    rule_id="GKE-023",
                    severity=Severity.WARNING,
                    message=f"Node pool '{name}' missing gpu_driver_installation_config",
                    file=file_path,
                    recommendation="Add gpu_driver_installation_config for automatic GPU driver installation",
                ))

    # GKE-024: Disk size for GPU nodes
    if is_gpu:
        disk_sizes = extract_number_values(block, "disk_size_gb")
        for ds in disk_sizes:
            if ds < MIN_GPU_DISK_SIZE_GB:
                findings.append(Finding(
                    rule_id="GKE-024",
                    severity=Severity.WARNING,
                    message=f"Node pool '{name}' disk size {ds}GB is below recommended minimum {MIN_GPU_DISK_SIZE_GB}GB for GPU nodes",
                    file=file_path,
                    recommendation=f"Increase disk_size_gb to at least {MIN_GPU_DISK_SIZE_GB}GB for model storage",
                ))

    # GKE-025: GPU taint
    if is_gpu:
        if not has_block(block, "taint"):
            findings.append(Finding(
                rule_id="GKE-025",
                severity=Severity.WARNING,
                message=f"GPU node pool '{name}' missing nvidia.com/gpu taint",
                file=file_path,
                recommendation="Add taint { key = 'nvidia.com/gpu', value = 'true', effect = 'NO_SCHEDULE' }",
            ))

    # GKE-026: Autoscaling
    if has_block(block, "autoscaling"):
        as_blocks = find_blocks(block, "autoscaling")
        for as_block in as_blocks:
            mins = extract_number_values(as_block, "min_node_count")
            maxs = extract_number_values(as_block, "max_node_count")
            for mn in mins:
                for mx in maxs:
                    if mx <= mn:
                        findings.append(Finding(
                            rule_id="GKE-026",
                            severity=Severity.ERROR,
                            message=f"Node pool '{name}' autoscaling max_node_count ({mx}) <= min_node_count ({mn})",
                            file=file_path,
                            recommendation="Set max_node_count greater than min_node_count",
                        ))
    else:
        findings.append(Finding(
            rule_id="GKE-026",
            severity=Severity.WARNING,
            message=f"Node pool '{name}' does not have autoscaling configured",
            file=file_path,
            recommendation="Enable autoscaling for dynamic workload management",
        ))

    # GKE-027: Shielded instance config
    node_config_blocks = find_blocks(block, "node_config")
    effective_block = node_config_blocks[0] if node_config_blocks else block
    if not has_block(effective_block, "shielded_instance_config"):
        # Check the outer block too
        if not has_block(block, "shielded_instance_config"):
            severity = Severity.WARNING if not strict else Severity.ERROR
            findings.append(Finding(
                rule_id="GKE-027",
                severity=severity,
                message=f"Node pool '{name}' missing shielded_instance_config (Secure Boot)",
                file=file_path,
                recommendation="Add shielded_instance_config { enable_secure_boot = true } for security hardening",
            ))

    # GKE-028: Auto-repair and auto-upgrade
    if has_block(block, "management"):
        mgmt_blocks = find_blocks(block, "management")
        for mgmt in mgmt_blocks:
            repairs = extract_bool_values(mgmt, "auto_repair")
            upgrades = extract_bool_values(mgmt, "auto_upgrade")
            if repairs and not repairs[0]:
                findings.append(Finding(
                    rule_id="GKE-028",
                    severity=Severity.WARNING,
                    message=f"Node pool '{name}' has auto_repair disabled",
                    file=file_path,
                    recommendation="Enable auto_repair for automatic node recovery",
                ))
            if upgrades and not upgrades[0]:
                findings.append(Finding(
                    rule_id="GKE-028",
                    severity=Severity.WARNING,
                    message=f"Node pool '{name}' has auto_upgrade disabled",
                    file=file_path,
                    recommendation="Enable auto_upgrade for automatic security patches",
                ))

    # GKE-029: Legacy endpoints disabled
    if has_block(block, "metadata"):
        meta_blocks = find_blocks(block, "metadata")
        for meta in meta_blocks:
            legacy = extract_string_values(meta, "disable-legacy-endpoints")
            if legacy and legacy[0] != "true":
                findings.append(Finding(
                    rule_id="GKE-029",
                    severity=Severity.WARNING,
                    message=f"Node pool '{name}' does not disable legacy metadata endpoints",
                    file=file_path,
                    recommendation='Set metadata { disable-legacy-endpoints = "true" }',
                ))

    # GKE-030: Node labels
    if not has_block(block, "labels"):
        # check for labels via var reference
        label_refs = re.findall(r'labels\s*=\s*(\S+)', block)
        if not label_refs:
            findings.append(Finding(
                rule_id="GKE-030",
                severity=Severity.INFO,
                message=f"Node pool '{name}' has no labels configured",
                file=file_path,
                recommendation="Add labels for node identification and scheduling",
            ))

    return findings


def validate_iam_bindings(
    tf_files: dict[str, str], strict: bool = False
) -> list[Finding]:
    """Validate IAM bindings for least-privilege."""
    findings: list[Finding] = []
    found_roles: set[str] = set()

    for file_path, content in tf_files.items():
        # Check google_project_iam_member resources
        iam_resources = find_resources(content, "google_project_iam_member")
        for name, block in iam_resources:
            roles = extract_string_values(block, "role")
            for role in roles:
                found_roles.add(role)
                # GKE-040: Overly permissive roles
                if role in OVERLY_PERMISSIVE_ROLES:
                    findings.append(Finding(
                        rule_id="GKE-040",
                        severity=Severity.ERROR,
                        message=f"IAM binding '{name}' uses overly permissive role '{role}'",
                        file=file_path,
                        recommendation=f"Replace with least-privilege roles. Avoid: {', '.join(sorted(OVERLY_PERMISSIVE_ROLES))}",
                    ))

        # Check google_project_iam_binding (less preferred)
        iam_bindings = find_resources(content, "google_project_iam_binding")
        for name, block in iam_bindings:
            findings.append(Finding(
                rule_id="GKE-041",
                severity=Severity.WARNING,
                message=f"IAM binding '{name}' uses google_project_iam_binding (prefer google_project_iam_member)",
                file=file_path,
                recommendation="Use google_project_iam_member for additive bindings",
            ))
            roles = extract_string_values(block, "role")
            for role in roles:
                found_roles.add(role)
                if role in OVERLY_PERMISSIVE_ROLES:
                    findings.append(Finding(
                        rule_id="GKE-040",
                        severity=Severity.ERROR,
                        message=f"IAM binding '{name}' uses overly permissive role '{role}'",
                        file=file_path,
                        recommendation="Replace with least-privilege roles",
                    ))

    # GKE-042: Check for required least-privilege roles
    missing_roles = REQUIRED_NODE_IAM_ROLES - found_roles
    if missing_roles and found_roles:
        for role in sorted(missing_roles):
            findings.append(Finding(
                rule_id="GKE-042",
                severity=Severity.INFO,
                message=f"Expected IAM role '{role}' not found in terraform files",
                recommendation=f"Add google_project_iam_member for '{role}' on the node service account",
            ))

    return findings


def validate_networking(
    tf_files: dict[str, str], strict: bool = False
) -> list[Finding]:
    """Validate networking configuration (VPC, subnets, NAT, firewall)."""
    findings: list[Finding] = []
    has_vpc = False
    has_nat = False

    for file_path, content in tf_files.items():
        # Check for VPC
        vpcs = find_resources(content, "google_compute_network")
        if vpcs:
            has_vpc = True
            for name, block in vpcs:
                # GKE-050: auto_create_subnetworks should be false
                auto_sub = extract_bool_values(block, "auto_create_subnetworks")
                if auto_sub and auto_sub[0]:
                    findings.append(Finding(
                        rule_id="GKE-050",
                        severity=Severity.WARNING,
                        message=f"VPC '{name}' has auto_create_subnetworks = true",
                        file=file_path,
                        recommendation="Set auto_create_subnetworks = false for explicit subnet control",
                    ))

        # Check for subnet with secondary ranges
        subnets = find_resources(content, "google_compute_subnetwork")
        if subnets:
            for name, block in subnets:
                # GKE-051: Private Google access
                pga = extract_bool_values(block, "private_ip_google_access")
                if pga and not pga[0]:
                    findings.append(Finding(
                        rule_id="GKE-051",
                        severity=Severity.WARNING,
                        message=f"Subnet '{name}' has private_ip_google_access disabled",
                        file=file_path,
                        recommendation="Enable private_ip_google_access for Google API access without public IPs",
                    ))

                # GKE-052: Secondary IP ranges for pods/services
                if not has_block(block, "secondary_ip_range"):
                    findings.append(Finding(
                        rule_id="GKE-052",
                        severity=Severity.ERROR,
                        message=f"Subnet '{name}' missing secondary_ip_range for pods/services",
                        file=file_path,
                        recommendation="Add secondary_ip_range blocks for pods and services CIDR ranges",
                    ))

        # Check for Cloud Router
        # Check for Cloud NAT
        nats = find_resources(content, "google_compute_router_nat")
        if nats:
            has_nat = True

        # Check for firewall rules
        firewalls = find_resources(content, "google_compute_firewall")
        for name, block in firewalls:
            # GKE-053: Overly permissive firewall rules
            source_ranges = extract_string_values(block, "source_ranges")
            if "0.0.0.0/0" in source_ranges:
                findings.append(Finding(
                    rule_id="GKE-053",
                    severity=Severity.ERROR if strict else Severity.WARNING,
                    message=f"Firewall rule '{name}' allows traffic from 0.0.0.0/0",
                    file=file_path,
                    recommendation="Restrict source_ranges to specific CIDR blocks",
                ))

    # GKE-054: NAT gateway for private nodes
    if has_vpc and not has_nat:
        findings.append(Finding(
            rule_id="GKE-054",
            severity=Severity.WARNING,
            message="No Cloud NAT found -- private nodes need NAT for outbound internet access",
            recommendation="Add google_compute_router_nat for egress from private nodes",
        ))

    return findings


def validate_labels(
    tf_files: dict[str, str], strict: bool = False
) -> list[Finding]:
    """Validate resource labels for cost tracking and operations."""
    findings: list[Finding] = []

    for file_path, content in tf_files.items():
        # Check variable defaults for labels -- use brace-aware extraction
        pattern = re.compile(r'variable\s+"labels"\s*\{', re.MULTILINE)
        for match in pattern.finditer(content):
            var_block = _extract_block(content, match.end())
            default_blocks = find_blocks(var_block, "default")
            for db in default_blocks:
                found_labels = set(RE_STRING_VALUE.findall(db))
                found_label_keys = {k for k, _v in found_labels}

                missing_required = REQUIRED_LABELS - found_label_keys
                for lbl in sorted(missing_required):
                    findings.append(Finding(
                        rule_id="GKE-060",
                        severity=Severity.WARNING,
                        message=f"Labels variable in '{file_path}' missing required label '{lbl}'",
                        file=file_path,
                        recommendation=f"Add '{lbl}' to default labels for resource tracking",
                    ))

                if strict:
                    missing_recommended = RECOMMENDED_LABELS - found_label_keys
                    for lbl in sorted(missing_recommended):
                        findings.append(Finding(
                            rule_id="GKE-061",
                            severity=Severity.INFO,
                            message=f"Labels variable in '{file_path}' missing recommended label '{lbl}'",
                            file=file_path,
                            recommendation=f"Consider adding '{lbl}' for improved cost attribution",
                        ))

    return findings


def validate_artifact_registry(
    tf_files: dict[str, str], strict: bool = False
) -> list[Finding]:
    """Validate Artifact Registry configuration."""
    findings: list[Finding] = []
    ar_found = False

    for file_path, content in tf_files.items():
        repos = find_resources(content, "google_artifact_registry_repository")
        if repos:
            ar_found = True
            for name, block in repos:
                formats = extract_string_values(block, "format")
                for fmt in formats:
                    if fmt != "DOCKER":
                        findings.append(Finding(
                            rule_id="GKE-070",
                            severity=Severity.INFO,
                            message=f"Artifact Registry '{name}' uses format '{fmt}' (expected DOCKER)",
                            file=file_path,
                        ))

    if not ar_found:
        # Check module references
        has_ar_var = any(
            "artifact_registry" in content
            for content in tf_files.values()
        )
        if not has_ar_var:
            findings.append(Finding(
                rule_id="GKE-071",
                severity=Severity.INFO,
                message="No Artifact Registry resource found",
                recommendation="Add google_artifact_registry_repository for container image storage",
            ))

    return findings


def validate_autopilot_detection(
    tf_files: dict[str, str], strict: bool = False
) -> list[Finding]:
    """Detect and validate GKE Autopilot vs Standard mode settings."""
    findings: list[Finding] = []

    for file_path, content in tf_files.items():
        clusters = find_resources(content, "google_container_cluster")
        for name, block in clusters:
            autopilot_vals = extract_bool_values(block, "enable_autopilot")
            if autopilot_vals and autopilot_vals[0]:
                findings.append(Finding(
                    rule_id="GKE-080",
                    severity=Severity.INFO,
                    message=f"Cluster '{name}' uses GKE Autopilot mode",
                    file=file_path,
                    recommendation=(
                        "Autopilot manages node pools automatically. "
                        "Verify GPU workload class is available in your region for OCR workers."
                    ),
                ))
                # Autopilot ignores manual node pool configs
                node_pools = find_resources(content, "google_container_node_pool")
                if node_pools:
                    findings.append(Finding(
                        rule_id="GKE-081",
                        severity=Severity.WARNING,
                        message=f"Cluster '{name}' uses Autopilot but has manual node pool definitions",
                        file=file_path,
                        recommendation="Autopilot manages node pools automatically; remove manual node pool resources",
                    ))

    return findings


def validate_service_account(
    tf_files: dict[str, str], strict: bool = False
) -> list[Finding]:
    """Validate service account configuration for GKE nodes."""
    findings: list[Finding] = []
    sa_found = False

    for file_path, content in tf_files.items():
        sas = find_resources(content, "google_service_account")
        if sas:
            sa_found = True

    if not sa_found:
        # Might be using default compute SA or referencing via variable
        has_sa_ref = any(
            "service_account" in content
            for content in tf_files.values()
        )
        if not has_sa_ref:
            findings.append(Finding(
                rule_id="GKE-090",
                severity=Severity.WARNING,
                message="No dedicated service account found for GKE nodes",
                recommendation="Create a dedicated service account with least-privilege for GKE nodes",
            ))

    return findings


# ---------------------------------------------------------------------------
# Main validation orchestrator
# ---------------------------------------------------------------------------


def validate_gke_terraform(
    terraform_dir: str,
    strict: bool = False,
) -> ValidationReport:
    """Run all GKE validation rules against terraform files.

    Args:
        terraform_dir: Path to the terraform directory to validate.
        strict: If True, treat warnings as errors for some rules.

    Returns:
        ValidationReport with all findings.
    """
    report = ValidationReport(terraform_dir=terraform_dir)

    tf_files = read_tf_files(terraform_dir)
    if not tf_files:
        report.findings.append(Finding(
            rule_id="GKE-000",
            severity=Severity.ERROR,
            message=f"No .tf files found in '{terraform_dir}'",
            recommendation="Check that --terraform-dir points to a directory with Terraform files",
        ))
        return report

    # Filter to GKE-relevant files (modules/gke/ and environment files referencing GKE)
    gke_files: dict[str, str] = {}
    for path, content in tf_files.items():
        if (
            "gke" in path.lower()
            or "google_container" in content
            or "google_compute" in content
            or "google_service_account" in content
            or "google_artifact_registry" in content
            or "google_project_iam" in content
            or 'cloud_provider' in content
            or 'module "gke"' in content
        ):
            gke_files[path] = content

    if not gke_files:
        report.findings.append(Finding(
            rule_id="GKE-000",
            severity=Severity.INFO,
            message="No GKE-related terraform files found",
            recommendation="Ensure terraform files contain GKE resources or module references",
        ))
        return report

    logger.info("Validating %d GKE-related terraform files", len(gke_files))

    # Run all validation categories
    report.findings.extend(validate_gke_cluster(gke_files, strict))
    report.findings.extend(validate_node_pools(gke_files, strict))
    report.findings.extend(validate_iam_bindings(gke_files, strict))
    report.findings.extend(validate_networking(gke_files, strict))
    report.findings.extend(validate_labels(gke_files, strict))
    report.findings.extend(validate_artifact_registry(gke_files, strict))
    report.findings.extend(validate_autopilot_detection(gke_files, strict))
    report.findings.extend(validate_service_account(gke_files, strict))

    logger.info(
        "Validation complete: %d errors, %d warnings, %d info",
        report.error_count,
        report.warning_count,
        report.info_count,
    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GKE Terraform configuration for OCR-Local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--terraform-dir",
        default="terraform",
        help="Path to terraform directory (default: terraform)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: escalate warnings to errors for some rules",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON and Markdown reports (default: stdout only)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON only (no Markdown summary)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    report = validate_gke_terraform(
        terraform_dir=args.terraform_dir,
        strict=args.strict,
    )

    # Output
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "gke-validation-report.json"
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("JSON report written to %s", json_path)

        if not args.json_only:
            md_path = out_dir / "gke-validation-report.md"
            md_path.write_text(report.to_markdown(), encoding="utf-8")
            logger.info("Markdown report written to %s", md_path)

    # Always print summary to stdout
    report_dict = report.to_dict()
    if args.json_only:
        print(json.dumps(report_dict, indent=2, ensure_ascii=False))
    else:
        print(report.to_markdown())

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
