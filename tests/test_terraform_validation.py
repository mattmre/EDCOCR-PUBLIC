"""Tests for Terraform validation tools.

Covers:
- Variable description/type/default validation
- Credential scanning
- Naming convention checks
- Tag/label compliance
- Provider version constraint checks
- Backend configuration checks
- Placeholder detection
- Sensitive variable detection
- Report generation (JSON + Markdown)
- Plan check output parsing
- Environment resolution
- CLI argument handling
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from scripts.terraform_plan_check import (
    CommandResult,
    PlanCheckReport,
    analyze_plan_output,
    find_terraform_binary,
    run_plan_check,
)
from scripts.validate_terraform import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    Finding,
    ValidationReport,
    check_backend_configuration,
    check_credential_patterns,
    check_placeholder_values,
    check_provider_version_constraints,
    check_required_variables,
    check_resource_naming,
    check_resource_tags,
    check_sensitive_variables,
    check_variable_descriptions,
    check_variable_types,
    parse_resources,
    parse_terraform_block,
    parse_variables,
    resolve_environments,
    run_validation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_terraform_root(tmp_path):
    """Create a minimal terraform module structure for testing."""
    # EKS module
    eks_dir = tmp_path / "modules" / "eks"
    eks_dir.mkdir(parents=True)

    (eks_dir / "main.tf").write_text(textwrap.dedent("""\
        terraform {
          required_version = ">= 1.5"

          required_providers {
            aws = {
              source  = "hashicorp/aws"
              version = ">= 5.0"
            }
          }
        }

        resource "aws_vpc" "main" {
          cidr_block = var.vpc_cidr

          tags = merge(var.tags, {
            Name = "${var.cluster_name}-vpc"
          })
        }

        resource "aws_eks_cluster" "main" {
          name     = var.cluster_name
          role_arn = aws_iam_role.eks_cluster.arn

          tags = var.tags
        }
    """), encoding="utf-8")

    (eks_dir / "variables.tf").write_text(textwrap.dedent("""\
        variable "cluster_name" {
          description = "Name of the EKS cluster"
          type        = string
        }

        variable "vpc_cidr" {
          description = "CIDR block for the VPC"
          type        = string
          default     = "10.0.0.0/16"
        }

        variable "tags" {
          description = "Tags to apply to all resources"
          type        = map(string)
          default = {
            Project   = "ocr-local"
            ManagedBy = "terraform"
          }
        }
    """), encoding="utf-8")

    (eks_dir / "outputs.tf").write_text(textwrap.dedent("""\
        output "cluster_name" {
          description = "Name of the EKS cluster"
          value       = aws_eks_cluster.main.name
        }
    """), encoding="utf-8")

    # GKE module (with some issues for testing)
    gke_dir = tmp_path / "modules" / "gke"
    gke_dir.mkdir(parents=True)

    (gke_dir / "main.tf").write_text(textwrap.dedent("""\
        terraform {
          required_version = ">= 1.5"

          required_providers {
            google = {
              source  = "hashicorp/google"
              version = ">= 5.0"
            }
          }
        }

        resource "google_container_cluster" "main" {
          name     = var.cluster_name
          project  = var.project_id
          location = var.region

          resource_labels = var.labels
        }
    """), encoding="utf-8")

    (gke_dir / "variables.tf").write_text(textwrap.dedent("""\
        variable "cluster_name" {
          description = "Name of the GKE cluster"
          type        = string
        }

        variable "project_id" {
          description = "GCP project ID"
          type        = string
        }

        variable "region" {
          description = "GCP region"
          type        = string
          default     = "us-central1"
        }

        variable "labels" {
          description = "Labels for resources"
          type        = map(string)
          default = {
            project    = "ocr-local"
            managed-by = "terraform"
          }
        }
    """), encoding="utf-8")

    # Shared module
    shared_dir = tmp_path / "modules" / "shared"
    shared_dir.mkdir(parents=True)
    (shared_dir / "keda.tf").write_text(textwrap.dedent("""\
        terraform {
          required_version = ">= 1.5"

          required_providers {
            helm = {
              source  = "hashicorp/helm"
              version = ">= 2.12"
            }
          }
        }

        variable "keda_namespace" {
          description = "Kubernetes namespace for KEDA"
          type        = string
          default     = "keda"
        }
    """), encoding="utf-8")

    (shared_dir / "monitoring.tf").write_text(textwrap.dedent("""\
        variable "monitoring_namespace" {
          description = "Kubernetes namespace for monitoring"
          type        = string
          default     = "monitoring"
        }

        variable "grafana_admin_password" {
          description = "Grafana admin password"
          type        = string
          sensitive   = true
          default     = ""
        }
    """), encoding="utf-8")

    # OKE module
    oke_dir = tmp_path / "modules" / "oke"
    oke_dir.mkdir(parents=True)
    (oke_dir / "main.tf").write_text(textwrap.dedent("""\
        terraform {
          required_version = ">= 1.5"

          required_providers {
            oci = {
              source  = "oracle/oci"
              version = ">= 5.0"
            }
          }
        }

        resource "oci_core_vcn" "main" {
          compartment_id = var.compartment_id
          display_name   = "${var.cluster_name}-vcn"
          freeform_tags  = var.freeform_tags
        }
    """), encoding="utf-8")

    (oke_dir / "variables.tf").write_text(textwrap.dedent("""\
        variable "cluster_name" {
          description = "Name of the OKE cluster"
          type        = string
        }

        variable "compartment_id" {
          description = "OCI compartment OCID"
          type        = string
        }

        variable "freeform_tags" {
          description = "Tags for resources"
          type        = map(string)
          default = {
            Project   = "ocr-local"
            ManagedBy = "terraform"
          }
        }
    """), encoding="utf-8")

    # Environments
    staging_dir = tmp_path / "environments" / "staging"
    staging_dir.mkdir(parents=True)
    (staging_dir / "main.tf").write_text(textwrap.dedent("""\
        terraform {
          required_version = ">= 1.5"

          # backend "s3" {
          #   bucket = "ocr-local-terraform-state"
          #   key    = "staging/terraform.tfstate"
          # }
        }

        variable "cloud_provider" {
          description = "Cloud provider to deploy to"
          type        = string
          default     = "aws"
        }
    """), encoding="utf-8")

    production_dir = tmp_path / "environments" / "production"
    production_dir.mkdir(parents=True)
    (production_dir / "main.tf").write_text(textwrap.dedent("""\
        terraform {
          required_version = ">= 1.5"

          # backend "s3" {
          #   bucket = "ocr-local-terraform-state"
          #   key    = "production/terraform.tfstate"
          # }
        }

        variable "cluster_name" {
          description = "Name of the cluster"
          type        = string
          default     = "ocr-local-production"
        }
    """), encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Test: Variable parsing
# ---------------------------------------------------------------------------


class TestParseVariables:
    def test_parse_basic_variable(self):
        lines = [
            'variable "name" {',
            '  description = "A name"',
            '  type        = string',
            '}',
        ]
        variables = parse_variables(lines)
        assert len(variables) == 1
        assert variables[0]["name"] == "name"
        assert variables[0]["has_description"] is True
        assert variables[0]["has_type"] is True
        assert variables[0]["has_default"] is False

    def test_parse_variable_with_default(self):
        lines = [
            'variable "region" {',
            '  description = "AWS region"',
            '  type        = string',
            '  default     = "us-east-1"',
            '}',
        ]
        variables = parse_variables(lines)
        assert len(variables) == 1
        assert variables[0]["has_default"] is True

    def test_parse_sensitive_variable(self):
        lines = [
            'variable "password" {',
            '  description = "DB password"',
            '  type        = string',
            '  sensitive   = true',
            '}',
        ]
        variables = parse_variables(lines)
        assert variables[0]["is_sensitive"] is True

    def test_parse_multiple_variables(self):
        lines = [
            'variable "a" {',
            '  type = string',
            '}',
            '',
            'variable "b" {',
            '  description = "B"',
            '  type = number',
            '  default = 1',
            '}',
        ]
        variables = parse_variables(lines)
        assert len(variables) == 2
        assert variables[0]["name"] == "a"
        assert variables[0]["has_description"] is False
        assert variables[1]["name"] == "b"
        assert variables[1]["has_default"] is True

    def test_parse_variable_with_nested_braces(self):
        lines = [
            'variable "tags" {',
            '  description = "Tags"',
            '  type        = map(string)',
            '  default = {',
            '    Project = "ocr"',
            '  }',
            '}',
        ]
        variables = parse_variables(lines)
        assert len(variables) == 1
        assert variables[0]["has_default"] is True

    def test_parse_empty_lines(self):
        variables = parse_variables([])
        assert variables == []


# ---------------------------------------------------------------------------
# Test: Resource parsing
# ---------------------------------------------------------------------------


class TestParseResources:
    def test_parse_basic_resource(self):
        lines = [
            'resource "aws_vpc" "main" {',
            '  cidr_block = "10.0.0.0/16"',
            '',
            '  tags = {',
            '    Name = "test"',
            '  }',
            '}',
        ]
        resources = parse_resources(lines)
        assert len(resources) == 1
        assert resources[0]["type"] == "aws_vpc"
        assert resources[0]["name"] == "main"
        assert len(resources[0]["content_lines"]) > 0

    def test_parse_no_resources(self):
        lines = [
            'variable "x" {',
            '  type = string',
            '}',
        ]
        resources = parse_resources(lines)
        assert resources == []


# ---------------------------------------------------------------------------
# Test: Terraform block parsing
# ---------------------------------------------------------------------------


class TestParseTerraformBlock:
    def test_parse_full_block(self):
        lines = [
            'terraform {',
            '  required_version = ">= 1.5"',
            '',
            '  required_providers {',
            '    aws = {',
            '      source  = "hashicorp/aws"',
            '      version = ">= 5.0"',
            '    }',
            '  }',
            '}',
        ]
        block = parse_terraform_block(lines)
        assert block["has_required_version"] is True
        assert block["has_required_providers"] is True
        assert block["has_backend"] is False

    def test_parse_commented_backend(self):
        lines = [
            'terraform {',
            '  required_version = ">= 1.5"',
            '  # backend "s3" {',
            '  #   bucket = "test"',
            '  # }',
            '}',
        ]
        block = parse_terraform_block(lines)
        assert block["backend_commented"] is True
        assert block["has_backend"] is False

    def test_parse_no_terraform_block(self):
        lines = [
            'variable "x" {',
            '  type = string',
            '}',
        ]
        block = parse_terraform_block(lines)
        assert block["has_required_version"] is False


# ---------------------------------------------------------------------------
# Test: Variable description checks
# ---------------------------------------------------------------------------


class TestCheckVariableDescriptions:
    def test_missing_description_is_error(self):
        report = ValidationReport(environment="test")
        variables = [{"name": "x", "line": 1, "has_description": False,
                      "has_type": True, "has_default": True, "is_sensitive": False}]
        check_variable_descriptions(variables, "test.tf", report)
        assert report.error_count == 1
        assert report.findings[0].rule == "VAR-DESC"

    def test_has_description_passes(self):
        report = ValidationReport(environment="test")
        variables = [{"name": "x", "line": 1, "has_description": True,
                      "has_type": True, "has_default": True, "is_sensitive": False}]
        check_variable_descriptions(variables, "test.tf", report)
        assert report.error_count == 0


# ---------------------------------------------------------------------------
# Test: Variable type checks
# ---------------------------------------------------------------------------


class TestCheckVariableTypes:
    def test_missing_type_is_error(self):
        report = ValidationReport(environment="test")
        variables = [{"name": "x", "line": 1, "has_description": True,
                      "has_type": False, "has_default": True, "is_sensitive": False}]
        check_variable_types(variables, "test.tf", report)
        assert report.error_count == 1
        assert report.findings[0].rule == "VAR-TYPE"

    def test_has_type_passes(self):
        report = ValidationReport(environment="test")
        variables = [{"name": "x", "line": 1, "has_description": True,
                      "has_type": True, "has_default": True, "is_sensitive": False}]
        check_variable_types(variables, "test.tf", report)
        assert report.error_count == 0


# ---------------------------------------------------------------------------
# Test: Required variables (info-level)
# ---------------------------------------------------------------------------


class TestCheckRequiredVariables:
    def test_no_default_is_info(self):
        report = ValidationReport(environment="test")
        variables = [{"name": "cluster_name", "line": 1, "has_description": True,
                      "has_type": True, "has_default": False, "is_sensitive": False}]
        check_required_variables(variables, "test.tf", report)
        assert report.info_count == 1
        assert report.findings[0].rule == "VAR-REQUIRED"

    def test_has_default_no_info(self):
        report = ValidationReport(environment="test")
        variables = [{"name": "region", "line": 1, "has_description": True,
                      "has_type": True, "has_default": True, "is_sensitive": False}]
        check_required_variables(variables, "test.tf", report)
        assert report.info_count == 0


# ---------------------------------------------------------------------------
# Test: Credential scanning
# ---------------------------------------------------------------------------


class TestCheckCredentialPatterns:
    def test_detect_aws_access_key(self):
        report = ValidationReport(environment="test")
        lines = ['  access_key = "AKIAIOSFODNN7EXAMPLE"']
        check_credential_patterns(lines, "test.tf", report)
        assert report.error_count >= 1
        messages = " ".join(f.message.lower() for f in report.findings)
        assert "access key" in messages

    def test_detect_embedded_private_key(self):
        report = ValidationReport(environment="test")
        lines = ['  private_key = "-----BEGIN RSA PRIVATE KEY-----"']
        check_credential_patterns(lines, "test.tf", report)
        assert report.error_count >= 1

    def test_no_credentials_clean(self):
        report = ValidationReport(environment="test")
        lines = [
            '  region = "us-east-1"',
            '  name   = "test-cluster"',
        ]
        check_credential_patterns(lines, "test.tf", report)
        assert report.error_count == 0

    def test_detect_aws_key_id_literal(self):
        report = ValidationReport(environment="test")
        lines = ['  key = "AKIAI44QH8DHBEXAMPLE"']
        check_credential_patterns(lines, "test.tf", report)
        assert report.error_count >= 1


# ---------------------------------------------------------------------------
# Test: Provider version constraints
# ---------------------------------------------------------------------------


class TestCheckProviderVersionConstraints:
    def test_missing_required_version(self):
        report = ValidationReport(environment="test")
        tf_block = {
            "has_required_version": False,
            "has_required_providers": True,
        }
        check_provider_version_constraints(tf_block, "test.tf", report)
        assert report.error_count == 1
        assert report.findings[0].rule == "TF-VERSION"

    def test_missing_required_providers(self):
        report = ValidationReport(environment="test")
        tf_block = {
            "has_required_version": True,
            "has_required_providers": False,
        }
        check_provider_version_constraints(tf_block, "test.tf", report)
        assert report.warning_count == 1
        assert report.findings[0].rule == "TF-PROVIDERS"

    def test_all_present_passes(self):
        report = ValidationReport(environment="test")
        tf_block = {
            "has_required_version": True,
            "has_required_providers": True,
        }
        check_provider_version_constraints(tf_block, "test.tf", report)
        assert report.error_count == 0
        assert report.warning_count == 0


# ---------------------------------------------------------------------------
# Test: Backend configuration
# ---------------------------------------------------------------------------


class TestCheckBackendConfiguration:
    def test_no_backend_in_environment_warns(self):
        report = ValidationReport(environment="test")
        tf_block = {"has_backend": False, "backend_commented": False}
        check_backend_configuration(tf_block, "environments/staging/main.tf", report)
        assert report.warning_count == 1

    def test_commented_backend_passes(self):
        report = ValidationReport(environment="test")
        tf_block = {"has_backend": False, "backend_commented": True}
        check_backend_configuration(tf_block, "environments/staging/main.tf", report)
        assert report.warning_count == 0

    def test_no_backend_in_module_ok(self):
        """Modules do not need backend config."""
        report = ValidationReport(environment="test")
        tf_block = {"has_backend": False, "backend_commented": False}
        check_backend_configuration(tf_block, "modules/eks/main.tf", report)
        assert report.warning_count == 0


# ---------------------------------------------------------------------------
# Test: Resource naming
# ---------------------------------------------------------------------------


class TestCheckResourceNaming:
    def test_resource_with_cluster_name_passes(self):
        report = ValidationReport(environment="test")
        resources = [{
            "type": "aws_vpc",
            "name": "main",
            "line": 1,
            "content_lines": ['  Name = "${var.cluster_name}-vpc"'],
        }]
        check_resource_naming(resources, "test.tf", report)
        assert report.warning_count == 0

    def test_iam_attachment_skipped(self):
        report = ValidationReport(environment="test")
        resources = [{
            "type": "aws_iam_role_policy_attachment",
            "name": "eks",
            "line": 1,
            "content_lines": ['  policy_arn = "arn:aws:iam::aws:policy/Test"'],
        }]
        check_resource_naming(resources, "test.tf", report)
        assert report.warning_count == 0


# ---------------------------------------------------------------------------
# Test: Resource tags
# ---------------------------------------------------------------------------


class TestCheckResourceTags:
    def test_aws_resource_without_tags_warns(self):
        report = ValidationReport(environment="test")
        resources = [{
            "type": "aws_vpc",
            "name": "main",
            "line": 1,
            "content_lines": ['  cidr_block = "10.0.0.0/16"'],
        }]
        check_resource_tags(resources, "test.tf", report, "eks")
        assert report.warning_count == 1
        assert "tags" in report.findings[0].message

    def test_aws_resource_with_tags_passes(self):
        report = ValidationReport(environment="test")
        resources = [{
            "type": "aws_vpc",
            "name": "main",
            "line": 1,
            "content_lines": ['  tags = var.tags'],
        }]
        check_resource_tags(resources, "test.tf", report, "eks")
        assert report.warning_count == 0

    def test_gke_resource_checks_labels(self):
        report = ValidationReport(environment="test")
        resources = [{
            "type": "google_container_cluster",
            "name": "main",
            "line": 1,
            "content_lines": ['  name = "test"'],
        }]
        check_resource_tags(resources, "test.tf", report, "gke")
        assert report.warning_count == 1
        assert "labels" in report.findings[0].message

    def test_oke_resource_checks_freeform_tags(self):
        report = ValidationReport(environment="test")
        resources = [{
            "type": "oci_core_vcn",
            "name": "main",
            "line": 1,
            "content_lines": ['  display_name = "test"'],
        }]
        check_resource_tags(resources, "test.tf", report, "oke")
        assert report.warning_count == 1
        assert "freeform_tags" in report.findings[0].message

    def test_non_taggable_resource_skipped(self):
        report = ValidationReport(environment="test")
        resources = [{
            "type": "aws_iam_role",
            "name": "main",
            "line": 1,
            "content_lines": ['  name = "test"'],
        }]
        check_resource_tags(resources, "test.tf", report, "eks")
        assert report.warning_count == 0


# ---------------------------------------------------------------------------
# Test: Sensitive variable detection
# ---------------------------------------------------------------------------


class TestCheckSensitiveVariables:
    def test_password_not_sensitive_warns(self):
        report = ValidationReport(environment="test")
        variables = [{"name": "db_password", "line": 1, "has_description": True,
                      "has_type": True, "has_default": False, "is_sensitive": False}]
        check_sensitive_variables(variables, "test.tf", report)
        assert report.warning_count == 1
        assert "sensitive" in report.findings[0].message.lower()

    def test_password_marked_sensitive_passes(self):
        report = ValidationReport(environment="test")
        variables = [{"name": "db_password", "line": 1, "has_description": True,
                      "has_type": True, "has_default": False, "is_sensitive": True}]
        check_sensitive_variables(variables, "test.tf", report)
        assert report.warning_count == 0

    def test_non_secret_variable_ok(self):
        report = ValidationReport(environment="test")
        variables = [{"name": "region", "line": 1, "has_description": True,
                      "has_type": True, "has_default": True, "is_sensitive": False}]
        check_sensitive_variables(variables, "test.tf", report)
        assert report.warning_count == 0


# ---------------------------------------------------------------------------
# Test: Placeholder detection
# ---------------------------------------------------------------------------


class TestCheckPlaceholderValues:
    def test_detect_placeholder(self):
        report = ValidationReport(environment="test")
        lines = ['  image_id = "ocid1.image.oc1..placeholder"']
        check_placeholder_values(lines, "test.tf", report)
        assert report.warning_count == 1

    def test_comment_todo_skipped(self):
        report = ValidationReport(environment="test")
        lines = ['  # TODO: add more config']
        check_placeholder_values(lines, "test.tf", report)
        assert report.warning_count == 0

    def test_inline_replace_me_detected(self):
        report = ValidationReport(environment="test")
        lines = ['  bucket = "REPLACE_ME"']
        check_placeholder_values(lines, "test.tf", report)
        assert report.warning_count >= 1


# ---------------------------------------------------------------------------
# Test: Environment resolution
# ---------------------------------------------------------------------------


class TestResolveEnvironments:
    def test_resolve_all(self):
        result = resolve_environments("all")
        assert len(result) == len({"eks", "gke", "oke", "shared", "staging", "production"})

    def test_resolve_single(self):
        result = resolve_environments("eks")
        assert len(result) == 1
        assert result[0] == ("eks", "modules/eks")

    def test_resolve_unknown_returns_empty(self):
        result = resolve_environments("azure")
        assert result == []

    def test_resolve_comma_separated(self):
        result = resolve_environments("eks,gke")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Test: Full validation run
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_validate_eks_module(self, tmp_terraform_root):
        report = run_validation(tmp_terraform_root, "eks")
        assert report.files_scanned >= 2  # main.tf + variables.tf + outputs.tf
        assert report.variables_checked > 0

    def test_validate_all_modules(self, tmp_terraform_root):
        report = run_validation(tmp_terraform_root, "all")
        assert report.files_scanned >= 6

    def test_validate_unknown_environment(self, tmp_terraform_root):
        report = run_validation(tmp_terraform_root, "azure")
        assert report.passed is False
        assert any(f.rule == "ENV-UNKNOWN" for f in report.findings)

    def test_strict_mode_fails_on_warnings(self, tmp_terraform_root):
        report = run_validation(tmp_terraform_root, "all", strict=True)
        # Some warnings are expected (backend config, etc.)
        if report.warning_count > 0:
            assert report.passed is False


# ---------------------------------------------------------------------------
# Test: Report generation
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_report_to_dict(self):
        report = ValidationReport(environment="test")
        report.add(Finding(
            rule="TEST", severity=SEVERITY_ERROR,
            message="test error", file="test.tf", line=1,
        ))
        d = report.to_dict()
        assert d["environment"] == "test"
        assert d["passed"] is False
        assert d["summary"]["errors"] == 1
        assert len(d["findings"]) == 1

    def test_report_to_markdown(self):
        report = ValidationReport(environment="test")
        report.add(Finding(
            rule="TEST", severity=SEVERITY_WARNING,
            message="test warning", file="test.tf",
        ))
        md = report.to_markdown()
        assert "# Terraform Validation Report: test" in md
        assert "PASSED" in md  # warnings don't fail by default
        assert "test warning" in md

    def test_empty_report_passed(self):
        report = ValidationReport(environment="test")
        assert report.passed is True
        assert report.error_count == 0
        assert "No findings" in report.to_markdown()

    def test_error_causes_failure(self):
        report = ValidationReport(environment="test")
        report.add(Finding(
            rule="X", severity=SEVERITY_ERROR,
            message="fail", file="x.tf",
        ))
        assert report.passed is False

    def test_finding_to_dict_with_line(self):
        f = Finding(rule="R", severity="error", message="m", file="f.tf", line=42)
        d = f.to_dict()
        assert d["line"] == 42

    def test_finding_to_dict_without_line(self):
        f = Finding(rule="R", severity="info", message="m", file="f.tf")
        d = f.to_dict()
        assert "line" not in d


# ---------------------------------------------------------------------------
# Test: Plan output analysis
# ---------------------------------------------------------------------------


class TestAnalyzePlanOutput:
    def test_no_changes(self):
        output = "No changes. Your infrastructure matches the configuration."
        analysis = analyze_plan_output(output)
        assert analysis.no_changes is True

    def test_plan_summary_parsed(self):
        output = "Plan: 15 to add, 2 to change, 1 to destroy."
        analysis = analyze_plan_output(output)
        assert analysis.to_add == 15
        assert analysis.to_change == 2
        assert analysis.to_destroy == 1
        assert analysis.has_destroys is True

    def test_individual_resources_parsed(self):
        output = textwrap.dedent("""\
            # aws_vpc.main will be created
            # aws_eks_cluster.main will be created
            # aws_subnet.old will be destroyed

            Plan: 2 to add, 0 to change, 1 to destroy.
        """)
        analysis = analyze_plan_output(output)
        assert len(analysis.resources) == 3
        assert analysis.to_add == 2
        assert analysis.to_destroy == 1

    def test_empty_output(self):
        analysis = analyze_plan_output("")
        assert analysis.no_changes is False
        assert analysis.to_add == 0


# ---------------------------------------------------------------------------
# Test: PlanCheckReport
# ---------------------------------------------------------------------------


class TestPlanCheckReport:
    def test_report_defaults_to_passed(self):
        report = PlanCheckReport(environment="staging")
        assert report.passed is True
        assert report.terraform_available is False

    def test_add_error_fails_report(self):
        report = PlanCheckReport(environment="staging")
        report.add_error("test error")
        assert report.passed is False
        assert len(report.errors) == 1

    def test_add_warning_keeps_passing(self):
        report = PlanCheckReport(environment="staging")
        report.add_warning("test warning")
        assert report.passed is True
        assert len(report.warnings) == 1

    def test_to_dict_includes_steps(self):
        report = PlanCheckReport(environment="staging")
        report.init_result = CommandResult("init", 0, "ok", "")
        report.validate_result = CommandResult("validate", 0, "ok", "")
        d = report.to_dict()
        assert d["init"]["success"] is True
        assert d["validate"]["success"] is True

    def test_to_markdown_output(self):
        report = PlanCheckReport(environment="staging")
        report.terraform_available = True
        report.terraform_version = "1.7.0"
        md = report.to_markdown()
        assert "# Terraform Plan Check: staging" in md
        assert "1.7.0" in md


# ---------------------------------------------------------------------------
# Test: CommandResult
# ---------------------------------------------------------------------------


class TestCommandResult:
    def test_success_on_zero_exit(self):
        r = CommandResult("cmd", 0, "out", "")
        assert r.success is True

    def test_failure_on_nonzero_exit(self):
        r = CommandResult("cmd", 1, "", "error")
        assert r.success is False


# ---------------------------------------------------------------------------
# Test: find_terraform_binary
# ---------------------------------------------------------------------------


class TestFindTerraformBinary:
    def test_returns_path_or_none(self):
        # This is a system-dependent test -- just verify it returns str or None
        result = find_terraform_binary()
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# Test: Plan check with missing terraform
# ---------------------------------------------------------------------------


class TestRunPlanCheck:
    def test_unknown_environment(self, tmp_terraform_root):
        report = run_plan_check(tmp_terraform_root, "azure")
        assert report.passed is False
        assert "Unknown environment" in report.errors[0]

    def test_missing_terraform_binary(self, tmp_terraform_root):
        with mock.patch("scripts.terraform_plan_check.find_terraform_binary", return_value=None):
            report = run_plan_check(tmp_terraform_root, "staging")
            assert report.passed is False
            assert report.terraform_available is False
            assert "not found" in report.errors[0].lower()

    def test_successful_init_and_validate(self, tmp_terraform_root):
        """Mock terraform binary to simulate successful init + validate."""
        mock_results = {
            "init": CommandResult("init", 0, "Terraform initialized", ""),
            "validate": CommandResult("validate", 0, "Success!", ""),
            "plan": CommandResult("plan", 1, "", "No valid credential sources found"),
        }

        call_count = [0]
        def mock_run_command(cmd, **kwargs):
            call_count[0] += 1
            if "init" in cmd:
                return mock_results["init"]
            elif "validate" in cmd:
                return mock_results["validate"]
            elif "plan" in cmd:
                return mock_results["plan"]
            elif "version" in cmd:
                return CommandResult("version", 0, '{"terraform_version":"1.7.0"}', "")
            return CommandResult("unknown", 1, "", "unknown command")

        with mock.patch("scripts.terraform_plan_check.find_terraform_binary", return_value="/usr/bin/terraform"), \
             mock.patch("scripts.terraform_plan_check.run_command", side_effect=mock_run_command):
            report = run_plan_check(tmp_terraform_root, "staging")
            assert report.terraform_available is True
            assert report.init_result is not None
            assert report.init_result.success is True
            assert report.validate_result is not None
            assert report.validate_result.success is True
            # Plan fails due to missing credentials (expected)
            assert report.passed is True  # credential failures are warnings

    def test_failed_init_stops_early(self, tmp_terraform_root):
        """If init fails, validate and plan should not run."""
        def mock_run_command(cmd, **kwargs):
            if "init" in cmd:
                return CommandResult("init", 1, "", "Error: failed to init")
            elif "version" in cmd:
                return CommandResult("version", 0, '{"terraform_version":"1.7.0"}', "")
            return CommandResult("unknown", 0, "", "")

        with mock.patch("scripts.terraform_plan_check.find_terraform_binary", return_value="/usr/bin/terraform"), \
             mock.patch("scripts.terraform_plan_check.run_command", side_effect=mock_run_command):
            report = run_plan_check(tmp_terraform_root, "staging")
            assert report.passed is False
            assert report.validate_result is None


# ---------------------------------------------------------------------------
# Test: CLI main functions
# ---------------------------------------------------------------------------


class TestValidateTerraformCLI:
    def test_main_json_output(self, tmp_terraform_root, capsys):
        from scripts.validate_terraform import main
        exit_code = main([
            "--environment", "eks",
            "--terraform-root", str(tmp_terraform_root),
            "--json-only",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "environment" in data
        assert "findings" in data
        assert isinstance(exit_code, int)

    def test_main_markdown_output(self, tmp_terraform_root, capsys):
        from scripts.validate_terraform import main
        main([
            "--environment", "all",
            "--terraform-root", str(tmp_terraform_root),
        ])
        captured = capsys.readouterr()
        assert "Terraform Validation Report" in captured.out

    def test_main_output_dir(self, tmp_terraform_root, tmp_path):
        from scripts.validate_terraform import main
        out_dir = tmp_path / "reports"
        main([
            "--environment", "eks",
            "--terraform-root", str(tmp_terraform_root),
            "--output-dir", str(out_dir),
        ])
        assert out_dir.exists()
        json_files = list(out_dir.glob("*.json"))
        md_files = list(out_dir.glob("*.md"))
        assert len(json_files) == 1
        assert len(md_files) == 1

    def test_main_missing_terraform_root(self, capsys):
        from scripts.validate_terraform import main
        exit_code = main([
            "--terraform-root", "/nonexistent/path/terraform",
        ])
        assert exit_code == 2


class TestPlanCheckCLI:
    def test_main_missing_terraform_root(self, capsys):
        from scripts.terraform_plan_check import main
        exit_code = main([
            "--terraform-root", "/nonexistent/path/terraform",
        ])
        assert exit_code == 2

    def test_main_json_output(self, tmp_terraform_root, capsys):
        from scripts.terraform_plan_check import main
        with mock.patch("scripts.terraform_plan_check.find_terraform_binary", return_value=None):
            main([
                "--environment", "staging",
                "--terraform-root", str(tmp_terraform_root),
                "--json-only",
            ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["terraform_available"] is False


# ---------------------------------------------------------------------------
# Test: Real module validation (against actual terraform/ in repo)
# ---------------------------------------------------------------------------


class TestRealModuleValidation:
    """Validate the actual terraform modules in the repository.

    These tests run against the real terraform/ directory and verify that
    the existing modules pass basic validation.
    """

    @pytest.fixture
    def real_terraform_root(self):
        """Find the real terraform root relative to the test file."""
        # Try multiple paths (repo root, worktree)
        candidates = [
            Path(__file__).resolve().parent.parent / "terraform",
            Path.cwd() / "terraform",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        pytest.skip("Real terraform/ directory not found")

    def test_eks_module_has_no_hardcoded_credentials(self, real_terraform_root):
        report = run_validation(real_terraform_root, "eks")
        cred_findings = [f for f in report.findings if f.rule == "CRED-HARDCODED"]
        assert len(cred_findings) == 0, (
            f"Found hardcoded credentials: {[f.message for f in cred_findings]}"
        )

    def test_all_variables_have_descriptions(self, real_terraform_root):
        report = run_validation(real_terraform_root, "all")
        desc_findings = [f for f in report.findings if f.rule == "VAR-DESC"]
        assert len(desc_findings) == 0, (
            f"Variables missing descriptions: "
            f"{[f'{f.file}:{f.line} {f.message}' for f in desc_findings]}"
        )

    def test_all_variables_have_types(self, real_terraform_root):
        report = run_validation(real_terraform_root, "all")
        type_findings = [f for f in report.findings if f.rule == "VAR-TYPE"]
        assert len(type_findings) == 0, (
            f"Variables missing types: "
            f"{[f'{f.file}:{f.line} {f.message}' for f in type_findings]}"
        )

    def test_modules_have_provider_version_constraints(self, real_terraform_root):
        report = run_validation(real_terraform_root, "all")
        version_findings = [f for f in report.findings if f.rule == "TF-VERSION"]
        assert len(version_findings) == 0, (
            f"Missing version constraints: "
            f"{[f'{f.file}: {f.message}' for f in version_findings]}"
        )

    def test_report_json_is_valid(self, real_terraform_root):
        report = run_validation(real_terraform_root, "all")
        d = report.to_dict()
        # Should be valid JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["environment"] == "all"
        assert "summary" in parsed
