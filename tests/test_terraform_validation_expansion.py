"""Tests for Terraform validation expansion.

Covers:
- Example tfvars files exist and contain expected variables with REPLACE markers
- OKE module does not contain placeholder OCIDs in default values
- Validation shell script exists and has correct structure
- Terraform deployment guide documentation exists
- OKE variables.tf declares node_image_id and gpu_node_image_id
- Environment main.tf files pass through OKE image variables
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_text(rel_path: str) -> str:
    """Read a project-relative file and return its contents."""
    full = PROJECT_ROOT / rel_path
    assert full.exists(), f"Expected file not found: {full}"
    return full.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test: tfvars.example files
# ---------------------------------------------------------------------------


class TestTfvarsExampleFiles:
    """Verify example tfvars files exist and contain expected content."""

    @pytest.mark.parametrize(
        "env_name",
        ["staging", "production"],
    )
    def test_tfvars_example_exists(self, env_name: str) -> None:
        path = PROJECT_ROOT / f"terraform/environments/{env_name}/terraform.tfvars.example"
        assert path.exists(), f"Missing terraform.tfvars.example for {env_name}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 100, "tfvars.example file appears too short"

    @pytest.mark.parametrize(
        "env_name",
        ["staging", "production"],
    )
    def test_tfvars_example_has_cloud_provider(self, env_name: str) -> None:
        content = _read_text(f"terraform/environments/{env_name}/terraform.tfvars.example")
        assert "cloud_provider" in content

    @pytest.mark.parametrize(
        "env_name",
        ["staging", "production"],
    )
    def test_tfvars_example_has_cluster_name(self, env_name: str) -> None:
        content = _read_text(f"terraform/environments/{env_name}/terraform.tfvars.example")
        assert "cluster_name" in content

    @pytest.mark.parametrize(
        "env_name",
        ["staging", "production"],
    )
    def test_tfvars_example_has_replace_markers(self, env_name: str) -> None:
        """Every tfvars.example should have REPLACE: comments for user-provided values."""
        content = _read_text(f"terraform/environments/{env_name}/terraform.tfvars.example")
        assert "# REPLACE:" in content, (
            f"tfvars.example for {env_name} missing '# REPLACE:' markers"
        )

    @pytest.mark.parametrize(
        "env_name",
        ["staging", "production"],
    )
    def test_tfvars_example_has_aws_section(self, env_name: str) -> None:
        content = _read_text(f"terraform/environments/{env_name}/terraform.tfvars.example")
        assert "aws_region" in content

    @pytest.mark.parametrize(
        "env_name",
        ["staging", "production"],
    )
    def test_tfvars_example_has_gcp_section(self, env_name: str) -> None:
        content = _read_text(f"terraform/environments/{env_name}/terraform.tfvars.example")
        assert "gcp_project_id" in content

    @pytest.mark.parametrize(
        "env_name",
        ["staging", "production"],
    )
    def test_tfvars_example_has_oracle_section(self, env_name: str) -> None:
        content = _read_text(f"terraform/environments/{env_name}/terraform.tfvars.example")
        assert "oci_compartment_id" in content

    @pytest.mark.parametrize(
        "env_name",
        ["staging", "production"],
    )
    def test_tfvars_example_has_oke_image_vars(self, env_name: str) -> None:
        """Example files should reference OKE image OCID variables."""
        content = _read_text(f"terraform/environments/{env_name}/terraform.tfvars.example")
        assert "oci_node_image_id" in content
        assert "oci_gpu_node_image_id" in content

    @pytest.mark.parametrize(
        "env_name",
        ["staging", "production"],
    )
    def test_tfvars_example_has_grafana_password(self, env_name: str) -> None:
        content = _read_text(f"terraform/environments/{env_name}/terraform.tfvars.example")
        assert "grafana_admin_password" in content

    def test_staging_defaults_differ_from_production(self) -> None:
        staging = _read_text("terraform/environments/staging/terraform.tfvars.example")
        production = _read_text("terraform/environments/production/terraform.tfvars.example")
        assert "ocr-local-staging" in staging
        assert "ocr-local-production" in production
        assert staging != production

    def test_tfvars_example_no_real_secrets(self) -> None:
        """Neither example file should contain actual credentials."""
        for env in ("staging", "production"):
            content = _read_text(
                f"terraform/environments/{env}/terraform.tfvars.example"
            )
            # Should not contain real-looking AWS keys
            assert "AKIA" not in content
            # Should not contain real-looking OCID values (more than 20 chars after prefix)
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Active (non-commented) lines should not have real OCIDs
                assert "ocid1.compartment.oc1..aaaaaaa" not in stripped


# ---------------------------------------------------------------------------
# Test: OKE placeholder OCIDs removed
# ---------------------------------------------------------------------------


class TestOkePlaceholderOcids:
    """Verify OKE module no longer contains placeholder image OCIDs."""

    def test_oke_main_no_placeholder_ocid(self) -> None:
        content = _read_text("terraform/modules/oke/main.tf")
        assert "ocid1.image.oc1..placeholder" not in content, (
            "OKE main.tf still contains placeholder image OCID"
        )

    def test_oke_variables_declares_node_image_id(self) -> None:
        content = _read_text("terraform/modules/oke/variables.tf")
        assert 'variable "node_image_id"' in content

    def test_oke_variables_declares_gpu_node_image_id(self) -> None:
        content = _read_text("terraform/modules/oke/variables.tf")
        assert 'variable "gpu_node_image_id"' in content

    def test_oke_main_references_node_image_var(self) -> None:
        content = _read_text("terraform/modules/oke/main.tf")
        assert "var.node_image_id" in content

    def test_oke_main_references_gpu_node_image_var(self) -> None:
        content = _read_text("terraform/modules/oke/main.tf")
        assert "var.gpu_node_image_id" in content

    def test_oke_variables_has_image_lookup_instructions(self) -> None:
        """Variables file should explain how to find the correct OCID."""
        content = _read_text("terraform/modules/oke/variables.tf")
        assert "oci compute image list" in content, (
            "OKE variables.tf should contain OCI CLI instructions for finding images"
        )

    def test_oke_no_placeholder_anywhere(self) -> None:
        """No .tf file in the OKE module should contain 'placeholder'."""
        oke_dir = PROJECT_ROOT / "terraform/modules/oke"
        for tf_file in oke_dir.glob("*.tf"):
            content = tf_file.read_text(encoding="utf-8")
            # Check for placeholder in non-comment lines
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert "placeholder" not in stripped.lower(), (
                    f"{tf_file.name}:{i} contains 'placeholder' in non-comment line"
                )


# ---------------------------------------------------------------------------
# Test: Environment main.tf passes OKE image variables
# ---------------------------------------------------------------------------


class TestEnvironmentOkePassthrough:
    """Verify staging and production main.tf pass image variables to OKE module."""

    @pytest.mark.parametrize("env_name", ["staging", "production"])
    def test_env_declares_oci_node_image_id(self, env_name: str) -> None:
        content = _read_text(f"terraform/environments/{env_name}/main.tf")
        assert "oci_node_image_id" in content

    @pytest.mark.parametrize("env_name", ["staging", "production"])
    def test_env_declares_oci_gpu_node_image_id(self, env_name: str) -> None:
        content = _read_text(f"terraform/environments/{env_name}/main.tf")
        assert "oci_gpu_node_image_id" in content

    @pytest.mark.parametrize("env_name", ["staging", "production"])
    def test_env_passes_image_vars_to_oke_module(self, env_name: str) -> None:
        content = _read_text(f"terraform/environments/{env_name}/main.tf")
        assert "node_image_id" in content
        assert "gpu_node_image_id" in content


# ---------------------------------------------------------------------------
# Test: Validation script
# ---------------------------------------------------------------------------


class TestValidationScript:
    """Verify the shell validation script exists and has correct structure."""

    def test_script_exists(self) -> None:
        path = PROJECT_ROOT / "scripts/validate_terraform.sh"
        assert path.exists(), "scripts/validate_terraform.sh not found"

    def test_script_is_executable(self) -> None:
        path = PROJECT_ROOT / "scripts/validate_terraform.sh"
        if os.name == "nt":
            # On Windows, check that git tracks executable bit via content
            pytest.skip("Executable bit check not reliable on Windows")
        else:
            assert os.access(path, os.X_OK), "validate_terraform.sh is not executable"

    def test_script_has_shebang(self) -> None:
        content = _read_text("scripts/validate_terraform.sh")
        assert content.startswith("#!/"), "Script missing shebang line"
        first_line = content.splitlines()[0]
        assert "bash" in first_line, "Shebang should reference bash"

    def test_script_has_set_options(self) -> None:
        """Script should use set -e or set -euo pipefail for safety."""
        content = _read_text("scripts/validate_terraform.sh")
        assert "set -" in content
        assert "pipefail" in content or "set -e" in content

    def test_script_checks_terraform_installed(self) -> None:
        content = _read_text("scripts/validate_terraform.sh")
        assert "command -v terraform" in content or "which terraform" in content

    def test_script_runs_fmt_check(self) -> None:
        content = _read_text("scripts/validate_terraform.sh")
        assert "terraform fmt" in content
        assert "-check" in content

    def test_script_runs_init_backend_false(self) -> None:
        content = _read_text("scripts/validate_terraform.sh")
        assert "terraform init" in content
        assert "-backend=false" in content

    def test_script_runs_validate(self) -> None:
        content = _read_text("scripts/validate_terraform.sh")
        assert "terraform validate" in content

    def test_script_checks_all_modules(self) -> None:
        content = _read_text("scripts/validate_terraform.sh")
        for module in ("modules/eks", "modules/gke", "modules/oke", "modules/shared"):
            assert module in content, f"Script should validate {module}"

    def test_script_reports_pass_fail(self) -> None:
        content = _read_text("scripts/validate_terraform.sh")
        # Script should have some form of pass/fail reporting
        content_lower = content.lower()
        assert "pass" in content_lower
        assert "fail" in content_lower

    def test_script_exits_nonzero_on_failure(self) -> None:
        """Script should exit non-zero when failures are detected."""
        content = _read_text("scripts/validate_terraform.sh")
        assert "exit 1" in content


# ---------------------------------------------------------------------------
# Test: Deployment guide documentation
# ---------------------------------------------------------------------------


class TestDeploymentGuide:
    """Verify the terraform deployment guide exists and covers key topics."""

    def test_guide_exists(self) -> None:
        path = PROJECT_ROOT / "docs/operations/terraform-deployment-guide.md"
        assert path.exists(), "Terraform deployment guide not found"

    def test_guide_covers_prerequisites(self) -> None:
        content = _read_text("docs/operations/terraform-deployment-guide.md")
        content_lower = content.lower()
        assert "prerequisite" in content_lower or "pre-requisite" in content_lower

    def test_guide_covers_terraform_cli(self) -> None:
        content = _read_text("docs/operations/terraform-deployment-guide.md")
        assert "terraform" in content.lower()

    def test_guide_covers_all_providers(self) -> None:
        content = _read_text("docs/operations/terraform-deployment-guide.md")
        assert "EKS" in content
        assert "GKE" in content
        assert "OKE" in content

    def test_guide_covers_validation_steps(self) -> None:
        content = _read_text("docs/operations/terraform-deployment-guide.md")
        assert "terraform fmt" in content
        assert "terraform validate" in content
        assert "terraform plan" in content

    def test_guide_covers_validation_gap(self) -> None:
        """Guide should explain what static validation does NOT cover."""
        content = _read_text("docs/operations/terraform-deployment-guide.md")
        content_lower = content.lower()
        assert "does not" in content_lower or "not cover" in content_lower

    def test_guide_covers_oke_image_lookup(self) -> None:
        content = _read_text("docs/operations/terraform-deployment-guide.md")
        assert "OCID" in content or "ocid" in content.lower()
        assert "image" in content.lower()

    def test_guide_covers_ci_integration(self) -> None:
        content = _read_text("docs/operations/terraform-deployment-guide.md")
        assert "CI" in content or "ci" in content.lower() or "github" in content.lower()

    def test_guide_references_validation_script(self) -> None:
        content = _read_text("docs/operations/terraform-deployment-guide.md")
        assert "validate_terraform.sh" in content
