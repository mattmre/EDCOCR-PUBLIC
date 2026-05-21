"""Tests for Terraform CI validation configuration.

Validates that:
- All .tf files exist and are non-empty
- terraform/.tflint.hcl exists with valid structure
- CI workflow YAML has a terraform-validate job with expected steps

Run with: python -m pytest tests/test_terraform_ci.py -v
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TERRAFORM_DIR = REPO_ROOT / "terraform"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Terraform files exist and are parseable
# ---------------------------------------------------------------------------

class TestTerraformFilesExist:
    """Verify all expected .tf files are present and non-empty."""

    EXPECTED_TF_FILES = [
        "environments/staging/main.tf",
        "environments/production/main.tf",
        "modules/eks/main.tf",
        "modules/eks/outputs.tf",
        "modules/eks/variables.tf",
        "modules/gke/main.tf",
        "modules/gke/outputs.tf",
        "modules/gke/variables.tf",
        "modules/oke/main.tf",
        "modules/oke/outputs.tf",
        "modules/oke/variables.tf",
        "modules/shared/keda.tf",
        "modules/shared/monitoring.tf",
    ]

    @pytest.mark.parametrize("tf_path", EXPECTED_TF_FILES)
    def test_tf_file_exists(self, tf_path):
        full_path = TERRAFORM_DIR / tf_path
        assert full_path.exists(), f"Missing Terraform file: {tf_path}"

    @pytest.mark.parametrize("tf_path", EXPECTED_TF_FILES)
    def test_tf_file_nonempty(self, tf_path):
        full_path = TERRAFORM_DIR / tf_path
        if full_path.exists():
            content = full_path.read_text(encoding="utf-8").strip()
            assert len(content) > 0, f"Terraform file is empty: {tf_path}"

    def test_tf_files_contain_terraform_block(self):
        """Root modules and child modules should declare a terraform block."""
        root_modules = [
            "environments/staging/main.tf",
            "environments/production/main.tf",
        ]
        child_modules = [
            "modules/eks/main.tf",
            "modules/gke/main.tf",
            "modules/oke/main.tf",
            "modules/shared/keda.tf",
        ]
        for tf_path in root_modules + child_modules:
            full_path = TERRAFORM_DIR / tf_path
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                assert "terraform {" in content or "terraform{" in content, (
                    f"{tf_path} missing terraform block"
                )

    def test_required_version_declared(self):
        """Root modules should declare required_version >= 1.5."""
        for env in ["staging", "production"]:
            main_tf = TERRAFORM_DIR / "environments" / env / "main.tf"
            if main_tf.exists():
                content = main_tf.read_text(encoding="utf-8")
                assert "required_version" in content, (
                    f"environments/{env}/main.tf missing required_version"
                )
                assert ">= 1.5" in content, (
                    f"environments/{env}/main.tf should require Terraform >= 1.5"
                )

    def test_total_tf_file_count(self):
        """Verify there are exactly 13 .tf files in the terraform directory."""
        tf_files = list(TERRAFORM_DIR.rglob("*.tf"))
        assert len(tf_files) == 13, (
            f"Expected 13 .tf files, found {len(tf_files)}: "
            + ", ".join(str(f.relative_to(TERRAFORM_DIR)) for f in tf_files)
        )


# ---------------------------------------------------------------------------
# TFLint configuration
# ---------------------------------------------------------------------------

class TestTflintConfig:
    """Validate terraform/.tflint.hcl exists and has expected content."""

    def test_tflint_config_exists(self):
        tflint_path = TERRAFORM_DIR / ".tflint.hcl"
        assert tflint_path.exists(), "terraform/.tflint.hcl not found"

    def test_tflint_config_nonempty(self):
        tflint_path = TERRAFORM_DIR / ".tflint.hcl"
        if tflint_path.exists():
            content = tflint_path.read_text(encoding="utf-8").strip()
            assert len(content) > 0, "terraform/.tflint.hcl is empty"

    def test_tflint_has_terraform_plugin(self):
        tflint_path = TERRAFORM_DIR / ".tflint.hcl"
        if tflint_path.exists():
            content = tflint_path.read_text(encoding="utf-8")
            assert 'plugin "terraform"' in content, (
                ".tflint.hcl should declare the terraform plugin"
            )

    def test_tflint_plugin_enabled(self):
        tflint_path = TERRAFORM_DIR / ".tflint.hcl"
        if tflint_path.exists():
            content = tflint_path.read_text(encoding="utf-8")
            assert "enabled = true" in content, (
                ".tflint.hcl terraform plugin should be enabled"
            )

    def test_tflint_has_naming_convention_rule(self):
        tflint_path = TERRAFORM_DIR / ".tflint.hcl"
        if tflint_path.exists():
            content = tflint_path.read_text(encoding="utf-8")
            assert "terraform_naming_convention" in content, (
                ".tflint.hcl should include terraform_naming_convention rule"
            )

    def test_tflint_has_documented_variables_rule(self):
        tflint_path = TERRAFORM_DIR / ".tflint.hcl"
        if tflint_path.exists():
            content = tflint_path.read_text(encoding="utf-8")
            assert "terraform_documented_variables" in content, (
                ".tflint.hcl should include terraform_documented_variables rule"
            )

    def test_tflint_has_unused_declarations_rule(self):
        tflint_path = TERRAFORM_DIR / ".tflint.hcl"
        if tflint_path.exists():
            content = tflint_path.read_text(encoding="utf-8")
            assert "terraform_unused_declarations" in content, (
                ".tflint.hcl should include terraform_unused_declarations rule"
            )


# ---------------------------------------------------------------------------
# CI workflow terraform-validate job
# ---------------------------------------------------------------------------

class TestCITerraformJob:
    """Validate the terraform-validate job in .github/workflows/ci.yml."""

    @pytest.fixture(autouse=True)
    def _load(self):
        assert CI_WORKFLOW.exists(), f"CI workflow not found at {CI_WORKFLOW}"
        self.cfg = _load_yaml(CI_WORKFLOW)
        self.jobs = self.cfg.get("jobs", {})

    def test_terraform_validate_job_exists(self):
        assert "terraform-validate" in self.jobs, (
            "CI workflow missing 'terraform-validate' job"
        )

    def test_runs_on_ubuntu(self):
        job = self.jobs.get("terraform-validate", {})
        assert job.get("runs-on") == "ubuntu-latest"

    def test_timeout_set(self):
        job = self.jobs.get("terraform-validate", {})
        timeout = job.get("timeout-minutes")
        assert timeout is not None, "terraform-validate should have a timeout"
        assert timeout <= 15, "Timeout should be 15 minutes or less"

    def test_has_checkout_step(self):
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        checkout_steps = [
            s for s in steps
            if s.get("uses", "").startswith("actions/checkout")
        ]
        assert len(checkout_steps) >= 1, "Missing checkout step"

    def test_has_setup_terraform_step(self):
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        tf_setup = [
            s for s in steps
            if "setup-terraform" in s.get("uses", "")
        ]
        assert len(tf_setup) >= 1, "Missing hashicorp/setup-terraform step"

    def test_terraform_version_pinned(self):
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        for s in steps:
            if "setup-terraform" in s.get("uses", ""):
                tf_version = s.get("with", {}).get("terraform_version", "")
                assert tf_version, "Terraform version should be pinned"
                assert tf_version.startswith("1.5"), (
                    f"Expected Terraform 1.5.x, got {tf_version}"
                )
                return
        pytest.fail("setup-terraform step not found")

    def test_has_fmt_check_step(self):
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        fmt_steps = [
            s for s in steps
            if "fmt" in s.get("name", "").lower() or "fmt" in s.get("run", "")
        ]
        assert len(fmt_steps) >= 1, "Missing terraform fmt check step"
        # Verify it uses -check flag
        for s in fmt_steps:
            run_cmd = s.get("run", "")
            if "fmt" in run_cmd:
                assert "-check" in run_cmd, "fmt step should use -check flag"

    def test_has_validate_staging_step(self):
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        validate_steps = [
            s for s in steps
            if "staging" in s.get("name", "").lower()
            and ("validate" in s.get("name", "").lower() or "validate" in s.get("run", ""))
        ]
        assert len(validate_steps) >= 1, "Missing terraform validate step for staging"

    def test_has_validate_production_step(self):
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        validate_steps = [
            s for s in steps
            if "production" in s.get("name", "").lower()
            and ("validate" in s.get("name", "").lower() or "validate" in s.get("run", ""))
        ]
        assert len(validate_steps) >= 1, "Missing terraform validate step for production"

    def test_validate_uses_backend_false(self):
        """terraform init should use -backend=false for CI validation."""
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        for s in steps:
            run_cmd = s.get("run", "")
            if "terraform init" in run_cmd and "terraform validate" in run_cmd:
                assert "-backend=false" in run_cmd, (
                    "terraform init should use -backend=false in CI"
                )

    def test_has_tflint_setup_step(self):
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        tflint_setup = [
            s for s in steps
            if "setup-tflint" in s.get("uses", "")
        ]
        assert len(tflint_setup) >= 1, "Missing terraform-linters/setup-tflint step"

    def test_has_tflint_run_steps(self):
        """At least one tflint execution step should exist."""
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        tflint_run = [
            s for s in steps
            if "tflint" in s.get("run", "") and "setup" not in s.get("uses", "")
        ]
        assert len(tflint_run) >= 1, "Missing tflint run step"

    def test_tflint_steps_are_blocking(self):
        """TFLint steps must be blocking (no continue-on-error)."""
        job = self.jobs.get("terraform-validate", {})
        steps = job.get("steps", [])
        tflint_run = [
            s for s in steps
            if "tflint" in s.get("name", "").lower()
            and "setup" not in s.get("name", "").lower()
            and s.get("run")  # has a run command (not just uses)
        ]
        assert len(tflint_run) >= 1, "Expected at least one TFLint run step"
        for s in tflint_run:
            assert s.get("continue-on-error") is not True, (
                f"TFLint step '{s.get('name')}' should not have continue-on-error"
            )

    def test_does_not_modify_existing_jobs(self):
        """Verify the existing 11 jobs still exist alongside terraform-validate."""
        expected_jobs = [
            "root-lint-and-tests",
            "coordinator-lint-and-tests",
            "sdk-tests",
            "typescript-sdk-tests",
            "version-consistency",
            "helm-lint",
            "resume-regression",
            "playwright-smoke",
            "security-audit",
            "generate-sbom",
            "helm-package",
        ]
        for job_name in expected_jobs:
            assert job_name in self.jobs, (
                f"Existing job '{job_name}' is missing from CI workflow"
            )
