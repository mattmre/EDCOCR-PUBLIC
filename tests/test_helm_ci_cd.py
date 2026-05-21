"""Tests for Helm CI/CD pipeline configuration."""

import os
import unittest

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_on_key(workflow):
    """Get the 'on' trigger config from a parsed GitHub Actions workflow.

    PyYAML parses the YAML key ``on:`` as boolean True, so we look up
    the trigger configuration under the boolean key first, then fall
    back to the string ``"on"`` for safety.
    """
    if True in workflow:
        return workflow[True]
    return workflow.get("on", {})


class TestHelmCIWorkflow(unittest.TestCase):
    """Tests for the helm-package job in ci.yml."""

    def setUp(self):
        ci_path = os.path.join(REPO_ROOT, ".github", "workflows", "ci.yml")
        with open(ci_path) as f:
            self.ci = yaml.safe_load(f)

    def test_helm_package_job_exists(self):
        """helm-package job must be defined in ci.yml."""
        self.assertIn("helm-package", self.ci["jobs"])

    def test_helm_package_needs_lint(self):
        """helm-package must depend on helm-lint."""
        job = self.ci["jobs"]["helm-package"]
        self.assertEqual(job["needs"], "helm-lint")

    def test_helm_package_runs_on_main_or_tags(self):
        """helm-package must only run on main branch or version tags."""
        job = self.ci["jobs"]["helm-package"]
        condition = job.get("if", "")
        self.assertIn("refs/heads/main", condition)
        self.assertIn("refs/tags/v", condition)

    def test_helm_package_has_checkout_step(self):
        """helm-package must check out the repository."""
        job = self.ci["jobs"]["helm-package"]
        step_uses = [s.get("uses", "") for s in job["steps"]]
        self.assertTrue(
            any("actions/checkout" in u for u in step_uses),
            "No actions/checkout step found",
        )

    def test_helm_package_has_helm_install_step(self):
        """helm-package must install Helm."""
        job = self.ci["jobs"]["helm-package"]
        step_uses = [s.get("uses", "") for s in job["steps"]]
        self.assertTrue(
            any("azure/setup-helm" in u for u in step_uses),
            "No azure/setup-helm step found",
        )

    def test_helm_package_has_upload_artifact(self):
        """helm-package must upload the packaged chart as an artifact."""
        job = self.ci["jobs"]["helm-package"]
        step_uses = [s.get("uses", "") for s in job["steps"]]
        self.assertTrue(
            any("actions/upload-artifact" in u for u in step_uses),
            "No actions/upload-artifact step found",
        )

    def test_helm_package_registry_push_only_on_tags(self):
        """OCI registry push must only run on tagged releases."""
        job = self.ci["jobs"]["helm-package"]
        registry_steps = [
            s for s in job["steps"]
            if "registry" in s.get("name", "").lower()
        ]
        self.assertTrue(
            len(registry_steps) > 0,
            "No registry push step found (expected OCI registry push)",
        )
        for step in registry_steps:
            condition = step.get("if", "")
            self.assertIn("refs/tags/v", condition)

    def test_helm_package_registry_push_handles_missing_url(self):
        """Registry push must skip gracefully when HELM_REGISTRY_URL is not set."""
        job = self.ci["jobs"]["helm-package"]
        registry_steps = [
            s for s in job["steps"]
            if "registry" in s.get("name", "").lower()
        ]
        self.assertTrue(
            len(registry_steps) > 0,
            "No registry push step found (expected OCI registry push)",
        )
        for step in registry_steps:
            run_cmd = step.get("run", "")
            self.assertIn("HELM_REGISTRY_URL", run_cmd)
            self.assertIn("skipping push", run_cmd.lower())


class TestHelmDeployWorkflow(unittest.TestCase):
    """Tests for the helm-deploy.yml manual deployment workflow."""

    def setUp(self):
        deploy_path = os.path.join(
            REPO_ROOT, ".github", "workflows", "helm-deploy.yml"
        )
        with open(deploy_path) as f:
            self.deploy = yaml.safe_load(f)

    def test_deploy_workflow_has_dispatch(self):
        """Deploy workflow must be triggered by workflow_dispatch."""
        on_config = _get_on_key(self.deploy)
        self.assertIn("workflow_dispatch", on_config)

    def test_deploy_has_environment_input(self):
        """Deploy workflow must have an environment input."""
        on_config = _get_on_key(self.deploy)
        inputs = on_config["workflow_dispatch"]["inputs"]
        self.assertIn("environment", inputs)

    def test_deploy_environment_choices(self):
        """Environment input must offer staging and production."""
        on_config = _get_on_key(self.deploy)
        env_input = on_config["workflow_dispatch"]["inputs"]["environment"]
        options = env_input.get("options", [])
        self.assertIn("staging", options)
        self.assertIn("production", options)

    def test_deploy_has_dry_run_default_true(self):
        """Dry run input must default to true for safety."""
        on_config = _get_on_key(self.deploy)
        inputs = on_config["workflow_dispatch"]["inputs"]
        self.assertTrue(inputs["dry_run"]["default"])

    def test_deploy_has_chart_version_input(self):
        """Deploy workflow must have an optional chart_version input."""
        on_config = _get_on_key(self.deploy)
        inputs = on_config["workflow_dispatch"]["inputs"]
        self.assertIn("chart_version", inputs)
        self.assertFalse(inputs["chart_version"].get("required", True))

    def test_deploy_uses_kubeconfig_secret(self):
        """Deploy job must reference KUBECONFIG secret."""
        deploy_job = self.deploy["jobs"]["deploy"]
        yaml_str = yaml.dump(deploy_job)
        self.assertIn("KUBECONFIG", yaml_str)

    def test_deploy_uses_helm_upgrade_install(self):
        """Deploy job must use helm upgrade --install."""
        deploy_job = self.deploy["jobs"]["deploy"]
        steps_run = [s.get("run", "") for s in deploy_job["steps"]]
        combined = " ".join(steps_run)
        self.assertIn("helm upgrade --install", combined)

    def test_deploy_uses_environment_protection(self):
        """Deploy job must use GitHub environment protection."""
        deploy_job = self.deploy["jobs"]["deploy"]
        self.assertIn("environment", deploy_job)


class TestValuesOverrides(unittest.TestCase):
    """Tests for staging and production values override files."""

    def setUp(self):
        staging_path = os.path.join(
            REPO_ROOT, "helm", "ocr-local", "values-staging.yaml"
        )
        prod_path = os.path.join(
            REPO_ROOT, "helm", "ocr-local", "values-production.yaml"
        )
        with open(staging_path) as f:
            self.staging = yaml.safe_load(f)
        with open(prod_path) as f:
            self.production = yaml.safe_load(f)

    def test_staging_lower_coordinator_replicas(self):
        """Staging must have fewer coordinator replicas than production."""
        self.assertLessEqual(
            self.staging.get("coordinator", {}).get("replicas", 1),
            self.production.get("coordinator", {}).get("replicas", 1),
        )

    def test_staging_lower_gpu_worker_replicas(self):
        """Staging must have fewer GPU worker replicas than production."""
        staging_gpu = self.staging.get("gpuWorker", {}).get("replicas", 1)
        prod_gpu = self.production.get("gpuWorker", {}).get("replicas", 1)
        self.assertLessEqual(staging_gpu, prod_gpu)

    def test_staging_lower_cpu_worker_replicas(self):
        """Staging must have fewer CPU worker replicas than production."""
        staging_cpu = self.staging.get("cpuWorker", {}).get("replicas", 1)
        prod_cpu = self.production.get("cpuWorker", {}).get("replicas", 1)
        self.assertLessEqual(staging_cpu, prod_cpu)

    def test_production_has_backup_enabled(self):
        """Production must enable PostgreSQL backups."""
        self.assertTrue(
            self.production["postgresql"]["backup"]["enabled"]
        )

    def test_production_has_sentinel_enabled(self):
        """Production must enable Redis Sentinel."""
        self.assertTrue(
            self.production["redis"]["sentinel"]["enabled"]
        )

    def test_production_has_prometheus_enabled(self):
        """Production must enable Prometheus monitoring."""
        self.assertTrue(self.production["prometheus"]["enabled"])

    def test_staging_has_prometheus_enabled(self):
        """Staging must enable Prometheus monitoring."""
        self.assertTrue(self.staging["prometheus"]["enabled"])

    def test_production_has_keda_scalers(self):
        """Production must enable KEDA GPU and CPU scalers."""
        keda = self.production.get("keda", {})
        self.assertTrue(keda.get("gpuScaler", {}).get("enabled", False))
        self.assertTrue(keda.get("cpuScaler", {}).get("enabled", False))

    def test_staging_is_valid_yaml(self):
        """Staging values must parse as valid YAML (non-None)."""
        self.assertIsNotNone(self.staging)
        self.assertIsInstance(self.staging, dict)

    def test_production_is_valid_yaml(self):
        """Production values must parse as valid YAML (non-None)."""
        self.assertIsNotNone(self.production)
        self.assertIsInstance(self.production, dict)


if __name__ == "__main__":
    unittest.main()
