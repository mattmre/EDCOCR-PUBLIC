"""Tests for LayoutLMv3 Dockerfile, KEDA scaler, and Helm deployment templates.

Validates:
  - Dockerfile.worker.layoutlm syntax and structure
  - KEDA ScaledObject template renders correctly for the ocr_layoutlm queue
  - values.yaml includes layoutlmWorker section with correct defaults
  - Deployment template structure for LayoutLMv3 worker
  - TriggerAuthentication guard covers layoutlmWorker
  - _helpers.tpl includes layoutlmWorkerImage helper

Run with: python -m pytest tests/test_layoutlm_dockerfile.py -v
"""

import os
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML required for Helm values tests")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COORDINATOR_DIR = PROJECT_ROOT / "coordinator"
HELM_DIR = PROJECT_ROOT / "helm" / "ocr-local"
TEMPLATES_DIR = HELM_DIR / "templates"
VALUES_PATH = HELM_DIR / "values.yaml"


@pytest.fixture()
def dockerfile_content():
    """Load the LayoutLMv3 Dockerfile content."""
    path = COORDINATOR_DIR / "Dockerfile.worker.layoutlm"
    if not path.exists():
        pytest.skip("Dockerfile.worker.layoutlm not found")
    return path.read_text(encoding="utf-8")


@pytest.fixture()
def keda_scaler_content():
    """Load the KEDA LayoutLMv3 scaler template."""
    path = TEMPLATES_DIR / "keda-layoutlm-scaler.yaml"
    if not path.exists():
        pytest.skip("keda-layoutlm-scaler.yaml not found")
    return path.read_text(encoding="utf-8")


@pytest.fixture()
def deployment_content():
    """Load the LayoutLMv3 worker deployment template."""
    path = TEMPLATES_DIR / "layoutlm-worker-deployment.yaml"
    if not path.exists():
        pytest.skip("layoutlm-worker-deployment.yaml not found")
    return path.read_text(encoding="utf-8")


@pytest.fixture()
def helpers_content():
    """Load the Helm _helpers.tpl template."""
    path = TEMPLATES_DIR / "_helpers.tpl"
    if not path.exists():
        pytest.skip("_helpers.tpl not found")
    return path.read_text(encoding="utf-8")


@pytest.fixture()
def helm_values():
    """Load and parse the Helm values.yaml."""
    if not VALUES_PATH.exists():
        pytest.skip("values.yaml not found")
    with open(VALUES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def trigger_auth_content():
    """Load the TriggerAuthentication template."""
    path = TEMPLATES_DIR / "keda-trigger-auth.yaml"
    if not path.exists():
        pytest.skip("keda-trigger-auth.yaml not found")
    return path.read_text(encoding="utf-8")


# ===========================================================================
# Dockerfile tests
# ===========================================================================


class TestLayoutlmDockerfile:
    """Validate Dockerfile.worker.layoutlm structure and content."""

    def test_dockerfile_exists(self):
        """Dockerfile.worker.layoutlm must exist in coordinator/."""
        path = COORDINATOR_DIR / "Dockerfile.worker.layoutlm"
        assert path.exists(), "coordinator/Dockerfile.worker.layoutlm not found"

    def test_base_image_is_python311(self, dockerfile_content):
        """Base image should be python:3.11-slim-bookworm."""
        assert "python:3.11-slim-bookworm" in dockerfile_content

    def test_installs_torch(self, dockerfile_content):
        """Dockerfile must install torch for LayoutLMv3."""
        assert "torch" in dockerfile_content

    def test_installs_transformers(self, dockerfile_content):
        """Dockerfile must install transformers for LayoutLMv3."""
        assert "transformers" in dockerfile_content

    def test_installs_timm(self, dockerfile_content):
        """Dockerfile must install timm (required by LayoutLMv3)."""
        assert "timm" in dockerfile_content

    def test_has_gpu_build_arg(self, dockerfile_content):
        """Dockerfile should support USE_GPU build arg."""
        assert "ARG USE_GPU" in dockerfile_content

    def test_has_model_build_arg(self, dockerfile_content):
        """Dockerfile should support LAYOUTLM_MODEL build arg."""
        assert "ARG LAYOUTLM_MODEL" in dockerfile_content

    def test_prebakes_model(self, dockerfile_content):
        """Dockerfile should pre-download the LayoutLMv3 model at build time."""
        assert "microsoft/layoutlmv3-base" in dockerfile_content
        assert "AutoModel.from_pretrained" in dockerfile_content
        assert "AutoTokenizer.from_pretrained" in dockerfile_content

    def test_sets_hf_home(self, dockerfile_content):
        """Dockerfile should set HF_HOME for model cache directory."""
        assert "HF_HOME" in dockerfile_content

    def test_celery_queue_is_layoutlm(self, dockerfile_content):
        """CMD should start celery worker on ocr_layoutlm queue."""
        assert "ocr_layoutlm" in dockerfile_content

    def test_sets_enable_layoutlm(self, dockerfile_content):
        """Dockerfile should set ENABLE_LAYOUTLM=true."""
        assert "ENABLE_LAYOUTLM=true" in dockerfile_content

    def test_copies_coordinator_requirements(self, dockerfile_content):
        """Dockerfile should install coordinator requirements."""
        assert "coordinator/requirements.txt" in dockerfile_content

    def test_sets_pythonpath(self, dockerfile_content):
        """Dockerfile should set PYTHONPATH to include app and coordinator."""
        assert "PYTHONPATH=/app:/app/coordinator" in dockerfile_content

    def test_sets_django_settings(self, dockerfile_content):
        """Dockerfile should set DJANGO_SETTINGS_MODULE."""
        assert "DJANGO_SETTINGS_MODULE=coordinator.settings" in dockerfile_content

    def test_workdir_is_app(self, dockerfile_content):
        """WORKDIR should be /app."""
        assert "WORKDIR /app" in dockerfile_content

    def test_has_from_statement(self, dockerfile_content):
        """Dockerfile must have a FROM statement."""
        assert re.search(r"^FROM\s+", dockerfile_content, re.MULTILINE)

    def test_hostname_includes_layoutlm(self, dockerfile_content):
        """Worker hostname should identify itself as layoutlm worker."""
        assert "worker-layoutlm@%h" in dockerfile_content

    def test_cpu_torch_install_uses_cpu_index(self, dockerfile_content):
        """CPU-only torch install should use PyTorch CPU index URL."""
        assert "download.pytorch.org/whl/cpu" in dockerfile_content


# ===========================================================================
# KEDA scaler template tests
# ===========================================================================


class TestKedaLayoutlmScaler:
    """Validate KEDA ScaledObject for LayoutLMv3 queue."""

    def test_scaler_template_exists(self):
        """keda-layoutlm-scaler.yaml must exist in templates/."""
        path = TEMPLATES_DIR / "keda-layoutlm-scaler.yaml"
        assert path.exists(), "keda-layoutlm-scaler.yaml not found"

    def test_kind_is_scaled_object(self, keda_scaler_content):
        """Template must define a ScaledObject resource."""
        assert "kind: ScaledObject" in keda_scaler_content

    def test_api_version_is_keda(self, keda_scaler_content):
        """Template must use keda.sh/v1alpha1 API."""
        assert "apiVersion: keda.sh/v1alpha1" in keda_scaler_content

    def test_queue_name_is_layoutlm(self, keda_scaler_content):
        """Trigger queueName must be ocr_layoutlm."""
        assert "queueName: ocr_layoutlm" in keda_scaler_content

    def test_trigger_type_is_rabbitmq(self, keda_scaler_content):
        """Trigger type must be rabbitmq."""
        assert "type: rabbitmq" in keda_scaler_content

    def test_has_authentication_ref(self, keda_scaler_content):
        """ScaledObject must reference rabbitmq-auth TriggerAuthentication."""
        assert "rabbitmq-auth" in keda_scaler_content
        assert "authenticationRef:" in keda_scaler_content

    def test_guard_condition_checks_enabled(self, keda_scaler_content):
        """Guard condition must check layoutlmWorker.enabled AND autoscaling.enabled."""
        first_line = keda_scaler_content.split("\n")[0]
        assert "layoutlmWorker.enabled" in first_line
        assert "layoutlmWorker.autoscaling.enabled" in first_line

    def test_component_label(self, keda_scaler_content):
        """Must have app.kubernetes.io/component: layoutlm-worker."""
        assert "app.kubernetes.io/component: layoutlm-worker" in keda_scaler_content

    def test_scale_target_ref(self, keda_scaler_content):
        """scaleTargetRef must reference layoutlm-worker deployment."""
        assert "layoutlm-worker" in keda_scaler_content

    def test_min_replicas_from_values(self, keda_scaler_content):
        """minReplicaCount should reference values."""
        assert ".Values.layoutlmWorker.autoscaling.minReplicas" in keda_scaler_content

    def test_max_replicas_from_values(self, keda_scaler_content):
        """maxReplicaCount should reference values with global cap."""
        assert ".Values.layoutlmWorker.autoscaling.maxReplicas" in keda_scaler_content
        assert ".Values.keda.maxReplicaCount" in keda_scaler_content

    def test_supports_scaling_strategy(self, keda_scaler_content):
        """Template should support aggressive/conservative scaling strategies."""
        assert "scalingStrategy" in keda_scaler_content
        assert "aggressive" in keda_scaler_content
        assert "conservative" in keda_scaler_content

    def test_has_cloud_provider_annotation(self, keda_scaler_content):
        """Template should include cloud-provider annotation."""
        assert "ocr-local/cloud-provider" in keda_scaler_content


# ===========================================================================
# Deployment template tests
# ===========================================================================


class TestLayoutlmWorkerDeployment:
    """Validate LayoutLMv3 worker Kubernetes Deployment."""

    def test_deployment_template_exists(self):
        """layoutlm-worker-deployment.yaml must exist in templates/."""
        path = TEMPLATES_DIR / "layoutlm-worker-deployment.yaml"
        assert path.exists(), "layoutlm-worker-deployment.yaml not found"

    def test_kind_is_deployment(self, deployment_content):
        """Template must define a Deployment resource."""
        assert "kind: Deployment" in deployment_content

    def test_guard_condition(self, deployment_content):
        """Guard condition must check layoutlmWorker.enabled."""
        assert "layoutlmWorker.enabled" in deployment_content

    def test_component_label(self, deployment_content):
        """Must have app.kubernetes.io/component: layoutlm-worker."""
        assert "app.kubernetes.io/component: layoutlm-worker" in deployment_content

    def test_celery_queue_command(self, deployment_content):
        """Container command must start celery with ocr_layoutlm queue."""
        assert "ocr_layoutlm" in deployment_content

    def test_hostname_is_layoutlm_worker(self, deployment_content):
        """Worker hostname should be layoutlm-worker."""
        assert "layoutlm-worker@%h" in deployment_content

    def test_enable_layoutlm_env(self, deployment_content):
        """Deployment must set ENABLE_LAYOUTLM=true."""
        assert "ENABLE_LAYOUTLM" in deployment_content

    def test_model_path_env(self, deployment_content):
        """Deployment must set LAYOUTLM_MODEL_PATH from values."""
        assert "LAYOUTLM_MODEL_PATH" in deployment_content

    def test_hf_home_env(self, deployment_content):
        """Deployment must set HF_HOME for model cache."""
        assert "HF_HOME" in deployment_content

    def test_startup_probe(self, deployment_content):
        """Deployment must have a startup probe checking torch+transformers."""
        assert "startupProbe" in deployment_content
        assert "import torch" in deployment_content
        assert "import transformers" in deployment_content

    def test_liveness_probe(self, deployment_content):
        """Deployment must have a liveness probe using celery inspect ping."""
        assert "livenessProbe" in deployment_content
        assert "inspect" in deployment_content
        assert "ping" in deployment_content

    def test_mtls_annotation(self, deployment_content):
        """Deployment should support mTLS annotation when enabled."""
        assert "workerSecurity.mTLS.enabled" in deployment_content

    def test_staging_volume(self, deployment_content):
        """Deployment should mount the encrypted staging volume."""
        assert "staging-volume" in deployment_content

    def test_shared_memory_gpu(self, deployment_content):
        """Deployment should mount /dev/shm when GPU is enabled."""
        assert "dshm" in deployment_content
        assert "/dev/shm" in deployment_content

    def test_autoscaling_replica_guard(self, deployment_content):
        """Replicas should be omitted when autoscaling is enabled."""
        assert "layoutlmWorker.autoscaling.enabled" in deployment_content

    def test_envfrom_configmap_and_secret(self, deployment_content):
        """Deployment must use envFrom for configmap and secret."""
        assert "configMapRef" in deployment_content
        assert "secretRef" in deployment_content

    def test_custom_env_loop(self, deployment_content):
        """Deployment should support custom env vars from values."""
        assert "layoutlmWorker.env" in deployment_content

    def test_uses_layoutlm_worker_image(self, deployment_content):
        """Deployment should use the layoutlmWorkerImage helper."""
        assert "layoutlmWorkerImage" in deployment_content


# ===========================================================================
# values.yaml tests
# ===========================================================================


class TestLayoutlmValuesYaml:
    """Validate layoutlmWorker section in values.yaml."""

    def test_layoutlm_worker_present(self, helm_values):
        """values.yaml must include layoutlmWorker section."""
        assert "layoutlmWorker" in helm_values

    def test_disabled_by_default(self, helm_values):
        """layoutlmWorker should be disabled by default."""
        assert helm_values["layoutlmWorker"]["enabled"] is False

    def test_replicas_default(self, helm_values):
        """Default replicas should be 1."""
        assert helm_values["layoutlmWorker"]["replicas"] == 1

    def test_concurrency_default(self, helm_values):
        """Default concurrency should be 2 (memory-heavy inference)."""
        assert helm_values["layoutlmWorker"]["concurrency"] == 2

    def test_queues_default(self, helm_values):
        """Default queue should be ocr_layoutlm."""
        assert helm_values["layoutlmWorker"]["queues"] == "ocr_layoutlm"

    def test_autoscaling_section(self, helm_values):
        """layoutlmWorker must have autoscaling section."""
        autoscaling = helm_values["layoutlmWorker"]["autoscaling"]
        assert "enabled" in autoscaling
        assert "minReplicas" in autoscaling
        assert "maxReplicas" in autoscaling
        assert "pollingInterval" in autoscaling
        assert "cooldownPeriod" in autoscaling
        assert "queueTarget" in autoscaling

    def test_autoscaling_disabled_by_default(self, helm_values):
        """Autoscaling should be disabled by default."""
        assert helm_values["layoutlmWorker"]["autoscaling"]["enabled"] is False

    def test_autoscaling_min_replicas(self, helm_values):
        """minReplicas should default to 0 (scale-to-zero supported)."""
        assert helm_values["layoutlmWorker"]["autoscaling"]["minReplicas"] == 0

    def test_autoscaling_max_replicas(self, helm_values):
        """maxReplicas should default to 3."""
        assert helm_values["layoutlmWorker"]["autoscaling"]["maxReplicas"] == 3

    def test_gpu_section(self, helm_values):
        """layoutlmWorker must have gpu section."""
        gpu = helm_values["layoutlmWorker"]["gpu"]
        assert "enabled" in gpu
        assert "count" in gpu

    def test_gpu_enabled_by_default(self, helm_values):
        """GPU should be enabled by default for LayoutLMv3."""
        assert helm_values["layoutlmWorker"]["gpu"]["enabled"] is True

    def test_model_section(self, helm_values):
        """layoutlmWorker must have model section."""
        model = helm_values["layoutlmWorker"]["model"]
        assert "name" in model
        assert "cacheDir" in model

    def test_model_name_default(self, helm_values):
        """Default model should be microsoft/layoutlmv3-base."""
        assert helm_values["layoutlmWorker"]["model"]["name"] == "microsoft/layoutlmv3-base"

    def test_model_cache_dir(self, helm_values):
        """Model cache directory should default to /models."""
        assert helm_values["layoutlmWorker"]["model"]["cacheDir"] == "/models"

    def test_resources_section(self, helm_values):
        """layoutlmWorker must have resources section."""
        resources = helm_values["layoutlmWorker"]["resources"]
        assert "requests" in resources
        assert "limits" in resources

    def test_memory_request_default(self, helm_values):
        """Memory request should be at least 2Gi."""
        mem = helm_values["layoutlmWorker"]["resources"]["requests"]["memory"]
        assert mem == "2Gi"

    def test_capabilities(self, helm_values):
        """Worker should have layoutlm capability."""
        caps = helm_values["layoutlmWorker"]["capabilities"]
        assert "layoutlm" in caps

    def test_image_section(self, helm_values):
        """Image section must include layoutlmWorker."""
        assert "layoutlmWorker" in helm_values["image"]
        assert "repository" in helm_values["image"]["layoutlmWorker"]

    def test_image_repository(self, helm_values):
        """Image repository should be ocr-local/worker-layoutlm."""
        assert helm_values["image"]["layoutlmWorker"]["repository"] == "ocr-local/worker-layoutlm"


# ===========================================================================
# _helpers.tpl tests
# ===========================================================================


class TestHelpersTemplate:
    """Validate _helpers.tpl includes layoutlmWorkerImage."""

    def test_layoutlm_worker_image_helper(self, helpers_content):
        """_helpers.tpl must define ocr-local.layoutlmWorkerImage."""
        assert "ocr-local.layoutlmWorkerImage" in helpers_content

    def test_layoutlm_image_uses_correct_values_path(self, helpers_content):
        """layoutlmWorkerImage helper should reference image.layoutlmWorker."""
        assert ".Values.image.layoutlmWorker.tag" in helpers_content
        assert ".Values.image.layoutlmWorker.repository" in helpers_content


# ===========================================================================
# TriggerAuthentication guard coverage tests
# ===========================================================================


class TestTriggerAuthGuardCoverage:
    """Validate TriggerAuthentication guard includes layoutlmWorker."""

    def test_trigger_auth_covers_layoutlm(self, trigger_auth_content):
        """TriggerAuthentication guard must include layoutlmWorker check."""
        first_line = trigger_auth_content.split("\n")[0]
        assert "layoutlmWorker" in first_line, (
            "TriggerAuthentication guard condition is missing layoutlmWorker. "
            "First line: " + first_line
        )

    def test_trigger_auth_checks_enabled_and_autoscaling(self, trigger_auth_content):
        """Guard must check both layoutlmWorker.enabled AND autoscaling.enabled."""
        first_line = trigger_auth_content.split("\n")[0]
        assert "layoutlmWorker.enabled" in first_line
        assert "layoutlmWorker.autoscaling.enabled" in first_line


# ===========================================================================
# validate_keda.py integration tests
# ===========================================================================


class TestValidateKedaIntegration:
    """Verify validate_keda.py knows about the LayoutLMv3 worker."""

    def test_expected_queues_includes_layoutlm(self):
        """EXPECTED_QUEUES must include layoutlm-worker."""
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
            ),
        )
        from scripts.validate_keda import EXPECTED_QUEUES

        assert "layoutlm-worker" in EXPECTED_QUEUES
        assert EXPECTED_QUEUES["layoutlm-worker"] == "ocr_layoutlm"

    def test_full_validation_finds_layoutlm_scaler(self):
        """Full validation should find 6 ScaledObjects including layoutlm."""
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
            ),
        )
        from scripts.validate_keda import run_validation

        chart_dir = HELM_DIR
        if not (chart_dir / "templates").is_dir():
            pytest.skip("Helm chart not found at expected path")

        report = run_validation(chart_dir)
        components = [so.component for so in report.scaled_objects]
        assert "layoutlm-worker" in components, (
            f"Expected layoutlm-worker in ScaledObject components, "
            f"found: {components}"
        )

    def test_full_validation_passes(self):
        """Full KEDA validation should pass with no errors after adding layoutlm."""
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
            ),
        )
        from scripts.validate_keda import run_validation

        chart_dir = HELM_DIR
        if not (chart_dir / "templates").is_dir():
            pytest.skip("Helm chart not found at expected path")

        report = run_validation(chart_dir)
        errors = [r for r in report.results if not r.passed and r.severity == "error"]
        assert len(errors) == 0, (
            "Validation errors:\n"
            + "\n".join(f"  {r.check}: {r.message}" for r in errors)
        )
