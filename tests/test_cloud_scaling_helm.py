"""Tests for cloud-native Helm chart values — scaling strategy and webhook enrichment.

These tests validate the new values.yaml keys are present and correctly
structured without requiring a running Kubernetes cluster.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Attempt to import YAML parser; skip if not available
yaml = pytest.importorskip("yaml", reason="PyYAML required for Helm values tests")

HELM_VALUES_PATH = Path(__file__).resolve().parent.parent / "helm" / "ocr-local" / "values.yaml"


@pytest.fixture()
def helm_values():
    """Load and parse the Helm values.yaml."""
    with open(HELM_VALUES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestCloudProviderConfig:
    """Test cloudProvider configuration in values.yaml."""

    def test_cloud_provider_present(self, helm_values):
        assert "cloudProvider" in helm_values

    def test_cloud_provider_default(self, helm_values):
        assert helm_values["cloudProvider"] == "generic"

    def test_valid_cloud_providers(self, helm_values):
        valid = {"aws", "gcp", "oracle", "generic"}
        assert helm_values["cloudProvider"] in valid


class TestKedaScalingStrategy:
    """Test KEDA scaling strategy configuration."""

    def test_keda_section_present(self, helm_values):
        assert "keda" in helm_values

    def test_scaling_strategy_present(self, helm_values):
        assert "scalingStrategy" in helm_values["keda"]

    def test_scaling_strategy_default(self, helm_values):
        assert helm_values["keda"]["scalingStrategy"] == "balanced"

    def test_valid_scaling_strategies(self, helm_values):
        valid = {"aggressive", "balanced", "conservative"}
        assert helm_values["keda"]["scalingStrategy"] in valid

    def test_max_replica_count_present(self, helm_values):
        assert "maxReplicaCount" in helm_values["keda"]

    def test_max_replica_count_positive(self, helm_values):
        assert helm_values["keda"]["maxReplicaCount"] > 0


class TestWebhookConfig:
    """Test webhook configuration in values.yaml."""

    def test_webhooks_section_present(self, helm_values):
        assert "webhooks" in helm_values

    def test_enrich_with_entities_present(self, helm_values):
        assert "enrichWithEntities" in helm_values["webhooks"]

    def test_enrich_with_entities_default_false(self, helm_values):
        assert helm_values["webhooks"]["enrichWithEntities"] is False


class TestExistingValuesUntouched:
    """Ensure existing values are not accidentally modified."""

    def test_gpu_worker_autoscaling_exists(self, helm_values):
        assert "gpuWorker" in helm_values
        assert "autoscaling" in helm_values["gpuWorker"]
        assert "enabled" in helm_values["gpuWorker"]["autoscaling"]

    def test_cpu_worker_autoscaling_exists(self, helm_values):
        assert "cpuWorker" in helm_values
        assert "autoscaling" in helm_values["cpuWorker"]

    def test_prometheus_section_exists(self, helm_values):
        assert "prometheus" in helm_values
        assert "enabled" in helm_values["prometheus"]

    def test_ingress_section_exists(self, helm_values):
        assert "ingress" in helm_values
        assert "enabled" in helm_values["ingress"]

    def test_storage_section_exists(self, helm_values):
        assert "storage" in helm_values
        assert "backend" in helm_values["storage"]

    def test_worker_security_exists(self, helm_values):
        assert "workerSecurity" in helm_values
        assert "mTLS" in helm_values["workerSecurity"]
