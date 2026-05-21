"""Tests for Helm chart RBAC / ServiceAccount configuration.

Validates that:
- values.yaml has rbac.create defaulting to true
- serviceaccount.yaml template exists with correct structure
- All deployment templates reference serviceAccountName
"""

import os

import pytest
import yaml

HELM_DIR = os.path.join(os.path.dirname(__file__), "..", "helm", "ocr-local")
VALUES_PATH = os.path.join(HELM_DIR, "values.yaml")
TEMPLATES_DIR = os.path.join(HELM_DIR, "templates")
SA_TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, "serviceaccount.yaml")
HELPERS_PATH = os.path.join(TEMPLATES_DIR, "_helpers.tpl")

# All deployment template files that should have serviceAccountName
DEPLOYMENT_TEMPLATES = [
    "coordinator-deployment.yaml",
    "gpu-worker-deployment.yaml",
    "cpu-worker-deployment.yaml",
    "celery-beat-deployment.yaml",
    "celery-coordinator-deployment.yaml",
    "flower-deployment.yaml",
    "cpu-ocr-worker-deployment.yaml",
    "layout-cpu-worker-deployment.yaml",
    "nlp-gpu-worker-deployment.yaml",
    "layoutlm-worker-deployment.yaml",
]


@pytest.fixture(scope="module")
def values():
    """Load the Helm values.yaml file."""
    with open(VALUES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestRbacValues:
    """Tests for RBAC configuration in values.yaml."""

    def test_rbac_section_exists(self, values):
        """values.yaml must have an rbac section."""
        assert "rbac" in values, "rbac section missing from values.yaml"

    def test_rbac_create_defaults_true(self, values):
        """rbac.create must default to true."""
        assert values["rbac"]["create"] is True, (
            "rbac.create must default to true for pod identity separation"
        )


class TestServiceAccountTemplate:
    """Tests for serviceaccount.yaml template."""

    def test_serviceaccount_template_exists(self):
        """serviceaccount.yaml template must exist."""
        assert os.path.isfile(SA_TEMPLATE_PATH), (
            f"serviceaccount.yaml template missing at {SA_TEMPLATE_PATH}"
        )

    def test_serviceaccount_template_has_conditional(self):
        """serviceaccount.yaml must be guarded by rbac.create."""
        with open(SA_TEMPLATE_PATH, encoding="utf-8") as f:
            content = f.read()
        assert ".Values.rbac.create" in content, (
            "serviceaccount.yaml must be conditional on .Values.rbac.create"
        )

    def test_serviceaccount_template_has_api_version(self):
        """serviceaccount.yaml must declare apiVersion: v1."""
        with open(SA_TEMPLATE_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "apiVersion: v1" in content

    def test_serviceaccount_template_has_kind(self):
        """serviceaccount.yaml must declare kind: ServiceAccount."""
        with open(SA_TEMPLATE_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "kind: ServiceAccount" in content

    def test_serviceaccount_template_uses_fullname(self):
        """serviceaccount.yaml must use ocr-local.fullname for the name."""
        with open(SA_TEMPLATE_PATH, encoding="utf-8") as f:
            content = f.read()
        assert 'include "ocr-local.fullname"' in content

    def test_serviceaccount_template_has_namespace(self):
        """serviceaccount.yaml must set namespace from Release."""
        with open(SA_TEMPLATE_PATH, encoding="utf-8") as f:
            content = f.read()
        assert ".Release.Namespace" in content

    def test_serviceaccount_template_has_labels(self):
        """serviceaccount.yaml must include standard labels."""
        with open(SA_TEMPLATE_PATH, encoding="utf-8") as f:
            content = f.read()
        assert 'include "ocr-local.labels"' in content


class TestHelpersServiceAccountName:
    """Tests for the serviceAccountName helper in _helpers.tpl."""

    def test_helper_defines_service_account_name(self):
        """_helpers.tpl must define ocr-local.serviceAccountName."""
        with open(HELPERS_PATH, encoding="utf-8") as f:
            content = f.read()
        assert 'define "ocr-local.serviceAccountName"' in content

    def test_helper_uses_rbac_create(self):
        """serviceAccountName helper must check .Values.rbac.create."""
        with open(HELPERS_PATH, encoding="utf-8") as f:
            content = f.read()
        assert ".Values.rbac.create" in content

    def test_helper_falls_back_to_default(self):
        """serviceAccountName helper must fall back to 'default' when rbac.create is false."""
        with open(HELPERS_PATH, encoding="utf-8") as f:
            content = f.read()
        # The helper should output "default" in the else branch
        assert "default" in content


class TestDeploymentServiceAccountName:
    """Tests that all deployment templates reference serviceAccountName."""

    @pytest.mark.parametrize("template_file", DEPLOYMENT_TEMPLATES)
    def test_deployment_has_service_account_name(self, template_file):
        """Each deployment template must reference serviceAccountName."""
        template_path = os.path.join(TEMPLATES_DIR, template_file)
        assert os.path.isfile(template_path), f"Template missing: {template_file}"

        with open(template_path, encoding="utf-8") as f:
            content = f.read()

        assert "serviceAccountName:" in content, (
            f"{template_file} missing serviceAccountName reference"
        )

    @pytest.mark.parametrize("template_file", DEPLOYMENT_TEMPLATES)
    def test_deployment_uses_helper(self, template_file):
        """Each deployment must use the ocr-local.serviceAccountName helper."""
        template_path = os.path.join(TEMPLATES_DIR, template_file)
        with open(template_path, encoding="utf-8") as f:
            content = f.read()

        assert 'include "ocr-local.serviceAccountName"' in content, (
            f"{template_file} must use the ocr-local.serviceAccountName helper"
        )
