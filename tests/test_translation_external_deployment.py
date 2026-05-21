"""Deployment-surface tests for the external EDC_TRANSLATION seam."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CHART = ROOT / "helm" / "ocr-local"

_REQUIRED_HELM_SETS = [
    "--set",
    "secrets.djangoSecretKey=test",
    "--set",
    "secrets.postgresPassword=test",
    "--set",
    "secrets.rabbitmqPassword=test",
    "--set",
    "secrets.redisPassword=test",
    "--set",
    "secrets.flowerPassword=test-flower-password",
]


def test_helm_external_translation_defaults_off():
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    production = yaml.safe_load(
        (CHART / "values-production.yaml").read_text(encoding="utf-8")
    )

    assert values["edcTranslation"]["preferExternal"] == "false"
    assert values["edcTranslation"]["url"] == ""
    assert values["edcTranslation"]["readinessPath"] == "/health"
    assert production["edcTranslation"]["preferExternal"] == "false"
    assert production["edcTranslation"]["readinessPath"] == "/health"


def test_staging_values_define_external_smoke_target_but_keep_preference_off():
    staging = yaml.safe_load(
        (CHART / "values-staging.yaml").read_text(encoding="utf-8")
    )

    assert staging["edcTranslation"]["preferExternal"] == "false"
    assert staging["edcTranslation"]["url"] == "http://edc-translation:8080"
    assert staging["edcTranslation"]["providerId"] == "deterministic_ci"
    assert staging["edcTranslation"]["readinessPath"] == "/health"


def test_compose_wires_optional_edc_translation_profile():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    coordinator = yaml.safe_load(
        (ROOT / "coordinator" / "docker-compose.coordinator.yml").read_text(
            encoding="utf-8"
        )
    )

    assert compose["services"]["edc-translation"]["profiles"] == ["edc-translation"]
    assert (
        coordinator["services"]["edc-translation"]["profiles"]
        == ["edc-translation"]
    )
    assert (
        coordinator["services"]["edc-translation"]["build"]["context"]
        == "${EDC_TRANSLATION_REPO:-../../EDC_TRANSLATION}"
    )
    assert (
        coordinator["services"]["django"]["environment"][
            "EDC_TRANSLATION_PREFER_EXTERNAL"
        ]
        == "${EDC_TRANSLATION_PREFER_EXTERNAL:-false}"
    )
    assert (
        "EDC_TRANSLATION_URL=${EDC_TRANSLATION_URL:-http://edc-translation:8080}"
        in compose["services"]["ocr-gpu"]["environment"]
    )
    assert (
        "EDC_TRANSLATION_READINESS_PATH=${EDC_TRANSLATION_READINESS_PATH:-/health}"
        in compose["services"]["ocr-gpu"]["environment"]
    )
    assert (
        coordinator["services"]["django"]["environment"][
            "EDC_TRANSLATION_READINESS_PATH"
        ]
        == "${EDC_TRANSLATION_READINESS_PATH:-/health}"
    )


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm CLI not available")
def test_helm_template_renders_external_translation_configmap_values():
    docs = _helm_template(CHART / "values-staging.yaml")
    configmap = next(
        doc
        for doc in docs
        if doc.get("kind") == "ConfigMap"
        and doc.get("metadata", {}).get("name") == "ocr-local-config"
    )

    assert configmap["data"]["EDC_TRANSLATION_PREFER_EXTERNAL"] == "false"
    assert configmap["data"]["EDC_TRANSLATION_URL"] == "http://edc-translation:8080"
    assert configmap["data"]["EDC_TRANSLATION_PROVIDER_ID"] == "deterministic_ci"
    assert configmap["data"]["EDC_TRANSLATION_TIMEOUT_SECONDS"] == "30"
    assert configmap["data"]["EDC_TRANSLATION_READINESS_PATH"] == "/health"


def _helm_template(values_path: Path) -> list[dict]:
    result = subprocess.run(
        [
            "helm",
            "template",
            "ocr-local",
            str(CHART),
            "--values",
            str(values_path),
            *_REQUIRED_HELM_SETS,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"helm template failed: {result.stderr}")
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]
