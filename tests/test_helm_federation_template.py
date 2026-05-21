"""Tests for the Plan C Phase 1 federation Helm overlay.

Validates that:
* Default install (`federation.enabled=false`) renders zero federation resources.
* With federation enabled and the example values applied, all 5 federation
  resource types (overlay ConfigMap, FederationUpstream CRs, Policy CR,
  NetworkPolicy, ApplicationSet) render.
* Coordinator and worker deployments only mount the federation-overlay
  ConfigMap as `envFrom` when federation is enabled.

These tests shell out to the local `helm` CLI. They are skipped if `helm` is
not on PATH or if the chart fails to render at all.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHART_PATH = REPO_ROOT / "helm" / "ocr-local"
EXAMPLE_VALUES = CHART_PATH / "values-federation-example.yaml"

# Minimal secret overrides so `required` calls in the chart don't abort the
# render. None of these are real credentials -- they exist only to clear the
# fail-on-missing-secret guards in templates/secret.yaml.
MIN_SECRETS = {
    "secrets.djangoSecretKey": "test-django-key",
    "secrets.postgresPassword": "test-pg",
    "secrets.rabbitmqPassword": "test-rmq",
    "secrets.redisPassword": "test-redis",
    "secrets.flowerPassword": "test-flower",
    "secrets.metricsApiKey": "test-metrics",
}


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _helm_template(*extra_args: str, values_files: list[Path] | None = None) -> str:
    cmd = ["helm", "template", "ocr-local", str(CHART_PATH)]
    for vf in values_files or []:
        cmd.extend(["--values", str(vf)])
    for k, v in MIN_SECRETS.items():
        cmd.extend(["--set", f"{k}={v}"])
    cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        pytest.fail(f"helm template failed: {proc.stderr}")
    return proc.stdout


def _parse_docs(rendered: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(rendered) if d]


pytestmark = pytest.mark.skipif(not _helm_available(), reason="helm CLI not on PATH")


def test_federation_disabled_renders_zero_federation_resources():
    """Default values (`federation.enabled=false`) must produce no federation resources."""
    rendered = _helm_template()
    docs = _parse_docs(rendered)

    federation_kinds = {"Federation", "Policy", "ApplicationSet"}
    fed_resources = [
        d for d in docs
        if d.get("kind") in federation_kinds
        and "federation" in (d.get("metadata", {}).get("name", "") or "").lower()
    ]
    # No federation-overlay ConfigMap
    overlay_cms = [
        d for d in docs
        if d.get("kind") == "ConfigMap"
        and d.get("metadata", {}).get("name", "").endswith("-federation-overlay")
    ]
    # No federation-egress NetworkPolicy
    fed_netpols = [
        d for d in docs
        if d.get("kind") == "NetworkPolicy"
        and d.get("metadata", {}).get("name", "").endswith("-federation-egress")
    ]

    assert fed_resources == [], f"Unexpected federation resources: {fed_resources}"
    assert overlay_cms == [], f"Unexpected overlay ConfigMap: {overlay_cms}"
    assert fed_netpols == [], f"Unexpected federation NetworkPolicy: {fed_netpols}"


def test_federation_enabled_renders_all_resources():
    """With `federation.enabled=true` plus example values, all 5 resource types render."""
    rendered = _helm_template(values_files=[EXAMPLE_VALUES])
    docs = _parse_docs(rendered)

    by_kind: dict[str, list[dict]] = {}
    for d in docs:
        by_kind.setdefault(d.get("kind", ""), []).append(d)

    # 1. Federation overlay ConfigMap
    overlay_cms = [
        d for d in by_kind.get("ConfigMap", [])
        if d.get("metadata", {}).get("name", "").endswith("-federation-overlay")
    ]
    assert len(overlay_cms) == 1, "expected exactly one federation-overlay ConfigMap"
    cm_data = overlay_cms[0].get("data", {})
    assert cm_data.get("OCR_CLUSTER_NAME") == "prod-us-east-1"
    assert "OCR_PRESIGNED_ENDPOINT" in cm_data
    assert cm_data.get("OCR_FEDERATION_ENABLED") == "true"

    # 2. RabbitMQ FederationUpstream CRs (one per peer cluster).
    # C2 widened the example to a 3-cluster registry; the template ranges
    # over `.federation.clusters` when populated and falls back to the legacy
    # `upstreamClusters` list otherwise. Either form must produce >= 2 CRs.
    fed_uprstreams = [
        d for d in by_kind.get("Federation", [])
        if d.get("apiVersion", "").startswith("rabbitmq.com/")
    ]
    assert len(fed_uprstreams) >= 2, (
        f"expected >= 2 Federation upstream CRs, got {len(fed_uprstreams)}"
    )

    # 3. RabbitMQ Policy CR(s). C2 emits one Policy per (cluster x queue-class)
    # combo when the dynamic registry is populated; when only the legacy
    # `upstreamClusters` list is used, exactly one aggregated policy renders.
    fed_policies = [
        d for d in by_kind.get("Policy", [])
        if d.get("apiVersion", "").startswith("rabbitmq.com/")
        and "fed-policy" in d.get("metadata", {}).get("name", "")
    ]
    assert len(fed_policies) >= 1, "expected at least one RabbitMQ federation Policy"
    # Across all rendered policies the federated queue classes must appear.
    all_patterns = " ".join(p.get("spec", {}).get("pattern", "") for p in fed_policies)
    assert "ocr_gpu" in all_patterns and "ocr_cpu" in all_patterns

    # 4. ArgoCD ApplicationSet
    appsets = by_kind.get("ApplicationSet", [])
    assert len(appsets) == 1, "expected exactly one ArgoCD ApplicationSet"

    # 5. Federation egress NetworkPolicy
    fed_netpols = [
        d for d in by_kind.get("NetworkPolicy", [])
        if d.get("metadata", {}).get("name", "").endswith("-federation-egress")
    ]
    assert len(fed_netpols) == 1, "expected exactly one federation egress NetworkPolicy"


def test_argocd_applicationset_carries_target_clusters():
    """ApplicationSet must include the targetClusters list-generator entries."""
    rendered = _helm_template(values_files=[EXAMPLE_VALUES])
    docs = _parse_docs(rendered)

    appsets = [d for d in docs if d.get("kind") == "ApplicationSet"]
    assert len(appsets) == 1
    spec = appsets[0]["spec"]
    generators = spec.get("generators", [])
    assert generators, "ApplicationSet must declare at least one generator"
    elements = generators[0].get("list", {}).get("elements", [])
    cluster_names = {e.get("cluster") for e in elements}
    # values-federation-example.yaml declares 3 clusters
    assert {"prod-us-east-1", "prod-eu-west-1", "airgap-gov-1"} <= cluster_names


def test_coordinator_gets_federation_envfrom_only_when_enabled():
    """Coordinator deployment mounts the overlay ConfigMap iff federation enabled."""
    # disabled -> no federation envFrom
    disabled_rendered = _helm_template()
    disabled_docs = _parse_docs(disabled_rendered)
    coord_disabled = next(
        d for d in disabled_docs
        if d.get("kind") == "Deployment"
        and d.get("metadata", {}).get("name", "") == "ocr-local-coordinator"
    )
    container = coord_disabled["spec"]["template"]["spec"]["containers"][0]
    env_from = container.get("envFrom", [])
    overlay_refs = [
        ef for ef in env_from
        if (ef.get("configMapRef", {}) or {}).get("name", "").endswith("-federation-overlay")
    ]
    assert overlay_refs == [], "coordinator must NOT mount federation overlay when disabled"

    # enabled -> federation envFrom present
    enabled_rendered = _helm_template(values_files=[EXAMPLE_VALUES])
    enabled_docs = _parse_docs(enabled_rendered)
    coord_enabled = next(
        d for d in enabled_docs
        if d.get("kind") == "Deployment"
        and d.get("metadata", {}).get("name", "") == "ocr-local-coordinator"
    )
    container = coord_enabled["spec"]["template"]["spec"]["containers"][0]
    env_from = container.get("envFrom", [])
    overlay_refs = [
        ef for ef in env_from
        if (ef.get("configMapRef", {}) or {}).get("name", "").endswith("-federation-overlay")
    ]
    assert len(overlay_refs) == 1, "coordinator must mount federation overlay when enabled"


@pytest.mark.parametrize(
    "deployment_suffix",
    ["-gpu-worker", "-cpu-worker"],
)
def test_workers_get_federation_envfrom_only_when_enabled(deployment_suffix):
    """GPU/CPU worker deployments mount the overlay ConfigMap iff federation enabled."""
    # disabled
    disabled_docs = _parse_docs(_helm_template())
    worker_disabled = next(
        d for d in disabled_docs
        if d.get("kind") == "Deployment"
        and d.get("metadata", {}).get("name", "").endswith(deployment_suffix)
    )
    env_from = worker_disabled["spec"]["template"]["spec"]["containers"][0].get(
        "envFrom", []
    )
    refs = [
        ef for ef in env_from
        if (ef.get("configMapRef", {}) or {}).get("name", "").endswith("-federation-overlay")
    ]
    assert refs == [], (
        f"{deployment_suffix} must NOT mount federation overlay when disabled"
    )

    # enabled
    enabled_docs = _parse_docs(_helm_template(values_files=[EXAMPLE_VALUES]))
    worker_enabled = next(
        d for d in enabled_docs
        if d.get("kind") == "Deployment"
        and d.get("metadata", {}).get("name", "").endswith(deployment_suffix)
    )
    env_from = worker_enabled["spec"]["template"]["spec"]["containers"][0].get(
        "envFrom", []
    )
    refs = [
        ef for ef in env_from
        if (ef.get("configMapRef", {}) or {}).get("name", "").endswith("-federation-overlay")
    ]
    assert len(refs) == 1, (
        f"{deployment_suffix} must mount federation overlay when enabled"
    )


def test_federation_example_values_file_exists():
    """The values-federation-example.yaml must ship with the chart."""
    assert EXAMPLE_VALUES.is_file(), f"missing {EXAMPLE_VALUES}"
    with open(EXAMPLE_VALUES, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["federation"]["enabled"] is True
    assert data["federation"]["clusterName"]
    assert len(data["federation"]["upstreamClusters"]) >= 1


def test_federation_disabled_in_default_values():
    """Chart default must keep federation OFF -- non-federated installs unchanged."""
    with open(CHART_PATH / "values.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["federation"]["enabled"] is False
    assert data["federation"]["argocd"]["enabled"] is False
