"""Tests for the Plan C Phase 1, item C2 federation operator overlay.

Validates the C2 expanded federation control plane:
* Cluster registry ConfigMap renders one entry per `federation.clusters[]`.
* Reconciler Deployment, Service, and RBAC manifests are emitted when
  enabled.
* Per-(peer x queue-class) Policy CRs are produced.
* All federation resources disappear when `federation.enabled=false`.

Like the C1 test module these tests shell out to the local `helm` CLI and
skip when it is not on PATH.
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

# Minimal secrets so `required` calls in the chart don't abort the render.
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


# ---------------------------------------------------------------------------
# Default-disabled state
# ---------------------------------------------------------------------------
def test_disabled_emits_no_operator_resources() -> None:
    rendered = _helm_template()
    docs = _parse_docs(rendered)
    suspects = [
        d for d in docs
        if (d.get("metadata", {}) or {}).get("name", "").startswith(
            "ocr-local-federation-"
        )
    ]
    assert suspects == [], (
        f"federation operator resources rendered while disabled: "
        f"{[s['metadata']['name'] for s in suspects]}"
    )


# ---------------------------------------------------------------------------
# Enabled state
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def enabled_docs() -> list[dict]:
    return _parse_docs(_helm_template(values_files=[EXAMPLE_VALUES]))


def test_cluster_registry_configmap_rendered(enabled_docs: list[dict]) -> None:
    cms = [
        d for d in enabled_docs
        if d.get("kind") == "ConfigMap"
        and d.get("metadata", {}).get("name", "")
        == "ocr-local-federation-cluster-registry"
    ]
    assert len(cms) == 1
    cm = cms[0]
    assert cm["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "Helm"
    payload = cm["data"]["clusters.json"]
    # Should be valid JSON with the expected schema.
    import json

    data = json.loads(payload)
    assert data["self"]["name"] == "prod-us-east-1"
    assert "ocr_gpu" in data["queue_classes"]
    assert "translation_batch" in data["queue_classes"]
    cluster_names = {c["name"] for c in data["clusters"]}
    assert cluster_names == {"prod-us-west-1", "prod-eu-central-1", "airgap-gov-1"}
    eu = next(c for c in data["clusters"] if c["name"] == "prod-eu-central-1")
    assert eu["tls"]["enabled"] is True
    assert eu["tls"]["ca_secret_ref"] == "rabbitmq-fed-eu-ca"
    assert eu["priority"] == 10


def test_reconciler_deployment_rendered(enabled_docs: list[dict]) -> None:
    deploys = [
        d for d in enabled_docs
        if d.get("kind") == "Deployment"
        and d.get("metadata", {}).get("name", "")
        == "ocr-local-federation-reconciler"
    ]
    assert len(deploys) == 1
    deploy = deploys[0]
    spec = deploy["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert container["name"] == "reconciler"
    assert container["command"][:3] == ["python", "-m", "federation.reconciler"]
    env = {e["name"]: e for e in container.get("env", [])}
    assert env["FEDERATION_REGISTRY_PATH"]["value"] == "/etc/federation/clusters.json"
    assert env["FEDERATION_METRICS_PORT"]["value"] == "9100"
    # Volume + mount for the registry ConfigMap.
    volumes = {v["name"]: v for v in spec.get("volumes", [])}
    assert "registry" in volumes
    assert (
        volumes["registry"]["configMap"]["name"]
        == "ocr-local-federation-cluster-registry"
    )
    # Pod-level non-root posture.
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert deploy["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "Helm"


def test_reconciler_service_rendered(enabled_docs: list[dict]) -> None:
    svcs = [
        d for d in enabled_docs
        if d.get("kind") == "Service"
        and d.get("metadata", {}).get("name", "")
        == "ocr-local-federation-reconciler"
    ]
    assert len(svcs) == 1
    ports = svcs[0]["spec"]["ports"]
    assert any(p.get("port") == 9100 for p in ports)


def test_reconciler_rbac_rendered(enabled_docs: list[dict]) -> None:
    sa = [
        d for d in enabled_docs
        if d.get("kind") == "ServiceAccount"
        and d.get("metadata", {}).get("name", "")
        == "ocr-local-federation-reconciler"
    ]
    role = [
        d for d in enabled_docs
        if d.get("kind") == "Role"
        and d.get("metadata", {}).get("name", "")
        == "ocr-local-federation-reconciler"
    ]
    rb = [
        d for d in enabled_docs
        if d.get("kind") == "RoleBinding"
        and d.get("metadata", {}).get("name", "")
        == "ocr-local-federation-reconciler"
    ]
    assert len(sa) == 1
    assert len(role) == 1
    assert len(rb) == 1
    role_rules = role[0]["rules"]
    # ConfigMap and Secret read-only access plus rabbitmq.com policy verbs.
    rabbit_rule = next(r for r in role_rules if "rabbitmq.com" in r["apiGroups"])
    assert "patch" in rabbit_rule["verbs"]
    assert "delete" in rabbit_rule["verbs"]
    cm_rule = next(r for r in role_rules if "configmaps" in r.get("resources", []))
    # Reads only -- no write verbs.
    assert set(cm_rule["verbs"]) <= {"get", "list", "watch"}


def test_per_cluster_per_queue_policies_rendered(enabled_docs: list[dict]) -> None:
    policies = [
        d for d in enabled_docs
        if d.get("kind") == "Policy"
        and d.get("apiVersion", "").startswith("rabbitmq.com/")
    ]
    # 3 clusters x 4 queue classes = 12 policies.
    assert len(policies) == 12
    # Patterns reference the bare queue class.
    patterns = {p["spec"]["pattern"] for p in policies}
    assert "^ocr_gpu.*$" in patterns
    assert "^ocr_cpu.*$" in patterns
    assert "^nlp.*$" in patterns
    assert "^translation_batch.*$" in patterns
    # Each policy targets a single peer upstream.
    upstreams = {p["spec"]["definition"]["federation-upstream"] for p in policies}
    assert upstreams == {
        "prod-us-west-1-upstream",
        "prod-eu-central-1-upstream",
        "airgap-gov-1-upstream",
    }
    for p in policies:
        assert p["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "Helm"


def test_federation_crs_match_dynamic_clusters(enabled_docs: list[dict]) -> None:
    feds = [
        d for d in enabled_docs
        if d.get("kind") == "Federation"
        and d.get("apiVersion", "").startswith("rabbitmq.com/")
    ]
    names = {f["metadata"]["name"] for f in feds}
    assert names == {
        "ocr-local-fed-upstream-prod-us-west-1",
        "ocr-local-fed-upstream-prod-eu-central-1",
        "ocr-local-fed-upstream-airgap-gov-1",
    }


def test_reconciler_disabled_when_subflag_off() -> None:
    rendered = _helm_template(
        "--set",
        "federation.enabled=true",
        "--set",
        "federation.reconciler.enabled=false",
        "--set",
        "federation.clusterName=prod-self",
    )
    docs = _parse_docs(rendered)
    deploy_names = [
        d.get("metadata", {}).get("name", "")
        for d in docs
        if d.get("kind") == "Deployment"
    ]
    assert "ocr-local-federation-reconciler" not in deploy_names
    svc_names = [
        d.get("metadata", {}).get("name", "")
        for d in docs
        if d.get("kind") == "Service"
    ]
    assert "ocr-local-federation-reconciler" not in svc_names


def test_default_values_keep_operator_off() -> None:
    """Defaults (`federation.enabled=false`) must produce zero operator manifests."""
    rendered = _helm_template()
    docs = _parse_docs(rendered)
    for d in docs:
        name = d.get("metadata", {}).get("name", "")
        assert "federation-reconciler" not in name
        assert "federation-cluster-registry" not in name
