"""Tests for the Plan C Phase 1 / item C7 federation discovery Helm overlay.

Verifies that:
* Default install renders neither the CronJob nor the discovery RBAC.
* Federation enabled but discovery disabled still renders nothing.
* Discovery enabled emits CronJob, ServiceAccount, Role, RoleBinding with
  the expected env vars, schedule, RBAC scope, and image inheritance.

These tests shell out to the local ``helm`` CLI. They are skipped when
``helm`` is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHART_PATH = REPO_ROOT / "helm" / "ocr-local"

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


def _helm_template(*extra_args: str) -> str:
    cmd = ["helm", "template", "ocr-local", str(CHART_PATH)]
    for k, v in MIN_SECRETS.items():
        cmd.extend(["--set", f"{k}={v}"])
    cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        pytest.fail(f"helm template failed: {proc.stderr}")
    return proc.stdout


def _parse_docs(rendered: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(rendered) if d]


def _resource(docs: list[dict], kind: str, suffix: str) -> dict | None:
    for d in docs:
        if d.get("kind") != kind:
            continue
        name = d.get("metadata", {}).get("name", "") or ""
        if name.endswith(suffix):
            return d
    return None


pytestmark = pytest.mark.skipif(
    not _helm_available(), reason="helm CLI not on PATH"
)


# ---------------------------------------------------------------------------
# Default install -- nothing rendered
# ---------------------------------------------------------------------------
def test_default_install_renders_no_discovery_cronjob():
    """``federation.enabled=false`` (default) -> no discovery CronJob."""
    rendered = _helm_template()
    docs = _parse_docs(rendered)
    assert _resource(docs, "CronJob", "-federation-discovery") is None
    assert _resource(docs, "ServiceAccount", "-federation-discovery") is None
    assert _resource(docs, "Role", "-federation-discovery") is None
    assert _resource(docs, "RoleBinding", "-federation-discovery") is None


def test_chart_default_discovery_block_is_disabled():
    """``values.yaml`` must keep ``federation.discovery.enabled`` false."""
    with open(CHART_PATH / "values.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    discovery = data["federation"].get("discovery", {})
    assert discovery.get("enabled") is False
    assert discovery.get("dryRun") is False
    assert discovery.get("serviceType") == "_ocr-federation._tcp"


# ---------------------------------------------------------------------------
# Federation enabled but discovery disabled
# ---------------------------------------------------------------------------
def test_federation_enabled_discovery_disabled_renders_no_cronjob():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.clusterName=test-local",
    )
    docs = _parse_docs(rendered)
    assert _resource(docs, "CronJob", "-federation-discovery") is None
    assert _resource(docs, "ServiceAccount", "-federation-discovery") is None


# ---------------------------------------------------------------------------
# Discovery enabled
# ---------------------------------------------------------------------------
def _enabled_template_args() -> list[str]:
    return [
        "--set", "federation.enabled=true",
        "--set", "federation.clusterName=test-local",
        "--set", "federation.discovery.enabled=true",
        "--set", "federation.discovery.domain=clusters.example.com",
    ]


def test_discovery_enabled_renders_cronjob():
    rendered = _helm_template(*_enabled_template_args())
    docs = _parse_docs(rendered)
    cj = _resource(docs, "CronJob", "-federation-discovery")
    assert cj is not None, "CronJob must be rendered when discovery enabled"
    assert cj["spec"]["schedule"] == "*/5 * * * *"
    assert cj["spec"]["concurrencyPolicy"] == "Forbid"


def test_cronjob_command_invokes_discovery_module():
    rendered = _helm_template(*_enabled_template_args())
    docs = _parse_docs(rendered)
    cj = _resource(docs, "CronJob", "-federation-discovery")
    container = cj["spec"]["jobTemplate"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert container["command"] == [
        "python",
        "-m",
        "federation.discovery",
    ]


def test_cronjob_env_carries_required_vars():
    rendered = _helm_template(*_enabled_template_args())
    docs = _parse_docs(rendered)
    cj = _resource(docs, "CronJob", "-federation-discovery")
    container = cj["spec"]["jobTemplate"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    env = {e["name"]: e for e in container.get("env", [])}
    assert env["OCR_FEDERATION_DISCOVERY_ENABLED"]["value"] == "true"
    assert (
        env["OCR_FEDERATION_DISCOVERY_DOMAIN"]["value"]
        == "clusters.example.com"
    )
    assert (
        env["OCR_FEDERATION_DISCOVERY_SERVICE_TYPE"]["value"]
        == "_ocr-federation._tcp"
    )
    # Namespace is always sourced from the downward API.
    assert env["OCR_FEDERATION_DISCOVERY_NAMESPACE"]["valueFrom"][
        "fieldRef"
    ]["fieldPath"] == "metadata.namespace"
    assert (
        env["OCR_FEDERATION_DISCOVERY_CONFIGMAP"]["value"]
        == "federation-clusters"
    )
    assert env["OCR_FEDERATION_DISCOVERY_DRY_RUN"]["value"] == "false"


def test_cronjob_dry_run_overrides_propagate():
    rendered = _helm_template(
        *_enabled_template_args(),
        "--set", "federation.discovery.dryRun=true",
        "--set", "federation.discovery.schedule=0 * * * *",
        "--set",
        "federation.discovery.configmapName=custom-fed-clusters",
    )
    docs = _parse_docs(rendered)
    cj = _resource(docs, "CronJob", "-federation-discovery")
    assert cj["spec"]["schedule"] == "0 * * * *"
    container = cj["spec"]["jobTemplate"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    env = {e["name"]: e for e in container.get("env", [])}
    assert env["OCR_FEDERATION_DISCOVERY_DRY_RUN"]["value"] == "true"
    assert (
        env["OCR_FEDERATION_DISCOVERY_CONFIGMAP"]["value"]
        == "custom-fed-clusters"
    )


def test_rbac_role_grants_scoped_to_named_configmap():
    rendered = _helm_template(*_enabled_template_args())
    docs = _parse_docs(rendered)
    role = _resource(docs, "Role", "-federation-discovery")
    assert role is not None
    rules = role.get("rules", [])
    assert len(rules) == 1
    rule = rules[0]
    assert rule["apiGroups"] == [""]
    assert rule["resources"] == ["configmaps"]
    assert sorted(rule["verbs"]) == ["get", "patch", "update"]
    # The Role binds the verbs to a single, named ConfigMap to enforce the
    # least-privilege requirement spelled out in the C7 charter.
    assert rule["resourceNames"] == ["federation-clusters"]


def test_rbac_role_does_not_grant_list_or_watch():
    rendered = _helm_template(*_enabled_template_args())
    docs = _parse_docs(rendered)
    role = _resource(docs, "Role", "-federation-discovery")
    rules = role.get("rules", [])
    for rule in rules:
        verbs = rule.get("verbs", [])
        assert "list" not in verbs
        assert "watch" not in verbs
        assert "create" not in verbs
        assert "delete" not in verbs


def test_rolebinding_targets_discovery_serviceaccount():
    rendered = _helm_template(*_enabled_template_args())
    docs = _parse_docs(rendered)
    rb = _resource(docs, "RoleBinding", "-federation-discovery")
    assert rb is not None
    subjects = rb.get("subjects", [])
    assert len(subjects) == 1
    assert subjects[0]["kind"] == "ServiceAccount"
    assert subjects[0]["name"].endswith("-federation-discovery")
    assert rb["roleRef"]["kind"] == "Role"
    assert rb["roleRef"]["name"].endswith("-federation-discovery")


def test_cronjob_runs_as_non_root_with_dropped_caps():
    rendered = _helm_template(*_enabled_template_args())
    docs = _parse_docs(rendered)
    cj = _resource(docs, "CronJob", "-federation-discovery")
    pod_spec = cj["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    container = pod_spec["containers"][0]
    sec = container["securityContext"]
    assert sec["allowPrivilegeEscalation"] is False
    assert sec["readOnlyRootFilesystem"] is True
    assert sec["capabilities"]["drop"] == ["ALL"]


def test_cronjob_uses_dedicated_service_account():
    rendered = _helm_template(*_enabled_template_args())
    docs = _parse_docs(rendered)
    cj = _resource(docs, "CronJob", "-federation-discovery")
    pod_spec = cj["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    sa = pod_spec.get("serviceAccountName", "")
    assert sa.endswith("-federation-discovery")
    # And the matching ServiceAccount resource exists.
    sa_doc = _resource(docs, "ServiceAccount", "-federation-discovery")
    assert sa_doc is not None
    assert sa_doc["metadata"]["name"] == sa
