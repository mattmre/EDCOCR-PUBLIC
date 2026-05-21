"""Tests for the Plan C Phase 1 C3 federated observability Helm overlay.

Validates that:
* Default install (`observability.federation.enabled=false`) renders zero
  observability-federation manifests.
* `mode=remote_write` with two endpoints renders the ConfigMap fragment plus
  the auth Secret stub.
* `mode=thanos_sidecar` renders the StatefulSet + Service.
* The multi-cluster Grafana dashboard ConfigMap renders with all 12 expected
  panel titles when enabled.

These tests shell out to the local `helm` CLI and are skipped if `helm` is
not on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHART_PATH = REPO_ROOT / "helm" / "ocr-local"
OBSERVABILITY_VALUES = CHART_PATH / "values-observability.yaml"

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


def test_observability_disabled_renders_zero_federation_resources():
    """Default values must produce no observability-federation resources."""
    rendered = _helm_template(values_files=[OBSERVABILITY_VALUES])
    docs = _parse_docs(rendered)

    fed_component_names = (
        "-prom-remote-write",
        "-prom-remote-write-auth",
        "-thanos-sidecar",
        "-grafana-multi-cluster-dashboard",
        "-observability-fed",
    )
    fed_resources = [
        d
        for d in docs
        if any(
            (d.get("metadata", {}).get("name", "") or "").endswith(suffix)
            for suffix in fed_component_names
        )
    ]
    assert fed_resources == [], (
        f"Unexpected observability federation resources: "
        f"{[d.get('metadata', {}).get('name') for d in fed_resources]}"
    )


def test_remote_write_mode_renders_configmap_and_secret():
    """`mode=remote_write` with 2 endpoints renders ConfigMap + Secret stub."""
    rendered = _helm_template(
        values_files=[OBSERVABILITY_VALUES],
        *[
            "--set", "observability.federation.enabled=true",
            "--set", "observability.federation.mode=remote_write",
            "--set", "observability.federation.cluster_label=test-cluster",
            "--set", "observability.federation.remote_write.endpoints[0].name=primary",
            "--set", "observability.federation.remote_write.endpoints[0].url=https://prom-a.example/api/v1/write",
            "--set", "observability.federation.remote_write.endpoints[1].name=secondary",
            "--set", "observability.federation.remote_write.endpoints[1].url=https://prom-b.example/api/v1/write",
        ],
    )
    docs = _parse_docs(rendered)

    cms = [
        d for d in docs
        if d.get("kind") == "ConfigMap"
        and d.get("metadata", {}).get("name", "").endswith("-prom-remote-write")
    ]
    assert len(cms) == 1, "expected exactly one prom-remote-write ConfigMap"
    body = cms[0]["data"]["remote-write.yaml"]
    assert "https://prom-a.example/api/v1/write" in body
    assert "https://prom-b.example/api/v1/write" in body
    assert "cluster: \"test-cluster\"" in body or "cluster: 'test-cluster'" in body

    secrets = [
        d for d in docs
        if d.get("kind") == "Secret"
        and d.get("metadata", {}).get("name", "").endswith("-prom-remote-write-auth")
    ]
    assert len(secrets) == 1, "expected the remote_write auth Secret stub"
    annotations = secrets[0]["metadata"].get("annotations", {})
    assert "kubectl edit" in annotations.get("ocr-local.io/note", "")

    sidecars = [
        d for d in docs
        if d.get("kind") == "StatefulSet"
        and d.get("metadata", {}).get("name", "").endswith("-thanos-sidecar")
    ]
    assert sidecars == [], "thanos-sidecar must NOT render in remote_write mode"


def test_thanos_sidecar_mode_renders_statefulset_and_service():
    """`mode=thanos_sidecar` renders StatefulSet + ClusterIP Service."""
    rendered = _helm_template(
        values_files=[OBSERVABILITY_VALUES],
        *[
            "--set", "observability.federation.enabled=true",
            "--set", "observability.federation.mode=thanos_sidecar",
            "--set", "observability.federation.thanos_sidecar.enabled=true",
            "--set", "observability.federation.cluster_label=ts-cluster",
        ],
    )
    docs = _parse_docs(rendered)

    sts = [
        d for d in docs
        if d.get("kind") == "StatefulSet"
        and d.get("metadata", {}).get("name", "").endswith("-thanos-sidecar")
    ]
    assert len(sts) == 1, "expected exactly one thanos-sidecar StatefulSet"
    container = sts[0]["spec"]["template"]["spec"]["containers"][0]
    assert container["name"] == "thanos-sidecar"
    assert container["image"].startswith("quay.io/thanos/thanos:")
    ports = {p["name"] for p in container["ports"]}
    assert "grpc" in ports and "http" in ports

    svcs = [
        d for d in docs
        if d.get("kind") == "Service"
        and d.get("metadata", {}).get("name", "").endswith("-thanos-sidecar")
    ]
    assert len(svcs) == 1, "expected exactly one thanos-sidecar Service"
    svc_ports = {p["name"]: p["port"] for p in svcs[0]["spec"]["ports"]}
    assert svc_ports.get("grpc") == 10901
    assert svc_ports.get("http") == 10902

    cms = [
        d for d in docs
        if d.get("kind") == "ConfigMap"
        and d.get("metadata", {}).get("name", "").endswith("-prom-remote-write")
    ]
    assert cms == [], "remote_write CM must NOT render in thanos_sidecar mode"


def test_multi_cluster_grafana_dashboard_has_all_panels():
    """Dashboard ConfigMap renders with all 12 expected panel titles."""
    rendered = _helm_template(
        values_files=[OBSERVABILITY_VALUES],
        *[
            "--set", "observability.federation.enabled=true",
            "--set", "observability.federation.mode=remote_write",
            "--set", "observability.grafana.multi_cluster_dashboard.enabled=true",
        ],
    )
    docs = _parse_docs(rendered)

    cms = [
        d for d in docs
        if d.get("kind") == "ConfigMap"
        and d.get("metadata", {}).get("name", "").endswith("-grafana-multi-cluster-dashboard")
    ]
    assert len(cms) == 1, "expected exactly one multi-cluster grafana ConfigMap"

    cm = cms[0]
    labels = cm["metadata"].get("labels", {})
    assert labels.get("grafana_dashboard") == "1", "must carry sidecar discovery label"

    raw_json = cm["data"]["ocr-multi-cluster.json"]
    dashboard = json.loads(raw_json)
    titles = [p["title"] for p in dashboard["panels"]]
    expected = [
        "Cluster Heartbeat",
        "Per-Cluster Job Throughput",
        "Per-Cluster Queue Depth",
        "Per-Cluster GPU Utilization",
        "Per-Cluster GPU VRAM",
        "Per-Cluster Worker Count",
        "Per-Cluster Failure Rate",
        "Per-Cluster Tail Latency p95",
        "Per-Cluster Tail Latency p99",
        "Per-Cluster Active Tenants",
        "Cross-Cluster Federation Lag",
        "Cross-Cluster Translation Throughput",
    ]
    assert titles == expected, f"panel order/titles drifted: {titles}"
    var_names = {v["name"] for v in dashboard["templating"]["list"]}
    assert {"cluster", "region"} <= var_names


def test_observability_values_file_exists_and_disabled_by_default():
    """The shipped overlay file must keep federation OFF by default."""
    assert OBSERVABILITY_VALUES.is_file(), f"missing {OBSERVABILITY_VALUES}"
    with open(OBSERVABILITY_VALUES, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["observability"]["federation"]["enabled"] is False
    assert data["observability"]["federation"]["mode"] == "remote_write"
    assert data["observability"]["grafana"]["multi_cluster_dashboard"]["enabled"] is False


def test_observability_networkpolicy_renders_when_enabled():
    """The opt-in NetworkPolicy renders only when both flags are set."""
    rendered = _helm_template(
        values_files=[OBSERVABILITY_VALUES],
        *[
            "--set", "observability.federation.enabled=true",
            "--set", "observability.federation.mode=thanos_sidecar",
            "--set", "observability.federation.thanos_sidecar.enabled=true",
            "--set", "observability.networkPolicy.enabled=true",
        ],
    )
    docs = _parse_docs(rendered)
    netpols = [
        d for d in docs
        if d.get("kind") == "NetworkPolicy"
        and d.get("metadata", {}).get("name", "").endswith("-observability-fed")
    ]
    assert len(netpols) == 1
    spec = netpols[0]["spec"]
    assert spec["podSelector"]["matchLabels"]["app.kubernetes.io/component"] == "thanos-sidecar"
    assert "Ingress" in spec["policyTypes"] and "Egress" in spec["policyTypes"]
