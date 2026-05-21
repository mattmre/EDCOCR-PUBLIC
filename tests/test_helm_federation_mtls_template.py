"""Tests for the Plan C Phase 1 / item C8 federation mTLS Helm overlay.

Verifies that the federation mTLS hardening templates render correctly under
the matrix of `federation.enabled` and `federation.mtls.enabled` toggles, and
that leaf-cert / NetworkPolicy / volume-mount wiring honours the values
documented in `helm/ocr-local/values.yaml`.

These tests shell out to the local ``helm`` CLI and are skipped when ``helm``
is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHART_PATH = REPO_ROOT / "helm" / "ocr-local"

# Minimal secret overrides to bypass `required` template guards.
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


def _by_kind_and_name(docs: list[dict], kind: str, name_suffix: str) -> dict | None:
    for d in docs:
        if d.get("kind") != kind:
            continue
        name = d.get("metadata", {}).get("name", "") or ""
        if name.endswith(name_suffix):
            return d
    return None


def _deployment_by_name(docs: list[dict], name: str) -> dict | None:
    for d in docs:
        if d.get("kind") != "Deployment":
            continue
        if d.get("metadata", {}).get("name", "") == name:
            return d
    return None


def _statefulset_by_name(docs: list[dict], name: str) -> dict | None:
    for d in docs:
        if d.get("kind") != "StatefulSet":
            continue
        if d.get("metadata", {}).get("name", "") == name:
            return d
    return None


def _federation_mtls_resources(docs: list[dict]) -> list[dict]:
    out = []
    for d in docs:
        name = d.get("metadata", {}).get("name", "") or ""
        if "federation-mtls" in name:
            out.append(d)
    return out


pytestmark = pytest.mark.skipif(not _helm_available(), reason="helm CLI not on PATH")


# ---------------------------------------------------------------------------
# Default install / federation off / mtls off
# ---------------------------------------------------------------------------
def test_default_install_renders_no_mtls_resources():
    """Default install (federation disabled) -> zero mTLS resources."""
    rendered = _helm_template()
    docs = _parse_docs(rendered)
    assert _federation_mtls_resources(docs) == []


def test_chart_default_mtls_block_is_disabled():
    """`values.yaml` keeps `federation.mtls.enabled` false by default."""
    with open(CHART_PATH / "values.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mtls = data["federation"].get("mtls", {})
    assert mtls.get("enabled") is False
    assert mtls.get("clusterIssuer") is False
    assert mtls.get("caRef") == ""
    assert mtls.get("certDuration") == "8760h"
    assert mtls.get("certRenewBefore") == "720h"
    assert mtls.get("peerLabel") == "ocr.local/federation-peer"


def test_federation_disabled_with_mtls_enabled_renders_nothing():
    """`mtls.enabled=true` is subordinate to `federation.enabled` -- no resources."""
    rendered = _helm_template(
        "--set", "federation.enabled=false",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    assert _federation_mtls_resources(docs) == []


def test_federation_enabled_mtls_disabled_renders_no_mtls_resources():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=false",
    )
    docs = _parse_docs(rendered)
    assert _federation_mtls_resources(docs) == []


# ---------------------------------------------------------------------------
# Self-signed bootstrap path
# ---------------------------------------------------------------------------
def test_self_signed_bootstrap_renders_full_chain():
    """`mtls.enabled=true` with no `caRef` -> bootstrap Issuer + CA cert + 3 leaves + NetworkPolicy."""
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)

    # Two Issuers (bootstrap + CA-backed).
    issuers = [
        d for d in docs
        if d.get("kind") == "Issuer"
        and "federation-mtls" in d.get("metadata", {}).get("name", "")
    ]
    assert len(issuers) == 2
    issuer_names = {d["metadata"]["name"] for d in issuers}
    assert "ocr-local-federation-mtls-bootstrap" in issuer_names
    assert "ocr-local-federation-mtls-ca" in issuer_names

    # No ClusterIssuer (default off).
    cluster_issuers = [
        d for d in docs if d.get("kind") == "ClusterIssuer"
    ]
    assert cluster_issuers == []

    # Four Certificates: CA, coordinator, rabbitmq, client.
    certs = [
        d for d in docs
        if d.get("kind") == "Certificate"
        and "federation-mtls" in d.get("metadata", {}).get("name", "")
    ]
    assert len(certs) == 4
    cert_names = {d["metadata"]["name"] for d in certs}
    assert {
        "ocr-local-federation-mtls-ca",
        "ocr-local-federation-mtls-coordinator",
        "ocr-local-federation-mtls-rabbitmq",
        "ocr-local-federation-mtls-client",
    } <= cert_names

    # NetworkPolicy.
    netpol = _by_kind_and_name(docs, "NetworkPolicy", "-federation-mtls-ingress")
    assert netpol is not None


def test_ca_is_marked_isCA_true():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    ca_cert = _by_kind_and_name(docs, "Certificate", "-federation-mtls-ca")
    assert ca_cert is not None
    assert ca_cert["spec"].get("isCA") is True


def test_leaf_certs_reference_chart_managed_ca_issuer():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    for suffix in ("coordinator", "rabbitmq", "client"):
        cert = _by_kind_and_name(docs, "Certificate", f"-federation-mtls-{suffix}")
        assert cert is not None
        issuer_ref = cert["spec"]["issuerRef"]
        assert issuer_ref["name"] == "ocr-local-federation-mtls-ca"
        assert issuer_ref["kind"] == "Issuer"
        assert issuer_ref["group"] == "cert-manager.io"


# ---------------------------------------------------------------------------
# External CA path
# ---------------------------------------------------------------------------
def test_caRef_skips_bootstrap_issuer_and_ca_cert():
    """Setting `caRef` -> no bootstrap Issuer, no CA Certificate, leaves use external Issuer."""
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
        "--set", "federation.mtls.caRef=external-fed-ca",
    )
    docs = _parse_docs(rendered)

    # Bootstrap Issuer must NOT render.
    issuers = [
        d for d in docs
        if d.get("kind") == "Issuer"
        and "federation-mtls" in d.get("metadata", {}).get("name", "")
    ]
    assert issuers == []

    # CA Certificate must NOT render.
    ca_cert = _by_kind_and_name(docs, "Certificate", "-federation-mtls-ca")
    assert ca_cert is None

    # Leaf certs MUST reference the external CA.
    for suffix in ("coordinator", "rabbitmq", "client"):
        cert = _by_kind_and_name(docs, "Certificate", f"-federation-mtls-{suffix}")
        assert cert is not None
        assert cert["spec"]["issuerRef"]["name"] == "external-fed-ca"


# ---------------------------------------------------------------------------
# ClusterIssuer kind switch
# ---------------------------------------------------------------------------
def test_clusterIssuer_renders_clusterIssuer_kind():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
        "--set", "federation.mtls.clusterIssuer=true",
    )
    docs = _parse_docs(rendered)

    issuers = [
        d for d in docs
        if d.get("kind") == "Issuer"
        and "federation-mtls" in d.get("metadata", {}).get("name", "")
    ]
    assert issuers == []  # no namespaced Issuers

    cluster_issuers = [
        d for d in docs
        if d.get("kind") == "ClusterIssuer"
        and "federation-mtls" in d.get("metadata", {}).get("name", "")
    ]
    assert len(cluster_issuers) == 2  # bootstrap + ca

    # Leaf certs reference ClusterIssuer kind.
    for suffix in ("coordinator", "rabbitmq", "client"):
        cert = _by_kind_and_name(docs, "Certificate", f"-federation-mtls-{suffix}")
        assert cert is not None
        assert cert["spec"]["issuerRef"]["kind"] == "ClusterIssuer"


# ---------------------------------------------------------------------------
# SAN inclusion
# ---------------------------------------------------------------------------
def test_coordinatorSan_appears_in_coordinator_cert():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
        "--set", "federation.mtls.coordinatorSan=fed.coordinator.example.com",
    )
    docs = _parse_docs(rendered)
    cert = _by_kind_and_name(docs, "Certificate", "-federation-mtls-coordinator")
    assert cert is not None
    assert "fed.coordinator.example.com" in cert["spec"]["dnsNames"]


def test_rabbitmqSan_appears_in_rabbitmq_cert():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
        "--set", "federation.mtls.rabbitmqSan=fed.rabbit.example.com",
    )
    docs = _parse_docs(rendered)
    cert = _by_kind_and_name(docs, "Certificate", "-federation-mtls-rabbitmq")
    assert cert is not None
    assert "fed.rabbit.example.com" in cert["spec"]["dnsNames"]


def test_coordinator_cert_includes_in_cluster_dns_san():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    cert = _by_kind_and_name(docs, "Certificate", "-federation-mtls-coordinator")
    assert cert is not None
    dns_names = cert["spec"]["dnsNames"]
    assert "coordinator.default.svc.cluster.local" in dns_names


# ---------------------------------------------------------------------------
# NetworkPolicy
# ---------------------------------------------------------------------------
def test_networkpolicy_peer_label_in_pod_selector():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
        "--set", "federation.mtls.peerLabel=custom.io/peer",
    )
    docs = _parse_docs(rendered)
    netpol = _by_kind_and_name(docs, "NetworkPolicy", "-federation-mtls-ingress")
    assert netpol is not None
    rendered_yaml = yaml.safe_dump(netpol)
    assert "custom.io/peer" in rendered_yaml


def test_networkpolicy_protects_federation_ports():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    netpol = _by_kind_and_name(docs, "NetworkPolicy", "-federation-mtls-ingress")
    assert netpol is not None
    ports = []
    for rule in netpol["spec"]["ingress"]:
        for p in rule.get("ports", []):
            ports.append(p["port"])
    assert 4369 in ports
    assert 5671 in ports
    assert 25672 in ports
    assert 8443 in ports  # default custodyPort


def test_networkpolicy_custodyPort_override():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
        "--set", "federation.mtls.custodyPort=9999",
    )
    docs = _parse_docs(rendered)
    netpol = _by_kind_and_name(docs, "NetworkPolicy", "-federation-mtls-ingress")
    assert netpol is not None
    ports = []
    for rule in netpol["spec"]["ingress"]:
        for p in rule.get("ports", []):
            ports.append(p["port"])
    assert 9999 in ports
    assert 8443 not in ports  # default replaced


# ---------------------------------------------------------------------------
# Cert duration / renewBefore propagation
# ---------------------------------------------------------------------------
def test_cert_duration_and_renewBefore_propagate():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
        "--set", "federation.mtls.certDuration=4380h",
        "--set", "federation.mtls.certRenewBefore=240h",
    )
    docs = _parse_docs(rendered)
    for suffix in ("coordinator", "rabbitmq", "client"):
        cert = _by_kind_and_name(docs, "Certificate", f"-federation-mtls-{suffix}")
        assert cert is not None
        assert cert["spec"]["duration"] == "4380h"
        assert cert["spec"]["renewBefore"] == "240h"


# ---------------------------------------------------------------------------
# Coordinator deployment volume mount
# ---------------------------------------------------------------------------
def test_coordinator_deployment_mounts_client_secret_when_mtls_enabled():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    deploy = _deployment_by_name(docs, "ocr-local-coordinator")
    assert deploy is not None
    spec = deploy["spec"]["template"]["spec"]
    container = spec["containers"][0]
    mount_names = [m["name"] for m in container.get("volumeMounts", [])]
    assert "federation-mtls-client" in mount_names

    volume_names = [v["name"] for v in spec.get("volumes", [])]
    assert "federation-mtls-client" in volume_names

    # The volume secret must reference the leaf client cert Secret.
    fed_volume = next(v for v in spec["volumes"] if v["name"] == "federation-mtls-client")
    assert fed_volume["secret"]["secretName"] == "ocr-local-federation-mtls-client-tls"


def test_coordinator_deployment_no_mtls_mount_when_mtls_disabled():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=false",
    )
    docs = _parse_docs(rendered)
    deploy = _deployment_by_name(docs, "ocr-local-coordinator")
    assert deploy is not None
    spec = deploy["spec"]["template"]["spec"]
    container = spec["containers"][0]
    mount_names = [m["name"] for m in container.get("volumeMounts", [])]
    assert "federation-mtls-client" not in mount_names


# ---------------------------------------------------------------------------
# RabbitMQ statefulset
# ---------------------------------------------------------------------------
def test_rabbitmq_statefulset_amqps_listener_config_when_mtls_enabled():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    sts = _statefulset_by_name(docs, "ocr-local-rabbitmq")
    assert sts is not None
    container = sts["spec"]["template"]["spec"]["containers"][0]
    env_names = {e["name"] for e in container.get("env", []) if isinstance(e, dict)}
    assert "RABBITMQ_FEDERATION_MTLS_ENABLED" in env_names
    assert "RABBITMQ_FEDERATION_SSL_CERTFILE" in env_names
    assert "RABBITMQ_FEDERATION_SSL_KEYFILE" in env_names
    assert "RABBITMQ_FEDERATION_SSL_CACERTFILE" in env_names

    # Volume mount for federation cert.
    mount_names = [m["name"] for m in container["volumeMounts"]]
    assert "federation-mtls-certs" in mount_names

    # Volume.
    volume_names = [v["name"] for v in sts["spec"]["template"]["spec"]["volumes"]]
    assert "federation-mtls-certs" in volume_names


def test_rabbitmq_statefulset_no_mtls_env_when_mtls_disabled():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=false",
    )
    docs = _parse_docs(rendered)
    sts = _statefulset_by_name(docs, "ocr-local-rabbitmq")
    assert sts is not None
    container = sts["spec"]["template"]["spec"]["containers"][0]
    env_names = {e["name"] for e in container.get("env", []) if isinstance(e, dict)}
    assert "RABBITMQ_FEDERATION_MTLS_ENABLED" not in env_names

    mount_names = [m["name"] for m in container["volumeMounts"]]
    assert "federation-mtls-certs" not in mount_names


# ---------------------------------------------------------------------------
# Federation overlay ConfigMap env vars
# ---------------------------------------------------------------------------
def test_overlay_configmap_exposes_mtls_env_when_enabled():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    cm = _by_kind_and_name(docs, "ConfigMap", "-federation-overlay")
    assert cm is not None
    data = cm.get("data", {})
    assert data.get("OCR_FEDERATION_MTLS_ENABLED") == "true"
    assert data.get("OCR_FEDERATION_MTLS_CERT_PATH") == "/etc/federation-mtls/tls.crt"
    assert data.get("OCR_FEDERATION_MTLS_KEY_PATH") == "/etc/federation-mtls/tls.key"
    assert data.get("OCR_FEDERATION_MTLS_CA_PATH") == "/etc/federation-mtls/ca.crt"


def test_overlay_configmap_omits_mtls_env_when_disabled():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=false",
    )
    docs = _parse_docs(rendered)
    cm = _by_kind_and_name(docs, "ConfigMap", "-federation-overlay")
    assert cm is not None
    data = cm.get("data", {})
    assert "OCR_FEDERATION_MTLS_ENABLED" not in data
    assert "OCR_FEDERATION_MTLS_CERT_PATH" not in data


# ---------------------------------------------------------------------------
# Cert privateKey hardening
# ---------------------------------------------------------------------------
def test_all_certs_use_ecdsa_p256():
    rendered = _helm_template(
        "--set", "federation.enabled=true",
        "--set", "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    fed_certs = [
        d for d in docs
        if d.get("kind") == "Certificate"
        and "federation-mtls" in d.get("metadata", {}).get("name", "")
    ]
    assert len(fed_certs) >= 3
    for cert in fed_certs:
        pk = cert["spec"].get("privateKey", {})
        assert pk.get("algorithm") == "ECDSA"
        assert pk.get("size") == 256
