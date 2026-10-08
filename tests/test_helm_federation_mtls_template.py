"""Defensive render checks for supported TLS and unsupported federation mTLS."""

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
    "secrets.apiKey": "test-api-key",
    "secrets.postgresPassword": "test-pg",
    "secrets.webhookSecretKey": "test-webhook",
    "secrets.piiEncryptionKey": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    "secrets.celeryTaskSigningKey": "test-signing",
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
        "--set",
        "federation.enabled=false",
        "--set",
        "federation.mtls.enabled=true",
    )
    docs = _parse_docs(rendered)
    assert _federation_mtls_resources(docs) == []


def test_federation_enabled_mtls_disabled_renders_no_mtls_resources():
    rendered = _helm_template(
        "--set",
        "federation.enabled=true",
        "--set",
        "federation.mtls.enabled=false",
    )
    docs = _parse_docs(rendered)
    assert _federation_mtls_resources(docs) == []


# Certificates remain as future implementation scaffolding. Rendering them is
# insufficient to establish mutually authenticated transport: reject the toggle.
@pytest.mark.parametrize("extra", [
    [],
    ["--set", "tls.enabled=true"],
    ["--set", "federation.mtls.caRef=owned-external-issuer"],
    ["--set", "federation.mtls.clusterIssuer=true"],
])
def test_incomplete_federation_mtls_transport_is_rejected(extra):
    cmd = ["helm", "template", "ocr-local", str(CHART_PATH)]
    for key, value in MIN_SECRETS.items():
        cmd.extend(["--set", f"{key}={value}"])
    cmd.extend(["--set", "federation.enabled=true", "--set", "federation.mtls.enabled=true", *extra])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "federation.mtls.enabled is unsupported" in result.stderr


def test_ordinary_broker_tls_does_not_claim_client_certificate_identity():
    documents = _parse_docs(_helm_template("--set", "tls.enabled=true"))
    broker = _by_kind_and_name(documents, "ConfigMap", "-rabbitmq-config")
    config = broker["data"]["rabbitmq.conf"]
    assert "listeners.tcp = none" in config
    assert "listeners.ssl.default = 5671" in config
    assert "ssl_options.verify = verify_none" in config
    assert "ssl_options.fail_if_no_peer_cert = false" in config


def test_backup_job_verifies_postgres_identity_when_tls_is_enabled():
    documents = _parse_docs(_helm_template("--set", "tls.enabled=true", "--set", "postgresql.backup.enabled=true"))
    job = _by_kind_and_name(documents, "CronJob", "-postgres-backup")
    pod = job["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    env = {entry["name"]: entry.get("value") for entry in container["env"]}
    assert env["PGSSLMODE"] == "verify-full"
    assert env["PGSSLROOTCERT"] == "/etc/backup-ca/ca.crt"
    ca = next(volume for volume in pod["volumes"] if volume["name"] == "backup-ca")
    assert ca["secret"]["items"] == [{"key": "ca.crt", "path": "ca.crt"}]
    assert any(mount["name"] == "backup-ca" and mount["readOnly"] for mount in container["volumeMounts"])
