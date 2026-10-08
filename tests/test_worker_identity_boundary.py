"""AEP R3-RC-06 + S2-02 Helm posture tests.

1. Incomplete federation mTLS transport must be refused.
2. GPU/CPU workers must not envFrom the full release Secret by default
   (leastSecretEnv); S3_SECRET_KEY absent unless workerInjectS3Credentials.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHART_PATH = REPO_ROOT / "helm" / "ocr-local"

MIN_SECRETS = {
    "secrets.djangoSecretKey": "test-django-key",
    "secrets.apiKey": "test-api-key",
    "secrets.postgresPassword": "test-pg",
    "secrets.rabbitmqPassword": "test-rmq",
    "secrets.redisPassword": "test-redis",
    "secrets.flowerPassword": "test-flower",
    "secrets.metricsApiKey": "test-metrics",
    "secrets.webhookSecretKey": "test-webhook",
    "secrets.piiEncryptionKey": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    "secrets.celeryTaskSigningKey": "test-signing",
}

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None, reason="helm CLI not on PATH"
)


def _helm_template(*extra: str) -> str:
    cmd = ["helm", "template", "ocr-local", str(CHART_PATH)]
    for k, v in MIN_SECRETS.items():
        cmd.extend(["--set", f"{k}={v}"])
    cmd.extend(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        pytest.fail(f"helm template failed: {proc.stderr}")
    return proc.stdout


def _docs(rendered: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(rendered) if d]


def test_incomplete_federation_mtls_overlay_is_refused():
    """A certificate/NetworkPolicy overlay cannot qualify mutual TLS."""
    with pytest.raises(pytest.fail.Exception, match="federation.mtls.enabled is unsupported"):
        _helm_template(
            "--set", "federation.enabled=true",
            "--set", "federation.mtls.enabled=true",
        )


def test_gpu_worker_default_no_full_secret_envfrom():
    """S2-02: least-secret production worker gets required keys explicitly."""
    rendered = _helm_template()
    docs = _docs(rendered)
    config = next(
        d
        for d in docs
        if d.get("kind") == "ConfigMap"
        and d.get("metadata", {}).get("name") == "ocr-local-config"
    )
    assert config["data"]["DEPLOYMENT_ENV"] == "production"
    dep = None
    for d in docs:
        if d.get("kind") == "Deployment" and "gpu-worker" in d.get("metadata", {}).get(
            "name", ""
        ):
            dep = d
            break
    assert dep is not None
    containers = (
        dep.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    )
    assert containers
    c0 = containers[0]
    env_from = c0.get("envFrom") or []
    secret_refs = [e for e in env_from if "secretRef" in e]
    assert secret_refs == [], f"unexpected full secret envFrom: {secret_refs}"
    env_list = c0.get("env") or []
    env_names = {e.get("name") for e in env_list}
    env_name_list = [e.get("name") for e in env_list]
    assert len(env_name_list) == len(set(env_name_list)), (
        "duplicate GPU worker env entries: "
        f"{sorted(name for name in set(env_name_list) if env_name_list.count(name) > 1)}"
    )
    assert "CELERY_BROKER_URL" in env_names
    assert "DATABASE_URL" in env_names
    by_name = {entry["name"]: entry for entry in env_list}
    assert by_name["CELERY_TASK_SIGNING_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "ocr-local-secret",
        "key": "CELERY_TASK_SIGNING_KEY",
    }
    assert "PII_ENCRYPTION_KEY" in env_names
    assert "WEBHOOK_SECRET_KEY" in env_names
    assert "S3_SECRET_KEY" not in env_names


def test_review_strong_auth_verifier_and_ca_are_renderable():
    rendered = _helm_template(
        "--set",
        "reviewStrongAuth.verifierUrl=https://verifier.example.test/verify",
        "--set",
        "reviewStrongAuth.caSecretName=review-auth-ca",
    )
    docs = _docs(rendered)
    config = next(d for d in docs if d.get("kind") == "ConfigMap")
    assert config["data"]["REVIEW_STRONG_AUTH_VERIFIER_URL"] == (
        "https://verifier.example.test/verify"
    )
    assert config["data"]["REVIEW_STRONG_AUTH_CA_FILE"] == ("/etc/review-auth/ca.crt")
    api = _deployment(docs, "ocr-local-api")
    pod = api["spec"]["template"]["spec"]
    assert any(volume["name"] == "review-auth-ca" for volume in pod["volumes"])


def test_api_keeps_sqlite_store_separate_from_coordinator_postgres():
    rendered = _helm_template()
    docs = _docs(rendered)
    api = _deployment(docs, "ocr-local-api")
    env = api["spec"]["template"]["spec"]["containers"][0]["env"]
    by_name = {item["name"]: item for item in env}
    assert "DATABASE_URL" not in by_name
    assert by_name["COORDINATOR_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "ocr-local-secret",
        "key": "DATABASE_URL",
    }
    assert by_name["API_DB_PATH"]["value"].endswith("/jobs.db")


def test_gpu_worker_s3_keys_opt_in():
    """S3 keys only when storage.backend=s3 and workerInjectS3Credentials."""
    rendered = _helm_template(
        "--set",
        "storage.backend=s3",
        "--set",
        "storage.workerInjectS3Credentials=true",
        "--set",
        "secrets.s3Endpoint=http://minio:9000",
        "--set",
        "secrets.s3Bucket=ocr",
        "--set",
        "secrets.s3AccessKey=ak",
        "--set",
        "secrets.s3SecretKey=sk",
        "--set",
        "storage.s3.networkPolicy.allowedCidrs[0]=10.0.0.0/8",
    )
    docs = _docs(rendered)
    dep = next(
        d
        for d in docs
        if d.get("kind") == "Deployment"
        and "gpu-worker" in d.get("metadata", {}).get("name", "")
    )
    c0 = dep["spec"]["template"]["spec"]["containers"][0]
    env_names = {e.get("name") for e in (c0.get("env") or [])}
    assert "S3_SECRET_KEY" in env_names
    assert "S3_ACCESS_KEY" in env_names


def test_cpu_ocr_worker_environment_has_no_duplicate_names():
    rendered = _helm_template(
        "--set",
        "cpuOcrWorker.enabled=true",
        "--set",
        "cpuOcrWorker.autoscaling.enabled=true",
    )
    docs = _docs(rendered)
    dep = _deployment(docs, "ocr-local-cpu-ocr-worker")
    env = dep["spec"]["template"]["spec"]["containers"][0].get("env") or []
    names = [item.get("name") for item in env]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    assert not duplicates, f"duplicate CPU OCR worker env entries: {duplicates}"


def test_rabbitmq_erlang_cookie_is_a_secret_data_key():
    rendered = _helm_template()
    docs = _docs(rendered)
    secret = next(
        item
        for item in docs
        if item.get("kind") == "Secret"
        and item.get("metadata", {}).get("name") == "ocr-local-secret"
    )
    cookie = secret.get("data", {}).get("RABBITMQ_ERLANG_COOKIE")
    assert cookie, "RabbitMQ Erlang cookie must render as a Secret data key"


def test_rabbitmq_native_config_has_queue_and_memory_settings():
    rendered = _helm_template()
    docs = _docs(rendered)
    config = next(
        item
        for item in docs
        if item.get("kind") == "ConfigMap"
        and item.get("metadata", {}).get("name") == "ocr-local-rabbitmq-config"
    )
    rabbitmq = next(
        item
        for item in docs
        if item.get("kind") == "StatefulSet"
        and item.get("metadata", {}).get("name") == "ocr-local-rabbitmq"
    )
    config_text = config["data"]["rabbitmq.conf"]
    assert "vm_memory_high_watermark.relative = 0.6" in config_text
    assert "default_queue_type = quorum" in config_text
    container = rabbitmq["spec"]["template"]["spec"]["containers"][0]
    env_names = {entry["name"] for entry in container.get("env", [])}
    assert "RABBITMQ_VM_MEMORY_HIGH_WATERMARK" not in env_names
    assert any(
        mount.get("mountPath") == "/etc/rabbitmq/rabbitmq.conf"
        for mount in container.get("volumeMounts", [])
    )
    assert container["startupProbe"]["timeoutSeconds"] == 10
    assert container["startupProbe"]["failureThreshold"] == 30
    assert container["livenessProbe"]["timeoutSeconds"] == 10


def test_rabbitmq_broker_url_uses_cross_namespace_cluster_dns():
    rendered = _helm_template(
        "--namespace",
        "ocr-system",
        "--set",
        "clusterDomain=cluster.test",
    )
    docs = _docs(rendered)
    secret = next(
        item
        for item in docs
        if item.get("kind") == "Secret"
        and item.get("metadata", {}).get("name") == "ocr-local-secret"
    )
    broker_url = base64.b64decode(secret["data"]["CELERY_BROKER_URL"]).decode()
    assert broker_url == (
        "amqp://ocr:test-rmq@ocr-local-rabbitmq.ocr-system.svc.cluster.test:5672//"
    )


def test_database_url_defaults_to_embedded_postgres_database():
    rendered = _helm_template()
    docs = _docs(rendered)
    secret = next(
        item
        for item in docs
        if item.get("kind") == "Secret"
        and item.get("metadata", {}).get("name") == "ocr-local-secret"
    )
    database_url = base64.b64decode(secret["data"]["DATABASE_URL"]).decode()
    assert database_url == (
        "postgres://ocr:test-pg@ocr-local-postgres:5432/ocr_coordinator"
    )


def _deployment(docs: list[dict], name_substr: str) -> dict:
    for d in docs:
        if d.get("kind") == "Deployment" and name_substr in d.get("metadata", {}).get(
            "name", ""
        ):
            return d
    raise AssertionError(f"Deployment containing {name_substr!r} not found")


def test_coordinator_default_no_full_secret_envfrom():
    """S1-05: coordinator must not envFrom full release Secret by default."""
    rendered = _helm_template()
    docs = _docs(rendered)
    dep = _deployment(docs, "ocr-local-coordinator")
    # Prefer the Gunicorn coordinator, not celery-coordinator
    if dep["metadata"]["name"] != "ocr-local-coordinator":
        dep = next(
            d
            for d in docs
            if d.get("kind") == "Deployment"
            and d.get("metadata", {}).get("name") == "ocr-local-coordinator"
        )
    c0 = dep["spec"]["template"]["spec"]["containers"][0]
    secret_refs = [e for e in (c0.get("envFrom") or []) if "secretRef" in e]
    assert secret_refs == [], f"unexpected full secret envFrom: {secret_refs}"
    env_names = {e.get("name") for e in (c0.get("env") or [])}
    assert "DJANGO_SECRET_KEY" in env_names
    assert "DATABASE_URL" in env_names
    assert "CELERY_BROKER_URL" in env_names
    assert "METRICS_API_KEY" in env_names
    # Blast-radius keys must not be ambient env on coordinator
    assert "FLOWER_PASSWORD" not in env_names
    assert "RABBITMQ_ERLANG_COOKIE" not in env_names
    assert "POSTGRES_PASSWORD" not in env_names


def test_coordinator_migrate_init_also_least_secret():
    rendered = _helm_template()
    docs = _docs(rendered)
    dep = next(
        d
        for d in docs
        if d.get("kind") == "Deployment"
        and d.get("metadata", {}).get("name") == "ocr-local-coordinator"
    )
    inits = dep["spec"]["template"]["spec"].get("initContainers") or []
    assert inits, "migrate initContainer missing"
    mig = inits[0]
    secret_refs = [e for e in (mig.get("envFrom") or []) if "secretRef" in e]
    assert secret_refs == []
    env_names = {e.get("name") for e in (mig.get("env") or [])}
    assert "DATABASE_URL" in env_names


def test_celery_beat_and_celery_coordinator_least_secret():
    rendered = _helm_template()
    docs = _docs(rendered)
    for name in ("ocr-local-celery-beat", "ocr-local-celery-coordinator"):
        dep = next(
            d
            for d in docs
            if d.get("kind") == "Deployment"
            and d.get("metadata", {}).get("name") == name
        )
        c0 = dep["spec"]["template"]["spec"]["containers"][0]
        secret_refs = [e for e in (c0.get("envFrom") or []) if "secretRef" in e]
        assert secret_refs == [], f"{name} still full secretRef envFrom"
        env_names = {e.get("name") for e in (c0.get("env") or [])}
        assert "CELERY_BROKER_URL" in env_names
        assert "FLOWER_PASSWORD" not in env_names
