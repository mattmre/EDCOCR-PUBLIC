"""Rendered S3 worker credential boundary, not template text matching."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CHART = Path(__file__).resolve().parents[1] / "helm" / "ocr-local"
pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is unavailable")


def _render(*extra: str) -> list[dict]:
    settings = {
        "secrets.djangoSecretKey": "test-django",
        "secrets.apiKey": "test-api",
        "secrets.postgresPassword": "test-db",
        "secrets.webhookSecretKey": "test-webhook",
        "secrets.piiEncryptionKey": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "secrets.celeryTaskSigningKey": "test-celery",
        "secrets.rabbitmqPassword": "test-broker",
        "secrets.redisPassword": "test-redis",
        "secrets.flowerPassword": "test-flower",
        "secrets.s3Endpoint": "https://s3.example.test",
        "secrets.s3Bucket": "test-bucket",
        "secrets.s3AccessKey": "test-access",
        "secrets.s3SecretKey": "test-secret",
        "secrets.presignedTaskKey": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "storage.backend": "s3",
        "storage.s3.usePresignedUrls": "true",
        "storage.s3.networkPolicy.allowedCidrs[0]": "192.0.2.0/24",
    }
    command = ["helm", "template", "rescue", str(CHART)]
    for key, value in settings.items():
        command.extend(["--set", f"{key}={value}"])
    command.extend(extra)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise AssertionError(result.stderr)
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def test_presigned_workers_have_no_sdk_credentials_and_coordinator_keeps_them() -> None:
    deployments = {
        document["metadata"]["name"]: document
        for document in _render()
        if document.get("kind") == "Deployment"
    }
    worker_names = [name for name in deployments if name.endswith("-worker")]
    assert worker_names
    for name in worker_names:
        container = deployments[name]["spec"]["template"]["spec"]["containers"][0]
        env = {item["name"]: item for item in container.get("env", [])}
        assert env["S3_PRESIGNED_ONLY_WORKER"]["value"] == "true"
        assert env["AWS_EC2_METADATA_DISABLED"]["value"] == "true"
        assert "S3_ENDPOINT" in env and "S3_BUCKET" in env
        assert "S3_ACCESS_KEY" not in env and "S3_SECRET_KEY" not in env
        assert not any("secretRef" in item for item in container.get("envFrom", []))
    coordinator = deployments["rescue-celery-coordinator"]["spec"]["template"]["spec"]["containers"][0]
    control_env = {item["name"]: item for item in coordinator["env"]}
    assert "S3_ACCESS_KEY" in control_env and "S3_SECRET_KEY" in control_env
    assert "S3_PRESIGNED_ONLY_WORKER" not in control_env


def test_full_secret_worker_mode_cannot_claim_credential_free() -> None:
    with pytest.raises(AssertionError, match="leastSecretEnv=true"):
        _render("--set", "workerSecurity.leastSecretEnv=false")


def _iam_render(*extra: str) -> list[dict]:
    return _render(
        "--set", "storage.s3.authMode=iam",
        "--set", "storage.s3.controlPlaneServiceAccount=owned-iam-coordinator",
        "--set", "storage.s3.presignedAllowedHosts[0]=owned-fixture.s3.us-east-1.amazonaws.com",
        "--set", "secrets.s3AccessKey=", "--set", "secrets.s3SecretKey=",
        "--set", "secrets.s3Endpoint=", *extra,
    )


def test_iam_identity_is_separate_from_presigned_workers() -> None:
    documents = _iam_render("--set", "ocrStagedPage.enabled=true", "--set", "tesseractWorker.enabled=true")
    control_names = {"rescue-api", "rescue-coordinator", "rescue-celery-coordinator", "rescue-celery-beat"}
    seen = set()
    for document in documents:
        if document.get("kind") != "Deployment":
            continue
        name = document["metadata"]["name"]
        pod = document["spec"]["template"]["spec"]
        if name in control_names:
            seen.add(name)
            assert pod["serviceAccountName"] == "owned-iam-coordinator"
        else:
            assert pod["serviceAccountName"] != "owned-iam-coordinator"
        for container in pod["containers"]:
            env = {entry["name"] for entry in container.get("env", [])}
            assert not {"S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_SESSION_TOKEN"} & env
    assert seen == control_names
    secret = next(item for item in documents if item.get("kind") == "Secret")
    assert not {"S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_SESSION_TOKEN"} & secret["data"].keys()
    config = next(item["data"] for item in documents if item.get("kind") == "ConfigMap")
    assert config["S3_PRESIGNED_ALLOWED_HOSTS"] == "owned-fixture.s3.us-east-1.amazonaws.com"


@pytest.mark.parametrize("setting", [
    "storage.s3.controlPlaneServiceAccount=", "storage.s3.controlPlaneServiceAccount=default",
    "storage.s3.controlPlaneServiceAccount=rescue-ocr-local", "storage.s3.usePresignedUrls=false",
    "storage.workerInjectS3Credentials=true", "secrets.s3AccessKey=fixture-key",
    "secrets.s3SessionToken=fixture-token", "storage.s3.authMode=typo",
    "storage.s3.presignedAllowedHosts=[]", "storage.s3.presignedAllowedHosts[0]=*.amazonaws.com",
])
def test_iam_render_refuses_ambiguous_identity_or_credentials(setting: str) -> None:
    with pytest.raises(AssertionError):
        _iam_render("--set", setting)


def test_nfs_root_matches_the_actual_shared_volume_mount() -> None:
    documents = _render(
        "--set", "storage.backend=nfs", "--set", "storage.s3.usePresignedUrls=false",
        "--set", "storage.nfs.persistence.enabled=true", "--set", "storage.nfs.path=/server/export",
        "--set", "storage.nfs.mountPath=/mnt/ocr-shared", "--set", "ocrStagedPage.enabled=true",
        "--set", "tesseractWorker.enabled=true",
    )
    config = next(item["data"] for item in documents if item.get("kind") == "ConfigMap")
    assert config["NFS_ROOT"] == "/mnt/ocr-shared"
    mounts = []
    for document in documents:
        if document.get("kind") != "Deployment":
            continue
        for container in document["spec"]["template"]["spec"]["containers"]:
            mounts.extend(mount for mount in container.get("volumeMounts", []) if mount["name"] == "nfs-data")
    assert mounts
    assert all(mount["mountPath"] == config["NFS_ROOT"] for mount in mounts)


def test_s3_cache_stays_inside_protected_staging_with_bounded_memory() -> None:
    documents = _render()
    config = next(item for item in documents if item.get("kind") == "ConfigMap")
    data = config["data"]
    assert data["S3_CACHE_DIR"] == "/tmp/staging/ocr-cache"
    assert float(data["S3_CACHE_MAX_SIZE_GB"]) == 1
    assert float(data["S3_CACHE_MAX_OBJECT_GB"]) == 0.5
    assert data["OCR_STAGE_LEASE_SECONDS"] == "1800"
    assert data["OCR_STAGE_RECOVERY_INTERVAL_SECONDS"] == "60"
    with pytest.raises(AssertionError, match="cacheDir must be beneath"):
        _render("--set", "storage.s3.cacheDir=/tmp/ocr-cache")
    with pytest.raises(AssertionError, match="capacity/headroom"):
        _render("--set", "storage.s3.cacheMaxSizeGB=1.5")
    with pytest.raises(AssertionError, match="immutable source half"):
        _render("--set", "storage.s3.cacheMaxObjectGB=0.75")
    _render(
        "--set", "workerSecurity.stagingVolume.emptyDir.sizeLimit=2.5Gi",
    )
    with pytest.raises(AssertionError, match="capacity/headroom"):
        _render(
            "--set", "workerSecurity.stagingVolume.emptyDir.sizeLimit=1.5Gi",
        )


def test_pvc_cache_bound_uses_encrypted_staging_capacity() -> None:
    _render(
        "--set", "workerSecurity.stagingVolume.type=pvc",
        "--set", "storage.s3.cacheMaxSizeGB=8",
        "--set", "storage.s3.cacheMaxObjectGB=4",
    )
    with pytest.raises(AssertionError, match="capacity/headroom"):
        _render(
            "--set", "workerSecurity.stagingVolume.type=pvc",
            "--set", "storage.s3.cacheMaxSizeGB=9",
        )


def test_stage_lease_and_beat_bounds_render_as_operational_settings() -> None:
    documents = _render(
        "--set", "ocrStagedPage.leaseSeconds=60",
        "--set", "ocrStagedPage.recoveryIntervalSeconds=5",
    )
    data = next(item["data"] for item in documents if item.get("kind") == "ConfigMap")
    assert data["OCR_STAGE_LEASE_SECONDS"] == "60"
    assert data["OCR_STAGE_RECOVERY_INTERVAL_SECONDS"] == "5"
    for name, value, message in (
        ("leaseSeconds", "19", "leaseSeconds must be between"),
        ("leaseSeconds", "3601", "leaseSeconds must be between"),
        ("recoveryIntervalSeconds", "4", "recoveryIntervalSeconds must be between"),
        ("recoveryIntervalSeconds", "301", "recoveryIntervalSeconds must be between"),
    ):
        with pytest.raises(AssertionError, match=message):
            _render("--set", f"ocrStagedPage.{name}={value}")


def test_staged_and_tesseract_workers_are_covered_by_restricted_network_policy() -> None:
    documents = _render(
        "--set", "ocrStagedPage.enabled=true",
        "--set", "tesseractWorker.enabled=true",
    )
    deployments = {
        item["metadata"]["name"]: item
        for item in documents if item.get("kind") == "Deployment"
    }
    policies = {
        item["metadata"]["name"]: item
        for item in documents if item.get("kind") == "NetworkPolicy"
    }
    policy = policies["rescue-worker-netpol"]["spec"]
    selected = set(policy["podSelector"]["matchExpressions"][0]["values"])
    for name in ("rescue-ocr-render", "rescue-ocr-finalize", "rescue-tesseract-worker"):
        component = deployments[name]["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/component"]
        assert component in selected
    assert policy["policyTypes"] == ["Ingress", "Egress"]
    assert policy["ingress"] == []
    ports = {
        item["port"]
        for rule in policy["egress"]
        for item in rule.get("ports", [])
    }
    assert {53, 5432, 5672, 6379, 443} <= ports
    s3_rules = [
        rule for rule in policy["egress"]
        if any(target.get("ipBlock", {}).get("cidr") == "192.0.2.0/24" for target in rule.get("to", []))
    ]
    assert len(s3_rules) == 1
    assert s3_rules[0]["ports"] == [{"protocol": "TCP", "port": 443}]


def test_api_defaults_match_qualified_single_pipeline_capacity() -> None:
    documents = _render()
    api = next(
        item for item in documents
        if item.get("kind") == "Deployment" and item["metadata"]["name"] == "rescue-api"
    )["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in api["env"] if "value" in item}
    assert {name: env[name] for name in (
        "MAX_CONCURRENT_JOBS", "MAX_QUEUED_JOBS", "NUM_WORKERS",
        "NUM_EXTRACTORS", "NUM_COMPRESSORS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OCR_NATIVE_THREADS_PER_PREDICTOR", "OCR_ENGINE_REPLICAS_PER_KEY",
        "OCR_INFERENCE_SLOTS_PER_DEVICE",
    )} == {
        "MAX_CONCURRENT_JOBS": "1", "MAX_QUEUED_JOBS": "4", "NUM_WORKERS": "2",
        "NUM_EXTRACTORS": "2", "NUM_COMPRESSORS": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OCR_NATIVE_THREADS_PER_PREDICTOR": "1",
        "OCR_ENGINE_REPLICAS_PER_KEY": "2", "OCR_INFERENCE_SLOTS_PER_DEVICE": "2",
    }
    assert api["resources"]["limits"] == {"cpu": "4", "memory": "8Gi"}
    with pytest.raises(AssertionError, match="per-pipeline planning memory budget"):
        _render("--set", "api.maxConcurrentJobs=4")
    with pytest.raises(AssertionError, match="native thread limit"):
        _render("--set", "api.pipeline.ompThreads=3")
    _render(
        "--set", "api.resources.limits.cpu=4000m",
        "--set", "api.resources.limits.memory=8192Mi",
    )
    _render(
        "--set", "api.resources.limits.cpu=4.5",
        "--set", "api.resources.limits.memory=9G",
    )
    _render(
        "--set", "api.resources.limits.cpu=4",
        "--set", "api.resources.limits.memory=8589934592",
    )
    with pytest.raises(AssertionError, match="native thread limit"):
        _render("--set", "api.resources.limits.cpu=0.5")
    with pytest.raises(AssertionError, match="per-pipeline planning memory budget"):
        _render("--set", "api.resources.limits.memory=1.5Gi")
    with pytest.raises(AssertionError, match="per-pipeline planning memory budget"):
        _render("--set", "api.resources.limits.memory=4Gi")


def test_s3_config_consumers_mount_the_protected_cache_volume() -> None:
    deployments = {
        item["metadata"]["name"]: item
        for item in _render() if item.get("kind") == "Deployment"
    }
    for name in (
        "rescue-api", "rescue-coordinator", "rescue-celery-coordinator",
        "rescue-celery-beat",
    ):
        pod = deployments[name]["spec"]["template"]["spec"]
        volume = next(item for item in pod["volumes"] if item["name"] == "s3-cache-staging")
        assert volume["emptyDir"] == {"medium": "Memory", "sizeLimit": "2Gi"}
        container = pod["containers"][0]
        assert {"name": "s3-cache-staging", "mountPath": "/tmp/staging"} in container["volumeMounts"]
    pvc_deployments = {
        item["metadata"]["name"]: item
        for item in _render("--set", "workerSecurity.stagingVolume.type=pvc")
        if item.get("kind") == "Deployment"
    }
    for name in ("rescue-api", "rescue-celery-coordinator"):
        volumes = pvc_deployments[name]["spec"]["template"]["spec"]["volumes"]
        cache = next(item for item in volumes if item["name"] == "s3-cache-staging")
        claim = cache["ephemeral"]["volumeClaimTemplate"]["spec"]
        assert claim["storageClassName"] == "encrypted-sc"
        assert claim["resources"]["requests"]["storage"] == "10Gi"
