"""Helm contract tests for the durable processing-engine OCR provider."""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHART_PATH = REPO_ROOT / "helm" / "ocr-local"
PRODUCTION_VALUES = CHART_PATH / "values-production.yaml"
ACCEPTANCE_VALUES = CHART_PATH / "values-processing-acceptance.yaml"
LOCAL_SANDBOX_VALUES = CHART_PATH / "values-local-sandbox.yaml"
ACCEPTANCE_DATABASE_URL = (
    "postgres://ocr:test-pg@postgres-pooler."
    "ocr-lab-processing-acceptance.svc:5432/ocr_coordinator"
)
MIN_SECRETS = {
    "secrets.djangoSecretKey": "test-django-key",
    "secrets.coordinatorApiKey": "test-coordinator-key",
    "secrets.apiKey": "test-api-key",
    "secrets.postgresPassword": "test-pg",
    "secrets.rabbitmqPassword": "test-rmq",
    "secrets.redisPassword": "test-redis",
    "secrets.celeryTaskSigningKey": "test-task-signing",
    "secrets.webhookSecretKey": "test-webhook-secret",
    "secrets.piiEncryptionKey": "0P-5W9e3q8G0lr2xW6iA_9h9x57DmeH3Qq2FVDgKqWQ=",
    "secrets.s3Endpoint": "https://object-store:9000",
    "secrets.s3Bucket": "ocr-test",
    "secrets.s3AccessKey": "test-s3-access",
    "secrets.s3SecretKey": "test-s3-secret",
}
PRODUCTION_RABBITMQ_OVERRIDES = (
    "rabbitmq.enabled=false",
    "secrets.celeryBrokerUrl=amqps://ocr:brokerpass@rabbit.example.invalid:5671/%2F",
    "keda.rabbitmqManagementHost=https://rabbit.example.invalid:15671/",
    "keda.rabbitmqManagementCaSecretName=production-rabbit-ca",
    "secrets.rabbitmqManagementUser=ocr-keda-monitor",
    "secrets.rabbitmqManagementPassword=test-rmq-monitor",
    "networkPolicy.externalDependencies.rabbitmq.namespaceSelector.matchLabels.role=messaging",
    "networkPolicy.externalDependencies.rabbitmq.podSelector.matchLabels.app=rabbitmq",
)

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None, reason="helm CLI not on PATH"
)


def _render(
    *overrides: str,
    production: bool = False,
    acceptance: bool = False,
    local_sandbox: bool = False,
) -> list[dict]:
    command = ["helm", "template", "ocr-provider", str(CHART_PATH)]
    assert not (production and acceptance)
    if production:
        command.extend(["--values", str(PRODUCTION_VALUES)])
        command.extend(["--set", "image.requireDigest=false"])
    if acceptance:
        command.extend(["--values", str(ACCEPTANCE_VALUES)])
        command.extend(["--set", f"secrets.databaseUrl={ACCEPTANCE_DATABASE_URL}"])
    if local_sandbox:
        command.extend(["--values", str(LOCAL_SANDBOX_VALUES)])
    for key, value in MIN_SECRETS.items():
        command.extend(["--set", f"{key}={value}"])
    for override in overrides:
        command.extend(["--set", override])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def _resource(documents: list[dict], kind: str, suffix: str) -> dict:
    expected_name = f"ocr-provider{suffix}"
    return next(
        document
        for document in documents
        if document.get("kind") == kind
        and document.get("metadata", {}).get("name", "") == expected_name
    )


def _container(deployment: dict, name: str) -> dict:
    return next(
        container
        for container in deployment["spec"]["template"]["spec"]["containers"]
        if container["name"] == name
    )


def test_coordinator_readiness_uses_authenticated_detailed_contract():
    documents = _render()
    deployment = _resource(documents, "Deployment", "-coordinator")
    coordinator = _container(deployment, "coordinator")
    env = {item["name"]: item for item in coordinator["env"]}
    command = coordinator["readinessProbe"]["exec"]["command"]

    assert "OCR_COORDINATOR_API_KEY" in env
    assert env["OCR_COORDINATOR_API_KEY"]["valueFrom"]["secretKeyRef"]["key"] == (
        "OCR_COORDINATOR_API_KEY"
    )
    assert "/api/v1/health/detailed" in command[-1]
    assert "OCR_COORDINATOR_HEALTH_TENANT_ID" in command[-1]
    for probe_name in ("startupProbe", "livenessProbe"):
        headers = coordinator[probe_name]["httpGet"]["httpHeaders"]
        assert {item["name"]: item["value"] for item in headers}["Host"] == "localhost"


def test_zero_worker_readiness_tracks_enabled_scale_to_zero_queue():
    default_documents = _render()
    default_coordinator = _resource(default_documents, "Deployment", "-coordinator")
    default_env = {
        item["name"]: item
        for item in _container(default_coordinator, "coordinator")["env"]
    }
    assert (
        default_env["OCR_COORDINATOR_ALLOW_ZERO_WORKER_READINESS"]["value"] == "false"
    )

    scaled_documents = _render(
        "cpuOcrWorker.enabled=true",
        "cpuOcrWorker.autoscaling.enabled=true",
        "cpuOcrWorker.autoscaling.minReplicas=0",
    )
    scaled_coordinator = _resource(scaled_documents, "Deployment", "-coordinator")
    scaled_env = {
        item["name"]: item
        for item in _container(scaled_coordinator, "coordinator")["env"]
    }
    assert scaled_env["OCR_COORDINATOR_ALLOW_ZERO_WORKER_READINESS"]["value"] == "true"


def test_coordinator_routing_policy_reaches_ingest_dispatcher():
    documents = _render("coordinator.env.OCR_TASK_ROUTING=cpu")
    config = _resource(documents, "ConfigMap", "-config")
    celery = _resource(documents, "Deployment", "-celery-coordinator")
    env_from = _container(celery, "celery-coordinator")["envFrom"]

    assert config["data"]["OCR_TASK_ROUTING"] == "cpu"
    assert {
        item["configMapRef"]["name"] for item in env_from if "configMapRef" in item
    } >= {"ocr-provider-config"}


def test_cpu_only_local_sandbox_routes_to_cpu_ocr_and_runs_cpu_general_worker():
    documents = _render(local_sandbox=True)
    config = _resource(documents, "ConfigMap", "-config")
    cpu_ocr = _resource(documents, "Deployment", "-cpu-ocr-worker")
    cpu_general = _resource(documents, "Deployment", "-cpu-worker")

    assert config["data"]["OCR_TASK_ROUTING"] == "cpu"
    assert cpu_ocr["spec"]["replicas"] == 1
    assert cpu_general["spec"]["replicas"] == 1


def test_coordinator_api_key_is_not_injected_into_execution_workers():
    documents = _render()
    worker = _resource(documents, "Deployment", "-cpu-worker")
    names = {item["name"] for item in _container(worker, "cpu-worker")["env"]}

    assert "OCR_COORDINATOR_API_KEY" not in names
    assert {"CELERY_TASK_SIGNING_KEY", "PII_ENCRYPTION_KEY"} <= names


def test_keda_http_scaler_uses_management_host_and_separate_rabbit_credentials():
    documents = _render(
        "cpuOcrWorker.enabled=true",
        "cpuOcrWorker.autoscaling.enabled=true",
    )
    secret = _resource(documents, "Secret", "-secret")
    trigger_auth = _resource(documents, "TriggerAuthentication", "-rabbitmq-auth")
    management_host = base64.b64decode(
        secret["data"]["RABBITMQ_MANAGEMENT_HOST"]
    ).decode()
    assert management_host == (
        "http://ocr-provider-rabbitmq.default.svc.cluster.local:15672/"
    )
    references = {
        item["parameter"]: item["key"]
        for item in trigger_auth["spec"]["secretTargetRef"]
    }
    assert references == {
        "host": "RABBITMQ_MANAGEMENT_HOST",
        "username": "RABBITMQ_MANAGEMENT_USER",
        "password": "RABBITMQ_MANAGEMENT_PASSWORD",
    }
    assert base64.b64decode(secret["data"]["RABBITMQ_MANAGEMENT_USER"]).decode() == (
        "ocr"
    )


def test_keda_tls_scaler_uses_broker_management_certificate_and_ca():
    documents = _render("tls.enabled=true", "cpuWorker.autoscaling.enabled=true")
    secret = _resource(documents, "Secret", "-secret")
    management_host = base64.b64decode(
        secret["data"]["RABBITMQ_MANAGEMENT_HOST"]
    ).decode()
    assert management_host == (
        "https://ocr-provider-rabbitmq.default.svc.cluster.local:15671/"
    )
    trigger_auth = _resource(documents, "TriggerAuthentication", "-rabbitmq-auth")
    references = {
        item["parameter"]: (item["name"], item["key"])
        for item in trigger_auth["spec"]["secretTargetRef"]
    }
    assert references["tls"] == ("ocr-provider-secret", "RABBITMQ_MANAGEMENT_TLS")
    assert references["ca"] == ("ocr-provider-ocr-local-ca-secret", "ca.crt")
    rabbitmq = _resource(documents, "StatefulSet", "-rabbitmq")
    container = rabbitmq["spec"]["template"]["spec"]["containers"][0]
    assert {port["containerPort"] for port in container["ports"]} >= {5671, 15671}
    assert 15672 not in {port["containerPort"] for port in container["ports"]}
    config = _resource(documents, "ConfigMap", "-rabbitmq-config")["data"][
        "rabbitmq.conf"
    ]
    assert "management.tcp.port = none" in config
    assert "management.ssl.port = 15671" in config


def test_cpu_ocr_worker_uses_dedicated_qualified_image_and_startup_gate():
    documents = _render(
        "cpuOcrWorker.enabled=true",
        "image.cpuOcrWorker.repository=registry.example/ocr-cpu",
        "image.cpuOcrWorker.tag=qualified",
    )
    deployment = _resource(documents, "Deployment", "-cpu-ocr-worker")
    worker = _container(deployment, "cpu-ocr-worker")

    assert worker["image"] == "registry.example/ocr-cpu:qualified"
    startup = worker["startupProbe"]["exec"]["command"][-1]
    assert "paddleocr" in startup
    assert "pytesseract" in startup
    assert "tesseract" in startup
    assert ".paddleocr" in startup
    assert worker["startupProbe"]["timeoutSeconds"] == 30
    assert worker["livenessProbe"]["timeoutSeconds"] == 15


def test_processing_acceptance_keeps_ocr_and_postprocessing_queues_isolated():
    documents = _render(acceptance=True)
    ocr_deployment = _resource(documents, "Deployment", "-cpu-ocr-worker")
    general_deployment = _resource(documents, "Deployment", "-cpu-worker")

    ocr_command = _container(ocr_deployment, "cpu-ocr-worker")["command"]
    general_command = _container(general_deployment, "cpu-worker")["command"]
    assert "--queues=ocr_cpu" in ocr_command
    assert "--queues=cpu_general,nlp_general" in general_command
    assert all("cpu_general" not in argument for argument in ocr_command)
    assert all("ocr_cpu" not in argument for argument in general_command)


def test_cpu_ocr_worker_rejects_foreign_queue_subscription():
    command = [
        "helm",
        "template",
        "ocr-provider",
        str(CHART_PATH),
        "--set",
        "cpuOcrWorker.enabled=true",
        "--set",
        r"cpuOcrWorker.queues=ocr_cpu\,cpu_general",
    ]
    for key, value in MIN_SECRETS.items():
        command.extend(["--set", f"{key}={value}"])

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "cpuOcrWorker.queues must be exactly ocr_cpu" in result.stderr


def test_capability_separation_renders_owned_workers_and_keda_queue():
    documents = _render(
        "ocrCapabilitySeparation.enabled=true",
        "cpuOcrWorker.enabled=true",
        "tesseractWorker.enabled=true",
        "tesseractWorker.autoscaling.enabled=true",
        "image.cpuOcrWorker.repository=registry.example/ocr-paddle",
        "image.tesseractWorker.repository=registry.example/ocr-tesseract",
    )
    config = _resource(documents, "ConfigMap", "-config")
    paddle = _container(
        _resource(documents, "Deployment", "-cpu-ocr-worker"),
        "cpu-ocr-worker",
    )
    tesseract = _container(
        _resource(documents, "Deployment", "-tesseract-worker"),
        "tesseract-worker",
    )
    scaler = _resource(documents, "ScaledObject", "-tesseract-worker-scaler")

    assert config["data"]["OCR_SEPARATE_TESSERACT_LANE"] == "true"
    assert "--queues=ocr_cpu" in paddle["command"]
    assert "--queues=ocr_tesseract" in tesseract["command"]
    startup = tesseract["startupProbe"]["exec"]["command"][-1]
    assert "import celery, pytesseract" in startup
    assert "import celery, paddle" not in startup
    assert scaler["spec"]["triggers"][0]["metadata"]["queueName"] == ("ocr_tesseract")
    assert scaler["spec"]["triggers"][0]["metadata"]["protocol"] == "http"
    assert scaler["spec"]["triggers"][0]["metadata"]["excludeUnacknowledged"] == (
        "false"
    )


def test_production_keda_requires_trusted_https_management_endpoint():
    command = [
        "helm",
        "template",
        "ocr-provider",
        str(CHART_PATH),
        "--values",
        str(PRODUCTION_VALUES),
        "--set",
        "image.requireDigest=false",
    ]
    for key, value in MIN_SECRETS.items():
        command.extend(["--set", f"{key}={value}"])
    for override in PRODUCTION_RABBITMQ_OVERRIDES[-2:]:
        command.extend(["--set", override])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "requires operator-managed RabbitMQ" in result.stderr

    insecure = command + ["--set", "tls.enabled=false"]
    insecure_result = subprocess.run(
        insecure, capture_output=True, text=True, check=False
    )
    assert insecure_result.returncode != 0
    assert "requires operator-managed RabbitMQ" in insecure_result.stderr

    documents = _render(*PRODUCTION_RABBITMQ_OVERRIDES, production=True)
    secret = _resource(documents, "Secret", "-secret")
    management_host = base64.b64decode(
        secret["data"]["RABBITMQ_MANAGEMENT_HOST"]
    ).decode()
    assert management_host == "https://rabbit.example.invalid:15671/"
    trigger_auth = _resource(documents, "TriggerAuthentication", "-rabbitmq-auth")
    refs = {
        item["parameter"]: (item["name"], item["key"])
        for item in trigger_auth["spec"]["secretTargetRef"]
    }
    assert refs["ca"] == ("production-rabbit-ca", "ca.crt")


def test_bundled_rabbitmq_does_not_claim_external_management_identity():
    documents = _render()
    rabbitmq = _resource(documents, "StatefulSet", "-rabbitmq")
    container = rabbitmq["spec"]["template"]["spec"]["containers"][0]
    env_names = {item["name"] for item in container["env"]}
    assert "RABBITMQ_MANAGEMENT_USER" not in env_names
    assert "lifecycle" not in container


def test_production_rabbitmq_egress_requires_narrow_external_selectors():
    command = [
        "helm",
        "template",
        "ocr-provider",
        str(CHART_PATH),
        "--values",
        str(PRODUCTION_VALUES),
        "--set",
        "image.requireDigest=false",
    ]
    for key, value in MIN_SECRETS.items():
        command.extend(["--set", f"{key}={value}"])
    for override in PRODUCTION_RABBITMQ_OVERRIDES[:-2]:
        command.extend(["--set", override])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "requires nonempty namespaceSelector.matchLabels" in result.stderr


def test_capability_separation_rejects_missing_fallback_worker():
    command = [
        "helm",
        "template",
        "ocr-provider",
        str(CHART_PATH),
        "--set",
        "ocrCapabilitySeparation.enabled=true",
        "--set",
        "cpuOcrWorker.enabled=true",
    ]
    for key, value in MIN_SECRETS.items():
        command.extend(["--set", f"{key}={value}"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "requires tesseractWorker.enabled=true" in result.stderr


def test_cpu_acceptance_postprocessing_has_required_runtime_and_queue_lanes():
    documents = _render(acceptance=True)
    config = _resource(documents, "ConfigMap", "-config")
    deployment = _resource(documents, "Deployment", "-cpu-worker")
    worker = _container(deployment, "cpu-worker")

    assert worker["image"].startswith("ocr-local/worker-cpu:")
    command = worker["command"]
    assert "--queues=cpu_general,nlp_general" in command
    assert deployment["spec"]["template"]["spec"]["nodeSelector"] == {
        "kubernetes.io/hostname": "ocr-lab-broker-ha-worker"
    }
    assert config["data"]["ALLOW_DEGRADED_NER"] == "true"


def test_processing_acceptance_uses_capability_separated_worker_images():
    documents = _render(acceptance=True)
    config = _resource(documents, "ConfigMap", "-config")
    paddle = _container(
        _resource(documents, "Deployment", "-cpu-ocr-worker"),
        "cpu-ocr-worker",
    )
    tesseract = _container(
        _resource(documents, "Deployment", "-tesseract-worker"),
        "tesseract-worker",
    )
    paddle_startup = paddle["startupProbe"]["exec"]["command"][-1]
    tesseract_startup = tesseract["startupProbe"]["exec"]["command"][-1]
    paddle_env = {item["name"]: item.get("value") for item in paddle["env"]}

    assert config["data"]["OCR_SEPARATE_TESSERACT_LANE"] == "true"
    assert paddle["image"].startswith("ocr-local/worker-paddle:")
    assert tesseract["image"].startswith("ocr-local/worker-tesseract:")
    assert "--queues=ocr_cpu" in paddle["command"]
    assert "--queues=ocr_tesseract" in tesseract["command"]
    assert paddle_env["OCR_WORKER_CAPABILITY"] == "paddle"
    assert "assert not shutil.which('tesseract')" in paddle_startup
    assert "import celery, paddle, paddleocr" in paddle_startup
    assert "import celery, pytesseract" in tesseract_startup
    assert "find_spec('paddleocr') is None" in tesseract_startup


def test_celery_worker_liveness_allows_inspect_command_to_finish():
    documents = _render()

    for suffix, container_name in (
        ("-celery-coordinator", "celery-coordinator"),
        ("-cpu-worker", "cpu-worker"),
    ):
        deployment = _resource(documents, "Deployment", suffix)
        assert (
            _container(deployment, container_name)["livenessProbe"]["timeoutSeconds"]
            == 15
        )


def test_shared_dependencies_are_selected_by_namespace_and_pod_identity():
    documents = _render(
        "storage.backend=s3",
        "secrets.s3Endpoint=https://object-store:9000",
        "secrets.s3Bucket=ocr",
        "secrets.s3AccessKey=test-access",
        "secrets.s3SecretKey=test-secret",
        "networkPolicy.externalDependencies.postgresql.enabled=true",
        "networkPolicy.externalDependencies.postgresql.namespaceSelector.matchLabels.scope=acceptance",
        "networkPolicy.externalDependencies.postgresql.podSelector.matchLabels.role=pooler",
        "networkPolicy.externalDependencies.rabbitmq.enabled=true",
        "networkPolicy.externalDependencies.rabbitmq.namespaceSelector.matchLabels.scope=broker",
        "networkPolicy.externalDependencies.rabbitmq.podSelector.matchLabels.role=rabbit",
        "networkPolicy.externalDependencies.redis.enabled=true",
        "networkPolicy.externalDependencies.redis.namespaceSelector.matchLabels.scope=cache",
        "networkPolicy.externalDependencies.redis.podSelector.matchLabels.role=redis-proxy",
        "networkPolicy.externalDependencies.s3.enabled=true",
        "networkPolicy.externalDependencies.s3.namespaceSelector.matchLabels.scope=acceptance",
        "networkPolicy.externalDependencies.s3.podSelector.matchLabels.role=object-store",
        "networkPolicy.externalDependencies.s3.port=9000",
    )

    for suffix in ("-worker-netpol", "-coordinator-netpol"):
        policy = _resource(documents, "NetworkPolicy", suffix)
        peers = [
            peer for rule in policy["spec"]["egress"] for peer in rule.get("to", [])
        ]
        identities = {
            (
                peer.get("namespaceSelector", {}).get("matchLabels", {}).get("scope"),
                peer.get("podSelector", {}).get("matchLabels", {}).get("role"),
            )
            for peer in peers
        }
        assert ("acceptance", "pooler") in identities
        assert ("broker", "rabbit") in identities
        assert ("cache", "redis-proxy") in identities
        assert ("acceptance", "object-store") in identities
        assert any(
            port["port"] == 9000
            for rule in policy["spec"]["egress"]
            for port in rule.get("ports", [])
        )

    control_policy = _resource(documents, "NetworkPolicy", "-coordinator-netpol")
    components = control_policy["spec"]["podSelector"]["matchExpressions"][0]["values"]
    assert {"coordinator", "celery-coordinator", "celery-beat"} <= set(components)


def test_redis_sentinel_wires_every_application_client_to_three_nodes():
    documents = _render("redis.sentinel.enabled=true")
    config = _resource(documents, "ConfigMap", "-config")
    secret = _resource(documents, "Secret", "-secret")
    replica_service = _resource(documents, "Service", "-redis-replica")
    redis_policy = _resource(documents, "NetworkPolicy", "-redis-netpol")

    assert config["data"]["REDIS_SENTINEL_MASTER_NAME"] == "ocr-master"
    nodes = config["data"]["REDIS_SENTINEL_NODES"].split(",")
    assert len(nodes) == 3
    assert all(node.endswith(":26379") for node in nodes)
    sentinel = _resource(documents, "StatefulSet", "-redis-sentinel")
    sentinel_command = sentinel["spec"]["template"]["spec"]["containers"][0]["command"][
        -1
    ]
    assert "sentinel resolve-hostnames yes" in sentinel_command
    assert "sentinel announce-hostnames yes" in sentinel_command
    result_backend = base64.b64decode(secret["data"]["CELERY_RESULT_BACKEND"]).decode()
    assert result_backend.count("sentinel://") == 3
    assert result_backend.count(";") == 2
    assert replica_service["spec"]["clusterIP"] == "None"
    assert {
        port["port"]
        for rule in redis_policy["spec"]["egress"]
        for port in rule["ports"]
    } >= {
        53,
        6379,
        26379,
    }

    for suffix in ("-redis-replica-pdb", "-redis-sentinel-pdb"):
        assert (
            _resource(documents, "PodDisruptionBudget", suffix)["spec"][
                "maxUnavailable"
            ]
            == 1
        )


def test_private_s3_ca_is_mounted_and_exported_to_botocore():
    documents = _render(
        "storage.backend=s3",
        "storage.s3.caSecretName=processing-object-ca",
        "secrets.s3Endpoint=https://object-store:9000",
        "secrets.s3Bucket=ocr",
        "secrets.s3AccessKey=test-access",
        "secrets.s3SecretKey=test-secret",
        "storage.s3.networkPolicy.allowedCidrs[0]=10.96.0.0/16",
    )
    config = _resource(documents, "ConfigMap", "-config")
    coordinator = _resource(documents, "Deployment", "-coordinator")
    pod_spec = coordinator["spec"]["template"]["spec"]

    assert config["data"]["AWS_CA_BUNDLE"] == "/etc/s3-ca/ca.crt"
    ca_volume = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == "s3-ca"
    )
    assert ca_volume["secret"]["secretName"] == "processing-object-ca"
    mount = next(
        item
        for item in _container(coordinator, "coordinator")["volumeMounts"]
        if item["name"] == "s3-ca"
    )
    assert mount["mountPath"] == "/etc/s3-ca/ca.crt"
    assert mount["readOnly"] is True


def test_dependency_client_tls_supports_mixed_transport_and_trust_roots():
    endpoint = "postgres://ocr:test-pg@external-db.example:5432/ocr_coordinator"
    documents = _render(
        "postgresql.enabled=false",
        f"secrets.databaseUrl={endpoint}",
        "postgresql.clientTls.enabled=true",
        "postgresql.clientTls.caSecretName=processing-postgres-ca",
        "postgresql.clientTls.caMountPath=/etc/postgresql-ca/ca.crt",
        "rabbitmq.clientTls.enabled=false",
        "redis.clientTls.enabled=false",
    )
    config = _resource(documents, "ConfigMap", "-config")
    coordinator = _resource(documents, "Deployment", "-coordinator")
    pod_spec = coordinator["spec"]["template"]["spec"]

    assert config["data"]["DATABASE_SSLMODE"] == "verify-full"
    assert config["data"]["DATABASE_SSLROOTCERT"] == "/etc/postgresql-ca/ca.crt"
    assert "CELERY_BROKER_USE_SSL" not in config["data"]
    assert "REDIS_SSL" not in config["data"]
    assert all(volume["name"] != "tls-ca" for volume in pod_spec["volumes"])
    database_ca = next(
        volume
        for volume in pod_spec["volumes"]
        if volume["name"] == "postgresql-client-ca"
    )
    assert database_ca["secret"]["secretName"] == "processing-postgres-ca"
    mount = next(
        item
        for item in _container(coordinator, "coordinator")["volumeMounts"]
        if item["name"] == "postgresql-client-ca"
    )
    assert mount["mountPath"] == "/etc/postgresql-ca/ca.crt"
    assert mount["subPath"] == "ca.crt"
    secret = _resource(documents, "Secret", "-secret")
    assert base64.b64decode(secret["data"]["DATABASE_URL"]).decode() == endpoint
    assert not any(
        document.get("kind") == "StatefulSet"
        and document.get("metadata", {}).get("name", "").endswith("-postgres")
        for document in documents
    )


def test_processing_acceptance_requires_external_database_url():
    command = ["helm", "template", "ocr-provider", str(CHART_PATH), "--values", str(ACCEPTANCE_VALUES)]
    for key, value in MIN_SECRETS.items():
        command.extend(["--set", f"{key}={value}"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "bundled chart alias" in result.stderr


def test_processing_acceptance_preserves_external_database_tls():
    documents = _render(acceptance=True)
    config = _resource(documents, "ConfigMap", "-config")
    secret = _resource(documents, "Secret", "-secret")
    assert config["data"]["DATABASE_SSLMODE"] == "verify-full"
    assert config["data"]["DATABASE_SSLROOTCERT"] == "/etc/postgresql-ca/ca.crt"
    assert base64.b64decode(secret["data"]["DATABASE_URL"]).decode() == ACCEPTANCE_DATABASE_URL
    assert not any(
        document.get("kind") == "StatefulSet" and document["metadata"]["name"].endswith("-postgres")
        for document in documents
    )


def test_dependency_client_tls_requires_an_explicit_ca_secret():
    command = ["helm", "template", "ocr-provider", str(CHART_PATH)]
    for key, value in MIN_SECRETS.items():
        command.extend(["--set", f"{key}={value}"])
    command.extend(["--set", "postgresql.clientTls.enabled=true"])

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "postgresql.clientTls.caSecretName is required" in result.stderr


def test_isolated_database_schema_runs_owner_check_before_migrate():
    documents = _render("coordinator.env.DATABASE_SCHEMA=ocr_processing")
    config = _resource(documents, "ConfigMap", "-config")
    deployment = _resource(documents, "Deployment", "-coordinator")
    init_containers = deployment["spec"]["template"]["spec"]["initContainers"]

    assert config["data"]["DATABASE_SCHEMA"] == "ocr_processing"
    assert [container["name"] for container in init_containers[:2]] == [
        "ensure-database-schema",
        "migrate",
    ]
    assert init_containers[0]["command"] == [
        "python",
        "manage.py",
        "ensure_database_schema",
    ]


def test_production_provider_has_replicated_control_plane_and_isolated_schema():
    documents = _render(*PRODUCTION_RABBITMQ_OVERRIDES, production=True)
    config = _resource(documents, "ConfigMap", "-config")
    coordinator = _resource(documents, "Deployment", "-coordinator")
    celery_coordinator = _resource(documents, "Deployment", "-celery-coordinator")

    assert coordinator["spec"]["replicas"] >= 2
    assert celery_coordinator["spec"]["replicas"] >= 2
    assert config["data"]["DATABASE_SCHEMA"] == "ocr_processing"
    for deployment in (coordinator, celery_coordinator):
        pod_spec = deployment["spec"]["template"]["spec"]
        topology_keys = {
            constraint["topologyKey"]
            for constraint in pod_spec["topologySpreadConstraints"]
        }
        assert topology_keys == {
            "kubernetes.io/hostname",
            "topology.kubernetes.io/zone",
        }
        assert pod_spec["affinity"]["podAntiAffinity"][
            "preferredDuringSchedulingIgnoredDuringExecution"
        ]

    coordinator_pdb = _resource(documents, "PodDisruptionBudget", "-coordinator-pdb")
    celery_pdb = _resource(documents, "PodDisruptionBudget", "-celery-coordinator-pdb")
    assert coordinator_pdb["spec"]["maxUnavailable"] == 1
    assert celery_pdb["spec"]["minAvailable"] == 1
