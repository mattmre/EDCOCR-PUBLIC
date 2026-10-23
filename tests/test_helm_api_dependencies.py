"""Render the API's real dependency network and TLS configuration together."""

import base64
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CHART = Path(__file__).resolve().parents[1] / "helm" / "ocr-local"
pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm required")


def render(**settings):
    namespace = settings.pop("namespace", None)
    command = ["helm", "template", "test", str(CHART)]
    if namespace:
        command.extend(["--namespace", namespace])
    for key in (
        "djangoSecretKey", "apiKey", "postgresPassword", "webhookSecretKey",
        "celeryTaskSigningKey", "rabbitmqPassword", "redisPassword", "flowerPassword",
    ):
        command.extend(["--set-string", f"secrets.{key}=render-test-only"])
    command.extend(["--set-string", "secrets.piiEncryptionKey=" + "A" * 43 + "="])
    for key, value in settings.items():
        command.extend(["--set", f"{key}={str(value).lower() if isinstance(value, bool) else value}"])
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return {doc["metadata"]["name"]: doc for doc in yaml.safe_load_all(result.stdout) if doc}


def _secret_value(docs, key):
    secret = next(doc for doc in docs.values() if key in doc.get("data", {}))
    return base64.b64decode(secret["data"][key]).decode()


@pytest.mark.parametrize("tls,ports", [(False, [5432, 5672, 6379]), (True, [5432, 5671, 6380])])
def test_api_egress_reaches_bundled_dependencies(tls, ports):
    docs = render(**{"tls.enabled": tls})
    api = docs["test-api-netpol"]["spec"]
    coordinator = docs["test-coordinator-netpol"]["spec"]
    assert api["podSelector"]["matchLabels"]["app.kubernetes.io/component"] == "api"
    assert api["egress"] == coordinator["egress"]
    for component, port in zip(["postgres", "rabbitmq", "redis"], ports):
        rule = next(rule for rule in api["egress"] if any(
            peer.get("podSelector", {}).get("matchLabels", {}).get("app.kubernetes.io/component") == component
            for peer in rule.get("to", [])
        ))
        assert rule["ports"] == [{"protocol": "TCP", "port": port}]


def test_api_egress_obeys_external_selectors_and_ports():
    settings = {}
    for dependency, port in [("postgresql", 15432), ("rabbitmq", 15671), ("redis", 16380)]:
        prefix = f"networkPolicy.externalDependencies.{dependency}"
        settings.update({
            f"{prefix}.enabled": True,
            f"{prefix}.namespaceSelector.matchLabels.environment": "infra",
            f"{prefix}.podSelector.matchLabels.service": dependency,
            f"{prefix}.port": port,
        })
    docs = render(**settings)
    rules = docs["test-api-netpol"]["spec"]["egress"]
    for dependency, port in [("postgresql", 15432), ("rabbitmq", 15671), ("redis", 16380)]:
        rule = next(rule for rule in rules if rule.get("ports") == [{"protocol": "TCP", "port": port}])
        assert rule["to"] == [{
            "namespaceSelector": {"matchLabels": {"environment": "infra"}},
            "podSelector": {"matchLabels": {"service": dependency}},
        }]


def test_api_mounts_dependency_ca_secrets():
    settings = {"tls.enabled": True, "storage.s3.caSecretName": "s3-trust"}
    for dependency in ["postgresql", "rabbitmq", "redis"]:
        settings[f"{dependency}.clientTls.enabled"] = True
        settings[f"{dependency}.clientTls.caSecretName"] = f"{dependency}-trust"
    docs = render(**settings)
    pod = docs["test-api"]["spec"]["template"]["spec"]
    coordinator = docs["test-coordinator"]["spec"]["template"]["spec"]
    expected = {"tls-ca", "postgresql-client-ca", "rabbitmq-client-ca", "redis-client-ca", "s3-ca"}
    mounts = {item["name"]: item for item in pod["containers"][0]["volumeMounts"]}
    volumes = {item["name"]: item for item in pod["volumes"]}
    coordinator_mounts = {item["name"]: item for item in coordinator["containers"][0]["volumeMounts"]}
    for name in expected:
        assert mounts[name] == coordinator_mounts[name]
        assert mounts[name]["readOnly"] is True
        assert volumes[name]["secret"]["secretName"]


def test_client_tls_alone_cannot_generate_an_unreachable_bundled_broker():
    with pytest.raises(AssertionError, match="client TLS alone does not configure a broker TLS listener"):
        render(**{
            "rabbitmq.clientTls.enabled": True,
            "rabbitmq.clientTls.caSecretName": "external-broker-ca",
        })


def test_external_broker_client_tls_preserves_explicit_endpoint():
    endpoint = "amqps://external-broker.example:5671//"
    docs = render(**{
        "rabbitmq.enabled": False,
        "rabbitmq.clientTls.enabled": True,
        "rabbitmq.clientTls.caSecretName": "external-broker-ca",
        "secrets.celeryBrokerUrl": endpoint,
    })
    secret = next(doc for doc in docs.values() if "CELERY_BROKER_URL" in doc.get("data", {}))
    assert base64.b64decode(secret["data"]["CELERY_BROKER_URL"]).decode() == endpoint



BUNDLED_BROKER_ALIASES = (
    "amqps://test-rabbitmq:5671//",
    "amqps://test-rabbitmq.default:5671//",
    "amqps://test-rabbitmq.default.svc:5671//",
    "amqps://test-rabbitmq.default.svc.cluster.local:5671//",
    "amqps://test-rabbitmq-0.test-rabbitmq:5671//",
    "amqps://test-rabbitmq-0.test-rabbitmq.default.svc.cluster.local:5671//",
)


@pytest.mark.parametrize("endpoint", BUNDLED_BROKER_ALIASES)
@pytest.mark.parametrize("client_tls", [False, True])
def test_explicit_internal_amqps_cannot_bypass_missing_listener(endpoint, client_tls):
    settings = {"secrets.celeryBrokerUrl": endpoint}
    if client_tls:
        settings.update({
            "rabbitmq.clientTls.enabled": True,
            "rabbitmq.clientTls.caSecretName": "broker-ca",
        })
    with pytest.raises(AssertionError, match="no TLS listener|no SSL listener"):
        render(**settings)


def test_namespaced_and_custom_domain_broker_aliases_are_rejected():
    with pytest.raises(AssertionError, match="no SSL listener"):
        render(
            namespace="ocr-lab",
            **{"secrets.celeryBrokerUrl": "amqps://test-rabbitmq.ocr-lab.svc.cluster.local:5671//"},
        )
    with pytest.raises(AssertionError, match="no SSL listener"):
        render(**{
            "clusterDomain": "example.internal",
            "secrets.celeryBrokerUrl": "amqps://test-rabbitmq.default.svc.example.internal:5671//",
        })


def test_plaintext_bundled_broker_override_is_preserved():
    endpoint = "amqp://ocr:render-test-only@test-rabbitmq.default.svc.cluster.local:5672//"
    docs = render(**{"secrets.celeryBrokerUrl": endpoint})
    assert _secret_value(docs, "CELERY_BROKER_URL") == endpoint
    ports = {port["containerPort"] for port in docs["test-rabbitmq"]["spec"]["template"]["spec"]["containers"][0]["ports"]}
    assert 5672 in ports
    assert 5671 not in ports
    assert "CELERY_BROKER_USE_SSL" not in docs["test-config"]["data"]


def test_explicit_external_amqps_is_preserved_with_bundled_plaintext_broker():
    endpoint = "amqps://external-broker.example:5671//"
    docs = render(**{
        "rabbitmq.clientTls.enabled": True,
        "rabbitmq.clientTls.caSecretName": "external-broker-ca",
        "secrets.celeryBrokerUrl": endpoint,
    })
    assert _secret_value(docs, "CELERY_BROKER_URL") == endpoint
    assert docs["test-config"]["data"]["CELERY_BROKER_USE_SSL"] == "true"
    ports = {port["containerPort"] for port in docs["test-rabbitmq"]["spec"]["template"]["spec"]["containers"][0]["ports"]}
    assert ports >= {5672}
    assert 5671 not in ports


def test_tls_enabled_generated_endpoints_match_listeners():
    docs = render(**{"tls.enabled": True})
    assert _secret_value(docs, "CELERY_BROKER_URL").startswith("amqps://")
    assert ":5671//" in _secret_value(docs, "CELERY_BROKER_URL")
    assert _secret_value(docs, "REDIS_URL").startswith("rediss://")
    assert _secret_value(docs, "REDIS_URL").endswith(":6380/0")
    assert _secret_value(docs, "CELERY_RESULT_BACKEND").endswith(":6380/1")
    rabbit_ports = {port["containerPort"] for port in docs["test-rabbitmq"]["spec"]["template"]["spec"]["containers"][0]["ports"]}
    redis_ports = {port["containerPort"] for port in docs["test-redis"]["spec"]["template"]["spec"]["containers"][0]["ports"]}
    assert 5671 in rabbit_ports and 5672 not in rabbit_ports
    assert 6380 in redis_ports and 6379 not in redis_ports
    assert "ssl=on" in docs["test-postgres"]["spec"]["template"]["spec"]["containers"][0]["args"]
    assert docs["test-config"]["data"]["DATABASE_SSLMODE"] == "verify-full"


def test_redis_client_tls_rejects_generated_and_partial_endpoints():
    with pytest.raises(AssertionError, match="partial override or generated endpoint"):
        render(**{
            "redis.clientTls.enabled": True,
            "redis.clientTls.caSecretName": "redis-ca",
        })
    with pytest.raises(AssertionError, match="partial override or generated endpoint"):
        render(**{
            "redis.clientTls.enabled": True,
            "redis.clientTls.caSecretName": "redis-ca",
            "secrets.redisUrl": "rediss://:secret@cache.example:6380/0",
        })
    with pytest.raises(AssertionError, match="partial override or generated endpoint"):
        render(**{
            "redis.clientTls.enabled": True,
            "redis.clientTls.caSecretName": "redis-ca",
            "secrets.celeryResultBackend": "rediss://:secret@results.example:6380/1",
        })


def test_redis_client_tls_rejects_bundled_aliases():
    with pytest.raises(AssertionError, match="no TLS listener"):
        render(**{
            "redis.clientTls.enabled": True,
            "redis.clientTls.caSecretName": "redis-ca",
            "secrets.redisUrl": "rediss://:secret@test-redis:6380/0",
            "secrets.celeryResultBackend": "rediss://:secret@results.example:6380/1",
        })
    with pytest.raises(AssertionError, match="no TLS listener"):
        render(**{
            "redis.clientTls.enabled": True,
            "redis.clientTls.caSecretName": "redis-ca",
            "secrets.redisUrl": "rediss://:secret@cache.example:6380/0",
            "secrets.celeryResultBackend": "sentinel://:secret@test-redis-sentinel-0.test-redis-sentinel.default.svc.cluster.local:26379/1",
        })


def test_external_redis_client_tls_preserves_cache_and_result_backend():
    cache = "rediss://:secret@cache.example:6380/0"
    result = "rediss://:secret@results.example:6380/1"
    docs = render(**{
        "redis.clientTls.enabled": True,
        "redis.clientTls.caSecretName": "redis-ca",
        "secrets.redisUrl": cache,
        "secrets.celeryResultBackend": result,
    })
    assert _secret_value(docs, "REDIS_URL") == cache
    assert _secret_value(docs, "CELERY_RESULT_BACKEND") == result
    assert docs["test-config"]["data"]["REDIS_SSL"] == "true"
    ports = {port["containerPort"] for port in docs["test-redis"]["spec"]["template"]["spec"]["containers"][0]["ports"]}
    assert 6379 in ports and 6380 not in ports


def test_external_sentinel_result_backend_is_preserved_with_client_tls():
    cache = "rediss://:secret@cache.example:6380/0"
    result = "sentinel://:secret@sentinel.example:26379/1;sentinel://:secret@sentinel-b.example:26379/1"
    docs = render(**{
        "redis.sentinel.enabled": True,
        "redis.clientTls.enabled": True,
        "redis.clientTls.caSecretName": "redis-ca",
        "secrets.redisUrl": cache,
        "secrets.celeryResultBackend": result,
    })
    assert _secret_value(docs, "REDIS_URL") == cache
    assert _secret_value(docs, "CELERY_RESULT_BACKEND") == result
    sentinel = docs["test-redis-sentinel"]["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    assert "6379" in sentinel
    assert "6380" not in sentinel


def test_bundled_sentinel_cannot_combine_with_tls_listener():
    with pytest.raises(AssertionError, match="cannot be combined with tls.enabled"):
        render(**{"tls.enabled": True, "redis.sentinel.enabled": True})


def test_plaintext_redis_override_is_preserved():
    endpoint = "redis://:render-test-only@test-redis:6379/0"
    docs = render(**{"secrets.redisUrl": endpoint})
    assert _secret_value(docs, "REDIS_URL") == endpoint
    assert "REDIS_SSL" not in docs["test-config"]["data"]


def test_postgres_client_tls_rejects_bundled_non_tls_server():
    with pytest.raises(AssertionError, match="no ssl=on listener"):
        render(**{
            "postgresql.clientTls.enabled": True,
            "postgresql.clientTls.caSecretName": "db-ca",
        })
    with pytest.raises(AssertionError, match="bundled chart alias"):
        render(**{
            "postgresql.clientTls.enabled": True,
            "postgresql.clientTls.caSecretName": "db-ca",
            "secrets.databaseUrl": "postgres://ocr:pw@test-postgres.default.svc.cluster.local:5432/ocr_coordinator",
        })


def test_external_database_url_keeps_verify_full_off_bundled_plaintext_server():
    endpoint = "postgres://ocr:pw@external-db.example:5432/ocr_coordinator"
    docs = render(**{
        "postgresql.clientTls.enabled": True,
        "postgresql.clientTls.caSecretName": "external-db-ca",
        "secrets.databaseUrl": endpoint,
    })
    assert _secret_value(docs, "DATABASE_URL") == endpoint
    assert docs["test-config"]["data"]["DATABASE_SSLMODE"] == "verify-full"
    assert docs["test-config"]["data"]["DATABASE_SSLROOTCERT"] == "/etc/postgresql-ca/ca.crt"
    args = docs["test-postgres"]["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "ssl=on" not in args

def test_frontend_runtime_environment_is_rendered():
    docs = render(**{"frontend.apiBaseUrl": "/ocr", "frontend.httpOnlySession": True})
    env = {item["name"]: item["value"] for item in docs["test-frontend"]["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["OCR_FRONTEND_API_BASE_URL"] == "/ocr"
    assert env["OCR_FRONTEND_HTTPONLY_SESSION"] == "true"


def test_api_sentinel_and_s3_network_rules_remain_valid():
    docs = render(**{
        "redis.sentinel.enabled": True,
        "storage.backend": "s3",
        "secrets.s3Endpoint": "https://s3.test.invalid",
        "secrets.s3Bucket": "render-test",
        "secrets.s3AccessKey": "render-test-only",
        "secrets.s3SecretKey": "render-test-only",
        "storage.s3.networkPolicy.allowedCidrs[0]": "192.0.2.0/24",
    })
    rules = docs["test-api-netpol"]["spec"]["egress"]
    sentinel = next(rule for rule in rules if rule.get("ports") == [{"protocol": "TCP", "port": 26379}])
    assert sentinel["to"][0]["podSelector"]["matchLabels"]["app.kubernetes.io/component"] == "redis-sentinel"
    assert any(rule.get("to") == [{"ipBlock": {"cidr": "192.0.2.0/24"}}] for rule in rules)
