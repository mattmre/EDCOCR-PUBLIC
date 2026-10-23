"""Rendered Helm schema ordering and bounded migration startup contracts."""

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

CHART = Path(__file__).resolve().parent.parent / "helm" / "ocr-local"
SECRETS = {
    "djangoSecretKey": "test-django",
    "apiKey": "test-api",
    "coordinatorApiKey": "test-coordinator",
    "postgresPassword": "test-db",
    "webhookSecretKey": "test-webhook",
    "piiEncryptionKey": "A" * 43 + "=",
    "celeryTaskSigningKey": "test-celery",
    "rabbitmqPassword": "test-broker",
    "redisPassword": "test-redis",
    "flowerPassword": "test-flower",
    "metricsApiKey": "test-metrics",
}
BASE_CONSUMERS = {
    "ocr-local-celery-beat",
    "ocr-local-celery-coordinator",
    "ocr-local-cpu-worker",
    "ocr-local-gpu-worker",
}
OPTIONAL_CONSUMERS = {
    "ocr-local-cpu-ocr-worker",
    "ocr-local-layout-cpu-worker",
    "ocr-local-layoutlm-worker",
    "ocr-local-nlp-gpu-worker",
    "ocr-local-tesseract-worker",
}
STAGED_CONSUMERS = {"ocr-local-ocr-render", "ocr-local-ocr-finalize"}

pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm unavailable")


def _render(*overrides, values_file=None):
    command = ["helm", "template", "ocr-local", str(CHART)]
    if values_file:
        command.extend(["-f", str(CHART / values_file)])
    for name, value in SECRETS.items():
        command.extend(["--set-string", f"secrets.{name}={value}"])
    for override in overrides:
        command.extend(["--set", override])
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    return {
        document["metadata"]["name"]: document
        for document in yaml.safe_load_all(result.stdout)
        if document and document.get("kind") == "Deployment"
    }


def _pod(deployment):
    return deployment["spec"]["template"]["spec"]


def _single_init(deployment):
    (init,) = _pod(deployment)["initContainers"]
    return init


@pytest.mark.parametrize(
    ("values_file", "overrides", "expected"),
    [
        (None, (), BASE_CONSUMERS),
        (
            "values-staged-spark-16.yaml", (),
            BASE_CONSUMERS | {"ocr-local-tesseract-worker"} | STAGED_CONSUMERS,
        ),
        (
            "values-staged-h100-32.yaml", (),
            BASE_CONSUMERS | {"ocr-local-tesseract-worker"} | STAGED_CONSUMERS,
        ),
        (
            None,
            (
                "cpuOcrWorker.enabled=true",
                "layoutCpuWorker.enabled=true",
                "layoutlmWorker.enabled=true",
                "nlpGpuWorker.enabled=true",
                "tesseractWorker.enabled=true",
                "postgresql.config.maxConnections=500",
            ),
            BASE_CONSUMERS | OPTIONAL_CONSUMERS,
        ),
    ],
)
def test_every_rendered_django_consumer_waits_for_read_only_schema(
    values_file, overrides, expected
):
    deployments = _render(*overrides, values_file=values_file)
    coordinator = deployments["ocr-local-coordinator"]
    assert coordinator["spec"]["replicas"] >= 2
    writer = _single_init(coordinator)
    assert writer["name"] == "migrate"
    assert writer["command"] == ["python", "-c"]
    assert writer["image"] == _pod(coordinator)["containers"][0]["image"]
    writer_script = writer["args"][0]
    assert "pg_try_advisory_lock" in writer_script
    assert "pg_advisory_unlock" in writer_script
    assert 'call_command("migrate", interactive=False)' in writer_script
    assert "deadline = time.monotonic() + 300" in writer_script

    assert expected <= deployments.keys()
    for name in expected:
        deployment = deployments[name]
        gate = _single_init(deployment)
        assert gate["name"] == "wait-for-coordinator-schema", name
        assert gate["command"] == ["python", "-c"], name
        assert gate["image"] == writer["image"], name
        script = gate["args"][0]
        assert '"migrate", "--check", "--noinput"' in script, name
        assert "subprocess.run(" in script and "timeout=min(10, remaining)" in script, name
        assert "deadline = time.monotonic() + 300" in script, name
        assert "call_command" not in script and "pg_try_advisory_lock" not in script, name
        assert gate["securityContext"] == _pod(deployment)["containers"][0]["securityContext"], name
        assert {"configMapRef": {"name": "ocr-local-config"}} in gate["envFrom"], name
        assert any(
            entry["name"] == "DATABASE_URL" and "secretKeyRef" in entry["valueFrom"]
            for entry in gate["env"]
        ), name

    assert not _pod(deployments["ocr-local-api"]).get("initContainers")
    for name in expected - {"ocr-local-gpu-worker"}:
        command = _pod(deployments[name])["containers"][0].get("command", [])
        if "celery" in command:
            assert command[:3] == ["/usr/bin/tini", "-g", "--"], name


def test_tls_mount_and_credentials_reach_readiness_gate_and_writer():
    deployments = _render(
        "cpuOcrWorker.enabled=true",
        "layoutCpuWorker.enabled=true",
        "layoutlmWorker.enabled=true",
        "nlpGpuWorker.enabled=true",
        "tesseractWorker.enabled=true",
        "postgresql.enabled=false",
        "secrets.databaseUrl=postgres://ocr:test-db@external-db.example:5432/ocr_coordinator",
        "postgresql.clientTls.enabled=true",
        "postgresql.clientTls.caSecretName=test-db-ca",
    )
    for name in BASE_CONSUMERS | OPTIONAL_CONSUMERS | {"ocr-local-coordinator"}:
        deployment = deployments[name]
        init = _single_init(deployment)
        pod = _pod(deployment)
        assert "postgresql-client-ca" in {m["name"] for m in init["volumeMounts"]}, name
        assert "postgresql-client-ca" in {v["name"] for v in pod["volumes"]}, name
        assert init["envFrom"] == pod["containers"][0]["envFrom"], name
        assert {e["name"] for e in init["env"]} <= {
            e["name"] for e in pod["containers"][0]["env"]
        }, name


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.mark.parametrize("mode", ["success", "stale", "hung"])
def test_read_only_gate_has_total_and_per_check_deadlines(monkeypatch, mode):
    script = _single_init(_render()["ocr-local-cpu-worker"])["args"][0]
    clock = _Clock()
    checks = []

    def fake_run(command, **kwargs):
        checks.append((command, kwargs))
        assert command[1:] == ["manage.py", "migrate", "--check", "--noinput"]
        assert kwargs["timeout"] <= 10
        assert kwargs["timeout"] > 0
        if mode == "hung":
            clock.now += kwargs["timeout"]
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if mode == "success" and len(checks) == 3:
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("time.monotonic", clock.monotonic)
    monkeypatch.setattr("time.sleep", clock.sleep)
    with pytest.raises(SystemExit) as result:
        exec(compile(script, "helm-schema-readiness", "exec"), {})
    assert result.value.code == (0 if mode == "success" else 1)
    assert len(checks) == (3 if mode == "success" else (20 if mode == "hung" else 60))
    assert clock.now <= 300


def test_writer_script_holds_distinct_session_through_migration_and_releases(monkeypatch):
    script = _single_init(_render()["ocr-local-coordinator"])["args"][0]
    events = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def execute(self, query, params):
            assert len(params) == 1 and isinstance(params[0], int)
            events.append(query.split("(")[0])

        def fetchone(self):
            return (True,)

    class Connection:
        def __init__(self):
            self.settings_dict = {"OPTIONS": {}}

        def copy(self, alias):
            assert alias == "migration_lock"
            events.append("copy")
            return self

        def cursor(self):
            return Cursor()

        def close(self):
            events.append("close")

    connection = Connection()
    django = types.ModuleType("django")
    django.setup = lambda: events.append("setup")
    conf = types.ModuleType("django.conf")
    conf.settings = types.SimpleNamespace(
        DATABASES={"default": {"NAME": "test-db"}}, DATABASE_SCHEMA="public"
    )
    management = types.ModuleType("django.core.management")

    def migrate(name, *, interactive):
        assert name == "migrate" and interactive is False
        assert events[-1] == "SELECT pg_try_advisory_lock"
        events.append("migrate")

    management.call_command = migrate
    db = types.ModuleType("django.db")
    db.OperationalError = type("OperationalError", (Exception,), {})
    db.connections = {"default": connection}
    modules = {
        "django": django,
        "django.conf": conf,
        "django.core": types.ModuleType("django.core"),
        "django.core.management": management,
        "django.db": db,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    exec(compile(script, "helm-serialized-migrate", "exec"), {})
    assert events == [
        "setup", "copy", "SELECT pg_try_advisory_lock", "migrate",
        "SELECT pg_advisory_unlock", "close",
    ]
    assert connection.settings_dict["OPTIONS"]["connect_timeout"] == 10
    assert "statement_timeout=" in connection.settings_dict["OPTIONS"]["options"]


def test_writer_never_migrates_when_lock_unavailable(monkeypatch):
    script = _single_init(_render()["ocr-local-coordinator"])["args"][0]
    clock = _Clock()
    calls = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def execute(self, query, _params):
            calls.append(query)

        def fetchone(self):
            return (False,)

    class Connection:
        def __init__(self):
            self.settings_dict = {"OPTIONS": {}}

        def copy(self, alias):
            assert alias == "migration_lock"
            return Connection()

        def cursor(self):
            return Cursor()

        def close(self):
            calls.append("close")

    django = types.ModuleType("django")
    django.setup = lambda: None
    conf = types.ModuleType("django.conf")
    conf.settings = types.SimpleNamespace(
        DATABASES={"default": {"NAME": "test-db"}}, DATABASE_SCHEMA="public"
    )
    management = types.ModuleType("django.core.management")

    def forbidden_migrate(*_args, **_kwargs):
        pytest.fail("a contender without the advisory lock must not migrate")

    management.call_command = forbidden_migrate
    db = types.ModuleType("django.db")
    db.OperationalError = type("OperationalError", (Exception,), {})
    db.connections = {"default": Connection()}
    for name, module in {
        "django": django,
        "django.conf": conf,
        "django.core": types.ModuleType("django.core"),
        "django.core.management": management,
        "django.db": db,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("time.monotonic", clock.monotonic)
    monkeypatch.setattr("time.sleep", clock.sleep)
    with pytest.raises(SystemExit) as result:
        exec(compile(script, "helm-serialized-migrate-contender", "exec"), {})
    assert result.value.code == 1
    assert calls.count("SELECT pg_try_advisory_lock(%s)") == 60
    assert calls.count("close") == 60
    assert not any("pg_advisory_unlock" in call for call in calls)
    assert clock.now == 300
