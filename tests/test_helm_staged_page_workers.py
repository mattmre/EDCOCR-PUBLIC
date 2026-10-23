"""Rendered staged OCR workers keep the established worker security contract."""

import os
import shutil
import subprocess
import sys
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

pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm unavailable")


def render(*overrides, values_file=None):
    command = ["helm", "template", "ocr-local", str(CHART)]
    if values_file is not None:
        command.extend(["-f", str(CHART / values_file)])
    for name, value in SECRETS.items():
        command.extend(["--set-string", f"secrets.{name}={value}"])
    for override in overrides:
        command.extend(["--set", override])
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _deployments(rendered):
    return {
        document["metadata"]["name"]: document
        for document in yaml.safe_load_all(rendered.stdout)
        if document and document.get("kind") == "Deployment"
    }


@pytest.mark.parametrize("values_file", [None, "values-staged-h100-32.yaml"])
def test_django_consumers_wait_for_single_migration_writer(values_file):
    rendered = render(values_file=values_file)
    assert rendered.returncode == 0, rendered.stderr
    deployments = _deployments(rendered)
    expected = {
        "ocr-local-celery-beat", "ocr-local-celery-coordinator",
        "ocr-local-cpu-worker", "ocr-local-gpu-worker",
    }
    if values_file:
        expected |= {
            "ocr-local-ocr-render", "ocr-local-ocr-finalize",
            "ocr-local-tesseract-worker",
        }
    for name in expected:
        gate, = deployments[name]["spec"]["template"]["spec"]["initContainers"]
        assert gate["name"] == "wait-for-coordinator-schema", name
        assert gate["image"] == deployments["ocr-local-coordinator"]["spec"]["template"]["spec"]["containers"][0]["image"]
        assert gate["command"] == ["python", "-c"]
        script, = gate["args"]
        assert '"manage.py", "migrate", "--check", "--noinput"' in script
        assert "deadline = time.monotonic() + 300" in script
        assert "timeout=min(10, remaining)" in script
        assert "DATABASE_URL" in {item["name"] for item in gate["env"]}
        assert {"configMapRef": {"name": "ocr-local-config"}} in gate["envFrom"]
    writer, = deployments["ocr-local-coordinator"]["spec"]["template"]["spec"]["initContainers"]
    assert writer["name"] == "migrate"
    assert writer["command"] == ["python", "-c"]
    writer_script, = writer["args"]
    assert "pg_try_advisory_lock" in writer_script
    assert 'call_command("migrate", interactive=False)' in writer_script
    assert not deployments["ocr-local-api"]["spec"]["template"]["spec"].get("initContainers")


def test_schema_gate_succeeds_after_migration_and_times_out_without_writer(tmp_path):
    rendered = render()
    assert rendered.returncode == 0, rendered.stderr
    gate, = _deployments(rendered)["ocr-local-cpu-worker"]["spec"]["template"]["spec"]["initContainers"]
    script, = gate["args"]
    count_file = tmp_path / "checks"
    manage = tmp_path / "manage.py"
    manage.write_text(
        'import os, pathlib, sys, time\n'
        'path = pathlib.Path(os.environ["GATE_CHECKS"])\n'
        'with path.open("a") as output: output.write(" ".join(sys.argv[1:]) + "\\n")\n'
        'count = len(path.read_text().splitlines())\n'
        'if os.environ.get("GATE_HANG") == "1": time.sleep(10)\n'
        'raise SystemExit(0 if count >= int(os.environ["GATE_SUCCESS_AFTER"]) else 1)\n',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "GATE_CHECKS": str(count_file),
        "GATE_SUCCESS_AFTER": "3",
    }
    fast_script = script.replace("time.sleep(min(5,", "time.sleep(min(0,")
    success = subprocess.run([sys.executable, "-c", fast_script], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    assert count_file.read_text(encoding="utf-8").splitlines() == [
        "migrate --check --noinput",
    ] * 3
    count_file.unlink()
    env["GATE_SUCCESS_AFTER"] = "999"
    env["GATE_HANG"] = "1"
    fast_script = fast_script.replace("time.monotonic() + 300", "time.monotonic() + 0.2")
    timeout = subprocess.run([sys.executable, "-c", fast_script], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert timeout.returncode != 0
    assert "unavailable after 300 seconds" in timeout.stderr
    assert len(count_file.read_text(encoding="utf-8").splitlines()) >= 1


def test_optional_django_workers_share_schema_gate_and_tls_mount():
    rendered = render(
        "cpuOcrWorker.enabled=true", "layoutCpuWorker.enabled=true",
        "nlpGpuWorker.enabled=true", "layoutlmWorker.enabled=true",
        "tesseractWorker.enabled=true",
        "postgresql.enabled=false",
        "secrets.databaseUrl=postgres://ocr:test-db@external-db.example:5432/ocr_coordinator",
        "postgresql.clientTls.enabled=true",
        "postgresql.clientTls.caSecretName=db-ca",
    )
    assert rendered.returncode == 0, rendered.stderr
    deployments = _deployments(rendered)
    for name in (
        "ocr-local-cpu-ocr-worker", "ocr-local-layout-cpu-worker",
        "ocr-local-nlp-gpu-worker", "ocr-local-layoutlm-worker",
        "ocr-local-tesseract-worker",
    ):
        pod = deployments[name]["spec"]["template"]["spec"]
        gate, = pod["initContainers"]
        assert gate["name"] == "wait-for-coordinator-schema", name
        assert "DATABASE_URL" in {item["name"] for item in gate["env"]}
        assert "postgresql-client-ca" in {mount["name"] for mount in gate["volumeMounts"]}
        assert "postgresql-client-ca" in {volume["name"] for volume in pod["volumes"]}


@pytest.mark.parametrize("values_file", ["values-staged-spark-16.yaml", "values-staged-h100-32.yaml"])
def test_rendered_worker_commands_keep_tini_as_pid_one(values_file):
    result = render(values_file=values_file)
    assert result.returncode == 0, result.stderr
    deployments = [
        item for item in yaml.safe_load_all(result.stdout)
        if item and item.get("kind") == "Deployment"
    ]
    checked = set()
    for deployment in deployments:
        name = deployment["metadata"]["name"]
        for container in deployment["spec"]["template"]["spec"]["containers"]:
            command = container.get("command", [])
            if "celery" in command or "gunicorn" in command:
                assert command[:3] == ["/usr/bin/tini", "-g", "--"], name
                checked.add(name)
            if name == "ocr-local-gpu-worker":
                assert not command, "GPU args must retain the image's verified entrypoint"
                assert "celery" in container["args"]
                checked.add(name)
    assert {"ocr-local-ocr-render", "ocr-local-ocr-finalize", "ocr-local-gpu-worker"} <= checked


@pytest.mark.parametrize(
    "override",
    [
        "coordinator.env.OCR_PER_GPU_STRICT=true",
        "coordinator.env.ENABLE_PER_GPU_QUEUES=true",
        "gpuWorker.env.ENABLE_PER_GPU_QUEUES=true",
    ],
)
def test_staged_shared_gpu_consumer_rejects_indexed_producer_topology(override):
    result = render(override, values_file="values-staged-spark-16.yaml")
    assert result.returncode != 0
    assert "explicitly matched consumer topology" in result.stderr


@pytest.mark.parametrize(
    ("values_file", "gpu_replicas", "cpu_replicas", "architecture"),
    [
        ("values-staged-spark-16.yaml", 16, 8, "arm64"),
        ("values-staged-h100-32.yaml", 32, 16, "amd64"),
    ],
)
def test_fleet_profiles_render_distinct_staged_capacity(
    values_file, gpu_replicas, cpu_replicas, architecture
):
    result = render(values_file=values_file)
    assert result.returncode == 0, result.stderr
    documents = [item for item in yaml.safe_load_all(result.stdout) if item]
    deployments = {
        item["metadata"]["name"]: item
        for item in documents if item.get("kind") == "Deployment"
    }
    gpu = deployments["ocr-local-gpu-worker"]
    assert gpu["spec"]["replicas"] == gpu_replicas
    assert gpu["spec"]["template"]["spec"]["nodeSelector"]["kubernetes.io/arch"] == architecture
    assert gpu["spec"]["template"]["spec"]["topologySpreadConstraints"]
    assert "--max-tasks-per-child=512" in gpu["spec"]["template"]["spec"]["containers"][0]["args"]
    for stage in ("render", "finalize"):
        deployment = deployments[f"ocr-local-ocr-{stage}"]
        assert deployment["spec"]["replicas"] == cpu_replicas
    config = next(item for item in documents if item.get("kind") == "ConfigMap" and item["metadata"]["name"] == "ocr-local-config")
    assert config["data"]["OCR_FAIR_FANOUT_ENABLED"] == "true"
    assert int(config["data"]["OCR_MAX_INFLIGHT_PAGES_PER_DOCUMENT"]) >= gpu_replicas


@pytest.mark.parametrize(
    ("values_file", "connections", "memory_request", "memory_limit"),
    [
        ("values-staged-spark-16.yaml", "200", "2Gi", "4Gi"),
        ("values-staged-h100-32.yaml", "320", "4Gi", "8Gi"),
    ],
)
def test_fleet_profiles_render_postgres_connection_and_memory_budget(
    values_file, connections, memory_request, memory_limit
):
    result = render(values_file=values_file)
    assert result.returncode == 0, result.stderr
    postgres = next(
        item for item in yaml.safe_load_all(result.stdout)
        if item and item.get("kind") == "StatefulSet"
        and item["metadata"]["name"] == "ocr-local-postgres"
    )["spec"]["template"]["spec"]["containers"][0]
    assert f"max_connections={connections}" in postgres["args"]
    assert postgres["resources"]["requests"]["memory"] == memory_request
    assert postgres["resources"]["limits"]["memory"] == memory_limit


def test_32_spark_requires_explicit_postgres_budget_override():
    fleet = (
        "gpuWorker.replicas=32",
        "ocrStagedPage.render.replicas=16",
        "ocrStagedPage.finalize.replicas=16",
    )
    undersized = render(*fleet, values_file="values-staged-spark-16.yaml")
    assert undersized.returncode != 0
    assert "staged OCR PostgreSQL connection budget requires maxConnections" in undersized.stderr
    sized = render(
        *fleet,
        "postgresql.config.maxConnections=320",
        "postgresql.resources.requests.memory=4Gi",
        "postgresql.resources.limits.memory=8Gi",
        values_file="values-staged-spark-16.yaml",
    )
    assert sized.returncode == 0, sized.stderr


def test_16_h100_can_use_the_smaller_explicit_postgres_budget():
    result = render(
        "gpuWorker.replicas=16",
        "ocrStagedPage.render.replicas=8",
        "ocrStagedPage.finalize.replicas=8",
        "tesseractWorker.replicas=8",
        "postgresql.config.maxConnections=200",
        "postgresql.resources.requests.memory=2Gi",
        "postgresql.resources.limits.memory=4Gi",
        values_file="values-staged-h100-32.yaml",
    )
    assert result.returncode == 0, result.stderr
    postgres = next(
        item for item in yaml.safe_load_all(result.stdout)
        if item and item.get("kind") == "StatefulSet"
        and item["metadata"]["name"] == "ocr-local-postgres"
    )["spec"]["template"]["spec"]["containers"][0]
    assert "max_connections=200" in postgres["args"]
    assert postgres["resources"]["requests"]["memory"] == "2Gi"
    assert postgres["resources"]["limits"]["memory"] == "4Gi"


def test_autoscaled_peak_replicas_are_included_in_postgres_budget():
    result = render(
        "ocrStagedPage.enabled=true",
        "tesseractWorker.enabled=true",
        "ocrStagedPage.render.autoscaling.enabled=true",
        "ocrStagedPage.render.autoscaling.maxReplicas=40",
        "ocrStagedPage.finalize.autoscaling.enabled=true",
        "ocrStagedPage.finalize.autoscaling.maxReplicas=40",
    )
    assert result.returncode != 0
    assert "staged OCR PostgreSQL connection budget requires maxConnections" in result.stderr


def test_threaded_gpu_task_slots_are_counted_as_potential_db_sessions():
    result = render(
        "ocrStagedPage.enabled=true",
        "tesseractWorker.enabled=true",
        "gpuWorker.replicas=40",
        "gpuWorker.stagedInference.pool=threads",
        "gpuWorker.stagedInference.concurrency=4",
        "gpuWorker.stagedInference.slotsPerDevice=4",
        "gpuWorker.stagedInference.replicasPerKey=4",
        "gpuWorker.stagedInference.deviceMemoryMiB=81920",
        "gpuWorker.stagedInference.modelReplicaMiB=4096",
        "gpuWorker.stagedInference.runtimeReserveMiB=8192",
    )
    assert result.returncode != 0
    assert "staged OCR PostgreSQL connection budget requires maxConnections" in result.stderr
    assert "potential sessions" in result.stderr


def test_staged_workers_require_fallback_consumer():
    result = render("ocrStagedPage.enabled=true")
    assert result.returncode != 0
    assert "requires tesseractWorker.enabled" in result.stderr


def test_staged_workers_keep_security_and_distinct_scalers():
    result = render(
        "ocrStagedPage.enabled=true",
        "tesseractWorker.enabled=true",
        "ocrStagedPage.render.autoscaling.enabled=true",
        "ocrStagedPage.finalize.autoscaling.enabled=true",
        "workerSecurity.stagingVolume.type=pvc",
        "haScheduling.enabled=true",
    )
    assert result.returncode == 0, result.stderr
    documents = [item for item in yaml.safe_load_all(result.stdout) if item]
    deployments = {
        item["metadata"]["name"]: item
        for item in documents if item.get("kind") == "Deployment"
    }
    scalers = {
        item["metadata"]["name"]: item
        for item in documents if item.get("kind") == "ScaledObject"
    }
    for stage in ("render", "finalize"):
        name = f"ocr-local-ocr-{stage}"
        deployment = deployments[name]
        assert "replicas" not in deployment["spec"]
        pod = deployment["spec"]["template"]
        assert pod["metadata"]["annotations"]["sidecar.istio.io/inject"] == "true"
        assert pod["spec"]["securityContext"]["runAsNonRoot"] is True
        assert pod["spec"]["topologySpreadConstraints"]
        assert pod["spec"]["volumes"][0]["ephemeral"]["volumeClaimTemplate"]
        worker = pod["spec"]["containers"][0]
        assert worker["securityContext"]["allowPrivilegeEscalation"] is False
        assert worker["startupProbe"] and worker["readinessProbe"] and worker["livenessProbe"]
        assert f"--queues=ocr_{stage}" in worker["command"]
        assert scalers[f"{name}-scaler"]["spec"]["triggers"][0]["metadata"]["queueName"] == f"ocr_{stage}"


def test_threaded_gpu_profile_has_one_process_and_bounded_owner_slots():
    result = render(
        "ocrStagedPage.enabled=true",
        "tesseractWorker.enabled=true",
        "gpuWorker.stagedInference.pool=threads",
        "gpuWorker.stagedInference.concurrency=4",
        "gpuWorker.stagedInference.slotsPerDevice=4",
        "gpuWorker.stagedInference.replicasPerKey=4",
        "gpuWorker.stagedInference.deviceMemoryMiB=81920",
        "gpuWorker.stagedInference.modelReplicaMiB=4096",
        "gpuWorker.stagedInference.runtimeReserveMiB=8192",
    )
    assert result.returncode == 0, result.stderr
    documents = [item for item in yaml.safe_load_all(result.stdout) if item]
    gpu = next(
        item for item in documents
        if item.get("kind") == "Deployment"
        and item["metadata"]["name"] == "ocr-local-gpu-worker"
    )["spec"]["template"]["spec"]["containers"][0]
    assert "--pool=threads" in gpu["args"]
    assert "--concurrency=4" in gpu["args"]
    assert not any(value.startswith("--max-tasks-per-child") for value in gpu["args"])
    env = {item["name"]: item.get("value") for item in gpu["env"]}
    assert env["OCR_INFERENCE_SLOTS_PER_DEVICE"] == "4"
    assert env["OCR_ENGINE_REPLICAS_PER_KEY"] == "4"


@pytest.mark.parametrize(
    "override",
    [
        "gpuWorker.stagedInference.slotsPerDevice=3",
        "gpuWorker.stagedInference.modelReplicaMiB=30000",
    ],
)
def test_threaded_gpu_profile_rejects_slot_or_memory_overcommit(override):
    result = render(
        "ocrStagedPage.enabled=true",
        "tesseractWorker.enabled=true",
        "gpuWorker.stagedInference.pool=threads",
        "gpuWorker.stagedInference.concurrency=4",
        "gpuWorker.stagedInference.slotsPerDevice=4",
        "gpuWorker.stagedInference.replicasPerKey=4",
        "gpuWorker.stagedInference.deviceMemoryMiB=81920",
        "gpuWorker.stagedInference.modelReplicaMiB=4096",
        "gpuWorker.stagedInference.runtimeReserveMiB=8192",
        override,
    )
    assert result.returncode != 0
    assert "staged threaded GPU profile requires" in result.stderr
