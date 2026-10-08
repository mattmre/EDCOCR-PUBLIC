"""Worker CPU-runtime alignment contract (WP-E).

PR #971 qualified the ROOT CPU image on native Paddle (python:3.12-slim-bookworm,
paddlepaddle==3.0.0 + paddleocr==2.9.1, native model preload, onnxruntime removed)
and propagated the HOME/cache/preload changes into the worker Dockerfiles. These
tests lock the deployment configs to the qualified native-Paddle recipe so the
shipped CPU-worker path does not silently select the now-dead ONNX backend or ship
an OCR worker image with no baked models.

The ONNX code path in ocr_local/infra/ocr_inference_backend.py stays as an explicit
opt-in and is intentionally NOT covered here.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ROOT_REQUIREMENTS = ROOT / "requirements.txt"
COMPOSE_CPU_ONLY = ROOT / "coordinator" / "docker-compose.cpu-only.yml"
COMPOSE_SOLO = ROOT / "docker-compose.solo.yml"
HELM_VALUES = ROOT / "helm" / "ocr-local" / "values.yaml"
GPU_DOCKERFILE = ROOT / "coordinator" / "Dockerfile.worker.gpu"
GPU_COMPOSE_FILES = (
    ROOT / "docker-compose.yml",
    ROOT / "coordinator" / "docker-compose.worker.yml",
    ROOT / "coordinator" / "docker-compose.multi-gpu.yml",
)
CPU_COMPOSE_FILES = (
    ROOT / "coordinator" / "docker-compose.cpu-only.yml",
    ROOT / "coordinator" / "docker-compose.worker.cpu.yml",
)
DJANGO_WORKER_COMPOSE_FILES = (
    ROOT / "coordinator" / "docker-compose.coordinator.yml",
    ROOT / "coordinator" / "docker-compose.worker.yml",
    ROOT / "coordinator" / "docker-compose.worker.cpu.yml",
    ROOT / "coordinator" / "docker-compose.multi-gpu.yml",
    ROOT / "coordinator" / "docker-compose.cpu-only.yml",
    ROOT / "coordinator" / "docker-compose.mock-prod-cp.yml",
    ROOT / "coordinator" / "docker-compose.ha.yml",
    ROOT / "coordinator" / "docker-compose.scale-test.yml",
    ROOT / "coordinator" / "docker-compose.isolation-poc.yml",
)

WORKER_DOCKERFILES = {
    "coordinator/Dockerfile.worker": ROOT / "coordinator" / "Dockerfile.worker",
    "coordinator/Dockerfile.worker.ocr": ROOT / "coordinator" / "Dockerfile.worker.ocr",
    "coordinator/Dockerfile.worker.nlp": ROOT / "coordinator" / "Dockerfile.worker.nlp",
}
HEALTHCHECK_DOCKERFILES = (
    *WORKER_DOCKERFILES.values(),
    ROOT / "coordinator" / "Dockerfile.worker.layoutlm",
    GPU_DOCKERFILE,
)


def _skip_model_preload_default(content: str) -> str:
    """Return the default value of the ARG SKIP_MODEL_PRELOAD build arg."""
    match = re.search(r"^ARG\s+SKIP_MODEL_PRELOAD=(\S+)", content, re.MULTILINE)
    assert match is not None, "ARG SKIP_MODEL_PRELOAD not declared"
    return match.group(1)


def _compose_backend_values(content: str) -> list[str]:
    """Return every OCR_INFERENCE_BACKEND value assigned in the compose file."""
    values: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.match(r"OCR_INFERENCE_BACKEND:\s*(\S+)", stripped)
        if match:
            values.append(match.group(1).strip().strip('"').strip("'"))
    return values


def _requirement_names(path: Path) -> list[str]:
    """Return the normalised package name of every requirement line."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)", line)
        if match:
            names.append(match.group(1).lower().replace("_", "-"))
    return names


# ---------------------------------------------------------------------------
# Worker Dockerfiles: base image, no onnxruntime, root requirements install
# ---------------------------------------------------------------------------


def test_worker_dockerfiles_pin_python312_bookworm():
    """Workers use a Python version compatible with the pinned Django 6 runtime."""
    for relative, path in WORKER_DOCKERFILES.items():
        content = path.read_text(encoding="utf-8")
        assert "FROM python:3.12-slim-bookworm" in content, relative
        assert "FROM python:3.11-slim" not in content, relative


def test_worker_dockerfiles_have_no_onnxruntime():
    """No worker image installs onnxruntime; native Paddle is the CPU runtime."""
    for relative, path in WORKER_DOCKERFILES.items():
        content = path.read_text(encoding="utf-8")
        assert "onnxruntime" not in content, relative


def test_worker_dockerfiles_install_root_requirements():
    """Each worker installs the root requirements.txt so pins flow through."""
    for relative, path in WORKER_DOCKERFILES.items():
        content = path.read_text(encoding="utf-8")
        assert "COPY requirements.txt /app/ocr-requirements.txt" in content, relative
        assert "/app/ocr-requirements.txt" in content, relative
        assert "pip install --no-cache-dir" in content, relative


def test_all_celery_worker_images_require_source_identity():
    paths = [*HEALTHCHECK_DOCKERFILES]
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert "ARG SOURCE_REVISION" in content, path
        assert "ARG SOURCE_TREE_SHA256" in content, path
        assert 'io.ocr_local.runtime-profile="worker"' in content, path
        assert "grep -Eq '^[0-9a-f]{40,64}$'" in content, path
        assert "grep -Eq '^[0-9a-f]{64}$'" in content, path
        assert "scripts/with_runtime_source.sh worker" in content, path


def test_paddle_worker_images_disable_vulnerable_imgaug_background_class():
    for path in (*WORKER_DOCKERFILES.values(), GPU_DOCKERFILE):
        content = path.read_text(encoding="utf-8")
        assert "COPY scripts/harden_imgaug_runtime.py" in content, path
        assert "python3 /tmp/harden_imgaug_runtime.py" in content, path


# ---------------------------------------------------------------------------
# Deployment configs: backend must be native paddle, never onnx
# ---------------------------------------------------------------------------


def test_compose_cpu_only_selects_paddle_backend():
    """docker-compose.cpu-only.yml must select native Paddle, not the dead ONNX path."""
    content = COMPOSE_CPU_ONLY.read_text(encoding="utf-8")
    values = _compose_backend_values(content)
    assert values, "OCR_INFERENCE_BACKEND not set in docker-compose.cpu-only.yml"
    assert "onnx" not in values, values
    assert "paddle" in values, values


def test_helm_cpu_ocr_worker_selects_paddle_backend():
    """helm values cpuOcrWorker env must select native Paddle, not ONNX."""
    with open(HELM_VALUES, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f)
    env = values["cpuOcrWorker"]["env"]
    assert env["OCR_INFERENCE_BACKEND"] == "paddle", env


# ---------------------------------------------------------------------------
# Model preload defaults: OCR-capable workers bake models; NLP-only does not
# ---------------------------------------------------------------------------


def test_ocr_capable_workers_default_to_baking_models():
    """Dockerfile.worker and Dockerfile.worker.ocr must preload models by default."""
    for relative in (
        "coordinator/Dockerfile.worker",
        "coordinator/Dockerfile.worker.ocr",
    ):
        content = WORKER_DOCKERFILES[relative].read_text(encoding="utf-8")
        assert _skip_model_preload_default(content) == "0", relative


def test_nlp_only_worker_skips_model_preload_by_default():
    """The NLP-only image has no OCR models to bake, so it keeps the skip default."""
    content = WORKER_DOCKERFILES["coordinator/Dockerfile.worker.nlp"].read_text(
        encoding="utf-8"
    )
    assert _skip_model_preload_default(content) == "1"


# ---------------------------------------------------------------------------
# Root requirements pins flow through to workers
# ---------------------------------------------------------------------------


def test_root_requirements_pin_paddle_and_omit_onnxruntime():
    """Workers install root requirements.txt: paddlepaddle pinned, no onnxruntime."""
    content = ROOT_REQUIREMENTS.read_text(encoding="utf-8")
    assert (
        "paddlepaddle==3.0.0" in "".join(content.split())
        or "paddlepaddle==3.0.0" in content
    )
    assert "onnxruntime" not in _requirement_names(ROOT_REQUIREMENTS)


# ---------------------------------------------------------------------------
# GPU workers: CUDA runtime and GPU wheel must be explicit
# ---------------------------------------------------------------------------


def test_gpu_worker_pins_cuda_runtime_and_gpu_paddle_wheel():
    content = GPU_DOCKERFILE.read_text(encoding="utf-8")

    assert "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:8aef630a" in content
    assert 'if [ "${TARGETARCH}" != "amd64" ]' in content
    assert "paddlepaddle-gpu==3.0.0" in content
    assert "sys.version_info >= (3, 12)" in content
    assert "pip install --no-deps" in content
    assert "packages/stable/cu126/" in content
    assert "gpu_worker_entrypoint.sh" in content
    assert "ARG SKIP_MODEL_PRELOAD=0" in content


def test_gpu_compose_topologies_use_gpu_worker_recipe():
    for path in GPU_COMPOSE_FILES:
        content = path.read_text(encoding="utf-8")
        assert "dockerfile: coordinator/Dockerfile.worker.gpu" in content, path
        assert "SOURCE_TREE_SHA256: ${SOURCE_TREE_SHA256:-unknown}" in content, path
    generator = (ROOT / "scripts" / "generate_multi_gpu_compose.py").read_text(
        encoding="utf-8"
    )
    assert "dockerfile: coordinator/Dockerfile.worker.gpu" in generator


def test_cpu_worker_compose_builds_receive_worker_source_identity():
    for path in CPU_COMPOSE_FILES:
        content = path.read_text(encoding="utf-8")
        assert "dockerfile: coordinator/Dockerfile.worker" in content, path
        assert "SOURCE_REVISION: ${SOURCE_REVISION:-unknown}" in content, path
        assert "SOURCE_TREE_SHA256: ${SOURCE_TREE_SHA256:-unknown}" in content, path


def test_worker_healthchecks_resolve_the_configured_celery_nodename():
    for path in HEALTHCHECK_DOCKERFILES:
        content = path.read_text(encoding="utf-8")
        assert "python3 /app/coordinator/worker_healthcheck.py" in content, path
        assert "-d celery@$(hostname)" not in content, path


def test_helm_workers_use_exact_local_healthcheck_and_gpu_entrypoint():
    templates = ROOT / "helm" / "ocr-local" / "templates"
    worker_templates = sorted(templates.glob("*worker-deployment.yaml"))
    assert worker_templates
    for path in worker_templates:
        content = path.read_text(encoding="utf-8")
        if path.name == "tesseract-worker-deployment.yaml":
            assert "tesseract-worker@$(hostname)" in content, path
            assert content.count("--hostname=tesseract-worker@%h") == 1, path
            continue
        assert "/app/coordinator/worker_healthcheck.py" in content, path
        assert "\n                - inspect\n" not in content, path

    gpu = (templates / "gpu-worker-deployment.yaml").read_text(encoding="utf-8")
    container = gpu.split("- name: gpu-worker", 1)[1]
    assert "\n          args:\n" in container
    assert "\n          command:\n" not in container.split("volumeMounts:", 1)[0]


def test_solo_healthcheck_follows_monolithic_monitor_heartbeat():
    """SOLO runs ocr_gpu_async.py, so it must not use the Celery-only probe."""
    values = yaml.safe_load(COMPOSE_SOLO.read_text(encoding="utf-8"))
    service = values["services"]["ocr-solo"]

    assert service["command"] == "python3 -u /app/ocr_gpu_async.py"
    assert service["environment"]["HEALTHCHECK_FILE"] == "/tmp/ocr_healthcheck"
    assert service["healthcheck"]["test"] == ["CMD", "sh", "/app/healthcheck.sh"]

    healthcheck = (ROOT / "healthcheck.sh").read_text(encoding="utf-8")
    assert "heartbeat file is missing" in healthcheck
    assert 'if [ ! -f "$HEALTHCHECK_FILE" ]' in healthcheck


def test_default_gpu_compose_overrides_celery_image_healthcheck():
    values = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = values["services"]["ocr-gpu"]
    assert service["command"] == "python3 -u /app/ocr_gpu_async.py"
    assert "HEALTHCHECK_FILE=/tmp/ocr_healthcheck" in service["environment"]
    assert service["healthcheck"]["test"] == ["CMD", "sh", "/app/healthcheck.sh"]


def test_api_image_replaces_gpu_worker_entrypoint_with_init():
    content = (ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["/usr/bin/tini", "-g", "--"]' in content
    assert 'CMD ["python3", "-m", "uvicorn"' in content


def test_every_django_worker_topology_wires_field_and_task_keys():
    required = {
        "CELERY_TASK_SIGNING_KEY",
        "PII_ENCRYPTION_KEY",
        "WEBHOOK_SECRET_KEY",
        "S3_PRESIGNED_TASK_KEY",
    }
    for path in DJANGO_WORKER_COMPOSE_FILES:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        django_services = [
            (name, service)
            for name, service in (values.get("services") or {}).items()
            if "DJANGO_SECRET_KEY" in (service.get("environment") or {})
        ]
        assert django_services, path
        for name, service in django_services:
            env = service["environment"]
            assert required <= set(env), f"{path}:{name} missing {required - set(env)}"
