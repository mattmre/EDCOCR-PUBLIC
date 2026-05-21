# Deployment Topology Matrix

This document describes every supported deployment mode for EDCOCR, the
hardware and software each mode requires, the environment variables that must be
set, and how to validate that a host satisfies the prerequisites.

> **Version**: 1.2.0 | **Last Updated**: 2026-05-20

---

## Topology Overview

| ID | Topology | GPU Required | Min RAM | Min VRAM | Docker Images | Primary Compose / Helm | Notes |
|----|----------|-------------|---------|----------|---------------|----------------------|-------|
| T1 | Single-node GPU | Yes | 32 GB | 16 GB | `ocr-gpu-local` | `docker-compose.yml` | Production default for single-server |
| T2 | Single-node CPU | No | 16 GB | -- | `ocr-gpu-local` (ONNX mode) | `docker-compose.yml` (override env) | Set `OCR_TASK_ROUTING=cpu` |
| T3 | Multi-GPU | Yes | 32 GB | N x 16 GB | `ocr-gpu-local` | Generated via `scripts/generate_multi_gpu_compose.py` | `ENABLE_PER_GPU_QUEUES=true` |
| T4 | Distributed (GPU workers) | Yes (workers) | Coord: 8 GB; Workers: 32 GB | 16 GB per worker | coordinator + worker | `coordinator/docker-compose.coordinator.yml` + `coordinator/docker-compose.worker.yml` | Phase M deployment |
| T5 | Distributed (CPU + GPU) | Partial | Varies | Varies | coordinator + ocr + nlp + cpu-ocr | Multiple compose files + `coordinator/docker-compose.isolation-poc.yml` | |
| T6 | Kubernetes | Optional | Cluster-dependent | Cluster-dependent | All images | `helm/ocr-local/` | Production-scale Helm chart |
| T7 | Air-gapped | Optional | Same as base topology | Same as base topology | Pre-bundled via `scripts/airgap-bundle.sh` | Same as base | No internet required |

---

## Detailed Topology Specifications

### T1: Single-node GPU

The default production deployment. A single Docker container runs the full
async pipeline with GPU-accelerated PaddleOCR.

**Prerequisites**
- NVIDIA GPU with >= 16 GB VRAM (24 GB recommended for 12 workers)
- Docker Engine >= 24.0 with `nvidia-container-toolkit` installed
- Host RAM >= 32 GB

**Docker Image**: `ocr-gpu-local` (built from root `Dockerfile`)

**Compose File**: `docker-compose.yml`

**Key Environment Variables**

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `NVIDIA_VISIBLE_DEVICES` | `all` | Yes | GPU device visibility |
| `NUM_WORKERS` | `12` | No | GPU OCR worker threads |
| `NUM_EXTRACTORS` | `8` | No | CPU image extraction threads |
| `NUM_COMPRESSORS` | `8` | No | Ghostscript compression threads |
| `IMAGE_QUEUE_SIZE` | `200` | No | RAM buffer for extracted images |
| `DPI` | `300` | No | Image resolution for OCR |

**Validation Steps**
1. `nvidia-smi` reports at least one GPU with >= 16 GB VRAM
2. `docker compose version` succeeds
3. `docker-compose.yml` exists in project root

---

### T2: Single-node CPU

Runs on hardware without a GPU. OCR is performed via ONNX Runtime or
Tesseract fallback. Throughput is lower but deployment is simpler.

**Prerequisites**
- No GPU required
- Docker Engine >= 24.0
- Host RAM >= 16 GB

**Docker Image**: `ocr-gpu-local` (same image, different runtime config)

**Compose File**: `docker-compose.yml` with environment overrides, or
`coordinator/docker-compose.cpu-only.yml` for distributed CPU-only.

**Key Environment Variables**

| Variable | Required Value | Description |
|----------|---------------|-------------|
| `OCR_TASK_ROUTING` | `cpu` | Route OCR tasks to CPU queue |
| `OCR_INFERENCE_BACKEND` | `onnx` | Use ONNX Runtime instead of PaddleOCR GPU |
| `OCR_ENGINE_SELECTION` | `auto` or `tesseract` | Engine selection strategy |

**Validation Steps**
1. `docker compose version` succeeds
2. `OCR_TASK_ROUTING` is set to `cpu`
3. No GPU device reservation in active compose file

---

### T3: Multi-GPU

Uses one Celery worker container per GPU, each pinned to a specific device
via `CUDA_VISIBLE_DEVICES`. Requires `ENABLE_PER_GPU_QUEUES=true`.

**Prerequisites**
- Multiple NVIDIA GPUs (each >= 16 GB VRAM)
- Docker Engine >= 24.0 with `nvidia-container-toolkit`
- Host RAM >= 32 GB

**Docker Images**: `ocr-gpu-local` (one per GPU)

**Compose File**: Generated via `scripts/generate_multi_gpu_compose.py`

**Key Environment Variables**

| Variable | Required | Description |
|----------|----------|-------------|
| `ENABLE_PER_GPU_QUEUES` | `true` | Enable per-GPU queue affinity |
| `GPU_COUNT` | N | Number of GPUs to use |

**Generation Command**
```bash
python scripts/generate_multi_gpu_compose.py --gpus 4 --output docker-compose.workers.yml
```

**Validation Steps**
1. `nvidia-smi` reports multiple GPUs
2. `ENABLE_PER_GPU_QUEUES=true` is set
3. Generated compose file exists
4. GPU count matches `GPU_COUNT` env var

---

### T4: Distributed (GPU Workers)

Coordinator (Django + PostgreSQL + RabbitMQ + Redis) runs on one node;
GPU workers connect via Celery and process OCR tasks.

**Prerequisites**
- Coordinator: 8 GB RAM, no GPU required
- Workers: 16 GB VRAM each, 32 GB RAM
- Network connectivity between coordinator and workers
- Docker Engine >= 24.0 on all nodes

**Docker Images**: `coordinator` (from `coordinator/Dockerfile.coordinator`) +
`worker` (from `coordinator/Dockerfile.worker`)

**Compose Files**:
- Coordinator: `coordinator/docker-compose.coordinator.yml`
- Worker: `coordinator/docker-compose.worker.yml`

**Key Environment Variables (Coordinator)**

| Variable | Required | Description |
|----------|----------|-------------|
| `DJANGO_SECRET_KEY` | Yes | Django secret key |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password |
| `RABBITMQ_PASSWORD` | Yes | RabbitMQ password |
| `DATABASE_URL` | Yes | PostgreSQL connection URL |
| `CELERY_BROKER_URL` | Yes | RabbitMQ connection URL |

**Key Environment Variables (Worker)**

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection URL |
| `CELERY_BROKER_URL` | Yes | RabbitMQ connection URL |
| `DJANGO_SECRET_KEY` | Yes | Django secret key |
| `NFS_ROOT` | Yes | Shared filesystem mount |

**Validation Steps**
1. Coordinator compose file exists
2. Worker compose file exists
3. All required env vars are set
4. `DATABASE_URL` and `CELERY_BROKER_URL` are valid URLs

---

### T5: Distributed (CPU + GPU Mixed)

separate OCR and NLP workers with optional
CPU-only OCR workers.

**Prerequisites**
- Same as T4, plus CPU-only worker nodes
- At least one GPU worker for OCR processing (or full CPU fleet)

**Docker Images**: coordinator + `Dockerfile.worker.ocr` + `Dockerfile.worker.nlp`

**Additional Compose Files**:
- `coordinator/docker-compose.isolation-poc.yml`
- `coordinator/docker-compose.cpu-only.yml`

**Key Environment Variables (additional)**

| Variable | Required | Description |
|----------|----------|-------------|
| `OCR_TASK_ROUTING` | `auto` or `cpu` | Task routing mode |
| `WORKER_QUEUES` | Varies | Queue assignments per worker type |

**Validation Steps**
1. All T4 validations pass
2. Isolation compose file exists
3. Worker queue assignments are configured

---

### T6: Kubernetes

Production-scale deployment via Helm chart with KEDA autoscaling,
PodDisruptionBudgets, network policies, and optional Redis Sentinel.

**Prerequisites**
- Kubernetes cluster >= 1.27
- `kubectl` configured with cluster access
- `helm` >= 3.12 installed
- GPU nodes with `nvidia.com/gpu` resource (optional)
- KEDA operator installed (for autoscaling, optional)

**Helm Chart**: `helm/ocr-local/`

**Key Values**

| Value Path | Required | Description |
|------------|----------|-------------|
| `secrets.djangoSecretKey` | Yes | Django secret key |
| `secrets.postgresPassword` | Yes | PostgreSQL password |
| `secrets.rabbitmqPassword` | Yes | RabbitMQ password |
| `gpuWorker.enabled` | No | Enable GPU workers |
| `cpuWorker.enabled` | No | Enable CPU workers |
| `redis.sentinel.enabled` | No | Enable Redis Sentinel HA |

**Validation Steps**
1. `kubectl version --client` succeeds
2. `helm version` succeeds
3. `helm/ocr-local/Chart.yaml` exists
4. `helm lint helm/ocr-local/` passes
5. Cluster connectivity works (optional)

---

### T7: Air-gapped

Any of the above topologies deployed without internet access. Docker images
and model files are pre-bundled for offline transfer.

**Prerequisites**
- Same as the base topology
- Images bundled via `scripts/airgap-bundle.sh` on an internet-connected host
- Bundle transferred to target host

**Scripts**:
- Bundle: `scripts/airgap-bundle.sh`
- Deploy: `scripts/airgap-deploy.sh`

**Validation Steps**
1. Base topology validations pass
2. All required Docker images are present locally (`docker images`)
3. PaddleOCR models exist in expected paths
4. FastText model (`lid.176.bin`) exists

---

## Quick Reference: Environment Variable Matrix

| Variable | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|----------|----|----|----|----|----|----|-----|
| `NVIDIA_VISIBLE_DEVICES` | Req | -- | Req | Workers | Partial | Values | Base |
| `OCR_TASK_ROUTING` | -- | `cpu` | -- | -- | `auto`/`cpu` | Values | Base |
| `ENABLE_PER_GPU_QUEUES` | -- | -- | `true` | -- | -- | Values | Base |
| `GPU_COUNT` | -- | -- | Req | -- | -- | Values | Base |
| `DJANGO_SECRET_KEY` | -- | -- | -- | Req | Req | Req | Base |
| `POSTGRES_PASSWORD` | -- | -- | -- | Req | Req | Req | Base |
| `RABBITMQ_PASSWORD` | -- | -- | -- | Req | Req | Req | Base |
| `DATABASE_URL` | -- | -- | -- | Req | Req | Req | Base |
| `CELERY_BROKER_URL` | -- | -- | -- | Req | Req | Req | Base |

Legend: **Req** = Required, **--** = Not applicable, **Base** = Inherits from base topology, **Values** = Set in Helm values.yaml

---

## Automated Validation

Run the topology validation script to check whether your host satisfies a
given deployment mode:

```bash
# Auto-detect topology and validate
python scripts/validate_topology.py

# Validate a specific topology
python scripts/validate_topology.py --topology single-gpu

# Use an env file
python scripts/validate_topology.py --env-file coordinator/.env

# JSON output for CI/CD pipelines
python scripts/validate_topology.py --json

# Include port availability checks
python scripts/validate_topology.py --check-ports

# Write a report file
python scripts/validate_topology.py --report topology-report.md
```

See `scripts/validate_topology.py` for implementation details.
