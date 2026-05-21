# Installation Guide

This guide covers every supported install path for EDCOCR. Pick the section that matches your target environment.

> **Not sure which topology to pick?** Read [`docs/DEPLOYMENT-DECISION-GUIDE.md`](docs/DEPLOYMENT-DECISION-GUIDE.md) first. It walks you through the decision and links back to the right section here.

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Docker (Recommended)](#2-docker-recommended)
3. [Kubernetes via Helm](#3-kubernetes-via-helm)
4. [Bare-Metal Python](#4-bare-metal-python)
5. [Air-Gapped Deployment](#5-air-gapped-deployment)
6. [SDK Installation](#6-sdk-installation)
7. [Post-Install Verification](#7-post-install-verification)
8. [Troubleshooting Install Issues](#8-troubleshooting-install-issues)

---

## 1. System Requirements

### Minimum (Development)

- 8 GB RAM
- 4 CPU cores
- 50 GB free disk
- Linux, macOS, or Windows with WSL2

### Recommended (Production)

- 32 GB RAM (64 GB for heavy NLP)
- 12+ CPU cores
- NVIDIA GPU with 16+ GB VRAM (24 GB recommended)
- 500 GB+ SSD storage
- Linux (Ubuntu 22.04 LTS, Debian 12, RHEL 9, or compatible)

### Software Prerequisites

| Component | Minimum | Notes |
|---|---|---|
| Docker Engine | 24.0+ | For Docker install path |
| docker-compose | v2.20+ | Bundled with modern Docker |
| NVIDIA Container Toolkit | latest | For GPU passthrough |
| Python | 3.10+ | For bare-metal install |
| Helm | 3.13+ | For Kubernetes install |
| kubectl | 1.27+ | For Kubernetes install |
| Poppler | latest | Required by `pdf2image` for PDF rasterization |
| Tesseract OCR | 5.0+ | Optional fallback engine |

---

## 2. Docker (Recommended)

The fastest path to a running EDCOCR instance.

### 2.1 Clone the Repository

```bash
git clone https://github.com/mattmre/EDCOCR-PUBLIC.git
cd EDCOCR-PUBLIC
```

### 2.2 Configure (Optional)

For a default single-host install, no configuration is required. Source documents go in `./ocr_source/`, results land in `./ocr_output/`.

To customize, copy and edit:

```bash
cp .env.example .env
# Edit .env for: API key, language tier, worker counts, storage backend
```

See [`docs/06-CONFIGURATION-REFERENCE.md`](docs/06-CONFIGURATION-REFERENCE.md) for every variable.

### 2.3 Build and Start

```bash
docker compose up -d --build
```

First build downloads PaddleOCR models for the configured language tier (about 3-5 GB for the core tier). Subsequent builds use the cache.

### 2.4 Verify

```bash
docker logs -f ocr_gpu_processor
# Look for: "Monitor heartbeat started" and "Worker N ready"

curl http://localhost:8000/api/v1/health/detailed
# Expect: 200 OK with status: ok
```

### 2.5 Quick CPU-Only Install

If you have no GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml up -d --build
```

This activates ONNX Runtime for 4-7x CPU speedup over the default PaddlePaddle CPU mode.

---

## 3. Kubernetes via Helm

For production deployments with autoscaling, HA, and observability.

### 3.1 Prerequisites

- A Kubernetes cluster (1.27+)
- A storage class supporting ReadWriteOnce (PVCs)
- Optional: KEDA installed for autoscaling
- Optional: cert-manager for TLS
- Optional: an Ingress controller (nginx-ingress, Traefik, etc.)
- Optional: Prometheus + Grafana for observability

### 3.2 Add Required Secrets

The Helm chart does **not** ship secrets. You must provide them.

Create `values-secret.yaml`:

```yaml
secrets:
  djangoSecretKey: "<openssl rand -hex 32>"
  postgresqlPassword: "<strong password>"
  rabbitmqPassword: "<strong password>"
  apiKey: "<API key for client auth>"
  webhookSecretKey: "<openssl rand -hex 32>"
  s3AccessKey: "<S3 access key>"
  s3SecretKey: "<S3 secret key>"
```

### 3.3 Install

```bash
helm install edcocr ./helm/ocr-local \
  --namespace edcocr \
  --create-namespace \
  -f helm/ocr-local/values.yaml \
  -f values-secret.yaml
```

For production overlays:

```bash
helm install edcocr ./helm/ocr-local \
  --namespace edcocr \
  --create-namespace \
  -f helm/ocr-local/values.yaml \
  -f helm/ocr-local/values-production.yaml \
  -f values-secret.yaml
```

### 3.4 HA Overlay (Optional)

Enable Redis Sentinel + RabbitMQ quorum queues + PostgreSQL backup CronJob:

```bash
helm upgrade edcocr ./helm/ocr-local \
  --reuse-values \
  --set redis.sentinel.enabled=true \
  --set rabbitmq.quorumQueues=true \
  --set postgresql.backup.enabled=true
```

### 3.5 Monitoring Overlay (Optional)

```bash
helm upgrade edcocr ./helm/ocr-local \
  --reuse-values \
  --set monitoring.serviceMonitor.enabled=true \
  --set monitoring.prometheusRule.enabled=true \
  --set monitoring.grafanaDashboard.enabled=true
```

### 3.6 Verify

```bash
kubectl get pods -n edcocr
kubectl logs -n edcocr -l app=edcocr-coordinator -f
kubectl port-forward -n edcocr svc/edcocr-coordinator 8000:8000
curl http://localhost:8000/api/v1/health/detailed
```

---

## 4. Bare-Metal Python

For developer workstations and CI environments.

### 4.1 System Dependencies

```bash
# Debian / Ubuntu
sudo apt-get install -y \
  python3.11 python3.11-venv python3.11-dev \
  poppler-utils tesseract-ocr \
  ghostscript libgl1 libglib2.0-0
```

```bash
# macOS (Homebrew)
brew install python@3.11 poppler tesseract ghostscript
```

```powershell
# Windows (Scoop)
scoop install python311 poppler tesseract ghostscript
```

### 4.2 Python Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3 Pre-Download Language Models

```bash
python download_models.py
# Use --cpu-only for CPU-only environments
# Use OCR_LANGUAGE_TIERS=core,extended for 45-language support
```

### 4.4 Run

```bash
# Sync pipeline (legacy, not recommended)
python OCR_GPU.py

# Production async pipeline
python ocr_gpu_async.py

# REST API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Coordinator (in a separate terminal)
cd coordinator && python manage.py runserver 0.0.0.0:8001
```

---

## 5. Air-Gapped Deployment

For environments without outbound network access.

### 5.1 Bundle (Connected Environment)

```bash
./scripts/airgap-bundle.sh
# Produces: edcocr-airgap-<version>.tar.gz
```

This bundles:
- All Docker images (coordinator, GPU worker, CPU worker, NLP worker)
- Pre-downloaded language models
- Helm chart
- Documentation

### 5.2 Transfer

Move the bundle to the air-gapped environment via your approved transfer mechanism (encrypted USB, write-once media, internal artifact registry, etc.).

### 5.3 Deploy (Air-Gapped Environment)

```bash
tar -xzf edcocr-airgap-<version>.tar.gz
cd edcocr-airgap-<version>
./scripts/airgap-deploy.sh
```

### 5.4 Validate

```bash
docker compose ps
# All services should report "healthy"

./scripts/env_preflight.py
# Verifies environment configuration
```

See [`docs/operations/production-cutover-runbook.md`](docs/operations/production-cutover-runbook.md) for the full air-gapped cutover procedure.

---

## 6. SDK Installation

### 6.1 Python SDK

```bash
pip install edcocr-sdk
```

```python
from edcocr_sdk import Client

client = Client(base_url="https://edcocr.example.com", api_key="...")
result = client.submit_and_wait(file="document.pdf")
print(result.text)
```

### 6.2 TypeScript SDK

```bash
npm install @edcocr/sdk
# or
yarn add @edcocr/sdk
# or
pnpm add @edcocr/sdk
```

```typescript
import { Client } from "@edcocr/sdk";

const client = new Client({
  baseUrl: "https://edcocr.example.com",
  apiKey: process.env.EDCOCR_API_KEY!,
});

const result = await client.submitAndWait({ file: "document.pdf" });
console.log(result.text);
```

See [`docs/08-SDK-REFERENCE.md`](docs/08-SDK-REFERENCE.md) for the complete SDK API.

---

## 7. Post-Install Verification

After any install method, run these checks.

### 7.1 Health Endpoint

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health/detailed
```

Expected response includes:
- `status: ok`
- `database: ok`
- `redis: ok`
- `queue: ok`
- `workers: N` (≥ 1)
- `pipeline_version: 4.1.0`

### 7.2 Submit a Test Job

```bash
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "X-API-Key: $API_KEY" \
  -F "file=@tests/fixtures/sample.pdf"
```

Expected: `201 Created` with a job ID.

### 7.3 Watch a Job

```bash
curl -H "X-API-Key: $API_KEY" \
  http://localhost:8000/api/v1/jobs/<job_id>
```

Expected progression: `PENDING` → `RUNNING` → `COMPLETED`.

### 7.4 Check Outputs

```bash
ls -lh ocr_output/EXPORT/PDF/
ls -lh ocr_output/EXPORT/TEXT/
cat ocr_output/custody.jsonl | tail -5
```

### 7.5 Verify Custody Chain

```bash
python scripts/verify_release_state.py --custody-log ocr_output/custody.jsonl
```

Expected: `PASS` with verified event count.

---

## 8. Troubleshooting Install Issues

### "CUDA not available" / "No CUDA-capable device"

```bash
nvidia-smi    # Verify GPU is visible
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If the `docker run` test fails, reinstall NVIDIA Container Toolkit:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### "Poppler not found" on Windows

```powershell
scoop install poppler
# Or download from: https://github.com/oschwartz10612/poppler-windows/releases
# Add bin/ directory to PATH
```

### Model download fails behind a proxy

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
docker compose build --build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTPS_PROXY
```

For fully offline environments, use [Section 5: Air-Gapped Deployment](#5-air-gapped-deployment).

### "Permission denied" on `ocr_temp/` or `ocr_output/`

```bash
sudo chown -R 1000:1000 ocr_source ocr_output ocr_temp
sudo chmod -R 755 ocr_source ocr_output ocr_temp
```

The container runs as UID 1000 by default.

### docker-compose service unhealthy

```bash
docker compose ps               # Identify unhealthy service
docker compose logs <service>   # Read the failure
docker inspect <container>      # Check healthcheck output
```

Most healthcheck failures indicate the service is still initializing. Wait 60-90 seconds and re-check.

### Helm install hangs

```bash
helm install edcocr ./helm/ocr-local --debug --timeout 10m
kubectl get pods -n edcocr -w
kubectl describe pod -n edcocr <pending pod>
```

Common causes: missing PVC storage class, insufficient cluster resources, missing image pull secret.

See [`docs/09-TROUBLESHOOTING.md`](docs/09-TROUBLESHOOTING.md) for a wider catalog of runtime issues.

---

## Next Steps

After a successful install:

- Read [`docs/02-QUICKSTART-5-MINUTE-SUCCESS.md`](docs/02-QUICKSTART-5-MINUTE-SUCCESS.md) for a guided first run.
- Skim [`ARCHITECTURE.md`](ARCHITECTURE.md) to understand what's running where.
- Set up monitoring per [`docs/10-MONITORING-OPERATIONS.md`](docs/10-MONITORING-OPERATIONS.md).
- Review [`docs/security-audit-checklist.md`](docs/security-audit-checklist.md) before exposing the API publicly.
