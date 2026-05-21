# Docker Deployment Guide

## Overview

The OCR pipeline is **production-ready for Docker deployment** with GPU acceleration via NVIDIA Container Toolkit. The deployment strategy uses a **multi-stage Dockerfile** to pre-download OCR models during build time, minimizing runtime startup latency.

The Dockerfile is cold-start safe: it bootstraps from `python:3.10-slim` by default and does not require a pre-existing local OCR image.

---

## Prerequisites

### 1. Docker Engine

**Minimum version**: 20.10+

**Installation**:

#### Linux
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker
```

#### Windows
Download **Docker Desktop** from https://www.docker.com/products/docker-desktop

**Settings**:
- Enable WSL 2 backend (Settings → General → Use WSL 2)
- Allocate resources (Settings → Resources):
  - CPU: 8+ cores recommended
  - Memory: 16+ GB recommended
  - Disk: 50+ GB for models + temp files

### 2. NVIDIA Container Toolkit

**Purpose**: Exposes GPU to Docker containers

**Requirements**:
- NVIDIA GPU with CUDA support (compute capability 6.0+)
- NVIDIA driver: 525+ (for CUDA 12.0)
- Linux host (WSL 2 on Windows)

**Installation**:

#### Linux
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

#### Windows (WSL 2)
```powershell
# In WSL 2 Ubuntu terminal
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker Desktop from Windows
```

**Verification**:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**Expected output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
|  0%   45C    P0    50W / 350W |   1024MiB / 24576MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Docker Desktop / Kubernetes Runtime Blocker

For local release-gate or deployed-stack evidence, Docker and Kubernetes must be live
runtimes, not only render targets. On Windows, first verify:

```powershell
docker version
kubectl get nodes --request-timeout=10s
```

If Docker Desktop's Linux engine returns a pipe 500, or `kubectl get nodes`
times out, record a deployed-stack runtime blocker and do not claim Product E2E
or production/live release credit from Helm rendering, kind manifests, or local-dev
evidence artifacts. Local-dev artifacts that use `environment=local-dev` must be
validated with their explicit `--allow-local-dev` flags and must continue to
fail strict production validation.

### 3. Docker Compose

**Minimum version**: 1.28+ (built-in to Docker Desktop)

**Verification**:
```bash
docker-compose --version
# OR (newer syntax)
docker compose version
```

---

## Repository Structure

```
EDCOCR/
├── Dockerfile                  # Multi-stage build definition
├── docker-compose.yml          # Service orchestration
├── ocr_gpu_async.py            # Main pipeline
├── download_models.py          # Model preloader script
├── run_ocr.bat                 # Windows: Start container
├── run_ocr_async.bat           # Windows: Start async pipeline
├── run_optimization.bat        # Windows: Run Ghostscript compression
│
├── ocr_source/                 # INPUT: Place documents here
│   └── (your PDFs and images)
│
├── ocr_output/                 # OUTPUT: Results appear here
│   ├── EXPORT/
│   │   ├── PDF/
│   │   └── TEXT/
│   └── logs/
│
└── models/                     # Pre-downloaded OCR models (created at build)
    ├── lid.176.bin
    └── paddleocr/
```

---

## Dockerfile Architecture

### Multi-Stage Build Strategy

```dockerfile
# ============================================
# STAGE 1: Model Preloader (CPU-only)
# ============================================
FROM python:3.10-slim AS model-preload

# Install CPU-only PaddlePaddle (smaller, faster build)
RUN pip install paddlepaddle==2.5.1

# Install PaddleOCR (triggers model downloads)
RUN pip install paddleocr==2.7.0

# Download 27+ language models (PaddleOCR only)
COPY download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py

# NOTE: FastText lid.176.bin is not downloaded by download_models.py itself.
# The Dockerfile adds it separately later with an explicit `ADD` instruction.
# See the final image stage for the authoritative build path.

# ============================================
# STAGE 2: GPU Runtime (final image)
# ============================================
FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    # ... (27 language packs)
    ghostscript \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with GPU support
RUN pip install --no-cache-dir \
    paddlepaddle-gpu==2.5.1 \
    paddleocr==2.7.0 \
    pdf2image==1.16.3 \
    pytesseract==0.3.10 \
    pillow==10.0.0 \
    fasttext-wheel==0.9.2 \
    pikepdf==8.10.1

# Copy pre-downloaded models from stage 1
COPY --from=model-preload /root/.paddleocr/ /root/.paddleocr/

# Create working directories
WORKDIR /app
RUN mkdir -p /app/ocr_source /app/ocr_output /app/ocr_temp /app/models

# Copy application code
COPY ocr_gpu_async.py /app/
COPY optimize_pdfs.py /app/

# Default command
CMD ["python3", "/app/ocr_gpu_async.py"]
```

### Why Multi-Stage?

| Benefit | Explanation |
|---------|-------------|
| **Faster builds** | Models downloaded once during build, not at runtime |
| **Smaller image** | CPU-only tools discarded, only GPU runtime kept |
| **Offline support** | Pre-baked models enable airgapped deployments |
| **Reproducibility** | Model versions locked at build time |

**Build time**: ~15-20 minutes (first build), ~2 minutes (cached layers)

**Image size**: ~8-10 GB (includes CUDA runtime + models)

---

## docker-compose.yml Configuration

```yaml
version: '3.8'

services:
  ocr-gpu:
    build:
      context: .
      dockerfile: Dockerfile

    container_name: ocr_gpu_processor

    # GPU reservation (REQUIRED)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    # Shared memory size (for PIL + OpenCV)
    shm_size: '8gb'

    # Volume mounts
    volumes:
      # Input documents (read-write)
      - ./ocr_source:/app/ocr_source

      # Output results (read-write)
      - ./ocr_output:/app/ocr_output

      # Code mount (read-only, for development)
      - ./ocr_gpu_async.py:/app/ocr_gpu_async.py:ro
      - ./optimize_pdfs.py:/app/optimize_pdfs.py:ro

    # Environment variables (optional overrides)
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONUNBUFFERED=1

    # Restart policy (production)
    restart: unless-stopped

    # Network mode (default bridge)
    # network_mode: bridge
```

### Key Configuration Options

#### GPU Reservation
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1              # Use 1 GPU
          capabilities: [gpu]
```

**Multi-GPU**: Change `count: all` to use all available GPUs (requires code changes to distribute work)

#### Shared Memory
```yaml
shm_size: '8gb'
```

**Purpose**: PIL and OpenCV use `/dev/shm` for temporary arrays

**Symptoms of insufficient shm**: `OSError: [Errno 28] No space left on device`

**Tuning**: Increase to 16gb for large documents (500+ pages)

#### Volume Mounts
```yaml
volumes:
  - ./ocr_source:/app/ocr_source          # Input
  - ./ocr_output:/app/ocr_output          # Output
  - ./ocr_gpu_async.py:/app/ocr_gpu_async.py:ro  # Development
```

**Read-only mounts** (`:ro`): Prevent accidental code modification inside container

**Production**: Remove code mounts, bake code into image

---

## Deployment Workflow

### Step 1: Prepare Directories

```bash
cd /path/to/EDCOCR-PUBLIC

# Create input/output directories (if not exist)
mkdir -p ocr_source ocr_output

# Place documents in ocr_source
cp /path/to/documents/*.pdf ocr_source/
```

### Step 2: Build Image

```bash
# Build with Docker Compose (recommended)
docker-compose build

# OR build manually
docker build -t ocr-gpu-local .
```

**First build**: ~15-20 minutes (downloads models, CUDA runtime, etc.)

**Subsequent builds**: ~2-5 minutes (cached layers)

> [!NOTE]
> Cold-start builds are expected to take longer because model preload runs during image build. This is by design for deterministic runtime startup.

**Build output**:
```
[+] Building 1234.5s (18/18) FINISHED
 => [model-preload 1/5] FROM python:3.10-slim
 => [model-preload 2/5] RUN pip install paddlepaddle==2.5.1
 => [model-preload 3/5] RUN pip install paddleocr==2.7.0
 => [model-preload 4/5] COPY download_models.py /tmp/
 => [model-preload 5/5] RUN python /tmp/download_models.py  # ← MODEL DOWNLOAD
 => [stage-1 1/8] FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04
 => [stage-1 2/8] RUN apt-get update && apt-get install -y ...
 => [stage-1 3/8] RUN pip install --no-cache-dir ...
 => [stage-1 4/8] COPY --from=model-preload /root/.paddleocr/ ...
 => [stage-1 5/8] WORKDIR /app
 => [stage-1 6/7] COPY ocr_gpu_async.py /app/
 => [stage-1 7/7] COPY optimize_pdfs.py /app/
 => exporting to image
```

### Step 3: Start Container

```bash
# Start in background (detached mode)
docker-compose up -d

# OR start with logs visible
docker-compose up
```

**Expected output**:
```
[+] Running 1/1
 ⠿ Container ocr_gpu_processor  Started
```

**Verify container is running**:
```bash
docker ps
```

```
CONTAINER ID   IMAGE                  COMMAND                  STATUS
a1b2c3d4e5f6   ocr_local_ocr-gpu     "python3 /app/OCR_GP…"   Up 30 seconds
```

### Step 4: Monitor Progress

```bash
# View logs (follow mode)
docker logs -f ocr_gpu_processor

# View last 100 lines
docker logs --tail 100 ocr_gpu_processor
```

**Log output**:
```
2024-01-15 14:30:22 - INFO - Starting OCR pipeline
2024-01-15 14:30:23 - INFO - Scheduler: Found 45 documents
2024-01-15 14:30:33 - INFO - [MONITOR] Queues: C=12 I=87 A=234 G=5 | PPM: 45.2 (avg: 42.1) | Docs/hr: 18.3
2024-01-15 14:30:43 - INFO - [MONITOR] Queues: C=8 I=65 A=312 G=8 | PPM: 48.7 (avg: 43.5) | Docs/hr: 19.1
...
2024-01-15 15:45:00 - INFO - Pipeline complete. Total pages: 3,421 | Time: 1h 14m 38s
```

### Step 5: Access Results

```bash
# On host machine
ls -lh ocr_output/EXPORT/PDF/
ls -lh ocr_output/EXPORT/TEXT/

# View logs
cat ocr_output/logs/ocr_pipeline_*.log

# Check for failures
cat ocr_output/failures.csv
```

### Step 6: Stop Container

```bash
# Graceful stop (waits for current page to finish)
docker-compose down

# Immediate stop (may leave temp files)
docker-compose kill
```

---

## Windows Batch Files

### run_ocr.bat

```batch
@echo off
echo Starting OCR GPU Processor...
docker-compose up -d
echo Container started. Use 'docker logs -f ocr_gpu_processor' to view logs.
pause
```

**Usage**: Double-click to start container

### run_ocr_async.bat

```batch
@echo off
echo Running OCR pipeline in async mode...
docker exec -it ocr_gpu_processor python3 /app/ocr_gpu_async.py
pause
```

**Usage**: Run pipeline manually (if container already running)

### run_optimization.bat

```batch
@echo off
echo Running PDF optimization (Ghostscript)...
docker exec -it ocr_gpu_processor python3 /app/optimize_pdfs.py
pause
```

**Usage**: Re-compress existing PDFs (separate from main pipeline)

---

## Advanced Operations

### Interactive Shell

```bash
# Enter container shell
docker exec -it ocr_gpu_processor bash

# Inside container
root@a1b2c3d4e5f6:/app# ls -l ocr_output/
root@a1b2c3d4e5f6:/app# python3 ocr_gpu_async.py
root@a1b2c3d4e5f6:/app# exit
```

### Manual Model Download

```bash
# Download additional language models
docker exec -it ocr_gpu_processor python3 -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='ar')  # Download Arabic models
"
```

### Resource Monitoring

```bash
# GPU utilization
docker exec ocr_gpu_processor nvidia-smi

# Container resource usage
docker stats ocr_gpu_processor
```

**Example output**:
```
CONTAINER ID   NAME                CPU %   MEM USAGE / LIMIT     GPU %   GPU MEM
a1b2c3d4e5f6   ocr_gpu_processor   850%    12.5GiB / 64GiB      45%     8.2GiB / 24GiB
```

---

## Troubleshooting

### Issue 1: "could not select device driver"

**Error**:
```
Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Cause**: NVIDIA Container Toolkit not installed

**Fix**:
```bash
# Install toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Issue 2: "No space left on device" (shm)

**Error**:
```
OSError: [Errno 28] No space left on device
```

**Cause**: Insufficient shared memory

**Fix**: Increase `shm_size` in docker-compose.yml
```yaml
shm_size: '16gb'  # Was '8gb'
```

### Issue 3: Out of Memory (GPU)

**Error**:
```
CUDA error: out of memory
```

**Cause**: Too many GPU workers for available VRAM

**Fix**: Reduce `NUM_WORKERS` in `ocr_gpu_async.py`
```python
NUM_WORKERS = 8  # Was 12 (reduce by 33%)
```

**Calculation**: Each worker uses ~600-800 MB VRAM

### Issue 4: Slow Performance

**Symptoms**: <10 pages/minute

**Diagnostics**:
```bash
# Check GPU utilization
docker exec ocr_gpu_processor nvidia-smi

# If GPU util < 30%: Increase NUM_WORKERS
# If GPU util > 95%: Already optimized
# If GPU mem > 90%: Reduce NUM_WORKERS
```

### Issue 5: Models Not Found

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/root/.paddleocr/whl/det/...'
```

**Cause**: Model preload stage failed during build

**Fix**: Rebuild image with verbose output
```bash
docker-compose build --progress=plain
```

Look for errors in model download stage.

### Issue 6: Permission Denied (ocr_output)

**Error** (Linux only):
```
PermissionError: [Errno 13] Permission denied: '/app/ocr_output/...'
```

**Cause**: Container runs as root, host directory owned by user

**Fix**: Set permissions on host
```bash
sudo chown -R $USER:$USER ocr_output/
chmod -R 755 ocr_output/
```

---

## Production Deployment

### 1. Use Named Volumes

```yaml
volumes:
  - ocr_source:/app/ocr_source
  - ocr_output:/app/ocr_output

volumes:
  ocr_source:
    driver: local
  ocr_output:
    driver: local
```

**Benefit**: Persist data across container recreations

### 2. Enable Logging

```yaml
services:
  ocr-gpu:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

**Benefit**: Prevent log files from filling disk

### 3. Health Checks

```yaml
services:
  ocr-gpu:
    healthcheck:
      test: ["CMD", "pgrep", "-f", "ocr_gpu_async.py"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

**Benefit**: Auto-restart on crashes

### 4. Resource Limits

```yaml
services:
  ocr-gpu:
    deploy:
      resources:
        limits:
          cpus: '16'
          memory: 32G
```

**Benefit**: Prevent resource exhaustion on shared hosts

### 5. Backup Automation

```bash
# Cron job (daily backup of output)
0 2 * * * rsync -av /path/to/ocr_output/ /backup/ocr_output_$(date +\%Y\%m\%d)/
```

---

## Performance Tuning

### Optimal Configuration (RTX 4090, 16-core CPU)

```yaml
# docker-compose.yml
shm_size: '16gb'

# ocr_gpu_async.py
NUM_EXTRACTORS = 12
NUM_WORKERS = 16
NUM_COMPRESSORS = 12
IMAGE_QUEUE_SIZE = 300
```

**Expected throughput**: 80-120 pages/minute

### Budget Configuration (GTX 1660 Ti, 8-core CPU)

```yaml
# docker-compose.yml
shm_size: '4gb'

# ocr_gpu_async.py
NUM_EXTRACTORS = 6
NUM_WORKERS = 6
NUM_COMPRESSORS = 6
IMAGE_QUEUE_SIZE = 100
```

**Expected throughput**: 30-50 pages/minute

---

## Related Documentation

- **Architecture**: `docs/architecture/pipeline-design.md`
- **Configuration**: `docs/06-CONFIGURATION-REFERENCE.md`
- **Data Flow**: `docs/architecture/data-flow.md`
