# 02: Quickstart (5-Minute Success)

## Goal
Get OCR processing running end-to-end in under 10 minutes with Docker, then verify outputs in `ocr_output/EXPORT`.

## Path A (Recommended): Docker GPU Runtime

### Prerequisites
| Requirement | Check |
|---|---|
| Docker Engine + Compose v2 | `docker compose version` |
| NVIDIA drivers (GPU mode) | `nvidia-smi` |
| NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi` |

### Step 1: Build and start
```bash
docker compose up -d --build
```

Cold-start note: the first build preloads OCR models and can take materially longer than incremental builds.

### Step 2: Confirm container state
```bash
docker ps --filter "name=ocr_gpu_processor"
```

Expected container name: `ocr_gpu_processor`.

### Step 3: Add input documents
Put files under `ocr_source/`. The scheduler scans recursively.

Examples:
```bash
# Bash
cp /path/to/source/*.pdf ocr_source/
```

```powershell
# PowerShell
Copy-Item -Path C:\path\to\source\* -Destination .\ocr_source\ -Recurse
```

### Step 4: Watch processing
```bash
docker logs -f ocr_gpu_processor
```

### Step 5: Validate output artifacts
```bash
# Bash
ls ocr_output/EXPORT/PDF
ls ocr_output/EXPORT/TEXT
cat ocr_output/failures.csv
```

```powershell
# PowerShell
Get-ChildItem .\ocr_output\EXPORT\PDF -Recurse
Get-ChildItem .\ocr_output\EXPORT\TEXT -Recurse
Get-Content .\ocr_output\failures.csv
```

---

## Runtime Environment Variables

### Pipeline Throughput Controls
| Variable | Default | Effect |
|---|---|---|
| `NUM_EXTRACTORS` | `8` | CPU extraction workers |
| `NUM_WORKERS` | `12` | GPU OCR workers |
| `NUM_COMPRESSORS` | `8` | PDF compressor workers |
| `IMAGE_QUEUE_SIZE` | `200` | In-memory image buffer depth |
| `EXTRACTOR_MODE` | `thread` | `thread` (default) or `process` for CPU extractor multiprocessing |
| `EXTRACTOR_PROCESS_WORKERS` | `NUM_EXTRACTORS` | Process workers when `EXTRACTOR_MODE=process` |
| `DPI` | `300` | OCR render resolution |

### Optional Feature Flags
| Variable | Default | Effect |
|---|---|---|
| `ENABLE_PREPROCESSING` | `false` | OpenCV preprocessing pipeline |
| `PREPROCESSING_LEVEL` | `standard` | `standard`, `enhanced`, `aggressive` |
| `ENABLE_DPI_ESCALATION` | `false` | Retries low-confidence pages at higher DPI |
| `ENABLE_NER` | `false` | Named entity sidecars |
| `ENABLE_HANDWRITING` | `false` | Handwriting detection sidecars |
| `ENABLE_CLASSIFICATION` | `false` | Document classification sidecars |
| `ENABLE_EXTRACTION` | `false` | Structured extraction sidecars |
| `ENABLE_VALIDATION` | `true` | Validation sidecars |

### API Security and Limits
| Variable | Default | Effect |
|---|---|---|
| `OCR_API_KEY` | empty | Enables API-key auth when set |
| `OCR_RATE_LIMIT` | `60/minute` | Read endpoint rate limit |
| `OCR_SUBMIT_RATE_LIMIT` | `10/minute` | Submit endpoint rate limit |

> [!TIP]
> Add variables under `services.ocr-gpu.environment` in `docker-compose.yml`, then restart with `docker compose up -d --force-recreate`.

---

## Path B: Local Developer Setup (Non-Docker)

Use this path for module-level development and tests. Containerized execution remains the primary production path.

### Step 1: Python environment
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: System dependencies
Install:
- `tesseract-ocr`
- `poppler-utils`
- `ghostscript`

### Step 3: Optional model preload
```bash
python download_models.py
```

### Step 4: Run app components
```bash
# API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

> [!WARNING]
> `ocr_gpu_async.py` validates source/output paths against `/app/...` defaults. Prefer Docker for full pipeline runs unless your local environment mirrors those paths.

---

## Path C: Distributed Coordinator Startup

### Step 1: Configure secrets
Copy `coordinator/.env.example` to `coordinator/.env` and set all required values.

Set `DEPLOYMENT_ENV=staging` for validation. Do not set `DEPLOYMENT_ENV=production` until readiness checks pass.

### Step 2: Start coordinator stack
```bash
cd coordinator
docker compose -f docker-compose.coordinator.yml up -d --build
```

### Step 3: Start worker stack
```bash
docker compose -f docker-compose.worker.yml up -d --build
```

### Step 4: Verify services
| Service | Default URL |
|---|---|
| Django | `http://localhost:8000` |
| Flower | `http://localhost:5555` |
| RabbitMQ UI | `http://localhost:15672` |

---

## Distributed Production Gate
- Run the checklist in `docs/deployment/distributed-readiness-checklist.md`.
- `DEPLOYMENT_ENV=production` requires `PRODUCTION_READINESS_ACK=true` in coordinator settings.
- Keep `PRODUCTION_READINESS_ACK=false` in development/staging until checklist evidence is complete.

---

## Quick Verification Checklist
- Container is healthy and logging throughput.
- At least one searchable PDF appears in `ocr_output/EXPORT/PDF`.
- Matching text artifact appears in `ocr_output/EXPORT/TEXT`.
- `ocr_output/failures.csv` has only expected errors.
