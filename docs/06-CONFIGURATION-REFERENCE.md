# 06: Configuration Reference

## Overview
This document lists runtime controls used by the monolithic pipeline, API service, and distributed coordinator.

> [!NOTE]
> Defaults below are taken directly from current code (`ocr_gpu_async.py`, `api/config.py`, `coordinator/coordinator/settings.py`, and related modules).

## Monolithic Pipeline (`ocr_gpu_async.py`)

### Throughput and Queue Controls
| Variable | Default | Description |
|---|---|---|
| `CHUNK_QUEUE_SIZE` | `50` | Scheduler to extractor queue depth |
| `IMAGE_QUEUE_SIZE` | `200` | Extractor to worker queue depth |
| `RESULT_QUEUE_SIZE` | `5000` | Worker to assembler queue depth |
| `COMPRESSION_QUEUE_SIZE` | `5000` | Assembler to compressor queue depth |
| `NUM_EXTRACTORS` | `8` | Extractor thread count |
| `NUM_WORKERS` | `12` | GPU worker thread count |
| `NUM_COMPRESSORS` | `8` | Compressor thread count |
| `PDF_CONVERSION_THREADS` | `1` | `pdf2image` internal threads |
| `EXTRACTOR_MODE` | `thread` | Extractor execution mode: `thread` or `process` |
| `EXTRACTOR_PROCESS_WORKERS` | `NUM_EXTRACTORS` | Process count used when `EXTRACTOR_MODE=process` |
| `THREAD_JOIN_TIMEOUT` | `30` | Join timeout during shutdown |
| `CHUNK_TARGET_SIZE` | `20` | Pages per scheduled chunk |
| `DPI` | `300` | Render DPI for OCR |

### Paths and Model Inputs
| Variable | Default | Description |
|---|---|---|
| `SOURCE_FOLDER` | `/app/ocr_source` | Input root |
| `OUTPUT_FOLDER` | `/app/ocr_output` | Output root |
| `TEMP_FOLDER` | `/app/ocr_temp` | Temp page artifacts |
| `LOG_DIR` | `/app/ocr_output/logs` | Pipeline logs |
| `FAILURE_REPORT` | `/app/ocr_output/failures.csv` | Failure CSV |
| `HEALTHCHECK_FILE` | `/app/ocr_healthcheck` | Heartbeat file |
| `FASTTEXT_MODEL_PATH` | `/app/models/lid.176.bin` | Language model path |

### Feature Toggles
| Variable | Default | Description |
|---|---|---|
| `ENABLE_PREPROCESSING` | `false` | Enable preprocessing before OCR |
| `PREPROCESSING_LEVEL` | `standard` | `standard`, `enhanced`, `aggressive` |
| `ENABLE_VALIDATION` | `true` | Validation sidecar generation |
| `ENABLE_DPI_ESCALATION` | `false` | Retry low-confidence pages at higher DPI |
| `DPI_CONFIDENCE_THRESHOLD` | `0.60` | Escalation threshold |
| `ENABLE_NER` | `false` | NER sidecar generation |
| `ENABLE_HANDWRITING` | `false` | Handwriting sidecar generation |
| `ENABLE_CLASSIFICATION` | `false` | Classification sidecar generation |
| `ENABLE_EXTRACTION` | `false` | Structured extraction sidecar generation |
| `ENABLE_SPECIALIST_ROUTING` | `false` | Routing sidecar generation from classification/entity signals |
| `CLASSIFICATION_MODE` | `heuristic` | `heuristic`, `ml`, or `ensemble` classifier mode |
| `ML_CLASSIFICATION_MODEL` | `microsoft/layoutlmv3-base` | LayoutLMv3 model path for ML classification |
| `ML_CLASSIFICATION_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for ML-only classification acceptance |
| `CLASSIFICATION_MULTI_LABEL_THRESHOLD` | `0.3` | Minimum document-level score to keep a secondary label |
| `CLASSIFICATION_MULTI_LABEL_MAX_LABELS` | `3` | Maximum labels written to `document_labels` |
| `CLASSIFICATION_PROFILE_PATH` | empty | Optional JSON file for customer profile overlays and route hints |

### Language Detection

Per-span language detection is an opt-in enhancement that labels each OCR line
with a BCP-47 language code, aggregates pages and documents, and writes a
`.language.json` sidecar under `EXPORT/LANGUAGE/`.  The pass runs in-line with
the GPU worker and falls through to a Unicode-script heuristic for short spans
where FastText is unreliable.  A dedicated CLI
(`python -m ocr_local.features.language_detection --doc <pdf>`) and REST
endpoint (`POST /api/v1/jobs/{id}/redetect-language`) can re-run detection
against an already-OCR'd PDF without touching the OCR text layer.

| Environment Variable | Default | Description |
|---|---|---|
| `ENABLE_PER_SPAN_LANGUAGE` | `false` | Enable per-span language detection (requires FastText `lid.176.bin` model) |
| `LANGUAGE_INCLUDE_SPANS` | `false` | Include full span list in `.language.json` sidecar (verbose mode) |
| `LANGUAGE_SHORT_SPAN_THRESHOLD` | `20` | Minimum non-whitespace chars before using FastText instead of the script heuristic |
| `LANGUAGE_CONFIDENCE_THRESHOLD` | `0.4` | Minimum FastText confidence; below this threshold spans fall back to `"und"` |
| `LANGUAGE_REDACT_SAMPLES` | `privilege_or_short_doc` | When to suppress `text_sample` (`true`, `false`, or `privilege_or_short_doc`) |

### Video Ingestion

Video files placed in the source directory are automatically detected and processed.
The pipeline extracts frames at a configurable sampling interval and feeds each frame
into the standard OCR page pipeline (language detection, OCR, assembly, compression).
No CLI flag is required to enable this feature -- any file with a recognized video
extension or magic-byte signature is accepted as an OCR source.

#### Supported Container Formats

| Extension | Container |
|---|---|
| `.mp4` | MPEG-4 Part 14 |
| `.avi` | Audio Video Interleave (RIFF/AVI) |
| `.mov` | Apple QuickTime |
| `.m4v` | MPEG-4 Visual |
| `.mpg`, `.mpeg` | MPEG-1 / MPEG-2 |
| `.mkv` | Matroska |
| `.webm` | WebM |

Files are also recognized by magic-byte signature (RIFF/AVI header, ISO base media `ftyp`
box, or Matroska/EBML header), so correctly structured files with non-standard extensions
will still be accepted.

#### Environment Variables

| Variable | Default | Min | Max | Description |
|---|---|---|---|---|
| `VIDEO_FRAME_SAMPLE_SECONDS` | `1.0` | `0.1` | `3600.0` | Interval in seconds between sampled frames. A value of `1.0` means one frame is sampled every second of video. Lower values produce more frames (and more OCR pages). |
| `VIDEO_MAX_FRAMES` | `300` | `1` | `10000` | Hard upper limit on the total number of frames extracted from a single video. When the sampling interval would produce more frames than this cap, the frame plan is deterministically downsampled while preserving the first and last frames. |

#### How Frame Sampling Works

1. **Probe** -- OpenCV reads the video metadata (frame count and FPS). If container
   metadata is missing or unreliable, the pipeline falls back to a full decode count.
2. **Plan** -- A frame plan is built by stepping through the video at intervals of
   `VIDEO_FRAME_SAMPLE_SECONDS * FPS` frames. The last frame of the video is always
   included to capture any trailing content.
3. **Cap** -- If the plan exceeds `VIDEO_MAX_FRAMES`, it is downsampled to exactly that
   many entries using uniform index selection that preserves the first and last frames.
4. **Extract** -- Selected frames are decoded via OpenCV, converted to PIL RGB images,
   and injected into the existing page pipeline. Each frame is treated as one "page" for
   OCR processing, assembly, compression, and all enabled sidecar outputs.

The sampling is deterministic: the same video with the same configuration always produces
the same set of frame indices, which supports crash-resume and reproducible processing.

#### Dependency

Video ingestion requires OpenCV (`cv2`). The Docker production image includes
`opencv-python-headless` by default. If OpenCV is not installed, video files will raise
a descriptive runtime error rather than silently fail.

#### Example Configuration

Process one frame every 2 seconds, capping at 500 frames per video:

```yaml
# docker-compose.override.yml
services:
  ocr-gpu:
    environment:
      - VIDEO_FRAME_SAMPLE_SECONDS=2.0
      - VIDEO_MAX_FRAMES=500
```

For a 10-minute video at 30 FPS (18,000 total frames), this configuration samples one
frame every 60th frame, producing 300 candidate frames. Since 300 is below the 500-frame
cap, all 300 frames are processed as OCR pages.

#### Caveats

- **Processing time scales with frame count.** A long video with a low sample interval
  can produce thousands of OCR pages. Use `VIDEO_MAX_FRAMES` to bound processing cost.
- **Audio tracks are ignored.** Only visual frames are extracted; there is no speech-to-text
  integration.
- **Codec support depends on the OpenCV build.** The Docker image includes FFmpeg-backed
  OpenCV which covers all listed container formats. Exotic codecs may require a custom
  OpenCV build.
- **Frame quality varies.** Motion blur, compression artifacts, and low resolution in
  source video can reduce OCR accuracy compared to scanned document images.

### Forensic-Core Defaults

These settings define the default forensic-grade processing path. They should remain
safe without any optional intelligence workers or model-specific extensions.

| Control | Default | Why it is core |
|---|---|---|
| `ENABLE_VALIDATION` | `true` | Validation evidence is part of the forensic baseline |
| custody logging | on unless `--no-custody` | Chain-of-custody remains part of the default integrity model |
| `API_AUDIT_LOG_ENABLED` | `true` | Request audit evidence stays on by default for service use |

### AI-Adjacent Controls

These toggles add semantic or model-driven sidecars on top of the forensic core.
They remain opt-in so the default pipeline promise does not silently shift from OCR
and evidence capture into optional document-intelligence behavior.

- Request-time AI-adjacent controls: `--enable-docintel`, `--docintel-mode`, `--enable-ner`, `--enable-classification`, `--enable-extraction`, `--enable-specialist-routing`
- Runtime AI-adjacent controls default off: `ENABLE_NER`, `ENABLE_CLASSIFICATION`, `ENABLE_EXTRACTION`, `ENABLE_SPECIALIST_ROUTING`, `ENABLE_LAYOUTLM`

> [!NOTE]
> These toggles are part of the AI-adjacent layer, not the minimum forensic-core contract. EDCOCR's core forensic deliverables remain OCR/fallback processing, primary PDF/TXT artifacts, and custody/validation evidence. See [docs/architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).

## Pipeline CLI Flags
| Flag | Description |
|---|---|
| `--enable-docintel` | Enable PP-Structure analysis |
| `--docintel-mode {layout_only,tables_only,full}` | DocIntel mode |
| `--export-tables` | Export table HTML/CSV outputs |
| `--enable-form-detection` | Form field detection (requires docintel) |
| `--enable-kv-extraction` | Key-value extraction (requires docintel) |
| `--enable-privilege-detection` | Privilege indicator detection (requires docintel) |
| `--no-custody` | Disable custody logging |
| `--enable-preprocessing` | Enable preprocessing |
| `--preprocessing-level {standard,enhanced,aggressive}` | Preprocessing intensity |
| `--enable-dpi-escalation` | Enable DPI escalation |
| `--enable-ner` | Enable NER |
| `--enable-handwriting` | Enable handwriting detection |
| `--enable-classification` | Enable classification |
| `--enable-extraction` | Enable structured extraction |
| `--enable-specialist-routing` | Enable routing sidecar output |
| `--extractor-mode {thread,process}` | Override extractor mode |
| `--extractor-process-workers N` | Set process worker count for `process` mode |

## API Service (`api/config.py`, `api/limits.py`)

### Server and Storage
| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API bind port |
| `EXPOSE_API_DOCS` | `false` | Mount `/docs`, `/redoc`, and `/openapi.json` when true |
| `SOURCE_FOLDER` | `/app/ocr_source` | Job source root |
| `OUTPUT_FOLDER` | `/app/ocr_output` | Job output root |
| `API_DB_PATH` | `${OUTPUT_FOLDER}/jobs.db` | SQLite DB path |
| `OCR_QUEUE_THRESHOLDS_PATH` | `${OUTPUT_FOLDER}/queue_thresholds.json` | Persisted queue alert threshold config used by `/api/v1/queues/{queue_name}/threshold` |

### Job Limits and Polling
| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | `5120` | Upload size limit |
| `MAX_REQUEST_BODY_SIZE` | `10485760` | Non-multipart request body cap in bytes; set `0` to disable |
| `MAX_CONCURRENT_JOBS` | `4` | Active job cap |
| `PIPELINE_SCRIPT` | `ocr_gpu_async.py` absolute path | Script used by job manager |
| `PIPELINE_POLL_INTERVAL` | `5` | Progress polling interval (seconds) |
| `JOB_PROCESSING_TIMEOUT_MINUTES` | `30` | Default per-job processing timeout before the API marks a pipeline run failed |
| `RESULT_RETENTION_DAYS` | `90` | Artifact retention window |

### Security and Rate Limits
| Variable | Default | Description |
|---|---|---|
| `OCR_API_KEY` | empty | Enables API-key auth when set |
| `ALLOW_UNAUTHENTICATED` | `false` | Development-only auth bypass |
| `ANONYMOUS_ROLE` | `viewer` | Role assigned when `ALLOW_UNAUTHENTICATED=true` |
| `API_ALLOWED_IPS` | empty | Optional comma-separated ingress allowlist for API clients (partial exact-match control; not proxy-aware) |
| `API_AUDIT_LOG_ENABLED` | `true` | Enable append-only request audit logging to JSONL |
| `API_AUDIT_LOG_PATH` | derived from `OUTPUT_FOLDER/logs/api-audit.jsonl` | Override request-audit JSONL path |
| `OCR_RATE_LIMIT` | `60/minute` | Read endpoints |
| `OCR_SUBMIT_RATE_LIMIT` | `10/minute` | Submit/retry endpoints |

### Feature Flags
| Variable | Default | Description |
|---|---|---|
| `ENABLE_TRANSFORMS` | `false` | Enable transform operations API (rotate, extract, merge) |
| `ENABLE_STAMPING` | `false` | Enable stamp operations API (Bates, designation) |
| `ENABLE_MULTITENANCY` | `false` | Enable tenant-scoped auth, quota, usage, and admin endpoints |
| `ENABLE_DASHBOARD` | `false` | Enable dashboard, fleet, alerts, queue-threshold, and analytics endpoints |
| `ENABLE_TRANSLATION_API` | `false` | Enable optional translation job, QE, tenant config, glossary, and batch endpoints |
| `OCR_FEDERATION_CUSTODY_ENABLED` | `false` | Enable federation custody ingest endpoint |
| `OCR_FEDERATION_CUSTODY_AUTH_TOKEN` | empty | Bearer token required by the federation custody ingest endpoint when enabled |

> [!NOTE]
> `enable_docintel`, LayoutLMv3, semantic search, and related model-assisted flows are optional AI-adjacent features layered on top of the forensic-core API contract. They are additive and should not be treated as baseline OCR evidence guarantees. See [docs/architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).

### Tenant Cost Accounting
| Variable | Default | Description |
|---|---|---|
| `TENANT_COST_PER_PAGE_USD` | `0.0` | Internal estimated cost rate per processed page |
| `TENANT_COST_PER_GIB_INGESTED_USD` | `0.0` | Internal estimated cost rate per GiB of ingested tenant source data |
| `TENANT_COST_PER_API_CALL_USD` | `0.0` | Internal estimated cost rate per authenticated tenant API request |
| `TENANT_COST_PER_PROCESSING_HOUR_USD` | `0.0` | Internal estimated cost rate per hour of accumulated tenant processing time |

### Tenant SLO Monitoring
| Variable | Default | Description |
|---|---|---|
| `TENANT_SLO_WINDOW_HOURS` | `24` | Rolling window size used for tenant SLO snapshots |
| `TENANT_SLO_TARGET_SUCCESS_RATE` | `0.95` | Minimum acceptable completed/(completed+failed+cancelled) ratio |
| `TENANT_SLO_TARGET_P95_PROCESSING_SECONDS` | `1800.0` | Maximum acceptable p95 tenant job processing latency in seconds |

### Webhooks
| Variable | Default | Description |
|---|---|---|
| `WEBHOOK_SECRET` | empty | Default HMAC secret |
| `WEBHOOK_TIMEOUT` | `30` | Delivery timeout |
| `WEBHOOK_MAX_RETRIES` | `3` | Retry attempts |
| `WEBHOOK_ALLOW_HTTP` | `false` | Permit HTTP webhooks |
| `WEBHOOK_ALLOW_PRIVATE` | `false` | Permit private-network targets |

### Durable Event Stream
| Variable | Default | Description |
|---|---|---|
| `API_EVENT_STREAM_ENABLED` | `false` | Enable append-only local JSONL publication for job and batch lifecycle events |
| `API_EVENT_STREAM_PATH` | `${OUTPUT_FOLDER}/logs/api-events.jsonl` | Durable event-stream sink used by the API managers |

## Distributed Coordinator (`coordinator/`)

### Django and Core Services
| Variable | Default | Description |
|---|---|---|
| `DJANGO_SECRET_KEY` | required | Django secret key |
| `DJANGO_DEBUG` | `False` | Debug mode |
| `DEPLOYMENT_ENV` | `development` | Deployment level: `development`, `staging`, `production` |
| `PRODUCTION_READINESS_ACK` | `false` | Required (`true`) when `DEPLOYMENT_ENV=production` |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Allowed host header values |
| `DATABASE_URL` | required | PostgreSQL connection URL |
| `REDIS_URL` | `redis://redis:6379/0` | Django cache backend URL |
| `CELERY_BROKER_URL` | required | RabbitMQ URL |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | Celery backend |
| `CELERY_USE_QUORUM_QUEUES` | `False` | Use RabbitMQ quorum queue declarations for `coordinator`, `ocr_gpu`, `cpu_general` |
| `STORAGE_BACKEND` | `nfs` | `nfs` (default) or `s3` (kickoff mode) |
| `NFS_ROOT` | `/shared` | Shared filesystem root |
| `S3_ENDPOINT` | empty | S3-compatible endpoint when `STORAGE_BACKEND=s3` |
| `S3_BUCKET` | empty | Bucket/container name for object storage |
| `S3_ACCESS_KEY` | empty | S3 access key |
| `S3_SECRET_KEY` | empty | S3 secret key |
| `S3_REGION` | empty | Optional S3 region |
| `MINIO_ROOT_USER` | empty | Local MinIO harness root/access key used by `coordinator/docker-compose.local-minio.yml` |
| `MINIO_ROOT_PASSWORD` | empty | Local MinIO harness root/secret key used by `coordinator/docker-compose.local-minio.yml` |

### Operations and Worker Settings
| Variable | Default | Description |
|---|---|---|
| `METRICS_API_KEY` | empty | Protects `/api/v1/metrics/` when set |
| `WORKER_QUEUES` | `cpu_general` fallback | Worker queue membership |
| `WORKER_CONCURRENCY` | `4` | Worker process concurrency |
| `CUDA_VISIBLE_DEVICES` | unset | Pin worker container to a specific GPU (multi-GPU template) |
| `JOB_PROCESSING_TIMEOUT_MINUTES` | `30` | Default stale-job timeout; `jobs.settings_json.processing_timeout_minutes` can override per job |
| `JOB_RETENTION_DAYS` | `30` | Cleanup retention for terminal jobs |

Distributed queue isolation keeps advanced AI-adjacent work off the default OCR path:
`extract_structured_data` routes to `nlp_general`, and LayoutLMv3 work routes to
`ocr_layoutlm` instead of the main OCR queue.

> [!NOTE]
> `coordinator/docker-compose.multi-gpu.yml` for one-worker-per-GPU scaling.

> [!WARNING]
> Production coordinator startup is guarded: `DEPLOYMENT_ENV=production` requires `PRODUCTION_READINESS_ACK=true`. Complete `docs/deployment/distributed-readiness-checklist.md` first.

## Example Compose Override
```yaml
services:
  ocr-gpu:
    environment:
      - NUM_WORKERS=16
      - NUM_EXTRACTORS=10
      - ENABLE_DPI_ESCALATION=true
      - ENABLE_NER=true
      - OCR_API_KEY=replace-with-secret
```

Apply changes:
```bash
docker compose up -d --force-recreate
```

## Tuning Guidelines

### Thread Pool Tuning

**NUM_EXTRACTORS**:
- CPU-bound, limited by GIL with diminishing returns above 16
- Bottleneck indicator: high `chunk_queue` depth
- Recommendation: 50-75% of CPU cores

**NUM_WORKERS**:
- GPU-bound, limited by VRAM (600-800 MB per worker)
- Bottleneck indicator: high `image_queue` depth, GPU utilization above 95%
- Recommendation: `floor(GPU_VRAM_GB / 0.7)`, then reduce for safety headroom

**NUM_COMPRESSORS**:
- CPU-bound (Ghostscript is single-threaded per process)
- Bottleneck indicator: high `compression_queue` depth
- Recommendation: match `NUM_EXTRACTORS`

### Queue Size Memory Impact

| Queue | Default | Per-Item Size | Worst-Case Total |
|---|---|---|---|
| `CHUNK_QUEUE_SIZE` | 50 | ~10 KB (metadata) | 500 KB |
| `IMAGE_QUEUE_SIZE` | 200 | ~20 MB (PIL image) | **4 GB** |
| `RESULT_QUEUE_SIZE` | 5000 | ~50 KB | 250 MB |
| `COMPRESSION_QUEUE_SIZE` | 5000 | ~8 bytes (path) | 40 KB |

`IMAGE_QUEUE_SIZE` dominates memory. Reduce to 100 or 50 on low-memory systems (8 GB RAM).
Increase to 300-500 on high-memory systems (64 GB RAM) for better GPU throughput.

### DPI Quality vs Performance

| DPI | Quality | Speed | Memory per A4 Page | Use Case |
|-----|---------|-------|--------------------|----------|
| 150 | Low | Fast | ~5 MB | Draft/preview |
| 200 | Medium | Moderate | ~9 MB | Informal documents |
| **300** | **High** | **Moderate** | **~20 MB** | **Production (default)** |
| 400 | Very High | Slow | ~35 MB | Archival/legal |
| 600 | Maximum | Very Slow | ~80 MB | Historical preservation |

### Language Detection Confidence

The two-pass language detection uses FastText to check the initial OCR result. If a different
language is detected above the confidence threshold (`LANG_CONFIDENCE_THRESHOLD`, default 0.4),
the page is re-OCRed with the detected language model.

| Threshold | Re-OCR Rate | Quality Impact |
|-----------|-------------|----------------|
| 0.2 | Very High | Many false positives |
| **0.4** | **Medium** | **Balanced (default)** |
| 0.6 | Low | May miss language switches |
| 0.8 | Very Low | Only obvious mismatches |

## Ghostscript Compression Presets

The `optimize_pdfs.py` compressor uses Ghostscript quality presets:

| Preset | DPI | Color | Use Case | Compression |
|--------|-----|-------|----------|-------------|
| `/screen` | 72 | Downsampled | Web preview | Maximum |
| `/ebook` | 150 | Optimized | E-readers | High |
| `/printer` | 300 | Preserved | Office printing | Medium |
| **`/prepress`** | **300** | **Preserved** | **Production/archival (default)** | **Low** |
| `/default` | Varies | Varies | Balanced | Medium |

## Logging Configuration

| Level | Visibility | Use Case |
|-------|------------|----------|
| `DEBUG` | All operations (page-level) | Development, debugging |
| **`INFO`** | **Progress, monitor stats** | **Production default** |
| `WARNING` | Recoverable issues (fallbacks) | Quiet operation |
| `ERROR` | Page-level failures | Critical errors only |

Override via the `LOG_LEVEL` constant in `ocr_gpu_async.py` or by setting `MONITOR_INTERVAL`
(default 10 seconds) to control reporting frequency.

### Monitor Output

```
[MONITOR] Queues: C=12 I=87 A=234 G=5 | PPM: 45.2 (avg: 42.1) | Docs/hr: 18.3
```

- `C`: chunk_queue depth
- `I`: image_queue depth
- `A`: assembly_queue (result_queue) depth
- `G`: compression_queue depth
- `PPM`: instantaneous pages/minute (last interval)
- `avg`: average PPM (session lifetime)
- `Docs/hr`: documents completed per hour

## Performance Profiles

### Profile 1: Maximum Throughput

Hardware: RTX 4090, 32-core CPU, 64 GB RAM. Expected: 120-150 PPM.

```yaml
environment:
  - NUM_EXTRACTORS=16
  - NUM_WORKERS=24
  - NUM_COMPRESSORS=16
  - IMAGE_QUEUE_SIZE=500
  - DPI=200
```

### Profile 2: Balanced (Default)

Hardware: RTX 3090, 16-core CPU, 32 GB RAM. Expected: 60-80 PPM.

```yaml
environment:
  - NUM_EXTRACTORS=8
  - NUM_WORKERS=12
  - NUM_COMPRESSORS=8
  - IMAGE_QUEUE_SIZE=200
  - DPI=300
```

### Profile 3: Maximum Quality

Hardware: RTX 3080, 12-core CPU, 16 GB RAM. Expected: 30-50 PPM.

```yaml
environment:
  - NUM_EXTRACTORS=6
  - NUM_WORKERS=8
  - NUM_COMPRESSORS=4
  - IMAGE_QUEUE_SIZE=100
  - DPI=400
```

### Profile 4: Budget GPU

Hardware: GTX 1660 Ti, 8-core CPU, 16 GB RAM. Expected: 20-30 PPM.

```yaml
environment:
  - NUM_EXTRACTORS=4
  - NUM_WORKERS=4
  - NUM_COMPRESSORS=4
  - IMAGE_QUEUE_SIZE=50
  - DPI=300
```

## Troubleshooting

### Low Pages/Minute

Check monitor output queue depths. If upstream queues (`C`, `I`) are empty but downstream
queues (`A`, `G`) are full, the bottleneck is the assembler or compressor -- increase
`NUM_COMPRESSORS`.

### High Memory Usage

If all queues are at or near capacity, reduce `IMAGE_QUEUE_SIZE` to 100 or 50.

### GPU Underutilized

If `nvidia-smi` shows GPU utilization below 30%, increase `NUM_WORKERS`.

## Safe Tuning Sequence
1. Tune thread/queue variables first.
2. Enable one feature flag at a time.
3. Validate output artifacts and failure logs.
4. Scale to distributed mode only after single-node behavior is stable.
