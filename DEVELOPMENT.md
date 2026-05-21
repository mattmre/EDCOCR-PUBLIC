# Development Guide

This guide is the working reference for anyone developing on EDCOCR. It covers the local dev loop, the production file layout, key conventions, and the gotchas that come up often enough to be worth writing down.

If you are looking for **how to install** EDCOCR for use, see [`INSTALL.md`](INSTALL.md). If you are looking for **how to contribute changes**, see [`CONTRIBUTING.md`](CONTRIBUTING.md). This document assumes you have both of those open already.

## Table of Contents

1. [Quickstart](#1-quickstart)
2. [Repository Layout](#2-repository-layout)
3. [Production File: ocr_gpu_async.py](#3-production-file-ocr_gpu_asyncpy)
4. [Key Modules](#4-key-modules)
5. [Adding a New Feature](#5-adding-a-new-feature)
6. [Adding a New Language](#6-adding-a-new-language)
7. [Adjusting Performance](#7-adjusting-performance)
8. [Testing Locally](#8-testing-locally)
9. [Common Gotchas](#9-common-gotchas)
10. [Style and Conventions](#10-style-and-conventions)

---

## 1. Quickstart

```bash
git clone https://github.com/mattmre/EDCOCR-PUBLIC.git
cd EDCOCR-PUBLIC
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

# Run the test suite
python -m pytest tests/ -v

# Run the production pipeline against test fixtures
mkdir -p ocr_source ocr_output
cp tests/fixtures/sample.pdf ocr_source/
python ocr_gpu_async.py
ls ocr_output/EXPORT/PDF/
```

---

## 2. Repository Layout

```
EDCOCR-PUBLIC/
├── ocr_gpu_async.py            Production pipeline (target all new work here)
├── ocr_local/                  Organized namespace package
│   ├── features/               Opt-in feature modules (NER, classification, etc.)
│   ├── ml/                     ML modules (LayoutLMv3, calibration)
│   ├── infra/                  Infrastructure modules (engine selection, caching)
│   └── config/                 Configuration + language registry
├── api/                        FastAPI REST + WebSocket + SSE
├── coordinator/                Django coordinator + Celery
├── sdk/
│   ├── python/                 Python SDK package
│   └── typescript/             TypeScript SDK package
├── helm/ocr-local/             Kubernetes Helm chart (26 templates)
├── schemas/                    JSON Schema definitions
├── scripts/                    Operator scripts + migration tools
├── tests/                      Pytest suite
├── docs/                       Documentation suite
├── presentation/               HTML presentations (landing, slides, architecture)
├── frontend/                   Optional admin UI
└── ocr_source/ + ocr_output/   Volume mounts (created at runtime)
```

### Top-Level Files

- `README.md` — Public landing page
- `ARCHITECTURE.md` — Top-level architecture with diagrams
- `INSTALL.md` — Installation guide
- `CONTRIBUTING.md` — Contribution guidelines
- `DEVELOPMENT.md` — This file
- `CHANGELOG.md` — Release history
- `SECURITY.md` — Security disclosure policy
- `LICENSE` — Apache 2.0
- `NOTICE` — Third-party attributions
- `Makefile` — Convenient targets (test, lint, build, run)
- `Dockerfile` — Single-host production image
- `docker-compose.yml` — Single-host orchestration
- `requirements.txt` — Pinned Python dependencies
- `ruff.toml` — Linter config
- `version.py` — Version string

---

## 3. Production File: ocr_gpu_async.py

**`ocr_gpu_async.py` is the production pipeline.** All new pipeline work targets this file.

Avoid:

- `OCR_GPU.py` — sync v2, functional but not production-grade.
- `legacy/OCRLOCAL.py` — v1, deprecated, do not use.

The file is large (~4000 lines) and uses threading (not asyncio) for parallelism. Key structural patterns:

- **Constants at module top** (ALL_CAPS naming).
- **Six thread functions**, one per pipeline stage: `scheduler_thread`, `extractor_thread`, `gpu_worker_thread`, `assembler_thread`, `compression_thread`, `monitor_thread`.
- **Queues** for inter-stage communication: `chunk_queue`, `image_queue`, `page_queue`, `compression_queue`.
- **`doc_registry`** for tracking per-document assembly state — protected by `_doc_registry_lock` (RLock).
- **`SHUTDOWN_EVENT`** for graceful shutdown on SIGTERM/SIGINT.

### When You Need to Modify `ocr_gpu_async.py`

- **Be aware of `global` declarations** in worker thread functions. Misplacement causes `SyntaxError` or `UnboundLocalError`. The test suite will catch this.
- **PRs touching `ocr_gpu_async.py` must merge sequentially.** Parallel merges produce hard-to-diagnose conflicts in thread-local logic.
- **Use `with _doc_registry_lock:`** for any read or write of `doc_registry`. The lock is an `RLock`, so re-entrant acquisition in exception handlers is safe.
- **Subprocess timeouts are enforced** at constants: `GS_TIMEOUT`, `TESSERACT_TIMEOUT`, `POPPLER_TIMEOUT`.

---

## 4. Key Modules

### Pipeline

| Module | Purpose |
|---|---|
| `ocr_gpu_async.py` | Production async pipeline |
| `optimize_pdfs.py` | Ghostscript PDF compression with integrity validation |
| `download_models.py` | Pre-downloads PaddleOCR models for 34-45 languages |
| `pipeline_config.py` | `PipelineConfig` dataclass + `create_pipeline_config()` |

### Recognition

| Module | Purpose |
|---|---|
| `ocr_local/infra/ocr_inference_backend.py` | ONNX Runtime / OpenVINO selection |
| `ocr_local/infra/engine_selection.py` | Quality-based Tesseract / PaddleOCR routing |
| `ocr_local/infra/page_routing.py` | Smart page-to-backend routing |
| `ocr_local/infra/adaptive_batch.py` | Complexity-aware dynamic batch sizing |
| `ocr_local/infra/page_cache.py` | Memory-mapped page caching with LRU eviction |
| `dpi_escalation.py` | Adaptive DPI escalation (300 to 450 to 600) |
| `preprocessing.py` | OpenCV deskew, denoise, binarize |
| `validation.py` | Per-page confidence + quality classification |
| `table_fallback.py` | Per-region table OCR fallback |
| `ocr_local/features/vertical_text.py` | CJK vertical text detection |

### Features

| Module | Purpose |
|---|---|
| `ner.py` | spaCy NER with forensic entity types |
| `extraction.py` | PaddleNLP UIE + regex fallback for structured fields |
| `classification.py` | Document classification ensemble |
| `handwriting.py` | Handwriting detection heuristics |
| `signature_verification.py` | Signature detection (advisory-only) |
| `custody.py` | Hash-chained JSONL audit logging |

### API

| Module | Purpose |
|---|---|
| `api/main.py` | FastAPI app + middleware |
| `api/auth.py` | API key authentication (X-API-Key + Bearer) |
| `api/routers/jobs.py` | 6 REST endpoints for job management |
| `api/routers/ws.py` | WebSocket endpoint for job progress |
| `api/routers/outputs.py` | Output manifest retrieval |
| `api/routers/review.py` | Human review queue |
| `api/routers/recall.py` | Entity recall search |
| `api/webhooks.py` | HMAC-SHA256 signed webhooks |
| `api/cloud_storage.py` | Azure Blob + GCS connectors |
| `api/event_bus.py` | Kafka + SNS/SQS integration |

### Coordinator

| Module | Purpose |
|---|---|
| `coordinator/coordinator/settings.py` | Django settings (env-driven) |
| `coordinator/jobs/models.py` | Job, Worker, PageResult, CustodyEvent, PiiEntity |
| `coordinator/jobs/tasks.py` | Celery task definitions |
| `coordinator/jobs/views.py` | Metrics + Prometheus endpoints |
| `coordinator/jobs/prometheus_metrics.py` | Custom ORM-backed collector |
| `coordinator/jobs/storage.py` | Dual NFS/S3 storage backend |
| `coordinator/jobs/presigned.py` | Presigned URL generation |

---

## 5. Adding a New Feature

Use this checklist when adding a new opt-in feature module.

1. **Place the module under `ocr_local/features/`** if it's an OCR feature, or the appropriate sub-package otherwise.
2. **Add a root-level shim** (`<feature>.py`) that re-exports `sys.modules[__name__] = importlib.import_module("ocr_local.features.<feature>")` for backward compatibility.
3. **Add a feature flag** to `pipeline_config.py` with `default=False`.
4. **Wire the flag check** at the integration point in `ocr_gpu_async.py` (typically the assembler thread or a GPU worker hook).
5. **Add a JSON Schema** in `schemas/<feature>.schema.json` if the feature emits sidecar output.
6. **Add a custody event type** in `custody.py` if the feature has audit relevance.
7. **Add unit tests** in `tests/test_<feature>.py`.
8. **Add an integration test** in `tests/test_pipeline_<feature>.py` that runs the pipeline with the flag enabled and asserts the sidecar.
9. **Document the feature** in [`docs/06-CONFIGURATION-REFERENCE.md`](docs/06-CONFIGURATION-REFERENCE.md) and add to the relevant capability matrix in [`README.md`](README.md).
10. **Update the changelog** with the feature under "Added".

---

## 6. Adding a New Language

1. **Add the language entry** to `ocr_local/config/language_config.py` with the appropriate tier (`core` or `extended`).
2. **Confirm PaddleOCR supports the language** — check the model list in [PaddleOCR's docs](https://github.com/PaddlePaddle/PaddleOCR).
3. **Add the model name** to the registry in `download_models.py` if it needs custom handling.
4. **Rebuild the Docker image** to download the new model.
5. **Add a fixture document** to `tests/fixtures/languages/<code>.pdf` (optional but recommended).
6. **Add a test** in `tests/test_language_<code>.py` (optional).
7. **Document the addition** in [`README.md`](README.md) language table.

---

## 7. Adjusting Performance

Edit constants at the top of `ocr_gpu_async.py`:

```python
NUM_EXTRACTORS = 8       # CPU image extraction threads
NUM_WORKERS = 12         # GPU OCR worker threads (tune by VRAM)
NUM_COMPRESSORS = 8      # Ghostscript compression threads
IMAGE_QUEUE_SIZE = 200   # RAM buffer (~200 * 20 MB = 4 GB)
CHUNK_QUEUE_SIZE = 50    # Scheduler to extractor buffer
DPI = 300                # Image resolution for OCR
```

VRAM guidance:

| VRAM | NUM_WORKERS |
|---|---|
| 16 GB | 8-12 |
| 24 GB | 12-16 |
| 48 GB | 16-24 |

For Kubernetes, set the same via `values.yaml`:

```yaml
gpuWorker:
  threads: 12
  imageQueueSize: 200
```

See [`docs/hardware-tuning-profiles.md`](docs/hardware-tuning-profiles.md) for benchmarked profiles.

---

## 8. Testing Locally

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Targeted Tests

```bash
python -m pytest tests/test_ner.py -v
python -m pytest -k "test_resume" -v
```

### Smoke Test the Pipeline

```bash
python scripts/smoke_pipeline.py
```

### End-to-End with Real Fixture

```bash
mkdir -p ocr_source ocr_output
cp tests/fixtures/sample.pdf ocr_source/
python ocr_gpu_async.py
# Output: ocr_output/EXPORT/PDF/sample.pdf
```

### Coordinator Tests

```bash
cd coordinator
python -m pytest tests/ -v
```

The coordinator suite needs Django settings configured. Most root tests gate Django imports with `pytest.importorskip('django')`.

### Lint

```bash
ruff check .
```

Pre-commit will also catch lint issues automatically.

---

## 9. Common Gotchas

A curated list. The full developer "lessons learned" file accumulates and is not all reproduced here — these are the ones you'll hit.

### Gotchas — Pipeline

1. **`.gitignore` has `*.txt`** — Any new `requirements-*.txt` must be explicitly excepted in `.gitignore`.
2. **PaddleOCR 2.9.1 is the production version.** `paddlex` is not used. `use_onnx=True` enables ONNX Runtime; `enable_hpi` is PaddleOCR 3.x only.
3. **`get_file_hash` is an alias for `get_path_based_doc_id`.** The function hashes the file path (for crash-resume directory naming), not file contents.
4. **`doc_registry` requires `_doc_registry_lock` (RLock).** All 15 access sites across extractor, worker, assembler, and monitor threads hold the lock.
5. **Subprocess timeouts enforced.** Ghostscript 300s, Tesseract 120s, Poppler 300s. On timeout, the page enters the failure pipeline.

### Gotchas — Configuration

6. **Feature flags default OFF.** Translation (`ENABLE_TRANSLATION=False`), per-span language (`ENABLE_PER_SPAN_LANGUAGE=False`), handwriting MT (`ENABLE_HANDWRITING_MT=False`).
7. **CPU worker builds need `--cpu-only` flag.** `download_models.py --cpu-only` forces `use_gpu=False` and `use_onnx=True` to avoid CUDA probe segfault.
8. **`OCR_TASK_ROUTING` env var** (`gpu`/`cpu`/`auto`) controls which queue OCR tasks dispatch to. Default is `gpu`. `auto` queries Worker model for online GPU workers.
9. **`OCR_ENGINE_SELECTION` env var** (`auto`/`paddle`/`tesseract`) controls per-page routing. Default is `paddle`.

### Gotchas — API + Tests

10. **Test auth must patch both `api.config` AND `api.auth` module vars.** `api.auth` caches config at import time.
11. **Coordinator tests need `coordinator/` on `sys.path`** or must run from `coordinator/`. Otherwise `coordinator.settings_test` import fails.
12. **`coordinator.jobs` import requires Django settings configured.** Root tests should gate with `importlib.util.find_spec('django')` or `pytest.importorskip`.
13. **conftest autouse fixtures may import api.job_manager / api.database lazily.** Wrap in `try/except ImportError` for SDK-only test runs.

### Gotchas — Operations

14. **Helm chart requires manual secrets.** `secrets.*` values must be overridden via `--set` or `values-secret.yaml`.
15. **Redis Sentinel and PostgreSQL backup are opt-in.** Defaults are `false`; enable in production overlay.
16. **`ocr_metrics.py` is standalone.** Pipeline modules import Prometheus counters without Django. Optional `prometheus_client` import with no-op fallback.
17. **Helm `appVersion` in `Chart.yaml` drifts from `version.py`.** Bump both together.

### Gotchas — Windows

18. **`nul` device file creation** can break `git add -A`. Always use specific file paths.
19. **Console encoding** — set `PYTHONIOENCODING=utf-8` for Unicode output (✓, ✗, → glyphs).

### Gotchas — Testing

20. **Pre-existing flaky tests:** `test_pipeline_records_tenant_processing_seconds`, `test_build_cost_summary_uses_provider_agnostic_rates`. Not blockers — retry.
21. **Grafana dashboard panel count is 53.** Five test files assert this. Update all five when adding new panels.
22. **NFS-to-S3 dry-run output uses `(projected)` labels.** Tests asserting bare `Files to upload:` will fail.

---

## 10. Style and Conventions

### Python

- **PEP 8** with `ruff` enforcement.
- **`from __future__ import annotations`** at the top.
- **Type hints** on public APIs.
- **Module-level constants** in ALL_CAPS.
- **No emojis** in code, comments, or commit messages.
- **`logger = logging.getLogger(__name__)`** at module top.
- **`logger.info("text %s", value)`** — lazy %s, never f-strings in log calls (so disabled log levels stay cheap).

### TypeScript

- **Strict mode** enabled.
- **ESM modules**, no CommonJS.
- **No `any`** unless boundary is genuinely untyped.
- **Prettier** for formatting.

### Markdown

- **Plain English.** Define jargon on first use.
- **Mermaid** for diagrams.
- **No emojis** in documentation.

### Commits

- Conventional Commits format: `type(scope): summary`
- Wrap message body at 72 columns.
- No AI co-author footers.

### Git

- Cut feature branches from **fresh `origin/main`**:
  ```bash
  git fetch origin
  git checkout -b feat/xxx origin/main
  ```
- Never reuse a merged branch. (Branch tracking can mislead about state.)
- Squash-merge PRs to `main`.
- Tag releases as `v<MAJOR>.<MINOR>.<PATCH>` (e.g. `v4.1.0`).

---

## Further Reading

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — top-level architecture
- [`docs/00-SYSTEM-BLUEPRINT.md`](docs/00-SYSTEM-BLUEPRINT.md) — system blueprint
- [`docs/06-CONFIGURATION-REFERENCE.md`](docs/06-CONFIGURATION-REFERENCE.md) — every env var
- [`docs/09-TROUBLESHOOTING.md`](docs/09-TROUBLESHOOTING.md) — runtime issues
- [`docs/benchmarking-methodology.md`](docs/benchmarking-methodology.md) — performance benchmarks
