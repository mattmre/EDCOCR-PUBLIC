# EDCOCR — Technical White Paper

**Forensic-Grade OCR for Electronic Discovery and Regulated Document Workflows**

**Version**: 4.1.0  ·  **License**: Apache 2.0  ·  **Date**: 2026-05-20

---

## Abstract

EDCOCR is an open-source, GPU-accelerated OCR platform engineered for workflows where every page must be accounted for and every extraction must be defensible. It combines CTC (Connectionist Temporal Classification) recognition — which cannot synthesize characters — with hash-chained custody logging, deterministic fallback behavior, and forensic image preservation, producing outputs suitable for legal discovery, regulated industries, and air-gapped environments.

The platform processes documents through a six-stage producer-consumer pipeline (31 concurrent threads) and ships in three deployment topologies: single-machine Docker, distributed Celery clusters, and Kubernetes via a 26-template Helm chart. It supports 45 languages with per-span detection, integrates with NFS/S3/Azure/GCS storage, and emits 14 structured output families plus a tamper-evident audit trail.

This document describes the design philosophy, system architecture, and operational characteristics that distinguish EDCOCR from generative-model OCR tools and best-effort cloud APIs.

---

## 1. Motivation

### 1.1 The Forensic Problem with Modern OCR

The dominant OCR tools of the 2020s have converged on generative architectures — encoder-decoder transformers that condition on image features and emit text tokens. These models achieve excellent accuracy on clean print, but they introduce three failure modes that are unacceptable for forensic workflows:

1. **Hallucination.** A generative recognizer can emit characters that were never on the page. There is no mathematical bound on what text it can produce given an image.
2. **Silent failure.** When the model is uncertain, it still emits the most likely token. There is rarely a meaningful "I cannot read this" signal exposed to the consumer.
3. **Opaque provenance.** Cloud APIs hide which model version processed which page, when, with what parameters. Reproducibility — the foundation of evidence — becomes impossible.

For workflows that produce evidence consumed by courts, regulators, or auditors, these failure modes are not acceptable.

### 1.2 The "Air-Gap" Problem

A significant fraction of regulated document processing occurs in environments where outbound network calls are contractually or legally prohibited:

- Classified government work (SCIF environments)
- Healthcare records subject to HIPAA-class restrictions
- Legal discovery under protective order
- Financial workflows touching customer PII

Cloud OCR APIs are non-starters for these workflows. EDCOCR's runtime never makes outbound calls; all models ship pre-baked into Docker images.

### 1.3 Audit, Reproducibility, and Custody

When OCR output is contested — by a litigant, a regulator, or an internal reviewer — the platform must answer four questions definitively:

1. **What was the original input?** (Cryptographic hash of the source file.)
2. **What model processed each page, with what parameters?** (Custody event per operation.)
3. **Did anyone tamper with the output between extraction and review?** (Hash-chained audit log.)
4. **Where did this specific character come from?** (Page-level confidence + DPI + engine.)

EDCOCR answers all four for every job, with no operator action required.

---

## 2. Design Principles

### 2.1 CTC-Only Recognition

EDCOCR uses PaddleOCR 2.9.1 with CTC decoding as the primary engine. CTC heads emit per-frame character probabilities and use a deterministic collapse function to map them to text. The decoder **cannot synthesize tokens** that have no probability mass somewhere in the input — a property that generative decoders do not share.

The fallback chain is also generative-free:
1. **PaddleOCR** (CTC) — primary
2. **Tesseract** (LSTM + CTC) — fallback
3. **Image-only preservation** — final fallback

There is no generative LLM substitution in the core OCR loop. The opt-in translation suite (Section 7) is the only place generative models touch the pipeline, and it is gated by feature flags, custody events, and per-tenant policy.

### 2.2 Forensic Image Preservation

If all OCR engines fail on a page, EDCOCR does **not** discard it. Instead, the rasterized page image is embedded directly into the final PDF, ensuring the document is always recoverable. The custody log records the failure with the original image hash, so reviewers know which pages need manual handling.

This is the inverse of cloud OCR APIs, which typically drop unreadable pages from output.

### 2.3 Hash-Chained Custody Log

Every operation — file ingestion, page OCR completion, language detection, classification, transform, stamp, output finalization — emits a custody event. Each event records:

- Operation type (e.g., `PAGE_OCR_OK`, `LANGUAGE_DETECTED`, `BATES_STAMP_APPLIED`)
- Timestamp (ISO 8601)
- Job ID, page index, tenant ID
- Operation-specific payload (model version, DPI, confidence, bbox)
- SHA-256 of the previous event

Tampering with any event breaks the hash chain at that point and every event after, making post-hoc modification detectable.

The log is written as JSONL (newline-delimited JSON) — append-only, easy to ship to external SIEM, and trivially diffable.

### 2.4 Deterministic Behavior

Given the same input file, model versions, and configuration, EDCOCR produces byte-identical output (modulo timestamps in the custody log). This is enforced through:

- Fixed PaddleOCR model checkpoints (no live model swaps)
- Pinned dependency versions
- Deterministic page chunking (path-hash based)
- Stable language detection (FastText with fixed seed)

### 2.5 Backpressure as a Feature

The pipeline uses bounded queues at every stage. When a downstream stage falls behind, upstream stages block rather than dropping work. This trades latency for guaranteed completion — the appropriate trade for batch document workflows.

---

## 3. Architecture

### 3.1 Producer-Consumer Pipeline

The monolithic pipeline runs six stages concurrently:

| Stage | Threads | Function |
|---|---:|---|
| 1. Scheduler | 1 | Scans `ocr_source/`, creates chunk tasks, handles page-level resume |
| 2. CPU Extractors | 8 | PDF/image → PIL images at 300 DPI |
| 3. GPU Workers | 12 | PaddleOCR → Tesseract fallback → image-only fallback |
| 4. Assembler | 1 | Collects pages via document registry, merges into final PDF + sidecars |
| 5. Compressors | 8 | Ghostscript optimization with integrity validation |
| 6. Monitor | 1 (daemon) | Real-time metrics + healthcheck heartbeat |

Queues:
- `chunk_queue` — Scheduler → Extractors
- `image_queue` — Extractors → Workers (size 200, ≈ 4 GB RAM budget)
- `assembly_queue` — Workers → Assembler
- `compression_queue` — Assembler → Compressors

This is implemented in `ocr_gpu_async.py`. The threading model (not asyncio) is intentional — the workload is CPU/GPU-bound, not I/O-bound, and threading minimizes context-switching overhead.

### 3.2 Distributed Topology

For horizontal scale, the pipeline runs as a Django coordinator + Celery worker fleet:

```
┌──────────────────────────────────────────────────────┐
│  FastAPI / Django Coordinator                        │
│  - Job submission API                                │
│  - Job state tracking (PostgreSQL)                   │
│  - WebSocket / SSE progress streaming                │
└────┬─────────────────────────────────────────────────┘
     │
     ├─→ RabbitMQ (quorum queues)
     │    ├─ ocr_gpu (or ocr_gpu_{0..N-1} for per-GPU affinity)
     │    ├─ ocr_cpu
     │    └─ ocr_nlp
     │
     ├─→ Redis (cache, Sentinel for HA, Streams for context windowing)
     │
     ├─→ PostgreSQL (job state, audit metadata)
     │
     └─→ Worker fleet
          ├─ GPU OCR workers (PaddleOCR + Tesseract)
          ├─ CPU OCR workers (ONNX Runtime, 4-7× speedup on CPU)
          └─ NLP workers (NER, classification, extraction)
```

Workers register their capabilities on startup; the coordinator routes tasks based on registered capability. KEDA autoscaling tracks queue depth and scales worker replica counts dynamically.

### 3.3 Storage Abstraction

The platform supports three storage backends through a unified interface:

| Backend | Use Case | Worker Access Pattern |
|---|---|---|
| Local filesystem | Single-machine, testing | Direct path |
| NFS | Multi-worker, on-prem | Shared mount |
| S3 / MinIO | Cloud-native, credential-free workers | Presigned URLs |

Workers in S3 mode receive presigned URLs from the coordinator and never see AWS credentials, simplifying multi-tenant deployments.

### 3.4 Observability

| Layer | Component | Purpose |
|---|---|---|
| Metrics | Prometheus (`/api/v1/prometheus/`) | 7 metric families, per-tenant cost/SLA gauges |
| Dashboards | Grafana (53 panels) | Queue depth, throughput, p50/95/99, GPU/CPU, fleet status |
| Alerts | PrometheusRule (5 rules) | Queue saturation, error rate, worker offline |
| Traces | OpenTelemetry | End-to-end job span propagation |
| Logs | Structured JSON | Stdout (Docker) + per-job NDJSON |
| Audit | Hash-chained JSONL | Custody events, immutable |

---

## 4. Output Schema

Every job produces a structured set of artifacts under `ocr_output/EXPORT/<family>/<subfolder>/<document>.<ext>`:

| Family | Format | Contents |
|---|---|---|
| `PDF/` | PDF | Searchable PDF with embedded OCR text layer |
| `TEXT/` | .txt | Plain text extraction |
| `STRUCTURE/` | JSON | Layout regions, table cells (HTML), figure positions |
| `NER/` | JSON | Named entities — CASE_NUMBER, BATES_NUMBER, PERSON, ORG |
| `ENTITIES/` | JSON | Consolidated entity index across pages |
| `CLASSIFICATION/` | JSON | Document type with confidence |
| `EXTRACTION/` | JSON | Structured fields — dates, amounts, addresses |
| `HANDWRITING/` | JSON | Handwriting regions with confidence |
| `SIGNATURE/` | JSON | Signature detection (advisory) |
| `VERTICAL/` | JSON | CJK vertical text reading order |
| `LANGUAGE/` | JSON | Per-span language with BCP-47 tags |
| `VALIDATION/` | JSON | Per-page confidence + quality classification |
| `RETRIEVAL/` | JSON + MD | Unified retrieval format for downstream search |
| Custody log | JSONL | Hash-chained audit events |

Each family has a JSON Schema definition in `schemas/`, ensuring consumers can validate output structure.

---

## 5. Languages

EDCOCR ships with a tiered language registry — 45 languages organized into two tiers:

**Core (34 languages, default, air-gapped ready)**
- Latin: en, fr, de, es, it, pt, nl, sv, da, fi, ro, pl, cs, hu, tr, vi
- Cyrillic: ru, uk, be, bg
- CJK: ch (Simplified), chinese_cht (Traditional), japan, korean
- Arabic / RTL: ar, fa, ur, ug
- Indic: hi, ta, te, kn
- Other: el (Greek), ka (Georgian)

**Extended (+11 languages, opt-in via `OCR_LANGUAGE_TIERS=core,extended`)**
- Latin: hr, sk, no, lt, lv, et, rs_latin
- Indic: bn (Bengali), mr (Marathi), ne (Nepali)
- Southeast Asian: th (Thai)

Language selection is two-pass: FastText runs first on a quick text sample, the best-matching PaddleOCR model is loaded, and detection re-runs on extracted text to catch mixed-script pages. Per-span detection (opt-in via `ENABLE_PER_SPAN_LANGUAGE=true`) emits a `.language.json` sidecar with BCP-47 tags at line, page, and document levels.

---

## 6. Security Posture

| Control | Implementation |
|---|---|
| Container hardening | All 6 Dockerfiles non-root (UID 1000), capabilities dropped, seccomp default |
| Authentication | X-API-Key header (default), optional OIDC Bearer |
| Rate limiting | slowapi per-endpoint, per-tenant limits |
| Per-tenant isolation | Tenant ID flows through every API, DB filter, and output path |
| Webhook secrets | Fernet-encrypted at rest, key from `WEBHOOK_SECRET_KEY` |
| SSRF protection | Outbound URL allowlist; metadata IPs blocked |
| Subprocess timeouts | Ghostscript 300s, Tesseract 120s, Poppler 300s |
| OpenAPI docs gating | `EXPOSE_API_DOCS=false` by default; prevents endpoint enumeration |
| Audit logging | Hash-chained JSONL; masked API key in request audit log |
| Database security | SQLite 0o600 perms; PostgreSQL backup CronJob; Redis Sentinel HA |
| Request size limit | 413 on oversized non-multipart bodies, configurable |
| WS idle timeout | Idle connection close; max connection age cap |

---

## 7. Translation Suite (Opt-In)

EDCOCR includes a pluggable translation layer that operates downstream of OCR. All translation is feature-gated and defaults OFF.

### 7.1 Engine Adapters

| Engine | License | Default |
|---|---|---|
| OPUS-MT | Apache 2.0 | Recommended for commercial use |
| MADLAD-400 | Apache 2.0 | Recommended quality tier |
| NLLB-200 | CC-BY-NC-4.0 | Opt-in via `allow_nc_licensed=True` |
| LLM tier (Ollama / vLLM / OpenAI-compatible) | varies | Opt-in, operator-managed endpoints |

### 7.2 Policy Controls

- Per-tenant `allow_nc_licensed` flag (NC license routing)
- Privilege-blocked content (legal privilege keywords) never routes to cloud generative engines
- Custody event on every engine selection (`MODEL_SELECTED`) and every translation (`TRANSLATION_COMPLETED` or `TRANSLATION_REJECTED`)
- Certification gated behind strong-auth review (PIV/CAC, OIDC MFA, hardware token)
- Generative translations are tagged `is_generative=true` in the sidecar

### 7.3 Forensic Integrity

The forensic-core (OCR pipeline) and AI-adjacent (translation, summarization) capability classes are separated by a contract documented at [architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md). A boundary validator (`scripts/validate_feature_boundary.py`) enforces:

- Language detection isolation from translation routing
- Certified-translation filter at write time
- Plugin signature validation

---

## 8. Deployment

### 8.1 Single Machine (Docker)

```bash
git clone https://github.com/mattmre/EDCOCR-PUBLIC.git
cd EDCOCR-PUBLIC
docker compose up -d --build
```

Source documents go in `ocr_source/`; results land in `ocr_output/EXPORT/`.

### 8.2 Kubernetes (Helm)

```bash
helm install edcocr helm/ocr-local/ \
  -f values-secret.yaml \
  --namespace edcocr \
  --create-namespace
```

The chart ships 26 templates including:
- GPU/CPU worker deployments with KEDA autoscalers
- PostgreSQL + RabbitMQ + Redis StatefulSets (with optional Sentinel HA)
- Prometheus ServiceMonitor + PrometheusRule + Grafana dashboard ConfigMap
- PodDisruptionBudgets, NetworkPolicies, and a configurable Ingress

### 8.3 Air-Gapped

```bash
./scripts/airgap-bundle.sh    # On a connected host
# (transfer the bundle to the isolated host)
./scripts/airgap-deploy.sh    # On the isolated host
```

The bundle includes all Docker images, pre-baked models (PaddleOCR + FastText + LayoutLMv3), and Helm charts. No outbound calls are made at runtime.

### 8.4 Multi-Cloud (Terraform)

Terraform modules ship for AWS EKS, GCP GKE, and Oracle OKE under `terraform/`. Each module provisions the cluster, configures KEDA, and renders the Helm release with environment-specific values.

---

## 9. Performance Characteristics

| Configuration | GPU Workers | Notes |
|---|---:|---|
| 16 GB VRAM | 8–12 | Default for consumer GPUs (RTX 4080 / A5000) |
| 24 GB VRAM | 12–16 | RTX 4090 / A6000 sweet spot |
| 48 GB VRAM | 16–24 | A100 / H100 maximum density |

CPU-only deployments (ONNX Runtime backend) achieve 4–7× speedup over the PaddleOCR default Python backend, making meaningful throughput available without GPU hardware.

Adaptive DPI escalation auto-retries low-confidence pages at 450 and 600 DPI; adaptive batch sizing scales batch dimensions to page complexity.

---

## 10. What EDCOCR Is Not

To set expectations clearly:

- **Not a document review platform.** EDCOCR extracts text and structure; downstream review happens in your platform of choice.
- **Not a generative document understanding system.** There is no "summarize this contract" endpoint in the core. Optional VLM and LLM integrations exist behind feature flags, but they are sidecars, not the main loop.
- **Not a hosted service.** EDCOCR is software you run. There is no SaaS endpoint operated by the project maintainers.
- **Not Cloud-API-equivalent.** Cloud OCR APIs may have higher accuracy on specific document types where they have trained extensively. EDCOCR optimizes for defensibility, auditability, and air-gap deployment instead.

---

## 11. Versioning and Stability

EDCOCR follows semantic versioning. The 2.x line guarantees REST API compatibility within the major version. Output schemas are versioned independently, and migration guides accompany schema-breaking releases.

The SDK packages (`edcocr-sdk` for Python, `@edcocr/sdk` for TypeScript) are versioned in lockstep with the server.

See [SDK-VERSIONING-POLICY.md](SDK-VERSIONING-POLICY.md) and [api-stability-contract.md](api-stability-contract.md) for the complete policy.

---

## 12. References

- [00-SYSTEM-BLUEPRINT.md](00-SYSTEM-BLUEPRINT.md) — System overview
- [01-TECH-STACK-DNA.md](01-TECH-STACK-DNA.md) — Tech stack
- [03-INFORMATION-FLOWS.md](03-INFORMATION-FLOWS.md) — Data flow diagrams
- [architecture/pipeline-design.md](architecture/pipeline-design.md) — Pipeline internals
- [architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md) — Forensic-AI boundary
- [API-REFERENCE.md](API-REFERENCE.md) — Full API reference
- [forensic/evidence-bundle-specification.md](forensic/evidence-bundle-specification.md) — Evidence bundle spec
- [compliance/](compliance/) — Compliance readiness matrices

---

*EDCOCR is open source under the Apache 2.0 license. The project is published at https://github.com/mattmre/EDCOCR-PUBLIC. Contributions welcome — see [CONTRIBUTING.md](../CONTRIBUTING.md).*
