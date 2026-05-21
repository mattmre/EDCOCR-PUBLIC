# EDCOCR Architecture

This document is the canonical top-level architecture reference for EDCOCR. Read it first if you are evaluating, integrating, or extending the platform.

For deeper-dive material, see [`docs/00-SYSTEM-BLUEPRINT.md`](docs/00-SYSTEM-BLUEPRINT.md) (system blueprint) and [`docs/03-INFORMATION-FLOWS.md`](docs/03-INFORMATION-FLOWS.md) (information flows).

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [System Overview](#2-system-overview)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Deployment Topologies](#4-deployment-topologies)
5. [Data Model](#5-data-model)
6. [Storage Architecture](#6-storage-architecture)
7. [Observability](#7-observability)
8. [Security Architecture](#8-security-architecture)
9. [Chain of Custody](#9-chain-of-custody)
10. [Failure Modes](#10-failure-modes)

---

## 1. Design Principles

EDCOCR's architecture is shaped by five non-negotiables:

| Principle | What It Means In Practice |
|---|---|
| **Zero hallucination** | CTC-only recognition. Generative AI never touches the recognition path. If it generates a character that wasn't there, it's not OCR — it's storytelling. |
| **Preserve the evidence** | OCR failure never deletes the source. A blank-text page falls back to image-only embedding, and a custody event records why. |
| **Deterministic recovery** | Page-level temp files mean the pipeline can crash anywhere and resume without rework. No "lost half a batch." |
| **Tamper-evident by construction** | SHA-256 hash-chained JSONL custody log. Append-only, replayable, signature-verifiable. |
| **Operable at any size** | Same code path from a laptop Docker run to a multi-cluster Kubernetes federation. |

These principles are load-bearing. Most architectural choices below are direct consequences of one or more of them.

---

## 2. System Overview

```mermaid
flowchart TB
    subgraph Clients["Clients"]
        C1[Python SDK]
        C2[TypeScript SDK]
        C3[REST API direct]
        C4[Webhook consumers]
    end

    subgraph Ingestion["Ingestion Layer"]
        API[FastAPI<br/>REST + WebSocket + SSE]
        Watcher[File Watcher<br/>local + FTP/SFTP]
        Object[Object Storage<br/>S3 · MinIO · Azure · GCS]
    end

    subgraph Coordination["Coordination Layer"]
        Coord[Django Coordinator]
        DB[(PostgreSQL)]
        Broker[(RabbitMQ)]
        Redis[(Redis<br/>Sentinel HA)]
    end

    subgraph Workers["Worker Layer"]
        WG[GPU OCR Workers]
        WC[CPU OCR Workers<br/>ONNX]
        WN[NLP Workers<br/>NER · UIE]
        WX[Compression Workers]
    end

    subgraph Output["Output Layer"]
        OutPDF[Searchable PDFs]
        OutTxt[Plain Text]
        OutSide[Sidecar JSONs<br/>NER · Tables · Classification]
        Custody[Custody Log<br/>JSONL hash chain]
    end

    subgraph Observability["Observability"]
        Prom[Prometheus]
        Graf[Grafana<br/>53 panels]
        Trace[OpenTelemetry]
    end

    C1 --> API
    C2 --> API
    C3 --> API
    Watcher --> API
    Object --> API
    API --> Coord
    Coord <--> DB
    Coord <--> Broker
    Coord <--> Redis
    Broker --> WG
    Broker --> WC
    Broker --> WN
    Broker --> WX
    WG --> Output
    WC --> Output
    WN --> Output
    WX --> OutPDF
    Output -.->|metrics| Prom
    Prom --> Graf
    Coord -.->|spans| Trace
    Output -->|completion| C4

    style Workers fill:#10b981,stroke:#065f46,color:#fff
    style Output fill:#f59e0b,stroke:#92400e,color:#fff
    style Custody fill:#ef4444,stroke:#7f1d1d,color:#fff
    style Observability fill:#8b5cf6,stroke:#4c1d95,color:#fff
```

### Component Roles

| Layer | Component | Responsibility |
|---|---|---|
| **Ingestion** | FastAPI | Accept jobs via REST/WebSocket, authenticate, validate, rate-limit, route. |
| | File Watcher | Poll mounted directories or FTP/SFTP shares for new documents. |
| | Object Storage | Stage documents from S3/MinIO/Azure/GCS via presigned URLs. |
| **Coordination** | Django Coordinator | Job lifecycle, worker registry, capability routing, admin UI. |
| | PostgreSQL | Durable storage for jobs, pages, custody events, tenants. |
| | RabbitMQ | Task queue with capability-based routing (`ocr_gpu_*`, `ocr_cpu`, `nlp`, etc.). |
| | Redis (+ Sentinel) | Job state cache, Celery result backend, rate-limit counters, Redis Streams for context windowing. |
| **Workers** | GPU OCR | PaddleOCR + Tesseract fallback + optional Document Intelligence. |
| | CPU OCR | ONNX Runtime / OpenVINO backend for CPU-only deployments. |
| | NLP | spaCy NER, PaddleNLP UIE, classification ensemble. |
| | Compression | Ghostscript optimization with integrity validation. |
| **Output** | Searchable PDFs | OCR text layer embedded in original PDF. |
| | Sidecar JSONs | One file per enrichment (NER, structure, classification, etc.). |
| | Custody Log | Hash-chained JSONL audit trail. |

---

## 3. Pipeline Architecture

The OCR pipeline is a **6-stage async producer-consumer model** with 31 threads.

```mermaid
flowchart LR
    subgraph S1["Stage 1<br/>Scheduler · 1 thread"]
        SCH[Scan files<br/>Create chunks<br/>Resume from temp]
    end
    subgraph S2["Stage 2<br/>Extractors · 8 threads"]
        EX[PDF/Image to PIL<br/>300 DPI]
    end
    subgraph S3["Stage 3<br/>GPU Workers · 12 threads"]
        GW[PaddleOCR<br/>Tesseract fallback<br/>Image-only fallback<br/>+ Document Intelligence]
    end
    subgraph S4["Stage 4<br/>Assembler · 1 thread"]
        AS[Merge pages<br/>Build PDF<br/>Write sidecars]
    end
    subgraph S5["Stage 5<br/>Compressors · 8 threads"]
        CO[Ghostscript<br/>Integrity check]
    end
    subgraph S6["Stage 6<br/>Monitor · 1 daemon"]
        MO[PPM metrics<br/>Heartbeat<br/>Healthcheck]
    end

    SCH -->|chunk_queue| EX
    EX -->|image_queue 200| GW
    GW -->|page_queue| AS
    AS -->|compression_queue| CO
    CO --> Output[ocr_output/EXPORT/PDF/]
    MO -.->|observe| SCH
    MO -.->|observe| EX
    MO -.->|observe| GW
    MO -.->|observe| AS
    MO -.->|observe| CO

    style S3 fill:#10b981,color:#fff
    style S4 fill:#3b82f6,color:#fff
    style S6 fill:#8b5cf6,color:#fff
```

### Thread Counts and Queues

| Stage | Threads | Queue Size | Purpose |
|---|---|---|---|
| Scheduler | 1 | — | File enumeration, chunk task creation |
| Extractors | 8 | 50 (chunk) | CPU-bound PDF/image to 300 DPI PIL conversion |
| GPU Workers | 12 | 200 (image) | OCR recognition + optional layout/table analysis |
| Assembler | 1 | — (registry) | Merge pages, write outputs, emit custody events |
| Compressors | 8 | — (FIFO) | Ghostscript optimization with integrity validation |
| Monitor | 1 daemon | — | Real-time metrics, heartbeat, healthcheck |

Thread counts are tunable via environment variables. See [`docs/06-CONFIGURATION-REFERENCE.md`](docs/06-CONFIGURATION-REFERENCE.md).

### Page Lifecycle

```mermaid
sequenceDiagram
    participant FS as File System
    participant SCH as Scheduler
    participant EX as Extractor
    participant W as GPU Worker
    participant AS as Assembler
    participant CO as Compressor
    participant OUT as Output

    FS->>SCH: new file detected
    SCH->>SCH: hash path → doc_id
    SCH->>FS: check ocr_temp/doc_id/
    Note over SCH: Resume gap detection
    SCH->>EX: chunk task (pages 1-N)
    EX->>EX: pdf to PIL @ 300 DPI
    EX->>W: page image
    W->>W: PaddleOCR recognition
    alt OCR succeeded
        W->>AS: page result + text
    else OCR failed, image preserved
        W->>AS: page result (image-only)
        W-->>OUT: custody event: OCR_FAILED_IMAGE_PRESERVED
    end
    AS->>AS: write per-page PDF to ocr_temp/
    AS->>AS: all pages in? merge!
    AS->>CO: final PDF
    CO->>CO: Ghostscript compress
    CO->>OUT: ocr_output/EXPORT/PDF/
    AS-->>OUT: sidecar JSONs
    AS-->>OUT: custody event: DOC_COMPLETE
```

### Resume Semantics

```mermaid
flowchart LR
    Start[Process restart] --> Scan[Scan ocr_temp/]
    Scan --> Detect[For each doc_id, list existing pages]
    Detect --> Gap[compute_resume_gap_chunks<br/>total - existing = missing]
    Gap --> Queue[Enqueue only missing chunks]
    Queue --> Continue[Pipeline resumes mid-job]

    style Start fill:#ef4444,color:#fff
    style Continue fill:#10b981,color:#fff
```

`compute_resume_gap_chunks(total_pages, existing_pages, chunk_target_size)` is the canonical helper. It uses explicit set difference and is covered by 34 regression tests in `tests/test_gap_detection.py`.

---

## 4. Deployment Topologies

EDCOCR supports four deployment patterns. Same code, different orchestration.

### 4.1 Single-Host Docker (Development & Small Production)

```mermaid
flowchart TB
    User[Operator] -->|docker compose up| Host
    subgraph Host["Single Host"]
        direction TB
        Compose[docker-compose.yml]
        Compose --> Coord[Coordinator]
        Compose --> Workers[GPU + CPU + NLP workers]
        Compose --> PG[(PostgreSQL)]
        Compose --> RMQ[(RabbitMQ)]
        Compose --> RD[(Redis)]
        Workers -.->|/dev/nvidia| GPU[GPU passthrough]
    end
    Host --> Storage[Local volumes<br/>ocr_source · ocr_output]

    style Workers fill:#10b981,color:#fff
```

**When to use:** Development, single-tenant on-premises, processing volumes under ~50k pages/day.

### 4.2 Distributed Coordinator + Worker Fleet

```mermaid
flowchart TB
    subgraph CoordHost["Coordinator Host"]
        API[FastAPI]
        DJ[Django Admin]
        PG[(PostgreSQL)]
        RMQ[(RabbitMQ)]
        RD[(Redis)]
    end

    subgraph WG1["GPU Worker Host 1"]
        W1G[GPU Workers]
    end
    subgraph WG2["GPU Worker Host 2"]
        W2G[GPU Workers]
    end
    subgraph WC1["CPU Worker Pool"]
        W1C[CPU OCR + NLP]
    end

    API <--> RMQ
    DJ <--> PG
    RMQ <--> W1G
    RMQ <--> W2G
    RMQ <--> W1C
    PG <--> W1G
    PG <--> W2G
    PG <--> W1C
    RD <--> W1G
    RD <--> W2G

    Storage[NFS · S3 · Azure · GCS]
    W1G <--> Storage
    W2G <--> Storage
    W1C <--> Storage
    API <--> Storage

    style CoordHost fill:#3b82f6,color:#fff
    style WG1 fill:#10b981,color:#fff
    style WG2 fill:#10b981,color:#fff
    style WC1 fill:#10b981,color:#fff
```

**When to use:** Mid-volume production, mixed GPU+CPU fleet, single-region deployment.

### 4.3 Kubernetes (KEDA Autoscaled)

```mermaid
flowchart TB
    subgraph Cluster["Kubernetes Cluster"]
        subgraph CP["Control Plane Namespace"]
            CoordD[Coordinator Deployment]
            CeleryD[Celery Beat + Coordinator Worker]
            FlowerD[Flower]
        end
        subgraph WP["Worker Namespace"]
            GPUWD[GPU Worker Deployment<br/>nvidia.com/gpu]
            CPUWD[CPU Worker Deployment]
            CPUOCRWD[CPU OCR Worker<br/>ONNX backend]
        end
        subgraph DP["Data Plane"]
            PGSS[PostgreSQL StatefulSet]
            PGBackup[Backup CronJob]
            RMQSS[RabbitMQ StatefulSet<br/>Quorum queues]
            RedisSS[Redis + Sentinel]
        end
        subgraph Obs["Observability"]
            SM[ServiceMonitor]
            PR[PrometheusRule<br/>5 alerts]
            GD[Grafana Dashboard<br/>53 panels]
        end
        KEDA[KEDA<br/>scales on queue depth]
    end

    KEDA -.->|scale| GPUWD
    KEDA -.->|scale| CPUWD
    KEDA -.->|scale| CPUOCRWD
    PGBackup -.->|hourly| PGSS

    Ingress[Ingress + NetworkPolicy] --> CoordD
    CoordD <--> PGSS
    CoordD <--> RMQSS
    CoordD <--> RedisSS
    GPUWD <--> RMQSS
    CPUWD <--> RMQSS

    style CP fill:#3b82f6,color:#fff
    style WP fill:#10b981,color:#fff
    style DP fill:#f59e0b,color:#fff
    style Obs fill:#8b5cf6,color:#fff
```

**When to use:** Production at scale, multi-tenant, autoscaling demand, formal HA requirements.

The Helm chart ships 26 templates covering everything above plus PDBs, NetworkPolicies, Ingress, Secrets, ConfigMaps, and KEDA ScaledObjects.

### 4.4 Air-Gapped Deployment

```mermaid
flowchart LR
    subgraph Connected["Connected Environment"]
        Build[docker build] --> Bundle[airgap-bundle.sh<br/>docker save + tar]
        Bundle --> Transport[Encrypted media transfer]
    end
    subgraph AirGap["Air-Gapped Environment"]
        Transport --> Deploy[airgap-deploy.sh<br/>docker load]
        Deploy --> Run[docker compose up]
    end

    style Connected fill:#3b82f6,color:#fff
    style AirGap fill:#ef4444,color:#fff
```

All 45 language models, FastText detector, spaCy models, and ML weights are baked into the Docker image. The runtime never requires outbound network access.

---

## 5. Data Model

```mermaid
erDiagram
    JOB ||--o{ PAGE_RESULT : "produces"
    JOB ||--o{ CUSTODY_EVENT : "emits"
    JOB ||--o{ PII_ENTITY : "extracts"
    JOB }o--|| TENANT : "owned by"
    JOB }o--|| WORKER : "processed by"
    WORKER ||--o{ HEARTBEAT : "reports"

    JOB {
        uuid id PK
        string status
        string priority
        json settings
        string source_path
        string output_dir
        timestamp created_at
        timestamp updated_at
        string tenant_id FK
        string worker_id FK
    }

    PAGE_RESULT {
        uuid id PK
        uuid job_id FK
        int page_number
        float ocr_confidence
        string quality_class
        bool degraded
        int dpi_used
        string engine_used
    }

    CUSTODY_EVENT {
        uuid id PK
        uuid job_id FK
        int seq
        string event_type
        json payload
        string sha256_self
        string sha256_prev
        timestamp ts
    }

    PII_ENTITY {
        uuid id PK
        uuid job_id FK
        int page_number
        string entity_type
        string entity_text
        json bbox
        float confidence
    }

    TENANT {
        string id PK
        json policy
        bool allow_nc_licensed
        bool allow_translation
    }

    WORKER {
        uuid id PK
        string hostname
        json capabilities
        string status
        timestamp last_heartbeat
    }
```

### Hash Chain Detail

Each `CUSTODY_EVENT` carries:

- `sha256_self` — hash of `(prev_hash || event_type || payload || ts)`
- `sha256_prev` — `sha256_self` from the preceding event in the same job

Verifying integrity is a linear walk: recompute each hash, confirm continuity, confirm signatures (if signing is enabled). Any tampering — insertion, deletion, edit — breaks the chain at the point of tampering.

---

## 6. Storage Architecture

EDCOCR supports two storage backends concurrently:

```mermaid
flowchart TB
    Worker[Worker] --> Decide{STORAGE_BACKEND}
    Decide -->|nfs| NFS[NFS Mount<br/>/mnt/ocr/]
    Decide -->|s3| S3[Object Storage<br/>S3 · MinIO · Azure · GCS]

    NFS --> Layout[Standard layout<br/>ocr_source/ · ocr_output/ · ocr_temp/]
    S3 --> Presigned[Presigned URLs<br/>credential-free workers]
    Presigned --> Layout

    style NFS fill:#f59e0b,color:#fff
    style S3 fill:#0ea5e9,color:#fff
```

| Backend | Pros | Cons |
|---|---|---|
| **NFS** | Simple, POSIX semantics, no auth complexity | Single mount point bottleneck, harder for multi-region |
| **S3-compatible** | Multi-region, credential-free workers via presigned URLs, scale-out | Eventual consistency for some operations, costs scale with API calls |

### Migration

`scripts/migrate_nfs_to_s3.py` performs SHA-256-verified bulk migration with resume support. See [`docs/operations/`](docs/operations/) for the runbook.

---

## 7. Observability

```mermaid
flowchart LR
    subgraph Pipeline["Pipeline + Coordinator"]
        Metrics[Prometheus Collector<br/>7 metric families<br/>ORM-backed]
    end
    subgraph Stack["Observability Stack"]
        Prom[(Prometheus)]
        Graf[Grafana<br/>53 panels]
        Alerts[Alertmanager]
        OTel[OpenTelemetry Collector]
        Jaeger[Jaeger / Tempo]
    end
    subgraph Audit["Audit"]
        Custody[Custody JSONL<br/>hash-chained]
        APIAudit[API Audit Log<br/>masked keys]
    end

    Metrics -->|/metrics| Prom
    Prom --> Graf
    Prom --> Alerts
    Pipeline -.->|spans| OTel
    OTel --> Jaeger
    Pipeline --> Custody
    Pipeline --> APIAudit

    style Metrics fill:#8b5cf6,color:#fff
    style Custody fill:#ef4444,color:#fff
```

### Metric Families

| Family | Examples |
|---|---|
| **Throughput** | `ocr_pages_processed`, `ocr_documents_completed` |
| **Latency** | `ocr_job_duration_seconds`, `ocr_page_duration_seconds` |
| **Queue** | `ocr_queue_depth`, `ocr_queue_lag_seconds` |
| **GPU** | `ocr_gpu_vram_bytes`, `ocr_gpu_utilization_percent` |
| **Cost** | `ocr_cost_estimate_total` (per tenant) |
| **SLA** | `ocr_sla_uptime_ratio`, `ocr_sla_throughput_compliance` |
| **Errors** | `ocr_job_failures_total`, `ocr_custody_chain_breaks_total` |

### Alert Rules

Five PrometheusRule alerts ship in the Helm chart:

1. `OCRQueueBackup` — sustained queue depth above threshold
2. `OCRWorkerDown` — heartbeat gap exceeds tolerance
3. `OCRJobFailureSpike` — failure ratio above baseline
4. `OCRSLABreached` — per-tenant SLA below contract
5. `OCRCustodyChainBreak` — hash chain integrity violation

---

## 8. Security Architecture

```mermaid
flowchart TB
    subgraph Ingress["Ingress"]
        Edge[Ingress Controller<br/>TLS termination]
        Net[NetworkPolicy<br/>allowlisted CIDRs]
    end
    subgraph App["Application"]
        Auth[API Key auth<br/>X-API-Key + Bearer]
        Rate[Rate Limiter<br/>per-tenant]
        Size[Size Limit<br/>413 on oversize]
        WSGuard[WebSocket idle timeout<br/>+ max-duration]
    end
    subgraph Data["Data"]
        Encrypt[Webhook secrets<br/>Fernet]
        SQLite[SQLite 0o600]
        Custody[Custody hash chain]
    end
    subgraph Runtime["Runtime"]
        NonRoot[Non-root containers<br/>UID 1000]
        Caps[capabilities.drop ALL]
        Seccomp[seccompProfile<br/>RuntimeDefault]
        ReadOnly[Read-only root filesystem]
    end

    Edge --> Auth
    Auth --> Rate
    Rate --> Size
    Size --> WSGuard
    App --> Data
    Data -.- Runtime

    style Ingress fill:#3b82f6,color:#fff
    style App fill:#10b981,color:#fff
    style Data fill:#f59e0b,color:#fff
    style Runtime fill:#8b5cf6,color:#fff
```

### Defense in Depth

| Layer | Control |
|---|---|
| **Network** | Ingress allowlist, NetworkPolicy, mTLS option for worker-coordinator |
| **Auth** | API key (X-API-Key or Bearer), 1024-byte timing-safe comparison |
| **Input** | Pydantic validation, 413 on oversize body, content-type checking |
| **Rate limiting** | Per-tenant, configurable via TenantPolicy |
| **WebSocket** | Idle timeout, max duration, payload size limit |
| **Storage** | SQLite 0o600 permissions, S3 cache 0o700 |
| **Secrets** | Webhook secrets encrypted with Fernet (WEBHOOK_SECRET_KEY) |
| **Runtime** | Non-root containers, dropped capabilities, seccomp profile |
| **Audit** | Hash-chained custody log, masked API key in audit middleware |
| **Tenant isolation** | Conditional `tenant_id` filter on all tenant-scoped endpoints |

---

## 9. Chain of Custody

The custody log is **the** forensic primitive. Every operation that touches a document or its derived artifacts emits an event.

```mermaid
sequenceDiagram
    participant J as Job
    participant E as Event Type
    participant C as Custody Log
    participant V as Verifier

    Note over J,C: Linear append-only chain
    J->>E: DOC_RECEIVED
    E->>C: sha256(prev=NULL, type, payload, ts)
    J->>E: OCR_STARTED
    E->>C: sha256(prev=H1, type, payload, ts)
    J->>E: PAGE_COMPLETE × N
    E->>C: sha256(prev=H2..N, type, payload, ts)
    J->>E: DOC_COMPLETE
    E->>C: sha256(prev=Hlast, type, payload, ts)

    Note over C,V: Verification walk
    V->>C: read events in order
    loop For each event
        V->>V: recompute sha256
        V->>V: compare with stored
        V->>V: confirm prev linkage
    end
    V-->>V: PASS or BREAK at position N
```

### Event Types

| Category | Events |
|---|---|
| **Lifecycle** | `DOC_RECEIVED`, `DOC_STARTED`, `DOC_COMPLETE`, `DOC_FAILED` |
| **Page-level** | `PAGE_OCR_START`, `PAGE_OCR_COMPLETE`, `OCR_FAILED_IMAGE_PRESERVED` |
| **Configuration** | `CONFIG_APPLIED`, `MODEL_LOAD_NC_LICENSE_ACKNOWLEDGED` |
| **Language** | `LANGUAGE_DETECTED`, `LANGUAGE_MIXED_SCRIPT`, `LANGUAGE_REDETECTED` |
| **Translation** | `TRANSLATION_REQUESTED`, `TRANSLATION_REJECTED`, `GENERATIVE_TRANSLATION_USED` |
| **Review** | `REVIEW_QUEUE_ENTERED`, `REVIEW_CERTIFIED`, `REVIEW_REJECTED` |
| **Output** | `OUTPUT_WRITTEN`, `OUTPUT_VERIFIED`, `OUTPUT_DELIVERED` |

### Verification

```bash
python scripts/verify_release_state.py --custody-log ocr_output/custody.jsonl
```

The verifier walks the chain, recomputes every hash, validates signatures (if signing is enabled), and reports the exact position of any break.

---

## 10. Failure Modes

EDCOCR is designed to fail visibly and recoverably, not silently.

```mermaid
flowchart TB
    Fail{Failure}
    Fail -->|Worker crash mid-job| Resume[Page-level resume<br/>continue from missing chunks]
    Fail -->|GPU OOM| Backoff[Adaptive batch sizing<br/>retry at smaller batch]
    Fail -->|OCR confidence low| DPI[DPI escalation<br/>retry at 450 then 600]
    Fail -->|OCR returns nothing| Preserve[Image-only embedding<br/>custody event]
    Fail -->|Ghostscript fails| Skip[Skip compression<br/>keep uncompressed]
    Fail -->|Queue depth exceeded| Reject[413 at API<br/>client retries with backoff]
    Fail -->|Subprocess timeout| Kill[Kill subprocess<br/>page enters failure pipeline]
    Fail -->|RabbitMQ down| Quorum[Quorum queues<br/>survive minority loss]
    Fail -->|PostgreSQL down| BackupJob[Backup CronJob<br/>restore from last hour]
    Fail -->|Redis down| Sentinel[Sentinel failover<br/>promote replica]
    Fail -->|Custody chain break| Alert[Prometheus alert<br/>fire immediately]

    style Resume fill:#10b981,color:#fff
    style Preserve fill:#10b981,color:#fff
    style Alert fill:#ef4444,color:#fff
```

### Subprocess Timeouts

External processes have enforced timeouts at module-level constants:

| Process | Timeout |
|---|---|
| Ghostscript | 300s (`GS_TIMEOUT`) |
| Tesseract | 120s (`TESSERACT_TIMEOUT`) |
| Poppler | 300s (`POPPLER_TIMEOUT`) |

On timeout, the subprocess is killed and the page/document enters the failure pipeline. No silent hangs.

---

## Further Reading

- [`docs/00-SYSTEM-BLUEPRINT.md`](docs/00-SYSTEM-BLUEPRINT.md) — Deeper system blueprint
- [`docs/03-INFORMATION-FLOWS.md`](docs/03-INFORMATION-FLOWS.md) — Information flow diagrams
- [`docs/06-CONFIGURATION-REFERENCE.md`](docs/06-CONFIGURATION-REFERENCE.md) — Every env var
- [`docs/FAILOVER-RUNBOOK.md`](docs/FAILOVER-RUNBOOK.md) — HA failover procedures
- [`docs/10-MONITORING-OPERATIONS.md`](docs/10-MONITORING-OPERATIONS.md) — Operating in production
- [`docs/security-audit-checklist.md`](docs/security-audit-checklist.md) — Security review
