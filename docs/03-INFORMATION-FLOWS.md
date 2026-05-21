# 03: Information Flows

## End-to-End Processing Flow
The monolithic pipeline uses bounded queues to move data from source ingestion to final artifacts.

```mermaid
flowchart LR
    A[Source File<br/>PDF or image] --> B[Scheduler]
    B --> C[chunk_queue]
    C --> D[Extractor Threads]
    D --> E[image_queue]
    E --> F[GPU Worker Threads]
    F --> G[assembly_queue]
    G --> H[Assembler]
    H --> I[compression_queue]
    I --> J[Compressor Threads]
    H --> K[Text + JSON Sidecars]
    J --> L[Final PDF]
    K --> M[ocr_output/EXPORT]
    L --> M
```

## Stage Narrative
| Stage | Input | Output | Primary Module |
|---|---|---|---|
| Scheduler | Source files in `SOURCE_FOLDER` | Page chunks + resume signals | `ocr_gpu_async.py` (`scheduler_thread`) |
| Extractors | Chunk metadata | PIL page images | `extractor_thread` |
| GPU workers | Page images | Per-page OCR results + temp PDFs | `worker_thread` |
| Assembler | Out-of-order page results | Ordered final document artifacts | `assembler_thread` |
| Compressors | Final PDF path | Optimized PDF | `compressor_thread` |
| Monitor | Queue and registry state | Throughput/health logs | `monitor_thread` |

## OCR Decision Flow
```mermaid
flowchart TD
    A[Page Image] --> B[PaddleOCR Pass]
    B --> C{Has text?}
    C -->|Yes| D{Language mismatch?}
    D -->|Yes| E[Re-run with detected language model]
    D -->|No| F[Keep pass result]
    C -->|No| G[Tesseract fallback]
    G --> H{Has text?}
    H -->|Yes| I[Use fallback text]
    H -->|No| J[Image-only page preservation]
    E --> K[Write per-page temp PDF and message]
    F --> K
    I --> K
    J --> K
```

> [!NOTE]
> The system records failures in `failures.csv` while keeping deliverables clean.

## API Job Lifecycle Flow
```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI Router
    participant DB as SQLite Jobs DB
    participant JM as JobManager
    participant P as OCR Pipeline
    participant WS as WebSocket
    participant WH as Webhook

    Client->>API: POST /api/v1/jobs
    API->>JM: submit(...)
    JM->>DB: create job(status=submitted)
    JM->>P: launch subprocess
    P-->>DB: update status/progress
    Client->>API: GET /api/v1/jobs/{id}
    API-->>Client: status + progress
    WS->>DB: poll status updates
    DB-->>WS: changed status
    WS-->>Client: progress/completed/failed event
    P-->>JM: exit code
    JM->>WH: deliver callback for terminal state
```

## Distributed Flow (Coordinator Mode)
```mermaid
flowchart TD
    A[Submit Job] --> B[ingest_document task]
    B --> C{Pages <= 20?}
    C -->|Yes| D[process_document on ocr_gpu queue]
    C -->|No| E[extract_pages]
    E --> F[chord of process_page tasks]
    F --> G[assemble_document]
    D --> G
    G --> H[chord: compress_pdf + extract_entities]
    H --> I[finalize_job]
```

## Barcode and OMR Extraction Flow
```mermaid
flowchart LR
    A[Page Image] --> B[pyzbar Decode]
    B --> C{Barcodes found?}
    C -->|Yes| D[Collect Results]
    C -->|No| E[python-zxing Fallback]
    E --> F{Barcodes found?}
    F -->|Yes| D
    F -->|No| G[No barcode output]
    D --> H[OMR Contour Analysis]
    G --> H
    H --> I{Checkboxes or radios detected?}
    I -->|Yes| J[Merge OMR results]
    I -->|No| K[Skip OMR]
    J --> L[".symbology.json"]
    K --> L
```

## Semantic Search Flow
```mermaid
flowchart LR
    A[OCR Text Output] --> B[Text Chunking]
    B --> C[sentence-transformers Encoder]
    C --> D[Dense Vector Embeddings]
    D --> E[(Vector Store)]
    F[Search Query] --> G[Query Embedding]
    G --> H[Cosine Similarity Search]
    E --> H
    H --> I[Ranked Results]
```

## LayoutLMv3 Extraction Flow
```mermaid
flowchart LR
    A[OCR Text + Bounding Boxes] --> D[LayoutLMv3 Model]
    B[Page Image] --> D
    D --> E[BIO Token Labels]
    E --> F[Entity Extraction and Grouping]
    F --> G[".semantic.json"]
```

## Event Bus Flow
```mermaid
flowchart LR
    A[Job Events] --> B[Event Bus Abstraction]
    B --> C{Backend}
    C -->|Kafka| D[Kafka Topic: ocr.jobs.events]
    C -->|In-process| E[Local Queue]
    D --> F[External Consumers]
    E --> G[Internal Handlers]
```

## File Watcher Ingestion Flow
```mermaid
flowchart LR
    A[Hot Folder] --> B[watchdog Observer]
    B --> C[Debounce Filter]
    C --> D{Submission Mode}
    D -->|API| E[POST /api/v1/jobs]
    D -->|Pipeline| F[Direct pipeline call]
    D -->|Distributed| G[Celery dispatch]
    E --> H[OCR Processing]
    F --> H
    G --> H
```

## Distributed Tracing Flow
```mermaid
flowchart LR
    A[Incoming Request] --> B[OTel SDK Instrumentation]
    B --> C[Span Creation]
    C --> D[Span Processor]
    D --> E[OTel Collector]
    E --> F[Jaeger or Tempo]
    E --> G[Prometheus Metrics Export]
    G --> H[Grafana Dashboards]
```

## API Surface (Operational Summary)
| Protocol | Endpoint | Purpose |
|---|---|---|
| REST | `POST /api/v1/jobs` | Submit job |
| REST | `GET /api/v1/jobs` | List jobs |
| REST | `GET /api/v1/jobs/{job_id}` | Poll job status |
| REST | `GET /api/v1/jobs/{job_id}/result` | Artifact metadata |
| REST | `GET /api/v1/jobs/{job_id}/result/download` | Artifact download |
| REST | `POST /api/v1/jobs/{job_id}/retry` | Retry failed/cancelled job |
| REST | `DELETE /api/v1/jobs/{job_id}` | Cancel job |
| REST | `GET /api/v1/health` | Health check |
| REST | `GET /api/v1/transforms` | List available transform operations |
| REST | `POST /api/v1/transforms/execute` | Execute transform operation |
| REST | `GET /api/v1/stamps` | List available stamp operations |
| REST | `POST /api/v1/stamps/execute` | Execute stamp operation |
| REST | `POST /api/v1/batch` | Submit batch of jobs |
| REST | `GET /api/v1/events` | Query job events |
| REST | `GET /api/v1/semantic/search` | Semantic search |
| REST | `GET /api/v1/fleet/status` | Worker fleet status |
| REST | `GET /api/v1/alerts` | Alert queries |
| REST | `GET /api/v1/analytics` | Processing analytics |
| REST | `GET /api/v1/dashboard` | Dashboard metrics |
| WebSocket | `/ws/jobs/{job_id}` | Realtime status updates |

## Data Contracts
| Contract | Location | Notes |
|---|---|---|
| Job record | SQLite (`api/database.py`) | Tracks status, timestamps, artifacts, webhook state |
| Page result | PostgreSQL (`coordinator/jobs/models.py`) | Distributed per-page tracking |
| Custody events | PostgreSQL + JSONL export | Hash-linked chain in distributed mode |
| Artifacts | `ocr_output/EXPORT/...` | Final deliverables and sidecars |
