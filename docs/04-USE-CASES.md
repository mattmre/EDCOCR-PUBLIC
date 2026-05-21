# 04: Use Cases

## Use Cases by User Type
| User Type | Primary Goal | Recommended Mode |
|---|---|---|
| Developer | Integrate OCR into existing software | FastAPI endpoints + webhooks |
| End User / Analyst | Convert evidence folders into searchable outputs | Monolithic Docker pipeline |
| Platform Engineer | Scale jobs across multiple worker nodes | Django + Celery coordinator |
| Compliance / Legal Ops | Preserve defensible auditability | Custody events + failure audit artifacts |

## Use Case A: Batch OCR for Case Review (End User)
### Problem
A reviewer has mixed PDFs and image scans and needs searchable outputs without changing folder structure.

### Workflow
1. Place source content under `ocr_source/`.
2. Run `docker compose up -d --build`.
3. Track progress in container logs.
4. Collect outputs from `ocr_output/EXPORT/PDF` and `ocr_output/EXPORT/TEXT`.

### Value
- Minimal setup.
- Deterministic fallback behavior.
- Output mirrors input hierarchy.

## Use Case B: API-Driven Ingestion (Developer)
### Problem
An application needs asynchronous OCR as a service with status polling and callback delivery.

### Workflow
1. Start API server (`uvicorn api.main:app`).
2. Submit jobs through `POST /api/v1/jobs`.
3. Poll `GET /api/v1/jobs/{job_id}` or subscribe to `/ws/jobs/{job_id}`.
4. Download artifacts from `GET /api/v1/jobs/{job_id}/result/download`.
5. Receive signed webhook events on terminal states.

### Value
- Standard integration surface.
- Built-in rate limits and API-key controls.
- Pluggable webhook delivery with HMAC signatures.

## Use Case C: Forensic Pipeline with Enrichment (Compliance / Legal Ops)
### Problem
A legal operations team must keep auditable outputs while extracting entities and structured fields.

### Workflow
1. Enable `ENABLE_NER=true`, `ENABLE_CLASSIFICATION=true`, `ENABLE_EXTRACTION=true`.
2. Process collection as normal.
3. Review sidecars in `EXPORT/NER`, `EXPORT/CLASSIFICATION`, `EXPORT/EXTRACTION`.
4. Validate anomalies from `failures.csv`.

### Value
- Keeps text extraction and feature enrichment separate.
- Supports downstream indexing and legal analytics.
- Preserves source evidence if OCR confidence is poor.

## Use Case D: Horizontal Scale-Out (Platform Engineer)
### Problem
A team needs to process larger queues than one host can sustain.

### Workflow
1. Deploy coordinator stack with `docker-compose.coordinator.yml`.
2. Deploy worker stacks with `docker-compose.worker.yml`.
3. Scale workers based on queue depth and GPU availability.
4. Monitor worker health and task throughput through Django and Flower.

### Value
- Queue-driven orchestration.
- Worker capability segmentation (`ocr_gpu`, `cpu_general`, `coordinator`).
- Built-in periodic cleanup and heartbeat checks.

## Feature-to-Outcome Matrix
| Feature | Outcome | Typical Consumer |
|---|---|---|
| Two-pass language routing | Better multilingual OCR accuracy | Global legal and archival teams |
| Tesseract fallback | Resilience on model failures | Operations and support teams |
| Image-only fallback | Evidence preservation when OCR fails | Compliance and legal teams |
| Sidecar exports | Machine-readable analysis outputs | Data engineering teams |
| Webhooks and WebSockets | Event-driven integration | Product and platform engineering |

> [!TIP]
> Start with batch-only delivery, then phase in sidecars and distributed orchestration after baseline throughput is stable.
