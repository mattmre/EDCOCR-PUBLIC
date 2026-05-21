# edcocr-sdk

Python SDK for the [EDCOCR](https://github.com/mattmre/EDCOCR-PUBLIC) forensic-grade OCR pipeline.

## Installation

```bash
pip install edcocr-sdk
```

For WebSocket streaming support:

```bash
pip install edcocr-sdk[websockets]
```

## Quick Start

### CLI Usage

The SDK package installs an operator CLI:

```bash
edcocr submit input.pdf --output out/
edcocr batch ./docs --tenant tenant-a
edcocr status job_abc123def456
edcocr export-bundle job_abc123def456 --out bundle.json
edcocr verify-custody job_abc123def456
```

Set `EDCOCR_API_URL` and `EDCOCR_API_KEY`, or pass `--api-url` and
`--api-key` before the subcommand.

### MCP-Style Tool Server

The package also installs `edcocr-mcp`, a dependency-light JSON-RPC tool server
for local MCP adapters. It exposes:

- `ocr_submit_document`
- `ocr_submit_batch`
- `ocr_get_job_status`
- `ocr_get_document_bundle`
- `ocr_get_evidence_bundle`
- `ocr_list_outputs`
- `ocr_validate_custody`

Run it with the same `EDCOCR_API_URL` and `EDCOCR_API_KEY` environment
variables used by the CLI.

### Synchronous Usage

```python
import os

from edcocr_sdk import EDCOCRClient

with EDCOCRClient("http://localhost:8000", api_key=os.environ["OCR_API_KEY"]) as client:
    # Check API health
    health = client.health_check
    print(f"API version: {health.version}, status: {health.status}")

    # Submit a document for OCR
    job = client.submit_job("document.pdf")
    print(f"Job submitted: {job.job_id}")

    # Wait for completion
    result = client.wait_for_completion(job.job_id, timeout=300)
    print(f"Job status: {result.status}")

    # Download the searchable PDF
    client.download_artifact(job.job_id, "pdf", output_path="output.pdf")

    # Retrieve OCR contract and custody surfaces
    bundle = client.get_document_bundle(job.job_id)
    evidence = client.get_evidence_bundle(job.job_id)
    assert client.verify_custody(job.job_id)
```

### Async Usage

```python
import asyncio
from edcocr_sdk import AsyncEDCOCRClient

async def main:
    async with AsyncEDCOCRClient("http://localhost:8000", api_key=os.environ["OCR_API_KEY"]) as client:
        job = await client.submit_job("document.pdf")
        result = await client.wait_for_completion(job.job_id)
        content = await client.download_artifact(job.job_id, "pdf")

asyncio.run(main)
```

### One-liner: Submit and Wait

```python
from edcocr_sdk import EDCOCRClient

with EDCOCRClient("http://localhost:8000", api_key=os.environ["OCR_API_KEY"]) as client:
    result = client.submit_and_wait("document.pdf", timeout=300)
    print(f"Completed: {result.is_success}")
```

## API Reference

### EDCOCRClient / AsyncEDCOCRClient

| Method | Description |
|---|---|
| `health_check` | Check API health status |
| `submit_job(file_path, ...)` | Submit a document for OCR |
| `get_job(job_id)` | Get job status and progress |
| `list_jobs(status, page, per_page)` | List jobs with pagination |
| `get_result(job_id)` | Get result metadata and artifact links |
| `download_artifact(job_id, type, output_path)` | Download a result artifact |
| `list_outputs(job_id)` | List output manifest artifacts |
| `get_document_bundle(job_id)` | Retrieve the EDC `DocumentBundle.v1` JSON |
| `export_document_bundle(job_id, output_path)` | Retrieve and write the `DocumentBundle.v1` JSON |
| `get_evidence_bundle(job_id)` | Retrieve OCR custody/evidence bundle JSON |
| `export_evidence_bundle(job_id, output_path)` | Retrieve and write OCR custody/evidence bundle JSON |
| `verify_custody(job_id)` | Return whether custody evidence is available and valid |
| `submit_batch(file_paths, source_paths, ...)` | Submit multiple documents for OCR |
| `list_batches(status, limit, offset)` | List OCR batches |
| `get_batch(batch_id)` | Get OCR batch status |
| `cancel_job(job_id)` | Cancel a running job |
| `retry_job(job_id)` | Retry a failed job |
| `wait_for_completion(job_id, poll_interval, timeout)` | Poll until job completes |
| `submit_and_wait(file_path, ...)` | Submit and poll in one call |

### Submit Options

```python
client.submit_job(
    file_path="document.pdf",       # Local file to upload
    # file_obj=open_file,           # Or: file-like object
    # source_path="/server/path",   # Or: server-side path
    enable_docintel=True,           # Enable document intelligence
    docintel_mode="full",           # layout_only | tables_only | full
    priority="normal",              # urgent | normal | low
    webhook_url="https://...",      # Completion webhook
    webhook_secret=os.environ["WEBHOOK_SIGNING_KEY"],  # HMAC-SHA256 signing key
    processing_timeout_minutes=30,  # Per-job timeout override
)
```

### Exception Hierarchy

All exceptions inherit from `OCRLocalError`:

| Exception | HTTP Status | Description |
|---|---|---|
| `AuthenticationError` | 401, 403 | Invalid or missing API key |
| `NotFoundError` | 404 | Resource not found |
| `RateLimitError` | 429 | Rate limit or queue capacity exceeded |
| `ValidationError` | 400, 422 | Invalid request parameters |
| `ConflictError` | 409 | Job not in expected state |
| `ServerError` | 5xx | Server-side error |
| `TimeoutError` | -- | Polling timeout exceeded |

```python
from edcocr_sdk import EDCOCRClient, NotFoundError, RateLimitError

with EDCOCRClient("http://localhost:8000", api_key=os.environ["OCR_API_KEY"]) as client:
    try:
        job = client.get_job("job_nonexistent")
    except NotFoundError:
        print("Job not found")
    except RateLimitError:
        print("Too many requests, retry later")
```

### Models

All response models are Pydantic v2 `BaseModel` instances with full type hints:

- `Job` -- Full job status with progress, settings, timestamps
- `JobSubmitResult` -- Submission response with job_id and links
- `JobResult` -- Completed job metadata with artifact URLs
- `JobListResponse` -- Paginated job list
- `HealthResponse` -- API health with version and job counts

## Requirements

- Python 3.9+
- httpx >= 0.24.0
- pydantic >= 2.0.0

## License

Apache-2.0
