> Historical v4.1.0 reference. For current install commands and verified scope, use [candidate scope](release-candidate.md) and the root installation guide.

# 08: SDK Reference

## Overview

EDCOCR provides published SDK packages for Python and TypeScript. Both packages wrap the REST API documented in [API-REFERENCE.md](API-REFERENCE.md) with typed interfaces for submission, polling, artifact download, and error handling.

| SDK | Location | Runtime Requirement |
|---|---|---|
| Python | `sdk/python/` (`edcocr-sdk`) | Python 3.9+ from wheel; 3.10+ source build, `httpx`, `pydantic` |
| TypeScript | `sdk/typescript/` (`@edcocr/sdk`) | Node.js 20.19+ or 22.12+ or modern browser with native `fetch` |

> [!NOTE]
> Repository-local compatibility shims also exist at `sdk/python_client.py` and `sdk/typescript/ocr_client.ts`, but the package roots above are the supported SDK distribution surfaces.

---

## Python SDK

### Installation

```bash
pip install edcocr-sdk
```

### Quick Start

```python
from edcocr_sdk import EDCOCRClient

client = EDCOCRClient("http://localhost:8000", api_key="my-secret-key")
job = client.submit_job(file_path="document.pdf")
result = client.wait_for_completion(job.job_id)
pdf_bytes = client.download_artifact(job.job_id, artifact_type="pdf")
client.close
```

### Methods

| Method | Description |
|---|---|
| `health_check` | Check API health |
| `submit_job(...)` | Submit a document for OCR |
| `get_job(job_id)` | Get current status of a job |
| `list_jobs(...)` | List jobs with optional filtering |
| `cancel_job(job_id)` | Cancel a queued or processing job |
| `download_artifact(job_id, artifact_type)` | Download a completed artifact |
| `wait_for_completion(job_id, ...)` | Poll until terminal state |
| `close` | Close the HTTP session |

---

## TypeScript SDK

### Installation

```bash
npm install @edcocr/sdk
```

### Quick Start

```typescript
import { EDCOCRClient } from "@edcocr/sdk";

const client = new EDCOCRClient({
  baseUrl: "http://localhost:8000",
  apiKey: "my-secret-key",
});

const job = await client.submitJob({ filePath: "document.pdf" });
const result = await client.waitForCompletion(job.job_id);
const pdf = await client.downloadArtifact(job.job_id, "pdf");
client.close;
```

### Notes

- Uses native `fetch` in Node.js 20.19+ or 22.12+ and modern browsers.
- Uses `EDCOCRClient` as the public client export.
- Method names align conceptually with the Python SDK, but follow TypeScript casing such as `submitJob` and `waitForCompletion`.

---

## Integration Notes

- Use `X-API-Key` for API key auth.
- Handle `429` and `503` responses with retry and backoff.
- Keep polling intervals modest to avoid rate-limit pressure.
