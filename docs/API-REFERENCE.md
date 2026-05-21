# API Reference

Comprehensive reference for the EDCOCR REST API, WebSocket streaming, webhook notifications, and coordinator metrics endpoints.

**Version**: 4.1.0

**Base URLs**:
- FastAPI server: `http://<host>:8000`
- Django coordinator: `http://<host>:8000`

**Interactive docs** (FastAPI server):
- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`
- OpenAPI schema: `GET /openapi.json`

These routes are mounted only when `EXPOSE_API_DOCS=true`. The default is
`false`, so production-like deployments do not expose the interactive docs or
raw schema.

---

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Health](#health)
- [Jobs](#jobs)
- [Batch Jobs](#batch-jobs)
- [Events](#events)
- [WebSocket Streaming](#websocket-streaming)
- [Transforms](#transforms-feature-gated)
- [Stamps](#stamps-feature-gated)
- [Output Manifest](#output-manifest)
- [Schema Registry](#schema-registry)
- [Queue Operations](#queue-operations)
- [Review Queue](#review-queue)
- [Exception Routing Rules](#exception-routing-rules)
- [Entity Recall](#entity-recall)
- [Semantic Search / VLM](#semantic-search--vlm)
- [Translation API](#translation-api-feature-gated)
- [Admin / Multi-Tenancy](#admin--multi-tenancy-feature-gated)
- [Webhooks](#webhooks)
- [Coordinator API](#coordinator-api-django)
- [Configuration Reference](#configuration-reference)
- [Endpoint Summary](#endpoint-summary)

---

## Authentication

The API supports three authentication methods, checked in the following order of priority:

### 1. Multi-Tenant API Key (when `ENABLE_MULTITENANCY=true`)

```
X-API-Key: <tenant-api-key>
```

Tenant API keys are created through the admin endpoints. Each key is bound to a tenant and carries specific permissions. The key hash is compared against the database.

### 2. Legacy API Key

```
X-API-Key: <key>
```

Compared against the `OCR_API_KEY` environment variable using constant-time comparison. When set, all requests (except exempt paths) must include this header.

### 3. OAuth2 Bearer Token (when `OAUTH2_ENABLED=true`)

```
Authorization: Bearer <jwt>
```

JWT is validated against the configured OIDC JWKS endpoint. Claims are mapped to RBAC roles.

### Unauthenticated Access

Unauthenticated access is allowed **only** when `ALLOW_UNAUTHENTICATED=true` is set. Anonymous callers receive the role configured by `ANONYMOUS_ROLE` (`viewer` by default), and this mode is intended for development environments only.

### Exempt Paths

The following paths do not require authentication:

| Path | Description |
|------|-------------|
| `/api/v1/health` | Health check |
| `/api/v1/health/detailed` | Detailed health check |
| `/api/v1/ready` | Readiness alias |
| `/api/v1/readiness` | Readiness alias |
| `/api/v1/translation/readiness` | OCR-side external translation readiness probe |
| `/docs` | Swagger UI; only mounted and exempt when `EXPOSE_API_DOCS=true` |
| `/redoc` | ReDoc; only mounted and exempt when `EXPOSE_API_DOCS=true` |
| `/openapi.json` | OpenAPI schema; only mounted and exempt when `EXPOSE_API_DOCS=true` |

### RBAC Roles

Three roles are defined (highest privilege first):

| Role | Description | Capabilities |
|------|-------------|--------------|
| `admin` | Full access | All operations, tenant management |
| `operator` | Write access | Submit, retry, cancel jobs; execute transforms/stamps |
| `viewer` | Read-only access | View job status, results, events |

Write operations (submit, retry, cancel, transforms, stamps) require `admin` or `operator`. The `admin` role always passes any role check.

### IP Allowlist

Optional host-based access control. Set `API_ALLOWED_IPS` (comma-separated) to restrict API access by `request.client.host`. Requests from non-allowlisted IPs receive `403 Forbidden`.

**Note**: When running behind a reverse proxy or load balancer, the allowlist compares the proxy source IP. Configure upstream IP forwarding accordingly.

### cURL Authentication Examples

```bash
# Legacy API key
curl -H "X-API-Key: your-secret-key" http://localhost:8000/api/v1/jobs

# OAuth2 Bearer token
curl -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIs..." http://localhost:8000/api/v1/jobs

# No auth (dev mode with ALLOW_UNAUTHENTICATED=true)
curl http://localhost:8000/api/v1/jobs
```

---

## Rate Limiting

Rate limits are enforced per remote IP address using slowapi. Exceeding the limit returns `429 Too Many Requests`.

| Scope | Default | Env Var |
|-------|---------|---------|
| Read endpoints (GET) | 60/minute | `OCR_RATE_LIMIT` |
| Submit endpoints (POST job/retry) | 10/minute | `OCR_SUBMIT_RATE_LIMIT` |
| Batch submit | 5/minute | Hardcoded |

### Rate Limit Response

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please try again later."
}
```

**Status code**: `429`

---

## Error Handling

### Standard Error Response

All error responses follow a consistent JSON structure:

```json
{
  "error": "error_code",
  "message": "Human-readable error description",
  "details": {}
}
```

The `details` field is optional and provides additional context for specific error types.

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| `400` | `invalid_request` | Missing or invalid request parameters |
| `400` | `invalid_job_id` | Job ID does not match format `^job_[0-9a-f]{12}$` |
| `400` | `invalid_batch_id` | Batch ID does not match format `^batch_[0-9a-f]{12}$` |
| `401` | `unauthorized` | Invalid or missing API key / bearer token |
| `403` | `forbidden` | Insufficient permissions or IP not allowlisted |
| `403` | `feature_disabled` | Feature gate is not enabled |
| `404` | `job_not_found` | Job does not exist |
| `404` | `batch_not_found` | Batch does not exist |
| `404` | `not_found` | Requested resource not found |
| `409` | `job_not_complete` | Job has not yet reached a terminal state |
| `409` | `invalid_state` | Operation not valid for current job state |
| `409` | `already_decided` | Review item already has a decision |
| `413` | `file_too_large` | Upload exceeds `MAX_UPLOAD_SIZE_MB` |
| `422` | (validation) | Pydantic validation errors |
| `429` | `queue_full` | Job queue is at `MAX_CONCURRENT_JOBS` capacity |
| `429` | `quota_exceeded` | Tenant quota exceeded (multi-tenancy) |
| `429` | `rate_limit_exceeded` | Too many requests |
| `500` | `internal_error` | Unexpected server error |
| `502` | `vlm_error` | VLM inference server error |
| `503` | `server_misconfigured` | API key required but not configured |
| `503` | `vlm_disabled` | VLM feature not enabled |
| `504` | `vlm_timeout` | VLM inference server timeout |

---

## Health

### GET /api/v1/health

Returns pipeline health status including job counts by status. **No authentication required.**

**Rate limit**: 60/minute

#### Response `200 OK`

```json
{
  "status": "healthy",
  "version": "4.1.0",
  "uptime_seconds": 3642.5,
  "jobs": {
    "submitted": 1,
    "processing": 2,
    "completed": 150,
    "failed": 3,
    "cancelled": 0
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Always `"healthy"` when the server is responding |
| `version` | `string` | Pipeline version (from `version.py`) |
| `uptime_seconds` | `float` | Seconds since server start |
| `jobs` | `object` | Job counts by status |

#### cURL

```bash
curl http://localhost:8000/api/v1/health
```

### GET /api/v1/health/detailed

Returns detailed health status with per-subsystem probe results. **No authentication required.**

**Rate limit**: 60/minute

Subsystems checked:
- **database** -- SQLite connectivity (lightweight `SELECT 1` probe)
- **disk_output** -- Output directory existence and free space (threshold: 1 GB)
- **disk_source** -- Source directory existence and free space (threshold: 1 GB)
- **models** -- FastText language-detection model availability
- **pipeline** -- Monitor heartbeat file age (stale > 60s = degraded, > 120s = unhealthy)

Overall status is derived from the worst individual check: if any subsystem is `unhealthy`, the response is `unhealthy`; if any is `degraded`, the response is `degraded`; otherwise `healthy`.

#### Response `200 OK` (all systems healthy)

```json
{
  "status": "healthy",
  "version": "4.1.0",
  "uptime_seconds": 3642.5,
  "jobs": {
    "submitted": 1,
    "processing": 2,
    "completed": 150,
    "failed": 3,
    "cancelled": 0
  },
  "checks": {
    "database": {
      "status": "healthy",
      "message": "SQLite OK",
      "latency_ms": 0.3
    },
    "disk_output": {
      "status": "healthy",
      "message": "Output: 42.1 GB free"
    },
    "disk_source": {
      "status": "healthy",
      "message": "Source: 42.1 GB free"
    },
    "models": {
      "status": "healthy",
      "message": "FastText model found: /app/lid.176.bin"
    },
    "pipeline": {
      "status": "healthy",
      "message": "Heartbeat: 12s ago"
    }
  }
}
```

#### Response `503 Service Unavailable` (one or more subsystems degraded/unhealthy)

```json
{
  "status": "unhealthy",
  "version": "4.1.0",
  "uptime_seconds": 7200.0,
  "jobs": {},
  "checks": {
    "database": {
      "status": "unhealthy",
      "message": "Database error: Connection refused"
    },
    "disk_output": {
      "status": "degraded",
      "message": "Output: 0.4 GB free (threshold: 1 GB)"
    },
    "models": {
      "status": "degraded",
      "message": "FastText model not found (language detection will use fallback)"
    },
    "pipeline": {
      "status": "unhealthy",
      "message": "Heartbeat stale: 180s old"
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | `"healthy"`, `"degraded"`, or `"unhealthy"` |
| `version` | `string` | Pipeline version (from `version.py`) |
| `uptime_seconds` | `float` | Seconds since server start |
| `jobs` | `object` | Job counts by status (may be empty on database failure) |
| `checks` | `object` | Per-subsystem check results |
| `checks.*.status` | `string` | `"healthy"`, `"degraded"`, or `"unhealthy"` |
| `checks.*.message` | `string` | Human-readable status detail |
| `checks.*.latency_ms` | `float?` | Probe latency in milliseconds (present for database check) |

#### cURL

```bash
curl http://localhost:8000/api/v1/health/detailed
```

---

## Jobs

All job endpoints are under `/api/v1/jobs`. Job IDs follow the format `job_` followed by 12 hex characters (e.g., `job_abc123def456`).

### POST /api/v1/jobs -- Submit Job

Submit a document for OCR processing. Provide either a file upload or a server-side `source_path`.

**Auth**: Required | **Rate limit**: 10/minute | **RBAC**: `admin`, `operator`

**Content type**: `multipart/form-data`

#### Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file upload | No* | -- | Document file to process |
| `source_path` | `string` | No* | -- | Server-side path (validated against `SOURCE_FOLDER`) |
| `priority` | `string` | No | `"normal"` | `"urgent"`, `"normal"`, or `"low"` |
| `enable_docintel` | `boolean` | No | `false` | Enable Document Intelligence extraction |
| `docintel_mode` | `string` | No | `"full"` | `"layout_only"`, `"tables_only"`, or `"full"` |
| `skip_ocr` | `boolean` | No | `false` | Skip OCR, perform NLP/DocIntel only |
| `processing_timeout_minutes` | `integer` | No | -- | Per-job timeout override (>= 1 minute) |
| `webhook_url` | `string` | No | -- | HTTPS callback URL (SSRF-validated) |
| `webhook_secret` | `string` | No | -- | HMAC-SHA256 signing secret for webhook |

\* One of `file` or `source_path` is required.

> [!NOTE]
> `enable_docintel` and `skip_ocr` are AI-adjacent controls. When omitted, a job
> follows the default forensic-core OCR path with the normal custody, validation,
> and failure-handling behavior. These controls add optional semantic or native-text behavior
> on top of that baseline; they do not redefine the core pipeline promise.

`enable_docintel` and `docintel_mode` are optional AI-adjacent enrichment controls. They may add structure sidecars, but they do not redefine the baseline forensic-core PDF/TXT and custody contract. See [architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).

#### Response `201 Created`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "submitted",
  "created_at": "2026-03-14T10:30:00",
  "priority": "normal",
  "source_file": "contract.pdf",
  "estimated_pages": null,
  "links": {
    "self": "/api/v1/jobs/job_a1b2c3d4e5f6",
    "result": "/api/v1/jobs/job_a1b2c3d4e5f6/result"
  }
}
```

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_request` | Neither file nor source_path provided |
| `404` | `not_found` | Source path does not exist |
| `413` | `file_too_large` | Upload exceeds `MAX_UPLOAD_SIZE_MB` (default 5120 MB) |
| `422` | (validation) | Invalid priority, docintel_mode, or other field |
| `429` | `queue_full` | Active jobs >= `MAX_CONCURRENT_JOBS` |
| `429` | `quota_exceeded` | Tenant quota exceeded (multi-tenancy) |

#### cURL Examples

```bash
# Upload a file
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "X-API-Key: your-key" \
  -F "file=@document.pdf" \
  -F "priority=normal"

# Use server-side path
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "X-API-Key: your-key" \
  -F "source_path=/app/ocr_source/document.pdf" \
  -F "enable_docintel=true" \
  -F "docintel_mode=full"

# With webhook notification
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "X-API-Key: your-key" \
  -F "file=@document.pdf" \
  -F "webhook_url=https://example.com/webhook" \
  -F "webhook_secret=my-hmac-secret"
```

---

### GET /api/v1/jobs -- List Jobs

List jobs with optional filtering and pagination.

**Auth**: Required | **Rate limit**: 60/minute

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | `string` | -- | Filter by status (`submitted`, `processing`, `completed`, `failed`, `cancelled`) |
| `batch_id` | `string` | -- | Filter by batch ID |
| `page` | `integer` | `1` | Page number (>= 1) |
| `per_page` | `integer` | `20` | Items per page (1-100) |

#### Response `200 OK`

```json
{
  "jobs": [
    {
      "job_id": "job_a1b2c3d4e5f6",
      "status": "completed",
      "created_at": "2026-03-14T10:30:00",
      "started_at": "2026-03-14T10:30:01",
      "completed_at": "2026-03-14T10:31:15",
      "priority": "normal",
      "source_file": "contract.pdf",
      "progress": {
        "total_pages": 10,
        "pages_completed": 10,
        "percent_complete": 100.0,
        "current_stage": "processing"
      },
      "settings": {
        "enable_docintel": false,
        "docintel_mode": "full"
      },
      "webhook_status": "delivered"
    }
  ],
  "total": 42,
  "page": 1,
  "per_page": 20
}
```

#### cURL

```bash
# List all jobs
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/jobs

# Filter by status with pagination
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/jobs?status=completed&page=2&per_page=50"
```

---

### GET /api/v1/jobs/{job_id} -- Job Status

Get detailed status and progress of a specific job.

**Auth**: Required | **Rate limit**: 60/minute

#### Path Parameters

| Param | Type | Format | Description |
|-------|------|--------|-------------|
| `job_id` | `string` | `^job_[0-9a-f]{12}$` | Job identifier |

#### Response `200 OK`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "processing",
  "created_at": "2026-03-14T10:30:00",
  "started_at": "2026-03-14T10:30:01",
  "completed_at": null,
  "priority": "normal",
  "source_file": "contract.pdf",
  "progress": {
    "total_pages": 10,
    "pages_completed": 5,
    "percent_complete": 50.0,
    "current_stage": "processing"
  },
  "settings": {
    "enable_docintel": true,
    "docintel_mode": "full"
  },
  "webhook_status": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | `submitted`, `processing`, `completed`, `failed`, or `cancelled` |
| `progress.total_pages` | `integer` | Total detected pages (0 until processing starts) |
| `progress.pages_completed` | `integer` | Pages processed so far |
| `progress.percent_complete` | `float` | Completion percentage (0.0 - 100.0) |
| `progress.current_stage` | `string` | Current pipeline stage |
| `settings` | `object` | Job configuration parameters |
| `webhook_status` | `string?` | `null`, `pending`, `delivered`, or `failed` |

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_job_id` | Malformed job ID |
| `404` | `job_not_found` | Job does not exist |

#### cURL

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6
```

---

### GET /api/v1/jobs/{job_id}/result -- Result Metadata

Get result metadata and download links for a completed or failed job.

**Auth**: Required | **Rate limit**: 60/minute

#### Response `200 OK`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "completed",
  "completed_at": "2026-03-14T10:31:15",
  "processing_time_seconds": 74.2,
  "artifacts": {
    "pdf": "/api/v1/jobs/job_a1b2c3d4e5f6/result/download?type=pdf",
    "text": "/api/v1/jobs/job_a1b2c3d4e5f6/result/download?type=text",
    "structure": "/api/v1/jobs/job_a1b2c3d4e5f6/result/download?type=structure"
  },
  "metadata": {
    "pages_processed": 10
  }
}
```

The `artifacts` object contains download URLs for each available output type. The `structure` artifact is only present when DocIntel is enabled.

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_job_id` | Malformed job ID |
| `404` | `job_not_found` | Job does not exist |
| `409` | `job_not_complete` | Job is still processing; includes progress percentage |

#### cURL

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/result
```

---

### GET /api/v1/jobs/{job_id}/result/download -- Download Artifact

Download a specific result artifact from a completed job.

**Auth**: Required | **Rate limit**: 60/minute

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `string` | `"pdf"` | Artifact type: `pdf`, `text`, or `structure` |

#### Response

Returns a file download with the appropriate content type:

| Type | Content-Type |
|------|-------------|
| `pdf` | `application/pdf` |
| `text` | `text/plain` |
| `structure` | `application/json` |
| (other) | `application/octet-stream` |

The `Content-Disposition` filename is `{job_id}.{type}`.

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_job_id` | Malformed job ID |
| `404` | `job_not_found` | Job does not exist |
| `404` | `artifact_not_found` | Requested artifact type is not available |
| `409` | `job_not_complete` | Job status is not `completed` |

#### cURL

```bash
# Download PDF
curl -H "X-API-Key: your-key" -o output.pdf \
  "http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/result/download?type=pdf"

# Download extracted text
curl -H "X-API-Key: your-key" -o output.txt \
  "http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/result/download?type=text"

# Download structure JSON
curl -H "X-API-Key: your-key" -o output.json \
  "http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/result/download?type=structure"
```

---

### POST /api/v1/jobs/{job_id}/retry -- Retry Failed Job

Create a new job from the original source of a failed or cancelled job. The original settings (priority, DocIntel, webhook) are preserved.

**Auth**: Required | **Rate limit**: 10/minute | **RBAC**: `admin`, `operator`

#### Response `201 Created`

Same schema as the job submission response (`JobSubmitResponse`). A new `job_id` is assigned.

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_job_id` | Malformed job ID |
| `404` | `job_not_found` | Original job does not exist |
| `404` | `source_missing` | Original source file no longer available |
| `409` | `invalid_state` | Job is not in `failed` or `cancelled` state |
| `429` | `queue_full` | Job queue at capacity |
| `429` | `quota_exceeded` | Tenant quota exceeded |

#### cURL

```bash
curl -X POST -H "X-API-Key: your-key" \
  http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/retry
```

---

### DELETE /api/v1/jobs/{job_id} -- Cancel Job

Cancel a running or submitted job. If the job has a running subprocess, it is killed. Jobs that are already in a terminal state (`completed`, `failed`, `cancelled`) are returned unchanged.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

#### Response `200 OK`

Returns the updated `JobStatusResponse` with `cancelled` status.

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "cancelled",
  "created_at": "2026-03-14T10:30:00",
  "started_at": "2026-03-14T10:30:01",
  "completed_at": "2026-03-14T10:30:05",
  "priority": "normal",
  "source_file": "contract.pdf",
  "progress": null,
  "settings": {},
  "webhook_status": null
}
```

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_job_id` | Malformed job ID |
| `404` | `job_not_found` | Job does not exist |

#### cURL

```bash
curl -X DELETE -H "X-API-Key: your-key" \
  http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6
```

---

## Batch Jobs

Batch endpoints allow submitting and managing multiple documents as a group. Batch IDs follow the format `batch_` followed by 12 hex characters.

### POST /api/v1/jobs/batch -- Submit Batch

Submit multiple documents for OCR processing as a batch.

**Auth**: Required | **Rate limit**: 5/minute | **RBAC**: `admin`, `operator`

**Content type**: `multipart/form-data`

#### Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `files` | file uploads | No* | `[]` | Multiple file uploads |
| `source_paths` | `string` | No* | -- | JSON array of server-side paths |
| `priority` | `string` | No | `"normal"` | `"urgent"`, `"normal"`, or `"low"` |
| `enable_docintel` | `boolean` | No | `false` | Enable Document Intelligence |
| `docintel_mode` | `string` | No | `"full"` | `"layout_only"`, `"tables_only"`, or `"full"` |
| `skip_ocr` | `boolean` | No | `false` | Skip OCR, perform NLP/DocIntel only |
| `processing_timeout_minutes` | `integer` | No | -- | Per-job timeout override (>= 1 minute) |
| `webhook_url` | `string` | No | -- | Batch-level webhook callback URL |
| `webhook_secret` | `string` | No | -- | HMAC-SHA256 signing secret for batch webhook |

\* At least one file upload or source_path is required.

As with single-job submission, DocIntel settings are optional AI-adjacent enrichment controls rather than part of the minimum forensic-core processing guarantee. See [architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).

**Max batch size**: `MAX_BATCH_SIZE` (default 50, maximum 500).

#### Response `201 Created`

```json
{
  "batch_id": "batch_f1e2d3c4b5a6",
  "status": "submitted",
  "created_at": "2026-03-14T10:30:00",
  "total_jobs": 3,
  "priority": "normal",
  "jobs": [
    { "job_id": "job_111111111111", "source_file": "doc1.pdf", "status": "submitted" },
    { "job_id": "job_222222222222", "source_file": "doc2.pdf", "status": "submitted" },
    { "job_id": "job_333333333333", "source_file": "doc3.pdf", "status": "submitted" }
  ],
  "links": {
    "self": "/api/v1/jobs/batch/batch_f1e2d3c4b5a6",
    "jobs": "/api/v1/jobs?batch_id=batch_f1e2d3c4b5a6"
  }
}
```

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_request` | No files or source_paths provided |
| `400` | `batch_too_large` | Batch exceeds `MAX_BATCH_SIZE` |
| `413` | `file_too_large` | Any file exceeds `MAX_UPLOAD_SIZE_MB` |
| `422` | (validation) | Invalid priority, docintel_mode, or source_paths JSON |

#### cURL

```bash
# Upload multiple files
curl -X POST http://localhost:8000/api/v1/jobs/batch \
  -H "X-API-Key: your-key" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@doc3.pdf" \
  -F "priority=normal"

# Server-side paths
curl -X POST http://localhost:8000/api/v1/jobs/batch \
  -H "X-API-Key: your-key" \
  -F 'source_paths=["/app/ocr_source/doc1.pdf","/app/ocr_source/doc2.pdf"]'
```

---

### GET /api/v1/jobs/batch -- List Batches

List batches with optional status filter and pagination.

**Auth**: Required | **Rate limit**: 60/minute

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | `string` | -- | Filter by batch status |
| `page` | `integer` | `1` | Page number (>= 1) |
| `per_page` | `integer` | `20` | Items per page (1-100) |

#### Response `200 OK`

```json
{
  "batches": [
    {
      "batch_id": "batch_f1e2d3c4b5a6",
      "status": "completed",
      "created_at": "2026-03-14T10:30:00",
      "completed_at": "2026-03-14T10:35:00",
      "processing_time": 300.0,
      "total_jobs": 3,
      "progress": {
        "submitted": 0,
        "processing": 0,
        "completed": 3,
        "failed": 0,
        "cancelled": 0,
        "percent_complete": 100.0
      },
      "jobs": [
        { "job_id": "job_111111111111", "source_file": "doc1.pdf", "status": "completed" }
      ],
      "settings": {},
      "webhook_status": "delivered"
    }
  ],
  "total": 5,
  "page": 1,
  "per_page": 20
}
```

---

### GET /api/v1/jobs/batch/{batch_id} -- Batch Status

Get status and progress of a batch, including all child job summaries.

**Auth**: Required | **Rate limit**: 60/minute

#### Response `200 OK`

Same schema as individual batch items in the list response above.

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_batch_id` | Malformed batch ID |
| `404` | `batch_not_found` | Batch does not exist |

---

### DELETE /api/v1/jobs/batch/{batch_id} -- Cancel Batch

Cancel all running and submitted jobs in a batch.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

#### Response `200 OK`

Updated `BatchStatusResponse` with cancelled jobs.

---

### POST /api/v1/jobs/batch/{batch_id}/retry -- Retry Batch

Retry all failed or cancelled jobs within a batch.

**Auth**: Required | **Rate limit**: 10/minute | **RBAC**: `admin`, `operator`

#### Response `200 OK`

Updated `BatchStatusResponse` with retried jobs.

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_batch_id` | Malformed batch ID |
| `404` | `batch_not_found` | Batch does not exist |
| `409` | `no_retryable_jobs` | No jobs in `failed` or `cancelled` state |

---

## Events

### GET /api/v1/jobs/{job_id}/events -- Replay Job Events

Replay stored lifecycle events for a job. Useful for reconstructing job history or reconnecting after a WebSocket disconnection.

**Auth**: Required | **RBAC**: `viewer`, `operator`, `admin`

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `since_id` | `string` | -- | Return events after this event ID |
| `limit` | `integer` | `100` | Maximum events to return (1-1000) |

#### Response `200 OK`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "events": [
    {
      "event_id": "evt_...",
      "event_type": "job.submitted",
      "timestamp": "2026-03-14T10:30:00Z",
      "data": {}
    },
    {
      "event_id": "evt_...",
      "event_type": "job.processing",
      "timestamp": "2026-03-14T10:30:01Z",
      "data": {}
    }
  ],
  "count": 2
}
```

#### cURL

```bash
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/events?limit=50"
```

---

### GET /api/v1/webhooks/dlq -- List Dead-Letter Queue

List webhook dead-letter queue entries (failed webhook deliveries).

**Auth**: Required | **RBAC**: `operator`, `admin`

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | `integer` | `100` | Maximum entries to return (1-1000) |

#### Response `200 OK`

```json
{
  "entries": [
    {
      "entry_id": "dlq_abc123def4567890",
      "job_id": "job_a1b2c3d4e5f6",
      "webhook_url": "https://example.com/webhook",
      "event_type": "job.completed",
      "payload": { "event": "job.completed", "job_id": "..." },
      "last_error": "HTTP 500",
      "attempts": 4,
      "created_at": "2026-03-14T10:31:20Z",
      "retried_at": null
    }
  ],
  "count": 1
}
```

---

### GET /api/v1/webhooks/dlq/{entry_id} -- Get DLQ Entry

Get a single DLQ entry by ID.

**Auth**: Required | **RBAC**: `operator`, `admin`

**DLQ entry ID format**: `^dlq_[0-9a-f]{16}$`

#### Error Responses

| Status | Condition |
|--------|-----------|
| `400` | Invalid entry_id format |
| `404` | DLQ entry not found |

---

### POST /api/v1/webhooks/dlq/{entry_id}/retry -- Retry DLQ Entry

Retry a failed webhook delivery from the dead-letter queue. Re-delivers the original payload to the webhook URL.

**Auth**: Required | **RBAC**: `operator`, `admin`

#### Response `200 OK` (success)

```json
{
  "status": "delivered",
  "entry_id": "dlq_abc123def4567890",
  "http_status": 200
}
```

#### Response `200 OK` (delivery failure)

```json
{
  "status": "failed",
  "entry_id": "dlq_abc123def4567890",
  "http_status": 500,
  "error": "HTTP 500"
}
```

#### Error Responses

| Status | Condition |
|--------|-----------|
| `400` | Invalid entry_id or URL validation failure |
| `404` | DLQ entry not found |
| `409` | Entry has already been retried |

---

## WebSocket Streaming

### WS /ws/jobs/{job_id} -- Real-Time Job Progress

Stream real-time progress updates for an OCR job via WebSocket.

**Protocol**: `ws://` or `wss://`

#### Authentication

Three methods are supported (checked in order):

1. **X-API-Key header** during the WebSocket handshake
2. **OAuth2 Bearer token** via the `token` query parameter: `ws://host/ws/jobs/{id}?token=jwt`
3. **Auth frame** as the first message after connection:
   ```json
   {"type": "auth", "api_key": "<value from OCR_API_KEY>"}
   ```

The auth frame must arrive within 5 seconds of connection acceptance.

#### Connection Limits

- Maximum 100 concurrent WebSocket connections across all jobs
- Exceeding the limit returns close code `4029`

#### Server Messages

| Type | Fields | When |
|------|--------|------|
| `connected` | `job_id`, `status` | Connection established, initial status sent |
| `progress` | `job_id`, `status` | Status changes during processing |
| `completed` | `job_id`, `status`, `output_path` | Job finishes successfully |
| `failed` | `job_id`, `status`, `error` | Job fails |
| `cancelled` | `job_id`, `status` | Job is cancelled |
| `error` | `message` | Auth failure, job not found, or internal error |
| `pong` | -- | Response to client `"ping"` text frame |

#### Message Examples

**Connected**:
```json
{"type": "connected", "job_id": "job_a1b2c3d4e5f6", "status": "submitted"}
```

**Progress update**:
```json
{"type": "progress", "job_id": "job_a1b2c3d4e5f6", "status": "processing"}
```

**Completed**:
```json
{"type": "completed", "job_id": "job_a1b2c3d4e5f6", "status": "completed", "output_path": "/app/ocr_output/job_a1b2c3d4e5f6"}
```

**Failed**:
```json
{"type": "failed", "job_id": "job_a1b2c3d4e5f6", "status": "failed", "error": "Pipeline exited with code 1"}
```

**Authentication error**:
```json
{"type": "error", "message": "Authentication failed"}
```

#### Close Codes

| Code | Reason |
|------|--------|
| `4001` | Authentication failed |
| `4004` | Job not found |
| `4029` | Too many connections |

#### Client Keepalive

Send `"ping"` as a text frame to receive `{"type": "pong"}`. The server polls job status every 1 second.

#### Behavior Notes

- If the job is already in a terminal state when connecting, the server sends the initial `connected` message followed immediately by the terminal status message, then closes.
- The server uses an in-memory connection map, which limits WebSocket streaming to single-worker deployments. Multi-worker deployments require a shared pub/sub backend.

#### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/jobs/job_a1b2c3d4e5f6');

ws.onopen =  => {
  // Auth via frame (if not using header)
  ws.send(JSON.stringify({type: 'auth', api_key: process.env.OCR_API_KEY}));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case 'connected':
      console.log('Connected, job status:', msg.status);
      break;
    case 'progress':
      console.log('Progress:', msg.status);
      break;
    case 'completed':
      console.log('Done! Output:', msg.output_path);
      break;
    case 'failed':
      console.error('Failed:', msg.error);
      break;
  }
};
```

---

## Transforms (Feature-Gated)

Document transform operations for forensic document processing. All transform endpoints are under `/api/v1/transforms`.

**Requires**: `ENABLE_TRANSFORMS=true` (returns `403 feature_disabled` otherwise)

### GET /api/v1/transforms -- List Transform Operations

List all registered transform operations with metadata.

**Auth**: Required | **Rate limit**: 60/minute

#### Response `200 OK`

```json
{
  "operations": [
    {
      "name": "redact",
      "description": "Redact sensitive content from documents",
      "version": "1.0.0",
      "supported_formats": ["pdf"],
      "output_format": "pdf",
      "parameters": {}
    }
  ],
  "total": 1
}
```

---

### GET /api/v1/transforms/{operation_id} -- Transform Operation Metadata

Get metadata for a specific transform operation.

**Auth**: Required | **Rate limit**: 60/minute

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `403` | `feature_disabled` | Transforms not enabled |
| `404` | `operation_not_found` | Operation does not exist |

---

### POST /api/v1/transforms/execute -- Execute Transform

Execute a transform operation on a server-side file. Execution is synchronous.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

**Content type**: `application/json`

#### Request Body

```json
{
  "operation_id": "redact",
  "input_path": "/app/ocr_source/document.pdf",
  "output_path": "/app/ocr_output/document_redacted.pdf",
  "params": {},
  "validate_input": true,
  "preserve_metadata": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation_id` | `string` | Yes | -- | Registered transform operation name |
| `input_path` | `string` | Yes | -- | Server-side input file path |
| `output_path` | `string` | Yes | -- | Server-side output file path |
| `params` | `object` | No | `{}` | Operation-specific parameters |
| `validate_input` | `boolean` | No | `true` | Validate input before transform |
| `preserve_metadata` | `boolean` | No | `true` | Preserve PDF/image metadata |

Path traversal protection is enforced. `input_path` must be within `SOURCE_FOLDER` or `OUTPUT_FOLDER`. `output_path` must be within `OUTPUT_FOLDER`.

#### Response `200 OK`

```json
{
  "success": true,
  "operation_id": "redact",
  "output_path": "/app/ocr_output/document_redacted.pdf",
  "error_message": null,
  "metadata": {
    "custody": {}
  },
  "pages_processed": 10,
  "warnings": []
}
```

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_config` | Invalid transform configuration |
| `400` | `execution_validation_failed` | Input validation failed during execution |
| `403` | `feature_disabled` | Transforms not enabled |
| `404` | `operation_not_found` | Operation does not exist |
| `404` | `input_not_found` | Input file not found |
| `500` | `execution_failed` | Transform execution error |
| `500` | `validation_gate_failed` | Output failed validation gate |

All error responses from transform execution include a `custody` object with chain-of-custody diagnostics.

---

## Stamps (Feature-Gated)

Document stamping operations (Bates numbering, designations, zone-based stamps). All stamp endpoints are under `/api/v1/stamps`.

**Requires**: `ENABLE_STAMPING=true` (returns `403 feature_disabled` otherwise)

### GET /api/v1/stamps -- List Stamp Operations

List all registered stamp operations with metadata.

**Auth**: Required | **Rate limit**: 60/minute

#### Response `200 OK`

```json
{
  "operations": [
    {
      "name": "bates",
      "description": "Apply sequential Bates numbering",
      "version": "1.0.0",
      "supported_formats": ["pdf"],
      "parameters": {
        "prefix": "string",
        "start_number": "integer",
        "suffix": "string",
        "separator": "string"
      }
    }
  ],
  "total": 1
}
```

---

### GET /api/v1/stamps/{operation_id} -- Stamp Operation Metadata

Get metadata for a specific stamp operation.

**Auth**: Required | **Rate limit**: 60/minute

---

### POST /api/v1/stamps/execute -- Execute Stamp

Execute a stamp operation on a server-side PDF file. Execution is synchronous.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

**Content type**: `application/json`

#### Request Body

```json
{
  "operation_id": "bates",
  "input_path": "/app/ocr_source/document.pdf",
  "output_path": "/app/ocr_output/document_stamped.pdf",
  "placement": "bottom_right",
  "params": {
    "prefix": "ABC",
    "start_number": 1,
    "suffix": "",
    "separator": ""
  },
  "validate_input": true,
  "check_overlap": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation_id` | `string` | Yes | -- | Registered stamp operation name |
| `input_path` | `string` | Yes | -- | Server-side input PDF path |
| `output_path` | `string` | Yes | -- | Server-side output path |
| `placement` | `string` | No | `"bottom_right"` | Stamp placement location |
| `params` | `object` | No | `{}` | Operation-specific parameters |
| `validate_input` | `boolean` | No | `true` | Validate input before stamping |
| `check_overlap` | `boolean` | No | `true` | Detect and warn about stamp overlaps |

#### Response `200 OK`

```json
{
  "success": true,
  "operation_id": "bates",
  "output_path": "/app/ocr_output/document_stamped.pdf",
  "error_message": null,
  "metadata": {
    "custody": {}
  },
  "pages_stamped": 10,
  "stamp_values": ["ABC000001", "ABC000002", "ABC000003"],
  "warnings": []
}
```

#### Error Responses

Same as transform execute endpoint, with `custody` diagnostics in all error responses.

---

## Output Manifest

Output manifest endpoints provide typed artifact discovery for completed jobs. Instead of manually scanning output directories, clients can query the manifest to find all artifacts (PDFs, text, NER results, etc.) produced by a job along with their types, sizes, and download paths.

All output endpoints require authentication.

### GET /api/v1/jobs/{job_id}/outputs -- List Job Outputs

List all output artifacts produced for a job. Returns a manifest with metadata for each artifact including type, filename, relative path, size, and schema version.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`, `viewer`

#### Path Parameters

| Param | Type | Format | Description |
|-------|------|--------|-------------|
| `job_id` | `string` | `^job_[0-9a-f]{12}$` | Job identifier |

#### Response `200 OK`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "artifacts": [
    {
      "output_type": "searchable_pdf",
      "filename": "contract.pdf",
      "relative_path": "EXPORT/PDF/contract.pdf",
      "size_bytes": 1048576,
      "mime_type": "application/pdf",
      "schema_version": "1.0"
    },
    {
      "output_type": "ocr_text",
      "filename": "contract.txt",
      "relative_path": "EXPORT/TEXT/contract.txt",
      "size_bytes": 12345,
      "mime_type": "text/plain",
      "schema_version": "1.0"
    },
    {
      "output_type": "ner",
      "filename": "contract.ner.json",
      "relative_path": "EXPORT/NER/contract.ner.json",
      "size_bytes": 5678,
      "mime_type": "application/json",
      "schema_version": "1.0"
    }
  ],
  "schema_versions": {
    "searchable_pdf": "1.0",
    "ocr_text": "1.0",
    "ner": "1.0"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `string` | Job identifier |
| `artifacts` | `array` | List of output artifacts |
| `artifacts[].output_type` | `string` | Output type identifier (see valid types below) |
| `artifacts[].filename` | `string` | Filename of the artifact |
| `artifacts[].relative_path` | `string` | Path relative to the job output directory |
| `artifacts[].size_bytes` | `integer` | File size in bytes |
| `artifacts[].mime_type` | `string` | MIME type of the artifact |
| `artifacts[].schema_version` | `string` | Schema version for this output type |
| `schema_versions` | `object` | Map of output type to schema version for all found types |

**Valid output types**: `searchable_pdf`, `ocr_text`, `structure`, `entities`, `ner`, `extraction`, `classification`, `validation`, `handwriting`, `signature`, `vertical`, `retrieval`, `custody`

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_job_id` | Malformed job ID |
| `404` | `job_not_found` | Job does not exist |

#### cURL

```bash
curl -H "X-API-Key: your-key" \
  http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/outputs
```

---

### GET /api/v1/jobs/{job_id}/outputs/{output_type} -- Download Output

Download a specific output artifact for a job. Returns the file content directly with the appropriate content type.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`, `viewer`

#### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| `job_id` | `string` | Job identifier (`^job_[0-9a-f]{12}$`) |
| `output_type` | `string` | Output type to download (e.g., `searchable_pdf`, `ocr_text`, `ner`) |

#### Response

Returns a file download with the appropriate content type:

| Output Type | Content-Type |
|-------------|-------------|
| `searchable_pdf` | `application/pdf` |
| `ocr_text` | `text/plain` |
| `structure`, `entities`, `ner`, `extraction`, `classification`, `validation`, `handwriting`, `signature`, `vertical`, `retrieval` | `application/json` |
| `custody` | `application/jsonl` |

The `Content-Disposition` header includes the original filename.

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_job_id` | Malformed job ID |
| `400` | `invalid_output_type` | Unknown output type |
| `404` | `job_not_found` | Job does not exist |
| `404` | `no_outputs` | Job has no output directory |
| `404` | `output_not_found` | Requested output type not found for this job |
| `404` | `file_not_found` | Artifact file missing from disk |

#### cURL

```bash
# Download the searchable PDF
curl -H "X-API-Key: your-key" -o output.pdf \
  http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/outputs/searchable_pdf

# Download NER results
curl -H "X-API-Key: your-key" -o entities.json \
  http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/outputs/ner

# Download extracted text
curl -H "X-API-Key: your-key" -o text.txt \
  http://localhost:8000/api/v1/jobs/job_a1b2c3d4e5f6/outputs/ocr_text
```

---

## Schema Registry

Schema registry endpoints provide JSON Schema definitions for all 14 output types. Clients can use these schemas to validate pipeline output or to discover the structure of each artifact type programmatically.

### GET /api/v1/schemas -- List Schemas

List all available output schema definitions.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`, `viewer`

#### Response `200 OK`

```json
{
  "schemas": [
    { "output_type": "ocr_text", "schema_version": "1.0" },
    { "output_type": "searchable_pdf", "schema_version": "1.0" },
    { "output_type": "structure", "schema_version": "1.0" },
    { "output_type": "entities", "schema_version": "1.0" },
    { "output_type": "ner", "schema_version": "1.0" },
    { "output_type": "extraction", "schema_version": "1.0" },
    { "output_type": "classification", "schema_version": "1.0" },
    { "output_type": "validation", "schema_version": "1.0" },
    { "output_type": "handwriting", "schema_version": "1.0" },
    { "output_type": "signature", "schema_version": "1.0" },
    { "output_type": "vertical", "schema_version": "1.0" },
    { "output_type": "custody", "schema_version": "1.0" },
    { "output_type": "retrieval", "schema_version": "1.0" },
    { "output_type": "output_manifest", "schema_version": "1.0" }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `schemas` | `array` | List of available schema definitions |
| `schemas[].output_type` | `string` | Output type name |
| `schemas[].schema_version` | `string` | Schema version string |

#### cURL

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/schemas
```

---

### GET /api/v1/schemas/{output_type} -- Get Schema

Get the JSON Schema definition for a specific output type. Returns the raw JSON Schema content.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`, `viewer`

#### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| `output_type` | `string` | Output type name (e.g., `ner`, `extraction`, `searchable_pdf`) |

#### Response `200 OK`

Returns the raw JSON Schema as `application/json`. The schema follows the [JSON Schema](https://json-schema.org/) specification.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NER Output",
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": { "type": "string" },
          "text": { "type": "string" },
          "confidence": { "type": "number" },
          "page": { "type": "integer" }
        }
      }
    }
  }
}
```

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_output_type` | Unknown output type |
| `404` | `schema_not_found` | Schema file not found on disk |
| `500` | `schemas_unavailable` | Schema package not installed |

#### cURL

```bash
# Get the NER output schema
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/schemas/ner

# Get the extraction output schema
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/schemas/extraction
```

---

## Queue Operations

Queue operations expose OCR queue alert thresholds for operator tuning. These
endpoints are available with the dashboard/fleet router and persist threshold
updates to `OCR_QUEUE_THRESHOLDS_PATH` or `${OUTPUT_FOLDER}/queue_thresholds.json`.
They configure alerting thresholds only; worker replica counts and autoscaling
remain deployment-controlled through Helm/KEDA values.

### GET /api/v1/queues/thresholds -- List Queue Thresholds

List configured queue alert thresholds.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

#### Response `200 OK`

```json
[
  {
    "queue_name": "ocr_gpu",
    "warning_depth": 500,
    "critical_depth": 1000,
    "warning_wait_seconds": 300.0,
    "critical_wait_seconds": 600.0
  }
]
```

### GET /api/v1/queues/{queue_name}/threshold -- Get Queue Threshold

Return the configured threshold for one queue.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `404` | `queue_threshold_not_found` | No threshold is configured for the queue |

### PUT /api/v1/queues/{queue_name}/threshold -- Update Queue Threshold

Create or replace queue alert thresholds. `critical_depth` must be greater than
or equal to `warning_depth`, and `critical_wait_seconds` must be greater than or
equal to `warning_wait_seconds`.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`

#### Request Body

```json
{
  "warning_depth": 500,
  "critical_depth": 1000,
  "warning_wait_seconds": 300.0,
  "critical_wait_seconds": 600.0
}
```

#### Response `200 OK`

```json
{
  "queue_name": "ocr_gpu",
  "warning_depth": 500,
  "critical_depth": 1000,
  "warning_wait_seconds": 300.0,
  "critical_wait_seconds": 600.0
}
```

#### cURL

```bash
curl -X PUT -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"warning_depth":500,"critical_depth":1000,"warning_wait_seconds":300,"critical_wait_seconds":600}' \
  http://localhost:8000/api/v1/queues/ocr_gpu/threshold
```

---

## Review Queue

Human review queue for documents that fail confidence thresholds or trigger exception routing rules. The review queue uses SQLite-backed storage and supports filtering, pagination, and decision workflows.

All review endpoints are under `/api/v1/review`.

Review IDs follow the format `rev_` followed by 12 hex characters (e.g., `rev_abc123def456`).

### GET /api/v1/review/queue -- List Review Queue

List review queue items with optional filtering by status and reason.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | `string` | `"pending"` | Filter by status: `pending`, `approved`, `rejected`, `reprocess` |
| `reason` | `string` | -- | Filter by review reason (see reasons below) |
| `limit` | `integer` | `50` | Page size (1-200) |
| `offset` | `integer` | `0` | Number of items to skip |

**Valid review reasons**: `low_confidence`, `degraded_quality`, `handwriting_detected`, `dpi_escalation_failed`, `image_only_pages`, `classification_uncertain`, `manual_flag`

#### Response `200 OK`

```json
{
  "items": [
    {
      "review_id": "rev_a1b2c3d4e5f6",
      "job_id": "job_111111111111",
      "reason": "low_confidence",
      "confidence": 0.35,
      "quality_classification": "degraded",
      "status": "pending",
      "reviewer": "",
      "decision_notes": "",
      "created_at": "2026-03-14T10:30:00",
      "reviewed_at": "",
      "metadata": {
        "document_name": "scan_001.pdf",
        "page_count": 5
      }
    }
  ],
  "total": 12
}
```

| Field | Type | Description |
|-------|------|-------------|
| `items` | `array` | List of review queue items |
| `items[].review_id` | `string` | Review item identifier |
| `items[].job_id` | `string` | Associated job identifier |
| `items[].reason` | `string` | Why the item was flagged for review |
| `items[].confidence` | `float` | Overall OCR confidence score (0.0-1.0) |
| `items[].quality_classification` | `string` | Quality classification from validation |
| `items[].status` | `string` | Current review status |
| `items[].reviewer` | `string` | Reviewer identity (empty if not yet reviewed) |
| `items[].decision_notes` | `string` | Notes from the reviewer |
| `items[].created_at` | `string` | When the item was added to the queue |
| `items[].reviewed_at` | `string` | When a decision was made (empty if pending) |
| `items[].metadata` | `object` | Additional context about the document |
| `total` | `integer` | Total items matching the filter |

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_status` | Invalid status filter value |
| `400` | `invalid_reason` | Invalid reason filter value |

#### cURL

```bash
# List pending items
curl -H "X-API-Key: your-key" \
  http://localhost:8000/api/v1/review/queue

# Filter by reason with pagination
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/review/queue?reason=low_confidence&limit=20&offset=0"

# List approved items
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/review/queue?status=approved"
```

---

### GET /api/v1/review/stats -- Review Queue Statistics

Return aggregate review queue statistics.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

#### Response `200 OK`

```json
{
  "pending": 12,
  "approved": 45,
  "rejected": 3,
  "reprocess": 2,
  "total": 62,
  "avg_review_seconds": 120.5,
  "oldest_pending": "2026-03-12T08:15:00"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `pending` | `integer` | Number of items awaiting review |
| `approved` | `integer` | Number of approved items |
| `rejected` | `integer` | Number of rejected items |
| `reprocess` | `integer` | Number of items flagged for reprocessing |
| `total` | `integer` | Total items across all statuses |
| `avg_review_seconds` | `float` | Average time from creation to decision (seconds) |
| `oldest_pending` | `string` | Timestamp of the oldest pending item (empty if none) |

#### cURL

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/review/stats
```

---

### GET /api/v1/review/{review_id} -- Get Review Item

Get details for a specific review item.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

#### Path Parameters

| Param | Type | Format | Description |
|-------|------|--------|-------------|
| `review_id` | `string` | `^rev_[0-9a-f]{12}$` | Review item identifier |

#### Response `200 OK`

Same schema as individual items in the queue list response (`ReviewItemResponse`).

```json
{
  "review_id": "rev_a1b2c3d4e5f6",
  "job_id": "job_111111111111",
  "reason": "low_confidence",
  "confidence": 0.35,
  "quality_classification": "degraded",
  "status": "pending",
  "reviewer": "",
  "decision_notes": "",
  "created_at": "2026-03-14T10:30:00",
  "reviewed_at": "",
  "metadata": {}
}
```

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_review_id` | Malformed review ID |
| `404` | `review_not_found` | Review item does not exist |

#### cURL

```bash
curl -H "X-API-Key: your-key" \
  http://localhost:8000/api/v1/review/rev_a1b2c3d4e5f6
```

---

### POST /api/v1/review/{review_id}/decision -- Submit Review Decision

Submit a review decision for a pending review item. Valid decisions are `approved`, `rejected`, or `reprocess`.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

**Content type**: `application/json`

#### Path Parameters

| Param | Type | Format | Description |
|-------|------|--------|-------------|
| `review_id` | `string` | `^rev_[0-9a-f]{12}$` | Review item identifier |

#### Request Body

```json
{
  "status": "approved",
  "reviewer": "jane.doe@example.com",
  "notes": "Quality acceptable after manual inspection"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `status` | `string` | Yes | -- | Decision: `"approved"`, `"rejected"`, or `"reprocess"` |
| `reviewer` | `string` | No | `""` | Reviewer identity |
| `notes` | `string` | No | `""` | Decision notes or comments |

#### Response `200 OK`

Returns the updated `ReviewItemResponse` with the decision applied.

```json
{
  "review_id": "rev_a1b2c3d4e5f6",
  "job_id": "job_111111111111",
  "reason": "low_confidence",
  "confidence": 0.35,
  "quality_classification": "degraded",
  "status": "approved",
  "reviewer": "jane.doe@example.com",
  "decision_notes": "Quality acceptable after manual inspection",
  "created_at": "2026-03-14T10:30:00",
  "reviewed_at": "2026-03-14T14:22:00",
  "metadata": {}
}
```

#### Error Responses

| Status | Error | Condition |
|--------|-------|-----------|
| `400` | `invalid_review_id` | Malformed review ID |
| `400` | `invalid_request` | Invalid decision status value |
| `404` | `review_not_found` | Review item does not exist |
| `409` | `already_decided` | Review item already has a decision |

#### cURL

```bash
# Approve a review item
curl -X POST http://localhost:8000/api/v1/review/rev_a1b2c3d4e5f6/decision \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"status": "approved", "reviewer": "jane.doe@example.com", "notes": "Looks good"}'

# Reject a review item
curl -X POST http://localhost:8000/api/v1/review/rev_a1b2c3d4e5f6/decision \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"status": "rejected", "reviewer": "john.smith", "notes": "Unreadable scan"}'

# Request reprocessing
curl -X POST http://localhost:8000/api/v1/review/rev_a1b2c3d4e5f6/decision \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"status": "reprocess", "reviewer": "ops-team"}'
```

---

## Exception Routing Rules

Exception routing rules define the conditions under which processed documents are automatically flagged for human review. Rules are evaluated against validation, classification, handwriting, and text data from each processed document.

Exception routing is opt-in via `ENABLE_EXCEPTION_ROUTING=true`.

### GET /api/v1/review/rules -- List Routing Rules

List all configured exception routing rules and their current enabled status. Returns both built-in and custom rules.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

#### Response `200 OK`

```json
{
  "rules": [
    {
      "name": "low_confidence",
      "reason": "low_confidence",
      "enabled": true,
      "description": "Overall confidence below 0.5"
    },
    {
      "name": "degraded_quality",
      "reason": "degraded_quality",
      "enabled": true,
      "description": "Quality classified as degraded or review_required"
    },
    {
      "name": "handwriting_detected",
      "reason": "handwriting_detected",
      "enabled": true,
      "description": "Handwriting content detected in document"
    },
    {
      "name": "image_only_pages",
      "reason": "image_only_pages",
      "enabled": true,
      "description": "More than 3 image-only pages"
    },
    {
      "name": "classification_uncertain",
      "reason": "classification_uncertain",
      "enabled": true,
      "description": "Classification confidence below 0.5"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `rules` | `array` | List of routing rules |
| `rules[].name` | `string` | Rule identifier |
| `rules[].reason` | `string` | Maps to a `ReviewReason` value in the review queue |
| `rules[].enabled` | `boolean` | Whether the rule is currently active |
| `rules[].description` | `string` | Human-readable description of the rule condition |

**Built-in rules**: `low_confidence`, `degraded_quality`, `handwriting_detected`, `image_only_pages`, `classification_uncertain`

**Custom rules**: Additional rules can be loaded from a JSON file specified by the `EXCEPTION_ROUTING_RULES_PATH` environment variable.

**Configurable thresholds**:

| Variable | Default | Description |
|----------|---------|-------------|
| `REVIEW_CONFIDENCE_THRESHOLD` | `0.5` | Confidence below this triggers `low_confidence` |
| `REVIEW_IMAGE_ONLY_THRESHOLD` | `3` | Image-only pages above this triggers `image_only_pages` |
| `REVIEW_CLASSIFICATION_CONFIDENCE_THRESHOLD` | `0.5` | Classification confidence below this triggers `classification_uncertain` |

#### cURL

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/review/rules
```

---

## Entity Recall

Cross-document entity and extraction search endpoints. The recall index provides full-text search across all entities (NER results) and extractions (structured fields) indexed from processed documents.

All recall endpoints are under `/api/v1`.

### GET /api/v1/entities -- Search Entities

Search indexed entities across all processed documents. Supports filtering by entity type, text content, job ID, and minimum confidence. Results are ordered by most recently indexed.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`, `viewer`

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `string` | -- | Filter by entity type (e.g., `PERSON`, `DATE`, `CASE_NUMBER`, `BATES_NUMBER`) |
| `q` | `string` | -- | Text search query (LIKE matching) |
| `job_id` | `string` | -- | Filter by job ID |
| `min_confidence` | `float` | `0.0` | Minimum confidence threshold (0.0-1.0) |
| `limit` | `integer` | `50` | Page size (1-200) |
| `offset` | `integer` | `0` | Number of items to skip |

#### Response `200 OK`

```json
{
  "results": [
    {
      "entity_id": "ent_a1b2c3d4e5f6",
      "job_id": "job_111111111111",
      "entity_type": "PERSON",
      "text": "John Smith",
      "confidence": 0.95,
      "source": "spacy",
      "page": 1,
      "document_name": "contract.pdf"
    },
    {
      "entity_id": "ent_f6e5d4c3b2a1",
      "job_id": "job_111111111111",
      "entity_type": "DATE",
      "text": "January 15, 2026",
      "confidence": 0.92,
      "source": "regex",
      "page": 2,
      "document_name": "contract.pdf"
    }
  ],
  "total": 145,
  "limit": 50,
  "offset": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `results` | `array` | List of matching entities |
| `results[].entity_id` | `string` | Entity identifier |
| `results[].job_id` | `string` | Job that produced this entity |
| `results[].entity_type` | `string` | Entity type (e.g., `PERSON`, `ORG`, `DATE`, `CASE_NUMBER`) |
| `results[].text` | `string` | Extracted entity text |
| `results[].confidence` | `float` | Detection confidence (0.0-1.0) |
| `results[].source` | `string` | Detection source (e.g., `spacy`, `regex`) |
| `results[].page` | `integer` | Page number where the entity was found |
| `results[].document_name` | `string` | Source document filename |
| `total` | `integer` | Total matching entities |
| `limit` | `integer` | Page size used |
| `offset` | `integer` | Offset used |

#### cURL

```bash
# Search for all PERSON entities
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/entities?type=PERSON"

# Text search with minimum confidence
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/entities?q=Smith&min_confidence=0.8"

# Filter by job
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/entities?job_id=job_111111111111&limit=100"
```

---

### GET /api/v1/extractions -- Search Extractions

Search indexed key-value extractions across all processed documents. Supports filtering by field name, value content, job ID, and minimum confidence. Results are ordered by most recently indexed.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`, `viewer`

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `field` | `string` | -- | Filter by field name (e.g., `invoice_number`, `date`, `amount`) |
| `q` | `string` | -- | Value search query (LIKE matching) |
| `job_id` | `string` | -- | Filter by job ID |
| `min_confidence` | `float` | `0.0` | Minimum confidence threshold (0.0-1.0) |
| `limit` | `integer` | `50` | Page size (1-200) |
| `offset` | `integer` | `0` | Number of items to skip |

#### Response `200 OK`

```json
{
  "results": [
    {
      "extraction_id": "ext_a1b2c3d4e5f6",
      "job_id": "job_111111111111",
      "field_name": "invoice_number",
      "field_value": "INV-2026-00123",
      "confidence": 0.88,
      "page": 1,
      "document_name": "invoice.pdf"
    },
    {
      "extraction_id": "ext_f6e5d4c3b2a1",
      "job_id": "job_222222222222",
      "field_name": "invoice_number",
      "field_value": "INV-2026-00456",
      "confidence": 0.91,
      "page": 1,
      "document_name": "invoice_batch_2.pdf"
    }
  ],
  "total": 38,
  "limit": 50,
  "offset": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `results` | `array` | List of matching extractions |
| `results[].extraction_id` | `string` | Extraction identifier |
| `results[].job_id` | `string` | Job that produced this extraction |
| `results[].field_name` | `string` | Extracted field name |
| `results[].field_value` | `string` | Extracted field value |
| `results[].confidence` | `float` | Extraction confidence (0.0-1.0) |
| `results[].page` | `integer` | Page number where the field was found |
| `results[].document_name` | `string` | Source document filename |
| `total` | `integer` | Total matching extractions |
| `limit` | `integer` | Page size used |
| `offset` | `integer` | Offset used |

#### cURL

```bash
# Search for all invoice numbers
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/extractions?field=invoice_number"

# Value search
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/extractions?q=2026-00123"

# Filter by job with confidence threshold
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/api/v1/extractions?job_id=job_111111111111&min_confidence=0.7"
```

---

### GET /api/v1/recall/stats -- Recall Index Statistics

Return entity and extraction index statistics. Provides an overview of the total indexed data.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`, `viewer`

#### Response `200 OK`

```json
{
  "total_entities": 1523,
  "total_extractions": 487,
  "unique_entity_types": 8,
  "unique_field_names": 12,
  "jobs_indexed": 42
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_entities` | `integer` | Total entities in the index |
| `total_extractions` | `integer` | Total extractions in the index |
| `unique_entity_types` | `integer` | Number of distinct entity types |
| `unique_field_names` | `integer` | Number of distinct extraction field names |
| `jobs_indexed` | `integer` | Number of jobs with indexed data |

#### cURL

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/recall/stats
```

---

## Semantic Search / VLM

VLM-backed semantic search and document analysis endpoints. All endpoints are under `/api/v1/search`.

These endpoints are AI-adjacent analyst-assist features. They are explicitly feature-gated, additive to the primary OCR pipeline, and must not be treated as the source of forensic truth for baseline OCR artifacts or custody evidence. See [architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).

**Requires**: `VLM_ENABLED=true` and a valid `VLM_ENDPOINT_URL`.

### POST /api/v1/search/semantic -- Semantic Search

Search documents using natural-language queries via VLM.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`, `viewer`

**Content type**: `application/json`

#### Request Body

```json
{
  "query": "What is the effective date of the contract?",
  "document_id": "doc_123",
  "max_results": 10,
  "min_score": 0.5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | `string` | Yes | -- | Natural-language search query (1-2000 chars) |
| `document_id` | `string` | No | -- | Scope search to a specific document |
| `max_results` | `integer` | No | `10` | Maximum results (1-100) |
| `min_score` | `float` | No | `0.0` | Minimum relevance score threshold (0.0-1.0) |

#### Response `200 OK`

```json
{
  "query": "What is the effective date of the contract?",
  "results": [
    {
      "text": "This agreement is effective as of January 1, 2026.",
      "score": 0.95,
      "page": 1,
      "document_id": "doc_123",
      "bbox": [72.0, 200.0, 540.0, 220.0]
    }
  ],
  "total": 1,
  "model": "vision-llm",
  "processing_time_ms": 450.0
}
```

---

### POST /api/v1/search/analyze -- Analyze Document

Analyze document pages using VLM for entity extraction and summarization.

**Auth**: Required | **Rate limit**: 60/minute | **RBAC**: `admin`, `operator`

**Content type**: `application/json`

#### Request Body

```json
{
  "pages": [
    {
      "page_number": 1,
      "text": "The parties agree to the following terms...",
      "image_b64": null
    }
  ],
  "prompt": "Extract all named entities and dates",
  "document_id": "doc_123"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `pages` | `array` | Yes | -- | List of page dicts (min 1) |
| `pages[].page_number` | `integer` | Yes | -- | Page number |
| `pages[].text` | `string` | Yes | -- | Page text content |
| `pages[].image_b64` | `string` | No | -- | Base64-encoded page image |
| `prompt` | `string` | No | -- | Analysis instruction (max 4000 chars) |
| `document_id` | `string` | No | -- | Document identifier |

#### Response `200 OK`

```json
{
  "entities": [
    {"type": "DATE", "text": "January 1, 2026", "page": 1}
  ],
  "summary": "A contractual agreement between two parties...",
  "relationships": [],
  "confidence": 0.92,
  "model": "vision-llm",
  "processing_time_ms": 1200.0
}
```

---

### GET /api/v1/search/vlm/health -- VLM Health Check

Check VLM gateway connectivity. **No authentication required** (allows monitoring probes).

**Rate limit**: 60/minute

#### Response `200 OK`

```json
{
  "vlm_enabled": true,
  "vlm_reachable": true,
  "model_name": "vision-llm",
  "endpoint_configured": true
}
```

---

## Translation API (Feature-Gated)

EDCOCR exposes translation-adjacent integration endpoints only when
`ENABLE_TRANSLATION_API=true`. These endpoints are optional and are not part of
the baseline OCR evidence contract. The OCR service also exposes
`GET /api/v1/translation/readiness` without that gate so operators can see
whether the external EXTERNAL_TRANSLATION dependency is reachable from EDCOCR.

The translation router enforces local/offline validation constraints: quality
estimation loads only from configured local model paths, and request-time model
downloads are not enabled by default.

### Translation Jobs and Quality Scoring

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/translation/jobs` | Submit a translation request stub or route through tenant-aware engine selection when requested |
| `POST` | `/api/v1/translation/score-pair` | Score one source/target pair with the local COMETKiwi quality estimator |

`POST /api/v1/translation/score-pair` returns `503` when the quality-estimation
runtime or configured local model is unavailable. The endpoint is rate-limited
to 10 requests per minute.

### Translation Tenant Configuration and Glossary

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/translation/tenants/{tenant_id}/config` | Read tenant translation routing config |
| `PUT` | `/api/v1/translation/tenants/{tenant_id}/config` | Create or replace tenant translation routing config |
| `GET` | `/api/v1/translation/tenants/{tenant_id}/glossary` | List tenant glossary entries |
| `POST` | `/api/v1/translation/tenants/{tenant_id}/glossary` | Create glossary entry |
| `PATCH` | `/api/v1/translation/tenants/{tenant_id}/glossary/{entry_id}` | Update glossary entry |
| `DELETE` | `/api/v1/translation/tenants/{tenant_id}/glossary/{entry_id}` | Delete glossary entry |

### Translation Batches

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/translation/batches` | Submit a batch translation request |
| `GET` | `/api/v1/translation/batches/{batch_id}` | Poll batch status |
| `GET` | `/api/v1/translation/batches/{batch_id}/results` | Fetch terminal batch results |
| `POST` | `/api/v1/translation/batches/{batch_id}/cancel` | Cancel a pending or running translation batch |

`POST /api/v1/translation/batches` is rate-limited to 5 requests per minute and
returns `503` if the Django-backed translation batch store is not configured.

---

## Admin / Multi-Tenancy (Feature-Gated)

Tenant and API key management endpoints. All endpoints are under `/api/v1/admin`.

**Requires**: `ENABLE_MULTITENANCY=true` (endpoints are not registered otherwise)

All admin endpoints require the `admin` permission. Platform-level operations additionally require `platform_admin`.

### POST /api/v1/admin/tenants -- Create Tenant

Create a new tenant organization.

**Permission**: `platform_admin`

**Content type**: `application/json`

#### Request Body

```json
{
  "name": "acme-corp",
  "display_name": "Acme Corporation",
  "tier": "standard",
  "max_concurrent_jobs": 4,
  "max_pages_per_month": 10000,
  "max_storage_bytes": 10737418240,
  "allowed_features": ["docintel"],
  "admin_email": "admin@acme.com"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `string` | Yes | -- | Tenant name (1-256 chars) |
| `display_name` | `string` | No | -- | Display name (max 256 chars) |
| `tier` | `string` | No | `"standard"` | `"free"`, `"standard"`, or `"enterprise"` |
| `max_concurrent_jobs` | `integer` | No | `4` | Max concurrent jobs (1-100) |
| `max_pages_per_month` | `integer` | No | `10000` | Monthly page quota (100-10M) |
| `max_storage_bytes` | `integer` | No | `10 GiB` | Storage quota (>= 1 MiB) |
| `allowed_features` | `array` | No | `[]` | Enabled features list |
| `admin_email` | `string` | No | -- | Admin contact email |

#### Response `201 Created`

```json
{
  "tenant_id": "tenant_a1b2c3d4e5f6",
  "name": "acme-corp",
  "display_name": "Acme Corporation",
  "status": "active",
  "tier": "standard",
  "created_at": "2026-03-14T10:30:00",
  "updated_at": null,
  "max_concurrent_jobs": 4,
  "max_pages_per_month": 10000,
  "max_storage_bytes": 10737418240,
  "allowed_features": ["docintel"],
  "admin_email": "admin@acme.com"
}
```

---

### GET /api/v1/admin/tenants -- List Tenants

List all tenants. Non-platform-admins see only their own tenant.

**Permission**: `admin`

#### Query Parameters

| Param | Type | Description |
|-------|------|-------------|
| `status` | `string` | Filter by status (`active`, `suspended`) |

---

### GET /api/v1/admin/tenants/{tenant_id} -- Tenant Details

Get tenant details including current usage summary.

**Permission**: `admin` (scoped to own tenant unless `platform_admin`)

#### Response `200 OK`

Includes all `TenantResponse` fields plus a `usage` object with current-period metrics.

---

### PUT /api/v1/admin/tenants/{tenant_id} -- Update Tenant

Update tenant configuration fields. Only provided fields are updated.

**Permission**: `admin` (scoped to own tenant unless `platform_admin`)

---

### POST /api/v1/admin/tenants/{tenant_id}/suspend -- Suspend Tenant

Suspend a tenant. All API keys become inactive.

**Permission**: `admin` (scoped to own tenant unless `platform_admin`)

---

### POST /api/v1/admin/tenants/{tenant_id}/activate -- Activate Tenant

Re-activate a suspended tenant.

**Permission**: `admin` (scoped to own tenant unless `platform_admin`)

---

### POST /api/v1/admin/tenants/{tenant_id}/keys -- Create API Key

Create a new API key for a tenant. The raw API key is returned only once in the response.

**Permission**: `admin` (scoped to own tenant unless `platform_admin`)

#### Request Body

```json
{
  "name": "Production Key",
  "permissions": ["submit", "read"],
  "expires_at": "2027-01-01T00:00:00Z"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `string` | No | -- | Key name (max 256 chars) |
| `permissions` | `array` | No | `["submit", "read"]` | Permission list |
| `expires_at` | `datetime` | No | -- | Key expiration (optional) |

Available permissions: `submit`, `read`, `admin`, `platform_admin`.

Only `platform_admin` callers can create keys with the `platform_admin` permission.

#### Response `201 Created`

```json
{
  "key_id": "key_a1b2c3d4e5f6",
  "api_key": "<newly generated value>",
  "name": "Production Key",
  "permissions": ["submit", "read"],
  "created_at": "2026-03-14T10:30:00",
  "expires_at": "2027-01-01T00:00:00Z"
}
```

**Important**: The `api_key` value is shown only once. Store it securely.

---

### DELETE /api/v1/admin/tenants/{tenant_id}/keys/{key_id} -- Revoke API Key

Revoke an API key. The key becomes immediately invalid.

**Permission**: `admin` (scoped to own tenant unless `platform_admin`)

**Key ID format**: `^key_[0-9a-f]{12}$`

**Response**: `204 No Content`

---

### GET /api/v1/admin/tenants/{tenant_id}/usage -- Usage Report

Get usage report for a tenant for a specific billing period.

**Permission**: `admin` (scoped to own tenant unless `platform_admin`)

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `period` | `string` | Current month | Billing period (e.g., `"2026-03"`) |

#### Response `200 OK`

```json
{
  "tenant_id": "tenant_a1b2c3d4e5f6",
  "period": "2026-03",
  "jobs_submitted": 150,
  "pages_processed": 4500,
  "storage_bytes_used": 2147483648,
  "api_calls": 1200,
  "processing_seconds": 7200.5,
  "estimated_costs": {
    "currency": "USD",
    "page_cost_usd": 4.50,
    "storage_ingest_cost_usd": 2.00,
    "api_call_cost_usd": 0.12,
    "processing_cost_usd": 2.00,
    "total_cost_usd": 8.62,
    "storage_gib_ingested": 2.0,
    "processing_hours": 2.0,
    "rates": {
      "per_page_usd": 0.001,
      "per_gib_ingested_usd": 1.0,
      "per_api_call_usd": 0.0001,
      "per_processing_hour_usd": 1.0
    }
  }
}
```

Cost rates are configured via environment variables (see [Configuration Reference](#configuration-reference)).

---

### GET /api/v1/admin/tenants/{tenant_id}/slo -- SLO Snapshot

Get a rolling-window SLO (Service Level Objective) snapshot for a tenant.

**Permission**: `admin` (scoped to own tenant unless `platform_admin`)

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `window_hours` | `integer` | `24` | Rolling window size in hours (>= 1) |

#### Response `200 OK`

```json
{
  "tenant_id": "tenant_a1b2c3d4e5f6",
  "window_hours": 24,
  "window_start": "2026-03-13T10:30:00Z",
  "window_end": "2026-03-14T10:30:00Z",
  "jobs_total": 100,
  "terminal_jobs": 98,
  "completed_jobs": 95,
  "failed_jobs": 3,
  "cancelled_jobs": 0,
  "active_jobs": 2,
  "pages_processed": 2500,
  "success_rate": 0.9694,
  "failure_rate": 0.0306,
  "avg_processing_seconds": 45.2,
  "p95_processing_seconds": 120.0,
  "throughput_jobs_per_hour": 4.08,
  "throughput_pages_per_hour": 104.2,
  "targets": {
    "success_rate_min": 0.95,
    "p95_processing_seconds_max": 1800.0
  },
  "status": {
    "success_rate_met": true,
    "p95_processing_met": true,
    "overall_met": true
  }
}
```

---

## Webhooks

Webhook notifications are delivered when jobs or batches reach terminal states (`completed`, `failed`, `cancelled`). Configure a webhook by providing `webhook_url` and optionally `webhook_secret` during job or batch submission.

### Payload Schema (Job)

```json
{
  "event": "job.completed",
  "timestamp": "2026-03-14T10:31:15.123456+00:00",
  "job_id": "job_a1b2c3d4e5f6",
  "status": "completed",
  "source_file": "contract.pdf",
  "processing": {
    "started_at": "2026-03-14T10:30:01.000000",
    "completed_at": "2026-03-14T10:31:15.000000",
    "processing_time_seconds": 74.0,
    "pages_completed": 10,
    "total_pages": 10
  },
  "error_message": null
}
```

When `WEBHOOK_ENRICH_ENTITIES=true` is set and the job completes successfully, an `entities` array is included with PII/PHI bounding box data:

```json
{
  "event": "job.completed",
  "entities": [
    {
      "entity_type": "PERSON_NAME",
      "confidence_score": 0.9512,
      "page_index": 0,
      "bounding_box": [100.0, 200.0, 300.0, 220.0]
    }
  ]
}
```

### Event Types

| Event | Trigger |
|-------|---------|
| `job.completed` | Job finishes successfully |
| `job.failed` | Job fails |
| `job.cancelled` | Job is cancelled |
| `batch.completed` | All batch jobs complete |
| `batch.partial_failure` | Some batch jobs failed |
| `batch.failed` | All batch jobs failed |
| `batch.cancelled` | Batch is cancelled |

### Payload Schema (Batch)

```json
{
  "event": "batch.completed",
  "timestamp": "2026-03-14T10:35:00.000000+00:00",
  "batch_id": "batch_f1e2d3c4b5a6",
  "status": "completed",
  "total_jobs": 3,
  "jobs_completed": 3,
  "jobs_failed": 0,
  "jobs_cancelled": 0,
  "processing_time_seconds": 300.0,
  "jobs": [
    { "job_id": "job_111111111111", "status": "completed", "source_file": "doc1.pdf" },
    { "job_id": "job_222222222222", "status": "completed", "source_file": "doc2.pdf" },
    { "job_id": "job_333333333333", "status": "completed", "source_file": "doc3.pdf" }
  ]
}
```

### HMAC-SHA256 Signature Verification

When a `webhook_secret` is provided (per-job or via the `WEBHOOK_SECRET` environment variable), each delivery includes signature headers:

| Header | Description |
|--------|-------------|
| `X-Webhook-Signature` | `sha256={hex_digest}` |
| `X-Webhook-Timestamp` | Unix timestamp (integer) |
| `X-Webhook-Event` | Event type (e.g., `job.completed`) |
| `X-Webhook-Job-ID` | Job or batch ID |

**Signature computation**:

```
signed_payload = "{timestamp}.{payload_json}"
signature = "sha256=" + HMAC-SHA256(secret, signed_payload)
```

#### Verification Example (Python)

```python
import hashlib
import hmac

def verify_webhook(payload_body: bytes, signature: str, timestamp: str, secret: str) -> bool:
    """Verify webhook HMAC-SHA256 signature."""
    message = f"{timestamp}.{payload_body.decode('utf-8')}"
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256).hexdigest
    return hmac.compare_digest(signature, expected)
```

### Delivery Details

| Property | Value |
|----------|-------|
| HTTP method | `POST` |
| Content-Type | `application/json` |
| User-Agent | `OCR-Pipeline-Webhook/{version}` |
| Timeout | `WEBHOOK_TIMEOUT` seconds (default 30, max 120) |
| Max retries | `WEBHOOK_MAX_RETRIES` (default 3, max 10) |
| Retry delays | 5s, 10s, 20s, 40s (exponential backoff) |
| Success codes | `2xx` range |

### Retry and Dead-Letter Queue

Failed deliveries are retried with exponential backoff. After all retries are exhausted:
- `webhook_status` is set to `"failed"` on the job/batch record
- The payload is added to the webhook dead-letter queue (DLQ)
- DLQ entries can be inspected and retried via the `/api/v1/webhooks/dlq` endpoints

### SSRF Protection

Webhook URLs are validated at both submission and delivery time:
- HTTPS required by default (override with `WEBHOOK_ALLOW_HTTP=true`)
- Private/internal IPs blocked by default (override with `WEBHOOK_ALLOW_PRIVATE=true`)
- DNS resolution is validated against known private ranges

### Secret Storage and `WEBHOOK_SECRET_KEY`

Per-job webhook secrets are encrypted at rest in the API database using a Fernet key derived from an environment variable.

| Env Var | Purpose |
|---------|---------|
| `WEBHOOK_SECRET_KEY` | **Preferred.** Master key used to derive the Fernet encryption key for per-job webhook secrets. Must be set in production. |
| `OCR_API_KEY` | Fallback key used if `WEBHOOK_SECRET_KEY` is not set. A WARNING is logged on first use. |
| `WEBHOOK_SECRET` | Default HMAC signing secret when a job does not carry its own `webhook_secret`. |
| `WEBHOOK_TIMEOUT` | HTTP request timeout in seconds (default 30, max 120). |
| `WEBHOOK_MAX_RETRIES` | Maximum retry attempts (default 3, max 10). |
| `WEBHOOK_ALLOW_HTTP` | Allow `http://` webhook URLs (default `false`). |
| `WEBHOOK_ALLOW_PRIVATE` | Allow private/loopback webhook targets (default `false`). |
| `WEBHOOK_ENRICH_ENTITIES` | Include PII/PHI bounding boxes in `job.completed` payloads (default `false`). |

> [!IMPORTANT]
> `WEBHOOK_SECRET_KEY` must be set **before** the first production webhook is created. Changing it later will invalidate all previously encrypted per-job secrets; plaintext legacy values continue to decrypt with a WARNING.

The key derivation uses `base64.urlsafe_b64encode(sha256(raw_key))`. Plaintext secrets from pre-encryption versions fall back transparently with a WARNING log so upgrades remain safe.

### Delivery Headers Summary

Every delivery attempt (initial and retry) sends the following headers. The signature and timestamp are refreshed on every retry so receivers that enforce timestamp freshness still accept re-delivered payloads:

| Header | Example | Notes |
|--------|---------|-------|
| `Content-Type` | `application/json` | Always JSON |
| `User-Agent` | `OCR-Pipeline-Webhook/4.1.0` | Version from `version.__version__` |
| `X-Webhook-Event` | `job.completed` | Event type |
| `X-Webhook-Job-ID` | `job_a1b2c3d4e5f6` | Job id (or `X-Webhook-Batch-ID` for batches) |
| `X-Webhook-Signature` | `sha256=c0ffee...` | Only present when a secret is configured |
| `X-Webhook-Timestamp` | `1741953075` | Refreshed every attempt |

### Webhook Status Values

| Status | Description |
|--------|-------------|
| `null` | No webhook configured |
| `pending` | Delivery in progress |
| `delivered` | Successfully delivered (2xx response) |
| `failed` | All delivery attempts exhausted |

---

## Coordinator API (Django)

The Django coordinator provides pipeline metrics for distributed deployments. These endpoints run on the coordinator HTTP surface (default port 8000).

### GET /api/v1/metrics/ -- Pipeline Metrics (JSON)

Return pipeline health metrics as JSON.

**Auth**: `METRICS_API_KEY` via `X-Api-Key` header or `Authorization: Bearer` header. Unauthenticated when `METRICS_API_KEY` is not set.

#### Response `200 OK`

```json
{
  "jobs": {
    "by_status": {
      "submitted": 2,
      "processing": 3,
      "completed": 150,
      "failed": 5,
      "cancelled": 1,
      "pending": 0
    },
    "total": 161,
    "error_rate_1h": 0.0312
  },
  "workers": {
    "by_status": {
      "online": 3,
      "offline": 1,
      "busy": 2,
      "draining": 0
    },
    "total": 6,
    "gpu_available": 4
  },
  "pages": {
    "total_processed": 4500,
    "avg_processing_time_ms": 245.3
  },
  "timestamp": "2026-03-14T10:30:00.000000+00:00"
}
```

#### cURL

```bash
# With API key
curl -H "X-Api-Key: your-metrics-key" http://localhost:8000/api/v1/metrics/

# With Bearer token
curl -H "Authorization: Bearer your-metrics-key" http://localhost:8000/api/v1/metrics/
```

---

### GET /api/v1/prometheus/ -- Prometheus Metrics

Return pipeline metrics in Prometheus text exposition format.

**Auth**: Same as `/api/v1/metrics/`.

**Content-Type**: `text/plain; version=0.0.4; charset=utf-8`

Returns 7 metric families from the custom ORM-backed Prometheus collector:

- `ocr_jobs_total` (counter by status)
- `ocr_jobs_active` (gauge)
- `ocr_pages_processed_total` (counter)
- `ocr_page_processing_seconds` (histogram)
- `ocr_workers_total` (gauge by status)
- `ocr_workers_gpu_available` (gauge)
- `ocr_error_rate_1h` (gauge)

#### cURL

```bash
curl -H "X-Api-Key: your-metrics-key" http://localhost:8000/api/v1/prometheus/
```

---

### GET /dashboard/ -- Operator Dashboard

HTML dashboard showing recent jobs and fleet status. Requires Django admin access.

---

## Configuration Reference

Environment variables that affect API behavior:

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port (1-65535) |
| `EXPOSE_API_DOCS` | `false` | Mount `/docs`, `/redoc`, and `/openapi.json` when true |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_API_KEY` | (empty) | Legacy API key for X-API-Key authentication |
| `ALLOW_UNAUTHENTICATED` | `false` | Allow unauthenticated access (dev mode only) |
| `ANONYMOUS_ROLE` | `viewer` | Role assigned when `ALLOW_UNAUTHENTICATED=true` |
| `API_ALLOWED_IPS` | (empty) | Comma-separated IP allowlist |
| `OAUTH2_ENABLED` | `false` | Enable OAuth2/OIDC bearer token auth |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_RATE_LIMIT` | `60/minute` | Default rate limit for read endpoints |
| `OCR_SUBMIT_RATE_LIMIT` | `10/minute` | Rate limit for submit/retry endpoints |

### Job Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_JOBS` | `4` | Maximum active jobs (1-64) |
| `MAX_UPLOAD_SIZE_MB` | `5120` | Maximum upload size in MB (max 51200) |
| `MAX_REQUEST_BODY_SIZE` | `10485760` | Non-multipart request body cap in bytes; `0` disables |
| `MAX_BATCH_SIZE` | `50` | Maximum jobs per batch (max 500) |
| `JOB_PROCESSING_TIMEOUT_MINUTES` | `30` | Default processing timeout (max 10080) |
| `PIPELINE_POLL_INTERVAL` | `5` | Progress polling interval in seconds |
| `RESULT_RETENTION_DAYS` | `90` | Days to retain results (max 3650) |

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `SOURCE_FOLDER` | `/app/ocr_source` | Input file root directory |
| `OUTPUT_FOLDER` | `/app/ocr_output` | Output file root directory |
| `API_DB_PATH` | `{OUTPUT_FOLDER}/jobs.db` | SQLite database path |
| `OCR_QUEUE_THRESHOLDS_PATH` | `{OUTPUT_FOLDER}/queue_thresholds.json` | Persisted queue alert threshold config |
| `PIPELINE_SCRIPT` | `ocr_gpu_async.py` | Pipeline script path |

### Webhooks

| Variable | Default | Description |
|----------|---------|-------------|
| `WEBHOOK_SECRET` | (empty) | Default HMAC secret for webhook signing |
| `WEBHOOK_TIMEOUT` | `30` | HTTP request timeout in seconds (max 120) |
| `WEBHOOK_MAX_RETRIES` | `3` | Maximum retry attempts (max 10) |
| `WEBHOOK_ALLOW_HTTP` | `false` | Allow non-HTTPS webhook URLs |
| `WEBHOOK_ALLOW_PRIVATE` | `false` | Allow private/internal IP webhook URLs |
| `WEBHOOK_ENRICH_ENTITIES` | `false` | Include PII entities in webhook payloads |

### Webhook Dead-Letter Queue

| Variable | Default | Description |
|----------|---------|-------------|
| `WEBHOOK_DLQ_ENABLED` | `true` | Enable webhook DLQ |
| `WEBHOOK_DLQ_PATH` | `{OUTPUT_FOLDER}/logs/webhook_dlq.jsonl` | DLQ file path |

### Event Store

| Variable | Default | Description |
|----------|---------|-------------|
| `EVENT_STORE_ENABLED` | `true` | Enable durable event store |
| `EVENT_STORE_PATH` | `{OUTPUT_FOLDER}/event_store.db` | SQLite event store path |
| `EVENT_RETENTION_HOURS` | `72` | Event retention (max 8760) |
| `API_EVENT_STREAM_ENABLED` | `false` | Enable JSONL event stream |
| `API_EVENT_STREAM_PATH` | `{OUTPUT_FOLDER}/logs/api-events.jsonl` | Event stream file path |

### Audit Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `API_AUDIT_LOG_ENABLED` | `true` | Enable API request audit logging |
| `API_AUDIT_LOG_PATH` | (empty) | Custom audit log file path |
| `API_AUDIT_EXCLUDE_HEALTH` | `false` | Exclude health checks from audit |

### Feature Gates

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TRANSFORMS` | `false` | Enable transform endpoints |
| `ENABLE_STAMPING` | `false` | Enable stamp endpoints |
| `ENABLE_MULTITENANCY` | `false` | Enable multi-tenant admin endpoints |
| `ENABLE_DASHBOARD` | `false` | Enable dashboard, fleet, alert, queue-threshold, and analytics endpoints |
| `ENABLE_TRANSLATION_API` | `false` | Enable optional translation job, tenant config, glossary, QE, and batch endpoints |
| `OCR_FEDERATION_CUSTODY_ENABLED` | `false` | Enable federation custody ingest endpoint |
| `VLM_ENABLED` | `false` | Enable VLM semantic search endpoints |
| `ENABLE_EXCEPTION_ROUTING` | `false` | Enable automatic exception routing to review queue |
| `ENABLE_RETRIEVAL_OUTPUT` | `false` | Enable unified retrieval output assembly |

### Multi-Tenancy Cost Rates

| Variable | Default | Description |
|----------|---------|-------------|
| `TENANT_COST_PER_PAGE_USD` | `0.0` | Cost per page processed |
| `TENANT_COST_PER_GIB_INGESTED_USD` | `0.0` | Cost per GiB of ingested storage |
| `TENANT_COST_PER_API_CALL_USD` | `0.0` | Cost per API call |
| `TENANT_COST_PER_PROCESSING_HOUR_USD` | `0.0` | Cost per processing hour |

### Multi-Tenancy SLO Targets

| Variable | Default | Description |
|----------|---------|-------------|
| `TENANT_SLO_WINDOW_HOURS` | `24` | Default SLO window (1-720 hours) |
| `TENANT_SLO_TARGET_SUCCESS_RATE` | `0.95` | Target success rate (0.0-1.0) |
| `TENANT_SLO_TARGET_P95_PROCESSING_SECONDS` | `1800.0` | Target p95 processing time |

### Exception Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `REVIEW_CONFIDENCE_THRESHOLD` | `0.5` | Confidence below this triggers low_confidence review |
| `REVIEW_IMAGE_ONLY_THRESHOLD` | `3` | Image-only pages above this triggers review |
| `REVIEW_CLASSIFICATION_CONFIDENCE_THRESHOLD` | `0.5` | Classification confidence below this triggers review |
| `EXCEPTION_ROUTING_RULES_PATH` | (empty) | Path to custom routing rules JSON file |

### Coordinator Metrics

| Variable | Default | Description |
|----------|---------|-------------|
| `METRICS_API_KEY` | (empty) | API key for coordinator metrics endpoints |

---

## Endpoint Summary

| Category | Endpoints | Feature Gate | Server |
|----------|-----------|-------------|--------|
| Health / readiness / features | 6 | Always on | FastAPI |
| Jobs | 7 | Always on | FastAPI |
| Batch | 5 | Always on | FastAPI |
| Events & DLQ | 4 | Always on | FastAPI |
| WebSocket | 1 | Always on | FastAPI |
| Output Manifest | 2 | Always on | FastAPI |
| Schema Registry | 2 | Always on | FastAPI |
| Dashboard / fleet / alerts / analytics / queue operations | 14 | `ENABLE_DASHBOARD` | FastAPI |
| Review Queue | 4 | Always on | FastAPI |
| Exception Routing | 1 | Always on | FastAPI |
| Entity Recall | 3 | Always on | FastAPI |
| Transforms | 3 | `ENABLE_TRANSFORMS` | FastAPI |
| Stamps | 3 | `ENABLE_STAMPING` | FastAPI |
| Semantic Search / VLM | 3 | `VLM_ENABLED` | FastAPI |
| Translation | 12 | `ENABLE_TRANSLATION_API` | FastAPI |
| Federation custody ingest | 1 | `OCR_FEDERATION_CUSTODY_ENABLED` | FastAPI |
| Admin / Multi-Tenancy | 10 | `ENABLE_MULTITENANCY` | FastAPI |
| Interactive docs | 3 | `EXPOSE_API_DOCS` | FastAPI |
| Coordinator Metrics | 3 | Separate server | Django |

The exact mounted surface depends on feature gates. Use `/openapi.json` in an
environment with `EXPOSE_API_DOCS=true` to inspect the live route set.

---

## Related Documentation

- [Information Flows](03-INFORMATION-FLOWS.md) -- Data flow diagrams
- [Configuration Reference](06-CONFIGURATION-REFERENCE.md) -- All env vars (pipeline + API)
- [Forensic-Core vs AI-Adjacent Boundary](architecture/forensic-ai-boundary-contract.md) -- Capability contract for optional intelligence layers
- [Transforms and Stamping](07-TRANSFORMS-STAMPING.md) -- Document transforms, stamps, and custody controls
- [System Blueprint](00-SYSTEM-BLUEPRINT.md) -- Architecture overview
- [Quickstart Guide](02-QUICKSTART-5-MINUTE-SUCCESS.md) -- Getting started
- [Failover Runbook](FAILOVER-RUNBOOK.md) -- Operational procedures

---

*Last Updated: 2026-05-20 | Version 4.1.0*
