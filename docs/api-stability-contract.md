# API Stability Contract — v1.0

**Effective**: v1.0.0 release
**Scope**: All `/api/v1/` REST endpoints, WebSocket endpoint, webhook payloads

## Versioning Policy

This project follows [Semantic Versioning 2.0.0](https://semver.org/).

### Version Format: MAJOR.MINOR.PATCH

- **MAJOR**: Breaking changes to stable API surface
- **MINOR**: Backward-compatible additions (new endpoints, optional fields)
- **PATCH**: Bug fixes, security patches, no API changes

### API Version Prefix

All REST endpoints use `/api/v1/` prefix. This prefix is part of the stability contract.

## Stability Tiers

### Tier 1: Stable (breaking change = major version bump)

These endpoints and models are covered by the stability guarantee:

**Job Management** (7 endpoints):
- POST /api/v1/jobs — Submit OCR job
- GET /api/v1/jobs — List jobs
- GET /api/v1/jobs/{job_id} — Get job status
- GET /api/v1/jobs/{job_id}/result — Get job result
- GET /api/v1/jobs/{job_id}/result/download — Download artifact
- POST /api/v1/jobs/{job_id}/retry — Retry failed job
- DELETE /api/v1/jobs/{job_id} — Cancel job

**Batch Management** (5 endpoints):
- POST /api/v1/jobs/batch — Submit batch
- GET /api/v1/jobs/batch — List batches
- GET /api/v1/jobs/batch/{batch_id} — Get batch status
- DELETE /api/v1/jobs/batch/{batch_id} — Cancel batch
- POST /api/v1/jobs/batch/{batch_id}/retry — Retry batch

**Health** (1 endpoint):
- GET /api/v1/health — Health check (no auth)

**WebSocket**:
- WS /ws/jobs/{job_id} — Real-time job progress

**Webhook Payload**: The webhook JSON schema (event, timestamp, job_id, status, source_file, processing, error_message) is stable.

**Core Models**: JobSubmitRequest, JobSubmitResponse, JobStatusResponse, JobListResponse, JobResultResponse, HealthResponse, ErrorResponse, BatchSubmitRequest, BatchSubmitResponse, BatchStatusResponse, BatchListResponse

### Tier 2: Beta (may change in minor versions)

These endpoints are functional but may see non-breaking additions or refinements:

**Transforms** (3 endpoints, gated by ENABLE_TRANSFORMS):
- GET/POST /api/v1/transforms/*

**Stamps** (3 endpoints, gated by ENABLE_STAMPING):
- GET/POST /api/v1/stamps/*

**Events & DLQ** (4 endpoints):
- GET /api/v1/jobs/{job_id}/events
- GET/POST /api/v1/webhooks/dlq/*

**Semantic Search** (3 endpoints):
- POST /api/v1/search/semantic
- POST /api/v1/search/analyze
- GET /api/v1/search/vlm/health

### Tier 3: Experimental (may change or be removed)

**Admin / Multi-Tenancy** (10 endpoints, gated by ENABLE_MULTITENANCY):
- All /api/v1/admin/* endpoints

## Breaking Change Definition

A **breaking change** is any of:
1. Removing an endpoint
2. Changing an endpoint's HTTP method
3. Removing a required request parameter
4. Removing a response field that was previously present
5. Changing the type of an existing response field
6. Changing authentication requirements (adding auth where none was required)
7. Changing error response codes for the same error condition

The following are **NOT** breaking changes:
1. Adding a new optional request parameter
2. Adding a new response field
3. Adding a new endpoint
4. Adding a new enum value to an existing string field
5. Improving error messages (without changing error codes)
6. Adding new WebSocket message types
7. Adding new webhook event types

## Deprecation Policy

1. Deprecated features are announced in CHANGELOG.md
2. Deprecated features include `Deprecated` header in HTTP responses
3. Deprecated features are maintained for at least 2 minor versions
4. Removal only happens in major version bumps

## Rate Limiting

Rate limits are operational, not contractual. They may be adjusted without a version bump.
Current defaults:
- Submit: 10/minute
- General: 60/minute
- Batch submit: 5/minute

## Authentication

- API key via `X-API-Key` header (stable)
- Roles: `viewer`, `operator`, `admin`, `platform_admin` (stable)
- WebSocket auth: header, first-frame JSON, or query param token (stable)

## Environment Variables

Environment variables prefixed with `OCR_` are considered part of the public interface.
New env vars may be added in minor versions. Existing env var semantics follow the breaking change policy.
