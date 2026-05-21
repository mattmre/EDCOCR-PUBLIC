# Audit Logging Completeness Assessment -- EDCOCR v1.0

## Overview

This document inventories all audit logging capabilities in the EDCOCR pipeline, identifies gaps where events are not currently logged, and provides recommendations for closing those gaps. It covers two independent audit logging systems: the file-based chain-of-custody module (`custody.py`) and the coordinator-level database audit table (`CustodyEvent` model).

**Last Updated**: 2026-05-20

---

## Audit Logging Systems

EDCOCR implements three distinct audit logging subsystems:

### 1. Chain of Custody (`custody.py`) -- File-Based JSONL

- **Purpose**: Tamper-evident processing audit trail for individual documents.
- **Format**: Append-only JSONL (one JSON object per line).
- **Location**: `ocr_output/custody/<document_hash>.custody.jsonl`
- **Integrity**: SHA-256 hash chain. Each event's `hash` field is the SHA-256 of the event payload, and the `prev_hash` field links to the preceding event's hash. Genesis events have `prev_hash: null`.
- **Thread safety**: Protected by `threading.Lock` per chain instance.

### 2. Coordinator CustodyEvent Model -- Database

- **Purpose**: Centralized audit trail for distributed pipeline jobs.
- **Storage**: PostgreSQL `jobs_custodyevent` table.
- **Fields**: `document_id`, `job` (FK), `event_type`, `timestamp`, `worker_hostname`, `data` (JSON), `prev_hash`, `event_hash`, `chain_finalized`.
- **Cascade**: Events are deleted when the parent `Job` is deleted via Django cascade.

### 3. API Audit Log (`api/audit.py`) -- File-Based JSONL

- **Purpose**: Request-level API audit trail for all REST API interactions.
- **Format**: Append-only JSONL with SHA-256 hash chain (same pattern as custody.py).
- **Location**: `ocr_output/logs/api-audit.jsonl` (configurable via `API_AUDIT_LOG_PATH`).
- **Fields**: Request ID, method, path, status code, auth outcome, identity, timestamp, duration, client IP.
- **Exclusions**: Health check paths are excluded by default. API keys are explicitly not logged.

---

## Events Currently Logged

### Chain of Custody Events (`custody.py`)

The following event types are defined in `EVENT_TYPES` and recorded during pipeline processing:

| Event Type | When Recorded | Data Fields |
|------------|--------------|-------------|
| `file_ingested` | Source file accepted for processing | source_path, file_hash, file_size, file_type |
| `page_extracted` | Page image extracted from document | page_num, dpi, image_size |
| `ocr_primary` | PaddleOCR successfully processed page | page_num, language, confidence, text_length |
| `ocr_fallback` | Tesseract fallback OCR processed page | page_num, language, confidence, text_length |
| `ocr_image_only` | OCR failed, page preserved as image | page_num, reason |
| `language_detected` | Language detection completed | detected_language, confidence |
| `language_reprocess` | Page re-processed with detected language model | page_num, original_language, new_language |
| `docintel_analysis` | Document Intelligence analysis completed | mode, regions_found, tables_found |
| `assembly_complete` | Final document assembled from pages | total_pages, output_path |
| `compression_complete` | PDF compression/optimization completed | original_size, compressed_size, ratio |
| `dpi_escalation` | Page re-extracted at higher DPI | page_num, original_dpi, new_dpi, reason |
| `processing_failed` | Processing stage failed | stage, error, page_num (if applicable) |

### Coordinator CustodyEvent Records

The coordinator records custody events during distributed job processing. These mirror the chain-of-custody events but are stored in PostgreSQL with additional fields for distributed tracking:

| Event Type | When Recorded | Additional Fields |
|------------|--------------|-------------------|
| `job_submitted` | Job created in coordinator | job_id, source_file, priority |
| `page_dispatched` | Page task sent to worker | page_num, worker_hostname, queue |
| `page_completed` | Worker completed page OCR | page_num, confidence, method |
| `page_failed` | Worker failed to process page | page_num, error_message |
| `job_completed` | All pages assembled | total_pages, processing_time_ms |
| `job_failed` | Job failed | error_message |

### API Audit Log Events

The API audit middleware records every API request (except excluded health paths):

| Field | Description |
|-------|-------------|
| `request_id` | Unique ID (from `X-Request-ID` header or auto-generated `req_` prefix) |
| `method` | HTTP method (GET, POST, DELETE, etc.) |
| `path` | Request URL path |
| `status_code` | HTTP response status code |
| `auth_outcome` | `authorized`, `unauthorized`, `forbidden`, or `unknown` |
| `identity` | Authenticated identity (tenant ID or key prefix) |
| `timestamp` | ISO 8601 UTC |
| `duration_ms` | Request processing time |
| `client_ip` | Client IP address (from `request.client.host`) |

---

## Events NOT Currently Logged (Gaps)

### Gap 1: Data Deletion Events

**Severity**: High

The `cleanup_old_jobs` management command deletes jobs, page results, custody events, and PII entities without recording the deletion in any audit log. This means the chain of custody has a silent gap when data is removed.

**What is missing**:
- No custody event for job record deletion
- No hash of deleted data preserved
- No record of who initiated the deletion or why
- NFS directory removal is not audited

**Recommendation**: Record a `data_deleted` custody event before performing the deletion. Include the count of deleted records, a SHA-256 hash of the serialized data, the requestor identity (e.g., `celery_beat` or admin username), and the retention policy that triggered the deletion.

### Gap 2: PII/PHI Entity Lifecycle

**Severity**: High

PII entities detected during processing are stored in the `PiiEntity` model but there is no audit trail for:
- When entities are accessed (queried by API consumers)
- When entities are deleted (either by cascade or explicit cleanup)
- GDPR right-to-erasure request handling

**Recommendation**: Add a `pii_access` event to the API audit log when PII entity endpoints are queried. Add a `pii_deleted` custody event when PII entities are removed.

### Gap 3: Configuration Changes

**Severity**: Medium

Changes to runtime configuration (environment variables, feature flags, tenant quotas) are not logged. An operator could change `ENABLE_SIGNATURE_VERIFICATION` or `OCR_TASK_ROUTING` without any audit record.

**What is missing**:
- No record of environment variable changes at startup
- No record of feature flag toggles
- No record of tenant quota modifications

**Recommendation**: Log effective configuration at application startup (excluding secrets). Log tenant quota changes via the admin API audit trail.

### Gap 4: Authentication Failures (Detailed)

**Severity**: Medium

The API audit log records `unauthorized` and `forbidden` outcomes but does not capture:
- The specific authentication method attempted
- Whether it was a key expiration, suspension, or invalid key
- Rate limit events with client identity

**Recommendation**: Enrich the `auth_outcome` field with sub-categories: `key_expired`, `key_suspended`, `key_invalid`, `tenant_suspended`, `rate_limited`.

### Gap 5: Worker Registration and Deregistration

**Severity**: Medium

Workers register via heartbeat but there is no audit event when:
- A new worker joins the fleet
- A worker goes offline (heartbeat timeout)
- A worker is manually drained or removed

**Recommendation**: Add coordinator-level audit events for `worker_registered`, `worker_offline`, and `worker_drained`.

### Gap 6: S3/Storage Backend Operations

**Severity**: Low

File operations on the S3 storage backend (upload, download, presigned URL generation) are not audited beyond standard S3 access logs (which are infrastructure-level, not application-level).

**What is missing**:
- No application-level record of presigned URL generation
- No record of which worker accessed which S3 object

**Recommendation**: Log presigned URL generation with job ID, object key, and expiry time. S3 access logging provides the download audit trail.

### Gap 7: Webhook Delivery Audit

**Severity**: Low

Webhook delivery attempts are logged at the application log level but not in the hash-chained audit trail. Failed deliveries and retry attempts are visible in logs but not in the custody chain.

**Recommendation**: Add `webhook_delivered` and `webhook_failed` custody events with delivery timestamp, HTTP status, and retry count.

---

## Log Integrity Verification

### Verifying Chain of Custody Files

The `custody.py` module provides built-in verification:

```python
from custody import verify_custody_file

# Verify a single custody file
is_valid, message = verify_custody_file("ocr_output/custody/abc123.custody.jsonl")
print(f"Valid: {is_valid}, Message: {message}")
```

The verification process:
1. Loads all events from the JSONL file.
2. For each event, verifies that `prev_hash` matches the hash of the preceding event (or `null` for the genesis event).
3. Recomputes the SHA-256 hash of each event's payload and compares it to the stored `hash` field.
4. Reports the first broken link or tampered event if integrity is violated.

### Verifying API Audit Logs

The API audit log (`api/audit.py`) uses the same hash-chain pattern. Verification can be performed by loading the JSONL file and checking the `prev_hash` linkage, following the same algorithm as custody chain verification.

### Batch Verification Script

For operational use, verify all custody files in a directory:

```bash
# Verify all custody chains
python -c "
import os, sys
from custody import verify_custody_file

custody_dir = 'ocr_output/custody'
if not os.path.isdir(custody_dir):
    print('No custody directory found'); sys.exit(0)

failed = 0
total = 0
for f in os.listdir(custody_dir):
    if f.endswith('.custody.jsonl'):
        total += 1
        path = os.path.join(custody_dir, f)
        ok, msg = verify_custody_file(path)
        if not ok:
            print(f'FAIL: {f}: {msg}')
            failed += 1

print(f'Verified {total} chains: {total - failed} passed, {failed} failed')
sys.exit(1 if failed else 0)
"
```

### Coordinator CustodyEvent Verification

The coordinator `CustodyEvent` model stores `prev_hash` and `event_hash` fields. Verification can be performed via a Django management command or direct database query:

```python
from jobs.models import CustodyEvent

# Verify chain for a specific document
events = CustodyEvent.objects.filter(document_id="abc123").order_by("timestamp")
prev_hash = ""
for event in events:
    if event.prev_hash != prev_hash:
        print(f"Broken chain at {event.id}: expected {prev_hash}, got {event.prev_hash}")
        break
    prev_hash = event.event_hash
else:
    print(f"Chain verified: {events.count} events")
```

---

## Gap Remediation Roadmap

| Priority | Gap ID | Description | Effort | Compliance Impact |
|----------|--------|-------------|--------|-------------------|
| P1 | Gap 1 | Data deletion audit events | 2 days | HIPAA 164.530(j), SOC2 CC5.3, FedRAMP SI-12 |
| P1 | Gap 2 | PII/PHI entity lifecycle logging | 2 days | HIPAA 164.312(b), GDPR Art. 17/30 |
| P2 | Gap 3 | Configuration change logging | 1 day | SOC2 CC7.1, FedRAMP CM-3 |
| P2 | Gap 4 | Detailed auth failure categories | 1 day | SOC2 CC6.1, FedRAMP AC-7 |
| P2 | Gap 5 | Worker registration audit events | 1 day | SOC2 CC7.2, FedRAMP CA-7 |
| P3 | Gap 6 | S3 storage operation logging | 1 day | SOC2 CC5.3, FedRAMP AU-2 |
| P3 | Gap 7 | Webhook delivery audit events | 0.5 day | SOC2 CC7.5 |

**Total estimated effort: 8.5 days**

---

## Summary

### Strengths

- **Tamper-evident audit trail**: SHA-256 hash-chained JSONL exceeds typical audit logging requirements for HIPAA, SOC 2, and FedRAMP. Integrity can be cryptographically verified at any time.
- **Dual-layer logging**: Both file-based (custody JSONL) and database-based (CustodyEvent) audit trails provide redundancy.
- **API request auditing**: Every API request is logged with correlation IDs, authentication outcomes, and timing information.
- **No secrets in logs**: API keys are explicitly excluded from audit logs.

### Weaknesses

- **Deletion events are not audited**: The most significant gap. Data disposal should always produce an audit record.
- **PII lifecycle is incomplete**: Entity creation is logged but access and deletion are not.
- **Configuration changes are invisible**: Feature flag and runtime configuration changes leave no audit trace.

### Compliance Assessment

| Framework | Audit Requirement | Current Status |
|-----------|-------------------|----------------|
| HIPAA 164.312(b) | Audit controls for ePHI access | Partial -- processing logged, access/deletion not logged |
| SOC 2 CC5.3 | Data protection audit trail | Partial -- processing logged, deletion not logged |
| SOC 2 CC7.1 | Change detection logging | Met for code changes, gap for runtime config |
| FedRAMP AU-2 | Event logging | Met for processing events, gap for data lifecycle |
| FedRAMP AU-12 | Audit record generation | Met -- automatic for all processing events |
| GDPR Art. 30 | Records of processing activities | Partial -- processing logged, PII lifecycle gaps |

---

## Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-03-24 | 1.0 | Initial audit logging completeness assessment |
