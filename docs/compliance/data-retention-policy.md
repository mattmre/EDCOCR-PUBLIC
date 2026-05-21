# Data Retention Policy -- EDCOCR v1.0

## Overview

This document defines the data classification, retention periods, deletion procedures, and audit requirements for all data classes processed and stored by the EDCOCR pipeline. It addresses HIPAA 164.530(j) record retention, SOC 2 CC5.3 data protection, FedRAMP SI-12 information management, and GDPR Article 17 right-to-erasure obligations.

## Data Classification

EDCOCR processes and stores data across five classification levels. Each level has distinct handling, retention, and destruction requirements.

### Level 1: Forensic Evidence (Highest Sensitivity)

- **Description**: Original source documents, searchable OCR PDFs, extracted text, and image-only fallback pages.
- **Examples**: `ocr_output/EXPORT/PDF/*.pdf`, `ocr_output/EXPORT/TEXT/*.txt`, source files in `ocr_source/`.
- **Sensitivity**: May contain PII, PHI, privileged legal content, or classified information depending on source material.
- **Storage locations**: NFS (`nfs_job_path`), S3 (`storage_backend_used=s3`), local filesystem.

### Level 2: PII/PHI Entities (High Sensitivity)

- **Description**: Named entities extracted by the NER module and PII/PHI spatial extraction pipeline.
- **Examples**: `PiiEntity` model records (SSN, DOB, EMAIL, PHONE, NAME, MEDICAL_RECORD_NUMBER), NER sidecar JSON files in `EXPORT/NER/`.
- **Sensitivity**: Contains personally identifiable and protected health information with spatial bounding boxes.
- **Storage locations**: PostgreSQL (`PiiEntity` table), NER JSON files on disk/S3.

### Level 3: Audit Logs (Medium-High Sensitivity)

- **Description**: Chain-of-custody JSONL files, API audit logs, coordinator `CustodyEvent` records.
- **Examples**: `ocr_output/custody/*.custody.jsonl`, `ocr_output/logs/api-audit.jsonl`, `CustodyEvent` table rows.
- **Sensitivity**: Contains document hashes, processing metadata, API request details, and authentication outcomes. Must be retained longer than source data for compliance verification.
- **Storage locations**: Local filesystem (JSONL), PostgreSQL (`CustodyEvent`), S3 (archived logs).

### Level 4: Job Metadata (Medium Sensitivity)

- **Description**: Job records, page results, worker assignments, and processing configuration.
- **Examples**: `Job` model, `PageResult` model, `Worker` model records.
- **Sensitivity**: Contains file paths, processing statistics, and configuration details. No direct PII unless embedded in file names.
- **Storage locations**: PostgreSQL (coordinator database).

### Level 5: Operational Data (Low Sensitivity)

- **Description**: Prometheus metrics, Grafana dashboards, temporary processing artifacts, crash-resume temp files.
- **Examples**: `ocr_temp/<doc_hash>/` page PDFs, Prometheus TSDB, log files in `ocr_output/logs/`.
- **Sensitivity**: Aggregated metrics and transient processing state. Minimal PII risk.
- **Storage locations**: Local filesystem, Prometheus TSDB, Redis (ephemeral context).

---

## Retention Periods

Retention periods are configurable via environment variables. The defaults below reflect a balanced approach suitable for most forensic and compliance use cases. Organizations must adjust these based on their specific regulatory obligations.

| Data Class | Env Variable | Default | Minimum (Compliance) | Maximum | Notes |
|------------|-------------|---------|----------------------|---------|-------|
| Forensic evidence (PDFs/text) | `DOCUMENT_RETENTION_DAYS` | 365 | 6 years (HIPAA), 7 years (SOC2) | Unlimited | Retain per litigation hold or regulatory mandate |
| PII/PHI entities | `PII_ENTITY_RETENTION_DAYS` | 90 | 0 (GDPR erasure) | 2555 (7 years) | Subject to right-to-erasure requests |
| Chain-of-custody logs | `AUDIT_LOG_RETENTION_DAYS` | 2555 (7 years) | 6 years (HIPAA) | Unlimited | Must outlive source data retention |
| API audit logs | `API_AUDIT_RETENTION_DAYS` | 2555 (7 years) | 6 years (HIPAA) | Unlimited | Hash-chained, append-only |
| Job metadata | `JOB_RETENTION_DAYS` | 30 | 7 | 3650 | Already implemented in `cleanup_old_jobs.py` |
| Job results (API) | `RESULT_RETENTION_DAYS` | 90 | 7 | 3650 | Already implemented in `api/config.py` |
| Event stream data | `EVENT_RETENTION_HOURS` | 72 | 1 | 8760 | Already implemented in `api/event_store.py` |
| Temp/crash-resume files | N/A (auto) | Until job completes | 0 | 30 | Cleaned by `purge_temp_files.py` |
| Prometheus metrics | `PROMETHEUS_RETENTION` | 15d | 7d | 90d | Configured in Prometheus server |
| Redis ephemeral context | `REDIS_CONTEXT_TTL` | 300s | 60s | 3600s | Auto-expires via Redis TTL |

### Compliance-Specific Overrides

- **HIPAA deployments**: Set `AUDIT_LOG_RETENTION_DAYS=2190` (6 years minimum per 45 CFR 164.530(j)).
- **SOC 2 deployments**: Set `AUDIT_LOG_RETENTION_DAYS=2555` (7 years recommended for audit evidence).
- **FedRAMP deployments**: Follow agency-specific retention schedules; minimum 3 years for system records per NARA GRS.
- **GDPR deployments**: Set `PII_ENTITY_RETENTION_DAYS` to the minimum period necessary for the documented processing purpose. Configure automated deletion or implement on-demand erasure.

---

## Environment Variable Status

| Env Variable | Status | Where Used |
|-------------|--------|------------|
| `JOB_RETENTION_DAYS` | Implemented | `cleanup_old_jobs.py`, `coordinator/jobs/tasks.py` |
| `RESULT_RETENTION_DAYS` | Implemented | `api/config.py` (default 90, max 3650) |
| `EVENT_RETENTION_HOURS` | Implemented | `api/config.py`, `api/event_store.py` (default 72) |
| `PII_ENTITY_RETENTION_DAYS` | Implemented | `cleanup_old_jobs.py --include-pii` (default 90) |
| `AUDIT_LOG_RETENTION_DAYS` | Implemented (constant) | `cleanup_old_jobs.py` (default 2555 = 7 years, read by `_get_retention_days`) |
| `DOCUMENT_RETENTION_DAYS` | Implemented (constant) | `cleanup_old_jobs.py` (default 365, read by `_get_retention_days`) |
| `LITIGATION_HOLD` | Implemented | `cleanup_old_jobs.py` -- when `true`, all automated deletions are suspended |
| `API_AUDIT_RETENTION_DAYS` | Proposed | Not yet implemented -- requires JSONL archive/rotation |

### Implementation Notes

1. **`PII_ENTITY_RETENTION_DAYS`**: Implemented as `--include-pii` flag on `cleanup_old_jobs` management command. Deletes `PiiEntity` rows older than the configured period and records a `pii_deleted` custody event. Run as a Celery Beat periodic task alongside `cleanup_old_jobs`.

2. **`LITIGATION_HOLD`**: When set to `true`, `yes`, or `1`, the `cleanup_old_jobs` command exits immediately without deleting anything and prints a warning. This provides a global emergency stop for all automated data deletion.

3. **Deletion audit trail**: All deletion operations (job cleanup and PII cleanup) now create `CustodyEvent` records before deletion, capturing the retention policy, count of records to be deleted, cutoff date, and reason.

4. **`AUDIT_LOG_RETENTION_DAYS`**: Custody JSONL files and `CustodyEvent` rows should be archived (not deleted) after the retention period. Archive to cold S3 storage with a custody event recording the archive action.

5. **`API_AUDIT_RETENTION_DAYS`**: Rotate `api-audit.jsonl` files by date. Archive rotated files to S3 with checksum verification before local deletion.

6. **`DOCUMENT_RETENTION_DAYS`**: The env var is read and available. A dedicated `cleanup_old_documents` management command targeting `EXPORT/PDF/`, `EXPORT/TEXT/`, and associated sidecar directories is recommended for a future iteration.

---

## Deletion Procedures

### Procedure 1: Job Metadata Deletion (Implemented)

**Trigger**: Automated via Celery Beat periodic task or manual `cleanup_old_jobs` command.

**Steps**:
1. Query `Job` records with `status IN (completed, failed, cancelled)` and `created_at < cutoff`.
2. For each matching job, remove NFS job directory if `nfs_job_path` is set and directory exists.
3. Django cascade delete removes `Job`, `PageResult`, `CustodyEvent`, and `PiiEntity` rows.
4. Log deletion count to stdout.

**Existing command**: `python manage.py cleanup_old_jobs --days N`

**Audit**: A `data_deleted` custody event is recorded before deletion, capturing the retention policy, count, cutoff date, and reason. The `LITIGATION_HOLD` env var blocks all automated deletions when active.

### Procedure 2: PII/PHI Entity Deletion (Proposed)

**Trigger**: Automated periodic task, or on-demand for GDPR right-to-erasure requests.

**Steps**:
1. Identify `PiiEntity` rows older than `PII_ENTITY_RETENTION_DAYS`.
2. For each entity batch, compute SHA-256 hash of the serialized entity data.
3. Record a deletion custody event: `{event_type: "pii_entities_deleted", data: {count, hash_of_deleted, requestor, reason}}`.
4. Delete the `PiiEntity` rows.
5. If NER sidecar JSON files exist on disk/S3, remove them and record the file hash.

**GDPR right-to-erasure flow**:
1. Receive erasure request with subject identifier (name, email, etc.).
2. Query `PiiEntity` for matching `entity_value` across all jobs.
3. Record the erasure request in the API audit log with the requestor identity.
4. Delete matching entities and record custody events.
5. Return confirmation with deletion count and timestamp.
6. Retain the deletion audit record (the audit log entry itself is not subject to erasure as it serves a legitimate compliance purpose per GDPR Article 17(3)(b)).

### Procedure 3: Audit Log Archival (Proposed)

**Trigger**: Automated periodic task.

**Steps**:
1. Identify custody JSONL files and API audit JSONL files older than `AUDIT_LOG_RETENTION_DAYS`.
2. Verify chain integrity (`verify_custody_file`) before archival.
3. Upload to cold S3 storage tier (Glacier or equivalent).
4. Verify S3 upload checksum matches local file checksum.
5. Record archival event: `{event_type: "audit_log_archived", data: {file, hash, s3_key, archive_date}}`.
6. Remove local file only after successful S3 verification.

**Important**: Audit logs must never be permanently deleted while the data they describe is still within its retention period. Audit log retention must always equal or exceed the longest data retention period in the system.

### Procedure 4: Document Output Deletion (Proposed)

**Trigger**: Automated periodic task or manual command.

**Steps**:
1. Identify output files in `EXPORT/PDF/`, `EXPORT/TEXT/`, and sidecar directories older than `DOCUMENT_RETENTION_DAYS`.
2. For each file, compute SHA-256 hash before deletion.
3. Record deletion custody event: `{event_type: "document_deleted", data: {file_path, hash, size_bytes, deletion_reason}}`.
4. Delete the file.
5. Verify deletion (confirm file no longer exists).

---

## Audit Trail for Deletion Events

All deletion operations must produce an auditable record containing:

| Field | Description | Example |
|-------|-------------|---------|
| `event_type` | Type of deletion | `pii_entities_deleted`, `document_deleted`, `job_cleaned` |
| `timestamp` | ISO 8601 UTC timestamp | `2026-03-24T14:30:00.000+00:00` |
| `requestor` | Identity of the person or system that initiated deletion | `celery_beat`, `admin@example.com`, `gdpr_request_42` |
| `reason` | Reason for deletion | `retention_policy`, `gdpr_erasure_request`, `manual_cleanup` |
| `count` | Number of records/files deleted | `47` |
| `hash_of_deleted` | SHA-256 hash of the serialized deleted data | `a1b2c3d4...` |
| `retention_policy` | Retention policy that triggered the deletion | `PII_ENTITY_RETENTION_DAYS=90` |

These deletion audit records are written to the chain-of-custody log and are themselves subject to the audit log retention period (longest retention in the system).

---

## Litigation Hold

When a litigation hold is in effect:

1. **All automated deletion is suspended** for documents within the hold scope.
2. The hold scope is defined by job IDs, date ranges, or document classifications.
3. A custody event records the hold activation: `{event_type: "litigation_hold_activated", data: {scope, authority, date}}`.
4. Hold release requires explicit authorization and a corresponding custody event.
5. Implementation: Set `LITIGATION_HOLD=true` environment variable to globally suspend all cleanup commands, or use a per-job `litigation_hold` field (proposed model addition).

---

## Verification Commands

### Verify chain-of-custody integrity

```bash
# Verify a single custody file
python -c "from custody import verify_custody_file; ok, msg = verify_custody_file('ocr_output/custody/abc123.custody.jsonl'); print(msg)"

# Verify all custody files
find ocr_output/custody/ -name '*.custody.jsonl' -exec python -c "
from custody import verify_custody_file
import sys
ok, msg = verify_custody_file(sys.argv[1])
print(f'{sys.argv[1]}: {msg}')
if not ok: sys.exit(1)
" {} \;
```

### Preview retention cleanup (dry run)

```bash
# Job metadata cleanup preview
python manage.py cleanup_old_jobs --days 30 --dry-run

# Orphaned temp file cleanup preview
python manage.py purge_temp_files --dry-run
```

---

## Regulatory Reference Matrix

| Requirement | HIPAA | SOC 2 | FedRAMP | GDPR |
|-------------|-------|-------|---------|------|
| Record retention period | 6 years (164.530(j)) | 7 years (recommended) | Per NARA GRS | Purpose-limited |
| Audit log retention | 6 years | 7 years | 3 years minimum | Purpose-limited |
| PHI/PII deletion | On authorization termination | Per data protection policy | Per agency schedule | Right to erasure (Art. 17) |
| Deletion audit trail | Required | Required (CC5.3) | Required (SI-12) | Required (accountability) |
| Encryption at rest | Addressable (164.312(a)(2)(iv)) | Required (CC5.3) | Required (SC-28) | Required (Art. 32) |

---

## Revision History

| Date | Version | Author | Change |
|------|---------|--------|--------|
| 2026-03-24 | 1.0 | Compliance remediation | Initial data retention policy |
