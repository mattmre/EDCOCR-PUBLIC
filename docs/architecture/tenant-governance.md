# Tenant governance and independent data stores

**Candidate**: v4.2.0-rc.1; source design with bounded local validation
**Scope**: Source-of-truth designation for cost and SLA data; dual-path purge documentation

---

## Overview

The EDCOCR platform has two independent data paths for tenant governance:

1. **Coordinator path** (Django + PostgreSQL): distributed pipeline data
2. **API path** (FastAPI + SQLite): local single-server job data

This document designates the authoritative source of truth for cost and SLA data, and documents the intentional dual-path purge design.

## Authoritative Sources of Truth

### Cost Data

| Layer | Module | Role |
|-------|--------|------|
| **Authoritative** | `api/database.py` `UsageRecord` table | SQLite-backed persistent cost records |
| Cache | `cost_tracking.CostTracker` | In-memory optimization cache; may be lost on restart |
| Bridge | `api/prometheus.py` | Exports `ocr_cost_estimate_total` Prometheus metric from cache |

The SQLite `UsageRecord` table in `api/database.py` is the authoritative cost source. The in-memory `CostTracker` singleton in `cost_tracking.py` is a convenience cache for hot-path access and Prometheus metric bridging. It is not durable across process restarts.

### SLA Data

| Layer | Module | Role |
|-------|--------|------|
| **Authoritative** | `api/slo.py` Job-table snapshot | Computed from SQLite Job records with sliding window query |
| Cache | `sla_monitoring.SLAMonitor` | In-memory sliding-window optimization cache |
| Bridge | `api/prometheus.py` | Exports per-tenant SLA gauges from cache |

The authoritative SLA source is the `api/slo.py` module, which computes SLO compliance from the persistent Job table. The `SLAMonitor` singleton in `sla_monitoring.py` maintains in-memory sliding windows for fast metric access but is not durable across restarts.

## Dual-Path Purge Design

The dual-path purge is **intentional**, not accidental. Each path purges the data it owns:

### Coordinator Purge (`purge_tenant` management command)

Handles:
- PostgreSQL Job, PageResult, PiiEntity, CustodyEvent records
- Coordinator-owned translation batches/inputs, tenant configs, and glossary entries
- NFS output directories (when `--include-output` is set)
- S3 objects (when `--include-output` is set and `STORAGE_BACKEND=s3`)
- In-memory cost tracking cache reset
- In-memory SLA monitoring window reset
- Custody event audit trail (recorded before deletion)
- Litigation hold enforcement

### API Purge

Handles:
- SQLite Job, UsageRecord tables
- Local filesystem output cleanup
- Rate limiter cache reset

### Why Two Paths

The coordinator and API server run as separate processes with separate databases. The coordinator uses PostgreSQL for distributed job management, while the API server uses SQLite for local job tracking. Both maintain their own tenant data, and both must be purged independently to achieve complete tenant data removal.

## Known Limitations

1. **Isolated Prometheus registries**: The coordinator and API server each have their own Prometheus registry. Metrics from one process are not visible in the other. A unified Prometheus scrape configuration is required to see both.

2. **Process-local rate limit cache**: `api/tenant_rate_limiter.py` maintains a process-local cache of tenant rate limits. Purging a tenant does not automatically clear rate limit state in other API server instances. The cache refreshes on its own schedule.

3. **Non-atomic cross-path purge**: There is no single transaction spanning both PostgreSQL and SQLite. If the coordinator purge succeeds but the API purge is not run (or vice versa), tenant data may remain in the un-purged database. Operators must run both purge paths.

4. **In-memory cache drift**: Between process restarts and purge operations, the in-memory caches in `CostTracker` and `SLAMonitor` may hold stale tenant data. These caches are automatically cleared during purge, but only within the same process.
5. **Translation-only tenants**: The coordinator purge command also purges independent translation rows when a tenant has no OCR `Job` rows; the API-plane purge remains a separate database/process operation.

## Candidate deletion limits

Stop tenant admission and drain work before purge. Staged OCR cleanup must
settle before deletion proceeds. S3 jobs retain backend-bound cleanup records
after the Job row is deleted, so the recovery scheduler can remove late uploads.
Cleanup is eventual and covers current objects; historical versions, object-lock
copies and backup replicas need separate lifecycle handling. See
[release scope](../release-candidate.md). Migration 0032 refuses reversal while
cleanup records exist, and older binaries do not service the records.
