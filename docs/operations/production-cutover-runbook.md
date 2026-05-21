# Production Cutover Runbook -- EDCOCR Distributed Pipeline

**Version**: 1.2.0 | **Last Updated**: 2026-05-20

This runbook covers the end-to-end procedure for promoting the EDCOCR distributed
pipeline from development/staging to production. It covers credential replacement,
storage migration, staged canary deployment, monitoring setup, and rollback
procedures.

**Prerequisites**: Familiarity with the [Failover Runbook](../FAILOVER-RUNBOOK.md)
and the [Distributed Readiness Checklist](../deployment/distributed-readiness-checklist.md).

---

## Table of Contents

1. [Pre-Cutover Checklist](#1-pre-cutover-checklist)
2. [S3 Credential Replacement](#2-s3-credential-replacement)
3. [NFS-to-S3 Migration](#3-nfs-to-s3-migration)
4. [Staged Canary Deployment](#4-staged-canary-deployment)
5. [Production Monitoring](#5-production-monitoring)
6. [Rollback Procedures](#6-rollback-procedures)
7. [Post-Cutover Validation](#7-post-cutover-validation)
8. [Appendix: Quick Reference Commands](#appendix-quick-reference-commands)

---

## 1. Pre-Cutover Checklist

Complete every item below before beginning the cutover. Do not proceed until all
items are checked.

### 1.1 Credential and Secret Rotation

- [ ] `DJANGO_SECRET_KEY` rotated to a cryptographically random value (minimum 50 characters).
- [ ] `POSTGRES_PASSWORD` rotated and stored in a secret manager.
- [ ] `RABBITMQ_PASSWORD` rotated and stored in a secret manager.
- [ ] `REDIS_PASSWORD` rotated and stored in a secret manager.
- [ ] `FLOWER_PASSWORD` rotated.
- [ ] `METRICS_API_KEY` set to a unique value for metrics endpoint authentication.
- [ ] `OCR_API_KEY` set for API job submission authentication (used by `X-API-Key` header on `/api/v1/jobs/` endpoints).
- [ ] S3 credentials replaced (see [Section 2](#2-s3-credential-replacement)).
- [ ] All secrets removed from version control and `.env` files excluded from git.

### 1.2 Environment Validation

Run the environment validation script with strict placeholder rejection:

```bash
python scripts/validate_phase7c_env.py \
  --env-file coordinator/.env \
  --strict-placeholders \
  --report docs/reports/phase7c-production-env-validation.md
```

**Expected result**: Exit code 0 and `PASS` status for all baseline and S3 keys.

If the `.env` file does not exist, bootstrap from the example and then replace
all placeholder values:

```bash
python scripts/validate_phase7c_env.py \
  --env-file coordinator/.env \
  --strict-placeholders \
  --bootstrap-from-example \
  --report docs/reports/phase7c-production-env-validation.md
```

### 1.3 Release Gate

Run the ```bash
LATEST_MERGED_PR_NUMBER=412  # replace with the newest merged PR on main

PYTHONIOENCODING=utf-8 python scripts/run_phase8_release_gate.py \
  --repo mattmre/EDCOCR-PUBLIC \
  --pull-request "$LATEST_MERGED_PR_NUMBER" \
  --expected-branch main \
  --env-file coordinator/.env \
  --report docs/reports/phase8-release-gate.md \
  --summary-json docs/reports/phase8-release-gate.json
```

**Expected result**: `READY` with exit code 0.

Review the generated report at `docs/reports/phase8-release-gate.md` for any
blockers. Common blockers:

| Blocker | Resolution |
|---------|-----------|
| `S3_ACCESS_KEY` uses known insecure default `minioadmin` | Replace with production credentials (Section 2) |
| PR not merged | Merge the target PR before proceeding |
| Worktree has pending changes | Commit or stash changes, then re-run |
| Operator-chain env validation failed | Fix `.env` values, then re-run |

### 1.4 Docker Images

- [ ] Production Docker images built and tagged with version `1.2.0`:

```bash
# Coordinator image
cd coordinator
docker build -f Dockerfile.coordinator -t ocr-coordinator:1.2.0 .

# Worker image
docker build -f Dockerfile.worker -t ocr-worker:1.2.0 ..
```

- [ ] Images pushed to the container registry (or bundled for air-gapped deployment):

```bash
# Air-gapped bundle (if no registry access)
bash scripts/airgap-bundle.sh
```

**Font packages for searchable-PDF text layer embedding:**

The Docker images install `fonts-noto-core` via apt for Latin/Cyrillic/Greek coverage. CJK
fonts (NotoSansSC, NotoSansJP, NotoSansKR, NotoSansTC) are downloaded from the Google
Noto GitHub releases during the Docker build.

- **Internet-connected build**: Fonts are fetched automatically by the Dockerfile.
- **Air-gapped deployment**: Pre-stage the CJK `.ttf` files and add a `COPY` instruction
  to place them in the font directory (default: `/usr/share/fonts/noto-cjk/` or the path
  specified by `NOTO_FONT_DIR`).
- **Custom font directory**: Set the `NOTO_FONT_DIR` environment variable to override the
  default font search path used by the `font_selector` module.

### 1.5 Backup Procedures

- [ ] PostgreSQL backup verified and recent (< 24 hours):

```bash
# Docker Compose
docker compose -f docker-compose.coordinator.yml exec postgres \
  pg_dump -U ocr -d ocr_coordinator --format=custom --compress=6 \
  -f /var/lib/postgresql/data/backup-$(date +%Y%m%d-%H%M%S).sql.gz

# Kubernetes
kubectl create job --from=cronjob/${RELEASE}-postgres-backup \
  ${RELEASE}-manual-backup-$(date +%s)
```

- [ ] NFS data snapshot taken (if migrating to S3).
- [ ] Rollback procedure tested in staging (see [Section 6](#6-rollback-procedures)).

### 1.6 Infrastructure Prerequisites

- [ ] Target S3 bucket created and accessible.
- [ ] Network connectivity verified between coordinator, workers, and storage backend.
- [ ] DNS records updated (if changing coordinator hostname).
- [ ] TLS certificates deployed (if terminating TLS at the coordinator or ingress).

---

## 2. S3 Credential Replacement

The development `.env` currently uses `minioadmin` as S3 credentials. These must
be replaced with production values before any production workload.

### 2.1 Files Containing S3 Credentials

| File | Variables | Purpose |
|------|-----------|---------|
| `coordinator/.env` | `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_ENDPOINT`, `S3_BUCKET`, `S3_REGION` | Runtime coordinator and worker configuration |
| `coordinator/.env.example` | Same variables | Template for new deployments (update with placeholder instructions) |
| `coordinator/docker-compose.coordinator.yml` | Inherits from `.env` via `*coordinator_env` anchor | Coordinator stack services |
| `coordinator/docker-compose.worker.yml` | Inherits from `.env` via environment block | Worker nodes |

**Note**: If using Kubernetes, S3 credentials are supplied via Helm `values-secret.yaml`
or `--set` flags, not `.env` files. See `helm/ocr-local/values.yaml` for the
`secrets.s3AccessKey` and `secrets.s3SecretKey` fields.

### 2.2 Step-by-Step Replacement

**Step 1: Generate production S3 credentials** from your cloud provider or MinIO
admin console.

**Step 2: Update `coordinator/.env`**:

```bash
# Back up the current .env
cp coordinator/.env coordinator/.env.bak

# Edit the S3 section
# Replace these values:
#   S3_ACCESS_KEY=minioadmin      -> S3_ACCESS_KEY=<production-access-key>
#   S3_SECRET_KEY=minioadmin      -> S3_SECRET_KEY=<production-secret-key>
#   S3_ENDPOINT=http://minio:9000 -> S3_ENDPOINT=<production-s3-endpoint>
#   S3_BUCKET=ocr-local           -> S3_BUCKET=<production-bucket-name>
#   S3_REGION=us-east-1           -> S3_REGION=<production-region>
```

Also update `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` if running a self-hosted
MinIO instance.

**Step 3: Validate the updated credentials**:

```bash
python scripts/validate_phase7c_env.py \
  --env-file coordinator/.env \
  --strict-placeholders \
  --report docs/reports/phase7c-s3-credential-validation.md
```

All S3 keys must show status `ok`. The script will flag `minioadmin` as
`insecure_default` if not replaced.

**Step 4: Test S3 connectivity**:

```bash
# Start only the infrastructure services
cd coordinator
docker compose -f docker-compose.coordinator.yml up -d postgres rabbitmq redis django

# Test S3 connectivity from the coordinator container
docker compose -f docker-compose.coordinator.yml exec django python -c "
from jobs.storage import S3Backend
import os
s3 = S3Backend(
    endpoint=os.environ.get('S3_ENDPOINT', ''),
    bucket=os.environ.get('S3_BUCKET', ''),
    access_key=os.environ.get('S3_ACCESS_KEY', ''),
    secret_key=os.environ.get('S3_SECRET_KEY', ''),
    region=os.environ.get('S3_REGION', ''))
# List objects to verify connectivity
objects = s3.list_objects('test-connectivity/')
print(f'S3 connection successful. Objects found: {len(objects)}')
"
```

### 2.3 Rollback If S3 Connection Fails

If S3 connectivity fails after credential replacement:

1. Restore the backed-up `.env`:
   ```bash
   cp coordinator/.env.bak coordinator/.env
   ```
2. Restart affected services:
   ```bash
   docker compose -f docker-compose.coordinator.yml restart django celery-coordinator
   ```
3. Set `STORAGE_BACKEND=nfs` in `.env` to fall back to NFS while diagnosing S3 issues.
4. Investigate: check endpoint URL, credentials, bucket permissions, network
   connectivity, and any firewall rules between the coordinator and S3.

---

## 3. NFS-to-S3 Migration

Use this section when migrating existing job artifacts from NFS shared storage to
S3. The migration script (`scripts/migrate_nfs_to_s3.py`) uploads all artifacts,
verifies SHA-256 checksums, and optionally removes NFS files after verification.

### 3.1 Prerequisites

- [ ] S3 bucket created and credentials configured (Section 2 complete).
- [ ] S3 connectivity verified from the coordinator host.
- [ ] NFS mount accessible at the configured `NFS_ROOT` path (default: `/shared`).
- [ ] Sufficient S3 storage capacity for all existing artifacts.
- [ ] No active jobs in `processing` or `ingesting` state (drain the pipeline first).

**Drain the pipeline before migration:**

```bash
# Check for active jobs
docker compose -f docker-compose.coordinator.yml exec django \
  python manage.py shell -c "
from jobs.models import Job
active = Job.objects.filter(status__in=['processing', 'ingesting', 'assembling'])
print(f'Active jobs: {active.count}')
for j in active:
    print(f'  {j.job_id}: {j.status}')
"

# Wait for all active jobs to complete before proceeding
```

### 3.2 Dry Run

Always perform a dry run first. The dry run counts files and bytes without
uploading anything:

```bash
# Set credentials via environment (preferred over CLI args)
export S3_ACCESS_KEY=<production-access-key>
export S3_SECRET_KEY=<production-secret-key>

python scripts/migrate_nfs_to_s3.py \
  --nfs-root /shared \
  --s3-endpoint <production-s3-endpoint> \
  --s3-bucket <production-bucket-name> \
  --s3-region <production-region> \
  --output docs/reports/nfs-to-s3-dryrun.json \
  --verbose
```

**Review the dry run output:**

```
============================================================
Migration Summary (DRY RUN)
============================================================
  Jobs total:        <N>
  Jobs migrated:     <N>
  Jobs with errors:  0
  Files uploaded:    <N>
  Files skipped:     0
  Files verified:    <N>
  Bytes transferred: <N>
  Elapsed:           <N>s
============================================================
```

If `Jobs with errors` is non-zero, investigate the errors in the verbose log
before proceeding.

### 3.3 Execute the Migration

```bash
export S3_ACCESS_KEY=<production-access-key>
export S3_SECRET_KEY=<production-secret-key>

python scripts/migrate_nfs_to_s3.py \
  --nfs-root /shared \
  --s3-endpoint <production-s3-endpoint> \
  --s3-bucket <production-bucket-name> \
  --s3-region <production-region> \
  --execute \
  --output docs/reports/nfs-to-s3-migration.json \
  --verbose
```

The script uploads each file, then downloads a copy and compares SHA-256 checksums
to verify integrity. Any checksum mismatch is reported as an error.

### 3.4 Resume on Interruption

If the migration is interrupted (network failure, process kill), resume from where
it left off. The `--resume` flag skips files already present in S3:

```bash
python scripts/migrate_nfs_to_s3.py \
  --nfs-root /shared \
  --s3-endpoint <production-s3-endpoint> \
  --s3-bucket <production-bucket-name> \
  --execute \
  --resume \
  --output docs/reports/nfs-to-s3-migration-resumed.json \
  --verbose
```

### 3.5 Verify and Clean Up NFS

After successful migration with zero errors, optionally remove NFS files:

```bash
python scripts/migrate_nfs_to_s3.py \
  --nfs-root /shared \
  --s3-endpoint <production-s3-endpoint> \
  --s3-bucket <production-bucket-name> \
  --execute \
  --delete-nfs \
  --output docs/reports/nfs-to-s3-cleanup.json \
  --verbose
```

**Important**: The `--delete-nfs` flag only deletes a job's NFS directory when all
files for that job have been verified. If any file has a checksum mismatch, the
entire job directory is retained on NFS.

### 3.6 Rollback: Keep NFS as Fallback

Until you are confident that S3 is stable:

1. Keep NFS mounted and data intact (do not use `--delete-nfs` initially).
2. Set `STORAGE_BACKEND=s3` in `coordinator/.env` for new jobs.
3. Existing NFS-backed jobs remain accessible via the dual backend in `jobs/storage.py`.
4. If S3 fails, revert to `STORAGE_BACKEND=nfs` and restart the coordinator.

---

## 4. Staged Canary Deployment

Deploy to production in four stages with monitoring checkpoints between each stage.
Do not advance to the next stage until the current stage passes its success criteria.

### 4.1 Stage 1: Coordinator Stack Health (24 hours)

**Objective**: Verify that the coordinator stack (Django, Celery Beat, PostgreSQL,
RabbitMQ, Redis) is stable with production credentials and S3 storage.

**Procedure:**

```bash
cd coordinator

# Set production environment
# In coordinator/.env:
#   DEPLOYMENT_ENV=staging       (not production yet)
#   PRODUCTION_READINESS_ACK=false
#   STORAGE_BACKEND=s3
#   S3_ACCESS_KEY=<production>
#   S3_SECRET_KEY=<production>

# Start the coordinator stack
docker compose -f docker-compose.coordinator.yml up -d

# Verify health
docker compose -f docker-compose.coordinator.yml exec postgres pg_isready -U ocr -d ocr_coordinator
docker compose -f docker-compose.coordinator.yml exec rabbitmq rabbitmqctl status
docker compose -f docker-compose.coordinator.yml exec redis redis-cli -a "$REDIS_PASSWORD" ping

# Check Django health endpoint
curl -s http://localhost:8000/api/v1/health/ | python -m json.tool
```

**Capture baseline metrics:**

```bash
python scripts/capture_phase7c_metrics.py \
  --api-url http://localhost:8000/api/v1/metrics/ \
  --api-key "$METRICS_API_KEY" \
  --env-file coordinator/.env \
  --report docs/reports/phase7c-baseline-metrics.md \
  --json-output docs/reports/phase7c-baseline-metrics.json
```

**Success criteria (24 hours):**

| Metric | Threshold | How to Check |
|--------|-----------|--------------|
| Coordinator uptime | 24h with no crashes | `docker compose ps` shows all services `Up` |
| Health endpoint | Returns 200 with `status: ok` | `curl http://localhost:8000/api/v1/health/` |
| PostgreSQL connectivity | `pg_isready` returns 0 | Healthcheck passing in `docker compose ps` |
| RabbitMQ connectivity | `rabbitmqctl status` succeeds | Healthcheck passing |
| Redis connectivity | `PONG` response | Healthcheck passing |
| S3 connectivity | No S3 errors in logs | `docker compose logs django \| grep -i s3` |

**Rollback**: If any check fails, stop the stack and investigate logs. See
[Section 6.1](-coordinator-stack-rollback).

### 4.2 Stage 2: Single Worker with Production Traffic (24-48 hours)

**Objective**: Verify end-to-end OCR processing with one GPU worker processing
real production documents.

**Procedure:**

```bash
# Start a single GPU worker
docker compose -f docker-compose.worker.yml up -d

# Verify worker registration
docker compose -f docker-compose.coordinator.yml exec django \
  python manage.py fleet_status

# Submit a test document
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "X-API-Key: $OCR_API_KEY" \
  -F "file=@test-document.pdf"

# Monitor job completion
curl -s -H "X-API-Key: $OCR_API_KEY" \
  http://localhost:8000/api/v1/jobs/<job-id>/ | python -m json.tool
```

**Capture canary metrics after processing representative documents:**

```bash
python scripts/capture_phase7c_metrics.py \
  --api-url http://localhost:8000/api/v1/metrics/ \
  --api-key "$METRICS_API_KEY" \
  --env-file coordinator/.env \
  --report docs/reports/phase7c-stage2-canary-metrics.md \
  --json-output docs/reports/phase7c-stage2-canary-metrics.json
```

**Success criteria (24-48 hours):**

| Metric | Threshold | Source |
|--------|-----------|--------|
| Job error rate (1h) | < 5% | `ocr_job_error_rate_1h` or metrics endpoint |
| Page processing time (avg) | < 5000ms | `ocr_page_processing_time_avg_ms` |
| GPU workers available | >= 1 | `ocr_gpu_workers_available` |
| Jobs stuck in processing | 0 for > 30 min | `ocr_jobs_total{status="processing"}` with no progress |
| S3 job error rate | < 15% | `ocr_s3_job_error_rate_1h` |
| Job completion rate | > 85% | `ocr_job_completion_rate_1h` |
| Chain of custody violations | 0 | `ocr_custody_violations_total` |

**Rollback**: Drain the worker, resubmit failed jobs. See [Section 6.2](-worker-rollback).

### 4.3 Stage 3: Fleet Expansion (3-5 days)

**Objective**: Scale to the target fleet size and verify throughput under sustained
load.

**Procedure:**

```bash
# Scale workers (adjust count based on GPU capacity)
docker compose -f docker-compose.worker.yml up -d --scale ocr-worker=4

# For multi-GPU per-queue affinity:
# Note: docker-compose.multi-gpu.yml is auto-generated by the script below.
# Do not edit it by hand -- re-run the generator when GPU count changes.
python scripts/generate_multi_gpu_compose.py --gpu-count 4 --per-gpu-queues
docker compose -f docker-compose.multi-gpu.yml up -d

# Monitor fleet
docker compose -f docker-compose.coordinator.yml exec django \
  python manage.py fleet_status
```

**Success criteria (3-5 days):**

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Sustained throughput | Matches capacity plan | Monitor pages/hour via Grafana or metrics endpoint |
| Error rate (24h rolling) | < 5% | No sustained increase in errors |
| Worker heartbeat age | < 120s for all workers | `fleet_status` command |
| Queue depth (ocr_gpu) | < 50 sustained | No unbounded growth |
| Memory/VRAM usage | Stable (no leaks) | Monitor over multi-day window |
| S3 latency | < 500ms p99 for uploads | Monitor S3 operation timings |

**Rollback**: Scale back to 1 worker and investigate. See [Section 6.2](-worker-rollback).

### 4.4 Stage 4: Full Production (2 weeks)

**Objective**: Confirm long-term stability with full production traffic.

**Procedure:**

```bash
# Promote to production
# In coordinator/.env:
#   DEPLOYMENT_ENV=production
#   PRODUCTION_READINESS_ACK=true

LATEST_MERGED_PR_NUMBER=412  # replace with the newest merged PR on main

# Restart coordinator to pick up environment change
docker compose -f docker-compose.coordinator.yml up -d

# Run the release gate one final time
PYTHONIOENCODING=utf-8 python scripts/run_phase8_release_gate.py \
  --repo mattmre/EDCOCR-PUBLIC \
  --pull-request "$LATEST_MERGED_PR_NUMBER" \
  --expected-branch main \
  --env-file coordinator/.env \
  --report docs/reports/phase8-production-release-gate.md \
  --summary-json docs/reports/phase8-production-release-gate.json
```

**Success criteria (2 weeks):**

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Overall availability | > 99.5% | Coordinator health endpoint uptime |
| Error rate (7d rolling) | < 3% | Sustained low error rate |
| No data loss incidents | 0 | No custody chain violations |
| Cleanup commands running | On schedule | `cleanup_old_jobs` and `purge_temp_files` via Celery Beat |
| Backup CronJob | Completing successfully | PostgreSQL backup retention verified |

After 2 weeks of stable operation, the cutover is complete. Archive all evidence
reports and proceed to [Section 7](#7-post-cutover-validation).

---

## 5. Production Monitoring

### 5.1 Prometheus Metrics

The coordinator exposes metrics via two endpoints:

| Endpoint | Format | Auth |
|----------|--------|------|
| `GET /api/v1/metrics/` | JSON | `X-Api-Key` or `Authorization: Bearer` with `METRICS_API_KEY` |
| `GET /api/v1/prometheus/` | Prometheus text | Same auth; unauthenticated when `METRICS_API_KEY` is unset |

```bash
# Quick metrics check
curl -s -H "X-Api-Key: $METRICS_API_KEY" \
  http://<coordinator>:8000/api/v1/metrics/ | python -m json.tool

# Prometheus scrape endpoint
curl -s -H "X-Api-Key: $METRICS_API_KEY" \
  http://<coordinator>:8000/api/v1/prometheus/
```

### 5.2 Prometheus Alert Rules

The following alert rules are defined in `helm/ocr-local/templates/prometheusrule.yaml`
(enabled with `prometheus.rules.enabled=true` in Helm values):

| Alert | Expression | Severity | For | Action |
|-------|-----------|----------|-----|--------|
| `OCRHighJobErrorRate` | `ocr_job_error_rate_1h > 0.1` | warning | 5m | Investigate `failures.csv` and logs. Check for bad input documents or worker issues. |
| `OCRNoGPUWorkersAvailable` | `ocr_gpu_workers_available == 0` | critical | 2m | Check worker pods/containers. Restart or scale workers immediately. |
| `OCRWorkersDrained` | `ocr_workers_total{online} + ocr_workers_total{busy} == 0` | critical | 5m | All workers offline. Follow [Worker Recovery](../FAILOVER-RUNBOOK.md#5-worker-recovery). |
| `OCRSlowPageProcessing` | `ocr_page_processing_time_avg_ms > 30000` | warning | 10m | Check GPU utilization and queue depth. May need more workers. |
| `OCRJobsStuck` | Processing jobs with no page progress in 30m | warning | 30m | Check worker logs. Restart stuck workers. Resubmit affected jobs. |
| `OCRCustodyChainViolation` | `ocr_custody_violations_total > 0` | critical | 1m | Forensic investigation required. Do not delete any data. |
| `OCRHighS3ErrorRate` | `ocr_s3_job_error_rate_1h > 0.15` | warning | 5m | Check S3 connectivity and credentials. Consider failback to NFS. |
| `OCRLowCompletionRate` | `ocr_job_completion_rate_1h < 0.85` | warning | 10m | Investigate failed jobs. May indicate systematic input quality issues. |
| `OCRJobTimeout` | `ocr_jobs_stuck_total > 0` | warning | 5m | Jobs stuck over 1 hour. Check `JOB_PROCESSING_TIMEOUT_MINUTES` setting. |

**Timeout configuration**: The `JOB_PROCESSING_TIMEOUT_MINUTES` environment variable
(default: 30) controls how long a job can remain in `processing` state before the
coordinator's stale-job cleanup marks it as failed. For long-running forensic jobs
(large documents, high-DPI escalation), consider increasing this value. Per-job
overrides are supported via `processing_timeout_minutes` in the job submission payload
or batch `settings_json`.

### 5.3 Key Metrics to Watch

| Metric | Warning | Critical | How to Check |
|--------|---------|----------|--------------|
| `ocr_gpu_workers_available` | < 2 | 0 | Metrics endpoint or Prometheus |
| `ocr_job_error_rate_1h` | > 0.05 | > 0.10 | Metrics endpoint |
| `ocr_page_processing_time_avg_ms` | > 15000 | > 30000 | Metrics endpoint |
| `ocr_s3_job_error_rate_1h` | > 0.10 | > 0.15 | Metrics endpoint |
| `ocr_job_completion_rate_1h` | < 0.90 | < 0.85 | Metrics endpoint |
| `ocr_custody_violations_total` | -- | > 0 | Metrics endpoint |
| RabbitMQ queue depth (ocr_gpu) | > 50 | > 200 | `rabbitmqctl list_queues` |
| RabbitMQ queue depth (coordinator) | > 100 | > 500 | `rabbitmqctl list_queues` |
| PostgreSQL connection count | > 150 | > 190 (of max 200) | `pg_stat_activity` |
| Redis memory usage | > 70% of maxmemory | > 90% | `redis-cli info memory` |
| Worker heartbeat age | > 120s | > 300s | `fleet_status` management command |

### 5.4 Grafana Dashboard

When `prometheus.grafana.enabled=true` in the Helm chart, a ConfigMap with a
7-panel Grafana dashboard is deployed automatically. The panels cover:

1. **Pipeline Throughput**: Pages processed per minute (`rate(ocr_pages_processed_total[5m])`)
2. **Job Status Distribution**: Stacked bar of jobs by status (completed, failed, processing)
3. **Worker Fleet**: Gauge of workers by status (online, busy, draining, offline)
4. **GPU Queue Depth**: RabbitMQ `ocr_gpu` queue message count
5. **Page Processing Latency**: Average `ocr_page_processing_time_avg_ms`
6. **Error Rate**: Job failure percentage over time
7. **Infrastructure Health**: PostgreSQL connections, Redis memory, RabbitMQ memory

For additional canary-specific panels, the canary monitoring dashboard
adds 10 panels and 4 alert rules covering S3-backed job metrics.

### 5.5 Manual Monitoring (No Prometheus)

When Prometheus is not available, use these spot check commands:

```bash
# Fleet status
docker compose -f docker-compose.coordinator.yml exec django \
  python manage.py fleet_status

# RabbitMQ queue depths
docker compose -f docker-compose.coordinator.yml exec rabbitmq \
  rabbitmqctl list_queues name messages consumers

# Redis memory
docker compose -f docker-compose.coordinator.yml exec redis \
  redis-cli -a "$REDIS_PASSWORD" info memory | grep used_memory_human

# PostgreSQL active connections
docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d ocr_coordinator -c "SELECT count(*) FROM pg_stat_activity WHERE state='active';"

# Celery worker status (via Flower API)
curl -u admin:$FLOWER_PASSWORD http://localhost:5555/api/workers

# Stuck jobs
docker compose -f docker-compose.coordinator.yml exec django \
  python manage.py shell -c "
from jobs.models import Job
stuck = Job.objects.filter(status__in=['processing', 'assembling', 'ingesting'])
print(f'Stuck jobs: {stuck.count}')
for j in stuck:
    print(f'  {j.job_id}: {j.status} ({j.pages_completed}/{j.total_pages})')
"
```

### 5.6 On-Call Procedures

1. **Alert fires**: Check the alert description in Prometheus/Grafana or the metrics endpoint.
2. **Identify scope**: Single worker, single service, or cluster-wide?
3. **Consult runbooks**: Use this document for cutover issues, the
   [Failover Runbook](../FAILOVER-RUNBOOK.md) for component failures.
4. **Escalate**: If custody chain violations are detected, escalate to forensic
   investigation immediately. Do not delete any data.
5. **Document**: Record the incident timeline, actions taken, and outcome.

---

## 6. Rollback Procedures

### 6.1 Coordinator Stack Rollback

**When to use**: Coordinator services fail to start or become unstable with
production credentials.

```bash
cd coordinator

# Stop all coordinator services
docker compose -f docker-compose.coordinator.yml down

# Restore the previous .env
cp coordinator/.env.bak coordinator/.env

# Restart with previous configuration
docker compose -f docker-compose.coordinator.yml up -d

# Verify health
docker compose -f docker-compose.coordinator.yml exec postgres pg_isready -U ocr -d ocr_coordinator
curl -s http://localhost:8000/api/v1/health/ | python -m json.tool
```

### 6.2 Worker Rollback

**When to use**: Workers fail to process jobs or exhibit high error rates.

**Step 1: Drain active workers**

```bash
# Option A: Via Celery remote control
docker compose -f docker-compose.coordinator.yml exec celery-coordinator \
  celery -A coordinator control shutdown --destination worker@<hostname>

# Option B: Scale to zero
docker compose -f docker-compose.worker.yml down
```

**Step 2: Resubmit failed jobs**

```bash
docker compose -f docker-compose.coordinator.yml exec django \
  python manage.py shell -c "
from jobs.models import Job
failed = Job.objects.filter(status='failed')
print(f'Failed jobs to resubmit: {failed.count}')
# Reset to pending for reprocessing
for j in failed:
    j.status = 'pending'
    j.save
    print(f'  Reset: {j.job_id}')
"
```

**Step 3: Deploy previous worker image**

```bash
# If using tagged images, roll back to the previous tag
# Edit docker-compose.worker.yml to use the previous image tag
docker compose -f docker-compose.worker.yml up -d
```

### 6.3 Docker Image Rollback

**When to use**: A new image version introduces bugs or incompatibilities.

```bash
# List available image tags
docker images | grep ocr-coordinator
docker images | grep ocr-worker

# Update compose files to use the previous tag
# Example: change 1.0.0 to 0.9.0 in the image field

# Restart with the old image
cd coordinator
docker compose -f docker-compose.coordinator.yml up -d
docker compose -f docker-compose.worker.yml up -d
```

For Kubernetes:

```bash
RELEASE=ocr-local

# Roll back the Helm release to the previous revision
helm rollback $RELEASE <PREVIOUS_REVISION>

# Or specify the previous image tag explicitly
helm upgrade $RELEASE helm/ocr-local/ \
  --set image.tag=<previous-tag> \
  -f values-production.yaml \
  -f values-secret.yaml
```

### 6.4 Database Rollback

**When to use**: Database migration causes data corruption or schema incompatibility.

Follow the restore procedure in the
[Failover Runbook, Section 8.3](../FAILOVER-RUNBOOK.md-restore-procedure).

Summary:

```bash
# Docker Compose
docker compose -f docker-compose.coordinator.yml stop django celery-coordinator celery-beat flower

docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d postgres -c "
    SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'ocr_coordinator';
  "
docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d postgres -c "DROP DATABASE IF EXISTS ocr_coordinator;"
docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d postgres -c "CREATE DATABASE ocr_coordinator OWNER ocr;"

docker cp ./backups/<backup-file>.sql.gz \
  $(docker compose -f docker-compose.coordinator.yml ps -q postgres):/tmp/restore.sql.gz

docker compose -f docker-compose.coordinator.yml exec postgres \
  pg_restore -U ocr -d ocr_coordinator --no-owner --verbose /tmp/restore.sql.gz

docker compose -f docker-compose.coordinator.yml up -d
```

### 6.5 S3 to NFS Failback

**When to use**: S3 becomes unreliable or inaccessible in production.

**Step 1: Switch storage backend**

```bash
# In coordinator/.env:
#   STORAGE_BACKEND=nfs

# Restart coordinator services
docker compose -f docker-compose.coordinator.yml restart django celery-coordinator
```

**Step 2: Verify NFS mount**

```bash
docker compose -f docker-compose.coordinator.yml exec django \
  ls -la /shared/jobs/ | head -20
```

**Step 3: New jobs will use NFS storage**. Existing S3-backed jobs remain
accessible as long as S3 credentials are still configured (the dual backend in
`jobs/storage.py` supports both).

### 6.6 Worker Fleet Drain

**When to use**: Before any infrastructure maintenance or rollback that affects
all workers.

```bash
# Option A: Graceful drain via Celery
docker compose -f docker-compose.coordinator.yml exec celery-coordinator \
  celery -A coordinator control shutdown

# Option B: Django admin
# Navigate to http://<coordinator>:8000/admin/jobs/worker/
# Select all workers > "Drain worker" action

# Option C: Kubernetes
kubectl scale deployment ${RELEASE}-gpu-worker --replicas=0
kubectl scale deployment ${RELEASE}-cpu-worker --replicas=0

# Wait for in-flight tasks to complete (check queue)
docker compose -f docker-compose.coordinator.yml exec rabbitmq \
  rabbitmqctl list_queues name messages consumers
# Wait until messages = 0 for ocr_gpu and cpu_general queues
```

---

## 7. Post-Cutover Validation

After completing Stage 4 of the canary deployment, run this final validation
checklist.

### 7.1 Release Gate (Final)

```bash
LATEST_MERGED_PR_NUMBER=412  # replace with the newest merged PR on main

PYTHONIOENCODING=utf-8 python scripts/run_phase8_release_gate.py \
  --repo mattmre/EDCOCR-PUBLIC \
  --pull-request "$LATEST_MERGED_PR_NUMBER" \
  --expected-branch main \
  --env-file coordinator/.env \
  --report docs/reports/phase8-production-final-gate.md \
  --summary-json docs/reports/phase8-production-final-gate.json
```

Result must be `READY`.

### 7.2 Health Endpoints

```bash
# Coordinator health
curl -s http://<coordinator>:8000/api/v1/health/ | python -m json.tool

# Metrics endpoint (verify non-empty response)
curl -s -H "X-Api-Key: $METRICS_API_KEY" \
  http://<coordinator>:8000/api/v1/metrics/ | python -m json.tool

# Flower dashboard (verify accessible)
curl -s -u admin:$FLOWER_PASSWORD http://<coordinator>:5555/api/workers | python -m json.tool
```

### 7.3 Monitoring and Alerting

- [ ] Prometheus is scraping the `/api/v1/prometheus/` endpoint (check Prometheus targets page).
- [ ] All alert rules are loaded (check Prometheus alerts page).
- [ ] Grafana dashboard is accessible and showing data.
- [ ] At least one test alert has been fired and received by the on-call system.

### 7.4 Operational Verification

- [ ] Worker fleet status shows expected number of workers:
  ```bash
  docker compose -f docker-compose.coordinator.yml exec django \
    python manage.py fleet_status
  ```
- [ ] Cleanup commands are running on schedule:
  ```bash
  docker compose -f docker-compose.coordinator.yml exec django \
    python manage.py shell -c "
  from django_celery_beat.models import PeriodicTask
  for t in PeriodicTask.objects.filter(enabled=True):
      print(f'{t.name}: {t.interval or t.crontab} (last_run: {t.last_run_at})')
  "
  ```
- [ ] PostgreSQL backup CronJob is completing (Kubernetes) or manual backup is scheduled.
- [ ] S3 bucket versioning is enabled (for production S3 backends).

### 7.5 Document Cutover Completion

Archive the following evidence:

- [ ] `docs/reports/phase7c-production-env-validation.md`
- [ ] `docs/reports/phase7c-baseline-metrics.md` and `.json`
- [ ] `docs/reports/phase7c-stage2-canary-metrics.md` and `.json`
- [ ] `docs/reports/phase8-production-final-gate.md` and `.json`
- [ ] `docs/reports/nfs-to-s3-migration.json` (if applicable)
- [ ] Incident log for any issues during cutover stages
- [ ] Date and time of `DEPLOYMENT_ENV=production` promotion
- [ ] Names of personnel who approved the cutover

---

## Appendix: Quick Reference Commands

```bash
# === PRE-CUTOVER ===
# Validate environment
python scripts/validate_phase7c_env.py --env-file coordinator/.env --strict-placeholders

# Run release gate
PYTHONIOENCODING=utf-8 python scripts/run_phase8_release_gate.py \
  --repo mattmre/EDCOCR-PUBLIC --pull-request <PR> --expected-branch main

# Capture metrics
python scripts/capture_phase7c_metrics.py \
  --api-url http://localhost:8000/api/v1/metrics/ --api-key "$METRICS_API_KEY" \
  --env-file coordinator/.env --report docs/reports/metrics.md --json-output docs/reports/metrics.json

# === MIGRATION ===
# NFS-to-S3 dry run
python scripts/migrate_nfs_to_s3.py --nfs-root /shared --s3-endpoint <URL> --s3-bucket <BUCKET>

# NFS-to-S3 execute
python scripts/migrate_nfs_to_s3.py --nfs-root /shared --s3-endpoint <URL> --s3-bucket <BUCKET> --execute

# NFS-to-S3 resume
python scripts/migrate_nfs_to_s3.py --nfs-root /shared --s3-endpoint <URL> --s3-bucket <BUCKET> --execute --resume

# === DEPLOYMENT ===
# Start coordinator stack
cd coordinator && docker compose -f docker-compose.coordinator.yml up -d

# Start workers
docker compose -f docker-compose.worker.yml up -d

# Start HA overlay
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml up -d

# Fleet status
docker compose -f docker-compose.coordinator.yml exec django python manage.py fleet_status

# === MONITORING ===
# Health check
curl -s http://localhost:8000/api/v1/health/

# Metrics
curl -s -H "X-Api-Key: $METRICS_API_KEY" http://localhost:8000/api/v1/metrics/

# Queue depths
docker compose -f docker-compose.coordinator.yml exec rabbitmq \
  rabbitmqctl list_queues name messages consumers

# === ROLLBACK ===
# Restore .env
cp coordinator/.env.bak coordinator/.env

# Switch storage backend
# Set STORAGE_BACKEND=nfs in coordinator/.env

# Drain workers
docker compose -f docker-compose.coordinator.yml exec celery-coordinator \
  celery -A coordinator control shutdown

# Database restore
docker compose -f docker-compose.coordinator.yml exec postgres \
  pg_restore -U ocr -d ocr_coordinator --no-owner --verbose /tmp/restore.sql.gz
```
