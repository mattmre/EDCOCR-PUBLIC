# S3 Storage Backend Rollout Runbook

**Purpose**: Staged rollout procedure for enabling S3 storage backend in production, with explicit rollback gates and operator actions.

**Status**: Operator runbook for staged S3 backend rollout with explicit rollback gates.

---

## Overview

This runbook guides operators through a staged rollout of S3 storage backend support, progressing from canary testing through full production deployment with defined rollback triggers at each stage.

### Prerequisites

- Storage-aware task pipeline available (`coordinator/jobs/storage.py`)
- MinIO integration tests passing (`coordinator/jobs/tests/test_storage.py`)
- S3 observability runbook reviewed (see `s3-observability-runbook.md`)
- Staging environment available with MinIO or AWS S3 backend
- Rollback plan reviewed and understood by team

---

## Rollout Stages

### Stage 0: Pre-Flight Validation

**Duration**: 1 day
**Goal**: Verify baseline system health before rollout

#### Actions

1. **Baseline Health Check**
   ```bash
   # Verify existing NFS-mode system health
   cd coordinator
   python -m pytest jobs/tests/test_storage.py -v
   python -m pytest jobs/tests/test_tasks.py -v

   # Verify docs validation
   cd ..
   python scripts/check_docs.py --full-scan
   python scripts/verify_docs_compat.py
   ```

2. **System Metrics Snapshot**
   - Record current throughput metrics (pages/hour, jobs/hour)
   - Record current error rates (task failures, retries)
   - Record current latency percentiles (p50, p95, p99)
   - Document baseline disk usage patterns

3. **Configuration Validation**
   ```bash
   # Validate environment configuration
   python scripts/env_preflight.py --env-file coordinator/.env

   # Confirm storage backend selection (expect NFS at baseline)
   grep -E "^STORAGE_BACKEND=" coordinator/.env
   ```

4. **Capture Baseline Metrics**

   Capture the metrics that are meaningful to your environment using the
   Prometheus endpoint (`/api/v1/prometheus/`) or your existing monitoring
   stack. Preserve a baseline snapshot (throughput, latency p50/p95/p99,
   task error/retry rate) for comparison against later stages.

#### Rollback Gate

- **Trigger**: Baseline tests fail or system shows degraded performance
- **Action**: **Do not proceed**. Fix baseline issues before starting rollout.

---

### Stage 1: Canary (Single Test Job)

**Duration**: 2-4 hours
**Goal**: Validate S3 backend with single controlled job

#### Actions

1. **Enable S3 Mode for Test Worker**
   ```bash
   # Create test configuration
   cd coordinator
   cp .env .env.nfs-backup

   # Update .env for S3 mode
   echo "STORAGE_BACKEND=s3" >> .env
   echo "S3_BUCKET=ocr-test-canary" >> .env
   echo "S3_ENDPOINT=http://minio:9000" >> .env  # or AWS S3 endpoint
   echo "S3_ACCESS_KEY=${S3_ACCESS_KEY:?Set S3_ACCESS_KEY to production value}" >> .env
   echo "S3_SECRET_KEY=${S3_SECRET_KEY:?Set S3_SECRET_KEY to production value}" >> .env
   echo "S3_REGION=us-east-1" >> .env            # optional for AWS S3

   # Validate S3 configuration before starting
   cd ..
   python scripts/env_preflight.py --env-file coordinator/.env
   ```
   Do not reuse placeholder credentials (e.g. `minioadmin`) for production
   S3 endpoints. Record the redacted validation evidence in your change
   management record before proceeding.

2. **Start Single Worker**
   ```bash
   # Start coordinator with S3 mode
   celery -A coordinator worker --loglevel=info --concurrency=1 -n canary@%h
   ```

3. **Submit Test Job**
   - Use small test document (2-5 pages)
   - Monitor logs for S3 operations
   - Verify output artifacts in S3 bucket

4. **Capture Post-Canary Metrics**

   Snapshot the same metrics captured at baseline and diff them against the
   baseline file to verify there are no regressions.

5. **Validation Checklist**
   - [ ] Job completes successfully (status: `completed`)
   - [ ] All output files present in S3 bucket (pages/, metadata.json, final PDF)
   - [ ] No S3-related exceptions in logs
   - [ ] Task execution time within 2x baseline
   - [ ] Storage backend field logged correctly

#### Observability

Monitor these signals (see `s3-observability-runbook.md` for details):
- S3 API response times (target: <500ms p95)
- Task retry rates (target: <5%)
- Worker error logs (filter: "S3", "storage", "upload", "download")

#### Rollback Gate

**Triggers**:
- Job fails to complete after 3 attempts
- S3 exceptions in logs (authentication, network, bucket access)
- Task execution time >3x baseline
- Output artifacts missing or corrupted

**Rollback Actions**:
```bash
# Stop S3 worker
pkill -f "celery.*canary"

# Restore NFS configuration
cd coordinator
mv .env.nfs-backup .env

# Restart NFS worker
celery -A coordinator worker --loglevel=info --concurrency=4
```

**Post-Rollback**:
- Review canary job logs
- Fix identified issues
- Re-run Stage 0 baseline validation
- Restart Stage 1 after fixes confirmed

---

### Stage 2: Limited (10% Traffic)

**Duration**: 1-2 days
**Goal**: Validate S3 backend under real traffic patterns

#### Actions

1. **Deploy S3-Mode Workers (10% Capacity)**
   ```bash
   # Example: If production has 10 workers, deploy 1 S3 worker

   # Update worker configuration
   cd coordinator
   cp .env .env.s3
   # Ensure STORAGE_BACKEND=s3 in .env.s3

   # Start S3 worker pool (example using systemd or supervisord)
   celery -A coordinator worker --loglevel=info --concurrency=2 -n s3-worker-1@%h
   ```

2. **Route 10% of Jobs to S3 Workers**
   - Option A: Use queue-based routing (add S3-specific queue)
   - Option B: Use weighted routing at job submission
   - Option C: Use time-based routing (e.g., every 10th job)

3. **Monitor for 24-48 Hours**
   - Track job completion rates (S3 vs NFS)
   - Track error rates (S3 vs NFS)
   - Track latency distributions (S3 vs NFS)
   - Capture timestamped metrics snapshots every 6 hours and store them
     alongside your change-management record.

4. **Record Stage 2 Evidence**
   - Save the metrics snapshots collected during the 24-48h window.
   - Record whether S3 credential validation evidence was reviewed.
   - Document the GO / rollback decision for the next stage.

#### Validation Checklist

- [ ] S3 job completion rate >95%
- [ ] S3 job error rate <5%
- [ ] S3 job latency within 2x NFS baseline
- [ ] No S3 authentication failures
- [ ] No worker crashes or restarts
- [ ] S3 bucket storage growth as expected

#### Rollback Gate

**Triggers**:
- S3 job completion rate <90%
- S3 job error rate >10%
- S3 job latency >3x NFS baseline
- Repeated S3 authentication failures
- Worker instability (crashes, OOM)

**Rollback Actions**:
```bash
# Stop S3 workers
pkill -f "celery.*s3-worker"

# Requeue any pending S3 jobs to NFS workers
cd coordinator
celery -A coordinator purge  # if safe to drop pending tasks
# OR manually requeue tasks using Flower UI or celery inspect

# Verify NFS workers handling full load
celery -A coordinator inspect active
```

**Post-Rollback**:
- Analyze S3 worker logs for failure patterns
- Review S3 bucket access logs
- Fix identified issues
- Consider infrastructure changes (network, S3 endpoint, credentials)
- Restart Stage 1 validation after fixes

---

### Stage 3: Broad (50% Traffic)

**Duration**: 3-5 days
**Goal**: Validate S3 backend at scale with mixed workload

#### Actions

1. **Scale S3 Worker Pool to 50%**
   ```bash
   # Deploy additional S3 workers to match 50% of NFS capacity
   # Example: 5 S3 workers if 10 total workers

   for i in {2..5}; do
     celery -A coordinator worker --loglevel=info --concurrency=2 -n s3-worker-$i@%h &
   done
   ```

2. **Update Routing to 50% S3**
   - Adjust queue weights or routing logic
   - Verify distribution via monitoring dashboard

3. **Monitor for 3-5 Days**
   - Track daily job volume distribution
   - Compare S3 vs NFS cost metrics
   - Monitor S3 bucket storage costs and growth
   - Capture a daily metrics snapshot using a day-indexed file name
     (e.g. `stage3-day1-metrics.json` ... `stage3-day5-metrics.json`).

4. **Record Stage 3 Evidence**
   - Retain all day-indexed metrics artifacts collected during the
     3-5 day window.
   - Record the final GO / rollback decision and any escalation notes.

#### Validation Checklist

- [ ] S3 job completion rate >97%
- [ ] S3 job error rate <3%
- [ ] S3 job latency stable (no degradation over time)
- [ ] No S3 API throttling errors
- [ ] Worker CPU/memory usage stable
- [ ] S3 storage costs within budget

#### Rollback Gate

**Triggers**:
- S3 job completion rate <95%
- S3 job error rate >5%
- S3 API throttling observed
- S3 storage costs exceed projections by >20%
- Worker resource exhaustion

**Rollback Actions**:
```bash
# Reduce S3 workers back to 10%
pkill -f "celery.*s3-worker-[2-5]"

# Verify single S3 worker still operational
celery -A coordinator inspect active | grep s3-worker-1

# Verify NFS workers handling increased load
celery -A coordinator inspect active | grep -v s3-worker
```

**Post-Rollback**:
- Analyze cost metrics and projections
- Review API throttling patterns and consider rate limiting
- Assess worker resource allocation
- Tune S3 worker configuration (concurrency, prefetch, timeouts)
- Decide: continue to Stage 4 or iterate at Stage 2/3

---

### Stage 4: Full (100% Traffic)

**Duration**: 2 weeks initial, ongoing monitoring
**Goal**: Complete migration to S3 backend

#### Actions

1. **Scale S3 Workers to 100%**
   ```bash
   # Deploy full S3 worker pool
   for i in {6..10}; do
     celery -A coordinator worker --loglevel=info --concurrency=2 -n s3-worker-$i@%h &
   done
   ```

2. **Decommission NFS Workers**
   ```bash
   # Gracefully stop NFS workers (allow in-flight jobs to complete)
   celery -A coordinator inspect active | grep -v s3-worker  # verify no active NFS jobs

   pkill -TERM -f "celery.*worker" && sleep 30  # graceful shutdown
   # Then start only S3 workers
   ```

3. **Update Default Configuration**
   ```bash
   # Set S3 as default in .env
   cd coordinator
   echo "STORAGE_BACKEND=s3" > .env
   echo "S3_BUCKET=ocr-production" >> .env
    echo "S3_ENDPOINT=<production-s3-endpoint>" >> .env
    # ... (add non-default S3 credentials)
   ```
   - Use `S3_ENDPOINT` to match application settings and env validation.
   - Do not reuse `minioadmin` or other local-only credentials for Stage4 execution evidence.

4. **Execute NFS-to-S3 migration in two steps (safety-required)**
   ```bash
   # Step A (required first): migrate with verification, keep NFS artifacts
   python scripts/migrate_nfs_to_s3.py \
     --nfs-root /shared \
     --s3-endpoint http://minio:9000 \
     --s3-bucket ocr-production \
     --execute \
     --output docs/reports/stage4-migration-execute.json
   ```
   - Do **not** use `--delete-nfs` in the initial execute pass.
   - Keep NFS artifacts available during early full-cutover stabilization.
   - Open delete gate only after Stage 4 is stable for 2 weeks and explicit approval is recorded.
   ```bash
   # Step B (post-window delete gate only): cleanup NFS after stable window
   python scripts/migrate_nfs_to_s3.py \
     --nfs-root /shared \
     --s3-endpoint http://minio:9000 \
     --s3-bucket ocr-production \
     --execute \
     --delete-nfs \
     --output docs/reports/stage4-migration-delete.json
   ```

5. **Monitor for 2 Weeks**
   - Daily health checks
   - Weekly cost reviews
   - Continuous error rate monitoring

#### Validation Checklist

- [ ] All jobs running on S3 backend
- [ ] Job completion rate >98%
- [ ] Job error rate <2%
- [ ] S3 storage growth linear and predictable
- [ ] No NFS workers in pool
- [ ] Documentation updated with S3 as default

#### Rollback Gate

**Triggers**:
- System-wide outage or critical failure
- S3 job completion rate <95%
- Unrecoverable S3 data loss
- Runaway storage costs

**Rollback Actions** (CRITICAL PATH):
```bash
# EMERGENCY: Full rollback to NFS
# 1. Stop all S3 workers immediately
pkill -9 -f "celery.*s3-worker"

# 2. Restore NFS configuration
cd coordinator
cp .env.nfs-backup .env

# 3. Start full NFS worker pool
for i in {1..10}; do
  celery -A coordinator worker --loglevel=info --concurrency=4 -n worker-$i@%h &
done

# 4. Verify workers online and processing
celery -A coordinator inspect active
celery -A coordinator inspect stats

# 5. Notify team and begin incident post-mortem
```

**Post-Rollback** (Full):
- Conduct incident post-mortem
- Review S3 backend architecture and configuration
- Assess data recovery needs (S3 -> NFS migration)
- Determine if S3 rollout should continue or be abandoned
- Update runbook with lessons learned

---

## Rollback Decision Matrix

| Symptom | Stage | Severity | Action |
|---------|-------|----------|--------|
| Single job failure | 1-2 | Low | Retry, monitor |
| <90% completion rate | 2-3 | Medium | Rollback to previous stage |
| <95% completion rate | 4 | High | Rollback to Stage 3 |
| S3 auth failures | Any | High | Rollback immediately |
| Worker crashes | Any | High | Rollback immediately |
| API throttling | 3-4 | Medium | Reduce load or rollback |
| Storage cost spike | 3-4 | Medium | Pause, analyze, decide |
| Data loss | Any | Critical | **EMERGENCY ROLLBACK** |

---

## Validation Commands Reference

### Pre-Flight
```bash
python -m ruff check .
cd coordinator && python -m pytest jobs/tests/test_storage.py -v
cd coordinator && python -m pytest jobs/tests/test_tasks.py -v
python scripts/check_docs.py --full-scan
```

### Canary
```bash
# Check S3 worker logs
tail -f /var/log/celery/canary.log | grep -E "(S3|storage|ERROR)"

# Verify S3 bucket contents
aws s3 ls s3://ocr-test-canary/jobs/ --recursive

# Check task metrics
celery -A coordinator inspect stats | jq '.["canary@hostname"].total'
```

### Limited/Broad/Full
```bash
# Monitor worker distribution
celery -A coordinator inspect active | grep -c "s3-worker"
celery -A coordinator inspect active | grep -c -v "s3-worker"

# Check S3 storage usage
aws s3 ls s3://ocr-production/ --recursive --summarize

# Monitor error rates
celery -A coordinator events  # watch for failures
```

---

## Communication Plan

### Stage 1 (Canary)
- **Notify**: Engineering team only
- **Channel**: Slack #ocr-dev
- **Frequency**: Start and end of stage

### Stage 2 (Limited)
- **Notify**: Engineering + Ops teams
- **Channel**: Slack #ocr-prod
- **Frequency**: Daily updates

### Stage 3 (Broad)
- **Notify**: Engineering + Ops + Product teams
- **Channel**: Slack #ocr-prod + email
- **Frequency**: Daily updates, weekly summary

### Stage 4 (Full)
- **Notify**: All stakeholders
- **Channel**: All channels + status page
- **Frequency**: Daily for first week, weekly thereafter

### Rollback Communication
- **Immediate**: Slack alert to #ocr-prod and #incidents
- **1 hour**: Incident summary and next steps
- **24 hours**: Post-mortem timeline and action items

---

## Success Criteria

### Canary Success
- Single test job completes successfully
- S3 operations logged and traceable
- No worker errors

### Limited Success
- 24-48 hours stable operation
- S3 completion rate matches NFS baseline
- No rollback triggers hit

### Broad Success
- 3-5 days stable operation at scale
- S3 latency stable and predictable
- Cost metrics within projections

### Full Success
- 2 weeks stable operation at 100%
- NFS workers decommissioned
- S3 set as default in configuration
- Team trained on S3 operations and troubleshooting

---

## Related Documentation

- `s3-observability-runbook.md` - Health signals and metrics for S3 backend
- `coordinator/jobs/storage.py` - Storage backend implementation
- `scripts/migrate_nfs_to_s3.py` - NFS-to-S3 migration script

---

**Approved By**: _TBD_
**Next Review**: After Stage 2 completion or any rollback event
