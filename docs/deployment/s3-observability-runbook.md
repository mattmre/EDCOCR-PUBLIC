# S3 Storage Backend Observability Runbook

**Purpose**: Define required health signals, metrics, log indicators, and alert thresholds for S3 backend operations.

**Status**: Operator reference for observable signals and verification checklists during S3 backend rollout and production operation.

---

## Overview

This runbook provides operators with observable signals and verification checklists for assessing S3 storage backend health during rollout and production operation. All signals are based on existing log output, test results, and storage system behavior — no additional monitoring infrastructure required.

### Key Principles

1. **Log-based observability**: Leverage existing Celery/Python logging
2. **Test-driven health checks**: Use pytest as validation tool
3. **Storage behavior verification**: Monitor S3 bucket state directly
4. **Manual inspection first**: Automate alerts only after patterns validated

---

## Health Signals by Category

### 1. Task Execution Health

#### Signal: Job Completion Rate

**Source**: Celery worker logs + test suite results

**Collection Method**:
```bash
# During rollout - monitor worker logs
tail -f /var/log/celery/worker.log | grep "Task.*succeeded"

# Count successful vs failed tasks over time window
grep "Task.*succeeded" /var/log/celery/worker.log | wc -l
grep "Task.*failed" /var/log/celery/worker.log | wc -l
```

**Expected Baseline** (from test validation):
- NFS mode: 100% completion in test suite (930 passed, 25 skipped)
- S3 mode: Target 95-98% completion rate in production

**Alert Thresholds**:
- **Warning**: Completion rate <97% over 1 hour window
- **Critical**: Completion rate <95% over 30 min window
- **Emergency**: Completion rate <90% over 15 min window

**Verification Checklist**:
- [ ] Compare S3 completion rate to NFS baseline
- [ ] Check for patterns in failed jobs (specific pages, file types, sizes)
- [ ] Verify no systemic S3 backend errors in logs

---

#### Signal: Task Retry Rate

**Source**: Celery retry logs

**Collection Method**:
```bash
# Monitor retry attempts
tail -f /var/log/celery/worker.log | grep "Retrying.*ingest_document\|process_page\|assemble_document"

# Count retries over time
grep "Retrying" /var/log/celery/worker.log | grep -E "$(date +%Y-%m-%d)" | wc -l
```

**Expected Baseline**:
- Normal: <5% of tasks require retry (network transients, rate limiting)
- Retry with exponential backoff should eventually succeed

**Alert Thresholds**:
- **Warning**: Retry rate >5% over 1 hour
- **Critical**: Retry rate >10% over 30 min
- **Emergency**: Same task retrying >5 times (suggests persistent failure)

**Verification Checklist**:
- [ ] Identify which task types are retrying (ingest, process, assemble)
- [ ] Check if retries eventually succeed or exhaust attempts
- [ ] Correlate retry spikes with S3 API throttling or network issues

---

#### Signal: Task Execution Time

**Source**: Celery task duration logs + pytest benchmarks

**Collection Method**:
```bash
# Extract task durations from logs
grep "Task.*succeeded in" /var/log/celery/worker.log | awk '{print $NF}' | sort -n

# Compare S3 vs NFS task times
grep "STORAGE_BACKEND=s3.*succeeded in" /var/log/celery/worker.log | awk '{print $NF}' | sort -n
grep "STORAGE_BACKEND=nfs.*succeeded in" /var/log/celery/worker.log | awk '{print $NF}' | sort -n
```

**Expected Baseline** (from existing behavior):
- NFS mode: Minimal I/O overhead, fast local disk access
- S3 mode: +20-50% latency acceptable for download/upload operations
- Single-page process: ~2-5 seconds
- Multi-page assembly: ~5-15 seconds

**Alert Thresholds**:
- **Warning**: P95 latency >2x baseline
- **Critical**: P95 latency >3x baseline
- **Emergency**: P95 latency >5x baseline or tasks timing out

**Verification Checklist**:
- [ ] Compare P50, P95, P99 latencies between S3 and NFS modes
- [ ] Check for latency spikes correlating with S3 API throttling
- [ ] Verify task timeouts configured appropriately (default: 600s)

---

### 2. S3 Backend Health

#### Signal: S3 API Response Times

**Source**: Application logs with timing instrumentation

**Collection Method**:
```bash
# Monitor S3 API calls in worker logs
tail -f /var/log/celery/worker.log | grep -E "S3.*upload|S3.*download|S3.*list"

# Extract timing data (if instrumented in storage.py)
# Example: "S3 upload to bucket/key completed in 234ms"
grep "S3.*completed in" /var/log/celery/worker.log | awk '{print $NF}' | sed 's/ms//' | sort -n
```

**Expected Baseline**:
- S3 upload (small files <1MB): <500ms P95
- S3 download (small files <1MB): <300ms P95
- S3 list operations: <200ms P95
- Large files (>10MB): Scale linearly with size

**Alert Thresholds**:
- **Warning**: P95 response time >1s for small files
- **Critical**: P95 response time >2s or timeouts >1%
- **Emergency**: S3 endpoint unreachable or auth failures

**Verification Checklist**:
- [ ] Check S3 endpoint availability (`aws s3 ls` or `curl` healthcheck)
- [ ] Verify network path to S3 (latency, packet loss)
- [ ] Review AWS/MinIO service status pages

---

#### Signal: S3 Authentication Failures

**Source**: Exception logs

**Collection Method**:
```bash
# Monitor for S3 auth errors
tail -f /var/log/celery/worker.log | grep -E "NoCredentialsError|InvalidAccessKeyId|SignatureDoesNotMatch"

# Count auth failures over time
grep -E "NoCredentialsError|InvalidAccessKeyId|SignatureDoesNotMatch" /var/log/celery/worker.log | wc -l
```

**Expected Baseline**:
- Zero authentication failures in steady state
- Credentials should be valid and rotated per policy

**Alert Thresholds**:
- **Critical**: Any authentication failure (suggests credential issue)
- **Emergency**: >5 auth failures in 5 minutes (suggests credential rotation issue)

**Verification Checklist**:
- [ ] Verify AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY set correctly
- [ ] Check credential expiration (if using temporary credentials)
- [ ] Verify IAM permissions for bucket operations (GetObject, PutObject, ListBucket, DeleteObject)

---

#### Signal: S3 Bucket Storage Growth

**Source**: S3 bucket metrics via AWS CLI or boto3

**Collection Method**:
```bash
# Check bucket size and object count
aws s3 ls s3://ocr-production/ --recursive --summarize

# Monitor growth over time (daily snapshot)
aws s3api list-objects-v2 --bucket ocr-production --output json | \
  jq '.Contents | length, ([.[].Size] | add)'
```

**Expected Baseline**:
- Linear growth with job volume
- Cleanup tasks should prevent unbounded growth
- Average job size: ~5-20MB per document

**Alert Thresholds**:
- **Warning**: Storage growth >2x expected rate
- **Critical**: Cleanup task failures causing accumulation
- **Emergency**: Storage quota exceeded or cost spike >20%

**Verification Checklist**:
- [ ] Verify `cleanup_completed_jobs` task running on schedule
- [ ] Check for jobs with missing or failed cleanup
- [ ] Review S3 lifecycle policies (if configured)

---

### 3. Worker Health

#### Signal: Worker CPU/Memory Usage

**Source**: System monitoring (top, htop, or container metrics)

**Collection Method**:
```bash
# Monitor worker processes
ps aux | grep celery | grep worker

# Check memory usage
pmap $(pgrep -f "celery.*worker") | tail -1

# Monitor CPU usage
top -b -n 1 -p $(pgrep -f "celery.*worker" | head -1)
```

**Expected Baseline**:
- CPU: <50% per worker under normal load
- Memory: ~500MB-2GB per worker (depends on concurrency and page size)
- No memory leaks or unbounded growth

**Alert Thresholds**:
- **Warning**: CPU >80% sustained or memory >80% of container limit
- **Critical**: Worker OOM kills or CPU throttling
- **Emergency**: Multiple workers crashing/restarting

**Verification Checklist**:
- [ ] Check for memory leaks (gradual growth over hours/days)
- [ ] Verify worker concurrency settings appropriate for resources
- [ ] Review task prefetch settings (avoid overloading workers)

---

#### Signal: Worker Queue Depth

**Source**: Celery inspect commands

**Collection Method**:
```bash
# Check active tasks per worker
celery -A coordinator inspect active | jq 'to_entries | .[] | {worker: .key, tasks: (.value | length)}'

# Check reserved (prefetched) tasks
celery -A coordinator inspect reserved

# Check queue lengths
celery -A coordinator inspect stats | jq '.[] | {queue_length: .total}'
```

**Expected Baseline**:
- Queue depth: <100 pending tasks under normal load
- Workers should consume tasks at steady rate
- Spikes during high-traffic periods acceptable if queues drain

**Alert Thresholds**:
- **Warning**: Queue depth >500 for >10 minutes
- **Critical**: Queue depth >1000 or continuously growing
- **Emergency**: Workers not consuming tasks (stuck/deadlocked)

**Verification Checklist**:
- [ ] Check if workers are accepting new tasks
- [ ] Verify no deadlocks or stuck tasks
- [ ] Review task routing and worker pool sizing

---

### 4. Storage Backend Integrity

#### Signal: S3 Object Consistency

**Source**: Test suite + manual verification

**Collection Method**:
```bash
# Run storage integration tests
cd coordinator
python -m pytest jobs/tests/test_storage.py -v -k s3

# Verify object contents match expectations
aws s3 cp s3://ocr-production/jobs/{job_id}/metadata.json - | jq .

# Compare page count with metadata
aws s3 ls s3://ocr-production/jobs/{job_id}/pages/ | wc -l
```

**Expected Baseline**:
- All uploaded objects readable and intact
- Metadata matches actual stored artifacts
- No corrupted or truncated files

**Alert Thresholds**:
- **Critical**: Any corrupted files detected
- **Emergency**: Data loss or inability to read stored objects

**Verification Checklist**:
- [ ] Verify metadata.json present for all completed jobs
- [ ] Check page count matches `pages/` directory contents
- [ ] Validate final output PDF/text files readable

---

#### Signal: Storage Backend Mode Tracking

**Source**: Job metadata + application logs

**Collection Method**:
```bash
# Check which storage backend was used for recent jobs
grep "storage_backend" /var/log/celery/worker.log | tail -20

# Verify job metadata includes backend field (if implemented)
aws s3 cp s3://ocr-production/jobs/{job_id}/metadata.json - | jq '.storage_backend'
```

**Expected Baseline**:
- During rollout: Mix of NFS and S3 depending on stage
- After rollout: 100% S3
- No mid-job backend switches (would indicate config issue)

**Alert Thresholds**:
- **Warning**: Unexpected backend mode for worker configuration
- **Critical**: Backend mode changing mid-job (suggests config race)

**Verification Checklist**:
- [ ] Confirm STORAGE_BACKEND env var matches observed behavior
- [ ] Verify no workers with mismatched configuration
- [ ] Check for race conditions during config updates

---

## Verification Checklists by Rollout Stage

### Pre-Rollout (Stage 0)

**Baseline Health Check**:
- [ ] Run full test suite: `pytest tests/ -v` (expect: 930 passed, 25 skipped)
- [ ] Run storage tests: `pytest coordinator/jobs/tests/test_storage.py -v` (expect: 13 passed)
- [ ] Run task tests: `pytest coordinator/jobs/tests/test_tasks.py -v` (expect: 115 passed)
- [ ] Verify docs: `python scripts/check_docs.py --full-scan` (expect: 0 findings)
- [ ] Verify lint: `python -m ruff check .` (expect: 0 errors)

**System Health Snapshot**:
- [ ] Record baseline job completion rate
- [ ] Record baseline task execution times (P50, P95, P99)
- [ ] Record baseline error/retry rates
- [ ] Document current disk usage and worker resource usage

---

### Canary Stage (Stage 1)

**Pre-Flight**:
- [ ] S3 credentials configured and validated
- [ ] S3 bucket created and accessible
- [ ] Test worker started with S3 mode

**During Canary**:
- [ ] Monitor test job logs for S3 operations
- [ ] Verify S3 uploads successful: `aws s3 ls s3://{bucket}/jobs/{job_id}/`
- [ ] Check task completion: `celery -A coordinator inspect active`
- [ ] Verify no authentication errors in logs

**Post-Canary**:
- [ ] Test job completed successfully
- [ ] All output artifacts present in S3
- [ ] Task execution time <3x baseline
- [ ] No rollback triggers hit

---

### Limited Stage (Stage 2)

**Pre-Flight**:
- [ ] Canary stage validated successfully
- [ ] S3 worker pool deployed (10% capacity)
- [ ] Routing configured to send 10% traffic to S3

**Daily Checks** (for 1-2 days):
- [ ] S3 job completion rate >95%
- [ ] S3 job error rate <5%
- [ ] S3 task latency <2x NFS baseline
- [ ] Worker CPU/memory usage stable
- [ ] No authentication failures
- [ ] Queue depth manageable (<100)

**Post-Limited**:
- [ ] 24-48 hours stable operation
- [ ] No rollback triggers hit
- [ ] Team comfortable with S3 operations

---

### Broad Stage (Stage 3)

**Pre-Flight**:
- [ ] Limited stage validated successfully
- [ ] S3 worker pool scaled to 50% capacity
- [ ] Routing configured to send 50% traffic to S3

**Daily Checks** (for 3-5 days):
- [ ] S3 job completion rate >97%
- [ ] S3 job error rate <3%
- [ ] S3 task latency stable (no degradation over time)
- [ ] S3 API response times <1s P95
- [ ] Worker resource usage stable
- [ ] S3 storage growth linear and predictable

**Weekly Review**:
- [ ] S3 storage costs within budget
- [ ] No API throttling observed
- [ ] Team trained on S3 troubleshooting

**Post-Broad**:
- [ ] 3-5 days stable operation at scale
- [ ] No rollback triggers hit
- [ ] Ready for full migration

---

### Full Stage (Stage 4)

**Pre-Flight**:
- [ ] Broad stage validated successfully
- [ ] NFS workers ready for decommission
- [ ] Rollback plan reviewed and understood

**Daily Checks** (for 2 weeks):
- [ ] All jobs running on S3 backend (100%)
- [ ] Job completion rate >98%
- [ ] Job error rate <2%
- [ ] Worker pool healthy and stable
- [ ] S3 storage growth predictable

**Weekly Review**:
- [ ] S3 storage costs reviewed and approved
- [ ] Cleanup tasks running successfully
- [ ] No NFS workers in pool

**Post-Full**:
- [ ] 2 weeks stable operation at 100%
- [ ] NFS workers decommissioned
- [ ] S3 set as default configuration
- [ ] Documentation updated

---

## Log Indicators Reference

### Healthy S3 Operations

```
INFO Task ingest_document[abc123] succeeded in 3.2s
INFO Uploaded to S3: s3://bucket/jobs/job_456/pages/page_001.png
INFO S3 backend: downloaded 5 pages for assembly
INFO Task assemble_document[def789] succeeded in 12.4s
```

### Warning Signs

```
WARNING Retrying ingest_document (attempt 2/5) due to S3 upload timeout
WARNING S3 API response time 1523ms (threshold: 1000ms)
WARNING Worker memory usage: 1.8GB (80% of limit)
```

### Critical Errors

```
ERROR Task ingest_document[abc123] failed: NoCredentialsError
ERROR S3 upload failed after 5 retries: ConnectionError
ERROR Worker crashed: OutOfMemoryError
CRITICAL S3 bucket not accessible: 403 Forbidden
```

---

## Response Procedures

### Procedure: S3 Authentication Failure

1. **Detect**: Monitor logs for `NoCredentialsError`, `InvalidAccessKeyId`, or `SignatureDoesNotMatch`
2. **Validate**: Test credentials manually: `aws s3 ls s3://{bucket}/`
3. **Fix**:
   - Verify AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in worker .env
   - Check IAM permissions for bucket operations
   - Rotate credentials if expired or compromised
4. **Restart**: Restart affected workers with corrected configuration
5. **Monitor**: Watch logs for successful S3 operations

---

### Procedure: S3 API Throttling

1. **Detect**: Monitor logs for `SlowDown`, `RequestLimitExceeded`, or 503 errors
2. **Assess**: Check if throttling is transient or sustained
3. **Mitigate**:
   - Reduce worker concurrency to lower request rate
   - Implement exponential backoff in retry logic (already present)
   - Contact AWS support to request limit increase (if needed)
4. **Monitor**: Track S3 API response times and request rates

---

### Procedure: Worker Resource Exhaustion

1. **Detect**: Monitor CPU >80% sustained or memory approaching limit
2. **Assess**: Check active tasks and worker configuration
3. **Mitigate**:
   - Reduce worker concurrency (e.g., from 4 to 2)
   - Adjust task prefetch settings to avoid overload
   - Scale horizontally (add more workers with lower concurrency)
4. **Restart**: Restart workers with new configuration
5. **Monitor**: Track CPU/memory usage and task throughput

---

### Procedure: Unexplained Job Failures

1. **Detect**: Job completion rate drops or error rate spikes
2. **Investigate**:
   - Review worker logs for exception patterns
   - Check S3 bucket for missing/corrupted artifacts
   - Verify task retry logs for persistent failures
3. **Isolate**: Identify if issue affects specific job types, file sizes, or workers
4. **Fix**: Address root cause (code bug, config issue, infrastructure problem)
5. **Validate**: Re-run failed jobs to confirm fix

---

## Alert Configuration (Future)

When transitioning to automated alerting, use these thresholds as starting points:

### High-Priority Alerts (PagerDuty/Opsgenie)
- S3 authentication failures (any occurrence)
- Job completion rate <95% for >30 minutes
- Worker crashes or OOM kills
- S3 endpoint unreachable

### Medium-Priority Alerts (Slack/Email)
- Job completion rate <97% for >1 hour
- Task retry rate >5% for >1 hour
- S3 API response time >1s P95 for >15 minutes
- Worker CPU/memory >80% sustained

### Low-Priority Alerts (Dashboard/Metrics)
- Queue depth >100 for >10 minutes
- S3 storage growth >2x expected
- Task latency >2x baseline

---

## Related Documentation

- `s3-rollout-runbook.md` - Staged rollout procedure and rollback gates
- `coordinator/jobs/storage.py` - Storage backend implementation
- `coordinator/jobs/tests/test_storage.py` - Storage tests and validation

---

**Approved By**: _TBD_
**Next Review**: After each rollout stage or incident
