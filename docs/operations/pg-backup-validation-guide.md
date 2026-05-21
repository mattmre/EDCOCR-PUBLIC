# PostgreSQL Backup Validation Guide

**Version**: 1.1.0 | **Last Updated**: 2026-05-20

This guide covers the PostgreSQL backup validation framework for the EDCOCR distributed pipeline. It validates backup configuration, tests backup integrity, and optionally performs restore-to-temporary-database verification.

---

## Table of Contents

1. [Backup Strategy Overview](#1-backup-strategy-overview)
2. [Validation Script Usage](#2-validation-script-usage)
3. [Restore Testing Procedure](#3-restore-testing-procedure)
4. [Scheduled Validation Setup](#4-scheduled-validation-setup)
5. [Troubleshooting Backup Failures](#5-troubleshooting-backup-failures)
6. [Integration with Helm CronJob](#6-integration-with-helm-cronjob)

---

## 1. Backup Strategy Overview

### What Gets Backed Up

The PostgreSQL backup CronJob (Helm: `postgresql.backup.enabled`) protects the **coordinator database**, which is the authoritative record of:

| Data | Table | Impact if Lost |
|------|-------|----------------|
| Job metadata | `jobs_job` | All job history, status, and tracking lost |
| Worker registry | `jobs_worker` | Worker fleet state and heartbeat history lost |
| Page results | `jobs_pageresult` | Per-page OCR confidence and quality data lost |
| Custody events | `jobs_custodyevent` | Forensic chain-of-custody audit trail lost |
| PII entities | `jobs_piientity` | Spatial PII extraction results lost |

### What is NOT Backed Up

- **Source documents** (`ocr_source/`) -- stored on NFS or S3; use filesystem snapshots or S3 versioning
- **Output documents** (`ocr_output/`) -- regenerable by reprocessing from source
- **Temp files** (`ocr_temp/`) -- ephemeral crash-resume data; safe to discard
- **RabbitMQ queues** -- in-flight tasks are requeued via `task_acks_late`
- **Redis cache** -- transient data; auto-rebuilds on restart

### Backup Format

The Helm CronJob uses PostgreSQL custom format (`--format=custom`) with compression level 6:

```
pg_dump --format=custom --compress=6 -f /backups/ocr-coordinator-YYYYMMDD-HHMMSS.sql.gz
```

Custom format advantages:
- Selective table/schema restore
- Parallel restore support
- Built-in compression
- `pg_restore --list` for content inspection without full restore

### Retention Policy

| Parameter | Default | Configurable Via |
|-----------|---------|------------------|
| Schedule | `0 2 * * *` (daily 2 AM UTC) | `postgresql.backup.schedule` |
| Retention | 7 backups | `postgresql.backup.retentionCount` |
| Storage | 20 Gi PVC | `postgresql.backup.storage` |
| Timeout | 3600s (1 hour) | `postgresql.backup.timeoutSeconds` |

Old backups beyond the retention count are automatically pruned by the CronJob.

---

## 2. Validation Script Usage

### Prerequisites

- Python 3.10+
- PostgreSQL client tools (`pg_dump`, `pg_restore`, `pg_isready`, `psql`) available in PATH
- Network access to the PostgreSQL instance

### Modes

#### Check Mode (`--check`)

Verifies that the backup infrastructure is correctly configured.

```bash
python scripts/pg_backup_validation.py --check \
    --database-url postgres://ocr:pass@localhost:5432/ocr_coordinator \
    --backup-dir /backups
```

Checks performed:
- Database URL format is valid
- Database is reachable (`pg_isready`)
- Backup directory exists and is writable
- `pg_dump` binary is available
- `pg_restore` binary is available
- Helm values are correct (if `--helm-values` is provided)

#### Verify Mode (`--verify`)

Inspects existing backup files for integrity and recency.

```bash
python scripts/pg_backup_validation.py --verify \
    --backup-dir /backups \
    --max-age-hours 24
```

Checks performed per file:
- File matches expected naming pattern (`ocr-coordinator-YYYYMMDD-HHMMSS.sql.gz`)
- File size exceeds minimum (default: 1024 bytes)
- File age is within threshold (default: 24 hours)
- `pg_restore --list` succeeds (valid custom-format archive)
- Table count extracted from archive listing

#### Backup Mode (`--backup`)

Triggers a manual `pg_dump` and verifies the output.

```bash
# Actual backup
python scripts/pg_backup_validation.py --backup \
    --database-url postgres://ocr:pass@localhost:5432/ocr_coordinator \
    --backup-dir /backups

# Dry run (no changes)
python scripts/pg_backup_validation.py --backup \
    --database-url postgres://ocr:pass@localhost:5432/ocr_coordinator \
    --backup-dir /backups \
    --dry-run
```

#### Restore Test Mode (`--restore-test`)

Creates a temporary database, restores the latest backup, compares row counts, then drops the temporary database.

```bash
python scripts/pg_backup_validation.py --restore-test \
    --database-url postgres://ocr:pass@localhost:5432/ocr_coordinator \
    --backup-dir /backups
```

**Warning**: This creates a temporary database named `ocr_restore_test_<timestamp>`. The database user must have `CREATE DATABASE` privileges.

#### Report Mode (`--report`)

Generates a comprehensive backup health report in both JSON and markdown formats.

```bash
python scripts/pg_backup_validation.py --report \
    --database-url postgres://ocr:pass@localhost:5432/ocr_coordinator \
    --backup-dir /backups \
    --output-dir ./reports
```

Output files:
- `pg-backup-validation-YYYYMMDD-HHMMSS.json` -- machine-readable report
- `pg-backup-validation-YYYYMMDD-HHMMSS.md` -- human-readable report

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--database-url` | `$DATABASE_URL` | PostgreSQL connection URL |
| `--backup-dir` | `/backups` | Directory containing backup files |
| `--output-dir` | `.` | Directory for report output |
| `--max-age-hours` | `24` | Maximum acceptable backup age |
| `--min-size-bytes` | `1024` | Minimum acceptable backup size |
| `--dry-run` | `false` | Show what would happen without changes |
| `--helm-values` | (none) | Path to Helm values.yaml for config checks |
| `--verbose` | `false` | Enable debug logging |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed |
| 1 | One or more checks failed |
| 2 | CLI argument error |

---

## 3. Restore Testing Procedure

### Automated Restore Test

The `--restore-test` mode automates the full restore verification workflow:

```bash
python scripts/pg_backup_validation.py --restore-test \
    --database-url postgres://ocr:pass@localhost:5432/ocr_coordinator \
    --backup-dir /backups
```

The script:
1. Finds the latest backup file in `--backup-dir`
2. Creates a temporary database (`ocr_restore_test_<timestamp>`)
3. Runs `pg_restore` into the temporary database
4. Queries row counts for `jobs_job`, `jobs_worker`, `jobs_pageresult`, `jobs_custodyevent`
5. Compares row counts between source and restored databases
6. Drops the temporary database (even on failure)

### Manual Restore Test

For more control, perform a manual restore test:

```bash
# 1. Create a temporary database
psql -U ocr -d postgres -c "CREATE DATABASE ocr_restore_test OWNER ocr;"

# 2. Restore the latest backup
pg_restore -U ocr -d ocr_restore_test --no-owner --verbose \
    /backups/ocr-coordinator-20260315-020000.sql.gz

# 3. Compare row counts
psql -U ocr -d ocr_coordinator -c "SELECT count(*) FROM jobs_job;"
psql -U ocr -d ocr_restore_test -c "SELECT count(*) FROM jobs_job;"

# 4. Clean up
psql -U ocr -d postgres -c "DROP DATABASE ocr_restore_test;"
```

### Kubernetes Restore Test

To run the restore test inside a Kubernetes cluster:

```bash
RELEASE=ocr-local

# Run from a coordinator pod
kubectl exec -it $(kubectl get pod -l app.kubernetes.io/component=coordinator -o name | head -1) -- \
    python scripts/pg_backup_validation.py --restore-test \
        --database-url "$DATABASE_URL" \
        --backup-dir /backups
```

Or create a one-off Job:

```bash
kubectl create job pg-restore-test --from=cronjob/${RELEASE}-postgres-backup -- \
    python /app/scripts/pg_backup_validation.py --restore-test \
        --database-url "$DATABASE_URL" \
        --backup-dir /backups
```

---

## 4. Scheduled Validation Setup

### Kubernetes CronJob

Add a validation CronJob alongside the backup CronJob. Create a custom values override:

```yaml
# values-backup-validation.yaml
postgresql:
  backup:
    enabled: true
    schedule: "0 2 * * *"     # Backup at 2 AM UTC
    retentionCount: 7
```

Then create a separate CronJob manifest for validation:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: pg-backup-validation
spec:
  schedule: "0 4 * * *"    # Validate at 4 AM UTC (after backup completes)
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: validator
              image: ocr-local/coordinator:latest
              command:
                - python
                - scripts/pg_backup_validation.py
                - --verify
                - --backup-dir
                - /backups
                - --max-age-hours
                - "26"
                - --report
                - --output-dir
                - /backups/reports
              env:
                - name: DATABASE_URL
                  valueFrom:
                    secretKeyRef:
                      name: ocr-local-secret
                      key: DATABASE_URL
              volumeMounts:
                - name: backup-storage
                  mountPath: /backups
          volumes:
            - name: backup-storage
              persistentVolumeClaim:
                claimName: ocr-local-postgres-backup
```

### Docker Compose

Add a scheduled validation check using cron on the host:

```bash
# /etc/cron.d/pg-backup-validation
0 4 * * * root docker compose -f /path/to/docker-compose.coordinator.yml exec -T postgres \
    python /app/scripts/pg_backup_validation.py --verify --backup-dir /backups --max-age-hours 26 \
    >> /var/log/pg-backup-validation.log 2>&1
```

### CI/CD Integration

Add backup validation as a periodic CI job:

```yaml
# .github/workflows/backup-validation.yml
name: Backup Validation
on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC
jobs:
  validate:
    runs-on: self-hosted  # Must have DB access
    steps:
      - uses: actions/checkout@v4
      - run: |
          python scripts/pg_backup_validation.py --report \
              --database-url "$DATABASE_URL" \
              --backup-dir /backups \
              --output-dir ./reports
      - uses: actions/upload-artifact@v4
        with:
          name: backup-validation-report
          path: reports/
```

---

## 5. Troubleshooting Backup Failures

### No Backup Files Found

**Symptom**: `--verify` reports 0 files in backup directory.

**Causes and fixes**:

| Cause | Fix |
|-------|-----|
| CronJob not enabled | Set `postgresql.backup.enabled: true` in Helm values |
| CronJob suspended | `kubectl get cronjob` -- check SUSPEND column |
| Wrong backup directory | Verify `--backup-dir` matches the CronJob mount path |
| PVC not mounted | Check PVC status: `kubectl get pvc` |
| Permissions | Backup container runs as UID 999 -- PVC must be writable |

### Backup File Too Small

**Symptom**: Verify reports "File too small" (< 1024 bytes).

**Causes**:
- `pg_dump` failed mid-write (OOM, timeout, disk full)
- Connection dropped during dump
- Database is empty (new deployment)

**Fix**:
```bash
# Check CronJob logs
kubectl logs job/<release>-postgres-backup-<id>

# Trigger a manual backup to see real-time output
python scripts/pg_backup_validation.py --backup \
    --database-url "$DATABASE_URL" \
    --backup-dir /backups \
    --verbose
```

### Backup Too Old

**Symptom**: Verify reports "File too old" (> max-age-hours).

**Causes**:
- CronJob schedule is wrong or hasn't fired
- CronJob is hitting `activeDeadlineSeconds` and timing out
- Clock skew between backup container and validation

**Fix**:
```bash
# Check CronJob status and last schedule time
kubectl get cronjob <release>-postgres-backup

# Check recent job history
kubectl get jobs --sort-by=.status.startTime | grep backup

# Manually trigger a backup
kubectl create job --from=cronjob/<release>-postgres-backup manual-backup-$(date +%s)
```

### pg_restore --list Fails

**Symptom**: Verify reports "pg_restore --list failed".

**Causes**:
- Backup file is corrupted (partial write, disk error)
- pg_restore version mismatch (backup created with newer PostgreSQL)
- File is not custom format (wrong `--format` flag)

**Fix**:
```bash
# Check the file header manually
file /backups/ocr-coordinator-*.sql.gz | head -5

# Try listing with verbose output
pg_restore --list --verbose /backups/<backup-file>

# Check PostgreSQL versions match
pg_dump --version
pg_restore --version
```

### Row Count Mismatch After Restore

**Symptom**: `--restore-test` reports row count differences.

**Causes**:
- Backup was taken while data was being written (expected, minor discrepancy)
- Backup predates recent data changes (normal for older backups)
- Foreign key constraint failures during restore

**Assessment**:
- Small discrepancies (< 1%) are normal for hot backups
- Large discrepancies indicate a backup timing or corruption issue
- Check pg_restore stderr for skipped tables or constraint errors

---

## 6. Integration with Helm CronJob

### Helm Values Reference

The backup CronJob is controlled by these `values.yaml` keys:

```yaml
postgresql:
  backup:
    enabled: false              # Master toggle (must be true for backups)
    schedule: "0 2 * * *"      # Cron schedule (daily at 2 AM UTC)
    retentionCount: 7           # Number of backups to keep
    storage: 20Gi               # PVC size for backup storage
    storageClass: ""            # StorageClass override (optional)
    historyLimit: 3             # Completed CronJob history
    timeoutSeconds: 3600        # Max backup duration (1 hour)
```

### Validating Helm Values

Use the `--helm-values` flag to validate your values file:

```bash
python scripts/pg_backup_validation.py --check \
    --helm-values helm/ocr-local/values.yaml \
    --database-url "$DATABASE_URL" \
    --backup-dir /backups
```

This checks:
- `postgresql.backup.enabled` is true
- `schedule` is a valid 5-field cron expression
- `retentionCount` is > 0

### Monitoring Backup Health

Combine the validation script with the existing Prometheus monitoring:

1. Run `--report` mode on a schedule
2. Parse the JSON output for `summary.overall_health`
3. Alert on `DEGRADED` or `CRITICAL` status
4. Include backup validation in the failover runbook drill

See [FAILOVER-RUNBOOK.md](../FAILOVER-RUNBOOK.md) Section 8 for the complete backup and restore procedure.

---

## Quick Reference

```bash
# Check everything is configured correctly
python scripts/pg_backup_validation.py --check --database-url "$DATABASE_URL"

# Verify existing backups are healthy
python scripts/pg_backup_validation.py --verify --backup-dir /backups --max-age-hours 24

# Take a manual backup
python scripts/pg_backup_validation.py --backup --database-url "$DATABASE_URL" --backup-dir /backups

# Full restore test
python scripts/pg_backup_validation.py --restore-test --database-url "$DATABASE_URL" --backup-dir /backups

# Generate comprehensive report
python scripts/pg_backup_validation.py --report --database-url "$DATABASE_URL" --backup-dir /backups --output-dir ./reports
```
