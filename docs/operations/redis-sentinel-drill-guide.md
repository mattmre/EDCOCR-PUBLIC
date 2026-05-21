# Redis Sentinel Failover Drill Guide

**Version**: 1.1.0 | **Last Updated**: 2026-05-20

This guide covers how to run, interpret, and schedule Redis Sentinel failover
drills for the EDCOCR distributed pipeline.  Drills validate that the
Sentinel topology is healthy and that application-layer reconnection works
correctly after a master promotion.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Pre-Drill Checklist](#3-pre-drill-checklist)
4. [Running a Drill](#4-running-a-drill)
5. [Interpreting Results](#5-interpreting-results)
6. [Expected Outcomes](#6-expected-outcomes)
7. [Troubleshooting Failed Drills](#7-troubleshooting-failed-drills)
8. [Scheduling Periodic Drills](#8-scheduling-periodic-drills)
9. [Reference](#9-reference)

---

## 1. Overview

The Redis Sentinel drill framework (`scripts/redis_sentinel_drill.py`) provides
four operational modes:

| Mode | Flag | Description |
|------|------|-------------|
| **Check** | `--check` | Validate Sentinel topology: quorum, auth, master reachability, replication sync |
| **Simulate** | `--simulate` | Trigger `SENTINEL FAILOVER` and measure promotion time |
| **Validate** | `--validate` | Post-failover app reconnection: Celery result backend, Django cache |
| **Report** | `--report` | Generate JSON + markdown drill report |

A full drill (all four modes) takes approximately 15-30 seconds in a healthy
environment.

### Architecture Context

In the EDCOCR distributed pipeline, Redis serves two roles:

1. **Celery result backend** -- stores task results, chord unlock keys, and
   page-level processing metadata.
2. **Django cache** -- caches fleet status, metrics queries, and admin data.

When Redis Sentinel is enabled (`redis.sentinel.enabled=true` in Helm values or
the Docker Compose HA overlay), the Sentinel cluster monitors a single master
and one or more replicas.  If the master becomes unreachable for
`down-after-milliseconds` (default: 5000ms), Sentinel promotes a replica and
reconfigures clients.

### When to Run Drills

- After initial Sentinel deployment or configuration changes
- Before and after maintenance windows
- Monthly as part of operational readiness validation
- After any infrastructure incident involving Redis

---

## 2. Prerequisites

### Software

- Python 3.10+
- Network access to Sentinel instances (port 26379) and Redis master/replicas (port 6379)
- Optional: `redis` Python package for richer error messages (falls back to raw TCP sockets)

### Infrastructure

- Redis Sentinel deployed (at least 3 Sentinel instances for production)
- At least 1 Redis replica configured (failover requires a promotion target)
- Redis master is accepting connections

### Credentials

Set these environment variables if authentication is enabled:

```bash
export REDIS_SENTINEL_PASSWORD="<sentinel-requirepass>"
export REDIS_PASSWORD="<redis-requirepass>"
```

---

## 3. Pre-Drill Checklist

Before running any drill, verify the following:

- [ ] **Sentinel instances are running**: at least 3 Sentinel pods/containers
- [ ] **Replica is connected**: `redis-cli -p 26379 sentinel replicas ocr-master` shows at least 1 replica
- [ ] **No active failover**: `redis-cli -p 26379 sentinel master ocr-master` shows `flags: master` (no `s_down` or `o_down`)
- [ ] **No critical jobs in flight**: check `fleet_status` for active processing jobs
- [ ] **Monitoring is active**: Prometheus/Grafana alerting is operational
- [ ] **Notification channel ready**: team is aware a drill is in progress
- [ ] **Credentials are set**: `REDIS_SENTINEL_PASSWORD` and `REDIS_PASSWORD` if applicable

### Quick Pre-Flight Check

```bash
# Docker Compose (HA overlay)
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec redis-sentinel1 redis-cli -p 26379 sentinel master ocr-master

# Kubernetes
SENTINEL_POD=$(kubectl get pod -l app.kubernetes.io/component=redis-sentinel -o name | head -1)
kubectl exec $SENTINEL_POD -- redis-cli -p 26379 sentinel master ocr-master
```

Look for:
- `flags: master` (no `s_down`, `o_down`, or `failover_in_progress`)
- `num-slaves` >= 1
- `num-other-sentinels` >= 2 (for quorum of 2)

---

## 4. Running a Drill

### Check Only (Non-Destructive)

Validates configuration without triggering any failover:

```bash
python scripts/redis_sentinel_drill.py \
  --check \
  --sentinel-host localhost \
  --sentinel-port 26379 \
  --master-name ocr-master
```

### Full Drill (Check + Simulate + Validate + Report)

Triggers an actual failover and validates reconnection:

```bash
python scripts/redis_sentinel_drill.py \
  --check --simulate --validate --report \
  --sentinel-host sentinel.prod.internal \
  --sentinel-port 26379 \
  --master-name ocr-master \
  --timeout 30 \
  --output-dir sentinel_drill_results
```

### Kubernetes Example

```bash
# Port-forward Sentinel to localhost
kubectl port-forward svc/ocr-local-redis-sentinel 26379:26379 &

# Run drill
REDIS_PASSWORD=$(kubectl get secret ocr-local-secret -o jsonpath='{.data.REDIS_PASSWORD}' | base64 -d)
REDIS_SENTINEL_PASSWORD=$REDIS_PASSWORD
export REDIS_PASSWORD REDIS_SENTINEL_PASSWORD

python scripts/redis_sentinel_drill.py \
  --check --simulate --validate --report \
  --sentinel-host localhost \
  --sentinel-port 26379 \
  --output-dir sentinel_drill_results
```

### Docker Compose HA Example

```bash
# From the coordinator directory
export REDIS_PASSWORD=$(grep REDIS_PASSWORD .env | cut -d= -f2)
export REDIS_SENTINEL_PASSWORD=$REDIS_PASSWORD

python scripts/redis_sentinel_drill.py \
  --check --simulate --validate --report \
  --sentinel-host redis-sentinel1 \
  --output-dir sentinel_drill_results
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--check` | off | Validate Sentinel configuration |
| `--simulate` | off | Trigger `SENTINEL FAILOVER` |
| `--validate` | off | Post-failover reconnection checks |
| `--report` | off | Generate JSON + markdown reports |
| `--sentinel-host` | `localhost` | Sentinel hostname |
| `--sentinel-port` | `26379` | Sentinel port |
| `--master-name` | `ocr-master` | Monitored master name |
| `--timeout` | `30` | Per-step timeout (seconds) |
| `--output-dir` | `sentinel_drill_results` | Report output directory |
| `--verbose`, `-v` | off | Debug logging |

---

## 5. Interpreting Results

### Output Files

The `--report` mode generates two files:

```
sentinel_drill_results/
  sentinel_drill_<id>.json    # Machine-readable full results
  sentinel_drill_<id>.md      # Human-readable summary
```

### Overall Status

| Status | Meaning |
|--------|---------|
| `passed` | All steps completed successfully |
| `partial` | Some steps passed, some failed -- review details |
| `failed` | All steps failed or critical steps failed |

### Key Metrics

- **Failover Duration**: Time from `SENTINEL FAILOVER` command to new master
  being confirmed.  RTO target is < 15 seconds.
- **Write/Read Round-Trip**: Confirms the new master accepts writes.
- **Celery Keyspace**: Confirms the result backend namespace is accessible.
- **Django Cache**: Confirms the cache backend namespace is accessible.

### JSON Report Schema

```json
{
  "drill_id": "abc123def456",
  "timestamp": "2026-03-15T12:00:00+00:00",
  "pipeline_version": "1.0.0",
  "sentinel_host": "sentinel.prod",
  "sentinel_port": 26379,
  "master_name": "ocr-master",
  "modes": ["check", "simulate", "validate", "report"],
  "status": "passed",
  "failover_triggered": true,
  "failover_duration_seconds": 3.456,
  "old_master": "10.0.0.1:6379",
  "new_master": "10.0.0.2:6379",
  "celery_reconnected": true,
  "cache_reconnected": true,
  "steps": [
    {
      "name": "sentinel_ping",
      "description": "Verify Sentinel responds to PING",
      "status": "passed",
      "duration_seconds": 0.01,
      "details": {},
      "error": ""
    }
  ]
}
```

---

## 6. Expected Outcomes

### Healthy Environment

| Step | Expected Status | Notes |
|------|----------------|-------|
| `sentinel_ping` | passed | Sentinel responds within timeout |
| `master_info` | passed | Master IP/port resolved, flags = `master` |
| `master_ping` | passed | Master responds to PING |
| `replicas_check` | passed | At least 1 replica found |
| `quorum_check` | passed | sentinel_count >= quorum |
| `replication_health` | passed | role = master, connected_slaves >= 1 |
| `record_master` | passed | Old master address captured |
| `trigger_failover` | passed | Sentinel returns OK |
| `wait_promotion` | passed | New master differs from old, < 15s |
| `new_master_ping` | passed | New master responds |
| `write_read_test` | passed | SET/GET round-trip succeeds |
| `celery_result_probe` | passed | Celery keyspace writable |
| `django_cache_probe` | passed | Django cache keyspace writable |

### After Failover

After a successful `--simulate`, the master role has swapped.  Running the
drill again will show the previous replica as the new master.  This is
expected and correct -- no manual intervention is needed.

---

## 7. Troubleshooting Failed Drills

### sentinel_ping Failed

**Cause**: Sentinel is unreachable.

**Actions**:
1. Verify Sentinel pods/containers are running.
2. Check network connectivity to Sentinel port (26379).
3. Verify `REDIS_SENTINEL_PASSWORD` matches Sentinel `requirepass`.
4. Check firewall/network policy rules.

```bash
# Quick connectivity test
redis-cli -h <sentinel-host> -p 26379 ping
```

### master_info Failed

**Cause**: Sentinel does not know about the monitored master.

**Actions**:
1. Verify master name matches (`--master-name` vs Sentinel config).
2. Check Sentinel logs for `+sdown` or `-sdown` events.
3. Confirm Sentinel configuration with `SENTINEL MASTER <name>`.

### replicas_check Failed

**Cause**: No replicas are connected to the master.

**Actions**:
1. Check replica pod/container status.
2. Verify replica configuration (`replicaof` directive).
3. Check `REDIS_PASSWORD` / `masterauth` on replicas.

### quorum_check Failed

**Cause**: Insufficient Sentinels for quorum.

**Actions**:
1. Scale Sentinel instances to meet quorum (typically 3 Sentinels with quorum of 2).
2. Verify all Sentinel instances have the same `sentinel monitor` configuration.

### trigger_failover Failed with NOGOODSLAVE

**Cause**: No eligible replica for promotion (all replicas down or lagging).

**Actions**:
1. Check replica health: `redis-cli info replication` on replicas.
2. Verify replication is not stalled (check `master_link_status`).
3. Wait for replication to catch up, then retry.

### wait_promotion Timed Out

**Cause**: Failover initiated but promotion did not complete in time.

**Actions**:
1. Check Sentinel logs for failover progress events.
2. Increase `--timeout` if network latency is high.
3. Verify no conflicting failover is in progress.
4. Check if `failover-timeout` in Sentinel config is too short.

### write_read_test Failed with READONLY

**Cause**: The node resolved as new master is still in replica mode.

**Actions**:
1. Wait a few seconds for Sentinel reconfiguration to propagate.
2. Re-run `--validate` mode only.
3. Check `SENTINEL MASTER <name>` to confirm the new master.

### celery_result_probe or django_cache_probe Failed

**Cause**: Application keyspace is not accessible on the new master.

**Actions**:
1. Verify the Celery result backend URL uses Sentinel transport
   (`sentinel://:password@sentinel1:26379/0`).
2. Verify `REDIS_SENTINEL_MASTER_NAME` matches the monitored master name.
3. Restart Celery workers if they cached the old master address.
4. Check Django `CACHES` configuration uses Sentinel-aware backend.

---

## 8. Scheduling Periodic Drills

### Recommended Schedule

| Frequency | Drill Type | When |
|-----------|-----------|------|
| **Weekly** | `--check` only | Off-peak hours, automated |
| **Monthly** | Full (`--check --simulate --validate --report`) | Scheduled maintenance window |
| **After changes** | Full | After any Redis/Sentinel config change |
| **Quarterly** | Full + manual verification | As part of DR drill program |

### Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: redis-sentinel-drill
spec:
  schedule: "0 3 * * 0"  # Weekly Sunday at 3 AM UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: drill
              image: ocr-local/coordinator:latest
              command:
                - python
                - scripts/redis_sentinel_drill.py
                - --check
                - --report
                - --sentinel-host
                - ocr-local-redis-sentinel
                - --output-dir
                - /shared/sentinel_drill_results
              env:
                - name: REDIS_SENTINEL_PASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: ocr-local-secret
                      key: REDIS_PASSWORD
                - name: REDIS_PASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: ocr-local-secret
                      key: REDIS_PASSWORD
              volumeMounts:
                - name: shared
                  mountPath: /shared
          volumes:
            - name: shared
              persistentVolumeClaim:
                claimName: ocr-local-shared
          restartPolicy: OnFailure
```

### Docker Compose Cron (via host crontab)

```bash
# /etc/cron.d/sentinel-drill
0 3 * * 0 root cd /opt/ocr-local && \
  docker compose -f coordinator/docker-compose.coordinator.yml \
    exec -T django python /app/scripts/redis_sentinel_drill.py \
    --check --report \
    --sentinel-host redis-sentinel1 \
    --output-dir /shared/sentinel_drill_results \
    2>&1 >> /var/log/sentinel-drill.log
```

### Alerting on Drill Failures

Integrate drill results into your monitoring pipeline:

```bash
# After drill, check exit code
python scripts/redis_sentinel_drill.py --check --report
if [ $? -ne 0 ]; then
  # Send alert via PagerDuty, Slack, etc.
  curl -X POST https://hooks.slack.com/services/... \
    -d '{"text":"Redis Sentinel drill FAILED - check results"}'
fi
```

---

## 9. Reference

### Related Documentation

- [Failover Runbook](../FAILOVER-RUNBOOK.md) -- Section 4 covers Redis failover procedures
- [Production Cutover Runbook](production-cutover-runbook.md) -- Redis HA setup during initial deployment
- [Helm Chart values.yaml](../../helm/ocr-local/values.yaml) -- `redis.sentinel.*` configuration

### Sentinel Configuration (from Helm values.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `redis.sentinel.enabled` | `false` | Enable Sentinel + replicas |
| `redis.sentinel.masterName` | `ocr-master` | Monitored master name |
| `redis.sentinel.replicas` | `1` | Number of read replicas |
| `redis.sentinel.sentinelReplicas` | `3` | Number of Sentinel instances |
| `redis.sentinel.quorum` | `2` | Sentinels required for failover |
| `redis.sentinel.downAfterMs` | `5000` | Master down detection threshold |
| `redis.sentinel.failoverTimeoutMs` | `10000` | Maximum failover duration |

### Recovery Time Objective

Per the failover runbook, the RTO target for Redis Sentinel failover is
**< 15 seconds**.  The drill measures actual promotion time against this
target and reports whether the RTO was met.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REDIS_SENTINEL_PASSWORD` | If auth enabled | Sentinel `requirepass` credential |
| `REDIS_PASSWORD` | If auth enabled | Master/replica `requirepass` credential |
