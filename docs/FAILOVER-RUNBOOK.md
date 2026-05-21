# Failover Runbook -- EDCOCR Distributed Pipeline

**Version**: 1.2.0 | **Last Updated**: 2026-05-20

This runbook covers failure detection, failover procedures, and recovery for every
stateful component in the EDCOCR distributed pipeline. Each section provides
procedures for both Docker Compose (single-host / multi-VPS) and Kubernetes (Helm
chart) deployments.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [PostgreSQL Failover](#2-postgresql-failover)
3. [RabbitMQ Failover](#3-rabbitmq-failover)
4. [Redis Failover](#4-redis-failover)
5. [Worker Recovery](#5-worker-recovery)
6. [Full Cluster Recovery](#6-full-cluster-recovery)
7. [Monitoring and Alerting](#7-monitoring-and-alerting)
8. [Backup and Restore](#8-backup-and-restore)

---

## 1. Architecture Overview

### Component Topology

```
                          +-------------------+
                          |   Load Balancer   |
                          |  (Ingress / LB)   |
                          +--------+----------+
                                   |
                    +--------------+--------------+
                    |                              |
           +-------v--------+           +---------v-------+
           |  Coordinator   |           |   Coordinator   |
           |  (Django +     |           |   (replica 2)   |
           |   Gunicorn)    |           |                 |
           +---+----+-------+           +---+----+--------+
               |    |                       |    |
       +-------+    +--------+    +---------+    +-------+
       |                     |    |                      |
+------v------+     +--------v----v-------+     +--------v-----+
| PostgreSQL  |     |     RabbitMQ        |     |    Redis     |
|  (primary)  |     |  (node1/2/3 in HA)  |     |  (master)    |
|             |     |   quorum queues     |     |              |
+------+------+     +-----+----+----+----+     +------+-------+
       |                  |    |    |                  |
       |           (HA overlay only)           +------v-------+
       |                                       | Redis Replica|
       |                                       +------+-------+
       |                                              |
       |                                    +---------v----------+
       |                                    | Sentinel 1 / 2 / 3 |
       |                                    +--------------------+
       |
+------v--------------------------------------------------------------+
|                        Celery Workers                                |
|                                                                      |
|  +-------------------+  +-------------------+  +-------------------+ |
|  | celery-coordinator|  | GPU Worker (N)    |  | CPU Worker (M)    | |
|  | queue: coordinator|  | queues: ocr_gpu,  |  | queue: cpu_general| |
|  | + celery-beat     |  |   cpu_general     |  |                   | |
|  +-------------------+  +-------------------+  +-------------------+ |
+----------------------------------------------------------------------+
```

### Component Dependencies

| Component | Depends On | Impact if Down |
|-----------|-----------|----------------|
| PostgreSQL | -- | All job tracking, worker registry, and admin lost. Pipeline halts. |
| RabbitMQ | -- | No new tasks dispatched. In-flight tasks continue. Pipeline stalls on completion. |
| Redis | -- | Result backend unavailable. Chord callbacks fail. Cache misses in Django. |
| Coordinator (Django) | PostgreSQL, RabbitMQ, Redis | Cannot submit jobs or view status. Workers continue in-flight tasks. |
| Celery Beat | PostgreSQL, RabbitMQ | Periodic tasks stop (heartbeat checks, cleanup). Workers unaffected. |
| GPU Workers | RabbitMQ, Redis, PostgreSQL, NFS/S3 | OCR processing halts. Jobs stall in "processing" state. |
| CPU Workers | RabbitMQ, Redis, PostgreSQL, NFS/S3 | Compression and NER halt. Jobs complete OCR but skip post-processing. |
| Flower | RabbitMQ | Monitoring dashboard unavailable. No operational impact. |

### Recovery Time Objectives

| Scenario | RTO Target | Notes |
|----------|-----------|-------|
| Single worker crash | < 1 minute | Celery auto-requeue via `task_reject_on_worker_lost` |
| Redis failover (Sentinel) | < 15 seconds | Automatic via Sentinel quorum (`down-after-milliseconds: 5000`) |
| RabbitMQ node loss (HA) | < 30 seconds | Quorum queues survive minority node loss |
| PostgreSQL restart | 1-5 minutes | Single-replica; requires pod/container restart |
| PostgreSQL restore from backup | 10-30 minutes | Depends on database size |
| Full cluster cold start | 5-10 minutes | Start order: PostgreSQL, RabbitMQ, Redis, Coordinator, Workers |

---

## 2. PostgreSQL Failover

PostgreSQL is a **single-replica** StatefulSet in Kubernetes and a single container
in Docker Compose. There is no automatic primary/replica failover. Recovery depends
on restarting the container or restoring from a backup.

### 2.1 Symptoms of Failure

**Application-level indicators:**

- Django returns HTTP 500 with `OperationalError: could not connect to server`
- Celery tasks fail with `django.db.utils.OperationalError`
- `fleet_status` management command fails
- Flower shows tasks stuck in PENDING (no DB write for status update)

**Infrastructure-level indicators:**

- Docker: `docker inspect` shows healthcheck status `unhealthy`
- Kubernetes: pod in `CrashLoopBackOff` or readiness probe failing (`pg_isready` returns non-zero)
- Prometheus alert: no specific PostgreSQL alert in the default rule set; monitor via custom PostgreSQL exporter

**Quick diagnosis:**

```bash
# Docker Compose
docker compose -f docker-compose.coordinator.yml exec postgres pg_isready -U ocr -d ocr_coordinator

# Kubernetes
kubectl exec -it $(kubectl get pod -l app.kubernetes.io/component=postgres -o name) -- \
  pg_isready -U ocr -d ocr_coordinator
```

### 2.2 Docker Compose Recovery

**Step 1: Check container status**

```bash
cd coordinator
docker compose -f docker-compose.coordinator.yml ps postgres
docker compose -f docker-compose.coordinator.yml logs --tail=100 postgres
```

**Step 2: Restart the container**

```bash
docker compose -f docker-compose.coordinator.yml restart postgres
```

**Step 3: Wait for healthy state**

```bash
# Poll until healthy (healthcheck: pg_isready -U ocr -d ocr_coordinator)
docker compose -f docker-compose.coordinator.yml exec postgres pg_isready -U ocr -d ocr_coordinator
```

**Step 4: Verify data integrity**

```bash
docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d ocr_coordinator -c "SELECT count(*) FROM jobs_job;"

docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d ocr_coordinator -c "SELECT count(*) FROM jobs_worker;"
```

**Step 5: Restart dependent services** (only if they did not reconnect automatically)

```bash
docker compose -f docker-compose.coordinator.yml restart django celery-coordinator celery-beat
```

### 2.3 Kubernetes Recovery

**Step 1: Check StatefulSet status**

```bash
RELEASE=ocr-local  # adjust to your Helm release name

kubectl get statefulset ${RELEASE}-postgres
kubectl describe pod ${RELEASE}-postgres-0
kubectl logs ${RELEASE}-postgres-0 --tail=100
```

**Step 2: Delete the pod to trigger restart** (PVC data is preserved)

```bash
kubectl delete pod ${RELEASE}-postgres-0
# StatefulSet controller recreates the pod automatically
kubectl rollout status statefulset/${RELEASE}-postgres --timeout=120s
```

**Step 3: If PVC data is corrupted, restore from backup** (see [Section 8](#8-backup-and-restore))

```bash
# Scale down to avoid writes during restore
kubectl scale statefulset ${RELEASE}-postgres --replicas=0
kubectl wait --for=delete pod/${RELEASE}-postgres-0 --timeout=60s

# Delete the corrupted PVC
kubectl delete pvc postgres-data-${RELEASE}-postgres-0

# The StatefulSet will create a fresh PVC on scale-up
kubectl scale statefulset ${RELEASE}-postgres --replicas=1
kubectl wait --for=condition=Ready pod/${RELEASE}-postgres-0 --timeout=120s

# Now restore from backup (see Section 8.3)
```

### 2.4 Connection String Update

If PostgreSQL moves to a new host (e.g., manual migration), update `DATABASE_URL`
in every service that connects to it.

**Docker Compose:** Edit `.env` and restart all services.

```bash
# .env
DATABASE_URL=postgres://ocr:<password>@<new-host>:5432/ocr_coordinator
```

```bash
docker compose -f docker-compose.coordinator.yml up -d
```

**Kubernetes:** Update the Helm secret and perform a rolling restart.

```bash
kubectl create secret generic ${RELEASE}-secret \
  --from-literal=DATABASE_URL="postgres://ocr:<password>@<new-host>:5432/ocr_coordinator" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/${RELEASE}-coordinator
kubectl rollout restart deployment/${RELEASE}-celery-coordinator
kubectl rollout restart deployment/${RELEASE}-celery-beat
```

---

## 3. RabbitMQ Failover

### 3.1 Symptoms of Failure

**Application-level indicators:**

- Celery workers log `amqp.exceptions.ConnectionForced` or `ConnectionRefusedError`
- New tasks are not dispatched (jobs stuck in "ingesting")
- Flower dashboard unreachable or shows 0 active workers

**Infrastructure-level indicators:**

- Docker: healthcheck (`rabbitmq-diagnostics -q ping`) failing
- Kubernetes: pod in `CrashLoopBackOff`, readiness probe failing
- Prometheus alert: `OCRNoGPUWorkersAvailable` or `OCRWorkersDrained` fires (workers disconnect when broker is down)

**Quick diagnosis:**

```bash
# Docker Compose (single node)
docker compose -f docker-compose.coordinator.yml exec rabbitmq rabbitmqctl status

# Docker Compose (HA overlay)
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec rabbitmq rabbitmqctl cluster_status

# Kubernetes
kubectl exec -it ${RELEASE}-rabbitmq-0 -- rabbitmqctl cluster_status
```

### 3.2 Docker Compose -- Single Node Recovery

```bash
# Restart the broker
docker compose -f docker-compose.coordinator.yml restart rabbitmq

# Verify
docker compose -f docker-compose.coordinator.yml exec rabbitmq rabbitmqctl status

# Check queues exist
docker compose -f docker-compose.coordinator.yml exec rabbitmq \
  rabbitmqctl list_queues name messages consumers
```

### 3.3 Docker Compose -- HA Overlay (3-Node Cluster)

The HA overlay (`docker-compose.ha.yml`) deploys a 3-node RabbitMQ cluster with
quorum queues enabled (`CELERY_USE_QUORUM_QUEUES=true`). Quorum queues replicate
messages across nodes using Raft consensus and tolerate the loss of any single node.

**Single node failure (automatic recovery):**

```bash
# Check which node is down
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec rabbitmq rabbitmqctl cluster_status

# Restart the failed node
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  restart rabbitmq2  # or rabbitmq3

# Verify cluster re-formation
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec rabbitmq rabbitmqctl cluster_status
```

**All nodes down (cold start):**

```bash
# Bring up the primary first
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  up -d rabbitmq

# Wait for healthy, then start replicas
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  up -d rabbitmq2 rabbitmq3

# Run the cluster init sidecar
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  up rabbitmq-cluster-init
```

**Verify quorum queue health:**

```bash
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec rabbitmq rabbitmqctl list_queues name type messages members online

# Expected: type=quorum for coordinator, ocr_gpu, cpu_general queues
```

### 3.4 Kubernetes Recovery

The default Helm chart deploys RabbitMQ as a **single-replica** StatefulSet.

```bash
# Check pod
kubectl get pod -l app.kubernetes.io/component=rabbitmq
kubectl logs ${RELEASE}-rabbitmq-0 --tail=50

# Restart
kubectl delete pod ${RELEASE}-rabbitmq-0
kubectl wait --for=condition=Ready pod/${RELEASE}-rabbitmq-0 --timeout=120s

# Verify
kubectl exec ${RELEASE}-rabbitmq-0 -- rabbitmqctl status
kubectl exec ${RELEASE}-rabbitmq-0 -- rabbitmqctl list_queues name messages consumers
```

### 3.5 Celery Broker Connection Retry

Celery is configured to automatically reconnect to the broker on startup and during
transient outages. From `coordinator/coordinator/celery.py`:

```python
app.conf.broker_connection_retry_on_startup = True
app.conf.broker_connection_max_retries = 10
app.conf.broker_transport_options = {
    'confirm_publish': True,
}
```

**What this means operationally:**

- Workers will retry broker connections up to 10 times on startup.
- If the broker is briefly unavailable, workers hold their connection and retry.
- `confirm_publish` ensures messages are not silently lost.
- After 10 failed retries, workers exit. Docker `restart: unless-stopped` or
  Kubernetes restart policy will restart them.

### 3.6 Queue Recovery and Message Durability

- **Quorum queues** (HA overlay): Messages replicated across 3 nodes. Survive
  single-node loss. If all nodes lose data, messages are lost.
- **Classic queues** (default single-node): Messages persisted to disk
  (`rabbitmq_data` volume). Survive broker restart but are lost if the volume
  is deleted.
- **`task_acks_late = True`**: Messages are only acknowledged after the task
  completes. If a worker crashes mid-task, the message is redelivered.
- **`task_reject_on_worker_lost = True`**: If a worker process is killed, the
  message is rejected and requeued, not acknowledged.

---

## 4. Redis Failover

Redis serves two roles in the pipeline:

1. **Celery result backend** (`redis://:password@redis:6379/0`) -- stores task results and chord unlock keys.
2. **Django cache** (`redis://:password@redis:6379/1`) -- caches fleet status and metrics queries.

### 4.1 Symptoms of Failure

**Application-level indicators:**

- Celery chord callbacks fail with `redis.exceptions.ConnectionError`
- Tasks complete but results are not stored; `AsyncResult.get` times out
- Django admin is slow (cache misses)
- Metrics endpoint returns stale data

**Infrastructure-level indicators:**

- Docker: healthcheck (`redis-cli -a $PASSWORD ping`) failing
- Kubernetes: pod readiness probe failing
- Celery workers log `redis.exceptions.ConnectionError: Error connecting to redis`

**Quick diagnosis:**

```bash
# Docker Compose
docker compose -f docker-compose.coordinator.yml exec redis redis-cli -a "$REDIS_PASSWORD" ping
docker compose -f docker-compose.coordinator.yml exec redis redis-cli -a "$REDIS_PASSWORD" info replication

# Kubernetes
kubectl exec -it ${RELEASE}-redis-0 -- redis-cli -a "$REDIS_PASSWORD" ping
kubectl exec -it ${RELEASE}-redis-0 -- redis-cli -a "$REDIS_PASSWORD" info replication
```

### 4.2 Docker Compose -- Single Node Recovery

```bash
docker compose -f docker-compose.coordinator.yml restart redis

# Verify
docker compose -f docker-compose.coordinator.yml exec redis redis-cli -a "$REDIS_PASSWORD" ping
```

### 4.3 Docker Compose -- HA Overlay (Sentinel)

The HA overlay deploys:
- 1 Redis master
- 1 Redis replica (`redis-replica`)
- 3 Sentinel instances (`redis-sentinel1`, `redis-sentinel2`, `redis-sentinel3`)

Sentinel configuration (`docker-compose.ha.yml`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `sentinel monitor ocr-master` | `redis 6379 2` | Monitor master at `redis:6379`, quorum of 2 sentinels to trigger failover |
| `down-after-milliseconds` | `5000` | Master considered down after 5 seconds of no response |
| `failover-timeout` | `10000` | Failover must complete within 10 seconds |
| `parallel-syncs` | `1` | Only 1 replica syncs at a time during failover |

**Automatic failover (master fails, replica promotes):**

When the master becomes unreachable for 5 seconds, Sentinel automatically:
1. Promotes `redis-replica` to master.
2. Reconfigures clients to point to the new master.

**Verify Sentinel status:**

```bash
# Check which node is master
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec redis-sentinel1 redis-cli -p 26379 sentinel master ocr-master

# Check replica status
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec redis-sentinel1 redis-cli -p 26379 sentinel replicas ocr-master

# Check all sentinels agree
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec redis-sentinel1 redis-cli -p 26379 sentinel sentinels ocr-master
```

**Manual failover (planned maintenance):**

```bash
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec redis-sentinel1 redis-cli -p 26379 sentinel failover ocr-master

# Wait a few seconds, then verify
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml \
  exec redis-sentinel1 redis-cli -p 26379 sentinel master ocr-master
```

### 4.4 Kubernetes with Sentinel

When `redis.sentinel.enabled=true` in `values.yaml`, the Helm chart deploys:
- 1 Redis master (StatefulSet, `${RELEASE}-redis`)
- N Redis replicas (StatefulSet, `${RELEASE}-redis-replica`, default 1)
- Q Sentinel instances (StatefulSet, `${RELEASE}-redis-sentinel`, default 3)

**Verify Sentinel status:**

```bash
# Get a sentinel pod
SENTINEL_POD=$(kubectl get pod -l app.kubernetes.io/component=redis-sentinel -o name | head -1)

# Check master info
kubectl exec $SENTINEL_POD -- redis-cli -p 26379 sentinel master ocr-master

# Force a manual failover
kubectl exec $SENTINEL_POD -- redis-cli -p 26379 sentinel failover ocr-master
```

**Recovery from total Redis loss:**

```bash
# Scale down sentinel and replica first
kubectl scale statefulset ${RELEASE}-redis-sentinel --replicas=0
kubectl scale statefulset ${RELEASE}-redis-replica --replicas=0

# Restart master
kubectl delete pod ${RELEASE}-redis-0
kubectl wait --for=condition=Ready pod/${RELEASE}-redis-0 --timeout=60s

# Scale replicas and sentinels back
kubectl scale statefulset ${RELEASE}-redis-replica --replicas=1
kubectl scale statefulset ${RELEASE}-redis-sentinel --replicas=3
```

### 4.5 Celery Result Backend with Sentinel

When using Sentinel, the Celery result backend URL and transport options must be
configured. From `coordinator/coordinator/settings.py`:

```python
# Set these environment variables:
# CELERY_RESULT_BACKEND=sentinel://sentinel1:26379/0
# REDIS_SENTINEL_MASTER_NAME=ocr-master
# REDIS_SENTINEL_PASSWORD=<password>  (optional)

CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS = {
    'master_name': 'ocr-master',
    'sentinel_kwargs': {
        'password': '<sentinel-password>',
    },
}
```

**Docker Compose .env for Sentinel mode:**

```bash
CELERY_RESULT_BACKEND=sentinel://:${REDIS_PASSWORD}@redis-sentinel1:26379/0
REDIS_SENTINEL_MASTER_NAME=ocr-master
REDIS_SENTINEL_PASSWORD=${REDIS_PASSWORD}
```

### 4.6 Impact of Redis Loss on Pipeline

Redis loss is **non-fatal** for basic OCR processing but degrades pipeline features:

| Feature | Impact | Recovery |
|---------|--------|----------|
| Task results | Chord callbacks fail; fan-out jobs do not assemble | Resubmit affected jobs after Redis recovery |
| Django cache | Slower metrics/admin queries | Auto-recovers on reconnect |
| Celery Beat state | Beat continues using DB scheduler (PostgreSQL), not Redis | No impact |
| In-flight tasks | Continue executing; results lost on completion | Tasks marked failed after timeout; resubmit |

---

## 5. Worker Recovery

### 5.1 GPU Worker Crash Recovery

Celery's reliability settings ensure tasks are requeued when a worker dies:

```python
# coordinator/coordinator/celery.py
app.conf.task_acks_late = True              # Ack only after task completes
app.conf.task_reject_on_worker_lost = True  # Requeue on worker death
app.conf.worker_prefetch_multiplier = 1     # Fetch 1 task at a time
app.conf.worker_max_tasks_per_child = 50    # Recycle after 50 tasks (leak protection)
```

**What happens when a GPU worker crashes:**

1. The OS kills the worker process (OOM, GPU fault, SIGKILL).
2. RabbitMQ detects the consumer disconnect.
3. The unacknowledged message is requeued (because `task_acks_late = True`).
4. Another available worker picks up the requeued task.
5. Docker `restart: unless-stopped` or Kubernetes restartPolicy restarts the worker.
6. On startup, the worker fires `worker_ready` signal and re-registers in the database.

**Manual verification after crash:**

```bash
# Docker Compose
docker compose -f docker-compose.worker.yml logs --tail=50 ocr-worker
docker compose -f docker-compose.coordinator.yml exec django \
  python manage.py fleet_status

# Kubernetes
kubectl get pods -l app.kubernetes.io/component=gpu-worker
kubectl logs <crashed-pod> --previous  # check crash reason
```

### 5.2 Worker Drain Procedure (Planned Maintenance)

To gracefully remove a worker from the pool without losing in-flight tasks:

**Option A: Celery remote control**

```bash
# Send warm shutdown (finish current tasks, then exit)
docker compose -f docker-compose.coordinator.yml exec celery-coordinator \
  celery -A coordinator control shutdown --destination worker@<hostname>

# Or from any container with broker access:
celery -A coordinator control shutdown --destination worker@<hostname>
```

**Option B: Django admin action**

1. Open Django admin at `http://<coordinator>:8000/admin/jobs/worker/`
2. Select the worker(s).
3. Choose "Drain worker" from the action dropdown.
4. Worker status changes to `DRAINING`. It finishes current tasks and stops accepting new ones.

**Option C: Kubernetes pod eviction**

```bash
# Cordon the node first (prevent new pods)
kubectl cordon <node-name>

# Drain with grace period matching max task duration
kubectl drain <node-name> --grace-period=600 --delete-emptydir-data
```

### 5.3 Fleet Status Check

```bash
# Docker Compose
docker compose -f docker-compose.coordinator.yml exec django \
  python manage.py fleet_status

# Kubernetes
kubectl exec -it $(kubectl get pod -l app.kubernetes.io/component=coordinator -o name | head -1) -- \
  python manage.py fleet_status
```

Expected output:

```
============================================================
  Fleet Status
============================================================

  Workers: 5 total
    Online:   3
    Busy:     2
    Draining: 0
    Offline:  0

  Hostname                       Status     GPU    Last Heartbeat
  ------------------------------ ---------- ------ --------------------
  worker@gpu-node-1              busy       Yes    12s ago
  worker@gpu-node-2              online     Yes    8s ago
  worker@cpu-node-1              online     No     5s ago
  ...

  Jobs:
    Processing: 3
    Completed:  147
    Total:      150

============================================================
```

### 5.4 Worker Re-Registration on Restart

Workers automatically re-register via the `worker_ready` Celery signal
(`coordinator/jobs/signals.py`). The signal handler:

1. Detects GPU availability, model, and VRAM.
2. Reads queue assignments from the consumer or `WORKER_QUEUES` env var.
3. Creates or updates the `Worker` database row with status `ONLINE`.
4. Logs recommended concurrency based on detected VRAM.

No manual intervention is required for worker re-registration.

### 5.5 Scaling Workers

**Docker Compose:**

```bash
# Scale to 5 workers
docker compose -f docker-compose.coordinator.yml -f docker-compose.scale-test.yml \
  up -d --scale ocr-worker=5
```

**Kubernetes (manual):**

```bash
kubectl scale deployment ${RELEASE}-gpu-worker --replicas=5
```

**Kubernetes (KEDA autoscaler, if enabled):**

Set `gpuWorker.autoscaling.enabled=true` in `values.yaml`. KEDA scales based on
RabbitMQ queue depth (`queueTarget: 5` messages per worker).

---

## 6. Full Cluster Recovery

Use this procedure when the entire stack is down (power outage, node failure,
disaster recovery).

### 6.1 Recovery Priority Order

Start services in strict dependency order:

```
1. PostgreSQL    (no dependencies -- all other services need it)
2. RabbitMQ      (no dependencies -- Celery workers need it)
3. Redis         (no dependencies -- result backend + cache)
4. Coordinator   (needs: PostgreSQL, RabbitMQ, Redis)
   - Django (runs migrations)
   - Celery Beat
   - Celery Coordinator Worker
   - Flower
5. GPU Workers   (needs: RabbitMQ, Redis, PostgreSQL, shared storage)
6. CPU Workers   (needs: RabbitMQ, Redis, PostgreSQL, shared storage)
```

### 6.2 Docker Compose Full Recovery

```bash
cd coordinator

# Step 1: Start infrastructure
docker compose -f docker-compose.coordinator.yml up -d postgres
docker compose -f docker-compose.coordinator.yml exec postgres pg_isready -U ocr -d ocr_coordinator
# Retry until "accepting connections"

docker compose -f docker-compose.coordinator.yml up -d rabbitmq
docker compose -f docker-compose.coordinator.yml exec rabbitmq rabbitmqctl status
# Wait for "running"

docker compose -f docker-compose.coordinator.yml up -d redis
docker compose -f docker-compose.coordinator.yml exec redis redis-cli -a "$REDIS_PASSWORD" ping
# Wait for "PONG"

# Step 2: Start coordinator services
docker compose -f docker-compose.coordinator.yml up -d django celery-coordinator celery-beat flower
# Django runs migrations automatically on startup

# Step 3: Start workers (on the same or remote VPS)
docker compose -f docker-compose.worker.yml up -d

# Step 4: Verify
docker compose -f docker-compose.coordinator.yml exec django python manage.py fleet_status
```

**With HA overlay:**

```bash
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml up -d postgres
# Wait for healthy...

docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml up -d rabbitmq rabbitmq2 rabbitmq3
docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml up rabbitmq-cluster-init
# Wait for "cluster formed successfully"

docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml up -d redis redis-replica redis-sentinel1 redis-sentinel2 redis-sentinel3

docker compose -f docker-compose.coordinator.yml -f docker-compose.ha.yml up -d django celery-coordinator celery-beat flower
```

### 6.3 Kubernetes Full Recovery

```bash
RELEASE=ocr-local
NAMESPACE=ocr

# Install or upgrade from Helm chart
helm upgrade --install $RELEASE helm/ocr-local/ \
  --namespace $NAMESPACE --create-namespace \
  -f values-production.yaml \
  -f values-secret.yaml

# Monitor rollout
kubectl -n $NAMESPACE rollout status statefulset/${RELEASE}-postgres --timeout=180s
kubectl -n $NAMESPACE rollout status statefulset/${RELEASE}-rabbitmq --timeout=180s
kubectl -n $NAMESPACE rollout status statefulset/${RELEASE}-redis --timeout=120s
kubectl -n $NAMESPACE rollout status deployment/${RELEASE}-coordinator --timeout=120s
kubectl -n $NAMESPACE rollout status deployment/${RELEASE}-gpu-worker --timeout=300s
```

### 6.4 Post-Recovery Validation Checklist

Run these checks after any recovery to confirm pipeline integrity.

```bash
# 1. Database connectivity
kubectl exec ${RELEASE}-postgres-0 -- pg_isready -U ocr

# 2. RabbitMQ queues exist with correct type
kubectl exec ${RELEASE}-rabbitmq-0 -- rabbitmqctl list_queues name type messages

# 3. Redis responds
kubectl exec ${RELEASE}-redis-0 -- redis-cli -a "$REDIS_PASSWORD" ping

# 4. Coordinator health endpoint
curl -s http://<coordinator>:8000/api/v1/health/ | python -m json.tool

# 5. Worker fleet status
kubectl exec $(kubectl get pod -l app.kubernetes.io/component=coordinator -o name | head -1) -- \
  python manage.py fleet_status

# 6. Metrics endpoint (with auth if METRICS_API_KEY is set)
curl -s -H "X-Api-Key: <key>" http://<coordinator>:8000/api/v1/metrics/ | python -m json.tool

# 7. Submit a test job and verify completion
curl -X POST http://<coordinator>:8000/api/v1/jobs/ \
  -H "X-API-Key: <key>" \
  -F "file=@test-document.pdf"

# 8. Check for stuck jobs from before the outage
kubectl exec $(kubectl get pod -l app.kubernetes.io/component=coordinator -o name | head -1) -- \
  python manage.py shell -c "
from jobs.models import Job
stuck = Job.objects.filter(status__in=['processing', 'assembling', 'ingesting'])
for j in stuck:
    print(f'{j.job_id}: {j.status} ({j.pages_completed}/{j.total_pages})')
"

# 9. Resubmit stuck jobs if needed
# Jobs with task_reject_on_worker_lost should auto-requeue, but
# verify after 5 minutes and manually resubmit if still stuck.
```

---

## 7. Monitoring and Alerting

### 7.1 Prometheus Alerts

The Helm chart includes a `PrometheusRule` resource (enabled with
`prometheus.rules.enabled=true`). These are the default alerts:

| Alert | Expression | Severity | For | Meaning |
|-------|-----------|----------|-----|---------|
| `OCRHighJobErrorRate` | `ocr_job_error_rate_1h > 0.1` | warning | 5m | More than 10% of jobs failing in the last hour |
| `OCRNoGPUWorkersAvailable` | `ocr_gpu_workers_available == 0` | critical | 2m | Zero GPU workers online -- OCR processing halted |
| `OCRWorkersDrained` | `ocr_workers_total{status="online"} + ocr_workers_total{status="busy"} == 0` | critical | 5m | All workers offline -- pipeline cannot process |
| `OCRSlowPageProcessing` | `ocr_page_processing_time_avg_ms > 30000` | warning | 10m | Average page OCR taking over 30 seconds |
| `OCRJobsStuck` | `increase(ocr_pages_processed_total[30m]) == 0` with processing jobs | warning | 30m | Jobs in processing but no pages completed |

### 7.2 Metrics Endpoint

The coordinator exposes two metrics endpoints:

- **JSON**: `GET /api/v1/metrics/` -- human-readable, used by `capture_phase7c_metrics.py`
- **Prometheus**: `GET /api/v1/prometheus/` -- scraped by ServiceMonitor

Authentication: If `METRICS_API_KEY` is set, both endpoints require either
`X-Api-Key: <key>` or `Authorization: Bearer <key>` header.

```bash
# Quick health check
curl -s -H "X-Api-Key: $METRICS_API_KEY" http://<coordinator>:8000/api/v1/metrics/ | python -m json.tool
```

### 7.3 Key Metrics to Watch

| Metric | Source | Warning Threshold | Critical Threshold |
|--------|--------|-------------------|-------------------|
| `ocr_gpu_workers_available` | Prometheus | < 2 | 0 |
| `ocr_jobs_total{status="failed"}` | Prometheus | Increasing trend | > 10% of total |
| `ocr_page_processing_time_avg_ms` | Prometheus | > 15000 | > 30000 |
| RabbitMQ queue depth (coordinator) | `rabbitmqctl list_queues` | > 100 | > 500 |
| RabbitMQ queue depth (ocr_gpu) | `rabbitmqctl list_queues` | > 50 | > 200 |
| PostgreSQL connection count | `pg_stat_activity` | > 150 (of max 200) | > 190 |
| Redis memory usage | `redis-cli info memory` | > 70% of maxmemory | > 90% |
| Worker heartbeat age | `fleet_status` command | > 120s | > 300s |

### 7.4 Grafana Dashboard Panels

When `prometheus.grafana.enabled=true`, the Helm chart deploys a ConfigMap with
a Grafana dashboard. Recommended panels:

1. **Pipeline Throughput**: Pages processed per minute (rate of `ocr_pages_processed_total`)
2. **Job Status Distribution**: Stacked bar of jobs by status
3. **Worker Fleet**: Gauge of workers by status (online / busy / draining / offline)
4. **GPU Queue Depth**: RabbitMQ `ocr_gpu` queue messages
5. **Page Processing Latency**: Histogram of `ocr_page_processing_time_avg_ms`
6. **Error Rate**: Percentage of failed jobs over time
7. **Infrastructure Health**: PostgreSQL connections, Redis memory, RabbitMQ memory

### 7.5 Manual Monitoring Commands

When Prometheus is not available, use these commands for spot checks:

```bash
# RabbitMQ queue depths
docker compose -f docker-compose.coordinator.yml exec rabbitmq \
  rabbitmqctl list_queues name messages consumers

# Redis memory
docker compose -f docker-compose.coordinator.yml exec redis \
  redis-cli -a "$REDIS_PASSWORD" info memory | grep used_memory_human

# PostgreSQL active connections
docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d ocr_coordinator -c "SELECT count(*) FROM pg_stat_activity WHERE state='active';"

# Celery active tasks (via Flower API)
curl -u admin:$FLOWER_PASSWORD http://<coordinator>:5555/api/workers
```

---

## 8. Backup and Restore

### 8.1 Kubernetes Backup CronJob

The Helm chart includes a PostgreSQL backup CronJob (enabled with
`postgresql.backup.enabled=true`).

**Configuration** (`values.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `postgresql.backup.schedule` | `"0 2 * * *"` | Cron schedule (daily at 2 AM UTC) |
| `postgresql.backup.retentionCount` | `7` | Number of backups to retain |
| `postgresql.backup.storage` | `20Gi` | PVC size for backup storage |
| `postgresql.backup.timeoutSeconds` | `3600` | Max backup duration (1 hour) |
| `postgresql.backup.historyLimit` | `3` | Completed CronJob history |

**What the CronJob does:**

1. Runs `pg_dump` in custom format with compression level 6.
2. Writes to `/backups/ocr-coordinator-<TIMESTAMP>.sql.gz`.
3. Prunes backups older than the retention count.
4. Backup PVC: `${RELEASE}-postgres-backup` (ReadWriteOnce).

### 8.2 Manual Backup Trigger

**Kubernetes:**

```bash
# Trigger a one-off backup job from the CronJob
kubectl create job --from=cronjob/${RELEASE}-postgres-backup ${RELEASE}-manual-backup-$(date +%s)

# Monitor
kubectl get jobs | grep manual-backup
kubectl logs job/${RELEASE}-manual-backup-<timestamp>
```

**Docker Compose:**

```bash
# Run pg_dump directly
docker compose -f docker-compose.coordinator.yml exec postgres \
  pg_dump -U ocr -d ocr_coordinator --format=custom --compress=6 \
  -f /var/lib/postgresql/data/backup-$(date +%Y%m%d-%H%M%S).sql.gz

# Copy backup out of the container
docker cp $(docker compose -f docker-compose.coordinator.yml ps -q postgres):/var/lib/postgresql/data/backup-*.sql.gz ./backups/
```

### 8.3 Restore Procedure

**Kubernetes restore from backup PVC:**

```bash
# Step 1: Scale down all services that write to PostgreSQL
kubectl scale deployment ${RELEASE}-coordinator --replicas=0
kubectl scale deployment ${RELEASE}-celery-coordinator --replicas=0
kubectl scale deployment ${RELEASE}-celery-beat --replicas=0
kubectl scale deployment ${RELEASE}-gpu-worker --replicas=0
kubectl scale deployment ${RELEASE}-cpu-worker --replicas=0

# Step 2: Identify the backup to restore
kubectl run backup-inspector --rm -it --restart=Never \
  --image=postgres:16-alpine \
  --overrides='{"spec":{"containers":[{"name":"inspector","image":"postgres:16-alpine","command":["ls","-lhrt","/backups/"],"volumeMounts":[{"name":"backup","mountPath":"/backups"}]}],"volumes":[{"name":"backup","persistentVolumeClaim":{"claimName":"'"${RELEASE}"'-postgres-backup"}}]}}'

# Step 3: Drop and recreate the database
kubectl exec ${RELEASE}-postgres-0 -- psql -U ocr -d postgres -c "
  SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'ocr_coordinator';
"
kubectl exec ${RELEASE}-postgres-0 -- psql -U ocr -d postgres -c "DROP DATABASE IF EXISTS ocr_coordinator;"
kubectl exec ${RELEASE}-postgres-0 -- psql -U ocr -d postgres -c "CREATE DATABASE ocr_coordinator OWNER ocr;"

# Step 4: Restore from backup
kubectl run pg-restore --rm -it --restart=Never \
  --image=postgres:16-alpine \
  --env="PGPASSWORD=<password>" \
  --overrides='{"spec":{"containers":[{"name":"restore","image":"postgres:16-alpine","command":["pg_restore","--host='"${RELEASE}"'-postgres","--username=ocr","--dbname=ocr_coordinator","--no-owner","--verbose","/backups/ocr-coordinator-YYYYMMDD-HHMMSS.sql.gz"],"volumeMounts":[{"name":"backup","mountPath":"/backups"}]}],"volumes":[{"name":"backup","persistentVolumeClaim":{"claimName":"'"${RELEASE}"'-postgres-backup"}}]}}'

# Step 5: Run Django migrations (in case backup predates schema changes)
kubectl exec $(kubectl get pod -l app.kubernetes.io/component=coordinator -o name | head -1) -- \
  python manage.py migrate --noinput

# Step 6: Scale services back up
kubectl scale deployment ${RELEASE}-coordinator --replicas=2
kubectl scale deployment ${RELEASE}-celery-coordinator --replicas=1
kubectl scale deployment ${RELEASE}-celery-beat --replicas=1  # spec is Deployment in Helm, keep replicas=1
kubectl scale deployment ${RELEASE}-gpu-worker --replicas=2
kubectl scale deployment ${RELEASE}-cpu-worker --replicas=1
```

**Docker Compose restore:**

```bash
# Step 1: Stop all services except PostgreSQL
docker compose -f docker-compose.coordinator.yml stop django celery-coordinator celery-beat flower

# Step 2: Copy backup into container
docker cp ./backups/ocr-coordinator-YYYYMMDD-HHMMSS.sql.gz \
  $(docker compose -f docker-compose.coordinator.yml ps -q postgres):/tmp/restore.sql.gz

# Step 3: Drop and recreate database
docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d postgres -c "
    SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'ocr_coordinator';
  "
docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d postgres -c "DROP DATABASE IF EXISTS ocr_coordinator;"
docker compose -f docker-compose.coordinator.yml exec postgres \
  psql -U ocr -d postgres -c "CREATE DATABASE ocr_coordinator OWNER ocr;"

# Step 4: Restore
docker compose -f docker-compose.coordinator.yml exec postgres \
  pg_restore -U ocr -d ocr_coordinator --no-owner --verbose /tmp/restore.sql.gz

# Step 5: Restart services
docker compose -f docker-compose.coordinator.yml up -d
```

### 8.4 Data Verification After Restore

```bash
# Check row counts for critical tables
psql -U ocr -d ocr_coordinator -c "
  SELECT 'jobs' AS table_name, count(*) FROM jobs_job
  UNION ALL
  SELECT 'workers', count(*) FROM jobs_worker
  UNION ALL
  SELECT 'page_results', count(*) FROM jobs_pageresult
  UNION ALL
  SELECT 'custody_events', count(*) FROM jobs_custodyevent;
"

# Verify no orphaned jobs (processing but no workers assigned)
psql -U ocr -d ocr_coordinator -c "
  SELECT job_id, status, pages_completed, total_pages
  FROM jobs_job
  WHERE status IN ('processing', 'assembling')
  ORDER BY created_at DESC
  LIMIT 10;
"
```

### 8.5 Shared Storage (NFS/S3) Backup Considerations

The backup CronJob only covers the PostgreSQL database. Document files on shared
storage (NFS or S3) require a separate backup strategy:

- **NFS**: Use filesystem-level snapshots (ZFS, LVM) or `rsync` to a backup volume.
- **S3**: Enable bucket versioning and cross-region replication at the object store level.

The OCR output directory (`ocr_output/`) and temp directory (`ocr_temp/`) can be
regenerated by reprocessing source documents. The database is the authoritative
record of job status and page results.

---

## Appendix A: Environment Variable Reference

These environment variables control failover-related behavior:

| Variable | Default | Used By | Purpose |
|----------|---------|---------|---------|
| `DATABASE_URL` | (required) | Coordinator, Workers | PostgreSQL connection string |
| `CELERY_BROKER_URL` | (required) | Coordinator, Workers | RabbitMQ AMQP connection |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | Coordinator, Workers | Redis result backend |
| `REDIS_URL` | `redis://redis:6379/1` | Django | Cache backend |
| `CELERY_USE_QUORUM_QUEUES` | `false` | Coordinator | Enable RabbitMQ quorum queues |
| `REDIS_SENTINEL_MASTER_NAME` | (empty) | Coordinator | Enable Sentinel mode for result backend |
| `REDIS_SENTINEL_PASSWORD` | (empty) | Coordinator | Sentinel auth password |
| `METRICS_API_KEY` | (empty) | Coordinator | Auth key for metrics endpoints |

## Appendix B: Quick Reference Commands

```bash
# === HEALTH CHECKS ===
# PostgreSQL
pg_isready -U ocr -d ocr_coordinator

# RabbitMQ
rabbitmqctl status
rabbitmqctl cluster_status        # HA only
rabbitmqctl list_queues name type messages consumers

# Redis
redis-cli -a "$REDIS_PASSWORD" ping
redis-cli -a "$REDIS_PASSWORD" info replication
redis-cli -p 26379 sentinel master ocr-master    # Sentinel only

# Fleet
python manage.py fleet_status

# === OPERATIONAL ===
# Drain a worker
celery -A coordinator control shutdown --destination worker@<hostname>

# Manual backup
pg_dump -U ocr -d ocr_coordinator --format=custom --compress=6 -f /backups/backup.sql.gz

# Restore
pg_restore -U ocr -d ocr_coordinator --no-owner --verbose /backups/backup.sql.gz

# Force Redis Sentinel failover
redis-cli -p 26379 sentinel failover ocr-master

# Check stuck jobs
python manage.py shell -c "from jobs.models import Job; print(Job.objects.filter(status='processing').count)"
```
