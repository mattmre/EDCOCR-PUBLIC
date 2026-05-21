# Cloud-Native Validation Guide -- EDCOCR on EKS, GKE, and OKE

**Version**: 1.2.0 | **Last Updated**: 2026-05-20

This guide covers the end-to-end process for validating a Kubernetes deployment of
EDCOCR across Amazon EKS, Google GKE, and Oracle OKE. It complements the
[Production Cutover Runbook](production-cutover-runbook.md) (which covers credential
rotation, storage migration, and canary deployment) by focusing on cloud-specific
cluster provisioning, Helm install validation, KEDA autoscaler verification, network
policy enforcement, PDB behavior, and a structured smoke test suite.

**Prerequisites**: Familiarity with the production cutover runbook, the Helm chart
(`helm/ocr-local/`), and the Terraform modules (`terraform/modules/`).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pre-flight Checklist](#2-pre-flight-checklist)
3. [Cloud-Specific Setup](#3-cloud-specific-setup)
4. [Helm Deploy Sequence](#4-helm-deploy-sequence)
5. [KEDA Autoscaler Validation](#5-keda-autoscaler-validation)
6. [Network Policy Validation](#6-network-policy-validation)
7. [PodDisruptionBudget Validation](#7-poddisruptionbudget-validation)
8. [Smoke Test Suite](#8-smoke-test-suite)
9. [Rollback Procedure](#9-rollback-procedure)

---

## 1. Overview

EDCOCR ships a production Helm chart (`helm/ocr-local/`, chart version 0.4.0,
appVersion 1.0.0) with 33 Kubernetes resource templates covering:

- **Coordinator** (Django + Gunicorn) with configurable replicas
- **Celery coordinator worker** (job lifecycle queue)
- **Celery Beat** (periodic tasks)
- **GPU OCR workers** with `nvidia.com/gpu` resource requests
- **CPU OCR workers** (ONNX Runtime backend, opt-in)
- **CPU-only workers** (compression, NER, postprocessing)
- **Layout CPU workers** (barcode/QR, OMR, opt-in)
- **NLP GPU workers** (VLM inference, opt-in)
- **LayoutLMv3 workers** (document understanding, opt-in)
- **Flower** (Celery monitoring dashboard, opt-in)
- **PostgreSQL**, **RabbitMQ**, and **Redis** StatefulSets
- **Redis Sentinel** (opt-in HA)
- **PostgreSQL backup CronJob** (opt-in)
- **KEDA ScaledObjects** for GPU, CPU, CPU-OCR, layout, NLP, and LayoutLM workers
- **NetworkPolicies** restricting worker and coordinator traffic
- **PodDisruptionBudgets** for GPU workers, coordinator, PostgreSQL, RabbitMQ, Redis, and Celery Beat
- **Ingress** with cert-manager TLS (opt-in)
- **Prometheus ServiceMonitor**, **PrometheusRules** (9 alerts + recording rules), and **Grafana dashboard ConfigMap** (opt-in)
- **Inter-service TLS certificates** via cert-manager (opt-in)

Terraform modules under `terraform/modules/` provide cluster provisioning for EKS,
GKE, and OKE, with shared modules for KEDA and Prometheus/Grafana. Environment
configurations under `terraform/environments/` offer staging and production presets.

This guide assumes the cluster already exists (provisioned via Terraform or manually)
and focuses on validating the Helm deployment inside it.

---

## 2. Pre-flight Checklist

Complete every item before running `helm install`. Failure on any item will produce a
non-functional deployment.

### 2.1 Helm Chart Lint

```bash
helm lint helm/ocr-local/
```

Expected: `1 chart(s) linted, 0 chart(s) failed`. Fix any errors before proceeding.

### 2.2 Required Secrets

The chart's `secret.yaml` template uses `required` for critical values. The following
secrets **must** be set via `--set` flags or a `values-secret.yaml` file at install
time. Empty defaults will cause `helm install` to fail (this is intentional).

| Secret Key | Description |
|---|---|
| `secrets.djangoSecretKey` | Django signing key (min 50 chars, cryptographically random) |
| `secrets.postgresPassword` | PostgreSQL password for the `ocr` user |
| `secrets.rabbitmqPassword` | RabbitMQ password |
| `secrets.redisPassword` | Redis AUTH password |
| `secrets.flowerPassword` | Flower dashboard password (required when `flower.enabled=true`) |

**Conditionally required** (when `storage.backend=s3`):

| Secret Key | Description |
|---|---|
| `secrets.s3Endpoint` | S3-compatible endpoint URL |
| `secrets.s3Bucket` | Bucket name |
| `secrets.s3AccessKey` | Access key |
| `secrets.s3SecretKey` | Secret key |

**Auto-generated if empty** (safe defaults derived from release name):

- `secrets.databaseUrl` -- defaults to `postgres://ocr:<pgPassword>@<release>-postgres:5432/ocr`
- `secrets.celeryBrokerUrl` -- defaults to `amqp://ocr:<rmqPassword>@<release>-rabbitmq:5672//`
- `secrets.redisUrl` -- defaults to `redis://:<redisPassword>@<release>-redis:6379/0`
- `secrets.celeryResultBackend` -- defaults to `redis://:<redisPassword>@<release>-redis:6379/1`

Prepare a `values-secret.yaml` (excluded from version control):

```yaml
secrets:
  djangoSecretKey: "<random-50+-char-string>"
  postgresPassword: "<strong-password>"
  rabbitmqPassword: "<strong-password>"
  redisPassword: "<strong-password>"
  flowerPassword: "<strong-password>"
  metricsApiKey: "<optional-metrics-auth-key>"
```

### 2.3 NVIDIA Device Plugin

GPU workers request `nvidia.com/gpu: 1` via their resource spec. The NVIDIA device
plugin DaemonSet must be installed on all GPU-capable nodes.

```bash
# Verify the device plugin is running
kubectl get daemonset -n kube-system | grep nvidia

# Verify GPU resources are allocatable
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"
```

If no GPUs appear, install the plugin:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
```

On GKE, the plugin is installed automatically when a GPU node pool is created.
On EKS, use the EKS-optimized AMI with GPU support, which includes the plugin.
On OKE, install the plugin manually via the manifest above.

### 2.4 Storage Classes

StatefulSets (PostgreSQL, RabbitMQ, Redis) require a storage class for PVC
provisioning. The chart uses `storageClass: ""` by default, which selects the
cluster's default storage class.

Verify a default storage class exists:

```bash
kubectl get storageclass
```

At least one class should be marked `(default)`. If not, set `storageClass` explicitly
in your values override for each StatefulSet. See Section 3 for cloud-specific
recommendations.

### 2.5 Ingress Controller (Optional)

Required only if `ingress.enabled=true`. The chart defaults to `className: nginx`.

```bash
# Verify the ingress controller is running
kubectl get pods -n ingress-nginx
kubectl get ingressclass
```

If using a different ingress controller, override `ingress.className` accordingly.

### 2.6 cert-manager (Optional)

Required when `ingress.tls=true` (for automated TLS certificates) or when
`tls.enabled=true` (for inter-service mTLS certificates).

```bash
# Verify cert-manager is running
kubectl get pods -n cert-manager

# Verify the ClusterIssuer exists
kubectl get clusterissuer letsencrypt-prod
```

### 2.7 KEDA (Optional)

Required when any worker autoscaling is enabled. See Section 5 for install instructions.

---

## 3. Cloud-Specific Setup

### 3.1 Amazon EKS

**Terraform module**: `terraform/modules/eks/`

**Recommended instance types** (from Terraform defaults):

| Node Pool | Instance Type | Notes |
|---|---|---|
| CPU | `m6i.2xlarge` (8 vCPU, 32 GiB) | Coordinator, CPU workers, infra |
| GPU | `g5.xlarge` (4 vCPU, 16 GiB, 1x A10G) | GPU OCR workers |

Alternative GPU instances: `g4dn.xlarge` (T4, lower cost), `p3.2xlarge` (V100, higher
throughput).

**Storage class**: `gp3` is the default on EKS 1.23+. Verify:

```bash
kubectl get storageclass gp3
```

If not present, create it:

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
EOF
```

**NVIDIA device plugin**: Included in the EKS-optimized GPU AMI. Verify via:

```bash
kubectl get ds -n kube-system nvidia-device-plugin-daemonset
```

**IAM Roles for Service Accounts (IRSA)**: Required for S3 access without embedding
credentials in pods. Create an IAM role with S3 permissions and annotate the service
account:

```bash
# Create the OIDC provider (one-time per cluster)
eksctl utils associate-iam-oidc-provider \
  --cluster ocr-local-production \
  --approve

# Create the IAM service account
eksctl create iamserviceaccount \
  --name ocr-local-sa \
  --namespace ocr \
  --cluster ocr-local-production \
  --attach-policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/OCRLocalS3Access \
  --approve
```

When using IRSA, you can leave `secrets.s3AccessKey` and `secrets.s3SecretKey` empty
and configure the S3 backend to use the instance profile instead.

**Key Helm values override** (`values-eks.yaml`):

```yaml
cloudProvider: "aws"

coordinator:
  env:
    DJANGO_ALLOWED_HOSTS: "ocr.example.com"

gpuWorker:
  nodeSelector:
    node.kubernetes.io/instance-type: g5.xlarge
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule

storage:
  backend: s3

ingress:
  enabled: true
  className: alb    # or nginx
  host: ocr.example.com
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
```

**Cluster provisioning via Terraform**:

```bash
cd terraform/environments/production
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: set cloud_provider = "aws", aws_region, cluster_name

terraform init
terraform plan -out=plan.tfplan
terraform apply plan.tfplan

# Configure kubectl
$(terraform output -raw kubeconfig_command)
```

---

### 3.2 Google GKE

**Terraform module**: `terraform/modules/gke/`

**Recommended machine types** (from Terraform defaults):

| Node Pool | Machine Type | Notes |
|---|---|---|
| CPU | `e2-standard-8` (8 vCPU, 32 GiB) | Coordinator, CPU workers, infra |
| GPU | `n1-standard-8` (8 vCPU, 30 GiB) + T4 accelerator | GPU OCR workers |

Alternative GPUs: `nvidia-l4` (L4, newer generation), `nvidia-tesla-v100` (V100).

**Storage class**: `standard-rwo` (SSD-backed) is the default on GKE Autopilot and
standard clusters. For premium performance, use `premium-rwo`.

```bash
kubectl get storageclass
```

**NVIDIA device plugin**: Automatically installed by GKE when a GPU node pool is
created. No manual action needed. Verify GPU availability:

```bash
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-tesla-t4
kubectl describe node <gpu-node> | grep nvidia.com/gpu
```

**Workload Identity**: Preferred method for GCS access. Bind a Kubernetes service
account to a GCP service account:

```bash
# Enable Workload Identity on the cluster (if not already)
gcloud container clusters update ocr-local-production \
  --workload-pool=<PROJECT_ID>.svc.id.goog

# Create a GCP service account
gcloud iam service-accounts create ocr-local-sa \
  --display-name="OCR-Local workload identity"

# Grant GCS access
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:ocr-local-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Bind to Kubernetes service account
gcloud iam service-accounts add-iam-policy-binding \
  ocr-local-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:<PROJECT_ID>.svc.id.goog[ocr/default]"
```

**Key Helm values override** (`values-gke.yaml`):

```yaml
cloudProvider: "gcp"

coordinator:
  env:
    DJANGO_ALLOWED_HOSTS: "ocr.example.com"

gpuWorker:
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-t4
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule

ingress:
  enabled: true
  className: gce    # GKE default ingress class
  host: ocr.example.com
```

**Cluster provisioning via Terraform**:

```bash
cd terraform/environments/production
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: set cloud_provider = "gcp", gcp_project_id, gcp_region

terraform init
terraform plan -out=plan.tfplan
terraform apply plan.tfplan

$(terraform output -raw kubeconfig_command)
```

---

### 3.3 Oracle OKE

**Terraform module**: `terraform/modules/oke/`

**Recommended shapes** (from Terraform defaults):

| Node Pool | Shape | Notes |
|---|---|---|
| CPU | `VM.Standard.E4.Flex` (configurable OCPUs/memory) | Coordinator, CPU workers, infra |
| GPU | `VM.GPU.A10.1` (1x A10) | GPU OCR workers |

Alternative GPU shapes: `BM.GPU2.2` (2x P100, bare metal), `VM.GPU3.1` (1x V100).

**Storage class**: `oci-bv` (OCI Block Volume) is the default CSI storage class on OKE.

```bash
kubectl get storageclass oci-bv
```

**NVIDIA device plugin**: Must be installed manually on OKE:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
```

**Instance Principal**: Preferred method for OCI Object Storage access. Attach a
dynamic group to GPU/CPU node pools and grant `objectstorage:*` permissions:

```
# Dynamic group rule (match by compartment)
Any {instance.compartment.id = '<COMPARTMENT_OCID>'}

# IAM policy
Allow dynamic-group ocr-local-nodes to manage objects in compartment <COMPARTMENT_NAME>
```

**Key Helm values override** (`values-oke.yaml`):

```yaml
cloudProvider: "oracle"

coordinator:
  env:
    DJANGO_ALLOWED_HOSTS: "ocr.example.com"

gpuWorker:
  nodeSelector:
    oci.oraclecloud.com/gpu: "true"
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule

storage:
  backend: s3    # OCI Object Storage is S3-compatible

secrets:
  s3Endpoint: "https://<namespace>.compat.objectstorage.<region>.oraclecloud.com"
  s3Bucket: "ocr-local"
  s3Region: "<region>"
```

**Cluster provisioning via Terraform**:

```bash
cd terraform/environments/production
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: set cloud_provider = "oracle", oci_compartment_id, oci_region

terraform init
terraform plan -out=plan.tfplan
terraform apply plan.tfplan

$(terraform output -raw kubeconfig_command)
```

---

## 4. Helm Deploy Sequence

### 4.1 Install KEDA (if using autoscaling)

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --version 2.14.0
```

Verify:

```bash
kubectl get pods -n keda
# Expect: keda-operator, keda-operator-metrics-apiserver, keda-admission-webhooks
```

### 4.2 Dry Run

Always perform a dry run before applying to the cluster. This catches template
rendering errors and missing required values.

```bash
helm install ocr-local helm/ocr-local/ \
  --namespace ocr \
  --create-namespace \
  --values values-secret.yaml \
  --values values-<cloud>.yaml \
  --dry-run --debug
```

Review the rendered manifests. Pay special attention to:

- The `Secret` resource: all required values must be non-empty
- GPU worker `resources.requests`: must include `nvidia.com/gpu: 1`
- StatefulSet `storageClassName`: must match an available storage class
- `DJANGO_ALLOWED_HOSTS`: must not be empty in production

### 4.3 Install

```bash
helm install ocr-local helm/ocr-local/ \
  --namespace ocr \
  --create-namespace \
  --values values-secret.yaml \
  --values values-<cloud>.yaml \
  --wait \
  --timeout 10m
```

The `--wait` flag blocks until all Deployments, StatefulSets, and Jobs are ready.
GPU workers may take several minutes due to the `startupProbe` that imports PaddlePaddle
(up to 5 minutes: `failureThreshold=30 * periodSeconds=10`).

### 4.4 Verify StatefulSets

```bash
kubectl get statefulset -n ocr

# Expected output (all replicas ready):
# NAME                    READY   AGE
# ocr-local-postgres      1/1     2m
# ocr-local-rabbitmq      1/1     2m
# ocr-local-redis         1/1     2m
```

If any StatefulSet is stuck at `0/1`, check PVC binding:

```bash
kubectl get pvc -n ocr
kubectl describe pvc <pvc-name> -n ocr
```

Common issues: missing storage class, insufficient capacity, zone mismatch.

### 4.5 Verify Deployments

```bash
kubectl get deployments -n ocr

# Expected output:
# NAME                           READY   AGE
# ocr-local-coordinator          2/2     3m
# ocr-local-celery-coordinator   1/1     3m
# ocr-local-celery-beat          1/1     3m
# ocr-local-gpu-worker           2/2     5m
# ocr-local-cpu-worker           1/1     3m
# ocr-local-flower               1/1     3m    (if flower.enabled=true)
```

If GPU workers are stuck in `Pending`:

```bash
kubectl describe pod -n ocr -l app.kubernetes.io/component=gpu-worker
```

Common issues: no GPU nodes available, insufficient `nvidia.com/gpu` resources,
node selector mismatch, missing tolerations.

### 4.6 Verify Coordinator Health

```bash
# Port-forward to the coordinator service
kubectl port-forward svc/ocr-local-coordinator 8000:8000 -n ocr &

# Health check
curl -s http://localhost:8000/api/v1/health/ | python -m json.tool

# Expected: {"status": "healthy", ...}
```

### 4.7 Run Django Migrations (Automatic)

Migrations run automatically on coordinator startup via the Gunicorn entrypoint.
Verify by checking the coordinator logs:

```bash
kubectl logs -n ocr deployment/ocr-local-coordinator | grep -i migrate
```

---

## 5. KEDA Autoscaler Validation

KEDA autoscaling is opt-in per worker type. When enabled, KEDA creates a
`ScaledObject` that watches the RabbitMQ queue depth and scales workers accordingly.

### 5.1 Enable Autoscaling

Add to your values override:

```yaml
gpuWorker:
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 10
    pollingInterval: 15     # seconds between queue checks
    cooldownPeriod: 300     # seconds before scale-down
    queueTarget: 5          # messages per replica trigger

keda:
  scalingStrategy: "balanced"   # "aggressive", "balanced", or "conservative"
  maxReplicaCount: 50           # global cap across all worker types
```

The `scalingStrategy` adjusts polling and cooldown timing:

| Strategy | Polling Interval | Cooldown Period |
|---|---|---|
| `aggressive` | 10s | 60s |
| `balanced` | 15s (default) | 120-300s (per worker type) |
| `conservative` | 30s | 600s |

### 5.2 Verify ScaledObject Status

```bash
kubectl get scaledobject -n ocr

# Expected:
# NAME                           SCALETARGETKIND   SCALETARGETNAME        MIN   MAX   TRIGGERS   ...   READY   ACTIVE
# ocr-local-gpu-worker-scaler    apps/v1.Deployment ocr-local-gpu-worker  1     10    rabbitmq         True    False
```

- `READY=True`: KEDA can communicate with RabbitMQ
- `ACTIVE=True`: Queue depth exceeds threshold; scaling is in progress
- `ACTIVE=False`: Queue is empty or below threshold; at minReplicas

Check for errors:

```bash
kubectl describe scaledobject ocr-local-gpu-worker-scaler -n ocr
```

Common issues: `TriggerAuthentication` secret reference mismatch (the chart creates
`<release>-rabbitmq-auth` referencing `CELERY_BROKER_URL` from the secret).

### 5.3 Test Scale-Up

Submit test jobs to the coordinator API and watch worker replicas increase:

```bash
# Terminal 1: Watch replica count
kubectl get deployment ocr-local-gpu-worker -n ocr -w

# Terminal 2: Submit jobs (adjust URL and API key)
for i in $(seq 1 20); do
  curl -s -X POST http://localhost:8000/api/v1/jobs/ \
    -H "X-API-Key: <your-api-key>" \
    -F "file=@test-document.pdf" &
done
wait
```

Within the polling interval (15-30 seconds), KEDA should begin scaling up GPU workers.

### 5.4 Test Scale-to-Zero (CPU Workers)

CPU workers and layout workers can scale to zero when `minReplicas: 0`:

```bash
# Verify scale-to-zero
kubectl get deployment ocr-local-cpu-worker -n ocr
# READY should be 0/0 when queue is empty
```

GPU workers typically use `minReplicas: 1` to avoid cold-start latency from PaddlePaddle
model loading.

---

## 6. Network Policy Validation

Network policies are opt-in (`networkPolicy.enabled: true`). When enabled, the chart
creates two policies:

1. **Worker policy** (`<release>-worker-netpol`): Workers accept no inbound connections
   and can only reach PostgreSQL (5432), RabbitMQ (5672), Redis (6379), and DNS (53).
2. **Coordinator policy** (`<release>-coordinator-netpol`): Coordinator accepts inbound
   on port 8000, can reach the same infra services plus external HTTPS (443) and HTTP
   (80) for webhook delivery.

### 6.1 Enable Network Policies

```yaml
networkPolicy:
  enabled: true
```

Requires a CNI that supports NetworkPolicy (Calico, Cilium, or the cloud-native CNI).
GKE, EKS with Calico, and OKE with Calico all support this.

### 6.2 Verify Policies Are Applied

```bash
kubectl get networkpolicy -n ocr

# Expected:
# NAME                            POD-SELECTOR                                            AGE
# ocr-local-worker-netpol         app.kubernetes.io/component in (gpu-worker,cpu-worker,...) 1m
# ocr-local-coordinator-netpol    app.kubernetes.io/component=coordinator                    1m
```

### 6.3 Test: Coordinator Can Reach Infrastructure

```bash
# Exec into coordinator and verify connectivity
kubectl exec -n ocr deployment/ocr-local-coordinator -- \
  python -c "import socket; socket.create_connection(('ocr-local-postgres', 5432), timeout=5); print('PostgreSQL: OK')"

kubectl exec -n ocr deployment/ocr-local-coordinator -- \
  python -c "import socket; socket.create_connection(('ocr-local-rabbitmq', 5672), timeout=5); print('RabbitMQ: OK')"

kubectl exec -n ocr deployment/ocr-local-coordinator -- \
  python -c "import socket; socket.create_connection(('ocr-local-redis', 6379), timeout=5); print('Redis: OK')"
```

### 6.4 Test: Workers Can Reach Infrastructure

```bash
kubectl exec -n ocr deployment/ocr-local-gpu-worker -- \
  python -c "import socket; socket.create_connection(('ocr-local-postgres', 5432), timeout=5); print('PostgreSQL: OK')"

kubectl exec -n ocr deployment/ocr-local-gpu-worker -- \
  python -c "import socket; socket.create_connection(('ocr-local-rabbitmq', 5672), timeout=5); print('RabbitMQ: OK')"
```

### 6.5 Test: Workers Cannot Receive Inbound Traffic

Workers have no ingress rules. Verify that a connection attempt from a test pod is
rejected:

```bash
# Get a GPU worker pod IP
WORKER_IP=$(kubectl get pod -n ocr -l app.kubernetes.io/component=gpu-worker \
  -o jsonpath='{.items[0].status.podIP}')

# Attempt connection from a temporary pod (should time out or be refused)
kubectl run netpol-test --rm -it --image=busybox --restart=Never -n ocr -- \
  sh -c "timeout 5 nc -zv $WORKER_IP 8000 2>&1 || echo 'Connection blocked (expected)'"
```

### 6.6 Test: Coordinator Can Reach External (Webhooks)

The coordinator policy allows egress to port 443 and 80 for webhook delivery:

```bash
kubectl exec -n ocr deployment/ocr-local-coordinator -- \
  python -c "import urllib.request; urllib.request.urlopen('https://httpbin.org/status/200', timeout=10); print('External HTTPS: OK')"
```

---

## 7. PodDisruptionBudget Validation

The chart creates PDBs for all critical components with `maxUnavailable: 1`, which
ensures that at most one pod is evicted at a time during voluntary disruptions (node
drains, cluster upgrades).

### 7.1 List PDBs

```bash
kubectl get pdb -n ocr

# Expected:
# NAME                          MIN AVAILABLE   MAX UNAVAILABLE   ALLOWED DISRUPTIONS   AGE
# ocr-local-gpu-worker-pdb      N/A             1                 1                     5m
# ocr-local-coordinator-pdb     N/A             1                 1                     5m
# ocr-local-postgres-pdb        N/A             1                 0                     5m
# ocr-local-rabbitmq-pdb        N/A             1                 0                     5m
# ocr-local-redis-pdb           N/A             1                 0                     5m
# ocr-local-celery-beat-pdb     N/A             1                 1                     5m
```

`ALLOWED DISRUPTIONS` shows how many pods can be evicted right now. For single-replica
StatefulSets (PostgreSQL, RabbitMQ, Redis), this is `0` -- a drain will block until
the pod is rescheduled elsewhere first.

### 7.2 Test Node Drain

Pick a non-critical node (not the one hosting the sole PostgreSQL pod) and drain it:

```bash
# Identify which node hosts each pod
kubectl get pods -n ocr -o wide

# Cordon the node first (prevent new scheduling)
kubectl cordon <node-name>

# Drain with PDB respect
kubectl drain <node-name> \
  --ignore-daemonsets \
  --delete-emptydir-data \
  --grace-period=60

# Verify pods were rescheduled
kubectl get pods -n ocr -o wide
```

Expected behavior:

- Pods on the drained node are evicted one at a time (respecting `maxUnavailable: 1`)
- GPU workers with KEDA may scale up a replacement before the evicted pod terminates
- Single-replica StatefulSets (PostgreSQL, RabbitMQ, Redis) will block the drain until
  rescheduled on another node -- this is correct and prevents data loss

After validation, uncordon the node:

```bash
kubectl uncordon <node-name>
```

### 7.3 Multi-Replica PDB Behavior

With `coordinator.replicas: 2` and `maxUnavailable: 1`, draining a node that hosts one
coordinator pod will evict it while the other continues serving traffic. The
coordinator service routes to the remaining healthy pod.

For GPU workers with `gpuWorker.replicas: 2`, the same applies. One worker can be
evicted while the other continues processing OCR jobs.

---

## 8. Smoke Test Suite

After a successful Helm install, run these end-to-end checks to verify the full
pipeline is operational.

### 8.1 API Connectivity

```bash
# Port-forward (if no ingress)
kubectl port-forward svc/ocr-local-coordinator 8000:8000 -n ocr &

# Health check
curl -sf http://localhost:8000/api/v1/health/
# Expected: 200 OK with {"status": "healthy"}
```

### 8.2 Submit a Test OCR Job

```bash
# Submit a single-page PDF
JOB_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/jobs/ \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@test-document.pdf")

JOB_ID=$(echo "$JOB_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Submitted job: $JOB_ID"
```

### 8.3 Monitor Job Progress

```bash
# Poll job status
while true; do
  STATUS=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID/ \
    -H "X-API-Key: <your-api-key>" | python -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "Job status: $STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  sleep 5
done
```

### 8.4 Verify Output Artifacts

```bash
# Retrieve job details with artifact locations
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID/ \
  -H "X-API-Key: <your-api-key>" | python -m json.tool
```

Check that the response includes:

- `status: "completed"`
- Output PDF location
- Extracted text location
- Page count and processing metadata

### 8.5 Verify Audit Log (Chain of Custody)

```bash
# Check coordinator logs for custody events
kubectl logs -n ocr deployment/ocr-local-celery-coordinator | grep -i custody
```

### 8.6 Verify Worker Logs

```bash
# GPU worker processing confirmation
kubectl logs -n ocr deployment/ocr-local-gpu-worker --tail=50 | grep -i "task.*succeeded"
```

### 8.7 Verify Prometheus Metrics (if enabled)

```bash
# Port-forward to the coordinator
kubectl port-forward svc/ocr-local-coordinator 8000:8000 -n ocr &

# Scrape the Prometheus endpoint
curl -s http://localhost:8000/api/v1/prometheus/ \
  -H "Authorization: Bearer <metrics-api-key>" | head -30

# Expected: Prometheus text format with metrics like:
# ocr_jobs_total{status="completed"} 1
# ocr_workers_total{status="online"} 2
```

### 8.8 Verify Flower Dashboard (if enabled)

```bash
kubectl port-forward svc/ocr-local-flower 5555:5555 -n ocr &

# Open http://localhost:5555 in a browser
# Authenticate with the configured Flower credentials
```

### 8.9 Full Smoke Test Script

Combine all checks into a single validation script:

```bash
#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${OCR_NAMESPACE:-ocr}"
API_KEY="${OCR_API_KEY:?Set OCR_API_KEY}"
BASE_URL="${OCR_BASE_URL:-http://localhost:8000}"
TEST_FILE="${OCR_TEST_FILE:-test-document.pdf}"

echo "=== EDCOCR Cloud-Native Smoke Test ==="
echo "Namespace: $NAMESPACE"
echo "Base URL:  $BASE_URL"

# 1. Health check
echo -n "Health check... "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/health/")
if [ "$HTTP_CODE" = "200" ]; then echo "PASS"; else echo "FAIL ($HTTP_CODE)"; exit 1; fi

# 2. Submit job
echo -n "Submitting test job... "
RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/jobs/" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@$TEST_FILE")
JOB_ID=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null)
if [ -n "$JOB_ID" ]; then echo "PASS (job_id=$JOB_ID)"; else echo "FAIL"; echo "$RESPONSE"; exit 1; fi

# 3. Wait for completion (timeout: 5 minutes)
echo -n "Waiting for completion... "
TIMEOUT=300
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
  STATUS=$(curl -s "$BASE_URL/api/v1/jobs/$JOB_ID/" \
    -H "X-API-Key: $API_KEY" | python -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
  if [ "$STATUS" = "completed" ]; then echo "PASS ($ELAPSED s)"; break; fi
  if [ "$STATUS" = "failed" ]; then echo "FAIL (job failed after $ELAPSED s)"; exit 1; fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done
if [ $ELAPSED -ge $TIMEOUT ]; then echo "FAIL (timeout after ${TIMEOUT}s, last status: $STATUS)"; exit 1; fi

# 4. Verify pods are healthy
echo -n "Pod health... "
UNHEALTHY=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep -v Running | grep -v Completed | wc -l)
if [ "$UNHEALTHY" -eq 0 ]; then echo "PASS"; else echo "WARN ($UNHEALTHY pods not Running)"; fi

# 5. Verify PDBs exist
echo -n "PDB check... "
PDB_COUNT=$(kubectl get pdb -n "$NAMESPACE" --no-headers | wc -l)
if [ "$PDB_COUNT" -ge 4 ]; then echo "PASS ($PDB_COUNT PDBs)"; else echo "WARN ($PDB_COUNT PDBs, expected >= 4)"; fi

echo "=== Smoke test complete ==="
```

---

## 9. Rollback Procedure

### 9.1 Helm Rollback

If a deployment is broken after upgrade:

```bash
# List release history
helm history ocr-local -n ocr

# Roll back to the previous revision
helm rollback ocr-local -n ocr

# Roll back to a specific revision
helm rollback ocr-local 3 -n ocr
```

Rollback restores the previous set of Kubernetes manifests. StatefulSet data
(PostgreSQL, RabbitMQ, Redis PVCs) is preserved across rollbacks.

### 9.2 When to Roll Back vs Patch Forward

**Roll back** when:

- Workers fail to start (image pull errors, crash loops, missing GPU resources)
- StatefulSets are stuck (PVC binding failures, storage class issues)
- Coordinator is unreachable (misconfigured secrets, ingress errors)
- Jobs are failing at a rate above the 10% error rate alert threshold

**Patch forward** when:

- A single configuration value is wrong (fix in values file, `helm upgrade`)
- A worker is slow but functional (adjust resource requests, `helm upgrade`)
- Autoscaling parameters need tuning (update KEDA values, `helm upgrade`)

```bash
# Patch forward with corrected values
helm upgrade ocr-local helm/ocr-local/ \
  --namespace ocr \
  --values values-secret.yaml \
  --values values-<cloud>.yaml \
  --wait \
  --timeout 10m
```

### 9.3 Emergency: Full Uninstall and Reinstall

If rollback and patching both fail:

```bash
# Uninstall the release (PVCs are preserved by default)
helm uninstall ocr-local -n ocr

# Verify PVCs still exist
kubectl get pvc -n ocr

# Reinstall with corrected values
helm install ocr-local helm/ocr-local/ \
  --namespace ocr \
  --values values-secret.yaml \
  --values values-<cloud>.yaml \
  --wait \
  --timeout 10m
```

StatefulSet PVCs survive `helm uninstall` because their `reclaimPolicy` defaults to
`Retain`. PostgreSQL, RabbitMQ, and Redis will reattach their existing volumes on
reinstall.

### 9.4 Post-Rollback Verification

After any rollback or reinstall:

1. Run the full smoke test from Section 8.9
2. Check for stuck jobs: `kubectl exec -n ocr deployment/ocr-local-coordinator -- python manage.py shell -c "from jobs.models import Job; print(Job.objects.filter(status='processing').count)"`
3. Verify Prometheus metrics are flowing (if enabled)
4. Check the Flower dashboard for worker connectivity

---

## Appendix: Quick Reference Commands

| Action | Command |
|---|---|
| Lint the chart | `helm lint helm/ocr-local/` |
| Dry-run install | `helm install ocr-local helm/ocr-local/ -n ocr --values values-secret.yaml --dry-run` |
| Install | `helm install ocr-local helm/ocr-local/ -n ocr --create-namespace --values values-secret.yaml --wait` |
| Upgrade | `helm upgrade ocr-local helm/ocr-local/ -n ocr --values values-secret.yaml --wait` |
| Rollback | `helm rollback ocr-local -n ocr` |
| Uninstall | `helm uninstall ocr-local -n ocr` |
| Check all pods | `kubectl get pods -n ocr -o wide` |
| Check PVCs | `kubectl get pvc -n ocr` |
| Check PDBs | `kubectl get pdb -n ocr` |
| Check ScaledObjects | `kubectl get scaledobject -n ocr` |
| Check NetworkPolicies | `kubectl get networkpolicy -n ocr` |
| Coordinator logs | `kubectl logs -n ocr deployment/ocr-local-coordinator --tail=100` |
| GPU worker logs | `kubectl logs -n ocr deployment/ocr-local-gpu-worker --tail=100` |
| Celery worker logs | `kubectl logs -n ocr deployment/ocr-local-celery-coordinator --tail=100` |
| Port-forward API | `kubectl port-forward svc/ocr-local-coordinator 8000:8000 -n ocr` |
| Port-forward Flower | `kubectl port-forward svc/ocr-local-flower 5555:5555 -n ocr` |
| RabbitMQ queue depth | `kubectl exec -n ocr statefulset/ocr-local-rabbitmq -- rabbitmqctl list_queues` |
| PostgreSQL shell | `kubectl exec -it -n ocr statefulset/ocr-local-postgres -- psql -U ocr -d ocr` |
| Redis CLI | `kubectl exec -it -n ocr statefulset/ocr-local-redis -- redis-cli -a <password>` |
