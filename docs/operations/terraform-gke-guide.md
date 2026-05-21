# GKE Deployment Guide for OCR-Local

**Status**: Production-ready configuration (template-validated, pending real-cluster validation)
**Module**: `terraform/modules/gke/`
**Validation script**: `scripts/validate_terraform_gke.py`

---

## Overview

This guide covers deploying OCR-Local on Google Kubernetes Engine (GKE) using the Terraform modules provided in this repository. The GKE module provisions a production-grade cluster with CPU and GPU node pools, VPC networking, IAM service accounts, and Artifact Registry for container images.

## Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed and authenticated:
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```
3. **Terraform >= 1.5** installed
4. **kubectl** installed
5. **GPU quota** approved for your target region (request via GCP Console > IAM & Admin > Quotas)
6. **Helm 3** installed (for OCR-Local chart deployment)

## Architecture

The GKE module creates the following resources:

```
+---------------------------+
|        GCP Project        |
|                           |
|  +---------------------+ |
|  |    VPC Network       | |
|  |  +----------------+  | |
|  |  |   Subnet       |  | |
|  |  | 10.0.0.0/20    |  | |
|  |  | Pods: /14      |  | |
|  |  | Services: /20  |  | |
|  |  +----------------+  | |
|  +---------------------+ |
|                           |
|  +---------------------+ |
|  |   GKE Cluster       | |
|  |  +------+ +------+  | |
|  |  | CPU  | | GPU  |  | |
|  |  | Pool | | Pool |  | |
|  |  +------+ +------+  | |
|  +---------------------+ |
|                           |
|  +---------------------+ |
|  | Artifact Registry   | |
|  +---------------------+ |
|                           |
|  Cloud Router + NAT       |
|  IAM Service Account      |
+---------------------------+
```

## Quick Start

```bash
# 1. Navigate to environment
cd terraform/environments/staging

# 2. Configure variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars:
#   cloud_provider = "gcp"
#   gcp_project_id = "your-project-id"
#   gcp_region     = "us-central1"

# 3. Initialize and apply
terraform init
terraform plan
terraform apply

# 4. Configure kubectl
gcloud container clusters get-credentials ocr-local-staging \
  --region us-central1 --project your-project-id

# 5. Deploy OCR-Local
helm install ocr-local ../../helm/ocr-local/ -f values-gke.yaml
```

## GKE-Specific Configuration

### Node Pool Setup

The module creates two node pools:

| Pool | Machine Type | Purpose | Autoscaling |
|------|-------------|---------|-------------|
| CPU  | e2-standard-8 | Coordinator, Celery workers, Redis, RabbitMQ, PostgreSQL | 1-10 nodes |
| GPU  | n1-standard-8 + T4 | OCR GPU workers (PaddleOCR inference) | 0-8 nodes |

#### GPU Node Pool Details

```hcl
# terraform/modules/gke/main.tf (excerpt)
resource "google_container_node_pool" "gpu" {
  # ...
  node_config {
    machine_type = var.gpu_machine_type  # n1-standard-8
    disk_size_gb = var.gpu_disk_size_gb  # 100+ GB for models

    guest_accelerator {
      type  = var.gpu_type   # nvidia-tesla-t4
      count = var.gpu_count  # 1 per node

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }
}
```

The GPU taint ensures only pods with matching tolerations (OCR workers) schedule onto GPU nodes.

#### Recommended GPU Types for OCR Workloads

| GPU Type | VRAM | OCR Workers | Cost Tier | Notes |
|----------|------|-------------|-----------|-------|
| nvidia-tesla-t4 | 16 GB | 8-12 | Low | Best price/performance for inference |
| nvidia-l4 | 24 GB | 12-16 | Medium | Newer generation, better throughput |
| nvidia-tesla-v100 | 16 GB | 8-12 | Medium | Older but widely available |
| nvidia-tesla-a100 | 40/80 GB | 16-24+ | High | Maximum throughput for large batches |

### Workload Identity

The module enables GKE Workload Identity, which is the recommended way to authenticate pods to GCP services:

```hcl
workload_identity_config {
  workload_pool = "${var.project_id}.svc.id.goog"
}
```

To bind a Kubernetes service account to a GCP service account:

```bash
# Create GCP service account for OCR workers
gcloud iam service-accounts create ocr-worker-sa \
  --display-name="OCR Worker Service Account"

# Grant permissions (e.g., GCS access for input/output)
gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:ocr-worker-sa@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Bind KSA to GSA
gcloud iam service-accounts add-iam-policy-binding \
  ocr-worker-sa@YOUR_PROJECT.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:YOUR_PROJECT.svc.id.goog[ocr-local/ocr-worker]"

# Annotate the Kubernetes service account
kubectl annotate serviceaccount ocr-worker \
  --namespace ocr-local \
  iam.gke.io/gcp-service-account=ocr-worker-sa@YOUR_PROJECT.iam.gserviceaccount.com
```

### VPC-Native Mode

The cluster uses VPC-native mode with secondary IP ranges for pods and services:

| Range | Default CIDR | Purpose |
|-------|-------------|---------|
| Subnet primary | 10.0.0.0/20 | Node IPs |
| Pods (secondary) | 10.4.0.0/14 | Pod IPs (~262k addresses) |
| Services (secondary) | 10.8.0.0/20 | Service ClusterIPs |

### Private Cluster

Nodes have private IPs only. A Cloud NAT gateway provides outbound internet access for pulling container images and model downloads:

```hcl
private_cluster_config {
  enable_private_nodes    = true
  enable_private_endpoint = false   # API server reachable from authorized networks
  master_ipv4_cidr_block  = "172.16.0.0/28"
}
```

### Release Channel

The module supports GKE release channels for automatic cluster version management:

| Channel | Stability | Recommended For |
|---------|-----------|-----------------|
| STABLE | Highest | Production |
| REGULAR | Balanced | Staging |
| RAPID | Newest | Development/testing |

Production deployments should use `STABLE`:
```hcl
release_channel = "STABLE"
```

## IAM Configuration

The module creates a dedicated service account for GKE nodes with least-privilege roles:

| Role | Purpose |
|------|---------|
| `roles/logging.logWriter` | Write logs to Cloud Logging |
| `roles/monitoring.metricWriter` | Write metrics to Cloud Monitoring |
| `roles/artifactregistry.reader` | Pull container images |

**Never assign** `roles/editor`, `roles/owner`, or `roles/compute.admin` to node service accounts.

## Cost Optimization

### Scale-to-Zero GPU Nodes

For staging environments, set `gpu_node_min_count = 0` so GPU nodes are deprovisioned when idle. KEDA autoscaling brings them up when OCR jobs are queued:

```hcl
gpu_node_min_count = 0
gpu_node_max_count = 4
```

Note: Scale-from-zero has a cold start delay of 2-5 minutes while GKE provisions new nodes and installs GPU drivers.

### Preemptible/Spot Nodes

For non-production, consider preemptible VMs for significant cost savings (60-91% discount). Add to node config:

```hcl
node_config {
  preemptible = true
  # or for Spot VMs:
  spot = true
}
```

**Warning**: Preemptible/spot nodes can be terminated at any time. Only use for fault-tolerant OCR workers with crash-resume capability.

### Committed Use Discounts

For production, purchase committed use contracts for predictable workloads (CPU node pool). GPU-accelerated committed use is available for sustained GPU inference loads.

### Regional vs Zonal

The module deploys a regional cluster by default (nodes across multiple zones). For cost savings in staging:

```hcl
# Use a specific zone instead of region
location = "us-central1-a"  # Instead of "us-central1"
```

## Security Best Practices

The module implements these GKE security features:

1. **Shielded Nodes**: Secure Boot enabled on all node pools
2. **Legacy Metadata Disabled**: `disable-legacy-endpoints = true`
3. **Private Nodes**: No public IPs on nodes
4. **VPC-Native**: Network policies supported
5. **Workload Identity**: No node-level credentials
6. **Dedicated Service Account**: Least-privilege IAM

### Additional Hardening (Manual)

These should be configured after initial deployment:

```bash
# Enable Binary Authorization (restrict container sources)
gcloud container clusters update ocr-local-production \
  --enable-binauthz --region us-central1

# Enable intranode visibility for network policy enforcement
gcloud container clusters update ocr-local-production \
  --enable-intra-node-visibility --region us-central1

# Configure authorized networks for API server access
gcloud container clusters update ocr-local-production \
  --enable-master-authorized-networks \
  --master-authorized-networks YOUR_OFFICE_CIDR
```

## Validation

Run the GKE-specific validation script before deploying:

```bash
# Basic validation
python scripts/validate_terraform_gke.py --terraform-dir terraform/

# Strict validation (recommended for production)
python scripts/validate_terraform_gke.py --terraform-dir terraform/ --strict

# Generate reports
python scripts/validate_terraform_gke.py \
  --terraform-dir terraform/ \
  --output-dir docs/reports/ \
  --strict
```

The script checks:
- Workload Identity configuration
- VPC-native mode (pod/service secondary ranges)
- Private cluster configuration
- Node pool autoscaling bounds
- GPU driver auto-installation
- GPU taints for scheduling isolation
- IAM least-privilege bindings
- Network configuration (NAT, firewall rules)
- Resource labels for cost tracking
- Autopilot vs Standard mode detection
- Shielded instance configuration
- Auto-repair and auto-upgrade settings

## Monitoring

After deployment, the shared Terraform module deploys Prometheus and Grafana:

```bash
# Access Grafana dashboard
kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80

# Import OCR-Local dashboard
# (available as ConfigMap if grafana.dashboard.enabled in Helm values)
```

Key metrics to monitor:
- `ocr_pages_processed_total` -- Pages processed by backend
- `ocr_job_duration_seconds` -- Processing time per job
- `ocr_queue_depth` -- RabbitMQ queue depth (KEDA scaling trigger)
- GPU utilization via `nvidia_gpu_duty_cycle`

## Troubleshooting

### GPU Nodes Not Scaling Up

1. Check quota: `gcloud compute regions describe us-central1 | grep -A2 NVIDIA`
2. Verify KEDA is running: `kubectl get pods -n keda`
3. Check KEDA scaler: `kubectl describe scaledobject -n ocr-local`
4. Check node pool events: `kubectl get events --sort-by=.lastTimestamp`

### GPU Drivers Not Installing

The module uses `gpu_driver_installation_config` for automatic driver installation. If drivers are missing:

```bash
# Check DaemonSet
kubectl get ds -n kube-system | grep nvidia

# Verify driver version
kubectl exec -it GPU_POD -- nvidia-smi
```

### Workload Identity Not Working

```bash
# Verify KSA annotation
kubectl describe sa ocr-worker -n ocr-local

# Test from a pod
kubectl run test --image=google/cloud-sdk --rm -it -- \
  gcloud auth list
```

### Private Cluster Cannot Pull Images

Verify Cloud NAT is active:

```bash
gcloud compute routers nats describe ocr-local-nat \
  --router=ocr-local-router --region=us-central1
```

## Environment-Specific Settings

### Staging

```hcl
# terraform/environments/staging/terraform.tfvars
cloud_provider = "gcp"
gcp_project_id = "ocr-local-staging"
gcp_region     = "us-central1"
cluster_name   = "ocr-local-staging"

# Smaller pools, scale-to-zero GPU
cpu_node_min_count = 1
cpu_node_max_count = 5
gpu_node_min_count = 0
gpu_node_max_count = 4
```

### Production

```hcl
# terraform/environments/production/terraform.tfvars
cloud_provider = "gcp"
gcp_project_id = "ocr-local-production"
gcp_region     = "us-central1"
cluster_name   = "ocr-local-production"

# Larger pools with minimum GPU headroom
release_channel    = "STABLE"
cpu_node_min_count = 2
cpu_node_max_count = 20
gpu_node_min_count = 1
gpu_node_max_count = 16
gpu_disk_size_gb   = 200

grafana_admin_password = "use-a-strong-password"
```

## Related Documentation

- [Terraform Module Validation Guide](terraform-validation.md) -- Overview of all cloud provider modules
- [Production Cutover Runbook](production-cutover-runbook.md) -- Step-by-step deployment guide
- [Failover Runbook](../FAILOVER-RUNBOOK.md) -- Operational failover procedures
- [CPU vs GPU Analysis](../cpu-vs-gpu-analysis.md) -- Deployment cost comparison
