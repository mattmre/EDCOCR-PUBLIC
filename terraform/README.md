# Terraform — OCR-Local Cloud-Native Deployment

Multi-cloud Terraform modules for deploying OCR-Local on managed Kubernetes services.

## Supported Cloud Providers

| Provider | Service | Module |
|----------|---------|--------|
| AWS | EKS (Elastic Kubernetes Service) | `modules/eks/` |
| GCP | GKE (Google Kubernetes Engine) | `modules/gke/` |
| Oracle | OKE (Oracle Kubernetes Engine) | `modules/oke/` |

## Architecture

Each cloud module provisions:

- **Managed Kubernetes cluster** with configurable version
- **CPU node pool** for coordinator, Celery workers, and supporting services
- **GPU node pool** with NVIDIA taints for OCR GPU workers (scale-to-zero capable)
- **VPC/VCN networking** with public and private subnets, NAT gateways
- **IAM roles / service accounts** with least-privilege access
- **Container registry** for OCR-Local Docker images

Shared modules deploy:

- **KEDA** for queue-length-based autoscaling of worker pods
- **Prometheus + Grafana** for metrics collection and dashboards

## Quick Start

```bash
# 1. Navigate to an environment
cd environments/staging

# 2. Copy and configure variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your cloud credentials and settings

# 3. Initialize Terraform
terraform init

# 4. Preview changes
terraform plan

# 5. Apply
terraform apply

# 6. Configure kubectl
# (the output includes the provider-specific kubeconfig command)

# 7. Deploy OCR-Local via Helm
helm install ocr-local ../../helm/ocr-local/ -f values-staging.yaml
```

## Directory Structure

```
terraform/
  modules/
    eks/             # AWS EKS cluster + VPC + IAM + ECR
    gke/             # GCP GKE cluster + VPC + IAM + Artifact Registry
    oke/             # Oracle OKE cluster + VCN + OCIR
    shared/          # KEDA + Prometheus/Grafana Helm deployments
  environments/
    staging/         # Staging environment (smaller node pools)
    production/      # Production environment (HA, larger pools)
```

## Prerequisites

- Terraform >= 1.5
- Cloud provider CLI configured with appropriate credentials:
  - AWS: `aws configure`
  - GCP: `gcloud auth application-default login`
  - Oracle: `oci setup config`
- kubectl installed

## Security Notes

- **Never commit `terraform.tfvars`** — it contains credentials
- GPU nodes use taints to prevent non-GPU workloads from scheduling
- Private subnets with NAT gateways for node egress
- Container registries have image scanning enabled (AWS/GCP)
- Service accounts use least-privilege IAM bindings

## Customization

Override variables in `terraform.tfvars` or via CLI:

```bash
terraform apply -var="cluster_name=my-ocr-cluster" -var="cloud_provider=gcp"
```

See `variables.tf` in each module for all configurable parameters.
