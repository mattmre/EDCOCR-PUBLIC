# Terraform Module Validation Guide

**Status**: Documentation
**Modules**: EKS, GKE, OKE (delivered)

---

## Module Locations

- `terraform/modules/eks/` -- Amazon EKS cluster
- `terraform/modules/gke/` -- Google GKE cluster
- `terraform/modules/oke/` -- Oracle OKE cluster
- `terraform/modules/shared/` -- KEDA autoscaling and monitoring (shared across providers)
- `terraform/environments/staging/` -- Staging environment composition
- `terraform/environments/production/` -- Production environment composition

## Validation Status

All modules are template-ready but have NOT been validated against real cloud accounts.

## Required Variables

Each module requires cloud-specific credentials and configuration.
See `terraform/modules/<provider>/variables.tf` for full variable reference.

### Common Variables (all providers)

| Variable | Description | Default |
|---|---|---|
| `cluster_name` | Name of the Kubernetes cluster | (required) |
| `cluster_version` | Kubernetes version | Provider-specific |

### AWS EKS

| Variable | Description | Default |
|---|---|---|
| `region` | AWS region | `us-east-1` |
| `vpc_cidr` | CIDR block for the VPC | `10.0.0.0/16` |
| `private_subnets` | CIDR blocks for private subnets | (required) |

### GCP GKE

| Variable | Description | Default |
|---|---|---|
| `project_id` | GCP project ID | (required) |
| `region` | GCP region | `us-central1` |

### Oracle OKE

| Variable | Description | Default |
|---|---|---|
| `compartment_id` | OCI compartment OCID | (required) |
| `region` | OCI region | `us-ashburn-1` |

## Validation Procedure

1. Configure cloud credentials (AWS/GCP/OCI CLI)
2. Choose target environment: `cd terraform/environments/staging`
3. Set `cloud_provider` variable to `aws`, `gcp`, or `oracle`
4. Run `terraform init && terraform plan`
5. Review plan output for expected resources
6. Apply in a non-production account: `terraform apply`
7. Verify cluster health: `kubectl get nodes`
8. Deploy EDCOCR Helm chart: `helm install ocr-local helm/ocr-local/`
9. Run smoke test with test documents
10. Tear down: `terraform destroy`

## Known Limitations

- GPU node pools require quota pre-approval from the cloud provider
- KEDA autoscaling requires KEDA operator pre-installation on the cluster
- S3/GCS/OCI Object Storage buckets are created but IAM policies may need adjustment
- Staging environment defaults to a single-node GPU pool; scale up via `gpu_node_count`
- Remote state backend blocks are commented out in environment configs; uncomment and configure for team use
