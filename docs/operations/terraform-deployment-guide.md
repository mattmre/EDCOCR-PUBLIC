# Terraform Deployment Guide

**Status**: Production reference
**Modules**: EKS (AWS), GKE (GCP), OKE (Oracle), Shared (KEDA + Monitoring)

---

## Pre-requisites

### Terraform CLI

Install Terraform >= 1.5:

```bash
# macOS
brew install terraform

# Linux (apt)
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Verify
terraform version
```

### Cloud Provider CLIs

| Provider | CLI Tool | Install |
|----------|----------|---------|
| AWS | `aws` | `pip install awscli` or `brew install awscli` |
| GCP | `gcloud` | [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install) |
| Oracle | `oci` | `pip install oci-cli` or [docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm) |

### Authentication

Each provider requires credentials configured before `terraform plan`:

**AWS:**
```bash
aws configure
# or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
```

**GCP:**
```bash
gcloud auth application-default login
# or export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**Oracle:**
```bash
oci setup config
# Creates ~/.oci/config with tenancy, user, key, fingerprint, region
```

---

## Module Overview

```
terraform/
  modules/
    eks/          -- AWS EKS cluster (VPC, node groups, ECR, IAM)
    gke/          -- GCP GKE cluster (VPC, node pools, Artifact Registry, IAM)
    oke/          -- Oracle OKE cluster (VCN, node pools, OCIR)
    shared/       -- KEDA autoscaling + Prometheus/Grafana monitoring
  environments/
    staging/      -- Staging composition (smaller pools, debug logging)
    production/   -- Production composition (HA, larger pools, stricter defaults)
```

Each environment's `main.tf` accepts a `cloud_provider` variable (`aws`, `gcp`, or `oracle`) and conditionally provisions the corresponding module.

---

## Per-Cloud Setup

### AWS (EKS)

1. Configure AWS credentials (see above).
2. Copy the tfvars example:
   ```bash
   cd terraform/environments/staging
   cp terraform.tfvars.example terraform.tfvars
   ```
3. Edit `terraform.tfvars`:
   ```hcl
   cloud_provider = "aws"
   aws_region     = "us-east-1"   # Your target region
   cluster_name   = "ocr-local-staging"
   ```
4. No AMI selection needed -- EKS managed node groups use AWS-optimized AMIs automatically.

**Key variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region | `us-east-1` |
| `cpu_node_instance_types` | EC2 types for CPU pool | `["m6i.2xlarge"]` |
| `gpu_node_instance_types` | EC2 types for GPU pool | `["g5.xlarge"]` |
| `gpu_node_disk_size` | Disk GB for GPU nodes | `100` |

### GCP (GKE)

1. Configure GCP credentials (see above).
2. Copy and edit tfvars:
   ```bash
   cd terraform/environments/staging
   cp terraform.tfvars.example terraform.tfvars
   ```
3. Edit `terraform.tfvars`:
   ```hcl
   cloud_provider = "gcp"
   gcp_project_id = "my-project-123"   # REPLACE: your GCP project ID
   gcp_region     = "us-central1"
   ```
4. Ensure the required GCP APIs are enabled:
   ```bash
   gcloud services enable container.googleapis.com compute.googleapis.com \
     artifactregistry.googleapis.com --project=my-project-123
   ```

**Key variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `gcp_project_id` | GCP project (required) | -- |
| `gcp_region` | GCP region | `us-central1` |
| `gpu_type` | GPU accelerator | `nvidia-tesla-t4` |
| `gpu_count` | GPUs per node | `1` |
| `release_channel` | GKE update channel | `REGULAR` |

### Oracle (OKE)

1. Configure OCI credentials (see above).
2. Copy and edit tfvars:
   ```bash
   cd terraform/environments/staging
   cp terraform.tfvars.example terraform.tfvars
   ```
3. Edit `terraform.tfvars`:
   ```hcl
   cloud_provider     = "oracle"
   oci_compartment_id = "ocid1.compartment.oc1..aaaa..."  # REPLACE
   oci_region         = "us-ashburn-1"
   ```

4. **Find the correct node image OCID** (required):

   OKE node images are region-specific. You must provide a valid OCID for your target region.

   **Option A -- OCI Console:**
   - Navigate to Compute > Images
   - Filter: Operating System = "Oracle Linux", Version = "8"
   - Select an image compatible with your target Kubernetes version
   - Copy the OCID from the image details page

   **Option B -- OCI CLI:**
   ```bash
   # List recent Oracle Linux 8 images for your region
   oci compute image list \
     --compartment-id <COMPARTMENT_OCID> \
     --operating-system "Oracle Linux" \
     --operating-system-version "8" \
     --shape "VM.Standard.E4.Flex" \
     --sort-by TIMECREATED \
     --sort-order DESC \
     --limit 5 \
     --query 'data[*].{id:id, name:"display-name", created:"time-created"}' \
     --output table
   ```

   **Option C -- OCI API (for GPU shapes):**
   ```bash
   oci compute image list \
     --compartment-id <COMPARTMENT_OCID> \
     --operating-system "Oracle Linux" \
     --shape "VM.GPU.A10.1" \
     --sort-by TIMECREATED \
     --sort-order DESC \
     --limit 5
   ```

5. Set the image OCIDs in your tfvars:
   ```hcl
   oci_node_image_id     = "ocid1.image.oc1.iad.aaaa..."  # CPU nodes
   oci_gpu_node_image_id = "ocid1.image.oc1.iad.aaaa..."  # GPU nodes
   ```

**Key variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `oci_compartment_id` | OCI compartment OCID (required) | -- |
| `oci_region` | OCI region | `us-ashburn-1` |
| `oci_node_image_id` | CPU node image OCID (required) | -- |
| `oci_gpu_node_image_id` | GPU node image OCID (required) | -- |
| `cpu_node_shape` | OCI shape for CPU pool | `VM.Standard.E4.Flex` |
| `gpu_node_shape` | OCI shape for GPU pool | `VM.GPU.A10.1` |

---

## Validation Steps

Run these checks before applying any changes.

### 1. Format Check

Verify all `.tf` files follow canonical formatting:

```bash
terraform fmt -check -recursive terraform/
```

Fix formatting issues automatically:

```bash
terraform fmt -recursive terraform/
```

### 2. Module Validation

Validate each module's syntax and internal consistency:

```bash
# EKS module
cd terraform/modules/eks
terraform init -backend=false
terraform validate

# GKE module
cd terraform/modules/gke
terraform init -backend=false
terraform validate

# OKE module
cd terraform/modules/oke
terraform init -backend=false
terraform validate

# Shared module
cd terraform/modules/shared
terraform init -backend=false
terraform validate
```

### 3. Plan (Dry Run)

After configuring your tfvars, generate an execution plan:

```bash
cd terraform/environments/staging
terraform init
terraform plan -out=tfplan
```

Review the plan output carefully. It shows what resources will be created, modified, or destroyed.

### 4. Automated Validation Script

Use the included validation script to run all checks at once:

```bash
bash scripts/validate_terraform.sh
```

The script checks:
- Terraform CLI is installed
- All `.tf` files are formatted correctly
- Each module passes `terraform validate`
- Reports pass/fail per module

### 5. Python Static Analysis

For deeper static checks (no Terraform CLI required):

```bash
python scripts/validate_terraform.py --environment all
```

---

## The Validation-to-Deployment Gap

Static validation (`terraform fmt`, `terraform validate`, and `scripts/validate_terraform.py`) catches structural errors but does NOT verify:

| What static validation checks | What it does NOT check |
|-------------------------------|------------------------|
| HCL syntax correctness | Cloud provider credentials are valid |
| Variable type constraints | Resources can actually be created |
| Required variables are declared | Quotas and limits in your account |
| Provider version constraints | Network connectivity to cloud APIs |
| Formatting consistency | IAM permissions are sufficient |
| Placeholder detection | Correct image OCIDs / AMIs for region |
| Naming convention compliance | Cost implications of the plan |

**Always run `terraform plan` with valid credentials against a real cloud account before `terraform apply`.** The plan output is the only way to confirm what resources will be created and at what cost.

### Common Deployment Failures Not Caught by Validation

1. **Insufficient quotas**: GPU instance quotas are often zero by default. Request increases before deploying GPU node pools.
2. **Wrong region for image OCIDs**: OKE node images are region-specific. An OCID from `us-ashburn-1` will not work in `eu-frankfurt-1`.
3. **Missing API enablement**: GKE requires `container.googleapis.com` enabled; OKE requires policies granting the user access to `CLUSTER_MANAGE`.
4. **Network conflicts**: VPC/VCN CIDR ranges may conflict with existing infrastructure in your account.

---

## Adding to CI

To run Terraform validation in GitHub Actions:

```yaml
- name: Terraform format check
  run: |
    terraform fmt -check -recursive terraform/

- name: Validate modules
  run: bash scripts/validate_terraform.sh
```

The validation script exits with a non-zero status if any module fails, making it suitable for CI gate checks.

---

## Applying Changes

Once the plan looks correct:

```bash
# Apply with the saved plan
terraform apply tfplan

# Or apply interactively
terraform apply
```

After apply completes, configure kubectl:

```bash
# The kubeconfig_command output tells you exactly what to run
terraform output kubeconfig_command
```

Then deploy the OCR-Local Helm chart:

```bash
helm upgrade --install ocr-local helm/ocr-local/ \
  --namespace ocr-local --create-namespace \
  -f helm/ocr-local/values.yaml \
  --set secrets.djangoSecretKey="$(openssl rand -hex 32)" \
  --set secrets.postgresPassword="$(openssl rand -hex 16)" \
  --set secrets.rabbitmqPassword="$(openssl rand -hex 16)"
```

See `docs/operations/production-cutover-runbook.md` for the full production deployment checklist.
