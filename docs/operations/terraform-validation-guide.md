# Terraform Validation Guide

**Status**: Production
**Modules**: EKS, GKE, OKE, Shared (KEDA + Monitoring)
**Tools**: `scripts/validate_terraform.py`, `scripts/terraform_plan_check.py`

---

## Overview

OCR-Local ships multi-cloud Terraform modules for deploying on managed Kubernetes services (AWS EKS, GCP GKE, Oracle OKE). Two validation tools ensure these modules remain structurally sound, credential-free, and compliant with project conventions.

| Tool | Purpose | Requires Terraform CLI |
|------|---------|----------------------|
| `validate_terraform.py` | Static analysis of `.tf` files (variables, credentials, tags, naming) | No |
| `terraform_plan_check.py` | Dry-run validation via `terraform init/validate/plan` | Yes |

## Quick Start

### Static Validation (No Dependencies)

```bash
# Validate all modules
python scripts/validate_terraform.py --environment all

# Validate EKS only
python scripts/validate_terraform.py --environment eks

# Strict mode (warnings = errors)
python scripts/validate_terraform.py --environment all --strict

# JSON output
python scripts/validate_terraform.py --environment all --json-only

# Save reports to directory
python scripts/validate_terraform.py --environment all --output-dir reports/terraform/
```

### Plan-Mode Validation (Requires Terraform CLI)

```bash
# Validate staging environment
python scripts/terraform_plan_check.py --environment staging

# Skip plan (init + validate only)
python scripts/terraform_plan_check.py --environment staging --skip-plan

# Production environment
python scripts/terraform_plan_check.py --environment production

# Save reports
python scripts/terraform_plan_check.py --environment staging --output-dir reports/terraform/
```

## Validation Rules

### Variable Checks

| Rule | Severity | Description |
|------|----------|-------------|
| `VAR-DESC` | Error | Every variable must have a `description` field |
| `VAR-TYPE` | Error | Every variable must have an explicit `type` constraint |
| `VAR-REQUIRED` | Info | Variables without `default` are flagged as required at apply time |
| `VAR-SENSITIVE` | Warning | Variables named `password`, `secret`, `token`, etc. should be marked `sensitive = true` |

### Security Checks

| Rule | Severity | Description |
|------|----------|-------------|
| `CRED-HARDCODED` | Error | Scans for hardcoded AWS access keys, secret keys, passwords, tokens, and private keys |
| `PLACEHOLDER` | Warning | Flags `placeholder`, `REPLACE_ME`, `TODO`, and `FIXME` in non-comment lines |

### Structure Checks

| Rule | Severity | Description |
|------|----------|-------------|
| `TF-VERSION` | Error | `terraform { required_version }` must be present |
| `TF-PROVIDERS` | Warning | `required_providers` block expected in modules |
| `BACKEND-CFG` | Warning | Backend configuration recommended for environment-level configs |
| `TF-BLOCK` | Warning | `main.tf` should contain a `terraform {}` block |
| `MODULE-MISSING` | Error | Target directory contains no `.tf` files |

### Convention Checks

| Rule | Severity | Description |
|------|----------|-------------|
| `RES-NAMING` | Warning | Resources should reference `var.cluster_name` in their naming |
| `RES-TAGS` | Warning | AWS resources need `tags`, GCP needs `labels`, OCI needs `freeform_tags` |

## Report Formats

Both tools emit reports in two formats.

### JSON Report

```json
{
  "environment": "all",
  "timestamp": "2026-03-15T12:00:00+00:00",
  "passed": true,
  "summary": {
    "files_scanned": 13,
    "variables_checked": 42,
    "errors": 0,
    "warnings": 3,
    "info": 5
  },
  "findings": [
    {
      "rule": "VAR-REQUIRED",
      "severity": "info",
      "message": "Variable 'cluster_name' has no default (required at apply time)",
      "file": "modules/eks/variables.tf",
      "line": 6
    }
  ]
}
```

### Markdown Report

The markdown report includes a summary table and a detailed findings table, suitable for inclusion in PR descriptions or documentation.

## How to Fix Common Issues

### VAR-DESC: Missing description

```hcl
# Before (fails)
variable "region" {
  type    = string
  default = "us-east-1"
}

# After (passes)
variable "region" {
  description = "AWS region for the cluster"
  type        = string
  default     = "us-east-1"
}
```

### VAR-TYPE: Missing type constraint

```hcl
# Before (fails)
variable "name" {
  description = "Cluster name"
  default     = "ocr-local"
}

# After (passes)
variable "name" {
  description = "Cluster name"
  type        = string
  default     = "ocr-local"
}
```

### VAR-SENSITIVE: Secret not marked sensitive

```hcl
# Before (warning)
variable "db_password" {
  description = "Database password"
  type        = string
}

# After (passes)
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}
```

### CRED-HARDCODED: Credential in source

Never put credentials in `.tf` files. Use variables, environment variables, or a secrets manager.

```hcl
# WRONG -- will be flagged
provider "aws" {
  access_key = "AKIAIOSFODNN7EXAMPLE"
  secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
}

# RIGHT -- use environment variables or IAM roles
provider "aws" {
  region = var.aws_region
  # Credentials sourced from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
  # or IAM instance profile
}
```

### RES-TAGS: Missing tags/labels

```hcl
# Before (warning)
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

# After (passes)
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"

  tags = merge(var.tags, {
    Name = "${var.cluster_name}-vpc"
  })
}
```

### PLACEHOLDER: Placeholder values

Replace placeholder values with actual resource identifiers or variable references before deployment.

```hcl
# Before (warning)
node_source_details {
  source_type = "IMAGE"
  image_id    = "ocid1.image.oc1..placeholder"  # flagged
}

# After (passes)
node_source_details {
  source_type = "IMAGE"
  image_id    = var.node_image_id
}
```

## EKS-Specific Configuration

### Required Variables

| Variable | Description |
|----------|-------------|
| `cluster_name` | Name of the EKS cluster (no default -- must be provided) |
| `region` | AWS region (default: `us-east-1`) |

### Recommended Pre-Deployment Checks

1. **Verify GPU quota**: EKS GPU node groups use `g5.xlarge` by default. Ensure your AWS account has GPU instance quota in the target region.

2. **Check VPC CIDR availability**: Default CIDR `10.0.0.0/16` must not overlap with existing VPCs in the target region.

3. **Enable EKS service-linked role**: If this is the first EKS cluster in the account, the service-linked role must exist:
   ```bash
   aws iam create-service-linked-role --aws-service-name eks.amazonaws.com
   ```

4. **Verify ECR repository access**: Ensure the node IAM role has ECR read permissions (handled by the module, but verify if using custom policies).

5. **Configure kubectl**: After deployment, configure kubectl using the output command:
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name ocr-local-staging
   ```

### Resource Naming Convention

All EKS resources follow the pattern `{cluster_name}-{resource_suffix}`:

| Resource | Name Pattern |
|----------|-------------|
| VPC | `{cluster_name}-vpc` |
| Subnets | `{cluster_name}-private-{index}`, `{cluster_name}-public-{index}` |
| NAT Gateway | `{cluster_name}-nat` |
| EKS Cluster | `{cluster_name}` |
| CPU Node Group | `{cluster_name}-cpu` |
| GPU Node Group | `{cluster_name}-gpu` |
| ECR Repositories | `ocr-local/coordinator`, `ocr-local/worker` |

### Tag Requirements

All resources must carry these tags:

| Tag Key | Expected Value |
|---------|---------------|
| `Project` | `ocr-local` |
| `ManagedBy` | `terraform` |
| `Environment` | `staging` or `production` |

## CI Integration

Add Terraform validation to your CI pipeline:

```yaml
# .github/workflows/ci.yml
terraform-validate:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    - name: Static validation
      run: python scripts/validate_terraform.py --environment all --strict --json-only
    - name: Plan check (syntax only)
      run: |
        # Install terraform for syntax validation
        curl -fsSL https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_linux_amd64.zip -o tf.zip
        unzip tf.zip && mv terraform /usr/local/bin/
        python scripts/terraform_plan_check.py --environment staging --skip-plan
```

## Known Limitations

- **Placeholder values in OKE module**: The `image_id` fields in the OKE module use `ocid1.image.oc1..placeholder` which must be replaced with actual OCI image OCIDs before deployment.
- **Backend configuration is commented out**: Both staging and production environments have the S3 backend block commented out. Uncomment and configure for team use with remote state.
- **Plan validation without credentials**: `terraform plan` will fail without cloud provider credentials. The plan check tool treats credential errors as warnings, not failures.
- **GPU node pools require quota**: Cloud provider GPU instance quotas must be requested separately and are not validated by these tools.
