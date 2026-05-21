# -----------------------------------------------------------------------------
# Staging Environment — OCR-Local Cloud Deployment
#
# Composes cloud provider module + shared infrastructure (KEDA, monitoring).
# Switch the cloud_provider local to target EKS, GKE, or OKE.
# -----------------------------------------------------------------------------

terraform {
  required_version = ">= 1.5"

  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.12, < 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.25, < 3.0"
    }
  }

  # Uncomment the backend block for remote state storage:
  # backend "s3" {
  #   bucket = "ocr-local-terraform-state"
  #   key    = "staging/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "cloud_provider" {
  description = "Cloud provider to deploy to (aws, gcp, oracle)"
  type        = string
  default     = "aws"

  validation {
    condition     = contains(["aws", "gcp", "oracle"], var.cloud_provider)
    error_message = "cloud_provider must be one of: aws, gcp, oracle"
  }
}

variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
  default     = "ocr-local-staging"
}

variable "environment" {
  description = "Environment label"
  type        = string
  default     = "staging"
}

# -- AWS-specific -----------------------------------------------------------

variable "aws_region" {
  description = "AWS region (used when cloud_provider = aws)"
  type        = string
  default     = "us-east-1"
}

# -- GCP-specific -----------------------------------------------------------

variable "gcp_project_id" {
  description = "GCP project ID (used when cloud_provider = gcp)"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP region (used when cloud_provider = gcp)"
  type        = string
  default     = "us-central1"
}

# -- Oracle-specific --------------------------------------------------------

variable "oci_compartment_id" {
  description = "OCI compartment OCID (used when cloud_provider = oracle)"
  type        = string
  default     = ""
}

variable "oci_region" {
  description = "OCI region (used when cloud_provider = oracle)"
  type        = string
  default     = "us-ashburn-1"
}

variable "oci_node_image_id" {
  description = "OCI image OCID for CPU node pool (required when cloud_provider = oracle)"
  type        = string
  default     = ""
}

variable "oci_gpu_node_image_id" {
  description = "OCI image OCID for GPU node pool (required when cloud_provider = oracle)"
  type        = string
  default     = ""
}

# -- Monitoring -------------------------------------------------------------

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
  default     = ""
}

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ocr-local"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# ---------------------------------------------------------------------------
# Cloud Provider Modules (conditional)
# ---------------------------------------------------------------------------

module "eks" {
  source = "../../modules/eks"
  count  = var.cloud_provider == "aws" ? 1 : 0

  cluster_name = var.cluster_name
  region       = var.aws_region

  # Staging: smaller node pools
  cpu_node_min_size     = 1
  cpu_node_max_size     = 5
  cpu_node_desired_size = 1
  gpu_node_min_size     = 0
  gpu_node_max_size     = 4
  gpu_node_desired_size = 0

  tags = {
    Project     = "ocr-local"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

module "gke" {
  source = "../../modules/gke"
  count  = var.cloud_provider == "gcp" ? 1 : 0

  cluster_name = var.cluster_name
  project_id   = var.gcp_project_id
  region       = var.gcp_region

  # Staging: smaller node pools
  cpu_node_min_count = 1
  cpu_node_max_count = 5
  gpu_node_min_count = 0
  gpu_node_max_count = 4

  labels = {
    project     = "ocr-local"
    environment = var.environment
    managed-by  = "terraform"
  }
}

module "oke" {
  source = "../../modules/oke"
  count  = var.cloud_provider == "oracle" ? 1 : 0

  cluster_name   = var.cluster_name
  compartment_id = var.oci_compartment_id
  region         = var.oci_region

  node_image_id     = var.oci_node_image_id
  gpu_node_image_id = var.oci_gpu_node_image_id

  # Staging: smaller node pools
  cpu_node_pool_size = 1
  gpu_node_pool_size = 0

  freeform_tags = {
    Project     = "ocr-local"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ---------------------------------------------------------------------------
# Shared Infrastructure
# ---------------------------------------------------------------------------

module "keda" {
  source = "../../modules/shared"

  keda_replicas          = 1 # Staging: single replica
  keda_log_level         = "debug"
  grafana_admin_password = var.grafana_admin_password
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "cloud_provider" {
  description = "Deployed cloud provider"
  value       = var.cloud_provider
}

output "cluster_name" {
  description = "Name of the deployed cluster"
  value       = var.cluster_name
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value = (
    var.cloud_provider == "aws" ? (length(module.eks) > 0 ? module.eks[0].kubeconfig_command : "") :
    var.cloud_provider == "gcp" ? (length(module.gke) > 0 ? module.gke[0].kubeconfig_command : "") :
    var.cloud_provider == "oracle" ? (length(module.oke) > 0 ? module.oke[0].kubeconfig_command : "") :
    ""
  )
}
