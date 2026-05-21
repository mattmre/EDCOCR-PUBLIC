# -----------------------------------------------------------------------------
# GCP GKE Module Variables
# EDCOCR — Cloud-Native Deployment
# -----------------------------------------------------------------------------

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the cluster"
  type        = string
  default     = "us-central1"
}

variable "network_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "ocr-local-vpc"
}

variable "subnet_cidr" {
  description = "CIDR range for the primary subnet"
  type        = string
  default     = "10.0.0.0/20"
}

variable "pods_cidr" {
  description = "Secondary CIDR range for pods"
  type        = string
  default     = "10.4.0.0/14"
}

variable "services_cidr" {
  description = "Secondary CIDR range for services"
  type        = string
  default     = "10.8.0.0/20"
}

variable "kubernetes_version" {
  description = "Kubernetes version for GKE (use release channel if unset)"
  type        = string
  default     = ""
}

variable "release_channel" {
  description = "GKE release channel (RAPID, REGULAR, STABLE)"
  type        = string
  default     = "REGULAR"
}

# -- CPU Node Pool ----------------------------------------------------------

variable "cpu_machine_type" {
  description = "Machine type for CPU node pool"
  type        = string
  default     = "e2-standard-8"
}

variable "cpu_node_min_count" {
  description = "Minimum number of CPU nodes per zone"
  type        = number
  default     = 1
}

variable "cpu_node_max_count" {
  description = "Maximum number of CPU nodes per zone"
  type        = number
  default     = 10
}

# -- GPU Node Pool ----------------------------------------------------------

variable "gpu_machine_type" {
  description = "Machine type for GPU node pool"
  type        = string
  default     = "n1-standard-8"
}

variable "gpu_type" {
  description = "GPU accelerator type (e.g., nvidia-tesla-t4, nvidia-l4)"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs per node"
  type        = number
  default     = 1
}

variable "gpu_node_min_count" {
  description = "Minimum number of GPU nodes per zone"
  type        = number
  default     = 0
}

variable "gpu_node_max_count" {
  description = "Maximum number of GPU nodes per zone"
  type        = number
  default     = 8
}

variable "gpu_disk_size_gb" {
  description = "Boot disk size in GB for GPU nodes"
  type        = number
  default     = 100
}

# -- Artifact Registry -----------------------------------------------------

variable "artifact_registry_name" {
  description = "Name of the Artifact Registry repository"
  type        = string
  default     = "ocr-local"
}

# -- Labels ----------------------------------------------------------------

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    project    = "ocr-local"
    managed-by = "terraform"
  }
}
