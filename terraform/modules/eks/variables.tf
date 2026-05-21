# -----------------------------------------------------------------------------
# AWS EKS Module Variables
# EDCOCR — Cloud-Native Deployment
# -----------------------------------------------------------------------------

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "cluster_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.29"
}

variable "region" {
  description = "AWS region for the cluster"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "availability_zones" {
  description = "Availability zones for subnet distribution"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

# -- CPU Node Pool ----------------------------------------------------------

variable "cpu_node_instance_types" {
  description = "EC2 instance types for the CPU node pool"
  type        = list(string)
  default     = ["m6i.2xlarge"]
}

variable "cpu_node_min_size" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 1
}

variable "cpu_node_max_size" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 10
}

variable "cpu_node_desired_size" {
  description = "Desired number of CPU nodes"
  type        = number
  default     = 2
}

# -- GPU Node Pool ----------------------------------------------------------

variable "gpu_node_instance_types" {
  description = "EC2 instance types for the GPU node pool (must have NVIDIA GPUs)"
  type        = list(string)
  default     = ["g5.xlarge"]
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 8
}

variable "gpu_node_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_node_disk_size" {
  description = "Disk size in GiB for GPU nodes (needs space for models)"
  type        = number
  default     = 100
}

# -- ECR -------------------------------------------------------------------

variable "ecr_repository_names" {
  description = "ECR repository names for OCR container images"
  type        = list(string)
  default     = ["ocr-local/coordinator", "ocr-local/worker"]
}

# -- Tags ------------------------------------------------------------------

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "ocr-local"
    ManagedBy   = "terraform"
    Environment = "staging"
  }
}
