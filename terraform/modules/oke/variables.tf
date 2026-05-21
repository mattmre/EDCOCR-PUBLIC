# -----------------------------------------------------------------------------
# Oracle OKE Module Variables
# EDCOCR — Cloud-Native Deployment
# -----------------------------------------------------------------------------

variable "cluster_name" {
  description = "Name of the OKE cluster"
  type        = string
}

variable "compartment_id" {
  description = "OCI compartment OCID"
  type        = string
}

variable "region" {
  description = "OCI region (e.g., us-ashburn-1)"
  type        = string
  default     = "us-ashburn-1"
}

variable "kubernetes_version" {
  description = "Kubernetes version for OKE"
  type        = string
  default     = "v1.29.1"
}

variable "vcn_cidr" {
  description = "CIDR block for the VCN"
  type        = string
  default     = "10.0.0.0/16"
}

variable "service_lb_subnet_cidr" {
  description = "CIDR for the service load balancer subnet"
  type        = string
  default     = "10.0.20.0/24"
}

variable "node_subnet_cidr" {
  description = "CIDR for the node subnet"
  type        = string
  default     = "10.0.10.0/24"
}

variable "api_endpoint_subnet_cidr" {
  description = "CIDR for the Kubernetes API endpoint subnet"
  type        = string
  default     = "10.0.0.0/28"
}

# -- CPU Node Pool ----------------------------------------------------------

variable "cpu_node_shape" {
  description = "OCI shape for CPU nodes"
  type        = string
  default     = "VM.Standard.E4.Flex"
}

variable "cpu_node_ocpus" {
  description = "Number of OCPUs per CPU node (flex shapes)"
  type        = number
  default     = 4
}

variable "cpu_node_memory_gb" {
  description = "Memory in GB per CPU node (flex shapes)"
  type        = number
  default     = 32
}

variable "cpu_node_pool_size" {
  description = "Number of nodes in the CPU pool"
  type        = number
  default     = 2
}

# -- GPU Node Pool ----------------------------------------------------------

variable "gpu_node_shape" {
  description = "OCI shape for GPU nodes (e.g., VM.GPU.A10.1)"
  type        = string
  default     = "VM.GPU.A10.1"
}

variable "gpu_node_pool_size" {
  description = "Number of nodes in the GPU pool"
  type        = number
  default     = 1
}

variable "gpu_boot_volume_gb" {
  description = "Boot volume size in GB for GPU nodes"
  type        = number
  default     = 100
}

# -- Node Images -----------------------------------------------------------
# OKE node pool images are region-specific. You MUST provide a valid OCID
# for your target OCI region. To find the correct OCID:
#   1. OCI Console > Compute > Images
#   2. Filter by "Oracle-Linux-8" (or "Oracle-Linux-8-GPU" for GPU nodes)
#   3. Select the image compatible with your target Kubernetes version
#   4. Copy the OCID from the image details page
#
# Alternatively, use the OCI CLI:
#   oci compute image list --compartment-id <COMPARTMENT_OCID> \
#     --operating-system "Oracle Linux" --operating-system-version "8" \
#     --shape <NODE_SHAPE> --sort-by TIMECREATED --sort-order DESC --limit 5

variable "node_image_id" {
  description = "OCI image OCID for CPU node pool (region-specific, see comments above)"
  type        = string
  # No default -- callers must provide a valid region-specific OCID.
  # Example for us-ashburn-1: "ocid1.image.oc1.iad.aaaaaaa..."
}

variable "gpu_node_image_id" {
  description = "OCI image OCID for GPU node pool (region-specific, see comments above)"
  type        = string
  # No default -- callers must provide a valid region-specific OCID.
  # Example for us-ashburn-1: "ocid1.image.oc1.iad.aaaaaaa..."
}

# -- Tags ------------------------------------------------------------------

variable "freeform_tags" {
  description = "Freeform tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "ocr-local"
    ManagedBy = "terraform"
  }
}
