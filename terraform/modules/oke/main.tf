# -----------------------------------------------------------------------------
# Oracle OKE Cluster Module
# EDCOCR — Cloud-Native Deployment
#
# Provisions:
#   - VCN with public/private subnets and security lists
#   - OKE cluster with CPU and GPU node pools
#   - OCIR repository for container images
# -----------------------------------------------------------------------------

terraform {
  required_version = ">= 1.5"

  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 5.0, < 6.0"
    }
  }
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_id
}

data "oci_core_services" "all" {
  filter {
    name   = "name"
    values = ["All .* Services In Oracle Services Network"]
    regex  = true
  }
}

# ---------------------------------------------------------------------------
# VCN
# ---------------------------------------------------------------------------

resource "oci_core_vcn" "main" {
  compartment_id = var.compartment_id
  display_name   = "${var.cluster_name}-vcn"
  cidr_blocks    = [var.vcn_cidr]
  dns_label      = "ocrlocal"
  freeform_tags  = var.freeform_tags
}

resource "oci_core_internet_gateway" "main" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-igw"
  enabled        = true
  freeform_tags  = var.freeform_tags
}

resource "oci_core_nat_gateway" "main" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-nat"
  freeform_tags  = var.freeform_tags
}

resource "oci_core_service_gateway" "main" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-sgw"
  freeform_tags  = var.freeform_tags

  services {
    service_id = data.oci_core_services.all.services[0].id
  }
}

# ---------------------------------------------------------------------------
# Route Tables
# ---------------------------------------------------------------------------

resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-public-rt"
  freeform_tags  = var.freeform_tags

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.main.id
  }
}

resource "oci_core_route_table" "private" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-private-rt"
  freeform_tags  = var.freeform_tags

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_nat_gateway.main.id
  }

  route_rules {
    destination       = data.oci_core_services.all.services[0].cidr_block
    destination_type  = "SERVICE_CIDR_BLOCK"
    network_entity_id = oci_core_service_gateway.main.id
  }
}

# ---------------------------------------------------------------------------
# Security Lists
# ---------------------------------------------------------------------------

resource "oci_core_security_list" "node" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-node-sl"
  freeform_tags  = var.freeform_tags

  # Allow all egress
  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
    stateless   = false
  }

  # Allow all intra-VCN traffic
  ingress_security_rules {
    source    = var.vcn_cidr
    protocol  = "all"
    stateless = false
  }
}

resource "oci_core_security_list" "api" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-api-sl"
  freeform_tags  = var.freeform_tags

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
    stateless   = false
  }

  # Kubernetes API access
  ingress_security_rules {
    source   = "0.0.0.0/0"
    protocol = "6" # TCP

    tcp_options {
      min = 6443
      max = 6443
    }

    stateless = false
  }
}

# ---------------------------------------------------------------------------
# Subnets
# ---------------------------------------------------------------------------

resource "oci_core_subnet" "api_endpoint" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.main.id
  display_name               = "${var.cluster_name}-api-subnet"
  cidr_block                 = var.api_endpoint_subnet_cidr
  route_table_id             = oci_core_route_table.public.id
  security_list_ids          = [oci_core_security_list.api.id]
  prohibit_public_ip_on_vnic = false
  freeform_tags              = var.freeform_tags
}

resource "oci_core_subnet" "node" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.main.id
  display_name               = "${var.cluster_name}-node-subnet"
  cidr_block                 = var.node_subnet_cidr
  route_table_id             = oci_core_route_table.private.id
  security_list_ids          = [oci_core_security_list.node.id]
  prohibit_public_ip_on_vnic = true
  freeform_tags              = var.freeform_tags
}

resource "oci_core_subnet" "service_lb" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.main.id
  display_name               = "${var.cluster_name}-svc-lb-subnet"
  cidr_block                 = var.service_lb_subnet_cidr
  route_table_id             = oci_core_route_table.public.id
  security_list_ids          = [oci_core_security_list.node.id]
  prohibit_public_ip_on_vnic = false
  freeform_tags              = var.freeform_tags
}

# ---------------------------------------------------------------------------
# OKE Cluster
# ---------------------------------------------------------------------------

resource "oci_containerengine_cluster" "main" {
  compartment_id     = var.compartment_id
  kubernetes_version = var.kubernetes_version
  name               = var.cluster_name
  vcn_id             = oci_core_vcn.main.id
  freeform_tags      = var.freeform_tags

  endpoint_config {
    is_public_ip_enabled = true
    subnet_id            = oci_core_subnet.api_endpoint.id
  }

  options {
    service_lb_subnet_ids = [oci_core_subnet.service_lb.id]
  }
}

# ---------------------------------------------------------------------------
# CPU Node Pool
# ---------------------------------------------------------------------------

resource "oci_containerengine_node_pool" "cpu" {
  cluster_id         = oci_containerengine_cluster.main.id
  compartment_id     = var.compartment_id
  kubernetes_version = var.kubernetes_version
  name               = "${var.cluster_name}-cpu"
  freeform_tags      = var.freeform_tags

  node_shape = var.cpu_node_shape

  node_shape_config {
    ocpus         = var.cpu_node_ocpus
    memory_in_gbs = var.cpu_node_memory_gb
  }

  node_config_details {
    size = var.cpu_node_pool_size

    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.node.id
    }

    freeform_tags = merge(var.freeform_tags, {
      "ocr-local/node-type" = "cpu"
    })
  }

  node_source_details {
    source_type = "IMAGE"
    image_id    = var.node_image_id
  }

  initial_node_labels {
    key   = "ocr-local/node-type"
    value = "cpu"
  }
}

# ---------------------------------------------------------------------------
# GPU Node Pool
# ---------------------------------------------------------------------------

resource "oci_containerengine_node_pool" "gpu" {
  cluster_id         = oci_containerengine_cluster.main.id
  compartment_id     = var.compartment_id
  kubernetes_version = var.kubernetes_version
  name               = "${var.cluster_name}-gpu"
  freeform_tags      = var.freeform_tags

  node_shape = var.gpu_node_shape

  node_config_details {
    size = var.gpu_node_pool_size

    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.node.id
    }

    freeform_tags = merge(var.freeform_tags, {
      "ocr-local/node-type" = "gpu"
    })
  }

  node_source_details {
    source_type             = "IMAGE"
    image_id                = var.gpu_node_image_id
    boot_volume_size_in_gbs = var.gpu_boot_volume_gb
  }

  initial_node_labels {
    key   = "ocr-local/node-type"
    value = "gpu"
  }

  initial_node_labels {
    key   = "nvidia.com/gpu"
    value = "true"
  }
}
