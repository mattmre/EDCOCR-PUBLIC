# -----------------------------------------------------------------------------
# GCP GKE Cluster Module
# EDCOCR — Cloud-Native Deployment
#
# Provisions:
#   - VPC network with subnet and secondary ranges
#   - GKE cluster with CPU and GPU node pools
#   - Service accounts with least-privilege IAM
#   - Artifact Registry for container images
#   - GPU driver auto-installation
# -----------------------------------------------------------------------------

terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0, < 6.0"
    }
  }
}

# ---------------------------------------------------------------------------
# VPC Network
# ---------------------------------------------------------------------------

resource "google_compute_network" "main" {
  name                    = var.network_name
  project                 = var.project_id
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "main" {
  name          = "${var.cluster_name}-subnet"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.main.id
  ip_cidr_range = var.subnet_cidr

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }

  private_ip_google_access = true
}

# ---------------------------------------------------------------------------
# Cloud Router + NAT (for private nodes)
# ---------------------------------------------------------------------------

resource "google_compute_router" "main" {
  name    = "${var.cluster_name}-router"
  project = var.project_id
  region  = var.region
  network = google_compute_network.main.id
}

resource "google_compute_router_nat" "main" {
  name                               = "${var.cluster_name}-nat"
  project                            = var.project_id
  region                             = var.region
  router                             = google_compute_router.main.name
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# ---------------------------------------------------------------------------
# Service Account for GKE Nodes
# ---------------------------------------------------------------------------

resource "google_service_account" "gke_nodes" {
  account_id   = "${var.cluster_name}-nodes"
  project      = var.project_id
  display_name = "GKE node SA for ${var.cluster_name}"
}

resource "google_project_iam_member" "gke_node_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_node_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_node_ar_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# ---------------------------------------------------------------------------
# GKE Cluster
# ---------------------------------------------------------------------------

resource "google_container_cluster" "main" {
  name     = var.cluster_name
  project  = var.project_id
  location = var.region

  network    = google_compute_network.main.id
  subnetwork = google_compute_subnetwork.main.id

  # Use a separately managed node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  dynamic "release_channel" {
    for_each = var.kubernetes_version == "" ? [1] : []
    content {
      channel = var.release_channel
    }
  }

  min_master_version = var.kubernetes_version != "" ? var.kubernetes_version : null

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  resource_labels = var.labels
}

# ---------------------------------------------------------------------------
# CPU Node Pool
# ---------------------------------------------------------------------------

resource "google_container_node_pool" "cpu" {
  name     = "${var.cluster_name}-cpu"
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.main.name

  autoscaling {
    min_node_count = var.cpu_node_min_count
    max_node_count = var.cpu_node_max_count
  }

  node_config {
    machine_type    = var.cpu_machine_type
    service_account = google_service_account.gke_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = merge(var.labels, {
      "ocr-local/node-type" = "cpu"
    })

    metadata = {
      disable-legacy-endpoints = "true"
    }

    shielded_instance_config {
      enable_secure_boot = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# ---------------------------------------------------------------------------
# GPU Node Pool
# ---------------------------------------------------------------------------

resource "google_container_node_pool" "gpu" {
  name     = "${var.cluster_name}-gpu"
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.main.name

  autoscaling {
    min_node_count = var.gpu_node_min_count
    max_node_count = var.gpu_node_max_count
  }

  node_config {
    machine_type    = var.gpu_machine_type
    disk_size_gb    = var.gpu_disk_size_gb
    service_account = google_service_account.gke_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    labels = merge(var.labels, {
      "ocr-local/node-type" = "gpu"
    })

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# ---------------------------------------------------------------------------
# Artifact Registry
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "main" {
  repository_id = var.artifact_registry_name
  project       = var.project_id
  location      = var.region
  format        = "DOCKER"
  description   = "Container images for OCR-Local pipeline"

  labels = var.labels
}
