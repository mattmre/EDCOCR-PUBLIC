# -----------------------------------------------------------------------------
# GCP GKE Module Outputs
# -----------------------------------------------------------------------------

output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.main.name
}

output "cluster_endpoint" {
  description = "Endpoint URL for the GKE cluster API server"
  value       = google_container_cluster.main.endpoint
}

output "cluster_ca_certificate" {
  description = "Base64-encoded CA certificate for the cluster"
  value       = google_container_cluster.main.master_auth[0].cluster_ca_certificate
}

output "network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.main.name
}

output "subnet_name" {
  description = "Name of the subnet"
  value       = google_compute_subnetwork.main.name
}

output "node_service_account_email" {
  description = "Email of the node service account"
  value       = google_service_account.gke_nodes.email
}

output "artifact_registry_url" {
  description = "URL of the Artifact Registry repository"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.main.repository_id}"
}

output "kubeconfig_command" {
  description = "gcloud command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.main.name} --region ${var.region} --project ${var.project_id}"
}
