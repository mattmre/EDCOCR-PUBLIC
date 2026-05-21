# -----------------------------------------------------------------------------
# Oracle OKE Module Outputs
# -----------------------------------------------------------------------------

output "cluster_id" {
  description = "OCID of the OKE cluster"
  value       = oci_containerengine_cluster.main.id
}

output "cluster_name" {
  description = "Name of the OKE cluster"
  value       = oci_containerengine_cluster.main.name
}

output "cluster_kubernetes_version" {
  description = "Kubernetes version running on the cluster"
  value       = oci_containerengine_cluster.main.kubernetes_version
}

output "vcn_id" {
  description = "OCID of the VCN"
  value       = oci_core_vcn.main.id
}

output "node_subnet_id" {
  description = "OCID of the node subnet"
  value       = oci_core_subnet.node.id
}

output "kubeconfig_command" {
  description = "OCI CLI command to configure kubectl"
  value       = "oci ce cluster create-kubeconfig --cluster-id ${oci_containerengine_cluster.main.id} --region ${var.region}"
}
