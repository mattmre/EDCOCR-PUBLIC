# -----------------------------------------------------------------------------
# AWS EKS Module Outputs
# -----------------------------------------------------------------------------

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = aws_eks_cluster.main.name
}

output "cluster_endpoint" {
  description = "Endpoint URL for the EKS cluster API server"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_certificate_authority" {
  description = "Base64-encoded certificate authority data for the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "node_role_arn" {
  description = "ARN of the IAM role used by worker nodes"
  value       = aws_iam_role.eks_node.arn
}

output "ecr_repository_urls" {
  description = "URLs of the ECR repositories"
  value       = { for k, v in aws_ecr_repository.images : k => v.repository_url }
}

output "kubeconfig_command" {
  description = "AWS CLI command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${aws_eks_cluster.main.name}"
}
