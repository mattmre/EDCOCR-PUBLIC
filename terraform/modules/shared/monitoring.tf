# -----------------------------------------------------------------------------
# Monitoring Stack (Prometheus + Grafana) via Helm
# EDCOCR — Observability
#
# Deploys kube-prometheus-stack for metrics collection, alerting, and
# dashboard visualization of OCR pipeline performance.
# -----------------------------------------------------------------------------

variable "monitoring_namespace" {
  description = "Kubernetes namespace for monitoring stack"
  type        = string
  default     = "monitoring"
}

variable "prometheus_chart_version" {
  description = "kube-prometheus-stack Helm chart version"
  type        = string
  default     = "58.0.0"
}

variable "prometheus_retention" {
  description = "Prometheus data retention period"
  type        = string
  default     = "15d"
}

variable "prometheus_storage_size" {
  description = "PVC size for Prometheus storage"
  type        = string
  default     = "50Gi"
}

variable "grafana_admin_password" {
  description = "Grafana admin password (set via tfvars, not in code)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "grafana_ingress_enabled" {
  description = "Enable ingress for Grafana"
  type        = bool
  default     = false
}

variable "grafana_ingress_host" {
  description = "Hostname for Grafana ingress"
  type        = string
  default     = "grafana.ocr.example.com"
}

# ---------------------------------------------------------------------------
# Monitoring Namespace
# ---------------------------------------------------------------------------

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = var.monitoring_namespace

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = "ocr-local"
    }
  }
}

# ---------------------------------------------------------------------------
# kube-prometheus-stack Helm Release
# ---------------------------------------------------------------------------

resource "helm_release" "prometheus_stack" {
  name       = "prometheus"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = var.prometheus_chart_version

  # Prometheus server configuration
  set {
    name  = "prometheus.prometheusSpec.retention"
    value = var.prometheus_retention
  }

  set {
    name  = "prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage"
    value = var.prometheus_storage_size
  }

  # Enable ServiceMonitor auto-discovery for OCR-Local
  set {
    name  = "prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues"
    value = "false"
  }

  set {
    name  = "prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues"
    value = "false"
  }

  # Grafana configuration
  set {
    name  = "grafana.enabled"
    value = "true"
  }

  set_sensitive {
    name  = "grafana.adminPassword"
    value = var.grafana_admin_password != "" ? var.grafana_admin_password : "changeme"
  }

  set {
    name  = "grafana.ingress.enabled"
    value = var.grafana_ingress_enabled
  }

  set {
    name  = "grafana.ingress.hosts[0]"
    value = var.grafana_ingress_host
  }

  # Alert manager
  set {
    name  = "alertmanager.enabled"
    value = "true"
  }

  depends_on = [kubernetes_namespace.monitoring]
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "monitoring_namespace" {
  description = "Namespace where monitoring stack is deployed"
  value       = kubernetes_namespace.monitoring.metadata[0].name
}

output "prometheus_chart_version" {
  description = "Installed kube-prometheus-stack chart version"
  value       = helm_release.prometheus_stack.version
}
