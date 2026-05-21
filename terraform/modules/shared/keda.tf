# -----------------------------------------------------------------------------
# KEDA Deployment via Helm
# EDCOCR — Event-Driven Autoscaling
#
# Deploys KEDA into the Kubernetes cluster for queue-length-based autoscaling
# of OCR GPU and CPU worker pods.
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
}

variable "keda_namespace" {
  description = "Kubernetes namespace for KEDA"
  type        = string
  default     = "keda"
}

variable "keda_chart_version" {
  description = "KEDA Helm chart version"
  type        = string
  default     = "2.14.0"
}

variable "keda_replicas" {
  description = "Number of KEDA operator replicas"
  type        = number
  default     = 2
}

variable "keda_log_level" {
  description = "Log level for KEDA operator (debug, info, error)"
  type        = string
  default     = "info"
}

variable "keda_metrics_server_enabled" {
  description = "Enable KEDA metrics server (for HPA integration)"
  type        = bool
  default     = true
}

# ---------------------------------------------------------------------------
# KEDA Namespace
# ---------------------------------------------------------------------------

resource "kubernetes_namespace" "keda" {
  metadata {
    name = var.keda_namespace

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = "ocr-local"
    }
  }
}

# ---------------------------------------------------------------------------
# KEDA Helm Release
# ---------------------------------------------------------------------------

resource "helm_release" "keda" {
  name       = "keda"
  namespace  = kubernetes_namespace.keda.metadata[0].name
  repository = "https://kedacore.github.io/charts"
  chart      = "keda"
  version    = var.keda_chart_version

  set {
    name  = "operator.replicaCount"
    value = var.keda_replicas
  }

  set {
    name  = "logging.operator.level"
    value = var.keda_log_level
  }

  set {
    name  = "metricsServer.enabled"
    value = var.keda_metrics_server_enabled
  }

  # Pod disruption budget for HA
  set {
    name  = "podDisruptionBudget.operator.minAvailable"
    value = "1"
  }

  # Resource requests for the operator
  set {
    name  = "resources.operator.requests.cpu"
    value = "100m"
  }

  set {
    name  = "resources.operator.requests.memory"
    value = "128Mi"
  }

  set {
    name  = "resources.operator.limits.cpu"
    value = "500m"
  }

  set {
    name  = "resources.operator.limits.memory"
    value = "256Mi"
  }

  depends_on = [kubernetes_namespace.keda]
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "keda_namespace" {
  description = "Namespace where KEDA is deployed"
  value       = kubernetes_namespace.keda.metadata[0].name
}

output "keda_chart_version" {
  description = "Installed KEDA chart version"
  value       = helm_release.keda.version
}
