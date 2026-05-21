{{/*
Expand the name of the chart.
*/}}
{{- define "ocr-local.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this
(by the DNS naming spec). If release name contains chart name it will be used
as a full name.
*/}}
{{- define "ocr-local.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "ocr-local.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/name: {{ include "ocr-local.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: ocr-local
{{- end }}

{{/*
Selector labels -- the minimal set used in matchLabels.
*/}}
{{- define "ocr-local.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ocr-local.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Resolve coordinator container image (repository:tag).
Falls back to Chart.appVersion when .Values.image.coordinator.tag is empty.
*/}}
{{- define "ocr-local.coordinatorImage" -}}
{{- $tag := default .Chart.AppVersion .Values.image.coordinator.tag -}}
{{- printf "%s:%s" .Values.image.coordinator.repository $tag -}}
{{- end }}

{{/*
Resolve worker container image (repository:tag).
Falls back to Chart.appVersion when .Values.image.worker.tag is empty.
*/}}
{{- define "ocr-local.workerImage" -}}
{{- $tag := default .Chart.AppVersion .Values.image.worker.tag -}}
{{- printf "%s:%s" .Values.image.worker.repository $tag -}}
{{- end }}

{{/*
Resolve NLP worker container image (repository:tag).
Falls back to Chart.appVersion when .Values.image.nlpWorker.tag is empty.
Used by nlp-gpu-worker deployment for VLM inference workloads.
*/}}
{{- define "ocr-local.nlpWorkerImage" -}}
{{- $tag := default .Chart.AppVersion .Values.image.nlpWorker.tag -}}
{{- printf "%s:%s" .Values.image.nlpWorker.repository $tag -}}
{{- end }}

{{/*
Resolve the ServiceAccount name for pod identity.
Uses the chart fullname when rbac.create is true, otherwise "default".
*/}}
{{- define "ocr-local.serviceAccountName" -}}
{{- if .Values.rbac.create -}}
{{ include "ocr-local.fullname" . }}
{{- else -}}
default
{{- end -}}
{{- end }}

{{/*
Resolve LayoutLMv3 worker container image (repository:tag).
Falls back to Chart.appVersion when .Values.image.layoutlmWorker.tag is empty.
Used by layoutlm-worker deployment for document understanding workloads.
*/}}
{{- define "ocr-local.layoutlmWorkerImage" -}}
{{- $tag := default .Chart.AppVersion .Values.image.layoutlmWorker.tag -}}
{{- printf "%s:%s" .Values.image.layoutlmWorker.repository $tag -}}
{{- end }}

{{/*
Resolve frontend operator-console container image (repository:tag).
Falls back to Chart.appVersion when .Values.image.frontend.tag is empty.
*/}}
{{- define "ocr-local.frontendImage" -}}
{{- $tag := default .Chart.AppVersion .Values.image.frontend.tag -}}
{{- printf "%s:%s" .Values.image.frontend.repository $tag -}}
{{- end }}
