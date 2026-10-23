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
Resolve a repository image from an optional immutable digest.  Production
overlays can require digests without forcing local development to know a
registry-specific sha256 value.
*/}}
{{- define "ocr-local.imageRef" -}}
{{- $image := .image -}}
{{- if and .requireDigest (not $image.digest) -}}
{{- fail (printf "image digest is required for %s when image.requireDigest=true" .name) -}}
{{- end -}}
{{- if $image.digest -}}
{{- if not (regexMatch "^sha256:[0-9a-fA-F]{64}$" $image.digest) -}}
{{- fail (printf "image digest for %s must be sha256:<64 hex chars>" .name) -}}
{{- end -}}
{{- printf "%s@%s" $image.repository $image.digest -}}
{{- else -}}
{{- printf "%s:%s" $image.repository (default .defaultTag $image.tag) -}}
{{- end -}}
{{- end }}

{{/* Resolve coordinator container image. */}}
{{- define "ocr-local.coordinatorImage" -}}
{{ include "ocr-local.imageRef" (dict "image" .Values.image.coordinator "defaultTag" .Chart.AppVersion "requireDigest" .Values.image.requireDigest "name" "coordinator") }}
{{- end }}

{{/* Resolve worker container image. */}}
{{- define "ocr-local.workerImage" -}}
{{ include "ocr-local.imageRef" (dict "image" .Values.image.worker "defaultTag" .Chart.AppVersion "requireDigest" .Values.image.requireDigest "name" "worker") }}
{{- end }}

{{/* Resolve the native Paddle/Tesseract CPU OCR worker image. */}}
{{- define "ocr-local.cpuOcrWorkerImage" -}}
{{ include "ocr-local.imageRef" (dict "image" .Values.image.cpuOcrWorker "defaultTag" .Chart.AppVersion "requireDigest" .Values.image.requireDigest "name" "cpuOcrWorker") }}
{{- end }}

{{/* Resolve the capability-owned Tesseract worker image. */}}
{{- define "ocr-local.tesseractWorkerImage" -}}
{{ include "ocr-local.imageRef" (dict "image" .Values.image.tesseractWorker "defaultTag" .Chart.AppVersion "requireDigest" .Values.image.requireDigest "name" "tesseractWorker") }}
{{- end }}

{{/* Stable pod DNS endpoints used by Redis Sentinel-aware clients. */}}
{{- define "ocr-local.redisSentinelNodes" -}}
{{- $nodes := list -}}
{{- range $index := until (int .Values.redis.sentinel.sentinelReplicas) -}}
{{- $nodes = append $nodes (printf "%s-redis-sentinel-%d.%s-redis-sentinel:26379" $.Release.Name $index $.Release.Name) -}}
{{- end -}}
{{- join "," $nodes -}}
{{- end }}

{{/* Celery accepts semicolon-delimited Sentinel URLs. */}}
{{- define "ocr-local.redisSentinelResultBackend" -}}
{{- $nodes := list -}}
{{- $root := .root -}}
{{- $password := .password -}}
{{- $db := int .db -}}
{{- range $index := until (int $root.Values.redis.sentinel.sentinelReplicas) -}}
{{- $nodes = append $nodes (printf "sentinel://:%s@%s-redis-sentinel-%d.%s-redis-sentinel:26379/%d" ($password | urlquery) $root.Release.Name $index $root.Release.Name $db) -}}
{{- end -}}
{{- join ";" $nodes -}}
{{- end }}

{{/* Resolve NLP worker container image. */}}
{{- define "ocr-local.nlpWorkerImage" -}}
{{ include "ocr-local.imageRef" (dict "image" .Values.image.nlpWorker "defaultTag" .Chart.AppVersion "requireDigest" .Values.image.requireDigest "name" "nlpWorker") }}
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

{{/* IAM identity belongs to control-plane pods, never presigned workers. */}}
{{- define "ocr-local.controlPlaneServiceAccountName" -}}
{{- if and (eq .Values.storage.backend "s3") (eq .Values.storage.s3.authMode "iam") -}}
{{ required "IAM mode requires storage.s3.controlPlaneServiceAccount" .Values.storage.s3.controlPlaneServiceAccount }}
{{- else -}}
{{ include "ocr-local.serviceAccountName" . }}
{{- end -}}
{{- end }}

{{/*
Optional registry credentials for the application images.  Keep this at the
Pod level so init containers and the main container use the same pull policy.
*/}}
{{- define "ocr-local.imagePullSecrets" -}}
{{- with .Values.image.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/* Resolve LayoutLMv3 worker container image. */}}
{{- define "ocr-local.layoutlmWorkerImage" -}}
{{ include "ocr-local.imageRef" (dict "image" .Values.image.layoutlmWorker "defaultTag" .Chart.AppVersion "requireDigest" .Values.image.requireDigest "name" "layoutlmWorker") }}
{{- end }}

{{/* Resolve frontend operator-console container image. */}}
{{- define "ocr-local.frontendImage" -}}
{{ include "ocr-local.imageRef" (dict "image" .Values.image.frontend "defaultTag" .Chart.AppVersion "requireDigest" .Values.image.requireDigest "name" "frontend") }}
{{- end }}

{{/* Resolve the FastAPI image (the Django coordinator is not the API). */}}
{{- define "ocr-local.apiImage" -}}
{{ include "ocr-local.imageRef" (dict "image" .Values.image.api "defaultTag" .Chart.AppVersion "requireDigest" .Values.image.requireDigest "name" "api") }}
{{- end }}

{{/*
Worker envFrom + env secret keys (S2-02 / R2 residual).

When workerSecurity.leastSecretEnv is true (default), workers envFrom only the
release ConfigMap and receive explicit secretKeyRef entries for broker/DB/Redis/
Django — never the full Secret (Postgres root, Flower password, Erlang cookie,
optional S3 unless storage.workerInjectS3Credentials=true).

When leastSecretEnv is false, legacy full secretRef envFrom is used.
*/}}
{{- define "ocr-local.workerEnvFrom" -}}
{{- if and (eq .Values.storage.backend "s3") .Values.storage.s3.usePresignedUrls (not .Values.storage.workerInjectS3Credentials) (not .Values.workerSecurity.leastSecretEnv) -}}
{{- fail "credential-free presigned workers require workerSecurity.leastSecretEnv=true; a full secretRef would expose S3 credentials" -}}
{{- end -}}
envFrom:
  - configMapRef:
      name: {{ .Release.Name }}-config
{{- if not .Values.workerSecurity.leastSecretEnv }}
  - secretRef:
      name: {{ .Release.Name }}-secret
{{- end }}
{{- if .Values.federation.enabled }}
  - configMapRef:
      name: {{ .Release.Name }}-federation-overlay
{{- end }}
{{- end }}

{{- define "ocr-local.workerSecretEnv" -}}
{{- if and (eq .Values.storage.backend "s3") .Values.storage.s3.usePresignedUrls (not .Values.storage.workerInjectS3Credentials) }}
- name: S3_PRESIGNED_ONLY_WORKER
  value: "true"
- name: AWS_EC2_METADATA_DISABLED
  value: "true"
{{- end }}
{{- if .Values.workerSecurity.leastSecretEnv }}
- name: DJANGO_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: DJANGO_SECRET_KEY
- name: RABBITMQ_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: RABBITMQ_PASSWORD
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: REDIS_PASSWORD
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: DATABASE_URL
- name: CELERY_BROKER_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: CELERY_BROKER_URL
- name: CELERY_RESULT_BACKEND
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: CELERY_RESULT_BACKEND
- name: CELERY_TASK_SIGNING_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: CELERY_TASK_SIGNING_KEY
- name: PII_ENCRYPTION_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: PII_ENCRYPTION_KEY
- name: WEBHOOK_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: WEBHOOK_SECRET_KEY
- name: REDIS_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: REDIS_URL
{{- if and (eq .Values.storage.backend "s3") (or .Values.storage.workerInjectS3Credentials .Values.storage.s3.usePresignedUrls) }}
- name: S3_ENDPOINT
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_ENDPOINT
- name: S3_BUCKET
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_BUCKET
{{- end }}
{{- if and (eq .Values.storage.backend "s3") .Values.storage.workerInjectS3Credentials }}
{{- if eq .Values.storage.s3.authMode "static" }}
- name: S3_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_ACCESS_KEY
- name: S3_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_SECRET_KEY
- name: S3_SESSION_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_SESSION_TOKEN
      optional: true
{{- end }}
{{- end }}
{{- if and (eq .Values.storage.backend "s3") (or .Values.storage.workerInjectS3Credentials .Values.storage.s3.usePresignedUrls) }}
- name: S3_REGION
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_REGION
      optional: true
{{- end }}
{{- if and (eq .Values.storage.backend "s3") .Values.storage.s3.usePresignedUrls }}
- name: S3_PRESIGNED_TASK_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_PRESIGNED_TASK_KEY
{{- end }}
{{- end }}
{{- end }}

{{/*
Coordinator / beat / celery-coordinator least-secret env (S1-05 residual).

When workerSecurity.leastSecretEnv is true (default), these pods envFrom only
the ConfigMap and receive explicit secretKeyRef for Django/DB/broker/Redis/
metrics/S3 — never the full Secret (Postgres root password alone, Flower
password, Erlang cookie as ambient env). Migrate init uses the same keys.
*/}}
{{- define "ocr-local.coordinatorEnvFrom" -}}
envFrom:
  - configMapRef:
      name: {{ .Release.Name }}-config
{{- if not .Values.workerSecurity.leastSecretEnv }}
  - secretRef:
      name: {{ .Release.Name }}-secret
{{- end }}

{{- if .Values.federation.enabled }}
  - configMapRef:
      name: {{ .Release.Name }}-federation-overlay
{{- end }}
{{- end }}

{{/*
Consumers wait for the coordinator web pod's sole migration writer.  Django's
"migrate --check" only reads migration state; it never applies migrations.
The bounded init restart keeps new workers and Beat from consuming messages
against an older schema during a Helm install or upgrade.
*/}}
{{- define "ocr-local.schemaReadinessInit" -}}
- name: wait-for-coordinator-schema
  image: {{ include "ocr-local.coordinatorImage" .root }}
  imagePullPolicy: {{ .root.Values.image.coordinator.pullPolicy | default "IfNotPresent" }}
  command: ["python", "-c"]
  args:
    - |
      import subprocess
      import sys
      import time

      deadline = time.monotonic() + 300
      while True:
          remaining = deadline - time.monotonic()
          if remaining <= 0:
              print("coordinator schema unavailable after 300 seconds", file=sys.stderr)
              raise SystemExit(1)
          try:
              check = subprocess.run(
                  [sys.executable, "manage.py", "migrate", "--check", "--noinput"],
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                  timeout=min(10, remaining), check=False,
              )
          except subprocess.TimeoutExpired:
              check = None
          if check is not None and check.returncode == 0:
              raise SystemExit(0)
          time.sleep(min(5, max(0, deadline - time.monotonic())))
  {{- if eq .envKind "worker" }}
  {{- include "ocr-local.workerEnvFrom" .root | nindent 2 }}
  env:
    {{- include "ocr-local.workerSecretEnv" .root | nindent 4 }}
  {{- else }}
  {{- include "ocr-local.coordinatorEnvFrom" .root | nindent 2 }}
  env:
    {{- include "ocr-local.coordinatorSecretEnv" .root | nindent 4 }}
  {{- end }}
  {{- if eq (include "ocr-local.hasClientCaVolumes" .root) "true" }}
  volumeMounts:
    {{- include "ocr-local.tlsCaVolumeMount" .root | nindent 4 }}
  {{- end }}
  {{- with .securityContext }}
  securityContext:
    {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end }}

{{/*
Coordinator web replicas are concurrent. Hold a separate PostgreSQL session
advisory lock for the entire migration command so only one pod performs DDL.
The session has bounded connect/query timeouts; lock acquisition has a bounded
monotonic deadline. Other consumers run only the read-only readiness gate.
*/}}
{{- define "ocr-local.serializedMigrationInit" -}}
- name: migrate
  image: {{ include "ocr-local.coordinatorImage" . }}
  imagePullPolicy: {{ .Values.image.coordinator.pullPolicy | default "IfNotPresent" }}
  command: ["python", "-c"]
  args:
    - |
      import hashlib
      import sys
      import time

      import django
      from django.conf import settings
      from django.core.management import call_command
      from django.db import OperationalError, connections

      django.setup()
      lock_identity = (
          str(settings.DATABASES["default"]["NAME"]) + ":"
          + str(settings.DATABASE_SCHEMA) + ":ocr-local:migrate"
      ).encode("utf-8")
      lock_key = int.from_bytes(
          hashlib.sha256(lock_identity).digest()[:8], "big", signed=True,
      )
      deadline = time.monotonic() + 300
      owner = None
      try:
          while time.monotonic() < deadline:
              candidate = connections["default"].copy(alias="migration_lock")
              original_options = candidate.settings_dict.get("OPTIONS", {})
              candidate.settings_dict["OPTIONS"] = {
                  **original_options,
                  "connect_timeout": 10,
                  "options": (
                      str(original_options.get("options", ""))
                      + " -c statement_timeout=10000 -c lock_timeout=10000"
                  ).strip(),
              }
              try:
                  with candidate.cursor() as cursor:
                      cursor.execute("SELECT pg_try_advisory_lock(%s)", [lock_key])
                      if cursor.fetchone()[0]:
                          owner = candidate
                          break
              except OperationalError:
                  pass
              finally:
                  if owner is not candidate:
                      candidate.close()
              time.sleep(min(5, max(0, deadline - time.monotonic())))
          if owner is None:
              print("coordinator migration lock unavailable after 300 seconds", file=sys.stderr)
              raise SystemExit(1)
          call_command("migrate", interactive=False)
      finally:
          if owner is not None:
              try:
                  with owner.cursor() as cursor:
                      cursor.execute("SELECT pg_advisory_unlock(%s)", [lock_key])
              finally:
                  owner.close()
  {{- include "ocr-local.coordinatorEnvFrom" . | nindent 2 }}
  env:
    {{- include "ocr-local.coordinatorSecretEnv" . | nindent 4 }}
  {{- if eq (include "ocr-local.hasClientCaVolumes" .) "true" }}
  volumeMounts:
    {{- include "ocr-local.tlsCaVolumeMount" . | nindent 4 }}
  {{- end }}
  {{- with .Values.coordinator.containerSecurityContext }}
  securityContext:
    {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end }}

{{- define "ocr-local.coordinatorSecretEnv" -}}
{{- if .Values.workerSecurity.leastSecretEnv }}
- name: DJANGO_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: DJANGO_SECRET_KEY
- name: OCR_COORDINATOR_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: OCR_COORDINATOR_API_KEY
- name: RABBITMQ_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: RABBITMQ_PASSWORD
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: REDIS_PASSWORD
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: DATABASE_URL
- name: CELERY_BROKER_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: CELERY_BROKER_URL
- name: CELERY_RESULT_BACKEND
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: CELERY_RESULT_BACKEND
- name: CELERY_TASK_SIGNING_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: CELERY_TASK_SIGNING_KEY
- name: REDIS_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: REDIS_URL
- name: WEBHOOK_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: WEBHOOK_SECRET_KEY
- name: PII_ENCRYPTION_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: PII_ENCRYPTION_KEY
{{- if .Values.secrets.metricsApiKey }}
- name: METRICS_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: METRICS_API_KEY
{{- end }}

{{- if eq .Values.storage.backend "s3" }}
- name: S3_ENDPOINT
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_ENDPOINT
- name: S3_BUCKET
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_BUCKET
{{- if eq .Values.storage.s3.authMode "static" }}
- name: S3_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_ACCESS_KEY
- name: S3_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_SECRET_KEY
- name: S3_SESSION_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_SESSION_TOKEN
      optional: true
{{- end }}
- name: S3_REGION
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_REGION
      optional: true
{{- end }}
{{- if and (eq .Values.storage.backend "s3") .Values.storage.s3.usePresignedUrls }}
- name: S3_PRESIGNED_TASK_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secret
      key: S3_PRESIGNED_TASK_KEY
{{- end }}
{{- end }}
{{- end }}

{{/* Failure-domain spread shared by replicated provider control/data planes. */}}
{{- define "ocr-local.haScheduling" -}}
{{- $root := .root -}}
{{- $component := .component -}}
{{- if $root.Values.haScheduling.enabled }}
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: kubernetes.io/hostname
    whenUnsatisfiable: {{ $root.Values.haScheduling.hostnameWhenUnsatisfiable }}
    labelSelector:
      matchLabels:
        {{- include "ocr-local.selectorLabels" $root | nindent 8 }}
        app.kubernetes.io/component: {{ $component }}
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: {{ $root.Values.haScheduling.zoneWhenUnsatisfiable }}
    labelSelector:
      matchLabels:
        {{- include "ocr-local.selectorLabels" $root | nindent 8 }}
        app.kubernetes.io/component: {{ $component }}
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          topologyKey: kubernetes.io/hostname
          labelSelector:
            matchLabels:
              {{- include "ocr-local.selectorLabels" $root | nindent 14 }}
              app.kubernetes.io/component: {{ $component }}
{{- end }}
{{- end }}

{{/* True when application pods require at least one dependency CA volume. */}}
{{- define "ocr-local.hasClientCaVolumes" -}}
{{- if or .Values.tls.enabled (and .Values.postgresql.clientTls.enabled .Values.postgresql.clientTls.caSecretName) (and .Values.rabbitmq.clientTls.enabled .Values.rabbitmq.clientTls.caSecretName) (and .Values.redis.clientTls.enabled .Values.redis.clientTls.caSecretName) .Values.storage.s3.caSecretName -}}true{{- else -}}false{{- end -}}
{{- end }}

{{/*
When an enabled KEDA worker queue is configured to scale to zero, coordinator
readiness must allow the idle state so the API can accept work that activates
the scaler. The detailed health response still reports workers as
not_required while all durable dependencies remain mandatory.
*/}}
{{- define "ocr-local.allowZeroWorkerReadiness" -}}
{{- $allow := or
  (and .Values.gpuWorker.enabled .Values.gpuWorker.autoscaling.enabled (eq (int .Values.gpuWorker.autoscaling.minReplicas) 0))
  (and .Values.cpuWorker.autoscaling.enabled (eq (int .Values.cpuWorker.autoscaling.minReplicas) 0))
  (and .Values.cpuOcrWorker.enabled .Values.cpuOcrWorker.autoscaling.enabled (eq (int .Values.cpuOcrWorker.autoscaling.minReplicas) 0))
  (and .Values.tesseractWorker.enabled .Values.tesseractWorker.autoscaling.enabled (eq (int .Values.tesseractWorker.autoscaling.minReplicas) 0))
  (and .Values.layoutCpuWorker.enabled .Values.layoutCpuWorker.autoscaling.enabled (eq (int .Values.layoutCpuWorker.autoscaling.minReplicas) 0))
  (and .Values.nlpGpuWorker.enabled .Values.nlpGpuWorker.autoscaling.enabled (eq (int .Values.nlpGpuWorker.autoscaling.minReplicas) 0))
  (and .Values.layoutlmWorker.enabled .Values.layoutlmWorker.autoscaling.enabled (eq (int .Values.layoutlmWorker.autoscaling.minReplicas) 0))
-}}
{{- if $allow -}}true{{- else -}}false{{- end -}}
{{- end }}

{{/*
Mount the chart-managed shared CA and independently configured external
dependency CAs. ConfigMap TLS switches are otherwise inert if client
containers cannot read their referenced trust roots.
*/}}
{{- define "ocr-local.tlsCaVolumeMount" -}}
{{- if .Values.tls.enabled }}
- name: tls-ca
  mountPath: /etc/tls
  readOnly: true
{{- end }}
{{- if and .Values.postgresql.clientTls.enabled .Values.postgresql.clientTls.caSecretName }}
- name: postgresql-client-ca
  mountPath: {{ .Values.postgresql.clientTls.caMountPath }}
  subPath: {{ .Values.postgresql.clientTls.caSecretKey }}
  readOnly: true
{{- end }}
{{- if and .Values.rabbitmq.clientTls.enabled .Values.rabbitmq.clientTls.caSecretName }}
- name: rabbitmq-client-ca
  mountPath: {{ .Values.rabbitmq.clientTls.caMountPath }}
  subPath: {{ .Values.rabbitmq.clientTls.caSecretKey }}
  readOnly: true
{{- end }}
{{- if and .Values.redis.clientTls.enabled .Values.redis.clientTls.caSecretName }}
- name: redis-client-ca
  mountPath: {{ .Values.redis.clientTls.caMountPath }}
  subPath: {{ .Values.redis.clientTls.caSecretKey }}
  readOnly: true
{{- end }}
{{- if .Values.storage.s3.caSecretName }}
- name: s3-ca
  mountPath: {{ .Values.storage.s3.caMountPath }}
  subPath: {{ .Values.storage.s3.caSecretKey }}
  readOnly: true
{{- end }}
{{- end }}

{{- define "ocr-local.tlsCaVolume" -}}
{{- if .Values.tls.enabled }}
- name: tls-ca
  secret:
    secretName: {{ include "ocr-local.fullname" . }}-ca-secret
    defaultMode: 0440
{{- end }}
{{- if and .Values.postgresql.clientTls.enabled .Values.postgresql.clientTls.caSecretName }}
- name: postgresql-client-ca
  secret:
    secretName: {{ .Values.postgresql.clientTls.caSecretName }}
    items:
      - key: {{ .Values.postgresql.clientTls.caSecretKey }}
        path: {{ .Values.postgresql.clientTls.caSecretKey }}
    defaultMode: 0440
{{- end }}
{{- if and .Values.rabbitmq.clientTls.enabled .Values.rabbitmq.clientTls.caSecretName }}
- name: rabbitmq-client-ca
  secret:
    secretName: {{ .Values.rabbitmq.clientTls.caSecretName }}
    items:
      - key: {{ .Values.rabbitmq.clientTls.caSecretKey }}
        path: {{ .Values.rabbitmq.clientTls.caSecretKey }}
    defaultMode: 0440
{{- end }}
{{- if and .Values.redis.clientTls.enabled .Values.redis.clientTls.caSecretName }}
- name: redis-client-ca
  secret:
    secretName: {{ .Values.redis.clientTls.caSecretName }}
    items:
      - key: {{ .Values.redis.clientTls.caSecretKey }}
        path: {{ .Values.redis.clientTls.caSecretKey }}
    defaultMode: 0440
{{- end }}
{{- if .Values.storage.s3.caSecretName }}
- name: s3-ca
  secret:
    secretName: {{ .Values.storage.s3.caSecretName }}
    items:
      - key: {{ .Values.storage.s3.caSecretKey }}
        path: {{ .Values.storage.s3.caSecretKey }}
    defaultMode: 0440
{{- end }}
{{- end }}

{{/* Dependency egress for workloads using Django/Celery clients. */}}
{{- define "ocr-local.applicationDependencyEgress" -}}
# PostgreSQL
- to:
    {{- if .Values.networkPolicy.externalDependencies.postgresql.enabled }}
    - namespaceSelector:
        {{- toYaml .Values.networkPolicy.externalDependencies.postgresql.namespaceSelector | nindent 8 }}
      podSelector:
        {{- toYaml .Values.networkPolicy.externalDependencies.postgresql.podSelector | nindent 8 }}
    {{- else }}
    - podSelector:
        matchLabels:
          app.kubernetes.io/component: postgres
    {{- end }}
  ports:
    - protocol: TCP
      port: {{ .Values.networkPolicy.externalDependencies.postgresql.port }}
# RabbitMQ
- to:
    {{- if .Values.networkPolicy.externalDependencies.rabbitmq.enabled }}
    - namespaceSelector:
        {{- toYaml .Values.networkPolicy.externalDependencies.rabbitmq.namespaceSelector | nindent 8 }}
      podSelector:
        {{- toYaml .Values.networkPolicy.externalDependencies.rabbitmq.podSelector | nindent 8 }}
    {{- else }}
    - podSelector:
        matchLabels:
          app.kubernetes.io/component: rabbitmq
    {{- end }}
  ports:
    {{- if .Values.networkPolicy.externalDependencies.rabbitmq.enabled }}
    - protocol: TCP
      port: {{ .Values.networkPolicy.externalDependencies.rabbitmq.port }}
    {{- else }}
    {{- if not .Values.tls.enabled }}
    - protocol: TCP
      port: 5672
    {{- end }}
    {{- if .Values.tls.enabled }}
    - protocol: TCP
      port: 5671
    {{- end }}
    {{- end }}
# Redis
- to:
    {{- if .Values.networkPolicy.externalDependencies.redis.enabled }}
    - namespaceSelector:
        {{- toYaml .Values.networkPolicy.externalDependencies.redis.namespaceSelector | nindent 8 }}
      podSelector:
        {{- toYaml .Values.networkPolicy.externalDependencies.redis.podSelector | nindent 8 }}
    {{- else }}
    - podSelector:
        matchLabels:
          app.kubernetes.io/component: redis
    {{- end }}
  ports:
    {{- if .Values.networkPolicy.externalDependencies.redis.enabled }}
    - protocol: TCP
      port: {{ .Values.networkPolicy.externalDependencies.redis.port }}
    {{- else }}
    {{- if not .Values.tls.enabled }}
    - protocol: TCP
      port: 6379
    {{- end }}
    {{- if .Values.tls.enabled }}
    - protocol: TCP
      port: 6380
    {{- end }}
    {{- end }}
{{- if .Values.redis.sentinel.enabled }}
# Redis Sentinel
- to:
    - podSelector:
        matchLabels:
          {{- include "ocr-local.selectorLabels" . | nindent 10 }}
          app.kubernetes.io/component: redis-sentinel
  ports:
    - protocol: TCP
      port: 26379
{{- end }}
# DNS (kube-dns / CoreDNS)
- ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
{{- if eq .Values.storage.backend "s3" }}
{{- if and (not .Values.networkPolicy.externalDependencies.s3.enabled) (not .Values.storage.s3.networkPolicy.allowedCidrs) }}
{{- fail "storage.s3.networkPolicy.allowedCidrs must be explicitly set when storage.backend=s3" }}
{{- end }}
- to:
    {{- if .Values.networkPolicy.externalDependencies.s3.enabled }}
    - namespaceSelector:
        {{- toYaml .Values.networkPolicy.externalDependencies.s3.namespaceSelector | nindent 8 }}
      podSelector:
        {{- toYaml .Values.networkPolicy.externalDependencies.s3.podSelector | nindent 8 }}
    {{- else }}
    {{- range .Values.storage.s3.networkPolicy.allowedCidrs }}
    - ipBlock:
        cidr: {{ . | quote }}
    {{- end }}
    {{- end }}
  ports:
    - protocol: TCP
      port: {{ .Values.networkPolicy.externalDependencies.s3.port }}
{{- end }}
{{- end }}
{{/* The global S3 cache path must resolve inside protected staging on every pod that consumes the ConfigMap. */}}
{{- define "ocr-local.s3CacheStagingVolume" -}}
{{- if eq .Values.storage.backend "s3" }}
- name: s3-cache-staging
  {{- if eq .Values.workerSecurity.stagingVolume.type "emptyDir" }}
  emptyDir:
    medium: {{ .Values.workerSecurity.stagingVolume.emptyDir.medium }}
    sizeLimit: {{ .Values.workerSecurity.stagingVolume.emptyDir.sizeLimit }}
  {{- else }}
  ephemeral:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: {{ .Values.workerSecurity.stagingVolume.pvc.storageClass | quote }}
        resources:
          requests:
            storage: {{ .Values.workerSecurity.stagingVolume.pvc.size }}
  {{- end }}
{{- end }}
{{- end }}

{{- define "ocr-local.s3CacheStagingMount" -}}
{{- if eq .Values.storage.backend "s3" }}
- name: s3-cache-staging
  mountPath: {{ .Values.workerSecurity.stagingVolume.mountPath | quote }}
{{- end }}
{{- end }}

{{/*
True when host is a chart Service DNS alias: short name, namespace, cluster
FQDN, or StatefulSet pod DNS. Release-qualified names only; unrelated hosts
that share a prefix are not aliases.
*/}}
{{- define "ocr-local.hostnameTargetsBundledService" -}}
{{- $root := .root -}}
{{- $host := lower (trimSuffix "." (trim (default "" .host))) -}}
{{- $svc := lower (printf "%s-%s" $root.Release.Name .service) -}}
{{- $ns := lower $root.Release.Namespace -}}
{{- $domain := lower (trimSuffix "." (default "cluster.local" $root.Values.clusterDomain)) -}}
{{- $suffixes := list "" (printf ".%s" $ns) (printf ".%s.svc" $ns) (printf ".%s.svc.%s" $ns $domain) -}}
{{- $matched := false -}}
{{- range $suffix := $suffixes -}}
{{- if eq $host (printf "%s%s" $svc $suffix) -}}
{{- $matched = true -}}
{{- end -}}
{{- end -}}
{{- $parts := splitn "." 2 $host -}}
{{- $left := $parts._0 -}}
{{- $right := default "" $parts._1 -}}
{{- if and (not $matched) $right (hasPrefix (printf "%s-" $svc) $left) -}}
{{- $ordinal := trimPrefix (printf "%s-" $svc) $left -}}
{{- if regexMatch "^[0-9]+$" $ordinal -}}
{{- range $suffix := $suffixes -}}
{{- if eq $right (printf "%s%s" $svc $suffix) -}}
{{- $matched = true -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if $matched }}true{{- else }}false{{- end -}}
{{- end -}}

{{/* True when any semicolon-delimited URL targets bundled Redis, replica, or Sentinel. */}}
{{- define "ocr-local.redisUrlTargetsBundled" -}}
{{- $root := .root -}}
{{- $matched := false -}}
{{- range splitList ";" (default "" .url) -}}
{{- $part := trim . -}}
{{- if $part -}}
{{- $host := (urlParse $part).hostname -}}
{{- range list "redis" "redis-sentinel" "redis-replica" -}}
{{- if eq (include "ocr-local.hostnameTargetsBundledService" (dict "root" $root "host" $host "service" .)) "true" -}}
{{- $matched = true -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if $matched }}true{{- else }}false{{- end -}}
{{- end -}}

{{/* True when every URL segment has a host and an allowed TLS scheme. */}}
{{- define "ocr-local.redisEndpointUsable" -}}
{{- $ok := true -}}
{{- $saw := false -}}
{{- $allowSentinel := .allowSentinel -}}
{{- range splitList ";" (default "" .url) -}}
{{- $part := trim . -}}
{{- if $part -}}
{{- $saw = true -}}
{{- $parsed := urlParse $part -}}
{{- $scheme := lower $parsed.scheme -}}
{{- $allowed := eq $scheme "rediss" -}}
{{- if and $allowSentinel (eq $scheme "sentinel") -}}
{{- $allowed = true -}}
{{- end -}}
{{- if or (not $allowed) (eq (trim $parsed.hostname) "") -}}
{{- $ok = false -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if and $ok $saw }}true{{- else }}false{{- end -}}
{{- end -}}
