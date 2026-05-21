# OpenTelemetry Production Configuration Guide

Operator guide for configuring distributed tracing in the EDCOCR pipeline
across all supported backends (Grafana Tempo, Jaeger, AWS X-Ray, Datadog).

## Environment Variables Reference

All configuration is driven by environment variables. The tracing module
(`api/tracing.py`) reads standard OTel env vars with legacy fallbacks.

### Standard OTel Variables (Preferred)

| Variable | Default | Description |
|---|---|---|
| `OTEL_ENABLED` | `true` | Master toggle for tracing (`true`/`false`) |
| `OTEL_SERVICE_NAME` | `ocr-local-api` | Service identity in trace spans |
| `OTEL_SERVICE_VERSION` | `version.__version__` | Service version in resource attributes |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP gRPC collector endpoint |
| `OTEL_EXPORTER_OTLP_HEADERS` | (empty) | Comma-separated `key=value` auth headers |
| `OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` | Sampler: `always_on`, `always_off`, `parentbased_traceidratio` |
| `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Sampling ratio (float 0.0--1.0) |
| `OTEL_ENVIRONMENT` | `development` | Deployment env label in resource attributes |
| `OTEL_EXPORTER` | `console` | Exporter backend: `console`, `otlp`, `jaeger` |

### Legacy Variables (Backward Compatible)

These are still honoured but standard names take precedence when both are set.

| Legacy Variable | Standard Replacement |
|---|---|
| `OTEL_SAMPLE_RATE` | `OTEL_TRACES_SAMPLER_ARG` |
| `OTEL_SAMPLING_STRATEGY` | `OTEL_TRACES_SAMPLER` |

### Resource Attributes

The tracing module sets these OTel resource attributes automatically:

| Attribute | Source |
|---|---|
| `service.name` | `OTEL_SERVICE_NAME` |
| `service.version` | `OTEL_SERVICE_VERSION` or `version.__version__` |
| `service.namespace` | `ocr-local` (hardcoded) |
| `deployment.environment` | `OTEL_ENVIRONMENT` or `DEPLOYMENT_ENV` |

---

## Grafana Cloud Tempo Integration

Grafana Cloud Tempo accepts traces via OTLP/gRPC with basic auth.

### Configuration

```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=https://tempo-us-central1.grafana.net:443
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64-encoded-instanceID:api-key>"
export OTEL_SERVICE_NAME=ocr-local-api
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1
export OTEL_ENVIRONMENT=production
```

### Getting the Credentials

1. Log in to Grafana Cloud and navigate to **Connections > Tempo**.
2. Copy the **Instance ID** and **API key** (generate one under API keys).
3. Base64-encode `<instanceID>:<api-key>`:

```bash
echo -n "123456:glc_abc123..." | base64
```

4. Set the result as the `Authorization=Basic <value>` header.

### Via OTel Collector (Recommended)

For production, route through an OTel Collector sidecar rather than
exporting directly from the application:

```bash
# Application points at the local collector
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317

# Collector config exports to Grafana Cloud
# (see otel/otel-collector-config.yaml)
```

Collector config snippet:

```yaml
exporters:
  otlp/grafana:
    endpoint: tempo-us-central1.grafana.net:443
    headers:
      Authorization: "Basic <base64-encoded-instanceID:api-key>"
    tls:
      insecure: false

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, memory_limiter]
      exporters: [otlp/grafana]
```

---

## Jaeger Integration

Jaeger 1.35+ supports OTLP natively on port 4317. No Thrift exporter needed.

### Direct OTLP/gRPC

```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
export OTEL_SERVICE_NAME=ocr-local-api
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1
```

### Docker Compose Development Stack

Use the bundled OTel overlay:

```bash
docker compose -f otel/docker-compose.otel.yml up -d
```

This starts Jaeger with OTLP receiver and the OTel Collector. View traces
at `http://localhost:16686`.

### Migration from Deprecated Jaeger Exporter

If you previously used `OTEL_EXPORTER=jaeger`, the application now
automatically redirects to the OTLP path with a deprecation warning.
To migrate explicitly:

1. Change `OTEL_EXPORTER=jaeger` to `OTEL_EXPORTER=otlp`.
2. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to your Jaeger OTLP port (4317).
3. Remove `opentelemetry-exporter-jaeger-thrift` from your environment.

---

## AWS X-Ray via ADOT Collector

AWS Distro for OpenTelemetry (ADOT) Collector converts OTLP traces
to X-Ray format.

### Configuration

```bash
# Application sends OTLP to the ADOT Collector sidecar
export OTEL_ENABLED=true
export OTEL_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://adot-collector:4317
export OTEL_SERVICE_NAME=ocr-local-api
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.05
export OTEL_ENVIRONMENT=production
```

### ADOT Collector Config

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  awsxray:
    region: us-east-1
    # IAM role-based auth via EC2 instance profile or EKS IRSA

processors:
  batch:
    timeout: 5s
    send_batch_size: 256

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [awsxray]
```

### EKS Deployment

Deploy the ADOT Collector as a DaemonSet or sidecar using the
[AWS ADOT EKS add-on](https://docs.aws.amazon.com/eks/latest/userguide/opentelemetry.html).
Ensure the pod service account has the `AWSXRayDaemonWriteAccess` IAM policy.

---

## Datadog Agent Integration

The Datadog Agent (v7.32+) includes an OTLP receiver that accepts
gRPC traces on port 4317.

### Configuration

```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://datadog-agent:4317
export OTEL_SERVICE_NAME=ocr-local-api
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1
export OTEL_ENVIRONMENT=production
```

### Datadog Agent Config (`datadog.yaml`)

```yaml
otlp_config:
  receiver:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

apm_config:
  enabled: true
```

### Kubernetes

Set `DD_OTLP_CONFIG_RECEIVER_PROTOCOLS_GRPC_ENDPOINT=0.0.0.0:4317` on
the Datadog Agent DaemonSet. The OCR worker pods reference the agent via
the node-local service (`datadog-agent.datadog.svc.cluster.local:4317`).

---

## Sampling Strategy Guide

Sampling controls what fraction of traces are recorded. Choose based on
traffic volume and budget.

### Recommended Settings by Environment

| Environment | Sampler | Rate | Rationale |
|---|---|---|---|
| Development | `always_on` | N/A | Full visibility for debugging |
| Staging | `parentbased_traceidratio` | `0.5` | Balanced cost and coverage |
| Production (low traffic) | `parentbased_traceidratio` | `0.5` | Adequate volume for analysis |
| Production (standard) | `parentbased_traceidratio` | `0.1` | Cost-effective, ~10% coverage |
| Production (high traffic) | `parentbased_traceidratio` | `0.01` | 1% sampling at scale |

### Why `parentbased_traceidratio`

In the OCR distributed pipeline, the coordinator dispatches work to
GPU/CPU workers via Celery. Without parent-based sampling, a sampled
coordinator trace might have missing worker spans (and vice versa).

`parentbased_traceidratio` inherits the parent's sampling decision via
W3C Trace Context propagation:

- If the parent span was sampled, all downstream services also sample.
- New root spans use the configured ratio.
- This guarantees complete end-to-end traces for sampled requests.

### Applying Configuration

```bash
# Production: 10% sampling with parent-based propagation
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1

# Development: sample everything
export OTEL_TRACES_SAMPLER=always_on

# Temporarily disable tracing (e.g., during maintenance)
export OTEL_TRACES_SAMPLER=always_off
```

---

## Verifying Traces Are Flowing

### Step 1: Check the Health Endpoint

```bash
# Submit a request that generates a trace
curl -s -H "X-API-Key: $OCR_API_KEY" http://localhost:8000/api/v1/health
```

### Step 2: Verify Application Logs

Look for the initialization message:

```
OpenTelemetry tracing initialised (exporter=otlp, sampler=parentbased_traceidratio, rate=0.10, endpoint=http://otel-collector:4317, env=production)
```

If you see this instead, the OTel packages are not installed:

```
OpenTelemetry SDK not available -- using LocalTracer fallback.
```

### Step 3: Check Collector Health

```bash
# OTel Collector health check
curl -s http://otel-collector:13133

# Collector internal metrics (look for received/exported span counts)
curl -s http://otel-collector:8888/metrics | grep otelcol_receiver_accepted_spans
curl -s http://otel-collector:8888/metrics | grep otelcol_exporter_sent_spans
```

### Step 4: Query the Backend

- **Jaeger**: Visit `http://localhost:16686`, select service `ocr-local-api`
- **Grafana Tempo**: Use Explore > Tempo data source, search by
  `service.name = "ocr-local-api"`
- **AWS X-Ray**: Open the X-Ray console, filter by service name
- **Datadog**: Navigate to APM > Traces, filter by `service:ocr-local-api`

---

## Kubernetes Deployment (Helm Values)

Add OTel environment variables to the coordinator and worker deployments
via Helm values.

### values-otel.yaml

```yaml
# Coordinator deployment
coordinator:
  env:
    - name: OTEL_ENABLED
      value: "true"
    - name: OTEL_EXPORTER
      value: "otlp"
    - name: OTEL_EXPORTER_OTLP_ENDPOINT
      value: "http://otel-collector:4317"
    - name: OTEL_SERVICE_NAME
      value: "ocr-local-coordinator"
    - name: OTEL_TRACES_SAMPLER
      value: "parentbased_traceidratio"
    - name: OTEL_TRACES_SAMPLER_ARG
      value: "0.1"
    - name: OTEL_ENVIRONMENT
      value: "production"
    # For Grafana Cloud (use a Secret reference instead of plain value):
    # - name: OTEL_EXPORTER_OTLP_HEADERS
    #   valueFrom:
    #     secretKeyRef:
    #       name: otel-secrets
    #       key: otlp-headers

# GPU worker deployment
gpuWorker:
  env:
    - name: OTEL_ENABLED
      value: "true"
    - name: OTEL_EXPORTER
      value: "otlp"
    - name: OTEL_EXPORTER_OTLP_ENDPOINT
      value: "http://otel-collector:4317"
    - name: OTEL_SERVICE_NAME
      value: "ocr-local-gpu-worker"
    - name: OTEL_TRACES_SAMPLER
      value: "parentbased_traceidratio"
    - name: OTEL_TRACES_SAMPLER_ARG
      value: "0.1"
    - name: OTEL_ENVIRONMENT
      value: "production"

# CPU worker deployment
cpuWorker:
  env:
    - name: OTEL_ENABLED
      value: "true"
    - name: OTEL_EXPORTER
      value: "otlp"
    - name: OTEL_EXPORTER_OTLP_ENDPOINT
      value: "http://otel-collector:4317"
    - name: OTEL_SERVICE_NAME
      value: "ocr-local-cpu-worker"
    - name: OTEL_TRACES_SAMPLER
      value: "parentbased_traceidratio"
    - name: OTEL_TRACES_SAMPLER_ARG
      value: "0.1"
    - name: OTEL_ENVIRONMENT
      value: "production"
```

### Apply the Overlay

```bash
helm upgrade ocr-local ./helm/ocr-local \
  -f helm/ocr-local/values.yaml \
  -f helm/ocr-local/values-otel.yaml
```

### Storing Auth Headers in Kubernetes Secrets

Never put auth tokens in plain Helm values. Create a secret:

```bash
kubectl create secret generic otel-secrets \
  --from-literal=otlp-headers="Authorization=Basic $(echo -n 'ID:KEY' | base64)"
```

Reference it in the deployment env via `valueFrom.secretKeyRef`.

---

## Docker Compose Configuration

Add the following environment block to your `docker-compose.yml` service
definitions.

### API Service

```yaml
services:
  ocr-api:
    # ...existing config...
    environment:
      # --- OpenTelemetry ---
      OTEL_ENABLED: "true"
      OTEL_EXPORTER: "otlp"
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector:4317"
      OTEL_SERVICE_NAME: "ocr-local-api"
      OTEL_TRACES_SAMPLER: "parentbased_traceidratio"
      OTEL_TRACES_SAMPLER_ARG: "0.1"
      OTEL_ENVIRONMENT: "production"
      # For Grafana Cloud direct export (no collector):
      # OTEL_EXPORTER_OTLP_ENDPOINT: "https://tempo-us-central1.grafana.net:443"
      # OTEL_EXPORTER_OTLP_HEADERS: "Authorization=Basic <base64>"
```

### GPU Worker Service

```yaml
  ocr-gpu-worker:
    # ...existing config...
    environment:
      OTEL_ENABLED: "true"
      OTEL_EXPORTER: "otlp"
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector:4317"
      OTEL_SERVICE_NAME: "ocr-local-gpu-worker"
      OTEL_TRACES_SAMPLER: "parentbased_traceidratio"
      OTEL_TRACES_SAMPLER_ARG: "0.1"
      OTEL_ENVIRONMENT: "production"
```

### With the OTel Collector Overlay

```bash
docker compose \
  -f docker-compose.yml \
  -f otel/docker-compose.otel.yml \
  up -d
```

---

## Prerequisites

Install the optional OTel Python packages (commented out in `requirements.txt`):

```bash
pip install \
  opentelemetry-api>=1.20.0 \
  opentelemetry-sdk>=1.20.0 \
  opentelemetry-exporter-otlp-proto-grpc>=1.20.0 \
  opentelemetry-instrumentation-fastapi>=0.41b0
```

Without these packages, the tracing module falls back to `LocalTracer`
transparently. No code changes are needed -- the pipeline runs identically
with or without OTel installed.

---

## Graceful Degradation

The tracing module is designed to never crash the application:

1. **No OTel packages**: Falls back to `LocalTracer` (in-memory spans for
   debugging). Logged at INFO level.
2. **Invalid env vars**: Clamped to safe defaults (e.g., sample rate outside
   0.0--1.0 is clamped, unknown sampler names fall back to
   `parentbased_traceidratio`).
3. **Collector unreachable**: The OTLP exporter retries in the background.
   Spans may be dropped but the application continues processing.
4. **FastAPI instrumentation missing**: Skipped silently; manual
   `trace_operation` still works.

---

## Security Considerations

- **Never log OTLP headers**: The `OTEL_EXPORTER_OTLP_HEADERS` value
  may contain bearer tokens or API keys. The tracing module does not
  log header values.
- **Use TLS in production**: Set the endpoint to `https://` when
  exporting directly to a cloud backend. The OTel Collector should also
  use TLS for its receiver and exporter connections.
- **Avoid PII in spans**: The OCR pipeline traces include operation names,
  page counts, and timing data, but never document content. Review any
  custom span attributes before enabling production export.
- **Kubernetes Secrets**: Store `OTEL_EXPORTER_OTLP_HEADERS` in a
  Kubernetes Secret, not in plain ConfigMap or Helm values.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| "LocalTracer fallback" in logs | OTel SDK not installed | `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc` |
| No traces in backend | `OTEL_EXPORTER=console` (default) | Set `OTEL_EXPORTER=otlp` |
| Traces sent but not visible | Sampling rate too low | Set `OTEL_TRACES_SAMPLER=always_on` temporarily |
| Collector OOM | High trace volume | Increase `OTEL_MEMORY_LIMIT_MIB` or reduce `OTEL_TRACES_SAMPLER_ARG` |
| Auth errors to Grafana Cloud | Bad headers format | Verify `OTEL_EXPORTER_OTLP_HEADERS` uses `key=value` format with base64 credentials |
| gRPC deadline exceeded | Network/firewall issue | Verify connectivity to `OTEL_EXPORTER_OTLP_ENDPOINT` from the pod/container |

---

## Further Reading

- [OpenTelemetry Environment Variable Spec](https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/)
- [OTel Python SDK](https://opentelemetry.io/docs/languages/python/)
- [Grafana Tempo OTLP Configuration](https://grafana.com/docs/tempo/latest/configuration/)
- [AWS ADOT Collector](https://aws-otel.github.io/docs/getting-started/collector)
- [Datadog OTLP Ingestion](https://docs.datadoghq.com/tracing/trace_collection/open_standards/otlp_ingest_in_the_agent/)
- [Jaeger OTLP Receiver](https://www.jaegertracing.io/docs/latest/apis/#opentelemetry-protocol-stable)
- [W3C Trace Context Specification](https://www.w3.org/TR/trace-context/)
