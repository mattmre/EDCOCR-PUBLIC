# OpenTelemetry Production Configuration

Production deployment guide for distributed tracing in the EDCOCR pipeline.

## Architecture Overview

```
  OCR API / Pipeline / Workers
          |
          | OTel SDK (opentelemetry-api + opentelemetry-sdk)
          | Traces exported via OTLP gRPC
          v
  +-------------------------+
  | OpenTelemetry Collector |
  | (otel-collector-contrib)|
  | - batch processor       |
  | - memory_limiter        |
  | - attributes processor  |
  +-------------------------+
     |          |          |
     v          v          v
  Jaeger    Prometheus   Grafana
  / Tempo   (metrics     Cloud
  (traces)   bridge)     (managed)
```

The tracing subsystem has three layers:

1. **Application SDK** (`api/tracing.py`): Instruments FastAPI requests and
   pipeline operations. When the OTel SDK packages are not installed, a
   zero-dependency `LocalTracer` fallback records spans in memory for
   debugging.

2. **OTel Collector** (`otel/otel-collector-config.yaml`): Receives OTLP
   telemetry, applies batching and memory limits, then exports to one or
   more backends.

3. **Trace Backend**: Jaeger (bundled for development), Grafana Tempo
   (recommended for production), or any OTLP-compatible backend.

## Prerequisites

Install the optional OTel packages (commented out in `requirements.txt`):

```bash
pip install \
  opentelemetry-api>=1.20.0 \
  opentelemetry-sdk>=1.20.0 \
  opentelemetry-exporter-otlp-proto-grpc>=1.20.0 \
  opentelemetry-instrumentation-fastapi>=0.41b0
```

Without these packages the tracing module falls back to `LocalTracer`
transparently. No code changes are needed.

## Quick Start (Docker Compose)

### Standalone (local development)

```bash
docker compose -f otel/docker-compose.otel.yml up -d
```

This starts the OTel Collector and Jaeger. View traces at
`http://localhost:16686`.

### With the Coordinator Stack

```bash
docker compose \
  -f coordinator/docker-compose.coordinator.yml \
  -f otel/docker-compose.otel.yml \
  up -d
```

Both stacks share the `coordinator_default` network, so the OCR API and
workers can reach the collector at `otel-collector:4317`.

### Configure the application

Set these environment variables on the API or worker containers:

```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_SERVICE_NAME=ocr-local-api
export OTEL_SAMPLE_RATE=0.1
export OTEL_SAMPLING_STRATEGY=parentbased
```

Then start the API server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables Reference

### Application-level (api/tracing.py)

| Variable                        | Default              | Description                                        |
|---------------------------------|----------------------|----------------------------------------------------|
| `OTEL_ENABLED`                  | `true`               | Master toggle for tracing                          |
| `OTEL_SERVICE_NAME`             | `ocr-pipeline`       | Service identity reported in trace spans           |
| `OTEL_EXPORTER`                 | `console`            | Exporter type: `console`, `otlp`, or `jaeger`      |
| `OTEL_EXPORTER_OTLP_ENDPOINT`  | `http://localhost:4317` | OTel Collector gRPC endpoint                    |
| `OTEL_SAMPLE_RATE`             | `1.0`                | Float 0.0-1.0 controlling trace sampling ratio     |
| `OTEL_SAMPLING_STRATEGY`       | `ratio`              | `ratio` (simple) or `parentbased` (distributed)    |

Setting `OTEL_EXPORTER=jaeger` emits a deprecation warning and falls back
to the OTLP exporter path. Jaeger 1.35+ accepts OTLP natively on port 4317.

### Collector-level (otel/otel-collector-config.yaml)

| Variable                   | Default        | Description                              |
|----------------------------|----------------|------------------------------------------|
| `OTEL_ENVIRONMENT`         | `development`  | Deployment environment label             |
| `OTEL_MEMORY_LIMIT_MIB`   | `512`          | Memory limiter hard cap (MiB)            |
| `OTEL_MEMORY_SPIKE_MIB`   | `128`          | Memory limiter spike allowance (MiB)     |
| `OTEL_BATCH_TIMEOUT`       | `5s`           | Batch processor flush interval           |
| `OTEL_BATCH_SIZE`          | `512`          | Max spans per batch export               |
| `OTEL_JAEGER_ENDPOINT`     | `jaeger:4317`  | Trace backend OTLP gRPC endpoint         |

## Production Sampling Strategy

Sampling controls the fraction of traces recorded. Use `parentbased`
sampling in distributed deployments so that all services in a request chain
agree on whether to sample a given trace.

| Environment   | Recommended Rate | Strategy       | Rationale                             |
|---------------|------------------|----------------|---------------------------------------|
| Development   | 1.0 (100%)       | `ratio`        | Full visibility for debugging         |
| Staging       | 0.5 (50%)        | `parentbased`  | Balanced cost and visibility          |
| Production    | 0.1 (10%)        | `parentbased`  | Cost-effective for steady traffic     |
| High-volume   | 0.01 (1%)        | `parentbased`  | Forensic pipeline at scale            |

Set via environment:

```bash
export OTEL_SAMPLE_RATE=0.1
export OTEL_SAMPLING_STRATEGY=parentbased
```

### How parentbased sampling works

When a request arrives with an upstream trace context (W3C Trace Context
header), the `parentbased` sampler inherits the parent's sampling decision.
If the parent was sampled, all downstream services also sample that trace,
giving complete end-to-end visibility. New root spans use the configured
`OTEL_SAMPLE_RATE` ratio.

This is critical for the OCR pipeline's distributed architecture where the
coordinator dispatches work to GPU/CPU workers via Celery. Without
parentbased sampling, individual workers might drop spans from an otherwise
sampled trace.

## Grafana Tempo as Alternative Backend

Grafana Tempo is the recommended production alternative to Jaeger for
large-scale trace storage.

### Swap Jaeger for Tempo

1. Point the collector at your Tempo instance:

```bash
export OTEL_JAEGER_ENDPOINT=tempo.internal:4317
```

2. Start only the collector (skip the Jaeger container):

```bash
docker compose -f otel/docker-compose.otel.yml up -d otel-collector
```

3. Add Tempo as a data source in Grafana and query traces via TraceQL.

### Grafana Cloud (managed)

Update `otel-collector-config.yaml` to export directly to Grafana Cloud:

```yaml
exporters:
  otlp/traces:
    endpoint: tempo-us-central1.grafana.net:443
    headers:
      Authorization: "Basic <base64-encoded-user:api-key>"
    tls:
      insecure: false
```

Restart the collector after modifying the config.

## Kubernetes Deployment

Use the Helm values overlay at `helm/ocr-local/values-otel.yaml`:

```bash
helm upgrade ocr-local ./helm/ocr-local \
  -f helm/ocr-local/values.yaml \
  -f helm/ocr-local/values-otel.yaml
```

This overlay:

- Enables the OTel Collector sidecar/deployment
- Configures Jaeger as the default trace backend
- Injects `OTEL_*` environment variables into coordinator and GPU worker pods
- Sets a 10% production sampling rate

For production Kubernetes clusters, consider the
[OpenTelemetry Operator](https://opentelemetry.io/docs/kubernetes/operator/)
for automated collector lifecycle management, auto-instrumentation injection,
and target allocator support.

### Custom collector image

Override the collector image in the values overlay:

```yaml
otel:
  collector:
    image: otel/opentelemetry-collector-contrib:0.100.0
```

### Disable Jaeger in production

When using Tempo or Grafana Cloud, disable the bundled Jaeger:

```yaml
otel:
  jaeger:
    enabled: false
  collector:
    endpoint: "http://tempo:4317"
```

## Troubleshooting

### Verify the collector is running

```bash
# Health check endpoint (returns 200 when healthy)
curl -s http://localhost:13133

# zpages — active span diagnostics
open http://localhost:55679/debug/tracez

# Internal metrics (receiver/exporter counts)
curl -s http://localhost:8888/metrics | grep otelcol_receiver
```

### Debug exporter

Temporarily enable the `debug` exporter in `otel-collector-config.yaml`
to print all received telemetry to stdout:

```yaml
exporters:
  debug:
    verbosity: detailed

service:
  pipelines:
    traces:
      exporters: [otlp/traces, debug]
```

### No traces appearing in Jaeger

1. Confirm `OTEL_ENABLED=true` and `OTEL_EXPORTER=otlp` are set on the
   application.
2. Verify the application can reach `OTEL_EXPORTER_OTLP_ENDPOINT` (default
   `http://localhost:4317` or `http://otel-collector:4317` inside Docker).
3. Check collector logs: `docker logs otel-collector 2>&1 | tail -20`.
4. Ensure `OTEL_SAMPLE_RATE` is not `0.0`.

### Collector out of memory

Increase the memory limiter:

```bash
export OTEL_MEMORY_LIMIT_MIB=1024
export OTEL_MEMORY_SPIKE_MIB=256
```

Or reduce batch size to lower peak memory:

```bash
export OTEL_BATCH_SIZE=256
```

### Application falls back to LocalTracer

This means the OTel SDK packages are not installed. Install them:

```bash
pip install opentelemetry-api opentelemetry-sdk \
  opentelemetry-exporter-otlp-proto-grpc
```

The application logs `OpenTelemetry SDK not available -- using LocalTracer
fallback` when this happens.

## Security Considerations

### TLS for OTLP endpoints

In production, the OTLP gRPC connection between the application and
collector (and between the collector and backend) should use TLS.

For the application SDK, the `OTLPSpanExporter` accepts TLS configuration:

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="https://otel-collector:4317",
    insecure=False,
    # credentials=... (for mTLS)
)
```

For the collector, configure TLS in the receiver and exporter sections of
`otel-collector-config.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        tls:
          cert_file: /certs/server.crt
          key_file: /certs/server.key
```

### Network isolation

The Docker Compose overlay connects services via the `coordinator_default`
network. In Kubernetes, use `NetworkPolicy` resources (already templated in
the Helm chart) to restrict which pods can reach the collector.

### Sensitive data in spans

Avoid recording PII or PHI in span attributes. The OCR pipeline traces
include operation names, page counts, and timing data, but never document
content. If custom instrumentation is added, review span attributes for
sensitive data before enabling production export.

## Migration from Deprecated Jaeger Exporter

The `opentelemetry-exporter-jaeger-thrift` package is deprecated. If your
deployment previously used `OTEL_EXPORTER=jaeger`, the application now
automatically falls back to the OTLP exporter path with a deprecation
warning.

To migrate explicitly:

1. Change `OTEL_EXPORTER=jaeger` to `OTEL_EXPORTER=otlp`.
2. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to your Jaeger OTLP port (4317).
3. Remove `opentelemetry-exporter-jaeger-thrift` from your environment.

No collector configuration changes are needed -- Jaeger 1.35+ accepts OTLP
on port 4317 by default.

## Service Endpoints Reference

| Service              | Port  | Protocol | Description                       |
|----------------------|-------|----------|-----------------------------------|
| OTel Collector gRPC  | 4317  | gRPC     | Primary OTLP telemetry receiver   |
| OTel Collector HTTP  | 4318  | HTTP     | Secondary OTLP receiver           |
| Collector health     | 13133 | HTTP     | Liveness probe endpoint           |
| Collector zpages     | 55679 | HTTP     | Internal span diagnostics         |
| Collector metrics    | 8888  | HTTP     | Internal Prometheus metrics       |
| Prometheus bridge    | 8889  | HTTP     | Exported Prometheus metrics       |
| Jaeger UI            | 16686 | HTTP     | Trace visualization dashboard     |
| Jaeger collector     | 14268 | HTTP     | Legacy Jaeger HTTP collector      |

## Further Reading

- [OpenTelemetry Collector documentation](https://opentelemetry.io/docs/collector/)
- [Grafana Tempo OTLP configuration](https://grafana.com/docs/tempo/latest/configuration/)
- [Jaeger OTLP receiver](https://www.jaegertracing.io/docs/latest/apis/#opentelemetry-protocol-stable)
- [OTel Python SDK](https://opentelemetry.io/docs/languages/python/)
- [W3C Trace Context specification](https://www.w3.org/TR/trace-context/)
