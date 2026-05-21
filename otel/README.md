# OpenTelemetry Collector for EDCOCR

Production-ready OpenTelemetry Collector configuration for distributed tracing,
metrics bridging, and log collection in the EDCOCR pipeline.

## Architecture

```
  OCR Pipeline / API / Workers
          |
          | OTLP (gRPC :4317 / HTTP :4318)
          v
  +-------------------+
  | OTel Collector    |
  | - batch           |
  | - memory_limiter  |
  | - attributes      |
  +-------------------+
     |        |       |
     v        v       v
  Jaeger  Prometheus  Debug
  (traces) (metrics)  (stdout)
```

## Quick Start (Local Development)

```bash
# Start the collector + Jaeger
docker compose -f otel/docker-compose.otel.yml up -d

# Configure the OCR API to send traces
export OTEL_ENABLED=true
export OTEL_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=ocr-local-api
export OTEL_SAMPLE_RATE=1.0

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# View traces in Jaeger UI
open http://localhost:16686
```

## Quick Start (With Coordinator Stack)

```bash
# Start everything together
docker compose \
  -f coordinator/docker-compose.coordinator.yml \
  -f otel/docker-compose.otel.yml \
  up -d

# The coordinator services automatically connect to the collector
# via the shared Docker network
```

## Service Endpoints

| Service            | Port  | Description                      |
|--------------------|-------|----------------------------------|
| OTLP gRPC         | 4317  | Primary telemetry receiver       |
| OTLP HTTP         | 4318  | HTTP telemetry receiver          |
| Jaeger UI          | 16686 | Trace visualization              |
| Prometheus metrics | 8889  | Scraped by Prometheus            |
| Health check       | 13133 | Collector liveness probe         |
| zpages             | 55679 | Collector internal diagnostics   |

## Environment Variables

### Collector Configuration

| Variable                  | Default       | Description                          |
|---------------------------|---------------|--------------------------------------|
| `OTEL_ENVIRONMENT`        | `development` | Deployment environment label         |
| `OTEL_MEMORY_LIMIT_MIB`  | `512`         | Memory limiter hard cap (MiB)        |
| `OTEL_MEMORY_SPIKE_MIB`  | `128`         | Memory limiter spike allowance (MiB) |
| `OTEL_BATCH_TIMEOUT`     | `5s`          | Batch processor flush interval       |
| `OTEL_BATCH_SIZE`        | `512`         | Max spans per batch export           |
| `OTEL_JAEGER_ENDPOINT`   | `jaeger:4317` | Trace backend OTLP gRPC endpoint     |

### Application Configuration (api/tracing.py)

| Variable                      | Default           | Description                       |
|-------------------------------|-------------------|-----------------------------------|
| `OTEL_ENABLED`                | `true`            | Enable/disable tracing            |
| `OTEL_SERVICE_NAME`           | `ocr-pipeline`    | Service name in traces            |
| `OTEL_EXPORTER`               | `console`         | Exporter type: console/otlp       |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `localhost:4317`  | Collector gRPC endpoint           |
| `OTEL_SAMPLE_RATE`          | `1.0`             | Trace sampling ratio (0.0 - 1.0)  |
| `OTEL_SAMPLING_STRATEGY`      | `ratio`           | Sampling strategy: ratio/parentbased |

## Production Deployment

### With Grafana Tempo

Replace the Jaeger backend with Grafana Tempo for production-scale trace storage:

1. Set `OTEL_JAEGER_ENDPOINT` to your Tempo OTLP gRPC endpoint
2. Remove the `jaeger` service from the compose file
3. Configure Tempo datasource in your Grafana instance

```bash
# Example: Tempo running on a separate host
export OTEL_JAEGER_ENDPOINT=tempo.internal:4317
docker compose -f otel/docker-compose.otel.yml up -d otel-collector
```

### With Grafana Cloud

For managed trace storage:

1. Update the `otlp/traces` exporter in `otel-collector-config.yaml`:

```yaml
exporters:
  otlp/traces:
    endpoint: tempo-us-central1.grafana.net:443
    headers:
      Authorization: "Basic <base64-encoded-user:api-key>"
    tls:
      insecure: false
```

2. Restart the collector.

### Sampling Strategy

For production workloads, reduce sampling to control costs and storage:

| Environment  | Recommended Rate | Strategy       |
|--------------|------------------|----------------|
| Development  | 1.0 (100%)       | ratio          |
| Staging      | 0.5 (50%)        | parentbased    |
| Production   | 0.1 (10%)        | parentbased    |
| High-volume  | 0.01 (1%)        | parentbased    |

Set via environment variables:

```bash
export OTEL_SAMPLE_RATE=0.1
export OTEL_SAMPLING_STRATEGY=parentbased
```

The `parentbased` strategy inherits the sampling decision from upstream services,
ensuring complete trace visibility when a trace is sampled.

### Memory Tuning

For high-throughput deployments, increase collector memory limits:

```bash
export OTEL_MEMORY_LIMIT_MIB=1024
export OTEL_MEMORY_SPIKE_MIB=256
export OTEL_BATCH_SIZE=1024
```

### Kubernetes Deployment

The Helm chart at `helm/ocr-local/` can be extended to include the collector.
A minimal sidecar configuration:

```yaml
# values-otel.yaml
otel:
  enabled: true
  image: otel/opentelemetry-collector-contrib:0.96.0
  resources:
    limits:
      memory: 512Mi
      cpu: 500m
```

For production Kubernetes deployments, consider using the
[OpenTelemetry Operator](https://opentelemetry.io/docs/kubernetes/operator/)
for automated collector lifecycle management.

## Troubleshooting

### Verify collector is receiving data

```bash
# Check collector health
curl http://localhost:13133

# Check zpages for active spans
open http://localhost:55679/debug/tracez

# Check collector internal metrics
curl http://localhost:8888/metrics | grep otelcol_receiver
```

### Common issues

1. **No traces in Jaeger**: Verify `OTEL_EXPORTER=otlp` and
   `OTEL_EXPORTER_OTLP_ENDPOINT` points to the collector.

2. **Collector OOM**: Increase `OTEL_MEMORY_LIMIT_MIB` or reduce
   `OTEL_BATCH_SIZE`.

3. **High latency**: Reduce `OTEL_BATCH_TIMEOUT` for faster exports
   at the cost of more network calls.

4. **Network errors**: When using with the coordinator stack, ensure
   services share the same Docker network.
