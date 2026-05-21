# Billing and SLA Policy -- Locked Reference

**Version**: 1.0.0
**Effective Date**: 2026-03-26
**Status**: LOCKED -- changes require versioned amendment with architecture review

---

## 1. Billing Policy

### 1.1 Purpose

This document codifies the cost-estimation formulas and SLA targets used by the
EDCOCR pipeline.  The pipeline provides **cost estimation**, not invoicing.
Operators translate estimates into billing through their own systems.

### 1.2 Cost Computation Formula

Total estimated cost for a tenant in a billing period:

```
total_cost = (pages_processed * COST_PER_PAGE)
           + (gpu_seconds     * COST_PER_GPU_SECOND)
           + (storage_gb      * COST_PER_GB_STORED)
           + (api_calls       * COST_PER_API_CALL)
```

All rates are denominated in **USD**.  The formula is versioned as
`BILLING_FORMULA_VERSION` in `cost_tracking.py` (currently `1.0.0`).

### 1.3 Default Rate Table

| Resource   | Env Variable            | Unit            | Default Rate | Notes                              |
|------------|-------------------------|-----------------|--------------|------------------------------------|
| OCR pages  | `COST_PER_PAGE`         | per page        | 0.01         | Includes all fallback attempts     |
| GPU time   | `COST_PER_GPU_SECOND`   | per GPU-second  | 0.001        | Measured from task start to end    |
| Storage    | `COST_PER_GB_STORED`    | per GB/month    | 0.05         | Input + output combined            |
| API calls  | `COST_PER_API_CALL`     | per request     | 0.0001       | All endpoints                      |

### 1.4 Billing Period

- Default: **monthly** (configurable via `BILLING_PERIOD_DAYS`, default 30).
- Period boundaries are calendar-aligned at midnight UTC.
- Partial periods at start/end of service are prorated.

### 1.5 Tenant Isolation

- Costs are tracked per-tenant via `tenant_id` in `cost_tracking.CostTracker`.
- Each tenant accumulates independent counters for pages, GPU seconds, storage,
  and API calls.
- Per-tenant rate overrides are applied at report generation time (see Section 3).

### 1.6 Estimation vs Actual Billing

| Aspect             | This System                     | Operator Responsibility       |
|--------------------|---------------------------------|-------------------------------|
| Resource counting  | Automatic (pipeline hooks)      | --                            |
| Rate application   | Default rates from env vars     | Override per contract         |
| Invoice generation | Not provided                    | Operator billing system       |
| Payment collection | Not provided                    | Operator billing system       |
| Dispute resolution | Audit log via `custody.py`      | Operator process              |

### 1.7 BillingFormula Dataclass

The `BillingFormula` dataclass in `cost_tracking.py` codifies the locked formula:

| Field                       | Type   | Default                    |
|-----------------------------|--------|----------------------------|
| `version`                   | str    | `"1.0.0"`                  |
| `cpu_cost_per_page`         | float  | `COST_PER_PAGE` (0.01)     |
| `gpu_cost_per_page`         | float  | `COST_PER_GPU_SECOND` (0.001) |
| `storage_cost_per_gb_month` | float  | `COST_PER_GB_STORED` (0.05)|
| `api_call_cost`             | float  | `COST_PER_API_CALL` (0.0001)|
| `effective_date`            | str    | Current date (YYYY-MM-DD)  |
| `locked`                    | bool   | `True`                     |

The `locked` flag must be `True` for production use.  Validation via
`validate_billing_formula` enforces this constraint.

---

## 2. SLA Policy

### 2.1 Default SLA Tiers

The pipeline ships with three pre-defined SLA tiers.  The **Silver** tier is the
system default, matching the defaults in `sla_monitoring.py`.

| Metric        | Gold     | Silver (default) | Bronze   | Env Variable               |
|---------------|----------|------------------|----------|----------------------------|
| Availability  | 99.9%    | 99.5%            | 99.0%    | `SLO_AVAILABILITY_TARGET`  |
| P95 Latency   | 30s/page | 60s/page         | 120s/page| `SLO_P95_LATENCY_TARGET`   |
| Error Rate    | < 0.1%   | < 1.0%           | < 2.0%   | `SLO_ERROR_RATE_BUDGET`    |
| Throughput    | 100 PPM  | 10 PPM           | 5 PPM    | `SLO_THROUGHPUT_TARGET`    |
| Recovery Time | 120s     | 300s             | 600s     | `SLO_RECOVERY_TIME_TARGET` |

**PPM** = pages per minute.

### 2.2 SLATargets Dataclass

The `SLATargets` dataclass in `sla_monitoring.py` codifies the locked targets:

| Field                      | Type   | Default (Silver)                  |
|----------------------------|--------|-----------------------------------|
| `version`                  | str    | `"1.0.0"`                         |
| `uptime_target`            | float  | 0.995 (99.5%)                     |
| `p95_latency_ms`           | float  | 30000.0 (30s * 1000)              |
| `error_rate_target`        | float  | 0.01 (1%)                         |
| `throughput_ppm_target`    | float  | 10.0                              |
| `recovery_time_seconds`    | float  | 300.0                             |

### 2.3 Five Built-in SLO Definitions

| # | Name            | Metric          | Target    | Unit              | Comparison |
|---|-----------------|-----------------|-----------|-------------------|------------|
| 1 | Availability    | availability    | 99.5      | percent           | gte        |
| 2 | Throughput      | throughput      | 10.0      | pages_per_minute  | gte        |
| 3 | Error Rate      | error_rate      | 1.0       | percent           | lte        |
| 4 | P95 Latency     | p95_latency     | 30.0      | seconds           | lte        |
| 5 | Recovery Time   | recovery_time   | 300.0     | seconds           | lte        |

### 2.4 Measurement Window

- Default: **sliding 1-hour window** (`MetricsWindow(window_seconds=3600)`).
- The `SLAMonitor` default evaluation window is **24 hours** (`window_hours=24`).
- Both are configurable at construction time.
- Samples older than the window are pruned automatically on read.

### 2.5 Violation Escalation

| Severity | Condition                     | Action                        |
|----------|-------------------------------|-------------------------------|
| Warning  | Margin < 10% of target        | Log warning                   |
| Alert    | SLO breached (margin < 0)     | Prometheus alert fires        |
| Incident | 2+ SLOs breached for > 1 hour | Operator escalation required  |

Violations are stored in `SLAMonitor._violations` (capped at 10,000 records)
and exposed via `get_violations`.

### 2.6 SLA Exclusions

The following conditions are excluded from SLA calculations:

1. **Scheduled maintenance**: Pre-announced windows with > 24h notice.
2. **Force majeure**: Natural disasters, infrastructure provider outages.
3. **Customer-caused failures**: Malformed input, auth failures, rate-limit hits.
4. **Dependency failures**: External model servers, third-party APIs.

Exclusions must be documented in the audit log (`custody.py`) with reason codes.

---

## 3. Operator Configuration

### 3.1 Setting Per-Tenant Rate Overrides

Per-tenant rate overrides are applied by setting environment variables before the
API process starts, or by modifying the `BillingFormula` at runtime:

```python
from cost_tracking import BillingFormula

# Custom rates for a high-volume tenant
custom = BillingFormula(
    cpu_cost_per_page=0.005,       # 50% discount
    gpu_cost_per_page=0.0008,
    storage_cost_per_gb_month=0.03,
    api_call_cost=0.0001)
```

For bulk configuration, operators can use the `CostTracker` persistence file
(JSON) and pre-populate tenant usage records.

### 3.2 Setting Per-Tenant SLA Tier

```python
from sla_monitoring import SLAMonitor, SLODefinition

monitor = SLAMonitor

# Gold tier for tenant "acme"
gold_slos = [
    SLODefinition("Availability", "availability", 99.9, "percent", "gte"),
    SLODefinition("Throughput", "throughput", 100.0, "pages_per_minute", "gte"),
    SLODefinition("Error Rate", "error_rate", 0.1, "percent", "lte"),
    SLODefinition("P95 Latency", "p95_latency", 30.0, "seconds", "lte"),
    SLODefinition("Recovery Time", "recovery_time", 120.0, "seconds", "lte"),
]
monitor.set_tenant_slos("acme", gold_slos)
```

### 3.3 Generating Billing Reports

```python
from cost_tracking import get_tracker

tracker = get_tracker(persist_path="/data/cost_tracking.json")

# Single tenant
report = tracker.get_cost_report("acme")

# All tenants
reports = tracker.get_all_cost_reports
```

Reports include usage counters and estimated cost breakdown (page, GPU, storage,
API call costs plus total).

### 3.4 Generating SLA Violation Reports

```python
from sla_monitoring import get_monitor

monitor = get_monitor

# Evaluate a single tenant
report = monitor.evaluate_tenant("acme")
print(f"Compliant: {report.overall_compliant}")
print(f"Violations: {report.violation_count}")

# Write JSON report
monitor.write_report_json(report, "/data/reports/", "acme-sla.json")

# Get violation history
violations = monitor.get_violations(tenant_id="acme", limit=50)
```

### 3.5 Prometheus Metrics

The following metrics are exported via `api/prometheus.py`:

| Metric Name                  | Type  | Labels      | Description                       |
|------------------------------|-------|-------------|-----------------------------------|
| `ocr_cost_estimate_total`    | Gauge | `tenant_id` | Total estimated cost (USD)        |
| `ocr_tenant_gpu_seconds`     | Gauge | `tenant_id` | GPU seconds consumed              |
| `ocr_tenant_storage_bytes`   | Gauge | `tenant_id` | Storage bytes consumed            |
| `ocr_sla_compliance_pct`     | Gauge | `tenant_id` | Overall SLA compliance percentage |
| `ocr_sla_violation_count`    | Gauge | `tenant_id` | Current violation count           |
| `ocr_sla_availability_pct`   | Gauge | `tenant_id` | Current availability percentage   |
| `ocr_sla_p95_latency_seconds`| Gauge | `tenant_id` | Current P95 latency (seconds)     |

### 3.6 Grafana Dashboard Panels

The Grafana dashboard (`grafana-dashboard-configmap.yaml`) includes:

- **Cost per Tenant**: `ocr_cost_estimate_total` by tenant_id
- **Tenant Storage Consumption**: `ocr_tenant_storage_bytes` by tenant_id
- **SLA Compliance Rate**: `ocr_sla_compliance_pct` by tenant_id
- **SLA Breach History**: historical violation tracking

---

## 4. Validation

Run the policy validation script to verify alignment between this document,
the source code constants, and the monitoring infrastructure:

```bash
python scripts/validate_billing_sla.py --project-root .
python scripts/validate_billing_sla.py --project-root . --json --report billing-sla-report.json
```

The script verifies:

1. Cost tracking constants match the policy rate table
2. SLA monitoring constants match the policy tier definitions
3. Prometheus metrics exist in `api/prometheus.py`
4. Grafana dashboard has cost/SLA panels
5. Billing formula and SLA target dataclasses are valid

---

## 5. Amendment Process

1. Draft amendment with rationale and effective date.
2. Update `BILLING_FORMULA_VERSION` or `SLA_FORMULA_VERSION` (semver bump).
3. Update this document with new values.
4. Run `scripts/validate_billing_sla.py` to verify alignment.
5. Submit PR with architecture review label.
6. Merge only after review approval.

---

**Last Updated**: 2026-05-20
