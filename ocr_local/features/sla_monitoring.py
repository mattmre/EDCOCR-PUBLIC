"""SLA/SLO monitoring for OCR pipeline.

Defines service level objectives, tracks compliance metrics, and generates
violation reports.  Supports per-tenant SLA overrides and sliding time-window
evaluation.

Opt-in via ENABLE_SLA_MONITORING=true.  All metric windows are thread-safe
and automatically prune expired samples.

Default SLOs (overridable via env vars):
  - Availability >= 99.5%    (SLO_AVAILABILITY_TARGET)
  - Throughput >= 10 ppm     (SLO_THROUGHPUT_TARGET)
  - Error rate <= 1%         (SLO_ERROR_RATE_BUDGET)
  - P95 latency <= 30s       (SLO_P95_LATENCY_TARGET)
  - Recovery time <= 300s    (SLO_RECOVERY_TIME_TARGET)
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = [
    "ENABLE_SLA_MONITORING",
    "DEFAULT_AVAILABILITY_TARGET",
    "DEFAULT_THROUGHPUT_TARGET",
    "DEFAULT_ERROR_RATE_BUDGET",
    "DEFAULT_P95_LATENCY_TARGET",
    "DEFAULT_RECOVERY_TIME_TARGET",
    "SLA_FORMULA_VERSION",
    "SLATargets",
    "SLODefinition",
    "SLOStatus",
    "SLAReport",
    "MetricsWindow",
    "SLAMonitor",
    "get_sla_targets",
    "validate_sla_targets",
    "get_monitor",
    "reset_global_monitor",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------

ENABLE_SLA_MONITORING = os.environ.get(
    "ENABLE_SLA_MONITORING", "false"
).lower() == "true"

# ---------------------------------------------------------------------------
# Default SLO targets (configurable via environment)
# ---------------------------------------------------------------------------

DEFAULT_AVAILABILITY_TARGET = float(
    os.environ.get("SLO_AVAILABILITY_TARGET", "99.5")
)  # percent
DEFAULT_THROUGHPUT_TARGET = float(
    os.environ.get("SLO_THROUGHPUT_TARGET", "10.0")
)  # pages per minute
DEFAULT_ERROR_RATE_BUDGET = float(
    os.environ.get("SLO_ERROR_RATE_BUDGET", "1.0")
)  # percent
DEFAULT_P95_LATENCY_TARGET = float(
    os.environ.get("SLO_P95_LATENCY_TARGET", "30.0")
)  # seconds per page
DEFAULT_RECOVERY_TIME_TARGET = float(
    os.environ.get("SLO_RECOVERY_TIME_TARGET", "300.0")
)  # seconds

# ---------------------------------------------------------------------------
# Versioned SLA formula
# ---------------------------------------------------------------------------

SLA_FORMULA_VERSION = "1.0.0"

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass
class SLATargets:
    """Locked, versioned SLA/SLO target definitions.

    Codifies the production service level targets so they can be validated,
    serialized, and audited alongside billing formulas.  All values are
    sourced from the existing module-level defaults.
    """

    version: str = field(default=SLA_FORMULA_VERSION)
    uptime_target: float = field(default=DEFAULT_AVAILABILITY_TARGET / 100.0)
    p95_latency_ms: float = field(
        default=DEFAULT_P95_LATENCY_TARGET * 1000.0
    )
    error_rate_target: float = field(
        default=DEFAULT_ERROR_RATE_BUDGET / 100.0
    )
    throughput_ppm_target: float = field(default=DEFAULT_THROUGHPUT_TARGET)
    recovery_time_seconds: float = field(default=DEFAULT_RECOVERY_TIME_TARGET)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON encoding."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def get_sla_targets() -> SLATargets:
    """Return the current SLA targets with production defaults."""
    return SLATargets()


def validate_sla_targets(targets: SLATargets) -> list[str]:
    """Validate SLA targets and return a list of error messages.

    An empty list indicates valid targets.
    """
    errors: list[str] = []

    # Version format
    if not _SEMVER_RE.match(targets.version):
        errors.append(
            f"Invalid version format '{targets.version}': expected MAJOR.MINOR.PATCH"
        )

    # Uptime must be in (0, 1]
    if not (0.0 < targets.uptime_target <= 1.0):
        errors.append(
            f"uptime_target must be in (0, 1], got {targets.uptime_target}"
        )

    # P95 latency must be positive
    if targets.p95_latency_ms <= 0:
        errors.append(
            f"p95_latency_ms must be positive, got {targets.p95_latency_ms}"
        )

    # Error rate must be in [0, 1)
    if not (0.0 <= targets.error_rate_target < 1.0):
        errors.append(
            f"error_rate_target must be in [0, 1), got {targets.error_rate_target}"
        )

    # Throughput must be positive
    if targets.throughput_ppm_target <= 0:
        errors.append(
            f"throughput_ppm_target must be positive, got {targets.throughput_ppm_target}"
        )

    # Recovery time must be positive
    if targets.recovery_time_seconds <= 0:
        errors.append(
            f"recovery_time_seconds must be positive, got {targets.recovery_time_seconds}"
        )

    return errors


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SLODefinition:
    """A single Service Level Objective."""

    name: str
    metric: str  # "availability", "throughput", "error_rate", "p95_latency", "recovery_time"
    target: float
    unit: str  # "percent", "pages_per_minute", "seconds", etc.
    comparison: str = "gte"  # "gte" (target is minimum) or "lte" (target is maximum)


@dataclass
class SLOStatus:
    """Current compliance status for an SLO."""

    definition: SLODefinition
    current_value: float
    compliant: bool
    margin: float  # distance from target (positive = good, negative = violation)
    window_start: str  # ISO timestamp
    window_end: str
    sample_count: int


@dataclass
class SLAReport:
    """Full SLA compliance report for a tenant or system."""

    tenant_id: str  # "system" for global
    report_time: str
    window_hours: int
    slo_statuses: list  # list of SLOStatus
    overall_compliant: bool
    violation_count: int
    compliance_percentage: float  # percent of SLOs met


# ---------------------------------------------------------------------------
# Sliding metric window
# ---------------------------------------------------------------------------


class MetricsWindow:
    """Thread-safe sliding window for collecting metric samples.

    Each sample is a ``(timestamp, value)`` pair.  Samples older than
    ``window_seconds`` are pruned automatically on read operations.
    """

    def __init__(self, window_seconds: int = 3600):
        self._lock = threading.Lock()
        self._samples: list[tuple[float, float]] = []
        self._window_seconds = window_seconds

    @property
    def window_seconds(self) -> int:
        return self._window_seconds

    def add_sample(self, value: float, timestamp: float | None = None):
        """Record a metric sample at the given (or current) time."""
        ts = timestamp or time.time()
        with self._lock:
            self._samples.append((ts, value))
            self._prune()

    def _prune(self):
        """Remove samples older than the window.  Caller must hold lock."""
        cutoff = time.time() - self._window_seconds
        self._samples = [(t, v) for t, v in self._samples if t >= cutoff]

    def get_samples(self) -> list[tuple[float, float]]:
        """Return a snapshot of current in-window samples."""
        with self._lock:
            self._prune()
            return list(self._samples)

    def percentile(self, pct: float) -> float | None:
        """Return the *pct*-th percentile of sample values, or ``None``."""
        samples = self.get_samples()
        if not samples:
            return None
        values = sorted(v for _, v in samples)
        idx = int(len(values) * pct / 100.0)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def average(self) -> float | None:
        """Return the arithmetic mean of sample values, or ``None``."""
        samples = self.get_samples()
        if not samples:
            return None
        return sum(v for _, v in samples) / len(samples)

    def count(self) -> int:
        """Return the number of in-window samples."""
        return len(self.get_samples())

    def rate(self, success_threshold: float = 0.0) -> float | None:
        """Return the percentage of samples whose value exceeds *success_threshold*.

        Returns ``None`` when there are no samples.
        """
        samples = self.get_samples()
        if not samples:
            return None
        successes = sum(1 for _, v in samples if v > success_threshold)
        return (successes / len(samples)) * 100.0

    def clear(self):
        """Discard all samples."""
        with self._lock:
            self._samples.clear()


# ---------------------------------------------------------------------------
# SLA monitor engine
# ---------------------------------------------------------------------------


class SLAMonitor:
    """Thread-safe SLA/SLO monitoring engine.

    Tracks metric windows per tenant, evaluates SLO compliance, and
    generates violation reports.

    NOTE: This is an in-memory optimization cache. The authoritative SLA
    source of truth is computed from the Job table via api/slo.py.
    Sliding window data here may be lost on process restart.
    """

    def __init__(self, window_hours: int = 24, persist_path: str | None = None):
        self._lock = threading.Lock()
        self._window_seconds = window_hours * 3600
        self._window_hours = window_hours
        self._persist_path = Path(persist_path) if persist_path else None
        # Metric windows: {tenant_id: {metric_name: MetricsWindow}}
        self._windows: dict[str, dict[str, MetricsWindow]] = {}
        # SLO overrides per tenant: {tenant_id: [SLODefinition]}
        self._tenant_slos: dict[str, list[SLODefinition]] = {}
        # Violation history
        self._violations: list[dict] = []

    @property
    def window_hours(self) -> int:
        return self._window_hours

    # -- internal helpers ---------------------------------------------------

    def _get_window(self, tenant_id: str, metric: str) -> MetricsWindow:
        """Return (or create) a MetricsWindow for *tenant_id* / *metric*."""
        with self._lock:
            if tenant_id not in self._windows:
                self._windows[tenant_id] = {}
            if metric not in self._windows[tenant_id]:
                self._windows[tenant_id][metric] = MetricsWindow(
                    self._window_seconds
                )
            return self._windows[tenant_id][metric]

    # -- SLO definitions ----------------------------------------------------

    def get_default_slos(self) -> list[SLODefinition]:
        """Return the five built-in SLO definitions."""
        return [
            SLODefinition(
                "Availability",
                "availability",
                DEFAULT_AVAILABILITY_TARGET,
                "percent",
                "gte",
            ),
            SLODefinition(
                "Throughput",
                "throughput",
                DEFAULT_THROUGHPUT_TARGET,
                "pages_per_minute",
                "gte",
            ),
            SLODefinition(
                "Error Rate",
                "error_rate",
                DEFAULT_ERROR_RATE_BUDGET,
                "percent",
                "lte",
            ),
            SLODefinition(
                "P95 Latency",
                "p95_latency",
                DEFAULT_P95_LATENCY_TARGET,
                "seconds",
                "lte",
            ),
            SLODefinition(
                "Recovery Time",
                "recovery_time",
                DEFAULT_RECOVERY_TIME_TARGET,
                "seconds",
                "lte",
            ),
        ]

    def set_tenant_slos(self, tenant_id: str, slos: list[SLODefinition]):
        """Override SLO definitions for a specific tenant."""
        with self._lock:
            self._tenant_slos[tenant_id] = list(slos)

    def get_tenant_slos(self, tenant_id: str) -> list[SLODefinition]:
        """Return tenant-specific SLOs, falling back to defaults."""
        with self._lock:
            return list(
                self._tenant_slos.get(tenant_id, self.get_default_slos())
            )

    # -- recording methods --------------------------------------------------

    def record_request(
        self, tenant_id: str, success: bool, latency_seconds: float
    ):
        """Record a completed request (success/failure + latency)."""
        self._get_window(tenant_id, "availability").add_sample(
            1.0 if success else 0.0
        )
        self._get_window(tenant_id, "latency").add_sample(latency_seconds)
        self._get_window("system", "availability").add_sample(
            1.0 if success else 0.0
        )
        self._get_window("system", "latency").add_sample(latency_seconds)

    def record_throughput(self, tenant_id: str, pages_per_minute: float):
        """Record an instantaneous throughput sample."""
        self._get_window(tenant_id, "throughput").add_sample(pages_per_minute)
        self._get_window("system", "throughput").add_sample(pages_per_minute)

    def record_error(self, tenant_id: str):
        """Record a processing error (contributes to error-rate SLO)."""
        self._get_window(tenant_id, "error_rate").add_sample(1.0)
        self._get_window("system", "error_rate").add_sample(1.0)

    def record_success(self, tenant_id: str):
        """Record a processing success (contributes to error-rate SLO)."""
        self._get_window(tenant_id, "error_rate").add_sample(0.0)
        self._get_window("system", "error_rate").add_sample(0.0)

    def record_recovery(self, tenant_id: str, recovery_seconds: float):
        """Record a recovery-time measurement."""
        self._get_window(tenant_id, "recovery_time").add_sample(
            recovery_seconds
        )
        self._get_window("system", "recovery_time").add_sample(
            recovery_seconds
        )

    # -- evaluation ---------------------------------------------------------

    def _evaluate_slo(self, tenant_id: str, slo: SLODefinition) -> SLOStatus:
        """Evaluate a single SLO against current window data.

        When no samples exist for a metric, the SLO is assumed compliant
        (no evidence of violation).
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=self._window_hours)

        if slo.metric == "availability":
            window = self._get_window(tenant_id, "availability")
            current = window.rate(0.5)
            if current is None:
                current = 100.0  # no data = assume healthy
        elif slo.metric == "throughput":
            window = self._get_window(tenant_id, "throughput")
            avg = window.average()
            # No data = assume at target (no evidence of violation)
            current = avg if avg is not None else slo.target
        elif slo.metric == "error_rate":
            window = self._get_window(tenant_id, "error_rate")
            samples = window.get_samples()
            if samples:
                errors = sum(1 for _, v in samples if v > 0.5)
                current = (errors / len(samples)) * 100.0
            else:
                current = 0.0
        elif slo.metric == "p95_latency":
            window = self._get_window(tenant_id, "latency")
            p95 = window.percentile(95)
            current = p95 if p95 is not None else 0.0
        elif slo.metric == "recovery_time":
            window = self._get_window(tenant_id, "recovery_time")
            p95 = window.percentile(95)
            current = p95 if p95 is not None else 0.0
        else:
            window = self._get_window(tenant_id, slo.metric)
            avg = window.average()
            if avg is None:
                current = slo.target if slo.comparison == "gte" else 0.0
            else:
                current = avg

        if slo.comparison == "gte":
            compliant = current >= slo.target
            margin = current - slo.target
        else:  # lte
            compliant = current <= slo.target
            margin = slo.target - current

        return SLOStatus(
            definition=slo,
            current_value=round(current, 4),
            compliant=compliant,
            margin=round(margin, 4),
            window_start=window_start.isoformat(),
            window_end=now.isoformat(),
            sample_count=window.count(),
        )

    def evaluate_tenant(self, tenant_id: str) -> SLAReport:
        """Evaluate all SLOs for a tenant and return an SLAReport."""
        slos = self.get_tenant_slos(tenant_id)
        statuses = [self._evaluate_slo(tenant_id, slo) for slo in slos]
        violations = [s for s in statuses if not s.compliant]

        now = datetime.now(timezone.utc)

        # Record violations in history (capped at 10000 to prevent unbounded growth)
        with self._lock:
            for v in violations:
                self._violations.append(
                    {
                        "tenant_id": tenant_id,
                        "slo_name": v.definition.name,
                        "current_value": v.current_value,
                        "target": v.definition.target,
                        "timestamp": now.isoformat(),
                    }
                )
            if len(self._violations) > 10000:
                self._violations = self._violations[-10000:]

        total = len(statuses)
        compliant_count = total - len(violations)

        return SLAReport(
            tenant_id=tenant_id,
            report_time=now.isoformat(),
            window_hours=self._window_hours,
            slo_statuses=statuses,
            overall_compliant=len(violations) == 0,
            violation_count=len(violations),
            compliance_percentage=round(
                (compliant_count / total * 100) if total > 0 else 100.0, 2
            ),
        )

    def evaluate_system(self) -> SLAReport:
        """Evaluate all SLOs at the global (system) level."""
        return self.evaluate_tenant("system")

    # -- violation history --------------------------------------------------

    def get_violations(
        self, tenant_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Return recent violation records, optionally filtered by tenant."""
        with self._lock:
            if tenant_id:
                filtered = [
                    v
                    for v in self._violations
                    if v["tenant_id"] == tenant_id
                ]
                return filtered[-limit:]
            return self._violations[-limit:]

    # -- reporting ----------------------------------------------------------

    def write_report_json(
        self,
        report: SLAReport,
        output_dir: str,
        filename: str = "sla-report.json",
    ) -> str:
        """Write an SLA report as JSON to *output_dir*/*filename*.

        Uses atomic tmp-then-rename pattern.  Returns the final file path.
        Path traversal characters in *filename* are stripped.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sanitize filename: strip traversal sequences, path separators,
        # and any residual non-alphanumeric-only artifacts.
        safe_name = (
            filename.replace("..", "").replace("/", "_").replace("\\", "_")
        )
        safe_name = safe_name.strip("_. ")
        if not safe_name or safe_name == ".json":
            safe_name = "sla-report.json"
        filepath = output_path / safe_name

        data = {
            "tenant_id": report.tenant_id,
            "report_time": report.report_time,
            "window_hours": report.window_hours,
            "overall_compliant": report.overall_compliant,
            "violation_count": report.violation_count,
            "compliance_percentage": report.compliance_percentage,
            "slo_statuses": [
                {
                    "name": s.definition.name,
                    "metric": s.definition.metric,
                    "target": s.definition.target,
                    "unit": s.definition.unit,
                    "current_value": s.current_value,
                    "compliant": s.compliant,
                    "margin": s.margin,
                    "sample_count": s.sample_count,
                }
                for s in report.slo_statuses
            ],
        }

        tmp = filepath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(filepath)
        logger.info("SLA report written to %s", filepath)
        return str(filepath)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_monitor: SLAMonitor | None = None
_monitor_lock = threading.Lock()


def get_monitor(**kwargs) -> SLAMonitor:
    """Return the global SLAMonitor singleton, creating it on first call.

    Keyword arguments are forwarded to the ``SLAMonitor`` constructor only
    when the singleton does not yet exist.
    """
    global _global_monitor
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = SLAMonitor(**kwargs)
        return _global_monitor


def reset_global_monitor():
    """Reset the global SLAMonitor singleton (primarily for tests)."""
    global _global_monitor
    with _monitor_lock:
        _global_monitor = None
