"""Per-tenant resource usage tracking and cost estimation.

Tracks pages processed, processing time (GPU-seconds approximation),
storage consumption, and API calls per tenant.  Provides cost estimation
based on configurable unit costs.

Gated behind ``ENABLE_COST_TRACKING`` (default: disabled).
"""

import copy
import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "ENABLE_COST_TRACKING",
    "COST_PER_PAGE",
    "COST_PER_GPU_SECOND",
    "COST_PER_GB_STORED",
    "COST_PER_API_CALL",
    "BILLING_FORMULA_VERSION",
    "BillingFormula",
    "TenantUsage",
    "CostTracker",
    "get_billing_formula",
    "validate_billing_formula",
    "get_tracker",
    "reset_global_tracker",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------

ENABLE_COST_TRACKING = os.environ.get("ENABLE_COST_TRACKING", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Configurable unit costs (USD per unit)
# ---------------------------------------------------------------------------

COST_PER_PAGE = float(os.environ.get("COST_PER_PAGE", "0.01"))
COST_PER_GPU_SECOND = float(os.environ.get("COST_PER_GPU_SECOND", "0.001"))
COST_PER_GB_STORED = float(os.environ.get("COST_PER_GB_STORED", "0.05"))
COST_PER_API_CALL = float(os.environ.get("COST_PER_API_CALL", "0.0001"))

# ---------------------------------------------------------------------------
# Versioned billing formula
# ---------------------------------------------------------------------------

BILLING_FORMULA_VERSION = "1.0.0"

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass
class BillingFormula:
    """Locked, versioned billing formula for cost calculations.

    All rates are denominated in USD.  The ``locked`` flag indicates that this
    formula has been finalized and should not be modified at runtime.  The
    ``effective_date`` marks the start of applicability (ISO-8601 date string).
    """

    version: str = field(default=BILLING_FORMULA_VERSION)
    cpu_cost_per_page: float = field(default=COST_PER_PAGE)
    gpu_cost_per_page: float = field(default=COST_PER_GPU_SECOND)
    storage_cost_per_gb_month: float = field(default=COST_PER_GB_STORED)
    api_call_cost: float = field(default=COST_PER_API_CALL)
    effective_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    locked: bool = True

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON encoding."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def get_billing_formula() -> BillingFormula:
    """Return the current locked billing formula with production defaults."""
    return BillingFormula()


def validate_billing_formula(formula: BillingFormula) -> list[str]:
    """Validate a billing formula and return a list of error messages.

    An empty list indicates a valid formula.
    """
    errors: list[str] = []

    # Version format (semantic versioning)
    if not _SEMVER_RE.match(formula.version):
        errors.append(
            f"Invalid version format '{formula.version}': expected MAJOR.MINOR.PATCH"
        )

    # All rates must be positive
    rate_fields = {
        "cpu_cost_per_page": formula.cpu_cost_per_page,
        "gpu_cost_per_page": formula.gpu_cost_per_page,
        "storage_cost_per_gb_month": formula.storage_cost_per_gb_month,
        "api_call_cost": formula.api_call_cost,
    }
    for name, value in rate_fields.items():
        if value <= 0:
            errors.append(f"{name} must be positive, got {value}")

    # Effective date must be a valid ISO date
    try:
        datetime.strptime(formula.effective_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        errors.append(
            f"Invalid effective_date '{formula.effective_date}': expected YYYY-MM-DD"
        )

    # Locked flag must be True for production use
    if not formula.locked:
        errors.append("Formula is not locked; set locked=True for production use")

    return errors


# ---------------------------------------------------------------------------
# TenantUsage dataclass
# ---------------------------------------------------------------------------


@dataclass
class TenantUsage:
    """Accumulated resource usage for a single tenant."""

    tenant_id: str
    pages_processed: int = 0
    gpu_seconds: float = 0.0
    storage_bytes: int = 0
    api_calls: int = 0
    jobs_submitted: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    first_activity: str = ""  # ISO timestamp
    last_activity: str = ""  # ISO timestamp

    @property
    def storage_gb(self) -> float:
        """Return storage in gigabytes."""
        return self.storage_bytes / (1024**3)

    def estimated_cost(self) -> dict:
        """Calculate estimated cost breakdown based on current unit costs."""
        page_cost = self.pages_processed * COST_PER_PAGE
        gpu_cost = self.gpu_seconds * COST_PER_GPU_SECOND
        storage_cost = self.storage_gb * COST_PER_GB_STORED
        api_cost = self.api_calls * COST_PER_API_CALL
        return {
            "page_cost": round(page_cost, 4),
            "gpu_cost": round(gpu_cost, 4),
            "storage_cost": round(storage_cost, 4),
            "api_cost": round(api_cost, 4),
            "total_cost": round(page_cost + gpu_cost + storage_cost + api_cost, 4),
            "currency": "USD",
        }


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Thread-safe per-tenant resource usage tracker.

    Supports in-memory tracking with optional JSON persistence.
    Persistence uses atomic writes (write to ``.tmp``, then rename) to
    avoid corruption on crash.

    NOTE: This is an in-memory optimization cache. The authoritative cost
    source of truth is the SQLite UsageRecord table (api/database.py).
    Data here may be lost on process restart.
    """

    def __init__(self, persist_path=None):
        self._lock = threading.Lock()
        self._tenants: dict[str, TenantUsage] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    # -- internal helpers ---------------------------------------------------

    def _get_or_create(self, tenant_id: str) -> TenantUsage:
        if tenant_id not in self._tenants:
            self._tenants[tenant_id] = TenantUsage(tenant_id=tenant_id)
        return self._tenants[tenant_id]

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _touch_activity(self, usage: TenantUsage) -> None:
        """Update last_activity (and first_activity if unset)."""
        usage.last_activity = self._now_iso()
        if not usage.first_activity:
            usage.first_activity = usage.last_activity

    # -- recording methods --------------------------------------------------

    def record_pages(self, tenant_id: str, page_count: int) -> None:
        """Record pages processed for a tenant."""
        with self._lock:
            usage = self._get_or_create(tenant_id)
            usage.pages_processed += page_count
            self._touch_activity(usage)

    def record_gpu_time(self, tenant_id: str, seconds: float) -> None:
        """Record GPU processing time (in seconds) for a tenant."""
        with self._lock:
            usage = self._get_or_create(tenant_id)
            usage.gpu_seconds += seconds
            self._touch_activity(usage)

    def record_storage(self, tenant_id: str, bytes_added: int) -> None:
        """Record storage bytes added for a tenant."""
        with self._lock:
            usage = self._get_or_create(tenant_id)
            usage.storage_bytes += bytes_added
            self._touch_activity(usage)

    def record_api_call(self, tenant_id: str) -> None:
        """Record a single API call for a tenant."""
        with self._lock:
            usage = self._get_or_create(tenant_id)
            usage.api_calls += 1
            self._touch_activity(usage)

    def record_job_submitted(self, tenant_id: str) -> None:
        """Record a job submission for a tenant."""
        with self._lock:
            usage = self._get_or_create(tenant_id)
            usage.jobs_submitted += 1
            self._touch_activity(usage)

    def record_job_completed(self, tenant_id: str) -> None:
        """Record a completed job for a tenant."""
        with self._lock:
            usage = self._get_or_create(tenant_id)
            usage.jobs_completed += 1
            self._touch_activity(usage)

    def record_job_failed(self, tenant_id: str) -> None:
        """Record a failed job for a tenant."""
        with self._lock:
            usage = self._get_or_create(tenant_id)
            usage.jobs_failed += 1
            self._touch_activity(usage)

    # -- query methods ------------------------------------------------------

    def get_usage(self, tenant_id: str) -> TenantUsage | None:
        """Return a copy of usage for *tenant_id*, or ``None`` if unknown."""
        with self._lock:
            usage = self._tenants.get(tenant_id)
            return copy.copy(usage) if usage is not None else None

    def get_all_usage(self) -> dict[str, TenantUsage]:
        """Return a shallow copy of the tenant usage dict."""
        with self._lock:
            return dict(self._tenants)

    def get_cost_report(self, tenant_id: str) -> dict | None:
        """Return a combined usage + cost report for a single tenant."""
        with self._lock:
            usage = self._tenants.get(tenant_id)
            if not usage:
                return None
            return {
                "tenant_id": tenant_id,
                "usage": asdict(usage),
                "cost": usage.estimated_cost(),
            }

    def get_all_cost_reports(self) -> list[dict]:
        """Return cost reports for all tenants, sorted by tenant_id."""
        with self._lock:
            return [
                {"tenant_id": tid, "usage": asdict(u), "cost": u.estimated_cost()}
                for tid, u in sorted(self._tenants.items())
            ]

    # -- management ---------------------------------------------------------

    def reset_tenant(self, tenant_id: str) -> None:
        """Remove all tracked usage for *tenant_id*."""
        with self._lock:
            self._tenants.pop(tenant_id, None)

    # -- persistence --------------------------------------------------------

    def persist(self) -> None:
        """Write current state to disk using atomic rename."""
        if not self._persist_path:
            return
        with self._lock:
            data = {tid: asdict(u) for tid, u in self._tenants.items()}
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._persist_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._persist_path)
        logger.debug("Persisted cost tracking data to %s", self._persist_path)

    def _load(self) -> None:
        """Load previously persisted state from disk."""
        try:
            with open(self._persist_path, encoding="utf-8") as f:
                data = json.load(f)
            for tid, fields in data.items():
                self._tenants[tid] = TenantUsage(**fields)
            logger.debug(
                "Loaded cost tracking data for %d tenants from %s",
                len(self._tenants),
                self._persist_path,
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "Failed to load cost tracking data from %s: %s",
                self._persist_path,
                exc,
            )


# ---------------------------------------------------------------------------
# Module-level global tracker (singleton convenience API)
# ---------------------------------------------------------------------------

_global_tracker: CostTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker(persist_path=None) -> CostTracker:
    """Return the module-level global :class:`CostTracker`, creating it on first call."""
    global _global_tracker
    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = CostTracker(persist_path=persist_path)
        return _global_tracker


def reset_global_tracker() -> None:
    """Reset the module-level global tracker (primarily for tests)."""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None
