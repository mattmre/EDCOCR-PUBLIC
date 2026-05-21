"""API versioning utilities and surface registry.

Provides tools for tracking API stability tiers, generating OpenAPI
metadata, and validating backward compatibility.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = [
    "StabilityTier",
    "EndpointRecord",
    "API_SURFACE",
    "get_stable_endpoints",
    "get_beta_endpoints",
    "get_experimental_endpoints",
    "check_backward_compatibility",
    "get_api_version",
    "get_version_header",
]


class StabilityTier(enum.Enum):
    STABLE = "stable"
    BETA = "beta"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class EndpointRecord:
    method: str  # HTTP method
    path: str  # URL path pattern
    name: str  # endpoint function name
    tier: StabilityTier
    auth_required: bool = True
    since_version: str = "1.0.0"
    deprecated: bool = False
    deprecated_in: str | None = None


# Complete API surface registry
API_SURFACE: tuple[EndpointRecord, ...] = (
    # Tier 1: Stable — Job Management
    EndpointRecord("POST", "/api/v1/jobs", "submit_job", StabilityTier.STABLE),
    EndpointRecord("GET", "/api/v1/jobs", "list_jobs", StabilityTier.STABLE),
    EndpointRecord("GET", "/api/v1/jobs/{job_id}", "get_job_status", StabilityTier.STABLE),
    EndpointRecord("GET", "/api/v1/jobs/{job_id}/result", "get_job_result", StabilityTier.STABLE),
    EndpointRecord("GET", "/api/v1/jobs/{job_id}/result/download", "download_artifact", StabilityTier.STABLE),
    EndpointRecord("POST", "/api/v1/jobs/{job_id}/retry", "retry_job", StabilityTier.STABLE),
    EndpointRecord("DELETE", "/api/v1/jobs/{job_id}", "cancel_job", StabilityTier.STABLE),
    # Tier 1: Stable — Batch Management
    EndpointRecord("POST", "/api/v1/jobs/batch", "submit_batch", StabilityTier.STABLE),
    EndpointRecord("GET", "/api/v1/jobs/batch", "list_batches", StabilityTier.STABLE),
    EndpointRecord("GET", "/api/v1/jobs/batch/{batch_id}", "get_batch_status", StabilityTier.STABLE),
    EndpointRecord("DELETE", "/api/v1/jobs/batch/{batch_id}", "cancel_batch", StabilityTier.STABLE),
    EndpointRecord("POST", "/api/v1/jobs/batch/{batch_id}/retry", "retry_batch", StabilityTier.STABLE),
    # Tier 1: Stable — Health
    EndpointRecord("GET", "/api/v1/health", "health_check", StabilityTier.STABLE, auth_required=False),
    EndpointRecord("GET", "/api/v1/health/detailed", "detailed_health_check", StabilityTier.STABLE, auth_required=False),
    EndpointRecord("GET", "/api/v1/ready", "ready_check", StabilityTier.STABLE, auth_required=False),
    EndpointRecord("GET", "/api/v1/readiness", "readiness_check", StabilityTier.STABLE, auth_required=False),
    EndpointRecord("GET", "/api/v1/translation/readiness", "external_translation_readiness_check", StabilityTier.STABLE, auth_required=False),
    # Tier 2: Beta — Transforms
    EndpointRecord("GET", "/api/v1/transforms", "list_transforms", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/transforms/{operation_id}", "get_transform_metadata", StabilityTier.BETA),
    EndpointRecord("POST", "/api/v1/transforms/execute", "execute_transform", StabilityTier.BETA),
    # Tier 2: Beta — Stamps
    EndpointRecord("GET", "/api/v1/stamps", "list_stamps", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/stamps/{operation_id}", "get_stamp_metadata", StabilityTier.BETA),
    EndpointRecord("POST", "/api/v1/stamps/execute", "execute_stamp", StabilityTier.BETA),
    # Tier 2: Beta — Events & DLQ
    EndpointRecord("GET", "/api/v1/jobs/{job_id}/events", "get_job_events", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/webhooks/dlq", "list_dlq_entries", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/webhooks/dlq/{entry_id}", "get_dlq_entry_endpoint", StabilityTier.BETA),
    EndpointRecord("POST", "/api/v1/webhooks/dlq/{entry_id}/retry", "retry_dlq_entry", StabilityTier.BETA),
    # Tier 2: Beta — Output Contracts
    EndpointRecord("GET", "/api/v1/jobs/{job_id}/outputs", "list_job_outputs", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/jobs/{job_id}/outputs/{output_type}", "get_job_output", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/jobs/{job_id}/document-bundle", "get_job_document_bundle", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/jobs/{job_id}/evidence-bundle", "get_job_evidence_bundle", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/schemas", "list_schemas", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/schemas/{output_type}", "get_schema", StabilityTier.BETA),
    # Tier 2: Beta — Queue Operations
    EndpointRecord("GET", "/api/v1/queues/thresholds", "queue_thresholds_list", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/queues/{queue_name}/threshold", "queue_threshold_get", StabilityTier.BETA),
    EndpointRecord("PUT", "/api/v1/queues/{queue_name}/threshold", "queue_threshold_update", StabilityTier.BETA),
    # Tier 2: Beta — Semantic Search
    EndpointRecord("POST", "/api/v1/search/semantic", "semantic_search", StabilityTier.BETA),
    EndpointRecord("POST", "/api/v1/search/analyze", "analyze_document", StabilityTier.BETA),
    EndpointRecord("GET", "/api/v1/search/vlm/health", "vlm_health", StabilityTier.BETA, auth_required=False),
    # Tier 3: Experimental — Admin
    EndpointRecord("POST", "/api/v1/admin/tenants", "admin_create_tenant", StabilityTier.EXPERIMENTAL),
    EndpointRecord("GET", "/api/v1/admin/tenants", "admin_list_tenants", StabilityTier.EXPERIMENTAL),
    EndpointRecord("GET", "/api/v1/admin/tenants/{tenant_id}", "admin_get_tenant", StabilityTier.EXPERIMENTAL),
    EndpointRecord("PUT", "/api/v1/admin/tenants/{tenant_id}", "admin_update_tenant", StabilityTier.EXPERIMENTAL),
    EndpointRecord("POST", "/api/v1/admin/tenants/{tenant_id}/suspend", "admin_suspend_tenant", StabilityTier.EXPERIMENTAL),
    EndpointRecord("POST", "/api/v1/admin/tenants/{tenant_id}/activate", "admin_activate_tenant", StabilityTier.EXPERIMENTAL),
    EndpointRecord("POST", "/api/v1/admin/tenants/{tenant_id}/keys", "admin_create_api_key", StabilityTier.EXPERIMENTAL),
    EndpointRecord("DELETE", "/api/v1/admin/tenants/{tenant_id}/keys/{key_id}", "admin_revoke_api_key", StabilityTier.EXPERIMENTAL),
    EndpointRecord("GET", "/api/v1/admin/tenants/{tenant_id}/usage", "admin_get_usage", StabilityTier.EXPERIMENTAL),
    EndpointRecord("GET", "/api/v1/admin/tenants/{tenant_id}/slo", "admin_get_slo", StabilityTier.EXPERIMENTAL),
    # Tier 2: Beta — Review Queue
    EndpointRecord("GET", "/api/v1/review/queue", "list_review_queue", StabilityTier.BETA, since_version="1.2.0"),
    EndpointRecord("GET", "/api/v1/review/stats", "review_stats", StabilityTier.BETA, since_version="1.2.0"),
    EndpointRecord("GET", "/api/v1/review/{review_id}", "get_review_item", StabilityTier.BETA, since_version="1.2.0"),
    EndpointRecord("POST", "/api/v1/review/{review_id}/decision", "review_decision", StabilityTier.BETA, since_version="1.2.0"),
    EndpointRecord("GET", "/api/v1/review/rules", "list_review_rules", StabilityTier.BETA, since_version="1.2.0"),
    # Tier 2: Beta --- Entity / Extraction Recall
    EndpointRecord("GET", "/api/v1/entities", "search_entities", StabilityTier.BETA, since_version="1.2.0"),
    EndpointRecord("GET", "/api/v1/extractions", "search_extractions", StabilityTier.BETA, since_version="1.2.0"),
    EndpointRecord("GET", "/api/v1/recall/stats", "recall_stats", StabilityTier.BETA, since_version="1.2.0"),
)


def get_stable_endpoints() -> list[EndpointRecord]:
    return [e for e in API_SURFACE if e.tier == StabilityTier.STABLE]


def get_beta_endpoints() -> list[EndpointRecord]:
    return [e for e in API_SURFACE if e.tier == StabilityTier.BETA]


def get_experimental_endpoints() -> list[EndpointRecord]:
    return [e for e in API_SURFACE if e.tier == StabilityTier.EXPERIMENTAL]


def get_api_version() -> str:
    """Return API version string from version.py."""
    try:
        from ocr_local.config.version import __version__
        return __version__
    except ImportError:
        return "0.0.0"


def get_version_header() -> dict[str, str]:
    """Return headers to include in API responses for version tracking."""
    return {
        "X-API-Version": get_api_version(),
        "X-API-Stability": "v1-stable",
    }


@dataclass
class CompatibilityReport:
    """Result of backward compatibility check."""
    compatible: bool = True
    removed_endpoints: list[str] = field(default_factory=list)
    changed_methods: list[str] = field(default_factory=list)
    new_endpoints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_backward_compatibility(
    previous: tuple[EndpointRecord, ...],
    current: tuple[EndpointRecord, ...],
) -> CompatibilityReport:
    """Check if current API surface is backward compatible with previous.

    Only checks Tier 1 (stable) endpoints for breaking changes.
    """
    report = CompatibilityReport()

    prev_stable = {(e.method, e.path): e for e in previous if e.tier == StabilityTier.STABLE}
    curr_stable = {(e.method, e.path): e for e in current if e.tier == StabilityTier.STABLE}
    curr_all = {(e.method, e.path) for e in current}

    # Check for removed stable endpoints
    for key, endpoint in prev_stable.items():
        if key not in curr_stable:
            # Check if path exists with different method
            matching_paths = [k for k in curr_all if k[1] == key[1]]
            if matching_paths:
                report.changed_methods.append(
                    f"{endpoint.method} {endpoint.path} → methods changed"
                )
            else:
                report.removed_endpoints.append(f"{endpoint.method} {endpoint.path}")
            report.compatible = False

    # Check for new endpoints
    prev_all = {(e.method, e.path) for e in previous}
    for key in curr_all - prev_all:
        report.new_endpoints.append(f"{key[0]} {key[1]}")

    # Check for auth changes on stable endpoints
    for key, endpoint in curr_stable.items():
        if key in prev_stable:
            prev_ep = prev_stable[key]
            if prev_ep.auth_required != endpoint.auth_required:
                if endpoint.auth_required and not prev_ep.auth_required:
                    report.warnings.append(
                        f"{endpoint.method} {endpoint.path}: auth now required (was optional)"
                    )
                    report.compatible = False

    return report
