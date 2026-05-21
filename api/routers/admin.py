"""Admin API endpoints for multi-tenant management.

All endpoints require the 'admin' permission in the caller's API key.
Registered only when ENABLE_MULTITENANCY is true.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.config import ENABLE_MULTITENANCY
from api.database import TenantApiKey, get_session_factory
from api.slo import build_tenant_slo_snapshot
from api.tenant_manager import (
    activate_tenant,
    build_cost_tracking_bridge,
    create_api_key,
    create_tenant,
    delete_tenant,
    get_tenant,
    list_tenants,
    purge_tenant_data,
    revoke_api_key,
    suspend_tenant,
    update_tenant,
)
from api.usage import build_cost_summary, get_usage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

_TENANT_ID_RE = re.compile(r"^tenant_[0-9a-f]{12}$")
_KEY_ID_RE = re.compile(r"^key_[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# Pydantic models for admin API
# ---------------------------------------------------------------------------


class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    display_name: Optional[str] = Field(None, max_length=256)
    tier: str = Field("standard", pattern=r"^(free|standard|enterprise)$")
    max_concurrent_jobs: int = Field(4, ge=1, le=100)
    max_pages_per_month: int = Field(10000, ge=100, le=10_000_000)
    max_storage_bytes: int = Field(10 * 1024**3, ge=1024**2)
    allowed_features: list[str] = Field(default_factory=list)
    admin_email: Optional[str] = Field(None, max_length=256)


class UpdateTenantRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    display_name: Optional[str] = Field(None, max_length=256)
    tier: Optional[str] = Field(None, pattern=r"^(free|standard|enterprise)$")
    max_concurrent_jobs: Optional[int] = Field(None, ge=1, le=100)
    max_pages_per_month: Optional[int] = Field(None, ge=100, le=10_000_000)
    max_storage_bytes: Optional[int] = Field(None, ge=1024**2)
    allowed_features: Optional[list[str]] = None
    admin_email: Optional[str] = Field(None, max_length=256)


class CreateApiKeyRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=256)
    permissions: list[str] = Field(default_factory=lambda: ["submit", "read"])
    expires_at: Optional[datetime] = None


class TenantResponse(BaseModel):
    tenant_id: str
    name: str
    display_name: Optional[str] = None
    status: str
    tier: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    max_concurrent_jobs: int
    max_pages_per_month: int
    max_storage_bytes: int
    allowed_features: list[str] = []
    admin_email: Optional[str] = None


class TenantDetailResponse(TenantResponse):
    usage: Optional[dict[str, Any]] = None


class UsageCostRatesResponse(BaseModel):
    per_page_usd: float = 0.0
    per_gib_ingested_usd: float = 0.0
    per_api_call_usd: float = 0.0
    per_processing_hour_usd: float = 0.0


class UsageCostSummaryResponse(BaseModel):
    currency: str = "USD"
    page_cost_usd: float = 0.0
    storage_ingest_cost_usd: float = 0.0
    api_call_cost_usd: float = 0.0
    processing_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    storage_gib_ingested: float = 0.0
    processing_hours: float = 0.0
    rates: UsageCostRatesResponse = Field(default_factory=UsageCostRatesResponse)


class ApiKeyCreatedResponse(BaseModel):
    key_id: str
    api_key: str  # raw key, shown only once
    name: Optional[str] = None
    permissions: list[str]
    created_at: datetime
    expires_at: Optional[datetime] = None


class UsageResponse(BaseModel):
    tenant_id: str
    period: str
    jobs_submitted: int = 0
    pages_processed: int = 0
    storage_bytes_used: int = 0
    api_calls: int = 0
    processing_seconds: float = 0.0
    estimated_costs: UsageCostSummaryResponse = Field(
        default_factory=UsageCostSummaryResponse
    )


class TenantSloTargetsResponse(BaseModel):
    success_rate_min: float = 0.95
    p95_processing_seconds_max: float = 1800.0


class TenantSloStatusResponse(BaseModel):
    success_rate_met: bool = True
    p95_processing_met: bool = True
    overall_met: bool = True


class TenantSloResponse(BaseModel):
    tenant_id: str
    window_hours: int
    window_start: datetime
    window_end: datetime
    jobs_total: int = 0
    terminal_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    active_jobs: int = 0
    pages_processed: int = 0
    success_rate: float = 1.0
    failure_rate: float = 0.0
    avg_processing_seconds: Optional[float] = None
    p95_processing_seconds: Optional[float] = None
    throughput_jobs_per_hour: float = 0.0
    throughput_pages_per_hour: float = 0.0
    targets: TenantSloTargetsResponse = Field(default_factory=TenantSloTargetsResponse)
    status: TenantSloStatusResponse = Field(default_factory=TenantSloStatusResponse)


class ErrorResponse(BaseModel):
    error: str
    message: str


class TenantPurgeDeletedCounts(BaseModel):
    jobs: int = 0
    usage_records: int = 0
    api_keys: int = 0


class TenantPurgeResponse(BaseModel):
    tenant_id: str
    deleted: TenantPurgeDeletedCounts


# ---------------------------------------------------------------------------
# Helper: extract caller's tenant and check admin permission
# ---------------------------------------------------------------------------


def _require_admin(
    request: Request,
    *,
    target_tenant_id: Optional[str] = None,
    require_platform_admin: bool = False,
) -> Optional[str]:
    """Verify admin permission and enforce tenant scope when applicable."""
    if not ENABLE_MULTITENANCY:
        raise HTTPException(status_code=404, detail="Multi-tenancy is not enabled")

    permissions = getattr(request.state, "tenant_permissions", None)
    if permissions is None:
        return None  # Legacy OCR_API_KEY acts as platform admin.

    if "admin" not in permissions:
        raise HTTPException(
            status_code=403,
            detail={"error": "forbidden", "message": "Admin permission required"},
        )

    caller_tenant_id = getattr(request.state, "tenant_id", None)
    is_platform_admin = "platform_admin" in permissions
    if require_platform_admin and not is_platform_admin:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "Platform admin permission required",
            },
        )

    if (
        target_tenant_id is not None
        and not is_platform_admin
        and caller_tenant_id != target_tenant_id
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "Admin access is limited to the caller's tenant",
            },
        )

    return None if is_platform_admin else caller_tenant_id


def _tenant_to_response(tenant) -> TenantResponse:
    """Convert a Tenant ORM object to a TenantResponse."""
    features = []
    if tenant.allowed_features:
        try:
            features = json.loads(tenant.allowed_features)
        except (json.JSONDecodeError, TypeError):
            features = []

    return TenantResponse(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        display_name=tenant.display_name,
        status=tenant.status,
        tier=tenant.tier,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
        max_concurrent_jobs=tenant.max_concurrent_jobs,
        max_pages_per_month=tenant.max_pages_per_month,
        max_storage_bytes=tenant.max_storage_bytes,
        allowed_features=features,
        admin_email=tenant.admin_email,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/tenants",
    response_model=TenantResponse,
    status_code=201,
    responses={403: {"model": ErrorResponse}},
)
async def admin_create_tenant(request: Request, body: CreateTenantRequest):
    """Create a new tenant."""
    _require_admin(request, require_platform_admin=True)
    session = get_session_factory()()
    try:
        tenant = create_tenant(
            name=body.name,
            display_name=body.display_name,
            tier=body.tier,
            max_concurrent_jobs=body.max_concurrent_jobs,
            max_pages_per_month=body.max_pages_per_month,
            max_storage_bytes=body.max_storage_bytes,
            allowed_features=body.allowed_features,
            admin_email=body.admin_email,
            session=session,
        )
        return _tenant_to_response(tenant)
    finally:
        session.close()


@router.get(
    "/tenants",
    response_model=list[TenantResponse],
    responses={403: {"model": ErrorResponse}},
)
async def admin_list_tenants(
    request: Request,
    status: Optional[str] = None,
):
    """List all tenants, optionally filtered by status."""
    caller_tenant_id = _require_admin(request)
    session = get_session_factory()()
    try:
        if caller_tenant_id is not None:
            tenant = get_tenant(caller_tenant_id, session=session)
            tenants = [tenant] if tenant and (status is None or tenant.status == status) else []
        else:
            tenants = list_tenants(status_filter=status, session=session)
        return [_tenant_to_response(t) for t in tenants]
    finally:
        session.close()


@router.get(
    "/tenants/{tenant_id}",
    response_model=TenantDetailResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_get_tenant(request: Request, tenant_id: str):
    """Get tenant details including current usage."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")

    session = get_session_factory()()
    try:
        tenant = get_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )
        _require_admin(request, target_tenant_id=tenant_id)

        resp = _tenant_to_response(tenant)
        usage = get_usage(tenant_id, session=session)
        usage_dict = None
        if usage:
            cost_summary = build_cost_summary(usage)
            usage_dict = {
                "period": usage.period,
                "jobs_submitted": usage.jobs_submitted,
                "pages_processed": usage.pages_processed,
                "storage_bytes_used": usage.storage_bytes_used,
                "api_calls": usage.api_calls,
                "processing_seconds": usage.processing_seconds,
                "estimated_costs": cost_summary,
            }

        return TenantDetailResponse(
            **resp.model_dump(),
            usage=usage_dict,
        )
    finally:
        session.close()


@router.put(
    "/tenants/{tenant_id}",
    response_model=TenantResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_update_tenant(request: Request, tenant_id: str, body: UpdateTenantRequest):
    """Update tenant configuration."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        updates = body.model_dump(exclude_unset=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        tenant = update_tenant(tenant_id, session=session, **updates)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )
        return _tenant_to_response(tenant)
    finally:
        session.close()


@router.post(
    "/tenants/{tenant_id}/suspend",
    response_model=TenantResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_suspend_tenant(request: Request, tenant_id: str):
    """Suspend a tenant (all API keys become inactive)."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        tenant = suspend_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )
        return _tenant_to_response(tenant)
    finally:
        session.close()


@router.post(
    "/tenants/{tenant_id}/activate",
    response_model=TenantResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_activate_tenant(request: Request, tenant_id: str):
    """Re-activate a suspended tenant."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        tenant = activate_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )
        return _tenant_to_response(tenant)
    finally:
        session.close()


@router.delete(
    "/tenants/{tenant_id}",
    response_model=TenantResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_delete_tenant(request: Request, tenant_id: str):
    """Soft-delete a tenant: marks status as 'deleted' and revokes all API keys.

    Data is preserved for audit trail.  Requires platform admin permission.
    """
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, require_platform_admin=True)

    session = get_session_factory()()
    try:
        tenant = delete_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )
        logger.info("Tenant %s soft-deleted via admin API", tenant_id)
        return _tenant_to_response(tenant)
    finally:
        session.close()


@router.delete(
    "/tenants/{tenant_id}/purge",
    response_model=TenantPurgeResponse,
    responses={
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def admin_purge_tenant(request: Request, tenant_id: str):
    """Hard-delete a tenant and all associated data (cascading cleanup).

    Permanently removes the tenant record along with all Jobs, UsageRecords,
    and TenantApiKeys.  Returns counts of deleted rows.

    Blocked when ``LITIGATION_HOLD`` environment variable is set to a truthy
    value (``1``, ``true``, or ``yes``).

    Requires platform admin permission.
    """
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, require_platform_admin=True)

    # Check litigation hold
    lit_hold = os.environ.get("LITIGATION_HOLD", "").lower()
    if lit_hold in ("1", "true", "yes"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "litigation_hold",
                "message": "LITIGATION_HOLD is active -- all data deletions are blocked.",
            },
        )

    session = get_session_factory()()
    try:
        result = purge_tenant_data(tenant_id, session=session)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )
        logger.info(
            "Tenant %s purged via admin API: %s",
            tenant_id,
            result["deleted"],
        )
        return TenantPurgeResponse(
            tenant_id=result["tenant_id"],
            deleted=TenantPurgeDeletedCounts(**result["deleted"]),
        )
    finally:
        session.close()


@router.get(
    "/tenants/{tenant_id}/cost-bridge",
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_get_cost_bridge(request: Request, tenant_id: str):
    """Return a cost report compatible with the standalone cost_tracking.py schema.

    Provides a bridge between the database-backed usage system and the
    standalone CostTracker format used by cost_tracking.py.
    """
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        tenant = get_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )

        report = build_cost_tracking_bridge(tenant_id, session=session)
        if not report:
            return {
                "tenant_id": tenant_id,
                "usage": {
                    "tenant_id": tenant_id,
                    "pages_processed": 0,
                    "gpu_seconds": 0.0,
                    "storage_bytes": 0,
                    "api_calls": 0,
                    "jobs_submitted": 0,
                    "jobs_completed": 0,
                    "jobs_failed": 0,
                    "first_activity": "",
                    "last_activity": "",
                },
                "cost": {
                    "page_cost": 0.0,
                    "gpu_cost": 0.0,
                    "storage_cost": 0.0,
                    "api_cost": 0.0,
                    "total_cost": 0.0,
                    "currency": "USD",
                },
            }
        return report
    finally:
        session.close()


@router.post(
    "/tenants/{tenant_id}/keys",
    response_model=ApiKeyCreatedResponse,
    status_code=201,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_create_api_key(request: Request, tenant_id: str, body: CreateApiKeyRequest):
    """Create a new API key for a tenant. The raw key is returned only once."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    caller_permissions = getattr(request.state, "tenant_permissions", None)
    if (
        caller_permissions is not None
        and "platform_admin" in body.permissions
        and "platform_admin" not in caller_permissions
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "Only platform admins can mint platform_admin keys",
            },
        )

    session = get_session_factory()()
    try:
        try:
            key_id, raw_key = create_api_key(
                tenant_id,
                name=body.name,
                permissions=body.permissions,
                expires_at=body.expires_at,
                session=session,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": str(exc)},
            )

        key_record = session.get(TenantApiKey, key_id)

        return ApiKeyCreatedResponse(
            key_id=key_id,
            api_key=raw_key,
            name=body.name,
            permissions=body.permissions,
            created_at=key_record.created_at if key_record else datetime.now(timezone.utc).replace(tzinfo=None),
            expires_at=body.expires_at,
        )
    finally:
        session.close()


@router.delete(
    "/tenants/{tenant_id}/keys/{key_id}",
    status_code=204,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_revoke_api_key(request: Request, tenant_id: str, key_id: str):
    """Revoke an API key."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    if not _KEY_ID_RE.match(key_id):
        raise HTTPException(status_code=400, detail="Invalid key_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        key_record = session.get(TenantApiKey, key_id)
        if not key_record or key_record.tenant_id != tenant_id:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"API key {key_id} not found"},
            )

        success = revoke_api_key(key_id, session=session)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"API key {key_id} not found"},
            )
    finally:
        session.close()


@router.get(
    "/tenants/{tenant_id}/usage",
    response_model=UsageResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_get_usage(
    request: Request,
    tenant_id: str,
    period: Optional[str] = None,
):
    """Get usage report for a tenant (defaults to current month)."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        tenant = get_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )

        usage = get_usage(tenant_id, period=period, session=session)
        if not usage:
            # Return zeroed usage for the requested period
            return UsageResponse(
                tenant_id=tenant_id,
                period=period or datetime.now(timezone.utc).strftime("%Y-%m"),
                estimated_costs=UsageCostSummaryResponse(
                    **build_cost_summary(None)
                ),
            )

        return UsageResponse(
            tenant_id=usage.tenant_id,
            period=usage.period,
            jobs_submitted=usage.jobs_submitted,
            pages_processed=usage.pages_processed,
            storage_bytes_used=usage.storage_bytes_used,
            api_calls=usage.api_calls,
            processing_seconds=usage.processing_seconds,
            estimated_costs=UsageCostSummaryResponse(
                **build_cost_summary(usage)
            ),
        )
    finally:
        session.close()


@router.get(
    "/tenants/{tenant_id}/slo",
    response_model=TenantSloResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_get_slo(
    request: Request,
    tenant_id: str,
    window_hours: Optional[int] = Query(None, ge=1),
):
    """Get a rolling tenant SLO snapshot."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        tenant = get_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )

        snapshot = build_tenant_slo_snapshot(
            tenant_id,
            window_hours=window_hours,
            session=session,
        )
        return TenantSloResponse(**snapshot)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Per-tenant rate limit management
# ---------------------------------------------------------------------------


class SetRateLimitRequest(BaseModel):
    rate_limit: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^\d+/(second|minute|hour|day|month|year)$",
        description="Rate limit string, e.g. '100/minute', '500/hour'.",
    )


class TenantRateLimitResponse(BaseModel):
    tenant_id: str
    rate_limit: Optional[str] = None


@router.put(
    "/tenants/{tenant_id}/rate-limit",
    response_model=TenantRateLimitResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def admin_set_tenant_rate_limit(
    request: Request,
    tenant_id: str,
    body: SetRateLimitRequest,
):
    """Set a per-tenant rate limit override.

    When configured, this rate limit replaces the global default for all
    API requests authenticated with this tenant's API keys.
    """
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        tenant = get_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )
    finally:
        session.close()

    from api.tenant_rate_limiter import set_tenant_rate_limit

    try:
        set_tenant_rate_limit(tenant_id, body.rate_limit)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_rate_limit", "message": str(exc)},
        )

    logger.info("Rate limit set for tenant %s: %s", tenant_id, body.rate_limit)
    return TenantRateLimitResponse(tenant_id=tenant_id, rate_limit=body.rate_limit)


@router.get(
    "/tenants/{tenant_id}/rate-limit",
    response_model=TenantRateLimitResponse,
    responses={
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def admin_get_tenant_rate_limit(
    request: Request,
    tenant_id: str,
):
    """Get the current per-tenant rate limit override (or null if unset)."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        tenant = get_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )
    finally:
        session.close()

    from api.tenant_rate_limiter import get_tenant_rate_limit

    current_limit = get_tenant_rate_limit(tenant_id)
    return TenantRateLimitResponse(tenant_id=tenant_id, rate_limit=current_limit)


@router.delete(
    "/tenants/{tenant_id}/rate-limit",
    response_model=TenantRateLimitResponse,
    responses={
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def admin_delete_tenant_rate_limit(
    request: Request,
    tenant_id: str,
):
    """Remove a per-tenant rate limit override (reverts to global default)."""
    if not _TENANT_ID_RE.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    _require_admin(request, target_tenant_id=tenant_id)

    session = get_session_factory()()
    try:
        tenant = get_tenant(tenant_id, session=session)
        if not tenant:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"Tenant {tenant_id} not found"},
            )
    finally:
        session.close()

    from api.tenant_rate_limiter import delete_tenant_rate_limit

    delete_tenant_rate_limit(tenant_id)
    logger.info("Rate limit removed for tenant %s", tenant_id)
    return TenantRateLimitResponse(tenant_id=tenant_id, rate_limit=None)
