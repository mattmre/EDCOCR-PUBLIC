"""Tenant lifecycle management — CRUD, API key management, resolution."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from api.database import Tenant, TenantApiKey, get_session_factory

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC time without tzinfo (matches project convention)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _generate_tenant_id() -> str:
    """Generate a unique tenant identifier: tenant_{hex12}."""
    return f"tenant_{uuid.uuid4().hex[:12]}"


def _generate_key_id() -> str:
    """Generate a unique API key identifier: key_{hex12}."""
    return f"key_{uuid.uuid4().hex[:12]}"


def _generate_api_key() -> str:
    """Generate a random 48-character API key: ocr_{urlsafe44}."""
    return f"ocr_{secrets.token_urlsafe(33)}"


def hash_api_key(raw_key: str) -> str:
    """Compute SHA-256 hex digest of a raw API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tenant CRUD
# ---------------------------------------------------------------------------


def create_tenant(
    *,
    name: str,
    display_name: Optional[str] = None,
    tier: str = "standard",
    max_concurrent_jobs: int = 4,
    max_pages_per_month: int = 10000,
    max_storage_bytes: int = 10 * 1024**3,
    allowed_features: Optional[list[str]] = None,
    admin_email: Optional[str] = None,
    session: Optional[Session] = None,
) -> Tenant:
    """Create a new tenant record and return it."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        tenant = Tenant(
            tenant_id=_generate_tenant_id(),
            name=name,
            display_name=display_name,
            status="active",
            tier=tier,
            created_at=_utcnow(),
            max_concurrent_jobs=max_concurrent_jobs,
            max_pages_per_month=max_pages_per_month,
            max_storage_bytes=max_storage_bytes,
            allowed_features=json.dumps(allowed_features or []),
            admin_email=admin_email,
        )
        session.add(tenant)
        session.commit()
        session.refresh(tenant)
        logger.info("Created tenant %s (%s)", tenant.tenant_id, name)
        return tenant
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def get_tenant(tenant_id: str, *, session: Optional[Session] = None) -> Optional[Tenant]:
    """Return a tenant by ID, or None if not found."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        return session.get(Tenant, tenant_id)
    finally:
        if own_session:
            session.close()


def list_tenants(
    status_filter: Optional[str] = None,
    *,
    session: Optional[Session] = None,
) -> list[Tenant]:
    """Return all tenants, optionally filtered by status."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        q = session.query(Tenant)
        if status_filter:
            q = q.filter(Tenant.status == status_filter)
        return q.order_by(Tenant.created_at.desc()).all()
    finally:
        if own_session:
            session.close()


def update_tenant(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
    **kwargs,
) -> Optional[Tenant]:
    """Update tenant fields. Returns updated tenant or None if not found."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        tenant = session.get(Tenant, tenant_id)
        if not tenant:
            return None

        allowed_fields = {
            "name", "display_name", "status", "tier",
            "max_concurrent_jobs", "max_pages_per_month",
            "max_storage_bytes", "allowed_features", "admin_email",
        }
        for key, value in kwargs.items():
            if key not in allowed_fields:
                continue
            if key == "allowed_features" and isinstance(value, list):
                value = json.dumps(value)
            setattr(tenant, key, value)

        tenant.updated_at = _utcnow()
        session.commit()
        session.refresh(tenant)
        logger.info("Updated tenant %s", tenant_id)
        return tenant
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def suspend_tenant(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> Optional[Tenant]:
    """Set tenant status to 'suspended'."""
    return update_tenant(tenant_id, session=session, status="suspended")


def activate_tenant(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> Optional[Tenant]:
    """Set tenant status to 'active'."""
    return update_tenant(tenant_id, session=session, status="active")


def delete_tenant(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> Optional[Tenant]:
    """Soft-delete a tenant and revoke all associated API keys.

    Sets tenant status to 'deleted' (data is preserved for audit trail)
    and revokes every active API key belonging to the tenant.
    Returns the updated tenant, or None if the tenant was not found.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        tenant = session.get(Tenant, tenant_id)
        if not tenant:
            return None

        tenant.status = "deleted"
        tenant.updated_at = _utcnow()

        # Revoke all active keys for this tenant
        active_keys = (
            session.query(TenantApiKey)
            .filter(
                TenantApiKey.tenant_id == tenant_id,
                TenantApiKey.status == "active",
            )
            .all()
        )
        revoked_count = 0
        for key in active_keys:
            key.status = "revoked"
            revoked_count += 1

        session.commit()
        session.refresh(tenant)
        logger.info(
            "Soft-deleted tenant %s, revoked %d API keys",
            tenant_id,
            revoked_count,
        )
        return tenant
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# Hard purge: cascading deletion of all tenant data
# ---------------------------------------------------------------------------


def purge_tenant_data(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> Optional[dict]:
    """Hard-delete all data associated with a tenant and remove the tenant record.

    Deletes Jobs, UsageRecords, TenantApiKeys, and the Tenant row itself.
    Returns a dict with deletion counts, or None if the tenant was not found.

    This is the API-layer equivalent of the coordinator ``purge_tenant``
    management command.  It operates on the SQLite-backed models only
    (Job, UsageRecord, TenantApiKey, Tenant).
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        tenant = session.get(Tenant, tenant_id)
        if not tenant:
            return None

        from api.database import Job, UsageRecord

        # Count before deletion for the response
        jobs_deleted = (
            session.query(Job)
            .filter(Job.tenant_id == tenant_id)
            .delete(synchronize_session="fetch")
        )
        usage_deleted = (
            session.query(UsageRecord)
            .filter(UsageRecord.tenant_id == tenant_id)
            .delete(synchronize_session="fetch")
        )
        keys_deleted = (
            session.query(TenantApiKey)
            .filter(TenantApiKey.tenant_id == tenant_id)
            .delete(synchronize_session="fetch")
        )

        # Delete the tenant record itself
        session.delete(tenant)
        session.commit()

        logger.info(
            "Purged tenant %s: %d jobs, %d usage records, %d API keys",
            tenant_id,
            jobs_deleted,
            usage_deleted,
            keys_deleted,
        )
        return {
            "tenant_id": tenant_id,
            "deleted": {
                "jobs": jobs_deleted,
                "usage_records": usage_deleted,
                "api_keys": keys_deleted,
            },
        }
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# Cost bridge: map database-backed usage to cost_tracking.py schema
# ---------------------------------------------------------------------------


def build_cost_tracking_bridge(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> Optional[dict]:
    """Return a cost report in the same schema as cost_tracking.TenantUsage.estimated_cost().

    Reads from the database-backed UsageRecord (api.usage) and formats
    the result to match the standalone cost_tracking.py schema, providing
    a bridge between the admin API and the standalone cost tracker.

    Returns None if no usage record exists for the current period.
    """
    from api.usage import get_usage

    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        usage = get_usage(tenant_id, session=session)
        if not usage:
            return None

        # Import standalone cost rates for compatibility
        try:
            from cost_tracking import (
                COST_PER_API_CALL,
                COST_PER_GB_STORED,
                COST_PER_GPU_SECOND,
                COST_PER_PAGE,
            )
        except ImportError:
            # Fallback defaults matching cost_tracking.py
            COST_PER_PAGE = 0.01
            COST_PER_GPU_SECOND = 0.001
            COST_PER_GB_STORED = 0.05
            COST_PER_API_CALL = 0.0001

        pages = int(usage.pages_processed or 0)
        storage_bytes = int(usage.storage_bytes_used or 0)
        api_calls = int(usage.api_calls or 0)
        processing_seconds = float(usage.processing_seconds or 0.0)
        storage_gb = storage_bytes / (1024**3)

        page_cost = pages * COST_PER_PAGE
        gpu_cost = processing_seconds * COST_PER_GPU_SECOND
        storage_cost = storage_gb * COST_PER_GB_STORED
        api_cost = api_calls * COST_PER_API_CALL

        return {
            "tenant_id": tenant_id,
            "usage": {
                "tenant_id": tenant_id,
                "pages_processed": pages,
                "gpu_seconds": processing_seconds,
                "storage_bytes": storage_bytes,
                "api_calls": api_calls,
                "jobs_submitted": int(usage.jobs_submitted or 0),
                "jobs_completed": 0,
                "jobs_failed": 0,
                "first_activity": "",
                "last_activity": "",
            },
            "cost": {
                "page_cost": round(page_cost, 4),
                "gpu_cost": round(gpu_cost, 4),
                "storage_cost": round(storage_cost, 4),
                "api_cost": round(api_cost, 4),
                "total_cost": round(
                    page_cost + gpu_cost + storage_cost + api_cost, 4
                ),
                "currency": "USD",
            },
        }
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# API Key management
# ---------------------------------------------------------------------------


def create_api_key(
    tenant_id: str,
    *,
    name: Optional[str] = None,
    permissions: Optional[list[str]] = None,
    expires_at: Optional[datetime] = None,
    session: Optional[Session] = None,
) -> tuple[str, str]:
    """Create a new API key for a tenant.

    Returns (key_id, raw_api_key). The raw key is returned only once
    and is never stored — only its SHA-256 hash is persisted.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        tenant = session.get(Tenant, tenant_id)
        if not tenant:
            raise ValueError(f"Tenant not found: {tenant_id}")

        raw_key = _generate_api_key()
        key_id = _generate_key_id()

        record = TenantApiKey(
            key_id=key_id,
            tenant_id=tenant_id,
            api_key_hash=hash_api_key(raw_key),
            name=name,
            status="active",
            permissions=json.dumps(permissions or ["submit", "read"]),
            created_at=_utcnow(),
            expires_at=expires_at,
        )
        session.add(record)
        session.commit()
        logger.info("Created API key %s for tenant %s", key_id, tenant_id)
        return key_id, raw_key
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def revoke_api_key(
    key_id: str,
    *,
    session: Optional[Session] = None,
) -> bool:
    """Revoke an API key. Returns True if key was found and revoked."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        record = session.get(TenantApiKey, key_id)
        if not record:
            return False
        record.status = "revoked"
        session.commit()
        logger.info("Revoked API key %s", key_id)
        return True
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def resolve_tenant_by_key(
    api_key_hash: str,
    *,
    session: Optional[Session] = None,
) -> Optional[tuple[Tenant, TenantApiKey]]:
    """Look up a tenant by the SHA-256 hash of the provided API key.

    Returns (Tenant, TenantApiKey) or None if no active match is found.
    Expired and revoked keys are excluded.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        key_record = (
            session.query(TenantApiKey)
            .filter(
                TenantApiKey.api_key_hash == api_key_hash,
                TenantApiKey.status == "active",
            )
            .first()
        )
        if not key_record:
            return None

        # Check expiry
        now = _utcnow()
        if key_record.expires_at and key_record.expires_at < now:
            return None

        tenant = session.get(Tenant, key_record.tenant_id)
        if not tenant or tenant.status != "active":
            return None

        # Avoid a write on every request under sustained traffic.
        if (
            key_record.last_used_at is None
            or now - key_record.last_used_at >= timedelta(minutes=5)
        ):
            key_record.last_used_at = now
            session.commit()

        return tenant, key_record
    finally:
        if own_session:
            session.close()
