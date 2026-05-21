"""Quota enforcement for multi-tenant OCR API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from api.database import Job, Tenant, UsageRecord, get_session_factory

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when a tenant operation would exceed a configured quota limit."""

    def __init__(
        self,
        tenant_id: str,
        limit_type: str,
        current: int,
        maximum: int,
    ) -> None:
        self.tenant_id = tenant_id
        self.limit_type = limit_type
        self.current = current
        self.maximum = maximum
        super().__init__(
            f"Quota exceeded for tenant {tenant_id}: "
            f"{limit_type} ({current}/{maximum})"
        )


def _current_period() -> str:
    """Return the current year-month string (e.g. '2026-03')."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _get_usage(
    tenant_id: str,
    period: str,
    session: Session,
) -> Optional[UsageRecord]:
    """Return the usage record for a tenant/period, or None."""
    return (
        session.query(UsageRecord)
        .filter(
            UsageRecord.tenant_id == tenant_id,
            UsageRecord.period == period,
        )
        .first()
    )


def check_job_quota(
    tenant: Tenant,
    *,
    session: Optional[Session] = None,
) -> bool:
    """Return True if the tenant can submit another concurrent job.

    Raises QuotaExceededError if the limit is reached.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        active = (
            session.query(Job)
            .filter(
                Job.tenant_id == tenant.tenant_id,
                Job.status.in_(["submitted", "processing"]),
            )
            .count()
        )
        if active >= tenant.max_concurrent_jobs:
            raise QuotaExceededError(
                tenant_id=tenant.tenant_id,
                limit_type="concurrent_jobs",
                current=active,
                maximum=tenant.max_concurrent_jobs,
            )
        return True
    finally:
        if own_session:
            session.close()


def check_page_quota(
    tenant: Tenant,
    page_count: int,
    *,
    session: Optional[Session] = None,
) -> bool:
    """Return True if the tenant can process additional pages this month.

    Raises QuotaExceededError if the limit would be exceeded.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        period = _current_period()
        usage = _get_usage(tenant.tenant_id, period, session)
        current_pages = usage.pages_processed if usage else 0
        if current_pages + page_count > tenant.max_pages_per_month:
            raise QuotaExceededError(
                tenant_id=tenant.tenant_id,
                limit_type="pages_per_month",
                current=current_pages,
                maximum=tenant.max_pages_per_month,
            )
        return True
    finally:
        if own_session:
            session.close()


def check_storage_quota(
    tenant: Tenant,
    file_size: int,
    *,
    session: Optional[Session] = None,
) -> bool:
    """Return True if the tenant has storage headroom for the given file size.

    Raises QuotaExceededError if the limit would be exceeded.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        period = _current_period()
        usage = _get_usage(tenant.tenant_id, period, session)
        current_bytes = usage.storage_bytes_used if usage else 0
        if current_bytes + file_size > tenant.max_storage_bytes:
            raise QuotaExceededError(
                tenant_id=tenant.tenant_id,
                limit_type="storage_bytes",
                current=current_bytes,
                maximum=tenant.max_storage_bytes,
            )
        return True
    finally:
        if own_session:
            session.close()
