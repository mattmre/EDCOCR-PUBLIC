"""Usage tracking for multi-tenant OCR API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import api.config as config
from api.database import UsageRecord, get_session_factory

logger = logging.getLogger(__name__)


def _current_period() -> str:
    """Return the current year-month string (e.g. '2026-03')."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _get_or_create_usage(
    tenant_id: str,
    period: str,
    session: Session,
) -> UsageRecord:
    """Return existing usage record or create a new one for the period."""
    query = (
        session.query(UsageRecord)
        .filter(
            UsageRecord.tenant_id == tenant_id,
            UsageRecord.period == period,
        )
    )
    record = query.first()
    if record is None:
        record = UsageRecord(
            tenant_id=tenant_id,
            period=period,
            jobs_submitted=0,
            pages_processed=0,
            storage_bytes_used=0,
            api_calls=0,
            processing_seconds=0.0,
        )
        savepoint = session.begin_nested()
        try:
            session.add(record)
            session.flush()
            savepoint.commit()
        except IntegrityError:
            savepoint.rollback()
            session.expire_all()
            record = query.first()
            if record is None:
                raise
    return record


def record_job_submitted(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> None:
    """Increment the jobs_submitted counter for the current period."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        period = _current_period()
        record = _get_or_create_usage(tenant_id, period, session)
        record.jobs_submitted += 1
        if own_session:
            session.commit()
        else:
            session.flush()
    except Exception:
        if own_session:
            session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def record_pages_processed(
    tenant_id: str,
    page_count: int,
    *,
    session: Optional[Session] = None,
) -> None:
    """Add to the pages_processed counter for the current period."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        period = _current_period()
        record = _get_or_create_usage(tenant_id, period, session)
        record.pages_processed += page_count
        if own_session:
            session.commit()
        else:
            session.flush()
    except Exception:
        if own_session:
            session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def record_storage_used(
    tenant_id: str,
    bytes_delta: int,
    *,
    session: Optional[Session] = None,
) -> None:
    """Add to the storage_bytes_used counter for the current period."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        period = _current_period()
        record = _get_or_create_usage(tenant_id, period, session)
        record.storage_bytes_used += bytes_delta
        if own_session:
            session.commit()
        else:
            session.flush()
    except Exception:
        if own_session:
            session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def record_api_call(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> None:
    """Increment the api_calls counter for the current period."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        period = _current_period()
        record = _get_or_create_usage(tenant_id, period, session)
        record.api_calls += 1
        if own_session:
            session.commit()
        else:
            session.flush()
    except Exception:
        if own_session:
            session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def record_processing_seconds(
    tenant_id: str,
    processing_seconds: float,
    *,
    session: Optional[Session] = None,
    period: Optional[str] = None,
) -> None:
    """Add processing time to the monthly usage record."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        usage_period = period or _current_period()
        record = _get_or_create_usage(tenant_id, usage_period, session)
        record.processing_seconds += max(0.0, float(processing_seconds))
        if own_session:
            session.commit()
        else:
            session.flush()
    except Exception:
        if own_session:
            session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def get_cost_rate_snapshot() -> dict[str, float]:
    """Return the current internal rate inputs used for cost estimation."""
    return {
        "per_page_usd": float(config.TENANT_COST_PER_PAGE_USD),
        "per_gib_ingested_usd": float(config.TENANT_COST_PER_GIB_INGESTED_USD),
        "per_api_call_usd": float(config.TENANT_COST_PER_API_CALL_USD),
        "per_processing_hour_usd": float(config.TENANT_COST_PER_PROCESSING_HOUR_USD),
    }


def build_cost_summary(record: Optional[UsageRecord]) -> dict[str, object]:
    """Derive a transparent estimated-cost summary from raw usage."""
    rates = get_cost_rate_snapshot()
    if record is None:
        pages_processed = 0
        storage_bytes_used = 0
        api_calls = 0
        processing_seconds = 0.0
    else:
        pages_processed = int(record.pages_processed or 0)
        storage_bytes_used = int(record.storage_bytes_used or 0)
        api_calls = int(record.api_calls or 0)
        processing_seconds = float(record.processing_seconds or 0.0)

    storage_gib_ingested = storage_bytes_used / float(1024**3)
    processing_hours = processing_seconds / 3600.0

    page_cost = pages_processed * rates["per_page_usd"]
    storage_cost = storage_gib_ingested * rates["per_gib_ingested_usd"]
    api_cost = api_calls * rates["per_api_call_usd"]
    processing_cost = processing_hours * rates["per_processing_hour_usd"]
    total_cost = page_cost + storage_cost + api_cost + processing_cost

    return {
        "currency": "USD",
        "page_cost_usd": round(page_cost, 6),
        "storage_ingest_cost_usd": round(storage_cost, 6),
        "api_call_cost_usd": round(api_cost, 6),
        "processing_cost_usd": round(processing_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "storage_gib_ingested": round(storage_gib_ingested, 6),
        "processing_hours": round(processing_hours, 6),
        "rates": rates,
    }


def get_usage(
    tenant_id: str,
    period: Optional[str] = None,
    *,
    session: Optional[Session] = None,
) -> Optional[UsageRecord]:
    """Return the usage record for a tenant/period (defaults to current month)."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        if period is None:
            period = _current_period()
        return (
            session.query(UsageRecord)
            .filter(
                UsageRecord.tenant_id == tenant_id,
                UsageRecord.period == period,
            )
            .first()
        )
    finally:
        if own_session:
            session.close()
