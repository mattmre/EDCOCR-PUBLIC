"""Recall API endpoints for cross-document entity and extraction search."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.database import Job, get_session_factory
from api.deps import pagination_params
from api.entity_index import EntityIndex
from api.identity import require_role
from api.limits import get_default_rate, limiter
from api.models import (
    EntitySearchResponse,
    EntitySearchResult,
    ExtractionSearchResponse,
    ExtractionSearchResult,
    RecallStatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["recall"])

# Module-level singleton; lazily initialized to avoid import-time DB access.
_entity_index: EntityIndex | None = None
_entity_index_lock = threading.Lock()


def _get_index() -> EntityIndex:
    """Return the module-level EntityIndex singleton (thread-safe)."""
    global _entity_index
    if _entity_index is None:
        with _entity_index_lock:
            if _entity_index is None:
                _entity_index = EntityIndex()
    return _entity_index


def _request_tenant_id(request: Request) -> str | None:
    """Return tenant scope for authenticated tenant keys."""
    return getattr(request.state, "tenant_id", None)


def _tenant_job_ids(tenant_id: str) -> list[str]:
    """Return all job IDs belonging to a tenant."""
    session = get_session_factory()()
    try:
        rows = session.query(Job.job_id).filter(Job.tenant_id == tenant_id).all()
        return [r[0] for r in rows]
    finally:
        session.close()


def _verify_job_tenant(job_id: str, tenant_id: str | None) -> None:
    """Verify a specific job belongs to the tenant; raise 404 otherwise."""
    if tenant_id is None:
        return
    session = get_session_factory()()
    try:
        exists = (
            session.query(Job.job_id)
            .filter(Job.job_id == job_id, Job.tenant_id == tenant_id)
            .first()
        )
        if not exists:
            raise HTTPException(
                status_code=404,
                detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
            )
    finally:
        session.close()


# ------------------------------------------------------------------
# GET /api/v1/entities --- Search entities across all jobs
# ------------------------------------------------------------------


@router.get(
    "/entities",
    name="search_entities",
    response_model=EntitySearchResponse,
)
@limiter.limit(get_default_rate())
async def search_entities(
    request: Request,
    type: Optional[str] = Query(
        default=None, description="Filter by entity type (e.g. PERSON, DATE)"
    ),
    q: Optional[str] = Query(
        default=None, description="Text search query (LIKE matching)"
    ),
    job_id: Optional[str] = Query(
        default=None, description="Filter by job ID"
    ),
    min_confidence: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    pagination: tuple[int, int] = Depends(pagination_params),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Search indexed entities across all processed documents.

    Supports filtering by entity type, text content, job ID, and
    minimum confidence. Results are ordered by most recently indexed.

    Accepts ``limit``/``offset`` query parameters (canonical) or the
    deprecated ``page``/``per_page`` pair for backward compatibility.
    """
    limit, offset = pagination
    tenant_id = _request_tenant_id(request)

    # Tenant isolation: restrict results to jobs owned by this tenant
    allowed_job_ids = None
    if tenant_id is not None:
        if job_id:
            _verify_job_tenant(job_id, tenant_id)
        else:
            allowed_job_ids = _tenant_job_ids(tenant_id)

    index = _get_index()
    results, total = index.search_entities(
        entity_type=type,
        text_query=q,
        job_id=job_id,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset,
        allowed_job_ids=allowed_job_ids,
    )

    return EntitySearchResponse(
        results=[
            EntitySearchResult(
                entity_id=r.entity_id,
                job_id=r.job_id,
                entity_type=r.entity_type,
                text=r.text,
                confidence=r.confidence,
                source=r.source,
                page=r.page,
                document_name=r.document_name,
            )
            for r in results
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


# ------------------------------------------------------------------
# GET /api/v1/extractions --- Search extractions across all jobs
# ------------------------------------------------------------------


@router.get(
    "/extractions",
    name="search_extractions",
    response_model=ExtractionSearchResponse,
)
@limiter.limit(get_default_rate())
async def search_extractions(
    request: Request,
    field: Optional[str] = Query(
        default=None, description="Filter by field name (e.g. invoice_number, date)"
    ),
    q: Optional[str] = Query(
        default=None, description="Value search query (LIKE matching)"
    ),
    job_id: Optional[str] = Query(
        default=None, description="Filter by job ID"
    ),
    min_confidence: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    pagination: tuple[int, int] = Depends(pagination_params),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Search indexed key-value extractions across all processed documents.

    Supports filtering by field name, value content, job ID, and
    minimum confidence. Results are ordered by most recently indexed.

    Accepts ``limit``/``offset`` query parameters (canonical) or the
    deprecated ``page``/``per_page`` pair for backward compatibility.
    """
    limit, offset = pagination
    tenant_id = _request_tenant_id(request)

    # Tenant isolation: restrict results to jobs owned by this tenant
    allowed_job_ids = None
    if tenant_id is not None:
        if job_id:
            _verify_job_tenant(job_id, tenant_id)
        else:
            allowed_job_ids = _tenant_job_ids(tenant_id)

    index = _get_index()
    results, total = index.search_extractions(
        field_name=field,
        value_query=q,
        job_id=job_id,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset,
        allowed_job_ids=allowed_job_ids,
    )

    return ExtractionSearchResponse(
        results=[
            ExtractionSearchResult(
                extraction_id=r.extraction_id,
                job_id=r.job_id,
                field_name=r.field_name,
                field_value=r.field_value,
                confidence=r.confidence,
                page=r.page,
                document_name=r.document_name,
            )
            for r in results
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


# ------------------------------------------------------------------
# GET /api/v1/recall/stats --- Index statistics
# ------------------------------------------------------------------


@router.get(
    "/recall/stats",
    name="recall_stats",
    response_model=RecallStatsResponse,
)
@limiter.limit(get_default_rate())
async def recall_stats(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Return entity and extraction index statistics.

    Includes total counts, unique types/fields, and number of
    indexed jobs.
    """
    index = _get_index()
    stats = index.stats()
    return RecallStatsResponse(**stats)
