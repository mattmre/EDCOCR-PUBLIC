"""Translation tenant config + glossary admin endpoints (Plan B Wave M2).

Feature-gated by ``ENABLE_TRANSLATION_API``.  When the flag is unset, the
router is not registered at all (see ``api.main``).  This module provides
an *additional* hard 404 guard inside each handler so the router can be
mounted unconditionally during isolated tests but still refuse to serve
when the flag is off.

Auth follows the global ``api_key_middleware`` -- no explicit dependency
on ``require_api_key`` is needed.

CRITICAL: tenant isolation is enforced via the ``tenant_id`` path
parameter.  Filters use the path tenant directly so cross-tenant access
is impossible.

CRITICAL: ``require_certified`` is accepted as a *tenant policy* flag
but does NOT bypass the ``certified=False`` enforcement at translation
sidecar write time.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/api/v1/translation", tags=["translation"])


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------


def _translation_api_enabled() -> bool:
    """Resolve ``ENABLE_TRANSLATION_API`` at call time (test-friendly)."""
    return os.environ.get("ENABLE_TRANSLATION_API", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _ensure_enabled() -> None:
    """Raise 404 when the translation admin API is disabled."""
    if not _translation_api_enabled():
        raise HTTPException(
            status_code=404,
            detail="Translation admin API is disabled",
        )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TenantConfigUpsert(BaseModel):
    target_languages: list[str] = Field(default_factory=list)
    preferred_engines: list[str] = Field(default_factory=list)
    allow_nc_licensed: bool = False
    require_certified: bool = False
    default_quality_tier: str = "standard"

    @field_validator("default_quality_tier")
    @classmethod
    def _quality_tier(cls, v: str) -> str:
        if v not in {"draft", "standard", "legal"}:
            raise ValueError(
                "default_quality_tier must be one of: draft, standard, legal"
            )
        return v


class TenantConfigOut(BaseModel):
    tenant_id: str
    target_languages: list[str]
    preferred_engines: list[str]
    allow_nc_licensed: bool
    require_certified: bool
    default_quality_tier: str
    created_at: str | None = None
    updated_at: str | None = None


class GlossaryEntryIn(BaseModel):
    source_term: str = Field(min_length=1, max_length=500)
    target_term: str = Field(min_length=1, max_length=500)
    source_lang: str = Field(min_length=1, max_length=10)
    target_lang: str = Field(min_length=1, max_length=10)
    case_sensitive: bool = False
    is_regex: bool = False
    priority: int = 100
    notes: str | None = None


class GlossaryEntryUpdate(BaseModel):
    source_term: str | None = Field(default=None, min_length=1, max_length=500)
    target_term: str | None = Field(default=None, min_length=1, max_length=500)
    source_lang: str | None = Field(default=None, min_length=1, max_length=10)
    target_lang: str | None = Field(default=None, min_length=1, max_length=10)
    case_sensitive: bool | None = None
    is_regex: bool | None = None
    priority: int | None = None
    notes: str | None = None


class GlossaryEntryOut(BaseModel):
    id: int
    tenant_id: str
    source_term: str
    target_term: str
    source_lang: str
    target_lang: str
    case_sensitive: bool
    is_regex: bool
    priority: int
    notes: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class GlossaryListOut(BaseModel):
    entries: list[GlossaryEntryOut]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Lazy Django imports
# ---------------------------------------------------------------------------


def _get_models() -> tuple[Any, Any]:
    """Lazy import the Django models.

    Raises ``HTTPException(503)`` when Django is not available.
    """
    try:
        from jobs.models import (  # type: ignore[import-not-found]
            GlossaryEntry,
            TranslationTenantConfig,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Coordinator models unavailable: {exc}",
        ) from exc
    return TranslationTenantConfig, GlossaryEntry


def _config_to_dict(row: Any) -> dict[str, Any]:
    return {
        "tenant_id": row.tenant_id,
        "target_languages": list(row.target_languages or []),
        "preferred_engines": list(row.preferred_engines or []),
        "allow_nc_licensed": bool(row.allow_nc_licensed),
        "require_certified": bool(row.require_certified),
        "default_quality_tier": row.default_quality_tier or "standard",
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _entry_to_dict(row: Any) -> dict[str, Any]:
    return {
        "id": int(row.pk),
        "tenant_id": row.tenant_id,
        "source_term": row.source_term,
        "target_term": row.target_term,
        "source_lang": row.source_lang,
        "target_lang": row.target_lang,
        "case_sensitive": bool(row.case_sensitive),
        "is_regex": bool(row.is_regex),
        "priority": int(row.priority),
        "notes": row.notes or "",
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Tenant config endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/tenants/{tenant_id}/config",
    response_model=TenantConfigOut,
    name="get_tenant_translation_config",
)
def get_tenant_config(tenant_id: str) -> TenantConfigOut:
    """Read the translation config for ``tenant_id``.

    Returns 404 when no row exists for the tenant.
    """
    _ensure_enabled()
    TranslationTenantConfig, _ = _get_models()
    row = TranslationTenantConfig.objects.filter(tenant_id=tenant_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="tenant config not found")
    return TenantConfigOut(**_config_to_dict(row))


@router.put(
    "/tenants/{tenant_id}/config",
    response_model=TenantConfigOut,
    name="upsert_tenant_translation_config",
)
def upsert_tenant_config(
    tenant_id: str,
    payload: TenantConfigUpsert,
) -> TenantConfigOut:
    """Create or update the translation config for ``tenant_id``."""
    _ensure_enabled()
    TranslationTenantConfig, _ = _get_models()

    defaults = {
        "target_languages": payload.target_languages,
        "preferred_engines": payload.preferred_engines,
        "allow_nc_licensed": payload.allow_nc_licensed,
        "require_certified": payload.require_certified,
        "default_quality_tier": payload.default_quality_tier,
    }
    row, _created = TranslationTenantConfig.objects.update_or_create(
        tenant_id=tenant_id,
        defaults=defaults,
    )
    return TenantConfigOut(**_config_to_dict(row))


# ---------------------------------------------------------------------------
# Glossary endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/tenants/{tenant_id}/glossary",
    response_model=GlossaryListOut,
    name="list_tenant_glossary",
)
def list_glossary(
    tenant_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    source_lang: str | None = Query(None),
    target_lang: str | None = Query(None),
) -> GlossaryListOut:
    """Paginated list of glossary entries for ``tenant_id``."""
    _ensure_enabled()
    _, GlossaryEntry = _get_models()

    qs = GlossaryEntry.objects.filter(tenant_id=tenant_id)
    if source_lang:
        qs = qs.filter(source_lang=source_lang)
    if target_lang:
        qs = qs.filter(target_lang=target_lang)
    qs = qs.order_by("priority", "id")

    total = qs.count()
    offset = (page - 1) * page_size
    rows = list(qs[offset:offset + page_size])
    return GlossaryListOut(
        entries=[GlossaryEntryOut(**_entry_to_dict(r)) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/tenants/{tenant_id}/glossary",
    response_model=GlossaryEntryOut,
    status_code=201,
    name="create_tenant_glossary_entry",
)
def create_glossary_entry(
    tenant_id: str,
    payload: GlossaryEntryIn,
) -> GlossaryEntryOut:
    """Create a glossary entry under ``tenant_id``.

    Returns 409 on duplicate literal entries (unique constraint on
    tenant_id + source_term + source_lang + target_lang for non-regex
    entries).
    """
    _ensure_enabled()
    _, GlossaryEntry = _get_models()

    # Lazy import to avoid Django at module import time.
    try:
        from django.db import IntegrityError  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - non-Django fallback
        raise HTTPException(
            status_code=503,
            detail=f"Django unavailable: {exc}",
        ) from exc

    try:
        row = GlossaryEntry.objects.create(
            tenant_id=tenant_id,
            source_term=payload.source_term,
            target_term=payload.target_term,
            source_lang=payload.source_lang,
            target_lang=payload.target_lang,
            case_sensitive=payload.case_sensitive,
            is_regex=payload.is_regex,
            priority=payload.priority,
            notes=payload.notes or "",
        )
    except IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"duplicate glossary entry: {exc}",
        ) from exc
    return GlossaryEntryOut(**_entry_to_dict(row))


@router.patch(
    "/tenants/{tenant_id}/glossary/{entry_id}",
    response_model=GlossaryEntryOut,
    name="update_tenant_glossary_entry",
)
def update_glossary_entry(
    tenant_id: str,
    entry_id: int,
    payload: GlossaryEntryUpdate,
) -> GlossaryEntryOut:
    """Patch fields on a glossary entry under ``tenant_id``.

    Returns 404 when the entry does not exist or does not belong to the
    tenant.  Returns 409 when the patch creates a duplicate literal
    entry.
    """
    _ensure_enabled()
    _, GlossaryEntry = _get_models()

    try:
        from django.db import IntegrityError  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - non-Django fallback
        raise HTTPException(
            status_code=503,
            detail=f"Django unavailable: {exc}",
        ) from exc

    row = GlossaryEntry.objects.filter(
        tenant_id=tenant_id, pk=entry_id,
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="glossary entry not found")

    update_fields = []
    for field_name in (
        "source_term", "target_term", "source_lang", "target_lang",
        "case_sensitive", "is_regex", "priority", "notes",
    ):
        value = getattr(payload, field_name)
        if value is not None:
            setattr(row, field_name, value)
            update_fields.append(field_name)

    if update_fields:
        try:
            row.save(update_fields=update_fields + ["updated_at"])
        except IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"duplicate glossary entry: {exc}",
            ) from exc

    return GlossaryEntryOut(**_entry_to_dict(row))


@router.delete(
    "/tenants/{tenant_id}/glossary/{entry_id}",
    status_code=204,
    name="delete_tenant_glossary_entry",
)
def delete_glossary_entry(tenant_id: str, entry_id: int) -> None:
    """Delete a glossary entry under ``tenant_id``."""
    _ensure_enabled()
    _, GlossaryEntry = _get_models()

    deleted, _ = GlossaryEntry.objects.filter(
        tenant_id=tenant_id, pk=entry_id,
    ).delete()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="glossary entry not found")
    return None
