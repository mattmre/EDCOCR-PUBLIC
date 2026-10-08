"""RC-01 / RC-04: multi-tenancy queryset scoping for Django admin ModelAdmins.

Under ``ENABLE_MULTITENANCY``, staff admin changelists must not fleet-scan
tenant-owned rows unless a superuser explicitly opts in with ``?fleet=1``.
Non-superuser staff (and superusers without fleet) require ``tenant_id``
query param or ``X-Tenant-Id`` header; missing scope returns an empty qs.

RC-04 staff membership: when ``STAFF_TENANT_MAP`` is configured (or production
enforcement is on), non-superuser staff may only scope to tenants listed for
their username — arbitrary ``?tenant_id=`` IDOR is refused.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Set


def multitenancy_enabled() -> bool:
    return os.environ.get("ENABLE_MULTITENANCY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def staff_tenant_map_required() -> bool:
    """True when non-superuser staff must be bound via STAFF_TENANT_MAP."""
    if os.environ.get("REQUIRE_STAFF_TENANT_MAP", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    dep = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
    return dep in ("production", "prod")


def parse_staff_tenant_map(raw: Optional[str] = None) -> dict[str, Set[str]]:
    """Parse STAFF_TENANT_MAP: JSON object or ``user:tenant[,tenant];user2:t``.

    Values may be a string or list of tenant ids. Usernames are casefolded.
    """
    text = raw if raw is not None else os.environ.get("STAFF_TENANT_MAP", "")
    text = (text or "").strip()
    if not text:
        return {}
    out: dict[str, Set[str]] = {}
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        for user, tenants in data.items():
            key = str(user).strip().casefold()
            if not key:
                continue
            if isinstance(tenants, str):
                vals = {t.strip() for t in tenants.split(",") if t.strip()}
            elif isinstance(tenants, (list, tuple, set)):
                vals = {str(t).strip() for t in tenants if str(t).strip()}
            else:
                vals = set()
            if vals:
                out[key] = vals
        return out
    # Compact form: alice:t1,t2;bob:t3
    for part in text.replace("\n", ";").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        user, rest = part.split(":", 1)
        key = user.strip().casefold()
        vals = {t.strip() for t in rest.split(",") if t.strip()}
        if key and vals:
            out[key] = vals
    return out


def staff_allowed_tenants(request: Any) -> Optional[Set[str]]:
    """Return allowed tenants for non-superuser staff, or None if unbound.

    * None — no membership enforcement (lab, map empty, not required)
    * empty set — user has no mapping (must get empty queryset)
    * non-empty set — only these tenants are allowed
    """
    mapping = parse_staff_tenant_map()
    if not mapping and not staff_tenant_map_required():
        return None
    user = getattr(request, "user", None)
    username = (
        getattr(user, "username", None)
        or getattr(user, "get_username", lambda: "")()
        or ""
    )
    username = str(username).strip().casefold()
    if not username:
        return set() if (mapping or staff_tenant_map_required()) else None
    if username not in mapping:
        # Map required or present but user missing → refuse all
        if mapping or staff_tenant_map_required():
            return set()
        return None
    return set(mapping[username])


def resolve_admin_tenant_id(request: Any) -> Optional[str]:
    """Return requested tenant scope, or None if unset."""
    tenant = (getattr(request, "GET", {}) or {}).get("tenant_id") or ""
    tenant = str(tenant).strip()
    if tenant:
        return tenant
    headers = getattr(request, "headers", None)
    if headers is not None:
        tenant = (headers.get("X-Tenant-Id") or "").strip()
        if tenant:
            return tenant
    meta = getattr(request, "META", None) or {}
    tenant = (meta.get("HTTP_X_TENANT_ID") or "").strip()
    return tenant or None


def admin_scope_mode(request: Any) -> str:
    """Return one of: ``disabled``, ``fleet``, ``tenant``, ``none``.

    * ``disabled`` — multi-tenancy off; caller should not filter
    * ``fleet`` — superuser explicit cross-tenant list
    * ``tenant`` — filter to resolve_admin_tenant_id(request)
    * ``none`` — refuse unscoped list (empty queryset)
    """
    if not multitenancy_enabled():
        return "disabled"
    user = getattr(request, "user", None)
    is_super = bool(getattr(user, "is_superuser", False))
    get = getattr(request, "GET", {}) or {}
    if is_super and str(get.get("fleet") or "") == "1":
        return "fleet"
    tenant = resolve_admin_tenant_id(request)
    allowed = None if is_super else staff_allowed_tenants(request)
    if allowed is not None:
        if not allowed:
            return "none"
        if tenant and tenant not in allowed:
            return "none"  # RC-04: refuse foreign tenant_id
        if not tenant:
            if len(allowed) == 1:
                # Implicit single binding
                return "tenant"
            return "none"
        return "tenant"
    if tenant:
        return "tenant"
    return "none"


def effective_admin_tenant_id(request: Any) -> Optional[str]:
    """Tenant id used for filtering after membership checks."""
    mode = admin_scope_mode(request)
    if mode != "tenant":
        return None
    tenant = resolve_admin_tenant_id(request)
    if tenant:
        return tenant
    user = getattr(request, "user", None)
    is_super = bool(getattr(user, "is_superuser", False))
    if is_super:
        return None
    allowed = staff_allowed_tenants(request)
    if allowed and len(allowed) == 1:
        return next(iter(allowed))
    return None


def apply_tenant_scope(
    qs: Any,
    request: Any,
    *,
    tenant_field: str = "tenant_id",
    job_tenant_field: str = "job__tenant_id",
    via_job: bool = False,
) -> Any:
    """Filter *qs* for RC-01/RC-04 admin multi-tenancy scope.

    Parameters
    ----------
    tenant_field:
        Direct column on the model (e.g. Job.tenant_id, GlossaryEntry).
    job_tenant_field:
        FK path when ``via_job=True`` (PageResult, CustodyEvent, extraction).
    via_job:
        When True, filter through the job FK tenant column.
    """
    mode = admin_scope_mode(request)
    if mode in ("disabled", "fleet"):
        return qs
    if mode == "none":
        return qs.none()
    tenant = effective_admin_tenant_id(request)
    if not tenant:
        return qs.none()
    field = job_tenant_field if via_job else tenant_field
    return qs.filter(**{field: tenant})
