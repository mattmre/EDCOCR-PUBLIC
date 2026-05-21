"""Tenant policy for translation engine routing."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_local.translation.custody_adapter import ReasonCode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TenantPolicy:
    tenant_id: str
    allow_cloud_translation: bool = False
    require_local_only: bool = False
    # Empty list means "all engines allowed"; otherwise only listed engine
    # ids are eligible for routing.
    allowed_engine_ids: list[str] = field(default_factory=list)
    blocked_engine_ids: list[str] = field(default_factory=list)
    max_monthly_chars: int | None = None  # None = unlimited
    max_monthly_tokens: int | None = None
    # NLLB is CC-BY-NC-4.0 -- set False for commercial-use tenants so the
    # router skips NLLB-derived engines.
    allow_nllb_commercial: bool = True
    residency_region: str | None = None  # e.g. "eu-west-1"
    # Wave M2 PR B14 -- engine ids in priority order for tenant-aware
    # routing.  Empty list means "use the default ranking (local-first,
    # legal>standard>draft)".  This is independent of ``allowed_engine_ids``
    # which is purely a filter; ``preferred_engine_ids`` only re-orders the
    # candidates that survive the existing filter rules.
    preferred_engine_ids: list[str] = field(default_factory=list)

    @property
    def allow_nc_licensed(self) -> bool:
        """Public alias for ``allow_nllb_commercial`` ().

        The DB column on ``TranslationTenantConfig`` is named
        ``allow_nc_licensed`` and the public-facing the development guide guidance
        references the same name.  Internally the dataclass field is
        ``allow_nllb_commercial`` because the original commit shipped with
        that name; this read-only alias lets callers use either spelling
        without divergence.
        """
        return self.allow_nllb_commercial


class PolicyDenied(Exception):
    """Raised when an engine cannot be selected because of tenant policy
    or a privilege guard.

    Custody events are emitted *before* this exception is raised so the
    audit trail still records the rejection when callers handle the
    exception.
    """

    def __init__(self, reason_code: "ReasonCode", message: str = "") -> None:
        self.reason_code = reason_code
        self.message = message
        super().__init__(f"{reason_code}: {message}")


def load_tenant_policy(tenant_id: str) -> TenantPolicy:
    """Load tenant policy.

    Production deployments will resolve this via DB/config; the default
    here is permissive so single-tenant pilot deployments work without
    extra configuration.
    """
    return TenantPolicy(tenant_id=tenant_id)


def get_tenant_policy(tenant_id: str | None) -> TenantPolicy:
    """Hydrate :class:`TenantPolicy` from the ``TranslationTenantConfig`` row.

    Wave M2 hydration helper.  When ``tenant_id`` is ``None`` (single-tenant
    mode) or no row exists, returns the safe permissive default
    (``allow_cloud_translation=False`` -- callers can override per request).

    Performs a lazy Django import so callers without DJANGO_SETTINGS_MODULE
    configured (the standalone ``ocr_gpu_async`` pipeline, the test suite)
    can call this helper safely.

    Mapping from the persisted row to TenantPolicy:
    - ``allow_nc_licensed`` -> ``allow_nllb_commercial`` (NC-licensed weights
      are eligible only when the tenant explicitly opts in)
    - ``preferred_engines`` -> ``allowed_engine_ids`` (when non-empty, only
      these engine ids are ranked)
    - ``require_certified`` is a tenant *policy* flag and is intentionally
      not propagated into TenantPolicy -- the certified=False enforcement
      lives at the sidecar write path.
    """
    safe_default = TenantPolicy(tenant_id=tenant_id or "default")

    if tenant_id is None:
        return safe_default

    try:
        from jobs.models import (
            TranslationTenantConfig,  # type: ignore[import-not-found]
        )
    except Exception as exc:  # pragma: no cover - non-Django fallback
        logger.debug(
            "get_tenant_policy: jobs.models not importable (%s); using default",
            exc,
        )
        return safe_default

    try:
        row = TranslationTenantConfig.objects.filter(tenant_id=tenant_id).first()
    except Exception as exc:  # pragma: no cover - DB unavailable
        logger.warning(
            "get_tenant_policy: lookup failed for tenant=%s (%s); using default",
            tenant_id,
            exc,
        )
        return safe_default

    if row is None:
        return safe_default

    preferred = list(row.preferred_engines or [])
    return TenantPolicy(
        tenant_id=tenant_id,
        # NC-licensed weights (NLLB CC-BY-NC-4.0) are only eligible when
        # the tenant has explicitly opted in.
        allow_nllb_commercial=bool(row.allow_nc_licensed),
        # When preferred_engines is set, restrict ranking to that list.
        allowed_engine_ids=preferred,
        # Same list also drives priority ordering in the tenant-aware
        # router (PR B14 router_v2).  A tenant that lists ['opus_mt',
        # 'nllb_200'] gets opus_mt first when both pass filtering.
        preferred_engine_ids=list(preferred),
    )


def compute_policy_hash(policy: TenantPolicy) -> str:
    """SHA-256 of the canonical JSON representation of ``policy``.

    The hash is intended to be embedded in custody events so reviewers
    can prove which policy version made a routing decision.
    """
    canonical = json.dumps(
        dataclasses.asdict(policy),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
