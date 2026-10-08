"""Tenant policy for translation engine routing."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ocr_local.translation.custody_adapter import ReasonCode

if TYPE_CHECKING:
    pass

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
    # LT-05 / S2-18: NC-licensed engines (NLLB) default refused for commercial
    # tenants; operators must set allow_nc_licensed / allow_nllb_commercial True.
    allow_nllb_commercial: bool = False
    # S2-04: when set (e.g. "eu-west-1" / "eu"), cloud engines are rejected
    # with RESIDENCY_VIOLATION until region-affinity cloud adapters exist.
    residency_region: str | None = None
    # S2-04: when True, delivery of uncertified translation results is blocked
    # (write path still forces certified=False — gotcha #88).
    require_certified: bool = False
    # Wave M2 PR B14 -- engine ids in priority order for tenant-aware
    # routing.  Empty list means "use the default ranking (local-first,
    # legal>standard>draft)".  This is independent of ``allowed_engine_ids``
    # which is purely a filter; ``preferred_engine_ids`` only re-orders the
    # candidates that survive the existing filter rules.
    preferred_engine_ids: list[str] = field(default_factory=list)

    @property
    def allow_nc_licensed(self) -> bool:
        """Public alias for ``allow_nllb_commercial`` (docs/release-candidate.md).

        The DB column on ``TranslationTenantConfig`` is named
        ``allow_nc_licensed`` and the public-facing docs/release-candidate.md guidance
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
    here permits local routing without extra configuration while refusing
    cloud translation and NC-licensed engines unless explicitly enabled.
    """
    return TenantPolicy(tenant_id=tenant_id)


def get_tenant_policy(tenant_id: str | None) -> TenantPolicy:
    """Hydrate :class:`TenantPolicy` from the ``TranslationTenantConfig`` row.

    Wave M2 hydration helper.  When ``tenant_id`` is ``None`` (single-tenant
    mode) or no row exists, returns the safe local default
    (``allow_cloud_translation=False`` and NC engines refused unless the
    tenant opts in).

    Performs a lazy Django import so callers without DJANGO_SETTINGS_MODULE
    configured (the standalone ``ocr_gpu_async`` pipeline, the test suite)
    can call this helper safely.

    Mapping from the persisted row to TenantPolicy:
    - ``allow_nc_licensed`` -> ``allow_nllb_commercial`` (NC-licensed weights
      are eligible only when the tenant explicitly opts in)
    - ``preferred_engines`` -> ``allowed_engine_ids`` (when non-empty, only
      these engine ids are ranked)
    - ``require_certified`` -> TenantPolicy.require_certified (S2-04 delivery gate)
    - ``residency_region`` -> TenantPolicy.residency_region (S2-04 cloud block)
    """
    safe_default = TenantPolicy(tenant_id=tenant_id or "default")
    # Post-sweep HIGH: production fail-closed hydrate — deny NC + cloud + require
    # local-only when DB/import fails (do not admit commercial NC engines).
    import os

    def _deny_all_policy(tid: str) -> TenantPolicy:
        return TenantPolicy(
            tenant_id=tid or "default",
            allow_cloud_translation=False,
            require_local_only=True,
            allow_nllb_commercial=False,
            allowed_engine_ids=[],
            blocked_engine_ids=[],
            require_certified=True,
        )

    prod = os.environ.get("DEPLOYMENT_ENV", "").strip().lower() in (
        "production",
        "prod",
    )

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
        if prod:
            return _deny_all_policy(tenant_id)
        return safe_default

    try:
        row = TranslationTenantConfig.objects.filter(tenant_id=tenant_id).first()
    except Exception as exc:  # pragma: no cover - DB unavailable
        logger.warning(
            "get_tenant_policy: lookup failed for tenant=%s (%s); using default",
            tenant_id,
            exc,
        )
        if prod:
            return _deny_all_policy(tenant_id)
        return safe_default

    if row is None:
        # PH1-17: missing config row in production is not an admit path —
        # deny-all (same as DB/import failure). Lab keeps safe_default so
        # tenants without a row still route local engines.
        if prod:
            logger.warning(
                "get_tenant_policy: no TranslationTenantConfig for tenant=%s "
                "under production; applying deny-all (PH1-17)",
                tenant_id,
            )
            return _deny_all_policy(tenant_id)
        return safe_default

    preferred = list(row.preferred_engines or [])
    residency = getattr(row, "residency_region", None) or None
    if isinstance(residency, str):
        residency = residency.strip() or None
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
        residency_region=residency,
        require_certified=bool(getattr(row, "require_certified", False)),
    )


def assert_certified_delivery_allowed(
    policy: TenantPolicy,
    *,
    certified: bool,
) -> None:
    """Fail closed when tenant requires certified output but payload is not.

    S2-04: ``require_certified`` is a delivery gate. Write path still forces
    ``certified=False`` at sidecar create (gotcha #88); delivery must block
    until review certify sets certified=True with real custody linkage.
    """
    if policy.require_certified and not certified:
        raise PolicyDenied(
            ReasonCode.TENANT_POLICY,
            (
                f"Tenant {policy.tenant_id!r} requires certified translation "
                "(require_certified=true); results not certified yet (S2-04)"
            ),
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
