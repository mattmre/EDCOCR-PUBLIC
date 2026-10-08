"""Governance: which engines a tenant/job may select.

Generalizes the translation suite's ``TenantPolicy`` + ``allow_nc_licensed``
pattern (``ocr_local/translation/policy.py``) from translation engines to all
pluggable OCR engines.  Pure stdlib, no Django, so the standalone pipeline and
the test suite can import it freely (docs/release-candidate.md).

The filter order is forensic-first by design: ``forensic_mode`` (default True)
drops every non-CTC/generative engine before any license consideration, so the
zero-generative-surface guarantee holds regardless of license.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from ocr_local.engines.capability import EngineCapability

# Stable reason codes -- embedded in custody ``LICENSE_BLOCKED`` /
# ``LANE_BOUNDARY_ENFORCED`` events; do not rename without a custody migration.
ADMITTED = "ADMITTED"
BLOCKED_FORENSIC_MODE = "BLOCKED_FORENSIC_MODE_NON_CTC"
BLOCKED_DENYLIST = "BLOCKED_DENYLIST"
BLOCKED_GENERATIVE = "BLOCKED_GENERATIVE_NOT_ALLOWED"
BLOCKED_NC_LICENSE = "BLOCKED_NC_LICENSE"
BLOCKED_COPYLEFT = "BLOCKED_COPYLEFT_NO_ENTERPRISE_KEY"
BLOCKED_GATED = "BLOCKED_GATED_LICENSE"
BLOCKED_THRESHOLD = "BLOCKED_GATED_THRESHOLD_EXCEEDED"
BLOCKED_CLOUD_EGRESS = "BLOCKED_CLOUD_EGRESS_NOT_ALLOWED"
BLOCKED_USE_RESTRICTIONS = "BLOCKED_USE_RESTRICTIONS_NOT_ACKNOWLEDGED"
BLOCKED_NOT_IN_ALLOWLIST = "BLOCKED_NOT_IN_ALLOWLIST"


@dataclass(frozen=True)
class EnginePolicy:
    """Per-tenant / per-job engine selection policy.

    ``forensic_mode`` defaults True -- the safe posture in which only
    deterministic CTC engines are eligible.  ``allow_nc_licensed`` mirrors the
    existing translation field (docs/release-candidate.md), generalized to all
    engines (e.g. LayoutLMv3 NC weights).
    """

    forensic_mode: bool = True
    allow_generative: bool = False
    allow_cloud_egress: bool = False
    allow_nc_licensed: bool = False
    allow_gated_licensed: bool = False
    gated_license_ack: bool = False
    allow_use_restrictions: bool = False
    enterprise_license_key: str | None = None
    allowed_engine_ids: tuple[str, ...] = ()
    blocked_engine_ids: tuple[str, ...] = ()
    preferred_engine_ids: tuple[str, ...] = ()
    max_monthly_mau_attested: int | None = None
    max_monthly_revenue_usd: int | None = None


@dataclass(frozen=True)
class EngineDecision:
    engine_id: str
    admitted: bool
    reason: str


def _threshold_exceeded(cap: EngineCapability, policy: EnginePolicy) -> bool:
    """True when a tenant-attested figure exceeds a GATED engine's threshold."""
    threshold = cap.commercial_threshold or {}
    mau = threshold.get("mau")
    revenue = threshold.get("revenue_usd")
    if mau is not None and policy.max_monthly_mau_attested is not None:
        if policy.max_monthly_mau_attested > mau:
            return True
    if revenue is not None and policy.max_monthly_revenue_usd is not None:
        if policy.max_monthly_revenue_usd > revenue:
            return True
    return False


def evaluate_engine(cap: EngineCapability, policy: EnginePolicy) -> EngineDecision:
    """Decide whether ``cap`` is admissible under ``policy``.

    Returns the first failing reason (filter order documented in the program
    doc, section 4.4) or :data:`ADMITTED`.
    """
    engine_id = cap.engine_id

    def deny(reason: str) -> EngineDecision:
        return EngineDecision(engine_id, False, reason)

    # 1. Forensic mode overrides everything: only CTC, non-generative engines.
    if policy.forensic_mode and not (cap.is_deterministic_ctc and not cap.is_generative):
        return deny(BLOCKED_FORENSIC_MODE)
    # 2. Explicit denylist.
    if engine_id in policy.blocked_engine_ids:
        return deny(BLOCKED_DENYLIST)
    # 3. Generative engines require an explicit opt-in.
    if cap.is_generative and not policy.allow_generative:
        return deny(BLOCKED_GENERATIVE)
    # 4. Non-commercial weights (generalized allow_nc_licensed).
    if cap.license_class == "NC" and not policy.allow_nc_licensed:
        return deny(BLOCKED_NC_LICENSE)
    # 5. Network copyleft (AGPL) requires a registered enterprise license key.
    if cap.license_class == "COPYLEFT_NET" and not policy.enterprise_license_key:
        return deny(BLOCKED_COPYLEFT)
    # 6. Revenue/MAU-gated or OpenRAIL: enterprise key bypasses; otherwise needs
    #    an acknowledged opt-in and the attested figures must stay under threshold.
    if cap.license_class == "GATED" and not policy.enterprise_license_key:
        if not (policy.allow_gated_licensed and policy.gated_license_ack):
            return deny(BLOCKED_GATED)
        if _threshold_exceeded(cap, policy):
            return deny(BLOCKED_THRESHOLD)
    # 7. Cloud egress opt-in.
    if cap.requires_cloud_egress and not policy.allow_cloud_egress:
        return deny(BLOCKED_CLOUD_EGRESS)
    # 8. Acceptable-use / behavioral restrictions require an explicit opt-in.
    if cap.use_restrictions and not policy.allow_use_restrictions:
        return deny(BLOCKED_USE_RESTRICTIONS)
    # 9. Allowlist (empty list = all eligible).
    if policy.allowed_engine_ids and engine_id not in policy.allowed_engine_ids:
        return deny(BLOCKED_NOT_IN_ALLOWLIST)
    return EngineDecision(engine_id, True, ADMITTED)


def rank_engines(
    caps: list[EngineCapability], policy: EnginePolicy
) -> list[EngineCapability]:
    """Order candidates: explicit preference first, else forensic-CTC first.

    An engine listed in ``preferred_engine_ids`` wins on its preference rank;
    ties (and unlisted engines) fall back to forensic-first then engine_id, so
    the deterministic CTC default is preferred whenever no explicit preference
    applies.
    """
    preferred = list(policy.preferred_engine_ids)

    def sort_key(cap: EngineCapability) -> tuple[int, int, str]:
        pref_rank = preferred.index(cap.engine_id) if cap.engine_id in preferred else len(preferred)
        forensic_first = 0 if (cap.is_deterministic_ctc and not cap.is_generative) else 1
        return (pref_rank, forensic_first, cap.engine_id)

    return sorted(caps, key=sort_key)


def filter_engines(
    caps: list[EngineCapability], policy: EnginePolicy
) -> list[EngineCapability]:
    """Return the admissible engines under ``policy``, ranked."""
    admitted = [cap for cap in caps if evaluate_engine(cap, policy).admitted]
    return rank_engines(admitted, policy)


def compute_policy_hash(policy: EnginePolicy) -> str:
    """SHA-256 of the canonical JSON of ``policy`` (for custody embedding).

    Mirrors ``ocr_local.translation.policy.compute_policy_hash`` so reviewers
    can prove which policy version admitted a given engine.
    """
    canonical = json.dumps(
        dataclasses.asdict(policy),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
