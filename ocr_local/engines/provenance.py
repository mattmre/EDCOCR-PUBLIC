"""Phase 2: per-page engine provenance + boundary custody events.

Pure dict builders -- the caller (worker / coordinator) appends the returned
dict to the hash-chained custody log (custody.py). Each event embeds
``compute_policy_hash(policy)`` so a reviewer can prove which policy version
admitted (or blocked) a given engine for a given page.
"""
from __future__ import annotations

from typing import Any

from ocr_local.engines.capability import EngineCapability
from ocr_local.engines.governance import EnginePolicy, compute_policy_hash

ENGINE_PROVENANCE = "ENGINE_PROVENANCE"
LICENSE_BLOCKED = "LICENSE_BLOCKED"
LANE_BOUNDARY_ENFORCED = "LANE_BOUNDARY_ENFORCED"


def build_engine_provenance_event(
    *,
    page_id: str,
    capability: EngineCapability,
    policy: EnginePolicy,
    engine_version: str | None = None,
    model_weights_sha256: str | None = None,
    egress: str = "local",
    fell_back_from: str | None = None,
) -> dict[str, Any]:
    """Record which engine produced a page and that policy permitted it."""
    decision = "ADMITTED" if not fell_back_from else f"FELL_BACK_FROM:{fell_back_from}"
    return {
        "event": ENGINE_PROVENANCE,
        "page_id": page_id,
        "lane": capability.lane,
        "engine_id": capability.engine_id,
        "engine_version": engine_version,
        "license_spdx": capability.license_spdx,
        "license_class": capability.license_class,
        "is_generative": capability.is_generative,
        "model_weights_sha256": model_weights_sha256,
        "policy_hash": compute_policy_hash(policy),
        "egress": egress,
        "decision": decision,
    }


def build_license_blocked_event(
    *,
    capability: EngineCapability,
    policy: EnginePolicy,
    reason: str,
    page_id: str | None = None,
) -> dict[str, Any]:
    """Record that an engine was dropped for a license/policy reason.

    Must be emitted BEFORE any ``PolicyDenied`` is raised so the audit trail
    survives even when the exception is caught upstream (cf. gotcha #87).
    """
    return {
        "event": LICENSE_BLOCKED,
        "page_id": page_id,
        "engine_id": capability.engine_id,
        "license_spdx": capability.license_spdx,
        "license_class": capability.license_class,
        "reason": reason,
        "policy_hash": compute_policy_hash(policy),
    }


def build_lane_boundary_event(
    *,
    capability: EngineCapability,
    policy: EnginePolicy,
    page_id: str | None = None,
) -> dict[str, Any]:
    """Record that forensic_mode blocked a generative engine for a page.

    Proves the zero-generative-surface promise held.
    """
    return {
        "event": LANE_BOUNDARY_ENFORCED,
        "page_id": page_id,
        "engine_id": capability.engine_id,
        "lane": capability.lane,
        "is_generative": capability.is_generative,
        "policy_hash": compute_policy_hash(policy),
    }
