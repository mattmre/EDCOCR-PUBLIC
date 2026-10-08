"""Engine capability catalog endpoint (Phase 2).

Exposes the built-in OCR engine catalog with license, lane, and routing
metadata so callers (operators, the UI) can see which engines exist, their
license posture, which lane (forensic-CTC vs AI-enhanced) each belongs to, and
which declared routing properties future policy can query. Static metadata;
catalog GET requires operator/viewer/admin (submit-only cannot browse).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.identity import require_role
from ocr_local.engines import catalog
from ocr_local.engines.capability import EngineCapability

router = APIRouter(tags=["engines"])


def _capability_to_dict(cap: EngineCapability) -> dict:
    return {
        "engine_id": cap.engine_id,
        "family": cap.family,
        "license_spdx": cap.license_spdx,
        "license_class": cap.license_class,
        "lane": cap.lane,
        "is_generative": cap.is_generative,
        "is_deterministic_ctc": cap.is_deterministic_ctc,
        "requires_cloud_egress": cap.requires_cloud_egress,
        "weights_redistributable": cap.weights_redistributable,
        "use_restrictions": cap.use_restrictions,
        "commercial_threshold": cap.commercial_threshold,
        "languages": list(cap.languages),
        "scripts": list(cap.scripts),
        "handwriting": cap.handwriting,
        "tables": cap.tables,
        "vertical": cap.vertical,
        "cost_class": cap.cost_class,
        "gpu_required": cap.gpu_required,
        "notes": cap.notes,
    }


def list_engines() -> dict:
    """List the built-in OCR engine catalog with license/lane/routing metadata."""
    engines = [_capability_to_dict(c) for c in catalog.BUILTIN_CAPABILITIES]
    return {
        "forensic_default": "paddle-ctc",
        "forensic_lane": [
            e["engine_id"] for e in engines if e["lane"] == "forensic_ctc"
        ],
        "ai_enhanced_lane": [
            e["engine_id"] for e in engines if e["lane"] == "ai_enhanced"
        ],
        "engines": engines,
    }


@router.get("/api/v1/engines", name="list_engines")
def list_engines_http(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
) -> dict:
    """Authenticated catalog GET; payload stays the static capability list."""
    del request
    return list_engines()
