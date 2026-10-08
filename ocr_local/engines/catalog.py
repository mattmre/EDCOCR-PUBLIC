"""Built-in :class:`EngineCapability` records for known engines.

Data only -- no engine implementations.  The forensic-lane engines
(``paddle-ctc``, ``tesseract``, ``easyocr``) describe the current production
cascade; the AI-enhanced records are the operator-greenlit candidates
(2026-06-19: PaddleOCR-VL, Granite-Docling, Surya) whose worker implementations
land in later phases.  ``dots-ocr`` is included for completeness but is
DEFERRED pending an Acceptable-Use Policy legal review.
"""
from __future__ import annotations

from ocr_local.engines.capability import EngineCapability
from ocr_local.engines.governance import EnginePolicy, evaluate_engine

# --- Forensic-CTC lane (current production cascade) -------------------------

PADDLE_CTC = EngineCapability(
    engine_id="paddle-ctc",
    family="paddleocr",
    license_spdx="Apache-2.0",
    license_class="PERMISSIVE",
    is_generative=False,
    is_deterministic_ctc=True,
    scripts=("multi",),
    cost_class="medium",
    notes="Production forensic default (PaddleOCR 2.9.1, CTC). Zero generative surface.",
)

TESSERACT = EngineCapability(
    engine_id="tesseract",
    family="tesseract",
    license_spdx="Apache-2.0",
    license_class="PERMISSIVE",
    is_generative=False,
    is_deterministic_ctc=True,
    scripts=("multi",),
    cost_class="low",
    notes="Forensic fallback engine (LSTM+CTC).",
)

EASYOCR = EngineCapability(
    engine_id="easyocr",
    family="easyocr",
    license_spdx="Apache-2.0",
    license_class="PERMISSIVE",
    is_generative=False,
    is_deterministic_ctc=True,
    scripts=("multi",),
    handwriting=True,
    cost_class="medium",
    notes="CRNN+CTC recognition. Optional forensic-lane alternative.",
)

# --- AI-enhanced lane (greenlit / conditional / deferred) -------------------

PADDLEOCR_VL = EngineCapability(
    engine_id="paddleocr-vl-0.9b",
    family="paddleocr-vl",
    license_spdx="Apache-2.0",
    license_class="PERMISSIVE",
    is_generative=True,
    is_deterministic_ctc=False,
    languages=("multi",),
    scripts=("multi",),
    handwriting=True,
    tables=True,
    cost_class="high",
    gpu_required=True,
    notes="GREENLIT (first AI engine). VLM 0.9B; same PaddlePaddle ecosystem; 109 languages.",
)

GRANITE_DOCLING = EngineCapability(
    engine_id="granite-docling-258m",
    family="granite-docling",
    license_spdx="Apache-2.0",
    license_class="PERMISSIVE",
    is_generative=True,
    is_deterministic_ctc=False,
    scripts=("latin",),
    tables=True,
    cost_class="medium",
    notes="GREENLIT (CPU pool). VLM 258M, CPU-feasible; DocTags -> tables/equations/code.",
)

SURYA = EngineCapability(
    engine_id="surya-ocr-2",
    family="surya",
    license_spdx="LicenseRef-AI-Pubs-OpenRAIL-M-modified",
    license_class="GATED",
    is_generative=False,
    is_deterministic_ctc=False,
    use_restrictions=True,
    weights_redistributable=False,
    commercial_threshold={"revenue_usd": 5_000_000},
    scripts=("multi",),
    cost_class="medium",
    notes=(
        "GREENLIT (operator-approved 2026-06-19) for the AI-enhanced lane. "
        "Weights remain OpenRAIL-gated (<$5M tier), so admission still requires "
        "allow_gated_licensed + gated_license_ack (or an enterprise key). Stays "
        "AI-enhanced, not forensic, pending a recognition-head CTC review."
    ),
)

DOTS_OCR = EngineCapability(
    engine_id="dots-ocr",
    family="dots.ocr",
    license_spdx="MIT",
    license_class="PERMISSIVE",
    is_generative=True,
    is_deterministic_ctc=False,
    use_restrictions=True,
    scripts=("multi",),
    tables=True,
    cost_class="high",
    gpu_required=True,
    notes=(
        "DEFERRED: MIT + Acceptable-Use Policy restricting ID/medical/biometric/"
        "GDPR/HIPAA extraction. Legal read pending before any use."
    ),
)

BUILTIN_CAPABILITIES: tuple[EngineCapability, ...] = (
    PADDLE_CTC,
    TESSERACT,
    EASYOCR,
    PADDLEOCR_VL,
    GRANITE_DOCLING,
    SURYA,
    DOTS_OCR,
)

CAPABILITY_BY_ID: dict[str, EngineCapability] = {
    cap.engine_id: cap for cap in BUILTIN_CAPABILITIES
}


def get_capability(engine_id: str) -> EngineCapability:
    """Look up a built-in capability record by exact engine id."""
    return CAPABILITY_BY_ID[engine_id]


def find_capabilities_for_routing(
    *,
    script: str | None = None,
    requires_handwriting: bool = False,
    requires_tables: bool = False,
    requires_vertical: bool = False,
    allow_gpu_required: bool = True,
    include_restricted: bool = False,
    policy: EnginePolicy | None = None,
    capabilities: tuple[EngineCapability, ...] = BUILTIN_CAPABILITIES,
) -> tuple[EngineCapability, ...]:
    """Return route-eligible capabilities whose declared metadata matches.

    The default excludes ``use_restrictions`` engines such as ``dots-ocr`` so
    deferred/legal-pending inventory cannot become an ordinary route candidate.
    """
    engine_policy = policy or EnginePolicy()
    return tuple(
        cap
        for cap in capabilities
        if evaluate_engine(cap, engine_policy).admitted
        and (include_restricted or not cap.use_restrictions)
        and cap.satisfies_routing(
            script=script,
            requires_handwriting=requires_handwriting,
            requires_tables=requires_tables,
            requires_vertical=requires_vertical,
            allow_gpu_required=allow_gpu_required,
        )
    )
