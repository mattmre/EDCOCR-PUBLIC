"""Engine capability + license/governance metadata.

One frozen record per pluggable OCR engine.  This is the single source of
truth the governance layer (:mod:`ocr_local.engines.governance`) filters on and
that custody ``ENGINE_PROVENANCE`` events embed.  Records key on the *exact*
engine/model id, never the family -- license terms differ per size/variant
(e.g. Qwen2.5-VL 7B is Apache-2.0 but 3B/72B are not).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# License class drives governance filtering (see governance.evaluate_engine).
#   PERMISSIVE    -- MIT / Apache-2.0 / BSD, commercial-clean
#   NC            -- non-commercial weights (e.g. LayoutLMv3 CC-BY-NC-SA-4.0)
#   COPYLEFT_NET  -- AGPL-style network copyleft (e.g. Ultralytics YOLO)
#   GATED         -- revenue/MAU-gated or OpenRAIL behavioral restriction
LicenseClass = Literal["PERMISSIVE", "NC", "COPYLEFT_NET", "GATED"]

# Lane the engine is eligible for.  The forensic lane is reserved for
# deterministic CTC engines with no generative surface.
Lane = Literal["forensic_ctc", "ai_enhanced"]

# Coarse routing cost class.  This is advisory metadata for policy/routing,
# not a benchmark result.
CostClass = Literal["low", "medium", "high", "unknown"]


@dataclass(frozen=True)
class EngineCapability:
    engine_id: str
    family: str
    license_spdx: str
    license_class: LicenseClass
    is_generative: bool
    is_deterministic_ctc: bool
    requires_cloud_egress: bool = False
    weights_redistributable: bool = True
    use_restrictions: bool = False
    # Threshold above which a GATED license requires a separate commercial
    # license, e.g. ``{"revenue_usd": 5_000_000}`` or ``{"mau": 100_000_000}``.
    commercial_threshold: dict[str, int] | None = None
    languages: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()
    handwriting: bool = False
    tables: bool = False
    vertical: bool = False
    cost_class: CostClass = "unknown"
    gpu_required: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.engine_id:
            raise ValueError("engine_id is required")
        if self.is_generative and self.is_deterministic_ctc:
            raise ValueError(
                f"engine {self.engine_id!r}: cannot be both generative and "
                "deterministic-CTC"
            )
        if self.scripts is None:
            raw_scripts = ()
        elif isinstance(self.scripts, str):
            raw_scripts = (self.scripts,)
        else:
            raw_scripts = self.scripts
        object.__setattr__(
            self,
            "scripts",
            tuple(
                str(item).strip().lower()
                for item in raw_scripts
                if str(item).strip()
            ),
        )
        if self.cost_class not in {"low", "medium", "high", "unknown"}:
            raise ValueError(
                f"engine {self.engine_id!r}: invalid cost_class "
                f"{self.cost_class!r}"
            )

    @property
    def lane(self) -> Lane:
        """Forensic lane requires deterministic CTC and no generative surface."""
        if self.is_deterministic_ctc and not self.is_generative:
            return "forensic_ctc"
        return "ai_enhanced"

    def supports_script(self, script: str | None) -> bool:
        """True when this capability declares support for ``script``.

        ``multi`` is the catalog wildcard for engines whose model card claims
        broad multilingual/script support.  Empty ``scripts`` means support has
        not been declared yet and should not satisfy a routing query.
        """
        normalized_script = script.strip().lower() if script else ""
        if not normalized_script:
            return True
        declared = set(self.scripts)
        return "multi" in declared or normalized_script in declared

    def satisfies_routing(
        self,
        *,
        script: str | None = None,
        requires_handwriting: bool = False,
        requires_tables: bool = False,
        requires_vertical: bool = False,
        allow_gpu_required: bool = True,
    ) -> bool:
        """Return whether declared routing metadata satisfies a query."""
        if not self.supports_script(script):
            return False
        if requires_handwriting and not self.handwriting:
            return False
        if requires_tables and not self.tables:
            return False
        if requires_vertical and not self.vertical:
            return False
        if not allow_gpu_required and self.gpu_required:
            return False
        return True
