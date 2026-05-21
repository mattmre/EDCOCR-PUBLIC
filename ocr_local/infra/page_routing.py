"""Smart page routing for the OCR pipeline.

Routes individual pages to different processing backends based on
complexity analysis of page features.  The :class:`PageRouter` evaluates
a prioritised list of :class:`RoutingRule` objects against extracted
:class:`PageFeatures` and emits a :class:`RoutingDecision` for each page.

Built-in rules handle common cases (handwritten content, tables, very
small pages, high/low complexity) while callers may register arbitrary
custom rules.

Typical usage::

    router = PageRouter()
    features = PageFeatures(page_number=1, width=2480, height=3508,
                            dpi=300, file_size_bytes=524288,
                            estimated_text_density=0.6)
    decision = router.route_page(features)
    print(decision.target, decision.reason)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RoutingTarget(Enum):
    """Available processing backends for page routing."""

    GPU_PADDLE = "gpu_paddle"
    GPU_TESSERACT = "gpu_tesseract"
    CPU_PADDLE = "cpu_paddle"
    CPU_TESSERACT = "cpu_tesseract"
    CPU_ONNX = "cpu_onnx"
    SKIP = "skip"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PageFeatures:
    """Extracted features of a single page used for routing decisions.

    Attributes:
        page_number: 1-based page index within the document.
        width: Page width in pixels.
        height: Page height in pixels.
        dpi: Dots per inch of the page image.
        file_size_bytes: Size of the page image on disk (bytes).
        estimated_text_density: Fraction of the page area covered by text
            (0.0 = no text, 1.0 = fully covered).
        has_tables: Whether the page contains tabular structures.
        has_images: Whether the page contains embedded images.
        is_handwritten: Whether the page is primarily handwritten.
        language: ISO-639 language code (e.g. ``"en"``).
        complexity_score: Overall complexity estimate (0.0–1.0).
    """

    page_number: int
    width: int
    height: int
    dpi: int = 300
    file_size_bytes: int = 0
    estimated_text_density: float = 0.5
    has_tables: bool = False
    has_images: bool = False
    is_handwritten: bool = False
    language: str = "en"
    complexity_score: float = 0.5


@dataclass
class RoutingDecision:
    """Outcome of routing a single page.

    Attributes:
        page_number: The page that was routed.
        target: The selected processing backend.
        confidence: Confidence in the routing decision (0.0–1.0).
        reason: Human-readable explanation of why this target was chosen.
        estimated_duration_ms: Rough estimate of processing time.
        priority: Processing priority (1 = lowest, 10 = highest).
    """

    page_number: int
    target: RoutingTarget
    confidence: float = 1.0
    reason: str = ""
    estimated_duration_ms: float = 0.0
    priority: int = 5


@dataclass
class RoutingRule:
    """A single routing rule evaluated against :class:`PageFeatures`.

    Attributes:
        name: Short descriptive name for the rule.
        condition: Callable that accepts a :class:`PageFeatures` and
            returns ``True`` when the rule matches.
        target: The :class:`RoutingTarget` to assign when matched.
        priority: Higher-priority rules are evaluated first; the first
            matching rule wins.
        reason: Human-readable explanation attached to the decision.
    """

    name: str
    condition: Callable[[PageFeatures], bool]
    target: RoutingTarget
    priority: int = 0
    reason: str = ""


# ---------------------------------------------------------------------------
# Duration estimation helpers
# ---------------------------------------------------------------------------

_DURATION_ESTIMATES_MS: dict[RoutingTarget, float] = {
    RoutingTarget.GPU_PADDLE: 120.0,
    RoutingTarget.GPU_TESSERACT: 200.0,
    RoutingTarget.CPU_PADDLE: 800.0,
    RoutingTarget.CPU_TESSERACT: 600.0,
    RoutingTarget.CPU_ONNX: 350.0,
    RoutingTarget.SKIP: 0.0,
}


def _estimate_duration(target: RoutingTarget, features: PageFeatures) -> float:
    """Return an estimated processing duration in milliseconds.

    The base estimate is scaled linearly by the page's complexity score
    so that more complex pages get higher duration predictions.
    """
    base = _DURATION_ESTIMATES_MS.get(target, 500.0)
    # Scale between 0.5x (simple) and 1.5x (complex)
    scale = 0.5 + features.complexity_score
    return round(base * scale, 2)


# ---------------------------------------------------------------------------
# Default rules
# ---------------------------------------------------------------------------


def _build_default_rules() -> list[RoutingRule]:
    """Return the built-in routing rules in priority order."""
    return [
        RoutingRule(
            name="skip_tiny_pages",
            condition=lambda f: f.width < 100 and f.height < 100,
            target=RoutingTarget.SKIP,
            priority=100,
            reason="Page dimensions too small to contain useful content",
        ),
        RoutingRule(
            name="handwritten_gpu",
            condition=lambda f: f.is_handwritten,
            target=RoutingTarget.GPU_PADDLE,
            priority=90,
            reason="Handwritten content requires GPU-accelerated recognition",
        ),
        RoutingRule(
            name="tables_gpu",
            condition=lambda f: f.has_tables,
            target=RoutingTarget.GPU_PADDLE,
            priority=80,
            reason="Tabular structures benefit from GPU-accelerated layout analysis",
        ),
        RoutingRule(
            name="high_complexity_gpu",
            condition=lambda f: f.complexity_score > 0.8,
            target=RoutingTarget.GPU_PADDLE,
            priority=70,
            reason="High complexity page routed to GPU for best accuracy",
        ),
        RoutingRule(
            name="low_complexity_onnx",
            condition=lambda f: f.complexity_score < 0.2,
            target=RoutingTarget.CPU_ONNX,
            priority=60,
            reason="Low complexity page routed to fast CPU ONNX path",
        ),
    ]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class PageRouter:
    """Routes pages to processing backends based on feature analysis.

    Parameters:
        rules: Optional list of :class:`RoutingRule` objects.  When
            ``None`` the built-in default rules are used.
        default_target: Fallback :class:`RoutingTarget` when no rule
            matches a page.
    """

    def __init__(
        self,
        rules: Optional[list[RoutingRule]] = None,
        default_target: RoutingTarget = RoutingTarget.GPU_PADDLE,
    ) -> None:
        self._rules: list[RoutingRule] = (
            list(rules) if rules is not None else _build_default_rules()
        )
        self._default_target = default_target
        self._stats: dict[RoutingTarget, int] = {}

    # -- Rule management ----------------------------------------------------

    def add_rule(self, rule: RoutingRule) -> None:
        """Append a routing rule and re-sort by descending priority."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)

    @property
    def rules(self) -> list[RoutingRule]:
        """Return a copy of the current rule list."""
        return list(self._rules)

    @property
    def default_target(self) -> RoutingTarget:
        """Return the configured default target."""
        return self._default_target

    # -- Routing ------------------------------------------------------------

    def route_page(self, features: PageFeatures) -> RoutingDecision:
        """Route a single page to a processing backend.

        Rules are evaluated in descending priority order; the first
        matching rule wins.  If no rule matches the page is assigned to
        *default_target*.
        """
        # Ensure rules are evaluated highest-priority-first
        sorted_rules = sorted(self._rules, key=lambda r: -r.priority)

        for rule in sorted_rules:
            try:
                if rule.condition(features):
                    decision = RoutingDecision(
                        page_number=features.page_number,
                        target=rule.target,
                        confidence=1.0,
                        reason=rule.reason,
                        estimated_duration_ms=_estimate_duration(
                            rule.target, features
                        ),
                        priority=min(max(rule.priority // 10, 1), 10),
                    )
                    self._record(decision.target)
                    logger.debug(
                        "Page %d routed to %s by rule '%s'",
                        features.page_number,
                        rule.target.value,
                        rule.name,
                    )
                    return decision
            except Exception:
                logger.warning(
                    "Rule '%s' raised an exception for page %d; skipping",
                    rule.name,
                    features.page_number,
                    exc_info=True,
                )

        # Fallback to default target
        decision = RoutingDecision(
            page_number=features.page_number,
            target=self._default_target,
            confidence=0.5,
            reason=f"No rule matched; using default target {self._default_target.value}",
            estimated_duration_ms=_estimate_duration(
                self._default_target, features
            ),
            priority=5,
        )
        self._record(decision.target)
        logger.debug(
            "Page %d fell through to default target %s",
            features.page_number,
            self._default_target.value,
        )
        return decision

    def route_batch(self, pages: list[PageFeatures]) -> list[RoutingDecision]:
        """Route a list of pages and return decisions in the same order."""
        return [self.route_page(f) for f in pages]

    # -- Statistics ---------------------------------------------------------

    def get_routing_stats(self) -> dict[str, int]:
        """Return a mapping of target name → count of routed pages."""
        return {t.value: c for t, c in self._stats.items()}

    def reset_stats(self) -> None:
        """Clear all accumulated routing statistics."""
        self._stats.clear()

    def _record(self, target: RoutingTarget) -> None:
        """Increment the routing counter for *target*."""
        self._stats[target] = self._stats.get(target, 0) + 1
