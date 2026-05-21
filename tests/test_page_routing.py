"""Tests for smart page routing module (page_routing.py).

Covers:
- RoutingTarget enum values and count
- RoutingTarget enum round-trip from value
- PageFeatures defaults and custom values
- PageFeatures boundary values (density, complexity 0 and 1)
- RoutingDecision defaults
- RoutingDecision custom values
- RoutingRule creation and condition callable
- Default rules list length and priority ordering
- PageRouter construction with defaults
- PageRouter construction with custom rules
- PageRouter construction with custom default_target
- PageRouter add_rule sorts by priority
- PageRouter route_page handwritten → GPU_PADDLE
- PageRouter route_page has_tables → GPU_PADDLE
- PageRouter route_page high complexity → GPU_PADDLE
- PageRouter route_page low complexity → CPU_ONNX
- PageRouter route_page tiny page → SKIP
- PageRouter route_page default fallback
- PageRouter route_page default fallback uses custom default_target
- PageRouter route_page fallback confidence is lower
- PageRouter route_page priority ordering (highest wins)
- PageRouter route_page custom rule overrides defaults
- PageRouter route_page faulty rule skipped gracefully
- PageRouter route_batch returns correct length
- PageRouter route_batch preserves page order
- PageRouter route_batch mixed routing
- PageRouter get_routing_stats initially empty
- PageRouter get_routing_stats after routing
- PageRouter reset_stats clears counters
- PageRouter stats accumulate across calls
- Duration estimate scales with complexity
- Duration estimate for SKIP is zero
- PageRouter route_page estimated_duration_ms is positive for non-skip
- PageRouter with empty rules list always falls through
- PageRouter route_page reason string present
- PageRouter route_batch empty list
- PageFeatures equality
- RoutingDecision page_number matches features
- PageRouter rules property returns copy
- PageRouter default_target property

Run with: python -m pytest tests/test_page_routing.py -v
"""


# Add project root to path

from page_routing import (
    PageFeatures,
    PageRouter,
    RoutingDecision,
    RoutingRule,
    RoutingTarget,
    _build_default_rules,
    _estimate_duration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_features(**overrides) -> PageFeatures:
    """Create a PageFeatures with sensible defaults, applying overrides."""
    defaults = dict(
        page_number=1,
        width=2480,
        height=3508,
        dpi=300,
        file_size_bytes=500_000,
        estimated_text_density=0.5,
        has_tables=False,
        has_images=False,
        is_handwritten=False,
        language="en",
        complexity_score=0.5,
    )
    defaults.update(overrides)
    return PageFeatures(**defaults)


# ---------------------------------------------------------------------------
# Tests: RoutingTarget enum
# ---------------------------------------------------------------------------


class TestRoutingTarget:
    def test_enum_has_six_members(self):
        assert len(RoutingTarget) == 6

    def test_gpu_paddle_value(self):
        assert RoutingTarget.GPU_PADDLE.value == "gpu_paddle"

    def test_gpu_tesseract_value(self):
        assert RoutingTarget.GPU_TESSERACT.value == "gpu_tesseract"

    def test_cpu_paddle_value(self):
        assert RoutingTarget.CPU_PADDLE.value == "cpu_paddle"

    def test_cpu_tesseract_value(self):
        assert RoutingTarget.CPU_TESSERACT.value == "cpu_tesseract"

    def test_cpu_onnx_value(self):
        assert RoutingTarget.CPU_ONNX.value == "cpu_onnx"

    def test_skip_value(self):
        assert RoutingTarget.SKIP.value == "skip"

    def test_enum_round_trip(self):
        for member in RoutingTarget:
            assert RoutingTarget(member.value) is member


# ---------------------------------------------------------------------------
# Tests: PageFeatures
# ---------------------------------------------------------------------------


class TestPageFeatures:
    def test_default_values(self):
        pf = PageFeatures(page_number=1, width=100, height=200)
        assert pf.page_number == 1
        assert pf.width == 100
        assert pf.height == 200
        assert pf.dpi == 300
        assert pf.file_size_bytes == 0
        assert pf.estimated_text_density == 0.5
        assert pf.has_tables is False
        assert pf.has_images is False
        assert pf.is_handwritten is False
        assert pf.language == "en"
        assert pf.complexity_score == 0.5

    def test_custom_values(self):
        pf = PageFeatures(
            page_number=5,
            width=800,
            height=600,
            dpi=150,
            file_size_bytes=1024,
            estimated_text_density=0.9,
            has_tables=True,
            has_images=True,
            is_handwritten=True,
            language="ja",
            complexity_score=0.95,
        )
        assert pf.page_number == 5
        assert pf.dpi == 150
        assert pf.estimated_text_density == 0.9
        assert pf.has_tables is True
        assert pf.has_images is True
        assert pf.is_handwritten is True
        assert pf.language == "ja"
        assert pf.complexity_score == 0.95

    def test_density_zero(self):
        pf = PageFeatures(page_number=1, width=100, height=100, estimated_text_density=0.0)
        assert pf.estimated_text_density == 0.0

    def test_density_one(self):
        pf = PageFeatures(page_number=1, width=100, height=100, estimated_text_density=1.0)
        assert pf.estimated_text_density == 1.0

    def test_complexity_zero(self):
        pf = PageFeatures(page_number=1, width=100, height=100, complexity_score=0.0)
        assert pf.complexity_score == 0.0

    def test_complexity_one(self):
        pf = PageFeatures(page_number=1, width=100, height=100, complexity_score=1.0)
        assert pf.complexity_score == 1.0

    def test_equality(self):
        a = PageFeatures(page_number=1, width=100, height=200)
        b = PageFeatures(page_number=1, width=100, height=200)
        assert a == b


# ---------------------------------------------------------------------------
# Tests: RoutingDecision
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    def test_defaults(self):
        rd = RoutingDecision(page_number=1, target=RoutingTarget.GPU_PADDLE)
        assert rd.page_number == 1
        assert rd.target is RoutingTarget.GPU_PADDLE
        assert rd.confidence == 1.0
        assert rd.reason == ""
        assert rd.estimated_duration_ms == 0.0
        assert rd.priority == 5

    def test_custom_values(self):
        rd = RoutingDecision(
            page_number=3,
            target=RoutingTarget.CPU_ONNX,
            confidence=0.8,
            reason="custom",
            estimated_duration_ms=42.5,
            priority=9,
        )
        assert rd.page_number == 3
        assert rd.target is RoutingTarget.CPU_ONNX
        assert rd.confidence == 0.8
        assert rd.reason == "custom"
        assert rd.estimated_duration_ms == 42.5
        assert rd.priority == 9


# ---------------------------------------------------------------------------
# Tests: RoutingRule
# ---------------------------------------------------------------------------


class TestRoutingRule:
    def test_creation(self):
        rule = RoutingRule(
            name="test_rule",
            condition=lambda f: f.has_tables,
            target=RoutingTarget.GPU_PADDLE,
            priority=50,
            reason="tables detected",
        )
        assert rule.name == "test_rule"
        assert rule.target is RoutingTarget.GPU_PADDLE
        assert rule.priority == 50
        assert rule.reason == "tables detected"

    def test_condition_callable(self):
        rule = RoutingRule(
            name="always_true",
            condition=lambda f: True,
            target=RoutingTarget.SKIP,
        )
        pf = _simple_features()
        assert rule.condition(pf) is True

    def test_default_priority(self):
        rule = RoutingRule(name="r", condition=lambda f: False, target=RoutingTarget.SKIP)
        assert rule.priority == 0

    def test_default_reason(self):
        rule = RoutingRule(name="r", condition=lambda f: False, target=RoutingTarget.SKIP)
        assert rule.reason == ""


# ---------------------------------------------------------------------------
# Tests: Default rules
# ---------------------------------------------------------------------------


class TestDefaultRules:
    def test_default_rules_count(self):
        rules = _build_default_rules()
        assert len(rules) == 5

    def test_default_rules_priority_ordering(self):
        rules = _build_default_rules()
        priorities = [r.priority for r in rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_default_rules_have_names(self):
        for rule in _build_default_rules():
            assert rule.name
            assert len(rule.name) > 0

    def test_default_rules_have_reasons(self):
        for rule in _build_default_rules():
            assert rule.reason
            assert len(rule.reason) > 0


# ---------------------------------------------------------------------------
# Tests: Duration estimation
# ---------------------------------------------------------------------------


class TestDurationEstimation:
    def test_skip_target_zero_duration(self):
        pf = _simple_features(complexity_score=0.5)
        assert _estimate_duration(RoutingTarget.SKIP, pf) == 0.0

    def test_scales_with_complexity(self):
        pf_low = _simple_features(complexity_score=0.0)
        pf_high = _simple_features(complexity_score=1.0)
        dur_low = _estimate_duration(RoutingTarget.GPU_PADDLE, pf_low)
        dur_high = _estimate_duration(RoutingTarget.GPU_PADDLE, pf_high)
        assert dur_high > dur_low

    def test_positive_for_non_skip(self):
        pf = _simple_features(complexity_score=0.5)
        for target in RoutingTarget:
            if target is not RoutingTarget.SKIP:
                assert _estimate_duration(target, pf) > 0


# ---------------------------------------------------------------------------
# Tests: PageRouter construction
# ---------------------------------------------------------------------------


class TestPageRouterConstruction:
    def test_default_construction(self):
        router = PageRouter()
        assert len(router.rules) == 5
        assert router.default_target is RoutingTarget.GPU_PADDLE

    def test_custom_rules(self):
        custom = [
            RoutingRule(name="r1", condition=lambda f: True, target=RoutingTarget.SKIP, priority=10),
        ]
        router = PageRouter(rules=custom)
        assert len(router.rules) == 1

    def test_custom_default_target(self):
        router = PageRouter(default_target=RoutingTarget.CPU_TESSERACT)
        assert router.default_target is RoutingTarget.CPU_TESSERACT

    def test_empty_rules_list(self):
        router = PageRouter(rules=[])
        assert len(router.rules) == 0

    def test_rules_property_returns_copy(self):
        router = PageRouter()
        rules = router.rules
        rules.clear()
        assert len(router.rules) == 5  # original unchanged


# ---------------------------------------------------------------------------
# Tests: PageRouter.add_rule
# ---------------------------------------------------------------------------


class TestPageRouterAddRule:
    def test_add_rule_increases_count(self):
        router = PageRouter()
        initial = len(router.rules)
        router.add_rule(
            RoutingRule(name="extra", condition=lambda f: False, target=RoutingTarget.SKIP, priority=50)
        )
        assert len(router.rules) == initial + 1

    def test_add_rule_sorts_by_priority(self):
        router = PageRouter(rules=[])
        router.add_rule(RoutingRule(name="low", condition=lambda f: False, target=RoutingTarget.SKIP, priority=10))
        router.add_rule(RoutingRule(name="high", condition=lambda f: False, target=RoutingTarget.SKIP, priority=90))
        router.add_rule(RoutingRule(name="mid", condition=lambda f: False, target=RoutingTarget.SKIP, priority=50))
        priorities = [r.priority for r in router.rules]
        assert priorities == sorted(priorities, reverse=True)


# ---------------------------------------------------------------------------
# Tests: PageRouter.route_page – built-in rules
# ---------------------------------------------------------------------------


class TestRoutePageBuiltinRules:
    def test_handwritten_routes_to_gpu_paddle(self):
        router = PageRouter()
        pf = _simple_features(is_handwritten=True)
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.GPU_PADDLE
        assert "andwrit" in decision.reason.lower()

    def test_tables_routes_to_gpu_paddle(self):
        router = PageRouter()
        pf = _simple_features(has_tables=True)
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.GPU_PADDLE
        assert "tabul" in decision.reason.lower()

    def test_high_complexity_routes_to_gpu_paddle(self):
        router = PageRouter()
        pf = _simple_features(complexity_score=0.9)
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.GPU_PADDLE
        assert "complex" in decision.reason.lower()

    def test_low_complexity_routes_to_cpu_onnx(self):
        router = PageRouter()
        pf = _simple_features(complexity_score=0.1)
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.CPU_ONNX
        assert "low" in decision.reason.lower() or "onnx" in decision.reason.lower()

    def test_tiny_page_routes_to_skip(self):
        router = PageRouter()
        pf = _simple_features(width=50, height=50)
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.SKIP
        assert "small" in decision.reason.lower() or "dimension" in decision.reason.lower()

    def test_tiny_page_only_when_both_dimensions_small(self):
        router = PageRouter()
        # Width small but height large → should NOT skip
        pf = _simple_features(width=50, height=3508)
        decision = router.route_page(pf)
        assert decision.target is not RoutingTarget.SKIP


# ---------------------------------------------------------------------------
# Tests: PageRouter.route_page – fallback
# ---------------------------------------------------------------------------


class TestRoutePageFallback:
    def test_default_fallback(self):
        router = PageRouter()
        pf = _simple_features(complexity_score=0.5)  # mid-range, no tables, no handwriting
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.GPU_PADDLE
        assert "default" in decision.reason.lower()

    def test_custom_default_fallback(self):
        router = PageRouter(default_target=RoutingTarget.CPU_TESSERACT)
        pf = _simple_features(complexity_score=0.5)
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.CPU_TESSERACT

    def test_fallback_confidence_lower(self):
        router = PageRouter()
        pf = _simple_features(complexity_score=0.5)
        decision = router.route_page(pf)
        assert decision.confidence < 1.0

    def test_empty_rules_always_falls_through(self):
        router = PageRouter(rules=[], default_target=RoutingTarget.CPU_ONNX)
        pf = _simple_features()
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.CPU_ONNX


# ---------------------------------------------------------------------------
# Tests: PageRouter.route_page – priority ordering
# ---------------------------------------------------------------------------


class TestRoutePagePriority:
    def test_highest_priority_wins(self):
        """When multiple rules match, the highest-priority one wins."""
        router = PageRouter()
        # Handwritten + tables both match; handwritten has higher priority
        pf = _simple_features(is_handwritten=True, has_tables=True, complexity_score=0.9)
        decision = router.route_page(pf)
        # Tiny page is highest priority (100) but won't match; handwritten (90) > tables (80) > high_complexity (70)
        assert "andwrit" in decision.reason.lower()

    def test_tiny_page_overrides_handwritten(self):
        """Skip rule has highest priority and overrides all others."""
        router = PageRouter()
        pf = _simple_features(width=50, height=50, is_handwritten=True)
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.SKIP

    def test_custom_rule_can_override(self):
        """A custom rule with very high priority overrides built-in rules."""
        custom_rule = RoutingRule(
            name="force_cpu",
            condition=lambda f: True,
            target=RoutingTarget.CPU_TESSERACT,
            priority=999,
            reason="Forced CPU processing",
        )
        router = PageRouter()
        router.add_rule(custom_rule)
        pf = _simple_features(is_handwritten=True)
        decision = router.route_page(pf)
        assert decision.target is RoutingTarget.CPU_TESSERACT


# ---------------------------------------------------------------------------
# Tests: PageRouter.route_page – error handling
# ---------------------------------------------------------------------------


class TestRoutePageErrorHandling:
    def test_faulty_rule_skipped(self):
        """A rule whose condition raises is silently skipped."""
        def bad_condition(f):
            raise ValueError("boom")

        faulty = RoutingRule(
            name="faulty",
            condition=bad_condition,
            target=RoutingTarget.SKIP,
            priority=999,
        )
        router = PageRouter(rules=[faulty])
        pf = _simple_features()
        decision = router.route_page(pf)
        # Falls through to default since faulty rule is skipped
        assert decision.target is RoutingTarget.GPU_PADDLE


# ---------------------------------------------------------------------------
# Tests: PageRouter.route_page – decision fields
# ---------------------------------------------------------------------------


class TestRoutePageDecisionFields:
    def test_page_number_matches(self):
        router = PageRouter()
        pf = _simple_features(page_number=7)
        decision = router.route_page(pf)
        assert decision.page_number == 7

    def test_reason_is_nonempty_for_matched_rule(self):
        router = PageRouter()
        pf = _simple_features(is_handwritten=True)
        decision = router.route_page(pf)
        assert len(decision.reason) > 0

    def test_reason_is_nonempty_for_fallback(self):
        router = PageRouter()
        pf = _simple_features(complexity_score=0.5)
        decision = router.route_page(pf)
        assert len(decision.reason) > 0

    def test_estimated_duration_positive_for_gpu(self):
        router = PageRouter()
        pf = _simple_features(is_handwritten=True, complexity_score=0.5)
        decision = router.route_page(pf)
        assert decision.estimated_duration_ms > 0

    def test_priority_clamped_1_to_10(self):
        router = PageRouter()
        for pf in [
            _simple_features(is_handwritten=True),
            _simple_features(complexity_score=0.1),
            _simple_features(width=50, height=50),
            _simple_features(complexity_score=0.5),
        ]:
            decision = router.route_page(pf)
            assert 1 <= decision.priority <= 10


# ---------------------------------------------------------------------------
# Tests: PageRouter.route_batch
# ---------------------------------------------------------------------------


class TestRouteBatch:
    def test_returns_correct_length(self):
        router = PageRouter()
        pages = [_simple_features(page_number=i) for i in range(1, 6)]
        decisions = router.route_batch(pages)
        assert len(decisions) == 5

    def test_preserves_page_order(self):
        router = PageRouter()
        pages = [_simple_features(page_number=i) for i in range(1, 4)]
        decisions = router.route_batch(pages)
        assert [d.page_number for d in decisions] == [1, 2, 3]

    def test_empty_batch(self):
        router = PageRouter()
        assert router.route_batch([]) == []

    def test_mixed_routing(self):
        router = PageRouter()
        pages = [
            _simple_features(page_number=1, is_handwritten=True),
            _simple_features(page_number=2, complexity_score=0.1),
            _simple_features(page_number=3, width=50, height=50),
        ]
        decisions = router.route_batch(pages)
        assert decisions[0].target is RoutingTarget.GPU_PADDLE
        assert decisions[1].target is RoutingTarget.CPU_ONNX
        assert decisions[2].target is RoutingTarget.SKIP


# ---------------------------------------------------------------------------
# Tests: PageRouter statistics
# ---------------------------------------------------------------------------


class TestRoutingStats:
    def test_initially_empty(self):
        router = PageRouter()
        assert router.get_routing_stats() == {}

    def test_after_routing(self):
        router = PageRouter()
        router.route_page(_simple_features(is_handwritten=True))
        stats = router.get_routing_stats()
        assert stats["gpu_paddle"] == 1

    def test_accumulates_across_calls(self):
        router = PageRouter()
        router.route_page(_simple_features(is_handwritten=True))
        router.route_page(_simple_features(is_handwritten=True))
        router.route_page(_simple_features(complexity_score=0.1))
        stats = router.get_routing_stats()
        assert stats["gpu_paddle"] == 2
        assert stats["cpu_onnx"] == 1

    def test_reset_clears_stats(self):
        router = PageRouter()
        router.route_page(_simple_features(is_handwritten=True))
        router.reset_stats()
        assert router.get_routing_stats() == {}

    def test_batch_updates_stats(self):
        router = PageRouter()
        pages = [
            _simple_features(page_number=1, is_handwritten=True),
            _simple_features(page_number=2, complexity_score=0.1),
            _simple_features(page_number=3, complexity_score=0.5),
        ]
        router.route_batch(pages)
        stats = router.get_routing_stats()
        total = sum(stats.values())
        assert total == 3
