"""Tests for exception_router.py -- exception routing rules engine."""

import json
import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure exception-routing env vars are at defaults for each test."""
    monkeypatch.delenv("ENABLE_EXCEPTION_ROUTING", raising=False)
    monkeypatch.delenv("REVIEW_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("REVIEW_IMAGE_ONLY_THRESHOLD", raising=False)
    monkeypatch.delenv("REVIEW_CLASSIFICATION_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("EXCEPTION_ROUTING_RULES_PATH", raising=False)


def _reload_module(monkeypatch, env_overrides=None):
    """Reload exception_router with custom env vars to pick up changes."""
    import importlib

    for key, val in (env_overrides or {}).items():
        monkeypatch.setenv(key, val)

    import exception_router

    importlib.reload(exception_router)
    return exception_router


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------


class TestRoutingRule:
    def test_creation_defaults(self):
        from exception_router import RoutingRule

        rule = RoutingRule(name="test", reason="low_confidence")
        assert rule.name == "test"
        assert rule.reason == "low_confidence"
        assert rule.enabled is True
        assert rule.description == ""

    def test_creation_with_all_fields(self):
        from exception_router import RoutingRule

        rule = RoutingRule(
            name="custom",
            reason="manual_flag",
            enabled=False,
            description="A custom rule",
        )
        assert rule.name == "custom"
        assert rule.enabled is False
        assert rule.description == "A custom rule"


class TestRoutingDecision:
    def test_defaults(self):
        from exception_router import RoutingDecision

        decision = RoutingDecision()
        assert decision.should_route is False
        assert decision.triggered_rules == []
        assert decision.reasons == []
        assert decision.confidence == 0.0
        assert decision.metadata == {}


# ---------------------------------------------------------------------------
# ExceptionRouter.evaluate()
# ---------------------------------------------------------------------------


class TestEvaluateLowConfidence:
    def test_low_confidence_routes(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={"overall_confidence": 0.3, "classification": "acceptable"}
        )
        assert decision.should_route is True
        assert "low_confidence" in decision.triggered_rules
        assert "low_confidence" in decision.reasons
        assert decision.confidence == 0.3

    def test_high_confidence_does_not_route(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={"overall_confidence": 0.9, "classification": "high_quality"}
        )
        # Only low_confidence rule checked here, not degraded
        assert "low_confidence" not in decision.triggered_rules

    def test_confidence_at_threshold_does_not_route(self):
        """Exactly at threshold should NOT route (strict less-than)."""
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={"overall_confidence": 0.5, "classification": "acceptable"}
        )
        assert "low_confidence" not in decision.triggered_rules


class TestEvaluateDegradedQuality:
    def test_degraded_quality_routes(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={"overall_confidence": 0.8, "classification": "degraded"}
        )
        assert decision.should_route is True
        assert "degraded_quality" in decision.triggered_rules
        assert "degraded_quality" in decision.reasons

    def test_review_required_quality_routes(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "overall_confidence": 0.8,
                "classification": "review_required",
            }
        )
        assert decision.should_route is True
        assert "degraded_quality" in decision.triggered_rules

    def test_acceptable_quality_does_not_route(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "overall_confidence": 0.8,
                "classification": "acceptable",
            }
        )
        assert "degraded_quality" not in decision.triggered_rules

    def test_high_quality_does_not_route(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "overall_confidence": 0.9,
                "classification": "high_quality",
            }
        )
        assert "degraded_quality" not in decision.triggered_rules


class TestEvaluateHandwriting:
    def test_handwriting_detected_routes(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            handwriting_data={
                "document_summary": {
                    "is_primarily_handwritten": True,
                    "total_handwritten_pages": 3,
                }
            }
        )
        assert decision.should_route is True
        assert "handwriting_detected" in decision.triggered_rules
        assert "handwriting_detected" in decision.reasons

    def test_handwriting_detected_flat_key(self):
        """Support flat handwriting_detected boolean (legacy format)."""
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            handwriting_data={"handwriting_detected": True}
        )
        assert decision.should_route is True
        assert "handwriting_detected" in decision.triggered_rules

    def test_no_handwriting_does_not_route(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            handwriting_data={
                "document_summary": {
                    "is_primarily_handwritten": False,
                    "total_handwritten_pages": 0,
                }
            }
        )
        assert "handwriting_detected" not in decision.triggered_rules


class TestEvaluateImageOnlyPages:
    def test_excessive_image_only_routes(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "overall_confidence": 0.8,
                "classification": "acceptable",
                "pages_image_only": 5,
            }
        )
        assert decision.should_route is True
        assert "image_only_pages" in decision.triggered_rules
        assert decision.metadata["image_only_pages"] == 5

    def test_few_image_only_does_not_route(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "overall_confidence": 0.8,
                "classification": "acceptable",
                "pages_image_only": 2,
            }
        )
        assert "image_only_pages" not in decision.triggered_rules

    def test_at_threshold_does_not_route(self):
        """Exactly at threshold should NOT route (strict greater-than)."""
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "overall_confidence": 0.8,
                "classification": "acceptable",
                "pages_image_only": 3,
            }
        )
        assert "image_only_pages" not in decision.triggered_rules


class TestEvaluateClassificationUncertain:
    def test_uncertain_classification_routes(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            classification_data={"confidence": 0.3, "doc_type": "unknown"}
        )
        assert decision.should_route is True
        assert "classification_uncertain" in decision.triggered_rules
        assert "classification_uncertain" in decision.reasons

    def test_confident_classification_does_not_route(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            classification_data={"confidence": 0.8, "doc_type": "invoice"}
        )
        assert "classification_uncertain" not in decision.triggered_rules


class TestEvaluateCustomPatterns:
    def test_custom_pattern_match_routes(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                [
                    {
                        "name": "privileged_keyword",
                        "pattern": r"attorney.client\s+privilege",
                        "reason": "manual_flag",
                        "description": "Privileged content detected",
                    }
                ],
                f,
            )
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            decision = router.evaluate(
                extracted_text="This document is subject to attorney-client privilege protection."
            )
            assert decision.should_route is True
            assert "privileged_keyword" in decision.triggered_rules
            assert "manual_flag" in decision.reasons
            assert decision.metadata.get("pattern_privileged_keyword") is True
        finally:
            os.unlink(rules_path)

    def test_custom_pattern_no_match(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                [{"name": "ssn_pattern", "pattern": r"\d{3}-\d{2}-\d{4}"}],
                f,
            )
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            decision = router.evaluate(extracted_text="No sensitive data here.")
            assert "ssn_pattern" not in decision.triggered_rules
        finally:
            os.unlink(rules_path)

    def test_invalid_regex_skipped(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                [
                    {
                        "name": "bad_regex",
                        "pattern": r"[invalid",
                        "reason": "manual_flag",
                    }
                ],
                f,
            )
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            # Invalid regex should be skipped during load
            assert len(router.custom_patterns) == 0
        finally:
            os.unlink(rules_path)


class TestEvaluateNoData:
    def test_no_data_does_not_route(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate()
        assert decision.should_route is False
        assert decision.triggered_rules == []
        assert decision.reasons == []

    def test_none_inputs(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data=None,
            classification_data=None,
            handwriting_data=None,
            extracted_text="",
        )
        assert decision.should_route is False


class TestEvaluateMultipleRules:
    def test_multiple_rules_triggered(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "overall_confidence": 0.2,
                "classification": "review_required",
                "pages_image_only": 10,
            },
            classification_data={"confidence": 0.1},
            handwriting_data={"document_summary": {"is_primarily_handwritten": True}},
        )
        assert decision.should_route is True
        assert "low_confidence" in decision.triggered_rules
        assert "degraded_quality" in decision.triggered_rules
        assert "handwriting_detected" in decision.triggered_rules
        assert "image_only_pages" in decision.triggered_rules
        assert "classification_uncertain" in decision.triggered_rules
        assert len(decision.triggered_rules) == 5

    def test_reasons_deduplicated_per_rule(self):
        """Each rule contributes exactly one reason entry."""
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "overall_confidence": 0.2,
                "classification": "review_required",
            }
        )
        assert decision.reasons.count("low_confidence") == 1
        assert decision.reasons.count("degraded_quality") == 1


class TestEvaluateWithFullReport:
    def test_full_validation_report_format(self):
        """Support the full validation JSON report with nested quality dict."""
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        full_report = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "quality": {
                "overall_confidence": 0.3,
                "classification": "degraded",
                "pages_image_only": 5,
            },
        }
        decision = router.evaluate(validation_data=full_report)
        assert decision.should_route is True
        assert "low_confidence" in decision.triggered_rules
        assert "degraded_quality" in decision.triggered_rules
        assert "image_only_pages" in decision.triggered_rules


# ---------------------------------------------------------------------------
# Env var threshold configuration
# ---------------------------------------------------------------------------


class TestEnvVarConfiguration:
    def test_custom_confidence_threshold(self, monkeypatch):
        mod = _reload_module(monkeypatch, {"REVIEW_CONFIDENCE_THRESHOLD": "0.8"})
        router = mod.ExceptionRouter()
        # 0.7 is below 0.8 threshold
        decision = router.evaluate(
            validation_data={"overall_confidence": 0.7, "classification": "acceptable"}
        )
        assert "low_confidence" in decision.triggered_rules

    def test_custom_image_only_threshold(self, monkeypatch):
        mod = _reload_module(monkeypatch, {"REVIEW_IMAGE_ONLY_THRESHOLD": "1"})
        router = mod.ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "overall_confidence": 0.8,
                "classification": "acceptable",
                "pages_image_only": 2,
            }
        )
        assert "image_only_pages" in decision.triggered_rules

    def test_enable_exception_routing_true(self, monkeypatch):
        mod = _reload_module(monkeypatch, {"ENABLE_EXCEPTION_ROUTING": "true"})
        assert mod.ENABLE_EXCEPTION_ROUTING is True

    def test_enable_exception_routing_false(self, monkeypatch):
        mod = _reload_module(monkeypatch, {"ENABLE_EXCEPTION_ROUTING": "false"})
        assert mod.ENABLE_EXCEPTION_ROUTING is False

    def test_enable_exception_routing_1(self, monkeypatch):
        mod = _reload_module(monkeypatch, {"ENABLE_EXCEPTION_ROUTING": "1"})
        assert mod.ENABLE_EXCEPTION_ROUTING is True

    def test_enable_exception_routing_yes(self, monkeypatch):
        mod = _reload_module(monkeypatch, {"ENABLE_EXCEPTION_ROUTING": "yes"})
        assert mod.ENABLE_EXCEPTION_ROUTING is True

    def test_enable_exception_routing_default(self, monkeypatch):
        mod = _reload_module(monkeypatch)
        assert mod.ENABLE_EXCEPTION_ROUTING is False


# ---------------------------------------------------------------------------
# get_rules()
# ---------------------------------------------------------------------------


class TestGetRules:
    def test_returns_default_rules(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        rules = router.get_rules()
        assert isinstance(rules, list)
        assert len(rules) >= 5
        names = [r["name"] for r in rules]
        assert "low_confidence" in names
        assert "degraded_quality" in names
        assert "handwriting_detected" in names
        assert "image_only_pages" in names
        assert "classification_uncertain" in names

    def test_rule_dict_structure(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        rules = router.get_rules()
        for rule in rules:
            assert "name" in rule
            assert "reason" in rule
            assert "enabled" in rule
            assert "description" in rule

    def test_includes_custom_rules(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                [
                    {
                        "name": "custom_one",
                        "pattern": "test",
                        "reason": "manual_flag",
                        "description": "Custom rule",
                    }
                ],
                f,
            )
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            rules = router.get_rules()
            names = [r["name"] for r in rules]
            assert "custom_one" in names
        finally:
            os.unlink(rules_path)


# ---------------------------------------------------------------------------
# Custom rules JSON loading
# ---------------------------------------------------------------------------


class TestCustomRulesLoading:
    def test_load_valid_rules(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                [
                    {"name": "r1", "pattern": r"\bconfidential\b", "reason": "manual_flag"},
                    {"name": "r2", "pattern": r"\bsecret\b"},
                ],
                f,
            )
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            assert len(router.custom_patterns) == 2
            assert router.custom_patterns[0]["name"] == "r1"
            assert router.custom_patterns[1]["reason"] == "manual_flag"  # default
        finally:
            os.unlink(rules_path)

    def test_load_nonexistent_file(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter(custom_rules_path="/nonexistent/path.json")
        assert len(router.custom_patterns) == 0

    def test_load_empty_path(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter(custom_rules_path="")
        assert len(router.custom_patterns) == 0

    def test_load_non_array_json(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"not": "an array"}, f)
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            assert len(router.custom_patterns) == 0
        finally:
            os.unlink(rules_path)

    def test_load_missing_name_skipped(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                [{"pattern": "test"}],  # missing name
                f,
            )
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            assert len(router.custom_patterns) == 0
        finally:
            os.unlink(rules_path)

    def test_load_missing_pattern_skipped(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                [{"name": "test"}],  # missing pattern
                f,
            )
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            assert len(router.custom_patterns) == 0
        finally:
            os.unlink(rules_path)

    def test_load_invalid_json_file(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json{{{")
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            assert len(router.custom_patterns) == 0
        finally:
            os.unlink(rules_path)

    def test_load_non_dict_entries_skipped(self):
        from exception_router import ExceptionRouter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                ["string_entry", 42, None, {"name": "valid", "pattern": "ok"}],
                f,
            )
            f.flush()
            rules_path = f.name

        try:
            router = ExceptionRouter(custom_rules_path=rules_path)
            assert len(router.custom_patterns) == 1
            assert router.custom_patterns[0]["name"] == "valid"
        finally:
            os.unlink(rules_path)


# ---------------------------------------------------------------------------
# Rule enable/disable
# ---------------------------------------------------------------------------


class TestRuleEnableDisable:
    def test_disabled_rule_not_evaluated(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        # Disable the low_confidence rule
        for rule in router.rules:
            if rule.name == "low_confidence":
                rule.enabled = False

        decision = router.evaluate(
            validation_data={"overall_confidence": 0.1, "classification": "acceptable"}
        )
        assert "low_confidence" not in decision.triggered_rules

    def test_enabled_rule_evaluated(self):
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={"overall_confidence": 0.1, "classification": "acceptable"}
        )
        assert "low_confidence" in decision.triggered_rules
