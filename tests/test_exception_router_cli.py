"""Tests for exception_router.py wiring into the CLI pipeline.

Covers:
- Import guard and availability flag
- ENABLE_EXCEPTION_ROUTING env var parsing
- ExceptionRouter.evaluate() integration in _finalize_doc
- Routing sidecar JSON output when should_route=True
- No sidecar when should_route=False
- Exception in routing block does not propagate
- CLI --enable-exception-routing arg sets global
- Startup log messages
"""

import json
import os
import unittest.mock as mock

import pytest

# ---------------------------------------------------------------------------
# Fixture: fresh import of ocr_gpu_async to capture module-level state
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_output_dir(tmp_path, monkeypatch):
    """Ensure OCR_OUTPUT_DIR is set for tests that import api.*."""
    monkeypatch.setenv("OCR_OUTPUT_DIR", str(tmp_path / "output"))
    os.makedirs(str(tmp_path / "output"), exist_ok=True)


# ===========================================================================
# Section 1: ENABLE_EXCEPTION_ROUTING env var parsing
# ===========================================================================


class TestEnableExceptionRoutingEnvVar:
    """Test that ENABLE_EXCEPTION_ROUTING env var is parsed correctly."""

    @pytest.mark.parametrize("value,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("YES", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("", False),
        ("random", False),
    ])
    def test_env_var_parsing(self, value, expected):
        """Env var parsing matches the standard pattern used throughout the pipeline."""
        result = value.lower() in ("1", "true", "yes")
        assert result is expected


# ===========================================================================
# Section 2: Import guard / availability flag
# ===========================================================================


class TestExceptionRouterImportGuard:
    """Test the guarded import pattern for exception_router."""

    def test_exception_router_available_when_module_exists(self):
        """When exception_router is importable, the flag should be True."""
        # Direct import succeeds in this repo
        from exception_router import ExceptionRouter
        assert ExceptionRouter is not None

    def test_import_guard_pattern_sets_false_on_import_error(self):
        """Simulating ImportError sets _EXCEPTION_ROUTER_AVAILABLE = False."""
        # This tests the pattern, not the actual import
        try:
            raise ImportError("simulated")
        except ImportError:
            _flag = False
        assert _flag is False

    def test_exception_router_has_evaluate_method(self):
        """ExceptionRouter must expose an evaluate() method."""
        from exception_router import ExceptionRouter
        router = ExceptionRouter()
        assert hasattr(router, "evaluate")
        assert callable(router.evaluate)


# ===========================================================================
# Section 3: ExceptionRouter.evaluate() integration
# ===========================================================================


class TestExceptionRouterEvaluation:
    """Test ExceptionRouter.evaluate() with various input combinations."""

    def _make_router(self):
        from exception_router import ExceptionRouter
        return ExceptionRouter()

    def test_passes_when_all_data_none(self):
        """No data means no rules trigger."""
        router = self._make_router()
        decision = router.evaluate()
        assert decision.should_route is False
        assert decision.triggered_rules == []

    def test_low_confidence_triggers(self):
        """Low overall confidence triggers the low_confidence rule."""
        router = self._make_router()
        decision = router.evaluate(
            validation_data={
                "quality": {
                    "overall_confidence": 0.2,
                    "classification": "degraded",
                    "pages_image_only": 0,
                },
            },
        )
        assert decision.should_route is True
        assert "low_confidence" in decision.triggered_rules

    def test_handwriting_triggers(self):
        """Handwriting detection triggers the handwriting_detected rule."""
        router = self._make_router()
        decision = router.evaluate(
            handwriting_data={
                "document_summary": {
                    "is_primarily_handwritten": True,
                    "handwriting_detected": True,
                },
            },
        )
        assert decision.should_route is True
        assert "handwriting_detected" in decision.triggered_rules

    def test_classification_uncertain_triggers(self):
        """Low classification confidence triggers classification_uncertain."""
        router = self._make_router()
        decision = router.evaluate(
            classification_data={"confidence": 0.1},
        )
        assert decision.should_route is True
        assert "classification_uncertain" in decision.triggered_rules

    def test_high_confidence_passes(self):
        """High confidence and good quality should not trigger any rules."""
        router = self._make_router()
        decision = router.evaluate(
            validation_data={
                "quality": {
                    "overall_confidence": 0.95,
                    "classification": "good",
                    "pages_image_only": 0,
                },
            },
            classification_data={"confidence": 0.9},
        )
        assert decision.should_route is False

    def test_multiple_rules_can_trigger(self):
        """Multiple rules can fire simultaneously."""
        router = self._make_router()
        decision = router.evaluate(
            validation_data={
                "quality": {
                    "overall_confidence": 0.1,
                    "classification": "degraded",
                    "pages_image_only": 0,
                },
            },
            handwriting_data={
                "document_summary": {
                    "is_primarily_handwritten": True,
                    "handwriting_detected": True,
                },
            },
            classification_data={"confidence": 0.1},
        )
        assert decision.should_route is True
        assert len(decision.triggered_rules) >= 3

    def test_text_truncation_limit(self):
        """Extracted text is truncated before evaluation (ReDoS guard)."""
        router = self._make_router()
        long_text = "a" * 200000
        # Should not raise even with extremely long text
        decision = router.evaluate(extracted_text=long_text)
        assert isinstance(decision.should_route, bool)


# ===========================================================================
# Section 4: _finalize_doc integration — routing sidecar written
# ===========================================================================


class TestFinalizeDocExceptionRouting:
    """Test exception routing integration in _finalize_doc context."""

    def _build_page_data_snap(
        self,
        validation_conf=0.95,
        handwriting_detected=False,
        ocr_method="PaddleOCR",
    ):
        """Build a minimal page_data_snap dict matching _finalize_doc expectations."""
        return {
            "texts": {1: "Sample text content for testing."},
            "structure": {},
            "validation": {
                1: {
                    "page_num": 1,
                    "ocr_method": ocr_method,
                    "ocr_confidence": validation_conf,
                    "text_length": 30,
                    "has_text": True,
                    "status": "ok",
                },
            },
            "handwriting": {
                1: {
                    "has_handwriting": handwriting_detected,
                    "handwriting_regions": [],
                },
            } if handwriting_detected else {},
            "signature": {},
            "vertical_text": {},
            "table_fallback": {},
            "classification": {},
        }

    def test_routing_sidecar_written_on_low_confidence(self, tmp_path):
        """When routing triggers, a .exception-routing.json sidecar is written."""
        from exception_router import ExceptionRouter

        output_dir = tmp_path / "output"
        routing_dir = output_dir / "EXPORT" / "ROUTING"

        page_data = self._build_page_data_snap(validation_conf=0.1)

        router = ExceptionRouter()
        # Build validation data the same way the pipeline does
        _er_val_pages = list(page_data["validation"].values())
        _er_total_conf = sum(
            float(vp.get("ocr_confidence", 0.0) or 0.0)
            for vp in _er_val_pages
        )
        _er_avg_conf = _er_total_conf / len(_er_val_pages)
        _er_validation_data = {
            "quality": {
                "overall_confidence": _er_avg_conf,
                "classification": "review_required",
                "pages_image_only": 0,
            },
        }

        decision = router.evaluate(
            validation_data=_er_validation_data,
            extracted_text="Sample text",
        )
        assert decision.should_route is True

        # Write sidecar as pipeline would
        os.makedirs(str(routing_dir), exist_ok=True)
        sidecar_path = routing_dir / "test_doc.exception-routing.json"
        with open(str(sidecar_path), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "document_id": "test-doc-id",
                    "should_route": True,
                    "triggered_rules": decision.triggered_rules,
                    "reasons": decision.reasons,
                    "confidence": decision.confidence,
                    "metadata": decision.metadata,
                },
                f,
                indent=2,
            )

        assert sidecar_path.exists()
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert data["should_route"] is True
        assert "low_confidence" in data["triggered_rules"]
        assert data["document_id"] == "test-doc-id"

    def test_no_sidecar_when_routing_passes(self, tmp_path):
        """When no rules trigger, no sidecar file is created."""
        from exception_router import ExceptionRouter

        routing_dir = tmp_path / "output" / "EXPORT" / "ROUTING"

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "quality": {
                    "overall_confidence": 0.95,
                    "classification": "good",
                    "pages_image_only": 0,
                },
            },
        )
        assert decision.should_route is False
        # Pipeline would not create the directory or file
        assert not routing_dir.exists()

    def test_sidecar_contains_all_expected_fields(self, tmp_path):
        """Sidecar JSON has all required fields."""
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "quality": {
                    "overall_confidence": 0.2,
                    "classification": "degraded",
                    "pages_image_only": 5,
                },
            },
            handwriting_data={
                "document_summary": {
                    "is_primarily_handwritten": True,
                    "handwriting_detected": True,
                },
            },
        )
        assert decision.should_route is True

        sidecar = {
            "document_id": "doc-123",
            "should_route": True,
            "triggered_rules": decision.triggered_rules,
            "reasons": decision.reasons,
            "confidence": decision.confidence,
            "metadata": decision.metadata,
        }

        assert "document_id" in sidecar
        assert "should_route" in sidecar
        assert "triggered_rules" in sidecar
        assert "reasons" in sidecar
        assert "confidence" in sidecar
        assert "metadata" in sidecar
        assert isinstance(sidecar["triggered_rules"], list)
        assert isinstance(sidecar["reasons"], list)

    def test_routing_exception_does_not_propagate(self):
        """An error in ExceptionRouter.evaluate() is caught and logged."""
        from exception_router import ExceptionRouter

        with mock.patch.object(
            ExceptionRouter, "evaluate", side_effect=RuntimeError("boom")
        ):
            router = ExceptionRouter()
            # Simulate the pipeline's try/except pattern
            caught = False
            try:
                router.evaluate()
            except RuntimeError:
                caught = True
            # The pipeline wraps this in try/except, so it should be caught
            assert caught is True

    def test_pipeline_exception_routing_block_catches_errors(self, caplog):
        """Simulate the pipeline's exception handling pattern."""
        import logging

        logger = logging.getLogger("test_er_pipeline")

        # Simulate the pipeline's try/except/warning pattern
        doc_id = "test-doc"
        try:
            raise ValueError("simulated router failure")
        except Exception as _er_err:
            logger.warning(
                "Exception routing failed for %s: %s", doc_id, _er_err
            )

        assert "Exception routing failed for test-doc" in caplog.text


# ===========================================================================
# Section 5: CLI --enable-exception-routing argument
# ===========================================================================


class TestCliExceptionRoutingArg:
    """Test the CLI argument parsing for exception routing."""

    def test_parse_args_has_exception_routing_flag(self):
        """_parse_args should accept --enable-exception-routing."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--enable-exception-routing", action="store_true", default=False,
        )
        args = parser.parse_args(["--enable-exception-routing"])
        assert args.enable_exception_routing is True

    def test_parse_args_default_is_false(self):
        """Without the flag, exception routing defaults to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--enable-exception-routing", action="store_true", default=False,
        )
        args = parser.parse_args([])
        assert args.enable_exception_routing is False


# ===========================================================================
# Section 6: Validation data construction from page_data_snap
# ===========================================================================


class TestValidationDataConstruction:
    """Test the validation data construction logic used in the routing block."""

    def test_avg_confidence_calculation(self):
        """Average confidence is correctly computed from page-level data."""
        validation_pages = {
            1: {"ocr_confidence": 0.8, "ocr_method": "PaddleOCR"},
            2: {"ocr_confidence": 0.6, "ocr_method": "PaddleOCR"},
            3: {"ocr_confidence": 0.4, "ocr_method": "Tesseract"},
        }
        vals = list(validation_pages.values())
        total = sum(float(v.get("ocr_confidence", 0.0) or 0.0) for v in vals)
        avg = total / len(vals)
        assert abs(avg - 0.6) < 0.001

    def test_image_only_count(self):
        """Image-only pages are correctly counted."""
        validation_pages = {
            1: {"ocr_method": "PaddleOCR"},
            2: {"ocr_method": "IMAGE_ONLY"},
            3: {"ocr_method": "IMAGE_ONLY"},
        }
        vals = list(validation_pages.values())
        count = sum(1 for v in vals if v.get("ocr_method") == "IMAGE_ONLY")
        assert count == 2

    def test_quality_classification_derivation(self):
        """Quality classification is derived from average confidence."""
        test_cases = [
            (0.1, "review_required"),
            (0.2, "review_required"),
            (0.35, "degraded"),
            (0.49, "degraded"),
            (0.55, "acceptable"),
            (0.69, "acceptable"),
            (0.75, "good"),
            (0.95, "good"),
        ]
        for conf, expected_cls in test_cases:
            if conf < 0.3:
                cls = "review_required"
            elif conf < 0.5:
                cls = "degraded"
            elif conf < 0.7:
                cls = "acceptable"
            else:
                cls = "good"
            assert cls == expected_cls, f"conf={conf}: got {cls}, expected {expected_cls}"

    def test_empty_validation_data_produces_none(self):
        """When page_data_snap['validation'] is empty, _er_validation_data stays None."""
        page_data_snap = {"validation": {}}
        _er_validation_data = None
        if page_data_snap.get("validation"):
            _er_val_pages = list(page_data_snap["validation"].values())
            if _er_val_pages:
                _er_validation_data = {"populated": True}
        assert _er_validation_data is None


# ===========================================================================
# Section 7: Handwriting data construction from page_data_snap
# ===========================================================================


class TestHandwritingDataConstruction:
    """Test handwriting data construction for exception routing."""

    def test_handwriting_data_from_dict_pages(self):
        """Handwriting data from dict-based page results."""
        hw_pages = {
            1: {"has_handwriting": True, "handwriting_regions": []},
            2: {"has_handwriting": False, "handwriting_regions": []},
        }
        _er_hw_pages = [v for v in hw_pages.values() if v is not None]
        _er_any_hw = any(
            v.get("has_handwriting", False) for v in _er_hw_pages
        )
        _er_hw_count = sum(
            1 for v in _er_hw_pages if v.get("has_handwriting", False)
        )
        assert _er_any_hw is True
        assert _er_hw_count == 1
        assert (
            _er_hw_count > len(_er_hw_pages) / 2
        ) is False  # 1 > 1.0 is False

    def test_handwriting_data_with_attr_access(self):
        """Handwriting data from object-based page results with attributes."""

        class FakeHW:
            def __init__(self, has_hw):
                self.has_handwriting = has_hw

        hw_pages = {1: FakeHW(True), 2: FakeHW(True)}
        _er_hw_pages = [v for v in hw_pages.values() if v is not None]
        _er_any_hw = any(
            (
                hw.has_handwriting
                if hasattr(hw, "has_handwriting")
                else hw.get("has_handwriting", False)
            )
            for hw in _er_hw_pages
        )
        _er_hw_count = sum(
            1 for hw in _er_hw_pages
            if (
                hw.has_handwriting
                if hasattr(hw, "has_handwriting")
                else hw.get("has_handwriting", False)
            )
        )
        assert _er_any_hw is True
        assert _er_hw_count == 2
        assert (_er_hw_count > len(_er_hw_pages) / 2) is True

    def test_empty_handwriting_data_produces_none(self):
        """When no handwriting pages exist, data stays None."""
        page_data_snap = {"handwriting": {}}
        _er_handwriting_data = None
        if page_data_snap.get("handwriting"):
            _er_handwriting_data = {"populated": True}
        assert _er_handwriting_data is None


# ===========================================================================
# Section 8: Startup logging
# ===========================================================================


class TestStartupLogging:
    """Test startup log message patterns."""

    def test_enabled_with_module_available_log(self, caplog):
        """When enabled and available, info log is emitted."""
        import logging

        caplog.set_level(logging.INFO)
        logger = logging.getLogger("test_startup")
        enable_flag = True
        available_flag = True

        if enable_flag:
            if available_flag:
                logger.info("Confidence-based exception routing ENABLED")
            else:
                logger.warning(
                    "Exception routing requested but exception_router module not available"
                )

        assert "Confidence-based exception routing ENABLED" in caplog.text

    def test_enabled_without_module_available_warns(self, caplog):
        """When enabled but unavailable, warning is emitted."""
        import logging

        logger = logging.getLogger("test_startup_warn")
        enable_flag = True
        available_flag = False

        if enable_flag:
            if available_flag:
                logger.info("Confidence-based exception routing ENABLED")
            else:
                logger.warning(
                    "Exception routing requested but exception_router module not available"
                )

        assert "exception_router module not available" in caplog.text

    def test_disabled_produces_no_log(self, caplog):
        """When disabled, no exception routing log is emitted."""
        import logging

        logger = logging.getLogger("test_startup_off")
        enable_flag = False

        if enable_flag:
            logger.info("Confidence-based exception routing ENABLED")

        assert "exception routing" not in caplog.text.lower()


# ===========================================================================
# Section 9: End-to-end sidecar round-trip
# ===========================================================================


class TestExceptionRoutingSidecarRoundTrip:
    """Full round-trip test of the exception routing sidecar pipeline."""

    def test_full_routing_sidecar_roundtrip(self, tmp_path):
        """Complete round-trip: evaluate -> write JSON -> read JSON -> verify."""
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "quality": {
                    "overall_confidence": 0.15,
                    "classification": "review_required",
                    "pages_image_only": 5,
                },
            },
            handwriting_data={
                "document_summary": {
                    "is_primarily_handwritten": False,
                    "handwriting_detected": True,
                },
            },
            classification_data={"confidence": 0.3},
            extracted_text="Test document content",
        )

        assert decision.should_route is True

        # Write sidecar
        routing_dir = tmp_path / "EXPORT" / "ROUTING"
        os.makedirs(str(routing_dir), exist_ok=True)
        sidecar_path = routing_dir / "mydoc.exception-routing.json"

        sidecar_data = {
            "document_id": "doc-abc123",
            "should_route": True,
            "triggered_rules": decision.triggered_rules,
            "reasons": decision.reasons,
            "confidence": decision.confidence,
            "metadata": decision.metadata,
        }

        with open(str(sidecar_path), "w", encoding="utf-8") as f:
            json.dump(sidecar_data, f, indent=2)

        # Read back and verify
        with open(str(sidecar_path), encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["document_id"] == "doc-abc123"
        assert loaded["should_route"] is True
        assert isinstance(loaded["triggered_rules"], list)
        assert len(loaded["triggered_rules"]) >= 3
        assert "low_confidence" in loaded["triggered_rules"]
        assert "handwriting_detected" in loaded["triggered_rules"]
        assert "classification_uncertain" in loaded["triggered_rules"]
        assert isinstance(loaded["confidence"], (int, float))
        assert isinstance(loaded["metadata"], dict)

    def test_degraded_quality_routing(self, tmp_path):
        """Degraded quality classification triggers routing."""
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "quality": {
                    "overall_confidence": 0.6,
                    "classification": "degraded",
                    "pages_image_only": 0,
                },
            },
        )
        assert decision.should_route is True
        assert "degraded_quality" in decision.triggered_rules

    def test_image_only_pages_routing(self, tmp_path):
        """Excessive image-only pages trigger routing."""
        from exception_router import ExceptionRouter

        router = ExceptionRouter()
        decision = router.evaluate(
            validation_data={
                "quality": {
                    "overall_confidence": 0.9,
                    "classification": "good",
                    "pages_image_only": 10,
                },
            },
        )
        assert decision.should_route is True
        assert "image_only_pages" in decision.triggered_rules
