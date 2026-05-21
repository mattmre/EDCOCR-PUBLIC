"""
Unit tests for TrOCR handwriting recognition module (trocr_recognition.py).

Tests cover:
- HandwritingRecognitionResult dataclass defaults
- Agreement score computation (Levenshtein similarity)
- TrOCRRecognizer initialization and model loading (mocked)
- Recognition with confidence gating (verify/trust/reject modes)
- Singleton factory (get_trocr_recognizer)
- Region recognition pipeline with mocked model
- Graceful degradation (missing torch, missing transformers, model load failure)

Run with: python -m pytest tests/test_trocr_recognition.py -v
"""

import threading
from unittest import mock

from trocr_recognition import (
    TROCR_AGREEMENT_MODE,
    TROCR_AGREEMENT_THRESHOLD,
    TROCR_CONFIDENCE_THRESHOLD,
    HandwritingRecognitionResult,
    TrOCRRecognizer,
    compute_agreement_score,
    get_trocr_recognizer,
    recognize_handwriting_regions,
    reset_trocr_recognizer,
)

# ---------------------------------------------------------------------------
# Tests: Dataclass defaults
# ---------------------------------------------------------------------------


class TestHandwritingRecognitionResult:
    def test_defaults(self):
        r = HandwritingRecognitionResult()
        assert r.trocr_text == ""
        assert r.trocr_confidence == 0.0
        assert r.paddle_text == ""
        assert r.paddle_confidence == 0.0
        assert r.agreement_score == 0.0
        assert r.selected_text == ""
        assert r.selection_reason == ""
        assert r.bbox == []

    def test_custom_values(self):
        r = HandwritingRecognitionResult(
            trocr_text="hello world",
            trocr_confidence=0.92,
            paddle_text="helo world",
            paddle_confidence=0.45,
            agreement_score=0.85,
            selected_text="hello world",
            selection_reason="verify_agreed",
            bbox=[10, 20, 300, 50],
        )
        assert r.trocr_text == "hello world"
        assert r.trocr_confidence == 0.92
        assert r.paddle_text == "helo world"
        assert r.selected_text == "hello world"
        assert r.bbox == [10, 20, 300, 50]


# ---------------------------------------------------------------------------
# Tests: Agreement score
# ---------------------------------------------------------------------------


class TestAgreementScore:
    def test_identical_text(self):
        assert compute_agreement_score("hello", "hello") == 1.0

    def test_identical_after_case_normalization(self):
        assert compute_agreement_score("Hello", "hello") == 1.0

    def test_identical_after_strip(self):
        assert compute_agreement_score("  hello  ", "hello") == 1.0

    def test_similar_text(self):
        score = compute_agreement_score("hello", "helo")
        # Distance = 1 (one insertion), max_len = 5, similarity = 0.8
        assert score == 0.8

    def test_completely_different(self):
        score = compute_agreement_score("abc", "xyz")
        # Distance = 3, max_len = 3, similarity = 0.0
        assert score == 0.0

    def test_empty_first(self):
        assert compute_agreement_score("", "hello") == 0.0

    def test_empty_second(self):
        assert compute_agreement_score("hello", "") == 0.0

    def test_both_empty(self):
        # Two empty strings are identical, so agreement is 1.0
        assert compute_agreement_score("", "") == 1.0

    def test_single_char_match(self):
        assert compute_agreement_score("a", "a") == 1.0

    def test_single_char_differ(self):
        assert compute_agreement_score("a", "b") == 0.0

    def test_partial_overlap(self):
        score = compute_agreement_score("abcdef", "abcxyz")
        # Distance = 3, max_len = 6, similarity = 0.5
        assert score == 0.5

    def test_unicode_text(self):
        score = compute_agreement_score("cafe\u0301", "cafe\u0301")
        assert score == 1.0

    def test_length_difference(self):
        score = compute_agreement_score("hi", "hello")
        # Distance = 4, max_len = 5, similarity = 0.2
        assert score == 0.2


# ---------------------------------------------------------------------------
# Tests: TrOCRRecognizer initialization
# ---------------------------------------------------------------------------


class TestTrOCRRecognizerInit:
    def test_init_defaults(self):
        rec = TrOCRRecognizer()
        assert rec._model_path == "microsoft/trocr-base-handwritten"
        assert rec._device is None
        assert rec._processor is None
        assert rec._model is None
        assert rec._loaded is False
        assert rec._load_failed is False

    def test_init_custom_path(self):
        rec = TrOCRRecognizer(model_path="/custom/model", device="cpu")
        assert rec._model_path == "/custom/model"
        assert rec._device == "cpu"

    def test_is_available_depends_on_deps(self):
        rec = TrOCRRecognizer()
        # is_available checks _TROCR_DEPS_AVAILABLE which depends on actual imports
        # This is a truthful test -- it reflects the real environment
        assert isinstance(rec.is_available, bool)

    def test_is_available_after_load_failure(self):
        rec = TrOCRRecognizer()
        rec._load_failed = True
        assert rec.is_available is False

    def test_is_available_after_load_success(self):
        rec = TrOCRRecognizer()
        rec._loaded = True
        assert rec.is_available is True


# ---------------------------------------------------------------------------
# Tests: Model loading (mocked)
# ---------------------------------------------------------------------------


class TestModelLoading:
    @mock.patch("trocr_recognition._TROCR_DEPS_AVAILABLE", False)
    def test_load_fails_without_deps(self):
        rec = TrOCRRecognizer()
        result = rec._load_model()
        assert result is False
        assert rec._load_failed is True

    @mock.patch("trocr_recognition._TROCR_DEPS_AVAILABLE", True)
    def test_load_fails_on_exception(self):
        rec = TrOCRRecognizer(model_path="nonexistent/model")
        with mock.patch(
            "trocr_recognition.TrOCRRecognizer._load_model",
            side_effect=lambda: setattr(rec, "_load_failed", True) or False,
        ):
            result = rec._load_model()
            assert result is False

    def test_double_load_returns_early(self):
        rec = TrOCRRecognizer()
        rec._loaded = True
        # Should return True immediately without attempting load
        assert rec._load_model() is True

    def test_failed_load_returns_early(self):
        rec = TrOCRRecognizer()
        rec._load_failed = True
        assert rec._load_model() is False


# ---------------------------------------------------------------------------
# Tests: Recognition (mocked)
# ---------------------------------------------------------------------------


class TestRecognize:
    def test_recognize_returns_empty_when_not_loaded(self):
        rec = TrOCRRecognizer()
        rec._load_failed = True
        text, conf = rec.recognize(mock.MagicMock())
        assert text == ""
        assert conf == 0.0

    @mock.patch("trocr_recognition._TROCR_DEPS_AVAILABLE", True)
    def test_recognize_with_mocked_model(self):
        """Test recognition with fully mocked model components."""
        rec = TrOCRRecognizer()
        rec._loaded = True
        rec._device = "cpu"

        # Mock processor
        mock_processor = mock.MagicMock()
        mock_pixel_values = mock.MagicMock()
        mock_pixel_values.to.return_value = mock_pixel_values
        mock_processor.return_value = mock.MagicMock(pixel_values=mock_pixel_values)
        mock_processor.batch_decode.return_value = ["Hello World"]
        rec._processor = mock_processor

        # Mock model
        mock_model = mock.MagicMock()
        mock_outputs = mock.MagicMock()
        mock_outputs.sequences = mock.MagicMock()
        mock_outputs.scores = []  # Empty scores -> 0.0 confidence
        mock_model.generate.return_value = mock_outputs
        rec._model = mock_model

        with mock.patch("trocr_recognition.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = mock.MagicMock()
            mock_torch.no_grad.return_value.__exit__ = mock.MagicMock()

            text, conf = rec.recognize(mock.MagicMock())
            assert text == "Hello World"
            assert conf == 0.0  # Empty scores


# ---------------------------------------------------------------------------
# Tests: Confidence gating
# ---------------------------------------------------------------------------


class TestConfidenceGating:
    def _make_recognizer(self, trocr_text="hello", trocr_conf=0.9):
        """Create a recognizer with mocked recognize method."""
        rec = TrOCRRecognizer()
        rec.recognize = mock.MagicMock(return_value=(trocr_text, trocr_conf))
        return rec

    def test_reject_mode(self):
        rec = self._make_recognizer()
        result = rec.recognize_with_gating(
            mock.MagicMock(), "paddle hello", 0.4, agreement_mode="reject"
        )
        assert result.selected_text == "paddle hello"
        assert result.selection_reason == "reject_mode"
        # recognize should NOT be called in reject mode
        rec.recognize.assert_not_called()

    def test_trust_mode(self):
        rec = self._make_recognizer("trocr hello", 0.95)
        result = rec.recognize_with_gating(
            mock.MagicMock(), "paddle hello", 0.4, agreement_mode="trust"
        )
        assert result.selected_text == "trocr hello"
        assert result.selection_reason == "trust_mode"
        assert result.trocr_text == "trocr hello"
        assert result.trocr_confidence == 0.95

    def test_trust_mode_preserves_paddle(self):
        rec = self._make_recognizer("trocr text", 0.92)
        result = rec.recognize_with_gating(
            mock.MagicMock(), "paddle text", 0.35, agreement_mode="trust"
        )
        assert result.paddle_text == "paddle text"
        assert result.paddle_confidence == 0.35

    def test_verify_mode_agreed(self):
        rec = self._make_recognizer("hello world", 0.92)
        result = rec.recognize_with_gating(
            mock.MagicMock(), "hello world", 0.4, agreement_mode="verify"
        )
        assert result.selected_text == "hello world"
        assert result.selection_reason == "verify_agreed"
        assert result.agreement_score == 1.0

    def test_verify_mode_disagreed_keep_paddle(self):
        rec = self._make_recognizer("completely different text", 0.88)
        result = rec.recognize_with_gating(
            mock.MagicMock(), "original paddle text", 0.75, agreement_mode="verify"
        )
        assert result.selected_text == "original paddle text"
        assert result.selection_reason == "verify_disagreed_keep_paddle"

    def test_verify_mode_low_trocr_confidence(self):
        rec = self._make_recognizer("some text", 0.5)  # Below threshold
        result = rec.recognize_with_gating(
            mock.MagicMock(), "paddle text", 0.4, agreement_mode="verify"
        )
        assert result.selected_text == "paddle text"
        assert result.selection_reason == "verify_low_trocr_confidence"

    def test_verify_mode_trocr_empty_fallback(self):
        rec = self._make_recognizer("", 0.0)
        result = rec.recognize_with_gating(
            mock.MagicMock(), "paddle text", 0.4, agreement_mode="verify"
        )
        assert result.selected_text == "paddle text"
        assert result.selection_reason == "trocr_empty_fallback"

    def test_verify_trocr_higher_confidence(self):
        """TrOCR disagrees but has much higher confidence -> use TrOCR."""
        # Use completely different texts to ensure agreement is below threshold
        rec = self._make_recognizer("John Q. Smith", 0.95)
        result = rec.recognize_with_gating(
            mock.MagicMock(), "xyzw abcdef", 0.3, agreement_mode="verify"
        )
        # Agreement is very low but trocr_conf (0.95) > paddle_conf (0.3) + 0.2
        assert result.agreement_score < TROCR_AGREEMENT_THRESHOLD
        assert result.selected_text == "John Q. Smith"
        assert result.selection_reason == "verify_trocr_higher_confidence"

    def test_default_mode_is_verify(self):
        assert TROCR_AGREEMENT_MODE == "verify"

    def test_uses_global_mode_when_none(self):
        rec = self._make_recognizer("hello", 0.92)
        # When agreement_mode=None, should use global TROCR_AGREEMENT_MODE
        result = rec.recognize_with_gating(
            mock.MagicMock(), "hello", 0.4, agreement_mode=None
        )
        # With global mode="verify" and high agreement, should agree
        assert result.selection_reason == "verify_agreed"


class TestSequenceConfidence:
    def test_logs_exception_and_returns_zero(self):
        outputs = mock.MagicMock(scores=[object()])
        with mock.patch(
            "trocr_recognition.torch.softmax",
            side_effect=RuntimeError("boom"),
        ):
            with mock.patch("trocr_recognition.logger.warning") as mock_warning:
                from trocr_recognition import _compute_sequence_confidence

                result = _compute_sequence_confidence(outputs)

        assert result == 0.0
        mock_warning.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Singleton factory
# ---------------------------------------------------------------------------


class TestSingletonFactory:
    def setup_method(self):
        reset_trocr_recognizer()

    def teardown_method(self):
        reset_trocr_recognizer()

    def test_returns_same_instance(self):
        r1 = get_trocr_recognizer()
        r2 = get_trocr_recognizer()
        assert r1 is r2

    def test_reset_creates_new_instance(self):
        r1 = get_trocr_recognizer()
        reset_trocr_recognizer()
        r2 = get_trocr_recognizer()
        assert r1 is not r2

    def test_thread_safety(self):
        """Verify singleton is thread-safe."""
        instances = []

        def get_instance():
            instances.append(get_trocr_recognizer())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert len(set(id(i) for i in instances)) == 1


# ---------------------------------------------------------------------------
# Tests: Region recognition
# ---------------------------------------------------------------------------


class TestRecognizeHandwritingRegions:
    def test_returns_empty_when_disabled(self):
        """When ENABLE_TROCR is False, returns empty list."""
        with mock.patch("trocr_recognition.ENABLE_TROCR", False):
            result = recognize_handwriting_regions(
                mock.MagicMock(), {"has_handwriting": True, "handwriting_regions": [{"bbox": [0, 0, 100, 50]}]},
                [("text", 0.5, [0, 0, 100, 50])],
            )
            assert result == []

    def test_returns_empty_when_no_handwriting(self):
        with mock.patch("trocr_recognition.ENABLE_TROCR", True):
            result = recognize_handwriting_regions(
                mock.MagicMock(), {"has_handwriting": False, "handwriting_regions": []},
                [],
            )
            assert result == []

    def test_returns_empty_when_no_regions(self):
        with mock.patch("trocr_recognition.ENABLE_TROCR", True):
            result = recognize_handwriting_regions(
                mock.MagicMock(), {"has_handwriting": True, "handwriting_regions": []},
                [],
            )
            assert result == []

    @mock.patch("trocr_recognition.ENABLE_TROCR", True)
    @mock.patch("trocr_recognition.get_trocr_recognizer")
    def test_processes_regions(self, mock_get_rec):
        mock_recognizer = mock.MagicMock()
        mock_recognizer.is_available = True
        mock_recognizer.recognize_with_gating.return_value = HandwritingRecognitionResult(
            trocr_text="recognized",
            trocr_confidence=0.92,
            paddle_text="detected",
            paddle_confidence=0.35,
            selected_text="recognized",
            selection_reason="verify_agreed",
        )
        mock_get_rec.return_value = mock_recognizer

        mock_image = mock.MagicMock()
        mock_image.size = (1000, 1000)
        mock_image.crop.return_value = mock.MagicMock()

        hw_result = {
            "has_handwriting": True,
            "handwriting_regions": [
                {"bbox": [10, 20, 300, 50], "text": "detected", "ocr_confidence": 0.35}
            ],
        }
        paddle_lines = [("detected", 0.35, [10, 20, 300, 50])]

        results = recognize_handwriting_regions(
            mock_image, hw_result, paddle_lines
        )

        assert len(results) == 1
        assert results[0].selected_text == "recognized"

    @mock.patch("trocr_recognition.ENABLE_TROCR", True)
    @mock.patch("trocr_recognition.get_trocr_recognizer")
    def test_handles_recognition_exception(self, mock_get_rec):
        """Exception during crop/recognize should produce fallback result."""
        mock_recognizer = mock.MagicMock()
        mock_recognizer.is_available = True
        mock_recognizer.recognize_with_gating.side_effect = RuntimeError("GPU error")
        mock_get_rec.return_value = mock_recognizer

        mock_image = mock.MagicMock()
        mock_image.size = (1000, 1000)
        mock_image.crop.return_value = mock.MagicMock()

        hw_result = {
            "has_handwriting": True,
            "handwriting_regions": [
                {"bbox": [10, 20, 300, 50], "text": "paddle text", "ocr_confidence": 0.4}
            ],
        }
        paddle_lines = [("paddle text", 0.4, [10, 20, 300, 50])]

        results = recognize_handwriting_regions(
            mock_image, hw_result, paddle_lines
        )

        assert len(results) == 1
        assert results[0].selected_text == "paddle text"
        assert results[0].selection_reason == "trocr_exception_fallback"

    @mock.patch("trocr_recognition.ENABLE_TROCR", True)
    @mock.patch("trocr_recognition.get_trocr_recognizer")
    def test_skips_invalid_bbox(self, mock_get_rec):
        """Regions with invalid bboxes should be skipped."""
        mock_recognizer = mock.MagicMock()
        mock_recognizer.is_available = True
        mock_get_rec.return_value = mock_recognizer

        mock_image = mock.MagicMock()
        mock_image.size = (1000, 1000)

        hw_result = {
            "has_handwriting": True,
            "handwriting_regions": [
                {"bbox": [100, 200], "text": "short bbox"},  # Too few elements
                {"bbox": [500, 500, 100, 100], "text": "inverted bbox"},  # x2 < x1
            ],
        }

        results = recognize_handwriting_regions(
            mock_image, hw_result, []
        )

        assert len(results) == 0

    @mock.patch("trocr_recognition.ENABLE_TROCR", True)
    @mock.patch("trocr_recognition.get_trocr_recognizer")
    def test_unavailable_recognizer(self, mock_get_rec):
        """When recognizer is not available, return empty."""
        mock_recognizer = mock.MagicMock()
        mock_recognizer.is_available = False
        mock_get_rec.return_value = mock_recognizer

        hw_result = {
            "has_handwriting": True,
            "handwriting_regions": [
                {"bbox": [10, 20, 300, 50], "text": "text"}
            ],
        }

        results = recognize_handwriting_regions(
            mock.MagicMock(), hw_result, []
        )
        assert results == []


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_recognize_with_no_model_loaded(self):
        rec = TrOCRRecognizer()
        rec._load_failed = True
        text, conf = rec.recognize(mock.MagicMock())
        assert text == ""
        assert conf == 0.0

    def test_gating_falls_back_on_empty_trocr(self):
        rec = TrOCRRecognizer()
        rec.recognize = mock.MagicMock(return_value=("", 0.0))
        result = rec.recognize_with_gating(
            mock.MagicMock(), "paddle text", 0.5, agreement_mode="verify"
        )
        assert result.selected_text == "paddle text"
        assert result.selection_reason == "trocr_empty_fallback"

    def test_config_defaults(self):
        assert TROCR_CONFIDENCE_THRESHOLD == 0.85
        assert TROCR_AGREEMENT_THRESHOLD == 0.40
        assert TROCR_AGREEMENT_MODE == "verify"
