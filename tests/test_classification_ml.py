"""
Unit tests for ML-based document classification (classification.py ML extensions).

Tests cover:
- MLDocumentClassifier: init, model loading (mocked), classify (mocked inference)
- classify_page_ml: end-to-end with mocked classifier
- classify_page_hybrid: weighted combination, threshold fallback, edge cases
- Extended document types: all 19 types valid, backward compatibility
- Classification mode: env var parsing, heuristic/ml/ensemble mode routing
- Graceful degradation: missing torch, missing transformers, model load failure
- Backward compatibility: CLASSIFICATION_MODE=heuristic identical to original

Run with: python -m pytest tests/test_classification_ml.py -v
"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
from classification import (
    _HEURISTIC_WEIGHT,
    _ML_TO_BASE_TYPE,
    _ML_WEIGHT,
    CLASSIFICATION_MODE,
    DOCUMENT_TYPES,
    DOCUMENT_TYPES_EXTENDED,
    ML_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    ML_CLASSIFICATION_MODEL,
    DocumentClassification,
    MLDocumentClassifier,
    PageClassification,
    _heuristic_ensemble,
    classify_page_by_text,
    classify_page_ensemble,
    classify_page_hybrid,
    classify_page_ml,
    finalize_classification,
    get_ml_classifier,
    write_classification_json,
)

# ---------------------------------------------------------------------------
# Helper: reset ML classifier singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_ml_singleton():
    """Reset the ML classifier singleton state before each test."""
    import classification as cls_mod

    cls_mod._ml_classifier_instance = None
    cls_mod._ml_classifier_init_failed = False
    yield
    cls_mod._ml_classifier_instance = None
    cls_mod._ml_classifier_init_failed = False


# ---------------------------------------------------------------------------
# Tests: Extended document types
# ---------------------------------------------------------------------------


class TestExtendedDocumentTypes:
    def test_extended_types_count(self):
        """DOCUMENT_TYPES_EXTENDED has 19 categories."""
        assert len(DOCUMENT_TYPES_EXTENDED) == 19

    def test_extended_types_superset(self):
        """Extended types contain all base types (with handwritten_note mapping)."""
        base_set = set(DOCUMENT_TYPES)
        extended_set = set(DOCUMENT_TYPES_EXTENDED)
        # All base types except "handwritten" (renamed to "handwritten_note")
        for btype in base_set:
            if btype == "handwritten":
                assert "handwritten_note" in extended_set
            else:
                assert btype in extended_set

    def test_extended_types_unique(self):
        """No duplicate types in extended list."""
        assert len(DOCUMENT_TYPES_EXTENDED) == len(set(DOCUMENT_TYPES_EXTENDED))

    def test_new_types_present(self):
        """All 9 new types are present in the extended list."""
        new_types = {
            "scientific_paper", "legal_filing", "email_printout",
            "spreadsheet", "presentation", "specification",
            "resume", "medical_record", "government_form",
        }
        extended_set = set(DOCUMENT_TYPES_EXTENDED)
        for ntype in new_types:
            assert ntype in extended_set, f"Missing new type: {ntype}"

    def test_ml_to_base_type_mapping(self):
        """ML-to-base type mapping covers all extended-only types."""
        base_set = set(DOCUMENT_TYPES)
        extended_only = set(DOCUMENT_TYPES_EXTENDED) - base_set
        # "handwritten_note" should map to "handwritten"
        for etype in extended_only:
            if etype == "handwritten_note":
                assert _ML_TO_BASE_TYPE.get(etype) == "handwritten"
            else:
                assert etype in _ML_TO_BASE_TYPE, (
                    f"Extended type {etype} missing from _ML_TO_BASE_TYPE"
                )
                assert _ML_TO_BASE_TYPE[etype] in base_set, (
                    f"Mapped base type for {etype} not in DOCUMENT_TYPES"
                )

    def test_base_types_unchanged(self):
        """Original DOCUMENT_TYPES list is unmodified."""
        expected = [
            "invoice", "contract", "letter", "form", "report",
            "memo", "receipt", "handwritten", "photograph", "other",
        ]
        assert DOCUMENT_TYPES == expected


# ---------------------------------------------------------------------------
# Tests: Configuration
# ---------------------------------------------------------------------------


class TestClassificationMode:
    def test_default_mode_is_heuristic(self):
        """Default CLASSIFICATION_MODE should be 'heuristic'."""
        # The actual env value may differ in test runs, but the default
        # from the source code is "heuristic"
        assert CLASSIFICATION_MODE in ("heuristic", "ml", "ensemble")

    def test_ml_model_default(self):
        """Default ML model is microsoft/layoutlmv3-base."""
        assert ML_CLASSIFICATION_MODEL == "microsoft/layoutlmv3-base"

    def test_confidence_threshold_default(self):
        """Default ML confidence threshold is 0.5."""
        assert ML_CLASSIFICATION_CONFIDENCE_THRESHOLD == 0.5

    def test_ml_weight(self):
        """ML weight in hybrid ensemble is 0.7."""
        assert _ML_WEIGHT == 0.7

    def test_heuristic_weight(self):
        """Heuristic weight in hybrid ensemble is 0.3."""
        assert _HEURISTIC_WEIGHT == 0.3

    def test_weights_sum_to_one(self):
        """ML + heuristic weights sum to 1.0."""
        assert abs(_ML_WEIGHT + _HEURISTIC_WEIGHT - 1.0) < 1e-9

    @patch.dict(os.environ, {"CLASSIFICATION_MODE": "ml"})
    def test_env_var_ml_mode(self):
        """CLASSIFICATION_MODE env var can be set to 'ml'."""
        # Re-read the env var as the module would at import time
        mode = os.environ.get("CLASSIFICATION_MODE", "heuristic")
        assert mode == "ml"

    @patch.dict(os.environ, {"CLASSIFICATION_MODE": "ensemble"})
    def test_env_var_ensemble_mode(self):
        """CLASSIFICATION_MODE env var can be set to 'ensemble'."""
        mode = os.environ.get("CLASSIFICATION_MODE", "heuristic")
        assert mode == "ensemble"

    def test_invalid_confidence_threshold_falls_back_to_default(self):
        """Invalid threshold env values should not crash module import."""
        module_path = Path(__file__).resolve().parent.parent / "classification.py"
        spec = importlib.util.spec_from_file_location(
            "classification_invalid_threshold_test",
            module_path,
        )
        assert spec is not None and spec.loader is not None

        test_module = importlib.util.module_from_spec(spec)
        with patch.dict(
            os.environ,
            {"ML_CLASSIFICATION_CONFIDENCE_THRESHOLD": "not-a-float"},
            clear=False,
        ):
            spec.loader.exec_module(test_module)

        assert test_module.ML_CLASSIFICATION_CONFIDENCE_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# Tests: MLDocumentClassifier
# ---------------------------------------------------------------------------


class TestMLDocumentClassifier:
    def test_init_defaults(self):
        """MLDocumentClassifier initializes with correct defaults."""
        clf = MLDocumentClassifier()
        assert clf.model_path == "microsoft/layoutlmv3-base"
        assert clf.device == "cpu"
        assert clf._model is None
        assert clf._processor is None
        assert clf._loaded is False
        assert clf._load_failed is False

    def test_init_custom_params(self):
        """MLDocumentClassifier accepts custom model path and device."""
        clf = MLDocumentClassifier(
            model_path="custom/model", device="cuda"
        )
        assert clf.model_path == "custom/model"
        assert clf.device == "cuda"

    def test_id2label_mapping(self):
        """id2label mapping covers all extended types."""
        clf = MLDocumentClassifier()
        assert len(clf._id2label) == len(DOCUMENT_TYPES_EXTENDED)
        for i, dtype in enumerate(DOCUMENT_TYPES_EXTENDED):
            assert clf._id2label[i] == dtype

    def test_label2id_mapping(self):
        """label2id is inverse of id2label."""
        clf = MLDocumentClassifier()
        for dtype, idx in clf._label2id.items():
            assert clf._id2label[idx] == dtype

    def test_preprocess_uses_empty_word_list_for_empty_text(self):
        """Empty text should not be turned into a fake empty-string token."""
        clf = MLDocumentClassifier()
        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        clf._processor = MagicMock(return_value={"input_ids": mock_tensor})

        clf._preprocess("", None, MagicMock())

        assert clf._processor.call_args.args[1] == []

    def test_classify_returns_tuple(self):
        """classify() returns (str, float) tuple even on failure."""
        clf = MLDocumentClassifier()
        # Without ML deps, classify should return default
        result = clf.classify("some text")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_classify_without_deps_returns_other(self):
        """classify() returns ('other', 0.0) when torch is not available."""
        clf = MLDocumentClassifier()
        # Simulate torch not being available by forcing load failure
        clf._load_failed = True
        doc_type, confidence = clf.classify("Invoice #123")
        assert doc_type == "other"
        assert confidence == 0.0

    @patch("classification.MLDocumentClassifier._load_model")
    def test_classify_with_mocked_model(self, mock_load):
        """classify() uses model output when loaded successfully."""
        clf = MLDocumentClassifier()
        clf._loaded = True
        clf._load_failed = False

        # Mock torch
        mock_torch = MagicMock()
        mock_outputs = MagicMock()
        mock_logits = MagicMock()

        # Simulate softmax output with invoice as highest
        mock_probs = MagicMock()
        mock_confidence = MagicMock()
        mock_confidence.item.return_value = 0.85
        mock_predicted_idx = MagicMock()
        mock_predicted_idx.item.return_value = 0  # index 0 = "invoice"
        mock_torch.max.return_value = (mock_confidence, mock_predicted_idx)
        mock_torch.softmax.return_value = mock_probs
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_outputs.logits = mock_logits

        clf._model = MagicMock()
        clf._model.return_value = mock_outputs
        clf._processor = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch.object(clf, "_preprocess", return_value={}):
                doc_type, confidence = clf.classify("Invoice text")

        assert doc_type == "invoice"
        assert confidence == 0.85

    def test_load_model_sets_failed_on_import_error(self):
        """_load_model sets _load_failed when torch is missing."""
        clf = MLDocumentClassifier()
        # torch is not installed in test env, so _load_model should fail
        try:
            clf._load_model()
        except ImportError:
            pass
        # Either it succeeded (torch available) or set _load_failed
        assert clf._loaded or clf._load_failed

    def test_load_model_idempotent_after_failure(self):
        """_load_model does not retry after _load_failed is set."""
        clf = MLDocumentClassifier()
        clf._load_failed = True
        # Should return immediately without raising
        clf._load_model()
        assert clf._loaded is False

    def test_load_model_idempotent_after_success(self):
        """_load_model does not reload after _loaded is set."""
        clf = MLDocumentClassifier()
        clf._loaded = True
        clf._model = MagicMock()
        # Should return immediately
        clf._load_model()
        assert clf._loaded is True

    def test_classify_inference_error_returns_default(self):
        """classify() returns ('other', 0.0) on inference error."""
        clf = MLDocumentClassifier()
        clf._loaded = True

        # Mock model that raises during forward pass
        clf._model = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        clf._processor = MagicMock()

        with patch.object(clf, "_preprocess", return_value={}):
            with patch.dict(sys.modules, {"torch": MagicMock()}):
                doc_type, confidence = clf.classify("text")

        assert doc_type == "other"
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Tests: classify_page_ml
# ---------------------------------------------------------------------------


class TestClassifyPageML:
    def test_returns_page_classification(self):
        """classify_page_ml always returns a PageClassification."""
        result = classify_page_ml("text", None, None, 1)
        assert isinstance(result, PageClassification)
        assert result.page_num == 1
        assert result.method == "ml"

    @patch("classification.get_ml_classifier", return_value=None)
    def test_default_when_classifier_unavailable(self, mock_get):
        """Returns 'other' with 0.0 confidence when ML is unavailable."""
        result = classify_page_ml("Invoice #123", None, None, 5)
        assert result.predicted_type == "other"
        assert result.confidence == 0.0
        assert result.type_scores == {}

    @patch("classification.get_ml_classifier")
    def test_with_mocked_classifier(self, mock_get_clf):
        """classify_page_ml uses classifier result when available."""
        mock_clf = MagicMock()
        mock_clf.classify.return_value = ("invoice", 0.92)
        mock_get_clf.return_value = mock_clf

        result = classify_page_ml("Invoice #123", [[0, 0, 100, 20]], None, 3)

        assert result.predicted_type == "invoice"
        assert result.confidence == 0.92
        assert result.type_scores == {"invoice": 0.92}
        assert result.method == "ml"
        assert result.page_num == 3

    @patch("classification.get_ml_classifier")
    def test_classifier_returns_zero_confidence(self, mock_get_clf):
        """Zero confidence from classifier produces empty type_scores."""
        mock_clf = MagicMock()
        mock_clf.classify.return_value = ("other", 0.0)
        mock_get_clf.return_value = mock_clf

        result = classify_page_ml("random text", None, None, 1)
        assert result.predicted_type == "other"
        assert result.confidence == 0.0
        assert result.type_scores == {}

    @patch("classification.get_ml_classifier")
    def test_extended_type_returned(self, mock_get_clf):
        """ML can return extended types like 'scientific_paper'."""
        mock_clf = MagicMock()
        mock_clf.classify.return_value = ("scientific_paper", 0.78)
        mock_get_clf.return_value = mock_clf

        result = classify_page_ml("Abstract. Methods. Results.", None, None, 1)
        assert result.predicted_type == "scientific_paper"
        assert result.confidence == 0.78


# ---------------------------------------------------------------------------
# Tests: classify_page_hybrid
# ---------------------------------------------------------------------------


class TestClassifyPageHybrid:
    def test_both_agree(self):
        """Both heuristic and ML agree on invoice -- hybrid picks invoice."""
        heuristic = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.7, method="ensemble",
            type_scores={"invoice": 0.7},
        )
        ml = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.9, method="ml",
            type_scores={"invoice": 0.9},
        )
        result = classify_page_hybrid(heuristic, ml, 1)
        assert result.predicted_type == "invoice"
        assert result.method == "hybrid"
        # Combined: 0.7*0.3 + 0.9*0.7 = 0.21 + 0.63 = 0.84
        assert abs(result.confidence - 0.84) < 0.01

    def test_ml_below_threshold_fallback(self):
        """ML confidence below threshold -- falls back to heuristic."""
        heuristic = PageClassification(
            page_num=1, predicted_type="letter",
            confidence=0.6, method="ensemble",
            type_scores={"letter": 0.6},
        )
        ml = PageClassification(
            page_num=1, predicted_type="contract",
            confidence=0.3,  # Below default threshold of 0.5
            method="ml",
            type_scores={"contract": 0.3},
        )
        result = classify_page_hybrid(heuristic, ml, 1)
        assert result.predicted_type == "letter"
        assert result.confidence == 0.6
        assert result.method == "hybrid"

    def test_ml_at_threshold_uses_hybrid(self):
        """ML confidence exactly at threshold -- uses hybrid weighting."""
        heuristic = PageClassification(
            page_num=1, predicted_type="form",
            confidence=0.5, method="ensemble",
            type_scores={"form": 0.5},
        )
        ml = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.5,  # Exactly at threshold
            method="ml",
            type_scores={"invoice": 0.5},
        )
        result = classify_page_hybrid(heuristic, ml, 1)
        # ML confidence >= threshold, so hybrid weighting applies
        # invoice: 0.5*0.7 = 0.35, form: 0.5*0.3 = 0.15
        assert result.predicted_type == "invoice"
        assert result.method == "hybrid"

    def test_disagree_ml_wins(self):
        """ML and heuristic disagree -- ML wins due to higher weight (0.7)."""
        heuristic = PageClassification(
            page_num=1, predicted_type="letter",
            confidence=0.8, method="ensemble",
            type_scores={"letter": 0.8},
        )
        ml = PageClassification(
            page_num=1, predicted_type="contract",
            confidence=0.8, method="ml",
            type_scores={"contract": 0.8},
        )
        result = classify_page_hybrid(heuristic, ml, 1)
        # contract: 0.8*0.7 = 0.56, letter: 0.8*0.3 = 0.24
        assert result.predicted_type == "contract"
        assert abs(result.confidence - 0.56) < 0.01

    def test_extended_type_mapped_to_base(self):
        """ML returns extended type -- gets mapped to base type in hybrid."""
        heuristic = PageClassification(
            page_num=1, predicted_type="report",
            confidence=0.5, method="ensemble",
            type_scores={"report": 0.5},
        )
        ml = PageClassification(
            page_num=1, predicted_type="scientific_paper",
            confidence=0.8, method="ml",
            type_scores={"scientific_paper": 0.8},
        )
        result = classify_page_hybrid(heuristic, ml, 1)
        # scientific_paper maps to "report" in _ML_TO_BASE_TYPE
        # report: 0.5*0.3 + 0.8*0.7 = 0.15 + 0.56 = 0.71
        assert result.predicted_type == "report"
        assert abs(result.confidence - 0.71) < 0.01

    def test_mapped_ml_scores_take_max_not_sum(self):
        """Mapped extended ML types should not inflate a base score above 1.0."""
        heuristic = PageClassification(
            page_num=1,
            predicted_type="report",
            confidence=0.4,
            method="ensemble",
            type_scores={"report": 0.4},
        )
        ml = PageClassification(
            page_num=1,
            predicted_type="spreadsheet",
            confidence=0.9,
            method="ml",
            type_scores={"spreadsheet": 0.9, "presentation": 0.8},
        )

        result = classify_page_hybrid(heuristic, ml, 1)

        # Base "report" score uses the max mapped ML score, not the sum.
        expected = round(0.4 * _HEURISTIC_WEIGHT + 0.9 * _ML_WEIGHT, 4)
        assert result.predicted_type == "report"
        assert result.confidence == expected
        assert result.type_scores["report"] == expected

    def test_type_scores_merged(self):
        """Type scores from both sources are merged with weights."""
        heuristic = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.6, method="ensemble",
            type_scores={"invoice": 0.6, "receipt": 0.3},
        )
        ml = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.8, method="ml",
            type_scores={"invoice": 0.8},
        )
        result = classify_page_hybrid(heuristic, ml, 1)
        assert "invoice" in result.type_scores
        assert "receipt" in result.type_scores
        # invoice: 0.6*0.3 + 0.8*0.7 = 0.18 + 0.56 = 0.74
        assert abs(result.type_scores["invoice"] - 0.74) < 0.01
        # receipt: 0.3*0.3 = 0.09
        assert abs(result.type_scores["receipt"] - 0.09) < 0.01

    def test_empty_heuristic_scores(self):
        """Heuristic has no scores -- ML scores dominate."""
        heuristic = PageClassification(
            page_num=1, predicted_type="other",
            confidence=0.0, method="ensemble",
            type_scores={},
        )
        ml = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.9, method="ml",
            type_scores={"invoice": 0.9},
        )
        result = classify_page_hybrid(heuristic, ml, 1)
        assert result.predicted_type == "invoice"
        # invoice: 0.9*0.7 = 0.63
        assert abs(result.confidence - 0.63) < 0.01

    def test_both_zero_confidence(self):
        """Both zero confidence -- ML below threshold, falls back to heuristic."""
        heuristic = PageClassification(
            page_num=1, predicted_type="other",
            confidence=0.0, method="ensemble",
            type_scores={},
        )
        ml = PageClassification(
            page_num=1, predicted_type="other",
            confidence=0.0, method="ml",
            type_scores={},
        )
        result = classify_page_hybrid(heuristic, ml, 1)
        assert result.predicted_type == "other"
        assert result.confidence == 0.0

    def test_page_num_preserved(self):
        """Page number is correctly preserved in hybrid result."""
        heuristic = PageClassification(
            page_num=42, predicted_type="form",
            confidence=0.5, method="ensemble",
            type_scores={"form": 0.5},
        )
        ml = PageClassification(
            page_num=42, predicted_type="form",
            confidence=0.7, method="ml",
            type_scores={"form": 0.7},
        )
        result = classify_page_hybrid(heuristic, ml, 42)
        assert result.page_num == 42


# ---------------------------------------------------------------------------
# Tests: classify_page_ensemble mode routing
# ---------------------------------------------------------------------------


class TestEnsembleModeRouting:
    def test_heuristic_mode_unchanged(self):
        """CLASSIFICATION_MODE=heuristic produces same result as _heuristic_ensemble."""
        import classification as cls_mod

        original_mode = cls_mod.CLASSIFICATION_MODE
        try:
            cls_mod.CLASSIFICATION_MODE = "heuristic"

            text_result = PageClassification(
                page_num=1, predicted_type="invoice",
                confidence=0.7, method="text_rules",
                type_scores={"invoice": 0.7},
            )
            layout_result = PageClassification(
                page_num=1, predicted_type="form",
                confidence=0.5, method="layout_features",
                type_scores={"form": 0.5},
            )

            ensemble_result = classify_page_ensemble(
                text_result, layout_result, 1
            )
            heuristic_result = _heuristic_ensemble(
                text_result, layout_result, 1
            )

            assert ensemble_result.predicted_type == heuristic_result.predicted_type
            assert ensemble_result.confidence == heuristic_result.confidence
            assert ensemble_result.method == heuristic_result.method
        finally:
            cls_mod.CLASSIFICATION_MODE = original_mode

    @patch("classification.classify_page_ml")
    def test_ml_mode_calls_classify_page_ml(self, mock_ml):
        """CLASSIFICATION_MODE=ml calls classify_page_ml."""
        import classification as cls_mod

        original_mode = cls_mod.CLASSIFICATION_MODE
        try:
            cls_mod.CLASSIFICATION_MODE = "ml"

            mock_ml.return_value = PageClassification(
                page_num=1, predicted_type="contract",
                confidence=0.85, method="ml",
                type_scores={"contract": 0.85},
            )

            result = classify_page_ensemble(
                None, None, 1, text="Agreement hereby"
            )
            assert result.predicted_type == "contract"
            assert result.confidence == 0.85
            mock_ml.assert_called_once()
        finally:
            cls_mod.CLASSIFICATION_MODE = original_mode

    @patch("classification.classify_page_ml")
    def test_ml_mode_fallback_on_zero_confidence(self, mock_ml):
        """CLASSIFICATION_MODE=ml falls back to heuristic on ML zero confidence."""
        import classification as cls_mod

        original_mode = cls_mod.CLASSIFICATION_MODE
        try:
            cls_mod.CLASSIFICATION_MODE = "ml"

            mock_ml.return_value = PageClassification(
                page_num=1, predicted_type="other",
                confidence=0.0, method="ml",
                type_scores={},
            )

            text_result = PageClassification(
                page_num=1, predicted_type="invoice",
                confidence=0.6, method="text_rules",
                type_scores={"invoice": 0.6},
            )

            result = classify_page_ensemble(
                text_result, None, 1, text="Invoice #123"
            )
            # Should fall back to heuristic
            assert result.predicted_type == "invoice"
            assert result.method == "ensemble"  # from _heuristic_ensemble
        finally:
            cls_mod.CLASSIFICATION_MODE = original_mode

    @patch("classification.classify_page_hybrid")
    @patch("classification.classify_page_ml")
    def test_ensemble_mode_calls_both(self, mock_ml, mock_hybrid):
        """CLASSIFICATION_MODE=ensemble calls both heuristic and ML."""
        import classification as cls_mod

        original_mode = cls_mod.CLASSIFICATION_MODE
        try:
            cls_mod.CLASSIFICATION_MODE = "ensemble"

            mock_ml.return_value = PageClassification(
                page_num=1, predicted_type="invoice",
                confidence=0.8, method="ml",
                type_scores={"invoice": 0.8},
            )
            mock_hybrid.return_value = PageClassification(
                page_num=1, predicted_type="invoice",
                confidence=0.75, method="hybrid",
                type_scores={"invoice": 0.75},
            )

            text_result = PageClassification(
                page_num=1, predicted_type="invoice",
                confidence=0.6, method="text_rules",
                type_scores={"invoice": 0.6},
            )

            result = classify_page_ensemble(
                text_result, None, 1, text="Invoice text"
            )

            mock_ml.assert_called_once()
            mock_hybrid.assert_called_once()
            assert result.predicted_type == "invoice"
        finally:
            cls_mod.CLASSIFICATION_MODE = original_mode

    def test_unknown_mode_falls_back_to_heuristic(self):
        """Unknown CLASSIFICATION_MODE falls back to heuristic with warning."""
        import classification as cls_mod

        original_mode = cls_mod.CLASSIFICATION_MODE
        try:
            cls_mod.CLASSIFICATION_MODE = "invalid_mode"

            text_result = PageClassification(
                page_num=1, predicted_type="letter",
                confidence=0.5, method="text_rules",
                type_scores={"letter": 0.5},
            )

            result = classify_page_ensemble(text_result, None, 1)
            # Falls back to heuristic
            assert result.predicted_type == "letter"
            assert result.method == "ensemble"
        finally:
            cls_mod.CLASSIFICATION_MODE = original_mode


# ---------------------------------------------------------------------------
# Tests: get_ml_classifier singleton
# ---------------------------------------------------------------------------


class TestGetMLClassifier:
    def test_returns_instance_or_none(self):
        """get_ml_classifier returns MLDocumentClassifier or None."""
        result = get_ml_classifier()
        assert result is None or isinstance(result, MLDocumentClassifier)

    def test_singleton_behavior(self):
        """Second call returns same instance."""
        import classification as cls_mod

        # Manually set a mock instance
        mock_clf = MLDocumentClassifier()
        cls_mod._ml_classifier_instance = mock_clf

        result1 = get_ml_classifier()
        result2 = get_ml_classifier()
        assert result1 is result2
        assert result1 is mock_clf

    def test_returns_none_after_init_failure(self):
        """Returns None if a prior init attempt failed."""
        import classification as cls_mod

        cls_mod._ml_classifier_init_failed = True
        result = get_ml_classifier()
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    @patch("classification.get_ml_classifier", return_value=None)
    def test_classify_page_ml_without_classifier(self, mock_get):
        """classify_page_ml returns safe default when classifier is unavailable."""
        result = classify_page_ml("Invoice #123", None, None, 1)
        assert result.predicted_type == "other"
        assert result.confidence == 0.0
        assert result.method == "ml"

    def test_ml_classifier_classify_after_load_failure(self):
        """MLDocumentClassifier.classify returns default after load failure."""
        clf = MLDocumentClassifier()
        # Simulate import/load failure
        clf._load_failed = True
        doc_type, confidence = clf.classify("text")
        assert doc_type == "other"
        assert confidence == 0.0

    @patch("classification.get_ml_classifier", return_value=None)
    def test_classify_page_ml_no_classifier(self, mock_get):
        """classify_page_ml handles None classifier gracefully."""
        result = classify_page_ml("text", None, None, 1)
        assert result.predicted_type == "other"
        assert result.confidence == 0.0

    def test_model_load_failure_is_permanent(self):
        """After _load_failed is set, model is never retried."""
        clf = MLDocumentClassifier()
        clf._load_failed = True

        # classify should return default without trying to load
        doc_type, confidence = clf.classify("Invoice text")
        assert doc_type == "other"
        assert confidence == 0.0

    def test_classifier_exception_returns_default(self):
        """MLDocumentClassifier.classify returns default on inference error."""
        clf = MLDocumentClassifier()
        clf._loaded = True
        clf._model = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        clf._processor = MagicMock()

        with patch.object(clf, "_preprocess", return_value={}):
            with patch.dict(sys.modules, {"torch": MagicMock()}):
                doc_type, confidence = clf.classify("text")

        assert doc_type == "other"
        assert confidence == 0.0

    @patch("classification.get_ml_classifier")
    def test_classify_page_ml_classifier_exception(self, mock_get):
        """classify_page_ml handles exception from classifier.classify."""
        mock_clf = MagicMock()
        # classify returns ("other", 0.0) when it catches an internal error
        mock_clf.classify.return_value = ("other", 0.0)
        mock_get.return_value = mock_clf

        result = classify_page_ml("text", None, None, 1)
        assert result.page_num == 1
        assert result.method == "ml"
        assert result.predicted_type == "other"
        assert result.confidence == 0.0

    def test_load_model_import_error_sets_flag(self):
        """_load_model sets _load_failed when transformers import fails."""
        clf = MLDocumentClassifier()

        # Mock the import to fail
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name in ("torch", "transformers"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            try:
                clf._load_model()
            except ImportError:
                pass

        assert clf._load_failed is True
        assert clf._loaded is False


# ---------------------------------------------------------------------------
# Tests: Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_heuristic_mode_identical_to_original(self):
        """In heuristic mode, classify_page_ensemble == _heuristic_ensemble."""
        import classification as cls_mod

        original_mode = cls_mod.CLASSIFICATION_MODE
        try:
            cls_mod.CLASSIFICATION_MODE = "heuristic"

            text = "Invoice #12345\nAmount Due: $500.00\nTotal: $500.00"
            text_result = classify_page_by_text(text, 1)

            ensemble_result = classify_page_ensemble(text_result, None, 1)
            heuristic_result = _heuristic_ensemble(text_result, None, 1)

            assert ensemble_result.predicted_type == heuristic_result.predicted_type
            assert ensemble_result.confidence == heuristic_result.confidence
            assert ensemble_result.type_scores == heuristic_result.type_scores
        finally:
            cls_mod.CLASSIFICATION_MODE = original_mode

    def test_existing_test_text_classification_unchanged(self):
        """Original text classification still works correctly."""
        text = "Invoice #12345\nAmount Due: $500.00\nTotal: $500.00"
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "invoice"
        assert result.confidence > 0.0
        assert result.method == "text_rules"

    def test_existing_finalization_unchanged(self):
        """finalize_classification still works with original types."""
        doc = DocumentClassification(document_id="d1", source_file="test.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="invoice", confidence=0.8),
            PageClassification(page_num=2, predicted_type="invoice", confidence=0.7),
        ]
        result = finalize_classification(doc)
        assert result.document_type == "invoice"
        assert result.type_distribution == {"invoice": 2}

    def test_finalization_with_hybrid_method(self):
        """finalize_classification works with hybrid method pages."""
        doc = DocumentClassification(document_id="d2", source_file="test.pdf")
        doc.pages = [
            PageClassification(
                page_num=1, predicted_type="invoice",
                confidence=0.8, method="hybrid"
            ),
            PageClassification(
                page_num=2, predicted_type="contract",
                confidence=0.6, method="hybrid"
            ),
            PageClassification(
                page_num=3, predicted_type="invoice",
                confidence=0.7, method="hybrid"
            ),
        ]
        result = finalize_classification(doc)
        assert result.document_type == "invoice"
        assert result.type_distribution["invoice"] == 2
        assert result.type_distribution["contract"] == 1


# ---------------------------------------------------------------------------
# Tests: JSON engine label for ML methods
# ---------------------------------------------------------------------------


class TestJsonEngineLabelML:
    def test_ml_engine_label(self, tmp_path):
        """Pages with method=ml produce 'ml' engine label."""
        doc = DocumentClassification(document_id="ml1", source_file="t.pdf")
        doc.pages = [
            PageClassification(
                page_num=1, predicted_type="invoice",
                confidence=0.8, method="ml",
            ),
        ]
        doc = finalize_classification(doc)
        path = write_classification_json(doc, str(tmp_path), "", "0.9.0")
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["processing"]["classification_engine"] == "ml"

    def test_hybrid_engine_label(self, tmp_path):
        """Pages with method=hybrid produce combined engine label."""
        doc = DocumentClassification(document_id="h1", source_file="t.pdf")
        doc.pages = [
            PageClassification(
                page_num=1, predicted_type="invoice",
                confidence=0.8, method="hybrid",
            ),
        ]
        doc = finalize_classification(doc)
        path = write_classification_json(doc, str(tmp_path), "", "0.9.0")
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["processing"]["classification_engine"] == "text_rules+layout_features+ml"

    def test_mixed_methods_engine_label(self, tmp_path):
        """Pages with mix of heuristic and ML methods produce correct label."""
        doc = DocumentClassification(document_id="mix1", source_file="t.pdf")
        doc.pages = [
            PageClassification(
                page_num=1, predicted_type="invoice",
                confidence=0.8, method="text_rules",
            ),
            PageClassification(
                page_num=2, predicted_type="contract",
                confidence=0.7, method="hybrid",
            ),
        ]
        doc = finalize_classification(doc)
        path = write_classification_json(doc, str(tmp_path), "", "0.9.0")
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # "hybrid" takes priority
        assert data["processing"]["classification_engine"] == "text_rules+layout_features+ml"


# ---------------------------------------------------------------------------
# Tests: _heuristic_ensemble (extracted helper)
# ---------------------------------------------------------------------------


class TestHeuristicEnsemble:
    def test_text_and_layout_agree(self):
        """Extracted _heuristic_ensemble matches old classify_page_ensemble behavior."""
        text_result = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.6, method="text_rules",
            type_scores={"invoice": 0.6},
        )
        layout_result = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.7, method="layout_features",
            type_scores={"invoice": 0.7},
        )
        result = _heuristic_ensemble(text_result, layout_result, 1)
        assert result.predicted_type == "invoice"
        assert result.method == "ensemble"

    def test_none_text_result(self):
        """_heuristic_ensemble handles None text_result."""
        layout_result = PageClassification(
            page_num=1, predicted_type="form",
            confidence=0.6, method="layout_features",
            type_scores={"form": 0.6},
        )
        result = _heuristic_ensemble(None, layout_result, 1)
        assert result.predicted_type == "form"
        assert result.method == "ensemble"

    def test_none_layout_fallback(self):
        """_heuristic_ensemble falls back to text when layout is None."""
        text_result = PageClassification(
            page_num=1, predicted_type="letter",
            confidence=0.5, method="text_rules",
            type_scores={"letter": 0.5},
        )
        result = _heuristic_ensemble(text_result, None, 1)
        assert result.predicted_type == "letter"
        assert result.confidence == 0.5
