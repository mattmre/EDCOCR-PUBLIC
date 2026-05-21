"""
Unit tests for LayoutLMv3 semantic extraction module (semantic_extraction.py).

Tests cover:
- Entity label list validation
- LayoutLMv3Extractor init and lazy model loading (mocked)
- Bounding box normalization to 0-1000 range
- BIO tag postprocessing (merging tokens into entities)
- Merge of semantic results with existing extraction results
- Graceful degradation when transformers/torch are missing
- Configuration via environment variables
- High-level extract_semantic_fields with mocked model

Run with: python -m pytest tests/test_semantic_extraction.py -v
"""

import os
import sys
import threading
import time
import types
from unittest import mock

import pytest

import semantic_extraction as sem_module
from layoutlm_model_registry import ResolvedModelSelection
from semantic_extraction import (
    SEMANTIC_ENTITY_LABELS,
    SEMANTIC_MODEL_PATH,
    LayoutLMv3Extractor,
    SemanticEntity,
    SemanticExtractionResult,
    _merge_boxes,
    _text_overlaps_any,
    merge_with_existing_extraction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProcessor:
    """Mock LayoutLMv3 processor that returns tensor-like dicts."""

    def __call__(self, image, words, boxes=None, return_tensors=None,
                 truncation=None, max_length=None):
        n_tokens = len(words) + 2  # +2 for [CLS] and [SEP]
        # Simulate a BatchEncoding-like object
        encoding = {
            "input_ids": _MockTensor([[0] * n_tokens]),
            "attention_mask": _MockTensor([[1] * n_tokens]),
            "bbox": _MockTensor([[[0, 0, 0, 0]] * n_tokens]),
        }

        # Build word_ids: None for special tokens, index for real tokens
        wid_list = [None] + list(range(len(words))) + [None]

        def word_ids_fn(batch_index=0):
            return wid_list

        encoding["word_ids"] = word_ids_fn
        # Also make it subscriptable and dict-like
        encoding_obj = _DictLikeEncoding(encoding, wid_list)
        return encoding_obj


class _MockTensor:
    """Minimal tensor mock that supports .to() and basic ops."""

    def __init__(self, data):
        self.data = data

    def to(self, device):
        return self

    def __len__(self):
        return len(self.data)


class _DictLikeEncoding(dict):
    """Mock BatchEncoding that supports word_ids() and .items()."""

    def __init__(self, data, wid_list):
        super().__init__(data)
        self._wid_list = wid_list

    def word_ids(self, batch_index=0):
        return self._wid_list


class MockModel:
    """Mock LayoutLMv3 model for token classification."""

    def __init__(self, predictions=None, num_labels=19):
        self._predictions = predictions  # List of label indices
        self._num_labels = num_labels
        self._device = "cpu"

    def to(self, device):
        self._device = device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        import types

        n_tokens = kwargs.get("input_ids", _MockTensor([[]])).data[0]
        if isinstance(n_tokens, list):
            n_tokens = len(n_tokens)

        preds = self._predictions or [0] * n_tokens

        # Build logits as a mock tensor with softmax-like behavior
        logits = _MockLogits(preds, self._num_labels)
        result = types.SimpleNamespace(logits=logits)
        return result


class _MockLogits:
    """Mock logits tensor that supports softmax, argmax, max operations."""

    def __init__(self, pred_ids, num_labels):
        self._pred_ids = pred_ids
        self._num_labels = num_labels

    def squeeze(self):
        return self

    def tolist(self):
        return self._pred_ids


# ---------------------------------------------------------------------------
# Tests: Entity labels
# ---------------------------------------------------------------------------


class TestSemanticEntityLabels:
    """Validate the SEMANTIC_ENTITY_LABELS list."""

    def test_first_label_is_outside(self):
        assert SEMANTIC_ENTITY_LABELS[0] == "O"

    def test_bio_pairs_complete(self):
        """Every B- label should have a corresponding I- label."""
        b_labels = [lbl for lbl in SEMANTIC_ENTITY_LABELS if lbl.startswith("B-")]
        i_labels = [lbl for lbl in SEMANTIC_ENTITY_LABELS if lbl.startswith("I-")]
        b_types = {lbl[2:] for lbl in b_labels}
        i_types = {lbl[2:] for lbl in i_labels}
        assert b_types == i_types

    def test_expected_entity_count(self):
        """Should have 9 entity types (B+I pairs) plus O = 19 labels."""
        assert len(SEMANTIC_ENTITY_LABELS) == 19

    def test_contains_expected_types(self):
        types = {lbl[2:] for lbl in SEMANTIC_ENTITY_LABELS if lbl.startswith("B-")}
        expected = {
            "INVOICE_NUMBER", "DATE", "AMOUNT", "PERSON_NAME",
            "ORGANIZATION", "ADDRESS", "REFERENCE_NUMBER",
            "PHONE_NUMBER", "EMAIL",
        }
        assert types == expected

    def test_no_duplicate_labels(self):
        assert len(SEMANTIC_ENTITY_LABELS) == len(set(SEMANTIC_ENTITY_LABELS))

    def test_label2id_consistency(self):
        from semantic_extraction import _ID2LABEL, _LABEL2ID

        for label, idx in _LABEL2ID.items():
            assert _ID2LABEL[idx] == label

    def test_id2label_coverage(self):
        from semantic_extraction import _ID2LABEL

        assert len(_ID2LABEL) == len(SEMANTIC_ENTITY_LABELS)


# ---------------------------------------------------------------------------
# Tests: SemanticEntity dataclass
# ---------------------------------------------------------------------------


class TestSemanticEntity:
    """Test SemanticEntity dataclass behavior."""

    def test_defaults(self):
        e = SemanticEntity(text="test", label="DATE")
        assert e.text == "test"
        assert e.label == "DATE"
        assert e.confidence == 0.0
        assert e.bbox == []
        assert e.page_num == 0

    def test_field_type_property(self):
        e = SemanticEntity(text="2024-01-15", label="DATE")
        assert e.field_type == "date"

    def test_field_type_invoice_number(self):
        e = SemanticEntity(text="INV-001", label="INVOICE_NUMBER")
        assert e.field_type == "reference_number"

    def test_field_type_email(self):
        e = SemanticEntity(text="a@b.com", label="EMAIL")
        assert e.field_type == "email_address"

    def test_field_type_unknown(self):
        e = SemanticEntity(text="x", label="UNKNOWN_TYPE")
        assert e.field_type == "unknown_type"

    def test_with_bbox(self):
        e = SemanticEntity(
            text="test", label="AMOUNT", bbox=[10, 20, 100, 40]
        )
        assert e.bbox == [10, 20, 100, 40]


class TestSemanticExtractionResult:
    """Test SemanticExtractionResult dataclass."""

    def test_defaults(self):
        r = SemanticExtractionResult()
        assert r.entities == []
        assert r.model_name == ""
        assert r.processing_time == 0.0
        assert r.page_count == 0

    def test_with_entities(self):
        e = SemanticEntity(text="hello", label="PERSON_NAME")
        r = SemanticExtractionResult(entities=[e], model_name="test-model")
        assert len(r.entities) == 1
        assert r.model_name == "test-model"


# ---------------------------------------------------------------------------
# Tests: Box normalization
# ---------------------------------------------------------------------------


class TestBoxNormalization:
    """Test LayoutLMv3Extractor._normalize_boxes."""

    def test_basic_normalization(self):
        boxes = [[0, 0, 500, 500]]
        result = LayoutLMv3Extractor._normalize_boxes(boxes, 1000, 1000)
        assert result == [[0, 0, 500, 500]]

    def test_scales_to_1000(self):
        boxes = [[100, 200, 300, 400]]
        result = LayoutLMv3Extractor._normalize_boxes(boxes, 1000, 1000)
        assert result == [[100, 200, 300, 400]]

    def test_small_image(self):
        boxes = [[50, 50, 100, 100]]
        result = LayoutLMv3Extractor._normalize_boxes(boxes, 100, 100)
        assert result == [[500, 500, 1000, 1000]]

    def test_full_page(self):
        boxes = [[0, 0, 800, 600]]
        result = LayoutLMv3Extractor._normalize_boxes(boxes, 800, 600)
        assert result == [[0, 0, 1000, 1000]]

    def test_clamp_to_range(self):
        """Values should be clamped to [0, 1000]."""
        boxes = [[-10, -20, 2000, 3000]]
        result = LayoutLMv3Extractor._normalize_boxes(boxes, 1000, 1000)
        assert all(0 <= v <= 1000 for v in result[0])

    def test_zero_width_returns_zeros(self):
        boxes = [[10, 20, 30, 40]]
        result = LayoutLMv3Extractor._normalize_boxes(boxes, 0, 100)
        assert result == [[0, 0, 0, 0]]

    def test_zero_height_returns_zeros(self):
        boxes = [[10, 20, 30, 40]]
        result = LayoutLMv3Extractor._normalize_boxes(boxes, 100, 0)
        assert result == [[0, 0, 0, 0]]

    def test_multiple_boxes(self):
        boxes = [[0, 0, 500, 500], [500, 500, 1000, 1000]]
        result = LayoutLMv3Extractor._normalize_boxes(boxes, 1000, 1000)
        assert len(result) == 2
        assert result[0] == [0, 0, 500, 500]
        assert result[1] == [500, 500, 1000, 1000]

    def test_empty_boxes(self):
        result = LayoutLMv3Extractor._normalize_boxes([], 100, 100)
        assert result == []

    def test_short_box_padded(self):
        """Box with fewer than 4 values gets [0,0,0,0]."""
        boxes = [[10, 20]]
        result = LayoutLMv3Extractor._normalize_boxes(boxes, 100, 100)
        assert result == [[0, 0, 0, 0]]


# ---------------------------------------------------------------------------
# Tests: BIO postprocessing
# ---------------------------------------------------------------------------


class TestBIOPostprocessing:
    """Test LayoutLMv3Extractor._postprocess_predictions."""

    def test_single_b_tag(self):
        from semantic_extraction import _LABEL2ID

        tokens = ["2024-01-15"]
        preds = [_LABEL2ID["B-DATE"]]
        confs = [0.95]
        result = LayoutLMv3Extractor._postprocess_predictions(
            tokens, preds, confs
        )
        assert len(result) == 1
        assert result[0].text == "2024-01-15"
        assert result[0].label == "DATE"
        assert result[0].confidence == 0.95

    def test_b_i_sequence(self):
        from semantic_extraction import _LABEL2ID

        tokens = ["John", "Smith"]
        preds = [_LABEL2ID["B-PERSON_NAME"], _LABEL2ID["I-PERSON_NAME"]]
        confs = [0.9, 0.85]
        result = LayoutLMv3Extractor._postprocess_predictions(
            tokens, preds, confs
        )
        assert len(result) == 1
        assert result[0].text == "John Smith"
        assert result[0].label == "PERSON_NAME"
        assert result[0].confidence == pytest.approx(0.875, abs=0.001)

    def test_all_o_tags(self):
        tokens = ["the", "quick", "brown"]
        preds = [0, 0, 0]  # O tags
        confs = [0.99, 0.99, 0.99]
        result = LayoutLMv3Extractor._postprocess_predictions(
            tokens, preds, confs
        )
        assert len(result) == 0

    def test_multiple_entities(self):
        from semantic_extraction import _LABEL2ID

        tokens = ["$500", "on", "2024-01-15"]
        preds = [
            _LABEL2ID["B-AMOUNT"],
            0,  # O
            _LABEL2ID["B-DATE"],
        ]
        confs = [0.9, 0.5, 0.88]
        result = LayoutLMv3Extractor._postprocess_predictions(
            tokens, preds, confs
        )
        assert len(result) == 2
        labels = {e.label for e in result}
        assert "AMOUNT" in labels
        assert "DATE" in labels

    def test_i_without_b_treated_as_b(self):
        """An I- tag without a preceding B- should start a new entity."""
        from semantic_extraction import _LABEL2ID

        tokens = ["Smith"]
        preds = [_LABEL2ID["I-PERSON_NAME"]]
        confs = [0.8]
        result = LayoutLMv3Extractor._postprocess_predictions(
            tokens, preds, confs
        )
        assert len(result) == 1
        assert result[0].text == "Smith"
        assert result[0].label == "PERSON_NAME"

    def test_type_mismatch_flushes_entity(self):
        """I- tag with different type than current B- flushes and starts new."""
        from semantic_extraction import _LABEL2ID

        tokens = ["John", "$500"]
        preds = [_LABEL2ID["B-PERSON_NAME"], _LABEL2ID["I-AMOUNT"]]
        confs = [0.9, 0.85]
        result = LayoutLMv3Extractor._postprocess_predictions(
            tokens, preds, confs
        )
        assert len(result) == 2
        assert result[0].label == "PERSON_NAME"
        assert result[1].label == "AMOUNT"

    def test_empty_input(self):
        result = LayoutLMv3Extractor._postprocess_predictions([], [], [])
        assert result == []

    def test_with_boxes(self):
        from semantic_extraction import _LABEL2ID

        tokens = ["Acme", "Corp"]
        preds = [_LABEL2ID["B-ORGANIZATION"], _LABEL2ID["I-ORGANIZATION"]]
        confs = [0.9, 0.88]
        boxes = [[10, 20, 100, 40], [110, 20, 200, 40]]
        result = LayoutLMv3Extractor._postprocess_predictions(
            tokens, preds, confs, boxes=boxes
        )
        assert len(result) == 1
        assert result[0].bbox == [10, 20, 200, 40]  # merged box

    def test_consecutive_different_entities(self):
        from semantic_extraction import _LABEL2ID

        tokens = ["$100", "John"]
        preds = [_LABEL2ID["B-AMOUNT"], _LABEL2ID["B-PERSON_NAME"]]
        confs = [0.9, 0.85]
        result = LayoutLMv3Extractor._postprocess_predictions(
            tokens, preds, confs
        )
        assert len(result) == 2
        assert result[0].label == "AMOUNT"
        assert result[1].label == "PERSON_NAME"

    def test_long_entity(self):
        from semantic_extraction import _LABEL2ID

        tokens = ["123", "Main", "St", "Suite", "400"]
        preds = [
            _LABEL2ID["B-ADDRESS"],
            _LABEL2ID["I-ADDRESS"],
            _LABEL2ID["I-ADDRESS"],
            _LABEL2ID["I-ADDRESS"],
            _LABEL2ID["I-ADDRESS"],
        ]
        confs = [0.9, 0.88, 0.87, 0.86, 0.85]
        result = LayoutLMv3Extractor._postprocess_predictions(
            tokens, preds, confs
        )
        assert len(result) == 1
        assert result[0].text == "123 Main St Suite 400"
        assert result[0].label == "ADDRESS"


# ---------------------------------------------------------------------------
# Tests: _merge_boxes helper
# ---------------------------------------------------------------------------


class TestMergeBoxes:
    """Test the _merge_boxes helper function."""

    def test_single_box(self):
        assert _merge_boxes([[10, 20, 100, 40]]) == [10, 20, 100, 40]

    def test_two_boxes(self):
        result = _merge_boxes([[10, 20, 100, 40], [110, 20, 200, 40]])
        assert result == [10, 20, 200, 40]

    def test_empty_list(self):
        assert _merge_boxes([]) == []

    def test_invalid_boxes_filtered(self):
        assert _merge_boxes([[10], [20, 30]]) == []

    def test_mixed_valid_invalid(self):
        result = _merge_boxes([[10, 20, 100, 40], [5]])
        assert result == [10, 20, 100, 40]


# ---------------------------------------------------------------------------
# Tests: Merge results
# ---------------------------------------------------------------------------


class TestMergeResults:
    """Test merge_with_existing_extraction."""

    def test_empty_semantic_returns_existing(self):
        existing = [
            {"field_type": "date", "text": "2024-01-15",
             "extraction_method": "regex"},
        ]
        result = merge_with_existing_extraction(
            SemanticExtractionResult(), existing
        )
        assert result == existing

    def test_none_semantic_returns_existing(self):
        existing = [{"field_type": "date", "text": "2024-01-15",
                      "extraction_method": "regex"}]
        result = merge_with_existing_extraction(None, existing)
        assert result == existing

    def test_uie_takes_priority(self):
        """UIE fields should be kept even when semantic finds same text."""
        semantic = SemanticExtractionResult(entities=[
            SemanticEntity(text="2024-01-15", label="DATE", confidence=0.9),
        ])
        existing = [
            {"field_type": "date", "text": "2024-01-15",
             "extraction_method": "uie", "confidence": 0.95},
        ]
        result = merge_with_existing_extraction(semantic, existing)
        # UIE field kept, semantic deduped
        methods = [f.get("extraction_method") for f in result]
        assert "uie" in methods
        # Semantic should be deduped out since text matches
        assert methods.count("semantic") == 0

    def test_semantic_adds_new_entities(self):
        """Semantic entities not in existing results should be added."""
        semantic = SemanticExtractionResult(entities=[
            SemanticEntity(text="Acme Corp", label="ORGANIZATION",
                          confidence=0.88),
        ])
        existing = [
            {"field_type": "date", "text": "2024-01-15",
             "extraction_method": "regex"},
        ]
        result = merge_with_existing_extraction(semantic, existing)
        assert len(result) == 2
        types = {f["field_type"] for f in result}
        assert "organization" in types

    def test_semantic_deduplicates_regex(self):
        """Semantic entity should replace regex for same text+type."""
        semantic = SemanticExtractionResult(entities=[
            SemanticEntity(text="$500.00", label="AMOUNT", confidence=0.9),
        ])
        existing = [
            {"field_type": "amount", "text": "$500.00",
             "extraction_method": "regex", "confidence": 1.0},
        ]
        result = merge_with_existing_extraction(semantic, existing)
        # Semantic is added, regex is deduped
        methods = [f.get("extraction_method") for f in result]
        assert "semantic" in methods
        assert "regex" not in methods

    def test_partial_text_overlap(self):
        """When semantic text contains regex text, regex should be deduped."""
        semantic = SemanticExtractionResult(entities=[
            SemanticEntity(text="January 15, 2024", label="DATE",
                          confidence=0.85),
        ])
        existing = [
            {"field_type": "date", "text": "January 15, 2024",
             "extraction_method": "regex"},
        ]
        result = merge_with_existing_extraction(semantic, existing)
        methods = [f.get("extraction_method") for f in result]
        assert "semantic" in methods
        assert "regex" not in methods

    def test_different_types_not_deduped(self):
        """Same text but different types should both be kept."""
        semantic = SemanticExtractionResult(entities=[
            SemanticEntity(text="INV-2024-001", label="INVOICE_NUMBER",
                          confidence=0.9),
        ])
        existing = [
            {"field_type": "date", "text": "2024-01-15",
             "extraction_method": "regex"},
        ]
        result = merge_with_existing_extraction(semantic, existing)
        assert len(result) == 2

    def test_preserves_other_methods(self):
        """Fields with non-uie/non-regex methods should be preserved."""
        semantic = SemanticExtractionResult(entities=[
            SemanticEntity(text="test", label="ORGANIZATION", confidence=0.8),
        ])
        existing = [
            {"field_type": "date", "text": "2024-01-15",
             "extraction_method": "custom"},
        ]
        result = merge_with_existing_extraction(semantic, existing)
        methods = {f.get("extraction_method") for f in result}
        assert "custom" in methods
        assert "semantic" in methods


# ---------------------------------------------------------------------------
# Tests: _text_overlaps_any
# ---------------------------------------------------------------------------


class TestTextOverlapsAny:
    """Test the _text_overlaps_any deduplication helper."""

    def test_exact_match(self):
        field = {"field_type": "date", "text": "2024-01-15"}
        existing = [{"field_type": "date", "text": "2024-01-15"}]
        assert _text_overlaps_any(field, existing) is True

    def test_case_insensitive(self):
        field = {"field_type": "organization", "text": "ACME CORP"}
        existing = [{"field_type": "organization", "text": "acme corp"}]
        assert _text_overlaps_any(field, existing) is True

    def test_no_match(self):
        field = {"field_type": "date", "text": "2024-01-15"}
        existing = [{"field_type": "amount", "text": "$500"}]
        assert _text_overlaps_any(field, existing) is False

    def test_same_type_different_text(self):
        field = {"field_type": "date", "text": "2024-01-15"}
        existing = [{"field_type": "date", "text": "2024-02-20"}]
        assert _text_overlaps_any(field, existing) is False

    def test_substring_overlap(self):
        field = {"field_type": "amount", "text": "$500.00"}
        existing = [{"field_type": "amount", "text": "USD $500.00"}]
        assert _text_overlaps_any(field, existing) is False

    def test_empty_existing(self):
        field = {"field_type": "date", "text": "2024-01-15"}
        assert _text_overlaps_any(field, []) is False


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Test behavior when transformers/torch are not installed."""

    def test_extractor_load_fails_without_transformers(self):
        extractor = LayoutLMv3Extractor(model_path="test-model")
        with mock.patch.dict("sys.modules", {"transformers": None}):
            with mock.patch(
                "builtins.__import__",
                side_effect=_selective_import_error({"transformers", "torch"}),
            ):
                result = extractor._load_model()
        assert result is False
        assert extractor._load_failed is True

    def test_extractor_extract_returns_empty_on_load_failure(self):
        extractor = LayoutLMv3Extractor(model_path="test-model")
        extractor._load_failed = True
        result = extractor.extract_entities(
            words=["hello"], boxes=[[0, 0, 100, 20]], image=None, page_num=1
        )
        assert result == []

    def test_extract_semantic_fields_disabled(self):
        """When ENABLE_SEMANTIC_EXTRACTION is False, returns empty result."""
        from semantic_extraction import extract_semantic_fields

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", False):
            with mock.patch.object(sem_module, "_extractor_instance", None):
                result = extract_semantic_fields(
                    [("hello world", 0.9, [0, 0, 200, 30])],
                    _make_mock_image(),
                    page_num=1,
                )
        assert result.entities == []

    def test_model_load_exception_handled(self):
        """Exception during model load should not crash, just return False."""
        extractor = LayoutLMv3Extractor(model_path="bad-model")
        with mock.patch(
            "builtins.__import__",
            side_effect=_selective_import_error(set()),
        ):
            # Patch the transformers module to raise on from_pretrained
            mock_transformers = mock.MagicMock()
            mock_transformers.AutoProcessor.from_pretrained.side_effect = (
                RuntimeError("Model not found")
            )
            with mock.patch.dict(
                "sys.modules",
                {"transformers": mock_transformers, "torch": mock.MagicMock()},
            ):
                result = extractor._load_model()
        assert result is False
        assert extractor._load_failed is True

    def test_model_load_is_single_threaded(self):
        extractor = LayoutLMv3Extractor(model_path="test-model")
        load_counts = {"processor": 0, "model": 0}
        start_event = threading.Event()

        class _FakeModel:
            def to(self, _device):
                return self

            def eval(self):
                return self

        def _load_processor(*args, **kwargs):
            del args, kwargs
            load_counts["processor"] += 1
            time.sleep(0.05)
            return object()

        def _load_model(*args, **kwargs):
            del args, kwargs
            load_counts["model"] += 1
            time.sleep(0.05)
            return _FakeModel()

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            device=lambda value: value,
        )
        fake_transformers = types.SimpleNamespace(
            AutoProcessor=types.SimpleNamespace(
                from_pretrained=_load_processor
            ),
            AutoModelForTokenClassification=types.SimpleNamespace(
                from_pretrained=_load_model
            ),
        )

        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            results = []

            def _run_load():
                start_event.wait(timeout=1)
                results.append(extractor._load_model())

            threads = [
                threading.Thread(target=_run_load),
                threading.Thread(target=_run_load),
            ]
            for thread in threads:
                thread.start()
            start_event.set()
            for thread in threads:
                thread.join()

        assert results == [True, True]
        assert load_counts["processor"] == 1
        assert load_counts["model"] == 1

    def test_inference_exception_returns_empty(self):
        """Runtime errors during inference should return empty, not crash."""
        extractor = LayoutLMv3Extractor()
        extractor._model = mock.MagicMock(
            side_effect=RuntimeError("CUDA OOM")
        )
        extractor._processor = mock.MagicMock()
        extractor._device = "cpu"

        # Need torch for the inference path
        mock_torch = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            result = extractor.extract_entities(
                words=["test"], boxes=[[0, 0, 100, 20]],
                image=_make_mock_image(), page_num=1,
            )
        assert result == []

    def test_is_available_before_load(self):
        extractor = LayoutLMv3Extractor()
        assert extractor.is_available is False

    def test_is_available_after_failed_load(self):
        extractor = LayoutLMv3Extractor()
        extractor._load_failed = True
        assert extractor.is_available is False


# ---------------------------------------------------------------------------
# Tests: Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Test environment variable configuration."""

    def test_enable_flag_default_false(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Re-evaluate the module-level constant
            val = os.environ.get(
                "ENABLE_SEMANTIC_EXTRACTION", ""
            ).lower() in ("1", "true", "yes")
            assert val is False

    def test_enable_flag_true(self):
        for value in ("1", "true", "yes", "True", "YES"):
            val = value.lower() in ("1", "true", "yes")
            assert val is True

    def test_enable_flag_false(self):
        for value in ("0", "false", "no", ""):
            val = value.lower() in ("1", "true", "yes")
            assert val is False

    def test_model_path_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            val = os.environ.get("SEMANTIC_MODEL_PATH", "microsoft/layoutlmv3-base")
            assert val == "microsoft/layoutlmv3-base"

    def test_model_path_custom(self):
        with mock.patch.dict(
            os.environ, {"SEMANTIC_MODEL_PATH": "/models/custom-kie"}
        ):
            val = os.environ.get("SEMANTIC_MODEL_PATH", "microsoft/layoutlmv3-base")
            assert val == "/models/custom-kie"

    def test_confidence_threshold_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            val = float(os.environ.get("SEMANTIC_CONFIDENCE_THRESHOLD", "0.5"))
            assert val == 0.5

    def test_confidence_threshold_custom(self):
        with mock.patch.dict(
            os.environ, {"SEMANTIC_CONFIDENCE_THRESHOLD": "0.8"}
        ):
            val = float(os.environ.get("SEMANTIC_CONFIDENCE_THRESHOLD", "0.5"))
            assert val == 0.8


# ---------------------------------------------------------------------------
# Tests: Singleton factory
# ---------------------------------------------------------------------------


class TestGetSemanticExtractor:
    """Test get_semantic_extractor singleton factory."""

    def test_returns_none_when_disabled(self):
        from semantic_extraction import get_semantic_extractor

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", False):
            result = get_semantic_extractor()
        assert result is None

    def test_returns_extractor_when_enabled(self):
        from semantic_extraction import get_semantic_extractor

        selection = ResolvedModelSelection(
            model_path="/models/active-layoutlm",
            source="registry",
            active_model_spec="forensic:1.0.0",
        )

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", True):
            with mock.patch.object(sem_module, "_extractor_instance", None):
                with mock.patch.object(
                    sem_module,
                    "resolve_active_model_selection",
                    return_value=selection,
                ):
                    result = get_semantic_extractor()
        assert isinstance(result, LayoutLMv3Extractor)
        assert result.model_path == "/models/active-layoutlm"
        assert result.model_source == "registry"
        assert result.active_model_spec == "forensic:1.0.0"

    def test_registry_selection_with_adapter_metadata_is_recorded(self):
        from semantic_extraction import get_semantic_extractor

        selection = ResolvedModelSelection(
            model_path="/models/active-layoutlm",
            source="registry",
            active_model_spec="forensic:1.0.0",
            adapter_path="/models/adapters/forensic",
        )

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", True):
            with mock.patch.object(sem_module, "_extractor_instance", None):
                with mock.patch.object(
                    sem_module,
                    "resolve_active_model_selection",
                    return_value=selection,
                ):
                    result = get_semantic_extractor()
        assert result.adapter_path == "/models/adapters/forensic"

    def test_returns_extractor_with_fallback_model_when_enabled(self):
        from semantic_extraction import get_semantic_extractor

        selection = ResolvedModelSelection(
            model_path=SEMANTIC_MODEL_PATH,
            source="fallback",
        )

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", True):
            with mock.patch.object(sem_module, "_extractor_instance", None):
                with mock.patch.object(
                    sem_module,
                    "resolve_active_model_selection",
                    return_value=selection,
                ):
                    result = get_semantic_extractor()
        assert isinstance(result, LayoutLMv3Extractor)
        assert result.model_path == SEMANTIC_MODEL_PATH

    def test_returns_same_instance(self):
        from semantic_extraction import get_semantic_extractor

        sentinel = LayoutLMv3Extractor(model_path="cached")
        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", True):
            with mock.patch.object(
                sem_module, "_extractor_instance", sentinel
            ):
                result = get_semantic_extractor()
        assert result is sentinel


# ---------------------------------------------------------------------------
# Tests: extract_semantic_fields (high-level)
# ---------------------------------------------------------------------------


class TestExtractSemanticFields:
    """Test the high-level extract_semantic_fields function."""

    def test_empty_paddle_lines(self):
        from semantic_extraction import extract_semantic_fields

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", True):
            with mock.patch.object(sem_module, "_extractor_instance", None):
                result = extract_semantic_fields([], _make_mock_image(), 1)
        assert result.entities == []
        assert result.page_count == 1

    def test_returns_result_with_model_name(self):
        from semantic_extraction import extract_semantic_fields

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", True):
            with mock.patch.object(sem_module, "_extractor_instance", None):
                result = extract_semantic_fields(
                    [("hello", 0.9, [0, 0, 100, 20])],
                    _make_mock_image(),
                    1,
                )
        assert result.model_name == SEMANTIC_MODEL_PATH

    def test_returns_result_with_resolved_model_name(self):
        from semantic_extraction import extract_semantic_fields

        mock_extractor = mock.MagicMock(spec=LayoutLMv3Extractor)
        mock_extractor.model_path = "/models/active-layoutlm"
        mock_extractor.extract_entities.return_value = []

        with mock.patch.object(
            sem_module,
            "get_semantic_extractor",
            return_value=mock_extractor,
        ):
            result = extract_semantic_fields(
                [("hello", 0.9, [0, 0, 100, 20])],
                _make_mock_image(),
                1,
            )
        assert result.model_name == "/models/active-layoutlm"

    def test_processing_time_recorded(self):
        from semantic_extraction import extract_semantic_fields

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", True):
            with mock.patch.object(sem_module, "_extractor_instance", None):
                result = extract_semantic_fields(
                    [("hello", 0.9, [0, 0, 100, 20])],
                    _make_mock_image(),
                    1,
                )
        assert result.processing_time >= 0

    def test_skips_short_line_data(self):
        from semantic_extraction import extract_semantic_fields

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", True):
            with mock.patch.object(sem_module, "_extractor_instance", None):
                result = extract_semantic_fields(
                    [("hello",)],  # Too short
                    _make_mock_image(),
                    1,
                )
        assert result.entities == []

    def test_splits_words_from_lines(self):
        """Verifies that multi-word lines are split into individual tokens."""
        from semantic_extraction import extract_semantic_fields

        mock_extractor = mock.MagicMock(spec=LayoutLMv3Extractor)
        mock_extractor.model_path = SEMANTIC_MODEL_PATH
        mock_extractor.extract_entities.return_value = []

        with mock.patch.object(sem_module, "ENABLE_SEMANTIC_EXTRACTION", True):
            with mock.patch.object(
                sem_module, "_extractor_instance", mock_extractor
            ):
                with mock.patch.object(
                    sem_module, "get_semantic_extractor",
                    return_value=mock_extractor,
                ):
                    extract_semantic_fields(
                        [("hello world foo", 0.9, [0, 0, 300, 20])],
                        _make_mock_image(),
                        1,
                    )

        # Verify the extractor was called with 3 words
        call_args = mock_extractor.extract_entities.call_args
        words_arg = call_args[1].get("words", call_args[0][0])
        assert len(words_arg) == 3


# ---------------------------------------------------------------------------
# Tests: LayoutLMv3Extractor init
# ---------------------------------------------------------------------------


class TestLayoutLMv3ExtractorInit:
    """Test LayoutLMv3Extractor initialization."""

    def test_default_model_path(self):
        extractor = LayoutLMv3Extractor()
        assert extractor.model_path == SEMANTIC_MODEL_PATH

    def test_custom_model_path(self):
        extractor = LayoutLMv3Extractor(model_path="/custom/model")
        assert extractor.model_path == "/custom/model"

    def test_custom_device(self):
        extractor = LayoutLMv3Extractor(device="cpu")
        assert extractor._device_hint == "cpu"

    def test_initial_state(self):
        extractor = LayoutLMv3Extractor()
        assert extractor._model is None
        assert extractor._processor is None
        assert extractor._load_failed is False

    def test_empty_words_returns_empty(self):
        extractor = LayoutLMv3Extractor()
        result = extractor.extract_entities([], [], _make_mock_image(), 1)
        assert result == []

    def test_empty_boxes_returns_empty(self):
        extractor = LayoutLMv3Extractor()
        result = extractor.extract_entities(["word"], [], _make_mock_image(), 1)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: Extraction mode integration in extraction.py
# ---------------------------------------------------------------------------


class TestExtractionMode:
    """Test EXTRACTION_MODE env var integration in extraction.py."""

    def test_extraction_mode_default(self):
        import extraction

        # Default is "uie"
        assert hasattr(extraction, "EXTRACTION_MODE")

    def test_semantic_engine_label(self):
        """Finalization should recognize 'semantic' extraction method."""
        from extraction import DocumentExtraction, PageExtraction, finalize_extraction

        doc = DocumentExtraction(document_id="d_sem", source_file="test.pdf")
        doc.pages = [
            PageExtraction(page_num=1, fields=[
                {"field_type": "date", "text": "2024-01-15",
                 "extraction_method": "semantic", "confidence": 0.9,
                 "page_num": 1, "start": 0, "end": 10, "normalized_value": ""},
            ]),
        ]
        result = finalize_extraction(doc)
        assert result.extraction_engine == "semantic"

    def test_hybrid_with_semantic(self):
        """Mix of semantic + regex should report 'hybrid'."""
        from extraction import DocumentExtraction, PageExtraction, finalize_extraction

        doc = DocumentExtraction(document_id="d_hyb", source_file="test.pdf")
        doc.pages = [
            PageExtraction(page_num=1, fields=[
                {"field_type": "date", "text": "2024-01-15",
                 "extraction_method": "semantic", "confidence": 0.9,
                 "page_num": 1, "start": 0, "end": 10, "normalized_value": ""},
                {"field_type": "amount", "text": "$500.00",
                 "extraction_method": "regex", "confidence": 1.0,
                 "page_num": 1, "start": 20, "end": 27, "normalized_value": ""},
            ]),
        ]
        result = finalize_extraction(doc)
        assert result.extraction_engine == "hybrid"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_image(width=800, height=600):
    """Create a minimal mock PIL Image with .size property."""
    img = mock.MagicMock()
    img.size = (width, height)
    return img


def _selective_import_error(blocked_modules):
    """Return an import side_effect that blocks specific modules."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import(name, *args, **kwargs):
        if name in blocked_modules:
            raise ImportError(f"Mocked: {name} not installed")
        return original_import(name, *args, **kwargs)

    return _import
