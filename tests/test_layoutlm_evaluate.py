"""Tests for layoutlm_evaluate — Evaluation & reporting for LayoutLMv3.

Covers evaluate_model (mocked ML), evaluate_predictions (mocked seqeval),
empty-report generation, and report file writing. All tests run WITHOUT
torch/transformers/seqeval installed.

Run with: python -m pytest tests/test_layoutlm_evaluate.py -v
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from layoutlm_evaluate import _empty_report, _write_report
from layoutlm_labels import build_label_set

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_label_set():
    """Minimal label set for focused tests."""
    return build_label_set("small", ["DATE", "AMOUNT"])


@pytest.fixture
def default_label_set():
    """Default 9-entity label set."""
    return build_label_set("default", [
        "INVOICE_NUMBER", "DATE", "AMOUNT", "PERSON_NAME",
        "ORGANIZATION", "ADDRESS", "REFERENCE_NUMBER",
        "PHONE_NUMBER", "EMAIL",
    ])


# ---------------------------------------------------------------------------
# _empty_report tests
# ---------------------------------------------------------------------------


class TestEmptyReport:
    """Tests for the _empty_report helper."""

    def test_structure(self, small_label_set):
        """Empty report has all required top-level keys."""
        report = _empty_report(small_label_set, "/models/test")
        assert "run_id" in report
        assert "timestamp" in report
        assert "overall" in report
        assert "per_entity" in report
        assert "metadata" in report

    def test_zero_metrics(self, small_label_set):
        """All metrics in an empty report are zero."""
        report = _empty_report(small_label_set, "/models/test")
        assert report["overall"]["f1"] == 0.0
        assert report["overall"]["precision"] == 0.0
        assert report["overall"]["recall"] == 0.0
        assert report["num_pages_evaluated"] == 0

    def test_per_entity_coverage(self, small_label_set):
        """Every entity type appears in per_entity breakdown."""
        report = _empty_report(small_label_set, "")
        assert "DATE" in report["per_entity"]
        assert "AMOUNT" in report["per_entity"]
        assert len(report["per_entity"]) == 2

    def test_metadata_fields(self, small_label_set):
        """Metadata declares correct evaluation type and tagging scheme."""
        report = _empty_report(small_label_set, "")
        assert report["metadata"]["evaluation_type"] == "token_classification"
        assert report["metadata"]["tagging_scheme"] == "BIO"
        assert report["metadata"]["metric_library"] == "seqeval"


# ---------------------------------------------------------------------------
# _write_report tests
# ---------------------------------------------------------------------------


class TestWriteReport:
    """Tests for report file writing."""

    def test_write_creates_file(self, tmp_path, small_label_set):
        """Report is written as valid JSON."""
        report = _empty_report(small_label_set, "")
        out_path = str(tmp_path / "report.json")
        _write_report(report, out_path)
        assert os.path.isfile(out_path)

        with open(out_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded["label_set"] == "small"

    def test_write_creates_parent_dirs(self, tmp_path, small_label_set):
        """Parent directories are created if they don't exist."""
        report = _empty_report(small_label_set, "")
        out_path = str(tmp_path / "sub" / "dir" / "report.json")
        _write_report(report, out_path)
        assert os.path.isfile(out_path)


# ---------------------------------------------------------------------------
# evaluate_model import guard tests
# ---------------------------------------------------------------------------


class TestEvaluateModelImports:
    """Tests that evaluate_model raises ImportError without ML deps."""

    def test_import_error_without_numpy(self, small_label_set):
        """ImportError when numpy is not installed."""
        with patch.dict("sys.modules", {"numpy": None}):
            from layoutlm_evaluate import evaluate_model
            with pytest.raises(ImportError, match="numpy"):
                evaluate_model("", [], small_label_set)

    def test_import_error_without_torch(self, small_label_set):
        """ImportError when torch is not installed."""
        mock_np = MagicMock()
        with patch.dict("sys.modules", {
            "numpy": mock_np,
            "torch": None,
        }):
            from layoutlm_evaluate import evaluate_model
            with pytest.raises(ImportError, match="torch"):
                evaluate_model("", [], small_label_set)

    def test_import_error_without_transformers(self, small_label_set):
        """ImportError when transformers is not installed."""
        mock_np = MagicMock()
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {
            "numpy": mock_np,
            "torch": mock_torch,
            "transformers": None,
        }):
            from layoutlm_evaluate import evaluate_model
            with pytest.raises(ImportError, match="transformers"):
                evaluate_model("", [], small_label_set)


# ---------------------------------------------------------------------------
# evaluate_predictions import guard tests
# ---------------------------------------------------------------------------


class TestEvaluatePredictionsImports:
    """Tests for evaluate_predictions with mocked seqeval."""

    def test_import_error_without_seqeval(self, small_label_set):
        """ImportError when seqeval is not installed."""
        true_labels = [["O", "B-DATE"]]
        pred_labels = [["O", "B-DATE"]]
        with patch.dict("sys.modules", {
            "seqeval": None,
            "seqeval.metrics": None,
        }):
            from layoutlm_evaluate import evaluate_predictions
            with pytest.raises(ImportError, match="seqeval"):
                evaluate_predictions(true_labels, pred_labels, small_label_set)

    def test_empty_predictions(self, small_label_set):
        """Empty input returns zeroed report without seqeval."""
        from layoutlm_evaluate import evaluate_predictions
        # Empty lists don't require seqeval at all
        report = evaluate_predictions([], [], small_label_set)
        assert report["overall"]["f1"] == 0.0
