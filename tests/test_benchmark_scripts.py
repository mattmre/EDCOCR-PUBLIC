"""Tests for ML benchmark framework code.

Tests the metric computation, result data classes, and synthetic generation
logic in the benchmark scripts. Does NOT run actual benchmarks against
real datasets or external ML models.

Run with: python -m pytest tests/test_benchmark_scripts.py -v
"""

import json
import os
import sys
from dataclasses import asdict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from scripts.benchmark_classification import (  # noqa: I001
    ClassificationBenchmarkResult,
    _generate_synthetic_text,
    compute_accuracy,
    compute_per_class_metrics,
    format_classification_report,
)
from scripts.benchmark_handwriting import (
    HandwritingBenchmarkResult,
    _edit_distance,
    compute_character_error_rate,
    compute_detection_metrics,
    compute_word_error_rate,
    format_handwriting_report,
)
from scripts.benchmark_watcher import (
    WatcherBenchmarkResult,
    compute_latency_percentiles,
    format_watcher_report,
)


# ===========================================================================
# Tests: compute_per_class_metrics (classification)
# ===========================================================================


class TestComputePerClassMetrics:
    """Tests for per-class precision, recall, F1 computation."""

    def test_perfect_classification(self):
        preds = ["a", "b", "c", "a", "b", "c"]
        truth = ["a", "b", "c", "a", "b", "c"]
        result = compute_per_class_metrics(preds, truth, ["a", "b", "c"])
        for cls in ["a", "b", "c"]:
            assert result[cls]["precision"] == 1.0
            assert result[cls]["recall"] == 1.0
            assert result[cls]["f1"] == 1.0
            assert result[cls]["support"] == 2

    def test_all_wrong(self):
        preds = ["b", "c", "a"]
        truth = ["a", "b", "c"]
        result = compute_per_class_metrics(preds, truth, ["a", "b", "c"])
        for cls in ["a", "b", "c"]:
            assert result[cls]["precision"] == 0.0
            assert result[cls]["recall"] == 0.0
            assert result[cls]["f1"] == 0.0

    def test_partial_correct(self):
        preds = ["a", "a", "b", "b"]
        truth = ["a", "b", "b", "a"]
        result = compute_per_class_metrics(preds, truth, ["a", "b"])
        # a: tp=1, fp=1, fn=1 -> precision=0.5, recall=0.5
        assert result["a"]["precision"] == 0.5
        assert result["a"]["recall"] == 0.5

    def test_missing_class_in_predictions(self):
        preds = ["a", "a", "a"]
        truth = ["a", "b", "c"]
        result = compute_per_class_metrics(preds, truth, ["a", "b", "c"])
        assert result["a"]["recall"] == 1.0
        assert result["b"]["recall"] == 0.0
        assert result["c"]["recall"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_per_class_metrics(["a"], ["a", "b"], ["a", "b"])

    def test_empty_inputs(self):
        result = compute_per_class_metrics([], [], ["a"])
        assert result["a"]["precision"] == 0.0
        assert result["a"]["recall"] == 0.0
        assert result["a"]["support"] == 0

    def test_single_class(self):
        preds = ["a", "a", "a"]
        truth = ["a", "a", "a"]
        result = compute_per_class_metrics(preds, truth, ["a"])
        assert result["a"]["precision"] == 1.0
        assert result["a"]["recall"] == 1.0
        assert result["a"]["f1"] == 1.0
        assert result["a"]["support"] == 3

    def test_f1_harmonic_mean(self):
        """F1 should be the harmonic mean of precision and recall."""
        preds = ["a", "a", "b", "b", "b"]
        truth = ["a", "b", "a", "b", "b"]
        result = compute_per_class_metrics(preds, truth, ["a", "b"])
        for cls in ["a", "b"]:
            p = result[cls]["precision"]
            r = result[cls]["recall"]
            expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            assert abs(result[cls]["f1"] - round(expected_f1, 4)) < 0.001


# ===========================================================================
# Tests: compute_accuracy (classification)
# ===========================================================================


class TestComputeAccuracy:
    """Tests for overall accuracy computation."""

    def test_perfect(self):
        assert compute_accuracy(["a", "b"], ["a", "b"]) == 1.0

    def test_all_wrong(self):
        assert compute_accuracy(["a", "a"], ["b", "b"]) == 0.0

    def test_half_correct(self):
        assert compute_accuracy(["a", "b"], ["a", "a"]) == 0.5

    def test_empty(self):
        assert compute_accuracy([], []) == 0.0


# ===========================================================================
# Tests: ClassificationBenchmarkResult
# ===========================================================================


class TestClassificationBenchmarkResult:
    """Tests for the classification benchmark result dataclass."""

    def test_default_values(self):
        result = ClassificationBenchmarkResult()
        assert result.total_documents == 0
        assert result.accuracy == 0.0
        assert result.passed is False
        assert result.per_class_metrics == {}

    def test_asdict(self):
        result = ClassificationBenchmarkResult(
            total_documents=10,
            correct=8,
            incorrect=2,
            accuracy=80.0,
            dataset="synthetic",
            model="test",
        )
        d = asdict(result)
        assert d["total_documents"] == 10
        assert d["accuracy"] == 80.0
        assert isinstance(d, dict)

    def test_json_serializable(self):
        result = ClassificationBenchmarkResult(
            total_documents=5,
            accuracy=90.0,
            per_class_metrics={"a": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 5}},
        )
        s = json.dumps(asdict(result))
        assert "total_documents" in s

    def test_passed_flag(self):
        result = ClassificationBenchmarkResult(
            accuracy=96.0,
            target_accuracy=95.0,
            passed=True,
        )
        assert result.passed is True
        result2 = ClassificationBenchmarkResult(
            accuracy=90.0,
            target_accuracy=95.0,
            passed=False,
        )
        assert result2.passed is False


# ===========================================================================
# Tests: _generate_synthetic_text (classification)
# ===========================================================================


class TestSyntheticBenchmark:
    """Tests for synthetic document generation."""

    def test_invoice_text_contains_keywords(self):
        text = _generate_synthetic_text("invoice", seed=42)
        text_lower = text.lower()
        assert "invoice" in text_lower
        assert "amount" in text_lower or "total" in text_lower

    def test_contract_text_contains_keywords(self):
        text = _generate_synthetic_text("contract", seed=42)
        text_lower = text.lower()
        assert "agreement" in text_lower or "contract" in text_lower

    def test_deterministic_with_same_seed(self):
        text1 = _generate_synthetic_text("letter", seed=123)
        text2 = _generate_synthetic_text("letter", seed=123)
        assert text1 == text2

    def test_different_seeds_different_text(self):
        text1 = _generate_synthetic_text("invoice", seed=1)
        text2 = _generate_synthetic_text("invoice", seed=999)
        # Same template but different random fill values
        assert text1 != text2

    def test_all_types_generate_nonempty(self):
        types = [
            "invoice", "contract", "letter", "form", "report",
            "memo", "receipt", "handwritten", "photograph", "other",
        ]
        for dtype in types:
            text = _generate_synthetic_text(dtype, seed=42)
            assert len(text) > 10, f"Empty text for type {dtype}"

    def test_unknown_type_falls_back_to_other(self):
        text = _generate_synthetic_text("nonexistent_type", seed=42)
        assert len(text) > 0  # Falls back to "other" template


# ===========================================================================
# Tests: edit_distance (handwriting)
# ===========================================================================


class TestEditDistance:
    """Tests for Levenshtein edit distance."""

    def test_identical(self):
        assert _edit_distance(list("abc"), list("abc")) == 0

    def test_empty_sequences(self):
        assert _edit_distance([], []) == 0

    def test_one_empty(self):
        assert _edit_distance(list("abc"), []) == 3
        assert _edit_distance([], list("abc")) == 3

    def test_single_substitution(self):
        assert _edit_distance(list("abc"), list("axc")) == 1

    def test_single_insertion(self):
        assert _edit_distance(list("ac"), list("abc")) == 1

    def test_single_deletion(self):
        assert _edit_distance(list("abc"), list("ac")) == 1

    def test_completely_different(self):
        assert _edit_distance(list("abc"), list("xyz")) == 3


# ===========================================================================
# Tests: compute_character_error_rate (handwriting)
# ===========================================================================


class TestHandwritingMetricsCER:
    """Tests for Character Error Rate computation."""

    def test_identical_strings(self):
        assert compute_character_error_rate("hello", "hello") == 0.0

    def test_completely_different(self):
        assert compute_character_error_rate("abc", "xyz") == 1.0

    def test_empty_reference_empty_hypothesis(self):
        assert compute_character_error_rate("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        assert compute_character_error_rate("", "abc") == 1.0

    def test_partial_errors(self):
        cer = compute_character_error_rate("hello", "hallo")
        assert 0.0 < cer < 1.0
        # One substitution out of 5 chars = 0.2
        assert abs(cer - 0.2) < 0.01

    def test_insertion_errors(self):
        cer = compute_character_error_rate("abc", "abxc")
        assert 0.0 < cer <= 1.0


# ===========================================================================
# Tests: compute_word_error_rate (handwriting)
# ===========================================================================


class TestHandwritingMetricsWER:
    """Tests for Word Error Rate computation."""

    def test_identical_text(self):
        assert compute_word_error_rate("hello world", "hello world") == 0.0

    def test_completely_different(self):
        wer = compute_word_error_rate("hello world", "foo bar")
        assert wer == 1.0

    def test_empty_reference_empty_hypothesis(self):
        assert compute_word_error_rate("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        assert compute_word_error_rate("", "hello") == 1.0

    def test_one_word_wrong(self):
        wer = compute_word_error_rate("hello world", "hello earth")
        assert abs(wer - 0.5) < 0.01

    def test_extra_words(self):
        wer = compute_word_error_rate("hello", "hello world")
        assert 0.0 < wer <= 1.0


# ===========================================================================
# Tests: compute_detection_metrics (handwriting)
# ===========================================================================


class TestDetectionMetrics:
    """Tests for binary detection metrics."""

    def test_perfect_detection(self):
        preds = [True, False, True, False]
        truth = [True, False, True, False]
        result = compute_detection_metrics(preds, truth)
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_all_false_positives(self):
        preds = [True, True]
        truth = [False, False]
        result = compute_detection_metrics(preds, truth)
        assert result["precision"] == 0.0
        assert result["fp"] == 2

    def test_all_false_negatives(self):
        preds = [False, False]
        truth = [True, True]
        result = compute_detection_metrics(preds, truth)
        assert result["recall"] == 0.0
        assert result["fn"] == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_detection_metrics([True], [True, False])

    def test_empty_inputs(self):
        result = compute_detection_metrics([], [])
        assert result["accuracy"] == 0.0

    def test_mixed_results(self):
        preds = [True, True, False, False]
        truth = [True, False, True, False]
        result = compute_detection_metrics(preds, truth)
        assert result["tp"] == 1
        assert result["fp"] == 1
        assert result["fn"] == 1
        assert result["tn"] == 1
        assert result["accuracy"] == 0.5


# ===========================================================================
# Tests: compute_latency_percentiles (watcher)
# ===========================================================================


class TestWatcherLatencyPercentiles:
    """Tests for latency percentile computation."""

    def test_empty_latencies(self):
        result = compute_latency_percentiles([])
        assert result["p50"] == 0.0
        assert result["p95"] == 0.0
        assert result["mean"] == 0.0

    def test_single_value(self):
        result = compute_latency_percentiles([100.0])
        assert result["p50"] == 100.0
        assert result["min"] == 100.0
        assert result["max"] == 100.0
        assert result["mean"] == 100.0
        assert result["stddev"] == 0.0

    def test_sorted_output(self):
        result = compute_latency_percentiles([50, 10, 90, 30, 70])
        assert result["min"] == 10.0
        assert result["max"] == 90.0
        assert result["p50"] <= result["p95"]
        assert result["p95"] <= result["p99"]

    def test_known_percentiles(self):
        # 100 values from 1 to 100
        latencies = list(range(1, 101))
        result = compute_latency_percentiles([float(x) for x in latencies])
        assert result["min"] == 1.0
        assert result["max"] == 100.0
        # p50 should be around 50
        assert 49 <= result["p50"] <= 51
        # p95 should be around 95
        assert 94 <= result["p95"] <= 96

    def test_two_values_has_stddev(self):
        result = compute_latency_percentiles([10.0, 20.0])
        assert result["stddev"] > 0


# ===========================================================================
# Tests: WatcherBenchmarkResult
# ===========================================================================


class TestWatcherBenchmarkResult:
    """Tests for the watcher benchmark result dataclass."""

    def test_default_values(self):
        result = WatcherBenchmarkResult()
        assert result.iterations == 0
        assert result.passed is False
        assert result.latencies_ms == []

    def test_passed_flag(self):
        result = WatcherBenchmarkResult(
            p95_ms=500.0,
            target_latency_s=10.0,
            passed=True,
        )
        assert result.passed is True

    def test_json_serializable(self):
        result = WatcherBenchmarkResult(
            iterations=10,
            latencies_ms=[1.0, 2.0, 3.0],
        )
        s = json.dumps(asdict(result))
        assert "iterations" in s


# ===========================================================================
# Tests: HandwritingBenchmarkResult
# ===========================================================================


class TestHandwritingBenchmarkResult:
    """Tests for the handwriting benchmark result dataclass."""

    def test_default_values(self):
        result = HandwritingBenchmarkResult()
        assert result.total_samples == 0
        assert result.passed is False

    def test_json_serializable(self):
        result = HandwritingBenchmarkResult(
            total_samples=50,
            detection_accuracy=85.0,
        )
        s = json.dumps(asdict(result))
        assert "total_samples" in s


# ===========================================================================
# Tests: Report formatting
# ===========================================================================


class TestReportFormatting:
    """Tests for human-readable report generation."""

    def test_classification_report_contains_key_fields(self):
        result = ClassificationBenchmarkResult(
            total_documents=100,
            correct=90,
            incorrect=10,
            accuracy=90.0,
            dataset="synthetic",
            model="test_model",
            target_accuracy=95.0,
            passed=False,
        )
        report = format_classification_report(result)
        assert "CLASSIFICATION" in report
        assert "90.0" in report
        assert "FAIL" in report

    def test_handwriting_report_contains_key_fields(self):
        result = HandwritingBenchmarkResult(
            total_samples=50,
            detection_accuracy=85.0,
            avg_character_error_rate=0.15,
            dataset="synthetic",
            target_accuracy=80.0,
            passed=True,
        )
        report = format_handwriting_report(result)
        assert "HANDWRITING" in report
        assert "85.0" in report
        assert "PASS" in report

    def test_watcher_report_contains_key_fields(self):
        result = WatcherBenchmarkResult(
            iterations=20,
            successful_detections=20,
            p95_ms=150.0,
            target_latency_s=10.0,
            passed=True,
        )
        report = format_watcher_report(result)
        assert "WATCHER" in report
        assert "150.0" in report
        assert "PASS" in report

    def test_classification_report_with_per_class(self):
        result = ClassificationBenchmarkResult(
            per_class_metrics={
                "invoice": {"precision": 0.95, "recall": 0.9, "f1": 0.92, "support": 10},
                "letter": {"precision": 0.8, "recall": 0.85, "f1": 0.82, "support": 10},
            },
        )
        report = format_classification_report(result)
        assert "invoice" in report
        assert "letter" in report
        assert "Precision" in report
