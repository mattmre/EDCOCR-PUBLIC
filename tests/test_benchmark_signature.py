"""Tests for scripts/benchmark_signature.py.

Covers:
- Corpus generation (creates expected files and manifest)
- Metric computation (known confusion matrix values)
- CLI argument parsing (subcommand dispatch)
- Report generation (Markdown format, confusion matrix)
- End-to-end generate + run flow
"""

import json
import os
import sys

import pytest

# Ensure project root and scripts directory are on sys.path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from scripts.benchmark_signature import (
    CATEGORY_NO_SIGNATURE,
    CATEGORY_NOISE_STAMP,
    CATEGORY_SIGNATURE,
    SignatureBenchmarkResult,
    compute_metrics,
    generate_corpus,
    generate_report,
)

# ---------------------------------------------------------------------------
# Corpus generation tests
# ---------------------------------------------------------------------------


class TestGenerateCorpus:
    """Test synthetic corpus generation."""

    def test_generates_correct_count(self, tmp_path):
        """Corpus generates the requested number of images."""
        manifest = generate_corpus(str(tmp_path), count=30, seed=1)
        assert len(manifest) == 30

    def test_creates_image_files(self, tmp_path):
        """All manifest entries have corresponding PNG files."""
        manifest = generate_corpus(str(tmp_path), count=15, seed=2)
        for fname in manifest:
            assert (tmp_path / fname).exists(), f"Missing image: {fname}"
            assert fname.endswith(".png")

    def test_creates_ground_truth_json(self, tmp_path):
        """ground_truth.json is created alongside images."""
        generate_corpus(str(tmp_path), count=12, seed=3)
        gt_path = tmp_path / "ground_truth.json"
        assert gt_path.exists()
        with open(str(gt_path), encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)
        assert len(data) == 12

    def test_category_distribution(self, tmp_path):
        """Categories are roughly equally distributed."""
        manifest = generate_corpus(str(tmp_path), count=60, seed=4)
        cats = list(manifest.values())
        sig_count = cats.count(CATEGORY_SIGNATURE)
        no_sig_count = cats.count(CATEGORY_NO_SIGNATURE)
        noise_count = cats.count(CATEGORY_NOISE_STAMP)
        assert sig_count == 20
        assert no_sig_count == 20
        assert noise_count == 20

    def test_valid_category_labels(self, tmp_path):
        """All labels are one of the three expected categories."""
        manifest = generate_corpus(str(tmp_path), count=9, seed=5)
        valid = {CATEGORY_SIGNATURE, CATEGORY_NO_SIGNATURE, CATEGORY_NOISE_STAMP}
        for cat in manifest.values():
            assert cat in valid, f"Unexpected category: {cat}"

    def test_reproducible_with_seed(self, tmp_path):
        """Same seed produces identical manifest."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        m1 = generate_corpus(str(dir1), count=20, seed=99)
        m2 = generate_corpus(str(dir2), count=20, seed=99)
        assert m1 == m2

    def test_images_are_valid_pngs(self, tmp_path):
        """Generated images can be opened by Pillow."""
        from PIL import Image

        manifest = generate_corpus(str(tmp_path), count=6, seed=6)
        for fname in manifest:
            img = Image.open(str(tmp_path / fname))
            assert img.size[0] > 0
            assert img.size[1] > 0

    def test_minimum_count(self, tmp_path):
        """Generating 3 images (one per category) works."""
        manifest = generate_corpus(str(tmp_path), count=3, seed=7)
        assert len(manifest) == 3
        cats = set(manifest.values())
        assert len(cats) >= 1  # At least one category present


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Test binary classification metric computation."""

    def test_perfect_predictions(self):
        """All correct predictions yield perfect metrics."""
        preds = [True, True, False, False, True]
        gt = [True, True, False, False, True]
        m = compute_metrics(preds, gt)
        assert m["tp"] == 3
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["tn"] == 2
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["accuracy"] == 1.0

    def test_all_wrong_predictions(self):
        """All incorrect predictions yield zero precision/recall."""
        preds = [True, True, True, True]
        gt = [False, False, False, False]
        m = compute_metrics(preds, gt)
        assert m["tp"] == 0
        assert m["fp"] == 4
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_known_confusion_matrix(self):
        """Verify specific known confusion matrix values."""
        # 3 TP, 1 FP, 2 FN, 4 TN
        preds = [True, True, True, True, False, False, False, False, False, False]
        gt = [True, True, True, False, True, True, False, False, False, False]
        m = compute_metrics(preds, gt)
        assert m["tp"] == 3
        assert m["fp"] == 1
        assert m["fn"] == 2
        assert m["tn"] == 4
        # precision = 3/4 = 0.75
        assert m["precision"] == 0.75
        # recall = 3/5 = 0.6
        assert m["recall"] == 0.6
        # f1 = 2 * 0.75 * 0.6 / (0.75 + 0.6) = 0.9 / 1.35 = 0.6667
        assert abs(m["f1"] - 0.6667) < 0.001
        # accuracy = 7/10 = 0.7
        assert m["accuracy"] == 0.7

    def test_empty_predictions(self):
        """Empty prediction list yields zero metrics."""
        m = compute_metrics([], [])
        assert m["tp"] == 0
        assert m["accuracy"] == 0.0

    def test_mismatched_lengths_raises(self):
        """Mismatched list lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            compute_metrics([True], [True, False])

    def test_all_negative_ground_truth(self):
        """All-negative ground truth with all-negative predictions."""
        preds = [False, False, False]
        gt = [False, False, False]
        m = compute_metrics(preds, gt)
        assert m["tn"] == 3
        assert m["precision"] == 0.0  # No positives predicted
        assert m["accuracy"] == 1.0

    def test_single_sample_positive(self):
        """Single positive sample correctly predicted."""
        m = compute_metrics([True], [True])
        assert m["tp"] == 1
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Test Markdown report generation."""

    def test_report_contains_metrics(self):
        """Report includes precision, recall, F1, accuracy."""
        metrics = {
            "tp": 10, "fp": 2, "tn": 8, "fn": 3,
            "precision": 0.8333, "recall": 0.7692,
            "f1": 0.8000, "accuracy": 0.7826,
        }
        report = generate_report(metrics)
        assert "Precision:" in report
        assert "Recall:" in report
        assert "F1:" in report
        assert "Accuracy:" in report

    def test_report_contains_confusion_matrix(self):
        """Report includes confusion matrix with TP/FP/FN/TN."""
        metrics = {
            "tp": 5, "fp": 1, "tn": 4, "fn": 2,
            "precision": 0.8333, "recall": 0.7143,
            "f1": 0.7692, "accuracy": 0.75,
        }
        report = generate_report(metrics)
        assert "CONFUSION MATRIX" in report
        assert "TP = 5" in report
        assert "FP = 1" in report
        assert "FN = 2" in report
        assert "TN = 4" in report

    def test_report_with_result_object(self):
        """Report includes timing and dataset info when result is provided."""
        metrics = {
            "tp": 3, "fp": 0, "tn": 3, "fn": 0,
            "precision": 1.0, "recall": 1.0,
            "f1": 1.0, "accuracy": 1.0,
        }
        result = SignatureBenchmarkResult(
            total_samples=6,
            dataset="test_corpus",
            target_f1=60.0,
            passed=True,
            avg_inference_time_ms=1.5,
            min_inference_time_ms=0.8,
            max_inference_time_ms=3.0,
            p95_inference_time_ms=2.5,
            timestamp="2026-01-01T00:00:00Z",
        )
        report = generate_report(metrics, result)
        assert "test_corpus" in report
        assert "PASS" in report
        assert "INFERENCE TIMING" in report

    def test_report_shows_fail_when_below_target(self):
        """Report shows FAIL when F1 is below target."""
        metrics = {
            "tp": 1, "fp": 5, "tn": 3, "fn": 4,
            "precision": 0.1667, "recall": 0.2,
            "f1": 0.1818, "accuracy": 0.3077,
        }
        result = SignatureBenchmarkResult(
            total_samples=13,
            target_f1=60.0,
            passed=False,
            dataset="bad_corpus",
        )
        report = generate_report(metrics, result)
        assert "FAIL" in report

    def test_report_per_category_breakdown(self):
        """Report includes per-category breakdown when available."""
        metrics = {
            "tp": 3, "fp": 1, "tn": 4, "fn": 2,
            "precision": 0.75, "recall": 0.6,
            "f1": 0.6667, "accuracy": 0.7,
        }
        result = SignatureBenchmarkResult(
            total_samples=10,
            per_category={
                "signature": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "accuracy": 0.9},
                "no_signature": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "accuracy": 0.8},
            },
        )
        report = generate_report(metrics, result)
        assert "PER-CATEGORY BREAKDOWN" in report
        assert "signature" in report
        assert "no_signature" in report

    def test_report_header(self):
        """Report starts with the benchmark title."""
        metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                   "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
        report = generate_report(metrics)
        assert "SIGNATURE DETECTION BENCHMARK" in report


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Test CLI argument parsing and subcommand dispatch."""

    def test_main_no_args_returns_1(self):
        """Calling main with no args prints help and returns 1."""
        from scripts.benchmark_signature import main

        sys.argv = ["benchmark_signature.py"]
        assert main() == 1

    def test_generate_subcommand_creates_files(self, tmp_path):
        """'generate' subcommand creates corpus files."""
        from scripts.benchmark_signature import main

        out_dir = str(tmp_path / "gen")
        sys.argv = [
            "benchmark_signature.py", "generate",
            "--output-dir", out_dir,
            "--count", "6",
            "--seed", "10",
        ]
        ret = main()
        assert ret == 0
        assert (tmp_path / "gen" / "ground_truth.json").exists()

    def test_report_missing_file_returns_1(self):
        """'report' with a non-existent file returns 1."""
        from scripts.benchmark_signature import main

        sys.argv = [
            "benchmark_signature.py", "report",
            "--results-file", "/nonexistent/results.json",
        ]
        assert main() == 1

    def test_report_subcommand_reads_json(self, tmp_path):
        """'report' subcommand reads and formats a results JSON file."""
        from scripts.benchmark_signature import main

        results = {
            "total_samples": 10,
            "true_positives": 3,
            "false_positives": 1,
            "true_negatives": 4,
            "false_negatives": 2,
            "precision": 0.75,
            "recall": 0.6,
            "f1": 0.6667,
            "accuracy": 0.7,
            "avg_inference_time_ms": 1.0,
            "min_inference_time_ms": 0.5,
            "max_inference_time_ms": 2.0,
            "p95_inference_time_ms": 1.8,
            "dataset": "test",
            "target_f1": 60.0,
            "passed": True,
            "timestamp": "2026-01-01T00:00:00Z",
            "per_category": {},
        }
        results_file = tmp_path / "results.json"
        with open(str(results_file), "w", encoding="utf-8") as fh:
            json.dump(results, fh)

        sys.argv = [
            "benchmark_signature.py", "report",
            "--results-file", str(results_file),
        ]
        ret = main()
        assert ret == 0


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end generate + run test."""

    def test_generate_and_run(self, tmp_path):
        """Generate a small corpus, run benchmark, verify metrics are populated."""
        corpus_dir = str(tmp_path / "corpus")
        manifest = generate_corpus(corpus_dir, count=12, seed=42)
        assert len(manifest) == 12

        try:
            from scripts.benchmark_signature import run_benchmark
        except ImportError:
            pytest.skip("benchmark_signature not importable")

        result, details = run_benchmark(corpus_dir)
        assert result.total_samples == 12
        assert len(details) == 12

        # All detail records should have required fields
        for d in details:
            assert "file" in d
            assert "category" in d
            assert "ground_truth" in d
            assert "predicted" in d
            assert "correct" in d
            assert "inference_ms" in d

        # Metrics should be populated
        assert result.precision >= 0.0
        assert result.recall >= 0.0
        assert result.f1 >= 0.0
        assert result.accuracy >= 0.0

    def test_run_with_output_file(self, tmp_path):
        """Run benchmark and save JSON output."""
        from scripts.benchmark_signature import main

        corpus_dir = str(tmp_path / "corpus")
        generate_corpus(corpus_dir, count=9, seed=55)

        output_file = str(tmp_path / "out.json")
        sys.argv = [
            "benchmark_signature.py", "run",
            "--corpus-dir", corpus_dir,
            "--output", output_file,
        ]
        main()

        assert os.path.exists(output_file)
        with open(output_file, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "total_samples" in data
        assert "details" in data
        assert data["total_samples"] == 9

    def test_run_missing_corpus_returns_empty(self, tmp_path):
        """Running against a missing corpus returns zero-sample result."""
        try:
            from scripts.benchmark_signature import run_benchmark
        except ImportError:
            pytest.skip("benchmark_signature not importable")

        result, details = run_benchmark(str(tmp_path / "nonexistent"))
        assert result.total_samples == 0
        assert len(details) == 0
