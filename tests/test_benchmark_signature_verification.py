"""
Unit tests for the signature verification benchmark script.

Tests cover:
- CLI argument parsing (build_parser)
- Binary metrics computation (compute_binary_metrics)
- Timing statistics computation (compute_timing_stats)
- Dry-run corpus validation (validate_corpus_structure)
- Synthetic image generation
- Report formatting (console and markdown)
- Benchmark result data class defaults
- End-to-end synthetic benchmark run

Run with: python -m pytest tests/test_benchmark_signature_verification.py -v
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from scripts.benchmark_signature_verification import (  # noqa: I001
    SignatureVerificationBenchmarkResult,
    _generate_blank_image,
    _generate_signature_image,
    build_parser,
    compute_binary_metrics,
    compute_timing_stats,
    format_console_report,
    format_markdown_report,
    main,
    validate_corpus_structure,
)


# ---------------------------------------------------------------------------
# Test CLI argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for CLI argument parser construction."""

    def test_parser_synthetic_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic"])
        assert args.synthetic is True
        assert args.corpus_dir is None
        assert args.dry_run is False

    def test_parser_corpus_dir(self):
        parser = build_parser()
        args = parser.parse_args(["--corpus-dir", "/tmp/corpus"])
        assert args.corpus_dir == "/tmp/corpus"
        assert args.synthetic is False

    def test_parser_dry_run_with_corpus(self):
        parser = build_parser()
        args = parser.parse_args(["--corpus-dir", "/tmp/corpus", "--dry-run"])
        assert args.dry_run is True
        assert args.corpus_dir == "/tmp/corpus"

    def test_parser_sample_size_default(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic"])
        assert args.sample_size == 50

    def test_parser_sample_size_custom(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic", "--sample-size", "200"])
        assert args.sample_size == 200

    def test_parser_output_paths(self):
        parser = build_parser()
        args = parser.parse_args([
            "--synthetic",
            "--output", "results.json",
            "--output-md", "report.md",
        ])
        assert args.output == "results.json"
        assert args.output_md == "report.md"

    def test_parser_target_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic"])
        assert args.target_precision == 70.0
        assert args.target_recall == 70.0

    def test_parser_target_custom(self):
        parser = build_parser()
        args = parser.parse_args([
            "--synthetic",
            "--target-precision", "80.0",
            "--target-recall", "85.0",
        ])
        assert args.target_precision == 80.0
        assert args.target_recall == 85.0

    def test_parser_verbose(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic", "-v"])
        assert args.verbose is True


# ---------------------------------------------------------------------------
# Test metric computation
# ---------------------------------------------------------------------------


class TestComputeBinaryMetrics:
    """Tests for binary classification metric computation."""

    def test_perfect_predictions(self):
        preds = [True, True, False, False]
        truth = [True, True, False, False]
        m = compute_binary_metrics(preds, truth)
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["tn"] == 2
        assert m["fn"] == 0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["fp_rate"] == 0.0
        assert m["fn_rate"] == 0.0

    def test_all_false_positives(self):
        preds = [True, True, True]
        truth = [False, False, False]
        m = compute_binary_metrics(preds, truth)
        assert m["tp"] == 0
        assert m["fp"] == 3
        assert m["tn"] == 0
        assert m["fn"] == 0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["fp_rate"] == 1.0
        assert m["fn_rate"] == 0.0

    def test_all_false_negatives(self):
        preds = [False, False, False]
        truth = [True, True, True]
        m = compute_binary_metrics(preds, truth)
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["tn"] == 0
        assert m["fn"] == 3
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["fp_rate"] == 0.0
        assert m["fn_rate"] == 1.0

    def test_mixed_results(self):
        preds = [True, False, True, False, True]
        truth = [True, True, False, False, True]
        m = compute_binary_metrics(preds, truth)
        assert m["tp"] == 2
        assert m["fp"] == 1
        assert m["tn"] == 1
        assert m["fn"] == 1
        assert m["precision"] == pytest.approx(0.6667, abs=0.001)
        assert m["recall"] == pytest.approx(0.6667, abs=0.001)

    def test_empty_predictions(self):
        m = compute_binary_metrics([], [])
        assert m["tp"] == 0
        assert m["accuracy"] == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            compute_binary_metrics([True], [True, False])

    def test_fp_rate_computation(self):
        # 1 FP out of 4 negatives
        preds = [True, False, False, False, True]
        truth = [True, False, False, False, False]
        m = compute_binary_metrics(preds, truth)
        assert m["fp"] == 1
        assert m["tn"] == 3
        # FP rate = 1 / (1 + 3) = 0.25
        assert m["fp_rate"] == 0.25

    def test_fn_rate_computation(self):
        # 1 FN out of 3 positives
        preds = [True, True, False, False, False]
        truth = [True, True, True, False, False]
        m = compute_binary_metrics(preds, truth)
        assert m["fn"] == 1
        assert m["tp"] == 2
        # FN rate = 1 / (1 + 2) = 0.3333
        assert m["fn_rate"] == pytest.approx(0.3333, abs=0.001)


class TestComputeTimingStats:
    """Tests for timing statistics computation."""

    def test_empty_timings(self):
        stats = compute_timing_stats([])
        assert stats["avg_ms"] == 0.0
        assert stats["min_ms"] == 0.0
        assert stats["max_ms"] == 0.0
        assert stats["p95_ms"] == 0.0

    def test_single_timing(self):
        stats = compute_timing_stats([5.0])
        assert stats["avg_ms"] == 5.0
        assert stats["min_ms"] == 5.0
        assert stats["max_ms"] == 5.0
        assert stats["p95_ms"] == 5.0

    def test_multiple_timings(self):
        timings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = compute_timing_stats(timings)
        assert stats["avg_ms"] == 5.5
        assert stats["min_ms"] == 1.0
        assert stats["max_ms"] == 10.0
        assert stats["p95_ms"] == 10.0

    def test_few_timings_p95_fallback(self):
        # Less than 5 timings: p95 falls back to max
        stats = compute_timing_stats([1.0, 2.0, 3.0])
        assert stats["p95_ms"] == 3.0


# ---------------------------------------------------------------------------
# Test corpus validation (dry-run)
# ---------------------------------------------------------------------------


class TestValidateCorpusStructure:
    """Tests for dry-run corpus structure validation."""

    def test_nonexistent_directory(self):
        report = validate_corpus_structure("/nonexistent/path/12345")
        assert report["valid"] is False
        assert len(report["issues"]) > 0
        assert "does not exist" in report["issues"][0]

    def test_valid_corpus_with_images(self, tmp_path):
        sig_dir = tmp_path / "signatures"
        no_sig_dir = tmp_path / "no_signatures"
        sig_dir.mkdir()
        no_sig_dir.mkdir()

        # Create dummy image files
        (sig_dir / "sig1.png").write_bytes(b"\x89PNG")
        (sig_dir / "sig2.jpg").write_bytes(b"\xff\xd8")
        (no_sig_dir / "blank1.png").write_bytes(b"\x89PNG")

        report = validate_corpus_structure(str(tmp_path))
        assert report["valid"] is True
        assert report["signatures_dir_exists"] is True
        assert report["no_signatures_dir_exists"] is True
        assert report["signatures_image_count"] == 2
        assert report["no_signatures_image_count"] == 1
        assert report["total_images"] == 3

    def test_missing_signatures_dir(self, tmp_path):
        (tmp_path / "no_signatures").mkdir()
        report = validate_corpus_structure(str(tmp_path))
        assert report["valid"] is False
        assert any("signatures/" in issue for issue in report["issues"])

    def test_missing_no_signatures_dir(self, tmp_path):
        (tmp_path / "signatures").mkdir()
        report = validate_corpus_structure(str(tmp_path))
        assert report["valid"] is False
        assert any("no_signatures/" in issue for issue in report["issues"])

    def test_empty_directories(self, tmp_path):
        (tmp_path / "signatures").mkdir()
        (tmp_path / "no_signatures").mkdir()
        report = validate_corpus_structure(str(tmp_path))
        assert report["valid"] is False
        assert report["total_images"] == 0

    def test_unsupported_file_types_not_counted(self, tmp_path):
        sig_dir = tmp_path / "signatures"
        no_sig_dir = tmp_path / "no_signatures"
        sig_dir.mkdir()
        no_sig_dir.mkdir()
        # Create non-image files
        (sig_dir / "readme.txt").write_text("not an image")
        (sig_dir / "data.csv").write_text("a,b,c")
        (no_sig_dir / "notes.md").write_text("# notes")
        report = validate_corpus_structure(str(tmp_path))
        assert report["valid"] is False
        assert report["signatures_image_count"] == 0
        assert report["no_signatures_image_count"] == 0

    def test_warning_for_empty_signatures_subdir(self, tmp_path):
        sig_dir = tmp_path / "signatures"
        no_sig_dir = tmp_path / "no_signatures"
        sig_dir.mkdir()
        no_sig_dir.mkdir()
        # Only put images in no_signatures
        (no_sig_dir / "blank.png").write_bytes(b"\x89PNG")
        report = validate_corpus_structure(str(tmp_path))
        # Still valid if total > 0, but should have a warning
        assert report["valid"] is True
        assert any("signatures/" in issue and "Warning" in issue for issue in report["issues"])


# ---------------------------------------------------------------------------
# Test synthetic image generation
# ---------------------------------------------------------------------------


class TestSyntheticImageGeneration:
    """Tests for synthetic image generation helpers."""

    def test_signature_image_shape(self):
        img = _generate_signature_image(seed=42, width=400, height=200)
        assert isinstance(img, np.ndarray)
        assert img.shape == (200, 400)
        assert img.dtype == np.uint8

    def test_signature_image_has_dark_pixels(self):
        img = _generate_signature_image(seed=42)
        # Should have some ink (dark pixels < 100)
        assert np.sum(img < 100) > 0

    def test_blank_image_shape(self):
        img = _generate_blank_image(seed=42, width=300, height=150)
        assert isinstance(img, np.ndarray)
        assert img.shape == (150, 300)
        assert img.dtype == np.uint8

    def test_blank_image_mostly_light(self):
        img = _generate_blank_image(seed=42)
        # Should be mostly light (> 200)
        light_ratio = np.sum(img > 200) / img.size
        assert light_ratio > 0.5

    def test_reproducibility(self):
        img1 = _generate_signature_image(seed=99)
        img2 = _generate_signature_image(seed=99)
        np.testing.assert_array_equal(img1, img2)

    def test_different_seeds_differ(self):
        img1 = _generate_signature_image(seed=1)
        img2 = _generate_signature_image(seed=2)
        assert not np.array_equal(img1, img2)


# ---------------------------------------------------------------------------
# Test report formatting
# ---------------------------------------------------------------------------


class TestFormatConsoleReport:
    """Tests for console report formatting."""

    def test_report_contains_header(self):
        result = SignatureVerificationBenchmarkResult(
            total_images=10,
            precision=0.8,
            recall=0.75,
            f1=0.7742,
            accuracy=0.8,
            passed=True,
            dataset="synthetic",
            model="test_model",
        )
        report = format_console_report(result)
        assert "SIGNATURE VERIFICATION BENCHMARK" in report
        assert "DETECTION METRICS" in report
        assert "CONFUSION MATRIX" in report

    def test_report_contains_metrics(self):
        result = SignatureVerificationBenchmarkResult(
            total_images=20,
            precision=0.85,
            recall=0.90,
            fp_rate=0.10,
            fn_rate=0.10,
            passed=True,
            dataset="synthetic",
        )
        report = format_console_report(result)
        assert "0.8500" in report
        assert "0.9000" in report
        assert "FP Rate" in report
        assert "FN Rate" in report

    def test_report_pass_label(self):
        result = SignatureVerificationBenchmarkResult(passed=True, dataset="test")
        report = format_console_report(result)
        assert "PASS" in report

    def test_report_fail_label(self):
        result = SignatureVerificationBenchmarkResult(passed=False, dataset="test")
        report = format_console_report(result)
        assert "FAIL" in report

    def test_report_includes_notes(self):
        result = SignatureVerificationBenchmarkResult(
            dataset="test",
            notes=["This is a test note"],
        )
        report = format_console_report(result)
        assert "This is a test note" in report
        assert "NOTES" in report


class TestFormatMarkdownReport:
    """Tests for markdown report formatting."""

    def test_markdown_has_heading(self):
        result = SignatureVerificationBenchmarkResult(
            dataset="test",
            precision=0.8,
            recall=0.7,
        )
        report = format_markdown_report(result)
        assert "# Signature Verification Benchmark Report" in report

    def test_markdown_has_tables(self):
        result = SignatureVerificationBenchmarkResult(
            dataset="test",
            precision=0.85,
            recall=0.90,
            true_positives=17,
            false_positives=3,
        )
        report = format_markdown_report(result)
        assert "| Metric | Value |" in report
        assert "| Precision |" in report
        assert "Confusion Matrix" in report

    def test_markdown_includes_experimental_note(self):
        result = SignatureVerificationBenchmarkResult(dataset="test")
        report = format_markdown_report(result)
        assert "experimental and advisory-only" in report

    def test_markdown_includes_notes(self):
        result = SignatureVerificationBenchmarkResult(
            dataset="test",
            notes=["Custom note here"],
        )
        report = format_markdown_report(result)
        assert "Custom note here" in report


# ---------------------------------------------------------------------------
# Test data class defaults
# ---------------------------------------------------------------------------


class TestBenchmarkResultDefaults:
    """Tests for SignatureVerificationBenchmarkResult defaults."""

    def test_default_values(self):
        result = SignatureVerificationBenchmarkResult()
        assert result.total_images == 0
        assert result.true_positives == 0
        assert result.false_positives == 0
        assert result.true_negatives == 0
        assert result.false_negatives == 0
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.accuracy == 0.0
        assert result.fp_rate == 0.0
        assert result.fn_rate == 0.0
        assert result.target_precision == 70.0
        assert result.target_recall == 70.0
        assert result.passed is False
        assert result.notes == []

    def test_notes_isolation(self):
        r1 = SignatureVerificationBenchmarkResult()
        r2 = SignatureVerificationBenchmarkResult()
        r1.notes.append("test")
        assert "test" not in r2.notes


# ---------------------------------------------------------------------------
# Test main CLI entry point
# ---------------------------------------------------------------------------


class TestMainCli:
    """Tests for the main() CLI entry point."""

    def test_dry_run_nonexistent_corpus(self):
        exit_code = main(["--corpus-dir", "/nonexistent/path/xyz", "--dry-run"])
        assert exit_code == 1

    def test_dry_run_valid_corpus(self, tmp_path):
        sig_dir = tmp_path / "signatures"
        no_sig_dir = tmp_path / "no_signatures"
        sig_dir.mkdir()
        no_sig_dir.mkdir()
        (sig_dir / "sig1.png").write_bytes(b"\x89PNG")
        (no_sig_dir / "blank1.png").write_bytes(b"\x89PNG")

        exit_code = main(["--corpus-dir", str(tmp_path), "--dry-run"])
        assert exit_code == 0

    def test_synthetic_benchmark_runs(self):
        # Small sample to keep test fast
        exit_code = main(["--synthetic", "--sample-size", "4"])
        # Exit code depends on detection performance; just verify it runs
        assert exit_code in (0, 1)

    def test_synthetic_with_json_output(self, tmp_path):
        output_path = tmp_path / "results.json"
        exit_code = main([
            "--synthetic",
            "--sample-size", "4",
            "--output", str(output_path),
        ])
        assert exit_code in (0, 1)
        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "total_images" in data
        assert "precision" in data
        assert "fp_rate" in data

    def test_synthetic_with_markdown_output(self, tmp_path):
        output_path = tmp_path / "report.md"
        exit_code = main([
            "--synthetic",
            "--sample-size", "4",
            "--output-md", str(output_path),
        ])
        assert exit_code in (0, 1)
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "Signature Verification Benchmark Report" in content

    def test_dry_run_requires_corpus_dir(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--dry-run"])
        assert exc_info.value.code == 2

    def test_no_mode_selected_errors(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Test synthetic benchmark integration
# ---------------------------------------------------------------------------


class TestSyntheticBenchmarkIntegration:
    """Integration tests for the synthetic benchmark runner."""

    def test_run_synthetic_benchmark_import(self):
        from scripts.benchmark_signature_verification import run_synthetic_benchmark
        result = run_synthetic_benchmark(sample_size=6)
        assert result.total_images == 6
        assert result.dataset == "synthetic"
        assert result.model == "signature_verification.analyze_signature_page"
        assert isinstance(result.timestamp, str)
        assert len(result.timestamp) > 0

    def test_run_synthetic_counts_add_up(self):
        from scripts.benchmark_signature_verification import run_synthetic_benchmark
        result = run_synthetic_benchmark(sample_size=10)
        total = (
            result.true_positives
            + result.false_positives
            + result.true_negatives
            + result.false_negatives
        )
        assert total == result.total_images

    def test_run_synthetic_metrics_bounded(self):
        from scripts.benchmark_signature_verification import run_synthetic_benchmark
        result = run_synthetic_benchmark(sample_size=10)
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.fp_rate <= 1.0
        assert 0.0 <= result.fn_rate <= 1.0

    def test_run_synthetic_has_notes(self):
        from scripts.benchmark_signature_verification import run_synthetic_benchmark
        result = run_synthetic_benchmark(sample_size=4)
        assert len(result.notes) > 0
        assert any("experimental" in n.lower() for n in result.notes)
