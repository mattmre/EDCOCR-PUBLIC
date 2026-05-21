"""
Unit tests for CJK vertical text benchmark script.

Tests cover:
- CLI argument parsing (build_parser)
- Kendall's tau metric computation
- Column precision metric computation
- Corpus structure validation (dry-run mode)
- Synthetic benchmark execution
- Report formatting (plain text + markdown)
- Sidecar JSON parsing
- Edge cases (empty inputs, missing fields, malformed data)

Run with: python -m pytest tests/test_benchmark_vertical_text.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from benchmark_vertical_text import (  # noqa: E402
    CORPUS_SUBDIRS,
    SIDECAR_SUFFIX,
    CategoryMetrics,
    VerticalTextBenchmarkResult,
    build_parser,
    compute_column_precision,
    compute_kendall_tau,
    format_benchmark_report,
    format_markdown_report,
    run_synthetic_benchmark,
    validate_corpus_structure,
)

# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_synthetic_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic"])
        assert args.synthetic is True
        assert args.corpus_dir is None
        assert args.dry_run is False

    def test_corpus_dir(self):
        parser = build_parser()
        args = parser.parse_args(["--corpus-dir", "/data/corpus"])
        assert args.corpus_dir == "/data/corpus"
        assert args.synthetic is False

    def test_dry_run(self):
        parser = build_parser()
        args = parser.parse_args(["--corpus-dir", "/data/corpus", "--dry-run"])
        assert args.dry_run is True
        assert args.corpus_dir == "/data/corpus"

    def test_sample_size(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic", "--sample-size", "50"])
        assert args.sample_size == 50

    def test_default_sample_size(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic"])
        assert args.sample_size == 30

    def test_output_json(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic", "--output", "results.json"])
        assert args.output == "results.json"

    def test_output_markdown(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic", "--output-md", "report.md"])
        assert args.output_md == "report.md"

    def test_target_direction_accuracy(self):
        parser = build_parser()
        args = parser.parse_args([
            "--synthetic", "--target-direction-accuracy", "95.0"
        ])
        assert args.target_direction_accuracy == 95.0

    def test_target_ordering_tau(self):
        parser = build_parser()
        args = parser.parse_args([
            "--synthetic", "--target-ordering-tau", "0.90"
        ])
        assert args.target_ordering_tau == 0.90

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic", "-v"])
        assert args.verbose is True

    def test_default_targets(self):
        parser = build_parser()
        args = parser.parse_args(["--synthetic"])
        assert args.target_direction_accuracy == 90.0
        assert args.target_ordering_tau == 0.80


# ---------------------------------------------------------------------------
# Tests: compute_kendall_tau
# ---------------------------------------------------------------------------


class TestComputeKendallTau:
    def test_perfect_agreement(self):
        order = ["a", "b", "c", "d"]
        tau = compute_kendall_tau(order, order)
        assert tau == 1.0

    def test_perfect_reversal(self):
        predicted = ["d", "c", "b", "a"]
        expected = ["a", "b", "c", "d"]
        tau = compute_kendall_tau(predicted, expected)
        assert tau == -1.0

    def test_partial_agreement(self):
        predicted = ["a", "c", "b", "d"]
        expected = ["a", "b", "c", "d"]
        tau = compute_kendall_tau(predicted, expected)
        # One discordant pair (c,b) out of 6 pairs: (5-1)/6 = 0.6667
        assert abs(tau - 0.6667) < 0.01

    def test_empty_lists(self):
        assert compute_kendall_tau([], []) == 0.0

    def test_single_element(self):
        assert compute_kendall_tau(["a"], ["a"]) == 0.0

    def test_no_common_items(self):
        predicted = ["x", "y", "z"]
        expected = ["a", "b", "c"]
        tau = compute_kendall_tau(predicted, expected)
        assert tau == 0.0

    def test_partial_overlap(self):
        predicted = ["a", "b", "x"]
        expected = ["a", "b", "c"]
        # Common items: a, b -> perfect order
        tau = compute_kendall_tau(predicted, expected)
        assert tau == 1.0

    def test_two_elements_swapped(self):
        predicted = ["b", "a"]
        expected = ["a", "b"]
        tau = compute_kendall_tau(predicted, expected)
        assert tau == -1.0

    def test_two_elements_correct(self):
        predicted = ["a", "b"]
        expected = ["a", "b"]
        tau = compute_kendall_tau(predicted, expected)
        assert tau == 1.0


# ---------------------------------------------------------------------------
# Tests: compute_column_precision
# ---------------------------------------------------------------------------


class TestComputeColumnPrecision:
    def test_exact_match(self):
        assert compute_column_precision(3, 3) == 1.0

    def test_overshoot(self):
        # predicted=4, expected=3 -> 1 - 1/4 = 0.75
        prec = compute_column_precision(4, 3)
        assert abs(prec - 0.75) < 0.01

    def test_undershoot(self):
        # predicted=2, expected=3 -> 1 - 1/3 = 0.6667
        prec = compute_column_precision(2, 3)
        assert abs(prec - 0.6667) < 0.01

    def test_zero_predicted_nonzero_expected(self):
        assert compute_column_precision(0, 3) == 0.0

    def test_zero_expected_zero_predicted(self):
        assert compute_column_precision(0, 0) == 1.0

    def test_zero_expected_nonzero_predicted(self):
        assert compute_column_precision(2, 0) == 0.0

    def test_single_column(self):
        assert compute_column_precision(1, 1) == 1.0

    def test_large_difference(self):
        # predicted=1, expected=10 -> 1 - 9/10 = 0.1
        prec = compute_column_precision(1, 10)
        assert abs(prec - 0.1) < 0.01


# ---------------------------------------------------------------------------
# Tests: validate_corpus_structure
# ---------------------------------------------------------------------------


class TestValidateCorpusStructure:
    def test_nonexistent_directory(self):
        result = validate_corpus_structure("/nonexistent/path/corpus")
        assert result["valid"] is False
        assert result["total_cases"] == 0
        assert len(result["errors"]) >= 1

    def test_empty_directory(self, tmp_path):
        result = validate_corpus_structure(str(tmp_path))
        assert result["valid"] is False
        assert "No recognized subdirectories" in result["errors"][0]

    def test_valid_corpus_with_one_category(self, tmp_path):
        # Create vertical_only with a valid pair
        vert_dir = tmp_path / "vertical_only"
        vert_dir.mkdir()

        # Create a dummy image
        (vert_dir / "test.png").write_bytes(b"fake image data")

        # Create a valid sidecar
        sidecar = {
            "direction": "vertical",
            "expected_order": ["line1", "line2"],
            "ocr_lines": [
                {"text": "line1", "box": [[0, 0], [30, 0], [30, 200], [0, 200]], "confidence": 0.9},
                {"text": "line2", "box": [[0, 220], [30, 220], [30, 420], [0, 420]], "confidence": 0.9},
            ],
            "page_width": 1000,
        }
        with open(vert_dir / "test.expected.json", "w") as f:
            json.dump(sidecar, f)

        result = validate_corpus_structure(str(tmp_path))
        assert result["valid"] is True
        assert result["total_cases"] == 1
        assert result["categories"]["vertical_only"]["valid_pairs"] == 1

    def test_missing_sidecar(self, tmp_path):
        vert_dir = tmp_path / "vertical_only"
        vert_dir.mkdir()
        (vert_dir / "orphan.png").write_bytes(b"fake image data")

        result = validate_corpus_structure(str(tmp_path))
        assert result["valid"] is True  # Missing sidecars are warnings, not failures
        cat = result["categories"]["vertical_only"]
        assert cat["image_count"] == 1
        assert cat["valid_pairs"] == 0
        assert len(cat["missing_sidecars"]) == 1

    def test_invalid_json_sidecar(self, tmp_path):
        vert_dir = tmp_path / "vertical_only"
        vert_dir.mkdir()
        (vert_dir / "bad.png").write_bytes(b"fake image data")
        (vert_dir / "bad.expected.json").write_text("not valid json {{{")

        result = validate_corpus_structure(str(tmp_path))
        assert result["valid"] is False
        assert len(result["errors"]) >= 1

    def test_missing_required_fields(self, tmp_path):
        vert_dir = tmp_path / "vertical_only"
        vert_dir.mkdir()
        (vert_dir / "incomplete.png").write_bytes(b"fake image data")

        sidecar = {"direction": "vertical"}  # Missing expected_order, ocr_lines, page_width
        with open(vert_dir / "incomplete.expected.json", "w") as f:
            json.dump(sidecar, f)

        result = validate_corpus_structure(str(tmp_path))
        assert result["valid"] is False
        cat = result["categories"]["vertical_only"]
        assert len(cat["invalid_sidecars"]) == 1

    def test_all_three_categories(self, tmp_path):
        for subdir_name in CORPUS_SUBDIRS:
            subdir = tmp_path / subdir_name
            subdir.mkdir()
            (subdir / "sample.png").write_bytes(b"fake")
            sidecar = {
                "direction": "horizontal",
                "expected_order": ["text"],
                "ocr_lines": [
                    {"text": "text", "box": [[0, 0], [400, 0], [400, 30], [0, 30]]}
                ],
                "page_width": 1000,
            }
            with open(subdir / "sample.expected.json", "w") as f:
                json.dump(sidecar, f)

        result = validate_corpus_structure(str(tmp_path))
        assert result["valid"] is True
        assert result["total_cases"] == 3
        for subdir_name in CORPUS_SUBDIRS:
            assert result["categories"][subdir_name]["exists"] is True
            assert result["categories"][subdir_name]["valid_pairs"] == 1

    def test_orphan_sidecar(self, tmp_path):
        """Sidecar without a matching image."""
        vert_dir = tmp_path / "vertical_only"
        vert_dir.mkdir()
        sidecar = {
            "direction": "vertical",
            "expected_order": ["text"],
            "ocr_lines": [
                {"text": "text", "box": [[0, 0], [30, 0], [30, 200], [0, 200]]}
            ],
            "page_width": 1000,
        }
        with open(vert_dir / "orphan.expected.json", "w") as f:
            json.dump(sidecar, f)

        result = validate_corpus_structure(str(tmp_path))
        cat = result["categories"]["vertical_only"]
        assert len(cat["missing_images"]) == 1


# ---------------------------------------------------------------------------
# Tests: Synthetic benchmark
# ---------------------------------------------------------------------------


class TestRunSyntheticBenchmark:
    def test_synthetic_runs_without_error(self):
        result = run_synthetic_benchmark(sample_size=10)
        assert result.total_cases == 10
        assert result.dataset == "synthetic"
        assert result.timestamp != ""

    def test_synthetic_produces_direction_accuracy(self):
        result = run_synthetic_benchmark(sample_size=15)
        assert 0.0 <= result.direction_accuracy <= 100.0

    def test_synthetic_produces_kendall_tau(self):
        result = run_synthetic_benchmark(sample_size=15)
        assert -1.0 <= result.avg_kendall_tau <= 1.0

    def test_synthetic_produces_column_precision(self):
        result = run_synthetic_benchmark(sample_size=15)
        assert 0.0 <= result.avg_column_precision <= 1.0

    def test_synthetic_timing_metrics(self):
        result = run_synthetic_benchmark(sample_size=10)
        assert result.avg_inference_time_ms >= 0.0
        assert result.min_inference_time_ms >= 0.0
        assert result.max_inference_time_ms >= result.min_inference_time_ms

    def test_synthetic_per_category(self):
        result = run_synthetic_benchmark(sample_size=15)
        assert "vertical_only" in result.per_category
        assert "mixed_layout" in result.per_category
        assert "horizontal_only" in result.per_category

    def test_synthetic_pass_with_generous_targets(self):
        result = run_synthetic_benchmark(
            sample_size=20,
            target_direction_accuracy=50.0,
            target_ordering_tau=0.0,
        )
        assert result.passed is True

    def test_synthetic_small_sample(self):
        result = run_synthetic_benchmark(sample_size=3)
        assert result.total_cases == 3

    def test_synthetic_deterministic_seed(self):
        """Two runs with the same parameters should produce the same results."""
        r1 = run_synthetic_benchmark(sample_size=10)
        r2 = run_synthetic_benchmark(sample_size=10)
        assert r1.direction_accuracy == r2.direction_accuracy
        assert r1.avg_kendall_tau == r2.avg_kendall_tau


# ---------------------------------------------------------------------------
# Tests: Report formatting
# ---------------------------------------------------------------------------


class TestFormatBenchmarkReport:
    def _make_result(self):
        return VerticalTextBenchmarkResult(
            total_cases=30,
            direction_accuracy=93.33,
            avg_kendall_tau=0.8721,
            avg_column_precision=0.9500,
            avg_inference_time_ms=0.15,
            min_inference_time_ms=0.05,
            max_inference_time_ms=0.42,
            p95_inference_time_ms=0.38,
            per_category={
                "vertical_only": {
                    "category": "vertical_only",
                    "total_cases": 12,
                    "direction_correct": 12,
                    "direction_accuracy": 100.0,
                    "avg_kendall_tau": 0.92,
                    "avg_column_precision": 1.0,
                    "avg_inference_time_ms": 0.1,
                    "errors": [],
                },
                "mixed_layout": {
                    "category": "mixed_layout",
                    "total_cases": 9,
                    "direction_correct": 7,
                    "direction_accuracy": 77.78,
                    "avg_kendall_tau": 0.78,
                    "avg_column_precision": 0.85,
                    "avg_inference_time_ms": 0.2,
                    "errors": [],
                },
                "horizontal_only": {
                    "category": "horizontal_only",
                    "total_cases": 9,
                    "direction_correct": 9,
                    "direction_accuracy": 100.0,
                    "avg_kendall_tau": 0.0,
                    "avg_column_precision": 0.0,
                    "avg_inference_time_ms": 0.08,
                    "errors": [],
                },
            },
            dataset="synthetic",
            target_direction_accuracy=90.0,
            target_ordering_tau=0.80,
            passed=True,
            timestamp="2026-03-12T00:00:00.000+00:00",
        )

    def test_plain_report_contains_header(self):
        result = self._make_result()
        report = format_benchmark_report(result)
        assert "CJK VERTICAL TEXT BENCHMARK" in report

    def test_plain_report_contains_metrics(self):
        result = self._make_result()
        report = format_benchmark_report(result)
        assert "93.33%" in report
        assert "0.8721" in report
        assert "PASS" in report

    def test_plain_report_contains_categories(self):
        result = self._make_result()
        report = format_benchmark_report(result)
        assert "vertical_only" in report
        assert "mixed_layout" in report
        assert "horizontal_only" in report

    def test_plain_report_with_empty_result(self):
        result = VerticalTextBenchmarkResult()
        report = format_benchmark_report(result)
        assert "CJK VERTICAL TEXT BENCHMARK" in report
        assert "0.00%" in report

    def test_plain_report_fail_result(self):
        result = self._make_result()
        result.passed = False
        report = format_benchmark_report(result)
        assert "FAIL" in report


class TestFormatMarkdownReport:
    def _make_result(self):
        return VerticalTextBenchmarkResult(
            total_cases=10,
            direction_accuracy=90.0,
            avg_kendall_tau=0.85,
            avg_column_precision=0.90,
            avg_inference_time_ms=0.2,
            min_inference_time_ms=0.1,
            max_inference_time_ms=0.5,
            p95_inference_time_ms=0.4,
            per_category={
                "vertical_only": {
                    "category": "vertical_only",
                    "total_cases": 4,
                    "direction_correct": 4,
                    "direction_accuracy": 100.0,
                    "avg_kendall_tau": 0.95,
                    "avg_column_precision": 1.0,
                    "avg_inference_time_ms": 0.15,
                    "errors": [],
                },
            },
            dataset="synthetic",
            target_direction_accuracy=90.0,
            target_ordering_tau=0.80,
            passed=True,
            timestamp="2026-03-12T00:00:00.000+00:00",
        )

    def test_markdown_contains_header(self):
        result = self._make_result()
        md = format_markdown_report(result)
        assert "# CJK Vertical Text Benchmark Report" in md

    def test_markdown_contains_summary_table(self):
        result = self._make_result()
        md = format_markdown_report(result)
        assert "| Direction Accuracy" in md
        assert "| Avg Kendall's Tau" in md

    def test_markdown_contains_category_table(self):
        result = self._make_result()
        md = format_markdown_report(result)
        assert "## Per-Category Breakdown" in md
        assert "| vertical_only" in md

    def test_markdown_contains_limitations(self):
        result = self._make_result()
        md = format_markdown_report(result)
        assert "## Known Limitations" in md

    def test_markdown_pass_status(self):
        result = self._make_result()
        md = format_markdown_report(result)
        assert "**Result**: PASS" in md

    def test_markdown_fail_status(self):
        result = self._make_result()
        result.passed = False
        md = format_markdown_report(result)
        assert "**Result**: FAIL" in md


# ---------------------------------------------------------------------------
# Tests: CategoryMetrics dataclass
# ---------------------------------------------------------------------------


class TestCategoryMetrics:
    def test_defaults(self):
        cat = CategoryMetrics()
        assert cat.category == ""
        assert cat.total_cases == 0
        assert cat.direction_correct == 0
        assert cat.direction_accuracy == 0.0
        assert cat.avg_kendall_tau == 0.0
        assert cat.avg_column_precision == 0.0
        assert cat.avg_inference_time_ms == 0.0
        assert cat.errors == []

    def test_custom_values(self):
        cat = CategoryMetrics(
            category="vertical_only",
            total_cases=10,
            direction_correct=9,
            direction_accuracy=90.0,
            avg_kendall_tau=0.85,
        )
        assert cat.category == "vertical_only"
        assert cat.total_cases == 10
        assert cat.direction_correct == 9


# ---------------------------------------------------------------------------
# Tests: VerticalTextBenchmarkResult dataclass
# ---------------------------------------------------------------------------


class TestVerticalTextBenchmarkResult:
    def test_defaults(self):
        r = VerticalTextBenchmarkResult()
        assert r.total_cases == 0
        assert r.direction_accuracy == 0.0
        assert r.avg_kendall_tau == 0.0
        assert r.avg_column_precision == 0.0
        assert r.per_category == {}
        assert r.passed is False

    def test_default_targets(self):
        r = VerticalTextBenchmarkResult()
        assert r.target_direction_accuracy == 90.0
        assert r.target_ordering_tau == 0.80


# ---------------------------------------------------------------------------
# Tests: JSON output round-trip
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_round_trip(self, tmp_path):
        from dataclasses import asdict

        result = run_synthetic_benchmark(sample_size=5)
        json_path = str(tmp_path / "results.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["total_cases"] == result.total_cases
        assert loaded["direction_accuracy"] == result.direction_accuracy
        assert loaded["dataset"] == "synthetic"
        assert "per_category" in loaded

    def test_markdown_output_file(self, tmp_path):
        result = run_synthetic_benchmark(sample_size=5)
        md_path = str(tmp_path / "report.md")

        md_report = format_markdown_report(result)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# CJK Vertical Text Benchmark Report" in content


# ---------------------------------------------------------------------------
# Tests: Dry-run mode with corpus
# ---------------------------------------------------------------------------


class TestDryRunMode:
    def test_dry_run_empty_corpus(self, tmp_path):
        """Dry run on empty directory reports invalid."""
        result = validate_corpus_structure(str(tmp_path))
        assert result["valid"] is False

    def test_dry_run_valid_minimal_corpus(self, tmp_path):
        """Dry run on minimal valid corpus."""
        vert_dir = tmp_path / "vertical_only"
        vert_dir.mkdir()

        (vert_dir / "page1.png").write_bytes(b"image data")
        sidecar = {
            "direction": "vertical",
            "expected_order": ["col1_line1", "col1_line2"],
            "ocr_lines": [
                {
                    "text": "col1_line1",
                    "box": [[770, 50], [800, 50], [800, 250], [770, 250]],
                    "confidence": 0.95,
                },
                {
                    "text": "col1_line2",
                    "box": [[770, 270], [800, 270], [800, 470], [770, 470]],
                    "confidence": 0.93,
                },
            ],
            "page_width": 1000,
        }
        with open(vert_dir / "page1.expected.json", "w") as f:
            json.dump(sidecar, f)

        result = validate_corpus_structure(str(tmp_path))
        assert result["valid"] is True
        assert result["total_cases"] == 1

    def test_supported_image_extensions(self, tmp_path):
        """All supported image extensions are recognized."""
        vert_dir = tmp_path / "vertical_only"
        vert_dir.mkdir()

        for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"):
            stem = f"test_{ext.replace('.', '')}"
            (vert_dir / f"{stem}{ext}").write_bytes(b"image data")
            sidecar = {
                "direction": "vertical",
                "expected_order": ["text"],
                "ocr_lines": [
                    {"text": "text", "box": [[0, 0], [30, 0], [30, 200], [0, 200]]}
                ],
                "page_width": 1000,
            }
            with open(vert_dir / f"{stem}.expected.json", "w") as f:
                json.dump(sidecar, f)

        result = validate_corpus_structure(str(tmp_path))
        assert result["categories"]["vertical_only"]["image_count"] == 7
        assert result["categories"]["vertical_only"]["valid_pairs"] == 7


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_corpus_subdirs(self):
        assert CORPUS_SUBDIRS == ("vertical_only", "mixed_layout", "horizontal_only")

    def test_sidecar_suffix(self):
        assert SIDECAR_SUFFIX == ".expected.json"
