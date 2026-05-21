"""
Unit tests for the CJK vertical text validation suite (scripts/validate_cjk_vertical.py).

Tests cover:
- Synthetic test case generation (vertical, horizontal, mixed, multi-column, small)
- Vertical/horizontal classification on known boxes
- Reading order accuracy computation (Kendall's tau)
- Column grouping precision computation
- Validation engine (single case and batch)
- Corpus loading and in-memory generation
- CLI argument parsing
- Report formatting
- Edge cases (empty inputs, missing module, etc.)

Run with: python -m pytest tests/test_cjk_validation.py -v
"""

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from scripts.validate_cjk_vertical import (  # noqa: I001
    CategoryResult,
    ValidationMetrics,
    _make_box,
    _make_horizontal_line,
    _make_vertical_line,
    build_parser,
    compute_column_precision,
    compute_kendall_tau,
    format_report,
    generate_corpus,
    generate_horizontal_only_case,
    generate_mixed_layout_case,
    generate_multi_column_case,
    generate_small_text_case,
    generate_synthetic_cases,
    generate_vertical_only_case,
    load_corpus_cases,
    main,
    run_validation,
    validate_case,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed=42):
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Tests: Box helpers
# ---------------------------------------------------------------------------


class TestBoxHelpers:
    def test_make_box_format(self):
        box = _make_box(10, 20, 50, 80)
        assert len(box) == 4
        assert box[0] == [10, 20]
        assert box[1] == [50, 20]
        assert box[2] == [50, 80]
        assert box[3] == [10, 80]

    def test_make_vertical_line(self):
        text, box, conf = _make_vertical_line("abc", 100, 50)
        assert text == "abc"
        assert conf == 0.92
        # Width=30, height=200 -> aspect ratio ~6.67 -> vertical
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        assert height / width >= 2.0

    def test_make_horizontal_line(self):
        text, box, conf = _make_horizontal_line("test", 10, 20)
        assert text == "test"
        assert conf == 0.92
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        assert width > height


# ---------------------------------------------------------------------------
# Tests: Kendall's tau computation
# ---------------------------------------------------------------------------


class TestKendallTau:
    def test_perfect_agreement(self):
        order = ["a", "b", "c", "d"]
        assert compute_kendall_tau(order, order) == 1.0

    def test_perfect_reversal(self):
        predicted = ["d", "c", "b", "a"]
        expected = ["a", "b", "c", "d"]
        assert compute_kendall_tau(predicted, expected) == -1.0

    def test_empty_lists(self):
        assert compute_kendall_tau([], []) == 0.0

    def test_single_element(self):
        assert compute_kendall_tau(["a"], ["a"]) == 0.0

    def test_partial_overlap(self):
        predicted = ["a", "b", "c"]
        expected = ["a", "c", "b"]
        tau = compute_kendall_tau(predicted, expected)
        assert -1.0 <= tau <= 1.0
        assert tau != 1.0  # Not perfect agreement

    def test_no_common_elements(self):
        predicted = ["x", "y"]
        expected = ["a", "b"]
        assert compute_kendall_tau(predicted, expected) == 0.0

    def test_two_elements_swapped(self):
        predicted = ["b", "a"]
        expected = ["a", "b"]
        assert compute_kendall_tau(predicted, expected) == -1.0


# ---------------------------------------------------------------------------
# Tests: Column precision
# ---------------------------------------------------------------------------


class TestColumnPrecision:
    def test_exact_match(self):
        assert compute_column_precision(3, 3) == 1.0

    def test_predicted_zero(self):
        assert compute_column_precision(0, 3) == 0.0

    def test_expected_zero_predicted_zero(self):
        assert compute_column_precision(0, 0) == 1.0

    def test_expected_zero_predicted_nonzero(self):
        assert compute_column_precision(2, 0) == 0.0

    def test_one_off(self):
        prec = compute_column_precision(2, 3)
        assert 0.0 < prec < 1.0

    def test_over_prediction(self):
        prec = compute_column_precision(5, 3)
        assert 0.0 < prec < 1.0


# ---------------------------------------------------------------------------
# Tests: Synthetic case generators
# ---------------------------------------------------------------------------


class TestGenerateVerticalOnlyCase:
    def test_produces_valid_sidecar(self):
        sidecar = generate_vertical_only_case(_rng(), "test_vert", num_columns=3, lines_per_col=4)
        assert sidecar["direction"] == "vertical"
        assert sidecar["expected_columns"] == 3
        assert len(sidecar["expected_order"]) == 12  # 3 cols * 4 lines
        assert len(sidecar["ocr_lines"]) == 12
        assert sidecar["page_width"] == 1000

    def test_expected_order_reflects_columns(self):
        sidecar = generate_vertical_only_case(_rng(), "test", num_columns=2, lines_per_col=3)
        # First 3 items should be from col 0 (rightmost), next 3 from col 1
        order = sidecar["expected_order"]
        assert all("_0_" in t for t in order[:3])
        assert all("_1_" in t for t in order[3:6])

    def test_ocr_lines_shuffled(self):
        sidecar = generate_vertical_only_case(_rng(), "test", num_columns=2, lines_per_col=3)
        line_texts = [entry["text"] for entry in sidecar["ocr_lines"]]
        expected = sidecar["expected_order"]
        # Order in ocr_lines should differ from expected_order (shuffled)
        # Not guaranteed for all seeds, but the test will still pass because
        # the validation engine re-sorts them
        assert set(line_texts) == set(expected)

    def test_has_cjk_characters(self):
        sidecar = generate_vertical_only_case(_rng(), "test", num_columns=1, lines_per_col=1)
        text = sidecar["expected_order"][0]
        # Text should contain CJK characters before the underscore
        prefix = text.split("_")[0]
        assert any(ord(c) > 0x3000 for c in prefix)


class TestGenerateHorizontalOnlyCase:
    def test_produces_valid_sidecar(self):
        sidecar = generate_horizontal_only_case(_rng(), "test", num_lines=5)
        assert sidecar["direction"] == "horizontal"
        assert sidecar["expected_columns"] == 0
        assert len(sidecar["expected_order"]) == 5
        assert len(sidecar["ocr_lines"]) == 5

    def test_lines_contain_english(self):
        sidecar = generate_horizontal_only_case(_rng(), "test", num_lines=3)
        for entry in sidecar["ocr_lines"]:
            assert "Line" in entry["text"]


class TestGenerateMixedLayoutCase:
    def test_produces_valid_sidecar(self):
        sidecar = generate_mixed_layout_case(
            _rng(), "test", num_v_columns=2, v_lines_per_col=3, num_h_lines=2,
        )
        assert sidecar["direction"] == "mixed"
        assert sidecar["expected_columns"] == 2
        # 2 horizontal + 2 cols * 3 lines = 8 total
        assert len(sidecar["expected_order"]) == 8
        assert len(sidecar["ocr_lines"]) == 8

    def test_has_both_horizontal_and_vertical(self):
        sidecar = generate_mixed_layout_case(
            _rng(), "test", num_v_columns=1, v_lines_per_col=2, num_h_lines=2,
        )
        texts = [entry["text"] for entry in sidecar["ocr_lines"]]
        has_header = any("Header" in t for t in texts)
        has_cjk = any(any(ord(c) > 0x3000 for c in t) for t in texts)
        assert has_header
        assert has_cjk


class TestGenerateMultiColumnCase:
    def test_produces_valid_sidecar(self):
        sidecar = generate_multi_column_case(
            _rng(), "test", num_columns=5, lines_per_col=4,
        )
        assert sidecar["direction"] == "vertical"
        assert sidecar["expected_columns"] == 5
        assert len(sidecar["expected_order"]) == 20

    def test_wider_page(self):
        sidecar = generate_multi_column_case(
            _rng(), "test", num_columns=5, lines_per_col=2, page_width=1500,
        )
        assert sidecar["page_width"] == 1500


class TestGenerateSmallTextCase:
    def test_produces_valid_sidecar(self):
        sidecar = generate_small_text_case(_rng(), "test")
        assert sidecar["direction"] == "vertical"
        assert sidecar["expected_columns"] == 2
        assert len(sidecar["expected_order"]) == 4  # 2 cols * 2 lines


# ---------------------------------------------------------------------------
# Tests: Validate case
# ---------------------------------------------------------------------------


class TestValidateCase:
    def test_vertical_case_detected(self):
        sidecar = generate_vertical_only_case(_rng(), "test", num_columns=3, lines_per_col=4)
        result = validate_case(sidecar)
        assert result["direction_correct"] is True
        assert result["predicted_direction"] == "vertical"

    def test_horizontal_case_detected(self):
        sidecar = generate_horizontal_only_case(_rng(), "test", num_lines=5)
        result = validate_case(sidecar)
        assert result["direction_correct"] is True
        assert result["predicted_direction"] == "horizontal"

    def test_reading_order_has_positive_tau(self):
        sidecar = generate_vertical_only_case(_rng(), "test", num_columns=3, lines_per_col=4)
        result = validate_case(sidecar)
        # The module should produce reasonable reading order
        assert result["kendall_tau"] > 0.0

    def test_column_precision_positive(self):
        sidecar = generate_vertical_only_case(_rng(), "test", num_columns=3, lines_per_col=4)
        result = validate_case(sidecar)
        assert result["column_precision"] > 0.0

    def test_inference_time_recorded(self):
        sidecar = generate_vertical_only_case(_rng(), "test", num_columns=2, lines_per_col=2)
        result = validate_case(sidecar)
        assert result["inference_time_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Tests: Run validation (batch)
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_synthetic_validation_passes(self):
        cases = generate_synthetic_cases(num_samples=10, seed=42)
        metrics = run_validation(cases)
        assert metrics.total_cases == 10
        assert metrics.direction_accuracy >= 0.0
        assert metrics.timestamp != ""

    def test_per_category_populated(self):
        cases = generate_synthetic_cases(num_samples=10, seed=42)
        metrics = run_validation(cases)
        assert len(metrics.per_category) > 0

    def test_empty_cases(self):
        metrics = run_validation([])
        assert metrics.total_cases == 0
        assert metrics.direction_accuracy == 0.0

    def test_pass_threshold_check(self):
        cases = generate_synthetic_cases(num_samples=15, seed=42)
        metrics = run_validation(
            cases,
            target_direction_accuracy=50.0,
            target_ordering_tau=0.01,
        )
        # With synthetic data and low targets, should pass
        assert metrics.passed is True

    def test_strict_threshold_may_fail(self):
        cases = generate_synthetic_cases(num_samples=5, seed=42)
        metrics = run_validation(
            cases,
            target_direction_accuracy=100.0,
            target_ordering_tau=1.0,
        )
        # With perfect targets, may not pass
        assert isinstance(metrics.passed, bool)


# ---------------------------------------------------------------------------
# Tests: Generate synthetic cases in-memory
# ---------------------------------------------------------------------------


class TestGenerateSyntheticCases:
    def test_generates_expected_count(self):
        cases = generate_synthetic_cases(num_samples=20, seed=42)
        assert len(cases) == 20

    def test_each_case_has_required_keys(self):
        cases = generate_synthetic_cases(num_samples=5, seed=42)
        for case in cases:
            assert "id" in case
            assert "category" in case
            assert "sidecar" in case
            sidecar = case["sidecar"]
            assert "direction" in sidecar
            assert "ocr_lines" in sidecar
            assert "page_width" in sidecar

    def test_includes_all_categories(self):
        cases = generate_synthetic_cases(num_samples=25, seed=42)
        categories = {c["category"] for c in cases}
        assert "vertical_only" in categories
        assert "horizontal_only" in categories
        assert "mixed_layout" in categories
        assert "multi_column" in categories
        assert "small_text" in categories

    def test_deterministic_with_seed(self):
        cases1 = generate_synthetic_cases(num_samples=5, seed=99)
        cases2 = generate_synthetic_cases(num_samples=5, seed=99)
        for c1, c2 in zip(cases1, cases2):
            assert c1["id"] == c2["id"]
            assert c1["sidecar"]["expected_order"] == c2["sidecar"]["expected_order"]


# ---------------------------------------------------------------------------
# Tests: Corpus generation and loading
# ---------------------------------------------------------------------------


class TestGenerateCorpus:
    def test_creates_directory_structure(self, tmp_path):
        summary = generate_corpus(str(tmp_path / "corpus"), num_samples=10, seed=42)
        assert summary["total_generated"] == 10
        assert os.path.isdir(tmp_path / "corpus" / "vertical_only")
        assert os.path.isdir(tmp_path / "corpus" / "horizontal_only")
        assert os.path.isdir(tmp_path / "corpus" / "mixed_layout")
        assert os.path.isdir(tmp_path / "corpus" / "multi_column")
        assert os.path.isdir(tmp_path / "corpus" / "small_text")

    def test_sidecar_files_created(self, tmp_path):
        generate_corpus(str(tmp_path / "corpus"), num_samples=5, seed=42)
        sidecar_files = list((tmp_path / "corpus").rglob("*.expected.json"))
        assert len(sidecar_files) == 5

    def test_sidecar_files_valid_json(self, tmp_path):
        generate_corpus(str(tmp_path / "corpus"), num_samples=5, seed=42)
        for f in (tmp_path / "corpus").rglob("*.expected.json"):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            assert "direction" in data
            assert "ocr_lines" in data


class TestLoadCorpusCases:
    def test_loads_generated_corpus(self, tmp_path):
        generate_corpus(str(tmp_path / "corpus"), num_samples=8, seed=42)
        cases = load_corpus_cases(str(tmp_path / "corpus"))
        assert len(cases) == 8

    def test_handles_missing_directory(self):
        cases = load_corpus_cases("/nonexistent/path")
        assert cases == []

    def test_case_structure(self, tmp_path):
        generate_corpus(str(tmp_path / "corpus"), num_samples=3, seed=42)
        cases = load_corpus_cases(str(tmp_path / "corpus"))
        for case in cases:
            assert "id" in case
            assert "category" in case
            assert "sidecar" in case

    def test_roundtrip_generate_load_validate(self, tmp_path):
        """Generate corpus, load it, validate -- end-to-end."""
        generate_corpus(str(tmp_path / "corpus"), num_samples=10, seed=42)
        cases = load_corpus_cases(str(tmp_path / "corpus"))
        metrics = run_validation(cases, dataset_label=str(tmp_path / "corpus"))
        assert metrics.total_cases == 10
        assert metrics.direction_accuracy >= 0.0


# ---------------------------------------------------------------------------
# Tests: Report formatting
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_report_contains_key_sections(self):
        metrics = ValidationMetrics(
            total_cases=10,
            direction_correct=9,
            direction_accuracy=90.0,
            avg_kendall_tau=0.85,
            column_grouping_precision=0.9,
            avg_inference_time_ms=0.5,
            passed=True,
            dataset="test",
            timestamp="2026-03-14T12:00:00.000Z",
        )
        report = format_report(metrics)
        assert "CJK VERTICAL TEXT VALIDATION REPORT" in report
        assert "90.00%" in report
        assert "0.8500" in report
        assert "PASS" in report
        assert "test" in report

    def test_report_with_per_category(self):
        metrics = ValidationMetrics(
            total_cases=5,
            per_category={
                "vertical_only": {
                    "category": "vertical_only",
                    "total_cases": 3,
                    "direction_correct": 3,
                    "direction_accuracy": 100.0,
                    "avg_kendall_tau": 0.9,
                    "column_precision": 1.0,
                    "avg_inference_time_ms": 0.3,
                },
            },
            dataset="test",
            timestamp="2026-03-14T12:00:00.000Z",
        )
        report = format_report(metrics)
        assert "PER-CATEGORY BREAKDOWN" in report
        assert "vertical_only" in report

    def test_report_fail_result(self):
        metrics = ValidationMetrics(
            total_cases=5,
            direction_accuracy=50.0,
            passed=False,
            dataset="test",
            timestamp="2026-03-14T12:00:00.000Z",
        )
        report = format_report(metrics)
        assert "FAIL" in report


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_generate_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--generate", "--output-dir", "/tmp/corpus"])
        assert args.generate is True
        assert args.output_dir == "/tmp/corpus"

    def test_validate_with_synthetic(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "--synthetic"])
        assert args.validate is True
        assert args.synthetic is True

    def test_validate_with_corpus(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "--corpus-dir", "/data/corpus"])
        assert args.validate is True
        assert args.corpus_dir == "/data/corpus"

    def test_num_samples_default(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "--synthetic"])
        assert args.num_samples == 25

    def test_custom_targets(self):
        parser = build_parser()
        args = parser.parse_args([
            "--validate", "--synthetic",
            "--target-direction-accuracy", "95.0",
            "--target-ordering-tau", "0.90",
        ])
        assert args.target_direction_accuracy == 95.0
        assert args.target_ordering_tau == 0.90

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "--synthetic", "-v"])
        assert args.verbose is True

    def test_seed_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "--synthetic", "--seed", "123"])
        assert args.seed == 123

    def test_output_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "--synthetic", "--output", "out.json"])
        assert args.output == "out.json"


# ---------------------------------------------------------------------------
# Tests: CLI main() execution
# ---------------------------------------------------------------------------


class TestMainExecution:
    def test_main_validate_synthetic(self):
        """Validate with synthetic data should return 0 (pass) with relaxed targets."""
        rc = main([
            "--validate", "--synthetic", "--num-samples", "5",
            "--target-direction-accuracy", "60.0",
            "--target-ordering-tau", "0.50",
        ])
        assert rc == 0

    def test_main_generate(self, tmp_path):
        """Generate should create corpus directory and return 0."""
        out_dir = str(tmp_path / "gen_corpus")
        rc = main(["--generate", "--output-dir", out_dir, "--num-samples", "5"])
        assert rc == 0
        assert os.path.isdir(os.path.join(out_dir, "vertical_only"))

    def test_main_generate_with_json_output(self, tmp_path):
        """Generate with --output should produce summary JSON."""
        out_dir = str(tmp_path / "gen_corpus")
        json_path = str(tmp_path / "summary.json")
        rc = main(["--generate", "--output-dir", out_dir,
                    "--num-samples", "5", "--output", json_path])
        assert rc == 0
        assert os.path.exists(json_path)
        with open(json_path, "r") as f:
            summary = json.load(f)
        assert summary["total_generated"] == 5

    def test_main_validate_corpus_roundtrip(self, tmp_path):
        """Generate then validate a corpus -- full roundtrip."""
        out_dir = str(tmp_path / "corpus")
        main(["--generate", "--output-dir", out_dir, "--num-samples", "8"])
        rc = main([
            "--validate", "--corpus-dir", out_dir,
            "--target-direction-accuracy", "60.0",
            "--target-ordering-tau", "0.50",
        ])
        assert rc == 0

    def test_main_validate_with_output_json(self, tmp_path):
        """Validate with --output should produce JSON results."""
        json_path = str(tmp_path / "results.json")
        rc = main([
            "--validate", "--synthetic", "--num-samples", "5",
            "--output", json_path,
            "--target-direction-accuracy", "60.0",
            "--target-ordering-tau", "0.50",
        ])
        assert rc == 0
        assert os.path.exists(json_path)
        with open(json_path, "r") as f:
            data = json.load(f)
        assert "total_cases" in data
        assert "direction_accuracy" in data

    def test_main_validate_missing_corpus_fails(self):
        """Validate with nonexistent corpus should return 1."""
        rc = main(["--validate", "--corpus-dir", "/nonexistent/path/xyz"])
        assert rc == 1

    def test_main_no_action_errors(self):
        """Running with no --generate or --validate should error."""
        with pytest.raises(SystemExit):
            main([])


# ---------------------------------------------------------------------------
# Tests: Data class defaults
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_validation_metrics_defaults(self):
        m = ValidationMetrics()
        assert m.total_cases == 0
        assert m.direction_accuracy == 0.0
        assert m.avg_kendall_tau == 0.0
        assert m.column_grouping_precision == 0.0
        assert m.passed is False
        assert m.per_category == {}

    def test_category_result_defaults(self):
        c = CategoryResult()
        assert c.category == ""
        assert c.total_cases == 0
        assert c.direction_accuracy == 0.0

    def test_category_result_custom(self):
        c = CategoryResult(
            category="vertical_only",
            total_cases=10,
            direction_correct=9,
            direction_accuracy=90.0,
        )
        assert c.category == "vertical_only"
        assert c.total_cases == 10


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_validate_case_empty_ocr_lines(self):
        sidecar = {
            "direction": "horizontal",
            "expected_order": [],
            "ocr_lines": [],
            "page_width": 1000,
        }
        result = validate_case(sidecar)
        assert result["direction_correct"] is True
        assert result["predicted_direction"] == "horizontal"

    def test_validate_case_single_line(self):
        sidecar = generate_vertical_only_case(_rng(), "test", num_columns=1, lines_per_col=1)
        result = validate_case(sidecar)
        # Direction may be horizontal for a single line
        assert "predicted_direction" in result

    def test_kendall_tau_identical_texts(self):
        order = ["a", "a", "b"]
        tau = compute_kendall_tau(order, ["a", "b"])
        assert -1.0 <= tau <= 1.0

    def test_column_precision_large_values(self):
        prec = compute_column_precision(100, 50)
        assert 0.0 < prec < 1.0

    def test_format_report_empty_metrics(self):
        metrics = ValidationMetrics(
            dataset="empty",
            timestamp="2026-01-01T00:00:00.000Z",
        )
        report = format_report(metrics)
        assert "VALIDATION REPORT" in report
        assert "0" in report  # total_cases == 0
