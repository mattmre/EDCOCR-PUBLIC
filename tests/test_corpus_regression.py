"""Tests for scripts/corpus_regression.py — corpus regression framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from scripts.corpus_regression import (
    _compute_aggregate,
    _stem_from_gt,
    _stratify,
    compute_corpus_metrics,
    compute_document_metrics,
    detect_regressions,
    format_markdown_report,
    format_text_summary,
    load_baseline,
    main,
    match_documents,
    save_baseline,
)

# Path to the test corpus fixtures
_FIXTURES = _PROJECT_ROOT / "tests" / "fixtures" / "corpus"
_GT_DIR = _FIXTURES / "ground-truth"
_OUT_DIR = _FIXTURES / "output"
_LANG_MAP = _FIXTURES / "language-map.json"
_CAT_MAP = _FIXTURES / "category-map.json"


# ---------------------------------------------------------------------------
# Document matching
# ---------------------------------------------------------------------------


class TestDocumentMatching:
    """Tests for output-to-ground-truth file matching."""

    def test_match_fixture_corpus(self):
        pairs = match_documents(_OUT_DIR, _GT_DIR)
        doc_ids = [p[0] for p in pairs]
        assert "sample-clean" in doc_ids
        assert "sample-degraded" in doc_ids

    def test_match_returns_sorted(self):
        pairs = match_documents(_OUT_DIR, _GT_DIR)
        doc_ids = [p[0] for p in pairs]
        assert doc_ids == sorted(doc_ids)

    def test_match_paths_are_files(self):
        pairs = match_documents(_OUT_DIR, _GT_DIR)
        for _, out_path, gt_path in pairs:
            assert out_path.is_file()
            assert gt_path.is_file()

    def test_stem_from_gt_double_extension(self):
        assert _stem_from_gt(Path("sample-clean.gt.txt")) == "sample-clean"

    def test_stem_from_gt_single_extension(self):
        assert _stem_from_gt(Path("sample-clean.txt")) == "sample-clean"

    def test_no_matches_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        pairs = match_documents(empty, _GT_DIR)
        assert pairs == []


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


class TestDocumentMetrics:
    """Tests for single-document metric computation."""

    def test_perfect_document(self):
        text = "The quick brown fox."
        metrics = compute_document_metrics(text, text)
        assert metrics["cer"] == 0.0
        assert metrics["wer"] == 0.0
        assert metrics["line_accuracy"] == 1.0

    def test_some_errors(self):
        gt = "hello world"
        out = "helo world"
        metrics = compute_document_metrics(out, gt)
        assert metrics["cer"] > 0.0
        assert metrics["wer"] > 0.0


class TestCorpusMetrics:
    """Tests for full corpus metric computation."""

    def test_fixture_corpus_has_all_documents(self):
        results = compute_corpus_metrics(_OUT_DIR, _GT_DIR)
        assert len(results["documents"]) == 5

    def test_aggregate_present(self):
        results = compute_corpus_metrics(_OUT_DIR, _GT_DIR)
        agg = results["aggregate"]
        assert "mean_cer" in agg
        assert "mean_wer" in agg
        assert "mean_line_accuracy" in agg
        assert agg["document_count"] == 5

    def test_language_stratification(self):
        lang_map = json.loads(_LANG_MAP.read_text(encoding="utf-8"))
        results = compute_corpus_metrics(_OUT_DIR, _GT_DIR, language_map=lang_map)
        assert "en" in results["by_language"]
        assert results["by_language"]["en"]["document_count"] == 3
        assert "multilingual" in results["by_language"]
        assert results["by_language"]["multilingual"]["document_count"] == 2

    def test_category_stratification(self):
        cat_map = json.loads(_CAT_MAP.read_text(encoding="utf-8"))
        results = compute_corpus_metrics(_OUT_DIR, _GT_DIR, category_map=cat_map)
        assert "printed_text" in results["by_category"]
        assert "degraded_scan" in results["by_category"]

    def test_empty_corpus_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        results = compute_corpus_metrics(empty, _GT_DIR)
        assert results["documents"] == {}


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


class TestRegressionDetection:
    """Tests for regression detection against baseline."""

    def _make_results(self, cer=0.05, wer=0.10, la=0.90):
        return {
            "documents": {
                "doc1": {"cer": cer, "wer": wer, "line_accuracy": la},
            },
        }

    def test_no_regression_within_threshold(self):
        baseline = self._make_results(cer=0.05, wer=0.10, la=0.90)
        current = self._make_results(cer=0.06, wer=0.11, la=0.89)
        regressions = detect_regressions(current, baseline, threshold=0.02)
        assert regressions == []

    def test_cer_regression_detected(self):
        baseline = self._make_results(cer=0.05)
        current = self._make_results(cer=0.10)
        regressions = detect_regressions(current, baseline, threshold=0.02)
        cer_regs = [r for r in regressions if r["metric"] == "cer"]
        assert len(cer_regs) == 1
        assert cer_regs[0]["delta"] > 0

    def test_wer_regression_detected(self):
        baseline = self._make_results(wer=0.10)
        current = self._make_results(wer=0.20)
        regressions = detect_regressions(current, baseline, threshold=0.02)
        wer_regs = [r for r in regressions if r["metric"] == "wer"]
        assert len(wer_regs) == 1

    def test_line_accuracy_regression_detected(self):
        baseline = self._make_results(la=0.95)
        current = self._make_results(la=0.80)
        regressions = detect_regressions(current, baseline, threshold=0.02)
        la_regs = [r for r in regressions if r["metric"] == "line_accuracy"]
        assert len(la_regs) == 1
        assert la_regs[0]["delta"] < 0  # negative = got worse

    def test_new_document_not_flagged(self):
        """A document not in the baseline is not a regression."""
        baseline = {"documents": {"old_doc": {"cer": 0.05, "wer": 0.10, "line_accuracy": 0.90}}}
        current = {"documents": {"new_doc": {"cer": 0.50, "wer": 0.50, "line_accuracy": 0.50}}}
        regressions = detect_regressions(current, baseline, threshold=0.02)
        assert regressions == []

    def test_exact_threshold_not_flagged(self):
        """Delta exactly equal to threshold is not a regression (must exceed).

        Uses 0.25/0.5 to avoid IEEE-754 representation artefacts that
        occur with values like 0.07-0.05.
        """
        baseline = self._make_results(cer=0.25)
        current = self._make_results(cer=0.5)
        regressions = detect_regressions(current, baseline, threshold=0.25)
        assert regressions == []

    def test_custom_threshold(self):
        baseline = self._make_results(cer=0.05)
        current = self._make_results(cer=0.06)
        # With tight threshold, this is a regression
        regressions = detect_regressions(current, baseline, threshold=0.005)
        cer_regs = [r for r in regressions if r["metric"] == "cer"]
        assert len(cer_regs) == 1


# ---------------------------------------------------------------------------
# Baseline save/load
# ---------------------------------------------------------------------------


class TestBaselineManagement:
    """Tests for baseline persistence."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "baseline.json"
        data = {"version": "1.0.0", "documents": {"d1": {"cer": 0.01}}}
        save_baseline(data, path)
        loaded = load_baseline(path)
        assert loaded == data

    def test_load_nonexistent(self, tmp_path):
        path = tmp_path / "does-not-exist.json"
        assert load_baseline(path) is None

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "baseline.json"
        save_baseline({"test": True}, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


class TestOutputFormatters:
    """Tests for text and markdown output."""

    @pytest.fixture()
    def sample_results(self):
        return {
            "version": "1.2.0",
            "timestamp": "2026-03-26T00:00:00+00:00",
            "documents": {
                "doc1": {"cer": 0.02, "wer": 0.05, "line_accuracy": 0.95, "language": "en"},
            },
            "aggregate": {"mean_cer": 0.02, "mean_wer": 0.05, "mean_line_accuracy": 0.95, "document_count": 1},
            "by_language": {"en": {"mean_cer": 0.02, "mean_wer": 0.05, "mean_line_accuracy": 0.95, "document_count": 1}},
            "by_category": {},
        }

    def test_text_summary_contains_header(self, sample_results):
        text = format_text_summary(sample_results)
        assert "Corpus Regression Report" in text

    def test_text_summary_contains_metrics(self, sample_results):
        text = format_text_summary(sample_results)
        assert "doc1" in text
        assert "0.0200" in text

    def test_text_summary_with_regressions(self, sample_results):
        regs = [{"doc_id": "doc1", "metric": "cer", "baseline_value": 0.01, "current_value": 0.05, "delta": 0.04}]
        text = format_text_summary(sample_results, regressions=regs)
        assert "REGRESSIONS DETECTED" in text

    def test_text_summary_no_regressions(self, sample_results):
        text = format_text_summary(sample_results, regressions=[])
        assert "No regressions detected" in text

    def test_markdown_report_structure(self, sample_results):
        md = format_markdown_report(sample_results)
        assert md.startswith("# Corpus Regression Report")
        assert "| Document |" in md

    def test_markdown_report_with_regressions(self, sample_results):
        regs = [{"doc_id": "doc1", "metric": "cer", "baseline_value": 0.01, "current_value": 0.05, "delta": 0.04}]
        md = format_markdown_report(sample_results, regressions=regs)
        assert "## Regressions" in md
        assert "doc1" in md


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the CLI entry point."""

    def test_basic_run(self):
        rc = main([
            "--corpus-dir", str(_OUT_DIR),
            "--ground-truth-dir", str(_GT_DIR),
        ])
        assert rc == 0

    def test_json_output(self, capsys):
        rc = main([
            "--corpus-dir", str(_OUT_DIR),
            "--ground-truth-dir", str(_GT_DIR),
            "--json",
        ])
        assert rc == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "documents" in data
        assert "aggregate" in data

    def test_save_and_compare_baseline(self, tmp_path):
        baseline_path = tmp_path / "baseline.json"
        # Save
        rc = main([
            "--corpus-dir", str(_OUT_DIR),
            "--ground-truth-dir", str(_GT_DIR),
            "--save-baseline",
            "--baseline-path", str(baseline_path),
        ])
        assert rc == 0
        assert baseline_path.exists()

        # Compare (no regressions against self)
        rc = main([
            "--corpus-dir", str(_OUT_DIR),
            "--ground-truth-dir", str(_GT_DIR),
            "--compare-baseline",
            "--baseline-path", str(baseline_path),
        ])
        assert rc == 0

    def test_markdown_report_written(self, tmp_path):
        report_path = tmp_path / "report.md"
        rc = main([
            "--corpus-dir", str(_OUT_DIR),
            "--ground-truth-dir", str(_GT_DIR),
            "--report", str(report_path),
        ])
        assert rc == 0
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "# Corpus Regression Report" in content

    def test_with_language_and_category_maps(self, capsys):
        rc = main([
            "--corpus-dir", str(_OUT_DIR),
            "--ground-truth-dir", str(_GT_DIR),
            "--language-map", str(_LANG_MAP),
            "--category-map", str(_CAT_MAP),
            "--json",
        ])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "en" in data["by_language"]
        assert "printed_text" in data["by_category"]

    def test_invalid_corpus_dir(self, tmp_path):
        rc = main([
            "--corpus-dir", str(tmp_path / "nonexistent"),
            "--ground-truth-dir", str(_GT_DIR),
        ])
        assert rc == 1

    def test_compare_without_baseline(self, tmp_path, capsys):
        """Compare flag with no baseline file should warn but not fail."""
        rc = main([
            "--corpus-dir", str(_OUT_DIR),
            "--ground-truth-dir", str(_GT_DIR),
            "--compare-baseline",
            "--baseline-path", str(tmp_path / "missing.json"),
        ])
        assert rc == 0  # no regressions since no baseline exists


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Tests for internal helper functions."""

    def test_compute_aggregate_empty(self):
        assert _compute_aggregate({}) == {}

    def test_compute_aggregate_single(self):
        docs = {"d": {"cer": 0.1, "wer": 0.2, "line_accuracy": 0.8}}
        agg = _compute_aggregate(docs)
        assert agg["mean_cer"] == pytest.approx(0.1)
        assert agg["document_count"] == 1

    def test_stratify_empty(self):
        assert _stratify({}, "language") == {}

    def test_stratify_with_data(self):
        docs = {
            "d1": {"cer": 0.1, "wer": 0.2, "line_accuracy": 0.8, "language": "en"},
            "d2": {"cer": 0.2, "wer": 0.3, "line_accuracy": 0.7, "language": "en"},
        }
        result = _stratify(docs, "language")
        assert "en" in result
        assert result["en"]["document_count"] == 2
        assert result["en"]["mean_cer"] == pytest.approx(0.15)
