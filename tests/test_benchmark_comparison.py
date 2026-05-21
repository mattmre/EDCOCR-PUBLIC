"""Tests for scripts/benchmark_comparison.py.

Covers PageMetrics, DocumentMetrics, AccuracyMetrics, BenchmarkResult
dataclasses, the _percentile helper, edit-distance computation, accuracy
computation, comparison-table generation, save/load round-trip, and report
generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmark_comparison import (
    AccuracyMetrics,
    BenchmarkResult,
    BenchmarkSuite,
    DocumentCategory,
    DocumentMetrics,
    PageMetrics,
    _build_run_root,
    _percentile,
    build_output_rel_stem,
    compute_accuracy,
    compute_edit_distance,
    discover_documents,
    execute_benchmark_suite,
    generate_comparison_table,
    generate_report,
    get_text_output_path,
    load_result,
    run_pipeline_for_document,
    save_result,
)

# ---------------------------------------------------------------------------
# Helper to build a minimal BenchmarkResult with documents
# ---------------------------------------------------------------------------


def _make_result(
    suite="standard",
    n_docs=3,
    pages_per_doc=5,
    time_per_doc_ms=1200.0,
    confidence=0.92,
):
    """Return a BenchmarkResult populated with *n_docs* documents."""
    docs = []
    for i in range(n_docs):
        pages = [
            PageMetrics(
                page_number=p + 1,
                processing_time_ms=time_per_doc_ms / pages_per_doc,
                char_count=500,
                word_count=80,
                confidence=confidence,
            )
            for p in range(pages_per_doc)
        ]
        doc = DocumentMetrics(
            document_id=f"doc-{i}",
            filename=f"file_{i}.pdf",
            category="printed_text",
            total_pages=pages_per_doc,
            total_processing_time_ms=time_per_doc_ms,
            pages=pages,
        )
        doc.compute_aggregates()
        docs.append(doc)

    result = BenchmarkResult(
        suite=suite,
        timestamp="20250101_120000",
        pipeline_version="1.0.0",
        documents=docs,
    )
    result.compute_summary()
    return result


# ---------------------------------------------------------------------------
# BenchmarkSuite enum
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    """Tests for the BenchmarkSuite enum."""

    def test_values(self):
        assert BenchmarkSuite.STANDARD.value == "standard"
        assert BenchmarkSuite.ACCURACY.value == "accuracy"
        assert BenchmarkSuite.STRESS.value == "stress"
        assert BenchmarkSuite.COLD_START.value == "cold_start"

    def test_member_count(self):
        assert len(BenchmarkSuite) == 4


# ---------------------------------------------------------------------------
# DocumentCategory enum
# ---------------------------------------------------------------------------


class TestDocumentCategory:
    """Tests for the DocumentCategory enum."""

    def test_values(self):
        assert DocumentCategory.PRINTED_TEXT.value == "printed_text"
        assert DocumentCategory.HANDWRITTEN.value == "handwritten"
        assert DocumentCategory.MIXED_LAYOUT.value == "mixed_layout"
        assert DocumentCategory.TABLE_HEAVY.value == "table_heavy"
        assert DocumentCategory.SCANNED_LOW_QUALITY.value == "scanned_low_quality"
        assert DocumentCategory.MULTILINGUAL.value == "multilingual"
        assert DocumentCategory.LARGE_DOCUMENT.value == "large_document"

    def test_member_count(self):
        assert len(DocumentCategory) == 7


# ---------------------------------------------------------------------------
# PageMetrics
# ---------------------------------------------------------------------------


class TestPageMetrics:
    """Tests for the PageMetrics dataclass."""

    def test_creation(self):
        pm = PageMetrics(
            page_number=1,
            processing_time_ms=250.5,
            char_count=1000,
            word_count=200,
        )
        assert pm.page_number == 1
        assert pm.processing_time_ms == 250.5
        assert pm.char_count == 1000
        assert pm.word_count == 200

    def test_defaults(self):
        pm = PageMetrics(page_number=1, processing_time_ms=0, char_count=0, word_count=0)
        assert pm.confidence == 0.0
        assert pm.dpi == 300
        assert pm.engine == "paddle"

    def test_custom_engine(self):
        pm = PageMetrics(
            page_number=2,
            processing_time_ms=100,
            char_count=50,
            word_count=10,
            engine="easyocr",
            dpi=600,
            confidence=0.95,
        )
        assert pm.engine == "easyocr"
        assert pm.dpi == 600
        assert pm.confidence == 0.95


# ---------------------------------------------------------------------------
# DocumentMetrics
# ---------------------------------------------------------------------------


class TestDocumentMetrics:
    """Tests for DocumentMetrics.compute_aggregates."""

    def test_compute_aggregates_basic(self):
        pages = [
            PageMetrics(1, 100, 500, 80, confidence=0.9),
            PageMetrics(2, 200, 600, 90, confidence=0.8),
        ]
        dm = DocumentMetrics(
            document_id="d1",
            filename="test.pdf",
            category="printed_text",
            total_pages=2,
            total_processing_time_ms=300.0,
            pages=pages,
        )
        dm.compute_aggregates()

        assert dm.total_chars == 1100
        assert dm.total_words == 170
        assert dm.avg_confidence == (0.9 + 0.8) / 2
        assert dm.throughput_pages_per_minute > 0

    def test_compute_aggregates_no_pages(self):
        dm = DocumentMetrics(
            document_id="d2",
            filename="empty.pdf",
            category="printed_text",
            total_pages=0,
            total_processing_time_ms=0,
        )
        dm.compute_aggregates()
        assert dm.total_chars == 0
        assert dm.total_words == 0

    def test_compute_aggregates_zero_confidence(self):
        pages = [PageMetrics(1, 100, 500, 80, confidence=0.0)]
        dm = DocumentMetrics(
            document_id="d3",
            filename="low.pdf",
            category="scanned_low_quality",
            total_pages=1,
            total_processing_time_ms=100.0,
            pages=pages,
        )
        dm.compute_aggregates()
        assert dm.avg_confidence == 0.0

    def test_throughput_calculation(self):
        pages = [PageMetrics(1, 30000, 500, 80, confidence=0.9)]
        dm = DocumentMetrics(
            document_id="d4",
            filename="slow.pdf",
            category="large_document",
            total_pages=1,
            total_processing_time_ms=60000.0,  # exactly 1 minute
            pages=pages,
        )
        dm.compute_aggregates()
        assert dm.throughput_pages_per_minute == 1.0


# ---------------------------------------------------------------------------
# AccuracyMetrics
# ---------------------------------------------------------------------------


class TestAccuracyMetrics:
    """Tests for AccuracyMetrics CER/WER properties."""

    def test_cer_perfect(self):
        am = AccuracyMetrics(
            document_id="a1",
            character_accuracy=1.0,
            total_characters=100,
            character_errors=0,
        )
        assert am.cer == 0.0

    def test_cer_with_errors(self):
        am = AccuracyMetrics(
            document_id="a2",
            total_characters=200,
            character_errors=20,
        )
        assert am.cer == 20 / 200

    def test_cer_zero_chars(self):
        am = AccuracyMetrics(document_id="a3", total_characters=0, character_errors=0)
        assert am.cer == 1.0

    def test_wer_perfect(self):
        am = AccuracyMetrics(
            document_id="a4",
            word_accuracy=1.0,
            total_words=50,
            word_errors=0,
        )
        assert am.wer == 0.0

    def test_wer_with_errors(self):
        am = AccuracyMetrics(
            document_id="a5",
            total_words=100,
            word_errors=10,
        )
        assert am.wer == 10 / 100

    def test_wer_zero_words(self):
        am = AccuracyMetrics(document_id="a6", total_words=0, word_errors=0)
        assert am.wer == 1.0


# ---------------------------------------------------------------------------
# BenchmarkResult.compute_summary
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Tests for BenchmarkResult.compute_summary."""

    def test_compute_summary_basic(self):
        result = _make_result(n_docs=3, pages_per_doc=5, time_per_doc_ms=1200)
        s = result.summary
        assert s["total_documents"] == 3
        assert s["total_pages"] == 15
        assert s["avg_time_per_doc_ms"] == 1200.0
        assert s["throughput_pages_per_minute"] > 0

    def test_compute_summary_empty(self):
        result = BenchmarkResult(suite="standard")
        result.compute_summary()
        assert result.summary == {}

    def test_compute_summary_with_accuracy(self):
        result = _make_result(n_docs=2)
        result.accuracy = [
            AccuracyMetrics("a1", character_accuracy=0.95, word_accuracy=0.90,
                            total_characters=100, total_words=20,
                            character_errors=5, word_errors=2),
            AccuracyMetrics("a2", character_accuracy=0.98, word_accuracy=0.96,
                            total_characters=200, total_words=40,
                            character_errors=4, word_errors=2),
        ]
        result.compute_summary()
        s = result.summary
        assert "avg_character_accuracy" in s
        assert "avg_word_accuracy" in s
        assert "avg_cer" in s
        assert "avg_wer" in s
        assert 0 < s["avg_character_accuracy"] <= 1.0
        assert 0 < s["avg_word_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    """Tests for the _percentile helper."""

    def test_empty_list(self):
        assert _percentile([], 95) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 95) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_multiple_values_p50(self):
        data = [10, 20, 30, 40, 50]
        result = _percentile(data, 50)
        assert result == 30.0

    def test_p95(self):
        data = list(range(1, 101))  # 1..100
        result = _percentile(data, 95)
        assert 94 < result <= 96

    def test_p99(self):
        data = list(range(1, 101))
        result = _percentile(data, 99)
        assert 98 < result <= 100

    def test_unsorted_input(self):
        data = [50, 10, 40, 20, 30]
        result = _percentile(data, 50)
        assert result == 30.0


# ---------------------------------------------------------------------------
# compute_edit_distance
# ---------------------------------------------------------------------------


class TestComputeEditDistance:
    """Tests for Levenshtein edit distance."""

    def test_identical_strings(self):
        assert compute_edit_distance("hello", "hello") == 0

    def test_insertion(self):
        assert compute_edit_distance("abc", "abcd") == 1

    def test_deletion(self):
        assert compute_edit_distance("abcd", "abc") == 1

    def test_substitution(self):
        assert compute_edit_distance("abc", "axc") == 1

    def test_empty_reference(self):
        assert compute_edit_distance("", "abc") == 3

    def test_empty_hypothesis(self):
        assert compute_edit_distance("abc", "") == 3

    def test_both_empty(self):
        assert compute_edit_distance("", "") == 0

    def test_complex_case(self):
        # "kitten" -> "sitting" requires 3 edits
        assert compute_edit_distance("kitten", "sitting") == 3


# ---------------------------------------------------------------------------
# compute_accuracy
# ---------------------------------------------------------------------------


class TestComputeAccuracy:
    """Tests for compute_accuracy."""

    def test_perfect_match(self):
        acc = compute_accuracy("hello world", "hello world")
        assert acc.character_accuracy == 1.0
        assert acc.word_accuracy == 1.0
        assert acc.character_errors == 0

    def test_partial_match(self):
        acc = compute_accuracy("hello world", "hallo world")
        assert 0.0 < acc.character_accuracy < 1.0
        assert acc.character_errors > 0

    def test_empty_reference(self):
        acc = compute_accuracy("", "some text")
        assert acc.total_characters == 0
        # character_accuracy should be clamped to 0
        assert acc.character_accuracy >= 0.0

    def test_empty_hypothesis(self):
        acc = compute_accuracy("hello world", "")
        assert acc.character_errors == len("hello world")

    def test_document_id_default(self):
        acc = compute_accuracy("a", "a")
        assert acc.document_id == ""


# ---------------------------------------------------------------------------
# generate_comparison_table
# ---------------------------------------------------------------------------


class TestGenerateComparisonTable:
    """Tests for markdown comparison table generation."""

    def test_contains_header(self):
        result = _make_result()
        table = generate_comparison_table(result)
        assert "# OCR Performance Benchmark Comparison" in table

    def test_contains_metrics(self):
        result = _make_result()
        table = generate_comparison_table(result)
        assert "Total Documents" in table
        assert "Total Pages" in table
        assert "Throughput" in table
        assert "EDCOCR" in table

    def test_contains_pipeline_version(self):
        result = _make_result()
        table = generate_comparison_table(result)
        assert "1.0.0" in table

    def test_accuracy_metrics_included(self):
        result = _make_result()
        result.accuracy = [
            AccuracyMetrics("a1", character_accuracy=0.95, word_accuracy=0.90,
                            total_characters=100, total_words=20,
                            character_errors=5, word_errors=2),
        ]
        result.compute_summary()
        table = generate_comparison_table(result)
        assert "Avg CER" in table
        assert "Avg WER" in table


# ---------------------------------------------------------------------------
# save_result / load_result round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadResult:
    """Tests for save_result and load_result."""

    def test_save_creates_file(self, tmp_path):
        result = _make_result()
        path = save_result(result, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_valid_json(self, tmp_path):
        result = _make_result()
        path = save_result(result, tmp_path)
        data = json.loads(path.read_text())
        assert data["suite"] == "standard"
        assert data["pipeline_version"] == "1.0.0"
        assert len(data["documents"]) == 3

    def test_roundtrip(self, tmp_path):
        original = _make_result(n_docs=2, pages_per_doc=3)
        path = save_result(original, tmp_path)
        loaded = load_result(path)

        assert loaded.suite == original.suite
        assert loaded.timestamp == original.timestamp
        assert loaded.pipeline_version == original.pipeline_version
        assert len(loaded.documents) == len(original.documents)
        assert loaded.documents[0].document_id == original.documents[0].document_id
        assert loaded.documents[0].total_pages == original.documents[0].total_pages

    def test_roundtrip_pages(self, tmp_path):
        original = _make_result(n_docs=1, pages_per_doc=4)
        path = save_result(original, tmp_path)
        loaded = load_result(path)
        assert len(loaded.documents[0].pages) == 4
        assert loaded.documents[0].pages[0].page_number == 1

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        result = _make_result()
        path = save_result(result, nested)
        assert nested.is_dir()
        assert path.exists()


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for generate_report."""

    def test_no_results(self, tmp_path):
        report = generate_report(tmp_path)
        assert report == "No benchmark results found."

    def test_with_result_file(self, tmp_path):
        result = _make_result()
        save_result(result, tmp_path)
        report = generate_report(tmp_path)
        assert "OCR Performance Benchmark Comparison" in report
        assert "Total Documents" in report

    def test_json_format(self, tmp_path):
        result = _make_result()
        save_result(result, tmp_path)
        report = generate_report(tmp_path, fmt="json")
        data = json.loads(report)
        assert data["suite"] == "standard"
        assert data["summary"]["total_documents"] == 3


class TestBenchmarkExecution:
    """Tests for real suite execution orchestration."""

    def test_discover_documents_filters_supported_extensions(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.pdf").write_bytes(b"%PDF-1.4")
        (input_dir / "b.png").write_bytes(b"not-a-real-png")
        (input_dir / "ignore.txt").write_text("skip")

        docs = discover_documents(input_dir)

        assert [doc.name for doc in docs] == ["a.pdf", "b.png"]

    def test_execute_standard_suite_writes_results(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "out"
        nested = input_dir / "cases"
        nested.mkdir(parents=True)
        doc = nested / "invoice.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")

        def fake_runner(document_path, source_dir, run_output_dir, pipeline_script):
            txt_path = get_text_output_path(document_path, source_dir, run_output_dir)
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text("invoice total due", encoding="utf-8")
            return 250.0

        result = execute_benchmark_suite(
            BenchmarkSuite.STANDARD,
            input_dir,
            output_dir,
            pipeline_runner=fake_runner,
        )

        assert result.suite == "standard"
        assert len(result.documents) == 1
        assert result.documents[0].document_id == "cases/invoice"
        assert result.documents[0].total_chars == len("invoice total due")
        assert result.summary["total_documents"] == 1
        assert (output_dir / "runs").is_dir()

    def test_execute_accuracy_suite_collects_ground_truth(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "out"
        ground_truth_dir = tmp_path / "truth"
        input_dir.mkdir()
        ground_truth_dir.mkdir()
        doc = input_dir / "record.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")
        (ground_truth_dir / "record.txt").write_text("matched text", encoding="utf-8")

        def fake_runner(document_path, source_dir, run_output_dir, pipeline_script):
            txt_path = get_text_output_path(document_path, source_dir, run_output_dir)
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text("matched text", encoding="utf-8")
            return 180.0

        result = execute_benchmark_suite(
            BenchmarkSuite.ACCURACY,
            input_dir,
            output_dir,
            ground_truth_dir=ground_truth_dir,
            pipeline_runner=fake_runner,
        )

        assert len(result.accuracy) == 1
        assert result.accuracy[0].character_accuracy == 1.0
        assert result.summary["avg_character_accuracy"] == 1.0
        assert result.summary["avg_word_accuracy"] == 1.0

    def test_build_output_rel_stem_preserves_non_pdf_extension_token(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        image = source_dir / "nested" / "scan.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"img")

        rel_stem = build_output_rel_stem(image, source_dir)

        assert rel_stem.as_posix() == "nested/scan__png"

    def test_build_run_root_sanitizes_invalid_stem_characters(self, tmp_path):
        run_root = _build_run_root(tmp_path, 7, Path("bad:name?.pdf"))

        assert run_root.name == "007_bad_name_"

    def test_run_pipeline_for_document_uses_utf8_capture(self, tmp_path):
        doc = tmp_path / "sample.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")

        with patch("benchmark_comparison.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = ""
            run_mock.return_value.stderr = ""

            run_pipeline_for_document(doc, tmp_path, tmp_path / "out", Path("pipe.py"))

        assert run_mock.call_args.kwargs["encoding"] == "utf-8"
        assert run_mock.call_args.kwargs["errors"] == "replace"
