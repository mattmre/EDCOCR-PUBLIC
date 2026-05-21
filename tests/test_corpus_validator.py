"""Tests for scripts/corpus_validator.py.

Validates the corpus validation framework including the CorpusFormat enum,
ValidationMetric / DocumentResult / CorpusReport dataclasses,
TextComparator utility (CER, WER, line_accuracy, normalize_text),
CorpusValidator class (construction, load_ground_truth, load_ocr_output,
validate_document, validate_corpus, discover_documents), and CLI parsing.
"""

import importlib
import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Import the module under test via importlib (it lives in scripts/)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_mod = importlib.import_module("corpus_validator")
CorpusFormat = _mod.CorpusFormat
ValidationMetric = _mod.ValidationMetric
DocumentResult = _mod.DocumentResult
CorpusReport = _mod.CorpusReport
TextComparator = _mod.TextComparator
CorpusValidator = _mod.CorpusValidator
build_parser = _mod.build_parser
main = _mod.main


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def comparator():
    """Return a TextComparator instance."""
    return TextComparator()


@pytest.fixture
def corpus_tree(tmp_path):
    """Create matching corpus and ground-truth directories with known files."""
    corpus = tmp_path / "corpus"
    gt = tmp_path / "ground_truth"
    corpus.mkdir()
    gt.mkdir()

    # doc_a: perfect match
    (corpus / "doc_a.txt").write_text("Hello world\nSecond line\n", encoding="utf-8")
    (gt / "doc_a.txt").write_text("Hello world\nSecond line\n", encoding="utf-8")

    # doc_b: minor error
    (corpus / "doc_b.txt").write_text("The quikc brown fox\n", encoding="utf-8")
    (gt / "doc_b.txt").write_text("The quick brown fox\n", encoding="utf-8")

    # doc_c: major error
    (corpus / "doc_c.txt").write_text("completely wrong output\n", encoding="utf-8")
    (gt / "doc_c.txt").write_text("The quick brown fox jumps over the lazy dog\n", encoding="utf-8")

    return corpus, gt


@pytest.fixture
def validator(corpus_tree):
    """Return a CorpusValidator wired to the corpus_tree fixture."""
    corpus, gt = corpus_tree
    return CorpusValidator(
        corpus_dir=str(corpus),
        ground_truth_dir=str(gt),
    )


# ===========================================================================
# CorpusFormat enum
# ===========================================================================


class TestCorpusFormat:
    """Tests for the CorpusFormat enum."""

    def test_plain_text_value(self):
        assert CorpusFormat.PLAIN_TEXT.value == "PLAIN_TEXT"

    def test_hocr_value(self):
        assert CorpusFormat.HOCR.value == "HOCR"

    def test_alto_xml_value(self):
        assert CorpusFormat.ALTO_XML.value == "ALTO_XML"

    def test_page_xml_value(self):
        assert CorpusFormat.PAGE_XML.value == "PAGE_XML"

    def test_custom_value(self):
        assert CorpusFormat.CUSTOM.value == "CUSTOM"

    def test_enum_member_count(self):
        assert len(CorpusFormat) == 5

    def test_from_string(self):
        assert CorpusFormat("PLAIN_TEXT") is CorpusFormat.PLAIN_TEXT

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            CorpusFormat("INVALID_FORMAT")


# ===========================================================================
# ValidationMetric dataclass
# ===========================================================================


class TestValidationMetric:
    """Tests for the ValidationMetric dataclass."""

    def test_construction(self):
        m = ValidationMetric(name="cer", value=0.02, threshold=0.05, passed=True)
        assert m.name == "cer"
        assert m.value == 0.02
        assert m.threshold == 0.05
        assert m.passed is True

    def test_failed_metric(self):
        m = ValidationMetric(name="wer", value=0.15, threshold=0.10, passed=False)
        assert m.passed is False

    def test_equality(self):
        a = ValidationMetric(name="cer", value=0.02, threshold=0.05, passed=True)
        b = ValidationMetric(name="cer", value=0.02, threshold=0.05, passed=True)
        assert a == b


# ===========================================================================
# DocumentResult dataclass
# ===========================================================================


class TestDocumentResult:
    """Tests for the DocumentResult dataclass."""

    def test_construction_defaults(self):
        r = DocumentResult(doc_id="doc1", source_path="/path/doc1.txt")
        assert r.doc_id == "doc1"
        assert r.source_path == "/path/doc1.txt"
        assert r.metrics == []
        assert r.overall_pass is False
        assert r.errors == []

    def test_with_metrics(self):
        m = ValidationMetric(name="cer", value=0.01, threshold=0.05, passed=True)
        r = DocumentResult(
            doc_id="d1",
            source_path="/d1.txt",
            metrics=[m],
            overall_pass=True,
        )
        assert len(r.metrics) == 1
        assert r.overall_pass is True

    def test_errors_list(self):
        r = DocumentResult(doc_id="d1", source_path="/d1.txt", errors=["file missing"])
        assert r.errors == ["file missing"]


# ===========================================================================
# CorpusReport dataclass
# ===========================================================================


class TestCorpusReport:
    """Tests for the CorpusReport dataclass."""

    def test_construction_defaults(self):
        r = CorpusReport(corpus_name="test", total_documents=0, passed=0, failed=0)
        assert r.corpus_name == "test"
        assert r.total_documents == 0
        assert r.metrics_summary == {}
        assert r.duration_seconds == 0.0
        assert r.results == []

    def test_to_dict(self):
        m = ValidationMetric(name="cer", value=0.02, threshold=0.05, passed=True)
        dr = DocumentResult(doc_id="d1", source_path="/d1.txt", metrics=[m], overall_pass=True)
        r = CorpusReport(
            corpus_name="my_corpus",
            total_documents=1,
            passed=1,
            failed=0,
            metrics_summary={"cer": 0.02},
            duration_seconds=1.234,
            results=[dr],
        )
        d = r.to_dict()
        assert d["corpus_name"] == "my_corpus"
        assert d["total_documents"] == 1
        assert d["passed"] == 1
        assert d["failed"] == 0
        assert d["duration_seconds"] == 1.234
        assert len(d["results"]) == 1
        assert d["results"][0]["doc_id"] == "d1"
        assert d["results"][0]["metrics"][0]["name"] == "cer"

    def test_to_dict_is_json_serializable(self):
        r = CorpusReport(corpus_name="c", total_documents=0, passed=0, failed=0)
        text = json.dumps(r.to_dict())
        assert isinstance(text, str)

    def test_summary_text_contains_header(self):
        r = CorpusReport(corpus_name="demo", total_documents=2, passed=1, failed=1)
        text = r.summary_text()
        assert "Corpus Validation Report" in text
        assert "demo" in text

    def test_summary_text_contains_counts(self):
        r = CorpusReport(corpus_name="c", total_documents=5, passed=3, failed=2)
        text = r.summary_text()
        assert "5" in text
        assert "3" in text
        assert "2" in text

    def test_metrics_summary_aggregation(self):
        """Metrics summary should hold average values."""
        r = CorpusReport(
            corpus_name="c",
            total_documents=2,
            passed=2,
            failed=0,
            metrics_summary={"cer": 0.025, "wer": 0.05},
        )
        assert r.metrics_summary["cer"] == pytest.approx(0.025)
        assert r.metrics_summary["wer"] == pytest.approx(0.05)


# ===========================================================================
# TextComparator — CER
# ===========================================================================


class TestCharacterErrorRate:
    """Tests for TextComparator.character_error_rate."""

    def test_identical_strings(self, comparator):
        assert comparator.character_error_rate("hello", "hello") == 0.0

    def test_completely_different(self, comparator):
        cer = comparator.character_error_rate("abc", "xyz")
        assert cer == pytest.approx(1.0)

    def test_empty_both(self, comparator):
        assert comparator.character_error_rate("", "") == 0.0

    def test_empty_ground_truth_nonempty_predicted(self, comparator):
        assert comparator.character_error_rate("abc", "") == 1.0

    def test_empty_predicted_nonempty_ground_truth(self, comparator):
        cer = comparator.character_error_rate("", "hello")
        assert cer == pytest.approx(1.0)  # 5 deletes / 5 chars

    def test_single_char_error(self, comparator):
        # "helo" vs "hello" => 1 insertion / 5 = 0.2
        cer = comparator.character_error_rate("helo", "hello")
        assert cer == pytest.approx(0.2)

    def test_cer_symmetry_not_guaranteed(self, comparator):
        """CER(a,b) != CER(b,a) in general because denominator differs."""
        cer_ab = comparator.character_error_rate("abc", "abcd")
        cer_ba = comparator.character_error_rate("abcd", "abc")
        # Just verify both are positive
        assert cer_ab > 0
        assert cer_ba > 0

    def test_cer_returns_float(self, comparator):
        result = comparator.character_error_rate("test", "text")
        assert isinstance(result, float)


# ===========================================================================
# TextComparator — WER
# ===========================================================================


class TestWordErrorRate:
    """Tests for TextComparator.word_error_rate."""

    def test_identical_sentences(self, comparator):
        assert comparator.word_error_rate("hello world", "hello world") == 0.0

    def test_completely_different(self, comparator):
        wer = comparator.word_error_rate("foo bar", "baz qux")
        assert wer == pytest.approx(1.0)

    def test_empty_both(self, comparator):
        assert comparator.word_error_rate("", "") == 0.0

    def test_empty_ground_truth_nonempty_predicted(self, comparator):
        assert comparator.word_error_rate("some words", "") == 1.0

    def test_one_word_wrong(self, comparator):
        wer = comparator.word_error_rate("the quick brown fox", "the quikc brown fox")
        # 1 substitution / 4 words = 0.25
        assert wer == pytest.approx(0.25)

    def test_extra_word(self, comparator):
        wer = comparator.word_error_rate("a b c d", "a b c")
        # 1 insertion / 3 words
        assert wer == pytest.approx(1.0 / 3.0)

    def test_missing_word(self, comparator):
        wer = comparator.word_error_rate("a b", "a b c")
        # 1 deletion / 3 words
        assert wer == pytest.approx(1.0 / 3.0)


# ===========================================================================
# TextComparator — line_accuracy
# ===========================================================================


class TestLineAccuracy:
    """Tests for TextComparator.line_accuracy."""

    def test_identical_lines(self, comparator):
        text = "line one\nline two\nline three"
        assert comparator.line_accuracy(text, text) == 1.0

    def test_all_different_lines(self, comparator):
        pred = "a\nb\nc"
        gt = "x\ny\nz"
        assert comparator.line_accuracy(pred, gt) == 0.0

    def test_empty_both(self, comparator):
        assert comparator.line_accuracy("", "") == 1.0

    def test_partial_match(self, comparator):
        pred = "line one\nwrong\nline three"
        gt = "line one\nline two\nline three"
        assert comparator.line_accuracy(pred, gt) == pytest.approx(2.0 / 3.0)

    def test_extra_predicted_lines(self, comparator):
        pred = "a\nb\nc\nd"
        gt = "a\nb"
        # 4 total, 2 match
        assert comparator.line_accuracy(pred, gt) == pytest.approx(0.5)

    def test_fewer_predicted_lines(self, comparator):
        pred = "a"
        gt = "a\nb\nc"
        # 3 total, 1 match
        assert comparator.line_accuracy(pred, gt) == pytest.approx(1.0 / 3.0)


# ===========================================================================
# TextComparator — normalize_text
# ===========================================================================


class TestNormalizeText:
    """Tests for TextComparator.normalize_text."""

    def test_strips_whitespace(self, comparator):
        assert comparator.normalize_text("  hello  ") == "hello"

    def test_collapses_interior_whitespace(self, comparator):
        assert comparator.normalize_text("a   b     c") == "a b c"

    def test_collapses_newlines(self, comparator):
        assert comparator.normalize_text("a\n\nb\tc") == "a b c"

    def test_empty_string(self, comparator):
        assert comparator.normalize_text("") == ""

    def test_unicode_normalization(self, comparator):
        # NFC normalization: combining é (e + ´) → single é
        composed = "\u00e9"     # é (single code point)
        decomposed = "e\u0301"  # e + combining acute
        assert comparator.normalize_text(composed) == comparator.normalize_text(decomposed)

    def test_only_whitespace(self, comparator):
        assert comparator.normalize_text("   \t\n  ") == ""


# ===========================================================================
# CorpusValidator — construction
# ===========================================================================


class TestCorpusValidatorConstruction:
    """Tests for CorpusValidator initialization."""

    def test_default_format(self, tmp_path):
        v = CorpusValidator(
            corpus_dir=str(tmp_path / "c"),
            ground_truth_dir=str(tmp_path / "gt"),
        )
        assert v.format is CorpusFormat.PLAIN_TEXT

    def test_custom_format(self, tmp_path):
        v = CorpusValidator(
            corpus_dir=str(tmp_path / "c"),
            ground_truth_dir=str(tmp_path / "gt"),
            format=CorpusFormat.HOCR,
        )
        assert v.format is CorpusFormat.HOCR

    def test_stores_dirs(self, tmp_path):
        c = str(tmp_path / "c")
        g = str(tmp_path / "gt")
        v = CorpusValidator(corpus_dir=c, ground_truth_dir=g)
        assert v.corpus_dir == c
        assert v.ground_truth_dir == g


# ===========================================================================
# CorpusValidator — load_ground_truth / load_ocr_output
# ===========================================================================


class TestLoadFiles:
    """Tests for load_ground_truth and load_ocr_output."""

    def test_load_ground_truth(self, validator, corpus_tree):
        text = validator.load_ground_truth("doc_a")
        assert "Hello world" in text

    def test_load_ocr_output(self, validator, corpus_tree):
        text = validator.load_ocr_output("doc_a")
        assert "Hello world" in text

    def test_load_ground_truth_missing(self, validator):
        with pytest.raises(FileNotFoundError):
            validator.load_ground_truth("nonexistent")

    def test_load_ocr_output_missing(self, validator):
        with pytest.raises(FileNotFoundError):
            validator.load_ocr_output("nonexistent")

    def test_load_ground_truth_hocr_extension(self, tmp_path):
        gt = tmp_path / "gt"
        gt.mkdir()
        (gt / "doc.hocr").write_text("hocr content", encoding="utf-8")
        v = CorpusValidator(
            corpus_dir=str(tmp_path),
            ground_truth_dir=str(gt),
            format=CorpusFormat.HOCR,
        )
        text = v.load_ground_truth("doc")
        assert text == "hocr content"


# ===========================================================================
# CorpusValidator — discover_documents
# ===========================================================================


class TestDiscoverDocuments:
    """Tests for discover_documents."""

    def test_discovers_matching_docs(self, validator):
        docs = validator.discover_documents()
        assert "doc_a" in docs
        assert "doc_b" in docs
        assert "doc_c" in docs

    def test_sorted_order(self, validator):
        docs = validator.discover_documents()
        assert docs == sorted(docs)

    def test_no_match_when_gt_missing(self, tmp_path):
        corpus = tmp_path / "corpus"
        gt = tmp_path / "gt"
        corpus.mkdir()
        gt.mkdir()
        (corpus / "only_in_corpus.txt").write_text("data", encoding="utf-8")
        v = CorpusValidator(corpus_dir=str(corpus), ground_truth_dir=str(gt))
        assert v.discover_documents() == []

    def test_no_match_when_corpus_missing(self, tmp_path):
        corpus = tmp_path / "corpus"
        gt = tmp_path / "gt"
        corpus.mkdir()
        gt.mkdir()
        (gt / "only_in_gt.txt").write_text("data", encoding="utf-8")
        v = CorpusValidator(corpus_dir=str(corpus), ground_truth_dir=str(gt))
        assert v.discover_documents() == []

    def test_only_intersecting_ids(self, tmp_path):
        corpus = tmp_path / "corpus"
        gt = tmp_path / "gt"
        corpus.mkdir()
        gt.mkdir()
        (corpus / "shared.txt").write_text("a", encoding="utf-8")
        (corpus / "only_corpus.txt").write_text("b", encoding="utf-8")
        (gt / "shared.txt").write_text("a", encoding="utf-8")
        (gt / "only_gt.txt").write_text("c", encoding="utf-8")
        v = CorpusValidator(corpus_dir=str(corpus), ground_truth_dir=str(gt))
        docs = v.discover_documents()
        assert docs == ["shared"]


# ===========================================================================
# CorpusValidator — validate_document
# ===========================================================================


class TestValidateDocument:
    """Tests for validate_document."""

    def test_perfect_match_passes(self, validator):
        result = validator.validate_document("doc_a")
        assert result.overall_pass is True
        assert len(result.metrics) == 3
        assert all(m.passed for m in result.metrics)

    def test_minor_error_with_default_thresholds(self, validator):
        result = validator.validate_document("doc_b")
        # "quikc" vs "quick" — CER is small, should be within defaults
        cer_metric = next(m for m in result.metrics if m.name == "cer")
        assert cer_metric.value > 0

    def test_major_error_fails(self, validator):
        result = validator.validate_document("doc_c")
        assert result.overall_pass is False
        assert len(result.errors) > 0

    def test_custom_strict_thresholds_fail(self, validator):
        result = validator.validate_document(
            "doc_b",
            thresholds={"cer": 0.001, "wer": 0.001, "line_accuracy": 1.0},
        )
        assert result.overall_pass is False

    def test_custom_lenient_thresholds_pass(self, validator):
        result = validator.validate_document(
            "doc_c",
            thresholds={"cer": 5.0, "wer": 5.0, "line_accuracy": 0.0},
        )
        assert result.overall_pass is True

    def test_missing_doc_returns_errors(self, validator):
        result = validator.validate_document("nonexistent")
        assert result.overall_pass is False
        assert len(result.errors) > 0

    def test_result_has_source_path(self, validator):
        result = validator.validate_document("doc_a")
        assert "doc_a" in result.source_path

    def test_metrics_have_thresholds(self, validator):
        result = validator.validate_document("doc_a")
        for m in result.metrics:
            assert m.threshold > 0 or m.name == "line_accuracy"

    def test_cer_metric_present(self, validator):
        result = validator.validate_document("doc_a")
        names = [m.name for m in result.metrics]
        assert "cer" in names

    def test_wer_metric_present(self, validator):
        result = validator.validate_document("doc_a")
        names = [m.name for m in result.metrics]
        assert "wer" in names

    def test_line_accuracy_metric_present(self, validator):
        result = validator.validate_document("doc_a")
        names = [m.name for m in result.metrics]
        assert "line_accuracy" in names


# ===========================================================================
# CorpusValidator — validate_corpus
# ===========================================================================


class TestValidateCorpus:
    """Tests for validate_corpus."""

    def test_validates_all_docs(self, validator):
        report = validator.validate_corpus()
        assert report.total_documents == 3

    def test_report_counts(self, validator):
        report = validator.validate_corpus()
        assert report.passed + report.failed == report.total_documents

    def test_metrics_summary_populated(self, validator):
        report = validator.validate_corpus()
        assert "cer" in report.metrics_summary
        assert "wer" in report.metrics_summary
        assert "line_accuracy" in report.metrics_summary

    def test_metrics_summary_averages(self, tmp_path):
        """Average metrics should be between min and max of individual docs."""
        corpus = tmp_path / "c"
        gt = tmp_path / "gt"
        corpus.mkdir()
        gt.mkdir()
        (corpus / "a.txt").write_text("hello world", encoding="utf-8")
        (gt / "a.txt").write_text("hello world", encoding="utf-8")
        (corpus / "b.txt").write_text("hello world", encoding="utf-8")
        (gt / "b.txt").write_text("hello world", encoding="utf-8")
        v = CorpusValidator(corpus_dir=str(corpus), ground_truth_dir=str(gt))
        report = v.validate_corpus()
        # Both identical, average CER should be 0
        assert report.metrics_summary["cer"] == pytest.approx(0.0)

    def test_duration_is_positive(self, validator):
        report = validator.validate_corpus()
        assert report.duration_seconds >= 0

    def test_corpus_name_from_dir(self, validator, corpus_tree):
        report = validator.validate_corpus()
        assert report.corpus_name == "corpus"

    def test_results_list(self, validator):
        report = validator.validate_corpus()
        assert len(report.results) == 3

    def test_custom_thresholds_applied(self, validator):
        report = validator.validate_corpus(thresholds={"cer": 10.0, "wer": 10.0, "line_accuracy": 0.0})
        assert report.passed == 3

    def test_empty_corpus(self, tmp_path):
        corpus = tmp_path / "c"
        gt = tmp_path / "gt"
        corpus.mkdir()
        gt.mkdir()
        v = CorpusValidator(corpus_dir=str(corpus), ground_truth_dir=str(gt))
        report = v.validate_corpus()
        assert report.total_documents == 0
        assert report.passed == 0
        assert report.failed == 0


# ===========================================================================
# DEFAULT_THRESHOLDS
# ===========================================================================


class TestDefaultThresholds:
    """Tests for CorpusValidator.DEFAULT_THRESHOLDS."""

    def test_cer_threshold(self):
        assert CorpusValidator.DEFAULT_THRESHOLDS["cer"] == 0.05

    def test_wer_threshold(self):
        assert CorpusValidator.DEFAULT_THRESHOLDS["wer"] == 0.10

    def test_line_accuracy_threshold(self):
        assert CorpusValidator.DEFAULT_THRESHOLDS["line_accuracy"] == 0.90

    def test_has_three_keys(self):
        assert len(CorpusValidator.DEFAULT_THRESHOLDS) == 3


# ===========================================================================
# CLI argument parsing
# ===========================================================================


class TestCLI:
    """Tests for CLI argument parsing and main entry point."""

    def test_parser_requires_corpus_dir(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_requires_ground_truth_dir(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--corpus-dir", "/tmp/c"])

    def test_parser_default_format(self):
        parser = build_parser()
        args = parser.parse_args(["--corpus-dir", "/c", "--ground-truth-dir", "/g"])
        assert args.format == "PLAIN_TEXT"

    def test_parser_custom_format(self):
        parser = build_parser()
        args = parser.parse_args(["--corpus-dir", "/c", "--ground-truth-dir", "/g", "--format", "HOCR"])
        assert args.format == "HOCR"

    def test_parser_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--corpus-dir", "/c", "--ground-truth-dir", "/g", "--json"])
        assert args.json is True

    def test_parser_thresholds(self):
        parser = build_parser()
        args = parser.parse_args([
            "--corpus-dir", "/c",
            "--ground-truth-dir", "/g",
            "--thresholds", '{"cer": 0.03}',
        ])
        assert args.thresholds == '{"cer": 0.03}'

    def test_parser_doc_id(self):
        parser = build_parser()
        args = parser.parse_args([
            "--corpus-dir", "/c",
            "--ground-truth-dir", "/g",
            "--doc-id", "my_doc",
        ])
        assert args.doc_id == "my_doc"

    def test_main_corpus_validation(self, validator, corpus_tree):
        corpus, gt = corpus_tree
        rc = main(["--corpus-dir", str(corpus), "--ground-truth-dir", str(gt)])
        # Some docs fail with default thresholds
        assert rc in (0, 1)

    def test_main_single_doc(self, corpus_tree):
        corpus, gt = corpus_tree
        rc = main([
            "--corpus-dir", str(corpus),
            "--ground-truth-dir", str(gt),
            "--doc-id", "doc_a",
        ])
        assert rc == 0  # perfect match

    def test_main_json_output(self, corpus_tree, capsys):
        corpus, gt = corpus_tree
        main([
            "--corpus-dir", str(corpus),
            "--ground-truth-dir", str(gt),
            "--json",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "corpus_name" in data

    def test_main_single_doc_json(self, corpus_tree, capsys):
        corpus, gt = corpus_tree
        main([
            "--corpus-dir", str(corpus),
            "--ground-truth-dir", str(gt),
            "--doc-id", "doc_a",
            "--json",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_documents"] == 1

    def test_main_invalid_thresholds(self, corpus_tree):
        corpus, gt = corpus_tree
        rc = main([
            "--corpus-dir", str(corpus),
            "--ground-truth-dir", str(gt),
            "--thresholds", "not-json",
        ])
        assert rc == 1
