"""
Unit tests for structured extraction module (extraction.py).

Tests cover:
- UIE extraction with mocked PaddleNLP Taskflow
- Regex extraction for dates, amounts, phones, emails, references
- Deduplication of overlapping UIE + regex results
- Finalization and summary stats
- JSON output format and file paths
- Graceful degradation when PaddleNLP is not available
- Edge cases: empty text, special characters

Run with: python -m pytest tests/test_extraction.py -v
"""

import json
import os
from unittest import mock

import extraction as extraction_module
from extraction import (
    DEFAULT_SCHEMA,
    DocumentExtraction,
    ExtractedField,
    PageExtraction,
    _deduplicate_fields,
    extract_fields_regex,
    extract_fields_uie,
    extract_page_fields,
    finalize_extraction,
    write_extraction_json,
)
from version import __version__

# ---------------------------------------------------------------------------
# Helpers: Mock UIE engine
# ---------------------------------------------------------------------------


class MockUIEEngine:
    """Callable mock that returns UIE-format extraction results."""

    def __init__(self, results):
        """results: list of dicts mapping schema labels to extraction lists."""
        self._results = results

    def __call__(self, text):
        return self._results


def _make_uie_result(**label_extractions):
    """Build a single UIE result dict.

    Usage: _make_uie_result(Date=[{"text": "2024-01-15", "start": 10, "end": 20, "probability": 0.95}])
    """
    return [label_extractions]


# ---------------------------------------------------------------------------
# Tests: Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_extracted_field_defaults(self):
        f = ExtractedField(field_type="date", text="2024-01-01")
        assert f.field_type == "date"
        assert f.text == "2024-01-01"
        assert f.confidence == 0.0
        assert f.page_num == 0
        assert f.start == 0
        assert f.end == 0
        assert f.extraction_method == ""
        assert f.normalized_value == ""

    def test_page_extraction_defaults(self):
        p = PageExtraction(page_num=1)
        assert p.page_num == 1
        assert p.fields == []

    def test_document_extraction_defaults(self):
        d = DocumentExtraction(document_id="doc1", source_file="test.pdf")
        assert d.document_id == "doc1"
        assert d.source_file == "test.pdf"
        assert d.pages == []
        assert d.total_fields == 0
        assert d.field_type_counts == {}
        assert d.extraction_engine == ""


# ---------------------------------------------------------------------------
# Tests: UIE extraction
# ---------------------------------------------------------------------------


class TestUIEExtraction:
    """Test extract_fields_uie with mocked PaddleNLP UIE engine."""

    def _patch_uie(self, engine, available=True):
        """Context manager to patch both _uie_engine and _PADDLENLP_AVAILABLE."""
        return mock.patch.multiple(
            extraction_module,
            _uie_engine=engine,
            _PADDLENLP_AVAILABLE=available,
            _uie_init_failed=False,
        )

    def test_uie_extracts_date(self):
        engine = MockUIEEngine(_make_uie_result(
            Date=[{"text": "2024-01-15", "start": 10, "end": 20, "probability": 0.95}],
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("Invoice dated 2024-01-15 was received.", 1)
        assert len(result) == 1
        assert result[0].field_type == "date"
        assert result[0].text == "2024-01-15"

    def test_uie_extracts_amount(self):
        engine = MockUIEEngine(_make_uie_result(
            Amount=[{"text": "$1,234.56", "start": 5, "end": 14, "probability": 0.88}],
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("Total $1,234.56 due.", 1)
        assert len(result) == 1
        assert result[0].field_type == "amount"
        assert result[0].text == "$1,234.56"

    def test_uie_extracts_person(self):
        engine = MockUIEEngine(_make_uie_result(
            **{"Person Name": [{"text": "John Smith", "start": 0, "end": 10, "probability": 0.92}]},
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("John Smith is the contact.", 1)
        assert len(result) == 1
        assert result[0].field_type == "person_name"

    def test_uie_extracts_organization(self):
        engine = MockUIEEngine(_make_uie_result(
            Organization=[{"text": "Acme Corp", "start": 0, "end": 9, "probability": 0.91}],
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("Acme Corp filed the claim.", 1)
        assert len(result) == 1
        assert result[0].field_type == "organization"
        assert result[0].text == "Acme Corp"

    def test_uie_extracts_address(self):
        engine = MockUIEEngine(_make_uie_result(
            Address=[{"text": "123 Main St, Anytown, CA 90210", "start": 0, "end": 30, "probability": 0.85}],
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("123 Main St, Anytown, CA 90210", 2)
        assert len(result) == 1
        assert result[0].field_type == "address"

    def test_uie_extracts_phone(self):
        engine = MockUIEEngine(_make_uie_result(
            **{"Phone Number": [{"text": "(555) 123-4567", "start": 5, "end": 19, "probability": 0.90}]},
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("Call (555) 123-4567 now.", 1)
        assert len(result) == 1
        assert result[0].field_type == "phone_number"

    def test_uie_extracts_email(self):
        engine = MockUIEEngine(_make_uie_result(
            **{"Email Address": [{"text": "user@example.com", "start": 6, "end": 22, "probability": 0.97}]},
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("Email user@example.com for info.", 1)
        assert len(result) == 1
        assert result[0].field_type == "email_address"

    def test_uie_extracts_reference(self):
        engine = MockUIEEngine(_make_uie_result(
            **{"Reference Number": [{"text": "INV-2024-001", "start": 0, "end": 12, "probability": 0.89}]},
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("INV-2024-001 is the invoice number.", 1)
        assert len(result) == 1
        assert result[0].field_type == "reference_number"

    def test_uie_multiple_entities(self):
        engine = MockUIEEngine(_make_uie_result(
            Date=[{"text": "2024-01-15", "start": 0, "end": 10, "probability": 0.95}],
            Amount=[{"text": "$500.00", "start": 20, "end": 27, "probability": 0.88}],
            **{"Person Name": [{"text": "Alice", "start": 30, "end": 35, "probability": 0.90}]},
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("2024-01-15 - invoice $500.00 for Alice", 1)
        assert len(result) == 3
        types = {f.field_type for f in result}
        assert "date" in types
        assert "amount" in types
        assert "person_name" in types

    def test_uie_empty_text(self):
        engine = MockUIEEngine([{}])
        with self._patch_uie(engine):
            result = extract_fields_uie("", 1)
        assert result == []

    def test_uie_not_available(self):
        with mock.patch.multiple(
            extraction_module,
            _uie_engine=None,
            _PADDLENLP_AVAILABLE=False,
            _uie_init_failed=False,
        ):
            result = extract_fields_uie("Some text with dates 2024-01-15", 1)
        assert result == []

    def test_uie_engine_failure(self):
        def failing_engine(text):
            raise RuntimeError("UIE model crashed")

        with self._patch_uie(failing_engine):
            result = extract_fields_uie("Some text here.", 1)
        assert result == []

    def test_uie_confidence_preserved(self):
        engine = MockUIEEngine(_make_uie_result(
            Date=[{"text": "2024-06-01", "start": 0, "end": 10, "probability": 0.9321}],
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("2024-06-01", 1)
        assert len(result) == 1
        assert result[0].confidence == 0.9321

    def test_uie_extraction_method(self):
        engine = MockUIEEngine(_make_uie_result(
            Date=[{"text": "2024-01-01", "start": 0, "end": 10, "probability": 0.9}],
            Amount=[{"text": "$100", "start": 15, "end": 19, "probability": 0.8}],
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("2024-01-01 total $100", 1)
        for f in result:
            assert f.extraction_method == "uie"

    def test_uie_page_num_preserved(self):
        engine = MockUIEEngine(_make_uie_result(
            Date=[{"text": "2024-03-20", "start": 0, "end": 10, "probability": 0.95}],
        ))
        with self._patch_uie(engine):
            result = extract_fields_uie("2024-03-20", 42)
        assert len(result) == 1
        assert result[0].page_num == 42


# ---------------------------------------------------------------------------
# Tests: Regex extraction
# ---------------------------------------------------------------------------


class TestRegexExtraction:
    """Test extract_fields_regex with various text patterns."""

    def test_iso_date(self):
        result = extract_fields_regex("Document dated 2024-01-15 received.", 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1
        assert any(f.text == "2024-01-15" for f in dates)

    def test_us_date(self):
        result = extract_fields_regex("Filed on 01/15/2024 by the court.", 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1
        assert any(f.text == "01/15/2024" for f in dates)

    def test_written_date(self):
        result = extract_fields_regex("Signed on January 15, 2024 by the CEO.", 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1
        assert any("January" in f.text for f in dates)

    def test_written_date_short(self):
        result = extract_fields_regex("Meeting on Jan 15, 2024 at the office.", 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1
        assert any("Jan" in f.text for f in dates)

    def test_amount_usd_symbol(self):
        result = extract_fields_regex("Total payment: $1,234.56 confirmed.", 1)
        amounts = [f for f in result if f.field_type == "amount"]
        assert len(amounts) >= 1
        assert any("1,234.56" in f.text for f in amounts)

    def test_amount_eur(self):
        result = extract_fields_regex("Price is EUR 100.00 per unit.", 1)
        amounts = [f for f in result if f.field_type == "amount"]
        assert len(amounts) >= 1
        assert any("EUR" in f.text or "100.00" in f.text for f in amounts)

    def test_amount_no_currency(self):
        """Bare number without currency symbol/code should NOT match as amount."""
        result = extract_fields_regex("The number is 1234 or maybe 5678.", 1)
        amounts = [f for f in result if f.field_type == "amount"]
        assert len(amounts) == 0

    def test_phone_us(self):
        result = extract_fields_regex("Call us at (555) 123-4567 today.", 1)
        phones = [f for f in result if f.field_type == "phone_number"]
        assert len(phones) >= 1
        assert any("555" in f.text and "4567" in f.text for f in phones)

    def test_phone_with_country_code(self):
        # The regex pattern optionally matches the country code prefix;
        # depending on word boundary rules, the +1 may or may not be included.
        result = extract_fields_regex("International: +1-555-123-4567 available.", 1)
        phones = [f for f in result if f.field_type == "phone_number"]
        assert len(phones) >= 1
        # Core phone digits should be present regardless of country code capture
        assert any("555" in f.text and "4567" in f.text for f in phones)

    def test_phone_dashes(self):
        result = extract_fields_regex("Dial 555-123-4567 for support.", 1)
        phones = [f for f in result if f.field_type == "phone_number"]
        assert len(phones) >= 1

    def test_email_simple(self):
        result = extract_fields_regex("Contact user@example.com for help.", 1)
        emails = [f for f in result if f.field_type == "email_address"]
        assert len(emails) >= 1
        assert any(f.text == "user@example.com" for f in emails)

    def test_email_complex(self):
        result = extract_fields_regex(
            "Send to first.last+tag@sub.domain.org for details.", 1
        )
        emails = [f for f in result if f.field_type == "email_address"]
        assert len(emails) >= 1
        assert any("first.last+tag@sub.domain.org" in f.text for f in emails)

    def test_reference_inv(self):
        result = extract_fields_regex("See invoice INV-2024-001 attached.", 1)
        refs = [f for f in result if f.field_type == "reference_number"]
        assert len(refs) >= 1
        assert any("INV" in f.text for f in refs)

    def test_reference_po(self):
        result = extract_fields_regex("Purchase order PO#12345 confirmed.", 1)
        refs = [f for f in result if f.field_type == "reference_number"]
        assert len(refs) >= 1
        assert any("PO" in f.text for f in refs)

    def test_reference_ref(self):
        result = extract_fields_regex("Reference REF: 12345 for your records.", 1)
        refs = [f for f in result if f.field_type == "reference_number"]
        assert len(refs) >= 1
        assert any("REF" in f.text for f in refs)

    def test_multiple_dates(self):
        text = "Filed on 2024-01-10, amended 2024-02-20, finalized 2024-03-30."
        result = extract_fields_regex(text, 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) == 3

    def test_no_matches(self):
        result = extract_fields_regex("Hello world, this is a simple test.", 1)
        assert result == []

    def test_empty_text(self):
        result = extract_fields_regex("", 1)
        assert result == []

    def test_extraction_method_regex(self):
        result = extract_fields_regex("Contact user@example.com on 2024-01-15.", 1)
        for f in result:
            assert f.extraction_method == "regex"

    def test_confidence_is_one(self):
        result = extract_fields_regex("$500.00 on 2024-06-01.", 1)
        for f in result:
            assert f.confidence == 1.0


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Test _deduplicate_fields logic."""

    def test_no_overlap_kept(self):
        """UIE and regex find different fields -- all kept."""
        uie_field = ExtractedField(
            field_type="date", text="2024-01-15",
            start=0, end=10, extraction_method="uie",
        )
        regex_field = ExtractedField(
            field_type="email_address", text="test@example.com",
            start=50, end=66, extraction_method="regex",
        )
        result = _deduplicate_fields([uie_field, regex_field])
        assert len(result) == 2

    def test_overlapping_same_type_deduped(self):
        """UIE and regex find same text, same type -- UIE preferred."""
        uie_field = ExtractedField(
            field_type="date", text="2024-01-15",
            start=10, end=20, extraction_method="uie",
        )
        regex_field = ExtractedField(
            field_type="date", text="2024-01-15",
            start=10, end=20, extraction_method="regex",
        )
        result = _deduplicate_fields([uie_field, regex_field])
        assert len(result) == 1
        assert result[0].extraction_method == "uie"

    def test_overlapping_different_type_kept(self):
        """Same text range but different types -- both kept."""
        uie_field = ExtractedField(
            field_type="date", text="2024-01-15",
            start=10, end=20, extraction_method="uie",
        )
        regex_field = ExtractedField(
            field_type="reference_number", text="2024-01-15",
            start=10, end=20, extraction_method="regex",
        )
        result = _deduplicate_fields([uie_field, regex_field])
        assert len(result) == 2

    def test_partial_overlap(self):
        """Partially overlapping ranges with same type -- regex removed."""
        uie_field = ExtractedField(
            field_type="amount", text="$1,234.56",
            start=10, end=19, extraction_method="uie",
        )
        # Regex finds a slightly different range but overlapping
        regex_field = ExtractedField(
            field_type="amount", text="$1,234.56",
            start=10, end=20, extraction_method="regex",
        )
        result = _deduplicate_fields([uie_field, regex_field])
        assert len(result) == 1
        assert result[0].extraction_method == "uie"

    def test_empty_inputs(self):
        result = _deduplicate_fields([])
        assert result == []


# ---------------------------------------------------------------------------
# Tests: extract_page_fields
# ---------------------------------------------------------------------------


class TestPageFields:
    """Test extract_page_fields combining UIE and regex."""

    def test_with_uie_enabled(self):
        """use_uie=True calls both UIE and regex."""
        engine = MockUIEEngine(_make_uie_result(
            Date=[{"text": "2024-01-15", "start": 0, "end": 10, "probability": 0.95}],
        ))
        with mock.patch.multiple(
            extraction_module,
            _uie_engine=engine,
            _PADDLENLP_AVAILABLE=True,
            _uie_init_failed=False,
        ):
            result = extract_page_fields(
                "2024-01-15 invoice for $500.00", 1, use_uie=True,
            )
        assert isinstance(result, PageExtraction)
        assert result.page_num == 1
        # Should have fields from both UIE and regex (deduped)
        assert len(result.fields) >= 1

    def test_without_uie(self):
        """use_uie=False uses regex only."""
        result = extract_page_fields(
            "Invoice dated 2024-01-15 for $500.00", 3, use_uie=False,
        )
        assert isinstance(result, PageExtraction)
        assert result.page_num == 3
        # Should have regex fields only
        for f in result.fields:
            assert f.get("extraction_method") == "regex"

    def test_uie_unavailable_fallback(self):
        """When _PADDLENLP_AVAILABLE is False, falls back to regex only."""
        with mock.patch.multiple(
            extraction_module,
            _uie_engine=None,
            _PADDLENLP_AVAILABLE=False,
            _uie_init_failed=False,
        ):
            result = extract_page_fields(
                "2024-01-15 $500.00 user@test.com", 1, use_uie=True,
            )
        # All fields should be regex-based since UIE was unavailable
        for f in result.fields:
            assert f.get("extraction_method") == "regex"

    def test_page_num_on_all_fields(self):
        result = extract_page_fields(
            "$100.00 on 2024-01-15 at user@test.com", 7, use_uie=False,
        )
        for f in result.fields:
            assert f.get("page_num") == 7

    def test_combined_results(self):
        """Both UIE and regex results are combined and deduplicated."""
        # UIE finds a person name, regex finds an email -- both should appear
        engine = MockUIEEngine(_make_uie_result(
            **{"Person Name": [{"text": "Jane Doe", "start": 0, "end": 8, "probability": 0.93}]},
        ))
        with mock.patch.multiple(
            extraction_module,
            _uie_engine=engine,
            _PADDLENLP_AVAILABLE=True,
            _uie_init_failed=False,
        ):
            result = extract_page_fields(
                "Jane Doe emailed admin@test.com on 2024-01-15", 1, use_uie=True,
            )
        types_found = {f["field_type"] for f in result.fields}
        assert "person_name" in types_found
        assert "email_address" in types_found


# ---------------------------------------------------------------------------
# Tests: Finalization
# ---------------------------------------------------------------------------


class TestFinalization:
    """Test finalize_extraction summary computation."""

    def test_finalize_empty(self):
        doc = DocumentExtraction(document_id="d1", source_file="empty.pdf")
        result = finalize_extraction(doc)
        assert result.total_fields == 0
        assert result.field_type_counts == {}
        assert result.extraction_engine == ""

    def test_finalize_counts(self):
        doc = DocumentExtraction(document_id="d2", source_file="multi.pdf")
        doc.pages = [
            PageExtraction(page_num=1, fields=[
                {"field_type": "date", "text": "2024-01-01", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 0, "end": 10, "normalized_value": ""},
                {"field_type": "amount", "text": "$100", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 15, "end": 19, "normalized_value": ""},
            ]),
            PageExtraction(page_num=2, fields=[
                {"field_type": "date", "text": "2024-02-01", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 2, "start": 0, "end": 10, "normalized_value": ""},
            ]),
        ]
        result = finalize_extraction(doc)
        assert result.total_fields == 3

    def test_finalize_engine_uie(self):
        doc = DocumentExtraction(document_id="d3", source_file="uie.pdf")
        doc.pages = [
            PageExtraction(page_num=1, fields=[
                {"field_type": "date", "text": "2024-01-01", "extraction_method": "uie",
                 "confidence": 0.95, "page_num": 1, "start": 0, "end": 10, "normalized_value": ""},
            ]),
        ]
        result = finalize_extraction(doc)
        assert result.extraction_engine == "uie"

    def test_finalize_engine_regex(self):
        doc = DocumentExtraction(document_id="d4", source_file="regex.pdf")
        doc.pages = [
            PageExtraction(page_num=1, fields=[
                {"field_type": "date", "text": "2024-01-01", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 0, "end": 10, "normalized_value": ""},
            ]),
        ]
        result = finalize_extraction(doc)
        assert result.extraction_engine == "regex"

    def test_finalize_engine_hybrid(self):
        doc = DocumentExtraction(document_id="d5", source_file="hybrid.pdf")
        doc.pages = [
            PageExtraction(page_num=1, fields=[
                {"field_type": "date", "text": "2024-01-01", "extraction_method": "uie",
                 "confidence": 0.95, "page_num": 1, "start": 0, "end": 10, "normalized_value": ""},
                {"field_type": "email_address", "text": "a@b.com", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 20, "end": 27, "normalized_value": ""},
            ]),
        ]
        result = finalize_extraction(doc)
        assert result.extraction_engine == "hybrid"

    def test_finalize_field_type_counts(self):
        doc = DocumentExtraction(document_id="d6", source_file="counts.pdf")
        doc.pages = [
            PageExtraction(page_num=1, fields=[
                {"field_type": "date", "text": "2024-01-01", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 0, "end": 10, "normalized_value": ""},
                {"field_type": "date", "text": "2024-02-01", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 15, "end": 25, "normalized_value": ""},
                {"field_type": "amount", "text": "$100", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 30, "end": 34, "normalized_value": ""},
                {"field_type": "email_address", "text": "a@b.com", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 40, "end": 47, "normalized_value": ""},
            ]),
        ]
        result = finalize_extraction(doc)
        assert result.field_type_counts["date"] == 2
        assert result.field_type_counts["amount"] == 1
        assert result.field_type_counts["email_address"] == 1
        assert result.total_fields == 4
        # Verify counts are sorted alphabetically
        keys = list(result.field_type_counts.keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Tests: write_extraction_json
# ---------------------------------------------------------------------------


class TestWriteJson:
    """Test JSON output file creation and structure."""

    def _make_finalized_doc(self, source_file="test_doc.pdf"):
        """Create a simple finalized DocumentExtraction for testing."""
        doc = DocumentExtraction(document_id="test_id", source_file=source_file)
        doc.pages = [
            PageExtraction(page_num=1, fields=[
                {"field_type": "date", "text": "2024-01-15", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 0, "end": 10,
                 "normalized_value": "2024-01-15"},
                {"field_type": "amount", "text": "$500.00", "extraction_method": "regex",
                 "confidence": 1.0, "page_num": 1, "start": 20, "end": 27,
                 "normalized_value": ""},
            ]),
        ]
        return finalize_extraction(doc)

    def test_json_file_created(self, tmp_path):
        doc = self._make_finalized_doc()
        result = write_extraction_json(doc, str(tmp_path), "", "0.5.0")
        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".extraction.json")

    def test_non_pdf_uses_ext_token(self, tmp_path):
        doc = self._make_finalized_doc(source_file="images/photo.jpg")
        result = write_extraction_json(doc, str(tmp_path), "", __version__)
        assert result is not None
        assert os.path.basename(result) == "photo__jpg.extraction.json"

    def test_json_schema_structure(self, tmp_path):
        doc = self._make_finalized_doc()
        result_path = write_extraction_json(doc, str(tmp_path), "", "0.5.0")
        assert result_path is not None

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Top-level keys
        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "test_id"
        assert data["source_file"] == "test_doc.pdf"
        assert "processing" in data
        assert data["processing"]["pipeline_version"] == "0.5.0"
        assert data["processing"]["extraction_engine"] == "regex"
        assert "extraction_summary" in data
        assert data["extraction_summary"]["total_fields"] == 2
        assert "date" in data["extraction_summary"]["field_type_counts"]
        assert "pages" in data
        assert len(data["pages"]) == 1
        assert len(data["pages"][0]["fields"]) == 2

    def test_json_subfolder_creation(self, tmp_path):
        doc = self._make_finalized_doc(source_file="sub/deep/report.pdf")
        result = write_extraction_json(doc, str(tmp_path), "sub/deep", "0.5.0")
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(
            str(tmp_path), "EXPORT", "EXTRACTION", "sub", "deep"
        )
        assert os.path.isdir(expected_dir)

    def test_path_traversal_protection(self, tmp_path):
        """Path traversal via '..' is neutralized by _sanitize_path_segment.

        The sanitizer strips '.' characters from path segments, turning '..'
        into '' which is filtered out. The result stays safely inside the
        extraction directory. This test verifies the output is contained
        within EXPORT/EXTRACTION/ regardless of traversal attempts.
        """
        doc = self._make_finalized_doc()
        extraction_root = os.path.join(str(tmp_path), "EXPORT", "EXTRACTION")
        result = write_extraction_json(doc, str(tmp_path), "../../etc", "0.5.0")
        # Sanitizer neutralizes ".." so the file is created safely inside
        # EXPORT/EXTRACTION (possibly in an "etc" subfolder).
        assert result is not None
        resolved = os.path.realpath(result)
        assert resolved.startswith(os.path.realpath(extraction_root))

    def test_empty_subfolder(self, tmp_path):
        doc = self._make_finalized_doc()
        result = write_extraction_json(doc, str(tmp_path), "", "0.5.0")
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(str(tmp_path), "EXPORT", "EXTRACTION")
        assert os.path.dirname(result) == expected_dir


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Test behavior when PaddleNLP is not available or fails."""

    def test_no_paddlenlp_extract_fields_uie(self):
        """extract_fields_uie returns empty list when PaddleNLP unavailable."""
        with mock.patch.multiple(
            extraction_module,
            _uie_engine=None,
            _PADDLENLP_AVAILABLE=False,
            _uie_init_failed=False,
        ):
            result = extract_fields_uie("Invoice 2024-01-15 for $500.00", 1)
        assert result == []

    def test_no_paddlenlp_extract_page_fields(self):
        """extract_page_fields falls back to regex when PaddleNLP unavailable."""
        with mock.patch.multiple(
            extraction_module,
            _uie_engine=None,
            _PADDLENLP_AVAILABLE=False,
            _uie_init_failed=False,
        ):
            result = extract_page_fields(
                "Contact user@test.com on 2024-01-15", 1, use_uie=True,
            )
        assert isinstance(result, PageExtraction)
        # Should have regex results (email + date)
        assert len(result.fields) >= 1
        for f in result.fields:
            assert f.get("extraction_method") == "regex"

    def test_uie_init_failure(self):
        """When _uie_init_failed is True, engine returns None."""
        with mock.patch.multiple(
            extraction_module,
            _uie_engine=None,
            _PADDLENLP_AVAILABLE=True,
            _uie_init_failed=True,
        ):
            result = extract_fields_uie("Some text 2024-01-01", 1)
        assert result == []

    def test_special_characters(self):
        """Unicode and special characters in text should not crash extraction."""
        text = "Amount: \u20ac1,234.56 paid by Ren\u00e9 M\u00fcller on 2024-01-15."
        result = extract_fields_regex(text, 1)
        # Should not raise; may or may not find the EUR amount depending on pattern
        assert isinstance(result, list)
        # ISO date should still be found
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1

    def test_very_long_text(self):
        """Extraction should complete on large texts without error."""
        text = "Invoice 2024-01-15 for $100.00. " * 1000
        result = extract_fields_regex(text, 1)
        assert isinstance(result, list)
        # Should find many dates and amounts
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: Date normalization (via regex extraction)
# ---------------------------------------------------------------------------


class TestDateNormalization:
    """Test that regex-extracted dates have correct normalized_value."""

    def test_iso_date_normalized(self):
        result = extract_fields_regex("Date: 2024-01-15 here.", 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1
        assert dates[0].normalized_value == "2024-01-15"

    def test_us_date_normalized(self):
        result = extract_fields_regex("Filed on 01/15/2024 today.", 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1
        assert dates[0].normalized_value == "2024-01-15"

    def test_written_date_normalized(self):
        result = extract_fields_regex("Signed on January 15, 2024 here.", 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1
        assert dates[0].normalized_value == "2024-01-15"

    def test_short_month_date_normalized(self):
        result = extract_fields_regex("Received Jan 15, 2024 at the office.", 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1
        assert dates[0].normalized_value == "2024-01-15"


# ---------------------------------------------------------------------------
# Tests: DEFAULT_SCHEMA constant
# ---------------------------------------------------------------------------


class TestDefaultSchema:
    """Test that DEFAULT_SCHEMA is properly defined."""

    def test_schema_has_all_types(self):
        expected = [
            "Date", "Amount", "Person Name", "Organization",
            "Address", "Reference Number", "Phone Number", "Email Address",
        ]
        assert DEFAULT_SCHEMA == expected

    def test_schema_length(self):
        assert len(DEFAULT_SCHEMA) == 8


# ---------------------------------------------------------------------------
# Tests: _field_to_dict helper
# ---------------------------------------------------------------------------


class TestFieldToDict:
    """Test _field_to_dict conversion."""

    def test_extracted_field_to_dict(self):
        from extraction import _field_to_dict

        f = ExtractedField(
            field_type="date", text="2024-01-15", confidence=0.95,
            page_num=1, start=10, end=20, extraction_method="uie",
            normalized_value="2024-01-15",
        )
        d = _field_to_dict(f)
        assert d["field_type"] == "date"
        assert d["text"] == "2024-01-15"
        assert d["confidence"] == 0.95
        assert d["page_num"] == 1
        assert d["start"] == 10
        assert d["end"] == 20
        assert d["extraction_method"] == "uie"
        assert d["normalized_value"] == "2024-01-15"

    def test_dict_passthrough(self):
        from extraction import _field_to_dict

        d = {"field_type": "amount", "text": "$100"}
        result = _field_to_dict(d)
        assert result == d

    def test_unknown_type_returns_empty(self):
        from extraction import _field_to_dict

        result = _field_to_dict(42)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: UIE type normalization
# ---------------------------------------------------------------------------


class TestUIETypeNormalization:
    """Test _normalize_uie_type mapping."""

    def test_known_labels(self):
        from extraction import _normalize_uie_type

        assert _normalize_uie_type("Date") == "date"
        assert _normalize_uie_type("Amount") == "amount"
        assert _normalize_uie_type("Person Name") == "person_name"
        assert _normalize_uie_type("Organization") == "organization"
        assert _normalize_uie_type("Address") == "address"
        assert _normalize_uie_type("Reference Number") == "reference_number"
        assert _normalize_uie_type("Phone Number") == "phone_number"
        assert _normalize_uie_type("Email Address") == "email_address"

    def test_unknown_label_fallback(self):
        from extraction import _normalize_uie_type

        assert _normalize_uie_type("Custom Field") == "custom_field"
        assert _normalize_uie_type("SomeNewType") == "somenewtype"


# ---------------------------------------------------------------------------
# Tests: Edge cases in regex extraction
# ---------------------------------------------------------------------------


class TestRegexEdgeCases:
    """Additional edge case tests for regex extraction."""

    def test_multiple_amounts_in_text(self):
        text = "Payment of $1,000.00 and $2,500.00 received."
        result = extract_fields_regex(text, 1)
        amounts = [f for f in result if f.field_type == "amount"]
        assert len(amounts) == 2

    def test_multiple_emails_in_text(self):
        text = "Send to alice@example.com and bob@example.org for review."
        result = extract_fields_regex(text, 1)
        emails = [f for f in result if f.field_type == "email_address"]
        assert len(emails) == 2

    def test_start_end_offsets_correct(self):
        text = "Date: 2024-06-15 is important."
        result = extract_fields_regex(text, 1)
        dates = [f for f in result if f.field_type == "date"]
        assert len(dates) >= 1
        d = dates[0]
        assert d.start >= 0
        assert d.end > d.start
        # Verify the extracted text matches the offset range
        assert text[d.start:d.end] == d.text

    def test_page_num_preserved_in_regex(self):
        result = extract_fields_regex("$500.00", 99)
        for f in result:
            assert f.page_num == 99

    def test_gbp_amount(self):
        result = extract_fields_regex("Cost: GBP 2,500.00 total.", 1)
        amounts = [f for f in result if f.field_type == "amount"]
        assert len(amounts) >= 1
        assert any("GBP" in f.text or "2,500.00" in f.text for f in amounts)

    def test_reference_so(self):
        result = extract_fields_regex("Sales order SO-12345 processed.", 1)
        refs = [f for f in result if f.field_type == "reference_number"]
        assert len(refs) >= 1

    def test_reference_doc(self):
        result = extract_fields_regex("Document DOC12345 archived.", 1)
        refs = [f for f in result if f.field_type == "reference_number"]
        assert len(refs) >= 1


# ---------------------------------------------------------------------------
# Tests: Finalization with dict pages (backward compatibility)
# ---------------------------------------------------------------------------


class TestFinalizationDictPages:
    """Test finalize_extraction with dict-based page data (not PageExtraction)."""

    def test_dict_pages_finalized(self):
        doc = DocumentExtraction(document_id="d_dict", source_file="dict_pages.pdf")
        doc.pages = [
            {
                "page_num": 1,
                "fields": [
                    {"field_type": "date", "text": "2024-01-01",
                     "extraction_method": "regex", "confidence": 1.0,
                     "page_num": 1, "start": 0, "end": 10, "normalized_value": ""},
                ],
            },
        ]
        result = finalize_extraction(doc)
        assert result.total_fields == 1
        assert result.field_type_counts["date"] == 1
        assert result.extraction_engine == "regex"

    def test_mixed_page_types(self):
        """Mix of PageExtraction objects and dicts should both work."""
        doc = DocumentExtraction(document_id="d_mix", source_file="mix.pdf")
        doc.pages = [
            PageExtraction(page_num=1, fields=[
                {"field_type": "date", "text": "2024-01-01",
                 "extraction_method": "uie", "confidence": 0.95,
                 "page_num": 1, "start": 0, "end": 10, "normalized_value": ""},
            ]),
            {
                "page_num": 2,
                "fields": [
                    {"field_type": "amount", "text": "$50",
                     "extraction_method": "regex", "confidence": 1.0,
                     "page_num": 2, "start": 0, "end": 3, "normalized_value": ""},
                ],
            },
        ]
        result = finalize_extraction(doc)
        assert result.total_fields == 2
        assert result.extraction_engine == "hybrid"
