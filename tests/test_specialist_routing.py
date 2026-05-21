"""Unit tests for specialist routing from classification output.

Tests cover:
- Built-in specialist selection for each document type
- Custom specialist loading from JSON config
- Confidence threshold filtering
- Extraction for each built-in doc type (invoice, contract, medical, legal, letter)
- Custom pattern matching
- Fallback when no specialist matches
- Integration with classification output (dataclass and dict)
- JSON output format and file paths
- SpecialistRouter singleton lifecycle
- Edge cases: empty text, missing fields, malformed config

Run with: python -m pytest tests/test_specialist_routing.py -v
"""

import json
import os

from classification import DocumentClassification
from specialist_routing import (
    SpecialistConfig,
    SpecialistField,
    SpecialistResult,
    SpecialistRouter,
    _specialist_field_to_dict,
    get_router,
    reset_router,
    write_specialist_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classification(doc_type="invoice", confidence=0.85):
    """Create a DocumentClassification with the given type and confidence."""
    dc = DocumentClassification(document_id="test_doc", source_file="test.pdf")
    dc.document_type = doc_type
    dc.document_confidence = confidence
    return dc


def _make_classification_dict(doc_type="invoice", confidence=0.85):
    """Create a dict-based classification result."""
    return {
        "document_type": doc_type,
        "document_confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Tests: Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_specialist_config_defaults(self):
        sc = SpecialistConfig(doc_type="test", extraction_fields=["a", "b"])
        assert sc.doc_type == "test"
        assert sc.extraction_fields == ["a", "b"]
        assert sc.confidence_threshold == 0.7
        assert sc.custom_patterns == {}
        assert sc.source == "builtin"

    def test_specialist_field_defaults(self):
        sf = SpecialistField(field_name="amount", text="$100.00")
        assert sf.field_name == "amount"
        assert sf.text == "$100.00"
        assert sf.confidence == 1.0
        assert sf.page_num == 0
        assert sf.extraction_method == "specialist_regex"

    def test_specialist_result_defaults(self):
        sr = SpecialistResult(
            document_id="doc1", source_file="test.pdf", doc_type="invoice"
        )
        assert sr.document_id == "doc1"
        assert sr.doc_type == "invoice"
        assert sr.fields == []
        assert sr.total_fields == 0
        assert sr.field_counts == {}

    def test_specialist_field_to_dict(self):
        sf = SpecialistField(
            field_name="total_amount",
            text="$1,234.56",
            confidence=1.0,
            page_num=1,
            start=10,
            end=20,
        )
        d = _specialist_field_to_dict(sf)
        assert d["field_name"] == "total_amount"
        assert d["text"] == "$1,234.56"
        assert d["confidence"] == 1.0
        assert d["page_num"] == 1
        assert d["start"] == 10
        assert d["end"] == 20
        assert d["extraction_method"] == "specialist_regex"


# ---------------------------------------------------------------------------
# Tests: SpecialistRouter — built-in defaults
# ---------------------------------------------------------------------------


class TestRouterDefaults:
    def test_builtin_specialists_loaded(self):
        router = SpecialistRouter()
        specialists = router.specialists
        assert "invoice" in specialists
        assert "contract" in specialists
        assert "medical_record" in specialists
        assert "legal_filing" in specialists
        assert "letter" in specialists
        assert "memo" in specialists
        assert "receipt" in specialists

    def test_invoice_specialist_has_expected_fields(self):
        router = SpecialistRouter()
        inv = router.specialists["invoice"]
        assert "invoice_number" in inv.extraction_fields
        assert "total_amount" in inv.extraction_fields
        assert "vendor" in inv.extraction_fields
        assert "due_date" in inv.extraction_fields

    def test_contract_specialist_has_expected_fields(self):
        router = SpecialistRouter()
        spec = router.specialists["contract"]
        assert "parties" in spec.extraction_fields
        assert "effective_date" in spec.extraction_fields
        assert "governing_law" in spec.extraction_fields

    def test_medical_record_specialist_has_expected_fields(self):
        router = SpecialistRouter()
        spec = router.specialists["medical_record"]
        assert "patient_name" in spec.extraction_fields
        assert "dob" in spec.extraction_fields
        assert "mrn" in spec.extraction_fields
        assert "diagnosis" in spec.extraction_fields

    def test_legal_filing_specialist_has_expected_fields(self):
        router = SpecialistRouter()
        spec = router.specialists["legal_filing"]
        assert "case_number" in spec.extraction_fields
        assert "court" in spec.extraction_fields
        assert "filing_date" in spec.extraction_fields
        assert "judge" in spec.extraction_fields

    def test_letter_specialist_has_expected_fields(self):
        router = SpecialistRouter()
        spec = router.specialists["letter"]
        assert "sender" in spec.extraction_fields
        assert "recipient" in spec.extraction_fields
        assert "subject" in spec.extraction_fields


# ---------------------------------------------------------------------------
# Tests: Routing logic
# ---------------------------------------------------------------------------


class TestRouting:
    def test_route_returns_specialist_for_matching_type(self):
        router = SpecialistRouter()
        cls_result = _make_classification("invoice", 0.85)
        config = router.route(cls_result)
        assert config is not None
        assert config.doc_type == "invoice"

    def test_route_returns_none_for_unknown_type(self):
        router = SpecialistRouter()
        cls_result = _make_classification("other", 0.85)
        config = router.route(cls_result)
        assert config is None

    def test_route_returns_none_below_threshold(self):
        router = SpecialistRouter()
        cls_result = _make_classification("invoice", 0.5)
        config = router.route(cls_result)
        assert config is None

    def test_route_at_exact_threshold(self):
        router = SpecialistRouter()
        cls_result = _make_classification("invoice", 0.7)
        config = router.route(cls_result)
        assert config is not None

    def test_route_with_dict_input(self):
        router = SpecialistRouter()
        cls_dict = _make_classification_dict("contract", 0.9)
        config = router.route(cls_dict)
        assert config is not None
        assert config.doc_type == "contract"

    def test_route_with_dict_below_threshold(self):
        router = SpecialistRouter()
        cls_dict = _make_classification_dict("contract", 0.3)
        config = router.route(cls_dict)
        assert config is None

    def test_route_with_none_input(self):
        router = SpecialistRouter()
        config = router.route(None)
        assert config is None


# ---------------------------------------------------------------------------
# Tests: Extraction — Invoice
# ---------------------------------------------------------------------------


class TestInvoiceExtraction:
    def test_extract_invoice_number(self):
        router = SpecialistRouter()
        config = router.specialists["invoice"]
        text = "Invoice #INV-2024-001\nBill To: Acme Corp"
        fields = router.extract_specialized(text, config)
        inv_fields = [f for f in fields if f.field_name == "invoice_number"]
        assert len(inv_fields) >= 1
        assert "INV-2024-001" in inv_fields[0].text

    def test_extract_total_amount(self):
        router = SpecialistRouter()
        config = router.specialists["invoice"]
        text = "Subtotal: $1,000.00\nTax: $80.00\nTotal: $1,080.00"
        fields = router.extract_specialized(text, config)
        total_fields = [f for f in fields if f.field_name == "total_amount"]
        assert len(total_fields) >= 1
        assert any("1,080.00" in f.text for f in total_fields)

    def test_extract_vendor(self):
        router = SpecialistRouter()
        config = router.specialists["invoice"]
        text = "From: Acme Industries LLC\nInvoice #1234"
        fields = router.extract_specialized(text, config)
        vendor_fields = [f for f in fields if f.field_name == "vendor"]
        assert len(vendor_fields) >= 1
        assert "Acme" in vendor_fields[0].text

    def test_extract_due_date(self):
        router = SpecialistRouter()
        config = router.specialists["invoice"]
        text = "Invoice Date: 01/15/2024\nDue Date: 02/15/2024"
        fields = router.extract_specialized(text, config)
        due_fields = [f for f in fields if f.field_name == "due_date"]
        assert len(due_fields) >= 1
        assert "02/15/2024" in due_fields[0].text

    def test_extract_tax(self):
        router = SpecialistRouter()
        config = router.specialists["invoice"]
        text = "Tax: $125.00\nTotal: $1,375.00"
        fields = router.extract_specialized(text, config)
        tax_fields = [f for f in fields if f.field_name == "tax"]
        assert len(tax_fields) >= 1
        assert "$125.00" in tax_fields[0].text


# ---------------------------------------------------------------------------
# Tests: Extraction — Contract
# ---------------------------------------------------------------------------


class TestContractExtraction:
    def test_extract_parties(self):
        router = SpecialistRouter()
        config = router.specialists["contract"]
        text = "This Agreement is between Acme Corporation and Beta LLC."
        fields = router.extract_specialized(text, config)
        party_fields = [f for f in fields if f.field_name == "parties"]
        assert len(party_fields) >= 1

    def test_extract_effective_date(self):
        router = SpecialistRouter()
        config = router.specialists["contract"]
        text = "This agreement is effective date: January 15, 2024."
        fields = router.extract_specialized(text, config)
        date_fields = [f for f in fields if f.field_name == "effective_date"]
        assert len(date_fields) >= 1
        assert "January 15, 2024" in date_fields[0].text

    def test_extract_term(self):
        router = SpecialistRouter()
        config = router.specialists["contract"]
        text = "The term of this agreement shall be 12 months from the effective date."
        fields = router.extract_specialized(text, config)
        term_fields = [f for f in fields if f.field_name == "term"]
        assert len(term_fields) >= 1
        assert "12 months" in term_fields[0].text

    def test_extract_governing_law(self):
        router = SpecialistRouter()
        config = router.specialists["contract"]
        text = "This agreement shall be governed by the laws of the State of California."
        fields = router.extract_specialized(text, config)
        law_fields = [f for f in fields if f.field_name == "governing_law"]
        assert len(law_fields) >= 1
        assert "California" in law_fields[0].text


# ---------------------------------------------------------------------------
# Tests: Extraction — Medical Record
# ---------------------------------------------------------------------------


class TestMedicalRecordExtraction:
    def test_extract_patient_name(self):
        router = SpecialistRouter()
        config = router.specialists["medical_record"]
        text = "Patient Name: John Doe\nDOB: 05/15/1980"
        fields = router.extract_specialized(text, config)
        name_fields = [f for f in fields if f.field_name == "patient_name"]
        assert len(name_fields) >= 1
        assert "John Doe" in name_fields[0].text

    def test_extract_dob(self):
        router = SpecialistRouter()
        config = router.specialists["medical_record"]
        text = "Patient: Jane Smith\nDate of Birth: 03/22/1975"
        fields = router.extract_specialized(text, config)
        dob_fields = [f for f in fields if f.field_name == "dob"]
        assert len(dob_fields) >= 1
        assert "03/22/1975" in dob_fields[0].text

    def test_extract_mrn(self):
        router = SpecialistRouter()
        config = router.specialists["medical_record"]
        text = "MRN: 12345678\nAdmission Date: 01/01/2024"
        fields = router.extract_specialized(text, config)
        mrn_fields = [f for f in fields if f.field_name == "mrn"]
        assert len(mrn_fields) >= 1
        assert "12345678" in mrn_fields[0].text

    def test_extract_diagnosis(self):
        router = SpecialistRouter()
        config = router.specialists["medical_record"]
        text = "Diagnosis: Acute bronchitis with secondary infection"
        fields = router.extract_specialized(text, config)
        diag_fields = [f for f in fields if f.field_name == "diagnosis"]
        assert len(diag_fields) >= 1
        assert "bronchitis" in diag_fields[0].text.lower()

    def test_extract_provider(self):
        router = SpecialistRouter()
        config = router.specialists["medical_record"]
        text = "Attending: Dr. Sarah Johnson\nSpecialty: Internal Medicine"
        fields = router.extract_specialized(text, config)
        # The pattern matches "Dr." prefix
        provider_fields = [f for f in fields if f.field_name == "provider"]
        assert len(provider_fields) >= 1


# ---------------------------------------------------------------------------
# Tests: Extraction — Legal Filing
# ---------------------------------------------------------------------------


class TestLegalFilingExtraction:
    def test_extract_case_number(self):
        router = SpecialistRouter()
        config = router.specialists["legal_filing"]
        text = "Case No. 2024-CV-12345\nIN THE SUPERIOR COURT"
        fields = router.extract_specialized(text, config)
        case_fields = [f for f in fields if f.field_name == "case_number"]
        assert len(case_fields) >= 1
        assert "2024-CV-12345" in case_fields[0].text

    def test_extract_court(self):
        router = SpecialistRouter()
        config = router.specialists["legal_filing"]
        text = "IN THE DISTRICT COURT OF the Northern District"
        fields = router.extract_specialized(text, config)
        court_fields = [f for f in fields if f.field_name == "court"]
        assert len(court_fields) >= 1

    def test_extract_filing_date(self):
        router = SpecialistRouter()
        config = router.specialists["legal_filing"]
        text = "Filed: 03/15/2024\nCase No. 2024-CV-001"
        fields = router.extract_specialized(text, config)
        date_fields = [f for f in fields if f.field_name == "filing_date"]
        assert len(date_fields) >= 1
        assert "03/15/2024" in date_fields[0].text

    def test_extract_judge(self):
        router = SpecialistRouter()
        config = router.specialists["legal_filing"]
        text = "Hon. Robert Smith, presiding\nCase No. 123"
        fields = router.extract_specialized(text, config)
        judge_fields = [f for f in fields if f.field_name == "judge"]
        assert len(judge_fields) >= 1
        assert "Robert Smith" in judge_fields[0].text


# ---------------------------------------------------------------------------
# Tests: Extraction — Correspondence
# ---------------------------------------------------------------------------


class TestCorrespondenceExtraction:
    def test_extract_sender(self):
        router = SpecialistRouter()
        config = router.specialists["letter"]
        text = "From: Acme Corporation\nTo: Beta LLC"
        fields = router.extract_specialized(text, config)
        sender_fields = [f for f in fields if f.field_name == "sender"]
        assert len(sender_fields) >= 1
        assert "Acme" in sender_fields[0].text

    def test_extract_recipient(self):
        router = SpecialistRouter()
        config = router.specialists["letter"]
        text = "From: Acme Corp\nDear John Smith,"
        fields = router.extract_specialized(text, config)
        recip_fields = [f for f in fields if f.field_name == "recipient"]
        assert len(recip_fields) >= 1

    def test_extract_subject(self):
        router = SpecialistRouter()
        config = router.specialists["letter"]
        text = "Subject: Quarterly Review Results\nDear Board Members,"
        fields = router.extract_specialized(text, config)
        subj_fields = [f for f in fields if f.field_name == "subject"]
        assert len(subj_fields) >= 1
        assert "Quarterly" in subj_fields[0].text

    def test_extract_reference_number(self):
        router = SpecialistRouter()
        config = router.specialists["letter"]
        text = "Ref: ABC-12345\nDear Sir,"
        fields = router.extract_specialized(text, config)
        ref_fields = [f for f in fields if f.field_name == "reference_number"]
        assert len(ref_fields) >= 1
        assert "ABC-12345" in ref_fields[0].text

    def test_memo_shares_correspondence_fields(self):
        router = SpecialistRouter()
        config = router.specialists["memo"]
        text = "From: Legal Department\nSubject: Policy Update"
        fields = router.extract_specialized(text, config)
        assert any(f.field_name == "sender" for f in fields)
        assert any(f.field_name == "subject" for f in fields)


# ---------------------------------------------------------------------------
# Tests: Custom config loading
# ---------------------------------------------------------------------------


class TestCustomConfig:
    def test_load_custom_config(self, tmp_path):
        config_file = tmp_path / "specialists.json"
        config_file.write_text(json.dumps({
            "specialists": {
                "purchase_order": {
                    "extraction_fields": ["po_number", "vendor", "total"],
                    "confidence_threshold": 0.8,
                    "custom_patterns": {
                        "po_number": r"PO[\s-]?(\d{4,10})"
                    }
                }
            }
        }), encoding="utf-8")

        router = SpecialistRouter(config_path=str(config_file))
        assert "purchase_order" in router.specialists
        spec = router.specialists["purchase_order"]
        assert spec.source == "custom"
        assert spec.confidence_threshold == 0.8
        assert "po_number" in spec.extraction_fields

    def test_custom_extraction_works(self, tmp_path):
        config_file = tmp_path / "specialists.json"
        config_file.write_text(json.dumps({
            "specialists": {
                "purchase_order": {
                    "extraction_fields": ["po_number"],
                    "custom_patterns": {
                        "po_number": r"PO[\s-]?(\d{4,10})"
                    }
                }
            }
        }), encoding="utf-8")

        router = SpecialistRouter(config_path=str(config_file))
        config = router.specialists["purchase_order"]
        fields = router.extract_specialized("Purchase Order PO-12345678", config)
        assert len(fields) >= 1
        assert fields[0].field_name == "po_number"
        assert "12345678" in fields[0].text

    def test_custom_augments_builtin(self, tmp_path):
        config_file = tmp_path / "specialists.json"
        config_file.write_text(json.dumps({
            "specialists": {
                "invoice": {
                    "extraction_fields": ["purchase_order"],
                    "custom_patterns": {
                        "purchase_order": r"PO[\s#:-]*(\d{4,10})"
                    }
                }
            }
        }), encoding="utf-8")

        router = SpecialistRouter(config_path=str(config_file))
        spec = router.specialists["invoice"]
        assert spec.source == "builtin+custom"
        # Should have both original and augmented fields
        assert "invoice_number" in spec.extraction_fields
        assert "purchase_order" in spec.extraction_fields

    def test_custom_pattern_extraction_merges(self, tmp_path):
        config_file = tmp_path / "specialists.json"
        config_file.write_text(json.dumps({
            "specialists": {
                "invoice": {
                    "extraction_fields": ["purchase_order"],
                    "custom_patterns": {
                        "purchase_order": r"PO[\s#:-]*(\d{4,10})"
                    }
                }
            }
        }), encoding="utf-8")

        router = SpecialistRouter(config_path=str(config_file))
        config = router.specialists["invoice"]
        text = "Invoice #INV-2024-001\nPO# 99887766"
        fields = router.extract_specialized(text, config)
        field_names = {f.field_name for f in fields}
        assert "invoice_number" in field_names
        assert "purchase_order" in field_names

    def test_invalid_config_file_graceful(self, tmp_path):
        """Non-existent config file logs warning but does not crash."""
        router = SpecialistRouter(config_path=str(tmp_path / "nonexistent.json"))
        assert "invoice" in router.specialists  # builtins still loaded

    def test_invalid_json_graceful(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not valid json{{{", encoding="utf-8")
        router = SpecialistRouter(config_path=str(config_file))
        assert "invoice" in router.specialists

    def test_invalid_regex_skipped(self, tmp_path):
        config_file = tmp_path / "bad_regex.json"
        config_file.write_text(json.dumps({
            "specialists": {
                "custom_type": {
                    "extraction_fields": ["bad_field"],
                    "custom_patterns": {
                        "bad_field": r"(unclosed group"
                    }
                }
            }
        }), encoding="utf-8")
        router = SpecialistRouter(config_path=str(config_file))
        assert "custom_type" in router.specialists

    def test_custom_config_bad_specialists_key(self, tmp_path):
        """specialists key that is not a dict should be skipped."""
        config_file = tmp_path / "bad_key.json"
        config_file.write_text(json.dumps({
            "specialists": "not a dict"
        }), encoding="utf-8")
        router = SpecialistRouter(config_path=str(config_file))
        # builtins still loaded
        assert "invoice" in router.specialists

    def test_custom_config_bad_entry_skipped(self, tmp_path):
        """Individual specialist entries that are not dicts should be skipped."""
        config_file = tmp_path / "bad_entry.json"
        config_file.write_text(json.dumps({
            "specialists": {
                "bad_entry": "not a dict",
                "good_entry": {
                    "extraction_fields": ["field1"],
                    "custom_patterns": {"field1": r"(test)"}
                }
            }
        }), encoding="utf-8")
        router = SpecialistRouter(config_path=str(config_file))
        assert "good_entry" in router.specialists
        assert "bad_entry" not in router.specialists


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_text_returns_no_fields(self):
        router = SpecialistRouter()
        config = router.specialists["invoice"]
        fields = router.extract_specialized("", config)
        assert fields == []

    def test_none_config_returns_no_fields(self):
        router = SpecialistRouter()
        fields = router.extract_specialized("some text", None)
        assert fields == []

    def test_text_with_no_matching_patterns(self):
        router = SpecialistRouter()
        config = router.specialists["invoice"]
        text = "This is a random paragraph with no financial data."
        fields = router.extract_specialized(text, config)
        assert fields == []

    def test_add_specialist_programmatic(self):
        router = SpecialistRouter()
        custom = SpecialistConfig(
            doc_type="blueprint",
            extraction_fields=["project_name"],
            confidence_threshold=0.6,
        )
        router.add_specialist(custom)
        assert "blueprint" in router.specialists
        assert router.specialists["blueprint"].confidence_threshold == 0.6


# ---------------------------------------------------------------------------
# Tests: process_document end-to-end
# ---------------------------------------------------------------------------


class TestProcessDocument:
    def test_process_invoice(self):
        router = SpecialistRouter()
        cls_result = _make_classification("invoice", 0.9)
        text = (
            "Invoice #INV-2024-001\n"
            "From: Acme Industries\n"
            "Due Date: 02/28/2024\n"
            "Total: $5,250.00\n"
            "Tax: $420.00"
        )
        result = router.process_document(cls_result, text, "doc1", "invoice.pdf")
        assert result is not None
        assert result.doc_type == "invoice"
        assert result.total_fields > 0
        assert result.confidence == 0.9
        field_names = {f["field_name"] for f in result.fields}
        assert "invoice_number" in field_names
        assert "total_amount" in field_names

    def test_process_no_match_returns_none(self):
        router = SpecialistRouter()
        cls_result = _make_classification("other", 0.9)
        result = router.process_document(cls_result, "Some text", "doc2", "other.pdf")
        assert result is None

    def test_process_below_threshold_returns_none(self):
        router = SpecialistRouter()
        cls_result = _make_classification("invoice", 0.3)
        result = router.process_document(cls_result, "Invoice #123", "doc3", "inv.pdf")
        assert result is None

    def test_process_with_dict_classification(self):
        router = SpecialistRouter()
        cls_dict = _make_classification_dict("contract", 0.85)
        text = "This Agreement is between Alpha Corp and Beta LLC effective date: 01/01/2025"
        result = router.process_document(cls_dict, text, "doc4", "contract.pdf")
        assert result is not None
        assert result.doc_type == "contract"
        assert result.total_fields > 0

    def test_process_field_counts_populated(self):
        router = SpecialistRouter()
        cls_result = _make_classification("invoice", 0.9)
        text = (
            "Invoice #INV-001\n"
            "Total: $100.00\n"
            "Tax: $8.00"
        )
        result = router.process_document(cls_result, text, "doc5", "inv.pdf")
        assert result is not None
        assert result.field_counts
        assert sum(result.field_counts.values()) == result.total_fields


# ---------------------------------------------------------------------------
# Tests: JSON output
# ---------------------------------------------------------------------------


class TestWriteSpecialistJson:
    def test_write_json_creates_file(self, tmp_path):
        result = SpecialistResult(
            document_id="doc1",
            source_file="invoice.pdf",
            doc_type="invoice",
            specialist_source="builtin",
            confidence=0.85,
            fields=[{
                "field_name": "invoice_number",
                "text": "INV-001",
                "confidence": 1.0,
                "page_num": 0,
                "start": 0,
                "end": 7,
                "extraction_method": "specialist_regex",
            }],
            total_fields=1,
            field_counts={"invoice_number": 1},
        )
        path = write_specialist_json(result, str(tmp_path), ".", "0.9.0")
        assert path is not None
        assert os.path.exists(path)
        assert path.endswith(".specialist.json")

    def test_json_schema_version(self, tmp_path):
        result = SpecialistResult(
            document_id="doc2",
            source_file="test.pdf",
            doc_type="invoice",
        )
        path = write_specialist_json(result, str(tmp_path), ".", "0.9.0")
        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "doc2"
        assert data["processing"]["specialist_doc_type"] == "invoice"
        assert data["processing"]["pipeline_version"] == "0.9.0"

    def test_json_with_subfolder(self, tmp_path):
        result = SpecialistResult(
            document_id="doc3",
            source_file="sub/dir/contract.pdf",
            doc_type="contract",
        )
        path = write_specialist_json(result, str(tmp_path), "sub/dir", "0.9.0")
        assert path is not None
        assert "sub" in path or "dir" in path

    def test_json_path_traversal_blocked(self, tmp_path):
        result = SpecialistResult(
            document_id="doc4",
            source_file="test.pdf",
            doc_type="invoice",
        )
        # Use a traversal deep enough to escape EXPORT/EXTRACTION base dir
        path = write_specialist_json(
            result, str(tmp_path), "../../../../../../../tmp/evil", "0.9.0"
        )
        # On platforms where realpath resolves the traversal outside the
        # extraction dir, this should be blocked. On Windows where the
        # resolved path may still land under tmp_path, we accept that
        # the protection functions as designed (resolved stays under root).
        if path is not None:
            extraction_base = os.path.realpath(
                os.path.join(str(tmp_path), "EXPORT", "EXTRACTION")
            )
            assert os.path.realpath(os.path.dirname(path)).startswith(
                extraction_base
            )

    def test_json_fields_included(self, tmp_path):
        result = SpecialistResult(
            document_id="doc5",
            source_file="medical.pdf",
            doc_type="medical_record",
            confidence=0.92,
            fields=[
                {
                    "field_name": "patient_name",
                    "text": "John Doe",
                    "confidence": 1.0,
                    "page_num": 1,
                    "start": 15,
                    "end": 23,
                    "extraction_method": "specialist_regex",
                },
                {
                    "field_name": "mrn",
                    "text": "12345678",
                    "confidence": 1.0,
                    "page_num": 1,
                    "start": 30,
                    "end": 38,
                    "extraction_method": "specialist_regex",
                },
            ],
            total_fields=2,
            field_counts={"patient_name": 1, "mrn": 1},
        )
        path = write_specialist_json(result, str(tmp_path), ".", "0.9.0")
        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["specialist_summary"]["total_fields"] == 2
        assert len(data["fields"]) == 2
        assert data["specialist_summary"]["field_counts"]["patient_name"] == 1


# ---------------------------------------------------------------------------
# Tests: Singleton router lifecycle
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_router_returns_same_instance(self):
        reset_router()
        r1 = get_router()
        r2 = get_router()
        assert r1 is r2

    def test_reset_router_creates_new_instance(self):
        reset_router()
        r1 = get_router()
        reset_router()
        r2 = get_router()
        assert r1 is not r2

    def test_reset_router_clears_state(self):
        reset_router()
        router = get_router()
        router.add_specialist(
            SpecialistConfig(doc_type="temp_type", extraction_fields=["x"])
        )
        assert "temp_type" in router.specialists
        reset_router()
        router2 = get_router()
        assert "temp_type" not in router2.specialists


# ---------------------------------------------------------------------------
# Tests: End-to-end integration with classification + JSON output
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_invoice_pipeline(self, tmp_path):
        """Simulate: classify -> route -> extract -> write JSON."""
        router = SpecialistRouter()

        cls_result = _make_classification("invoice", 0.92)
        text = (
            "INVOICE\n"
            "Invoice #INV-2024-0567\n"
            "From: Global Supplies Inc.\n"
            "Due Date: March 15, 2024\n"
            "Subtotal: $2,500.00\n"
            "Tax: $200.00\n"
            "Total: $2,700.00\n"
        )

        result = router.process_document(
            cls_result, text, "test_inv_001", "invoices/inv-2024.pdf"
        )
        assert result is not None
        assert result.doc_type == "invoice"
        assert result.total_fields > 0

        path = write_specialist_json(result, str(tmp_path), "invoices", "0.9.0")
        assert path is not None

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["specialist_summary"]["doc_type"] == "invoice"
        assert data["processing"]["classification_confidence"] == 0.92
        assert len(data["fields"]) > 0

    def test_full_contract_pipeline(self, tmp_path):
        router = SpecialistRouter()

        cls_result = _make_classification("contract", 0.88)
        text = (
            "AGREEMENT\n"
            "This Agreement is between Alpha Corp and Beta LLC.\n"
            "Effective Date: January 1, 2025\n"
            "The term of this contract shall be 24 months.\n"
            "Governing Law: This agreement shall be governed by the laws of "
            "the State of New York.\n"
        )

        result = router.process_document(
            cls_result, text, "test_contract_001", "contracts/agreement.pdf"
        )
        assert result is not None
        assert result.doc_type == "contract"

        path = write_specialist_json(result, str(tmp_path), "contracts", "0.9.0")
        assert path is not None

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        field_names = {f["field_name"] for f in data["fields"]}
        assert "parties" in field_names
        assert "effective_date" in field_names
        assert "term" in field_names
        assert "governing_law" in field_names
