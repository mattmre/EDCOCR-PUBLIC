"""
Unit tests for document classification module (classification.py).

Tests cover:
- Text pattern matching for each document type
- Layout feature analysis
- Ensemble weighted combination
- Finalization: majority vote, distribution, confidence
- JSON output format and file paths
- Edge cases: empty text, no layout, single page

Run with: python -m pytest tests/test_classification.py -v
"""

import json
import os
import re
import sys

# Add project root to path
from classification import (
    DOCUMENT_TYPES,
    DocumentClassification,
    PageClassification,
    _load_classification_profiles,
    classify_page_by_layout,
    classify_page_by_text,
    classify_page_ensemble,
    finalize_classification,
    write_classification_json,
)
from version import __version__

# ---------------------------------------------------------------------------
# Tests: Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_page_classification_defaults(self):
        pc = PageClassification(page_num=1)
        assert pc.predicted_type == "other"
        assert pc.confidence == 0.0
        assert pc.method == ""
        assert pc.type_scores == {}
        assert pc.is_handwritten is False
        assert pc.profile_matches == []
        assert pc.page_num == 1

    def test_document_classification_defaults(self):
        dc = DocumentClassification(document_id="doc1", source_file="test.pdf")
        assert dc.document_type == "other"
        assert dc.document_confidence == 0.0
        assert dc.pages == []
        assert dc.type_distribution == {}
        assert dc.document_type_scores == {}
        assert dc.document_labels == []
        assert dc.custom_profile_matches == []
        assert dc.document_id == "doc1"
        assert dc.source_file == "test.pdf"

    def test_document_types_list(self):
        """DOCUMENT_TYPES contains the 10 expected categories."""
        expected = {
            "invoice", "contract", "letter", "form", "report",
            "memo", "receipt", "handwritten", "photograph", "other",
        }
        assert set(DOCUMENT_TYPES) == expected
        assert len(DOCUMENT_TYPES) == 10


# ---------------------------------------------------------------------------
# Tests: Text classification (Tier 1)
# ---------------------------------------------------------------------------


class TestTextClassification:
    def test_invalid_profile_payload_returns_empty(self, tmp_path):
        monkeypatch_root = tmp_path / "root"
        monkeypatch_root.mkdir()
        profile_path = monkeypatch_root / "profiles.json"
        profile_path.write_text('["not-an-object"]', encoding="utf-8")
        module = sys.modules["classification"]
        original_root = module._CLASSIFICATION_PROFILE_ROOT
        module._CLASSIFICATION_PROFILE_ROOT = str(monkeypatch_root)
        try:
            assert _load_classification_profiles("profiles.json") == []
        finally:
            module._CLASSIFICATION_PROFILE_ROOT = original_root

    def test_profile_path_outside_root_is_blocked(self, tmp_path):
        allowed_root = tmp_path / "allowed"
        allowed_root.mkdir()
        outside_path = tmp_path / "outside.json"
        outside_path.write_text('{"profiles": []}', encoding="utf-8")
        module = sys.modules["classification"]
        original_root = module._CLASSIFICATION_PROFILE_ROOT
        module._CLASSIFICATION_PROFILE_ROOT = str(allowed_root)
        try:
            assert _load_classification_profiles(str(outside_path)) == []
        finally:
            module._CLASSIFICATION_PROFILE_ROOT = original_root

    def test_invoice_text(self):
        text = "Invoice #12345\nAmount Due: $500.00\nTotal: $500.00"
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "invoice"
        assert result.confidence > 0.0

    def test_contract_text(self):
        text = (
            "This Agreement is entered into between the parties.\n"
            "WHEREAS the undersigned hereby agrees to the terms."
        )
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "contract"
        assert result.confidence > 0.0

    def test_letter_text(self):
        text = "Dear Mr. Johnson,\n\nThank you for your inquiry.\n\nSincerely,\nJane"
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "letter"
        assert result.confidence > 0.0

    def test_form_text(self):
        text = (
            "Please complete all required fields.\n"
            "Applicant Name: ___________\n"
            "Signature line: ___________"
        )
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "form"
        assert result.confidence > 0.0

    def test_report_text(self):
        text = (
            "Executive Summary\n"
            "This report presents the findings of the investigation.\n"
            "Methodology: The team used the following approach.\n"
            "Conclusion: The results support the hypothesis."
        )
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "report"
        assert result.confidence > 0.0

    def test_memo_text(self):
        text = "Memorandum\nTo: All Staff\nFrom: Management\nSubject: Policy Update"
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "memo"
        assert result.confidence > 0.0

    def test_receipt_text(self):
        text = "Receipt\nTransaction ID: 78901\nPaid: $25.00\nChange Due: $5.00"
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "receipt"
        assert result.confidence > 0.0

    def test_empty_text(self):
        result = classify_page_by_text("", 1)
        assert result.predicted_type == "other"
        assert result.confidence == 0.0
        assert result.type_scores == {}

    def test_generic_text(self):
        result = classify_page_by_text("Hello world, this is a simple sentence.", 1)
        assert result.predicted_type == "other"
        assert result.confidence == 0.0

    def test_case_insensitive(self):
        """Both 'INVOICE' and 'invoice' should match the invoice type."""
        upper = classify_page_by_text("INVOICE for services rendered", 1)
        lower = classify_page_by_text("invoice for services rendered", 1)
        assert upper.predicted_type == "invoice"
        assert lower.predicted_type == "invoice"
        assert upper.confidence == lower.confidence

    def test_multiple_type_matches(self):
        """Text matching multiple types should pick the higher-scoring one."""
        text = (
            "Invoice Total: $100.00\n"
            "Receipt of payment. Transaction completed. Paid in full."
        )
        result = classify_page_by_text(text, 1)
        # Both invoice and receipt keywords match; the one with more hits wins
        assert result.predicted_type in ("invoice", "receipt")
        assert result.confidence > 0.0
        # Verify multiple types scored
        assert len(result.type_scores) >= 2

    def test_invoice_negative(self):
        """Text about letters should not match invoice."""
        text = "Dear friend, Sincerely yours, best regards"
        result = classify_page_by_text(text, 1)
        assert result.predicted_type != "invoice"

    def test_contract_negative(self):
        """Random unrelated text should not match contract."""
        text = "The weather is nice today. Let's go for a walk."
        result = classify_page_by_text(text, 1)
        assert result.predicted_type != "contract"

    def test_memo_without_colocation(self):
        """Just 'memo' without To/From/Subject co-located fields.

        The memo rule requires co_locate fields to boost the score, so a bare
        'memo' keyword will match but with fewer hits.
        """
        text = "This is a memo about the upcoming event."
        result = classify_page_by_text(text, 1)
        # 'memo' keyword matches, but co-locate requirement may lower confidence
        if result.predicted_type == "memo":
            assert result.confidence > 0.0

    def test_memo_with_colocation(self):
        """'Memorandum' + To/From/Subject triggers co-locate boost."""
        text = "Memorandum\nTo: Team\nFrom: Director\nSubject: Budget"
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "memo"
        # Confidence should be higher than without co-location
        memo_no_coloc = classify_page_by_text("This is a memo about updates.", 1)
        if memo_no_coloc.predicted_type == "memo":
            assert result.confidence >= memo_no_coloc.confidence

    def test_type_scores_populated(self):
        """type_scores dict should have entries for each matched type."""
        text = "Invoice Total: $100.00"
        result = classify_page_by_text(text, 1)
        assert "invoice" in result.type_scores
        assert result.type_scores["invoice"] > 0.0

    def test_method_is_text_rules(self):
        result = classify_page_by_text("some text", 1)
        assert result.method == "text_rules"

    def test_page_num_preserved(self):
        result = classify_page_by_text("Dear Sir,", 42)
        assert result.page_num == 42

    def test_whitespace_handling(self):
        """Text with extra whitespace should still match keywords."""
        text = "   Invoice   \n\n\n   Amount   Due   :   $100   "
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "invoice"

    def test_multiline_text(self):
        """Keywords spread across multiple lines should be detected."""
        text = "Line1: agreement\nLine2: hereby\nLine3: parties"
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "contract"
        assert result.confidence > 0.0

    def test_customer_profile_matches(self, monkeypatch):
        monkeypatch.setattr(
            sys.modules["classification"],
            "_CLASSIFICATION_PROFILES",
            [
                {
                    "name": "invoice_packet",
                    "base_type": "invoice",
                    "route": "finance_invoice_review",
                    "weight": 1.0,
                    "patterns": [
                        re.compile(r"\bwire instructions\b", re.IGNORECASE),
                        re.compile(r"\bremit to\b", re.IGNORECASE),
                    ],
                }
            ],
        )
        result = classify_page_by_text(
            "Invoice\nWire instructions enclosed\nPlease remit to accounts payable",
            1,
        )
        assert result.predicted_type == "invoice"
        assert result.profile_matches[0]["name"] == "invoice_packet"
        assert result.profile_matches[0]["route"] == "finance_invoice_review"


# ---------------------------------------------------------------------------
# Tests: Layout classification (Tier 2)
# ---------------------------------------------------------------------------


class TestLayoutClassification:
    def test_invoice_layout(self):
        """Tables with monetary text content should suggest invoice."""
        regions = [
            {"type": "text", "text": "Amount Due: $500"},
            {"type": "text", "text": "Total: $1000"},
        ]
        tables = [{"html": "<table>...</table>"}]
        result = classify_page_by_layout(regions, tables, [], 1)
        assert "invoice" in result.type_scores
        assert result.type_scores["invoice"] > 0.0

    def test_form_layout(self):
        """Many form fields should classify as form."""
        fields = [
            {"label": "Name", "value": ""},
            {"label": "DOB", "value": ""},
            {"label": "Address", "value": ""},
        ]
        result = classify_page_by_layout([], [], fields, 1)
        assert result.predicted_type == "form"
        assert result.confidence > 0.0

    def test_photograph_layout(self):
        """More figure regions than text should classify as photograph."""
        regions = [
            {"type": "figure"},
            {"type": "figure"},
            {"type": "figure"},
            {"type": "text"},
        ]
        result = classify_page_by_layout(regions, [], [], 1)
        assert result.predicted_type == "photograph"
        assert result.confidence > 0.0

    def test_report_layout(self):
        """Multiple titles + text blocks should suggest report."""
        regions = [
            {"type": "title"},
            {"type": "title"},
            {"type": "text"},
            {"type": "text"},
            {"type": "text"},
        ]
        result = classify_page_by_layout(regions, [], [], 1)
        assert "report" in result.type_scores
        assert result.type_scores["report"] > 0.0

    def test_empty_layout(self):
        """No regions, no tables, no fields should give zero confidence."""
        result = classify_page_by_layout([], [], [], 1)
        assert result.predicted_type == "other"
        assert result.confidence == 0.0
        assert result.type_scores == {}

    def test_tables_only(self):
        """Tables without monetary content should still produce some classification."""
        tables = [{"html": "<table>data</table>"}]
        result = classify_page_by_layout([], tables, [], 1)
        # Tables without money suggest report
        assert result.predicted_type in ("report", "other")
        assert result.confidence >= 0.0

    def test_form_fields_only(self):
        """Form fields alone should classify as form."""
        fields = [{"label": "Field1"}, {"label": "Field2"}]
        result = classify_page_by_layout([], [], fields, 1)
        assert result.predicted_type == "form"
        assert result.confidence > 0.0

    def test_mixed_layout(self):
        """Tables + form fields + text regions produce classification."""
        regions = [
            {"type": "text", "text": "Total: $200"},
        ]
        tables = [{"html": "<table>...</table>"}]
        fields = [{"label": "Name"}, {"label": "Date"}, {"label": "Amount"}]
        result = classify_page_by_layout(regions, tables, fields, 1)
        assert result.predicted_type in ("form", "invoice", "receipt")
        assert result.confidence > 0.0

    def test_method_is_layout_features(self):
        regions = [{"type": "text"}]
        result = classify_page_by_layout(regions, [], [], 1)
        assert result.method == "layout_features"

    def test_single_text_region(self):
        """Minimal layout with a single text region -- low or no signal."""
        regions = [{"type": "text"}]
        result = classify_page_by_layout(regions, [], [], 1)
        # Single text region without other features lacks strong signal
        assert result.confidence <= 0.5

    def test_many_figures(self):
        """Many figure regions with no text should score high as photograph."""
        regions = [{"type": "figure"} for _ in range(5)]
        result = classify_page_by_layout(regions, [], [], 1)
        assert result.predicted_type == "photograph"
        assert result.confidence > 0.5

    def test_layout_with_titles(self):
        """Titles + text blocks should score for report."""
        regions = (
            [{"type": "title"} for _ in range(3)]
            + [{"type": "text"} for _ in range(4)]
        )
        result = classify_page_by_layout(regions, [], [], 1)
        assert "report" in result.type_scores


# ---------------------------------------------------------------------------
# Tests: Ensemble classification
# ---------------------------------------------------------------------------


class TestEnsemble:
    def test_text_and_layout_agree(self):
        """Both tiers agree on invoice -- ensemble should also say invoice."""
        text_result = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.6, method="text_rules",
            type_scores={"invoice": 0.6},
        )
        layout_result = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.7, method="layout_features",
            type_scores={"invoice": 0.7},
        )
        result = classify_page_ensemble(text_result, layout_result, 1)
        assert result.predicted_type == "invoice"
        assert result.method == "ensemble"
        # Combined score: 0.6*0.6 + 0.7*0.4 = 0.36 + 0.28 = 0.64
        assert result.confidence > 0.5

    def test_text_overrides_layout(self):
        """Text says contract strongly, layout says other -- text weight (0.6) wins."""
        text_result = PageClassification(
            page_num=1, predicted_type="contract",
            confidence=0.8, method="text_rules",
            type_scores={"contract": 0.8},
        )
        layout_result = PageClassification(
            page_num=1, predicted_type="form",
            confidence=0.3, method="layout_features",
            type_scores={"form": 0.3},
        )
        result = classify_page_ensemble(text_result, layout_result, 1)
        # contract: 0.8*0.6 = 0.48, form: 0.3*0.4 = 0.12
        assert result.predicted_type == "contract"

    def test_layout_contribution(self):
        """Layout says form strongly, text says other -- form may win."""
        text_result = PageClassification(
            page_num=1, predicted_type="other",
            confidence=0.0, method="text_rules",
            type_scores={},
        )
        layout_result = PageClassification(
            page_num=1, predicted_type="form",
            confidence=0.9, method="layout_features",
            type_scores={"form": 0.9},
        )
        result = classify_page_ensemble(text_result, layout_result, 1)
        # form: 0.9*0.4 = 0.36
        assert result.predicted_type == "form"
        assert result.confidence > 0.0

    def test_layout_zero_confidence_fallback(self):
        """Layout confidence=0.0 should fall back to text result."""
        text_result = PageClassification(
            page_num=1, predicted_type="letter",
            confidence=0.5, method="text_rules",
            type_scores={"letter": 0.5},
        )
        layout_result = PageClassification(
            page_num=1, predicted_type="other",
            confidence=0.0, method="layout_features",
            type_scores={},
        )
        result = classify_page_ensemble(text_result, layout_result, 1)
        assert result.predicted_type == "letter"
        assert result.confidence == 0.5
        assert result.method == "ensemble"

    def test_layout_none_fallback(self):
        """Missing layout result should fall back to text-only classification."""
        text_result = PageClassification(
            page_num=1, predicted_type="letter",
            confidence=0.5, method="text_rules",
            type_scores={"letter": 0.5},
        )
        result = classify_page_ensemble(text_result, None, 1)
        assert result.predicted_type == "letter"
        assert result.confidence == 0.5
        assert result.method == "ensemble"

    def test_method_is_ensemble(self):
        text_result = PageClassification(
            page_num=1, predicted_type="other",
            confidence=0.0, method="text_rules",
            type_scores={},
        )
        layout_result = PageClassification(
            page_num=1, predicted_type="other",
            confidence=0.0, method="layout_features",
            type_scores={},
        )
        result = classify_page_ensemble(text_result, layout_result, 1)
        assert result.method == "ensemble"

    def test_type_scores_merged(self):
        """type_scores from text and layout should be merged with weights."""
        text_result = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.5, method="text_rules",
            type_scores={"invoice": 0.5, "receipt": 0.2},
        )
        layout_result = PageClassification(
            page_num=1, predicted_type="form",
            confidence=0.4, method="layout_features",
            type_scores={"form": 0.4, "receipt": 0.3},
        )
        result = classify_page_ensemble(text_result, layout_result, 1)
        assert "invoice" in result.type_scores
        assert "form" in result.type_scores
        assert "receipt" in result.type_scores
        # receipt merged: 0.2*0.6 + 0.3*0.4 = 0.12 + 0.12 = 0.24
        assert abs(result.type_scores["receipt"] - 0.24) < 0.01

    def test_different_predictions(self):
        """Text=invoice, layout=form -- weighted winner selected."""
        text_result = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.5, method="text_rules",
            type_scores={"invoice": 0.5},
        )
        layout_result = PageClassification(
            page_num=1, predicted_type="form",
            confidence=0.5, method="layout_features",
            type_scores={"form": 0.5},
        )
        result = classify_page_ensemble(text_result, layout_result, 1)
        # invoice: 0.5*0.6=0.30, form: 0.5*0.4=0.20 -- invoice wins
        assert result.predicted_type == "invoice"

    def test_both_low_confidence(self):
        """Both tiers have low confidence -- result has low confidence too."""
        text_result = PageClassification(
            page_num=1, predicted_type="invoice",
            confidence=0.1, method="text_rules",
            type_scores={"invoice": 0.1},
        )
        layout_result = PageClassification(
            page_num=1, predicted_type="form",
            confidence=0.1, method="layout_features",
            type_scores={"form": 0.1},
        )
        result = classify_page_ensemble(text_result, layout_result, 1)
        assert result.confidence < 0.2


# ---------------------------------------------------------------------------
# Tests: Finalization (document-level aggregation)
# ---------------------------------------------------------------------------


class TestFinalization:
    def test_finalize_empty_document(self):
        doc = DocumentClassification(document_id="d1", source_file="empty.pdf")
        result = finalize_classification(doc)
        assert result.document_type == "other"
        assert result.document_confidence == 0.0
        assert result.type_distribution == {}
        assert result.document_labels == []
        assert result.custom_profile_matches == []

    def test_finalize_unanimous(self):
        """All pages classified as the same type -- high confidence."""
        doc = DocumentClassification(document_id="d2", source_file="test.pdf")
        doc.pages = [
            PageClassification(page_num=i, predicted_type="invoice", confidence=0.8)
            for i in range(1, 4)
        ]
        result = finalize_classification(doc)
        assert result.document_type == "invoice"
        assert result.document_confidence == 0.8
        assert result.type_distribution == {"invoice": 3}

    def test_finalize_majority_vote(self):
        """3 invoice, 1 letter, 1 form -- invoice wins by majority."""
        doc = DocumentClassification(document_id="d3", source_file="mixed.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="invoice", confidence=0.7),
            PageClassification(page_num=2, predicted_type="invoice", confidence=0.6),
            PageClassification(page_num=3, predicted_type="invoice", confidence=0.8),
            PageClassification(page_num=4, predicted_type="letter", confidence=0.9),
            PageClassification(page_num=5, predicted_type="form", confidence=0.5),
        ]
        result = finalize_classification(doc)
        assert result.document_type == "invoice"

    def test_finalize_type_distribution(self):
        """Correct page counts per type in distribution."""
        doc = DocumentClassification(document_id="d4", source_file="dist.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="invoice", confidence=0.7),
            PageClassification(page_num=2, predicted_type="letter", confidence=0.5),
            PageClassification(page_num=3, predicted_type="invoice", confidence=0.6),
        ]
        result = finalize_classification(doc)
        assert result.type_distribution["invoice"] == 2
        assert result.type_distribution["letter"] == 1

    def test_finalize_document_confidence(self):
        """Document confidence is average of winning type's page confidences."""
        doc = DocumentClassification(document_id="d5", source_file="conf.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="invoice", confidence=0.6),
            PageClassification(page_num=2, predicted_type="invoice", confidence=0.8),
            PageClassification(page_num=3, predicted_type="letter", confidence=0.9),
        ]
        result = finalize_classification(doc)
        assert result.document_type == "invoice"
        # Average confidence of invoice pages: (0.6 + 0.8) / 2 = 0.7
        assert abs(result.document_confidence - 0.7) < 0.01

    def test_finalize_tie_breaking(self):
        """Equal vote counts -- type with higher avg confidence wins."""
        doc = DocumentClassification(document_id="d6", source_file="tie.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="invoice", confidence=0.5),
            PageClassification(page_num=2, predicted_type="letter", confidence=0.9),
        ]
        result = finalize_classification(doc)
        # Both have 1 page each; letter has higher avg confidence
        assert result.document_type == "letter"
        assert abs(result.document_confidence - 0.9) < 0.01

    def test_finalize_single_page(self):
        """Single-page document -- document type matches that page."""
        doc = DocumentClassification(document_id="d7", source_file="single.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="report", confidence=0.65),
        ]
        result = finalize_classification(doc)
        assert result.document_type == "report"
        assert result.document_confidence == 0.65

    def test_finalize_all_different(self):
        """Each page has a different type -- picks one (all tied at 1 vote)."""
        doc = DocumentClassification(document_id="d8", source_file="diverse.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="invoice", confidence=0.3),
            PageClassification(page_num=2, predicted_type="letter", confidence=0.4),
            PageClassification(page_num=3, predicted_type="form", confidence=0.9),
        ]
        result = finalize_classification(doc)
        # All 1 vote each; highest avg confidence wins => form (0.9)
        assert result.document_type == "form"
        assert abs(result.document_confidence - 0.9) < 0.01

    def test_finalize_builds_multi_label_summary(self):
        doc = DocumentClassification(document_id="d9", source_file="multilabel.pdf")
        doc.pages = [
            PageClassification(
                page_num=1,
                predicted_type="invoice",
                confidence=0.9,
                type_scores={"invoice": 0.9, "receipt": 0.4},
            ),
            PageClassification(
                page_num=2,
                predicted_type="receipt",
                confidence=0.8,
                type_scores={"invoice": 0.35, "receipt": 0.8},
            ),
        ]
        result = finalize_classification(doc)
        assert result.document_type == "invoice"
        assert result.document_type_scores["invoice"] >= result.document_type_scores["receipt"]
        assert result.document_labels[0]["label"] == "invoice"
        assert any(label["label"] == "receipt" for label in result.document_labels)

    def test_finalize_summarizes_custom_profiles(self):
        doc = DocumentClassification(document_id="d10", source_file="profile.pdf")
        doc.pages = [
            PageClassification(
                page_num=1,
                predicted_type="invoice",
                confidence=0.8,
                type_scores={"invoice": 0.8},
                profile_matches=[
                    {
                        "name": "invoice_packet",
                        "base_type": "invoice",
                        "route": "finance_invoice_review",
                        "confidence": 0.9,
                    }
                ],
            )
        ]
        result = finalize_classification(doc)
        assert result.custom_profile_matches[0]["name"] == "invoice_packet"
        assert result.custom_profile_matches[0]["route"] == "finance_invoice_review"


# ---------------------------------------------------------------------------
# Tests: JSON output
# ---------------------------------------------------------------------------


class TestWriteJson:
    def test_json_file_created(self, tmp_path):
        """Write to tmp dir -- file should exist."""
        doc = DocumentClassification(
            document_id="test_id", source_file="subfolder/doc.pdf"
        )
        doc.pages = [
            PageClassification(page_num=1, predicted_type="invoice", confidence=0.7,
                               method="text_rules", type_scores={"invoice": 0.7}),
        ]
        doc = finalize_classification(doc)
        result = write_classification_json(doc, str(tmp_path), "subfolder", "0.5.0")
        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".classification.json")

    def test_non_pdf_uses_ext_token(self, tmp_path):
        doc = DocumentClassification(
            document_id="img_doc",
            source_file="images/photo.png",
        )
        doc.pages = []
        doc = finalize_classification(doc)
        result = write_classification_json(doc, str(tmp_path), "", __version__)
        assert result is not None
        assert os.path.basename(result) == "photo__png.classification.json"

    def test_json_schema_structure(self, tmp_path):
        """Parsed JSON should contain all required top-level keys."""
        doc = DocumentClassification(
            document_id="schema_test", source_file="test.pdf"
        )
        doc.pages = [
            PageClassification(page_num=1, predicted_type="contract", confidence=0.6,
                               method="text_rules", type_scores={"contract": 0.6}),
        ]
        doc = finalize_classification(doc)
        result_path = write_classification_json(doc, str(tmp_path), "", "0.5.0")
        assert result_path is not None

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "schema_test"
        assert data["source_file"] == "test.pdf"
        assert "processing" in data
        assert data["processing"]["pipeline_version"] == "0.5.0"
        assert data["processing"]["classification_engine"] == "text_rules"
        assert "document_summary" in data
        assert data["document_summary"]["document_type"] == "contract"
        assert "pages" in data
        assert len(data["pages"]) == 1
        assert data["pages"][0]["predicted_type"] == "contract"

    def test_json_includes_multilabel_summary(self, tmp_path):
        doc = DocumentClassification(document_id="summary_test", source_file="test.pdf")
        doc.pages = [
            PageClassification(
                page_num=1,
                predicted_type="invoice",
                confidence=0.8,
                method="text_rules",
                type_scores={"invoice": 0.8, "receipt": 0.35},
                profile_matches=[
                    {
                        "name": "invoice_packet",
                        "base_type": "invoice",
                        "route": "finance_invoice_review",
                        "confidence": 0.9,
                    }
                ],
            )
        ]
        doc = finalize_classification(doc)
        result_path = write_classification_json(doc, str(tmp_path), "", "0.5.0")
        assert result_path is not None

        with open(result_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        summary = data["document_summary"]
        assert summary["document_labels"][0]["label"] == "invoice"
        assert summary["custom_profile_matches"][0]["name"] == "invoice_packet"

    def test_json_subfolder_creation(self, tmp_path):
        """Non-empty subfolder should create nested directories."""
        doc = DocumentClassification(
            document_id="dir_test", source_file="deep/nested/doc.pdf"
        )
        doc.pages = []
        doc = finalize_classification(doc)
        result = write_classification_json(doc, str(tmp_path), "deep/nested", "0.5.0")
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(
            str(tmp_path), "EXPORT", "CLASSIFICATION", "deep", "nested"
        )
        assert os.path.isdir(expected_dir)

    def test_path_traversal_protection(self, tmp_path):
        """Subfolder with '..' should be neutralized by sanitizer.

        The _sanitize_path_segment function strips dots from path segments,
        so '..' becomes empty and is filtered out. The result is a safe path
        inside the CLASSIFICATION directory (not None, since the traversal
        is prevented by sanitization rather than outright rejection).
        """
        doc = DocumentClassification(
            document_id="traversal", source_file="evil.pdf"
        )
        doc.pages = []
        doc = finalize_classification(doc)
        result = write_classification_json(
            doc, str(tmp_path), "../../etc", "0.5.0"
        )
        # Sanitizer strips '..' segments; result is safe inside classification dir
        assert result is not None
        assert os.path.exists(result)
        classification_root = os.path.realpath(
            os.path.join(str(tmp_path), "EXPORT", "CLASSIFICATION")
        )
        assert os.path.realpath(result).startswith(classification_root)

    def test_empty_subfolder(self):
        """Empty subfolder should write to the CLASSIFICATION root."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            doc = DocumentClassification(
                document_id="root_test", source_file="root_doc.pdf"
            )
            doc.pages = []
            doc = finalize_classification(doc)
            result = write_classification_json(doc, tmp_dir, "", "0.5.0")
            assert result is not None
            assert os.path.exists(result)
            expected_dir = os.path.join(tmp_dir, "EXPORT", "CLASSIFICATION")
            assert os.path.dirname(result) == expected_dir


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_text_handling(self):
        """None or empty text should produce safe 'other' classification."""
        result_none = classify_page_by_text(None, 1)
        assert result_none.predicted_type == "other"
        assert result_none.confidence == 0.0

        result_empty = classify_page_by_text("", 1)
        assert result_empty.predicted_type == "other"
        assert result_empty.confidence == 0.0

    def test_very_long_text(self):
        """Long text (10000 chars) should complete without error."""
        text = "invoice total amount due " * 400  # ~10000 chars
        result = classify_page_by_text(text, 1)
        assert result.predicted_type == "invoice"
        assert result.confidence > 0.0

    def test_special_characters_in_text(self):
        """Unicode and special characters should not crash classification."""
        text = "Invoice \u00e9\u00e8\u00ea \u2603 \u00a9 Amount Due: \u20ac100 Total: \u00a5500"
        result = classify_page_by_text(text, 1)
        # Should still detect invoice keywords
        assert result.predicted_type == "invoice"
        assert result.confidence > 0.0

    def test_is_handwritten_flag(self):
        """PageClassification.is_handwritten can be set and read."""
        pc = PageClassification(page_num=1, is_handwritten=True)
        assert pc.is_handwritten is True

        pc2 = PageClassification(page_num=2, is_handwritten=False)
        assert pc2.is_handwritten is False

    def test_negative_page_num(self):
        """Negative page_num should still work without error."""
        result = classify_page_by_text("Dear Sir,", -1)
        assert result.page_num == -1
        assert result.predicted_type == "letter"

    def test_write_json_with_special_filename(self, tmp_path):
        """Filename with spaces should be handled correctly."""
        doc = DocumentClassification(
            document_id="special_name", source_file="my document file.pdf"
        )
        doc.pages = [
            PageClassification(page_num=1, predicted_type="other", confidence=0.0,
                               method="text_rules"),
        ]
        doc = finalize_classification(doc)
        result = write_classification_json(doc, str(tmp_path), "", "0.5.0")
        assert result is not None
        assert os.path.exists(result)
        assert "my document file" in os.path.basename(result)

    def test_concurrent_safe_construction(self):
        """Multiple PageClassification instances should be independent."""
        pc1 = PageClassification(page_num=1, predicted_type="invoice", confidence=0.8)
        pc2 = PageClassification(page_num=2, predicted_type="letter", confidence=0.5)
        pc3 = PageClassification(page_num=3)

        # Verify they are independent
        assert pc1.predicted_type == "invoice"
        assert pc2.predicted_type == "letter"
        assert pc3.predicted_type == "other"

        # Modify one, others unchanged
        pc1.type_scores["invoice"] = 0.8
        assert pc2.type_scores == {}
        assert pc3.type_scores == {}


# ---------------------------------------------------------------------------
# Tests: Finalization with dict pages (backward compatibility)
# ---------------------------------------------------------------------------


class TestFinalizationDictPages:
    def test_finalize_with_dict_pages(self):
        """finalize_classification should accept dict pages as well as dataclasses."""
        doc = DocumentClassification(document_id="dict_test", source_file="dict.pdf")
        doc.pages = [
            {"predicted_type": "invoice", "confidence": 0.7},
            {"predicted_type": "invoice", "confidence": 0.5},
            {"predicted_type": "letter", "confidence": 0.6},
        ]
        result = finalize_classification(doc)
        assert result.document_type == "invoice"
        assert result.type_distribution["invoice"] == 2
        assert result.type_distribution["letter"] == 1

    def test_finalize_mixed_pages(self):
        """finalize_classification should handle mix of dicts and dataclass pages."""
        doc = DocumentClassification(document_id="mix_test", source_file="mix.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="form", confidence=0.8),
            {"predicted_type": "form", "confidence": 0.6},
        ]
        result = finalize_classification(doc)
        assert result.document_type == "form"
        assert result.type_distribution["form"] == 2


# ---------------------------------------------------------------------------
# Tests: JSON engine label detection
# ---------------------------------------------------------------------------


class TestJsonEngineLabel:
    def test_text_rules_engine_label(self, tmp_path):
        """Pages with method=text_rules should produce 'text_rules' engine."""
        doc = DocumentClassification(document_id="t1", source_file="t.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="invoice",
                               confidence=0.5, method="text_rules"),
        ]
        doc = finalize_classification(doc)
        path = write_classification_json(doc, str(tmp_path), "", "0.5.0")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["processing"]["classification_engine"] == "text_rules"

    def test_ensemble_engine_label(self, tmp_path):
        """Pages with method=ensemble should produce combined engine label."""
        doc = DocumentClassification(document_id="t2", source_file="t.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="invoice",
                               confidence=0.5, method="ensemble"),
        ]
        doc = finalize_classification(doc)
        path = write_classification_json(doc, str(tmp_path), "", "0.5.0")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["processing"]["classification_engine"] == "text_rules+layout_features"

    def test_layout_features_engine_label(self, tmp_path):
        """Pages with method=layout_features should produce that engine label."""
        doc = DocumentClassification(document_id="t3", source_file="t.pdf")
        doc.pages = [
            PageClassification(page_num=1, predicted_type="form",
                               confidence=0.5, method="layout_features"),
        ]
        doc = finalize_classification(doc)
        path = write_classification_json(doc, str(tmp_path), "", "0.5.0")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["processing"]["classification_engine"] == "layout_features"
