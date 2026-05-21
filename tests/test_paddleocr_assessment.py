"""Tests for PaddleOCR 3.x migration assessment document.

Validates that the research document exists and contains all
required sections for a complete migration assessment.
"""

import os


class TestPaddleOCRAssessmentDoc:
    def setup_method(self):
        self.doc_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "docs",
            "research",
            "paddleocr-3x-assessment.md",
        )

    def test_doc_exists(self):
        assert os.path.isfile(self.doc_path)

    def test_has_executive_summary(self):
        content = open(self.doc_path).read()
        assert "## Executive Summary" in content

    def test_has_risk_matrix(self):
        content = open(self.doc_path).read()
        assert "## Migration Risk Matrix" in content

    def test_has_decision(self):
        content = open(self.doc_path).read()
        assert "## Decision" in content

    def test_mentions_ctc_safety(self):
        content = open(self.doc_path).read()
        assert "CTC" in content

    def test_mentions_current_version(self):
        content = open(self.doc_path).read()
        assert "2.9.1" in content

    def test_has_compatibility_checklist(self):
        content = open(self.doc_path).read()
        assert "## Compatibility Checklist" in content

    def test_has_migration_path(self):
        content = open(self.doc_path).read()
        assert "## Recommended Migration Path" in content

    def test_recommendation_is_defer(self):
        content = open(self.doc_path).read()
        assert "DEFER" in content
