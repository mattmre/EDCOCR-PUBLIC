"""Unit tests for multi-label document classification module.

Tests cover:
- CustomTaxonomy dataclass defaults
- Taxonomy loading from JSON config files
- Config file validation and error handling
- Multi-label classification with keyword matching
- Exclusive (single-label) mode
- Confidence threshold filtering
- Empty/missing text handling
- Integration merge with existing classification output
- Environment-based classifier creation
- Edge cases: duplicate names, large configs, special characters

Run with: python -m pytest tests/test_multi_label_classification.py -v
"""

import json
import os

# Add project root to path
from multi_label_classification import (
    CustomTaxonomy,
    MultiLabelClassifier,
    TaxonomyMatch,
    create_classifier_from_env,
    load_taxonomies_from_file,
    merge_multi_label_results,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_taxonomy_config(taxonomies: list) -> dict:
    """Build a taxonomy config payload dict."""
    return {"taxonomies": taxonomies}


def _write_config(tmp_path, config: dict) -> str:
    """Write a config dict to a JSON file and return the path."""
    config_path = os.path.join(str(tmp_path), "taxonomies.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
    return config_path


def _sample_department_taxonomy() -> dict:
    """Return a sample department taxonomy config dict."""
    return {
        "name": "department",
        "labels": ["HR", "Finance", "Legal", "Engineering", "Sales"],
        "rules": {
            "HR": ["employee", "benefits", "onboarding", "performance review"],
            "Finance": ["invoice", "payment", "budget", "expense"],
            "Legal": ["contract", "agreement", "clause", "liability"],
            "Engineering": ["code", "deploy", "architecture", "sprint"],
            "Sales": ["client", "revenue", "pipeline", "quota"],
        },
        "exclusive": False,
        "confidence_threshold": 0.4,
    }


def _sample_priority_taxonomy() -> dict:
    """Return a sample priority taxonomy config dict."""
    return {
        "name": "priority",
        "labels": ["urgent", "normal", "low"],
        "rules": {
            "urgent": ["immediate", "asap", "critical", "emergency"],
            "normal": ["standard", "routine", "regular"],
            "low": ["whenever", "optional", "no rush"],
        },
        "exclusive": True,
        "confidence_threshold": 0.3,
    }


# ---------------------------------------------------------------------------
# Tests: Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_custom_taxonomy_defaults(self):
        t = CustomTaxonomy(name="test")
        assert t.name == "test"
        assert t.labels == []
        assert t.rules == {}
        assert t.exclusive is False
        assert t.confidence_threshold == 0.5

    def test_taxonomy_match_defaults(self):
        m = TaxonomyMatch(label="foo")
        assert m.label == "foo"
        assert m.confidence == 0.0
        assert m.matched_keywords == 0
        assert m.total_keywords == 0

    def test_taxonomy_match_to_dict(self):
        m = TaxonomyMatch(
            label="HR", confidence=0.75, matched_keywords=3, total_keywords=4
        )
        d = m.to_dict()
        assert d["label"] == "HR"
        assert d["confidence"] == 0.75
        assert d["matched_keywords"] == 3
        assert d["total_keywords"] == 4


# ---------------------------------------------------------------------------
# Tests: Taxonomy loading from JSON
# ---------------------------------------------------------------------------


class TestTaxonomyLoading:
    def test_load_valid_config(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 1
        assert taxonomies[0].name == "department"
        assert len(taxonomies[0].labels) == 5
        assert "HR" in taxonomies[0].rules
        assert taxonomies[0].exclusive is False
        assert taxonomies[0].confidence_threshold == 0.4

    def test_load_multiple_taxonomies(self, tmp_path):
        config = _make_taxonomy_config([
            _sample_department_taxonomy(),
            _sample_priority_taxonomy(),
        ])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 2
        names = [t.name for t in taxonomies]
        assert "department" in names
        assert "priority" in names

    def test_load_empty_path(self):
        taxonomies = load_taxonomies_from_file("")
        assert taxonomies == []

    def test_load_missing_file(self):
        taxonomies = load_taxonomies_from_file("/nonexistent/path.json")
        assert taxonomies == []

    def test_load_invalid_json(self, tmp_path):
        path = os.path.join(str(tmp_path), "bad.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        taxonomies = load_taxonomies_from_file(path)
        assert taxonomies == []

    def test_load_non_object_payload(self, tmp_path):
        path = _write_config(tmp_path, ["not", "an", "object"])
        taxonomies = load_taxonomies_from_file(path)
        assert taxonomies == []

    def test_load_missing_taxonomies_key(self, tmp_path):
        path = _write_config(tmp_path, {"other_key": []})
        taxonomies = load_taxonomies_from_file(path)
        assert taxonomies == []

    def test_load_taxonomies_not_list(self, tmp_path):
        path = _write_config(tmp_path, {"taxonomies": "not-a-list"})
        taxonomies = load_taxonomies_from_file(path)
        assert taxonomies == []

    def test_skip_taxonomy_without_name(self, tmp_path):
        config = _make_taxonomy_config([
            {
                "labels": ["A"],
                "rules": {"A": ["keyword"]},
            }
        ])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 0

    def test_skip_taxonomy_without_rules(self, tmp_path):
        config = _make_taxonomy_config([
            {"name": "empty_rules", "labels": ["A"], "rules": {}},
        ])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 0

    def test_skip_duplicate_taxonomy_names(self, tmp_path):
        tax = _sample_department_taxonomy()
        config = _make_taxonomy_config([tax, tax])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 1

    def test_taxonomy_non_dict_entry(self, tmp_path):
        config = _make_taxonomy_config(["not-a-dict"])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 0

    def test_taxonomy_labels_not_list(self, tmp_path):
        config = _make_taxonomy_config([{
            "name": "bad_labels",
            "labels": "not-a-list",
            "rules": {"A": ["keyword"]},
        }])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        # Labels validation logs warning but rules still work;
        # label "A" gets auto-added from rules
        assert len(taxonomies) == 1
        assert "A" in taxonomies[0].labels

    def test_taxonomy_rules_not_dict(self, tmp_path):
        config = _make_taxonomy_config([{
            "name": "bad_rules",
            "labels": ["A"],
            "rules": ["not-a-dict"],
        }])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 0

    def test_rule_label_not_in_labels_list(self, tmp_path):
        config = _make_taxonomy_config([{
            "name": "mismatch",
            "labels": ["A"],
            "rules": {
                "A": ["keyword_a"],
                "B": ["keyword_b"],
            },
        }])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 1
        # Only label "A" should have rules; "B" is skipped
        assert "A" in taxonomies[0].rules
        assert "B" not in taxonomies[0].rules

    def test_invalid_confidence_threshold(self, tmp_path):
        tax = _sample_department_taxonomy()
        tax["confidence_threshold"] = "not-a-number"
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 1
        assert taxonomies[0].confidence_threshold == 0.5

    def test_confidence_threshold_clamped(self, tmp_path):
        tax = _sample_department_taxonomy()
        tax["confidence_threshold"] = 5.0
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert taxonomies[0].confidence_threshold == 1.0

        tax["confidence_threshold"] = -1.0
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert taxonomies[0].confidence_threshold == 0.0

    def test_exclusive_mode_true(self, tmp_path):
        tax = _sample_priority_taxonomy()
        assert tax["exclusive"] is True
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert taxonomies[0].exclusive is True

    def test_empty_keyword_list_skipped(self, tmp_path):
        config = _make_taxonomy_config([{
            "name": "empty_kw",
            "labels": ["A"],
            "rules": {"A": []},
        }])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        # No valid rules after validation
        assert len(taxonomies) == 0

    def test_labels_auto_added_from_rules(self, tmp_path):
        """Labels not in the labels list but present in rules are auto-added."""
        config = _make_taxonomy_config([{
            "name": "auto_add",
            "labels": [],
            "rules": {"X": ["keyword_x"], "Y": ["keyword_y"]},
        }])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 1
        assert "X" in taxonomies[0].labels
        assert "Y" in taxonomies[0].labels


# ---------------------------------------------------------------------------
# Tests: Multi-label classification
# ---------------------------------------------------------------------------


class TestMultiLabelClassification:
    def test_classify_matches_keywords(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify(
            "The employee benefits package includes onboarding materials "
            "and performance review documents"
        )
        dept = results["taxonomies"]["department"]
        labels = [m["label"] for m in dept]
        assert "HR" in labels

    def test_classify_multiple_labels(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify(
            "The employee benefits need to be invoiced. "
            "Please process the payment and expense report."
        )
        dept = results["taxonomies"]["department"]
        labels = [m["label"] for m in dept]
        # Both HR and Finance keywords present
        assert "HR" in labels
        assert "Finance" in labels

    def test_classify_empty_text(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify("")
        assert results["taxonomies"]["department"] == []
        assert results["total_labels_matched"] == 0

    def test_classify_none_text(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify(None)
        assert results["taxonomies"]["department"] == []

    def test_classify_whitespace_only(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify("   \n\t  ")
        assert results["taxonomies"]["department"] == []

    def test_classify_no_match(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify("The weather is nice today.")
        dept = results["taxonomies"]["department"]
        assert dept == []

    def test_classify_case_insensitive(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify(
            "EMPLOYEE BENEFITS ONBOARDING PERFORMANCE REVIEW"
        )
        dept = results["taxonomies"]["department"]
        labels = [m["label"] for m in dept]
        assert "HR" in labels

    def test_classify_confidence_score(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        # Match all 4 HR keywords
        results = classifier.classify(
            "employee benefits onboarding performance review"
        )
        dept = results["taxonomies"]["department"]
        hr_match = next(m for m in dept if m["label"] == "HR")
        assert hr_match["confidence"] == 1.0
        assert hr_match["matched_keywords"] == 4
        assert hr_match["total_keywords"] == 4

    def test_classify_partial_match_confidence(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        # Match 2 of 4 HR keywords
        results = classifier.classify("employee benefits")
        dept = results["taxonomies"]["department"]
        hr_match = next(m for m in dept if m["label"] == "HR")
        assert hr_match["confidence"] == 0.5
        assert hr_match["matched_keywords"] == 2
        assert hr_match["total_keywords"] == 4


# ---------------------------------------------------------------------------
# Tests: Exclusive mode
# ---------------------------------------------------------------------------


class TestExclusiveMode:
    def test_exclusive_returns_single_label(self, tmp_path):
        config = _make_taxonomy_config([_sample_priority_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify(
            "This is an immediate critical emergency. "
            "Please handle this standard routine task."
        )
        priority = results["taxonomies"]["priority"]
        # Exclusive mode: only one label returned
        assert len(priority) == 1
        # Urgent has more keyword hits
        assert priority[0]["label"] == "urgent"

    def test_exclusive_picks_highest_confidence(self, tmp_path):
        tax = {
            "name": "severity",
            "labels": ["high", "low"],
            "rules": {
                "high": ["critical", "severe"],
                "low": ["minor"],
            },
            "exclusive": True,
            "confidence_threshold": 0.3,
        }
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify("This is a critical and severe issue")
        severity = results["taxonomies"]["severity"]
        assert len(severity) == 1
        assert severity[0]["label"] == "high"
        assert severity[0]["confidence"] == 1.0


# ---------------------------------------------------------------------------
# Tests: Confidence threshold
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    def test_below_threshold_filtered(self, tmp_path):
        tax = {
            "name": "strict",
            "labels": ["A", "B"],
            "rules": {
                "A": ["keyword1", "keyword2", "keyword3", "keyword4"],
                "B": ["word1"],
            },
            "exclusive": False,
            "confidence_threshold": 0.5,
        }
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        # Match 1 of 4 A keywords (0.25) -> below 0.5 threshold
        # Match 1 of 1 B keywords (1.0) -> above 0.5 threshold
        results = classifier.classify("keyword1 word1")
        matches = results["taxonomies"]["strict"]
        labels = [m["label"] for m in matches]
        assert "B" in labels
        assert "A" not in labels

    def test_zero_threshold_returns_all(self, tmp_path):
        tax = {
            "name": "permissive",
            "labels": ["A", "B"],
            "rules": {
                "A": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"],
                "B": ["y1"],
            },
            "exclusive": False,
            "confidence_threshold": 0.0,
        }
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        # Even 1 of 10 matches (0.1) is above 0.0 threshold
        results = classifier.classify("x1 y1")
        matches = results["taxonomies"]["permissive"]
        labels = [m["label"] for m in matches]
        assert "A" in labels
        assert "B" in labels


# ---------------------------------------------------------------------------
# Tests: Multiple taxonomies
# ---------------------------------------------------------------------------


class TestMultipleTaxonomies:
    def test_classify_with_multiple_taxonomies(self, tmp_path):
        config = _make_taxonomy_config([
            _sample_department_taxonomy(),
            _sample_priority_taxonomy(),
        ])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify(
            "Employee benefits need immediate critical attention"
        )
        assert "department" in results["taxonomies"]
        assert "priority" in results["taxonomies"]
        dept = results["taxonomies"]["department"]
        priority = results["taxonomies"]["priority"]
        assert any(m["label"] == "HR" for m in dept)
        assert any(m["label"] == "urgent" for m in priority)

    def test_taxonomy_count_in_results(self, tmp_path):
        config = _make_taxonomy_config([
            _sample_department_taxonomy(),
            _sample_priority_taxonomy(),
        ])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify("employee immediate")
        assert results["taxonomy_count"] == 2

    def test_total_labels_matched(self, tmp_path):
        config = _make_taxonomy_config([
            _sample_department_taxonomy(),
            _sample_priority_taxonomy(),
        ])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify(
            "employee benefits onboarding performance review "
            "immediate asap critical emergency"
        )
        assert results["total_labels_matched"] >= 2


# ---------------------------------------------------------------------------
# Tests: Classifier construction
# ---------------------------------------------------------------------------


class TestClassifierConstruction:
    def test_init_with_taxonomies(self):
        import re as _re

        tax = CustomTaxonomy(
            name="test",
            labels=["A"],
            rules={"A": [_re.compile(r"keyword", _re.IGNORECASE)]},
        )
        classifier = MultiLabelClassifier(taxonomies=[tax])
        assert classifier.taxonomy_count == 1
        assert classifier.taxonomy_names == ["test"]

    def test_init_with_config_path(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        assert classifier.taxonomy_count == 1

    def test_init_no_args(self):
        classifier = MultiLabelClassifier()
        assert classifier.taxonomy_count == 0
        results = classifier.classify("some text")
        assert results["taxonomies"] == {}
        assert results["total_labels_matched"] == 0

    def test_classify_single_taxonomy(self, tmp_path):
        config = _make_taxonomy_config([
            _sample_department_taxonomy(),
            _sample_priority_taxonomy(),
        ])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify_single_taxonomy(
            "employee benefits", "department"
        )
        assert isinstance(results, list)
        assert any(m["label"] == "HR" for m in results)

    def test_classify_single_taxonomy_not_found(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify_single_taxonomy(
            "employee benefits", "nonexistent"
        )
        assert results == []


# ---------------------------------------------------------------------------
# Tests: Integration merge
# ---------------------------------------------------------------------------


class TestMergeResults:
    def test_merge_adds_multi_label_key(self):
        report = {
            "schema_version": "1.0",
            "document_summary": {"document_type": "invoice"},
        }
        ml_results = {
            "taxonomies": {"department": [{"label": "Finance", "confidence": 0.8}]},
            "taxonomy_count": 1,
            "total_labels_matched": 1,
        }
        merged = merge_multi_label_results(report, ml_results)
        assert "multi_label_results" in merged
        assert merged["multi_label_results"]["taxonomy_count"] == 1

    def test_merge_preserves_existing_fields(self):
        report = {
            "schema_version": "1.0",
            "document_id": "doc1",
            "document_summary": {"document_type": "invoice"},
        }
        ml_results = {"taxonomies": {}, "taxonomy_count": 0}
        merged = merge_multi_label_results(report, ml_results)
        assert merged["schema_version"] == "1.0"
        assert merged["document_id"] == "doc1"
        assert merged["document_summary"]["document_type"] == "invoice"

    def test_merge_non_dict_report_returns_unchanged(self):
        result = merge_multi_label_results("not-a-dict", {})
        assert result == "not-a-dict"

    def test_merge_non_dict_results_returns_unchanged(self):
        report = {"key": "value"}
        result = merge_multi_label_results(report, "not-a-dict")
        assert result == {"key": "value"}
        assert "multi_label_results" not in result


# ---------------------------------------------------------------------------
# Tests: Environment-based creation
# ---------------------------------------------------------------------------


class TestCreateFromEnv:
    def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_LABEL_CLASSIFICATION", "false")
        # Re-import to pick up new env value
        import multi_label_classification as mlc

        original = mlc.ENABLE_MULTI_LABEL_CLASSIFICATION
        mlc.ENABLE_MULTI_LABEL_CLASSIFICATION = False
        try:
            result = create_classifier_from_env()
            assert result is None
        finally:
            mlc.ENABLE_MULTI_LABEL_CLASSIFICATION = original

    def test_enabled_no_path_returns_none(self, monkeypatch):
        import multi_label_classification as mlc

        orig_enabled = mlc.ENABLE_MULTI_LABEL_CLASSIFICATION
        orig_path = mlc.CLASSIFICATION_TAXONOMY_PATH
        mlc.ENABLE_MULTI_LABEL_CLASSIFICATION = True
        mlc.CLASSIFICATION_TAXONOMY_PATH = ""
        try:
            result = create_classifier_from_env()
            assert result is None
        finally:
            mlc.ENABLE_MULTI_LABEL_CLASSIFICATION = orig_enabled
            mlc.CLASSIFICATION_TAXONOMY_PATH = orig_path

    def test_enabled_with_valid_config(self, tmp_path, monkeypatch):
        import multi_label_classification as mlc

        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)

        orig_enabled = mlc.ENABLE_MULTI_LABEL_CLASSIFICATION
        orig_path = mlc.CLASSIFICATION_TAXONOMY_PATH
        mlc.ENABLE_MULTI_LABEL_CLASSIFICATION = True
        mlc.CLASSIFICATION_TAXONOMY_PATH = path
        try:
            result = create_classifier_from_env()
            assert result is not None
            assert result.taxonomy_count == 1
        finally:
            mlc.ENABLE_MULTI_LABEL_CLASSIFICATION = orig_enabled
            mlc.CLASSIFICATION_TAXONOMY_PATH = orig_path

    def test_enabled_with_empty_config(self, tmp_path, monkeypatch):
        import multi_label_classification as mlc

        config = _make_taxonomy_config([])
        path = _write_config(tmp_path, config)

        orig_enabled = mlc.ENABLE_MULTI_LABEL_CLASSIFICATION
        orig_path = mlc.CLASSIFICATION_TAXONOMY_PATH
        mlc.ENABLE_MULTI_LABEL_CLASSIFICATION = True
        mlc.CLASSIFICATION_TAXONOMY_PATH = path
        try:
            result = create_classifier_from_env()
            assert result is None
        finally:
            mlc.ENABLE_MULTI_LABEL_CLASSIFICATION = orig_enabled
            mlc.CLASSIFICATION_TAXONOMY_PATH = orig_path


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unicode_keywords(self, tmp_path):
        tax = {
            "name": "language",
            "labels": ["french", "german"],
            "rules": {
                "french": ["facture", "montant"],
                "german": ["rechnung", "betrag"],
            },
            "exclusive": False,
            "confidence_threshold": 0.3,
        }
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify("Voici la facture et le montant total")
        matches = results["taxonomies"]["language"]
        assert any(m["label"] == "french" for m in matches)

    def test_special_regex_chars_in_keywords(self, tmp_path):
        """Keywords with regex-special chars are escaped and match literally."""
        tax = {
            "name": "symbols",
            "labels": ["money"],
            "rules": {
                "money": ["$100", "(total)", "item.price"],
            },
            "exclusive": False,
            "confidence_threshold": 0.3,
        }
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify("The $100 (total) is the item.price")
        matches = results["taxonomies"]["symbols"]
        assert any(m["label"] == "money" for m in matches)
        money = next(m for m in matches if m["label"] == "money")
        assert money["matched_keywords"] == 3

    def test_very_long_text(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        text = "employee benefits " * 5000  # ~90k chars
        results = classifier.classify(text)
        dept = results["taxonomies"]["department"]
        assert any(m["label"] == "HR" for m in dept)

    def test_sorted_by_confidence_descending(self, tmp_path):
        tax = {
            "name": "sorted",
            "labels": ["A", "B", "C"],
            "rules": {
                "A": ["a1"],
                "B": ["b1", "b2"],
                "C": ["c1", "c2", "c3"],
            },
            "exclusive": False,
            "confidence_threshold": 0.0,
        }
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify("a1 b1 b2 c1 c2 c3")
        matches = results["taxonomies"]["sorted"]
        confidences = [m["confidence"] for m in matches]
        assert confidences == sorted(confidences, reverse=True)

    def test_deterministic_tie_breaking(self, tmp_path):
        """Labels with equal confidence are sorted alphabetically."""
        tax = {
            "name": "tie",
            "labels": ["Banana", "Apple"],
            "rules": {
                "Banana": ["fruit"],
                "Apple": ["fruit"],
            },
            "exclusive": False,
            "confidence_threshold": 0.0,
        }
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify("fruit")
        matches = results["taxonomies"]["tie"]
        assert len(matches) == 2
        assert matches[0]["label"] == "Apple"
        assert matches[1]["label"] == "Banana"

    def test_multi_word_keyword_match(self, tmp_path):
        tax = {
            "name": "phrases",
            "labels": ["legal"],
            "rules": {
                "legal": ["terms and conditions", "binding agreement"],
            },
            "exclusive": False,
            "confidence_threshold": 0.3,
        }
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify(
            "Please read the terms and conditions of this binding agreement"
        )
        matches = results["taxonomies"]["phrases"]
        assert any(m["label"] == "legal" for m in matches)
        legal = next(m for m in matches if m["label"] == "legal")
        assert legal["matched_keywords"] == 2

    def test_taxonomy_names_property(self, tmp_path):
        config = _make_taxonomy_config([
            _sample_department_taxonomy(),
            _sample_priority_taxonomy(),
        ])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        assert set(classifier.taxonomy_names) == {"department", "priority"}

    def test_taxonomy_count_property(self, tmp_path):
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        assert classifier.taxonomy_count == 1

    def test_classify_with_layout_features_param(self, tmp_path):
        """layout_features param is accepted but currently unused."""
        config = _make_taxonomy_config([_sample_department_taxonomy()])
        path = _write_config(tmp_path, config)
        classifier = MultiLabelClassifier(config_path=path)
        results = classifier.classify(
            "employee benefits",
            layout_features={"tables": 2, "figures": 1},
        )
        dept = results["taxonomies"]["department"]
        assert any(m["label"] == "HR" for m in dept)

    def test_name_too_long_skipped(self, tmp_path):
        tax = {
            "name": "a" * 300,
            "labels": ["X"],
            "rules": {"X": ["keyword"]},
        }
        config = _make_taxonomy_config([tax])
        path = _write_config(tmp_path, config)
        taxonomies = load_taxonomies_from_file(path)
        assert len(taxonomies) == 0
