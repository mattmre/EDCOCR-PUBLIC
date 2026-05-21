"""Comprehensive tests for ML Model Training & Accuracy tools (Items 16-20).

Tests cover:
- Item 16: finetune_legal_corpus (label configs, report generation, CLI)
- Item 17: finetune_medical_corpus (labels, HIPAA mode, PHI recall, CLI)
- Item 18: ab_test_layout_engines (IoU, entity F1, Mann-Whitney, reporting)
- Item 19: benchmark_accuracy (CER, WER, edit distance, F1, dataset loading)
- Item 20: calibrate_confidence (ECE, reliability bins, calibration methods)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Ensure project root and scripts/ are on sys.path
_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TEST_DIR.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for p in [str(_PROJECT_ROOT), str(_SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ===================================================================
# Item 16: finetune_legal_corpus
# ===================================================================

class TestLegalCorpusLabels:
    """Test legal entity type definitions and label set building."""

    def test_legal_entity_types_defined(self):
        from finetune_legal_corpus import LEGAL_ENTITY_TYPES
        assert len(LEGAL_ENTITY_TYPES) == 8
        assert "CLAUSE_NUMBER" in LEGAL_ENTITY_TYPES
        assert "PARTY_NAME" in LEGAL_ENTITY_TYPES
        assert "EFFECTIVE_DATE" in LEGAL_ENTITY_TYPES
        assert "JURISDICTION" in LEGAL_ENTITY_TYPES
        assert "SIGNATURE_BLOCK" in LEGAL_ENTITY_TYPES
        assert "EXHIBIT_REF" in LEGAL_ENTITY_TYPES
        assert "BATES_NUMBER" in LEGAL_ENTITY_TYPES
        assert "PRIVILEGE_MARKER" in LEGAL_ENTITY_TYPES

    def test_legal_type_map_complete(self):
        from finetune_legal_corpus import LEGAL_ENTITY_TYPES, LEGAL_TYPE_MAP
        for etype in LEGAL_ENTITY_TYPES:
            assert etype in LEGAL_TYPE_MAP, f"Missing type_map for {etype}"

    def test_build_legal_label_set(self):
        from finetune_legal_corpus import build_legal_label_set
        ls = build_legal_label_set()
        assert ls.name == "legal"
        assert len(ls.entity_types) == 8
        # BIO expansion: O + 2 * 8 = 17 labels
        assert ls.num_labels == 17
        assert "O" in ls.label2id
        assert "B-CLAUSE_NUMBER" in ls.label2id
        assert "I-PRIVILEGE_MARKER" in ls.label2id

    def test_legal_label_set_id_roundtrip(self):
        from finetune_legal_corpus import build_legal_label_set
        ls = build_legal_label_set()
        for label, idx in ls.label2id.items():
            assert ls.id2label[idx] == label


class TestLegalConfusionMatrix:
    """Test confusion matrix and per-label F1 helpers."""

    def test_confusion_summary_perfect(self):
        from finetune_legal_corpus import compute_confusion_summary
        true = ["A", "B", "A", "B"]
        pred = ["A", "B", "A", "B"]
        cm = compute_confusion_summary(true, pred, ["A", "B"])
        assert cm["A"]["A"] == 2
        assert cm["A"]["B"] == 0
        assert cm["B"]["B"] == 2

    def test_confusion_summary_with_errors(self):
        from finetune_legal_corpus import compute_confusion_summary
        true = ["A", "A", "B", "B"]
        pred = ["A", "B", "A", "B"]
        cm = compute_confusion_summary(true, pred, ["A", "B"])
        assert cm["A"]["A"] == 1
        assert cm["A"]["B"] == 1
        assert cm["B"]["A"] == 1
        assert cm["B"]["B"] == 1

    def test_per_label_f1_perfect(self):
        from finetune_legal_corpus import compute_per_label_f1
        true = ["A", "B", "A", "B"]
        pred = ["A", "B", "A", "B"]
        metrics = compute_per_label_f1(true, pred, ["A", "B"])
        assert metrics["A"]["f1"] == 1.0
        assert metrics["B"]["f1"] == 1.0

    def test_per_label_f1_zero_support(self):
        from finetune_legal_corpus import compute_per_label_f1
        true = ["A", "A"]
        pred = ["A", "A"]
        metrics = compute_per_label_f1(true, pred, ["A", "B"])
        assert metrics["B"]["f1"] == 0.0
        assert metrics["B"]["support"] == 0


class TestLegalReport:
    """Test legal training report generation."""

    def test_generate_report_structure(self):
        from finetune_legal_corpus import (
            LegalFinetuneConfig,
            generate_training_report,
        )
        config = LegalFinetuneConfig()
        report = generate_training_report(
            config=config,
            train_result={"train_loss": 0.5, "eval_f1": 0.85},
        )
        assert report["pipeline"] == "legal_corpus_finetune"
        assert "timestamp" in report
        assert report["training_result"]["train_loss"] == 0.5

    def test_generate_report_writes_file(self):
        from finetune_legal_corpus import (
            LegalFinetuneConfig,
            generate_training_report,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report.json")
            generate_training_report(
                config=LegalFinetuneConfig(),
                train_result={"train_loss": 0.3},
                output_path=out_path,
            )
            assert Path(out_path).is_file()
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
            assert data["pipeline"] == "legal_corpus_finetune"


class TestLegalCLI:
    """Test legal corpus CLI argument parsing."""

    def test_default_args(self):
        from finetune_legal_corpus import parse_args
        config = parse_args([])
        assert config.data_dir == "./data/legal"
        assert config.output_dir == "./models/legal-v1"
        assert config.epochs == 30

    def test_custom_args(self):
        from finetune_legal_corpus import parse_args
        config = parse_args([
            "--data-dir", "/tmp/legal_data",
            "--epochs", "50",
            "--learning-rate", "1e-4",
            "--use-lora",
        ])
        assert config.data_dir == "/tmp/legal_data"
        assert config.epochs == 50
        assert config.learning_rate == 1e-4
        assert config.use_lora is True


# ===================================================================
# Item 17: finetune_medical_corpus
# ===================================================================

class TestMedicalCorpusLabels:
    """Test medical entity type definitions."""

    def test_medical_entity_types_defined(self):
        from finetune_medical_corpus import MEDICAL_ENTITY_TYPES
        assert len(MEDICAL_ENTITY_TYPES) == 10
        assert "PATIENT_NAME" in MEDICAL_ENTITY_TYPES
        assert "MRN" in MEDICAL_ENTITY_TYPES
        assert "DIAGNOSIS_CODE" in MEDICAL_ENTITY_TYPES
        assert "MEDICATION" in MEDICAL_ENTITY_TYPES
        assert "HIPAA_IDENTIFIER" in MEDICAL_ENTITY_TYPES

    def test_medical_type_map_complete(self):
        from finetune_medical_corpus import MEDICAL_ENTITY_TYPES, MEDICAL_TYPE_MAP
        for etype in MEDICAL_ENTITY_TYPES:
            assert etype in MEDICAL_TYPE_MAP

    def test_phi_entity_types_subset(self):
        from finetune_medical_corpus import (
            MEDICAL_ENTITY_TYPES,
            PHI_ENTITY_TYPES,
        )
        for phi in PHI_ENTITY_TYPES:
            assert phi in MEDICAL_ENTITY_TYPES

    def test_build_medical_label_set(self):
        from finetune_medical_corpus import build_medical_label_set
        ls = build_medical_label_set()
        assert ls.name == "medical"
        assert len(ls.entity_types) == 10
        # O + 2 * 10 = 21
        assert ls.num_labels == 21


class TestHIPAAHandling:
    """Test HIPAA-aware features."""

    def test_redact_phi_ssn(self):
        from finetune_medical_corpus import redact_phi
        text = "Patient SSN: 123-45-6789"
        result = redact_phi(text)
        assert "[REDACTED]" in result
        assert "123-45-6789" not in result

    def test_redact_phi_date(self):
        from finetune_medical_corpus import redact_phi
        text = "DOB: 1/15/1990"
        result = redact_phi(text)
        assert "[REDACTED]" in result

    def test_redact_phi_no_match(self):
        from finetune_medical_corpus import redact_phi
        text = "Normal text without PHI"
        result = redact_phi(text)
        assert result == text

    def test_hipaa_strict_report_redacts_path(self):
        from finetune_medical_corpus import (
            MedicalFinetuneConfig,
            generate_medical_report,
        )
        config = MedicalFinetuneConfig(hipaa_strict=True)
        report = generate_medical_report(
            config=config,
            train_result={"train_loss": 0.4},
        )
        assert report["config"]["data_dir"] == "[REDACTED]"
        assert report["hipaa_strict"] is True

    def test_non_hipaa_report_keeps_path(self):
        from finetune_medical_corpus import (
            MedicalFinetuneConfig,
            generate_medical_report,
        )
        config = MedicalFinetuneConfig(
            hipaa_strict=False,
            data_dir="/some/path",
        )
        report = generate_medical_report(
            config=config,
            train_result={},
        )
        assert report["config"]["data_dir"] == "/some/path"


class TestPHIDetectionRecall:
    """Test PHI detection recall computation."""

    def test_perfect_phi_recall(self):
        from finetune_medical_corpus import compute_phi_detection_recall
        true_labels = ["B-PATIENT_NAME", "I-PATIENT_NAME", "O", "B-MRN"]
        pred_labels = ["B-PATIENT_NAME", "I-PATIENT_NAME", "O", "B-MRN"]
        result = compute_phi_detection_recall(true_labels, pred_labels)
        assert result["overall_phi_recall"] == 1.0

    def test_zero_phi_recall(self):
        from finetune_medical_corpus import compute_phi_detection_recall
        true_labels = ["B-PATIENT_NAME", "B-MRN"]
        pred_labels = ["O", "O"]
        result = compute_phi_detection_recall(true_labels, pred_labels)
        assert result["overall_phi_recall"] == 0.0

    def test_partial_phi_recall(self):
        from finetune_medical_corpus import compute_phi_detection_recall
        true_labels = ["B-PATIENT_NAME", "B-MRN", "B-MEDICATION"]
        pred_labels = ["B-PATIENT_NAME", "O", "B-MEDICATION"]
        result = compute_phi_detection_recall(true_labels, pred_labels)
        # PATIENT_NAME detected, MRN missed; MEDICATION not a PHI type
        assert result["PATIENT_NAME_recall"] == 1.0
        assert result["MRN_recall"] == 0.0

    def test_empty_labels(self):
        from finetune_medical_corpus import compute_phi_detection_recall
        result = compute_phi_detection_recall([], [])
        assert result["overall_phi_recall"] == 0.0


class TestMedicalCLI:
    """Test medical corpus CLI parsing."""

    def test_default_args(self):
        from finetune_medical_corpus import parse_args
        config = parse_args([])
        assert config.data_dir == "./data/medical"
        assert config.epochs == 30

    def test_hipaa_strict_flag(self):
        from finetune_medical_corpus import parse_args
        config = parse_args(["--hipaa-strict"])
        assert config.hipaa_strict is True


class TestMedicalPerLabelAccuracy:
    """Test per-label accuracy computation."""

    def test_perfect_accuracy(self):
        from finetune_medical_corpus import compute_per_label_accuracy
        true = ["A", "B", "A"]
        pred = ["A", "B", "A"]
        result = compute_per_label_accuracy(true, pred, ["A", "B"])
        assert result["A"]["precision"] == 1.0
        assert result["A"]["recall"] == 1.0
        assert result["B"]["f1"] == 1.0

    def test_mixed_accuracy(self):
        from finetune_medical_corpus import compute_per_label_accuracy
        true = ["A", "A", "B"]
        pred = ["A", "B", "B"]
        result = compute_per_label_accuracy(true, pred, ["A", "B"])
        assert result["A"]["recall"] == 0.5  # 1 hit, 1 miss
        assert result["B"]["recall"] == 1.0  # 1 hit, 0 miss


# ===================================================================
# Item 18: ab_test_layout_engines
# ===================================================================

class TestIoUComputation:
    """Test bounding box IoU computation."""

    def test_perfect_overlap(self):
        from ab_test_layout_engines import BoundingBox, compute_iou
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(0, 0, 10, 10)
        assert compute_iou(a, b) == 1.0

    def test_no_overlap(self):
        from ab_test_layout_engines import BoundingBox, compute_iou
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        from ab_test_layout_engines import BoundingBox, compute_iou
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 5, 15, 15)
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        iou = compute_iou(a, b)
        assert abs(iou - 25.0 / 175.0) < 1e-6

    def test_zero_area_box(self):
        from ab_test_layout_engines import BoundingBox, compute_iou
        a = BoundingBox(0, 0, 0, 0)
        b = BoundingBox(0, 0, 10, 10)
        assert compute_iou(a, b) == 0.0

    def test_contained_box(self):
        from ab_test_layout_engines import BoundingBox, compute_iou
        a = BoundingBox(0, 0, 20, 20)
        b = BoundingBox(5, 5, 15, 15)
        # Intersection: 10x10 = 100
        # Union: 400 + 100 - 100 = 400
        iou = compute_iou(a, b)
        assert abs(iou - 100.0 / 400.0) < 1e-6


class TestMeanIoU:
    """Test mean IoU across box sets."""

    def test_mean_iou_identical_sets(self):
        from ab_test_layout_engines import BoundingBox, compute_mean_iou
        boxes_a = [BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 30, 30)]
        boxes_b = [BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 30, 30)]
        assert compute_mean_iou(boxes_a, boxes_b) == 1.0

    def test_mean_iou_empty_sets(self):
        from ab_test_layout_engines import BoundingBox, compute_mean_iou
        assert compute_mean_iou([], []) == 0.0
        assert compute_mean_iou([BoundingBox(0, 0, 10, 10)], []) == 0.0


class TestEntityF1:
    """Test entity comparison F1."""

    def test_perfect_match(self):
        from ab_test_layout_engines import compute_entity_f1
        p, r, f1 = compute_entity_f1(["A", "B"], ["A", "B"])
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_no_overlap(self):
        from ab_test_layout_engines import compute_entity_f1
        p, r, f1 = compute_entity_f1(["A"], ["B"])
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_empty_lists(self):
        from ab_test_layout_engines import compute_entity_f1
        p, r, f1 = compute_entity_f1([], [])
        assert f1 == 1.0

    def test_partial_match(self):
        from ab_test_layout_engines import compute_entity_f1
        p, r, f1 = compute_entity_f1(["A", "B", "C"], ["A", "B", "D"])
        # tp=2 (A, B), fp=1 (D), fn=1 (C)
        assert abs(p - 2 / 3) < 1e-6
        assert abs(r - 2 / 3) < 1e-6


class TestMannWhitneyU:
    """Test Mann-Whitney U test implementation."""

    def test_identical_samples(self):
        from ab_test_layout_engines import mann_whitney_u_test
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        u, p = mann_whitney_u_test(a, b)
        # Identical samples should not be significant
        assert p > 0.05

    def test_very_different_samples(self):
        from ab_test_layout_engines import mann_whitney_u_test
        a = [1.0] * 30
        b = [100.0] * 30
        u, p = mann_whitney_u_test(a, b)
        # Very different samples should be significant
        assert p < 0.05

    def test_empty_samples(self):
        from ab_test_layout_engines import mann_whitney_u_test
        u, p = mann_whitney_u_test([], [1.0])
        assert p == 1.0

    def test_single_element_samples(self):
        from ab_test_layout_engines import mann_whitney_u_test
        u, p = mann_whitney_u_test([1.0], [2.0])
        assert 0.0 <= p <= 1.0


class TestABTestCLI:
    """Test A/B test CLI parsing."""

    def test_parse_args(self):
        from ab_test_layout_engines import parse_args
        args = parse_args(["--test-dir", "/tmp/test"])
        assert args.test_dir == "/tmp/test"
        assert args.engines == "both"
        assert args.output_dir == "./ab_results"


class TestABTestReport:
    """Test A/B test report generation."""

    def test_generate_report_creates_files(self):
        from ab_test_layout_engines import ComparisonResult, generate_comparison_report
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ComparisonResult(
                engine_a="layoutlm",
                engine_b="ppstructure",
                num_documents=10,
                iou_scores=[0.8, 0.9],
                avg_iou=0.85,
                entity_f1=0.75,
                entity_precision=0.8,
                entity_recall=0.7,
                latency_a_ms=[10.0, 20.0],
                latency_b_ms=[15.0, 25.0],
                avg_latency_a_ms=15.0,
                avg_latency_b_ms=20.0,
                mann_whitney_u=5.0,
                mann_whitney_p=0.1,
                significant=False,
                timestamp="2026-01-01T00:00:00Z",
            )
            paths = generate_comparison_report(result, tmpdir)
            assert Path(paths["json"]).is_file()
            assert Path(paths["markdown"]).is_file()


class TestABTestEngineExecution:
    """Test the real engine wiring for Item 18."""

    @patch("ab_test_layout_engines._normalize_ppstructure_output")
    @patch("ab_test_layout_engines._run_ppstructure_page")
    @patch("ab_test_layout_engines._load_document_images")
    def test_run_engine_ppstructure_uses_normalized_outputs(
        self,
        mock_load_document_images,
        mock_run_ppstructure_page,
        mock_normalize_ppstructure_output,
        tmp_path,
    ):
        from ab_test_layout_engines import _run_engine

        doc = tmp_path / "sample.png"
        doc.write_bytes(b"fake-image")
        expected_regions = [{"type": "paragraph", "bbox": [0, 0, 100, 100]}]
        expected_entities = [{"type": "DATE", "text": "2026-03-17"}]

        mock_load_document_images.return_value = [object()]
        mock_run_ppstructure_page.return_value = [{"type": "text"}]
        mock_normalize_ppstructure_output.return_value = (
            expected_regions,
            expected_entities,
            [("dated 2026-03-17", 0.9, [0, 0, 100, 20])],
        )

        result = _run_engine("ppstructure", doc)

        assert result.error == ""
        assert result.regions == expected_regions
        assert result.entities == expected_entities

    @patch("ab_test_layout_engines._run_layoutlm_entities")
    @patch("ab_test_layout_engines._run_layoutlm_regions")
    @patch("ab_test_layout_engines._normalize_ppstructure_output")
    @patch("ab_test_layout_engines._run_ppstructure_page")
    @patch("ab_test_layout_engines._load_document_images")
    def test_run_engine_layoutlm_uses_repo_adapters(
        self,
        mock_load_document_images,
        mock_run_ppstructure_page,
        mock_normalize_ppstructure_output,
        mock_run_layoutlm_regions,
        mock_run_layoutlm_entities,
        tmp_path,
    ):
        from ab_test_layout_engines import _run_engine

        doc = tmp_path / "sample.png"
        doc.write_bytes(b"fake-image")

        mock_load_document_images.return_value = [object()]
        mock_run_ppstructure_page.return_value = [{"type": "text"}]
        mock_normalize_ppstructure_output.return_value = (
            [{"type": "paragraph", "bbox": [0, 0, 100, 100]}],
            [{"type": "DATE", "text": "2026-03-17"}],
            [("Invoice dated 2026-03-17", 0.9, [0, 0, 120, 20])],
        )
        mock_run_layoutlm_regions.return_value = [{"type": "paragraph", "bbox": [0, 0, 120, 120]}]
        mock_run_layoutlm_entities.return_value = [{"type": "DATE", "text": "2026-03-17", "source": "layoutlm"}]

        result = _run_engine("layoutlm", doc)

        assert result.error == ""
        assert result.regions == [{"type": "paragraph", "bbox": [0, 0, 120, 120]}]
        assert result.entities == [{"type": "DATE", "text": "2026-03-17", "source": "layoutlm"}]
        assert all(entity.get("text") != "placeholder" for entity in result.entities)
        mock_run_layoutlm_regions.assert_called_once()
        mock_run_layoutlm_entities.assert_called_once()

    def test_extract_entities_from_text_uses_heuristic_confidence_and_international_phone(self):
        from ab_test_layout_engines import (
            _REGEX_ENTITY_CONFIDENCE,
            _extract_entities_from_text,
        )

        entities = _extract_entities_from_text(
            "Contact +44 020 7946 0958 on 2026-03-17",
            [0, 0, 120, 20],
            source="ppstructure",
            page_num=1,
        )

        assert {entity["type"] for entity in entities} >= {"DATE", "PHONE_NUMBER"}
        assert all(entity["confidence"] == _REGEX_ENTITY_CONFIDENCE for entity in entities)

    def test_coerce_bbox_accepts_four_point_polygon(self):
        from ab_test_layout_engines import _coerce_bbox

        bbox = _coerce_bbox([[0, 0], [10, 0], [10, 10], [0, 10]])

        assert bbox == [0.0, 0.0, 10.0, 10.0]

    def test_coerce_bbox_returns_fallback_for_unparseable_shape(self):
        from ab_test_layout_engines import _coerce_bbox

        bbox = _coerce_bbox(["bad", object(), None, []], fallback=[1, 2, 3, 4])

        assert bbox == [1, 2, 3, 4]

    def test_normalize_ppstructure_output_coerces_polygon_bboxes(self):
        from ab_test_layout_engines import _normalize_ppstructure_output

        raw_result = [
            {
                "type": "text",
                "bbox": [[0, 0], [100, 0], [100, 50], [0, 50]],
                "score": 0.9,
                "res": [
                    {
                        "text": "Invoice 2026-03-17",
                        "bbox": [[5, 5], [95, 5], [95, 25], [5, 25]],
                        "confidence": 0.8,
                    }
                ],
            }
        ]

        regions, entities, paddle_lines = _normalize_ppstructure_output(
            raw_result,
            page_num=1,
        )

        assert regions == [
            {
                "type": "paragraph",
                "bbox": [0.0, 0.0, 100.0, 50.0],
                "confidence": 0.9,
                "page_num": 1,
            }
        ]
        assert paddle_lines == [
            ("Invoice 2026-03-17", 0.8, [5.0, 5.0, 95.0, 25.0])
        ]
        assert {entity["type"] for entity in entities} >= {"DATE"}


# ===================================================================
# Item 19: benchmark_accuracy
# ===================================================================

class TestLevenshteinDistance:
    """Test edit distance computation."""

    def test_identical_strings(self):
        from benchmark_accuracy import levenshtein_distance
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        from benchmark_accuracy import levenshtein_distance
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_substitution(self):
        from benchmark_accuracy import levenshtein_distance
        assert levenshtein_distance("cat", "car") == 1

    def test_single_insertion(self):
        from benchmark_accuracy import levenshtein_distance
        assert levenshtein_distance("cat", "cats") == 1

    def test_single_deletion(self):
        from benchmark_accuracy import levenshtein_distance
        assert levenshtein_distance("cats", "cat") == 1

    def test_completely_different(self):
        from benchmark_accuracy import levenshtein_distance
        assert levenshtein_distance("abc", "xyz") == 3

    def test_symmetric(self):
        from benchmark_accuracy import levenshtein_distance
        assert levenshtein_distance("abc", "aec") == levenshtein_distance("aec", "abc")


class TestCER:
    """Test Character Error Rate computation."""

    def test_perfect_cer(self):
        from benchmark_accuracy import compute_cer
        assert compute_cer("hello world", "hello world") == 0.0

    def test_empty_reference(self):
        from benchmark_accuracy import compute_cer
        assert compute_cer("", "") == 0.0
        assert compute_cer("", "something") == 1.0

    def test_partial_error(self):
        from benchmark_accuracy import compute_cer
        # "hello" -> "hallo": 1 substitution, len=5
        cer = compute_cer("hello", "hallo")
        assert abs(cer - 0.2) < 1e-6

    def test_complete_error(self):
        from benchmark_accuracy import compute_cer
        cer = compute_cer("abc", "xyz")
        assert cer == 1.0


class TestWER:
    """Test Word Error Rate computation."""

    def test_perfect_wer(self):
        from benchmark_accuracy import compute_wer
        assert compute_wer("hello world", "hello world") == 0.0

    def test_empty_reference(self):
        from benchmark_accuracy import compute_wer
        assert compute_wer("", "") == 0.0
        assert compute_wer("", "word") == 1.0

    def test_one_word_wrong(self):
        from benchmark_accuracy import compute_wer
        # 3 words, 1 substitution -> WER = 1/3
        wer = compute_wer("the quick fox", "the slow fox")
        assert abs(wer - 1 / 3) < 1e-6

    def test_all_words_wrong(self):
        from benchmark_accuracy import compute_wer
        wer = compute_wer("hello world", "goodbye earth")
        assert wer == 1.0

    def test_insertion(self):
        from benchmark_accuracy import compute_wer
        # "hello" -> "hello world": 1 insertion, ref len=1
        wer = compute_wer("hello", "hello world")
        assert wer == 1.0  # 1 insertion / 1 ref word


class TestPrecisionRecallF1:
    """Test token-level precision/recall/F1."""

    def test_perfect_match(self):
        from benchmark_accuracy import compute_precision_recall_f1
        p, r, f1 = compute_precision_recall_f1(
            ["a", "b", "c"], ["a", "b", "c"],
        )
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_no_match(self):
        from benchmark_accuracy import compute_precision_recall_f1
        p, r, f1 = compute_precision_recall_f1(
            ["a", "b"], ["c", "d"],
        )
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_empty_both(self):
        from benchmark_accuracy import compute_precision_recall_f1
        p, r, f1 = compute_precision_recall_f1([], [])
        assert f1 == 1.0


class TestDatasetLoading:
    """Test dataset pair loading."""

    def test_load_custom_pairs(self):
        from benchmark_accuracy import load_custom_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = Path(tmpdir) / "sample.gt.txt"
            pred_path = Path(tmpdir) / "sample.pred.txt"
            gt_path.write_text("hello world", encoding="utf-8")
            pred_path.write_text("hello earth", encoding="utf-8")

            pairs = load_custom_pairs(tmpdir)
            assert len(pairs) == 1
            assert pairs[0][0] == "sample"
            assert pairs[0][1] == "hello world"
            assert pairs[0][2] == "hello earth"

    def test_load_custom_pairs_missing_pred(self):
        from benchmark_accuracy import load_custom_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = Path(tmpdir) / "doc.gt.txt"
            gt_path.write_text("reference text", encoding="utf-8")

            pairs = load_custom_pairs(tmpdir)
            assert len(pairs) == 1
            assert pairs[0][2] == ""  # no prediction file

    def test_load_funsd_pairs(self):
        from benchmark_accuracy import load_funsd_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "form": [
                    {
                        "words": [
                            {"text": "Name"},
                            {"text": "John"},
                        ],
                    },
                ],
            }
            json_path = Path(tmpdir) / "page1.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")

            pairs = load_funsd_pairs(tmpdir)
            assert len(pairs) == 1
            assert "Name John" in pairs[0][1]

    def test_load_empty_dir(self):
        from benchmark_accuracy import load_custom_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs = load_custom_pairs(tmpdir)
            assert pairs == []


class TestBenchmarkAccuracyCLI:
    """Test accuracy benchmark CLI parsing."""

    def test_parse_args(self):
        from benchmark_accuracy import parse_args
        args = parse_args(["--dataset-dir", "/tmp/data"])
        assert args.dataset_dir == "/tmp/data"
        assert args.dataset_format == "custom"
        assert args.engine == "paddle"

    def test_parse_args_full(self):
        from benchmark_accuracy import parse_args
        args = parse_args([
            "--dataset-dir", "/data",
            "--dataset-format", "funsd",
            "--engine", "tesseract",
            "--output-dir", "/output",
        ])
        assert args.dataset_format == "funsd"
        assert args.engine == "tesseract"


# ===================================================================
# Item 20: calibrate_confidence
# ===================================================================

class TestECE:
    """Test Expected Calibration Error computation."""

    def test_perfect_calibration(self):
        from calibrate_confidence import compute_ece
        # All predictions at 1.0, all correct
        preds = [1.0] * 10
        labels = [1] * 10
        ece = compute_ece(preds, labels, n_bins=10)
        assert ece == 0.0

    def test_worst_calibration(self):
        from calibrate_confidence import compute_ece
        # All predictions at 1.0, all wrong
        preds = [1.0] * 10
        labels = [0] * 10
        ece = compute_ece(preds, labels, n_bins=10)
        assert abs(ece - 1.0) < 1e-6

    def test_empty_data(self):
        from calibrate_confidence import compute_ece
        assert compute_ece([], [], n_bins=10) == 0.0

    def test_moderate_calibration(self):
        from calibrate_confidence import compute_ece
        # 50% confidence, 50% accuracy -> ECE near 0
        preds = [0.5] * 100
        labels = [1] * 50 + [0] * 50
        ece = compute_ece(preds, labels, n_bins=10)
        assert ece < 0.05  # Should be well-calibrated


class TestReliabilityBins:
    """Test reliability diagram bin computation."""

    def test_bin_structure(self):
        from calibrate_confidence import compute_reliability_bins
        preds = [0.1, 0.5, 0.9]
        labels = [0, 1, 1]
        bins = compute_reliability_bins(preds, labels, n_bins=10)
        assert len(bins) == 10
        for b in bins:
            assert "bin_start" in b
            assert "bin_end" in b
            assert "avg_confidence" in b
            assert "avg_accuracy" in b
            assert "count" in b

    def test_empty_bins(self):
        from calibrate_confidence import compute_reliability_bins
        bins = compute_reliability_bins([], [], n_bins=10)
        assert bins == []


class TestTemperatureScaling:
    """Test temperature scaling calibration."""

    def test_returns_calibrated_values(self):
        from calibrate_confidence import apply_temperature_scaling
        confs = [0.9, 0.8, 0.7, 0.6, 0.95]
        labels = [1, 1, 0, 0, 1]
        calibrated, temp = apply_temperature_scaling(confs, labels)
        assert len(calibrated) == len(confs)
        assert temp > 0
        for c in calibrated:
            assert 0.0 <= c <= 1.0

    def test_empty_input(self):
        from calibrate_confidence import apply_temperature_scaling
        calibrated, temp = apply_temperature_scaling([], [])
        assert calibrated == []
        assert temp == 1.0


class TestPlattScaling:
    """Test Platt scaling calibration."""

    def test_returns_calibrated_values(self):
        from calibrate_confidence import apply_platt_scaling
        confs = [0.9, 0.8, 0.7, 0.6, 0.95]
        labels = [1, 1, 0, 0, 1]
        calibrated, a, b = apply_platt_scaling(confs, labels)
        assert len(calibrated) == len(confs)
        for c in calibrated:
            assert 0.0 <= c <= 1.0

    def test_empty_input(self):
        from calibrate_confidence import apply_platt_scaling
        calibrated, a, b = apply_platt_scaling([], [])
        assert calibrated == []


class TestIsotonicRegression:
    """Test isotonic regression calibration."""

    def test_without_sklearn(self):
        from calibrate_confidence import apply_isotonic_regression
        # This test may succeed or fail depending on sklearn availability
        confs = [0.9, 0.8, 0.7, 0.6]
        labels = [1, 1, 0, 0]
        calibrated, success = apply_isotonic_regression(confs, labels)
        assert len(calibrated) == len(confs)
        # success depends on sklearn availability

    def test_empty_input(self):
        from calibrate_confidence import apply_isotonic_regression
        calibrated, success = apply_isotonic_regression([], [])
        assert calibrated == []
        assert success is False


class TestCalibrationDataLoading:
    """Test calibration data loading from JSON files."""

    def test_load_valid_data(self):
        from calibrate_confidence import load_calibration_data
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "predictions": [
                    {"label": "B-DATE", "confidence": 0.9, "logit": 2.0},
                    {"label": "O", "confidence": 0.5, "logit": 0.0},
                ],
                "ground_truth": ["B-DATE", "O"],
            }
            json_path = Path(tmpdir) / "batch1.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")

            result = load_calibration_data(tmpdir)
            assert len(result.confidences) == 2
            assert len(result.labels) == 2
            assert result.confidences[0] == 0.9

    def test_load_empty_dir(self):
        from calibrate_confidence import load_calibration_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_calibration_data(tmpdir)
            assert len(result.confidences) == 0

    def test_load_invalid_dir(self):
        from calibrate_confidence import load_calibration_data
        result = load_calibration_data("/nonexistent/path")
        assert len(result.confidences) == 0

    def test_load_multiple_files(self):
        from calibrate_confidence import load_calibration_data
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                data = {
                    "predictions": [
                        {"label": "A", "confidence": 0.8},
                    ],
                    "ground_truth": ["A"],
                }
                json_path = Path(tmpdir) / f"batch{i}.json"
                json_path.write_text(json.dumps(data), encoding="utf-8")

            result = load_calibration_data(tmpdir)
            assert len(result.confidences) == 3


class TestCalibrationPipeline:
    """Test the full calibration pipeline."""

    def test_pipeline_temperature(self):
        from calibrate_confidence import run_calibration_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            out_dir = Path(tmpdir) / "output"

            data = {
                "predictions": [
                    {"label": "A", "confidence": 0.9, "logit": 2.0},
                    {"label": "A", "confidence": 0.8, "logit": 1.0},
                    {"label": "B", "confidence": 0.6, "logit": 0.5},
                    {"label": "A", "confidence": 0.7, "logit": 0.8},
                ],
                "ground_truth": ["A", "A", "A", "B"],
            }
            json_path = data_dir / "batch.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")

            reports = run_calibration_pipeline(
                data_dir=str(data_dir),
                method="temperature",
                output_dir=str(out_dir),
            )
            assert len(reports) == 1
            assert reports[0].method == "temperature"
            assert reports[0].num_samples == 4
            assert 0.0 <= reports[0].ece_after <= 1.0

    def test_pipeline_all_methods(self):
        from calibrate_confidence import run_calibration_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            out_dir = Path(tmpdir) / "output"

            data = {
                "predictions": [
                    {"label": "A", "confidence": 0.9},
                    {"label": "B", "confidence": 0.3},
                ],
                "ground_truth": ["A", "A"],
            }
            json_path = data_dir / "batch.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")

            reports = run_calibration_pipeline(
                data_dir=str(data_dir),
                method="all",
                output_dir=str(out_dir),
            )
            # Should have temperature, platt, and possibly isotonic
            method_names = [r.method for r in reports]
            assert "temperature" in method_names
            assert "platt" in method_names

    def test_pipeline_empty_data(self):
        from calibrate_confidence import run_calibration_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            reports = run_calibration_pipeline(
                data_dir=tmpdir,
                method="temperature",
                output_dir=os.path.join(tmpdir, "out"),
            )
            assert reports == []


class TestCalibrationCLI:
    """Test calibration CLI parsing."""

    def test_parse_args(self):
        from calibrate_confidence import parse_args
        args = parse_args(["--data-dir", "/tmp/cal"])
        assert args.data_dir == "/tmp/cal"
        assert args.method == "all"
        assert args.n_bins == 10

    def test_parse_args_full(self):
        from calibrate_confidence import parse_args
        args = parse_args([
            "--data-dir", "/data",
            "--method", "platt",
            "--n-bins", "20",
            "--output-dir", "/out",
        ])
        assert args.method == "platt"
        assert args.n_bins == 20


# ===================================================================
# Cross-cutting integration tests
# ===================================================================

class TestAccuracyBenchmarkIntegration:
    """Integration test: run accuracy benchmark on synthetic data."""

    def test_run_benchmark_on_custom_pairs(self):
        from benchmark_accuracy import run_accuracy_benchmark
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            out_dir = Path(tmpdir) / "output"

            # Create ground truth and prediction pairs
            (data_dir / "doc1.gt.txt").write_text("hello world", encoding="utf-8")
            (data_dir / "doc1.pred.txt").write_text("hello world", encoding="utf-8")
            (data_dir / "doc2.gt.txt").write_text("the quick brown fox", encoding="utf-8")
            (data_dir / "doc2.pred.txt").write_text("the quick brown dog", encoding="utf-8")

            result = run_accuracy_benchmark(
                dataset_dir=str(data_dir),
                dataset_format="custom",
                engine="test_engine",
                output_dir=str(out_dir),
            )
            assert result.num_samples == 2
            assert result.avg_cer >= 0.0
            assert result.avg_wer >= 0.0
            assert 0.0 <= result.f1 <= 1.0
            # First doc is perfect, second has 1 word error
            assert result.avg_wer > 0.0


class TestCalibrationIntegration:
    """Integration test: full calibration on synthetic data."""

    def test_calibration_improves_or_maintains_ece(self):
        from calibrate_confidence import (
            apply_temperature_scaling,
            compute_ece,
        )
        # Overconfident model: predicts 0.95 but only 60% accurate
        confs = [0.95] * 50 + [0.95] * 50
        labels = [1] * 30 + [0] * 20 + [1] * 30 + [0] * 20

        ece_before = compute_ece(confs, labels, n_bins=10)
        calibrated, _ = apply_temperature_scaling(confs, labels)
        ece_after = compute_ece(calibrated, labels, n_bins=10)

        # Temperature scaling should improve or maintain ECE
        assert ece_after <= ece_before + 0.01  # Allow tiny tolerance


class TestEdgeCases:
    """Test edge cases across all tools."""

    def test_cer_single_char(self):
        from benchmark_accuracy import compute_cer
        assert compute_cer("a", "b") == 1.0
        assert compute_cer("a", "a") == 0.0

    def test_wer_single_word(self):
        from benchmark_accuracy import compute_wer
        assert compute_wer("hello", "hello") == 0.0
        assert compute_wer("hello", "goodbye") == 1.0

    def test_iou_touching_boxes(self):
        from ab_test_layout_engines import BoundingBox, compute_iou
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(10, 0, 20, 10)  # Adjacent, not overlapping
        assert compute_iou(a, b) == 0.0

    def test_ece_all_same_bin(self):
        from calibrate_confidence import compute_ece
        preds = [0.55] * 100
        labels = [1] * 55 + [0] * 45
        ece = compute_ece(preds, labels, n_bins=10)
        assert ece < 0.05

    def test_mann_whitney_with_ties(self):
        from ab_test_layout_engines import mann_whitney_u_test
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [1.0, 1.0, 1.0, 1.0, 1.0]
        u, p = mann_whitney_u_test(a, b)
        assert p > 0.05  # Identical => not significant
