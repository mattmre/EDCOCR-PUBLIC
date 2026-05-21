"""Tests for scripts/benchmark_doclaynet.py — DocLayNet layout analysis benchmark."""

import json
import sys
from pathlib import Path
from unittest import mock

# Ensure project root is on sys.path
_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TEST_DIR.parent

# Ensure scripts dir is on sys.path for direct import
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from scripts.benchmark_doclaynet import (
    AnnotatedPage,
    BBox,
    BenchmarkReport,
    LayoutRegion,
    ModelResult,
    compute_ap_at_iou,
    compute_iou,
    compute_map,
    compute_per_type_f1,
    compute_per_type_iou,
    format_markdown_report,
    format_text_report,
    generate_corpus,
    generate_synthetic_page,
    load_corpus,
    run_heuristic,
)

# ---------------------------------------------------------------------------
# BBox tests
# ---------------------------------------------------------------------------


class TestBBox:
    """Tests for the BBox data class."""

    def test_area_positive(self):
        box = BBox(10, 20, 110, 120)
        assert box.area() == 100 * 100

    def test_area_zero_width(self):
        box = BBox(50, 50, 50, 100)
        assert box.area() == 0

    def test_area_zero_height(self):
        box = BBox(50, 50, 100, 50)
        assert box.area() == 0

    def test_area_inverted_returns_zero(self):
        box = BBox(100, 100, 50, 50)
        assert box.area() == 0

    def test_to_list(self):
        box = BBox(1, 2, 3, 4)
        assert box.to_list() == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# IoU computation tests
# ---------------------------------------------------------------------------


class TestComputeIoU:
    """Tests for compute_iou with known bounding boxes."""

    def test_perfect_overlap(self):
        box = BBox(0, 0, 100, 100)
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        box_a = BBox(0, 0, 50, 50)
        box_b = BBox(100, 100, 200, 200)
        assert compute_iou(box_a, box_b) == 0.0

    def test_partial_overlap(self):
        box_a = BBox(0, 0, 100, 100)
        box_b = BBox(50, 50, 150, 150)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert abs(compute_iou(box_a, box_b) - expected) < 1e-6

    def test_one_inside_other(self):
        outer = BBox(0, 0, 200, 200)
        inner = BBox(50, 50, 100, 100)
        # Intersection = 50*50 = 2500
        # Union = 40000 + 2500 - 2500 = 40000
        expected = 2500 / 40000
        assert abs(compute_iou(outer, inner) - expected) < 1e-6

    def test_touching_edge(self):
        box_a = BBox(0, 0, 50, 50)
        box_b = BBox(50, 0, 100, 50)
        # Touching at edge, no overlap area
        assert compute_iou(box_a, box_b) == 0.0

    def test_zero_area_boxes(self):
        box_a = BBox(0, 0, 0, 0)
        box_b = BBox(0, 0, 0, 0)
        assert compute_iou(box_a, box_b) == 0.0

    def test_half_overlap_horizontal(self):
        box_a = BBox(0, 0, 100, 100)
        box_b = BBox(50, 0, 150, 100)
        # Intersection: 50x100 = 5000
        # Union: 10000 + 10000 - 5000 = 15000
        expected = 5000 / 15000
        assert abs(compute_iou(box_a, box_b) - expected) < 1e-6


# ---------------------------------------------------------------------------
# Per-type IoU tests
# ---------------------------------------------------------------------------


class TestComputePerTypeIoU:
    """Tests for compute_per_type_iou."""

    def test_perfect_match_single_type(self):
        gt = [LayoutRegion("title", BBox(0, 0, 100, 50))]
        pred = [LayoutRegion("title", BBox(0, 0, 100, 50))]
        result = compute_per_type_iou(pred, gt, ["title"])
        assert result["title"] == 1.0

    def test_no_predictions(self):
        gt = [LayoutRegion("title", BBox(0, 0, 100, 50))]
        result = compute_per_type_iou([], gt, ["title"])
        assert result["title"] == 0.0

    def test_no_ground_truth(self):
        pred = [LayoutRegion("title", BBox(0, 0, 100, 50))]
        result = compute_per_type_iou(pred, [], ["title"])
        # No GT of type title -> not included in result
        assert "title" not in result

    def test_multiple_types(self):
        gt = [
            LayoutRegion("title", BBox(0, 0, 100, 50)),
            LayoutRegion("paragraph", BBox(0, 100, 200, 300)),
        ]
        pred = [
            LayoutRegion("title", BBox(0, 0, 100, 50)),
            LayoutRegion("paragraph", BBox(0, 100, 200, 300)),
        ]
        result = compute_per_type_iou(pred, gt, ["title", "paragraph"])
        assert result["title"] == 1.0
        assert result["paragraph"] == 1.0

    def test_type_mismatch(self):
        gt = [LayoutRegion("title", BBox(0, 0, 100, 50))]
        pred = [LayoutRegion("paragraph", BBox(0, 0, 100, 50))]
        result = compute_per_type_iou(pred, gt, ["title", "paragraph"])
        assert result["title"] == 0.0


# ---------------------------------------------------------------------------
# AP and mAP tests
# ---------------------------------------------------------------------------


class TestComputeAP:
    """Tests for compute_ap_at_iou and compute_map."""

    def test_perfect_predictions(self):
        gt = [LayoutRegion("title", BBox(0, 0, 100, 50), confidence=1.0)]
        pred = [LayoutRegion("title", BBox(0, 0, 100, 50), confidence=0.9)]
        ap = compute_ap_at_iou(pred, gt, 0.5, "title")
        assert ap == 1.0

    def test_no_predictions_no_gt(self):
        ap = compute_ap_at_iou([], [], 0.5, "title")
        assert ap == 1.0  # No GT and no preds -> perfect

    def test_no_predictions_with_gt(self):
        gt = [LayoutRegion("title", BBox(0, 0, 100, 50))]
        ap = compute_ap_at_iou([], gt, 0.5, "title")
        assert ap == 0.0

    def test_predictions_no_gt(self):
        pred = [LayoutRegion("title", BBox(0, 0, 100, 50), confidence=0.9)]
        ap = compute_ap_at_iou(pred, [], 0.5, "title")
        assert ap == 0.0  # False positives

    def test_mixed_precision_recall(self):
        gt = [
            LayoutRegion("title", BBox(0, 0, 100, 50)),
            LayoutRegion("title", BBox(0, 100, 100, 150)),
        ]
        pred = [
            LayoutRegion("title", BBox(0, 0, 100, 50), confidence=0.9),
            LayoutRegion("title", BBox(500, 500, 600, 550), confidence=0.5),
        ]
        ap = compute_ap_at_iou(pred, gt, 0.5, "title")
        # First pred matches, second doesn't, one GT unmatched
        assert 0.0 < ap < 1.0

    def test_map_single_type(self):
        gt = [LayoutRegion("title", BBox(0, 0, 100, 50))]
        pred = [LayoutRegion("title", BBox(0, 0, 100, 50), confidence=0.9)]
        m = compute_map(pred, gt, ["title"], 0.5)
        assert m == 1.0

    def test_map_multiple_types(self):
        gt = [
            LayoutRegion("title", BBox(0, 0, 100, 50)),
            LayoutRegion("paragraph", BBox(0, 100, 200, 300)),
        ]
        pred = [
            LayoutRegion("title", BBox(0, 0, 100, 50), confidence=0.9),
            LayoutRegion("paragraph", BBox(0, 100, 200, 300), confidence=0.8),
        ]
        m = compute_map(pred, gt, ["title", "paragraph"], 0.5)
        assert m == 1.0

    def test_map_75_stricter_threshold(self):
        gt = [LayoutRegion("title", BBox(0, 0, 100, 100))]
        # Prediction with lower IoU
        pred = [LayoutRegion("title", BBox(0, 0, 60, 100), confidence=0.9)]
        # IoU = 60*100 / (100*100+60*100-60*100) = 6000/10000 = 0.6
        m50 = compute_map(pred, gt, ["title"], 0.5)
        m75 = compute_map(pred, gt, ["title"], 0.75)
        assert m50 == 1.0  # Passes 0.5 threshold
        assert m75 == 0.0  # Fails 0.75 threshold


# ---------------------------------------------------------------------------
# F1 score tests
# ---------------------------------------------------------------------------


class TestComputePerTypeF1:
    """Tests for compute_per_type_f1."""

    def test_perfect_f1(self):
        gt = [LayoutRegion("title", BBox(0, 0, 100, 50))]
        pred = [LayoutRegion("title", BBox(0, 0, 100, 50), confidence=0.9)]
        result = compute_per_type_f1(pred, gt, ["title"], 0.5)
        assert result["title"] == 1.0

    def test_zero_f1_no_overlap(self):
        gt = [LayoutRegion("title", BBox(0, 0, 100, 50))]
        pred = [LayoutRegion("title", BBox(500, 500, 600, 550), confidence=0.9)]
        result = compute_per_type_f1(pred, gt, ["title"], 0.5)
        assert result["title"] == 0.0

    def test_empty_inputs(self):
        result = compute_per_type_f1([], [], ["title"])
        assert "title" not in result

    def test_multiple_types_mixed(self):
        gt = [
            LayoutRegion("title", BBox(0, 0, 100, 50)),
            LayoutRegion("paragraph", BBox(0, 100, 200, 300)),
        ]
        pred = [
            LayoutRegion("title", BBox(0, 0, 100, 50), confidence=0.9),
            # paragraph prediction is wrong location
            LayoutRegion("paragraph", BBox(500, 500, 700, 700), confidence=0.5),
        ]
        result = compute_per_type_f1(pred, gt, ["title", "paragraph"], 0.5)
        assert result["title"] == 1.0
        assert result["paragraph"] == 0.0


# ---------------------------------------------------------------------------
# Synthetic document generation tests
# ---------------------------------------------------------------------------


class TestSyntheticGeneration:
    """Tests for generate_synthetic_page and generate_corpus."""

    def test_generate_page_returns_annotated_page(self):
        page, img = generate_synthetic_page(
            page_width=800, page_height=600, seed=42,
        )
        assert isinstance(page, AnnotatedPage)
        assert page.width == 800
        assert page.height == 600
        assert len(page.regions) > 0

    def test_generate_page_regions_within_bounds(self):
        page, img = generate_synthetic_page(
            page_width=1000, page_height=1200, seed=0,
        )
        for region in page.regions:
            assert region.bbox.x1 >= 0
            assert region.bbox.y1 >= 0
            assert region.bbox.x2 <= 1000
            assert region.bbox.y2 <= 1200
            assert region.bbox.x2 > region.bbox.x1
            assert region.bbox.y2 > region.bbox.y1

    def test_generate_page_returns_pil_image(self):
        from PIL import Image
        page, img = generate_synthetic_page(
            page_width=500, page_height=400, seed=1,
        )
        assert isinstance(img, Image.Image)
        assert img.size == (500, 400)

    def test_generate_page_deterministic(self):
        page1, _ = generate_synthetic_page(seed=42)
        page2, _ = generate_synthetic_page(seed=42)
        assert len(page1.regions) == len(page2.regions)
        for r1, r2 in zip(page1.regions, page2.regions):
            assert r1.region_type == r2.region_type
            assert r1.bbox.to_list() == r2.bbox.to_list()

    def test_generate_page_different_seeds(self):
        page1, _ = generate_synthetic_page(seed=0)
        page2, _ = generate_synthetic_page(seed=99)
        # Different seeds should produce different layouts
        regions1 = [(r.region_type, r.bbox.to_list()) for r in page1.regions]
        regions2 = [(r.region_type, r.bbox.to_list()) for r in page2.regions]
        assert regions1 != regions2

    def test_generate_page_has_valid_region_types(self):
        from scripts.benchmark_doclaynet import LAYOUT_REGION_TYPES
        page, _ = generate_synthetic_page(seed=5)
        for region in page.regions:
            assert region.region_type in LAYOUT_REGION_TYPES

    def test_generate_corpus_creates_files(self, tmp_path):
        output_dir = str(tmp_path / "corpus")
        paths = generate_corpus(output_dir, count=3, page_width=500, page_height=400)
        assert len(paths) == 3

        images_dir = tmp_path / "corpus" / "images"
        annotations_dir = tmp_path / "corpus" / "annotations"
        assert images_dir.is_dir()
        assert annotations_dir.is_dir()

        assert len(list(images_dir.glob("*.png"))) == 3
        assert len(list(annotations_dir.glob("*.json"))) == 3

    def test_generate_corpus_annotations_valid_json(self, tmp_path):
        output_dir = str(tmp_path / "corpus")
        paths = generate_corpus(output_dir, count=2, page_width=500, page_height=400)

        for ann_path in paths:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "image_path" in data
            assert "width" in data
            assert "height" in data
            assert "regions" in data
            assert len(data["regions"]) > 0
            for region in data["regions"]:
                assert "region_type" in region
                assert "bbox" in region
                assert len(region["bbox"]) == 4

    def test_load_corpus_roundtrip(self, tmp_path):
        output_dir = str(tmp_path / "corpus")
        generate_corpus(output_dir, count=3, page_width=500, page_height=400)
        pages = load_corpus(output_dir)
        assert len(pages) == 3
        for page, img in pages:
            assert isinstance(page, AnnotatedPage)
            assert len(page.regions) > 0
            assert img.size == (500, 400)


# ---------------------------------------------------------------------------
# Model runner tests
# ---------------------------------------------------------------------------


class TestModelRunners:
    """Tests for model runner functions with mocked models."""

    def test_run_ppstructure_import_error(self):
        """run_ppstructure returns empty list when paddleocr is not installed."""
        with mock.patch.dict(sys.modules, {"paddleocr": None}):
            from PIL import Image

            from scripts.benchmark_doclaynet import run_ppstructure
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            result = run_ppstructure(img)
            assert result == []

    def test_run_layoutlmv3_import_error(self):
        """run_layoutlmv3 returns empty list when classification is unavailable."""
        with mock.patch.dict(sys.modules, {"classification": None}):
            from PIL import Image

            from scripts.benchmark_doclaynet import run_layoutlmv3
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            result = run_layoutlmv3(img)
            assert result == []

    def test_run_heuristic_with_classification(self):
        """run_heuristic returns a list of regions when classification is available."""
        # Mock classify_page_by_text to return a known result
        mock_classify = mock.MagicMock(return_value={
            "predicted_type": "invoice",
            "confidence": 0.85,
        })
        with mock.patch.dict(
            "sys.modules",
            {"classification": mock.MagicMock(classify_page_by_text=mock_classify)},
        ):
            # Need to reimport to pick up the mock
            result = run_heuristic("INVOICE #12345", 800, 600)
            assert len(result) >= 1
            assert result[0].region_type in ("table", "paragraph", "list")
            assert result[0].bbox.x2 == 800
            assert result[0].bbox.y2 == 600

    def test_run_heuristic_import_error(self):
        """run_heuristic returns empty list when classification is unavailable."""
        with mock.patch.dict(sys.modules, {"classification": None}):
            from scripts.benchmark_doclaynet import run_heuristic
            result = run_heuristic("some text")
            assert result == []


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Tests for report formatting functions."""

    def _make_sample_report(self):
        """Create a sample BenchmarkReport for testing."""
        return BenchmarkReport(
            corpus_dir="/tmp/test_corpus",
            total_pages=10,
            region_types=["title", "paragraph", "table"],
            models=[
                ModelResult(
                    model_name="ppstructure",
                    available=True,
                    total_pages=10,
                    per_type_iou={"title": 0.85, "paragraph": 0.72, "table": 0.91},
                    mean_iou=0.8267,
                    map_50=0.78,
                    map_75=0.55,
                    per_type_f1={"title": 0.90, "paragraph": 0.80, "table": 0.95},
                    mean_f1=0.8833,
                    avg_inference_ms=150.5,
                    p95_inference_ms=210.3,
                    min_inference_ms=120.0,
                    max_inference_ms=250.0,
                    timestamp="2026-03-15T00:00:00.000Z",
                ),
                ModelResult(
                    model_name="layoutlmv3",
                    available=False,
                    total_pages=10,
                    timestamp="2026-03-15T00:00:00.000Z",
                ),
                ModelResult(
                    model_name="heuristic",
                    available=True,
                    total_pages=10,
                    per_type_iou={"title": 0.30, "paragraph": 0.25},
                    mean_iou=0.275,
                    map_50=0.20,
                    map_75=0.05,
                    per_type_f1={"title": 0.40, "paragraph": 0.35},
                    mean_f1=0.375,
                    avg_inference_ms=2.5,
                    p95_inference_ms=5.0,
                    min_inference_ms=1.0,
                    max_inference_ms=8.0,
                    timestamp="2026-03-15T00:00:00.000Z",
                ),
            ],
            timestamp="2026-03-15T00:00:00.000Z",
        )

    def test_text_report_contains_model_names(self):
        report = self._make_sample_report()
        text = format_text_report(report)
        assert "ppstructure" in text
        assert "layoutlmv3" in text
        assert "heuristic" in text

    def test_text_report_contains_metrics(self):
        report = self._make_sample_report()
        text = format_text_report(report)
        assert "mIoU" in text
        assert "mAP@50" in text
        assert "mAP@75" in text
        assert "mF1" in text

    def test_text_report_contains_header(self):
        report = self._make_sample_report()
        text = format_text_report(report)
        assert "DOCLAYNET LAYOUT ANALYSIS BENCHMARK" in text

    def test_markdown_report_contains_tables(self):
        report = self._make_sample_report()
        md = format_markdown_report(report)
        assert "| Model |" in md
        assert "| Region Type |" in md
        assert "## Overall Comparison" in md
        assert "## Per-Type IoU" in md
        assert "## Per-Type F1 Score" in md
        assert "## Inference Timing" in md

    def test_markdown_report_shows_availability(self):
        report = self._make_sample_report()
        md = format_markdown_report(report)
        assert "| ppstructure | Yes |" in md
        assert "| layoutlmv3 | No |" in md

    def test_markdown_report_contains_corpus_info(self):
        report = self._make_sample_report()
        md = format_markdown_report(report)
        assert "/tmp/test_corpus" in md
        assert "10" in md

    def test_text_report_empty_report(self):
        report = BenchmarkReport()
        text = format_text_report(report)
        assert "DOCLAYNET LAYOUT ANALYSIS BENCHMARK" in text

    def test_report_json_roundtrip(self):
        """Verify that report can be serialized to JSON and reconstructed."""
        from dataclasses import asdict
        report = self._make_sample_report()
        json_str = json.dumps(asdict(report), indent=2)
        data = json.loads(json_str)
        assert data["corpus_dir"] == "/tmp/test_corpus"
        assert data["total_pages"] == 10
        assert len(data["models"]) == 3
        assert data["models"][0]["model_name"] == "ppstructure"
        assert data["models"][0]["mean_iou"] == 0.8267


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_generate_command(self):
        """Test 'generate' subcommand parsing."""
        from scripts.benchmark_doclaynet import main
        with mock.patch(
            "sys.argv",
            ["benchmark_doclaynet.py", "generate", "--count", "5", "--output-dir", "/tmp/out"],
        ):
            with mock.patch(
                "scripts.benchmark_doclaynet.generate_corpus"
            ) as mock_gen:
                mock_gen.return_value = []
                result = main()
                assert result == 0
                mock_gen.assert_called_once_with(
                    output_dir="/tmp/out",
                    count=5,
                    page_width=2550,
                    page_height=3300,
                )

    def test_run_command(self):
        """Test 'run' subcommand parsing."""
        from scripts.benchmark_doclaynet import main
        with mock.patch(
            "sys.argv",
            [
                "benchmark_doclaynet.py", "run",
                "--corpus-dir", "/tmp/corpus",
                "--models", "heuristic",
            ],
        ):
            with mock.patch(
                "scripts.benchmark_doclaynet.run_benchmark"
            ) as mock_run:
                mock_run.return_value = BenchmarkReport()
                result = main()
                assert result == 0
                mock_run.assert_called_once_with(
                    corpus_dir="/tmp/corpus",
                    models=["heuristic"],
                )

    def test_run_command_all_models(self):
        """Test 'run' with --models all expands to full model list."""
        from scripts.benchmark_doclaynet import main
        with mock.patch(
            "sys.argv",
            [
                "benchmark_doclaynet.py", "run",
                "--corpus-dir", "/tmp/corpus",
                "--models", "all",
            ],
        ):
            with mock.patch(
                "scripts.benchmark_doclaynet.run_benchmark"
            ) as mock_run:
                mock_run.return_value = BenchmarkReport()
                result = main()
                assert result == 0
                mock_run.assert_called_once_with(
                    corpus_dir="/tmp/corpus",
                    models=["ppstructure", "layoutlmv3", "heuristic"],
                )

    def test_report_command(self, tmp_path):
        """Test 'report' subcommand with a JSON results file."""
        from dataclasses import asdict
        results_file = tmp_path / "results.json"
        report = BenchmarkReport(
            corpus_dir="/tmp/corpus",
            total_pages=5,
            region_types=["title"],
            models=[
                ModelResult(
                    model_name="heuristic",
                    available=True,
                    total_pages=5,
                    mean_iou=0.5,
                ),
            ],
            timestamp="2026-01-01T00:00:00.000Z",
        )
        results_file.write_text(json.dumps(asdict(report), indent=2))

        from scripts.benchmark_doclaynet import main
        with mock.patch(
            "sys.argv",
            [
                "benchmark_doclaynet.py", "report",
                "--results", str(results_file),
                "--format", "markdown",
            ],
        ):
            result = main()
            assert result == 0

    def test_no_command_shows_help(self, capsys):
        """Test that running without a command returns 1."""
        from scripts.benchmark_doclaynet import main
        with mock.patch("sys.argv", ["benchmark_doclaynet.py"]):
            result = main()
            assert result == 1

    def test_report_missing_file(self, tmp_path):
        """Test report with non-existent results file returns 1."""
        from scripts.benchmark_doclaynet import main
        with mock.patch(
            "sys.argv",
            [
                "benchmark_doclaynet.py", "report",
                "--results", str(tmp_path / "nonexistent.json"),
            ],
        ):
            result = main()
            assert result == 1


# ---------------------------------------------------------------------------
# Integration: benchmark runner with mocked models
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    """Tests for run_benchmark with mocked model availability."""

    def test_run_benchmark_no_models_available(self, tmp_path):
        """Benchmark with no available models should still produce a report."""
        output_dir = str(tmp_path / "corpus")
        generate_corpus(output_dir, count=2, page_width=200, page_height=200)

        with mock.patch(
            "scripts.benchmark_doclaynet._check_model_available",
            return_value=False,
        ):
            from scripts.benchmark_doclaynet import run_benchmark
            report = run_benchmark(
                corpus_dir=output_dir,
                models=["ppstructure", "layoutlmv3"],
            )
            assert report.total_pages == 2
            assert len(report.models) == 2
            for m in report.models:
                assert not m.available
                assert m.mean_iou == 0.0

    def test_run_benchmark_empty_corpus(self, tmp_path):
        """Benchmark on empty corpus returns empty report."""
        output_dir = str(tmp_path / "empty_corpus")
        Path(output_dir).mkdir(parents=True)

        from scripts.benchmark_doclaynet import run_benchmark
        report = run_benchmark(corpus_dir=output_dir, models=["heuristic"])
        assert report.total_pages == 0
        assert len(report.models) == 0

    def test_run_benchmark_heuristic_only(self, tmp_path):
        """Benchmark with heuristic model runs successfully."""
        output_dir = str(tmp_path / "corpus")
        generate_corpus(output_dir, count=3, page_width=200, page_height=200)

        from scripts.benchmark_doclaynet import run_benchmark

        # Mock _check_model_available to return True only for heuristic
        def mock_check(model_name):
            return model_name == "heuristic"

        with mock.patch(
            "scripts.benchmark_doclaynet._check_model_available",
            side_effect=mock_check,
        ):
            # Mock run_heuristic to return regions that match GT
            with mock.patch(
                "scripts.benchmark_doclaynet.run_heuristic",
            ) as mock_heuristic:
                mock_heuristic.return_value = [
                    LayoutRegion("paragraph", BBox(0, 0, 200, 200), confidence=0.6),
                ]
                report = run_benchmark(
                    corpus_dir=output_dir,
                    models=["heuristic"],
                )
                assert report.total_pages == 3
                assert len(report.models) == 1
                assert report.models[0].model_name == "heuristic"
                assert report.models[0].available
                assert report.models[0].avg_inference_ms >= 0
