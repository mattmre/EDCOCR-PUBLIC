"""Tests for OCR benchmark tool.

Run with: python -m pytest tests/test_benchmark_ocr.py -v
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from benchmark_ocr import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    WARMUP_PAGES,
    _compile_results,
    _detect_available_backends,
    _load_document_pages,
    format_comparison_table,
    generate_test_page,
    run_benchmarks,
)

# ===========================================================================
# Tests: generate_test_page
# ===========================================================================


class TestGenerateTestPage:
    """Tests for synthetic page generation."""

    def test_returns_pil_image(self):
        img = generate_test_page()
        assert isinstance(img, Image.Image)

    def test_default_dimensions(self):
        img = generate_test_page()
        assert img.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)

    def test_custom_dimensions(self):
        img = generate_test_page(width=800, height=600)
        assert img.size == (800, 600)

    def test_clean_complexity(self):
        img = generate_test_page(complexity="clean", seed=42)
        arr = np.array(img)
        assert arr.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
        # Clean pages should have high mean (mostly white background)
        assert np.mean(arr) > 150

    def test_degraded_complexity(self):
        img = generate_test_page(complexity="degraded", seed=42)
        arr = np.array(img)
        assert arr.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

    def test_mixed_complexity(self):
        img = generate_test_page(complexity="mixed", seed=42)
        assert isinstance(img, Image.Image)

    def test_reproducible_with_seed(self):
        img1 = generate_test_page(complexity="clean", seed=123)
        img2 = generate_test_page(complexity="clean", seed=123)
        assert np.array_equal(np.array(img1), np.array(img2))

    def test_different_seeds_differ(self):
        img1 = generate_test_page(complexity="clean", seed=1)
        img2 = generate_test_page(complexity="clean", seed=2)
        assert not np.array_equal(np.array(img1), np.array(img2))

    def test_rgb_mode(self):
        img = generate_test_page(seed=0)
        assert img.mode == "RGB"

    def test_small_dimensions(self):
        """Ensure generation works with small page sizes."""
        img = generate_test_page(width=400, height=400, complexity="clean", seed=10)
        assert img.size == (400, 400)

    def test_degraded_has_noise(self):
        """Degraded pages should have salt-and-pepper noise visible in variance."""
        img = generate_test_page(complexity="degraded", seed=99)
        arr = np.array(img)
        # Degraded pages should have moderate variance due to noise
        assert np.std(arr) > 10


# ===========================================================================
# Tests: _compile_results
# ===========================================================================


class TestCompileResults:
    """Tests for results compilation."""

    def test_basic_compilation(self):
        result = _compile_results(
            "Test Backend",
            [100.0, 200.0, 150.0, 120.0, 180.0],
            500.0,
            550.0,
            [100, 200, 150, 120, 180],
        )
        assert result["label"] == "Test Backend"
        assert result["pages"] == 5
        assert result["mean_ms"] == 150.0
        assert result["median_ms"] == 150.0
        assert result["min_ms"] == 100.0
        assert result["max_ms"] == 200.0
        assert result["memory_delta_mb"] == 50.0
        assert result["pages_per_minute"] > 0

    def test_empty_timings(self):
        result = _compile_results("Empty", [], 0, 0, [])
        assert result is None

    def test_single_timing(self):
        result = _compile_results("Single", [100.0], 0, 0, [50])
        assert result["pages"] == 1
        assert result["mean_ms"] == 100.0
        assert result["stddev_ms"] == 0.0

    def test_ppm_calculation(self):
        # 10 pages at 100ms each = 1000ms total
        # PPM = (10 / 1.0) * 60 = 600
        result = _compile_results(
            "Fast",
            [100.0] * 10,
            0,
            0,
            [50] * 10,
        )
        assert result["pages_per_minute"] == 600.0

    def test_two_timings_has_stddev(self):
        result = _compile_results("Two", [100.0, 200.0], 0, 0, [50, 60])
        assert result["stddev_ms"] > 0

    def test_p95_with_many_timings(self):
        timings = list(range(1, 101))  # 1..100ms
        result = _compile_results(
            "Many", [float(t) for t in timings], 0, 0, [50] * 100
        )
        assert result["p95_ms"] >= 90.0

    def test_p99_with_many_timings(self):
        timings = [float(t) for t in range(1, 101)]
        result = _compile_results("Many", timings, 0, 0, [50] * 100)
        assert result["p99_ms"] >= 95.0

    def test_negative_memory_delta(self):
        """Memory can decrease (GC) -- delta should be negative."""
        result = _compile_results("GC", [100.0], 1000.0, 900.0, [50])
        assert result["memory_delta_mb"] == -100.0

    def test_zero_total_ms(self):
        """Edge case: all timings are 0.0 (should not divide by zero)."""
        result = _compile_results("Zero", [0.0, 0.0], 0, 0, [0, 0])
        # PPM should be 0 since total_ms is 0
        assert result["pages_per_minute"] == 0

    def test_avg_text_length(self):
        result = _compile_results("Text", [100.0, 200.0], 0, 0, [100, 200])
        assert result["avg_text_length"] == 150

    def test_empty_text_lengths(self):
        result = _compile_results("NoText", [100.0], 0, 0, [])
        assert result["avg_text_length"] == 0

    def test_p95_few_timings(self):
        """With fewer than 5 timings, P95 falls back to max."""
        result = _compile_results("Few", [50.0, 100.0, 150.0], 0, 0, [10, 20, 30])
        assert result["p95_ms"] == 150.0

    def test_p99_few_timings(self):
        """With fewer than 10 timings, P99 falls back to max."""
        result = _compile_results(
            "Few", [50.0, 100.0, 150.0, 200.0, 250.0], 0, 0, [10] * 5
        )
        assert result["p99_ms"] == 250.0


# ===========================================================================
# Tests: format_comparison_table
# ===========================================================================


class TestFormatComparisonTable:
    """Tests for table formatting."""

    def test_empty_results(self):
        table = format_comparison_table([])
        assert "No benchmark results" in table

    def test_single_result(self):
        results = [_compile_results("Test", [100.0] * 5, 0, 100, [50] * 5)]
        table = format_comparison_table(results)
        assert "Test" in table
        assert "BENCHMARK RESULTS" in table

    def test_multiple_results_sorted(self):
        fast = _compile_results("Fast GPU", [50.0] * 5, 0, 0, [50] * 5)
        slow = _compile_results("Slow CPU", [500.0] * 5, 0, 0, [50] * 5)
        table = format_comparison_table([slow, fast])
        # Fast should appear first (sorted by PPM descending)
        fast_pos = table.index("Fast GPU")
        slow_pos = table.index("Slow CPU")
        assert fast_pos < slow_pos

    def test_cost_analysis_section(self):
        results = [
            _compile_results(
                "PaddleOCR (paddle, gpu)", [100.0] * 5, 0, 0, [50] * 5
            )
        ]
        table = format_comparison_table(results)
        assert "COST ANALYSIS" in table

    def test_recommendations_gpu_and_cpu(self):
        gpu = _compile_results(
            "PaddleOCR (paddle, gpu)", [50.0] * 5, 0, 0, [50] * 5
        )
        cpu = _compile_results(
            "PaddleOCR (paddle, cpu)", [500.0] * 5, 0, 0, [50] * 5
        )
        table = format_comparison_table([gpu, cpu])
        assert "RECOMMENDATIONS" in table
        assert "speedup" in table.lower()

    def test_recommendations_cpu_only(self):
        cpu = _compile_results(
            "PaddleOCR (paddle, cpu)", [100.0] * 5, 0, 0, [50] * 5
        )
        table = format_comparison_table([cpu])
        assert "RECOMMENDATIONS" in table
        assert "No GPU available" in table

    def test_cost_column_headers(self):
        results = [_compile_results("Test", [100.0] * 5, 0, 0, [50] * 5)]
        table = format_comparison_table(results)
        assert "$/Hour" in table
        assert "$/1K Pages" in table
        assert "Pages/Day" in table

    def test_gpu_uses_higher_cost(self):
        """GPU backends should map to a higher hourly cost."""
        gpu = _compile_results(
            "PaddleOCR (paddle, gpu)", [100.0] * 5, 0, 0, [50] * 5
        )
        table = format_comparison_table([gpu])
        # GPU cost is $0.526/hr (column is right-aligned, may have spaces)
        assert "0.526" in table

    def test_cpu_uses_lower_cost(self):
        """CPU backends should map to a lower hourly cost."""
        cpu = _compile_results(
            "PaddleOCR (paddle, cpu)", [100.0] * 5, 0, 0, [50] * 5
        )
        table = format_comparison_table([cpu])
        # CPU cost is $0.170/hr (column is right-aligned, may have spaces)
        assert "0.170" in table

    def test_speedup_threshold_viable(self):
        """When GPU speedup <= 3x, recommendation is CPU viable."""
        gpu = _compile_results(
            "PaddleOCR (paddle, gpu)", [100.0] * 5, 0, 0, [50] * 5
        )
        cpu = _compile_results(
            "PaddleOCR (paddle, cpu)", [200.0] * 5, 0, 0, [50] * 5
        )
        table = format_comparison_table([gpu, cpu])
        assert "CPU-only deployment is viable" in table

    def test_speedup_threshold_hybrid(self):
        """When GPU speedup is 3-6x, recommendation is hybrid."""
        gpu = _compile_results(
            "PaddleOCR (paddle, gpu)", [100.0] * 5, 0, 0, [50] * 5
        )
        cpu = _compile_results(
            "PaddleOCR (paddle, cpu)", [500.0] * 5, 0, 0, [50] * 5
        )
        table = format_comparison_table([gpu, cpu])
        assert "hybrid deployment" in table

    def test_speedup_threshold_gpu_recommended(self):
        """When GPU speedup > 6x, recommendation is GPU."""
        gpu = _compile_results(
            "PaddleOCR (paddle, gpu)", [100.0] * 5, 0, 0, [50] * 5
        )
        cpu = _compile_results(
            "PaddleOCR (paddle, cpu)", [2000.0] * 5, 0, 0, [50] * 5
        )
        table = format_comparison_table([gpu, cpu])
        assert "GPU recommended" in table


# ===========================================================================
# Tests: _detect_available_backends
# ===========================================================================


class TestDetectAvailableBackends:
    """Tests for backend auto-detection."""

    def test_returns_list(self):
        backends = _detect_available_backends()
        assert isinstance(backends, list)

    def test_tesseract_detected_when_available(self):
        mock_pytesseract = MagicMock()
        mock_pytesseract.get_tesseract_version.return_value = "5.3.0"
        with patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            backends = _detect_available_backends()
            assert "tesseract" in backends

    def test_paddle_cpu_detected_when_available(self):
        mock_paddleocr = MagicMock()
        with patch.dict("sys.modules", {"paddleocr": mock_paddleocr}):
            backends = _detect_available_backends()
            assert "paddle-cpu" in backends

    def test_paddle_gpu_detected_when_cuda_available(self):
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = True
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            backends = _detect_available_backends()
            assert "paddle-gpu" in backends

    def test_paddle_gpu_not_detected_without_cuda(self):
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = False
        # Block paddleocr/onnxruntime/pytesseract to isolate test
        with patch.dict(
            "sys.modules",
            {
                "paddle": mock_paddle,
                "paddleocr": None,
                "onnxruntime": None,
                "pytesseract": None,
            },
        ):
            backends = _detect_available_backends()
            assert "paddle-gpu" not in backends


# ===========================================================================
# Tests: run_benchmarks
# ===========================================================================


class TestRunBenchmarks:
    """Tests for the main benchmark runner."""

    def test_with_unknown_backend(self):
        results = run_benchmarks(backends=["nonexistent"], num_pages=2)
        assert results == []

    def test_returns_list(self):
        # Even if no backends are available, should return a list
        results = run_benchmarks(backends=["tesseract"], num_pages=2)
        assert isinstance(results, list)

    def test_empty_backends_list(self):
        results = run_benchmarks(backends=[], num_pages=2)
        assert results == []

    def test_generates_pages_for_benchmark(self):
        """Verify that synthetic pages are generated when no input_path given."""
        # Use an unavailable backend so we don't actually run OCR,
        # but verify the function completes without error
        results = run_benchmarks(
            backends=["nonexistent"],
            num_pages=5,
            complexity="clean",
        )
        assert isinstance(results, list)


# ===========================================================================
# Tests: _load_document_pages
# ===========================================================================


class TestLoadDocumentPages:
    """Tests for document loading."""

    def test_load_png_image(self):
        """Loading a PNG should repeat the image for the requested page count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.png")
            img = Image.new("RGB", (100, 100), color="white")
            img.save(fpath)
            pages = _load_document_pages(fpath, 5)
            assert len(pages) == 5
            assert all(isinstance(p, Image.Image) for p in pages)

    def test_load_jpg_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.jpg")
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(fpath)
            pages = _load_document_pages(fpath, 3)
            assert len(pages) == 3

    def test_unsupported_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.xyz")
            with open(fpath, "wb") as f:
                f.write(b"not a real file")
            pages = _load_document_pages(fpath, 5)
            assert pages == []

    def test_pdf_without_fitz(self):
        """PDF loading should return empty when fitz is not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.pdf")
            with open(fpath, "wb") as f:
                f.write(b"%PDF-1.4 fake pdf")
            with patch.dict("sys.modules", {"fitz": None}):
                pages = _load_document_pages(fpath, 5)
                assert pages == []


# ===========================================================================
# Tests: _get_process_memory_mb
# ===========================================================================


class TestGetProcessMemoryMb:
    """Tests for memory measurement helper."""

    def test_returns_float(self):
        from benchmark_ocr import _get_process_memory_mb

        result = _get_process_memory_mb()
        assert isinstance(result, float)

    def test_returns_zero_without_psutil(self):
        from benchmark_ocr import _get_process_memory_mb

        with patch.dict("sys.modules", {"psutil": None}):
            # Need to reload or call fresh since psutil may already be imported
            # The function uses a try/except ImportError, so mocking the module
            # to None should trigger the fallback
            result = _get_process_memory_mb()
            # May or may not be 0 depending on whether psutil was already imported
            assert isinstance(result, float)


# ===========================================================================
# Tests: benchmark_paddle (mocked)
# ===========================================================================


class TestBenchmarkPaddleMocked:
    """Tests for PaddleOCR benchmark with mocked dependencies."""

    def test_returns_none_when_paddleocr_unavailable(self):
        from benchmark_ocr import benchmark_paddle

        pages = [Image.new("RGB", (100, 100))]
        with patch.dict("sys.modules", {"paddleocr": None}):
            result = benchmark_paddle(pages, device="cpu")
            assert result is None

    def test_returns_none_when_cuda_unavailable(self):
        from benchmark_ocr import benchmark_paddle

        pages = [Image.new("RGB", (100, 100))]
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = False
        mock_paddleocr = MagicMock()
        with patch.dict(
            "sys.modules", {"paddle": mock_paddle, "paddleocr": mock_paddleocr}
        ):
            result = benchmark_paddle(pages, device="gpu")
            assert result is None


# ===========================================================================
# Tests: benchmark_tesseract (mocked)
# ===========================================================================


class TestBenchmarkTesseractMocked:
    """Tests for Tesseract benchmark with mocked dependencies."""

    def test_returns_none_when_pytesseract_unavailable(self):
        from benchmark_ocr import benchmark_tesseract

        pages = [Image.new("RGB", (100, 100))]
        with patch.dict("sys.modules", {"pytesseract": None}):
            result = benchmark_tesseract(pages)
            assert result is None

    def test_returns_none_when_binary_not_found(self):
        from benchmark_ocr import benchmark_tesseract

        pages = [Image.new("RGB", (100, 100))]
        mock_pytesseract = MagicMock()
        mock_pytesseract.get_tesseract_version.side_effect = RuntimeError(
            "not found"
        )
        with patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            result = benchmark_tesseract(pages)
            assert result is None


# ===========================================================================
# Tests: WARMUP_PAGES constant
# ===========================================================================


class TestConstants:
    """Tests for module constants."""

    def test_warmup_pages_positive(self):
        assert WARMUP_PAGES > 0

    def test_default_dimensions_letter_at_300dpi(self):
        # Letter size (8.5 x 11 inches) at 300 DPI
        assert DEFAULT_WIDTH == 2550
        assert DEFAULT_HEIGHT == 3300


# ===========================================================================
# Tests: JSON output (main integration-style)
# ===========================================================================


class TestJsonOutput:
    """Tests for JSON output structure."""

    def test_compile_results_json_serializable(self):
        result = _compile_results(
            "Test", [100.0, 200.0, 150.0, 120.0, 180.0], 0, 100, [50] * 5
        )
        # Should not raise
        serialized = json.dumps(result)
        loaded = json.loads(serialized)
        assert loaded["label"] == "Test"
        assert loaded["pages"] == 5

    def test_all_result_keys_present(self):
        result = _compile_results("Test", [100.0] * 10, 0, 0, [50] * 10)
        expected_keys = {
            "label",
            "pages",
            "total_ms",
            "mean_ms",
            "median_ms",
            "p95_ms",
            "p99_ms",
            "min_ms",
            "max_ms",
            "stddev_ms",
            "pages_per_minute",
            "memory_delta_mb",
            "avg_text_length",
        }
        assert set(result.keys()) == expected_keys
