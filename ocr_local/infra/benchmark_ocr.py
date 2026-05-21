"""OCR inference benchmark tool.

Measures real OCR performance across different backends and engines
to help users make data-driven CPU vs GPU deployment decisions.

Usage:
    python benchmark_ocr.py                        # Auto-detect and benchmark all available backends
    python benchmark_ocr.py --backends paddle-gpu   # Benchmark specific backends
    python benchmark_ocr.py --pages 50              # Process 50 pages per backend
    python benchmark_ocr.py --input sample.pdf      # Use a real document
    python benchmark_ocr.py --output results.json   # Save structured results
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default benchmark parameters
DEFAULT_PAGES = 20
DEFAULT_WIDTH = 2550  # Letter size at 300 DPI
DEFAULT_HEIGHT = 3300
WARMUP_PAGES = 3


def generate_test_page(
    width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, complexity="mixed", seed=None,
    language="en",
):
    """Generate a synthetic document page for benchmarking.

    Parameters
    ----------
    width, height : int
        Page dimensions in pixels.
    complexity : str
        "clean" -- crisp text on white background
        "degraded" -- noisy, low-contrast text
        "mixed" -- random mix of clean and degraded
    seed : int, optional
        Random seed for reproducibility.
    language : str
        Language code controlling the character set overlaid on the page.
        Supported: "en" (ASCII), "de" (Latin + umlauts), "ru" (Cyrillic),
        "zh" (CJK), "ar" (Arabic).

    Returns
    -------
    PIL.Image.Image
        Generated page image (RGB).
    """
    rng = np.random.RandomState(seed)

    if complexity == "mixed":
        complexity = rng.choice(["clean", "degraded"])

    # Compute safe margins based on actual dimensions
    margin = min(150, height // 4, width // 4)
    max_x_start = max(1, min(300, width // 4))

    if complexity == "clean":
        # White background, dark text-like patterns
        img = np.ones((height, width, 3), dtype=np.uint8) * 245
        # Simulate text lines with varying line lengths
        y = margin
        while y < height - margin:
            x_start = rng.randint(max(1, max_x_start // 2), max_x_start + 1)
            max_line = max(1, width - x_start - 10)
            line_width = rng.randint(
                max(1, max_line // 3), max(2, max_line)
            )
            char_height = rng.randint(10, 16)
            # Clamp to image bounds
            end_x = min(x_start + line_width, width)
            end_y = min(y + char_height, height)
            actual_w = end_x - x_start
            actual_h = end_y - y
            if actual_w <= 0 or actual_h <= 0:
                y += rng.randint(20, 45)
                continue
            # Draw text-like horizontal band
            img[y:end_y, x_start:end_x] = rng.randint(
                10, 50, (actual_h, actual_w, 3)
            )
            # Add inter-character gaps
            num_gaps = rng.randint(5, 20)
            for _ in range(num_gaps):
                gap_x = rng.randint(x_start, max(x_start + 1, end_x - 10))
                gap_w = rng.randint(3, 15)
                gap_end = min(gap_x + gap_w, end_x)
                img[y:end_y, gap_x:gap_end] = 245
            y += rng.randint(20, 45)
    else:
        # Degraded: noisy background, faded text
        bg_noise = rng.randint(180, 230, (height, width, 3), dtype=np.uint8)
        img = bg_noise
        y = margin
        while y < height - margin:
            x_start = rng.randint(max(1, max_x_start // 2), max_x_start + 1)
            max_line = max(1, width - x_start - 10)
            line_width = rng.randint(
                max(1, max_line // 4), max(2, max_line)
            )
            char_height = rng.randint(10, 18)
            text_color = rng.randint(60, 120)
            end_x = min(x_start + line_width, width)
            end_y = min(y + char_height, height)
            if end_x > x_start and end_y > y:
                img[y:end_y, x_start:end_x] = text_color
            y += rng.randint(25, 50)
        # Add salt-and-pepper noise
        noise_mask = rng.random((height, width)) < 0.005
        img[noise_mask] = rng.choice([0, 255], size=noise_mask.sum())[:, np.newaxis]

    pil_img = Image.fromarray(img, "RGB")

    # Overlay language-specific text using PIL ImageDraw
    if language != "en":
        _overlay_language_text(pil_img, language, rng)

    return pil_img


# Character sets for corpus diversity benchmarking
_LANGUAGE_CHARSETS = {
    "en": "The quick brown fox jumps over the lazy dog. 0123456789",
    "de": "Ää Öö Üü ß Straße Größe Ärger Übung Lösung Würde",
    "ru": "Привет мир Быстрая коричневая лиса перепрыгнула через ленивую собаку",
    "zh": "你好世界 快速的棕色狐狸跳过懒惰的狗 人工智能文字识别",
    "ar": "مرحبا بالعالم الثعلب البني السريع يقفز فوق الكلب الكسول",
}


def _overlay_language_text(pil_img, language, rng):
    """Overlay Unicode text on the page image using PIL ImageDraw.

    Uses the default font (no external font files needed).  The text may
    render as boxes on systems without the appropriate font, but the
    pixel-level data is still useful for OCR benchmarking.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(pil_img)
    charset = _LANGUAGE_CHARSETS.get(language, _LANGUAGE_CHARSETS["en"])
    words = charset.split()
    y = 150
    while y < pil_img.height - 150:
        # Build a line from random words
        n_words = rng.randint(3, min(8, len(words) + 1))
        indices = rng.randint(0, len(words), size=n_words)
        line = " ".join(words[i] for i in indices)
        x = rng.randint(50, max(51, min(300, pil_img.width // 4)))
        draw.text((x, y), line, fill=(30, 30, 30))
        y += rng.randint(30, 55)


def _get_process_memory_mb():
    """Get current process memory usage in MB."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def benchmark_paddle(pages, device="gpu", use_onnx=False, lang="en"):
    """Benchmark PaddleOCR inference.

    Parameters
    ----------
    pages : list[PIL.Image.Image]
        Test page images to process.
    device : str
        "gpu" or "cpu".
    use_onnx : bool
        If True, enable ONNX Runtime inference (CPU only, PaddleOCR 2.x).
    lang : str
        Language code for PaddleOCR.

    Returns
    -------
    dict or None
        Timing results, or None if the backend is unavailable.
    """
    backend_name = "paddle"
    if use_onnx:
        backend_name = "onnx"

    label = f"PaddleOCR ({backend_name}, {device})"
    logger.info("Benchmarking %s with %d pages...", label, len(pages))

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.warning("PaddleOCR not available, skipping %s", label)
        return None

    # Check GPU availability
    if device == "gpu":
        try:
            import paddle

            if not paddle.device.is_compiled_with_cuda():
                logger.warning("CUDA not available, skipping GPU benchmark")
                return None
        except ImportError:
            logger.warning("PaddlePaddle not available, skipping GPU benchmark")
            return None

    kwargs = {
        "use_angle_cls": True,
        "lang": lang,
        "use_gpu": (device == "gpu"),
        "show_log": False,
    }
    if use_onnx and device == "cpu":
        kwargs["use_onnx"] = True

    try:
        engine = PaddleOCR(**kwargs)
    except TypeError:
        # Older PaddleOCR versions may not support use_onnx
        kwargs.pop("use_onnx", None)
        engine = PaddleOCR(**kwargs)

    # Warmup
    logger.info("  Warming up (%d pages)...", WARMUP_PAGES)
    for i in range(min(WARMUP_PAGES, len(pages))):
        engine.ocr(np.array(pages[i]))

    # Benchmark
    mem_before = _get_process_memory_mb()
    timings = []
    text_lengths = []

    for i, page in enumerate(pages):
        img_np = np.array(page)
        start = time.perf_counter()
        result = engine.ocr(img_np)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)

        # Extract text length from PaddleOCR 2.x result format
        # result is list of pages, each page is list of (box, (text, conf))
        text_len = 0
        if result:
            for page_result in result:
                if page_result:
                    for line in page_result:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                                text_len += len(str(text_info[0]))
        text_lengths.append(text_len)

        if (i + 1) % 10 == 0:
            logger.info("  Processed %d/%d pages", i + 1, len(pages))

    mem_after = _get_process_memory_mb()

    return _compile_results(label, timings, mem_before, mem_after, text_lengths)


def benchmark_tesseract(pages, lang="eng"):
    """Benchmark Tesseract OCR inference.

    Parameters
    ----------
    pages : list[PIL.Image.Image]
        Test page images to process.
    lang : str
        Language code for Tesseract.

    Returns
    -------
    dict or None
        Timing results, or None if Tesseract is unavailable.
    """
    label = "Tesseract 5 (CPU)"
    logger.info("Benchmarking %s with %d pages...", label, len(pages))

    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not available, skipping Tesseract benchmark")
        return None

    # Verify Tesseract is installed
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        logger.warning("Tesseract binary not found, skipping Tesseract benchmark")
        return None

    # Warmup
    logger.info("  Warming up (%d pages)...", WARMUP_PAGES)
    for i in range(min(WARMUP_PAGES, len(pages))):
        pytesseract.image_to_string(pages[i], lang=lang)

    # Benchmark
    mem_before = _get_process_memory_mb()
    timings = []
    text_lengths = []

    for i, page in enumerate(pages):
        start = time.perf_counter()
        text = pytesseract.image_to_string(page, lang=lang)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)
        text_lengths.append(len(text))

        if (i + 1) % 10 == 0:
            logger.info("  Processed %d/%d pages", i + 1, len(pages))

    mem_after = _get_process_memory_mb()

    return _compile_results(label, timings, mem_before, mem_after, text_lengths)


def _compile_results(label, timings, mem_before, mem_after, text_lengths):
    """Compile benchmark timing data into a results dict.

    Parameters
    ----------
    label : str
        Human-readable backend label.
    timings : list[float]
        Per-page timing in milliseconds.
    mem_before, mem_after : float
        Process memory usage before/after benchmark (MB).
    text_lengths : list[int]
        Character count extracted per page.

    Returns
    -------
    dict or None
        Compiled results, or None if no timings were recorded.
    """
    if not timings:
        return None

    total_ms = sum(timings)
    pages_per_minute = (len(timings) / (total_ms / 1000)) * 60 if total_ms > 0 else 0

    return {
        "label": label,
        "pages": len(timings),
        "total_ms": round(total_ms, 1),
        "mean_ms": round(statistics.mean(timings), 1),
        "median_ms": round(statistics.median(timings), 1),
        "p95_ms": round(
            sorted(timings)[int(len(timings) * 0.95)], 1
        )
        if len(timings) >= 5
        else round(max(timings), 1),
        "p99_ms": round(
            sorted(timings)[int(len(timings) * 0.99)], 1
        )
        if len(timings) >= 10
        else round(max(timings), 1),
        "min_ms": round(min(timings), 1),
        "max_ms": round(max(timings), 1),
        "stddev_ms": round(statistics.stdev(timings), 1)
        if len(timings) >= 2
        else 0.0,
        "pages_per_minute": round(pages_per_minute, 1),
        "memory_delta_mb": round(mem_after - mem_before, 1),
        "avg_text_length": round(statistics.mean(text_lengths), 0)
        if text_lengths
        else 0,
    }


def format_comparison_table(results):
    """Format benchmark results as a human-readable comparison table.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from benchmark functions.

    Returns
    -------
    str
        Formatted table string.
    """
    if not results:
        return "No benchmark results to display."

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("OCR BENCHMARK RESULTS")
    lines.append("=" * 90)
    lines.append("")

    # Header
    header = (
        f"{'Backend':<30} {'Mean(ms)':>10} {'P95(ms)':>10} "
        f"{'PPM':>10} {'Mem(MB)':>10}"
    )
    lines.append(header)
    lines.append("-" * 90)

    # Sort by pages_per_minute descending
    sorted_results = sorted(
        results, key=lambda r: r["pages_per_minute"], reverse=True
    )
    fastest_ppm = sorted_results[0]["pages_per_minute"] if sorted_results else 1

    for r in sorted_results:
        _ratio = r["pages_per_minute"] / fastest_ppm if fastest_ppm > 0 else 0
        row = (
            f"{r['label']:<30} "
            f"{r['mean_ms']:>10.1f} "
            f"{r['p95_ms']:>10.1f} "
            f"{r['pages_per_minute']:>10.1f} "
            f"{r['memory_delta_mb']:>10.1f}"
        )
        lines.append(row)

    lines.append("-" * 90)
    lines.append("")

    # Cost analysis
    lines.append("COST ANALYSIS (estimated)")
    lines.append("-" * 90)

    lines.append(
        f"{'Configuration':<35} {'$/Hour':>8} {'$/1K Pages':>12} {'Pages/Day':>12}"
    )
    lines.append("-" * 90)

    for r in sorted_results:
        ppm = r["pages_per_minute"]
        if ppm <= 0:
            continue
        pages_per_hour = ppm * 60

        # Match backend to likely instance type
        if "gpu" in r["label"].lower():
            cost_hr = 0.526
        else:
            cost_hr = 0.170

        cost_per_1k = (cost_hr / pages_per_hour) * 1000 if pages_per_hour > 0 else 0
        pages_per_day = pages_per_hour * 24

        lines.append(
            f"{r['label']:<35} "
            f"${cost_hr:>7.3f} "
            f"${cost_per_1k:>11.4f} "
            f"{pages_per_day:>12,.0f}"
        )

    lines.append("-" * 90)
    lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 90)

    gpu_results = [r for r in sorted_results if "gpu" in r["label"].lower()]
    cpu_results = [r for r in sorted_results if "gpu" not in r["label"].lower()]

    if gpu_results and cpu_results:
        best_gpu = gpu_results[0]
        best_cpu = cpu_results[0]
        speedup = (
            best_gpu["pages_per_minute"] / best_cpu["pages_per_minute"]
            if best_cpu["pages_per_minute"] > 0
            else float("inf")
        )
        cpu_workers_needed = max(1, round(speedup))

        lines.append(
            f"  Best GPU backend: {best_gpu['label']} "
            f"({best_gpu['pages_per_minute']:.0f} PPM)"
        )
        lines.append(
            f"  Best CPU backend: {best_cpu['label']} "
            f"({best_cpu['pages_per_minute']:.0f} PPM)"
        )
        lines.append(f"  GPU speedup: {speedup:.1f}x")
        lines.append(f"  CPU workers needed to match GPU: ~{cpu_workers_needed}")
        lines.append("")

        if speedup <= 3:
            lines.append(
                "  -> CPU-only deployment is viable for most workloads"
            )
        elif speedup <= 6:
            lines.append(
                "  -> Consider hybrid deployment "
                "(GPU for complex docs, CPU for clean)"
            )
        else:
            lines.append(
                "  -> GPU recommended for high-throughput requirements"
            )
    elif cpu_results:
        best_cpu = cpu_results[0]
        lines.append(
            f"  Best CPU backend: {best_cpu['label']} "
            f"({best_cpu['pages_per_minute']:.0f} PPM)"
        )
        lines.append("  No GPU available for comparison")
        lines.append(
            "  -> Use ONNX Runtime or OpenVINO for best CPU performance"
        )

    lines.append("")
    return "\n".join(lines)


def run_benchmarks(
    backends=None,
    num_pages=DEFAULT_PAGES,
    complexity="mixed",
    lang="en",
    input_path=None,
    languages=None,
):
    """Run OCR benchmarks across specified backends.

    Parameters
    ----------
    backends : list[str], optional
        List of backends to benchmark. Default: auto-detect available.
        Options: "paddle-gpu", "paddle-cpu", "onnx", "tesseract"
    num_pages : int
        Number of pages to process per backend.
    complexity : str
        Page complexity: "clean", "degraded", "mixed"
    lang : str
        Language code for OCR engines.
    input_path : str, optional
        Path to a real document to use instead of synthetic pages.
    languages : list[str], optional
        List of language codes for corpus diversity.  Pages are
        distributed round-robin across languages.  Default: ``["en"]``.

    Returns
    -------
    list[dict]
        Benchmark results for each backend.
    """
    if languages is None:
        languages = ["en"]

    # Generate or load test pages
    if input_path:
        pages = _load_document_pages(input_path, num_pages)
    else:
        logger.info(
            "Generating %d synthetic %s test pages (languages=%s)...",
            num_pages, complexity, ",".join(languages),
        )
        pages = [
            generate_test_page(
                complexity=complexity, seed=i,
                language=languages[i % len(languages)],
            )
            for i in range(num_pages)
        ]

    if not pages:
        logger.error("No test pages available")
        return []

    # Auto-detect available backends
    if backends is None:
        backends = _detect_available_backends()

    results = []

    for backend in backends:
        result = None
        if backend == "paddle-gpu":
            result = benchmark_paddle(pages, device="gpu", lang=lang)
        elif backend == "paddle-cpu":
            result = benchmark_paddle(pages, device="cpu", lang=lang)
        elif backend == "onnx":
            result = benchmark_paddle(pages, device="cpu", use_onnx=True, lang=lang)
        elif backend == "tesseract":
            tess_lang = "eng" if lang == "en" else lang
            result = benchmark_tesseract(pages, lang=tess_lang)
        else:
            logger.warning("Unknown backend: %s", backend)

        if result:
            results.append(result)

    return results


def _detect_available_backends():
    """Detect which OCR backends are available on this system.

    Returns
    -------
    list[str]
        List of available backend identifiers.
    """
    backends = []

    # Check PaddleOCR + GPU
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            backends.append("paddle-gpu")
    except Exception:
        pass

    # Check PaddleOCR CPU
    try:
        import paddleocr  # noqa: F401

        backends.append("paddle-cpu")
    except Exception:
        pass

    # Check ONNX Runtime
    try:
        import onnxruntime  # noqa: F401
        import paddleocr  # noqa: F401

        backends.append("onnx")
    except Exception:
        pass

    # Check Tesseract
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        backends.append("tesseract")
    except Exception:
        pass

    if not backends:
        logger.warning("No OCR backends detected!")
    else:
        logger.info("Detected backends: %s", ", ".join(backends))

    return backends


def _load_document_pages(path, max_pages):
    """Load pages from a real document file.

    Parameters
    ----------
    path : str
        Path to the document file (PDF or image).
    max_pages : int
        Maximum number of pages to load.

    Returns
    -------
    list[PIL.Image.Image]
        Loaded page images.
    """
    path = Path(path)
    pages = []

    if path.suffix.lower() == ".pdf":
        try:
            import fitz

            doc = fitz.open(str(path))
            for i, pg in enumerate(doc):
                if i >= max_pages:
                    break
                pix = pg.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(img)
            doc.close()
        except ImportError:
            logger.error(
                "PyMuPDF (fitz) required for PDF input. "
                "Install with: pip install PyMuPDF"
            )
    elif path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
        img = Image.open(str(path)).convert("RGB")
        pages = [img] * max_pages  # Repeat same image for benchmark
    else:
        logger.error("Unsupported file format: %s", path.suffix)

    return pages


def main():
    """CLI entry point for the OCR benchmark tool."""
    parser = argparse.ArgumentParser(
        description="OCR Inference Benchmark -- Compare CPU vs GPU performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_ocr.py                          # Auto-detect and benchmark all
  python benchmark_ocr.py --backends paddle-cpu onnx tesseract
  python benchmark_ocr.py --pages 50 --complexity clean
  python benchmark_ocr.py --input documents/sample.pdf --output results.json
        """,
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["paddle-gpu", "paddle-cpu", "onnx", "tesseract"],
        help="Backends to benchmark (default: auto-detect)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=DEFAULT_PAGES,
        help=f"Number of pages per backend (default: {DEFAULT_PAGES})",
    )
    parser.add_argument(
        "--complexity",
        choices=["clean", "degraded", "mixed"],
        default="mixed",
        help="Synthetic page complexity (default: mixed)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="OCR language code (default: en)",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to a real document (PDF or image) to benchmark with",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Save structured results to JSON file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en",
        help="Comma-separated language codes for corpus diversity (default: en). "
             "Supported: en, de, ru, zh, ar",
    )

    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    results = run_benchmarks(
        backends=args.backends,
        num_pages=args.pages,
        complexity=args.complexity,
        lang=args.lang,
        input_path=args.input_path,
        languages=languages,
    )

    # Display results
    table = format_comparison_table(results)
    print(table)

    # Build corpus_info
    corpus_info = {
        "languages": languages,
        "formats": [args.complexity],
        "page_count": args.pages,
    }

    # Save JSON if requested
    if args.output_path:
        output = {
            "benchmark_config": {
                "pages": args.pages,
                "complexity": args.complexity,
                "lang": args.lang,
                "input": args.input_path,
                "languages": languages,
            },
            "corpus_info": corpus_info,
            "results": results,
        }
        with open(args.output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", args.output_path)

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
