"""Performance benchmark framework for OCR pipeline comparison.

Measures throughput, accuracy, and cost metrics to compare the OCR pipeline
against commercial alternatives (ABBYY, Azure Document Intelligence, Google
Document AI).

Usage:
    python scripts/benchmark_comparison.py --suite standard --output-dir benchmark_results/
    python scripts/benchmark_comparison.py --suite accuracy --ground-truth data/ground_truth/
    python scripts/benchmark_comparison.py --report benchmark_results/ --format markdown
"""

import argparse
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_PIPELINE_SCRIPT = _PROJECT_ROOT / "ocr_gpu_async.py"
_SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}


class BenchmarkSuite(Enum):
    STANDARD = "standard"  # Throughput + latency
    ACCURACY = "accuracy"  # Character/word accuracy vs ground truth
    STRESS = "stress"  # High volume sustained throughput
    COLD_START = "cold_start"  # Startup time measurement


class DocumentCategory(Enum):
    PRINTED_TEXT = "printed_text"
    HANDWRITTEN = "handwritten"
    MIXED_LAYOUT = "mixed_layout"
    TABLE_HEAVY = "table_heavy"
    SCANNED_LOW_QUALITY = "scanned_low_quality"
    MULTILINGUAL = "multilingual"
    LARGE_DOCUMENT = "large_document"


@dataclass
class PageMetrics:
    """Per-page performance metrics."""

    page_number: int
    processing_time_ms: float
    char_count: int
    word_count: int
    confidence: float = 0.0
    dpi: int = 300
    engine: str = "paddle"


@dataclass
class DocumentMetrics:
    """Per-document aggregate metrics."""

    document_id: str
    filename: str
    category: str
    total_pages: int
    total_processing_time_ms: float
    pages: list = field(default_factory=list)
    total_chars: int = 0
    total_words: int = 0
    avg_confidence: float = 0.0
    throughput_pages_per_minute: float = 0.0
    memory_peak_mb: float = 0.0

    def compute_aggregates(self):
        if self.pages:
            self.total_chars = sum(p.char_count for p in self.pages)
            self.total_words = sum(p.word_count for p in self.pages)
            confs = [p.confidence for p in self.pages if p.confidence > 0]
            self.avg_confidence = statistics.mean(confs) if confs else 0.0
            if self.total_processing_time_ms > 0:
                minutes = self.total_processing_time_ms / 60000
                self.throughput_pages_per_minute = (
                    self.total_pages / minutes if minutes > 0 else 0
                )


@dataclass
class AccuracyMetrics:
    """Accuracy comparison against ground truth."""

    document_id: str
    character_accuracy: float = 0.0  # 1 - CER
    word_accuracy: float = 0.0  # 1 - WER
    total_characters: int = 0
    total_words: int = 0
    character_errors: int = 0
    word_errors: int = 0

    @property
    def cer(self) -> float:
        """Character Error Rate."""
        return (
            self.character_errors / self.total_characters
            if self.total_characters > 0
            else 1.0
        )

    @property
    def wer(self) -> float:
        """Word Error Rate."""
        return (
            self.word_errors / self.total_words if self.total_words > 0 else 1.0
        )


@dataclass
class BenchmarkResult:
    """Complete benchmark run result."""

    suite: str
    timestamp: str = ""
    pipeline_version: str = ""
    system_info: dict = field(default_factory=dict)
    documents: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def compute_summary(self):
        if not self.documents:
            return

        all_times = [d.total_processing_time_ms for d in self.documents]
        all_pages = sum(d.total_pages for d in self.documents)
        total_time = sum(all_times)
        confidence_values = [
            d.avg_confidence for d in self.documents if d.avg_confidence > 0
        ]

        self.summary = {
            "total_documents": len(self.documents),
            "total_pages": all_pages,
            "total_time_ms": total_time,
            "avg_time_per_doc_ms": (
                statistics.mean(all_times) if all_times else 0
            ),
            "median_time_per_doc_ms": (
                statistics.median(all_times) if all_times else 0
            ),
            "p95_time_per_doc_ms": (
                _percentile(all_times, 95) if all_times else 0
            ),
            "p99_time_per_doc_ms": (
                _percentile(all_times, 99) if all_times else 0
            ),
            "throughput_pages_per_minute": (
                (all_pages / (total_time / 60000)) if total_time > 0 else 0
            ),
            "avg_confidence": (
                statistics.mean(confidence_values) if confidence_values else 0
            ),
        }

        if self.accuracy:
            self.summary["avg_character_accuracy"] = statistics.mean(
                a.character_accuracy for a in self.accuracy
            )
            self.summary["avg_word_accuracy"] = statistics.mean(
                a.word_accuracy for a in self.accuracy
            )
            self.summary["avg_cer"] = statistics.mean(
                a.cer for a in self.accuracy
            )
            self.summary["avg_wer"] = statistics.mean(
                a.wer for a in self.accuracy
            )


def _percentile(data: list, pct: float) -> float:
    """Calculate percentile value from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


def compute_edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein edit distance between two strings."""
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    return prev[m]


def compute_accuracy(reference: str, hypothesis: str) -> AccuracyMetrics:
    """Compute character and word accuracy metrics."""
    char_errors = compute_edit_distance(reference, hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()
    word_edit_dist = compute_edit_distance(
        "\n".join(ref_words), "\n".join(hyp_words)
    )

    total_chars = max(len(reference), 1)
    total_words = max(len(ref_words), 1)

    return AccuracyMetrics(
        document_id="",
        character_accuracy=max(0.0, 1.0 - char_errors / total_chars),
        word_accuracy=max(0.0, 1.0 - word_edit_dist / total_words),
        total_characters=len(reference),
        total_words=len(ref_words),
        character_errors=char_errors,
        word_errors=word_edit_dist,
    )


def generate_comparison_table(
    our_result: BenchmarkResult, competitor_data: dict | None = None
) -> str:
    """Generate markdown comparison table."""
    lines = [
        "# OCR Performance Benchmark Comparison",
        "",
        "| Metric | EDCOCR |",
        "|--------|----------|",
    ]

    s = our_result.summary
    lines.append(f"| Total Documents | {s.get('total_documents', 0)} |")
    lines.append(f"| Total Pages | {s.get('total_pages', 0)} |")
    lines.append(
        f"| Avg Time/Doc (ms) | {s.get('avg_time_per_doc_ms', 0):.1f} |"
    )
    lines.append(
        f"| Median Time/Doc (ms) | {s.get('median_time_per_doc_ms', 0):.1f} |"
    )
    lines.append(
        f"| P95 Time/Doc (ms) | {s.get('p95_time_per_doc_ms', 0):.1f} |"
    )
    lines.append(
        f"| P99 Time/Doc (ms) | {s.get('p99_time_per_doc_ms', 0):.1f} |"
    )
    lines.append(
        f"| Throughput (pages/min) | {s.get('throughput_pages_per_minute', 0):.1f} |"
    )
    lines.append(
        f"| Avg Confidence | {s.get('avg_confidence', 0):.3f} |"
    )

    if "avg_cer" in s:
        lines.append(f"| Avg CER | {s.get('avg_cer', 0):.4f} |")
        lines.append(f"| Avg WER | {s.get('avg_wer', 0):.4f} |")
        lines.append(
            f"| Avg Char Accuracy | {s.get('avg_character_accuracy', 0):.3f} |"
        )
        lines.append(
            f"| Avg Word Accuracy | {s.get('avg_word_accuracy', 0):.3f} |"
        )

    lines.append("")
    lines.append(f"*Pipeline Version: {our_result.pipeline_version}*")
    lines.append(f"*Suite: {our_result.suite}*")

    return "\n".join(lines)


def save_result(result: BenchmarkResult, output_dir: Path):
    """Save benchmark result to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = result.timestamp or time.strftime("%Y%m%d_%H%M%S")

    data = {
        "suite": result.suite,
        "timestamp": result.timestamp,
        "pipeline_version": result.pipeline_version,
        "system_info": result.system_info,
        "summary": result.summary,
        "documents": [asdict(d) for d in result.documents],
        "accuracy": [asdict(a) for a in result.accuracy],
    }

    json_path = output_dir / f"benchmark_{result.suite}_{timestamp}.json"
    json_path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Saved benchmark result to %s", json_path)
    return json_path


def load_result(path: Path) -> BenchmarkResult:
    """Load benchmark result from JSON file."""
    data = json.loads(path.read_text())
    result = BenchmarkResult(
        suite=data.get("suite", ""),
        timestamp=data.get("timestamp", ""),
        pipeline_version=data.get("pipeline_version", ""),
        system_info=data.get("system_info", {}),
        summary=data.get("summary", {}),
    )
    for doc_data in data.get("documents", []):
        pages = [PageMetrics(**p) for p in doc_data.pop("pages", [])]
        doc = DocumentMetrics(**doc_data)
        doc.pages = pages
        result.documents.append(doc)
    for acc_data in data.get("accuracy", []):
        result.accuracy.append(AccuracyMetrics(**acc_data))
    return result


def generate_report(result_dir: Path, fmt: str = "markdown") -> str:
    """Generate report from saved benchmark results."""
    results = []
    for f in sorted(result_dir.glob("benchmark_*.json")):
        results.append(load_result(f))

    if not results:
        return "No benchmark results found."

    latest = results[-1]
    if fmt == "json":
        return json.dumps(
            {
                "suite": latest.suite,
                "timestamp": latest.timestamp,
                "pipeline_version": latest.pipeline_version,
                "system_info": latest.system_info,
                "summary": latest.summary,
                "documents": [asdict(d) for d in latest.documents],
                "accuracy": [asdict(a) for a in latest.accuracy],
            },
            indent=2,
        )
    return generate_comparison_table(latest)


def discover_documents(input_dir: Path) -> list[Path]:
    """Return supported benchmark documents beneath *input_dir*."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    documents = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_DOCUMENT_EXTENSIONS
    )
    if not documents:
        raise ValueError(
            f"No benchmark documents found in {input_dir} with supported extensions "
            f"{sorted(_SUPPORTED_DOCUMENT_EXTENSIONS)}"
        )
    return documents


def _sanitize_path_segment(segment: str) -> str:
    """Return a filesystem-safe path segment."""
    sanitized = []
    for ch in segment:
        if ch in '<>:"|?*\x00' or ord(ch) < 32:
            sanitized.append("_")
        else:
            sanitized.append(ch)
    return "".join(sanitized).strip(". ")


def build_output_rel_stem(document_path: Path, source_dir: Path) -> Path:
    """Match the pipeline's output naming for a source document."""
    try:
        rel_path = document_path.relative_to(source_dir)
    except ValueError:
        rel_path = Path(document_path.name)

    clean_parts = [_sanitize_path_segment(part) for part in rel_path.parts]
    clean_rel_path = Path(*clean_parts) if clean_parts else Path(document_path.name)
    rel_stem = clean_rel_path.with_suffix("")
    if document_path.suffix.lower() == ".pdf":
        return rel_stem

    ext_token = "".join(
        ch if ch.isalnum() else "_" for ch in document_path.suffix.lower().lstrip(".")
    )
    if not ext_token:
        ext_token = "img"
    return rel_stem.parent / f"{rel_stem.name}__{ext_token}"


def get_text_output_path(
    document_path: Path, source_dir: Path, output_dir: Path
) -> Path:
    """Return the expected plain-text artifact path for *document_path*."""
    rel_stem = build_output_rel_stem(document_path, source_dir)
    return output_dir / "EXPORT" / "TEXT" / rel_stem.parent / f"{rel_stem.name}.txt"


def _copy_document_for_run(
    document_path: Path, input_dir: Path, run_source_dir: Path
) -> Path:
    """Mirror a benchmark document into an isolated run source directory."""
    relative_path = document_path.relative_to(input_dir)
    copied_doc = run_source_dir / relative_path
    copied_doc.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(document_path, copied_doc)
    return copied_doc


def _build_run_root(output_dir: Path, index: int, relative_path: Path) -> Path:
    """Return a safe artifact directory for a single benchmark document."""
    safe_stem = _sanitize_path_segment(relative_path.stem) or f"doc_{index:03d}"
    return output_dir / "runs" / f"{index:03d}_{safe_stem}"


def _resolve_ground_truth_path(
    document_path: Path, input_dir: Path, ground_truth_dir: Path
) -> Path:
    """Resolve the matching ground-truth text file for *document_path*."""
    relative_path = document_path.relative_to(input_dir)
    candidates = [
        ground_truth_dir / relative_path.with_suffix(".txt"),
        ground_truth_dir / relative_path.with_suffix(".gt.txt"),
        ground_truth_dir / relative_path.parent / f"{relative_path.stem}.txt",
        ground_truth_dir / relative_path.parent / f"{relative_path.stem}.gt.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No ground-truth text found for {document_path} under {ground_truth_dir}"
    )


def _estimate_page_count(document_path: Path) -> int:
    """Estimate page count, using PDF metadata when available."""
    if document_path.suffix.lower() != ".pdf":
        return 1

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return 1

    try:
        with fitz.open(document_path) as pdf:
            return max(len(pdf), 1)
    except Exception:
        logger.warning("Failed to inspect page count for %s", document_path)
        return 1


def _distribute_count(total: int, buckets: int) -> list[int]:
    """Split *total* as evenly as possible across *buckets*."""
    if buckets <= 0:
        return []
    base, remainder = divmod(max(total, 0), buckets)
    return [base + (1 if idx < remainder else 0) for idx in range(buckets)]


def _build_page_metrics(
    total_pages: int, processing_time_ms: float, text: str
) -> list[PageMetrics]:
    """Create evenly distributed page metrics from document-level outputs."""
    page_count = max(total_pages, 1)
    char_counts = _distribute_count(len(text), page_count)
    word_counts = _distribute_count(len(text.split()), page_count)
    per_page_time = processing_time_ms / page_count if page_count else 0.0
    return [
        PageMetrics(
            page_number=index + 1,
            processing_time_ms=per_page_time,
            char_count=char_counts[index],
            word_count=word_counts[index],
        )
        for index in range(page_count)
    ]


def run_pipeline_for_document(
    document_path: Path,
    source_dir: Path,
    output_dir: Path,
    pipeline_script: Path,
) -> float:
    """Run the OCR pipeline for a single benchmark document and return elapsed ms."""
    env = os.environ.copy()
    env["SOURCE_FOLDER"] = str(source_dir)
    env["OCR_SOURCE_DIR"] = str(source_dir)
    env["OUTPUT_FOLDER"] = str(output_dir)
    env["OCR_OUTPUT_DIR"] = str(output_dir)

    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(pipeline_script)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed for {document_path} with exit code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return elapsed_ms


def execute_benchmark_suite(
    suite: BenchmarkSuite,
    input_dir: Path,
    output_dir: Path,
    *,
    ground_truth_dir: Path | None = None,
    pipeline_script: Path | None = None,
    pipeline_runner: Callable[[Path, Path, Path, Path], float] | None = None,
    enable_profiling: bool = False,
) -> BenchmarkResult:
    """Execute a real benchmark suite and return the populated result."""
    profiler = None
    if enable_profiling:
        try:
            sys.path.insert(0, str(_PROJECT_ROOT))
            from profiling import PipelineProfiler

            profiler = PipelineProfiler()
        except ImportError:
            logger.warning("profiling module not available; --profile ignored")

    documents = discover_documents(input_dir)
    pipeline_script = pipeline_script or _DEFAULT_PIPELINE_SCRIPT
    pipeline_runner = pipeline_runner or run_pipeline_for_document

    if suite == BenchmarkSuite.ACCURACY and ground_truth_dir is None:
        raise ValueError("Accuracy suite requires a ground-truth directory.")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = BenchmarkResult(
        suite=suite.value,
        timestamp=time.strftime("%Y%m%d_%H%M%S"),
        pipeline_version="unknown",
        system_info={
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "input_dir": str(input_dir),
            "pipeline_script": str(pipeline_script),
            "execution_mode": "per_document_subprocess",
        },
    )

    try:
        from ocr_local.config.version import __version__

        result.pipeline_version = __version__
    except Exception:
        logger.debug("Unable to import pipeline version", exc_info=True)

    for index, document_path in enumerate(documents, start=1):
        relative_path = document_path.relative_to(input_dir)
        run_root = _build_run_root(output_dir, index, relative_path)
        run_source_dir = run_root / "source"
        run_output_dir = run_root / "ocr_output"
        benchmark_doc = _copy_document_for_run(
            document_path, input_dir, run_source_dir
        )

        if profiler is not None:
            with profiler.stage(f"document_{index}"):
                elapsed_ms = pipeline_runner(
                    benchmark_doc,
                    run_source_dir,
                    run_output_dir,
                    pipeline_script,
                )
        else:
            elapsed_ms = pipeline_runner(
                benchmark_doc,
                run_source_dir,
                run_output_dir,
                pipeline_script,
            )
        text_output_path = get_text_output_path(
            benchmark_doc,
            run_source_dir,
            run_output_dir,
        )
        output_text = (
            text_output_path.read_text(encoding="utf-8", errors="replace")
            if text_output_path.is_file()
            else ""
        )

        total_pages = _estimate_page_count(document_path)
        doc_metrics = DocumentMetrics(
            document_id=str(relative_path.with_suffix("")).replace("\\", "/"),
            filename=document_path.name,
            category=DocumentCategory.PRINTED_TEXT.value,
            total_pages=total_pages,
            total_processing_time_ms=elapsed_ms,
            pages=_build_page_metrics(total_pages, elapsed_ms, output_text),
        )
        doc_metrics.compute_aggregates()
        result.documents.append(doc_metrics)

        if suite == BenchmarkSuite.ACCURACY and ground_truth_dir is not None:
            truth_path = _resolve_ground_truth_path(
                document_path,
                input_dir,
                ground_truth_dir,
            )
            accuracy = compute_accuracy(
                truth_path.read_text(encoding="utf-8", errors="replace"),
                output_text,
            )
            accuracy.document_id = doc_metrics.document_id
            result.accuracy.append(accuracy)

    result.compute_summary()
    result.system_info["documents_benchmarked"] = len(result.documents)
    result.system_info["artifacts_dir"] = str(output_dir / "runs")

    if profiler is not None:
        result.system_info["profiling"] = profiler.report()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="OCR pipeline performance benchmark"
    )
    parser.add_argument(
        "--suite",
        choices=[s.value for s in BenchmarkSuite],
        default="standard",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results/"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="ocr_source/",
        help="Input directory containing benchmark documents",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Ground truth directory for accuracy suite",
    )
    parser.add_argument(
        "--pipeline-script",
        type=str,
        default=str(_DEFAULT_PIPELINE_SCRIPT),
        help="Path to the OCR pipeline entry point",
    )
    parser.add_argument(
        "--report", type=str, help="Generate report from results directory"
    )
    parser.add_argument(
        "--format", choices=["markdown", "json"], default="markdown"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable pipeline profiling (per-stage timing, memory, CPU time)",
    )
    args = parser.parse_args()

    if args.report:
        report = generate_report(Path(args.report), args.format)
        print(report)
        return

    suite = BenchmarkSuite(args.suite)
    if suite == BenchmarkSuite.ACCURACY and not args.ground_truth:
        parser.error("--ground-truth is required for the accuracy suite")

    result = execute_benchmark_suite(
        suite,
        Path(args.input_dir),
        Path(args.output_dir),
        ground_truth_dir=Path(args.ground_truth) if args.ground_truth else None,
        pipeline_script=Path(args.pipeline_script),
        enable_profiling=args.profile,
    )
    json_path = save_result(result, Path(args.output_dir))
    print(
        f"Executed benchmark suite '{args.suite}' "
        f"for {len(result.documents)} documents."
    )
    print(f"Results saved to: {json_path}")
    if args.format == "json":
        print(json.dumps(result.summary, indent=2))
    else:
        print(generate_comparison_table(result))


if __name__ == "__main__":
    main()
