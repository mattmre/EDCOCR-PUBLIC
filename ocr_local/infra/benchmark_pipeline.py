"""
Performance Benchmarking Framework for EDCOCR Pipeline.

Usage:
    python benchmark_pipeline.py --mode simulate    # Mock mode (no GPU needed)
    python benchmark_pipeline.py --mode live        # Real pipeline benchmark
    python benchmark_pipeline.py --report latest    # Display latest results
    python benchmark_pipeline.py --compare run1.json run2.json

Metrics collected:
    - Pages per minute (PPM)
    - Average time per page (ms)
    - Peak memory usage (MB)
    - GPU memory usage (MB) [live mode only]
    - Queue throughput rates
    - Pipeline stage timings
"""
import argparse
import json
import os
import queue
import random
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# Optional: psutil for system metrics (graceful fallback)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from ocr_local.config.version import __version__
except ImportError:
    __version__ = "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_DIR = "benchmark_results"

# Default pipeline config matching ocr_gpu_async.py constants
DEFAULT_CONFIG = {
    "NUM_EXTRACTORS": 8,
    "NUM_WORKERS": 12,
    "NUM_COMPRESSORS": 8,
    "IMAGE_QUEUE_SIZE": 200,
    "CHUNK_QUEUE_SIZE": 50,
    "RESULT_QUEUE_SIZE": 5000,
    "COMPRESSION_QUEUE_SIZE": 5000,
    "DPI": 300,
}

# Simulate mode timing distributions (milliseconds)
# Based on documented estimates in document-intelligence-roadmap.md:
#   OCR only: ~200-500ms per page
SIMULATE_EXTRACTION_MS = (30, 80)    # PDF rasterization at 300 DPI
SIMULATE_OCR_MS = (200, 500)         # PaddleOCR PP-OCRv4
SIMULATE_ASSEMBLY_MS = (10, 40)      # Page stitching + text write
SIMULATE_COMPRESSION_MS = (50, 200)  # Ghostscript /prepress

# Simulated memory model constants
SIMULATE_BASE_MEMORY_MB = 400.0
SIMULATE_MEMORY_PER_PAGE_MB = 0.02
SIMULATE_MEMORY_JITTER_LOW = 0.8
SIMULATE_MEMORY_JITTER_HIGH = 1.2

# Supported file extensions for live benchmarking
SUPPORTED_LIVE_EXTENSIONS = {
    ".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png",
    ".bmp", ".gif", ".webp", ".jp2",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IOCounters:
    """Disk and network I/O counters captured during a benchmark run."""

    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    net_sent_mb: float = 0.0
    net_recv_mb: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Container for a single benchmark run's results."""

    run_id: str
    timestamp: str
    mode: str  # "simulate" or "live"
    pipeline_version: str
    config: dict

    # Timing
    total_duration_seconds: float = 0.0
    pages_processed: int = 0
    pages_per_minute: float = 0.0
    avg_time_per_page_ms: float = 0.0
    p50_time_per_page_ms: float = 0.0
    p95_time_per_page_ms: float = 0.0
    p99_time_per_page_ms: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0

    # Stage timings (avg ms per page)
    extraction_avg_ms: float = 0.0
    ocr_avg_ms: float = 0.0
    assembly_avg_ms: float = 0.0
    compression_avg_ms: float = 0.0

    # Throughput
    extraction_queue_throughput: float = 0.0  # items/sec
    ocr_queue_throughput: float = 0.0
    assembly_queue_throughput: float = 0.0

    # Per-page detail (not serialized in summary)
    page_timings: list = field(default_factory=list)

    # I/O counters (optional)
    io_counters: object = field(default=None)

    def to_dict(self):
        """Serialize to dict for JSON storage (excludes raw page_timings)."""
        d = asdict(self)
        d.pop("page_timings", None)
        # Convert IOCounters to dict if present
        if d.get("io_counters") is not None:
            io = d["io_counters"]
            if isinstance(io, dict):
                d["io_counters"] = io
        else:
            d.pop("io_counters", None)
        return d


# ---------------------------------------------------------------------------
# System metrics collection
# ---------------------------------------------------------------------------

def collect_system_metrics():
    """Collect current system memory and CPU metrics via psutil.

    Returns a dict with memory_mb, cpu_percent, or empty values if
    psutil is unavailable.
    """
    if not HAS_PSUTIL:
        return {"memory_mb": 0.0, "cpu_percent": 0.0}

    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    return {
        "memory_mb": mem_info.rss / (1024 * 1024),
        "cpu_percent": proc.cpu_percent(interval=0.1),
    }


def collect_io_baseline():
    """Capture a baseline snapshot of disk and network I/O counters.

    Returns a dict with raw byte counts, or zeros if psutil is
    unavailable or the platform doesn't expose the counters.
    """
    baseline = {
        "disk_read": 0, "disk_write": 0,
        "net_sent": 0, "net_recv": 0,
    }
    if not HAS_PSUTIL:
        return baseline
    try:
        dio = psutil.disk_io_counters()
        if dio:
            baseline["disk_read"] = dio.read_bytes
            baseline["disk_write"] = dio.write_bytes
    except Exception:
        pass
    try:
        nio = psutil.net_io_counters()
        if nio:
            baseline["net_sent"] = nio.bytes_sent
            baseline["net_recv"] = nio.bytes_recv
    except Exception:
        pass
    return baseline


def compute_io_delta(baseline, current):
    """Compute the difference between two I/O snapshots.

    Args:
        baseline: dict from ``collect_io_baseline()`` (start).
        current: dict from ``collect_io_baseline()`` (end).

    Returns:
        IOCounters with delta values in megabytes.
    """
    mb = 1024 * 1024
    return IOCounters(
        disk_read_mb=round((current["disk_read"] - baseline["disk_read"]) / mb, 2),
        disk_write_mb=round((current["disk_write"] - baseline["disk_write"]) / mb, 2),
        net_sent_mb=round((current["net_sent"] - baseline["net_sent"]) / mb, 2),
        net_recv_mb=round((current["net_recv"] - baseline["net_recv"]) / mb, 2),
    )


def check_regression(metrics, baseline_path):
    """Check benchmark metrics against regression thresholds.

    Args:
        metrics: BenchmarkMetrics from the current run.
        baseline_path: Path to a ``baseline.json`` file containing
            ``{"version": "...", "thresholds": {...}}``.

    Returns:
        dict with ``"passed"`` (bool) and ``"checks"`` (list of dicts).
    """
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    thresholds = baseline.get("thresholds", {})
    checks = []

    # simulate_ppm_min — pages per minute must be >= threshold
    if "simulate_ppm_min" in thresholds:
        thr = thresholds["simulate_ppm_min"]
        actual = metrics.pages_per_minute
        checks.append({
            "metric": "simulate_ppm_min",
            "actual": actual,
            "threshold": thr,
            "passed": actual >= thr,
        })

    # p95_latency_ms_max — p95 must be <= threshold
    if "p95_latency_ms_max" in thresholds:
        thr = thresholds["p95_latency_ms_max"]
        actual = metrics.p95_time_per_page_ms
        checks.append({
            "metric": "p95_latency_ms_max",
            "actual": actual,
            "threshold": thr,
            "passed": actual <= thr,
        })

    # peak_memory_mb_max — peak memory must be <= threshold
    if "peak_memory_mb_max" in thresholds:
        thr = thresholds["peak_memory_mb_max"]
        actual = metrics.peak_memory_mb
        checks.append({
            "metric": "peak_memory_mb_max",
            "actual": actual,
            "threshold": thr,
            "passed": actual <= thr,
        })

    passed = all(c["passed"] for c in checks)
    return {"passed": passed, "checks": checks}


# ---------------------------------------------------------------------------
# Simulate mode
# ---------------------------------------------------------------------------

def _sample_ms(low_high):
    """Draw a random sample from a uniform distribution (ms)."""
    low, high = low_high
    return random.uniform(low, high)


def simulate_pipeline(num_pages=100, config=None):
    """Run a simulated benchmark with synthetic timing distributions.

    This does NOT require GPU, PaddleOCR, or any heavyweight dependencies.
    It generates realistic per-page timing data based on documented
    performance characteristics of the EDCOCR pipeline.

    Args:
        num_pages: Number of pages to simulate.
        config: Pipeline config dict (uses DEFAULT_CONFIG if None).

    Returns:
        BenchmarkMetrics with populated fields.
    """
    config = config or dict(DEFAULT_CONFIG)

    io_baseline = collect_io_baseline()

    metrics = BenchmarkMetrics(
        run_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode="simulate",
        pipeline_version=__version__,
        config=config,
        pages_processed=num_pages,
    )

    extraction_times = []
    ocr_times = []
    assembly_times = []
    compression_times = []
    total_page_times = []
    memory_samples = []

    for page_idx in range(num_pages):
        ext_ms = _sample_ms(SIMULATE_EXTRACTION_MS)
        ocr_ms = _sample_ms(SIMULATE_OCR_MS)
        asm_ms = _sample_ms(SIMULATE_ASSEMBLY_MS)
        comp_ms = _sample_ms(SIMULATE_COMPRESSION_MS)

        extraction_times.append(ext_ms)
        ocr_times.append(ocr_ms)
        assembly_times.append(asm_ms)
        compression_times.append(comp_ms)
        total_page_times.append(ext_ms + ocr_ms + asm_ms + comp_ms)

        # Simulate memory sampling (base ~400MB + ~20MB per queued page)
        queued_pages = min(page_idx, config.get("IMAGE_QUEUE_SIZE", 200))
        mem_mb = SIMULATE_BASE_MEMORY_MB + queued_pages * SIMULATE_MEMORY_PER_PAGE_MB * random.uniform(SIMULATE_MEMORY_JITTER_LOW, SIMULATE_MEMORY_JITTER_HIGH)
        memory_samples.append(mem_mb)

    # In a real pipeline, pages are processed in parallel across workers.
    # The effective wall-clock time per page depends on concurrency.
    num_workers = config.get("NUM_WORKERS", 12)
    # Simulated wall-clock: sum(ocr) / num_workers + serial overhead
    simulated_wall_clock = (
        sum(ocr_times) / num_workers
        + sum(extraction_times) / config.get("NUM_EXTRACTORS", 8)
        + sum(assembly_times)  # single assembler thread
        + sum(compression_times) / config.get("NUM_COMPRESSORS", 8)
    ) / 1000.0  # convert ms to seconds

    metrics.total_duration_seconds = simulated_wall_clock

    # Pages per minute
    if simulated_wall_clock > 0:
        metrics.pages_per_minute = (num_pages / simulated_wall_clock) * 60.0

    # Per-page timing stats
    sorted_times = sorted(total_page_times)
    metrics.avg_time_per_page_ms = statistics.mean(total_page_times)
    metrics.p50_time_per_page_ms = _percentile(sorted_times, 50)
    metrics.p95_time_per_page_ms = _percentile(sorted_times, 95)
    metrics.p99_time_per_page_ms = _percentile(sorted_times, 99)

    # Stage averages
    metrics.extraction_avg_ms = statistics.mean(extraction_times)
    metrics.ocr_avg_ms = statistics.mean(ocr_times)
    metrics.assembly_avg_ms = statistics.mean(assembly_times)
    metrics.compression_avg_ms = statistics.mean(compression_times)

    # Memory
    metrics.peak_memory_mb = max(memory_samples)
    metrics.avg_memory_mb = statistics.mean(memory_samples)

    # Throughput (items/sec based on simulated wall clock)
    if simulated_wall_clock > 0:
        metrics.extraction_queue_throughput = num_pages / simulated_wall_clock
        metrics.ocr_queue_throughput = num_pages / simulated_wall_clock
        metrics.assembly_queue_throughput = num_pages / simulated_wall_clock

    metrics.page_timings = total_page_times

    # I/O delta
    io_current = collect_io_baseline()
    metrics.io_counters = compute_io_delta(io_baseline, io_current)

    return metrics


def _percentile(sorted_data, pct):
    """Compute the pct-th percentile from pre-sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


# ---------------------------------------------------------------------------
# Live mode — queue instrumentation and real pipeline benchmarking
# ---------------------------------------------------------------------------

class InstrumentedQueue:
    """Transparent wrapper around queue.Queue that records put/get timestamps.

    Delegates all public Queue methods so the pipeline sees no difference.
    Timing data is stored in thread-safe lists for post-run aggregation.
    """

    def __init__(self, wrapped, name):
        self._q = wrapped
        self.name = name
        self._lock = threading.Lock()
        self.put_times = []   # perf_counter timestamps
        self.get_times = []
        self.wait_times = []  # seconds blocked in get()

    # -- instrumented ops ---------------------------------------------------

    def put(self, item, block=True, timeout=None):
        with self._lock:
            self._q.put(item, block=block, timeout=timeout)
            ts = time.perf_counter()
            self.put_times.append(ts)

    def get(self, block=True, timeout=None):
        with self._lock:
            t0 = time.perf_counter()
            item = self._q.get(block=block, timeout=timeout)
            t1 = time.perf_counter()
            self.get_times.append(t1)
            self.wait_times.append(t1 - t0)
        return item

    # -- delegated ops (full Queue interface) --------------------------------

    def qsize(self):
        return self._q.qsize()

    def empty(self):
        return self._q.empty()

    def full(self):
        return self._q.full()

    def put_nowait(self, item):
        self.put(item, block=False)

    def get_nowait(self):
        return self.get(block=False)

    def task_done(self):
        return self._q.task_done()

    def join(self):
        return self._q.join()

    @property
    def maxsize(self):
        return self._q.maxsize

    # -- metrics helpers ----------------------------------------------------

    def throughput(self):
        """Items per second based on get() timestamps."""
        with self._lock:
            if len(self.get_times) < 2:
                return 0.0
            span = self.get_times[-1] - self.get_times[0]
            return (len(self.get_times) - 1) / span if span > 0 else 0.0

    def avg_wait_ms(self):
        """Average time blocked in get(), in milliseconds."""
        with self._lock:
            return statistics.mean(self.wait_times) * 1000 if self.wait_times else 0.0

    def avg_transit_ms(self):
        """Average time between a put() and the next get() (queue transit).

        Approximated by pairing put/get events in order. Only meaningful when
        the queue is roughly FIFO with 1:1 put/get correspondence.
        """
        with self._lock:
            n = min(len(self.put_times), len(self.get_times))
            if n == 0:
                return 0.0
            deltas = [
                (self.get_times[i] - self.put_times[i]) * 1000
                for i in range(n)
                if self.get_times[i] >= self.put_times[i]
            ]
            return statistics.mean(deltas) if deltas else 0.0


def _memory_sampler(samples, stop_flag, interval=2.0):
    """Background thread that appends memory samples until *stop_flag* is set."""
    while not stop_flag.is_set():
        m = collect_system_metrics()["memory_mb"]
        if m > 0:
            samples.append(m)
        stop_flag.wait(interval)


def instrument_live_pipeline(input_dir, config=None):
    """Run the real OCR pipeline with queue-level instrumentation.

    Wraps the four inter-stage queues with ``InstrumentedQueue`` to collect
    timing data, then calls ``ocr_gpu_async.main()`` against *input_dir*.
    A background thread samples process memory every 2 seconds.

    After the pipeline finishes, global counters and queue timing data are
    aggregated into a ``BenchmarkMetrics`` result.

    Args:
        input_dir: Path to directory containing source documents.
        config: Pipeline config overrides.

    Returns:
        BenchmarkMetrics with populated fields from actual processing.

    Raises:
        ImportError: If production dependencies are not available.
        FileNotFoundError: If input_dir does not exist.
    """
    config = config or dict(DEFAULT_CONFIG)

    io_baseline = collect_io_baseline()

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Check for GPU dependencies
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        raise ImportError(
            "Live mode requires PaddleOCR and GPU dependencies. "
            "Use --mode simulate for testing without GPU."
        )

    # Import the production pipeline (must happen after paddleocr check)
    import ocr_gpu_async  # noqa: E402

    # Count input files
    source_files = []
    for root, _dirs, files in os.walk(input_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_LIVE_EXTENSIONS:
                source_files.append(os.path.join(root, fname))

    if not source_files:
        print(f"No supported source files found in {input_dir}")
        return BenchmarkMetrics(
            run_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode="live",
            pipeline_version=__version__,
            config=config,
        )

    print(f"Found {len(source_files)} source files for live benchmark")

    # --- Apply config overrides to the pipeline module ---------------------
    ocr_gpu_async.SOURCE_FOLDER = os.path.abspath(input_dir)
    for key in DEFAULT_CONFIG:
        if key in config:
            setattr(ocr_gpu_async, key, config[key])

    # --- Wrap the four inter-stage queues with instrumentation -------------
    q_chunk = InstrumentedQueue(
        queue.Queue(maxsize=config.get("CHUNK_QUEUE_SIZE", 50)), "chunk",
    )
    q_image = InstrumentedQueue(
        queue.Queue(maxsize=config.get("IMAGE_QUEUE_SIZE", 200)), "image",
    )
    q_assembly = InstrumentedQueue(
        queue.Queue(maxsize=config.get("RESULT_QUEUE_SIZE", 5000)), "assembly",
    )
    q_compression = InstrumentedQueue(
        queue.Queue(maxsize=config.get("COMPRESSION_QUEUE_SIZE", 5000)), "compression",
    )

    ocr_gpu_async.chunk_queue = q_chunk
    ocr_gpu_async.image_queue = q_image
    ocr_gpu_async.assembly_queue = q_assembly
    ocr_gpu_async.compression_queue = q_compression

    # Reset global counters so metrics start from zero
    ocr_gpu_async.global_pages_processed = 0
    ocr_gpu_async.global_docs_processed = 0
    ocr_gpu_async.start_time_global = time.time()

    # --- Start memory sampler ----------------------------------------------
    mem_samples = [collect_system_metrics()["memory_mb"]]
    mem_stop = threading.Event()
    mem_thread = threading.Thread(
        target=_memory_sampler, args=(mem_samples, mem_stop), daemon=True,
    )
    mem_thread.start()

    # --- Run the pipeline --------------------------------------------------
    wall_start = time.perf_counter()
    try:
        ocr_gpu_async.main()
    finally:
        mem_stop.set()
        mem_thread.join(timeout=5)
    wall_end = time.perf_counter()

    # --- Collect final memory sample ---------------------------------------
    final_mem = collect_system_metrics()["memory_mb"]
    if final_mem > 0:
        mem_samples.append(final_mem)

    # --- Read global counters from the pipeline ----------------------------
    pages = ocr_gpu_async.global_pages_processed
    elapsed = wall_end - wall_start

    # --- Build per-page timing approximation from queue data ---------------
    # Each page goes through image_queue (put by extractor, get by worker)
    # and assembly_queue (put by worker, get by assembler).
    # We pair puts/gets to approximate per-page latency.
    page_latencies = []
    n_img = min(len(q_image.put_times), len(q_image.get_times))
    n_asm = min(len(q_assembly.put_times), len(q_assembly.get_times))
    n_pairs = min(n_img, n_asm)
    for i in range(n_pairs):
        # extraction ≈ chunk_get → image_put  (approximated as image transit)
        # ocr       ≈ image_get  → assembly_put
        # The total per-page latency is image_put → assembly_get
        lat_ms = (q_assembly.get_times[i] - q_image.put_times[i]) * 1000
        if lat_ms > 0:
            page_latencies.append(lat_ms)

    sorted_lat = sorted(page_latencies)

    # --- Populate metrics --------------------------------------------------
    ppm = (pages / elapsed) * 60 if elapsed > 0 else 0.0
    mem_positive = [m for m in mem_samples if m > 0] or [0.0]

    metrics = BenchmarkMetrics(
        run_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode="live",
        pipeline_version=__version__,
        config=config,
        total_duration_seconds=elapsed,
        pages_processed=pages,
        pages_per_minute=ppm,
        avg_time_per_page_ms=(elapsed * 1000 / pages) if pages else 0.0,
        p50_time_per_page_ms=_percentile(sorted_lat, 50),
        p95_time_per_page_ms=_percentile(sorted_lat, 95),
        p99_time_per_page_ms=_percentile(sorted_lat, 99),
        peak_memory_mb=max(mem_positive),
        avg_memory_mb=statistics.mean(mem_positive),
        extraction_avg_ms=q_chunk.avg_transit_ms(),
        ocr_avg_ms=q_image.avg_transit_ms(),
        assembly_avg_ms=q_assembly.avg_transit_ms(),
        compression_avg_ms=q_compression.avg_transit_ms(),
        extraction_queue_throughput=q_chunk.throughput(),
        ocr_queue_throughput=q_image.throughput(),
        assembly_queue_throughput=q_assembly.throughput(),
        page_timings=sorted_lat,
    )

    # I/O delta
    io_current = collect_io_baseline()
    metrics.io_counters = compute_io_delta(io_baseline, io_current)

    return metrics


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(metrics, output_dir=None):
    """Save benchmark metrics to a JSON file.

    Args:
        metrics: BenchmarkMetrics instance.
        output_dir: Directory for results (defaults to BENCHMARK_DIR).

    Returns:
        Path to the saved JSON file.
    """
    output_dir = output_dir or BENCHMARK_DIR
    os.makedirs(output_dir, exist_ok=True)

    filename = f"benchmark_{metrics.mode}_{metrics.run_id}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    return filepath


def load_results(filepath):
    """Load benchmark metrics from a JSON file.

    Args:
        filepath: Path to the JSON results file.

    Returns:
        BenchmarkMetrics instance.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # BenchmarkMetrics expects page_timings but it's excluded from JSON
    data.setdefault("page_timings", [])
    # Reconstruct IOCounters from dict if present
    io_data = data.pop("io_counters", None)
    if isinstance(io_data, dict):
        data["io_counters"] = IOCounters(**io_data)
    return BenchmarkMetrics(**data)


def find_latest_result(results_dir=None):
    """Find the most recent benchmark result file.

    Args:
        results_dir: Directory to search (defaults to BENCHMARK_DIR).

    Returns:
        Path to the latest JSON file, or None if no results exist.
    """
    results_dir = results_dir or BENCHMARK_DIR
    if not os.path.isdir(results_dir):
        return None

    json_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.startswith("benchmark_") and f.endswith(".json")
    ]

    if not json_files:
        return None

    return max(json_files, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_report(results_dir=None):
    """Generate a text summary of all benchmark runs.

    Args:
        results_dir: Directory containing result JSON files.

    Returns:
        Formatted report string.
    """
    results_dir = results_dir or BENCHMARK_DIR
    if not os.path.isdir(results_dir):
        return "No benchmark results found."

    json_files = sorted([
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.startswith("benchmark_") and f.endswith(".json")
    ], key=os.path.getmtime)

    if not json_files:
        return "No benchmark results found."

    lines = []
    lines.append("=" * 72)
    lines.append("  EDCOCR Performance Benchmark Report")
    lines.append("=" * 72)
    lines.append("")

    for fpath in json_files:
        m = load_results(fpath)
        lines.append(f"Run: {m.run_id}  |  Mode: {m.mode}  |  {m.timestamp}")
        lines.append(f"  Pipeline Version: {m.pipeline_version}")
        lines.append(f"  Pages Processed:  {m.pages_processed}")
        lines.append(f"  Total Duration:   {m.total_duration_seconds:.2f}s")
        lines.append(f"  Pages/Minute:     {m.pages_per_minute:.1f}")
        lines.append(f"  Avg ms/page:      {m.avg_time_per_page_ms:.1f}")
        lines.append(f"  P50 ms/page:      {m.p50_time_per_page_ms:.1f}")
        lines.append(f"  P95 ms/page:      {m.p95_time_per_page_ms:.1f}")
        lines.append(f"  P99 ms/page:      {m.p99_time_per_page_ms:.1f}")
        lines.append(f"  Peak Memory:      {m.peak_memory_mb:.1f} MB")
        lines.append(f"  Stage Avg (ms):   ext={m.extraction_avg_ms:.1f}  "
                      f"ocr={m.ocr_avg_ms:.1f}  "
                      f"asm={m.assembly_avg_ms:.1f}  "
                      f"comp={m.compression_avg_ms:.1f}")
        if m.io_counters is not None and isinstance(m.io_counters, IOCounters):
            lines.append(f"  I/O:              disk_r={m.io_counters.disk_read_mb:.1f}MB  "
                          f"disk_w={m.io_counters.disk_write_mb:.1f}MB  "
                          f"net_s={m.io_counters.net_sent_mb:.1f}MB  "
                          f"net_r={m.io_counters.net_recv_mb:.1f}MB")
        lines.append("-" * 72)

    return "\n".join(lines)


def compare_runs(run1_path, run2_path):
    """Side-by-side comparison of two benchmark runs.

    Args:
        run1_path: Path to first (baseline) result JSON.
        run2_path: Path to second (comparison) result JSON.

    Returns:
        Formatted comparison string with deltas.
    """
    m1 = load_results(run1_path)
    m2 = load_results(run2_path)

    def _delta(v1, v2, unit="", invert=False):
        """Format value with delta. invert=True means lower is better."""
        if v1 == 0:
            return f"{v2:.1f}{unit} (no baseline)"
        diff = v2 - v1
        pct = (diff / v1) * 100
        sign = "+" if diff >= 0 else ""
        # For metrics where lower is better (ms/page, memory), positive = worse
        if invert:
            indicator = "WORSE" if diff > 0 else "BETTER"
        else:
            indicator = "BETTER" if diff > 0 else "WORSE"
        return f"{v2:.1f}{unit} ({sign}{pct:.1f}% {indicator})"

    lines = []
    lines.append("=" * 72)
    lines.append("  Benchmark Comparison")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Baseline:    {m1.run_id} ({m1.mode}, {m1.timestamp})")
    lines.append(f"  Comparison:  {m2.run_id} ({m2.mode}, {m2.timestamp})")
    lines.append(f"  Versions:    {m1.pipeline_version} -> {m2.pipeline_version}")
    lines.append("")
    lines.append(f"  {'Metric':<30} {'Baseline':>12} {'Current':>30}")
    lines.append("  " + "-" * 68)

    rows = [
        ("Pages Processed", m1.pages_processed, f"{m2.pages_processed}"),
        ("Total Duration (s)", m1.total_duration_seconds,
         _delta(m1.total_duration_seconds, m2.total_duration_seconds, "s", invert=True)),
        ("Pages/Minute", m1.pages_per_minute,
         _delta(m1.pages_per_minute, m2.pages_per_minute, "", invert=False)),
        ("Avg ms/page", m1.avg_time_per_page_ms,
         _delta(m1.avg_time_per_page_ms, m2.avg_time_per_page_ms, "ms", invert=True)),
        ("P95 ms/page", m1.p95_time_per_page_ms,
         _delta(m1.p95_time_per_page_ms, m2.p95_time_per_page_ms, "ms", invert=True)),
        ("P99 ms/page", m1.p99_time_per_page_ms,
         _delta(m1.p99_time_per_page_ms, m2.p99_time_per_page_ms, "ms", invert=True)),
        ("Peak Memory (MB)", m1.peak_memory_mb,
         _delta(m1.peak_memory_mb, m2.peak_memory_mb, "MB", invert=True)),
        ("Extraction avg ms", m1.extraction_avg_ms,
         _delta(m1.extraction_avg_ms, m2.extraction_avg_ms, "ms", invert=True)),
        ("OCR avg ms", m1.ocr_avg_ms,
         _delta(m1.ocr_avg_ms, m2.ocr_avg_ms, "ms", invert=True)),
        ("Assembly avg ms", m1.assembly_avg_ms,
         _delta(m1.assembly_avg_ms, m2.assembly_avg_ms, "ms", invert=True)),
        ("Compression avg ms", m1.compression_avg_ms,
         _delta(m1.compression_avg_ms, m2.compression_avg_ms, "ms", invert=True)),
    ]

    for label, baseline_val, current_str in rows:
        if isinstance(baseline_val, float):
            lines.append(f"  {label:<30} {baseline_val:>12.1f} {current_str:>30}")
        else:
            lines.append(f"  {label:<30} {str(baseline_val):>12} {current_str:>30}")

    lines.append("")

    # Performance target check
    if m1.pages_per_minute > 0 and m2.pages_per_minute > 0:
        slowdown = 1.0 - (m2.pages_per_minute / m1.pages_per_minute)
        lines.append("  Performance Targets:")
        lines.append(f"    Slowdown: {slowdown * 100:.1f}%")
        if slowdown <= 0.5:
            lines.append("    [PASS] Within 50% threshold (layout-only target)")
        else:
            lines.append("    [FAIL] Exceeds 50% threshold (layout-only target)")
        if slowdown <= 1.0:
            lines.append("    [PASS] Within 100% threshold (full DocIntel target)")
        else:
            lines.append("    [FAIL] Exceeds 100% threshold (full DocIntel target)")

    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark(mode="simulate", num_pages=100, config=None,
                  enable_profile=False):
    """Main entry point for running a benchmark.

    Args:
        mode: "simulate" or "live".
        num_pages: Number of pages (simulate mode only).
        config: Pipeline config overrides.
        enable_profile: If True, wrap the run with FlameGraphProfiler.

    Returns:
        BenchmarkMetrics instance.
    """
    print(f"Starting {mode} benchmark ({num_pages} pages)...")

    flame = None
    if enable_profile:
        from profiling import FlameGraphProfiler
        flame = FlameGraphProfiler()
        flame.start()

    try:
        if mode == "simulate":
            metrics = simulate_pipeline(num_pages=num_pages, config=config)
        elif mode == "live":
            input_dir = config.pop("input_dir", "ocr_source") if config else "ocr_source"
            metrics = instrument_live_pipeline(input_dir, config=config)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'simulate' or 'live'.")
    finally:
        if flame is not None:
            flame.stop()

    filepath = save_results(metrics)
    print(f"Results saved to: {filepath}")
    print(f"  Pages/Minute: {metrics.pages_per_minute:.1f}")
    print(f"  Avg ms/page:  {metrics.avg_time_per_page_ms:.1f}")
    print(f"  Peak Memory:  {metrics.peak_memory_mb:.1f} MB")

    # Print I/O counters if available
    if metrics.io_counters is not None:
        io = metrics.io_counters
        if isinstance(io, IOCounters):
            print(f"  Disk Read:    {io.disk_read_mb:.2f} MB")
            print(f"  Disk Write:   {io.disk_write_mb:.2f} MB")
            print(f"  Net Sent:     {io.net_sent_mb:.2f} MB")
            print(f"  Net Recv:     {io.net_recv_mb:.2f} MB")

    # Save profile data if profiling was enabled
    if flame is not None:
        profile_dir = os.path.join(BENCHMARK_DIR, "profiles")
        os.makedirs(profile_dir, exist_ok=True)
        prof_path = os.path.join(
            profile_dir, f"profile_{metrics.run_id}.prof"
        )
        flame.dump_stats(prof_path)
        print(f"  Profile:      {prof_path}")
        report = flame.generate_report()
        report_path = os.path.join(
            profile_dir, f"profile_{metrics.run_id}.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  Profile JSON: {report_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="EDCOCR Performance Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["simulate", "live"], default="simulate",
        help="Benchmark mode: simulate (no GPU) or live (real pipeline)",
    )
    parser.add_argument(
        "--pages", type=int, default=100,
        help="Number of pages to simulate (simulate mode only)",
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Input directory for live mode",
    )
    parser.add_argument(
        "--report", nargs="?", const="all", default=None,
        help="Display benchmark report. Use 'latest' for most recent run.",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("RUN1", "RUN2"),
        help="Compare two benchmark result files",
    )
    parser.add_argument(
        "--results-dir", type=str, default=BENCHMARK_DIR,
        help="Directory for benchmark results",
    )
    parser.add_argument(
        "--profile", action="store_true", default=False,
        help="Enable cProfile-based profiling around the benchmark run",
    )
    parser.add_argument(
        "--check-regression", type=str, default=None, metavar="PATH",
        help="Check metrics against regression baseline JSON file",
    )

    args = parser.parse_args()

    # Report mode
    if args.report is not None:
        if args.report == "latest":
            latest = find_latest_result(args.results_dir)
            if latest:
                m = load_results(latest)
                print(f"Latest run: {m.run_id} ({m.mode}, {m.timestamp})")
                print(f"  Pages/Minute: {m.pages_per_minute:.1f}")
                print(f"  Avg ms/page:  {m.avg_time_per_page_ms:.1f}")
                print(f"  P95 ms/page:  {m.p95_time_per_page_ms:.1f}")
                print(f"  Peak Memory:  {m.peak_memory_mb:.1f} MB")
            else:
                print("No benchmark results found.")
        else:
            print(generate_report(args.results_dir))
        return

    # Compare mode
    if args.compare:
        print(compare_runs(args.compare[0], args.compare[1]))
        return

    # Run benchmark
    config = dict(DEFAULT_CONFIG)
    if args.input_dir:
        config["input_dir"] = args.input_dir

    metrics = run_benchmark(
        mode=args.mode, num_pages=args.pages, config=config,
        enable_profile=args.profile,
    )

    # Regression check
    if args.check_regression:
        result = check_regression(metrics, args.check_regression)
        print()
        print("Regression Check:")
        for c in result["checks"]:
            status = "PASS" if c["passed"] else "FAIL"
            print(f"  [{status}] {c['metric']}: {c['actual']:.1f} "
                  f"(threshold: {c['threshold']:.1f})")
        if not result["passed"]:
            print("REGRESSION DETECTED — exiting with code 1")
            raise SystemExit(1)
        else:
            print("All regression checks passed.")


if __name__ == "__main__":
    main()
