"""
Distributed Pipeline Scale Testing Framework for EDCOCR.

Validates the coordinator + worker architecture under load by generating
synthetic PDF corpora, submitting jobs, monitoring completion, and
collecting fleet-wide metrics.

Usage:
    python scale_test.py --mode simulate              # Mock mode (no Django)
    python scale_test.py --mode live --jobs 50         # Real distributed test
    python scale_test.py --mode crash-recovery         # Worker crash test
    python scale_test.py --report latest               # Display latest results
    python scale_test.py --compare run1.json run2.json # Compare two runs

Modes:
    simulate       - Synthetic timing simulation (no Django/Celery required)
    live           - Submit real jobs to the coordinator via Celery
    crash-recovery - Submit large document, kill a worker, verify retry

Requirements:
    - simulate mode: No external dependencies
    - live/crash-recovery mode: Django coordinator stack running
"""

import argparse
import json
import os
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

try:
    from ocr_local.config.version import __version__
except ImportError:
    __version__ = "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCALE_TEST_DIR = "scale_test_results"

# Corpus generation parameters
CORPUS_PROFILES = {
    "small": {"min_pages": 1, "max_pages": 5, "weight": 0.60},
    "medium": {"min_pages": 10, "max_pages": 20, "weight": 0.25},
    "large": {"min_pages": 50, "max_pages": 100, "weight": 0.15},
}

# Simulate mode timing distributions (milliseconds per page)
# Based on distributed overhead estimates
SIM_INGEST_MS = (100, 300)        # File copy to NFS + DB record creation
SIM_OCR_PER_PAGE_MS = (200, 600)  # PaddleOCR on GPU worker
SIM_ASSEMBLY_MS = (50, 150)       # PDF merge + text concatenation
SIM_COMPRESS_MS = (40, 120)       # Ghostscript per-page
SIM_NER_MS = (20, 80)             # spaCy entity extraction
SIM_FANOUT_OVERHEAD_MS = (200, 500)  # Chord setup + callback overhead

# Fan-out threshold matching coordinator/jobs/tasks.py
FANOUT_THRESHOLD = 20

# Default test configuration
DEFAULT_CONFIG = {
    "num_jobs": 20,
    "num_workers": 1,
    "worker_concurrency": 4,
    "fanout_threshold": FANOUT_THRESHOLD,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WorkerStats:
    """Per-worker performance statistics."""

    hostname: str
    pages_processed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_page_time_ms: float = 0.0
    ppm: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class ScaleTestResult:
    """Container for a single scale test run's results."""

    test_id: str
    timestamp: str
    mode: str
    pipeline_version: str
    config: dict

    # Job-level metrics
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0

    # Page-level metrics
    total_pages: int = 0
    pages_processed: int = 0
    pages_failed: int = 0

    # Timing
    duration_seconds: float = 0.0
    fleet_ppm: float = 0.0
    avg_job_duration_seconds: float = 0.0
    p50_job_duration_seconds: float = 0.0
    p95_job_duration_seconds: float = 0.0
    p99_job_duration_seconds: float = 0.0

    # Worker stats
    per_worker_stats: list = field(default_factory=list)

    # Error summary
    error_summary: dict = field(default_factory=dict)

    # Corpus info
    corpus_info: dict = field(default_factory=dict)

    # Raw job durations (not serialized)
    job_durations: list = field(default_factory=list)

    def to_dict(self):
        """Serialize to dict for JSON storage (excludes raw timings)."""
        d = asdict(self)
        d.pop("job_durations", None)
        return d


# ---------------------------------------------------------------------------
# Synthetic PDF corpus generation
# ---------------------------------------------------------------------------

def _minimal_pdf_bytes(num_pages):
    """Generate a minimal valid PDF with the given number of pages.

    Creates a valid PDF with text content on each page without requiring
    any external PDF libraries. Each page contains a simple text string.
    """
    # Build PDF objects
    objects = []
    page_refs = []

    # Object 1: Catalog (placeholder, updated later)
    objects.append(None)
    # Object 2: Pages (placeholder, updated later)
    objects.append(None)
    # Object 3: Font
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Create page objects
    for page_idx in range(num_pages):
        # Content stream
        text = f"Scale test page {page_idx + 1} of {num_pages} - {uuid.uuid4().hex[:8]}"
        stream_content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        stream_obj = (
            b"<< /Length " + str(len(stream_content)).encode() + b" >>\n"
            b"stream\n" + stream_content + b"\nendstream"
        )
        stream_num = len(objects) + 1
        objects.append(stream_obj)

        # Page object
        page_num = len(objects) + 1
        page_obj = (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Contents " + str(stream_num).encode() + b" 0 R "
            b"/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objects.append(page_obj)
        page_refs.append(f"{page_num} 0 R")

    # Now fill in catalog and pages
    kids_str = " ".join(page_refs)
    objects[1] = (
        b"<< /Type /Pages /Kids [" + kids_str.encode() + b"] "
        b"/Count " + str(num_pages).encode() + b" >>"
    )
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"

    # Serialize PDF
    parts = [b"%PDF-1.4\n"]
    offsets = []
    for i, obj in enumerate(objects):
        offsets.append(len(b"".join(parts)))
        obj_num = i + 1
        parts.append(f"{obj_num} 0 obj\n".encode())
        parts.append(obj + b"\n")
        parts.append(b"endobj\n")

    # Cross-reference table
    xref_offset = len(b"".join(parts))
    parts.append(b"xref\n")
    parts.append(f"0 {len(objects) + 1}\n".encode())
    parts.append(b"0000000000 65535 f \n")
    for offset in offsets:
        parts.append(f"{offset:010d} 00000 n \n".encode())

    parts.append(b"trailer\n")
    parts.append(
        f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    )
    parts.append(b"startxref\n")
    parts.append(f"{xref_offset}\n".encode())
    parts.append(b"%%EOF\n")

    return b"".join(parts)


def generate_corpus(num_jobs, output_dir, seed=None):
    """Generate a synthetic PDF corpus for scale testing.

    Creates PDFs of varying sizes based on CORPUS_PROFILES weights.

    Args:
        num_jobs: Number of PDF files to generate.
        output_dir: Directory to write PDFs into.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with 'path', 'pages', 'size_category' per document.
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Build weighted category list
    categories = []
    for cat_name, profile in CORPUS_PROFILES.items():
        count = max(1, round(num_jobs * profile["weight"]))
        categories.extend([cat_name] * count)
    random.shuffle(categories)
    categories = categories[:num_jobs]

    # Pad if rounding caused fewer items
    while len(categories) < num_jobs:
        categories.append("small")

    corpus = []
    for i, category in enumerate(categories):
        profile = CORPUS_PROFILES[category]
        num_pages = random.randint(profile["min_pages"], profile["max_pages"])
        filename = f"scale_test_{i:04d}_{category}_{num_pages}p.pdf"
        filepath = os.path.join(output_dir, filename)

        pdf_bytes = _minimal_pdf_bytes(num_pages)
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)

        corpus.append({
            "path": filepath,
            "pages": num_pages,
            "size_category": category,
            "size_bytes": len(pdf_bytes),
        })

    return corpus


def corpus_summary(corpus):
    """Generate a summary dict for a corpus."""
    total_pages = sum(d["pages"] for d in corpus)
    by_category = {}
    for d in corpus:
        cat = d["size_category"]
        if cat not in by_category:
            by_category[cat] = {"count": 0, "pages": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["pages"] += d["pages"]

    return {
        "total_documents": len(corpus),
        "total_pages": total_pages,
        "by_category": by_category,
        "avg_pages_per_doc": total_pages / len(corpus) if corpus else 0,
    }


# ---------------------------------------------------------------------------
# Simulate mode
# ---------------------------------------------------------------------------

def _sample_ms(low_high):
    """Draw a random sample from a uniform distribution (ms)."""
    low, high = low_high
    return random.uniform(low, high)


def simulate_distributed(num_jobs=20, num_workers=1, config=None, seed=None):
    """Simulate distributed pipeline behavior with synthetic timing.

    Models the coordinator/worker architecture without requiring Django
    or Celery. Simulates job ingestion, page-level OCR processing,
    assembly, compression, and NER extraction.

    Args:
        num_jobs: Number of documents to simulate.
        num_workers: Number of simulated workers.
        config: Override configuration dict.
        seed: Random seed for reproducibility.

    Returns:
        ScaleTestResult with populated metrics.
    """
    if seed is not None:
        random.seed(seed)

    config = config or dict(DEFAULT_CONFIG)
    config["num_jobs"] = num_jobs
    config["num_workers"] = num_workers

    result = ScaleTestResult(
        test_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode="simulate",
        pipeline_version=__version__,
        config=config,
        total_jobs=num_jobs,
    )

    # Generate corpus metadata (no actual files)
    corpus_meta = []
    categories = []
    for cat_name, profile in CORPUS_PROFILES.items():
        count = max(1, round(num_jobs * profile["weight"]))
        categories.extend([cat_name] * count)
    random.shuffle(categories)
    categories = categories[:num_jobs]
    while len(categories) < num_jobs:
        categories.append("small")

    for i, category in enumerate(categories):
        profile = CORPUS_PROFILES[category]
        num_pages = random.randint(profile["min_pages"], profile["max_pages"])
        corpus_meta.append({"pages": num_pages, "category": category})

    total_pages = sum(d["pages"] for d in corpus_meta)
    result.total_pages = total_pages

    # Simulate worker names
    worker_names = [f"worker-{i}@host-{i}" for i in range(num_workers)]
    worker_page_counts = {w: 0 for w in worker_names}
    worker_page_times = {w: [] for w in worker_names}

    # Simulate job processing
    job_durations = []
    completed = 0
    failed = 0
    pages_processed = 0

    for job_meta in corpus_meta:
        num_pages = job_meta["pages"]

        # Simulate ingestion
        ingest_ms = _sample_ms(SIM_INGEST_MS)

        # Simulate OCR processing
        uses_fanout = num_pages > FANOUT_THRESHOLD
        fanout_overhead_ms = _sample_ms(SIM_FANOUT_OVERHEAD_MS) if uses_fanout else 0

        page_times = []
        for _ in range(num_pages):
            ocr_ms = _sample_ms(SIM_OCR_PER_PAGE_MS)
            page_times.append(ocr_ms)

        # With multiple workers, pages are distributed
        if num_workers > 1 and uses_fanout:
            # Simulated parallel processing: divide by workers
            parallel_factor = min(num_workers, num_pages)
            effective_ocr_ms = sum(page_times) / parallel_factor
        else:
            effective_ocr_ms = sum(page_times)

        # Assembly + compression + NER
        assembly_ms = _sample_ms(SIM_ASSEMBLY_MS)
        compress_ms = sum(_sample_ms(SIM_COMPRESS_MS) for _ in range(num_pages))
        ner_ms = _sample_ms(SIM_NER_MS)

        total_job_ms = (
            ingest_ms + fanout_overhead_ms + effective_ocr_ms
            + assembly_ms + compress_ms + ner_ms
        )

        # Simulate occasional failures (2% rate)
        if random.random() < 0.02:
            failed += 1
        else:
            completed += 1
            pages_processed += num_pages

        job_durations.append(total_job_ms / 1000.0)  # Convert to seconds

        # Assign pages to workers (round-robin simulation)
        for p_idx in range(num_pages):
            worker = worker_names[p_idx % num_workers]
            worker_page_counts[worker] += 1
            worker_page_times[worker].append(page_times[p_idx])

    # Compute overall metrics
    result.duration_seconds = sum(job_durations)  # Simulated total time
    result.completed_jobs = completed
    result.failed_jobs = failed
    result.pages_processed = pages_processed

    if result.duration_seconds > 0:
        result.fleet_ppm = (pages_processed / result.duration_seconds) * 60
    else:
        result.fleet_ppm = 0.0

    # Job duration stats
    result.job_durations = job_durations
    if job_durations:
        result.avg_job_duration_seconds = statistics.mean(job_durations)
        sorted_durations = sorted(job_durations)
        n = len(sorted_durations)
        result.p50_job_duration_seconds = sorted_durations[int(n * 0.50)]
        result.p95_job_duration_seconds = sorted_durations[min(int(n * 0.95), n - 1)]
        result.p99_job_duration_seconds = sorted_durations[min(int(n * 0.99), n - 1)]

    # Per-worker stats
    for worker_name in worker_names:
        pages = worker_page_counts[worker_name]
        times = worker_page_times[worker_name]
        avg_time = statistics.mean(times) if times else 0.0
        worker_ppm = (pages / result.duration_seconds) * 60 if result.duration_seconds > 0 else 0.0
        result.per_worker_stats.append(WorkerStats(
            hostname=worker_name,
            pages_processed=pages,
            tasks_completed=pages,
            avg_page_time_ms=avg_time,
            ppm=worker_ppm,
        ).to_dict())

    # Corpus info
    result.corpus_info = {
        "total_documents": num_jobs,
        "total_pages": total_pages,
        "avg_pages_per_doc": total_pages / num_jobs if num_jobs > 0 else 0,
        "fanout_count": sum(1 for m in corpus_meta if m["pages"] > FANOUT_THRESHOLD),
        "single_worker_count": sum(1 for m in corpus_meta if m["pages"] <= FANOUT_THRESHOLD),
    }

    if failed > 0:
        result.error_summary = {"simulated_random_failure": failed}

    return result


# ---------------------------------------------------------------------------
# Live mode (requires Django coordinator stack)
# ---------------------------------------------------------------------------

def _setup_django():
    """Initialize Django settings for ORM access."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coordinator.coordinator.settings")
    import django
    django.setup()


def live_distributed(num_jobs=20, corpus_dir=None, config=None, poll_interval=2.0,
                     timeout=3600, seed=None):
    """Run a live scale test against the distributed coordinator.

    Generates a synthetic corpus, submits jobs via Celery, monitors
    completion, and collects fleet-wide metrics from Django models.

    Args:
        num_jobs: Number of documents to submit.
        corpus_dir: Directory for generated PDFs (default: scale_test_corpus/).
        config: Override configuration dict.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait for all jobs.
        seed: Random seed for reproducibility.

    Returns:
        ScaleTestResult with populated metrics.
    """
    _setup_django()
    from jobs.models import Job, PageResult, Worker
    from jobs.tasks import ingest_document

    config = config or dict(DEFAULT_CONFIG)
    config["num_jobs"] = num_jobs

    corpus_dir = corpus_dir or "scale_test_corpus"
    corpus = generate_corpus(num_jobs, corpus_dir, seed=seed)
    corpus_info = corpus_summary(corpus)

    result = ScaleTestResult(
        test_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode="live",
        pipeline_version=__version__,
        config=config,
        total_jobs=num_jobs,
        total_pages=corpus_info["total_pages"],
        corpus_info=corpus_info,
    )

    print(f"Corpus generated: {corpus_info['total_documents']} docs, "
          f"{corpus_info['total_pages']} pages")

    # Submit all jobs — create Job records first, then dispatch via Celery
    job_ids = []
    for doc in corpus:
        job = Job.objects.create(
            source_file=doc["path"],
            status=Job.Status.SUBMITTED,
        )
        ingest_document.delay(str(job.job_id))
        job_ids.append(job.job_id)
    print(f"Submitted {len(job_ids)} jobs")

    # Monitor until all complete or timeout
    start_time = time.monotonic()
    terminal_statuses = {"completed", "failed", "cancelled"}

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            print(f"Timeout after {timeout}s")
            break

        jobs = Job.objects.filter(job_id__in=job_ids)
        statuses = list(jobs.values_list("status", flat=True))

        done_count = sum(1 for s in statuses if s in terminal_statuses)
        in_progress = len(statuses) - done_count
        print(f"\r  [{elapsed:.0f}s] {done_count}/{len(job_ids)} done, "
              f"{in_progress} in progress", end="", flush=True)

        if done_count >= len(job_ids):
            break

        time.sleep(poll_interval)

    print()  # Newline after progress

    # Collect metrics
    result.duration_seconds = time.monotonic() - start_time

    jobs = Job.objects.filter(job_id__in=job_ids)
    result.completed_jobs = jobs.filter(status="completed").count()
    result.failed_jobs = jobs.filter(status="failed").count()
    result.cancelled_jobs = jobs.filter(status="cancelled").count()

    # Page-level metrics
    completed_job_ids = list(
        jobs.filter(status="completed").values_list("job_id", flat=True)
    )
    page_results = PageResult.objects.filter(job_id__in=completed_job_ids)
    result.pages_processed = page_results.filter(status="completed").count()
    result.pages_failed = page_results.filter(status="failed").count()

    # Job duration stats
    job_durations = []
    for job in jobs.filter(status="completed"):
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()
            job_durations.append(duration)

    result.job_durations = job_durations
    if job_durations:
        result.avg_job_duration_seconds = statistics.mean(job_durations)
        sorted_d = sorted(job_durations)
        n = len(sorted_d)
        result.p50_job_duration_seconds = sorted_d[int(n * 0.50)]
        result.p95_job_duration_seconds = sorted_d[min(int(n * 0.95), n - 1)]
        result.p99_job_duration_seconds = sorted_d[min(int(n * 0.99), n - 1)]

    # Fleet PPM
    if result.duration_seconds > 0:
        result.fleet_ppm = (result.pages_processed / result.duration_seconds) * 60

    # Per-worker stats — only workers that participated in this test run
    active_hostnames = page_results.values_list(
        "worker_hostname", flat=True
    ).distinct()
    workers = Worker.objects.filter(hostname__in=active_hostnames)
    for worker in workers:
        worker_pages = page_results.filter(
            worker_hostname=worker.hostname, status="completed"
        )
        page_count = worker_pages.count()
        avg_time = 0.0
        times = list(worker_pages.values_list("processing_time_ms", flat=True))
        if times:
            avg_time = statistics.mean(times)
        w_ppm = (page_count / result.duration_seconds) * 60 if result.duration_seconds > 0 else 0.0

        result.per_worker_stats.append(WorkerStats(
            hostname=worker.hostname,
            pages_processed=page_count,
            tasks_completed=worker.tasks_completed,
            tasks_failed=worker.tasks_failed,
            avg_page_time_ms=avg_time,
            ppm=w_ppm,
        ).to_dict())

    # Error summary
    error_messages = list(
        jobs.filter(status="failed")
        .exclude(error_message="")
        .values_list("error_message", flat=True)
    )
    error_counts = {}
    for msg in error_messages:
        # Truncate long error messages for grouping
        key = msg[:100] if len(msg) > 100 else msg
        error_counts[key] = error_counts.get(key, 0) + 1
    result.error_summary = error_counts

    return result


# ---------------------------------------------------------------------------
# Crash recovery test
# ---------------------------------------------------------------------------

def crash_recovery_test(corpus_dir=None, config=None, poll_interval=2.0,
                        timeout=600):
    """Test worker crash recovery by killing a worker during processing.

    Submits a large document (100+ pages) that triggers chord fan-out,
    waits for processing to start, then reports instructions for manual
    worker kill (since Docker control requires host access).

    Args:
        corpus_dir: Directory for test PDF.
        config: Override configuration dict.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait.

    Returns:
        ScaleTestResult with crash recovery metrics.
    """
    _setup_django()
    from jobs.models import Job, PageResult
    from jobs.tasks import ingest_document

    config = config or dict(DEFAULT_CONFIG)
    config["mode"] = "crash-recovery"
    corpus_dir = corpus_dir or "scale_test_corpus"
    os.makedirs(corpus_dir, exist_ok=True)

    # Generate one large document (100 pages, triggers fan-out)
    large_pdf_path = os.path.join(corpus_dir, "crash_test_100p.pdf")
    pdf_bytes = _minimal_pdf_bytes(100)
    with open(large_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    result = ScaleTestResult(
        test_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode="crash-recovery",
        pipeline_version=__version__,
        config=config,
        total_jobs=1,
        total_pages=100,
    )

    print("Submitting 100-page document for crash recovery test...")
    job = Job.objects.create(
        source_file=large_pdf_path,
        status=Job.Status.SUBMITTED,
    )
    ingest_document.delay(str(job.job_id))

    start_time = time.monotonic()
    terminal_statuses = {"completed", "failed", "cancelled"}

    # Wait for processing to start
    print("Waiting for job to start processing...")
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            print("Timeout waiting for processing to start")
            result.error_summary = {"timeout": "Job never started processing"}
            return result

        job.refresh_from_db()
        if job.status in ("processing", "assembling"):
            pages_done = PageResult.objects.filter(
                job=job, status="completed"
            ).count()
            print(f"\n  Job is {job.status} ({pages_done} pages done)")
            print("  *** NOW: Kill a worker with 'docker kill <worker_container>' ***")
            print("  Monitoring for completion/retry...")
            break
        time.sleep(poll_interval)

    # Monitor until completion or timeout
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            print(f"\nTimeout after {timeout}s")
            break

        job.refresh_from_db()
        pages_done = PageResult.objects.filter(
            job=job, status="completed"
        ).count()
        print(f"\r  [{elapsed:.0f}s] Status: {job.status}, "
              f"Pages: {pages_done}/100", end="", flush=True)

        if job.status in terminal_statuses:
            break
        time.sleep(poll_interval)

    print()

    # Collect results
    result.duration_seconds = time.monotonic() - start_time
    job.refresh_from_db()

    if job.status == "completed":
        result.completed_jobs = 1
        result.pages_processed = PageResult.objects.filter(
            job=job, status="completed"
        ).count()
        print(f"  PASS: Job completed with {result.pages_processed}/100 pages")
    else:
        result.failed_jobs = 1
        result.error_summary = {"final_status": job.status,
                                "error": job.error_message}
        print(f"  RESULT: Job ended with status '{job.status}'")

    return result


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(result, results_dir=None):
    """Save a ScaleTestResult to JSON.

    Args:
        result: ScaleTestResult instance.
        results_dir: Output directory (default: SCALE_TEST_DIR).

    Returns:
        Path to the saved JSON file.
    """
    results_dir = results_dir or SCALE_TEST_DIR
    os.makedirs(results_dir, exist_ok=True)

    filename = f"scale_{result.mode}_{result.test_id}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return filepath


def load_results(filepath):
    """Load a ScaleTestResult from JSON.

    Args:
        filepath: Path to the JSON results file.

    Returns:
        ScaleTestResult instance.
    """
    with open(filepath) as f:
        data = json.load(f)

    # Reconstruct dataclass (ignore unknown keys for forward compat)
    known_fields = {f.name for f in ScaleTestResult.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return ScaleTestResult(**filtered)


def find_latest_result(results_dir=None):
    """Find the most recently saved result file.

    Args:
        results_dir: Directory to search (default: SCALE_TEST_DIR).

    Returns:
        Path to the latest result file, or None.
    """
    results_dir = results_dir or SCALE_TEST_DIR
    if not os.path.isdir(results_dir):
        return None

    files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.startswith("scale_") and f.endswith(".json")
    ]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(result):
    """Generate a human-readable text report from a ScaleTestResult.

    Args:
        result: ScaleTestResult instance.

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 72,
        f"  Scale Test Report: {result.test_id}",
        f"  Mode: {result.mode} | Version: {result.pipeline_version}",
        f"  Timestamp: {result.timestamp}",
        "=" * 72,
        "",
        "  Job Summary:",
        f"    Total Jobs:     {result.total_jobs}",
        f"    Completed:      {result.completed_jobs}",
        f"    Failed:         {result.failed_jobs}",
        f"    Cancelled:      {result.cancelled_jobs}",
        f"    Success Rate:   {_pct(result.completed_jobs, result.total_jobs)}",
        "",
        "  Page Summary:",
        f"    Total Pages:    {result.total_pages}",
        f"    Processed:      {result.pages_processed}",
        f"    Failed:         {result.pages_failed}",
        "",
        "  Performance:",
        f"    Duration:       {result.duration_seconds:.1f}s",
        f"    Fleet PPM:      {result.fleet_ppm:.1f}",
        f"    Avg Job Time:   {result.avg_job_duration_seconds:.2f}s",
        f"    P50 Job Time:   {result.p50_job_duration_seconds:.2f}s",
        f"    P95 Job Time:   {result.p95_job_duration_seconds:.2f}s",
        f"    P99 Job Time:   {result.p99_job_duration_seconds:.2f}s",
    ]

    # Corpus info
    if result.corpus_info:
        lines.extend([
            "",
            "  Corpus:",
            f"    Documents:      {result.corpus_info.get('total_documents', 'N/A')}",
            f"    Total Pages:    {result.corpus_info.get('total_pages', 'N/A')}",
            f"    Avg Pages/Doc:  {result.corpus_info.get('avg_pages_per_doc', 0):.1f}",
        ])
        if "fanout_count" in result.corpus_info:
            lines.append(
                f"    Fan-out Jobs:   {result.corpus_info['fanout_count']} "
                f"(>{FANOUT_THRESHOLD} pages)"
            )

    # Per-worker stats
    if result.per_worker_stats:
        lines.extend(["", "  Per-Worker Stats:"])
        lines.append(f"    {'Hostname':<30} {'Pages':>8} {'PPM':>8} {'Avg ms':>8}")
        lines.append(f"    {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        for ws in result.per_worker_stats:
            hostname = ws.get("hostname", "unknown")
            pages = ws.get("pages_processed", 0)
            ppm = ws.get("ppm", 0.0)
            avg_ms = ws.get("avg_page_time_ms", 0.0)
            lines.append(f"    {hostname:<30} {pages:>8} {ppm:>8.1f} {avg_ms:>8.1f}")

    # Error summary
    if result.error_summary:
        lines.extend(["", "  Errors:"])
        for err_type, count in result.error_summary.items():
            lines.append(f"    {err_type}: {count}")

    lines.extend(["", "=" * 72])
    return "\n".join(lines)


def compare_runs(path1, path2):
    """Compare two scale test results side-by-side.

    Args:
        path1: Path to baseline result JSON.
        path2: Path to current result JSON.

    Returns:
        Formatted comparison string.
    """
    r1 = load_results(path1)
    r2 = load_results(path2)

    def _delta(v1, v2, unit="", higher_is_better=True):
        if v1 == 0:
            return "N/A"
        diff = v2 - v1
        pct = (diff / v1) * 100 if v1 != 0 else 0
        direction = "BETTER" if (diff > 0) == higher_is_better else "WORSE"
        return f"{v2:.1f}{unit} ({diff:+.1f}, {pct:+.1f}% {direction})"

    lines = [
        "=" * 72,
        "  Scale Test Comparison",
        f"  Baseline: {r1.test_id} ({r1.mode})",
        f"  Current:  {r2.test_id} ({r2.mode})",
        "=" * 72,
        "",
        f"  {'Metric':<30} {'Baseline':>12} {'Current':>30}",
        f"  {'-'*30} {'-'*12} {'-'*30}",
    ]

    rows = [
        ("Total Jobs", r1.total_jobs,
         str(r2.total_jobs)),
        ("Completed Jobs", r1.completed_jobs,
         str(r2.completed_jobs)),
        ("Failed Jobs", r1.failed_jobs,
         str(r2.failed_jobs)),
        ("Total Pages", r1.total_pages,
         str(r2.total_pages)),
        ("Fleet PPM", r1.fleet_ppm,
         _delta(r1.fleet_ppm, r2.fleet_ppm, "", higher_is_better=True)),
        ("Avg Job Duration (s)", r1.avg_job_duration_seconds,
         _delta(r1.avg_job_duration_seconds, r2.avg_job_duration_seconds, "s",
                higher_is_better=False)),
        ("P95 Job Duration (s)", r1.p95_job_duration_seconds,
         _delta(r1.p95_job_duration_seconds, r2.p95_job_duration_seconds, "s",
                higher_is_better=False)),
    ]

    for label, baseline_val, current_str in rows:
        if isinstance(baseline_val, float):
            lines.append(f"  {label:<30} {baseline_val:>12.1f} {current_str:>30}")
        else:
            lines.append(f"  {label:<30} {str(baseline_val):>12} {current_str:>30}")

    lines.extend(["", "=" * 72])
    return "\n".join(lines)


def _pct(numerator, denominator):
    """Format a percentage string."""
    if denominator == 0:
        return "N/A"
    return f"{(numerator / denominator) * 100:.1f}%"


def generate_all_reports(results_dir=None):
    """Generate a summary report of all saved results.

    Args:
        results_dir: Directory containing result files.

    Returns:
        Formatted string with all results.
    """
    results_dir = results_dir or SCALE_TEST_DIR
    if not os.path.isdir(results_dir):
        return "No results directory found."

    files = sorted([
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.startswith("scale_") and f.endswith(".json")
    ], key=os.path.getmtime)

    if not files:
        return "No scale test results found."

    lines = [
        "=" * 72,
        f"  Scale Test Results Summary ({len(files)} runs)",
        "=" * 72,
        "",
        f"  {'ID':<14} {'Mode':<16} {'Jobs':>6} {'Pages':>8} "
        f"{'PPM':>8} {'Success':>8} {'Duration':>10}",
        f"  {'-'*14} {'-'*16} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}",
    ]

    for fp in files:
        r = load_results(fp)
        success = _pct(r.completed_jobs, r.total_jobs)
        lines.append(
            f"  {r.test_id:<14} {r.mode:<16} {r.total_jobs:>6} "
            f"{r.total_pages:>8} {r.fleet_ppm:>8.1f} {success:>8} "
            f"{r.duration_seconds:>9.1f}s"
        )

    lines.extend(["", "=" * 72])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_scale_test(mode="simulate", num_jobs=20, num_workers=1, config=None,
                   seed=None, results_dir=None):
    """Main entry point for running a scale test.

    Args:
        mode: "simulate", "live", or "crash-recovery".
        num_jobs: Number of jobs to submit.
        num_workers: Number of simulated workers (simulate mode).
        config: Configuration overrides.
        seed: Random seed for reproducibility.
        results_dir: Output directory for results.

    Returns:
        ScaleTestResult instance.
    """
    print(f"Starting {mode} scale test ({num_jobs} jobs, {num_workers} workers)...")

    if mode == "simulate":
        result = simulate_distributed(
            num_jobs=num_jobs, num_workers=num_workers,
            config=config, seed=seed,
        )
    elif mode == "live":
        result = live_distributed(
            num_jobs=num_jobs, config=config, seed=seed,
        )
    elif mode == "crash-recovery":
        result = crash_recovery_test(config=config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simulate', 'live', or 'crash-recovery'.")

    filepath = save_results(result, results_dir=results_dir)
    print(f"\nResults saved to: {filepath}")
    print(f"  Fleet PPM:      {result.fleet_ppm:.1f}")
    print(f"  Jobs OK/Fail:   {result.completed_jobs}/{result.failed_jobs}")
    print(f"  Pages:          {result.pages_processed}")
    print(f"  Duration:       {result.duration_seconds:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="EDCOCR Distributed Pipeline Scale Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["simulate", "live", "crash-recovery"],
        default="simulate",
        help="Test mode: simulate (no stack), live (real coordinator), "
             "crash-recovery (worker kill test)",
    )
    parser.add_argument(
        "--jobs", type=int, default=20,
        help="Number of jobs to submit (default: 20)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of simulated workers (simulate mode only, default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible tests",
    )
    parser.add_argument(
        "--report", nargs="?", const="all", default=None,
        help="Display report. Use 'latest' for most recent run.",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("RUN1", "RUN2"),
        help="Compare two scale test result files",
    )
    parser.add_argument(
        "--results-dir", type=str, default=SCALE_TEST_DIR,
        help="Directory for test results",
    )

    args = parser.parse_args()

    # Report mode
    if args.report is not None:
        if args.report == "latest":
            latest = find_latest_result(args.results_dir)
            if latest:
                r = load_results(latest)
                print(generate_report(r))
            else:
                print("No scale test results found.")
        else:
            print(generate_all_reports(args.results_dir))
        return

    # Compare mode
    if args.compare:
        print(compare_runs(args.compare[0], args.compare[1]))
        return

    # Run test
    run_scale_test(
        mode=args.mode,
        num_jobs=args.jobs,
        num_workers=args.workers,
        seed=args.seed,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
