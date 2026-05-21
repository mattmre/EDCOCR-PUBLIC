#!/usr/bin/env python3
"""Benchmark S3 presigned URL generation and access performance.

Measures presigned URL generation latency, upload/download throughput via
presigned URLs, and concurrent access patterns. Handles missing boto3/minio
gracefully and generates structured reports.

Usage:
    python scripts/benchmark_presigned_urls.py --endpoint http://minio:9000 \\
        --bucket test-bench --iterations 100

    python scripts/benchmark_presigned_urls.py --endpoint http://minio:9000 \\
        --bucket test-bench --concurrency 50 --output-dir results/

    python scripts/benchmark_presigned_urls.py --dry-run --iterations 50
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("benchmark_presigned_urls")

# ---------------------------------------------------------------------------
# Graceful dependency handling
# ---------------------------------------------------------------------------

_boto3 = None
_urllib3 = None

try:
    import boto3  # type: ignore[import-untyped]
    from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]

    _boto3 = boto3
except ImportError:
    pass

try:
    import urllib3  # type: ignore[import-untyped]

    _urllib3 = urllib3
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    """Percentile latency statistics."""

    count: int = 0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    stddev_ms: float = 0.0


@dataclass
class ThroughputResult:
    """Upload or download throughput measurement."""

    operation: str = ""
    total_bytes: int = 0
    total_time_s: float = 0.0
    throughput_mbps: float = 0.0
    requests_completed: int = 0
    requests_failed: int = 0


@dataclass
class ConcurrencyResult:
    """Concurrent access pattern measurement."""

    concurrency_level: int = 0
    latency: LatencyStats = field(default_factory=LatencyStats)
    success_count: int = 0
    failure_count: int = 0
    total_time_s: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete presigned URL benchmark report."""

    endpoint: str = ""
    bucket: str = ""
    iterations: int = 0
    timestamp: str = ""
    url_generation: LatencyStats = field(default_factory=LatencyStats)
    upload_throughput: ThroughputResult = field(default_factory=ThroughputResult)
    download_throughput: ThroughputResult = field(default_factory=ThroughputResult)
    concurrency_results: list = field(default_factory=list)
    dry_run: bool = False
    errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Latency computation helpers
# ---------------------------------------------------------------------------


def compute_latency_stats(timings_ms: list[float]) -> LatencyStats:
    """Compute percentile latency statistics from timing measurements.

    Parameters
    ----------
    timings_ms : list[float]
        List of latency measurements in milliseconds.

    Returns
    -------
    LatencyStats
        Computed statistics including p50, p95, p99.
    """
    if not timings_ms:
        return LatencyStats()

    sorted_t = sorted(timings_ms)
    n = len(sorted_t)

    return LatencyStats(
        count=n,
        mean_ms=round(statistics.mean(sorted_t), 3),
        min_ms=round(sorted_t[0], 3),
        max_ms=round(sorted_t[-1], 3),
        p50_ms=round(sorted_t[min(int(n * 0.50), n - 1)], 3),
        p95_ms=round(sorted_t[min(int(n * 0.95), n - 1)], 3),
        p99_ms=round(sorted_t[min(int(n * 0.99), n - 1)], 3),
        stddev_ms=round(statistics.stdev(sorted_t), 3) if n >= 2 else 0.0,
    )


# ---------------------------------------------------------------------------
# S3 client setup
# ---------------------------------------------------------------------------


def create_s3_client(
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str = "us-east-1",
):
    """Create an S3 client for presigned URL operations.

    Parameters
    ----------
    endpoint : str
        S3-compatible endpoint URL.
    access_key : str
        Access key ID.
    secret_key : str
        Secret access key.
    region : str
        AWS region (default: us-east-1).

    Returns
    -------
    boto3.client
        Configured S3 client.

    Raises
    ------
    ImportError
        If boto3 is not available.
    """
    if _boto3 is None:
        raise ImportError(
            "boto3 is required for presigned URL benchmarks. "
            "Install with: pip install boto3"
        )

    config = BotoConfig(
        signature_version="s3v4",
        retries={"max_attempts": 0},
    )

    return _boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=config,
    )


# ---------------------------------------------------------------------------
# Benchmark: URL generation latency
# ---------------------------------------------------------------------------


def benchmark_url_generation(
    s3_client,
    bucket: str,
    iterations: int,
) -> LatencyStats:
    """Benchmark presigned URL generation latency.

    Parameters
    ----------
    s3_client : boto3.client
        S3 client.
    bucket : str
        S3 bucket name.
    iterations : int
        Number of URLs to generate.

    Returns
    -------
    LatencyStats
        URL generation latency statistics.
    """
    timings = []

    for i in range(iterations):
        key = f"bench/test-{i:06d}.bin"

        start = time.perf_counter()
        s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)

    return compute_latency_stats(timings)


def benchmark_url_generation_dry_run(iterations: int) -> LatencyStats:
    """Simulate URL generation timing for dry-run mode.

    Parameters
    ----------
    iterations : int
        Number of simulated iterations.

    Returns
    -------
    LatencyStats
        Simulated latency statistics.
    """
    import random

    rng = random.Random(42)
    timings = [rng.gauss(0.1, 0.02) for _ in range(iterations)]
    timings = [max(0.01, t) for t in timings]
    return compute_latency_stats(timings)


# ---------------------------------------------------------------------------
# Benchmark: upload/download throughput
# ---------------------------------------------------------------------------

# Default test payload size (1 KB)
_DEFAULT_PAYLOAD_SIZE = 1024


def benchmark_upload_throughput(
    s3_client,
    bucket: str,
    iterations: int,
    payload_size: int = _DEFAULT_PAYLOAD_SIZE,
) -> ThroughputResult:
    """Benchmark upload throughput via presigned PUT URLs.

    Parameters
    ----------
    s3_client : boto3.client
        S3 client.
    bucket : str
        S3 bucket name.
    iterations : int
        Number of upload operations.
    payload_size : int
        Size of each upload payload in bytes.

    Returns
    -------
    ThroughputResult
        Upload throughput measurements.
    """
    if _urllib3 is None:
        return ThroughputResult(
            operation="upload",
            requests_failed=iterations,
        )

    http = _urllib3.PoolManager()
    payload = os.urandom(payload_size)
    completed = 0
    failed = 0
    total_bytes = 0

    start = time.perf_counter()

    for i in range(iterations):
        key = f"bench/upload-{i:06d}.bin"
        try:
            url = s3_client.generate_presigned_url(
                "put_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=3600,
            )
            resp = http.request("PUT", url, body=payload, timeout=30.0)
            if 200 <= resp.status < 300:
                completed += 1
                total_bytes += payload_size
            else:
                failed += 1
        except Exception as exc:
            logger.debug("Upload %d failed: %s", i, exc)
            failed += 1

    elapsed = time.perf_counter() - start
    throughput = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    return ThroughputResult(
        operation="upload",
        total_bytes=total_bytes,
        total_time_s=round(elapsed, 3),
        throughput_mbps=round(throughput, 3),
        requests_completed=completed,
        requests_failed=failed,
    )


def benchmark_download_throughput(
    s3_client,
    bucket: str,
    iterations: int,
    payload_size: int = _DEFAULT_PAYLOAD_SIZE,
) -> ThroughputResult:
    """Benchmark download throughput via presigned GET URLs.

    Assumes upload benchmark has already placed objects in the bucket.

    Parameters
    ----------
    s3_client : boto3.client
        S3 client.
    bucket : str
        S3 bucket name.
    iterations : int
        Number of download operations.
    payload_size : int
        Expected size of each download payload.

    Returns
    -------
    ThroughputResult
        Download throughput measurements.
    """
    if _urllib3 is None:
        return ThroughputResult(
            operation="download",
            requests_failed=iterations,
        )

    http = _urllib3.PoolManager()
    completed = 0
    failed = 0
    total_bytes = 0

    start = time.perf_counter()

    for i in range(iterations):
        key = f"bench/upload-{i:06d}.bin"
        try:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=3600,
            )
            resp = http.request("GET", url, timeout=30.0)
            if 200 <= resp.status < 300:
                completed += 1
                total_bytes += len(resp.data)
            else:
                failed += 1
        except Exception as exc:
            logger.debug("Download %d failed: %s", i, exc)
            failed += 1

    elapsed = time.perf_counter() - start
    throughput = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    return ThroughputResult(
        operation="download",
        total_bytes=total_bytes,
        total_time_s=round(elapsed, 3),
        throughput_mbps=round(throughput, 3),
        requests_completed=completed,
        requests_failed=failed,
    )


# ---------------------------------------------------------------------------
# Benchmark: concurrent access
# ---------------------------------------------------------------------------


def benchmark_concurrent_access(
    s3_client,
    bucket: str,
    concurrency: int,
    iterations_per_worker: int = 10,
) -> ConcurrencyResult:
    """Benchmark concurrent presigned URL generation and access.

    Parameters
    ----------
    s3_client : boto3.client
        S3 client.
    bucket : str
        S3 bucket name.
    concurrency : int
        Number of concurrent workers.
    iterations_per_worker : int
        Number of URL generations per worker.

    Returns
    -------
    ConcurrencyResult
        Concurrent access performance results.
    """
    timings = []
    success_count = 0
    failure_count = 0

    def _worker(worker_id: int) -> list[float]:
        worker_timings = []
        for i in range(iterations_per_worker):
            key = f"bench/concurrent-{worker_id:04d}-{i:04d}.bin"
            start = time.perf_counter()
            try:
                s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=3600,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                worker_timings.append(elapsed_ms)
            except Exception:
                elapsed_ms = (time.perf_counter() - start) * 1000
                worker_timings.append(-elapsed_ms)  # negative = failure
        return worker_timings

    overall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_worker, wid): wid
            for wid in range(concurrency)
        }
        for future in as_completed(futures):
            worker_timings = future.result()
            for t in worker_timings:
                if t >= 0:
                    timings.append(t)
                    success_count += 1
                else:
                    failure_count += 1

    overall_time = time.perf_counter() - overall_start

    return ConcurrencyResult(
        concurrency_level=concurrency,
        latency=compute_latency_stats(timings),
        success_count=success_count,
        failure_count=failure_count,
        total_time_s=round(overall_time, 3),
    )


def benchmark_concurrent_dry_run(
    concurrency: int,
    iterations_per_worker: int = 10,
) -> ConcurrencyResult:
    """Simulate concurrent access for dry-run mode.

    Parameters
    ----------
    concurrency : int
        Simulated concurrency level.
    iterations_per_worker : int
        Iterations per simulated worker.

    Returns
    -------
    ConcurrencyResult
        Simulated concurrency results.
    """
    import random

    rng = random.Random(concurrency)
    total = concurrency * iterations_per_worker
    timings = [rng.gauss(0.15 * (1 + concurrency / 100), 0.03) for _ in range(total)]
    timings = [max(0.01, t) for t in timings]

    return ConcurrencyResult(
        concurrency_level=concurrency,
        latency=compute_latency_stats(timings),
        success_count=total,
        failure_count=0,
        total_time_s=round(sum(timings) / 1000 / concurrency, 3),
    )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_benchmark_objects(
    s3_client,
    bucket: str,
    prefix: str = "bench/",
) -> int:
    """Remove benchmark objects from the S3 bucket.

    Parameters
    ----------
    s3_client : boto3.client
        S3 client.
    bucket : str
        S3 bucket name.
    prefix : str
        Object key prefix to clean up.

    Returns
    -------
    int
        Number of objects deleted.
    """
    deleted = 0
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
                deleted += 1
    except Exception as exc:
        logger.warning("Cleanup failed: %s", exc)
    return deleted


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def format_report_markdown(report: BenchmarkReport) -> str:
    """Format benchmark report as markdown.

    Parameters
    ----------
    report : BenchmarkReport
        Benchmark results.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# S3 Presigned URL Benchmark Report",
        "",
        f"**Endpoint**: {report.endpoint}",
        f"**Bucket**: {report.bucket}",
        f"**Iterations**: {report.iterations}",
        f"**Timestamp**: {report.timestamp}",
        f"**Dry Run**: {report.dry_run}",
        "",
        "## URL Generation Latency",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Count | {report.url_generation.count} |",
        f"| Mean | {report.url_generation.mean_ms:.3f} ms |",
        f"| Min | {report.url_generation.min_ms:.3f} ms |",
        f"| Max | {report.url_generation.max_ms:.3f} ms |",
        f"| p50 | {report.url_generation.p50_ms:.3f} ms |",
        f"| p95 | {report.url_generation.p95_ms:.3f} ms |",
        f"| p99 | {report.url_generation.p99_ms:.3f} ms |",
        f"| Std Dev | {report.url_generation.stddev_ms:.3f} ms |",
        "",
    ]

    if not report.dry_run:
        lines.extend([
            "## Upload Throughput",
            "",
            f"- Completed: {report.upload_throughput.requests_completed}",
            f"- Failed: {report.upload_throughput.requests_failed}",
            f"- Total bytes: {report.upload_throughput.total_bytes:,}",
            f"- Time: {report.upload_throughput.total_time_s:.3f} s",
            f"- Throughput: {report.upload_throughput.throughput_mbps:.3f} MB/s",
            "",
            "## Download Throughput",
            "",
            f"- Completed: {report.download_throughput.requests_completed}",
            f"- Failed: {report.download_throughput.requests_failed}",
            f"- Total bytes: {report.download_throughput.total_bytes:,}",
            f"- Time: {report.download_throughput.total_time_s:.3f} s",
            f"- Throughput: {report.download_throughput.throughput_mbps:.3f} MB/s",
            "",
        ])

    if report.concurrency_results:
        lines.extend([
            "## Concurrent Access",
            "",
            "| Concurrency | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Success | Failed |",
            "|-------------|-----------|----------|----------|----------|---------|--------|",
        ])
        for cr in report.concurrency_results:
            cr_data = cr if isinstance(cr, dict) else asdict(cr)
            lat = cr_data.get("latency", {})
            lines.append(
                f"| {cr_data.get('concurrency_level', 0)} "
                f"| {lat.get('mean_ms', 0):.3f} "
                f"| {lat.get('p50_ms', 0):.3f} "
                f"| {lat.get('p95_ms', 0):.3f} "
                f"| {lat.get('p99_ms', 0):.3f} "
                f"| {cr_data.get('success_count', 0)} "
                f"| {cr_data.get('failure_count', 0)} |"
            )
        lines.append("")

    if report.errors:
        lines.extend([
            "## Errors",
            "",
        ])
        for err in report.errors:
            lines.append(f"- {err}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    endpoint: str = "http://localhost:9000",
    bucket: str = "test-bench",
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    iterations: int = 100,
    concurrency_levels: Optional[list[int]] = None,
    dry_run: bool = False,
) -> BenchmarkReport:
    """Run the complete presigned URL benchmark suite.

    Parameters
    ----------
    endpoint : str
        S3-compatible endpoint URL.
    bucket : str
        Bucket name for benchmark objects.
    access_key : str, optional
        S3 access key (falls back to S3_ACCESS_KEY env var).
    secret_key : str, optional
        S3 secret key (falls back to S3_SECRET_KEY env var).
    iterations : int
        Number of iterations for each benchmark phase.
    concurrency_levels : list[int], optional
        Concurrency levels to test (default: [10, 50, 100]).
    dry_run : bool
        If True, simulate without actual S3 access.

    Returns
    -------
    BenchmarkReport
        Complete benchmark results.
    """
    if concurrency_levels is None:
        concurrency_levels = [10, 50, 100]

    report = BenchmarkReport(
        endpoint=endpoint,
        bucket=bucket,
        iterations=iterations,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        dry_run=dry_run,
    )

    if dry_run:
        logger.info("Running in dry-run mode (no S3 access)")
        report.url_generation = benchmark_url_generation_dry_run(iterations)

        for level in concurrency_levels:
            result = benchmark_concurrent_dry_run(level)
            report.concurrency_results.append(asdict(result))

        return report

    # Live benchmark
    access_key = access_key or os.environ.get("S3_ACCESS_KEY", "minioadmin")
    secret_key = secret_key or os.environ.get("S3_SECRET_KEY", "minioadmin")

    try:
        s3_client = create_s3_client(endpoint, access_key, secret_key)
    except ImportError as exc:
        report.errors.append(str(exc))
        return report

    # Phase 1: URL generation latency
    logger.info("Phase 1: URL generation latency (%d iterations)...", iterations)
    try:
        report.url_generation = benchmark_url_generation(
            s3_client, bucket, iterations
        )
    except Exception as exc:
        report.errors.append(f"URL generation failed: {exc}")

    # Phase 2: Upload throughput
    logger.info("Phase 2: Upload throughput (%d iterations)...", iterations)
    try:
        report.upload_throughput = benchmark_upload_throughput(
            s3_client, bucket, iterations
        )
    except Exception as exc:
        report.errors.append(f"Upload benchmark failed: {exc}")

    # Phase 3: Download throughput
    logger.info("Phase 3: Download throughput (%d iterations)...", iterations)
    try:
        report.download_throughput = benchmark_download_throughput(
            s3_client, bucket, iterations
        )
    except Exception as exc:
        report.errors.append(f"Download benchmark failed: {exc}")

    # Phase 4: Concurrent access
    for level in concurrency_levels:
        logger.info("Phase 4: Concurrent access (concurrency=%d)...", level)
        try:
            result = benchmark_concurrent_access(
                s3_client, bucket, level,
                iterations_per_worker=max(1, iterations // level),
            )
            report.concurrency_results.append(asdict(result))
        except Exception as exc:
            report.errors.append(f"Concurrency {level} failed: {exc}")

    # Cleanup
    logger.info("Cleaning up benchmark objects...")
    try:
        deleted = cleanup_benchmark_objects(s3_client, bucket)
        logger.info("Deleted %d benchmark objects", deleted)
    except Exception as exc:
        report.errors.append(f"Cleanup failed: {exc}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for presigned URL benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark S3 presigned URL performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_presigned_urls.py --dry-run --iterations 50
  python scripts/benchmark_presigned_urls.py --endpoint http://minio:9000 --bucket test
  python scripts/benchmark_presigned_urls.py --concurrency 10 50 100 --output-dir results/
        """,
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:9000",
        help="S3-compatible endpoint URL (default: http://localhost:9000)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="ocr-bench",
        help="S3 bucket name (default: ocr-bench)",
    )
    parser.add_argument(
        "--access-key",
        type=str,
        help="S3 access key (default: S3_ACCESS_KEY env var)",
    )
    parser.add_argument(
        "--secret-key",
        type=str,
        help="S3 secret key (default: S3_SECRET_KEY env var)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark phase (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[10, 50, 100],
        help="Concurrency levels to test (default: 10 50 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for output reports (JSON + markdown)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate benchmark without S3 access",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    report = run_benchmark(
        endpoint=args.endpoint,
        bucket=args.bucket,
        access_key=args.access_key,
        secret_key=args.secret_key,
        iterations=args.iterations,
        concurrency_levels=args.concurrency,
        dry_run=args.dry_run,
    )

    # Print markdown report
    md_report = format_report_markdown(report)
    print(md_report)

    # Save outputs
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "presigned-url-benchmark.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("JSON report saved to %s", json_path)

        md_path = out_dir / "presigned-url-benchmark.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        logger.info("Markdown report saved to %s", md_path)

    if report.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
