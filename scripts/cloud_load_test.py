"""Cloud load testing harness for OCR-Local pipeline.

Submits concurrent OCR jobs against a cloud-deployed API endpoint,
measures throughput, latency percentiles, and optionally validates
KEDA autoscaling behavior via the Kubernetes API.

Usage:
    python scripts/cloud_load_test.py \\
        --api-url https://ocr.example.com \\
        --api-key my-api-key \\
        --num-jobs 100 \\
        --concurrency 10 \\
        --duration-minutes 5

    # With KEDA validation (requires kubectl configured):
    python scripts/cloud_load_test.py \\
        --api-url https://ocr.example.com \\
        --api-key my-api-key \\
        --num-jobs 500 \\
        --validate-keda \\
        --keda-namespace ocr-local \\
        --keda-max-pods 20

This module can also be imported and used programmatically:

    from scripts.cloud_load_test import CloudLoadTester
    tester = CloudLoadTester("https://ocr.example.com", "my-key", concurrency=50)
    tester.run_load_test(num_jobs=1000, duration_minutes=10)
    report = tester.generate_report()
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic PDF generator (minimal valid PDF for load testing)
# ---------------------------------------------------------------------------

_MINIMAL_PDF = (
    b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
)


def _create_synthetic_pdf(page_count: int = 1) -> bytes:
    """Return minimal valid PDF bytes for load testing.

    For simplicity, the generated PDF always has a single blank page
    regardless of page_count (the parameter is accepted for interface
    compatibility but does not affect content).
    """
    return _MINIMAL_PDF


# ---------------------------------------------------------------------------
# Job result tracking
# ---------------------------------------------------------------------------


class JobResult:
    """Tracks the outcome of a single load-test job submission."""

    __slots__ = (
        "job_id", "status", "submit_time", "response_time",
        "http_status", "error",
    )

    def __init__(self):
        self.job_id: str = ""
        self.status: str = "pending"
        self.submit_time: float = 0.0
        self.response_time: float = 0.0
        self.http_status: int = 0
        self.error: Optional[str] = None


# ---------------------------------------------------------------------------
# CloudLoadTester
# ---------------------------------------------------------------------------


class CloudLoadTester:
    """Load testing harness for cloud-deployed OCR pipeline.

    Parameters
    ----------
    api_url : str
        Base URL of the OCR API (e.g. ``https://ocr.example.com``).
    api_key : str
        API key for authentication (sent as ``X-API-Key`` header).
    concurrency : int
        Maximum number of concurrent job submissions.
    request_timeout : int
        HTTP request timeout in seconds for each submission.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        concurrency: int = 50,
        request_timeout: int = 60,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.concurrency = max(1, concurrency)
        self.request_timeout = request_timeout

        self._results: list[JobResult] = []
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._stopped = threading.Event()

    # -- Internal helpers ---------------------------------------------------

    def _submit_job(self, job_index: int) -> JobResult:
        """Submit a single OCR job and record the result."""
        result = JobResult()
        pdf_data = _create_synthetic_pdf()

        boundary = f"----LoadTestBoundary{job_index:08d}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; '
            f'filename="loadtest_{job_index:06d}.pdf"\r\n'
            f"Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8")
        body += pdf_data
        body += f"\r\n--{boundary}--\r\n".encode("utf-8")

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }

        url = f"{self.api_url}/api/v1/jobs"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        result.submit_time = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                result.http_status = resp.getcode()
                resp_data = json.loads(resp.read().decode("utf-8"))
                result.job_id = resp_data.get("job_id", "")
                result.status = "submitted"
        except urllib.error.HTTPError as exc:
            result.http_status = exc.code
            result.error = f"HTTP {exc.code}: {exc.reason}"
            result.status = "error"
        except (urllib.error.URLError, OSError) as exc:
            result.error = str(exc)[:200]
            result.status = "error"

        result.response_time = time.monotonic() - result.submit_time

        with self._lock:
            self._results.append(result)

        return result

    # -- Public API ---------------------------------------------------------

    def run_load_test(
        self,
        num_jobs: int,
        duration_minutes: int = 10,
    ) -> list[JobResult]:
        """Submit jobs and measure throughput and latency.

        Parameters
        ----------
        num_jobs : int
            Total number of jobs to submit.
        duration_minutes : int
            Maximum duration in minutes (test stops if exceeded even
            if not all jobs have been submitted).

        Returns
        -------
        list of JobResult
            Results for each submitted job.
        """
        self._results = []
        self._stopped.clear()
        self._start_time = time.monotonic()
        deadline = self._start_time + (duration_minutes * 60)

        submitted = 0
        logger.info(
            "Starting load test: %d jobs, concurrency=%d, timeout=%d min",
            num_jobs, self.concurrency, duration_minutes,
        )

        with ThreadPoolExecutor(
            max_workers=self.concurrency,
            thread_name_prefix="loadtest",
        ) as pool:
            futures = []
            for i in range(num_jobs):
                if time.monotonic() >= deadline or self._stopped.is_set():
                    break
                futures.append(pool.submit(self._submit_job, i))
                submitted += 1

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.warning("Job submission raised: %s", exc)

        self._end_time = time.monotonic()
        logger.info(
            "Load test complete: %d/%d jobs submitted in %.1f seconds",
            submitted, num_jobs, self._end_time - self._start_time,
        )
        return list(self._results)

    def validate_keda_scaling(
        self,
        expected_max_pods: int,
        timeout: int = 90,
        namespace: str = "default",
        deployment_name: str = "gpu-worker",
    ) -> dict:
        """Verify KEDA scaled workers within timeout.

        Uses ``kubectl`` to query pod counts. Requires kubectl to be
        configured and accessible.

        Parameters
        ----------
        expected_max_pods : int
            Expected maximum pod count after scaling.
        timeout : int
            Seconds to wait for scaling to occur.
        namespace : str
            Kubernetes namespace for the worker deployment.
        deployment_name : str
            Name (or label selector) for the worker deployment.

        Returns
        -------
        dict
            Scaling validation results including observed pod counts
            and whether the target was reached.
        """
        import subprocess

        start = time.monotonic()
        max_observed = 0
        observations: list[dict] = []

        logger.info(
            "Validating KEDA scaling: expecting up to %d pods in %s/%s within %ds",
            expected_max_pods, namespace, deployment_name, timeout,
        )

        while time.monotonic() - start < timeout:
            try:
                cmd = [
                    "kubectl", "get", "pods",
                    "-n", namespace,
                    "-l", f"app.kubernetes.io/component={deployment_name}",
                    "--no-headers",
                ]
                output = subprocess.check_output(
                    cmd, text=True, timeout=10, stderr=subprocess.DEVNULL,
                )
                running = sum(
                    1 for line in output.strip().split("\n")
                    if line and "Running" in line
                )
            except (subprocess.SubprocessError, OSError):
                running = -1

            max_observed = max(max_observed, running)
            observations.append({
                "elapsed_seconds": round(time.monotonic() - start, 1),
                "running_pods": running,
            })

            if running >= expected_max_pods:
                break

            time.sleep(5)

        result = {
            "target_pods": expected_max_pods,
            "max_observed_pods": max_observed,
            "target_reached": max_observed >= expected_max_pods,
            "elapsed_seconds": round(time.monotonic() - start, 1),
            "observations": observations,
        }

        if result["target_reached"]:
            logger.info(
                "KEDA scaling validated: reached %d pods in %.1fs",
                max_observed, result["elapsed_seconds"],
            )
        else:
            logger.warning(
                "KEDA scaling did not reach target: observed %d/%d pods in %.1fs",
                max_observed, expected_max_pods, result["elapsed_seconds"],
            )

        return result

    def stop(self) -> None:
        """Signal the load test to stop early."""
        self._stopped.set()

    def generate_report(self) -> dict:
        """Generate load test report with metrics.

        Returns
        -------
        dict
            Report containing summary statistics, latency percentiles,
            status distribution, and per-second throughput.
        """
        if not self._results:
            return {"error": "No results to report"}

        total = len(self._results)
        successful = [r for r in self._results if r.status == "submitted"]
        failed = [r for r in self._results if r.status == "error"]
        response_times = [r.response_time for r in self._results]

        # Status code distribution
        status_counts: dict[int, int] = defaultdict(int)
        for r in self._results:
            if r.http_status:
                status_counts[r.http_status] += 1

        # Latency percentiles
        sorted_times = sorted(response_times)
        duration = self._end_time - self._start_time if self._end_time else 0.0

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_url": self.api_url,
            "concurrency": self.concurrency,
            "summary": {
                "total_jobs": total,
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": round(len(successful) / total * 100, 2) if total else 0,
                "duration_seconds": round(duration, 2),
                "throughput_jobs_per_second": round(total / duration, 2) if duration > 0 else 0,
            },
            "latency": {
                "min_ms": round(min(response_times) * 1000, 1) if response_times else 0,
                "max_ms": round(max(response_times) * 1000, 1) if response_times else 0,
                "mean_ms": round(statistics.mean(response_times) * 1000, 1) if response_times else 0,
                "median_ms": round(statistics.median(response_times) * 1000, 1) if response_times else 0,
                "p95_ms": round(sorted_times[int(len(sorted_times) * 0.95)] * 1000, 1) if sorted_times else 0,
                "p99_ms": round(sorted_times[int(len(sorted_times) * 0.99)] * 1000, 1) if sorted_times else 0,
                "stdev_ms": round(statistics.stdev(response_times) * 1000, 1) if len(response_times) > 1 else 0,
            },
            "status_codes": dict(sorted(status_counts.items())),
            "errors": [
                {"job_index": i, "error": r.error}
                for i, r in enumerate(self._results)
                if r.error
            ][:20],  # Cap error list at 20 entries
        }

        return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Run cloud load test from the command line."""
    parser = argparse.ArgumentParser(
        description="Cloud load testing for OCR-Local pipeline",
    )
    parser.add_argument(
        "--api-url", required=True,
        help="Base URL of the OCR API",
    )
    parser.add_argument(
        "--api-key", required=True,
        help="API key for authentication",
    )
    parser.add_argument(
        "--num-jobs", type=int, default=100,
        help="Number of jobs to submit (default: 100)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Maximum concurrent submissions (default: 10)",
    )
    parser.add_argument(
        "--duration-minutes", type=int, default=10,
        help="Maximum test duration in minutes (default: 10)",
    )
    parser.add_argument(
        "--request-timeout", type=int, default=60,
        help="HTTP timeout per request in seconds (default: 60)",
    )
    parser.add_argument(
        "--validate-keda", action="store_true",
        help="Validate KEDA autoscaling after load test",
    )
    parser.add_argument(
        "--keda-namespace", default="default",
        help="Kubernetes namespace for KEDA validation (default: default)",
    )
    parser.add_argument(
        "--keda-deployment", default="gpu-worker",
        help="Worker deployment name for KEDA validation (default: gpu-worker)",
    )
    parser.add_argument(
        "--keda-max-pods", type=int, default=10,
        help="Expected max pods for KEDA validation (default: 10)",
    )
    parser.add_argument(
        "--keda-timeout", type=int, default=90,
        help="Timeout in seconds for KEDA scaling (default: 90)",
    )
    parser.add_argument(
        "--output", "-o", default="-",
        help="Output file for report JSON (default: stdout)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tester = CloudLoadTester(
        api_url=args.api_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
    )

    tester.run_load_test(
        num_jobs=args.num_jobs,
        duration_minutes=args.duration_minutes,
    )

    report = tester.generate_report()

    if args.validate_keda:
        keda_result = tester.validate_keda_scaling(
            expected_max_pods=args.keda_max_pods,
            timeout=args.keda_timeout,
            namespace=args.keda_namespace,
            deployment_name=args.keda_deployment,
        )
        report["keda_scaling"] = keda_result

    report_json = json.dumps(report, indent=2)

    if args.output == "-":
        print(report_json)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_json + "\n")
        logger.info("Report written to %s", args.output)

    # Exit with error code if success rate is below 90%
    success_rate = report.get("summary", {}).get("success_rate", 0)
    if success_rate < 90:
        logger.error("Load test failed: success rate %.1f%% < 90%%", success_rate)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
