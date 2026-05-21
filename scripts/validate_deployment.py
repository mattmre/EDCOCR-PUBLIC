#!/usr/bin/env python3
"""Deployment validation pack for EDCOCR.

Validates a running EDCOCR deployment by checking health, submitting a
test job, verifying output contracts, and checking schema compliance.

Usage:
    python scripts/validate_deployment.py --base-url http://localhost:8000 --api-key MY_KEY
    python scripts/validate_deployment.py --base-url http://localhost:8000 --api-key MY_KEY --profile single-node
    python scripts/validate_deployment.py --base-url http://localhost:8000 --api-key MY_KEY --skip-job-submit
    python scripts/validate_deployment.py --base-url http://localhost:8000 --api-key MY_KEY --format markdown
"""

from __future__ import annotations

import argparse
import io
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFILE_SINGLE_NODE = "single-node"
PROFILE_MULTI_GPU = "multi-gpu"
PROFILE_KUBERNETES = "kubernetes"

ALL_PROFILES = [PROFILE_SINGLE_NODE, PROFILE_MULTI_GPU, PROFILE_KUBERNETES]

# All 14 expected schema types from schemas/__init__.py
EXPECTED_SCHEMA_TYPES = (
    "ocr_text",
    "searchable_pdf",
    "structure",
    "entities",
    "ner",
    "extraction",
    "classification",
    "validation",
    "handwriting",
    "signature",
    "vertical",
    "custody",
    "retrieval",
    "output_manifest",
)

DEFAULT_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2

# Minimal valid PDF content for test job submission.  This is a simple
# 1-page PDF that renders the text "Deployment validation test".
_MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
    b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"5 0 obj<</Length 44>>stream\nBT /F1 12 Tf 100 700 Td "
    b"(Deployment validation test) Tj ET\nendstream\nendobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000340 00000 n \n"
    b"trailer<</Size 6/Root 1 0 R>>\n"
    b"startxref\n434\n%%EOF\n"
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class DeploymentReport:
    """Aggregated deployment validation report."""

    base_url: str = ""
    profile: str = PROFILE_SINGLE_NODE
    timestamp: str = ""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only, optional requests fallback)
# ---------------------------------------------------------------------------

_session = None  # type: ignore[assignment]


def _have_requests() -> bool:
    """Check if the requests library is available."""
    try:
        import requests  # noqa: F401
        return True
    except ImportError:
        return False


_skip_tls_verify = False  # Set via --skip-tls-verify CLI flag


def _make_ssl_context() -> ssl.SSLContext:
    """Create an SSL context respecting the --skip-tls-verify flag."""
    if _skip_tls_verify:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context()


def _http_get(
    url: str,
    api_key: str,
    timeout: int = 30,
) -> tuple[int, dict | None, dict[str, str]]:
    """Perform an HTTP GET and return (status_code, json_body, headers).

    Uses ``requests`` if available, otherwise falls back to ``urllib``.
    """
    if _have_requests():
        import requests

        kwargs: dict = {
            "headers": {"X-API-Key": api_key},
            "timeout": timeout,
        }
        if _skip_tls_verify:
            kwargs["verify"] = False

        resp = requests.get(url, **kwargs)
        try:
            body = resp.json()
        except Exception:
            body = None
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.status_code, body, headers

    # stdlib fallback
    req = urllib.request.Request(url, headers={"X-API-Key": api_key})
    ctx = _make_ssl_context()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        raw = resp.read()
        try:
            body = json.loads(raw)
        except Exception:
            body = None
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.status, body, headers
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            body = json.loads(raw)
        except Exception:
            body = None
        headers = {k.lower(): v for k, v in e.headers.items()}
        return e.code, body, headers


def _http_post_multipart(
    url: str,
    api_key: str,
    file_bytes: bytes,
    filename: str = "test.pdf",
    timeout: int = 30,
) -> tuple[int, dict | None, dict[str, str]]:
    """POST a file via multipart/form-data.

    Uses ``requests`` if available, otherwise falls back to manual
    multipart encoding with ``urllib``.
    """
    if _have_requests():
        import requests

        files = {"file": (filename, io.BytesIO(file_bytes), "application/pdf")}
        kwargs: dict = {
            "headers": {"X-API-Key": api_key},
            "files": files,
            "timeout": timeout,
        }
        if _skip_tls_verify:
            kwargs["verify"] = False

        resp = requests.post(url, **kwargs)
        try:
            body = resp.json()
        except Exception:
            body = None
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.status_code, body, headers

    # stdlib multipart fallback
    boundary = "----DeploymentValidation9876543210"
    body_parts = []
    body_parts.append(f"--{boundary}".encode())
    body_parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode()
    )
    body_parts.append(b"Content-Type: application/pdf")
    body_parts.append(b"")
    body_parts.append(file_bytes)
    body_parts.append(f"--{boundary}--".encode())
    body_bytes = b"\r\n".join(body_parts)

    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={
            "X-API-Key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    ctx = _make_ssl_context()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        raw = resp.read()
        try:
            resp_body = json.loads(raw)
        except Exception:
            resp_body = None
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.status, resp_body, headers
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            resp_body = json.loads(raw)
        except Exception:
            resp_body = None
        headers = {k.lower(): v for k, v in e.headers.items()}
        return e.code, resp_body, headers


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_health(base_url: str, api_key: str) -> CheckResult:
    """Probe GET /api/v1/health and verify 200 response."""
    t0 = time.monotonic()
    try:
        status, body, _headers = _http_get(f"{base_url}/api/v1/health", api_key)
        ms = (time.monotonic() - t0) * 1000
        if status != 200:
            return CheckResult(
                name="health_check",
                passed=False,
                detail=f"Expected 200, got {status}",
                duration_ms=round(ms, 1),
            )
        if body and body.get("status") == "healthy":
            return CheckResult(
                name="health_check",
                passed=True,
                detail=f"Status: healthy, version: {body.get('version', 'unknown')}",
                duration_ms=round(ms, 1),
            )
        return CheckResult(
            name="health_check",
            passed=True,
            detail=f"Status: {body.get('status', 'unknown') if body else 'no body'}",
            duration_ms=round(ms, 1),
        )
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="health_check",
            passed=False,
            detail=f"Connection failed: {e}",
            duration_ms=round(ms, 1),
        )


def check_detailed_health(
    base_url: str,
    api_key: str,
    profile: str = PROFILE_SINGLE_NODE,
) -> CheckResult:
    """Probe GET /api/v1/health/detailed and verify subsystem status."""
    t0 = time.monotonic()
    try:
        status, body, _headers = _http_get(
            f"{base_url}/api/v1/health/detailed", api_key
        )
        ms = (time.monotonic() - t0) * 1000
        if status != 200:
            return CheckResult(
                name="detailed_health_check",
                passed=False,
                detail=f"Expected 200, got {status}",
                duration_ms=round(ms, 1),
            )
        if not body:
            return CheckResult(
                name="detailed_health_check",
                passed=False,
                detail="Empty response body",
                duration_ms=round(ms, 1),
            )

        overall = body.get("status", "unknown")
        checks = body.get("checks", {})
        parts = [f"overall={overall}"]
        for name, info in checks.items():
            sub_status = info.get("status", "unknown") if isinstance(info, dict) else str(info)
            parts.append(f"{name}={sub_status}")

        # Profile-specific checks
        if profile == PROFILE_MULTI_GPU:
            # For multi-GPU, we just verify detailed health is accessible
            # (GPU-specific subsystems would appear in checks if available)
            pass
        elif profile == PROFILE_KUBERNETES:
            # For k8s, we just verify detailed health works
            pass

        passed = overall in ("healthy", "degraded")
        return CheckResult(
            name="detailed_health_check",
            passed=passed,
            detail="; ".join(parts),
            duration_ms=round(ms, 1),
        )
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="detailed_health_check",
            passed=False,
            detail=f"Connection failed: {e}",
            duration_ms=round(ms, 1),
        )


def check_schema_list(base_url: str, api_key: str) -> CheckResult:
    """Verify GET /api/v1/schemas returns all expected schema types."""
    t0 = time.monotonic()
    try:
        status, body, _headers = _http_get(f"{base_url}/api/v1/schemas", api_key)
        ms = (time.monotonic() - t0) * 1000
        if status != 200:
            return CheckResult(
                name="schema_list",
                passed=False,
                detail=f"Expected 200, got {status}",
                duration_ms=round(ms, 1),
            )
        if not body or "schemas" not in body:
            return CheckResult(
                name="schema_list",
                passed=False,
                detail="Response missing 'schemas' field",
                duration_ms=round(ms, 1),
            )

        returned_types = {s.get("output_type") for s in body["schemas"]}
        expected = set(EXPECTED_SCHEMA_TYPES)
        missing = expected - returned_types
        if missing:
            return CheckResult(
                name="schema_list",
                passed=False,
                detail=f"Missing schema types: {sorted(missing)}",
                duration_ms=round(ms, 1),
            )

        return CheckResult(
            name="schema_list",
            passed=True,
            detail=f"All {len(expected)} schema types present",
            duration_ms=round(ms, 1),
        )
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="schema_list",
            passed=False,
            detail=f"Connection failed: {e}",
            duration_ms=round(ms, 1),
        )


def check_schema_retrieval(base_url: str, api_key: str) -> list[CheckResult]:
    """Verify GET /api/v1/schemas/{type} returns valid JSON Schema for each type."""
    results = []
    for schema_type in EXPECTED_SCHEMA_TYPES:
        t0 = time.monotonic()
        name = f"schema_get_{schema_type}"
        try:
            status, body, _headers = _http_get(
                f"{base_url}/api/v1/schemas/{schema_type}", api_key
            )
            ms = (time.monotonic() - t0) * 1000
            if status != 200:
                results.append(CheckResult(
                    name=name,
                    passed=False,
                    detail=f"Expected 200, got {status}",
                    duration_ms=round(ms, 1),
                ))
                continue

            if not body or not isinstance(body, dict):
                results.append(CheckResult(
                    name=name,
                    passed=False,
                    detail="Response is not a valid JSON object",
                    duration_ms=round(ms, 1),
                ))
                continue

            # Minimal JSON Schema validation: must have at least "type" or
            # "$schema" or "properties"
            has_schema_markers = any(
                k in body for k in ("type", "$schema", "properties", "title")
            )
            if not has_schema_markers:
                results.append(CheckResult(
                    name=name,
                    passed=False,
                    detail="Response lacks JSON Schema markers (type/$schema/properties/title)",
                    duration_ms=round(ms, 1),
                ))
                continue

            results.append(CheckResult(
                name=name,
                passed=True,
                detail=f"Valid schema: title={body.get('title', 'N/A')}",
                duration_ms=round(ms, 1),
            ))
        except Exception as e:
            ms = (time.monotonic() - t0) * 1000
            results.append(CheckResult(
                name=name,
                passed=False,
                detail=f"Error: {e}",
                duration_ms=round(ms, 1),
            ))
    return results


def check_version_header(base_url: str, api_key: str) -> CheckResult:
    """Verify X-API-Version header is present in responses."""
    t0 = time.monotonic()
    try:
        status, _body, headers = _http_get(f"{base_url}/api/v1/health", api_key)
        ms = (time.monotonic() - t0) * 1000
        version = headers.get("x-api-version")
        if not version:
            return CheckResult(
                name="version_header",
                passed=False,
                detail="X-API-Version header not found in response",
                duration_ms=round(ms, 1),
            )
        return CheckResult(
            name="version_header",
            passed=True,
            detail=f"X-API-Version: {version}",
            duration_ms=round(ms, 1),
        )
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="version_header",
            passed=False,
            detail=f"Connection failed: {e}",
            duration_ms=round(ms, 1),
        )


def check_job_submission(
    base_url: str,
    api_key: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[CheckResult]:
    """Submit a test job, poll for completion, and verify output manifest.

    Returns multiple CheckResult entries covering submission, completion,
    and output validation.
    """
    results = []

    # --- Step 1: Submit job ---
    t0 = time.monotonic()
    try:
        status, body, _headers = _http_post_multipart(
            f"{base_url}/api/v1/jobs",
            api_key,
            _MINIMAL_PDF_BYTES,
            filename="deployment_test.pdf",
            timeout=30,
        )
        ms = (time.monotonic() - t0) * 1000
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        results.append(CheckResult(
            name="job_submit",
            passed=False,
            detail=f"Submission failed: {e}",
            duration_ms=round(ms, 1),
        ))
        return results

    if status not in (200, 201, 202):
        results.append(CheckResult(
            name="job_submit",
            passed=False,
            detail=f"Expected 200/201/202, got {status}: {body}",
            duration_ms=round(ms, 1),
        ))
        return results

    job_id = None
    if body:
        job_id = body.get("job_id") or body.get("id")
    if not job_id:
        results.append(CheckResult(
            name="job_submit",
            passed=False,
            detail=f"No job_id in response: {body}",
            duration_ms=round(ms, 1),
        ))
        return results

    results.append(CheckResult(
        name="job_submit",
        passed=True,
        detail=f"Job submitted: {job_id}",
        duration_ms=round(ms, 1),
    ))

    # --- Step 2: Poll for completion ---
    t0 = time.monotonic()
    deadline = time.monotonic() + timeout
    final_status = None
    while time.monotonic() < deadline:
        try:
            s, b, _h = _http_get(f"{base_url}/api/v1/jobs/{job_id}", api_key)
            if s == 200 and b:
                final_status = b.get("status")
                if final_status in ("completed", "failed"):
                    break
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_SECONDS)

    ms = (time.monotonic() - t0) * 1000
    if final_status == "completed":
        results.append(CheckResult(
            name="job_completion",
            passed=True,
            detail=f"Job completed in {ms / 1000:.1f}s",
            duration_ms=round(ms, 1),
        ))
    elif final_status == "failed":
        results.append(CheckResult(
            name="job_completion",
            passed=False,
            detail="Job failed",
            duration_ms=round(ms, 1),
        ))
        return results
    else:
        results.append(CheckResult(
            name="job_completion",
            passed=False,
            detail=f"Job did not complete within {timeout}s (status: {final_status})",
            duration_ms=round(ms, 1),
        ))
        return results

    # --- Step 3: Verify output manifest ---
    t0 = time.monotonic()
    try:
        s, manifest, _h = _http_get(
            f"{base_url}/api/v1/jobs/{job_id}/outputs", api_key
        )
        ms = (time.monotonic() - t0) * 1000
        if s != 200:
            results.append(CheckResult(
                name="job_output_manifest",
                passed=False,
                detail=f"Expected 200, got {s}",
                duration_ms=round(ms, 1),
            ))
            return results

        if not manifest or "artifacts" not in manifest:
            results.append(CheckResult(
                name="job_output_manifest",
                passed=False,
                detail="Manifest missing 'artifacts' field",
                duration_ms=round(ms, 1),
            ))
            return results

        artifact_types = [a.get("output_type") for a in manifest["artifacts"]]
        results.append(CheckResult(
            name="job_output_manifest",
            passed=True,
            detail=f"Found {len(artifact_types)} artifacts: {artifact_types}",
            duration_ms=round(ms, 1),
        ))
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        results.append(CheckResult(
            name="job_output_manifest",
            passed=False,
            detail=f"Manifest retrieval failed: {e}",
            duration_ms=round(ms, 1),
        ))

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_json_report(report: DeploymentReport) -> str:
    """Format report as JSON string."""
    data = {
        "base_url": report.base_url,
        "profile": report.profile,
        "timestamp": report.timestamp,
        "summary": {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "all_passed": report.all_passed,
        },
        "checks": [asdict(c) for c in report.checks],
    }
    return json.dumps(data, indent=2)


def format_markdown_report(report: DeploymentReport) -> str:
    """Format report as Markdown string."""
    lines = []
    lines.append("# Deployment Validation Report")
    lines.append("")
    lines.append(f"- **Base URL**: {report.base_url}")
    lines.append(f"- **Profile**: {report.profile}")
    lines.append(f"- **Timestamp**: {report.timestamp}")
    lines.append(f"- **Result**: {'PASS' if report.all_passed else 'FAIL'}")
    lines.append(f"- **Checks**: {report.passed}/{report.total} passed")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Result | Detail | Duration |")
    lines.append("|-------|--------|--------|----------|")
    for c in report.checks:
        result = "PASS" if c.passed else "FAIL"
        detail = c.detail.replace("|", "\\|")
        lines.append(f"| {c.name} | {result} | {detail} | {c.duration_ms:.0f}ms |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_validation(
    base_url: str,
    api_key: str,
    profile: str = PROFILE_SINGLE_NODE,
    skip_job_submit: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> DeploymentReport:
    """Execute all validation checks and return a report."""
    report = DeploymentReport(
        base_url=base_url,
        profile=profile,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # 1. Health check
    report.checks.append(check_health(base_url, api_key))

    # 2. Detailed health check
    report.checks.append(check_detailed_health(base_url, api_key, profile=profile))

    # 3. Schema list
    report.checks.append(check_schema_list(base_url, api_key))

    # 4. Schema retrieval for each type
    report.checks.extend(check_schema_retrieval(base_url, api_key))

    # 5. Version header check
    report.checks.append(check_version_header(base_url, api_key))

    # 6. Job submission (optional)
    if not skip_job_submit:
        report.checks.extend(
            check_job_submission(base_url, api_key, timeout=timeout)
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate a running EDCOCR deployment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Profiles:\n"
            "  single-node   Basic health + schema + optional job test (default)\n"
            "  multi-gpu     All single-node checks + multi-GPU awareness\n"
            "  kubernetes    All single-node checks + Kubernetes subsystem awareness\n"
        ),
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="API base URL (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="API key for authentication",
    )
    parser.add_argument(
        "--profile",
        choices=ALL_PROFILES,
        default=PROFILE_SINGLE_NODE,
        help="Topology profile (default: single-node)",
    )
    parser.add_argument(
        "--skip-job-submit",
        action="store_true",
        help="Skip the job submission test",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Job completion timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        dest="output_format",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--output",
        help="Write report to file instead of stdout",
    )
    parser.add_argument(
        "--skip-tls-verify",
        action="store_true",
        help=(
            "Disable TLS certificate verification (NOT recommended for "
            "production; use only for self-signed certificates in test "
            "environments)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on all-pass, 1 on any failure."""
    global _skip_tls_verify
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.skip_tls_verify:
        _skip_tls_verify = True
        print(
            "WARNING: TLS certificate verification is disabled. "
            "Do not use this in production.",
            file=sys.stderr,
        )

    # Strip trailing slash from base URL
    base_url = args.base_url.rstrip("/")

    report = run_validation(
        base_url=base_url,
        api_key=args.api_key,
        profile=args.profile,
        skip_job_submit=args.skip_job_submit,
        timeout=args.timeout,
    )

    if args.output_format == "markdown":
        output = format_markdown_report(report)
    else:
        output = format_json_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
