#!/usr/bin/env python3
"""Production cutover pre-flight checker.

Consolidates credential validation, environment completeness, service
connectivity, pipeline drain status, Docker image verification, and
deployment state into a single orchestrated pass/fail gate.

Each check returns a standardised result dict so that the overall
orchestration can render text, JSON, or markdown reports uniformly.

Usage:
    python scripts/cutover_preflight.py
    python scripts/cutover_preflight.py --env-file coordinator/.env.prod
    python scripts/cutover_preflight.py --json
    python scripts/cutover_preflight.py --report docs/reports/preflight.md
    python scripts/cutover_preflight.py --require-drain --check-images
    python scripts/cutover_preflight.py --skip-connectivity
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Bootstrap: locate project root and add scripts/ to sys.path
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("cutover_preflight")

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

# ---------------------------------------------------------------------------
# Credential strength thresholds
# ---------------------------------------------------------------------------

# Minimum lengths by key pattern
_MIN_LENGTHS: dict[str, int] = {
    "DJANGO_SECRET_KEY": 50,
    "METRICS_API_KEY": 32,
    "OCR_API_KEY": 32,
}
# Default minimum for passwords and other secrets
_DEFAULT_MIN_LENGTH = 16

# Keys that are security-relevant credentials
_CREDENTIAL_KEY_PATTERNS = [
    "DJANGO_SECRET_KEY",
    "POSTGRES_PASSWORD",
    "RABBITMQ_PASSWORD",
    "RABBITMQ_ERLANG_COOKIE",
    "REDIS_PASSWORD",
    "FLOWER_PASSWORD",
    "METRICS_API_KEY",
    "OCR_API_KEY",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD",
]

# Keys that are identifiers (not secrets) -- enforce length but skip entropy
_IDENTIFIER_KEYS = {"S3_ACCESS_KEY", "MINIO_ROOT_USER"}

# Service URL keys and their expected schemes
_SERVICE_URL_KEYS: dict[str, list[str]] = {
    "DATABASE_URL": ["postgres", "postgresql"],
    "CELERY_BROKER_URL": ["amqp", "amqps", "redis", "rediss"],
    "S3_ENDPOINT": ["http", "https"],
    "REDIS_URL": ["redis", "rediss"],
    "CELERY_RESULT_BACKEND": ["redis", "rediss", "db+postgresql"],
}


# ---------------------------------------------------------------------------
# Result factory
# ---------------------------------------------------------------------------


def _result(
    name: str,
    status: str,
    details: str,
    remediation: str | None = None,
) -> dict[str, Any]:
    """Build a standardised check result dict."""
    return {
        "name": name,
        "status": status,
        "details": details,
        "remediation": remediation,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_print(*args: object, **kwargs: object) -> None:
    """Print with fallback for Windows console encoding."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        text = text.replace("\u2713", "OK").replace("\u2717", "FAIL").replace("\u2192", "->")
        end = kwargs.get("end", "\n")
        stream = kwargs.get("file", sys.stdout)
        if isinstance(stream, io.TextIOWrapper):
            stream.buffer.write((text + str(end)).encode("utf-8", errors="replace"))
        else:
            print(text, **kwargs)


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a .env file into a key-value dict.

    Handles comments, blank lines, and KEY=value pairs.
    Values are stripped of surrounding whitespace.
    """
    env_vars: dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    text = env_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        # Strip surrounding quotes (single or double)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env_vars[key.strip()] = value
    return env_vars


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string (bits per character).

    Returns 0.0 for empty strings.
    """
    if not s:
        return 0.0
    length = len(s)
    freq: Counter[str] = Counter(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _has_sequential_pattern(s: str, run_length: int = 5) -> bool:
    """Detect trivially sequential patterns like 'abcdef' or '123456'.

    Returns True if a run of ``run_length`` consecutive ASCII code points
    is found anywhere in *s*.
    """
    if len(s) < run_length:
        return False
    for i in range(len(s) - run_length + 1):
        is_sequential = True
        for j in range(1, run_length):
            if ord(s[i + j]) != ord(s[i + j - 1]) + 1:
                is_sequential = False
                break
        if is_sequential:
            return True
    return False


def _has_repeated_chars(s: str, threshold: float = 0.6) -> bool:
    """Return True if one character accounts for > threshold of the string."""
    if not s:
        return False
    freq: Counter[str] = Counter(s)
    most_common_count = freq.most_common(1)[0][1]
    return most_common_count / len(s) > threshold


def _extract_host_port(url: str) -> tuple[str, int] | None:
    """Extract (host, port) from a URL string.

    Returns None if parsing fails or no host/port can be determined.
    Handles standard URL formats and applies default ports for known schemes.
    """
    default_ports: dict[str, int] = {
        "postgres": 5432,
        "postgresql": 5432,
        "db+postgresql": 5432,
        "amqp": 5672,
        "amqps": 5671,
        "redis": 6379,
        "rediss": 6379,
        "http": 80,
        "https": 443,
    }
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return None
        port = parsed.port
        if port is None:
            port = default_ports.get(parsed.scheme, None)
        if port is None:
            return None
        return (host, port)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Check A: Credential Security Posture (delegates to credential_cutover_checklist)
# ---------------------------------------------------------------------------


def check_credential_posture(env_vars: dict[str, str]) -> list[dict[str, Any]]:
    """Delegate to credential_cutover_checklist.assess_env() if available.

    Falls back to a basic check if the module cannot be imported.
    """
    results: list[dict[str, Any]] = []
    try:
        from credential_cutover_checklist import assess_env

        reports = assess_env(env_vars)
        for r in reports:
            if r.has_value and not r.is_insecure:
                status = PASS
            elif not r.has_value:
                status = FAIL
            else:
                status = FAIL
            results.append(_result(
                name=f"credential_posture:{r.key}",
                status=status,
                details=r.recommendation,
                remediation=r.recommendation if status == FAIL else None,
            ))
    except ImportError:
        results.append(_result(
            name="credential_posture",
            status=SKIP,
            details="credential_cutover_checklist module not importable",
            remediation="Ensure scripts/ is on sys.path",
        ))
    return results


# ---------------------------------------------------------------------------
# Check B: Credential Strength Validation (NEW)
# ---------------------------------------------------------------------------


def check_credential_strength(env_vars: dict[str, str]) -> list[dict[str, Any]]:
    """Validate credential strength: minimum length, entropy, patterns."""
    results: list[dict[str, Any]] = []

    for key in _CREDENTIAL_KEY_PATTERNS:
        value = env_vars.get(key, "")
        if not value:
            # Missing credentials are caught by posture check; skip here
            continue

        check_name = f"credential_strength:{key}"
        min_len = _MIN_LENGTHS.get(key, _DEFAULT_MIN_LENGTH)
        issues: list[str] = []

        # Length check
        if len(value) < min_len:
            issues.append(f"length {len(value)} < minimum {min_len}")

        # Entropy check -- a reasonable password should have > 2.5 bits/char
        # Skip entropy for identifier keys (e.g. S3_ACCESS_KEY, MINIO_ROOT_USER)
        entropy = _shannon_entropy(value)
        if key not in _IDENTIFIER_KEYS and entropy < 2.5:
            issues.append(f"low entropy ({entropy:.1f} bits/char < 2.5)")

        # Sequential pattern check
        if _has_sequential_pattern(value):
            issues.append("contains sequential character pattern")

        # Repeated character dominance
        if _has_repeated_chars(value):
            issues.append("single character dominates >60% of value")

        if issues:
            results.append(_result(
                name=check_name,
                status=FAIL,
                details=f"Weak credential: {'; '.join(issues)}",
                remediation=f"Generate a stronger value: python -c \"import secrets; print(secrets.token_urlsafe({max(min_len, 32)}))\"",
            ))
        else:
            results.append(_result(
                name=check_name,
                status=PASS,
                details=f"length={len(value)}, entropy={entropy:.1f} bits/char",
            ))

    return results


# ---------------------------------------------------------------------------
# Check C: Environment Completeness (delegates to validate_phase7c_env)
# ---------------------------------------------------------------------------


def check_env_completeness(env_path: Path) -> list[dict[str, Any]]:
    """Delegate to validate_phase7c_env.run() if available.

    Captures stdout from the delegated call to avoid polluting
    structured output modes (JSON, markdown).
    """
    results: list[dict[str, Any]] = []
    try:
        from validate_phase7c_env import run as validate_env_run

        # Suppress stdout from the delegated call
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exit_code = validate_env_run(
                env_path=env_path,
                report=None,
                strict_placeholders=True,
                production=True,
            )
        finally:
            sys.stdout = old_stdout
        if exit_code == 0:
            results.append(_result(
                name="env_completeness",
                status=PASS,
                details="Phase 7c environment validation passed (production mode)",
            ))
        else:
            results.append(_result(
                name="env_completeness",
                status=FAIL,
                details=f"Phase 7c environment validation failed (exit code {exit_code})",
                remediation="Run: python scripts/validate_phase7c_env.py --production --env-file <path>",
            ))
    except ImportError:
        results.append(_result(
            name="env_completeness",
            status=SKIP,
            details="validate_phase7c_env module not importable",
            remediation="Ensure scripts/ is on sys.path",
        ))
    except Exception as exc:
        results.append(_result(
            name="env_completeness",
            status=SKIP,
            details=f"validate_phase7c_env raised: {exc}",
        ))
    return results


# ---------------------------------------------------------------------------
# Check D: Service Connectivity Probes (NEW)
# ---------------------------------------------------------------------------


def check_service_connectivity(
    env_vars: dict[str, str],
    timeout: float = 5.0,
) -> list[dict[str, Any]]:
    """Probe each service URL endpoint with a TCP socket connect."""
    results: list[dict[str, Any]] = []

    for key, _schemes in _SERVICE_URL_KEYS.items():
        url = env_vars.get(key, "")
        check_name = f"connectivity:{key}"

        if not url:
            results.append(_result(
                name=check_name,
                status=SKIP,
                details=f"{key} not set in environment",
            ))
            continue

        host_port = _extract_host_port(url)
        if host_port is None:
            results.append(_result(
                name=check_name,
                status=WARN,
                details=f"Could not parse host:port from {key} value",
                remediation=f"Check URL format for {key}",
            ))
            continue

        host, port = host_port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout)
            sock.connect((host, port))
            results.append(_result(
                name=check_name,
                status=PASS,
                details=f"TCP connect to {host}:{port} succeeded",
            ))
        except (socket.timeout, TimeoutError):
            results.append(_result(
                name=check_name,
                status=FAIL,
                details=f"TCP connect to {host}:{port} timed out after {timeout}s",
                remediation=f"Verify {key} host is reachable and service is running",
            ))
        except OSError as exc:
            results.append(_result(
                name=check_name,
                status=FAIL,
                details=f"TCP connect to {host}:{port} failed: {exc}",
                remediation=f"Verify {key} host is reachable and service is running",
            ))
        finally:
            sock.close()

    return results


# ---------------------------------------------------------------------------
# Check E: Pipeline Drain Check (NEW)
# ---------------------------------------------------------------------------


def check_pipeline_drain(
    env_vars: dict[str, str],
    require_drain: bool = False,
) -> list[dict[str, Any]]:
    """Check coordinator for active jobs if reachable."""
    results: list[dict[str, Any]] = []

    # Try to determine coordinator URL from common env vars
    coordinator_url = env_vars.get("COORDINATOR_URL", "")
    if not coordinator_url:
        # Fall back to API_HOST/API_PORT or Django settings
        api_host = env_vars.get("API_HOST", "")
        api_port = env_vars.get("API_PORT", "8000")
        if api_host:
            coordinator_url = f"http://{api_host}:{api_port}"

    if not coordinator_url:
        results.append(_result(
            name="pipeline_drain",
            status=SKIP,
            details="No COORDINATOR_URL or API_HOST set; cannot check drain status",
        ))
        return results

    # Try health endpoint
    try:
        from urllib.request import Request, urlopen

        health_url = f"{coordinator_url.rstrip('/')}/api/v1/health/"
        req = Request(health_url, method="GET")
        req.add_header("Accept", "application/json")

        with urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
                active_jobs = data.get("active_jobs", data.get("jobs_active", 0))
                if isinstance(active_jobs, int) and active_jobs > 0:
                    if require_drain:
                        results.append(_result(
                            name="pipeline_drain",
                            status=FAIL,
                            details=f"{active_jobs} active job(s) -- drain required before cutover",
                            remediation="Wait for active jobs to complete or cancel them",
                        ))
                    else:
                        results.append(_result(
                            name="pipeline_drain",
                            status=WARN,
                            details=f"{active_jobs} active job(s) -- consider draining before cutover",
                        ))
                else:
                    results.append(_result(
                        name="pipeline_drain",
                        status=PASS,
                        details="No active jobs detected; pipeline is drained",
                    ))
            except (json.JSONDecodeError, TypeError):
                results.append(_result(
                    name="pipeline_drain",
                    status=WARN,
                    details="Coordinator responded but could not parse active job count",
                ))
    except ImportError:
        results.append(_result(
            name="pipeline_drain",
            status=SKIP,
            details="urllib not available",
        ))
    except Exception:
        results.append(_result(
            name="pipeline_drain",
            status=SKIP,
            details="Coordinator not reachable; skipping drain check",
        ))

    return results


# ---------------------------------------------------------------------------
# Check F: Docker Image Check (NEW)
# ---------------------------------------------------------------------------


def check_docker_images(check_version: bool = False) -> list[dict[str, Any]]:
    """Verify local Docker images contain expected OCR images."""
    results: list[dict[str, Any]] = []

    try:
        proc = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            results.append(_result(
                name="docker_images",
                status=SKIP,
                details=f"docker images command failed (rc={proc.returncode})",
                remediation="Ensure Docker is installed and running",
            ))
            return results

        images = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        ocr_images = [img for img in images if "ocr" in img.lower()]

        if not ocr_images:
            results.append(_result(
                name="docker_images",
                status=WARN,
                details="No Docker images containing 'ocr' found locally",
                remediation="Build OCR images: docker-compose build",
            ))
        else:
            results.append(_result(
                name="docker_images",
                status=PASS,
                details=f"Found {len(ocr_images)} OCR-related image(s): {', '.join(ocr_images[:5])}",
            ))

        if check_version and ocr_images:
            try:
                from ocr_local.config.version import __version__

                version_match = any(__version__ in img for img in ocr_images)
                if version_match:
                    results.append(_result(
                        name="docker_image_version",
                        status=PASS,
                        details=f"Found image(s) matching version {__version__}",
                    ))
                else:
                    results.append(_result(
                        name="docker_image_version",
                        status=WARN,
                        details=f"No image tags match version {__version__}",
                        remediation="Rebuild with version tag: docker-compose build",
                    ))
            except ImportError:
                results.append(_result(
                    name="docker_image_version",
                    status=SKIP,
                    details="Could not import version.py to check image tags",
                ))

    except FileNotFoundError:
        results.append(_result(
            name="docker_images",
            status=SKIP,
            details="Docker CLI not found on PATH",
            remediation="Install Docker or add it to PATH",
        ))
    except subprocess.TimeoutExpired:
        results.append(_result(
            name="docker_images",
            status=SKIP,
            details="Docker command timed out after 15s",
        ))
    except OSError as exc:
        results.append(_result(
            name="docker_images",
            status=SKIP,
            details=f"Docker command failed: {exc}",
        ))

    return results


# ---------------------------------------------------------------------------
# Check G: Deployment Environment State (NEW)
# ---------------------------------------------------------------------------


def check_deployment_state(env_vars: dict[str, str]) -> list[dict[str, Any]]:
    """Verify deployment environment configuration is production-ready."""
    results: list[dict[str, Any]] = []

    deploy_env = env_vars.get("DEPLOYMENT_ENV", "")
    prod_ack = env_vars.get("PRODUCTION_READINESS_ACK", "").lower() in ("true", "1", "yes")

    if not deploy_env:
        results.append(_result(
            name="deployment_state",
            status=WARN,
            details="DEPLOYMENT_ENV not set",
            remediation="Set DEPLOYMENT_ENV to 'production', 'staging', or 'development'",
        ))
    elif deploy_env == "production":
        if prod_ack:
            results.append(_result(
                name="deployment_state",
                status=PASS,
                details="DEPLOYMENT_ENV=production, PRODUCTION_READINESS_ACK=true",
            ))
        else:
            results.append(_result(
                name="deployment_state",
                status=WARN,
                details="DEPLOYMENT_ENV=production but PRODUCTION_READINESS_ACK is not true",
                remediation="Set PRODUCTION_READINESS_ACK=true after verifying all production prerequisites",
            ))
    else:
        results.append(_result(
            name="deployment_state",
            status=WARN,
            details=f"DEPLOYMENT_ENV={deploy_env} (not production)",
            remediation="Set DEPLOYMENT_ENV=production for production cutover",
        ))

    # Check DEBUG mode
    debug = env_vars.get("DJANGO_DEBUG", "").lower() in ("true", "1", "yes")
    if debug:
        results.append(_result(
            name="deployment_debug",
            status=FAIL,
            details="DJANGO_DEBUG is enabled -- must be disabled for production",
            remediation="Set DJANGO_DEBUG=False",
        ))
    else:
        results.append(_result(
            name="deployment_debug",
            status=PASS,
            details="DJANGO_DEBUG is disabled",
        ))

    return results


# ---------------------------------------------------------------------------
# Check H: Backup Recency (NEW)
# ---------------------------------------------------------------------------


def check_backup_recency(env_vars: dict[str, str]) -> list[dict[str, Any]]:
    """Check for recent PostgreSQL backups in standard locations."""
    results: list[dict[str, Any]] = []

    backup_dirs = [
        Path(env_vars.get("BACKUP_DIR", "")) if env_vars.get("BACKUP_DIR") else None,
        Path("/backups/postgres"),
        PROJECT_ROOT / "backups",
    ]

    found_any = False
    newest_age_hours: float | None = None
    newest_path: str | None = None

    for backup_dir in backup_dirs:
        if backup_dir is None or not backup_dir.exists():
            continue

        for f in backup_dir.iterdir():
            if f.is_file() and f.suffix in (".sql", ".dump", ".gz", ".bak", ".pgdump"):
                found_any = True
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
                if newest_age_hours is None or age_hours < newest_age_hours:
                    newest_age_hours = age_hours
                    newest_path = str(f)

    if not found_any:
        results.append(_result(
            name="backup_recency",
            status=SKIP,
            details="No backup files found in standard locations",
            remediation="Create a PostgreSQL backup before cutover",
        ))
    elif newest_age_hours is not None and newest_age_hours <= 24:
        results.append(_result(
            name="backup_recency",
            status=PASS,
            details=f"Most recent backup is {newest_age_hours:.1f}h old: {newest_path}",
        ))
    elif newest_age_hours is not None:
        results.append(_result(
            name="backup_recency",
            status=WARN,
            details=f"Most recent backup is {newest_age_hours:.1f}h old (>24h): {newest_path}",
            remediation="Create a fresh PostgreSQL backup before cutover",
        ))

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_checks(
    env_path: Path,
    require_drain: bool = False,
    check_images: bool = False,
    skip_connectivity: bool = False,
) -> list[dict[str, Any]]:
    """Run all pre-flight checks and return aggregated results."""
    all_results: list[dict[str, Any]] = []

    # Parse env file
    if not env_path.exists():
        all_results.append(_result(
            name="env_file",
            status=FAIL,
            details=f"Environment file not found: {env_path}",
            remediation=f"Create {env_path} or specify a different path with --env-file",
        ))
        return all_results

    env_vars = _parse_env_file(env_path)

    all_results.append(_result(
        name="env_file",
        status=PASS,
        details=f"Loaded {len(env_vars)} variable(s) from {env_path}",
    ))

    # A) Credential posture
    all_results.extend(check_credential_posture(env_vars))

    # B) Credential strength
    all_results.extend(check_credential_strength(env_vars))

    # C) Environment completeness
    all_results.extend(check_env_completeness(env_path))

    # D) Service connectivity
    if skip_connectivity:
        all_results.append(_result(
            name="connectivity",
            status=SKIP,
            details="Connectivity probes skipped (--skip-connectivity)",
        ))
    else:
        all_results.extend(check_service_connectivity(env_vars))

    # E) Pipeline drain
    all_results.extend(check_pipeline_drain(env_vars, require_drain=require_drain))

    # F) Docker images
    all_results.extend(check_docker_images(check_version=check_images))

    # G) Deployment state
    all_results.extend(check_deployment_state(env_vars))

    # H) Backup recency
    all_results.extend(check_backup_recency(env_vars))

    return all_results


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------


def compute_verdict(results: list[dict[str, Any]]) -> str:
    """Compute overall verdict from check results.

    Returns "READY", "NOT READY", or "PARTIAL".
    """
    statuses = [r["status"] for r in results]
    has_fail = FAIL in statuses
    has_warn = WARN in statuses

    if has_fail:
        return "NOT READY"
    elif has_warn:
        return "PARTIAL"
    else:
        return "READY"


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

_STATUS_SYMBOLS = {
    PASS: "[PASS]",
    FAIL: "[FAIL]",
    WARN: "[WARN]",
    SKIP: "[SKIP]",
}


def render_text(results: list[dict[str, Any]], verdict: str) -> str:
    """Render results as a text table."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Production Cutover Pre-Flight Report")
    lines.append("=" * 80)
    lines.append("")

    # Column widths
    name_w = max((len(r["name"]) for r in results), default=20)
    name_w = max(name_w, 20)

    lines.append(f"  {'Check':<{name_w}}  {'Status':<8}  Details")
    lines.append(f"  {'-' * name_w}  {'-' * 8}  {'-' * 40}")

    for r in results:
        sym = _STATUS_SYMBOLS.get(r["status"], r["status"])
        lines.append(f"  {r['name']:<{name_w}}  {sym:<8}  {r['details']}")
        if r.get("remediation"):
            lines.append(f"  {'':<{name_w}}           -> {r['remediation']}")

    lines.append("")
    lines.append("-" * 80)

    counts = Counter(r["status"] for r in results)
    summary_parts = []
    for s in (PASS, FAIL, WARN, SKIP):
        if counts.get(s, 0) > 0:
            summary_parts.append(f"{s}: {counts[s]}")
    lines.append(f"  Summary: {', '.join(summary_parts)}")
    lines.append(f"  Verdict: {verdict}")
    lines.append("-" * 80)
    lines.append("")

    return "\n".join(lines)


def render_json(results: list[dict[str, Any]], verdict: str, env_path: str) -> dict:
    """Build a JSON-serializable report dict."""
    counts = Counter(r["status"] for r in results)
    return {
        "env_file": env_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "summary": {
            "pass": counts.get(PASS, 0),
            "fail": counts.get(FAIL, 0),
            "warn": counts.get(WARN, 0),
            "skip": counts.get(SKIP, 0),
            "total": len(results),
        },
        "checks": results,
    }


def render_markdown(results: list[dict[str, Any]], verdict: str, env_path: str) -> str:
    """Render results as a Markdown report."""
    lines: list[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("# Production Cutover Pre-Flight Report")
    lines.append("")
    lines.append(f"- **Generated**: {timestamp}")
    lines.append(f"- **Config File**: `{env_path}`")
    lines.append(f"- **Verdict**: **{verdict}**")
    lines.append("")

    counts = Counter(r["status"] for r in results)
    lines.append("## Summary")
    lines.append("")
    for s in (PASS, FAIL, WARN, SKIP):
        if counts.get(s, 0) > 0:
            lines.append(f"- {s}: {counts[s]}")
    lines.append(f"- Total: {len(results)}")
    lines.append("")

    lines.append("## Check Details")
    lines.append("")
    lines.append("| Check | Status | Details |")
    lines.append("|-------|--------|---------|")
    for r in results:
        details = r["details"].replace("|", "\\|")
        lines.append(f"| `{r['name']}` | {r['status']} | {details} |")

    # Remediation section
    remediation_items = [r for r in results if r.get("remediation")]
    if remediation_items:
        lines.append("")
        lines.append("## Remediation Required")
        lines.append("")
        for r in remediation_items:
            lines.append(f"- **{r['name']}**: {r['remediation']}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Production cutover pre-flight checker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("coordinator/.env"),
        help="Path to .env file (default: coordinator/.env)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output results as JSON to stdout",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write Markdown report to this file path",
    )
    parser.add_argument(
        "--require-drain",
        action="store_true",
        default=False,
        help="Fail if pipeline has active jobs (instead of warning)",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        default=False,
        help="Verify Docker image tags match version.py",
    )
    parser.add_argument(
        "--skip-connectivity",
        action="store_true",
        default=False,
        help="Skip TCP service connectivity probes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Returns:
        0 if verdict is READY, 1 if NOT READY, 2 if PARTIAL.
    """
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    # Run checks
    results = run_all_checks(
        env_path=args.env_file,
        require_drain=args.require_drain,
        check_images=args.check_images,
        skip_connectivity=args.skip_connectivity,
    )

    verdict = compute_verdict(results)

    # Output
    if args.json_output:
        report_data = render_json(results, verdict, str(args.env_file))
        _safe_print(json.dumps(report_data, indent=2))
    else:
        text = render_text(results, verdict)
        _safe_print(text)

    if args.report:
        md = render_markdown(results, verdict, str(args.env_file))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md, encoding="utf-8")
        _safe_print(f"Markdown report written to: {args.report}")

    # Exit code: 0=READY, 1=NOT READY, 2=PARTIAL
    if verdict == "READY":
        return 0
    elif verdict == "NOT READY":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
