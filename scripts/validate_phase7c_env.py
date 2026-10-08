#!/usr/bin/env python3
"""Phase 7c environment validation script.

This script validates coordinator .env configuration for Phase 7c rollout,
verifying required keys for baseline and S3 storage backend modes.

Usage:
    python scripts/validate_phase7c_env.py
    python scripts/validate_phase7c_env.py --env-file coordinator/.env.prod
    python scripts/validate_phase7c_env.py --report validation-report.md
    python scripts/validate_phase7c_env.py --strict-placeholders --bootstrap-from-example
    python scripts/validate_phase7c_env.py --production
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _safe_print(*args: object, **kwargs: object) -> None:
    """Print with fallback for Windows console encoding (cp1252 cannot render Unicode symbols)."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        text = text.replace("\u2713", "OK").replace("\u2717", "FAIL")
        end = kwargs.get("end", "\n")
        stream = kwargs.get("file", sys.stdout)
        if isinstance(stream, io.TextIOWrapper):
            stream.buffer.write((text + str(end)).encode("utf-8", errors="replace"))
        else:
            print(text, **kwargs)


@dataclass(frozen=True)
class ValidationResult:
    """Single validation check result."""
    key: str
    status: str  # "ok", "missing", "empty", "placeholder", "insecure_default"
    message: str


# Required keys for baseline coordinator operation
BASELINE_REQUIRED_KEYS = [
    "DJANGO_SECRET_KEY",
    "POSTGRES_PASSWORD",
    "RABBITMQ_PASSWORD",
    "RABBITMQ_ERLANG_COOKIE",
    "DATABASE_URL",
    "CELERY_BROKER_URL",
]

# Additional required keys when STORAGE_BACKEND=s3
S3_REQUIRED_KEYS = [
    "S3_ENDPOINT",
    "S3_BUCKET",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
]

# Placeholder patterns for strict validation
PLACEHOLDER_PATTERNS = [
    "change-me",
    "example",
    "your_",
    "placeholder",
]

# Exact placeholder values for common secrets
EXACT_PLACEHOLDER_VALUES = [
    "password",
    "secret",
]

# Additional required keys for production deployment (--production flag)
PRODUCTION_REQUIRED_KEYS = [
    "FLOWER_PASSWORD",
    "METRICS_API_KEY",
    "OCR_API_KEY",
]

# Known insecure default values that should never pass production validation.
# Checked case-insensitively against the full value.
KNOWN_INSECURE_DEFAULTS = {
    "minioadmin",
    "change-me",
    "change-me-in-production",
    "password",
    "secret",
    "admin",
    "test",
}

# Known insecure S3 credential defaults that should never pass production validation
KNOWN_INSECURE_S3_DEFAULTS = {
    "minioadmin",
}


def _looks_like_placeholder_token(value: str) -> bool:
    """Check whether a standalone token looks like placeholder content."""
    value_lower = value.lower()
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern in value_lower:
            return True
    return value_lower in EXACT_PLACEHOLDER_VALUES


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse .env file into key-value dict.

    Handles:
    - Comments (lines starting with #)
    - Blank lines
    - KEY=value pairs (strips whitespace)
    - Empty values (KEY= becomes empty string)
    """
    env_vars: dict[str, str] = {}

    if not env_path.exists():
        return env_vars

    text = env_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()

        # Skip comments and blank lines
        if not line or line.startswith("#"):
            continue

        # Parse KEY=value
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        env_vars[key.strip()] = value.strip()

    return env_vars


def _is_placeholder_value(value: str) -> bool:
    """Check if value looks like a placeholder.

    Returns True if value contains common placeholder patterns
    or matches exact placeholder values.
    """
    if _looks_like_placeholder_token(value):
        return True

    if "://" in value:
        from urllib.parse import urlparse

        parsed = urlparse(value)
        if parsed.username and _looks_like_placeholder_token(parsed.username):
            return True
        if parsed.password and _looks_like_placeholder_token(parsed.password):
            return True

    return False


def _validate_baseline(env_vars: dict[str, str], strict_placeholders: bool = True) -> list[ValidationResult]:
    """Validate baseline required environment variables."""
    results: list[ValidationResult] = []

    for key in BASELINE_REQUIRED_KEYS:
        if key not in env_vars:
            results.append(ValidationResult(
                key=key,
                status="missing",
                message=f"Required baseline key '{key}' not found in .env",
            ))
        elif not env_vars[key]:
            results.append(ValidationResult(
                key=key,
                status="empty",
                message=f"Required baseline key '{key}' is empty",
            ))
        elif strict_placeholders and _is_placeholder_value(env_vars[key]):
            results.append(ValidationResult(
                key=key,
                status="placeholder",
                message=f"Value appears to be a placeholder: '{env_vars[key]}'",
            ))
        else:
            results.append(ValidationResult(
                key=key,
                status="ok",
                message="Present and non-empty",
            ))

    return results


def _validate_s3_mode(env_vars: dict[str, str], strict_placeholders: bool = True) -> list[ValidationResult]:
    """Validate S3-specific required environment variables."""
    results: list[ValidationResult] = []

    mode = env_vars.get("S3_AUTH_MODE", "static").strip().lower()
    if mode not in {"static", "iam"}:
        return [ValidationResult(key="S3_AUTH_MODE", status="invalid_format", message="Must be static or iam")]
    required_keys = S3_REQUIRED_KEYS
    if mode == "iam":
        required_keys = ["S3_BUCKET"]
        if env_vars.get("S3_USE_PRESIGNED_URLS", "").lower() in {"true", "1", "yes"}:
            required_keys.append("S3_PRESIGNED_ALLOWED_HOSTS")
        for key in ("S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_SESSION_TOKEN"):
            if env_vars.get(key):
                results.append(ValidationResult(
                    key=key, status="invalid_format", message="IAM mode must not receive explicit S3 credentials",
                ))
        results.append(ValidationResult(
            key="S3_AUTH_MODE", status="ok",
            message="IAM provider configuration selected; live identity and bucket authorization remain target checks",
        ))

    for key in required_keys:
        if key not in env_vars:
            results.append(ValidationResult(
                key=key,
                status="missing",
                message=f"Required S3 key '{key}' not found in .env",
            ))
        elif not env_vars[key]:
            results.append(ValidationResult(
                key=key,
                status="empty",
                message=f"Required S3 key '{key}' is empty",
            ))
        elif key in ("S3_ACCESS_KEY", "S3_SECRET_KEY") and env_vars[key].strip().lower() in KNOWN_INSECURE_S3_DEFAULTS:
            results.append(ValidationResult(
                key=key,
                status="insecure_default",
                message=f"Value uses known insecure default credential '{env_vars[key]}'",
            ))
        elif strict_placeholders and _is_placeholder_value(env_vars[key]):
            results.append(ValidationResult(
                key=key,
                status="placeholder",
                message=f"Value appears to be a placeholder: '{env_vars[key]}'",
            ))
        else:
            results.append(ValidationResult(
                key=key,
                status="ok",
                message="Present and non-empty",
            ))

    return results


def _is_insecure_default(value: str) -> bool:
    """Check if value matches a known insecure default (case-insensitive)."""
    return value.strip().lower() in KNOWN_INSECURE_DEFAULTS


def _validate_production(env_vars: dict[str, str], strict_placeholders: bool = True) -> list[ValidationResult]:
    """Validate production-required environment variables.

    These keys are only checked when --production is passed.
    Each key is checked for presence, emptiness, insecure defaults, and placeholders.
    """
    results: list[ValidationResult] = []

    for key in PRODUCTION_REQUIRED_KEYS:
        if key not in env_vars:
            results.append(ValidationResult(
                key=key,
                status="missing",
                message=f"Required production key '{key}' not found in .env",
            ))
        elif not env_vars[key]:
            results.append(ValidationResult(
                key=key,
                status="empty",
                message=f"Required production key '{key}' is empty",
            ))
        elif _is_insecure_default(env_vars[key]):
            results.append(ValidationResult(
                key=key,
                status="insecure_default",
                message="Value matches a known insecure default",
            ))
        elif strict_placeholders and _is_placeholder_value(env_vars[key]):
            results.append(ValidationResult(
                key=key,
                status="placeholder",
                message="Value appears to be a placeholder",
            ))
        else:
            results.append(ValidationResult(
                key=key,
                status="ok",
                message="Present and non-empty",
            ))

    return results


def _validate_all_insecure_defaults(
    env_vars: dict[str, str],
    all_keys: list[str],
) -> list[ValidationResult]:
    """Check all security-relevant keys against KNOWN_INSECURE_DEFAULTS.

    Returns only insecure_default results for keys that have values matching
    the insecure defaults set but were not already caught by their primary
    validation pass. This is additive -- callers combine with other results.
    """
    results: list[ValidationResult] = []
    for key in all_keys:
        value = env_vars.get(key, "")
        if value and _is_insecure_default(value):
            results.append(ValidationResult(
                key=key,
                status="insecure_default",
                message="Value matches a known insecure default",
            ))
    return results


# ---------------------------------------------------------------------------
# Extended validation helpers (C-15)
# ---------------------------------------------------------------------------

def _check_bool_format(value: str) -> bool:
    """Return True if *value* looks like a valid boolean string."""
    return value.lower() in ("true", "false", "1", "0", "yes", "no")


def _check_numeric(value: str, *, positive: bool = False) -> bool:
    """Return True if *value* is a valid number (optionally >0)."""
    try:
        num = float(value)
        if positive:
            return num > 0
        return True
    except (ValueError, TypeError):
        return False


def _check_int(value: str, *, positive: bool = False) -> bool:
    """Return True if *value* is a valid integer (optionally >0)."""
    try:
        num = int(value)
        if positive:
            return num > 0
        return True
    except (ValueError, TypeError):
        return False


def _check_port(value: str) -> bool:
    """Return True if *value* is a valid port number (1-65535)."""
    try:
        port = int(value)
        return 1 <= port <= 65535
    except (ValueError, TypeError):
        return False


def _check_url_format(value: str) -> bool:
    """Return True if *value* looks like a URL (has scheme + host)."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(value)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def _validate_coordinator_extended(env_vars: dict[str, str]) -> list[ValidationResult]:
    """Validate extended coordinator configuration variables.

    Covers Redis, S3 extended options, context window, and job settings.
    Missing optional vars produce ``warn`` status, not ``missing``.
    """
    results: list[ValidationResult] = []

    # ---- Redis ----
    val = env_vars.get("REDIS_URL")
    if val is not None:
        if _check_url_format(val):
            results.append(ValidationResult(key="REDIS_URL", status="ok", message="Valid URL format"))
        else:
            results.append(ValidationResult(key="REDIS_URL", status="invalid_format", message=f"Does not look like a valid URL: '{val}'"))
    else:
        results.append(ValidationResult(key="REDIS_URL", status="warn", message="Not set (optional)"))

    # If REDIS_SENTINEL_* vars present, check they are non-empty
    for sentinel_key in [k for k in env_vars if k.startswith("REDIS_SENTINEL_")]:
        val = env_vars[sentinel_key]
        if val:
            results.append(ValidationResult(key=sentinel_key, status="ok", message="Present and non-empty"))
        else:
            results.append(ValidationResult(key=sentinel_key, status="warn", message="Set but empty"))

    # ---- S3 extended ----
    for s3_key in ("S3_REGION",):
        val = env_vars.get(s3_key)
        if val is not None:
            if val:
                results.append(ValidationResult(key=s3_key, status="ok", message="Present and non-empty"))
            else:
                results.append(ValidationResult(key=s3_key, status="warn", message="Set but empty"))
        else:
            results.append(ValidationResult(key=s3_key, status="warn", message="Not set (optional)"))

    for s3_bool_key in ("S3_USE_PRESIGNED_URLS",):
        val = env_vars.get(s3_bool_key)
        if val is not None:
            if _check_bool_format(val):
                results.append(ValidationResult(key=s3_bool_key, status="ok", message=f"Valid boolean: {val}"))
            else:
                results.append(ValidationResult(key=s3_bool_key, status="invalid_format", message=f"Expected boolean, got: '{val}'"))
        else:
            results.append(ValidationResult(key=s3_bool_key, status="warn", message="Not set (optional)"))

    for s3_num_key in ("S3_PRESIGNED_URL_EXPIRY", "S3_CACHE_MAX_SIZE_GB"):
        val = env_vars.get(s3_num_key)
        if val is not None:
            if _check_numeric(val):
                results.append(ValidationResult(key=s3_num_key, status="ok", message=f"Valid numeric: {val}"))
            else:
                results.append(ValidationResult(key=s3_num_key, status="invalid_format", message=f"Expected numeric value, got: '{val}'"))
        else:
            results.append(ValidationResult(key=s3_num_key, status="warn", message="Not set (optional)"))

    # ---- Context window ----
    val = env_vars.get("CONTEXT_WINDOW_ENABLED")
    if val is not None:
        if _check_bool_format(val):
            results.append(ValidationResult(key="CONTEXT_WINDOW_ENABLED", status="ok", message=f"Valid boolean: {val}"))
        else:
            results.append(ValidationResult(key="CONTEXT_WINDOW_ENABLED", status="invalid_format", message=f"Expected boolean, got: '{val}'"))
    else:
        results.append(ValidationResult(key="CONTEXT_WINDOW_ENABLED", status="warn", message="Not set (optional)"))

    for ctx_int_key in ("CONTEXT_WINDOW_SIZE", "CONTEXT_STORE_TTL"):
        val = env_vars.get(ctx_int_key)
        if val is not None:
            if _check_int(val):
                results.append(ValidationResult(key=ctx_int_key, status="ok", message=f"Valid integer: {val}"))
            else:
                results.append(ValidationResult(key=ctx_int_key, status="invalid_format", message=f"Expected integer, got: '{val}'"))
        else:
            results.append(ValidationResult(key=ctx_int_key, status="warn", message="Not set (optional)"))

    # ---- Job settings ----
    for job_key in ("JOB_PROCESSING_TIMEOUT_MINUTES", "JOB_RETENTION_DAYS"):
        val = env_vars.get(job_key)
        if val is not None:
            if _check_numeric(val, positive=True):
                results.append(ValidationResult(key=job_key, status="ok", message=f"Valid positive numeric: {val}"))
            else:
                results.append(ValidationResult(key=job_key, status="invalid_format", message=f"Expected positive numeric, got: '{val}'"))
        else:
            results.append(ValidationResult(key=job_key, status="warn", message="Not set (optional)"))

    return results


def _validate_api_config(env_vars: dict[str, str]) -> list[ValidationResult]:
    """Validate API server configuration variables.

    Covers server settings, limits, security, and feature flags.
    Missing optional vars produce ``warn`` status.
    """
    results: list[ValidationResult] = []

    # ---- Server ----
    val = env_vars.get("API_HOST")
    if val is not None:
        if val:
            results.append(ValidationResult(key="API_HOST", status="ok", message=f"Set to: {val}"))
        else:
            results.append(ValidationResult(key="API_HOST", status="warn", message="Set but empty"))
    else:
        results.append(ValidationResult(key="API_HOST", status="warn", message="Not set (optional)"))

    val = env_vars.get("API_PORT")
    if val is not None:
        if _check_port(val):
            results.append(ValidationResult(key="API_PORT", status="ok", message=f"Valid port: {val}"))
        else:
            results.append(ValidationResult(key="API_PORT", status="invalid_format", message=f"Expected port 1-65535, got: '{val}'"))
    else:
        results.append(ValidationResult(key="API_PORT", status="warn", message="Not set (optional)"))

    # ---- Limits ----
    for limit_key in ("MAX_UPLOAD_SIZE_MB", "MAX_CONCURRENT_JOBS", "MAX_BATCH_SIZE"):
        val = env_vars.get(limit_key)
        if val is not None:
            if _check_numeric(val, positive=True):
                results.append(ValidationResult(key=limit_key, status="ok", message=f"Valid positive numeric: {val}"))
            else:
                results.append(ValidationResult(key=limit_key, status="invalid_format", message=f"Expected positive numeric, got: '{val}'"))
        else:
            results.append(ValidationResult(key=limit_key, status="warn", message="Not set (optional)"))

    # ---- Security ----
    # WEBHOOK_SECRET: should be non-empty when webhooks are configured
    webhooks_enabled = env_vars.get("ENABLE_WEBHOOKS", "").lower() in ("true", "1", "yes")
    val = env_vars.get("WEBHOOK_SECRET")
    if val is not None:
        if val:
            results.append(ValidationResult(key="WEBHOOK_SECRET", status="ok", message="Present and non-empty"))
        else:
            status = "invalid_format" if webhooks_enabled else "warn"
            msg = "Empty (required when webhooks enabled)" if webhooks_enabled else "Set but empty"
            results.append(ValidationResult(key="WEBHOOK_SECRET", status=status, message=msg))
    else:
        if webhooks_enabled:
            results.append(ValidationResult(key="WEBHOOK_SECRET", status="invalid_format", message="Missing but webhooks are enabled"))
        else:
            results.append(ValidationResult(key="WEBHOOK_SECRET", status="warn", message="Not set (optional)"))

    for wh_num in ("WEBHOOK_TIMEOUT", "WEBHOOK_MAX_RETRIES"):
        val = env_vars.get(wh_num)
        if val is not None:
            if _check_numeric(val):
                results.append(ValidationResult(key=wh_num, status="ok", message=f"Valid numeric: {val}"))
            else:
                results.append(ValidationResult(key=wh_num, status="invalid_format", message=f"Expected numeric, got: '{val}'"))
        else:
            results.append(ValidationResult(key=wh_num, status="warn", message="Not set (optional)"))

    # ---- Feature flags ----
    for flag_key in ("ENABLE_TRANSFORMS", "ENABLE_STAMPING", "ENABLE_MULTITENANCY"):
        val = env_vars.get(flag_key)
        if val is not None:
            if _check_bool_format(val):
                results.append(ValidationResult(key=flag_key, status="ok", message=f"Valid boolean: {val}"))
            else:
                results.append(ValidationResult(key=flag_key, status="invalid_format", message=f"Expected boolean, got: '{val}'"))
        else:
            results.append(ValidationResult(key=flag_key, status="warn", message="Not set (optional)"))

    return results


def _validate_pipeline_config(env_vars: dict[str, str]) -> list[ValidationResult]:
    """Validate pipeline configuration variables.

    Covers thread counts, queue sizes, and DPI setting.
    Missing optional vars produce ``warn`` status.
    """
    results: list[ValidationResult] = []

    # ---- Thread counts ----
    for key in ("NUM_EXTRACTORS", "NUM_WORKERS", "NUM_COMPRESSORS"):
        val = env_vars.get(key)
        if val is not None:
            if _check_numeric(val, positive=True):
                results.append(ValidationResult(key=key, status="ok", message=f"Valid positive numeric: {val}"))
            else:
                results.append(ValidationResult(key=key, status="invalid_format", message=f"Expected positive numeric, got: '{val}'"))
        else:
            results.append(ValidationResult(key=key, status="warn", message="Not set (optional)"))

    # ---- Queue sizes ----
    for key in ("CHUNK_QUEUE_SIZE", "IMAGE_QUEUE_SIZE"):
        val = env_vars.get(key)
        if val is not None:
            if _check_numeric(val, positive=True):
                results.append(ValidationResult(key=key, status="ok", message=f"Valid positive numeric: {val}"))
            else:
                results.append(ValidationResult(key=key, status="invalid_format", message=f"Expected positive numeric, got: '{val}'"))
        else:
            results.append(ValidationResult(key=key, status="warn", message="Not set (optional)"))

    # ---- DPI ----
    val = env_vars.get("DPI")
    if val is not None:
        try:
            dpi_val = int(val)
            if 150 <= dpi_val <= 1200:
                results.append(ValidationResult(key="DPI", status="ok", message=f"Valid DPI: {dpi_val}"))
            else:
                results.append(ValidationResult(key="DPI", status="invalid_format", message=f"DPI {dpi_val} outside valid range 150-1200"))
        except (ValueError, TypeError):
            results.append(ValidationResult(key="DPI", status="invalid_format", message=f"Expected integer, got: '{val}'"))
    else:
        results.append(ValidationResult(key="DPI", status="warn", message="Not set (optional)"))

    return results


def _check_storage_backend(env_vars: dict[str, str]) -> str:
    """Determine storage backend mode from env vars.

    Returns: "nfs" (default), "s3", or "invalid"
    """
    backend = env_vars.get("STORAGE_BACKEND", "nfs").strip().lower()

    if backend not in ("nfs", "s3"):
        return "invalid"

    return backend


def _print_summary(
    env_path: Path,
    backend: str,
    baseline_results: list[ValidationResult],
    s3_results: list[ValidationResult] | None,
    production_results: list[ValidationResult] | None = None,
    extended_sections: dict[str, list[ValidationResult]] | None = None,
) -> None:
    """Print concise validation summary to stdout."""
    def _count_status(results: list[ValidationResult], status: str) -> int:
        return sum(1 for result in results if result.status == status)

    _safe_print("=" * 70)
    _safe_print("Phase 7c Environment Validation")
    _safe_print("=" * 70)
    _safe_print(f"Config file: {env_path}")
    _safe_print(f"Storage backend: {backend}")
    _safe_print()

    # Baseline checks
    _safe_print("Baseline Configuration:")
    baseline_passed = all(r.status == "ok" for r in baseline_results)
    for result in baseline_results:
        status_symbol = "\u2713" if result.status == "ok" else "\u2717"
        _safe_print(f"  {status_symbol} {result.key}: {result.message}")
    _safe_print(
        "  Summary: "
        f"ok={_count_status(baseline_results, 'ok')}, "
        f"missing={_count_status(baseline_results, 'missing')}, "
        f"empty={_count_status(baseline_results, 'empty')}, "
        f"placeholder={_count_status(baseline_results, 'placeholder')}, "
        f"insecure_default={_count_status(baseline_results, 'insecure_default')}"
    )
    _safe_print()

    # S3 checks (if applicable)
    s3_passed = True
    if s3_results:
        _safe_print("S3 Storage Configuration:")
        s3_passed = all(r.status == "ok" for r in s3_results)
        for result in s3_results:
            status_symbol = "\u2713" if result.status == "ok" else "\u2717"
            _safe_print(f"  {status_symbol} {result.key}: {result.message}")
        _safe_print(
            "  Summary: "
            f"ok={_count_status(s3_results, 'ok')}, "
            f"missing={_count_status(s3_results, 'missing')}, "
            f"empty={_count_status(s3_results, 'empty')}, "
            f"placeholder={_count_status(s3_results, 'placeholder')}, "
            f"insecure_default={_count_status(s3_results, 'insecure_default')}"
        )
        _safe_print()

    # Production checks (if applicable)
    production_passed = True
    if production_results:
        _safe_print("Production Configuration:")
        production_passed = all(r.status == "ok" for r in production_results)
        for result in production_results:
            status_symbol = "\u2713" if result.status == "ok" else "\u2717"
            _safe_print(f"  {status_symbol} {result.key}: {result.message}")
        _safe_print(
            "  Summary: "
            f"ok={_count_status(production_results, 'ok')}, "
            f"missing={_count_status(production_results, 'missing')}, "
            f"empty={_count_status(production_results, 'empty')}, "
            f"placeholder={_count_status(production_results, 'placeholder')}, "
            f"insecure_default={_count_status(production_results, 'insecure_default')}"
        )
        _safe_print()

    # Extended validation sections (C-15)
    extended_passed = True
    if extended_sections:
        for section_name, section_results in extended_sections.items():
            _safe_print(f"{section_name}:")
            section_ok = all(r.status in ("ok", "warn") for r in section_results)
            if not section_ok:
                extended_passed = False
            for result in section_results:
                if result.status == "ok":
                    status_symbol = "\u2713"
                elif result.status == "warn":
                    status_symbol = "~"
                else:
                    status_symbol = "\u2717"
                _safe_print(f"  {status_symbol} {result.key}: {result.message}")
            _safe_print(
                "  Summary: "
                f"ok={_count_status(section_results, 'ok')}, "
                f"warn={_count_status(section_results, 'warn')}, "
                f"invalid_format={_count_status(section_results, 'invalid_format')}"
            )
            _safe_print()

    # Overall status
    overall_passed = baseline_passed and s3_passed and production_passed and extended_passed
    _safe_print("=" * 70)
    if overall_passed:
        _safe_print("\u2713 PASS: All required environment variables validated")
    else:
        _safe_print("\u2717 FAIL: Validation errors detected")
    _safe_print("=" * 70)


def _write_missing_file_report(report_path: Path, env_path: Path) -> None:
    """Write markdown report for missing env file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Phase 7c Environment Validation Report",
        "",
        f"- **Generated**: {timestamp}",
        f"- **Config File**: `{env_path}`",
        "- **Overall Status**: ✗ FAIL",
        "",
        "## Error",
        "",
        f"Environment file not found at path: `{env_path}`",
        "",
        "**Action Required**: Create the environment file with required configuration keys.",
        "",
        "### Required Baseline Keys",
        "",
        "- `DJANGO_SECRET_KEY`",
        "- `POSTGRES_PASSWORD`",
        "- `RABBITMQ_PASSWORD`",
        "",
        "### Required S3 Keys (if STORAGE_BACKEND=s3)",
        "",
        "- `S3_ENDPOINT`",
        "- `S3_BUCKET`",
        "- `S3_ACCESS_KEY`",
        "- `S3_SECRET_KEY`",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown_report(
    report_path: Path,
    env_path: Path,
    backend: str,
    baseline_results: list[ValidationResult],
    s3_results: list[ValidationResult] | None,
) -> None:
    """Write markdown validation report."""
    def _count_status(results: list[ValidationResult], status: str) -> int:
        return sum(1 for result in results if result.status == status)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    baseline_passed = all(r.status == "ok" for r in baseline_results)
    s3_passed = s3_results is None or all(r.status == "ok" for r in s3_results)
    overall_passed = baseline_passed and s3_passed

    lines = [
        "# Phase 7c Environment Validation Report",
        "",
        f"- **Generated**: {timestamp}",
        f"- **Config File**: `{env_path}`",
        f"- **Storage Backend**: `{backend}`",
        f"- **Overall Status**: {'✓ PASS' if overall_passed else '✗ FAIL'}",
        "",
        "## Baseline Configuration",
        "",
        f"Status: {'✓ PASS' if baseline_passed else '✗ FAIL'}",
        "",
        "| Key | Status | Message |",
        "|-----|--------|---------|",
    ]

    for result in baseline_results:
        status_symbol = "✓" if result.status == "ok" else "✗"
        lines.append(f"| `{result.key}` | {status_symbol} {result.status} | {result.message} |")

    if s3_results:
        lines.extend([
            "",
            "## S3 Storage Configuration",
            "",
            f"Status: {'✓ PASS' if s3_passed else '✗ FAIL'}",
            "",
            "| Key | Status | Message |",
            "|-----|--------|---------|",
        ])

        for result in s3_results:
            status_symbol = "✓" if result.status == "ok" else "✗"
            lines.append(f"| `{result.key}` | {status_symbol} {result.status} | {result.message} |")

    lines.extend([
        "",
        "## Summary",
        "",
        f"- Baseline checks: {sum(1 for r in baseline_results if r.status == 'ok')}/{len(baseline_results)} passed",
        f"- Baseline insecure defaults: {_count_status(baseline_results, 'insecure_default')}",
    ])

    if s3_results:
        lines.append(f"- S3 checks: {sum(1 for r in s3_results if r.status == 'ok')}/{len(s3_results)} passed")
        lines.append(f"- S3 insecure defaults: {_count_status(s3_results, 'insecure_default')}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bootstrap_from_example(env_path: Path) -> bool:
    """Bootstrap .env file from .env.example if it doesn't exist.

    Args:
        env_path: Path to target .env file

    Returns:
        True if bootstrap was successful or not needed, False if failed
    """
    # If target already exists, no need to bootstrap
    if env_path.exists():
        return True

    # Find sibling .env.example
    example_path = env_path.parent / ".env.example"
    if not example_path.exists():
        print(f"ERROR: Cannot bootstrap - example file not found: {example_path}", file=sys.stderr)
        return False

    # Copy example to target
    try:
        shutil.copy2(example_path, env_path)
        _safe_print(f"\u2713 Bootstrapped {env_path} from {example_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to bootstrap from example: {e}", file=sys.stderr)
        return False


def run(
    env_path: Path,
    report: Path | None,
    strict_placeholders: bool = True,
    bootstrap_from_example: bool = False,
    production: bool = False,
    validate_coordinator: bool = False,
    validate_api: bool = False,
    validate_pipeline: bool = False,
) -> int:
    """Run environment validation.

    Args:
        env_path: Path to .env file to validate
        report: Optional path to write markdown report
        strict_placeholders: If True, treat placeholder values as invalid
        bootstrap_from_example: If True, copy from .env.example when env file is missing
        production: If True, also validate production-required keys and check
            all values against the expanded KNOWN_INSECURE_DEFAULTS set
        validate_coordinator: If True, run extended coordinator validation
        validate_api: If True, run API config validation
        validate_pipeline: If True, run pipeline config validation

    Returns:
        0 if validation passed, non-zero if validation failed or errors occurred
    """
    # Bootstrap from example if requested and file is missing
    if bootstrap_from_example and not env_path.exists():
        if not _bootstrap_from_example(env_path):
            return 2

    # Check if env file exists
    if not env_path.exists():
        print(f"ERROR: Environment file not found: {env_path}", file=sys.stderr)

        # If report requested, still write a report documenting missing file
        if report:
            try:
                _write_missing_file_report(report, env_path)
                print(f"\nReport written to: {report}")
            except Exception as e:
                print(f"ERROR: Failed to write report: {e}", file=sys.stderr)

        return 2

    # Parse environment file
    try:
        env_vars = _parse_env_file(env_path)
    except Exception as e:
        print(f"ERROR: Failed to parse environment file: {e}", file=sys.stderr)
        return 2

    # Determine storage backend
    backend = _check_storage_backend(env_vars)
    if backend == "invalid":
        actual_value = env_vars.get("STORAGE_BACKEND", "")
        print(f"ERROR: Invalid STORAGE_BACKEND='{actual_value}' (must be 'nfs' or 's3')", file=sys.stderr)
        return 1

    # Run baseline validation
    baseline_results = _validate_baseline(env_vars, strict_placeholders)

    # Run S3 validation if in S3 mode
    s3_results = None
    if backend == "s3":
        s3_results = _validate_s3_mode(env_vars, strict_placeholders)

    # Run production validation if --production flag
    production_results = None
    if production:
        production_results = _validate_production(env_vars, strict_placeholders)

    # Run extended validation tiers (C-15)
    extended_sections: dict[str, list[ValidationResult]] = {}
    if validate_coordinator:
        extended_sections["Coordinator Extended Configuration"] = _validate_coordinator_extended(env_vars)
    if validate_api:
        extended_sections["API Configuration"] = _validate_api_config(env_vars)
    if validate_pipeline:
        extended_sections["Pipeline Configuration"] = _validate_pipeline_config(env_vars)

    # Print summary
    _print_summary(
        env_path, backend, baseline_results, s3_results,
        production_results, extended_sections or None,
    )

    # Write optional report
    if report:
        try:
            _write_markdown_report(report, env_path, backend, baseline_results, s3_results)
            print(f"\nReport written to: {report}")
        except Exception as e:
            print(f"ERROR: Failed to write report: {e}", file=sys.stderr)
            return 2

    # Determine exit code
    baseline_passed = all(r.status == "ok" for r in baseline_results)
    s3_passed = s3_results is None or all(r.status == "ok" for r in s3_results)
    production_passed = production_results is None or all(r.status == "ok" for r in production_results)

    # Extended sections: warn is acceptable, only invalid_format / missing fail
    extended_passed = True
    for section_results in extended_sections.values():
        if any(r.status not in ("ok", "warn") for r in section_results):
            extended_passed = False
            break

    return 0 if (baseline_passed and s3_passed and production_passed and extended_passed) else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Phase 7c environment configuration."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("coordinator/.env"),
        help="Path to .env file (default: coordinator/.env)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional markdown report output path",
    )
    parser.add_argument(
        "--strict-placeholders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Treat placeholder values as validation failures "
            "(default: true; use --no-strict-placeholders to allow placeholders)"
        ),
    )
    parser.add_argument(
        "--bootstrap-from-example",
        action="store_true",
        default=False,
        help="Copy .env.example to --env-file if env file is missing (default: false)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        default=False,
        help=(
            "Enable production validation: check RABBITMQ_ERLANG_COOKIE, "
            "FLOWER_PASSWORD, METRICS_API_KEY, OCR_API_KEY and validate all "
            "values against known insecure defaults (default: false)"
        ),
    )
    # Extended validation tiers (C-15)
    parser.add_argument(
        "--validate-all",
        action="store_true",
        default=False,
        help="Enable all extended validation tiers (coordinator, API, pipeline)",
    )
    parser.add_argument(
        "--validate-coordinator",
        action="store_true",
        default=False,
        help="Enable extended coordinator validation (Redis, S3 extended, context window, job settings)",
    )
    parser.add_argument(
        "--validate-api",
        action="store_true",
        default=False,
        help="Enable API configuration validation (server, limits, security, feature flags)",
    )
    parser.add_argument(
        "--validate-pipeline",
        action="store_true",
        default=False,
        help="Enable pipeline configuration validation (threads, queues, DPI)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # --validate-all enables all extended tiers
    validate_coordinator = args.validate_coordinator or args.validate_all
    validate_api = args.validate_api or args.validate_all
    validate_pipeline = args.validate_pipeline or args.validate_all

    return run(
        env_path=args.env_file,
        report=args.report,
        strict_placeholders=args.strict_placeholders,
        bootstrap_from_example=args.bootstrap_from_example,
        production=args.production,
        validate_coordinator=validate_coordinator,
        validate_api=validate_api,
        validate_pipeline=validate_pipeline,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
