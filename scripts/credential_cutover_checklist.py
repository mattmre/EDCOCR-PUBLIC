#!/usr/bin/env python3
"""Credential cutover checklist for production deployment.

Reads a coordinator .env file and reports the security posture of every
security-relevant key: whether it is present, whether its value matches a
known insecure default, and a recommendation for remediation.

NEVER prints actual credential values -- only presence/insecurity status.

Usage:
    python scripts/credential_cutover_checklist.py
    python scripts/credential_cutover_checklist.py --env-file coordinator/.env.prod
    python scripts/credential_cutover_checklist.py --json-report report.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Security-relevant keys and insecure defaults
# ---------------------------------------------------------------------------

# All keys that carry secrets or auth tokens
SECURITY_RELEVANT_KEYS = [
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

# Substring patterns that indicate a placeholder value
_PLACEHOLDER_PATTERNS = [
    "change-me",
    "example",
    "your_",
    "placeholder",
]

# Exact (case-insensitive) values that are considered insecure
_INSECURE_EXACT = {
    "minioadmin",
    "change-me",
    "change-me-in-production",
    "password",
    "secret",
    "admin",
    "test",
}

# Per-key human-readable recommendations
_RECOMMENDATIONS: dict[str, str] = {
    "DJANGO_SECRET_KEY": "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(48))\"",
    "POSTGRES_PASSWORD": "Use a strong random password (32+ chars)",
    "RABBITMQ_PASSWORD": "Use a strong random password (32+ chars)",
    "RABBITMQ_ERLANG_COOKIE": "Must be identical across all RabbitMQ nodes; generate a random string",
    "REDIS_PASSWORD": "Use a strong random password (32+ chars)",
    "FLOWER_PASSWORD": "Use a strong random password; rotate periodically",
    "METRICS_API_KEY": "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\"",
    "OCR_API_KEY": "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\"",
    "S3_ACCESS_KEY": "Use IAM role or generate dedicated MinIO access key",
    "S3_SECRET_KEY": "Use IAM role or generate dedicated MinIO secret key",
    "MINIO_ROOT_USER": "Use a non-default admin username",
    "MINIO_ROOT_PASSWORD": "Use a strong random password (32+ chars)",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyReport:
    """Assessment of a single security-relevant key."""
    key: str
    has_value: bool
    is_insecure: bool
    recommendation: str


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
    """Parse .env file into key-value dict."""
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
        env_vars[key.strip()] = value.strip()
    return env_vars


def _is_insecure_value(value: str) -> bool:
    """Check whether a value is a known insecure default or placeholder."""
    lower = value.lower()
    # Check exact insecure values
    if lower in _INSECURE_EXACT:
        return True
    # Check substring placeholder patterns
    for pattern in _PLACEHOLDER_PATTERNS:
        if pattern in lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def assess_env(env_vars: dict[str, str]) -> list[KeyReport]:
    """Assess all security-relevant keys in the environment.

    Returns a list of KeyReport objects, one per SECURITY_RELEVANT_KEYS entry.
    """
    reports: list[KeyReport] = []
    for key in SECURITY_RELEVANT_KEYS:
        value = env_vars.get(key, "")
        has_value = bool(value)
        is_insecure = has_value and _is_insecure_value(value)
        recommendation = _RECOMMENDATIONS.get(key, "Review and set a secure value")
        if has_value and not is_insecure:
            recommendation = "OK -- value appears secure"
        elif not has_value:
            recommendation = f"MISSING -- {_RECOMMENDATIONS.get(key, 'set a secure value')}"
        else:
            recommendation = f"INSECURE -- {_RECOMMENDATIONS.get(key, 'replace with a secure value')}"
        reports.append(KeyReport(
            key=key,
            has_value=has_value,
            is_insecure=is_insecure,
            recommendation=recommendation,
        ))
    return reports


def print_table(reports: list[KeyReport]) -> None:
    """Print a human-readable table of results to stdout."""
    # Column widths
    key_w = max(len(r.key) for r in reports)
    has_w = 9  # "has_value"
    ins_w = 11  # "is_insecure"

    header = (
        f"{'Key':<{key_w}}  {'Has Value':<{has_w}}  {'Is Insecure':<{ins_w}}  Recommendation"
    )
    separator = "-" * len(header)

    _safe_print()
    _safe_print("Credential Cutover Checklist")
    _safe_print(separator)
    _safe_print(header)
    _safe_print(separator)

    for r in reports:
        has_str = "yes" if r.has_value else "NO"
        ins_str = "YES" if r.is_insecure else "no"
        _safe_print(f"{r.key:<{key_w}}  {has_str:<{has_w}}  {ins_str:<{ins_w}}  {r.recommendation}")

    _safe_print(separator)

    insecure_count = sum(1 for r in reports if r.is_insecure)
    missing_count = sum(1 for r in reports if not r.has_value)
    secure_count = sum(1 for r in reports if r.has_value and not r.is_insecure)

    _safe_print(f"  Secure: {secure_count}  |  Insecure: {insecure_count}  |  Missing: {missing_count}")
    _safe_print(separator)
    _safe_print()


def build_json_report(reports: list[KeyReport], env_path: str) -> dict:
    """Build a JSON-serializable report dict."""
    insecure_count = sum(1 for r in reports if r.is_insecure)
    missing_count = sum(1 for r in reports if not r.has_value)
    secure_count = sum(1 for r in reports if r.has_value and not r.is_insecure)
    all_secure = insecure_count == 0 and missing_count == 0

    return {
        "env_file": str(env_path),
        "all_secure": all_secure,
        "summary": {
            "secure": secure_count,
            "insecure": insecure_count,
            "missing": missing_count,
            "total": len(reports),
        },
        "keys": [asdict(r) for r in reports],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(env_path: Path, json_report: Path | None = None) -> int:
    """Run the credential cutover checklist.

    Args:
        env_path: Path to .env file
        json_report: Optional path to write JSON report

    Returns:
        0 if all keys are secure, 1 if any insecure or missing values found
    """
    if not env_path.exists():
        print(f"ERROR: Environment file not found: {env_path}", file=sys.stderr)
        return 2

    env_vars = _parse_env_file(env_path)
    reports = assess_env(env_vars)

    # Print human-readable table
    print_table(reports)

    # Build JSON report
    report_data = build_json_report(reports, str(env_path))

    # Write JSON report if requested
    if json_report:
        json_report.parent.mkdir(parents=True, exist_ok=True)
        json_report.write_text(
            json.dumps(report_data, indent=2) + "\n",
            encoding="utf-8",
        )
        _safe_print(f"JSON report written to: {json_report}")

    # Also print JSON to stdout for piping
    # (the table was already printed above)

    return 0 if report_data["all_secure"] else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Credential cutover checklist for production deployment."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("coordinator/.env"),
        help="Path to .env file (default: coordinator/.env)",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional path to write JSON report",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)
    return run(env_path=args.env_file, json_report=args.json_report)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
