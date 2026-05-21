"""Credential management hardening: validation, rotation tracking, and secret scanning.

Scans the project for default, weak, exposed, or hardcoded credentials and
generates an actionable audit report.

Run: python scripts/credential_audit.py [--project-root DIR] [--json] [--strict]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CredentialType(Enum):
    """Categories of credentials tracked by the auditor."""

    DATABASE = "database"
    API_KEY = "api_key"
    S3_ACCESS = "s3_access"
    S3_SECRET = "s3_secret"
    DJANGO_SECRET = "django_secret"
    RABBITMQ = "rabbitmq"
    REDIS = "redis"
    SMTP = "smtp"
    OAUTH = "oauth"
    CUSTOM = "custom"


class CredentialStatus(Enum):
    """Result of evaluating a single credential."""

    SECURE = "secure"
    WEAK = "weak"
    DEFAULT = "default"
    EXPOSED = "exposed"
    EXPIRED = "expired"
    MISSING = "missing"


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


@dataclass
class CredentialFinding:
    """A single credential audit finding."""

    credential_type: CredentialType
    status: CredentialStatus
    location: str
    message: str
    severity: str  # "critical", "high", "medium", "low"
    recommendation: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known default / placeholder values that must never reach production.
KNOWN_DEFAULTS: set[str] = {
    "minioadmin",
    "changeme",
    "password",
    "secret",
    "admin",
    "django-insecure",
    "default",
    "test",
}

# Substrings that mark a value as a placeholder.
_PLACEHOLDER_SUBSTRINGS: list[str] = [
    "changeme",
    "change-me",
    "change_me",
    "django-insecure",
    "replace-me",
    "replace_me",
    "placeholder",
    "example",
]

# Regex patterns for hardcoded secret assignments in Python source.
_HARDCODED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"""(?:_PASSWORD|_SECRET|_KEY)\s*=\s*["'][^"']{1,}["']""",
        re.IGNORECASE,
    ),
]

# Map env-var key substrings to CredentialType.
_KEY_TYPE_MAP: list[tuple[str, CredentialType]] = [
    ("s3_access", CredentialType.S3_ACCESS),
    ("s3_secret", CredentialType.S3_SECRET),
    ("django", CredentialType.DJANGO_SECRET),
    ("rabbitmq", CredentialType.RABBITMQ),
    ("redis", CredentialType.REDIS),
    ("smtp", CredentialType.SMTP),
    ("oauth", CredentialType.OAUTH),
    ("database", CredentialType.DATABASE),
    ("postgres", CredentialType.DATABASE),
    ("db_", CredentialType.DATABASE),
    ("api_key", CredentialType.API_KEY),
    ("api_secret", CredentialType.API_KEY),
]


def _classify_key(key: str) -> CredentialType:
    """Return the :class:`CredentialType` best matching *key*."""
    key_lower = key.lower()
    for substring, ctype in _KEY_TYPE_MAP:
        if substring in key_lower:
            return ctype
    return CredentialType.CUSTOM


# ---------------------------------------------------------------------------
# CredentialAuditor
# ---------------------------------------------------------------------------


class CredentialAuditor:
    """Scans a project tree for credential hygiene issues.

    Parameters
    ----------
    project_root:
        Absolute or relative path to the project root directory.
    """

    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root).resolve()

    # -- public API ---------------------------------------------------------

    def scan_env_files(self) -> list[CredentialFinding]:
        """Scan ``.env`` files for default or weak credentials."""
        findings: list[CredentialFinding] = []
        env_files = self._collect_env_files()

        for env_path in env_files:
            try:
                content = env_path.read_text(errors="replace")
            except Exception:
                continue

            for lineno, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue

                key, _, value = stripped.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if not value:
                    continue

                key_lower = key.lower()
                is_secret_key = any(
                    s in key_lower
                    for s in ("password", "secret", "key", "token")
                )
                if not is_secret_key:
                    continue

                status, msg = self.check_credential_strength(value)
                if status in (CredentialStatus.SECURE,):
                    continue

                rel = self._rel(env_path)
                location = f"{rel}:{lineno}"
                severity = "critical" if status == CredentialStatus.DEFAULT else "high"
                findings.append(
                    CredentialFinding(
                        credential_type=_classify_key(key),
                        status=status,
                        location=location,
                        message=f"{key} has {status.value} value",
                        severity=severity,
                        recommendation=msg,
                    )
                )

        return findings

    def scan_source_code(self) -> list[CredentialFinding]:
        """Scan Python / YAML source files for hardcoded secrets."""
        findings: list[CredentialFinding] = []
        extensions = ("*.py", "*.yml", "*.yaml")
        source_files: list[Path] = []
        for ext in extensions:
            source_files.extend(self.project_root.glob(ext))
            source_files.extend(self.project_root.glob(f"**/{ext}"))

        # De-duplicate (glob + recursive glob may overlap)
        seen: set[Path] = set()
        unique: list[Path] = []
        for p in source_files:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)

        for src_path in unique:
            # Skip .env.example-style files
            if "example" in src_path.name.lower():
                continue
            # Skip test files
            if "test" in src_path.name.lower():
                continue
            # Skip node_modules / __pycache__
            parts_lower = [p.lower() for p in src_path.parts]
            if "node_modules" in parts_lower or "__pycache__" in parts_lower:
                continue

            try:
                content = src_path.read_text(errors="replace")
            except Exception:
                continue

            for lineno, line in enumerate(content.splitlines(), 1):
                line_stripped = line.strip()
                if line_stripped.startswith("#"):
                    continue
                # Ignore env lookups
                if "os.environ" in line or "getenv" in line:
                    continue

                for pat in _HARDCODED_PATTERNS:
                    match = pat.search(line)
                    if match:
                        rel = self._rel(src_path)
                        location = f"{rel}:{lineno}"
                        findings.append(
                            CredentialFinding(
                                credential_type=CredentialType.CUSTOM,
                                status=CredentialStatus.EXPOSED,
                                location=location,
                                message=f"Potential hardcoded secret in {rel}",
                                severity="high",
                                recommendation=(
                                    "Move credential to environment variable or secrets manager."
                                ),
                            )
                        )
                        break  # one finding per line

        return findings

    def check_credential_strength(
        self, value: str
    ) -> tuple[CredentialStatus, str]:
        """Evaluate password / token strength.

        Returns a ``(status, recommendation)`` tuple.
        """
        if not value or not value.strip():
            return (
                CredentialStatus.MISSING,
                "Credential is empty. Provide a strong value.",
            )

        normalised = value.strip().lower()

        # Check exact known defaults
        if normalised in KNOWN_DEFAULTS:
            return (
                CredentialStatus.DEFAULT,
                f"'{value}' is a well-known default. Generate a unique credential.",
            )

        # Check placeholder substrings
        for sub in _PLACEHOLDER_SUBSTRINGS:
            if sub in normalised:
                return (
                    CredentialStatus.DEFAULT,
                    f"Value contains placeholder substring '{sub}'. Use a real credential.",
                )

        # Length check
        if len(value) < 12:
            return (
                CredentialStatus.WEAK,
                "Credential is shorter than 12 characters. Use at least 12.",
            )

        # All lowercase
        if value.isalpha() and value == value.lower():
            return (
                CredentialStatus.WEAK,
                "Credential is all lowercase letters. Mix case, digits, and symbols.",
            )

        # All digits
        if value.isdigit():
            return (
                CredentialStatus.WEAK,
                "Credential is all digits. Mix letters, digits, and symbols.",
            )

        return (CredentialStatus.SECURE, "Credential meets strength requirements.")

    def audit_all(self) -> list[CredentialFinding]:
        """Run every available scan and return aggregated findings."""
        findings: list[CredentialFinding] = []
        findings.extend(self.scan_env_files())
        findings.extend(self.scan_source_code())
        return findings

    def generate_report(self, findings: list[CredentialFinding]) -> dict:
        """Build a summary report dictionary from *findings*."""
        by_severity: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        by_status: dict[str, int] = {}
        for f in findings:
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
            status_val = f.status.value
            by_status[status_val] = by_status.get(status_val, 0) + 1

        return {
            "total_findings": len(findings),
            "by_severity": by_severity,
            "by_status": by_status,
            "findings": [
                {
                    "credential_type": f.credential_type.value,
                    "status": f.status.value,
                    "location": f.location,
                    "message": f.message,
                    "severity": f.severity,
                    "recommendation": f.recommendation,
                }
                for f in findings
            ],
        }

    # -- helpers ------------------------------------------------------------

    def _collect_env_files(self) -> list[Path]:
        """Gather ``.env`` files under the project root."""
        env_files: list[Path] = []
        for pattern in (".env", "*.env", "**/.env", "**/*.env"):
            env_files.extend(self.project_root.glob(pattern))

        # Explicit coordinator/.env
        coord_env = self.project_root / "coordinator" / ".env"
        if coord_env.exists() and coord_env not in env_files:
            env_files.append(coord_env)

        # De-duplicate preserving order
        seen: set[Path] = set()
        unique: list[Path] = []
        for p in env_files:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)

        # Exclude .env.example files
        return [p for p in unique if ".example" not in p.name]

    def _rel(self, path: Path) -> str:
        """Return *path* relative to project root (as a string)."""
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Locate the project root by looking for ``version.py``."""
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "version.py").exists():
        return candidate
    candidate = Path.cwd()
    if (candidate / "version.py").exists():
        return candidate
    return Path.cwd()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Credential audit scanner for EDCOCR",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any critical or high severity findings",
    )
    args = parser.parse_args(argv)

    root = Path(args.project_root) if args.project_root else _find_project_root()
    auditor = CredentialAuditor(str(root))
    findings = auditor.audit_all()
    report = auditor.generate_report(findings)

    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)

    if args.strict:
        crit_high = report["by_severity"].get("critical", 0) + report["by_severity"].get("high", 0)
        if crit_high > 0:
            sys.exit(1)


def _print_report(report: dict) -> None:
    """Pretty-print a credential audit report to stdout."""
    total = report["total_findings"]
    print("=" * 60)
    print("Credential Audit Report")
    print("=" * 60)
    print(f"\nTotal findings: {total}")
    print(f"  Critical: {report['by_severity'].get('critical', 0)}")
    print(f"  High:     {report['by_severity'].get('high', 0)}")
    print(f"  Medium:   {report['by_severity'].get('medium', 0)}")
    print(f"  Low:      {report['by_severity'].get('low', 0)}")
    print()

    if not report["findings"]:
        print("No credential issues found. ✓")
        return

    for f in report["findings"]:
        icon = "!" if f["severity"] in ("critical", "high") else "·"
        print(f"  [{f['severity'].upper():>8}] {icon} {f['location']}")
        print(f"             {f['message']}")
        print(f"             Fix: {f['recommendation']}")
        print()


if __name__ == "__main__":
    main()
