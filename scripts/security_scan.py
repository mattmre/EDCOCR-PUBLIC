"""Security posture scanner for EDCOCR deployment.

Checks for common security misconfigurations and generates a report.
Run: python scripts/security_scan.py [--json] [--strict]
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Finding:
    check_id: str
    title: str
    severity: Severity
    description: str
    recommendation: str
    passed: bool
    details: str = ""


@dataclass
class ScanReport:
    findings: list = field(default_factory=list)
    passed: int = 0
    failed: int = 0

    def add(self, finding: Finding):
        self.findings.append(finding)
        if finding.passed:
            self.passed += 1
        else:
            self.failed += 1

    def to_dict(self):
        return {
            "summary": {
                "passed": self.passed,
                "failed": self.failed,
                "total": self.passed + self.failed,
            },
            "findings": [
                {
                    "check_id": f.check_id,
                    "title": f.title,
                    "severity": f.severity.value,
                    "passed": f.passed,
                    "description": f.description,
                    "recommendation": f.recommendation,
                    "details": f.details,
                }
                for f in self.findings
            ],
        }


def _find_project_root():
    """Find project root by looking for version.py."""
    p = Path(__file__).resolve().parent.parent
    if (p / "version.py").exists():
        return p
    # Fallback to cwd
    p = Path.cwd()
    if (p / "version.py").exists():
        return p
    return Path.cwd()


def _is_git_tracked(root: Path, filepath: Path) -> bool:
    """Check whether a file is tracked by git (committed or staged)."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(filepath)],
            cwd=str(root), capture_output=True, text=True,
        )
        return result.returncode == 0
    except Exception:
        # If git is unavailable, assume tracked (safer to flag than to miss)
        return True


def check_env_files(root: Path, report: ScanReport):
    """Check for secrets in committed env files."""
    env_files = list(root.rglob("*.env")) + list(root.rglob(".env"))
    # Also check coordinator/.env specifically
    coord_env = root / "coordinator" / ".env"
    if coord_env.exists() and coord_env not in env_files:
        env_files.append(coord_env)

    # Only scan files actually tracked by git — untracked local env files
    # are not a version-control risk.
    env_files = [ef for ef in env_files if _is_git_tracked(root, ef)]

    placeholder_values = {
        "changeme",
        "change-me",
        "password",
        "secret",
        "minioadmin",
        "admin",
    }

    found_secrets = False
    details_lines = []
    for ef in env_files:
        try:
            content = ef.read_text(errors="replace")
            for line in content.splitlines():
                line_stripped = line.strip()
                if line_stripped.startswith("#") or "=" not in line_stripped:
                    continue
                key, _, value = line_stripped.partition("=")
                value = value.strip().strip("'\"")
                if not value:
                    continue
                key_lower = key.strip().lower()
                if any(
                    s in key_lower
                    for s in ("password", "secret", "key", "token")
                ):
                    if value.lower() not in placeholder_values and not value.startswith(
                        "change-me"
                    ):
                        found_secrets = True
                        details_lines.append(
                            f"  {ef.relative_to(root)}: {key.strip()}=***"
                        )
        except Exception:
            pass

    report.add(
        Finding(
            check_id="SEC-001",
            title="No secrets in committed env files",
            severity=Severity.CRITICAL,
            description="Environment files should not contain real secrets in version control.",
            recommendation="Add env files to .gitignore, use .env.example with placeholders.",
            passed=not found_secrets,
            details="\n".join(details_lines)
            if details_lines
            else "No secret values detected.",
        )
    )


def check_gitignore_env(root: Path, report: ScanReport):
    """Check that .gitignore excludes env files."""
    gitignore = root / ".gitignore"
    has_env_ignore = False
    if gitignore.exists():
        content = gitignore.read_text(errors="replace")
        # Check for .env patterns
        for line in content.splitlines():
            line = line.strip()
            if line in (".env", "*.env", ".env*", "coordinator/.env"):
                has_env_ignore = True
                break

    report.add(
        Finding(
            check_id="SEC-002",
            title=".gitignore excludes .env files",
            severity=Severity.HIGH,
            description="The .gitignore should explicitly exclude .env files to prevent accidental secret commits.",
            recommendation="Add '.env' and 'coordinator/.env' to .gitignore.",
            passed=has_env_ignore,
        )
    )


def check_dockerfile_user(root: Path, report: ScanReport):
    """Check that Dockerfiles use non-root user."""
    dockerfiles = list(root.rglob("Dockerfile*"))
    all_have_user = True
    details_lines = []
    for df in dockerfiles:
        if df.name.startswith("Dockerfile"):
            try:
                content = df.read_text(errors="replace")
                has_user = bool(
                    re.search(r"^\s*USER\s+\w+", content, re.MULTILINE)
                )
                if not has_user:
                    all_have_user = False
                    details_lines.append(
                        f"  {df.relative_to(root)}: No USER directive"
                    )
            except Exception:
                pass

    report.add(
        Finding(
            check_id="SEC-003",
            title="Dockerfiles use non-root user",
            severity=Severity.CRITICAL,
            description="Containers should not run as root to limit blast radius of container escape.",
            recommendation="Add 'RUN useradd -r -s /bin/false ocr && USER ocr' to Dockerfiles.",
            passed=all_have_user,
            details="\n".join(details_lines)
            if details_lines
            else "All Dockerfiles have USER directive.",
        )
    )


def check_cors_config(root: Path, report: ScanReport):
    """Check for CORS middleware configuration."""
    main_py = root / "api" / "main.py"
    has_cors = False
    if main_py.exists():
        content = main_py.read_text(errors="replace")
        has_cors = "CORSMiddleware" in content

    report.add(
        Finding(
            check_id="SEC-004",
            title="CORS middleware configured",
            severity=Severity.MEDIUM,
            description="If browser clients access the API, CORS must be configured with explicit origin allowlist.",
            recommendation="Add CORSMiddleware if browser access is needed. Skip if API-only.",
            passed=has_cors,
            details="CORSMiddleware found."
            if has_cors
            else "No CORSMiddleware in api/main.py.",
        )
    )


def check_tls_config(root: Path, report: ScanReport):
    """Check for TLS configuration."""
    compose_files = list(root.rglob("docker-compose*.yml")) + list(
        root.rglob("docker-compose*.yaml")
    )
    has_tls = False
    for cf in compose_files:
        try:
            content = cf.read_text(errors="replace")
            if (
                "443:" in content
                or "ssl" in content.lower()
                or "tls" in content.lower()
            ):
                has_tls = True
                break
        except Exception:
            pass

    report.add(
        Finding(
            check_id="SEC-005",
            title="TLS termination configured",
            severity=Severity.HIGH,
            description="API and inter-service communication should use TLS in production.",
            recommendation="Add nginx/traefik reverse proxy with TLS, or enable TLS on services directly.",
            passed=has_tls,
        )
    )


def check_dependency_pinning(root: Path, report: ScanReport):
    """Check that dependencies are pinned."""
    req = root / "requirements.txt"
    unpinned = []
    if req.exists():
        for line in req.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Check for version specifier
            if ">=" in line and "==" not in line:
                pkg = line.split(">=")[0].strip()
                unpinned.append(pkg)

    report.add(
        Finding(
            check_id="SEC-006",
            title="Dependencies fully pinned",
            severity=Severity.MEDIUM,
            description="All production dependencies should have exact version pins for reproducible builds.",
            recommendation="Use pip freeze or pip-compile for exact pins.",
            passed=len(unpinned) == 0,
            details=f"Unpinned packages: {', '.join(unpinned)}"
            if unpinned
            else "All packages pinned.",
        )
    )


def check_auth_bypass(root: Path, report: ScanReport):
    """Check for authentication bypass flags."""
    auth_py = root / "api" / "auth.py"
    has_bypass = False
    if auth_py.exists():
        content = auth_py.read_text(errors="replace")
        has_bypass = "ALLOW_UNAUTHENTICATED" in content

    report.add(
        Finding(
            check_id="SEC-007",
            title="No auth bypass in production",
            severity=Severity.MEDIUM,
            description="ALLOW_UNAUTHENTICATED flag should not be used in production deployments.",
            recommendation="Ensure ALLOW_UNAUTHENTICATED is not set in production. Consider removing the flag or restricting to viewer role.",
            passed=not has_bypass,
            details="ALLOW_UNAUTHENTICATED flag found in auth.py. Ensure it is not set in production."
            if has_bypass
            else "",
        )
    )


def check_hardcoded_secrets(root: Path, report: ScanReport):
    """Scan Python files for hardcoded secrets."""
    secret_patterns = [
        re.compile(
            r"""(?:password|secret|token|api_key)\s*=\s*["'][^"']{8,}["']""",
            re.IGNORECASE,
        ),
    ]

    findings_list = []
    py_files = (
        list(root.glob("*.py"))
        + list(root.glob("api/*.py"))
        + list(root.glob("scripts/*.py"))
    )
    for pf in py_files:
        try:
            content = pf.read_text(errors="replace")
            for i, line in enumerate(content.splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                for pat in secret_patterns:
                    if pat.search(line):
                        # Skip test files, examples, and known safe patterns
                        if (
                            "test" in pf.name.lower()
                            or "example" in line.lower()
                            or "placeholder" in line.lower()
                        ):
                            continue
                        if "os.environ" in line or "getenv" in line:
                            continue
                        findings_list.append(f"  {pf.relative_to(root)}:{i}")
        except Exception:
            pass

    report.add(
        Finding(
            check_id="SEC-008",
            title="No hardcoded secrets in source",
            severity=Severity.HIGH,
            description="Source code should not contain hardcoded passwords, tokens, or API keys.",
            recommendation="Use environment variables or a secrets manager for all credentials.",
            passed=len(findings_list) == 0,
            details="\n".join(findings_list)
            if findings_list
            else "No hardcoded secrets detected.",
        )
    )


def check_rate_limit_auth(root: Path, report: ScanReport):
    """Check for rate limiting on authentication endpoints."""
    limits_py = root / "api" / "limits.py"
    has_auth_rate_limit = False

    if limits_py and limits_py.exists():
        content = limits_py.read_text(errors="replace")
        if "auth" in content.lower() and "limit" in content.lower():
            has_auth_rate_limit = True

    report.add(
        Finding(
            check_id="SEC-009",
            title="Auth endpoints rate-limited",
            severity=Severity.MEDIUM,
            description="Authentication endpoints should have specific rate limits to prevent brute-force attacks.",
            recommendation="Add per-IP rate limiting on failed authentication attempts (e.g., 5 failures/minute).",
            passed=has_auth_rate_limit,
        )
    )


def run_scan(root: Path) -> ScanReport:
    """Run all security checks and return report."""
    report = ScanReport()

    check_env_files(root, report)
    check_gitignore_env(root, report)
    check_dockerfile_user(root, report)
    check_cors_config(root, report)
    check_tls_config(root, report)
    check_dependency_pinning(root, report)
    check_auth_bypass(root, report)
    check_hardcoded_secrets(root, report)
    check_rate_limit_auth(root, report)

    return report


def print_report(report: ScanReport):
    """Print human-readable report."""
    print("=" * 60)
    print("Security Scan Report")
    print("=" * 60)
    print(
        f"\nPassed: {report.passed}  |  Failed: {report.failed}  |  Total: {report.passed + report.failed}\n"
    )

    # Group by severity
    for severity in Severity:
        findings = [
            f
            for f in report.findings
            if f.severity == severity and not f.passed
        ]
        if not findings:
            continue
        print(f"\n--- {severity.value.upper()} ---")
        for f in findings:
            icon = "PASS" if f.passed else "FAIL"
            print(f"  [{icon}] {f.check_id}: {f.title}")
            print(f"         {f.description}")
            if f.details:
                for line in f.details.splitlines():
                    print(f"         {line}")
            print(f"         Fix: {f.recommendation}")
            print()

    # Show passed
    passed = [f for f in report.findings if f.passed]
    if passed:
        print("\n--- PASSED ---")
        for f in passed:
            print(f"  [PASS] {f.check_id}: {f.title}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Security posture scanner for EDCOCR"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output JSON report"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any critical/high findings",
    )
    parser.add_argument(
        "--root", type=str, help="Project root directory"
    )
    args = parser.parse_args()

    root = Path(args.root) if args.root else _find_project_root()
    report = run_scan(root)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)

    if args.strict:
        critical_high = [
            f
            for f in report.findings
            if not f.passed
            and f.severity in (Severity.CRITICAL, Severity.HIGH)
        ]
        if critical_high:
            sys.exit(1)


if __name__ == "__main__":
    main()
