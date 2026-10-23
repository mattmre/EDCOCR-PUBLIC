#!/usr/bin/env python3
"""Pre-release checklist validator.

Validates that all release artifacts are ready for distribution:

  A) Version consistency  -- all sources agree on version
  B) CHANGELOG            -- current version has an entry with content
  C) Git state            -- clean tree, on main, up-to-date with remote
  D) Local validation     -- complete root test suite and release-state checks
  E) Docker images        -- can build successfully (optional, --check-docker)
  F) SDK packages         -- Python package builds (optional, --check-sdks)
  G) Release evidence     -- evidence bundle exists for current version (optional)
  H) Billing/SLA policy   -- locked cost and SLA contracts match runtime sources
  I) Tag readiness        -- version tag does not already exist

Usage:
    python scripts/release_checklist.py [--check-docker] [--check-sdks] [--json] [--report PATH]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.check_version_consistency import read_version_py
    from scripts.release_evidence_contract import verify_release_evidence_bundle
    from scripts.validate_billing_sla import run_validation as validate_billing_sla
except ImportError:  # Direct execution: scripts/ is sys.path[0].
    from check_version_consistency import read_version_py
    from release_evidence_contract import verify_release_evidence_bundle
    from validate_billing_sla import run_validation as validate_billing_sla

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def read_current_version(root: Path) -> str:
    """Read the canonical version through ``version.py`` or its forwarding shim."""
    return read_version_py(root / "version.py")


# ---------------------------------------------------------------------------
# Individual checklist items
# ---------------------------------------------------------------------------


def check_version_consistency(root: Path) -> dict:
    """A) All version sources agree."""
    script_path = root / "scripts" / "check_version_consistency.py"
    if not script_path.exists():
        return {
            "name": "Version consistency",
            "passed": False,
            "detail": f"Error: {script_path} not found",
        }
    try:
        spec = importlib.util.spec_from_file_location(
            "check_version_consistency", str(script_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        versions = mod.collect_versions(root)
        ok, report = mod.check_consistency(versions)
        return {
            "name": "Version consistency",
            "passed": ok,
            "detail": report.splitlines()[0] if report else "",
        }
    except Exception as exc:
        return {
            "name": "Version consistency",
            "passed": False,
            "detail": f"Error: {exc}",
        }


def check_changelog(root: Path) -> dict:
    """B) CHANGELOG has entry for current version with content."""
    changelog_path = root / "CHANGELOG.md"
    if not changelog_path.exists():
        return {
            "name": "CHANGELOG entry",
            "passed": False,
            "detail": "CHANGELOG.md not found",
        }

    try:
        version = read_current_version(root)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "name": "CHANGELOG entry",
            "passed": False,
            "detail": f"Cannot read version: {exc}",
        }

    text = changelog_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^##\s*\[" + re.escape(version) + r"\][^\n]*\n?", re.MULTILINE
    )
    match = pattern.search(text)
    if not match:
        return {
            "name": "CHANGELOG entry",
            "passed": False,
            "detail": f"No entry for [{version}]",
        }

    # Check for content after the header
    start = match.end()
    next_section = re.search(r"^## ", text[start:], re.MULTILINE)
    if next_section:
        section_text = text[start : start + next_section.start()]
    else:
        section_text = text[start:]

    has_content = len(section_text.strip()) > 0
    if not has_content:
        return {
            "name": "CHANGELOG entry",
            "passed": False,
            "detail": f"Entry for [{version}] exists but has no content",
        }

    return {
        "name": "CHANGELOG entry",
        "passed": True,
        "detail": f"Entry for [{version}] found with content",
    }


def check_git_state(root: Path) -> dict:
    """C) Working tree is clean, on main, up-to-date with remote."""
    issues = []

    try:
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=30,
        )
        if result.stdout.strip():
            issues.append("uncommitted changes")

        # Check current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=10,
        )
        branch = result.stdout.strip()
        if branch != "main":
            issues.append(f"on branch '{branch}', not 'main'")

        # Check if up-to-date with remote
        subprocess.run(
            ["git", "fetch", "origin", "main"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=30,
        )
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=10,
        )
        behind = result.stdout.strip()
        if behind and int(behind) > 0:
            issues.append(f"behind origin/main by {behind} commits")

    except (subprocess.TimeoutExpired, OSError) as exc:
        issues.append(f"git error: {exc}")

    if issues:
        return {
            "name": "Git state",
            "passed": False,
            "detail": "; ".join(issues),
        }
    return {
        "name": "Git state",
        "passed": True,
        "detail": f"Clean tree on {branch}, up-to-date with origin/main",
    }


def check_local_validation(root: Path) -> dict:
    """D) Run the authoritative local release validation gate."""
    try:
        state_result = subprocess.run(
            [sys.executable, "scripts/verify_release_state.py"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=60,
        )
        if state_result.returncode != 0:
            return {
                "name": "Local validation",
                "passed": False,
                "detail": "Release-state verification failed: "
                + (state_result.stderr or state_result.stdout).strip()[:200],
            }

        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=1800,
        )
        output = test_result.stdout + test_result.stderr
        passed_match = re.search(r"(\d+) passed", output)
        passed = int(passed_match.group(1)) if passed_match else 0
        failed_match = re.search(r"(\d+) failed", output)
        failed = int(failed_match.group(1)) if failed_match else 0
        skipped_match = re.search(r"(\d+) skipped", output)
        skipped = int(skipped_match.group(1)) if skipped_match else 0
        if test_result.returncode == 0 and failed == 0 and passed > 0:
            return {
                "name": "Local validation",
                "passed": True,
                "detail": (
                    f"release-state verified; {passed} passed locally"
                    + (f", {skipped} skipped" if skipped else "")
                ),
            }
        return {
            "name": "Local validation",
            "passed": False,
            "detail": f"local test suite failed: {passed} passed, {failed} failed",
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "name": "Local validation",
            "passed": False,
            "detail": f"Error: {exc}",
        }


def check_docker_build(root: Path) -> dict:
    """E) Docker image builds successfully (optional)."""
    try:
        result = subprocess.run(
            ["docker", "compose", "build", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=300,
        )
        if result.returncode == 0:
            return {
                "name": "Docker build",
                "passed": True,
                "detail": "docker compose build --dry-run succeeded",
            }
        # Fall back to checking Dockerfile exists
        dockerfile = root / "Dockerfile"
        if dockerfile.exists():
            return {
                "name": "Docker build",
                "passed": False,
                "detail": f"Build failed: {result.stderr.strip()[:200]}",
            }
        return {
            "name": "Docker build",
            "passed": False,
            "detail": "Dockerfile not found",
        }
    except FileNotFoundError:
        return {
            "name": "Docker build",
            "passed": False,
            "detail": "docker not found",
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "name": "Docker build",
            "passed": False,
            "detail": f"Error: {exc}",
        }


def check_sdk_packages(root: Path) -> dict:
    """F) Python SDK package builds (optional)."""
    sdk_path = root / "sdk" / "python"
    pyproject = sdk_path / "pyproject.toml"
    if not pyproject.exists():
        return {
            "name": "SDK packages",
            "passed": False,
            "detail": "sdk/python/pyproject.toml not found",
        }

    try:
        result = subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--no-isolation"],
            capture_output=True,
            text=True,
            cwd=str(sdk_path),
            timeout=120,
        )
        if result.returncode == 0:
            return {
                "name": "SDK packages",
                "passed": True,
                "detail": "Python SDK sdist build succeeded",
            }
        return {
            "name": "SDK packages",
            "passed": False,
            "detail": f"Build failed: {result.stderr.strip()[:200]}",
        }
    except FileNotFoundError:
        return {
            "name": "SDK packages",
            "passed": False,
            "detail": "python build module not available",
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "name": "SDK packages",
            "passed": False,
            "detail": f"Error: {exc}",
        }


def check_release_evidence(
    root: Path,
    evidence_dir: Path | None = None,
    signing_fingerprint: str | None = None,
) -> dict:
    """G) Production evidence bundle is complete, intact, and signed."""
    try:
        version = read_current_version(root)
    except (FileNotFoundError, ValueError):
        return {
            "name": "Release evidence",
            "passed": False,
            "detail": "Cannot read version",
        }

    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "name": "Release evidence",
            "passed": False,
            "detail": f"Cannot read source commit: {exc}",
        }
    expected_commit = commit_result.stdout.strip()
    if commit_result.returncode != 0 or not expected_commit:
        return {
            "name": "Release evidence",
            "passed": False,
            "detail": "Cannot read source commit",
        }

    candidates = (
        [Path(evidence_dir)]
        if evidence_dir is not None
        else [
            root / "docs" / "release-evidence" / f"v{version}",
            root / "release-evidence" / f"v{version}",
        ]
    )

    existing = [candidate.resolve() for candidate in candidates if candidate.is_dir()]
    if not existing:
        requested = f" at {evidence_dir}" if evidence_dir is not None else ""
        return {
            "name": "Release evidence",
            "passed": False,
            "detail": f"No evidence bundle found for v{version}{requested}",
        }

    verification = verify_release_evidence_bundle(
        existing[0],
        expected_version=version,
        expected_commit=expected_commit,
        require_production=True,
        require_signature=True,
        expected_signer_fingerprint=signing_fingerprint,
    )
    if verification.valid:
        return {
            "name": "Release evidence",
            "passed": True,
            "detail": f"Verified signed production bundle: {existing[0]}",
        }

    detail = "; ".join(verification.errors[:3])
    if len(verification.errors) > 3:
        detail += f" (+{len(verification.errors) - 3} more)"

    return {
        "name": "Release evidence",
        "passed": False,
        "detail": f"Evidence bundle invalid: {detail}",
    }


def check_billing_sla_policy(root: Path) -> dict:
    """H) Locked billing and SLA policy matches active runtime sources."""
    try:
        report = validate_billing_sla(root)
    except (OSError, ValueError) as exc:
        return {
            "name": "Billing/SLA policy",
            "passed": False,
            "detail": f"Validation error: {exc}",
        }

    total = report.passed + report.failed
    if report.all_passed:
        detail = f"{report.passed}/{total} policy checks passed"
    else:
        failures = [check.name for check in report.checks if not check.passed]
        detail = f"{report.failed}/{total} policy checks failed: " + ", ".join(
            failures[:3]
        )
        if len(failures) > 3:
            detail += f" (+{len(failures) - 3} more)"

    return {
        "name": "Billing/SLA policy",
        "passed": report.all_passed,
        "detail": detail,
    }


def check_tag_readiness(root: Path) -> dict:
    """I) Version tag does not already exist."""
    try:
        version = read_current_version(root)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "name": "Tag readiness",
            "passed": False,
            "detail": f"Cannot read version: {exc}",
        }

    tag = f"v{version}"
    try:
        result = subprocess.run(
            ["git", "tag", "--list", tag],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=10,
        )
        if result.stdout.strip():
            return {
                "name": "Tag readiness",
                "passed": False,
                "detail": f"Tag '{tag}' already exists",
            }
        return {
            "name": "Tag readiness",
            "passed": True,
            "detail": f"Tag '{tag}' does not exist (ready to create)",
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "name": "Tag readiness",
            "passed": False,
            "detail": f"git error: {exc}",
        }


# ---------------------------------------------------------------------------
# Checklist orchestrator
# ---------------------------------------------------------------------------


def run_checklist(
    root: Path,
    *,
    check_docker: bool = False,
    check_sdks: bool = False,
    evidence_dir: Path | None = None,
    signing_fingerprint: str | None = None,
) -> list[dict]:
    """Run all checklist items and return results."""
    results = []

    # Always run core checks
    results.append(check_version_consistency(root))
    results.append(check_changelog(root))
    results.append(check_git_state(root))
    results.append(check_local_validation(root))

    # Optional checks
    if check_docker:
        results.append(check_docker_build(root))
    if check_sdks:
        results.append(check_sdk_packages(root))

    # Always run lightweight checks
    results.append(
        check_release_evidence(
            root,
            evidence_dir=evidence_dir,
            signing_fingerprint=signing_fingerprint,
        )
    )
    results.append(check_billing_sla_policy(root))
    results.append(check_tag_readiness(root))

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_text_table(results: list[dict]) -> str:
    """Format results as a human-readable text table."""
    lines = []
    lines.append("=" * 60)
    lines.append("Pre-Release Checklist")
    lines.append("=" * 60)
    lines.append("")

    # Column widths
    name_width = max(len(r["name"]) for r in results) + 2

    for r in results:
        marker = "[PASS]" if r["passed"] else "[FAIL]"
        name = r["name"].ljust(name_width)
        lines.append(f"  {marker} {name} {r['detail']}")

    lines.append("")
    lines.append("-" * 60)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    all_pass = passed == total

    verdict = "READY" if all_pass else "NOT READY"
    lines.append(f"Result: {verdict} ({passed}/{total} checks passed)")
    lines.append("-" * 60)

    return "\n".join(lines)


def format_markdown_report(results: list[dict]) -> str:
    """Format results as a markdown report."""
    lines = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    all_pass = passed == total

    lines.append("# Pre-Release Checklist Report")
    lines.append("")
    lines.append(f"**Generated**: {timestamp}")
    lines.append(f"**Verdict**: {'READY' if all_pass else 'NOT READY'}")
    lines.append(f"**Checks**: {passed}/{total} passed")
    lines.append("")

    lines.append("| Check | Status | Detail |")
    lines.append("|-------|--------|--------|")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = r["detail"].replace("|", "\\|")
        lines.append(f"| {r['name']} | {status} | {detail} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point. Returns 0 if all checks pass, 1 otherwise."""
    parser = argparse.ArgumentParser(
        description="Pre-release checklist: validate all release artifacts."
    )
    parser.add_argument(
        "--check-docker",
        action="store_true",
        help="Include Docker build check",
    )
    parser.add_argument(
        "--check-sdks",
        action="store_true",
        help="Include SDK package build check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="Write markdown report to the given file path",
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        help="Project root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--evidence-dir",
        metavar="DIR",
        type=Path,
        help="Protected release evidence bundle to verify",
    )
    parser.add_argument(
        "--signing-fingerprint",
        help="Approved full GPG fingerprint for the release evidence signer",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT

    results = run_checklist(
        root,
        check_docker=args.check_docker,
        check_sdks=args.check_sdks,
        evidence_dir=args.evidence_dir,
        signing_fingerprint=args.signing_fingerprint,
    )

    all_pass = all(r["passed"] for r in results)

    if args.json:
        output = {
            "verdict": "READY" if all_pass else "NOT READY",
            "checks": results,
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_text_table(results))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(format_markdown_report(results), encoding="utf-8")
        if not args.json:
            print(f"\nReport written to: {report_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
