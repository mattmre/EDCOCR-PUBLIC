#!/usr/bin/env python3
"""Master release-state verification script.

Aggregates multiple verification checks into a single orchestrated run:

A) Extended Version Consistency (14 sources)
   - Base 7 sources from check_version_consistency.collect_versions
   - Additional: openapi.json, api/tracing.py, otel config, TS SDK User-Agent,
     CHANGELOG.md, sdk/typescript/ocr_client.ts
B) Docs Canon Validation (13-file spine from docs/release-candidate.md)
C) Helm Chart Consistency
D) CHANGELOG Validation
E) CI Workflow Integrity
F) Summary Report (text, --json, --report)

Exits 0 if all checks pass, 1 if any fail.
"""

import argparse
import importlib.util
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LOG = logging.getLogger("verify_release_state")

# Canonical docs spine from docs/release-candidate.md (13 files)
DOCS_CANON_SPINE = [
    "docs/README.md",
    "docs/EXECUTIVE-SUMMARY.md",
    "docs/00-SYSTEM-BLUEPRINT.md",
    "docs/01-TECH-STACK-DNA.md",
    "docs/02-QUICKSTART-5-MINUTE-SUCCESS.md",
    "docs/03-INFORMATION-FLOWS.md",
    "docs/04-USE-CASES.md",
    "docs/05-INTERACTIVE-WALKTHROUGH.md",
    "docs/06-CONFIGURATION-REFERENCE.md",
    "docs/FAILOVER-RUNBOOK.md",
    "docs/cpu-vs-gpu-analysis.md",
    "docs/operations/production-cutover-runbook.md",
    "docs/API-REFERENCE.md",
]

# CI workflow files that must exist
CI_WORKFLOW_FILES = [
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".github/workflows/docker-publish.yml",
    ".github/workflows/sdk-publish.yml",
]

# Python versions expected in CI. This is release-state truth only: it verifies
# that CI advertises the migration target without implying Docker/runtime proof.
CI_PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]

# Additional version source paths (beyond the base 7)
EXTENDED_VERSION_SOURCES = {
    "docs/openapi.json": "docs/openapi.json",
    "api/tracing.py": "api/tracing.py",
    "otel/otel-collector-config.yaml": "otel/otel-collector-config.yaml",
    "sdk/typescript/ocr_client.ts": "sdk/typescript/ocr_client.ts",
    "CHANGELOG.md": "CHANGELOG.md",
}


# ---------------------------------------------------------------------------
# Version extraction helpers
# ---------------------------------------------------------------------------


def _load_check_version_consistency_module():
    """Load the sibling version-consistency module with its reader helpers."""
    existing = sys.modules.get("check_version_consistency")
    if existing is not None and hasattr(existing, "read_version_py"):
        return existing

    script_path = SCRIPT_DIR / "check_version_consistency.py"
    spec = importlib.util.spec_from_file_location(
        "_verify_release_state_check_version_consistency", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load check_version_consistency from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_canonical_version(root: Path) -> str:
    """Read the canonical version from version.py."""
    module = _load_check_version_consistency_module()
    return module.read_version_py(root / "version.py")


def extract_openapi_version(path: Path) -> str:
    """Extract info.version from openapi.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("info", {}).get("version")
    if not version:
        raise ValueError(f"No info.version found in {path}")
    return version


def extract_tracing_version(path: Path) -> str:
    """Extract _SERVICE_VERSION_DEFAULT from api/tracing.py."""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'^_SERVICE_VERSION_DEFAULT\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE
    )
    if not match:
        raise ValueError(f"No _SERVICE_VERSION_DEFAULT found in {path}")
    return match.group(1)


def extract_otel_config_version(path: Path) -> str:
    """Extract service.version from otel-collector-config.yaml (regex)."""
    text = path.read_text(encoding="utf-8")
    # Look for: key: service.version / value: "X.Y.Z"
    match = re.search(
        r'key:\s*service\.version\s*\n\s*value:\s*["\']?([^"\'"\n]+)["\']?',
        text,
    )
    if not match:
        raise ValueError(f"No service.version resource attribute found in {path}")
    return match.group(1).strip()


def extract_ts_sdk_useragent_version(path: Path) -> str:
    """Extract version from TypeScript SDK User-Agent header string."""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"""['"]User-Agent['"]\s*:\s*['"]ocr-local-typescript-sdk/([^'"]+)['"]""",
        text,
    )
    if not match:
        raise ValueError(f"No User-Agent version found in {path}")
    return match.group(1)


def extract_changelog_version(path: Path) -> str:
    """Extract the latest numbered release section header from CHANGELOG.md.

    The changelog may start with an ``[Unreleased]`` section. Release-state
    verification should compare the canonical version to the latest numbered
    release entry instead of that rolling placeholder.
    """
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r'^##\s*\[([^\]]+)\]', text, re.MULTILINE)
    for version in matches:
        if re.fullmatch(r"\d+\.\d+\.\d+", version):
            return version
    raise ValueError(f"No numbered version section header found in {path}")


# Map of extended source labels to their extraction functions
EXTENDED_EXTRACTORS = {
    "docs/openapi.json": extract_openapi_version,
    "api/tracing.py": extract_tracing_version,
    "otel/otel-collector-config.yaml": extract_otel_config_version,
    "sdk/typescript/ocr_client.ts": extract_ts_sdk_useragent_version,
    "CHANGELOG.md": extract_changelog_version,
}


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_version_consistency(root: Path) -> dict:
    """A) Extended version consistency check across 14 sources.

    Returns dict with 'passed', 'canonical', 'sources', 'mismatches'.
    """
    errors = []

    # Import base 7-source collector
    try:
        sys.path.insert(0, str(root / "scripts"))
        from check_version_consistency import collect_versions

        base_versions = collect_versions(root)
    except Exception as exc:
        errors.append(f"Failed to import check_version_consistency: {exc}")
        base_versions = {}
    finally:
        if str(root / "scripts") in sys.path:
            sys.path.remove(str(root / "scripts"))

    # Collect extended sources
    extended_versions = {}
    for label, rel_path in EXTENDED_VERSION_SOURCES.items():
        full_path = root / rel_path
        if not full_path.exists():
            extended_versions[label] = f"FILE_NOT_FOUND ({rel_path})"
            continue
        try:
            extended_versions[label] = EXTENDED_EXTRACTORS[label](full_path)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            extended_versions[label] = f"PARSE_ERROR ({exc})"

    # Merge all versions
    all_versions = {**base_versions, **extended_versions}

    # Read canonical version
    try:
        canonical = read_canonical_version(root)
    except (FileNotFoundError, ValueError):
        canonical = None

    # Find mismatches
    mismatches = {}
    for label, ver in all_versions.items():
        if canonical and ver != canonical:
            mismatches[label] = ver

    passed = len(mismatches) == 0 and len(errors) == 0 and canonical is not None
    return {
        "passed": passed,
        "canonical": canonical,
        "sources": all_versions,
        "mismatches": mismatches,
        "errors": errors,
        "total_sources": len(all_versions),
    }


def check_docs_canon(root: Path) -> dict:
    """B) Docs canon validation.

    Checks that all 13 files in the canonical docs spine exist and are non-empty.
    """
    missing = []
    empty = []
    present = []

    for rel_path in DOCS_CANON_SPINE:
        full_path = root / rel_path
        if not full_path.exists():
            missing.append(rel_path)
        elif full_path.stat().st_size == 0:
            empty.append(rel_path)
        else:
            present.append(rel_path)

    passed = len(missing) == 0 and len(empty) == 0
    return {
        "passed": passed,
        "total_files": len(DOCS_CANON_SPINE),
        "present": len(present),
        "missing": missing,
        "empty": empty,
    }


def check_helm_chart(root: Path) -> dict:
    """C) Helm chart consistency.

    Verifies appVersion matches version.py and chart version is set.
    """
    chart_path = root / "helm" / "ocr-local" / "Chart.yaml"
    result = {
        "passed": False,
        "chart_path": "helm/ocr-local/Chart.yaml",
        "app_version": None,
        "chart_version": None,
        "canonical_version": None,
        "app_version_matches": False,
        "chart_version_set": False,
    }

    if not chart_path.exists():
        result["error"] = "Chart.yaml not found"
        return result

    text = chart_path.read_text(encoding="utf-8")

    # Extract appVersion
    app_match = re.search(
        r'^appVersion:\s*["\']?([^"\'"\n]+)["\']?', text, re.MULTILINE
    )
    if app_match:
        result["app_version"] = app_match.group(1).strip()

    # Extract chart version
    ver_match = re.search(r'^version:\s*["\']?([^"\'"\n]+)["\']?', text, re.MULTILINE)
    if ver_match:
        result["chart_version"] = ver_match.group(1).strip()

    # Read canonical
    try:
        canonical = read_canonical_version(root)
        result["canonical_version"] = canonical
    except (FileNotFoundError, ValueError):
        canonical = None

    result["app_version_matches"] = (
        result["app_version"] == canonical if canonical else False
    )
    result["chart_version_set"] = bool(
        result["chart_version"] and result["chart_version"].strip()
    )
    result["passed"] = result["app_version_matches"] and result["chart_version_set"]
    return result


def check_changelog(root: Path) -> dict:
    """D) CHANGELOG validation.

    Verifies that CHANGELOG.md has an entry for the current version with content.
    """
    changelog_path = root / "CHANGELOG.md"
    result = {
        "passed": False,
        "changelog_exists": False,
        "has_current_version_entry": False,
        "entry_has_content": False,
        "latest_version": None,
        "canonical_version": None,
    }

    if not changelog_path.exists():
        return result

    result["changelog_exists"] = True
    text = changelog_path.read_text(encoding="utf-8")

    try:
        canonical = read_canonical_version(root)
        result["canonical_version"] = canonical
    except (FileNotFoundError, ValueError):
        return result

    # Check for a section header matching the canonical version.
    # Match the full header line so we skip the date portion.
    pattern = re.compile(
        r'^##\s*\[' + re.escape(canonical) + r'\][^\n]*\n?', re.MULTILINE
    )
    match = pattern.search(text)
    if match:
        result["has_current_version_entry"] = True
        # Check that there's content after the header line before the next ##
        start = match.end()
        next_section = re.search(r'^## ', text[start:], re.MULTILINE)
        if next_section:
            section_text = text[start : start + next_section.start()]
        else:
            section_text = text[start:]
        # Strip whitespace and check for non-empty content
        result["entry_has_content"] = len(section_text.strip()) > 0

    # Extract latest version
    latest_match = re.search(r'^##\s*\[([^\]]+)\]', text, re.MULTILINE)
    if latest_match:
        result["latest_version"] = latest_match.group(1)

    result["passed"] = (
        result["has_current_version_entry"] and result["entry_has_content"]
    )
    return result


def check_ci_workflows(root: Path) -> dict:
    """E) CI workflow integrity.

    Verifies key CI workflow files exist and reference correct Python versions.
    """
    missing_files = []
    present_files = []
    python_version_issues = []

    for rel_path in CI_WORKFLOW_FILES:
        full_path = root / rel_path
        if full_path.exists():
            present_files.append(rel_path)
        else:
            missing_files.append(rel_path)

    # Check ci.yml for Python versions
    ci_path = root / ".github" / "workflows" / "ci.yml"
    if ci_path.exists():
        ci_text = ci_path.read_text(encoding="utf-8")
        for py_ver in CI_PYTHON_VERSIONS:
            # Check for the version in quotes (as in YAML matrix)
            if f'"{py_ver}"' not in ci_text and f"'{py_ver}'" not in ci_text:
                python_version_issues.append(
                    f"Python {py_ver} not found in ci.yml matrix"
                )

    passed = len(missing_files) == 0 and len(python_version_issues) == 0
    return {
        "passed": passed,
        "total_expected": len(CI_WORKFLOW_FILES),
        "present": present_files,
        "missing": missing_files,
        "python_version_issues": python_version_issues,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_text_report(results: dict) -> str:
    """Generate a human-readable text summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("Release State Verification Report")
    lines.append("=" * 60)
    lines.append("")

    all_passed = all(r["passed"] for r in results.values())

    # A) Version consistency
    vc = results["version_consistency"]
    status = "PASS" if vc["passed"] else "FAIL"
    lines.append(f"[{status}] A) Version Consistency ({vc['total_sources']} sources)")
    if vc["canonical"]:
        lines.append(f"       Canonical version: {vc['canonical']}")
    if vc.get("errors"):
        for err in vc["errors"]:
            lines.append(f"       ERROR: {err}")
    if vc["mismatches"]:
        for label, ver in sorted(vc["mismatches"].items()):
            lines.append(f"       MISMATCH: {label} = {ver}")
    lines.append("")

    # B) Docs canon
    dc = results["docs_canon"]
    status = "PASS" if dc["passed"] else "FAIL"
    lines.append(
        f"[{status}] B) Docs Canon ({dc['present']}/{dc['total_files']} present)"
    )
    if dc["missing"]:
        for f in dc["missing"]:
            lines.append(f"       MISSING: {f}")
    if dc["empty"]:
        for f in dc["empty"]:
            lines.append(f"       EMPTY: {f}")
    lines.append("")

    # C) Helm chart
    hc = results["helm_chart"]
    status = "PASS" if hc["passed"] else "FAIL"
    lines.append(f"[{status}] C) Helm Chart Consistency")
    if hc.get("app_version"):
        lines.append(f"       appVersion: {hc['app_version']}")
    if hc.get("chart_version"):
        lines.append(f"       chartVersion: {hc['chart_version']}")
    if not hc["app_version_matches"]:
        lines.append("       appVersion does not match version.py")
    lines.append("")

    # D) CHANGELOG
    cl = results["changelog"]
    status = "PASS" if cl["passed"] else "FAIL"
    lines.append(f"[{status}] D) CHANGELOG Validation")
    if cl.get("latest_version"):
        lines.append(f"       Latest entry: [{cl['latest_version']}]")
    if not cl["has_current_version_entry"]:
        lines.append("       No entry for current version")
    elif not cl["entry_has_content"]:
        lines.append("       Entry exists but has no content")
    lines.append("")

    # E) CI workflows
    ci = results["ci_workflows"]
    status = "PASS" if ci["passed"] else "FAIL"
    lines.append(
        f"[{status}] E) CI Workflow Integrity "
        f"({len(ci['present'])}/{ci['total_expected']} files)"
    )
    if ci["missing"]:
        for f in ci["missing"]:
            lines.append(f"       MISSING: {f}")
    if ci["python_version_issues"]:
        for issue in ci["python_version_issues"]:
            lines.append(f"       ISSUE: {issue}")
    lines.append("")

    # Summary
    lines.append("-" * 60)
    overall = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    lines.append(f"Overall: {overall}")
    lines.append("-" * 60)

    return "\n".join(lines)


def generate_markdown_report(results: dict) -> str:
    """Generate a markdown report suitable for file output."""
    lines = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    all_passed = all(r["passed"] for r in results.values())

    lines.append("# Release State Verification Report")
    lines.append("")
    lines.append(f"**Generated**: {timestamp}")
    lines.append(
        f"**Overall**: {'PASS' if all_passed else 'FAIL'}"
    )
    lines.append("")

    # Summary table
    lines.append("| Check | Status |")
    lines.append("|-------|--------|")
    check_labels = {
        "version_consistency": "A) Version Consistency",
        "docs_canon": "B) Docs Canon Validation",
        "helm_chart": "C) Helm Chart Consistency",
        "changelog": "D) CHANGELOG Validation",
        "ci_workflows": "E) CI Workflow Integrity",
    }
    for key, label in check_labels.items():
        status = "PASS" if results[key]["passed"] else "FAIL"
        lines.append(f"| {label} | {status} |")
    lines.append("")

    # A) Version consistency details
    vc = results["version_consistency"]
    lines.append("## A) Version Consistency")
    lines.append("")
    lines.append(
        f"- **Canonical version**: {vc.get('canonical', 'UNKNOWN')}"
    )
    lines.append(f"- **Total sources checked**: {vc['total_sources']}")
    lines.append(f"- **Mismatches**: {len(vc['mismatches'])}")
    lines.append(f"- **Errors**: {len(vc.get('errors', []))}")
    if vc.get("errors"):
        lines.append("")
        for err in vc["errors"]:
            lines.append(f"- **Error**: {err}")
    if vc["mismatches"]:
        lines.append("")
        lines.append("| Source | Found Version |")
        lines.append("|--------|---------------|")
        for label, ver in sorted(vc["mismatches"].items()):
            lines.append(f"| `{label}` | `{ver}` |")
    lines.append("")

    # B) Docs canon details
    dc = results["docs_canon"]
    lines.append("## B) Docs Canon Validation")
    lines.append("")
    lines.append(f"- **Files expected**: {dc['total_files']}")
    lines.append(f"- **Files present**: {dc['present']}")
    if dc["missing"]:
        lines.append(f"- **Missing**: {', '.join(f'`{f}`' for f in dc['missing'])}")
    if dc["empty"]:
        lines.append(f"- **Empty**: {', '.join(f'`{f}`' for f in dc['empty'])}")
    lines.append("")

    # C) Helm chart details
    hc = results["helm_chart"]
    lines.append("## C) Helm Chart Consistency")
    lines.append("")
    lines.append(f"- **appVersion**: {hc.get('app_version', 'N/A')}")
    lines.append(f"- **chartVersion**: {hc.get('chart_version', 'N/A')}")
    lines.append(
        f"- **appVersion matches version.py**: {hc['app_version_matches']}"
    )
    lines.append("")

    # D) CHANGELOG details
    cl = results["changelog"]
    lines.append("## D) CHANGELOG Validation")
    lines.append("")
    lines.append(f"- **CHANGELOG exists**: {cl['changelog_exists']}")
    lines.append(
        f"- **Current version entry**: {cl['has_current_version_entry']}"
    )
    lines.append(f"- **Entry has content**: {cl['entry_has_content']}")
    lines.append("")

    # E) CI workflow details
    ci = results["ci_workflows"]
    lines.append("## E) CI Workflow Integrity")
    lines.append("")
    lines.append(
        f"- **Workflow files**: {len(ci['present'])}/{ci['total_expected']}"
    )
    if ci["missing"]:
        lines.append(
            f"- **Missing**: {', '.join(f'`{f}`' for f in ci['missing'])}"
        )
    if ci["python_version_issues"]:
        for issue in ci["python_version_issues"]:
            lines.append(f"- **Issue**: {issue}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all_checks(root: Path) -> dict:
    """Run all verification checks and return results dict."""
    results = {}
    results["version_consistency"] = check_version_consistency(root)
    results["docs_canon"] = check_docs_canon(root)
    results["helm_chart"] = check_helm_chart(root)
    results["changelog"] = check_changelog(root)
    results["ci_workflows"] = check_ci_workflows(root)
    return results


def main() -> int:
    """Entry point. Returns 0 if all pass, 1 if any fail."""
    parser = argparse.ArgumentParser(
        description="Master release-state verification for OCR_LOCAL"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="Write a markdown report to the given file path",
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        help="Project root directory (default: auto-detect)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT

    results = run_all_checks(root)
    all_passed = all(r["passed"] for r in results.values())

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(generate_text_report(results))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            generate_markdown_report(results), encoding="utf-8"
        )
        if not args.json:
            print(f"\nMarkdown report written to: {report_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
