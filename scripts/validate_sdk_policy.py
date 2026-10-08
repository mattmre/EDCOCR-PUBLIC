#!/usr/bin/env python3
"""Validate that SDK implementations comply with the versioning policy.

Checks:
  A) Version alignment — all SDK versions match version.py
  B) Python SDK structure — pyproject.toml has required metadata
  C) TypeScript SDK structure — package.json has required fields
  D) API stability — api/versioning.py exists with endpoint registry
  E) Deprecation markers — deprecated APIs have runtime warnings
  F) Policy document — SDK-VERSIONING-POLICY.md exists and is non-empty
  G) CI publish workflow — sdk-publish.yml has Python and TypeScript jobs

Usage:
    python scripts/validate_sdk_policy.py
    python scripts/validate_sdk_policy.py --project-root /path/to/repo
    python scripts/validate_sdk_policy.py --json
    python scripts/validate_sdk_policy.py --report docs/reports/sdk-policy.md
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregated validation report."""

    passed: bool = True
    checks: list[CheckResult] = field(default_factory=list)
    timestamp: str = ""
    project_root: str = ""

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        if not result.passed:
            self.passed = False


# ---------------------------------------------------------------------------
# Safe print helper (Windows cp1252 console)
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
        if isinstance(stream, io.TextIOWrapper) and hasattr(stream, "buffer"):
            stream.buffer.write((text + str(end)).encode("utf-8", errors="replace"))
        else:
            print(text, end=end)


# ---------------------------------------------------------------------------
# Check A: Version alignment
# ---------------------------------------------------------------------------


def check_version_alignment(root: Path) -> CheckResult:
    """Verify all SDK version sources match version.py."""
    version_py = root / "version.py"
    if not version_py.exists():
        return CheckResult(
            name="version_alignment",
            passed=False,
            message="version.py not found",
        )

    canonical = _extract_canonical_version(root)
    if canonical is None:
        return CheckResult(
            name="version_alignment",
            passed=False,
            message="Could not parse __version__ from version.py or ocr_local/config/version.py",
        )

    # Collect versions from SDK sources
    sources: dict[str, str | None] = {}
    _collect_version(sources, root, "sdk/python/pyproject.toml", _extract_pyproject_version)
    _collect_version(sources, root, "sdk/python/src/edcocr_sdk/__init__.py", _extract_dunder_version)
    _collect_version(sources, root, "sdk/python/src/edcocr_sdk/client.py", _extract_sdk_version_py)
    _collect_version(sources, root, "sdk/typescript/package.json", _extract_package_json_version)
    _collect_version(sources, root, "sdk/typescript/src/client.ts", _extract_sdk_version_ts)
    _collect_version(sources, root, "helm/ocr-local/Chart.yaml", _extract_chart_app_version)

    mismatches: list[str] = []
    missing: list[str] = []
    for label, ver in sources.items():
        if ver is None:
            missing.append(label)
        elif ver != canonical:
            mismatches.append(f"{label}: {ver} (expected {canonical})")

    details: list[str] = [f"canonical version: {canonical}"]
    for label, ver in sorted(sources.items()):
        details.append(f"{label}: {ver if ver is not None else 'FILE_NOT_FOUND'}")

    if mismatches:
        return CheckResult(
            name="version_alignment",
            passed=False,
            message=f"Version mismatch in {len(mismatches)} source(s)",
            details=details + ["Mismatches:"] + mismatches,
        )

    if missing:
        return CheckResult(
            name="version_alignment",
            passed=False,
            message=f"{len(missing)} version source file(s) not found",
            details=details + ["Missing:"] + missing,
        )

    return CheckResult(
        name="version_alignment",
        passed=True,
        message=f"All {len(sources) + 1} version sources match: {canonical}",
        details=details,
    )


# Version extraction helpers ------------------------------------------------


def _extract_canonical_version(root: Path) -> str | None:
    """Return the canonical product version.

    ``version.py`` is retained as a backward-compatibility shim in newer trees,
    while the actual ``__version__`` lives under ``ocr_local/config``.
    """
    for rel_path in ("version.py", "ocr_local/config/version.py"):
        path = root / rel_path
        if not path.exists():
            continue
        try:
            return _extract_dunder_version(path)
        except ValueError:
            continue
    return None


def _collect_version(
    out: dict[str, str | None],
    root: Path,
    rel_path: str,
    extractor: object,
) -> None:
    full = root / rel_path
    if not full.exists():
        out[rel_path] = None
        return
    try:
        out[rel_path] = extractor(full)  # type: ignore[operator]
    except (ValueError, KeyError, json.JSONDecodeError):
        out[rel_path] = None


def _extract_dunder_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        raise ValueError(f"No __version__ in {path}")
    return m.group(1)


def _extract_pyproject_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise ValueError(f"No version in {path}")
    return m.group(1)


def _extract_sdk_version_py(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'^SDK_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        raise ValueError(f"No SDK_VERSION in {path}")
    return m.group(1)


def _extract_package_json_version(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    ver = data.get("version")
    if not ver:
        raise ValueError(f"No version in {path}")
    return ver


def _extract_sdk_version_ts(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"""^export\s+const\s+SDK_VERSION\s*=\s*['"]([^'"]+)['"]""", text, re.MULTILINE)
    if not m:
        raise ValueError(f"No SDK_VERSION in {path}")
    return m.group(1)


def _extract_chart_app_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'^appVersion:\s*["\']?([^"\'\n]+)["\']?', text, re.MULTILINE)
    if not m:
        raise ValueError(f"No appVersion in {path}")
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# Check B: Python SDK structure
# ---------------------------------------------------------------------------

# Required pyproject.toml fields
PYTHON_SDK_REQUIRED_FIELDS = ("name", "version", "license")
PYTHON_SDK_REQUIRES_PYTHON_KEY = "requires-python"


def check_python_sdk_structure(root: Path) -> CheckResult:
    """Verify Python SDK pyproject.toml has required metadata."""
    pyproject = root / "sdk" / "python" / "pyproject.toml"
    if not pyproject.exists():
        return CheckResult(
            name="python_sdk_structure",
            passed=False,
            message="sdk/python/pyproject.toml not found",
        )

    text = pyproject.read_text(encoding="utf-8")
    issues: list[str] = []

    for fld in PYTHON_SDK_REQUIRED_FIELDS:
        pattern = rf'^{re.escape(fld)}\s*='
        if not re.search(pattern, text, re.MULTILINE):
            issues.append(f"Missing required field: {fld}")

    if not re.search(rf'^{re.escape(PYTHON_SDK_REQUIRES_PYTHON_KEY)}\s*=', text, re.MULTILINE):
        issues.append(f"Missing required field: {PYTHON_SDK_REQUIRES_PYTHON_KEY}")

    # Check for py.typed marker
    py_typed = root / "sdk" / "python" / "src" / "edcocr_sdk" / "py.typed"
    if not py_typed.exists():
        issues.append("Missing py.typed marker for PEP 561")

    if issues:
        return CheckResult(
            name="python_sdk_structure",
            passed=False,
            message=f"{len(issues)} issue(s) in Python SDK structure",
            details=issues,
        )

    return CheckResult(
        name="python_sdk_structure",
        passed=True,
        message="Python SDK structure is valid",
        details=[f"All required fields present in {pyproject.relative_to(root)}"],
    )


# ---------------------------------------------------------------------------
# Check C: TypeScript SDK structure
# ---------------------------------------------------------------------------

TS_SDK_REQUIRED_FIELDS = ("name", "version", "main", "types")
TS_SDK_NODE_FLOOR = "^20.19.0 || >=22.12.0"


def check_typescript_sdk_structure(root: Path) -> CheckResult:
    """Verify TypeScript SDK package.json has required fields."""
    pkg_json = root / "sdk" / "typescript" / "package.json"
    if not pkg_json.exists():
        return CheckResult(
            name="typescript_sdk_structure",
            passed=False,
            message="sdk/typescript/package.json not found",
        )

    try:
        data = json.loads(pkg_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckResult(
            name="typescript_sdk_structure",
            passed=False,
            message=f"Invalid JSON in package.json: {exc}",
        )

    issues: list[str] = []
    for fld in TS_SDK_REQUIRED_FIELDS:
        if fld not in data or not data[fld]:
            issues.append(f"Missing or empty required field: {fld}")

    # Check engines.node constraint
    engines = data.get("engines", {})
    if "node" not in engines:
        issues.append(f"Missing engines.node constraint (should require {TS_SDK_NODE_FLOOR})")
    elif engines["node"] != TS_SDK_NODE_FLOOR:
        issues.append(
            f"engines.node must match supported SDK/toolchain floor {TS_SDK_NODE_FLOOR}; "
            f"found {engines['node']!r}"
        )

    dev_runtime = data.get("devEngines", {}).get("runtime", {})
    if dev_runtime.get("name") != "node":
        issues.append("Missing devEngines.runtime.name=node")
    if dev_runtime.get("version") != engines.get("node"):
        issues.append("devEngines.runtime.version must match engines.node")
    if dev_runtime.get("onFail") != "error":
        issues.append("devEngines.runtime.onFail must be error")

    node_types = data.get("devDependencies", {}).get("@types/node", "")
    if not node_types.startswith("^20."):
        issues.append("@types/node must stay on the Node 20 type surface for the declared support floor")

    if issues:
        return CheckResult(
            name="typescript_sdk_structure",
            passed=False,
            message=f"{len(issues)} issue(s) in TypeScript SDK structure",
            details=issues,
        )

    return CheckResult(
        name="typescript_sdk_structure",
        passed=True,
        message="TypeScript SDK structure is valid",
        details=[f"All required fields present in {pkg_json.relative_to(root)}"],
    )


# ---------------------------------------------------------------------------
# Check D: API stability contract
# ---------------------------------------------------------------------------


def check_api_stability(root: Path) -> CheckResult:
    """Verify api/versioning.py exists with endpoint registry."""
    versioning = root / "api" / "versioning.py"
    if not versioning.exists():
        return CheckResult(
            name="api_stability",
            passed=False,
            message="api/versioning.py not found",
        )

    text = versioning.read_text(encoding="utf-8")
    issues: list[str] = []

    if "API_SURFACE" not in text:
        issues.append("API_SURFACE registry not found in api/versioning.py")

    if "StabilityTier" not in text:
        issues.append("StabilityTier enum not found in api/versioning.py")

    if "check_backward_compatibility" not in text:
        issues.append("check_backward_compatibility function not found")

    # Count stable endpoints
    stable_count = len(re.findall(r"StabilityTier\.STABLE", text))

    if issues:
        return CheckResult(
            name="api_stability",
            passed=False,
            message=f"{len(issues)} issue(s) in API stability contract",
            details=issues,
        )

    return CheckResult(
        name="api_stability",
        passed=True,
        message="API stability contract is valid",
        details=[
            "api/versioning.py contains API_SURFACE, StabilityTier, check_backward_compatibility",
            f"Stable endpoints registered: {stable_count}",
        ],
    )


# ---------------------------------------------------------------------------
# Check E: Deprecation markers
# ---------------------------------------------------------------------------


def check_deprecation_markers(root: Path) -> CheckResult:
    """Verify deprecated APIs have runtime warnings in SDK code."""
    versioning = root / "api" / "versioning.py"
    if not versioning.exists():
        return CheckResult(
            name="deprecation_markers",
            passed=True,
            message="api/versioning.py not found; skipping deprecation check",
        )

    text = versioning.read_text(encoding="utf-8")
    deprecated_endpoints = re.findall(r'deprecated=True', text)

    if not deprecated_endpoints:
        return CheckResult(
            name="deprecation_markers",
            passed=True,
            message="No deprecated endpoints found; no warnings required",
            details=["All endpoints are active"],
        )

    # If there are deprecated endpoints, check SDK code for warnings
    issues: list[str] = []

    py_client = root / "sdk" / "python" / "src" / "edcocr_sdk" / "client.py"
    if py_client.exists():
        py_text = py_client.read_text(encoding="utf-8")
        has_py_warnings = "warnings.warn" in py_text or "import warnings" in py_text
        if not has_py_warnings:
            issues.append(
                "Python SDK client.py has no warnings.warn for deprecated endpoints"
            )

    ts_client = root / "sdk" / "typescript" / "src" / "client.ts"
    if ts_client.exists():
        ts_text = ts_client.read_text(encoding="utf-8")
        has_ts_deprecated = "@deprecated" in ts_text or "console.warn" in ts_text
        if not has_ts_deprecated:
            issues.append(
                "TypeScript SDK client.ts has no @deprecated or console.warn markers"
            )

    if issues:
        return CheckResult(
            name="deprecation_markers",
            passed=False,
            message=f"{len(deprecated_endpoints)} deprecated endpoint(s) but SDK warnings missing",
            details=issues,
        )

    return CheckResult(
        name="deprecation_markers",
        passed=True,
        message=f"{len(deprecated_endpoints)} deprecated endpoint(s) with SDK warnings present",
    )


# ---------------------------------------------------------------------------
# Check F: Policy document
# ---------------------------------------------------------------------------


def check_policy_document(root: Path) -> CheckResult:
    """Verify SDK-VERSIONING-POLICY.md exists and is non-empty."""
    policy = root / "docs" / "SDK-VERSIONING-POLICY.md"
    if not policy.exists():
        return CheckResult(
            name="policy_document",
            passed=False,
            message="docs/SDK-VERSIONING-POLICY.md not found",
        )

    content = policy.read_text(encoding="utf-8").strip()
    if not content:
        return CheckResult(
            name="policy_document",
            passed=False,
            message="docs/SDK-VERSIONING-POLICY.md is empty",
        )

    # Check for key sections
    expected_sections = [
        "Versioning Strategy",
        "Compatibility Matrix",
        "Deprecation",
        "Breaking Change",
        "Release Process",
        "Support Lifecycle",
    ]
    found: list[str] = []
    missing: list[str] = []
    for section in expected_sections:
        if section.lower() in content.lower():
            found.append(section)
        else:
            missing.append(section)

    if missing:
        return CheckResult(
            name="policy_document",
            passed=False,
            message=f"Policy document missing {len(missing)} expected section(s)",
            details=[f"Missing: {', '.join(missing)}", f"Found: {', '.join(found)}"],
        )

    return CheckResult(
        name="policy_document",
        passed=True,
        message=f"Policy document present with {len(found)} sections ({len(content)} bytes)",
        details=[f"Sections found: {', '.join(found)}"],
    )


# ---------------------------------------------------------------------------
# Check G: CI publish workflow
# ---------------------------------------------------------------------------


def check_ci_publish_workflow(root: Path) -> CheckResult:
    """Verify sdk-publish.yml exists with Python and TypeScript jobs."""
    workflow = root / ".github" / "workflows" / "sdk-publish.yml"
    if not workflow.exists():
        return CheckResult(
            name="ci_publish_workflow",
            passed=False,
            message=".github/workflows/sdk-publish.yml not found",
        )

    text = workflow.read_text(encoding="utf-8")
    issues: list[str] = []

    # Check for Python publish job
    if "sdk-python-v" not in text:
        issues.append("No sdk-python-v tag trigger found")
    if "pypi" not in text.lower():
        issues.append("No PyPI publishing step found")

    # Check for TypeScript publish job
    if "sdk-ts-v" not in text:
        issues.append("No sdk-ts-v tag trigger found")
    if "npm" not in text.lower():
        issues.append("No npm publishing step found")

    # Check for version consistency step
    if "check_version_consistency" not in text:
        issues.append("No version consistency check step found")

    if issues:
        return CheckResult(
            name="ci_publish_workflow",
            passed=False,
            message=f"{len(issues)} issue(s) in CI publish workflow",
            details=issues,
        )

    return CheckResult(
        name="ci_publish_workflow",
        passed=True,
        message="CI publish workflow has Python and TypeScript jobs",
        details=[
            "Tag triggers: sdk-python-v*, sdk-ts-v*",
            "Version consistency check present",
            "PyPI and npm publishing configured",
        ],
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_version_alignment,
    check_python_sdk_structure,
    check_typescript_sdk_structure,
    check_api_stability,
    check_deprecation_markers,
    check_policy_document,
    check_ci_publish_workflow,
]


def run_all_checks(root: Path) -> ValidationReport:
    """Execute all policy checks and return aggregated report."""
    report = ValidationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        project_root=str(root),
    )
    for check_fn in ALL_CHECKS:
        result = check_fn(root)
        report.add(result)
    return report


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def format_text(report: ValidationReport) -> str:
    """Format report as human-readable text."""
    lines: list[str] = []
    lines.append("SDK Policy Validation Report")
    lines.append("=" * 40)
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(f"Root:      {report.project_root}")
    lines.append("")

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"[{status}] {check.name}: {check.message}")
        for detail in check.details:
            lines.append(f"       {detail}")
        lines.append("")

    passed = sum(1 for c in report.checks if c.passed)
    total = len(report.checks)
    lines.append("-" * 40)
    lines.append(f"Result: {passed}/{total} checks passed")
    if report.passed:
        lines.append("Overall: PASS")
    else:
        lines.append("Overall: FAIL")

    return "\n".join(lines)


def format_json(report: ValidationReport) -> str:
    """Format report as JSON."""
    data = {
        "passed": report.passed,
        "timestamp": report.timestamp,
        "project_root": report.project_root,
        "checks": [asdict(c) for c in report.checks],
        "summary": {
            "total": len(report.checks),
            "passed": sum(1 for c in report.checks if c.passed),
            "failed": sum(1 for c in report.checks if not c.passed),
        },
    }
    return json.dumps(data, indent=2)


def format_markdown(report: ValidationReport) -> str:
    """Format report as markdown."""
    lines: list[str] = []
    lines.append("# SDK Policy Validation Report")
    lines.append("")
    lines.append(f"**Generated**: {report.timestamp}")
    lines.append(f"**Project root**: `{report.project_root}`")
    lines.append("")

    passed = sum(1 for c in report.checks if c.passed)
    total = len(report.checks)
    overall = "PASS" if report.passed else "FAIL"
    lines.append(f"**Result**: {passed}/{total} checks passed ({overall})")
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Status | Message |")
    lines.append("|-------|--------|---------|")
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"| {check.name} | {status} | {check.message} |")

    lines.append("")
    lines.append("## Details")
    lines.append("")
    for check in report.checks:
        if check.details:
            lines.append(f"### {check.name}")
            lines.append("")
            for detail in check.details:
                lines.append(f"- {detail}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate SDK implementations against versioning policy.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write markdown report to file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run SDK policy validation. Returns 0 on pass, 1 on failure, 2 on error."""
    parser = build_parser()
    args = parser.parse_args(argv)

    root = args.project_root if args.project_root else DEFAULT_PROJECT_ROOT

    if not root.exists():
        _safe_print(f"Error: project root not found: {root}", file=sys.stderr)
        return 2

    try:
        report = run_all_checks(root)
    except Exception as exc:
        _safe_print(f"Error running checks: {exc}", file=sys.stderr)
        return 2

    # Output results
    if args.json_output:
        _safe_print(format_json(report))
    else:
        _safe_print(format_text(report))

    # Write markdown report if requested
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(format_markdown(report), encoding="utf-8")
        _safe_print(f"\nReport written to {args.report}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
