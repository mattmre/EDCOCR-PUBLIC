#!/usr/bin/env python3
"""Upgrade readiness and config drift validation.

Validates that an environment is ready for a version upgrade by checking:
  A) Version compatibility (valid upgrade path)
  B) Config drift detection (env vars vs baseline snapshot)
  C) Deprecated settings (env vars that should be migrated)
  D) Required migrations (changelog breaking-change scan)
  E) Dependency compatibility (known-bad version combos)

Exit codes:
  0 = READY (no blockers)
  1 = NOT READY (blockers found)
  2 = WARNINGS ONLY (no blockers, but actionable warnings)

Usage:
    # Check upgrade readiness to a target version
    python scripts/upgrade_readiness.py --target-version 1.3.0

    # Save current env as baseline
    python scripts/upgrade_readiness.py --save-baseline

    # Compare current env against saved baseline
    python scripts/upgrade_readiness.py --compare-baseline

    # JSON output
    python scripts/upgrade_readiness.py --target-version 1.3.0 --json

    # Markdown report
    python scripts/upgrade_readiness.py --target-version 1.3.0 --report report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Locate project root (scripts/ is one level below)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default baseline location
DEFAULT_BASELINE_PATH = _PROJECT_ROOT / "docs" / "reports" / "config-baseline.json"

# Env var prefixes we consider "project-relevant"
MONITORED_PREFIXES: tuple[str, ...] = (
    "OCR_",
    "ENABLE_",
    "DEPLOYMENT_",
    "CELERY_",
    "S3_",
    "REDIS_",
    "DJANGO_",
    "STORAGE_",
    "NUM_",
    "CHUNK_",
    "MAX_",
    "API_",
    "WEBHOOK_",
    "JOB_",
    "METRICS_",
    "TENANT_",
    "RABBITMQ_",
    "NFS_",
    "GPU_",
    "VIDEO_",
)

MONITORED_EXACT_NAMES: frozenset[str] = frozenset({
    "DATABASE_URL",
    "DPI",
    "TEMP_FOLDER",
    "LOG_DIR",
    "PRODUCTION_READINESS_ACK",
    "CLASSIFICATION_MODE",
    "INFERENCE_BACKEND",
    "PYTHONIOENCODING",
})

# ---------------------------------------------------------------------------
# Deprecated settings registry
# ---------------------------------------------------------------------------
# Each entry maps an old env-var name to migration guidance.
#   replacement: the new env-var (or None if removed entirely)
#   since:       version where deprecation was introduced
#   removed:     version where the setting was fully removed (or None)
#   guidance:    human-readable migration note
DEPRECATED_SETTINGS: dict[str, dict[str, Any]] = {
    "ENABLE_HPI": {
        "replacement": "OCR_INFERENCE_BACKEND",
        "since": "1.0.0",
        "removed": None,
        "guidance": (
            "enable_hpi is PaddleOCR 3.x only. Use OCR_INFERENCE_BACKEND=onnx "
            "with PaddleOCR 2.9.1 instead."
        ),
    },
    "PADDLEX_MODEL": {
        "replacement": None,
        "since": "0.9.0",
        "removed": "1.0.0",
        "guidance": (
            "PaddleX is not used in this project. PaddleOCR 2.9.1 is the "
            "production engine."
        ),
    },
    "OCR_GPU_WORKERS": {
        "replacement": "NUM_WORKERS",
        "since": "0.5.0",
        "removed": None,
        "guidance": "Renamed to NUM_WORKERS for consistency with pipeline constants.",
    },
    "OCR_EXTRACTORS": {
        "replacement": "NUM_EXTRACTORS",
        "since": "0.5.0",
        "removed": None,
        "guidance": "Renamed to NUM_EXTRACTORS for consistency with pipeline constants.",
    },
    "OCR_COMPRESSORS": {
        "replacement": "NUM_COMPRESSORS",
        "since": "0.5.0",
        "removed": None,
        "guidance": "Renamed to NUM_COMPRESSORS for consistency with pipeline constants.",
    },
    "USE_GPU": {
        "replacement": "OCR_INFERENCE_BACKEND",
        "since": "0.7.0",
        "removed": None,
        "guidance": (
            "GPU detection is now automatic. Use OCR_INFERENCE_BACKEND "
            "to select paddle/onnx/openvino."
        ),
    },
    "ENABLE_PADDLEX": {
        "replacement": None,
        "since": "0.9.0",
        "removed": "1.0.0",
        "guidance": "PaddleX is not supported. Remove this setting.",
    },
    "OCR_CONSENSUS_MODE": {
        "replacement": None,
        "since": "1.0.0",
        "removed": None,
        "guidance": (
            "Consensus mode was a research prototype (item15). "
            "Use OCR_ENGINE_SELECTION for engine routing instead."
        ),
    },
    "CPU_ONLY_BUILD": {
        "replacement": None,
        "since": "1.1.0",
        "removed": None,
        "guidance": (
            "CPU_ONLY_BUILD is a build-time flag only. For runtime CPU mode, "
            "use OCR_TASK_ROUTING=cpu and OCR_INFERENCE_BACKEND=onnx."
        ),
    },
}

# ---------------------------------------------------------------------------
# Known dependency incompatibilities
# ---------------------------------------------------------------------------
# Each entry: (package_a, version_predicate_a, package_b, version_predicate_b, explanation)
# version_predicate uses simple ops: ">=X.Y", "<X.Y", "==X.Y"

KNOWN_INCOMPATIBILITIES: list[dict[str, str]] = [
    {
        "package_a": "numpy",
        "condition_a": ">=2.0.0",
        "package_b": "paddlepaddle",
        "condition_b": "<3.0.0",
        "severity": "blocker",
        "explanation": (
            "numpy 2.x breaks PaddlePaddle ABI. Pin numpy<2.0.0 until "
            "PaddlePaddle 3.x is adopted."
        ),
    },
    {
        "package_a": "opencv-python-headless",
        "condition_a": ">=4.12.0",
        "package_b": "numpy",
        "condition_b": "<2.0.0",
        "severity": "blocker",
        "explanation": (
            "opencv-python-headless 4.12+ requires numpy>=2 on Python 3.9+. "
            "Pin opencv-python-headless<4.12 with numpy<2."
        ),
    },
    {
        "package_a": "paddlepaddle",
        "condition_a": ">=3.0.0",
        "package_b": "paddleocr",
        "condition_b": "<3.0.0",
        "severity": "blocker",
        "explanation": (
            "PaddlePaddle 3.x requires PaddleOCR 3.x. Do not upgrade "
            "PaddlePaddle without upgrading PaddleOCR."
        ),
    },
    {
        "package_a": "django",
        "condition_a": ">=6.0",
        "package_b": None,
        "condition_b": None,
        "severity": "warning",
        "explanation": (
            "Django 6.x may require Python 3.12+ and has potential "
            "breaking changes. Review release notes before upgrading."
        ),
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result from a single readiness check."""

    category: str
    status: str  # "pass", "fail", "warn", "skip"
    summary: str
    details: list[str] = field(default_factory=list)


@dataclass
class ReadinessReport:
    """Aggregate readiness report."""

    version_current: str
    version_target: str | None
    timestamp: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def has_blockers(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == "warn" for c in self.checks)

    @property
    def exit_code(self) -> int:
        if self.has_blockers:
            return 1
        if self.has_warnings:
            return 2
        return 0


# ---------------------------------------------------------------------------
# Version utilities
# ---------------------------------------------------------------------------

def read_current_version(project_root: Path | None = None) -> str:
    """Read __version__ from version.py via regex."""
    root = project_root or _PROJECT_ROOT
    version_file = root / "version.py"
    if not version_file.exists():
        return "0.0.0"
    text = version_file.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if match:
        return match.group(1)
    return "0.0.0"


def parse_semver(version_str: str) -> tuple[int, int, int]:
    """Parse a semver string into (major, minor, patch).

    Tolerates pre-release suffixes like ``1.0.0-rc1`` by stripping them.
    """
    clean = version_str.split("-")[0].split("+")[0]
    parts = clean.split(".")
    if len(parts) < 3:
        parts.extend(["0"] * (3 - len(parts)))
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def is_valid_upgrade_path(
    current: str,
    target: str,
) -> tuple[bool, str]:
    """Check if upgrading from *current* to *target* is a valid path.

    Rules:
    - Target must be strictly greater than current.
    - Minor/patch within same major: always valid.
    - Major bump: valid but requires migration review.
    - Downgrade: never valid.

    Returns (is_valid, reason_string).
    """
    cur = parse_semver(current)
    tgt = parse_semver(target)

    if tgt == cur:
        return False, f"Target {target} is the same as current {current}"

    if tgt < cur:
        return False, f"Downgrade from {current} to {target} is not supported"

    if tgt[0] > cur[0]:
        return True, (
            f"Major version bump {current} -> {target}. "
            "Migration guide review required."
        )

    # Same major, higher minor or patch
    return True, f"Valid upgrade path: {current} -> {target}"


# ---------------------------------------------------------------------------
# A) Version compatibility check
# ---------------------------------------------------------------------------

def check_version_compatibility(
    current: str,
    target: str | None,
) -> CheckResult:
    """Check whether the upgrade path is valid."""
    if target is None:
        return CheckResult(
            category="Version Compatibility",
            status="skip",
            summary="No target version specified; skipping version check.",
        )

    valid, reason = is_valid_upgrade_path(current, target)
    cur = parse_semver(current)
    tgt = parse_semver(target)

    if not valid:
        return CheckResult(
            category="Version Compatibility",
            status="fail",
            summary=reason,
        )

    # Major bump => warning (requires migration review)
    if tgt[0] > cur[0]:
        return CheckResult(
            category="Version Compatibility",
            status="warn",
            summary=reason,
            details=[
                "Check CHANGELOG.md for breaking changes.",
                "Review docs/operations/production-cutover-runbook.md.",
            ],
        )

    return CheckResult(
        category="Version Compatibility",
        status="pass",
        summary=reason,
    )


# ---------------------------------------------------------------------------
# B) Config drift detection
# ---------------------------------------------------------------------------

def _is_monitored(name: str) -> bool:
    """Return True if *name* matches monitored prefixes or exact names."""
    if name in MONITORED_EXACT_NAMES:
        return True
    for prefix in MONITORED_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def capture_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Capture monitored env vars from the given or real environment."""
    source = env if env is not None else dict(os.environ)
    return {k: v for k, v in sorted(source.items()) if _is_monitored(k)}


def load_env_file(path: str | Path) -> dict[str, str]:
    """Load a .env file and return parsed key=value pairs.

    Ignores comments (#) and blank lines. Strips surrounding quotes.
    """
    result: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return result
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def save_baseline(
    env: dict[str, str],
    path: str | Path,
) -> None:
    """Save env snapshot as a JSON baseline file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variables": env,
    }
    p.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> dict[str, str]:
    """Load a previously saved baseline snapshot."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("variables", data)


def detect_drift(
    baseline: dict[str, str],
    current: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    """Compare baseline against current env and categorise differences."""
    added: list[dict[str, str]] = []
    removed: list[dict[str, str]] = []
    changed: list[dict[str, str]] = []

    for key in sorted(baseline):
        if key not in current:
            removed.append({"key": key, "baseline_value": baseline[key]})
        elif current[key] != baseline[key]:
            changed.append({
                "key": key,
                "baseline_value": baseline[key],
                "current_value": current[key],
            })

    for key in sorted(current):
        if key not in baseline:
            added.append({"key": key, "current_value": current[key]})

    return {"added": added, "removed": removed, "changed": changed}


def check_config_drift(
    baseline_path: str | Path | None,
    current_env: dict[str, str],
) -> CheckResult:
    """Compare current env against a saved baseline."""
    if baseline_path is None:
        return CheckResult(
            category="Config Drift",
            status="skip",
            summary="No baseline path provided; skipping drift check.",
        )

    p = Path(baseline_path)
    if not p.exists():
        return CheckResult(
            category="Config Drift",
            status="skip",
            summary=f"Baseline file not found: {p}",
        )

    try:
        baseline = load_baseline(p)
    except (json.JSONDecodeError, OSError) as exc:
        return CheckResult(
            category="Config Drift",
            status="fail",
            summary=f"Failed to load baseline: {exc}",
        )

    drift = detect_drift(baseline, current_env)
    total = sum(len(drift[c]) for c in ("added", "removed", "changed"))

    if total == 0:
        return CheckResult(
            category="Config Drift",
            status="pass",
            summary="No configuration drift detected.",
        )

    details: list[str] = []
    for item in drift["added"]:
        details.append(f"ADDED: {item['key']} = {item['current_value']}")
    for item in drift["removed"]:
        details.append(f"REMOVED: {item['key']} (was: {item['baseline_value']})")
    for item in drift["changed"]:
        details.append(
            f"CHANGED: {item['key']}: "
            f"{item['baseline_value']} -> {item['current_value']}"
        )

    return CheckResult(
        category="Config Drift",
        status="warn",
        summary=(
            f"Config drift detected: {len(drift['added'])} added, "
            f"{len(drift['removed'])} removed, "
            f"{len(drift['changed'])} changed."
        ),
        details=details,
    )


# ---------------------------------------------------------------------------
# C) Deprecated settings detection
# ---------------------------------------------------------------------------

def check_deprecated_settings(
    current_env: dict[str, str],
    deprecated: dict[str, dict[str, Any]] | None = None,
) -> CheckResult:
    """Scan env for deprecated settings."""
    registry = deprecated if deprecated is not None else DEPRECATED_SETTINGS
    found: list[str] = []

    for var_name, info in sorted(registry.items()):
        if var_name in current_env:
            replacement = info.get("replacement")
            since = info.get("since", "unknown")
            removed = info.get("removed")
            guidance = info.get("guidance", "")

            parts = [f"{var_name}: deprecated since {since}"]
            if removed:
                parts.append(f"(removed in {removed})")
            if replacement:
                parts.append(f"-> use {replacement}")
            if guidance:
                parts.append(f"  {guidance}")
            found.append(" ".join(parts))

    if not found:
        return CheckResult(
            category="Deprecated Settings",
            status="pass",
            summary="No deprecated settings detected in environment.",
        )

    # If any deprecated setting was fully removed, that is a blocker
    has_removed = any(
        var_name in current_env and registry[var_name].get("removed") is not None
        for var_name in registry
    )

    return CheckResult(
        category="Deprecated Settings",
        status="fail" if has_removed else "warn",
        summary=f"Found {len(found)} deprecated setting(s) in environment.",
        details=found,
    )


# ---------------------------------------------------------------------------
# D) Required migrations check
# ---------------------------------------------------------------------------

def parse_changelog_versions(changelog_path: Path) -> list[tuple[str, str]]:
    """Parse CHANGELOG.md and return list of (version, section_text) tuples."""
    if not changelog_path.exists():
        return []

    text = changelog_path.read_text(encoding="utf-8")
    # Match lines like "## [1.2.0] - 2026-03-26"
    version_pattern = re.compile(r"^## \[([^\]]+)\]", re.MULTILINE)
    matches = list(version_pattern.finditer(text))

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        version = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end]
        sections.append((version, section_text))

    return sections


def check_required_migrations(
    current_version: str,
    target_version: str | None,
    project_root: Path | None = None,
) -> CheckResult:
    """Check changelog for breaking changes between current and target version."""
    if target_version is None:
        return CheckResult(
            category="Required Migrations",
            status="skip",
            summary="No target version specified; skipping migration check.",
        )

    root = project_root or _PROJECT_ROOT
    changelog_path = root / "CHANGELOG.md"

    if not changelog_path.exists():
        return CheckResult(
            category="Required Migrations",
            status="skip",
            summary="CHANGELOG.md not found; cannot check for breaking changes.",
        )

    current_sv = parse_semver(current_version)
    target_sv = parse_semver(target_version)

    sections = parse_changelog_versions(changelog_path)
    relevant_sections: list[tuple[str, str]] = []

    for version, section_text in sections:
        try:
            sv = parse_semver(version)
        except (ValueError, IndexError):
            continue
        # Include sections strictly between current (exclusive) and target (inclusive)
        if current_sv < sv <= target_sv:
            relevant_sections.append((version, section_text))

    if not relevant_sections:
        return CheckResult(
            category="Required Migrations",
            status="pass",
            summary="No changelog entries found between current and target version.",
        )

    # Scan for breaking/migration keywords
    breaking_pattern = re.compile(
        r"(breaking|migration|BREAKING|migrate|incompatible)",
        re.IGNORECASE,
    )

    migration_notes: list[str] = []
    for version, section_text in relevant_sections:
        hits = breaking_pattern.findall(section_text)
        if hits:
            # Extract the relevant lines
            for line in section_text.splitlines():
                if breaking_pattern.search(line):
                    migration_notes.append(f"[{version}] {line.strip()}")

    # Also check for migration scripts
    scripts_dir = root / "scripts"
    migration_scripts: list[str] = []
    if scripts_dir.exists():
        for p in sorted(scripts_dir.iterdir()):
            if p.is_file() and "migrat" in p.name.lower():
                migration_scripts.append(p.name)

    details: list[str] = []
    if migration_notes:
        details.append("Breaking/migration references in changelog:")
        details.extend(f"  {note}" for note in migration_notes)
    if migration_scripts:
        details.append("Migration scripts available:")
        details.extend(f"  scripts/{s}" for s in migration_scripts)

    if migration_notes:
        return CheckResult(
            category="Required Migrations",
            status="warn",
            summary=(
                f"Found {len(migration_notes)} breaking/migration reference(s) "
                f"in changelog between {current_version} and {target_version}."
            ),
            details=details,
        )

    return CheckResult(
        category="Required Migrations",
        status="pass",
        summary=(
            f"No breaking changes found in changelog between "
            f"{current_version} and {target_version}."
        ),
        details=details if migration_scripts else [],
    )


# ---------------------------------------------------------------------------
# E) Dependency compatibility check
# ---------------------------------------------------------------------------

def parse_requirements(requirements_path: Path) -> dict[str, str]:
    """Parse requirements.txt into {package_name: version_spec}.

    Returns the raw version specifier string (e.g. "==2.6.2", ">=0.42.0,<1.0.0").
    Commented-out lines are ignored.
    """
    result: dict[str, str] = {}
    if not requirements_path.exists():
        return result

    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle extras like uvicorn[standard]
        # Regex: name[extras]version_spec
        match = re.match(
            r"([a-zA-Z0-9_-]+)(?:\[[^\]]+\])?\s*(.*)",
            line,
        )
        if match:
            pkg = match.group(1).lower().replace("-", "_")
            spec = match.group(2).strip()
            if spec:
                result[pkg] = spec

    return result


def _extract_pinned_version(spec: str) -> str | None:
    """Extract the exact version from a pin like '==2.6.2'.

    For range specs like '>=0.42.0,<1.0.0', returns the lower bound.
    Returns None if no version can be determined.
    """
    # Exact pin
    m = re.match(r"==\s*([0-9][0-9a-zA-Z.*]+)", spec)
    if m:
        return m.group(1)
    # Range pin — return lower bound
    m = re.match(r">=\s*([0-9][0-9a-zA-Z.*]+)", spec)
    if m:
        return m.group(1)
    return None


def _version_matches_condition(version_str: str, condition: str) -> bool:
    """Check if a version string matches a simple condition.

    Supports: >=X.Y.Z, <X.Y.Z, ==X.Y.Z, >X.Y.Z, <=X.Y.Z
    """
    m = re.match(r"(>=|<=|==|>|<)\s*([0-9][0-9.]*)", condition)
    if not m:
        return False

    op = m.group(1)
    cond_ver = parse_semver(m.group(2))

    try:
        actual = parse_semver(version_str)
    except (ValueError, IndexError):
        return False

    if op == ">=":
        return actual >= cond_ver
    if op == "<=":
        return actual <= cond_ver
    if op == "==":
        return actual == cond_ver
    if op == ">":
        return actual > cond_ver
    if op == "<":
        return actual < cond_ver
    return False


def check_dependency_compatibility(
    project_root: Path | None = None,
    incompatibilities: list[dict[str, str]] | None = None,
) -> CheckResult:
    """Check for known-bad dependency version combinations."""
    root = project_root or _PROJECT_ROOT
    req_path = root / "requirements.txt"
    packages = parse_requirements(req_path)

    if not packages:
        return CheckResult(
            category="Dependency Compatibility",
            status="skip",
            summary="requirements.txt not found or empty.",
        )

    rules = incompatibilities if incompatibilities is not None else KNOWN_INCOMPATIBILITIES
    blockers: list[str] = []
    warnings: list[str] = []

    for rule in rules:
        pkg_a = rule["package_a"].lower().replace("-", "_")
        cond_a = rule["condition_a"]

        if pkg_a not in packages:
            continue

        ver_a = _extract_pinned_version(packages[pkg_a])
        if ver_a is None:
            continue

        if not _version_matches_condition(ver_a, cond_a):
            continue

        # Package A matches its condition — now check B if present
        pkg_b_raw = rule.get("package_b")
        if pkg_b_raw is not None:
            pkg_b = pkg_b_raw.lower().replace("-", "_")
            cond_b = rule.get("condition_b", "")

            if pkg_b not in packages:
                continue

            ver_b = _extract_pinned_version(packages[pkg_b])
            if ver_b is None:
                continue

            if not _version_matches_condition(ver_b, cond_b):
                continue

        # Both conditions match — this is a hit
        severity = rule.get("severity", "warning")
        explanation = rule["explanation"]
        msg = f"[{severity.upper()}] {explanation}"

        if severity == "blocker":
            blockers.append(msg)
        else:
            warnings.append(msg)

    details = blockers + warnings

    if blockers:
        return CheckResult(
            category="Dependency Compatibility",
            status="fail",
            summary=f"Found {len(blockers)} dependency blocker(s).",
            details=details,
        )

    if warnings:
        return CheckResult(
            category="Dependency Compatibility",
            status="warn",
            summary=f"Found {len(warnings)} dependency warning(s).",
            details=details,
        )

    return CheckResult(
        category="Dependency Compatibility",
        status="pass",
        summary="No known dependency incompatibilities detected.",
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "pass": "PASS",
    "fail": "FAIL",
    "warn": "WARN",
    "skip": "SKIP",
}


def format_text_report(report: ReadinessReport) -> str:
    """Format the report as a human-readable text table."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("UPGRADE READINESS REPORT")
    lines.append("=" * 70)
    lines.append(f"Current version : {report.version_current}")
    if report.version_target:
        lines.append(f"Target version  : {report.version_target}")
    lines.append(f"Timestamp       : {report.timestamp}")
    lines.append("-" * 70)
    lines.append("")

    for check in report.checks:
        icon = _STATUS_ICONS.get(check.status, check.status.upper())
        lines.append(f"[{icon}] {check.category}")
        lines.append(f"       {check.summary}")
        for detail in check.details:
            lines.append(f"       - {detail}")
        lines.append("")

    lines.append("-" * 70)
    if report.has_blockers:
        lines.append("Result: NOT READY (blockers found)")
    elif report.has_warnings:
        lines.append("Result: WARNINGS (review recommended)")
    else:
        lines.append("Result: READY")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_json_report(report: ReadinessReport) -> str:
    """Format the report as JSON."""
    data = {
        "version_current": report.version_current,
        "version_target": report.version_target,
        "timestamp": report.timestamp,
        "ready": not report.has_blockers,
        "exit_code": report.exit_code,
        "checks": [
            {
                "category": c.category,
                "status": c.status,
                "summary": c.summary,
                "details": c.details,
            }
            for c in report.checks
        ],
    }
    return json.dumps(data, indent=2, sort_keys=True)


def format_markdown_report(report: ReadinessReport) -> str:
    """Format the report as a Markdown document."""
    lines: list[str] = []
    lines.append("# Upgrade Readiness Report")
    lines.append("")
    lines.append(f"- **Current version**: {report.version_current}")
    if report.version_target:
        lines.append(f"- **Target version**: {report.version_target}")
    lines.append(f"- **Generated**: {report.timestamp}")
    lines.append("")

    status_emoji = {
        "pass": "PASS",
        "fail": "FAIL",
        "warn": "WARN",
        "skip": "SKIP",
    }

    lines.append("## Checks")
    lines.append("")
    lines.append("| Category | Status | Summary |")
    lines.append("|----------|--------|---------|")
    for check in report.checks:
        s = status_emoji.get(check.status, check.status.upper())
        lines.append(f"| {check.category} | {s} | {check.summary} |")

    lines.append("")

    # Details for non-pass checks
    for check in report.checks:
        if check.details:
            lines.append(f"### {check.category}")
            lines.append("")
            for detail in check.details:
                lines.append(f"- {detail}")
            lines.append("")

    lines.append("## Result")
    lines.append("")
    if report.has_blockers:
        lines.append("**NOT READY** -- blockers found that must be resolved.")
    elif report.has_warnings:
        lines.append("**WARNINGS** -- review recommended before proceeding.")
    else:
        lines.append("**READY** -- all checks passed.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_readiness_checks(
    target_version: str | None = None,
    env_file: str | None = None,
    baseline_path: str | None = None,
    compare_baseline: bool = False,
    project_root: Path | None = None,
    env_override: dict[str, str] | None = None,
) -> ReadinessReport:
    """Run all readiness checks and return the aggregate report."""
    root = project_root or _PROJECT_ROOT
    current_version = read_current_version(root)

    # Build the effective environment
    effective_env: dict[str, str]
    if env_override is not None:
        effective_env = capture_env(env_override)
    elif env_file:
        file_env = load_env_file(env_file)
        # Merge file env on top of real env
        merged = dict(os.environ)
        merged.update(file_env)
        effective_env = capture_env(merged)
    else:
        effective_env = capture_env()

    report = ReadinessReport(
        version_current=current_version,
        version_target=target_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # A) Version compatibility
    report.checks.append(
        check_version_compatibility(current_version, target_version)
    )

    # B) Config drift
    bl_path = baseline_path if compare_baseline else None
    report.checks.append(
        check_config_drift(bl_path, effective_env)
    )

    # C) Deprecated settings
    report.checks.append(
        check_deprecated_settings(effective_env)
    )

    # D) Required migrations
    report.checks.append(
        check_required_migrations(current_version, target_version, root)
    )

    # E) Dependency compatibility
    report.checks.append(
        check_dependency_compatibility(root)
    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Upgrade readiness and config drift validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0 = READY (no blockers)\n"
            "  1 = NOT READY (blockers found)\n"
            "  2 = WARNINGS ONLY\n"
        ),
    )
    parser.add_argument(
        "--target-version",
        type=str,
        default=None,
        help="Target version to upgrade to (e.g. 1.3.0)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to a .env file to load before checking",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current env vars as a baseline snapshot and exit",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare current env against saved baseline",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help=f"Path to baseline JSON file (default: {DEFAULT_BASELINE_PATH})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="PATH",
        help="Write a Markdown report to the given file path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_path = args.baseline_path or str(DEFAULT_BASELINE_PATH)

    # Handle --save-baseline mode
    if args.save_baseline:
        env = capture_env()
        save_baseline(env, baseline_path)
        print(f"Baseline saved to {baseline_path} ({len(env)} variables captured)")
        return 0

    # Run all checks
    report = run_readiness_checks(
        target_version=args.target_version,
        env_file=args.env_file,
        baseline_path=baseline_path,
        compare_baseline=args.compare_baseline,
    )

    # Output
    if args.json_output:
        print(format_json_report(report))
    else:
        print(format_text_report(report))

    # Write markdown report if requested
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            format_markdown_report(report) + "\n",
            encoding="utf-8",
        )
        print(f"\nMarkdown report written to {args.report}")

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
