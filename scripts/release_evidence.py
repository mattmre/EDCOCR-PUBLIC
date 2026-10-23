#!/usr/bin/env python3
"""Release evidence bundle generator.

Orchestrates all existing validation tools into a single evidence capture
pipeline with a timestamped output bundle.  This is a NEW wrapper that
does NOT modify any existing scripts -- it invokes them as subprocesses,
normalizes results into a common schema, and writes an evidence bundle
directory containing raw artifacts, a manifest.json, and a summary.md.

Tier 1 (static checks, always run):
  A) Version Consistency  -- verify_release_state.py --json
  B) RC Validation        -- rc_validation.py --json [--skip-tests]
  C) Release Gate         -- run_phase8_release_gate.py --dry-run --json-report

Tier 2 (operational checks, run with --full):
  D) Cutover Preflight    -- cutover_preflight.py --json --skip-connectivity

Exit codes:
  0 = all PASS
  1 = any FAIL
  2 = PARTIAL (warnings only, no failures)

Usage:
    python scripts/release_evidence.py
    python scripts/release_evidence.py --full
    python scripts/release_evidence.py --skip-tests --json
    python scripts/release_evidence.py --pull-request 536 --evidence-dir docs/reports/my-run
    python scripts/release_evidence.py --full --production-gates /secure/evidence/production-gates.json

Local bundles are explicitly marked as dry-run evidence. A production-scoped
bundle additionally requires an operator-authored GO manifest covering every
required live gate. The resulting manifest must be detached-signed before the
release checklist will accept it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.release_evidence_contract import (
        BUNDLE_SCHEMA_VERSION,
        REQUIRED_PRODUCTION_GATES,
        is_timestamp,
    )
except ImportError:  # Direct execution: scripts/ is sys.path[0].
    from release_evidence_contract import (
        BUNDLE_SCHEMA_VERSION,
        REQUIRED_PRODUCTION_GATES,
        is_timestamp,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Subprocess timeout defaults (seconds)
STATIC_CHECK_TIMEOUT = 30
TEST_TIMEOUT = 1800
OPERATIONAL_CHECK_TIMEOUT = 60

# Status constants
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"
ERROR = "ERROR"
EVIDENCE_CREDENTIAL_SCAN_ARTIFACT = "evidence-secret-scan.json"


# ---------------------------------------------------------------------------
# Evidence data model
# ---------------------------------------------------------------------------


@dataclass
class EvidenceCheck:
    """Result of a single evidence check."""

    name: str
    category: str  # "static" or "operational"
    status: str  # PASS, FAIL, WARN, SKIP, ERROR
    summary: str
    details: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    artifact_path: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = STATIC_CHECK_TIMEOUT,
) -> tuple[int, str, str, float]:
    """Run a subprocess and return (returncode, stdout, stderr, duration).

    If the subprocess fails to start or times out, returns a non-zero code
    with appropriate stderr text.
    """
    start = time.monotonic()
    effective_cwd = str(cwd) if cwd else str(PROJECT_ROOT)
    try:
        result = subprocess.run(
            cmd,
            cwd=effective_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - start
        return result.returncode, result.stdout, result.stderr, duration
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        return -1, "", f"Command timed out after {timeout}s", duration
    except FileNotFoundError as exc:
        duration = time.monotonic() - start
        return -2, "", f"Command not found: {exc}", duration
    except OSError as exc:
        duration = time.monotonic() - start
        return -3, "", f"OS error: {exc}", duration


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_version() -> str:
    """Read version from the canonical package source."""
    version_path = PROJECT_ROOT / "ocr_local" / "config" / "version.py"
    if not version_path.exists():
        return "unknown"
    text = version_path.read_text(encoding="utf-8")
    import re

    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return match.group(1) if match else "unknown"


def _read_git_commit() -> str:
    """Return the exact source commit used to generate the bundle."""
    code, stdout, _stderr, _duration = _run_subprocess(
        ["git", "rev-parse", "HEAD"], timeout=STATIC_CHECK_TIMEOUT
    )
    return stdout.strip() if code == 0 else "unknown"


def _write_artifact(bundle_dir: Path, name: str, content: str) -> Path:
    """Write a sub-artifact into the evidence bundle directory."""
    path = bundle_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def _collect_production_evidence(
    source_manifest_path: Path,
    bundle_dir: Path,
    *,
    version: str,
    source_commit: str,
) -> tuple[dict[str, dict], list[dict], list[str]]:
    """Validate operator gate metadata and copy its evidence into the bundle."""
    errors: list[str] = []
    try:
        payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, [], [f"cannot read production gate manifest: {exc}"]
    if not isinstance(payload, dict):
        return {}, [], ["production gate manifest root must be an object"]
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append(
            f"production gate manifest schema_version must be {BUNDLE_SCHEMA_VERSION}"
        )
    if payload.get("release_version") != version:
        errors.append("production gate manifest release_version does not match source")
    if payload.get("source_commit") != source_commit:
        errors.append("production gate manifest source_commit does not match HEAD")
    if payload.get("decision") != "GO":
        errors.append("production gate manifest decision must be GO")

    raw_gates = payload.get("gates")
    if not isinstance(raw_gates, dict):
        return {}, [], errors + ["production gate manifest gates must be an object"]

    source_root = source_manifest_path.resolve().parent
    production_dir = bundle_dir / "production"
    normalized_gates: dict[str, dict] = {}
    copied_artifacts: list[dict] = []

    for gate_name in REQUIRED_PRODUCTION_GATES:
        raw_gate = raw_gates.get(gate_name)
        if not isinstance(raw_gate, dict):
            errors.append(f"production gate missing: {gate_name}")
            continue
        if raw_gate.get("status") != PASS:
            errors.append(f"production gate {gate_name} is not PASS")
        approved_by = str(raw_gate.get("approved_by", "")).strip()
        completed_at = str(raw_gate.get("completed_at", "")).strip()
        if not approved_by:
            errors.append(f"production gate {gate_name} has no approved_by")
        if not is_timestamp(completed_at):
            errors.append(f"production gate {gate_name} has invalid completed_at")
        raw_artifacts = raw_gate.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            errors.append(f"production gate {gate_name} has no artifacts")
            continue

        normalized_paths: list[str] = []
        for index, raw_artifact in enumerate(raw_artifacts, start=1):
            if not isinstance(raw_artifact, str) or not raw_artifact.strip():
                errors.append(
                    f"production gate {gate_name} has an invalid artifact path"
                )
                continue
            relative_source = Path(raw_artifact)
            if relative_source.is_absolute() or ".." in relative_source.parts:
                errors.append(
                    f"production gate {gate_name} artifact must be relative: {raw_artifact!r}"
                )
                continue
            unresolved_source = source_root / relative_source
            candidate = source_root
            contains_symlink = False
            for part in relative_source.parts:
                candidate = candidate / part
                if candidate.is_symlink():
                    contains_symlink = True
                    break
            if contains_symlink:
                errors.append(
                    f"production gate {gate_name} artifact contains symlink: "
                    f"{raw_artifact!r}"
                )
                continue
            source = unresolved_source.resolve()
            try:
                source.relative_to(source_root)
            except ValueError:
                errors.append(
                    f"production gate {gate_name} artifact escapes evidence root: "
                    f"{raw_artifact!r}"
                )
                continue
            if not source.is_file() or source.is_symlink():
                errors.append(
                    f"production gate {gate_name} artifact missing or unsafe: "
                    f"{raw_artifact!r}"
                )
                continue

            destination_dir = production_dir / gate_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{index:02d}-{source.name}"
            shutil.copy2(source, destination)
            bundle_path = str(destination.relative_to(bundle_dir))
            normalized_paths.append(bundle_path)
            copied_artifacts.append(
                {
                    "name": bundle_path,
                    "path": bundle_path,
                    "sha256": _sha256_file(destination),
                }
            )

        normalized_gates[gate_name] = {
            "status": raw_gate.get("status"),
            "approved_by": approved_by,
            "completed_at": completed_at,
            "artifacts": normalized_paths,
        }

    return normalized_gates, copied_artifacts, errors


# ---------------------------------------------------------------------------
# Tier 1: Static checks
# ---------------------------------------------------------------------------


def check_version_consistency(bundle_dir: Path) -> EvidenceCheck:
    """A) Version Consistency -- invoke verify_release_state.py --json."""
    script = SCRIPT_DIR / "verify_release_state.py"
    if not script.exists():
        return EvidenceCheck(
            name="version_consistency",
            category="static",
            status=ERROR,
            summary="verify_release_state.py not found",
            details=[f"Expected at: {script}"],
        )

    cmd = [sys.executable, str(script), "--json"]
    code, stdout, stderr, duration = _run_subprocess(cmd, timeout=STATIC_CHECK_TIMEOUT)

    artifact_path: str | None = None
    if stdout.strip():
        art = _write_artifact(bundle_dir, "version_consistency.json", stdout)
        artifact_path = str(art.relative_to(bundle_dir))

    if code < 0:
        return EvidenceCheck(
            name="version_consistency",
            category="static",
            status=ERROR,
            summary=f"Subprocess error: {stderr.strip()}",
            details=[stderr.strip()],
            duration_seconds=duration,
            artifact_path=artifact_path,
        )

    # Parse JSON output
    details: list[str] = []
    try:
        data = json.loads(stdout)
        vc = data.get("version_consistency", data)
        passed = vc.get("passed", False)
        canonical = vc.get("canonical", "unknown")
        mismatches = vc.get("mismatches", {})
        total = vc.get("total_sources", 0)

        details.append(f"Canonical version: {canonical}")
        details.append(f"Total sources checked: {total}")
        if mismatches:
            for label, ver in sorted(mismatches.items()):
                details.append(f"MISMATCH: {label} = {ver}")

        status = PASS if passed else FAIL
        summary = (
            f"All {total} version sources consistent ({canonical})"
            if passed
            else f"{len(mismatches)} version mismatch(es) found"
        )
    except (json.JSONDecodeError, KeyError) as exc:
        status = ERROR
        summary = f"Failed to parse output: {exc}"
        details.append(stdout[:500] if stdout else "(no output)")

    return EvidenceCheck(
        name="version_consistency",
        category="static",
        status=status,
        summary=summary,
        details=details,
        duration_seconds=duration,
        artifact_path=artifact_path,
    )


def check_rc_validation(bundle_dir: Path, *, skip_tests: bool = False) -> EvidenceCheck:
    """B) RC Validation -- invoke rc_validation.py --json [--skip-tests]."""
    script = SCRIPT_DIR / "rc_validation.py"
    if not script.exists():
        return EvidenceCheck(
            name="rc_validation",
            category="static",
            status=ERROR,
            summary="rc_validation.py not found",
            details=[f"Expected at: {script}"],
        )

    cmd = [sys.executable, str(script), "--json"]
    if skip_tests:
        cmd.append("--skip-tests")

    timeout = TEST_TIMEOUT if not skip_tests else STATIC_CHECK_TIMEOUT
    code, stdout, stderr, duration = _run_subprocess(cmd, timeout=timeout)

    artifact_path: str | None = None
    if stdout.strip():
        art = _write_artifact(bundle_dir, "rc_validation.json", stdout)
        artifact_path = str(art.relative_to(bundle_dir))

    if code < 0:
        return EvidenceCheck(
            name="rc_validation",
            category="static",
            status=ERROR,
            summary=f"Subprocess error: {stderr.strip()}",
            details=[stderr.strip()],
            duration_seconds=duration,
            artifact_path=artifact_path,
        )

    # Parse JSON output
    details: list[str] = []
    try:
        data = json.loads(stdout)
        overall = data.get("overall_status", "unknown")
        gates = data.get("gates", [])

        fail_count = 0
        warn_count = 0
        skip_count = 0
        pass_count = 0

        for gate in gates:
            gate_id = gate.get("gate_id", "?")
            gate_name = gate.get("name", "?")
            gate_status = gate.get("status", "?")
            gate_msg = gate.get("message", "")
            details.append(f"{gate_id} ({gate_name}): {gate_status} - {gate_msg}")

            if gate_status == "fail":
                fail_count += 1
            elif gate_status == "warn":
                warn_count += 1
            elif gate_status == "skip":
                skip_count += 1
            elif gate_status == "pass":
                pass_count += 1

        if overall == "pass":
            status = PASS
        elif overall == "fail":
            status = FAIL
        elif overall == "warn":
            status = WARN
        else:
            status = WARN

        summary = (
            f"RC validation: {pass_count} passed, {fail_count} failed, "
            f"{warn_count} warned, {skip_count} skipped"
        )
    except (json.JSONDecodeError, KeyError) as exc:
        status = ERROR
        summary = f"Failed to parse output: {exc}"
        details.append(stdout[:500] if stdout else "(no output)")

    return EvidenceCheck(
        name="rc_validation",
        category="static",
        status=status,
        summary=summary,
        details=details,
        duration_seconds=duration,
        artifact_path=artifact_path,
    )


def check_release_gate(
    bundle_dir: Path, *, pull_request: int | None = None
) -> EvidenceCheck:
    """C) Release Gate -- invoke run_phase8_release_gate.py --dry-run."""
    script = SCRIPT_DIR / "run_phase8_release_gate.py"
    if not script.exists():
        return EvidenceCheck(
            name="release_gate",
            category="static",
            status=ERROR,
            summary="run_phase8_release_gate.py not found",
            details=[f"Expected at: {script}"],
        )

    json_report_path = bundle_dir / "release_gate.json"
    cmd = [
        sys.executable,
        str(script),
        "--repo",
        "mattmre/OCR_LOCAL",
        "--dry-run",
        "--json-report",
        str(json_report_path),
    ]
    if pull_request:
        cmd.extend(["--pull-request", str(pull_request)])

    code, stdout, stderr, duration = _run_subprocess(
        cmd, timeout=OPERATIONAL_CHECK_TIMEOUT
    )

    artifact_path: str | None = None
    if json_report_path.exists():
        artifact_path = str(json_report_path.relative_to(bundle_dir))

    if code < 0:
        return EvidenceCheck(
            name="release_gate",
            category="static",
            status=ERROR,
            summary=f"Subprocess error: {stderr.strip()}",
            details=[stderr.strip()],
            duration_seconds=duration,
            artifact_path=artifact_path,
        )

    # Parse JSON report if it was written
    details: list[str] = []
    if json_report_path.exists():
        try:
            data = json.loads(json_report_path.read_text(encoding="utf-8"))
            overall = data.get("overall_status", "UNKNOWN")
            gates = data.get("gates", [])
            blockers = data.get("blockers", [])

            for gate in gates:
                gate_name = gate.get("gate_name", "?")
                gate_status = gate.get("status", "?")
                gate_detail = gate.get("detail", "")
                details.append(f"{gate_name}: {gate_status} - {gate_detail}")

            if blockers:
                for blocker in blockers:
                    details.append(f"BLOCKER: {blocker}")

            if overall in ("READY", "DRY-RUN-READY"):
                status = PASS
            elif overall == "ERROR":
                status = ERROR
            else:
                status = WARN

            pr_label = f" (PR #{pull_request})" if pull_request else ""
            summary = f"Release gate{pr_label}: {overall}"
        except (json.JSONDecodeError, KeyError) as exc:
            status = ERROR
            summary = f"Failed to parse release gate JSON: {exc}"
    else:
        # No JSON report written -- fall back to stdout/exit code
        if code == 0:
            status = PASS
            summary = "Release gate: passed (dry-run)"
        else:
            status = WARN
            summary = f"Release gate: exit code {code}"
        if stdout.strip():
            details.append(stdout.strip()[:500])
        if stderr.strip():
            details.append(stderr.strip()[:500])

    return EvidenceCheck(
        name="release_gate",
        category="static",
        status=status,
        summary=summary,
        details=details,
        duration_seconds=duration,
        artifact_path=artifact_path,
    )


# ---------------------------------------------------------------------------
# Tier 2: Operational checks
# ---------------------------------------------------------------------------


def check_cutover_preflight(bundle_dir: Path) -> EvidenceCheck:
    """D) Cutover Preflight -- invoke cutover_preflight.py --json."""
    script = SCRIPT_DIR / "cutover_preflight.py"
    if not script.exists():
        return EvidenceCheck(
            name="cutover_preflight",
            category="operational",
            status=ERROR,
            summary="cutover_preflight.py not found",
            details=[f"Expected at: {script}"],
        )

    cmd = [
        sys.executable,
        str(script),
        "--json",
        "--skip-connectivity",
    ]
    code, stdout, stderr, duration = _run_subprocess(
        cmd, timeout=OPERATIONAL_CHECK_TIMEOUT
    )

    artifact_path: str | None = None
    if stdout.strip():
        art = _write_artifact(bundle_dir, "cutover_preflight.json", stdout)
        artifact_path = str(art.relative_to(bundle_dir))

    if code < 0:
        return EvidenceCheck(
            name="cutover_preflight",
            category="operational",
            status=ERROR,
            summary=f"Subprocess error: {stderr.strip()}",
            details=[stderr.strip()],
            duration_seconds=duration,
            artifact_path=artifact_path,
        )

    # Parse JSON output
    details: list[str] = []
    try:
        data = json.loads(stdout)
        verdict = data.get("verdict", "UNKNOWN")
        checks = data.get("checks", [])
        summary_data = data.get("summary", {})

        pass_count = summary_data.get("pass", 0)
        fail_count = summary_data.get("fail", 0)
        warn_count = summary_data.get("warn", 0)
        skip_count = summary_data.get("skip", 0)

        for check in checks:
            cname = check.get("name", "?")
            cstatus = check.get("status", "?")
            cdetails = check.get("details", "")
            details.append(f"{cname}: {cstatus} - {cdetails}")

        if verdict == "READY":
            status = PASS
        elif verdict == "NOT READY":
            status = FAIL
        elif verdict == "PARTIAL":
            status = WARN
        else:
            status = WARN

        summary = (
            f"Cutover preflight: {verdict} "
            f"({pass_count} pass, {fail_count} fail, {warn_count} warn, {skip_count} skip)"
        )
    except (json.JSONDecodeError, KeyError) as exc:
        status = ERROR
        summary = f"Failed to parse output: {exc}"
        details.append(stdout[:500] if stdout else "(no output)")

    return EvidenceCheck(
        name="cutover_preflight",
        category="operational",
        status=status,
        summary=summary,
        details=details,
        duration_seconds=duration,
        artifact_path=artifact_path,
    )


def check_evidence_secret_scan(bundle_dir: Path) -> EvidenceCheck:
    """Scan every pre-manifest bundle artifact for high-severity secrets.

    The report records the exact files read. It excludes itself because it
    cannot scan bytes that have not been written; the signed manifest covers
    the completed report.
    """
    script = SCRIPT_DIR / "scan_evidence_secrets.py"
    report_path = bundle_dir / EVIDENCE_CREDENTIAL_SCAN_ARTIFACT
    if not script.exists():
        return EvidenceCheck(
            name="evidence_secret_scan",
            category="operational",
            status=ERROR,
            summary="scan_evidence_secrets.py not found",
            details=[f"Expected at: {script}"],
        )

    candidates = [
        path.resolve()
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name != EVIDENCE_CREDENTIAL_SCAN_ARTIFACT
    ]
    if not candidates:
        return EvidenceCheck(
            name="evidence_secret_scan",
            category="operational",
            status=ERROR,
            summary="No bundle artifacts available for secret scanning",
        )

    cmd = [sys.executable, str(script)]
    for candidate in candidates:
        cmd.extend(["--path", str(candidate)])
    cmd.extend(["--output", str(report_path)])
    code, stdout, stderr, duration = _run_subprocess(
        cmd, timeout=OPERATIONAL_CHECK_TIMEOUT
    )

    if not report_path.is_file():
        return EvidenceCheck(
            name="evidence_secret_scan",
            category="operational",
            status=ERROR,
            summary="Evidence secret scan did not produce a report",
            details=[stderr.strip() or stdout.strip()],
            duration_seconds=duration,
        )

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return EvidenceCheck(
            name="evidence_secret_scan",
            category="operational",
            status=ERROR,
            summary=f"Evidence secret scan report is invalid: {exc}",
            duration_seconds=duration,
            artifact_path=EVIDENCE_CREDENTIAL_SCAN_ARTIFACT,
        )
    if not isinstance(report, dict):
        return EvidenceCheck(
            name="evidence_secret_scan",
            category="operational",
            status=ERROR,
            summary="Evidence secret scan report root must be an object",
            duration_seconds=duration,
            artifact_path=EVIDENCE_CREDENTIAL_SCAN_ARTIFACT,
        )

    findings = report.get("findings")
    high_findings = (
        [
            finding
            for finding in findings
            if isinstance(finding, dict) and finding.get("severity") == "high"
        ]
        if isinstance(findings, list)
        else []
    )
    report["release_evidence_scan"] = {
        "schema_version": 1,
        "scanner_exit_code": code,
        "covered_artifacts": [
            str(candidate.relative_to(bundle_dir.resolve())) for candidate in candidates
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if code == 0 and isinstance(findings, list) and not high_findings:
        status = PASS
        summary = f"Evidence secret scan passed for {len(candidates)} artifact(s)"
        details: list[str] = []
    elif code == 1 or high_findings:
        status = FAIL
        summary = (
            f"Evidence secret scan found {len(high_findings)} high-severity "
            "finding(s)"
        )
        details = [
            f"{finding.get('path', '?')}:{finding.get('line', '?')} "
            f"{finding.get('pattern_name', '?')}"
            for finding in high_findings[:10]
        ]
    else:
        status = ERROR
        summary = f"Evidence secret scan failed with exit code {code}"
        details = [stderr.strip() or stdout.strip()]

    return EvidenceCheck(
        name="evidence_secret_scan",
        category="operational",
        status=status,
        summary=summary,
        details=details,
        duration_seconds=duration,
        artifact_path=EVIDENCE_CREDENTIAL_SCAN_ARTIFACT,
    )


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------


def compute_verdict(checks: list[EvidenceCheck]) -> str:
    """Compute overall verdict from check statuses.

    Returns "PASS", "FAIL", or "PARTIAL".
    """
    statuses = [c.status for c in checks]
    if FAIL in statuses or ERROR in statuses:
        return "FAIL"
    if WARN in statuses:
        return "PARTIAL"
    return "PASS"


def verdict_exit_code(verdict: str) -> int:
    """Map verdict to exit code: PASS=0, FAIL=1, PARTIAL=2."""
    if verdict == "PASS":
        return 0
    elif verdict == "FAIL":
        return 1
    else:
        return 2


# ---------------------------------------------------------------------------
# Bundle output
# ---------------------------------------------------------------------------


def build_manifest(
    checks: list[EvidenceCheck],
    bundle_dir: Path,
    version: str,
    overall_status: str,
    *,
    source_commit: str = "unknown",
    evidence_scope: str = "local-dry-run",
    decision: str = "NO-GO",
    production_gates: dict[str, dict] | None = None,
    extra_artifacts: list[dict] | None = None,
) -> dict:
    """Build the manifest.json payload."""
    artifacts = []
    for check in checks:
        if check.artifact_path:
            full_path = bundle_dir / check.artifact_path
            if full_path.exists():
                artifacts.append(
                    {
                        "name": check.artifact_path,
                        "path": check.artifact_path,
                        "sha256": _sha256_file(full_path),
                    }
                )

    summary_path = bundle_dir / "summary.md"
    if summary_path.is_file():
        artifacts.append(
            {
                "name": "summary.md",
                "path": "summary.md",
                "sha256": _sha256_file(summary_path),
            }
        )
    artifacts.extend(extra_artifacts or [])

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": version,
        "source_commit": source_commit,
        "evidence_scope": evidence_scope,
        "decision": decision,
        "overall_status": overall_status,
        "checks": [asdict(c) for c in checks],
        "production_gates": production_gates or {},
        "artifacts": artifacts,
    }


def build_summary_md(
    checks: list[EvidenceCheck],
    overall_status: str,
    version: str,
) -> str:
    """Build a human-readable markdown summary."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Release Evidence Summary",
        "",
        f"- **Generated**: {timestamp}",
        f"- **Version**: {version}",
        f"- **Overall Status**: {overall_status}",
        "",
        "## Check Results",
        "",
        "| Check | Category | Status | Summary | Duration |",
        "|-------|----------|--------|---------|----------|",
    ]

    for c in checks:
        dur = f"{c.duration_seconds:.1f}s" if c.duration_seconds > 0 else "-"
        lines.append(f"| {c.name} | {c.category} | {c.status} | {c.summary} | {dur} |")

    # Detail sections
    for c in checks:
        if c.details:
            lines.append("")
            lines.append(f"### {c.name}")
            lines.append("")
            for detail in c.details:
                lines.append(f"- {detail}")
            if c.artifact_path:
                lines.append(f"- Artifact: `{c.artifact_path}`")

    lines.append("")
    return "\n".join(lines)


def build_text_summary(checks: list[EvidenceCheck], overall_status: str) -> str:
    """Build a text summary for stdout output."""
    lines = [
        "=" * 60,
        "Release Evidence Report",
        "=" * 60,
        "",
    ]

    for c in checks:
        dur = f" ({c.duration_seconds:.1f}s)" if c.duration_seconds > 0 else ""
        lines.append(f"  [{c.status}] {c.name}{dur}")
        lines.append(f"         {c.summary}")
        if c.status in (FAIL, ERROR) and c.details:
            for detail in c.details[:5]:
                lines.append(f"         | {detail}")
        lines.append("")

    lines.append("-" * 60)
    lines.append(f"  Overall: {overall_status}")
    lines.append("-" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_evidence(
    *,
    full: bool = False,
    skip_tests: bool = False,
    pull_request: int | None = None,
    evidence_dir: Path | None = None,
    json_output: bool = False,
    production_gates_path: Path | None = None,
) -> int:
    """Run the evidence capture pipeline.

    Returns exit code: 0=PASS, 1=FAIL, 2=PARTIAL.
    """
    version = _read_version()
    source_commit = _read_git_commit()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Determine bundle directory
    if evidence_dir:
        bundle_dir = Path(evidence_dir).resolve()
    else:
        bundle_dir = (
            PROJECT_ROOT / "docs" / "reports" / f"evidence-{timestamp}"
        ).resolve()

    bundle_dir.mkdir(parents=True, exist_ok=True)

    checks: list[EvidenceCheck] = []

    # --- Tier 1: Static checks ---
    checks.append(check_version_consistency(bundle_dir))
    checks.append(check_rc_validation(bundle_dir, skip_tests=skip_tests))
    checks.append(check_release_gate(bundle_dir, pull_request=pull_request))

    # --- Tier 2: Operational checks (--full only) ---
    if full:
        checks.append(check_cutover_preflight(bundle_dir))

    production_gates: dict[str, dict] = {}
    production_artifacts: list[dict] = []
    evidence_scope = "local-operational-dry-run" if full else "local-dry-run"
    decision = "NO-GO"
    if production_gates_path is not None:
        production_errors = [] if full else ["--production-gates requires --full"]
        if full:
            production_gates, production_artifacts, collection_errors = (
                _collect_production_evidence(
                    Path(production_gates_path).resolve(),
                    bundle_dir,
                    version=version,
                    source_commit=source_commit,
                )
            )
            production_errors.extend(collection_errors)
        checks.append(
            EvidenceCheck(
                name="production_evidence",
                category="operational",
                status=FAIL if production_errors else PASS,
                summary=(
                    f"Production evidence has {len(production_errors)} error(s)"
                    if production_errors
                    else "All required production gates and artifacts are present"
                ),
                details=production_errors,
            )
        )
    summary_path = bundle_dir / "summary.md"
    if production_gates_path is not None and not production_errors:
        # The first scan captures the ordinary evidence files. Re-run after
        # adding the result so final summary bytes are also covered. The scan
        # report itself is integrity-covered by the manifest, not self-scanned.
        summary_path.write_text(
            build_summary_md(checks, compute_verdict(checks), version),
            encoding="utf-8",
        )
        checks.append(check_evidence_secret_scan(bundle_dir))
        overall_status = compute_verdict(checks)
        summary_path.write_text(
            build_summary_md(checks, overall_status, version), encoding="utf-8"
        )
        checks[-1] = check_evidence_secret_scan(bundle_dir)
        overall_status = compute_verdict(checks)
        if checks[-1].status == PASS:
            evidence_scope = "production"
            decision = "GO"
        else:
            evidence_scope = "production-invalid"
            decision = "NO-GO"
    else:
        overall_status = compute_verdict(checks)
        summary_path.write_text(
            build_summary_md(checks, overall_status, version), encoding="utf-8"
        )

    # Write integrity-covered manifest and checksum sidecar.
    manifest = build_manifest(
        checks,
        bundle_dir,
        version,
        overall_status,
        source_commit=source_commit,
        evidence_scope=evidence_scope,
        decision=decision,
        production_gates=production_gates,
        extra_artifacts=production_artifacts,
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (bundle_dir / "manifest.sha256").write_text(
        f"{_sha256_file(manifest_path)}  manifest.json\n", encoding="utf-8"
    )

    # Output
    if json_output:
        print(json.dumps(manifest, indent=2))
    else:
        text = build_text_summary(checks, overall_status)
        print(text)
        print(f"\n  Evidence bundle: {bundle_dir}")

    return verdict_exit_code(overall_status)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Release evidence bundle generator -- unified gate orchestration with artifact capture.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Include Tier 2 operational checks (cutover preflight)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        default=False,
        help="Skip test suite in RC validation (RC-004)",
    )
    parser.add_argument(
        "--pull-request",
        type=int,
        default=None,
        metavar="PR_NUM",
        help="PR number for release gate check",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Custom evidence bundle output directory (default: docs/reports/evidence-{timestamp}/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output manifest JSON to stdout instead of text summary",
    )
    parser.add_argument(
        "--production-gates",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Operator-authored GO manifest for all required production gates; "
            "referenced artifacts are copied into the bundle"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    return run_evidence(
        full=args.full,
        skip_tests=args.skip_tests,
        pull_request=args.pull_request,
        evidence_dir=args.evidence_dir,
        json_output=args.json_output,
        production_gates_path=args.production_gates,
    )


if __name__ == "__main__":
    sys.exit(main())
