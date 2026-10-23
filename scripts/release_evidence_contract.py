"""Integrity and authenticity checks for production release evidence bundles."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

BUNDLE_SCHEMA_VERSION = 1
REQUIRED_PRODUCTION_GATES = (
    "local_validation",
    "phase8_rollout",
    "credentials",
    "storage_migration",
    "ha_dr",
    "keda_scaling",
    "node_loss",
    "approved_corpora",
    "telemetry_7d",
    "billing_sla",
    "public_release_approval",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_CREDENTIAL_SCAN_ARTIFACT = "evidence-secret-scan.json"


@dataclass
class BundleVerification:
    """Result of validating a release evidence bundle."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_timestamp(value: object) -> bool:
    """Return whether *value* is a timezone-aware ISO-8601 timestamp."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _path_contains_symlink(root: Path, relative: Path) -> bool:
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            return True
    return False


def _resolve_bundle_file(bundle_dir: Path, raw_path: object) -> tuple[Path | None, str]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "artifact path must be a non-empty string"
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None, f"artifact path escapes bundle: {raw_path!r}"
    if _path_contains_symlink(bundle_dir, relative):
        return None, f"artifact path contains symlink: {raw_path!r}"
    resolved = (bundle_dir / relative).resolve()
    try:
        resolved.relative_to(bundle_dir.resolve())
    except ValueError:
        return None, f"artifact path escapes bundle: {raw_path!r}"
    return resolved, ""


def _verify_signature(
    bundle_dir: Path,
    manifest_path: Path,
    expected_signer_fingerprint: str | None,
) -> str | None:
    normalized_fingerprint = (
        (expected_signer_fingerprint or "").replace(" ", "").upper()
    )
    if not normalized_fingerprint:
        return "approved release signing fingerprint is required"
    signature_path = bundle_dir / "manifest.json.sig"
    if not signature_path.is_file():
        return "detached signature manifest.json.sig is missing"
    try:
        result = subprocess.run(
            [
                "gpg",
                "--batch",
                "--status-fd",
                "1",
                "--verify",
                str(signature_path),
                str(manifest_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return "gpg is not installed"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unable to verify detached signature: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return f"detached signature verification failed: {detail[:240]}"
    valid_fingerprints = {
        line.split()[2].upper()
        for line in result.stdout.splitlines()
        if line.startswith("[GNUPG:] VALIDSIG ") and len(line.split()) >= 3
    }
    if normalized_fingerprint not in valid_fingerprints:
        return (
            "detached signature signer does not match approved fingerprint "
            f"{normalized_fingerprint}"
        )
    return None


def _verify_secret_scan(bundle_dir: Path, artifact_paths: set[str]) -> list[str]:
    """Verify the in-bundle scan report covers every signed artifact."""
    if EVIDENCE_CREDENTIAL_SCAN_ARTIFACT not in artifact_paths:
        return ["evidence secret scan report is not integrity-covered"]

    report_path, path_error = _resolve_bundle_file(
        bundle_dir, EVIDENCE_CREDENTIAL_SCAN_ARTIFACT
    )
    if path_error or report_path is None or not report_path.is_file():
        return ["evidence secret scan report is missing or unsafe"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"evidence secret scan report is invalid: {exc}"]
    if not isinstance(report, dict):
        return ["evidence secret scan report root must be an object"]

    errors: list[str] = []
    scan = report.get("release_evidence_scan")
    if not isinstance(scan, dict):
        return ["evidence secret scan metadata is missing"]
    if scan.get("schema_version") != 1:
        errors.append("evidence secret scan schema_version must be 1")
    if scan.get("scanner_exit_code") != 0:
        errors.append("evidence secret scan did not complete successfully")

    findings = report.get("findings")
    if not isinstance(findings, list):
        errors.append("evidence secret scan findings must be a list")
    elif any(
        isinstance(finding, dict) and finding.get("severity") == "high"
        for finding in findings
    ):
        errors.append("evidence secret scan contains high-severity findings")

    covered = scan.get("covered_artifacts")
    if not isinstance(covered, list) or not all(
        isinstance(path, str) and path for path in covered
    ):
        errors.append("evidence secret scan coverage must be a list of paths")
        return errors
    expected = artifact_paths - {EVIDENCE_CREDENTIAL_SCAN_ARTIFACT}
    if len(covered) != len(set(covered)):
        errors.append("evidence secret scan coverage contains duplicate paths")
    if set(covered) != expected:
        missing = sorted(expected - set(covered))
        unexpected = sorted(set(covered) - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        errors.append(
            "evidence secret scan coverage is incomplete: " + "; ".join(details)
        )
    return errors


def verify_release_evidence_bundle(
    bundle_dir: Path,
    *,
    expected_version: str,
    expected_commit: str,
    require_production: bool = True,
    require_signature: bool = True,
    expected_signer_fingerprint: str | None = None,
) -> BundleVerification:
    """Validate structure, hashes, release identity, gates, and signature."""
    bundle_dir = Path(bundle_dir).resolve()
    errors: list[str] = []
    manifest_path = bundle_dir / "manifest.json"

    if not bundle_dir.is_dir():
        return BundleVerification(False, [f"bundle directory not found: {bundle_dir}"])
    if not manifest_path.is_file():
        return BundleVerification(False, ["manifest.json is missing"])

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return BundleVerification(False, [f"manifest.json is invalid: {exc}"])
    if not isinstance(manifest, dict):
        return BundleVerification(False, ["manifest.json root must be an object"])

    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {BUNDLE_SCHEMA_VERSION}, "
            f"found {manifest.get('schema_version')!r}"
        )
    if manifest.get("version") != expected_version:
        errors.append(
            f"bundle version {manifest.get('version')!r} does not match "
            f"{expected_version!r}"
        )
    if manifest.get("source_commit") != expected_commit:
        errors.append(
            f"bundle source_commit {manifest.get('source_commit')!r} does not match "
            f"{expected_commit!r}"
        )
    if manifest.get("overall_status") != "PASS":
        errors.append(
            f"bundle overall_status must be 'PASS', found "
            f"{manifest.get('overall_status')!r}"
        )

    checks = manifest.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("bundle checks must be a non-empty list")
    else:
        failed_checks = [
            str(check.get("name", "?"))
            for check in checks
            if not isinstance(check, dict) or check.get("status") != "PASS"
        ]
        if failed_checks:
            errors.append("non-PASS bundle checks: " + ", ".join(failed_checks))

    artifacts = manifest.get("artifacts")
    artifact_paths: set[str] = set()
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("bundle artifacts must be a non-empty list")
        artifacts = []

    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"artifact {index} must be an object")
            continue
        raw_path = artifact.get("path")
        path, path_error = _resolve_bundle_file(bundle_dir, raw_path)
        if path_error:
            errors.append(path_error)
            continue
        assert path is not None
        normalized = str(Path(str(raw_path)))
        if normalized in artifact_paths:
            errors.append(f"duplicate artifact path: {normalized}")
            continue
        artifact_paths.add(normalized)
        expected_hash = artifact.get("sha256")
        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(
            expected_hash
        ):
            errors.append(f"artifact {normalized} has invalid SHA-256")
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(f"artifact file missing or unsafe: {normalized}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            errors.append(
                f"artifact hash mismatch for {normalized}: "
                f"expected {expected_hash}, found {actual_hash}"
            )

    if "summary.md" not in artifact_paths:
        errors.append("summary.md is not integrity-covered by the manifest")

    allowed_unlisted = {
        "manifest.json",
        "manifest.sha256",
        "manifest.json.sig",
    }
    for candidate in bundle_dir.rglob("*"):
        relative = str(candidate.relative_to(bundle_dir))
        if candidate.is_symlink():
            errors.append(f"bundle contains symlink: {relative}")
        elif candidate.is_file() and relative not in artifact_paths | allowed_unlisted:
            errors.append(f"bundle file is not integrity-covered: {relative}")

    checksum_path = bundle_dir / "manifest.sha256"
    if not checksum_path.is_file():
        errors.append("manifest.sha256 is missing")
    else:
        checksum_text = checksum_path.read_text(encoding="utf-8").strip()
        expected_line = f"{sha256_file(manifest_path)}  manifest.json"
        if checksum_text != expected_line:
            errors.append("manifest.sha256 does not match manifest.json")

    if require_production:
        if manifest.get("evidence_scope") != "production":
            errors.append(
                "bundle evidence_scope is not 'production'; local/dry-run bundles "
                "cannot satisfy release readiness"
            )
        if manifest.get("decision") != "GO":
            errors.append("production bundle decision must be 'GO'")
        production_gates = manifest.get("production_gates")
        if not isinstance(production_gates, dict):
            errors.append("production_gates must be an object")
            production_gates = {}
        for gate_name in REQUIRED_PRODUCTION_GATES:
            gate = production_gates.get(gate_name)
            if not isinstance(gate, dict):
                errors.append(f"production gate missing: {gate_name}")
                continue
            if gate.get("status") != "PASS":
                errors.append(f"production gate {gate_name} is not PASS")
            if not str(gate.get("approved_by", "")).strip():
                errors.append(f"production gate {gate_name} has no approved_by")
            if not is_timestamp(gate.get("completed_at")):
                errors.append(f"production gate {gate_name} has invalid completed_at")
            gate_artifacts = gate.get("artifacts")
            if not isinstance(gate_artifacts, list) or not gate_artifacts:
                errors.append(f"production gate {gate_name} has no artifacts")
                continue
            for gate_artifact in gate_artifacts:
                if gate_artifact not in artifact_paths:
                    errors.append(
                        f"production gate {gate_name} references unverified artifact "
                        f"{gate_artifact!r}"
                    )
        errors.extend(_verify_secret_scan(bundle_dir, artifact_paths))
        if not any(
            isinstance(check, dict)
            and check.get("name") == "evidence_secret_scan"
            and check.get("status") == "PASS"
            for check in checks
        ):
            errors.append("production bundle has no passing evidence_secret_scan check")

    if require_signature:
        signature_error = _verify_signature(
            bundle_dir, manifest_path, expected_signer_fingerprint
        )
        if signature_error:
            errors.append(signature_error)

    return BundleVerification(not errors, errors, manifest)
