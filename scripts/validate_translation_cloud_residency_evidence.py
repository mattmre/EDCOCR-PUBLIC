#!/usr/bin/env python3
"""Validate cloud translation residency/BAA evidence without API calls.

This is a local evidence-artifact validator. It does not call Google Cloud or
any paid provider. Production cloud translation remains blocked until an
operator supplies a real artifact that this validator accepts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TRUTHY = {"1", "true", "yes", "on"}
ROUTING_POLICY_RE = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_PATH = (
    ROOT
    / "docs"
    / "planning"
    / "2026-04-24-translation-swarm"
    / "cloud-residency-evidence.schema.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        default=os.environ.get("TRANSLATION_CLOUD_RESIDENCY_EVIDENCE", ""),
        help="JSON evidence artifact path; defaults to TRANSLATION_CLOUD_RESIDENCY_EVIDENCE",
    )
    return parser.parse_args(argv)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def validate_schema_shape(data: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    allowed = {
        "deployment_env",
        "gemini_provider",
        "ai_studio_enabled",
        "region",
        "retention_policy",
        "training_disabled",
        "phi_or_legal_production",
        "baa_evidence_path",
        "compliance_evidence_path",
        "routing_policy_sha256",
        "notes",
    }
    required = {
        "deployment_env",
        "gemini_provider",
        "ai_studio_enabled",
        "region",
        "retention_policy",
        "training_disabled",
        "phi_or_legal_production",
        "routing_policy_sha256",
    }

    extra = sorted(set(data) - allowed)
    if extra:
        problems.append(f"unexpected evidence field(s): {', '.join(extra)}")

    missing = sorted(required - set(data))
    if missing:
        problems.append(f"missing required evidence field(s): {', '.join(missing)}")

    if "ai_studio_enabled" in data and not isinstance(data["ai_studio_enabled"], bool):
        problems.append("ai_studio_enabled must be a boolean")
    if "training_disabled" in data and not isinstance(data["training_disabled"], bool):
        problems.append("training_disabled must be a boolean")
    if "phi_or_legal_production" in data and not isinstance(data["phi_or_legal_production"], bool):
        problems.append("phi_or_legal_production must be a boolean")

    routing_sha = data.get("routing_policy_sha256")
    if not isinstance(routing_sha, str) or not ROUTING_POLICY_RE.fullmatch(routing_sha):
        problems.append("routing_policy_sha256 must be a 64-character lowercase hex SHA-256")

    if _truthy(data.get("phi_or_legal_production")) and "baa_evidence_path" not in data:
        problems.append("baa_evidence_path is required when PHI/legal production is claimed")

    return problems


def validate_evidence(path: Path | None) -> list[str]:
    problems: list[str] = []
    if path is None:
        return ["missing evidence path: pass --evidence or set TRANSLATION_CLOUD_RESIDENCY_EVIDENCE"]
    if not path.is_file():
        return [f"evidence artifact not found: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"evidence artifact is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["evidence artifact must be a JSON object"]

    problems.extend(validate_schema_shape(data))

    if data.get("deployment_env") != "production":
        problems.append("deployment_env must be 'production' for production cloud evidence")
    if data.get("gemini_provider") != "vertex":
        problems.append("gemini_provider must be 'vertex'; AI Studio is not a production path")
    if _truthy(data.get("ai_studio_enabled")):
        problems.append("ai_studio_enabled must be false for production")
    if not isinstance(data.get("region"), str) or not data["region"].strip():
        problems.append("region must be explicit")
    if data.get("retention_policy") != "zero_retention":
        problems.append("retention_policy must be 'zero_retention'")
    if not _truthy(data.get("training_disabled")):
        problems.append("training_disabled must be true")

    if _truthy(data.get("phi_or_legal_production")):
        baa_path = data.get("baa_evidence_path")
        if not isinstance(baa_path, str) or not baa_path.strip():
            problems.append("baa_evidence_path is required when PHI/legal production is claimed")
        else:
            candidate = Path(baa_path)
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            if not candidate.exists():
                problems.append(f"baa_evidence_path does not exist: {candidate}")

    compliance_path = data.get("compliance_evidence_path")
    if isinstance(compliance_path, str) and compliance_path.strip():
        candidate = Path(compliance_path)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if not candidate.exists():
            problems.append(f"compliance_evidence_path does not exist: {candidate}")

    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evidence = Path(args.evidence) if str(args.evidence).strip() else None
    problems = validate_evidence(evidence)
    if problems:
        print("cloud residency evidence validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("cloud residency evidence validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
