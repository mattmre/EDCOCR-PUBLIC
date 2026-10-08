#!/usr/bin/env python3
"""Validate production approval evidence for a translation model bundle.

This is stricter than ``check_translation_model_bundle.py``. It fails closed
unless an operator-supplied approval artifact ties a concrete model bundle to
source/license/provenance, conversion command, and quality evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_translation_model_bundle import check_model_bundle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", default=os.environ.get("TRANSLATION_MODEL_APPROVAL", ""))
    parser.add_argument("--model-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_str(data: dict[str, Any], key: str, problems: list[str]) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{key} is required")
        return ""
    return value.strip()


def validate_approval(path: Path, model_dir_override: Path | None = None) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return [f"approval artifact not found: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"approval artifact is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["approval artifact must be a JSON object"]

    allowed = {
        "schema_version",
        "approval_status",
        "approved_by",
        "approval_ticket",
        "model_dir",
        "model_id",
        "source_url",
        "source_revision",
        "license",
        "conversion_command",
        "provenance_json_sha256",
        "model_bundle_sha256",
        "quality_report_path",
        "min_exact_match_rate",
        "min_segments",
        "operator_notes",
    }
    extra = sorted(set(data) - allowed)
    if extra:
        problems.append(f"unexpected approval field(s): {', '.join(extra)}")

    if data.get("schema_version") != "translation-model-approval-v1":
        problems.append("schema_version must be translation-model-approval-v1")
    if data.get("approval_status") != "approved":
        problems.append("approval_status must be approved")
    for key in (
        "approved_by",
        "approval_ticket",
        "model_id",
        "source_url",
        "source_revision",
        "license",
        "conversion_command",
        "provenance_json_sha256",
        "model_bundle_sha256",
        "quality_report_path",
    ):
        _require_str(data, key, problems)

    model_dir_value = model_dir_override or Path(str(data.get("model_dir", "")))
    if not model_dir_value.is_absolute():
        model_dir_value = path.parent / model_dir_value
    problems.extend(check_model_bundle(model_dir_value))

    provenance = model_dir_value / "provenance.json"
    if provenance.exists():
        actual = _sha256_file(provenance)
        if actual != data.get("provenance_json_sha256"):
            problems.append("provenance_json_sha256 does not match model bundle provenance.json")

    digest_file = model_dir_value / "MODEL_SHA256"
    if digest_file.exists():
        actual_bundle = digest_file.read_text(encoding="utf-8").strip()
        if actual_bundle != data.get("model_bundle_sha256"):
            problems.append("model_bundle_sha256 does not match MODEL_SHA256")

    quality_report = Path(str(data.get("quality_report_path", "")))
    if not quality_report.is_absolute():
        quality_report = path.parent / quality_report
    if not quality_report.is_file():
        problems.append(f"quality_report_path does not exist: {quality_report}")
    else:
        try:
            quality = json.loads(quality_report.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"quality_report_path is not valid JSON: {exc}")
        else:
            exact = quality.get("exact_match_rate")
            segments = quality.get("segments")
            min_exact = data.get("min_exact_match_rate", 0.0)
            min_segments = data.get("min_segments", 1)
            if not isinstance(exact, int | float) or exact < min_exact:
                problems.append("quality exact_match_rate is below approval threshold")
            if not isinstance(segments, int) or segments < min_segments:
                problems.append("quality segment count is below approval threshold")

    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(args.approval).strip():
        print(
            "missing approval artifact: pass --approval or set TRANSLATION_MODEL_APPROVAL",
            file=sys.stderr,
        )
        return 1
    problems = validate_approval(Path(args.approval), args.model_dir)
    if problems:
        print("translation model approval validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("translation model approval validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
