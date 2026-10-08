#!/usr/bin/env python3
"""Validate production TSA assurance evidence without making network calls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_local.translation.tsa import _extract_rfc3161_status  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default=os.environ.get("TRANSLATION_TSA_ASSURANCE", ""))
    return parser.parse_args(argv)


def _path(base: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(value)
    return p if p.is_absolute() else base / p


def validate_tsa_assurance(path: Path) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return [f"TSA assurance artifact not found: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"TSA assurance artifact is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["TSA assurance artifact must be a JSON object"]
    if data.get("schema_version") != "translation-tsa-assurance-v1":
        problems.append("schema_version must be translation-tsa-assurance-v1")
    if data.get("approval_status") != "approved":
        problems.append("approval_status must be approved")
    for key in ("tsa_url", "tsa_policy_oid", "approved_by", "approval_ticket"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            problems.append(f"{key} is required")

    tsr_path = _path(path.parent, data.get("tsr_path"))
    if tsr_path is None or not tsr_path.is_file():
        problems.append("tsr_path must reference a stored RFC3161 timestamp response")
    else:
        try:
            status = _extract_rfc3161_status(tsr_path.read_bytes())
        except Exception as exc:
            problems.append(f"tsr_path is not a parseable RFC3161 TimeStampResp: {exc}")
        else:
            if status not in {0, 1}:
                problems.append(f"RFC3161 TimeStampResp status is not granted/grantedWithMods: {status}")

    for key in ("certificate_chain_path", "trust_store_path", "verification_log_path"):
        p = _path(path.parent, data.get(key))
        if p is None or not p.is_file():
            problems.append(f"{key} must reference an operator-supplied file")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(args.evidence).strip():
        print(
            "missing TSA assurance artifact: pass --evidence or set TRANSLATION_TSA_ASSURANCE",
            file=sys.stderr,
        )
        return 1
    problems = validate_tsa_assurance(Path(args.evidence))
    if problems:
        print("translation TSA assurance validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("translation TSA assurance validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
