#!/usr/bin/env python3
"""Validate Plan C production/federation evidence artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default=os.environ.get("TRANSLATION_CLUSTER_EVIDENCE", ""))
    parser.add_argument(
        "--allow-local-dev",
        action="store_true",
        help=(
            "validate local Docker/kind rehearsal evidence shape without "
            "granting production BHS credit"
        ),
    )
    return parser.parse_args(argv)


def _require_bool(data: dict[str, Any], key: str, problems: list[str]) -> None:
    if data.get(key) is not True:
        problems.append(f"{key} must be true")


def _require_bool_field(data: dict[str, Any], key: str, problems: list[str]) -> None:
    if not isinstance(data.get(key), bool):
        problems.append(f"{key} must be a boolean")


def _require_file(base: Path, data: dict[str, Any], key: str, problems: list[str]) -> None:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{key} is required")
        return
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    if not p.is_file():
        problems.append(f"{key} does not exist: {p}")
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    needles = {
        "helm_log_path": ("STATUS:", "MANIFEST:"),
        "failover_log_path": ("failover",),
        "residency_log_path": ("residency",),
        "mtls_log_path": ("mtls", "identity"),
    }
    for needle in needles.get(key, ()):
        if needle.lower() not in text.lower():
            problems.append(f"{key} does not contain expected marker: {needle}")


def validate_cluster_evidence(path: Path, *, allow_local_dev: bool = False) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return [f"cluster evidence artifact not found: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"cluster evidence artifact is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["cluster evidence artifact must be a JSON object"]
    if data.get("schema_version") != "translation-cluster-evidence-v1":
        problems.append("schema_version must be translation-cluster-evidence-v1")
    environment = data.get("environment")
    if allow_local_dev:
        if environment not in {"local-dev", "production"}:
            problems.append("environment must be local-dev or production")
    elif environment != "production":
        problems.append("environment must be production for production BHS credit")
    clusters = data.get("clusters")
    min_clusters = 1 if allow_local_dev and environment == "local-dev" else 2
    if not isinstance(clusters, list) or len(clusters) < min_clusters:
        if allow_local_dev and environment == "local-dev":
            problems.append("clusters must list at least one local-dev cluster")
        else:
            problems.append("clusters must list at least two production clusters")
    _require_bool(data, "helm_install_or_upgrade_passed", problems)
    drill_keys = (
        "outbox_failover_drill_passed",
        "residency_routing_drill_passed",
        "mtls_identity_drill_passed",
        "federation_health_passed",
    )
    for key in drill_keys:
        if allow_local_dev and environment == "local-dev":
            _require_bool_field(data, key, problems)
        else:
            _require_bool(data, key, problems)
    for key in ("helm_log_path", "failover_log_path", "residency_log_path", "mtls_log_path"):
        _require_file(path.parent, data, key, problems)
    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(args.evidence).strip():
        print(
            "missing cluster evidence artifact: pass --evidence or set TRANSLATION_CLUSTER_EVIDENCE",
            file=sys.stderr,
        )
        return 1
    problems = validate_cluster_evidence(
        Path(args.evidence),
        allow_local_dev=args.allow_local_dev,
    )
    if problems:
        print("translation cluster evidence validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("translation cluster evidence validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
