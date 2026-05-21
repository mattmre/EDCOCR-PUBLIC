#!/usr/bin/env python3
"""Validate production-grade plugin runtime attestation evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", default=os.environ.get("PLUGIN_RUNTIME_ATTESTATION", ""))
    parser.add_argument(
        "--allow-local-dev",
        action="store_true",
        help=(
            "validate a local/dev attestation shape without granting production "
            "release credit"
        ),
    )
    return parser.parse_args(argv)


def validate_attestation(path: Path, *, allow_local_dev: bool = False) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return [f"plugin runtime attestation not found: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"plugin runtime attestation is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["plugin runtime attestation must be a JSON object"]
    if data.get("schema_version") != "plugin-runtime-attestation-v1":
        problems.append("schema_version must be plugin-runtime-attestation-v1")
    environment = data.get("environment")
    if allow_local_dev:
        if environment not in {"local-dev", "production"}:
            problems.append("environment must be local-dev or production")
    elif environment != "production":
        problems.append("environment must be production for production release credit")
    for key in ("runner_id", "runner_version", "workload_identity", "timestamp"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            problems.append(f"{key} is required")
    image_digest = data.get("image_digest")
    if not isinstance(image_digest, str) or not image_digest.startswith("sha256:"):
        problems.append("image_digest must be a sha256:<digest> reference")
    elif not SHA256_RE.fullmatch(image_digest.removeprefix("sha256:")):
        problems.append("image_digest must contain a 64-character lowercase hex digest")
    signature = data.get("attestation_signature_sha256")
    if not isinstance(signature, str) or not SHA256_RE.fullmatch(signature):
        problems.append("attestation_signature_sha256 must be a 64-character lowercase hex SHA-256")
    for key in (
        "out_of_process",
        "network_disabled",
        "read_only_rootfs",
        "non_root_user",
        "cpu_limit_enforced",
        "memory_limit_enforced",
        "wall_time_limit_enforced",
        "no_new_privileges",
        "capabilities_dropped",
        "adversarial_escape_tests_passed",
    ):
        if data.get(key) is not True:
            problems.append(f"{key} must be true")
    if not isinstance(data.get("seccomp_profile"), str) or not data["seccomp_profile"].strip():
        problems.append("seccomp_profile is required")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(args.attestation).strip():
        print(
            "missing plugin runtime attestation: pass --attestation or set PLUGIN_RUNTIME_ATTESTATION",
            file=sys.stderr,
        )
        return 1
    problems = validate_attestation(
        Path(args.attestation),
        allow_local_dev=args.allow_local_dev,
    )
    if problems:
        print("plugin runtime attestation validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("plugin runtime attestation validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
