#!/usr/bin/env python3
"""Validate Plan B translation provenance posture for release profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_local.translation.provenance import (  # noqa: E402
    ProvenanceMissingError,
    load_provenance_from_dir,
    validate_provenance,
)

TRUTHY = {"1", "true", "yes", "on"}
TRANSLATION_FLAG = "ENABLE_TRANSLATION"
PROVENANCE_FLAG = "OCR_TRANSLATION_ENFORCE_PROVENANCE"
DEFAULT_PROFILE_FILES = (
    ROOT / "helm" / "ocr-local" / "values-staging.yaml",
    ROOT / "helm" / "ocr-local" / "values-production.yaml",
)
DEFAULT_MODEL_DIRS = (ROOT / "models" / "translation",)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def _walk_env_maps(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}"
            if key == "env" and isinstance(value, dict):
                found.append((child_path, value))
            found.extend(_walk_env_maps(value, child_path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk_env_maps(value, f"{path}[{index}]"))
    return found


def validate_profile_file(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    problems: list[str] = []
    for env_path, env in _walk_env_maps(data):
        if not _truthy(env.get(TRANSLATION_FLAG, "false")):
            continue
        if not _truthy(env.get(PROVENANCE_FLAG, "false")):
            problems.append(
                f"{path}: {env_path} sets {TRANSLATION_FLAG}=true without "
                f"{PROVENANCE_FLAG}=true"
            )
    return problems


def _candidate_model_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if (root / "provenance.json").is_file():
        return [root]
    return sorted(
        child for child in root.iterdir() if child.is_dir() and (child / "provenance.json").is_file()
    )


def validate_model_dirs(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for root in paths:
        for model_dir in _candidate_model_dirs(root):
            try:
                validate_provenance(load_provenance_from_dir(model_dir), enforce=True)
            except (OSError, ProvenanceMissingError) as exc:
                problems.append(f"{model_dir}: invalid provenance.json: {exc}")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-file",
        action="append",
        type=Path,
        default=None,
        help="Helm values/release profile YAML file to validate",
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        type=Path,
        default=None,
        help="model bundle directory or parent directory containing model bundles",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile_files = args.profile_file or list(DEFAULT_PROFILE_FILES)
    model_dirs = args.model_dir or list(DEFAULT_MODEL_DIRS)

    problems: list[str] = []
    for profile_file in profile_files:
        if not profile_file.is_file():
            problems.append(f"{profile_file}: release profile file not found")
            continue
        problems.extend(validate_profile_file(profile_file))
    problems.extend(validate_model_dirs(model_dirs))

    if problems:
        print("translation release-profile validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    print("translation release-profile validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
