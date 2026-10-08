#!/usr/bin/env python3
"""Check local prerequisites for a real CT2 translation model bundle.

This is a prerequisite checker, not an inference proof. It verifies that the
operator supplied a model directory and that the directory carries enforced
Plan B provenance. A live CT2 evidence run still needs the real runtime and
weights supplied via ``TRANSLATION_CT2_MODEL_DIR``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_local.translation.provenance import (  # noqa: E402
    ProvenanceCorruptError,
    ProvenanceMissingError,
    load_provenance_from_dir,
    validate_provenance,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="CT2 model bundle directory. Defaults to TRANSLATION_CT2_MODEL_DIR.",
    )
    return parser.parse_args(argv)


def check_model_bundle(model_dir: Path) -> list[str]:
    problems: list[str] = []
    if not model_dir.exists():
        return [f"model directory does not exist: {model_dir}"]
    if not model_dir.is_dir():
        return [f"model path is not a directory: {model_dir}"]

    try:
        provenance = validate_provenance(
            load_provenance_from_dir(model_dir),
            enforce=True,
        )
    except (FileNotFoundError, ProvenanceCorruptError, ProvenanceMissingError) as exc:
        problems.append(str(exc))
    else:
        if provenance.weights_sha256 == "not_loaded":
            problems.append(
                "weights_sha256 is 'not_loaded'; real CT2 evidence requires a "
                "model bundle digest for the supplied weights"
            )

    candidate_files = [path for path in model_dir.rglob("*") if path.is_file()]
    non_metadata = [
        path
        for path in candidate_files
        if path.name not in {"provenance.json", "MODEL_SHA256", "README.md"}
    ]
    if not non_metadata:
        problems.append(
            "model bundle has no non-metadata files; real CT2 evidence requires weights/tokenizer files"
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_model_dir = os.environ.get("TRANSLATION_CT2_MODEL_DIR", "").strip()
    if args.model_dir is None and not env_model_dir:
        print(
            "missing model directory: pass --model-dir or set TRANSLATION_CT2_MODEL_DIR",
            file=sys.stderr,
        )
        return 1
    model_dir = args.model_dir or Path(env_model_dir)

    problems = check_model_bundle(model_dir)
    if problems:
        print("translation model-bundle prerequisite check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    print(
        "translation model-bundle prerequisite check passed; run CT2 inference evidence separately"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
