#!/usr/bin/env python3
"""Run local bootstrap checks over the translation golden corpus.

This harness intentionally separates cheap local reference checks from true
COMETKiwi/Qwen runtime evidence. It can verify corpus shape and exact-reference
fixture scoring without downloading large models or calling paid APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "tests" / "corpus" / "translation_golden" / "v1.0.0" / "manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--candidate-mode",
        choices=("reference",),
        default="reference",
        help="reference mode uses the human reference as deterministic candidate",
    )
    parser.add_argument("--min-exact-match", type=float, default=1.0)
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    return data


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def score_reference_mode(manifest: dict[str, Any]) -> dict[str, Any]:
    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("manifest must contain non-empty segments")
    exact = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("segment must be an object")
        candidate = segment.get("reference")
        reference = segment.get("reference")
        if not isinstance(candidate, str) or not isinstance(reference, str):
            raise ValueError("segment reference must be a string")
        if _normalise(candidate) == _normalise(reference):
            exact += 1
    return {
        "mode": "reference",
        "segment_count": len(segments),
        "exact_match_rate": exact / len(segments),
        "claim": "bootstrap harness only; not COMETKiwi or Qwen live evidence",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = _load_manifest(args.manifest)
        result = score_reference_mode(manifest)
    except Exception as exc:
        print(f"translation golden-corpus scoring failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["exact_match_rate"] < args.min_exact_match:
        print(
            f"exact_match_rate below threshold: {result['exact_match_rate']} < {args.min_exact_match}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
