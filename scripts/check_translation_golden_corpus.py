#!/usr/bin/env python3
"""Check the Plan B Qwen/COMETKiwi golden translation corpus prerequisite.

The checker validates corpus shape only. It does not claim Qwen or COMETKiwi
live evidence unless the referenced model/runtime evidence is supplied and
executed by the operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = ROOT / "tests" / "corpus" / "translation_golden" / "v1.0.0"
REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "corpus_id",
    "language_pairs",
    "license_attribution",
    "segments",
    "cometkiwi_baselines",
    "qwen_comparison_report",
}
REQUIRED_SEGMENT_KEYS = {
    "id",
    "source_lang",
    "target_lang",
    "source",
    "reference",
    "domain",
    "source_fixture",
    "source_row",
    "license",
    "sha256",
}
APPROVAL_ENV = "TRANSLATION_GOLDEN_CORPUS_UPDATE_APPROVED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _normalize_digest(value: object) -> str:
    return str(value).replace("-", "").lower()


def _format_digest(value: str) -> str:
    normalized = _normalize_digest(value)
    if not _SHA256_RE.fullmatch(normalized):
        return value
    return "-".join(normalized[index : index + 8] for index in range(0, 64, 8))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Golden corpus directory containing manifest.json",
    )
    parser.add_argument(
        "--min-segments",
        type=int,
        default=200,
        help="Minimum curated segment count for the local prerequisite gate",
    )
    parser.add_argument("--manifest-hash", type=Path, default=None)
    parser.add_argument("--update-manifest-hash", action="store_true")
    return parser.parse_args(argv)


def _segment_hash(segment: dict[str, Any]) -> str:
    payload = {
        "id": segment.get("id"),
        "source_lang": segment.get("source_lang"),
        "target_lang": segment.get("target_lang"),
        "source": segment.get("source"),
        "reference": segment.get("reference"),
        "domain": segment.get("domain"),
        "source_fixture": segment.get("source_fixture"),
        "source_row": segment.get("source_row"),
        "license": segment.get("license"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(corpus_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, [f"manifest not found: {manifest_path}"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"manifest is not valid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, ["manifest must be a JSON object"]
    return payload, []


def check_golden_corpus(corpus_dir: Path, *, min_segments: int) -> list[str]:
    manifest, problems = _load_manifest(corpus_dir)
    if manifest is None:
        return problems

    missing = sorted(REQUIRED_MANIFEST_KEYS.difference(manifest))
    if missing:
        problems.append(f"manifest missing required keys: {', '.join(missing)}")

    language_pairs = manifest.get("language_pairs")
    if not isinstance(language_pairs, list) or not language_pairs:
        problems.append("language_pairs must be a non-empty list")

    segments = manifest.get("segments")
    if not isinstance(segments, list):
        problems.append("segments must be a list")
        segments = []
    if len(segments) < min_segments:
        problems.append(
            f"segments has {len(segments)} entries; requires at least {min_segments}"
        )

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            problems.append(f"segments[{index}] must be an object")
            continue
        missing_segment = sorted(REQUIRED_SEGMENT_KEYS.difference(segment))
        if missing_segment:
            problems.append(
                f"segments[{index}] missing required keys: {', '.join(missing_segment)}"
            )
            continue
        if not isinstance(segment.get("license"), str) or not segment["license"].strip():
            problems.append(f"segments[{index}] license must be non-empty")
        if not isinstance(segment.get("source_fixture"), str) or not segment["source_fixture"].strip():
            problems.append(f"segments[{index}] source_fixture must be non-empty")
        expected_hash = _segment_hash(segment)
        if _normalize_digest(segment.get("sha256")) != expected_hash:
            problems.append(f"segments[{index}] sha256 mismatch")

    for key in ("cometkiwi_baselines", "qwen_comparison_report"):
        value = manifest.get(key)
        if not isinstance(value, dict) or not value:
            problems.append(f"{key} must be a non-empty object")

    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = check_golden_corpus(args.corpus_dir, min_segments=args.min_segments)
    manifest_path = args.corpus_dir / "manifest.json"
    hash_path = args.manifest_hash or args.corpus_dir / "manifest.sha256"
    if manifest_path.is_file():
        actual_manifest_hash = _file_hash(manifest_path)
        approved = os.environ.get(APPROVAL_ENV, "").strip().lower() == "true"
        if args.update_manifest_hash:
            if not approved:
                print(
                    f"refusing to update {hash_path}: set {APPROVAL_ENV}=true for approved regeneration",
                    file=sys.stderr,
                )
                return 1
            hash_path.write_text(_format_digest(actual_manifest_hash) + "\n", encoding="utf-8")
            print(f"updated translation golden-corpus manifest hash: {hash_path}")
        elif not hash_path.is_file():
            problems.append(f"manifest hash not found: {hash_path}")
        else:
            recorded_hash = hash_path.read_text(encoding="utf-8").strip()
            if _normalize_digest(recorded_hash) != actual_manifest_hash:
                problems.append("manifest.sha256 mismatch")
    if problems:
        print("translation golden-corpus prerequisite check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    print("translation golden-corpus prerequisite check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
