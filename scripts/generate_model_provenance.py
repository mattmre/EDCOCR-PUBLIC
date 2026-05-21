#!/usr/bin/env python3
"""Generate ``provenance.json`` for a translation model bundle.

Plan B Wave M2 PR B19 (E-B-008 / RED-07).  Writes a SLSA v1.0 + in-toto
+ CycloneDX SBOM provenance descriptor next to the model weights so
``ocr_local.translation.provenance.validate_engine_provenance`` can
gate engine binding when ``translation_enforce_provenance=True``.

Usage::

    python scripts/generate_model_provenance.py \\
        --model-dir /opt/models/opus-mt-en-fr \\
        --slsa-uri https://models.example/opus-mt-en-fr/slsa-v1.intoto.jsonl \\
        --intoto-attestation /opt/attestations/opus-mt-en-fr.intoto.jsonl \\
        --sbom /opt/sboms/opus-mt-en-fr.cdx.json \\
        --license CC-BY-4.0 \\
        --runtime-version 4.6.1

The script computes SHA-256 over the in-toto attestation file and the
SBOM file, and over every weights file inside ``model_dir`` (excluding
the ``provenance.json`` it is about to write).  The aggregate
weights digest is written as ``weights_sha256`` -- a SHA-256 of a
canonical sorted list of ``(relative_path, file_sha256)`` pairs so two
identical model bundles produce the same digest regardless of
filesystem ordering.

Exit codes:
* ``0`` -- provenance.json written successfully
* ``2`` -- argument validation failed (e.g. missing file)
* ``3`` -- attestation / SBOM file unreadable
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

# Reuse the validator's constants and digest helper so the generator
# and the consumer agree on file name and hashing semantics.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ocr_local.translation.provenance import (  # noqa: E402
    PROVENANCE_FILE_NAME,
    compute_file_sha256,
    validate_provenance,
)

EXIT_OK = 0
EXIT_ARG_ERROR = 2
EXIT_IO_ERROR = 3


@dataclasses.dataclass(frozen=True)
class GeneratorArgs:
    model_dir: Path
    slsa_uri: str
    intoto_attestation: Path
    sbom: Path
    license: str
    runtime_version: str
    overwrite: bool


def _parse_args(argv: list[str] | None = None) -> GeneratorArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Generate provenance.json for a translation model bundle "
            "(Plan B Wave M2 PR B19, per E-B-008 / RED-07)."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing the model weights",
    )
    parser.add_argument(
        "--slsa-uri",
        required=True,
        help="URI of the SLSA v1.0 provenance statement (https:// or oci://)",
    )
    parser.add_argument(
        "--intoto-attestation",
        required=True,
        type=Path,
        help="Path to the in-toto attestation envelope (JSONL)",
    )
    parser.add_argument(
        "--sbom",
        required=True,
        type=Path,
        help="Path to the CycloneDX SBOM (JSON)",
    )
    parser.add_argument(
        "--license",
        required=True,
        help="SPDX license identifier (e.g. Apache-2.0, CC-BY-4.0)",
    )
    parser.add_argument(
        "--runtime-version",
        required=True,
        help="ctranslate2 (or other) runtime version pinned to the bundle",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing provenance.json (default: refuse)",
    )
    ns = parser.parse_args(argv)
    return GeneratorArgs(
        model_dir=ns.model_dir,
        slsa_uri=ns.slsa_uri,
        intoto_attestation=ns.intoto_attestation,
        sbom=ns.sbom,
        license=ns.license,
        runtime_version=ns.runtime_version,
        overwrite=ns.overwrite,
    )


def aggregate_weights_digest(model_dir: Path) -> str:
    """Compute a deterministic SHA-256 over all weights files.

    The aggregate is SHA-256 of a JSON-encoded sorted list of
    ``[relative_path, file_sha256]`` pairs (relative paths use forward
    slashes for cross-platform stability).  Files named
    ``provenance.json`` and any directory entries are excluded -- the
    generator must not hash the artifact it is about to write, and
    aggregate hashing is over file contents only.
    """

    pairs: list[tuple[str, str]] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == PROVENANCE_FILE_NAME:
            continue
        rel = path.relative_to(model_dir).as_posix()
        pairs.append((rel, compute_file_sha256(path)))
    canonical = json.dumps(pairs, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_provenance_dict(args: GeneratorArgs) -> dict[str, str]:
    if not args.model_dir.is_dir():
        print(
            f"error: --model-dir {args.model_dir} is not a directory",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_ARG_ERROR)
    if not args.intoto_attestation.is_file():
        print(
            f"error: --intoto-attestation {args.intoto_attestation} not found",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_ARG_ERROR)
    if not args.sbom.is_file():
        print(f"error: --sbom {args.sbom} not found", file=sys.stderr)
        raise SystemExit(EXIT_ARG_ERROR)

    try:
        intoto_sha = compute_file_sha256(args.intoto_attestation)
        sbom_sha = compute_file_sha256(args.sbom)
        weights_sha = aggregate_weights_digest(args.model_dir)
    except OSError as exc:
        print(f"error: failed to hash provenance inputs: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_IO_ERROR) from exc

    return {
        "slsa_provenance_uri": args.slsa_uri,
        "intoto_attestation_sha256": intoto_sha,
        "sbom_sha256": sbom_sha,
        "weights_sha256": weights_sha,
        "license": args.license,
        "runtime_version": args.runtime_version,
    }


def write_provenance(model_dir: Path, payload: dict[str, str], *, overwrite: bool) -> Path:
    out_path = model_dir / PROVENANCE_FILE_NAME
    if out_path.exists() and not overwrite:
        print(
            f"error: {out_path} already exists; pass --overwrite to replace",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_ARG_ERROR)
    # Validate before writing -- the same code path engines will run.
    validate_provenance(payload, enforce=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_provenance_dict(args)
    out_path = write_provenance(args.model_dir, payload, overwrite=args.overwrite)
    print(f"wrote {out_path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
