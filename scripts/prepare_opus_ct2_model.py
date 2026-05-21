#!/usr/bin/env python3
"""Prepare a free OPUS-MT CTranslate2 model for local evidence.

The default model is Helsinki-NLP/opus-mt-en-fr. The command downloads from
Hugging Face and converts to an int8 CPU CTranslate2 bundle. It makes no paid
API calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Helsinki-NLP/opus-mt-en-fr")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/translation_models/opus-mt-en-fr-ct2"),
    )
    parser.add_argument("--quantization", default="int8")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError as exc:
        print(
            "missing ctranslate2 converter; run: python -m pip install -r requirements-translation.txt transformers",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        TransformersConverter(args.model_id).convert(
            str(args.output_dir),
            quantization=args.quantization,
            force=True,
        )
    except Exception as exc:
        print(f"OPUS-MT CT2 model preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"prepared OPUS-MT CT2 model: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
