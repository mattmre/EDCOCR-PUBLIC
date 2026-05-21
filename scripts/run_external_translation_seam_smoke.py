"""Smoke-test the OCR -> EXTERNAL_TRANSLATION bundle seam.

This script assumes the standalone EXTERNAL_TRANSLATION service is already running.
It submits a tiny DocumentBundle through the OCR-side external client and prints
a compact JSON summary suitable for runbooks and deployment checks.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from ocr_local.document_bundle import build_document_bundle
from ocr_local.translation.external_client import (
    DEFAULT_TRANSLATION_SERVICE_URL,
    TranslationServiceClient)
from ocr_local.translation.readiness import external_translation_readiness
from pipeline_config import create_pipeline_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("EXTERNAL_TRANSLATION_URL", DEFAULT_TRANSLATION_SERVICE_URL),
        help="Base URL for the standalone EXTERNAL_TRANSLATION service.")
    parser.add_argument(
        "--provider",
        default=os.environ.get("EXTERNAL_TRANSLATION_PROVIDER_ID", "deterministic_ci"),
        help="Provider id to request from EXTERNAL_TRANSLATION.")
    parser.add_argument(
        "--target",
        default=os.environ.get("EXTERNAL_TRANSLATION_SMOKE_TARGET", "fr"),
        help="Target language for the smoke translation.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("EXTERNAL_TRANSLATION_TIMEOUT_SECONDS", "30")),
        help="HTTP timeout in seconds.")
    parser.add_argument(
        "--readiness-path",
        default=os.environ.get("EXTERNAL_TRANSLATION_READINESS_PATH", "/health"),
        help="Readiness endpoint path to probe before dispatch.")
    args = parser.parse_args(argv)

    readiness_config = create_pipeline_config(
        env={
            "EXTERNAL_TRANSLATION_PREFER": "true",
            "EXTERNAL_TRANSLATION_URL": args.url,
            "EXTERNAL_TRANSLATION_TIMEOUT_SECONDS": str(args.timeout),
            "EXTERNAL_TRANSLATION_READINESS_PATH": args.readiness_path,
        }
    )
    readiness = external_translation_readiness(
        readiness_config,
        source_language="en",
        target_language=args.target)
    if not readiness.ready:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "url": args.url,
                    "readiness": readiness.status,
                    "message": readiness.message,
                },
                sort_keys=True)
        )
        return 2

    client = TranslationServiceClient(args.url, timeout=args.timeout)
    engines = client.list_engines()
    bundle = _smoke_document_bundle()
    translated = client.translate_bundle(
        bundle,
        target_language=args.target,
        provider_id=args.provider)
    first_span = translated["translated_spans"][0]
    print(
        json.dumps(
            {
                "status": "ok",
                "url": args.url,
                "readiness": readiness.status,
                "provider_id": translated["engine_provider"]["id"],
                "engine_count": len(engines),
                "target_language": translated["target_language"],
                "translated_text": first_span["translated_text"],
                "source_bundle_sha256": translated["source_bundle_sha256"],
            },
            sort_keys=True)
    )
    return 0


def _smoke_document_bundle() -> dict:
    return build_document_bundle(
        document_id="edc-smoke",
        source_file_name="edc-smoke.pdf",
        spans=[
            {
                "span_id": "smoke-1",
                "page_number": 1,
                "text": "Hello.",
                "bbox": [0.0, 0.0, 100.0, 12.0],
                "language": "en",
            }
        ],
        language_metadata={
            "primary_language": "en",
            "detected_languages": ["en"],
            "source": "smoke",
        },
        ocr_engine_metadata={
            "engine_id": "ocr_local.smoke",
            "engine_version": "phase3",
        },
        custody_chain_head="smoke",
        tenant_policy_hash="smoke")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
