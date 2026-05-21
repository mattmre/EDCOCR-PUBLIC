"""Command-line entry point for EDCOCR / EDCOCR workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from edcocr_sdk.client import EDCOCRClient
from edcocr_sdk.exceptions import OCRLocalError


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-url",
        default=os.environ.get("EDCOCR_API_URL", "http://localhost:8000"),
        help="OCR API base URL. Defaults to EDCOCR_API_URL or http://localhost:8000.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EDCOCR_API_KEY", ""),
        help="OCR API key. Defaults to EDCOCR_API_KEY.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _client(args: argparse.Namespace) -> EDCOCRClient:
    return EDCOCRClient(args.api_url, api_key=args.api_key, timeout=args.timeout)


def _handle_submit(args: argparse.Namespace) -> int:
    with _client(args) as client:
        result = client.submit_job(
            file_path=args.input,
            priority=args.priority,
            enable_docintel=args.enable_docintel,
            docintel_mode=args.docintel_mode,
        )
        payload = result.model_dump(mode="json", by_alias=True)
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{result.job_id}.submit.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        _print_json(payload)
    return 0


def _handle_batch(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Batch input directory not found: {input_dir}")

    file_paths = [p for p in sorted(input_dir.iterdir()) if p.is_file()]
    if not file_paths:
        raise ValueError(f"No files found in batch input directory: {input_dir}")

    with _client(args) as client:
        payload = client.submit_batch(
            file_paths=file_paths,
            priority=args.priority,
            enable_docintel=args.enable_docintel,
            docintel_mode=args.docintel_mode,
        )
        if args.tenant:
            payload.setdefault("client_context", {})["tenant"] = args.tenant
        _print_json(payload)
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    with _client(args) as client:
        job = client.get_job(args.job_id)
        _print_json(job.model_dump(mode="json"))
    return 0


def _handle_export_bundle(args: argparse.Namespace) -> int:
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _client(args) as client:
        bundle = client.export_document_bundle(args.job_id, output_path)
        _print_json(
            {
                "job_id": args.job_id,
                "output_path": str(output_path),
                "schema_version": bundle.get("schema_version"),
            }
        )
    return 0


def _handle_verify_custody(args: argparse.Namespace) -> int:
    with _client(args) as client:
        evidence = client.get_evidence_bundle(args.job_id)
        custody = evidence.get("custody", {})
        valid = bool(custody.get("available") and custody.get("valid"))
        _print_json(
            {
                "job_id": args.job_id,
                "custody_available": bool(custody.get("available")),
                "custody_valid": bool(custody.get("valid")),
                "chain_head": custody.get("chain_head"),
            }
        )
    return 0 if valid else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edc-ocr",
        description="EDCOCR operator CLI for OCR job, batch, bundle, and custody workflows.",
    )
    _add_common_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Submit one document for OCR.")
    submit.add_argument("input", help="Document path to upload.")
    submit.add_argument("--output", help="Directory for submit metadata JSON.")
    submit.add_argument("--priority", choices=("low", "normal", "urgent"), default="normal")
    submit.add_argument("--enable-docintel", action="store_true")
    submit.add_argument("--docintel-mode", choices=("layout_only", "tables_only", "full"), default="full")
    submit.set_defaults(func=_handle_submit)

    batch = subparsers.add_parser("batch", help="Submit all files in a directory as a batch.")
    batch.add_argument("input_dir", help="Directory containing documents to upload.")
    batch.add_argument("--tenant", help="Tenant identifier for local operator context.")
    batch.add_argument("--priority", choices=("low", "normal", "urgent"), default="normal")
    batch.add_argument("--enable-docintel", action="store_true")
    batch.add_argument("--docintel-mode", choices=("layout_only", "tables_only", "full"), default="full")
    batch.set_defaults(func=_handle_batch)

    status = subparsers.add_parser("status", help="Get OCR job status.")
    status.add_argument("job_id")
    status.set_defaults(func=_handle_status)

    export_bundle = subparsers.add_parser(
        "export-bundle",
        help="Export a completed job's DocumentBundle v1 JSON.",
    )
    export_bundle.add_argument("job_id")
    export_bundle.add_argument("--out", required=True, help="Output JSON path.")
    export_bundle.set_defaults(func=_handle_export_bundle)

    verify = subparsers.add_parser(
        "verify-custody",
        help="Verify OCR custody evidence for a job.",
    )
    verify.add_argument("job_id")
    verify.set_defaults(func=_handle_verify_custody)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, OCRLocalError) as exc:
        print(f"edc-ocr: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
