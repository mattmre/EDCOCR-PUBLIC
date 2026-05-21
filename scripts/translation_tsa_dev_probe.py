#!/usr/bin/env python3
"""Run one low-volume dev/live RFC3161 TSA probe.

This script is evidence that a configured TSA endpoint responds and that the
TSR bytes are stored and hashed. It is not production legal assurance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_local.translation.tsa import HTTPRFC3161TSAClient, anchor_certification_payload  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("TRANSLATION_TSA_URL", ""))
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=ROOT / "out" / "translation_tsa_dev_probe",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.url:
        print("missing TSA URL: pass --url or set TRANSLATION_TSA_URL", file=sys.stderr)
        return 1

    payload = {
        "proof": "translation-swarm-dev-tsa-probe",
        "scope": "dev/live proof only; not paid production legal assurance",
    }
    anchor = anchor_certification_payload(
        payload,
        tsa_client=HTTPRFC3161TSAClient(args.url, timeout=args.timeout),
        tsa_store_dir=args.store_dir,
    )
    print(json.dumps(anchor.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
