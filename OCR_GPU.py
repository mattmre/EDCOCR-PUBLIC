"""Deprecated sync OCR entrypoint.

This shim is intentionally side-effect free on import.
The legacy synchronous implementation was moved to:
  legacy/OCR_GPU_sync_legacy.py
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

LEGACY_SCRIPT = Path(__file__).resolve().parent / "legacy" / "OCR_GPU_sync_legacy.py"
DEPRECATION_MESSAGE = (
    "OCR_GPU.py (sync pipeline) is deprecated and no longer the production path.\n"
    "Use ocr_gpu_async.py for supported operation.\n"
    "If you intentionally need the legacy sync implementation, rerun with --run-legacy-sync."
)


def _run_legacy_sync() -> None:
    if not LEGACY_SCRIPT.exists():
        raise FileNotFoundError(
            f"Legacy sync pipeline not found at {LEGACY_SCRIPT}. "
            "Restore legacy/OCR_GPU_sync_legacy.py or use ocr_gpu_async.py."
        )
    runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deprecated sync OCR pipeline shim.",
    )
    parser.add_argument(
        "--run-legacy-sync",
        action="store_true",
        help="Run deprecated legacy sync pipeline from legacy/OCR_GPU_sync_legacy.py.",
    )
    args = parser.parse_args(argv)

    if not args.run_legacy_sync:
        print(DEPRECATION_MESSAGE, file=sys.stderr)
        return 1

    print("Running deprecated sync pipeline from legacy/OCR_GPU_sync_legacy.py", file=sys.stderr)
    _run_legacy_sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

