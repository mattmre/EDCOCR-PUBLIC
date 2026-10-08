#!/usr/bin/env python3
"""Run the translation tenant-config evidence lane.

This wraps the coordinator/Django pytest invocation validated by the hunter
pass so maintenance review evidence can be reproduced from the repository root.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        str(Path("..") / "tests" / "test_translation_tenant_config.py"),
        "-c",
        "pytest.ini",
        "-rs",
        "--tb=short",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional file that receives combined stdout/stderr for BHS evidence",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    coordinator_dir = _PROJECT_ROOT / "coordinator"
    command = build_command()
    print(f"running in {coordinator_dir}: {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=coordinator_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.stdout, encoding="utf-8", newline="\n")
        print(f"wrote tenant-config harness output: {args.output}")
    else:
        print(result.stdout, end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
