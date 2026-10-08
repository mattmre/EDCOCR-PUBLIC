#!/usr/bin/env python3
"""Ping the exact Celery worker nodename used by the container process."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path


def resolve_nodename(argv: list[str], hostname: str) -> str:
    override = os.environ.get("CELERY_WORKER_HEALTHCHECK_NODE", "").strip()
    if override:
        template = override
    else:
        template = ""
        for index, arg in enumerate(argv):
            if arg in {"--hostname", "-n"} and index + 1 < len(argv):
                template = argv[index + 1]
                break
            if arg.startswith("--hostname="):
                template = arg.split("=", 1)[1]
                break
        if not template:
            raise ValueError("Celery worker command has no --hostname/-n argument")
    return template.replace("%h", hostname)


def resolve_celery_app(argv: list[str]) -> str:
    """Use the same Celery application as PID 1 instead of guessing it."""
    for index, arg in enumerate(argv):
        if arg in {"-A", "--app"} and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--app="):
            return arg.split("=", 1)[1]
    return "coordinator.coordinator"


def main() -> int:
    try:
        argv = Path("/proc/1/cmdline").read_bytes().decode().split("\0")
        nodename = resolve_nodename(argv, socket.gethostname())
        celery_app = resolve_celery_app(argv)
        subprocess.run(
            [
                "celery",
                "-A",
                celery_app,
                "inspect",
                "ping",
                "--timeout=10",
                "-d",
                nodename,
            ],
            check=True,
            timeout=12,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"worker healthcheck failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
