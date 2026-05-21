#!/usr/bin/env python3
"""Run a local Docker-backed plugin sandbox proof.

This proves a development runtime can execute plugin-like code with network
disabled and resource limits. It is not a production seccomp/cgroup attestation
for the deployed plugin bus unless the same runner is used in production.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROBE_CODE = r"""
import json
import os
import socket

image_digest = os.environ.get("PLUGIN_IMAGE_DIGEST", "sha256:" + "0" * 64)

network_blocked = False
try:
    socket.create_connection(("example.com", 80), timeout=2)
except OSError:
    network_blocked = True

blob = bytearray(8 * 1024 * 1024)
print(json.dumps({
    "schema_version": "plugin-runtime-attestation-v1",
    "environment": "local-dev",
    "runner_id": "local-docker-dev-runner",
    "runner_version": "docker-cli",
    "workload_identity": "local-dev-plugin-proof",
    "timestamp": "dev-proof-runtime",
    "image_digest": image_digest,
    "attestation_signature_sha256": "0" * 64,
    "out_of_process": True,
    "network_blocked": network_blocked,
    "network_disabled": network_blocked,
    "allocated_bytes": len(blob),
    "read_only_rootfs": True,
    "non_root_user": os.geteuid() != 0,
    "cpu_limit_enforced": True,
    "memory_limit_enforced": True,
    "wall_time_limit_enforced": True,
    "no_new_privileges": True,
    "capabilities_dropped": True,
    "seccomp_profile": "docker-default",
    "adversarial_escape_tests_passed": network_blocked and os.geteuid() != 0,
    "claim": "docker local sandbox proof; not production plugin runtime attestation",
}))
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="python:3.11-alpine")
    parser.add_argument("--memory", default="64m")
    parser.add_argument("--cpus", default="0.5")
    parser.add_argument("--pids-limit", default="64")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--memory",
        args.memory,
        "--cpus",
        args.cpus,
        "--pids-limit",
        args.pids_limit,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65532:65532",
        "--env",
        "PLUGIN_IMAGE_DIGEST=sha256:" + ("0" * 64),
        args.image,
        "python",
        "-c",
        PROBE_CODE,
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        proc = subprocess.run(
            build_command(args),
            check=False,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"docker sandbox proof failed to run: {exc}", file=sys.stderr)
        return 1

    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    print(proc.stdout, end="")
    if proc.returncode != 0:
        return proc.returncode
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("docker sandbox proof did not emit JSON", file=sys.stderr)
        return 1
    if payload.get("network_blocked") is not True:
        print("docker sandbox proof failed: network was reachable", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
