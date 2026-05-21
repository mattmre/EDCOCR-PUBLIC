"""Command construction tests for Docker plugin sandbox proof."""

from __future__ import annotations

from argparse import Namespace

from scripts import prove_plugin_sandbox_docker as proof


def test_docker_command_uses_runtime_isolation_flags():
    cmd = proof.build_command(
        Namespace(
            image="python:3.11-alpine",
            memory="64m",
            cpus="0.5",
            pids_limit="64",
            timeout=60,
        )
    )

    assert "--network" in cmd
    assert "none" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop" in cmd
    assert "ALL" in cmd
    assert "--security-opt" in cmd
    assert "no-new-privileges" in cmd
    assert "--user" in cmd
    assert "65532:65532" in cmd
