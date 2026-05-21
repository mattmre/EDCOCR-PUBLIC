"""Tests for Docker-backed plugin sandbox proof command construction."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from scripts import prove_plugin_sandbox_docker as proof


def test_build_command_disables_network_and_sets_limits():
    args = proof.parse_args([])
    cmd = proof.build_command(args)

    assert "--network" in cmd
    assert "none" in cmd
    assert "--memory" in cmd
    assert "--cpus" in cmd
    assert "--pids-limit" in cmd
    assert "no-new-privileges" in cmd


def test_main_requires_network_blocked(monkeypatch):
    proc = MagicMock(
        returncode=0,
        stdout=json.dumps({"network_blocked": True, "allocated_bytes": 1}),
        stderr="",
    )
    monkeypatch.setattr(proof.subprocess, "run", MagicMock(return_value=proc))

    assert proof.main([]) == 0


def test_main_fails_when_network_reachable(monkeypatch):
    proc = MagicMock(
        returncode=0,
        stdout=json.dumps({"network_blocked": False, "allocated_bytes": 1}),
        stderr="",
    )
    monkeypatch.setattr(proof.subprocess, "run", MagicMock(return_value=proc))

    assert proof.main([]) == 1
