"""Tests for the translation tenant-config harness wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import run_translation_tenant_config_harness as harness


def test_build_command_matches_verified_coordinator_harness():
    assert harness.build_command() == [
        sys.executable,
        "-m",
        "pytest",
        str(Path("..") / "tests" / "test_translation_tenant_config.py"),
        "-c",
        "pytest.ini",
        "-rs",
        "--tb=short",
    ]


def test_harness_can_capture_evidence_output(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, cwd, text, stdout, stderr, check):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "text": text,
                "stdout": stdout,
                "stderr": stderr,
                "check": check,
            }
        )
        return SimpleNamespace(returncode=0, stdout="39 passed\n")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    output = tmp_path / "tenant-config.txt"

    assert harness.main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == "39 passed\n"
    assert calls[0]["cwd"] == Path(harness._PROJECT_ROOT) / "coordinator"
    assert calls[0]["command"] == harness.build_command()
