"""Rendered SOLO deployment boundaries for fresh CPU and GPU installs."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("cpu", [False, True])
def test_solo_build_and_input_ownership(cpu):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose is required for deployment rendering")
    version = subprocess.run(["docker", "compose", "version"], capture_output=True)
    if version.returncode:
        pytest.skip("Docker Compose plugin is not installed")
    command = ["docker", "compose", "-f", "docker-compose.solo.yml"]
    if cpu:
        command += ["-f", "docker-compose.solo.cpu.yml"]
    env = {key: value for key, value in os.environ.items() if not key.startswith("SOLO_")}
    env["SOLO_IMAGE"] = "owned-solo-fixture:custom-tag"
    result = subprocess.run(
        command + ["--profile", "api", "config", "--format", "json"],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True,
        text=True, check=True,
    )
    services = json.loads(result.stdout)["services"]
    worker, api = services["ocr-solo"], services["ocr-api"]
    assert worker["image"] == env["SOLO_IMAGE"]
    assert api["build"]["additional_contexts"]["solo-runtime"] == "service:ocr-solo"
    assert api["build"]["args"]["BASE_IMAGE"] == "solo-runtime"
    watched = next(v["source"] for v in worker["volumes"] if v["target"] == "/app/ocr_source")
    uploaded = next(v["source"] for v in api["volumes"] if v["target"] == "/app/api_source")
    assert Path(watched) not in (Path(uploaded), *Path(uploaded).parents)
    devices = worker["deploy"]["resources"].get("reservations", {}).get("devices", [])
    if cpu:
        assert devices == []
        assert worker["build"]["dockerfile"] == "Dockerfile"
        assert worker["environment"]["NUM_WORKERS"] == "1"
    else:
        assert devices[0]["driver"] == "nvidia"
        assert worker["build"]["dockerfile"] == "coordinator/Dockerfile.worker.gpu"
    assert api["environment"]["MAX_CONCURRENT_JOBS"] == "1"
