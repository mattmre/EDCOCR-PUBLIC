"""Process-level seam test against the standalone EDC_TRANSLATION service."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


def test_ocr_external_facade_translates_against_edc_process():
    edc_repo = _edc_translation_repo_or_skip()
    _uvicorn_or_skip(edc_repo)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _start_edc_translation(edc_repo, port)

    try:
        _wait_for_health(proc, base_url)

        from ocr_local.translation.api import translate_document
        from pipeline_config import create_pipeline_config

        cfg = create_pipeline_config(
            env={
                "ENABLE_TRANSLATION": "true",
                "TRANSLATION_TARGET_LANGUAGES": "fr",
                "EDC_TRANSLATION_PREFER_EXTERNAL": "true",
                "EDC_TRANSLATION_URL": base_url,
                "EDC_TRANSLATION_PROVIDER_ID": "deterministic_ci",
                "EDC_TRANSLATION_TIMEOUT_SECONDS": "5",
            }
        )

        result = translate_document(
            doc_path="/tmp/edc-process-seam.pdf",
            target_languages=["fr"],
            tenant_id="tenant-process-e2e",
            page_data_snap={"texts": {1: "Hello."}, "detected_language": "en"},
            config=cfg,
        )

        assert len(result) == 1
        doc = result[0]
        assert doc.processing["translation_service"] == "external"
        assert doc.engine["id"] == "deterministic_ci"
        assert doc.pages[0].spans[0].target_text == "Hello. [en->fr]"
        assert doc.custody["tenant_id"] == "tenant-process-e2e"
        assert doc.stats["source_bundle_sha256"]
    finally:
        _stop_process(proc)


def _edc_translation_repo_or_skip() -> Path:
    configured = os.environ.get("EDC_TRANSLATION_REPO")
    repo = (
        Path(configured)
        if configured
        else Path(__file__).resolve().parents[2] / "EDC_TRANSLATION"
    )
    if not repo.joinpath("edc_translation", "api.py").is_file():
        pytest.skip("EDC_TRANSLATION repo is not available for process e2e")
    return repo


def _uvicorn_or_skip(repo: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import uvicorn"],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("uvicorn is not installed for EDC_TRANSLATION process e2e")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_edc_translation(repo: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(repo)
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "edc_translation.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(repo),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_health(proc: subprocess.Popen, base_url: str) -> None:
    deadline = time.time() + float(
        os.environ.get("EDC_TRANSLATION_E2E_STARTUP_TIMEOUT", "20")
    )
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(
                "EDC_TRANSLATION process exited before health check "
                f"(returncode={proc.returncode})"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    pytest.fail("EDC_TRANSLATION process did not become healthy")


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
