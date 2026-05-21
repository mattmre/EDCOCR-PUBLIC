"""Tests for the edc-ocr SDK command-line entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SDK_SRC = str(Path(__file__).resolve().parent.parent / "sdk" / "python" / "src")
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)

from edcocr_sdk import cli  # noqa: E402


class _ModelResult:
    def __init__(self, payload):
        self._payload = payload
        self.job_id = payload["job_id"]

    def model_dump(self, **kwargs):
        return dict(self._payload)


class _FakeClient:
    instances = []

    def __init__(self, base_url, api_key="", timeout=30.0):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.calls = []
        _FakeClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def submit_job(self, **kwargs):
        self.calls.append(("submit_job", kwargs))
        return _ModelResult(
            {
                "job_id": "job_abc123def456",
                "status": "submitted",
                "created_at": "2026-05-14T00:00:00",
                "priority": kwargs.get("priority", "normal"),
                "source_file": str(kwargs.get("file_path", "")),
            }
        )

    def submit_batch(self, **kwargs):
        self.calls.append(("submit_batch", kwargs))
        return {
            "batch_id": "batch_abc123def456",
            "status": "submitted",
            "total_jobs": len(kwargs.get("file_paths", [])),
        }

    def get_job(self, job_id):
        self.calls.append(("get_job", job_id))
        return _ModelResult({"job_id": job_id, "status": "completed"})

    def export_document_bundle(self, job_id, output_path):
        self.calls.append(("export_document_bundle", job_id, output_path))
        payload = {"schema_version": "DocumentBundle.v1", "document": {"document_id": job_id}}
        Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def get_evidence_bundle(self, job_id):
        self.calls.append(("get_evidence_bundle", job_id))
        return {
            "job_id": job_id,
            "custody": {"available": True, "valid": True, "chain_head": "abc123"},
        }


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    _FakeClient.instances = []
    monkeypatch.setattr(cli, "EDCOCRClient", _FakeClient)


def _json_from_stdout(capsys):
    return json.loads(capsys.readouterr().out)


def test_submit_writes_metadata(tmp_path, capsys):
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF")
    out_dir = tmp_path / "out"

    assert cli.main(["--api-url", "http://api", "submit", str(pdf), "--output", str(out_dir)]) == 0

    payload = _json_from_stdout(capsys)
    assert payload["job_id"] == "job_abc123def456"
    assert (out_dir / "job_abc123def456.submit.json").exists()
    assert _FakeClient.instances[0].base_url == "http://api"


def test_batch_submits_directory_files(tmp_path, capsys):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "a.pdf").write_bytes(b"a")
    (input_dir / "b.pdf").write_bytes(b"b")

    assert cli.main(["batch", str(input_dir), "--tenant", "tenant-a"]) == 0

    payload = _json_from_stdout(capsys)
    assert payload["batch_id"] == "batch_abc123def456"
    assert payload["total_jobs"] == 2
    assert payload["client_context"]["tenant"] == "tenant-a"


def test_status_outputs_job_status(capsys):
    assert cli.main(["status", "job_abc123def456"]) == 0
    assert _json_from_stdout(capsys)["status"] == "completed"


def test_export_bundle_writes_document_bundle(tmp_path, capsys):
    out = tmp_path / "bundle.json"
    assert cli.main(["export-bundle", "job_abc123def456", "--out", str(out)]) == 0
    payload = _json_from_stdout(capsys)
    assert payload["schema_version"] == "DocumentBundle.v1"
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == "DocumentBundle.v1"


def test_verify_custody_returns_zero_when_valid(capsys):
    assert cli.main(["verify-custody", "job_abc123def456"]) == 0
    payload = _json_from_stdout(capsys)
    assert payload["custody_available"] is True
    assert payload["custody_valid"] is True
