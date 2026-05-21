"""Tests for the EDCOCR MCP-style tool server."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_SDK_SRC = str(Path(__file__).resolve().parent.parent / "sdk" / "python" / "src")
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)

from edcocr_sdk.mcp_server import OCRToolServer, handle_jsonrpc, serve_json_lines  # noqa: E402


class _ModelResult:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, **kwargs):
        return dict(self._payload)


class _FakeClient:
    def submit_job(self, **kwargs):
        return _ModelResult({"job_id": "job_abc123def456", "status": "submitted"})

    def submit_batch(self, **kwargs):
        return {"batch_id": "batch_abc123def456", "status": "submitted"}

    def get_job(self, job_id):
        return _ModelResult({"job_id": job_id, "status": "completed"})

    def get_document_bundle(self, job_id):
        return {"schema_version": "DocumentBundle.v1", "document": {"document_id": job_id}}

    def get_evidence_bundle(self, job_id):
        return {
            "job_id": job_id,
            "custody": {"available": True, "valid": True, "chain_head": "abc123"},
        }

    def list_outputs(self, job_id):
        return {"job_id": job_id, "artifacts": [{"output_type": "ocr_text"}]}


def test_list_tools_contains_required_ocr_tools():
    server = OCRToolServer(_FakeClient())
    names = {tool["name"] for tool in server.list_tools()}
    assert names == {
        "ocr_submit_document",
        "ocr_submit_batch",
        "ocr_get_job_status",
        "ocr_get_document_bundle",
        "ocr_get_evidence_bundle",
        "ocr_list_outputs",
        "ocr_validate_custody",
    }


def test_dispatches_job_status_tool():
    server = OCRToolServer(_FakeClient())
    result = server.call_tool("ocr_get_job_status", {"job_id": "job_abc123def456"})
    assert result == {"job_id": "job_abc123def456", "status": "completed"}


def test_validate_custody_returns_valid_flag():
    server = OCRToolServer(_FakeClient())
    result = server.call_tool("ocr_validate_custody", {"job_id": "job_abc123def456"})
    assert result["valid"] is True
    assert result["custody"]["chain_head"] == "abc123"


def test_missing_required_argument_returns_jsonrpc_error():
    server = OCRToolServer(_FakeClient())
    response = handle_jsonrpc(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "ocr_get_document_bundle", "arguments": {}},
        },
    )
    assert response["error"]["code"] == -32000
    assert "job_id" in response["error"]["message"]


def test_tools_call_returns_mcp_content_text():
    server = OCRToolServer(_FakeClient())
    response = handle_jsonrpc(
        server,
        {
            "jsonrpc": "2.0",
            "id": "abc",
            "method": "tools/call",
            "params": {
                "name": "ocr_get_document_bundle",
                "arguments": {"job_id": "job_abc123def456"},
            },
        },
    )
    text = response["result"]["content"][0]["text"]
    assert json.loads(text)["schema_version"] == "DocumentBundle.v1"


def test_json_lines_server_handles_initialize_and_tool_list():
    server = OCRToolServer(_FakeClient())
    input_lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]
    output = io.StringIO()
    assert serve_json_lines(server, input_lines, output) == 0
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]["result"]["serverInfo"]["name"] == "edc-ocr-tools"
    assert responses[1]["result"]["tools"][0]["name"].startswith("ocr_")
