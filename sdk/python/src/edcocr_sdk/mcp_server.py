"""Dependency-light MCP-style tool server for EDCOCR / EDCOCR.

The server exposes OCR operator tools over newline-delimited JSON-RPC so it can
be wrapped by MCP adapters without duplicating OCR client logic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TextIO

from edcocr_sdk.client import EDCOCRClient
from edcocr_sdk.exceptions import OCRLocalError


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


def _string_property(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="ocr_submit_document",
        description="Submit one document to EDCOCR.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": _string_property("Local file path to upload."),
                "source_path": _string_property("Server-side source path."),
                "priority": _string_property("Job priority: low, normal, or urgent."),
            },
        },
    ),
    ToolDefinition(
        name="ocr_submit_batch",
        description="Submit multiple documents to EDCOCR as a batch.",
        input_schema={
            "type": "object",
            "properties": {
                "file_paths": {"type": "array", "items": {"type": "string"}},
                "source_paths": {"type": "array", "items": {"type": "string"}},
                "priority": _string_property("Batch priority: low, normal, or urgent."),
            },
        },
    ),
    ToolDefinition(
        name="ocr_get_job_status",
        description="Get OCR job status.",
        input_schema={
            "type": "object",
            "properties": {"job_id": _string_property("OCR job id.")},
            "required": ["job_id"],
        },
    ),
    ToolDefinition(
        name="ocr_get_document_bundle",
        description="Retrieve a job's DocumentBundle v1.",
        input_schema={
            "type": "object",
            "properties": {"job_id": _string_property("OCR job id.")},
            "required": ["job_id"],
        },
    ),
    ToolDefinition(
        name="ocr_get_evidence_bundle",
        description="Retrieve a job's OCR evidence bundle.",
        input_schema={
            "type": "object",
            "properties": {"job_id": _string_property("OCR job id.")},
            "required": ["job_id"],
        },
    ),
    ToolDefinition(
        name="ocr_list_outputs",
        description="List OCR output artifacts for a job.",
        input_schema={
            "type": "object",
            "properties": {"job_id": _string_property("OCR job id.")},
            "required": ["job_id"],
        },
    ),
    ToolDefinition(
        name="ocr_validate_custody",
        description="Validate OCR custody evidence for a job.",
        input_schema={
            "type": "object",
            "properties": {"job_id": _string_property("OCR job id.")},
            "required": ["job_id"],
        },
    ),
)


class OCRToolServer:
    """Small tool dispatcher backed by :class:`EDCOCRClient`."""

    def __init__(self, client: EDCOCRClient) -> None:
        self.client = client
        self._handlers: dict[str, ToolHandler] = {
            "ocr_submit_document": self.ocr_submit_document,
            "ocr_submit_batch": self.ocr_submit_batch,
            "ocr_get_job_status": self.ocr_get_job_status,
            "ocr_get_document_bundle": self.ocr_get_document_bundle,
            "ocr_get_evidence_bundle": self.ocr_get_evidence_bundle,
            "ocr_list_outputs": self.ocr_list_outputs,
            "ocr_validate_custody": self.ocr_validate_custody,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP-compatible tool metadata."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in TOOL_DEFINITIONS
        ]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch a tool call by name."""
        if name not in self._handlers:
            raise ValueError(f"Unknown OCR tool: {name}")
        return self._handlers[name](arguments or {})

    def ocr_submit_document(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self.client.submit_job(
            file_path=arguments.get("file_path"),
            source_path=arguments.get("source_path"),
            priority=arguments.get("priority"),
        )
        return result.model_dump(mode="json", by_alias=True)

    def ocr_submit_batch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.client.submit_batch(
            file_paths=arguments.get("file_paths"),
            source_paths=arguments.get("source_paths"),
            priority=arguments.get("priority", "normal"),
        )

    def ocr_get_job_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        job = self.client.get_job(_required(arguments, "job_id"))
        return job.model_dump(mode="json")

    def ocr_get_document_bundle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.client.get_document_bundle(_required(arguments, "job_id"))

    def ocr_get_evidence_bundle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.client.get_evidence_bundle(_required(arguments, "job_id"))

    def ocr_list_outputs(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.client.list_outputs(_required(arguments, "job_id"))

    def ocr_validate_custody(self, arguments: dict[str, Any]) -> dict[str, Any]:
        evidence = self.client.get_evidence_bundle(_required(arguments, "job_id"))
        custody = evidence.get("custody", {})
        return {
            "job_id": arguments["job_id"],
            "valid": bool(custody.get("available") and custody.get("valid")),
            "custody": custody,
        }


def _required(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required argument: {key}")
    return value


def _success(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_jsonrpc(server: OCRToolServer, request: dict[str, Any]) -> dict[str, Any]:
    """Handle one JSON-RPC request."""
    request_id = request.get("id")
    method = request.get("method")
    try:
        if method == "initialize":
            return _success(request_id, {"serverInfo": {"name": "edc-ocr-tools"}})
        if method == "tools/list":
            return _success(request_id, {"tools": server.list_tools()})
        if method == "tools/call":
            params = request.get("params") or {}
            return _success(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                server.call_tool(
                                    str(params.get("name", "")),
                                    params.get("arguments") or {},
                                ),
                                sort_keys=True,
                            ),
                        }
                    ]
                },
            )
        return _error(request_id, -32601, f"Unsupported method: {method}")
    except (OCRLocalError, ValueError, FileNotFoundError) as exc:
        return _error(request_id, -32000, str(exc))


def serve_json_lines(
    server: OCRToolServer,
    input_stream: Iterable[str],
    output_stream: TextIO,
) -> int:
    """Serve newline-delimited JSON-RPC requests."""
    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = _error(None, -32700, f"Parse error: {exc}")
        else:
            response = handle_jsonrpc(server, request)
        output_stream.write(json.dumps(response, sort_keys=True) + "\n")
        output_stream.flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EDCOCR MCP-style tool server.")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("EDCOCR_API_URL", "http://localhost:8000"),
        help="OCR API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EDCOCR_API_KEY", ""),
        help="OCR API key.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with EDCOCRClient(args.api_url, api_key=args.api_key, timeout=args.timeout) as client:
        server = OCRToolServer(client)
        return serve_json_lines(server, sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
