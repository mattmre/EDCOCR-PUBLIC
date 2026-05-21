"""Tests for Python SDK output and schema methods.

All HTTP interactions are mocked -- no real network calls are made.
"""

import json
from unittest.mock import MagicMock, patch

from sdk.python_client import OcrClient

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mock_response(status_code=200, json_data=None, text="", content=b"",
                   content_type="application/json"):
    """Build a mock ``requests.Response``."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.json.return_value = json_data or {}
    resp.text = text or (str(json_data) if json_data else "")
    resp.content = content
    return resp


# ------------------------------------------------------------------
# OcrClient.get_outputs
# ------------------------------------------------------------------


class TestGetOutputs:
    """Test the get_outputs method."""

    @patch("sdk.python_client._get_requests")
    def test_get_outputs_returns_manifest(self, mock_get_req):
        manifest = {
            "job_id": "job_abc123def456",
            "artifacts": [
                {
                    "output_type": "ocr_text",
                    "filename": "doc.txt",
                    "relative_path": "TEXT/doc.txt",
                    "size_bytes": 1024,
                    "schema_version": "1.0.0",
                },
                {
                    "output_type": "searchable_pdf",
                    "filename": "doc.pdf",
                    "relative_path": "PDF/doc.pdf",
                    "size_bytes": 50000,
                    "schema_version": "1.0.0",
                },
            ],
            "schema_versions": {
                "ocr_text": "1.0.0",
                "searchable_pdf": "1.0.0",
            },
        }
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(200, json_data=manifest)
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h", api_key="k")
        result = c.get_outputs("job_abc123def456")

        assert result["job_id"] == "job_abc123def456"
        assert len(result["artifacts"]) == 2
        assert result["artifacts"][0]["output_type"] == "ocr_text"
        assert result["schema_versions"]["searchable_pdf"] == "1.0.0"

    @patch("sdk.python_client._get_requests")
    def test_get_outputs_makes_correct_request(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(200, json_data={"job_id": "j1", "artifacts": []})
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://host:8000")
        c.get_outputs("j1")

        call_args = mock_session.request.call_args
        assert call_args[0][0] == "GET"
        assert "/api/v1/jobs/j1/outputs" in call_args[0][1]


# ------------------------------------------------------------------
# OcrClient.get_output
# ------------------------------------------------------------------


class TestGetOutput:
    """Test the get_output method for binary downloads."""

    @patch("sdk.python_client._get_requests")
    def test_get_output_returns_bytes(self, mock_get_req):
        payload = b"plain text content here"
        mock_resp = _mock_response(200, content=payload)
        mock_get_req.return_value.get.return_value = mock_resp

        c = OcrClient("http://h", api_key="k")
        data = c.get_output("job_abc123def456", "ocr_text")
        assert data == payload

    @patch("sdk.python_client._get_requests")
    def test_get_output_calls_correct_url(self, mock_get_req):
        mock_resp = _mock_response(200, content=b"x")
        mock_get_req.return_value.get.return_value = mock_resp

        c = OcrClient("http://host:8000", api_key="secret")
        c.get_output("j1", "ner")

        call_args = mock_get_req.return_value.get.call_args
        url = call_args[0][0]
        assert url == "http://host:8000/api/v1/jobs/j1/outputs/ner"

    @patch("sdk.python_client._get_requests")
    def test_get_output_passes_api_key_header(self, mock_get_req):
        mock_resp = _mock_response(200, content=b"x")
        mock_get_req.return_value.get.return_value = mock_resp

        c = OcrClient("http://h", api_key="my-key")
        c.get_output("j1", "searchable_pdf")

        call_kwargs = mock_get_req.return_value.get.call_args[1]
        assert call_kwargs["headers"]["X-API-Key"] == "my-key"

    @patch("sdk.python_client._get_requests")
    def test_get_output_no_api_key(self, mock_get_req):
        mock_resp = _mock_response(200, content=b"x")
        mock_get_req.return_value.get.return_value = mock_resp

        c = OcrClient("http://h")
        c.get_output("j1", "ocr_text")

        call_kwargs = mock_get_req.return_value.get.call_args[1]
        assert "X-API-Key" not in call_kwargs["headers"]


# ------------------------------------------------------------------
# OcrClient.get_output_json
# ------------------------------------------------------------------


class TestGetOutputJson:
    """Test the get_output_json method for JSON sidecar downloads."""

    @patch("sdk.python_client._get_requests")
    def test_get_output_json_parses_response(self, mock_get_req):
        json_payload = {"entities": [{"type": "PERSON", "text": "John"}]}
        mock_resp = _mock_response(
            200, content=json.dumps(json_payload).encode("utf-8")
        )
        mock_get_req.return_value.get.return_value = mock_resp

        c = OcrClient("http://h", api_key="k")
        result = c.get_output_json("j1", "ner")

        assert isinstance(result, dict)
        assert result["entities"][0]["type"] == "PERSON"

    @patch("sdk.python_client._get_requests")
    def test_get_output_json_with_nested_structure(self, mock_get_req):
        json_payload = {
            "pages": [
                {
                    "page_number": 1,
                    "layout_regions": [
                        {"type": "text", "bbox": [0, 0, 100, 100]},
                    ],
                },
            ],
        }
        mock_resp = _mock_response(
            200, content=json.dumps(json_payload).encode("utf-8")
        )
        mock_get_req.return_value.get.return_value = mock_resp

        c = OcrClient("http://h")
        result = c.get_output_json("j1", "structure")

        assert result["pages"][0]["page_number"] == 1
        assert len(result["pages"][0]["layout_regions"]) == 1


# ------------------------------------------------------------------
# OcrClient.list_schemas
# ------------------------------------------------------------------


class TestListSchemas:
    """Test the list_schemas method."""

    @patch("sdk.python_client._get_requests")
    def test_list_schemas_returns_list(self, mock_get_req):
        schemas_resp = {
            "schemas": [
                {"output_type": "ocr_text", "schema_version": "1.0.0"},
                {"output_type": "ner", "schema_version": "1.0.0"},
                {"output_type": "structure", "schema_version": "1.0.0"},
            ],
        }
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(200, json_data=schemas_resp)
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        result = c.list_schemas()

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["output_type"] == "ocr_text"
        assert result[1]["schema_version"] == "1.0.0"

    @patch("sdk.python_client._get_requests")
    def test_list_schemas_empty(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(200, json_data={"schemas": []})
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        result = c.list_schemas()
        assert result == []

    @patch("sdk.python_client._get_requests")
    def test_list_schemas_missing_key_returns_empty(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(200, json_data={})
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        result = c.list_schemas()
        assert result == []


# ------------------------------------------------------------------
# OcrClient.get_schema
# ------------------------------------------------------------------


class TestGetSchema:
    """Test the get_schema method."""

    @patch("sdk.python_client._get_requests")
    def test_get_schema_returns_dict(self, mock_get_req):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "NER Output",
            "type": "object",
            "properties": {
                "entities": {"type": "array"},
            },
        }
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(200, json_data=schema)
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        result = c.get_schema("ner")

        assert isinstance(result, dict)
        assert result["title"] == "NER Output"
        assert "properties" in result

    @patch("sdk.python_client._get_requests")
    def test_get_schema_makes_correct_request(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(200, json_data={"type": "object"})
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://host:8000")
        c.get_schema("extraction")

        call_args = mock_session.request.call_args
        assert call_args[0][0] == "GET"
        assert "/api/v1/schemas/extraction" in call_args[0][1]
