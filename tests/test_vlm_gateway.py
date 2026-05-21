"""Tests for VLM inference gateway client."""

from unittest import mock

import httpx
import pytest

from vlm_config import VLMConfig
from vlm_gateway import (
    VLMAuthError,
    VLMConnectionError,
    VLMGateway,
    VLMGatewayError,
    VLMServerError,
    VLMTimeoutError,
)


def _make_config(**overrides) -> VLMConfig:
    """Create a test VLMConfig with sensible defaults."""
    defaults = {
        "enabled": True,
        "endpoint_url": "http://vlm-test:8080",
        "api_key": "test-key",
        "model_name": "test-model",
        "max_context_pages": 5,
        "timeout_seconds": 10,
        "retry_attempts": 0,  # no retries for fast tests
    }
    defaults.update(overrides)
    return VLMConfig(**defaults)


class TestVLMGatewayInit:
    """Tests for VLMGateway initialization and properties."""

    def test_enabled_property_true(self):
        gw = VLMGateway(_make_config(enabled=True))
        assert gw.enabled is True

    def test_enabled_property_false_when_disabled(self):
        gw = VLMGateway(_make_config(enabled=False))
        assert gw.enabled is False

    def test_enabled_property_false_when_no_url(self):
        gw = VLMGateway(_make_config(enabled=True, endpoint_url=""))
        assert gw.enabled is False

    def test_context_manager(self):
        gw = VLMGateway(_make_config())
        with gw as g:
            assert g is gw
        # Client should be cleaned up (no error)

    def test_close_idempotent(self):
        gw = VLMGateway(_make_config())
        gw.close()
        gw.close()  # Should not raise


class TestVLMGatewayHealthCheck:
    """Tests for VLMGateway.health_check()."""

    def test_disabled_returns_false(self):
        gw = VLMGateway(_make_config(enabled=False))
        assert gw.health_check() is False

    def test_successful_health_check(self):
        gw = VLMGateway(_make_config())
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.get.return_value = mock_resp
            assert gw.health_check() is True

    def test_health_check_fallback_to_models(self):
        """When /health fails, try /v1/models."""
        gw = VLMGateway(_make_config())
        health_resp = mock.MagicMock()
        health_resp.status_code = 404
        models_resp = mock.MagicMock()
        models_resp.status_code = 200

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.get.side_effect = [health_resp, models_resp]
            assert gw.health_check() is True

    def test_health_check_connection_error(self):
        gw = VLMGateway(_make_config())
        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.get.side_effect = httpx.ConnectError("refused")
            assert gw.health_check() is False

    def test_health_check_timeout(self):
        gw = VLMGateway(_make_config())
        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.get.side_effect = httpx.ReadTimeout("timeout")
            assert gw.health_check() is False


class TestVLMGatewayAnalyzeDocument:
    """Tests for VLMGateway.analyze_document()."""

    def test_disabled_raises(self):
        gw = VLMGateway(_make_config(enabled=False))
        with pytest.raises(VLMGatewayError, match="disabled"):
            gw.analyze_document(pages=[{"page_number": 1, "text": "hello"}])

    def test_successful_analysis(self):
        gw = VLMGateway(_make_config())
        response_data = {
            "entities": [{"type": "PERSON", "text": "John Doe"}],
            "summary": "A test document.",
            "relationships": [],
            "confidence": 0.95,
            "model": "test-model",
        }
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response_data

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.return_value = mock_resp
            result = gw.analyze_document(
                pages=[{"page_number": 1, "text": "John Doe signed the contract."}],
                prompt="Extract entities.",
                document_id="doc_001",
            )

        assert result["entities"] == [{"type": "PERSON", "text": "John Doe"}]
        assert result["summary"] == "A test document."
        assert result["confidence"] == 0.95
        assert "processing_time_ms" in result

    def test_page_truncation(self):
        """Pages beyond max_context_pages are truncated."""
        gw = VLMGateway(_make_config(max_context_pages=2))
        pages = [{"page_number": i, "text": f"page {i}"} for i in range(5)]

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"entities": []}

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.return_value = mock_resp
            gw.analyze_document(pages=pages)

            # Verify only 2 pages were sent
            call_args = mock_client.return_value.request.call_args
            sent_payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert len(sent_payload["pages"]) == 2

    def test_auth_error(self):
        gw = VLMGateway(_make_config())
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.return_value = mock_resp
            with pytest.raises(VLMAuthError):
                gw.analyze_document(pages=[{"page_number": 1, "text": "test"}])

    def test_server_error(self):
        gw = VLMGateway(_make_config(retry_attempts=0))
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.return_value = mock_resp
            with pytest.raises(VLMServerError):
                gw.analyze_document(pages=[{"page_number": 1, "text": "test"}])

    def test_connection_error(self):
        gw = VLMGateway(_make_config(retry_attempts=0))
        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.side_effect = httpx.ConnectError("refused")
            with pytest.raises(VLMConnectionError):
                gw.analyze_document(pages=[{"page_number": 1, "text": "test"}])

    def test_timeout_error(self):
        gw = VLMGateway(_make_config(retry_attempts=0))
        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.side_effect = httpx.ReadTimeout("timeout")
            with pytest.raises(VLMTimeoutError):
                gw.analyze_document(pages=[{"page_number": 1, "text": "test"}])


class TestVLMGatewaySemanticSearch:
    """Tests for VLMGateway.semantic_search()."""

    def test_disabled_raises(self):
        gw = VLMGateway(_make_config(enabled=False))
        with pytest.raises(VLMGatewayError, match="disabled"):
            gw.semantic_search(query="test")

    def test_successful_search(self):
        gw = VLMGateway(_make_config())
        response_data = {
            "results": [
                {
                    "text": "John Doe signed the agreement.",
                    "score": 0.92,
                    "page": 3,
                    "document_id": "doc_001",
                    "bbox": [100, 200, 500, 240],
                },
            ],
        }
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response_data

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.return_value = mock_resp
            results = gw.semantic_search(
                query="Who signed the agreement?",
                document_id="doc_001",
                max_results=5,
                min_score=0.5,
            )

        assert len(results) == 1
        assert results[0]["score"] == 0.92
        assert results[0]["page"] == 3

    def test_empty_results(self):
        gw = VLMGateway(_make_config())
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.return_value = mock_resp
            results = gw.semantic_search(query="nonexistent content")

        assert results == []

    def test_search_with_document_scope(self):
        gw = VLMGateway(_make_config())
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.return_value = mock_resp
            gw.semantic_search(query="test", document_id="doc_specific")

            call_args = mock_client.return_value.request.call_args
            sent_payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert sent_payload["document_id"] == "doc_specific"


class TestVLMGatewayRetry:
    """Tests for retry logic in _request_with_retry."""

    def test_retry_on_server_error(self):
        gw = VLMGateway(_make_config(retry_attempts=2))

        error_resp = mock.MagicMock()
        error_resp.status_code = 503
        error_resp.text = "Service Unavailable"

        ok_resp = mock.MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"results": []}

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.side_effect = [error_resp, ok_resp]
            with mock.patch("vlm_gateway.time.sleep"):  # skip actual waits
                result = gw._request_with_retry("POST", "/test", json_body={})

        assert result == {"results": [], "processing_time_ms": mock.ANY}
        assert mock_client.return_value.request.call_count == 2

    def test_no_retry_on_auth_error(self):
        gw = VLMGateway(_make_config(retry_attempts=3))

        auth_resp = mock.MagicMock()
        auth_resp.status_code = 401
        auth_resp.text = "Unauthorized"

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.return_value = auth_resp
            with pytest.raises(VLMAuthError):
                gw._request_with_retry("POST", "/test", json_body={})

        # Should only try once (no retries)
        assert mock_client.return_value.request.call_count == 1

    def test_no_retry_on_client_error(self):
        gw = VLMGateway(_make_config(retry_attempts=3))

        error_resp = mock.MagicMock()
        error_resp.status_code = 422
        error_resp.text = "Validation error"

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.return_value = error_resp
            with pytest.raises(VLMGatewayError):
                gw._request_with_retry("POST", "/test", json_body={})

        assert mock_client.return_value.request.call_count == 1

    def test_retry_on_connection_error(self):
        gw = VLMGateway(_make_config(retry_attempts=1))

        ok_resp = mock.MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.side_effect = [
                httpx.ConnectError("refused"),
                ok_resp,
            ]
            with mock.patch("vlm_gateway.time.sleep"):
                result = gw._request_with_retry("GET", "/test")

        assert result["ok"] is True
        assert mock_client.return_value.request.call_count == 2

    def test_retry_exhausted_connection(self):
        gw = VLMGateway(_make_config(retry_attempts=1))

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.side_effect = httpx.ConnectError("refused")
            with mock.patch("vlm_gateway.time.sleep"):
                with pytest.raises(VLMConnectionError, match="unreachable"):
                    gw._request_with_retry("GET", "/test")

        assert mock_client.return_value.request.call_count == 2  # 1 attempt + 1 retry

    def test_retry_exhausted_timeout(self):
        gw = VLMGateway(_make_config(retry_attempts=1))

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.side_effect = httpx.ReadTimeout("timeout")
            with mock.patch("vlm_gateway.time.sleep"):
                with pytest.raises(VLMTimeoutError, match="timed out"):
                    gw._request_with_retry("GET", "/test")

    def test_unexpected_exception_wraps(self):
        gw = VLMGateway(_make_config(retry_attempts=0))

        with mock.patch.object(gw, "_get_client") as mock_client:
            mock_client.return_value.request.side_effect = RuntimeError("boom")
            with pytest.raises(VLMGatewayError, match="Unexpected"):
                gw._request_with_retry("GET", "/test")


class TestVLMGatewayHTTPClient:
    """Tests for the HTTP client initialization."""

    def test_client_sets_auth_header(self):
        gw = VLMGateway(_make_config(api_key="my-secret-key"))
        client = gw._get_client()
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer my-secret-key"
        gw.close()

    def test_client_no_auth_header_without_key(self):
        gw = VLMGateway(_make_config(api_key=""))
        client = gw._get_client()
        assert "Authorization" not in client.headers
        gw.close()

    def test_client_reused(self):
        gw = VLMGateway(_make_config())
        client1 = gw._get_client()
        client2 = gw._get_client()
        assert client1 is client2
        gw.close()

    def test_close_resets_client(self):
        gw = VLMGateway(_make_config())
        gw._get_client()
        assert gw._client is not None
        gw.close()
        assert gw._client is None

    def test_client_base_url(self):
        gw = VLMGateway(_make_config(endpoint_url="http://vlm:9090/v1/"))
        client = gw._get_client()
        # httpx normalizes base_url
        assert "vlm" in str(client.base_url)
        gw.close()
