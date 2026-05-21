from __future__ import annotations

import json
import io
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from ocr_local.translation.external_client import (
    TranslationServiceClient,
    TranslationServiceError,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "edc_contracts"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_list_engines_calls_external_service(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return _Response({"engines": [{"id": "passthrough"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = TranslationServiceClient("http://translation.local", timeout=7)

    engines = client.list_engines()

    assert engines == [{"id": "passthrough"}]
    request, timeout = calls[0]
    assert request.full_url == "http://translation.local/api/v1/translation/engines"
    assert request.get_method() == "GET"
    assert timeout == 7


def test_check_readiness_reports_ready(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return _Response({"status": "healthy", "service": "edc_translation"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = TranslationServiceClient("http://translation.local", timeout=2)

    status = client.check_readiness()

    assert status.ready is True
    assert status.status == "ready"
    assert status.enabled is True
    assert status.url == "http://translation.local/health"
    assert calls[0][0].get_method() == "GET"
    assert calls[0][1] == 2


def test_check_readiness_reports_unavailable_status(monkeypatch):
    def fake_urlopen(_request, timeout):
        return _Response({"status": "degraded"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = TranslationServiceClient("http://translation.local")

    status = client.check_readiness()

    assert status.ready is False
    assert status.status == "unavailable"
    assert "degraded" in status.message


def test_check_readiness_reports_http_unavailable(monkeypatch):
    def fake_urlopen(_request, timeout):
        raise HTTPError(
            "http://translation.local/health",
            503,
            "Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"status":"unavailable"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = TranslationServiceClient("http://translation.local")

    status = client.check_readiness()

    assert status.ready is False
    assert status.status == "unavailable"
    assert "HTTP 503" in status.message


def test_check_readiness_reports_unreachable(monkeypatch):
    def fake_urlopen(_request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = TranslationServiceClient("http://translation.local")

    status = client.check_readiness()

    assert status.ready is False
    assert status.status == "unreachable"
    assert "connection refused" in status.message


def test_check_readiness_adds_auto_route_query(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _Response({"status": "ready"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = TranslationServiceClient("http://translation.local")

    status = client.check_readiness(
        path="/api/v1/translation/readiness/auto-route",
        source_language="en",
        target_language="fr",
    )

    assert status.ready is True
    assert captured["url"].endswith(
        "/api/v1/translation/readiness/auto-route?source_language=en&target_language=fr"
    )


def test_translate_bundle_validates_request_and_response(monkeypatch):
    document_bundle = _load("document-bundle-v1.valid.json")
    translation_bundle = _load("translation-bundle-v1.valid.json")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response({"translation_bundle": translation_bundle})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    operator_placeholder = "placeholder"
    client = TranslationServiceClient(
        "http://translation.local/",
        api_key=operator_placeholder,
        timeout=5,
    )

    out = client.translate_bundle(
        document_bundle,
        target_language="fr",
        provider_id="passthrough",
    )

    assert out == translation_bundle
    assert captured["url"] == "http://translation.local/api/v1/translation/bundles"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 5
    assert captured["headers"]["X-api-key"] == operator_placeholder
    assert captured["payload"]["target_language"] == "fr"
    assert captured["payload"]["provider_id"] == "passthrough"
    assert captured["payload"]["document_bundle"]["schema_version"] == "document-bundle-v1"


def test_translate_bundle_rejects_invalid_document_bundle(monkeypatch):
    document_bundle = _load("document-bundle-v1.valid.json")
    document_bundle.pop("source_ocr_sha256")

    def fake_urlopen(_request, _timeout):  # pragma: no cover - must not call
        raise AssertionError("urlopen should not be called for invalid input")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = TranslationServiceClient()

    with pytest.raises(Exception):
        client.translate_bundle(document_bundle, target_language="fr")


def test_translate_bundle_rejects_invalid_translation_response(monkeypatch):
    document_bundle = _load("document-bundle-v1.valid.json")
    bad_translation = _load("translation-bundle-v1.valid.json")
    bad_translation.pop("source_bundle_sha256")

    def fake_urlopen(_request, _timeout):
        return _Response({"translation_bundle": bad_translation})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = TranslationServiceClient()

    with pytest.raises(Exception):
        client.translate_bundle(document_bundle, target_language="fr")


def test_service_unavailable_is_wrapped(monkeypatch):
    def fake_urlopen(_request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = TranslationServiceClient()

    with pytest.raises(TranslationServiceError, match="unavailable"):
        client.list_engines()


def test_from_env(monkeypatch):
    monkeypatch.setenv("EDC_TRANSLATION_URL", "http://env.translation")
    operator_placeholder = "placeholder"
    monkeypatch.setenv("EDC_TRANSLATION_API_KEY", operator_placeholder)
    monkeypatch.setenv("EDC_TRANSLATION_TIMEOUT_SECONDS", "3")

    client = TranslationServiceClient.from_env()

    assert client.base_url == "http://env.translation"
    assert client.api_key == operator_placeholder
    assert client.timeout == 3.0
