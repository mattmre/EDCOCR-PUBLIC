"""Tests for the operator EXTERNAL_TRANSLATION seam smoke script."""

from __future__ import annotations

import json

from ocr_local.contracts import canonical_json_sha256, validate_contract_payload
from ocr_local.translation.external_client import TranslationReadinessStatus
from scripts import run_external_translation_seam_smoke as smoke


def test_smoke_document_bundle_is_contract_valid():
    bundle = smoke._smoke_document_bundle()

    validate_contract_payload(bundle, "document-bundle-v1")
    assert bundle["spans"][0]["text"] == "Hello."


def test_smoke_script_prints_compact_summary(monkeypatch, capsys):
    class _Client:
        def __init__(self, base_url, *, timeout=30.0):
            self.base_url = base_url
            self.timeout = timeout

        def list_engines(self):
            return [{"id": "deterministic_ci"}]

        def translate_bundle(self, document_bundle, *, target_language, provider_id):
            return {
                "engine_provider": {"id": provider_id},
                "target_language": target_language,
                "source_bundle_sha256": canonical_json_sha256(document_bundle),
                "translated_spans": [
                    {
                        "translated_text": (
                            f"{document_bundle['spans'][0]['text']} [en->{target_language}]"
                        )
                    }
                ],
            }

    monkeypatch.setattr(smoke, "TranslationServiceClient", _Client)
    monkeypatch.setattr(
        smoke,
        "external_translation_readiness",
        lambda *_args, **_kwargs: TranslationReadinessStatus(
            status="ready",
            ready=True,
            enabled=True,
            url="http://example.test/health",
            message="ready"))

    assert (
        smoke.main(
            [
                "--url",
                "http://example.test",
                "--provider",
                "deterministic_ci",
                "--target",
                "fr",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["readiness"] == "ready"
    assert payload["url"] == "http://example.test"
    assert payload["provider_id"] == "deterministic_ci"
    assert payload["translated_text"] == "Hello. [en->fr]"


def test_smoke_script_blocks_when_readiness_is_not_green(monkeypatch, capsys):
    class _Client:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("translation client should not be constructed")

    monkeypatch.setattr(smoke, "TranslationServiceClient", _Client)
    monkeypatch.setattr(
        smoke,
        "external_translation_readiness",
        lambda *_args, **_kwargs: TranslationReadinessStatus(
            status="unreachable",
            ready=False,
            enabled=True,
            url="http://example.test/health",
            message="connection refused"))

    assert smoke.main(["--url", "http://example.test"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["readiness"] == "unreachable"
    assert payload["message"] == "connection refused"
