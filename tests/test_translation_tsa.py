"""Tests for local RFC3161/TSA certified translation proof scaffolding."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ocr_local.features.custody import EVENT_TYPES
from ocr_local.translation.custody_adapter import emit_translation_reviewed
from ocr_local.translation.tsa import (
    FakeRFC3161TSAClient,
    HTTPRFC3161TSAClient,
    OutageTSAClient,
    TSATimestampUnavailable,
    anchor_certification_payload,
    build_rfc3161_timestamp_query,
)


def test_fake_tsa_anchor_writes_tsr_and_returns_hash(tmp_path):
    anchor = anchor_certification_payload(
        {"reviewer_id": "r1", "decision": "approve", "job_id": "job-1"},
        tsa_client=FakeRFC3161TSAClient(),
        tsa_store_dir=tmp_path,
    )

    assert anchor.clock_source == "rfc3161_tsa"
    assert len(anchor.rfc3161_tsr_sha256) == 64
    assert len(anchor.tsa_request_sha256) == 64
    assert (tmp_path / f"{anchor.rfc3161_tsr_sha256}.tsr").is_file()


def test_certified_review_adds_tsa_fields(tmp_path):
    chain = MagicMock()

    emit_translation_reviewed(
        chain,
        reviewer_id="reviewer-1",
        auth_method="piv_cac",
        decision="approve",
        job_id="job-1",
        certified=True,
        tsa_client=FakeRFC3161TSAClient(),
        tsa_store_dir=tmp_path,
    )

    event_type, payload = chain.log_event.call_args.args
    assert event_type == "TRANSLATION_REVIEWED"
    assert payload["certified"] is True
    assert payload["clock_source"] == "rfc3161_tsa"
    assert len(payload["rfc3161_tsr_sha256"]) == 64
    assert len(payload["tsa_request_sha256"]) == 64
    assert (tmp_path / f"{payload['rfc3161_tsr_sha256']}.tsr").is_file()


def test_certified_review_refuses_when_tsa_missing(tmp_path):
    chain = MagicMock()

    with pytest.raises(TSATimestampUnavailable):
        emit_translation_reviewed(
            chain,
            reviewer_id="reviewer-1",
            auth_method="piv_cac",
            decision="approve",
            job_id="job-1",
            certified=True,
            tsa_client=OutageTSAClient(),
            tsa_store_dir=tmp_path,
        )

    assert chain.log_event.call_count == 1
    event_type, payload = chain.log_event.call_args.args
    assert event_type == "CUSTODY_TSA_WARNING"
    assert payload["certification_refused"] is True


def test_certified_review_requires_tsa_client(tmp_path):
    chain = MagicMock()

    with pytest.raises(TSATimestampUnavailable):
        emit_translation_reviewed(
            chain,
            reviewer_id="reviewer-1",
            auth_method="piv_cac",
            decision="approve",
            job_id="job-1",
            certified=True,
            tsa_store_dir=tmp_path,
        )

    assert chain.log_event.call_count == 0


def test_custody_tsa_warning_event_registered():
    assert "CUSTODY_TSA_WARNING" in EVENT_TYPES


def test_rfc3161_query_contains_sha256_digest():
    digest = "a" * 64
    query = build_rfc3161_timestamp_query(digest)

    assert bytes.fromhex(digest) in query
    assert b"\x06\t`\x86H\x01e\x03\x04\x02\x01" in query


def test_http_rfc3161_client_posts_timestamp_query(monkeypatch):
    class _Response:
        content = b"\x30\x08\x30\x03\x02\x01\x00\x04\x01x"

        def raise_for_status(self):
            return None

    post = MagicMock(return_value=_Response())
    monkeypatch.setitem(__import__("sys").modules, "requests", MagicMock(post=post))

    tsr = HTTPRFC3161TSAClient("http://timestamp.example").timestamp("b" * 64)

    assert tsr == _Response.content
    _, kwargs = post.call_args
    assert kwargs["headers"]["Content-Type"] == "application/timestamp-query"
    assert bytes.fromhex("b" * 64) in kwargs["data"]
