"""Tests for ``ocr_local.translation.custody_adapter`` -- 18 tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ocr_local.features.custody import EVENT_TYPES
from ocr_local.translation.custody_adapter import (
    ReasonCode,
    emit_glossary_applied,
    emit_quality_below_threshold,
    emit_translation_applied,
    emit_translation_fallback,
    emit_translation_rejected,
    emit_translation_reviewed,
    emit_translation_skipped,
)


def _mock_chain() -> MagicMock:
    chain = MagicMock()
    return chain


def test_emit_applied_calls_log_event():
    chain = _mock_chain()
    emit_translation_applied(
        chain,
        engine_id="local_ct2_opus",
        src="en",
        tgt="fr",
        span_count=5,
        char_count=120,
        tenant_id="tenant-a",
        model_id="opus-en-fr",
        weights_sha256="abc123",
    )
    assert chain.log_event.call_count == 1
    args, _ = chain.log_event.call_args
    assert args[0] == "TRANSLATION_APPLIED"


def test_emit_applied_has_required_fields():
    chain = _mock_chain()
    emit_translation_applied(
        chain,
        engine_id="local_ct2_opus",
        src="en",
        tgt="fr",
        span_count=5,
        char_count=120,
        tenant_id="tenant-a",
        model_id="opus-en-fr",
        weights_sha256="deadbeef",
    )
    payload = chain.log_event.call_args.args[1]
    for key in ("engine_id", "src_lang", "tgt_lang", "weights_sha256"):
        assert key in payload, f"missing required field: {key}"
    assert payload["weights_sha256"] == "deadbeef"


def test_emit_rejected_has_reason_code():
    chain = _mock_chain()
    emit_translation_rejected(
        chain,
        reason_code=ReasonCode.PRIVILEGE_BLOCKED,
        engine_id="cloud_x",
        tenant_id="tenant-a",
    )
    payload = chain.log_event.call_args.args[1]
    assert "reason_code" in payload


def test_emit_rejected_reason_code_is_string():
    chain = _mock_chain()
    emit_translation_rejected(
        chain,
        reason_code=ReasonCode.TENANT_POLICY,
        tenant_id="tenant-a",
    )
    payload = chain.log_event.call_args.args[1]
    assert isinstance(payload["reason_code"], str)
    assert payload["reason_code"] == "TENANT_POLICY"


def test_emit_fallback_has_from_to():
    chain = _mock_chain()
    emit_translation_fallback(
        chain,
        from_engine="cloud_x",
        to_engine="local_ct2_opus",
        reason="cloud_unavailable",
    )
    payload = chain.log_event.call_args.args[1]
    assert payload["from_engine"] == "cloud_x"
    assert payload["to_engine"] == "local_ct2_opus"
    assert payload["reason"] == "cloud_unavailable"


def test_emit_quality_below_threshold_fields():
    chain = _mock_chain()
    emit_quality_below_threshold(
        chain,
        engine_id="local_ct2_opus",
        score=0.42,
        threshold=0.7,
        span_id="s17",
    )
    payload = chain.log_event.call_args.args[1]
    assert payload["score"] == 0.42
    assert payload["threshold"] == 0.7
    assert payload["span_id"] == "s17"


def test_emit_glossary_applied_fields():
    chain = _mock_chain()
    emit_glossary_applied(
        chain,
        glossary_id="legal-glossary-v1",
        glossary_hash="cafef00d",
        hit_count=12,
    )
    payload = chain.log_event.call_args.args[1]
    assert payload["glossary_id"] == "legal-glossary-v1"
    assert payload["glossary_hash"] == "cafef00d"
    assert payload["hit_count"] == 12


@pytest.mark.parametrize("auth_method", ["piv_cac", "oidc_mfa", "hardware_token"])
def test_emit_reviewed_valid_auth(auth_method):
    chain = _mock_chain()
    emit_translation_reviewed(
        chain,
        reviewer_id="reviewer-1",
        auth_method=auth_method,
        decision="approve",
        job_id="job-42",
    )
    assert chain.log_event.call_count == 1


@pytest.mark.parametrize("auth_method", ["password", "api_key", "basic", ""])
def test_emit_reviewed_invalid_auth_raises(auth_method):
    chain = _mock_chain()
    with pytest.raises(ValueError):
        emit_translation_reviewed(
            chain,
            reviewer_id="reviewer-1",
            auth_method=auth_method,
            decision="approve",
            job_id="job-42",
        )
    # Must not have logged anything
    assert chain.log_event.call_count == 0


def test_emit_reviewed_fields():
    chain = _mock_chain()
    emit_translation_reviewed(
        chain,
        reviewer_id="reviewer-7",
        auth_method="piv_cac",
        decision="reject",
        job_id="job-99",
    )
    payload = chain.log_event.call_args.args[1]
    assert payload["reviewer_id"] == "reviewer-7"
    assert payload["decision"] == "reject"
    assert payload["job_id"] == "job-99"


def test_emit_skipped_fields():
    chain = _mock_chain()
    emit_translation_skipped(chain, reason="text_only_bypass", tenant_id="tenant-z")
    payload = chain.log_event.call_args.args[1]
    assert payload["reason"] == "text_only_bypass"
    assert payload["tenant_id"] == "tenant-z"


def test_all_emitters_call_log_event_once():
    chain = _mock_chain()
    emit_translation_applied(
        chain,
        engine_id="e",
        src="en",
        tgt="fr",
        span_count=1,
        char_count=1,
        tenant_id="t",
        model_id="m",
        weights_sha256="h",
    )
    emit_translation_rejected(chain, reason_code=ReasonCode.PRIVILEGE_BLOCKED)
    emit_translation_fallback(chain, from_engine="a", to_engine="b", reason="x")
    emit_quality_below_threshold(
        chain, engine_id="e", score=0.1, threshold=0.5, span_id="s"
    )
    emit_glossary_applied(chain, glossary_id="g", glossary_hash="h", hit_count=1)
    emit_translation_reviewed(
        chain,
        reviewer_id="r",
        auth_method="piv_cac",
        decision="approve",
        job_id="j",
    )
    emit_translation_skipped(chain, reason="r")
    assert chain.log_event.call_count == 7


def test_reason_codes_all_defined():
    expected = {
        "PRIVILEGE_BLOCKED",
        "UNSUPPORTED_LANGUAGE",
        "SCHEMA_DRIFT",
        "COST_CEILING_EXCEEDED",
        "TENANT_POLICY",
        "RESIDENCY_VIOLATION",
        "MISSING_MFA",
        # Plan B Wave M2 model-cache lifecycle reason codes.
        "MODEL_DOWNLOADED",
        "MODEL_INTEGRITY_VERIFIED",
        "MODEL_INTEGRITY_FAILED",
        "MODEL_EVICTED",
        "MODEL_LOAD_BLOCKED_NC_LICENSE",
        "MODEL_PINNED",
        "MODEL_UNPINNED",
        # Plan B Wave M2 PR B14 -- tenant-aware router (router_v2).
        "TENANT_ENGINE_SELECTED",
        "TENANT_ENGINE_NO_CANDIDATES",
        "TENANT_POLICY_HYDRATED",
        # Plan B Wave M2 PR B17 -- batch translation scheduling.
        "BATCH_SUBMITTED",
        "BATCH_FAN_OUT",
        "BATCH_INPUT_COMPLETED",
        "BATCH_INPUT_FAILED",
        "BATCH_CANCELLED",
        "BATCH_COMPLETED",
        "BATCH_REJECTED_CERTIFIED",
        # Plan B Wave 3 B15 -- COMETKiwi quality estimation.
        "QUALITY_ESTIMATED",
    }
    actual = {rc.value for rc in ReasonCode}
    assert actual == expected


def test_privilege_blocked_emit_before_raise():
    """Integration with router: log_event must be called before exception propagates."""
    from ocr_local.translation.models import EngineCapability, TranslationRequest
    from ocr_local.translation.policy import PolicyDenied, TenantPolicy
    from ocr_local.translation.router import select_engine

    # Stub engine class that looks like cloud
    class _FakeCloud:
        capability = EngineCapability(
            id="fake_cloud_xx",
            is_local=False,
            is_cloud=True,
            supports_pairs="any",
            quality_class="standard",
            latency_class="standard",
            license="Apache-2.0",
            provider_retention_class="zero_retention_with_baa",
            deployment_envs=["cloud"],
        )

        def __init__(self):
            pass

    from ocr_local.translation.engines import ENGINE_REGISTRY

    original = dict(ENGINE_REGISTRY)
    passthrough = original.get("passthrough")
    try:
        ENGINE_REGISTRY.clear()
        if passthrough is not None:
            ENGINE_REGISTRY["passthrough"] = passthrough
        ENGINE_REGISTRY[_FakeCloud.capability.id] = _FakeCloud
        chain = _mock_chain()
        req = TranslationRequest(
            src_lang="en",
            tgt_lang="ja",
            privilege_flag=True,
            tenant_id="tenant-a",
        )
        tenant = TenantPolicy(tenant_id="tenant-a", allow_cloud_translation=True)
        with pytest.raises(PolicyDenied):
            select_engine(req, tenant, chain)
        # log_event MUST have been called before the raise
        assert chain.log_event.called
        # And the call must be a TRANSLATION_REJECTED with PRIVILEGE_BLOCKED
        args = chain.log_event.call_args.args
        assert args[0] == "TRANSLATION_REJECTED"
        assert args[1]["reason_code"] == "PRIVILEGE_BLOCKED"
    finally:
        ENGINE_REGISTRY.clear()
        ENGINE_REGISTRY.update(original)


def test_reason_code_is_str_enum():
    assert isinstance(ReasonCode.PRIVILEGE_BLOCKED, str)


def test_emit_applied_extra_kwargs_passed():
    chain = _mock_chain()
    emit_translation_applied(
        chain,
        engine_id="e",
        src="en",
        tgt="fr",
        span_count=1,
        char_count=1,
        tenant_id="t",
        model_id="m",
        weights_sha256="h",
        custom_field="custom_value",
        nested={"k": "v"},
    )
    payload = chain.log_event.call_args.args[1]
    assert payload["custom_field"] == "custom_value"
    assert payload["nested"] == {"k": "v"}


def test_emit_rejected_none_engine_id_allowed():
    chain = _mock_chain()
    emit_translation_rejected(
        chain,
        reason_code=ReasonCode.UNSUPPORTED_LANGUAGE,
        engine_id=None,
        tenant_id="tenant-a",
    )
    payload = chain.log_event.call_args.args[1]
    assert payload["engine_id"] is None


def test_custody_events_in_event_types():
    expected = {
        "TRANSLATION_APPLIED",
        "TRANSLATION_REJECTED",
        "QUALITY_BELOW_THRESHOLD",
        "TRANSLATION_FALLBACK",
        "TRANSLATION_REVIEWED",
        "GLOSSARY_APPLIED",
        "TRANSLATION_SKIPPED",
        "CUSTODY_TSA_WARNING",
    }
    for event_name in expected:
        assert event_name in EVENT_TYPES, f"{event_name} not in EVENT_TYPES"
