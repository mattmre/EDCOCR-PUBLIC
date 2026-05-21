"""Tests for ``ocr_local.translation.policy`` and routing-policy
interactions in ``ocr_local.translation.router`` -- 22 tests."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from ocr_local.translation.custody_adapter import ReasonCode
from ocr_local.translation.engines import ENGINE_REGISTRY
from ocr_local.translation.models import EngineCapability, TranslationRequest
from ocr_local.translation.policy import (
    PolicyDenied,
    TenantPolicy,
    compute_policy_hash,
    load_tenant_policy,
)
from ocr_local.translation.router import select_engine


def _mock_chain() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Fake engine helpers (registered/unregistered per test for isolation).
# ---------------------------------------------------------------------------


def _make_fake_engine(
    *,
    engine_id: str,
    is_local: bool,
    is_cloud: bool,
    quality: str = "standard",
    license_str: str = "Apache-2.0",
    supports_pairs="any",
):
    cap = EngineCapability(
        id=engine_id,
        is_local=is_local,
        is_cloud=is_cloud,
        supports_pairs=supports_pairs,
        quality_class=quality,
        latency_class="standard",
        license=license_str,
        provider_retention_class="local_only" if is_local else "zero_retention_with_baa",
        deployment_envs=["local"] if is_local else ["cloud"],
    )

    class _FakeEngine:
        capability = cap

        def __init__(self):
            pass

    _FakeEngine.__name__ = f"Fake_{engine_id}"
    return _FakeEngine


@contextmanager
def _registered(*engine_classes):
    original = dict(ENGINE_REGISTRY)
    passthrough = original.get("passthrough")
    try:
        ENGINE_REGISTRY.clear()
        if passthrough is not None:
            ENGINE_REGISTRY["passthrough"] = passthrough
        for cls in engine_classes:
            ENGINE_REGISTRY[cls.capability.id] = cls
        yield
    finally:
        ENGINE_REGISTRY.clear()
        ENGINE_REGISTRY.update(original)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_tenant_policy_returns_default():
    p = load_tenant_policy("tenant-a")
    assert p.tenant_id == "tenant-a"
    # Defaults are permissive
    assert p.require_local_only is False
    assert p.allow_nllb_commercial is True
    assert p.allowed_engine_ids == []


def test_compute_policy_hash_stable():
    p1 = TenantPolicy(tenant_id="tenant-a")
    p2 = TenantPolicy(tenant_id="tenant-a")
    assert compute_policy_hash(p1) == compute_policy_hash(p2)


def test_compute_policy_hash_differs():
    p1 = TenantPolicy(tenant_id="tenant-a")
    p2 = TenantPolicy(tenant_id="tenant-b")
    assert compute_policy_hash(p1) != compute_policy_hash(p2)


def test_policy_denied_has_reason_code():
    err = PolicyDenied(ReasonCode.PRIVILEGE_BLOCKED, "x")
    assert err.reason_code == ReasonCode.PRIVILEGE_BLOCKED


def test_policy_denied_str():
    err = PolicyDenied(ReasonCode.TENANT_POLICY, "blocked")
    assert "TENANT_POLICY" in str(err)


def test_privilege_flag_blocks_cloud():
    cloud = _make_fake_engine(engine_id="fake_cloud_pol", is_local=False, is_cloud=True)
    with _registered(cloud):
        chain = _mock_chain()
        req = TranslationRequest(
            src_lang="en",
            tgt_lang="ja",
            privilege_flag=True,
            tenant_id="tenant-a",
        )
        tenant = TenantPolicy(tenant_id="tenant-a", allow_cloud_translation=True)
        with pytest.raises(PolicyDenied) as exc_info:
            select_engine(req, tenant, chain)
        assert exc_info.value.reason_code == ReasonCode.PRIVILEGE_BLOCKED


def test_privilege_block_emits_custody_before_raise():
    cloud = _make_fake_engine(engine_id="fake_cloud_pol2", is_local=False, is_cloud=True)
    with _registered(cloud):
        chain = _mock_chain()
        # Capture call ordering: chain.log_event must be called BEFORE the
        # exception escapes select_engine.
        call_order: list[str] = []

        def _spy_log_event(*args, **kwargs):
            call_order.append("log_event")

        chain.log_event.side_effect = _spy_log_event

        req = TranslationRequest(
            src_lang="en",
            tgt_lang="ja",
            privilege_flag=True,
            tenant_id="tenant-a",
        )
        tenant = TenantPolicy(tenant_id="tenant-a", allow_cloud_translation=True)
        try:
            select_engine(req, tenant, chain)
        except PolicyDenied:
            call_order.append("raised")

        assert call_order == ["log_event", "raised"]


def test_cloud_blocked_by_tenant_policy():
    cloud = _make_fake_engine(engine_id="fake_cloud_pol3", is_local=False, is_cloud=True)
    with _registered(cloud):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(tenant_id="tenant-a", allow_cloud_translation=False)
        with pytest.raises(PolicyDenied) as exc_info:
            select_engine(req, tenant, chain)
        assert exc_info.value.reason_code == ReasonCode.TENANT_POLICY


def test_tenant_policy_block_emits_custody():
    cloud = _make_fake_engine(engine_id="fake_cloud_pol4", is_local=False, is_cloud=True)
    with _registered(cloud):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(tenant_id="tenant-a", allow_cloud_translation=False)
        with pytest.raises(PolicyDenied):
            select_engine(req, tenant, chain)
        assert chain.log_event.called
        args = chain.log_event.call_args.args
        assert args[0] == "TRANSLATION_REJECTED"
        assert args[1]["reason_code"] == "TENANT_POLICY"


def test_unsupported_language_emits_custody():
    chain = _mock_chain()
    req = TranslationRequest(src_lang="xx", tgt_lang="yy", tenant_id="tenant-a")
    tenant = TenantPolicy(tenant_id="tenant-a")
    # Engines registered support either "any" or matching pairs.  With xx->yy
    # plus only the passthrough engine in the registry (which is excluded
    # for src!=tgt), there must be no candidates.  We register a fake that
    # only supports en->fr to make the test deterministic.
    fake = _make_fake_engine(
        engine_id="fake_local_xy",
        is_local=True,
        is_cloud=False,
        supports_pairs=[("en", "fr")],
    )
    with _registered(fake):
        with pytest.raises(PolicyDenied) as exc_info:
            select_engine(req, tenant, chain)
        assert exc_info.value.reason_code == ReasonCode.UNSUPPORTED_LANGUAGE
        args = chain.log_event.call_args.args
        assert args[0] == "TRANSLATION_REJECTED"
        assert args[1]["reason_code"] == "UNSUPPORTED_LANGUAGE"


def test_nllb_blocked_for_commercial():
    nllb = _make_fake_engine(
        engine_id="fake_nllb",
        is_local=True,
        is_cloud=False,
        license_str="CC-BY-NC-4.0",
    )
    with _registered(nllb):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(
            tenant_id="tenant-a", allow_nllb_commercial=False
        )
        with pytest.raises(PolicyDenied) as exc_info:
            select_engine(req, tenant, chain)
        assert exc_info.value.reason_code == ReasonCode.UNSUPPORTED_LANGUAGE


def test_passthrough_for_same_language():
    chain = _mock_chain()
    req = TranslationRequest(src_lang="en", tgt_lang="en", tenant_id="tenant-a")
    tenant = TenantPolicy(tenant_id="tenant-a")
    engine = select_engine(req, tenant, chain)
    assert engine.capability.id == "passthrough"
    # No reject custody event for same-language passthrough.
    assert not chain.log_event.called


def test_allowed_engine_ids_filters():
    a = _make_fake_engine(engine_id="fake_a", is_local=True, is_cloud=False)
    b = _make_fake_engine(engine_id="fake_b", is_local=True, is_cloud=False)
    with _registered(a, b):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(tenant_id="tenant-a", allowed_engine_ids=["fake_a"])
        engine = select_engine(req, tenant, chain)
        assert engine.capability.id == "fake_a"


def test_blocked_engine_ids_excludes():
    a = _make_fake_engine(engine_id="fake_a2", is_local=True, is_cloud=False, quality="legal")
    b = _make_fake_engine(engine_id="fake_b2", is_local=True, is_cloud=False, quality="standard")
    with _registered(a, b):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(tenant_id="tenant-a", blocked_engine_ids=["fake_a2"])
        engine = select_engine(req, tenant, chain)
        assert engine.capability.id == "fake_b2"


def test_gemini_ai_studio_blocked_in_production(monkeypatch):
    gemini = _make_fake_engine(
        engine_id="cloud_gemini_ai_studio",
        is_local=False,
        is_cloud=True,
    )
    monkeypatch.setenv("DEPLOYMENT_ENV", "production")
    with _registered(gemini):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(tenant_id="tenant-a", allow_cloud_translation=True)
        with pytest.raises(PolicyDenied) as exc_info:
            select_engine(req, tenant, chain)
        assert exc_info.value.reason_code == ReasonCode.UNSUPPORTED_LANGUAGE


def test_local_engine_preferred_over_cloud():
    local = _make_fake_engine(engine_id="fake_local_pref", is_local=True, is_cloud=False)
    cloud = _make_fake_engine(engine_id="fake_cloud_pref", is_local=False, is_cloud=True)
    with _registered(local, cloud):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(tenant_id="tenant-a", allow_cloud_translation=True)
        engine = select_engine(req, tenant, chain)
        assert engine.capability.is_local is True


def test_legal_quality_preferred_over_standard():
    standard = _make_fake_engine(
        engine_id="fake_std", is_local=True, is_cloud=False, quality="standard"
    )
    legal = _make_fake_engine(
        engine_id="fake_legal", is_local=True, is_cloud=False, quality="legal"
    )
    with _registered(standard, legal):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(tenant_id="tenant-a")
        engine = select_engine(req, tenant, chain)
        assert engine.capability.quality_class == "legal"


def test_policy_hash_includes_tenant_id():
    p1 = TenantPolicy(tenant_id="tenant-a", allow_cloud_translation=True)
    p2 = TenantPolicy(tenant_id="tenant-b", allow_cloud_translation=True)
    assert compute_policy_hash(p1) != compute_policy_hash(p2)


def test_require_local_only_excludes_cloud():
    local = _make_fake_engine(engine_id="fake_local_lo", is_local=True, is_cloud=False)
    cloud = _make_fake_engine(engine_id="fake_cloud_lo", is_local=False, is_cloud=True)
    with _registered(local, cloud):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(
            tenant_id="tenant-a",
            require_local_only=True,
            allow_cloud_translation=True,
        )
        engine = select_engine(req, tenant, chain)
        assert engine.capability.is_local is True


def test_residency_not_blocking_without_cloud():
    local = _make_fake_engine(engine_id="fake_local_res", is_local=True, is_cloud=False)
    with _registered(local):
        chain = _mock_chain()
        req = TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="tenant-a")
        tenant = TenantPolicy(
            tenant_id="tenant-a",
            residency_region="eu-west-1",
        )
        engine = select_engine(req, tenant, chain)
        assert engine.capability.is_local is True


def test_all_reason_codes_are_strings():
    for rc in ReasonCode:
        assert isinstance(rc.value, str)
        assert isinstance(rc, str)


def test_reason_code_privilege_blocked_value():
    assert ReasonCode.PRIVILEGE_BLOCKED == "PRIVILEGE_BLOCKED"
    assert ReasonCode.PRIVILEGE_BLOCKED.value == "PRIVILEGE_BLOCKED"
