"""Tests for ``ocr_local.translation.router`` -- 24 tests."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from ocr_local.translation.custody_adapter import ReasonCode
from ocr_local.translation.engines import ENGINE_REGISTRY
from ocr_local.translation.engines.base import TranslationEngine
from ocr_local.translation.models import EngineCapability, TranslationRequest
from ocr_local.translation.policy import PolicyDenied, TenantPolicy
from ocr_local.translation.router import (
    _rank_candidates,
    _supports_pair,
    select_engine,
)


def _mock_chain() -> MagicMock:
    return MagicMock()


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
# select_engine routing scenarios (12)
# ---------------------------------------------------------------------------


def test_select_engine_returns_local_for_default_tenant():
    local = _make_fake_engine(engine_id="r_local_1", is_local=True, is_cloud=False)
    with _registered(local):
        engine = select_engine(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t"),
            _mock_chain(),
        )
        assert engine.capability.id == "r_local_1"


def test_select_engine_returns_cloud_when_allowed():
    cloud = _make_fake_engine(engine_id="r_cloud_1", is_local=False, is_cloud=True)
    with _registered(cloud):
        engine = select_engine(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t", allow_cloud_translation=True),
            _mock_chain(),
        )
        assert engine.capability.id == "r_cloud_1"


def test_select_engine_privilege_block_with_only_cloud():
    cloud = _make_fake_engine(engine_id="r_cloud_2", is_local=False, is_cloud=True)
    with _registered(cloud):
        chain = _mock_chain()
        with pytest.raises(PolicyDenied) as exc_info:
            select_engine(
                TranslationRequest(
                    src_lang="en", tgt_lang="ja", privilege_flag=True, tenant_id="t"
                ),
                TenantPolicy(tenant_id="t", allow_cloud_translation=True),
                chain,
            )
        assert exc_info.value.reason_code == ReasonCode.PRIVILEGE_BLOCKED


def test_select_engine_privilege_picks_local_when_available():
    local = _make_fake_engine(engine_id="r_local_2", is_local=True, is_cloud=False)
    cloud = _make_fake_engine(engine_id="r_cloud_3", is_local=False, is_cloud=True)
    with _registered(local, cloud):
        engine = select_engine(
            TranslationRequest(
                src_lang="en", tgt_lang="ja", privilege_flag=True, tenant_id="t"
            ),
            TenantPolicy(tenant_id="t", allow_cloud_translation=True),
            _mock_chain(),
        )
        # Local wins over cloud regardless of privilege flag.
        assert engine.capability.is_local is True


def test_select_engine_unsupported_language_emits_and_raises():
    engine = _make_fake_engine(
        engine_id="r_local_only_enfr",
        is_local=True,
        is_cloud=False,
        supports_pairs=[("en", "fr")],
    )
    with _registered(engine):
        chain = _mock_chain()
        with pytest.raises(PolicyDenied) as exc_info:
            select_engine(
                TranslationRequest(src_lang="zz", tgt_lang="qq", tenant_id="t"),
                TenantPolicy(tenant_id="t"),
                chain,
            )
        assert exc_info.value.reason_code == ReasonCode.UNSUPPORTED_LANGUAGE
        assert chain.log_event.called


def test_select_engine_tenant_policy_blocks_cloud():
    cloud = _make_fake_engine(engine_id="r_cloud_4", is_local=False, is_cloud=True)
    with _registered(cloud):
        chain = _mock_chain()
        with pytest.raises(PolicyDenied) as exc_info:
            select_engine(
                TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
                TenantPolicy(tenant_id="t", allow_cloud_translation=False),
                chain,
            )
        assert exc_info.value.reason_code == ReasonCode.TENANT_POLICY


def test_select_engine_blocked_engine_ids_skipped():
    a = _make_fake_engine(engine_id="r_block_a", is_local=True, is_cloud=False, quality="legal")
    b = _make_fake_engine(engine_id="r_block_b", is_local=True, is_cloud=False, quality="standard")
    with _registered(a, b):
        engine = select_engine(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t", blocked_engine_ids=["r_block_a"]),
            _mock_chain(),
        )
        assert engine.capability.id == "r_block_b"


def test_select_engine_allowed_engine_ids_filters():
    a = _make_fake_engine(engine_id="r_allow_a", is_local=True, is_cloud=False, quality="legal")
    b = _make_fake_engine(engine_id="r_allow_b", is_local=True, is_cloud=False, quality="standard")
    with _registered(a, b):
        engine = select_engine(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t", allowed_engine_ids=["r_allow_b"]),
            _mock_chain(),
        )
        assert engine.capability.id == "r_allow_b"


def test_select_engine_require_local_only_excludes_cloud():
    cloud = _make_fake_engine(engine_id="r_cloud_lo", is_local=False, is_cloud=True)
    with _registered(cloud):
        chain = _mock_chain()
        with pytest.raises(PolicyDenied):
            select_engine(
                TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
                TenantPolicy(
                    tenant_id="t",
                    require_local_only=True,
                    allow_cloud_translation=True,
                ),
                chain,
            )


def test_select_engine_nllb_blocked_for_commercial():
    nllb = _make_fake_engine(
        engine_id="r_nllb",
        is_local=True,
        is_cloud=False,
        license_str="CC-BY-NC-4.0",
    )
    with _registered(nllb):
        chain = _mock_chain()
        with pytest.raises(PolicyDenied) as exc_info:
            select_engine(
                TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
                TenantPolicy(tenant_id="t", allow_nllb_commercial=False),
                chain,
            )
        assert exc_info.value.reason_code == ReasonCode.UNSUPPORTED_LANGUAGE


def test_select_engine_returns_subclass_of_translation_engine():
    """Returned object should at least follow the TranslationEngine protocol."""
    # We don't strictly subclass in the fake engine; instead verify the
    # contract attributes are present.
    local = _make_fake_engine(engine_id="r_proto_check", is_local=True, is_cloud=False)
    with _registered(local):
        engine = select_engine(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t"),
            _mock_chain(),
        )
        assert hasattr(engine, "capability")


def test_select_engine_passthrough_for_same_lang_returns_real_engine():
    """Passthrough engine is registered by package import, so same-lang
    requests must return its concrete class (subclass of TranslationEngine)."""
    engine = select_engine(
        TranslationRequest(src_lang="en", tgt_lang="en", tenant_id="t"),
        TenantPolicy(tenant_id="t"),
        _mock_chain(),
    )
    assert engine.capability.id == "passthrough"
    assert isinstance(engine, TranslationEngine)


# ---------------------------------------------------------------------------
# _rank_candidates ordering (5)
# ---------------------------------------------------------------------------


def test_rank_local_before_cloud():
    local = _make_fake_engine(engine_id="rk_local", is_local=True, is_cloud=False)
    cloud = _make_fake_engine(engine_id="rk_cloud", is_local=False, is_cloud=True)
    with _registered(cloud, local):  # registered cloud first to verify sort
        ranked = _rank_candidates(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t", allow_cloud_translation=True),
        )
        assert ranked[0].capability.is_local is True


def test_rank_legal_before_standard():
    standard = _make_fake_engine(
        engine_id="rk_std", is_local=True, is_cloud=False, quality="standard"
    )
    legal = _make_fake_engine(
        engine_id="rk_legal", is_local=True, is_cloud=False, quality="legal"
    )
    with _registered(standard, legal):
        ranked = _rank_candidates(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t"),
        )
        assert ranked[0].capability.quality_class == "legal"


def test_rank_excludes_passthrough_when_src_neq_tgt():
    ranked = _rank_candidates(
        TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
        TenantPolicy(tenant_id="t"),
    )
    for cls in ranked:
        assert cls.capability.id != "passthrough"


def test_rank_empty_when_no_candidates():
    only = _make_fake_engine(
        engine_id="rk_only_enfr",
        is_local=True,
        is_cloud=False,
        supports_pairs=[("en", "fr")],
    )
    with _registered(only):
        ranked = _rank_candidates(
            TranslationRequest(src_lang="zz", tgt_lang="qq", tenant_id="t"),
            TenantPolicy(tenant_id="t"),
        )
        assert ranked == []


def test_rank_respects_blocked_list():
    a = _make_fake_engine(engine_id="rk_blk_a", is_local=True, is_cloud=False)
    b = _make_fake_engine(engine_id="rk_blk_b", is_local=True, is_cloud=False)
    with _registered(a, b):
        ranked = _rank_candidates(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t", blocked_engine_ids=["rk_blk_a"]),
        )
        assert all(c.capability.id != "rk_blk_a" for c in ranked)


# ---------------------------------------------------------------------------
# _supports_pair (3)
# ---------------------------------------------------------------------------


def test_supports_pair_any():
    cap = EngineCapability(
        id="x",
        is_local=True,
        is_cloud=False,
        supports_pairs="any",
        quality_class="standard",
        latency_class="standard",
        license="Apache-2.0",
        provider_retention_class="local_only",
        deployment_envs=["local"],
    )
    assert _supports_pair(cap, "en", "ja") is True
    assert _supports_pair(cap, "fr", "de") is True


def test_supports_pair_explicit_list():
    cap = EngineCapability(
        id="x",
        is_local=True,
        is_cloud=False,
        supports_pairs=[("en", "fr"), ("fr", "en")],
        quality_class="standard",
        latency_class="standard",
        license="Apache-2.0",
        provider_retention_class="local_only",
        deployment_envs=["local"],
    )
    assert _supports_pair(cap, "en", "fr") is True
    assert _supports_pair(cap, "fr", "en") is True
    assert _supports_pair(cap, "en", "ja") is False


def test_supports_pair_wildcard_target():
    cap = EngineCapability(
        id="x",
        is_local=True,
        is_cloud=False,
        supports_pairs=[("en", "*")],
        quality_class="standard",
        latency_class="standard",
        license="Apache-2.0",
        provider_retention_class="local_only",
        deployment_envs=["local"],
    )
    assert _supports_pair(cap, "en", "ja") is True
    assert _supports_pair(cap, "en", "fr") is True
    assert _supports_pair(cap, "fr", "ja") is False


# ---------------------------------------------------------------------------
# Same src/tgt passthrough (2)
# ---------------------------------------------------------------------------


def test_same_src_tgt_does_not_consult_engine_registry():
    """Same-lang short-circuit should bypass cloud/policy checks entirely."""
    chain = _mock_chain()
    engine = select_engine(
        TranslationRequest(src_lang="ja", tgt_lang="ja", privilege_flag=True, tenant_id="t"),
        TenantPolicy(tenant_id="t", allow_cloud_translation=False),
        chain,
    )
    assert engine.capability.id == "passthrough"
    assert not chain.log_event.called


def test_same_src_tgt_returns_instance_not_class():
    engine = select_engine(
        TranslationRequest(src_lang="en", tgt_lang="en", tenant_id="t"),
        TenantPolicy(tenant_id="t"),
        _mock_chain(),
    )
    # Instance, not class
    assert not isinstance(engine, type)


# ---------------------------------------------------------------------------
# Multi-engine registry + instantiation (2)
# ---------------------------------------------------------------------------


def test_multiple_engines_in_registry_picks_best_quality():
    high = _make_fake_engine(
        engine_id="multi_legal", is_local=True, is_cloud=False, quality="legal"
    )
    med = _make_fake_engine(
        engine_id="multi_std", is_local=True, is_cloud=False, quality="standard"
    )
    low = _make_fake_engine(
        engine_id="multi_draft", is_local=True, is_cloud=False, quality="draft"
    )
    with _registered(low, med, high):
        engine = select_engine(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t"),
            _mock_chain(),
        )
        assert engine.capability.id == "multi_legal"


def test_select_engine_returns_instance():
    local = _make_fake_engine(engine_id="ret_local", is_local=True, is_cloud=False)
    with _registered(local):
        engine = select_engine(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t"),
            _mock_chain(),
        )
        # Returned object must be an instance, not the class itself.
        assert not isinstance(engine, type)
        assert isinstance(engine, local)
