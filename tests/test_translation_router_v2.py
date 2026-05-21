"""Tests for ``ocr_local.translation.router`` v2 -- Plan B Wave M2 PR B14.

Covers tenant-aware engine selection, glossary preprocessing,
NC-license filtering, cache-availability filtering, custody event
emission, and the REST endpoint integration.
"""
from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from ocr_local.translation.custody_adapter import ReasonCode
from ocr_local.translation.engines import ENGINE_REGISTRY
from ocr_local.translation.models import EngineCapability
from ocr_local.translation.policy import TenantPolicy
from ocr_local.translation.router import (
    NoEligibleEngineError,
    _is_engine_cached,
    _order_candidates,
    prepare_translation_input,
    select_engine,
    select_engine_for_tenant,
)


def _mock_chain() -> MagicMock:
    return MagicMock()


def _make_fake_engine(
    *,
    engine_id: str,
    is_local: bool = True,
    is_cloud: bool = False,
    quality: str = "standard",
    license_str: str = "Apache-2.0",
    supports_pairs="any",
    accept_kwargs: bool = True,
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

    if accept_kwargs:
        class _FakeEngine:
            capability = cap

            def __init__(self, model_id: str = "", tenant_policy=None):
                self.model_id = model_id
                self.tenant_policy = tenant_policy
    else:
        class _FakeEngine:  # type: ignore[no-redef]
            capability = cap

            def __init__(self):
                self.model_id = ""
                self.tenant_policy = None

    _FakeEngine.__name__ = f"Fake_{engine_id}"
    return _FakeEngine


@contextmanager
def _registered(*engine_classes) -> Iterator[None]:
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


@contextmanager
def _stubbed_cache(cached_pairs: list[tuple[str, str]]) -> Iterator[None]:
    """Stub :func:`list_cached_models` to return the given (engine, model_id) pairs."""
    from ocr_local.translation import cache as cache_mod

    fake_entries = []
    for engine_id, model_id in cached_pairs:
        fake_entries.append(
            cache_mod.CachedModelInfo(
                engine=engine_id,
                model_id=model_id,
                path=cache_mod.DEFAULT_CACHE_DIR / engine_id / model_id,
                size_bytes=0,
                license="Apache-2.0",
                nc_licensed=False,
                pinned=False,
                access_time=0.0,
            )
        )

    with patch.object(cache_mod, "list_cached_models", return_value=fake_entries):
        yield


# ---------------------------------------------------------------------------
# Tenant policy hydration (3)
# ---------------------------------------------------------------------------


def test_get_tenant_policy_returns_default_when_tenant_id_none():
    from ocr_local.translation.policy import get_tenant_policy

    policy = get_tenant_policy(None)
    assert isinstance(policy, TenantPolicy)
    assert policy.tenant_id == "default"
    assert policy.allow_nllb_commercial is True  # default permissive


def test_get_tenant_policy_returns_default_when_django_unavailable(monkeypatch):
    """When Django/jobs.models is not importable, the helper must not crash."""
    import sys

    from ocr_local.translation.policy import get_tenant_policy

    # Force the lazy import to fail by patching sys.modules.
    fake_modules = dict(sys.modules)
    fake_modules.pop("jobs.models", None)
    monkeypatch.setattr(sys, "modules", fake_modules)
    # Build a dummy import-error path -- the helper catches Exception.
    policy = get_tenant_policy("any-tenant")
    assert isinstance(policy, TenantPolicy)


@pytest.mark.skipif(
    importlib.util.find_spec("django") is None,
    reason="Django not installed",
)
def test_get_tenant_policy_hydrates_preferred_engine_ids():
    """Hydration populates both ``allowed_engine_ids`` and ``preferred_engine_ids``."""
    from ocr_local.translation.policy import get_tenant_policy

    fake_row = MagicMock()
    fake_row.preferred_engines = ["opus_a", "opus_b"]
    fake_row.allow_nc_licensed = True
    fake_qs = MagicMock()
    fake_qs.first.return_value = fake_row
    fake_objects = MagicMock()
    fake_objects.filter.return_value = fake_qs

    fake_module = MagicMock()
    fake_module.TranslationTenantConfig.objects = fake_objects

    import sys

    with patch.dict(sys.modules, {"jobs.models": fake_module}):
        policy = get_tenant_policy("acme-corp")

    assert policy.tenant_id == "acme-corp"
    assert policy.allow_nllb_commercial is True
    assert policy.allowed_engine_ids == ["opus_a", "opus_b"]
    assert policy.preferred_engine_ids == ["opus_a", "opus_b"]


# ---------------------------------------------------------------------------
# NC-license filtering via select_engine_for_tenant (2)
# ---------------------------------------------------------------------------


def test_v2_drops_nc_licensed_when_tenant_disallows():
    nllb = _make_fake_engine(
        engine_id="rv2_nllb",
        license_str="CC-BY-NC-4.0",
    )
    apache = _make_fake_engine(
        engine_id="rv2_opus",
        license_str="Apache-2.0",
    )
    fake_policy = TenantPolicy(
        tenant_id="commercial",
        allow_nllb_commercial=False,
    )

    with _registered(nllb, apache), _stubbed_cache(
        [("rv2_nllb", "rv2_nllb"), ("rv2_opus", "rv2_opus")]
    ):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            engine = select_engine_for_tenant(
                text="hello",
                source_lang="en",
                target_lang="ja",
                tenant_id="commercial",
            )
    assert engine.capability.id == "rv2_opus"


def test_v2_keeps_nc_licensed_when_tenant_allows():
    nllb = _make_fake_engine(
        engine_id="rv2_nllb_ok",
        license_str="CC-BY-NC-4.0",
        quality="legal",
    )
    fake_policy = TenantPolicy(
        tenant_id="research",
        allow_nllb_commercial=True,
    )

    with _registered(nllb), _stubbed_cache([("rv2_nllb_ok", "rv2_nllb_ok")]):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            engine = select_engine_for_tenant(
                text="hello",
                source_lang="en",
                target_lang="ja",
                tenant_id="research",
            )
    assert engine.capability.id == "rv2_nllb_ok"


# ---------------------------------------------------------------------------
# Cache-availability filter (3)
# ---------------------------------------------------------------------------


def test_v2_drops_engine_when_model_not_cached_and_no_download():
    only_local = _make_fake_engine(engine_id="rv2_uncached")
    fake_policy = TenantPolicy(tenant_id="t")

    with _registered(only_local), _stubbed_cache([]):  # nothing cached
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            with pytest.raises(NoEligibleEngineError) as exc_info:
                select_engine_for_tenant(
                    text="hello",
                    source_lang="en",
                    target_lang="ja",
                    tenant_id="t",
                    allow_download=False,
                )
    assert any(
        "not cached" in r for r in exc_info.value.filter_reasons
    )


def test_v2_keeps_engine_when_model_is_cached():
    cached = _make_fake_engine(engine_id="rv2_cached")
    fake_policy = TenantPolicy(tenant_id="t")

    with _registered(cached), _stubbed_cache([("rv2_cached", "rv2_cached")]):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            engine = select_engine_for_tenant(
                text="hello",
                source_lang="en",
                target_lang="ja",
                tenant_id="t",
                allow_download=False,
            )
    assert engine.capability.id == "rv2_cached"


def test_is_engine_cached_skips_cloud_engines():
    cloud = _make_fake_engine(
        engine_id="rv2_cloud", is_local=False, is_cloud=True
    )
    fake_policy = TenantPolicy(tenant_id="t", allow_cloud_translation=True)
    ok, reason = _is_engine_cached(cloud, fake_policy, allow_download=False)
    assert ok is True
    assert reason == ""


# ---------------------------------------------------------------------------
# Preferred-engine ranking (3)
# ---------------------------------------------------------------------------


def test_v2_preferred_engine_wins_over_default_quality_ranking():
    """Tenant-preferred standard engine should beat default-ranked legal engine."""
    legal_default = _make_fake_engine(
        engine_id="rv2_legal", quality="legal"
    )
    standard_pref = _make_fake_engine(
        engine_id="rv2_pref_std", quality="standard"
    )
    fake_policy = TenantPolicy(
        tenant_id="t",
        preferred_engine_ids=["rv2_pref_std", "rv2_legal"],
    )

    with _registered(legal_default, standard_pref), _stubbed_cache(
        [("rv2_legal", "rv2_legal"), ("rv2_pref_std", "rv2_pref_std")]
    ):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            engine = select_engine_for_tenant(
                text="hello",
                source_lang="en",
                target_lang="ja",
                tenant_id="t",
            )
    assert engine.capability.id == "rv2_pref_std"


def test_order_candidates_default_when_no_preferred():
    """Without preferred_engine_ids, falls back to local-first + legal>standard ranking."""
    cloud = _make_fake_engine(
        engine_id="oc_cloud", is_local=False, is_cloud=True, quality="legal"
    )
    local_std = _make_fake_engine(
        engine_id="oc_local_std", quality="standard"
    )
    local_legal = _make_fake_engine(
        engine_id="oc_local_legal", quality="legal"
    )
    policy = TenantPolicy(tenant_id="t")
    ordered = _order_candidates([cloud, local_std, local_legal], policy)
    assert ordered[0].capability.id == "oc_local_legal"
    assert ordered[1].capability.id == "oc_local_std"
    assert ordered[2].capability.id == "oc_cloud"


def test_order_candidates_preferred_block_then_default():
    a = _make_fake_engine(engine_id="oc_a", quality="standard")
    b = _make_fake_engine(engine_id="oc_b", quality="legal")
    c = _make_fake_engine(engine_id="oc_c", quality="legal")
    policy = TenantPolicy(
        tenant_id="t",
        preferred_engine_ids=["oc_a"],
    )
    ordered = _order_candidates([c, b, a], policy)
    # oc_a is preferred -> position 0
    assert ordered[0].capability.id == "oc_a"
    # oc_b and oc_c are not preferred -> ranked by default (both legal local).
    assert {ordered[1].capability.id, ordered[2].capability.id} == {"oc_b", "oc_c"}


# ---------------------------------------------------------------------------
# Empty-candidate raises with reasons + emits TENANT_ENGINE_NO_CANDIDATES (2)
# ---------------------------------------------------------------------------


def test_v2_empty_candidates_raises_with_reasons():
    only = _make_fake_engine(
        engine_id="rv2_only_enfr",
        supports_pairs=[("en", "fr")],
    )
    fake_policy = TenantPolicy(tenant_id="t")
    with _registered(only), _stubbed_cache([("rv2_only_enfr", "rv2_only_enfr")]):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            with pytest.raises(NoEligibleEngineError) as exc_info:
                select_engine_for_tenant(
                    text="hello",
                    source_lang="zz",
                    target_lang="qq",
                    tenant_id="t",
                )
    assert "rv2_only_enfr" in " ".join(exc_info.value.filter_reasons)
    assert exc_info.value.tenant_id == "t"


def test_v2_empty_candidates_emits_custody_before_raise():
    only = _make_fake_engine(
        engine_id="rv2_emit_block",
        license_str="CC-BY-NC-4.0",
    )
    fake_policy = TenantPolicy(tenant_id="t", allow_nllb_commercial=False)
    chain = _mock_chain()

    with _registered(only), _stubbed_cache([]):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            with pytest.raises(NoEligibleEngineError):
                select_engine_for_tenant(
                    text="hello",
                    source_lang="en",
                    target_lang="ja",
                    tenant_id="t",
                    custody_chain=chain,
                )

    # First arg of log_event is the event name; second arg is the payload.
    event_calls = [c.args for c in chain.log_event.call_args_list]
    event_names = [args[0] for args in event_calls]
    assert "TENANT_ENGINE_NO_CANDIDATES" in event_names
    # And it should fire BEFORE there's any "TENANT_ENGINE_SELECTED".
    assert "TENANT_ENGINE_SELECTED" not in event_names


# ---------------------------------------------------------------------------
# Successful selection emits TENANT_ENGINE_SELECTED + HYDRATED (2)
# ---------------------------------------------------------------------------


def test_v2_success_emits_tenant_engine_selected_and_hydrated():
    engine_cls = _make_fake_engine(engine_id="rv2_sel_ok")
    fake_policy = TenantPolicy(tenant_id="acme")
    chain = _mock_chain()

    with _registered(engine_cls), _stubbed_cache([("rv2_sel_ok", "rv2_sel_ok")]):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            engine = select_engine_for_tenant(
                text="hello",
                source_lang="en",
                target_lang="ja",
                tenant_id="acme",
                custody_chain=chain,
            )

    assert engine.capability.id == "rv2_sel_ok"
    event_names = [c.args[0] for c in chain.log_event.call_args_list]
    assert "TENANT_POLICY_HYDRATED" in event_names
    assert "TENANT_ENGINE_SELECTED" in event_names

    # Sanity-check the SELECTED payload contains tenant + engine + lang.
    selected_payload = next(
        c.args[1] for c in chain.log_event.call_args_list
        if c.args[0] == "TENANT_ENGINE_SELECTED"
    )
    assert selected_payload["tenant_id"] == "acme"
    assert selected_payload["engine_id"] == "rv2_sel_ok"
    assert selected_payload["source_lang"] == "en"
    assert selected_payload["target_lang"] == "ja"
    assert selected_payload["reason_code"] == str(ReasonCode.TENANT_ENGINE_SELECTED)


def test_v2_passes_model_id_and_tenant_policy_to_engine_constructor():
    engine_cls = _make_fake_engine(engine_id="rv2_kwargs")
    fake_policy = TenantPolicy(tenant_id="t")

    with _registered(engine_cls), _stubbed_cache([("rv2_kwargs", "rv2_kwargs")]):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            engine = select_engine_for_tenant(
                text="hello",
                source_lang="en",
                target_lang="ja",
                tenant_id="t",
            )
    assert engine.model_id == "rv2_kwargs"
    assert engine.tenant_policy is fake_policy


def test_v2_engine_constructor_without_kwargs_is_fallback_compatible():
    """Engines whose __init__ doesn't accept model_id should still be selectable."""
    engine_cls = _make_fake_engine(engine_id="rv2_legacy_init", accept_kwargs=False)
    fake_policy = TenantPolicy(tenant_id="t")

    with _registered(engine_cls), _stubbed_cache([("rv2_legacy_init", "rv2_legacy_init")]):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            engine = select_engine_for_tenant(
                text="hello",
                source_lang="en",
                target_lang="ja",
                tenant_id="t",
            )
    assert engine.capability.id == "rv2_legacy_init"


# ---------------------------------------------------------------------------
# Same-language short-circuit (1)
# ---------------------------------------------------------------------------


def test_v2_same_lang_returns_passthrough_without_filtering():
    chain = _mock_chain()
    engine = select_engine_for_tenant(
        text="hello",
        source_lang="en",
        target_lang="en",
        tenant_id=None,
        custody_chain=chain,
    )
    assert engine.capability.id == "passthrough"


# ---------------------------------------------------------------------------
# Privilege flag (1)
# ---------------------------------------------------------------------------


def test_v2_privilege_flag_drops_cloud_engines():
    cloud = _make_fake_engine(
        engine_id="rv2_cloud_priv", is_local=False, is_cloud=True
    )
    local = _make_fake_engine(engine_id="rv2_local_priv")
    fake_policy = TenantPolicy(tenant_id="t", allow_cloud_translation=True)

    with _registered(cloud, local), _stubbed_cache(
        [("rv2_local_priv", "rv2_local_priv")]
    ):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            engine = select_engine_for_tenant(
                text="hello",
                source_lang="en",
                target_lang="ja",
                tenant_id="t",
                privilege_flag=True,
            )
    # Cloud engine should be filtered out; local survives.
    assert engine.capability.id == "rv2_local_priv"


# ---------------------------------------------------------------------------
# Glossary preprocessing (3)
# ---------------------------------------------------------------------------


def test_prepare_translation_input_returns_unchanged_when_no_entries():
    """No glossary entries -> text passes through, hits empty."""
    text = "Hello world"
    out_text, hits = prepare_translation_input(
        tenant_id=None,
        text=text,
        source_lang="en",
        target_lang="fr",
    )
    assert out_text == text
    assert hits == []


def test_prepare_translation_input_applies_entries_and_emits_custody():
    from ocr_local.translation.glossary import GlossaryEntry

    entries = [
        GlossaryEntry(
            term_id="g1",
            tenant_id="t",
            source_term="Acme",
            target_term="ACME Inc.",
            source_lang="en",
            target_lang="fr",
        )
    ]

    chain = _mock_chain()
    with patch(
        "ocr_local.translation.glossary.load_tenant_glossary",
        return_value=entries,
    ):
        out_text, hits = prepare_translation_input(
            tenant_id="t",
            text="Acme reports.",
            source_lang="en",
            target_lang="fr",
            custody_chain=chain,
        )
    assert "ACME Inc." in out_text
    assert len(hits) == 1
    # Custody emitted exactly once for the GLOSSARY_APPLIED event.
    event_names = [c.args[0] for c in chain.log_event.call_args_list]
    assert event_names.count("GLOSSARY_APPLIED") == 1


def test_prepare_translation_input_no_emit_when_no_hits():
    from ocr_local.translation.glossary import GlossaryEntry

    entries = [
        GlossaryEntry(
            term_id="g1",
            tenant_id="t",
            source_term="DOES_NOT_OCCUR",
            target_term="X",
            source_lang="en",
            target_lang="fr",
        )
    ]
    chain = _mock_chain()
    with patch(
        "ocr_local.translation.glossary.load_tenant_glossary",
        return_value=entries,
    ):
        out_text, hits = prepare_translation_input(
            tenant_id="t",
            text="Hello world",
            source_lang="en",
            target_lang="fr",
            custody_chain=chain,
        )
    assert out_text == "Hello world"
    assert hits == []
    event_names = [c.args[0] for c in chain.log_event.call_args_list]
    assert "GLOSSARY_APPLIED" not in event_names


# ---------------------------------------------------------------------------
# REST endpoint integration (3)
# ---------------------------------------------------------------------------


pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture
def api_client():
    """Build an isolated FastAPI app with only the translation router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers.translation import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def test_rest_legacy_path_when_use_tenant_router_false(api_client):
    """Legacy stub path returned when ``use_tenant_router=False``."""
    payload = {
        "source_text": "hello",
        "target_languages": ["fr"],
        "tenant_id": "default",
        "use_tenant_router": False,
    }
    response = api_client.post("/api/v1/translation/jobs", json=payload)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["selected_engines"] is None  # legacy path doesn't populate
    assert "Wave M1 stub" in body["message"]


def test_rest_tenant_router_v2_returns_selected_engine(api_client):
    """``use_tenant_router=True`` invokes router_v2 and surfaces the engine."""
    engine_cls = _make_fake_engine(engine_id="rest_v2_engine")
    fake_policy = TenantPolicy(tenant_id="acme")

    with _registered(engine_cls), _stubbed_cache(
        [("rest_v2_engine", "rest_v2_engine")]
    ):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            payload = {
                "source_text": "Hello world",
                "source_language": "en",
                "target_languages": ["ja"],
                "tenant_id": "acme",
                "use_tenant_router": True,
            }
            response = api_client.post("/api/v1/translation/jobs", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["selected_engines"] == {"ja": "rest_v2_engine"}


def test_rest_tenant_router_no_eligible_engine_records_failure(api_client):
    fake_policy = TenantPolicy(tenant_id="t")

    # No engines registered besides the package defaults; src=zz tgt=qq has
    # no support so router_v2 raises NoEligibleEngineError.
    with _stubbed_cache([]):
        with patch(
            "ocr_local.translation.router.get_tenant_policy",
            return_value=fake_policy,
        ):
            payload = {
                "source_text": "hello",
                "source_language": "zz",
                "target_languages": ["qq"],
                "tenant_id": "t",
                "use_tenant_router": True,
            }
            response = api_client.post("/api/v1/translation/jobs", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["selected_engines"] is None
    assert "qq" in body["message"]


# ---------------------------------------------------------------------------
# Backward-compatibility (legacy select_engine still works) (2)
# ---------------------------------------------------------------------------


def test_legacy_select_engine_signature_unchanged():
    """The legacy ``select_engine(req, tenant, chain)`` signature must still work."""
    from ocr_local.translation.models import TranslationRequest

    engine_cls = _make_fake_engine(engine_id="bw_compat_engine")
    with _registered(engine_cls):
        engine = select_engine(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t"),
            _mock_chain(),
        )
    assert engine.capability.id == "bw_compat_engine"


def test_legacy_select_engine_does_not_emit_v2_events():
    """Legacy path must not emit any of the new tenant-aware events."""
    from ocr_local.translation.models import TranslationRequest

    engine_cls = _make_fake_engine(engine_id="bw_no_emit")
    chain = _mock_chain()
    with _registered(engine_cls):
        select_engine(
            TranslationRequest(src_lang="en", tgt_lang="ja", tenant_id="t"),
            TenantPolicy(tenant_id="t"),
            chain,
        )
    event_names = [c.args[0] for c in chain.log_event.call_args_list]
    assert "TENANT_POLICY_HYDRATED" not in event_names
    assert "TENANT_ENGINE_SELECTED" not in event_names
    assert "TENANT_ENGINE_NO_CANDIDATES" not in event_names
