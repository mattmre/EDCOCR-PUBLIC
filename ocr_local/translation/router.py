"""Translation engine router.

Selects the best translation engine for a given request and tenant
policy, with a hard-block path for privilege-flagged documents.

CRITICAL: every reject path emits the corresponding ``TRANSLATION_REJECTED``
custody event *before* raising :class:`PolicyDenied`.  The audit panel
requires that the rejection be visible in the custody chain even when
callers swallow the exception.

Wave M2 PR B14 (router_v2) adds the tenant-aware entry point
:func:`select_engine_for_tenant` which:

* Hydrates :class:`TenantPolicy` via :func:`get_tenant_policy` (lazy).
* Filters NC-licensed engines per ``policy.allow_nllb_commercial``.
* Filters by cache availability (engine model not present, downloads
  disabled or air-gapped) -- engines whose weights cannot be loaded
  are dropped from the candidate list.
* Re-ranks remaining candidates by ``policy.preferred_engine_ids`` first,
  then the default local/quality ordering.
* Emits ``TENANT_POLICY_HYDRATED``, ``TENANT_ENGINE_SELECTED``, or
  ``TENANT_ENGINE_NO_CANDIDATES`` custody events with full filter reasons.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ocr_local.translation.custody_adapter import (
    ReasonCode,
    emit_tenant_engine_no_candidates,
    emit_tenant_engine_selected,
    emit_tenant_policy_hydrated,
    emit_translation_rejected,
)
from ocr_local.translation.policy import (
    PolicyDenied,
    TenantPolicy,
    compute_policy_hash,
    get_tenant_policy,
)

if TYPE_CHECKING:
    from ocr_local.features.custody import CustodyChain
    from ocr_local.translation.engines.base import TranslationEngine
    from ocr_local.translation.models import EngineCapability, TranslationRequest


class NoEligibleEngineError(RuntimeError):
    """Raised by :func:`select_engine_for_tenant` when no candidate engine
    survives tenant-aware filtering.

    The custody event ``TENANT_ENGINE_NO_CANDIDATES`` is emitted BEFORE
    this exception is raised (gotcha #87) so the audit trail captures
    the rejection even when callers swallow the exception.
    """

    def __init__(self, tenant_id: str, message: str, filter_reasons: list[str]) -> None:
        self.tenant_id = tenant_id
        self.filter_reasons = list(filter_reasons)
        super().__init__(message)


def _deployment_env() -> str:
    """Resolve deployment env at call time (test-friendly)."""
    return os.environ.get("DEPLOYMENT_ENV", "development")


def select_engine(
    req: "TranslationRequest",
    tenant: TenantPolicy,
    chain: "CustodyChain",
) -> "TranslationEngine":
    """Select the best available engine for ``req`` under ``tenant``.

    Reject paths emit ``TRANSLATION_REJECTED`` BEFORE raising.
    """
    from ocr_local.translation.engines import get_engine

    # Same-language requests always passthrough -- no policy check needed
    # (no translation actually happens, so no cloud/privilege exposure).
    if req.src_lang == req.tgt_lang:
        return get_engine("passthrough")()

    candidates = _rank_candidates(req, tenant)

    if not candidates:
        emit_translation_rejected(
            chain,
            reason_code=ReasonCode.UNSUPPORTED_LANGUAGE,
            message=f"No engine supports {req.src_lang}->{req.tgt_lang}",
            tenant_id=req.tenant_id,
        )
        raise PolicyDenied(
            ReasonCode.UNSUPPORTED_LANGUAGE,
            f"No engine supports {req.src_lang}->{req.tgt_lang}",
        )

    engine_cls = candidates[0]
    cap: EngineCapability = engine_cls.capability

    # Privilege hard-block: privilege_flag forbids cloud engines, period.
    # This check comes BEFORE the tenant cloud-policy check so the more
    # specific rejection reason wins.
    if cap.is_cloud and req.privilege_flag:
        emit_translation_rejected(
            chain,
            reason_code=ReasonCode.PRIVILEGE_BLOCKED,
            engine_id=cap.id,
            message="Privilege-flagged document cannot use cloud translation engine",
            tenant_id=req.tenant_id,
        )
        raise PolicyDenied(
            ReasonCode.PRIVILEGE_BLOCKED,
            "Privilege-flagged document cannot use cloud translation engine",
        )

    # Tenant policy: cloud blocked when not explicitly allowed.
    if cap.is_cloud and not tenant.allow_cloud_translation:
        emit_translation_rejected(
            chain,
            reason_code=ReasonCode.TENANT_POLICY,
            engine_id=cap.id,
            message="Tenant policy prohibits cloud translation",
            tenant_id=req.tenant_id,
        )
        raise PolicyDenied(
            ReasonCode.TENANT_POLICY,
            "Tenant policy prohibits cloud translation",
        )

    return engine_cls()


def _rank_candidates(
    req: "TranslationRequest",
    tenant: TenantPolicy,
) -> list[type["TranslationEngine"]]:
    """Return engine classes ordered by preference for ``req`` + ``tenant``."""
    from ocr_local.translation.engines import ENGINE_REGISTRY

    deployment_env = _deployment_env()
    candidates: list[type["TranslationEngine"]] = []

    for engine_id, engine_cls in ENGINE_REGISTRY.items():
        cap: EngineCapability = engine_cls.capability

        # Passthrough is only for same-language; never a translation
        # candidate when src != tgt.
        if engine_id == "passthrough":
            continue

        # Tenant allow/block lists.
        if engine_id in tenant.blocked_engine_ids:
            continue
        if tenant.allowed_engine_ids and engine_id not in tenant.allowed_engine_ids:
            continue

        # Local-only tenant policy excludes cloud engines.
        if tenant.require_local_only and cap.is_cloud:
            continue

        # Language pair support.
        if not _supports_pair(cap, req.src_lang, req.tgt_lang):
            continue

        # NLLB-derived engines use a non-commercial license -- skip when
        # tenant cannot use NC-licensed weights.
        if not tenant.allow_nllb_commercial and "nc" in cap.license.lower():
            continue

        # Gemini AI Studio is not allowed in production.
        if deployment_env == "production" and engine_id == "cloud_gemini_ai_studio":
            continue

        candidates.append(engine_cls)

    return _order_candidates(candidates, tenant)


def _order_candidates(
    candidates: list[type["TranslationEngine"]],
    tenant: TenantPolicy,
) -> list[type["TranslationEngine"]]:
    """Order ``candidates`` by tenant preference first, then default ranking.

    Default ranking: local-first, then ``legal > standard > draft``.
    Tenant ``preferred_engine_ids`` (when non-empty) overrides the default
    by giving listed engines a stable rank index 0..N-1; engines not in
    the list keep their default ordering after the preferred block.
    """
    _quality_order = {"legal": 0, "standard": 1, "draft": 2}
    preferred = list(getattr(tenant, "preferred_engine_ids", []) or [])
    pref_index = {eid: i for i, eid in enumerate(preferred)}

    def sort_key(c: type["TranslationEngine"]) -> tuple:
        cap = c.capability
        # Preferred engines come first (stable order from the list);
        # everything else gets the same neutral preference rank so the
        # default local/quality ordering wins.
        pref_rank = pref_index.get(cap.id, len(preferred) + 1)
        return (
            pref_rank,
            0 if cap.is_local else 1,
            _quality_order.get(cap.quality_class, 99),
        )

    return sorted(candidates, key=sort_key)


# ---------------------------------------------------------------------------
# Wave M2 PR B14 -- tenant-aware router (router_v2)
# ---------------------------------------------------------------------------


def _engine_default_model_id(engine_cls: type["TranslationEngine"]) -> str:
    """Return the default model_id for ``engine_cls`` -- the engine id.

    Engines may carry multiple model variants behind the same adapter
    (e.g. opus_en_fr vs opus_en_de), but for cache-availability filtering
    we treat the engine_id itself as the canonical model_id.  Tenants
    that need a specific model variant should pin via ``preferred_engine_ids``
    in conjunction with explicit allow-list semantics.
    """
    return engine_cls.capability.id


def _is_engine_cached(
    engine_cls: type["TranslationEngine"],
    tenant: TenantPolicy,
    *,
    allow_download: bool,
) -> tuple[bool, str]:
    """Probe cache availability for ``engine_cls``.

    Returns ``(eligible, reason)``.  Cloud engines and the passthrough
    engine are always eligible -- they don't depend on a local cache.
    Local engines are checked against :func:`list_cached_models`; if the
    requested model isn't present and ``allow_download`` is False (or the
    pipeline is air-gapped), the engine is dropped with a reason string.

    Catches :class:`ModelNotCachedError` from a probe-style
    :func:`get_translation_model_path` call when the listing path is
    unreliable (no cache_state.json yet).  Any other exception is treated
    as ineligible with the exception message as the reason.
    """
    cap = engine_cls.capability
    if cap.is_cloud or cap.id == "passthrough":
        return True, ""
    if not cap.is_local:
        # Engines that are neither cloud nor local (rare) -- skip cache check.
        return True, ""

    model_id = _engine_default_model_id(engine_cls)
    try:
        from ocr_local.translation.cache import (
            ModelNotCachedError,
            list_cached_models,
        )
    except Exception as exc:  # pragma: no cover -- cache module unavailable
        return False, f"cache module import failed: {exc}"

    try:
        cached = list_cached_models()
    except Exception as exc:
        # Non-fatal -- if we can't read cache state, assume not cached.
        return False, f"cache state read failed: {exc}"

    has_match = any(
        info.engine == cap.id and info.model_id == model_id for info in cached
    )
    if has_match:
        return True, ""

    if not allow_download:
        return False, f"model {cap.id}/{model_id} not cached and downloads disabled"

    # ``allow_download=True`` -- probe via get_translation_model_path so the
    # download lifecycle, manifest verify, NC-license filter all run.  If
    # the probe raises, drop the candidate.
    try:
        from ocr_local.translation.cache import get_translation_model_path

        get_translation_model_path(
            cap.id,
            model_id,
            allow_download=True,
            tenant_policy=tenant,
        )
        return True, ""
    except ModelNotCachedError as exc:
        return False, f"download blocked: {exc}"
    except PolicyDenied as exc:
        # NC-license block surfaces here when downloading fresh weights.
        return False, f"policy denied: {exc}"
    except Exception as exc:  # pragma: no cover -- defensive
        return False, f"cache probe failed: {exc}"


def select_engine_for_tenant(
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    tenant_id: str | None,
    allow_download: bool = False,
    custody_chain: "CustodyChain | None" = None,
    privilege_flag: bool = False,
) -> "TranslationEngine":
    """Tenant-aware engine selection for Plan B Wave M2.

    Hydrates :class:`TenantPolicy` from the persisted config, filters
    candidates by NC-license + cache availability, ranks by tenant
    preference, and emits the corresponding custody events.

    Returns an instantiated :class:`TranslationEngine`.  Raises
    :class:`NoEligibleEngineError` when the candidate list is empty
    after filtering -- the ``TENANT_ENGINE_NO_CANDIDATES`` custody event
    is emitted BEFORE the raise (gotcha #87) so the audit log always
    captures the rejection.

    Same-language requests short-circuit to the passthrough engine
    without consulting the registry, matching the legacy
    :func:`select_engine` behaviour.
    """
    from ocr_local.translation.engines import ENGINE_REGISTRY, get_engine

    chain = custody_chain  # may be None -- helpers no-op when so

    # Hydrate tenant policy and emit the hydration event up front so the
    # audit trail reflects which policy version drove the routing call.
    tenant = get_tenant_policy(tenant_id)
    if chain is not None:
        try:
            policy_hash = compute_policy_hash(tenant)
            emit_tenant_policy_hydrated(
                chain,
                tenant_id=tenant.tenant_id,
                policy_hash=policy_hash,
            )
        except Exception:
            # Custody emission must never block routing.
            pass

    # Same-language short-circuit -- no policy check, no cache filter.
    if source_lang == target_lang:
        engine_cls = get_engine("passthrough")
        return engine_cls()

    deployment_env = _deployment_env()
    filter_reasons: list[str] = []
    raw_candidates: list[type["TranslationEngine"]] = []

    for engine_id, engine_cls in ENGINE_REGISTRY.items():
        cap: EngineCapability = engine_cls.capability

        if engine_id == "passthrough":
            continue

        if engine_id in tenant.blocked_engine_ids:
            filter_reasons.append(f"{engine_id}: tenant block list")
            continue
        if tenant.allowed_engine_ids and engine_id not in tenant.allowed_engine_ids:
            filter_reasons.append(f"{engine_id}: not in tenant allow list")
            continue

        if tenant.require_local_only and cap.is_cloud:
            filter_reasons.append(f"{engine_id}: tenant require_local_only")
            continue

        if not _supports_pair(cap, source_lang, target_lang):
            filter_reasons.append(
                f"{engine_id}: unsupported pair {source_lang}->{target_lang}"
            )
            continue

        if not tenant.allow_nllb_commercial and "nc" in cap.license.lower():
            filter_reasons.append(
                f"{engine_id}: NC license blocked (allow_nllb_commercial=False)"
            )
            continue

        if deployment_env == "production" and engine_id == "cloud_gemini_ai_studio":
            filter_reasons.append(f"{engine_id}: not allowed in production")
            continue

        if privilege_flag and cap.is_cloud:
            filter_reasons.append(
                f"{engine_id}: privilege-flagged document blocks cloud engine"
            )
            continue

        if cap.is_cloud and not tenant.allow_cloud_translation:
            filter_reasons.append(
                f"{engine_id}: tenant policy prohibits cloud translation"
            )
            continue

        # Cache-availability probe -- only meaningful for local engines.
        ok, reason = _is_engine_cached(
            engine_cls, tenant, allow_download=allow_download
        )
        if not ok:
            filter_reasons.append(f"{engine_id}: {reason}")
            continue

        raw_candidates.append(engine_cls)

    if not raw_candidates:
        message = (
            f"No eligible translation engine for tenant={tenant.tenant_id!r} "
            f"src={source_lang!r} tgt={target_lang!r}"
        )
        if chain is not None:
            try:
                emit_tenant_engine_no_candidates(
                    chain,
                    tenant_id=tenant.tenant_id,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    filter_reasons=filter_reasons,
                )
            except Exception:
                pass
        raise NoEligibleEngineError(tenant.tenant_id, message, filter_reasons)

    ranked = _order_candidates(raw_candidates, tenant)
    engine_cls = ranked[0]
    selected_model_id = _engine_default_model_id(engine_cls)

    if chain is not None:
        try:
            emit_tenant_engine_selected(
                chain,
                tenant_id=tenant.tenant_id,
                engine_id=engine_cls.capability.id,
                model_id=selected_model_id,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        except Exception:
            pass

    # Instantiate.  Engines that resolve a model via the cache module
    # accept ``model_id`` + ``tenant_policy`` kwargs; pass them when the
    # constructor accepts them, fall back to no-arg construction otherwise.
    try:
        return engine_cls(model_id=selected_model_id, tenant_policy=tenant)
    except TypeError:
        return engine_cls()


def prepare_translation_input(
    *,
    tenant_id: str | None,
    text: str,
    source_lang: str,
    target_lang: str,
    span_index: int = 0,
    custody_chain: "CustodyChain | None" = None,
    glossary_id: str = "tenant_default",
):
    """Apply tenant glossary to ``text`` BEFORE engine selection.

    Glossary preprocessing influences downstream engine choice (e.g.
    length-based latency tier) so it must run before
    :func:`select_engine_for_tenant`.  Returns ``(modified_text, hits)``.

    When ``tenant_id`` is None the glossary lookup short-circuits to an
    empty list and the original text is returned unchanged.

    Emits a single ``GLOSSARY_APPLIED`` custody event per call when
    ``hits`` is non-empty.  When the glossary entry list is empty or no
    entries match, no custody event is emitted (callers can rely on
    ``len(hits) > 0`` as the emission predicate).
    """
    from ocr_local.translation.glossary import (
        apply_glossary,
        emit_glossary_applied_for_span,
        load_tenant_glossary,
    )

    entries = load_tenant_glossary(tenant_id, source_lang, target_lang)
    modified_text, hits = apply_glossary(text, entries, span_index=span_index)

    if hits and custody_chain is not None:
        try:
            import hashlib

            # Glossary content hash so reviewers can prove which set of
            # entries drove the modifications.
            payload = "\n".join(
                f"{e.term_id}|{e.source_term}|{e.target_term}|{e.priority}"
                for e in entries
            )
            ghash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            emit_glossary_applied_for_span(
                custody_chain,
                glossary_id=glossary_id,
                glossary_hash=ghash,
                hits=hits,
            )
        except Exception:
            # Custody emission must never block translation.
            pass

    return modified_text, hits


def _supports_pair(cap: "EngineCapability", src: str, tgt: str) -> bool:
    if cap.supports_pairs == "any":
        return True
    if isinstance(cap.supports_pairs, str):
        return False
    return (src, tgt) in cap.supports_pairs or (src, "*") in cap.supports_pairs
