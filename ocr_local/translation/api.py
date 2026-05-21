"""Translation document facade -- orchestrates span routing -> engine -> sidecar -> custody.

This module is the high-level entry point used by the OCR pipeline
assembler thread (see ``ocr_gpu_async.py`` finalize block).  It is
deliberately fail-open: any exception during translation is logged and
results in an empty list being returned, so a translation failure can
never fail the OCR job itself.

The facade reads per-span language metadata from ``page_data_snap``
when Plan A (per-span language detection) has populated it, and falls
back to a document-level detected language otherwise.  When no
language information is available at all, the source language defaults
to ``"en"`` and ``language_source="default_fallback"`` is recorded in
the resulting ``DocumentTranslation.processing`` block.

Engine selection delegates to :func:`ocr_local.translation.router.select_engine`.
When no engine satisfies the request (e.g. on the SDK CI lane where only
``passthrough`` is registered), the facade falls back to passthrough so
the sidecar shape is still produced and downstream consumers can wire
against the contract.  The fallback is recorded as
``engine.id == "passthrough"`` and is visible in the sidecar.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from ocr_local.document_bundle import build_document_bundle
from ocr_local.translation.custody_adapter import emit_translation_applied
from ocr_local.translation.external_client import (
    DEFAULT_TRANSLATION_SERVICE_URL,
    TranslationServiceClient,
)
from ocr_local.translation.engines import get_engine
from ocr_local.translation.metrics import record_translation_chars
from ocr_local.translation.models import (
    DocumentTranslation,
    PageTranslation,
    SpanTranslation,
    TranslationRequest,
)
from ocr_local.translation.policy import (
    PolicyDenied,
    compute_policy_hash,
    load_tenant_policy,
)
from ocr_local.translation.readiness import external_translation_readiness
from ocr_local.translation.router import (
    NoEligibleEngineError,
    prepare_translation_input,
    select_engine,
    select_engine_for_tenant,
)
from ocr_local.translation.segmentation import segment_to_sentences

if TYPE_CHECKING:
    from ocr_local.features.custody import CustodyChain
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

__all__ = ["translate_document"]


def translate_document(
    doc_path: str,
    target_languages: list[str],
    tenant_id: str,
    *,
    page_data_snap: dict | None = None,
    custody_chain: "CustodyChain | None" = None,
    output_dir: str = "EXPORT/TRANSLATION",
    config: "PipelineConfig | None" = None,
) -> list[DocumentTranslation]:
    """Translate a document to one or more target languages.

    Reads per-span language metadata from ``page_data_snap['languages']``
    (Plan A output).  Falls back to a document-level language hint when
    Plan A metadata is absent.

    Returns a list of :class:`DocumentTranslation` -- one per target
    language.  Never raises: errors are logged and an empty list is
    returned (fail-open).
    """
    readiness = _external_translation_preflight_status(
        config,
        page_data_snap=page_data_snap,
        target_languages=target_languages,
    )
    if readiness.ready:
        try:
            return _translate_document_external(
                doc_path=doc_path,
                target_languages=target_languages,
                tenant_id=tenant_id,
                page_data_snap=page_data_snap,
                custody_chain=custody_chain,
                config=config,
            )
        except Exception as exc:
            logger.warning(
                "External translation service failed for %s; falling back to "
                "in-repo translation: %s",
                doc_path,
                exc,
            )
    elif readiness.enabled:
        logger.warning(
            "External translation dispatch disabled by readiness preflight for %s: %s",
            doc_path,
            readiness.message,
        )

    results: list[DocumentTranslation] = []

    # Load tenant policy once -- it doesn't change per target language.
    try:
        tenant = load_tenant_policy(tenant_id)
        policy_hash = compute_policy_hash(tenant)
    except Exception as exc:
        logger.warning(
            "Translation: failed to load tenant policy for %s: %s",
            tenant_id, exc,
        )
        return results

    for tgt_lang in target_languages:
        try:
            doc = _translate_to_language(
                doc_path=doc_path,
                tgt_lang=tgt_lang,
                tenant=tenant,
                tenant_id=tenant_id,
                policy_hash=policy_hash,
                page_data_snap=page_data_snap,
                custody_chain=custody_chain,
                output_dir=output_dir,
                config=config,
            )
            results.append(doc)
        except Exception as exc:
            logger.warning(
                "Translation to %s failed for %s: %s",
                tgt_lang, doc_path, exc,
            )
            # fail-open: OCR job continues; sidecar omitted for this language.
    return results


def _should_prefer_external_translation(config: "PipelineConfig | None") -> bool:
    """Return True when callers opt into the external translation service."""

    env_value = os.environ.get("EDC_TRANSLATION_PREFER_EXTERNAL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    config_value = getattr(config, "translation_prefer_external_service", False)
    return _truthy_config_value(config_value) or env_value


def _external_translation_preflight_status(
    config: "PipelineConfig | None",
    *,
    page_data_snap: dict | None,
    target_languages: list[str],
):
    """Run the external readiness preflight when preference is enabled."""

    if not _should_prefer_external_translation(config):
        return external_translation_readiness(config)
    source_language, _source = _resolve_source_language(page_data_snap)
    target_language = target_languages[0] if target_languages else "en"
    return external_translation_readiness(
        config,
        source_language=source_language,
        target_language=target_language,
    )


def _translate_document_external(
    *,
    doc_path: str,
    target_languages: list[str],
    tenant_id: str,
    page_data_snap: dict | None,
    custody_chain: "CustodyChain | None",
    config: "PipelineConfig | None",
) -> list[DocumentTranslation]:
    """Translate through EDC_TRANSLATION and preserve legacy return shape."""

    if not target_languages:
        return []
    tenant = load_tenant_policy(tenant_id)
    policy_hash = compute_policy_hash(tenant)
    document_bundle = _build_external_document_bundle(
        doc_path=doc_path,
        page_data_snap=page_data_snap,
        custody_chain=custody_chain,
        config=config,
        tenant_policy_hash=policy_hash,
    )
    client = _external_translation_client(config)
    provider_id = _external_provider_id(config)

    results: list[DocumentTranslation] = []
    for target_language in target_languages:
        translation_bundle = client.translate_bundle(
            document_bundle,
            target_language=target_language,
            provider_id=provider_id,
        )
        results.append(
            _document_translation_from_bundle(
                translation_bundle,
                source_file=os.path.basename(doc_path),
                tenant_id=tenant_id,
                tenant_policy_hash=policy_hash,
            )
        )
    return results


def _build_external_document_bundle(
    *,
    doc_path: str,
    page_data_snap: dict | None,
    custody_chain: "CustodyChain | None",
    config: "PipelineConfig | None",
    tenant_policy_hash: str,
) -> dict:
    text_by_page = _external_text_by_page(page_data_snap)
    src_lang, language_source = _resolve_source_language(page_data_snap)
    spans: list[dict] = []
    for page_num, page_text in text_by_page.items():
        if not page_text:
            continue
        try:
            page_int = int(page_num)
        except (TypeError, ValueError):
            page_int = 1
        for segment in segment_to_sentences(str(page_text), src_lang):
            spans.append(
                {
                    "span_id": f"p{page_int}_{segment['span_id']}",
                    "page_number": page_int,
                    "text": segment["text"],
                    "bbox": segment["bbox"],
                    "language": src_lang,
                }
            )

    return build_document_bundle(
        document_id=hashlib.sha256(doc_path.encode("utf-8")).hexdigest()[:16],
        source_file_name=os.path.basename(doc_path),
        spans=spans,
        language_metadata={
            "primary_language": src_lang,
            "detected_languages": [src_lang],
            "source": f"ocr_local_external_adapter:{language_source}",
        },
        ocr_engine_metadata={
            "engine_id": "ocr_local",
            "engine_version": str(
                getattr(config, "pipeline_version", "external_adapter")
            ),
        },
        custody_chain_head=_external_custody_head(custody_chain),
        privilege_flags={
            "privileged": bool(
                page_data_snap and page_data_snap.get("privilege_flag", False)
            )
        },
        tenant_policy_hash=tenant_policy_hash,
    )


def _truthy_config_value(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _external_translation_client(
    config: "PipelineConfig | None",
) -> TranslationServiceClient:
    url = getattr(config, "translation_external_service_url", None) or os.environ.get(
        "EDC_TRANSLATION_URL",
        DEFAULT_TRANSLATION_SERVICE_URL,
    )
    timeout = getattr(config, "translation_external_timeout_seconds", None)
    if timeout is None:
        timeout = float(os.environ.get("EDC_TRANSLATION_TIMEOUT_SECONDS", "30"))
    return TranslationServiceClient(
        str(url),
        api_key=os.environ.get("EDC_TRANSLATION_API_KEY"),
        timeout=float(timeout),
    )


def _external_provider_id(config: "PipelineConfig | None") -> str:
    return str(
        getattr(config, "translation_external_provider_id", "")
        or os.environ.get("EDC_TRANSLATION_PROVIDER_ID", "passthrough")
    )


def _external_text_by_page(page_data_snap: dict | None) -> dict:
    if not page_data_snap:
        return {}
    return page_data_snap.get("text_by_page") or page_data_snap.get("texts") or {}


def _external_custody_head(custody_chain: "CustodyChain | None") -> str:
    if custody_chain is not None:
        for attr in ("chain_head", "head", "current_hash"):
            value = getattr(custody_chain, attr, None)
            if value:
                return str(value)
    return "n/a"


def _document_translation_from_bundle(
    translation_bundle: dict,
    *,
    source_file: str,
    tenant_id: str,
    tenant_policy_hash: str,
) -> DocumentTranslation:
    grouped: dict[int, list[SpanTranslation]] = defaultdict(list)
    for span in translation_bundle.get("translated_spans", []):
        page_number = int(span.get("page_number", 1))
        grouped[page_number].append(
            SpanTranslation(
                span_id=span["span_id"],
                source_text=span["source_text"],
                target_text=span["translated_text"],
                source_bbox=list(span["source_bbox"]),
                source_bboxes=[list(bbox) for bbox in span["source_bboxes"]],
                source_language=span["source_language"],
                target_language=span["target_language"],
                confidence=float(span["confidence"]),
                quality_score=span.get("quality_score"),
                engine_id=span["engine_id"],
                glossary_hits=list(span.get("glossary_hits", [])),
            )
        )
    pages = [
        PageTranslation(page_num=page_num, spans=spans)
        for page_num, spans in sorted(grouped.items())
    ]
    provider = translation_bundle["engine_provider"]
    provenance = translation_bundle["model_provenance"]
    quality = translation_bundle["quality_scores"]
    return DocumentTranslation(
        schema_version="1.0",
        document_id=translation_bundle["document_id"],
        source_file=source_file,
        source_language=pages[0].spans[0].source_language if pages else "und",
        target_language=translation_bundle["target_language"],
        certified=bool(translation_bundle["certified"]),
        engine={
            "id": provider["id"],
            "is_local": provider["is_local"],
            "license": provider["license"],
            "provider_retention_class": provider["provider_retention_class"],
            "weights_sha256": provenance.get("weights_sha256", "unknown"),
        },
        quality={
            "mean_score": quality["mean_score"],
            "below_threshold_count": quality["below_threshold_count"],
            "quality_class": quality["quality_class"],
        },
        pages=pages,
        stats={
            "total_chars": sum(
                len(span.source_text) for page in pages for span in page.spans
            ),
            "total_spans": sum(len(page.spans) for page in pages),
            "page_count": len(pages),
            "source_bundle_sha256": translation_bundle["source_bundle_sha256"],
        },
        custody={
            "chain_head": translation_bundle["custody_chain_head"],
            "clock_source": "external_service",
            "source_ocr_sha256": translation_bundle["source_ocr_sha256"],
            "tenant_policy_hash": tenant_policy_hash,
            "tenant_id": tenant_id,
        },
        processing={
            "pipeline_version": "external-edc-translation",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "enable_translation": True,
            "translation_service": "external",
        },
    )


def _translate_to_language(
    doc_path,
    tgt_lang,
    tenant,
    tenant_id,
    policy_hash,
    page_data_snap,
    custody_chain,
    output_dir,
    config,
) -> DocumentTranslation:
    """Internal: translate one target language.  May raise -- caller wraps."""
    # Determine source language from Plan A metadata or fallback.
    src_lang, language_source = _resolve_source_language(page_data_snap)

    privilege_flag = bool(
        page_data_snap and page_data_snap.get("privilege_flag", False)
    )

    req = TranslationRequest(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        tenant_id=tenant_id,
        privilege_flag=privilege_flag,
    )

    # Router needs a custody chain; fall back to a no-op chain when caller
    # didn't provide one (custody events are still no-ops, not errors).
    chain_for_router = (
        custody_chain if custody_chain is not None else _make_noop_chain()
    )
    # Wave M2 PR B14 -- prefer tenant-aware router when a real tenant_id is
    # supplied (i.e. not the legacy "default" sentinel).  Falls back to the
    # legacy router on NoEligibleEngineError so existing single-tenant
    # deployments keep working unchanged.
    use_v2 = bool(tenant_id) and tenant_id != "default"
    engine = _instantiate_engine(
        req,
        tenant,
        chain_for_router,
        use_tenant_router=use_v2,
        privilege_flag=privilege_flag,
    )

    pages_translated: list[PageTranslation] = []
    total_chars = 0

    if page_data_snap and "text_by_page" in page_data_snap:
        text_by_page = page_data_snap["text_by_page"] or {}
        for page_num, page_text in text_by_page.items():
            if not page_text:
                continue
            # Apply glossary preprocessing BEFORE segmentation so the
            # downstream engine sees the tenant-overridden terminology.
            try:
                modified_text, _hits = prepare_translation_input(
                    tenant_id=tenant_id,
                    text=page_text,
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    custody_chain=custody_chain,
                )
            except Exception as exc:
                logger.debug(
                    "Translation: glossary prep failed for page %s (non-fatal): %s",
                    page_num, exc,
                )
                modified_text = page_text
            segs = segment_to_sentences(modified_text, src_lang)
            if not segs:
                continue
            spans_out = engine.translate_spans(segs, src_lang, tgt_lang)
            total_chars += sum(len(s["text"]) for s in segs)
            try:
                page_int = int(page_num)
            except (TypeError, ValueError):
                page_int = 0
            pages_translated.append(
                PageTranslation(page_num=page_int, spans=list(spans_out))
            )

    if custody_chain is not None:
        try:
            emit_translation_applied(
                custody_chain,
                engine_id=engine.capability.id,
                src=src_lang,
                tgt=tgt_lang,
                span_count=sum(len(p.spans) for p in pages_translated),
                char_count=total_chars,
                tenant_id=tenant_id,
                model_id=engine.capability.id,
                weights_sha256=engine.model_provenance().get(
                    "weights_sha256", "unknown"
                ),
                language_source=language_source,
                tenant_policy_hash=policy_hash,
            )
        except Exception:
            logger.debug(
                "Translation: custody emit failed (non-fatal)", exc_info=True,
            )

    try:
        record_translation_chars(tenant_id, engine.capability.id, total_chars)
    except Exception:
        logger.debug(
            "Translation: metrics record failed (non-fatal)", exc_info=True,
        )

    return DocumentTranslation(
        schema_version="1.0",
        document_id=hashlib.sha256(doc_path.encode("utf-8")).hexdigest()[:16],
        source_file=os.path.basename(doc_path),
        source_language=src_lang,
        target_language=tgt_lang,
        certified=False,
        engine={
            "id": engine.capability.id,
            "is_local": engine.capability.is_local,
            "license": engine.capability.license,
            "provider_retention_class": engine.capability.provider_retention_class,
            "weights_sha256": engine.model_provenance().get(
                "weights_sha256", "unknown"
            ),
        },
        quality={
            "mean_score": None,
            "below_threshold_count": 0,
            "quality_class": engine.capability.quality_class,
        },
        pages=pages_translated,
        stats={
            "total_chars": total_chars,
            "total_spans": sum(len(p.spans) for p in pages_translated),
            "page_count": len(pages_translated),
        },
        custody={
            "chain_head": "n/a",
            "clock_source": "system",
            "source_ocr_sha256": "n/a",
            "tenant_policy_hash": policy_hash,
            "tenant_id": tenant_id,
        },
        processing={
            "pipeline_version": "4.1.0",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "enable_translation": True,
            "language_source": language_source,
        },
    )


def _instantiate_engine(
    req,
    tenant,
    chain,
    *,
    use_tenant_router: bool = False,
    privilege_flag: bool = False,
):
    """Resolve engine via router and return an instance.

    On :class:`PolicyDenied`, re-raises so the caller's fail-open path
    swallows it (privilege/tenant/policy rejections must NOT silently
    fall back to passthrough -- that would mask audit-relevant denials).

    On ``UNSUPPORTED_LANGUAGE`` (no engine for src->tgt) we fall back to
    passthrough so the sidecar contract still produces output on the SDK
    CI lane where only passthrough is registered; the fallback is
    visible because ``engine.id == "passthrough"`` in the sidecar.

    Wave M2 PR B14 -- when ``use_tenant_router`` is True, the
    :func:`select_engine_for_tenant` path is used.  An empty candidate
    list (raising :class:`NoEligibleEngineError`) falls back to the
    legacy router so deployments with no per-tenant config registered
    still produce output.
    """
    if use_tenant_router:
        try:
            return select_engine_for_tenant(
                text="",
                source_lang=req.src_lang,
                target_lang=req.tgt_lang,
                tenant_id=req.tenant_id,
                allow_download=False,
                custody_chain=chain,
                privilege_flag=privilege_flag,
            )
        except NoEligibleEngineError:
            # No tenant-level candidates -- fall through to legacy router.
            pass
        except PolicyDenied:
            # Policy/privilege denial -- propagate so caller records it.
            raise

    try:
        engine_or_cls = select_engine(req, tenant, chain)
    except PolicyDenied as exc:
        # UNSUPPORTED_LANGUAGE is the only PolicyDenied we transparently
        # downgrade to passthrough -- privilege/tenant denials must
        # propagate so the OCR job's fail-open wrapper records them.
        from ocr_local.translation.custody_adapter import ReasonCode
        if exc.reason_code == ReasonCode.UNSUPPORTED_LANGUAGE:
            engine_cls = get_engine("passthrough")
            return engine_cls()
        raise

    if isinstance(engine_or_cls, type):
        return engine_or_cls()
    return engine_or_cls


def _resolve_source_language(page_data_snap: dict | None) -> tuple[str, str]:
    """Return ``(src_lang, language_source)``.

    ``language_source`` is one of ``"plan_a"``, ``"document_level"``, or
    ``"default_fallback"``.
    """
    if page_data_snap:
        langs = page_data_snap.get("languages")
        if langs:
            lang_counts: Counter = Counter()
            for page_langs in langs.values():
                seq = page_langs if isinstance(page_langs, list) else []
                for span in seq:
                    if isinstance(span, dict) and "language" in span:
                        lang_counts[span["language"]] += 1
            if lang_counts:
                return lang_counts.most_common(1)[0][0], "plan_a"
        page_language = page_data_snap.get("language")
        if page_language:
            lang_counts = Counter()
            for item in page_language.values():
                language = None
                weight = 1
                if isinstance(item, dict):
                    language = item.get("primary_language") or item.get("language")
                    weight = int(item.get("span_count") or 1)
                else:
                    language = getattr(item, "primary_language", None) or getattr(
                        item,
                        "language",
                        None,
                    )
                    weight = int(getattr(item, "span_count", 1) or 1)
                if language:
                    lang_counts[str(language)] += max(weight, 1)
            if lang_counts:
                return lang_counts.most_common(1)[0][0], "plan_a"
        doc_lang = page_data_snap.get("detected_language")
        if doc_lang:
            return doc_lang, "document_level"
    return "en", "default_fallback"


def _make_noop_chain():
    """Return a no-op CustodyChain stand-in for callers that did not pass one."""

    class _NoopChain:
        def log_event(self, *_args, **_kwargs):
            return None

        def append_event(self, *_args, **_kwargs):
            return None

    return _NoopChain()
