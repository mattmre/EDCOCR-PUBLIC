"""Translation-side adapter for ``DocumentBundle v1`` inputs."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping

from ocr_local.contracts import canonical_json_sha256, validate_contract_payload
from ocr_local.translation.custody_adapter import ReasonCode
from ocr_local.translation.engines import get_engine
from ocr_local.translation.models import TranslationRequest
from ocr_local.translation.policy import (
    PolicyDenied,
    compute_policy_hash,
    load_tenant_policy,
)
from ocr_local.translation.router import select_engine

TRANSLATION_BUNDLE_SCHEMA_VERSION = "translation-bundle-v1"


def translate_document_bundle(
    document_bundle: dict[str, Any],
    *,
    target_language: str,
    tenant_id: str = "default",
    engine_id: str | None = None,
    custody_chain: Any | None = None,
    validate_input: bool = True,
    validate_output: bool = True,
) -> dict[str, Any]:
    """Translate a ``DocumentBundle`` into a schema-valid ``TranslationBundle``.

    ``engine_id`` is optional.  Tests and local contract probes can pass
    ``"passthrough"`` to exercise the existing dependency-free translation
    stub; production callers can omit it and use the normal router.
    """

    if validate_input:
        validate_contract_payload(document_bundle, "document-bundle-v1")

    source_language = _source_language(document_bundle)
    source_spans = _source_spans_for_engine(document_bundle)
    tenant_policy = load_tenant_policy(tenant_id)
    policy_hash = compute_policy_hash(tenant_policy)
    chain = custody_chain or _NoopChain()

    if engine_id is None:
        request = TranslationRequest(
            src_lang=source_language,
            tgt_lang=target_language,
            tenant_id=tenant_id,
            privilege_flag=bool(
                document_bundle.get("privilege_flags", {}).get("privileged", False)
            ),
        )
        try:
            engine = select_engine(request, tenant_policy, chain)
        except PolicyDenied as exc:
            if exc.reason_code != ReasonCode.UNSUPPORTED_LANGUAGE:
                raise
            engine = get_engine("passthrough")()
    else:
        engine = get_engine(engine_id)()

    translated = engine.translate_spans(
        source_spans,
        source_language,
        target_language,
    )
    source_bundle_sha256 = canonical_json_sha256(document_bundle)
    translated_spans = _translated_spans(
        translated,
        source_spans,
        source_language=source_language,
        target_language=target_language,
        engine_id=engine.capability.id,
    )

    bundle: dict[str, Any] = {
        "schema_version": TRANSLATION_BUNDLE_SCHEMA_VERSION,
        "document_id": document_bundle["document_id"],
        "source_ocr_sha256": document_bundle["source_ocr_sha256"],
        "source_bundle_sha256": source_bundle_sha256,
        "target_language": target_language,
        "translated_spans": translated_spans,
        "engine_provider": {
            "id": engine.capability.id,
            "family": _provider_family(engine.capability.id),
            "is_local": bool(engine.capability.is_local),
            "is_cloud": bool(engine.capability.is_cloud),
            "license": engine.capability.license,
            "provider_retention_class": engine.capability.provider_retention_class,
        },
        "model_provenance": _model_provenance(engine),
        "quality_scores": _quality_scores(
            translated_spans,
            quality_class=engine.capability.quality_class,
        ),
        "certified": False,
        "custody_chain_head": _custody_chain_head(
            document_bundle,
            custody_chain=custody_chain,
        ),
        "artifact_manifest": _artifact_manifest(source_bundle_sha256),
        "glossary_hits": sorted(
            {
                hit
                for span in translated_spans
                for hit in span.get("glossary_hits", [])
            }
        ),
    }
    bundle["model_provenance"]["tenant_policy_hash"] = policy_hash

    if validate_output:
        validate_translation_bundle(bundle)
    return bundle


def validate_translation_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``TranslationBundle v1`` payload."""

    return validate_contract_payload(bundle, TRANSLATION_BUNDLE_SCHEMA_VERSION)


def _source_language(document_bundle: Mapping[str, Any]) -> str:
    language_metadata = document_bundle.get("language_metadata") or {}
    primary = language_metadata.get("primary_language")
    if primary:
        return str(primary)
    detected = language_metadata.get("detected_languages") or []
    if detected:
        return str(detected[0])
    for span in document_bundle.get("spans", []):
        if span.get("language"):
            return str(span["language"])
    return "und"


def _source_spans_for_engine(document_bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "span_id": span["span_id"],
            "text": span["text"],
            "bbox": span["bbox"],
            "bboxes": span.get("bboxes", [span["bbox"]]),
            "page_number": span["page_number"],
        }
        for span in document_bundle.get("spans", [])
    ]


def _translated_spans(
    translated: list[Any],
    source_spans: list[dict[str, Any]],
    *,
    source_language: str,
    target_language: str,
    engine_id: str,
) -> list[dict[str, Any]]:
    source_by_id = {span["span_id"]: span for span in source_spans}
    out: list[dict[str, Any]] = []
    for index, translated_span in enumerate(translated):
        span_id = _span_value(translated_span, "span_id", source_spans[index]["span_id"])
        source = source_by_id.get(span_id, source_spans[index])
        out.append(
            {
                "span_id": span_id,
                "page_number": int(source["page_number"]),
                "source_text": _span_value(
                    translated_span,
                    "source_text",
                    source["text"],
                ),
                "translated_text": _span_value(
                    translated_span,
                    "target_text",
                    source["text"],
                ),
                "source_bbox": source["bbox"],
                "source_bboxes": source.get("bboxes", [source["bbox"]]),
                "source_language": _span_value(
                    translated_span,
                    "source_language",
                    source_language,
                ),
                "target_language": _span_value(
                    translated_span,
                    "target_language",
                    target_language,
                ),
                "confidence": float(
                    _span_value(translated_span, "confidence", 1.0)
                ),
                "quality_score": _span_value(translated_span, "quality_score", None),
                "engine_id": _span_value(translated_span, "engine_id", engine_id),
                "glossary_hits": list(
                    _span_value(translated_span, "glossary_hits", [])
                ),
            }
        )
    return out


def _span_value(span: Any, field_name: str, default: Any) -> Any:
    if is_dataclass(span):
        return getattr(span, field_name, default)
    if isinstance(span, Mapping):
        return span.get(field_name, default)
    return getattr(span, field_name, default)


def _provider_family(engine_id: str) -> str:
    lowered = engine_id.lower()
    if engine_id == "passthrough":
        return "passthrough"
    if "ct2" in lowered or "opus" in lowered or "nllb" in lowered or "madlad" in lowered:
        return "ct2_nmt"
    if "cloud" in lowered or "vertex" in lowered or "gemini" in lowered:
        return "llm_cloud"
    if "qwen" in lowered or "llm" in lowered:
        return "llm_local"
    return "unknown"


def _model_provenance(engine: Any) -> dict[str, Any]:
    provenance = dict(engine.model_provenance())
    provenance.setdefault("weights_sha256", "unknown")
    provenance.setdefault("runtime", "ocr_local.translation")
    provenance.setdefault("runtime_version", engine.runtime_info().get("version", "unknown"))
    return provenance


def _quality_scores(
    translated_spans: list[dict[str, Any]],
    *,
    quality_class: str,
) -> dict[str, Any]:
    scores = [
        span["quality_score"]
        for span in translated_spans
        if span.get("quality_score") is not None
    ]
    mean_score = sum(scores) / len(scores) if scores else None
    return {
        "mean_score": mean_score,
        "below_threshold_count": sum(1 for score in scores if score < 0.7),
        "quality_class": quality_class,
    }


def _custody_chain_head(
    document_bundle: Mapping[str, Any],
    *,
    custody_chain: Any | None,
) -> str:
    if custody_chain is not None:
        for attr in ("chain_head", "head", "current_hash"):
            value = getattr(custody_chain, attr, None)
            if value:
                return str(value)
    return str(document_bundle["custody_chain_head"])


def _artifact_manifest(source_bundle_sha256: str) -> dict[str, Any]:
    return {
        "artifacts": [
            {
                "artifact_id": "source_document_bundle",
                "artifact_type": "document_bundle",
                "sha256": source_bundle_sha256,
            },
            {
                "artifact_id": "translation_bundle",
                "artifact_type": "translation_bundle",
            },
        ]
    }


class _NoopChain:
    def log_event(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def append_event(self, *_args: Any, **_kwargs: Any) -> None:
        return None
