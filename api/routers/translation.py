"""Translation job submission endpoint -- Plan B Wave M1 + M2 PR B14.

This router is feature-gated: it is only registered with the FastAPI
application when ``ENABLE_TRANSLATION_API`` is truthy in the
environment.  The global ``api_key_middleware`` establishes identity and
the route dependencies below enforce operator access even in development
mode.

The single-job endpoint is retained as an explicit capability boundary.
It validates tenant scope and may perform a routing preview, but it does not
create durable work. Callers must use the batch endpoint for an operational
async translation job.
"""

from __future__ import annotations

import logging

from asgiref.sync import sync_to_async
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from api.identity import require_role
from api.language_tags import canonicalize_language_tag, canonicalize_language_tags
from api.limits import limiter
from api.routers.translation_scope import bind_requested_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/translation", tags=["translation"])

# B15 -- COMETKiwi QE estimator instance is cached per-process so the
# model is loaded at most once per worker.  Lazy-built on first request.
_QE_ESTIMATOR = None
_QE_LOAD_FAILED = False
_QE_LOAD_FAILED_REASON: str | None = None


def _get_qe_estimator():
    """Return a cached :class:`CometKiwiEstimator` or ``None`` on failure.

    Reads :class:`QualityEstimationConfig` from environment vars on each
    cache miss so operators can swap models without restarting (the
    cache is invalidated only when the previous load failed).
    """
    global _QE_ESTIMATOR, _QE_LOAD_FAILED, _QE_LOAD_FAILED_REASON

    if _QE_ESTIMATOR is not None:
        return _QE_ESTIMATOR
    if _QE_LOAD_FAILED:
        return None

    from ocr_local.translation.quality_estimation import (
        CometKiwiEstimator,
        ModelNotCachedError,
        QualityEstimationConfig,
    )

    cfg = QualityEstimationConfig.from_env()
    try:
        estimator = CometKiwiEstimator(cfg, allow_download=False)
    except ModelNotCachedError as exc:
        _QE_LOAD_FAILED = True
        _QE_LOAD_FAILED_REASON = str(exc)
        logger.info("Translation QE estimator unavailable: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover -- defensive
        _QE_LOAD_FAILED = True
        _QE_LOAD_FAILED_REASON = f"unexpected_error: {exc}"
        logger.warning("Translation QE estimator failed to construct: %s", exc)
        return None

    _QE_ESTIMATOR = estimator
    return _QE_ESTIMATOR


def _reset_qe_estimator_cache() -> None:
    """Test-only: invalidate the cached estimator state."""
    global _QE_ESTIMATOR, _QE_LOAD_FAILED, _QE_LOAD_FAILED_REASON
    _QE_ESTIMATOR = None
    _QE_LOAD_FAILED = False
    _QE_LOAD_FAILED_REASON = None


class TranslationJobSubmit(BaseModel):
    """Translation job submission request.

    At least one of ``source_job_id``, ``source_uri``, or ``source_text``
    must be provided.  ``target_languages`` is required and must contain
    at least one BCP-47 entry.
    """

    source_job_id: str | None = Field(
        default=None, max_length=64
    )  # re-translate from existing OCR job
    source_uri: str | None = Field(
        default=None, max_length=2048
    )  # S3/NFS URI to a document
    source_text: str | None = Field(
        default=None, max_length=100_000
    )  # plain text for standalone translation
    target_languages: list[str] = Field(
        ..., min_length=1, max_length=32
    )  # BCP-47 codes, e.g. ["fr", "es"]
    # Default tenant id is "default" for backward compatibility; pass
    # ``None`` (or omit when None default supported) to indicate
    # single-tenant / anonymous mode and fall back to the legacy router.
    tenant_id: str | None = Field(default="default", max_length=128)
    source_language: str | None = Field(
        default=None, max_length=16
    )  # BCP-47 source hint (router_v2 only)
    quality: str = Field(default="standard", max_length=32)
    latency: str = Field(default="standard", max_length=32)
    webhook_url: str | None = Field(default=None, max_length=2_048)
    # When True, route through the tenant-aware router_v2
    # (``select_engine_for_tenant``).  When False, use the legacy stub
    # path that returns ``status="queued"`` without engine selection.
    use_tenant_router: bool = False
    allow_download: bool = False  # router_v2 cache filter -- see PR B14
    # ``certified=True`` is set ONLY by the review queue after strong-auth
    # promotion; the submit path always defaults to False.
    certify_after_review: bool = False

    @field_validator("target_languages")
    @classmethod
    def _at_least_one_target(cls, v: list[str]) -> list[str]:
        return canonicalize_language_tags(v)

    @field_validator("source_language")
    @classmethod
    def _source_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return canonicalize_language_tag(value)

    def has_source(self) -> bool:
        """Return True when at least one source field is populated."""
        return bool(self.source_job_id or self.source_uri or self.source_text)


class TranslationJobResponse(BaseModel):
    """Translation job submission response."""

    job_id: str
    status: str = "queued"
    target_languages: list[str]
    certified: bool = False
    message: str = ""
    # Wave M2 PR B14 -- populated when ``use_tenant_router=True`` and
    # router_v2 successfully selected an engine for at least one target
    # language.  Maps target language -> selected engine id.
    selected_engines: dict[str, str] | None = None


@router.post(
    "/jobs",
    status_code=202,
    response_model=TranslationJobResponse,
    name="submit_translation_job",
)
@limiter.limit("5/minute")
async def submit_translation_job(
    request: Request,
    req: TranslationJobSubmit,
    _auth: None = Depends(require_role("admin", "operator")),
) -> TranslationJobResponse:
    """Submit a translation job.

    Translation job dispatch is not implemented by this single-job endpoint.
    It fails closed instead of returning a synthetic ``queued`` job that
    cannot be polled, cancelled, or delivered. Use the durable batch endpoint
    for the implemented async path.
    """
    if not req.has_source():
        raise HTTPException(
            status_code=422,
            detail=("One of source_job_id, source_uri, or source_text is required"),
        )

    effective_tenant_id = bind_requested_tenant(
        request,
        req.tenant_id,
        allow_default_alias=True,
    )
    # This endpoint historically returned a synthetic UUID for both paths.
    # Do not represent engine selection as durable work until a coordinator
    # job, status lifecycle, and output/custody path exist.
    if not req.use_tenant_router:
        raise HTTPException(
            status_code=501,
            detail=(
                "Single translation-job dispatch is unavailable; no job was "
                "created. Use POST /api/v1/translation/batches for the durable "
                "async translation path."
            ),
        )

    # Tenant-aware path (router_v2) -- resolve engine per target language.
    # Engine selection is best-effort here, but this endpoint still fails
    # closed because selection is not durable work and cannot be polled.
    from ocr_local.translation.router import (
        NoEligibleEngineError,
        prepare_translation_input,
        select_engine_for_tenant,
    )

    src_lang = req.source_language or "en"
    text_for_routing = req.source_text or ""
    selected: dict[str, str] = {}
    failure_reasons: list[str] = []

    for tgt in req.target_languages:
        try:
            # Glossary preprocessing must run BEFORE engine selection so
            # length-sensitive routing decisions see the modified text.
            modified_text, _hits = await sync_to_async(
                prepare_translation_input,
                thread_sensitive=True,
            )(
                tenant_id=effective_tenant_id,
                text=text_for_routing,
                source_lang=src_lang,
                target_lang=tgt,
            )
            engine = await sync_to_async(
                select_engine_for_tenant,
                thread_sensitive=True,
            )(
                text=modified_text,
                source_lang=src_lang,
                target_lang=tgt,
                tenant_id=effective_tenant_id,
                allow_download=req.allow_download,
            )
            selected[tgt] = engine.capability.id
        except NoEligibleEngineError:
            failure_reasons.append(f"{tgt}: no eligible translation engine")
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning(
                "submit_translation_job: router_v2 failed for tgt=%s tenant=%s: %s",
                tgt,
                effective_tenant_id,
                exc,
            )
            failure_reasons.append(f"{tgt}: engine selection unavailable")

    raise HTTPException(
        status_code=501,
        detail={
            "message": (
                "Single translation-job dispatch is unavailable; engine "
                "selection was not persisted and no job was created. Use POST "
                "/api/v1/translation/batches for the durable async path."
            ),
            "selected_engines": selected or None,
            "routing_failures": failure_reasons,
        },
    )


# ---------------------------------------------------------------------------
# B15 -- POST /api/v1/translation/score-pair (COMETKiwi QE)
# ---------------------------------------------------------------------------


class ScorePairRequest(BaseModel):
    """Single ``(source, target)`` pair for QE scoring."""

    # COMET-style scoring is a sentence/pair operation.  An unbounded body
    # here could make one rate-limited request allocate model-sized tensors
    # far beyond the normal OCR span budget.
    source: str = Field(..., min_length=1, max_length=20_000)
    target: str = Field(..., min_length=1, max_length=20_000)
    source_lang: str = Field(..., min_length=2, max_length=16)
    target_lang: str = Field(..., min_length=2, max_length=16)


class ScorePairResponse(BaseModel):
    """QE result for a single pair."""

    score: float | None
    available: bool
    reason: str | None = None
    model_id: str | None = None
    threshold_warn: float
    threshold_reject: float


@router.post(
    "/score-pair",
    status_code=200,
    response_model=ScorePairResponse,
    name="translation_score_pair",
)
@limiter.limit("10/minute")
async def translation_score_pair(
    request: Request,
    body: ScorePairRequest,
    _auth: None = Depends(require_role("admin", "operator")),
) -> ScorePairResponse:
    """Score a single (source, MT) pair via COMETKiwi.

    Returns 503 when the QE model is not available on the server (e.g.
    ``unbabel-comet`` not installed, or the local model path was unset
    in air-gapped mode).  Operator RBAC is enforced in addition to the
    global API-key middleware because this endpoint consumes model resources.

    Rate-limited to 10 requests per minute per key (API-key bucket via
    the shared :data:`api.limits.limiter`).
    """
    estimator = _get_qe_estimator()
    if estimator is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Translation QE unavailable: "
                f"{_QE_LOAD_FAILED_REASON or 'estimator not initialised'}"
            ),
        )

    score = estimator.score_pair(body.source, body.target)
    if not score.available:
        # comet not installed at runtime -> 503; empty input -> 200 with
        # available=False.  We distinguish via the recorded reason.
        if score.reason == "comet_not_installed":
            raise HTTPException(
                status_code=503,
                detail="Translation QE unavailable: comet_not_installed",
            )

    return ScorePairResponse(
        score=score.score,
        available=score.available,
        reason=score.reason,
        model_id=score.model_id,
        threshold_warn=estimator.config.score_threshold_warn,
        threshold_reject=estimator.config.score_threshold_reject,
    )
