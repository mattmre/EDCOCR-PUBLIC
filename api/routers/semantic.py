"""Semantic search API endpoint (VLM-backed).

Provides RAG-ready document querying via a connected VLM inference server.
Requires VLM_ENABLED=true and a valid VLM_ENDPOINT_URL.
"""

import json
import logging
import math
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field, StrictInt, StrictStr

from api.identity import require_role
from api.limits import get_default_rate, limiter
from api.models import ErrorResponse

logger = logging.getLogger(__name__)

# Guard VLM gateway imports. The current VLM surface is an HTTP client for an
# external inference server and requires optional gateway-client dependencies
# such as httpx. Missing client dependencies must not break FastAPI application
# startup; the semantic endpoints degrade to HTTP 503 instead. See maintenance review.
try:
    from vlm_config import VLMConfig, load_vlm_config
    from vlm_gateway import (
        VLMAuthError,
        VLMConnectionError,
        VLMGateway,
        VLMGatewayError,
        VLMServerError,
        VLMTimeoutError,
    )

    VLM_AVAILABLE = True
except ImportError as _vlm_import_error:  # pragma: no cover - import guard
    VLM_AVAILABLE = False
    _VLM_IMPORT_ERROR = _vlm_import_error
    logger.warning(
        "VLM gateway dependencies are not installed; semantic endpoints "
        "will return HTTP 503. Install requirements-vlm.txt to enable. (%s)",
        _vlm_import_error,
    )

    # Provide lightweight sentinels so type references below still resolve.
    VLMConfig = Any  # type: ignore[assignment,misc]

    def load_vlm_config():  # type: ignore[no-redef]
        raise RuntimeError("VLM dependencies are not installed")

    class VLMGatewayError(Exception):  # type: ignore[no-redef]
        """Placeholder when VLM gateway is unavailable."""

    class VLMAuthError(VLMGatewayError):  # type: ignore[no-redef]
        """Placeholder when VLM gateway is unavailable."""

    class VLMConnectionError(VLMGatewayError):  # type: ignore[no-redef]
        """Placeholder when VLM gateway is unavailable."""

    class VLMTimeoutError(VLMGatewayError):  # type: ignore[no-redef]
        """Placeholder when VLM gateway is unavailable."""

    class VLMServerError(VLMGatewayError):  # type: ignore[no-redef]
        """Placeholder when VLM gateway is unavailable."""

    class VLMGateway:  # type: ignore[no-redef]
        """Placeholder when VLM gateway is unavailable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("VLM dependencies are not installed")


def _require_vlm_available() -> None:
    """Raise HTTP 503 if VLM dependencies are not installed."""
    if not VLM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Semantic VLM features require VLM dependencies. "
                "Set VLM_ENABLED=true and install requirements-vlm.txt"
            ),
        )


router = APIRouter(prefix="/api/v1/search", tags=["semantic"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SemanticSearchRequest(BaseModel):
    """Request body for semantic search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language search query.",
    )
    document_id: Optional[str] = Field(
        None,
        max_length=256,
        description="Optional document ID to scope the search.",
    )
    max_results: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of results to return.",
    )
    min_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold (0.0-1.0).",
    )


class SemanticSearchResult(BaseModel):
    """A single semantic search result."""

    text: str = Field(..., description="Matched text passage.")
    score: float = Field(..., description="Relevance score (0.0-1.0).")
    page: int = Field(..., description="Page number where the match was found.")
    document_id: str = Field("", description="Source document identifier.")
    bbox: list[float] = Field(
        default_factory=list,
        description="Bounding box [x1, y1, x2, y2] of the matched region.",
    )


class SemanticSearchResponse(BaseModel):
    """Response containing semantic search results."""

    query: str
    results: list[SemanticSearchResult]
    total: int
    model: str = ""
    processing_time_ms: float = 0.0


class DocumentAnalysisRequest(BaseModel):
    """Request body for VLM document analysis."""

    class DocumentAnalysisPage(BaseModel):
        """Strict page payload accepted by the VLM analysis endpoint."""

        page_number: StrictInt = Field(..., ge=1, description="Page number.")
        text: StrictStr = Field(
            ...,
            max_length=200_000,
            description="OCR text for the page.",
        )
        image_b64: Optional[StrictStr] = Field(
            None,
            max_length=16 * 1024 * 1024,
            description="Optional inline page image; disabled by gateway policy by default.",
        )

        model_config = {"extra": "forbid"}

    pages: list[DocumentAnalysisPage] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of page dicts with page_number, text, and optional image_b64.",
    )
    prompt: Optional[str] = Field(
        None,
        max_length=4000,
        description="Analysis instruction or prompt.",
    )
    document_id: Optional[str] = Field(
        None,
        max_length=256,
        description="Optional document identifier.",
    )


class DocumentAnalysisResponse(BaseModel):
    """Response from VLM document analysis."""

    entities: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    model: str = ""
    processing_time_ms: float = 0.0


class VLMHealthResponse(BaseModel):
    """VLM gateway health status."""

    vlm_enabled: bool
    vlm_reachable: bool
    model_name: str = ""
    endpoint_configured: bool = False


_MAX_VLM_RESULT_TEXT = 64 * 1024
_MAX_VLM_RESULT_COUNT = 100
_MAX_VLM_MODEL_LENGTH = 256
_MAX_VLM_SUMMARY_LENGTH = 64 * 1024
_MAX_VLM_STRUCTURED_BYTES = 512 * 1024


def _finite_score(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VLMGatewayError(f"VLM response field {field} is invalid") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise VLMGatewayError(f"VLM response field {field} is outside [0, 1]")
    return result


def _sanitize_semantic_results(
    results: object, *, max_results: int, expected_document_id: str = ""
) -> list[dict[str, Any]]:
    """Validate and bound untrusted provider result JSON before response use."""
    if not isinstance(results, list) or len(results) > max_results:
        raise VLMGatewayError("VLM response contains too many or invalid results")
    normalized: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            raise VLMGatewayError("VLM result entries must be JSON objects")
        text = result.get("text", "")
        document_id = result.get("document_id", "")
        page = result.get("page", 0)
        bbox = result.get("bbox", [])
        if not isinstance(text, str) or len(text) > _MAX_VLM_RESULT_TEXT:
            raise VLMGatewayError("VLM result text exceeds the safety limit")
        if not isinstance(document_id, str) or len(document_id) > 256:
            raise VLMGatewayError("VLM result document_id is invalid")
        if expected_document_id:
            if document_id != expected_document_id:
                continue
        else:
            # Unpinned analyze: never let the provider mint a document_id.
            document_id = ""
        if type(page) is not int or page < 0:
            raise VLMGatewayError("VLM result page is invalid")
        if not isinstance(bbox, list) or len(bbox) not in (0, 4):
            raise VLMGatewayError("VLM result bbox must contain four coordinates")
        normalized_bbox = []
        for coordinate in bbox:
            try:
                numeric = float(coordinate)
            except (TypeError, ValueError) as exc:
                raise VLMGatewayError("VLM result bbox is invalid") from exc
            if not math.isfinite(numeric):
                raise VLMGatewayError("VLM result bbox contains a non-finite value")
            normalized_bbox.append(numeric)
        normalized.append(
            {
                "text": text,
                "score": _finite_score(result.get("score", 0.0), field="score"),
                "page": page,
                "document_id": expected_document_id or document_id or "",
                "bbox": normalized_bbox,
            }
        )
    return normalized


def _sanitize_analysis_response(result: object, model_name: str) -> dict[str, Any]:
    """Bound provider-controlled analysis fields before Pydantic serialization."""
    if not isinstance(result, dict):
        raise VLMGatewayError("VLM analysis response must be a JSON object")
    try:
        encoded = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VLMGatewayError("VLM analysis response is not JSON serializable") from exc
    if len(encoded) > _MAX_VLM_STRUCTURED_BYTES:
        raise VLMGatewayError("VLM analysis response exceeds the safety limit")
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])
    summary = result.get("summary", "")
    if not isinstance(entities, list) or len(entities) > _MAX_VLM_RESULT_COUNT:
        raise VLMGatewayError("VLM analysis entities are invalid or too numerous")
    if (
        not isinstance(relationships, list)
        or len(relationships) > _MAX_VLM_RESULT_COUNT
    ):
        raise VLMGatewayError("VLM analysis relationships are invalid or too numerous")
    if not all(isinstance(item, dict) for item in (*entities, *relationships)):
        raise VLMGatewayError("VLM analysis structured fields must contain objects")
    if not isinstance(summary, str) or len(summary) > _MAX_VLM_SUMMARY_LENGTH:
        raise VLMGatewayError("VLM analysis summary exceeds the safety limit")
    processing_time = result.get("processing_time_ms", 0.0)
    try:
        processing_time = float(processing_time)
    except (TypeError, ValueError) as exc:
        raise VLMGatewayError("VLM analysis processing time is invalid") from exc
    if not math.isfinite(processing_time) or processing_time < 0.0:
        raise VLMGatewayError("VLM analysis processing time is invalid")
    return {
        "entities": entities,
        "summary": summary,
        "relationships": relationships,
        "confidence": _finite_score(result.get("confidence", 0.0), field="confidence"),
        "model": str(result.get("model", model_name))[:_MAX_VLM_MODEL_LENGTH],
        "processing_time_ms": processing_time,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_vlm_config() -> VLMConfig:
    """Load VLM config from environment."""
    return load_vlm_config()


def _check_vlm_enabled(config: VLMConfig) -> None:
    """Raise 503 if VLM is not enabled or misconfigured."""
    if not config.enabled:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "vlm_disabled",
                "message": (
                    "VLM features are disabled. "
                    "Set VLM_ENABLED=true and configure VLM_ENDPOINT_URL."
                ),
            },
        )
    errors = config.validate()
    if errors:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "vlm_misconfigured",
                "message": "VLM configuration is invalid.",
                "details": {"errors": errors},
            },
        )


def _request_tenant_id(request: Request) -> Optional[str]:
    """Return authenticated tenant scope, if any."""
    return getattr(request.state, "tenant_id", None)


def _is_platform_admin(request: Request) -> bool:
    """True when server-attached tenant_permissions includes platform_admin.

    Trusts only ``request.state.tenant_permissions`` (API key path), never free-form
    JWT ``identity.claims["permissions"]`` (R4-RC-11 / RC-04 amp).
    """
    from api.authz import is_platform_admin

    return is_platform_admin(request)


def _assert_document_access(
    request: Request,
    document_id: Optional[str],
    *,
    require_document_id: bool = False,
) -> None:
    """Pin document_id to a job the caller may access before any VLM work.

    Policy (maintenance review):
    - Single-tenant / anonymous (``request.state.tenant_id is None``): no-op.
    - Multi-tenant semantic search: ``document_id`` required (HTTP 400).
    - Multi-tenant with ``document_id``: verify via ``JobManager.get_job``;
      foreign or missing → HTTP 404 (no existence leak).
    - ``platform_admin`` may access any existing job (lookup without tenant filter).
    - Analyze without ``document_id`` remains allowed (client-supplied page text).
    """
    auth_tid = _request_tenant_id(request)
    is_pa = _is_platform_admin(request)

    if auth_tid is None and not is_pa:
        from api.config import ENABLE_MULTITENANCY

        if ENABLE_MULTITENANCY:
            # OAuth/anonymous under MT must not run unscoped VLM (RC-01 / R4-RC-13).
            from api.authz import require_tenant_scope_or_platform

            require_tenant_scope_or_platform(request)
        # Single-tenant / anonymous: keep optional document_id for backward compat.
        return

    doc_id = (document_id or "").strip() or None
    if not doc_id:
        if require_document_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "document_id_required",
                    "message": (
                        "document_id is required for multi-tenant semantic search. "
                        "Unscoped VLM queries are not permitted."
                    ),
                },
            )
        return

    from api.job_manager import get_manager

    manager = get_manager()
    # Platform admins may cross-tenant; regular tenants are pinned to auth_tid.
    scope = None if is_pa else auth_tid
    job = manager.get_job(doc_id, tenant_id=scope)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "document_not_found",
                "message": f"Document {doc_id} not found.",
            },
        )


def _handle_gateway_error(exc: VLMGatewayError) -> HTTPException:
    """Convert a VLMGatewayError to an appropriate HTTPException."""
    if isinstance(exc, VLMAuthError):
        return HTTPException(
            status_code=502,
            detail={
                "error": "vlm_auth_error",
                "message": "VLM inference server rejected credentials.",
            },
        )
    if isinstance(exc, VLMConnectionError):
        return HTTPException(
            status_code=502,
            detail={
                "error": "vlm_connection_error",
                "message": "VLM inference server is unreachable.",
            },
        )
    if isinstance(exc, VLMTimeoutError):
        return HTTPException(
            status_code=504,
            detail={
                "error": "vlm_timeout",
                "message": "VLM inference server request timed out.",
            },
        )
    if isinstance(exc, VLMServerError):
        return HTTPException(
            status_code=502,
            detail={
                "error": "vlm_server_error",
                "message": "VLM inference server returned an error.",
            },
        )
    return HTTPException(
        status_code=502,
        detail={
            "error": "vlm_error",
            "message": "VLM inference request failed.",
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/semantic",
    name="semantic_search",
    response_model=SemanticSearchResponse,
    status_code=200,
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def semantic_search(
    request: Request,
    body: Annotated[SemanticSearchRequest, Body(...)],
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Search documents using semantic understanding via VLM.

    Sends the query to the configured VLM inference server and returns
    ranked results with relevance scores and bounding boxes.

    Multi-tenant callers must supply a document_id owned by their tenant;
    ownership is verified before any VLM gateway construction (BD002).
    """
    _require_vlm_available()
    config = _get_vlm_config()
    _check_vlm_enabled(config)

    # Fail-closed job pin before VLM (maintenance review).
    pinned_document_id = str(body.document_id or "").strip()
    _assert_document_access(
        request,
        pinned_document_id,
        require_document_id=True,
    )

    gateway = VLMGateway(config)
    try:
        results = gateway.semantic_search(
            query=body.query,
            document_id=pinned_document_id or None,
            max_results=body.max_results,
            min_score=body.min_score,
        )
    except VLMGatewayError as exc:
        logger.error("Semantic search failed: %s", exc)
        raise _handle_gateway_error(exc)
    finally:
        gateway.close()

    try:
        bounded_results = _sanitize_semantic_results(
            results,
            max_results=body.max_results,
            expected_document_id=pinned_document_id,
        )
        parsed_results = [SemanticSearchResult(**result) for result in bounded_results]
    except VLMGatewayError as exc:
        raise _handle_gateway_error(exc)

    return SemanticSearchResponse(
        query=body.query,
        results=parsed_results,
        total=len(parsed_results),
        model=config.model_name,
    )


@router.post(
    "/analyze",
    name="analyze_document",
    response_model=DocumentAnalysisResponse,
    status_code=200,
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def analyze_document(
    request: Request,
    body: Annotated[DocumentAnalysisRequest, Body(...)],
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Analyze document pages using a VLM for entity extraction and summarization.

    Sends pages (text and optional images) to the VLM inference server
    for deep document understanding.

    Body-only analysis without document_id is allowed (client-supplied text).
    When document_id is present under multi-tenant auth, ownership is pinned
    before any VLM gateway construction (BD002).
    """
    _require_vlm_available()
    config = _get_vlm_config()
    _check_vlm_enabled(config)

    # Pin document_id when present; body-only pages remain allowed (BD002).
    _assert_document_access(
        request,
        body.document_id,
        require_document_id=False,
    )

    gateway = VLMGateway(config)
    try:
        result = gateway.analyze_document(
            pages=[page.model_dump(exclude_none=True) for page in body.pages],
            prompt=body.prompt,
            document_id=body.document_id,
        )
    except VLMGatewayError as exc:
        logger.error("Document analysis failed: %s", exc)
        raise _handle_gateway_error(exc)
    finally:
        gateway.close()

    try:
        bounded = _sanitize_analysis_response(result, config.model_name)
        return DocumentAnalysisResponse(**bounded)
    except VLMGatewayError as exc:
        raise _handle_gateway_error(exc)


@router.get(
    "/vlm/health",
    name="vlm_health",
    response_model=VLMHealthResponse,
    responses={503: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def vlm_health(request: Request):
    """Check VLM gateway health and connectivity.

    Returns whether the VLM feature is enabled and whether the inference
    server is reachable. Does not require authentication to allow
    monitoring probes.
    """
    if not VLM_AVAILABLE:
        return VLMHealthResponse(
            vlm_enabled=False,
            vlm_reachable=False,
            endpoint_configured=False,
        )

    config = _get_vlm_config()

    if not config.enabled:
        return VLMHealthResponse(
            vlm_enabled=False,
            vlm_reachable=False,
            endpoint_configured=bool(config.endpoint_url),
        )

    gateway = VLMGateway(config)
    try:
        reachable = gateway.health_check()
    except Exception:
        reachable = False
    finally:
        gateway.close()

    if not reachable:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "vlm_unavailable",
                "message": "VLM inference server is unavailable.",
            },
        )

    return VLMHealthResponse(
        vlm_enabled=True,
        vlm_reachable=reachable,
        model_name=config.model_name,
        endpoint_configured=bool(config.endpoint_url),
    )
