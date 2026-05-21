"""Semantic search API endpoint (VLM-backed).

Provides RAG-ready document querying via a connected VLM inference server.
Requires VLM_ENABLED=true and a valid VLM_ENDPOINT_URL.
"""

import logging
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.identity import require_role
from api.limits import get_default_rate, limiter
from api.models import ErrorResponse

logger = logging.getLogger(__name__)

# Guard VLM gateway imports — the VLM stack depends on heavy optional
# dependencies (httpx, torch, transformers, sentence-transformers, ...).
# A missing dependency must not break FastAPI application startup; the
# semantic endpoints degrade to HTTP 503 instead. See .
try:
    from vlm_config import VLMConfig, load_vlm_config
    from vlm_gateway import (
        VLMAuthError,
        VLMConnectionError,
        VLMGateway,
        VLMGatewayError,
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
                "Semantic search requires VLM dependencies. "
                "Set ENABLE_VLM=true and install requirements-vlm.txt"
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

    pages: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of page dicts with page_number, text, and optional image_b64.",
    )
    prompt: Optional[str] = Field(
        None,
        max_length=4000,
        description="Analysis instruction or prompt.",
    )
    document_id: Optional[str] = Field(
        None,
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
    return HTTPException(
        status_code=502,
        detail={
            "error": "vlm_error",
            "message": str(exc),
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
    """
    _require_vlm_available()
    config = _get_vlm_config()
    _check_vlm_enabled(config)

    gateway = VLMGateway(config)
    try:
        results = gateway.semantic_search(
            query=body.query,
            document_id=body.document_id,
            max_results=body.max_results,
            min_score=body.min_score,
        )
    except VLMGatewayError as exc:
        logger.error("Semantic search failed: %s", exc)
        raise _handle_gateway_error(exc)
    finally:
        gateway.close()

    parsed_results = [
        SemanticSearchResult(
            text=r.get("text", ""),
            score=float(r.get("score", 0.0)),
            page=int(r.get("page", 0)),
            document_id=r.get("document_id", ""),
            bbox=r.get("bbox", []),
        )
        for r in results
    ]

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
    """
    _require_vlm_available()
    config = _get_vlm_config()
    _check_vlm_enabled(config)

    gateway = VLMGateway(config)
    try:
        result = gateway.analyze_document(
            pages=body.pages,
            prompt=body.prompt,
            document_id=body.document_id,
        )
    except VLMGatewayError as exc:
        logger.error("Document analysis failed: %s", exc)
        raise _handle_gateway_error(exc)
    finally:
        gateway.close()

    return DocumentAnalysisResponse(
        entities=result.get("entities", []),
        summary=result.get("summary", ""),
        relationships=result.get("relationships", []),
        confidence=float(result.get("confidence", 0.0)),
        model=result.get("model", config.model_name),
        processing_time_ms=float(result.get("processing_time_ms", 0.0)),
    )


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

    return VLMHealthResponse(
        vlm_enabled=True,
        vlm_reachable=reachable,
        model_name=config.model_name,
        endpoint_configured=bool(config.endpoint_url),
    )
