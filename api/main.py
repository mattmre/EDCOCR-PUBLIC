"""FastAPI application factory for the OCR REST API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.audit import ApiAuditMiddleware
from api.auth import api_key_middleware, validate_auth_configuration
from api.config import (
    CORS_ALLOWED_ORIGINS,
    ENABLE_MULTITENANCY,
    EXPOSE_API_DOCS,
    MAX_REQUEST_BODY_SIZE,
)
from api.database import get_engine
from api.limits import limiter
from api.prometheus import router as prometheus_router
from api.request_size_limit import RequestSizeLimitMiddleware
from api.routers import (
    batch,
    events,
    health,
    job_logs,
    jobs,
    outputs,
    recall,
    review,
    semantic,
    stamps,
    transforms,
    ws,
)
from api.security_headers import SecurityHeadersMiddleware
from ocr_local.config.version import __version__

logger = logging.getLogger(__name__)

# OpenAPI tag metadata — descriptions shown in /docs and /redoc
OPENAPI_TAGS = [
    {"name": "jobs", "description": "OCR job submission, status, and result retrieval"},
    {"name": "batch", "description": "Batch job submission and management"},
    {"name": "health", "description": "Service health check"},
    {"name": "websocket", "description": "Real-time job progress streaming via WebSocket"},
    {"name": "transforms", "description": "Document transform operations (clean, image, PDF)"},
    {"name": "stamps", "description": "Document stamping (Bates, designation, zone-based)"},
    {"name": "events", "description": "Job event history and webhook dead-letter queue"},
    {"name": "job-logs", "description": "Per-job NDJSON log streaming"},
    {"name": "semantic", "description": "Semantic search and VLM document analysis"},
    {"name": "outputs", "description": "Job output manifest and schema retrieval"},
    {"name": "prometheus", "description": "Prometheus metrics endpoint"},
    {"name": "admin", "description": "Multi-tenant administration (requires ENABLE_MULTITENANCY)"},
    {"name": "dashboard", "description": "Pipeline throughput dashboard (requires ENABLE_DASHBOARD)"},
    {"name": "fleet", "description": "Worker fleet status (requires ENABLE_DASHBOARD)"},
    {"name": "alerts", "description": "Queue depth alerts (requires ENABLE_DASHBOARD)"},
    {"name": "analytics", "description": "Historical processing analytics (requires ENABLE_DASHBOARD)"},
    {"name": "review", "description": "Human review queue for low-confidence documents"},
    {"name": "recall", "description": "Cross-document entity and extraction search"},
]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler.

    on shutdown, gracefully drain the JobManager worker threads.
    Worker threads are non-daemon so we must explicitly call ``shutdown()``
    to allow in-flight jobs to complete with full audit trail instead of
    being killed on interpreter exit.
    """
    try:
        yield
    finally:
        try:
            from api.job_manager import _manager_instance
        except Exception:
            return
        if _manager_instance is None:
            return
        try:
            timeout = float(os.environ.get("JOB_MANAGER_SHUTDOWN_TIMEOUT", "30"))
        except (TypeError, ValueError):
            timeout = 30.0
        try:
            _manager_instance.shutdown(timeout=timeout)
        except Exception:
            logger.exception("JobManager shutdown failed during FastAPI shutdown")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    # Gate OpenAPI docs exposure.  When EXPOSE_API_DOCS is false
    # (the default) the interactive docs, ReDoc, and raw schema endpoints
    # are not mounted, preventing unauthenticated API enumeration.
    docs_url = "/docs" if EXPOSE_API_DOCS else None
    redoc_url = "/redoc" if EXPOSE_API_DOCS else None
    openapi_url = "/openapi.json" if EXPOSE_API_DOCS else None

    app = FastAPI(
        title="EDCOCR API",
        description="REST API for forensic-grade document OCR processing.",
        version=__version__,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        openapi_tags=OPENAPI_TAGS,
        lifespan=_lifespan,
    )
    if not EXPOSE_API_DOCS:
        logger.info(
            "OpenAPI docs disabled (EXPOSE_API_DOCS=false); "
            "/docs, /redoc, /openapi.json are not mounted"
        )

    # Ensure database tables exist
    get_engine()
    try:
        from api.job_manager import recover_interrupted_jobs

        recover_interrupted_jobs()
    except Exception:
        logger.exception("Interrupted job recovery failed during startup")
    validate_auth_configuration()

    # Initialize distributed tracing (OpenTelemetry or local fallback)
    # configure_tracing() handles init + optional FastAPI auto-instrumentation
    try:
        from api.tracing import configure_tracing
        configure_tracing(app)
    except Exception:
        logger.debug("Tracing initialization skipped")

    # Register middlewares — order matters:
    # ApiAuditMiddleware runs FIRST (outermost), then SlowAPIMiddleware,
    # then the auth middleware. This ensures request audit logging captures
    # rate-limit responses and auth denials without changing auth behavior.
    app.middleware("http")(api_key_middleware)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(ApiAuditMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    # Reject oversized non-multipart bodies before downstream handlers
    # read them.  Added last so it runs first (Starlette's add_middleware
    # wraps the existing stack).
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_size=MAX_REQUEST_BODY_SIZE,
    )

    # CORS middleware — opt-in via CORS_ALLOWED_ORIGINS env var.
    # Empty default means no CORS middleware is added (conservative).
    if CORS_ALLOWED_ORIGINS:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(CORS_ALLOWED_ORIGINS),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled for origins: %s", CORS_ALLOWED_ORIGINS)

    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Register routers — batch must come before jobs to avoid path conflicts
    # (/api/v1/jobs/batch vs /api/v1/jobs/{job_id})
    app.include_router(batch.router)
    app.include_router(jobs.router)
    app.include_router(health.router)
    app.include_router(ws.router)
    app.include_router(transforms.router)
    app.include_router(stamps.router)
    app.include_router(events.router)
    app.include_router(job_logs.router)
    app.include_router(outputs.router)
    app.include_router(semantic.router)
    app.include_router(review.router)
    app.include_router(recall.router)
    app.include_router(prometheus_router)

    # Register admin router when multi-tenancy is enabled
    if ENABLE_MULTITENANCY:
        from api.routers import admin
        app.include_router(admin.router)
        logger.info("Multi-tenancy enabled: admin endpoints registered")

    # Translation (Plan B Wave M1, feature-gated -- default off)
    ENABLE_TRANSLATION_API = os.environ.get(
        "ENABLE_TRANSLATION_API", ""
    ).lower() in ("1", "true", "yes")
    if ENABLE_TRANSLATION_API:
        from api.routers.translation import router as translation_router
        from api.routers.translation_admin import router as translation_admin_router
        from api.routers.translation_batch import router as translation_batch_router
        app.include_router(translation_router)
        app.include_router(translation_admin_router)
        app.include_router(translation_batch_router)
        logger.info(
            "Translation API endpoints enabled (Plan B Wave M1+M2: admin/glossary/batch)"
        )

    # Federation custody chain (Plan C Phase 1 C6, feature-gated -- default off)
    ENABLE_FEDERATION_CUSTODY = os.environ.get(
        "OCR_FEDERATION_CUSTODY_ENABLED", ""
    ).lower() in ("1", "true", "yes", "on")
    if ENABLE_FEDERATION_CUSTODY:
        from api.routers.federation_custody import router as federation_custody_router
        app.include_router(federation_custody_router)
        logger.info(
            "Federation custody-chain ingest endpoint enabled "
            "(Plan C Phase 1 C6: /api/v1/federation/custody/ingest)"
        )

    # Dashboard / fleet / alerts / analytics (feature-gated)
    ENABLE_DASHBOARD = os.environ.get("ENABLE_DASHBOARD", "").lower() in ("1", "true", "yes")
    if ENABLE_DASHBOARD:
        from api.routers import alerts, analytics_router, dashboard, fleet
        app.include_router(dashboard.router)
        app.include_router(fleet.router)
        app.include_router(alerts.router)
        app.include_router(analytics_router.router)
        logger.info("Dashboard endpoints enabled (dashboard, fleet, alerts, analytics)")

    # Override OpenAPI schema to inject security scheme metadata.
    # Auth is implemented as raw HTTP middleware (api/auth.py), so FastAPI
    # cannot auto-discover the security schemes.  We inject them manually
    # so that /docs, /redoc, and the exported openapi.json advertise the
    # correct authentication requirements.
    def custom_openapi():  # type: ignore[no-untyped-def]
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
        )
        openapi_schema.setdefault("components", {})
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": (
                    "API key for authentication. "
                    "Set via OCR_API_KEY environment variable."
                ),
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "OAuth2 Bearer token (when OAUTH2_ENABLED=true).",
            },
        }
        openapi_schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    return app


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Return 429 with JSON error when rate limit is exceeded."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please try again later.",
        },
    )


# Uvicorn entry point: `uvicorn api.main:app`
app = create_app()
