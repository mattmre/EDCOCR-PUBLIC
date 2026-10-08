"""FastAPI application factory for the OCR REST API."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
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
    assert_api_audit_policy,
    assert_cors_origins_policy,
)
from api.database import assert_sqlite_single_writer_policy, get_engine
from api.limits import assert_rate_limit_storage_policy, limiter
from api.prometheus import router as prometheus_router
from api.request_size_limit import RequestSizeLimitMiddleware
from api.routers import (
    batch,
    engines,
    events,
    health,
    job_logs,
    jobs,
    outputs,
    recall,
    review,
    semantic,
    session_auth,
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
    {
        "name": "websocket",
        "description": "Real-time job progress streaming via WebSocket",
    },
    {
        "name": "transforms",
        "description": "Document transform operations (clean, image, PDF)",
    },
    {
        "name": "stamps",
        "description": "Document stamping (Bates, designation, zone-based)",
    },
    {
        "name": "events",
        "description": "Job event history and webhook dead-letter queue",
    },
    {"name": "job-logs", "description": "Per-job NDJSON log streaming"},
    {"name": "semantic", "description": "Semantic search and VLM document analysis"},
    {"name": "outputs", "description": "Job output manifest and schema retrieval"},
    {"name": "prometheus", "description": "Prometheus metrics endpoint"},
    {
        "name": "admin",
        "description": "Multi-tenant administration (requires ENABLE_MULTITENANCY)",
    },
    {
        "name": "dashboard",
        "description": "Pipeline throughput dashboard (requires ENABLE_DASHBOARD)",
    },
    {"name": "fleet", "description": "Worker fleet status (requires ENABLE_DASHBOARD)"},
    {"name": "alerts", "description": "Queue depth alerts (requires ENABLE_DASHBOARD)"},
    {
        "name": "analytics",
        "description": "Historical processing analytics (requires ENABLE_DASHBOARD)",
    },
    {
        "name": "review",
        "description": "Human review queue for low-confidence documents",
    },
    {"name": "recall", "description": "Cross-document entity and extraction search"},
]


def _translation_handoff_reconcile_loop(stop_event: threading.Event) -> None:
    """Retry optional translation handoffs on a shutdown-safe daemon thread."""
    from api.background_work import register_writer
    from api.routers.translation_batch import reconcile_translation_batch_handoffs

    try:
        interval = float(os.environ.get("TRANSLATION_RECONCILE_INTERVAL_SECONDS", "30"))
    except (TypeError, ValueError):
        interval = 30.0
    interval = min(max(interval, 1.0), 3600.0)
    while not stop_event.is_set():
        finish = register_writer()
        try:
            reconcile_translation_batch_handoffs()
        except Exception:
            # Translation is opt-in. A coordinator outage must not make
            # unrelated OCR/API surfaces unavailable, and the next interval
            # will retry the durable rows without requiring a restart.
            logger.exception("Translation handoff reconciliation failed; retrying")
        finally:
            finish()
        if stop_event.wait(interval):
            return


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler.

    maintenance review: on shutdown, gracefully drain the JobManager worker threads.
    Worker threads are non-daemon so we must explicitly call ``shutdown()``
    to allow in-flight jobs to complete with full audit trail instead of
    being killed on interpreter exit.
    """
    from api import database
    from api.background_work import wait_for_writers
    from api.job_manager import get_manager, recover_interrupted_jobs
    from api.process_owner import ApiProcessOwner, undrained_owners

    owner = ApiProcessOwner(database._engine_db_path or database.DB_PATH)
    owner.acquire()
    manager = None
    drained = True
    translation_stop: threading.Event | None = None
    translation_reconciler: threading.Thread | None = None
    try:
        get_engine()
        try:
            recover_interrupted_jobs()
        except Exception as exc:
            logger.exception("Interrupted job recovery failed during startup")
            raise RuntimeError(
                "Interrupted job recovery failed; refusing to serve stale job state. "
                "Check the logged job IDs, surviving pipeline processes and /proc "
                "visibility; preserve the database PID fences and source files."
            ) from exc
        manager = get_manager()
        manager.start()
        if os.environ.get("ENABLE_TRANSLATION_API", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            translation_stop = threading.Event()
            translation_reconciler = threading.Thread(
                target=_translation_handoff_reconcile_loop,
                args=(translation_stop,),
                name="translation-handoff-reconciler",
                daemon=True,
            )
            translation_reconciler.start()
        yield
    finally:
        try:
            timeout = float(os.environ.get("JOB_MANAGER_SHUTDOWN_TIMEOUT", "30"))
            if not math.isfinite(timeout) or timeout < 0:
                timeout = 30.0
        except (TypeError, ValueError):
            timeout = 30.0
        shutdown_deadline = time.monotonic() + timeout
        if translation_stop is not None:
            translation_stop.set()
        if translation_reconciler is not None:
            translation_reconciler.join(timeout=min(timeout, 1.0))
            if translation_reconciler.is_alive():
                drained = False
                logger.warning(
                    "Translation reconciler still blocked during shutdown; "
                    "daemon thread will not delay process exit"
                )
        if manager is not None:
            try:
                drained = manager.shutdown(
                    timeout=max(0.0, shutdown_deadline - time.monotonic())
                ) and drained
            except Exception:
                drained = False
                logger.exception("JobManager shutdown failed during FastAPI shutdown")
        try:
            from api.webhooks import shutdown_webhook_pool

            # Registered delivery writers share the remaining deadline below.
            # Waiting inside the pool would ignore the lifespan timeout.
            shutdown_webhook_pool(wait=False)
        except Exception:
            drained = False
            logger.exception("Webhook delivery shutdown failed")
        # Redetect and progress writers can outlive their HTTP request/job row.
        # ASGI has stopped request admission; the manager has stopped producers.
        drained = wait_for_writers(
            timeout=max(0.0, shutdown_deadline - time.monotonic())
        ) and drained
        if drained:
            owner.release()
        else:
            undrained_owners.append(owner)
            logger.error("API ownership retained until process exit: shutdown is incomplete")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    # maintenance review: Gate OpenAPI docs exposure.  When EXPOSE_API_DOCS is false
    # (the default) the interactive docs, ReDoc, and raw schema endpoints
    # are not mounted, preventing unauthenticated API enumeration.
    docs_url = "/docs" if EXPOSE_API_DOCS else None
    redoc_url = "/redoc" if EXPOSE_API_DOCS else None
    openapi_url = "/openapi.json" if EXPOSE_API_DOCS else None

    app = FastAPI(
        title="Industrial Adaptive OCR API",
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

    # Validate configuration before any database or interrupted-job mutation.
    # Recovery belongs to the lifespan, after exclusive process ownership.
    validate_auth_configuration()
    # PH3-11: multi-worker rate-limit storage required in production.
    assert_rate_limit_storage_policy()
    # Forensic request audit cannot be disabled in production.
    assert_api_audit_policy()
    # S2-03: SQLite multi-worker not multi-process honest in production.
    assert_sqlite_single_writer_policy()

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
    # maintenance review: Reject oversized non-multipart bodies before downstream handlers
    # read them.  Added last so it runs first (Starlette's add_middleware
    # wraps the existing stack).
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_size=MAX_REQUEST_BODY_SIZE,
    )

    # CORS middleware — opt-in via CORS_ALLOWED_ORIGINS env var (maintenance review).
    # Empty default means no CORS middleware is added (conservative).
    # LT-08 / S2-12: refuse wildcard origins with credentials (especially production).
    if CORS_ALLOWED_ORIGINS:
        from fastapi.middleware.cors import CORSMiddleware

        assert_cors_origins_policy(CORS_ALLOWED_ORIGINS)
        origins = list(CORS_ALLOWED_ORIGINS)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "X-API-Key",
                "X-Request-Id",
            ],
        )
        logger.info("CORS enabled for origins: %s", CORS_ALLOWED_ORIGINS)

    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Register routers — batch must come before jobs to avoid path conflicts
    # (/api/v1/jobs/batch vs /api/v1/jobs/{job_id})
    app.include_router(batch.router)
    app.include_router(session_auth.router)
    app.include_router(jobs.router)
    app.include_router(health.router)
    app.include_router(engines.router)
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
    ENABLE_TRANSLATION_API = os.environ.get("ENABLE_TRANSLATION_API", "").lower() in (
        "1",
        "true",
        "yes",
    )
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
    ENABLE_DASHBOARD = os.environ.get("ENABLE_DASHBOARD", "").lower() in (
        "1",
        "true",
        "yes",
    )
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


async def _rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> Response:
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
