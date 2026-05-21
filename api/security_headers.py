"""Security headers middleware for the OCR REST API.

Injects standard HTTP security headers on every response.
CSP is skipped for OpenAPI/Swagger UI paths that require inline styles.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from api.config import CSP_POLICY

# Paths serving Swagger UI / ReDoc that need inline styles/scripts
_DOCS_PATHS = frozenset({"/docs", "/openapi.json", "/redoc"})

# Static security headers applied to every response
_STATIC_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds HTTP security headers to all responses.

    CSP (``Content-Security-Policy``) is applied only on non-docs paths
    because Swagger UI and ReDoc require inline scripts/styles.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        for header, value in _STATIC_HEADERS.items():
            response.headers[header] = value

        # Apply CSP only on non-docs paths
        if request.url.path not in _DOCS_PATHS:
            response.headers["Content-Security-Policy"] = CSP_POLICY

        return response
