"""LT-10: HttpOnly API key session cookie (BFF-lite residual close).

POST /api/v1/auth/session sets an HttpOnly Secure cookie holding the API key
so the browser console need not keep the key in sessionStorage/localStorage.
DELETE clears the cookie.

Does not replace strong SSO; it removes XSS-readable storage for the lab console.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.config import API_COOKIE_MAX_AGE
from api.limits import get_default_rate, limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

COOKIE_NAME = "ocr_api_key"
# 8 hours default session; parsed once by the bounded API config helper.
_COOKIE_MAX_AGE = API_COOKIE_MAX_AGE


class SessionBody(BaseModel):
    api_key: str = Field(..., min_length=1, max_length=512)


def _secure_cookie() -> bool:
    dep = (os.environ.get("DEPLOYMENT_ENV") or "").strip().lower()
    if dep in ("production", "prod"):
        return True
    return (os.environ.get("OCR_API_COOKIE_SECURE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


@router.post("/session")
@limiter.limit("5/minute")
async def create_session(
    request: Request, body: SessionBody, response: Response
) -> dict:
    """Set an HttpOnly cookie only for a currently valid API key (LT-10)."""
    del request
    key = body.api_key.strip()
    if not key:
        return JSONResponse(
            status_code=400,
            content={"error": "empty_key", "message": "api_key required"},
        )

    valid = False
    try:
        from api import config
        from api.auth import _safe_compare_api_key, get_api_key

        if config.ENABLE_MULTITENANCY:
            from api.database import get_session_factory
            from api.tenant_manager import hash_api_key, resolve_tenant_by_key

            session = get_session_factory()()
            try:
                valid = (
                    resolve_tenant_by_key(hash_api_key(key), session=session)
                    is not None
                )
            finally:
                session.close()

        # Legacy/break-glass deployments may authenticate with OCR_API_KEY.
        # Read both the auth module and the environment-backed config value so
        # this endpoint remains correct when test/runtime configuration is
        # reloaded after module import.
        expected_keys = {
            str(get_api_key() or "").strip(),
            str(getattr(config, "OCR_API_KEY", "") or "").strip(),
            os.environ.get("OCR_API_KEY", "").strip(),
        }
        valid = valid or any(
            _safe_compare_api_key(key, expected)
            for expected in expected_keys
            if expected
        )
    except Exception:
        logger.exception("Session key validation failed")
        valid = False

    if not valid:
        return JSONResponse(
            status_code=401,
            content={
                "error": "invalid_key",
                "message": "api_key is not a valid active API credential",
            },
        )
    response.set_cookie(
        key=COOKIE_NAME,
        value=key,
        httponly=True,
        secure=_secure_cookie(),
        samesite="strict",
        max_age=_COOKIE_MAX_AGE,
        path="/",
    )
    return {"status": "ok", "storage": "httpOnly_cookie", "max_age": _COOKIE_MAX_AGE}


@router.delete("/session")
async def clear_session(request: Request, response: Response) -> dict:
    """Clear HttpOnly API key cookie and terminate its live sockets.

    PR1133 AC2: an already-authenticated socket must not keep answering
    ``pong`` after logout.  The logged-out credential is read before the
    cookie is cleared and only sockets authenticated with that exact
    credential are invalidated.
    """
    raw_key = (request.cookies.get(COOKIE_NAME) or "").strip()
    if raw_key:
        try:
            from api.routers.ws import invalidate_ws_sessions_for_key

            closed = await invalidate_ws_sessions_for_key(raw_key)
            logger.info("Session logout invalidated %d socket(s)", closed)
        except Exception:
            logger.exception("WebSocket session invalidation failed")
            raise HTTPException(
                status_code=503, detail="Session logout incomplete; retry required"
            ) from None
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"status": "ok", "cleared": True}


@router.get("/session/probe", name="probe_session")
@limiter.limit(get_default_rate())
async def probe_session(request: Request) -> dict:
    """Confirm middleware authentication without requiring job-read scope."""
    identity = getattr(request.state, "identity", None)
    if identity is None or getattr(identity, "auth_method", "none") == "none":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "message": "Authentication required"},
        )
    return {"authenticated": True}


@router.get("/session")
async def session_status(request: Request) -> dict:
    """Report whether an HttpOnly session cookie is present (value not returned)."""
    present = bool(request.cookies.get(COOKIE_NAME))
    return {"cookie_present": present, "cookie_name": COOKIE_NAME}
