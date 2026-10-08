"""WebSocket endpoint for real-time job progress streaming.

NOTE: The in-memory ``_connections`` dict limits this to single-worker
deployments.  For multi-worker (gunicorn --workers >1) you would need a
shared pub/sub backend (e.g. Redis Pub/Sub or a broadcast library).
Authenticated clients should either:
- send ``X-API-Key`` during the WebSocket handshake, or
- send ``{"type": "auth", "api_key": "..."}`` as the first frame after connect,
  or
- send ``{"type": "auth", "token": "..."}`` as the first frame (OAuth2 bearer).

This avoids long-lived credentials in query strings, which are commonly exposed
in logs, browser history, and proxy traces (maintenance review).  When the operator console
uses its opt-in HttpOnly session, the browser WebSocket handshake carries the
same-origin session cookie and no script-readable API key is available for an
auth frame.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from api.config import OCR_API_KEY
from api.database import Job, get_session_factory
from api.identity import VALID_ROLES, AuthIdentity
from ocr_distributed.ocr_utils import get_env_int as _safe_int

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)

# Active WebSocket connections: {job_id: set of WebSocket}
# Single-worker only; see module docstring for scaling notes.
_connections: dict[str, set[WebSocket]] = {}
_ws_admission_lock = threading.Lock()
_pending_connections = 0
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# Maximum concurrent WebSocket connections (DoS mitigation)
MAX_WS_CONNECTIONS = 100
WS_AUTH_TIMEOUT_SECONDS = 5.0
WS_AUTH_COOKIE_SUBPROTOCOL = "ocr-local.auth.cookie.v1"
WS_AUTH_FRAME_SUBPROTOCOL = "ocr-local.auth.frame.v1"
_WS_AUTH_SUBPROTOCOLS = frozenset(
    {WS_AUTH_COOKIE_SUBPROTOCOL, WS_AUTH_FRAME_SUBPROTOCOL}
)
WS_MAX_MESSAGE_BYTES = _safe_int(
    "WS_MAX_MESSAGE_BYTES",
    64 * 1024,
    min_val=1024,
    max_val=4 * 1024 * 1024,
)


def _ws_message_within_limit(message: object) -> bool:
    """Bound authentication and keepalive frames before JSON parsing."""
    if not isinstance(message, str):
        return False
    return len(message.encode("utf-8")) <= WS_MAX_MESSAGE_BYTES


def _reserve_ws_slot() -> bool:
    """Reserve capacity before accepting an authentication handshake."""
    global _pending_connections  # noqa: PLW0603
    with _ws_admission_lock:
        active = sum(len(sockets) for sockets in _connections.values())
        if active + _pending_connections >= MAX_WS_CONNECTIONS:
            return False
        _pending_connections += 1
        return True


def _release_ws_slot() -> None:
    """Release a pending handshake reservation."""
    global _pending_connections  # noqa: PLW0603
    with _ws_admission_lock:
        if _pending_connections:
            _pending_connections -= 1


def _credential_fingerprint(raw_key: str) -> str:
    """Return a non-reversible fingerprint for an API-key credential."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _mark_ws_credential(websocket: WebSocket, raw_key: str) -> None:
    """Pin the presented credential fingerprint onto the socket state.

    Lets session DELETE invalidate exactly the sockets that authenticated
    with the logged-out credential (PR1133 logout gap).
    """
    if raw_key:
        websocket.state.ws_cred_fp = _credential_fingerprint(raw_key)


# Sockets invalidated by session DELETE (ids).  The handler loop checks
# membership before answering ping/pong so a logged-out socket gets
# close/no-pong instead of another pong (PR1133 AC2).
_invalidated_ws_ids: set[int] = set()


async def invalidate_ws_sessions_for_key(raw_key: str) -> int:
    """Terminate live sockets authenticated with ``raw_key`` (logout).

    Marks matching sockets invalidated (the handler loop self-closes and
    stops answering ping) and best-effort closes them now.  Scoped to the
    presented credential so one logout cannot DoS unrelated sockets.
    Returns the number of sockets invalidated.
    """
    target = (
        _credential_fingerprint(raw_key.strip()) if raw_key and raw_key.strip() else ""
    )
    if not target:
        return 0
    with _ws_admission_lock:
        victims = [
            ws
            for sockets in _connections.values()
            for ws in sockets
            if getattr(getattr(ws, "state", None), "ws_cred_fp", None) == target
        ]
        for ws in victims:
            _invalidated_ws_ids.add(id(ws))
    for ws in victims:
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=4001, reason="Session terminated")
        except Exception:
            pass
    if victims:
        logger.info("Invalidated %d WebSocket session(s) on logout", len(victims))
    return len(victims)


# maintenance review: Per-connection idle + max-duration limits to prevent WebSocket
# connection exhaustion DoS.  Idle = no client frames received within the
# configured window.  Duration = wall-clock time since the connection was
# accepted.  Both are bounded via env vars for operator tuning.
WS_IDLE_TIMEOUT_SECONDS = _safe_int(
    "WS_IDLE_TIMEOUT_SECONDS",
    300,  # 5 minutes
    min_val=0,
    max_val=24 * 60 * 60,
)
WS_MAX_DURATION_SECONDS = _safe_int(
    "WS_MAX_DURATION_SECONDS",
    3600,  # 1 hour
    min_val=0,
    max_val=24 * 60 * 60,
)


def _now_monotonic() -> float:
    """Return a monotonic timestamp for maintenance review timeout accounting.

    Wrapped in a module-level function so tests can mock the clock
    without clobbering ``time.monotonic`` globally (which would break
    asyncio's internal timers).
    """
    return time.monotonic()


@contextmanager
def _db_session():
    """Yield a short-lived DB session and ensure it is closed."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def _cookie_origin_allowed(websocket: WebSocket) -> bool:
    """Require a same-origin or exact configured Origin for cookie auth."""
    from api.auth import _canonical_http_origin
    from api.config import CORS_ALLOWED_ORIGINS

    headers = getattr(websocket, "headers", None)
    if not hasattr(headers, "get"):
        return False
    origin = _canonical_http_origin(headers.get("origin", ""))
    if origin is None:
        return False

    scope = getattr(websocket, "scope", {})
    scheme = scope.get("scheme", "") if isinstance(scope, dict) else ""
    if not scheme:
        try:
            scheme = websocket.url.scheme
        except (AttributeError, TypeError, ValueError):
            scheme = ""
    scheme = str(scheme).lower()
    if scheme == "ws":
        scheme = "http"
    elif scheme == "wss":
        scheme = "https"
    request_host = headers.get("host", "")
    same_origin = _canonical_http_origin(f"{scheme}://{request_host}")
    if same_origin is not None and origin == same_origin:
        return True

    return any(
        str(configured).strip() != "*"
        and origin == _canonical_http_origin(str(configured).strip())
        for configured in CORS_ALLOWED_ORIGINS
    )


def _select_auth_subprotocol(websocket: WebSocket) -> tuple[Optional[str], bool]:
    """Select one explicit auth mode, preserving no-selector legacy clients."""
    scope = getattr(websocket, "scope", {})
    if not isinstance(scope, dict):
        return None, False
    offered = scope.get("subprotocols") or []
    if not isinstance(offered, (list, tuple)):
        return None, False
    if not offered:
        return None, True
    if len(offered) != 1:
        return None, False
    selected = offered[0]
    return (
        (selected, selected in _WS_AUTH_SUBPROTOCOLS)
        if isinstance(selected, str)
        else (None, False)
    )


def _ambient_api_cookie_present(websocket: WebSocket) -> bool:
    """Return whether the handshake includes the session cookie, even if blank."""
    cookies = getattr(websocket, "cookies", None)
    return hasattr(cookies, "get") and cookies.get("ocr_api_key") is not None


def _header_api_key(websocket: WebSocket) -> str:
    headers = getattr(websocket, "headers", None)
    key = headers.get("x-api-key", "") if hasattr(headers, "get") else ""
    return key if isinstance(key, str) else ""


async def _accept_ws(websocket: WebSocket, auth_subprotocol: Optional[str]) -> None:
    if auth_subprotocol is None:
        await websocket.accept()
    else:
        await websocket.accept(subprotocol=auth_subprotocol)


async def _authenticate_ws(
    websocket: WebSocket,
    api_key: Optional[str] = None,
    auth_subprotocol: Optional[str] = None,
) -> bool:
    """Authenticate a WebSocket using cookie/header/auth-frame API keys or OAuth2 token.

    OAuth2 bearer tokens must be sent via the auth frame
    (``{"type": "auth", "token": "..."}``) -- never as a query parameter, to
    avoid credential leakage in proxy logs and browser history (maintenance review).
    """
    from api.auth import _safe_compare_api_key, is_auth_enforced
    from api.oauth2_config import APIKEY_ROLE, OAUTH2_ENABLED

    def _set_apikey_identity() -> bool:
        websocket.state.identity = AuthIdentity(
            subject="apikey",
            role=APIKEY_ROLE,
            auth_method="apikey",
        )
        return True

    def _try_tenant_api_key(provided: str) -> bool:
        """Resolve multi-tenant API key and pin websocket.state.tenant_id (RC-02)."""
        from api.config import ENABLE_MULTITENANCY

        if not ENABLE_MULTITENANCY or not provided:
            return False
        try:
            from api.auth import (
                _identity_role_from_permissions,
                _parse_tenant_permissions,
            )
            from api.database import get_session_factory
            from api.tenant_manager import hash_api_key, resolve_tenant_by_key

            key_hash = hash_api_key(provided)
            session = get_session_factory()()
            try:
                result = resolve_tenant_by_key(key_hash, session=session)
                if result is None:
                    return False
                tenant, key_record = result
                permissions = _parse_tenant_permissions(key_record.permissions)
                if permissions is None:
                    logger.warning(
                        "Rejecting WebSocket tenant key %s with malformed permissions",
                        getattr(key_record, "key_id", "unknown"),
                    )
                    return False
                websocket.state.tenant_id = tenant.tenant_id
                websocket.state.tenant_permissions = permissions
                websocket.state.identity = AuthIdentity(
                    subject=tenant.tenant_id,
                    role=_identity_role_from_permissions(permissions),
                    auth_method="apikey",
                    claims={
                        "tenant_id": tenant.tenant_id,
                        "permissions": permissions,
                    },
                )
                _mark_ws_credential(websocket, provided)
                return True
            finally:
                session.close()
        except Exception:
            logger.exception("WebSocket multi-tenant key resolution failed")
            return False

    def _try_oauth2_token(token_value: str) -> bool:
        """Validate an OAuth2 bearer token and set identity on success.

        RC-02: tenant is pinned through ``apply_oauth_tenant_binding`` (claim
        path + membership map + tenant existence), matching HTTP. A raw JWT
        tenant claim is never enough for fleet or tenant-scoped open.
        """
        if not OAUTH2_ENABLED:
            return False
        try:
            from api.oauth2 import extract_role, validate_jwt

            claims = validate_jwt(token_value)
            role = extract_role(claims)
            if role in VALID_ROLES:
                identity = AuthIdentity(
                    subject=claims.get("sub", "unknown"),
                    role=role,
                    auth_method="oauth2",
                    email=claims.get("email"),
                    name=claims.get("name"),
                    claims=claims,
                )
                websocket.state.identity = identity
                # Same membership + existence pin as HTTP (not a raw JWT claim).
                from api.auth import apply_oauth_tenant_binding

                apply_oauth_tenant_binding(websocket.state, identity)
                return True
        except (ValueError, ImportError, RuntimeError):
            pass
        return False

    header_key = _header_api_key(websocket)

    if auth_subprotocol not in (
        None,
        WS_AUTH_COOKIE_SUBPROTOCOL,
        WS_AUTH_FRAME_SUBPROTOCOL,
    ):
        return False

    # Native browser WebSocket clients cannot set arbitrary request headers,
    # but they do carry cookies. Cookie auth requires an explicit cookie-mode
    # selector and a trusted Origin; frame mode never falls back to ambient
    # cookie authority.
    if (
        auth_subprotocol != WS_AUTH_FRAME_SUBPROTOCOL
        and api_key is None
        and not header_key
    ):
        raw_cookies = getattr(websocket, "cookies", None)
        cookie_key = (
            raw_cookies.get("ocr_api_key", "") if hasattr(raw_cookies, "get") else ""
        )
        if isinstance(cookie_key, str) and cookie_key:
            if not _cookie_origin_allowed(websocket):
                return False
            api_key = cookie_key

    if (
        auth_subprotocol == WS_AUTH_COOKIE_SUBPROTOCOL
        and api_key is None
        and not header_key
    ):
        # Cookie mode cannot fall back to a queued auth frame.
        return False

    # Multi-tenant API keys first (pin tenant_id on the socket state).
    if api_key is not None and _try_tenant_api_key(api_key):
        return True
    if header_key and _try_tenant_api_key(header_key):
        return True

    if OCR_API_KEY:
        if api_key is not None:
            ok = (
                bool(api_key)
                and _safe_compare_api_key(api_key, OCR_API_KEY)
                and _set_apikey_identity()
            )
            if ok:
                _mark_ws_credential(websocket, api_key)
                # Legacy key under MT gets platform permissions (same as HTTP).
                from api.config import ENABLE_MULTITENANCY

                if ENABLE_MULTITENANCY:
                    from api.authz import LEGACY_PLATFORM_PERMISSIONS

                    websocket.state.tenant_permissions = list(
                        LEGACY_PLATFORM_PERMISSIONS
                    )
                    websocket.state.tenant_id = None
            return ok

        if header_key and _safe_compare_api_key(header_key, OCR_API_KEY):
            _set_apikey_identity()
            _mark_ws_credential(websocket, header_key)
            from api.config import ENABLE_MULTITENANCY

            if ENABLE_MULTITENANCY:
                from api.authz import LEGACY_PLATFORM_PERMISSIONS

                websocket.state.tenant_permissions = list(LEGACY_PLATFORM_PERMISSIONS)
                websocket.state.tenant_id = None
            return True

    if auth_subprotocol == WS_AUTH_COOKIE_SUBPROTOCOL:
        # A rejected cookie or explicit key must not fall through to a frame.
        return False

    # Wait for auth frame (API key or OAuth2 token)
    from api.config import ENABLE_MULTITENANCY as _MT

    if OCR_API_KEY or OAUTH2_ENABLED or _MT:
        if api_key is not None:
            # Already tried via header path above; don't wait for frame
            return False

        receive_text = getattr(websocket, "receive_text", None)
        if receive_text is None:
            return False

        try:
            receive_text_result = receive_text()
        except TypeError:
            return False

        if not inspect.isawaitable(receive_text_result):
            return False

        try:
            raw_message = await asyncio.wait_for(
                receive_text_result,
                timeout=WS_AUTH_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, WebSocketDisconnect):
            return False
        except (KeyError, TypeError, UnicodeDecodeError):
            # Starlette's receive_text() raises when a binary frame arrives.
            # Treat it as an unsupported data type rather than leaking a
            # parser exception or leaving the handshake half-open.
            try:
                await websocket.close(code=1003, reason="Text frames required")
            except Exception:
                pass
            return False

        if not _ws_message_within_limit(raw_message):
            return False

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return False

        if isinstance(payload, dict) and payload.get("type") == "auth":
            # API key via auth frame (tenant key or legacy)
            if isinstance(payload.get("api_key"), str):
                if _try_tenant_api_key(payload["api_key"]):
                    return True
                if OCR_API_KEY and _safe_compare_api_key(
                    payload["api_key"], OCR_API_KEY
                ):
                    _set_apikey_identity()
                    _mark_ws_credential(websocket, payload["api_key"])
                    if _MT:
                        from api.authz import LEGACY_PLATFORM_PERMISSIONS

                        websocket.state.tenant_permissions = list(
                            LEGACY_PLATFORM_PERMISSIONS
                        )
                        websocket.state.tenant_id = None
                    return True
                return False
            # OAuth2 token via auth frame (replaces deprecated ?token= query param -- maintenance review)
            if isinstance(payload.get("token"), str):
                return _try_oauth2_token(payload["token"])
        return False

    # BE001 AC3: same single helper as HTTP — WS and HTTP always agree.
    if not is_auth_enforced():
        from api.identity import _anonymous_role

        websocket.state.identity = AuthIdentity(
            subject="anonymous",
            role=_anonymous_role(),
            auth_method="none",
        )
        return True

    return False


async def _send_json(ws: WebSocket, data: dict) -> bool:
    """Send JSON message to WebSocket, return False if connection closed."""
    try:
        if id(ws) in _invalidated_ws_ids:
            return False
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json(data)
            return True
    except Exception:
        return False
    return False


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(
    websocket: WebSocket,
    job_id: str,
):
    """Stream real-time progress updates for an OCR job.

    Messages sent (server -> client):
    - {"type": "connected", "job_id": "...", "status": "..."}
    - {"type": "progress", "job_id": "...", "status": "..."}
    - {"type": "completed", "job_id": "...", "status": "completed", "result_available": true}
    - {"type": "failed", "job_id": "...", "status": "failed", "error": "..."}
    - {"type": "cancelled", "job_id": "...", "status": "cancelled"}
    - {"type": "error", "message": "..."}
    - {"type": "pong"} (in response to "ping")

    Auth frames (client -> server, must be the first frame):
    - {"type": "auth", "api_key": "..."} -- API key authentication
    - {"type": "auth", "token": "..."} -- OAuth2 bearer token (replaces
      deprecated ``?token=`` query param; see maintenance review)
    """
    # HTTP middleware does not run for WebSocket scopes. Enforce its ingress
    # allowlist before accepting a socket or reserving admission capacity.
    from api.auth import get_allowed_ips

    allowed_ips = get_allowed_ips()
    client = getattr(websocket, "client", None)
    if allowed_ips and (client.host if client else "") not in allowed_ips:
        await websocket.close(code=1008, reason="Client is not allowlisted")
        return

    auth_subprotocol, protocol_valid = _select_auth_subprotocol(websocket)
    if not protocol_valid:
        await websocket.accept()
        await _send_json(
            websocket,
            {
                "type": "error",
                "message": "Unsupported WebSocket authentication protocol",
            },
        )
        await websocket.close(code=1002, reason="Unsupported authentication protocol")
        return

    # Rate limit: reserve capacity before accepting.  Counting only
    # authenticated sockets lets unauthenticated clients hold the auth
    # timeout open and bypass the active-connection ceiling.
    slot_reserved = _reserve_ws_slot()
    if not slot_reserved:
        await _accept_ws(websocket, auth_subprotocol)
        await _send_json(
            websocket, {"type": "error", "message": "Too many connections"}
        )
        await websocket.close(code=4029, reason="Too many connections")
        return

    def release_reserved_slot() -> None:
        nonlocal slot_reserved
        if slot_reserved:
            _release_ws_slot()
            slot_reserved = False

    global _event_loop, _pending_connections

    try:
        await _accept_ws(websocket, auth_subprotocol)
        _event_loop = asyncio.get_running_loop()
    except BaseException:
        release_reserved_slot()
        raise

    # Authenticate after accept so frame-mode clients can send an auth frame.
    # An ambient cookie without a selector is ambiguous: retain legacy header
    # clients, but require cookie users to negotiate cookie mode explicitly.
    try:
        authenticated = False
        if auth_subprotocol is None and _ambient_api_cookie_present(websocket):
            header_key = _header_api_key(websocket)
            if header_key:
                authenticated = await _authenticate_ws(
                    websocket,
                    api_key=header_key,
                    auth_subprotocol=auth_subprotocol,
                )
                identity = getattr(websocket.state, "identity", None)
                authenticated = authenticated and (
                    getattr(identity, "auth_method", None) == "apikey"
                )
            if not authenticated:
                release_reserved_slot()
                await _send_json(
                    websocket,
                    {
                        "type": "error",
                        "message": "Unsupported WebSocket authentication protocol",
                    },
                )
                await websocket.close(
                    code=1002,
                    reason="Unsupported authentication protocol",
                )
                return
        else:
            authenticated = await _authenticate_ws(
                websocket,
                auth_subprotocol=auth_subprotocol,
            )
    except BaseException:
        release_reserved_slot()
        raise
    if not authenticated:
        release_reserved_slot()
        await _send_json(
            websocket,
            {"type": "error", "message": "Authentication failed"},
        )
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Tenant scope (RC-02): under multitenancy require tenant-scoped API key or
    # platform_admin — never open all jobs when tenant_id is unset (OAuth /
    # legacy fail-open residual).
    from api.authz import is_platform_admin, require_tenant_scope_or_platform
    from api.config import ENABLE_MULTITENANCY

    tenant_id = getattr(websocket.state, "tenant_id", None)
    # Only trust tenant_id from multi-tenant API-key auth (server-set), not
    # free-form OAuth claims (same trust boundary as R4-RC-11).
    try:
        # Build a minimal request-like adapter for authz helpers.
        class _WSReq:
            def __init__(self, state):
                self.state = state

        ws_req = _WSReq(websocket.state)
        if ENABLE_MULTITENANCY and not is_platform_admin(ws_req):
            tenant_id = require_tenant_scope_or_platform(ws_req)
        elif is_platform_admin(ws_req):
            tenant_id = None  # platform may open any job
    except Exception as exc:
        # HTTPException from require_tenant_scope_or_platform
        from fastapi import HTTPException as _HTTPException

        if isinstance(exc, _HTTPException):
            release_reserved_slot()
            await _send_json(
                websocket,
                {
                    "type": "error",
                    "message": "Tenant scope required for WebSocket job access",
                },
            )
            await websocket.close(code=4003, reason="Tenant scope required")
            return
        release_reserved_slot()
        raise

    # Submit-only keys cannot browse evidence over WebSocket either.
    try:
        from types import SimpleNamespace

        from api.authz import enforce_tenant_capability

        ws_req.method = "GET"
        ws_req.url = SimpleNamespace(path=f"/api/v1/jobs/{job_id}/ws")
        enforce_tenant_capability(ws_req)
    except Exception as exc:
        from fastapi import HTTPException as _HTTPException

        if isinstance(exc, _HTTPException):
            release_reserved_slot()
            await _send_json(
                websocket,
                {
                    "type": "error",
                    "message": "This operation requires tenant capability 'read'.",
                },
            )
            await websocket.close(code=4003, reason="Tenant capability required")
            return
        release_reserved_slot()
        raise

    # Verify job exists and belongs to this tenant
    try:
        with _db_session() as session:
            query = session.query(Job).filter(Job.job_id == job_id)
            if tenant_id is not None:
                query = query.filter(Job.tenant_id == tenant_id)
            job = query.first()
            if not job:
                release_reserved_slot()
                await _send_json(
                    websocket, {"type": "error", "message": f"Job {job_id} not found"}
                )
                await websocket.close(code=4004, reason="Job not found")
                return
            initial_status = job.status
    except BaseException:
        release_reserved_slot()
        raise

    try:
        # Register connection inside try so finally always cleans up (maintenance review)
        with _ws_admission_lock:
            if job_id not in _connections:
                _connections[job_id] = set()
            _connections[job_id].add(websocket)
            _pending_connections -= 1
            slot_reserved = False

        logger.info("WebSocket connected for job %s", job_id)

        # Send initial status
        await _send_json(
            websocket,
            {
                "type": "connected",
                "job_id": job_id,
                "status": initial_status,
            },
        )

        # If already terminal, send final status and close
        if initial_status in ("completed", "failed", "cancelled"):
            terminal_payload = {
                "type": initial_status,
                "job_id": job_id,
                "status": initial_status,
            }
            if initial_status == "completed":
                # Keep the terminal-on-connect contract equivalent to the
                # transition path without exposing the persisted filesystem
                # path.
                terminal_payload["result_available"] = bool(job.result_path)
            await _send_json(websocket, terminal_payload)
            return

        # Poll for updates (1 second interval).  Track last_activity and
        # connect_start for maintenance review idle + max-duration timeouts.
        last_status = initial_status
        connect_start = _now_monotonic()
        last_activity = connect_start
        while True:
            try:
                # maintenance review: Enforce max connection duration regardless of
                # activity.  Kills long-lived leaked connections that a
                # caller "forgot" about.
                now = _now_monotonic()
                if (
                    WS_MAX_DURATION_SECONDS > 0
                    and (now - connect_start) >= WS_MAX_DURATION_SECONDS
                ):
                    logger.info(
                        "WebSocket max duration (%ds) reached for job %s; "
                        "closing connection",
                        WS_MAX_DURATION_SECONDS,
                        job_id,
                    )
                    await _send_json(
                        websocket,
                        {
                            "type": "error",
                            "message": "Max connection duration exceeded",
                        },
                    )
                    try:
                        await websocket.close(code=4008, reason="Max duration exceeded")
                    except Exception:
                        pass
                    break

                # maintenance review: Enforce idle timeout.  If no client frame has
                # been received within the configured window, close the
                # connection.
                if (
                    WS_IDLE_TIMEOUT_SECONDS > 0
                    and (now - last_activity) >= WS_IDLE_TIMEOUT_SECONDS
                ):
                    logger.info(
                        "WebSocket idle timeout (%ds) reached for job %s; "
                        "closing connection",
                        WS_IDLE_TIMEOUT_SECONDS,
                        job_id,
                    )
                    await _send_json(
                        websocket,
                        {
                            "type": "error",
                            "message": "Idle timeout exceeded",
                        },
                    )
                    try:
                        await websocket.close(code=4008, reason="Idle timeout")
                    except Exception:
                        pass
                    break

                # PR1133: a logged-out socket must not answer again.  Check
                # before blocking on receive (fast close) and again after
                # (a ping that arrived before DELETE must get no pong).
                if id(websocket) in _invalidated_ws_ids:
                    try:
                        await websocket.close(code=4001, reason="Session terminated")
                    except Exception:
                        pass
                    break

                # Check for incoming messages (ping/close)
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    if id(websocket) in _invalidated_ws_ids:
                        try:
                            await websocket.close(
                                code=4001, reason="Session terminated"
                            )
                        except Exception:
                            pass
                        break
                    # Any client frame counts as activity for idle accounting.
                    last_activity = _now_monotonic()
                    if not _ws_message_within_limit(msg):
                        await _send_json(
                            websocket,
                            {
                                "type": "error",
                                "message": "WebSocket message exceeds size limit",
                            },
                        )
                        await websocket.close(
                            code=1009,
                            reason="WebSocket message exceeds size limit",
                        )
                        break
                    if msg == "ping":
                        await _send_json(websocket, {"type": "pong"})
                except asyncio.TimeoutError:
                    pass
                except WebSocketDisconnect:
                    break
                except (KeyError, TypeError, UnicodeDecodeError):
                    try:
                        await websocket.close(
                            code=1003,
                            reason="Text frames required",
                        )
                    except Exception:
                        pass
                    break

                # Query current job status (tenant-scoped)
                with _db_session() as session:
                    ws_query = session.query(Job).filter(Job.job_id == job_id)
                    if tenant_id is not None:
                        ws_query = ws_query.filter(Job.tenant_id == tenant_id)
                    job = ws_query.first()
                    if not job:
                        await _send_json(
                            websocket,
                            {"type": "error", "message": "Job disappeared"},
                        )
                        break

                    current_status = job.status

                    # Send progress update if status changed
                    if current_status != last_status:
                        if current_status == "completed":
                            await _send_json(
                                websocket,
                                {
                                    "type": "completed",
                                    "job_id": job_id,
                                    "status": "completed",
                                    # Do not disclose the absolute filesystem
                                    # path; REST clients use the authenticated
                                    # artifact route to retrieve results.
                                    "result_available": bool(job.result_path),
                                },
                            )
                            break
                        elif current_status == "failed":
                            await _send_json(
                                websocket,
                                {
                                    "type": "failed",
                                    "job_id": job_id,
                                    "status": "failed",
                                    "error": job.error_message or "",
                                },
                            )
                            break
                        elif current_status == "cancelled":
                            await _send_json(
                                websocket,
                                {
                                    "type": "cancelled",
                                    "job_id": job_id,
                                    "status": "cancelled",
                                },
                            )
                            break
                        else:
                            await _send_json(
                                websocket,
                                {
                                    "type": "progress",
                                    "job_id": job_id,
                                    "status": current_status,
                                },
                            )
                        last_status = current_status

            except WebSocketDisconnect:
                break

    finally:
        # Cleanup connection
        with _ws_admission_lock:
            if job_id in _connections:
                _connections[job_id].discard(websocket)
                if not _connections[job_id]:
                    del _connections[job_id]
            _invalidated_ws_ids.discard(id(websocket))
        release_reserved_slot()
        logger.info("WebSocket disconnected for job %s", job_id)


async def notify_job_update(job_id: str, data: dict) -> None:
    """Broadcast a job update to all connected WebSocket clients.

    Called from job_manager when job status changes.
    """
    with _ws_admission_lock:
        sockets = set(_connections.get(job_id, set()))
    if not sockets:
        return

    dead: set[WebSocket] = set()
    for ws in sockets:
        if not await _send_json(ws, data):
            dead.add(ws)

    # Cleanup dead connections
    with _ws_admission_lock:
        if job_id in _connections:
            _connections[job_id] -= dead
            if not _connections[job_id]:
                del _connections[job_id]


def _log_notify_future(future) -> None:
    """Log background websocket notification failures without raising."""
    try:
        future.result()
    except Exception:
        logger.exception("WebSocket notification task failed")


def notify_job_update_sync(job_id: str, data: dict) -> None:
    """Schedule a job update broadcast from synchronous code."""
    loop = _event_loop
    if loop is None or loop.is_closed():
        return

    coroutine = notify_job_update(job_id, data)
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        future.add_done_callback(_log_notify_future)
        return

    if running_loop is loop:
        running_loop.create_task(coroutine)
        return

    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    future.add_done_callback(_log_notify_future)
