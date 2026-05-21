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
in logs, browser history, and proxy traces.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import secrets
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
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# Maximum concurrent WebSocket connections (DoS mitigation)
MAX_WS_CONNECTIONS = 100
WS_AUTH_TIMEOUT_SECONDS = 5.0

# Per-connection idle + max-duration limits to prevent WebSocket
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
    """Return a monotonic timestamp for  timeout accounting.

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


async def _authenticate_ws(
    websocket: WebSocket,
    api_key: Optional[str] = None,
) -> bool:
    """Authenticate a WebSocket using header/auth-frame API keys or OAuth2 token.

    OAuth2 bearer tokens must be sent via the auth frame
    (``{"type": "auth", "token": "..."}``) -- never as a query parameter, to
    avoid credential leakage in proxy logs and browser history.
    """
    from api.auth import allow_unauthenticated
    from api.oauth2_config import APIKEY_ROLE, OAUTH2_ENABLED

    def _set_apikey_identity() -> bool:
        websocket.state.identity = AuthIdentity(
            subject="apikey",
            role=APIKEY_ROLE,
            auth_method="apikey",
        )
        return True

    def _try_oauth2_token(token_value: str) -> bool:
        """Validate an OAuth2 bearer token and set identity on success."""
        if not OAUTH2_ENABLED:
            return False
        try:
            from api.oauth2 import extract_role, validate_jwt

            claims = validate_jwt(token_value)
            role = extract_role(claims)
            if role in VALID_ROLES:
                websocket.state.identity = AuthIdentity(
                    subject=claims.get("sub", "unknown"),
                    role=role,
                    auth_method="oauth2",
                    email=claims.get("email"),
                    name=claims.get("name"),
                    claims=claims,
                )
                return True
        except (ValueError, ImportError, RuntimeError):
            pass
        return False

    if OCR_API_KEY:
        if api_key is not None:
            return bool(api_key) and secrets.compare_digest(api_key, OCR_API_KEY) and _set_apikey_identity()

        raw_headers = getattr(websocket, "headers", None)
        header_key = (
            raw_headers.get("x-api-key", "")
            if hasattr(raw_headers, "get")
            else ""
        )
        if not isinstance(header_key, str):
            header_key = ""
        if header_key and secrets.compare_digest(header_key, OCR_API_KEY):
            return _set_apikey_identity()

    # Wait for auth frame (API key or OAuth2 token)
    if OCR_API_KEY or OAUTH2_ENABLED:
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

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return False

        if isinstance(payload, dict) and payload.get("type") == "auth":
            # API key via auth frame
            if isinstance(payload.get("api_key"), str) and OCR_API_KEY:
                if secrets.compare_digest(payload["api_key"], OCR_API_KEY):
                    return _set_apikey_identity()
                return False
            # OAuth2 token via auth frame (replaces deprecated ?token= query param -- )
            if isinstance(payload.get("token"), str):
                return _try_oauth2_token(payload["token"])
        return False

    if not OCR_API_KEY and not OAUTH2_ENABLED and allow_unauthenticated():
        from api.config import ANONYMOUS_ROLE

        websocket.state.identity = AuthIdentity(
            subject="anonymous",
            role=ANONYMOUS_ROLE,
            auth_method="none",
        )
        return True

    return False


async def _send_json(ws: WebSocket, data: dict) -> bool:
    """Send JSON message to WebSocket, return False if connection closed."""
    try:
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
    - {"type": "completed", "job_id": "...", "status": "completed", "output_path": "..."}
    - {"type": "failed", "job_id": "...", "status": "failed", "error": "..."}
    - {"type": "cancelled", "job_id": "...", "status": "cancelled"}
    - {"type": "error", "message": "..."}
    - {"type": "pong"} (in response to "ping")

    Auth frames (client -> server, must be the first frame):
    - {"type": "auth", "api_key": "..."} -- API key authentication
    - {"type": "auth", "token": "..."} -- OAuth2 bearer token (replaces
      deprecated ``?token=`` query param; see )
    """
    # Rate limit: cap total concurrent WebSocket connections
    total = sum(len(s) for s in _connections.values())
    if total >= MAX_WS_CONNECTIONS:
        await websocket.accept()
        await _send_json(
            websocket, {"type": "error", "message": "Too many connections"}
        )
        await websocket.close(code=4029, reason="Too many connections")
        return

    global _event_loop

    await websocket.accept()
    _event_loop = asyncio.get_running_loop()

    # Authenticate after accept so browser clients can send an auth frame.
    if not await _authenticate_ws(websocket):
        await _send_json(
            websocket,
            {"type": "error", "message": "Authentication failed"},
        )
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Extract tenant scope (set by multi-tenant auth or from identity claims)
    tenant_id = getattr(websocket.state, "tenant_id", None)
    if tenant_id is None:
        ws_identity = getattr(websocket.state, "identity", None)
        if ws_identity is not None:
            tenant_id = getattr(ws_identity, "claims", {}).get("tenant_id")

    # Verify job exists and belongs to this tenant
    with _db_session() as session:
        query = session.query(Job).filter(Job.job_id == job_id)
        if tenant_id is not None:
            query = query.filter(Job.tenant_id == tenant_id)
        job = query.first()
        if not job:
            await _send_json(
                websocket, {"type": "error", "message": f"Job {job_id} not found"}
            )
            await websocket.close(code=4004, reason="Job not found")
            return
        initial_status = job.status

    try:
        # Register connection inside try so finally always cleans up
        if job_id not in _connections:
            _connections[job_id] = set()
        _connections[job_id].add(websocket)

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
            await _send_json(
                websocket,
                {
                    "type": initial_status,
                    "job_id": job_id,
                    "status": initial_status,
                },
            )
            return

        # Poll for updates (1 second interval).  Track last_activity and
        # connect_start for  idle + max-duration timeouts.
        last_status = initial_status
        connect_start = _now_monotonic()
        last_activity = connect_start
        while True:
            try:
                # Enforce max connection duration regardless of
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
                        await websocket.close(
                            code=4008, reason="Max duration exceeded"
                        )
                    except Exception:
                        pass
                    break

                # Enforce idle timeout.  If no client frame has
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
                        await websocket.close(
                            code=4008, reason="Idle timeout"
                        )
                    except Exception:
                        pass
                    break

                # Check for incoming messages (ping/close)
                try:
                    msg = await asyncio.wait_for(
                        websocket.receive_text(), timeout=1.0
                    )
                    # Any client frame counts as activity for idle accounting.
                    last_activity = _now_monotonic()
                    if msg == "ping":
                        await _send_json(websocket, {"type": "pong"})
                except asyncio.TimeoutError:
                    pass

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
                                    "output_path": job.result_path or "",
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
        if job_id in _connections:
            _connections[job_id].discard(websocket)
            if not _connections[job_id]:
                del _connections[job_id]
        logger.info("WebSocket disconnected for job %s", job_id)


async def notify_job_update(job_id: str, data: dict) -> None:
    """Broadcast a job update to all connected WebSocket clients.

    Called from job_manager when job status changes.
    """
    if job_id not in _connections:
        return

    dead: set[WebSocket] = set()
    for ws in _connections[job_id]:
        if not await _send_json(ws, data):
            dead.add(ws)

    # Cleanup dead connections
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
