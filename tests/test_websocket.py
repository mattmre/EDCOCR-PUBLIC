"""Tests for WebSocket real-time job progress streaming."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from api.database import Job, get_session_factory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_ws_connections():
    """Reset WebSocket connection registry between tests."""
    from api.routers.ws import _connections

    _connections.clear()
    yield
    _connections.clear()


@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with isolated DB and temp dirs."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.OCR_API_KEY", ""),
        patch("api.routers.ws.OCR_API_KEY", ""),
        patch("api.auth.OCR_API_KEY", ""),
        patch("api.config.ALLOW_UNAUTHENTICATED", True),
        patch("api.auth.ALLOW_UNAUTHENTICATED", True),
        patch("api.job_manager.config") as mock_config,
    ):
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        yield TestClient(app)


@pytest.fixture()
def auth_client(tmp_path):
    """FastAPI TestClient with API key authentication enabled."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.OCR_API_KEY", "test-secret-key"),
        patch("api.routers.ws.OCR_API_KEY", "test-secret-key"),
        patch("api.routers.ws.WS_AUTH_TIMEOUT_SECONDS", 0.05),
        patch("api.auth.OCR_API_KEY", "test-secret-key"),
        patch("api.config.ALLOW_UNAUTHENTICATED", False),
        patch("api.auth.ALLOW_UNAUTHENTICATED", False),
        patch("api.job_manager.config") as mock_config,
    ):
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        yield TestClient(app)


def _create_job(status="submitted", **kwargs):
    """Insert a job record directly into the test database."""
    factory = get_session_factory()
    session = factory()
    job_id = kwargs.pop("job_id", "test-job-001")
    job = Job(
        job_id=job_id,
        status=status,
        source_file=kwargs.pop("source_file", "test.pdf"),
        priority=kwargs.pop("priority", "normal"),
        result_path=kwargs.pop("result_path", None),
        error_message=kwargs.pop("error_message", None),
    )
    for key, value in kwargs.items():
        setattr(job, key, value)
    session.add(job)
    session.commit()
    session.close()
    return job_id


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestWSAuthentication:
    def test_connect_without_auth_when_disabled(self, client):
        """WebSocket connects when no API key is configured."""
        _create_job(status="completed", job_id="auth-test-1")
        with client.websocket_connect("/ws/jobs/auth-test-1") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"

    def test_connect_with_valid_api_key_header(self, auth_client):
        """WebSocket connects with a valid X-API-Key header."""
        _create_job(status="completed", job_id="auth-test-2")
        with auth_client.websocket_connect(
            "/ws/jobs/auth-test-2",
            headers={"X-API-Key": "test-secret-key"},
        ) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"

    def test_connect_with_valid_auth_message(self, auth_client):
        """WebSocket connects when the first frame carries auth."""
        _create_job(status="completed", job_id="auth-test-2b")
        with auth_client.websocket_connect("/ws/jobs/auth-test-2b") as ws:
            ws.send_json({"type": "auth", "api_key": "test-secret-key"})
            msg = ws.receive_json()
            assert msg["type"] == "connected"

    def test_reject_invalid_api_key(self, auth_client):
        """WebSocket rejects connection with the wrong auth frame key."""
        _create_job(status="submitted", job_id="auth-test-3")
        with auth_client.websocket_connect("/ws/jobs/auth-test-3") as ws:
            ws.send_json({"type": "auth", "api_key": "wrong-key"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Authentication failed" in msg["message"]

    def test_reject_missing_api_key_when_required(self, auth_client):
        """WebSocket rejects connection when API key is required but missing."""
        _create_job(status="submitted", job_id="auth-test-4")
        with auth_client.websocket_connect("/ws/jobs/auth-test-4") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Authentication failed" in msg["message"]

    def test_reject_empty_api_key_when_required(self, auth_client):
        """WebSocket rejects connection with an empty auth frame key."""
        _create_job(status="submitted", job_id="auth-test-5")
        with auth_client.websocket_connect("/ws/jobs/auth-test-5") as ws:
            ws.send_json({"type": "auth", "api_key": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Authentication failed" in msg["message"]


# ---------------------------------------------------------------------------
# Connection lifecycle tests
# ---------------------------------------------------------------------------


class TestWSConnection:
    def test_connect_sends_initial_status(self, client):
        """On connect, server sends a 'connected' message with current status."""
        _create_job(status="processing", job_id="conn-test-1")
        with client.websocket_connect("/ws/jobs/conn-test-1") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert msg["job_id"] == "conn-test-1"
            assert msg["status"] == "processing"

    def test_connect_submitted_job(self, client):
        """Can connect to a job in submitted state."""
        _create_job(status="submitted", job_id="conn-test-2")
        with client.websocket_connect("/ws/jobs/conn-test-2") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert msg["status"] == "submitted"

    def test_disconnect_cleans_up_connections(self, client):
        """Connection registry is cleaned up after disconnect."""
        from api.routers.ws import _connections

        _create_job(status="completed", job_id="conn-test-3")
        with client.websocket_connect("/ws/jobs/conn-test-3"):
            pass  # connect and immediately disconnect

        # After context manager exits, connection should be cleaned up
        assert "conn-test-3" not in _connections or len(_connections.get("conn-test-3", set())) == 0


# ---------------------------------------------------------------------------
# Registry cleanup tests
# ---------------------------------------------------------------------------


class TestWSRegistryCleanup:
    """Verify _connections registry is always cleaned up, including early-return
    paths for terminal jobs."""

    def test_terminal_completed_cleans_registry(self, client):
        """Connecting to a completed job must leave _connections clean."""
        from api.routers.ws import _connections

        _create_job(status="completed", job_id="reg-term-1")
        with client.websocket_connect("/ws/jobs/reg-term-1") as ws:
            ws.receive_json()  # connected
            ws.receive_json()  # completed

        assert "reg-term-1" not in _connections

    def test_terminal_failed_cleans_registry(self, client):
        """Connecting to a failed job must leave _connections clean."""
        from api.routers.ws import _connections

        _create_job(status="failed", job_id="reg-term-2", error_message="err")
        with client.websocket_connect("/ws/jobs/reg-term-2") as ws:
            ws.receive_json()  # connected
            ws.receive_json()  # failed

        assert "reg-term-2" not in _connections

    def test_terminal_cancelled_cleans_registry(self, client):
        """Connecting to a cancelled job must leave _connections clean."""
        from api.routers.ws import _connections

        _create_job(status="cancelled", job_id="reg-term-3")
        with client.websocket_connect("/ws/jobs/reg-term-3") as ws:
            ws.receive_json()  # connected
            ws.receive_json()  # cancelled

        assert "reg-term-3" not in _connections

    def test_normal_disconnect_cleans_registry(self, client):
        """Normal client disconnect during polling cleans up registry."""
        from api.routers.ws import _connections

        _create_job(status="completed", job_id="reg-norm-1")
        with client.websocket_connect("/ws/jobs/reg-norm-1"):
            pass  # connect + immediate disconnect

        assert "reg-norm-1" not in _connections

    def test_multiple_connect_disconnect_cycles(self, client):
        """Registry stays clean across repeated connect/disconnect cycles."""
        from api.routers.ws import _connections

        _create_job(status="completed", job_id="reg-cycle-1")

        for _ in range(5):
            with client.websocket_connect("/ws/jobs/reg-cycle-1") as ws:
                ws.receive_json()  # connected
                ws.receive_json()  # completed

        assert "reg-cycle-1" not in _connections
        # Entire dict should be empty -- no residual keys from any cycle
        assert len(_connections) == 0

    def test_job_not_found_does_not_leak(self, client):
        """Connecting to a nonexistent job must not leave registry entries."""
        from api.routers.ws import _connections

        with client.websocket_connect("/ws/jobs/reg-ghost-1") as ws:
            ws.receive_json()  # error

        assert "reg-ghost-1" not in _connections


# ---------------------------------------------------------------------------
# Job not found tests
# ---------------------------------------------------------------------------


class TestWSJobNotFound:
    def test_nonexistent_job_sends_error_and_closes(self, client):
        """Connecting to a nonexistent job sends error message then closes."""
        with client.websocket_connect("/ws/jobs/nonexistent-job") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "not found" in msg["message"]

    def test_nonexistent_job_error_includes_job_id(self, client):
        """Error message for nonexistent job includes the requested job ID."""
        with client.websocket_connect("/ws/jobs/missing-xyz") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "missing-xyz" in msg["message"]


# ---------------------------------------------------------------------------
# Terminal state tests
# ---------------------------------------------------------------------------


class TestWSTerminalStates:
    def test_completed_job_sends_completed_and_closes(self, client):
        """Connecting to a completed job sends connected + completed messages."""
        _create_job(
            status="completed",
            job_id="term-test-1",
            result_path="/output/term-test-1",
        )
        with client.websocket_connect("/ws/jobs/term-test-1") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            assert connected["status"] == "completed"

            completed = ws.receive_json()
            assert completed["type"] == "completed"
            assert completed["job_id"] == "term-test-1"
            assert completed["status"] == "completed"

    def test_failed_job_sends_failed_and_closes(self, client):
        """Connecting to a failed job sends connected + failed messages."""
        _create_job(
            status="failed",
            job_id="term-test-2",
            error_message="Pipeline exited with code 1",
        )
        with client.websocket_connect("/ws/jobs/term-test-2") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            assert connected["status"] == "failed"

            failed = ws.receive_json()
            assert failed["type"] == "failed"
            assert failed["job_id"] == "term-test-2"
            assert failed["status"] == "failed"

    def test_cancelled_job_sends_cancelled_and_closes(self, client):
        """Connecting to a cancelled job sends connected + cancelled messages."""
        _create_job(status="cancelled", job_id="term-test-3")
        with client.websocket_connect("/ws/jobs/term-test-3") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            assert connected["status"] == "cancelled"

            cancelled = ws.receive_json()
            assert cancelled["type"] == "cancelled"
            assert cancelled["job_id"] == "term-test-3"
            assert cancelled["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Progress update tests
# ---------------------------------------------------------------------------


class TestWSProgressUpdates:
    def test_status_change_to_processing_sends_progress(self, client):
        """When job status changes from submitted to processing, a progress
        message is sent."""
        job_id = "prog-test-1"
        _create_job(status="submitted", job_id=job_id)

        factory = get_session_factory()

        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            assert connected["status"] == "submitted"

            # Update job status in DB while connected
            session = factory()
            job = session.query(Job).filter(Job.job_id == job_id).first()
            job.status = "processing"
            job.pages_completed = 2
            job.total_pages = 10
            session.commit()
            session.close()

            # Send ping to trigger poll cycle
            ws.send_text("ping")

            # Collect messages until we see a progress message
            messages = []
            try:
                for _ in range(5):
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg["type"] == "progress":
                        break
            except Exception:
                pass

            types = [m["type"] for m in messages]
            assert "progress" in types
            progress_msg = next(m for m in messages if m["type"] == "progress")
            assert progress_msg["status"] == "processing"

    def test_status_change_to_completed_sends_completed(self, client):
        """When a job transitions to completed, the completed message is sent
        and the connection closes."""
        job_id = "prog-test-2"
        _create_job(status="submitted", job_id=job_id)

        # After initial query, update to completed
        factory = get_session_factory()

        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            assert connected["status"] == "submitted"

            # Update job status in DB while connected
            session = factory()
            job = session.query(Job).filter(Job.job_id == job_id).first()
            job.status = "completed"
            job.result_path = "/output/completed"
            session.commit()
            session.close()

            # Send ping to trigger poll cycle
            ws.send_text("ping")

            # We should get either pong or completed (order depends on timing)
            messages = []
            try:
                for _ in range(5):
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg["type"] == "completed":
                        break
            except Exception:
                pass

            types = [m["type"] for m in messages]
            assert "completed" in types

    def test_status_change_to_failed_sends_failed(self, client):
        """When a job transitions to failed, the failed message is sent."""
        job_id = "prog-test-3"
        _create_job(status="processing", job_id=job_id)

        factory = get_session_factory()

        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            assert connected["status"] == "processing"

            # Update job status to failed
            session = factory()
            job = session.query(Job).filter(Job.job_id == job_id).first()
            job.status = "failed"
            job.error_message = "OCR engine crashed"
            session.commit()
            session.close()

            # Trigger poll cycle
            ws.send_text("ping")

            messages = []
            try:
                for _ in range(5):
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg["type"] == "failed":
                        break
            except Exception:
                pass

            types = [m["type"] for m in messages]
            assert "failed" in types
            failed_msg = next(m for m in messages if m["type"] == "failed")
            assert failed_msg["error"] == "OCR engine crashed"


# ---------------------------------------------------------------------------
# Ping/pong keepalive tests
# ---------------------------------------------------------------------------


class TestWSPingPong:
    def test_ping_receives_pong(self, client):
        """Sending 'ping' text receives a pong response."""
        _create_job(status="processing", job_id="ping-test-2")

        with client.websocket_connect("/ws/jobs/ping-test-2") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"

            ws.send_text("ping")
            pong = ws.receive_json()
            assert pong["type"] == "pong"


# ---------------------------------------------------------------------------
# Broadcast (notify_job_update) tests
# ---------------------------------------------------------------------------


class TestWSBroadcast:
    def test_notify_no_connections_does_nothing(self):
        """notify_job_update with no connected clients does not error."""
        import asyncio

        from api.routers.ws import notify_job_update

        asyncio.run(
            notify_job_update("no-such-job", {"type": "completed"})
        )
        # No error means success

    def test_notify_sends_to_connected_client(self):
        """notify_job_update broadcasts data to all registered connections."""
        import asyncio

        from api.routers.ws import _connections, notify_job_update

        # Create a mock WebSocket
        mock_ws = MagicMock()
        mock_ws.client_state = WebSocketState.CONNECTED

        sent_data = []

        async def mock_send_json(data):
            sent_data.append(data)

        mock_ws.send_json = mock_send_json

        # Register mock connection
        _connections["broadcast-test-1"] = {mock_ws}

        data = {"type": "completed", "job_id": "broadcast-test-1", "status": "completed"}
        asyncio.run(
            notify_job_update("broadcast-test-1", data)
        )

        assert len(sent_data) == 1
        assert sent_data[0]["type"] == "completed"

    def test_notify_removes_dead_connections(self):
        """notify_job_update cleans up connections that fail to send."""
        import asyncio

        from api.routers.ws import _connections, notify_job_update

        # Create a mock WebSocket that fails on send
        mock_ws = MagicMock()
        mock_ws.client_state = WebSocketState.DISCONNECTED

        _connections["broadcast-test-2"] = {mock_ws}

        data = {"type": "completed", "job_id": "broadcast-test-2"}
        asyncio.run(
            notify_job_update("broadcast-test-2", data)
        )

        # Dead connection should be cleaned up
        assert "broadcast-test-2" not in _connections

    def test_notify_to_multiple_clients(self):
        """notify_job_update sends to all registered connections for a job."""
        import asyncio

        from api.routers.ws import _connections, notify_job_update

        sent_data_1 = []
        sent_data_2 = []

        mock_ws_1 = MagicMock()
        mock_ws_1.client_state = WebSocketState.CONNECTED

        async def mock_send_1(data):
            sent_data_1.append(data)

        mock_ws_1.send_json = mock_send_1

        mock_ws_2 = MagicMock()
        mock_ws_2.client_state = WebSocketState.CONNECTED

        async def mock_send_2(data):
            sent_data_2.append(data)

        mock_ws_2.send_json = mock_send_2

        _connections["broadcast-test-3"] = {mock_ws_1, mock_ws_2}

        data = {"type": "progress", "job_id": "broadcast-test-3", "status": "processing"}
        asyncio.run(
            notify_job_update("broadcast-test-3", data)
        )

        assert len(sent_data_1) == 1
        assert len(sent_data_2) == 1
        assert sent_data_1[0]["type"] == "progress"
        assert sent_data_2[0]["type"] == "progress"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestWSHelpers:
    def test_send_json_to_connected_ws(self):
        """_send_json returns True for connected WebSocket."""
        import asyncio

        from api.routers.ws import _send_json

        mock_ws = MagicMock()
        mock_ws.client_state = WebSocketState.CONNECTED

        sent = []

        async def mock_send(data):
            sent.append(data)

        mock_ws.send_json = mock_send

        result = asyncio.run(
            _send_json(mock_ws, {"type": "test"})
        )
        assert result is True
        assert len(sent) == 1

    def test_send_json_to_disconnected_ws(self):
        """_send_json returns False for disconnected WebSocket."""
        import asyncio

        from api.routers.ws import _send_json

        mock_ws = MagicMock()
        mock_ws.client_state = WebSocketState.DISCONNECTED

        result = asyncio.run(
            _send_json(mock_ws, {"type": "test"})
        )
        assert result is False

    def test_send_json_handles_exception(self):
        """_send_json returns False when send_json raises."""
        import asyncio

        from api.routers.ws import _send_json

        mock_ws = MagicMock()
        mock_ws.client_state = WebSocketState.CONNECTED

        async def mock_send(data):
            raise RuntimeError("Connection lost")

        mock_ws.send_json = mock_send

        result = asyncio.run(
            _send_json(mock_ws, {"type": "test"})
        )
        assert result is False

    def test_authenticate_ws_no_key_configured(self):
        """When OCR_API_KEY is empty, authentication always succeeds."""
        import asyncio

        with (
            patch("api.routers.ws.OCR_API_KEY", ""),
            patch("api.oauth2_config.OAUTH2_ENABLED", False),
            patch("api.auth.ALLOW_UNAUTHENTICATED", True),
        ):
            from api.routers.ws import _authenticate_ws

            mock_ws = MagicMock()
            mock_ws.headers = {}
            result = asyncio.run(
                _authenticate_ws(mock_ws)
            )
            assert result is True

    def test_authenticate_ws_correct_header_key(self):
        """Authentication succeeds with a matching X-API-Key header."""
        import asyncio

        with patch("api.routers.ws.OCR_API_KEY", "secret-123"):
            from api.routers.ws import _authenticate_ws

            mock_ws = MagicMock()
            mock_ws.headers = {"x-api-key": "secret-123"}
            result = asyncio.run(
                _authenticate_ws(mock_ws)
            )
            assert result is True

    def test_authenticate_ws_correct_auth_message(self):
        """Authentication succeeds with a valid auth frame."""
        import asyncio

        with patch("api.routers.ws.OCR_API_KEY", "secret-123"):
            from api.routers.ws import _authenticate_ws

            mock_ws = MagicMock()
            mock_ws.headers = {}
            mock_ws.receive_text = AsyncMock(
                return_value='{"type":"auth","api_key":"secret-123"}'
            )
            result = asyncio.run(
                _authenticate_ws(mock_ws)
            )
            assert result is True

    def test_authenticate_ws_wrong_key(self):
        """Authentication fails with a non-matching auth frame key."""
        import asyncio

        with patch("api.routers.ws.OCR_API_KEY", "secret-123"):
            from api.routers.ws import _authenticate_ws

            mock_ws = MagicMock()
            mock_ws.headers = {}
            mock_ws.receive_text = AsyncMock(
                return_value='{"type":"auth","api_key":"wrong-key"}'
            )
            result = asyncio.run(
                _authenticate_ws(mock_ws)
            )
            assert result is False

    def test_max_connections_limit(self, client):
        """Exceeding MAX_WS_CONNECTIONS returns an error and closes."""
        from api.routers.ws import _connections

        _create_job(status="completed", job_id="limit-test")

        # Simulate MAX_WS_CONNECTIONS existing connections
        mock_ws_set = {MagicMock() for _ in range(100)}
        _connections["other-job"] = mock_ws_set

        with client.websocket_connect("/ws/jobs/limit-test") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Too many" in msg["message"]

        _connections.clear()

    def test_authenticate_ws_none_key_when_required(self):
        """Authentication fails when key is required but None provided."""
        import asyncio

        with patch("api.routers.ws.OCR_API_KEY", "secret-123"):
            from api.routers.ws import _authenticate_ws

            mock_ws = MagicMock()
            mock_ws.headers = {}
            mock_ws.receive_text = AsyncMock(
                return_value='{"type":"auth","api_key":""}'
            )
            result = asyncio.run(
                _authenticate_ws(mock_ws)
            )
            assert result is False


# ---------------------------------------------------------------------------
# OAuth2 token via auth frame (not query string)
# ---------------------------------------------------------------------------


class TestWSAuthFrameOAuth2:
    """Tests for OAuth2 bearer token authentication via auth frame.

    Validates that:
    - token query parameter is removed from endpoint signature
    - OAuth2 tokens are accepted via auth frame
    - auth frame API key still works alongside OAuth2
    """

    def test_endpoint_has_no_token_query_param(self):
        """job_progress_ws must not accept a token query parameter."""
        import inspect

        from api.routers.ws import job_progress_ws

        sig = inspect.signature(job_progress_ws)
        assert "token" not in sig.parameters, \
            "token query parameter must be removed to prevent credential leakage"

    def test_oauth2_token_via_auth_frame_succeeds(self):
        """Auth frame with token validates JWT when OAuth2 is enabled."""
        import asyncio
        import json

        mock_ws = MagicMock()
        mock_ws.state = MagicMock()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "auth", "token": "valid-jwt"}),
        )

        claims = {"sub": "oauth-user", "roles": ["operator"], "email": "u@x.com"}

        with (
            patch("api.routers.ws.OCR_API_KEY", ""),
            patch("api.oauth2_config.OAUTH2_ENABLED", True),
            patch("api.oauth2.validate_jwt", return_value=claims),
            patch("api.oauth2.extract_role", return_value="operator"),
        ):
            from api.routers.ws import _authenticate_ws

            result = asyncio.run(_authenticate_ws(mock_ws))

        assert result is True
        identity = mock_ws.state.identity
        assert identity.subject == "oauth-user"
        assert identity.role == "operator"
        assert identity.auth_method == "oauth2"
        assert identity.email == "u@x.com"

    def test_oauth2_token_auth_frame_invalid_jwt_rejected(self):
        """Auth frame with invalid JWT returns False."""
        import asyncio
        import json

        mock_ws = MagicMock()
        mock_ws.state = MagicMock()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "auth", "token": "bad-jwt"}),
        )

        with (
            patch("api.routers.ws.OCR_API_KEY", ""),
            patch("api.oauth2_config.OAUTH2_ENABLED", True),
            patch("api.oauth2.validate_jwt", side_effect=ValueError("expired")),
        ):
            from api.routers.ws import _authenticate_ws

            result = asyncio.run(_authenticate_ws(mock_ws))

        assert result is False

    def test_oauth2_token_auth_frame_ignored_when_disabled(self):
        """Auth frame token is not processed when OAUTH2_ENABLED is False."""
        import asyncio
        import json

        mock_ws = MagicMock()
        mock_ws.state = MagicMock()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "auth", "token": "some-token"}),
        )

        with (
            patch("api.routers.ws.OCR_API_KEY", ""),
            patch("api.oauth2_config.OAUTH2_ENABLED", False),
            patch("api.auth.ALLOW_UNAUTHENTICATED", False),
        ):
            from api.routers.ws import _authenticate_ws

            result = asyncio.run(_authenticate_ws(mock_ws))

        assert result is False

    def test_api_key_auth_frame_still_works(self):
        """API key via auth frame is unaffected by  changes."""
        import asyncio

        mock_ws = MagicMock()
        mock_ws.state = MagicMock()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value='{"type":"auth","api_key":"my-secret"}',
        )

        with patch("api.routers.ws.OCR_API_KEY", "my-secret"):
            from api.routers.ws import _authenticate_ws

            result = asyncio.run(_authenticate_ws(mock_ws))

        assert result is True

    def test_auth_frame_wrong_api_key_returns_false(self):
        """Auth frame with wrong API key returns False without waiting."""
        import asyncio

        mock_ws = MagicMock()
        mock_ws.state = MagicMock()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value='{"type":"auth","api_key":"wrong"}',
        )

        with patch("api.routers.ws.OCR_API_KEY", "correct-key"):
            from api.routers.ws import _authenticate_ws

            result = asyncio.run(_authenticate_ws(mock_ws))

        assert result is False

    def test_auth_frame_with_both_apikey_and_token_prefers_apikey(self):
        """When auth frame has both api_key and token, api_key is tried first."""
        import asyncio
        import json

        mock_ws = MagicMock()
        mock_ws.state = MagicMock()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value=json.dumps({
                "type": "auth",
                "api_key": "my-key",
                "token": "jwt-token",
            }),
        )

        with patch("api.routers.ws.OCR_API_KEY", "my-key"):
            from api.routers.ws import _authenticate_ws

            result = asyncio.run(_authenticate_ws(mock_ws))

        assert result is True
        assert mock_ws.state.identity.auth_method == "apikey"


# ---------------------------------------------------------------------------
# WebSocket idle timeout + max-duration limits
# ---------------------------------------------------------------------------


class TestWSTimeoutConfig:
    """Verify  timeout constants are env-configurable."""

    def test_default_constants_present(self):
        from api.routers import ws as ws_module

        assert hasattr(ws_module, "WS_IDLE_TIMEOUT_SECONDS")
        assert hasattr(ws_module, "WS_MAX_DURATION_SECONDS")
        # Defaults should be positive integers.
        assert isinstance(ws_module.WS_IDLE_TIMEOUT_SECONDS, int)
        assert isinstance(ws_module.WS_MAX_DURATION_SECONDS, int)
        assert ws_module.WS_IDLE_TIMEOUT_SECONDS > 0
        assert ws_module.WS_MAX_DURATION_SECONDS > 0

    def test_idle_timeout_env_override(self, monkeypatch):
        """WS_IDLE_TIMEOUT_SECONDS env var is honored at import time."""
        import importlib

        monkeypatch.setenv("WS_IDLE_TIMEOUT_SECONDS", "42")
        from api.routers import ws as ws_module

        importlib.reload(ws_module)
        assert ws_module.WS_IDLE_TIMEOUT_SECONDS == 42
        # Restore defaults for subsequent tests.
        monkeypatch.delenv("WS_IDLE_TIMEOUT_SECONDS", raising=False)
        importlib.reload(ws_module)

    def test_max_duration_env_override(self, monkeypatch):
        """WS_MAX_DURATION_SECONDS env var is honored at import time."""
        import importlib

        monkeypatch.setenv("WS_MAX_DURATION_SECONDS", "123")
        from api.routers import ws as ws_module

        importlib.reload(ws_module)
        assert ws_module.WS_MAX_DURATION_SECONDS == 123
        monkeypatch.delenv("WS_MAX_DURATION_SECONDS", raising=False)
        importlib.reload(ws_module)


class _FakeWebSocket:
    """Minimal WebSocket test double used for direct handler unit tests.

    The real ``fastapi.testclient.TestClient`` makes it awkward to assert
    on close-frame ordering because the client side buffers frames across
    ``close()`` calls.  For  we only need to verify the handler's
    *server-side* decisions, so a simple stub is sufficient.
    """

    def __init__(self, initial_status: str = "processing"):
        self.client_state = WebSocketState.CONNECTED
        self.state = MagicMock()
        self.state.tenant_id = None
        self.state.identity = None
        self.headers = {}
        self.sent: list = []
        self.close_calls: list = []
        self.accept_called = False
        # Queue of frames the fake client "sends" to the handler.  Each
        # item is either a string (delivered by ``receive_text``) or an
        # exception instance (raised by ``receive_text``).
        self._incoming: list = []

    def enqueue_incoming(self, *frames):
        self._incoming.extend(frames)

    async def accept(self):
        self.accept_called = True

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_text(self):
        # When the queue is empty, simulate a slow/idle client by raising
        # asyncio.TimeoutError the same way ``asyncio.wait_for`` does in
        # the real handler path.  This keeps the handler's loop cycling
        # without bringing back a frame.
        import asyncio as _asyncio

        if not self._incoming:
            raise _asyncio.TimeoutError()
        frame = self._incoming.pop(0)
        if isinstance(frame, BaseException):
            raise frame
        return frame

    async def close(self, code=1000, reason=""):
        self.close_calls.append((code, reason))
        self.client_state = WebSocketState.DISCONNECTED


class TestWSIdleTimeout:
    """Direct handler tests for  timeout behavior.

    These tests invoke ``job_progress_ws`` with a fake WebSocket to avoid
    TestClient buffering quirks when asserting on close-frame ordering.
    """

    @staticmethod
    def _run_handler(job_id: str, ws: _FakeWebSocket) -> None:
        import asyncio

        from api.routers.ws import job_progress_ws

        asyncio.run(job_progress_ws(ws, job_id))

    def test_idle_timeout_fires_for_inactive_connection(self, client):
        """When no client frames arrive and the idle window has elapsed,
        the handler must emit the idle error and close the connection.

        We monkey-patch ``time.monotonic`` so the loop sees an arbitrarily
        large wall-clock jump without actually sleeping.
        """
        _create_job(status="processing", job_id="idle-direct-1")
        ws = _FakeWebSocket()

        t0 = [1000.0]
        call_count = [0]

        def fake_monotonic():
            # First call (connect_start/last_activity baseline) returns t0.
            # Subsequent calls jump far into the future so the idle window
            # is guaranteed exceeded.
            call_count[0] += 1
            if call_count[0] <= 2:
                return t0[0]
            return t0[0] + 10_000.0

        with (
            patch("api.routers.ws.WS_IDLE_TIMEOUT_SECONDS", 60),
            patch("api.routers.ws.WS_MAX_DURATION_SECONDS", 36_000),
            patch("api.routers.ws._now_monotonic", side_effect=fake_monotonic),
        ):
            self._run_handler("idle-direct-1", ws)

        types = [m.get("type") for m in ws.sent]
        assert "connected" in types
        assert any(
            m.get("type") == "error" and "idle" in m.get("message", "").lower()
            for m in ws.sent
        ), f"no idle-timeout error in {ws.sent!r}"
        assert ws.close_calls, "websocket.close() was not called"
        assert ws.close_calls[0][0] == 4008

    def test_max_duration_fires_for_long_connection(self, client):
        """When the wall-clock duration exceeds the cap, the handler must
        emit the max-duration error and close the connection."""
        _create_job(status="processing", job_id="duration-direct-1")
        ws = _FakeWebSocket()

        call_count = [0]

        def fake_monotonic():
            call_count[0] += 1
            # Baseline calls get t=1000, then jump 10 hours.
            if call_count[0] <= 2:
                return 1000.0
            return 1000.0 + 36_000.0

        with (
            patch("api.routers.ws.WS_IDLE_TIMEOUT_SECONDS", 3600),
            patch("api.routers.ws.WS_MAX_DURATION_SECONDS", 60),
            patch("api.routers.ws._now_monotonic", side_effect=fake_monotonic),
        ):
            self._run_handler("duration-direct-1", ws)

        assert any(
            m.get("type") == "error"
            and "max connection duration" in m.get("message", "").lower()
            for m in ws.sent
        ), f"no max-duration error in {ws.sent!r}"
        assert ws.close_calls
        assert ws.close_calls[0][0] == 4008

    def test_disabled_timeouts_do_not_fire(self, client):
        """Setting either limit to 0 must disable that limit entirely.

        We jump the clock forward by a huge amount; if the check was not
        properly gated the handler would still emit the error.  Instead
        we rely on the fake client raising ``WebSocketDisconnect`` to end
        the loop cleanly.
        """
        _create_job(status="processing", job_id="disabled-direct-1")
        ws = _FakeWebSocket()
        from fastapi import WebSocketDisconnect
        ws.enqueue_incoming(WebSocketDisconnect())

        call_count = [0]

        def fake_monotonic():
            call_count[0] += 1
            if call_count[0] <= 2:
                return 1000.0
            return 1_000_000.0  # 11.5 days later

        with (
            patch("api.routers.ws.WS_IDLE_TIMEOUT_SECONDS", 0),
            patch("api.routers.ws.WS_MAX_DURATION_SECONDS", 0),
            patch("api.routers.ws._now_monotonic", side_effect=fake_monotonic),
        ):
            self._run_handler("disabled-direct-1", ws)

        assert not any(
            m.get("type") == "error"
            and (
                "idle" in m.get("message", "").lower()
                or "max connection" in m.get("message", "").lower()
            )
            for m in ws.sent
        ), f"timeout error fired despite being disabled: {ws.sent!r}"

    def test_terminal_status_skips_timeout_path(self, client):
        """Terminal jobs short-circuit before the poll loop, so tight
        timeouts have no effect on the completed-message path."""
        _create_job(status="completed", job_id="disable-direct-1")
        ws = _FakeWebSocket()

        with (
            patch("api.routers.ws.WS_IDLE_TIMEOUT_SECONDS", 1),
            patch("api.routers.ws.WS_MAX_DURATION_SECONDS", 1),
        ):
            self._run_handler("disable-direct-1", ws)

        types = [m.get("type") for m in ws.sent]
        assert types[:2] == ["connected", "completed"]
        # No timeout error should be emitted for terminal jobs.
        assert not any(
            m.get("type") == "error"
            and (
                "idle" in m.get("message", "").lower()
                or "max connection" in m.get("message", "").lower()
            )
            for m in ws.sent
        )

    def test_ping_resets_idle_activity(self, client):
        """A client frame received before the idle window expires resets
        the last_activity marker, so the loop can reach the next poll."""
        _create_job(status="processing", job_id="ping-direct-1")
        ws = _FakeWebSocket()
        # Deliver a ping frame, then a WebSocketDisconnect to end the loop.
        from fastapi import WebSocketDisconnect
        ws.enqueue_incoming("ping", WebSocketDisconnect())

        with (
            patch("api.routers.ws.WS_IDLE_TIMEOUT_SECONDS", 3600),
            patch("api.routers.ws.WS_MAX_DURATION_SECONDS", 3600),
        ):
            self._run_handler("ping-direct-1", ws)

        # A pong should have been sent in response to the ping.
        assert any(m.get("type") == "pong" for m in ws.sent)
        # And no timeout error should have been emitted.
        assert not any(
            m.get("type") == "error"
            and (
                "idle" in m.get("message", "").lower()
                or "max connection" in m.get("message", "").lower()
            )
            for m in ws.sent
        )
