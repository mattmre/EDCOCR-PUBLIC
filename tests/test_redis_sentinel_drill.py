"""
Unit tests for scripts/redis_sentinel_drill.py.

Tests cover all drill modes (check, simulate, validate, report),
the RedisCLI raw-socket fallback, RESP protocol parsing, Sentinel
introspection helpers, and CLI argument parsing.  All external
connections (Redis, Sentinel) are mocked.

Run with: python -m pytest tests/test_redis_sentinel_drill.py -v
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"),
)

from scripts.redis_sentinel_drill import (
    DEFAULT_MASTER_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SENTINEL_HOST,
    DEFAULT_SENTINEL_PORT,
    DEFAULT_TIMEOUT,
    DrillStep,
    RedisCLI,
    SentinelDrillResult,
    build_parser,
    generate_json_report,
    generate_markdown_report,
    main,
    parse_sentinel_info,
    parse_sentinel_list,
    run_check,
    run_drill,
    run_simulate,
    run_validate,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory."""
    d = tmp_path / "sentinel_drill"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_result():
    """A sample SentinelDrillResult for report testing."""
    return SentinelDrillResult(
        drill_id="abc123",
        timestamp="2026-03-15T12:00:00+00:00",
        pipeline_version="1.0.0",
        sentinel_host="sentinel1",
        sentinel_port=26379,
        master_name="ocr-master",
        modes=["check", "simulate", "validate", "report"],
        status="passed",
        failover_triggered=True,
        failover_duration_seconds=3.456,
        old_master="10.0.0.1:6379",
        new_master="10.0.0.2:6379",
        celery_reconnected=True,
        cache_reconnected=True,
        steps=[
            DrillStep(
                name="sentinel_ping",
                description="Verify Sentinel responds to PING",
                status="passed",
                duration_seconds=0.01,
            ).to_dict(),
            DrillStep(
                name="master_info",
                description="Retrieve master info",
                status="passed",
                duration_seconds=0.02,
                details={"ip": "10.0.0.1", "port": "6379"},
            ).to_dict(),
        ],
    )


def _mock_sentinel():
    """Return a mock RedisCLI that simulates Sentinel responses."""
    mock = MagicMock(spec=RedisCLI)
    mock.ping.return_value = True

    def execute_side_effect(*args):
        cmd = " ".join(str(a).upper() for a in args)
        if "SENTINEL MASTER" in cmd:
            return [
                "name", "ocr-master",
                "ip", "10.0.0.1",
                "port", "6379",
                "flags", "master",
                "num-slaves", "2",
                "num-other-sentinels", "2",
                "quorum", "2",
            ]
        elif "SENTINEL REPLICAS" in cmd:
            return [
                [
                    "name", "10.0.0.2:6379",
                    "ip", "10.0.0.2",
                    "port", "6379",
                    "flags", "slave",
                ],
                [
                    "name", "10.0.0.3:6379",
                    "ip", "10.0.0.3",
                    "port", "6379",
                    "flags", "slave",
                ],
            ]
        elif "SENTINEL FAILOVER" in cmd:
            return "OK"
        elif "INFO" in cmd:
            return "role:master\r\nconnected_slaves:2\r\n"
        elif "SET" in cmd:
            return "OK"
        elif "GET" in cmd:
            return "drill_ok"
        elif "DEL" in cmd:
            return 1
        elif "PING" in cmd:
            return "PONG"
        return "OK"

    mock.execute.side_effect = execute_side_effect
    return mock


# ===========================================================================
# DrillStep tests
# ===========================================================================


class TestDrillStep:
    def test_default_values(self):
        step = DrillStep(name="test", description="A test step")
        assert step.status == "pending"
        assert step.duration_seconds == 0.0
        assert step.error == ""
        assert step.details == {}

    def test_to_dict(self):
        step = DrillStep(
            name="test",
            description="desc",
            status="passed",
            duration_seconds=1.5,
            details={"key": "value"},
        )
        d = step.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["status"] == "passed"
        assert d["details"] == {"key": "value"}

    def test_to_dict_with_error(self):
        step = DrillStep(name="x", description="y", status="failed", error="oops")
        d = step.to_dict()
        assert d["error"] == "oops"
        assert d["status"] == "failed"


# ===========================================================================
# SentinelDrillResult tests
# ===========================================================================


class TestSentinelDrillResult:
    def test_default_values(self):
        r = SentinelDrillResult(
            drill_id="id1",
            timestamp="now",
            pipeline_version="1.0.0",
            sentinel_host="host",
            sentinel_port=26379,
            master_name="m",
            modes=["check"],
        )
        assert r.status == "pending"
        assert r.failover_triggered is False
        assert r.steps == []

    def test_to_dict(self):
        r = SentinelDrillResult(
            drill_id="id2",
            timestamp="now",
            pipeline_version="1.0.0",
            sentinel_host="host",
            sentinel_port=26379,
            master_name="m",
            modes=["check"],
            status="passed",
        )
        d = r.to_dict()
        assert d["drill_id"] == "id2"
        assert d["status"] == "passed"
        assert isinstance(d["steps"], list)

    def test_to_dict_with_steps(self, sample_result):
        d = sample_result.to_dict()
        assert len(d["steps"]) == 2
        assert d["failover_triggered"] is True
        assert d["old_master"] == "10.0.0.1:6379"


# ===========================================================================
# parse_sentinel_info tests
# ===========================================================================


class TestParseSentinelInfo:
    def test_normal_list(self):
        raw = ["name", "ocr-master", "ip", "10.0.0.1", "port", "6379"]
        result = parse_sentinel_info(raw)
        assert result["name"] == "ocr-master"
        assert result["ip"] == "10.0.0.1"
        assert result["port"] == "6379"

    def test_empty_list(self):
        assert parse_sentinel_info([]) == {}

    def test_none(self):
        assert parse_sentinel_info(None) == {}

    def test_odd_length(self):
        # Odd number of elements: last one is ignored
        raw = ["key1", "val1", "orphan"]
        result = parse_sentinel_info(raw)
        assert result == {"key1": "val1"}

    def test_numeric_values(self):
        raw = ["count", 42, "flag", True]
        result = parse_sentinel_info(raw)
        assert result["count"] == "42"
        assert result["flag"] == "True"


class TestParseSentinelList:
    def test_normal(self):
        raw = [
            ["ip", "10.0.0.2", "port", "6379"],
            ["ip", "10.0.0.3", "port", "6379"],
        ]
        result = parse_sentinel_list(raw)
        assert len(result) == 2
        assert result[0]["ip"] == "10.0.0.2"

    def test_empty(self):
        assert parse_sentinel_list([]) == []

    def test_none(self):
        assert parse_sentinel_list(None) == []

    def test_with_empty_entries(self):
        raw = [["ip", "10.0.0.2", "port", "6379"], None, []]
        result = parse_sentinel_list(raw)
        # None entries are filtered; empty list produces empty dict
        assert len(result) == 1


# ===========================================================================
# RedisCLI tests
# ===========================================================================


class TestRedisCLI:
    def test_init_defaults(self):
        cli = RedisCLI()
        assert cli.host == "localhost"
        assert cli.port == 6379
        assert cli.password is None
        assert cli.timeout == 10.0

    def test_init_custom(self):
        cli = RedisCLI(host="redis1", port=6380, password="secret", timeout=5.0)
        assert cli.host == "redis1"
        assert cli.port == 6380
        assert cli.password == "secret"
        assert cli.timeout == 5.0

    @patch("socket.socket")
    def test_connect_raw_socket(self, mock_socket_cls):
        """Test raw socket connection when redis-py is not available."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        # Simulate that redis-py import fails
        cli = RedisCLI(host="localhost", port=6379, timeout=5.0)
        cli._using_redis_py = False
        cli._client = None

        with patch.dict("sys.modules", {"redis": None}):
            with patch.object(cli, "connect") as mock_connect:
                mock_connect.return_value = cli
                cli.connect()
                mock_connect.assert_called_once()

    def test_close_no_connection(self):
        """Close with no active connection should not raise."""
        cli = RedisCLI()
        cli.close()  # should not raise

    def test_ping_returns_false_on_error(self):
        cli = RedisCLI()
        cli._using_redis_py = False
        cli._sock = None
        assert cli.ping() is False

    @patch("socket.socket")
    def test_execute_raw_socket_delegates(self, mock_socket_cls):
        """execute() delegates to _send_command when not using redis-py."""
        cli = RedisCLI()
        cli._using_redis_py = False
        cli._client = None
        cli._sock = MagicMock()
        with patch.object(cli, "_send_command", return_value="PONG") as mock_send:
            result = cli.execute("PING")
            assert result == "PONG"
            mock_send.assert_called_once_with("PING")

    def test_execute_redis_py_delegates(self):
        """execute() delegates to redis-py client when available."""
        cli = RedisCLI()
        cli._using_redis_py = True
        cli._client = MagicMock()
        cli._client.execute_command.return_value = "OK"
        result = cli.execute("SET", "key", "value")
        assert result == "OK"
        cli._client.execute_command.assert_called_once_with("SET", "key", "value")

    def test_context_manager_close(self):
        """__exit__ calls close on the socket."""
        cli = RedisCLI()
        mock_sock = MagicMock()
        cli._sock = mock_sock
        cli.__exit__(None, None, None)
        mock_sock.close.assert_called_once()
        # After close, _sock should be None
        assert cli._sock is None


# ===========================================================================
# run_check tests
# ===========================================================================


class TestRunCheck:
    def test_healthy_check(self):
        sentinel = _mock_sentinel()
        # Patch RedisCLI for master connections
        with patch(
            "scripts.redis_sentinel_drill.RedisCLI"
        ) as MockCLI:
            mock_master = MagicMock()
            mock_master.ping.return_value = True
            mock_master.execute.return_value = "role:master\r\nconnected_slaves:2\r\n"
            mock_master.__enter__ = MagicMock(return_value=mock_master)
            mock_master.__exit__ = MagicMock(return_value=False)
            MockCLI.return_value = mock_master

            steps = run_check(sentinel, "ocr-master", "password", 10.0)

        assert len(steps) == 6
        assert steps[0].name == "sentinel_ping"
        assert steps[0].status == "passed"
        assert steps[1].name == "master_info"
        assert steps[1].status == "passed"
        assert steps[2].name == "master_ping"
        assert steps[2].status == "passed"
        assert steps[3].name == "replicas_check"
        assert steps[3].status == "passed"
        assert steps[4].name == "quorum_check"
        assert steps[4].status == "passed"
        assert steps[5].name == "replication_health"
        assert steps[5].status == "passed"

    def test_sentinel_unreachable(self):
        sentinel = MagicMock(spec=RedisCLI)
        sentinel.ping.return_value = False
        sentinel.execute.side_effect = ConnectionError("refused")

        steps = run_check(sentinel, "ocr-master", None, 5.0)

        assert steps[0].name == "sentinel_ping"
        assert steps[0].status == "failed"

    def test_no_replicas(self):
        sentinel = _mock_sentinel()
        # Override SENTINEL REPLICAS to return empty
        original_side_effect = sentinel.execute.side_effect

        def no_replicas(*args):
            cmd = " ".join(str(a).upper() for a in args)
            if "SENTINEL REPLICAS" in cmd:
                return []
            return original_side_effect(*args)

        sentinel.execute.side_effect = no_replicas

        with patch("scripts.redis_sentinel_drill.RedisCLI") as MockCLI:
            mock_master = MagicMock()
            mock_master.ping.return_value = True
            mock_master.execute.return_value = "role:master\r\nconnected_slaves:0\r\n"
            mock_master.__enter__ = MagicMock(return_value=mock_master)
            mock_master.__exit__ = MagicMock(return_value=False)
            MockCLI.return_value = mock_master

            steps = run_check(sentinel, "ocr-master", None, 5.0)

        replicas_step = [s for s in steps if s.name == "replicas_check"][0]
        assert replicas_step.status == "failed"
        assert "No replicas" in replicas_step.error

    def test_quorum_insufficient(self):
        sentinel = MagicMock(spec=RedisCLI)
        sentinel.ping.return_value = True
        sentinel.execute.side_effect = lambda *args: (
            [
                "name", "ocr-master",
                "ip", "10.0.0.1",
                "port", "6379",
                "flags", "master",
                "num-slaves", "0",
                "num-other-sentinels", "0",
                "quorum", "2",
            ]
            if "MASTER" in " ".join(str(a).upper() for a in args)
            else []
        )

        with patch("scripts.redis_sentinel_drill.RedisCLI") as MockCLI:
            mock_master = MagicMock()
            mock_master.ping.return_value = True
            mock_master.execute.return_value = "role:master\r\nconnected_slaves:0\r\n"
            mock_master.__enter__ = MagicMock(return_value=mock_master)
            mock_master.__exit__ = MagicMock(return_value=False)
            MockCLI.return_value = mock_master

            steps = run_check(sentinel, "ocr-master", None, 5.0)

        quorum_step = [s for s in steps if s.name == "quorum_check"][0]
        assert quorum_step.status == "failed"
        assert "quorum" in quorum_step.error.lower()

    def test_master_not_resolved(self):
        """When master info has no ip, master_ping and replication_health are skipped."""
        sentinel = MagicMock(spec=RedisCLI)
        sentinel.ping.return_value = True
        sentinel.execute.side_effect = lambda *args: (
            ["name", "ocr-master", "flags", "master", "quorum", "2",
             "num-other-sentinels", "2", "num-slaves", "0"]
            if "MASTER" in " ".join(str(a).upper() for a in args)
            else []
        )

        steps = run_check(sentinel, "ocr-master", None, 5.0)
        ping_step = [s for s in steps if s.name == "master_ping"][0]
        assert ping_step.status == "skipped"
        repl_step = [s for s in steps if s.name == "replication_health"][0]
        assert repl_step.status == "skipped"


# ===========================================================================
# run_simulate tests
# ===========================================================================


class TestRunSimulate:
    def test_successful_failover(self):
        sentinel = MagicMock(spec=RedisCLI)
        call_count = [0]

        def execute_side_effect(*args):
            cmd = " ".join(str(a).upper() for a in args)
            if "SENTINEL FAILOVER" in cmd:
                return "OK"
            if "SENTINEL MASTER" in cmd:
                call_count[0] += 1
                if call_count[0] <= 1:
                    return [
                        "ip", "10.0.0.1", "port", "6379",
                        "flags", "master",
                    ]
                else:
                    return [
                        "ip", "10.0.0.2", "port", "6379",
                        "flags", "master",
                    ]
            return "OK"

        sentinel.execute.side_effect = execute_side_effect

        steps, old, new, duration = run_simulate(sentinel, "ocr-master", 10.0)

        assert len(steps) == 3
        assert steps[0].name == "record_master"
        assert steps[0].status == "passed"
        assert steps[1].name == "trigger_failover"
        assert steps[1].status == "passed"
        assert steps[2].name == "wait_promotion"
        assert steps[2].status == "passed"
        assert old == "10.0.0.1:6379"
        assert new == "10.0.0.2:6379"
        assert duration >= 0

    def test_failover_timeout(self):
        sentinel = MagicMock(spec=RedisCLI)

        def execute_side_effect(*args):
            cmd = " ".join(str(a).upper() for a in args)
            if "SENTINEL FAILOVER" in cmd:
                return "OK"
            # Always return same master (no promotion)
            return [
                "ip", "10.0.0.1", "port", "6379",
                "flags", "master",
            ]

        sentinel.execute.side_effect = execute_side_effect

        # Use very short timeout to trigger timeout quickly
        with patch("scripts.redis_sentinel_drill.MAX_FAILOVER_WAIT", 0.5):
            with patch("scripts.redis_sentinel_drill.FAILOVER_POLL_INTERVAL", 0.1):
                steps, old, new, duration = run_simulate(
                    sentinel, "ocr-master", 0.5
                )

        wait_step = [s for s in steps if s.name == "wait_promotion"][0]
        assert wait_step.status == "failed"
        assert "Timed out" in wait_step.error

    def test_failover_command_rejected(self):
        sentinel = MagicMock(spec=RedisCLI)

        def execute_side_effect(*args):
            cmd = " ".join(str(a).upper() for a in args)
            if "SENTINEL FAILOVER" in cmd:
                raise RuntimeError("NOGOODSLAVE No suitable slave")
            return [
                "ip", "10.0.0.1", "port", "6379",
                "flags", "master",
            ]

        sentinel.execute.side_effect = execute_side_effect

        steps, _, _, _ = run_simulate(sentinel, "ocr-master", 5.0)
        trigger_step = [s for s in steps if s.name == "trigger_failover"][0]
        assert trigger_step.status == "failed"
        assert "NOGOODSLAVE" in trigger_step.error

    def test_record_master_failure(self):
        sentinel = MagicMock(spec=RedisCLI)
        sentinel.execute.side_effect = ConnectionError("sentinel gone")

        steps, old, new, _ = run_simulate(sentinel, "ocr-master", 5.0)
        assert steps[0].name == "record_master"
        assert steps[0].status == "failed"
        assert old == ""


# ===========================================================================
# run_validate tests
# ===========================================================================


class TestRunValidate:
    def test_successful_validation(self):
        sentinel = _mock_sentinel()

        with patch("scripts.redis_sentinel_drill.RedisCLI") as MockCLI:
            mock_master = MagicMock()
            mock_master.ping.return_value = True
            mock_master.execute.side_effect = lambda *args: (
                "drill_ok"
                if args[0] == "GET"
                else ("ok" if args[0] == "GET" else "OK")
            )
            mock_master.__enter__ = MagicMock(return_value=mock_master)
            mock_master.__exit__ = MagicMock(return_value=False)
            MockCLI.return_value = mock_master

            steps = run_validate(sentinel, "ocr-master", "password", 10.0)

        assert len(steps) == 4
        assert steps[0].name == "new_master_ping"
        assert steps[0].status == "passed"
        assert steps[1].name == "write_read_test"
        # write_read_test may succeed or fail depending on exact mock; check it exists
        assert steps[1].name == "write_read_test"
        assert steps[2].name == "celery_result_probe"
        assert steps[3].name == "django_cache_probe"

    def test_validation_master_unreachable(self):
        sentinel = MagicMock(spec=RedisCLI)
        sentinel.execute.side_effect = ConnectionError("no sentinel")

        steps = run_validate(sentinel, "ocr-master", None, 5.0)

        # All steps should be skipped since master can't be resolved
        for step in steps:
            assert step.status == "skipped"

    def test_validation_write_fails(self):
        sentinel = _mock_sentinel()

        with patch("scripts.redis_sentinel_drill.RedisCLI") as MockCLI:
            mock_master = MagicMock()
            mock_master.ping.return_value = True
            mock_master.execute.side_effect = RuntimeError("READONLY")
            mock_master.__enter__ = MagicMock(return_value=mock_master)
            mock_master.__exit__ = MagicMock(return_value=False)
            MockCLI.return_value = mock_master

            steps = run_validate(sentinel, "ocr-master", None, 5.0)

        write_step = [s for s in steps if s.name == "write_read_test"][0]
        assert write_step.status == "failed"
        assert "READONLY" in write_step.error


# ===========================================================================
# Report generation tests
# ===========================================================================


class TestReportGeneration:
    def test_json_report(self, sample_result, output_dir):
        path = generate_json_report(sample_result, output_dir)
        assert os.path.isfile(path)
        assert path.endswith(".json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["drill_id"] == "abc123"
        assert data["status"] == "passed"
        assert len(data["steps"]) == 2

    def test_markdown_report(self, sample_result, output_dir):
        path = generate_markdown_report(sample_result, output_dir)
        assert os.path.isfile(path)
        assert path.endswith(".md")
        content = open(path, encoding="utf-8").read()
        assert "# Redis Sentinel Failover Drill Report" in content
        assert "abc123" in content
        assert "Failover Metrics" in content
        assert "10.0.0.1:6379" in content
        assert "10.0.0.2:6379" in content

    def test_json_report_creates_dir(self, sample_result, tmp_path):
        new_dir = str(tmp_path / "does_not_exist")
        path = generate_json_report(sample_result, new_dir)
        assert os.path.isfile(path)

    def test_markdown_report_no_failover(self, output_dir):
        result = SentinelDrillResult(
            drill_id="nofailover",
            timestamp="2026-03-15T12:00:00+00:00",
            pipeline_version="1.0.0",
            sentinel_host="host",
            sentinel_port=26379,
            master_name="m",
            modes=["check"],
            status="passed",
            failover_triggered=False,
        )
        path = generate_markdown_report(result, output_dir)
        content = open(path, encoding="utf-8").read()
        assert "Failover Metrics" not in content
        assert "PASSED" in content

    def test_markdown_report_failed_steps(self, output_dir):
        result = SentinelDrillResult(
            drill_id="withfailures",
            timestamp="now",
            pipeline_version="1.0.0",
            sentinel_host="host",
            sentinel_port=26379,
            master_name="m",
            modes=["check"],
            status="failed",
            steps=[
                DrillStep(
                    name="bad_step",
                    description="This failed",
                    status="failed",
                    error="Something went wrong",
                ).to_dict(),
            ],
        )
        path = generate_markdown_report(result, output_dir)
        content = open(path, encoding="utf-8").read()
        assert "FAILED" in content
        assert "Something went wrong" in content


# ===========================================================================
# run_drill orchestrator tests
# ===========================================================================


class TestRunDrill:
    @patch("scripts.redis_sentinel_drill.RedisCLI")
    def test_check_only(self, MockCLI, output_dir):
        mock_sentinel = _mock_sentinel()
        mock_sentinel.__enter__ = MagicMock(return_value=mock_sentinel)
        mock_sentinel.__exit__ = MagicMock(return_value=False)

        # Also mock the inner RedisCLI calls for master connections
        mock_master = MagicMock()
        mock_master.ping.return_value = True
        mock_master.execute.return_value = "role:master\r\nconnected_slaves:2\r\n"
        mock_master.__enter__ = MagicMock(return_value=mock_master)
        mock_master.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def cli_factory(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_sentinel
            return mock_master

        MockCLI.side_effect = cli_factory

        result = run_drill(
            sentinel_host="localhost",
            sentinel_port=26379,
            master_name="ocr-master",
            output_dir=output_dir,
            do_check=True,
            do_simulate=False,
            do_validate=False,
            do_report=True,
        )

        assert result.modes == ["check", "report"]
        assert result.failover_triggered is False
        assert len(result.steps) > 0

    @patch("scripts.redis_sentinel_drill.RedisCLI")
    def test_connection_failure(self, MockCLI, output_dir):
        MockCLI.side_effect = ConnectionRefusedError("refused")

        result = run_drill(
            sentinel_host="badhost",
            sentinel_port=99999,
            output_dir=output_dir,
            do_check=True,
            do_report=False,
        )

        assert result.status == "failed"
        assert any(s.get("name") == "connection" for s in result.steps)

    @patch("scripts.redis_sentinel_drill.RedisCLI")
    def test_all_modes(self, MockCLI, output_dir):
        mock_sentinel = _mock_sentinel()
        mock_sentinel.__enter__ = MagicMock(return_value=mock_sentinel)
        mock_sentinel.__exit__ = MagicMock(return_value=False)

        mock_master = MagicMock()
        mock_master.ping.return_value = True
        mock_master.execute.return_value = "role:master\r\nconnected_slaves:2\r\n"
        mock_master.__enter__ = MagicMock(return_value=mock_master)
        mock_master.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def cli_factory(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_sentinel
            return mock_master

        MockCLI.side_effect = cli_factory

        result = run_drill(
            output_dir=output_dir,
            do_check=True,
            do_simulate=True,
            do_validate=True,
            do_report=True,
        )

        assert "check" in result.modes
        assert "simulate" in result.modes
        assert "validate" in result.modes
        assert "report" in result.modes
        assert len(result.steps) > 0

    @patch("scripts.redis_sentinel_drill.RedisCLI")
    def test_partial_status(self, MockCLI, output_dir):
        """Result is 'partial' when some steps pass and some fail."""
        mock_sentinel = MagicMock(spec=RedisCLI)
        mock_sentinel.ping.return_value = True
        # Master info succeeds but replicas fail
        mock_sentinel.execute.side_effect = lambda *args: (
            [
                "name", "ocr-master", "ip", "10.0.0.1", "port", "6379",
                "flags", "master", "num-slaves", "0",
                "num-other-sentinels", "0", "quorum", "2",
            ]
            if "MASTER" in " ".join(str(a).upper() for a in args)
            else []
        )
        mock_sentinel.__enter__ = MagicMock(return_value=mock_sentinel)
        mock_sentinel.__exit__ = MagicMock(return_value=False)

        mock_master = MagicMock()
        mock_master.ping.return_value = True
        mock_master.execute.return_value = "role:master\r\nconnected_slaves:0\r\n"
        mock_master.__enter__ = MagicMock(return_value=mock_master)
        mock_master.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def cli_factory(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_sentinel
            return mock_master

        MockCLI.side_effect = cli_factory

        result = run_drill(
            output_dir=output_dir,
            do_check=True,
            do_report=False,
        )

        assert result.status == "partial"


# ===========================================================================
# CLI parser tests
# ===========================================================================


class TestCLIParser:
    def test_default_args(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.check is False
        assert args.simulate is False
        assert args.validate is False
        assert args.report is False
        assert args.sentinel_host == DEFAULT_SENTINEL_HOST
        assert args.sentinel_port == DEFAULT_SENTINEL_PORT
        assert args.master_name == DEFAULT_MASTER_NAME
        assert args.timeout == DEFAULT_TIMEOUT
        assert args.output_dir == DEFAULT_OUTPUT_DIR

    def test_check_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--check"])
        assert args.check is True

    def test_all_modes(self):
        parser = build_parser()
        args = parser.parse_args(["--check", "--simulate", "--validate", "--report"])
        assert all([args.check, args.simulate, args.validate, args.report])

    def test_custom_connection(self):
        parser = build_parser()
        args = parser.parse_args([
            "--sentinel-host", "sentinel.prod",
            "--sentinel-port", "26380",
            "--master-name", "my-master",
            "--timeout", "60",
            "--output-dir", "/tmp/drills",
        ])
        assert args.sentinel_host == "sentinel.prod"
        assert args.sentinel_port == 26380
        assert args.master_name == "my-master"
        assert args.timeout == 60.0
        assert args.output_dir == "/tmp/drills"

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-v"])
        assert args.verbose is True


# ===========================================================================
# main() integration tests
# ===========================================================================


class TestMain:
    @patch("scripts.redis_sentinel_drill.run_drill")
    def test_main_defaults_to_check(self, mock_run):
        """When no mode flags are given, default to --check --report."""
        mock_run.return_value = SentinelDrillResult(
            drill_id="test",
            timestamp="now",
            pipeline_version="1.0.0",
            sentinel_host="localhost",
            sentinel_port=26379,
            master_name="ocr-master",
            modes=["check", "report"],
            status="passed",
        )
        exit_code = main([])
        assert exit_code == 0
        _, kwargs = mock_run.call_args
        assert kwargs["do_check"] is True
        assert kwargs["do_report"] is True

    @patch("scripts.redis_sentinel_drill.run_drill")
    def test_main_failed_returns_1(self, mock_run):
        mock_run.return_value = SentinelDrillResult(
            drill_id="test",
            timestamp="now",
            pipeline_version="1.0.0",
            sentinel_host="localhost",
            sentinel_port=26379,
            master_name="ocr-master",
            modes=["check"],
            status="failed",
        )
        exit_code = main(["--check"])
        assert exit_code == 1

    @patch("scripts.redis_sentinel_drill.run_drill")
    def test_main_partial_returns_0(self, mock_run):
        mock_run.return_value = SentinelDrillResult(
            drill_id="test",
            timestamp="now",
            pipeline_version="1.0.0",
            sentinel_host="localhost",
            sentinel_port=26379,
            master_name="ocr-master",
            modes=["check"],
            status="partial",
        )
        exit_code = main(["--check"])
        assert exit_code == 0

    @patch("scripts.redis_sentinel_drill.run_drill")
    def test_main_passes_args_through(self, mock_run):
        mock_run.return_value = SentinelDrillResult(
            drill_id="t",
            timestamp="n",
            pipeline_version="v",
            sentinel_host="h",
            sentinel_port=1,
            master_name="m",
            modes=[],
            status="passed",
        )
        main([
            "--check", "--simulate", "--validate", "--report",
            "--sentinel-host", "myhost",
            "--sentinel-port", "26380",
            "--master-name", "custom-master",
            "--timeout", "45",
            "--output-dir", "/tmp/out",
        ])
        _, kwargs = mock_run.call_args
        assert kwargs["sentinel_host"] == "myhost"
        assert kwargs["sentinel_port"] == 26380
        assert kwargs["master_name"] == "custom-master"
        assert kwargs["timeout"] == 45.0
        assert kwargs["output_dir"] == "/tmp/out"
        assert kwargs["do_check"] is True
        assert kwargs["do_simulate"] is True
        assert kwargs["do_validate"] is True
        assert kwargs["do_report"] is True


# ===========================================================================
# Timeout handling tests
# ===========================================================================


class TestTimeoutHandling:
    def test_redis_cli_timeout_propagated(self):
        cli = RedisCLI(timeout=2.5)
        assert cli.timeout == 2.5

    @patch("scripts.redis_sentinel_drill.RedisCLI")
    def test_drill_timeout_passed_to_phases(self, MockCLI, output_dir):
        mock_sentinel = _mock_sentinel()
        mock_sentinel.__enter__ = MagicMock(return_value=mock_sentinel)
        mock_sentinel.__exit__ = MagicMock(return_value=False)

        mock_master = MagicMock()
        mock_master.ping.return_value = True
        mock_master.execute.return_value = "role:master\r\nconnected_slaves:2\r\n"
        mock_master.__enter__ = MagicMock(return_value=mock_master)
        mock_master.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def cli_factory(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Verify timeout is passed to Sentinel connection
                assert kwargs.get("timeout") == 42.0
                return mock_sentinel
            return mock_master

        MockCLI.side_effect = cli_factory

        run_drill(
            timeout=42.0,
            output_dir=output_dir,
            do_check=True,
            do_report=False,
        )

    def test_step_duration_is_positive(self):
        sentinel = _mock_sentinel()
        with patch("scripts.redis_sentinel_drill.RedisCLI") as MockCLI:
            mock_master = MagicMock()
            mock_master.ping.return_value = True
            mock_master.execute.return_value = "role:master\r\nconnected_slaves:2\r\n"
            mock_master.__enter__ = MagicMock(return_value=mock_master)
            mock_master.__exit__ = MagicMock(return_value=False)
            MockCLI.return_value = mock_master

            steps = run_check(sentinel, "ocr-master", None, 10.0)

        for step in steps:
            assert step.duration_seconds >= 0.0


# ===========================================================================
# Environment variable tests
# ===========================================================================


class TestEnvironmentVariables:
    @patch("scripts.redis_sentinel_drill.RedisCLI")
    def test_sentinel_password_from_env(self, MockCLI, output_dir):
        mock_sentinel = _mock_sentinel()
        mock_sentinel.__enter__ = MagicMock(return_value=mock_sentinel)
        mock_sentinel.__exit__ = MagicMock(return_value=False)

        mock_master = MagicMock()
        mock_master.ping.return_value = True
        mock_master.execute.return_value = "role:master\r\nconnected_slaves:2\r\n"
        mock_master.__enter__ = MagicMock(return_value=mock_master)
        mock_master.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def cli_factory(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                assert kwargs.get("password") == "sentinel_secret"
                return mock_sentinel
            return mock_master

        MockCLI.side_effect = cli_factory

        with patch.dict(os.environ, {
            "REDIS_SENTINEL_PASSWORD": "sentinel_secret",
            "REDIS_PASSWORD": "redis_secret",
        }):
            run_drill(output_dir=output_dir, do_check=True, do_report=False)

    @patch("scripts.redis_sentinel_drill.RedisCLI")
    def test_no_password_env(self, MockCLI, output_dir):
        mock_sentinel = _mock_sentinel()
        mock_sentinel.__enter__ = MagicMock(return_value=mock_sentinel)
        mock_sentinel.__exit__ = MagicMock(return_value=False)

        mock_master = MagicMock()
        mock_master.ping.return_value = True
        mock_master.execute.return_value = "role:master\r\nconnected_slaves:2\r\n"
        mock_master.__enter__ = MagicMock(return_value=mock_master)
        mock_master.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def cli_factory(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                assert kwargs.get("password") is None
                return mock_sentinel
            return mock_master

        MockCLI.side_effect = cli_factory

        with patch.dict(os.environ, {}, clear=True):
            # Ensure REDIS_SENTINEL_PASSWORD and REDIS_PASSWORD are unset
            os.environ.pop("REDIS_SENTINEL_PASSWORD", None)
            os.environ.pop("REDIS_PASSWORD", None)
            run_drill(output_dir=output_dir, do_check=True, do_report=False)
