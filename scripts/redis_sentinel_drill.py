#!/usr/bin/env python3
"""Redis Sentinel failover drill framework.

Validates Redis Sentinel configuration and simulates failover scenarios
for the EDCOCR distributed pipeline.  Designed to be run against a
live Sentinel deployment (Docker Compose HA overlay or Kubernetes with
``redis.sentinel.enabled=true``).

Modes:
    --check      Validate Sentinel topology (quorum, auth, replication).
    --simulate   Trigger ``SENTINEL FAILOVER`` and measure promotion time.
    --validate   Post-failover app reconnection checks (Celery, Django cache).
    --report     Generate a drill report from a previous or in-progress run.

Connection:
    --sentinel-host   Sentinel hostname (default: localhost)
    --sentinel-port   Sentinel port     (default: 26379)
    --master-name     Monitored master  (default: ocr-master)
    --timeout         Per-step timeout  (default: 30s)
    --output-dir      Directory for JSON + markdown reports

Environment variables:
    REDIS_SENTINEL_PASSWORD   Sentinel ``requirepass`` credential
    REDIS_PASSWORD            Master/replica ``requirepass`` credential

Usage:
    python scripts/redis_sentinel_drill.py --check
    python scripts/redis_sentinel_drill.py --simulate --validate
    python scripts/redis_sentinel_drill.py --report --output-dir drill_results
    python scripts/redis_sentinel_drill.py --check --simulate --validate --report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ocr_local.config.version import __version__
except ImportError:
    __version__ = "unknown"

logger = logging.getLogger("redis_sentinel_drill")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SENTINEL_HOST = "localhost"
DEFAULT_SENTINEL_PORT = 26379
DEFAULT_MASTER_NAME = "ocr-master"
DEFAULT_TIMEOUT = 30
DEFAULT_OUTPUT_DIR = "sentinel_drill_results"
FAILOVER_POLL_INTERVAL = 0.5  # seconds between promotion checks
MAX_FAILOVER_WAIT = 60  # absolute upper bound for failover wait


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DrillStep:
    """A single step in a Sentinel drill."""

    name: str
    description: str
    status: str = "pending"  # pending | passed | failed | skipped
    duration_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentinelDrillResult:
    """Container for a complete Sentinel drill run."""

    drill_id: str
    timestamp: str
    pipeline_version: str
    sentinel_host: str
    sentinel_port: int
    master_name: str
    modes: list[str]

    # Overall outcome
    status: str = "pending"  # pending | passed | failed | partial
    steps: list[dict] = field(default_factory=list)

    # Failover-specific metrics
    failover_triggered: bool = False
    failover_duration_seconds: float = 0.0
    old_master: str = ""
    new_master: str = ""

    # App reconnection
    celery_reconnected: bool = False
    cache_reconnected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Redis protocol helpers (raw socket, no redis-py dependency)
# ---------------------------------------------------------------------------


class RedisCLI:
    """Minimal Redis client using raw RESP protocol over TCP.

    Avoids requiring the ``redis`` Python package so the drill script
    can run in minimal environments.  Falls back to the ``redis``
    package if available, for richer error messages.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._using_redis_py = False
        self._client: Any = None  # redis.Redis if available

    # -- context manager ---------------------------------------------------

    def connect(self) -> "RedisCLI":
        """Open a TCP connection (or redis-py connection)."""
        # Try redis-py first for better error handling
        try:
            import redis as redis_lib

            kwargs: dict[str, Any] = {
                "host": self.host,
                "port": self.port,
                "socket_timeout": self.timeout,
                "socket_connect_timeout": self.timeout,
                "decode_responses": True,
            }
            if self.password:
                kwargs["password"] = self.password
            self._client = redis_lib.Redis(**kwargs)
            self._client.ping()
            self._using_redis_py = True
            return self
        except Exception:
            pass

        # Fall back to raw socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        if self.password:
            self._send_command("AUTH", self.password)
        return self

    def close(self) -> None:
        if self._using_redis_py and self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self) -> "RedisCLI":
        return self.connect()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- command execution -------------------------------------------------

    def execute(self, *args: str) -> Any:
        """Execute a Redis command and return the parsed response."""
        if self._using_redis_py and self._client is not None:
            return self._client.execute_command(*args)
        return self._send_command(*args)

    def ping(self) -> bool:
        """Return True if the server responds with PONG."""
        try:
            result = self.execute("PING")
            return str(result).upper().strip() in ("PONG", "True", "+PONG")
        except Exception:
            return False

    # -- raw RESP helpers --------------------------------------------------

    def _send_command(self, *args: str) -> Any:
        """Encode *args* as a RESP array, send, and parse the reply."""
        if self._sock is None:
            raise ConnectionError("Not connected")
        # Build RESP request
        parts = [f"*{len(args)}\r\n"]
        for arg in args:
            encoded = str(arg)
            parts.append(f"${len(encoded)}\r\n{encoded}\r\n")
        self._sock.sendall("".join(parts).encode())
        return self._read_response()

    def _read_response(self) -> Any:
        """Read and parse one RESP response."""
        if self._sock is None:
            raise ConnectionError("Not connected")
        line = self._read_line()
        prefix = line[0:1]
        payload = line[1:]

        if prefix == "+":
            return payload
        elif prefix == "-":
            raise RuntimeError(f"Redis error: {payload}")
        elif prefix == ":":
            return int(payload)
        elif prefix == "$":
            length = int(payload)
            if length == -1:
                return None
            data = self._read_exact(length + 2)  # +2 for trailing \r\n
            return data[:-2]  # strip \r\n
        elif prefix == "*":
            count = int(payload)
            if count == -1:
                return None
            return [self._read_response() for _ in range(count)]
        else:
            return line

    def _read_line(self) -> str:
        """Read bytes until \\r\\n."""
        if self._sock is None:
            raise ConnectionError("Not connected")
        buf = b""
        while not buf.endswith(b"\r\n"):
            chunk = self._sock.recv(1)
            if not chunk:
                raise ConnectionError("Connection closed")
            buf += chunk
        return buf.decode().rstrip("\r\n")

    def _read_exact(self, n: int) -> str:
        """Read exactly *n* bytes from the socket."""
        if self._sock is None:
            raise ConnectionError("Not connected")
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed")
            buf += chunk
        return buf.decode()


# ---------------------------------------------------------------------------
# Sentinel introspection helpers
# ---------------------------------------------------------------------------


def parse_sentinel_info(raw: list | None) -> dict[str, str]:
    """Convert a flat key-value list from ``SENTINEL MASTER`` to a dict.

    Sentinel replies with alternating key/value elements::

        ["name", "ocr-master", "ip", "10.0.0.1", "port", "6379", ...]
    """
    if raw is None:
        return {}
    result: dict[str, str] = {}
    items = list(raw)
    for i in range(0, len(items) - 1, 2):
        result[str(items[i])] = str(items[i + 1])
    return result


def parse_sentinel_list(raw: list | None) -> list[dict[str, str]]:
    """Parse a list-of-lists response (e.g. ``SENTINEL REPLICAS``)."""
    if raw is None:
        return []
    return [parse_sentinel_info(entry) for entry in raw if entry]


# ---------------------------------------------------------------------------
# Drill phases
# ---------------------------------------------------------------------------


def run_check(
    sentinel: RedisCLI,
    master_name: str,
    redis_password: str | None,
    timeout: float,
) -> list[DrillStep]:
    """Validate Sentinel configuration and topology.

    Checks performed:
      1. Sentinel PING
      2. Master is reachable via Sentinel
      3. Master responds to PING
      4. At least one replica configured
      5. Quorum is achievable (sentinels >= quorum)
      6. Replication is in sync (no stale replicas)
    """
    steps: list[DrillStep] = []

    # 1 -- Sentinel reachable
    step = DrillStep(name="sentinel_ping", description="Verify Sentinel responds to PING")
    t0 = time.monotonic()
    try:
        if sentinel.ping():
            step.status = "passed"
        else:
            step.status = "failed"
            step.error = "Sentinel did not respond with PONG"
    except Exception as exc:
        step.status = "failed"
        step.error = str(exc)
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 2 -- Master info from Sentinel
    step = DrillStep(name="master_info", description="Retrieve master info from Sentinel")
    t0 = time.monotonic()
    master_info: dict[str, str] = {}
    try:
        raw = sentinel.execute("SENTINEL", "MASTER", master_name)
        master_info = parse_sentinel_info(raw)
        if master_info.get("ip") and master_info.get("port"):
            step.status = "passed"
            step.details = {
                "ip": master_info.get("ip", ""),
                "port": master_info.get("port", ""),
                "flags": master_info.get("flags", ""),
                "num-slaves": master_info.get("num-slaves", "0"),
                "num-other-sentinels": master_info.get("num-other-sentinels", "0"),
                "quorum": master_info.get("quorum", ""),
            }
        else:
            step.status = "failed"
            step.error = "Master info missing ip/port"
    except Exception as exc:
        step.status = "failed"
        step.error = str(exc)
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 3 -- Master PING
    step = DrillStep(name="master_ping", description="Verify master responds to PING")
    t0 = time.monotonic()
    master_host = master_info.get("ip", "")
    master_port = int(master_info.get("port", "6379"))
    if master_host:
        try:
            with RedisCLI(
                host=master_host,
                port=master_port,
                password=redis_password,
                timeout=timeout,
            ) as master_cli:
                if master_cli.ping():
                    step.status = "passed"
                    step.details = {"host": master_host, "port": master_port}
                else:
                    step.status = "failed"
                    step.error = "Master did not respond with PONG"
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
    else:
        step.status = "skipped"
        step.error = "Master host not determined"
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 4 -- Replicas configured
    step = DrillStep(
        name="replicas_check",
        description="Verify at least one replica is configured",
    )
    t0 = time.monotonic()
    try:
        raw_replicas = sentinel.execute("SENTINEL", "REPLICAS", master_name)
        replicas = parse_sentinel_list(raw_replicas)
        step.details = {"replica_count": len(replicas)}
        if replicas:
            step.status = "passed"
            step.details["replicas"] = [
                {"ip": r.get("ip", ""), "port": r.get("port", "")}
                for r in replicas
            ]
        else:
            step.status = "failed"
            step.error = "No replicas found; failover will not be possible"
    except Exception as exc:
        step.status = "failed"
        step.error = str(exc)
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 5 -- Quorum achievable
    step = DrillStep(
        name="quorum_check",
        description="Verify enough Sentinels for quorum",
    )
    t0 = time.monotonic()
    try:
        quorum = int(master_info.get("quorum", "0"))
        num_sentinels = int(master_info.get("num-other-sentinels", "0")) + 1
        step.details = {
            "quorum": quorum,
            "total_sentinels": num_sentinels,
        }
        if num_sentinels >= quorum and quorum > 0:
            step.status = "passed"
        else:
            step.status = "failed"
            step.error = (
                f"Sentinel count ({num_sentinels}) < quorum ({quorum}); "
                "failover will not trigger"
            )
    except Exception as exc:
        step.status = "failed"
        step.error = str(exc)
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 6 -- Replication health
    step = DrillStep(
        name="replication_health",
        description="Verify replication sync status on master",
    )
    t0 = time.monotonic()
    if master_host:
        try:
            with RedisCLI(
                host=master_host,
                port=master_port,
                password=redis_password,
                timeout=timeout,
            ) as master_cli:
                info_raw = master_cli.execute("INFO", "replication")
                info_str = str(info_raw)
                role = ""
                connected_slaves = 0
                for line in info_str.splitlines():
                    if line.startswith("role:"):
                        role = line.split(":")[1].strip()
                    if line.startswith("connected_slaves:"):
                        connected_slaves = int(line.split(":")[1].strip())
                step.details = {
                    "role": role,
                    "connected_slaves": connected_slaves,
                }
                if role == "master" and connected_slaves > 0:
                    step.status = "passed"
                elif role == "master" and connected_slaves == 0:
                    step.status = "failed"
                    step.error = "Master has 0 connected slaves"
                else:
                    step.status = "failed"
                    step.error = f"Unexpected role: {role}"
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
    else:
        step.status = "skipped"
        step.error = "Master host not determined"
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    return steps


def run_simulate(
    sentinel: RedisCLI,
    master_name: str,
    timeout: float,
) -> tuple[list[DrillStep], str, str, float]:
    """Trigger ``SENTINEL FAILOVER`` and measure promotion time.

    Returns (steps, old_master, new_master, failover_duration).
    """
    steps: list[DrillStep] = []
    old_master = ""
    new_master = ""
    failover_duration = 0.0

    # 1 -- Record current master
    step = DrillStep(name="record_master", description="Record current master before failover")
    t0 = time.monotonic()
    try:
        raw = sentinel.execute("SENTINEL", "MASTER", master_name)
        info = parse_sentinel_info(raw)
        old_master = f"{info.get('ip', '')}:{info.get('port', '')}"
        step.status = "passed"
        step.details = {"old_master": old_master}
    except Exception as exc:
        step.status = "failed"
        step.error = str(exc)
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 2 -- Issue failover
    step = DrillStep(name="trigger_failover", description="Send SENTINEL FAILOVER command")
    t0 = time.monotonic()
    try:
        result = sentinel.execute("SENTINEL", "FAILOVER", master_name)
        result_str = str(result).upper()
        if "OK" in result_str:
            step.status = "passed"
        else:
            step.status = "failed"
            step.error = f"Unexpected response: {result}"
    except Exception as exc:
        step.status = "failed"
        step.error = str(exc)
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 3 -- Wait for new master
    step = DrillStep(
        name="wait_promotion",
        description="Wait for replica promotion to complete",
    )
    t0 = time.monotonic()
    deadline = time.monotonic() + min(timeout, MAX_FAILOVER_WAIT)
    promoted = False
    try:
        while time.monotonic() < deadline:
            raw = sentinel.execute("SENTINEL", "MASTER", master_name)
            info = parse_sentinel_info(raw)
            current = f"{info.get('ip', '')}:{info.get('port', '')}"
            flags = info.get("flags", "")
            # Failover is complete when master has changed and flags
            # do not indicate failover-in-progress
            if current != old_master and "master" in flags and "s_down" not in flags:
                new_master = current
                promoted = True
                break
            time.sleep(FAILOVER_POLL_INTERVAL)

        failover_duration = round(time.monotonic() - t0, 3)
        if promoted:
            step.status = "passed"
            step.details = {
                "new_master": new_master,
                "promotion_seconds": failover_duration,
            }
        else:
            step.status = "failed"
            step.error = "Timed out waiting for promotion"
            step.details = {"waited_seconds": failover_duration}
    except Exception as exc:
        step.status = "failed"
        step.error = str(exc)
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    return steps, old_master, new_master, failover_duration


def run_validate(
    sentinel: RedisCLI,
    master_name: str,
    redis_password: str | None,
    timeout: float,
) -> list[DrillStep]:
    """Verify post-failover application reconnection.

    Checks:
      1. New master responds to PING
      2. New master accepts writes (SET/GET round-trip)
      3. Celery result backend connectivity (simulated key check)
      4. Django cache connectivity (simulated key check)
    """
    steps: list[DrillStep] = []

    # Resolve current master from Sentinel
    master_host = ""
    master_port = 6379
    try:
        raw = sentinel.execute("SENTINEL", "MASTER", master_name)
        info = parse_sentinel_info(raw)
        master_host = info.get("ip", "")
        master_port = int(info.get("port", "6379"))
    except Exception:
        pass

    # 1 -- New master PING
    step = DrillStep(name="new_master_ping", description="Verify new master responds to PING")
    t0 = time.monotonic()
    if master_host:
        try:
            with RedisCLI(
                host=master_host,
                port=master_port,
                password=redis_password,
                timeout=timeout,
            ) as cli:
                if cli.ping():
                    step.status = "passed"
                    step.details = {"host": master_host, "port": master_port}
                else:
                    step.status = "failed"
                    step.error = "New master did not respond with PONG"
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
    else:
        step.status = "skipped"
        step.error = "Could not resolve new master from Sentinel"
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 2 -- Write/read round-trip
    step = DrillStep(
        name="write_read_test",
        description="Verify SET/GET round-trip on new master",
    )
    t0 = time.monotonic()
    test_key = f"sentinel_drill:{uuid.uuid4().hex[:8]}"
    test_value = "drill_ok"
    if master_host:
        try:
            with RedisCLI(
                host=master_host,
                port=master_port,
                password=redis_password,
                timeout=timeout,
            ) as cli:
                cli.execute("SET", test_key, test_value, "EX", "60")
                got = cli.execute("GET", test_key)
                if str(got) == test_value:
                    step.status = "passed"
                    step.details = {"key": test_key, "value_match": True}
                else:
                    step.status = "failed"
                    step.error = f"Value mismatch: expected {test_value!r}, got {got!r}"
                # Cleanup
                try:
                    cli.execute("DEL", test_key)
                except Exception:
                    pass
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
    else:
        step.status = "skipped"
        step.error = "Could not resolve new master"
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 3 -- Celery result backend probe
    step = DrillStep(
        name="celery_result_probe",
        description="Verify Celery result backend key namespace is accessible",
    )
    t0 = time.monotonic()
    if master_host:
        try:
            with RedisCLI(
                host=master_host,
                port=master_port,
                password=redis_password,
                timeout=timeout,
            ) as cli:
                # Celery stores results under celery-task-meta-* keys
                # We just verify the keyspace is queryable
                cli.execute("SET", "celery-task-meta-drill-test", "ok", "EX", "10")
                got = cli.execute("GET", "celery-task-meta-drill-test")
                if got:
                    step.status = "passed"
                    step.details = {"celery_keyspace": "accessible"}
                    cli.execute("DEL", "celery-task-meta-drill-test")
                else:
                    step.status = "failed"
                    step.error = "Could not read back Celery test key"
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
    else:
        step.status = "skipped"
        step.error = "Could not resolve new master"
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    # 4 -- Django cache probe
    step = DrillStep(
        name="django_cache_probe",
        description="Verify Django cache keyspace is accessible",
    )
    t0 = time.monotonic()
    if master_host:
        try:
            with RedisCLI(
                host=master_host,
                port=master_port,
                password=redis_password,
                timeout=timeout,
            ) as cli:
                cache_key = ":1:django_drill_test"
                cli.execute("SET", cache_key, "ok", "EX", "10")
                got = cli.execute("GET", cache_key)
                if got:
                    step.status = "passed"
                    step.details = {"django_cache": "accessible"}
                    cli.execute("DEL", cache_key)
                else:
                    step.status = "failed"
                    step.error = "Could not read back Django cache test key"
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
    else:
        step.status = "skipped"
        step.error = "Could not resolve new master"
    step.duration_seconds = round(time.monotonic() - t0, 3)
    steps.append(step)

    return steps


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_json_report(result: SentinelDrillResult, output_dir: str) -> str:
    """Write a JSON drill report and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"sentinel_drill_{result.drill_id}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return filepath


def generate_markdown_report(result: SentinelDrillResult, output_dir: str) -> str:
    """Write a markdown summary and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"sentinel_drill_{result.drill_id}.md"
    filepath = os.path.join(output_dir, filename)

    lines: list[str] = []
    lines.append("# Redis Sentinel Failover Drill Report")
    lines.append("")
    lines.append(f"- **Drill ID**: `{result.drill_id}`")
    lines.append(f"- **Timestamp**: {result.timestamp}")
    lines.append(f"- **Pipeline Version**: {result.pipeline_version}")
    lines.append(f"- **Sentinel**: `{result.sentinel_host}:{result.sentinel_port}`")
    lines.append(f"- **Master Name**: `{result.master_name}`")
    lines.append(f"- **Modes**: {', '.join(result.modes)}")
    lines.append(f"- **Overall Status**: **{result.status.upper()}**")
    lines.append("")

    if result.failover_triggered:
        lines.append("## Failover Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Old Master | `{result.old_master}` |")
        lines.append(f"| New Master | `{result.new_master}` |")
        lines.append(f"| Failover Duration | {result.failover_duration_seconds:.3f}s |")
        rto_target = 15.0
        rto_met = result.failover_duration_seconds <= rto_target
        lines.append(f"| RTO Target | {rto_target}s |")
        lines.append(f"| RTO Met | {'Yes' if rto_met else 'No'} |")
        lines.append("")

    lines.append("## Steps")
    lines.append("")
    lines.append("| # | Step | Description | Status | Duration |")
    lines.append("|---|------|-------------|--------|----------|")
    for i, step_dict in enumerate(result.steps, 1):
        name = step_dict.get("name", "")
        desc = step_dict.get("description", "")
        status = step_dict.get("status", "pending")
        duration = step_dict.get("duration_seconds", 0.0)
        error = step_dict.get("error", "")
        status_display = status.upper()
        if error:
            status_display += f" ({error[:60]})"
        lines.append(f"| {i} | `{name}` | {desc} | {status_display} | {duration:.3f}s |")
    lines.append("")

    passed = sum(1 for s in result.steps if s.get("status") == "passed")
    failed = sum(1 for s in result.steps if s.get("status") == "failed")
    skipped = sum(1 for s in result.steps if s.get("status") == "skipped")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Passed**: {passed}")
    lines.append(f"- **Failed**: {failed}")
    lines.append(f"- **Skipped**: {skipped}")
    lines.append(f"- **Total**: {len(result.steps)}")
    lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return filepath


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_drill(
    sentinel_host: str = DEFAULT_SENTINEL_HOST,
    sentinel_port: int = DEFAULT_SENTINEL_PORT,
    master_name: str = DEFAULT_MASTER_NAME,
    timeout: float = DEFAULT_TIMEOUT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    do_check: bool = True,
    do_simulate: bool = False,
    do_validate: bool = False,
    do_report: bool = True,
) -> SentinelDrillResult:
    """Execute the requested drill phases and return the result."""
    sentinel_password = os.environ.get("REDIS_SENTINEL_PASSWORD", "") or None
    redis_password = os.environ.get("REDIS_PASSWORD", "") or None

    modes: list[str] = []
    if do_check:
        modes.append("check")
    if do_simulate:
        modes.append("simulate")
    if do_validate:
        modes.append("validate")
    if do_report:
        modes.append("report")

    result = SentinelDrillResult(
        drill_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        pipeline_version=__version__,
        sentinel_host=sentinel_host,
        sentinel_port=sentinel_port,
        master_name=master_name,
        modes=modes,
    )

    logger.info(
        "Starting Sentinel drill %s (modes=%s, sentinel=%s:%d, master=%s)",
        result.drill_id,
        modes,
        sentinel_host,
        sentinel_port,
        master_name,
    )

    try:
        with RedisCLI(
            host=sentinel_host,
            port=sentinel_port,
            password=sentinel_password,
            timeout=timeout,
        ) as sentinel:
            # Phase 1: Check
            if do_check:
                logger.info("--- Phase: CHECK ---")
                check_steps = run_check(sentinel, master_name, redis_password, timeout)
                result.steps.extend([s.to_dict() for s in check_steps])

            # Phase 2: Simulate
            if do_simulate:
                logger.info("--- Phase: SIMULATE ---")
                sim_steps, old_master, new_master, duration = run_simulate(
                    sentinel, master_name, timeout
                )
                result.steps.extend([s.to_dict() for s in sim_steps])
                result.failover_triggered = True
                result.old_master = old_master
                result.new_master = new_master
                result.failover_duration_seconds = duration

            # Phase 3: Validate
            if do_validate:
                logger.info("--- Phase: VALIDATE ---")
                val_steps = run_validate(
                    sentinel, master_name, redis_password, timeout
                )
                result.steps.extend([s.to_dict() for s in val_steps])
                result.celery_reconnected = any(
                    s.to_dict().get("name") == "celery_result_probe"
                    and s.to_dict().get("status") == "passed"
                    for s in val_steps
                )
                result.cache_reconnected = any(
                    s.to_dict().get("name") == "django_cache_probe"
                    and s.to_dict().get("status") == "passed"
                    for s in val_steps
                )

    except Exception as exc:
        logger.error("Sentinel drill failed to connect: %s", exc)
        result.steps.append(
            DrillStep(
                name="connection",
                description="Connect to Sentinel",
                status="failed",
                error=str(exc),
            ).to_dict()
        )

    # Determine overall status
    failed_count = sum(1 for s in result.steps if s.get("status") == "failed")
    passed_count = sum(1 for s in result.steps if s.get("status") == "passed")
    if failed_count == 0 and passed_count > 0:
        result.status = "passed"
    elif failed_count > 0 and passed_count > 0:
        result.status = "partial"
    elif failed_count > 0:
        result.status = "failed"

    # Phase 4: Report
    if do_report:
        logger.info("--- Phase: REPORT ---")
        json_path = generate_json_report(result, output_dir)
        md_path = generate_markdown_report(result, output_dir)
        logger.info("JSON report: %s", json_path)
        logger.info("Markdown report: %s", md_path)

    logger.info("Drill %s completed: %s", result.drill_id, result.status)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Redis Sentinel failover drill framework for EDCOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate Sentinel configuration (quorum, auth, replication)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Trigger SENTINEL FAILOVER and measure promotion time",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Post-failover reconnection checks (Celery, Django cache)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate drill report (JSON + markdown)",
    )
    parser.add_argument(
        "--sentinel-host",
        default=DEFAULT_SENTINEL_HOST,
        help=f"Sentinel hostname (default: {DEFAULT_SENTINEL_HOST})",
    )
    parser.add_argument(
        "--sentinel-port",
        type=int,
        default=DEFAULT_SENTINEL_PORT,
        help=f"Sentinel port (default: {DEFAULT_SENTINEL_PORT})",
    )
    parser.add_argument(
        "--master-name",
        default=DEFAULT_MASTER_NAME,
        help=f"Monitored master name (default: {DEFAULT_MASTER_NAME})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-step timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Default to --check if no mode specified
    if not any([args.check, args.simulate, args.validate, args.report]):
        args.check = True
        args.report = True

    result = run_drill(
        sentinel_host=args.sentinel_host,
        sentinel_port=args.sentinel_port,
        master_name=args.master_name,
        timeout=args.timeout,
        output_dir=args.output_dir,
        do_check=args.check,
        do_simulate=args.simulate,
        do_validate=args.validate,
        do_report=args.report,
    )

    # Print summary to stdout
    print(json.dumps(result.to_dict(), indent=2, default=str))

    return 0 if result.status in ("passed", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
