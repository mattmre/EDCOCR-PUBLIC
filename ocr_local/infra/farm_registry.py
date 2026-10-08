"""ES-10 — Multi-host farm registry (registration, pools, heartbeat, drain).

In-process / file-backed multi-host registry implementing the ES-9
``cross_server_agent`` + ``pool_analyzer`` contracts without requiring a live
multi-VM cluster.

Honest residual:
  - This is a **local multi-host stand-in** (multiple host records in one
    process or shared JSON store), not multi-VM network farm proof (ES-12).
  - Aligns field names with ``coordinator.jobs.models.Worker`` for future
    Django sync; does not replace the ORM Worker table in production path
    until wired (ES-11 routing may consume this registry).

Env:
  OCR_FARM_REGISTRY_PATH=/path/to/registry.json  (optional durable store)
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ocr_distributed.text_io import read_text_bounded

DEFAULT_HEARTBEAT_TTL_S = 30.0
MAX_FARM_REGISTRY_BYTES = 16 * 1024 * 1024
MAX_FARM_HOSTS = 10_000
KNOWN_CAPABILITIES = (
    "gpu_ocr",
    "cpu_ocr",
    "nlp",
    "compression",
    "docintel",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_ts(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        # support Z
        t = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(t).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


@dataclass
class HostRecord:
    """One farm host / worker node (mirrors Worker PK=hostname semantics)."""

    hostname: str
    status: str = "online"  # online | draining | offline
    capabilities: list[str] = field(default_factory=list)
    last_heartbeat: str = ""
    registered_at: str = ""
    load: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HostRecord:
        return cls(
            hostname=str(d["hostname"]),
            status=str(d.get("status") or "online"),
            capabilities=list(d.get("capabilities") or []),
            last_heartbeat=str(d.get("last_heartbeat") or ""),
            registered_at=str(d.get("registered_at") or ""),
            load=dict(d.get("load") or {}),
            meta=dict(d.get("meta") or {}),
        )


class FarmRegistry:
    """Thread-safe multi-host registry with capability pools and drain."""

    def __init__(
        self,
        *,
        path: Path | str | None = None,
        heartbeat_ttl_s: float = DEFAULT_HEARTBEAT_TTL_S,
    ) -> None:
        self.path = Path(path) if path else None
        self.heartbeat_ttl_s = float(heartbeat_ttl_s)
        self._lock = threading.RLock()
        self._hosts: dict[str, HostRecord] = {}
        if self.path and self.path.is_file():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        data = json.loads(
            read_text_bounded(self.path, max_bytes=MAX_FARM_REGISTRY_BYTES)
        )
        hosts = data.get("hosts") or {}
        if not isinstance(hosts, dict):
            raise ValueError("farm registry hosts must be an object")
        if len(hosts) > MAX_FARM_HOSTS:
            raise ValueError(
                f"farm registry exceeds the {MAX_FARM_HOSTS}-host safety limit"
            )
        self._hosts = {
            k: HostRecord.from_dict(v)
            if isinstance(v, dict)
            else HostRecord(hostname=k)
            for k, v in hosts.items()
        }

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": _utc_now(),
            "hosts": {k: v.to_dict() for k, v in self._hosts.items()},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def register(
        self,
        hostname: str,
        capabilities: Iterable[str],
        *,
        load: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> HostRecord:
        caps = [c for c in capabilities if c]
        now = _utc_now()
        with self._lock:
            existing = self._hosts.get(hostname)
            rec = HostRecord(
                hostname=hostname,
                status="online",
                capabilities=caps,
                last_heartbeat=now,
                registered_at=existing.registered_at if existing else now,
                load=dict(load or {}),
                meta=dict(meta or {}),
            )
            self._hosts[hostname] = rec
            self._save()
            return rec

    def heartbeat(
        self,
        hostname: str,
        *,
        load: dict[str, Any] | None = None,
    ) -> HostRecord:
        with self._lock:
            rec = self._hosts.get(hostname)
            if rec is None:
                raise KeyError(f"host not registered: {hostname}")
            if rec.status == "offline":
                rec.status = "online"
            # drain stays draining until drain_complete
            rec.last_heartbeat = _utc_now()
            if load is not None:
                rec.load = dict(load)
            self._hosts[hostname] = rec
            self._save()
            return rec

    def drain(self, hostname: str) -> HostRecord:
        """Mark host draining — no new admit (ES-10); still heartbeats."""
        with self._lock:
            rec = self._hosts.get(hostname)
            if rec is None:
                raise KeyError(f"host not registered: {hostname}")
            rec.status = "draining"
            rec.last_heartbeat = _utc_now()
            self._hosts[hostname] = rec
            self._save()
            return rec

    def drain_complete(self, hostname: str) -> HostRecord:
        """Finish drain: offline and remove from online pools."""
        with self._lock:
            rec = self._hosts.get(hostname)
            if rec is None:
                raise KeyError(f"host not registered: {hostname}")
            rec.status = "offline"
            rec.last_heartbeat = _utc_now()
            self._hosts[hostname] = rec
            self._save()
            return rec

    def expire_stale(self, *, now: float | None = None) -> list[str]:
        """Mark hosts offline when heartbeat TTL exceeded. Returns expired names."""
        now_ts = now if now is not None else time.time()
        expired: list[str] = []
        with self._lock:
            for name, rec in list(self._hosts.items()):
                if rec.status == "offline":
                    continue
                last = _parse_ts(rec.last_heartbeat)
                if last <= 0 or (now_ts - last) > self.heartbeat_ttl_s:
                    rec.status = "offline"
                    self._hosts[name] = rec
                    expired.append(name)
            if expired:
                self._save()
        return expired

    def get(self, hostname: str) -> HostRecord | None:
        with self._lock:
            return self._hosts.get(hostname)

    def list_hosts(self, *, status: str | None = None) -> list[HostRecord]:
        with self._lock:
            rows = list(self._hosts.values())
        if status:
            rows = [r for r in rows if r.status == status]
        return sorted(rows, key=lambda r: r.hostname)

    def online_hosts(self) -> list[HostRecord]:
        """Online + draining (still present) but not offline."""
        with self._lock:
            return sorted(
                [r for r in self._hosts.values() if r.status in ("online", "draining")],
                key=lambda r: r.hostname,
            )

    def admit_eligible(self, capability: str | None = None) -> list[HostRecord]:
        """Hosts that may accept new work (online only, not draining/offline)."""
        with self._lock:
            rows = [r for r in self._hosts.values() if r.status == "online"]
        if capability:
            rows = [r for r in rows if capability in r.capabilities]
        return sorted(rows, key=lambda r: r.hostname)

    def capability_pools(self) -> dict[str, dict[str, Any]]:
        """Pool analyzer view: counts by capability for online admit-eligible."""
        eligible = self.admit_eligible()
        pools: dict[str, dict[str, Any]] = {}
        for cap in KNOWN_CAPABILITIES:
            hosts = [h.hostname for h in eligible if cap in h.capabilities]
            pools[cap] = {
                "capability": cap,
                "online_count": len(hosts),
                "hosts": hosts,
            }
        # also include any unknown caps present
        for h in eligible:
            for cap in h.capabilities:
                if cap not in pools:
                    pools[cap] = {
                        "capability": cap,
                        "online_count": 0,
                        "hosts": [],
                    }
                    hosts = [x.hostname for x in eligible if cap in x.capabilities]
                    pools[cap]["online_count"] = len(hosts)
                    pools[cap]["hosts"] = hosts
        return pools

    def summary(self) -> dict[str, Any]:
        with self._lock:
            statuses = Counter(r.status for r in self._hosts.values())
            n = len(self._hosts)
        return {
            "host_count": n,
            "status_counts": dict(statuses),
            "admit_eligible": len(self.admit_eligible()),
            "pools": self.capability_pools(),
            "heartbeat_ttl_s": self.heartbeat_ttl_s,
            "path": str(self.path) if self.path else None,
        }


def registry_from_env(
    env: dict[str, str] | None = None,
    *,
    heartbeat_ttl_s: float = DEFAULT_HEARTBEAT_TTL_S,
) -> FarmRegistry:
    e = env if env is not None else os.environ
    path = e.get("OCR_FARM_REGISTRY_PATH") or None
    return FarmRegistry(path=path, heartbeat_ttl_s=heartbeat_ttl_s)
