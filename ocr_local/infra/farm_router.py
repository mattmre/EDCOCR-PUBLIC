"""ES-11 — Cross-server job routing with no double-commit + custody hooks.

Builds on ES-10 FarmRegistry:

  - Route work units to admit-eligible hosts by capability
  - Claim/lease with unique claim_id so the same unit cannot be dual-owned
  - Complete / fail / requeue transitions
  - Optional custody chain events (production CustodyChain) for continuity

Honest residual:
  - Local multi-host stand-in (same process registry), not multi-VM (ES-12)
  - Not yet wired into Celery chord path; this is the farm routing contract
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ocr_local.infra.farm_registry import FarmRegistry


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class WorkUnit:
    """One farm-routable unit of work (doc or page)."""

    unit_id: str
    capability: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending|claimed|completed|failed
    owner_host: str | None = None
    claim_id: str | None = None
    claimed_at: str | None = None
    completed_at: str | None = None
    attempts: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DoubleCommitError(RuntimeError):
    """Raised when a second claim or foreign complete is attempted."""


class FarmRouter:
    """Capability-aware router with exclusive claim leases."""

    def __init__(
        self,
        registry: FarmRegistry,
        *,
        custody_dir: Path | str | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.registry = registry
        self.custody_dir = Path(custody_dir) if custody_dir else None
        self.max_attempts = max_attempts
        self._lock = threading.RLock()
        self._units: dict[str, WorkUnit] = {}
        self._claims: dict[str, str] = {}  # claim_id -> unit_id
        self._custody_events: list[dict[str, Any]] = []

    def submit(
        self, unit_id: str, capability: str, payload: dict[str, Any] | None = None
    ) -> WorkUnit:
        with self._lock:
            if unit_id in self._units:
                raise DoubleCommitError(f"unit already submitted: {unit_id}")
            unit = WorkUnit(
                unit_id=unit_id,
                capability=capability,
                payload=dict(payload or {}),
                status="pending",
            )
            self._units[unit_id] = unit
            self._emit_custody(unit_id, "file_ingested", {"capability": capability})
            return unit

    def route_next(self, capability: str | None = None) -> WorkUnit | None:
        """Claim the next pending unit onto an admit-eligible host."""
        with self._lock:
            pending = [
                u
                for u in self._units.values()
                if u.status == "pending"
                and (capability is None or u.capability == capability)
            ]
            pending.sort(key=lambda u: u.unit_id)
            for unit in pending:
                hosts = self.registry.admit_eligible(unit.capability)
                if not hosts:
                    continue
                # Round-robin-ish: least attempts then hostname order (already sorted)
                host = hosts[0]
                return self._claim_unlocked(unit.unit_id, host.hostname)
            return None

    def claim(self, unit_id: str, hostname: str) -> WorkUnit:
        with self._lock:
            return self._claim_unlocked(unit_id, hostname)

    def _claim_unlocked(self, unit_id: str, hostname: str) -> WorkUnit:
        unit = self._units.get(unit_id)
        if unit is None:
            raise KeyError(f"unknown unit: {unit_id}")
        if unit.status == "claimed":
            raise DoubleCommitError(
                f"unit {unit_id} already claimed by {unit.owner_host} "
                f"claim_id={unit.claim_id}"
            )
        if unit.status == "completed":
            raise DoubleCommitError(f"unit {unit_id} already completed")
        if unit.status not in ("pending", "failed"):
            raise DoubleCommitError(
                f"unit {unit_id} not claimable status={unit.status}"
            )

        # Host must be admit-eligible for capability
        eligible = {h.hostname for h in self.registry.admit_eligible(unit.capability)}
        if hostname not in eligible:
            raise DoubleCommitError(
                f"host {hostname} not admit-eligible for {unit.capability}"
            )

        claim_id = uuid.uuid4().hex
        unit.status = "claimed"
        unit.owner_host = hostname
        unit.claim_id = claim_id
        unit.claimed_at = _utc_now()
        unit.attempts += 1
        self._units[unit_id] = unit
        self._claims[claim_id] = unit_id
        self._emit_custody(
            unit_id,
            "page_extracted",
            {
                "owner_host": hostname,
                "claim_id": claim_id,
                "attempt": unit.attempts,
            },
        )
        return unit

    def complete(
        self, unit_id: str, claim_id: str, *, result: dict[str, Any] | None = None
    ) -> WorkUnit:
        with self._lock:
            unit = self._units.get(unit_id)
            if unit is None:
                raise KeyError(f"unknown unit: {unit_id}")
            if unit.status == "completed":
                raise DoubleCommitError(f"unit {unit_id} already completed")
            if unit.status != "claimed" or unit.claim_id != claim_id:
                raise DoubleCommitError(
                    f"complete rejected: unit={unit_id} status={unit.status} "
                    f"claim_match={unit.claim_id == claim_id}"
                )
            unit.status = "completed"
            unit.completed_at = _utc_now()
            if result:
                unit.payload = {**unit.payload, "result": result}
            self._units[unit_id] = unit
            self._emit_custody(
                unit_id,
                "assembly_complete",
                {
                    "owner_host": unit.owner_host,
                    "claim_id": claim_id,
                    "result_keys": list((result or {}).keys()),
                },
            )
            return unit

    def fail_and_requeue(
        self,
        unit_id: str,
        claim_id: str,
        *,
        error: str,
    ) -> WorkUnit:
        """Release claim for requeue, or fail permanently if max attempts."""
        with self._lock:
            unit = self._units.get(unit_id)
            if unit is None:
                raise KeyError(f"unknown unit: {unit_id}")
            if unit.status != "claimed" or unit.claim_id != claim_id:
                raise DoubleCommitError(
                    f"fail rejected: unit={unit_id} status={unit.status}"
                )
            unit.last_error = error
            if unit.attempts >= self.max_attempts:
                unit.status = "failed"
                unit.owner_host = None
                unit.claim_id = None
                self._emit_custody(
                    unit_id,
                    "processing_failed",
                    {"error": error, "attempts": unit.attempts},
                )
            else:
                unit.status = "pending"
                unit.owner_host = None
                unit.claim_id = None
                unit.claimed_at = None
                self._emit_custody(
                    unit_id,
                    "processing_resumed",
                    {"error": error, "attempts": unit.attempts, "requeued": True},
                )
            self._units[unit_id] = unit
            return unit

    def get(self, unit_id: str) -> WorkUnit | None:
        with self._lock:
            return self._units.get(unit_id)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_status: dict[str, int] = {}
            for u in self._units.values():
                by_status[u.status] = by_status.get(u.status, 0) + 1
            return {
                "units": len(self._units),
                "by_status": by_status,
                "claims_active": sum(
                    1 for u in self._units.values() if u.status == "claimed"
                ),
                "custody_events": len(self._custody_events),
            }

    def custody_event_log(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._custody_events)

    def _emit_custody(
        self, unit_id: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """Append in-memory custody record; optionally write production chain."""
        rec = {
            "unit_id": unit_id,
            "event_type": event_type,
            "timestamp": _utc_now(),
            "data": data,
        }
        self._custody_events.append(rec)
        if not self.custody_dir:
            return
        try:
            from ocr_local.features.custody import CustodyChain  # noqa: WPS433

            # Stable document_id per unit
            doc_id = hashlib.sha256(unit_id.encode()).hexdigest()[:16]
            chain = CustodyChain(
                doc_id, f"farm://unit/{unit_id}", str(self.custody_dir)
            )
            chain.append_event(event_type, {"unit_id": unit_id, **data})
        except Exception as exc:  # noqa: BLE001
            # Do not break routing if custody write fails — record residual
            self._custody_events.append(
                {
                    "unit_id": unit_id,
                    "event_type": "CUSTODY_TSA_WARNING",
                    "timestamp": _utc_now(),
                    "data": {"error": repr(exc), "orig": event_type},
                }
            )
