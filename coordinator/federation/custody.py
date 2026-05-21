"""Cross-cluster custody chain -- Plan C Phase 1, item C6.

Forensic feature that links per-cluster custody segments into a single
end-to-end chain so an auditor can verify a job's full lineage across
federated clusters.

Scope
=====
This module **only** emits and replicates ``CrossClusterCustodyEvent``
records that describe what happened at the *cluster boundary* of a job's
lifecycle (handed off, received, rebalanced).  Per-page custody events
remain in the existing ``coordinator.jobs`` custody pipeline.

Integration surface
===================
1. ``CustodyChainReconcilerHook`` is the object plugged into the C2
   reconciler (``Reconciler.custody_chain_engine``).  Its
   ``tick(now=...)`` method is exception-isolated by the reconciler and
   pulls fresh ``_RebalanceObservation`` records from a
   ``rebalance_provider`` callable (typically wired to the C5 failover
   engine's emit queue).

2. ``CrossClusterCustodyChain.replicate_to_peer`` is called whenever a
   job hand-off, receive, or rebalance event needs to be sent to a peer
   cluster's ``/api/v1/federation/custody/ingest`` endpoint.
   Replication failure never blocks the local write; it is logged and
   returned as ``False``.

3. ``build_custody_chain_from_env()`` is the factory that lifts
   configuration out of process environment variables.  When the
   feature is disabled (``OCR_FEDERATION_CUSTODY_ENABLED`` unset or
   not truthy), the factory returns ``None`` and callers must treat
   that as "feature off".

Hashing & signing
=================
Each event has a SHA-256 hash computed over a canonical-JSON
representation of the event body (without the signature field).  The
hash chain is anchored to the previous event's hash for ``HANDED_OFF``
and ``RECEIVED`` events.  ``REBALANCED`` events anchor to the Job row
in the coordinator DB (no parent hash required) -- ``verify_chain``
skips the parent-hash continuity check for those.

The signature is HMAC-SHA256 over the same canonical payload using
``OCR_FEDERATION_CUSTODY_HMAC_KEY``.  Peer clusters must verify both
the hash continuity and the signature before accepting an event.

Insecure mode
=============
``OCR_FEDERATION_CUSTODY_INSECURE=true`` emits an audit-log warning
when the engine is constructed.  It still requires an HMAC key (we do
not turn off signing entirely); the flag is intended as a documented
audit-trail marker for non-production environments.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Optional Prometheus dependency: same shim pattern as the reconciler /
# failover engine so unit tests can run on minimal interpreters.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter  # type: ignore[import-not-found]

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - executed in unit tests
    _PROMETHEUS_AVAILABLE = False

    class _NoopCounter:
        def labels(self, **_kwargs: Any) -> "_NoopCounter":
            return self

        def inc(self, _amount: float = 1.0) -> None:
            return None

    Counter = _NoopCounter  # type: ignore[assignment,misc]


_LOG = logging.getLogger("federation.custody")


# ---------------------------------------------------------------------------
# Event-type constants
# ---------------------------------------------------------------------------
EVENT_JOB_HANDED_OFF = "JOB_HANDED_OFF"
EVENT_JOB_RECEIVED = "JOB_RECEIVED"
EVENT_JOB_REBALANCED = "JOB_REBALANCED"

VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {EVENT_JOB_HANDED_OFF, EVENT_JOB_RECEIVED, EVENT_JOB_REBALANCED}
)

# Default reason code for received events.
_RECEIVED_DEFAULT_REASON = "job_received_from_peer"


# ---------------------------------------------------------------------------
# Prometheus counter (lazy-init under lock to avoid "Duplicated timeseries"
# when the module is re-imported by multiple tests).
# ---------------------------------------------------------------------------
_COUNTER_LOCK = threading.Lock()
_EVENTS_COUNTER: Any | None = None


def _get_events_counter() -> Any:
    """Return the module-level events counter, creating it on first use."""
    global _EVENTS_COUNTER
    if _EVENTS_COUNTER is not None:
        return _EVENTS_COUNTER
    with _COUNTER_LOCK:
        if _EVENTS_COUNTER is None:
            try:
                _EVENTS_COUNTER = Counter(
                    "ocr_federation_custody_events_total",
                    "Cross-cluster custody chain events emitted",
                    labelnames=("event_type", "cluster"),
                )
            except ValueError:
                # prometheus_client raises ValueError when a metric of
                # the same name already exists in the default registry.
                # Fall back to a no-op so the caller is unaffected.
                class _Local:
                    def labels(self, **_kw: Any) -> "_Local":
                        return self

                    def inc(self, _amount: float = 1.0) -> None:
                        return None

                _EVENTS_COUNTER = _Local()
    return _EVENTS_COUNTER


# ---------------------------------------------------------------------------
# Canonical JSON + signing helpers
# ---------------------------------------------------------------------------
def canonical_json(payload: dict[str, Any]) -> str:
    """Return the canonical JSON encoding (str) used for hashing + signing.

    Keys are sorted, separators have no whitespace.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return canonical_json(payload).encode("utf-8")


def _sha256_hex(payload: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical encoding of ``payload``."""
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def compute_event_hash(payload: dict[str, Any]) -> str:
    """Public helper: SHA-256 of the canonical body (signature stripped)."""
    body = {k: v for k, v in payload.items() if k != "signature"}
    return _sha256_hex(body)


def compute_signature(payload: dict[str, Any], hmac_key: str) -> str:
    """Return the HMAC-SHA256 hex digest of ``payload`` under ``hmac_key``.

    When ``hmac_key`` is empty, return an empty string.  This lets the
    caller treat "no key" as "no signature" without raising.
    """
    if not hmac_key:
        return ""
    body = {k: v for k, v in payload.items() if k != "signature"}
    return hmac.new(
        hmac_key.encode("utf-8"),
        _canonical_bytes(body),
        hashlib.sha256,
    ).hexdigest()


def verify_signature(
    event_or_payload: "CrossClusterCustodyEvent | dict[str, Any]",
    hmac_key: str,
    *,
    insecure: bool = False,
) -> bool:
    """Constant-time signature verification.

    Accepts either a ``CrossClusterCustodyEvent`` or a raw ``dict``.

    When ``insecure=True`` the function returns ``True`` whenever the
    payload includes *some* signature.  Production deployments should
    keep ``insecure=False``.
    """
    if isinstance(event_or_payload, CrossClusterCustodyEvent):
        payload = event_or_payload.to_signing_payload_with_signature()
    else:
        payload = dict(event_or_payload)

    sig = payload.get("signature")
    if not isinstance(sig, str) or not sig:
        return False
    if insecure:
        return True
    if not hmac_key:
        return False
    expected = compute_signature(payload, hmac_key)
    if not expected:
        return False
    return hmac.compare_digest(sig, expected)


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossClusterCustodyEvent:
    """A single cross-cluster custody chain entry.

    The ``signature`` is computed *over the rest of the fields* using the
    cluster-local HMAC key.  ``parent_event_hash`` is the SHA-256 hash
    of the previous event in the same job's chain (or empty for
    ``JOB_REBALANCED`` events, which anchor to the Job row).
    """

    job_id: str
    source_cluster: str
    target_cluster: str
    parent_event_hash: str
    event_type: str
    timestamp_utc: str
    dispatch_reason: str
    signature: str = ""

    def to_signing_payload(self) -> dict[str, Any]:
        """Return the body used for signing (signature field excluded)."""
        return {
            "job_id": self.job_id,
            "source_cluster": self.source_cluster,
            "target_cluster": self.target_cluster,
            "parent_event_hash": self.parent_event_hash,
            "event_type": self.event_type,
            "timestamp_utc": self.timestamp_utc,
            "dispatch_reason": self.dispatch_reason,
        }

    def to_signing_payload_with_signature(self) -> dict[str, Any]:
        """Full payload including signature (used by ``verify_signature``)."""
        body = self.to_signing_payload()
        body["signature"] = self.signature
        return body

    def to_payload(self) -> dict[str, Any]:
        """Alias for ``to_signing_payload_with_signature``.

        Kept for symmetry with the wire shape consumed by the
        ``federation_custody`` ingest endpoint.
        """
        return self.to_signing_payload_with_signature()

    def signed(self, hmac_key: str) -> "CrossClusterCustodyEvent":
        """Return a copy of this event with ``signature`` populated."""
        sig = compute_signature(self.to_signing_payload(), hmac_key)
        return CrossClusterCustodyEvent(
            job_id=self.job_id,
            source_cluster=self.source_cluster,
            target_cluster=self.target_cluster,
            parent_event_hash=self.parent_event_hash,
            event_type=self.event_type,
            timestamp_utc=self.timestamp_utc,
            dispatch_reason=self.dispatch_reason,
            signature=sig,
        )


def _utc_iso(ts: float | None = None) -> str:
    """Return an ISO-8601 timestamp with millisecond precision and ``+00:00``."""
    if ts is None:
        ts = time.time()
    seconds = int(ts)
    millis = int((ts - seconds) * 1000)
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
    return f"{base}.{millis:03d}+00:00"


# ---------------------------------------------------------------------------
# Rebalance observation (used by the reconciler hook)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _RebalanceObservation:
    """A pending rebalance the failover engine wants logged.

    ``CustodyChainReconcilerHook`` consumes these via a
    ``rebalance_provider`` callable.  Each (job_id, target_cluster) pair
    is emitted at most once across reconciler ticks (see
    ``CustodyChainReconcilerHook._seen``).
    """

    job_id: str
    source_cluster: str
    target_cluster: str
    reason: str


# ---------------------------------------------------------------------------
# Custody chain
# ---------------------------------------------------------------------------
class CrossClusterCustodyChain:
    """Append-only, hash-chained custody log spanning federated clusters.

    Instances are thread-safe at the Python-object level: each public
    mutation goes through ``_lock``.  Persistence on the receiving side
    happens via the ``federation_custody`` ingest endpoint, which writes
    events to the coordinator's SQLite store.
    """

    def __init__(
        self,
        *,
        local_cluster: str,
        hmac_key: str,
        auth_token: str = "",
        insecure: bool = False,
        clock: Callable[[], float] | None = None,
        opener: Any = None,
    ) -> None:
        if not local_cluster:
            raise ValueError("local_cluster must be a non-empty string")
        if not hmac_key:
            raise ValueError("hmac_key must be a non-empty string")
        self.local_cluster = local_cluster
        self.hmac_key = hmac_key
        self.auth_token = auth_token
        self._insecure = bool(insecure)
        self._clock: Callable[[], float] = clock or time.time
        self._opener: Callable[..., Any] | None = opener
        self._lock = threading.Lock()
        if self._insecure:
            _LOG.warning(
                "custody.insecure_mode_enabled cluster=%s -- HMAC signatures "
                "will be accepted without strict verification on receive",
                local_cluster,
            )

    # -- helpers (also used by tests) -------------------------------------
    @staticmethod
    def _sha256_hex(payload: dict[str, Any]) -> str:
        """Public-for-tests helper: SHA-256 hex of canonical encoding."""
        return _sha256_hex(payload)

    # -- public surface ---------------------------------------------------
    def record_handoff(
        self,
        job_id: str,
        target_cluster: str,
        parent_event_hash: str,
        dispatch_reason: str,
    ) -> CrossClusterCustodyEvent:
        """Record a JOB_HANDED_OFF event."""
        return self._build_event(
            job_id=job_id,
            source_cluster=self.local_cluster,
            target_cluster=target_cluster,
            parent_event_hash=parent_event_hash,
            event_type=EVENT_JOB_HANDED_OFF,
            dispatch_reason=dispatch_reason,
        )

    def record_receive(
        self,
        job_id: str,
        source_cluster: str,
        parent_event_hash: str,
    ) -> CrossClusterCustodyEvent:
        """Record a JOB_RECEIVED event.

        ``parent_event_hash`` is the SHA-256 of the corresponding
        ``JOB_HANDED_OFF`` event from the source cluster.
        """
        return self._build_event(
            job_id=job_id,
            source_cluster=source_cluster,
            target_cluster=self.local_cluster,
            parent_event_hash=parent_event_hash,
            event_type=EVENT_JOB_RECEIVED,
            dispatch_reason=_RECEIVED_DEFAULT_REASON,
        )

    def record_rebalance(
        self,
        job_id: str,
        source_cluster: str,
        target_cluster: str,
        reason: str,
    ) -> CrossClusterCustodyEvent:
        """Record a JOB_REBALANCED event.

        Rebalance events anchor to the Job row in the coordinator
        database, not to a previous event hash, so
        ``parent_event_hash`` is intentionally empty.
        """
        return self._build_event(
            job_id=job_id,
            source_cluster=source_cluster,
            target_cluster=target_cluster,
            parent_event_hash="",
            event_type=EVENT_JOB_REBALANCED,
            dispatch_reason=reason,
        )

    def verify_chain(
        self,
        events: list[CrossClusterCustodyEvent],
    ) -> tuple[bool, str]:
        """Walk an ordered chain and verify hash continuity + signatures.

        Returns ``(ok, message)``.  Empty chains are considered valid.
        """
        if not events:
            return True, "empty chain (0 events)"

        previous_hash: str | None = None
        for idx, event in enumerate(events):
            if event.event_type not in VALID_EVENT_TYPES:
                return (
                    False,
                    f"event_type guard failed at index {idx}: {event.event_type!r}",
                )
            if not verify_signature(
                event, self.hmac_key, insecure=self._insecure
            ):
                return (
                    False,
                    f"signature verification failed at index {idx}",
                )
            if event.event_type != EVENT_JOB_REBALANCED:
                if previous_hash is not None and event.parent_event_hash != previous_hash:
                    return (
                        False,
                        f"parent-hash mismatch at index {idx}: "
                        f"expected {previous_hash!r}, "
                        f"got {event.parent_event_hash!r}",
                    )
            previous_hash = _sha256_hex(event.to_signing_payload())
        return True, f"chain ok ({len(events)} events)"

    def replicate_to_peer(
        self,
        event: CrossClusterCustodyEvent,
        url: str,
        *,
        timeout: float = 5.0,
    ) -> bool:
        """POST a single event to a peer's custody-ingest endpoint.

        Returns ``True`` on 2xx response, ``False`` on transport,
        protocol, or auth failure.  Never raises.
        """
        if not url:
            return False
        body = json.dumps(event.to_payload()).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.auth_token:
            req.add_header("Authorization", f"Bearer {self.auth_token}")
        opener = self._opener
        try:
            if opener is not None:
                resp = opener(req, timeout=timeout)
            else:
                resp = urllib.request.urlopen(req, timeout=timeout)
            try:
                status = getattr(resp, "status", None)
                if status is None and hasattr(resp, "getcode"):
                    status = resp.getcode()
                ok = bool(status and 200 <= int(status) < 300)
                if not ok:
                    _LOG.warning(
                        "custody.replicate.http_status status=%s url=%s",
                        status,
                        url,
                    )
                return ok
            finally:
                close = getattr(resp, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        except urllib.error.HTTPError as exc:
            _LOG.warning(
                "custody.replicate.http_error url=%s status=%s",
                url,
                getattr(exc, "code", "?"),
            )
            return False
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            _LOG.warning(
                "custody.replicate.transport_error url=%s error=%s",
                url,
                exc,
            )
            return False
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning(
                "custody.replicate.unexpected url=%s error=%s",
                url,
                exc,
            )
            return False

    # -- internals --------------------------------------------------------
    def _build_event(
        self,
        *,
        job_id: str,
        source_cluster: str,
        target_cluster: str,
        parent_event_hash: str,
        event_type: str,
        dispatch_reason: str,
    ) -> CrossClusterCustodyEvent:
        if not job_id:
            raise ValueError("job_id must be a non-empty string")
        if not target_cluster:
            raise ValueError("target_cluster must be a non-empty string")
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"unknown event_type: {event_type!r}")
        timestamp = _utc_iso(self._clock())
        unsigned = CrossClusterCustodyEvent(
            job_id=job_id,
            source_cluster=source_cluster,
            target_cluster=target_cluster,
            parent_event_hash=parent_event_hash,
            event_type=event_type,
            timestamp_utc=timestamp,
            dispatch_reason=dispatch_reason,
            signature="",
        )
        with self._lock:
            event = unsigned.signed(self.hmac_key)
        try:
            _get_events_counter().labels(
                event_type=event_type, cluster=self.local_cluster
            ).inc()
        except Exception:  # pragma: no cover - prometheus optional
            pass
        return event


# ---------------------------------------------------------------------------
# Reconciler hook
# ---------------------------------------------------------------------------
class CustodyChainReconcilerHook:
    """Reconciler-friendly wrapper around a ``CrossClusterCustodyChain``.

    The C2 reconciler calls ``tick(now=...)`` once per cycle.  The hook
    pulls a fresh batch of pending ``_RebalanceObservation`` records
    from a caller-supplied provider and emits a JOB_REBALANCED custody
    event for each one that has not been emitted before.

    Idempotency is keyed on ``(job_id, target_cluster)`` so a stuck
    failover queue does not produce duplicate custody entries.
    """

    def __init__(
        self,
        chain: CrossClusterCustodyChain,
        *,
        rebalance_provider: Callable[[], Iterable[_RebalanceObservation]] | None = None,
    ) -> None:
        self.chain = chain
        self.rebalance_provider = rebalance_provider
        self._lock = threading.Lock()
        self._seen: set[tuple[str, str]] = set()

    def tick(self, *, now: float | None = None) -> int:
        """Emit JOB_REBALANCED events for any pending observations.

        Returns the number of new events emitted this tick.  Provider
        and per-observation failures are isolated -- a single bad
        observation cannot stop the rest of the batch.
        """
        if self.rebalance_provider is None:
            return 0
        try:
            observations = list(self.rebalance_provider())
        except Exception as exc:
            _LOG.warning(
                "custody.tick.provider_error error=%s", exc
            )
            return 0
        emitted = 0
        with self._lock:
            for obs in observations:
                if not isinstance(obs, _RebalanceObservation):
                    continue
                key = (obs.job_id, obs.target_cluster)
                if key in self._seen:
                    continue
                try:
                    self.chain.record_rebalance(
                        job_id=obs.job_id,
                        source_cluster=obs.source_cluster,
                        target_cluster=obs.target_cluster,
                        reason=obs.reason,
                    )
                except Exception as exc:
                    _LOG.warning(
                        "custody.tick.emit_error job_id=%s error=%s",
                        obs.job_id,
                        exc,
                    )
                    continue
                self._seen.add(key)
                emitted += 1
        return emitted


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_custody_chain_from_env(
    env: dict[str, str] | None = None,
) -> CrossClusterCustodyChain | None:
    """Build a custody chain from process env vars; return ``None`` if disabled.

    Required env vars (when enabled):
      * ``OCR_FEDERATION_CUSTODY_ENABLED`` -- truthy to activate.
      * ``OCR_CLUSTER_NAME`` -- this cluster's name.
      * ``OCR_FEDERATION_CUSTODY_HMAC_KEY`` -- shared signing key.

    Optional:
      * ``OCR_FEDERATION_CUSTODY_AUTH_TOKEN`` -- bearer token forwarded
        on outbound replication POSTs.
      * ``OCR_FEDERATION_CUSTODY_INSECURE`` -- emit an audit warning
        and accept signatures without strict verification on receive.
    """
    src = env if env is not None else os.environ

    enabled = src.get("OCR_FEDERATION_CUSTODY_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled:
        return None

    local_cluster = src.get("OCR_CLUSTER_NAME", "").strip()
    if not local_cluster:
        _LOG.warning(
            "custody.factory.missing OCR_CLUSTER_NAME"
        )
        return None

    hmac_key = src.get("OCR_FEDERATION_CUSTODY_HMAC_KEY", "").strip()
    if not hmac_key:
        _LOG.warning(
            "custody.factory.missing OCR_FEDERATION_CUSTODY_HMAC_KEY"
        )
        return None

    auth_token = src.get("OCR_FEDERATION_CUSTODY_AUTH_TOKEN", "").strip()
    insecure = src.get("OCR_FEDERATION_CUSTODY_INSECURE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return CrossClusterCustodyChain(
        local_cluster=local_cluster,
        hmac_key=hmac_key,
        auth_token=auth_token,
        insecure=insecure,
    )


__all__ = [
    "CrossClusterCustodyChain",
    "CrossClusterCustodyEvent",
    "CustodyChainReconcilerHook",
    "EVENT_JOB_HANDED_OFF",
    "EVENT_JOB_RECEIVED",
    "EVENT_JOB_REBALANCED",
    "VALID_EVENT_TYPES",
    "build_custody_chain_from_env",
    "canonical_json",
    "compute_event_hash",
    "compute_signature",
    "verify_signature",
]
