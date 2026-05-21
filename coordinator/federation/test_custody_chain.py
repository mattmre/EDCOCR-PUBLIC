"""Unit tests for the cross-cluster custody chain (Plan C Phase 1, item C6).

These tests cover:

* Event creation and signing
* Chain verification: happy path, broken parent-hash, bad signature,
  empty chain, single-event chain, mixed event types
* Replication: 200 OK, 401 unauthorized, 422 signature mismatch,
  network error, empty URL
* Reconciler integration via ``CustodyChainReconcilerHook``: emit on
  rebalance, no emit when ``custody_chain_engine=None``, exception
  isolation, idempotency
* Factory: returns ``None`` when disabled, returns engine when enabled,
  validates required env vars
* Insecure mode: env var override, audit log warning emitted

Prometheus metrics use a lazy-init counter (or a no-op shim when
``prometheus_client`` is absent) so the suite is deterministic.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest

from federation.custody import (
    EVENT_JOB_HANDED_OFF,
    EVENT_JOB_REBALANCED,
    EVENT_JOB_RECEIVED,
    VALID_EVENT_TYPES,
    CrossClusterCustodyChain,
    CrossClusterCustodyEvent,
    CustodyChainReconcilerHook,
    _RebalanceObservation,
    build_custody_chain_from_env,
    canonical_json,
    compute_signature,
    verify_signature,
)

HMAC_KEY = "shared-test-hmac-secret-do-not-use-in-prod"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def chain() -> CrossClusterCustodyChain:
    return CrossClusterCustodyChain(
        local_cluster="cluster-a",
        hmac_key=HMAC_KEY,
        auth_token="bearer-token-abc",
        clock=lambda: 1_700_000_000.123,
    )


class _FakeResponse:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status = status
        self._buf = io.BytesIO(body)

    def read(self) -> bytes:
        return self._buf.read()

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Event creation + signing
# ---------------------------------------------------------------------------
class TestEventCreation:
    def test_record_handoff_returns_signed_event(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        event = chain.record_handoff(
            job_id="job-1",
            target_cluster="cluster-b",
            parent_event_hash="abc123",
            dispatch_reason="region_affinity",
        )
        assert isinstance(event, CrossClusterCustodyEvent)
        assert event.job_id == "job-1"
        assert event.source_cluster == "cluster-a"
        assert event.target_cluster == "cluster-b"
        assert event.parent_event_hash == "abc123"
        assert event.event_type == EVENT_JOB_HANDED_OFF
        assert event.dispatch_reason == "region_affinity"
        assert event.signature  # non-empty
        assert verify_signature(event, HMAC_KEY)

    def test_record_receive_uses_local_as_target(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        event = chain.record_receive(
            job_id="job-2",
            source_cluster="cluster-b",
            parent_event_hash="def456",
        )
        assert event.source_cluster == "cluster-b"
        assert event.target_cluster == "cluster-a"
        assert event.event_type == EVENT_JOB_RECEIVED
        assert event.dispatch_reason == "job_received_from_peer"

    def test_record_rebalance_has_empty_parent_hash(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        event = chain.record_rebalance(
            job_id="job-3",
            source_cluster="cluster-down",
            target_cluster="cluster-a",
            reason="cluster_unhealthy",
        )
        assert event.event_type == EVENT_JOB_REBALANCED
        assert event.parent_event_hash == ""
        assert event.dispatch_reason == "cluster_unhealthy"
        assert verify_signature(event, HMAC_KEY)

    def test_signature_is_deterministic(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        e1 = chain.record_handoff("job-x", "cluster-b", "h1", "reason")
        e2 = chain.record_handoff("job-x", "cluster-b", "h1", "reason")
        # Timestamps are frozen by the fixture clock, so signatures match.
        assert e1.signature == e2.signature

    def test_canonical_json_is_stable(self) -> None:
        a = canonical_json({"b": 1, "a": 2})
        b = canonical_json({"a": 2, "b": 1})
        assert a == b == '{"a":2,"b":1}'

    def test_compute_signature_with_empty_key_returns_empty(self) -> None:
        assert compute_signature({"k": "v"}, "") == ""

    def test_event_type_validation(self, chain: CrossClusterCustodyChain) -> None:
        # _build_event is internal but the public methods exhaust the
        # valid event types, so we exercise the negative path via reach-
        # in to confirm the guard works.
        with pytest.raises(ValueError, match="event_type"):
            chain._build_event(
                job_id="job-z",
                source_cluster="x",
                target_cluster="y",
                parent_event_hash="",
                event_type="BOGUS",
                dispatch_reason="r",
            )

    def test_empty_job_id_rejected(self, chain: CrossClusterCustodyChain) -> None:
        with pytest.raises(ValueError, match="job_id"):
            chain.record_handoff("", "cluster-b", "h", "r")

    def test_empty_target_cluster_rejected(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        with pytest.raises(ValueError, match="target_cluster"):
            chain.record_handoff("job-1", "", "h", "r")


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------
class TestChainVerification:
    def test_empty_chain_is_valid(self, chain: CrossClusterCustodyChain) -> None:
        ok, msg = chain.verify_chain([])
        assert ok is True
        assert "empty" in msg

    def test_single_event_verifies(self, chain: CrossClusterCustodyChain) -> None:
        e = chain.record_handoff("job-1", "cluster-b", "anchor", "reason")
        ok, msg = chain.verify_chain([e])
        assert ok is True
        assert "1" in msg

    def test_happy_path_two_events_chains_correctly(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        e1 = chain.record_handoff("job-1", "cluster-b", "anchor", "reason")
        # The second event's parent_event_hash must be the SHA-256 of
        # e1's signing payload, computed via the same helper.
        prev_hash = chain._sha256_hex(e1.to_signing_payload())
        e2 = chain.record_handoff(
            "job-1", "cluster-c", prev_hash, "rebalance"
        )
        ok, msg = chain.verify_chain([e1, e2])
        assert ok is True, msg
        assert "2" in msg

    def test_broken_parent_hash_detected(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        e1 = chain.record_handoff("job-1", "cluster-b", "anchor", "reason")
        e2 = chain.record_handoff(
            "job-1", "cluster-c", "wrong-hash", "rebalance"
        )
        ok, msg = chain.verify_chain([e1, e2])
        assert ok is False
        assert "parent-hash" in msg

    def test_bad_signature_detected(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        e = chain.record_handoff("job-1", "cluster-b", "anchor", "reason")
        tampered = CrossClusterCustodyEvent(
            job_id=e.job_id,
            source_cluster=e.source_cluster,
            target_cluster=e.target_cluster,
            parent_event_hash=e.parent_event_hash,
            event_type=e.event_type,
            timestamp_utc=e.timestamp_utc,
            dispatch_reason="tampered-reason",  # changes signing payload
            signature=e.signature,
        )
        ok, msg = chain.verify_chain([tampered])
        assert ok is False
        assert "signature" in msg

    def test_unknown_event_type_detected(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        # Bypass _build_event to construct an unsigned bogus event.
        bogus = CrossClusterCustodyEvent(
            job_id="job-x",
            source_cluster="cluster-a",
            target_cluster="cluster-b",
            parent_event_hash="",
            event_type="BOGUS_EVENT",
            timestamp_utc="2026-01-01T00:00:00.000+00:00",
            dispatch_reason="x",
            signature="deadbeef",
        )
        ok, msg = chain.verify_chain([bogus])
        # The event_type guard runs first; signature mismatch would also
        # trigger but the type guard is more specific.
        assert ok is False
        assert "event_type" in msg or "signature" in msg

    def test_rebalance_event_skips_parent_hash_check(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        """JOB_REBALANCED events anchor to a Job row, not a previous hash."""
        e1 = chain.record_handoff("job-1", "cluster-b", "anchor", "reason")
        # A rebalance event with empty parent_event_hash should still
        # verify even though it does not chain to e1.
        e2 = chain.record_rebalance(
            "job-1", "cluster-down", "cluster-a", "cluster_unhealthy"
        )
        ok, msg = chain.verify_chain([e1, e2])
        assert ok is True, msg


# ---------------------------------------------------------------------------
# Replication
# ---------------------------------------------------------------------------
class TestReplication:
    def test_replicate_to_peer_200_ok(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_opener(req: Any, timeout: float | None = None) -> _FakeResponse:
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["data"] = req.data
            return _FakeResponse(200, b'{"status": "ok"}')

        chain._opener = fake_opener
        e = chain.record_handoff("job-1", "cluster-b", "h", "r")
        ok = chain.replicate_to_peer(
            e, "https://peer.example/api/v1/federation/custody/ingest"
        )
        assert ok is True
        # Bearer header includes the configured token.
        auth = next(
            (v for k, v in captured["headers"].items() if k.lower() == "authorization"),
            "",
        )
        assert auth == "Bearer bearer-token-abc"
        body = json.loads(captured["data"].decode("utf-8"))
        assert body["job_id"] == "job-1"
        assert body["signature"] == e.signature

    def test_replicate_to_peer_401_returns_false(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        chain._opener = lambda req, timeout=None: _FakeResponse(401)
        e = chain.record_handoff("job-1", "cluster-b", "h", "r")
        ok = chain.replicate_to_peer(e, "https://peer.example/x")
        assert ok is False

    def test_replicate_to_peer_422_returns_false(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        chain._opener = lambda req, timeout=None: _FakeResponse(422)
        e = chain.record_handoff("job-1", "cluster-b", "h", "r")
        ok = chain.replicate_to_peer(e, "https://peer.example/x")
        assert ok is False

    def test_replicate_to_peer_network_error_returns_false(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        import urllib.error

        def boom(req: Any, timeout: float | None = None) -> Any:
            raise urllib.error.URLError("connection refused")

        chain._opener = boom
        e = chain.record_handoff("job-1", "cluster-b", "h", "r")
        ok = chain.replicate_to_peer(e, "https://peer.example/x")
        assert ok is False

    def test_replicate_to_peer_empty_url_returns_false(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        e = chain.record_handoff("job-1", "cluster-b", "h", "r")
        ok = chain.replicate_to_peer(e, "")
        assert ok is False

    def test_replicate_without_auth_token_omits_header(self) -> None:
        chain = CrossClusterCustodyChain(
            local_cluster="cluster-a",
            hmac_key=HMAC_KEY,
            auth_token="",
            clock=lambda: 1.0,
        )
        captured: dict[str, Any] = {}

        def fake_opener(req: Any, timeout: float | None = None) -> _FakeResponse:
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            return _FakeResponse(200)

        chain._opener = fake_opener
        e = chain.record_handoff("job-1", "cluster-b", "h", "r")
        chain.replicate_to_peer(e, "https://peer.example/x")
        assert "authorization" not in captured["headers"]


# ---------------------------------------------------------------------------
# Reconciler integration via the hook
# ---------------------------------------------------------------------------
class TestReconcilerHook:
    def test_emits_on_rebalance(self, chain: CrossClusterCustodyChain) -> None:
        observations = [
            _RebalanceObservation(
                job_id="job-1",
                source_cluster="cluster-down",
                target_cluster="cluster-a",
                reason="cluster_unhealthy",
            ),
            _RebalanceObservation(
                job_id="job-2",
                source_cluster="cluster-down",
                target_cluster="cluster-a",
                reason="cluster_unhealthy",
            ),
        ]
        hook = CustodyChainReconcilerHook(
            chain, rebalance_provider=lambda: observations
        )
        emitted = hook.tick()
        assert emitted == 2

    def test_no_emit_when_no_observations(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        hook = CustodyChainReconcilerHook(
            chain, rebalance_provider=lambda: []
        )
        assert hook.tick() == 0

    def test_idempotent_across_ticks(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        """Same (job_id, target) pair must not be emitted twice."""
        observations = [
            _RebalanceObservation(
                job_id="job-1",
                source_cluster="cluster-down",
                target_cluster="cluster-a",
                reason="cluster_unhealthy",
            ),
        ]
        hook = CustodyChainReconcilerHook(
            chain, rebalance_provider=lambda: observations
        )
        first = hook.tick()
        second = hook.tick()
        assert first == 1
        assert second == 0

    def test_provider_exception_is_isolated(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        def bad_provider() -> Any:
            raise RuntimeError("DB exploded")

        hook = CustodyChainReconcilerHook(
            chain, rebalance_provider=bad_provider
        )
        # Must not raise.
        assert hook.tick() == 0

    def test_emit_exception_isolated_per_observation(
        self, chain: CrossClusterCustodyChain
    ) -> None:
        observations = [
            _RebalanceObservation(
                job_id="",  # invalid -> ValueError inside record_rebalance
                source_cluster="cluster-down",
                target_cluster="cluster-a",
                reason="x",
            ),
            _RebalanceObservation(
                job_id="job-2",
                source_cluster="cluster-down",
                target_cluster="cluster-a",
                reason="x",
            ),
        ]
        hook = CustodyChainReconcilerHook(
            chain, rebalance_provider=lambda: observations
        )
        # The first observation raises inside record_rebalance and is
        # swallowed; the second still emits.
        assert hook.tick() == 1


# ---------------------------------------------------------------------------
# Factory + env vars
# ---------------------------------------------------------------------------
class TestFactory:
    def test_returns_none_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OCR_FEDERATION_CUSTODY_ENABLED", raising=False)
        assert build_custody_chain_from_env() is None

    def test_returns_none_when_explicitly_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_ENABLED", "false")
        assert build_custody_chain_from_env() is None

    def test_returns_engine_when_fully_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_ENABLED", "true")
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_HMAC_KEY", "key-xyz")
        monkeypatch.setenv("OCR_CLUSTER_NAME", "cluster-a")
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_AUTH_TOKEN", "token-abc")
        engine = build_custody_chain_from_env()
        assert engine is not None
        assert isinstance(engine, CrossClusterCustodyChain)
        assert engine.local_cluster == "cluster-a"

    def test_returns_none_when_hmac_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_ENABLED", "true")
        monkeypatch.delenv("OCR_FEDERATION_CUSTODY_HMAC_KEY", raising=False)
        monkeypatch.setenv("OCR_CLUSTER_NAME", "cluster-a")
        assert build_custody_chain_from_env() is None

    def test_returns_none_when_cluster_name_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_ENABLED", "true")
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_HMAC_KEY", "key-xyz")
        monkeypatch.delenv("OCR_CLUSTER_NAME", raising=False)
        assert build_custody_chain_from_env() is None


# ---------------------------------------------------------------------------
# Insecure mode
# ---------------------------------------------------------------------------
class TestInsecureMode:
    def test_insecure_flag_emits_audit_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="federation.custody"):
            CrossClusterCustodyChain(
                local_cluster="cluster-a",
                hmac_key=HMAC_KEY,
                insecure=True,
            )
        assert any(
            "insecure" in rec.message.lower()
            for rec in caplog.records
        )

    def test_insecure_via_env_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_ENABLED", "true")
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_HMAC_KEY", "k")
        monkeypatch.setenv("OCR_CLUSTER_NAME", "c")
        monkeypatch.setenv("OCR_FEDERATION_CUSTODY_INSECURE", "true")
        engine = build_custody_chain_from_env()
        assert engine is not None
        assert engine._insecure is True

    def test_insecure_default_is_false(self) -> None:
        c = CrossClusterCustodyChain(
            local_cluster="cluster-a", hmac_key=HMAC_KEY
        )
        assert c._insecure is False


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------
class TestConstructorGuards:
    def test_empty_local_cluster_rejected(self) -> None:
        with pytest.raises(ValueError, match="local_cluster"):
            CrossClusterCustodyChain(local_cluster="", hmac_key=HMAC_KEY)

    def test_empty_hmac_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="hmac_key"):
            CrossClusterCustodyChain(local_cluster="cluster-a", hmac_key="")

    def test_valid_event_types_constants(self) -> None:
        assert EVENT_JOB_HANDED_OFF in VALID_EVENT_TYPES
        assert EVENT_JOB_RECEIVED in VALID_EVENT_TYPES
        assert EVENT_JOB_REBALANCED in VALID_EVENT_TYPES
        assert len(VALID_EVENT_TYPES) == 3
