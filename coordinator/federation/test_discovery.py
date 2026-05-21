"""Unit tests for the federation DNS-SD discovery module (Plan C Phase 1, item C7).

These tests exercise the module without dnspython or the kubernetes
client installed: a fake resolver feeds synthetic PTR/SRV/TXT answers,
and a fake CoreV1Api stand-in records ConfigMap patches.
"""

from __future__ import annotations

import json
import socket
from typing import Any
from unittest import mock

import pytest

from federation.discovery import (
    DnsSdDiscovery,
    ReconcileResult,
    RegistryReconciler,
    ServiceRecord,
    build_discovery_from_env,
    main,
)


# ---------------------------------------------------------------------------
# Fake DNS infrastructure
# ---------------------------------------------------------------------------
class _FakeRdata:
    """Minimal stand-in for a dnspython rdata record."""

    def __init__(
        self,
        *,
        target: str | None = None,
        priority: int = 0,
        weight: int = 0,
        port: int = 0,
        strings: list[bytes] | None = None,
    ) -> None:
        self.target = target
        self.priority = priority
        self.weight = weight
        self.port = port
        self.strings = strings


class _FakeAnswer:
    def __init__(self, rdatas: list[_FakeRdata], *, ttl: int = 60) -> None:
        self._rdatas = rdatas
        self.ttl = ttl

    def __iter__(self):
        return iter(self._rdatas)


class _FakeResolver:
    """Drives a record map keyed by ``(name, rdtype)``."""

    def __init__(self, records: dict[tuple[str, str], _FakeAnswer]) -> None:
        self._records = records
        self.calls: list[tuple[str, str]] = []

    def resolve(self, name: str, rdtype: str) -> _FakeAnswer:
        key = (str(name).rstrip("."), rdtype.upper())
        self.calls.append(key)
        if key not in self._records:
            raise LookupError(f"NXDOMAIN: {key}")
        return self._records[key]


def _txt(attrs: dict[str, str]) -> _FakeRdata:
    return _FakeRdata(
        strings=[f"{k}={v}".encode("utf-8") for k, v in attrs.items()]
    )


def _build_resolver(
    domain: str = "clusters.example.com",
    *,
    instances: list[tuple[str, dict[str, Any]]] | None = None,
) -> _FakeResolver:
    """Build a fake resolver with one or more instances.

    Each instance entry is ``(short_name, {host, port, priority, weight, txt})``.
    """
    instances = instances or [
        (
            "prod-eu-west-1",
            {
                "host": "rabbit.eu-west-1.example.com",
                "port": 5671,
                "priority": 10,
                "weight": 5,
                "txt": {
                    "region": "eu-west-1",
                    "tier": "core",
                    "protocol_version": "1",
                },
            },
        )
    ]
    service = "_ocr-federation._tcp"
    ptr_name = f"{service}.{domain}"
    records: dict[tuple[str, str], _FakeAnswer] = {}

    ptr_rdatas = []
    for short, _ in instances:
        full = f"{short}.{ptr_name}"
        ptr_rdatas.append(_FakeRdata(target=full + "."))
    records[(ptr_name, "PTR")] = _FakeAnswer(ptr_rdatas, ttl=300)

    for short, info in instances:
        full = f"{short}.{ptr_name}"
        records[(full, "SRV")] = _FakeAnswer(
            [
                _FakeRdata(
                    target=info["host"] + ".",
                    priority=int(info.get("priority", 0)),
                    weight=int(info.get("weight", 0)),
                    port=int(info.get("port", 5671)),
                )
            ],
            ttl=120,
        )
        records[(full, "TXT")] = _FakeAnswer(
            [_txt(info.get("txt", {}))], ttl=120
        )

    return _FakeResolver(records)


def _ok_host_resolver(host: str) -> list[str]:
    """Stand-in host resolver that always returns a non-loopback IP."""
    return ["10.0.0.1"]


def _loopback_host_resolver(host: str) -> list[str]:
    return ["127.0.0.1"]


def _broken_host_resolver(host: str) -> list[str]:
    raise socket.gaierror("name or service not known")


# ---------------------------------------------------------------------------
# DnsSdDiscovery.discover -- happy path
# ---------------------------------------------------------------------------
class TestDiscoverHappyPath:
    def test_single_instance_discovered(self):
        resolver = _build_resolver()
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_ok_host_resolver,
        )
        records = d.discover()
        assert len(records) == 1
        rec = records[0]
        assert rec.cluster_name == "prod-eu-west-1"
        assert rec.host == "rabbit.eu-west-1.example.com"
        assert rec.port == 5671
        assert rec.priority == 10
        assert rec.weight == 5
        assert rec.txt_attrs["region"] == "eu-west-1"
        assert rec.txt_attrs["tier"] == "core"
        assert rec.txt_attrs["protocol_version"] == "1"

    def test_multiple_instances_discovered(self):
        instances = [
            (
                "us-east",
                {
                    "host": "rabbit.us-east.example.com",
                    "port": 5671,
                    "priority": 10,
                    "weight": 1,
                    "txt": {
                        "region": "us-east-1",
                        "tier": "core",
                        "protocol_version": "1",
                    },
                },
            ),
            (
                "eu-west",
                {
                    "host": "rabbit.eu-west.example.com",
                    "port": 5671,
                    "priority": 20,
                    "weight": 1,
                    "txt": {
                        "region": "eu-west-1",
                        "tier": "core",
                        "protocol_version": "1",
                    },
                },
            ),
        ]
        resolver = _build_resolver(instances=instances)
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_ok_host_resolver,
        )
        records = d.discover()
        names = sorted(r.cluster_name for r in records)
        assert names == ["eu-west", "us-east"]

    def test_record_includes_ttl_and_timestamp(self):
        resolver = _build_resolver()
        clock = mock.Mock(return_value=1234.5)
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_ok_host_resolver,
            clock=clock,
        )
        records = d.discover()
        assert records[0].discovered_at == 1234.5
        # TTL plumbed through from the SRV answer.
        assert records[0].ttl_seconds == 120


# ---------------------------------------------------------------------------
# DnsSdDiscovery.discover -- rejection paths
# ---------------------------------------------------------------------------
class TestDiscoverRejections:
    def test_missing_required_txt_attr_drops_record(self):
        resolver = _build_resolver(
            instances=[
                (
                    "bad",
                    {
                        "host": "rabbit.bad.example.com",
                        "port": 5671,
                        "priority": 10,
                        "weight": 1,
                        "txt": {
                            # Missing 'protocol_version'
                            "region": "eu-west-1",
                            "tier": "core",
                        },
                    },
                )
            ]
        )
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_ok_host_resolver,
        )
        assert d.discover() == []

    def test_empty_required_txt_attr_drops_record(self):
        resolver = _build_resolver(
            instances=[
                (
                    "bad",
                    {
                        "host": "rabbit.bad.example.com",
                        "port": 5671,
                        "priority": 10,
                        "weight": 1,
                        "txt": {
                            "region": "",
                            "tier": "core",
                            "protocol_version": "1",
                        },
                    },
                )
            ]
        )
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_ok_host_resolver,
        )
        assert d.discover() == []

    def test_port_below_1024_rejected(self):
        resolver = _build_resolver(
            instances=[
                (
                    "low-port",
                    {
                        "host": "rabbit.example.com",
                        "port": 80,
                        "priority": 10,
                        "weight": 1,
                        "txt": {
                            "region": "eu-west-1",
                            "tier": "core",
                            "protocol_version": "1",
                        },
                    },
                )
            ]
        )
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_ok_host_resolver,
        )
        assert d.discover() == []

    def test_port_above_65535_rejected(self):
        rec = ServiceRecord(
            cluster_name="x",
            host="ok.example.com",
            port=70000,
            priority=10,
            weight=1,
            txt_attrs={
                "region": "r",
                "tier": "t",
                "protocol_version": "1",
            },
        )
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=_build_resolver(),
            host_resolver=_ok_host_resolver,
        )
        ok, err = d.validate_record(rec)
        assert ok is False
        assert "outside" in err

    def test_unresolvable_host_rejected(self):
        resolver = _build_resolver()
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_broken_host_resolver,
        )
        assert d.discover() == []

    def test_loopback_host_rejected_by_default(self):
        resolver = _build_resolver()
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_loopback_host_resolver,
        )
        assert d.discover() == []

    def test_loopback_host_allowed_with_flag(self):
        resolver = _build_resolver()
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_loopback_host_resolver,
            allow_loopback=True,
        )
        assert len(d.discover()) == 1

    def test_no_ptr_records_returns_empty_list(self):
        # Resolver with no records at all.
        resolver = _FakeResolver({})
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=resolver,
            host_resolver=_ok_host_resolver,
        )
        assert d.discover() == []

    def test_resolver_unavailable_returns_empty_list(self):
        # No resolver and dnspython not actually present in test env -- the
        # builder lazy-imports and falls back to None.
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=None,
            host_resolver=_ok_host_resolver,
        )
        # Force the resolver to remain None even if dnspython is installed.
        with mock.patch.object(d, "_get_resolver", return_value=None):
            assert d.discover() == []


class TestValidateRecord:
    def test_missing_cluster_name(self):
        rec = ServiceRecord(
            cluster_name="",
            host="x.example.com",
            port=5671,
            priority=10,
            weight=1,
            txt_attrs={
                "region": "r",
                "tier": "t",
                "protocol_version": "1",
            },
        )
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=_build_resolver(),
            host_resolver=_ok_host_resolver,
        )
        ok, err = d.validate_record(rec)
        assert ok is False
        assert "cluster_name" in err

    def test_missing_host(self):
        rec = ServiceRecord(
            cluster_name="x",
            host="",
            port=5671,
            priority=10,
            weight=1,
            txt_attrs={
                "region": "r",
                "tier": "t",
                "protocol_version": "1",
            },
        )
        d = DnsSdDiscovery(
            "clusters.example.com",
            resolver=_build_resolver(),
            host_resolver=_ok_host_resolver,
        )
        ok, err = d.validate_record(rec)
        assert ok is False
        assert "host" in err

    def test_constructor_rejects_empty_domain(self):
        with pytest.raises(ValueError):
            DnsSdDiscovery("")

    def test_constructor_rejects_bad_service_type(self):
        with pytest.raises(ValueError):
            DnsSdDiscovery(
                "clusters.example.com", service_type="ocr-federation._tcp"
            )


# ---------------------------------------------------------------------------
# Reconciler diff
# ---------------------------------------------------------------------------
def _record(
    name: str,
    *,
    host: str | None = None,
    port: int = 5671,
    priority: int = 10,
    region: str = "eu-west-1",
) -> ServiceRecord:
    return ServiceRecord(
        cluster_name=name,
        host=host or f"rabbit.{name}.example.com",
        port=port,
        priority=priority,
        weight=1,
        txt_attrs={
            "region": region,
            "tier": "core",
            "protocol_version": "1",
        },
        discovered_at=0.0,
        ttl_seconds=60,
    )


class TestReconcileDiff:
    def test_empty_existing_all_records_added(self):
        rec_a = _record("a")
        rec_b = _record("b")
        r = RegistryReconciler()
        result = r.reconcile([rec_a, rec_b], existing=None)
        assert {x.cluster_name for x in result.to_add} == {"a", "b"}
        assert result.to_update == []
        assert result.to_remove == []

    def test_idempotent_when_no_change(self):
        rec_a = _record("a")
        existing = {
            "version": 1,
            "clusters": [rec_a.to_registry_entry()],
        }
        r = RegistryReconciler()
        result = r.reconcile([rec_a], existing=existing)
        assert result.is_noop()

    def test_changed_record_shows_as_update(self):
        rec_old = _record("a", port=5671)
        rec_new = _record("a", port=5672)
        existing = {
            "version": 1,
            "clusters": [rec_old.to_registry_entry()],
        }
        r = RegistryReconciler()
        result = r.reconcile([rec_new], existing=existing)
        assert result.to_add == []
        assert [x.cluster_name for x in result.to_update] == ["a"]
        assert result.to_remove == []

    def test_missing_discovered_peer_is_removed(self):
        rec_a = _record("a")
        existing = {
            "version": 1,
            "clusters": [rec_a.to_registry_entry()],
        }
        r = RegistryReconciler()
        result = r.reconcile([], existing=existing)
        assert result.to_remove == ["a"]

    def test_static_peer_without_discovery_marker_is_kept(self):
        # An entry from values.yaml has no 'discovery' block; it must NOT
        # be removed by a DNS-SD pass that does not see it.
        existing = {
            "version": 1,
            "clusters": [
                {
                    "name": "static-peer",
                    "region": "us-east-1",
                    "rabbitmq_uri": "amqps://...",
                    "management_uri": "https://...",
                    "priority": 5,
                    "tags": [],
                    "tls": {"enabled": False, "ca_secret_ref": ""},
                }
            ],
        }
        r = RegistryReconciler()
        result = r.reconcile([], existing=existing)
        assert result.to_remove == []

    def test_volatile_timestamp_change_does_not_force_update(self):
        rec = _record("a")
        old_entry = rec.to_registry_entry()
        # Simulate an older timestamp on disk.
        old_entry["discovery"]["discovered_at"] = 100.0
        existing = {"version": 1, "clusters": [old_entry]}
        # New record has different timestamp but same logical content.
        new_rec = ServiceRecord(
            cluster_name=rec.cluster_name,
            host=rec.host,
            port=rec.port,
            priority=rec.priority,
            weight=rec.weight,
            txt_attrs=dict(rec.txt_attrs),
            discovered_at=999.0,
            ttl_seconds=60,
        )
        r = RegistryReconciler()
        result = r.reconcile([new_rec], existing=existing)
        assert result.is_noop()


# ---------------------------------------------------------------------------
# Reconciler apply
# ---------------------------------------------------------------------------
class _FakeApi:
    """In-memory CoreV1Api that records patches."""

    def __init__(
        self,
        *,
        existing_data: str | None = None,
        raise_on_patch: Exception | None = None,
        raise_on_read: Exception | None = None,
    ) -> None:
        self._existing_data = existing_data
        self._raise_on_patch = raise_on_patch
        self._raise_on_read = raise_on_read
        self.patches: list[dict[str, Any]] = []

    def read_namespaced_config_map(
        self, *, name: str, namespace: str
    ) -> Any:
        if self._raise_on_read:
            raise self._raise_on_read
        cm = mock.Mock()
        cm.data = (
            {"clusters.json": self._existing_data}
            if self._existing_data is not None
            else {}
        )
        return cm

    def patch_namespaced_config_map(
        self, *, name: str, namespace: str, body: dict[str, Any]
    ) -> Any:
        if self._raise_on_patch:
            raise self._raise_on_patch
        self.patches.append(
            {"name": name, "namespace": namespace, "body": body}
        )
        return mock.Mock()


class TestReconcilerApply:
    def test_dry_run_does_not_call_api(self):
        rec = _record("a")
        result = ReconcileResult(to_add=[rec])
        api = _FakeApi()
        r = RegistryReconciler(kubernetes_api=api, dry_run=True)
        counts = r.apply_to_configmap(
            result,
            namespace="ocr-local",
            configmap_name="federation-clusters",
        )
        assert api.patches == []
        assert counts["added"] == 1

    def test_noop_returns_zero_counts(self):
        result = ReconcileResult()
        api = _FakeApi()
        r = RegistryReconciler(kubernetes_api=api)
        counts = r.apply_to_configmap(
            result,
            namespace="ocr-local",
            configmap_name="federation-clusters",
        )
        assert counts == {
            "added": 0,
            "updated": 0,
            "removed": 0,
            "errors": 0,
        }
        assert api.patches == []

    def test_add_writes_patch(self):
        rec = _record("a")
        result = ReconcileResult(to_add=[rec])
        api = _FakeApi()
        r = RegistryReconciler(kubernetes_api=api)
        counts = r.apply_to_configmap(
            result,
            namespace="ocr-local",
            configmap_name="federation-clusters",
        )
        assert counts["added"] == 1
        assert len(api.patches) == 1
        body = api.patches[0]["body"]
        text = body["data"]["clusters.json"]
        parsed = json.loads(text)
        assert any(c.get("name") == "a" for c in parsed["clusters"])

    def test_remove_writes_patch_without_peer(self):
        rec_a = _record("a")
        existing = {"version": 1, "clusters": [rec_a.to_registry_entry()]}
        result = ReconcileResult(to_remove=["a"])
        api = _FakeApi()
        r = RegistryReconciler(kubernetes_api=api)
        counts = r.apply_to_configmap(
            result,
            namespace="ocr-local",
            configmap_name="federation-clusters",
            existing=existing,
        )
        assert counts["removed"] == 1
        body = api.patches[0]["body"]
        text = body["data"]["clusters.json"]
        parsed = json.loads(text)
        assert all(c.get("name") != "a" for c in parsed.get("clusters", []))

    def test_patch_failure_increments_errors(self):
        rec = _record("a")
        result = ReconcileResult(to_add=[rec])
        api = _FakeApi(raise_on_patch=RuntimeError("rbac denied"))
        r = RegistryReconciler(kubernetes_api=api)
        counts = r.apply_to_configmap(
            result,
            namespace="ocr-local",
            configmap_name="federation-clusters",
        )
        assert counts["errors"] == 1
        assert counts["added"] == 0

    def test_kubernetes_unavailable_increments_errors(self):
        rec = _record("a")
        result = ReconcileResult(to_add=[rec])
        # No api passed and lazy import returns None.
        r = RegistryReconciler(kubernetes_api=None, dry_run=False)
        with mock.patch.object(r, "_get_api", return_value=None):
            counts = r.apply_to_configmap(
                result,
                namespace="ocr-local",
                configmap_name="federation-clusters",
            )
        assert counts["errors"] == 1


# ---------------------------------------------------------------------------
# build_discovery_from_env
# ---------------------------------------------------------------------------
class TestBuildDiscoveryFromEnv:
    def test_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.delenv("OCR_FEDERATION_DISCOVERY_ENABLED", raising=False)
        assert build_discovery_from_env() is None

    def test_returns_none_when_explicit_false(self, monkeypatch):
        monkeypatch.setenv("OCR_FEDERATION_DISCOVERY_ENABLED", "false")
        assert build_discovery_from_env() is None

    def test_returns_none_when_domain_missing(self, monkeypatch):
        monkeypatch.setenv("OCR_FEDERATION_DISCOVERY_ENABLED", "true")
        monkeypatch.delenv(
            "OCR_FEDERATION_DISCOVERY_DOMAIN", raising=False
        )
        assert build_discovery_from_env() is None

    def test_returns_discovery_when_enabled_and_domain_set(self, monkeypatch):
        monkeypatch.setenv("OCR_FEDERATION_DISCOVERY_ENABLED", "true")
        monkeypatch.setenv(
            "OCR_FEDERATION_DISCOVERY_DOMAIN", "clusters.example.com"
        )
        d = build_discovery_from_env()
        assert d is not None
        assert d.domain == "clusters.example.com"
        assert d.service_type == "_ocr-federation._tcp"

    def test_allow_loopback_propagates(self, monkeypatch):
        monkeypatch.setenv("OCR_FEDERATION_DISCOVERY_ENABLED", "1")
        monkeypatch.setenv(
            "OCR_FEDERATION_DISCOVERY_DOMAIN", "clusters.example.com"
        )
        monkeypatch.setenv(
            "OCR_FEDERATION_DISCOVERY_ALLOW_LOOPBACK", "true"
        )
        d = build_discovery_from_env()
        assert d is not None
        assert d._allow_loopback is True


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
class TestCli:
    def test_main_returns_zero_when_disabled(self, monkeypatch):
        monkeypatch.delenv("OCR_FEDERATION_DISCOVERY_ENABLED", raising=False)
        assert main(None) == 0

    def test_main_requires_namespace_and_configmap(self, monkeypatch):
        monkeypatch.setenv("OCR_FEDERATION_DISCOVERY_ENABLED", "true")
        monkeypatch.setenv(
            "OCR_FEDERATION_DISCOVERY_DOMAIN", "clusters.example.com"
        )
        monkeypatch.delenv(
            "OCR_FEDERATION_DISCOVERY_NAMESPACE", raising=False
        )
        monkeypatch.delenv(
            "OCR_FEDERATION_DISCOVERY_CONFIGMAP", raising=False
        )
        assert main(None) == 1

    def test_main_runs_dry_run_pass(self, monkeypatch):
        # Configure env for a full dry-run pass; no kubernetes calls happen.
        monkeypatch.setenv("OCR_FEDERATION_DISCOVERY_ENABLED", "true")
        monkeypatch.setenv(
            "OCR_FEDERATION_DISCOVERY_DOMAIN", "clusters.example.com"
        )
        monkeypatch.setenv(
            "OCR_FEDERATION_DISCOVERY_NAMESPACE", "ocr-local"
        )
        monkeypatch.setenv(
            "OCR_FEDERATION_DISCOVERY_CONFIGMAP", "federation-clusters"
        )
        monkeypatch.setenv("OCR_FEDERATION_DISCOVERY_DRY_RUN", "true")

        # Patch DnsSdDiscovery so the CLI does not need a real resolver.
        with mock.patch(
            "federation.discovery.build_discovery_from_env",
            return_value=mock.Mock(
                discover=mock.Mock(return_value=[_record("a")])
            ),
        ):
            with mock.patch(
                "federation.discovery.RegistryReconciler._get_api",
                return_value=None,
            ):
                rc = main(None)
        assert rc == 0


# ---------------------------------------------------------------------------
# ServiceRecord.to_registry_entry
# ---------------------------------------------------------------------------
class TestServiceRecord:
    def test_to_registry_entry_includes_discovery_marker(self):
        rec = _record("a")
        entry = rec.to_registry_entry()
        assert entry["name"] == "a"
        assert entry["region"] == "eu-west-1"
        assert "discovery" in entry
        assert entry["discovery"]["host"] == "rabbit.a.example.com"
        assert entry["discovery"]["port"] == 5671

    def test_to_registry_entry_tags_sorted(self):
        rec = _record("a")
        entry = rec.to_registry_entry()
        assert entry["tags"] == sorted(entry["tags"])

    def test_record_dataclass_immutable(self):
        rec = _record("a")
        with pytest.raises((AttributeError, Exception)):
            rec.cluster_name = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Prometheus counters
# ---------------------------------------------------------------------------
class TestPrometheusCounters:
    def test_counters_lazy_init_does_not_raise(self):
        from federation.discovery import (
            _counter_apply_errors,
            _counter_records_discovered,
        )

        # Calling twice exercises the cache path.
        c1 = _counter_records_discovered()
        c2 = _counter_records_discovered()
        assert c1 is c2

        e1 = _counter_apply_errors()
        e2 = _counter_apply_errors()
        assert e1 is e2

    def test_counter_inc_is_safe(self):
        from federation.discovery import _counter_records_discovered

        c = _counter_records_discovered()
        # Both real and noop counters expose labels(...).inc().
        c.labels(cluster="x").inc()


# ---------------------------------------------------------------------------
# TXT parsing edge cases
# ---------------------------------------------------------------------------
class TestTxtParsing:
    def test_parses_string_chunks(self):
        from federation.discovery import _parse_txt_strings

        ans = _FakeAnswer(
            [_FakeRdata(strings=[b"region=us-east-1", b"tier=core"])]
        )
        out = _parse_txt_strings(ans)
        assert out == {"region": "us-east-1", "tier": "core"}

    def test_skips_chunks_without_equals(self):
        from federation.discovery import _parse_txt_strings

        ans = _FakeAnswer([_FakeRdata(strings=[b"noeq", b"k=v"])])
        out = _parse_txt_strings(ans)
        assert out == {"k": "v"}

    def test_handles_undecodable_bytes(self):
        from federation.discovery import _parse_txt_strings

        ans = _FakeAnswer(
            [_FakeRdata(strings=[b"\xff\xfe\xfd", b"good=value"])]
        )
        out = _parse_txt_strings(ans)
        assert out == {"good": "value"}
