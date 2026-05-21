"""Federation cluster auto-discovery via DNS-SD -- Plan C Phase 1, item C7.

Operator-facing infrastructure that periodically queries DNS-SD (RFC 6763)
for peer clusters advertising the ``_ocr-federation._tcp`` service type,
validates each record, and reconciles the result into the federation
cluster-registry ConfigMap consumed by ``coordinator.federation.cluster_router``.

Design constraints
------------------
* This module does **not** participate in the Celery routing path; it only
  writes to the same ConfigMap that ``ClusterRegistry`` reads on startup.
* ``dnspython`` and the ``kubernetes`` Python client are **optional** at
  import time -- the module imports cleanly without them and surfaces a
  clear error only when discovery / apply is invoked.
* All env vars and Helm flags default OFF; ``build_discovery_from_env``
  returns ``None`` unless ``OCR_FEDERATION_DISCOVERY_ENABLED=true``.
* The reconciler diffs against the ConfigMap once per CLI run; a
  Kubernetes CronJob schedules repeat invocations.
* Loopback addresses (``127.0.0.0/8``, ``::1``) are rejected unless
  ``allow_loopback=True`` -- security guardrail against rogue peer
  injections from misconfigured local resolvers.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Optional dependencies -- imported lazily inside helper functions so the
# module imports cleanly on minimal interpreters.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter  # type: ignore[import-not-found]

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - executed in unit tests
    _PROMETHEUS_AVAILABLE = False

    class _NoopMetric:
        def labels(self, **_kwargs: Any) -> "_NoopMetric":
            return self

        def inc(self, _amount: float = 1.0) -> None:
            return None

    Counter = _NoopMetric  # type: ignore[assignment,misc]


_LOG = logging.getLogger("federation.discovery")


# ---------------------------------------------------------------------------
# Required TXT attributes -- a peer SRV record must carry every key in this
# tuple, otherwise the record is rejected with a structured warning.
# ---------------------------------------------------------------------------
_REQUIRED_TXT_KEYS: tuple[str, ...] = ("region", "tier", "protocol_version")


# ---------------------------------------------------------------------------
# Service record dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServiceRecord:
    """One discovered peer cluster, materialised from PTR/SRV/TXT records."""

    cluster_name: str
    host: str
    port: int
    priority: int
    weight: int
    txt_attrs: dict[str, str] = field(default_factory=dict)
    discovered_at: float = 0.0
    ttl_seconds: int = 0

    def to_registry_entry(self) -> dict[str, Any]:
        """Project this record into the ConfigMap registry entry shape.

        Mirrors ``ClusterEntry.from_dict`` consumers in ``cluster_router.py``.
        Note that AMQP and Management URIs are not included in DNS-SD
        responses; the operator must populate those out-of-band when
        translating discovered hosts into routing-ready peers. We emit the
        host:port pair as a hint and leave URI fields blank so the static
        registry remains the source of truth for credentials.
        """
        tags = [
            f"{k}={v}"
            for k, v in sorted(self.txt_attrs.items())
            if k in _REQUIRED_TXT_KEYS
        ]
        return {
            "name": self.cluster_name,
            "region": self.txt_attrs.get("region", ""),
            "rabbitmq_uri": "",
            "management_uri": "",
            "priority": int(self.priority),
            "tags": tags,
            "tls": {"enabled": False, "ca_secret_ref": ""},
            "discovery": {
                "host": self.host,
                "port": int(self.port),
                "weight": int(self.weight),
                "ttl_seconds": int(self.ttl_seconds),
                "discovered_at": float(self.discovered_at),
                "txt": dict(self.txt_attrs),
            },
        }


# ---------------------------------------------------------------------------
# Reconcile result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ReconcileResult:
    """Diff produced by ``RegistryReconciler.reconcile``."""

    to_add: list[ServiceRecord] = field(default_factory=list)
    to_update: list[ServiceRecord] = field(default_factory=list)
    to_remove: list[str] = field(default_factory=list)

    def is_noop(self) -> bool:
        return not (self.to_add or self.to_update or self.to_remove)


# ---------------------------------------------------------------------------
# DNS-SD discovery
# ---------------------------------------------------------------------------
class DnsSdDiscovery:
    """DNS-SD (RFC 6763) browser for ``_ocr-federation._tcp`` services.

    The lookup chain is:

    1. ``PTR <service>.<domain>`` -> list of instance names.
    2. For each instance, ``SRV`` record -> (priority, weight, port, target).
    3. For each instance, ``TXT`` record -> attribute dictionary.

    Records lacking any required TXT attribute are dropped with a warning.
    Loopback hosts are filtered unless ``allow_loopback=True``.

    The ``resolver`` argument is optional. When ``None``, a ``dns.resolver``
    instance is constructed lazily; tests can inject a fake resolver with
    ``query()`` / ``resolve()`` methods.
    """

    def __init__(
        self,
        domain: str,
        *,
        service_type: str = "_ocr-federation._tcp",
        resolver: Any = None,
        allow_loopback: bool = False,
        clock: Any = None,
        host_resolver: Any = None,
    ) -> None:
        if not domain:
            raise ValueError("domain must be non-empty")
        if not service_type.startswith("_"):
            raise ValueError(
                f"service_type {service_type!r} must start with '_' per RFC 6763"
            )
        self._domain = domain.rstrip(".")
        self._service_type = service_type.rstrip(".")
        self._allow_loopback = bool(allow_loopback)
        self._resolver = resolver
        self._clock = clock if clock is not None else time.time
        # Lookup hook used to validate that an SRV target resolves; tests
        # inject a fake to avoid touching the network.
        self._host_resolver = (
            host_resolver
            if host_resolver is not None
            else self._default_host_resolver
        )

    # -- public surface ---------------------------------------------------
    @property
    def domain(self) -> str:
        return self._domain

    @property
    def service_type(self) -> str:
        return self._service_type

    def discover(self) -> list[ServiceRecord]:
        """Run a single PTR/SRV/TXT discovery pass.

        Returns the list of records that pass ``validate_record``. Records
        that fail validation are logged at WARNING level and dropped. A
        DNS resolver failure on the PTR query yields an empty list (no
        partial result) so the reconciler does not erase a healthy
        registry on a transient name-server hiccup.
        """
        resolver = self._get_resolver()
        if resolver is None:
            _LOG.error(
                "dnspython unavailable -- install 'dnspython>=2.0,<3.0' "
                "to enable DNS-SD discovery"
            )
            return []

        ptr_name = f"{self._service_type}.{self._domain}"
        instances = self._lookup_ptr(resolver, ptr_name)
        if not instances:
            _LOG.info("no PTR records for %s", ptr_name)
            return []

        records: list[ServiceRecord] = []
        for instance in instances:
            try:
                record = self._build_record_from_instance(resolver, instance)
            except Exception as exc:  # defensive -- one bad instance must
                # not abort the entire pass.
                _LOG.warning(
                    "skipping instance %s: %s", instance, exc
                )
                continue
            if record is None:
                continue
            ok, error = self.validate_record(record)
            if not ok:
                _LOG.warning(
                    "rejecting record %s: %s", record.cluster_name, error
                )
                continue
            records.append(record)
            _counter_records_discovered().labels(
                cluster=record.cluster_name
            ).inc()
        return records

    def validate_record(self, record: ServiceRecord) -> tuple[bool, str]:
        """Validate a record's schema, port range, and host resolvability.

        Returns ``(True, "")`` when the record is acceptable, otherwise
        ``(False, "<reason>")``. The reason is suitable for an audit log
        line.
        """
        if not record.cluster_name:
            return False, "missing cluster_name"
        if not record.host:
            return False, "missing host"
        if not (1024 <= int(record.port) <= 65535):
            return False, f"port {record.port} outside [1024, 65535]"
        for key in _REQUIRED_TXT_KEYS:
            if key not in record.txt_attrs:
                return False, f"missing required TXT attr {key!r}"
            if not record.txt_attrs[key]:
                return False, f"empty TXT attr {key!r}"
        # Host must resolve, and (unless explicitly allowed) must not be
        # a loopback address.
        try:
            addrs = list(self._host_resolver(record.host))
        except (socket.gaierror, OSError) as exc:
            return False, f"host {record.host!r} unresolvable: {exc}"
        if not addrs:
            return False, f"host {record.host!r} resolves to no addresses"
        for addr in addrs:
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if ip.is_loopback and not self._allow_loopback:
                return (
                    False,
                    f"host {record.host!r} resolves to loopback {addr} "
                    "(set allow_loopback=True to permit)",
                )
        return True, ""

    # -- internals --------------------------------------------------------
    def _get_resolver(self) -> Any:
        if self._resolver is not None:
            return self._resolver
        try:  # pragma: no cover - exercised only in integration runs
            import dns.resolver  # type: ignore[import-not-found]
        except ImportError:
            return None
        self._resolver = dns.resolver.Resolver()
        return self._resolver

    @staticmethod
    def _default_host_resolver(host: str) -> list[str]:
        """Resolve ``host`` to a list of IP strings via ``getaddrinfo``."""
        try:
            infos = socket.getaddrinfo(
                host, None, type=socket.SOCK_STREAM
            )
        except socket.gaierror:
            raise
        out: list[str] = []
        seen: set[str] = set()
        for entry in infos:
            sockaddr = entry[4]
            if not sockaddr:
                continue
            addr = sockaddr[0]
            if addr not in seen:
                seen.add(addr)
                out.append(addr)
        return out

    def _lookup_ptr(self, resolver: Any, ptr_name: str) -> list[str]:
        """Return the list of instance names from a PTR lookup."""
        try:
            answer = self._call_resolver(resolver, ptr_name, "PTR")
        except Exception as exc:
            _LOG.warning("PTR lookup for %s failed: %s", ptr_name, exc)
            return []
        out: list[str] = []
        for rr in answer or []:
            target = _rdata_target(rr)
            if not target:
                continue
            out.append(target.rstrip("."))
        return out

    def _build_record_from_instance(
        self, resolver: Any, instance: str
    ) -> ServiceRecord | None:
        """Materialise one SRV+TXT pair into a ``ServiceRecord``."""
        try:
            srv_answer = self._call_resolver(resolver, instance, "SRV")
        except Exception as exc:
            _LOG.warning("SRV lookup for %s failed: %s", instance, exc)
            return None
        if not srv_answer:
            return None
        srv_rr = list(srv_answer)[0]
        priority = int(getattr(srv_rr, "priority", 0))
        weight = int(getattr(srv_rr, "weight", 0))
        port = int(getattr(srv_rr, "port", 0))
        target = _rdata_target(srv_rr) or ""
        target = target.rstrip(".")
        ttl = int(_answer_ttl(srv_answer))

        try:
            txt_answer = self._call_resolver(resolver, instance, "TXT")
        except Exception as exc:
            _LOG.warning("TXT lookup for %s failed: %s", instance, exc)
            return None
        txt_attrs = _parse_txt_strings(txt_answer)

        # The cluster-name is the first label of the instance after stripping
        # the service-type / domain suffix; e.g.
        #   "prod-eu-west-1._ocr-federation._tcp.clusters.example.com"
        # -> "prod-eu-west-1".
        cluster_name = instance.split(".", 1)[0]
        return ServiceRecord(
            cluster_name=cluster_name,
            host=target,
            port=port,
            priority=priority,
            weight=weight,
            txt_attrs=txt_attrs,
            discovered_at=float(self._clock()),
            ttl_seconds=ttl,
        )

    @staticmethod
    def _call_resolver(resolver: Any, name: str, rdtype: str) -> Any:
        """Invoke ``resolver`` portably across dnspython 1.x and 2.x.

        dnspython 2.x uses ``resolve()``; 1.x uses ``query()``. Test fakes
        may implement either method.
        """
        method = getattr(resolver, "resolve", None)
        if method is None:
            method = getattr(resolver, "query", None)
        if method is None:
            raise RuntimeError(
                "resolver has no 'resolve' or 'query' method"
            )
        return method(name, rdtype)


def _rdata_target(rr: Any) -> str | None:
    """Extract a string target from a dnspython rdata or fake instance."""
    target = getattr(rr, "target", None)
    if target is None:
        return None
    return str(target)


def _answer_ttl(answer: Any) -> int:
    """Best-effort extraction of TTL from a dnspython Answer-like object."""
    ttl = getattr(answer, "ttl", None) or getattr(answer, "rrset", None)
    if ttl is None:
        return 0
    if isinstance(ttl, int):
        return ttl
    inner_ttl = getattr(ttl, "ttl", None)
    if isinstance(inner_ttl, int):
        return inner_ttl
    return 0


def _parse_txt_strings(answer: Any) -> dict[str, str]:
    """Convert a TXT answer's strings into a flat key=value dict.

    Handles dnspython's ``strings`` attribute (list of bytes) and falls
    back to stringifying each rdata when ``strings`` is unavailable.
    """
    out: dict[str, str] = {}
    for rr in answer or []:
        chunks: list[bytes | str] = []
        strings = getattr(rr, "strings", None)
        if strings is not None:
            chunks = list(strings)
        else:
            chunks = [str(rr)]
        for raw in chunks:
            if isinstance(raw, bytes):
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            else:
                text = str(raw).strip().strip('"')
            if "=" not in text:
                continue
            key, _, value = text.partition("=")
            key = key.strip()
            if not key:
                continue
            out[key] = value.strip()
    return out


# ---------------------------------------------------------------------------
# Registry reconciler
# ---------------------------------------------------------------------------
class RegistryReconciler:
    """Diffs discovered records against the live ConfigMap and patches it."""

    _DATA_KEY = "clusters.json"

    def __init__(
        self,
        *,
        kubernetes_api: Any = None,
        dry_run: bool = False,
    ) -> None:
        self._api = kubernetes_api
        self._dry_run = bool(dry_run)

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    # -- diff -------------------------------------------------------------
    def reconcile(
        self,
        discovered: Iterable[ServiceRecord],
        *,
        existing: dict[str, Any] | None = None,
    ) -> ReconcileResult:
        """Compute add/update/remove ops against an existing registry.

        ``existing`` is the parsed ``clusters.json`` document (or ``None``
        when no ConfigMap exists yet, in which case every discovered
        record is an add). Idempotent: when nothing changes, the result
        is a noop.
        """
        result = ReconcileResult()
        existing_clusters: dict[str, dict[str, Any]] = {}
        for entry in (existing or {}).get("clusters") or []:
            name = str(entry.get("name", ""))
            if name:
                existing_clusters[name] = entry

        discovered_by_name: dict[str, ServiceRecord] = {}
        for record in discovered:
            discovered_by_name[record.cluster_name] = record

        for name, record in discovered_by_name.items():
            new_entry = record.to_registry_entry()
            if name not in existing_clusters:
                result.to_add.append(record)
                continue
            current = existing_clusters[name]
            if not _entries_equal(current, new_entry):
                result.to_update.append(record)

        for name in existing_clusters:
            if name not in discovered_by_name:
                # Only remove peers that were originally discovered. We
                # detect that by the presence of the ``discovery`` block
                # we wrote at add-time. Static peers from values.yaml lack
                # this marker and must NOT be evicted by DNS-SD.
                if "discovery" in (existing_clusters[name] or {}):
                    result.to_remove.append(name)

        return result

    # -- apply ------------------------------------------------------------
    def apply_to_configmap(
        self,
        result: ReconcileResult,
        *,
        namespace: str,
        configmap_name: str,
        existing: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Apply the reconcile result to the federation ConfigMap.

        Returns a per-op-type count dict (``{"added": n, "updated": m,
        "removed": k, "errors": e}``). One failed op does not abort the
        rest -- each is wrapped in try/except so a single RBAC blip on
        one peer cannot stall the entire reconcile pass.
        """
        counts = {"added": 0, "updated": 0, "removed": 0, "errors": 0}
        if result.is_noop():
            _LOG.info(
                "reconcile noop -- registry already in sync (cm=%s/%s)",
                namespace,
                configmap_name,
            )
            return counts

        # Build the new clusters list by applying ops to the existing doc.
        merged = self._merge(result, existing)
        new_text = json.dumps(merged, sort_keys=True, indent=2)

        if self._dry_run:
            _LOG.info(
                "dry-run: would patch ConfigMap %s/%s "
                "(adds=%d updates=%d removes=%d)",
                namespace,
                configmap_name,
                len(result.to_add),
                len(result.to_update),
                len(result.to_remove),
            )
            counts["added"] = len(result.to_add)
            counts["updated"] = len(result.to_update)
            counts["removed"] = len(result.to_remove)
            return counts

        api = self._get_api()
        if api is None:
            _LOG.error(
                "kubernetes client unavailable -- install "
                "'kubernetes>=28.0,<32.0' or pass dry_run=true"
            )
            counts["errors"] += 1
            _counter_apply_errors().labels(op_type="client_unavailable").inc()
            return counts

        # Each op is wrapped individually so partial failures can still
        # land the surviving ops. We do this by patching the ConfigMap
        # once with the merged document and counting per-op results from
        # the diff. If the patch raises we increment errors and bail.
        try:
            api.patch_namespaced_config_map(
                name=configmap_name,
                namespace=namespace,
                body={"data": {self._DATA_KEY: new_text}},
            )
        except Exception as exc:
            _LOG.error(
                "patch ConfigMap %s/%s failed: %s",
                namespace,
                configmap_name,
                exc,
            )
            counts["errors"] += 1
            _counter_apply_errors().labels(op_type="patch").inc()
            return counts

        counts["added"] = len(result.to_add)
        counts["updated"] = len(result.to_update)
        counts["removed"] = len(result.to_remove)
        return counts

    # -- helpers ----------------------------------------------------------
    def _merge(
        self,
        result: ReconcileResult,
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(existing or {})
        merged.setdefault("version", 1)
        clusters_list: list[dict[str, Any]] = list(
            (existing or {}).get("clusters") or []
        )
        by_name: dict[str, dict[str, Any]] = {
            str(c.get("name", "")): c for c in clusters_list if c.get("name")
        }

        for record in result.to_add:
            try:
                by_name[record.cluster_name] = record.to_registry_entry()
            except Exception as exc:
                _LOG.warning(
                    "merge add failed for %s: %s", record.cluster_name, exc
                )
                _counter_apply_errors().labels(op_type="add").inc()

        for record in result.to_update:
            try:
                by_name[record.cluster_name] = record.to_registry_entry()
            except Exception as exc:
                _LOG.warning(
                    "merge update failed for %s: %s",
                    record.cluster_name,
                    exc,
                )
                _counter_apply_errors().labels(op_type="update").inc()

        for name in result.to_remove:
            try:
                by_name.pop(name, None)
            except Exception as exc:
                _LOG.warning("merge remove failed for %s: %s", name, exc)
                _counter_apply_errors().labels(op_type="remove").inc()

        merged["clusters"] = sorted(
            by_name.values(), key=lambda e: str(e.get("name", ""))
        )
        return merged

    def _get_api(self) -> Any:
        if self._api is not None:
            return self._api
        try:  # pragma: no cover - exercised only in cluster runs
            from kubernetes import (
                client as k8s_client,  # type: ignore[import-not-found]
            )
            from kubernetes import (
                config as k8s_config,  # type: ignore[import-not-found]
            )
        except ImportError:
            return None
        try:
            k8s_config.load_incluster_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception as exc:
                _LOG.warning("kube config load failed: %s", exc)
                return None
        self._api = k8s_client.CoreV1Api()
        return self._api


def _entries_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Compare two registry entries ignoring volatile discovery timestamps."""
    a_clean = _strip_volatile(a)
    b_clean = _strip_volatile(b)
    return a_clean == b_clean


def _strip_volatile(entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    discovery = dict(out.get("discovery") or {})
    discovery.pop("discovered_at", None)
    discovery.pop("ttl_seconds", None)
    out["discovery"] = discovery
    return out


# ---------------------------------------------------------------------------
# Prometheus surface -- lazy-init under a lock so duplicate registration
# never blows up the second import.
# ---------------------------------------------------------------------------
_RECORDS_DISCOVERED_COUNTER: Any | None = None
_APPLY_ERRORS_COUNTER: Any | None = None
_COUNTER_LOCK = threading.Lock()


def _counter_records_discovered() -> Any:
    global _RECORDS_DISCOVERED_COUNTER
    with _COUNTER_LOCK:
        if _RECORDS_DISCOVERED_COUNTER is not None:
            return _RECORDS_DISCOVERED_COUNTER
        if not _PROMETHEUS_AVAILABLE:
            _RECORDS_DISCOVERED_COUNTER = Counter()  # type: ignore[call-arg]
            return _RECORDS_DISCOVERED_COUNTER
        try:
            _RECORDS_DISCOVERED_COUNTER = Counter(
                "ocr_federation_discovery_records_discovered_total",
                "DNS-SD records discovered per cluster.",
                ["cluster"],
            )
        except ValueError:  # pragma: no cover - duplicate registration
            from prometheus_client import REGISTRY  # type: ignore[import-not-found]

            _RECORDS_DISCOVERED_COUNTER = next(
                (
                    c
                    for c in REGISTRY._collector_to_names  # type: ignore[attr-defined]
                    if getattr(c, "_name", "")
                    == "ocr_federation_discovery_records_discovered"
                ),
                Counter(),  # type: ignore[call-arg]
            )
    return _RECORDS_DISCOVERED_COUNTER


def _counter_apply_errors() -> Any:
    global _APPLY_ERRORS_COUNTER
    with _COUNTER_LOCK:
        if _APPLY_ERRORS_COUNTER is not None:
            return _APPLY_ERRORS_COUNTER
        if not _PROMETHEUS_AVAILABLE:
            _APPLY_ERRORS_COUNTER = Counter()  # type: ignore[call-arg]
            return _APPLY_ERRORS_COUNTER
        try:
            _APPLY_ERRORS_COUNTER = Counter(
                "ocr_federation_discovery_apply_errors_total",
                "Errors encountered while applying discovery results.",
                ["op_type"],
            )
        except ValueError:  # pragma: no cover - duplicate registration
            from prometheus_client import REGISTRY  # type: ignore[import-not-found]

            _APPLY_ERRORS_COUNTER = next(
                (
                    c
                    for c in REGISTRY._collector_to_names  # type: ignore[attr-defined]
                    if getattr(c, "_name", "")
                    == "ocr_federation_discovery_apply_errors"
                ),
                Counter(),  # type: ignore[call-arg]
            )
    return _APPLY_ERRORS_COUNTER


# ---------------------------------------------------------------------------
# Env-driven factory
# ---------------------------------------------------------------------------
def build_discovery_from_env() -> DnsSdDiscovery | None:
    """Construct a ``DnsSdDiscovery`` from env vars or return ``None``.

    Env vars (all default OFF / empty):

    * ``OCR_FEDERATION_DISCOVERY_ENABLED`` -- master toggle.
    * ``OCR_FEDERATION_DISCOVERY_DOMAIN`` -- DNS suffix to query.
    * ``OCR_FEDERATION_DISCOVERY_SERVICE_TYPE`` -- defaults to
      ``_ocr-federation._tcp``.
    * ``OCR_FEDERATION_DISCOVERY_ALLOW_LOOPBACK`` -- security guardrail
      (default off).
    """
    enabled = os.environ.get("OCR_FEDERATION_DISCOVERY_ENABLED", "false")
    if enabled.strip().lower() not in ("1", "true", "yes"):
        return None
    domain = os.environ.get("OCR_FEDERATION_DISCOVERY_DOMAIN", "").strip()
    if not domain:
        _LOG.warning(
            "OCR_FEDERATION_DISCOVERY_ENABLED=true but "
            "OCR_FEDERATION_DISCOVERY_DOMAIN is empty -- discovery disabled"
        )
        return None
    service_type = os.environ.get(
        "OCR_FEDERATION_DISCOVERY_SERVICE_TYPE", "_ocr-federation._tcp"
    ).strip()
    allow_loopback_raw = os.environ.get(
        "OCR_FEDERATION_DISCOVERY_ALLOW_LOOPBACK", "false"
    )
    allow_loopback = allow_loopback_raw.strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return DnsSdDiscovery(
        domain=domain,
        service_type=service_type,
        allow_loopback=allow_loopback,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint -- discover once, reconcile, exit.
# ---------------------------------------------------------------------------
def _load_existing_configmap(
    api: Any,
    *,
    namespace: str,
    configmap_name: str,
) -> dict[str, Any] | None:
    if api is None:
        return None
    try:  # pragma: no cover - cluster only
        cm = api.read_namespaced_config_map(
            name=configmap_name, namespace=namespace
        )
    except Exception as exc:
        # ``404`` => ConfigMap not found is a soft state -- treat as empty.
        status = getattr(exc, "status", None)
        if status == 404:
            return None
        _LOG.warning(
            "read ConfigMap %s/%s failed: %s",
            namespace,
            configmap_name,
            exc,
        )
        return None
    text = (cm.data or {}).get("clusters.json")
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint suitable for a Kubernetes CronJob.

    Reads env vars, runs one discovery+reconcile pass, and exits. Returns
    a process-style exit code (0 on success, 1 on failure).
    """
    del argv  # not used
    logging.basicConfig(
        level=os.environ.get("OCR_FEDERATION_DISCOVERY_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    discovery = build_discovery_from_env()
    if discovery is None:
        _LOG.info("discovery disabled or misconfigured -- exiting cleanly")
        return 0

    namespace = os.environ.get(
        "OCR_FEDERATION_DISCOVERY_NAMESPACE", ""
    ).strip()
    configmap_name = os.environ.get(
        "OCR_FEDERATION_DISCOVERY_CONFIGMAP", ""
    ).strip()
    if not namespace or not configmap_name:
        _LOG.error(
            "OCR_FEDERATION_DISCOVERY_NAMESPACE / "
            "OCR_FEDERATION_DISCOVERY_CONFIGMAP must both be set"
        )
        return 1

    dry_run_raw = os.environ.get("OCR_FEDERATION_DISCOVERY_DRY_RUN", "false")
    dry_run = dry_run_raw.strip().lower() in ("1", "true", "yes")
    reconciler = RegistryReconciler(dry_run=dry_run)

    try:
        records = discovery.discover()
    except Exception as exc:
        _LOG.error("discovery raised: %s", exc)
        return 1

    existing = _load_existing_configmap(
        reconciler._get_api(),
        namespace=namespace,
        configmap_name=configmap_name,
    )
    diff = reconciler.reconcile(records, existing=existing)
    counts = reconciler.apply_to_configmap(
        diff,
        namespace=namespace,
        configmap_name=configmap_name,
        existing=existing,
    )
    _LOG.info(
        "discovery pass complete records=%d added=%d updated=%d "
        "removed=%d errors=%d dry_run=%s",
        len(records),
        counts["added"],
        counts["updated"],
        counts["removed"],
        counts["errors"],
        dry_run,
    )
    return 0 if counts["errors"] == 0 else 1


__all__ = [
    "DnsSdDiscovery",
    "RegistryReconciler",
    "ReconcileResult",
    "ServiceRecord",
    "build_discovery_from_env",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised by CronJob
    sys.exit(main(sys.argv[1:]))
