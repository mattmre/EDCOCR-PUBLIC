"""Federation reconciler -- Plan C Phase 1, item C2.

Watches a ConfigMap-backed cluster registry, probes each peer's RabbitMQ
Management API for liveness, and adds/removes per-peer federation Policy
resources accordingly. Exposes Prometheus metrics on a configurable port
(default 9100).

Design constraints:
* Standard library HTTP -- no aiohttp / httpx dependency.
* The ``kubernetes`` Python client is **optional**: when missing, the
  reconciler logs a structured error and exits cleanly. This keeps unit
  tests dependency-free.
* All Prometheus metric registration goes through ``prometheus_client``
  when available; a no-op shim is used otherwise so unit tests pass on
  hosts without it.
* All log lines are structured JSON for ingestion by the cluster log shipper.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Optional dependencies -- imports are guarded so unit tests can run on a
# minimal interpreter and the reconciler can fail loudly (but cleanly) when
# deployed without the kubernetes client baked in.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised in container environments only
    from kubernetes import client as k8s_client  # type: ignore[import-not-found]
    from kubernetes import config as k8s_config  # type: ignore[import-not-found]

    _KUBERNETES_AVAILABLE = True
except ImportError:  # pragma: no cover - executed in unit tests
    k8s_client = None  # type: ignore[assignment]
    k8s_config = None  # type: ignore[assignment]
    _KUBERNETES_AVAILABLE = False

try:
    from prometheus_client import (  # type: ignore[import-not-found]
        CollectorRegistry,
        Counter,
        Gauge,
        start_http_server,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - executed in unit tests
    _PROMETHEUS_AVAILABLE = False

    class _NoopMetric:
        def labels(self, **_kwargs: Any) -> "_NoopMetric":
            return self

        def set(self, _value: float) -> None:
            return None

        def inc(self, _amount: float = 1.0) -> None:
            return None

    Gauge = Counter = _NoopMetric  # type: ignore[assignment,misc]
    CollectorRegistry = object  # type: ignore[assignment,misc]

    def start_http_server(*_args: Any, **_kwargs: Any) -> None:  # type: ignore[no-redef]
        return None


# ---------------------------------------------------------------------------
# Structured JSON logger
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, sort_keys=True)


def _build_logger(name: str = "federation.reconciler") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("FEDERATION_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


_LOG = _build_logger()


def _log(level: int, msg: str, **fields: Any) -> None:
    """Emit a structured log line; ``fields`` is folded into the JSON payload."""
    _LOG.log(level, msg, extra={"extra_fields": fields})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeerCluster:
    """Single peer entry parsed from the registry ConfigMap."""

    name: str
    region: str
    rabbitmq_uri: str
    management_uri: str
    priority: int
    tags: tuple[str, ...]
    tls_enabled: bool
    tls_ca_secret_ref: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeerCluster":
        tls = data.get("tls") or {}
        return cls(
            name=str(data["name"]),
            region=str(data.get("region", "")),
            rabbitmq_uri=str(data.get("rabbitmq_uri", "")),
            management_uri=str(data.get("management_uri", "")),
            priority=int(data.get("priority", 10)),
            tags=tuple(str(t) for t in (data.get("tags") or [])),
            tls_enabled=bool(tls.get("enabled", False)),
            tls_ca_secret_ref=str(tls.get("ca_secret_ref", "")),
        )


@dataclass
class Registry:
    """Parsed federation cluster-registry document."""

    self_name: str
    queue_classes: tuple[str, ...]
    healthcheck_interval_seconds: int
    unhealthy_threshold_seconds: int
    clusters: list[PeerCluster] = field(default_factory=list)

    @classmethod
    def from_json(cls, text: str) -> "Registry":
        data = json.loads(text)
        self_block = data.get("self") or {}
        return cls(
            self_name=str(self_block.get("name", "")),
            queue_classes=tuple(str(q) for q in (data.get("queue_classes") or [])),
            healthcheck_interval_seconds=int(
                data.get("healthcheck_interval_seconds", 30)
            ),
            unhealthy_threshold_seconds=int(
                data.get("unhealthy_threshold_seconds", 120)
            ),
            clusters=[
                PeerCluster.from_dict(c) for c in (data.get("clusters") or [])
            ],
        )


@dataclass
class PeerHealth:
    """Tracks consecutive-fail timing for one peer.

    ``last_success_ts`` defaults to ``None`` rather than ``0.0`` so that the
    "we have never seen this peer up" state is unambiguous (a literal ``0.0``
    would be falsy, which previously short-circuited the unhealthy threshold
    check on the very first reconcile cycle).
    """

    healthy: bool = True
    last_success_ts: float | None = None
    last_check_ts: float = 0.0
    consecutive_failures: int = 0


# ---------------------------------------------------------------------------
# Health probing -- pure stdlib so this is trivially mockable.
# ---------------------------------------------------------------------------
def _probe_management_api(
    management_uri: str,
    *,
    timeout: float = 5.0,
    opener: Any = None,
) -> tuple[bool, dict[str, Any]]:
    """Probe RabbitMQ Management API at ``/api/aliveness-test/%2F``.

    Returns ``(is_healthy, payload)``. Network failures and non-2xx responses
    map to ``(False, {})``. The ``opener`` argument exists for tests that
    inject a fake URL opener; production passes ``None`` and uses
    :func:`urllib.request.urlopen`.
    """
    if not management_uri:
        return False, {}
    base = management_uri.rstrip("/")
    url = f"{base}/api/aliveness-test/%2F"
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "application/json")
    open_fn = opener if opener is not None else urllib.request.urlopen
    try:
        with open_fn(request, timeout=timeout) as response:  # type: ignore[arg-type]
            status = getattr(response, "status", None) or response.getcode()
            body = response.read()
        if 200 <= int(status) < 300:
            try:
                payload = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                payload = {}
            return True, payload
        return False, {}
    except (urllib.error.URLError, OSError, TimeoutError):
        return False, {}


def _probe_queue_metrics(
    management_uri: str,
    queue: str,
    *,
    timeout: float = 5.0,
    opener: Any = None,
) -> dict[str, Any]:
    """Best-effort fetch of ``messages_unacknowledged`` for a single queue."""
    if not management_uri:
        return {}
    base = management_uri.rstrip("/")
    encoded = urllib.parse.quote(queue, safe="")
    url = f"{base}/api/queues/%2F/{encoded}"
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "application/json")
    open_fn = opener if opener is not None else urllib.request.urlopen
    try:
        with open_fn(request, timeout=timeout) as response:  # type: ignore[arg-type]
            status = getattr(response, "status", None) or response.getcode()
            body = response.read()
        if 200 <= int(status) < 300:
            try:
                return json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}
    except (urllib.error.URLError, OSError, TimeoutError):
        return {}
    return {}


# ---------------------------------------------------------------------------
# Policy reconciliation
# ---------------------------------------------------------------------------
def _policy_name(release: str, peer: str, queue_class: str) -> str:
    safe_qc = queue_class.replace("_", "-")
    return f"{release}-fed-policy-{peer}-{safe_qc}"


def _policy_body(
    *,
    release: str,
    peer: str,
    queue_class: str,
    priority: int,
    namespace: str,
    max_overflow: int,
) -> dict[str, Any]:
    """Build the RabbitMQ Policy CR body for one (peer, queue-class) pair."""
    name = _policy_name(release, peer, queue_class)
    return {
        "apiVersion": "rabbitmq.com/v1beta1",
        "kind": "Policy",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/component": "federation",
                "app.kubernetes.io/managed-by": "federation-reconciler",
                "federation.ocr-local.io/peer": peer,
                "federation.ocr-local.io/queue-class": queue_class,
            },
        },
        "spec": {
            "name": name,
            "vhost": "/",
            "pattern": f"^{queue_class}.*$",
            "applyTo": "queues",
            "priority": int(priority),
            "definition": {
                "federation-upstream": f"{peer}-upstream",
                "max-length": int(max_overflow),
                "overflow": "reject-publish",
            },
            "rabbitmqClusterReference": {
                "name": f"{release}-rabbitmq",
            },
        },
    }


# ---------------------------------------------------------------------------
# Prometheus metrics surface
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    peer_up: Any
    messages_in_total: Any
    messages_out_total: Any
    lag_seconds: Any

    @classmethod
    def build(cls, registry: Any | None = None) -> "Metrics":
        if not _PROMETHEUS_AVAILABLE:  # pragma: no cover - tested via env
            shim = Gauge()
            return cls(shim, shim, shim, shim)
        kwargs = {"registry": registry} if registry is not None else {}
        return cls(
            peer_up=Gauge(
                "ocr_rabbitmq_federation_peer_up",
                "Federation peer cluster liveness (1=healthy, 0=down).",
                ["cluster"],
                **kwargs,
            ),
            messages_in_total=Counter(
                "ocr_rabbitmq_federation_messages_in_total",
                "Messages flowing into the local cluster from a peer (best-effort).",
                ["cluster", "queue"],
                **kwargs,
            ),
            messages_out_total=Counter(
                "ocr_rabbitmq_federation_messages_out_total",
                "Messages flowing out of the local cluster towards a peer.",
                ["cluster", "queue"],
                **kwargs,
            ),
            lag_seconds=Gauge(
                "ocr_rabbitmq_federation_lag_seconds",
                "Approximate replication lag in seconds (placeholder, derived "
                "from peer messages_unacknowledged).",
                ["cluster", "queue"],
                **kwargs,
            ),
        )


# ---------------------------------------------------------------------------
# Reconciler core
# ---------------------------------------------------------------------------
class Reconciler:
    """Stateful reconcile loop driven by a registry document.

    The class is split from the entry-point so unit tests can drive a single
    reconcile cycle deterministically.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        release_name: str,
        namespace: str,
        max_overflow_messages: int = 10000,
        metrics: Metrics | None = None,
        opener: Any = None,
        kubernetes_api: Any = None,
        failover_engine: Any = None,
        custody_chain_engine: Any = None,
    ) -> None:
        self.registry = registry
        self.release_name = release_name
        self.namespace = namespace
        self.max_overflow_messages = max_overflow_messages
        self.metrics = metrics or Metrics.build()
        self._opener = opener
        self._kubernetes_api = kubernetes_api
        # Plan C Phase 1, item C5 -- optional failover engine.  When None
        # (the default), the reconciler behaves exactly as before; when
        # provided, ``reconcile_once`` invokes ``failover_engine.tick()``
        # after policy add/remove logic on each cycle.
        self.failover_engine = failover_engine
        # Plan C Phase 1, item C6 -- optional cross-cluster custody-chain
        # hook.  When None, behaviour is unchanged.  When provided, the
        # reconciler calls ``custody_chain_engine.tick()`` after the
        # failover engine on each cycle.  Custody emission is wrapped in
        # try/except so it can never block reconciliation.
        self.custody_chain_engine = custody_chain_engine
        self._peer_health: dict[str, PeerHealth] = {
            c.name: PeerHealth() for c in registry.clusters
        }
        self._installed_policies: set[str] = set()

    # -- public surface ---------------------------------------------------
    def reconcile_once(self, *, now: float | None = None) -> dict[str, PeerHealth]:
        """Run a single reconcile cycle and return the updated health map."""
        ts = float(now if now is not None else time.time())
        for peer in self.registry.clusters:
            healthy, payload = _probe_management_api(
                peer.management_uri, opener=self._opener
            )
            health = self._peer_health.setdefault(peer.name, PeerHealth())
            health.last_check_ts = ts
            if healthy:
                health.healthy = True
                health.last_success_ts = ts
                health.consecutive_failures = 0
                self.metrics.peer_up.labels(cluster=peer.name).set(1)
                self._sync_queue_metrics(peer)
                self._ensure_policies(peer)
            else:
                health.consecutive_failures += 1
                self.metrics.peer_up.labels(cluster=peer.name).set(0)
                threshold = max(
                    1, int(self.registry.unhealthy_threshold_seconds)
                )
                # If we have never observed a healthy probe, treat the
                # current cycle as the start of the unhealthy window.
                last_ok = (
                    health.last_success_ts
                    if health.last_success_ts is not None
                    else ts
                )
                if (ts - last_ok) >= threshold:
                    health.healthy = False
                    self._remove_policies(peer)
            _log(
                logging.DEBUG,
                "peer.health",
                cluster=peer.name,
                healthy=health.healthy,
                consecutive_failures=health.consecutive_failures,
                payload=payload,
            )

        # Plan C Phase 1, item C5 -- after policy reconciliation, give the
        # failover engine (if wired) a chance to drain in-flight jobs from
        # any peers that have been unhealthy beyond the configured
        # threshold.  The engine is responsible for its own enabled flag,
        # dry-run handling, and exception isolation; the reconciler tick
        # only kicks it.
        if self.failover_engine is not None:
            try:
                report = self.failover_engine.tick()
                _log(
                    logging.DEBUG,
                    "failover.tick.dispatched",
                    report=getattr(report, "as_dict", lambda: {})(),
                )
            except Exception as exc:  # pragma: no cover - defensive
                _log(
                    logging.WARNING,
                    "failover.tick.error",
                    error=str(exc),
                )

        # Plan C Phase 1, item C6 -- after failover, kick the cross-
        # cluster custody chain hook so any jobs that the failover engine
        # just rebalanced get a JOB_REBALANCED custody event.  The hook
        # is fully exception-isolated; custody emission must never block
        # reconciliation.
        if self.custody_chain_engine is not None:
            try:
                emitted = self.custody_chain_engine.tick(now=ts)
                _log(
                    logging.DEBUG,
                    "custody.tick.dispatched",
                    emitted=int(emitted) if emitted is not None else 0,
                )
            except Exception as exc:  # pragma: no cover - defensive
                _log(
                    logging.WARNING,
                    "custody.tick.error",
                    error=str(exc),
                )

        return dict(self._peer_health)

    def run_forever(self, stop_event: threading.Event) -> None:  # pragma: no cover
        """Loop forever (used by the container entrypoint)."""
        interval = max(1, int(self.registry.healthcheck_interval_seconds))
        while not stop_event.is_set():
            try:
                self.reconcile_once()
            except Exception as exc:  # pragma: no cover - defensive
                _log(logging.ERROR, "reconcile.error", error=str(exc))
            stop_event.wait(interval)

    # -- helpers ----------------------------------------------------------
    def _sync_queue_metrics(self, peer: PeerCluster) -> None:
        for queue_class in self.registry.queue_classes:
            payload = _probe_queue_metrics(
                peer.management_uri, queue_class, opener=self._opener
            )
            unacked = payload.get("messages_unacknowledged", 0) or 0
            try:
                lag = float(unacked)
            except (TypeError, ValueError):
                lag = 0.0
            self.metrics.lag_seconds.labels(
                cluster=peer.name, queue=queue_class
            ).set(lag)
            messages_in = payload.get("message_stats", {}).get(
                "publish_in", 0
            ) if isinstance(payload.get("message_stats"), dict) else 0
            messages_out = payload.get("message_stats", {}).get(
                "publish_out", 0
            ) if isinstance(payload.get("message_stats"), dict) else 0
            if messages_in:
                self.metrics.messages_in_total.labels(
                    cluster=peer.name, queue=queue_class
                ).inc(int(messages_in))
            if messages_out:
                self.metrics.messages_out_total.labels(
                    cluster=peer.name, queue=queue_class
                ).inc(int(messages_out))

    def _ensure_policies(self, peer: PeerCluster) -> None:
        for queue_class in self.registry.queue_classes:
            name = _policy_name(self.release_name, peer.name, queue_class)
            if name in self._installed_policies:
                continue
            body = _policy_body(
                release=self.release_name,
                peer=peer.name,
                queue_class=queue_class,
                priority=peer.priority,
                namespace=self.namespace,
                max_overflow=self.max_overflow_messages,
            )
            self._apply_policy(body)
            self._installed_policies.add(name)

    def _remove_policies(self, peer: PeerCluster) -> None:
        for queue_class in self.registry.queue_classes:
            name = _policy_name(self.release_name, peer.name, queue_class)
            if name not in self._installed_policies:
                continue
            self._delete_policy(name)
            self._installed_policies.discard(name)

    # -- Kubernetes glue --------------------------------------------------
    def _apply_policy(self, body: dict[str, Any]) -> None:
        api = self._kubernetes_api
        if api is None:
            _log(
                logging.INFO,
                "policy.apply.skip",
                reason="no_k8s_client",
                name=body["metadata"]["name"],
            )
            return
        try:
            api.create_namespaced_custom_object(
                group="rabbitmq.com",
                version="v1beta1",
                namespace=self.namespace,
                plural="policies",
                body=body,
            )
            _log(logging.INFO, "policy.apply.created", name=body["metadata"]["name"])
        except Exception as exc:  # pragma: no cover - covered via mocks
            _log(
                logging.WARNING,
                "policy.apply.error",
                name=body["metadata"]["name"],
                error=str(exc),
            )

    def _delete_policy(self, name: str) -> None:
        api = self._kubernetes_api
        if api is None:
            _log(
                logging.INFO,
                "policy.delete.skip",
                reason="no_k8s_client",
                name=name,
            )
            return
        try:
            api.delete_namespaced_custom_object(
                group="rabbitmq.com",
                version="v1beta1",
                namespace=self.namespace,
                plural="policies",
                name=name,
            )
            _log(logging.INFO, "policy.delete.removed", name=name)
        except Exception as exc:  # pragma: no cover - covered via mocks
            _log(
                logging.WARNING,
                "policy.delete.error",
                name=name,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------
def load_registry_from_path(path: str) -> Registry:
    """Read the cluster-registry JSON document mounted into the pod."""
    with open(path, "r", encoding="utf-8") as f:
        return Registry.from_json(f.read())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _build_kubernetes_api() -> Any | None:
    if not _KUBERNETES_AVAILABLE:
        _log(
            logging.ERROR,
            "k8s.client.missing",
            note="kubernetes python client not installed; running in observe-only mode",
        )
        return None
    try:  # pragma: no cover - exercised only in cluster
        k8s_config.load_incluster_config()
    except Exception:  # pragma: no cover - dev-mode fallback
        try:
            k8s_config.load_kube_config()
        except Exception as exc:
            _log(logging.ERROR, "k8s.config.error", error=str(exc))
            return None
    return k8s_client.CustomObjectsApi()  # pragma: no cover


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin glue
    del argv  # entry point currently takes no flags
    registry_path = os.environ.get(
        "FEDERATION_REGISTRY_PATH", "/etc/federation/clusters.json"
    )
    metrics_port = int(os.environ.get("FEDERATION_METRICS_PORT", "9100"))
    namespace = os.environ.get("FEDERATION_NAMESPACE", "default")
    release = os.environ.get("FEDERATION_RELEASE", "ocr-local")

    try:
        registry = load_registry_from_path(registry_path)
    except (OSError, ValueError) as exc:
        _log(logging.ERROR, "registry.load.error", path=registry_path, error=str(exc))
        return 2

    if _PROMETHEUS_AVAILABLE:
        start_http_server(metrics_port)
        _log(logging.INFO, "metrics.server.started", port=metrics_port)
    else:
        _log(logging.WARNING, "metrics.server.disabled", reason="prometheus_client_missing")

    api = _build_kubernetes_api()
    reconciler = Reconciler(
        registry,
        release_name=release,
        namespace=namespace,
        kubernetes_api=api,
    )

    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        _log(logging.INFO, "shutdown.signal", signal=signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    _log(
        logging.INFO,
        "reconciler.start",
        peers=[c.name for c in registry.clusters],
        queue_classes=list(registry.queue_classes),
    )
    reconciler.run_forever(stop_event)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
