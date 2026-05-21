"""Federation cluster router -- Plan C Phase 1, item C4.

Multi-strategy cross-cluster job router consulted by the Celery callable
router (``coordinator.coordinator.celery._route_task``) at task dispatch
time. Selects a destination cluster (or local) for a new job based on a
small set of strategies and emits a ``ClusterRoutingDecision``.

Design constraints
------------------
* Stdlib-only HTTP -- no aiohttp / httpx dependency.
* The ``kubernetes`` Python client is **optional** (lazy import).
* ``prometheus_client`` is optional; a no-op shim falls in when missing
  so unit tests pass on minimal interpreters.
* The custom-callable strategy resolves a dotted import path with
  ``importlib.import_module`` + ``getattr`` only -- never ``eval`` /
  ``exec``. Resolution failures fall back to the ``local`` strategy.
* Importable in non-Django, non-Celery contexts. The Celery hook
  instantiates this lazily so a single-cluster install pays no cost.

The router is intentionally minimal: state lives either in-process
(round-robin cursor, load cache) or in Redis (best-effort). All
federation behaviour defaults OFF; with disabled or empty-registry
the router always returns ``ClusterRoutingDecision(cluster_name=None,
queue_name=base_queue, reason_code="local_default")``.
"""

from __future__ import annotations

import importlib
import itertools
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only inside Kubernetes pods
    from kubernetes import client as k8s_client  # type: ignore[import-not-found]
    from kubernetes import config as k8s_config  # type: ignore[import-not-found]

    _KUBERNETES_AVAILABLE = True
except ImportError:  # pragma: no cover - executed in unit tests
    k8s_client = None  # type: ignore[assignment]
    k8s_config = None  # type: ignore[assignment]
    _KUBERNETES_AVAILABLE = False

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

        def set(self, _value: float) -> None:
            return None

    Counter = _NoopMetric  # type: ignore[assignment,misc]


_LOG = logging.getLogger("federation.cluster_router")


_StrategyName = Literal[
    "local",
    "priority",
    "round_robin",
    "load_aware",
    "region_affinity",
    "custom",
]
_VALID_STRATEGIES: tuple[str, ...] = (
    "local",
    "priority",
    "round_robin",
    "load_aware",
    "region_affinity",
    "custom",
)


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterRoutingDecision:
    """Outcome of a single ``select_cluster`` call.

    ``cluster_name=None`` means "stay local". ``queue_name`` always carries
    the final queue Celery should dispatch to (for federated peers this is
    ``<base_queue>.<cluster_name>``).
    """

    cluster_name: str | None
    queue_name: str
    reason_code: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cluster registry -- ConfigMap or YAML-file backed
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterEntry:
    name: str
    region: str
    rabbitmq_uri: str
    management_uri: str
    priority: int
    tags: tuple[str, ...]
    tls_enabled: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClusterEntry":
        tls = data.get("tls") or {}
        return cls(
            name=str(data["name"]),
            region=str(data.get("region", "")),
            rabbitmq_uri=str(data.get("rabbitmq_uri", "")),
            management_uri=str(data.get("management_uri", "")),
            priority=int(data.get("priority", 10)),
            tags=tuple(str(t) for t in (data.get("tags") or [])),
            tls_enabled=bool(tls.get("enabled", False)),
        )


HealthProvider = Callable[[str], bool]


class ClusterRegistry:
    """Loads the federation peer registry from a ConfigMap or YAML file.

    The registry is polled periodically (default 30 s) so live edits to the
    ConfigMap or YAML file propagate without a restart. In Kubernetes, the
    ``kubernetes`` client is loaded lazily; in dev/test we point at a YAML
    file via ``OCR_FEDERATION_REGISTRY_PATH``.

    Health is sourced from a pluggable provider so unit tests can inject
    fake liveness without standing up Prometheus.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        configmap_name: str | None = None,
        configmap_namespace: str | None = None,
        poll_seconds: float = 30.0,
        health_provider: HealthProvider | None = None,
        clock: Callable[[], float] | None = None,
        kubernetes_api: Any = None,
    ) -> None:
        self._path = path
        self._configmap_name = configmap_name
        self._configmap_namespace = configmap_namespace
        self._poll_seconds = max(1.0, float(poll_seconds))
        self._health_provider: HealthProvider = (
            health_provider if health_provider is not None else (lambda _name: True)
        )
        self._clock = clock if clock is not None else time.time
        self._kubernetes_api = kubernetes_api
        self._lock = threading.RLock()
        self._clusters: list[ClusterEntry] = []
        self._self_name: str = ""
        self._last_loaded_ts: float = 0.0
        self._loaded_once = False
        # First load is best-effort; failures leave the registry empty so
        # the router falls back to "local".
        try:
            self._reload_locked(force=True)
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("registry initial load failed: %s", exc)

    # -- public surface ---------------------------------------------------
    def clusters(self) -> list[ClusterEntry]:
        """Return the current peer list (may trigger a poll)."""
        with self._lock:
            self._maybe_reload_locked()
            return list(self._clusters)

    def self_name(self) -> str:
        with self._lock:
            self._maybe_reload_locked()
            return self._self_name

    def is_peer_healthy(self, name: str) -> bool:
        try:
            return bool(self._health_provider(name))
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("health provider raised for %s: %s", name, exc)
            return False

    def force_reload(self) -> None:
        with self._lock:
            self._reload_locked(force=True)

    # -- internals --------------------------------------------------------
    def _maybe_reload_locked(self) -> None:
        if not self._loaded_once:
            self._reload_locked(force=True)
            return
        now = self._clock()
        if (now - self._last_loaded_ts) >= self._poll_seconds:
            self._reload_locked(force=False)

    def _reload_locked(self, *, force: bool) -> None:
        del force  # currently advisory; reserved for retry/back-off
        data: dict[str, Any] | None = None
        if self._path:
            data = self._load_from_path(self._path)
        elif self._configmap_name and self._configmap_namespace:
            data = self._load_from_configmap(
                self._configmap_name, self._configmap_namespace
            )
        if data is None:
            # No source available -- keep whatever we had (often empty).
            self._loaded_once = True
            self._last_loaded_ts = self._clock()
            return
        self._self_name = str((data.get("self") or {}).get("name", ""))
        self._clusters = [
            ClusterEntry.from_dict(c) for c in (data.get("clusters") or [])
        ]
        self._last_loaded_ts = self._clock()
        self._loaded_once = True

    @staticmethod
    def _load_from_path(path: str) -> dict[str, Any] | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            _LOG.warning("registry file %s unreadable: %s", path, exc)
            return None
        # Accept both YAML and JSON; YAML is optional, JSON is stdlib.
        try:
            import yaml  # type: ignore[import-not-found]

            return yaml.safe_load(text) or {}
        except ImportError:  # pragma: no cover - PyYAML always available locally
            try:
                return json.loads(text)
            except ValueError as exc:
                _LOG.warning("registry file %s parse failed: %s", path, exc)
                return None
        except Exception as exc:
            _LOG.warning("registry yaml parse failed: %s", exc)
            return None

    def _load_from_configmap(
        self, name: str, namespace: str
    ) -> dict[str, Any] | None:
        api = self._kubernetes_api
        if api is None and _KUBERNETES_AVAILABLE:  # pragma: no cover
            try:
                k8s_config.load_incluster_config()
            except Exception:
                try:
                    k8s_config.load_kube_config()
                except Exception as exc:
                    _LOG.warning("kube config load failed: %s", exc)
                    return None
            api = k8s_client.CoreV1Api()
            self._kubernetes_api = api
        if api is None:
            return None
        try:  # pragma: no cover - exercised only inside K8s
            cm = api.read_namespaced_config_map(name=name, namespace=namespace)
            text = (cm.data or {}).get("clusters.json") or (cm.data or {}).get(
                "clusters.yaml"
            )
            if not text:
                return None
            try:
                return json.loads(text)
            except ValueError:
                import yaml  # type: ignore[import-not-found]

                return yaml.safe_load(text) or {}
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("configmap %s/%s read failed: %s", namespace, name, exc)
            return None


# ---------------------------------------------------------------------------
# Prometheus surface
# ---------------------------------------------------------------------------
_DECISIONS_COUNTER: Any | None = None
_COUNTER_LOCK = threading.Lock()


def _get_decisions_counter() -> Any:
    """Lazy-initialise the decisions counter exactly once.

    Defined as a module-singleton because Counter() is registered against
    the default Prometheus registry; calling it twice in the same process
    would raise a ``Duplicated timeseries`` error.
    """
    global _DECISIONS_COUNTER
    with _COUNTER_LOCK:
        if _DECISIONS_COUNTER is not None:
            return _DECISIONS_COUNTER
        if not _PROMETHEUS_AVAILABLE:
            _DECISIONS_COUNTER = Counter()  # type: ignore[call-arg]
        else:
            try:
                _DECISIONS_COUNTER = Counter(
                    "ocr_cluster_router_decisions_total",
                    "Cross-cluster routing decisions by strategy and outcome.",
                    ["strategy", "decision"],
                )
            except ValueError:
                # Already registered; fetch by name from the global registry.
                from prometheus_client import REGISTRY  # type: ignore[import-not-found]

                _DECISIONS_COUNTER = next(
                    (
                        c
                        for c in REGISTRY._collector_to_names  # type: ignore[attr-defined]
                        if getattr(c, "_name", "") == "ocr_cluster_router_decisions"
                    ),
                    Counter(),  # type: ignore[call-arg]
                )
    return _DECISIONS_COUNTER


# ---------------------------------------------------------------------------
# Load probe -- 5s cache to avoid hammering peer Management APIs
# ---------------------------------------------------------------------------
def _probe_messages_unacknowledged(
    management_uri: str,
    queue: str,
    *,
    timeout: float = 5.0,
    opener: Any = None,
) -> int | None:
    if not management_uri:
        return None
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
        if not (200 <= int(status) < 300):
            return None
        payload = json.loads(body.decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, UnicodeDecodeError):
        return None
    val = payload.get("messages_unacknowledged", 0) or 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Custom-callable resolver (importlib only -- never eval/exec)
# ---------------------------------------------------------------------------
def _resolve_custom_callable(dotted: str) -> Callable[..., Any] | None:
    """Resolve ``module.path:attr`` or ``module.path.attr`` to a callable.

    Returns ``None`` if the path is empty, unresolvable, or the symbol is
    not callable. Never executes arbitrary code beyond importing the
    declared module.
    """
    if not dotted:
        return None
    if ":" in dotted:
        module_name, attr = dotted.split(":", 1)
    else:
        if "." not in dotted:
            return None
        module_name, attr = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        _LOG.warning("custom callable %s not importable: %s", dotted, exc)
        return None
    target = getattr(module, attr, None)
    if not callable(target):
        _LOG.warning("custom callable %s is not callable", dotted)
        return None
    return target


# ---------------------------------------------------------------------------
# Cluster router
# ---------------------------------------------------------------------------
class ClusterRouter:
    """Multi-strategy router that picks a destination cluster.

    Strategies (the env var ``OCR_FEDERATION_ROUTING_STRATEGY`` selects one
    at construction time, but the ``strategy`` argument wins when
    provided explicitly):

    * ``local`` (default) -- always returns ``cluster_name=None``.
    * ``priority`` -- highest peer priority wins; ties broken by name.
    * ``round_robin`` -- rotates across the healthy peer set.
    * ``load_aware`` -- prefers the peer with the lowest
      ``messages_unacknowledged`` count (5s cached).
    * ``region_affinity`` -- prefers a peer whose region matches
      ``kwargs["origin_region"]``; falls back to ``priority``.
    * ``custom`` -- delegates to an importable callable.
    """

    def __init__(
        self,
        registry: ClusterRegistry,
        *,
        strategy: _StrategyName,
        local_cluster_name: str,
        custom_callable: str = "",
        load_aware_cache_seconds: float = 5.0,
        load_probe: Callable[[str, str], int | None] | None = None,
        clock: Callable[[], float] | None = None,
        round_robin_state: Callable[[], int] | None = None,
    ) -> None:
        if strategy not in _VALID_STRATEGIES:
            _LOG.warning(
                "unknown strategy %r; falling back to 'local'", strategy
            )
            strategy = "local"
        self._registry = registry
        self._strategy: str = strategy
        self._local_name = local_cluster_name
        self._custom_callable_path = custom_callable
        self._custom_callable = _resolve_custom_callable(custom_callable)
        self._load_aware_cache_seconds = max(0.0, float(load_aware_cache_seconds))
        self._load_probe = load_probe if load_probe is not None else (
            lambda mgmt, q: _probe_messages_unacknowledged(mgmt, q)
        )
        self._clock = clock if clock is not None else time.time
        self._rr_counter = (
            round_robin_state
            if round_robin_state is not None
            else (lambda _it=itertools.count(): next(_it))
        )
        self._load_cache: dict[str, tuple[float, int]] = {}
        self._load_cache_lock = threading.Lock()

    # -- public surface ---------------------------------------------------
    @property
    def strategy(self) -> str:
        return self._strategy

    def select_cluster(
        self,
        *,
        task_name: str,
        base_queue: str,
        kwargs: dict[str, Any] | None,
    ) -> ClusterRoutingDecision:
        kwargs = kwargs or {}
        # 1. Load registry; if empty, route locally regardless of strategy.
        peers = self._registry.clusters()
        local_decision = ClusterRoutingDecision(
            cluster_name=None,
            queue_name=base_queue,
            reason_code="local_default",
            metadata={"strategy": self._strategy},
        )
        if not peers:
            return self._record(local_decision)

        if self._strategy == "local":
            return self._record(local_decision)

        # 2. Filter for healthy peers (the local cluster is always considered
        # "healthy" -- it is implicit, not in the peer registry).
        healthy_peers = [
            p for p in peers if self._registry.is_peer_healthy(p.name)
        ]
        if not healthy_peers:
            return self._record(
                ClusterRoutingDecision(
                    cluster_name=None,
                    queue_name=base_queue,
                    reason_code="no_healthy_peers",
                    metadata={"strategy": self._strategy},
                )
            )

        # 3. Dispatch to the strategy-specific picker.
        if self._strategy == "priority":
            chosen = self._pick_priority(healthy_peers)
            return self._record(self._build_decision(chosen, base_queue, "priority"))
        if self._strategy == "round_robin":
            chosen = self._pick_round_robin(healthy_peers)
            return self._record(
                self._build_decision(chosen, base_queue, "round_robin")
            )
        if self._strategy == "load_aware":
            chosen, load = self._pick_load_aware(healthy_peers, base_queue)
            return self._record(
                self._build_decision(
                    chosen, base_queue, "load_aware", extra={"load": load}
                )
            )
        if self._strategy == "region_affinity":
            chosen = self._pick_region_affinity(
                healthy_peers, kwargs.get("origin_region")
            )
            return self._record(
                self._build_decision(chosen, base_queue, "region_affinity")
            )
        if self._strategy == "custom":
            if self._custom_callable is None:
                return self._record(
                    ClusterRoutingDecision(
                        cluster_name=None,
                        queue_name=base_queue,
                        reason_code="custom_unresolved",
                        metadata={
                            "strategy": "custom",
                            "callable": self._custom_callable_path,
                        },
                    )
                )
            try:
                returned = self._custom_callable(
                    task_name=task_name,
                    base_queue=base_queue,
                    kwargs=kwargs,
                    peers=healthy_peers,
                    local_cluster_name=self._local_name,
                )
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning("custom callable raised: %s", exc)
                return self._record(
                    ClusterRoutingDecision(
                        cluster_name=None,
                        queue_name=base_queue,
                        reason_code="custom_error",
                        metadata={"strategy": "custom", "error": str(exc)},
                    )
                )
            decision = self._coerce_custom_return(returned, base_queue)
            return self._record(decision)

        # Should be unreachable; defensive fall-through.
        return self._record(local_decision)

    # -- strategy implementations ----------------------------------------
    def _pick_priority(self, peers: list[ClusterEntry]) -> ClusterEntry:
        # Highest priority wins; tie-break by name (deterministic).
        return sorted(peers, key=lambda p: (-int(p.priority), p.name))[0]

    def _pick_round_robin(self, peers: list[ClusterEntry]) -> ClusterEntry:
        sorted_peers = sorted(peers, key=lambda p: p.name)
        idx = int(self._rr_counter()) % len(sorted_peers)
        return sorted_peers[idx]

    def _pick_load_aware(
        self, peers: list[ClusterEntry], base_queue: str
    ) -> tuple[ClusterEntry, int | None]:
        scored: list[tuple[int, ClusterEntry]] = []
        for p in peers:
            load = self._cached_load(p, base_queue)
            scored.append(
                (load if load is not None else _UNKNOWN_LOAD_SENTINEL, p)
            )
        # Lowest load wins; ties broken by name. Unknown loads sort last.
        scored.sort(key=lambda t: (t[0], t[1].name))
        load, peer = scored[0]
        actual = None if load == _UNKNOWN_LOAD_SENTINEL else load
        return peer, actual

    def _cached_load(
        self, peer: ClusterEntry, queue: str
    ) -> int | None:
        cache_key = f"{peer.name}::{queue}"
        now = self._clock()
        with self._load_cache_lock:
            cached = self._load_cache.get(cache_key)
            if cached is not None:
                ts, val = cached
                if (now - ts) < self._load_aware_cache_seconds:
                    return val
        load = self._load_probe(peer.management_uri, queue)
        if load is not None:
            with self._load_cache_lock:
                self._load_cache[cache_key] = (now, load)
        return load

    def _pick_region_affinity(
        self, peers: list[ClusterEntry], origin_region: Any
    ) -> ClusterEntry:
        if isinstance(origin_region, str) and origin_region:
            same_region = [p for p in peers if p.region == origin_region]
            if same_region:
                return self._pick_priority(same_region)
        return self._pick_priority(peers)

    # -- helpers ---------------------------------------------------------
    def _build_decision(
        self,
        peer: ClusterEntry,
        base_queue: str,
        reason_code: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> ClusterRoutingDecision:
        meta: dict[str, Any] = {
            "strategy": self._strategy,
            "peer": peer.name,
            "region": peer.region,
            "priority": peer.priority,
        }
        if extra:
            meta.update(extra)
        return ClusterRoutingDecision(
            cluster_name=peer.name,
            queue_name=f"{base_queue}.{peer.name}",
            reason_code=reason_code,
            metadata=meta,
        )

    def _coerce_custom_return(
        self, returned: Any, base_queue: str
    ) -> ClusterRoutingDecision:
        if isinstance(returned, ClusterRoutingDecision):
            return returned
        if returned is None:
            return ClusterRoutingDecision(
                cluster_name=None,
                queue_name=base_queue,
                reason_code="custom_local",
                metadata={"strategy": "custom"},
            )
        if isinstance(returned, str) and returned:
            return ClusterRoutingDecision(
                cluster_name=returned,
                queue_name=f"{base_queue}.{returned}",
                reason_code="custom_picked",
                metadata={"strategy": "custom", "peer": returned},
            )
        # Anything else falls back to local.
        return ClusterRoutingDecision(
            cluster_name=None,
            queue_name=base_queue,
            reason_code="custom_invalid_return",
            metadata={"strategy": "custom"},
        )

    def _record(
        self, decision: ClusterRoutingDecision
    ) -> ClusterRoutingDecision:
        try:
            counter = _get_decisions_counter()
            decision_label = (
                "local" if decision.cluster_name is None else "peer"
            )
            counter.labels(
                strategy=self._strategy, decision=decision_label
            ).inc()
        except Exception:  # pragma: no cover - defensive
            pass
        return decision


_UNKNOWN_LOAD_SENTINEL: int = 1 << 30


# ---------------------------------------------------------------------------
# Construction helpers used by the Celery hook
# ---------------------------------------------------------------------------
def build_router_from_env() -> ClusterRouter | None:
    """Construct a ``ClusterRouter`` from environment variables.

    Returns ``None`` when ``OCR_FEDERATION_ROUTING_ENABLED`` is unset or
    falsy, signalling the Celery hook to keep its existing single-cluster
    behaviour.
    """
    enabled = os.environ.get("OCR_FEDERATION_ROUTING_ENABLED", "false")
    if enabled.strip().lower() not in ("1", "true", "yes"):
        return None

    strategy_raw = os.environ.get(
        "OCR_FEDERATION_ROUTING_STRATEGY", "local"
    ).strip().lower()
    if strategy_raw not in _VALID_STRATEGIES:
        _LOG.warning(
            "OCR_FEDERATION_ROUTING_STRATEGY=%r invalid; using 'local'",
            strategy_raw,
        )
        strategy_raw = "local"

    poll_seconds = _safe_float(
        os.environ.get("OCR_FEDERATION_REGISTRY_POLL_SECONDS"), 30.0
    )
    cache_seconds = _safe_float(
        os.environ.get("OCR_FEDERATION_LOAD_AWARE_CACHE_SECONDS"), 5.0
    )

    path = os.environ.get("OCR_FEDERATION_REGISTRY_PATH") or os.environ.get(
        "FEDERATION_REGISTRY_PATH"
    )
    cm_name = os.environ.get("OCR_FEDERATION_REGISTRY_CONFIGMAP")
    cm_ns = os.environ.get(
        "OCR_FEDERATION_REGISTRY_NAMESPACE"
    ) or os.environ.get("FEDERATION_NAMESPACE")
    local_name = os.environ.get("OCR_CLUSTER_NAME", "")

    registry = ClusterRegistry(
        path=path,
        configmap_name=cm_name,
        configmap_namespace=cm_ns,
        poll_seconds=poll_seconds,
    )
    return ClusterRouter(
        registry,
        strategy=strategy_raw,  # type: ignore[arg-type]
        local_cluster_name=local_name,
        custom_callable=os.environ.get(
            "OCR_FEDERATION_ROUTER_CALLABLE", ""
        ),
        load_aware_cache_seconds=cache_seconds,
    )


def _safe_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "ClusterEntry",
    "ClusterRegistry",
    "ClusterRouter",
    "ClusterRoutingDecision",
    "HealthProvider",
    "build_router_from_env",
]
