"""Federation failover engine -- Plan C Phase 1, item C5.

Detects peer clusters that have been unhealthy beyond a configurable
threshold and re-dispatches their in-flight PostgreSQL ``Job`` rows to
healthy peers (or back to the local cluster).

Scope boundary
==============
This engine deals with **PostgreSQL ``Job`` rows + Celery dispatch** only.
It does NOT drain messages already sitting on a federated queue at the
peer broker -- those are recovered by RabbitMQ's federation upstream when
the peer comes back, or via operator intervention when the peer is gone
for good.  This intentionally narrow scope keeps the engine safe to run
opportunistically (no broker write paths, no policy changes).

Integration surface
===================
The engine plugs into two seams:

1. The C2 reconciler (``coordinator.federation.reconciler``) calls
   ``FailoverEngine.tick()`` once per reconcile cycle when the engine is
   wired in.  Default behaviour is unchanged -- when the engine is
   ``None`` the reconciler does not invoke anything new.

2. ``coordinator.jobs.tasks`` calls ``FailoverEngine.tick()``
   opportunistically when a worker reports a node-level failure
   ("cluster_unhealthy") or when a soft-time-limit fires for a job whose
   assigned cluster is no longer healthy.

Lazy import of the C4 router
============================
``ClusterRouter`` lives in the parallel C4 lane (``cluster_router.py``).
The import is guarded so this module loads cleanly when C4 has not yet
landed -- behaviour collapses to "always route to local cluster".
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Optional C4 dependency: lazy import.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on parallel lane landing
    from coordinator.federation.cluster_router import (  # type: ignore[import-not-found]
        ClusterRouter,
    )
except ImportError:  # pragma: no cover - tested via mocks
    ClusterRouter = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Optional Prometheus dependency: same shim pattern as the reconciler so
# unit tests can run on minimal interpreters.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (  # type: ignore[import-not-found]
        CollectorRegistry,
        Counter,
        Gauge,
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

        def dec(self, _amount: float = 1.0) -> None:
            return None

    Gauge = Counter = _NoopMetric  # type: ignore[assignment,misc]
    CollectorRegistry = object  # type: ignore[assignment,misc]


_LOG = logging.getLogger("federation.failover")


# ---------------------------------------------------------------------------
# Custody event reason codes
# ---------------------------------------------------------------------------
# The coordinator's ``CustodyEvent.event_type`` field is a free-form
# CharField (not an enum), so reason codes are string constants.  Keeping
# them here keeps the event vocabulary discoverable from one place.
JOB_FAILOVER_REROUTED = "JOB_FAILOVER_REROUTED"
JOB_FAILOVER_SKIPPED = "JOB_FAILOVER_SKIPPED"


# ---------------------------------------------------------------------------
# Decisions exposed via Prometheus labels
# ---------------------------------------------------------------------------
DECISION_REROUTED = "rerouted"
DECISION_DRY_RUN = "dry_run"
DECISION_NO_DESTINATION = "no_destination"
DECISION_PEER_HEALTHY = "peer_healthy"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FailoverPolicy:
    """Tunable knobs for the failover engine.  Defaults are conservative."""

    enabled: bool = False
    unhealthy_threshold_seconds: int = 120
    min_in_flight_to_failover: int = 1
    max_concurrent_failovers: int = 100
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "FailoverPolicy":
        """Load policy from environment variables.

        All variables are prefixed ``OCR_FEDERATION_FAILOVER_``.
        """
        def _bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=_bool("OCR_FEDERATION_FAILOVER_ENABLED", False),
            unhealthy_threshold_seconds=_int(
                "OCR_FEDERATION_FAILOVER_UNHEALTHY_SECONDS", 120
            ),
            min_in_flight_to_failover=_int(
                "OCR_FEDERATION_FAILOVER_MIN_IN_FLIGHT", 1
            ),
            max_concurrent_failovers=_int(
                "OCR_FEDERATION_FAILOVER_MAX_CONCURRENT", 100
            ),
            dry_run=_bool("OCR_FEDERATION_FAILOVER_DRY_RUN", False),
        )


@dataclass
class FailoverReport:
    """Aggregate counters for one tick of the engine."""

    peers_unhealthy: int = 0
    jobs_examined: int = 0
    jobs_failed_over: int = 0
    jobs_skipped_dry_run: int = 0
    jobs_skipped_no_destination: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "peers_unhealthy": self.peers_unhealthy,
            "jobs_examined": self.jobs_examined,
            "jobs_failed_over": self.jobs_failed_over,
            "jobs_skipped_dry_run": self.jobs_skipped_dry_run,
            "jobs_skipped_no_destination": self.jobs_skipped_no_destination,
            "duration_seconds": max(0.0, self.ended_at - self.started_at),
        }


# ---------------------------------------------------------------------------
# Metrics surface
# ---------------------------------------------------------------------------
@dataclass
class FailoverMetrics:
    failover_total: Any
    in_progress: Any

    @classmethod
    def build(cls, registry: Any | None = None) -> "FailoverMetrics":
        if not _PROMETHEUS_AVAILABLE:  # pragma: no cover - tested via env
            shim = Gauge()
            return cls(shim, shim)
        kwargs = {"registry": registry} if registry is not None else {}
        return cls(
            failover_total=Counter(
                "ocr_federation_failover_total",
                "Federation failover decisions, labelled by source cluster "
                "and decision (rerouted|dry_run|no_destination|peer_healthy).",
                ["cluster", "decision"],
                **kwargs,
            ),
            in_progress=Gauge(
                "ocr_federation_failover_in_progress",
                "Number of jobs currently being failed over (gauge).",
                **kwargs,
            ),
        )


# ---------------------------------------------------------------------------
# Built-in fallback router (used when C4's ClusterRouter is absent).
# ---------------------------------------------------------------------------
class _LocalOnlyRouter:
    """Tiny fallback that always returns ``"local"``.

    Used when the C4 ``ClusterRouter`` lane has not landed yet, or when a
    deployment intentionally disables federation routing.  Returning a
    constant string keeps the dispatch path simple and observable.
    """

    LOCAL = "local"

    def select_cluster(
        self,
        *,
        queue: str | None = None,
        tenant_id: str | None = None,
        priority: str | None = None,
        exclude: tuple[str, ...] = (),
    ) -> str | None:
        del queue, tenant_id, priority
        if self.LOCAL in exclude:
            return None
        return self.LOCAL


# ---------------------------------------------------------------------------
# Failover engine
# ---------------------------------------------------------------------------
class FailoverEngine:
    """Detects unhealthy peers and re-dispatches their in-flight jobs.

    Concurrency
    -----------
    ``tick()`` is safe to invoke concurrently from multiple threads (the
    reconciler tick + an opportunistic worker retry hook may collide).  An
    internal :class:`threading.RLock` serialises the inspection-and-dispatch
    loop and tracks an in-flight set so two parallel ticks never failover
    the same job twice.

    Side effects
    ------------
    On a real (``dry_run=False``) failover, ``tick()``:

    * Inserts a ``CustodyEvent`` row of type ``JOB_FAILOVER_REROUTED``.
    * Calls ``revoke_callback(celery_task_id)`` (best effort -- exceptions
      are logged and swallowed).
    * Updates ``Job.assigned_cluster`` to the new destination.
    * Calls ``dispatch_callback(job, new_cluster)`` to actually re-dispatch
      the work.  In production this lands in ``coordinator.jobs.tasks``;
      tests inject a mock.
    """

    def __init__(
        self,
        registry: Any,
        health_provider: Callable[[], dict[str, Any]],
        *,
        broker_url: str | None = None,
        db_session_factory: Callable[[], Any] | None = None,
        policy: FailoverPolicy | None = None,
        router: Any = None,
        metrics: FailoverMetrics | None = None,
        clock: Callable[[], float] = time.time,
        revoke_callback: Callable[[str], None] | None = None,
        dispatch_callback: Callable[[Any, str], None] | None = None,
    ) -> None:
        self.registry = registry
        self.health_provider = health_provider
        self.broker_url = broker_url
        self._db_session_factory = db_session_factory
        self.policy = policy or FailoverPolicy()
        self.router = router or _LocalOnlyRouter()
        self.metrics = metrics or FailoverMetrics.build()
        self._clock = clock
        self._revoke_callback = revoke_callback
        self._dispatch_callback = dispatch_callback

        self._lock = threading.RLock()
        self._in_flight_failover_ids: set[str] = set()

    # -- public API -------------------------------------------------------
    def tick(self) -> FailoverReport:
        """Run a single pass over the unhealthy-peer set.

        Returns a :class:`FailoverReport`.  The method holds an internal
        lock for its entire duration so concurrent callers see consistent
        state; callers that need fast non-blocking behaviour should
        gate the call themselves (e.g. a time-budget check).
        """
        report = FailoverReport(started_at=self._clock())
        try:
            with self._lock:
                if not self.policy.enabled:
                    _LOG.debug(
                        "failover.tick.disabled",
                        extra={"policy": "enabled=False"},
                    )
                    report.ended_at = self._clock()
                    return report

                unhealthy_peers = self._collect_unhealthy_peers()
                report.peers_unhealthy = len(unhealthy_peers)
                if not unhealthy_peers:
                    report.ended_at = self._clock()
                    return report

                for peer_name in unhealthy_peers:
                    self._failover_peer(peer_name, report)
                    if (
                        report.jobs_failed_over
                        >= self.policy.max_concurrent_failovers
                    ):
                        _LOG.info(
                            "failover.tick.max_concurrent_reached",
                            extra={
                                "max_concurrent": self.policy.max_concurrent_failovers,
                            },
                        )
                        break
        finally:
            report.ended_at = self._clock()
            try:
                self.metrics.in_progress.set(0)
            except Exception:  # pragma: no cover - metric shim is no-op
                pass
        _LOG.info("failover.tick.report", extra={"report": report.as_dict()})
        return report

    def run_loop(self, stop_event: threading.Event, interval_seconds: int = 30) -> None:
        """Loop ``tick()`` forever until ``stop_event`` is set.

        Intended for an optional daemon thread; the reconciler-driven
        integration is the primary execution path.
        """
        interval = max(1, int(interval_seconds))
        while not stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.error("failover.run_loop.error", extra={"error": str(exc)})
            stop_event.wait(interval)

    # -- internal --------------------------------------------------------
    def _collect_unhealthy_peers(self) -> list[str]:
        health_map = self.health_provider() or {}
        threshold = max(1, int(self.policy.unhealthy_threshold_seconds))
        now = self._clock()
        unhealthy: list[str] = []
        for peer_name, health in health_map.items():
            healthy_flag = bool(getattr(health, "healthy", True))
            last_success = getattr(health, "last_success_ts", None)
            if healthy_flag:
                continue
            elapsed = now - (last_success if last_success is not None else now)
            if elapsed >= threshold:
                unhealthy.append(peer_name)
        return unhealthy

    def _failover_peer(self, peer_name: str, report: FailoverReport) -> None:
        jobs = self._list_in_flight_jobs(peer_name)
        if len(jobs) < self.policy.min_in_flight_to_failover:
            _LOG.info(
                "failover.peer.below_min_in_flight",
                extra={
                    "peer": peer_name,
                    "in_flight": len(jobs),
                    "min": self.policy.min_in_flight_to_failover,
                },
            )
            return

        for job in jobs:
            report.jobs_examined += 1
            job_pk = self._job_pk(job)
            if job_pk is None or job_pk in self._in_flight_failover_ids:
                continue

            destination = self._select_destination(job, peer_name)
            if destination is None or destination == peer_name:
                report.jobs_skipped_no_destination += 1
                self.metrics.failover_total.labels(
                    cluster=peer_name, decision=DECISION_NO_DESTINATION
                ).inc()
                _LOG.warning(
                    "failover.job.no_destination",
                    extra={"peer": peer_name, "job_id": str(job_pk)},
                )
                continue

            if self.policy.dry_run:
                report.jobs_skipped_dry_run += 1
                self.metrics.failover_total.labels(
                    cluster=peer_name, decision=DECISION_DRY_RUN
                ).inc()
                _LOG.info(
                    "failover.job.dry_run",
                    extra={
                        "peer": peer_name,
                        "destination": destination,
                        "job_id": str(job_pk),
                    },
                )
                continue

            self._in_flight_failover_ids.add(job_pk)
            try:
                self.metrics.in_progress.inc()
                self._reroute_job(job, peer_name, destination)
                report.jobs_failed_over += 1
                self.metrics.failover_total.labels(
                    cluster=peer_name, decision=DECISION_REROUTED
                ).inc()
            finally:
                self._in_flight_failover_ids.discard(job_pk)
                try:
                    self.metrics.in_progress.dec()
                except Exception:  # pragma: no cover - shim
                    pass

            if (
                report.jobs_failed_over
                >= self.policy.max_concurrent_failovers
            ):
                break

    def _select_destination(self, job: Any, peer_name: str) -> str | None:
        """Ask the router for a new home for ``job``.

        We exclude the currently-unhealthy peer from the candidate set.
        The C4 ``ClusterRouter`` accepts an ``exclude`` tuple; the local
        fallback router honours the same contract.
        """
        if self.router is None:
            return _LocalOnlyRouter.LOCAL
        kwargs: dict[str, Any] = {}
        # Pass through any optional hints the job carries.  Both routers
        # tolerate unknown kwargs: ``_LocalOnlyRouter.select_cluster`` has
        # explicit parameters and ``ClusterRouter`` accepts ``**kwargs``.
        for key in ("priority", "tenant_id"):
            value = getattr(job, key, None)
            if value:
                kwargs[key] = value
        kwargs["exclude"] = (peer_name,)
        try:
            return self.router.select_cluster(**kwargs)
        except TypeError:
            # Older router signatures may not accept ``exclude=``; degrade
            # gracefully by retrying without it.
            try:
                return self.router.select_cluster()
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning(
                    "failover.router.select_failed",
                    extra={"error": str(exc)},
                )
                return None
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning(
                "failover.router.select_failed",
                extra={"error": str(exc)},
            )
            return None

    def _reroute_job(self, job: Any, peer_name: str, destination: str) -> None:
        """Revoke the old Celery task, record custody, dispatch to ``destination``.

        All side effects are wrapped in try/except so a partial failure on
        one job does not abort the whole tick.
        """
        celery_task_id = getattr(job, "celery_task_id", "") or ""
        if celery_task_id and self._revoke_callback is not None:
            try:
                self._revoke_callback(celery_task_id)
            except Exception as exc:  # pragma: no cover - best effort
                _LOG.warning(
                    "failover.revoke.error",
                    extra={
                        "task_id": celery_task_id,
                        "error": str(exc),
                    },
                )

        # Custody event + DB row update happen via the Django ORM when
        # available; tests use an in-memory mock to assert the call.
        self._record_custody_event(job, peer_name, destination)
        self._update_job_cluster(job, destination)

        if self._dispatch_callback is None:
            _LOG.warning(
                "failover.dispatch.no_callback",
                extra={"job_id": str(self._job_pk(job))},
            )
            return
        try:
            self._dispatch_callback(job, destination)
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.error(
                "failover.dispatch.error",
                extra={
                    "job_id": str(self._job_pk(job)),
                    "destination": destination,
                    "error": str(exc),
                },
            )

    # -- DB / ORM glue ---------------------------------------------------
    def _list_in_flight_jobs(self, peer_name: str) -> list[Any]:
        """Return Job rows whose ``assigned_cluster`` matches ``peer_name``.

        The query goes through the optional ``db_session_factory`` so unit
        tests can inject a deterministic list without touching Django.
        """
        if self._db_session_factory is not None:
            try:
                return list(self._db_session_factory()(peer_name))
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning(
                    "failover.db.list_error",
                    extra={"peer": peer_name, "error": str(exc)},
                )
                return []
        # Default path: try the Django ORM.
        try:  # pragma: no cover - exercised in coordinator integration
            from jobs.models import Job  # type: ignore[import-not-found]

            qs = Job.objects.filter(
                assigned_cluster=peer_name,
                status__in=[
                    Job.Status.SUBMITTED,
                    Job.Status.INGESTING,
                    Job.Status.PROCESSING,
                ],
            )
            return list(qs)
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning(
                "failover.db.import_error",
                extra={"peer": peer_name, "error": str(exc)},
            )
            return []

    def _record_custody_event(
        self, job: Any, peer_name: str, destination: str
    ) -> None:
        try:  # pragma: no cover - exercised in coordinator integration
            from django.utils import timezone  # type: ignore[import-not-found]
            from jobs.models import CustodyEvent  # type: ignore[import-not-found]

            CustodyEvent.objects.create(
                document_id=getattr(job, "source_hash", "") or "",
                job=job,
                event_type=JOB_FAILOVER_REROUTED,
                timestamp=timezone.now(),
                worker_hostname="",
                data={
                    "from_cluster": peer_name,
                    "to_cluster": destination,
                    "reason": "cluster_unhealthy",
                },
            )
        except Exception as exc:  # pragma: no cover - tests use mocks
            _LOG.warning(
                "failover.custody.error",
                extra={
                    "job_id": str(self._job_pk(job)),
                    "error": str(exc),
                },
            )

    def _update_job_cluster(self, job: Any, destination: str) -> None:
        try:
            setattr(job, "assigned_cluster", destination)
            save_fn = getattr(job, "save", None)
            if callable(save_fn):
                save_fn(update_fields=["assigned_cluster"])
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning(
                "failover.update.error",
                extra={
                    "job_id": str(self._job_pk(job)),
                    "error": str(exc),
                },
            )

    @staticmethod
    def _job_pk(job: Any) -> str | None:
        """Return a stable primary-key string for ``job``.

        Works with both Django ``Job`` instances (which expose ``job_id``)
        and the lightweight stand-in objects used by unit tests.
        """
        for attr in ("job_id", "pk", "id"):
            value = getattr(job, attr, None)
            if value is not None:
                return str(value)
        return None


# ---------------------------------------------------------------------------
# Convenience: opportunistic-tick API for the worker retry path.
# ---------------------------------------------------------------------------
_DEFAULT_ENGINE_LOCK = threading.RLock()
_DEFAULT_ENGINE: FailoverEngine | None = None


def set_default_engine(engine: FailoverEngine | None) -> None:
    """Register ``engine`` as the global opportunistic-tick handle.

    The reconciler installs the engine here so worker-side retry hooks in
    ``coordinator.jobs.tasks`` can fire a tick without holding a direct
    reference to the reconciler.
    """
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        _DEFAULT_ENGINE = engine


def get_default_engine() -> FailoverEngine | None:
    with _DEFAULT_ENGINE_LOCK:
        return _DEFAULT_ENGINE


def opportunistic_tick() -> FailoverReport | None:
    """Run a tick on the registered default engine, if any.

    Returns ``None`` when no engine is registered (the common case in
    single-cluster deployments).  Exceptions are caught and logged.
    """
    engine = get_default_engine()
    if engine is None:
        return None
    try:
        return engine.tick()
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning(
            "failover.opportunistic_tick.error",
            extra={"error": str(exc)},
        )
        return None
