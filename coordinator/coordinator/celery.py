import itertools
import os
import threading

from celery import Celery
from kombu import Queue

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coordinator.settings")

app = Celery("coordinator")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.conf.imports = tuple(app.conf.imports or ()) + ("jobs.tasks_translation_batch",)
app.autodiscover_tasks()
# LayoutLM lives in a deliberately separate task module.  Register it through
# Celery's deferred autodiscovery signal so Django's app registry is ready
# before the module imports Job/Worker models.
app.autodiscover_tasks(["jobs"], related_name="tasks_layoutlm")


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Parse an operational integer without allowing expansion-by-config."""
    try:
        value = int(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


# R2-RC-01 / Sweep C C06: HMAC task authenticity (broker ≠ pipeline authority).
# When CELERY_TASK_SIGNING_KEY is set (or production requires it), every
# shared_task inherits SignedTask so apply_async MACs kwargs/args into a message
# header and workers verify before run. Production / required signing must not
# silently drop the base.
import logging as _celery_logging

_celery_log = _celery_logging.getLogger(__name__)
try:
    from jobs.task_signing import SignedTask

    app.Task = SignedTask
except (
    Exception
) as _signed_task_exc:  # pragma: no cover - import-order / unit isolation
    if os.environ.get(
        "DEPLOYMENT_ENV", ""
    ).strip().lower() == "production" or os.environ.get(
        "ENABLE_CELERY_TASK_SIGNING", ""
    ).strip().lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "SignedTask base required but failed to install (C06 fail-closed)"
        ) from _signed_task_exc
    _celery_log.warning("SignedTask not installed (lab only): %s", _signed_task_exc)

# ---------------------------------------------------------------------------
# Per-GPU queue affinity (Phase 7A)
# ---------------------------------------------------------------------------
# When ENABLE_PER_GPU_QUEUES=true and GPU_COUNT > 0, each GPU worker
# subscribes to its own ocr_gpu_{N} queue.  Tasks are dispatched
# round-robin across the per-GPU queues.  When disabled (default),
# behaviour is identical to the shared ocr_gpu queue.
_GPU_COUNT = _bounded_env_int("GPU_COUNT", 0, minimum=0, maximum=256)
_PER_GPU_QUEUES_ENABLED = (
    os.environ.get("ENABLE_PER_GPU_QUEUES", "false").lower() in ("1", "true", "yes")
    and _GPU_COUNT > 0
)

# GPU task round-robin counter (thread-safe via itertools.count)
_gpu_round_robin_counter = itertools.count()


# ---------------------------------------------------------------------------
# Federation cluster router (Plan C Phase 1, item C4)
# ---------------------------------------------------------------------------
# When OCR_FEDERATION_ROUTING_ENABLED=true and a non-trivial peer registry
# is loaded, we consult the cluster router with the queue computed by the
# per-GPU/CPU logic above as the *base* queue. The router may rewrite the
# destination to ``<base_queue>.<peer-cluster-name>`` so the message flows
# through the RabbitMQ federation upstream link to the chosen peer.
# Behaviour with the flag off is identical to pre-C4.
_federation_router = None
_federation_router_lock = threading.Lock()
_federation_router_initialised = False
_federation_queue_lock = threading.Lock()
_OCR_AFFINITY_TASKS = frozenset(
    {
        "jobs.tasks.process_document",
        "jobs.tasks.process_page",
        "jobs.tasks.infer_page",
    }
)


def _declare_federation_destination(queue_name: str) -> None:
    """Declare a dynamic peer queue with the same durability as local queues.

    Federation appends a peer suffix at routing time, so these destinations
    cannot all be listed in the static import-time queue set. In quorum mode
    implicit queue creation is deliberately disabled; explicitly adding the
    selected destination here keeps fail-closed routing without downgrading it
    to a classic queue.
    """
    queues = app.amqp.queues
    if queue_name in queues:
        return
    quorum = os.environ.get("CELERY_USE_QUORUM_QUEUES", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    queue_arguments = {"x-queue-type": "quorum"} if quorum else {"x-max-priority": 10}
    with _federation_queue_lock:
        if queue_name not in queues:
            queues.add(
                Queue(
                    queue_name,
                    routing_key=queue_name,
                    queue_arguments=queue_arguments,
                )
            )


def _get_federation_router():
    """Lazily build the federation router; cache on first call.

    Returns ``None`` when federation routing is disabled. The router
    object is process-local and safe to cache across calls because it
    polls the registry internally.
    """
    global _federation_router, _federation_router_initialised
    if _federation_router_initialised:
        return _federation_router
    with _federation_router_lock:
        if _federation_router_initialised:
            return _federation_router
        try:
            from federation.cluster_router import build_router_from_env

            _federation_router = build_router_from_env()
        except Exception:  # pragma: no cover - defensive
            _federation_router = None
        _federation_router_initialised = True
    return _federation_router


def _maybe_apply_federation_routing(*, name, base_queue, kwargs):
    """Optionally rewrite the destination queue via the cluster router.

    Returns the (possibly rewritten) queue name. When federation routing
    is disabled or the router decides to stay local, ``base_queue`` is
    returned unchanged.
    """
    router = _get_federation_router()
    if router is None:
        return base_queue
    try:
        decision = router.select_cluster(
            task_name=name,
            base_queue=base_queue,
            kwargs=kwargs or {},
        )
    except Exception:  # pragma: no cover - defensive
        return base_queue
    if decision.queue_name != base_queue:
        _declare_federation_destination(decision.queue_name)
    return decision.queue_name


def resolve_ocr_dispatch_queue(name, base_queue, kwargs=None):
    """Resolve current affinity/federation for a durable logical OCR queue.

    B09: shared ``ocr_gpu`` remains the durable contract unless operators
    explicitly enable the strict per-GPU producer/consumer topology.

    Durable outbox rows intentionally persist ``ocr_gpu`` rather than a
    topology-specific ``ocr_gpu_N`` destination. Publication can then adapt
    safely when GPU_COUNT changes between row creation and relay. Legacy
    dedicated rows are normalized through the same path so a scale-down does
    not turn them into undeclared queues and dead-letter otherwise valid work.
    """
    if name not in _OCR_AFFINITY_TASKS:
        return base_queue
    if base_queue.startswith("ocr_gpu_") and base_queue[8:].isdigit():
        base_queue = "ocr_gpu"
    if base_queue == "ocr_gpu" and _PER_GPU_QUEUES_ENABLED:
        strict = os.environ.get("OCR_PER_GPU_STRICT", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if strict:
            idx = next(_gpu_round_robin_counter) % _GPU_COUNT
            base_queue = f"ocr_gpu_{idx}"
    return _maybe_apply_federation_routing(
        name=name,
        base_queue=base_queue,
        kwargs=kwargs or {},
    )


def _route_task(name, args, kwargs, options, task=None, **kw):
    """Route tasks to appropriate queues.

    When per-GPU queues are enabled, OCR inference tasks are distributed
    round-robin across ocr_gpu_{N} queues.
    All other tasks use their standard queue.

    When federation routing is enabled (Plan C C4), the queue computed
    by the per-GPU/CPU logic is fed into the cluster router which may
    rewrite the destination to ``<base_queue>.<peer-cluster>``.
    """
    _STATIC_ROUTES = {
        "jobs.tasks.ingest_document": "coordinator",
        "jobs.tasks.assemble_document": "coordinator",
        "jobs.tasks.dispatch_postprocessing": "coordinator",
        "jobs.tasks.finalize_job": "coordinator",
        "jobs.tasks.check_worker_heartbeats": "coordinator",
        "jobs.tasks.relay_task_dispatch_outbox": "coordinator",
        "jobs.tasks.cleanup_stale_jobs": "coordinator",
        "jobs.tasks.redrive_stale_assembling_jobs": "coordinator",
        "jobs.tasks.cleanup_completed_jobs": "coordinator",
        "jobs.tasks.cleanup_expired_presigned_capabilities": "coordinator",
        "jobs.tasks.republish_pending_custody": "coordinator",
        "jobs.tasks.extract_pages": "coordinator",
        "jobs.tasks.cleanup_page_stage": "coordinator",
        "jobs.tasks.recover_page_stages": "coordinator",
        "jobs.tasks.render_page": "ocr_render",
        "jobs.tasks.finalize_page": "ocr_finalize",
        "jobs.tasks.chord_error_handler": "coordinator",
        "jobs.tasks.compress_pdf": "cpu_general",
        "jobs.tasks.extract_entities": "cpu_general",
        "jobs.tasks.extract_structured_data": "nlp_general",
        "jobs.tasks.process_text_only": "cpu_general",
        # LayoutLM: only route when ENABLE_LAYOUTLM (PH2-17 ghost-queue close).
        # Producer: jobs.tasks.process_page may dispatch after OCR when enabled.
        "jobs.tasks_layoutlm.run_layoutlm_extraction": "ocr_layoutlm",
        "jobs.tasks_translation_batch.translate_batch_input": "translation_batch",
    }

    # PH2-17: refuse ghost ocr_layoutlm route when feature is off.
    if name == "jobs.tasks_layoutlm.run_layoutlm_extraction":
        _ll_on = os.environ.get("ENABLE_LAYOUTLM", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not _ll_on:
            # No queue affinity → falls through to default; task also refuses.
            return {"queue": "coordinator"}

    if name in _STATIC_ROUTES:
        return {"queue": _STATIC_ROUTES[name]}

    if name in _OCR_AFFINITY_TASKS:
        # Capability ownership takes precedence over device affinity.  Only the
        # coordinator may add this signed task kwarg; unknown values never
        # synthesize a queue name and therefore cannot create ghost lanes.
        engine_lane = str((kwargs or {}).get("ocr_engine_lane", "")).strip().lower()
        if engine_lane == "tesseract":
            final_queue = _maybe_apply_federation_routing(
                name=name, base_queue="ocr_tesseract", kwargs=kwargs
            )
            return {"queue": final_queue}

        # Check for explicit CPU routing via OCR_TASK_ROUTING env var.
        # When set to "cpu", OCR tasks go to the ocr_cpu queue instead of
        # ocr_gpu.  When "auto", tasks.py handles dynamic queue selection
        # at dispatch time; the router falls back to ocr_gpu so Celery has
        # a valid default.
        ocr_routing = os.environ.get("OCR_TASK_ROUTING", "gpu").lower().strip()
        if ocr_routing == "cpu":
            base_queue = "ocr_cpu"
        else:
            base_queue = "ocr_gpu"
        final_queue = resolve_ocr_dispatch_queue(
            name=name,
            base_queue=base_queue,
            kwargs=kwargs,
        )
        return {"queue": final_queue}

    return None


app.conf.task_routes = (_route_task,)

# ---------------------------------------------------------------------------
# Queue declarations with priority support (x-max-priority: 10)
# ---------------------------------------------------------------------------
# RabbitMQ uses x-max-priority to enable priority ordering within a queue.
# Celery tasks dispatched with priority=0..9 are dequeued in priority order.
_PRIORITY_QUEUE_ARGS = {"x-max-priority": 10}

# Optional quorum queues for HA (RabbitMQ quorum does not support classic
# x-max-priority on the same queue declaration — queue declaration compatibility).
_QUORUM_QUEUE_ARGS = {"x-queue-type": "quorum"}
if os.environ.get("CELERY_USE_QUORUM_QUEUES", "false").lower() in ("1", "true", "yes"):
    _quorum_queues = [
        Queue(
            "coordinator",
            routing_key="coordinator",
            queue_arguments=dict(_QUORUM_QUEUE_ARGS),
        ),
        Queue(
            "ocr_gpu",
            routing_key="ocr_gpu",
            queue_arguments=dict(_QUORUM_QUEUE_ARGS),
        ),
        Queue(
            "ocr_cpu",
            routing_key="ocr_cpu",
            queue_arguments=dict(_QUORUM_QUEUE_ARGS),
        ),
        Queue(
            "ocr_tesseract",
            routing_key="ocr_tesseract",
            queue_arguments=dict(_QUORUM_QUEUE_ARGS),
        ),
        Queue("ocr_render", routing_key="ocr_render", queue_arguments=dict(_QUORUM_QUEUE_ARGS)),
        Queue("ocr_finalize", routing_key="ocr_finalize", queue_arguments=dict(_QUORUM_QUEUE_ARGS)),
        Queue(
            "cpu_general",
            routing_key="cpu_general",
            queue_arguments=dict(_QUORUM_QUEUE_ARGS),
        ),
        Queue(
            "nlp_general",
            routing_key="nlp_general",
            queue_arguments=dict(_QUORUM_QUEUE_ARGS),
        ),
        Queue(
            "ocr_layoutlm",
            routing_key="ocr_layoutlm",
            queue_arguments=dict(_QUORUM_QUEUE_ARGS),
        ),
        Queue(
            "translation_batch",
            routing_key="translation_batch",
            queue_arguments=dict(_QUORUM_QUEUE_ARGS),
        ),
    ]
    if _PER_GPU_QUEUES_ENABLED:
        _quorum_queues.extend(
            Queue(
                f"ocr_gpu_{i}",
                routing_key=f"ocr_gpu_{i}",
                queue_arguments=dict(_QUORUM_QUEUE_ARGS),
            )
            for i in range(_GPU_COUNT)
        )
    app.conf.task_queues = _quorum_queues
    # An undeclared destination would otherwise be auto-created as a classic
    # queue and silently weaken the quorum durability contract.  Fail closed;
    # every route enabled in this process must be declared above.
    app.conf.task_default_queue = "coordinator"
    app.conf.task_create_missing_queues = False
elif _PER_GPU_QUEUES_ENABLED:
    # Per-GPU queues alongside the shared fallback and non-GPU queues
    _base_queues = [
        Queue(
            "coordinator",
            routing_key="coordinator",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue("ocr_gpu", routing_key="ocr_gpu", queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue("ocr_cpu", routing_key="ocr_cpu", queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue(
            "ocr_tesseract",
            routing_key="ocr_tesseract",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue("ocr_render", routing_key="ocr_render", queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue("ocr_finalize", routing_key="ocr_finalize", queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue(
            "cpu_general",
            routing_key="cpu_general",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue(
            "nlp_general",
            routing_key="nlp_general",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue(
            "ocr_layoutlm",
            routing_key="ocr_layoutlm",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue(
            "translation_batch",
            routing_key="translation_batch",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
    ]
    _gpu_queues = [
        Queue(
            f"ocr_gpu_{i}",
            routing_key=f"ocr_gpu_{i}",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        )
        for i in range(_GPU_COUNT)
    ]
    app.conf.task_queues = _base_queues + _gpu_queues
else:
    # Default queue declarations with priority support
    app.conf.task_queues = [
        Queue(
            "coordinator",
            routing_key="coordinator",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue("ocr_gpu", routing_key="ocr_gpu", queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue("ocr_cpu", routing_key="ocr_cpu", queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue(
            "ocr_tesseract",
            routing_key="ocr_tesseract",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue("ocr_render", routing_key="ocr_render", queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue("ocr_finalize", routing_key="ocr_finalize", queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue(
            "cpu_general",
            routing_key="cpu_general",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue(
            "nlp_general",
            routing_key="nlp_general",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue(
            "ocr_layoutlm",
            routing_key="ocr_layoutlm",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
        Queue(
            "translation_batch",
            routing_key="translation_batch",
            queue_arguments=_PRIORITY_QUEUE_ARGS,
        ),
    ]

# Reliability settings
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True
app.conf.worker_prefetch_multiplier = 1
app.conf.worker_max_tasks_per_child = _bounded_env_int(
    "CELERY_WORKER_MAX_TASKS_PER_CHILD", 50, minimum=1, maximum=10_000
)

# Result backend TTL (maintenance review): prevent unbounded Redis growth from retained
# task results.  Default 24h; override via CELERY_RESULT_EXPIRES (seconds).
_result_expires = _bounded_env_int(
    "CELERY_RESULT_EXPIRES",
    86400,
    minimum=0,
    maximum=30 * 24 * 3600,
)
app.conf.result_expires = _result_expires

# Broker connection resilience (Phase 7D)
app.conf.broker_connection_retry_on_startup = True
app.conf.broker_connection_max_retries = 10
app.conf.broker_transport_options = {
    "confirm_publish": True,
}
