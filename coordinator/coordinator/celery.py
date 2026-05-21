import itertools
import os
import threading

from celery import Celery
from kombu import Queue

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coordinator.settings')

app = Celery('coordinator')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.conf.imports = tuple(app.conf.imports or ()) + (
    'jobs.tasks_translation_batch',
)
app.autodiscover_tasks()

# ---------------------------------------------------------------------------
# Per-GPU queue affinity (Phase 7A)
# ---------------------------------------------------------------------------
# When ENABLE_PER_GPU_QUEUES=true and GPU_COUNT > 0, each GPU worker
# subscribes to its own ocr_gpu_{N} queue.  Tasks are dispatched
# round-robin across the per-GPU queues.  When disabled (default),
# behaviour is identical to the shared ocr_gpu queue.
try:
    _GPU_COUNT = max(0, int(os.environ.get('GPU_COUNT', '0')))
except (ValueError, TypeError):
    _GPU_COUNT = 0
_PER_GPU_QUEUES_ENABLED = (
    os.environ.get('ENABLE_PER_GPU_QUEUES', 'false').lower()
    in ('1', 'true', 'yes')
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


def _maybe_apply_federation_routing(
    *, name, base_queue, kwargs
):
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
    return decision.queue_name


def _route_task(name, args, kwargs, options, task=None, **kw):
    """Route tasks to appropriate queues.

    When per-GPU queues are enabled, GPU tasks (process_document,
    process_page) are distributed round-robin across ocr_gpu_{N} queues.
    All other tasks use their standard queue.

    When federation routing is enabled (Plan C C4), the queue computed
    by the per-GPU/CPU logic is fed into the cluster router which may
    rewrite the destination to ``<base_queue>.<peer-cluster>``.
    """
    _STATIC_ROUTES = {
        'jobs.tasks.ingest_document': 'coordinator',
        'jobs.tasks.assemble_document': 'coordinator',
        'jobs.tasks.finalize_job': 'coordinator',
        'jobs.tasks.check_worker_heartbeats': 'coordinator',
        'jobs.tasks.cleanup_stale_jobs': 'coordinator',
        'jobs.tasks.cleanup_completed_jobs': 'coordinator',
        'jobs.tasks.extract_pages': 'coordinator',
        'jobs.tasks.chord_error_handler': 'coordinator',
        'jobs.tasks.compress_pdf': 'cpu_general',
        'jobs.tasks.extract_entities': 'cpu_general',
        'jobs.tasks.extract_structured_data': 'nlp_general',
        'jobs.tasks.process_text_only': 'cpu_general',
        'jobs.tasks_layoutlm.run_layoutlm_extraction': 'ocr_layoutlm',
        'jobs.tasks_translation_batch.translate_batch_input': 'translation_batch',
    }

    if name in _STATIC_ROUTES:
        return {'queue': _STATIC_ROUTES[name]}

    if name in ('jobs.tasks.process_document', 'jobs.tasks.process_page'):
        # Check for explicit CPU routing via OCR_TASK_ROUTING env var.
        # When set to "cpu", OCR tasks go to the ocr_cpu queue instead of
        # ocr_gpu.  When "auto", tasks.py handles dynamic queue selection
        # at dispatch time; the router falls back to ocr_gpu so Celery has
        # a valid default.
        ocr_routing = os.environ.get('OCR_TASK_ROUTING', 'gpu').lower().strip()
        if ocr_routing == 'cpu':
            base_queue = 'ocr_cpu'
        elif _PER_GPU_QUEUES_ENABLED:
            idx = next(_gpu_round_robin_counter) % _GPU_COUNT
            base_queue = f'ocr_gpu_{idx}'
        else:
            base_queue = 'ocr_gpu'
        final_queue = _maybe_apply_federation_routing(
            name=name, base_queue=base_queue, kwargs=kwargs
        )
        return {'queue': final_queue}

    return None


app.conf.task_routes = (_route_task,)

# ---------------------------------------------------------------------------
# Queue declarations with priority support (x-max-priority: 10)
# ---------------------------------------------------------------------------
# RabbitMQ uses x-max-priority to enable priority ordering within a queue.
# Celery tasks dispatched with priority=0..9 are dequeued in priority order.
_PRIORITY_QUEUE_ARGS = {'x-max-priority': 10}

# Optional quorum queues for HA experiments (Phase 7 kickoff).
if os.environ.get('CELERY_USE_QUORUM_QUEUES', 'false').lower() in ('1', 'true', 'yes'):
    app.conf.task_queues = [
        Queue(
            'coordinator',
            routing_key='coordinator',
            queue_arguments={'x-queue-type': 'quorum', 'x-max-priority': 10},
        ),
        Queue(
            'ocr_gpu',
            routing_key='ocr_gpu',
            queue_arguments={'x-queue-type': 'quorum', 'x-max-priority': 10},
        ),
        Queue(
            'ocr_cpu',
            routing_key='ocr_cpu',
            queue_arguments={'x-queue-type': 'quorum', 'x-max-priority': 10},
        ),
        Queue(
            'cpu_general',
            routing_key='cpu_general',
            queue_arguments={'x-queue-type': 'quorum', 'x-max-priority': 10},
        ),
        Queue(
            'nlp_general',
            routing_key='nlp_general',
            queue_arguments={'x-queue-type': 'quorum', 'x-max-priority': 10},
        ),
        Queue(
            'ocr_layoutlm',
            routing_key='ocr_layoutlm',
            queue_arguments={'x-queue-type': 'quorum', 'x-max-priority': 10},
        ),
        Queue(
            'translation_batch',
            routing_key='translation_batch',
            queue_arguments={'x-queue-type': 'quorum', 'x-max-priority': 10},
        ),
    ]
elif _PER_GPU_QUEUES_ENABLED:
    # Per-GPU queues alongside the shared fallback and non-GPU queues
    _base_queues = [
        Queue('coordinator', routing_key='coordinator',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('ocr_gpu', routing_key='ocr_gpu',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('ocr_cpu', routing_key='ocr_cpu',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('cpu_general', routing_key='cpu_general',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('nlp_general', routing_key='nlp_general',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('ocr_layoutlm', routing_key='ocr_layoutlm',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('translation_batch', routing_key='translation_batch',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
    ]
    _gpu_queues = [
        Queue(f'ocr_gpu_{i}', routing_key=f'ocr_gpu_{i}',
              queue_arguments=_PRIORITY_QUEUE_ARGS)
        for i in range(_GPU_COUNT)
    ]
    app.conf.task_queues = _base_queues + _gpu_queues
else:
    # Default queue declarations with priority support
    app.conf.task_queues = [
        Queue('coordinator', routing_key='coordinator',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('ocr_gpu', routing_key='ocr_gpu',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('ocr_cpu', routing_key='ocr_cpu',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('cpu_general', routing_key='cpu_general',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('nlp_general', routing_key='nlp_general',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('ocr_layoutlm', routing_key='ocr_layoutlm',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
        Queue('translation_batch', routing_key='translation_batch',
              queue_arguments=_PRIORITY_QUEUE_ARGS),
    ]

# Reliability settings
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True
app.conf.worker_prefetch_multiplier = 1
app.conf.worker_max_tasks_per_child = 50

# Result backend TTL: prevent unbounded Redis growth from retained
# task results.  Default 24h; override via CELERY_RESULT_EXPIRES (seconds).
try:
    _result_expires = int(os.environ.get('CELERY_RESULT_EXPIRES', '86400'))
except (ValueError, TypeError):
    _result_expires = 86400
app.conf.result_expires = max(0, _result_expires)

# Broker connection resilience (Phase 7D)
app.conf.broker_connection_retry_on_startup = True
app.conf.broker_connection_max_retries = 10
app.conf.broker_transport_options = {
    'confirm_publish': True,
}
