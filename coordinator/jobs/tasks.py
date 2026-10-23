"""Celery task definitions for the distributed OCR pipeline.

Tasks are organized into capability-owned queue categories:
- coordinator: Job lifecycle management (ingest, assemble, finalize)
- ocr_gpu: GPU-intensive OCR processing (process_document, process_page)
- ocr_cpu: Paddle CPU OCR processing
- ocr_tesseract: Tesseract-only fallback processing
- cpu_general: CPU-bound tasks (compression, entity extraction)

OCR task routing is controlled by the OCR_TASK_ROUTING env var:
- "gpu" (default): Route to ocr_gpu queue
- "cpu": Route to ocr_cpu queue
- "auto": Route to ocr_gpu if GPU workers are online, else ocr_cpu

Task granularity follows a hybrid model:
- Classified PDF/image OCR: durable page admission and leased per-page tasks
- Native text and explicit skip_ocr: text-only processing
"""

import atexit
import hashlib
import hmac
import io
import json
import logging
import os
import shutil
import socket
import stat
import tempfile
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
from functools import lru_cache, partial, wraps

from celery import chord, shared_task
from celery.exceptions import MaxRetriesExceededError
from celery.result import allow_join_result
from django.conf import settings
from django.db import connection, transaction
from django.db import models as db_models
from django.db.models import Count, Exists, F, OuterRef, Q
from django.db.models.functions import Cast
from django.utils import timezone

from ocr_distributed.tenant_purge_fs import remove_path_no_follow, safe_nfs_job_path
from ocr_distributed.text_io import read_text_bounded
from ocr_local.config.language_config import (
    LANGUAGE_REGISTRY,
    get_tesseract_code,
)
from ocr_local.engines import catalog
from ocr_local.engines.adapters.tesseract import TESSERACT_TIMEOUT
from ocr_local.engines.governance import (
    BLOCKED_FORENSIC_MODE,
    EnginePolicy,
    compute_policy_hash,
    evaluate_engine,
)
from ocr_local.engines.provenance import (
    build_engine_provenance_event,
    build_lane_boundary_event,
    build_license_blocked_event,
)
from ocr_local.engines.runtime_pool import (
    EngineKey,
    EngineRuntimePool,
    production_native_call_timeout,
)
from ocr_local.language_routing import (
    LanguageRoute,
    OCRCandidate,
    compare_cross_engine_candidates,
    compare_grounded_word_candidates,
    compare_ocr_candidates,
    probe_image_script_candidates,
    route_language,
    text_supports_script,
)
from ocr_local.native_pdf import assess_native_pdf_page
from ocr_local.ocr_overlay import (
    bounded_source_raster_size,
    compare_source_grounded_text_revision,
    map_rotated_detector_lines_to_source,
    probe_page_reading_direction,
    read_source_tesseract_words,
    tesseract_data_to_lines_with_review,
)
from ocr_local.pdf_geometry import PageGeometry
from ocr_local.visual_coverage import (
    CoverageAssessment,
    assess_candidate_coverage,
    assess_raster_text_regions,
)

from .artifact_generations import (
    ACTIVE_OUTPUT_POINTER_KEY,
    ArtifactPointerError,
    build_active_pointer,
    generation_locations,
    make_generation_id,
    replace_pointer_pdf,
    resolve_active_artifacts,
)
from .extraction_models import DocumentChunk, ExtractedEntity, ExtractedFormValue
from .litigation_hold import is_litigation_hold_active
from .models import (
    CustodyEvent,
    Job,
    PageOCRByteBudget,
    PageOCRRasterRetirement,
    PageOCRStage,
    PageResult,
    PiiEntity,
    PresignedTaskCapability,
    TaskDispatchOutbox,
    Worker,
)
from .outbox import (
    enqueue_task_dispatch,
    publish_task_dispatch_outbox,
    resolve_task_dispatch_queue,
)
from .presigned import (
    create_presigned_capability,
    generate_compress_pdf_urls,
    generate_extract_entities_urls,
    generate_extract_structured_data_urls,
    generate_layoutlm_urls,
    generate_process_page_urls,
    generate_stage_urls,
    generate_text_only_source_urls,
    is_presigned_mode,
    open_presigned_capability,
    open_presigned_urls,
    page_attempt_keys,
    seal_presigned_urls,
    text_only_attempt_identity,
    text_only_attempt_keys,
    validate_stage_capability,
)
from .source_cache import SourceIdentity, lease_presigned_source
from .storage import CachedS3Backend, _open_local_parent, create_storage_backend

logger = logging.getLogger(__name__)

_OCR_ENGINE_POOL = None
_OCR_ENGINE_POOL_PID = None
_OCR_ENGINE_POOL_LOCK = threading.Lock()


def _reset_ocr_pool_after_fork():
    global _OCR_ENGINE_POOL, _OCR_ENGINE_POOL_PID, _OCR_ENGINE_POOL_LOCK
    # An inherited parent lock may be held by a thread that does not exist in
    # the child.  Never acquire or close parent CUDA/predictor state here.
    _OCR_ENGINE_POOL = None
    _OCR_ENGINE_POOL_PID = None
    _OCR_ENGINE_POOL_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_ocr_pool_after_fork)


def _ocr_runtime_pool():
    """Initialize predictor owners only in the process executing OCR."""
    global _OCR_ENGINE_POOL, _OCR_ENGINE_POOL_PID
    pid = os.getpid()
    if _OCR_ENGINE_POOL is not None and _OCR_ENGINE_POOL_PID == pid:
        return _OCR_ENGINE_POOL
    with _OCR_ENGINE_POOL_LOCK:
        if _OCR_ENGINE_POOL is None or _OCR_ENGINE_POOL_PID != pid:
            _OCR_ENGINE_POOL = EngineRuntimePool(
                max_entries=_bounded_operational_int(
                    "OCR_ENGINE_POOL_MAX_ENTRIES", 4, minimum=1, maximum=16
                ),
                max_replicas_per_key=_bounded_operational_int(
                    "OCR_ENGINE_REPLICAS_PER_KEY", 1, minimum=1, maximum=4
                ),
                max_replicas_per_device=_bounded_operational_int(
                    "OCR_INFERENCE_SLOTS_PER_DEVICE", 1, minimum=1, maximum=8
                ),
                max_pending_per_replica=_bounded_operational_int(
                    "OCR_ENGINE_MAX_PENDING", 8, minimum=1, maximum=32
                ),
                native_call_timeout_seconds=production_native_call_timeout(),
            )
            _OCR_ENGINE_POOL_PID = pid
            atexit.register(_close_ocr_runtime_pool, pid)
    return _OCR_ENGINE_POOL


def _close_ocr_runtime_pool(owner_pid):
    if os.getpid() == owner_pid and _OCR_ENGINE_POOL is not None:
        _OCR_ENGINE_POOL.close()


def _run_paddle_ocr(lang_code, device, raster, engine_factory):
    from ocr_local.config.language_config import get_paddle_runtime_code
    from ocr_local.infra.ocr_inference_backend import (
        OCR_DROP_SCORE,
        get_backend,
        get_detection_limit_side_len,
        get_gpu_device_configuration,
        get_native_threads_per_predictor,
        get_recognizer_batch_size,
    )

    if device.startswith("gpu"):
        try:
            import paddle

            get_gpu_device_configuration(device)
            if not paddle.device.is_compiled_with_cuda():
                raise RuntimeError("Paddle is not built with CUDA")
            if paddle.device.cuda.device_count() < 1:
                raise RuntimeError("no visible CUDA device for OCR inference")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("CUDA runtime could not be verified") from exc
        # The constructor selects the requested local GPU on its owner thread.
        # An unrelated caller's ambient default place may still be CPU.
        actual_device = "gpu:0"
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible = f"CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}"
        else:
            visible = f"NVIDIA:{os.environ.get('NVIDIA_VISIBLE_DEVICES', '')}"
        actual_device = f"{actual_device}|visible:{visible}"
    else:
        actual_device = "cpu"
    backend = "paddle" if device.startswith("gpu") else get_backend()
    runtime_language = get_paddle_runtime_code(lang_code)
    key = EngineKey(
        device=actual_device,
        language=runtime_language,
        model_family=runtime_language,
        backend=backend,
        config_identity=(
            device,
            get_detection_limit_side_len(),
            OCR_DROP_SCORE,
            get_native_threads_per_predictor(),
            get_recognizer_batch_size(),
            os.environ.get("PADDLEOCR_HOME", ""),
            os.environ.get("PADDLEOCR_MODEL_DIR", ""),
            os.environ.get("OCR_MODEL_PATH", ""),
            True,  # use_angle_cls
        ),
    )
    return _ocr_runtime_pool().run(
        key,
        lambda: engine_factory(lang_code, device),
        lambda engine, image: engine.ocr(image),
        raster,
    )


def _page_script_evidence(image, job) -> tuple[str, ...]:
    """Bounded OSD probe on the current raster, with no language assertion."""
    if (job.settings_json or {}).get("language_hint"):
        return ()
    if os.environ.get("OCR_PAGE_SCRIPT_PROBE", "true").lower() in {
        "0", "false", "no"
    }:
        return ()
    timeout = _bounded_operational_int(
        "OCR_PAGE_SCRIPT_PROBE_TIMEOUT", 3, minimum=1, maximum=10
    )
    try:
        import pytesseract

        data = pytesseract.image_to_osd(
            image, output_type=pytesseract.Output.DICT, timeout=timeout
        )
        script = str(data.get("script", "")).strip().lower()
        script_confidence = float(data.get("script_conf", 0))
        if script in {"greek", "cyrillic"} and script_confidence >= 1:
            # OSD confuses these two on actual rescue pages; corroborate only
            # this ambiguous case to avoid three OCR probes per ordinary page.
            return probe_image_script_candidates(
                image, timeout_seconds=timeout
            )
        if script in {"greek", "cyrillic"}:
            return ()
        if script and script_confidence >= 10:
            return (script,)
    except Exception as exc:
        logger.debug("Per-page script probe unavailable: %s", exc)
    return ()


def _page_reading_direction(image, geometry, route):
    """CPU-only, source-raster evidence for metadata rotation decisions."""
    rotation = int(getattr(geometry, "source_rotation", 0) or 0)
    if rotation not in (90, 180, 270):
        return {"orientation": 0, "reason": "no_source_rotation"}
    entry = LANGUAGE_REGISTRY.get(route.model_language)
    script = entry.script if entry is not None else ""
    if script not in {"latin", "greek", "cyrillic"}:
        return {"orientation": None, "reason": "script_probe_unsupported"}
    traineddata = get_tesseract_code(route.model_language)
    evidence = probe_page_reading_direction(
        image, rotation, traineddata=traineddata, script=script,
    )
    return {
        **asdict(evidence),
        "reference_text": evidence.reference_text[:4096],
    }


@lru_cache(maxsize=4)
def _cached_language_detector(model_path, pid, device, inode, size, mtime_ns):
    del pid, device, inode, size, mtime_ns
    from ocr_distributed.language import LanguageDetector

    return LanguageDetector(model_path)


def _language_detector_for_model(model_path):
    """Reuse a model only within the worker and exact on-disk revision."""
    try:
        identity = os.stat(model_path)
        key = (
            os.path.realpath(model_path), os.getpid(), identity.st_dev,
            identity.st_ino, identity.st_size, identity.st_mtime_ns,
        )
    except OSError:
        key = (model_path, os.getpid(), 0, 0, 0, 0)
    return _cached_language_detector(*key)


def _fasttext_model_path():
    """Use the baked worker model unless an operator selects another asset."""
    return os.environ.get("FASTTEXT_MODEL_PATH") or "/app/models/lid.176.bin"


def _page_text_language(text):
    """Return a qualified per-page language from actual OCR text, if any."""
    language, confidence = _page_text_language_evidence(text)
    return language if confidence >= 0.7 else None


def _page_text_language_evidence(text):
    """Return bounded candidate evidence; medium confidence is not detection."""
    if len(text.strip()) < 20:
        return None, 0.0
    model_path = _fasttext_model_path()
    try:
        language, confidence = _language_detector_for_model(
            model_path
        ).detect_from_text(text, confidence_threshold=0.4)
        return language, confidence
    except Exception as exc:
        logger.debug("Per-page text language probe unavailable: %s", exc)
    return None, 0.0


def _pin_rls_for_job(job) -> None:
    """Verify the task-scoped connection pin matches the persisted job."""
    try:
        from .rls import (
            _is_postgres,
            get_control_plane_rls_context,
            get_tenant_rls_context,
        )

        if _is_postgres():
            expected = str(getattr(job, "tenant_id", None) or "").strip()
            actual = get_tenant_rls_context()
            blank_tenant_authorized = (
                not expected and get_control_plane_rls_context() == "on"
            )
            tenant_pin_mismatch = bool(expected) and actual != expected
            if (not expected and not blank_tenant_authorized) or tenant_pin_mismatch:
                raise RuntimeError(
                    "worker RLS pin does not match the persisted job tenant"
                )
    except RuntimeError:
        # C05 / S3-14: production worker pin is fail-closed.
        raise
    except Exception:
        logger.debug("RLS pin verification skipped for non-Postgres lab", exc_info=True)


def _tenant_bound_task(func):
    """Wrap a job worker so its tenant pin exists before ORM lookup.

    The public task signature keeps ``tenant_id`` optional for SQLite/manual
    lab calls. Production Postgres execution fails closed when the coordinator
    did not persist and pass the tenant binding.
    """

    @wraps(func)
    def wrapper(self, job_id, *args, **kwargs):
        tenant_id = kwargs.pop("tenant_id", None)
        from .rls import worker_tenant_rls_context

        with worker_tenant_rls_context(tenant_id):
            return func(self, job_id, *args, tenant_id=tenant_id, **kwargs)

    return wrapper


def _tenant_bound_finalize_task(func):
    """Tenant-pin the chord callback whose first argument is header results."""

    @wraps(func)
    def wrapper(self, header_results, job_id, *args, **kwargs):
        tenant_id = kwargs.pop("tenant_id", None)
        from .rls import worker_tenant_rls_context

        with worker_tenant_rls_context(tenant_id):
            return func(
                self,
                header_results,
                job_id,
                *args,
                tenant_id=tenant_id,
                **kwargs,
            )

    return wrapper


def _tenant_bound_chord_error_task(func):
    """Pin the tenant before a flexible Celery errback looks up a job."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        tenant_id = kwargs.pop("tenant_id", None)
        from .rls import worker_tenant_rls_context

        with worker_tenant_rls_context(tenant_id):
            return func(self, *args, tenant_id=tenant_id, **kwargs)

    return wrapper


def _tenant_for_task(job):
    """Return a non-empty tenant kwarg only for tenant-bound jobs."""
    value = str(getattr(job, "tenant_id", None) or "").strip()
    return {"tenant_id": value} if value else {}


try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

# Documents with more pages than this threshold trigger fan-out processing
FANOUT_THRESHOLD = 20
ASSEMBLY_STREAMING_THRESHOLD_PAGES = int(
    os.environ.get("ASSEMBLY_STREAMING_THRESHOLD_PAGES", "1000")
)
ASSEMBLY_CHECKPOINT_INTERVAL_PAGES = int(
    os.environ.get("ASSEMBLY_CHECKPOINT_INTERVAL_PAGES", "250")
)

# Webhook retry delays (seconds) between attempts on transient failure
_WEBHOOK_RETRY_DELAYS = [5, 30, 60]

# RabbitMQ message priority mapping (0-9, higher = more priority)
_CELERY_PRIORITY_MAP = {"urgent": 9, "normal": 5, "low": 1}


def _reject_if_job_terminal(job):
    """Return a skip/cancel payload if the job is already terminal.

    Celery messages must not re-enter OCR/ingest after COMPLETED/FAILED/CANCELLED
    (R2-RC-01 residual: job-state CAS for delayed or redelivered tasks).
    """
    status = job.status
    status_val = getattr(status, "value", status)
    if status_val in (
        Job.Status.COMPLETED,
        Job.Status.FAILED,
        Job.Status.CANCELLED,
        "completed",
        "failed",
        "cancelled",
    ):
        if status_val in (Job.Status.CANCELLED, "cancelled"):
            return {"status": "cancelled", "job_id": str(job.job_id)}
        return {
            "status": "skipped",
            "job_id": str(job.job_id),
            "reason": f"job already {status_val}",
            "message": getattr(job, "error_message", None)
            or f"job was already {status_val}",
        }
    return None


def _reject_stale_dispatch_generation(job, expected_generation):
    """Reject a broker delivery whose durable outbox generation was revoked."""
    if expected_generation is None:
        return None
    try:
        expected = int(expected_generation)
    except (TypeError, ValueError):
        return {
            "status": "stale",
            "job_id": str(job.job_id),
            "reason": "invalid dispatch generation",
        }
    actual = int(job.dispatch_generation or 0)
    if expected != actual:
        return {
            "status": "stale",
            "job_id": str(job.job_id),
            "reason": f"dispatch generation revoked ({expected} != {actual})",
        }
    return None


def _claim_ingest_lease(job, task_id):
    """Atomically claim the submitted-to-ingesting stage for one delivery.

    A redelivered ingest message must not reset an active stage and repeat
    source hashing, uploads, custody events, and downstream publication.
    Stale ingest rows remain visible to the existing stale-job recovery path;
    this helper only prevents a second live delivery from taking ownership.
    """
    token = str(task_id or f"ingest:{job.job_id}")
    updated = (
        Job.objects.filter(
            pk=job.pk,
        )
        .filter(
            db_models.Q(status=Job.Status.SUBMITTED)
            | db_models.Q(status=Job.Status.INGESTING, celery_task_id=token)
        )
        .filter(db_models.Q(celery_task_id="") | db_models.Q(celery_task_id=token))
        .update(
            status=Job.Status.INGESTING,
            started_at=timezone.now(),
            celery_task_id=token,
        )
    )
    return token if updated == 1 else None


class _ProcessingLeaseLost(RuntimeError):
    """Raised when a worker no longer owns the processing state of a job."""


class _OcrFallbackRequired(RuntimeError):
    """Signal that a Paddle-owned page must continue on the fallback lane."""


_OCR_ENGINE_LANES = frozenset({"combined", "paddle", "tesseract"})


def _separate_tesseract_lane_enabled() -> bool:
    """Return whether new work uses page-fenced capability separation."""
    return os.environ.get("OCR_SEPARATE_TESSERACT_LANE", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolve_ocr_engine_lane(requested_lane=None) -> str:
    """Resolve task intent against the executable capability in this worker.

    ``combined`` is the backward-compatible development/rollback shape. A
    capability-specific worker refuses a task addressed to the other engine,
    even if broker ACL or operator error delivers it to the wrong queue.
    """
    requested = str(requested_lane or "combined").strip().lower()
    capability = os.environ.get("OCR_WORKER_CAPABILITY", "combined").strip().lower()
    if requested not in _OCR_ENGINE_LANES:
        raise ValueError(f"unsupported OCR engine lane: {requested!r}")
    if capability not in _OCR_ENGINE_LANES:
        raise RuntimeError(f"invalid OCR worker capability: {capability!r}")
    if capability == "combined":
        return requested
    if requested == "combined":
        return capability
    if requested != capability:
        raise RuntimeError(
            f"OCR task lane {requested!r} is not owned by {capability!r} worker"
        )
    return requested


def _claim_processing_lease(job, task_id):
    """Mint one durable execution nonce, independent of reused broker IDs."""
    delivery_id = str(task_id or "")
    now = timezone.now()
    with transaction.atomic():
        current = Job.objects.select_for_update().get(pk=job.pk)
        if (
            current.status != Job.Status.PROCESSING
            or current.dispatch_generation != job.dispatch_generation
            or current.tenant_id != job.tenant_id
            or current.source_hash != job.source_hash
            or current.source_object_key != job.source_object_key
            or current.source_file != job.source_file
        ):
            return None
        if current.celery_task_id and current.celery_task_id != delivery_id:
            return None
        if (
            current.processing_attempt_token
            and current.processing_attempt_identity == text_only_attempt_identity(current)
            and current.processing_lease_expires_at
            and current.processing_lease_expires_at > now
        ):
            return None
        token = uuid.uuid4().hex
        current.processing_attempt_token = token
        current.processing_lease_expires_at = now + timedelta(
            seconds=_native_processing_lease_seconds(current)
        )
        if not current.celery_task_id:
            current.celery_task_id = delivery_id
        current.processing_attempt_identity = text_only_attempt_identity(current)
        if current.started_at is None:
            current.started_at = now
        current.save(update_fields=[
            "processing_attempt_token", "processing_lease_expires_at",
            "processing_attempt_identity", "celery_task_id", "started_at",
        ])
        job.celery_task_id = current.celery_task_id
        job.processing_attempt_identity = current.processing_attempt_identity
        return token


def _native_processing_lease_seconds(job):
    """Keep native takeover sooner than the job's own processing deadline."""
    configured = _bounded_operational_int(
        "OCR_NATIVE_PROCESSING_LEASE_SECONDS", 600, minimum=20, maximum=1800,
    )
    return min(configured, max(20, _get_job_processing_timeout_minutes(job) * 30))


def _renew_processing_lease(job_id, token, expected_job=None):
    """Refresh the task lease or stop before the next durable write."""
    query = Job.objects.filter(
        pk=job_id,
        status=Job.Status.PROCESSING,
        processing_attempt_token=token,
        processing_attempt_identity=text_only_attempt_identity(expected_job) if expected_job is not None else db_models.F("processing_attempt_identity"),
        processing_lease_expires_at__gt=timezone.now(),
    )
    if expected_job is not None:
        query = query.filter(
            dispatch_generation=expected_job.dispatch_generation,
            tenant_id=expected_job.tenant_id,
            source_hash=expected_job.source_hash,
            source_object_key=expected_job.source_object_key,
            source_file=expected_job.source_file,
        )
    updated = query.update(
        started_at=timezone.now(),
        processing_lease_expires_at=timezone.now() + timedelta(
            seconds=_native_processing_lease_seconds(
                expected_job or Job.objects.get(pk=job_id)
            )
        )
    )
    if not updated:
        raise _ProcessingLeaseLost(f"processing lease lost for job {job_id}")


def _fail_processing_job_if_owned(
    job_id, token, message, *, expected_job=None, worker_hostname="",
    failure_data=None,
):
    """Fail a processing job only while this task still owns its lease.

    Failure handling is itself a durable write.  A worker that lost its lease
    must not overwrite a recovery worker's state (or a terminal state) with a
    late exception from the old delivery.
    """
    with transaction.atomic():
        locked = Job.objects.select_for_update().get(pk=job_id)
        if (
            not token
            or locked.status != Job.Status.PROCESSING
            or locked.processing_attempt_token != token
            or not locked.processing_lease_expires_at
            or locked.processing_lease_expires_at <= timezone.now()
            or locked.processing_attempt_identity != text_only_attempt_identity(locked)
        ):
            return 0
        if expected_job is not None and (
            locked.processing_attempt_identity != text_only_attempt_identity(expected_job)
            or locked.dispatch_generation != expected_job.dispatch_generation
            or locked.tenant_id != expected_job.tenant_id
            or locked.source_hash != expected_job.source_hash
            or locked.source_object_key != expected_job.source_object_key
            or locked.source_file != expected_job.source_file
            or locked.total_pages != expected_job.total_pages
        ):
            return 0
        locked.status = Job.Status.FAILED
        locked.error_message = message
        locked.completed_at = timezone.now()
        locked.processing_attempt_token = ""
        locked.processing_attempt_identity = ""
        locked.processing_lease_expires_at = None
        locked.save(update_fields=[
            "status", "error_message", "completed_at",
            "processing_attempt_token", "processing_attempt_identity",
            "processing_lease_expires_at",
        ])
        event_data = {"error": message}
        if failure_data:
            event_data.update(failure_data)
        _record_custody_event(
            locked,
            locked.source_hash[:16] if locked.source_hash else str(locked.job_id)[:16],
            "processing_failed",
            data=event_data,
            worker_hostname=worker_hostname or socket.gethostname(),
        )
        PresignedTaskCapability.objects.filter(job=locked).delete()
        return 1


def _mark_processing_failure(job, lease_token, message, hostname):
    """Terminalize a processing failure and preserve its custody boundary."""
    return bool(_fail_processing_job_if_owned(
        job.job_id, lease_token, message,
        expected_job=job, worker_hostname=hostname,
    ))


def _claim_page_processing_lease(job, page_num, task_id, document_id, hostname):
    """Claim one fan-out page without serializing the whole document.

    A document-level lease would incorrectly fence legitimate parallel page
    workers.  PageResult is therefore the ownership row for ``process_page``;
    a redelivery may reclaim its own token, while a different live delivery
    cannot overwrite a page already being processed.
    """
    # A redelivery of the same Celery task is a new publication attempt.  Its
    # owner token must never reuse an artifact namespace held by a dead owner.
    token = uuid.uuid4().hex
    page, _ = PageResult.objects.get_or_create(
        job=job,
        page_num=page_num,
        defaults={
            "document_id": document_id,
            "status": "pending",
            "worker_hostname": hostname,
        },
    )
    if page.status not in {"pending", "failed", "processing"}:
        return None
    now = timezone.now()
    updated = (
        PageResult.objects.filter(
            pk=page.pk,
        )
        .filter(
            db_models.Q(status__in=("pending", "failed"))
            | db_models.Q(status="processing", lease_expires_at__lt=now)
        )
        .update(
            document_id=document_id,
            status="processing",
            celery_task_id=token,
            admitted_generation=job.dispatch_generation,
            lease_expires_at=now + timedelta(
                seconds=_stage_lease_seconds() if _staged_page_ocr_enabled() else 1800
            ),
            worker_hostname=hostname,
            needs_review=True,
        )
    )
    return token if updated == 1 else None


def _renew_page_processing_lease(job_id, page_num, token):
    """Verify that the page worker still owns its page row."""
    updated = PageResult.objects.filter(
        job_id=job_id,
        page_num=page_num,
        status="processing",
        celery_task_id=token,
        lease_expires_at__gt=timezone.now(),
    ).update(
        worker_hostname=socket.gethostname(),
        lease_expires_at=timezone.now() + timedelta(
            seconds=_stage_lease_seconds() if _staged_page_ocr_enabled() else 1800
        ),
    )
    if not updated:
        raise _ProcessingLeaseLost(
            f"processing lease lost for job {job_id} page {page_num}"
        )
    parent_updated = Job.objects.filter(
        pk=job_id,
        status__in=(Job.Status.PROCESSING, Job.Status.ASSEMBLING),
    ).update(started_at=timezone.now())
    if not parent_updated:
        raise _ProcessingLeaseLost(
            f"parent processing lease lost for job {job_id} page {page_num}"
        )


def _commit_page_result_if_owned(job, page_num, lease_token, page_defaults):
    """Publish a terminal page only while its parent identity is current.

    The preflight lease renewal intentionally precedes the transaction: it
    cannot hold a PageResult row lock while waiting for the Job lock. The
    decisive parent check and page CAS then share one transaction, locking
    Job first. Cancellation or generation replacement that commits before
    this lock is rejected; one that waits for it observes the completed page.
    """
    _renew_page_processing_lease(job.job_id, page_num, lease_token)
    with transaction.atomic():
        parent = Job.objects.select_for_update().get(pk=job.pk)
        if (
            parent.status not in (Job.Status.PROCESSING, Job.Status.ASSEMBLING)
            or parent.dispatch_generation != job.dispatch_generation
            or parent.tenant_id != job.tenant_id
            or parent.source_hash != job.source_hash
            or parent.source_object_key != job.source_object_key
            or parent.source_file != job.source_file
        ):
            raise _ProcessingLeaseLost(
                f"parent identity changed for job {job.job_id} page {page_num}"
            )
        updated = PageResult.objects.filter(
            job_id=parent.pk,
            page_num=page_num,
            status="processing",
            celery_task_id=lease_token,
            admitted_generation=parent.dispatch_generation,
            lease_expires_at__gt=timezone.now(),
        ).update(**page_defaults)
        if updated != 1:
            raise _ProcessingLeaseLost(
                f"processing lease lost for job {job.job_id} page {page_num}"
            )


def _get_celery_priority(job) -> int:
    """Return the RabbitMQ message priority for a job."""
    return _CELERY_PRIORITY_MAP.get(getattr(job, "priority", "normal"), 5)


def _format_processing_timeout(minutes: int) -> str:
    unit = "minute" if minutes == 1 else "minutes"
    return f"{minutes} {unit}"


def _get_job_processing_timeout_minutes(job) -> int:
    """Return the per-job timeout override or the coordinator default."""
    max_minutes = max(
        1,
        min(
            10_080,
            int(getattr(settings, "MAX_PROCESSING_TIMEOUT_MINUTES", 10_080)),
        ),
    )
    default_minutes = min(
        max_minutes,
        max(1, int(getattr(settings, "JOB_PROCESSING_TIMEOUT_MINUTES", 30))),
    )
    raw_value = (job.settings_json or {}).get("processing_timeout_minutes")
    if raw_value is None:
        return default_minutes
    try:
        minutes = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid processing_timeout_minutes for job %s: %r; using default %d",
            job.job_id,
            raw_value,
            default_minutes,
        )
        return default_minutes
    if minutes < 1:
        logger.warning(
            "Non-positive processing_timeout_minutes for job %s: %r; using default %d",
            job.job_id,
            raw_value,
            default_minutes,
        )
        return default_minutes
    if minutes > max_minutes:
        logger.warning(
            "Capping processing_timeout_minutes for job %s: %r -> %d",
            job.job_id,
            raw_value,
            max_minutes,
        )
        return max_minutes
    return minutes


# ---------------------------------------------------------------------------
# Plan C Phase 1, item C5 -- federation failover hooks.
#
# These helpers fire opportunistically from the OCR task retry path.  They
# are no-ops when the failover engine is not registered (the default
# single-cluster mode) or when ``OCR_FEDERATION_FAILOVER_ENABLED`` is
# false.  All exception paths swallow errors -- the failover side channel
# must never break the underlying OCR job.
# ---------------------------------------------------------------------------
DEFAULT_FAILOVER_RUNNING_TIMEOUT_SECONDS = 600


def _get_failover_running_timeout_seconds() -> int:
    """Return the soft-time-limit threshold for cluster-unhealthy failover.

    Read from ``OCR_FEDERATION_FAILOVER_RUNNING_TIMEOUT_SECONDS`` so
    operators can tune the window without redeploying code.
    """
    raw = os.environ.get("OCR_FEDERATION_FAILOVER_RUNNING_TIMEOUT_SECONDS")
    if not raw:
        return DEFAULT_FAILOVER_RUNNING_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_FAILOVER_RUNNING_TIMEOUT_SECONDS
    return max(1, value)


def _trigger_failover_opportunistically(
    job, *, reason: str = "cluster_unhealthy"
) -> None:
    """Mark a job's last-failure-reason and kick the registered failover engine.

    The federation engine lives in ``coordinator.federation.failover`` and
    keeps a process-global handle so this hook can fire without a direct
    reference.  When no engine is registered (the common case) the call
    is a cheap no-op.
    """
    try:
        # Best-effort: record the failure reason so operators can see
        # why a job stalled before the failover engine reroutes it.
        if job is not None:
            try:
                job.last_failure_reason = reason
                job.save(update_fields=["last_failure_reason"])
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to record last_failure_reason on job %s",
                    getattr(job, "job_id", "?"),
                )

        from coordinator.federation.failover import opportunistic_tick

        report = opportunistic_tick()
        if report is not None:
            logger.info(
                "Federation failover opportunistic tick: %s",
                report.as_dict(),
            )
    except ImportError:  # pragma: no cover - federation pkg always present
        return
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Federation failover hook errored: %s", exc)


def _is_running_too_long_on_unhealthy_cluster(job) -> bool:
    """Return True if ``job`` has been running longer than the soft limit.

    Used as a gate from inside the worker retry path: if a job has been
    running long enough on its currently-assigned cluster AND that cluster
    is no longer healthy, the failover engine should be invoked.  The
    actual health lookup is delegated to the engine itself; this helper
    only checks the elapsed-time portion of the gate.
    """
    if job is None:
        return False
    started_at = getattr(job, "started_at", None)
    if started_at is None:
        return False
    threshold = _get_failover_running_timeout_seconds()
    try:
        elapsed = (timezone.now() - started_at).total_seconds()
    except Exception:  # pragma: no cover - defensive
        return False
    return elapsed >= threshold


def _get_ocr_queue(ocr_engine_lane=None):
    """Determine the stable logical OCR queue for a durable outbox row.

    Reads the OCR_TASK_ROUTING environment variable:
    - "gpu" (default): Route to ocr_gpu queue
    - "cpu": Route to ocr_cpu queue
    - "auto": Route to ocr_gpu if GPU workers are online, else ocr_cpu

    Per-GPU affinity is intentionally resolved at broker publication time,
    not persisted here. This keeps pending rows valid when GPU_COUNT changes.

    Returns:
        Queue name string for Celery task dispatch.
    """
    if str(ocr_engine_lane or "").strip().lower() == "tesseract":
        return "ocr_tesseract"
    routing = os.environ.get("OCR_TASK_ROUTING", "gpu").lower().strip()
    if routing == "cpu":
        return "ocr_cpu"
    if routing == "auto":
        # Check if any GPU workers are online
        gpu_online = Worker.objects.filter(
            gpu_available=True,
            status__in=[Worker.Status.ONLINE, Worker.Status.BUSY],
        ).exists()
        return "ocr_gpu" if gpu_online else "ocr_cpu"
    return "ocr_gpu"


def _select_initial_ocr_engine_lane(job) -> str:
    """Choose an admitted initial capability without executing either engine."""
    policy = _engine_policy_from_job_settings(job.settings_json)
    route = route_language(
        job.detected_language,
        explicit_hint=(job.settings_json or {}).get("language_hint"),
    )
    paddle = evaluate_engine(catalog.PADDLE_CTC, policy)
    tesseract = evaluate_engine(catalog.TESSERACT, policy)
    if (route.engine == "tesseract" or not paddle.admitted) and tesseract.admitted:
        return "tesseract"
    return "paddle"


_storage_backend_instance = None
_storage_backend_lock = threading.Lock()


def _create_storage_backend():
    """Create a new storage backend instance from Django settings.

    When using S3, wraps the backend in CachedS3Backend to avoid
    redundant downloads across fan-out page processing tasks.
    """
    backend_name = getattr(settings, "STORAGE_BACKEND", "nfs")
    backend = create_storage_backend(
        backend_name=backend_name,
        nfs_root=settings.NFS_ROOT,
        s3_endpoint=getattr(settings, "S3_ENDPOINT", ""),
        s3_bucket=getattr(settings, "S3_BUCKET", ""),
        s3_access_key=getattr(settings, "S3_ACCESS_KEY", ""),
        s3_secret_key=getattr(settings, "S3_SECRET_KEY", ""),
        s3_region=getattr(settings, "S3_REGION", ""),
        s3_auth_mode=getattr(settings, "S3_AUTH_MODE", "static"),
        s3_session_token=getattr(settings, "S3_SESSION_TOKEN", ""),
        presigned_only_worker=getattr(settings, "S3_PRESIGNED_ONLY_WORKER", False),
    )
    if backend_name.strip().lower() == "s3":
        cache_dir = getattr(settings, "S3_CACHE_DIR", "/tmp/ocr-cache")
        max_gb = getattr(settings, "S3_CACHE_MAX_SIZE_GB", 1)
        backend = CachedS3Backend(
            inner=backend,
            cache_dir=cache_dir,
            max_size_bytes=int(max_gb * 1024**3),
            max_object_bytes=int(
                getattr(settings, "S3_CACHE_MAX_OBJECT_GB", 0.5) * 1024**3
            ),
        )
    return backend


def _get_storage_backend():
    """Return the module-level storage backend singleton.

    Uses double-checked locking for thread-safe lazy initialization.
    The backend is created once per worker process and reused across
    all task invocations, preventing redundant S3 downloads when
    processing fan-out page tasks.
    """
    global _storage_backend_instance
    if _storage_backend_instance is None:
        with _storage_backend_lock:
            if _storage_backend_instance is None:
                _storage_backend_instance = _create_storage_backend()
    return _storage_backend_instance


def _reset_storage_backend():
    """Reset the storage backend singleton (for testing only)."""
    global _storage_backend_instance
    with _storage_backend_lock:
        _storage_backend_instance = None


def _get_backend_for_job(job):
    """Get storage backend, honoring the backend locked at ingest time.

    If the job has a ``storage_backend_used`` value and it differs from the
    current ``STORAGE_BACKEND`` setting, log a warning and return a backend
    matching the original ingest backend so artifacts are found in the
    correct location.
    """
    backend = _get_storage_backend()
    persisted_settings = (
        job.settings_json if isinstance(job.settings_json, dict) else {}
    )
    persisted_fingerprint = persisted_settings.get("storage_binding_fingerprint")
    production = os.environ.get("DEPLOYMENT_ENV", "").strip().lower() in {
        "production",
        "prod",
    }
    if job.storage_backend_used and not persisted_fingerprint and production:
        raise RuntimeError(
            f"Job {job.job_id} has no immutable storage binding fingerprint"
        )
    if job.storage_backend_used and persisted_fingerprint:
        current_fingerprint = _storage_binding_fingerprint(job.storage_backend_used)
        if persisted_fingerprint != current_fingerprint:
            raise RuntimeError(
                f"Storage binding changed for job {job.job_id}; refusing to read a "
                "different NFS root or S3 endpoint/bucket"
            )
    if job.storage_backend_used and backend.backend_name != job.storage_backend_used:
        logger.warning(
            "Backend mismatch for job %s: ingested=%s, current=%s. Using ingested backend.",
            job.job_id,
            job.storage_backend_used,
            backend.backend_name,
        )
        try:
            backend = create_storage_backend(
                backend_name=job.storage_backend_used,
                nfs_root=settings.NFS_ROOT,
                s3_endpoint=getattr(settings, "S3_ENDPOINT", ""),
                s3_bucket=getattr(settings, "S3_BUCKET", ""),
                s3_access_key=getattr(settings, "S3_ACCESS_KEY", ""),
                s3_secret_key=getattr(settings, "S3_SECRET_KEY", ""),
                s3_region=getattr(settings, "S3_REGION", ""),
                s3_auth_mode=getattr(settings, "S3_AUTH_MODE", "static"),
                s3_session_token=getattr(settings, "S3_SESSION_TOKEN", ""),
                presigned_only_worker=getattr(settings, "S3_PRESIGNED_ONLY_WORKER", False),
            )
        except Exception:
            logger.exception(
                "Failed to create the persisted %s backend for job %s; refusing "
                "to fall back to the current storage configuration",
                job.storage_backend_used,
                job.job_id,
            )
            raise RuntimeError(
                f"Persisted storage backend {job.storage_backend_used!r} "
                f"cannot be created for job {job.job_id}"
            ) from None
    return backend


def _storage_binding_fingerprint(backend_name: str) -> str:
    """Fingerprint the non-secret storage location bound to a job.

    ``storage_backend_used`` alone is insufficient: changing an NFS mount or
    S3 bucket while a job is in flight can make a later worker read unrelated
    bytes.  Credentials are intentionally excluded so normal credential
    rotation does not orphan a job.
    """
    normalized = str(backend_name or "").strip().lower()
    binding = {"backend": normalized}
    if normalized == "nfs":
        binding["root"] = os.path.realpath(
            os.path.abspath(os.path.expanduser(str(getattr(settings, "NFS_ROOT", ""))))
        )
    elif normalized == "s3":
        binding.update(
            {
                "endpoint": str(getattr(settings, "S3_ENDPOINT", "") or "").strip(),
                "bucket": str(getattr(settings, "S3_BUCKET", "") or "").strip(),
                "region": str(getattr(settings, "S3_REGION", "") or "").strip(),
            }
        )
    else:
        raise RuntimeError(f"Unsupported storage backend for binding: {backend_name!r}")
    return hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _job_storage_key(job_id, subpath=""):
    """Return storage key for job artifacts."""
    if subpath:
        return f"jobs/{job_id}/{subpath}"
    return f"jobs/{job_id}"


_PRESIGNED_PHASE_KEYS = {
    "render_page": {
        "input": ("source_get",),
        "output": ("raster_put", "raster_delete"),
    },
    "infer_page": {
        "input": ("raster_get",),
    },
    "finalize_page": {
        "input": ("source_get", "raster_get"),
        "output": (
            "raster_delete", "page_pdf_put", "page_text_put",
            "page_pdf_delete", "page_text_delete",
            "candidate_evidence_put", "candidate_evidence_delete",
            "page_pdf_key", "page_text_key", "candidate_evidence_key",
        ),
    },
    "process_page": {
        "input": ("source_get",),
        "output": (
            "page_pdf_put",
            "page_text_put",
            "page_pdf_delete",
            "page_text_delete",
            "candidate_evidence_put",
            "candidate_evidence_delete",
            "page_pdf_key", "page_text_key", "candidate_evidence_key",
        ),
    },
    "process_document": {
        "input": ("source_get",),
        "output": (
            "page_pdf_put",
            "page_text_put",
            "page_pdf_delete",
            "page_text_delete",
            "candidate_evidence_put",
            "candidate_evidence_delete",
        ),
    },
    "process_text_only": {
        "input": ("source_get",),
        "output": (
            "page_pdf_put",
            "page_text_put",
            "page_pdf_delete",
            "page_text_delete",
            "page_pdf_key",
            "page_text_key",
        ),
    },
    "layoutlm": {
        "input": ("source_get", "text_get", "text_key"),
        "output": ("entities_put", "entities_delete"),
    },
    "compress_pdf": {
        "input": ("pdf_get",),
        "output": (
            "pdf_put",
            "pdf_delete",
            "generation_pdf_put",
            "generation_pdf_delete",
        ),
    },
    "extract_entities": {
        "input": ("text_get",),
        "output": ("ner_put", "ner_delete"),
    },
    "extract_structured_data": {
        "input": ("text_get",),
        "output": ("ext_put", "ext_delete"),
    },
}


@shared_task(
    bind=True,
    name="jobs.tasks.issue_presigned_urls",
    queue="coordinator",
    acks_late=True,
    reject_on_worker_lost=True,
    # Celery logs a successful task's return value at INFO. The actual sealed
    # result must reach the caller, but even encrypted bearer grants must not
    # be copied into logs/events. saferepr(dict, 0) emits no map values.
    resultrepr_maxsize=0,
)
@_tenant_bound_task
def issue_presigned_urls(
    self,
    job_id,
    capability_id,
    operation,
    phase,
    page_num=None,
    tenant_id=None,
    lease_token=None,
    raster_sha256=None,
    page_lease_token=None,
):
    """Issue a sealed bearer-URL bundle immediately before one I/O phase.

    Worker messages carry only a revocable job/operation capability.  The
    coordinator, which owns S3 signing authority, validates it and returns an
    encrypted result.  Queue residence and long inference therefore consume
    capability lifetime, never the short bearer-URL lifetime.
    """
    del self, tenant_id
    job = Job.objects.get(job_id=job_id)
    _pin_rls_for_job(job)
    if job.status in _TERMINAL_JOB_STATUSES:
        raise RuntimeError("presigned URL issuance refused for terminal job")
    # Validate exact job, operation, revocation, and opaque-capability expiry.
    if operation not in {"render_page", "infer_page", "finalize_page"}:
        open_presigned_capability(str(job_id), capability_id, operation)
    phases = _PRESIGNED_PHASE_KEYS.get(operation)
    if phases is None or phase not in phases:
        raise ValueError("unsupported presigned operation phase")

    backend = _get_backend_for_job(job)
    if backend.backend_name != "s3":
        raise RuntimeError("presigned URL issuance requires the S3 backend")
    document_id = job.source_hash[:16] if job.source_hash else str(job.job_id)[:16]
    if operation in {"render_page", "infer_page", "finalize_page"}:
        try:
            requested_page = int(page_num)
        except (TypeError, ValueError) as exc:
            raise ValueError("staged presigned issuance requires page_num") from exc
        stage = PageOCRStage.objects.get(
            job=job, page_num=requested_page,
            generation=job.dispatch_generation,
        )
        validate_stage_capability(
            job, stage, capability_id, operation,
            lease_token=lease_token, raster_sha256=raster_sha256,
        )
        urls = generate_stage_urls(
            backend, job, stage, operation, document_id,
            page_lease_token=page_lease_token,
        )
    elif operation == "process_text_only" and phase == "input":
        urls = generate_text_only_source_urls(backend, job)
    elif operation in {"process_page", "process_document", "process_text_only"}:
        try:
            requested_page = int(page_num)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "page-scoped presigned issuance requires page_num"
            ) from exc
        if requested_page < 1 or requested_page > int(job.total_pages or 1):
            raise ValueError("presigned page number is outside the job boundary")
        if operation == "process_page" and phase == "output" and not page_lease_token:
            raise RuntimeError("page output grant requires a live page lease")
        if operation == "process_text_only" and phase == "output" and not lease_token:
            raise RuntimeError("native output grant requires a live document lease")
        urls = generate_process_page_urls(
            backend,
            job,
            requested_page,
            document_id,
            page_lease_token=page_lease_token if operation == "process_page" else None,
            document_lease_token=lease_token if operation == "process_text_only" and phase == "output" else None,
        )
    elif operation == "layoutlm":
        try:
            requested_page = int(page_num)
        except (TypeError, ValueError) as exc:
            raise ValueError("LayoutLM presigned issuance requires page_num") from exc
        if requested_page < 1 or requested_page > int(job.total_pages or 1):
            raise ValueError("LayoutLM page number is outside the job boundary")
        urls = generate_layoutlm_urls(backend, job, requested_page, document_id)
    elif operation == "compress_pdf":
        urls = generate_compress_pdf_urls(backend, job)
    elif operation == "extract_entities":
        urls = generate_extract_entities_urls(backend, job)
    else:
        urls = generate_extract_structured_data_urls(backend, job)

    selected = {key: urls[key] for key in phases[phase] if key in urls}
    if operation in {"process_page", "render_page", "finalize_page"} and phase == "input":
        if not isinstance(backend, CachedS3Backend):
            raise RuntimeError("presigned page source requires immutable source cache")
        source_key = _job_storage_key(
            job_id, f"source/{os.path.basename(job.source_file)}"
        )
        revision, size = backend.source_revision_info(source_key)
        selected["source_revision_token"] = revision
        selected["source_size_bytes"] = str(size)
    if not selected:
        raise RuntimeError("presigned URL issuance produced an empty phase bundle")
    return seal_presigned_urls(selected)


def _request_fresh_presigned_urls(
    job,
    capability_id,
    operation,
    phase,
    *,
    page_num=None,
    lease_token=None,
    raster_sha256=None,
    page_lease_token=None,
):
    """Synchronously exchange an opaque capability for one short-lived bundle."""
    if capability_id is None:
        return None
    timeout = max(
        5,
        min(
            120,
            int(getattr(settings, "S3_PRESIGNED_ISSUE_TIMEOUT_SECONDS", 30)),
        ),
    )
    kwargs = _tenant_for_task(job)
    if lease_token is not None:
        kwargs["lease_token"] = lease_token
    if raster_sha256 is not None:
        kwargs["raster_sha256"] = raster_sha256
    if page_lease_token is not None:
        kwargs["page_lease_token"] = page_lease_token
    args = [
        str(job.job_id),
        str(capability_id),
        operation,
        phase,
        page_num,
    ]
    if getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False):
        result = issue_presigned_urls.apply(args=args, kwargs=kwargs)
    else:
        result = issue_presigned_urls.apply_async(
            args=args,
            kwargs=kwargs,
            queue="coordinator",
        )
    try:
        with allow_join_result():
            sealed = result.get(
                timeout=timeout,
                propagate=True,
                disable_sync_subtasks=False,
            )
        return open_presigned_urls(sealed)
    finally:
        try:
            result.forget()
        except Exception:
            logger.debug("Could not forget presigned issuance result", exc_info=True)


def _nfs_job_path(job_id):
    """Return the canonical local workspace directory for a job."""
    return os.path.join(settings.NFS_ROOT, "jobs", str(job_id))


def _nfs_job_path_for_job(job):
    """Resolve a persisted NFS path only when it is bound to this job.

    Production workers fail closed on a poisoned ``nfs_job_path`` row.  The
    explicit DEBUG fallback preserves the historical temporary-directory test
    fixtures while making that compatibility visible and non-production-only.
    """
    canonical = _nfs_job_path(job.job_id)
    raw_path = getattr(job, "nfs_job_path", None)
    if not raw_path:
        return canonical
    safe_path = safe_nfs_job_path(
        raw_path,
        getattr(settings, "NFS_ROOT", None),
        job_id=str(job.job_id),
    )
    if safe_path is not None:
        return str(safe_path)
    deployment_env = (
        str(
            os.environ.get(
                "DEPLOYMENT_ENV",
                getattr(settings, "DEPLOYMENT_ENV", "development"),
            )
            or "development"
        )
        .strip()
        .lower()
    )
    if deployment_env not in {"production", "prod"}:
        logger.warning(
            "Using legacy unbound nfs_job_path for non-production job=%s; "
            "production workers refuse this value",
            job.job_id,
        )
        return str(raw_path)
    raise RuntimeError("Persisted nfs_job_path is not canonical for this job")


def _ensure_job_dirs(job_path):
    """Create the standard NFS directory layout for a job."""
    for subdir in [
        "source",
        "temp",
        "output/EXPORT/CLASSIFICATION",
        "output/EXPORT/CUSTODY",
        "output/EXPORT/EXTRACTION",
        "output/EXPORT/HANDWRITING",
        "output/EXPORT/NER",
        "output/EXPORT/PDF",
        "output/EXPORT/STRUCTURE",
        "output/EXPORT/TEXT",
        "output/EXPORT/VALIDATION",
    ]:
        os.makedirs(os.path.join(job_path, subdir), exist_ok=True)


def _mkdir_tree_no_follow(path: str, root: str) -> None:
    """Create a directory tree while keeping every component under *root*.

    NFS job directories are shared with workers and may contain operator- or
    tenant-controlled names.  ``os.makedirs`` follows a replaced ancestor;
    descriptor-relative opens keep the root and each descendant pinned for the
    complete create operation.
    """
    root_abs = os.path.realpath(os.path.abspath(os.path.expanduser(root)))
    target_abs = os.path.abspath(path)
    deployment_env = (
        str(
            os.environ.get(
                "DEPLOYMENT_ENV", getattr(settings, "DEPLOYMENT_ENV", "development")
            )
        )
        .strip()
        .lower()
    )
    try:
        relative = os.path.relpath(target_abs, root_abs)
    except ValueError as exc:
        raise RuntimeError("NFS job path is on a different filesystem root") from exc
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        if deployment_env in {"production", "prod"}:
            raise RuntimeError("NFS job path escapes NFS_ROOT")
        # Development fixtures may deliberately use a temporary job root
        # outside the configured shared root.  Bind the descriptor walk to
        # the nearest existing fixture ancestor, while production remains
        # strictly NFS_ROOT-contained.
        fixture_root = target_abs
        while not os.path.isdir(fixture_root):
            parent = os.path.dirname(fixture_root)
            if parent == fixture_root:
                raise RuntimeError("NFS fixture root is unavailable")
            fixture_root = parent
        root_abs = os.path.realpath(fixture_root)
        relative = os.path.relpath(target_abs, root_abs)

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    current_fd = os.open(root_abs, flags)
    try:
        if relative == os.curdir:
            return
        for component in relative.split(os.sep):
            if component in {"", os.curdir}:
                continue
            if component == os.pardir:
                raise RuntimeError("NFS job path contains a parent traversal")
            try:
                os.mkdir(component, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            child_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = child_fd
    finally:
        os.close(current_fd)


def _ensure_nfs_job_dirs(job_path: str) -> None:
    """Create the standard NFS layout without following symlink ancestors."""
    root = getattr(settings, "NFS_ROOT", "")
    _mkdir_tree_no_follow(job_path, root)
    for subdir in [
        "source",
        "temp",
        "output/EXPORT/CLASSIFICATION",
        "output/EXPORT/CUSTODY",
        "output/EXPORT/EXTRACTION",
        "output/EXPORT/HANDWRITING",
        "output/EXPORT/NER",
        "output/EXPORT/PDF",
        "output/EXPORT/STRUCTURE",
        "output/EXPORT/TEXT",
        "output/EXPORT/VALIDATION",
    ]:
        _mkdir_tree_no_follow(os.path.join(job_path, subdir), root)


def _record_custody_event(job, document_id, event_type, data=None, worker_hostname=""):
    """Record a custody event with immediate hash linking (R4-RC-01).

    Events are sealed at insert time against the previous tip hash so finalize
    cannot re-derive history from mutable DB rows. Finalize only verifies and
    exports sealed envelopes.
    """
    import hashlib
    import json

    bounded_data = data or {}
    encoded_data = json.dumps(bounded_data, sort_keys=True, default=str).encode("utf-8")
    if len(encoded_data) > 256 * 1024:
        raise ValueError("custody event data exceeds the 256 KiB bound")

    # The previous implementation selected the tip and inserted the next row
    # in separate, unlocked operations.  Page tasks run concurrently, so two
    # writers could both observe the same tip and create a forked chain.  Lock
    # the job row as the per-job/document chain mutex; all custody writers for
    # one job serialize while unrelated jobs remain concurrent.
    with transaction.atomic():
        locked_job = Job.objects.select_for_update().get(pk=job.pk)
        now = timezone.now()
        prev = (
            CustodyEvent.objects.filter(
                job=locked_job,
                document_id=document_id,
            )
            .order_by("-id")
            .only("event_hash", "chain_id")
            .first()
        )
        prev_hash = (prev.event_hash if prev and prev.event_hash else None) or None
        chain_id = (
            str(prev.chain_id or "").strip() if prev is not None else ""
        ) or hashlib.sha256(
            f"job\0{locked_job.job_id}\0{document_id}".encode("utf-8")
        ).hexdigest()
        event_dict = {
            "chain_id": chain_id,
            "hash_version": 2,
            "document_id": document_id,
            "event_type": event_type,
            "timestamp": now.isoformat(timespec="milliseconds"),
            "data": bounded_data,
            "prev_hash": prev_hash,
        }
        event_hash = hashlib.sha256(
            json.dumps(event_dict, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return CustodyEvent.objects.create(
            document_id=document_id,
            job=locked_job,
            chain_id=chain_id,
            hash_version=2,
            event_type=event_type,
            timestamp=now,
            worker_hostname=worker_hostname,
            data=bounded_data,
            prev_hash=prev_hash or "",
            event_hash=event_hash,
            chain_finalized=False,
        )


def _record_control_plane_custody_event(
    *, tenant_id, document_id, event_type, data=None, worker_hostname=""
):
    """Seal a jobless tenant/control-plane event into its own hash chain."""
    bounded_tenant = str(tenant_id or "").strip()
    bounded_document = str(document_id or "").strip()
    if not bounded_tenant or len(bounded_tenant) > 128:
        raise ValueError("control-plane custody requires a bounded tenant_id")
    if not bounded_document or len(bounded_document) > 64:
        raise ValueError("control-plane custody requires a bounded document_id")
    bounded_data = data or {}
    encoded_data = json.dumps(bounded_data, sort_keys=True, default=str).encode("utf-8")
    if len(encoded_data) > 256 * 1024:
        raise ValueError("custody event data exceeds the 256 KiB bound")

    with transaction.atomic():
        if connection.vendor == "postgresql":
            lock_key = hashlib.sha256(
                f"{bounded_tenant}\0{bounded_document}".encode("utf-8")
            ).digest()[:8]
            signed_key = int.from_bytes(lock_key, "big", signed=True)
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", [signed_key])
        now = timezone.now()
        prev = (
            CustodyEvent.objects.select_for_update()
            .filter(
                job__isnull=True,
                tenant_id=bounded_tenant,
                document_id=bounded_document,
            )
            .order_by("-id")
            .only("event_hash", "chain_id")
            .first()
        )
        prev_hash = (prev.event_hash if prev and prev.event_hash else None) or None
        chain_id = (
            str(prev.chain_id or "").strip() if prev is not None else ""
        ) or hashlib.sha256(
            f"control\0{bounded_tenant}\0{bounded_document}".encode("utf-8")
        ).hexdigest()
        event_dict = {
            "chain_id": chain_id,
            "hash_version": 2,
            "document_id": bounded_document,
            "event_type": event_type,
            "timestamp": now.isoformat(timespec="milliseconds"),
            "data": bounded_data,
            "prev_hash": prev_hash,
        }
        event_hash = hashlib.sha256(
            json.dumps(event_dict, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return CustodyEvent.objects.create(
            document_id=bounded_document,
            job=None,
            tenant_id=bounded_tenant,
            chain_id=chain_id,
            hash_version=2,
            event_type=event_type,
            timestamp=now,
            worker_hostname=worker_hostname,
            data=bounded_data,
            prev_hash=prev_hash or "",
            event_hash=event_hash,
            # Jobless control-plane events have no later per-job artifact
            # publication phase. Their insert is the durable finalization;
            # leaving this false makes a correctly sealed row permanently
            # ineligible for verified audit-log rotation.
            chain_finalized=True,
        )


def _tuple_policy_value(value):
    """Normalize settings JSON scalars/lists into EnginePolicy tuple fields."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _int_policy_value(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_policy_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


_ENGINE_POLICY_TUPLE_FIELDS = (
    "allowed_engine_ids",
    "blocked_engine_ids",
    "preferred_engine_ids",
)
_ENGINE_POLICY_INT_FIELDS = (
    "max_monthly_mau_attested",
    "max_monthly_revenue_usd",
)
_ENGINE_POLICY_BOOL_FIELDS = (
    "forensic_mode",
    "allow_generative",
    "allow_cloud_egress",
    "allow_nc_licensed",
    "allow_use_restrictions",
    "allow_gated_licensed",
    "gated_license_ack",
)
_ENGINE_POLICY_VALUE_FIELDS = ("enterprise_license_key",)
_ENGINE_POLICY_FIELDS = (
    *_ENGINE_POLICY_TUPLE_FIELDS,
    *_ENGINE_POLICY_INT_FIELDS,
    *_ENGINE_POLICY_BOOL_FIELDS,
    *_ENGINE_POLICY_VALUE_FIELDS,
)


def _engine_policy_from_job_settings(settings_json) -> EnginePolicy:
    """Build an EnginePolicy from job.settings_json.

    Supported forms:
    - {"engine_policy": {...}}
    - {"ocr_engine_policy": {...}}

    Unknown keys are ignored so older clients can keep sending unrelated
    settings. Tuple and integer fields are normalized from JSON-friendly values.
    """
    raw_settings = settings_json or {}
    if not isinstance(raw_settings, dict):
        return EnginePolicy()

    raw_policy = (
        raw_settings.get("engine_policy") or raw_settings.get("ocr_engine_policy") or {}
    )
    if not isinstance(raw_policy, dict):
        return EnginePolicy()

    kwargs = {}
    for key, value in raw_policy.items():
        if key not in _ENGINE_POLICY_FIELDS:
            continue
        if key in _ENGINE_POLICY_TUPLE_FIELDS:
            kwargs[key] = _tuple_policy_value(value)
        elif key in _ENGINE_POLICY_INT_FIELDS:
            kwargs[key] = _int_policy_value(value)
        elif key in _ENGINE_POLICY_BOOL_FIELDS:
            kwargs[key] = _bool_policy_value(value)
        else:
            kwargs[key] = value
    return EnginePolicy(**kwargs)


def _settings_engine_ids(settings_json, policy: EnginePolicy) -> tuple[str, ...]:
    raw_settings = settings_json or {}
    if not isinstance(raw_settings, dict):
        return ()

    ids = []
    for key in (
        "ocr_engine",
        "ocr_engine_id",
        "requested_engine",
        "requested_engine_id",
        "engine_id",
    ):
        value = raw_settings.get(key)
        if isinstance(value, str) and value.strip():
            ids.append(value.strip())

    for value in policy.preferred_engine_ids:
        if value:
            ids.append(str(value))

    deduped = []
    seen = set()
    for engine_id in ids:
        if engine_id not in seen:
            seen.add(engine_id)
            deduped.append(engine_id)
    return tuple(deduped)


def _requested_non_cascade_capabilities(settings_json, policy: EnginePolicy):
    cascade_engine_ids = {
        catalog.PADDLE_CTC.engine_id,
        catalog.TESSERACT.engine_id,
    }
    caps = []
    for engine_id in _settings_engine_ids(settings_json, policy):
        if engine_id in cascade_engine_ids:
            continue
        cap = catalog.CAPABILITY_BY_ID.get(engine_id)
        if cap is not None:
            caps.append(cap)
    return tuple(caps)


def _engine_page_id(document_id, page_num) -> str | None:
    if page_num is None:
        return None
    return f"{document_id}:{page_num}"


def _record_engine_event(job, document_id, hostname, event):
    event_type = event.get("event", "ENGINE_PROVENANCE")
    data = {key: value for key, value in event.items() if key != "event"}
    _record_custody_event(
        job,
        document_id,
        event_type,
        data=data,
        worker_hostname=hostname,
    )


def _engine_event_exists(job, document_id, event_type, capability, policy, reason=None):
    query = CustodyEvent.objects.filter(
        job=job,
        document_id=document_id,
        event_type=event_type,
        data__engine_id=capability.engine_id,
        data__policy_hash=compute_policy_hash(policy),
    )
    if reason is not None:
        query = query.filter(data__reason=reason)
    return query.exists()


def _record_engine_denial(
    job,
    document_id,
    hostname,
    page_num,
    capability,
    policy,
    reason,
    *,
    dedupe=False,
):
    if reason == BLOCKED_FORENSIC_MODE:
        event = build_lane_boundary_event(
            capability=capability,
            policy=policy,
            page_id=_engine_page_id(document_id, page_num),
        )
    else:
        event = build_license_blocked_event(
            capability=capability,
            policy=policy,
            reason=reason,
            page_id=_engine_page_id(document_id, page_num),
        )
    if dedupe and _engine_event_exists(
        job,
        document_id,
        event.get("event", "ENGINE_PROVENANCE"),
        capability,
        policy,
        reason=None if reason == BLOCKED_FORENSIC_MODE else reason,
    ):
        return
    _record_engine_event(job, document_id, hostname, event)


def _record_engine_provenance(job, document_id, hostname, page_num, capability, policy):
    _record_engine_event(
        job,
        document_id,
        hostname,
        build_engine_provenance_event(
            page_id=_engine_page_id(document_id, page_num),
            capability=capability,
            policy=policy,
        ),
    )


def _record_requested_engine_denials(job, document_id, hostname, policy):
    for capability in _requested_non_cascade_capabilities(job.settings_json, policy):
        decision = evaluate_engine(capability, policy)
        if decision.admitted:
            logger.debug(
                "Requested engine %s is policy-admitted but not wired "
                "into the coordinator OCR cascade",
                capability.engine_id,
            )
            continue
        _record_engine_denial(
            job,
            document_id,
            hostname,
            None,
            capability,
            policy,
            decision.reason,
            dedupe=True,
        )


def _compute_file_hash(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    parent_fd = -1
    fd = -1
    try:
        try:
            parent_fd, basename, _ = _open_local_parent(filepath, create=False)
            fd = os.open(
                basename,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except RuntimeError as exc:
            raise OSError("source hash path has an unsafe parent") from exc
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError("source hash requires a regular file")
        if file_stat.st_nlink != 1:
            raise OSError("source hash refuses multiply-linked files")
        with os.fdopen(fd, "rb") as f:
            fd = -1
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
    finally:
        if fd >= 0:
            os.close(fd)
        if parent_fd >= 0:
            os.close(parent_fd)
    return sha256.hexdigest()


def _is_sha256_hex(value):
    """Return True only for a canonical lowercase SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _checkpoint_merged_pdf(merged_pdf, checkpoint_path):
    """Persist and reopen merged PDF to release intermediate merge memory."""
    import fitz

    checkpoint_dir = os.path.dirname(checkpoint_path) or "."
    fd, next_checkpoint = tempfile.mkstemp(
        dir=checkpoint_dir,
        prefix=".merged-checkpoint.",
        suffix=".pdf",
    )
    os.close(fd)
    os.remove(next_checkpoint)
    try:
        merged_pdf.save(next_checkpoint, garbage=3, deflate=True)
        merged_pdf.close()
        os.replace(next_checkpoint, checkpoint_path)
        return fitz.open(checkpoint_path)
    finally:
        try:
            os.remove(next_checkpoint)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Ingest helpers (extracted from ingest_document for testability)
# ---------------------------------------------------------------------------


class _IngestError(Exception):
    """Internal error raised by ingest helpers to signal a failed ingest step.

    Carries a user-facing ``message`` that becomes the job error_message and
    the return payload of the ingest_document task.
    """


def _setup_ingest_work_dir(job, job_id, storage_mode):
    """Create the working directory and determine source_path.

    For NFS mode, creates the standard job directory under NFS_ROOT.
    For S3 mode, creates a temporary directory.

    Returns:
        (job_path, source_path) tuple.
    """
    if storage_mode == "nfs":
        job_path = _nfs_job_path(job_id)
        _ensure_nfs_job_dirs(job_path)
        job.nfs_job_path = job_path
        job.save(update_fields=["nfs_job_path"])
    else:
        # S3 mode: use temp directory
        job_path = tempfile.mkdtemp(prefix=f"job_{job_id}_")
        _ensure_job_dirs(job_path)

    source_path = os.path.join(job_path, "source", os.path.basename(job.source_file))
    return job_path, source_path


def _validate_and_resolve_source(job, source_path):
    """Validate that the source file is accessible and within allowed root.

    If the file is not at source_path but job.source_file is an absolute path
    to an existing file, copies it into source_path after a path-containment
    check.

    Raises:
        _IngestError: If the source file is outside the allowed root or missing.
    """
    if _safe_regular_path_exists(source_path):
        return

    # Check if source_file is an absolute path (direct upload)
    if _safe_regular_path_exists(job.source_file):
        # Path containment: only allow files within NFS_ROOT.
        real_source = os.path.realpath(job.source_file)
        allowed_root = os.path.realpath(settings.NFS_ROOT)
        if (
            not real_source.startswith(allowed_root + os.sep)
            and real_source != allowed_root
        ):
            raise _IngestError(
                f"Source file outside allowed directory: {job.source_file}"
            )
        _copy_source_no_follow(real_source, source_path)
    else:
        raise _IngestError(f"Source file not found: {job.source_file}")


def _copy_source_no_follow(source_path, destination_path):
    """Copy a coordinator source through an identity-checked file handle."""
    source_parent_fd = -1
    destination_parent_fd = -1
    source_fd = -1
    destination_fd = -1
    try:
        source_parent_fd, source_basename, _ = _open_local_parent(
            source_path, create=False
        )
        source_fd = os.open(
            source_basename,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=source_parent_fd,
        )
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise _IngestError("Source path must reference a regular file")
        if getattr(source_stat, "st_nlink", 1) > 1:
            raise _IngestError("Source path has multiple hard links")
        destination_parent_fd, destination_basename, _ = _open_local_parent(
            destination_path
        )
        destination_fd = os.open(
            destination_basename,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=destination_parent_fd,
        )
        with os.fdopen(source_fd, "rb") as source_file:
            source_fd = -1
            with os.fdopen(destination_fd, "wb") as destination:
                destination_fd = -1
                shutil.copyfileobj(source_file, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
    except OSError as exc:
        raise _IngestError(f"Source copy failed: {exc}") from exc
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_parent_fd >= 0:
            os.close(source_parent_fd)
        if destination_parent_fd >= 0:
            os.close(destination_parent_fd)


def _safe_regular_path_exists(path: str | None) -> bool:
    """Check one local file without following symlinked parents or leaves."""
    if not isinstance(path, str) or not path:
        return False
    parent_fd = -1
    fd = -1
    try:
        try:
            parent_fd, basename, _ = _open_local_parent(path, create=False)
            fd = os.open(
                basename,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except (FileNotFoundError, NotADirectoryError, RuntimeError, OSError):
            return False
        metadata = os.fstat(fd)
        return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
    finally:
        if fd >= 0:
            os.close(fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _upload_source_to_s3(backend, job_id, source_path):
    """Upload the source file to S3 backend.

    Raises:
        _IngestError: If the upload fails.
    """
    try:
        source_key = _job_storage_key(job_id, f"source/{os.path.basename(source_path)}")
        backend.upload_file(source_path, source_key)
        logger.info("Uploaded source file to S3: %s", source_key)
    except Exception as exc:
        raise _IngestError(f"S3 upload failed: {exc}") from exc


def _materialize_preuploaded_source(job, backend, source_path):
    """Download and verify a source committed by durable API admission.

    Returns ``False`` for legacy jobs whose source still arrives through the
    coordinator's local/NFS path.  Once ``submission_ready`` is committed,
    malformed metadata or changed object bytes fail closed and may not fall
    back to a pod-local source.
    """
    if not getattr(job, "submission_ready", False):
        return False
    filename = os.path.basename(job.source_file)
    expected_key = _job_storage_key(job.job_id, f"source/{filename}")
    if (
        not filename
        or job.source_file != filename
        or job.source_object_key != expected_key
    ):
        raise _IngestError("Pre-uploaded source binding is not canonical")
    expected_hash = str(job.source_hash or "")
    expected_size = int(job.source_size_bytes or 0)
    if len(expected_hash) != 64 or expected_size < 1:
        raise _IngestError("Pre-uploaded source integrity metadata is incomplete")
    try:
        backend.download_file(job.source_object_key, source_path)
        if not _safe_regular_path_exists(source_path):
            raise OSError("download did not create a regular file")
        actual_size = os.path.getsize(source_path)
        actual_hash = _compute_file_hash(source_path)
    except Exception as exc:
        raise _IngestError(f"Pre-uploaded source download failed: {exc}") from exc
    if actual_size != expected_size or not hmac.compare_digest(
        actual_hash, expected_hash
    ):
        raise _IngestError("Pre-uploaded source integrity check failed")
    return True


def _coordinator_expansion_limit(
    name: str,
    default: int,
    minimum: int = 1,
    maximum: int = 1_000_000,
) -> int:
    """Read a bounded expansion limit for coordinator fallback paths."""
    try:
        value = int(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_operational_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Parse an operational knob while keeping task work and sleep bounded."""
    try:
        value = int(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


FANOUT_IMMEDIATE_PUBLISH_LIMIT = _bounded_operational_int(
    "FANOUT_IMMEDIATE_PUBLISH_LIMIT", 100, minimum=1, maximum=1_000
)


def _fair_fanout_enabled() -> bool:
    return _staged_page_ocr_enabled() or os.environ.get("OCR_FAIR_FANOUT_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _staged_page_ocr_enabled() -> bool:
    return os.environ.get("OCR_STAGED_PAGE_OCR_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _max_inflight_pages_per_document() -> int:
    return _bounded_operational_int(
        "OCR_MAX_INFLIGHT_PAGES_PER_DOCUMENT", 8, minimum=1, maximum=4096
    )


def _admit_pending_pages_locked(job, *, max_new=None):
    """Reserve bounded per-job page slots and outbox contracts atomically.

    Caller holds the Job row lock and a transaction. A pending row counts as
    in flight as soon as it is admitted, including broker delay and retries.
    Only a terminal page releases its slot; redelivery cannot inflate it.
    """
    if not _fair_fanout_enabled() or job.status != Job.Status.PROCESSING:
        return []
    generation = int(job.dispatch_generation)
    active = PageResult.objects.filter(
        job=job,
        admitted_generation=generation,
        status__in=("pending", "processing"),
    ).count()
    capacity = max(0, _max_inflight_pages_per_document() - active)
    if max_new is not None:
        capacity = min(capacity, max_new)
    if capacity == 0:
        return []
    pending = list(
        PageResult.objects.filter(job=job, status="pending")
        .exclude(admitted_generation=generation)
        .order_by("page_num")[:capacity]
    )
    if not pending:
        return []
    staged = _staged_page_ocr_enabled() and job.source_type in {"pdf", "image"}
    lane = _select_initial_ocr_engine_lane(job) if _separate_tesseract_lane_enabled() and not staged else None
    queue = "ocr_render" if staged else _get_ocr_queue(lane)
    rows = []
    for page in pending:
        kwargs = {"presigned_capability_id": None, **_tenant_for_task(job)}
        if lane:
            kwargs["ocr_engine_lane"] = lane
        if staged:
            PageOCRStage.objects.get_or_create(
                job=job,
                page_num=page.page_num,
                generation=generation,
                defaults={
                    "tenant_id": job.tenant_id,
                    "source_sha256": job.source_hash,
                },
            )
        rows.append(
            enqueue_task_dispatch(
                job=job,
                stage=f"{'render_page' if staged else 'process_page'}:{page.page_num}",
                generation=str(generation),
                task_name="jobs.tasks.render_page" if staged else "jobs.tasks.process_page",
                args=[str(job.job_id), page.page_num, generation] if staged else [str(job.job_id), page.page_num],
                kwargs=kwargs,
                queue=queue,
                priority=_get_celery_priority(job),
            )
        )
        page.admitted_generation = generation
        page.save(update_fields=["admitted_generation"])
    job.last_page_admission_at = timezone.now()
    job.save(update_fields=["last_page_admission_at"])
    return rows


def _validate_job_page_budget(job) -> None:
    """Reject malformed or over-budget persisted page counts at task entry."""
    total_pages = getattr(job, "total_pages", None)
    max_pages = _coordinator_expansion_limit("MAX_PDF_PAGES", 5000)
    if isinstance(total_pages, bool) or not isinstance(total_pages, int):
        raise ValueError("job total_pages is not a valid integer")
    if total_pages < 1:
        raise ValueError("job total_pages must be at least 1")
    if total_pages > max_pages:
        raise ValueError(
            f"job total_pages {total_pages} exceeds MAX_PDF_PAGES={max_pages}"
        )


def _load_source_classifiers():
    """Import or define fallback file classification functions.

    When ``ocr_distributed`` is available the canonical
    :func:`classify_source_file` is used with ``include_coordinator_types=True``
    so that text-like ingest targets (``.txt``, ``.md``, ``.csv``, ``.json``)
    are accepted at the coordinator level.

    Returns:
        (classify_source_file, get_source_page_count) callable tuple.
    """
    try:
        from functools import partial  # noqa: I001
        from ocr_distributed.ocr_utils import (
            classify_source_file as _canonical_classify,
            get_source_page_count,
        )

        classify_source_file = partial(
            _canonical_classify,
            include_coordinator_types=True,
        )
        return classify_source_file, get_source_page_count
    except ModuleNotFoundError:
        logger.warning(
            "ocr_distributed package unavailable in coordinator runtime; "
            "using local ingest fallback classifier"
        )

    # --- Minimal fallback for air-gapped coordinator-only deployments ---

    def classify_source_file(path: str) -> tuple[str | None, str | None]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return "pdf", None
        text_exts = {".txt", ".md", ".csv", ".json"}
        if ext in text_exts:
            return "text", None
        image_exts = {
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".gif",
            ".webp",
            ".jp2",
        }
        if ext in image_exts:
            return "image", None
        return None, f"Unsupported extension: {ext or '<none>'}"

    def get_source_page_count(path: str, source_type: str) -> int:
        max_pages = _coordinator_expansion_limit("MAX_PDF_PAGES", 5000)
        if source_type == "pdf":
            import fitz

            with fitz.open(path) as pdf:
                count = len(pdf)
                if count > max_pages:
                    raise ValueError(
                        f"PDF has {count} pages which exceeds MAX_PDF_PAGES={max_pages}"
                    )
                return count
        if source_type == "image":
            try:
                from PIL import Image

                max_pixels = _coordinator_expansion_limit(
                    "MAX_IMAGE_PIXELS",
                    100_000_000,
                    minimum=0,
                    maximum=2_000_000_000,
                )
                with Image.open(path) as image:
                    width, height = image.size
                    frames = max(1, int(getattr(image, "n_frames", 1) or 1))
                    total_pixels = int(width) * int(height) * frames
                    if max_pixels > 0 and total_pixels > max_pixels:
                        raise ValueError(
                            f"Image has {total_pixels} pixels which exceeds "
                            f"MAX_IMAGE_PIXELS={max_pixels}"
                        )
                    if frames > max_pages:
                        raise ValueError(
                            f"Image has {frames} frames which exceeds "
                            f"MAX_PDF_PAGES={max_pages}"
                        )
                    return frames
            except ImportError:
                pass
        return 1

    return classify_source_file, get_source_page_count


def _classify_and_count_pages(job, source_path):
    """Classify the source file type and determine its page count.

    Updates job.source_type and job.total_pages in the database.

    Returns:
        (source_type, total_pages) tuple.

    Raises:
        _IngestError: If the file type is unsupported or classification/counting fails.
    """
    try:
        classify_source_file, get_source_page_count = _load_source_classifiers()

        source_type, warning = classify_source_file(source_path)
        if not source_type:
            raise _IngestError(f"Unsupported file: {warning}")
        if warning:
            logger.warning(
                "File classification warning for %s: %s", source_path, warning
            )
        job.source_type = source_type
        job.save(update_fields=["source_type"])
    except _IngestError:
        raise
    except Exception as exc:
        raise _IngestError(f"File classification failed: {exc}") from exc

    try:
        total_pages = get_source_page_count(source_path, source_type)
        job.total_pages = total_pages
        job.save(update_fields=["total_pages"])
    except Exception as exc:
        raise _IngestError(f"Page count failed: {exc}") from exc

    return source_type, total_pages


def _detect_language(job, source_path, source_type):
    """Detect the document language (best-effort, unknown remains 'und').

    Updates job.detected_language in the database.  Never raises.
    """
    try:
        from ocr_distributed.language import LanguageDetector

        model_path = _fasttext_model_path()
        detector = LanguageDetector(model_path)
        if source_type == "pdf":
            lang = detector.detect_from_pdf(source_path)
        else:
            lang = "und"
        job.detected_language = lang
        job.save(update_fields=["detected_language"])
    except Exception:
        job.detected_language = "und"
        job.save(update_fields=["detected_language"])


def _log_worker_availability(job_id):
    """Log a warning/info message about available workers (advisory only)."""
    gpu_workers = Worker.objects.filter(
        status__in=[Worker.Status.ONLINE, Worker.Status.BUSY],
        gpu_available=True,
    )
    if not gpu_workers.exists():
        any_workers = Worker.objects.filter(
            status__in=[Worker.Status.ONLINE, Worker.Status.BUSY],
        )
        if not any_workers.exists():
            logger.warning(
                "No online workers available for job %s -- "
                "proceeding anyway (queue will hold task until worker connects)",
                job_id,
            )
        else:
            logger.info(
                "No GPU workers online for job %s -- "
                "CPU workers available (Tesseract fallback)",
                job_id,
            )


def _dispatch_processing(job, job_id, total_pages):
    """Transition job to PROCESSING and enqueue the root task durably.

    Returns:
        Result dict with status, job_id, total_pages, and mode.

    The state transition and outbox insert share one database transaction.
    The immediate relay is best effort; the beat-driven relay repairs a
    coordinator crash or broker outage after the transaction commits.
    """
    with transaction.atomic():
        locked_job = Job.objects.select_for_update().get(pk=job_id)
        terminal = _reject_if_job_terminal(locked_job)
        if terminal is not None:
            return terminal
        if locked_job.status != Job.Status.INGESTING:
            return {
                "status": "skipped",
                "job_id": str(job_id),
                "reason": (
                    "ingest dispatch requires ingesting state; "
                    f"observed={locked_job.status}"
                ),
            }
        locked_job.status = Job.Status.PROCESSING
        locked_job.dispatch_generation = int(locked_job.dispatch_generation or 0) + 1
        # The ingest delivery token must not fence the new processing
        # delivery.  The worker or relay records the broker task id next.
        locked_job.celery_task_id = ""
        locked_job.save(
            update_fields=["status", "dispatch_generation", "celery_task_id"]
        )

        celery_priority = _get_celery_priority(locked_job)
        skip_ocr = bool((locked_job.settings_json or {}).get("skip_ocr", False))
        # Coordinator-level classification accepts text-like inputs.  They
        # must use the native-text path even when skip_ocr was not explicit.
        text_source = locked_job.source_type == "text"
        if skip_ocr or text_source:
            task_name = "jobs.tasks.process_text_only"
            queue = "cpu_general"
            mode = "skip_ocr" if skip_ocr else "text_only"
            logger.info(
                "Job %s using native text processing (skip_ocr=%s, source_type=%s)",
                job_id,
                skip_ocr,
                locked_job.source_type,
            )
        elif locked_job.source_type in {"pdf", "image"}:
            # Every classified OCR page gets its own PageResult lease and
            # immutable output namespace, including one-page jobs. A whole-
            # document retry cannot safely reuse fixed page output keys.
            task_name = "jobs.tasks.extract_pages"
            queue = "coordinator"
            mode = (
                "capability_fanout"
                if _staged_page_ocr_enabled() or _separate_tesseract_lane_enabled()
                else "fanout"
            )
        else:
            raise ValueError(
                f"classified processing source type is unsupported: {locked_job.source_type!r}"
            )

        if queue != "ocr_gpu" and task_name.endswith("process_document"):
            logger.info("Job %s routed to %s queue", job_id, queue)

        # Tenant identity is data used to establish the worker's RLS context,
        # never an authority for the worker.  Empty local jobs preserve the
        # legacy task shape; production tenant-bound jobs carry the value and
        # the worker verifies it against the persisted row.
        dispatch_kwargs = {}
        if str(locked_job.tenant_id or "").strip():
            dispatch_kwargs["tenant_id"] = str(locked_job.tenant_id).strip()
        outbox_row = enqueue_task_dispatch(
            job=locked_job,
            stage="root_processing",
            generation=str(locked_job.dispatch_generation),
            task_name=task_name,
            args=[str(job_id)],
            kwargs=dispatch_kwargs,
            queue=queue,
            priority=celery_priority,
        )

    # Publish outside the state transaction.  A failed immediate attempt is
    # retained as FAILED/available_at in the outbox for the periodic relay;
    # the job remains PROCESSING rather than being falsely terminalized.
    published = publish_task_dispatch_outbox(
        dispatch_id=outbox_row.dispatch_id,
        dispatcher=_publish_root_outbox_row,
    )
    row_result = published[0] if published else {}
    if row_result.get("status") != TaskDispatchOutbox.Status.PUBLISHED:
        logger.warning(
            "Root processing dispatch deferred for job %s; outbox=%s",
            job_id,
            outbox_row.dispatch_id,
        )

    return {
        "status": (
            "dispatched"
            if row_result.get("status") == TaskDispatchOutbox.Status.PUBLISHED
            else "queued"
        ),
        "job_id": str(job_id),
        "total_pages": total_pages,
        "mode": mode,
        "dispatch_id": str(outbox_row.dispatch_id),
    }


def _enqueue_assembly_dispatch(job_id, *, expected_processing_token=None, expected_job=None):
    """Persist the PROCESSING -> assembly handoff before broker publication.

    The unique ``(job, stage, generation)`` outbox key makes concurrent page
    reconciliation and worker redelivery converge on one logical assembly
    delivery.  Publication happens after commit so a broker outage cannot
    erase the handoff.
    """
    with transaction.atomic():
        job = Job.objects.select_for_update().get(job_id=job_id)
        terminal = _reject_if_job_terminal(job)
        if terminal is not None:
            return terminal
        if job.status != Job.Status.PROCESSING:
            return {
                "status": "skipped",
                "job_id": str(job_id),
                "reason": f"assembly handoff requires processing; observed={job.status}",
            }
        if (
            expected_processing_token is not None
            and (
                job.processing_attempt_token != expected_processing_token
                or job.processing_lease_expires_at is None
                or job.processing_lease_expires_at <= timezone.now()
            )
        ):
            return {
                "status": "stale",
                "job_id": str(job_id),
                "reason": "processing lease changed before assembly handoff",
            }
        if expected_job is not None and (
            job.dispatch_generation != expected_job.dispatch_generation
            or job.tenant_id != expected_job.tenant_id
            or job.source_hash != expected_job.source_hash
            or job.source_object_key != expected_job.source_object_key
            or job.source_file != expected_job.source_file
            or job.processing_attempt_identity != text_only_attempt_identity(expected_job)
            or job.processing_attempt_identity != text_only_attempt_identity(job)
        ):
            return {"status": "stale", "job_id": str(job_id), "reason": "assembly source identity changed"}
        row = enqueue_task_dispatch(
            job=job,
            stage="assemble_document",
            generation=str(job.dispatch_generation),
            task_name="jobs.tasks.assemble_document",
            args=[str(job_id)],
            kwargs=_tenant_for_task(job),
            queue="coordinator",
            priority=_get_celery_priority(job),
        )

    transaction.on_commit(
        partial(publish_task_dispatch_outbox, dispatch_id=row.dispatch_id)
    )
    return {
        "status": "queued",
        "job_id": str(job_id),
        "dispatch_id": str(row.dispatch_id),
    }


def _publish_root_outbox_row(row):
    """Resolve the small root-task allowlist without serializing task code."""
    task = {
        "jobs.tasks.process_document": process_document,
        "jobs.tasks.extract_pages": extract_pages,
        "jobs.tasks.process_text_only": process_text_only,
    }.get(row.task_name)
    if task is None:
        raise RuntimeError(f"unsupported root outbox task {row.task_name}")
    job = Job.objects.get(pk=row.job_id)
    if str(row.generation) != str(job.dispatch_generation):
        raise RuntimeError("refusing stale-generation root outbox publication")
    kwargs = dict(row.kwargs or {})
    kwargs["dispatch_generation"] = int(row.generation)
    if (
        row.task_name
        in {
            "jobs.tasks.process_document",
            "jobs.tasks.process_text_only",
        }
        and is_presigned_mode()
    ):
        operation = (
            "process_document"
            if row.task_name == "jobs.tasks.process_document"
            else "process_text_only"
        )
        kwargs["presigned_capability_id"] = create_presigned_capability(
            job, operation, {}
        )
    options = {
        "queue": resolve_task_dispatch_queue(row, kwargs),
        "priority": row.priority,
    }
    if kwargs:
        return task.apply_async(args=list(row.args or []), kwargs=kwargs, **options)
    return task.apply_async(args=list(row.args or []), **options)


def _cleanup_temp_dir(job_path, storage_mode):
    """Remove the temporary working directory if we are in S3 mode.

    Safe to call multiple times; missing directories are already clean.  A
    failed removal is returned to the caller so ingest cannot dispatch while
    local OCR bytes remain on disk.
    """
    if storage_mode == "nfs" or not job_path:
        return True
    if not os.path.lexists(job_path):
        return True
    try:
        if os.path.islink(job_path):
            os.unlink(job_path)
        elif os.path.isdir(job_path):
            shutil.rmtree(job_path)
        else:
            os.remove(job_path)
    except OSError:
        logger.exception("Could not remove temporary job workspace: %s", job_path)
        return False
    if os.path.lexists(job_path):
        logger.error("Temporary job workspace remains after cleanup: %s", job_path)
        return False
    return True


# ---------------------------------------------------------------------------
# Coordinator Queue Tasks
# ---------------------------------------------------------------------------


@shared_task(
    bind=True,
    name="jobs.tasks.ingest_document",
    queue="coordinator",
    max_retries=2,
    default_retry_delay=30,
)
@_tenant_bound_task
def ingest_document(self, job_id, tenant_id=None, dispatch_generation=None):
    """Validate source file, compute hash, detect language, and dispatch.

    This task runs on the coordinator and decides whether to process the
    document as a single task or fan out across multiple workers.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        logger.error("Job %s not found", job_id)
        return {"status": "error", "message": f"Job {job_id} not found"}
    _pin_rls_for_job(job)

    stale = _reject_stale_dispatch_generation(job, dispatch_generation)
    if stale is not None:
        return stale

    terminal = _reject_if_job_terminal(job)
    if terminal is not None:
        return terminal

    ingest_token = _claim_ingest_lease(job, getattr(self.request, "id", None))
    if ingest_token is None:
        job.refresh_from_db()
        return {
            "status": "skipped",
            "job_id": str(job_id),
            "reason": "ingest stage is already owned or no longer submitted",
            "message": f"job is already {job.status}",
        }
    job.refresh_from_db()

    backend = _get_storage_backend()
    storage_mode = backend.backend_name
    job.storage_backend_used = storage_mode
    job_settings = dict(job.settings_json or {})
    job_settings["storage_binding_fingerprint"] = _storage_binding_fingerprint(
        storage_mode
    )
    job.settings_json = job_settings
    job.save(
        update_fields=[
            "status",
            "started_at",
            "celery_task_id",
            "storage_backend_used",
            "settings_json",
        ]
    )

    job_path = None
    ingest_error = None
    cleanup_failed = False
    try:
        # Set up working directory and resolve source path
        job_path, source_path = _setup_ingest_work_dir(job, job_id, storage_mode)

        # Durable API admission stores the source before returning. Legacy
        # coordinator callers still use the bounded local/NFS handoff.
        preuploaded = storage_mode == "s3" and _materialize_preuploaded_source(
            job, backend, source_path
        )
        if not preuploaded:
            _validate_and_resolve_source(job, source_path)

        # Only legacy S3 jobs need an upload during ingest.
        if storage_mode == "s3" and not preuploaded:
            _upload_source_to_s3(backend, job_id, source_path)

        # The source hash is the document identity used by custody and output
        # keys.  A missing hash would make those bindings ambiguous, so ingest
        # fails closed instead of dispatching an unbound document.
        try:
            job.source_hash = _compute_file_hash(source_path)
            job.save(update_fields=["source_hash"])
        except OSError as exc:
            logger.error("Failed to hash %s: %s", source_path, exc)
            raise _IngestError("Source hash could not be established") from exc

        # Classify file type and get page count
        source_type, total_pages = _classify_and_count_pages(job, source_path)

        # Detect language; image-only sources retain honest unknown status.
        _detect_language(job, source_path, source_type)

    except _IngestError as exc:
        ingest_error = str(exc)
    finally:
        # Always clean up temp directory for non-NFS modes.
        # This fixes the leak where some error paths forgot to clean up.
        cleanup_failed = not _cleanup_temp_dir(job_path, storage_mode)

    if cleanup_failed:
        job.status = Job.Status.FAILED
        job.error_message = "Temporary job workspace cleanup failed; dispatch refused"
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])
        return {"status": "error", "message": job.error_message}

    if ingest_error is not None:
        job.status = Job.Status.FAILED
        job.error_message = ingest_error
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])
        return {"status": "error", "message": job.error_message}

    # Record custody event
    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
    _record_custody_event(
        job,
        document_id,
        "file_ingested",
        data={
            "source_path": job.source_file,
            "source_hash": job.source_hash,
            "source_type": source_type,
            "total_pages": total_pages,
        },
        worker_hostname=socket.gethostname(),
    )

    # Advisory log about worker availability
    _log_worker_availability(job_id)

    # Dispatch processing tasks
    return _dispatch_processing(job, job_id, total_pages)


def _new_assembly_staging_path(directory, base_name, suffix):
    """Reserve a collision-safe same-filesystem staging pathname."""
    fd, path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{base_name}.",
        suffix=suffix,
    )
    os.close(fd)
    os.remove(path)
    return path


def _durably_commit_assembly_file(staged_path, final_path):
    """Fsync a complete staged artifact and atomically publish it."""
    fd = os.open(staged_path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    from optimize_pdfs import _durable_replace

    _durable_replace(staged_path, final_path)


def _job_artifact_root(job, storage_mode):
    """Return the NFS job root path when applicable."""
    if storage_mode == "nfs":
        return _nfs_job_path_for_job(job)
    return ""


def _resolve_job_output_artifacts(job, backend=None):
    """Resolve active PDF/TEXT locators via generation pointer (legacy only if absent)."""
    backend = backend or _get_backend_for_job(job)
    storage_mode = backend.backend_name
    return resolve_active_artifacts(
        job_id=job.job_id,
        source_file=job.source_file,
        settings_json=job.settings_json,
        backend=storage_mode,
        job_root=_job_artifact_root(job, storage_mode),
    )


def _fail_assembly_job(
    job_id,
    document_id,
    message,
    failure_data,
    backend,
    *,
    claim_token,
):
    """Persist and publish a truthful terminal assembly failure."""
    with transaction.atomic():
        job = Job.objects.select_for_update().get(job_id=job_id)
        if job.status in _TERMINAL_JOB_STATUSES:
            return {
                "status": str(job.status),
                "job_id": str(job_id),
                "message": job.error_message or f"job was already {job.status}",
            }
        if (job.settings_json or {}).get("_assembly_claim_token") != claim_token:
            return {
                "status": "superseded",
                "job_id": str(job_id),
                "message": "assembly ownership was superseded",
            }
        _record_custody_event(
            job,
            document_id,
            "assembly_failed",
            data={"error": message, **failure_data},
            worker_hostname=socket.gethostname(),
        )
        job.status = Job.Status.FAILED
        job.error_message = message
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])
        if backend.backend_name == "nfs":
            _schedule_nfs_custody_publish(
                job,
                document_id,
                _nfs_job_path_for_job(job),
            )
        else:
            _schedule_s3_custody_publish(job, document_id, backend)
    return {"status": "failed", "job_id": str(job_id), "message": message}


def _dispatch_postprocessing(post_tasks, callback):
    # Optional lab hold so chaos drills can kill nlp/cpu consumers after
    # assembly_complete but before the post-assembly chord is enqueued.
    # Default 0 — no production delay.
    hold = _bounded_operational_int(
        "ASSEMBLING_POST_HOLD_SECONDS", 0, minimum=0, maximum=3_600
    )
    if hold > 0:
        logger.warning(
            "ASSEMBLING_POST_HOLD_SECONDS=%s — delaying post-assembly chord dispatch",
            hold,
        )
        time.sleep(hold)
    chord(post_tasks)(callback)


# MP-B1: orphaned assembling recovery (post-assembly chord lost consumer)
ASSEMBLING_REDRIVE_AFTER_SECONDS = _bounded_operational_int(
    "ASSEMBLING_REDRIVE_AFTER_SECONDS",
    90,
    minimum=1,
    maximum=7 * 24 * 3600,
)
ASSEMBLING_REDRIVE_MAX = _bounded_operational_int(
    "ASSEMBLING_REDRIVE_MAX", 3, minimum=0, maximum=100
)


def should_redrive_assembling_job(
    *,
    status,
    has_assembly_complete,
    assembly_complete_age_seconds,
    redrive_count,
    after_seconds=None,
    max_redrives=None,
):
    """Pure predicate: redrive post-assembly when stuck after assembly_complete.

    Unit-testable without Django ORM or Celery. Returns True only when the job
    is assembling, assembly artifacts were published, the chord appears stale,
    and redrive budget remains.
    """
    after = ASSEMBLING_REDRIVE_AFTER_SECONDS if after_seconds is None else after_seconds
    max_r = ASSEMBLING_REDRIVE_MAX if max_redrives is None else max_redrives
    if status not in (Job.Status.ASSEMBLING, "assembling"):
        return False
    if not has_assembly_complete:
        return False
    if redrive_count >= max_r:
        return False
    if assembly_complete_age_seconds is None:
        return False
    return float(assembly_complete_age_seconds) >= float(after)


def build_postprocessing_chord_signatures(job_id, presigned=None, tenant_id=None):
    """Build compress + NER + structured-data signatures for a job redrive."""
    presigned = presigned or {}
    tenant_kwargs = {"tenant_id": tenant_id} if str(tenant_id or "").strip() else {}
    job = None
    if presigned:
        job = Job.objects.get(job_id=job_id)
    capability_ids = {
        "compress": create_presigned_capability(
            job, "compress_pdf", presigned.get("compress")
        )
        if job is not None
        else None,
        "entities": create_presigned_capability(
            job, "extract_entities", presigned.get("entities")
        )
        if job is not None
        else None,
        "structured": create_presigned_capability(
            job, "extract_structured_data", presigned.get("structured")
        )
        if job is not None
        else None,
    }
    return [
        compress_pdf.s(
            str(job_id),
            presigned_capability_id=capability_ids["compress"],
            **tenant_kwargs,
        ),
        extract_entities.s(
            str(job_id),
            presigned_capability_id=capability_ids["entities"],
            **tenant_kwargs,
        ),
        extract_structured_data.s(
            str(job_id),
            presigned_capability_id=capability_ids["structured"],
            **tenant_kwargs,
        ),
    ]


@shared_task(
    bind=True,
    name="jobs.tasks.dispatch_postprocessing",
    queue="coordinator",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
    default_retry_delay=30,
)
@_tenant_bound_task
def dispatch_postprocessing(self, job_id, tenant_id=None, dispatch_generation=None):
    """Resolve fresh capabilities and publish the post-assembly chord."""
    del self, tenant_id
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}
    _pin_rls_for_job(job)
    stale = _reject_stale_dispatch_generation(job, dispatch_generation)
    if stale is not None:
        return stale
    terminal = _reject_if_job_terminal(job)
    if terminal is not None:
        return terminal
    if (
        job.status != Job.Status.ASSEMBLING
        or not CustodyEvent.objects.filter(
            job=job, event_type="assembly_complete"
        ).exists()
    ):
        return {
            "status": "skipped",
            "job_id": str(job_id),
            "reason": "postprocessing requires assembling + assembly_complete",
        }

    presigned = {}
    if is_presigned_mode():
        backend = _get_backend_for_job(job)
        if backend.backend_name == "s3":
            presigned = {
                "compress": {},
                "entities": {},
                "structured": {},
            }
    tenant_kwargs = _tenant_for_task(job)
    post_tasks = build_postprocessing_chord_signatures(
        job_id,
        presigned=presigned,
        tenant_id=tenant_kwargs.get("tenant_id"),
    )
    callback = finalize_job.s(str(job_id), **tenant_kwargs)
    # Celery invokes linked errbacks with different positional context across
    # direct and queued paths.  Keep the authoritative job identity in named
    # immutable task metadata instead of inferring it from those arguments.
    errback = chord_error_handler.s(
        bound_job_id=str(job_id),
        bound_generation=int(job.dispatch_generation),
        **tenant_kwargs,
    )
    for task_sig in post_tasks:
        task_sig.link_error(errback)
    _dispatch_postprocessing(post_tasks, callback)
    return {"status": "dispatched", "job_id": str(job_id)}


@transaction.atomic
def redrive_assembling_postprocessing(job_id):
    """Re-dispatch the post-assembly chord for a stuck assembling job.

    Returns a result dict. Increments settings_json.assembling_redrive_count.
    """
    try:
        # Keep the row lock through the predicate, claim update, custody
        # event, and dispatch.  Stale cleanup/retry cannot turn this into a
        # redrive for a job that changed state between the scan and dispatch.
        job = Job.objects.select_for_update().get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    if job.status != Job.Status.ASSEMBLING:
        return {
            "status": "skipped",
            "reason": f"status={job.status}",
            "job_id": str(job_id),
        }

    settings = job.settings_json if isinstance(job.settings_json, dict) else {}
    redrive_count = int(settings.get("assembling_redrive_count") or 0)
    has_complete = CustodyEvent.objects.filter(
        job=job, event_type="assembly_complete"
    ).exists()
    last_complete = (
        CustodyEvent.objects.filter(job=job, event_type="assembly_complete")
        .order_by("-timestamp")
        .first()
    )
    age = None
    if last_complete and last_complete.timestamp:
        age = (timezone.now() - last_complete.timestamp).total_seconds()

    # Force redrive when called with force=True from explicit operator recovery.
    force = bool(settings.get("assembling_redrive_force"))
    age_for_pred = float(age) if age is not None else None
    if force and has_complete:
        # Bypass age gate; still respect redrive budget.
        age_for_pred = float(ASSEMBLING_REDRIVE_AFTER_SECONDS) + 1.0
    if not should_redrive_assembling_job(
        status=job.status,
        has_assembly_complete=has_complete,
        assembly_complete_age_seconds=age_for_pred,
        redrive_count=redrive_count,
        after_seconds=0 if force else None,
    ):
        if redrive_count >= ASSEMBLING_REDRIVE_MAX and has_complete:
            job.status = Job.Status.FAILED
            job.error_message = (
                "post-assembly chord exhausted redrive budget "
                f"({ASSEMBLING_REDRIVE_MAX}); job left assembling without finalize"
            )
            job.last_failure_reason = "assembling_redrive_exhausted"
            job.completed_at = timezone.now()
            job.save(
                update_fields=[
                    "status",
                    "error_message",
                    "last_failure_reason",
                    "completed_at",
                ]
            )
            return {
                "status": "failed",
                "job_id": str(job_id),
                "reason": "redrive_exhausted",
            }
        return {
            "status": "skipped",
            "job_id": str(job_id),
            "reason": "predicate_false",
            "has_assembly_complete": has_complete,
            "redrive_count": redrive_count,
            "age_seconds": age,
        }

    settings = dict(settings)
    settings["assembling_redrive_count"] = redrive_count + 1
    settings["assembling_redrive_last_at"] = timezone.now().isoformat()
    # Refresh the lifecycle lease while the post-processing chord is being
    # claimed so stale cleanup cannot fail the job before its redrive runs.
    job.started_at = timezone.now()
    job.settings_json = settings
    job.save(update_fields=["settings_json", "started_at"])

    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
    _record_custody_event(
        job,
        document_id,
        "assembling_redrive",
        data={
            "redrive_count": redrive_count + 1,
            "reason": "stale_or_manual_redrive",
        },
        worker_hostname=socket.gethostname(),
    )

    outbox_row = enqueue_task_dispatch(
        job=job,
        stage=f"postprocessing_redrive:{redrive_count + 1}",
        generation=str(job.dispatch_generation),
        task_name="jobs.tasks.dispatch_postprocessing",
        args=[str(job_id)],
        kwargs=_tenant_for_task(job),
        queue="coordinator",
        priority=_get_celery_priority(job),
    )
    transaction.on_commit(
        partial(publish_task_dispatch_outbox, dispatch_id=outbox_row.dispatch_id)
    )
    return {
        "status": "redispatched",
        "job_id": str(job_id),
        "redrive_count": redrive_count + 1,
    }


@shared_task(
    bind=True,
    name="jobs.tasks.assemble_document",
    queue="coordinator",
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
)
@_tenant_bound_task
def assemble_document(self, job_id, tenant_id=None, dispatch_generation=None):
    """Merge page PDFs and dispatch the post-processing custody boundary.

    For fan-out processing, this is the chord callback that runs after all
    process_page tasks complete. It collects per-page temp PDFs, merges them
    in order, and writes the final output to storage backend.

    Retries up to 3 times with exponential backoff (jittered, max 60s) on
    transient failures such as S3 upload timeouts or storage backend errors.
    Permanent failures (job not found, cancelled) are not retried.
    """
    import fitz

    request_id = getattr(self.request, "id", None)
    assembly_claim_token = (
        f"{request_id or 'local'}:{socket.gethostname()}:{os.getpid()}:"
        f"{threading.get_ident()}:{time.monotonic_ns()}"
    )

    with transaction.atomic():
        try:
            job = Job.objects.select_for_update().get(job_id=job_id)
        except Job.DoesNotExist:
            logger.error("Job %s not found for assembly", job_id)
            return {"status": "error", "message": f"Job {job_id} not found"}
        _pin_rls_for_job(job)

        stale = _reject_stale_dispatch_generation(job, dispatch_generation)
        if stale is not None:
            return stale

        if job.status in _TERMINAL_JOB_STATUSES:
            return {
                "status": str(job.status),
                "job_id": str(job_id),
                "message": job.error_message or f"job was already {job.status}",
            }

        backend = _get_backend_for_job(job)
        storage_mode = backend.backend_name
        document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
        claimed_settings = dict(job.settings_json or {})
        claimed_settings["_assembly_claim_token"] = assembly_claim_token
        if job.status == Job.Status.PROCESSING:
            updated = Job.objects.filter(
                pk=job.pk,
                status=Job.Status.PROCESSING,
            ).update(
                status=Job.Status.ASSEMBLING,
                settings_json=claimed_settings,
                started_at=timezone.now(),
            )
            if updated != 1:
                raise RuntimeError("assembly state transition reservation was lost")
            job.status = Job.Status.ASSEMBLING
            job.settings_json = claimed_settings
        elif job.status == Job.Status.ASSEMBLING:
            updated = Job.objects.filter(
                pk=job.pk,
                status=Job.Status.ASSEMBLING,
            ).update(settings_json=claimed_settings, started_at=timezone.now())
            if updated != 1:
                raise RuntimeError("assembly ownership claim was lost")
            job.settings_json = claimed_settings
        elif job.status != Job.Status.ASSEMBLING:
            message = (
                f"assembly requires processing/assembling state; observed={job.status}"
            )
            _record_custody_event(
                job,
                document_id,
                "assembly_failed",
                data={"error": message, "pages_assembled": 0},
                worker_hostname=socket.gethostname(),
            )
            job.status = Job.Status.FAILED
            job.error_message = message
            job.completed_at = timezone.now()
            job.save(update_fields=["status", "error_message", "completed_at"])
            if storage_mode == "nfs":
                _schedule_nfs_custody_publish(
                    job,
                    document_id,
                    _nfs_job_path_for_job(job),
                )
            else:
                _schedule_s3_custody_publish(job, document_id, backend)
            return {"status": "failed", "job_id": str(job_id), "message": message}

    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    assembly_generation_id = make_generation_id(assembly_claim_token)
    job_path = _nfs_job_path_for_job(job) if storage_mode == "nfs" else ""
    try:
        pdf_location, text_location = generation_locations(
            job_id=job_id,
            source_file=job.source_file,
            backend=storage_mode,
            generation_id=assembly_generation_id,
            job_root=job_path or "",
        )
    except ArtifactPointerError as exc:
        return _fail_assembly_job(
            job_id,
            document_id,
            f"assembly generation path is invalid ({exc})",
            {
                "failure_kind": "invalid_generation_path",
                "pages_assembled": 0,
                "total_pages": job.total_pages,
            },
            backend,
            claim_token=assembly_claim_token,
        )

    # Setup working directory. Immutable generation-scoped destinations are
    # the only publication target for this claim; legacy EXPORT/ paths are
    # not dual-written.
    work_dir = ""
    if storage_mode == "nfs":
        temp_dir = os.path.join(job_path, "temp", document_id)
        final_pdf_path = pdf_location.locator
        final_text_path = text_location.locator
        output_pdf_dir = os.path.dirname(final_pdf_path)
        output_text_dir = os.path.dirname(final_text_path)
        nfs_root = getattr(settings, "NFS_ROOT", job_path)
        _mkdir_tree_no_follow(temp_dir, nfs_root)
        _mkdir_tree_no_follow(output_pdf_dir, nfs_root)
        _mkdir_tree_no_follow(output_text_dir, nfs_root)
    else:
        work_dir = tempfile.mkdtemp(prefix=f"assemble_{job_id}_")
        temp_dir = os.path.join(work_dir, "temp")
        output_pdf_dir = os.path.join(work_dir, "output", "PDF")
        output_text_dir = os.path.join(work_dir, "output", "TEXT")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_pdf_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)
        final_pdf_path = os.path.join(output_pdf_dir, f"{base_name}.pdf")
        final_text_path = os.path.join(output_text_dir, f"{base_name}.txt")

    # Collect page PDFs in order
    merged_pdf = fitz.open()
    checkpoint_interval = (
        ASSEMBLY_CHECKPOINT_INTERVAL_PAGES
        if ASSEMBLY_CHECKPOINT_INTERVAL_PAGES > 0
        else 0
    )
    checkpoint_enabled = (
        checkpoint_interval > 0
        and job.total_pages >= ASSEMBLY_STREAMING_THRESHOLD_PAGES
    )
    checkpoint_path = os.path.join(temp_dir, "_merged_checkpoint.pdf")
    pages_assembled = 0
    staged_text_path = _new_assembly_staging_path(
        output_text_dir,
        base_name,
        ".assembling.txt",
    )
    staged_pdf_path = _new_assembly_staging_path(
        output_pdf_dir,
        base_name,
        ".assembling.pdf",
    )
    assembly_failure = None

    try:
        with open(staged_text_path, "x", encoding="utf-8", newline="\n") as text_out:
            for page_num in range(1, job.total_pages + 1):
                if page_num == 1 or page_num % 10 == 0:
                    renewed = Job.objects.filter(
                        pk=job.pk,
                        status=Job.Status.ASSEMBLING,
                        settings_json=claimed_settings,
                    ).update(started_at=timezone.now())
                    if renewed != 1:
                        raise _ProcessingLeaseLost(
                            f"assembly lease lost for job {job.job_id}"
                        )
                page_result = PageResult.objects.filter(
                    job=job, page_num=page_num,
                ).only(
                    "temp_pdf_path", "ocr_provenance", "admitted_generation",
                    "ocr_method", "celery_task_id", "document_id", "status",
                ).first()
                if page_result is None or page_result.status not in {
                    "ok", "fallback", "image_only",
                }:
                    assembly_failure = (
                        page_num, "missing_accepted_page_result",
                        f"page {page_num} has no accepted page result",
                    )
                    break
                if (
                    page_result.admitted_generation != job.dispatch_generation
                    or page_result.document_id != document_id
                ):
                    assembly_failure = (
                        page_num, "stale_page_identity",
                        f"page {page_num} is not admitted for this source generation",
                    )
                    break
                page_pdf_path = os.path.join(temp_dir, f"{page_num}.pdf")
                page_text_path = os.path.join(temp_dir, f"{page_num}.txt")
                source_pdf_locator = (
                    page_result.temp_pdf_path if page_result and page_result.temp_pdf_path
                    else ""
                )
                source_text_locator = (
                    (page_result.ocr_provenance or {}).get("temp_text_path", "")
                    if page_result else ""
                )
                if page_result and page_result.ocr_method in {"NativeText", "NativePDF"}:
                    provenance = page_result.ocr_provenance or {}
                    if (
                        page_result.status != "ok"
                        or page_result.document_id != document_id
                        or provenance.get("source_sha256") != job.source_hash
                        or provenance.get("tenant_id") != job.tenant_id
                        or provenance.get("source_object_key") != job.source_object_key
                        or provenance.get("source_file") != job.source_file
                        or provenance.get("document_attempt_identity") is None
                    ):
                        assembly_failure = (
                            page_num, "native_page_identity_mismatch",
                            f"page {page_num} native winner is not bound to current source",
                        )
                        break
                    if storage_mode == "s3":
                        native_keys = text_only_attempt_keys(
                            job, page_num, document_id, page_result.celery_task_id,
                        )
                        expected_pdf = native_keys["page_pdf_key"]
                        expected_text = native_keys["page_text_key"]
                    else:
                        native_dir = os.path.join(
                            _nfs_job_path_for_job(job), "temp", document_id,
                            _native_text_attempt_suffix(job, page_result.celery_task_id),
                        )
                        expected_pdf = os.path.join(native_dir, f"{page_num}.pdf")
                        expected_text = os.path.join(native_dir, f"{page_num}.txt")
                    if source_pdf_locator != expected_pdf or source_text_locator != expected_text:
                        assembly_failure = (
                            page_num, "native_page_locator_mismatch",
                            f"page {page_num} native winner locator differs from its execution nonce",
                        )
                        break
                if source_pdf_locator or source_text_locator:
                    if not (source_pdf_locator and source_text_locator):
                        assembly_failure = (
                            page_num, "incomplete_page_locator",
                            f"page {page_num} has only one committed artifact locator",
                        )
                        break
                    if storage_mode == "s3":
                        expected_prefix = _job_storage_key(
                            job_id, f"temp/{document_id}/"
                        )
                        if not all(
                            str(locator).startswith(expected_prefix)
                            for locator in (source_pdf_locator, source_text_locator)
                        ):
                            assembly_failure = (
                                page_num, "invalid_page_locator",
                                f"page {page_num} artifact locator escapes its job",
                            )
                            break
                    else:
                        expected_root = os.path.realpath(
                            os.path.join(job_path, "temp", document_id)
                        )
                        if not all(
                            os.path.commonpath([
                                expected_root, os.path.realpath(str(locator))
                            ]) == expected_root
                            for locator in (source_pdf_locator, source_text_locator)
                        ):
                            assembly_failure = (
                                page_num, "invalid_page_locator",
                                f"page {page_num} artifact locator escapes its job",
                            )
                            break
                if source_pdf_locator and source_text_locator and storage_mode == "nfs":
                    page_pdf_path = source_pdf_locator
                    page_text_path = source_text_locator
                if storage_mode == "s3":
                    try:
                        pdf_key = source_pdf_locator or _job_storage_key(
                            job_id, f"temp/{document_id}/{page_num}.pdf"
                        )
                        text_key = source_text_locator or _job_storage_key(
                            job_id, f"temp/{document_id}/{page_num}.txt"
                        )
                        if not backend.exists(pdf_key):
                            assembly_failure = (
                                page_num,
                                "missing_page_pdf",
                                f"page {page_num} PDF artifact is missing",
                            )
                            break
                        if not backend.exists(text_key):
                            assembly_failure = (
                                page_num,
                                "missing_page_text",
                                f"page {page_num} text artifact is missing",
                            )
                            break
                        backend.download_file(pdf_key, page_pdf_path)
                        backend.download_file(text_key, page_text_path)
                    except Exception as exc:
                        assembly_failure = (
                            page_num,
                            "page_artifact_download_failed",
                            f"page {page_num} artifact download failed "
                            f"({type(exc).__name__})",
                        )
                        break

                if not os.path.isfile(page_pdf_path):
                    assembly_failure = (
                        page_num,
                        "missing_page_pdf",
                        f"page {page_num} PDF artifact is missing",
                    )
                    break
                try:
                    with fitz.open(page_pdf_path) as page_doc:
                        if page_doc.is_encrypted or page_doc.needs_pass:
                            raise ValueError("page PDF is encrypted")
                        if page_doc.page_count != 1:
                            raise ValueError(
                                f"page PDF contains {page_doc.page_count} pages"
                            )
                        merged_pdf.insert_pdf(page_doc)
                except Exception as exc:
                    assembly_failure = (
                        page_num,
                        "corrupt_page_pdf",
                        f"page {page_num} PDF artifact is corrupt or unreadable "
                        f"({type(exc).__name__})",
                    )
                    break

                if not os.path.isfile(page_text_path):
                    assembly_failure = (
                        page_num,
                        "missing_page_text",
                        f"page {page_num} text artifact is missing",
                    )
                    break
                try:
                    page_text = read_text_bounded(page_text_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    assembly_failure = (
                        page_num,
                        "unreadable_page_text",
                        f"page {page_num} text artifact is unreadable "
                        f"({type(exc).__name__})",
                    )
                    break
                expected_text_sha256 = (page_result.ocr_provenance or {}).get(
                    "text_sha256"
                )
                if (
                    not _is_sha256_hex(expected_text_sha256)
                    or not hmac.compare_digest(
                        hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
                        expected_text_sha256,
                    )
                ):
                    assembly_failure = (
                        page_num, "page_text_digest_mismatch",
                        f"page {page_num} text artifact is not bound to its accepted result",
                    )
                    break

                pages_assembled += 1
                if pages_assembled > 1:
                    text_out.write("\n\n")
                text_out.write(page_text)
                if checkpoint_enabled and pages_assembled % checkpoint_interval == 0:
                    merged_pdf = _checkpoint_merged_pdf(merged_pdf, checkpoint_path)
                    logger.info(
                        "Assembly checkpointed for job %s at %d/%d pages",
                        job_id,
                        pages_assembled,
                        job.total_pages,
                    )
            text_out.flush()
            os.fsync(text_out.fileno())

        if assembly_failure is None and (
            job.total_pages <= 0
            or pages_assembled != job.total_pages
            or merged_pdf.page_count != job.total_pages
        ):
            assembly_failure = (
                pages_assembled + 1,
                "incomplete_page_set",
                "page artifact set is incomplete: "
                f"assembled={pages_assembled}, total={job.total_pages}",
            )

        if assembly_failure is not None:
            failed_page, failure_kind, failure_message = assembly_failure
            merged_pdf.close()
            merged_pdf = None
            return _fail_assembly_job(
                job_id,
                document_id,
                failure_message,
                {
                    "failure_kind": failure_kind,
                    "failure_page": failed_page,
                    "pages_assembled": pages_assembled,
                    "total_pages": job.total_pages,
                },
                backend,
                claim_token=assembly_claim_token,
            )

        merged_pdf.save(staged_pdf_path, deflate=True)
        merged_pdf.close()
        merged_pdf = None

        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)

        try:
            # Short pre-publication ownership check.  The lock is released
            # before filesystem/S3 I/O so cancellation and recovery stay live.
            with transaction.atomic():
                publication_job = Job.objects.select_for_update().get(job_id=job_id)
                if publication_job.status in _TERMINAL_JOB_STATUSES:
                    return {
                        "status": str(publication_job.status),
                        "job_id": str(job_id),
                        "message": publication_job.error_message
                        or f"job was already {publication_job.status}",
                    }
                if publication_job.status != Job.Status.ASSEMBLING:
                    raise RuntimeError(
                        "assembly publication requires assembling state; "
                        f"observed={publication_job.status}"
                    )
                if (publication_job.settings_json or {}).get(
                    "_assembly_claim_token"
                ) != assembly_claim_token:
                    return {
                        "status": "superseded",
                        "job_id": str(job_id),
                        "message": "assembly ownership was superseded",
                    }

            # Hash staged bytes before visibility commit so the pointer binds
            # to immutable content identity.
            pdf_sha256 = _compute_file_hash(staged_pdf_path)
            text_sha256 = _compute_file_hash(staged_text_path)

            if storage_mode == "nfs":
                _durably_commit_assembly_file(staged_text_path, final_text_path)
                staged_text_path = ""
                _durably_commit_assembly_file(staged_pdf_path, final_pdf_path)
                staged_pdf_path = ""
                output_text_path = final_text_path
                output_pdf_path = final_pdf_path
            else:
                text_key = text_location.locator
                pdf_key = pdf_location.locator
                backend.upload_file(staged_text_path, text_key)
                backend.upload_file(staged_pdf_path, pdf_key)
                output_text_path = text_key
                output_pdf_path = pdf_key

            if _discard_published_artifacts_if_stale(
                job,
                backend,
                (
                    (output_text_path, None),
                    (output_pdf_path, None),
                ),
            ):
                fenced_status = (
                    Job.objects.filter(job_id=job_id)
                    .values_list("status", flat=True)
                    .first()
                )
                return {
                    "status": str(fenced_status or "stale"),
                    "job_id": str(job_id),
                    "message": "job was fenced during assembly publication",
                }

            # Visibility commit: single settings_json pointer update. Canceled
            # or superseded jobs keep any prior pointer (no write below).
            active_pointer = build_active_pointer(
                job_id=job_id,
                source_file=job.source_file,
                backend=storage_mode,
                revision=assembly_generation_id,
                pdf_generation_id=assembly_generation_id,
                pdf_sha256=pdf_sha256,
                text_generation_id=assembly_generation_id,
                text_sha256=text_sha256,
            )

            # Reconcile after canonical publication under a fresh short lock.
            # Cancellation/supersession during I/O suppresses success custody
            # and post-processing even though orphaned generation bytes may
            # need later cleanup.
            with transaction.atomic():
                locked_job = Job.objects.select_for_update().get(job_id=job_id)
                if locked_job.status in _TERMINAL_JOB_STATUSES:
                    return {
                        "status": str(locked_job.status),
                        "job_id": str(job_id),
                        "message": locked_job.error_message
                        or f"job was already {locked_job.status}",
                    }
                if locked_job.status != Job.Status.ASSEMBLING:
                    raise RuntimeError(
                        "assembly reconciliation requires assembling state; "
                        f"observed={locked_job.status}"
                    )
                if (locked_job.settings_json or {}).get(
                    "_assembly_claim_token"
                ) != assembly_claim_token:
                    return {
                        "status": "superseded",
                        "job_id": str(job_id),
                        "message": "assembly ownership was superseded",
                    }

                locked_job.pages_completed = pages_assembled
                reconciled_settings = dict(locked_job.settings_json or {})
                reconciled_settings.pop("_assembly_claim_token", None)
                reconciled_settings[ACTIVE_OUTPUT_POINTER_KEY] = active_pointer
                locked_job.settings_json = reconciled_settings
                locked_job.save(update_fields=["pages_completed", "settings_json"])

                if not CustodyEvent.objects.filter(
                    job=locked_job,
                    document_id=document_id,
                    event_type="assembly_complete",
                ).exists():
                    _record_custody_event(
                        locked_job,
                        document_id,
                        "assembly_complete",
                        data={
                            "pages_assembled": pages_assembled,
                            "total_pages": locked_job.total_pages,
                            "output_pdf": output_pdf_path,
                            "output_text": output_text_path,
                            "artifact_generation_id": assembly_generation_id,
                            "pdf_sha256": pdf_sha256,
                            "text_sha256": text_sha256,
                        },
                        worker_hostname=socket.gethostname(),
                    )

                postprocessing_outbox = enqueue_task_dispatch(
                    job=locked_job,
                    stage="postprocessing",
                    generation=str(locked_job.dispatch_generation),
                    task_name="jobs.tasks.dispatch_postprocessing",
                    args=[str(job_id)],
                    kwargs=_tenant_for_task(locked_job),
                    queue="coordinator",
                    priority=_get_celery_priority(locked_job),
                )
                transaction.on_commit(
                    partial(
                        publish_task_dispatch_outbox,
                        dispatch_id=postprocessing_outbox.dispatch_id,
                    )
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            if storage_mode == "s3":
                logger.error(
                    "Failed to upload assembly artifacts to S3 for job %s: %s",
                    job_id,
                    exc,
                )
                try:
                    retry_num = self.request.retries + 1
                    logger.warning(
                        "Retrying assemble_document for job %s (attempt %d/%d) "
                        "after S3 upload failure: %s",
                        job_id,
                        retry_num,
                        self.max_retries,
                        exc,
                    )
                    raise self.retry(exc=exc)
                except MaxRetriesExceededError:
                    logger.error(
                        "assemble_document permanently failed for job %s after "
                        "%d retries (S3 upload): %s",
                        job_id,
                        self.max_retries,
                        exc,
                    )
                    return _fail_assembly_job(
                        job_id,
                        document_id,
                        f"S3 upload failed during assembly: {type(exc).__name__}",
                        {
                            "failure_kind": "assembly_publication_failed",
                            "pages_assembled": pages_assembled,
                            "total_pages": job.total_pages,
                        },
                        backend,
                        claim_token=assembly_claim_token,
                    )
            return _fail_assembly_job(
                job_id,
                document_id,
                f"assembly publication failed ({type(exc).__name__})",
                {
                    "failure_kind": "assembly_publication_failed",
                    "pages_assembled": pages_assembled,
                    "total_pages": job.total_pages,
                },
                backend,
                claim_token=assembly_claim_token,
            )

        return {
            "status": "assembled",
            "job_id": str(job_id),
            "pages_assembled": pages_assembled,
        }
    finally:
        if merged_pdf is not None:
            merged_pdf.close()
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
        for staged_path in (staged_text_path, staged_pdf_path):
            if staged_path and os.path.isfile(staged_path):
                try:
                    os.remove(staged_path)
                except OSError:
                    logger.warning(
                        "Could not remove assembly staging file for job %s",
                        job_id,
                    )
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


_TERMINAL_JOB_STATUSES = frozenset(
    (Job.Status.COMPLETED, Job.Status.FAILED, Job.Status.CANCELLED)
)
_POSTPROCESSING_STAGES = frozenset(("compression", "entities", "structured_data"))


def _validate_postprocessing_results(header_results, job_id):
    """Validate every result delivered by the post-processing chord header."""
    if not isinstance(header_results, (list, tuple)) or len(header_results) != 3:
        raise RuntimeError("post-processing chord returned an invalid result set")

    by_stage = {}
    for result in header_results:
        if not isinstance(result, dict):
            raise RuntimeError("post-processing chord returned a non-object result")
        stage = result.get("stage")
        if stage not in _POSTPROCESSING_STAGES or stage in by_stage:
            raise RuntimeError(
                "post-processing chord returned missing/duplicate stages"
            )
        if result.get("job_id") != str(job_id):
            raise RuntimeError(f"post-processing {stage} result belongs to another job")
        by_stage[stage] = result

    if frozenset(by_stage) != _POSTPROCESSING_STAGES:
        raise RuntimeError("post-processing chord did not return every required stage")

    compression = by_stage["compression"]
    expected_dispositions = {
        "compressed": "APPLIED",
        "skipped_visual_budget": "SKIPPED_VISUAL_BUDGET",
        "skipped_candidate_rejected": "SKIPPED_CANDIDATE_REJECTED",
    }
    comp_status = compression.get("status")
    if (
        comp_status not in expected_dispositions
        or expected_dispositions[comp_status] != compression.get("disposition")
        or not _is_sha256_hex(compression.get("pdf_sha256"))
        or isinstance(compression.get("custody_event_id"), bool)
        or not isinstance(compression.get("custody_event_id"), int)
    ):
        raise RuntimeError("post-processing compression result is invalid")

    entities = by_stage["entities"]
    degraded_entities_allowed = (
        bool(getattr(settings, "ALLOW_DEGRADED_NER", False))
        and entities.get("status") == "degraded_regex"
        and entities.get("engine") == "regex_only"
    )
    if (entities.get("status") != "extracted" and not degraded_entities_allowed) or (
        isinstance(entities.get("entity_count"), bool)
        or not isinstance(entities.get("entity_count"), int)
        or entities["entity_count"] < 0
    ):
        raise RuntimeError("post-processing entity extraction did not succeed")

    structured = by_stage["structured_data"]
    structured_ok = structured.get("status") == "extracted"
    structured_not_requested = (
        structured.get("status") == "skipped"
        and structured.get("reason") == "not_requested"
    )
    if not (structured_ok or structured_not_requested):
        raise RuntimeError("requested structured extraction did not succeed")
    return by_stage


def _build_custody_records(events):
    """Export ID-ordered sealed envelopes; never rewrite existing tip hashes.

    R4-RC-01: if an event already has ``event_hash``, that sealed hash is the
    authority. Re-deriving from mutable ``data`` and overwriting is forbidden.
    Unsealed legacy rows (blank hash) are sealed once here, then marked
    ``chain_finalized``.
    """
    records = []
    prev_hash = None
    for event in events:
        stored_prev = (event.prev_hash or "") or None
        stored_hash = (event.event_hash or "").strip()
        if stored_hash:
            # Sealed tip — export as stored; refuse silent rehash.
            export_prev = stored_prev
            if prev_hash is not None and stored_prev not in (prev_hash, None, ""):
                # allow genesis null vs ""
                if stored_prev != prev_hash:
                    raise RuntimeError(
                        "custody chain tip mismatch: event "
                        f"{event.id} prev_hash does not link prior tip "
                        "(possible forge/rewrite)"
                    )
            # PA-A10: re-verify sealed hash against seal-at-insert envelope.
            # Mismatch => post-seal data mutation (or foreign hash). Opt-out via
            # CUSTODY_EXPORT_REVERIFY=false only for emergency recovery tooling.
            verify_env = (
                os.environ.get("CUSTODY_EXPORT_REVERIFY", "true").strip().lower()
            )
            if verify_env not in ("0", "false", "no", "off"):
                ts = event.timestamp.isoformat(timespec="milliseconds")
                data = event.data if event.data is not None else {}
                # Match _record_custody_event envelope (prev_hash None for genesis).
                prev_for_hash = stored_prev if stored_prev not in (None, "") else None
                candidates = [
                    {
                        "document_id": event.document_id,
                        "event_type": event.event_type,
                        "timestamp": ts,
                        "data": data,
                        "prev_hash": prev_for_hash,
                    },
                    {
                        "document_id": event.document_id,
                        "event_type": event.event_type,
                        "timestamp": ts,
                        "data": data,
                        "prev_hash": stored_prev or "",
                    },
                ]
                if int(getattr(event, "hash_version", 1) or 1) >= 2:
                    candidates = [
                        {
                            **candidate,
                            "chain_id": event.chain_id,
                            "hash_version": int(event.hash_version),
                        }
                        for candidate in candidates
                    ]
                recomputed = {
                    hashlib.sha256(
                        json.dumps(c, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest()
                    for c in candidates
                }
                if stored_hash not in recomputed:
                    raise RuntimeError(
                        "custody sealed hash re-verify failed: event "
                        f"{event.id} event_hash does not match data envelope "
                        "(possible post-seal mutation; PA-A10)"
                    )
            event_dict = {
                "chain_id": event.chain_id,
                "hash_version": int(getattr(event, "hash_version", 1) or 1),
                "document_id": event.document_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(timespec="milliseconds"),
                "data": event.data,
                "prev_hash": export_prev,
                "hash": stored_hash,
            }
            records.append(event_dict)
            event.chain_finalized = True
            prev_hash = stored_hash
            continue

        # RC-07 / R4-RC-01: blank-hash reseal from mutable DB is a forge surface.
        # Default refuse; opt-in only for one-shot legacy migration tooling.
        # PA-A18: production never honors LEGACY blank reseal (hard refuse).
        dep = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
        allow_legacy = os.environ.get(
            "CUSTODY_ALLOW_LEGACY_BLANK_RESEAL", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        if dep == "production":
            allow_legacy = False
        if not allow_legacy:
            raise RuntimeError(
                "custody event "
                f"{event.id} has blank event_hash; refuse reseal from mutable "
                "DB (RC-07/R4-RC-01). Re-emit sealed events or set "
                "CUSTODY_ALLOW_LEGACY_BLANK_RESEAL=true for one-shot migration."
            )
        event_dict = {
            "chain_id": event.chain_id,
            "hash_version": int(getattr(event, "hash_version", 1) or 1),
            "document_id": event.document_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(timespec="milliseconds"),
            "data": event.data,
            "prev_hash": prev_hash,
        }
        event_hash = hashlib.sha256(
            json.dumps(event_dict, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        records.append({**event_dict, "hash": event_hash})
        event.prev_hash = prev_hash or ""
        event.event_hash = event_hash
        event.chain_finalized = True
        prev_hash = event_hash
    return records


def _write_custody_records(file_obj, records):
    for record in records:
        file_obj.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _durable_publish_nfs_custody(records, jsonl_path):
    """Atomically publish a flushed JSONL file on the destination filesystem."""
    custody_dir = os.path.dirname(jsonl_path)
    configured_root = getattr(settings, "NFS_ROOT", custody_dir)
    root_for_create = configured_root
    configured_root_abs = os.path.realpath(
        os.path.abspath(os.path.expanduser(configured_root))
    )
    custody_dir_abs = os.path.abspath(custody_dir)
    try:
        contained = (
            os.path.commonpath([configured_root_abs, custody_dir_abs])
            == configured_root_abs
        )
    except ValueError:
        contained = False
    if not contained:
        deployment_env = (
            str(
                os.environ.get(
                    "DEPLOYMENT_ENV", getattr(settings, "DEPLOYMENT_ENV", "development")
                )
            )
            .strip()
            .lower()
        )
        if deployment_env in {"production", "prod"}:
            raise RuntimeError("NFS custody path escapes NFS_ROOT")
        # Lightweight/local fixtures historically use an isolated temporary
        # job root.  Keep that compatibility explicit while production
        # workers remain bound to the configured NFS root.
        root_for_create = os.path.dirname(custody_dir_abs)
    deployment_env = (
        str(
            os.environ.get(
                "DEPLOYMENT_ENV", getattr(settings, "DEPLOYMENT_ENV", "development")
            )
        )
        .strip()
        .lower()
    )
    if not os.path.isdir(root_for_create):
        if deployment_env in {"production", "prod"}:
            raise RuntimeError("NFS custody root is unavailable")
        os.makedirs(root_for_create, mode=0o700, exist_ok=True)
    _mkdir_tree_no_follow(custody_dir, root_for_create)
    fd, temp_path = tempfile.mkstemp(
        dir=custody_dir,
        prefix=f".{os.path.basename(jsonl_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file_obj:
            fd = -1
            _write_custody_records(file_obj, records)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        # This helper uses os.replace + parent-directory fsync on POSIX and
        # MoveFileExW(REPLACE_EXISTING|WRITE_THROUGH) on Windows.
        from optimize_pdfs import _durable_replace

        _durable_replace(temp_path, jsonl_path)
        temp_path = ""
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


def _write_durable_s3_custody_temp(records):
    """Create a closed, flushed, fsynced JSONL file for an S3 upload."""
    fd, jsonl_path = tempfile.mkstemp(prefix="custody_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file_obj:
            fd = -1
            _write_custody_records(file_obj, records)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        return jsonl_path
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            os.remove(jsonl_path)
        except FileNotFoundError:
            pass
        raise


def _locked_custody_events(job, document_id):
    return list(
        CustodyEvent.objects.select_for_update()
        .filter(job=job, document_id=document_id)
        .order_by("id")
    )


def _prepare_custody_chain(job, document_id):
    """Hash and persist an ID-ordered chain inside the current transaction."""
    events = _locked_custody_events(job, document_id)
    records = _build_custody_records(events)
    if events:
        CustodyEvent.objects.bulk_update(
            events,
            ("prev_hash", "event_hash", "chain_finalized"),
        )
    return records


def _durable_publish_s3_custody(records, job_id, document_id, backend):
    """Fsync a closed temp JSONL and upload it as one S3 object."""
    jsonl_path = _write_durable_s3_custody_temp(records)
    try:
        custody_key = _job_storage_key(
            job_id,
            f"output/EXPORT/CUSTODY/{document_id}.custody.jsonl",
        )
        backend.upload_file(jsonl_path, custody_key)
    finally:
        try:
            os.remove(jsonl_path)
        except FileNotFoundError:
            pass


def _publish_custody_after_commit(job_id, document_id, records, publisher):
    """Publish only after commit; convert publication errors into failure custody."""
    try:
        publisher(records)
    except Exception as exc:
        publish_exc = exc
        message = f"custody finalization/export failed after commit: {exc}"
    else:
        try:
            Job.objects.filter(
                job_id=job_id,
                status__in=_TERMINAL_JOB_STATUSES,
            ).update(custody_published_at=timezone.now())
        except Exception:
            # The external artifact is already durable. Leave the nullable
            # marker for the periodic reconciler, which will observe the
            # existing artifact and repair the marker without failing the job.
            logger.exception(
                "Custody published but marker update failed for job %s", job_id
            )
        return

    failure_records = None
    try:
        with transaction.atomic():
            job = Job.objects.select_for_update().get(job_id=job_id)
            _record_finalization_failure(job, document_id, message)
            if job.status != Job.Status.CANCELLED:
                job.status = Job.Status.FAILED
                job.error_message = message
                job.completed_at = timezone.now()
                job.save(update_fields=["status", "error_message", "completed_at"])
            failure_records = _prepare_custody_chain(job, document_id)
    except Exception:
        logger.critical(
            "Could not persist post-commit custody failure for job %s",
            job_id,
            exc_info=True,
        )
    if failure_records is not None:
        try:
            publisher(failure_records)
        except Exception:
            logger.critical(
                "Could not publish failure-shaped custody for job %s",
                job_id,
                exc_info=True,
            )
    raise RuntimeError(f"Cannot finalize job {job_id}: {message}") from publish_exc


def _schedule_nfs_custody_publish(job, document_id, job_path):
    """Prepare NFS custody now and schedule durable publication after commit."""
    records = _prepare_custody_chain(job, document_id)
    custody_dir = os.path.join(job_path, "output", "EXPORT", "CUSTODY")
    jsonl_path = os.path.join(custody_dir, f"{document_id}.custody.jsonl")
    publisher = partial(_durable_publish_nfs_custody, jsonl_path=jsonl_path)
    transaction.on_commit(
        partial(
            _publish_custody_after_commit,
            job.job_id,
            document_id,
            records,
            publisher,
        )
    )


def _schedule_s3_custody_publish(job, document_id, backend):
    """Prepare S3 custody now and schedule upload after commit."""
    records = _prepare_custody_chain(job, document_id)
    publisher = partial(
        _durable_publish_s3_custody,
        job_id=job.job_id,
        document_id=document_id,
        backend=backend,
    )
    transaction.on_commit(
        partial(
            _publish_custody_after_commit,
            job.job_id,
            document_id,
            records,
            publisher,
        )
    )


def _finalize_custody_chain(job, document_id, job_path):
    """Compatibility wrapper: schedule NFS custody for post-commit publish."""
    _schedule_nfs_custody_publish(job, document_id, job_path)


def _finalize_custody_chain_s3(job, document_id, backend):
    """Compatibility wrapper: schedule S3 custody for post-commit upload."""
    _schedule_s3_custody_publish(job, document_id, backend)


@shared_task(name="jobs.tasks.republish_pending_custody", queue="coordinator")
def republish_pending_custody(limit=1000):
    from .rls import control_plane_rls_context

    with control_plane_rls_context():
        return _republish_pending_custody_control_plane(limit=limit)


def _republish_pending_custody_control_plane(limit=1000):
    """Repair terminal jobs whose post-commit custody publication was lost.

    Database finalization and artifact publication intentionally use separate
    commit boundaries.  A coordinator crash in that interval must become a
    bounded, observable retry rather than a permanent custody gap.
    """
    try:
        scan_limit = int(limit)
    except (TypeError, ValueError):
        return {"scanned": 0, "published": 0, "failures": ["invalid limit"]}
    scan_limit = max(1, min(scan_limit, 1000))
    candidates = list(
        Job.objects.filter(
            status__in=_TERMINAL_JOB_STATUSES,
            custody_published_at__isnull=True,
        ).order_by("completed_at", "job_id")[:scan_limit]
    )
    published = 0
    failures = []
    for candidate in candidates:
        document_id = (
            candidate.source_hash[:16]
            if candidate.source_hash
            else str(candidate.job_id)[:16]
        )
        try:
            backend = _get_backend_for_job(candidate)
            if backend.backend_name == "nfs":
                job_path = _nfs_job_path_for_job(candidate)
                jsonl_path = os.path.join(
                    job_path,
                    "output",
                    "EXPORT",
                    "CUSTODY",
                    f"{document_id}.custody.jsonl",
                )
                try:
                    metadata = os.lstat(jsonl_path)
                    exists = stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
                except FileNotFoundError:
                    exists = False
                if exists:
                    Job.objects.filter(
                        pk=candidate.pk,
                        status__in=_TERMINAL_JOB_STATUSES,
                        custody_published_at__isnull=True,
                    ).update(custody_published_at=timezone.now())
                    continue
            else:
                custody_key = _job_storage_key(
                    candidate.job_id,
                    f"output/EXPORT/CUSTODY/{document_id}.custody.jsonl",
                )
                if backend.exists(custody_key):
                    Job.objects.filter(
                        pk=candidate.pk,
                        status__in=_TERMINAL_JOB_STATUSES,
                        custody_published_at__isnull=True,
                    ).update(custody_published_at=timezone.now())
                    continue

            with transaction.atomic():
                job = Job.objects.select_for_update().get(pk=candidate.pk)
                if job.status not in _TERMINAL_JOB_STATUSES:
                    continue
                records = _prepare_custody_chain(job, document_id)
                if not records:
                    continue
                if backend.backend_name == "nfs":
                    job_path = _nfs_job_path_for_job(job)
                    jsonl_path = os.path.join(
                        job_path,
                        "output",
                        "EXPORT",
                        "CUSTODY",
                        f"{document_id}.custody.jsonl",
                    )
                else:
                    custody_key = _job_storage_key(
                        job.job_id,
                        f"output/EXPORT/CUSTODY/{document_id}.custody.jsonl",
                    )
            if backend.backend_name == "nfs":
                _durable_publish_nfs_custody(records, jsonl_path)
            else:
                _durable_publish_s3_custody(
                    records,
                    job_id=job.job_id,
                    document_id=document_id,
                    backend=backend,
                )
            Job.objects.filter(
                pk=candidate.pk,
                status__in=_TERMINAL_JOB_STATUSES,
            ).update(custody_published_at=timezone.now())
            published += 1
        except Exception as exc:
            failures.append(f"job {candidate.job_id}: {type(exc).__name__}")
            logger.exception("Could not republish custody for job %s", candidate.job_id)
    return {"scanned": len(candidates), "published": published, "failures": failures}


def _summarize_page_results(job):
    page_results = list(PageResult.objects.filter(job=job).order_by("page_num"))
    counts = {status: 0 for status in ("ok", "fallback", "image_only", "failed")}
    invalid_statuses = []
    for page_result in page_results:
        if page_result.status in counts:
            counts[page_result.status] += 1
        else:
            invalid_statuses.append(page_result.status)

    confidences = [
        page_result.ocr_confidence
        for page_result in page_results
        if page_result.status in {"ok", "fallback"}
        and page_result.ocr_confidence >= 0
    ]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    review_pages = [
        row.page_num
        for row in page_results
        if row.needs_review
        or not isinstance(row.ocr_provenance, dict)
        or not row.ocr_provenance.get("assessment_version")
    ]
    actual_page_numbers = [page_result.page_num for page_result in page_results]
    expected_page_numbers = list(range(1, job.total_pages + 1))
    coverage_complete = (
        actual_page_numbers == expected_page_numbers and not invalid_statuses
    )
    pages_accounted_for = sum(counts.values())
    summary = {
        "total_pages": job.total_pages,
        "pages_ok": counts["ok"],
        "pages_fallback": counts["fallback"],
        "pages_image_only": counts["image_only"],
        "pages_failed": counts["failed"],
        "pages_accounted_for": pages_accounted_for,
        "average_confidence": round(average_confidence, 4),
        "pages_needing_review": len(review_pages),
        "review_page_numbers": review_pages[:100],
        "review_page_numbers_truncated": len(review_pages) > 100,
    }
    if counts["failed"]:
        failure_message = f"{counts['failed']} page(s) failed during processing"
    elif not coverage_complete:
        failure_message = (
            "page-result coverage is incomplete or non-terminal: "
            f"accounted_for={pages_accounted_for}, total={job.total_pages}"
        )
    else:
        failure_message = ""
    return summary, failure_message


def _record_finalization_failure(job, document_id, message, *, causal_event_id=None):
    data = {"error": message}
    if causal_event_id is not None:
        data["causal_event_id"] = causal_event_id
    return _record_custody_event(
        job,
        document_id,
        "processing_failed",
        data=data,
        worker_hostname=socket.gethostname(),
    )


def _terminal_job_result(job, job_id):
    status = str(job.status)
    return {
        "status": status,
        "job_id": str(job_id),
        "message": job.error_message or f"job was already {status}",
    }


def _finalization_snapshot(job, document_id, validated_results):
    """Validate the DB-only finalization inputs and return immutable IDs/data."""
    assembly_event = (
        CustodyEvent.objects.filter(
            job=job,
            document_id=document_id,
            event_type="assembly_complete",
        )
        .order_by("-id")
        .first()
    )
    compression_event = (
        CustodyEvent.objects.filter(
            job=job,
            document_id=document_id,
            event_type__in=("compression_complete", "compression_skipped"),
        )
        .order_by("-id")
        .first()
    )
    assembly_data = (
        assembly_event.data
        if assembly_event is not None and isinstance(assembly_event.data, dict)
        else {}
    )
    compression_data = (
        compression_event.data
        if compression_event is not None and isinstance(compression_event.data, dict)
        else {}
    )
    pages_assembled = assembly_data.get("pages_assembled")
    pdf_sha256 = compression_data.get("pdf_sha256", "")
    allowed_event_dispositions = {
        "compression_complete": frozenset(("APPLIED",)),
        "compression_skipped": frozenset((
            "SKIPPED_VISUAL_BUDGET", "SKIPPED_CANDIDATE_REJECTED",
        )),
    }
    compression_result = validated_results["compression"]
    if (
        assembly_event is None
        or compression_event is None
        or isinstance(pages_assembled, bool)
        or not isinstance(pages_assembled, int)
        or pages_assembled != job.total_pages
        or compression_event.id <= assembly_event.id
        or compression_data.get("causal_event_id") != assembly_event.id
        or compression_result.get("custody_event_id") != compression_event.id
        or compression_data.get("disposition") not in
        allowed_event_dispositions.get(compression_event.event_type, ())
        or compression_result.get("disposition") != compression_data.get("disposition")
        or compression_result.get("pdf_sha256") != pdf_sha256
        or not _is_sha256_hex(pdf_sha256)
    ):
        raise RuntimeError(
            "content-bound compression custody outcome is missing or invalid"
        )

    result_summary, page_failure = _summarize_page_results(job)
    return {
        "assembly_event_id": assembly_event.id,
        "compression_event_id": compression_event.id,
        "compression_data": dict(compression_data),
        "result_summary": result_summary,
        "page_failure": page_failure,
    }


def _verify_finalization_artifact(job, snapshot):
    """Verify the content-bound final PDF without a DB transaction or row lock."""
    backend = _get_backend_for_job(job)
    compression_data = snapshot["compression_data"]
    pdf_sha256 = compression_data["pdf_sha256"]
    active = _resolve_job_output_artifacts(job, backend)
    pdf_locator = active.pdf.locator
    if backend.backend_name == "nfs":
        job_path = _nfs_job_path_for_job(job)
        if not os.path.isfile(pdf_locator):
            raise FileNotFoundError(
                f"assembled PDF missing before custody finalization: {pdf_locator}"
            )
        if compression_data.get("pdf_path") != pdf_locator:
            raise RuntimeError(
                "compression custody locator does not identify the final PDF"
            )
        if active.pdf.sha256 and active.pdf.sha256 != pdf_sha256:
            raise RuntimeError(
                "active PDF pointer hash no longer matches compression custody"
            )
        if _compute_file_hash(pdf_locator) != pdf_sha256:
            raise RuntimeError(
                "assembled PDF hash no longer matches compression custody outcome"
            )
        return backend, job_path

    pdf_key = pdf_locator
    if compression_data.get("pdf_path") != pdf_key:
        raise RuntimeError(
            "compression custody locator does not identify the final PDF object"
        )
    if active.pdf.sha256 and active.pdf.sha256 != pdf_sha256:
        raise RuntimeError(
            "active PDF pointer hash no longer matches compression custody"
        )
    verification_backend = (
        backend.inner if isinstance(backend, CachedS3Backend) else backend
    )
    if not verification_backend.exists(pdf_key):
        raise FileNotFoundError(
            f"assembled PDF object missing before custody finalization: {pdf_key}"
        )
    verify_tmp = tempfile.NamedTemporaryFile(
        suffix=".pdf",
        prefix=f"finalize_{job.job_id}_",
        delete=False,
    )
    verify_path = verify_tmp.name
    verify_tmp.close()
    try:
        verification_backend.download_file(pdf_key, verify_path)
        if not os.path.isfile(verify_path):
            raise FileNotFoundError(
                "assembled PDF object download did not create a local file"
            )
        if _compute_file_hash(verify_path) != pdf_sha256:
            raise RuntimeError(
                "assembled PDF object hash no longer matches compression custody outcome"
            )
    finally:
        try:
            os.remove(verify_path)
        except FileNotFoundError:
            pass
    return backend, pdf_key


def _format_finalization_error(exc):
    message = str(exc)
    if "custody finalization/export failed after commit:" in message:
        return message
    if not message.startswith("custody finalization/export failed:"):
        if "compression custody outcome" not in message and not message.startswith(
            "post-processing"
        ):
            message = f"custody finalization/export failed: {message}"
    return message


def _fail_finalization(job_id, document_id, message):
    """Persist a failure and schedule failure-shaped custody after commit."""
    with transaction.atomic():
        job = Job.objects.select_for_update().get(job_id=job_id)
        if job.status in _TERMINAL_JOB_STATUSES:
            return _terminal_job_result(job, job_id)
        _record_finalization_failure(job, document_id, message)
        job.status = Job.Status.FAILED
        job.error_message = message
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])
        backend = _get_backend_for_job(job)
        if backend.backend_name == "nfs":
            _schedule_nfs_custody_publish(
                job,
                document_id,
                _nfs_job_path_for_job(job),
            )
        else:
            _schedule_s3_custody_publish(job, document_id, backend)
    return None


@shared_task(bind=True, name="jobs.tasks.finalize_job", queue="coordinator")
@_tenant_bound_finalize_task
def finalize_job(self, header_results, job_id, tenant_id=None):
    """Finalize through short DB phases with artifact I/O outside row locks."""
    document_id = str(job_id)[:16]
    try:
        with transaction.atomic():
            try:
                job = Job.objects.select_for_update().get(job_id=job_id)
            except Job.DoesNotExist:
                return {"status": "error", "message": f"Job {job_id} not found"}
            _pin_rls_for_job(job)
            if job.status in _TERMINAL_JOB_STATUSES:
                return _terminal_job_result(job, job_id)
            document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
            if job.status != Job.Status.ASSEMBLING:
                raise RuntimeError(
                    "post-processing callback requires the job to be assembling"
                )
            validated_results = _validate_postprocessing_results(
                header_results,
                job_id,
            )
            snapshot = _finalization_snapshot(
                job,
                document_id,
                validated_results,
            )
    except Exception as exc:
        message = _format_finalization_error(exc)
        terminal_result = _fail_finalization(job_id, document_id, message)
        if terminal_result is not None:
            return terminal_result
        raise RuntimeError(f"Cannot finalize job {job_id}: {message}") from exc

    try:
        backend, artifact_locator = _verify_finalization_artifact(job, snapshot)
    except Exception as exc:
        message = _format_finalization_error(exc)
        terminal_result = _fail_finalization(job_id, document_id, message)
        if terminal_result is not None:
            return terminal_result
        raise RuntimeError(f"Cannot finalize job {job_id}: {message}") from exc

    try:
        with transaction.atomic():
            locked_job = Job.objects.select_for_update().get(job_id=job_id)
            if locked_job.status in _TERMINAL_JOB_STATUSES:
                return _terminal_job_result(locked_job, job_id)
            if locked_job.status != Job.Status.ASSEMBLING:
                raise RuntimeError(
                    "post-processing reconciliation requires the job to be assembling"
                )

            current = _finalization_snapshot(
                locked_job,
                document_id,
                validated_results,
            )
            stable_fields = (
                "assembly_event_id",
                "compression_event_id",
                "compression_data",
                "result_summary",
                "page_failure",
            )
            if any(current[field] != snapshot[field] for field in stable_fields):
                raise RuntimeError(
                    "finalization inputs changed during artifact verification"
                )

            page_failure = current["page_failure"]
            result_summary = current["result_summary"]
            if page_failure:
                _record_finalization_failure(
                    locked_job,
                    document_id,
                    page_failure,
                    causal_event_id=current["compression_event_id"],
                )

            desired_status = Job.Status.FAILED if page_failure else Job.Status.COMPLETED
            locked_job.status = desired_status
            locked_job.completed_at = timezone.now()
            locked_job.result_summary = result_summary
            locked_job.pages_failed = result_summary["pages_failed"]
            locked_job.error_message = page_failure or ""
            locked_job.save(
                update_fields=[
                    "status",
                    "completed_at",
                    "result_summary",
                    "pages_failed",
                    "error_message",
                ]
            )
            if backend.backend_name == "nfs":
                _schedule_nfs_custody_publish(
                    locked_job,
                    document_id,
                    artifact_locator,
                )
            else:
                _schedule_s3_custody_publish(
                    locked_job,
                    document_id,
                    backend,
                )

            task_status = "failed" if page_failure else "completed"
            result = {
                "status": task_status,
                "job_id": str(job_id),
                "summary": result_summary,
            }
            webhook_job = locked_job
    except Exception as exc:
        message = _format_finalization_error(exc)
        terminal_result = _fail_finalization(job_id, document_id, message)
        if "custody finalization/export failed after commit:" in str(exc):
            raise RuntimeError(f"Cannot finalize job {job_id}: {message}") from exc
        if terminal_result is not None:
            return terminal_result
        raise RuntimeError(f"Cannot finalize job {job_id}: {message}") from exc

    if webhook_job.webhook_url:
        _send_webhook(webhook_job, result["summary"])
    logger.info(
        "Job %s finalized (%s): %s",
        job_id,
        result["status"],
        result["summary"],
    )
    return result


def _deliver_webhook_with_retry(
    url,
    body,
    headers,
    max_attempts=3,
    *,
    allow_http=False,
    allow_private=False,
):
    """Attempt webhook delivery with exponential backoff.

    Returns ``True`` on success, ``False`` after all attempts are exhausted.
    """
    import urllib.request

    from ocr_distributed.ssrf import (  # noqa: F811 – lazy import
        pinned_url_for_request,
        safe_opener,
    )

    for attempt in range(max_attempts):
        try:
            with pinned_url_for_request(
                url,
                allow_http=allow_http,
                allow_private=allow_private,
            ) as delivery_url:
                req = urllib.request.Request(
                    delivery_url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                safe_opener.open(req, timeout=10)
            return True
        except Exception as exc:
            logger.warning(
                "Webhook attempt %d/%d failed: %s",
                attempt + 1,
                max_attempts,
                exc,
            )
            if attempt < max_attempts - 1:
                time.sleep(_WEBHOOK_RETRY_DELAYS[attempt])
    return False


def _send_webhook(job, payload):
    """Send HMAC-SHA256 signed webhook notification with SSRF protection."""
    import hmac

    from ocr_distributed.ssrf import validate_webhook_url

    # TOCTOU: re-validate URL at delivery time
    try:
        validate_webhook_url(
            job.webhook_url,
            allow_http=getattr(settings, "WEBHOOK_ALLOW_HTTP", False),
            allow_private=getattr(settings, "WEBHOOK_ALLOW_PRIVATE", False),
        )
    except ValueError as exc:
        logger.warning("Webhook URL for job %s failed validation: %s", job.job_id, exc)
        job.webhook_status = "failed"
        job.save(update_fields=["webhook_status"])
        return

    event_name = "job.failed" if job.status == Job.Status.FAILED else "job.completed"
    body = json.dumps(
        {
            "event": event_name,
            "job_id": str(job.job_id),
            "status": job.status,
            "result": payload,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}

    if job.webhook_secret:
        # LT-07: single canonical scheme — timestamped X-Webhook-Signature
        # (matches api.webhooks.compute_signature). Legacy body-only
        # X-OCR-Signature only when WEBHOOK_LEGACY_BODY_HMAC=true (lab).
        timestamp = int(time.time())
        payload_json = body.decode("utf-8")
        api_msg = f"{timestamp}.{payload_json}".encode("utf-8")
        api_sig = hmac.new(
            job.webhook_secret.encode("utf-8"),
            api_msg,
            hashlib.sha256,
        ).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={api_sig}"
        headers["X-Webhook-Timestamp"] = str(timestamp)
        import os as _os_wh

        if (_os_wh.environ.get("WEBHOOK_LEGACY_BODY_HMAC") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            legacy_sig = hmac.new(
                job.webhook_secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-OCR-Signature"] = f"sha256={legacy_sig}"

    success = _deliver_webhook_with_retry(
        job.webhook_url,
        body,
        headers,
        allow_http=getattr(settings, "WEBHOOK_ALLOW_HTTP", False),
        allow_private=getattr(settings, "WEBHOOK_ALLOW_PRIVATE", False),
    )
    job.webhook_status = "delivered" if success else "failed"
    if not success:
        logger.error("Webhook delivery failed for job %s after all retries", job.job_id)
    job.save(update_fields=["webhook_status"])


# ---------------------------------------------------------------------------
# Periodic Tasks (Celery Beat)
# ---------------------------------------------------------------------------


def _terminalize_dead_dispatch(dispatch_id):
    """Fail the owning job and seal custody when publication is exhausted."""
    now = timezone.now()
    with transaction.atomic():
        row = (
            TaskDispatchOutbox.objects.select_for_update()
            .select_related("job")
            .get(pk=dispatch_id)
        )
        if row.status != TaskDispatchOutbox.Status.DEAD:
            return False
        job = Job.objects.select_for_update().get(pk=row.job_id)
        if str(row.generation) != str(job.dispatch_generation):
            # DEAD belongs to an earlier execution generation.  It remains an
            # immutable audit row but has no authority to terminalize a retry.
            return False
        if job.status in _TERMINAL_JOB_STATUSES:
            return False
        message = (
            f"Task dispatch dead-lettered at stage {row.stage}: "
            f"{row.last_error or 'broker publication retry budget exhausted'}"
        )[:8_000]
        job.status = Job.Status.FAILED
        job.error_message = message
        job.last_failure_reason = "dispatch_dead_lettered"
        job.completed_at = now
        job.save(
            update_fields=[
                "status",
                "error_message",
                "last_failure_reason",
                "completed_at",
            ]
        )
        TaskDispatchOutbox.objects.filter(
            job=job,
            status__in=[
                TaskDispatchOutbox.Status.PENDING,
                TaskDispatchOutbox.Status.FAILED,
            ],
        ).update(
            status=TaskDispatchOutbox.Status.DEAD,
            dead_at=now,
            last_error="owning job terminalized after dispatch dead-letter",
        )
        document_id = job.source_hash[:16] if job.source_hash else str(job.job_id)[:16]
        _record_custody_event(
            job,
            document_id,
            "dispatch_dead_lettered",
            data={
                "dispatch_id": str(row.dispatch_id),
                "stage": row.stage,
                "task_name": row.task_name,
                "attempts": row.attempts,
                "error": row.last_error,
            },
            worker_hostname=socket.gethostname(),
        )
    return True


def _reconcile_dead_dispatches(limit=100):
    """Repair DEAD-outbox/job terminalization crash gaps idempotently."""
    from .rls import control_plane_rls_context

    limit = max(1, min(1_000, int(limit)))
    with control_plane_rls_context():
        dispatch_ids = list(
            TaskDispatchOutbox.objects.filter(
                status=TaskDispatchOutbox.Status.DEAD,
            )
            .exclude(job__status__in=_TERMINAL_JOB_STATUSES)
            .filter(
                generation=Cast(
                    db_models.F("job__dispatch_generation"),
                    output_field=db_models.CharField(),
                )
            )
            .order_by("dead_at", "dispatch_id")
            .values_list("dispatch_id", flat=True)[:limit]
        )
        reconciled = 0
        failures = []
        for dispatch_id in dispatch_ids:
            try:
                reconciled += int(bool(_terminalize_dead_dispatch(dispatch_id)))
            except Exception as exc:
                failures.append(f"{dispatch_id}: {exc}")
                logger.exception("Failed to reconcile dead dispatch %s", dispatch_id)
    return {
        "examined": len(dispatch_ids),
        "reconciled": reconciled,
        "failures": failures,
    }


@shared_task(name="jobs.tasks.check_worker_heartbeats", queue="coordinator")
def check_worker_heartbeats():
    """Mark workers as offline if heartbeat is stale (>2 minutes)."""
    cutoff = timezone.now() - timezone.timedelta(minutes=2)
    stale = Worker.objects.filter(
        status__in=[Worker.Status.ONLINE, Worker.Status.BUSY],
        last_heartbeat__lt=cutoff,
    )
    count = stale.update(status=Worker.Status.OFFLINE)
    if count:
        logger.warning("Marked %d workers as offline (stale heartbeat)", count)
    return {"stale_workers": count}


@shared_task(name="jobs.tasks.relay_task_dispatch_outbox", queue="coordinator")
def relay_task_dispatch_outbox(limit=100):
    """Republish pending coordinator task contracts after crash/broker gaps."""
    dead_reconciliation = _reconcile_dead_dispatches(limit=limit)
    admitted = _recover_fair_page_admission(limit=limit)
    results = publish_task_dispatch_outbox(limit=limit)
    published = sum(
        result.get("status") == TaskDispatchOutbox.Status.PUBLISHED
        for result in results
    )
    failed = sum(
        result.get("status") == TaskDispatchOutbox.Status.FAILED for result in results
    )
    dead = sum(
        result.get("status") == TaskDispatchOutbox.Status.DEAD for result in results
    )
    return {
        "status": "ok" if failed == 0 and dead == 0 else "degraded",
        "examined": len(results),
        "published": published,
        "failed": failed,
        "dead": dead,
        "dead_reconciliation": dead_reconciliation,
        "pages_admitted": admitted,
    }


def _recover_fair_page_admission(*, limit=100) -> int:
    """Repair completion-to-next-page crash gaps one job at a time.

    Each scan admits at most one page per job in least-recently-admitted
    order. Normal page completion replenishes capacity immediately; this
    periodic path covers coordinator death before that reconciliation.
    """
    if not _fair_fanout_enabled():
        return 0
    from .rls import control_plane_rls_context

    admitted = 0
    with control_plane_rls_context():
        pending = PageResult.objects.filter(
            job_id=OuterRef("pk"), status="pending"
        ).exclude(admitted_generation=OuterRef("dispatch_generation"))
        job_ids = list(
            Job.objects.filter(
                status=Job.Status.PROCESSING,
                fanout_dispatched=True,
                total_pages__gt=0,
            )
            .annotate(
                active_page_count=Count(
                    "page_results",
                    filter=db_models.Q(
                        page_results__admitted_generation=db_models.F("dispatch_generation"),
                        page_results__status__in=("pending", "processing"),
                    ),
                ),
                has_unadmitted_page=Exists(pending),
            )
            .filter(
                active_page_count__lt=_max_inflight_pages_per_document(),
                has_unadmitted_page=True,
            )
            .order_by("last_page_admission_at", "created_at", "job_id")
            .values_list("job_id", flat=True)[: max(1, min(int(limit), 1000))]
        )
        for job_id in job_ids:
            with transaction.atomic():
                job = (
                    Job.objects.select_for_update(skip_locked=True)
                    .filter(job_id=job_id, status=Job.Status.PROCESSING)
                    .first()
                )
                if job is not None:
                    admitted += len(_admit_pending_pages_locked(job, max_new=1))
    return admitted


@shared_task(name="jobs.tasks.cleanup_stale_jobs", queue="coordinator")
def cleanup_stale_jobs():
    from .rls import control_plane_rls_context

    with control_plane_rls_context():
        return _cleanup_stale_jobs_control_plane()


def _cleanup_stale_jobs_control_plane():
    """Flag jobs stuck in processing beyond their configured timeout."""
    _recover_staged_pages()
    now = timezone.now()
    active_jobs = list(
        Job.objects.filter(
            status__in=[
                Job.Status.INGESTING,
                Job.Status.PROCESSING,
                Job.Status.ASSEMBLING,
            ],
            started_at__isnull=False,
        )
    )
    stale_jobs = []
    for job in active_jobs:
        timeout_minutes = _get_job_processing_timeout_minutes(job)
        cutoff = now - timezone.timedelta(minutes=timeout_minutes)
        if job.started_at < cutoff:
            if job.status == Job.Status.PROCESSING and _staged_page_ocr_enabled():
                current_stages = PageOCRStage.objects.filter(
                    job=job,
                    generation=job.dispatch_generation,
                    tenant_id=job.tenant_id,
                    source_sha256=job.source_hash,
                )
                first_stage = current_stages.order_by("created_at").first()
                stage_failed = current_stages.filter(
                    status__in=("failed", "obsolete"),
                ).exists()
                active_stage = current_stages.filter(
                    status__in=(
                        "pending", "rendering", "rendered", "infering",
                        "inferred", "cleanup",
                    ),
                ).exists()
                if first_stage and active_stage and not stage_failed:
                    hard_seconds = max(
                        _bounded_operational_int(
                            "OCR_STAGE_JOB_HARD_TIMEOUT_SECONDS", 6 * 3600,
                            minimum=1800,
                            maximum=7 * 24 * 3600,
                        ),
                        timeout_minutes * 60,
                    )
                    if now < first_stage.created_at + timedelta(seconds=hard_seconds):
                        # A lost finalizer's page lease expires at 30 minutes.
                        # Stage recovery ran above and can re-dispatch it; the
                        # ordinary 30-minute job timeout must not win here.
                        continue
            stale_jobs.append((job, timeout_minutes))

    failed_count = 0
    for job, timeout_minutes in stale_jobs:
        logger.warning(
            "Stale job detected: %s (started %s)", job.job_id, job.started_at
        )
        error_message = (
            "Job exceeded processing timeout "
            f"({_format_processing_timeout(timeout_minutes)})"
        )
        # Completion, cancellation, and stale-job cleanup can race across
        # beat processes.  Treat this as a compare-and-set: only the same
        # active state observed during the scan may be failed.  A late
        # cleanup pass must never overwrite a terminal result published by a
        # worker after the scan.
        updated = Job.objects.filter(
            pk=job.pk,
            status__in=[
                Job.Status.INGESTING,
                Job.Status.PROCESSING,
                Job.Status.ASSEMBLING,
            ],
            started_at=job.started_at,
        ).update(
            status=Job.Status.FAILED,
            error_message=error_message,
            completed_at=now,
        )
        if updated:
            failed_count += 1
        else:
            logger.info(
                "Skipped stale cleanup for job %s because its state changed",
                job.job_id,
            )
    return {"stale_jobs": failed_count}


@shared_task(name="jobs.tasks.redrive_stale_assembling_jobs", queue="coordinator")
def redrive_stale_assembling_jobs():
    from .rls import control_plane_rls_context

    with control_plane_rls_context():
        return _redrive_stale_assembling_jobs_control_plane()


def _redrive_stale_assembling_jobs_control_plane():
    """MP-B1: Re-dispatch post-assembly chords for orphaned assembling jobs.

    Targets jobs that reached assembly_complete but never finalized (lost chord
    consumer, empty nlp_general queue, worker kill mid-group). Does not require
    operator shell re-dispatch.
    """
    results = []
    assembling = Job.objects.filter(status=Job.Status.ASSEMBLING)
    for job in assembling.iterator():
        result = redrive_assembling_postprocessing(str(job.job_id))
        results.append(result)
    redispatched = sum(1 for r in results if r.get("status") == "redispatched")
    failed = sum(1 for r in results if r.get("status") == "failed")
    if redispatched or failed:
        logger.warning(
            "assembling redrive: redispatched=%d failed=%d scanned=%d",
            redispatched,
            failed,
            len(results),
        )
    return {
        "scanned": len(results),
        "redispatched": redispatched,
        "failed": failed,
        "results": results,
    }


@shared_task(name="jobs.tasks.cleanup_completed_jobs", queue="coordinator")
def cleanup_completed_jobs():
    from .rls import control_plane_rls_context

    with control_plane_rls_context():
        return _cleanup_completed_jobs_control_plane()


def _cleanup_completed_jobs_control_plane():
    """Delete completed/failed/cancelled jobs older than JOB_RETENTION_DAYS.

    Runs as a daily Celery Beat task. Retention period is configurable
    via the JOB_RETENTION_DAYS environment variable (default: 30).

    Respects LITIGATION_HOLD env var — skips all deletions when active.
    Emits a CustodyEvent recording the cleanup action before deletion.
    """
    if is_litigation_hold_active():
        logger.info("LITIGATION_HOLD active — cleanup_completed_jobs skipped")
        return {"deleted": 0, "litigation_hold": True}

    try:
        retention_days = int(os.environ.get("JOB_RETENTION_DAYS", "30"))
    except (TypeError, ValueError):
        logger.error("Invalid JOB_RETENTION_DAYS; refusing automated deletion")
        return {"deleted": 0, "cleanup_failures": ["invalid JOB_RETENTION_DAYS"]}
    if retention_days <= 0:
        logger.error("JOB_RETENTION_DAYS must be positive; refusing automated deletion")
        return {"deleted": 0, "cleanup_failures": ["non-positive JOB_RETENTION_DAYS"]}
    cutoff = timezone.now() - timezone.timedelta(days=retention_days)

    candidate_ids = list(
        Job.objects.filter(
            status__in=[Job.Status.COMPLETED, Job.Status.FAILED, Job.Status.CANCELLED],
            completed_at__isnull=False,
            completed_at__lt=cutoff,
        )
        .order_by("job_id")
        .values_list("job_id", flat=True)
    )
    count = len(candidate_ids)
    if count == 0:
        return {
            "deleted": 0,
            "nfs_cleaned": 0,
            "s3_jobs_cleaned": 0,
            "s3_objects_cleaned": 0,
            "cleanup_failures": [],
        }

    # One committed saga per job: START commits before external I/O, then a
    # committed SUCCESS/FAIL boundary, then a short transaction deletes the
    # still-eligible row. No transaction spans NFS/S3, and one failed job does
    # not roll back evidence or deletion for an earlier successful job.
    nfs_cleaned = 0
    s3_jobs_cleaned = 0
    s3_objects_cleaned = 0
    deleted = 0
    cleanup_failures = []
    for candidate_id in candidate_ids:
        with transaction.atomic():
            job = (
                Job.objects.select_for_update()
                .filter(
                    job_id=candidate_id,
                    status__in=[
                        Job.Status.COMPLETED,
                        Job.Status.FAILED,
                        Job.Status.CANCELLED,
                    ],
                    completed_at__isnull=False,
                    completed_at__lt=cutoff,
                )
                .first()
            )
            if job is None:
                continue
            original_status = job.status
            # Retention must preserve the same staged cleanup/capacity
            # authority as an explicit tenant purge. A prefix retirement
            # cannot release a stage's reserved-byte counter after cascade.
            if PageOCRStage.objects.filter(job=job).filter(
                ~Q(status__in=("done", "obsolete", "failed"))
                | Q(reserved_bytes__gt=0)
                | ~Q(lease_token="")
                | Q(lease_expires_at__isnull=False)
                | Q(publish_safe_after__isnull=False)
            ).exists():
                cleanup_failures.append(f"job {job.job_id}: staged OCR cleanup is not drained")
                continue
            job.status = Job.Status.CANCELLED
            job.dispatch_generation = int(job.dispatch_generation or 0) + 1
            job.celery_task_id = ""
            job.last_failure_reason = "retention_deletion_fence"
            job.save(
                update_fields=[
                    "status",
                    "dispatch_generation",
                    "celery_task_id",
                    "last_failure_reason",
                ]
            )
            fenced_generation = job.dispatch_generation
            TaskDispatchOutbox.objects.filter(
                job=job,
                status__in=[
                    TaskDispatchOutbox.Status.PENDING,
                    TaskDispatchOutbox.Status.FAILED,
                ],
            ).update(
                status=TaskDispatchOutbox.Status.DEAD,
                dead_at=timezone.now(),
                last_error="retention deletion fenced dispatch",
            )
            PresignedTaskCapability.objects.filter(job=job).delete()
            audit_tenant = str(job.tenant_id or "").strip() or "__unscoped__"
            audit_document = f"retention-{job.job_id}"
            try:
                _record_control_plane_custody_event(
                    tenant_id=audit_tenant,
                    document_id=audit_document,
                    event_type="data_deletion_started",
                    data={
                        "action": "cleanup_completed_jobs",
                        "job_id": str(job.job_id),
                        "retention_days": retention_days,
                        "original_status": original_status,
                        "fenced_generation": fenced_generation,
                    },
                    worker_hostname="celery-beat",
                )
            except Exception as exc:
                cleanup_failures.append(
                    f"job {job.job_id}: deletion-start custody failed ({exc})"
                )
                transaction.set_rollback(True)
                continue

        audit_tenant = str(job.tenant_id or "").strip() or "__unscoped__"
        audit_document = f"retention-{job.job_id}"
        backend = None
        job_failure = None
        try:
            backend = _get_backend_for_job(job)
            storage_mode = backend.backend_name
            if storage_mode == "nfs":
                raw_path = getattr(job, "nfs_job_path", None)
                candidate = safe_nfs_job_path(
                    raw_path or _nfs_job_path(job.job_id),
                    getattr(settings, "NFS_ROOT", None),
                    job_id=str(job.job_id),
                )
                if candidate is None:
                    job_failure = f"job {job.job_id}: untrusted nfs_job_path"
                elif os.path.lexists(candidate):
                    try:
                        if remove_path_no_follow(candidate):
                            nfs_cleaned += 1
                        else:
                            job_failure = f"job {job.job_id}: NFS directory remains"
                    except (OSError, RuntimeError) as exc:
                        job_failure = f"job {job.job_id}: NFS removal failed ({exc})"
            else:
                job_prefix = _job_storage_key(job.job_id) + "/"
                keys = backend.list_objects(job_prefix)
                if keys:
                    deleted_count = backend.delete_many(keys)
                    remaining = backend.list_objects(job_prefix)
                    if deleted_count != len(keys) or remaining:
                        job_failure = (
                            f"job {job.job_id}: deleted {deleted_count} of "
                            f"{len(keys)} S3 objects; remaining={len(remaining)}"
                        )
                    else:
                        s3_jobs_cleaned += 1
                        s3_objects_cleaned += deleted_count
                        logger.info(
                            "Deleted %d/%d S3 objects for job %s",
                            deleted_count,
                            len(keys),
                            job.job_id,
                        )
        except Exception as exc:
            job_failure = f"job {job.job_id}: storage cleanup failed ({exc})"
            logger.exception("Failed to clean storage for job %s", job.job_id)

        if job_failure is not None:
            cleanup_failures.append(job_failure)
            try:
                _record_control_plane_custody_event(
                    tenant_id=audit_tenant,
                    document_id=audit_document,
                    event_type="data_deletion_failed",
                    data={
                        "action": "cleanup_completed_jobs",
                        "job_id": str(job.job_id),
                        "retention_days": retention_days,
                        "failure": job_failure,
                    },
                    worker_hostname="celery-beat",
                )
            except Exception as exc:
                cleanup_failures.append(
                    f"job {job.job_id}: deletion-failure custody failed ({exc})"
                )
            continue

        try:
            with transaction.atomic():
                locked_job = (
                    Job.objects.select_for_update()
                    .filter(
                        job_id=job.job_id,
                        status=Job.Status.CANCELLED,
                        dispatch_generation=fenced_generation,
                        completed_at__isnull=False,
                        completed_at__lt=cutoff,
                    )
                    .first()
                )
                if locked_job is None:
                    raise RuntimeError(
                        "retention deletion fence changed after storage cleanup"
                    )
                from .artifact_retirement import retain_cleanup_authority

                retired_prefix = retain_cleanup_authority(locked_job, backend)
                _record_control_plane_custody_event(
                    tenant_id=audit_tenant,
                    document_id=audit_document,
                    event_type="data_deleted",
                    data={
                        "action": "cleanup_completed_jobs",
                        "job_id": str(job.job_id),
                        "retention_days": retention_days,
                        "storage_backend": backend.backend_name,
                        "fenced_generation": fenced_generation,
                        "late_s3_cleanup_scheduled": retired_prefix,
                    },
                    worker_hostname="celery-beat",
                )
                ExtractedFormValue.objects.filter(job=locked_job).delete()
                ExtractedEntity.objects.filter(job=locked_job).delete()
                locked_job.delete()
                deleted += 1
        except Exception as exc:
            job_failure = f"job {job.job_id}: database cleanup failed ({exc})"
            cleanup_failures.append(job_failure)
            try:
                _record_control_plane_custody_event(
                    tenant_id=audit_tenant,
                    document_id=audit_document,
                    event_type="data_deletion_failed",
                    data={
                        "action": "cleanup_completed_jobs",
                        "job_id": str(job.job_id),
                        "failure": job_failure,
                        "phase": "database_delete",
                    },
                    worker_hostname="celery-beat",
                )
            except Exception as custody_exc:
                cleanup_failures.append(
                    f"job {job.job_id}: database-failure custody failed ({custody_exc})"
                )

    logger.info(
        "Cleaned up %d old jobs (%d NFS dirs, %d S3 jobs/%d objects removed)",
        deleted,
        nfs_cleaned,
        s3_jobs_cleaned,
        s3_objects_cleaned,
    )
    return {
        "deleted": deleted,
        "nfs_cleaned": nfs_cleaned,
        "s3_jobs_cleaned": s3_jobs_cleaned,
        "s3_objects_cleaned": s3_objects_cleaned,
        "cleanup_failures": cleanup_failures,
    }


@shared_task(
    name="jobs.tasks.cleanup_expired_presigned_capabilities", queue="coordinator"
)
def cleanup_expired_presigned_capabilities():
    """Delete expired job-bound presigned URL capabilities.

    Capability rows contain encrypted bearer URL material and are not
    evidence records.  They are therefore removed as soon as their expiry
    has passed, independently of the longer job-retention and litigation
    hold policies.  The bounded batch keeps a stale-row backlog from making
    one coordinator task unbounded.
    """
    from .rls import control_plane_rls_context

    with control_plane_rls_context():
        terminal_statuses = [
            Job.Status.COMPLETED,
            Job.Status.FAILED,
            Job.Status.CANCELLED,
        ]
        expired_ids = list(
            PresignedTaskCapability.objects.filter(
                expires_at__lte=timezone.now(),
            )
            .filter(
                db_models.Q(job__status__in=terminal_statuses)
                | ~db_models.Q(
                    dispatch_generation=db_models.F("job__dispatch_generation")
                )
            )
            .order_by("expires_at")
            .values_list("capability_id", flat=True)[:10_000]
        )
        if not expired_ids:
            return {"deleted": 0}

        # Preserve the current generation's capability while its job is
        # active.  A constant-size JIT grant can be renewed safely in
        # open_presigned_capability(); an expired legacy bearer grant remains
        # unusable there but must keep its durable opaque ID until a queued
        # delivery has had a chance to fail closed. Terminal and revoked-
        # generation rows are eligible for prompt deletion.
        deleted, _ = PresignedTaskCapability.objects.filter(
            capability_id__in=expired_ids,
        ).delete()
    logger.info("Deleted %d expired presigned task capabilities", deleted)
    return {"deleted": deleted}


@shared_task(name="jobs.tasks.cleanup_pii_entities", queue="coordinator")
def cleanup_pii_entities():
    from .rls import control_plane_rls_context

    with control_plane_rls_context():
        return _cleanup_pii_entities_control_plane()


def _cleanup_pii_entities_control_plane():
    """Delete PII entity records older than PII_ENTITY_RETENTION_DAYS.

    Runs as a daily Celery Beat task.  Only deletes PII entities belonging
    to completed or failed jobs (active jobs are never touched).

    Respects LITIGATION_HOLD env var — skips all deletions when active.
    Emits a CustodyEvent recording the purge action.
    """
    if is_litigation_hold_active():
        logger.info("LITIGATION_HOLD active — cleanup_pii_entities skipped")
        return {"deleted": 0, "litigation_hold": True}

    try:
        retention_days = int(os.environ.get("PII_ENTITY_RETENTION_DAYS", "90"))
    except (TypeError, ValueError):
        logger.error("Invalid PII_ENTITY_RETENTION_DAYS; refusing automated deletion")
        return {
            "deleted": 0,
            "cleanup_failures": ["invalid PII_ENTITY_RETENTION_DAYS"],
        }
    if retention_days <= 0:
        logger.error(
            "PII_ENTITY_RETENTION_DAYS must be positive; refusing automated deletion"
        )
        return {
            "deleted": 0,
            "cleanup_failures": ["non-positive PII_ENTITY_RETENTION_DAYS"],
        }
    cutoff = timezone.now() - timezone.timedelta(days=retention_days)

    terminal_statuses = [
        Job.Status.COMPLETED,
        Job.Status.FAILED,
        Job.Status.CANCELLED,
    ]
    candidate_ids = set(
        PiiEntity.objects.filter(
            created_at__lt=cutoff,
            job__status__in=terminal_statuses,
        ).values_list("job_id", flat=True)
    )
    extracted_pii_types = {
        "SSN",
        "DOB",
        "EMAIL",
        "PHONE",
        "NAME",
        "ADDRESS",
        "CREDIT_CARD",
        "BANK_ACCOUNT",
        "PASSPORT",
        "DRIVERS_LICENSE",
    }
    candidate_ids.update(
        ExtractedEntity.objects.filter(
            created_at__lt=cutoff,
            entity_type__in=extracted_pii_types,
            job__status__in=terminal_statuses,
        ).values_list("job_id", flat=True)
    )
    candidate_ids.update(
        ExtractedFormValue.objects.filter(
            created_at__lt=cutoff,
            job__status__in=terminal_statuses,
        ).values_list("job_id", flat=True)
    )
    candidate_ids.update(
        DocumentChunk.objects.filter(
            created_at__lt=cutoff,
            contains_pii=True,
            job__status__in=terminal_statuses,
        ).values_list("job_id", flat=True)
    )
    if not candidate_ids:
        return {
            "deleted": 0,
            "extracted_pii_deleted": 0,
            "form_values_deleted": 0,
            "pii_chunks_deleted": 0,
            "retention_days": retention_days,
        }
    from jobs.management.commands.purge_pii import erase_pii_for_job

    totals = {
        "pii_entities": 0,
        "extracted_entities": 0,
        "form_values": 0,
        "pii_chunks": 0,
    }
    failures = []
    for job in Job.objects.filter(job_id__in=candidate_ids).order_by("job_id"):
        result = erase_pii_for_job(
            job,
            purge_scope=f"retention_days={retention_days}",
            source="celery-beat",
        )
        failures.extend(result["failures"])
        for key in totals:
            totals[key] += result["deleted"].get(key, 0)

    return {
        "deleted": totals["pii_entities"],
        "extracted_pii_deleted": totals["extracted_entities"],
        "form_values_deleted": totals["form_values"],
        "pii_chunks_deleted": totals["pii_chunks"],
        "retention_days": retention_days,
        "cleanup_failures": failures,
    }


@shared_task(name="jobs.tasks.cleanup_output_files", queue="coordinator")
def cleanup_output_files():
    """Remove output files for old completed jobs.

    Wraps the ``cleanup_output`` management command as a weekly Celery Beat
    task.  Reads ``OUTPUT_RETENTION_DAYS`` env var (default: 90).

    Respects LITIGATION_HOLD env var — skips when active.
    """
    if is_litigation_hold_active():
        logger.info("LITIGATION_HOLD active — cleanup_output_files skipped")
        return {"status": "skipped", "litigation_hold": True}

    from django.core.management import call_command

    try:
        retention_days = int(os.environ.get("OUTPUT_RETENTION_DAYS", "90"))
    except (TypeError, ValueError):
        logger.error("Invalid OUTPUT_RETENTION_DAYS; refusing automated cleanup")
        return {"status": "error", "message": "invalid OUTPUT_RETENTION_DAYS"}
    if retention_days <= 0:
        logger.error(
            "OUTPUT_RETENTION_DAYS must be positive; refusing automated cleanup"
        )
        return {"status": "error", "message": "non-positive OUTPUT_RETENTION_DAYS"}
    try:
        call_command(
            "cleanup_output", "--confirm", f"--retention-days={retention_days}"
        )
    except Exception:
        logger.exception("cleanup_output command failed")
        return {"status": "error"}
    return {"status": "ok", "retention_days": retention_days}


@shared_task(name="jobs.tasks.rotate_audit_logs_task", queue="coordinator")
def rotate_audit_logs_task():
    """Archive and rotate old CustodyEvent records.

    Wraps the ``rotate_audit_logs`` management command as a monthly Celery
    Beat task.  Reads ``AUDIT_LOG_RETENTION_DAYS`` env var (default: 2555).

    Respects LITIGATION_HOLD env var — skips when active.
    """
    if is_litigation_hold_active():
        logger.info("LITIGATION_HOLD active — rotate_audit_logs_task skipped")
        return {"status": "skipped", "litigation_hold": True}

    from django.core.management import call_command

    try:
        retention_days = int(os.environ.get("AUDIT_LOG_RETENTION_DAYS", "2555"))
    except (TypeError, ValueError):
        logger.error("Invalid AUDIT_LOG_RETENTION_DAYS; refusing automated rotation")
        return {"status": "error", "message": "invalid AUDIT_LOG_RETENTION_DAYS"}
    if retention_days <= 0:
        logger.error(
            "AUDIT_LOG_RETENTION_DAYS must be positive; refusing automated rotation"
        )
        return {
            "status": "error",
            "message": "non-positive AUDIT_LOG_RETENTION_DAYS",
        }
    try:
        call_command(
            "rotate_audit_logs", "--confirm", f"--retention-days={retention_days}"
        )
    except Exception:
        logger.exception("rotate_audit_logs command failed")
        return {"status": "error"}
    return {"status": "ok", "retention_days": retention_days}


# ---------------------------------------------------------------------------
# OCR GPU Queue Tasks
# ---------------------------------------------------------------------------


@shared_task(
    bind=True,
    name="jobs.tasks.process_document",
    queue="ocr_gpu",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
)
@_tenant_bound_task
def process_document(
    self,
    job_id,
    presigned_capability_id=None,
    ocr_engine_lane=None,
    tenant_id=None,
    dispatch_generation=None,
):
    """Reconcile an old broker delivery through the durable page outbox."""
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}
    _pin_rls_for_job(job)

    stale = _reject_stale_dispatch_generation(job, dispatch_generation)
    if stale is not None:
        return stale

    terminal = _reject_if_job_terminal(job)
    if terminal is not None:
        return terminal

    # Old brokers may still deliver a pre-upgrade process_document contract.
    # Reconcile it through the existing durable per-page outbox. Native text
    # and explicit skip_ocr jobs are owned by process_text_only.
    if job.source_type in {"pdf", "image"}:
        return extract_pages.run(
            str(job_id), tenant_id=tenant_id,
            dispatch_generation=dispatch_generation,
        )
    # A malformed or obsolete root task must not leave a job PROCESSING.
    with transaction.atomic():
        locked = Job.objects.select_for_update().get(pk=job.pk)
        if locked.status == Job.Status.PROCESSING and locked.dispatch_generation == job.dispatch_generation:
            locked.status = Job.Status.FAILED
            locked.error_message = "process_document requires a classified PDF or image"
            locked.save(update_fields=["status", "error_message"])
    return {
        "status": "error",
        "job_id": str(job_id),
        "message": "process_document requires a classified PDF or image for page-fenced redirect",
    }


@shared_task(
    bind=True,
    name="jobs.tasks.extract_pages",
    queue="coordinator",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
    default_retry_delay=30,
)
@_tenant_bound_task
def extract_pages(self, job_id, tenant_id=None, dispatch_generation=None):
    """Admit classified OCR pages into durable page-fenced processing.

    Persists one outbox contract per page. The final page reconciles durable
    PageResult state and creates the assembly outbox contract; no chord
    publication window exists between the database claim and the broker.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}
    _pin_rls_for_job(job)

    stale = _reject_stale_dispatch_generation(job, dispatch_generation)
    if stale is not None:
        return stale

    terminal = _reject_if_job_terminal(job)
    if terminal is not None:
        return terminal

    outbox_ids = []
    expected_identity = (
        int(job.dispatch_generation or 0),
        str(job.tenant_id or ""),
        str(job.source_hash or ""),
        str(job.source_object_key or ""),
        str(job.source_file or ""),
        str(job.source_type or ""),
    )
    try:
        separate_capabilities = _separate_tesseract_lane_enabled()
        fair_fanout = _fair_fanout_enabled()
        with transaction.atomic():
            job = Job.objects.select_for_update().get(job_id=job_id)
            current_identity = (
                int(job.dispatch_generation or 0),
                str(job.tenant_id or ""),
                str(job.source_hash or ""),
                str(job.source_object_key or ""),
                str(job.source_file or ""),
                str(job.source_type or ""),
            )
            if current_identity != expected_identity:
                return {"status": "stale", "job_id": str(job_id), "reason": "job identity changed before page admission"}
            stale = _reject_stale_dispatch_generation(job, dispatch_generation)
            if stale is not None:
                return stale
            if job.status != Job.Status.PROCESSING or job.fanout_dispatched:
                return {
                    "status": "skipped",
                    "job_id": str(job_id),
                    "reason": "fanout already dispatched or job state changed",
                }
            try:
                if job.source_type not in {"pdf", "image"}:
                    raise ValueError("OCR page admission requires classified PDF or image")
                _validate_job_page_budget(job)
            except ValueError as exc:
                message = f"Refusing extract_pages expansion: {exc}"
                job.status = Job.Status.FAILED
                job.error_message = message
                job.completed_at = timezone.now()
                job.save(update_fields=["status", "error_message", "completed_at"])
                return {"status": "error", "job_id": str(job_id), "message": message}
            document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
            ocr_engine_lane = (
                _select_initial_ocr_engine_lane(job) if separate_capabilities else None
            )
            ocr_queue = _get_ocr_queue(ocr_engine_lane)
            celery_priority = _get_celery_priority(job)
            _record_requested_engine_denials(
                job,
                document_id,
                socket.gethostname(),
                _engine_policy_from_job_settings(job.settings_json),
            )
            for page_num in range(1, job.total_pages + 1):
                PageResult.objects.get_or_create(
                    job=job,
                    page_num=page_num,
                    defaults={"document_id": document_id, "status": "pending"},
                )
                if fair_fanout:
                    continue
                dispatch_kwargs = {
                    # The relay mints this immediately before broker publish.
                    # Deferred outbox rows therefore cannot outlive a URL
                    # created during fan-out enumeration.
                    "presigned_capability_id": None,
                    **_tenant_for_task(job),
                }
                if ocr_engine_lane:
                    dispatch_kwargs["ocr_engine_lane"] = ocr_engine_lane
                row = enqueue_task_dispatch(
                    job=job,
                    stage=f"process_page:{page_num}",
                    generation=str(job.dispatch_generation),
                    task_name="jobs.tasks.process_page",
                    args=[str(job_id), page_num],
                    kwargs=dispatch_kwargs,
                    queue=ocr_queue,
                    priority=celery_priority,
                )
                outbox_ids.append(row.dispatch_id)
            if fair_fanout:
                outbox_ids.extend(
                    row.dispatch_id for row in _admit_pending_pages_locked(job)
                )
            _record_custody_event(
                job,
                document_id,
                "pages_extracted",
                data={"total_pages": job.total_pages},
                worker_hostname=socket.gethostname(),
            )
            job.fanout_dispatched = True
            job.started_at = timezone.now()
            job.save(update_fields=["fanout_dispatched", "started_at"])

        for dispatch_id in outbox_ids[:FANOUT_IMMEDIATE_PUBLISH_LIMIT]:
            transaction.on_commit(
                partial(publish_task_dispatch_outbox, dispatch_id=dispatch_id)
            )

        return {
            "status": "fanout_dispatched",
            "job_id": str(job_id),
            "page_tasks": len(outbox_ids),
            "pages_pending_admission": job.total_pages - len(outbox_ids),
            "outbox_committed": len(outbox_ids),
            "immediate_publish_scheduled": min(
                len(outbox_ids), FANOUT_IMMEDIATE_PUBLISH_LIMIT
            ),
        }
    except Exception as exc:
        # Transactional creation means an exception cannot leave a partial
        # page set. Broker failures are represented by FAILED/DEAD outbox rows
        # and repaired or surfaced by the relay instead of failing the job.
        logger.exception("Fan-out outbox creation failed for job %s", job_id)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        message = "Fan-out outbox creation failed after retry budget"
        Job.objects.filter(
            pk=job.pk,
            status=Job.Status.PROCESSING,
            fanout_dispatched=False,
        ).update(
            status=Job.Status.FAILED,
            error_message=message,
            completed_at=timezone.now(),
        )
        return {"status": "error", "job_id": str(job_id), "message": message}


def _job_id_from_chord_error(request, traceback=None, job_id=None):
    """Recover the document id across Celery errback calling conventions.

    Celery 5.x has two valid errback paths: newer workers invoke a multi-arg
    errback directly, while older/chord-backend paths enqueue it with only a
    task id or omit the traceback argument.  The previous fixed four-argument
    signature turned either variation into a TypeError, leaving the job in
    ASSEMBLING until the stale-redrive budget was exhausted.
    """
    candidates = [job_id]
    if traceback is not None:
        candidates.append(traceback)
    request_args = getattr(request, "args", None) or ()
    candidates.extend(request_args)
    request_kwargs = getattr(request, "kwargs", None) or {}
    candidates.extend(
        request_kwargs.get(key)
        for key in ("job_id", "id")
        if request_kwargs.get(key) is not None
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return str(uuid.UUID(str(candidate)))
        except (ValueError, TypeError, AttributeError):
            continue
    return None


@shared_task(bind=True, name="jobs.tasks.chord_error_handler", queue="coordinator")
@_tenant_bound_chord_error_task
def chord_error_handler(
    self,
    request=None,
    exc=None,
    traceback=None,
    job_id=None,
    *args,
    tenant_id=None,
    bound_job_id=None,
    bound_generation=None,
    **kwargs,
):
    """Handle chord failures when any process_page task raises an exception.

    Sets the job status to FAILED and records a custody event with the error.
    Without this handler, failed chords leave jobs stuck in 'processing' until
    the cleanup_stale_jobs periodic task catches them after the configured
    processing timeout.
    """
    # Keep this handler tolerant of Celery's direct and queued errback forms.
    # ``args``/``kwargs`` are intentionally accepted for forward compatibility
    # with broker/backend versions that add context fields.
    del self, args, kwargs
    job_id = _job_id_from_chord_error(
        request, traceback, bound_job_id or job_id
    )
    if not job_id:
        logger.error(
            "chord_error_handler: could not recover job id from failed task %s",
            getattr(request, "id", "<unknown>"),
        )
        return {"status": "error", "message": "chord error missing job id"}

    with transaction.atomic():
        try:
            job = Job.objects.select_for_update().get(job_id=job_id)
        except Job.DoesNotExist:
            logger.error("chord_error_handler: Job %s not found", job_id)
            return {"status": "error", "message": f"Job {job_id} not found"}

        if bound_generation is not None and int(bound_generation) != int(
            job.dispatch_generation
        ):
            return {
                "status": "stale",
                "job_id": str(job_id),
                "message": "failed task belongs to an older dispatch generation",
            }
        if tenant_id is not None and str(tenant_id) != str(job.tenant_id or ""):
            return {
                "status": "stale",
                "job_id": str(job_id),
                "message": "failed task tenant binding differs from job",
            }

        if job.status in _TERMINAL_JOB_STATUSES:
            status = str(job.status)
            return {
                "status": status,
                "job_id": str(job_id),
                "message": job.error_message or f"job was already {status}",
            }

        failed_task_id = getattr(request, "id", None) or (
            request if isinstance(request, str) else "unknown"
        )
        error_detail = str(exc) if exc is not None else f"task {failed_task_id} failed"
        error_msg = f"Chord processing failed: {error_detail}"
        updated = Job.objects.filter(
            pk=job.pk,
            status__in=(
                Job.Status.SUBMITTED,
                Job.Status.INGESTING,
                Job.Status.PROCESSING,
                Job.Status.ASSEMBLING,
            ),
        ).update(
            status=Job.Status.FAILED,
            error_message=error_msg,
            completed_at=timezone.now(),
        )
        if updated != 1:
            job.refresh_from_db()
            status = str(job.status)
            return {
                "status": status,
                "job_id": str(job_id),
                "message": job.error_message or f"job became {status}",
            }

        document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
        _record_custody_event(
            job,
            document_id,
            "chord_failed",
            data={"error": error_detail, "task_id": str(failed_task_id)},
            worker_hostname=socket.gethostname(),
        )

    logger.error("Chord failed for job %s: %s", job_id, exc)
    return {"status": "failed", "job_id": str(job_id), "error": error_msg}


_SUCCESSFUL_PAGE_STATUSES = frozenset({"ok", "fallback", "image_only"})


def _reconcile_fanout_page(job_id, page_num, *, enqueue_layoutlm=False):
    """Reconcile durable page state and enqueue its downstream contracts."""
    dispatch_rows = []
    with transaction.atomic():
        job = Job.objects.select_for_update().get(job_id=job_id)
        if job.status != Job.Status.PROCESSING:
            return dispatch_rows

        statuses = list(
            PageResult.objects.filter(job=job).values_list("status", flat=True)
        )
        completed = sum(status in _SUCCESSFUL_PAGE_STATUSES for status in statuses)
        failed = sum(status == "failed" for status in statuses)
        job.pages_completed = completed

        if failed:
            job.status = Job.Status.FAILED
            job.error_message = f"Fan-out processing failed for {failed} page(s)"
            job.completed_at = timezone.now()
            job.save(
                update_fields=[
                    "pages_completed",
                    "status",
                    "error_message",
                    "completed_at",
                ]
            )
            document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
            if not CustodyEvent.objects.filter(
                job=job,
                event_type="fanout_reconciled_failed",
            ).exists():
                _record_custody_event(
                    job,
                    document_id,
                    "fanout_reconciled_failed",
                    data={"failed_pages": failed, "completed_pages": completed},
                    worker_hostname=socket.gethostname(),
                )
            return dispatch_rows

        job.save(update_fields=["pages_completed"])
        if enqueue_layoutlm:
            dispatch_rows.append(
                enqueue_task_dispatch(
                    job=job,
                    stage=f"layoutlm_page:{page_num}",
                    generation=str(job.dispatch_generation),
                    task_name="jobs.tasks_layoutlm.run_layoutlm_extraction",
                    args=[str(job_id), int(page_num)],
                    kwargs=_tenant_for_task(job),
                    queue="ocr_layoutlm",
                    priority=_get_celery_priority(job),
                )
            )

        if job.total_pages > 0 and completed == job.total_pages:
            dispatch_rows.append(
                enqueue_task_dispatch(
                    job=job,
                    stage="assemble_document",
                    generation=str(job.dispatch_generation),
                    task_name="jobs.tasks.assemble_document",
                    args=[str(job_id)],
                    kwargs=_tenant_for_task(job),
                    queue="coordinator",
                    priority=_get_celery_priority(job),
                )
            )
        elif _fair_fanout_enabled():
            dispatch_rows.extend(_admit_pending_pages_locked(job))
    return dispatch_rows


def _publish_dispatch_rows(rows):
    """Best-effort immediate relay; durable rows remain for Beat on failure."""
    for row in rows:
        transaction.on_commit(
            partial(publish_task_dispatch_outbox, dispatch_id=row.dispatch_id)
        )


def _stage_raster_locator(job, page_num, generation, backend, token):
    relative = f"stages/{generation}/{page_num}/raster-{token}.png"
    if backend.backend_name == "nfs":
        return os.path.join(_nfs_job_path_for_job(job), relative)
    return _job_storage_key(job.job_id, relative)


def _stage_publish_safe_after(backend):
    """Set a conservative reservation hold for an uncertain S3 PUT.

    The renderer has a one-hour hard task limit. Signing lifetime and a
    transfer estimate extend the hold, but they cannot prove that a remote
    server will never commit later. The durable exact-key retirement sweep
    continues after this hold and must be qualified against the real backend.
    """
    if backend.backend_name == "nfs":
        return None
    max_page_bytes, _ = _stage_limits()
    transfer_seconds = ((max_page_bytes + 1024 * 1024 - 1) // (1024 * 1024)) * 5
    signing_seconds = int(getattr(settings, "S3_PRESIGNED_URL_EXPIRY", 300))
    return timezone.now() + timedelta(
        seconds=max(3600, signing_seconds + transfer_seconds + 120)
    )


def _stage_fence(stage, job):
    if (
        job.status != Job.Status.PROCESSING
        or stage.job_id != job.pk
        or stage.generation != job.dispatch_generation
        or stage.tenant_id != job.tenant_id
        or stage.source_sha256 != job.source_hash
    ):
        raise _ProcessingLeaseLost("staged page tenant/source/generation fence failed")


def _stage_limits():
    per_page = _bounded_operational_int(
        "OCR_STAGE_MAX_PAGE_BYTES", 256 * 1024 * 1024,
        minimum=1024 * 1024, maximum=512 * 1024 * 1024,
    )
    global_limit = _bounded_operational_int(
        "OCR_STAGE_GLOBAL_BYTES", 16 * 1024 * 1024 * 1024,
        minimum=per_page, maximum=1024 * 1024 * 1024 * 1024,
    )
    return per_page, global_limit


def _stage_lease_seconds():
    """Bounded worker-loss recovery clock, with production default 30 minutes."""
    return _bounded_operational_int(
        "OCR_STAGE_LEASE_SECONDS", 1800, minimum=20, maximum=3600
    )


def _stage_reserve(stage_id, job_id, backend=None):
    """Reserve worst-case raster bytes under one fleet-wide DB row lock."""
    per_page, global_limit = _stage_limits()
    with transaction.atomic():
        budget = PageOCRByteBudget.objects.select_for_update().get(pk=1)
        stage = PageOCRStage.objects.select_for_update().get(pk=stage_id)
        job = Job.objects.select_for_update().get(pk=job_id)
        _stage_fence(stage, job)
        if stage.status not in {"pending", "rendering"}:
            return stage.status
        if stage.status == "rendering":
            return "busy"
        additional = max(0, per_page - stage.reserved_bytes)
        if budget.reserved_bytes + additional > global_limit:
            return "saturated"
        token = uuid.uuid4().hex
        stage.status = "rendering"
        stage.lease_token = token
        stage.lease_expires_at = timezone.now() + timedelta(seconds=_stage_lease_seconds())
        if backend is None:
            backend = _get_backend_for_job(job)
        stage.raster_locator = _stage_raster_locator(
            job, stage.page_num, stage.generation, backend, token
        )
        stage.reserved_bytes = per_page
        stage.attempts += 1
        stage.render_attempts += 1
        stage.save(update_fields=["status", "lease_token", "lease_expires_at", "raster_locator", "reserved_bytes", "attempts", "render_attempts", "updated_at"])
        budget.reserved_bytes += additional
        budget.save(update_fields=["reserved_bytes"])
        return token


def _stage_release_locked(stage):
    if not stage.reserved_bytes:
        return
    budget = PageOCRByteBudget.objects.select_for_update().get(pk=1)
    if budget.reserved_bytes < stage.reserved_bytes:
        raise RuntimeError("staged raster budget underflow")
    budget.reserved_bytes -= stage.reserved_bytes
    budget.save(update_fields=["reserved_bytes"])
    stage.reserved_bytes = 0


def _stage_next_dispatch(stage, operation, queue, *, retry_index=None, ocr_engine_lane=None):
    job = stage.job
    stage_key = f"{operation}:{stage.page_num}"
    if retry_index is not None:
        stage_key += f":retry:{retry_index}"
    dispatch_kwargs = _tenant_for_task(job)
    if ocr_engine_lane is not None:
        dispatch_kwargs["ocr_engine_lane"] = ocr_engine_lane
    row = enqueue_task_dispatch(
        job=job,
        stage=stage_key,
        generation=str(stage.generation),
        task_name=f"jobs.tasks.{operation}",
        args=[str(job.job_id), stage.page_num, stage.generation],
        kwargs=dispatch_kwargs,
        queue=queue,
        priority=_get_celery_priority(job),
    )
    transaction.on_commit(
        partial(publish_task_dispatch_outbox, dispatch_id=row.dispatch_id)
    )


def _stage_fail_uncommitted_page_locked(job, stage, reason, now):
    """Fail only this generation's uncommitted page under the parent lock.

    Recovery already holds budget, stage, then Job. A finalizer can have
    committed its PageResult before dying during stage cleanup; that terminal
    winner must remain intact. A live finalizer lease likewise gets its
    bounded chance to commit before recovery marks the page failed.
    """
    page = PageResult.objects.select_for_update().filter(
        job=job, page_num=stage.page_num,
    ).first()
    if page is None or page.admitted_generation != stage.generation:
        return "obsolete"
    if page.status in _SUCCESSFUL_PAGE_STATUSES or page.status == "failed":
        return "terminal"
    if (
        page.status == "processing"
        and page.lease_expires_at is not None
        and page.lease_expires_at > now
    ):
        return "live_owner"
    if page.status not in {"pending", "processing"}:
        return "obsolete"
    page.status = "failed"
    page.needs_review = True
    page.lease_expires_at = None
    page.ocr_provenance = {
        "assessment_version": "ocr_rescue_v1",
        "assessment_state": "failed",
        "failure_reason": str(reason)[:256],
    }
    page.save(update_fields=[
        "status", "needs_review", "lease_expires_at", "ocr_provenance",
    ])
    _record_custody_event(
        job,
        job.source_hash[:16] if job.source_hash else str(job.pk)[:16],
        "processing_failed",
        data={"page_num": stage.page_num, "error": str(reason)[:256]},
        worker_hostname=socket.gethostname(),
    )
    return "failed"


def _recover_staged_pages(*, limit=100):
    """Re-drive lost stage workers and reclaim reservations after cancellation."""
    if not _staged_page_ocr_enabled():
        return 0
    from .rls import control_plane_rls_context

    now = timezone.now()
    stale_before = now - timedelta(seconds=_stage_lease_seconds())
    pending_before = now - timedelta(seconds=30)
    recovered = 0
    cleanup_ids = []
    recoverable_statuses = ("pending", "rendering", "rendered", "infering", "inferred")
    if not is_litigation_hold_active():
        recoverable_statuses += ("cleanup",)
    with control_plane_rls_context():
        candidates = list(
            PageOCRStage.objects.filter(
                status__in=recoverable_statuses,
            ).filter(
                Q(status="cleanup")
                | Q(status="pending", updated_at__lt=pending_before)
                | Q(updated_at__lt=stale_before)
                | Q(job__status__in=_TERMINAL_JOB_STATUSES)
                | ~Q(generation=F("job__dispatch_generation"))
                | ~Q(source_sha256=F("job__source_hash"))
                | ~Q(tenant_id=F("job__tenant_id"))
            ).order_by("updated_at").values_list("pk", flat=True)[:limit]
        )
        for stage_id in candidates:
            with transaction.atomic():
                PageOCRByteBudget.objects.select_for_update().get(pk=1)
                stage = PageOCRStage.objects.select_for_update().get(pk=stage_id)
                job = Job.objects.select_for_update().get(pk=stage.job_id)
                if stage.status == "cleanup":
                    cleanup_ids.append(stage.pk)
                    continue
                if stage.status not in {"pending", "rendering", "rendered", "infering", "inferred"}:
                    continue
                if (
                    job.status != Job.Status.PROCESSING
                    or stage.generation != job.dispatch_generation
                    or stage.source_sha256 != job.source_hash
                    or stage.tenant_id != job.tenant_id
                ):
                    stage.status = "cleanup"
                    stage.last_error = "stale_job_state_or_generation"
                    stage.save(update_fields=["status", "last_error", "updated_at"])
                    cleanup_ids.append(stage.pk)
                    recovered += 1
                    continue
                if stage.status in {"rendering", "infering"} and stage.lease_expires_at and stage.lease_expires_at > now:
                    continue
                capacity_timeout = _bounded_operational_int(
                    "OCR_STAGE_MAX_CAPACITY_WAIT_SECONDS", 3600,
                    minimum=60, maximum=86400,
                )
                budget_exhausted = (
                    stage.status == "pending"
                    and not stage.render_attempts
                    and (now - stage.created_at).total_seconds() >= capacity_timeout
                )
                retry_exhausted = (
                    (stage.status in {"pending", "rendering"} and stage.render_attempts >= 4)
                    or (stage.status in {"rendered", "infering"} and stage.infer_attempts >= 4)
                    or stage.redrive_count >= 100
                )
                if (
                    retry_exhausted
                    and stage.last_error == "cross_engine_pending"
                    and stage.inference
                ):
                    retained = dict(stage.inference)
                    route = dict(retained.get("route") or {})
                    route["needs_review"] = True
                    retained["route"] = route
                    evidence = dict(retained.get("rerun_evidence") or {})
                    evidence["cross_engine_unavailable"] = "retry_budget_exhausted"
                    retained["rerun_evidence"] = evidence
                    stage.inference = retained
                    stage.status = "inferred"
                    stage.last_error = "cross_engine_unavailable"
                    stage.lease_token = ""
                    stage.lease_expires_at = None
                    stage.save(update_fields=["inference", "status", "last_error", "lease_token", "lease_expires_at", "updated_at"])
                    stage.redrive_count += 1
                    stage.save(update_fields=["redrive_count", "updated_at"])
                    _stage_next_dispatch(
                        stage, "finalize_page", "ocr_finalize",
                        retry_index=f"recovery:{stage.redrive_count}",
                    )
                    recovered += 1
                    continue
                if budget_exhausted or retry_exhausted:
                    failure_reason = (
                        "staged page capacity wait exceeded"
                        if budget_exhausted else "staged page recovery budget exhausted"
                    )
                    page_disposition = _stage_fail_uncommitted_page_locked(
                        job, stage, failure_reason, now,
                    )
                    if page_disposition == "live_owner":
                        continue
                    stage.status = "cleanup"
                    stage.last_error = (
                        failure_reason if page_disposition == "failed"
                        else f"stage_cleanup:{page_disposition}"
                    )
                    stage.save(update_fields=["status", "last_error", "updated_at"])
                    cleanup_ids.append(stage.pk)
                    if page_disposition == "failed":
                        _reconcile_fanout_page(job.pk, stage.page_num)
                    recovered += 1
                    continue
                if stage.status == "rendering":
                    stage.status = "cleanup"
                    stage.lease_token = ""
                    stage.lease_expires_at = None
                    stage.last_error = "retry_render:render_lease_expired"
                    stage.save(update_fields=["status", "lease_token", "lease_expires_at", "last_error", "updated_at"])
                    cleanup_ids.append(stage.pk)
                    recovered += 1
                    continue
                elif stage.status == "infering":
                    stage.status = "rendered"
                    stage.lease_token = ""
                    stage.lease_expires_at = None
                    stage.save(update_fields=["status", "lease_token", "lease_expires_at", "updated_at"])
                operation = {
                    "pending": "render_page",
                    "rendered": "infer_page",
                    "inferred": "finalize_page",
                }[stage.status]
                queue = {
                    "render_page": "ocr_render",
                    "infer_page": _get_ocr_queue(),
                    "finalize_page": "ocr_finalize",
                }[operation]
                fallback_lane = (
                    operation == "infer_page"
                    and stage.last_error == "cross_engine_pending"
                )
                if fallback_lane:
                    queue = "ocr_tesseract"
                stage.redrive_count += 1
                stage.save(update_fields=["redrive_count", "updated_at"])
                _stage_next_dispatch(
                    stage, operation, queue,
                    retry_index=f"recovery:{stage.redrive_count}",
                    ocr_engine_lane="tesseract" if fallback_lane else None,
                )
                recovered += 1
        for stage_id in cleanup_ids:
            try:
                stage = PageOCRStage.objects.get(pk=stage_id)
                _stage_cleanup_artifact(stage_id, _get_backend_for_job(stage.job))
            except Exception:
                logger.exception("Staged artifact cleanup failed for stage %s", stage_id)
        with transaction.atomic():
            budget = PageOCRByteBudget.objects.select_for_update().get(pk=1)
            actual = sum(
                PageOCRStage.objects.values_list("reserved_bytes", flat=True)
            ) + sum(
                PageOCRRasterRetirement.objects.values_list("reserved_bytes", flat=True)
            )
            if budget.reserved_bytes != actual:
                budget.reserved_bytes = actual
                budget.save(update_fields=["reserved_bytes"])
    return recovered


@contextmanager
def _stage_source_lease(job, page_num, backend, stage, token, presigned_urls=None, capability_id=None):
    def authorize_source():
        _check_stage_render_owner(stage.pk, job.pk, token)

    authorize_source()
    if backend.backend_name == "nfs":
        path = os.path.join(
            _nfs_job_path_for_job(job), "source", os.path.basename(job.source_file)
        )
        if not hmac.compare_digest(_compute_file_hash(path), job.source_hash):
            raise RuntimeError("staged source hash changed before render")
        yield path
        authorize_source()
        return
    if not isinstance(backend, CachedS3Backend):
        raise RuntimeError("staged S3 source requires immutable source cache")
    key = _job_storage_key(job.job_id, f"source/{os.path.basename(job.source_file)}")
    if presigned_urls is not None:
        revision = presigned_urls["source_revision_token"]
        size = int(presigned_urls["source_size_bytes"])
        original_authorize = authorize_source

        def authorize_source():
            original_authorize()
            validate_stage_capability(
                job, stage, capability_id, "render_page", lease_token=token
            )
    else:
        revision, size = backend.source_revision_info(key)
    identity = SourceIdentity(
        str(job.tenant_id or f"local:{job.pk}"), str(job.pk),
        int(job.dispatch_generation), key, job.source_hash, revision, size,
    )
    if presigned_urls is not None:
        source_lease = lease_presigned_source(
            backend._source_cache, identity, presigned_urls["source_get"],
            authorize=authorize_source,
        )
    else:
        source_lease = backend.lease_source(identity, authorize=authorize_source)
    with source_lease as path:
        yield path
        authorize_source()


def _check_stage_render_owner(stage_id, job_id, token):
    current = Job.objects.get(pk=job_id)
    current_stage = PageOCRStage.objects.get(pk=stage_id)
    _stage_fence(current_stage, current)
    if (
        current_stage.status != "rendering"
        or current_stage.lease_token != token
        or current_stage.lease_expires_at <= timezone.now()
    ):
        raise _ProcessingLeaseLost("staged source lease no longer owns render")


def _stage_publish_raster(stage, job, backend, payload, token, presigned_urls=None):
    locator = stage.raster_locator
    if locator != _stage_raster_locator(job, stage.page_num, stage.generation, backend, token):
        raise RuntimeError("staged raster locator does not match owner token")
    if backend.backend_name == "nfs":
        _mkdir_tree_no_follow(os.path.dirname(locator), settings.NFS_ROOT)
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=os.path.dirname(locator), suffix=".png", delete=False
            ) as temporary:
                temporary_path = temporary.name
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, locator)
            temporary_path = ""
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".png") as temporary:
            temporary.write(payload)
            temporary.flush()
            if presigned_urls is not None:
                from .presigned_io import upload_presigned

                upload_presigned(
                    temporary.name, presigned_urls["raster_put"],
                    job_id=str(job.pk), expected_key=locator,
                )
            else:
                backend.upload_file(temporary.name, locator)
    return locator


def _stage_preflight_raster(source_path, page_num, source_type, reserved_bytes, dpi=300):
    """Reject oversized decoded pages before allocating the render raster."""
    if source_type == "pdf":
        import fitz

        with fitz.open(source_path) as document:
            page = document[page_num - 1]
            width = int(page.rect.width * dpi / 72 + 1)
            height = int(page.rect.height * dpi / 72 + 1)
    elif source_type == "image":
        from PIL import Image

        with Image.open(source_path) as source:
            width, height = source.size
    else:
        raise RuntimeError("staged OCR supports PDF and image sources")
    if width <= 0 or height <= 0 or width * height * 4 > reserved_bytes:
        raise RuntimeError("rendered raster exceeds reserved decoded-byte budget")


def _stage_read_raster(stage, backend, presigned_urls=None):
    if backend.backend_name == "nfs":
        with open(stage.raster_locator, "rb") as handle:
            payload = handle.read(stage.raster_size_bytes + 1)
    else:
        with tempfile.NamedTemporaryFile(suffix=".png") as temporary:
            if presigned_urls is not None:
                from .presigned_io import download_presigned

                download_presigned(
                    presigned_urls["raster_get"], temporary.name,
                    job_id=str(stage.job_id), expected_key=stage.raster_locator,
                )
            else:
                backend.download_file(stage.raster_locator, temporary.name)
            with open(temporary.name, "rb") as handle:
                payload = handle.read(stage.raster_size_bytes + 1)
    if len(payload) != stage.raster_size_bytes or not hmac.compare_digest(
        hashlib.sha256(payload).hexdigest(), stage.raster_sha256
    ):
        raise RuntimeError("staged raster bytes changed")
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as source:
        source.load()
        image = source.convert("RGB")
    if image.size != (stage.raster_width, stage.raster_height):
        raise RuntimeError("staged raster dimensions changed")
    image.info["_ocr_page_geometry"] = PageGeometry(**stage.geometry)
    return image


def _stage_inference_to_json(inferred):
    result = dict(inferred)
    result["route"] = asdict(result["route"])
    result["coverage_assessment"] = asdict(result["coverage_assessment"])
    result["engine_policy"] = None
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > 4 * 1024 * 1024:
        raise RuntimeError("staged OCR result exceeds durable metadata budget")
    return json.loads(encoded)


def _stage_inference_from_json(payload, job):
    result = dict(payload)
    result["route"] = LanguageRoute(**result["route"])
    result["coverage_assessment"] = CoverageAssessment(
        **result["coverage_assessment"]
    )
    result["engine_policy"] = _engine_policy_from_job_settings(job.settings_json)
    return result


def _compare_inference_candidates(prior, candidate):
    """Compare page-point OCR candidates with independent region/language evidence."""
    prior_lang, prior_support = _page_text_language_evidence(prior["page_text"])
    new_lang, new_support = _page_text_language_evidence(candidate["page_text"])
    first = OCRCandidate(
        prior["page_text"], prior["ocr_method"], prior["confidence"],
        prior["coverage_assessment"].coverage, True,
        prior_support if prior_lang == prior["model_language"] else 0.0,
    )
    second = OCRCandidate(
        candidate["page_text"], candidate["ocr_method"], candidate["confidence"],
        candidate["coverage_assessment"].coverage,
        (
            True if candidate["ocr_method"] == "PaddleOCR"
            else text_supports_script(
                candidate["page_text"],
                LANGUAGE_REGISTRY[candidate["model_language"]].script,
            ) if candidate["model_language"] in LANGUAGE_REGISTRY else False
        ),
        new_support if new_lang == candidate["model_language"] else 0.0,
    )
    comparator = (
        compare_ocr_candidates if first.engine == second.engine
        else compare_cross_engine_candidates
    )
    decision = comparator(first, second)
    if (
        not decision.use_candidate
        and first.engine == second.engine == "Tesseract"
        and prior.get("actual_runtime_model") == candidate.get("actual_runtime_model")
    ):
        grounded = compare_grounded_word_candidates(
            first, second,
            prior_lines=prior["ocr_lines"],
            candidate_lines=candidate["ocr_lines"],
        )
        if grounded.use_candidate:
            return grounded
    return decision


def _verify_paddle_text_revision(inferred, image, transform):
    """Read source once on CPU and accept only a grounded text revision.

    This runs in the CPU finalizer for staged pages, or in a combined legacy
    worker. The Paddle-only inference worker never imports Tesseract. The
    alternate text is retained in bounded candidate evidence regardless of
    selection, and a selected reading names Tesseract as its actual engine.
    """
    if inferred["ocr_method"] != "PaddleOCR" or inferred["overlay_issues"]:
        return inferred
    prior_audit = inferred.get("rerun_evidence")
    if isinstance(prior_audit, dict) and prior_audit.get("cross_engine"):
        # A weak Paddle candidate has already had one source Tesseract read.
        return inferred
    language = inferred["model_language"]
    entry = LANGUAGE_REGISTRY.get(language)
    if not entry or not entry.tesseract_code:
        return inferred
    if not evaluate_engine(catalog.TESSERACT, inferred["engine_policy"]).admitted:
        return inferred
    prior_text = str(inferred["page_text"])
    if not prior_text.strip():
        return inferred
    if len(prior_text) > 128_000 or len(inferred["ocr_lines"]) > 2048:
        result = dict(inferred)
        result["route"] = replace(inferred["route"], needs_review=True)
        result["candidate_disposition"] = "source_verification_evidence_over_budget"
        return result
    rotation = int(inferred.get("reading_rotation") or 0)
    if rotation not in (0, 90, 180, 270):
        return inferred
    result = dict(inferred)
    width, height = image.size
    sampled_width, sampled_height = bounded_source_raster_size(image.size)
    geometry = image.info.get("_ocr_page_geometry")
    raster_audit = {
        "background_pixels": [width, height],
        "verification_pixels": [sampled_width, sampled_height],
        "max_verification_pixels": 2_000_000,
        "timeout_seconds": 5.0,
        "tesseract_psm": 3,
        "ocr_rotation_degrees": rotation,
    }
    if isinstance(geometry, PageGeometry):
        raster_audit["background_dpi"] = [
            round(72 * width / geometry.width_pt, 3),
            round(72 * height / geometry.height_pt, 3),
        ]
        raster_audit["verification_dpi"] = [
            round(72 * sampled_width / geometry.width_pt, 3),
            round(72 * sampled_height / geometry.height_pt, 3),
        ]
    try:
        pixel_lines, issues = read_source_tesseract_words(
            image, traineddata=entry.tesseract_code, rotation=rotation,
            timeout_seconds=5.0, max_pixels=2_000_000,
        )
        if len(pixel_lines) > 2048:
            raise RuntimeError("source verification word budget exceeded")
        candidate_text = " ".join(text for text, _, _ in pixel_lines)
        if len(candidate_text) > 128_000:
            raise RuntimeError("source verification text budget exceeded")
        pdf_lines, mapping_issues = transform.map_lines_with_review(pixel_lines)
        raster_evidence = assess_raster_text_regions(image)
        coverage = assess_candidate_coverage(
            raster_evidence, pixel_lines, candidate_raster_size=image.size,
        )
        decision = compare_source_grounded_text_revision(
            inferred["ocr_lines"], pdf_lines,
            prior_coverage={
                "status": inferred["coverage_assessment"].status,
                "coverage": inferred["coverage_assessment"].coverage,
            },
            candidate_coverage={
                "status": coverage.status, "coverage": coverage.coverage,
            },
            script_supported=text_supports_script(candidate_text, entry.script)
            and not issues and not mapping_issues,
        )

        def audit_lines(lines):
            rows = []
            for text, box, score in lines:
                if len(box) != 4:
                    raise ValueError("source verification word box is malformed")
                coordinates = (
                    [[float(value) for value in point] for point in box]
                    if hasattr(box[0], "__iter__")
                    else [float(value) for value in box]
                )
                rows.append({
                    "text": str(text), "box": coordinates,
                    "confidence": float(score),
                })
            return rows

        audit = {
            "prior_engine": "PaddleOCR", "prior_text": prior_text,
            "prior_word_lines": audit_lines(inferred["ocr_lines"]),
            "candidate_engine": "Tesseract", "candidate_text": candidate_text,
            "candidate_word_lines": audit_lines(pdf_lines),
            "candidate_runtime_model": entry.tesseract_code,
            "candidate_confidence_basis": "tesseract_word_confidence_percent_scaled",
            "candidate_coverage": coverage.coverage,
            "issue_count": len(issues) + len(mapping_issues),
            "selected": decision.use_candidate, "reason": decision.reason,
            "selected_engine": "Tesseract" if decision.use_candidate else "PaddleOCR",
            "raster": raster_audit,
        }
        existing_audit = result.get("rerun_evidence")
        candidate_audit = {
            **(existing_audit if isinstance(existing_audit, dict) else {}),
            "source_text_verification": audit,
        }
        if len(json.dumps(candidate_audit, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 8 * 1024 * 1024 - 16 * 1024:
            result["route"] = replace(inferred["route"], needs_review=True)
            result["candidate_disposition"] = "source_verification_evidence_over_budget"
            return result
        if decision.use_candidate:
            scores = [score for _, _, score in pdf_lines]
            result.update(
                ocr_lines=pdf_lines, page_text=candidate_text,
                ocr_method="Tesseract", actual_runtime_model=entry.tesseract_code,
                attempted_tesseract_model=entry.tesseract_code,
                confidence=sum(scores) / len(scores),
                coverage_assessment=coverage,
                candidate_disposition=decision.reason,
                route=replace(inferred["route"], needs_review=True),
            )
        elif decision.reason != "source_revision_no_text_change":
            result["route"] = replace(inferred["route"], needs_review=True)
            result["candidate_disposition"] = decision.reason
    except Exception as exc:
        logger.warning("Bounded source text verification failed: %s", type(exc).__name__)
        result["route"] = replace(inferred["route"], needs_review=True)
        result["candidate_disposition"] = "source_verification_unavailable"
        return result
    result["rerun_evidence"] = candidate_audit
    return result


def _needs_source_dpi_retry(inferred):
    """Retry source PDF when text regions are missing or Tesseract is unsure."""
    weak_word = any(
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and 0 <= score < 0.70
        for _text, _box, score in inferred["ocr_lines"]
    ) if inferred["ocr_method"] == "Tesseract" else False
    return bool(
        inferred["coverage_assessment"].needs_retry
        or (
            inferred["ocr_method"] == "Tesseract"
            and (inferred["confidence"] < 0.85 or weak_word)
        )
    )


@shared_task(bind=True, name="jobs.tasks.render_page", queue="ocr_render", acks_late=True, reject_on_worker_lost=True, max_retries=3, time_limit=3600)
@_tenant_bound_task
def render_page(self, job_id, page_num, generation, tenant_id=None, dispatch_generation=None, presigned_capability_id=None):
    """CPU render after reserving fleet bytes; dispatch inference by IDs only."""
    del tenant_id
    job = Job.objects.get(pk=job_id)
    _pin_rls_for_job(job)
    stage = PageOCRStage.objects.get(job=job, page_num=page_num, generation=generation)
    _stage_fence(stage, job)
    backend = _get_backend_for_job(job)
    token = _stage_reserve(stage.pk, job.pk, backend)
    if token in {"saturated", "busy"}:
        return {"status": token, "page_num": page_num}
    if token != "rendering" and len(token) != 32:
        return {"status": token}
    stage.refresh_from_db()
    locator = stage.raster_locator
    publish_confirmed = False
    try:
        from ocr_distributed.ocr_utils import iter_source_pages

        source_urls = None
        if backend.backend_name == "s3" and is_presigned_mode():
            if not presigned_capability_id:
                raise RuntimeError("staged render requires a presigned capability")
            source_urls = _request_fresh_presigned_urls(
                job, presigned_capability_id, "render_page", "input",
                page_num=page_num, lease_token=token,
            )
        with _stage_source_lease(
            job, page_num, backend, stage, token,
            presigned_urls=source_urls,
            capability_id=presigned_capability_id,
        ) as source_path:
            _stage_preflight_raster(
                source_path, page_num, job.source_type, stage.reserved_bytes,
                dpi=stage.retry_dpi,
            )
            image, geometry = next(
                iter_source_pages(
                    source_path, page_num, page_num, job.source_type,
                    dpi=stage.retry_dpi,
                )
            )
            if image.width * image.height * 4 > stage.reserved_bytes:
                raise RuntimeError("rendered raster exceeds reserved decoded-byte budget")
            script_evidence = _page_script_evidence(image, job)
            reading_route = route_language(
                job.detected_language,
                explicit_hint=(job.settings_json or {}).get("language_hint"),
                script_evidence=script_evidence,
            )
            reading_direction = _page_reading_direction(
                image, geometry, reading_route
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
        if len(payload) > stage.reserved_bytes:
            raise RuntimeError("rendered raster exceeds reserved page bytes")
        _check_stage_render_owner(stage.pk, job.pk, token)
        output_urls = None
        if source_urls is not None:
            output_urls = _request_fresh_presigned_urls(
                job, presigned_capability_id, "render_page", "output",
                page_num=page_num, lease_token=token,
            )
        publish_safe_after = _stage_publish_safe_after(backend)
        if publish_safe_after is not None:
            updated = PageOCRStage.objects.filter(
                pk=stage.pk, status="rendering", lease_token=token,
                lease_expires_at__gt=timezone.now(),
            ).update(publish_safe_after=publish_safe_after)
            if updated != 1:
                raise _ProcessingLeaseLost("render owner changed before raster PUT")
        locator = _stage_publish_raster(
            stage, job, backend, payload, token, presigned_urls=output_urls
        )
        publish_confirmed = True
        with transaction.atomic():
            locked = PageOCRStage.objects.select_for_update().get(pk=stage.pk)
            current = Job.objects.select_for_update().get(pk=job.pk)
            _stage_fence(locked, current)
            if (
                locked.status != "rendering"
                or locked.lease_token != token
                or locked.lease_expires_at <= timezone.now()
            ):
                raise _ProcessingLeaseLost("render ownership changed")
            locked.status = "rendered"
            locked.raster_locator = locator
            locked.raster_sha256 = hashlib.sha256(payload).hexdigest()
            locked.raster_size_bytes = len(payload)
            locked.raster_width, locked.raster_height = image.size
            locked.raster_dpi = stage.retry_dpi
            locked.geometry = asdict(geometry)
            locked.script_evidence = list(script_evidence[:3])
            locked.reading_direction = reading_direction
            locked.publish_safe_after = None
            locked.last_error = ""
            locked.lease_token = ""
            locked.lease_expires_at = None
            locked.save()
            _stage_next_dispatch(
                locked, "infer_page", _get_ocr_queue(),
                retry_index=(
                    f"render:{locked.render_attempts}:{locked.raster_sha256[:12]}"
                ),
            )
        return {"status": "rendered", "page_num": page_num}
    except Exception as exc:
        with transaction.atomic():
            PageOCRByteBudget.objects.select_for_update().get(pk=1)
            locked = PageOCRStage.objects.select_for_update().get(pk=stage.pk)
            if locked.status == "rendering" and locked.lease_token == token:
                locked.status = "cleanup"
                locked.lease_token = ""
                locked.lease_expires_at = None
                locked.last_error = f"retry_render:{str(exc)[:1000]}"
                locked.save()
            if (
                publish_confirmed and locator
                and locked.status == "cleanup"
                and locked.raster_locator == locator
            ):
                # This worker received the PUT completion, so it can shorten
                # the crash-only quarantine and let coordinator cleanup run.
                locked.publish_safe_after = None
                locked.save(update_fields=["publish_safe_after", "updated_at"])
        if backend.backend_name == "s3" and is_presigned_mode():
            try:
                cleanup_page_stage.apply_async(args=[stage.pk], queue="coordinator")
            except Exception:
                logger.exception("Could not dispatch failed render cleanup %s", stage.pk)
        else:
            _stage_cleanup_artifact(stage.pk, backend)
        raise


@shared_task(bind=True, name="jobs.tasks.infer_page", queue="ocr_gpu", acks_late=True, reject_on_worker_lost=True, max_retries=3)
@_tenant_bound_task
def infer_page(self, job_id, page_num, generation, tenant_id=None, dispatch_generation=None, presigned_capability_id=None, ocr_engine_lane=None):
    """GPU-ready inference of one immutable raster, with no page PDF work."""
    del self, tenant_id
    job = Job.objects.get(pk=job_id)
    _pin_rls_for_job(job)
    with transaction.atomic():
        locked = PageOCRStage.objects.select_for_update().get(
            job=job, page_num=page_num, generation=generation
        )
        current = Job.objects.select_for_update().get(pk=job.pk)
        _stage_fence(locked, current)
        if locked.status != "rendered":
            return {"status": locked.status}
        if locked.last_error == "cross_engine_pending":
            # An at-least-once delivery of the original Paddle inference may
            # arrive after its durable CPU fallback handoff. Only a worker
            # that actually resolves to Tesseract may claim this stage, even
            # when the handoff has no retained first inference yet.
            try:
                resolved_lane = _resolve_ocr_engine_lane(ocr_engine_lane)
            except (RuntimeError, ValueError):
                resolved_lane = None
            if resolved_lane != "tesseract":
                return {"status": "lane_mismatch", "expected_lane": "tesseract"}
        token = uuid.uuid4().hex
        locked.status = "infering"
        locked.lease_token = token
        locked.lease_expires_at = timezone.now() + timedelta(seconds=_stage_lease_seconds())
        locked.attempts += 1
        locked.infer_attempts += 1
        locked.save(update_fields=["status", "lease_token", "lease_expires_at", "attempts", "infer_attempts", "updated_at"])
        stage_id = locked.pk
        raster_sha256 = locked.raster_sha256
    try:
        stage = PageOCRStage.objects.get(pk=stage_id)
        backend = _get_backend_for_job(job)
        raster_urls = None
        if backend.backend_name == "s3" and is_presigned_mode():
            if not presigned_capability_id:
                raise RuntimeError("staged inference requires a presigned capability")
            raster_urls = _request_fresh_presigned_urls(
                job, presigned_capability_id, "infer_page", "input",
                page_num=page_num, lease_token=token,
                raster_sha256=raster_sha256,
            )
        image = _stage_read_raster(stage, backend, raster_urls)
        geometry = image.info["_ocr_page_geometry"]
        evidence = assess_raster_text_regions(image)
        inferred = _infer_page_ocr(
            job, image, np.asarray(image) if np is not None else None,
            geometry.for_raster(*image.size), evidence, False, page_num,
            job.source_hash[:16] if job.source_hash else str(job_id)[:16],
            socket.gethostname(), ocr_engine_lane,
            preferred_language=(stage.inference.get("fallback_language") if stage.inference else None),
            script_evidence_override=stage.script_evidence,
            reading_direction_override=stage.reading_direction,
        )
        prior = _stage_inference_from_json(stage.inference, job) if stage.inference else None
        cross_engine_pending = (
            prior is None
            and inferred.get("request_cross_engine")
            and evaluate_engine(catalog.TESSERACT, inferred["engine_policy"]).admitted
        )
        if cross_engine_pending:
            durable = _stage_inference_to_json(inferred)
            with transaction.atomic():
                locked = PageOCRStage.objects.select_for_update().get(pk=stage_id)
                current = Job.objects.select_for_update().get(pk=job.pk)
                _stage_fence(locked, current)
                if locked.status != "infering" or locked.lease_token != token:
                    return {"status": "stale"}
                locked.inference = durable
                locked.status = "rendered"
                locked.last_error = "cross_engine_pending"
                locked.lease_token = ""
                locked.lease_expires_at = None
                locked.save(update_fields=["inference", "status", "last_error", "lease_token", "lease_expires_at", "updated_at"])
                _stage_next_dispatch(
                    locked, "infer_page", "ocr_tesseract",
                    retry_index=(
                        f"fallback:{locked.render_attempts}:"
                        f"{locked.infer_attempts}:{locked.raster_sha256[:8]}"
                    ),
                    ocr_engine_lane="tesseract",
                )
            return {"status": "cross_engine_retry", "page_num": page_num}
        if prior is not None:
            choice = _compare_inference_candidates(prior, inferred)
            best = inferred if choice.use_candidate else prior
            audit = dict(prior.get("rerun_evidence") or {})
            audit.update(inferred.get("rerun_evidence") or {})
            kind = (
                "cross_engine" if stage.last_error == "cross_engine_pending"
                else "coverage_retry"
            )
            records = list((prior.get("rerun_evidence") or {}).get(kind, []))
            records.append({
                "dpi": stage.raster_dpi,
                "engine": inferred["ocr_method"],
                "prior_text": prior["page_text"],
                "text": inferred["page_text"],
                "text_sha256": hashlib.sha256(inferred["page_text"].encode("utf-8")).hexdigest(),
                "coverage": inferred["coverage_assessment"].coverage,
                "confidence": inferred["confidence"],
                "disposition": choice.reason,
                "selected": choice.use_candidate,
            })
            audit[kind] = records
            best = dict(best)
            best["rerun_evidence"] = audit
            best["candidate_disposition"] = choice.reason
            best["route"] = replace(best["route"], needs_review=True)
            inferred = best
        durable = _stage_inference_to_json(inferred)
        with transaction.atomic():
            locked = PageOCRStage.objects.select_for_update().get(pk=stage_id)
            current = Job.objects.select_for_update().get(pk=job.pk)
            _stage_fence(locked, current)
            if (
                locked.status != "infering"
                or locked.lease_token != token
                or locked.lease_expires_at <= timezone.now()
                or locked.raster_sha256 != raster_sha256
            ):
                return {"status": "stale"}
            locked.inference = durable
            next_dpi = (
                450 if locked.raster_dpi == 300 else
                600 if locked.raster_dpi == 450 else None
            )
            retry_coverage = (
                job.source_type == "pdf"
                and next_dpi is not None
                and _needs_source_dpi_retry(inferred)
            )
            locked.status = "cleanup" if retry_coverage else "inferred"
            if retry_coverage:
                locked.retry_dpi = next_dpi
                locked.last_error = f"retry_coverage:{next_dpi}"
            else:
                locked.last_error = ""
            locked.lease_token = ""
            locked.lease_expires_at = None
            locked.save(update_fields=["inference", "status", "retry_dpi", "last_error", "lease_token", "lease_expires_at", "updated_at"])
            if not retry_coverage:
                _stage_next_dispatch(
                    locked, "finalize_page", "ocr_finalize",
                    retry_index=(
                        f"render:{locked.render_attempts}:infer:"
                        f"{locked.infer_attempts}:{locked.raster_sha256[:8]}"
                    ),
                )
        if retry_coverage:
            try:
                cleanup_page_stage.apply_async(args=[stage_id], queue="coordinator")
            except Exception:
                logger.exception("Could not dispatch coverage retry cleanup %s", stage_id)
        return {"status": "coverage_retry" if retry_coverage else "inferred", "page_num": page_num}
    except _OcrFallbackRequired as exc:
        if _resolve_ocr_engine_lane(ocr_engine_lane) != "paddle":
            raise
        with transaction.atomic():
            locked = PageOCRStage.objects.select_for_update().get(pk=stage_id)
            current = Job.objects.select_for_update().get(pk=job.pk)
            _stage_fence(locked, current)
            if locked.status != "infering" or locked.lease_token != token:
                return {"status": "stale"}
            locked.status = "rendered"
            locked.last_error = "cross_engine_pending"
            locked.lease_token = ""
            locked.lease_expires_at = None
            locked.save(update_fields=["status", "last_error", "lease_token", "lease_expires_at", "updated_at"])
            _stage_next_dispatch(
                locked, "infer_page", "ocr_tesseract",
                retry_index=(
                    f"fallback:{locked.render_attempts}:"
                    f"{locked.infer_attempts}:{locked.raster_sha256[:8]}"
                ),
                ocr_engine_lane="tesseract",
            )
        logger.info("Staged page %s requires CPU Tesseract: %s", stage_id, exc)
        return {"status": "tesseract_handoff", "page_num": page_num}
    except Exception as exc:
        with transaction.atomic():
            locked = PageOCRStage.objects.select_for_update().get(pk=stage_id)
            if locked.status == "infering" and locked.lease_token == token:
                locked.status = "rendered"
                locked.lease_token = ""
                locked.lease_expires_at = None
                locked.last_error = (
                    "cross_engine_pending" if ocr_engine_lane == "tesseract"
                    else str(exc)[:1024]
                )
                locked.save(update_fields=["status", "lease_token", "lease_expires_at", "last_error", "updated_at"])
        raise


def _stage_complete(stage_id, backend):
    """Release reservation only after terminal custody and raster deletion."""
    with transaction.atomic():
        PageOCRByteBudget.objects.select_for_update().get(pk=1)
        stage = PageOCRStage.objects.select_for_update().get(pk=stage_id)
        current = Job.objects.select_for_update().get(pk=stage.job_id)
        stale = (
            stage.generation != current.dispatch_generation
            or stage.tenant_id != current.tenant_id
            or stage.source_sha256 != current.source_hash
        )
        terminal = PageResult.objects.filter(
            job=stage.job,
            page_num=stage.page_num,
            admitted_generation=stage.generation,
            status__in=("ok", "fallback", "image_only", "failed"),
        ).exists()
        if not stale and not terminal:
            return
        stage.status = "cleanup"
        if stale:
            stage.last_error = "stale_job_source_or_generation"
        stage.lease_token = ""
        stage.lease_expires_at = None
        stage.inference = {}
        stage.save(update_fields=["status", "last_error", "lease_token", "lease_expires_at", "inference", "updated_at"])
    if backend.backend_name == "s3" and is_presigned_mode():
        try:
            cleanup_page_stage.apply_async(args=[stage_id], queue="coordinator")
        except Exception:
            # The cleanup row and reservation remain durable for Beat recovery.
            logger.exception("Could not dispatch staged cleanup %s", stage_id)
    else:
        _stage_cleanup_artifact(stage_id, backend)


def _stage_cleanup_artifact(stage_id, backend):
    if is_litigation_hold_active():
        return
    stage = PageOCRStage.objects.get(pk=stage_id)
    if stage.status != "cleanup":
        return
    if stage.publish_safe_after and stage.publish_safe_after > timezone.now():
        if backend.backend_name != "s3":
            return
        # Move the uncertain exact key and its global byte reservation to a
        # durable independent tombstone. The page can retry with a new token
        # when capacity permits, while the old key keeps being swept even if
        # this stage advances or the parent job is later deleted.
        with transaction.atomic():
            PageOCRByteBudget.objects.select_for_update().get(pk=1)
            locked = PageOCRStage.objects.select_for_update().get(pk=stage_id)
            current = Job.objects.select_for_update().get(pk=locked.job_id)
            if (
                locked.status == "cleanup" and locked.raster_locator
                and locked.publish_safe_after
                and locked.publish_safe_after > timezone.now()
            ):
                PageOCRRasterRetirement.objects.get_or_create(
                    locator=locked.raster_locator,
                    defaults={
                        "job_uuid": current.pk,
                        "tenant_id": locked.tenant_id,
                        "generation": locked.generation,
                        "page_num": locked.page_num,
                        "backend_binding": _storage_binding_fingerprint("s3"),
                        "reserved_bytes": locked.reserved_bytes,
                        "safe_to_release_after": locked.publish_safe_after,
                        "next_sweep_at": timezone.now(),
                    },
                )
                locked.reserved_bytes = 0
                locked.raster_locator = ""
                locked.raster_sha256 = ""
                locked.publish_safe_after = None
                locked.save(update_fields=[
                    "reserved_bytes", "raster_locator", "raster_sha256",
                    "publish_safe_after", "updated_at",
                ])
        stage = PageOCRStage.objects.get(pk=stage_id)
        if stage.status != "cleanup":
            return
    try:
        if not stage.raster_locator:
            pass
        elif backend.backend_name == "nfs":
            os.unlink(stage.raster_locator)
        else:
            backend.delete(stage.raster_locator)
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception("Could not remove completed staged raster %s", stage.pk)
        return
    with transaction.atomic():
        PageOCRByteBudget.objects.select_for_update().get(pk=1)
        locked = PageOCRStage.objects.select_for_update().get(pk=stage_id)
        if locked.status != "cleanup" or locked.raster_locator != stage.raster_locator:
            return
        current = Job.objects.select_for_update().get(pk=locked.job_id)
        _stage_release_locked(locked)
        # Cleanup has either removed the exact artifact or transferred its
        # uncertain PUT to an independent retirement row. Do not strand stale
        # ownership metadata on terminal stages that recovery no longer scans.
        locked.lease_token = ""
        locked.lease_expires_at = None
        locked.publish_safe_after = None
        retry_render = locked.last_error.startswith(("retry_render:", "retry_coverage:"))
        current_authorized = (
            current.status == Job.Status.PROCESSING
            and locked.generation == current.dispatch_generation
            and locked.tenant_id == current.tenant_id
            and locked.source_sha256 == current.source_hash
        )
        if retry_render and current_authorized and locked.render_attempts < 4:
            locked.status = "pending"
            locked.raster_locator = ""
            locked.raster_sha256 = ""
            locked.publish_safe_after = None
            locked.redrive_count += 1
            locked.save(update_fields=["status", "raster_locator", "raster_sha256", "publish_safe_after", "lease_token", "lease_expires_at", "reserved_bytes", "redrive_count", "updated_at"])
            _stage_next_dispatch(
                locked, "render_page", "ocr_render",
                retry_index=f"cleanup:{locked.redrive_count}",
            )
        else:
            locked.status = (
                "obsolete" if not current_authorized or locked.last_error.startswith("stale_")
                else "failed" if locked.last_error.startswith(("staged page ", "retry_render:"))
                else "done"
            )
            locked.save(update_fields=["status", "reserved_bytes", "lease_token", "lease_expires_at", "publish_safe_after", "updated_at"])


def _sweep_retired_stage_rasters(*, limit=100):
    """Repeatedly delete exact old keys, including after reservation release."""
    if is_litigation_hold_active():
        return 0
    from .rls import control_plane_rls_context

    now = timezone.now()
    swept = 0
    with control_plane_rls_context():
        ids = list(
            PageOCRRasterRetirement.objects.filter(next_sweep_at__lte=now)
            .order_by("next_sweep_at").values_list("pk", flat=True)[:limit]
        )
        for retirement_id in ids:
            row = PageOCRRasterRetirement.objects.get(pk=retirement_id)
            if row.backend_binding != _storage_binding_fingerprint("s3"):
                PageOCRRasterRetirement.objects.filter(pk=row.pk).update(
                    last_error="storage_binding_changed",
                    next_sweep_at=now + timedelta(minutes=5),
                )
                continue
            try:
                backend = _get_storage_backend()
                if backend.backend_name != "s3":
                    raise RuntimeError("retired raster backend is not S3")
                backend.delete(row.locator)
            except Exception as exc:
                PageOCRRasterRetirement.objects.filter(pk=row.pk).update(
                    last_error=type(exc).__name__[:256],
                    next_sweep_at=now + timedelta(minutes=5),
                )
                continue
            with transaction.atomic():
                budget = PageOCRByteBudget.objects.select_for_update().get(pk=1)
                locked = PageOCRRasterRetirement.objects.select_for_update().get(
                    pk=retirement_id
                )
                if locked.reserved_bytes and now >= locked.safe_to_release_after:
                    if budget.reserved_bytes < locked.reserved_bytes:
                        raise RuntimeError("retired raster budget underflow")
                    budget.reserved_bytes -= locked.reserved_bytes
                    budget.save(update_fields=["reserved_bytes"])
                    locked.reserved_bytes = 0
                locked.last_swept_at = now
                locked.sweep_count += 1
                locked.last_error = ""
                # Keep the tombstone after releasing capacity. A remote PUT
                # that commits exceptionally late is still found and deleted
                # by later exact-key sweeps; no newer attempt shares this key.
                locked.next_sweep_at = now + timedelta(
                    minutes=5 if locked.reserved_bytes else 60
                )
                locked.save(update_fields=[
                    "reserved_bytes", "last_swept_at", "sweep_count",
                    "last_error", "next_sweep_at",
                ])
            swept += 1
    return swept


@shared_task(name="jobs.tasks.recover_page_stages", queue="coordinator")
def recover_page_stages():
    """Beat repair for staged worker loss and uncertain retired raster PUTs."""
    from .artifact_retirement import sweep_retired_job_artifacts

    try:
        retired_jobs = sweep_retired_job_artifacts()
    except Exception as exc:
        logger.error(
            "Retired job artifact sweep failed: %s", type(exc).__name__,
        )
        retired_jobs = {"swept": 0, "failed": 0, "error": type(exc).__name__}
    recovered = _recover_staged_pages()
    swept = _sweep_retired_stage_rasters()
    return {"recovered": recovered, "retired_swept": swept, "retired_jobs": retired_jobs}


@shared_task(name="jobs.tasks.cleanup_page_stage", queue="coordinator", acks_late=True)
def cleanup_page_stage(stage_id):
    """Credentialed coordinator cleanup; also retried by durable stage sweep."""
    from .rls import control_plane_rls_context

    with control_plane_rls_context():
        stage = PageOCRStage.objects.select_related("job").get(pk=stage_id)
        _stage_cleanup_artifact(stage_id, _get_backend_for_job(stage.job))
        return {"status": PageOCRStage.objects.get(pk=stage_id).status}


@shared_task(bind=True, name="jobs.tasks.finalize_page", queue="ocr_finalize", acks_late=True, reject_on_worker_lost=True, max_retries=3)
@_tenant_bound_task
def finalize_page(self, job_id, page_num, generation, tenant_id=None, dispatch_generation=None, presigned_capability_id=None):
    """CPU PDF/text publication through the same fenced page implementation."""
    del self, dispatch_generation
    stage = PageOCRStage.objects.get(
        job_id=job_id, page_num=page_num, generation=generation
    )
    if stage.status == "done":
        return {"status": "already_finalized", "page_num": page_num}
    if stage.status != "inferred":
        return {"status": stage.status, "page_num": page_num}
    return process_page.run(
        job_id, page_num, tenant_id=tenant_id,
        dispatch_generation=generation, stage_id=stage.pk,
        presigned_capability_id=presigned_capability_id,
    )


@shared_task(
    bind=True,
    name="jobs.tasks.process_page",
    queue="ocr_gpu",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
    default_retry_delay=60,
)
@_tenant_bound_task
def process_page(
    self,
    job_id,
    page_num,
    presigned_capability_id=None,
    ocr_engine_lane=None,
    tenant_id=None,
    dispatch_generation=None,
    stage_id=None,
):
    """Process a single page of a document.

    Idempotent: re-processing overwrites the same temp files.

    Args:
        presigned_capability_id: Optional job-bound capability UUID. The
            worker resolves its encrypted URL map after loading the job.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}
    _pin_rls_for_job(job)

    stale = _reject_stale_dispatch_generation(job, dispatch_generation)
    if stale is not None:
        return stale

    terminal = _reject_if_job_terminal(job)
    if terminal is not None:
        return terminal

    hostname = socket.gethostname()
    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
    page_lease_token = _claim_page_processing_lease(
        job,
        page_num,
        getattr(self.request, "id", None),
        document_id,
        hostname,
    )
    if not page_lease_token:
        _publish_dispatch_rows(_reconcile_fanout_page(job_id, page_num))
        return {
            "status": "stale",
            "job_id": str(job_id),
            "page_num": page_num,
            "message": "page processing lease is owned by another task or page is finalized",
        }
    Worker.objects.filter(hostname=hostname).update(
        status=Worker.Status.BUSY,
        current_task_id=page_lease_token,
        last_heartbeat=timezone.now(),
    )

    backend = None
    storage_mode = ""
    temp_dir = ""
    page_result = None
    staged_image = None
    staged_inference = None

    try:
        backend = _get_backend_for_job(job)
        storage_mode = backend.backend_name
        stage_input_urls = None
        if stage_id is not None:
            stage = PageOCRStage.objects.get(pk=stage_id)
            _stage_fence(stage, job)
            if stage.page_num != page_num or stage.status != "inferred":
                raise _ProcessingLeaseLost("finalization stage is not ready")
            if storage_mode == "s3" and is_presigned_mode():
                if not presigned_capability_id:
                    raise RuntimeError("staged finalization requires a presigned capability")
                stage_input_urls = _request_fresh_presigned_urls(
                    job, presigned_capability_id, "finalize_page", "input",
                    page_num=page_num, raster_sha256=stage.raster_sha256,
                    page_lease_token=page_lease_token,
                )
            staged_image = _stage_read_raster(stage, backend, stage_input_urls)
            staged_inference = _stage_inference_from_json(stage.inference, job)
        presigned_urls = None
        presigned_output_refresher = None
        if storage_mode == "s3" and is_presigned_mode():
            if not presigned_capability_id:
                raise RuntimeError("process_page S3 worker requires a capability")
            presigned_operation = "finalize_page" if stage_id is not None else "process_page"
            presigned_urls = stage_input_urls or _request_fresh_presigned_urls(
                job,
                presigned_capability_id,
                presigned_operation,
                "input",
                page_num=page_num,
                raster_sha256=(stage.raster_sha256 if stage_id is not None else None),
                page_lease_token=page_lease_token,
            )
            presigned_output_refresher = partial(
                _request_fresh_presigned_urls,
                job,
                presigned_capability_id,
                presigned_operation,
                "output",
                page_num=page_num,
                raster_sha256=(stage.raster_sha256 if stage_id is not None else None),
                page_lease_token=page_lease_token,
            )
        _renew_page_processing_lease(job_id, page_num, page_lease_token)
        if storage_mode == "nfs":
            job_path = _nfs_job_path_for_job(job)
            source_lease = nullcontext(
                os.path.join(job_path, "source", os.path.basename(job.source_file))
            )
        else:
            # Share an immutable source snapshot across page workers. Every
            # lease rechecks this page's authority before granting a warm hit.
            import tempfile

            temp_dir = tempfile.mkdtemp(prefix=f"page_{job_id}_{page_num}_")
            if not isinstance(backend, CachedS3Backend):
                raise RuntimeError("S3 page source requires immutable source cache")
            source_key = _job_storage_key(
                job_id, f"source/{os.path.basename(job.source_file)}"
            )
            if presigned_urls is not None:
                if not presigned_urls.get("source_get"):
                    raise RuntimeError("presigned page source URL is missing")
                revision = presigned_urls["source_revision_token"]
                size = int(presigned_urls["source_size_bytes"])
            else:
                revision, size = backend.source_revision_info(source_key)
            source_identity = SourceIdentity(
                str(job.tenant_id or f"local:{job.pk}"),
                str(job.pk),
                int(job.dispatch_generation),
                source_key,
                job.source_hash,
                revision,
                size,
            )

            def authorize_source():
                if presigned_urls is not None:
                    if stage_id is not None:
                        validate_stage_capability(
                            job, stage, presigned_capability_id, "finalize_page",
                            raster_sha256=stage.raster_sha256,
                        )
                    else:
                        open_presigned_capability(
                            str(job_id), presigned_capability_id, "process_page"
                        )
                _renew_page_processing_lease(job_id, page_num, page_lease_token)

            if presigned_urls is not None:
                source_lease = lease_presigned_source(
                    backend._source_cache,
                    source_identity,
                    presigned_urls["source_get"],
                    authorize=authorize_source,
                )
            else:
                source_lease = backend.lease_source(
                    source_identity, authorize=authorize_source
                )

        with source_lease as source_path:
            page_result = _process_single_page(
                job,
                source_path,
                page_num,
                document_id,
                hostname,
                backend,
                storage_mode,
                s3_temp_dir=temp_dir,
                presigned_urls=presigned_urls,
                lease_token=page_lease_token,
                source_fence=getattr(source_path, "check", None),
                presigned_output_refresher=presigned_output_refresher,
                ocr_engine_lane=ocr_engine_lane,
                prepared_image=staged_image,
                precomputed_inference=staged_inference,
            )
    except _OcrFallbackRequired as exc:
        if _resolve_ocr_engine_lane(ocr_engine_lane) != "paddle":
            raise RuntimeError(
                "fallback handoff originated outside Paddle lane"
            ) from exc
        _renew_page_processing_lease(job_id, page_num, page_lease_token)
        _record_custody_event(
            job,
            document_id,
            "ocr_fallback_dispatched",
            data={
                "page_num": page_num,
                "from_lane": "paddle",
                "to_lane": "tesseract",
                "reason": str(exc),
            },
            worker_hostname=hostname,
        )
        retry_kwargs = {
            "presigned_capability_id": presigned_capability_id,
            "ocr_engine_lane": "tesseract",
        }
        if str(tenant_id or "").strip():
            retry_kwargs["tenant_id"] = str(tenant_id).strip()
        if dispatch_generation is not None:
            retry_kwargs["dispatch_generation"] = dispatch_generation
        raise self.retry(
            exc=exc,
            args=(str(job_id), int(page_num)),
            kwargs=retry_kwargs,
            queue="ocr_tesseract",
            countdown=0,
        )
    except _ProcessingLeaseLost as exc:
        logger.warning(
            "process_page stopped after lease loss for job %s page %s: %s",
            job_id,
            page_num,
            exc,
        )
        _publish_dispatch_rows(_reconcile_fanout_page(job_id, page_num))
        return {
            "status": "stale",
            "job_id": str(job_id),
            "page_num": page_num,
            "message": "page processing lease was fenced before the next durable write",
        }
    except Exception as exc:
        logger.error(
            "process_page failed for job %s page %d: %s", job_id, page_num, exc
        )
        try:
            _record_page_failed(
                job,
                page_num,
                document_id,
                hostname,
                str(exc),
                lease_token=page_lease_token,
            )
            _reconcile_fanout_page(job_id, page_num)
            if stage_id is not None and backend is not None:
                _stage_complete(stage_id, backend)
        except _ProcessingLeaseLost:
            return {
                "status": "stale",
                "job_id": str(job_id),
                "page_num": page_num,
            }
        # Increment failure counter
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        raise
    finally:
        if storage_mode == "s3" and temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        Worker.objects.filter(hostname=hostname).update(
            status=Worker.Status.ONLINE,
            current_task_id="",
            last_heartbeat=timezone.now(),
        )

    # B02 / Sweep B: domain soft-failure must not Celery-SUCCESS the chord
    # header (suppresses link_error / chord_error_handler and inflates
    # pages_completed). PageResult already marked failed; raise for errbacks.
    if isinstance(page_result, dict) and page_result.get("status") == "failed":
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        _reconcile_fanout_page(job_id, page_num)
        raise RuntimeError(
            "page soft-failed: "
            f"{page_result.get('error') or 'unknown'} "
            f"(job={job_id} page={page_num}; B02 fail-closed)"
        )

    reported_status = (
        page_result.get("status") if isinstance(page_result, dict) else None
    )
    if reported_status in _SUCCESSFUL_PAGE_STATUSES:
        PageResult.objects.filter(
            job_id=job_id,
            page_num=page_num,
            status="processing",
            celery_task_id=page_lease_token,
        ).update(status=reported_status)

    if not Job.objects.filter(job_id=job_id, status=Job.Status.PROCESSING).exists():
        return {
            "status": "stale",
            "job_id": str(job_id),
            "page_num": page_num,
            "message": "job state changed before page progress commit",
        }
    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )

    enqueue_layoutlm = False
    try:
        from jobs.layoutlm_config import ENABLE_LAYOUTLM

        enqueue_layoutlm = bool(ENABLE_LAYOUTLM)
    except Exception:
        logger.exception(
            "LayoutLM configuration failed for job=%s page=%s (non-fatal to OCR)",
            job_id,
            page_num,
        )
    dispatch_rows = _reconcile_fanout_page(
        job_id,
        page_num,
        enqueue_layoutlm=enqueue_layoutlm,
    )
    _publish_dispatch_rows(dispatch_rows)
    if stage_id is not None and backend is not None:
        _stage_complete(stage_id, backend)

    return {
        "status": "page_processed",
        "job_id": str(job_id),
        "page_num": page_num,
        "page_status": (page_result or {}).get("status")
        if isinstance(page_result, dict)
        else None,
    }


def _infer_page_ocr(
    job, img, img_array, transform, raster_evidence, geometry_uncertain,
    page_num, document_id, hostname, ocr_engine_lane, preferred_language=None,
    script_evidence_override=None,
    reading_direction_override=None,
):
    """Shared bounded recognition path for inline and staged page execution."""
    from ocr_distributed.ocr_utils import create_paddle_engine, extract_paddle_lines

    coverage_assessment = assess_candidate_coverage(raster_evidence, ())
    engine_lane = _resolve_ocr_engine_lane(ocr_engine_lane)
    script_evidence = (
        tuple(script_evidence_override)
        if script_evidence_override is not None
        else () if engine_lane == "paddle" else _page_script_evidence(img, job)
    )
    route = route_language(
        job.detected_language,
        explicit_hint=(
            preferred_language or (job.settings_json or {}).get("language_hint")
        ),
        script_evidence=script_evidence,
    )
    if geometry_uncertain:
        route = replace(route, needs_review=True)
    geometry = img.info.get("_ocr_page_geometry")
    source_rotation = (
        int(geometry.source_rotation)
        if isinstance(geometry, PageGeometry) else 0
    )
    if reading_direction_override is not None:
        reading_direction = dict(reading_direction_override)
    elif engine_lane == "paddle" and source_rotation:
        reading_direction = {
            "orientation": None, "reason": "cpu_direction_probe_unavailable"
        }
    else:
        reading_direction = _page_reading_direction(img, geometry, route)
    reading_rotation = reading_direction.get("orientation")
    if reading_rotation not in (0, 90, 180, 270):
        reading_rotation = 0
        route = replace(route, needs_review=True)
    model_language = route.model_language
    # A combined worker preserves the rollback/development behavior. A
    # capability worker imports and executes only its owned engine.
    ocr_method = "Tesseract" if engine_lane == "tesseract" else "PaddleOCR"
    ocr_lines = []
    ocr_lines_pre_mapped = False
    overlay_issues = []
    candidate_disposition = "not_attempted"
    rerun_evidence = None
    attempted_tesseract_model = ""
    confidence = 0.0
    page_text = ""
    engine_policy = _engine_policy_from_job_settings(job.settings_json)
    lang_code = model_language
    if engine_lane in {"combined", "paddle"}:
        paddle_decision = evaluate_engine(catalog.PADDLE_CTC, engine_policy)
        if paddle_decision.admitted and route.engine == "paddle":
            try:
                if _gpu_available():
                    local_gpu = os.environ.get("OCR_GPU_LOCAL_DEVICE", "").strip()
                    device = f"gpu:{local_gpu}" if local_gpu else "gpu"
                else:
                    device = "cpu"
                result = (
                    _run_paddle_ocr(
                        lang_code, device, img_array, create_paddle_engine
                    )
                    if img_array is not None
                    else None
                )
                ocr_lines = extract_paddle_lines(result)
                coverage_assessment = assess_candidate_coverage(
                    raster_evidence, ocr_lines, candidate_raster_size=img.size
                )
                if ocr_lines and not (job.settings_json or {}).get(
                    "language_hint"
                ):
                    first_text = "\n".join(text for text, _, _ in ocr_lines)
                    qualified_language = _page_text_language(first_text)
                    evidence_language, evidence_confidence = (
                        _page_text_language_evidence(first_text)
                    )
                    page_language = qualified_language or evidence_language
                    if page_language and page_language != route.detected_label:
                        candidate_route = route_language(page_language)
                        if (
                            candidate_route.engine == "paddle"
                            and candidate_route.runtime_model == route.runtime_model
                        ):
                            route = replace(
                                route,
                                detected_label=(
                                    page_language if qualified_language
                                    else route.detected_label
                                ),
                                reason=(
                                    "per_page_text_detected_same_model"
                                    if qualified_language else "per_page_model_candidate"
                                ),
                                needs_review=(
                                    coverage_assessment.needs_review
                                    or not qualified_language
                                ),
                            )
                            candidate_disposition = "same_model_detection"
                        elif candidate_route.engine == "paddle":
                            try:
                                rerun_result = _run_paddle_ocr(
                                    candidate_route.model_language,
                                    device,
                                    img_array,
                                    create_paddle_engine,
                                )
                                rerun_lines = extract_paddle_lines(rerun_result)
                            except Exception as rerun_exc:
                                logger.warning(
                                    "Language rerun failed for page %d: %s",
                                    page_num,
                                    rerun_exc,
                                )
                                rerun_lines = []
                            rerun_coverage = assess_candidate_coverage(
                                raster_evidence,
                                rerun_lines,
                                candidate_raster_size=img.size,
                            )
                            first_confidences = [
                                score for _, _, score in ocr_lines if score >= 0
                            ]
                            rerun_confidences = [
                                score for _, _, score in rerun_lines if score >= 0
                            ]
                            rerun_text = "\n".join(
                                text for text, _, _ in rerun_lines
                            )
                            rerun_language, rerun_language_confidence = (
                                _page_text_language_evidence(rerun_text)
                            )
                            first_support = (
                                evidence_confidence
                                if evidence_language == model_language else 0.0
                            )
                            retry_support = (
                                rerun_language_confidence
                                if rerun_language == candidate_route.model_language
                                else 0.0
                            )
                            decision = compare_ocr_candidates(
                                OCRCandidate(
                                    first_text,
                                    "paddle",
                                    sum(first_confidences) / len(first_confidences)
                                    if first_confidences
                                    else None,
                                    coverage_assessment.coverage,
                                    True,
                                    first_support,
                                ),
                                OCRCandidate(
                                    rerun_text,
                                    "paddle",
                                    sum(rerun_confidences) / len(rerun_confidences)
                                    if rerun_confidences
                                    else None,
                                    rerun_coverage.coverage,
                                    True,
                                    retry_support,
                                ),
                            )
                            candidate_disposition = decision.reason
                            rerun_evidence = {
                                "first": {
                                    "model_language": model_language,
                                    "text": first_text,
                                    "language_support": first_support,
                                    "text_sha256": hashlib.sha256(
                                        first_text.encode("utf-8")
                                    ).hexdigest(),
                                    "coverage": coverage_assessment.coverage,
                                },
                                "retry": {
                                    "model_language": candidate_route.model_language,
                                    "text": rerun_text,
                                    "language_support": retry_support,
                                    "coverage": rerun_coverage.coverage,
                                },
                                "selected": "retry" if decision.use_candidate else "first",
                                "disposition": decision.reason,
                            }
                            rerun_evidence["retry"]["text_sha256"] = (
                                hashlib.sha256(
                                    rerun_evidence["retry"]["text"].encode("utf-8")
                                ).hexdigest()
                            )
                            if decision.use_candidate:
                                ocr_lines = rerun_lines
                                coverage_assessment = rerun_coverage
                                route = replace(
                                    candidate_route,
                                    detected_label=(
                                        candidate_route.detected_label
                                        if rerun_language_confidence >= 0.7
                                        and rerun_language == candidate_route.model_language
                                        else route.detected_label
                                    ),
                                    reason="per_page_rerun_accepted",
                                    needs_review=(
                                        decision.needs_review
                                        or rerun_coverage.needs_review
                                        or rerun_language_confidence < 0.7
                                    ),
                                )
                            else:
                                route = replace(route, needs_review=True)
                        else:
                            candidate_disposition = (
                                "detected_unsupported_language_review"
                            )
                            route = replace(route, needs_review=True)
            except Exception as exc:
                logger.warning("PaddleOCR failed for page %d: %s", page_num, exc)
                ocr_lines = []
        elif paddle_decision.admitted:
            logger.info(
                "PaddleOCR 2.9.1 has no recognition model for language %s; "
                "routing page %d to Tesseract",
                lang_code,
                page_num,
            )
        else:
            _record_engine_denial(
                job,
                document_id,
                hostname,
                page_num,
                catalog.PADDLE_CTC,
                engine_policy,
                paddle_decision.reason,
            )
            logger.warning(
                "PaddleOCR denied by EnginePolicy for page %d: %s",
                page_num,
                paddle_decision.reason,
            )

    if (
        ocr_lines and ocr_method == "PaddleOCR"
        and reading_rotation in (90, 180, 270)
        and str(reading_direction.get("reference_text") or "").strip()
        and img_array is not None
    ):
        # The independent CPU raster probe qualified this OCR-only reading
        # direction. Keep the visual source unchanged and inverse-map the
        # second same-model detector geometry back to its original pixels.
        reference = "".join(
            char.casefold()
            for char in str(reading_direction["reference_text"])
            if char.isalnum()
        )
        original_text = " ".join(text for text, _, _ in ocr_lines)
        original_normalized = "".join(
            char.casefold() for char in original_text if char.isalnum()
        )
        try:
            rotated_image = img.rotate(reading_rotation, expand=True)
            rotated_result = _run_paddle_ocr(
                route.model_language, device, np.asarray(rotated_image),
                create_paddle_engine,
            )
            rotated_lines = extract_paddle_lines(rotated_result)
            _pixel_lines, pdf_lines, rotated_issues = (
                map_rotated_detector_lines_to_source(
                    rotated_lines, transform, reading_rotation
                )
            )
            rotated_text = " ".join(text for text, _, _ in pdf_lines)
            rotated_normalized = "".join(
                char.casefold() for char in rotated_text if char.isalnum()
            )
            rotated_raster_evidence = assess_raster_text_regions(rotated_image)
            rotated_coverage = assess_candidate_coverage(
                rotated_raster_evidence, rotated_lines,
                candidate_raster_size=rotated_image.size,
            )
            rotated_scores = [score for _, _, score in pdf_lines if score >= 0]
            rotated_confidence = (
                sum(rotated_scores) / len(rotated_scores)
                if rotated_scores else 0.0
            )
            script = LANGUAGE_REGISTRY[route.model_language].script
            accepted = (
                bool(reference) and bool(rotated_normalized)
                and rotated_normalized == reference
                and original_normalized != reference
                and not rotated_issues
                and rotated_confidence >= 0.70
                and rotated_coverage.status == "measured"
                and (
                    rotated_coverage.coverage + 0.02 >= coverage_assessment.coverage
                    if coverage_assessment.status == "measured"
                    else rotated_coverage.coverage >= 0.90
                )
                and text_supports_script(rotated_text, script)
            )
            existing_audit = rerun_evidence if isinstance(rerun_evidence, dict) else {}
            rerun_evidence = {
                **existing_audit,
                "orientation_retry": {
                    "source_rotation": source_rotation,
                    "qualified_rotation": reading_rotation,
                    "prior_text": original_text[:4096],
                    "candidate_text": rotated_text[:4096],
                    "reference_sha256": hashlib.sha256(
                        reference.encode("utf-8")
                    ).hexdigest(),
                    "candidate_confidence": rotated_confidence,
                    "candidate_coverage": rotated_coverage.coverage,
                    "selected": accepted,
                    "reason": (
                        "independent_direction_text_match"
                        if accepted else "direction_candidate_not_corroborated"
                    ),
                },
            }
            if accepted:
                ocr_lines = pdf_lines
                ocr_lines_pre_mapped = True
                coverage_assessment = rotated_coverage
                candidate_disposition = "independent_direction_text_match"
                route = replace(route, needs_review=True)
            else:
                route = replace(route, needs_review=True)
        except Exception as exc:
            logger.warning(
                "Bounded rotation OCR candidate failed for page %d: %s",
                page_num, exc,
            )
            route = replace(route, needs_review=True)

    model_language = route.model_language
    lang_code = model_language
    actual_runtime_model = route.runtime_model

    if not ocr_lines and engine_lane == "paddle":
        tesseract_decision = evaluate_engine(catalog.TESSERACT, engine_policy)
        if tesseract_decision.admitted:
            raise _OcrFallbackRequired(
                f"Paddle produced no admitted text for page {page_num}"
            )
        _record_engine_denial(
            job,
            document_id,
            hostname,
            page_num,
            catalog.TESSERACT,
            engine_policy,
            tesseract_decision.reason,
        )
        ocr_method = "ImageOnly"

    # Combined workers execute the legacy inline fallback. A Tesseract
    # worker enters here directly and never initializes Paddle.
    if not ocr_lines and engine_lane in {"combined", "tesseract"}:
        ocr_method = "Tesseract"
        tesseract_decision = evaluate_engine(catalog.TESSERACT, engine_policy)
        if tesseract_decision.admitted:
            try:
                import pytesseract

                tess_runtime_model = (
                    route.runtime_model
                    if route.engine == "tesseract"
                    else get_tesseract_code(lang_code)
                )
                attempted_tesseract_model = tess_runtime_model
                tess_data = pytesseract.image_to_data(
                    img,
                    lang=tess_runtime_model,
                    output_type=pytesseract.Output.DICT,
                    timeout=TESSERACT_TIMEOUT,
                )
                actual_runtime_model = tess_runtime_model
                ocr_lines, tess_issues = tesseract_data_to_lines_with_review(
                    tess_data, transform=transform
                )
                tess_raster_spans = []
                for index, text in enumerate(tess_data.get("text", ())):
                    if not str(text or "").strip():
                        continue
                    try:
                        left = float(tess_data["left"][index])
                        top = float(tess_data["top"][index])
                        width = float(tess_data["width"][index])
                        height = float(tess_data["height"][index])
                        score = float(tess_data["conf"][index])
                    except (IndexError, KeyError, TypeError, ValueError):
                        continue
                    if score < 0:
                        continue
                    tess_raster_spans.append(
                        (str(text), (left, top, left + width, top + height), score / 100)
                    )
                coverage_assessment = assess_candidate_coverage(
                    raster_evidence,
                    tess_raster_spans,
                    candidate_raster_size=img.size,
                )
                overlay_issues.extend(
                    {"reason": issue["reason"], "index": issue["index"]}
                    for issue in tess_issues[:8]
                )
                raw_words = [
                    str(text).strip()
                    for text in tess_data.get("text", ())
                    if str(text or "").strip()
                ]
                page_text = " ".join(raw_words)
                if len(tess_issues) > 8:
                    overlay_issues.append(
                        {"reason": "additional_unplaceable_words", "count": len(tess_issues) - 8}
                    )
                if not raw_words:
                    ocr_method = "ImageOnly"
                    model_language = ""
                    actual_runtime_model = ""
                    route = replace(route, needs_review=True)
            except Exception as exc:
                logger.warning(
                    "Tesseract fallback failed for page %d: %s", page_num, exc
                )
                if engine_lane == "tesseract":
                    raise RuntimeError(
                        f"Tesseract lane failed for page {page_num}"
                    ) from exc
                overlay_issues.append(
                    {"reason": "tesseract_execution_failed", "error_type": type(exc).__name__}
                )
                ocr_method = "ImageOnly"
                model_language = ""
                actual_runtime_model = ""
        else:
            _record_engine_denial(
                job,
                document_id,
                hostname,
                page_num,
                catalog.TESSERACT,
                engine_policy,
                tesseract_decision.reason,
            )
            logger.warning(
                "Tesseract denied by EnginePolicy for page %d: %s",
                page_num,
                tesseract_decision.reason,
            )
            ocr_method = "ImageOnly"

    # Paddle boxes are raster pixels; Tesseract boxes were transformed by
    # tesseract_data_to_lines above.  A zero confidence remains evidence.
    if ocr_lines and ocr_method == "PaddleOCR":
        page_text = "\n".join(txt for txt, _, _ in ocr_lines)
        confidences = [c for _, _, c in ocr_lines if c >= 0]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if ocr_lines_pre_mapped:
            mapped_issues = []
        else:
            ocr_lines, mapped_issues = transform.map_lines_with_review(ocr_lines)
        overlay_issues.extend(
            {"reason": issue["reason"], "index": issue["index"]}
            for issue in mapped_issues[:8]
        )
    elif ocr_lines and ocr_method == "Tesseract":
        confidences = [c for _, _, c in ocr_lines]
        confidence = sum(confidences) / len(confidences)

    evidence_language, evidence_confidence = _page_text_language_evidence(page_text)
    if (
        ocr_method == "Tesseract"
        and evidence_language == model_language
        and evidence_confidence >= 0.7
    ):
        route = replace(
            route,
            detected_label=evidence_language,
            reason="post_tesseract_text_confirmed",
        )
    fallback_language = (
        model_language if model_language != "en"
        else evidence_language if evidence_confidence >= 0.4
        else model_language
    )
    target_support = (
        evidence_confidence if evidence_language == fallback_language else 0.0
    )
    request_cross_engine = (
        ocr_method == "PaddleOCR"
        and bool(page_text.strip())
        and (
            confidence < 0.55
            or (fallback_language != "en" and target_support < 0.85)
        )
    )
    return {
        "ocr_lines": ocr_lines,
        "route": route,
        "model_language": model_language,
        "actual_runtime_model": actual_runtime_model,
        "candidate_disposition": candidate_disposition,
        "confidence": confidence,
        "page_text": page_text,
        "engine_policy": engine_policy,
        "lang_code": lang_code,
        "attempted_tesseract_model": attempted_tesseract_model,
        "script_evidence": script_evidence,
        "reading_direction": reading_direction,
        "reading_rotation": reading_rotation,
        # The qualified rotation retry has already mapped its quads back to
        # source PDF coordinates. Applying that rotation again at insertion
        # mirrors searchable boxes even though the OCR text is correct.
        "overlay_rotation": 0 if ocr_lines_pre_mapped else reading_rotation,
        "coverage_assessment": coverage_assessment,
        "ocr_method": ocr_method,
        "overlay_issues": overlay_issues,
        "rerun_evidence": rerun_evidence,
        "request_cross_engine": request_cross_engine,
        "fallback_language": fallback_language,
    }


def _retry_low_coverage_from_source(
    job, source_path, page_num, base_image, base_evidence, first,
    document_id, hostname, ocr_engine_lane,
):
    """Re-render authorized source at bounded DPI and keep the best candidate."""
    from ocr_distributed.ocr_utils import iter_source_images

    if (
        base_evidence.status != "supported"
        or not _needs_source_dpi_retry(first)
        or job.source_type not in {"pdf", "image"}
    ):
        return base_image, first
    # Image sources have no independent higher-DPI render; upscaling the same
    # pixels cannot create evidence and is not a completeness retry.
    if job.source_type == "image":
        return base_image, first
    current_image = base_image
    current = first
    retry_records = []
    for dpi in (450, 600):
        try:
            if not hmac.compare_digest(
                _compute_file_hash(source_path), job.source_hash
            ):
                raise _ProcessingLeaseLost("source changed before OCR DPI retry")
            _stage_preflight_raster(
                source_path, page_num, job.source_type, _stage_limits()[0], dpi=dpi,
            )
            retry_image = next(
                iter_source_images(
                    source_path, page_num, page_num, job.source_type, dpi=dpi
                )
            )
            if not hmac.compare_digest(
                _compute_file_hash(source_path), job.source_hash
            ):
                raise _ProcessingLeaseLost("source changed during OCR DPI retry")
            if retry_image.width * retry_image.height * 4 > _stage_limits()[0]:
                retry_records.append({"dpi": dpi, "disposition": "decoded_byte_limit"})
                break
            retry_geometry = retry_image.info.get("_ocr_page_geometry")
            if not isinstance(retry_geometry, PageGeometry):
                raise RuntimeError("retry render lacks authorized source geometry")
            candidate = _infer_page_ocr(
                job, retry_image,
                np.asarray(retry_image) if np is not None else None,
                retry_geometry.for_raster(*retry_image.size),
                base_evidence, False, page_num, document_id, hostname,
                ocr_engine_lane,
            )
            decision = _compare_inference_candidates(current, candidate)
            retry_records.append({
                "dpi": dpi,
                "text": candidate["page_text"],
                "text_sha256": hashlib.sha256(candidate["page_text"].encode("utf-8")).hexdigest(),
                "coverage": candidate["coverage_assessment"].coverage,
                "confidence": candidate["confidence"],
                "disposition": decision.reason,
                "selected": decision.use_candidate,
            })
            if decision.use_candidate:
                current_image = retry_image
                current = candidate
                current["candidate_disposition"] = decision.reason
            if not _needs_source_dpi_retry(current):
                break
        except _ProcessingLeaseLost:
            raise
        except Exception as exc:
            retry_records.append({
                "dpi": dpi, "disposition": "retry_failed",
                "error_type": type(exc).__name__,
            })
    if retry_records:
        current = dict(current)
        current["route"] = replace(current["route"], needs_review=True)
        existing = current.get("rerun_evidence")
        current["rerun_evidence"] = {
            **(existing if isinstance(existing, dict) else {}),
            "coverage_retry": retry_records,
        }
    return current_image, current


def _process_single_page(
    job,
    source_path,
    page_num,
    document_id,
    hostname,
    backend=None,
    storage_mode=None,
    s3_temp_dir="",
    presigned_urls=None,
    lease_token=None,
    publish_fence=None,
    source_fence=None,
    presigned_output_refresher=None,
    ocr_engine_lane=None,
    prepared_image=None,
    precomputed_inference=None,
):
    """Core OCR processing for a single page.

    Uses PaddleOCR as primary engine with Tesseract fallback and image-only
    as last resort. Writes temp PDF and text files to storage backend.

    Args:
        presigned_urls: Optional dict with page_pdf_put and page_text_put
            presigned S3 URLs for uploading results without credentials.

    Returns dict with OCR result metadata.
    """
    import fitz

    from ocr_distributed.ocr_utils import (
        img_to_bytes,
        insert_text_line,
        iter_source_images,
    )

    # Determine storage mode if not provided
    if backend is None:
        backend = _get_storage_backend()
        storage_mode = backend.backend_name

    # Setup temp directory
    owns_temp_dir = False
    if storage_mode == "nfs":
        job_path = _nfs_job_path_for_job(job)
        temp_dir = os.path.join(job_path, "temp", document_id)
        if lease_token:
            temp_dir = os.path.join(
                temp_dir, f"g{int(job.dispatch_generation)}", f"p{int(page_num)}",
                f"l{lease_token}",
            )
        _mkdir_tree_no_follow(temp_dir, getattr(settings, "NFS_ROOT", job_path))
    else:
        if s3_temp_dir:
            temp_dir = s3_temp_dir
        else:
            import tempfile

            temp_dir = tempfile.mkdtemp(prefix=f"ocr_page_{page_num}_")
            owns_temp_dir = True

    start_time = time.monotonic()

    try:
        # Extract page image
        images = (
            [prepared_image]
            if prepared_image is not None
            else list(
                iter_source_images(
                    source_path,
                    page_num,
                    page_num,
                    job.source_type,
                )
            )
        )
        if not images:
            return _record_page_failed(
                job,
                page_num,
                document_id,
                hostname,
                "No image extracted",
                lease_token=lease_token,
            )

        img = images[0]
        geometry = img.info.get("_ocr_page_geometry")
        geometry_uncertain = not isinstance(geometry, PageGeometry)
        if geometry_uncertain:
            # Synthetic/in-memory callers have no source-page metadata.  The
            # real source iterator attaches the authorized render's geometry.
            geometry = PageGeometry.from_image_pixels(
                *img.size, dpi=img.info.get("dpi")
            )
        transform = geometry.for_raster(*img.size)
        img_array = None
        if np is not None:
            try:
                img_array = np.array(img)
            except Exception:
                img_array = None

        # This reference is measured from source pixels before looking at OCR
        # output.  Unknown evidence is reviewable; it never authorizes skipping
        # OCR or selecting the longer candidate.
        if precomputed_inference is None:
            raster_evidence = assess_raster_text_regions(img)
            engine_lane = _resolve_ocr_engine_lane(ocr_engine_lane)
            prior_inference = None
            if engine_lane == "tesseract" and lease_token:
                page_row = PageResult.objects.get(job=job, page_num=page_num)
                envelope = (page_row.ocr_provenance or {}).get("temporary_candidate")
                if (
                    isinstance(envelope, dict)
                    and envelope.get("generation") == job.dispatch_generation
                    and envelope.get("source_sha256") == job.source_hash
                    and envelope.get("tenant_id") == job.tenant_id
                ):
                    prior_inference = _stage_inference_from_json(envelope["inference"], job)
            inferred = _infer_page_ocr(
                job, img, img_array, transform, raster_evidence, geometry_uncertain,
                page_num, document_id, hostname, ocr_engine_lane,
                preferred_language=(
                    prior_inference.get("fallback_language")
                    if prior_inference is not None else None
                ),
            )
            if (
                inferred.get("request_cross_engine")
                and engine_lane in {"paddle", "combined"}
                and evaluate_engine(catalog.TESSERACT, inferred["engine_policy"]).admitted
            ):
                if engine_lane == "paddle" and lease_token:
                    envelope = {
                        "assessment_state": "cross_engine_pending",
                        "temporary_candidate": {
                            "generation": job.dispatch_generation,
                            "source_sha256": job.source_hash,
                            "tenant_id": job.tenant_id,
                            "inference": _stage_inference_to_json(inferred),
                        },
                    }
                    updated = PageResult.objects.filter(
                        job=job, page_num=page_num, status="processing",
                        celery_task_id=lease_token,
                    ).update(ocr_provenance=envelope)
                    if updated != 1:
                        raise _ProcessingLeaseLost("cross-engine candidate lost page lease")
                    raise _OcrFallbackRequired("weak Paddle candidate needs independent Tesseract check")
                if engine_lane == "combined":
                    prior_inference = inferred
                    inferred = _infer_page_ocr(
                        job, img, img_array, transform, raster_evidence,
                        geometry_uncertain, page_num, document_id, hostname,
                        "tesseract", preferred_language=inferred["fallback_language"],
                    )
            if prior_inference is not None:
                choice = _compare_inference_candidates(prior_inference, inferred)
                selected = inferred if choice.use_candidate else prior_inference
                selected = dict(selected)
                audit = dict(selected.get("rerun_evidence") or {})
                audit["cross_engine"] = [{
                    "prior_engine": prior_inference["ocr_method"],
                    "prior_text": prior_inference["page_text"],
                    "candidate_engine": inferred["ocr_method"],
                    "candidate_text": inferred["page_text"],
                    "disposition": choice.reason,
                    "selected": choice.use_candidate,
                }]
                selected["rerun_evidence"] = audit
                selected["candidate_disposition"] = choice.reason
                selected["route"] = replace(selected["route"], needs_review=True)
                inferred = selected
            img, inferred = _retry_low_coverage_from_source(
                job, source_path, page_num, img, raster_evidence, inferred,
                document_id, hostname, ocr_engine_lane,
            )
            geometry = img.info.get("_ocr_page_geometry", geometry)
            transform = geometry.for_raster(*img.size)
        else:
            inferred = precomputed_inference
        if (
            precomputed_inference is not None
            or _resolve_ocr_engine_lane(ocr_engine_lane) == "combined"
        ):
            inferred = _verify_paddle_text_revision(inferred, img, transform)
        ocr_lines = inferred["ocr_lines"]
        route = inferred["route"]
        model_language = inferred["model_language"]
        actual_runtime_model = inferred["actual_runtime_model"]
        candidate_disposition = inferred["candidate_disposition"]
        confidence = inferred["confidence"]
        page_text = inferred["page_text"]
        engine_policy = inferred["engine_policy"]
        lang_code = inferred["lang_code"]
        attempted_tesseract_model = inferred["attempted_tesseract_model"]
        script_evidence = inferred["script_evidence"]
        reading_direction = inferred.get("reading_direction") or {}
        reading_rotation = int(inferred.get("reading_rotation") or 0)
        overlay_rotation = int(inferred.get("overlay_rotation", reading_rotation) or 0)
        coverage_assessment = inferred["coverage_assessment"]
        ocr_method = inferred["ocr_method"]
        overlay_issues = inferred["overlay_issues"]
        rerun_evidence = inferred["rerun_evidence"]

        # Build output PDF page
        # The page background is an evidentiary view of the authorized raster.
        # JPEG85 can erase fine print and thin strokes independently of OCR.
        img_bytes = img_to_bytes(img, fmt="PNG")
        page_pdf = fitz.open()
        page_rect = fitz.Rect(0, 0, geometry.width_pt, geometry.height_pt)
        pdf_page = page_pdf.new_page(
            width=geometry.width_pt, height=geometry.height_pt
        )
        pdf_page.insert_image(page_rect, stream=img_bytes)

        # Insert text overlay
        if ocr_lines:
            for txt, box, _ in ocr_lines:
                if box:
                    try:
                        inserted = insert_text_line(
                            pdf_page, txt, box, lang_code=lang_code,
                            source_rotation=(
                                overlay_rotation
                                if ocr_method == "PaddleOCR" else 0
                            ),
                        )
                        if inserted is False:
                            overlay_issues.append({"reason": "text_insert_rejected"})
                    except Exception as exc:
                        overlay_issues.append(
                            {"reason": f"text_insert_failed:{type(exc).__name__}"}
                        )

        # Save temp page PDF
        temp_pdf_path = os.path.join(temp_dir, f"{page_num}.pdf")
        page_pdf.save(temp_pdf_path, deflate=True)
        page_pdf.close()

        # Verify the saved searchable layer against retained OCR text. Font
        # fallback can otherwise silently replace unsupported glyphs with
        # dots while still reporting a successful PyMuPDF insertion.
        if page_text.strip() and not overlay_issues:
            with fitz.open(temp_pdf_path) as saved_page:
                extracted_text = saved_page[0].get_text(sort=False)

            def normalized_text(value):
                return "".join(
                    char
                    for char in unicodedata.normalize("NFC", value)
                    if not char.isspace()
                )

            if normalized_text(page_text) != normalized_text(extracted_text):
                overlay_issues.append({"reason": "pdf_text_mismatch"})

        # Save temp page text
        temp_text_path = os.path.join(temp_dir, f"{page_num}.txt")
        with open(temp_text_path, "w", encoding="utf-8") as f:
            f.write(page_text)

        candidate_evidence_key = ""
        candidate_evidence_path = ""
        candidate_evidence_sha256 = ""
        if rerun_evidence is not None:
            attempt_keys = (
                page_attempt_keys(job, page_num, document_id, lease_token)
                if lease_token else None
            )
            candidate_relative = (
                attempt_keys["candidate_evidence_key"].removeprefix(
                    _job_storage_key(job.job_id) + "/"
                ) if attempt_keys else
                f"audit/ocr-candidates/g{int(job.dispatch_generation)}/p{page_num}.json"
            )
            candidate_evidence_key = _job_storage_key(job.job_id, candidate_relative)
            payload = {
                "schema": "ocr_candidate_evidence_v1",
                "job_id": str(job.job_id),
                "tenant_id": str(job.tenant_id),
                "dispatch_generation": int(job.dispatch_generation),
                "page_num": int(page_num),
                "source_sha256": job.source_hash,
                "source_object_key": str(job.source_object_key or ""),
                "raster_sha256": hashlib.sha256(img_bytes).hexdigest(),
                "candidates": rerun_evidence,
            }
            candidate_bytes = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if len(candidate_bytes) > 8 * 1024 * 1024:
                raise RuntimeError("OCR candidate evidence exceeds bounded artifact size")
            candidate_evidence_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
            if storage_mode == "nfs":
                candidate_evidence_path = os.path.join(job_path, candidate_relative)
                _mkdir_tree_no_follow(
                    os.path.dirname(candidate_evidence_path),
                    getattr(settings, "NFS_ROOT", job_path),
                )
            else:
                candidate_evidence_path = os.path.join(
                    temp_dir, f"{page_num}.candidates.json"
                )
            with open(candidate_evidence_path, "wb") as evidence_file:
                evidence_file.write(candidate_bytes)

        # Upload to backend if S3 mode.  Keep the exact externally visible
        # locators so a cancellation fence that wins during I/O can remove
        # both artifacts before this worker returns stale.
        published_pdf_locator = temp_pdf_path
        published_text_locator = temp_text_path
        published_pdf_delete_url = None
        published_text_delete_url = None
        published_candidate_delete_url = None
        if publish_fence is None and lease_token:
            publish_fence = partial(
                _renew_page_processing_lease,
                job.job_id,
                page_num,
                lease_token,
            )
        if storage_mode == "s3":
            attempted_upload = False
            publication_targets = ()
            try:
                if presigned_output_refresher is not None:
                    presigned_urls = presigned_output_refresher()
                if source_fence is not None:
                    source_fence()
                if publish_fence is not None:
                    publish_fence()
                attempt_keys = (
                    page_attempt_keys(job, page_num, document_id, lease_token)
                    if lease_token else None
                )
                pdf_key = (
                    attempt_keys["page_pdf_key"] if attempt_keys else
                    _job_storage_key(job.job_id, f"temp/{document_id}/{page_num}.pdf")
                )
                text_key = (
                    attempt_keys["page_text_key"] if attempt_keys else
                    _job_storage_key(job.job_id, f"temp/{document_id}/{page_num}.txt")
                )
                if presigned_urls and lease_token:
                    for grant_name, expected_key in (
                        ("page_pdf_key", pdf_key),
                        ("page_text_key", text_key),
                        ("candidate_evidence_key", attempt_keys["candidate_evidence_key"]),
                    ):
                        if presigned_urls.get(grant_name) != expected_key:
                            raise RuntimeError("page output grant locator changed")
                published_pdf_delete_url = (
                    presigned_urls.get("page_pdf_delete") if presigned_urls else None
                )
                published_text_delete_url = (
                    presigned_urls.get("page_text_delete") if presigned_urls else None
                )
                published_candidate_delete_url = (
                    presigned_urls.get("candidate_evidence_delete")
                    if presigned_urls else None
                )
                publication_targets = (
                    (pdf_key, published_pdf_delete_url),
                    (text_key, published_text_delete_url),
                )
                if candidate_evidence_path:
                    publication_targets += (
                        (candidate_evidence_key, published_candidate_delete_url),
                    )
                if presigned_urls and presigned_urls.get("page_pdf_put"):
                    from .presigned_io import upload_presigned

                    if not all((published_pdf_delete_url, published_text_delete_url)):
                        raise RuntimeError("presigned output requires both cleanup URLs")
                    if candidate_evidence_path and not all(
                        (
                            presigned_urls.get("candidate_evidence_put"),
                            published_candidate_delete_url,
                        )
                    ):
                        raise RuntimeError(
                            "presigned output requires candidate evidence grants"
                        )
                    attempted_upload = True
                    if candidate_evidence_path:
                        upload_presigned(
                            candidate_evidence_path,
                            presigned_urls["candidate_evidence_put"],
                            job_id=str(job.job_id),
                            expected_key=candidate_evidence_key,
                        )
                    upload_presigned(
                        temp_pdf_path,
                        presigned_urls["page_pdf_put"],
                        job_id=str(job.job_id),
                        expected_key=pdf_key,
                    )
                    upload_presigned(
                        temp_text_path,
                        presigned_urls["page_text_put"],
                        job_id=str(job.job_id),
                        expected_key=text_key,
                    )
                else:
                    attempted_upload = True
                    if candidate_evidence_path:
                        backend.upload_file(
                            candidate_evidence_path, candidate_evidence_key
                        )
                    backend.upload_file(temp_pdf_path, pdf_key)
                    backend.upload_file(temp_text_path, text_key)
                # Use S3 keys as the temp paths for DB record
                temp_pdf_path = pdf_key
                temp_text_path = text_key
                published_pdf_locator = pdf_key
                published_text_locator = text_key
                if candidate_evidence_path:
                    candidate_evidence_path = candidate_evidence_key
            except Exception as exc:
                if attempted_upload:
                    # A transport exception can arrive after remote PUT has
                    # committed. Remove both exact keys, including the one
                    # whose acknowledgment was lost, before failing the page.
                    _discard_published_artifacts(job, backend, publication_targets)
                logger.error(
                    "Failed to upload page %d artifacts to S3: %s", page_num, exc
                )
                raise

        # The purge fence and this post-publication check form a handshake:
        # if the fence committed first, this worker removes its late bytes;
        # if it commits after this check, the purge's storage sweep removes
        # them.  A database/check failure also fails closed by deleting them.
        if publish_fence is not None or source_fence is not None:

            def checked_publish_fence():
                if source_fence is not None:
                    source_fence()
                if publish_fence is not None:
                    publish_fence()

            _renew_or_discard_published_artifacts(
                checked_publish_fence,
                job,
                backend,
                (
                    (published_pdf_locator, published_pdf_delete_url),
                    (published_text_locator, published_text_delete_url),
                    *(
                        ((candidate_evidence_path, published_candidate_delete_url),)
                        if candidate_evidence_path
                        else ()
                    ),
                ),
            )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Determine status
        if overlay_issues:
            status = "failed"
        elif ocr_method == "ImageOnly":
            model_language = ""
            actual_runtime_model = ""
            status = "image_only"
        elif ocr_method == "Tesseract":
            status = "fallback"
        else:
            status = "ok"

        if ocr_method == "PaddleOCR":
            _record_engine_provenance(
                job,
                document_id,
                hostname,
                page_num,
                catalog.PADDLE_CTC,
                engine_policy,
            )
        elif ocr_method == "Tesseract":
            _record_engine_provenance(
                job,
                document_id,
                hostname,
                page_num,
                catalog.TESSERACT,
                engine_policy,
            )

        # Record page result in DB.  Fan-out workers use a conditional update
        # so a stale delivery cannot publish over a newer page owner.  The
        # single-worker document path retains the legacy upsert behavior.
        page_defaults = {
            "document_id": document_id,
            "ocr_method": ocr_method,
            "ocr_language": model_language,
            "ocr_confidence": confidence,
            "needs_review": bool(
                route.needs_review
                or coverage_assessment.needs_review
                or overlay_issues
                or ocr_method == "ImageOnly"
            ),
            "ocr_provenance": {
                "assessment_version": "ocr_rescue_v1",
                "assessment_state": "assessed",
                "detected_language": route.detected_label,
                "model_language": model_language,
                "runtime_model": actual_runtime_model,
                "attempted_tesseract_model": attempted_tesseract_model,
                "route_reason": route.reason,
                "script_evidence": list(script_evidence[:2]),
                "reading_direction": {
                    "orientation": reading_direction.get("orientation"),
                    "reason": reading_direction.get("reason", ""),
                    "score": reading_direction.get("score", 0.0),
                },
                "page_geometry": {
                    "width_pt": geometry.width_pt,
                    "height_pt": geometry.height_pt,
                    "source_kind": geometry.source_kind,
                    "source_rotation": geometry.source_rotation,
                    "source_uncertain": geometry_uncertain,
                },
                "candidate_disposition": candidate_disposition,
                "image_only_reason": (
                    "no_recognized_text_after_primary_and_fallback"
                    if ocr_method == "ImageOnly" and not overlay_issues
                    else "ocr_or_placement_failure"
                    if ocr_method == "ImageOnly"
                    else ""
                ),
                "candidate_evidence": (
                    {
                        "locator": candidate_evidence_path,
                        "sha256": candidate_evidence_sha256,
                        "selected_text_sha256": hashlib.sha256(
                            page_text.encode("utf-8")
                        ).hexdigest(),
                    }
                    if candidate_evidence_path
                    else None
                ),
                "visual_coverage": {
                    "status": coverage_assessment.status,
                    "reason": coverage_assessment.reason,
                    "coverage": coverage_assessment.coverage,
                    "region_coverage": coverage_assessment.region_coverage,
                    "component_coverage": coverage_assessment.component_coverage,
                    "character_support": coverage_assessment.character_support,
                    "missing_region_count": len(coverage_assessment.missing_regions),
                    "needs_retry": coverage_assessment.needs_retry,
                },
                "geometry_issues": overlay_issues[:8],
                "geometry_issue_count": len(overlay_issues),
                "text_sha256": hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
            },
            "text_length": len(page_text),
            "status": status,
            "worker_hostname": hostname,
            "processing_time_ms": elapsed_ms,
            "temp_pdf_path": temp_pdf_path,
        }
        if lease_token:
            page_defaults["ocr_provenance"]["temp_text_path"] = temp_text_path
        if source_fence is not None:
            if storage_mode == "s3":
                _renew_or_discard_published_artifacts(
                    source_fence,
                    job,
                    backend,
                    (
                        (published_pdf_locator, published_pdf_delete_url),
                        (published_text_locator, published_text_delete_url),
                        *(
                            ((candidate_evidence_path, published_candidate_delete_url),)
                            if candidate_evidence_path
                            else ()
                        ),
                    ),
                )
            else:
                source_fence()
        if lease_token:
            page_defaults["celery_task_id"] = lease_token

            def commit_page_result():
                _commit_page_result_if_owned(
                    job, page_num, lease_token, page_defaults
                )

            _renew_or_discard_published_artifacts(
                commit_page_result, job, backend,
                (
                    (published_pdf_locator, published_pdf_delete_url),
                    (published_text_locator, published_text_delete_url),
                    *(
                        ((candidate_evidence_path, published_candidate_delete_url),)
                        if candidate_evidence_path else ()
                    ),
                ),
            )
        else:
            PageResult.objects.update_or_create(
                job=job,
                page_num=page_num,
                defaults=page_defaults,
            )

        # Record custody event
        custody_event_type = {
            "PaddleOCR": "ocr_primary",
            "Tesseract": "ocr_fallback",
            "ImageOnly": "ocr_image_only",
        }.get(ocr_method, "ocr_primary")

        _record_custody_event(
            job,
            document_id,
            custody_event_type,
            data={
                "page_num": page_num,
                "method": ocr_method,
                "confidence": round(confidence, 4),
                "text_length": len(page_text),
                "detected_language": route.detected_label,
                "model_language": model_language,
                "runtime_model": actual_runtime_model,
                "attempted_tesseract_model": attempted_tesseract_model,
                "language_route_reason": route.reason,
                "language_needs_review": route.needs_review,
                "needs_review": bool(
                    route.needs_review or overlay_issues or ocr_method == "ImageOnly"
                ),
                "geometry_issues": overlay_issues[:8],
                "geometry_issue_count": len(overlay_issues),
                "candidate_disposition": candidate_disposition,
            },
            worker_hostname=hostname,
        )

        return {
            "page_num": page_num,
            "method": ocr_method,
            "confidence": confidence,
            "text_length": len(page_text),
            "status": status,
            "elapsed_ms": elapsed_ms,
        }
    finally:
        if storage_mode == "s3" and owns_temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def _record_page_failed(
    job, page_num, document_id, hostname, error_msg, lease_token=None
):
    """Record a failed page result."""
    page_defaults = {
        "document_id": document_id,
        "status": "failed",
        "worker_hostname": hostname,
        "needs_review": True,
        "ocr_provenance": {
            "assessment_version": "ocr_rescue_v1",
            "assessment_state": "failed",
            "failure_reason": str(error_msg)[:256],
        },
    }
    if lease_token:
        page_defaults["celery_task_id"] = lease_token
        _commit_page_result_if_owned(job, page_num, lease_token, page_defaults)
    else:
        PageResult.objects.update_or_create(
            job=job,
            page_num=page_num,
            defaults=page_defaults,
        )
    _record_custody_event(
        job,
        document_id,
        "processing_failed",
        data={"page_num": page_num, "error": error_msg},
        worker_hostname=hostname,
    )
    return {"page_num": page_num, "status": "failed", "error": error_msg}


def _gpu_available():
    """Check if GPU is available for PaddleOCR."""
    try:
        import paddle

        return (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CPU General Queue Tasks
# ---------------------------------------------------------------------------

_COMPRESSION_DEV_LOCKS = {}
_COMPRESSION_DEV_LOCKS_GUARD = threading.Lock()


def _compression_advisory_key(job_id):
    """Return a deterministic signed PostgreSQL bigint key for a job UUID."""
    digest = hashlib.sha256(str(job_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


@contextmanager
def _compression_execution_lock(job_id):
    """Serialize compression without holding a Job row lock during I/O.

    Production PostgreSQL uses a session advisory lock.  SQLite and other
    non-PostgreSQL backends use a process-local keyed mutex for dev/tests only;
    that fallback does not provide cross-process exclusion.
    """
    if connection.vendor == "postgresql":
        key = _compression_advisory_key(job_id)
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_lock(%s)", [key])
        success = False
        try:
            yield
            success = True
        finally:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(%s)", [key])
                    row = cursor.fetchone()
                if not row or row[0] is not True:
                    raise RuntimeError("PostgreSQL compression advisory unlock failed")
            except Exception as unlock_exc:
                # Never return a possibly locked pooled session to the pool.
                connection.close()
                # Do not shadow an exception from the task body with unlock noise.
                if success:
                    raise RuntimeError(
                        "PostgreSQL compression advisory unlock failed"
                    ) from unlock_exc
        return

    lock_key = str(job_id)
    with _COMPRESSION_DEV_LOCKS_GUARD:
        entry = _COMPRESSION_DEV_LOCKS.get(lock_key)
        if entry is None:
            entry = {"lock": threading.Lock(), "users": 0}
            _COMPRESSION_DEV_LOCKS[lock_key] = entry
        entry["users"] += 1
    entry["lock"].acquire()
    try:
        yield
    finally:
        entry["lock"].release()
        with _COMPRESSION_DEV_LOCKS_GUARD:
            entry["users"] -= 1
            if entry["users"] == 0:
                _COMPRESSION_DEV_LOCKS.pop(lock_key, None)


def _compression_result_from_event(job, assembly_event, compression_event):
    data = compression_event.data if isinstance(compression_event.data, dict) else {}
    expected_status = {
        ("compression_complete", "APPLIED"): "compressed",
        ("compression_skipped", "SKIPPED_VISUAL_BUDGET"): "skipped_visual_budget",
        ("compression_skipped", "SKIPPED_CANDIDATE_REJECTED"): (
            "skipped_candidate_rejected"
        ),
    }.get((compression_event.event_type, data.get("disposition")))
    if (
        expected_status is None
        or compression_event.id <= assembly_event.id
        or data.get("causal_event_id") != assembly_event.id
        or not _is_sha256_hex(data.get("pdf_sha256"))
    ):
        raise RuntimeError("existing compression custody event is invalid")
    return {
        "stage": "compression",
        "status": expected_status,
        "job_id": str(job.job_id),
        "disposition": data["disposition"],
        "reason": data.get("reason", ""),
        "pdf_sha256": data["pdf_sha256"],
        "custody_event_id": compression_event.id,
        "idempotent_redelivery": True,
    }


def _discard_postprocessing_artifact(
    job,
    backend,
    locator,
    *,
    presigned_delete_url=None,
):
    """Remove bytes published after a tenant/job cancellation fence."""
    if not locator:
        return
    if backend.backend_name == "nfs":
        job_root = os.path.realpath(_nfs_job_path_for_job(job))
        candidate = os.path.realpath(str(locator))
        try:
            if os.path.commonpath([job_root, candidate]) != job_root:
                raise RuntimeError("stale postprocessing artifact escapes job root")
        except ValueError as exc:
            raise RuntimeError(
                "stale postprocessing artifact escapes job root"
            ) from exc
        if candidate == job_root:
            raise RuntimeError("refusing to discard the complete job root")
        if os.path.lexists(candidate) and not remove_path_no_follow(candidate):
            raise RuntimeError("stale postprocessing artifact remains on NFS")
        return

    if presigned_delete_url:
        from .presigned_io import delete_presigned

        delete_presigned(
            presigned_delete_url,
            job_id=str(job.job_id),
            expected_key=str(locator),
        )
    else:
        backend.delete(str(locator))
    if isinstance(backend, CachedS3Backend):
        backend.invalidate(str(locator))


def _discard_published_artifacts(job, backend, artifacts) -> None:
    """Best-effort all exact artifacts, then surface any cleanup failure."""
    failures = []
    for locator, delete_url in artifacts:
        try:
            _discard_postprocessing_artifact(
                job,
                backend,
                locator,
                presigned_delete_url=delete_url,
            )
        except Exception as exc:
            failures.append(exc)
            logger.exception(
                "Could not discard stale artifact %s for job %s",
                locator,
                job.job_id,
            )
    if failures:
        raise RuntimeError(
            f"could not discard {len(failures)} stale artifact(s) for job {job.job_id}"
        ) from failures[0]


def _renew_or_discard_published_artifacts(renew, job, backend, artifacts) -> None:
    """Run the durable ownership fence after I/O or remove every publication."""
    try:
        outcome = renew()
    except Exception:
        _discard_published_artifacts(job, backend, artifacts)
        raise
    if outcome is False:
        # A current-generation page may already have a committed winner from
        # an earlier native execution nonce. This attempt's distinct keys are
        # unused and must not accumulate or disturb the winner.
        _discard_published_artifacts(job, backend, artifacts)
    return outcome


def _discard_published_artifacts_if_stale(job, backend, artifacts) -> bool:
    """Reconcile a set of external writes against one durable job-state read."""
    try:
        current = _postprocessing_publish_is_current(job.job_id)
    except Exception:
        _discard_published_artifacts(job, backend, artifacts)
        raise
    if current:
        return False
    _discard_published_artifacts(job, backend, artifacts)
    return True


def _postprocessing_publish_is_current(job_id) -> bool:
    """Return true only while a postprocessing artifact may be committed."""
    return (
        Job.objects.filter(
            job_id=job_id,
        )
        .exclude(status__in=_TERMINAL_JOB_STATUSES)
        .exists()
    )


def _discard_if_postprocessing_stale(
    job,
    backend,
    locator,
    *,
    presigned_delete_url=None,
) -> bool:
    """Reconcile one external publish against the durable job fence."""
    return _discard_published_artifacts_if_stale(
        job,
        backend,
        ((locator, presigned_delete_url),),
    )


@shared_task(
    bind=True,
    name="jobs.tasks.compress_pdf",
    queue="cpu_general",
    acks_late=True,
    reject_on_worker_lost=True,
)
@_tenant_bound_task
def compress_pdf(self, job_id, presigned_capability_id=None, tenant_id=None):
    """Idempotently optimize the assembled PDF under a per-job execution lock."""
    with _compression_execution_lock(job_id):
        with transaction.atomic():
            try:
                job = Job.objects.select_for_update().get(job_id=job_id)
            except Job.DoesNotExist:
                return {"status": "error", "message": f"Job {job_id} not found"}
            if job.status in _TERMINAL_JOB_STATUSES:
                return {
                    "stage": "compression",
                    "status": "already_finalized",
                    "terminal_status": str(job.status),
                    "job_id": str(job_id),
                }
            if job.status != Job.Status.ASSEMBLING:
                raise RuntimeError(
                    f"compression requires assembling state; observed={job.status}"
                )
            document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
            assembly_event = (
                CustodyEvent.objects.filter(
                    job=job,
                    document_id=document_id,
                    event_type="assembly_complete",
                )
                .order_by("-id")
                .first()
            )
            if assembly_event is None:
                raise RuntimeError("compression has no causal assembly event")
            existing = (
                CustodyEvent.objects.filter(
                    job=job,
                    document_id=document_id,
                    event_type__in=("compression_complete", "compression_skipped"),
                )
                .order_by("-id")
                .first()
            )
            if existing is not None:
                return _compression_result_from_event(job, assembly_event, existing)
            assembly_event_id = assembly_event.id

        presigned_urls = None
        if _get_backend_for_job(job).backend_name == "s3" and is_presigned_mode():
            if not presigned_capability_id:
                raise RuntimeError("compression S3 worker requires a capability")
            presigned_urls = _request_fresh_presigned_urls(
                job,
                presigned_capability_id,
                "compress_pdf",
                "input",
            )

        return _compress_pdf_once(
            self,
            job,
            assembly_event_id,
            presigned_urls=presigned_urls,
            presigned_capability_id=presigned_capability_id,
        )


def _compress_pdf_once(
    self,
    job,
    assembly_event_id,
    presigned_urls=None,
    presigned_capability_id=None,
):
    """Perform external compression, then reconcile in one short transaction.

    When an active generation pointer exists, compression stages a *new* PDF
    generation and switches the pointer only after the new bytes are durable.
    Terminal/canceled jobs keep the prior pointer. Jobs without a pointer
    (legacy) still optimize the fixed EXPORT path in place.
    """
    job_id = job.job_id

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    hostname = socket.gethostname()
    job_path = _job_artifact_root(job, storage_mode)
    pointer_mode = ACTIVE_OUTPUT_POINTER_KEY in (job.settings_json or {})

    work_dir = ""
    new_pdf_key = ""
    published_delete_url = None
    try:
        active = _resolve_job_output_artifacts(job, backend)
        source_pdf_locator = active.pdf.locator

        work_dir = tempfile.mkdtemp(prefix=f"compress_{job_id}_")
        pdf_path = os.path.join(work_dir, f"{base_name}.working.pdf")

        if storage_mode == "nfs":
            if not os.path.isfile(source_pdf_locator):
                raise FileNotFoundError(
                    f"assembled PDF is missing: {source_pdf_locator}"
                )
            if pointer_mode:
                shutil.copy2(source_pdf_locator, pdf_path)
            else:
                # Legacy in-place optimize on the assembled EXPORT path.
                pdf_path = source_pdf_locator
        else:
            if presigned_urls and presigned_urls.get("pdf_get"):
                from .presigned_io import download_presigned

                download_presigned(
                    presigned_urls["pdf_get"],
                    pdf_path,
                    job_id=str(job_id),
                    expected_key=source_pdf_locator,
                )
            else:
                if not backend.exists(source_pdf_locator):
                    raise FileNotFoundError(
                        f"assembled PDF object is missing: {source_pdf_locator}"
                    )
                backend.download_file(source_pdf_locator, pdf_path)

        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(
                f"assembled PDF is missing after storage retrieval: {pdf_path}"
            )

        original_pdf_sha256 = _compute_file_hash(pdf_path)
        if active.pdf.sha256 and not hmac.compare_digest(
            original_pdf_sha256, active.pdf.sha256
        ):
            raise RuntimeError("assembled PDF hash differs from active generation")

        document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]

        from optimize_pdfs import OptimizationDisposition, optimize_pdf_detailed

        outcome = optimize_pdf_detailed(pdf_path, quality="/prepress")
        disposition = outcome.disposition

        if disposition is OptimizationDisposition.APPLIED:
            custody_event = "compression_complete"
            status = "compressed"
            logger.info("Compressed PDF for job %s", job_id)
        elif disposition in {
            OptimizationDisposition.SKIPPED_VISUAL_BUDGET,
            OptimizationDisposition.SKIPPED_CANDIDATE_REJECTED,
        }:
            custody_event = "compression_skipped"
            status = (
                "skipped_visual_budget"
                if disposition is OptimizationDisposition.SKIPPED_VISUAL_BUDGET
                else "skipped_candidate_rejected"
            )
            logger.info(
                "Skipped PDF compression for job %s (%s): %s",
                job_id,
                disposition.value,
                outcome.reason,
            )
        else:
            # FAILED_PRESERVED / FAILED_INDETERMINATE — fail closed; never
            # log/upload/emit compression_complete after a failed result.
            raise RuntimeError(
                f"PDF optimization failed ({disposition.value}): {outcome.reason}"
            )

        # Bind the exact bytes before upload/event creation.
        pdf_sha256 = _compute_file_hash(pdf_path)
        if disposition is not OptimizationDisposition.APPLIED and not hmac.compare_digest(
            pdf_sha256, original_pdf_sha256
        ):
            raise RuntimeError("skipped PDF optimization changed assembled bytes")
        compression_generation_id = None
        published_pdf_path = source_pdf_locator
        output_presigned_urls = None
        if (
            disposition is OptimizationDisposition.APPLIED
            and storage_mode == "s3"
            and is_presigned_mode()
        ):
            output_presigned_urls = _request_fresh_presigned_urls(
                job,
                presigned_capability_id,
                "compress_pdf",
                "output",
            )

        if pointer_mode and disposition is OptimizationDisposition.APPLIED:
            compression_generation_id = make_generation_id(
                f"{job_id}:{assembly_event_id}:compress"
            )
            new_pdf_location, _ = generation_locations(
                job_id=job_id,
                source_file=job.source_file,
                backend=storage_mode,
                generation_id=compression_generation_id,
                job_root=job_path,
            )
            if storage_mode == "nfs":
                _mkdir_tree_no_follow(
                    os.path.dirname(new_pdf_location.locator),
                    getattr(settings, "NFS_ROOT", job_path),
                )
                staged = _new_assembly_staging_path(
                    os.path.dirname(new_pdf_location.locator),
                    base_name,
                    ".compressing.pdf",
                )
                shutil.copy2(pdf_path, staged)
                _durably_commit_assembly_file(staged, new_pdf_location.locator)
                published_pdf_path = new_pdf_location.locator
            else:
                new_pdf_key = new_pdf_location.locator
                if output_presigned_urls and output_presigned_urls.get(
                    "generation_pdf_put"
                ):
                    from .presigned_io import upload_presigned

                    upload_presigned(
                        pdf_path,
                        output_presigned_urls["generation_pdf_put"],
                        job_id=str(job_id),
                        expected_key=new_pdf_key,
                    )
                    published_delete_url = output_presigned_urls.get(
                        "generation_pdf_delete"
                    )
                else:
                    backend.upload_file(pdf_path, new_pdf_key)
                if isinstance(backend, CachedS3Backend):
                    for cache_key in (source_pdf_locator, new_pdf_key):
                        backend.invalidate(cache_key)
                published_pdf_path = new_pdf_key
        elif storage_mode == "s3" and disposition is OptimizationDisposition.APPLIED:
            # Legacy fixed-key upload of optimized bytes.
            pdf_key = source_pdf_locator
            if output_presigned_urls and output_presigned_urls.get("pdf_put"):
                from .presigned_io import upload_presigned

                upload_presigned(
                    pdf_path,
                    output_presigned_urls["pdf_put"],
                    job_id=str(job_id),
                    expected_key=pdf_key,
                )
                published_delete_url = output_presigned_urls.get("pdf_delete")
            else:
                backend.upload_file(pdf_path, pdf_key)
            if isinstance(backend, CachedS3Backend):
                backend.invalidate(pdf_key)
            published_pdf_path = pdf_key

        if (
            disposition is OptimizationDisposition.APPLIED
            and _discard_if_postprocessing_stale(
                job,
                backend,
                published_pdf_path,
                presigned_delete_url=published_delete_url,
            )
        ):
            return {
                "stage": "compression",
                "status": "already_finalized",
                "terminal_status": "cancelled_or_deleted",
                "job_id": str(job_id),
            }

        with transaction.atomic():
            locked_job = Job.objects.select_for_update().get(job_id=job_id)
            if locked_job.status in _TERMINAL_JOB_STATUSES:
                return {
                    "stage": "compression",
                    "status": "already_finalized",
                    "terminal_status": str(locked_job.status),
                    "job_id": str(job_id),
                }
            if locked_job.status != Job.Status.ASSEMBLING:
                raise RuntimeError(
                    "compression reconciliation requires assembling state; "
                    f"observed={locked_job.status}"
                )
            current_assembly = (
                CustodyEvent.objects.filter(
                    job=locked_job,
                    document_id=document_id,
                    event_type="assembly_complete",
                )
                .order_by("-id")
                .first()
            )
            if current_assembly is None or current_assembly.id != assembly_event_id:
                raise RuntimeError("compression assembly ownership was superseded")
            existing = (
                CustodyEvent.objects.filter(
                    job=locked_job,
                    document_id=document_id,
                    event_type__in=("compression_complete", "compression_skipped"),
                )
                .order_by("-id")
                .first()
            )
            if existing is not None:
                return _compression_result_from_event(
                    locked_job,
                    current_assembly,
                    existing,
                )

            if pointer_mode and compression_generation_id:
                settings_json = dict(locked_job.settings_json or {})
                prior_pointer = settings_json.get(ACTIVE_OUTPUT_POINTER_KEY)
                if prior_pointer is None:
                    raise RuntimeError(
                        "compression requires an active assembly artifact pointer"
                    )
                settings_json[ACTIVE_OUTPUT_POINTER_KEY] = replace_pointer_pdf(
                    prior_pointer,
                    job_id=job_id,
                    source_file=locked_job.source_file,
                    revision=compression_generation_id,
                    pdf_generation_id=compression_generation_id,
                    pdf_sha256=pdf_sha256,
                )
                locked_job.settings_json = settings_json
                locked_job.save(update_fields=["settings_json"])

            custody_data = {
                "causal_event_id": assembly_event_id,
                "pdf_path": published_pdf_path,
                "disposition": disposition.value,
                "reason": outcome.reason,
                "pdf_sha256": pdf_sha256,
                "metrics": {
                    str(key): float(value)
                    for key, value in outcome.metrics.items()
                    if isinstance(value, (int, float))
                },
            }
            if compression_generation_id:
                custody_data["artifact_generation_id"] = compression_generation_id
            custody_row = _record_custody_event(
                locked_job,
                document_id,
                custody_event,
                data=custody_data,
                worker_hostname=hostname,
            )

        Worker.objects.filter(hostname=hostname).update(
            tasks_completed=db_models.F("tasks_completed") + 1,
        )
        result = {
            "stage": "compression",
            "status": status,
            "job_id": str(job_id),
            "disposition": disposition.value,
            "reason": outcome.reason,
            "pdf_sha256": pdf_sha256,
            "custody_event_id": custody_row.id,
        }
        if compression_generation_id:
            result["artifact_generation_id"] = compression_generation_id
        return result
    except Exception as exc:
        # Optimizer / upload / custody failures must fail the Celery task so
        # the header chord errback can run. Never return success-shaped error
        # dicts from this block (that would finalize the chord as success).
        logger.error("PDF compression failed for job %s: %s", job_id, exc)
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        raise
    finally:
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


@shared_task(
    bind=True,
    name="jobs.tasks.extract_entities",
    queue="cpu_general",
    acks_late=True,
    reject_on_worker_lost=True,
)
@_tenant_bound_task
def extract_entities(self, job_id, presigned_capability_id=None, tenant_id=None):
    """Run Named Entity Recognition on extracted text.

    Args:
        presigned_capability_id: Optional job-bound capability UUID.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist as exc:
        raise RuntimeError(f"Job {job_id} not found for entity extraction") from exc
    _pin_rls_for_job(job)
    if job.status in _TERMINAL_JOB_STATUSES:
        return {
            "stage": "entities",
            "status": "already_finalized",
            "job_id": str(job_id),
        }
    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name
    presigned_urls = None
    if storage_mode == "s3" and is_presigned_mode():
        if not presigned_capability_id:
            raise RuntimeError("NER S3 worker requires a capability")
        presigned_urls = _request_fresh_presigned_urls(
            job,
            presigned_capability_id,
            "extract_entities",
            "input",
        )
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    hostname = socket.gethostname()
    active = _resolve_job_output_artifacts(job, backend)

    # Setup paths based on storage backend
    work_dir = ""
    published_ner_locator = ""
    published_delete_url = None
    if storage_mode == "nfs":
        job_path = _nfs_job_path_for_job(job)
        text_path = active.text.locator
        ner_dir = os.path.join(job_path, "output", "EXPORT", "NER")
        _mkdir_tree_no_follow(ner_dir, getattr(settings, "NFS_ROOT", job_path))
        ner_path = os.path.join(ner_dir, f"{base_name}.ner.json")
        published_ner_locator = ner_path
    else:
        work_dir = tempfile.mkdtemp(prefix=f"ner_{job_id}_")
        text_path = os.path.join(work_dir, f"{base_name}.txt")
        ner_path = os.path.join(work_dir, f"{base_name}.ner.json")
        text_key = active.text.locator
        try:
            if presigned_urls and presigned_urls.get("text_get"):
                from .presigned_io import download_presigned

                download_presigned(
                    presigned_urls["text_get"],
                    text_path,
                    job_id=str(job_id),
                    expected_key=text_key,
                )
            else:
                if not backend.exists(text_key):
                    logger.warning("No text file for NER for job %s", job_id)
                    if os.path.isdir(work_dir):
                        shutil.rmtree(work_dir, ignore_errors=True)
                    raise FileNotFoundError(
                        f"requested NER text object is missing: {text_key}"
                    )
                backend.download_file(text_key, text_path)
        except Exception as exc:
            logger.error("S3 text download failed for job %s: %s", job_id, exc)
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            Worker.objects.filter(hostname=hostname).update(
                tasks_failed=db_models.F("tasks_failed") + 1,
            )
            raise RuntimeError(f"S3 text download failed: {exc}") from exc

    if not os.path.isfile(text_path):
        logger.warning("No text file for NER for job %s", job_id)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise FileNotFoundError(f"requested NER text file is missing: {text_path}")

    try:
        text = read_text_bounded(text_path)
    except (OSError, UnicodeError, ValueError) as exc:
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"could not read requested NER text: {exc}") from exc

    try:
        from ner import extract_entities as run_ner

        # Document-level text: page_num=0 (API requires int page_num).
        entities = run_ner(text, page_num=0)
        ner_status = "extracted"
        ner_engine = "unknown"
        try:
            import ner as ner_mod

            spacy_flag = getattr(ner_mod, "_SPACY_AVAILABLE", None)
            # The canonical module exposes an explicit flag.  Test doubles and
            # compatible third-party NER providers may not; absence of the flag
            # is not evidence that their returned entities are regex-only.
            if spacy_flag is False:
                ner_engine = "regex_only"
                ner_status = "degraded_regex"
            elif spacy_flag is True:
                ner_engine = "spacy_or_regex"
            else:
                ner_engine = "unknown"
        except Exception:
            ner_engine = "unknown"
    except ImportError as exc:
        logger.warning("NER module not available: %s", exc)
        document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
        _record_custody_event(
            job,
            document_id,
            "NER_UNAVAILABLE",
            data={"reason": "import_error", "error": str(exc)},
            worker_hostname=hostname,
        )
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        # Missing NER is a chord failure, not a successful empty extraction.
        # Returning an "unavailable" sidecar would let downstream finalization
        # treat a missing forensic stage as completed work.
        raise RuntimeError(f"NER module is unavailable: {exc}") from exc
    except TypeError as exc:
        # Call-signature drift (e.g. missing page_num) must not silent-empty.
        logger.error("NER call signature failed for job %s: %s", job_id, exc)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"NER call contract failed: {exc}") from exc
    except Exception as exc:
        logger.error("NER failed for job %s: %s", job_id, exc)
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"requested NER failed: {exc}") from exc

    # Honest sidecar: never pretend full spaCy success when engine unavailable.
    if ner_status == "unavailable":
        ner_payload = {
            "status": "unavailable",
            "engine": ner_engine,
            "entities": entities if isinstance(entities, list) else [],
            "entity_count": 0,
        }
    else:
        # Normalize Entity dataclasses / dicts to JSON-safe list
        if (
            entities
            and hasattr(entities[0], "__dict__")
            and not isinstance(entities[0], dict)
        ):
            entity_list = [
                e if isinstance(e, dict) else getattr(e, "__dict__", {"text": str(e)})
                for e in entities
            ]
        else:
            entity_list = entities if isinstance(entities, list) else []
        ner_payload = {
            "status": ner_status,
            "engine": ner_engine,
            "entities": entity_list,
            "entity_count": len(entity_list),
        }
        if ner_status == "degraded_regex":
            document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
            _record_custody_event(
                job,
                document_id,
                "NER_UNAVAILABLE",
                data={
                    "reason": "spacy_unavailable_regex_fallback",
                    "entity_count": len(entity_list),
                },
                worker_hostname=hostname,
            )

    # Write NER results
    try:
        with open(ner_path, "w", encoding="utf-8") as f:
            json.dump(ner_payload, f, indent=2, default=str)

        # Upload to S3 if needed
        if storage_mode == "s3":
            output_presigned_urls = None
            if is_presigned_mode():
                output_presigned_urls = _request_fresh_presigned_urls(
                    job,
                    presigned_capability_id,
                    "extract_entities",
                    "output",
                )
            if output_presigned_urls:
                from .presigned_io import upload_presigned

                upload_presigned(
                    ner_path,
                    output_presigned_urls["ner_put"],
                    job_id=str(job_id),
                    expected_key=_job_storage_key(
                        job_id, f"output/EXPORT/NER/{base_name}.ner.json"
                    ),
                )
                published_delete_url = output_presigned_urls.get("ner_delete")
            else:
                ner_key = _job_storage_key(
                    job_id, f"output/EXPORT/NER/{base_name}.ner.json"
                )
                backend.upload_file(ner_path, ner_key)
            published_ner_locator = _job_storage_key(
                job_id, f"output/EXPORT/NER/{base_name}.ner.json"
            )
        if _discard_if_postprocessing_stale(
            job,
            backend,
            published_ner_locator,
            presigned_delete_url=published_delete_url,
        ):
            return {
                "stage": "entities",
                "status": "already_finalized",
                "job_id": str(job_id),
            }
    except Exception as exc:
        logger.error("NER result write failed for job %s: %s", job_id, exc)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"requested NER result publication failed: {exc}") from exc
    finally:
        # Cleanup temp directory for S3 mode
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )

    return {
        "stage": "entities",
        "status": ner_payload.get("status", "extracted"),
        "job_id": str(job_id),
        "entity_count": int(ner_payload.get("entity_count") or 0),
        "engine": ner_payload.get("engine"),
    }


@shared_task(
    bind=True,
    name="jobs.tasks.extract_structured_data",
    queue="nlp_general",
    acks_late=True,
    reject_on_worker_lost=True,
)
@_tenant_bound_task
def extract_structured_data(self, job_id, presigned_capability_id=None, tenant_id=None):
    """Run PaddleNLP-based structured data extraction on the OCR output text.

    This fulfills the Phase 9 worker isolation design, decoupling heavy NLP
    extraction from the primary OCR workers so they can be versioned independently.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist as exc:
        raise RuntimeError(f"Job {job_id} not found for structured extraction") from exc
    _pin_rls_for_job(job)
    if job.status in _TERMINAL_JOB_STATUSES:
        return {
            "stage": "structured_data",
            "status": "already_finalized",
            "job_id": str(job_id),
        }
    if not job.settings_json.get("enable_docintel", False):
        # If extraction wasn't requested, just skip it.
        return {
            "stage": "structured_data",
            "status": "skipped",
            "reason": "not_requested",
            "job_id": str(job_id),
        }

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name
    presigned_urls = None
    if storage_mode == "s3" and is_presigned_mode():
        if not presigned_capability_id:
            raise RuntimeError("structured extraction S3 worker requires a capability")
        presigned_urls = _request_fresh_presigned_urls(
            job,
            presigned_capability_id,
            "extract_structured_data",
            "input",
        )
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    hostname = socket.gethostname()
    active = _resolve_job_output_artifacts(job, backend)

    # Setup paths based on storage backend
    work_dir = ""
    published_ext_locator = ""
    published_delete_url = None
    if storage_mode == "nfs":
        job_path = _nfs_job_path_for_job(job)
        text_path = active.text.locator
        ext_dir = os.path.join(job_path, "output", "EXPORT", "EXTRACTION")
        _mkdir_tree_no_follow(ext_dir, getattr(settings, "NFS_ROOT", job_path))
        ext_path = os.path.join(ext_dir, f"{base_name}.extraction.json")
        published_ext_locator = ext_path
    else:
        work_dir = tempfile.mkdtemp(prefix=f"ext_{job_id}_")
        text_path = os.path.join(work_dir, f"{base_name}.txt")
        ext_path = os.path.join(work_dir, f"{base_name}.extraction.json")
        text_key = active.text.locator
        try:
            if presigned_urls and presigned_urls.get("text_get"):
                from .presigned_io import download_presigned

                download_presigned(
                    presigned_urls["text_get"],
                    text_path,
                    job_id=str(job_id),
                    expected_key=text_key,
                )
            else:
                if not backend.exists(text_key):
                    logger.warning(
                        "No text file for structured extraction for job %s", job_id
                    )
                    if os.path.isdir(work_dir):
                        shutil.rmtree(work_dir, ignore_errors=True)
                    raise FileNotFoundError(
                        f"requested structured-extraction text object is missing: {text_key}"
                    )
                backend.download_file(text_key, text_path)
        except Exception as exc:
            logger.error("S3 text download failed for job %s: %s", job_id, exc)
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            Worker.objects.filter(hostname=hostname).update(
                tasks_failed=db_models.F("tasks_failed") + 1,
            )
            raise RuntimeError(
                f"S3 structured-extraction download failed: {exc}"
            ) from exc

    if not os.path.isfile(text_path):
        logger.warning("No text file for structured extraction for job %s", job_id)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise FileNotFoundError(
            f"requested structured-extraction text file is missing: {text_path}"
        )

    try:
        text = read_text_bounded(text_path)
    except (OSError, UnicodeError, ValueError) as exc:
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(
            f"could not read requested structured-extraction text: {exc}"
        ) from exc

    try:
        from extraction import (
            DocumentExtraction,
            extract_page_fields,
            finalize_extraction,
        )

        doc_ext = DocumentExtraction(
            document_id=str(job_id)[:16], source_file=base_name
        )
        # For simplicity, treat the whole document as page 1 for extraction if we only have the merged text.
        # Alternatively, we could split by double-newline if pages were separated.
        page_texts = text.split("\n\n")
        for i, page_text in enumerate(page_texts, start=1):
            if page_text.strip():
                page_ext = extract_page_fields(page_text, i, use_uie=True)
                doc_ext.pages.append(page_ext)

        finalize_extraction(doc_ext)
        import json

        with open(ext_path, "w", encoding="utf-8") as f:
            json.dump(doc_ext.to_dict(), f, indent=2)

    except ImportError as exc:
        logger.warning("Extraction module not available")
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(
            "requested structured-extraction module is unavailable"
        ) from exc
    except Exception as exc:
        logger.error("Structured extraction failed for job %s: %s", job_id, exc)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        raise RuntimeError(f"requested structured extraction failed: {exc}") from exc

    # Upload to S3 if needed
    if storage_mode == "s3":
        try:
            ext_key = _job_storage_key(
                job_id, f"output/EXPORT/EXTRACTION/{base_name}.extraction.json"
            )
            output_presigned_urls = None
            if is_presigned_mode():
                output_presigned_urls = _request_fresh_presigned_urls(
                    job,
                    presigned_capability_id,
                    "extract_structured_data",
                    "output",
                )
            if output_presigned_urls:
                from .presigned_io import upload_presigned

                upload_presigned(
                    ext_path,
                    output_presigned_urls["ext_put"],
                    job_id=str(job_id),
                    expected_key=ext_key,
                )
                published_delete_url = output_presigned_urls.get("ext_delete")
            else:
                backend.upload_file(ext_path, ext_key)
            published_ext_locator = ext_key
        except Exception as exc:
            logger.error(
                "Failed to upload structured extraction result for job %s: %s",
                job_id,
                exc,
            )
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(
                f"requested structured-extraction upload failed: {exc}"
            ) from exc
        finally:
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)

    if _discard_if_postprocessing_stale(
        job,
        backend,
        published_ext_locator,
        presigned_delete_url=published_delete_url,
    ):
        return {
            "stage": "structured_data",
            "status": "already_finalized",
            "job_id": str(job_id),
        }

    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )
    return {
        "stage": "structured_data",
        "status": "extracted",
        "job_id": str(job_id),
    }


def _native_text_attempt_suffix(job, lease_token):
    """Stable namespace for one document owner, never a raw broker token."""
    digest = hashlib.sha256(str(lease_token).encode("utf-8")).hexdigest()[:32]
    return f"g{int(job.dispatch_generation)}/d{digest}"


class _NativeSourceIntegrityError(ValueError):
    def __init__(self, message, expected_sha256, observed_sha256):
        super().__init__(message)
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256


def _native_source_failure_data(exc):
    if isinstance(exc, _NativeSourceIntegrityError):
        return {
            "failure_kind": "native_source_digest_mismatch",
            "expected_source_sha256": exc.expected_sha256,
            "observed_source_sha256": exc.observed_sha256,
        }
    return None


def _native_source_byte_limit():
    return _bounded_operational_int(
        "OCR_COORDINATOR_MAX_UPLOAD_BYTES", 512 * 1024 * 1024,
        minimum=1, maximum=8 * 1024 * 1024 * 1024,
    )


def _verify_native_source_bytes(job, path):
    """Bind consumed private source bytes to the admitted hash and size."""
    expected_hash = str(job.source_hash or "")
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        raise ValueError("native source hash is missing or invalid")
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("native source must be a private regular file")
    limit = _native_source_byte_limit()
    if info.st_size > limit:
        raise ValueError("native source exceeds upload byte limit")
    expected_size = int(job.source_size_bytes or 0)
    if expected_size and info.st_size != expected_size:
        raise ValueError("native source size changed after admission")
    actual_hash = _compute_file_hash(path)
    if not hmac.compare_digest(actual_hash, expected_hash):
        raise _NativeSourceIntegrityError(
            "native source SHA-256 changed after admission", expected_hash, actual_hash,
        )


def _snapshot_native_nfs_source(job, source_path, private_path):
    """Copy one bounded NFS source through a no-follow descriptor, then hash."""
    parent_fd = -1
    source_fd = -1
    try:
        parent_fd, basename, _ = _open_local_parent(source_path, create=False)
        source_fd = os.open(
            basename,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("native NFS source must be a regular single-link file")
        limit = _native_source_byte_limit()
        if before.st_size > limit:
            raise ValueError("native source exceeds upload byte limit")
        copied = 0
        with os.fdopen(source_fd, "rb") as source_stream:
            source_fd = -1
            with open(private_path, "xb") as destination:
                while chunk := source_stream.read(min(1024 * 1024, limit + 1 - copied)):
                    copied += len(chunk)
                    if copied > limit:
                        raise ValueError("native source grew beyond upload byte limit")
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
        _verify_native_source_bytes(job, private_path)
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _commit_native_page_if_owned(
    job, lease_token, page_num, document_id, page_defaults,
    *, text_locator, custody_data=None,
):
    """Commit native/skip output, progress and custody under one parent fence."""
    text_hash = (page_defaults.get("ocr_provenance") or {}).get("text_sha256")
    if (
        not isinstance(text_hash, str)
        or len(text_hash) != 64
        or any(char not in "0123456789abcdef" for char in text_hash)
    ):
        raise ValueError("native page text identity must be a SHA-256 digest")
    _renew_processing_lease(job.job_id, lease_token, job)
    with transaction.atomic():
        parent = Job.objects.select_for_update().get(pk=job.pk)
        if (
            parent.status != Job.Status.PROCESSING
            or parent.processing_attempt_token != lease_token
            or parent.processing_attempt_identity != text_only_attempt_identity(job)
            or parent.processing_attempt_identity != text_only_attempt_identity(parent)
            or parent.processing_lease_expires_at is None
            or parent.processing_lease_expires_at <= timezone.now()
            or parent.dispatch_generation != job.dispatch_generation
            or parent.tenant_id != job.tenant_id
            or parent.source_hash != job.source_hash
            or parent.source_object_key != job.source_object_key
            or parent.source_file != job.source_file
        ):
            raise _ProcessingLeaseLost(
                f"native page parent lease lost for job {job.job_id} page {page_num}"
            )
        expected_document_id = parent.source_hash[:16] if parent.source_hash else str(parent.pk)[:16]
        if document_id != expected_document_id or page_defaults.get("document_id") != expected_document_id:
            raise _ProcessingLeaseLost("native page document identity changed")
        page = PageResult.objects.select_for_update().filter(
            job=parent, page_num=page_num,
        ).first()
        if page is not None and page.admitted_generation == parent.dispatch_generation:
            prior = page.ocr_provenance or {}
            if (
                prior.get("source_sha256") != parent.source_hash
                or prior.get("tenant_id") != parent.tenant_id
                or prior.get("source_object_key") != parent.source_object_key
                or prior.get("source_file") != parent.source_file
            ):
                raise _ProcessingLeaseLost(
                    "native source replacement requires a new dispatch generation"
                )
            if page.status == "ok":
                prior_token = str(page.celery_task_id or "")
                if (
                    page.document_id != expected_document_id
                    or page.ocr_method != page_defaults.get("ocr_method")
                    or len(prior_token) != 32
                    or any(c not in "0123456789abcdef" for c in prior_token)
                    or prior.get("document_attempt_identity") != parent.processing_attempt_identity
                ):
                    raise _ProcessingLeaseLost("native page has a foreign terminal winner")
                if parent.storage_backend_used == "s3":
                    prior_keys = text_only_attempt_keys(
                        parent, page_num, expected_document_id, prior_token,
                    )
                    expected_pdf = prior_keys["page_pdf_key"]
                    expected_text = prior_keys["page_text_key"]
                else:
                    prior_dir = os.path.join(
                        parent.nfs_job_path, "temp", expected_document_id,
                        _native_text_attempt_suffix(parent, prior_token),
                    )
                    expected_pdf = os.path.join(prior_dir, f"{page_num}.pdf")
                    expected_text = os.path.join(prior_dir, f"{page_num}.txt")
                if (
                    page.temp_pdf_path != expected_pdf
                    or prior.get("temp_text_path") != expected_text
                ):
                    raise _ProcessingLeaseLost("native page terminal locator is not canonical")
                return False
            if page.status in _SUCCESSFUL_PAGE_STATUSES or page.status == "failed":
                raise _ProcessingLeaseLost(
                    f"native page already has a terminal winner for job {job.job_id} page {page_num}"
                )
        if page is None:
            page = PageResult(job=parent, page_num=page_num)
        for field, value in page_defaults.items():
            setattr(page, field, value)
        page.admitted_generation = parent.dispatch_generation
        page.celery_task_id = lease_token
        provenance = dict(page.ocr_provenance or {})
        provenance["temp_text_path"] = text_locator
        provenance["source_sha256"] = parent.source_hash
        provenance["tenant_id"] = parent.tenant_id
        provenance["source_object_key"] = parent.source_object_key
        provenance["source_file"] = parent.source_file
        provenance["document_attempt_identity"] = parent.processing_attempt_identity
        page.ocr_provenance = provenance
        page.save()
        parent.pages_completed = max(parent.pages_completed, page_num)
        parent.save(update_fields=["pages_completed"])
        if custody_data is not None:
            _record_custody_event(
                parent, document_id, "text_extracted_natively",
                data=custody_data, worker_hostname=page_defaults["worker_hostname"],
            )
        return True


def _handoff_native_pdf_to_ocr(job, lease_token, page_num, assessment):
    """Atomically fence native publication and enqueue the normal OCR lane."""
    with transaction.atomic():
        parent = Job.objects.select_for_update().get(pk=job.pk)
        _renew_processing_lease(job.pk, lease_token, job)
        decision = {
            "page_num": page_num,
            "requested_skip_ocr": bool((parent.settings_json or {}).get("skip_ocr")),
            "effective_skip_ocr": False,
            "coverage": asdict(assessment),
            "previous_generation": parent.dispatch_generation,
        }
        parent.settings_json = {
            **(parent.settings_json or {}), "skip_ocr": False,
            "native_ocr_fallback": decision,
        }
        parent.dispatch_generation += 1
        parent.celery_task_id = ""
        parent.processing_attempt_token = ""
        parent.processing_attempt_identity = ""
        parent.processing_lease_expires_at = None
        parent.fanout_dispatched = False
        parent.pages_completed = 0
        parent.pages_failed = 0
        parent.save(update_fields=[
            "settings_json", "dispatch_generation", "celery_task_id",
            "processing_attempt_token", "processing_attempt_identity",
            "processing_lease_expires_at", "fanout_dispatched",
            "pages_completed", "pages_failed",
        ])
        PresignedTaskCapability.objects.filter(job=parent).delete()
        # An upgraded worker may be resuming a native attempt that published
        # a prefix before this preflight existed. Preserve its artifacts and
        # custody, but make every page eligible for the successor generation.
        PageResult.objects.filter(job=parent).update(
            status="pending", needs_review=True, celery_task_id="",
            lease_expires_at=None,
        )
        _record_custody_event(
            parent, parent.source_hash[:16], "native_pdf_ocr_required",
            data={**decision, "dispatch_generation": parent.dispatch_generation},
            worker_hostname=socket.gethostname(),
        )
        row = enqueue_task_dispatch(
            job=parent, stage="root_processing",
            generation=str(parent.dispatch_generation),
            task_name="jobs.tasks.extract_pages", args=[str(parent.pk)],
            kwargs=_tenant_for_task(parent), queue="coordinator",
            priority=_get_celery_priority(parent),
        )
    # The periodic outbox relay also publishes this row after process loss or
    # broker failure. The old delivery cannot claim the successor generation.
    transaction.on_commit(partial(
        publish_task_dispatch_outbox, dispatch_id=row.dispatch_id,
        dispatcher=_publish_root_outbox_row,
    ), robust=True)
    return {
        "status": "ocr_queued", "job_id": str(job.pk),
        "dispatch_id": str(row.dispatch_id), "reason": assessment.reason,
    }


@shared_task(
    bind=True,
    name="jobs.tasks.process_text_only",
    queue="cpu_general",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=45,
)
@_tenant_bound_task
def process_text_only(
    self,
    job_id,
    presigned_capability_id=None,
    tenant_id=None,
    dispatch_generation=None,
):
    """Extract native text only when every PDF page has measured coverage.

    Used when skip_ocr=True is passed via the API. Writes out the necessary
    temp page artifacts (text and PDF) and then dispatches assemble_document
    just like a normal OCR completion would. Incomplete or unknown PDF coverage
    durably hands off to normal OCR before any native page is published.
    """
    import fitz

    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}
    _pin_rls_for_job(job)

    stale = _reject_stale_dispatch_generation(job, dispatch_generation)
    if stale is not None:
        return stale

    terminal = _reject_if_job_terminal(job)
    if terminal is not None:
        return terminal

    lease_token = _claim_processing_lease(job, getattr(self.request, "id", None))
    if not lease_token:
        current = Job.objects.get(pk=job.pk)
        if (
            current.status == Job.Status.PROCESSING
            and current.dispatch_generation == job.dispatch_generation
            and current.processing_attempt_token
            and current.processing_attempt_identity == text_only_attempt_identity(current)
            and current.celery_task_id == str(getattr(self.request, "id", None) or "")
            and current.processing_lease_expires_at
            and current.processing_lease_expires_at > timezone.now()
        ):
            # A duplicate delivery of the same Celery ID must wait for the
            # live execution nonce to expire. A bounded retry survives worker
            # loss without permitting concurrent writes to the same keys.
            if self.request.retries < self.max_retries:
                remaining = (current.processing_lease_expires_at - timezone.now()).total_seconds()
                raise self.retry(countdown=max(1, min(60, int(remaining) + 1)))
        return {
            "status": "stale",
            "job_id": str(job_id),
            "message": "processing lease is owned by another task or job is no longer active",
        }

    # This terminal failure is bound to the newly claimed execution owner;
    # a parent replacement between preflight and failure cannot be failed by
    # the old delivery.  The same budget applies to native PDF expansion.
    try:
        _validate_job_page_budget(job)
    except ValueError as exc:
        message = f"text-only page budget rejected: {exc}"
        logger.error("%s for job %s", message, job_id)
        if not _fail_processing_job_if_owned(
            job_id, lease_token, message, expected_job=job,
        ):
            return {"status": "stale", "job_id": str(job_id), "message": message}
        return {"status": "error", "job_id": str(job_id), "message": message}

    hostname = socket.gethostname()

    work_dir = ""
    try:
        backend = _get_backend_for_job(job)
        storage_mode = backend.backend_name
        document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
        attempt_suffix = _native_text_attempt_suffix(job, lease_token)
        presigned_urls = None
        if storage_mode == "s3" and is_presigned_mode():
            if not presigned_capability_id:
                raise RuntimeError("text-only S3 worker requires a capability")
            presigned_urls = _request_fresh_presigned_urls(
                job,
                presigned_capability_id,
                "process_text_only",
                "input",
                page_num=1,
            )
        if storage_mode == "nfs":
            _renew_processing_lease(job_id, lease_token, job)
            job_path = _nfs_job_path_for_job(job)
            nfs_source_path = os.path.join(
                job_path, "source", os.path.basename(job.source_file)
            )
            work_dir = tempfile.mkdtemp(prefix=f"text_only_{job_id}_")
            source_path = os.path.join(work_dir, os.path.basename(job.source_file))
            temp_dir = os.path.join(job_path, "temp", document_id, attempt_suffix)
            _mkdir_tree_no_follow(temp_dir, getattr(settings, "NFS_ROOT", job_path))
            _snapshot_native_nfs_source(job, nfs_source_path, source_path)
            _renew_processing_lease(job_id, lease_token, job)
        else:
            work_dir = tempfile.mkdtemp(prefix=f"text_only_{job_id}_")
            source_path = os.path.join(work_dir, os.path.basename(job.source_file))
            temp_dir = os.path.join(work_dir, "temp", document_id, attempt_suffix)
            os.makedirs(temp_dir, exist_ok=True)
            _renew_processing_lease(job_id, lease_token, job)
            source_key = _job_storage_key(
                job_id, f"source/{os.path.basename(job.source_file)}"
            )
            if presigned_urls and presigned_urls.get("source_get"):
                from .presigned_io import download_presigned

                download_presigned(
                    presigned_urls["source_get"],
                    source_path,
                    job_id=str(job_id),
                    expected_key=source_key,
                    max_bytes=_native_source_byte_limit(),
                )
            else:
                backend.download_file(
                    source_key, source_path, max_bytes=_native_source_byte_limit(),
                )
            _verify_native_source_bytes(job, source_path)
            _renew_processing_lease(job_id, lease_token, job)
        Worker.objects.filter(hostname=hostname).update(
            status=Worker.Status.BUSY,
            current_task_id=lease_token,
            last_heartbeat=timezone.now(),
        )
    except _ProcessingLeaseLost:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "stale", "job_id": str(job_id)}
    except Exception as exc:
        message = (
            str(exc) if isinstance(exc, _NativeSourceIntegrityError)
            else f"native processing setup failed ({type(exc).__name__})"
        )
        logger.error("%s for job %s", message, job_id)
        try:
            if not _fail_processing_job_if_owned(
                job_id, lease_token, message, expected_job=job,
                failure_data=_native_source_failure_data(exc),
            ):
                return {"status": "stale", "job_id": str(job_id)}
            return {"status": "error", "job_id": str(job_id), "message": message}
        finally:
            if work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
            Worker.objects.filter(hostname=hostname, current_task_id=lease_token).update(
                status=Worker.Status.ONLINE,
                current_task_id="",
                last_heartbeat=timezone.now(),
            )

    try:
        if job.source_type == "text":
            _renew_processing_lease(job_id, lease_token, job)
            if job.total_pages != 1:
                raise ValueError("native text admission must have exactly one page")
            # For raw text/markdown/json files, just copy the text into page 1
            # and generate a dummy blank PDF page for structural compatibility
            # Invalid bytes are evidence. Never silently drop them from a
            # native-text result; fail and retain the original source.
            page_text = read_text_bounded(source_path)

            page_num = 1
            temp_pdf_path = os.path.join(temp_dir, f"{page_num}.pdf")
            temp_text_path = os.path.join(temp_dir, f"{page_num}.txt")

            # Save dummy empty PDF page
            page_pdf = fitz.open()
            page_pdf.new_page(width=595, height=842)  # A4 size
            page_pdf.save(temp_pdf_path, deflate=True)
            page_pdf.close()

            # Save text
            with open(temp_text_path, "w", encoding="utf-8") as f:
                f.write(page_text)

            page_presigned_urls = None
            if storage_mode == "s3":
                pdf_key = _job_storage_key(job_id, f"temp/{document_id}/{attempt_suffix}/{page_num}.pdf")
                text_key = _job_storage_key(
                    job_id, f"temp/{document_id}/{attempt_suffix}/{page_num}.txt"
                )
                if is_presigned_mode():
                    page_presigned_urls = _request_fresh_presigned_urls(
                        job,
                        presigned_capability_id,
                        "process_text_only",
                        "output",
                        page_num=page_num,
                        lease_token=lease_token,
                    )
                    from .presigned_io import upload_presigned

                    upload_presigned(
                        temp_pdf_path,
                        page_presigned_urls["page_pdf_put"],
                        job_id=str(job_id),
                        expected_key=pdf_key,
                    )
                    upload_presigned(
                        temp_text_path,
                        page_presigned_urls["page_text_put"],
                        job_id=str(job_id),
                        expected_key=text_key,
                    )
                else:
                    backend.upload_file(temp_pdf_path, pdf_key)
                    backend.upload_file(temp_text_path, text_key)

            # Record DB result
            _renew_or_discard_published_artifacts(
                partial(_commit_native_page_if_owned, job, lease_token, page_num, document_id, {
                    "document_id": document_id,
                    "ocr_method": "NativeText",
                    "ocr_language": job.detected_language or "und",
                    "ocr_confidence": 1.0,
                    "text_length": len(page_text),
                    "ocr_provenance": {"text_sha256": hashlib.sha256(page_text.encode("utf-8")).hexdigest()},
                    "status": "ok",
                    "worker_hostname": hostname,
                    "processing_time_ms": 1,
                    "temp_pdf_path": pdf_key if storage_mode == "s3" else temp_pdf_path,
                }, text_locator=text_key if storage_mode == "s3" else temp_text_path,
                custody_data={"page_num": page_num, "method": "NativeText", "text_length": len(page_text)}),
                job,
                backend,
                (
                    (
                        pdf_key if storage_mode == "s3" else temp_pdf_path,
                        (
                            page_presigned_urls.get("page_pdf_delete")
                            if page_presigned_urls
                            else None
                        ),
                    ),
                    (
                        text_key if storage_mode == "s3" else temp_text_path,
                        (
                            page_presigned_urls.get("page_text_delete")
                            if page_presigned_urls
                            else None
                        ),
                    ),
                ),
            )

        elif job.source_type == "pdf":
            with fitz.open(source_path) as doc:
                if doc.page_count != job.total_pages or doc.page_count > _coordinator_expansion_limit("MAX_PDF_PAGES", 5000):
                    raise ValueError("native PDF page count changed after admission")
                coverage_by_page = []
                for page_index in range(doc.page_count):
                    _renew_processing_lease(job_id, lease_token, job)
                    assessment = assess_native_pdf_page(doc[page_index])
                    if assessment.status != "measured" or assessment.needs_retry or assessment.needs_review:
                        return _handoff_native_pdf_to_ocr(
                            job, lease_token, page_index + 1, assessment,
                        )
                    coverage_by_page.append(asdict(assessment))
                for page_num in range(1, doc.page_count + 1):
                    _renew_processing_lease(job_id, lease_token, job)
                    page = doc[page_num - 1]
                    page_text = page.get_text()

                    temp_pdf_path = os.path.join(temp_dir, f"{page_num}.pdf")
                    temp_text_path = os.path.join(temp_dir, f"{page_num}.txt")

                    # Save single page PDF
                    page_pdf = fitz.open()
                    page_pdf.insert_pdf(
                        doc, from_page=page_num - 1, to_page=page_num - 1
                    )
                    page_pdf.save(temp_pdf_path, deflate=True)
                    page_pdf.close()

                    # Save text
                    with open(temp_text_path, "w", encoding="utf-8") as f:
                        f.write(page_text)

                    # S3 upload
                    page_presigned_urls = None
                    if storage_mode == "s3":
                        pdf_key = _job_storage_key(
                            job_id, f"temp/{document_id}/{attempt_suffix}/{page_num}.pdf"
                        )
                        text_key = _job_storage_key(
                            job_id, f"temp/{document_id}/{attempt_suffix}/{page_num}.txt"
                        )
                        if is_presigned_mode():
                            page_presigned_urls = _request_fresh_presigned_urls(
                                job,
                                presigned_capability_id,
                                "process_text_only",
                                "output",
                                page_num=page_num,
                                lease_token=lease_token,
                            )
                            from .presigned_io import upload_presigned

                            upload_presigned(
                                temp_pdf_path,
                                page_presigned_urls["page_pdf_put"],
                                job_id=str(job_id),
                                expected_key=pdf_key,
                            )
                            upload_presigned(
                                temp_text_path,
                                page_presigned_urls["page_text_put"],
                                job_id=str(job_id),
                                expected_key=text_key,
                            )
                        else:
                            backend.upload_file(temp_pdf_path, pdf_key)
                            backend.upload_file(temp_text_path, text_key)

                    # Record DB result
                    _renew_or_discard_published_artifacts(
                        partial(_commit_native_page_if_owned, job, lease_token, page_num, document_id, {
                            "document_id": document_id,
                            "ocr_method": "NativePDF",
                            "ocr_language": job.detected_language or "und",
                            "ocr_confidence": 1.0,
                            "text_length": len(page_text),
                            "ocr_provenance": {
                                "text_sha256": hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
                                "native_coverage": coverage_by_page[page_num - 1],
                            },
                            "status": "ok",
                            "worker_hostname": hostname,
                            "processing_time_ms": 10,
                            "temp_pdf_path": pdf_key if storage_mode == "s3" else temp_pdf_path,
                        }, text_locator=text_key if storage_mode == "s3" else temp_text_path,
                        custody_data={
                            "page_num": page_num, "method": "NativePDF",
                            "text_length": len(page_text),
                            "native_coverage": coverage_by_page[page_num - 1],
                        }),
                        job,
                        backend,
                        (
                            (
                                pdf_key if storage_mode == "s3" else temp_pdf_path,
                                (
                                    page_presigned_urls.get("page_pdf_delete")
                                    if page_presigned_urls
                                    else None
                                ),
                            ),
                            (
                                text_key if storage_mode == "s3" else temp_text_path,
                                (
                                    page_presigned_urls.get("page_text_delete")
                                    if page_presigned_urls
                                    else None
                                ),
                            ),
                        ),
                    )
        else:
            raise ValueError(
                f"skip_ocr is not supported for {job.source_type} documents."
            )

    except _ProcessingLeaseLost as exc:
        logger.warning(
            "process_text_only stopped after lease loss for job %s: %s", job_id, exc
        )
        return {
            "status": "stale",
            "job_id": str(job_id),
            "message": "processing lease was fenced before the next durable write",
        }
    except Exception as exc:
        message = f"Processing failed: {exc}"
        if not _fail_processing_job_if_owned(job_id, lease_token, message, expected_job=job):
            return {"status": "stale", "job_id": str(job_id)}
        logger.error("process_text_only failed for job %s: %s", job_id, exc)
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        return {"status": "error", "job_id": str(job_id), "message": message}
    finally:
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        Worker.objects.filter(hostname=hostname, current_task_id=lease_token).update(
            status=Worker.Status.ONLINE,
            current_task_id="",
            last_heartbeat=timezone.now(),
        )

    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )

    try:
        _renew_processing_lease(job_id, lease_token, job)
    except _ProcessingLeaseLost:
        logger.warning("process_text_only will not assemble fenced job %s", job_id)
        return {
            "status": "stale",
            "job_id": str(job_id),
            "message": "processing lease was fenced before assembly dispatch",
        }
    assembly_dispatch = _enqueue_assembly_dispatch(
        job_id,
        expected_processing_token=lease_token,
        expected_job=job,
    )
    if assembly_dispatch.get("status") == "stale":
        return assembly_dispatch
    return {
        "status": "processed",
        "job_id": str(job_id),
        "pages_processed": job.total_pages,
    }


# ---------------------------------------------------------------------------
# Worker Registration (called from Celery signals)
# ---------------------------------------------------------------------------


def register_worker(
    hostname,
    queues=None,
    capabilities=None,
    gpu_available=False,
    gpu_model="",
    gpu_vram_mb=0,
    gpu_index=None,
    cpu_cores=0,
    ram_mb=0,
    pipeline_version="",
):
    """Register or update a worker in the database.

    Called from worker_init signal handler.
    """
    Worker.objects.update_or_create(
        hostname=hostname,
        defaults={
            "status": Worker.Status.ONLINE,
            "queues": queues or [],
            "capabilities": capabilities or [],
            "gpu_available": gpu_available,
            "gpu_model": gpu_model,
            "gpu_vram_mb": gpu_vram_mb,
            "gpu_index": gpu_index,
            "cpu_cores": cpu_cores,
            "ram_mb": ram_mb,
            "last_heartbeat": timezone.now(),
            "pipeline_version": pipeline_version,
        },
    )


def unregister_worker(hostname):
    """Mark a worker as offline. Called from worker_shutdown signal."""
    Worker.objects.filter(hostname=hostname).update(
        status=Worker.Status.OFFLINE,
        current_task_id="",
    )
