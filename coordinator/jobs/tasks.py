"""Celery task definitions for the distributed OCR pipeline.

Tasks are organized into four queue categories:
- coordinator: Job lifecycle management (ingest, assemble, finalize)
- ocr_gpu: GPU-intensive OCR processing (process_document, process_page)
- ocr_cpu: CPU-only OCR processing (same tasks, ONNX Runtime / Tesseract)
- cpu_general: CPU-bound tasks (compression, entity extraction)

OCR task routing is controlled by the OCR_TASK_ROUTING env var:
- "gpu" (default): Route to ocr_gpu queue
- "cpu": Route to ocr_cpu queue
- "auto": Route to ocr_gpu if GPU workers are online, else ocr_cpu

Task granularity follows a hybrid model:
- Documents <= FANOUT_THRESHOLD pages: single process_document task
- Documents > FANOUT_THRESHOLD pages: fan-out via chord
"""

import hashlib
import json
import logging
import os
import shutil
import socket
import tempfile
import threading
import time

from celery import chord, shared_task
from celery.exceptions import MaxRetriesExceededError
from django.conf import settings
from django.db import models as db_models
from django.utils import timezone

from .extraction_models import ExtractedEntity, ExtractedFormValue
from .litigation_hold import is_litigation_hold_active
from .models import CustodyEvent, Job, PageResult, PiiEntity, Worker
from .presigned import (
    generate_compress_pdf_urls,
    generate_extract_entities_urls,
    generate_process_page_urls,
    is_presigned_mode,
)
from .storage import CachedS3Backend, create_storage_backend

logger = logging.getLogger(__name__)

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


def _get_celery_priority(job) -> int:
    """Return the RabbitMQ message priority for a job."""
    return _CELERY_PRIORITY_MAP.get(getattr(job, "priority", "normal"), 5)


def _format_processing_timeout(minutes: int) -> str:
    unit = "minute" if minutes == 1 else "minutes"
    return f"{minutes} {unit}"


def _get_job_processing_timeout_minutes(job) -> int:
    """Return the per-job timeout override or the coordinator default."""
    default_minutes = max(1, int(getattr(settings, "JOB_PROCESSING_TIMEOUT_MINUTES", 30)))
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


def _get_ocr_queue():
    """Determine the OCR task queue based on routing configuration.

    Reads the OCR_TASK_ROUTING environment variable:
    - "gpu" (default): Route to ocr_gpu queue
    - "cpu": Route to ocr_cpu queue
    - "auto": Route to ocr_gpu if GPU workers are online, else ocr_cpu

    Returns:
        Queue name string for Celery task dispatch.
    """
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
    return "ocr_gpu"  # default


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
    )
    if backend_name.strip().lower() == "s3":
        cache_dir = getattr(settings, "S3_CACHE_DIR", "/tmp/ocr-cache")
        max_gb = getattr(settings, "S3_CACHE_MAX_SIZE_GB", 10)
        backend = CachedS3Backend(
            inner=backend,
            cache_dir=cache_dir,
            max_size_bytes=int(max_gb * 1024**3),
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
    if job.storage_backend_used and backend.backend_name != job.storage_backend_used:
        logger.warning(
            "Backend mismatch for job %s: ingested=%s, current=%s. Using ingested backend.",
            job.job_id, job.storage_backend_used, backend.backend_name,
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
            )
        except Exception:
            logger.error(
                "Failed to create %s backend for job %s, using current config",
                job.storage_backend_used, job.job_id,
            )
            backend = _get_storage_backend()
    return backend


def _job_storage_key(job_id, subpath=""):
    """Return storage key for job artifacts."""
    if subpath:
        return f"jobs/{job_id}/{subpath}"
    return f"jobs/{job_id}"


def _nfs_job_path(job_id):
    """Return the canonical local workspace directory for a job."""
    return os.path.join(settings.NFS_ROOT, "jobs", str(job_id))


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


def _record_custody_event(job, document_id, event_type, data=None,
                          worker_hostname=""):
    """Record an unlinked custody event in PostgreSQL.

    Hash chain linking is deferred to assemble_document for chronological
    ordering across distributed workers.
    """
    CustodyEvent.objects.create(
        document_id=document_id,
        job=job,
        event_type=event_type,
        timestamp=timezone.now(),
        worker_hostname=worker_hostname,
        data=data or {},
    )


def _compute_file_hash(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _checkpoint_merged_pdf(merged_pdf, checkpoint_path):
    """Persist and reopen merged PDF to release intermediate merge memory."""
    import fitz

    merged_pdf.save(checkpoint_path, garbage=3, deflate=True)
    merged_pdf.close()
    return fitz.open(checkpoint_path)


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
        _ensure_job_dirs(job_path)
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
    if os.path.isfile(source_path):
        return

    # Check if source_file is an absolute path (direct upload)
    if os.path.isfile(job.source_file):
        # Path containment: only allow files within NFS_ROOT.
        real_source = os.path.realpath(job.source_file)
        allowed_root = os.path.realpath(settings.NFS_ROOT)
        if not real_source.startswith(allowed_root + os.sep) and real_source != allowed_root:
            raise _IngestError(
                f"Source file outside allowed directory: {job.source_file}"
            )
        shutil.copy2(job.source_file, source_path)
    else:
        raise _IngestError(f"Source file not found: {job.source_file}")


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
            _canonical_classify, include_coordinator_types=True,
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
            ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
            ".gif", ".webp", ".jp2",
        }
        if ext in image_exts:
            return "image", None
        return None, f"Unsupported extension: {ext or '<none>'}"

    def get_source_page_count(path: str, source_type: str) -> int:
        if source_type == "pdf":
            import fitz

            with fitz.open(path) as pdf:
                return len(pdf)
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
            logger.warning("File classification warning for %s: %s", source_path, warning)
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
    """Detect the document language (best-effort, defaults to 'en').

    Updates job.detected_language in the database.  Never raises.
    """
    try:
        from ocr_distributed.language import LanguageDetector
        model_path = os.path.join(settings.NFS_ROOT, "models", "lid.176.bin")
        detector = LanguageDetector(model_path)
        if source_type == "pdf":
            lang = detector.detect_from_pdf(source_path)
        else:
            lang = "en"
        job.detected_language = lang
        job.save(update_fields=["detected_language"])
    except Exception:
        job.detected_language = "en"
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
    """Transition job to PROCESSING and dispatch to the appropriate queue.

    Returns:
        Result dict with status, job_id, total_pages, and mode.
    """
    job.status = Job.Status.PROCESSING
    job.save(update_fields=["status"])

    celery_priority = _get_celery_priority(job)
    skip_ocr = job.settings_json.get("skip_ocr", False)

    if skip_ocr:
        logger.info("Job %s skipping OCR engine pass (skip_ocr=True)", job_id)
        process_text_only.apply_async(
            args=[str(job_id)], queue="cpu_general",
            priority=celery_priority,
        )
        return {
            "status": "dispatched",
            "job_id": str(job_id),
            "total_pages": total_pages,
            "mode": "skip_ocr",
        }

    # Determine OCR queue based on routing configuration
    ocr_queue = _get_ocr_queue()
    if ocr_queue != "ocr_gpu":
        logger.info("Job %s routed to %s queue", job_id, ocr_queue)

    if total_pages <= FANOUT_THRESHOLD:
        # Single-worker processing (both NFS and S3 now storage-aware)
        process_document.apply_async(
            args=[str(job_id)], queue=ocr_queue,
            priority=celery_priority,
        )
    else:
        # Fan-out: extract pages, then process in parallel, then assemble
        extract_pages.apply_async(
            args=[str(job_id)], priority=celery_priority,
        )

    # Note: presigned URLs are generated at dispatch time in extract_pages
    # and assemble_document, not here, because the coordinator tasks that
    # dispatch worker tasks have direct backend access.

    return {
        "status": "dispatched",
        "job_id": str(job_id),
        "total_pages": total_pages,
        "mode": "single" if total_pages <= FANOUT_THRESHOLD else "fanout",
    }


def _cleanup_temp_dir(job_path, storage_mode):
    """Remove the temporary working directory if we are in S3 mode.

    Safe to call multiple times; silently ignores missing directories.
    """
    if storage_mode != "nfs" and job_path and os.path.isdir(job_path):
        shutil.rmtree(job_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Coordinator Queue Tasks
# ---------------------------------------------------------------------------

@shared_task(bind=True, name="jobs.tasks.ingest_document", queue="coordinator",
             max_retries=2, default_retry_delay=30)
def ingest_document(self, job_id):
    """Validate source file, compute hash, detect language, and dispatch.

    This task runs on the coordinator and decides whether to process the
    document as a single task or fan out across multiple workers.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        logger.error("Job %s not found", job_id)
        return {"status": "error", "message": f"Job {job_id} not found"}

    if job.status == Job.Status.CANCELLED:
        return {"status": "cancelled", "job_id": str(job_id)}

    job.status = Job.Status.INGESTING
    job.started_at = timezone.now()
    job.celery_task_id = self.request.id or ""

    backend = _get_storage_backend()
    storage_mode = backend.backend_name
    job.storage_backend_used = storage_mode
    job.save(update_fields=["status", "started_at", "celery_task_id", "storage_backend_used"])

    job_path = None
    try:
        # Set up working directory and resolve source path
        job_path, source_path = _setup_ingest_work_dir(job, job_id, storage_mode)

        # Validate source file exists and is within allowed root
        _validate_and_resolve_source(job, source_path)

        # Upload to S3 backend if applicable
        if storage_mode == "s3":
            _upload_source_to_s3(backend, job_id, source_path)

        # Compute file hash (best-effort)
        try:
            job.source_hash = _compute_file_hash(source_path)
            job.save(update_fields=["source_hash"])
        except OSError as exc:
            logger.error("Failed to hash %s: %s", source_path, exc)

        # Classify file type and get page count
        source_type, total_pages = _classify_and_count_pages(job, source_path)

        # Detect language (best-effort, defaults to 'en')
        _detect_language(job, source_path, source_type)

    except _IngestError as exc:
        job.status = Job.Status.FAILED
        job.error_message = str(exc)
        job.save(update_fields=["status", "error_message"])
        return {"status": "error", "message": job.error_message}
    finally:
        # Always clean up temp directory for non-NFS modes.
        # This fixes the leak where some error paths forgot to clean up.
        _cleanup_temp_dir(job_path, storage_mode)

    # Record custody event
    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
    _record_custody_event(
        job, document_id, "file_ingested",
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


@shared_task(bind=True, name="jobs.tasks.assemble_document", queue="coordinator",
             max_retries=3, retry_backoff=True, retry_backoff_max=60,
             retry_jitter=True)
def assemble_document(self, job_id):
    """Merge page PDFs into final document and finalize custody chain.

    For fan-out processing, this is the chord callback that runs after all
    process_page tasks complete. It collects per-page temp PDFs, merges them
    in order, and writes the final output to storage backend.

    Retries up to 3 times with exponential backoff (jittered, max 60s) on
    transient failures such as S3 upload timeouts or storage backend errors.
    Permanent failures (job not found, cancelled) are not retried.
    """
    import fitz

    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        logger.error("Job %s not found for assembly", job_id)
        return {"status": "error", "message": f"Job {job_id} not found"}

    if job.status == Job.Status.CANCELLED:
        return {"status": "cancelled", "job_id": str(job_id)}

    job.status = Job.Status.ASSEMBLING
    job.save(update_fields=["status"])

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name

    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]

    # Setup working directory
    work_dir = ""
    if storage_mode == "nfs":
        job_path = job.nfs_job_path or _nfs_job_path(job_id)
        temp_dir = os.path.join(job_path, "temp", document_id)
        output_pdf_dir = os.path.join(job_path, "output", "EXPORT", "PDF")
        output_text_dir = os.path.join(job_path, "output", "EXPORT", "TEXT")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_pdf_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)
    else:
        # S3 mode: use temp directory
        import tempfile
        work_dir = tempfile.mkdtemp(prefix=f"assemble_{job_id}_")
        temp_dir = os.path.join(work_dir, "temp")
        output_pdf_dir = os.path.join(work_dir, "output", "PDF")
        output_text_dir = os.path.join(work_dir, "output", "TEXT")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_pdf_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)

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
    output_text_path = os.path.join(output_text_dir, f"{base_name}.txt")

    try:
        with open(output_text_path, "w", encoding="utf-8") as text_out:
            for page_num in range(1, job.total_pages + 1):
                if storage_mode == "nfs":
                    page_pdf_path = os.path.join(temp_dir, f"{page_num}.pdf")
                    page_text_path = os.path.join(temp_dir, f"{page_num}.txt")
                else:
                    # S3 mode: download from backend
                    page_pdf_path = os.path.join(temp_dir, f"{page_num}.pdf")
                    page_text_path = os.path.join(temp_dir, f"{page_num}.txt")
                    try:
                        pdf_key = _job_storage_key(job_id, f"temp/{document_id}/{page_num}.pdf")
                        text_key = _job_storage_key(job_id, f"temp/{document_id}/{page_num}.txt")
                        if backend.exists(pdf_key):
                            backend.download_file(pdf_key, page_pdf_path)
                        if backend.exists(text_key):
                            backend.download_file(text_key, page_text_path)
                    except Exception as exc:
                        logger.warning("Failed to download page %d artifacts: %s", page_num, exc)

                if os.path.isfile(page_pdf_path):
                    try:
                        page_doc = fitz.open(page_pdf_path)
                        merged_pdf.insert_pdf(page_doc)
                        page_doc.close()
                        pages_assembled += 1

                        if checkpoint_enabled and pages_assembled % checkpoint_interval == 0:
                            merged_pdf = _checkpoint_merged_pdf(merged_pdf, checkpoint_path)
                            logger.info(
                                "Assembly checkpointed for job %s at %d/%d pages",
                                job_id,
                                pages_assembled,
                                job.total_pages,
                            )
                    except Exception as exc:
                        logger.error("Failed to merge page %d for job %s: %s",
                                     page_num, job_id, exc)
                else:
                    logger.warning("Missing page PDF %d for job %s", page_num, job_id)

                page_text = ""
                if os.path.isfile(page_text_path):
                    try:
                        with open(page_text_path, "r", encoding="utf-8") as f:
                            page_text = f.read()
                    except OSError:
                        page_text = ""

                if page_num > 1:
                    text_out.write("\n\n")
                text_out.write(page_text)

        # Write merged PDF
        output_pdf_path = os.path.join(output_pdf_dir, f"{base_name}.pdf")
        if merged_pdf.page_count > 0:
            merged_pdf.save(output_pdf_path)
        merged_pdf.close()
        merged_pdf = None

        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)

        # Upload to backend if S3 mode
        if storage_mode == "s3":
            try:
                text_key = _job_storage_key(job_id, f"output/EXPORT/TEXT/{base_name}.txt")
                backend.upload_file(output_text_path, text_key)
                output_text_path = text_key
                if os.path.isfile(output_pdf_path):
                    pdf_key = _job_storage_key(job_id, f"output/EXPORT/PDF/{base_name}.pdf")
                    backend.upload_file(output_pdf_path, pdf_key)
                    output_pdf_path = pdf_key
                else:
                    output_pdf_path = ""
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                logger.error("Failed to upload assembly artifacts to S3 for job %s: %s",
                             job_id, exc)
                try:
                    retry_num = self.request.retries + 1
                    logger.warning(
                        "Retrying assemble_document for job %s (attempt %d/%d) "
                        "after S3 upload failure: %s",
                        job_id, retry_num, self.max_retries, exc,
                    )
                    raise self.retry(exc=exc)
                except MaxRetriesExceededError:
                    logger.error(
                        "assemble_document permanently failed for job %s after "
                        "%d retries (S3 upload): %s",
                        job_id, self.max_retries, exc,
                    )
                    job.refresh_from_db()
                    job.status = Job.Status.FAILED
                    job.error_message = f"S3 upload failed during assembly: {exc}"
                    job.save(update_fields=["status", "error_message"])
                    return {
                        "status": "error",
                        "job_id": str(job_id),
                        "message": f"S3 upload failed: {exc}",
                    }

        # Finalize custody chain
        if storage_mode == "nfs":
            _finalize_custody_chain(job, document_id, job_path)
        else:
            _finalize_custody_chain_s3(job, document_id, backend)

        # Record assembly event (idempotent: skip if already recorded on retry)
        if not CustodyEvent.objects.filter(
            job=job, document_id=document_id, event_type="assembly_complete"
        ).exists():
            _record_custody_event(
                job, document_id, "assembly_complete",
                data={
                    "pages_assembled": pages_assembled,
                    "total_pages": job.total_pages,
                    "output_pdf": output_pdf_path,
                },
                worker_hostname=socket.gethostname(),
            )

        # Update job progress
        job.pages_completed = pages_assembled
        job.save(update_fields=["pages_completed"])

        # Dispatch post-processing chord for both backends (now storage-aware).
        compress_presigned = None
        entities_presigned = None
        ext_presigned = None
        if storage_mode == "s3" and is_presigned_mode():
            compress_presigned = generate_compress_pdf_urls(backend, job)
            entities_presigned = generate_extract_entities_urls(backend, job)
            from .presigned import generate_extract_structured_data_urls
            ext_presigned = generate_extract_structured_data_urls(backend, job)

        post_tasks = [
            compress_pdf.s(str(job_id), presigned_urls=compress_presigned),
            extract_entities.s(str(job_id), presigned_urls=entities_presigned),
            extract_structured_data.s(str(job_id), presigned_urls=ext_presigned),
        ]
        callback = finalize_job.si(str(job_id))
        errback = chord_error_handler.s(str(job_id))
        for task_sig in post_tasks:
            task_sig.link_error(errback)
        chord(post_tasks)(callback)

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
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


def _finalize_custody_chain(job, document_id, job_path):
    """Sort custody events chronologically and compute hash chain.

    Two-phase linking: workers write unlinked events during processing;
    this function sorts and links them into a verified hash chain, then
    exports to JSONL on NFS.
    """
    events = CustodyEvent.objects.filter(
        job=job, document_id=document_id
    ).order_by("timestamp")

    prev_hash = None
    for event in events:
        event_dict = {
            "document_id": event.document_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(timespec="milliseconds"),
            "data": event.data,
            "prev_hash": prev_hash,
        }
        event_bytes = json.dumps(event_dict, sort_keys=True, default=str).encode("utf-8")
        event_hash = hashlib.sha256(event_bytes).hexdigest()

        event.prev_hash = prev_hash or ""
        event.event_hash = event_hash
        event.chain_finalized = True
        event.save(update_fields=["prev_hash", "event_hash", "chain_finalized"])
        prev_hash = event_hash

    # Export to JSONL on NFS
    custody_dir = os.path.join(job_path, "output", "EXPORT", "CUSTODY")
    os.makedirs(custody_dir, exist_ok=True)
    jsonl_path = os.path.join(custody_dir, f"{document_id}.custody.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for event in events:
            event_dict = {
                "document_id": event.document_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(timespec="milliseconds"),
                "data": event.data,
                "prev_hash": event.prev_hash,
                "hash": event.event_hash,
            }
            f.write(json.dumps(event_dict, default=str) + "\n")


def _finalize_custody_chain_s3(job, document_id, backend):
    """Sort custody events and compute hash chain, then upload to S3."""
    events = CustodyEvent.objects.filter(
        job=job, document_id=document_id
    ).order_by("timestamp")

    prev_hash = None
    for event in events:
        event_dict = {
            "document_id": event.document_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(timespec="milliseconds"),
            "data": event.data,
            "prev_hash": prev_hash,
        }
        event_bytes = json.dumps(event_dict, sort_keys=True, default=str).encode("utf-8")
        event_hash = hashlib.sha256(event_bytes).hexdigest()

        event.prev_hash = prev_hash or ""
        event.event_hash = event_hash
        event.chain_finalized = True
        event.save(update_fields=["prev_hash", "event_hash", "chain_finalized"])
        prev_hash = event_hash

    # Export to temp JSONL and upload
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl", delete=False) as f:
        jsonl_path = f.name
        for event in events:
            event_dict = {
                "document_id": event.document_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(timespec="milliseconds"),
                "data": event.data,
                "prev_hash": event.prev_hash,
                "hash": event.event_hash,
            }
            f.write(json.dumps(event_dict, default=str) + "\n")

    try:
        custody_key = _job_storage_key(job.job_id, f"output/EXPORT/CUSTODY/{document_id}.custody.jsonl")
        backend.upload_file(jsonl_path, custody_key)
    finally:
        if os.path.isfile(jsonl_path):
            os.remove(jsonl_path)


@shared_task(bind=True, name="jobs.tasks.finalize_job", queue="coordinator")
def finalize_job(self, job_id):
    """Mark job terminal state, compute result summary, trigger webhook."""
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    if job.status == Job.Status.CANCELLED:
        return {"status": "cancelled", "job_id": str(job_id)}

    # Compute result summary from page results
    page_results = PageResult.objects.filter(job=job)
    pages_ok = page_results.filter(status="ok").count()
    pages_fallback = page_results.filter(status="fallback").count()
    pages_image_only = page_results.filter(status="image_only").count()
    pages_failed = page_results.filter(status="failed").count()

    confidences = [p.ocr_confidence for p in page_results if p.ocr_confidence > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    result_summary = {
        "total_pages": job.total_pages,
        "pages_ok": pages_ok,
        "pages_fallback": pages_fallback,
        "pages_image_only": pages_image_only,
        "pages_failed": pages_failed,
        "average_confidence": round(avg_confidence, 4),
    }

    has_failed_pages = pages_failed > 0
    job.status = Job.Status.FAILED if has_failed_pages else Job.Status.COMPLETED
    job.completed_at = timezone.now()
    job.result_summary = result_summary
    job.pages_failed = pages_failed
    update_fields = ["status", "completed_at", "result_summary", "pages_failed"]
    if has_failed_pages and not job.error_message:
        job.error_message = f"{pages_failed} page(s) failed during processing"
        update_fields.append("error_message")
    job.save(update_fields=update_fields)

    # Trigger webhook if configured
    if job.webhook_url:
        _send_webhook(job, result_summary)

    task_status = "failed" if has_failed_pages else "completed"
    logger.info("Job %s finalized (%s): %s", job_id, task_status, result_summary)
    return {"status": task_status, "job_id": str(job_id), "summary": result_summary}


def _deliver_webhook_with_retry(url, body, headers, max_attempts=3):
    """Attempt webhook delivery with exponential backoff.

    Returns ``True`` on success, ``False`` after all attempts are exhausted.
    """
    import urllib.request

    from ocr_distributed.ssrf import safe_opener  # noqa: F811 – lazy import

    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
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
        logger.warning(
            "Webhook URL for job %s failed validation: %s", job.job_id, exc
        )
        job.webhook_status = "failed"
        job.save(update_fields=["webhook_status"])
        return

    event_name = "job.failed" if job.status == Job.Status.FAILED else "job.completed"
    body = json.dumps({
        "event": event_name,
        "job_id": str(job.job_id),
        "status": job.status,
        "result": payload,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}

    if job.webhook_secret:
        signature = hmac.new(
            job.webhook_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        headers["X-OCR-Signature"] = f"sha256={signature}"

    success = _deliver_webhook_with_retry(job.webhook_url, body, headers)
    job.webhook_status = "delivered" if success else "failed"
    if not success:
        logger.error("Webhook delivery failed for job %s after all retries", job.job_id)
    job.save(update_fields=["webhook_status"])


# ---------------------------------------------------------------------------
# Periodic Tasks (Celery Beat)
# ---------------------------------------------------------------------------

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


@shared_task(name="jobs.tasks.cleanup_stale_jobs", queue="coordinator")
def cleanup_stale_jobs():
    """Flag jobs stuck in processing beyond their configured timeout."""
    now = timezone.now()
    active_jobs = list(Job.objects.filter(
        status__in=[Job.Status.PROCESSING, Job.Status.ASSEMBLING],
        started_at__isnull=False,
    ))
    stale_jobs = []
    for job in active_jobs:
        timeout_minutes = _get_job_processing_timeout_minutes(job)
        cutoff = now - timezone.timedelta(minutes=timeout_minutes)
        if job.started_at < cutoff:
            stale_jobs.append((job, timeout_minutes))

    for job, timeout_minutes in stale_jobs:
        logger.warning("Stale job detected: %s (started %s)", job.job_id, job.started_at)
        job.status = Job.Status.FAILED
        job.error_message = (
            "Job exceeded processing timeout "
            f"({_format_processing_timeout(timeout_minutes)})"
        )
        job.save(update_fields=["status", "error_message"])
    return {"stale_jobs": len(stale_jobs)}


@shared_task(name="jobs.tasks.cleanup_completed_jobs", queue="coordinator")
def cleanup_completed_jobs():
    """Delete completed/failed/cancelled jobs older than JOB_RETENTION_DAYS.

    Runs as a daily Celery Beat task. Retention period is configurable
    via the JOB_RETENTION_DAYS environment variable (default: 30).

    Respects LITIGATION_HOLD env var — skips all deletions when active.
    Emits a CustodyEvent recording the cleanup action before deletion.
    """
    if is_litigation_hold_active():
        logger.info("LITIGATION_HOLD active — cleanup_completed_jobs skipped")
        return {"deleted": 0, "litigation_hold": True}

    retention_days = int(os.environ.get("JOB_RETENTION_DAYS", "30"))
    cutoff = timezone.now() - timezone.timedelta(days=retention_days)

    old_jobs = Job.objects.filter(
        status__in=[Job.Status.COMPLETED, Job.Status.FAILED, Job.Status.CANCELLED],
        created_at__lt=cutoff,
    )
    count = old_jobs.count()
    if count == 0:
        return {"deleted": 0}

    # Record CustodyEvent BEFORE deletion so the FK still exists
    try:
        anchor_job = Job.objects.first()
        if anchor_job is not None:
            CustodyEvent.objects.create(
                document_id=f"cleanup-jobs-{timezone.now().isoformat()}",
                job=anchor_job,
                event_type="data_deleted",
                data={
                    "action": "cleanup_completed_jobs",
                    "retention_policy": f"JOB_RETENTION_DAYS={retention_days}",
                    "jobs_to_delete": count,
                    "cutoff_date": cutoff.isoformat(),
                    "reason": "retention_policy",
                    "source": "celery_beat",
                },
            )
    except Exception:
        logger.exception("Failed to record cleanup custody event")

    # Clean up extraction data linked to old jobs (explicit audit trail
    # before cascade delete removes them silently)
    form_values_deleted = ExtractedFormValue.objects.filter(job__in=old_jobs).count()
    entities_deleted = ExtractedEntity.objects.filter(job__in=old_jobs).count()
    if form_values_deleted or entities_deleted:
        ExtractedFormValue.objects.filter(job__in=old_jobs).delete()
        ExtractedEntity.objects.filter(job__in=old_jobs).delete()
        logger.info(
            "Pre-cascade cleanup: %d form values, %d extracted entities removed",
            form_values_deleted, entities_deleted,
        )

    # Remove storage artifacts (per-job backend for mixed NFS/S3 environments)
    nfs_cleaned = 0
    s3_jobs_cleaned = 0
    s3_objects_cleaned = 0
    for job in old_jobs:
        backend = _get_backend_for_job(job)
        storage_mode = backend.backend_name
        if storage_mode == "nfs":
            if job.nfs_job_path and os.path.isdir(job.nfs_job_path):
                shutil.rmtree(job.nfs_job_path, ignore_errors=True)
                nfs_cleaned += 1
        else:
            try:
                job_prefix = _job_storage_key(job.job_id)
                keys = backend.list_objects(job_prefix)
                if keys:
                    deleted_count = backend.delete_many(keys)
                    s3_jobs_cleaned += 1
                    s3_objects_cleaned += deleted_count
                    logger.info("Deleted %d/%d S3 objects for job %s", deleted_count, len(keys), job.job_id)
            except Exception as exc:
                logger.error("Failed to delete S3 objects for job %s: %s", job.job_id, exc)

    total_deleted, per_model = old_jobs.delete()
    deleted = per_model.get("jobs.Job", 0)
    logger.info(
        "Cleaned up %d old jobs (%d NFS dirs, %d S3 jobs/%d objects removed)",
        deleted, nfs_cleaned, s3_jobs_cleaned, s3_objects_cleaned,
    )
    return {
        "deleted": deleted,
        "nfs_cleaned": nfs_cleaned,
        "s3_jobs_cleaned": s3_jobs_cleaned,
        "s3_objects_cleaned": s3_objects_cleaned,
    }


@shared_task(name="jobs.tasks.cleanup_pii_entities", queue="coordinator")
def cleanup_pii_entities():
    """Delete PII entity records older than PII_ENTITY_RETENTION_DAYS.

    Runs as a daily Celery Beat task.  Only deletes PII entities belonging
    to completed or failed jobs (active jobs are never touched).

    Respects LITIGATION_HOLD env var — skips all deletions when active.
    Emits a CustodyEvent recording the purge action.
    """
    if is_litigation_hold_active():
        logger.info("LITIGATION_HOLD active — cleanup_pii_entities skipped")
        return {"deleted": 0, "litigation_hold": True}

    retention_days = int(os.environ.get("PII_ENTITY_RETENTION_DAYS", "90"))
    cutoff = timezone.now() - timezone.timedelta(days=retention_days)

    old_pii = PiiEntity.objects.filter(
        created_at__lt=cutoff,
        job__status__in=[Job.Status.COMPLETED, Job.Status.FAILED, Job.Status.CANCELLED],
    )
    count = old_pii.count()
    if count == 0:
        return {"deleted": 0}

    # Record CustodyEvent BEFORE deletion
    try:
        anchor_job = Job.objects.first()
        if anchor_job is not None:
            CustodyEvent.objects.create(
                document_id=f"cleanup-pii-{timezone.now().isoformat()}",
                job=anchor_job,
                event_type="pii_deleted",
                data={
                    "action": "cleanup_pii_entities",
                    "retention_policy": f"PII_ENTITY_RETENTION_DAYS={retention_days}",
                    "entities_to_delete": count,
                    "cutoff_date": cutoff.isoformat(),
                    "reason": "retention_policy",
                    "source": "celery_beat",
                },
            )
    except Exception:
        logger.exception("Failed to record PII cleanup custody event")

    deleted, _ = old_pii.delete()
    logger.info("Cleaned up %d old PII entities (retention=%d days)", deleted, retention_days)

    # Also clean up PII-typed ExtractedEntity records under the same retention
    extracted_pii_types = {
        "SSN", "DOB", "EMAIL", "PHONE", "NAME", "ADDRESS",
        "CREDIT_CARD", "BANK_ACCOUNT", "PASSPORT", "DRIVERS_LICENSE",
    }
    old_extracted = ExtractedEntity.objects.filter(
        created_at__lt=cutoff,
        entity_type__in=extracted_pii_types,
        job__status__in=[Job.Status.COMPLETED, Job.Status.FAILED, Job.Status.CANCELLED],
    )
    extracted_count = old_extracted.count()
    if extracted_count > 0:
        old_extracted.delete()
        logger.info(
            "Cleaned up %d old PII-typed ExtractedEntity records (retention=%d days)",
            extracted_count, retention_days,
        )

    return {
        "deleted": deleted,
        "extracted_pii_deleted": extracted_count,
        "retention_days": retention_days,
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

    retention_days = int(os.environ.get("OUTPUT_RETENTION_DAYS", "90"))
    try:
        call_command("cleanup_output", "--confirm", f"--retention-days={retention_days}")
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

    retention_days = int(os.environ.get("AUDIT_LOG_RETENTION_DAYS", "2555"))
    try:
        call_command("rotate_audit_logs", "--confirm", f"--retention-days={retention_days}")
    except Exception:
        logger.exception("rotate_audit_logs command failed")
        return {"status": "error"}
    return {"status": "ok", "retention_days": retention_days}


# ---------------------------------------------------------------------------
# OCR GPU Queue Tasks
# ---------------------------------------------------------------------------

@shared_task(bind=True, name="jobs.tasks.process_document",
             queue="ocr_gpu", acks_late=True, reject_on_worker_lost=True,
             max_retries=3, retry_backoff=True, retry_backoff_max=60,
             retry_jitter=True)
def process_document(self, job_id, presigned_urls=None):
    """Process an entire document on a single worker (for docs <= FANOUT_THRESHOLD pages).

    Runs PaddleOCR with Tesseract fallback on each page, produces per-page
    temp PDFs, then calls assemble_document.

    Retries up to 3 times with exponential backoff (jittered, max 60s) on
    transient failures such as S3 timeouts or database connection drops.
    Permanent failures (job not found, cancelled, KeyboardInterrupt) are
    not retried.

    Args:
        presigned_urls: Optional dict with presigned S3 URLs (source_get, pages).
            When provided, the worker uses HTTP GET/PUT instead of S3 credentials.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    if job.status == Job.Status.CANCELLED:
        return {"status": "cancelled", "job_id": str(job_id)}

    hostname = socket.gethostname()
    job.assigned_worker = hostname
    job.save(update_fields=["assigned_worker"])

    # Update worker status
    Worker.objects.filter(hostname=hostname).update(
        status=Worker.Status.BUSY,
        current_task_id=self.request.id or "",
        last_heartbeat=timezone.now(),
    )

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name
    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]

    # Setup paths based on storage backend
    work_dir = ""
    if storage_mode == "nfs":
        job_path = job.nfs_job_path or _nfs_job_path(job_id)
        source_path = os.path.join(job_path, "source", os.path.basename(job.source_file))
    else:
        # S3 mode: download source to temp directory
        import tempfile
        work_dir = tempfile.mkdtemp(prefix=f"process_doc_{job_id}_")
        source_path = os.path.join(work_dir, os.path.basename(job.source_file))
        try:
            if presigned_urls and presigned_urls.get("source_get"):
                from .presigned_io import download_presigned
                download_presigned(presigned_urls["source_get"], source_path)
            else:
                source_key = _job_storage_key(job_id, f"source/{os.path.basename(job.source_file)}")
                backend.download_file(source_key, source_path)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            logger.error("S3 source download failed for job %s: %s", job_id, exc)
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            Worker.objects.filter(hostname=hostname).update(
                status=Worker.Status.ONLINE,
                current_task_id="",
            )
            try:
                retry_num = self.request.retries + 1
                logger.warning(
                    "Retrying process_document for job %s (attempt %d/%d) "
                    "after S3 download failure: %s",
                    job_id, retry_num, self.max_retries, exc,
                )
                raise self.retry(exc=exc)
            except MaxRetriesExceededError:
                logger.error(
                    "process_document permanently failed for job %s after "
                    "%d retries (S3 download): %s",
                    job_id, self.max_retries, exc,
                )
                job.status = Job.Status.FAILED
                job.error_message = f"S3 source download failed: {exc}"
                job.save(update_fields=["status", "error_message"])
                Worker.objects.filter(hostname=hostname).update(
                    tasks_failed=db_models.F("tasks_failed") + 1,
                )
                return {"status": "error", "job_id": str(job_id), "message": f"Download failed: {exc}"}

    try:
        for page_num in range(1, job.total_pages + 1):
            _process_single_page(
                job,
                source_path,
                page_num,
                document_id,
                hostname,
                backend,
                storage_mode,
                s3_temp_dir=work_dir,
            )
            # Update progress
            job.pages_completed = page_num
            job.save(update_fields=["pages_completed"])
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        logger.error("process_document failed for job %s: %s", job_id, exc)
        # Reset worker status before retry/failure
        Worker.objects.filter(hostname=hostname).update(
            status=Worker.Status.ONLINE,
            current_task_id="",
            last_heartbeat=timezone.now(),
        )
        # Cleanup temp directory for S3 mode before retry
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

        # Plan C Phase 1, item C5 -- if the job has been running long
        # enough on its assigned cluster, opportunistically kick the
        # federation failover engine.  No-op when failover is disabled
        # or the engine is unregistered (default single-cluster mode).
        if _is_running_too_long_on_unhealthy_cluster(job):
            _trigger_failover_opportunistically(
                job, reason="cluster_unhealthy"
            )

        try:
            retry_num = self.request.retries + 1
            logger.warning(
                "Retrying process_document for job %s (attempt %d/%d) "
                "after processing failure: %s",
                job_id, retry_num, self.max_retries, exc,
            )
            raise self.retry(exc=exc)
        except MaxRetriesExceededError:
            logger.error(
                "process_document permanently failed for job %s after "
                "%d retries: %s",
                job_id, self.max_retries, exc,
            )
            job.refresh_from_db()
            job.status = Job.Status.FAILED
            job.error_message = f"Processing failed: {exc}"
            job.save(update_fields=["status", "error_message"])
            # Increment failure counter
            Worker.objects.filter(hostname=hostname).update(
                tasks_failed=db_models.F("tasks_failed") + 1,
            )
            return {"status": "error", "job_id": str(job_id), "message": str(exc)}
    finally:
        # Cleanup temp directory for S3 mode
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        Worker.objects.filter(hostname=hostname).update(
            status=Worker.Status.ONLINE,
            current_task_id="",
            last_heartbeat=timezone.now(),
        )

    # Increment success counter
    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )

    # Proceed to assembly
    assemble_document.delay(str(job_id))

    return {
        "status": "processed",
        "job_id": str(job_id),
        "pages_processed": job.total_pages,
    }


@shared_task(bind=True, name="jobs.tasks.extract_pages",
             queue="coordinator", acks_late=True, reject_on_worker_lost=True)
def extract_pages(self, job_id):
    """Extract page images for fan-out processing (docs > FANOUT_THRESHOLD pages).

    Creates a Celery chord: group of process_page tasks -> assemble_document.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    if job.status == Job.Status.CANCELLED:
        return {"status": "cancelled", "job_id": str(job_id)}

    # Create PageResult rows for tracking
    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
    for page_num in range(1, job.total_pages + 1):
        PageResult.objects.get_or_create(
            job=job, page_num=page_num,
            defaults={"document_id": document_id, "status": "pending"},
        )

    _record_custody_event(
        job, document_id, "pages_extracted",
        data={"total_pages": job.total_pages},
        worker_hostname=socket.gethostname(),
    )

    # Create the chord: process all pages in parallel, then assemble
    use_presigned = is_presigned_mode()
    if use_presigned:
        backend = _get_backend_for_job(job)

    # Determine OCR queue for page tasks
    ocr_queue = _get_ocr_queue()
    if ocr_queue != "ocr_gpu":
        logger.info("Fan-out for job %s routed to %s queue", job_id, ocr_queue)

    celery_priority = _get_celery_priority(job)

    page_tasks = []
    for page_num in range(1, job.total_pages + 1):
        presigned_urls = None
        if use_presigned:
            presigned_urls = generate_process_page_urls(
                backend, job, page_num, document_id,
            )
        page_tasks.append(
            process_page.s(
                str(job_id), page_num, presigned_urls=presigned_urls,
            ).set(queue=ocr_queue, priority=celery_priority)
        )

    callback = assemble_document.si(str(job_id)).set(priority=celery_priority)
    errback = chord_error_handler.s(str(job_id))
    for task_sig in page_tasks:
        task_sig.link_error(errback)
    chord(page_tasks)(callback)

    return {
        "status": "fanout_dispatched",
        "job_id": str(job_id),
        "page_tasks": len(page_tasks),
    }


@shared_task(bind=True, name="jobs.tasks.chord_error_handler", queue="coordinator")
def chord_error_handler(self, request, exc, traceback, job_id):
    """Handle chord failures when any process_page task raises an exception.

    Sets the job status to FAILED and records a custody event with the error.
    Without this handler, failed chords leave jobs stuck in 'processing' until
    the cleanup_stale_jobs periodic task catches them after the configured
    processing timeout.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        logger.error("chord_error_handler: Job %s not found", job_id)
        return {"status": "error", "message": f"Job {job_id} not found"}

    error_msg = f"Chord processing failed: {exc}"
    job.status = Job.Status.FAILED
    job.error_message = error_msg
    job.save(update_fields=["status", "error_message"])

    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
    _record_custody_event(
        job, document_id, "chord_failed",
        data={"error": str(exc), "task_id": str(request.id) if request else ""},
        worker_hostname=socket.gethostname(),
    )

    logger.error("Chord failed for job %s: %s", job_id, exc)
    return {"status": "failed", "job_id": str(job_id), "error": error_msg}


@shared_task(bind=True, name="jobs.tasks.process_page",
             queue="ocr_gpu", acks_late=True, reject_on_worker_lost=True,
             max_retries=1, default_retry_delay=60)
def process_page(self, job_id, page_num, presigned_urls=None):
    """Process a single page of a document.

    Idempotent: re-processing overwrites the same temp files.

    Args:
        presigned_urls: Optional dict with presigned S3 URLs (source_get,
            page_pdf_put, page_text_put). When provided, the worker uses
            HTTP GET/PUT instead of S3 credentials.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    if job.status == Job.Status.CANCELLED:
        return {"status": "cancelled", "job_id": str(job_id)}

    hostname = socket.gethostname()
    Worker.objects.filter(hostname=hostname).update(
        last_heartbeat=timezone.now(),
    )

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name

    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
    temp_dir = ""

    try:
        if storage_mode == "nfs":
            job_path = job.nfs_job_path or _nfs_job_path(job_id)
            source_path = os.path.join(job_path, "source", os.path.basename(job.source_file))
        else:
            # S3 mode: download source to local temp first.
            import tempfile

            temp_dir = tempfile.mkdtemp(prefix=f"page_{job_id}_{page_num}_")
            source_path = os.path.join(temp_dir, os.path.basename(job.source_file))
            if presigned_urls and presigned_urls.get("source_get"):
                from .presigned_io import download_presigned
                download_presigned(presigned_urls["source_get"], source_path)
            else:
                source_key = _job_storage_key(job_id, f"source/{os.path.basename(job.source_file)}")
                backend.download_file(source_key, source_path)

        _process_single_page(
            job,
            source_path,
            page_num,
            document_id,
            hostname,
            backend,
            storage_mode,
            s3_temp_dir=temp_dir,
            presigned_urls=presigned_urls,
        )
    except Exception as exc:
        logger.error("process_page failed for job %s page %d: %s",
                     job_id, page_num, exc)
        PageResult.objects.update_or_create(
            job=job, page_num=page_num,
            defaults={
                "document_id": document_id,
                "status": "failed",
                "worker_hostname": hostname,
                "celery_task_id": self.request.id or "",
            },
        )
        _record_custody_event(
            job, document_id, "processing_failed",
            data={"page_num": page_num, "error": str(exc)},
            worker_hostname=hostname,
        )
        # Increment failure counter
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        raise
    finally:
        if storage_mode == "s3" and temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Increment job progress and worker success counter atomically
    Job.objects.filter(job_id=job_id).update(
        pages_completed=db_models.F("pages_completed") + 1,
    )
    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )

    return {
        "status": "page_processed",
        "job_id": str(job_id),
        "page_num": page_num,
    }


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
        _resolve_text_font,
        create_paddle_engine,
        extract_paddle_lines,
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
        job_path = job.nfs_job_path or _nfs_job_path(job.job_id)
        temp_dir = os.path.join(job_path, "temp", document_id)
        os.makedirs(temp_dir, exist_ok=True)
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
        images = list(iter_source_images(
            source_path, page_num, page_num,
            job.source_type,
        ))
        if not images:
            return _record_page_failed(
                job, page_num, document_id, hostname,
                "No image extracted",
            )

        img = images[0]
        img_array = None
        if np is not None:
            try:
                img_array = np.array(img)
            except Exception:
                img_array = None

        # Try PaddleOCR
        ocr_method = "PaddleOCR"
        ocr_lines = []
        confidence = 0.0
        page_text = ""

        try:
            lang_code = job.detected_language or "en"
            device = "gpu" if _gpu_available() else "cpu"
            engine = create_paddle_engine(lang_code, device)
            result = engine.ocr(img_array) if img_array is not None else None
            ocr_lines = extract_paddle_lines(result)
        except Exception as exc:
            logger.warning("PaddleOCR failed for page %d: %s", page_num, exc)
            ocr_lines = []

        # Tesseract fallback
        if not ocr_lines:
            ocr_method = "Tesseract"
            try:
                import pytesseract
                page_text = pytesseract.image_to_string(img)
            except Exception as exc:
                logger.warning("Tesseract fallback failed for page %d: %s", page_num, exc)
                ocr_method = "ImageOnly"

        # Compute confidence and text from PaddleOCR results
        if ocr_lines:
            page_text = "\n".join(txt for txt, _, _ in ocr_lines)
            confidences = [c for _, _, c in ocr_lines if c > 0]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Build output PDF page
        img_bytes = img_to_bytes(img)
        page_pdf = fitz.open()
        page_rect = fitz.Rect(0, 0, img.width, img.height)
        pdf_page = page_pdf.new_page(width=img.width, height=img.height)
        pdf_page.insert_image(page_rect, stream=img_bytes)

        # Insert text overlay
        if ocr_lines:
            for txt, box, _ in ocr_lines:
                if box:
                    insert_text_line(pdf_page, txt, box, lang_code=lang_code)
        elif page_text and ocr_method == "Tesseract":
            fontname, fontfile = _resolve_text_font(lang_code)
            font_kwargs = {"fontname": fontname}
            if fontfile:
                font_kwargs["fontfile"] = fontfile
            pdf_page.insert_text((72, 72), page_text, fontsize=1,
                                 render_mode=3, **font_kwargs)

        # Save temp page PDF
        temp_pdf_path = os.path.join(temp_dir, f"{page_num}.pdf")
        page_pdf.save(temp_pdf_path)
        page_pdf.close()

        # Save temp page text
        temp_text_path = os.path.join(temp_dir, f"{page_num}.txt")
        with open(temp_text_path, "w", encoding="utf-8") as f:
            f.write(page_text)

        # Upload to backend if S3 mode
        if storage_mode == "s3":
            try:
                pdf_key = _job_storage_key(job.job_id, f"temp/{document_id}/{page_num}.pdf")
                text_key = _job_storage_key(job.job_id, f"temp/{document_id}/{page_num}.txt")
                if presigned_urls and presigned_urls.get("page_pdf_put"):
                    from .presigned_io import upload_presigned
                    upload_presigned(temp_pdf_path, presigned_urls["page_pdf_put"])
                    upload_presigned(temp_text_path, presigned_urls["page_text_put"])
                else:
                    backend.upload_file(temp_pdf_path, pdf_key)
                    backend.upload_file(temp_text_path, text_key)
                # Use S3 keys as the temp paths for DB record
                temp_pdf_path = pdf_key
                temp_text_path = text_key
            except Exception as exc:
                logger.error("Failed to upload page %d artifacts to S3: %s", page_num, exc)
                raise

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Determine status
        if ocr_method == "ImageOnly":
            status = "image_only"
        elif ocr_method == "Tesseract":
            status = "fallback"
        else:
            status = "ok"

        # Record page result in DB
        PageResult.objects.update_or_create(
            job=job, page_num=page_num,
            defaults={
                "document_id": document_id,
                "ocr_method": ocr_method,
                "ocr_language": job.detected_language or "en",
                "ocr_confidence": confidence,
                "text_length": len(page_text),
                "status": status,
                "worker_hostname": hostname,
                "processing_time_ms": elapsed_ms,
                "temp_pdf_path": temp_pdf_path,
            },
        )

        # Record custody event
        custody_event_type = {
            "PaddleOCR": "ocr_primary",
            "Tesseract": "ocr_fallback",
            "ImageOnly": "ocr_image_only",
        }.get(ocr_method, "ocr_primary")

        _record_custody_event(
            job, document_id, custody_event_type,
            data={
                "page_num": page_num,
                "method": ocr_method,
                "confidence": round(confidence, 4),
                "text_length": len(page_text),
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


def _record_page_failed(job, page_num, document_id, hostname, error_msg):
    """Record a failed page result."""
    PageResult.objects.update_or_create(
        job=job, page_num=page_num,
        defaults={
            "document_id": document_id,
            "status": "failed",
            "worker_hostname": hostname,
        },
    )
    _record_custody_event(
        job, document_id, "processing_failed",
        data={"page_num": page_num, "error": error_msg},
        worker_hostname=hostname,
    )
    return {"page_num": page_num, "status": "failed", "error": error_msg}


def _gpu_available():
    """Check if GPU is available for PaddleOCR."""
    try:
        import paddle
        return paddle.device.is_compiled_with_cuda()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CPU General Queue Tasks
# ---------------------------------------------------------------------------

@shared_task(bind=True, name="jobs.tasks.compress_pdf",
             queue="cpu_general", acks_late=True, reject_on_worker_lost=True)
def compress_pdf(self, job_id, presigned_urls=None):
    """Run Ghostscript optimization on the assembled PDF.

    Args:
        presigned_urls: Optional dict with pdf_get and pdf_put presigned URLs.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    hostname = socket.gethostname()

    # Setup paths based on storage backend
    work_dir = ""
    if storage_mode == "nfs":
        job_path = job.nfs_job_path or _nfs_job_path(job_id)
        pdf_path = os.path.join(job_path, "output", "EXPORT", "PDF", f"{base_name}.pdf")
    else:
        # S3 mode: download PDF to temp directory
        import tempfile
        work_dir = tempfile.mkdtemp(prefix=f"compress_{job_id}_")
        pdf_path = os.path.join(work_dir, f"{base_name}.pdf")
        pdf_key = _job_storage_key(job_id, f"output/EXPORT/PDF/{base_name}.pdf")
        try:
            if presigned_urls and presigned_urls.get("pdf_get"):
                from .presigned_io import download_presigned
                download_presigned(presigned_urls["pdf_get"], pdf_path)
            else:
                if not backend.exists(pdf_key):
                    logger.warning("No PDF to compress for job %s", job_id)
                    if os.path.isdir(work_dir):
                        shutil.rmtree(work_dir, ignore_errors=True)
                    return {"status": "skipped", "reason": "no_pdf"}
                backend.download_file(pdf_key, pdf_path)
        except Exception as exc:
            logger.error("S3 PDF download failed for job %s: %s", job_id, exc)
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            Worker.objects.filter(hostname=hostname).update(
                tasks_failed=db_models.F("tasks_failed") + 1,
            )
            return {"status": "error", "message": f"S3 download failed: {exc}"}

    if not os.path.isfile(pdf_path):
        logger.warning("No PDF to compress for job %s", job_id)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "skipped", "reason": "no_pdf"}

    try:
        from optimize_pdfs import optimize_pdf
        optimize_pdf(pdf_path, quality="/prepress")
        logger.info("Compressed PDF for job %s", job_id)

        # Upload back to S3 if needed
        if storage_mode == "s3":
            if presigned_urls and presigned_urls.get("pdf_put"):
                from .presigned_io import upload_presigned
                upload_presigned(pdf_path, presigned_urls["pdf_put"])
            else:
                backend.upload_file(pdf_path, pdf_key)
    except Exception as exc:
        logger.error("PDF compression failed for job %s: %s", job_id, exc)
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        return {"status": "error", "message": str(exc)}
    finally:
        # Cleanup temp directory for S3 mode
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

    # Record custody event
    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]
    _record_custody_event(
        job, document_id, "compression_complete",
        data={"pdf_path": pdf_path if storage_mode == "nfs" else pdf_key},
        worker_hostname=hostname,
    )

    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )

    return {"status": "compressed", "job_id": str(job_id)}


@shared_task(bind=True, name="jobs.tasks.extract_entities",
             queue="cpu_general", acks_late=True, reject_on_worker_lost=True)
def extract_entities(self, job_id, presigned_urls=None):
    """Run Named Entity Recognition on extracted text.

    Args:
        presigned_urls: Optional dict with text_get and ner_put presigned URLs.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    hostname = socket.gethostname()

    # Setup paths based on storage backend
    work_dir = ""
    if storage_mode == "nfs":
        job_path = job.nfs_job_path or _nfs_job_path(job_id)
        text_path = os.path.join(job_path, "output", "EXPORT", "TEXT", f"{base_name}.txt")
        ner_dir = os.path.join(job_path, "output", "EXPORT", "NER")
        os.makedirs(ner_dir, exist_ok=True)
        ner_path = os.path.join(ner_dir, f"{base_name}.ner.json")
    else:
        # S3 mode: download text to temp directory
        import tempfile
        work_dir = tempfile.mkdtemp(prefix=f"ner_{job_id}_")
        text_path = os.path.join(work_dir, f"{base_name}.txt")
        ner_path = os.path.join(work_dir, f"{base_name}.ner.json")
        text_key = _job_storage_key(job_id, f"output/EXPORT/TEXT/{base_name}.txt")
        try:
            if presigned_urls and presigned_urls.get("text_get"):
                from .presigned_io import download_presigned
                download_presigned(presigned_urls["text_get"], text_path)
            else:
                if not backend.exists(text_key):
                    logger.warning("No text file for NER for job %s", job_id)
                    if os.path.isdir(work_dir):
                        shutil.rmtree(work_dir, ignore_errors=True)
                    return {"status": "skipped", "reason": "no_text"}
                backend.download_file(text_key, text_path)
        except Exception as exc:
            logger.error("S3 text download failed for job %s: %s", job_id, exc)
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            Worker.objects.filter(hostname=hostname).update(
                tasks_failed=db_models.F("tasks_failed") + 1,
            )
            return {"status": "error", "message": f"S3 download failed: {exc}"}

    if not os.path.isfile(text_path):
        logger.warning("No text file for NER for job %s", job_id)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "skipped", "reason": "no_text"}

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "error", "message": str(exc)}

    try:
        from ner import extract_entities as run_ner
        entities = run_ner(text)
    except ImportError:
        logger.warning("NER module not available")
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "skipped", "reason": "ner_unavailable"}
    except Exception as exc:
        logger.error("NER failed for job %s: %s", job_id, exc)
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "error", "message": str(exc)}

    # Write NER results
    try:
        with open(ner_path, "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=2, default=str)

        # Upload to S3 if needed
        if storage_mode == "s3":
            if presigned_urls and presigned_urls.get("ner_put"):
                from .presigned_io import upload_presigned
                upload_presigned(ner_path, presigned_urls["ner_put"])
            else:
                ner_key = _job_storage_key(job_id, f"output/EXPORT/NER/{base_name}.ner.json")
                backend.upload_file(ner_path, ner_key)
    except Exception as exc:
        logger.error("NER result write failed for job %s: %s", job_id, exc)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "error", "message": str(exc)}
    finally:
        # Cleanup temp directory for S3 mode
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )

    return {"status": "extracted", "job_id": str(job_id), "entity_count": len(entities)}


@shared_task(bind=True, name="jobs.tasks.extract_structured_data",
             queue="nlp_general", acks_late=True, reject_on_worker_lost=True)
def extract_structured_data(self, job_id, presigned_urls=None):
    """Run PaddleNLP-based structured data extraction on the OCR output text.
    
    This fulfills the Phase 9 worker isolation design, decoupling heavy NLP
    extraction from the primary OCR workers so they can be versioned independently.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    if not job.settings_json.get("enable_docintel", False):
        # If extraction wasn't requested, just skip it.
        return {"status": "skipped", "reason": "not_requested"}

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    hostname = socket.gethostname()

    # Setup paths based on storage backend
    work_dir = ""
    if storage_mode == "nfs":
        job_path = job.nfs_job_path or _nfs_job_path(job_id)
        text_path = os.path.join(job_path, "output", "EXPORT", "TEXT", f"{base_name}.txt")
        ext_dir = os.path.join(job_path, "output", "EXPORT", "EXTRACTION")
        os.makedirs(ext_dir, exist_ok=True)
        ext_path = os.path.join(ext_dir, f"{base_name}.extraction.json")
    else:
        # S3 mode: download text to temp directory
        import tempfile
        work_dir = tempfile.mkdtemp(prefix=f"ext_{job_id}_")
        text_path = os.path.join(work_dir, f"{base_name}.txt")
        ext_path = os.path.join(work_dir, f"{base_name}.extraction.json")
        text_key = _job_storage_key(job_id, f"output/EXPORT/TEXT/{base_name}.txt")
        try:
            if presigned_urls and presigned_urls.get("text_get"):
                from .presigned_io import download_presigned
                download_presigned(presigned_urls["text_get"], text_path)
            else:
                if not backend.exists(text_key):
                    logger.warning("No text file for structured extraction for job %s", job_id)
                    if os.path.isdir(work_dir):
                        shutil.rmtree(work_dir, ignore_errors=True)
                    return {"status": "skipped", "reason": "no_text"}
                backend.download_file(text_key, text_path)
        except Exception as exc:
            logger.error("S3 text download failed for job %s: %s", job_id, exc)
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            Worker.objects.filter(hostname=hostname).update(
                tasks_failed=db_models.F("tasks_failed") + 1,
            )
            return {"status": "error", "message": f"S3 download failed: {exc}"}

    if not os.path.isfile(text_path):
        logger.warning("No text file for structured extraction for job %s", job_id)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "skipped", "reason": "no_text"}

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "error", "message": str(exc)}

    try:
        from extraction import (
            DocumentExtraction,
            extract_page_fields,
            finalize_extraction,
        )
        
        doc_ext = DocumentExtraction(document_id=str(job_id)[:16], source_file=base_name)
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
            
    except ImportError:
        logger.warning("Extraction module not available")
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "skipped", "reason": "extraction_unavailable"}
    except Exception as exc:
        logger.error("Structured extraction failed for job %s: %s", job_id, exc)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        return {"status": "error", "message": str(exc)}

    # Upload to S3 if needed
    if storage_mode == "s3":
        try:
            ext_key = _job_storage_key(job_id, f"output/EXPORT/EXTRACTION/{base_name}.extraction.json")
            if presigned_urls and presigned_urls.get("ext_put"):
                from .presigned_io import upload_presigned
                upload_presigned(ext_path, presigned_urls["ext_put"])
            else:
                backend.upload_file(ext_path, ext_key)
        except Exception as exc:
            logger.error("Failed to upload structured extraction result for job %s: %s", job_id, exc)
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            return {"status": "error", "message": f"S3 upload failed: {exc}"}
        finally:
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)

    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )
    return {"status": "extracted", "job_id": str(job_id)}
    """Run Named Entity Recognition on extracted text.

    Args:
        presigned_urls: Optional dict with text_get and ner_put presigned URLs.
    """
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    hostname = socket.gethostname()

    # Setup paths based on storage backend
    work_dir = ""
    if storage_mode == "nfs":
        job_path = job.nfs_job_path or _nfs_job_path(job_id)
        text_path = os.path.join(job_path, "output", "EXPORT", "TEXT", f"{base_name}.txt")
        ner_dir = os.path.join(job_path, "output", "EXPORT", "NER")
        os.makedirs(ner_dir, exist_ok=True)
        ner_path = os.path.join(ner_dir, f"{base_name}.ner.json")
    else:
        # S3 mode: download text to temp directory
        import tempfile
        work_dir = tempfile.mkdtemp(prefix=f"ner_{job_id}_")
        text_path = os.path.join(work_dir, f"{base_name}.txt")
        ner_path = os.path.join(work_dir, f"{base_name}.ner.json")
        text_key = _job_storage_key(job_id, f"output/EXPORT/TEXT/{base_name}.txt")
        try:
            if presigned_urls and presigned_urls.get("text_get"):
                from .presigned_io import download_presigned
                download_presigned(presigned_urls["text_get"], text_path)
            else:
                if not backend.exists(text_key):
                    logger.warning("No text file for NER for job %s", job_id)
                    if os.path.isdir(work_dir):
                        shutil.rmtree(work_dir, ignore_errors=True)
                    return {"status": "skipped", "reason": "no_text"}
                backend.download_file(text_key, text_path)
        except Exception as exc:
            logger.error("S3 text download failed for job %s: %s", job_id, exc)
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            Worker.objects.filter(hostname=hostname).update(
                tasks_failed=db_models.F("tasks_failed") + 1,
            )
            return {"status": "error", "message": f"S3 download failed: {exc}"}

    if not os.path.isfile(text_path):
        logger.warning("No text file for NER for job %s", job_id)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "skipped", "reason": "no_text"}

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "error", "message": str(exc)}

    try:
        from ner import extract_entities as run_ner
        entities = run_ner(text)
    except ImportError:
        logger.warning("NER module not available")
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "skipped", "reason": "ner_unavailable"}
    except Exception as exc:
        logger.error("NER failed for job %s: %s", job_id, exc)
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "error", "message": str(exc)}

    # Write NER results
    try:
        with open(ner_path, "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=2, default=str)

        # Upload to S3 if needed
        if storage_mode == "s3":
            if presigned_urls and presigned_urls.get("ner_put"):
                from .presigned_io import upload_presigned
                upload_presigned(ner_path, presigned_urls["ner_put"])
            else:
                ner_key = _job_storage_key(job_id, f"output/EXPORT/NER/{base_name}.ner.json")
                backend.upload_file(ner_path, ner_key)
    except Exception as exc:
        logger.error("NER result write failed for job %s: %s", job_id, exc)
        if storage_mode == "s3" and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "error", "message": str(exc)}
    finally:
        # Cleanup temp directory for S3 mode
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )

    return {"status": "extracted", "job_id": str(job_id), "entity_count": len(entities)}


@shared_task(bind=True, name="jobs.tasks.process_text_only",
             queue="cpu_general", acks_late=True, reject_on_worker_lost=True)
def process_text_only(self, job_id, presigned_urls=None):
    """Bypass OCR and extract text natively from PDF using PyMuPDF.
    
    Used when skip_ocr=True is passed via the API. Writes out the necessary
    temp page artifacts (text and PDF) and then dispatches assemble_document
    just like a normal OCR completion would.
    """
    import fitz

    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return {"status": "error", "message": f"Job {job_id} not found"}

    if job.status == Job.Status.CANCELLED:
        return {"status": "cancelled", "job_id": str(job_id)}

    hostname = socket.gethostname()
    Worker.objects.filter(hostname=hostname).update(
        status=Worker.Status.BUSY,
        current_task_id=self.request.id or "",
        last_heartbeat=timezone.now(),
    )

    backend = _get_backend_for_job(job)
    storage_mode = backend.backend_name
    document_id = job.source_hash[:16] if job.source_hash else str(job_id)[:16]

    # Setup paths
    work_dir = ""
    if storage_mode == "nfs":
        job_path = job.nfs_job_path or _nfs_job_path(job_id)
        source_path = os.path.join(job_path, "source", os.path.basename(job.source_file))
        temp_dir = os.path.join(job_path, "temp", document_id)
        os.makedirs(temp_dir, exist_ok=True)
    else:
        import tempfile
        work_dir = tempfile.mkdtemp(prefix=f"text_only_{job_id}_")
        source_path = os.path.join(work_dir, os.path.basename(job.source_file))
        temp_dir = os.path.join(work_dir, "temp", document_id)
        os.makedirs(temp_dir, exist_ok=True)
        try:
            source_key = _job_storage_key(job_id, f"source/{os.path.basename(job.source_file)}")
            backend.download_file(source_key, source_path)
        except Exception as exc:
            logger.error("S3 source download failed for job %s: %s", job_id, exc)
            job.status = Job.Status.FAILED
            job.error_message = f"S3 source download failed: {exc}"
            job.save(update_fields=["status", "error_message"])
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            return {"status": "error", "job_id": str(job_id), "message": f"Download failed: {exc}"}

    try:
        if job.source_type == "text":
            # For raw text/markdown/json files, just copy the text into page 1
            # and generate a dummy blank PDF page for structural compatibility
            with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
                page_text = f.read()

            page_num = 1
            temp_pdf_path = os.path.join(temp_dir, f"{page_num}.pdf")
            temp_text_path = os.path.join(temp_dir, f"{page_num}.txt")

            # Save dummy empty PDF page
            page_pdf = fitz.open()
            page_pdf.new_page(width=595, height=842) # A4 size
            page_pdf.save(temp_pdf_path)
            page_pdf.close()

            # Save text
            with open(temp_text_path, "w", encoding="utf-8") as f:
                f.write(page_text)

            if storage_mode == "s3":
                pdf_key = _job_storage_key(job_id, f"temp/{document_id}/{page_num}.pdf")
                text_key = _job_storage_key(job_id, f"temp/{document_id}/{page_num}.txt")
                backend.upload_file(temp_pdf_path, pdf_key)
                backend.upload_file(temp_text_path, text_key)

            # Record DB result
            PageResult.objects.update_or_create(
                job=job, page_num=page_num,
                defaults={
                    "document_id": document_id,
                    "ocr_method": "NativeText",
                    "ocr_language": job.detected_language or "en",
                    "ocr_confidence": 1.0,
                    "text_length": len(page_text),
                    "status": "ok",
                    "worker_hostname": hostname,
                    "processing_time_ms": 1,
                    "temp_pdf_path": pdf_key if storage_mode == "s3" else temp_pdf_path,
                },
            )
            
            job.pages_completed = page_num
            job.save(update_fields=["pages_completed"])
            
        elif job.source_type == "pdf":
            with fitz.open(source_path) as doc:
                for page_num in range(1, doc.page_count + 1):
                    page = doc[page_num - 1]
                    page_text = page.get_text()

                    temp_pdf_path = os.path.join(temp_dir, f"{page_num}.pdf")
                    temp_text_path = os.path.join(temp_dir, f"{page_num}.txt")

                    # Save single page PDF
                    page_pdf = fitz.open()
                    page_pdf.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
                    page_pdf.save(temp_pdf_path)
                    page_pdf.close()

                    # Save text
                    with open(temp_text_path, "w", encoding="utf-8") as f:
                        f.write(page_text)

                    # S3 upload
                    if storage_mode == "s3":
                        pdf_key = _job_storage_key(job_id, f"temp/{document_id}/{page_num}.pdf")
                        text_key = _job_storage_key(job_id, f"temp/{document_id}/{page_num}.txt")
                        backend.upload_file(temp_pdf_path, pdf_key)
                        backend.upload_file(temp_text_path, text_key)

                    # Record DB result
                    PageResult.objects.update_or_create(
                        job=job, page_num=page_num,
                        defaults={
                            "document_id": document_id,
                            "ocr_method": "NativePDF",
                            "ocr_language": job.detected_language or "en",
                            "ocr_confidence": 1.0,  # Native text is considered 100%
                            "text_length": len(page_text),
                            "status": "ok",
                            "worker_hostname": hostname,
                            "processing_time_ms": 10,
                            "temp_pdf_path": pdf_key if storage_mode == "s3" else temp_pdf_path,
                        },
                    )
                    
                    # Custody event
                    _record_custody_event(
                        job, document_id, "text_extracted_natively",
                        data={
                            "page_num": page_num,
                            "method": "NativePDF",
                            "text_length": len(page_text),
                        },
                        worker_hostname=hostname,
                    )

                    job.pages_completed = page_num
                    job.save(update_fields=["pages_completed"])
        else:
            raise ValueError(f"skip_ocr is not supported for {job.source_type} documents.")

    except Exception as exc:
        job.status = Job.Status.FAILED
        job.error_message = f"Processing failed: {exc}"
        job.save(update_fields=["status", "error_message"])
        logger.error("process_text_only failed for job %s: %s", job_id, exc)
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        return {"status": "error", "job_id": str(job_id), "message": str(exc)}
    finally:
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        Worker.objects.filter(hostname=hostname).update(
            status=Worker.Status.ONLINE,
            current_task_id="",
            last_heartbeat=timezone.now(),
        )

    Worker.objects.filter(hostname=hostname).update(
        tasks_completed=db_models.F("tasks_completed") + 1,
    )

    assemble_document.delay(str(job_id))
    return {
        "status": "processed",
        "job_id": str(job_id),
        "pages_processed": job.total_pages,
    }


# ---------------------------------------------------------------------------
# Worker Registration (called from Celery signals)
# ---------------------------------------------------------------------------

def register_worker(hostname, queues=None, capabilities=None,
                    gpu_available=False, gpu_model="", gpu_vram_mb=0,
                    gpu_index=None, cpu_cores=0, ram_mb=0,
                    pipeline_version=""):
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
