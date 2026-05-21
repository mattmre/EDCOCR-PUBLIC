"""Celery task definitions for LayoutLMv3 inference.

Tasks run on the dedicated ``ocr_layoutlm`` queue so that LayoutLMv3
workers can be scaled independently of the primary OCR and NLP workers.

The task operates on a *pre-OCR'd* page: it reads the page's OCR text and
bounding boxes from the ``PageResult`` row (or the page temp text file),
reconstructs the page image, runs LayoutLMv3 token classification, and
writes the result as a ``.entities.json`` sidecar in the job's output
directory.

All imports of ``torch``, ``transformers``, and ``semantic_extraction``
are lazy so that the module can be imported (and tasks registered by
Celery) without those heavy dependencies present at import time.
"""

import json
import logging
import os
import shutil
import socket
import time

from celery import shared_task
from django.conf import settings
from django.db import models as db_models
from django.utils import timezone

from ocr_local.ml.layoutlm_model_registry import resolve_active_model_selection

from .layoutlm_config import (
    ENABLE_LAYOUTLM,
    LAYOUTLM_CONFIDENCE_THRESHOLD,
    LAYOUTLM_DEVICE,
    LAYOUTLM_MODEL_PATH,
    LAYOUTLM_QUEUE,
    LAYOUTLM_REGISTRY_DIR,
    LAYOUTLM_TASK_TIMEOUT,
)
from .models import CustodyEvent, Job, Worker
from .storage import CachedS3Backend, create_storage_backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (mirrors tasks.py private helpers)
# ---------------------------------------------------------------------------


def _get_storage_backend():
    """Create storage backend instance from Django settings."""
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


def _get_backend_for_job(job):
    """Get storage backend, honoring the backend locked at ingest time."""
    backend = _get_storage_backend()
    if job.storage_backend_used and backend.backend_name != job.storage_backend_used:
        logger.warning(
            "Backend mismatch for layoutlm job %s: ingested=%s, current=%s. Using ingested.",
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
                "Failed to create %s backend for layoutlm job %s, using current config",
                job.storage_backend_used, job.job_id,
            )
            backend = _get_storage_backend()
    return backend


def _nfs_job_path(job_id):
    """Return the canonical local workspace directory for a job."""
    return os.path.join(settings.NFS_ROOT, "jobs", str(job_id))


def _job_storage_key(job_id, subpath=""):
    """Return storage key for job artifacts."""
    if subpath:
        return f"jobs/{job_id}/{subpath}"
    return f"jobs/{job_id}"


def _record_custody_event(job, document_id, event_type, data=None,
                          worker_hostname=""):
    """Record a custody event in PostgreSQL."""
    CustodyEvent.objects.create(
        document_id=document_id,
        job=job,
        event_type=event_type,
        timestamp=timezone.now(),
        worker_hostname=worker_hostname,
        data=data or {},
    )


# ---------------------------------------------------------------------------
# LayoutLMv3 extraction task
# ---------------------------------------------------------------------------


@shared_task(
    bind=True,
    name="jobs.tasks_layoutlm.run_layoutlm_extraction",
    queue=LAYOUTLM_QUEUE,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=1,
    default_retry_delay=60,
    soft_time_limit=LAYOUTLM_TASK_TIMEOUT,
    time_limit=LAYOUTLM_TASK_TIMEOUT + 30,
)
def run_layoutlm_extraction(self, job_id, page_number, presigned_urls=None):
    """Run LayoutLMv3 token classification on a pre-OCR'd page.

    This task is dispatched *after* OCR completes for a page.  It reads the
    page's OCR text (from the temp text file), reconstructs the page image
    from the source document, and runs LayoutLMv3 inference to produce
    semantic entities.

    Args:
        job_id: UUID string of the job.
        page_number: 1-based page number.
        presigned_urls: Optional dict with presigned S3 URLs for
            source_get, text_get, entities_put.
    """
    start_time = time.monotonic()

    # --- Gate: check if LayoutLMv3 is enabled ---
    if not ENABLE_LAYOUTLM:
        return {
            "status": "skipped",
            "reason": "layoutlm_disabled",
            "job_id": str(job_id),
            "page_number": page_number,
        }

    # --- Load job ---
    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        logger.error("LayoutLMv3 task: Job %s not found", job_id)
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
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]

    work_dir = ""
    try:
        # --- Set up paths ---
        if storage_mode == "nfs":
            job_path = job.nfs_job_path or _nfs_job_path(job_id)
            text_path = os.path.join(
                job_path, "temp", document_id, f"{page_number}.txt"
            )
            source_path = os.path.join(
                job_path, "source", os.path.basename(job.source_file)
            )
            entities_dir = os.path.join(
                job_path, "output", "EXPORT", "ENTITIES"
            )
            os.makedirs(entities_dir, exist_ok=True)
        else:
            import tempfile
            work_dir = tempfile.mkdtemp(
                prefix=f"layoutlm_{job_id}_{page_number}_"
            )
            source_path = os.path.join(
                work_dir, os.path.basename(job.source_file)
            )
            text_path = os.path.join(work_dir, f"{page_number}.txt")
            entities_dir = os.path.join(work_dir, "entities")
            os.makedirs(entities_dir, exist_ok=True)

            # Download source document
            if presigned_urls and presigned_urls.get("source_get"):
                from .presigned_io import download_presigned
                download_presigned(presigned_urls["source_get"], source_path)
            else:
                source_key = _job_storage_key(
                    job_id,
                    f"source/{os.path.basename(job.source_file)}",
                )
                backend.download_file(source_key, source_path)

            # Download page text
            if presigned_urls and presigned_urls.get("text_get"):
                from .presigned_io import download_presigned
                download_presigned(presigned_urls["text_get"], text_path)
            else:
                text_key = _job_storage_key(
                    job_id,
                    f"temp/{document_id}/{page_number}.txt",
                )
                if backend.exists(text_key):
                    backend.download_file(text_key, text_path)

        # --- Read OCR text for word/box reconstruction ---
        page_text = ""
        if os.path.isfile(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                page_text = f.read()

        if not page_text.strip():
            logger.info(
                "LayoutLMv3: No text for job %s page %d, skipping",
                job_id, page_number,
            )
            return {
                "status": "skipped",
                "reason": "no_text",
                "job_id": str(job_id),
                "page_number": page_number,
            }

        # --- Render page image from source document ---
        page_image = _render_page_image(source_path, page_number, job.source_type)

        if page_image is None:
            logger.warning(
                "LayoutLMv3: Could not render image for job %s page %d",
                job_id, page_number,
            )
            return {
                "status": "skipped",
                "reason": "no_image",
                "job_id": str(job_id),
                "page_number": page_number,
            }

        # --- Build word-level tokens and dummy bounding boxes ---
        words = page_text.split()
        if not words:
            return {
                "status": "skipped",
                "reason": "no_words",
                "job_id": str(job_id),
                "page_number": page_number,
            }

        img_w, img_h = page_image.size
        # Distribute words evenly across the page as dummy bboxes.
        # Real bboxes would come from PaddleOCR line output stored in
        # PageResult or a dedicated sidecar; this is a reasonable fallback
        # that still gives LayoutLMv3 spatial signal.
        word_height = max(1, img_h // max(len(words), 1))
        boxes = []
        for i, _word in enumerate(words):
            y1 = min(i * word_height, img_h - 1)
            y2 = min(y1 + word_height, img_h)
            boxes.append([0, y1, img_w, y2])

        # --- Lazy import and run LayoutLMv3 extractor ---
        try:
            from semantic_extraction import LayoutLMv3Extractor
        except ImportError:
            logger.warning(
                "LayoutLMv3: semantic_extraction module not available"
            )
            return {
                "status": "skipped",
                "reason": "module_unavailable",
                "job_id": str(job_id),
                "page_number": page_number,
            }

        model_selection = resolve_active_model_selection(
            fallback_model_path=LAYOUTLM_MODEL_PATH,
            registry_dir=LAYOUTLM_REGISTRY_DIR,
        )

        if model_selection.source == "registry":
            logger.info(
                "LayoutLMv3 worker using active registry model %s -> %s",
                model_selection.active_model_spec,
                model_selection.model_path,
            )
            if model_selection.adapter_path:
                logger.info(
                    "LayoutLMv3 registry entry includes adapter_path=%s; "
                    "worker inference uses model_path only in this pass.",
                    model_selection.adapter_path,
                )

        extractor = LayoutLMv3Extractor(
            model_path=model_selection.model_path,
            device=LAYOUTLM_DEVICE if LAYOUTLM_DEVICE != "auto" else None,
        )

        entities = extractor.extract_entities(
            words=words,
            boxes=boxes,
            image=page_image,
            page_num=page_number,
        )

        # Filter by confidence threshold
        entities = [
            e for e in entities
            if e.confidence >= LAYOUTLM_CONFIDENCE_THRESHOLD
        ]

        processing_time = round(time.monotonic() - start_time, 4)

        # --- Write .entities.json sidecar ---
        entities_output = {
            "schema_version": "1.0",
            "source": "layoutlmv3",
            "model": model_selection.model_path,
            "model_source": model_selection.source,
            "active_model_spec": model_selection.active_model_spec,
            "job_id": str(job_id),
            "page_number": page_number,
            "document_id": document_id,
            "processing_time_seconds": processing_time,
            "confidence_threshold": LAYOUTLM_CONFIDENCE_THRESHOLD,
            "entity_count": len(entities),
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "field_type": e.field_type,
                    "confidence": e.confidence,
                    "bbox": e.bbox,
                    "page_num": e.page_num,
                }
                for e in entities
            ],
        }

        entities_filename = f"{base_name}_p{page_number}.entities.json"
        entities_path = os.path.join(entities_dir, entities_filename)

        with open(entities_path, "w", encoding="utf-8") as f:
            json.dump(entities_output, f, indent=2, ensure_ascii=False)

        # Upload to S3 if needed
        if storage_mode == "s3":
            if presigned_urls and presigned_urls.get("entities_put"):
                from .presigned_io import upload_presigned
                upload_presigned(entities_path, presigned_urls["entities_put"])
            else:
                entities_key = _job_storage_key(
                    job_id,
                    f"output/EXPORT/ENTITIES/{entities_filename}",
                )
                backend.upload_file(entities_path, entities_key)

        # --- Record custody event ---
        _record_custody_event(
            job, document_id, "layoutlm_extraction",
            data={
                "page_number": page_number,
                "entity_count": len(entities),
                "model": model_selection.model_path,
                "model_source": model_selection.source,
                "active_model_spec": model_selection.active_model_spec,
                "processing_time_seconds": processing_time,
            },
            worker_hostname=hostname,
        )

        # Update worker stats
        Worker.objects.filter(hostname=hostname).update(
            tasks_completed=db_models.F("tasks_completed") + 1,
        )

        logger.info(
            "LayoutLMv3: job %s page %d - %d entities in %.2fs",
            job_id, page_number, len(entities), processing_time,
        )

        return {
            "status": "completed",
            "job_id": str(job_id),
            "page_number": page_number,
            "entity_count": len(entities),
            "model": model_selection.model_path,
            "model_source": model_selection.source,
            "active_model_spec": model_selection.active_model_spec,
            "processing_time_seconds": processing_time,
        }

    except Exception as exc:
        logger.error(
            "LayoutLMv3 extraction failed for job %s page %d: %s",
            job_id, page_number, exc,
        )
        Worker.objects.filter(hostname=hostname).update(
            tasks_failed=db_models.F("tasks_failed") + 1,
        )
        _record_custody_event(
            job, document_id, "layoutlm_extraction_failed",
            data={
                "page_number": page_number,
                "error": str(exc),
            },
            worker_hostname=hostname,
        )
        raise
    finally:
        if storage_mode == "s3" and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Page image rendering helper
# ---------------------------------------------------------------------------


def _render_page_image(source_path, page_number, source_type):
    """Render a page image from the source document.

    Returns a PIL Image or None on failure. Handles both PDF and image
    source types.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not available for page image rendering")
        return None

    try:
        if source_type in ("image", None):
            # Single-page image source
            return Image.open(source_path).convert("RGB")

        if source_type == "pdf":
            try:
                import fitz
                with fitz.open(source_path) as doc:
                    if page_number < 1 or page_number > len(doc):
                        return None
                    page = doc[page_number - 1]
                    # Render at 150 DPI for LayoutLMv3 (lower than OCR's 300)
                    mat = fitz.Matrix(150 / 72, 150 / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes(
                        "RGB", (pix.width, pix.height), pix.samples
                    )
                    return img
            except ImportError:
                logger.warning("PyMuPDF not available for PDF rendering")
                return None

        return None

    except Exception as exc:
        logger.warning(
            "Failed to render page %d from %s: %s",
            page_number, source_path, exc,
        )
        return None
