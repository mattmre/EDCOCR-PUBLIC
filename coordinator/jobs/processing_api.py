"""Authenticated processing-engine compatibility API over the durable coordinator."""

from __future__ import annotations

import errno
import hashlib
import hmac
import logging
import math
import os
import re
import stat
import tempfile
import uuid
from datetime import timedelta
from pathlib import PurePosixPath

from django.conf import settings
from django.core.cache import cache
from django.db import IntegrityError, connection, transaction
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from ocr_local.document_bundle import build_document_bundle

from .artifact_generations import (
    ArtifactPointerError,
    _validate_nfs_relative_path,
    resolve_active_artifacts,
)
from .models import CustodyEvent, Job, Worker
from .outbox import enqueue_task_dispatch, publish_task_dispatch_outbox
from .tasks import (
    _get_backend_for_job,
    _get_storage_backend,
    _nfs_job_path_for_job,
)

_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_DEFAULT_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_BUNDLE_TEXT_BYTES = 128 * 1024 * 1024
_API_INGEST_GENERATION = "0"
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
logger = logging.getLogger(__name__)


def _broker_ready() -> bool:
    from coordinator.celery import app as celery_app

    with celery_app.connection_for_read(connect_timeout=2) as broker_connection:
        broker_connection.connect()
    return True


def _cache_ready() -> bool:
    key = f"ocr:readiness:{uuid.uuid4().hex}"
    try:
        cache.set(key, "ready", timeout=10)
        return cache.get(key) == "ready"
    finally:
        cache.delete(key)


def _schema_ready() -> bool:
    expected = str(getattr(settings, "DATABASE_SCHEMA", "public"))
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT current_schema(), "
            "pg_has_role(CURRENT_USER, nspowner, 'USAGE') "
            "FROM pg_namespace WHERE nspname = %s",
            [expected],
        )
        row = cursor.fetchone()
    return bool(row and row[0] == expected and row[1])


def _json_error(status: int, code: str, message: str) -> JsonResponse:
    return JsonResponse({"detail": {"error": code, "message": message}}, status=status)


def _configured_api_key() -> str:
    return str(
        getattr(settings, "OCR_COORDINATOR_API_KEY", "")
        or os.environ.get("OCR_COORDINATOR_API_KEY", "")
        or getattr(settings, "OCR_API_KEY", "")
        or os.environ.get("OCR_API_KEY", "")
    ).strip()


def _authenticate(request):
    expected = _configured_api_key()
    if not expected:
        return _json_error(
            503, "auth_unavailable", "OCR coordinator authentication is not configured."
        )
    provided = request.META.get("HTTP_X_API_KEY", "")
    if not isinstance(provided, str) or not hmac.compare_digest(
        provided.encode("utf-8", errors="surrogatepass"),
        expected.encode("utf-8", errors="surrogatepass"),
    ):
        response = _json_error(401, "unauthorized", "A valid X-API-Key is required.")
        response["WWW-Authenticate"] = "ApiKey"
        return response
    return None


def _tenant(request) -> str | None:
    value = request.META.get("HTTP_X_TENANT_ID", "")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if _TENANT_RE.fullmatch(value) else None


def _positive_limit(name: str, default: int) -> int:
    try:
        value = int(getattr(settings, name, default))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _safe_filename(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    if value != os.path.basename(value) or value in {".", ".."}:
        return None
    encoded = value.encode("utf-8", errors="strict")
    return value if len(encoded) <= 255 else None


def _regular_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the inode attributes that must remain stable during a read."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _assert_regular_read_fd(fd: int, *, label: str) -> os.stat_result:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular file")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"{label} must not have multiple hard links")
    return metadata


def _read_regular_fd(fd: int, *, max_bytes: int, label: str) -> bytes:
    """Read a held regular-file descriptor with a size and mutation fence."""
    before = _assert_regular_read_fd(fd, label=label)
    if before.st_size > max_bytes:
        raise ValueError("ocr_text_too_large")

    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 1024 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    after = _assert_regular_read_fd(fd, label=label)
    if _regular_file_identity(before) != _regular_file_identity(after):
        raise RuntimeError(f"{label} changed during read")
    if len(raw) != before.st_size:
        raise RuntimeError(f"{label} changed during read")
    if len(raw) > max_bytes:
        raise ValueError("ocr_text_too_large")
    return raw


def _read_regular_file_nofollow(path: str, *, max_bytes: int) -> bytes:
    """Read the private S3 download file without following its final link."""
    fd = -1
    try:
        fd = os.open(path, _FILE_OPEN_FLAGS)
        return _read_regular_fd(fd, max_bytes=max_bytes, label="S3 OCR download")
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError("S3 OCR download must not be a symlink") from exc
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def _open_directory_path_nofollow(path: str) -> int:
    """Open an absolute directory path by walking from the filesystem root."""
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        raise RuntimeError("NFS job root must be an absolute path")
    if any(component in {os.curdir, os.pardir} for component in path.split(os.sep)):
        raise RuntimeError("NFS job root is not canonical")
    absolute = os.path.abspath(path)
    current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in absolute.split(os.sep):
            if component in {"", os.curdir}:
                continue
            if component == os.pardir:
                raise RuntimeError("NFS job root contains a parent traversal")
            try:
                child_fd = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=current_fd,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RuntimeError("NFS job root contains a symlink") from exc
                raise
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _read_nfs_locator(job_root: str, relative_path: str, *, max_bytes: int) -> bytes:
    """Read a validated locator below a held, authorized NFS job root."""
    relative = _validate_nfs_relative_path(relative_path)
    parts = PurePosixPath(relative).parts
    if not parts:
        raise ArtifactPointerError("NFS artifact locator is empty")

    parent_fd = _open_directory_path_nofollow(job_root)
    file_fd = -1
    try:
        for component in parts[:-1]:
            try:
                child_fd = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RuntimeError(
                        "NFS artifact locator contains a symlink"
                    ) from exc
                raise
            os.close(parent_fd)
            parent_fd = child_fd

        try:
            file_fd = os.open(
                parts[-1],
                _FILE_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RuntimeError("NFS artifact locator is a symlink") from exc
            raise
        return _read_regular_fd(
            file_fd,
            max_bytes=max_bytes,
            label="NFS OCR artifact",
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def _stage_upload(upload) -> tuple[str, int, str]:
    limit = _positive_limit(
        "OCR_COORDINATOR_MAX_UPLOAD_BYTES", _DEFAULT_MAX_UPLOAD_BYTES
    )
    digest = hashlib.sha256()
    size = 0
    chunks = iter(upload.chunks(chunk_size=1024 * 1024))
    prefix_chunks: list[bytes] = []
    prefix_size = 0
    while prefix_size < 4:
        chunk = next(chunks, b"")
        if not chunk:
            break
        prefix_chunks.append(chunk)
        prefix_size += len(chunk)
        if prefix_size > limit:
            break
    prefix = b"".join(prefix_chunks)
    if prefix.startswith(b"MZ") or prefix.startswith(b"\x7fELF"):
        raise ValueError("executable_refused")
    fd, path = tempfile.mkstemp(prefix="ocr-coordinator-upload-", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as output:

            def _write_chunk(chunk: bytes) -> None:
                nonlocal size
                if not chunk:
                    return
                size += len(chunk)
                if size > limit:
                    raise ValueError("upload_too_large")
                digest.update(chunk)
                output.write(chunk)

            for chunk in prefix_chunks:
                _write_chunk(chunk)
            for chunk in chunks:
                _write_chunk(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size < 1:
            raise ValueError("upload_empty")
        return path, size, digest.hexdigest()
    except Exception:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise


def _canonical_source_key(job_id, filename: str) -> str:
    return f"jobs/{job_id}/source/{filename}"


def _dispatch_ingest(job: Job) -> None:
    with transaction.atomic():
        locked = Job.objects.select_for_update().get(pk=job.pk)
        if not locked.submission_ready:
            raise RuntimeError("source admission is not committed")
        outbox = enqueue_task_dispatch(
            job=locked,
            stage="api_ingest",
            # API admission is the first dispatch edge and always owns
            # generation zero. Idempotent POST retries can arrive after
            # ingest has advanced the job to a new processing generation;
            # they must reuse this original contract rather than publishing a
            # second ingest that could overwrite the processing task lease.
            generation=_API_INGEST_GENERATION,
            task_name="jobs.tasks.ingest_document",
            args=[str(locked.job_id)],
            kwargs={"tenant_id": str(locked.tenant_id)},
            queue="coordinator",
        )
        dispatch_id = outbox.dispatch_id

        def publish_after_commit() -> None:
            publish_task_dispatch_outbox(dispatch_id=dispatch_id)

        # The request middleware may own an outer transaction. Defer broker
        # I/O until that transaction commits; if an unexpected callback error
        # occurs, the durable outbox remains available to the periodic relay.
        transaction.on_commit(publish_after_commit, robust=True)


@csrf_exempt
@require_http_methods(["POST"])
def submit_job(request):
    if auth_error := _authenticate(request):
        return auth_error
    tenant_id = _tenant(request)
    if tenant_id is None:
        return _json_error(400, "tenant_required", "A valid X-Tenant-ID is required.")
    raw_key = request.META.get("HTTP_IDEMPOTENCY_KEY", "")
    if (
        not isinstance(raw_key, str)
        or not raw_key.strip()
        or len(raw_key.encode("utf-8")) > 512
    ):
        return _json_error(
            400, "idempotency_key_required", "A bounded Idempotency-Key is required."
        )
    upload = request.FILES.get("file")
    filename = _safe_filename(getattr(upload, "name", None))
    if upload is None or filename is None:
        return _json_error(400, "file_required", "One safe multipart file is required.")
    key_hash = hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()
    try:
        staged_path, source_size, source_sha256 = _stage_upload(upload)
    except ValueError as exc:
        code = str(exc)
        status = 413 if code == "upload_too_large" else 400
        return _json_error(
            status, code, "The submitted file violates the upload contract."
        )
    try:
        backend = _get_storage_backend()
        if backend.backend_name != "s3":
            return _json_error(
                503,
                "durable_storage_required",
                "Coordinator API admission requires S3 storage.",
            )
        try:
            with transaction.atomic():
                try:
                    job, created = Job.objects.select_for_update().get_or_create(
                        tenant_id=tenant_id,
                        submission_key_hash=key_hash,
                        defaults={
                            "source_file": filename,
                            "source_hash": source_sha256,
                            "source_size_bytes": source_size,
                        },
                    )
                except IntegrityError:
                    job = Job.objects.select_for_update().get(
                        tenant_id=tenant_id, submission_key_hash=key_hash
                    )
                    created = False
                if (
                    job.source_file != filename
                    or job.source_hash != source_sha256
                    or job.source_size_bytes != source_size
                ):
                    return _json_error(
                        409,
                        "idempotency_conflict",
                        "The idempotency key is already bound to different source bytes.",
                    )
                object_key = _canonical_source_key(job.job_id, filename)
                if job.source_object_key and job.source_object_key != object_key:
                    return _json_error(
                        409,
                        "source_binding_conflict",
                        "The durable source binding changed.",
                    )
            if not job.submission_ready:
                backend.upload_file(staged_path, object_key)
                with transaction.atomic():
                    job = Job.objects.select_for_update().get(pk=job.pk)
                    if not job.submission_ready:
                        job.source_object_key = object_key
                        job.submission_ready = True
                        job.storage_backend_used = "s3"
                        job.save(
                            update_fields=[
                                "source_object_key",
                                "submission_ready",
                                "storage_backend_used",
                            ]
                        )
            _dispatch_ingest(job)
        except Exception:
            logger.exception("Durable OCR submission failed for tenant=%s", tenant_id)
            return _json_error(
                503,
                "submission_unavailable",
                "Durable OCR admission is temporarily unavailable.",
            )
        response = JsonResponse({"job_id": str(job.job_id), "status": job.status})
        response.status_code = 201 if created else 200
        return response
    finally:
        try:
            os.unlink(staged_path)
        except FileNotFoundError:
            pass


def _authorized_job(request, job_id):
    if auth_error := _authenticate(request):
        return None, auth_error
    tenant_id = _tenant(request)
    if tenant_id is None:
        return None, _json_error(
            400, "tenant_required", "A valid X-Tenant-ID is required."
        )
    try:
        return Job.objects.get(job_id=job_id, tenant_id=tenant_id), None
    except (Job.DoesNotExist, ValueError):
        return None, _json_error(404, "job_not_found", "OCR job was not found.")


@require_GET
def get_job(request, job_id):
    job, error = _authorized_job(request, job_id)
    if error:
        return error
    return JsonResponse(
        {
            "job_id": str(job.job_id),
            "status": job.status,
            "error_message": job.error_message,
            "source_file_sha256": job.source_hash,
            "total_pages": job.total_pages,
            "pages_completed": job.pages_completed,
            "pages_failed": job.pages_failed,
        }
    )


def _read_active_text(job: Job) -> tuple[bytes, object]:
    backend = _get_backend_for_job(job)
    backend_name = backend.backend_name
    job_root = _nfs_job_path_for_job(job) if backend_name == "nfs" else ""
    artifacts = resolve_active_artifacts(
        job_id=job.job_id,
        source_file=job.source_file,
        settings_json=job.settings_json,
        backend=backend_name,
        job_root=job_root,
    )
    limit = _positive_limit(
        "OCR_COORDINATOR_MAX_BUNDLE_TEXT_BYTES", _DEFAULT_MAX_BUNDLE_TEXT_BYTES
    )
    if artifacts.backend == "s3":
        fd, path = tempfile.mkstemp(prefix="ocr-coordinator-bundle-", suffix=".txt")
        os.close(fd)
        try:
            backend.download_file(artifacts.text.locator, path, max_bytes=limit)
            raw = _read_regular_file_nofollow(path, max_bytes=limit)
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Could not remove temporary OCR bundle %s", path)
    elif artifacts.backend == "nfs":
        raw = _read_nfs_locator(job_root, artifacts.text.relative_path, max_bytes=limit)
    else:
        raise ArtifactPointerError("unsupported artifact backend")
    if (
        artifacts.text.sha256
        and hashlib.sha256(raw).hexdigest() != artifacts.text.sha256
    ):
        raise ValueError("ocr_text_hash_mismatch")
    return raw, artifacts


@require_GET
def get_document_bundle(request, job_id):
    job, error = _authorized_job(request, job_id)
    if error:
        return error
    if job.status != Job.Status.COMPLETED:
        return _json_error(409, "job_not_complete", "OCR output is not yet complete.")
    try:
        raw, artifacts = _read_active_text(job)
    except (ArtifactPointerError, OSError, RuntimeError, ValueError):
        return _json_error(
            404, "output_unavailable", "The committed OCR output is unavailable."
        )
    text = raw.decode("utf-8", errors="replace")
    page_rows = list(job.page_results.order_by("page_num"))
    confidences = [
        row.ocr_confidence
        for row in page_rows
        if row.status in {"ok", "fallback"}
        and isinstance(row.ocr_provenance, dict)
        and row.ocr_provenance.get("assessment_version")
        and math.isfinite(row.ocr_confidence)
        and row.ocr_confidence >= 0
    ]
    confidence = sum(confidences) / len(confidences) if confidences else None
    custody_head = (
        CustodyEvent.objects.filter(job=job)
        .exclude(event_hash="")
        .order_by("-timestamp", "-id")
        .values_list("event_hash", flat=True)
        .first()
        or "n/a"
    )
    spans = []
    page_mapping_valid = bool(page_rows) and len(page_rows) == job.total_pages
    cursor = 0
    if page_mapping_valid:
        for index, row in enumerate(page_rows):
            provenance = row.ocr_provenance if isinstance(row.ocr_provenance, dict) else {}
            expected_hash = provenance.get("text_sha256")
            if (
                row.page_num != index + 1
                or not isinstance(row.text_length, int)
                or row.text_length < 0
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
            ):
                page_mapping_valid = False
                break
            page_text = text[cursor : cursor + row.text_length]
            cursor += row.text_length
            if hashlib.sha256(page_text.encode("utf-8")).hexdigest() != expected_hash:
                page_mapping_valid = False
                break
            spans.append(
                {
                    "span_id": f"ocr-page-{row.page_num}",
                    "page_number": row.page_num,
                    "text": page_text,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "language": provenance.get("detected_language") or "und",
                    "metadata": {
                        "source_artifact": artifacts.text.relative_path,
                        "location_scope": "page_text_artifact",
                        "bbox_scope": "unlocated",
                    },
                }
            )
            if (
                row.status in {"ok", "fallback"}
                and provenance.get("assessment_version")
                and math.isfinite(row.ocr_confidence)
                and row.ocr_confidence >= 0
            ):
                spans[-1]["confidence"] = row.ocr_confidence
            if index + 1 < len(page_rows):
                if text[cursor : cursor + 2] != "\n\n":
                    page_mapping_valid = False
                    break
                cursor += 2
        page_mapping_valid = page_mapping_valid and cursor == len(text)
    if not page_mapping_valid:
        spans = [
            {
                "span_id": "ocr-document-unlocated",
                "page_number": 1,  # DocumentBundle v1 requires an integer >= 1.
                "text": text,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "language": job.detected_language or "und",
                "metadata": {
                    "source_artifact": artifacts.text.relative_path,
                    "location_scope": "document_unlocated",
                    "page_number_is_schema_placeholder": True,
                },
            }
        ]
    if confidence is not None and not page_mapping_valid:
        spans[0]["confidence"] = confidence
    page_quality = []
    for row in page_rows:
        provenance = row.ocr_provenance if isinstance(row.ocr_provenance, dict) else {}
        assessed = bool(provenance.get("assessment_version"))
        page_quality.append(
            {
                "page_number": row.page_num,
                "status": row.status,
                "assessment_state": provenance.get("assessment_state", "not_assessed")
                if assessed
                else "not_assessed",
                "assessment_version": provenance.get("assessment_version", ""),
                "needs_review": bool(row.needs_review or not assessed),
                "model_language": row.ocr_language,
                "provenance": provenance,
            }
        )
    bundle = build_document_bundle(
        document_id=str(job.job_id),
        source_file_name=job.source_file,
        source_file_sha256=job.source_hash,
        spans=spans,
        source_ocr_sha256=hashlib.sha256(raw).hexdigest(),
        language_metadata={
            "primary_language": job.detected_language or "und",
            "detected_languages": [job.detected_language or "und"],
            "source": "ocr_coordinator",
        },
        ocr_engine_metadata={
            "engine_id": "ocr_local.coordinator",
            "engine_version": str(
                job.result_summary.get("pipeline_version", "coordinator")
            ),
            "job_id": str(job.job_id),
            "average_confidence": confidence,
            "text_location_scope": (
                "verified_page_artifacts" if page_mapping_valid else "document_unlocated"
            ),
            "pages_needing_review": sum(
                page["needs_review"] for page in page_quality
            ),
            "page_quality": page_quality,
        },
        custody_chain_head=custody_head,
        artifact_manifest={
            "artifacts": [
                {
                    "artifact_id": "ocr_text",
                    "artifact_type": "ocr_text",
                    "path": artifacts.text.relative_path,
                    "size_bytes": len(raw),
                    "mime_type": "text/plain",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            ]
        },
        validate=True,
    )
    return JsonResponse(bundle)


@require_GET
def detailed_health(request):
    if auth_error := _authenticate(request):
        return auth_error
    if _tenant(request) is None:
        return _json_error(400, "tenant_required", "A valid X-Tenant-ID is required.")
    checks = {
        "database": {"status": "down"},
        "schema": {"status": "down"},
        "broker": {"status": "down"},
        "cache": {"status": "down"},
        "storage": {"status": "down"},
        "workers": {"status": "down"},
    }
    try:
        Job.objects.only("job_id").order_by("job_id").first()
        checks["database"]["status"] = "healthy"
    except Exception:
        pass
    try:
        checks["schema"]["status"] = "healthy" if _schema_ready() else "down"
    except Exception:
        pass
    try:
        checks["broker"]["status"] = "healthy" if _broker_ready() else "down"
    except Exception:
        pass
    try:
        checks["cache"]["status"] = "healthy" if _cache_ready() else "down"
    except Exception:
        pass
    try:
        backend = _get_storage_backend()
        if backend.backend_name == "s3":
            checks["storage"]["status"] = "healthy"
        elif backend.backend_name == "nfs":
            root_fd = os.open(
                backend.root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fstat(root_fd)
            finally:
                os.close(root_fd)
            checks["storage"]["status"] = "healthy"
    except Exception:
        pass
    try:
        worker_available = Worker.objects.filter(
            status__in=["online", "busy"],
            last_heartbeat__gte=timezone.now() - timedelta(minutes=2),
        ).exists()
        allow_zero_workers = os.environ.get(
            "OCR_COORDINATOR_ALLOW_ZERO_WORKER_READINESS", ""
        ).strip().lower() in {"1", "true", "yes"}
        checks["workers"]["required"] = not allow_zero_workers
        checks["workers"]["status"] = (
            "healthy"
            if worker_available
            else "not_required"
            if allow_zero_workers
            else "down"
        )
    except Exception:
        pass
    ready = all(
        check["status"] in {"healthy", "not_required"} for check in checks.values()
    )
    return JsonResponse(
        {
            "status": "healthy" if ready else "down",
            "readiness": "ready" if ready else "not_ready",
            "checks": checks,
        },
        status=200 if ready else 503,
    )
