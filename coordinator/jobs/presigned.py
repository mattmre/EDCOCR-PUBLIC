"""Presigned URL generation for credential-free worker access.

When S3_USE_PRESIGNED_URLS=true, the coordinator generates short-lived
presigned GET/PUT URLs and stores them in encrypted, job-bound capability
rows. Workers receive only opaque capability identifiers via Celery and use
HTTP GET/PUT directly -- no S3 credentials required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from django.conf import settings
from django.utils import timezone

if TYPE_CHECKING:
    from .models import Job
    from .storage import StorageBackend

logger = logging.getLogger(__name__)

_MAX_SEALED_TASK_TOKEN_BYTES = 128 * 1024
_PRESIGNED_TASK_TOKEN_PREFIX = "v1."
_STAGE_OPERATIONS = frozenset({"render_page", "infer_page", "finalize_page"})
_STAGE_MARKER_KEYS = frozenset({
    "_stage_contract", "stage_id", "stage_page_num", "stage_generation",
    "stage_tenant_id", "stage_source_sha256", "stage_raster_locator",
    "stage_raster_sha256",
})


def _presigned_task_fernet():
    """Return the explicit Fernet used to keep bearer URLs out of Celery."""
    raw_key = os.environ.get("S3_PRESIGNED_TASK_KEY", "").strip()
    if not raw_key:
        raw_key = str(getattr(settings, "S3_PRESIGNED_TASK_KEY", "") or "").strip()
    if not raw_key:
        raise RuntimeError(
            "S3_PRESIGNED_TASK_KEY is required when presigned URLs cross a broker"
        )
    try:
        from cryptography.fernet import Fernet

        return Fernet(raw_key.encode("ascii"))
    except (ImportError, ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError("S3_PRESIGNED_TASK_KEY must be a valid Fernet key") from exc


def seal_presigned_urls(urls: dict[str, str] | None) -> str | None:
    """Encrypt a presigned URL map before storing it in a capability row."""
    if urls is None or isinstance(urls, str):
        return urls
    if not isinstance(urls, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in urls.items()
    ):
        raise ValueError("presigned task payload must be a string-keyed URL map")
    payload = json.dumps(urls, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > _MAX_SEALED_TASK_TOKEN_BYTES:
        raise ValueError("presigned task payload exceeds the broker bound")
    token = _presigned_task_fernet().encrypt(payload).decode("ascii")
    if len(token) + len(_PRESIGNED_TASK_TOKEN_PREFIX) > _MAX_SEALED_TASK_TOKEN_BYTES:
        raise ValueError("sealed presigned task payload exceeds the broker bound")
    return _PRESIGNED_TASK_TOKEN_PREFIX + token


def open_presigned_urls(value: dict[str, str] | str | None) -> dict[str, str] | None:
    """Decrypt a sealed task payload.

    Raw dictionaries are intentionally rejected.  A dictionary in a Celery
    task argument is already a cleartext bearer-URL payload and can be read by
    any broker observer; old lab callers must re-submit sealed task tokens.
    """
    if value is None:
        return value
    if isinstance(value, dict):
        raise ValueError("presigned task payload must be a sealed token")
    if not isinstance(value, str) or not value.startswith(_PRESIGNED_TASK_TOKEN_PREFIX):
        raise ValueError("presigned task payload is not a sealed token")
    if len(value.encode("utf-8")) > _MAX_SEALED_TASK_TOKEN_BYTES:
        raise ValueError("sealed presigned task payload exceeds the broker bound")
    try:
        payload = _presigned_task_fernet().decrypt(
            value[len(_PRESIGNED_TASK_TOKEN_PREFIX) :].encode("ascii")
        )
        decoded = json.loads(payload)
    except Exception as exc:
        raise ValueError("sealed presigned task payload could not be opened") from exc
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in decoded.items()
    ):
        raise ValueError("sealed presigned task payload is not a URL map")
    return decoded


def create_presigned_capability(
    job: "Job",
    operation: str,
    urls: dict[str, str] | None,
) -> str | None:
    """Persist URLs and return only an opaque identifier for Celery.

    The URL map is sealed before it is written to the job-owned capability
    row.  The row's foreign key, operation, and expiry are checked again by
    the worker at resolution time.  ``None`` keeps non-S3 and credentialed
    paths unchanged.
    """
    if urls is None:
        return None
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("presigned capability operation must be non-empty")
    from .models import PresignedTaskCapability

    token = seal_presigned_urls(urls)
    capability = PresignedTaskCapability.objects.create(
        job_id=job.job_id,
        operation=operation.strip(),
        dispatch_generation=int(job.dispatch_generation or 0),
        payload=token,
        expires_at=timezone.now() + timedelta(seconds=get_capability_expiry()),
    )
    return str(capability.capability_id)


def open_presigned_capability(
    job_id: str,
    capability_id: str | None,
    operation: str,
) -> dict[str, str] | None:
    """Resolve a worker capability only when it matches job and operation."""
    if capability_id is None:
        return None
    if not isinstance(capability_id, str) or not capability_id.strip():
        raise ValueError("presigned capability id must be a non-empty string")
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("presigned capability operation must be non-empty")
    try:
        capability_uuid = UUID(capability_id.strip())
    except ValueError as exc:
        raise ValueError("presigned capability id must be a UUID") from exc
    from django.db import transaction

    from .models import Job, PresignedTaskCapability

    with transaction.atomic():
        capability = (
            PresignedTaskCapability.objects.select_for_update()
            .select_related("job")
            .filter(
                capability_id=capability_uuid,
                job_id=job_id,
                operation=operation.strip(),
            )
            .only(
                "payload",
                "expires_at",
                "dispatch_generation",
                "job__status",
                "job__dispatch_generation",
            )
            .first()
        )
        if capability is None:
            raise ValueError("presigned capability is missing or job-bound incorrectly")
        if capability.job.status in {
            Job.Status.COMPLETED,
            Job.Status.FAILED,
            Job.Status.CANCELLED,
        }:
            raise ValueError("presigned capability belongs to a terminal job")
        if int(capability.dispatch_generation) != int(
            capability.job.dispatch_generation or 0
        ):
            raise ValueError("presigned capability dispatch generation was revoked")

        payload = open_presigned_urls(capability.payload)
        if capability.expires_at <= timezone.now():
            # Only the current constant-size JIT capability is renewable.
            # Legacy rows containing bearer URLs must fail closed because
            # extending the row cannot extend the URL signatures themselves.
            if payload and not (
                operation in _STAGE_OPERATIONS
                and payload.get("_stage_contract") == "v1"
                and set(payload) == _STAGE_MARKER_KEYS
            ):
                raise ValueError("expired legacy presigned capability cannot refresh")
            capability.expires_at = timezone.now() + timedelta(
                seconds=get_capability_expiry()
            )
            capability.save(update_fields=["expires_at"])
        return payload


def is_presigned_mode() -> bool:
    """Check if presigned URL mode is enabled."""
    return getattr(settings, "S3_USE_PRESIGNED_URLS", False)


def get_expiry() -> int:
    """Get configured presigned URL expiry in seconds.

    Keep the library fallback aligned with the coordinator setting (300s), so
    a partial settings import cannot silently create one-hour bearer URLs.
    """
    return getattr(settings, "S3_PRESIGNED_URL_EXPIRY", 300)


def get_capability_expiry() -> int:
    """Lifetime of the revocable opaque authorization, not of a bearer URL."""
    configured = int(getattr(settings, "S3_PRESIGNED_CAPABILITY_EXPIRY", 86400))
    return max(get_expiry(), min(configured, 604800))


def _job_storage_key(job_id: str, subpath: str = "") -> str:
    """Return storage key for job artifacts (mirrors tasks._job_storage_key)."""
    if subpath:
        return f"jobs/{job_id}/{subpath}"
    return f"jobs/{job_id}"


def _stage_raster_key(job: Job, stage) -> str:
    """Accept only the exact per-attempt key recorded by stage reservation."""
    prefix = _job_storage_key(
        str(job.job_id), f"stages/{int(stage.generation)}/{int(stage.page_num)}/"
    )
    locator = str(stage.raster_locator or "")
    if not re.fullmatch(re.escape(prefix) + r"raster-[0-9a-f]{32}\.png", locator):
        raise ValueError("staged raster locator is not an exact job/page/attempt key")
    return locator


def page_attempt_keys(job: Job, page_num: int, document_id: str, lease_token: str) -> dict[str, str]:
    """Derive output keys that an older page owner can never share with a new one."""
    if (
        not re.fullmatch(r"[0-9a-f]{32}", str(lease_token or ""))
        or not re.fullmatch(r"[a-f0-9]{1,64}", str(document_id or ""))
        or isinstance(page_num, bool)
        or not 1 <= int(page_num) <= int(job.total_pages or 0)
        or not re.fullmatch(r"[a-f0-9]{64}", str(job.source_hash or ""))
    ):
        raise ValueError("page output attempt identity is invalid")
    generation = int(job.dispatch_generation)
    page = int(page_num)
    root = _job_storage_key(
        str(job.job_id),
        f"temp/{document_id}/g{generation}/p{page}/l{lease_token}",
    )
    return {
        "page_pdf_key": f"{root}/{page}.pdf",
        "page_text_key": f"{root}/{page}.txt",
        "candidate_evidence_key": _job_storage_key(
            str(job.job_id),
            f"audit/ocr-candidates/g{generation}/p{page}/l{lease_token}.json",
        ),
    }


def text_only_attempt_keys(
    job: Job, page_num: int, document_id: str, document_lease_token: str,
) -> dict[str, str]:
    """Give each native-text document owner a private output namespace."""
    if (
        not re.fullmatch(r"[a-f0-9]{64}", str(job.source_hash or ""))
        or document_id != job.source_hash[:16]
        or isinstance(page_num, bool)
        or not 1 <= int(page_num) <= int(job.total_pages or 0)
        or not re.fullmatch(r"[0-9a-f]{32}", str(document_lease_token or ""))
    ):
        raise ValueError("text-only output attempt identity is invalid")
    generation = int(job.dispatch_generation)
    page = int(page_num)
    digest = hashlib.sha256(document_lease_token.encode("utf-8")).hexdigest()[:32]
    root = _job_storage_key(
        str(job.job_id), f"temp/{document_id}/g{generation}/d{digest}",
    )
    return {
        "page_pdf_key": f"{root}/{page}.pdf",
        "page_text_key": f"{root}/{page}.txt",
    }


def text_only_attempt_identity(job: Job) -> str:
    """Seal the parent, source and broker delivery observed by an execution."""
    fields = [
        str(job.job_id), int(job.dispatch_generation), str(job.tenant_id),
        str(job.source_hash), str(job.source_object_key), str(job.source_file),
        str(job.celery_task_id),
    ]
    return hashlib.sha256(
        json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_text_only_output_lease(job: Job, document_lease_token: str) -> None:
    """Sign only for a live, unique native-text execution epoch."""
    from .models import Job as JobModel

    job.refresh_from_db()
    if (
        job.status != JobModel.Status.PROCESSING
        or not re.fullmatch(r"[0-9a-f]{32}", str(document_lease_token or ""))
        or job.processing_attempt_token != document_lease_token
        or job.processing_attempt_identity != text_only_attempt_identity(job)
        or job.processing_lease_expires_at is None
        or job.processing_lease_expires_at <= timezone.now()
        or not re.fullmatch(r"[a-f0-9]{64}", str(job.source_hash or ""))
    ):
        raise ValueError("text-only output lease is missing, stale, or revoked")


def validate_page_output_lease(job: Job, page_num: int, lease_token: str) -> None:
    """Refuse signing for a page owner that lost its active processing lease."""
    from .models import Job as JobModel
    from .models import PageResult

    job.refresh_from_db()
    if job.status != JobModel.Status.PROCESSING:
        raise ValueError("page output job is not processing")
    if not re.fullmatch(r"[0-9a-f]{32}", str(lease_token or "")):
        raise ValueError("page output lease token is invalid")
    if not PageResult.objects.filter(
        job=job,
        page_num=page_num,
        status="processing",
        celery_task_id=lease_token,
        admitted_generation=job.dispatch_generation,
        lease_expires_at__gt=timezone.now(),
    ).exists():
        raise ValueError("page output lease is missing, expired, or different")


def create_stage_capability(job: Job, stage, operation: str) -> str:
    """Create one opaque stage grant carrying sealed, non-bearer row identity."""
    from .models import Job as JobModel

    if operation not in _STAGE_OPERATIONS:
        raise ValueError("unsupported staged presigned operation")
    if (
        job.status != JobModel.Status.PROCESSING
        or stage.job_id != job.pk
        or stage.generation != job.dispatch_generation
        or stage.tenant_id != job.tenant_id
        or stage.source_sha256 != job.source_hash
        or not 1 <= int(stage.page_num) <= int(job.total_pages or 0)
    ):
        raise ValueError("staged capability job, tenant, source, or generation changed")
    if operation != "render_page":
        _stage_raster_key(job, stage)
        if not re.fullmatch(r"[0-9a-f]{64}", str(stage.raster_sha256 or "")):
            raise ValueError("staged capability needs an immutable raster digest")
    markers = {
        "_stage_contract": "v1",
        "stage_id": str(stage.pk),
        "stage_page_num": str(stage.page_num),
        "stage_generation": str(stage.generation),
        "stage_tenant_id": str(stage.tenant_id),
        "stage_source_sha256": str(stage.source_sha256),
        "stage_raster_locator": str(stage.raster_locator or "") if operation != "render_page" else "",
        "stage_raster_sha256": str(stage.raster_sha256 or "") if operation != "render_page" else "",
    }
    return create_presigned_capability(job, operation, markers)


def validate_stage_capability(
    job: Job,
    stage,
    capability_id: str,
    operation: str,
    *,
    lease_token: str | None = None,
    raster_sha256: str | None = None,
) -> dict[str, str]:
    """Recheck current stage authority before signing each short bearer URL."""
    from .models import Job as JobModel

    if operation not in _STAGE_OPERATIONS:
        raise ValueError("unsupported staged presigned operation")
    job.refresh_from_db()
    stage.refresh_from_db()
    markers = open_presigned_capability(str(job.job_id), capability_id, operation)
    if (
        not isinstance(markers, dict)
        or markers.get("_stage_contract") != "v1"
        or set(markers) != _STAGE_MARKER_KEYS
    ):
        raise ValueError("staged capability contract is missing")
    current = {
        "stage_id": str(stage.pk),
        "stage_page_num": str(stage.page_num),
        "stage_generation": str(stage.generation),
        "stage_tenant_id": str(stage.tenant_id),
        "stage_source_sha256": str(stage.source_sha256),
    }
    if any(markers.get(name) != value for name, value in current.items()) or (
        job.status != JobModel.Status.PROCESSING
        or stage.job_id != job.pk
        or stage.generation != job.dispatch_generation
        or stage.tenant_id != job.tenant_id
        or stage.source_sha256 != job.source_hash
        or not 1 <= int(stage.page_num) <= int(job.total_pages or 0)
    ):
        raise ValueError("staged capability row, tenant, source, or generation changed")
    if operation in {"render_page", "infer_page"}:
        expected_status = "rendering" if operation == "render_page" else "infering"
        if (
            not lease_token
            or stage.status != expected_status
            or stage.lease_token != lease_token
            or stage.lease_expires_at is None
            or stage.lease_expires_at <= timezone.now()
        ):
            raise ValueError("staged capability lease is missing, expired, or different")
    elif stage.status != "inferred":
        raise ValueError("staged finalization is not ready")
    _stage_raster_key(job, stage)
    if operation != "render_page":
        if (
            not raster_sha256
            or raster_sha256 != stage.raster_sha256
            or markers.get("stage_raster_locator") != stage.raster_locator
            or markers.get("stage_raster_sha256") != stage.raster_sha256
        ):
            raise ValueError("staged raster digest or locator changed")
    return markers


def generate_stage_urls(
    backend: StorageBackend,
    job: Job,
    stage,
    operation: str,
    document_id: str | None = None,
    page_lease_token: str | None = None,
) -> dict[str, str]:
    """Sign only operation-specific keys after stage capability validation."""
    if operation not in _STAGE_OPERATIONS or backend.backend_name != "s3":
        raise ValueError("staged URLs require an S3 stage operation")
    if (
        stage.job_id != job.pk
        or stage.generation != job.dispatch_generation
        or stage.tenant_id != job.tenant_id
        or stage.source_sha256 != job.source_hash
        or not 1 <= int(stage.page_num) <= int(job.total_pages or 0)
    ):
        raise ValueError("staged URL job, tenant, source, or generation changed")
    raster_key = _stage_raster_key(job, stage)
    expiry = get_expiry()
    if operation == "render_page":
        source_name = os.path.basename(job.source_file)
        if not source_name:
            raise ValueError("staged source name is missing")
        return {
            "source_get": backend.presigned_url(
                _job_storage_key(str(job.job_id), f"source/{source_name}"), expiry
            ),
            "raster_put": backend.presigned_upload_url(raster_key, expiry),
            "raster_delete": backend.presigned_delete_url(raster_key, expiry),
        }
    if operation == "infer_page":
        return {"raster_get": backend.presigned_url(raster_key, expiry)}
    document_id = document_id or (job.source_hash[:16] if job.source_hash else str(job.job_id)[:16])
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", document_id):
        raise ValueError("staged document identifier is not canonical")
    if page_lease_token is None:
        raise ValueError("staged finalization requires an active page output lease")
    page_urls = generate_process_page_urls(
        backend, job, int(stage.page_num), document_id,
        page_lease_token=page_lease_token,
    )
    return {
        "source_get": page_urls["source_get"],
        "raster_get": backend.presigned_url(raster_key, expiry),
        "raster_delete": backend.presigned_delete_url(raster_key, expiry),
        **{name: page_urls[name] for name in (
            "page_pdf_put", "page_text_put", "page_pdf_delete", "page_text_delete",
            "candidate_evidence_put", "candidate_evidence_delete",
        )},
        **{name: page_urls[name] for name in (
            "page_pdf_key", "page_text_key", "candidate_evidence_key",
        )},
    }


def generate_process_page_urls(
    backend: StorageBackend,
    job: Job,
    page_num: int,
    document_id: str,
    *,
    page_lease_token: str | None = None,
    document_lease_token: str | None = None,
) -> dict[str, str]:
    """Generate presigned URLs needed by a process_page worker task.

    Returns dict with keys:
        source_get: GET URL for the source document
        page_pdf_put: PUT URL for uploading the page PDF result
        page_text_put: PUT URL for uploading the page text result
        page_pdf_delete: DELETE URL for reconciling a post-cancellation upload
        page_text_delete: DELETE URL for reconciling a post-cancellation upload
    """
    expiry = get_expiry()
    job_id = str(job.job_id)
    source_filename = os.path.basename(job.source_file)
    if page_lease_token is not None and document_lease_token is not None:
        raise ValueError("page and document output leases are mutually exclusive")
    if document_lease_token is not None:
        validate_text_only_output_lease(job, document_lease_token)
        attempt_keys = text_only_attempt_keys(job, page_num, document_id, document_lease_token)
        pdf_key = attempt_keys["page_pdf_key"]
        text_key = attempt_keys["page_text_key"]
        candidate_key = None
    elif page_lease_token is not None:
        validate_page_output_lease(job, page_num, page_lease_token)
        attempt_keys = page_attempt_keys(job, page_num, document_id, page_lease_token)
        pdf_key = attempt_keys["page_pdf_key"]
        text_key = attempt_keys["page_text_key"]
        candidate_key = attempt_keys["candidate_evidence_key"]
    else:
        # Retained only for the document-level legacy path. Page-task JIT
        # issuance must supply its active PageResult lease token.
        attempt_keys = {}
        pdf_key = _job_storage_key(job_id, f"temp/{document_id}/{page_num}.pdf")
        text_key = _job_storage_key(job_id, f"temp/{document_id}/{page_num}.txt")
        candidate_key = _job_storage_key(
            job_id,
            f"audit/ocr-candidates/g{int(job.dispatch_generation)}/p{int(page_num)}.json",
        )

    return {
        "source_get": backend.presigned_url(
            _job_storage_key(job_id, f"source/{source_filename}"),
            expiry,
        ),
        "page_pdf_put": backend.presigned_upload_url(
            pdf_key,
            expiry,
        ),
        "page_text_put": backend.presigned_upload_url(
            text_key,
            expiry,
        ),
        "page_pdf_delete": backend.presigned_delete_url(
            pdf_key,
            expiry,
        ),
        "page_text_delete": backend.presigned_delete_url(
            text_key,
            expiry,
        ),
        **(
            {
                "candidate_evidence_put": backend.presigned_upload_url(candidate_key, expiry),
                "candidate_evidence_delete": backend.presigned_delete_url(candidate_key, expiry),
            }
            if candidate_key is not None else {}
        ),
        **attempt_keys,
    }


def generate_text_only_source_urls(backend: StorageBackend, job: Job) -> dict[str, str]:
    """Mint only the native worker's source GET during its input phase."""
    if not re.fullmatch(r"[a-f0-9]{64}", str(job.source_hash or "")):
        raise ValueError("text-only source identity is invalid")
    source_filename = os.path.basename(job.source_file)
    source_key = _job_storage_key(str(job.job_id), f"source/{source_filename}")
    if not source_filename or (
        job.source_object_key and job.source_object_key != source_key
    ):
        raise ValueError("text-only source object key differs from sealed job identity")
    return {"source_get": backend.presigned_url(source_key, get_expiry())}


def generate_process_document_urls(
    backend: StorageBackend,
    job: Job,
    document_id: str,
) -> dict[str, str]:
    """Generate the bounded URL set for the single-worker document path."""
    expiry = get_expiry()
    job_id = str(job.job_id)
    source_filename = os.path.basename(job.source_file)
    urls = {
        "source_get": backend.presigned_url(
            _job_storage_key(job_id, f"source/{source_filename}"), expiry
        )
    }
    for page_num in range(1, int(job.total_pages or 0) + 1):
        candidate_key = _job_storage_key(
            job_id,
            f"audit/ocr-candidates/g{int(job.dispatch_generation)}/p{page_num}.json",
        )
        urls[f"page_{page_num}_pdf_put"] = backend.presigned_upload_url(
            _job_storage_key(job_id, f"temp/{document_id}/{page_num}.pdf"), expiry
        )
        urls[f"page_{page_num}_text_put"] = backend.presigned_upload_url(
            _job_storage_key(job_id, f"temp/{document_id}/{page_num}.txt"), expiry
        )
        urls[f"page_{page_num}_pdf_delete"] = backend.presigned_delete_url(
            _job_storage_key(job_id, f"temp/{document_id}/{page_num}.pdf"), expiry
        )
        urls[f"page_{page_num}_text_delete"] = backend.presigned_delete_url(
            _job_storage_key(job_id, f"temp/{document_id}/{page_num}.txt"), expiry
        )
        urls[f"page_{page_num}_candidate_evidence_put"] = (
            backend.presigned_upload_url(candidate_key, expiry)
        )
        urls[f"page_{page_num}_candidate_evidence_delete"] = (
            backend.presigned_delete_url(candidate_key, expiry)
        )
    return urls


def generate_layoutlm_urls(
    backend: StorageBackend, job: Job, page_num: int, document_id: str
) -> dict[str, str]:
    """Generate the complete credential-free LayoutLM page capability."""
    expiry = get_expiry()
    job_id = str(job.job_id)
    source_name = os.path.basename(job.source_file)
    base_name = os.path.splitext(source_name)[0]
    entities_name = f"{base_name}_p{page_num}.entities.json"
    from .models import PageResult

    page = PageResult.objects.filter(job=job, page_num=page_num).first()
    legacy_key = _job_storage_key(job_id, f"temp/{document_id}/{page_num}.txt")
    text_key = legacy_key
    if page is not None:
        if document_id != job.source_hash[:16] or page.document_id != document_id:
            raise ValueError("LayoutLM page document identity differs from source")
        if page.admitted_generation not in (None, job.dispatch_generation):
            raise ValueError("LayoutLM page belongs to an obsolete generation")
        if page.status not in {"ok", "fallback"}:
            raise ValueError("LayoutLM page text is not a committed OCR result")
        locator = (page.ocr_provenance or {}).get("temp_text_path", "")
        if locator:
            if not isinstance(locator, str):
                raise ValueError("LayoutLM committed text locator is malformed")
            if "/l" in locator:
                expected_key = page_attempt_keys(
                    job, page_num, document_id, page.celery_task_id,
                )["page_text_key"]
                if locator != expected_key:
                    raise ValueError("LayoutLM committed text locator differs from job/page lease")
            elif "/d" in locator:
                provenance = page.ocr_provenance or {}
                if (
                    page.ocr_method not in {"NativeText", "NativePDF"}
                    or provenance.get("source_sha256") != job.source_hash
                    or provenance.get("tenant_id") != job.tenant_id
                    or provenance.get("source_object_key") != job.source_object_key
                    or provenance.get("source_file") != job.source_file
                    or not re.fullmatch(r"[0-9a-f]{32}", str(page.celery_task_id or ""))
                    or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("document_attempt_identity") or ""))
                ):
                    raise ValueError("LayoutLM native winner source or attempt identity changed")
                native_keys = text_only_attempt_keys(
                    job, page_num, document_id, page.celery_task_id,
                )
                if (
                    locator != native_keys["page_text_key"]
                    or page.temp_pdf_path != native_keys["page_pdf_key"]
                ):
                    raise ValueError("LayoutLM committed native text locator differs from document attempt")
            elif locator != legacy_key:
                raise ValueError("LayoutLM committed text locator differs from legacy page key")
            text_key = locator
        elif page.ocr_method in {"NativeText", "NativePDF"} or any(
            marker in str(page.temp_pdf_path or "") for marker in ("/l", "/d")
        ):
            raise ValueError("LayoutLM leased page has no committed text locator")
    return {
        "source_get": backend.presigned_url(
            _job_storage_key(job_id, f"source/{source_name}"), expiry
        ),
        "text_get": backend.presigned_url(text_key, expiry),
        "text_key": text_key,
        "entities_put": backend.presigned_upload_url(
            _job_storage_key(job_id, f"output/EXPORT/ENTITIES/{entities_name}"),
            expiry,
        ),
        "entities_delete": backend.presigned_delete_url(
            _job_storage_key(job_id, f"output/EXPORT/ENTITIES/{entities_name}"),
            expiry,
        ),
    }


def _active_pdf_text_keys(job: Job) -> tuple[str, str]:
    """Resolve active PDF/TEXT object keys (generation pointer or legacy)."""
    from .artifact_generations import resolve_active_artifacts

    artifacts = resolve_active_artifacts(
        job_id=job.job_id,
        source_file=job.source_file,
        settings_json=job.settings_json,
        backend="s3",
    )
    return artifacts.pdf.locator, artifacts.text.locator


def generate_compress_pdf_urls(
    backend: StorageBackend,
    job: Job,
) -> dict[str, str]:
    """Generate presigned URLs needed by a compress_pdf worker task."""
    expiry = get_expiry()
    pdf_key, _ = _active_pdf_text_keys(job)

    urls = {
        "pdf_get": backend.presigned_url(pdf_key, expiry),
        # PUT targets the same active PDF key only for legacy jobs without a
        # generation pointer. Generation-aware compress stages a new key via
        # the storage backend and ignores a mismatched put URL.
        "pdf_put": backend.presigned_upload_url(pdf_key, expiry),
        "pdf_delete": backend.presigned_delete_url(pdf_key, expiry),
    }
    from .artifact_generations import (
        ACTIVE_OUTPUT_POINTER_KEY,
        generation_locations,
        make_generation_id,
    )

    if ACTIVE_OUTPUT_POINTER_KEY in (job.settings_json or {}):
        assembly = (
            job.custody_events.filter(event_type="assembly_complete")
            .order_by("-id")
            .first()
        )
        if assembly is None:
            raise ValueError("compression capability requires assembly custody")
        generation_id = make_generation_id(f"{job.job_id}:{assembly.id}:compress")
        pdf_location, _ = generation_locations(
            job_id=job.job_id,
            source_file=job.source_file,
            backend="s3",
            generation_id=generation_id,
            job_root="",
        )
        urls["generation_pdf_put"] = backend.presigned_upload_url(
            pdf_location.locator, expiry
        )
        urls["generation_pdf_delete"] = backend.presigned_delete_url(
            pdf_location.locator, expiry
        )
    return urls


def generate_extract_entities_urls(
    backend: StorageBackend,
    job: Job,
) -> dict[str, str]:
    """Generate presigned URLs needed by an extract_entities worker task."""
    expiry = get_expiry()
    job_id = str(job.job_id)
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    _, text_key = _active_pdf_text_keys(job)
    ner_key = _job_storage_key(job_id, f"output/EXPORT/NER/{base_name}.ner.json")

    return {
        "text_get": backend.presigned_url(text_key, expiry),
        "ner_put": backend.presigned_upload_url(ner_key, expiry),
        "ner_delete": backend.presigned_delete_url(ner_key, expiry),
    }


def generate_extract_structured_data_urls(
    backend: StorageBackend,
    job: Job,
) -> dict[str, str]:
    """Generate presigned URLs needed by an extract_structured_data worker task."""
    expiry = get_expiry()
    job_id = str(job.job_id)
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    _, text_key = _active_pdf_text_keys(job)
    ext_key = _job_storage_key(
        job_id, f"output/EXPORT/EXTRACTION/{base_name}.extraction.json"
    )

    return {
        "text_get": backend.presigned_url(text_key, expiry),
        "ext_put": backend.presigned_upload_url(ext_key, expiry),
        "ext_delete": backend.presigned_delete_url(ext_key, expiry),
    }
