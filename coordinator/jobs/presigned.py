"""Presigned URL generation for credential-free worker access.

When S3_USE_PRESIGNED_URLS=true, the coordinator generates short-lived
presigned GET/PUT URLs and passes them to workers via Celery task arguments.
Workers use HTTP GET/PUT directly -- no S3 credentials required.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from django.conf import settings

if TYPE_CHECKING:
    from .models import Job
    from .storage import StorageBackend

logger = logging.getLogger(__name__)


def is_presigned_mode() -> bool:
    """Check if presigned URL mode is enabled."""
    return getattr(settings, "S3_USE_PRESIGNED_URLS", False)


def get_expiry() -> int:
    """Get configured presigned URL expiry in seconds."""
    return getattr(settings, "S3_PRESIGNED_URL_EXPIRY", 3600)


def _job_storage_key(job_id: str, subpath: str = "") -> str:
    """Return storage key for job artifacts (mirrors tasks._job_storage_key)."""
    if subpath:
        return f"jobs/{job_id}/{subpath}"
    return f"jobs/{job_id}"


def generate_process_page_urls(
    backend: StorageBackend,
    job: Job,
    page_num: int,
    document_id: str,
) -> dict[str, str]:
    """Generate presigned URLs needed by a process_page worker task.

    Returns dict with keys:
        source_get: GET URL for the source document
        page_pdf_put: PUT URL for uploading the page PDF result
        page_text_put: PUT URL for uploading the page text result
    """
    expiry = get_expiry()
    job_id = str(job.job_id)
    source_filename = os.path.basename(job.source_file)

    return {
        "source_get": backend.presigned_url(
            _job_storage_key(job_id, f"source/{source_filename}"), expiry,
        ),
        "page_pdf_put": backend.presigned_upload_url(
            _job_storage_key(job_id, f"temp/{document_id}/{page_num}.pdf"), expiry,
        ),
        "page_text_put": backend.presigned_upload_url(
            _job_storage_key(job_id, f"temp/{document_id}/{page_num}.txt"), expiry,
        ),
    }


def generate_compress_pdf_urls(
    backend: StorageBackend,
    job: Job,
) -> dict[str, str]:
    """Generate presigned URLs needed by a compress_pdf worker task."""
    expiry = get_expiry()
    job_id = str(job.job_id)
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    pdf_key = _job_storage_key(job_id, f"output/EXPORT/PDF/{base_name}.pdf")

    return {
        "pdf_get": backend.presigned_url(pdf_key, expiry),
        "pdf_put": backend.presigned_upload_url(pdf_key, expiry),
    }


def generate_extract_entities_urls(
    backend: StorageBackend,
    job: Job,
) -> dict[str, str]:
    """Generate presigned URLs needed by an extract_entities worker task."""
    expiry = get_expiry()
    job_id = str(job.job_id)
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    text_key = _job_storage_key(job_id, f"output/EXPORT/TEXT/{base_name}.txt")
    ner_key = _job_storage_key(job_id, f"output/EXPORT/NER/{base_name}.ner.json")

    return {
        "text_get": backend.presigned_url(text_key, expiry),
        "ner_put": backend.presigned_upload_url(ner_key, expiry),
    }


def generate_extract_structured_data_urls(
    backend: StorageBackend,
    job: Job,
) -> dict[str, str]:
    """Generate presigned URLs needed by an extract_structured_data worker task."""
    expiry = get_expiry()
    job_id = str(job.job_id)
    base_name = os.path.splitext(os.path.basename(job.source_file))[0]
    text_key = _job_storage_key(job_id, f"output/EXPORT/TEXT/{base_name}.txt")
    ext_key = _job_storage_key(job_id, f"output/EXPORT/EXTRACTION/{base_name}.extraction.json")

    return {
        "text_get": backend.presigned_url(text_key, expiry),
        "ext_put": backend.presigned_upload_url(ext_key, expiry),
    }
