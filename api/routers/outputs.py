"""Output manifest and schema retrieval endpoints.

Provides:
- ``GET /api/v1/jobs/{job_id}/outputs`` -- list all outputs for a job
- ``GET /api/v1/jobs/{job_id}/outputs/{output_type}`` -- download a specific output
- ``GET /api/v1/schemas`` -- list available output schemas
- ``GET /api/v1/schemas/{output_type}`` -- get a specific schema definition
"""

from __future__ import annotations

import logging
from copy import deepcopy
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from api.database import get_session_factory
from api.identity import require_role
from api.limits import get_default_rate, limiter
from api.models import (
    ErrorResponse,
    OutputArtifactResponse,
    OutputManifestResponse,
    SchemaListItem,
    SchemaListResponse,
)
from api.output_manifest import (
    OutputManifest,
    VALID_OUTPUT_TYPES,
    build_manifest,
)
from api.path_safety import ensure_path_within_roots
from ocr_local.contracts import canonical_json_sha256
from ocr_local.document_bundle import build_document_bundle
from ocr_local.features.custody import verify_custody_file

logger = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12}$")

router = APIRouter(tags=["outputs"])


def _validate_job_id(job_id: str) -> None:
    """Raise 400 if job_id is malformed."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )


def _request_tenant_id(request: Request) -> str | None:
    """Return tenant scope for authenticated tenant keys."""
    return getattr(request.state, "tenant_id", None)


def _get_job_or_404(job_id: str, tenant_id: str | None = None):
    """Look up a job by ID, raising 404 if not found or wrong tenant."""
    from api.database import Job

    session = get_session_factory()()
    try:
        query = session.query(Job).filter(Job.job_id == job_id)
        if tenant_id is not None:
            query = query.filter(Job.tenant_id == tenant_id)
        job = query.first()
        if not job:
            raise HTTPException(
                status_code=404,
                detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
            )
        return job
    finally:
        session.close()


def _first_artifact(manifest: OutputManifest, output_type: str):
    return next(
        (artifact for artifact in manifest.artifacts if artifact.output_type == output_type),
        None,
    )


def _read_text_artifact(job, manifest: OutputManifest) -> tuple[str, str]:
    artifact = _first_artifact(manifest, "ocr_text")
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "ocr_text_not_found",
                "message": f"Job {job.job_id} has no OCR text artifact.",
            },
        )
    text_path = ensure_path_within_roots(
        path_value=str(Path(job.result_path) / artifact.relative_path),
        field_name="ocr_text_path",
        allowed_roots=[job.result_path],
    )
    if not text_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "ocr_text_missing",
                "message": f"OCR text artifact is missing on disk: {artifact.filename}",
            },
        )
    return text_path.read_text(encoding="utf-8", errors="replace"), artifact.relative_path


def _custody_summary(job, manifest: OutputManifest) -> dict[str, Any]:
    artifact = _first_artifact(manifest, "custody")
    if artifact is None:
        return {
            "available": False,
            "valid": None,
            "message": "No custody artifact found for this job.",
            "relative_path": None,
            "chain_head": "n/a",
        }
    custody_path = ensure_path_within_roots(
        path_value=str(Path(job.result_path) / artifact.relative_path),
        field_name="custody_path",
        allowed_roots=[job.result_path],
    )
    if not custody_path.is_file():
        return {
            "available": False,
            "valid": False,
            "message": f"Custody artifact is missing on disk: {artifact.filename}",
            "relative_path": artifact.relative_path,
            "chain_head": "n/a",
        }
    valid, message = verify_custody_file(str(custody_path))
    chain_head = "n/a"
    try:
        lines = [line for line in custody_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            import json

            chain_head = str(json.loads(lines[-1]).get("hash") or "n/a")
    except Exception:
        chain_head = "n/a"
    return {
        "available": True,
        "valid": valid,
        "message": message,
        "relative_path": artifact.relative_path,
        "chain_head": chain_head,
    }


def _document_bundle_from_job(job, request: Request) -> dict[str, Any]:
    if not job.result_path:
        raise HTTPException(
            status_code=404,
            detail={"error": "no_outputs", "message": f"Job {job.job_id} has no output directory."},
        )
    manifest = build_manifest(job.job_id, job.result_path)
    text, text_relative_path = _read_text_artifact(job, manifest)
    custody = _custody_summary(job, manifest)
    spans = [
        {
            "span_id": "ocr-text-1",
            "page_number": 1,
            "text": text,
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "language": "und",
            "metadata": {
                "source_artifact": text_relative_path,
                "extraction": "job_output_text",
            },
        }
    ]
    artifacts = [
        {
            "artifact_id": artifact.output_type,
            "artifact_type": artifact.output_type,
            "path": artifact.relative_path,
            "size_bytes": artifact.size_bytes,
            "mime_type": artifact.mime_type,
        }
        for artifact in manifest.artifacts
    ]
    if custody["available"]:
        artifacts.append(
            {
                "artifact_id": "custody_verification",
                "artifact_type": "custody_verification",
                "valid": custody["valid"],
                "message": custody["message"],
            }
        )
    bundle = build_document_bundle(
        document_id=job.job_id,
        source_file_name=job.source_file,
        source_file_sha256=job.source_hash,
        spans=spans,
        language_metadata={
            "primary_language": "und",
            "detected_languages": ["und"],
            "source": "job_output_text",
        },
        ocr_engine_metadata={
            "engine_id": "ocr_local.job_outputs",
            "engine_version": "job-output-export",
            "job_id": job.job_id,
        },
        custody_chain_head=custody["chain_head"],
        artifact_manifest={"artifacts": artifacts},
        validate=True,
    )
    bundle["artifact_manifest"]["document_bundle_url"] = str(
        request.url_for("get_job_document_bundle", job_id=job.job_id)
    )
    bundle["artifact_manifest"]["evidence_bundle_url"] = str(
        request.url_for("get_job_evidence_bundle", job_id=job.job_id)
    )
    return bundle


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/outputs -- List all outputs for a job
# ------------------------------------------------------------------


@router.get(
    "/api/v1/jobs/{job_id}/outputs",
    name="list_job_outputs",
    response_model=OutputManifestResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def list_job_outputs(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """List all output artifacts produced for a job.

    Returns a manifest with metadata for each artifact including type,
    filename, relative path, size, and schema version.
    """
    _validate_job_id(job_id)
    tenant_id = _request_tenant_id(request)
    job = _get_job_or_404(job_id, tenant_id=tenant_id)

    if not job.result_path:
        return OutputManifestResponse(job_id=job_id, artifacts=[], schema_versions={})

    manifest = build_manifest(job_id, job.result_path)
    return OutputManifestResponse(
        job_id=manifest.job_id,
        artifacts=[
            OutputArtifactResponse(
                output_type=a.output_type,
                filename=a.filename,
                relative_path=a.relative_path,
                size_bytes=a.size_bytes,
                mime_type=a.mime_type,
                schema_version=a.schema_version,
            )
            for a in manifest.artifacts
        ],
        schema_versions=manifest.schema_versions,
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/outputs/{output_type} -- Get specific output
# ------------------------------------------------------------------


@router.get(
    "/api/v1/jobs/{job_id}/outputs/{output_type}",
    name="get_job_output",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def get_job_output(
    request: Request,
    job_id: str,
    output_type: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Download a specific output artifact for a job.

    Returns the file content directly. JSON sidecar files are returned
    with ``application/json`` content type; PDFs as ``application/pdf``;
    text as ``text/plain``.
    """
    _validate_job_id(job_id)

    if output_type not in VALID_OUTPUT_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_output_type",
                "message": f"Unknown output type: {output_type}. Valid types: {sorted(VALID_OUTPUT_TYPES)}",
            },
        )

    tenant_id = _request_tenant_id(request)
    job = _get_job_or_404(job_id, tenant_id=tenant_id)

    if not job.result_path:
        raise HTTPException(
            status_code=404,
            detail={"error": "no_outputs", "message": f"Job {job_id} has no output directory."},
        )

    manifest = build_manifest(job_id, job.result_path)
    matching = [a for a in manifest.artifacts if a.output_type == output_type]

    if not matching:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "output_not_found",
                "message": f"No '{output_type}' output found for job {job_id}.",
            },
        )

    # Return the first matching artifact
    artifact = matching[0]
    file_path = Path(job.result_path) / artifact.relative_path

    # Path safety: ensure the resolved file is within the job's output dir
    safe_path = ensure_path_within_roots(
        path_value=str(file_path),
        field_name="artifact_path",
        allowed_roots=[job.result_path],
    )

    if not safe_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "file_not_found",
                "message": f"Artifact file not found on disk: {artifact.filename}",
            },
        )

    return FileResponse(
        path=str(safe_path),
        media_type=artifact.mime_type,
        filename=artifact.filename,
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/document-bundle -- Export DocumentBundle v1
# ------------------------------------------------------------------


@router.get(
    "/api/v1/jobs/{job_id}/document-bundle",
    name="get_job_document_bundle",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def get_job_document_bundle(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Return a contract-valid OCR ``DocumentBundle v1`` for a completed job."""

    _validate_job_id(job_id)
    job = _get_job_or_404(job_id, tenant_id=_request_tenant_id(request))
    bundle = _document_bundle_from_job(job, request)
    return JSONResponse(
        content=bundle,
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}.document-bundle.json"'
        },
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/evidence-bundle -- OCR evidence/custody bundle
# ------------------------------------------------------------------


@router.get(
    "/api/v1/jobs/{job_id}/evidence-bundle",
    name="get_job_evidence_bundle",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def get_job_evidence_bundle(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Return OCR-owned artifact and custody evidence for a job."""

    _validate_job_id(job_id)
    job = _get_job_or_404(job_id, tenant_id=_request_tenant_id(request))
    if not job.result_path:
        raise HTTPException(
            status_code=404,
            detail={"error": "no_outputs", "message": f"Job {job_id} has no output directory."},
        )
    manifest = build_manifest(job_id, job.result_path)
    custody = _custody_summary(job, manifest)
    document_bundle = _document_bundle_from_job(job, request)
    stable_document_bundle = deepcopy(document_bundle)
    stable_document_bundle.get("artifact_manifest", {}).pop("document_bundle_url", None)
    stable_document_bundle.get("artifact_manifest", {}).pop("evidence_bundle_url", None)
    evidence = {
        "schema_version": "ocr-evidence-bundle-v1",
        "job_id": job.job_id,
        "source_file": job.source_file,
        "source_file_sha256": job.source_hash,
        "status": job.status,
        "custody": custody,
        "document_bundle_sha256": canonical_json_sha256(stable_document_bundle),
        "document_bundle_url": str(
            request.url_for("get_job_document_bundle", job_id=job.job_id)
        ),
        "artifacts": [
            {
                "output_type": artifact.output_type,
                "filename": artifact.filename,
                "relative_path": artifact.relative_path,
                "size_bytes": artifact.size_bytes,
                "mime_type": artifact.mime_type,
                "schema_version": artifact.schema_version,
            }
            for artifact in manifest.artifacts
        ],
    }
    return JSONResponse(
        content=evidence,
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}.evidence-bundle.json"'
        },
    )


# ------------------------------------------------------------------
# GET /api/v1/schemas -- List available schemas
# ------------------------------------------------------------------


@router.get(
    "/api/v1/schemas",
    name="list_schemas",
    response_model=SchemaListResponse,
    responses={},
)
@limiter.limit(get_default_rate())
async def list_schemas(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """List all available output schema definitions.

    Returns the output type name and schema version for each defined
    schema in the ``schemas/`` package.
    """
    try:
        from schemas import OUTPUT_TYPES, SCHEMA_VERSION
    except ImportError:
        return SchemaListResponse(schemas=[])

    items = [
        SchemaListItem(output_type=ot, schema_version=SCHEMA_VERSION)
        for ot in OUTPUT_TYPES
    ]
    return SchemaListResponse(schemas=items)


# ------------------------------------------------------------------
# GET /api/v1/schemas/{output_type} -- Get specific schema
# ------------------------------------------------------------------


@router.get(
    "/api/v1/schemas/{output_type}",
    name="get_schema",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def get_schema(
    request: Request,
    output_type: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Get the JSON Schema definition for a specific output type.

    Returns the raw JSON Schema content.
    """
    try:
        from schemas import OUTPUT_TYPES, load_schema
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail={"error": "schemas_unavailable", "message": "Schema package not installed."},
        )

    if output_type not in OUTPUT_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_output_type",
                "message": f"Unknown output type: {output_type}. Valid types: {sorted(OUTPUT_TYPES)}",
            },
        )

    try:
        schema = load_schema(output_type)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "schema_not_found",
                "message": f"Schema file not found for output type: {output_type}",
            },
        )

    return JSONResponse(content=schema)
