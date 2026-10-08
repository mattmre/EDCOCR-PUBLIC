"""Output manifest and schema retrieval endpoints.

Provides:
- ``GET /api/v1/jobs/{job_id}/outputs`` -- list all outputs for a job
- ``GET /api/v1/jobs/{job_id}/outputs/{output_type}`` -- download a specific output
- ``GET /api/v1/schemas`` -- list available output schemas
- ``GET /api/v1/schemas/{output_type}`` -- get a specific schema definition
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api.custody_authority import (
    CustodyAuthorityError,
    CustodyBytesCandidate,
    select_job_custody_chain,
)
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
    VALID_OUTPUT_TYPES,
    OutputManifest,
    build_manifest_from_root_fd,
)
from api.path_safety import (
    _bind_job_directory_under_root,
    _BoundJobDirectory,
    _directory_fd_alias,
    read_regular_file_bounded_under_root_fd,
    secure_file_response_under_root_fd,
)
from ocr_local.contracts import canonical_json_sha256
from ocr_local.document_bundle import build_document_bundle

logger = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_MAX_BUNDLE_TEXT_BYTES = 16 * 1024 * 1024
_MAX_BUNDLE_CUSTODY_BYTES = 16 * 1024 * 1024
_MAX_BUNDLE_CUSTODY_EVENTS = 100_000
_MAX_BUNDLE_CUSTODY_CANDIDATES = 128
_MAX_BUNDLE_CUSTODY_AGGREGATE_BYTES = 64 * 1024 * 1024

router = APIRouter(tags=["outputs"])


def _validate_job_id(job_id: str) -> None:
    """Raise 400 if job_id is malformed."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )


def _require_source_hash(job) -> str:
    """Return a durable content hash; never substitute a filename digest."""
    source_hash = str(getattr(job, "source_hash", "") or "").strip().lower()
    if not _SHA256_RE.fullmatch(source_hash):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "source_hash_unavailable",
                "message": "The job has no durable source content hash; refusing bundle export.",
            },
        )
    return source_hash


def _request_tenant_id(request: Request) -> str | None:
    """Return the request's authorized tenant scope.

    Output, document-bundle, and evidence-bundle routes are all data planes;
    an unscoped principal must fail closed under multitenancy before a job
    lookup or artifact read occurs.
    """
    from api.authz import require_tenant_scope_or_platform

    return require_tenant_scope_or_platform(request)


def _assert_job_deliverable_not_held(job, *extra_paths: str) -> None:
    """LT-13: fail-closed HITL hold for all content-export surfaces.

    Used by file download, document-bundle, and evidence-bundle so SDK/MCP
    cannot bypass hold via JSON text export while FileResponse returns 423.
    """
    try:
        from exception_router import assert_deliverable_not_held
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "hold_check_unavailable",
                "message": (
                    "LT-13 deliverable hold module unavailable; refusing output."
                ),
            },
        ) from exc
    # ``Job.result_path`` is persisted data and may be stale or poisoned; it
    # must never be handed to the hold walker before the canonical output root
    # has been resolved.  Callers pass only already-jail-checked paths here.
    paths = [str(p) for p in extra_paths if p]
    try:
        assert_deliverable_not_held(*paths)
    except ValueError as exc:
        raise HTTPException(
            status_code=423,
            detail={
                "error": "deliverable_held",
                "message": "Deliverable is under a litigation hold.",
            },
        ) from exc
    except OSError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "hold_check_failed",
                "message": "LT-13 hold check failed closed; output withheld.",
            },
        )


def _assert_job_outputs_released(job) -> None:
    """Refuse artifact content until processing completed successfully.

    ``result_path`` can be populated while a worker is still writing output.
    Treating its presence as release authority would expose partial evidence
    from queued, processing, failed, or cancelled jobs.
    """
    if str(getattr(job, "status", "") or "").lower() != "completed":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "job_not_complete",
                "message": f"Job is {getattr(job, 'status', 'unknown')}.",
            },
        )


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
                detail={
                    "error": "job_not_found",
                    "message": f"Job {job_id} not found.",
                },
            )
        return job
    finally:
        session.close()


@dataclass
class _BoundOutputRoot:
    namespace: _BoundJobDirectory
    descriptor_alias: Path

    @property
    def output_root(self) -> Path:
        return self.namespace.job_root

    @property
    def root_fd(self) -> int:
        return self.namespace.job_fd

    def assert_unchanged(self) -> None:
        self.namespace.assert_unchanged()

    def close(self) -> None:
        self.namespace.close()


def _bind_job_output_root(job) -> _BoundOutputRoot:
    """Bind the authorized job root once for manifest, reads, and response."""
    from api import config

    namespace: _BoundJobDirectory | None = None
    try:
        namespace = _bind_job_directory_under_root(
            config.OUTPUT_FOLDER,
            job.job_id,
            persisted_result_path=job.result_path,
        )
        bound_root = _directory_fd_alias(namespace.job_fd)
        return _BoundOutputRoot(namespace, bound_root)
    except (OSError, ValueError) as exc:
        if namespace is not None:
            namespace.close()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "output_unavailable",
                "message": "Job output cannot be bound safely on this host.",
            },
        ) from exc


def _assert_bound_output_unchanged(binding: _BoundOutputRoot) -> None:
    try:
        binding.assert_unchanged()
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "output_changed",
                "message": "Job output changed during response preparation.",
            },
        ) from exc


def _first_artifact(manifest: OutputManifest, output_type: str):
    return next(
        (
            artifact
            for artifact in manifest.artifacts
            if artifact.output_type == output_type
        ),
        None,
    )


def _read_text_artifact(
    job,
    manifest: OutputManifest,
    *,
    binding: _BoundOutputRoot | None = None,
) -> tuple[str, str]:
    artifact = _first_artifact(manifest, "ocr_text")
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "ocr_text_not_found",
                "message": f"Job {job.job_id} has no OCR text artifact.",
            },
        )
    owns_binding = binding is None
    if binding is None:
        binding = _bind_job_output_root(job)
    try:
        _assert_bound_output_unchanged(binding)
        # Use paths below the descriptor alias. The alias itself is a symlink,
        # but every final checked component is inside the held root inode.
        _assert_job_deliverable_not_held(
            job,
            str(binding.descriptor_alias / "EXPORT"),
            str(binding.descriptor_alias / artifact.relative_path),
        )
        raw = read_regular_file_bounded_under_root_fd(
            binding.root_fd,
            artifact.relative_path,
            max_bytes=_MAX_BUNDLE_TEXT_BYTES,
            expected_device=artifact.device,
            expected_inode=artifact.inode,
            expected_size=artifact.size_bytes,
            expected_mtime_ns=artifact.mtime_ns,
            expected_ctime_ns=artifact.ctime_ns,
        )
        _assert_bound_output_unchanged(binding)
    except ValueError as exc:
        if "exceeds" in str(exc):
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "ocr_text_too_large",
                    "message": "OCR text artifact exceeds the document-bundle size limit.",
                },
            ) from exc
        raise HTTPException(
            status_code=404,
            detail={
                "error": "ocr_text_missing",
                "message": "OCR text artifact is not a safe regular file.",
            },
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "ocr_text_missing",
                "message": "Unable to read OCR text artifact.",
            },
        ) from exc
    finally:
        if owns_binding:
            binding.close()
    return raw.decode("utf-8", errors="replace"), artifact.relative_path


def _custody_summary(
    job,
    manifest: OutputManifest,
    *,
    binding: _BoundOutputRoot | None = None,
) -> dict[str, Any]:
    artifacts = [
        artifact for artifact in manifest.artifacts if artifact.output_type == "custody"
    ]
    if not artifacts:
        return {
            "available": False,
            "valid": None,
            "message": "No custody artifact found for this job.",
            "relative_path": None,
            "chain_head": "n/a",
        }
    if (
        len(artifacts) > _MAX_BUNDLE_CUSTODY_CANDIDATES
        or sum(artifact.size_bytes for artifact in artifacts)
        > _MAX_BUNDLE_CUSTODY_AGGREGATE_BYTES
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "custody_chain_bounds_exceeded",
                "message": "Custody artifacts exceed aggregate export bounds.",
            },
        )
    owns_binding = binding is None
    if binding is None:
        binding = _bind_job_output_root(job)
    try:
        _assert_bound_output_unchanged(binding)
        candidates: list[CustodyBytesCandidate] = []
        for artifact in artifacts:
            raw = read_regular_file_bounded_under_root_fd(
                binding.root_fd,
                artifact.relative_path,
                max_bytes=_MAX_BUNDLE_CUSTODY_BYTES,
                expected_device=artifact.device,
                expected_inode=artifact.inode,
                expected_size=artifact.size_bytes,
                expected_mtime_ns=artifact.mtime_ns,
                expected_ctime_ns=artifact.ctime_ns,
            )
            candidates.append(
                CustodyBytesCandidate(
                    relative_path=artifact.relative_path,
                    raw=raw,
                    device=artifact.device,
                    inode=artifact.inode,
                    size_bytes=artifact.size_bytes,
                    nlink=artifact.nlink,
                )
            )
        _assert_bound_output_unchanged(binding)
        selected = select_job_custody_chain(
            job_id=str(job.job_id),
            source_hash=_require_source_hash(job),
            candidates=candidates,
            max_bytes=_MAX_BUNDLE_CUSTODY_BYTES,
            max_events=_MAX_BUNDLE_CUSTODY_EVENTS,
            max_candidates=_MAX_BUNDLE_CUSTODY_CANDIDATES,
            max_total_bytes=_MAX_BUNDLE_CUSTODY_AGGREGATE_BYTES,
            max_total_events=_MAX_BUNDLE_CUSTODY_EVENTS,
        )
    except CustodyAuthorityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": exc.code, "message": str(exc)},
        ) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "custody_chain_invalid",
                "message": "Custody artifacts changed or cannot be read safely.",
            },
        ) from exc
    finally:
        if owns_binding:
            binding.close()
    return {
        "available": True,
        "valid": True,
        "message": selected.verification_message,
        "relative_path": selected.relative_path,
        "chain_head": selected.chain_head,
    }


def _document_bundle_from_job(
    job,
    request: Request,
    *,
    binding: _BoundOutputRoot | None = None,
    manifest: OutputManifest | None = None,
    custody: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not job.result_path:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_outputs",
                "message": f"Job {job.job_id} has no output directory.",
            },
        )
    owns_binding = binding is None
    if binding is None:
        binding = _bind_job_output_root(job)
    try:
        _assert_bound_output_unchanged(binding)
        _assert_job_deliverable_not_held(
            job,
            str(binding.descriptor_alias / "EXPORT"),
        )
        source_hash = _require_source_hash(job)
        if manifest is None:
            manifest = build_manifest_from_root_fd(
                job.job_id,
                str(binding.output_root),
                binding.root_fd,
            )
        text, text_relative_path = _read_text_artifact(
            job,
            manifest,
            binding=binding,
        )
        if custody is None:
            custody = _custody_summary(
                job,
                manifest,
                binding=binding,
            )
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
            source_file_sha256=source_hash,
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
        _assert_bound_output_unchanged(binding)
        return bundle
    finally:
        if owns_binding:
            binding.close()


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

    binding = _bind_job_output_root(job)
    try:
        manifest = build_manifest_from_root_fd(
            job_id,
            str(binding.output_root),
            binding.root_fd,
        )
        _assert_bound_output_unchanged(binding)
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
    finally:
        binding.close()


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
    _assert_job_outputs_released(job)

    if not job.result_path:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_outputs",
                "message": f"Job {job_id} has no output directory.",
            },
        )

    binding = _bind_job_output_root(job)
    try:
        manifest = build_manifest_from_root_fd(
            job_id,
            str(binding.output_root),
            binding.root_fd,
        )
        if output_type == "custody":
            custody = _custody_summary(job, manifest, binding=binding)
            matching = [
                artifact
                for artifact in manifest.artifacts
                if artifact.output_type == "custody"
                and artifact.relative_path == custody["relative_path"]
            ]
        else:
            matching = [
                artifact
                for artifact in manifest.artifacts
                if artifact.output_type == output_type
            ]

        if not matching:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "output_not_found",
                    "message": f"No '{output_type}' output found for job {job_id}.",
                },
            )

        artifact = matching[0]
        _assert_bound_output_unchanged(binding)
        _assert_job_deliverable_not_held(
            job,
            str(binding.descriptor_alias / artifact.relative_path),
        )
        _assert_bound_output_unchanged(binding)
        return secure_file_response_under_root_fd(
            binding.root_fd,
            artifact.relative_path,
            media_type=artifact.mime_type,
            filename=artifact.filename,
            expected_device=artifact.device,
            expected_inode=artifact.inode,
            expected_size=artifact.size_bytes,
            expected_mtime_ns=artifact.mtime_ns,
            expected_ctime_ns=artifact.ctime_ns,
        )
    finally:
        binding.close()


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
    _assert_job_outputs_released(job)
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
    _assert_job_outputs_released(job)
    if not job.result_path:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_outputs",
                "message": f"Job {job_id} has no output directory.",
            },
        )
    binding = _bind_job_output_root(job)
    try:
        _assert_bound_output_unchanged(binding)
        _assert_job_deliverable_not_held(
            job,
            str(binding.descriptor_alias / "EXPORT"),
        )
        manifest = build_manifest_from_root_fd(
            job_id,
            str(binding.output_root),
            binding.root_fd,
        )
        custody = _custody_summary(
            job,
            manifest,
            binding=binding,
        )
        document_bundle = _document_bundle_from_job(
            job,
            request,
            binding=binding,
            manifest=manifest,
            custody=custody,
        )
        source_hash = _require_source_hash(job)
        stable_document_bundle = deepcopy(document_bundle)
        stable_document_bundle.get("artifact_manifest", {}).pop(
            "document_bundle_url", None
        )
        stable_document_bundle.get("artifact_manifest", {}).pop(
            "evidence_bundle_url", None
        )
        evidence = {
            "schema_version": "ocr-evidence-bundle-v1",
            "job_id": job.job_id,
            "source_file": job.source_file,
            "source_file_sha256": source_hash,
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
        _assert_bound_output_unchanged(binding)
        return JSONResponse(
            content=evidence,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{job_id}.evidence-bundle.json"'
                )
            },
        )
    finally:
        binding.close()


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
            detail={
                "error": "schemas_unavailable",
                "message": "Schema package not installed.",
            },
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
