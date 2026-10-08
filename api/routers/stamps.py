"""Stamp operation API endpoints."""

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api import config
from api.identity import require_role
from api.limits import get_default_rate, limiter
from api.models import ErrorResponse
from api.path_safety import (
    bind_authorized_operation_io,
    ensure_path_within_roots,
    sandbox_params_paths,
)
from custody_hooks import (
    create_custody_chain_for_operation,
    get_custody_diagnostics_summary,
    record_stamp_lifecycle,
)
from ocr_distributed.stamps.base import (
    StampConfig,
    StampError,
    StampPlacement,
    StampValidationError,
)
from ocr_distributed.stamps.builtin import register_builtin_stamps
from ocr_distributed.stamps.registry import get_stamp_registry
from validation_gates import (
    validate_stamp_output,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/stamps", tags=["stamps"])

# --- Request/Response Models ---


class StampExecuteRequest(BaseModel):
    """Request to execute a stamp operation."""

    operation_id: str = Field(..., description="Registered stamp operation name")
    input_path: str = Field(..., description="Path to input PDF file (server-side)")
    output_path: str = Field(..., description="Path where output should be written")
    placement: StampPlacement = Field(
        StampPlacement.BOTTOM_RIGHT, description="Stamp placement location"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters (e.g., bates_prefix, designation_text)",
    )
    validate_input: bool = Field(
        True, description="Whether to validate input before stamping"
    )
    check_overlap: bool = Field(
        True, description="Whether to detect and warn about stamp overlaps"
    )


class StampExecuteResponse(BaseModel):
    """Response from stamp execution."""

    success: bool
    operation_id: str
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    pages_stamped: int = 0
    stamp_values: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StampOperationMetadata(BaseModel):
    """Metadata about a registered stamp operation."""

    name: str
    description: str
    version: str
    supported_formats: list[str]
    parameters: dict[str, Any]


class StampListResponse(BaseModel):
    """List of available stamp operations."""

    operations: list[StampOperationMetadata]
    total: int


# --- Helpers ---


def _check_feature_enabled():
    """Raise 403 if stamping feature is disabled."""
    if not config.ENABLE_STAMPING:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "feature_disabled",
                "message": "Stamp operations are disabled. Set ENABLE_STAMPING=true to enable.",
            },
        )


def _ensure_registry_initialized():
    """Ensure stamp registry has built-in operations registered."""
    registry = get_stamp_registry()
    if not registry.list_operations():
        register_builtin_stamps(registry)
    return registry


# --- Endpoints ---


@router.get(
    "",
    name="list_stamps",
    response_model=StampListResponse,
    responses={403: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def list_stamps(
    request: Request,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """List all available stamp operations with metadata."""
    _check_feature_enabled()

    registry = _ensure_registry_initialized()
    all_metadata = registry.list_all_metadata()

    operations = [
        StampOperationMetadata(
            name=meta["name"],
            description=meta.get("description", ""),
            version=meta.get("version", "1.0.0"),
            supported_formats=meta.get("supported_formats", []),
            parameters=meta.get("parameters", {}),
        )
        for meta in all_metadata
    ]

    return StampListResponse(operations=operations, total=len(operations))


@router.get(
    "/{operation_id}",
    name="get_stamp_metadata",
    response_model=StampOperationMetadata,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def get_stamp_metadata(
    request: Request,
    operation_id: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Get metadata for a specific stamp operation."""
    _check_feature_enabled()

    registry = _ensure_registry_initialized()
    metadata = registry.get_metadata(operation_id)

    if not metadata:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "operation_not_found",
                "message": f"Stamp operation '{operation_id}' not found.",
            },
        )

    return StampOperationMetadata(
        name=metadata["name"],
        description=metadata.get("description", ""),
        version=metadata.get("version", "1.0.0"),
        supported_formats=metadata.get("supported_formats", []),
        parameters=metadata.get("parameters", {}),
    )


@router.post(
    "/execute",
    name="execute_stamp",
    response_model=StampExecuteResponse,
    status_code=200,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def execute_stamp(
    request: Request,
    req: StampExecuteRequest,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Execute a stamp operation on a server-side PDF file.

    This endpoint performs synchronous execution suitable for short operations.
    For long-running stamping, consider using the job queue instead.
    """
    _check_feature_enabled()

    from api.authz import require_platform_admin_for_shared_paths

    require_platform_admin_for_shared_paths(request)

    registry = _ensure_registry_initialized()

    # Validate operation exists
    operation = registry.get(req.operation_id)
    if not operation:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "operation_not_found",
                "message": f"Stamp operation '{req.operation_id}' not found.",
            },
        )

    input_path = str(
        ensure_path_within_roots(
            path_value=req.input_path,
            field_name="input_path",
            allowed_roots=(config.SOURCE_FOLDER, config.OUTPUT_FOLDER),
        )
    )
    output_path = str(
        ensure_path_within_roots(
            path_value=req.output_path,
            field_name="output_path",
            allowed_roots=(config.OUTPUT_FOLDER,),
        )
    )

    try:
        operation_io = bind_authorized_operation_io(
            input_path=input_path,
            output_path=output_path,
            read_roots=(config.SOURCE_FOLDER, config.OUTPUT_FOLDER),
            write_root=config.OUTPUT_FOLDER,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "input_not_found",
                "message": "Input file not found.",
            },
        ) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "input_unavailable",
                "message": "Input or output path cannot be bound safely.",
            },
        ) from exc

    try:
        return _execute_bound_stamp(
            operation=operation,
            req=req,
            request=request,
            input_path=input_path,
            output_path=output_path,
            operation_io=operation_io,
        )
    finally:
        operation_io.close()


def _execute_bound_stamp(
    *,
    operation,
    req,
    request: Request,
    input_path: str,
    output_path: str,
    operation_io,
):
    params = sandbox_params_paths(
        dict(req.params or {}),
        read_roots=(config.SOURCE_FOLDER, config.OUTPUT_FOLDER),
        write_roots=(config.OUTPUT_FOLDER,),
    )
    try:
        params = operation_io.bind_path_params(
            params,
            read_roots=(config.SOURCE_FOLDER, config.OUTPUT_FOLDER),
            write_root=config.OUTPUT_FOLDER,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "input_not_found",
                "message": "Input file not found.",
            },
        ) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "input_unavailable",
                "message": "Input or output path cannot be bound safely.",
            },
        ) from exc
    if req.operation_id == "bates":
        params.setdefault("allocate", True)
        caller_tid = getattr(request.state, "tenant_id", None)
        if isinstance(caller_tid, str) and caller_tid:
            params.setdefault("tenant_id", caller_tid)
        try:
            import fitz  # PyMuPDF

            with fitz.open(operation_io.input_exec_path) as _doc:
                params["_page_count"] = int(_doc.page_count)
        except Exception:
            params.setdefault("_page_count", 1)

    try:
        stamp_config = StampConfig(
            operation_name=req.operation_id,
            placement=req.placement,
            params=params,
            validate_input=req.validate_input,
            check_overlap=req.check_overlap,
        )
    except StampValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_config",
                "message": str(exc),
            },
        )

    validation_errors = operation.validate_config(stamp_config)
    if validation_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_failed",
                "message": "Stamp configuration validation failed",
                "details": {"errors": validation_errors},
            },
        )

    custody_dir = str(Path(config.OUTPUT_FOLDER) / "EXPORT" / "CUSTODY" / "ops")
    Path(custody_dir).mkdir(parents=True, exist_ok=True)
    custody_chain = create_custody_chain_for_operation(
        input_path=input_path,
        custody_dir=custody_dir,
        document_id=operation_io.document_id,
    )

    try:
        operation_io.assert_snapshot_unchanged()
        result = operation.execute(
            operation_io.input_exec_path,
            operation_io.output_exec_path,
            stamp_config,
        )
    except StampValidationError as exc:
        custody_diag = record_stamp_lifecycle(
            custody_chain=custody_chain,
            operation_id=req.operation_id,
            input_path=input_path,
            output_path=None,
            placement=req.placement.value,
            params=req.params,
            success=False,
            error_message=str(exc),
            input_hash=operation_io.input_hash,
            input_nlink=operation_io.input_nlink,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "execution_validation_failed",
                "message": str(exc),
                "custody": get_custody_diagnostics_summary(custody_diag),
            },
        )
    except StampError as exc:
        logger.error(f"Stamp execution error: {exc}", exc_info=True)
        custody_diag = record_stamp_lifecycle(
            custody_chain=custody_chain,
            operation_id=req.operation_id,
            input_path=input_path,
            output_path=None,
            placement=req.placement.value,
            params=req.params,
            success=False,
            error_message=str(exc),
            input_hash=operation_io.input_hash,
            input_nlink=operation_io.input_nlink,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "execution_failed",
                "message": "Stamp execution failed; inspect the custody record.",
                "custody": get_custody_diagnostics_summary(custody_diag),
            },
        )
    except Exception as exc:
        logger.error(f"Unexpected stamp error: {exc}", exc_info=True)
        custody_diag = record_stamp_lifecycle(
            custody_chain=custody_chain,
            operation_id=req.operation_id,
            input_path=input_path,
            output_path=None,
            placement=req.placement.value,
            params=req.params,
            success=False,
            error_message=f"Unexpected error: {type(exc).__name__}",
            input_hash=operation_io.input_hash,
            input_nlink=operation_io.input_nlink,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred during stamp execution.",
                "custody": get_custody_diagnostics_summary(custody_diag),
            },
        )

    output_hash = None
    if result.success:
        try:
            output_hash = operation_io.hash_output()
        except (OSError, ValueError) as exc:
            custody_diag = record_stamp_lifecycle(
                custody_chain=custody_chain,
                operation_id=req.operation_id,
                input_path=input_path,
                output_path=None,
                placement=req.placement.value,
                params=req.params,
                success=False,
                error_message=str(exc),
                input_hash=operation_io.input_hash,
                input_nlink=operation_io.input_nlink,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "output_changed",
                    "message": "Stamp output left the authorized parent.",
                    "custody": get_custody_diagnostics_summary(custody_diag),
                },
            ) from exc
        bates_prefix = (
            req.params.get("prefix", "") if req.operation_id == "bates" else ""
        )
        bates_suffix = (
            req.params.get("suffix", "") if req.operation_id == "bates" else ""
        )
        bates_separator = (
            req.params.get("separator", "") if req.operation_id == "bates" else ""
        )
        continuity_values = result.stamp_values if req.operation_id == "bates" else []
        validation_passed, validation_diag = validate_stamp_output(
            output_path=operation_io.output_exec_path,
            stamp_values=continuity_values,
            warnings=result.warnings,
            prefix=bates_prefix,
            suffix=bates_suffix,
            separator=bates_separator,
            check_conflicts=req.check_overlap,
        )
        if not validation_passed:
            custody_diag = record_stamp_lifecycle(
                custody_chain=custody_chain,
                operation_id=req.operation_id,
                input_path=input_path,
                output_path=output_path,
                placement=req.placement.value,
                params=req.params,
                success=False,
                error_message="Validation gate failed",
                stamp_values=result.stamp_values,
                metadata=result.metadata,
                input_hash=operation_io.input_hash,
                output_hash=output_hash,
                input_nlink=operation_io.input_nlink,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "validation_gate_failed",
                    "message": "Stamp output failed validation gate",
                    "validation": validation_diag,
                    "custody": get_custody_diagnostics_summary(custody_diag),
                },
            )

    custody_diag = record_stamp_lifecycle(
        custody_chain=custody_chain,
        operation_id=req.operation_id,
        input_path=input_path,
        output_path=output_path if result.success else None,
        placement=req.placement.value,
        params=req.params,
        success=result.success,
        error_message=result.error_message,
        stamp_values=result.stamp_values,
        metadata=result.metadata,
        input_hash=operation_io.input_hash,
        output_hash=output_hash,
        input_nlink=operation_io.input_nlink,
    )
    custody_summary = get_custody_diagnostics_summary(custody_diag)
    if result.success and not custody_summary.get("custody_durable", False):
        logger.critical(
            "Stamp %s produced output without a durable custody record",
            req.operation_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "custody_persistence_failed",
                "message": (
                    "Stamp output was not released because its forensic "
                    "custody record is not durable."
                ),
                "custody": custody_summary,
            },
        )
    enriched_metadata = {
        **result.metadata,
        "custody": custody_summary,
    }
    return StampExecuteResponse(
        success=result.success,
        operation_id=req.operation_id,
        output_path=output_path if result.success else result.output_path,
        error_message=result.error_message,
        metadata=enriched_metadata,
        pages_stamped=result.pages_stamped,
        stamp_values=result.stamp_values,
        warnings=result.warnings,
    )
