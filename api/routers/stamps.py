"""Stamp operation API endpoints."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api import config
from api.identity import require_role
from api.limits import get_default_rate, limiter
from api.models import ErrorResponse
from api.path_safety import ensure_path_within_roots
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
async def list_stamps(request: Request):
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
async def get_stamp_metadata(request: Request, operation_id: str):
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

    # Validate input file exists
    if not os.path.isfile(input_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "input_not_found",
                "message": f"Input file not found: {input_path}",
            },
        )

    # Build stamp config
    try:
        stamp_config = StampConfig(
            operation_name=req.operation_id,
            placement=req.placement,
            params=req.params,
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

    # Validate config
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

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize custody chain
    custody_chain = create_custody_chain_for_operation(
        input_path=input_path,
        custody_dir="",  # Disable file output for API operations
    )
    
    # Execute stamp
    try:
        result = operation.execute(input_path, output_path, stamp_config)
    except StampValidationError as exc:
        # Record failure in custody chain
        custody_diag = record_stamp_lifecycle(
            custody_chain=custody_chain,
            operation_id=req.operation_id,
            input_path=input_path,
            output_path=None,
            placement=req.placement.value,
            params=req.params,
            success=False,
            error_message=str(exc),
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
        
        # Record failure in custody chain
        custody_diag = record_stamp_lifecycle(
            custody_chain=custody_chain,
            operation_id=req.operation_id,
            input_path=input_path,
            output_path=None,
            placement=req.placement.value,
            params=req.params,
            success=False,
            error_message=str(exc),
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "execution_failed",
                "message": str(exc),
                "custody": get_custody_diagnostics_summary(custody_diag),
            },
        )
    except Exception as exc:
        logger.error(f"Unexpected stamp error: {exc}", exc_info=True)
        
        # Record failure in custody chain
        custody_diag = record_stamp_lifecycle(
            custody_chain=custody_chain,
            operation_id=req.operation_id,
            input_path=input_path,
            output_path=None,
            placement=req.placement.value,
            params=req.params,
            success=False,
            error_message=f"Unexpected error: {type(exc).__name__}",
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred during stamp execution.",
                "custody": get_custody_diagnostics_summary(custody_diag),
            },
        )
    
    # Validate output if operation succeeded
    if result.success and result.output_path:
        # Extract Bates params if present
        bates_prefix = req.params.get("prefix", "") if req.operation_id == "bates" else ""
        bates_suffix = req.params.get("suffix", "") if req.operation_id == "bates" else ""
        bates_separator = req.params.get("separator", "") if req.operation_id == "bates" else ""
        continuity_values = result.stamp_values if req.operation_id == "bates" else []
        
        validation_passed, validation_diag = validate_stamp_output(
            output_path=result.output_path,
            stamp_values=continuity_values,
            warnings=result.warnings,
            prefix=bates_prefix,
            suffix=bates_suffix,
            separator=bates_separator,
            check_conflicts=req.check_overlap,
        )
        
        if not validation_passed:
            # Record failure in custody chain
            custody_diag = record_stamp_lifecycle(
                custody_chain=custody_chain,
                operation_id=req.operation_id,
                input_path=input_path,
                output_path=result.output_path,
                placement=req.placement.value,
                params=req.params,
                success=False,
                error_message="Validation gate failed",
                stamp_values=result.stamp_values,
                metadata=result.metadata,
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
    
    # Record success in custody chain
    custody_diag = record_stamp_lifecycle(
        custody_chain=custody_chain,
        operation_id=req.operation_id,
        input_path=input_path,
        output_path=result.output_path,
        placement=req.placement.value,
        params=req.params,
        success=result.success,
        error_message=result.error_message,
        stamp_values=result.stamp_values,
        metadata=result.metadata,
    )
    
    # Enrich metadata with custody diagnostics
    enriched_metadata = {
        **result.metadata,
        "custody": get_custody_diagnostics_summary(custody_diag),
    }

    # Build response
    return StampExecuteResponse(
        success=result.success,
        operation_id=req.operation_id,
        output_path=result.output_path,
        error_message=result.error_message,
        metadata=enriched_metadata,
        pages_stamped=result.pages_stamped,
        stamp_values=result.stamp_values,
        warnings=result.warnings,
    )
