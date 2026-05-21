"""Transform operation API endpoints."""

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
    record_transform_lifecycle,
)
from ocr_distributed.transforms.base import (
    TransformConfig,
    TransformError,
    TransformValidationError,
)
from ocr_distributed.transforms.builtin import register_builtin_transforms
from ocr_distributed.transforms.registry import get_transform_registry
from validation_gates import (
    validate_transform_output,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/transforms", tags=["transforms"])

# --- Request/Response Models ---


class TransformExecuteRequest(BaseModel):
    """Request to execute a transform operation."""

    operation_id: str = Field(..., description="Registered transform operation name")
    input_path: str = Field(..., description="Path to input file (server-side)")
    output_path: str = Field(..., description="Path where output should be written")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Operation-specific parameters"
    )
    validate_input: bool = Field(
        True, description="Whether to validate input before transform"
    )
    preserve_metadata: bool = Field(
        True, description="Whether to preserve PDF/image metadata"
    )


class TransformExecuteResponse(BaseModel):
    """Response from transform execution."""

    success: bool
    operation_id: str
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    pages_processed: int = 0
    warnings: list[str] = Field(default_factory=list)


class TransformOperationMetadata(BaseModel):
    """Metadata about a registered transform operation."""

    name: str
    description: str
    version: str
    supported_formats: list[str]
    output_format: str
    parameters: dict[str, Any]


class TransformListResponse(BaseModel):
    """List of available transform operations."""

    operations: list[TransformOperationMetadata]
    total: int


# --- Helpers ---


def _check_feature_enabled():
    """Raise 403 if transform feature is disabled."""
    if not config.ENABLE_TRANSFORMS:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "feature_disabled",
                "message": "Transform operations are disabled. Set ENABLE_TRANSFORMS=true to enable.",
            },
        )


def _ensure_registry_initialized():
    """Ensure transform registry has built-in operations registered."""
    registry = get_transform_registry()
    if not registry.list_operations():
        register_builtin_transforms(registry)
    return registry


# --- Endpoints ---


@router.get(
    "",
    name="list_transforms",
    response_model=TransformListResponse,
    responses={403: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def list_transforms(request: Request):
    """List all available transform operations with metadata."""
    _check_feature_enabled()

    registry = _ensure_registry_initialized()
    all_metadata = registry.list_all_metadata()

    operations = [
        TransformOperationMetadata(
            name=meta["name"],
            description=meta.get("description", ""),
            version=meta.get("version", "1.0.0"),
            supported_formats=meta.get("supported_formats", []),
            output_format=meta.get("output_format", ""),
            parameters=meta.get("parameters", {}),
        )
        for meta in all_metadata
    ]

    return TransformListResponse(operations=operations, total=len(operations))


@router.get(
    "/{operation_id}",
    name="get_transform_metadata",
    response_model=TransformOperationMetadata,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def get_transform_metadata(request: Request, operation_id: str):
    """Get metadata for a specific transform operation."""
    _check_feature_enabled()

    registry = _ensure_registry_initialized()
    metadata = registry.get_metadata(operation_id)

    if not metadata:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "operation_not_found",
                "message": f"Transform operation '{operation_id}' not found.",
            },
        )

    return TransformOperationMetadata(
        name=metadata["name"],
        description=metadata.get("description", ""),
        version=metadata.get("version", "1.0.0"),
        supported_formats=metadata.get("supported_formats", []),
        output_format=metadata.get("output_format", ""),
        parameters=metadata.get("parameters", {}),
    )


@router.post(
    "/execute",
    name="execute_transform",
    response_model=TransformExecuteResponse,
    status_code=200,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def execute_transform(
    request: Request,
    req: TransformExecuteRequest,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Execute a transform operation on a server-side file.
    
    This endpoint performs synchronous execution suitable for short operations.
    For long-running transforms, consider using the job queue instead.
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
                "message": f"Transform operation '{req.operation_id}' not found.",
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

    # Build transform config
    try:
        transform_config = TransformConfig(
            operation_name=req.operation_id,
            params=req.params,
            validate_input=req.validate_input,
            preserve_metadata=req.preserve_metadata,
        )
    except TransformValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_config",
                "message": str(exc),
            },
        )

    # Validate config
    validation_errors = operation.validate_config(transform_config)
    if validation_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_failed",
                "message": "Transform configuration validation failed",
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
    
    # Execute transform
    try:
        result = operation.execute(input_path, output_path, transform_config)
    except TransformValidationError as exc:
        # Record failure in custody chain
        custody_diag = record_transform_lifecycle(
            custody_chain=custody_chain,
            operation_id=req.operation_id,
            input_path=input_path,
            output_path=None,
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
    except TransformError as exc:
        logger.error(f"Transform execution error: {exc}", exc_info=True)
        
        # Record failure in custody chain
        custody_diag = record_transform_lifecycle(
            custody_chain=custody_chain,
            operation_id=req.operation_id,
            input_path=input_path,
            output_path=None,
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
        logger.error(f"Unexpected transform error: {exc}", exc_info=True)
        
        # Record failure in custody chain
        custody_diag = record_transform_lifecycle(
            custody_chain=custody_chain,
            operation_id=req.operation_id,
            input_path=input_path,
            output_path=None,
            params=req.params,
            success=False,
            error_message=f"Unexpected error: {type(exc).__name__}",
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred during transform execution.",
                "custody": get_custody_diagnostics_summary(custody_diag),
            },
        )
    
    # Validate output if operation succeeded
    if result.success and result.output_path:
        validation_passed, validation_diag = validate_transform_output(result.output_path)
        
        if not validation_passed:
            # Record failure in custody chain
            custody_diag = record_transform_lifecycle(
                custody_chain=custody_chain,
                operation_id=req.operation_id,
                input_path=input_path,
                output_path=result.output_path,
                params=req.params,
                success=False,
                error_message=f"Validation gate failed: {validation_diag.get('message', 'Unknown')}",
                metadata=result.metadata,
            )
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "validation_gate_failed",
                    "message": "Transform output failed validation gate",
                    "validation": validation_diag,
                    "custody": get_custody_diagnostics_summary(custody_diag),
                },
            )
    
    # Record success in custody chain
    custody_diag = record_transform_lifecycle(
        custody_chain=custody_chain,
        operation_id=req.operation_id,
        input_path=input_path,
        output_path=result.output_path,
        params=req.params,
        success=result.success,
        error_message=result.error_message,
        metadata=result.metadata,
    )
    
    # Enrich metadata with custody and validation diagnostics
    enriched_metadata = {
        **result.metadata,
        "custody": get_custody_diagnostics_summary(custody_diag),
    }

    # Build response
    return TransformExecuteResponse(
        success=result.success,
        operation_id=req.operation_id,
        output_path=result.output_path,
        error_message=result.error_message,
        metadata=enriched_metadata,
        pages_processed=result.pages_processed,
        warnings=result.warnings,
    )
