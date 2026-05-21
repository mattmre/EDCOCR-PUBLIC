"""Custody hook utilities for recording transform/stamp lifecycle events.

This module provides helper functions to record transform and stamp operations
in the custody chain with input/output hashes for forensic audit trails.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ocr_local.features.custody import CustodyChain, compute_file_hash

# --- Event Type Constants ---

EVENT_TRANSFORM_START = "transform_start"
EVENT_TRANSFORM_COMPLETE = "transform_complete"
EVENT_TRANSFORM_FAILED = "transform_failed"

EVENT_STAMP_START = "stamp_start"
EVENT_STAMP_COMPLETE = "stamp_complete"
EVENT_STAMP_FAILED = "stamp_failed"


# --- Custody Hook Helpers ---


def record_transform_lifecycle(
    custody_chain: CustodyChain,
    operation_id: str,
    input_path: str,
    output_path: Optional[str],
    params: dict[str, Any],
    success: bool,
    error_message: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Record transform operation lifecycle events in custody chain.
    
    Args:
        custody_chain: CustodyChain instance to record events
        operation_id: Transform operation identifier
        input_path: Path to input file
        output_path: Path to output file (None if operation failed)
        params: Transform operation parameters
        success: Whether operation succeeded
        error_message: Error message if operation failed
        metadata: Additional metadata to include
        
    Returns:
        Dictionary containing custody diagnostics (hashes, event info)
    """
    diagnostics: dict[str, Any] = {
        "operation_type": "transform",
        "operation_id": operation_id,
    }
    
    # Compute input hash
    try:
        input_hash = compute_file_hash(input_path)
        diagnostics["input_hash"] = input_hash
    except (OSError, IOError) as exc:
        diagnostics["input_hash_error"] = str(exc)
        input_hash = None
    
    # Record start event
    start_data = {
        "operation_id": operation_id,
        "input_path": input_path,
        "input_hash": input_hash,
        "params": params,
    }
    custody_chain.append_event(EVENT_TRANSFORM_START, start_data)
    diagnostics["start_event_recorded"] = True
    
    # Record completion or failure event
    if success and output_path:
        # Compute output hash
        try:
            output_hash = compute_file_hash(output_path)
            diagnostics["output_hash"] = output_hash
        except (OSError, IOError) as exc:
            diagnostics["output_hash_error"] = str(exc)
            output_hash = None
        
        complete_data = {
            "operation_id": operation_id,
            "output_path": output_path,
            "output_hash": output_hash,
            "metadata": metadata or {},
        }
        custody_chain.append_event(EVENT_TRANSFORM_COMPLETE, complete_data)
        diagnostics["complete_event_recorded"] = True
        
    else:
        failed_data = {
            "operation_id": operation_id,
            "error_message": error_message,
        }
        custody_chain.append_event(EVENT_TRANSFORM_FAILED, failed_data)
        diagnostics["failed_event_recorded"] = True
    
    # Add chain summary
    diagnostics["custody_chain_hash"] = custody_chain.get_summary().get("chain_hash")
    
    return diagnostics


def record_stamp_lifecycle(
    custody_chain: CustodyChain,
    operation_id: str,
    input_path: str,
    output_path: Optional[str],
    placement: str,
    params: dict[str, Any],
    success: bool,
    error_message: Optional[str] = None,
    stamp_values: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Record stamp operation lifecycle events in custody chain.
    
    Args:
        custody_chain: CustodyChain instance to record events
        operation_id: Stamp operation identifier
        input_path: Path to input file
        output_path: Path to output file (None if operation failed)
        placement: Stamp placement location
        params: Stamp operation parameters
        success: Whether operation succeeded
        error_message: Error message if operation failed
        stamp_values: List of stamp values applied (for Bates operations)
        metadata: Additional metadata to include
        
    Returns:
        Dictionary containing custody diagnostics (hashes, event info)
    """
    diagnostics: dict[str, Any] = {
        "operation_type": "stamp",
        "operation_id": operation_id,
    }
    
    # Compute input hash
    try:
        input_hash = compute_file_hash(input_path)
        diagnostics["input_hash"] = input_hash
    except (OSError, IOError) as exc:
        diagnostics["input_hash_error"] = str(exc)
        input_hash = None
    
    # Record start event
    start_data = {
        "operation_id": operation_id,
        "input_path": input_path,
        "input_hash": input_hash,
        "placement": placement,
        "params": params,
    }
    custody_chain.append_event(EVENT_STAMP_START, start_data)
    diagnostics["start_event_recorded"] = True
    
    # Record completion or failure event
    if success and output_path:
        # Compute output hash
        try:
            output_hash = compute_file_hash(output_path)
            diagnostics["output_hash"] = output_hash
        except (OSError, IOError) as exc:
            diagnostics["output_hash_error"] = str(exc)
            output_hash = None
        
        complete_data = {
            "operation_id": operation_id,
            "output_path": output_path,
            "output_hash": output_hash,
            "stamp_values": stamp_values or [],
            "metadata": metadata or {},
        }
        custody_chain.append_event(EVENT_STAMP_COMPLETE, complete_data)
        diagnostics["complete_event_recorded"] = True
        
    else:
        failed_data = {
            "operation_id": operation_id,
            "error_message": error_message,
        }
        custody_chain.append_event(EVENT_STAMP_FAILED, failed_data)
        diagnostics["failed_event_recorded"] = True
    
    # Add chain summary
    diagnostics["custody_chain_hash"] = custody_chain.get_summary().get("chain_hash")
    
    return diagnostics


def create_custody_chain_for_operation(
    input_path: str,
    custody_dir: str = "",
) -> CustodyChain:
    """Create a new custody chain for an operation.
    
    Args:
        input_path: Path to input file (used to generate document ID)
        custody_dir: Directory for custody files (empty string disables file output)
        
    Returns:
        CustodyChain instance ready for event recording
    """
    # Generate document ID from input path hash
    try:
        file_hash = compute_file_hash(input_path)
        document_id = file_hash[:16]  # Use first 16 chars of hash
    except (OSError, IOError):
        # Fallback to filename-based ID
        document_id = os.path.basename(input_path).replace(".", "_")
    
    return CustodyChain(
        document_id=document_id,
        source_path=input_path,
        custody_dir=custody_dir,
    )


def get_custody_diagnostics_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Extract key custody diagnostics for API/CLI response metadata.
    
    Args:
        diagnostics: Full diagnostics dictionary from record_*_lifecycle functions
        
    Returns:
        Summarized diagnostics suitable for inclusion in API responses
    """
    summary = {
        "custody_recorded": True,
        "operation_type": diagnostics.get("operation_type"),
        "operation_id": diagnostics.get("operation_id"),
    }
    
    if "input_hash" in diagnostics:
        summary["input_hash"] = diagnostics["input_hash"]
    
    if "output_hash" in diagnostics:
        summary["output_hash"] = diagnostics["output_hash"]
    
    if "custody_chain_hash" in diagnostics:
        summary["custody_chain_hash"] = diagnostics["custody_chain_hash"]
    
    # Include any errors
    errors = {}
    if "input_hash_error" in diagnostics:
        errors["input_hash"] = diagnostics["input_hash_error"]
    if "output_hash_error" in diagnostics:
        errors["output_hash"] = diagnostics["output_hash_error"]
    
    if errors:
        summary["custody_errors"] = errors
    
    return summary
