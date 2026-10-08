"""Custody hook utilities for recording transform/stamp lifecycle events.

This module provides helper functions to record transform and stamp operations
in the custody chain with input/output hashes for forensic audit trails.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Optional

from ocr_local.features.custody import (
    CustodyChain,
    compute_file_hash,
    validate_custody_document_id,
)

# --- Event Type Constants ---

EVENT_TRANSFORM_START = "transform_start"
EVENT_TRANSFORM_COMPLETE = "transform_complete"
EVENT_TRANSFORM_FAILED = "transform_failed"

EVENT_STAMP_START = "stamp_start"
EVENT_STAMP_COMPLETE = "stamp_complete"
EVENT_STAMP_FAILED = "stamp_failed"


# --- Custody Hook Helpers ---


def _update_custody_durability_diagnostics(
    diagnostics: dict[str, Any], custody_chain: CustodyChain
) -> None:
    """Expose whether lifecycle events are durable and still trustworthy.

    ``CustodyChain`` intentionally keeps a best-effort in-memory event when a
    disk append fails so callers can inspect the attempted event.  That event
    must not be presented as a durable forensic record, however.  Keep the
    historical ``custody_recorded`` flag for in-memory callers, while adding
    explicit durability and compromise signals for API/CLI consumers.
    """
    compromised = bool(getattr(custody_chain, "integrity_compromised", False))
    filepath = getattr(custody_chain, "_filepath", None)
    diagnostics["custody_integrity_compromised"] = compromised
    diagnostics["custody_durable"] = bool(filepath) and not compromised
    diagnostics["custody_recorded"] = not compromised


def record_transform_lifecycle(
    custody_chain: CustodyChain,
    operation_id: str,
    input_path: str,
    output_path: Optional[str],
    params: dict[str, Any],
    success: bool,
    error_message: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    input_hash: Optional[str] = None,
    output_hash: Optional[str] = None,
    input_nlink: Optional[int] = None,
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

    if input_hash is None:
        try:
            input_hash = compute_file_hash(input_path)
        except (OSError, IOError) as exc:
            diagnostics["input_hash_error"] = str(exc)
            input_hash = None
    if input_hash is not None:
        diagnostics["input_hash"] = input_hash
    if input_nlink is not None:
        diagnostics["input_nlink"] = input_nlink

    # Record start event
    start_data = {
        "operation_id": operation_id,
        "input_path": input_path,
        "input_hash": input_hash,
        "input_nlink": input_nlink,
        "params": params,
    }
    custody_chain.append_event(EVENT_TRANSFORM_START, start_data)
    diagnostics["start_event_recorded"] = not bool(
        getattr(custody_chain, "integrity_compromised", False)
    )

    # Record completion or failure event
    if success and output_path:
        if output_hash is None:
            try:
                output_hash = compute_file_hash(output_path)
            except (OSError, IOError) as exc:
                diagnostics["output_hash_error"] = str(exc)
                output_hash = None
        if output_hash is not None:
            diagnostics["output_hash"] = output_hash

        complete_data = {
            "operation_id": operation_id,
            "output_path": output_path,
            "output_hash": output_hash,
            "metadata": metadata or {},
        }
        custody_chain.append_event(EVENT_TRANSFORM_COMPLETE, complete_data)
        diagnostics["complete_event_recorded"] = not bool(
            getattr(custody_chain, "integrity_compromised", False)
        )

    else:
        failed_data = {
            "operation_id": operation_id,
            "error_message": error_message,
        }
        custody_chain.append_event(EVENT_TRANSFORM_FAILED, failed_data)
        diagnostics["failed_event_recorded"] = not bool(
            getattr(custody_chain, "integrity_compromised", False)
        )

    # Add chain summary
    diagnostics["custody_chain_hash"] = custody_chain.get_summary().get("chain_hash")
    _update_custody_durability_diagnostics(diagnostics, custody_chain)

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
    input_hash: Optional[str] = None,
    output_hash: Optional[str] = None,
    input_nlink: Optional[int] = None,
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

    if input_hash is None:
        try:
            input_hash = compute_file_hash(input_path)
        except (OSError, IOError) as exc:
            diagnostics["input_hash_error"] = str(exc)
            input_hash = None
    if input_hash is not None:
        diagnostics["input_hash"] = input_hash
    if input_nlink is not None:
        diagnostics["input_nlink"] = input_nlink

    # Record start event
    start_data = {
        "operation_id": operation_id,
        "input_path": input_path,
        "input_hash": input_hash,
        "input_nlink": input_nlink,
        "placement": placement,
        "params": params,
    }
    custody_chain.append_event(EVENT_STAMP_START, start_data)
    diagnostics["start_event_recorded"] = not bool(
        getattr(custody_chain, "integrity_compromised", False)
    )

    # Record completion or failure event
    if success and output_path:
        if output_hash is None:
            try:
                output_hash = compute_file_hash(output_path)
            except (OSError, IOError) as exc:
                diagnostics["output_hash_error"] = str(exc)
                output_hash = None
        if output_hash is not None:
            diagnostics["output_hash"] = output_hash

        complete_data = {
            "operation_id": operation_id,
            "output_path": output_path,
            "output_hash": output_hash,
            "stamp_values": stamp_values or [],
            "metadata": metadata or {},
        }
        custody_chain.append_event(EVENT_STAMP_COMPLETE, complete_data)
        diagnostics["complete_event_recorded"] = not bool(
            getattr(custody_chain, "integrity_compromised", False)
        )

    else:
        failed_data = {
            "operation_id": operation_id,
            "error_message": error_message,
        }
        custody_chain.append_event(EVENT_STAMP_FAILED, failed_data)
        diagnostics["failed_event_recorded"] = not bool(
            getattr(custody_chain, "integrity_compromised", False)
        )

    # Add chain summary
    diagnostics["custody_chain_hash"] = custody_chain.get_summary().get("chain_hash")
    _update_custody_durability_diagnostics(diagnostics, custody_chain)

    return diagnostics


def _operation_document_id(input_path: str) -> str:
    """Bind an op-local document id to path, content, size, and mtime.

    Content-hash-only ids fork from OCR resume ids and collide on copies.
    Basename fallback is refused: two ``scan.pdf`` paths must not share a tip.
    This identifier does not claim to encode a filesystem inode.
    """
    abs_path = os.path.abspath(input_path)
    stat = os.stat(abs_path)
    content_sha256 = compute_file_hash(abs_path)
    payload = f"{abs_path}|{content_sha256}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def create_custody_chain_for_operation(
    input_path: str,
    custody_dir: str = "",
    document_id: str | None = None,
) -> CustodyChain:
    """Create a new custody chain for an operation.

    Args:
        input_path: Path to input file (used to generate document ID)
        custody_dir: Directory for custody files (empty string disables file output)
        document_id: Optional caller-supplied forensic id (job/resume id)

    Returns:
        CustodyChain instance ready for event recording
    """
    if document_id is not None:
        resolved_id = validate_custody_document_id(document_id)
    else:
        try:
            resolved_id = _operation_document_id(input_path)
        except (OSError, IOError) as exc:
            raise ValueError(
                f"cannot establish forensic document_id for {input_path}"
            ) from exc

    return CustodyChain(
        document_id=resolved_id,
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
        "custody_recorded": bool(
            diagnostics.get(
                "custody_recorded",
                not diagnostics.get("custody_integrity_compromised", False),
            )
        ),
        "operation_type": diagnostics.get("operation_type"),
        "operation_id": diagnostics.get("operation_id"),
    }

    if "custody_durable" in diagnostics:
        summary["custody_durable"] = bool(diagnostics["custody_durable"])
    if "custody_integrity_compromised" in diagnostics:
        summary["custody_integrity_compromised"] = bool(
            diagnostics["custody_integrity_compromised"]
        )

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
