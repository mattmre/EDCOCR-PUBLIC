"""Post-operation validation gates for transform and stamp operations.

This module provides reusable validation helpers that can be integrated into
transform/stamp execution workflows to ensure data integrity and quality.

Validation gates include:
- Output reference integrity (file exists, non-empty, hashable)
- Bates continuity (no gaps, duplicates, or non-sequential numeric suffixes)
- Stamp placement conflicts (overlap warnings treated as validation failures)
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ValidationGateResult:
    """Result of a validation gate check.
    
    Attributes:
        passed: Whether validation passed
        gate_name: Name of the validation gate
        message: Human-readable message
        details: Additional diagnostic details
    """
    passed: bool
    gate_name: str
    message: str
    details: dict[str, Any]


class ValidationGateError(Exception):
    """Raised when a validation gate fails."""
    pass


# --- Output Reference Integrity Gate ---


def validate_output_reference_integrity(
    output_path: str,
    min_size_bytes: int = 1,
) -> ValidationGateResult:
    """Validate that output file exists, is non-empty, and can be hashed.
    
    Args:
        output_path: Path to output file
        min_size_bytes: Minimum acceptable file size in bytes (default: 1)
        
    Returns:
        ValidationGateResult with pass/fail status and diagnostics
    """
    details: dict[str, Any] = {"output_path": output_path}
    
    # Check file exists
    if not os.path.exists(output_path):
        return ValidationGateResult(
            passed=False,
            gate_name="output_reference_integrity",
            message=f"Output file does not exist: {output_path}",
            details=details,
        )
    
    # Check file is not empty
    try:
        file_size = os.path.getsize(output_path)
        details["file_size_bytes"] = file_size
        
        if file_size < min_size_bytes:
            return ValidationGateResult(
                passed=False,
                gate_name="output_reference_integrity",
                message=f"Output file is too small: {file_size} bytes (min: {min_size_bytes})",
                details=details,
            )
    except OSError as exc:
        return ValidationGateResult(
            passed=False,
            gate_name="output_reference_integrity",
            message=f"Cannot access output file: {exc}",
            details=details,
        )
    
    # Compute hash to verify file is readable and complete
    try:
        sha256 = hashlib.sha256()
        with open(output_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()
        details["sha256_hash"] = file_hash
    except (OSError, IOError) as exc:
        return ValidationGateResult(
            passed=False,
            gate_name="output_reference_integrity",
            message=f"Cannot hash output file: {exc}",
            details=details,
        )
    
    return ValidationGateResult(
        passed=True,
        gate_name="output_reference_integrity",
        message="Output file integrity validated",
        details=details,
    )


# --- Bates Continuity Gate ---


def validate_bates_continuity(
    stamp_values: list[str],
    prefix: str = "",
    suffix: str = "",
    separator: str = "",
) -> ValidationGateResult:
    """Validate Bates number continuity: no gaps, duplicates, sequential numeric suffixes.
    
    Args:
        stamp_values: List of Bates stamp values applied
        prefix: Expected Bates prefix (empty string if none)
        suffix: Expected Bates suffix (empty string if none)
        separator: Separator between prefix/number/suffix
        
    Returns:
        ValidationGateResult with pass/fail status and diagnostics
    """
    details: dict[str, Any] = {
        "total_stamps": len(stamp_values),
        "prefix": prefix,
        "suffix": suffix,
        "separator": separator,
    }
    
    if not stamp_values:
        return ValidationGateResult(
            passed=False,
            gate_name="bates_continuity",
            message="No stamp values provided for validation",
            details=details,
        )
    
    # Parse numeric parts from stamp values
    numeric_parts = []
    for idx, stamp in enumerate(stamp_values):
        numeric_part = _extract_bates_numeric(stamp, prefix, suffix, separator)
        if numeric_part is None:
            return ValidationGateResult(
                passed=False,
                gate_name="bates_continuity",
                message=f"Failed to extract numeric part from stamp at index {idx}: '{stamp}'",
                details={**details, "invalid_stamp": stamp, "invalid_index": idx},
            )
        numeric_parts.append(numeric_part)
    
    details["numeric_range"] = {
        "first": numeric_parts[0],
        "last": numeric_parts[-1],
    }
    
    # Check for duplicates
    seen = set()
    duplicates = []
    for idx, num in enumerate(numeric_parts):
        if num in seen:
            duplicates.append((idx, num, stamp_values[idx]))
        seen.add(num)
    
    if duplicates:
        return ValidationGateResult(
            passed=False,
            gate_name="bates_continuity",
            message=f"Found {len(duplicates)} duplicate Bates numbers",
            details={
                **details,
                "duplicates": [{"index": idx, "number": num, "stamp": stamp} for idx, num, stamp in duplicates[:5]],
            },
        )
    
    # Check for sequential continuity (no gaps)
    expected = numeric_parts[0]
    gaps = []
    for idx, num in enumerate(numeric_parts):
        if num != expected:
            gaps.append((idx, expected, num, stamp_values[idx]))
        expected += 1
    
    if gaps:
        return ValidationGateResult(
            passed=False,
            gate_name="bates_continuity",
            message=f"Found {len(gaps)} gaps in Bates sequence",
            details={
                **details,
                "gaps": [
                    {"index": idx, "expected": exp, "actual": act, "stamp": stamp}
                    for idx, exp, act, stamp in gaps[:5]
                ],
            },
        )
    
    return ValidationGateResult(
        passed=True,
        gate_name="bates_continuity",
        message=f"Bates continuity validated: {len(stamp_values)} sequential stamps",
        details=details,
    )


def _extract_bates_numeric(
    stamp: str,
    prefix: str,
    suffix: str,
    separator: str,
) -> Optional[int]:
    """Extract numeric part from a Bates stamp string.
    
    Args:
        stamp: Full Bates stamp string (e.g., "PROD-000123-CONF")
        prefix: Bates prefix
        suffix: Bates suffix
        separator: Separator character
        
    Returns:
        Numeric part as integer, or None if cannot be extracted
    """
    # Build regex pattern based on prefix/suffix/separator
    parts = []
    if prefix:
        parts.append(re.escape(prefix))
    parts.append(r"(\d+)")  # Capture numeric part
    if suffix:
        parts.append(re.escape(suffix))
    
    pattern = re.escape(separator).join(parts) if separator else "".join(parts)
    
    match = re.fullmatch(pattern, stamp)
    if match:
        return int(match.group(1))
    return None


# --- Stamp Placement Conflict Gate ---


def validate_stamp_placement_no_conflicts(
    warnings: list[str],
    strict_mode: bool = True,
) -> ValidationGateResult:
    """Validate that stamp placement has no conflicts (treats overlap warnings as failures).
    
    Args:
        warnings: List of warning messages from stamp operation
        strict_mode: If True, any overlap warning causes failure; if False, only
                     explicit "conflict" or "critical" warnings fail
        
    Returns:
        ValidationGateResult with pass/fail status and diagnostics
    """
    details: dict[str, Any] = {
        "total_warnings": len(warnings),
        "strict_mode": strict_mode,
    }
    
    # Keywords that indicate placement conflicts
    conflict_keywords = ["overlap", "conflict", "collision", "obscured"]
    if not strict_mode:
        conflict_keywords = ["conflict", "critical"]
    
    conflicts = []
    for warning in warnings:
        warning_lower = warning.lower()
        if any(keyword in warning_lower for keyword in conflict_keywords):
            conflicts.append(warning)
    
    details["conflict_warnings"] = conflicts[:10]  # Limit to first 10
    
    if conflicts:
        return ValidationGateResult(
            passed=False,
            gate_name="stamp_placement_no_conflicts",
            message=f"Found {len(conflicts)} stamp placement conflict(s)",
            details=details,
        )
    
    return ValidationGateResult(
        passed=True,
        gate_name="stamp_placement_no_conflicts",
        message="No stamp placement conflicts detected",
        details=details,
    )


# --- Composite Validation Runner ---


def run_validation_gates(
    gates: list[ValidationGateResult],
    raise_on_failure: bool = False,
) -> tuple[bool, list[ValidationGateResult]]:
    """Run multiple validation gates and aggregate results.
    
    Args:
        gates: List of ValidationGateResult objects to evaluate
        raise_on_failure: If True, raise ValidationGateError on first failure
        
    Returns:
        Tuple of (all_passed, gate_results)
        
    Raises:
        ValidationGateError: If raise_on_failure is True and any gate fails
    """
    failed_gates = [g for g in gates if not g.passed]
    
    if failed_gates and raise_on_failure:
        first_failure = failed_gates[0]
        raise ValidationGateError(
            f"Validation gate '{first_failure.gate_name}' failed: {first_failure.message}"
        )
    
    all_passed = len(failed_gates) == 0
    return all_passed, gates


# --- Convenience Functions for Common Patterns ---


def validate_transform_output(
    output_path: str,
    min_size_bytes: int = 1,
) -> tuple[bool, dict[str, Any]]:
    """Convenience function to validate transform operation output.
    
    Args:
        output_path: Path to transform output file
        min_size_bytes: Minimum acceptable file size
        
    Returns:
        Tuple of (passed, diagnostics_dict)
    """
    result = validate_output_reference_integrity(output_path, min_size_bytes)
    return result.passed, {
        "validation_gate": result.gate_name,
        "passed": result.passed,
        "message": result.message,
        **result.details,
    }


def validate_stamp_output(
    output_path: str,
    stamp_values: list[str],
    warnings: list[str],
    prefix: str = "",
    suffix: str = "",
    separator: str = "",
    check_conflicts: bool = True,
    min_size_bytes: int = 1,
) -> tuple[bool, dict[str, Any]]:
    """Convenience function to validate stamp operation output.
    
    Args:
        output_path: Path to stamped output file
        stamp_values: List of Bates stamp values applied
        warnings: List of warning messages from stamp operation
        prefix: Bates prefix (for Bates operations)
        suffix: Bates suffix (for Bates operations)
        separator: Bates separator (for Bates operations)
        check_conflicts: Whether to check for stamp placement conflicts
        min_size_bytes: Minimum acceptable file size
        
    Returns:
        Tuple of (all_passed, aggregated_diagnostics_dict)
    """
    gates = []
    
    # Gate 1: Output reference integrity
    gates.append(validate_output_reference_integrity(output_path, min_size_bytes))
    
    # Gate 2: Bates continuity (if stamp values provided)
    if stamp_values:
        gates.append(validate_bates_continuity(stamp_values, prefix, suffix, separator))
    
    # Gate 3: Placement conflicts
    if check_conflicts:
        gates.append(validate_stamp_placement_no_conflicts(warnings))
    
    all_passed = all(g.passed for g in gates)
    
    diagnostics = {
        "all_gates_passed": all_passed,
        "gates": [
            {
                "gate_name": g.gate_name,
                "passed": g.passed,
                "message": g.message,
                "details": g.details,
            }
            for g in gates
        ],
    }
    
    return all_passed, diagnostics
