#!/usr/bin/env python3
"""CLI script for transform and stamp operations.

Provides command-line interface to execute transform and stamp operations
with parity to API contracts. Supports single operations and chained operations.

Usage:
    # Single transform operation
    python scripts/transform_stamp_cli.py transform pdf_rotate \\
        --input input.pdf --output output.pdf \\
        --params '{"angle": 90}'
    
    # Single stamp operation
    python scripts/transform_stamp_cli.py stamp bates \\
        --input input.pdf --output output.pdf \\
        --placement bottom_right \\
        --params '{"prefix": "ABC", "start_number": 1000}'
    
    # Chained operations
    python scripts/transform_stamp_cli.py chain \\
        --input input.pdf --output final.pdf \\
        --operations '[
            {"type": "transform", "id": "pdf_rotate", "params": {"angle": 90}},
            {"type": "stamp", "id": "bates", "placement": "bottom_right", 
             "params": {"prefix": "XYZ"}}
        ]'
    
    # JSON output mode
    python scripts/transform_stamp_cli.py transform pdf_rotate \\
        --input input.pdf --output output.pdf \\
        --params '{"angle": 90}' \\
        --json-output result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from custody_hooks import (
    create_custody_chain_for_operation,
    get_custody_diagnostics_summary,
    record_stamp_lifecycle,
    record_transform_lifecycle,
)
from ocr_distributed.stamps.base import (
    StampConfig,
    StampError,
    StampPlacement,
    StampValidationError,
)
from ocr_distributed.stamps.builtin import register_builtin_stamps
from ocr_distributed.stamps.registry import get_stamp_registry
from ocr_distributed.transforms.base import (
    TransformConfig,
    TransformError,
    TransformValidationError,
)
from ocr_distributed.transforms.builtin import register_builtin_transforms
from ocr_distributed.transforms.registry import get_transform_registry
from validation_gates import (
    validate_stamp_output,
    validate_transform_output,
)

logger = logging.getLogger(__name__)


# --- Execution Summary Models ---


class ExecutionSummary:
    """Summary of operation execution for JSON output."""

    def __init__(self):
        self.success = True
        self.operations = []
        self.error_message = None
        self.final_output_path = None

    def add_operation(self, op_type: str, op_id: str, success: bool, details: dict):
        """Add operation result to summary."""
        self.operations.append({
            "type": op_type,
            "operation_id": op_id,
            "success": success,
            "details": details,
        })
        if not success:
            self.success = False

    def set_error(self, error_message: str):
        """Set overall error message."""
        self.success = False
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "operations": self.operations,
            "error_message": self.error_message,
            "final_output_path": self.final_output_path,
        }


# --- Registry Initialization ---


def ensure_registries_initialized():
    """Ensure transform and stamp registries have built-in operations registered."""
    transform_registry = get_transform_registry()
    if not transform_registry.list_operations():
        register_builtin_transforms(transform_registry)

    stamp_registry = get_stamp_registry()
    if not stamp_registry.list_operations():
        register_builtin_stamps(stamp_registry)


# --- Single Operation Execution ---


def execute_transform(
    operation_id: str,
    input_path: str,
    output_path: str,
    params: dict[str, Any],
    validate_input: bool = True,
    preserve_metadata: bool = True,
    enable_custody: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Execute a single transform operation.

    Args:
        operation_id: Registered transform operation name
        input_path: Path to input file
        output_path: Path where output should be written
        params: Operation-specific parameters
        validate_input: Whether to validate input before transform
        preserve_metadata: Whether to preserve PDF/image metadata
        enable_custody: Whether to record custody events

    Returns:
        Tuple of (success, details) where details contains result info
    """
    registry = get_transform_registry()
    operation = registry.get(operation_id)

    if not operation:
        return False, {
            "error": "operation_not_found",
            "message": f"Transform operation '{operation_id}' not found",
            "available_operations": registry.list_operations(),
        }

    # Validate input file exists
    if not Path(input_path).exists():
        return False, {
            "error": "input_not_found",
            "message": f"Input file not found: {input_path}",
        }

    # Build and validate config
    try:
        config = TransformConfig(
            operation_name=operation_id,
            params=params,
            validate_input=validate_input,
            preserve_metadata=preserve_metadata,
        )
    except TransformValidationError as e:
        return False, {
            "error": "config_validation_error",
            "message": str(e),
        }

    # Validate config against operation
    validation_errors = operation.validate_config(config)
    if validation_errors:
        return False, {
            "error": "config_validation_error",
            "message": "Configuration validation failed",
            "validation_errors": validation_errors,
        }

    # Initialize custody chain if enabled
    custody_chain = None
    if enable_custody:
        custody_chain = create_custody_chain_for_operation(
            input_path=input_path,
            custody_dir="",  # Disable file output for CLI
        )

    # Execute operation
    try:
        result = operation.execute(input_path, output_path, config)
        
        # Validate output if operation succeeded
        if result.success and result.output_path:
            validation_passed, validation_diag = validate_transform_output(result.output_path)
            
            if not validation_passed:
                # Record failure if custody enabled
                if custody_chain:
                    custody_diag = record_transform_lifecycle(
                        custody_chain=custody_chain,
                        operation_id=operation_id,
                        input_path=input_path,
                        output_path=result.output_path,
                        params=params,
                        success=False,
                        error_message=f"Validation gate failed: {validation_diag.get('message', 'Unknown')}",
                        metadata=result.metadata,
                    )
                    validation_diag["custody"] = get_custody_diagnostics_summary(custody_diag)
                
                return False, {
                    "error": "validation_gate_failed",
                    "message": "Transform output failed validation gate",
                    "validation": validation_diag,
                }
        
        # Record success in custody chain
        custody_diag_dict = {}
        if custody_chain:
            custody_diag = record_transform_lifecycle(
                custody_chain=custody_chain,
                operation_id=operation_id,
                input_path=input_path,
                output_path=result.output_path,
                params=params,
                success=result.success,
                error_message=result.error_message,
                metadata=result.metadata,
            )
            custody_diag_dict = get_custody_diagnostics_summary(custody_diag)
        
        return result.success, {
            "output_path": result.output_path,
            "error_message": result.error_message,
            "metadata": result.metadata,
            "pages_processed": result.pages_processed,
            "warnings": result.warnings,
            "custody": custody_diag_dict if custody_diag_dict else None,
        }
        
    except (TransformError, TransformValidationError) as e:
        # Record failure in custody chain
        if custody_chain:
            custody_diag = record_transform_lifecycle(
                custody_chain=custody_chain,
                operation_id=operation_id,
                input_path=input_path,
                output_path=None,
                params=params,
                success=False,
                error_message=str(e),
            )
            custody_summary = get_custody_diagnostics_summary(custody_diag)
        else:
            custody_summary = None
        
        return False, {
            "error": "execution_error",
            "message": str(e),
            "custody": custody_summary,
        }
    except Exception as e:
        logger.exception(f"Unexpected error executing transform '{operation_id}'")
        
        # Record failure in custody chain
        if custody_chain:
            custody_diag = record_transform_lifecycle(
                custody_chain=custody_chain,
                operation_id=operation_id,
                input_path=input_path,
                output_path=None,
                params=params,
                success=False,
                error_message=f"Unexpected error: {type(e).__name__}: {e}",
            )
            custody_summary = get_custody_diagnostics_summary(custody_diag)
        else:
            custody_summary = None
        
        return False, {
            "error": "unexpected_error",
            "message": f"Unexpected error: {type(e).__name__}: {e}",
            "custody": custody_summary,
        }


def execute_stamp(
    operation_id: str,
    input_path: str,
    output_path: str,
    placement: str,
    params: dict[str, Any],
    validate_input: bool = True,
    check_overlap: bool = True,
    enable_custody: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Execute a single stamp operation.

    Args:
        operation_id: Registered stamp operation name
        input_path: Path to input PDF file
        output_path: Path where output should be written
        placement: Stamp placement location (e.g., 'bottom_right')
        params: Operation-specific parameters
        validate_input: Whether to validate input before stamping
        check_overlap: Whether to detect and warn about stamp overlaps
        enable_custody: Whether to record custody events

    Returns:
        Tuple of (success, details) where details contains result info
    """
    registry = get_stamp_registry()
    operation = registry.get(operation_id)

    if not operation:
        return False, {
            "error": "operation_not_found",
            "message": f"Stamp operation '{operation_id}' not found",
            "available_operations": registry.list_operations(),
        }

    # Validate input file exists
    if not Path(input_path).exists():
        return False, {
            "error": "input_not_found",
            "message": f"Input file not found: {input_path}",
        }

    # Parse placement
    try:
        placement_enum = StampPlacement(placement)
    except ValueError:
        return False, {
            "error": "invalid_placement",
            "message": f"Invalid placement '{placement}', must be one of: "
            f"{[p.value for p in StampPlacement]}",
        }

    # Build and validate config
    try:
        config = StampConfig(
            operation_name=operation_id,
            placement=placement_enum,
            params=params,
            validate_input=validate_input,
            check_overlap=check_overlap,
        )
    except StampValidationError as e:
        return False, {
            "error": "config_validation_error",
            "message": str(e),
        }

    # Validate config against operation
    validation_errors = operation.validate_config(config)
    if validation_errors:
        return False, {
            "error": "config_validation_error",
            "message": "Configuration validation failed",
            "validation_errors": validation_errors,
        }

    # Initialize custody chain if enabled
    custody_chain = None
    if enable_custody:
        custody_chain = create_custody_chain_for_operation(
            input_path=input_path,
            custody_dir="",  # Disable file output for CLI
        )

    # Execute operation
    try:
        result = operation.execute(input_path, output_path, config)
        
        # Validate output if operation succeeded
        if result.success and result.output_path:
            # Extract Bates params if present
            bates_prefix = params.get("prefix", "") if operation_id == "bates" else ""
            bates_suffix = params.get("suffix", "") if operation_id == "bates" else ""
            bates_separator = params.get("separator", "") if operation_id == "bates" else ""
            continuity_values = result.stamp_values if operation_id == "bates" else []
            
            validation_passed, validation_diag = validate_stamp_output(
                output_path=result.output_path,
                stamp_values=continuity_values,
                warnings=result.warnings,
                prefix=bates_prefix,
                suffix=bates_suffix,
                separator=bates_separator,
                check_conflicts=check_overlap,
            )
            
            if not validation_passed:
                # Record failure if custody enabled
                if custody_chain:
                    custody_diag = record_stamp_lifecycle(
                        custody_chain=custody_chain,
                        operation_id=operation_id,
                        input_path=input_path,
                        output_path=result.output_path,
                        placement=placement,
                        params=params,
                        success=False,
                        error_message="Validation gate failed",
                        stamp_values=result.stamp_values,
                        metadata=result.metadata,
                    )
                    validation_diag["custody"] = get_custody_diagnostics_summary(custody_diag)
                
                return False, {
                    "error": "validation_gate_failed",
                    "message": "Stamp output failed validation gate",
                    "validation": validation_diag,
                }
        
        # Record success in custody chain
        custody_diag_dict = {}
        if custody_chain:
            custody_diag = record_stamp_lifecycle(
                custody_chain=custody_chain,
                operation_id=operation_id,
                input_path=input_path,
                output_path=result.output_path,
                placement=placement,
                params=params,
                success=result.success,
                error_message=result.error_message,
                stamp_values=result.stamp_values,
                metadata=result.metadata,
            )
            custody_diag_dict = get_custody_diagnostics_summary(custody_diag)
        
        return result.success, {
            "output_path": result.output_path,
            "error_message": result.error_message,
            "metadata": result.metadata,
            "pages_stamped": result.pages_stamped,
            "stamp_values": result.stamp_values,
            "warnings": result.warnings,
            "custody": custody_diag_dict if custody_diag_dict else None,
        }
        
    except (StampError, StampValidationError) as e:
        # Record failure in custody chain
        if custody_chain:
            custody_diag = record_stamp_lifecycle(
                custody_chain=custody_chain,
                operation_id=operation_id,
                input_path=input_path,
                output_path=None,
                placement=placement,
                params=params,
                success=False,
                error_message=str(e),
            )
            custody_summary = get_custody_diagnostics_summary(custody_diag)
        else:
            custody_summary = None
        
        return False, {
            "error": "execution_error",
            "message": str(e),
            "custody": custody_summary,
        }
    except Exception as e:
        logger.exception(f"Unexpected error executing stamp '{operation_id}'")
        
        # Record failure in custody chain
        if custody_chain:
            custody_diag = record_stamp_lifecycle(
                custody_chain=custody_chain,
                operation_id=operation_id,
                input_path=input_path,
                output_path=None,
                placement=placement,
                params=params,
                success=False,
                error_message=f"Unexpected error: {type(e).__name__}: {e}",
            )
            custody_summary = get_custody_diagnostics_summary(custody_diag)
        else:
            custody_summary = None
        
        return False, {
            "error": "unexpected_error",
            "message": f"Unexpected error: {type(e).__name__}: {e}",
            "custody": custody_summary,
        }


# --- Chained Operations ---


def execute_chain(
    input_path: str,
    output_path: str,
    operations: list[dict[str, Any]],
) -> tuple[bool, ExecutionSummary]:
    """Execute a chain of transform and stamp operations.

    Operations are executed in sequence, with the output of each operation
    becoming the input to the next. Intermediate files are stored in temp
    directory and cleaned up automatically.

    Args:
        input_path: Path to initial input file
        output_path: Path where final output should be written
        operations: List of operation specifications, each containing:
            - type: "transform" or "stamp"
            - id: operation identifier
            - params: operation parameters
            - placement: (stamp only) placement location

    Returns:
        Tuple of (success, ExecutionSummary)
    """
    summary = ExecutionSummary()

    if not operations:
        summary.set_error("No operations specified")
        return False, summary

    # Validate input exists
    if not Path(input_path).exists():
        summary.set_error(f"Input file not found: {input_path}")
        return False, summary

    # Create temp directory for intermediates
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        current_input = input_path

        for idx, op_spec in enumerate(operations):
            op_type = op_spec.get("type")
            op_id = op_spec.get("id")
            op_params = op_spec.get("params", {})

            # Determine output path: last operation goes to final output, others to temp
            if idx == len(operations) - 1:
                current_output = output_path
            else:
                current_output = str(temp_path / f"intermediate_{idx}.pdf")

            # Execute operation based on type
            if op_type == "transform":
                success, details = execute_transform(
                    operation_id=op_id,
                    input_path=current_input,
                    output_path=current_output,
                    params=op_params,
                )
                summary.add_operation("transform", op_id, success, details)

            elif op_type == "stamp":
                placement = op_spec.get("placement", "bottom_right")
                success, details = execute_stamp(
                    operation_id=op_id,
                    input_path=current_input,
                    output_path=current_output,
                    placement=placement,
                    params=op_params,
                )
                summary.add_operation("stamp", op_id, success, details)

            else:
                summary.add_operation(
                    "unknown",
                    op_id or "unknown",
                    False,
                    {"error": "invalid_operation_type", "message": f"Invalid operation type: {op_type}"},
                )
                return False, summary

            # If operation failed, stop chain
            if not success:
                return False, summary

            # Update current_input for next operation
            current_input = current_output

    summary.final_output_path = output_path
    return True, summary


# --- Command Handlers ---


def handle_transform(args: argparse.Namespace) -> int:
    """Handle transform subcommand.

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    ensure_registries_initialized()

    # Parse params JSON
    try:
        params = json.loads(args.params) if args.params else {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in --params: {e}", file=sys.stderr)
        return 1

    # Execute transform
    success, details = execute_transform(
        operation_id=args.operation_id,
        input_path=args.input,
        output_path=args.output,
        params=params,
    )

    # Build summary
    summary = ExecutionSummary()
    summary.add_operation("transform", args.operation_id, success, details)
    if success:
        summary.final_output_path = args.output
    else:
        summary.set_error(details.get("message", "Transform operation failed"))

    # Output JSON if requested
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    if args.json_stdout:
        print(json.dumps(summary.to_dict(), indent=2))
    elif success:
        print(f"✓ Transform successful: {args.output}")
    else:
        print(f"✗ Transform failed: {details.get('message', 'Unknown error')}", file=sys.stderr)
        if "validation_errors" in details:
            for error in details["validation_errors"]:
                print(f"  - {error}", file=sys.stderr)

    return 0 if success else 1


def handle_stamp(args: argparse.Namespace) -> int:
    """Handle stamp subcommand.

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    ensure_registries_initialized()

    # Parse params JSON
    try:
        params = json.loads(args.params) if args.params else {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in --params: {e}", file=sys.stderr)
        return 1

    # Execute stamp
    success, details = execute_stamp(
        operation_id=args.operation_id,
        input_path=args.input,
        output_path=args.output,
        placement=args.placement,
        params=params,
    )

    # Build summary
    summary = ExecutionSummary()
    summary.add_operation("stamp", args.operation_id, success, details)
    if success:
        summary.final_output_path = args.output
    else:
        summary.set_error(details.get("message", "Stamp operation failed"))

    # Output JSON if requested
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    if args.json_stdout:
        print(json.dumps(summary.to_dict(), indent=2))
    elif success:
        print(f"✓ Stamp successful: {args.output}")
        if details.get("stamp_values"):
            print(f"  Stamp values: {', '.join(details['stamp_values'][:3])}...")
    else:
        print(f"✗ Stamp failed: {details.get('message', 'Unknown error')}", file=sys.stderr)
        if "validation_errors" in details:
            for error in details["validation_errors"]:
                print(f"  - {error}", file=sys.stderr)

    return 0 if success else 1


def handle_chain(args: argparse.Namespace) -> int:
    """Handle chain subcommand.

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    ensure_registries_initialized()

    # Parse operations JSON
    try:
        operations = json.loads(args.operations)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in --operations: {e}", file=sys.stderr)
        return 1

    if not isinstance(operations, list):
        print("Error: --operations must be a JSON array", file=sys.stderr)
        return 1

    # Execute chain
    success, summary = execute_chain(
        input_path=args.input,
        output_path=args.output,
        operations=operations,
    )

    # Output JSON if requested
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    if args.json_stdout:
        print(json.dumps(summary.to_dict(), indent=2))
    elif success:
        print(f"✓ Chain successful ({len(operations)} operations): {args.output}")
    else:
        print(f"✗ Chain failed: {summary.error_message or 'Unknown error'}", file=sys.stderr)
        for op in summary.operations:
            status = "✓" if op["success"] else "✗"
            print(f"  {status} {op['type']} {op['operation_id']}", file=sys.stderr)

    return 0 if success else 1


def handle_list(args: argparse.Namespace) -> int:
    """Handle list subcommand to show available operations.

    Returns:
        Exit code: 0 (always succeeds)
    """
    ensure_registries_initialized()

    transform_registry = get_transform_registry()
    stamp_registry = get_stamp_registry()

    if args.json_stdout:
        output = {
            "transforms": transform_registry.list_all_metadata(),
            "stamps": stamp_registry.list_all_metadata(),
        }
        print(json.dumps(output, indent=2))
    else:
        print("Available Transform Operations:")
        for metadata in transform_registry.list_all_metadata():
            print(f"  - {metadata['name']}: {metadata['description']}")

        print("\nAvailable Stamp Operations:")
        for metadata in stamp_registry.list_all_metadata():
            print(f"  - {metadata['name']}: {metadata['description']}")

    return 0


# --- Argument Parsing ---


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CLI for transform and stamp operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Transform subcommand
    transform_parser = subparsers.add_parser(
        "transform", help="Execute a single transform operation"
    )
    transform_parser.add_argument(
        "operation_id", help="Transform operation name (e.g., pdf_rotate)"
    )
    transform_parser.add_argument("--input", required=True, help="Input file path")
    transform_parser.add_argument("--output", required=True, help="Output file path")
    transform_parser.add_argument(
        "--params", default="{}", help="Operation parameters as JSON string"
    )
    transform_parser.add_argument(
        "--json-output", help="Write execution summary JSON to file"
    )
    transform_parser.add_argument(
        "--json-stdout", action="store_true", help="Output JSON to stdout"
    )

    # Stamp subcommand
    stamp_parser = subparsers.add_parser(
        "stamp", help="Execute a single stamp operation"
    )
    stamp_parser.add_argument(
        "operation_id", help="Stamp operation name (e.g., bates)"
    )
    stamp_parser.add_argument("--input", required=True, help="Input PDF file path")
    stamp_parser.add_argument("--output", required=True, help="Output file path")
    stamp_parser.add_argument(
        "--placement",
        default="bottom_right",
        help="Stamp placement (default: bottom_right)",
    )
    stamp_parser.add_argument(
        "--params", default="{}", help="Operation parameters as JSON string"
    )
    stamp_parser.add_argument(
        "--json-output", help="Write execution summary JSON to file"
    )
    stamp_parser.add_argument(
        "--json-stdout", action="store_true", help="Output JSON to stdout"
    )

    # Chain subcommand
    chain_parser = subparsers.add_parser(
        "chain", help="Execute a chain of transform and stamp operations"
    )
    chain_parser.add_argument("--input", required=True, help="Input file path")
    chain_parser.add_argument("--output", required=True, help="Final output file path")
    chain_parser.add_argument(
        "--operations",
        required=True,
        help="JSON array of operations with type, id, params, placement",
    )
    chain_parser.add_argument(
        "--json-output", help="Write execution summary JSON to file"
    )
    chain_parser.add_argument(
        "--json-stdout", action="store_true", help="Output JSON to stdout"
    )

    # List subcommand
    list_parser = subparsers.add_parser(
        "list", help="List available operations"
    )
    list_parser.add_argument(
        "--json-stdout", action="store_true", help="Output JSON to stdout"
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Main entry point.

    Args:
        argv: Command line arguments

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return 0 if code == 0 else 1

    if not args.command:
        print("Error: No command specified. Use --help for usage.", file=sys.stderr)
        return 1

    # Route to command handler
    if args.command == "transform":
        return handle_transform(args)
    elif args.command == "stamp":
        return handle_stamp(args)
    elif args.command == "chain":
        return handle_chain(args)
    elif args.command == "list":
        return handle_list(args)
    else:
        print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
