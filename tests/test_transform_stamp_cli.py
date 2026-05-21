"""Tests for transform_stamp_cli.py script.

Validates CLI script functionality for transform, stamp, and chain operations.
Tests both successful execution and failure modes with appropriate error handling.

Run with: python -m pytest tests/test_transform_stamp_cli.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import transform_stamp_cli as cli

# --- Fixtures ---


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF file for testing."""
    # Use existing test fixture if available
    fixture_pdf = Path(__file__).parent / "fixtures" / "sample.pdf"
    if fixture_pdf.exists():
        return str(fixture_pdf)
    
    # Fallback: create minimal PDF (requires PyMuPDF)
    try:
        import fitz
        pdf_path = tmp_path / "test_input.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)  # Letter size
        page.insert_text((100, 100), "Test Page 1")
        page = doc.new_page(width=612, height=792)
        page.insert_text((100, 100), "Test Page 2")
        doc.save(str(pdf_path))
        doc.close()
        return str(pdf_path)
    except ImportError:
        pytest.skip("PyMuPDF required for PDF creation")


@pytest.fixture
def output_path(tmp_path):
    """Generate output file path."""
    return str(tmp_path / "output.pdf")


@pytest.fixture
def json_output_path(tmp_path):
    """Generate JSON output file path."""
    return str(tmp_path / "output.json")


# --- Registry Tests ---


def test_ensure_registries_initialized():
    """Test that registries are properly initialized."""
    cli.ensure_registries_initialized()
    
    from ocr_distributed.stamps.registry import get_stamp_registry
    from ocr_distributed.transforms.registry import get_transform_registry
    
    transform_registry = get_transform_registry()
    stamp_registry = get_stamp_registry()
    
    # Should have built-in operations registered
    assert len(transform_registry.list_operations()) > 0
    assert len(stamp_registry.list_operations()) > 0


# --- Transform Tests ---


def test_execute_transform_success(sample_pdf, output_path):
    """Test successful transform execution."""
    cli.ensure_registries_initialized()
    
    success, details = cli.execute_transform(
        operation_id="pdf_rotate",
        input_path=sample_pdf,
        output_path=output_path,
        params={"angle": 90},
    )
    
    assert success is True
    assert details["output_path"] == output_path
    assert details["pages_processed"] > 0
    assert Path(output_path).exists()


def test_execute_transform_invalid_operation(sample_pdf, output_path):
    """Test transform with invalid operation ID."""
    cli.ensure_registries_initialized()
    
    success, details = cli.execute_transform(
        operation_id="nonexistent_operation",
        input_path=sample_pdf,
        output_path=output_path,
        params={},
    )
    
    assert success is False
    assert details["error"] == "operation_not_found"
    assert "nonexistent_operation" in details["message"]
    assert "available_operations" in details


def test_execute_transform_missing_input(output_path):
    """Test transform with missing input file."""
    cli.ensure_registries_initialized()
    
    success, details = cli.execute_transform(
        operation_id="pdf_rotate",
        input_path="nonexistent_file.pdf",
        output_path=output_path,
        params={"angle": 90},
    )
    
    assert success is False
    assert details["error"] == "input_not_found"


def test_execute_transform_invalid_params(sample_pdf, output_path):
    """Test transform with invalid parameters."""
    cli.ensure_registries_initialized()
    
    success, details = cli.execute_transform(
        operation_id="pdf_rotate",
        input_path=sample_pdf,
        output_path=output_path,
        params={"angle": 45},  # Invalid angle
    )
    
    assert success is False
    assert details["error"] == "config_validation_error"
    assert "validation_errors" in details


# --- Stamp Tests ---


def test_execute_stamp_success(sample_pdf, output_path):
    """Test successful stamp execution."""
    cli.ensure_registries_initialized()
    
    success, details = cli.execute_stamp(
        operation_id="bates",
        input_path=sample_pdf,
        output_path=output_path,
        placement="bottom_right",
        params={"prefix": "TEST", "start_number": 1000},
    )
    
    assert success is True
    assert details["output_path"] == output_path
    assert details["pages_stamped"] > 0
    assert len(details["stamp_values"]) > 0
    assert Path(output_path).exists()


def test_execute_stamp_invalid_operation(sample_pdf, output_path):
    """Test stamp with invalid operation ID."""
    cli.ensure_registries_initialized()
    
    success, details = cli.execute_stamp(
        operation_id="nonexistent_stamp",
        input_path=sample_pdf,
        output_path=output_path,
        placement="bottom_right",
        params={},
    )
    
    assert success is False
    assert details["error"] == "operation_not_found"
    assert "nonexistent_stamp" in details["message"]


def test_execute_stamp_invalid_placement(sample_pdf, output_path):
    """Test stamp with invalid placement."""
    cli.ensure_registries_initialized()
    
    success, details = cli.execute_stamp(
        operation_id="bates",
        input_path=sample_pdf,
        output_path=output_path,
        placement="invalid_placement",
        params={"prefix": "TEST"},
    )
    
    assert success is False
    assert details["error"] == "invalid_placement"


def test_execute_stamp_missing_required_params(sample_pdf, output_path):
    """Test stamp with missing required parameters."""
    cli.ensure_registries_initialized()
    
    success, details = cli.execute_stamp(
        operation_id="designation",
        input_path=sample_pdf,
        output_path=output_path,
        placement="top_center",
        params={},  # Missing required 'text' param
    )
    
    assert success is False
    assert details["error"] == "config_validation_error"


# --- Chain Tests ---


def test_execute_chain_success(sample_pdf, output_path):
    """Test successful chained operations."""
    cli.ensure_registries_initialized()
    
    operations = [
        {
            "type": "transform",
            "id": "pdf_rotate",
            "params": {"angle": 90},
        },
        {
            "type": "stamp",
            "id": "bates",
            "placement": "bottom_right",
            "params": {"prefix": "CHAIN", "start_number": 1},
        },
    ]
    
    success, summary = cli.execute_chain(
        input_path=sample_pdf,
        output_path=output_path,
        operations=operations,
    )
    
    assert success is True
    assert len(summary.operations) == 2
    assert summary.operations[0]["type"] == "transform"
    assert summary.operations[0]["success"] is True
    assert summary.operations[1]["type"] == "stamp"
    assert summary.operations[1]["success"] is True
    assert summary.final_output_path == output_path
    assert Path(output_path).exists()


def test_execute_chain_empty_operations(sample_pdf, output_path):
    """Test chain with empty operations list."""
    cli.ensure_registries_initialized()
    
    success, summary = cli.execute_chain(
        input_path=sample_pdf,
        output_path=output_path,
        operations=[],
    )
    
    assert success is False
    assert summary.error_message == "No operations specified"


def test_execute_chain_missing_input(output_path):
    """Test chain with missing input file."""
    cli.ensure_registries_initialized()
    
    operations = [
        {"type": "transform", "id": "pdf_rotate", "params": {"angle": 90}},
    ]
    
    success, summary = cli.execute_chain(
        input_path="nonexistent.pdf",
        output_path=output_path,
        operations=operations,
    )
    
    assert success is False
    assert "not found" in summary.error_message.lower()


def test_execute_chain_failure_stops_chain(sample_pdf, output_path):
    """Test that chain stops on first failure."""
    cli.ensure_registries_initialized()
    
    operations = [
        {
            "type": "transform",
            "id": "pdf_rotate",
            "params": {"angle": 45},  # Invalid angle
        },
        {
            "type": "stamp",
            "id": "bates",
            "placement": "bottom_right",
            "params": {"prefix": "TEST"},
        },
    ]
    
    success, summary = cli.execute_chain(
        input_path=sample_pdf,
        output_path=output_path,
        operations=operations,
    )
    
    assert success is False
    # First operation should have failed
    assert summary.operations[0]["success"] is False
    # Second operation should not have been executed
    assert len(summary.operations) == 1


def test_execute_chain_invalid_operation_type(sample_pdf, output_path):
    """Test chain with invalid operation type."""
    cli.ensure_registries_initialized()
    
    operations = [
        {"type": "invalid_type", "id": "some_op", "params": {}},
    ]
    
    success, summary = cli.execute_chain(
        input_path=sample_pdf,
        output_path=output_path,
        operations=operations,
    )
    
    assert success is False
    assert summary.operations[0]["success"] is False
    assert "invalid_operation_type" in summary.operations[0]["details"]["error"]


# --- CLI Integration Tests ---


def test_handle_transform_success(sample_pdf, output_path, json_output_path, monkeypatch):
    """Test transform command handler."""
    argv = [
        "transform",
        "pdf_rotate",
        "--input",
        sample_pdf,
        "--output",
        output_path,
        "--params",
        '{"angle": 90}',
        "--json-output",
        json_output_path,
    ]
    
    args = cli.parse_args(argv)
    exit_code = cli.handle_transform(args)
    
    assert exit_code == 0
    assert Path(output_path).exists()
    assert Path(json_output_path).exists()
    
    # Validate JSON output
    with open(json_output_path) as f:
        result = json.load(f)
    assert result["success"] is True
    assert result["final_output_path"] == output_path


def test_handle_transform_invalid_json_params(sample_pdf, output_path, capsys):
    """Test transform with invalid JSON params."""
    argv = [
        "transform",
        "pdf_rotate",
        "--input",
        sample_pdf,
        "--output",
        output_path,
        "--params",
        "invalid json {",
    ]
    
    args = cli.parse_args(argv)
    exit_code = cli.handle_transform(args)
    
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Invalid JSON" in captured.err


def test_handle_stamp_success(sample_pdf, output_path, json_output_path):
    """Test stamp command handler."""
    argv = [
        "stamp",
        "bates",
        "--input",
        sample_pdf,
        "--output",
        output_path,
        "--placement",
        "bottom_right",
        "--params",
        '{"prefix": "ABC"}',
        "--json-output",
        json_output_path,
    ]
    
    args = cli.parse_args(argv)
    exit_code = cli.handle_stamp(args)
    
    assert exit_code == 0
    assert Path(output_path).exists()
    assert Path(json_output_path).exists()
    
    # Validate JSON output
    with open(json_output_path) as f:
        result = json.load(f)
    assert result["success"] is True


def test_handle_chain_success(sample_pdf, output_path, json_output_path):
    """Test chain command handler."""
    operations_json = json.dumps([
        {"type": "transform", "id": "pdf_rotate", "params": {"angle": 90}},
        {"type": "stamp", "id": "bates", "placement": "bottom_center", "params": {"prefix": "XYZ"}},
    ])
    
    argv = [
        "chain",
        "--input",
        sample_pdf,
        "--output",
        output_path,
        "--operations",
        operations_json,
        "--json-output",
        json_output_path,
    ]
    
    args = cli.parse_args(argv)
    exit_code = cli.handle_chain(args)
    
    assert exit_code == 0
    assert Path(output_path).exists()
    assert Path(json_output_path).exists()
    
    # Validate JSON output
    with open(json_output_path) as f:
        result = json.load(f)
    assert result["success"] is True
    assert len(result["operations"]) == 2


def test_handle_chain_invalid_json_operations(sample_pdf, output_path, capsys):
    """Test chain with invalid JSON operations."""
    argv = [
        "chain",
        "--input",
        sample_pdf,
        "--output",
        output_path,
        "--operations",
        "not valid json [",
    ]
    
    args = cli.parse_args(argv)
    exit_code = cli.handle_chain(args)
    
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Invalid JSON" in captured.err


def test_handle_list_operations(capsys):
    """Test list command handler."""
    argv = ["list"]
    
    args = cli.parse_args(argv)
    exit_code = cli.handle_list(args)
    
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Transform Operations" in captured.out
    assert "Stamp Operations" in captured.out


def test_handle_list_json_output(capsys):
    """Test list command with JSON output."""
    argv = ["list", "--json-stdout"]
    
    args = cli.parse_args(argv)
    exit_code = cli.handle_list(args)
    
    assert exit_code == 0
    captured = capsys.readouterr()
    
    result = json.loads(captured.out)
    assert "transforms" in result
    assert "stamps" in result
    assert len(result["transforms"]) > 0
    assert len(result["stamps"]) > 0


def test_main_no_command(capsys):
    """Test main with no command specified."""
    exit_code = cli.main([])
    
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "No command specified" in captured.err


def test_main_unknown_command(capsys):
    """Test main with unknown command."""
    exit_code = cli.main(["unknown_command"])
    
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


# --- ExecutionSummary Tests ---


def test_execution_summary_add_operation():
    """Test ExecutionSummary.add_operation."""
    summary = cli.ExecutionSummary()
    
    summary.add_operation(
        "transform",
        "pdf_rotate",
        True,
        {"output_path": "test.pdf", "pages_processed": 5},
    )
    
    assert len(summary.operations) == 1
    assert summary.operations[0]["type"] == "transform"
    assert summary.operations[0]["operation_id"] == "pdf_rotate"
    assert summary.operations[0]["success"] is True
    assert summary.success is True


def test_execution_summary_failure_marks_overall_failure():
    """Test that failed operation marks summary as failed."""
    summary = cli.ExecutionSummary()
    
    summary.add_operation("transform", "pdf_rotate", False, {"error": "test error"})
    
    assert summary.success is False


def test_execution_summary_set_error():
    """Test ExecutionSummary.set_error."""
    summary = cli.ExecutionSummary()
    
    summary.set_error("Test error message")
    
    assert summary.success is False
    assert summary.error_message == "Test error message"


def test_execution_summary_to_dict():
    """Test ExecutionSummary.to_dict serialization."""
    summary = cli.ExecutionSummary()
    summary.add_operation("transform", "pdf_rotate", True, {"output_path": "test.pdf"})
    summary.final_output_path = "final.pdf"
    
    result = summary.to_dict()
    
    assert result["success"] is True
    assert len(result["operations"]) == 1
    assert result["final_output_path"] == "final.pdf"
    assert result["error_message"] is None


# --- Argument Parsing Tests ---


def test_parse_args_transform():
    """Test argument parsing for transform command."""
    argv = [
        "transform",
        "pdf_rotate",
        "--input",
        "input.pdf",
        "--output",
        "output.pdf",
        "--params",
        '{"angle": 90}',
    ]
    
    args = cli.parse_args(argv)
    
    assert args.command == "transform"
    assert args.operation_id == "pdf_rotate"
    assert args.input == "input.pdf"
    assert args.output == "output.pdf"
    assert args.params == '{"angle": 90}'


def test_parse_args_stamp():
    """Test argument parsing for stamp command."""
    argv = [
        "stamp",
        "bates",
        "--input",
        "input.pdf",
        "--output",
        "output.pdf",
        "--placement",
        "top_center",
        "--params",
        '{"prefix": "ABC"}',
    ]
    
    args = cli.parse_args(argv)
    
    assert args.command == "stamp"
    assert args.operation_id == "bates"
    assert args.input == "input.pdf"
    assert args.output == "output.pdf"
    assert args.placement == "top_center"
    assert args.params == '{"prefix": "ABC"}'


def test_parse_args_chain():
    """Test argument parsing for chain command."""
    operations = '[{"type": "transform", "id": "pdf_rotate"}]'
    argv = [
        "chain",
        "--input",
        "input.pdf",
        "--output",
        "output.pdf",
        "--operations",
        operations,
    ]
    
    args = cli.parse_args(argv)
    
    assert args.command == "chain"
    assert args.input == "input.pdf"
    assert args.output == "output.pdf"
    assert args.operations == operations


def test_parse_args_list():
    """Test argument parsing for list command."""
    argv = ["list", "--json-stdout"]
    
    args = cli.parse_args(argv)
    
    assert args.command == "list"
    assert args.json_stdout is True


# --- Windows Path Compatibility Tests ---


def test_windows_paths_transform(sample_pdf, tmp_path):
    """Test transform with Windows-style paths."""
    cli.ensure_registries_initialized()
    
    # Use Windows-style path separators
    output_path = str(tmp_path / "output.pdf").replace("/", "\\")
    
    success, details = cli.execute_transform(
        operation_id="pdf_rotate",
        input_path=sample_pdf,
        output_path=output_path,
        params={"angle": 90},
    )
    
    assert success is True
    assert Path(output_path).exists()


def test_windows_paths_chain(sample_pdf, tmp_path):
    """Test chain with Windows-style paths."""
    cli.ensure_registries_initialized()
    
    output_path = str(tmp_path / "chain_output.pdf").replace("/", "\\")
    
    operations = [
        {"type": "transform", "id": "pdf_rotate", "params": {"angle": 90}},
    ]
    
    success, summary = cli.execute_chain(
        input_path=sample_pdf,
        output_path=output_path,
        operations=operations,
    )
    
    assert success is True
    assert Path(output_path).exists()
