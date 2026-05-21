"""Phase F tests for validation gates and custody hooks."""

from __future__ import annotations

from custody import CustodyChain
from custody_hooks import (
    get_custody_diagnostics_summary,
    record_stamp_lifecycle,
    record_transform_lifecycle,
)
from validation_gates import (
    validate_bates_continuity,
    validate_output_reference_integrity,
    validate_stamp_output,
    validate_stamp_placement_no_conflicts,
    validate_transform_output,
)


def test_validate_output_reference_integrity_success(tmp_path):
    output_file = tmp_path / "output.pdf"
    output_file.write_bytes(b"pdf-content")

    result = validate_output_reference_integrity(str(output_file))

    assert result.passed is True
    assert result.details["file_size_bytes"] > 0
    assert "sha256_hash" in result.details


def test_validate_output_reference_integrity_missing(tmp_path):
    result = validate_output_reference_integrity(str(tmp_path / "missing.pdf"))
    assert result.passed is False


def test_validate_bates_continuity_detects_gap():
    result = validate_bates_continuity(["PROD000001", "PROD000003"], prefix="PROD")
    assert result.passed is False
    assert "gap" in result.message.lower()


def test_validate_stamp_conflict_detection():
    result = validate_stamp_placement_no_conflicts(
        ["Page 1: Stamp overlaps text zone by 20.0%"]
    )
    assert result.passed is False


def test_validate_stamp_output_success(tmp_path):
    output_file = tmp_path / "stamp.pdf"
    output_file.write_bytes(b"stamp-content")

    passed, diagnostics = validate_stamp_output(
        output_path=str(output_file),
        stamp_values=["BATES000001", "BATES000002"],
        warnings=[],
        prefix="BATES",
    )

    assert passed is True
    assert diagnostics["all_gates_passed"] is True


def test_record_transform_lifecycle_success(tmp_path):
    input_file = tmp_path / "input.pdf"
    output_file = tmp_path / "output.pdf"
    input_file.write_bytes(b"input")
    output_file.write_bytes(b"output")

    chain = CustodyChain("doc1", str(input_file))
    diagnostics = record_transform_lifecycle(
        custody_chain=chain,
        operation_id="pdf_extract",
        input_path=str(input_file),
        output_path=str(output_file),
        params={"pages": [1]},
        success=True,
    )

    summary = get_custody_diagnostics_summary(diagnostics)
    assert summary["custody_recorded"] is True
    assert "input_hash" in summary
    assert "output_hash" in summary
    assert chain.verify_chain()[0] is True


def test_record_stamp_lifecycle_success(tmp_path):
    input_file = tmp_path / "input.pdf"
    output_file = tmp_path / "output.pdf"
    input_file.write_bytes(b"input")
    output_file.write_bytes(b"output")

    chain = CustodyChain("doc2", str(input_file))
    diagnostics = record_stamp_lifecycle(
        custody_chain=chain,
        operation_id="bates",
        input_path=str(input_file),
        output_path=str(output_file),
        placement="bottom_right",
        params={"prefix": "BATES"},
        success=True,
        stamp_values=["BATES000001"],
    )

    summary = get_custody_diagnostics_summary(diagnostics)
    assert summary["operation_type"] == "stamp"
    assert "output_hash" in summary
    assert chain.verify_chain()[0] is True


def test_validate_transform_output_success(tmp_path):
    output_file = tmp_path / "transform.pdf"
    output_file.write_bytes(b"transform-content")

    passed, diagnostics = validate_transform_output(str(output_file))

    assert passed is True
    assert diagnostics["passed"] is True
