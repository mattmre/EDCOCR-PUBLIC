"""Phase F CLI integration tests for custody/hash diagnostics."""

from __future__ import annotations

import pytest

from scripts.transform_stamp_cli import (  # noqa: E402
    ensure_registries_initialized,
    execute_stamp,
    execute_transform,
)


@pytest.fixture(autouse=True)
def _init_registries():
    ensure_registries_initialized()


@pytest.fixture()
def sample_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Phase F sample")
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Phase F page 2")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_cli_transform_includes_custody_hash(sample_pdf, tmp_path):
    output_path = str(tmp_path / "extract.pdf")
    success, details = execute_transform(
        operation_id="pdf_extract",
        input_path=sample_pdf,
        output_path=output_path,
        params={"pages": [1]},
        enable_custody=True,
    )

    assert success is True
    assert details["custody"]["operation_type"] == "transform"
    assert "input_hash" in details["custody"]
    assert "output_hash" in details["custody"]


def test_cli_stamp_includes_custody_hash(sample_pdf, tmp_path):
    output_path = str(tmp_path / "stamped.pdf")
    success, details = execute_stamp(
        operation_id="bates",
        input_path=sample_pdf,
        output_path=output_path,
        placement="bottom_right",
        params={"prefix": "TST", "start": 1},
        enable_custody=True,
    )

    assert success is True
    assert details["custody"]["operation_type"] == "stamp"
    assert "input_hash" in details["custody"]
    assert "output_hash" in details["custody"]


def test_cli_transform_custody_can_be_disabled(sample_pdf, tmp_path):
    output_path = str(tmp_path / "extract-no-custody.pdf")
    success, details = execute_transform(
        operation_id="pdf_extract",
        input_path=sample_pdf,
        output_path=output_path,
        params={"pages": [1]},
        enable_custody=False,
    )

    assert success is True
    assert details["custody"] is None
