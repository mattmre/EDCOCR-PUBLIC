"""Tests for transform and stamp API endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with isolated DB and temp dirs."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with patch("api.config.SOURCE_FOLDER", str(source)), patch(
        "api.config.OUTPUT_FOLDER", str(output)
    ), patch("api.job_manager.config") as mock_config:
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        yield TestClient(app)


@pytest.fixture()
def sample_pdf(tmp_path):
    """Create a minimal PDF for testing."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not available")

    pdf_path = tmp_path / "source" / "sample.pdf"
    pdf_path.parent.mkdir(exist_ok=True)

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Test PDF Page 1", fontsize=12)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Test PDF Page 2", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    return str(pdf_path)


# ---------------------------------------------------------------------------
# Transform API Tests
# ---------------------------------------------------------------------------


def test_list_transforms_disabled(client):
    """Transform listing fails when feature is disabled."""
    with patch("api.config.ENABLE_TRANSFORMS", False):
        resp = client.get("/api/v1/transforms")
        assert resp.status_code == 403
        body = resp.json()
        assert body["detail"]["error"] == "feature_disabled"
        assert "ENABLE_TRANSFORMS" in body["detail"]["message"]


def test_list_transforms_enabled(client):
    """Transform listing succeeds when feature is enabled."""
    with patch("api.config.ENABLE_TRANSFORMS", True):
        resp = client.get("/api/v1/transforms")
        assert resp.status_code == 200
        body = resp.json()
        assert "operations" in body
        assert "total" in body
        assert body["total"] >= 0

        # Check operation structure if any operations are registered
        if body["total"] > 0:
            op = body["operations"][0]
            assert "name" in op
            assert "description" in op
            assert "version" in op
            assert "supported_formats" in op
            assert "parameters" in op


def test_get_transform_metadata_disabled(client):
    """Transform metadata lookup fails when feature is disabled."""
    with patch("api.config.ENABLE_TRANSFORMS", False):
        resp = client.get("/api/v1/transforms/pdf_extract")
        assert resp.status_code == 403
        body = resp.json()
        assert body["detail"]["error"] == "feature_disabled"


def test_get_transform_metadata_not_found(client):
    """Transform metadata returns 404 for unknown operation."""
    with patch("api.config.ENABLE_TRANSFORMS", True):
        resp = client.get("/api/v1/transforms/nonexistent_operation")
        assert resp.status_code == 404
        body = resp.json()
        assert body["detail"]["error"] == "operation_not_found"
        assert "nonexistent_operation" in body["detail"]["message"]


def test_get_transform_metadata_success(client):
    """Transform metadata lookup succeeds for known operation."""
    with patch("api.config.ENABLE_TRANSFORMS", True):
        # Try to get any registered operation
        list_resp = client.get("/api/v1/transforms")
        if list_resp.json()["total"] == 0:
            pytest.skip("No transform operations registered")

        op_name = list_resp.json()["operations"][0]["name"]
        resp = client.get(f"/api/v1/transforms/{op_name}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == op_name
        assert "description" in body
        assert "version" in body


def test_execute_transform_disabled(client):
    """Transform execution fails when feature is disabled."""
    with patch("api.config.ENABLE_TRANSFORMS", False):
        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_extract",
                "input_path": "C:\\test\\input.pdf",
                "output_path": "C:\\test\\output.pdf",
            },
        )
        assert resp.status_code == 403
        body = resp.json()
        assert body["detail"]["error"] == "feature_disabled"


def test_execute_transform_operation_not_found(client):
    """Transform execution fails for unknown operation."""
    with patch("api.config.ENABLE_TRANSFORMS", True):
        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "nonexistent_op",
                "input_path": "C:\\test\\input.pdf",
                "output_path": "C:\\test\\output.pdf",
            },
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["detail"]["error"] == "operation_not_found"


def test_execute_transform_input_not_found(client, tmp_path):
    """Transform execution fails when input file doesn't exist."""
    with patch("api.config.ENABLE_TRANSFORMS", True):
        nonexistent_path = str(tmp_path / "source" / "nonexistent.pdf")
        output_path = str(tmp_path / "output" / "result.pdf")

        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_extract",
                "input_path": nonexistent_path,
                "output_path": output_path,
            },
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["detail"]["error"] == "input_not_found"


def test_execute_transform_rejects_input_outside_allowed_roots(client, sample_pdf, tmp_path):
    """Transform execution rejects existing input files outside configured roots."""
    with patch("api.config.ENABLE_TRANSFORMS", True):
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir(exist_ok=True)
        outside_input = outside_dir / "outside.pdf"
        outside_input.write_bytes(Path(sample_pdf).read_bytes())
        output_path = str(tmp_path / "output" / "result.pdf")

        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_extract",
                "input_path": str(outside_input),
                "output_path": output_path,
                "params": {"pages": [1]},
            },
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "path_not_allowed"


def test_execute_transform_rejects_output_outside_allowed_roots(client, sample_pdf, tmp_path):
    """Transform execution rejects output paths outside configured roots."""
    with patch("api.config.ENABLE_TRANSFORMS", True):
        outside_output = tmp_path / "outside" / "result.pdf"

        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_extract",
                "input_path": sample_pdf,
                "output_path": str(outside_output),
                "params": {"pages": [1]},
            },
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "path_not_allowed"


def test_execute_transform_validation_failure(client, sample_pdf, tmp_path):
    """Transform execution fails when config validation fails."""
    with patch("api.config.ENABLE_TRANSFORMS", True):
        output_path = str(tmp_path / "output" / "result.pdf")

        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_extract",
                "input_path": sample_pdf,
                "output_path": output_path,
                "params": {},  # Missing required 'pages' parameter
            },
        )
        # Should fail validation
        assert resp.status_code == 400
        body = resp.json()
        assert body["detail"]["error"] == "validation_failed"


def test_execute_transform_success(client, sample_pdf, tmp_path):
    """Transform execution succeeds with valid inputs."""
    with patch("api.config.ENABLE_TRANSFORMS", True):
        output_path = str(tmp_path / "output" / "result.pdf")

        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_extract",
                "input_path": sample_pdf,
                "output_path": output_path,
                "params": {"pages": [1]},  # Extract first page only
            },
        )

        if resp.status_code != 200:
            print(f"Response: {resp.json()}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["operation_id"] == "pdf_extract"
        assert body["output_path"] == output_path
        assert body["pages_processed"] >= 1
        assert Path(output_path).exists()


# ---------------------------------------------------------------------------
# Stamp API Tests
# ---------------------------------------------------------------------------


def test_list_stamps_disabled(client):
    """Stamp listing fails when feature is disabled."""
    with patch("api.config.ENABLE_STAMPING", False):
        resp = client.get("/api/v1/stamps")
        assert resp.status_code == 403
        body = resp.json()
        assert body["detail"]["error"] == "feature_disabled"
        assert "ENABLE_STAMPING" in body["detail"]["message"]


def test_list_stamps_enabled(client):
    """Stamp listing succeeds when feature is enabled."""
    with patch("api.config.ENABLE_STAMPING", True):
        resp = client.get("/api/v1/stamps")
        assert resp.status_code == 200
        body = resp.json()
        assert "operations" in body
        assert "total" in body
        assert body["total"] >= 0

        # Check operation structure if any operations are registered
        if body["total"] > 0:
            op = body["operations"][0]
            assert "name" in op
            assert "description" in op
            assert "version" in op
            assert "supported_formats" in op
            assert "parameters" in op


def test_get_stamp_metadata_disabled(client):
    """Stamp metadata lookup fails when feature is disabled."""
    with patch("api.config.ENABLE_STAMPING", False):
        resp = client.get("/api/v1/stamps/bates")
        assert resp.status_code == 403
        body = resp.json()
        assert body["detail"]["error"] == "feature_disabled"


def test_get_stamp_metadata_not_found(client):
    """Stamp metadata returns 404 for unknown operation."""
    with patch("api.config.ENABLE_STAMPING", True):
        resp = client.get("/api/v1/stamps/nonexistent_operation")
        assert resp.status_code == 404
        body = resp.json()
        assert body["detail"]["error"] == "operation_not_found"
        assert "nonexistent_operation" in body["detail"]["message"]


def test_get_stamp_metadata_success(client):
    """Stamp metadata lookup succeeds for known operation."""
    with patch("api.config.ENABLE_STAMPING", True):
        # Try to get any registered operation
        list_resp = client.get("/api/v1/stamps")
        if list_resp.json()["total"] == 0:
            pytest.skip("No stamp operations registered")

        op_name = list_resp.json()["operations"][0]["name"]
        resp = client.get(f"/api/v1/stamps/{op_name}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == op_name
        assert "description" in body
        assert "version" in body


def test_execute_stamp_disabled(client):
    """Stamp execution fails when feature is disabled."""
    with patch("api.config.ENABLE_STAMPING", False):
        resp = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "bates",
                "input_path": "C:\\test\\input.pdf",
                "output_path": "C:\\test\\output.pdf",
            },
        )
        assert resp.status_code == 403
        body = resp.json()
        assert body["detail"]["error"] == "feature_disabled"


def test_execute_stamp_operation_not_found(client):
    """Stamp execution fails for unknown operation."""
    with patch("api.config.ENABLE_STAMPING", True):
        resp = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "nonexistent_op",
                "input_path": "C:\\test\\input.pdf",
                "output_path": "C:\\test\\output.pdf",
            },
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["detail"]["error"] == "operation_not_found"


def test_execute_stamp_input_not_found(client, tmp_path):
    """Stamp execution fails when input file doesn't exist."""
    with patch("api.config.ENABLE_STAMPING", True):
        nonexistent_path = str(tmp_path / "source" / "nonexistent.pdf")
        output_path = str(tmp_path / "output" / "result.pdf")

        resp = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "bates",
                "input_path": nonexistent_path,
                "output_path": output_path,
            },
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["detail"]["error"] == "input_not_found"


def test_execute_stamp_rejects_input_outside_allowed_roots(client, sample_pdf, tmp_path):
    """Stamp execution rejects existing input files outside configured roots."""
    with patch("api.config.ENABLE_STAMPING", True):
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir(exist_ok=True)
        outside_input = outside_dir / "outside.pdf"
        outside_input.write_bytes(Path(sample_pdf).read_bytes())
        output_path = str(tmp_path / "output" / "result.pdf")

        resp = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "bates",
                "input_path": str(outside_input),
                "output_path": output_path,
                "params": {"prefix": "TEST", "start": 1},
            },
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "path_not_allowed"


def test_execute_stamp_rejects_output_outside_allowed_roots(client, sample_pdf, tmp_path):
    """Stamp execution rejects output paths outside configured roots."""
    with patch("api.config.ENABLE_STAMPING", True):
        outside_output = tmp_path / "outside" / "result.pdf"

        resp = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "bates",
                "input_path": sample_pdf,
                "output_path": str(outside_output),
                "params": {"prefix": "TEST", "start": 1},
            },
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "path_not_allowed"


def test_execute_stamp_validation_failure(client, sample_pdf, tmp_path):
    """Stamp execution fails when config validation fails."""
    with patch("api.config.ENABLE_STAMPING", True):
        output_path = str(tmp_path / "output" / "result.pdf")

        resp = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "bates",
                "input_path": sample_pdf,
                "output_path": output_path,
                "params": {"start": 0, "width": 6},  # Invalid start (must be >= 1)
            },
        )
        # Should fail validation
        assert resp.status_code == 400
        body = resp.json()
        assert body["detail"]["error"] == "validation_failed"


def test_execute_stamp_success(client, sample_pdf, tmp_path):
    """Stamp execution succeeds with valid inputs."""
    with patch("api.config.ENABLE_STAMPING", True):
        output_path = str(tmp_path / "output" / "result.pdf")

        resp = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "bates",
                "input_path": sample_pdf,
                "output_path": output_path,
                "placement": "bottom_right",
                "params": {"prefix": "TEST", "start": 1000},
            },
        )

        if resp.status_code != 200:
            print(f"Response: {resp.json()}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["operation_id"] == "bates"
        assert body["output_path"] == output_path
        assert body["pages_stamped"] >= 2
        assert len(body["stamp_values"]) >= 2
        assert Path(output_path).exists()


def test_execute_stamp_with_placement(client, sample_pdf, tmp_path):
    """Stamp execution respects placement parameter."""
    with patch("api.config.ENABLE_STAMPING", True):
        output_path = str(tmp_path / "output" / "result_top.pdf")

        resp = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "bates",
                "input_path": sample_pdf,
                "output_path": output_path,
                "placement": "top_center",
                "params": {"prefix": "TOP", "start": 1},
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert Path(output_path).exists()


def test_execute_stamp_designation(client, sample_pdf, tmp_path):
    """Designation stamp execution succeeds."""
    with patch("api.config.ENABLE_STAMPING", True):
        output_path = str(tmp_path / "output" / "result_designation.pdf")

        resp = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "designation",
                "input_path": sample_pdf,
                "output_path": output_path,
                "placement": "top_center",
                "params": {
                    "text": "CONFIDENTIAL",
                    "font_size": 14,
                    "font_color": [1.0, 0.0, 0.0],
                },
            },
        )

        if resp.status_code != 200:
            print(f"Response: {resp.json()}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["operation_id"] == "designation"
        assert Path(output_path).exists()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


def test_transform_then_stamp(client, sample_pdf, tmp_path):
    """Chain transform and stamp operations."""
    with patch("api.config.ENABLE_TRANSFORMS", True), patch(
        "api.config.ENABLE_STAMPING", True
    ):
        # First, extract a page
        extracted_path = str(tmp_path / "output" / "extracted.pdf")
        resp1 = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_extract",
                "input_path": sample_pdf,
                "output_path": extracted_path,
                "params": {"pages": [1]},
            },
        )
        assert resp1.status_code == 200
        assert Path(extracted_path).exists()

        # Then, stamp the extracted page
        stamped_path = str(tmp_path / "output" / "stamped.pdf")
        resp2 = client.post(
            "/api/v1/stamps/execute",
            json={
                "operation_id": "bates",
                "input_path": extracted_path,
                "output_path": stamped_path,
                "params": {"prefix": "CHAIN", "start_number": 1},
            },
        )
        assert resp2.status_code == 200
        body2 = resp2.json()
        assert body2["success"] is True
        assert body2["pages_stamped"] == 1
        assert Path(stamped_path).exists()
