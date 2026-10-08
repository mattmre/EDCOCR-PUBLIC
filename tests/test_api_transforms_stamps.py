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

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.ANONYMOUS_ROLE", "operator"),
        patch("api.auth.ANONYMOUS_ROLE", "operator"),
        patch("api.job_manager.config") as mock_config,
    ):
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


def test_execute_transform_rejects_input_outside_allowed_roots(
    client, sample_pdf, tmp_path
):
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


def test_execute_transform_rejects_output_outside_allowed_roots(
    client, sample_pdf, tmp_path
):
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
    """Transform execution writes its forensic chain to the ops namespace."""
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
        custody_ops = tmp_path / "output" / "EXPORT" / "CUSTODY" / "ops"
        custody_files = list(custody_ops.glob("*.custody.jsonl"))
        assert len(custody_files) == 1
        assert '"event_type": "transform_start"' in custody_files[0].read_text(
            encoding="utf-8"
        )


def test_execute_transform_hashes_snapshot_not_swapped_input(
    client, sample_pdf, tmp_path
):
    """Custody must attest the bytes the engine read, not a later pathname swap."""
    from ocr_distributed.transforms.base import TransformResult

    original = Path(sample_pdf).read_bytes()
    swapped = b"%PDF-1.0 swapped-input-bytes\n%%EOF\n"

    class SnapshotOp:
        def validate_config(self, _config):
            return []

        def execute(self, input_path, output_path, _config):
            Path(sample_pdf).write_bytes(swapped)
            operated = Path(input_path).read_bytes()
            Path(output_path).write_bytes(operated)
            return TransformResult(
                success=True,
                output_path=output_path,
                pages_processed=1,
                metadata={
                    "operated_sha256": __import__("hashlib")
                    .sha256(operated)
                    .hexdigest()
                },
            )

    with (
        patch("api.config.ENABLE_TRANSFORMS", True),
        patch("api.routers.transforms.get_transform_registry") as mock_registry,
    ):
        mock_registry.return_value.get.return_value = SnapshotOp()
        output_path = str(tmp_path / "output" / "result.pdf")
        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_extract",
                "input_path": sample_pdf,
                "output_path": output_path,
                "params": {"pages": [1]},
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    input_hash = body["metadata"]["custody"]["input_hash"]
    output_hash = body["metadata"]["custody"]["output_hash"]
    assert input_hash == __import__("hashlib").sha256(original).hexdigest()
    assert output_hash == __import__("hashlib").sha256(original).hexdigest()
    assert input_hash != __import__("hashlib").sha256(swapped).hexdigest()
    assert Path(output_path).read_bytes() == original


def test_execute_transform_rejects_parent_symlink_escape(client, sample_pdf, tmp_path):
    """A swapped output parent must not receive a durable success."""
    from ocr_distributed.transforms.base import TransformResult

    output_path = str(tmp_path / "output" / "nested" / "result.pdf")
    outside = tmp_path / "outside"
    outside.mkdir()

    class EscapeOp:
        def validate_config(self, _config):
            return []

        def execute(self, _input_path, output_exec_path, _config):
            parent = Path(output_path).parent
            parent.rename(tmp_path / "output" / "nested-real")
            parent.symlink_to(outside)
            Path(output_exec_path).write_bytes(b"descriptor-bound")
            Path(output_path).write_bytes(b"escaped")
            return TransformResult(
                success=True,
                output_path=output_path,
                pages_processed=1,
            )

    with (
        patch("api.config.ENABLE_TRANSFORMS", True),
        patch("api.routers.transforms.get_transform_registry") as mock_registry,
    ):
        mock_registry.return_value.get.return_value = EscapeOp()
        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_extract",
                "input_path": sample_pdf,
                "output_path": output_path,
                "params": {"pages": [1]},
            },
        )

    assert resp.status_code == 200, resp.text
    assert Path(tmp_path / "output" / "nested-real" / "result.pdf").read_bytes() == (
        b"descriptor-bound"
    )
    assert (outside / "result.pdf").read_bytes() == b"escaped"
    body = resp.json()
    assert (
        body["metadata"]["custody"]["output_hash"]
        == __import__("hashlib").sha256(b"descriptor-bound").hexdigest()
    )


def test_bind_path_params_rewrites_extra_keys_and_ignores_lexical_swap(tmp_path):
    """Unit: extra file snapshots keep suffix; lexical swap does not change bytes."""
    from api.path_safety import bind_authorized_operation_io

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    primary = source / "in.pdf"
    extra = source / "insert-src.pdf"
    listed_a = source / "merge-a.pdf"
    listed_b = source / "merge-b.pdf"
    primary.write_bytes(b"%PDF-1.0 primary\n%%EOF\n")
    extra_bytes = b"%PDF-1.0 extra-original\n%%EOF\n"
    extra.write_bytes(extra_bytes)
    listed_a.write_bytes(b"A-orig")
    listed_b.write_bytes(b"B-orig")
    out_dir = output / "split-dest"
    out_dir.mkdir()
    outp = output / "result.pdf"
    operation_io = bind_authorized_operation_io(
        input_path=str(primary),
        output_path=str(outp),
        read_roots=(str(source), str(output)),
        write_root=str(output),
    )
    try:
        rewritten = operation_io.bind_path_params(
            {
                "source_pdf": str(extra),
                "input_files": [str(listed_a), str(listed_b)],
                "output_dir": str(out_dir),
                "pages": [1],
            },
            read_roots=(str(source), str(output)),
            write_root=str(output),
        )
        bound_source = rewritten["source_pdf"]
        assert bound_source.startswith("/proc/self/fd/") or bound_source.startswith(
            "/dev/fd/"
        )
        assert bound_source.endswith(".pdf")
        assert rewritten["output_dir"].startswith("/proc/self/fd/") or rewritten[
            "output_dir"
        ].startswith("/dev/fd/")
        extra.write_bytes(b"%PDF-1.0 extra-swapped\n%%EOF\n")
        listed_a.write_bytes(b"A-swap")
        listed_b.write_bytes(b"B-swap")
        lexical_dir = output / "split-dest-swapped"
        lexical_dir.mkdir()
        out_dir.rename(output / "split-dest-real")
        (output / "split-dest").symlink_to(lexical_dir)
        assert Path(bound_source).read_bytes() == extra_bytes
        assert Path(rewritten["input_files"][0]).read_bytes() == b"A-orig"
        assert Path(rewritten["input_files"][1]).read_bytes() == b"B-orig"
        marker = Path(rewritten["output_dir"]) / "written.bin"
        marker.write_bytes(b"bound-dir")
        assert (output / "split-dest-real" / "written.bin").read_bytes() == b"bound-dir"
        assert not (lexical_dir / "written.bin").exists()
        assert rewritten["pages"] == [1]
    finally:
        operation_io.close()


def test_execute_transform_source_pdf_uses_snapshot_not_swapped_file(
    client, sample_pdf, tmp_path
):
    """Extra source_pdf must be the snapshot, not a later lexical swap."""
    from ocr_distributed.transforms.base import TransformResult

    extra = tmp_path / "source" / "insert-src.pdf"
    original = b"%PDF-1.0 extra-original-bytes\n%%EOF\n"
    swapped = b"%PDF-1.0 extra-swapped-bytes\n%%EOF\n"
    extra.write_bytes(original)

    class ExtraSourceOp:
        def validate_config(self, _config):
            return []

        def execute(self, _input_path, output_path, config):
            extra.write_bytes(swapped)
            operated = Path(config.params["source_pdf"]).read_bytes()
            Path(output_path).write_bytes(operated)
            return TransformResult(
                success=True,
                output_path=output_path,
                pages_processed=1,
            )

    with (
        patch("api.config.ENABLE_TRANSFORMS", True),
        patch("api.routers.transforms.get_transform_registry") as mock_registry,
    ):
        mock_registry.return_value.get.return_value = ExtraSourceOp()
        output_path = str(tmp_path / "output" / "result.pdf")
        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_insert",
                "input_path": sample_pdf,
                "output_path": output_path,
                "params": {"source_pdf": str(extra), "position": 0},
            },
        )

    assert resp.status_code == 200, resp.text
    assert Path(output_path).read_bytes() == original
    assert Path(output_path).read_bytes() != swapped


def test_execute_transform_input_files_use_snapshot_not_swapped_files(
    client, sample_pdf, tmp_path
):
    """Extra input_files entries must be snapshots, not later lexical swaps."""
    from ocr_distributed.transforms.base import TransformResult

    file_a = tmp_path / "source" / "merge-a.pdf"
    file_b = tmp_path / "source" / "merge-b.pdf"
    orig_a = b"%PDF-1.0 merge-a-original\n%%EOF\n"
    orig_b = b"%PDF-1.0 merge-b-original\n%%EOF\n"
    swap_a = b"%PDF-1.0 merge-a-swapped\n%%EOF\n"
    swap_b = b"%PDF-1.0 merge-b-swapped\n%%EOF\n"
    file_a.write_bytes(orig_a)
    file_b.write_bytes(orig_b)

    class ExtraFilesOp:
        def validate_config(self, _config):
            return []

        def execute(self, _input_path, output_path, config):
            file_a.write_bytes(swap_a)
            file_b.write_bytes(swap_b)
            bound = config.params["input_files"]
            operated = Path(bound[0]).read_bytes() + Path(bound[1]).read_bytes()
            Path(output_path).write_bytes(operated)
            return TransformResult(
                success=True,
                output_path=output_path,
                pages_processed=2,
            )

    with (
        patch("api.config.ENABLE_TRANSFORMS", True),
        patch("api.routers.transforms.get_transform_registry") as mock_registry,
    ):
        mock_registry.return_value.get.return_value = ExtraFilesOp()
        output_path = str(tmp_path / "output" / "merged.pdf")
        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_merge",
                "input_path": sample_pdf,
                "output_path": output_path,
                "params": {"input_files": [str(file_a), str(file_b)]},
            },
        )

    assert resp.status_code == 200, resp.text
    assert Path(output_path).read_bytes() == orig_a + orig_b
    assert Path(output_path).read_bytes() != swap_a + swap_b


def test_execute_transform_output_dir_stays_on_bound_directory(
    client, sample_pdf, tmp_path
):
    """output_dir writes must land on the bound directory, not a lexical swap."""
    from ocr_distributed.transforms.base import TransformResult

    bound_dir = tmp_path / "output" / "bound-out"
    bound_dir.mkdir()
    swap_dir = tmp_path / "output" / "swap-out"
    swap_dir.mkdir()

    class ExtraDirOp:
        def validate_config(self, _config):
            return []

        def execute(self, _input_path, output_path, config):
            lexical = tmp_path / "output" / "bound-out"
            real = tmp_path / "output" / "bound-out-real"
            lexical.rename(real)
            lexical.symlink_to(swap_dir)
            dest = Path(config.params["output_dir"]) / "written.bin"
            dest.write_bytes(b"from-bound-dir")
            Path(output_path).write_bytes(b"primary-ok")
            return TransformResult(
                success=True,
                output_path=output_path,
                pages_processed=1,
            )

    with (
        patch("api.config.ENABLE_TRANSFORMS", True),
        patch("api.routers.transforms.get_transform_registry") as mock_registry,
    ):
        mock_registry.return_value.get.return_value = ExtraDirOp()
        output_path = str(tmp_path / "output" / "result.pdf")
        resp = client.post(
            "/api/v1/transforms/execute",
            json={
                "operation_id": "pdf_split",
                "input_path": sample_pdf,
                "output_path": output_path,
                "params": {"output_dir": str(bound_dir)},
            },
        )

    assert resp.status_code == 200, resp.text
    assert (tmp_path / "output" / "bound-out-real" / "written.bin").read_bytes() == (
        b"from-bound-dir"
    )
    assert not (swap_dir / "written.bin").exists()


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


def test_execute_stamp_rejects_input_outside_allowed_roots(
    client, sample_pdf, tmp_path
):
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


def test_execute_stamp_rejects_output_outside_allowed_roots(
    client, sample_pdf, tmp_path
):
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
    """Stamp execution writes its forensic chain to the ops namespace."""
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
        custody_ops = tmp_path / "output" / "EXPORT" / "CUSTODY" / "ops"
        custody_files = list(custody_ops.glob("*.custody.jsonl"))
        assert len(custody_files) == 1
        assert '"event_type": "stamp_start"' in custody_files[0].read_text(
            encoding="utf-8"
        )


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
    with (
        patch("api.config.ENABLE_TRANSFORMS", True),
        patch("api.config.ENABLE_STAMPING", True),
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
