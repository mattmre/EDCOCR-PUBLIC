"""Tests for output manifest and schema retrieval API endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import Job, get_engine, reset_engine
from ocr_local.document_bundle import validate_document_bundle
from ocr_local.features.custody import CustodyChain

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

    with patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.config.OCR_API_KEY", "test-key-outputs"), \
         patch("api.config.ALLOW_UNAUTHENTICATED", False), \
         patch("api.auth.OCR_API_KEY", "test-key-outputs"), \
         patch("api.auth.ALLOW_UNAUTHENTICATED", False), \
         patch("api.job_manager.config") as mock_config:
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64

        from api.main import create_app
        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        c = TestClient(app)
        # Auto-inject auth header for all requests
        c.headers.update({"X-API-Key": "test-key-outputs"})
        yield c


@pytest.fixture()
def job_with_outputs(tmp_path):
    """Create a job record with a populated output directory.

    Returns (job_id, output_dir_path).
    """
    from datetime import datetime, timezone

    from api.database import get_session_factory

    job_id = "job_aaa111bbb222"
    output_dir = tmp_path / "output" / job_id
    export = output_dir / "EXPORT"

    # Create output structure
    for subdir, ext in [
        ("PDF", ".pdf"),
        ("TEXT", ".txt"),
        ("NER", ".ner.json"),
        ("VALIDATION", ".validation.json"),
    ]:
        d = export / subdir
        d.mkdir(parents=True, exist_ok=True)
        (d / f"document{ext}").write_bytes(b"test content for " + subdir.encode())
    chain = CustodyChain(job_id, "document.pdf", str(output_dir))
    chain.append_event("file_ingested", {"source_path": "document.pdf"})
    chain.append_event("assembly_complete", {"output_dir": str(output_dir)})

    # Create job record
    session_factory = get_session_factory()
    session = session_factory()
    try:
        job = Job(
            job_id=job_id,
            status="completed",
            source_file="document.pdf",
            result_path=str(output_dir),
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(job)
        session.commit()
    finally:
        session.close()

    return job_id, output_dir


@pytest.fixture()
def job_no_outputs(tmp_path):
    """Create a job record with no result_path set.

    Returns the job_id.
    """
    from datetime import datetime, timezone

    from api.database import get_session_factory

    job_id = "job_bbb222ccc333"
    session_factory = get_session_factory()
    session = session_factory()
    try:
        job = Job(
            job_id=job_id,
            status="submitted",
            source_file="other.pdf",
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(job)
        session.commit()
    finally:
        session.close()

    return job_id


# ---------------------------------------------------------------------------
# Schema listing endpoints
# ---------------------------------------------------------------------------


class TestSchemaListEndpoint:
    def test_list_schemas_returns_200(self, client):
        resp = client.get("/api/v1/schemas")
        assert resp.status_code == 200
        data = resp.json()
        assert "schemas" in data
        assert len(data["schemas"]) > 0

    def test_list_schemas_contains_expected_types(self, client):
        resp = client.get("/api/v1/schemas")
        data = resp.json()
        types = {s["output_type"] for s in data["schemas"]}
        # Check a subset of expected types
        for expected in ["ocr_text", "searchable_pdf", "structure", "ner", "custody"]:
            assert expected in types, f"{expected} not in schema list"

    def test_list_schemas_has_versions(self, client):
        resp = client.get("/api/v1/schemas")
        data = resp.json()
        for item in data["schemas"]:
            assert "schema_version" in item
            assert item["schema_version"]  # Non-empty string


class TestGetSchemaEndpoint:
    def test_get_valid_schema(self, client):
        resp = client.get("/api/v1/schemas/ocr_text")
        assert resp.status_code == 200
        data = resp.json()
        assert "$schema" in data or "title" in data or "type" in data

    def test_get_schema_returns_json_schema(self, client):
        resp = client.get("/api/v1/schemas/structure")
        assert resp.status_code == 200
        data = resp.json()
        # Should look like a JSON Schema document
        assert "properties" in data or "type" in data

    def test_get_schema_for_each_output_type(self, client):
        """Verify all listed schemas are retrievable."""
        list_resp = client.get("/api/v1/schemas")
        assert list_resp.status_code == 200
        for item in list_resp.json()["schemas"]:
            schema_resp = client.get(f"/api/v1/schemas/{item['output_type']}")
            assert schema_resp.status_code == 200, (
                f"Failed to get schema for {item['output_type']}: {schema_resp.status_code}"
            )

    def test_get_schema_invalid_type_returns_400(self, client):
        resp = client.get("/api/v1/schemas/nonexistent_type")
        assert resp.status_code == 400
        data = resp.json()
        assert data["detail"]["error"] == "invalid_output_type"

    def test_get_schema_empty_type_redirects_to_list(self, client):
        resp = client.get("/api/v1/schemas/")
        # Trailing slash is treated as the list endpoint (redirect or 200)
        assert resp.status_code in (200, 307)


# ---------------------------------------------------------------------------
# Job output manifest endpoints
# ---------------------------------------------------------------------------


class TestListJobOutputsEndpoint:
    def test_list_outputs_for_job(self, client, job_with_outputs):
        job_id, _ = job_with_outputs
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert len(data["artifacts"]) >= 4  # PDF, TEXT, NER, VALIDATION
        types = {a["output_type"] for a in data["artifacts"]}
        assert "searchable_pdf" in types
        assert "ocr_text" in types
        assert "ner" in types
        assert "validation" in types

    def test_list_outputs_has_file_metadata(self, client, job_with_outputs):
        job_id, _ = job_with_outputs
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs")
        data = resp.json()
        for artifact in data["artifacts"]:
            assert "filename" in artifact
            assert "relative_path" in artifact
            assert "size_bytes" in artifact
            assert artifact["size_bytes"] > 0
            assert "output_type" in artifact

    def test_list_outputs_has_schema_versions(self, client, job_with_outputs):
        job_id, _ = job_with_outputs
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs")
        data = resp.json()
        assert "schema_versions" in data
        assert len(data["schema_versions"]) > 0

    def test_list_outputs_job_no_result_path(self, client, job_no_outputs):
        job_id = job_no_outputs
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["artifacts"] == []

    def test_list_outputs_invalid_job_id_returns_400(self, client):
        resp = client.get("/api/v1/jobs/invalid_id/outputs")
        assert resp.status_code == 400
        data = resp.json()
        assert data["detail"]["error"] == "invalid_job_id"

    def test_list_outputs_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/v1/jobs/job_000000000000/outputs")
        assert resp.status_code == 404
        data = resp.json()
        assert data["detail"]["error"] == "job_not_found"


# ---------------------------------------------------------------------------
# Job output download endpoint
# ---------------------------------------------------------------------------


class TestGetJobOutputEndpoint:
    def test_download_pdf_output(self, client, job_with_outputs):
        job_id, _ = job_with_outputs
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs/searchable_pdf")
        assert resp.status_code == 200
        assert b"test content for PDF" in resp.content

    def test_download_text_output(self, client, job_with_outputs):
        job_id, _ = job_with_outputs
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs/ocr_text")
        assert resp.status_code == 200
        assert b"test content for TEXT" in resp.content

    def test_download_ner_output(self, client, job_with_outputs):
        job_id, _ = job_with_outputs
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs/ner")
        assert resp.status_code == 200

    def test_download_invalid_output_type_returns_400(self, client, job_with_outputs):
        job_id, _ = job_with_outputs
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs/nonexistent_type")
        assert resp.status_code == 400
        data = resp.json()
        assert data["detail"]["error"] == "invalid_output_type"

    def test_download_missing_output_returns_404(self, client, job_with_outputs):
        job_id, _ = job_with_outputs
        # Signature output was not created in the fixture
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs/signature")
        assert resp.status_code == 404
        data = resp.json()
        assert data["detail"]["error"] == "output_not_found"

    def test_download_invalid_job_id_returns_400(self, client):
        resp = client.get("/api/v1/jobs/bad/outputs/ocr_text")
        assert resp.status_code == 400

    def test_download_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/v1/jobs/job_000000000000/outputs/ocr_text")
        assert resp.status_code == 404

    def test_download_job_no_result_path_returns_404(self, client, job_no_outputs):
        job_id = job_no_outputs
        resp = client.get(f"/api/v1/jobs/{job_id}/outputs/ocr_text")
        assert resp.status_code == 404
        data = resp.json()
        assert data["detail"]["error"] == "no_outputs"


# ---------------------------------------------------------------------------
# DocumentBundle and evidence endpoints
# ---------------------------------------------------------------------------


class TestDocumentBundleEndpoint:
    def test_get_document_bundle_returns_contract_valid_payload(self, client, job_with_outputs):
        job_id, _ = job_with_outputs

        resp = client.get(f"/api/v1/jobs/{job_id}/document-bundle")

        assert resp.status_code == 200
        data = resp.json()
        validate_document_bundle(data)
        assert data["schema_version"] == "document-bundle-v1"
        assert data["document_id"] == job_id
        assert data["source_file_name"] == "document.pdf"
        assert data["spans"][0]["text"].startswith("test content for TEXT")
        assert data["custody_chain_head"] != "n/a"
        assert resp.headers["content-disposition"].endswith(".document-bundle.json\"")

    def test_get_document_bundle_missing_text_returns_404(self, client, job_with_outputs):
        job_id, output_dir = job_with_outputs
        for path in (output_dir / "EXPORT" / "TEXT").glob("*"):
            path.unlink()

        resp = client.get(f"/api/v1/jobs/{job_id}/document-bundle")

        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "ocr_text_not_found"

    def test_get_document_bundle_job_no_outputs_returns_404(self, client, job_no_outputs):
        resp = client.get(f"/api/v1/jobs/{job_no_outputs}/document-bundle")

        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "no_outputs"


class TestEvidenceBundleEndpoint:
    def test_get_evidence_bundle_reports_artifacts_and_custody(self, client, job_with_outputs):
        job_id, _ = job_with_outputs

        resp = client.get(f"/api/v1/jobs/{job_id}/evidence-bundle")

        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_version"] == "ocr-evidence-bundle-v1"
        assert data["job_id"] == job_id
        assert data["custody"]["available"] is True
        assert data["custody"]["valid"] is True
        assert data["document_bundle_sha256"]
        assert "document_bundle_url" in data
        assert {artifact["output_type"] for artifact in data["artifacts"]} >= {
            "ocr_text",
            "custody",
        }

    def test_document_bundle_hash_is_host_independent(self, client, job_with_outputs):
        job_id, _ = job_with_outputs

        first = client.get(
            f"/api/v1/jobs/{job_id}/evidence-bundle",
            headers={"host": "first.example.test", "X-API-Key": "test-key-outputs"},
        )
        second = client.get(
            f"/api/v1/jobs/{job_id}/evidence-bundle",
            headers={"host": "second.example.test", "X-API-Key": "test-key-outputs"},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["document_bundle_url"] != second.json()["document_bundle_url"]
        assert first.json()["document_bundle_sha256"] == second.json()["document_bundle_sha256"]

    def test_get_evidence_bundle_job_no_outputs_returns_404(self, client, job_no_outputs):
        resp = client.get(f"/api/v1/jobs/{job_no_outputs}/evidence-bundle")

        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "no_outputs"


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------


class TestOutputEndpointsAuth:
    """Verify auth is enforced on all output/schema endpoints.

    When OCR_API_KEY is set and no key is provided, requests should get 401.
    """

    @pytest.fixture()
    def auth_client(self, tmp_path):
        """Client with authentication enforced."""
        reset_engine()
        db_file = str(tmp_path / "auth_test.db")
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        with patch("api.config.DB_PATH", db_file), \
             patch("api.database.DB_PATH", db_file), \
             patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.config.OCR_API_KEY", "test-secret-key"), \
             patch("api.config.ALLOW_UNAUTHENTICATED", False), \
             patch("api.auth.OCR_API_KEY", "test-secret-key"), \
             patch("api.auth.ALLOW_UNAUTHENTICATED", False), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            reset_engine()
            get_engine(db_file)

            from api.main import create_app
            app = create_app()
            app.state.limiter.enabled = False
            app.state.limiter.reset()
            yield TestClient(app)
            reset_engine()

    def test_list_schemas_requires_auth(self, auth_client):
        resp = auth_client.get("/api/v1/schemas")
        assert resp.status_code == 401

    def test_get_schema_requires_auth(self, auth_client):
        resp = auth_client.get("/api/v1/schemas/ocr_text")
        assert resp.status_code == 401

    def test_list_outputs_requires_auth(self, auth_client):
        resp = auth_client.get("/api/v1/jobs/job_aaa111bbb222/outputs")
        assert resp.status_code == 401

    def test_get_output_requires_auth(self, auth_client):
        resp = auth_client.get("/api/v1/jobs/job_aaa111bbb222/outputs/ocr_text")
        assert resp.status_code == 401

    def test_get_document_bundle_requires_auth(self, auth_client):
        resp = auth_client.get("/api/v1/jobs/job_aaa111bbb222/document-bundle")
        assert resp.status_code == 401

    def test_get_evidence_bundle_requires_auth(self, auth_client):
        resp = auth_client.get("/api/v1/jobs/job_aaa111bbb222/evidence-bundle")
        assert resp.status_code == 401

    def test_auth_with_valid_key_passes(self, auth_client):
        resp = auth_client.get(
            "/api/v1/schemas",
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code == 200
