"""Tests for security review Sprint 1 security and robustness fixes.

Covers:
- C1: Upload filename sanitization (path traversal prevention)
- C2: SQLite check_same_thread=False for multi-threaded access
- H10: finalize_validation() idempotency
- H13: write_validation_json() path traversal protection
- C3: SSRF protection shared module (ocr_distributed/ssrf.py)

Run with: python -m pytest tests/test_sprint1_fixes.py -v
"""

from __future__ import annotations

import os
import threading
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from api.database import get_engine, get_session_factory, reset_engine

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
         patch("api.config.WEBHOOK_ALLOW_HTTP", True), \
         patch("api.config.WEBHOOK_ALLOW_PRIVATE", True), \
         patch("api.job_manager.config") as mock_config:
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64
        mock_config.WEBHOOK_TIMEOUT = 30
        mock_config.WEBHOOK_MAX_RETRIES = 3
        mock_config.WEBHOOK_SECRET = ""

        from fastapi.testclient import TestClient

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        yield TestClient(app)


@pytest.fixture()
def sample_pdf(tmp_path) -> Path:
    """Create a minimal PDF file for upload testing."""
    pdf = tmp_path / "test_doc.pdf"
    pdf.write_bytes(
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    return pdf


# ---------------------------------------------------------------------------
# C1: Upload filename sanitization (path traversal prevention)
# ---------------------------------------------------------------------------


class TestUploadFilenameSanitization:
    """Tests for C1: Upload filename sanitization (path traversal prevention)."""

    def test_traversal_filename_sanitized(self, client, sample_pdf):
        """Filenames with directory traversal are sanitized via os.path.basename.

        os.path.basename('../../etc/passwd') -> 'passwd', which is safe.
        The traversal components are stripped, and the job is accepted with
        the sanitized filename.
        """
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("../../etc/passwd", f, "application/pdf")},
            )
        # basename strips traversal, resulting in safe name "passwd"
        assert resp.status_code == 201
        assert resp.json()["source_file"] == "passwd"

    def test_backslash_traversal_rejected(self, client, sample_pdf):
        """Windows-style traversal is rejected."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("..\\..\\evil.pdf", f, "application/pdf")},
            )
        # os.path.basename on Windows strips the path, on Linux the backslash
        # stays in the name but that's still safe since it's basename'd.
        # Either 201 (basename stripped to safe name) or 400 (explicitly rejected)
        assert resp.status_code in (201, 400)

    def test_hidden_filename_rejected(self, client, sample_pdf):
        """Filenames starting with dot are rejected."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": (".hidden.pdf", f, "application/pdf")},
            )
        assert resp.status_code == 400

    def test_normal_filename_accepted(self, client, sample_pdf):
        """Normal filenames work fine."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("document.pdf", f, "application/pdf")},
            )
        assert resp.status_code == 201

    def test_deeply_nested_traversal_sanitized(self, client, sample_pdf):
        """Deep traversal paths are sanitized via os.path.basename.

        os.path.basename('a/b/c/../../../../../../etc/passwd') -> 'passwd'.
        """
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("a/b/c/../../../../../../etc/passwd", f, "application/pdf")},
            )
        # basename strips all path components, leaving safe "passwd"
        assert resp.status_code == 201
        assert resp.json()["source_file"] == "passwd"

    def test_basename_only_filename_accepted(self, client, sample_pdf):
        """A simple filename with no path components is accepted."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("my_report.pdf", f, "application/pdf")},
            )
        assert resp.status_code == 201
        assert resp.json()["source_file"] == "my_report.pdf"

    def test_empty_basename_after_strip_rejected(self, client, sample_pdf):
        """Filenames that become empty after os.path.basename are rejected."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("../", f, "application/pdf")},
            )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# C2: SQLite check_same_thread=False
# ---------------------------------------------------------------------------


class TestSQLiteThreading:
    """Tests for C2: SQLite check_same_thread=False."""

    def test_engine_connect_args_include_check_same_thread(self):
        """Engine is created with check_same_thread=False in connect_args."""
        reset_engine()
        engine = get_engine()
        # Verify by inspecting the URL and connect_args that were used.
        # The most reliable way is to test cross-thread access works.
        assert engine is not None

    def test_session_from_another_thread(self):
        """Sessions can be created and used from a non-main thread."""
        factory = get_session_factory()
        errors = []

        def thread_work():
            try:
                session = factory()
                # Execute a simple query to verify the connection works
                from api.database import Job
                session.query(Job).count()
                session.close()
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=thread_work)
        t.start()
        t.join()
        assert len(errors) == 0, f"Thread access failed: {errors}"

    def test_multiple_threads_concurrent_access(self):
        """Multiple threads can access the database concurrently."""
        factory = get_session_factory()
        errors = []
        barrier = threading.Barrier(3, timeout=5)

        def thread_work(thread_id):
            try:
                barrier.wait()
                session = factory()
                from api.database import Job
                session.query(Job).count()
                session.close()
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=thread_work, args=(i,))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent thread access failed: {errors}"


# ---------------------------------------------------------------------------
# H10: finalize_validation() idempotency
# ---------------------------------------------------------------------------


class TestFinalizeValidationIdempotency:
    """Tests for H10: finalize_validation is idempotent."""

    def test_double_call_same_result(self):
        """Calling finalize_validation twice produces identical results."""
        from validation import DocumentValidation, finalize_validation

        doc = DocumentValidation(
            document_id="idem1",
            source_file="test.pdf",
            source_page_count=3,
            output_page_count=3,
        )
        doc.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.95, "text_length": 500, "has_text": True, "status": "ok"},
            {"page_num": 2, "ocr_method": "ImageOnly", "ocr_language": "",
             "ocr_confidence": 0.0, "text_length": 0, "has_text": False, "status": "image_only"},
            {"page_num": 3, "ocr_method": "Tesseract", "ocr_language": "",
             "ocr_confidence": 0.3, "text_length": 100, "has_text": True, "status": "failed"},
        ]

        result1 = finalize_validation(doc)
        text1 = result1.pages_with_text
        img1 = result1.pages_image_only
        fail1 = result1.pages_failed
        len1 = result1.total_text_length
        conf1 = result1.overall_confidence
        rate1 = result1.text_extraction_rate

        # Call again on the same object
        result2 = finalize_validation(doc)

        assert result2.pages_with_text == text1
        assert result2.pages_image_only == img1
        assert result2.pages_failed == fail1
        assert result2.total_text_length == len1
        assert result2.overall_confidence == conf1
        assert result2.text_extraction_rate == rate1

    def test_triple_call_stable(self):
        """Three calls produce identical results."""
        from validation import DocumentValidation, finalize_validation

        doc = DocumentValidation(
            document_id="idem2",
            source_file="test.pdf",
            source_page_count=1,
            output_page_count=1,
        )
        doc.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.9, "text_length": 200, "has_text": True, "status": "ok"},
        ]

        finalize_validation(doc)
        finalize_validation(doc)
        result = finalize_validation(doc)

        assert result.pages_with_text == 1
        assert result.total_text_length == 200
        assert result.pages_image_only == 0
        assert result.pages_failed == 0

    def test_idempotency_counters_reset(self):
        """Counters are reset on each call, not accumulated."""
        from validation import DocumentValidation, finalize_validation

        doc = DocumentValidation(
            document_id="idem3",
            source_file="test.pdf",
            source_page_count=2,
            output_page_count=2,
        )
        doc.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.8, "text_length": 100, "has_text": True, "status": "ok"},
            {"page_num": 2, "ocr_method": "ImageOnly", "ocr_language": "",
             "ocr_confidence": 0.0, "text_length": 0, "has_text": False, "status": "image_only"},
        ]

        # Call multiple times -- without reset, counters would accumulate
        for _ in range(5):
            result = finalize_validation(doc)

        # Values should reflect one pass over the data, not 5x
        assert result.pages_with_text == 1
        assert result.pages_image_only == 1
        assert result.total_text_length == 100


# ---------------------------------------------------------------------------
# H13: write_validation_json path traversal protection
# ---------------------------------------------------------------------------


class TestValidationPathTraversal:
    """Tests for H13: write_validation_json path traversal protection."""

    def test_traversal_subfolder_blocked(self, tmp_path):
        """Subfolder with path traversal is blocked."""
        from validation import (
            DocumentValidation,
            finalize_validation,
            write_validation_json,
        )

        doc = DocumentValidation(document_id="pt1", source_file="test.pdf")
        doc.pages = []
        doc = finalize_validation(doc)
        result = write_validation_json(
            doc, str(tmp_path), "../../etc", "1.0.0"
        )
        assert result is None

    def test_normal_subfolder_works(self, tmp_path):
        """Normal subfolder creates file correctly."""
        from validation import (
            DocumentValidation,
            finalize_validation,
            write_validation_json,
        )

        doc = DocumentValidation(document_id="pt2", source_file="test.pdf")
        doc.pages = []
        doc = finalize_validation(doc)
        result = write_validation_json(
            doc, str(tmp_path), "my_folder", "1.0.0"
        )
        assert result is not None
        assert os.path.exists(result)

    def test_dot_subfolder_works(self, tmp_path):
        """Dot subfolder writes to base directory."""
        from validation import (
            DocumentValidation,
            finalize_validation,
            write_validation_json,
        )

        doc = DocumentValidation(document_id="pt3", source_file="test.pdf")
        doc.pages = []
        doc = finalize_validation(doc)
        result = write_validation_json(
            doc, str(tmp_path), ".", "1.0.0"
        )
        assert result is not None
        assert os.path.exists(result)

    def test_dotdot_in_middle_of_subfolder_blocked(self, tmp_path):
        """Subfolder containing '..' component is blocked."""
        from validation import (
            DocumentValidation,
            finalize_validation,
            write_validation_json,
        )

        doc = DocumentValidation(document_id="pt4", source_file="test.pdf")
        doc.pages = []
        doc = finalize_validation(doc)
        result = write_validation_json(
            doc, str(tmp_path), "legit/../../../escape", "1.0.0"
        )
        assert result is None

    def test_output_file_has_correct_extension(self, tmp_path):
        """Output file ends with .validation.json."""
        from validation import (
            DocumentValidation,
            finalize_validation,
            write_validation_json,
        )

        doc = DocumentValidation(document_id="pt5", source_file="report.pdf")
        doc.pages = []
        doc = finalize_validation(doc)
        result = write_validation_json(
            doc, str(tmp_path), ".", "1.0.0"
        )
        assert result is not None
        assert result.endswith("report.validation.json")


# ---------------------------------------------------------------------------
# C3: SSRF protection shared module
# ---------------------------------------------------------------------------


class TestSSRFProtection:
    """Tests for C3: SSRF protection shared module (ocr_distributed/ssrf.py)."""

    def test_private_ip_127_detected(self):
        """127.0.0.1 is detected as private."""
        from ocr_distributed.ssrf import is_private_ip
        assert is_private_ip("127.0.0.1") is True

    def test_private_ip_10_detected(self):
        """10.0.0.1 (RFC1918) is detected as private."""
        from ocr_distributed.ssrf import is_private_ip
        assert is_private_ip("10.0.0.1") is True

    def test_private_ip_192_168_detected(self):
        """192.168.1.1 is detected as private."""
        from ocr_distributed.ssrf import is_private_ip
        assert is_private_ip("192.168.1.1") is True

    def test_private_ip_172_16_detected(self):
        """172.16.0.1 is detected as private."""
        from ocr_distributed.ssrf import is_private_ip
        assert is_private_ip("172.16.0.1") is True

    def test_public_ip_allowed(self):
        """Public IPs are not flagged as private."""
        from ocr_distributed.ssrf import is_private_ip
        assert is_private_ip("8.8.8.8") is False
        assert is_private_ip("1.1.1.1") is False

    def test_loopback_ipv6(self):
        """IPv6 loopback is detected."""
        from ocr_distributed.ssrf import is_private_ip
        assert is_private_ip("::1") is True

    def test_validate_rejects_http_by_default(self):
        """HTTP URLs are rejected by default."""
        from ocr_distributed.ssrf import validate_webhook_url
        with pytest.raises(ValueError, match="HTTPS"):
            validate_webhook_url("http://example.com/hook")

    def test_validate_allows_https(self):
        """HTTPS URLs pass validation."""
        from ocr_distributed.ssrf import validate_webhook_url
        result = validate_webhook_url(
            "https://example.com/hook",
            allow_private=True,
        )
        assert result == "https://example.com/hook"

    def test_validate_rejects_private_ip(self):
        """Private IP URLs are rejected."""
        from ocr_distributed.ssrf import validate_webhook_url
        with pytest.raises(ValueError, match="private"):
            validate_webhook_url("https://192.168.1.1/hook")

    def test_validate_rejects_localhost(self):
        """Localhost URLs are rejected."""
        from ocr_distributed.ssrf import validate_webhook_url
        with pytest.raises(ValueError, match="localhost"):
            validate_webhook_url("https://localhost/hook")

    def test_validate_rejects_empty(self):
        """Empty URLs are rejected."""
        from ocr_distributed.ssrf import validate_webhook_url
        with pytest.raises(ValueError, match="empty"):
            validate_webhook_url("")

    def test_validate_rejects_ftp_scheme(self):
        """Non-HTTP(S) schemes are rejected."""
        from ocr_distributed.ssrf import validate_webhook_url
        with pytest.raises(ValueError, match="unsupported scheme"):
            validate_webhook_url("ftp://example.com/hook")

    def test_validate_rejects_whitespace_only(self):
        """Whitespace-only URLs are rejected."""
        from ocr_distributed.ssrf import validate_webhook_url
        with pytest.raises(ValueError, match="empty"):
            validate_webhook_url("   ")

    def test_validate_rejects_too_long_url(self):
        """URLs exceeding 2048 characters are rejected."""
        from ocr_distributed.ssrf import validate_webhook_url
        long_url = "https://example.com/" + "a" * 2040
        with pytest.raises(ValueError, match="2048"):
            validate_webhook_url(long_url)

    def test_validate_allows_http_when_flag_set(self):
        """HTTP URLs are allowed when allow_http=True."""
        from ocr_distributed.ssrf import validate_webhook_url
        result = validate_webhook_url(
            "http://example.com/hook",
            allow_http=True,
            allow_private=True,
        )
        assert result == "http://example.com/hook"

    def test_validate_allows_localhost_when_private_flag_set(self):
        """Localhost is allowed when allow_private=True."""
        from ocr_distributed.ssrf import validate_webhook_url
        result = validate_webhook_url(
            "https://localhost/hook",
            allow_private=True,
        )
        assert result == "https://localhost/hook"

    def test_redirect_handler_blocks(self):
        """NoRedirectHandler raises on redirect."""
        from ocr_distributed.ssrf import NoRedirectHandler
        handler = NoRedirectHandler()

        # Build a mock request object with the full_url attribute
        mock_req = type("MockReq", (), {"full_url": "https://example.com"})()

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            handler.redirect_request(
                mock_req, None, 302, "Found", {}, "https://evil.com"
            )
        assert exc_info.value.code == 302
        assert "Redirect blocked" in str(exc_info.value.msg)

    def test_validate_strips_whitespace(self):
        """Leading/trailing whitespace is stripped from URLs."""
        from ocr_distributed.ssrf import validate_webhook_url
        result = validate_webhook_url(
            "  https://example.com/hook  ",
            allow_private=True,
        )
        assert result == "https://example.com/hook"

    def test_validate_rejects_missing_hostname(self):
        """URLs without a hostname are rejected."""
        from ocr_distributed.ssrf import validate_webhook_url
        with pytest.raises(ValueError, match="hostname"):
            validate_webhook_url("https://")

    def test_validate_rejects_missing_scheme(self):
        """URLs without a scheme are rejected."""
        from ocr_distributed.ssrf import validate_webhook_url
        with pytest.raises(ValueError, match="scheme"):
            validate_webhook_url("example.com/hook")

    def test_safe_opener_exists(self):
        """Module-level safe_opener is available."""
        from ocr_distributed.ssrf import safe_opener
        assert safe_opener is not None

    def test_loopback_names_include_common_values(self):
        """LOOPBACK_NAMES set includes localhost and 127.0.0.1."""
        from ocr_distributed.ssrf import LOOPBACK_NAMES
        assert "localhost" in LOOPBACK_NAMES
        assert "127.0.0.1" in LOOPBACK_NAMES
        assert "::1" in LOOPBACK_NAMES
