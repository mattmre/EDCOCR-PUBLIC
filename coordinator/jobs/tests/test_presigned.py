"""Tests for Phase 7C presigned URL generation (coordinator side).

Tests the coordinator-side URL generation module that creates short-lived
presigned GET/PUT URLs for credential-free worker access.
"""

import uuid
from unittest.mock import MagicMock

from django.test import SimpleTestCase, TestCase, override_settings

from jobs.presigned import (
    generate_compress_pdf_urls,
    generate_extract_entities_urls,
    generate_process_page_urls,
    get_expiry,
    is_presigned_mode,
)
from jobs.storage import CachedS3Backend, S3Backend, StorageBackend


class MockBackend:
    """Simple mock storage backend with presigned URL support."""

    def presigned_url(self, key, expires=3600):
        return f"https://s3.example.com/{key}?get&expires={expires}"

    def presigned_upload_url(self, key, expires=3600):
        return f"https://s3.example.com/{key}?put&expires={expires}"


def _make_mock_job(source_file="/data/test-doc.pdf", source_hash="abc123def456"):
    """Create a mock Job object with required fields."""
    job = MagicMock()
    job.job_id = uuid.uuid4()
    job.source_file = source_file
    job.source_hash = source_hash
    return job


class TestIsPresignedMode(TestCase):
    """Tests for is_presigned_mode() settings detection."""

    def test_disabled_by_default(self):
        """Without S3_USE_PRESIGNED_URLS setting, is_presigned_mode() returns False."""
        # settings_test.py does not define S3_USE_PRESIGNED_URLS, so getattr falls back to False
        result = is_presigned_mode()
        self.assertFalse(result)

    @override_settings(S3_USE_PRESIGNED_URLS=True)
    def test_enabled_via_setting(self):
        """With S3_USE_PRESIGNED_URLS=True, is_presigned_mode() returns True."""
        result = is_presigned_mode()
        self.assertTrue(result)

    def test_get_expiry_default(self):
        """get_expiry() returns 3600 by default when S3_PRESIGNED_URL_EXPIRY is not set."""
        result = get_expiry()
        self.assertEqual(result, 3600)


class TestGenerateProcessPageUrls(SimpleTestCase):
    """Tests for generate_process_page_urls() URL generation."""

    def setUp(self):
        self.backend = MockBackend()
        self.job = _make_mock_job()
        self.page_num = 5
        self.document_id = "doc-hash-abc123"

    def test_returns_source_get_url(self):
        """Result has 'source_get' key containing a GET presigned URL for the source document."""
        urls = generate_process_page_urls(
            self.backend, self.job, self.page_num, self.document_id,
        )

        self.assertIn("source_get", urls)
        source_url = urls["source_get"]
        self.assertIn("?get&", source_url)
        # URL should reference the source filename
        self.assertIn("test-doc.pdf", source_url)
        # URL should be under the job's storage key
        self.assertIn(f"jobs/{self.job.job_id}/source/", source_url)

    def test_returns_page_pdf_put_url(self):
        """Result has 'page_pdf_put' key containing a PUT presigned URL for the page PDF."""
        urls = generate_process_page_urls(
            self.backend, self.job, self.page_num, self.document_id,
        )

        self.assertIn("page_pdf_put", urls)
        pdf_url = urls["page_pdf_put"]
        self.assertIn("?put&", pdf_url)
        # URL should reference the page number and document ID
        self.assertIn(f"temp/{self.document_id}/{self.page_num}.pdf", pdf_url)

    def test_returns_page_text_put_url(self):
        """Result has 'page_text_put' key containing a PUT presigned URL for the page text."""
        urls = generate_process_page_urls(
            self.backend, self.job, self.page_num, self.document_id,
        )

        self.assertIn("page_text_put", urls)
        text_url = urls["page_text_put"]
        self.assertIn("?put&", text_url)
        # URL should reference the page number and document ID
        self.assertIn(f"temp/{self.document_id}/{self.page_num}.txt", text_url)


class TestGenerateCompressPdfUrls(SimpleTestCase):
    """Tests for generate_compress_pdf_urls() URL generation."""

    def setUp(self):
        self.backend = MockBackend()
        self.job = _make_mock_job()

    def test_returns_pdf_get_url(self):
        """Result has 'pdf_get' key containing a GET presigned URL for the assembled PDF."""
        urls = generate_compress_pdf_urls(self.backend, self.job)

        self.assertIn("pdf_get", urls)
        pdf_get_url = urls["pdf_get"]
        self.assertIn("?get&", pdf_get_url)
        # URL should reference the output PDF path with base name (no extension)
        self.assertIn("output/EXPORT/PDF/test-doc.pdf", pdf_get_url)
        self.assertIn(f"jobs/{self.job.job_id}/", pdf_get_url)

    def test_returns_pdf_put_url(self):
        """Result has 'pdf_put' key containing a PUT presigned URL for the compressed PDF."""
        urls = generate_compress_pdf_urls(self.backend, self.job)

        self.assertIn("pdf_put", urls)
        pdf_put_url = urls["pdf_put"]
        self.assertIn("?put&", pdf_put_url)
        # PUT URL should target the same path as the GET URL
        self.assertIn("output/EXPORT/PDF/test-doc.pdf", pdf_put_url)


class TestGenerateExtractEntitiesUrls(SimpleTestCase):
    """Tests for generate_extract_entities_urls() URL generation."""

    def setUp(self):
        self.backend = MockBackend()
        self.job = _make_mock_job()

    def test_returns_text_get_url(self):
        """Result has 'text_get' key containing a GET presigned URL for the extracted text."""
        urls = generate_extract_entities_urls(self.backend, self.job)

        self.assertIn("text_get", urls)
        text_get_url = urls["text_get"]
        self.assertIn("?get&", text_get_url)
        # URL should reference the TEXT export path
        self.assertIn("output/EXPORT/TEXT/test-doc.txt", text_get_url)
        self.assertIn(f"jobs/{self.job.job_id}/", text_get_url)

    def test_returns_ner_put_url(self):
        """Result has 'ner_put' key containing a PUT presigned URL for NER output."""
        urls = generate_extract_entities_urls(self.backend, self.job)

        self.assertIn("ner_put", urls)
        ner_put_url = urls["ner_put"]
        self.assertIn("?put&", ner_put_url)
        # URL should reference the NER export path
        self.assertIn("output/EXPORT/NER/test-doc.ner.json", ner_put_url)


class TestPresignedUploadUrl(SimpleTestCase):
    """Tests for presigned_upload_url() on storage backend classes."""

    def test_base_raises_not_implemented(self):
        """StorageBackend.presigned_upload_url() raises NotImplementedError."""
        # StorageBackend is abstract, but presigned_upload_url has a concrete default
        # that raises NotImplementedError. We create a minimal concrete subclass to test it.
        class MinimalBackend(StorageBackend):
            @property
            def backend_name(self):
                return "minimal"

            def upload_file(self, local_path, key):
                pass

            def download_file(self, key, local_path):
                pass

            def delete(self, key):
                pass

            def exists(self, key):
                return False

            def list_objects(self, prefix):
                return []

        backend = MinimalBackend()
        with self.assertRaises(NotImplementedError):
            backend.presigned_upload_url("some/key", expires=300)

    def test_s3_backend_generates_put_url(self):
        """S3Backend.presigned_upload_url() calls generate_presigned_url with 'put_object'."""
        mock_client = MagicMock()
        mock_client.generate_presigned_url.return_value = "https://s3.example.com/put-url"

        backend = S3Backend(
            endpoint="http://s3.local",
            bucket="test-bucket",
            access_key="key",
            secret_key="secret",
            region="us-east-1",
            client=mock_client,
        )

        url = backend.presigned_upload_url("jobs/1/output.pdf", expires=600)

        self.assertEqual(url, "https://s3.example.com/put-url")
        mock_client.generate_presigned_url.assert_called_once_with(
            "put_object",
            Params={"Bucket": "test-bucket", "Key": "jobs/1/output.pdf"},
            ExpiresIn=600,
        )

    def test_cached_s3_delegates(self):
        """CachedS3Backend.presigned_upload_url() delegates to inner S3Backend."""
        mock_inner = MagicMock(spec=S3Backend)
        mock_inner.backend_name = "s3"
        mock_inner.presigned_upload_url.return_value = "https://s3.example.com/cached-put"

        cached = CachedS3Backend(inner=mock_inner, cache_dir="/tmp/test-cache-presigned")

        url = cached.presigned_upload_url("jobs/1/output.pdf", expires=900)

        self.assertEqual(url, "https://s3.example.com/cached-put")
        mock_inner.presigned_upload_url.assert_called_once_with("jobs/1/output.pdf", 900)
