"""Tests for Phase 7C presigned URL generation (coordinator side).

Tests the coordinator-side URL generation module that creates short-lived
presigned GET/PUT URLs for credential-free worker access.
"""

import uuid
from datetime import timedelta
from importlib import import_module
from inspect import signature
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet
from django.apps import apps
from django.db import connection
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from jobs.models import Job, PageResult, PresignedTaskCapability
from jobs.presigned import (
    create_presigned_capability,
    generate_compress_pdf_urls,
    generate_extract_entities_urls,
    generate_process_page_urls,
    get_capability_expiry,
    get_expiry,
    is_presigned_mode,
    open_presigned_capability,
    open_presigned_urls,
    seal_presigned_urls,
)
from jobs.storage import CachedS3Backend, S3Backend, StorageBackend
from jobs.tasks import (
    _request_fresh_presigned_urls,
    cleanup_expired_presigned_capabilities,
    issue_presigned_urls,
)


class MockBackend:
    """Simple mock storage backend with presigned URL support."""

    backend_name = "s3"

    def presigned_url(self, key, expires=3600):
        return f"https://s3.example.com/{key}?get&expires={expires}"

    def presigned_upload_url(self, key, expires=3600):
        return f"https://s3.example.com/{key}?put&expires={expires}"

    def presigned_delete_url(self, key, expires=3600):
        return f"https://s3.example.com/{key}?delete&expires={expires}"


def _make_mock_job(
    source_file="/data/test-doc.pdf",
    source_hash="abc123def456",  # pragma: allowlist secret
):
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
        """get_expiry() matches the coordinator's 300-second default."""
        result = get_expiry()
        self.assertEqual(result, 300)

    def test_opaque_capability_lives_longer_than_bearer_url(self):
        self.assertEqual(get_capability_expiry(), 86400)
        self.assertGreater(get_capability_expiry(), get_expiry())


class TestSealedPresignedTaskPayload(SimpleTestCase):
    """Bearer URLs must not be serialized as cleartext Celery kwargs."""

    def test_round_trip_uses_a_sealed_string(self):
        key = Fernet.generate_key().decode("ascii")
        with override_settings(S3_PRESIGNED_TASK_KEY=key):
            token = seal_presigned_urls(
                {"source_get": "https://s3.example/jobs/job-1/source.pdf"}
            )
            self.assertIsInstance(token, str)
            self.assertTrue(token.startswith("v1."))
            self.assertNotIn("s3.example", token)
            self.assertEqual(
                open_presigned_urls(token),
                {"source_get": "https://s3.example/jobs/job-1/source.pdf"},
            )

    def test_missing_key_fails_closed(self):
        with override_settings(S3_PRESIGNED_TASK_KEY=""):
            with self.assertRaisesRegex(RuntimeError, "required"):
                seal_presigned_urls({"source_get": "https://s3.example/file"})

    def test_raw_url_map_is_rejected_when_opening_task_payload(self):
        with self.assertRaisesRegex(ValueError, "sealed token"):
            open_presigned_urls({"source_get": "https://s3.example/file"})


class TestPresignedTaskCapability(TestCase):
    """Broker messages carry an id, while URLs stay job-bound in the DB."""

    def setUp(self):
        self.key = Fernet.generate_key().decode("ascii")
        self.job = Job.objects.create(source_file="source.pdf")

    @override_settings(S3_PRESIGNED_TASK_KEY="")
    def test_capability_requires_task_key(self):
        with self.assertRaisesRegex(RuntimeError, "required"):
            create_presigned_capability(
                self.job,
                "process_page",
                {"source_get": "https://s3.example/jobs/source.pdf"},
            )

    @override_settings(S3_PRESIGNED_TASK_KEY="")
    def test_capability_is_opaque_and_job_bound(self):
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            capability_id = create_presigned_capability(
                self.job,
                "process_page",
                {"source_get": "https://s3.example/jobs/source.pdf"},
            )

            self.assertIsNotNone(capability_id)
            self.assertNotIn("s3.example", str(capability_id))
            self.assertEqual(
                open_presigned_capability(
                    str(self.job.job_id), capability_id, "process_page"
                ),
                {"source_get": "https://s3.example/jobs/source.pdf"},
            )
            with self.assertRaises(ValueError):
                open_presigned_capability(
                    str(uuid.uuid4()), capability_id, "process_page"
                )
            with self.assertRaises(ValueError):
                open_presigned_capability(
                    str(self.job.job_id), capability_id, "compress_pdf"
                )

    @override_settings(S3_PRESIGNED_TASK_KEY="")
    def test_expired_capability_is_rejected(self):
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            capability_id = create_presigned_capability(
                self.job,
                "process_page",
                {"source_get": "https://s3.example/jobs/source.pdf"},
            )
        PresignedTaskCapability.objects.filter(capability_id=capability_id).update(
            expires_at=timezone.now() - timedelta(seconds=1)
        )
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            with self.assertRaises(ValueError):
                open_presigned_capability(
                    str(self.job.job_id), capability_id, "process_page"
                )

    @override_settings(S3_PRESIGNED_TASK_KEY="")
    def test_expired_jit_capability_refreshes_for_same_active_generation(self):
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            capability_id = create_presigned_capability(
                self.job,
                "process_page",
                {},
            )
        PresignedTaskCapability.objects.filter(capability_id=capability_id).update(
            expires_at=timezone.now() - timedelta(seconds=1)
        )
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            self.assertEqual(
                open_presigned_capability(
                    str(self.job.job_id), capability_id, "process_page"
                ),
                {},
            )
        self.assertGreater(
            PresignedTaskCapability.objects.get(capability_id=capability_id).expires_at,
            timezone.now(),
        )

    @override_settings(S3_PRESIGNED_TASK_KEY="")
    def test_dispatch_generation_change_revokes_capability(self):
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            capability_id = create_presigned_capability(
                self.job,
                "process_page",
                {},
            )
        Job.objects.filter(pk=self.job.pk).update(dispatch_generation=1)
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            with self.assertRaisesRegex(ValueError, "generation was revoked"):
                open_presigned_capability(
                    str(self.job.job_id), capability_id, "process_page"
                )

    def test_expired_capabilities_are_cleaned_without_touching_live_rows(self):
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            expired_id = create_presigned_capability(
                self.job,
                "process_page",
                {"source_get": "https://s3.example/jobs/expired.pdf"},
            )
            live_id = create_presigned_capability(
                self.job,
                "process_page",
                {"source_get": "https://s3.example/jobs/live.pdf"},
            )
        PresignedTaskCapability.objects.filter(capability_id=expired_id).update(
            expires_at=timezone.now() - timedelta(seconds=1)
        )
        self.job.status = Job.Status.COMPLETED
        self.job.save(update_fields=["status"])

        result = cleanup_expired_presigned_capabilities.run()

        self.assertEqual(result, {"deleted": 1})
        self.assertFalse(
            PresignedTaskCapability.objects.filter(capability_id=expired_id).exists()
        )
        self.assertTrue(
            PresignedTaskCapability.objects.filter(capability_id=live_id).exists()
        )

    def test_expired_current_generation_jit_capability_survives_cleanup(self):
        self.job.status = Job.Status.PROCESSING
        self.job.dispatch_generation = 7
        self.job.save(update_fields=["status", "dispatch_generation"])
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            capability_id = create_presigned_capability(
                self.job,
                "process_page",
                {},
            )
        PresignedTaskCapability.objects.filter(capability_id=capability_id).update(
            expires_at=timezone.now() - timedelta(seconds=1)
        )

        result = cleanup_expired_presigned_capabilities.run()

        assert result == {"deleted": 0}
        assert PresignedTaskCapability.objects.filter(
            capability_id=capability_id
        ).exists()

    def test_expired_revoked_generation_jit_capability_is_cleaned(self):
        self.job.status = Job.Status.PROCESSING
        self.job.dispatch_generation = 7
        self.job.save(update_fields=["status", "dispatch_generation"])
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            capability_id = create_presigned_capability(
                self.job,
                "process_page",
                {},
            )
        PresignedTaskCapability.objects.filter(capability_id=capability_id).update(
            dispatch_generation=6,
            expires_at=timezone.now() - timedelta(seconds=1),
        )

        result = cleanup_expired_presigned_capabilities.run()

        assert result == {"deleted": 1}
        assert not PresignedTaskCapability.objects.filter(
            capability_id=capability_id
        ).exists()

    def test_cleanup_filters_eligibility_before_bounded_slice(self):
        self.job.status = Job.Status.PROCESSING
        self.job.dispatch_generation = 7
        self.job.save(update_fields=["status", "dispatch_generation"])
        expired_at = timezone.now() - timedelta(seconds=2)
        PresignedTaskCapability.objects.bulk_create(
            [
                PresignedTaskCapability(
                    job=self.job,
                    operation="process_page",
                    dispatch_generation=7,
                    payload="",
                    expires_at=expired_at,
                )
                for _ in range(10_000)
            ],
            batch_size=500,
        )
        revoked = PresignedTaskCapability.objects.create(
            job=self.job,
            operation="process_page",
            dispatch_generation=6,
            payload="",
            expires_at=timezone.now() - timedelta(seconds=1),
        )

        result = cleanup_expired_presigned_capabilities.run()

        assert result == {"deleted": 1}
        assert not PresignedTaskCapability.objects.filter(pk=revoked.pk).exists()
        assert (
            PresignedTaskCapability.objects.filter(
                job=self.job,
                dispatch_generation=7,
            ).count()
            == 10_000
        )

    def test_migration_backfills_live_capability_generation(self):
        migration = import_module(
            "jobs.migrations.0025_presigned_capability_generation"
        )

        self.job.dispatch_generation = 11
        self.job.save(update_fields=["dispatch_generation"])
        capability = PresignedTaskCapability.objects.create(
            job=self.job,
            operation="process_page",
            dispatch_generation=0,
            payload="",
            expires_at=timezone.now() + timedelta(days=1),
        )

        migration.backfill_dispatch_generation(
            apps,
            SimpleNamespace(connection=connection),
        )

        capability.refresh_from_db()
        assert capability.dispatch_generation == 11

    def test_layoutlm_task_accepts_only_capability_reference(self):
        from jobs.tasks_layoutlm import run_layoutlm_extraction

        parameters = signature(run_layoutlm_extraction.run).parameters
        self.assertIn("presigned_capability_id", parameters)
        self.assertNotIn("presigned_urls", parameters)

    def test_worker_exchanges_opaque_capability_for_page_output_just_in_time(self):
        self.job.status = Job.Status.PROCESSING
        self.job.total_pages = 2
        self.job.source_hash = "a" * 64
        self.job.storage_backend_used = "s3"
        self.job.save(
            update_fields=[
                "status",
                "total_pages",
                "source_hash",
                "storage_backend_used",
            ]
        )
        backend = MockBackend()
        page_token = uuid.uuid4().hex
        PageResult.objects.create(
            job=self.job, document_id=self.job.source_hash[:16], page_num=2,
            status="processing", celery_task_id=page_token,
            admitted_generation=self.job.dispatch_generation,
            lease_expires_at=timezone.now() + timedelta(minutes=10),
        )
        with override_settings(
            S3_PRESIGNED_TASK_KEY=self.key,
            S3_USE_PRESIGNED_URLS=True,
        ):
            capability_id = create_presigned_capability(
                self.job,
                "process_page",
                {},
            )
            with patch("jobs.tasks._get_backend_for_job", return_value=backend):
                urls = _request_fresh_presigned_urls(
                    self.job,
                    capability_id,
                    "process_page",
                    "output",
                    page_num=2,
                    page_lease_token=page_token,
                )

        self.assertEqual(
            set(urls),
            {
                "page_pdf_put",
                "page_text_put",
                "page_pdf_delete",
                "page_text_delete",
                "candidate_evidence_put",
                "candidate_evidence_delete",
                "page_pdf_key",
                "page_text_key",
                "candidate_evidence_key",
            },
        )
        self.assertTrue(
            all(
                "/p2/l" in value
                if key.startswith("candidate_evidence")
                else "/p2/l" in value
                for key, value in urls.items()
            )
        )

    def test_page_input_bundle_seals_authoritative_source_revision(self):
        self.job.status = Job.Status.PROCESSING
        self.job.total_pages = 1
        self.job.source_hash = "a" * 64
        self.job.save(update_fields=["status", "total_pages", "source_hash"])
        backend = MagicMock(spec=CachedS3Backend)
        backend.backend_name = "s3"
        backend.source_revision_info.return_value = ("revision-bound-to-head", 4096)
        source_key = f"jobs/{self.job.job_id}/source/source.pdf"
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            capability_id = create_presigned_capability(self.job, "process_page", {})
            with (
                patch("jobs.tasks._get_backend_for_job", return_value=backend),
                patch(
                    "jobs.tasks.generate_process_page_urls",
                    return_value={"source_get": "https://s3.example/source"},
                ),
            ):
                token = issue_presigned_urls.run(
                    str(self.job.job_id), capability_id, "process_page", "input", 1
                )
            bundle = open_presigned_urls(token)
        assert bundle == {
            "source_get": "https://s3.example/source",
            "source_revision_token": "revision-bound-to-head",
            "source_size_bytes": "4096",
        }
        backend.source_revision_info.assert_called_once_with(source_key)

    def test_terminal_job_cannot_receive_fresh_bearer_urls(self):
        self.job.status = Job.Status.CANCELLED
        self.job.total_pages = 1
        self.job.save(update_fields=["status", "total_pages"])
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            capability_id = create_presigned_capability(
                self.job,
                "process_page",
                {},
            )
            with self.assertRaisesRegex(RuntimeError, "terminal job"):
                issue_presigned_urls.run(
                    str(self.job.job_id),
                    capability_id,
                    "process_page",
                    "output",
                    1,
                )

    def test_opaque_capability_size_is_constant_at_max_page_budget(self):
        self.job.total_pages = 5000
        self.job.save(update_fields=["total_pages"])
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            capability_id = create_presigned_capability(
                self.job,
                "process_text_only",
                {},
            )
        row = PresignedTaskCapability.objects.get(capability_id=capability_id)

        self.assertLess(len(row.payload.encode("utf-8")), 512)
        with override_settings(S3_PRESIGNED_TASK_KEY=self.key):
            self.assertEqual(
                open_presigned_capability(
                    str(self.job.job_id),
                    capability_id,
                    "process_text_only",
                ),
                {},
            )


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
            self.backend,
            self.job,
            self.page_num,
            self.document_id,
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
            self.backend,
            self.job,
            self.page_num,
            self.document_id,
        )

        self.assertIn("page_pdf_put", urls)
        pdf_url = urls["page_pdf_put"]
        self.assertIn("?put&", pdf_url)
        # URL should reference the page number and document ID
        self.assertIn(f"temp/{self.document_id}/{self.page_num}.pdf", pdf_url)

    def test_returns_page_text_put_url(self):
        """Result has 'page_text_put' key containing a PUT presigned URL for the page text."""
        urls = generate_process_page_urls(
            self.backend,
            self.job,
            self.page_num,
            self.document_id,
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
        self.assertIn("pdf_delete", urls)
        self.assertIn("?delete&", urls["pdf_delete"])


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
        self.assertIn("ner_delete", urls)
        self.assertIn("?delete&", urls["ner_delete"])


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
        mock_client.generate_presigned_url.return_value = (
            "https://s3.example.com/put-url"
        )

        backend = S3Backend(
            endpoint="http://s3.local",
            bucket="test-bucket",
            access_key="key",
            secret_key="secret",  # pragma: allowlist secret
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

