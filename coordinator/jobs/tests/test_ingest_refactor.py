"""Tests for ingest_document refactored helpers.

Validates that:
- Extracted helper functions work correctly in isolation.
- Temporary work directories are always cleaned up on error.
- The _IngestError flow correctly marks jobs as FAILED.

Run with: cd coordinator && python -m pytest jobs/tests/test_ingest_refactor.py -v
"""

import os
import shutil
import sys
import tempfile
import types
import uuid
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from jobs.models import Job
from jobs.tasks import (
    _classify_and_count_pages,
    _cleanup_temp_dir,
    _detect_language,
    _dispatch_processing,
    _IngestError,
    _load_source_classifiers,
    _log_worker_availability,
    _setup_ingest_work_dir,
    _upload_source_to_s3,
    _validate_and_resolve_source,
    ingest_document,
)


class TestIngestError(TestCase):
    """Tests for the _IngestError exception class."""

    def test_is_exception(self):
        err = _IngestError("test message")
        assert isinstance(err, Exception)

    def test_str_representation(self):
        err = _IngestError("test message")
        assert str(err) == "test message"


class TestSetupIngestWorkDir(TestCase):
    """Tests for _setup_ingest_work_dir helper."""

    def test_nfs_mode_creates_dirs_and_saves_path(self):
        job = Job.objects.create(
            source_file="/some/path/test.pdf",
            status=Job.Status.SUBMITTED,
        )
        with tempfile.TemporaryDirectory() as nfs_root:
            with override_settings(NFS_ROOT=nfs_root):
                job_path, source_path = _setup_ingest_work_dir(
                    job, str(job.job_id), "nfs"
                )

        assert os.path.basename(source_path) == "test.pdf"
        assert "source" in source_path
        job.refresh_from_db()
        assert job.nfs_job_path == job_path

    def test_s3_mode_creates_temp_dir(self):
        job = Job.objects.create(
            source_file="/some/path/test.pdf",
            status=Job.Status.SUBMITTED,
        )
        job_path, source_path = _setup_ingest_work_dir(
            job, str(job.job_id), "s3"
        )
        try:
            assert os.path.isdir(job_path)
            assert os.path.basename(source_path) == "test.pdf"
            # Verify subdirectories were created
            assert os.path.isdir(os.path.join(job_path, "source"))
            assert os.path.isdir(os.path.join(job_path, "temp"))
        finally:
            shutil.rmtree(job_path, ignore_errors=True)


class TestValidateAndResolveSource(TestCase):
    """Tests for _validate_and_resolve_source helper."""

    def test_existing_source_path_passes(self):
        """If source file already exists at source_path, no error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "test.pdf")
            with open(source_path, "wb") as f:
                f.write(b"%PDF-1.4 test")

            job = Job.objects.create(
                source_file=source_path,
                status=Job.Status.SUBMITTED,
            )
            # Should not raise
            _validate_and_resolve_source(job, source_path)

    def test_copies_absolute_source_within_allowed_root(self):
        """Copies file from job.source_file to source_path when within NFS_ROOT."""
        with tempfile.TemporaryDirectory() as nfs_root:
            # Create source file within NFS_ROOT
            upload_dir = os.path.join(nfs_root, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            original = os.path.join(upload_dir, "test.pdf")
            with open(original, "wb") as f:
                f.write(b"%PDF-1.4 test content")

            # Target source_path doesn't exist yet
            dest_dir = os.path.join(nfs_root, "jobs", "dest", "source")
            os.makedirs(dest_dir, exist_ok=True)
            source_path = os.path.join(dest_dir, "test.pdf")

            with override_settings(NFS_ROOT=nfs_root):
                job = Job.objects.create(
                    source_file=original,
                    status=Job.Status.SUBMITTED,
                )
                _validate_and_resolve_source(job, source_path)

            assert os.path.isfile(source_path)

    def test_rejects_source_outside_allowed_root(self):
        """Raises _IngestError when source file is outside NFS_ROOT."""
        with tempfile.TemporaryDirectory() as allowed_root, \
             tempfile.TemporaryDirectory() as other_root:
            outside_file = os.path.join(other_root, "outside.pdf")
            with open(outside_file, "wb") as f:
                f.write(b"%PDF-1.4 outside content")

            dest_path = os.path.join(allowed_root, "jobs", "dest", "source", "outside.pdf")
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            with override_settings(NFS_ROOT=allowed_root):
                job = Job.objects.create(
                    source_file=outside_file,
                    status=Job.Status.SUBMITTED,
                )
                with self.assertRaises(_IngestError) as ctx:
                    _validate_and_resolve_source(job, dest_path)
                assert "outside allowed directory" in str(ctx.exception)

    def test_missing_source_raises(self):
        """Raises _IngestError when source file does not exist anywhere."""
        job = Job.objects.create(
            source_file="/nonexistent/path/missing.pdf",
            status=Job.Status.SUBMITTED,
        )
        with self.assertRaises(_IngestError) as ctx:
            _validate_and_resolve_source(job, "/tmp/also_missing.pdf")
        assert "Source file not found" in str(ctx.exception)


class TestUploadSourceToS3(TestCase):
    """Tests for _upload_source_to_s3 helper."""

    def test_successful_upload(self):
        backend = MagicMock()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            source_path = f.name
        try:
            _upload_source_to_s3(backend, "job-123", source_path)
            backend.upload_file.assert_called_once()
        finally:
            os.unlink(source_path)

    def test_upload_failure_raises_ingest_error(self):
        backend = MagicMock()
        backend.upload_file.side_effect = RuntimeError("S3 timeout")
        with self.assertRaises(_IngestError) as ctx:
            _upload_source_to_s3(backend, "job-123", "/tmp/test.pdf")
        assert "S3 upload failed" in str(ctx.exception)


class TestLoadSourceClassifiers(TestCase):
    """Tests for _load_source_classifiers helper."""

    def test_fallback_classifiers_work(self):
        """When ocr_distributed is unavailable, fallback classifiers work."""
        # Remove ocr_distributed from sys.modules temporarily
        saved = {}
        for key in list(sys.modules.keys()):
            if key.startswith("ocr_distributed"):
                saved[key] = sys.modules.pop(key)
        try:
            with patch.dict(sys.modules, {"ocr_distributed": None, "ocr_distributed.ocr_utils": None}):
                classify, count = _load_source_classifiers()

            result_type, warning = classify("/some/file.pdf")
            assert result_type == "pdf"
            assert warning is None

            result_type, warning = classify("/some/file.png")
            assert result_type == "image"

            result_type, warning = classify("/some/file.xyz")
            assert result_type is None
            assert "Unsupported" in warning
        finally:
            sys.modules.update(saved)


class TestClassifyAndCountPages(TestCase):
    """Tests for _classify_and_count_pages helper."""

    def test_unsupported_file_raises_ingest_error(self):
        job = Job.objects.create(
            source_file="/test.xyz",
            status=Job.Status.SUBMITTED,
        )
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"unknown content")
            path = f.name
        try:
            # Force use of fallback classifiers
            saved = {}
            for key in list(sys.modules.keys()):
                if key.startswith("ocr_distributed"):
                    saved[key] = sys.modules.pop(key)
            try:
                with patch.dict(sys.modules, {
                    "ocr_distributed": None,
                    "ocr_distributed.ocr_utils": None,
                }):
                    with self.assertRaises(_IngestError) as ctx:
                        _classify_and_count_pages(job, path)
                    assert "Unsupported file" in str(ctx.exception)
            finally:
                sys.modules.update(saved)
        finally:
            os.unlink(path)

    def test_classification_exception_raises_ingest_error(self):
        """If classification itself throws, wraps as _IngestError."""
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.SUBMITTED,
        )
        with patch("jobs.tasks._load_source_classifiers") as mock_load:
            mock_load.side_effect = RuntimeError("import failed")
            with self.assertRaises(_IngestError) as ctx:
                _classify_and_count_pages(job, "/tmp/fake.pdf")
            assert "File classification failed" in str(ctx.exception)


class TestDetectLanguage(TestCase):
    """Tests for _detect_language helper."""

    def test_defaults_to_en_on_failure(self):
        """Language detection failure defaults to 'en' without raising."""
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.SUBMITTED,
        )
        # Mock the import to fail
        with patch.dict(sys.modules, {"ocr_distributed.language": None}):
            _detect_language(job, "/nonexistent.pdf", "pdf")

        job.refresh_from_db()
        assert job.detected_language == "en"


class TestCleanupTempDir(TestCase):
    """Tests for _cleanup_temp_dir helper."""

    def test_removes_dir_for_s3_mode(self):
        tmpdir = tempfile.mkdtemp(prefix="test_cleanup_")
        assert os.path.isdir(tmpdir)
        _cleanup_temp_dir(tmpdir, "s3")
        assert not os.path.isdir(tmpdir)

    def test_preserves_dir_for_nfs_mode(self):
        tmpdir = tempfile.mkdtemp(prefix="test_cleanup_")
        try:
            _cleanup_temp_dir(tmpdir, "nfs")
            assert os.path.isdir(tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_handles_none_path(self):
        """No error when job_path is None."""
        _cleanup_temp_dir(None, "s3")

    def test_handles_missing_dir(self):
        """No error when directory does not exist."""
        _cleanup_temp_dir("/nonexistent/path/12345", "s3")


class TestWorkDirCleanupOnError(TestCase):
    """Integration tests verifying temp dirs are cleaned on all error paths.

    This is the key bug fix: previously some error paths in ingest_document
    leaked temporary directories in S3 mode.
    """

    def _make_fake_modules(self, classify_fn=None, page_count_fn=None):
        """Build fake ocr_distributed modules for patching.

        Note: classify_fn is wrapped to accept **kwargs because the canonical
        ``_load_source_classifiers`` now passes ``include_coordinator_types``
        via ``functools.partial``.
        """
        fake_pkg = types.ModuleType("ocr_distributed")
        fake_ocr_utils = types.ModuleType("ocr_distributed.ocr_utils")
        fake_language = types.ModuleType("ocr_distributed.language")

        _raw_classify = classify_fn or (lambda _p: ("pdf", None))
        fake_ocr_utils.classify_source_file = lambda _p, **_kw: _raw_classify(_p)
        fake_ocr_utils.get_source_page_count = page_count_fn or (lambda _p, _t: 1)

        class _FakeDetector:
            def __init__(self, _):
                pass
            def detect_from_pdf(self, _):
                return "en"

        fake_language.LanguageDetector = _FakeDetector
        fake_pkg.ocr_utils = fake_ocr_utils
        fake_pkg.language = fake_language
        return {
            "ocr_distributed": fake_pkg,
            "ocr_distributed.ocr_utils": fake_ocr_utils,
            "ocr_distributed.language": fake_language,
        }

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_temp_dir_cleaned_on_classification_failure(
        self, mock_get_storage, mock_dispatch
    ):
        """When file classification fails, S3 temp dir must be cleaned up."""
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage.return_value = backend

        with tempfile.TemporaryDirectory() as nfs_root:
            source_dir = os.path.join(nfs_root, "uploads")
            os.makedirs(source_dir, exist_ok=True)
            test_file = os.path.join(source_dir, "test.xyz")
            with open(test_file, "wb") as f:
                f.write(b"unknown content")

            # Use classifiers that return unsupported
            modules = self._make_fake_modules(
                classify_fn=lambda _p: (None, "Unsupported extension: .xyz")
            )

            with override_settings(NFS_ROOT=nfs_root):
                job = Job.objects.create(
                    source_file=test_file,
                    status=Job.Status.SUBMITTED,
                )

                # Track what tempdir is created
                created_tempdirs = []
                orig_mkdtemp = tempfile.mkdtemp

                def _tracking_mkdtemp(**kwargs):
                    d = orig_mkdtemp(**kwargs)
                    created_tempdirs.append(d)
                    return d

                with patch.object(tempfile, "mkdtemp", side_effect=_tracking_mkdtemp):
                    with patch.dict(sys.modules, modules):
                        result = ingest_document.run(str(job.job_id))

                assert result["status"] == "error"
                assert "Unsupported" in result["message"]
                # Verify temp dir was cleaned up
                for d in created_tempdirs:
                    assert not os.path.isdir(d), f"Temp dir leaked: {d}"

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_temp_dir_cleaned_on_page_count_failure(
        self, mock_get_storage, mock_dispatch
    ):
        """When page counting fails, S3 temp dir must be cleaned up."""
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage.return_value = backend

        def _fail_page_count(_path, _type):
            raise RuntimeError("PDF corrupted")

        with tempfile.TemporaryDirectory() as nfs_root:
            source_dir = os.path.join(nfs_root, "uploads")
            os.makedirs(source_dir, exist_ok=True)
            test_file = os.path.join(source_dir, "test.pdf")
            with open(test_file, "wb") as f:
                f.write(b"%PDF-1.4 test content")

            modules = self._make_fake_modules(page_count_fn=_fail_page_count)

            with override_settings(NFS_ROOT=nfs_root):
                job = Job.objects.create(
                    source_file=test_file,
                    status=Job.Status.SUBMITTED,
                )

                created_tempdirs = []
                orig_mkdtemp = tempfile.mkdtemp

                def _tracking_mkdtemp(**kwargs):
                    d = orig_mkdtemp(**kwargs)
                    created_tempdirs.append(d)
                    return d

                with patch.object(tempfile, "mkdtemp", side_effect=_tracking_mkdtemp):
                    with patch.dict(sys.modules, modules):
                        result = ingest_document.run(str(job.job_id))

                assert result["status"] == "error"
                assert "Page count failed" in result["message"]
                for d in created_tempdirs:
                    assert not os.path.isdir(d), f"Temp dir leaked: {d}"

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_temp_dir_cleaned_on_s3_upload_failure(
        self, mock_get_storage, mock_dispatch
    ):
        """When S3 upload fails, temp dir must be cleaned up."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.upload_file.side_effect = RuntimeError("S3 timeout")
        mock_get_storage.return_value = backend

        with tempfile.TemporaryDirectory() as nfs_root:
            source_dir = os.path.join(nfs_root, "uploads")
            os.makedirs(source_dir, exist_ok=True)
            test_file = os.path.join(source_dir, "test.pdf")
            with open(test_file, "wb") as f:
                f.write(b"%PDF-1.4 test content")

            with override_settings(NFS_ROOT=nfs_root):
                job = Job.objects.create(
                    source_file=test_file,
                    status=Job.Status.SUBMITTED,
                )

                created_tempdirs = []
                orig_mkdtemp = tempfile.mkdtemp

                def _tracking_mkdtemp(**kwargs):
                    d = orig_mkdtemp(**kwargs)
                    created_tempdirs.append(d)
                    return d

                with patch.object(tempfile, "mkdtemp", side_effect=_tracking_mkdtemp):
                    result = ingest_document.run(str(job.job_id))

                assert result["status"] == "error"
                assert "S3 upload failed" in result["message"]
                for d in created_tempdirs:
                    assert not os.path.isdir(d), f"Temp dir leaked: {d}"

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_temp_dir_cleaned_on_source_not_found(
        self, mock_get_storage, mock_dispatch
    ):
        """When source file is missing, S3 temp dir must be cleaned up."""
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage.return_value = backend

        with tempfile.TemporaryDirectory() as nfs_root:
            with override_settings(NFS_ROOT=nfs_root):
                job = Job.objects.create(
                    source_file="/nonexistent/missing.pdf",
                    status=Job.Status.SUBMITTED,
                )

                created_tempdirs = []
                orig_mkdtemp = tempfile.mkdtemp

                def _tracking_mkdtemp(**kwargs):
                    d = orig_mkdtemp(**kwargs)
                    created_tempdirs.append(d)
                    return d

                with patch.object(tempfile, "mkdtemp", side_effect=_tracking_mkdtemp):
                    result = ingest_document.run(str(job.job_id))

                assert result["status"] == "error"
                assert "Source file not found" in result["message"]
                for d in created_tempdirs:
                    assert not os.path.isdir(d), f"Temp dir leaked: {d}"

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_temp_dir_cleaned_on_path_containment_failure(
        self, mock_get_storage, mock_dispatch
    ):
        """When path containment check fails, S3 temp dir must be cleaned up."""
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage.return_value = backend

        with tempfile.TemporaryDirectory() as allowed_root, \
             tempfile.TemporaryDirectory() as other_root:
            outside_file = os.path.join(other_root, "outside.pdf")
            with open(outside_file, "wb") as f:
                f.write(b"%PDF-1.4 outside content")

            with override_settings(NFS_ROOT=allowed_root):
                job = Job.objects.create(
                    source_file=outside_file,
                    status=Job.Status.SUBMITTED,
                )

                created_tempdirs = []
                orig_mkdtemp = tempfile.mkdtemp

                def _tracking_mkdtemp(**kwargs):
                    d = orig_mkdtemp(**kwargs)
                    created_tempdirs.append(d)
                    return d

                with patch.object(tempfile, "mkdtemp", side_effect=_tracking_mkdtemp):
                    result = ingest_document.run(str(job.job_id))

                assert result["status"] == "error"
                assert "outside allowed directory" in result["message"]
                for d in created_tempdirs:
                    assert not os.path.isdir(d), f"Temp dir leaked: {d}"


class TestDispatchProcessing(TestCase):
    """Tests for _dispatch_processing helper."""

    @patch("jobs.tasks.process_document.apply_async")
    def test_small_doc_dispatches_single_worker(self, mock_dispatch):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.INGESTING,
        )
        result = _dispatch_processing(job, str(job.job_id), total_pages=5)
        assert result["mode"] == "single"
        mock_dispatch.assert_called_once()
        job.refresh_from_db()
        assert job.status == Job.Status.PROCESSING

    @patch("jobs.tasks.extract_pages.apply_async")
    def test_large_doc_dispatches_fanout(self, mock_dispatch):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.INGESTING,
        )
        result = _dispatch_processing(job, str(job.job_id), total_pages=50)
        assert result["mode"] == "fanout"
        mock_dispatch.assert_called_once()

    @patch("jobs.tasks.process_text_only.apply_async")
    def test_skip_ocr_dispatches_text_only(self, mock_dispatch):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.INGESTING,
            settings_json={"skip_ocr": True},
        )
        result = _dispatch_processing(job, str(job.job_id), total_pages=5)
        assert result["mode"] == "skip_ocr"
        mock_dispatch.assert_called_once()


class TestLogWorkerAvailability(TestCase):
    """Tests for _log_worker_availability helper (advisory, no side effects)."""

    def test_no_workers_logs_warning(self):
        """No crash when no workers exist."""
        # Should not raise
        _log_worker_availability(str(uuid.uuid4()))
