"""Retired whole-document deliveries preserve page-fenced recovery semantics."""

import uuid
from unittest.mock import patch

from django.test import TestCase

from jobs.models import Job, PageResult, TaskDispatchOutbox
from jobs.tasks import extract_pages, process_document


class TestLegacyRootDelivery(TestCase):
    @patch("jobs.tasks._process_single_page")
    @patch("jobs.tasks.extract_pages.run")
    def test_redelivery_redirects_without_unleased_processing(self, extract, legacy_ocr):
        job = Job.objects.create(
            source_file="/test.pdf", source_type="pdf",
            status=Job.Status.PROCESSING, dispatch_generation=1,
        )
        extract.return_value = {"status": "fanout_dispatched"}
        for _ in range(2):
            assert process_document.run(str(job.job_id), dispatch_generation=1)["status"] == "fanout_dispatched"
        assert extract.call_count == 2
        legacy_ocr.assert_not_called()

    @patch("jobs.tasks.extract_pages.run")
    def test_stale_delivery_and_terminal_job_do_not_redrive(self, extract):
        job = Job.objects.create(
            source_file="/test.pdf", source_type="pdf",
            status=Job.Status.PROCESSING, dispatch_generation=2,
        )
        assert process_document.run(str(job.job_id), dispatch_generation=1)["status"] != "fanout_dispatched"
        job.status = Job.Status.CANCELLED
        job.save(update_fields=["status"])
        assert process_document.run(str(job.job_id), dispatch_generation=2)["status"] == "cancelled"
        extract.assert_not_called()

    @patch("jobs.tasks.extract_pages.run")
    def test_malformed_source_type_fails_closed(self, extract):
        job = Job.objects.create(
            source_file="/test.bin", source_type="",
            status=Job.Status.PROCESSING, dispatch_generation=1,
        )
        result = process_document.run(str(job.job_id), dispatch_generation=1)
        job.refresh_from_db()
        assert result["status"] == "error"
        assert job.status == Job.Status.FAILED
        extract.assert_not_called()

    def test_missing_job_is_error(self):
        assert process_document.run(str(uuid.uuid4()))["status"] == "error"


class TestPageAdmissionParentFence(TestCase):
    @patch("jobs.tasks._fair_fanout_enabled")
    def test_generation_change_before_parent_lock_creates_no_pages(self, fair_fanout):
        job = Job.objects.create(
            source_file="/test.pdf", source_type="pdf", source_hash="a" * 64,
            status=Job.Status.PROCESSING, dispatch_generation=1, total_pages=2,
        )

        def supersede():
            Job.objects.filter(pk=job.pk).update(dispatch_generation=2)
            return False

        fair_fanout.side_effect = supersede
        result = extract_pages.run(str(job.job_id), dispatch_generation=1)
        assert result["status"] == "stale"
        assert not PageResult.objects.filter(job=job).exists()
        assert not TaskDispatchOutbox.objects.filter(job=job).exists()

    @patch("jobs.tasks._fair_fanout_enabled")
    def test_source_change_before_parent_lock_creates_no_pages(self, fair_fanout):
        job = Job.objects.create(
            source_file="/test.pdf", source_type="pdf", source_hash="a" * 64,
            status=Job.Status.PROCESSING, dispatch_generation=1, total_pages=2,
        )

        def replace_source():
            Job.objects.filter(pk=job.pk).update(source_hash="b" * 64)
            return False

        fair_fanout.side_effect = replace_source
        result = extract_pages.run(str(job.job_id), dispatch_generation=1)
        assert result["status"] == "stale"
        assert not PageResult.objects.filter(job=job).exists()
        assert not TaskDispatchOutbox.objects.filter(job=job).exists()
