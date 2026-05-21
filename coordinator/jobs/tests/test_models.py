"""Tests for Django coordinator models.

Tests Job, Worker, PageResult, and CustodyEvent model creation,
field defaults, relationships, constraints, and properties.

Run with: cd coordinator && python -m pytest jobs/tests/test_models.py -v
"""

import uuid
from datetime import timedelta

import pytest
from django.db import IntegrityError
from django.test import TestCase
from django.utils import timezone

from jobs.models import CustodyEvent, Job, PageResult, Worker


class TestJobModel(TestCase):
    """Tests for the Job model."""

    def test_create_job_with_defaults(self):
        job = Job.objects.create(source_file="/path/to/document.pdf")
        assert job.status == Job.Status.SUBMITTED
        assert job.priority == Job.Priority.NORMAL
        assert job.total_pages == 0
        assert job.pages_completed == 0
        assert job.pages_failed == 0
        assert job.detected_language == "en"
        assert job.source_hash == ""
        assert job.source_type == ""
        assert job.error_message == ""
        assert job.settings_json == {}
        assert job.result_summary == {}
        assert job.webhook_url == ""
        assert job.webhook_secret == ""
        assert job.webhook_status == ""
        assert job.nfs_job_path == ""
        assert job.celery_task_id == ""
        assert job.assigned_worker == ""

    def test_job_id_is_uuid(self):
        job = Job.objects.create(source_file="/test.pdf")
        assert isinstance(job.job_id, uuid.UUID)

    def test_job_id_is_primary_key(self):
        job = Job.objects.create(source_file="/test.pdf")
        fetched = Job.objects.get(pk=job.job_id)
        assert fetched.source_file == "/test.pdf"

    def test_status_choices(self):
        valid_statuses = [c[0] for c in Job.Status.choices]
        assert "submitted" in valid_statuses
        assert "ingesting" in valid_statuses
        assert "processing" in valid_statuses
        assert "assembling" in valid_statuses
        assert "completed" in valid_statuses
        assert "failed" in valid_statuses
        assert "cancelled" in valid_statuses

    def test_priority_choices(self):
        valid_priorities = [c[0] for c in Job.Priority.choices]
        assert "urgent" in valid_priorities
        assert "normal" in valid_priorities
        assert "low" in valid_priorities

    def test_processing_time_seconds_both_set(self):
        now = timezone.now()
        job = Job.objects.create(
            source_file="/test.pdf",
            started_at=now - timedelta(seconds=120),
            completed_at=now,
        )
        assert job.processing_time_seconds == pytest.approx(120.0, abs=1)

    def test_processing_time_seconds_not_started(self):
        job = Job.objects.create(source_file="/test.pdf")
        assert job.processing_time_seconds is None

    def test_processing_time_seconds_not_completed(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            started_at=timezone.now(),
        )
        assert job.processing_time_seconds is None

    def test_str_representation(self):
        job = Job.objects.create(source_file="/test.pdf")
        result = str(job)
        assert str(job.job_id) in result
        assert "submitted" in result

    def test_ordering_by_created_at_desc(self):
        j1 = Job.objects.create(source_file="/first.pdf")
        j2 = Job.objects.create(source_file="/second.pdf")
        jobs = list(Job.objects.all())
        # Most recent first (descending)
        assert jobs[0].pk == j2.pk
        assert jobs[1].pk == j1.pk

    def test_status_can_be_set(self):
        job = Job.objects.create(source_file="/test.pdf")
        job.status = Job.Status.PROCESSING
        job.save(update_fields=["status"])
        job.refresh_from_db()
        assert job.status == Job.Status.PROCESSING


class TestWorkerModel(TestCase):
    """Tests for the Worker model."""

    def test_create_worker_with_defaults(self):
        worker = Worker.objects.create(hostname="worker-01")
        assert worker.status == Worker.Status.OFFLINE
        assert worker.queues == []
        assert worker.capabilities == []
        assert worker.concurrency == 4
        assert worker.gpu_available is False
        assert worker.gpu_model == ""
        assert worker.gpu_vram_mb == 0
        assert worker.cpu_cores == 0
        assert worker.ram_mb == 0
        assert worker.tasks_completed == 0
        assert worker.tasks_failed == 0
        assert worker.current_task_id == ""
        assert worker.pipeline_version == ""

    def test_hostname_is_primary_key(self):
        Worker.objects.create(hostname="worker-pk")
        fetched = Worker.objects.get(pk="worker-pk")
        assert fetched.hostname == "worker-pk"

    def test_status_choices(self):
        valid = [c[0] for c in Worker.Status.choices]
        assert "online" in valid
        assert "busy" in valid
        assert "offline" in valid
        assert "draining" in valid

    def test_str_representation(self):
        worker = Worker.objects.create(hostname="worker-test", status=Worker.Status.ONLINE)
        result = str(worker)
        assert "worker-test" in result
        assert "online" in result.lower()

    def test_ordering_by_hostname(self):
        Worker.objects.create(hostname="worker-b")
        Worker.objects.create(hostname="worker-a")
        workers = list(Worker.objects.all())
        assert workers[0].hostname == "worker-a"
        assert workers[1].hostname == "worker-b"


class TestPageResultModel(TestCase):
    """Tests for the PageResult model."""

    def test_create_page_result(self):
        job = Job.objects.create(source_file="/test.pdf", total_pages=5)
        pr = PageResult.objects.create(
            job=job,
            document_id="abc123",
            page_num=1,
            ocr_method="PaddleOCR",
            ocr_confidence=0.95,
            text_length=500,
            status="ok",
        )
        assert pr.job_id == job.job_id
        assert pr.page_num == 1
        assert pr.status == "ok"

    def test_fk_relationship(self):
        job = Job.objects.create(source_file="/test.pdf")
        PageResult.objects.create(job=job, document_id="d1", page_num=1)
        PageResult.objects.create(job=job, document_id="d1", page_num=2)
        assert job.page_results.count() == 2

    def test_unique_together_constraint(self):
        job = Job.objects.create(source_file="/test.pdf")
        PageResult.objects.create(job=job, document_id="d1", page_num=1)
        with pytest.raises(IntegrityError):
            PageResult.objects.create(job=job, document_id="d1", page_num=1)

    def test_cascade_delete(self):
        job = Job.objects.create(source_file="/test.pdf")
        PageResult.objects.create(job=job, document_id="d1", page_num=1)
        job.delete()
        assert PageResult.objects.count() == 0

    def test_defaults(self):
        job = Job.objects.create(source_file="/test.pdf")
        pr = PageResult.objects.create(job=job, document_id="d1", page_num=1)
        assert pr.ocr_method == ""
        assert pr.ocr_language == ""
        assert pr.ocr_confidence == 0.0
        assert pr.text_length == 0
        assert pr.status == "pending"
        assert pr.worker_hostname == ""
        assert pr.processing_time_ms == 0
        assert pr.celery_task_id == ""
        assert pr.temp_pdf_path == ""

    def test_str_representation(self):
        job = Job.objects.create(source_file="/test.pdf")
        pr = PageResult.objects.create(job=job, document_id="d1", page_num=3, status="ok")
        result = str(pr)
        assert "3" in result
        assert "ok" in result

    def test_ordering(self):
        job = Job.objects.create(source_file="/test.pdf")
        PageResult.objects.create(job=job, document_id="d1", page_num=3)
        PageResult.objects.create(job=job, document_id="d1", page_num=1)
        PageResult.objects.create(job=job, document_id="d1", page_num=2)
        pages = list(PageResult.objects.all())
        assert [p.page_num for p in pages] == [1, 2, 3]


class TestCustodyEventModel(TestCase):
    """Tests for the CustodyEvent model."""

    def test_create_custody_event(self):
        job = Job.objects.create(source_file="/test.pdf")
        event = CustodyEvent.objects.create(
            document_id="abc123def456",
            job=job,
            event_type="file_ingested",
            data={"source_hash": "deadbeef"},
        )
        assert event.document_id == "abc123def456"
        assert event.event_type == "file_ingested"
        assert event.data == {"source_hash": "deadbeef"}

    def test_defaults(self):
        job = Job.objects.create(source_file="/test.pdf")
        event = CustodyEvent.objects.create(
            document_id="d1", job=job, event_type="test",
        )
        assert event.worker_hostname == ""
        assert event.data == {}
        assert event.prev_hash == ""
        assert event.event_hash == ""
        assert event.chain_finalized is False

    def test_set_null_on_delete(self):
        """CustodyEvent.job is SET_NULL — events survive job deletion."""
        job = Job.objects.create(source_file="/test.pdf")
        CustodyEvent.objects.create(document_id="d1", job=job, event_type="test")
        job.delete()
        assert CustodyEvent.objects.count() == 1
        event = CustodyEvent.objects.first()
        assert event.job is None

    def test_str_representation(self):
        job = Job.objects.create(source_file="/test.pdf")
        event = CustodyEvent.objects.create(
            document_id="d1", job=job, event_type="file_ingested",
        )
        result = str(event)
        assert "file_ingested" in result
        assert "d1" in result

    def test_ordering(self):
        job = Job.objects.create(source_file="/test.pdf")
        now = timezone.now()
        CustodyEvent.objects.create(
            document_id="d1", job=job, event_type="second",
            timestamp=now + timedelta(seconds=10),
        )
        CustodyEvent.objects.create(
            document_id="d1", job=job, event_type="first",
            timestamp=now,
        )
        events = list(CustodyEvent.objects.all())
        assert events[0].event_type == "first"
        assert events[1].event_type == "second"

    def test_fk_related_name(self):
        job = Job.objects.create(source_file="/test.pdf")
        CustodyEvent.objects.create(document_id="d1", job=job, event_type="a")
        CustodyEvent.objects.create(document_id="d1", job=job, event_type="b")
        assert job.custody_events.count() == 2
