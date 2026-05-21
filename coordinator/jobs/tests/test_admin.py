"""Tests for Django admin configuration.

Tests the custom admin actions, display methods, and badge rendering
for JobAdmin, WorkerAdmin, PageResultAdmin, and CustodyEventAdmin.

Run with: cd coordinator && python -m pytest jobs/tests/test_admin.py -v
"""


from unittest.mock import patch

from django.contrib.admin.sites import AdminSite
from django.test import RequestFactory, TestCase

from jobs.admin import (
    CustodyEventAdmin,
    CustodyEventInline,
    JobAdmin,
    PageResultInline,
    WorkerAdmin,
)
from jobs.models import CustodyEvent, Job, Worker


class TestJobAdmin(TestCase):
    """Tests for JobAdmin custom methods and actions."""

    def setUp(self):
        self.site = AdminSite()
        self.admin = JobAdmin(Job, self.site)
        self.factory = RequestFactory()

    def test_job_id_short(self):
        job = Job.objects.create(source_file="/test.pdf")
        result = self.admin.job_id_short(job)
        assert len(result) == 8
        assert result == str(job.job_id)[:8]

    def test_source_file_name(self):
        job = Job.objects.create(source_file="/path/to/document.pdf")
        result = self.admin.source_file_name(job)
        assert result == "document.pdf"

    def test_status_badge_submitted(self):
        job = Job.objects.create(source_file="/test.pdf", status="submitted")
        result = self.admin.status_badge(job)
        assert "Submitted" in result
        assert "#6c757d" in result

    def test_status_badge_processing(self):
        job = Job.objects.create(source_file="/test.pdf", status="processing")
        result = self.admin.status_badge(job)
        assert "Processing" in result
        assert "#007bff" in result

    def test_status_badge_completed(self):
        job = Job.objects.create(source_file="/test.pdf", status="completed")
        result = self.admin.status_badge(job)
        assert "Completed" in result
        assert "#28a745" in result

    def test_status_badge_failed(self):
        job = Job.objects.create(source_file="/test.pdf", status="failed")
        result = self.admin.status_badge(job)
        assert "Failed" in result
        assert "#dc3545" in result

    def test_progress_bar_zero_pages(self):
        job = Job.objects.create(source_file="/test.pdf", total_pages=0)
        result = self.admin.progress_bar(job)
        assert result == "-"

    def test_progress_bar_partial(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            total_pages=10,
            pages_completed=3,
        )
        result = self.admin.progress_bar(job)
        assert "3/10" in result
        assert "30%" in result

    def test_progress_bar_complete(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            total_pages=5,
            pages_completed=5,
        )
        result = self.admin.progress_bar(job)
        assert "5/5" in result
        assert "100%" in result

    def _make_admin_request(self):
        """Create a request with messages middleware support for admin actions."""
        from django.contrib.messages.storage.fallback import FallbackStorage
        request = self.factory.post("/admin/")
        request.user = type("User", (), {"has_perm": lambda self, x: True})()
        # Attach messages storage so message_user() works
        setattr(request, "session", "session")
        setattr(request, "_messages", FallbackStorage(request))
        return request

    def test_retry_failed_jobs_action(self):
        j1 = Job.objects.create(source_file="/a.pdf", status="failed", error_message="err")
        j2 = Job.objects.create(source_file="/b.pdf", status="completed")
        j3 = Job.objects.create(source_file="/c.pdf", status="failed", error_message="err2")
        request = self._make_admin_request()
        queryset = Job.objects.all()
        self.admin.retry_failed_jobs(request, queryset)
        j1.refresh_from_db()
        j2.refresh_from_db()
        j3.refresh_from_db()
        assert j1.status == "submitted"
        assert j1.error_message == ""
        assert j2.status == "completed"  # Not changed
        assert j3.status == "submitted"

    def test_cancel_running_jobs_action(self):
        j1 = Job.objects.create(source_file="/a.pdf", status="processing")
        j2 = Job.objects.create(source_file="/b.pdf", status="completed")
        j3 = Job.objects.create(source_file="/c.pdf", status="ingesting")
        j4 = Job.objects.create(source_file="/d.pdf", status="submitted")
        request = self._make_admin_request()
        queryset = Job.objects.all()
        self.admin.cancel_running_jobs(request, queryset)
        j1.refresh_from_db()
        j2.refresh_from_db()
        j3.refresh_from_db()
        j4.refresh_from_db()
        assert j1.status == "cancelled"
        assert j2.status == "completed"  # Not changed
        assert j3.status == "cancelled"
        assert j4.status == "cancelled"


class TestWorkerAdmin(TestCase):
    """Tests for WorkerAdmin custom methods."""

    def setUp(self):
        self.site = AdminSite()
        self.admin = WorkerAdmin(Worker, self.site)

    def test_status_badge_online(self):
        worker = Worker.objects.create(hostname="w1", status="online")
        result = self.admin.status_badge(worker)
        assert "Online" in result
        assert "#28a745" in result

    def test_status_badge_offline(self):
        worker = Worker.objects.create(hostname="w2", status="offline")
        result = self.admin.status_badge(worker)
        assert "Offline" in result
        assert "#dc3545" in result

    def test_capabilities_display_with_caps(self):
        worker = Worker.objects.create(
            hostname="w3",
            capabilities=["ocr", "compress", "ner"],
        )
        result = self.admin.capabilities_display(worker)
        assert "ocr" in result
        assert "compress" in result
        assert "ner" in result

    def test_capabilities_display_empty(self):
        worker = Worker.objects.create(hostname="w4", capabilities=[])
        result = self.admin.capabilities_display(worker)
        assert result == "-"

    def test_gpu_info_cpu_only(self):
        worker = Worker.objects.create(hostname="w5", gpu_available=False)
        result = self.admin.gpu_info(worker)
        assert result == "CPU only"

    def test_gpu_info_with_gpu(self):
        worker = Worker.objects.create(
            hostname="w6",
            gpu_available=True,
            gpu_model="NVIDIA RTX 4090",
            gpu_vram_mb=24576,
        )
        result = self.admin.gpu_info(worker)
        assert "NVIDIA RTX 4090" in result
        assert "24576" in result


class TestCustodyEventAdmin(TestCase):
    """Tests for CustodyEventAdmin custom methods."""

    def setUp(self):
        self.site = AdminSite()
        self.admin = CustodyEventAdmin(CustodyEvent, self.site)

    def test_event_hash_short_with_hash(self):
        job = Job.objects.create(source_file="/test.pdf")
        event = CustodyEvent.objects.create(
            document_id="d1", job=job, event_type="test",
            event_hash="abcdef1234567890abcdef1234567890",
        )
        result = self.admin.event_hash_short(event)
        assert result == "abcdef123456..."

    def test_event_hash_short_empty(self):
        job = Job.objects.create(source_file="/test.pdf")
        event = CustodyEvent.objects.create(
            document_id="d1", job=job, event_type="test",
            event_hash="",
        )
        result = self.admin.event_hash_short(event)
        assert result == "-"


class TestJobAdminInlines(TestCase):
    """Tests for PageResultInline and CustodyEventInline on JobAdmin."""

    def setUp(self):
        self.site = AdminSite()
        self.admin = JobAdmin(Job, self.site)

    def test_page_result_inline_registered(self):
        inline_classes = [type(i) for i in self.admin.get_inline_instances(None)]
        assert PageResultInline in inline_classes

    def test_custody_event_inline_registered(self):
        inline_classes = [type(i) for i in self.admin.get_inline_instances(None)]
        assert CustodyEventInline in inline_classes

    def test_inlines_are_read_only(self):
        for inline_cls in [PageResultInline, CustodyEventInline]:
            inline = inline_cls(Job, self.site)
            assert inline.extra == 0
            assert inline.can_delete is False
            assert inline.has_add_permission(None) is False


class TestWorkerAdminActions(TestCase):
    """Tests for M3 WorkerAdmin actions (drain, mark_offline, ping)."""

    def setUp(self):
        self.admin = WorkerAdmin(Worker, AdminSite())
        self.factory = RequestFactory()

    def _make_admin_request(self):
        from django.contrib.messages.storage.fallback import FallbackStorage
        request = self.factory.post("/admin/")
        request.user = type("User", (), {"has_perm": lambda self, x: True})()
        setattr(request, "session", "session")
        setattr(request, "_messages", FallbackStorage(request))
        return request

    def test_drain_workers_sets_draining_status(self):
        Worker.objects.create(
            hostname="w1", status=Worker.Status.ONLINE, queues=["ocr_gpu"]
        )
        Worker.objects.create(
            hostname="w2", status=Worker.Status.BUSY, queues=["cpu_general"]
        )
        Worker.objects.create(hostname="w3", status=Worker.Status.OFFLINE)

        request = self._make_admin_request()
        qs = Worker.objects.all()
        with patch("coordinator.celery.app"):
            self.admin.drain_workers(request, qs)

        w1 = Worker.objects.get(hostname="w1")
        w2 = Worker.objects.get(hostname="w2")
        w3 = Worker.objects.get(hostname="w3")
        assert w1.status == Worker.Status.DRAINING
        assert w2.status == Worker.Status.DRAINING
        assert w3.status == Worker.Status.OFFLINE  # unchanged

    def test_drain_workers_uses_worker_queues(self):
        """Drain should cancel consumers for each worker's registered queues."""
        Worker.objects.create(
            hostname="w1",
            status=Worker.Status.ONLINE,
            queues=["ocr_gpu", "cpu_general"],
        )

        request = self._make_admin_request()
        qs = Worker.objects.all()
        with patch("coordinator.celery.app") as mock_app:
            self.admin.drain_workers(request, qs)
            # cancel_consumer called once per queue from worker.queues
            assert mock_app.control.cancel_consumer.call_count == 2
            mock_app.control.cancel_consumer.assert_any_call(
                "ocr_gpu", destination=["w1"]
            )
            mock_app.control.cancel_consumer.assert_any_call(
                "cpu_general", destination=["w1"]
            )

    def test_drain_workers_no_queues_still_drains(self):
        """Worker with empty queues list should still be set to draining."""
        Worker.objects.create(
            hostname="w1", status=Worker.Status.ONLINE, queues=[]
        )

        request = self._make_admin_request()
        qs = Worker.objects.all()
        with patch("coordinator.celery.app") as mock_app:
            self.admin.drain_workers(request, qs)
            mock_app.control.cancel_consumer.assert_not_called()

        assert Worker.objects.get(hostname="w1").status == Worker.Status.DRAINING

    def test_drain_workers_handles_celery_error(self):
        Worker.objects.create(hostname="w1", status=Worker.Status.ONLINE)

        request = self._make_admin_request()
        qs = Worker.objects.all()
        with patch("coordinator.celery.app") as mock_app:
            mock_app.control.cancel_consumer.side_effect = Exception("connection refused")
            # Should not raise
            self.admin.drain_workers(request, qs)

    def test_mark_workers_offline(self):
        Worker.objects.create(hostname="w1", status=Worker.Status.ONLINE, current_task_id="t1")
        Worker.objects.create(hostname="w2", status=Worker.Status.DRAINING, current_task_id="t2")
        Worker.objects.create(hostname="w3", status=Worker.Status.OFFLINE)

        request = self._make_admin_request()
        qs = Worker.objects.all()
        self.admin.mark_workers_offline(request, qs)

        assert Worker.objects.get(hostname="w1").status == Worker.Status.OFFLINE
        assert Worker.objects.get(hostname="w1").current_task_id == ""
        assert Worker.objects.get(hostname="w2").status == Worker.Status.OFFLINE
        assert Worker.objects.get(hostname="w3").status == Worker.Status.OFFLINE

    def test_ping_workers_reports_responsive(self):
        Worker.objects.create(hostname="w1", status=Worker.Status.ONLINE)
        Worker.objects.create(hostname="w2", status=Worker.Status.ONLINE)

        request = self._make_admin_request()
        qs = Worker.objects.all()
        with patch("coordinator.celery.app") as mock_app:
            mock_app.control.ping.return_value = [{"w1": {"ok": "pong"}}]
            self.admin.ping_workers(request, qs)

        # Check message was set (1 responsive, 1 unresponsive)
        messages = list(request._messages)
        assert len(messages) == 1
        assert "1 responsive" in str(messages[0])
        assert "1 unresponsive" in str(messages[0])

    def test_ping_workers_updates_heartbeat(self):
        """Responsive workers should get their last_heartbeat updated."""
        Worker.objects.create(
            hostname="w1", status=Worker.Status.ONLINE, last_heartbeat=None
        )
        Worker.objects.create(
            hostname="w2", status=Worker.Status.ONLINE, last_heartbeat=None
        )

        request = self._make_admin_request()
        qs = Worker.objects.all()
        with patch("coordinator.celery.app") as mock_app:
            mock_app.control.ping.return_value = [{"w1": {"ok": "pong"}}]
            self.admin.ping_workers(request, qs)

        w1 = Worker.objects.get(hostname="w1")
        w2 = Worker.objects.get(hostname="w2")
        assert w1.last_heartbeat is not None  # Updated
        assert w2.last_heartbeat is None  # Not responsive, not updated

    def test_ping_workers_handles_celery_error(self):
        Worker.objects.create(hostname="w1", status=Worker.Status.ONLINE)

        request = self._make_admin_request()
        qs = Worker.objects.all()
        with patch("coordinator.celery.app") as mock_app:
            mock_app.control.ping.side_effect = Exception("connection refused")
            self.admin.ping_workers(request, qs)

        messages = list(request._messages)
        assert len(messages) == 1
        assert "failed" in str(messages[0]).lower()
