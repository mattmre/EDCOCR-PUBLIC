"""Tests for api/job_manager.py -- direct unit tests for JobManager methods."""

from __future__ import annotations

from subprocess import TimeoutExpired
from unittest.mock import MagicMock, patch

import pytest

from api.database import Job, UsageRecord, get_session_factory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager(tmp_path):
    """Create a JobManager with isolated config."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with patch("api.job_manager.config") as mock_config:
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 0.1
        mock_config.JOB_PROCESSING_TIMEOUT_MINUTES = 30
        mock_config.MAX_CONCURRENT_JOBS = 64
        mock_config.WEBHOOK_TIMEOUT = 30
        mock_config.WEBHOOK_MAX_RETRIES = 0
        mock_config.WEBHOOK_SECRET = ""

        from api.job_manager import JobManager

        yield JobManager(get_session_factory())


def _insert_job(job_id, status="submitted", **kwargs):
    """Insert a job directly into the database."""
    factory = get_session_factory()
    session = factory()
    job = Job(
        job_id=job_id,
        status=status,
        source_file=kwargs.pop("source_file", "test.pdf"),
        priority=kwargs.pop("priority", "normal"),
    )
    for key, value in kwargs.items():
        setattr(job, key, value)
    session.add(job)
    session.commit()
    session.close()
    return job_id


# ---------------------------------------------------------------------------
# _run_pipeline Tests
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Tests for JobManager._run_pipeline()."""

    def test_pipeline_success_sets_completed(self, manager, tmp_path):
        """Successful pipeline sets status to completed."""
        job_id = _insert_job("job_aaaaaaaaaaaa")
        source_dir = str(tmp_path / "source")
        output_dir = str(tmp_path / "output")

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, source_dir, output_dir, {})

        job = manager.get_job(job_id)
        assert job.status == "completed"
        assert job.pid is None
        assert job.completed_at is not None
        assert job.processing_time is not None

    def test_pipeline_failure_sets_failed(self, manager, tmp_path):
        """Non-zero exit code sets status to failed."""
        job_id = _insert_job("job_bbbbbbbbbbbb")
        source_dir = str(tmp_path / "source")
        output_dir = str(tmp_path / "output")

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, source_dir, output_dir, {})

        job = manager.get_job(job_id)
        assert job.status == "failed"
        assert "exited with code 1" in job.error_message.lower()

    def test_pipeline_exception_sets_failed(self, manager, tmp_path):
        """Exception during subprocess launch sets status to failed."""
        job_id = _insert_job("job_cccccccccccc")
        source_dir = str(tmp_path / "source")
        output_dir = str(tmp_path / "output")

        with patch("api.job_manager.subprocess.Popen", side_effect=OSError("spawn failed")), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, source_dir, output_dir, {})

        job = manager.get_job(job_id)
        assert job.status == "failed"
        assert job.error_message is not None

    def test_pipeline_nonexistent_job_returns_early(self, manager, tmp_path):
        """Pipeline returns immediately for nonexistent job."""
        with patch("api.job_manager.subprocess.Popen") as mock_popen:
            manager._run_pipeline("job_999999999999", str(tmp_path), str(tmp_path), {})
        mock_popen.assert_not_called()

    def test_pipeline_stores_pid(self, manager, tmp_path):
        """Pipeline stores subprocess PID on the job row."""
        job_id = _insert_job("job_dddddddddddd")

        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        pid_captured = []

        def capture_pid(*args, **kwargs):
            """Side effect for proc.wait that reads PID from DB."""
            factory = get_session_factory()
            session = factory()
            job = session.get(Job, job_id)
            if job and job.pid:
                pid_captured.append(job.pid)
            session.close()
            return 0

        mock_proc.wait.side_effect = capture_pid

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        assert 42 in pid_captured

    def test_pipeline_docintel_adds_flags(self, manager, tmp_path):
        """DocIntel settings add --enable-docintel and --docintel-mode flags."""
        job_id = _insert_job("job_eeeeeeeeeeee")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(
                job_id, str(tmp_path), str(tmp_path),
                {"enable_docintel": True, "docintel_mode": "tables_only"},
            )

        cmd = mock_popen.call_args[0][0]
        assert "--enable-docintel" in cmd
        assert "--docintel-mode" in cmd
        assert "tables_only" in cmd

    def test_pipeline_docintel_full_mode_omits_mode_flag(self, manager, tmp_path):
        """DocIntel with mode=full omits --docintel-mode flag (it is the default)."""
        job_id = _insert_job("job_ffffffaaaaaa")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(
                job_id, str(tmp_path), str(tmp_path),
                {"enable_docintel": True, "docintel_mode": "full"},
            )

        cmd = mock_popen.call_args[0][0]
        assert "--enable-docintel" in cmd
        assert "--docintel-mode" not in cmd

    def test_pipeline_sets_started_at(self, manager, tmp_path):
        """Pipeline sets started_at timestamp before launching subprocess."""
        job_id = _insert_job("job_startedattt")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        job = manager.get_job(job_id)
        assert job.started_at is not None

    def test_pipeline_uses_default_processing_timeout(self, manager, tmp_path):
        """Pipeline waits using the configured default timeout when unset per-job."""
        job_id = _insert_job("job_defaulttout")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        mock_proc.wait.assert_called_once_with(timeout=30 * 60)

    def test_pipeline_timeout_kills_process_and_marks_failed(self, manager, tmp_path):
        """A timed-out pipeline is killed and marked failed."""
        job_id = _insert_job("job_timeoutkill")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = -9
        mock_proc.wait.side_effect = [
            TimeoutExpired(cmd="echo", timeout=120),
            -9,
        ]

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(
                job_id,
                str(tmp_path),
                str(tmp_path),
                {"processing_timeout_minutes": 2},
            )

        job = manager.get_job(job_id)
        assert job.status == "failed"
        assert "2 minutes" in (job.error_message or "")
        mock_proc.kill.assert_called_once()

    def test_pipeline_clears_pid_after_completion(self, manager, tmp_path):
        """PID is cleared to None after pipeline finishes (success or failure)."""
        job_id = _insert_job("job_pidcleartest")

        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        job = manager.get_job(job_id)
        assert job.pid is None

    def test_pipeline_fires_webhook_on_success(self, manager, tmp_path):
        """Pipeline calls _fire_webhook after successful completion."""
        job_id = _insert_job("job_webhooksuc1")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook") as mock_webhook, \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        mock_webhook.assert_called_once()

    def test_pipeline_fires_webhook_on_failure(self, manager, tmp_path):
        """Pipeline calls _fire_webhook after failed completion."""
        job_id = _insert_job("job_webhookfail")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 2
        mock_proc.wait.return_value = 2

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook") as mock_webhook, \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        mock_webhook.assert_called_once()

    def test_pipeline_fires_webhook_on_exception(self, manager, tmp_path):
        """Pipeline calls _fire_webhook even when an exception occurs."""
        job_id = _insert_job("job_webhookexc1")

        with patch("api.job_manager.subprocess.Popen", side_effect=RuntimeError("boom")), \
             patch.object(manager, "_fire_webhook") as mock_webhook, \
             patch.object(manager, "_publish_job_event"):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        mock_webhook.assert_called_once()

    def test_pipeline_publishes_processing_and_completed_events(self, manager, tmp_path):
        """Pipeline emits durable events at processing start and successful completion."""
        job_id = _insert_job("job_eventsuccess")

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event") as mock_publish:
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        event_types = [call.args[1] for call in mock_publish.call_args_list]
        assert "job.processing" in event_types
        assert "job.completed" in event_types

    def test_pipeline_records_tenant_processing_seconds(self, manager, tmp_path):
        """Successful tenant jobs persist processing-time usage."""
        tenant_id = "tenant_costtrack"
        _insert_job(
            "job_costtenant1",
            tenant_id=tenant_id,
            pages_completed=3,
        )

        factory = get_session_factory()
        session = factory()
        session.add(
            UsageRecord(
                tenant_id=tenant_id,
                period="2026-03",
                jobs_submitted=0,
                pages_processed=0,
                storage_bytes_used=0,
                api_calls=0,
                processing_seconds=0.0,
            )
        )
        session.commit()
        session.close()

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("api.job_manager.subprocess.Popen", return_value=mock_proc), \
             patch.object(manager, "_monitor_progress"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"), \
             patch("api.job_manager.datetime") as mock_datetime:
            from datetime import datetime, timezone

            mock_datetime.now.side_effect = [
                datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 10, 12, 0, 5, tzinfo=timezone.utc),
            ]
            mock_datetime.timezone = timezone
            manager._run_pipeline("job_costtenant1", str(tmp_path), str(tmp_path), {})

        session = factory()
        usage = (
            session.query(UsageRecord)
            .filter(UsageRecord.tenant_id == tenant_id, UsageRecord.period == "2026-03")
            .first()
        )
        assert usage is not None
        assert usage.processing_seconds == pytest.approx(5.0)
        session.close()


# ---------------------------------------------------------------------------
# cancel_job Tests
# ---------------------------------------------------------------------------


class TestCancelJobWithPid:
    """Tests for cancel_job() with running subprocess."""

    def test_cancel_with_pid_sends_signal(self, manager):
        """Cancel sends kill signal to running process."""
        job_id = _insert_job("job_ffffffffffff", status="processing", pid=99999)

        with patch("api.job_manager.os.kill") as mock_kill, \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            result = manager.cancel_job(job_id)

        assert result.status == "cancelled"
        assert result.pid is None
        mock_kill.assert_called_once()
        # Verify signal argument -- SIGKILL on Unix, 9 fallback on Windows
        call_args = mock_kill.call_args[0]
        assert call_args[0] == 99999

    def test_cancel_with_pid_swallows_oserror(self, manager):
        """Cancel handles OSError from os.kill gracefully."""
        job_id = _insert_job("job_111111111111", status="processing", pid=99999)

        with patch("api.job_manager.os.kill", side_effect=OSError("No such process")), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            result = manager.cancel_job(job_id)

        assert result.status == "cancelled"

    def test_cancel_completed_job_is_noop(self, manager):
        """Cancel of completed job returns job without changes."""
        job_id = _insert_job("job_222222222222", status="completed")

        result = manager.cancel_job(job_id)
        assert result.status == "completed"

    def test_cancel_failed_job_is_noop(self, manager):
        """Cancel of failed job returns job without changes."""
        job_id = _insert_job("job_failednoop1", status="failed")

        result = manager.cancel_job(job_id)
        assert result.status == "failed"

    def test_cancel_already_cancelled_is_noop(self, manager):
        """Cancel of already cancelled job returns job without changes."""
        job_id = _insert_job("job_cancelledno", status="cancelled")

        result = manager.cancel_job(job_id)
        assert result.status == "cancelled"

    def test_cancel_nonexistent_returns_none(self, manager):
        """Cancel of nonexistent job returns None."""
        result = manager.cancel_job("job_000000000000")
        assert result is None

    def test_cancel_submitted_job_without_pid(self, manager):
        """Cancel of submitted job (no PID yet) transitions to cancelled."""
        job_id = _insert_job("job_nopidcancel", status="submitted")

        with patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event"):
            result = manager.cancel_job(job_id)

        assert result.status == "cancelled"
        assert result.completed_at is not None

    def test_cancel_fires_webhook(self, manager):
        """Cancel fires webhook after status transition."""
        job_id = _insert_job("job_cancelwebhk", status="processing", pid=99999)

        with patch("api.job_manager.os.kill"), \
             patch.object(manager, "_fire_webhook") as mock_webhook, \
             patch.object(manager, "_publish_job_event"):
            manager.cancel_job(job_id)

        mock_webhook.assert_called_once()

    def test_cancel_terminal_does_not_fire_webhook(self, manager):
        """Cancel of already-terminal job does not fire webhook."""
        job_id = _insert_job("job_termnowebhk", status="completed")

        with patch.object(manager, "_fire_webhook") as mock_webhook:
            manager.cancel_job(job_id)

        mock_webhook.assert_not_called()

    def test_cancel_publishes_cancelled_event(self, manager):
        """Cancel emits a durable cancelled event for non-terminal jobs."""
        job_id = _insert_job("job_cancelpub01", status="processing", pid=99999)

        with patch("api.job_manager.os.kill"), \
             patch.object(manager, "_fire_webhook"), \
             patch.object(manager, "_publish_job_event") as mock_publish:
            manager.cancel_job(job_id)

        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[1] == "job.cancelled"


class TestSubmitJob:
    """Tests for submitted-job event publication."""

    def test_submit_publishes_submitted_event(self, manager, tmp_path):
        """Submitting a job emits a durable submitted event after commit."""
        fake_thread = MagicMock()

        with patch("api.job_manager.threading.Thread", return_value=fake_thread), \
             patch.object(manager, "_publish_job_event") as mock_publish:
            job = manager.submit(
                upload_filename="input.pdf",
                upload_content=b"test",
            )

        assert job.status == "submitted"
        assert fake_thread.start.call_count == manager._priority_queue._max_concurrent
        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[1] == "job.submitted"


# ---------------------------------------------------------------------------
# get_result_artifacts Tests
# ---------------------------------------------------------------------------


class TestGetResultArtifacts:
    """Tests for get_result_artifacts()."""

    def test_returns_all_artifact_types(self, manager, tmp_path):
        """Discovers PDF, TEXT, and STRUCTURE artifacts."""
        result_dir = tmp_path / "results"
        pdf_dir = result_dir / "EXPORT" / "PDF"
        text_dir = result_dir / "EXPORT" / "TEXT"
        struct_dir = result_dir / "EXPORT" / "STRUCTURE"
        pdf_dir.mkdir(parents=True)
        text_dir.mkdir(parents=True)
        struct_dir.mkdir(parents=True)

        (pdf_dir / "doc.pdf").write_bytes(b"%PDF-1.0")
        (text_dir / "doc.txt").write_text("Hello world")
        (struct_dir / "doc.json").write_text("{}")

        job_id = _insert_job(
            "job_333333333333",
            status="completed",
            result_path=str(result_dir),
        )

        artifacts = manager.get_result_artifacts(job_id)
        assert "pdf" in artifacts
        assert "text" in artifacts
        assert "structure" in artifacts
        # Verify artifact paths point to actual files
        assert artifacts["pdf"].endswith("doc.pdf")
        assert artifacts["text"].endswith("doc.txt")
        assert artifacts["structure"].endswith("doc.json")

    def test_returns_empty_for_no_result_path(self, manager):
        """Returns empty dict when job has no result_path."""
        job_id = _insert_job("job_444444444444", status="completed")

        artifacts = manager.get_result_artifacts(job_id)
        assert artifacts == {}

    def test_returns_empty_for_nonexistent_job(self, manager):
        """Returns empty dict for nonexistent job."""
        artifacts = manager.get_result_artifacts("job_000000000000")
        assert artifacts == {}

    def test_partial_artifacts_pdf_only(self, manager, tmp_path):
        """Returns only existing artifact types."""
        result_dir = tmp_path / "results"
        pdf_dir = result_dir / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"%PDF-1.0")

        job_id = _insert_job(
            "job_555555555555",
            status="completed",
            result_path=str(result_dir),
        )

        artifacts = manager.get_result_artifacts(job_id)
        assert "pdf" in artifacts
        assert "text" not in artifacts
        assert "structure" not in artifacts

    def test_empty_export_dirs(self, manager, tmp_path):
        """Returns empty dict when EXPORT subdirectories exist but are empty."""
        result_dir = tmp_path / "results"
        (result_dir / "EXPORT" / "PDF").mkdir(parents=True)
        (result_dir / "EXPORT" / "TEXT").mkdir(parents=True)

        job_id = _insert_job(
            "job_emptyexport",
            status="completed",
            result_path=str(result_dir),
        )

        artifacts = manager.get_result_artifacts(job_id)
        assert artifacts == {}

    def test_no_export_dir_at_all(self, manager, tmp_path):
        """Returns empty dict when result_path exists but has no EXPORT dir."""
        result_dir = tmp_path / "results"
        result_dir.mkdir()

        job_id = _insert_job(
            "job_noexportdir",
            status="completed",
            result_path=str(result_dir),
        )

        artifacts = manager.get_result_artifacts(job_id)
        assert artifacts == {}


# ---------------------------------------------------------------------------
# _fire_webhook Tests
# ---------------------------------------------------------------------------


class TestFireWebhook:
    """Tests for _fire_webhook()."""

    def test_no_webhook_url_is_noop(self, manager):
        """No-op when job has no webhook_url."""
        job = MagicMock()
        job.webhook_url = None

        with patch("api.webhooks.start_webhook_delivery") as mock_deliver:
            manager._fire_webhook(job)

        mock_deliver.assert_not_called()

    def test_empty_webhook_url_is_noop(self, manager):
        """No-op when job webhook_url is empty string."""
        job = MagicMock()
        job.webhook_url = ""

        with patch("api.webhooks.start_webhook_delivery") as mock_deliver:
            manager._fire_webhook(job)

        mock_deliver.assert_not_called()

    def test_with_webhook_url_calls_delivery(self, manager):
        """Calls start_webhook_delivery when webhook_url is set."""
        job = MagicMock()
        job.webhook_url = "https://example.com/hook"
        job.job_id = "job_666666666666"

        with patch("api.webhooks.start_webhook_delivery") as mock_deliver:
            manager._fire_webhook(job)

        mock_deliver.assert_called_once()
        # Verify job_id was passed as first argument
        call_args = mock_deliver.call_args
        assert call_args[0][0] == "job_666666666666"

    def test_webhook_passes_config_values(self, manager):
        """Webhook delivery receives timeout, retries, and secret from config."""
        job = MagicMock()
        job.webhook_url = "https://example.com/hook"
        job.job_id = "job_configcheck1"

        with patch("api.webhooks.start_webhook_delivery") as mock_deliver:
            manager._fire_webhook(job)

        call_kwargs = mock_deliver.call_args[1]
        assert call_kwargs["webhook_timeout"] == 30
        assert call_kwargs["webhook_max_retries"] == 0
        assert call_kwargs["webhook_secret_default"] == ""


# ---------------------------------------------------------------------------
# check_queue_capacity Tests
# ---------------------------------------------------------------------------


class TestCheckQueueCapacity:
    """Tests for check_queue_capacity()."""

    def test_empty_queue_has_capacity(self, manager):
        """Empty queue returns True."""
        assert manager.check_queue_capacity() is True

    def test_full_queue_no_capacity(self, manager):
        """Queue at MAX_CONCURRENT_JOBS returns False."""
        # MAX_CONCURRENT_JOBS is 64 in our mock config.
        # Insert 64 active jobs.
        for i in range(64):
            _insert_job(f"job_full_{i:04d}", status="submitted")

        assert manager.check_queue_capacity() is False

    def test_completed_jobs_dont_count(self, manager):
        """Completed/failed/cancelled jobs do not reduce capacity."""
        _insert_job("job_done_1", status="completed")
        _insert_job("job_done_2", status="failed")
        _insert_job("job_done_3", status="cancelled")

        assert manager.check_queue_capacity() is True


# ---------------------------------------------------------------------------
# get_job / list_jobs Tests
# ---------------------------------------------------------------------------


class TestQueryMethods:
    """Tests for get_job() and list_jobs()."""

    def test_get_job_exists(self, manager):
        """get_job returns the job when it exists."""
        _insert_job("job_gettest1234")
        job = manager.get_job("job_gettest1234")
        assert job is not None
        assert job.job_id == "job_gettest1234"

    def test_get_job_nonexistent(self, manager):
        """get_job returns None for nonexistent job."""
        job = manager.get_job("job_doesnotexst")
        assert job is None

    def test_list_jobs_empty(self, manager):
        """list_jobs returns empty list and zero total."""
        jobs, total = manager.list_jobs()
        assert jobs == []
        assert total == 0

    def test_list_jobs_with_filter(self, manager):
        """list_jobs filters by status."""
        _insert_job("job_listfilter1", status="submitted")
        _insert_job("job_listfilter2", status="completed")
        _insert_job("job_listfilter3", status="submitted")

        jobs, total = manager.list_jobs(status="submitted")
        assert total == 2
        assert len(jobs) == 2

    def test_list_jobs_pagination(self, manager):
        """list_jobs supports limit and offset."""
        for i in range(5):
            _insert_job(f"job_paginate_{i:02d}")

        jobs, total = manager.list_jobs(limit=2, offset=0)
        assert total == 5
        assert len(jobs) == 2

        jobs2, total2 = manager.list_jobs(limit=2, offset=2)
        assert total2 == 5
        assert len(jobs2) == 2

        jobs3, total3 = manager.list_jobs(limit=2, offset=4)
        assert total3 == 5
        assert len(jobs3) == 1

    def test_list_jobs_pagination_legacy_page_per_page(self, manager):
        """list_jobs backward-compat: legacy page/per_page still works."""
        for i in range(5):
            _insert_job(f"job_legacy_pg_{i:02d}")

        jobs, total = manager.list_jobs(page=1, per_page=2)
        assert total == 5
        assert len(jobs) == 2

        jobs2, total2 = manager.list_jobs(page=2, per_page=2)
        assert total2 == 5
        assert len(jobs2) == 2

        jobs3, total3 = manager.list_jobs(page=3, per_page=2)
        assert total3 == 5
        assert len(jobs3) == 1


# ---------------------------------------------------------------------------
# Graceful Shutdown Tests
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Tests for JobManager.shutdown() and non-daemon worker threads."""

    def test_worker_threads_are_non_daemon(self, manager):
        """Worker threads must be non-daemon so in-flight jobs are not killed
        when the interpreter exits."""
        manager._ensure_workers()
        try:
            assert len(manager._worker_threads) > 0
            for t in manager._worker_threads:
                assert t.daemon is False, (
                    f"Worker thread {t.name} is daemon=True; daemon threads "
                    "are killed at interpreter exit without cleanup."
                )
        finally:
            assert manager.shutdown(timeout=5.0) is True

    def test_shutdown_joins_idle_workers(self, manager):
        """shutdown() must unblock idle workers waiting in get()."""
        manager._ensure_workers()
        threads = list(manager._worker_threads)
        assert threads, "Expected at least one worker thread"

        # All workers are idle (queue is empty); shutdown should unblock them
        # via the PriorityJobQueue shutdown signal.
        result = manager.shutdown(timeout=5.0)
        assert result is True
        for t in threads:
            assert not t.is_alive(), f"Worker {t.name} still alive after shutdown"

    def test_shutdown_is_idempotent(self, manager):
        """Calling shutdown() twice is safe and returns True both times."""
        manager._ensure_workers()
        assert manager.shutdown(timeout=5.0) is True
        # Second call must be a no-op.
        assert manager.shutdown(timeout=5.0) is True

    def test_shutdown_without_started_workers_is_noop(self, manager):
        """shutdown() before any job is submitted should return True immediately."""
        # _ensure_workers has not been called; no threads exist.
        assert not manager._worker_threads
        assert manager.shutdown(timeout=5.0) is True

    def test_shutdown_returns_false_when_worker_stuck(self, manager):
        """If a worker is still running past the deadline, shutdown returns False."""
        import threading as _threading

        block_event = _threading.Event()

        def _blocking_loop() -> None:
            # Simulate a worker thread that ignores the shutdown signal long
            # enough to exceed our test deadline.
            block_event.wait(timeout=10.0)

        fake = _threading.Thread(
            target=_blocking_loop, name="fake-stuck-worker", daemon=True
        )
        fake.start()
        manager._workers_started = True
        manager._worker_threads.append(fake)
        try:
            result = manager.shutdown(timeout=0.3)
            assert result is False
        finally:
            block_event.set()
            fake.join(timeout=5.0)
