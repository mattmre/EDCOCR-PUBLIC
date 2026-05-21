"""Tests for dashboard collector wiring in api/job_manager.py.

Verifies that MetricsCollector, AnalyticsStore, and QueueMonitor
are populated at job lifecycle points (submit and completion).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from api.database import Job, get_session_factory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager(tmp_path):
    """Create a JobManager with isolated config and patched pipeline."""
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


def _seed_source_file(tmp_path, manager):
    """Write a tiny PDF-like file into the source folder and return its path."""
    import pathlib

    # manager fixture patches config.SOURCE_FOLDER to tmp_path/source
    source_dir = pathlib.Path(manager._session_factory().bind.url.database).parent / "source"
    # Fallback: extract from the patched config
    from api.job_manager import config
    source_dir = pathlib.Path(config.SOURCE_FOLDER)
    target = source_dir / "sample.pdf"
    target.write_bytes(b"%PDF-1.4 test content")
    return str(target)


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
# Submit wiring tests
# ---------------------------------------------------------------------------


class TestDashboardWiringOnSubmit:
    """Verify collectors are updated when a job is submitted."""

    @patch("api.job_manager.JobManager._run_pipeline")
    def test_submit_updates_stage_metrics(self, mock_pipeline, manager, tmp_path):
        """MetricsCollector.update_stage is called on submit."""
        from api.dashboard import get_collector

        collector = get_collector()
        collector.reset()

        source_file = tmp_path / "source" / "test.pdf"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_bytes(b"%PDF-1.4 test")

        manager.submit(source_path=str(source_file))

        # After submit, the stage metrics should have been updated
        snapshot = collector.get_snapshot()
        stage_names = [s.stage for s in snapshot.stages]
        assert "processing" in stage_names

    @patch("api.job_manager.JobManager._run_pipeline")
    def test_submit_updates_queue_depth(self, mock_pipeline, manager, tmp_path):
        """QueueMonitor.record_depth is called on submit."""
        from api.queue_alerting import get_queue_monitor

        monitor = get_queue_monitor()
        monitor.reset()

        source_file = tmp_path / "source" / "test.pdf"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_bytes(b"%PDF-1.4 test")

        manager.submit(source_path=str(source_file))

        snapshot = monitor.get_snapshot()
        queue_names = [q["queue_name"] for q in snapshot.queues]
        assert "processing" in queue_names
        # Depth should be at least 1 (the newly submitted job)
        for q in snapshot.queues:
            if q["queue_name"] == "processing":
                assert q["depth"] >= 1

    @patch("api.job_manager.JobManager._run_pipeline")
    def test_submit_collector_failure_does_not_break_pipeline(
        self, mock_pipeline, manager, tmp_path
    ):
        """If the dashboard collector throws, submit still succeeds."""
        source_file = tmp_path / "source" / "test.pdf"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_bytes(b"%PDF-1.4 test")

        with patch("api.dashboard.get_collector", side_effect=RuntimeError("boom")):
            # Should not raise
            job = manager.submit(source_path=str(source_file))
            assert job.status == "submitted"


# ---------------------------------------------------------------------------
# Pipeline completion wiring tests
# ---------------------------------------------------------------------------


class TestDashboardWiringOnComplete:
    """Verify collectors are populated after job completion."""

    def test_completion_records_throughput_and_latency(self, manager, tmp_path):
        """MetricsCollector receives throughput and latency on pipeline success."""
        from api.dashboard import get_collector

        collector = get_collector()
        collector.reset()

        _insert_job("job_test001", status="processing", pages_completed=5)

        source_dir = str(tmp_path / "src")
        output_dir = str(tmp_path / "out")

        with patch("subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.pid = 9999
            proc.returncode = 0
            proc.poll.return_value = 0
            proc.wait.return_value = 0
            mock_popen.return_value = proc

            manager._run_pipeline("job_test001", source_dir, output_dir, {})

        snapshot = collector.get_snapshot()
        # At least one throughput point should have been recorded
        assert snapshot.completed_jobs >= 1 or snapshot.total_jobs >= 1
        # Latency should have been recorded (total_ms > 0 possible if test is fast)
        # Check that the data structures were populated
        assert len(collector._latency) >= 1
        assert collector._latency[-1].job_id == "job_test001"

    def test_completion_records_analytics(self, manager, tmp_path):
        """AnalyticsStore.record_job is called on pipeline success."""
        from api.analytics import get_analytics_store

        store = get_analytics_store()
        store.reset()

        _insert_job("job_test002", status="processing", pages_completed=3)

        source_dir = str(tmp_path / "src")
        output_dir = str(tmp_path / "out")

        with patch("subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.pid = 9999
            proc.returncode = 0
            proc.poll.return_value = 0
            proc.wait.return_value = 0
            mock_popen.return_value = proc

            manager._run_pipeline("job_test002", source_dir, output_dir, {})

        assert store.record_count >= 1
        # Verify the record has the correct job_id
        with store._lock:
            job_ids = [r.job_id for r in store._records]
        assert "job_test002" in job_ids

    def test_completion_updates_queue_depth(self, manager, tmp_path):
        """QueueMonitor.record_depth is called on pipeline completion."""
        from api.queue_alerting import get_queue_monitor

        monitor = get_queue_monitor()
        monitor.reset()

        _insert_job("job_test003", status="processing")

        source_dir = str(tmp_path / "src")
        output_dir = str(tmp_path / "out")

        with patch("subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.pid = 9999
            proc.returncode = 0
            proc.poll.return_value = 0
            proc.wait.return_value = 0
            mock_popen.return_value = proc

            manager._run_pipeline("job_test003", source_dir, output_dir, {})

        snapshot = monitor.get_snapshot()
        queue_names = [q["queue_name"] for q in snapshot.queues]
        assert "processing" in queue_names

    def test_failed_pipeline_still_records_metrics(self, manager, tmp_path):
        """Dashboard collectors are updated even when the pipeline fails."""
        from api.analytics import get_analytics_store
        from api.dashboard import get_collector

        collector = get_collector()
        collector.reset()
        store = get_analytics_store()
        store.reset()

        _insert_job("job_test004", status="processing")

        source_dir = str(tmp_path / "src")
        output_dir = str(tmp_path / "out")

        with patch("subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.pid = 9999
            proc.returncode = 1  # failure
            proc.poll.return_value = 1
            proc.wait.return_value = 1
            mock_popen.return_value = proc

            manager._run_pipeline("job_test004", source_dir, output_dir, {})

        # Analytics should record the failure
        with store._lock:
            matching = [r for r in store._records if r.job_id == "job_test004"]
        assert len(matching) == 1
        assert matching[0].success is False

        # Job counts should reflect the failure
        assert collector._job_counts["failed"] >= 1

    def test_exception_path_records_metrics(self, manager, tmp_path):
        """Dashboard collectors are called even in the exception handler path."""
        from api.analytics import get_analytics_store

        store = get_analytics_store()
        store.reset()

        _insert_job("job_test005", status="processing")

        source_dir = str(tmp_path / "src")
        output_dir = str(tmp_path / "out")

        # Force an exception by making Popen raise
        with patch("subprocess.Popen", side_effect=OSError("mock spawn failure")):
            manager._run_pipeline("job_test005", source_dir, output_dir, {})

        # The exception handler should still have recorded the failure
        with store._lock:
            matching = [r for r in store._records if r.job_id == "job_test005"]
        assert len(matching) == 1
        assert matching[0].success is False

    def test_collector_failure_does_not_break_pipeline_completion(
        self, manager, tmp_path
    ):
        """If dashboard collector throws, the pipeline still completes."""
        _insert_job("job_test006", status="processing")

        source_dir = str(tmp_path / "src")
        output_dir = str(tmp_path / "out")

        with patch("subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.pid = 9999
            proc.returncode = 0
            proc.poll.return_value = 0
            proc.wait.return_value = 0
            mock_popen.return_value = proc

            with patch(
                "api.dashboard.get_collector",
                side_effect=RuntimeError("collector boom"),
            ):
                # Should not raise
                manager._run_pipeline("job_test006", source_dir, output_dir, {})

        # Verify job still completed
        session = get_session_factory()()
        job = session.get(Job, "job_test006")
        assert job.status == "completed"
        session.close()
