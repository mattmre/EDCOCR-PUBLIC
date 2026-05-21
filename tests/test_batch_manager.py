"""Tests for api/batch_manager.py -- unit tests for BatchManager methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from api.batch_manager import BatchManager, derive_batch_status
from api.database import Batch, Job, get_session_factory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager(tmp_path):
    """Create a BatchManager with a mock JobManager."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with patch("api.batch_manager.config") as mock_config:
        mock_config.MAX_BATCH_SIZE = 50
        mock_config.WEBHOOK_TIMEOUT = 30
        mock_config.WEBHOOK_MAX_RETRIES = 0
        mock_config.WEBHOOK_SECRET = ""

        factory = get_session_factory()

        # Create a mock job manager
        mock_jm = MagicMock()
        mock_jm.submit.side_effect = _make_submit_side_effect(factory)
        mock_jm.cancel_job.side_effect = _make_cancel_side_effect(factory)
        mock_jm.retry_job.side_effect = _make_retry_side_effect(factory)

        yield BatchManager(factory, job_manager=mock_jm)


def _make_submit_side_effect(factory):
    """Create a submit side effect that inserts jobs into DB."""
    import uuid

    def _submit(**kwargs):
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        session = factory()
        job = Job(
            job_id=job_id,
            status="submitted",
            source_file=kwargs.get("upload_filename", kwargs.get("source_path", "test.pdf")),
            priority=kwargs.get("priority", "normal"),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        session.close()
        return job

    return _submit


def _make_cancel_side_effect(factory):
    """Create a cancel side effect that updates job status."""
    def _cancel(job_id):
        session = factory()
        job = session.get(Job, job_id)
        if job and job.status not in ("completed", "failed", "cancelled"):
            job.status = "cancelled"
            session.commit()
            session.refresh(job)
        session.close()
        return job

    return _cancel


def _make_retry_side_effect(factory):
    """Create a retry side effect that creates a new job."""
    import uuid

    def _retry(job_id):
        session = factory()
        original = session.get(Job, job_id)
        if not original:
            return None
        new_id = f"job_{uuid.uuid4().hex[:12]}"
        new_job = Job(
            job_id=new_id,
            status="submitted",
            source_file=original.source_file,
            priority=original.priority,
        )
        session.add(new_job)
        session.commit()
        session.refresh(new_job)
        session.close()
        return new_job

    return _retry


def _insert_batch(batch_id, status="submitted", total_jobs=0, **kwargs):
    """Insert a batch directly into the database."""
    factory = get_session_factory()
    session = factory()
    batch = Batch(
        batch_id=batch_id,
        status=status,
        total_jobs=total_jobs,
        priority=kwargs.pop("priority", "normal"),
    )
    for key, value in kwargs.items():
        setattr(batch, key, value)
    session.add(batch)
    session.commit()
    session.close()


def _insert_job(job_id, status="submitted", batch_id=None, **kwargs):
    """Insert a job directly into the database."""
    factory = get_session_factory()
    session = factory()
    job = Job(
        job_id=job_id,
        status=status,
        source_file=kwargs.pop("source_file", "test.pdf"),
        priority=kwargs.pop("priority", "normal"),
        batch_id=batch_id,
    )
    for key, value in kwargs.items():
        setattr(job, key, value)
    session.add(job)
    session.commit()
    session.close()


# ---------------------------------------------------------------------------
# Submit batch
# ---------------------------------------------------------------------------


class TestBatchSubmit:
    def test_submit_creates_batch_and_jobs(self, manager):
        """Submitting a batch creates a Batch record and child jobs."""
        with patch.object(manager, "_publish_batch_event"):
            batch, jobs = manager.submit_batch(
                files=[("file1.pdf", b"content1"), ("file2.pdf", b"content2")],
            )
        assert batch.batch_id.startswith("batch_")
        assert batch.total_jobs == 2
        assert len(jobs) == 2
        assert batch.status == "submitted"

    def test_submit_with_source_paths(self, manager):
        """Submitting with source_paths creates child jobs."""
        with patch.object(manager, "_publish_batch_event"):
            batch, jobs = manager.submit_batch(
                source_paths=["/path/to/file1.pdf", "/path/to/file2.pdf"],
            )
        assert batch.total_jobs == 2
        assert len(jobs) == 2

    def test_submit_with_files_and_paths(self, manager):
        """Submitting with both files and paths creates all child jobs."""
        with patch.object(manager, "_publish_batch_event"):
            batch, jobs = manager.submit_batch(
                files=[("file1.pdf", b"content")],
                source_paths=["/path/to/file2.pdf"],
            )
        assert batch.total_jobs == 2
        assert len(jobs) == 2

    def test_submit_empty_raises_error(self, manager):
        """Submitting with no files or paths raises ValueError."""
        with pytest.raises(ValueError, match="At least one file"):
            manager.submit_batch()

    def test_submit_exceeds_max_size_raises_error(self, manager):
        """Exceeding MAX_BATCH_SIZE raises ValueError."""
        with patch("api.batch_manager.config") as mc:
            mc.MAX_BATCH_SIZE = 1
            with pytest.raises(ValueError, match="exceeds maximum"):
                manager.submit_batch(
                    files=[("f1.pdf", b"c1"), ("f2.pdf", b"c2")],
                )

    def test_submit_sets_priority(self, manager):
        """Batch priority is set on the batch record."""
        with patch.object(manager, "_publish_batch_event"):
            batch, _ = manager.submit_batch(
                files=[("f.pdf", b"c")],
                priority="urgent",
            )
        assert batch.priority == "urgent"

    def test_submit_sets_webhook(self, manager):
        """Batch webhook fields are set and secret is encrypted at rest."""
        from api.config import decrypt_webhook_secret

        with patch.object(manager, "_publish_batch_event"):
            batch, _ = manager.submit_batch(
                files=[("f.pdf", b"c")],
                webhook_url="https://example.com/hook",
                webhook_secret="secret123",
            )
        assert batch.webhook_url == "https://example.com/hook"
        # Secret should be encrypted at rest (not stored as plaintext)
        assert batch.webhook_secret != "secret123"
        assert decrypt_webhook_secret(batch.webhook_secret) == "secret123"

    def test_submit_sets_processing_timeout_override(self, manager):
        """Batch settings and child jobs receive the timeout override."""
        with patch.object(manager, "_publish_batch_event"):
            batch, _ = manager.submit_batch(
                files=[("f.pdf", b"c")],
                processing_timeout_minutes=12,
            )
        assert batch.settings["processing_timeout_minutes"] == 12
        assert manager._job_manager.submit.call_args.kwargs["processing_timeout_minutes"] == 12

    def test_child_jobs_have_batch_id(self, manager):
        """Child jobs are associated with the batch."""
        with patch.object(manager, "_publish_batch_event"):
            batch, jobs = manager.submit_batch(
                files=[("f.pdf", b"c")],
            )
        factory = get_session_factory()
        session = factory()
        db_job = session.get(Job, jobs[0].job_id)
        assert db_job.batch_id == batch.batch_id
        session.close()

    def test_submit_publishes_submitted_event(self, manager):
        """Batch submit emits a durable submitted event after child jobs are linked."""
        with patch.object(manager, "_publish_batch_event") as mock_publish:
            manager.submit_batch(files=[("file1.pdf", b"content1")])

        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[1] == "batch.submitted"


# ---------------------------------------------------------------------------
# Batch status derivation
# ---------------------------------------------------------------------------


class TestBatchStatusDerivation:
    def test_all_completed(self):
        """All jobs completed -> batch completed."""
        jobs = [MagicMock(status="completed") for _ in range(3)]
        assert derive_batch_status(jobs) == "completed"

    def test_all_failed(self):
        """All jobs failed -> batch failed."""
        jobs = [MagicMock(status="failed") for _ in range(3)]
        assert derive_batch_status(jobs) == "failed"

    def test_all_cancelled(self):
        """All jobs cancelled -> batch cancelled."""
        jobs = [MagicMock(status="cancelled") for _ in range(3)]
        assert derive_batch_status(jobs) == "cancelled"

    def test_partial_failure(self):
        """Mix of completed and failed -> partial_failure."""
        jobs = [
            MagicMock(status="completed"),
            MagicMock(status="failed"),
            MagicMock(status="completed"),
        ]
        assert derive_batch_status(jobs) == "partial_failure"

    def test_still_processing(self):
        """Some jobs still processing -> processing."""
        jobs = [
            MagicMock(status="completed"),
            MagicMock(status="processing"),
        ]
        assert derive_batch_status(jobs) == "processing"

    def test_still_submitted(self):
        """All jobs submitted -> submitted."""
        jobs = [MagicMock(status="submitted") for _ in range(3)]
        assert derive_batch_status(jobs) == "submitted"

    def test_empty_jobs(self):
        """No jobs -> submitted."""
        assert derive_batch_status([]) == "submitted"

    def test_mix_terminal_and_processing(self):
        """Mix of completed and processing -> processing."""
        jobs = [
            MagicMock(status="completed"),
            MagicMock(status="processing"),
            MagicMock(status="submitted"),
        ]
        assert derive_batch_status(jobs) == "processing"

    def test_completed_failed_cancelled_mix(self):
        """Mix of all three terminal types -> partial_failure."""
        jobs = [
            MagicMock(status="completed"),
            MagicMock(status="failed"),
            MagicMock(status="cancelled"),
        ]
        assert derive_batch_status(jobs) == "partial_failure"


# ---------------------------------------------------------------------------
# Cancel batch
# ---------------------------------------------------------------------------


class TestBatchCancel:
    def test_cancel_batch_cancels_non_terminal_jobs(self, manager):
        """Cancelling a batch cancels non-terminal child jobs."""
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=3)
        _insert_job("job_aaaaaaaaaaaa", status="submitted", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", status="processing", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_cccccccccccc", status="completed", batch_id="batch_aaaaaaaaaaaa")

        with patch.object(manager, "_publish_batch_event"):
            batch = manager.cancel_batch("batch_aaaaaaaaaaaa")
        assert batch is not None
        assert batch.status == "cancelled"

        # Verify the mock cancel was called for non-terminal jobs
        cancel_calls = manager._job_manager.cancel_job.call_args_list
        cancelled_ids = {c.args[0] for c in cancel_calls}
        assert "job_aaaaaaaaaaaa" in cancelled_ids
        assert "job_bbbbbbbbbbbb" in cancelled_ids
        assert "job_cccccccccccc" not in cancelled_ids

    def test_cancel_nonexistent_batch_returns_none(self, manager):
        """Cancelling a non-existent batch returns None."""
        result = manager.cancel_batch("batch_000000000000")
        assert result is None

    def test_cancel_already_terminal_batch(self, manager):
        """Cancelling an already terminal batch returns it unchanged."""
        _insert_batch("batch_aaaaaaaaaaaa", status="completed", total_jobs=1)
        with patch.object(manager, "_publish_batch_event"):
            batch = manager.cancel_batch("batch_aaaaaaaaaaaa")
        assert batch.status == "completed"

    def test_cancel_publishes_cancelled_event(self, manager):
        """Cancelling a batch emits a durable cancelled event."""
        _insert_batch("batch_cancelpub01", total_jobs=1)
        _insert_job("job_cancelpub01", status="processing", batch_id="batch_cancelpub01")

        with patch.object(manager, "_publish_batch_event") as mock_publish:
            manager.cancel_batch("batch_cancelpub01")

        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[1] == "batch.cancelled"


# ---------------------------------------------------------------------------
# Retry batch
# ---------------------------------------------------------------------------


class TestBatchRetry:
    def test_retry_batch_retries_failed_jobs(self, manager):
        """Retrying a batch creates new jobs for failed children."""
        _insert_batch("batch_aaaaaaaaaaaa", status="partial_failure", total_jobs=2)
        _insert_job("job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", status="failed", batch_id="batch_aaaaaaaaaaaa")

        with patch.object(manager, "_publish_batch_event"):
            result = manager.retry_batch("batch_aaaaaaaaaaaa")
        assert result is not None
        batch, new_jobs = result
        assert batch.status == "processing"
        assert len(new_jobs) == 1

    def test_retry_nonexistent_batch_returns_none(self, manager):
        """Retrying a non-existent batch returns None."""
        with patch.object(manager, "_publish_batch_event"):
            result = manager.retry_batch("batch_000000000000")
        assert result is None

    def test_retry_no_failed_jobs_raises_error(self, manager):
        """Retrying with no failed jobs raises ValueError."""
        _insert_batch("batch_aaaaaaaaaaaa", status="completed", total_jobs=1)
        _insert_job("job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa")

        with pytest.raises(ValueError, match="No failed or cancelled"):
            manager.retry_batch("batch_aaaaaaaaaaaa")

    def test_retry_publishes_processing_event(self, manager):
        """Retrying a batch emits a durable processing event."""
        _insert_batch("batch_retrypub01", status="partial_failure", total_jobs=2)
        _insert_job("job_retrydone01", status="completed", batch_id="batch_retrypub01")
        _insert_job("job_retryfail01", status="failed", batch_id="batch_retrypub01")

        with patch.object(manager, "_publish_batch_event") as mock_publish:
            manager.retry_batch("batch_retrypub01")

        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[1] == "batch.processing"


# ---------------------------------------------------------------------------
# Batch completion detection
# ---------------------------------------------------------------------------


class TestBatchCompletion:
    def test_batch_completes_when_all_jobs_terminal(self, manager):
        """Batch status updates to completed when all jobs are terminal."""
        _insert_batch("batch_aaaaaaaaaaaa", status="processing", total_jobs=2)
        _insert_job("job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", status="completed", batch_id="batch_aaaaaaaaaaaa")

        with patch.object(manager, "_publish_batch_event"):
            manager.check_batch_completion("batch_aaaaaaaaaaaa")

        batch = manager.get_batch("batch_aaaaaaaaaaaa")
        assert batch.status == "completed"
        assert batch.completed_at is not None
        assert batch.jobs_completed == 2

    def test_batch_partial_failure(self, manager):
        """Batch is partial_failure when some jobs fail."""
        _insert_batch("batch_aaaaaaaaaaaa", status="processing", total_jobs=2)
        _insert_job("job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", status="failed", batch_id="batch_aaaaaaaaaaaa")

        with patch.object(manager, "_publish_batch_event"):
            manager.check_batch_completion("batch_aaaaaaaaaaaa")

        batch = manager.get_batch("batch_aaaaaaaaaaaa")
        assert batch.status == "partial_failure"
        assert batch.jobs_completed == 1
        assert batch.jobs_failed == 1

    def test_batch_does_not_complete_while_processing(self, manager):
        """Batch stays processing when some jobs are not terminal."""
        _insert_batch("batch_aaaaaaaaaaaa", status="processing", total_jobs=2)
        _insert_job("job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", status="processing", batch_id="batch_aaaaaaaaaaaa")

        with patch.object(manager, "_publish_batch_event"):
            manager.check_batch_completion("batch_aaaaaaaaaaaa")

        batch = manager.get_batch("batch_aaaaaaaaaaaa")
        assert batch.status == "processing"  # Unchanged
        assert batch.completed_at is None

    def test_batch_completion_fires_webhook(self, manager):
        """Batch webhook is fired when batch completes."""
        _insert_batch(
            "batch_aaaaaaaaaaaa",
            status="processing",
            total_jobs=1,
            webhook_url="https://example.com/hook",
        )
        _insert_job("job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa")

        with patch("api.batch_manager.BatchManager._fire_batch_webhook") as mock_fire:
            with patch.object(manager, "_publish_batch_event"):
                manager.check_batch_completion("batch_aaaaaaaaaaaa")
            mock_fire.assert_called_once_with("batch_aaaaaaaaaaaa")

    def test_batch_completion_nonexistent(self, manager):
        """Checking completion for non-existent batch does nothing."""
        # Should not raise
        with patch.object(manager, "_publish_batch_event"):
            manager.check_batch_completion("batch_000000000000")

    def test_batch_all_failed_status(self, manager):
        """All failed jobs -> batch failed."""
        _insert_batch("batch_aaaaaaaaaaaa", status="processing", total_jobs=2)
        _insert_job("job_aaaaaaaaaaaa", status="failed", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", status="failed", batch_id="batch_aaaaaaaaaaaa")

        with patch.object(manager, "_publish_batch_event"):
            manager.check_batch_completion("batch_aaaaaaaaaaaa")

        batch = manager.get_batch("batch_aaaaaaaaaaaa")
        assert batch.status == "failed"
        assert batch.jobs_failed == 2

    def test_batch_already_terminal_skipped(self, manager):
        """Already terminal batch is not re-processed."""
        _insert_batch(
            "batch_aaaaaaaaaaaa",
            status="completed",
            total_jobs=1,
        )
        _insert_job("job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa")

        with patch("api.batch_manager.BatchManager._fire_batch_webhook") as mock_fire:
            with patch.object(manager, "_publish_batch_event"):
                manager.check_batch_completion("batch_aaaaaaaaaaaa")
            mock_fire.assert_not_called()

    def test_completion_publishes_terminal_event(self, manager):
        """All-terminal completion emits the derived terminal event."""
        _insert_batch("batch_termpub01", status="processing", total_jobs=2)
        _insert_job("job_termpub001", status="completed", batch_id="batch_termpub01")
        _insert_job("job_termpub002", status="failed", batch_id="batch_termpub01")

        with patch.object(manager, "_publish_batch_event") as mock_publish:
            manager.check_batch_completion("batch_termpub01")

        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[1] == "batch.partial_failure"


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestBatchQuery:
    def test_get_batch(self, manager):
        """get_batch returns the batch record."""
        _insert_batch("batch_aaaaaaaaaaaa")
        batch = manager.get_batch("batch_aaaaaaaaaaaa")
        assert batch is not None
        assert batch.batch_id == "batch_aaaaaaaaaaaa"

    def test_get_batch_not_found(self, manager):
        """get_batch returns None for non-existent batch."""
        assert manager.get_batch("batch_000000000000") is None

    def test_get_batch_jobs(self, manager):
        """get_batch_jobs returns child jobs."""
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=2)
        _insert_job("job_aaaaaaaaaaaa", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_cccccccccccc")  # Not in batch

        jobs = manager.get_batch_jobs("batch_aaaaaaaaaaaa")
        assert len(jobs) == 2
        job_ids = {j.job_id for j in jobs}
        assert "job_aaaaaaaaaaaa" in job_ids
        assert "job_bbbbbbbbbbbb" in job_ids

    def test_get_batch_jobs_empty(self, manager):
        """get_batch_jobs returns empty for batch with no jobs."""
        _insert_batch("batch_aaaaaaaaaaaa")
        jobs = manager.get_batch_jobs("batch_aaaaaaaaaaaa")
        assert jobs == []
