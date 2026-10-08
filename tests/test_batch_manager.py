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
            source_file=kwargs.get(
                "upload_filename", kwargs.get("source_path", "test.pdf")
            ),
            priority=kwargs.get("priority", "normal"),
            batch_id=kwargs.get("batch_id"),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        session.close()
        return job

    return _submit


def _make_cancel_side_effect(factory):
    """Create a cancel side effect that updates job status."""

    def _cancel(job_id, tenant_id=None):
        session = factory()
        job = session.get(Job, job_id)
        if job and tenant_id is not None and job.tenant_id not in (None, tenant_id):
            session.close()
            return None
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

    def _retry(job_id, tenant_id=None, batch_id=None):
        session = factory()
        original = session.get(Job, job_id)
        if not original:
            session.close()
            return None
        if tenant_id is not None and original.tenant_id not in (None, tenant_id):
            session.close()
            return None
        new_id = f"job_{uuid.uuid4().hex[:12]}"
        new_job = Job(
            job_id=new_id,
            status="submitted",
            source_file=original.source_file,
            priority=original.priority,
            tenant_id=original.tenant_id,
            batch_id=batch_id,
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
        assert (
            manager._job_manager.submit.call_args.kwargs["processing_timeout_minutes"]
            == 12
        )

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

    def test_submit_reports_child_rejections(self, manager):
        """A failed child submission is durable and visible to the caller."""
        manager._job_manager.submit.side_effect = [
            RuntimeError("rejected by upload policy"),
            _make_submit_side_effect(get_session_factory())(upload_filename="ok.pdf"),
        ]
        with patch.object(manager, "_publish_batch_event"):
            batch, jobs = manager.submit_batch(
                files=[("bad.pdf", b"bad"), ("ok.pdf", b"ok")],
            )

        assert len(jobs) == 1
        assert batch.total_jobs == 1
        assert batch.settings["rejected_inputs"] == ["bad.pdf"]


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
        _insert_job(
            "job_aaaaaaaaaaaa", status="submitted", batch_id="batch_aaaaaaaaaaaa"
        )
        _insert_job(
            "job_bbbbbbbbbbbb", status="processing", batch_id="batch_aaaaaaaaaaaa"
        )
        _insert_job(
            "job_cccccccccccc", status="completed", batch_id="batch_aaaaaaaaaaaa"
        )

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
        _insert_job(
            "job_cancelpub01", status="processing", batch_id="batch_cancelpub01"
        )

        with patch.object(manager, "_publish_batch_event") as mock_publish:
            manager.cancel_batch("batch_cancelpub01")

        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[1] == "batch.cancelled"


# ---------------------------------------------------------------------------
# Retry batch
# ---------------------------------------------------------------------------


class TestBatchRetry:
    def test_retry_claims_all_children_until_replacements_are_admitted(self, manager):
        """A fast first failure cannot finalize while a later retry is in flight."""
        batch_id = "batch_retryfence01"
        _insert_batch(batch_id, status="failed", total_jobs=2)
        _insert_job("job_retryfence00", status="failed", batch_id=batch_id)
        _insert_job("job_retryfence01", status="failed", batch_id=batch_id)
        factory = get_session_factory()

        def retry_with_fast_first_failure(job_id, tenant_id=None, batch_id=None):
            session = factory()
            original = session.get(Job, job_id)
            replacement_id = (
                "job_newfence00" if job_id.endswith("00") else "job_newfence01"
            )
            original.retried_to = replacement_id
            original.settings = {
                "retried_to": replacement_id,
                "superseded": True,
                "previous_batch_id": batch_id,
            }
            replacement = Job(
                job_id=replacement_id,
                source_file=original.source_file,
                status="failed" if job_id.endswith("00") else "submitted",
                batch_id=batch_id,
            )
            session.add(replacement)
            session.commit()
            session.refresh(replacement)
            session.close()
            # Production child completion runs before retry_job returns.
            manager.check_batch_completion(batch_id)
            return replacement

        manager._job_manager.retry_job.side_effect = retry_with_fast_first_failure
        with patch.object(manager, "_publish_batch_event"):
            batch, replacements = manager.retry_batch(batch_id)

        assert len(replacements) == 2
        assert batch.status == "processing"
        session = factory()
        assert (
            session.query(Job)
            .filter(Job.batch_id == batch_id, Job.retry_claimed == 1)
            .count()
            == 0
        )
        second = session.get(Job, "job_newfence01")
        second.status = "completed"
        session.commit()
        session.close()

        manager.check_batch_completion(batch_id)
        session = factory()
        assert session.get(Batch, batch_id).status == "partial_failure"
        session.close()

    def test_reconciler_recovers_abandoned_retry_claims(self, manager):
        """A dead retry request cannot leave a batch permanently processing."""
        batch_id = "batch_retryrecover01"
        _insert_batch(batch_id, status="processing", total_jobs=2)
        _insert_job(
            "job_retryrecover00",
            status="failed",
            batch_id=batch_id,
            retry_claimed=1,
            retried_to="job_newrecover00",
        )
        _insert_job(
            "job_retryrecover01",
            status="failed",
            batch_id=batch_id,
            retry_claimed=1,
        )
        _insert_job(
            "job_newrecover00",
            status="completed",
            batch_id=batch_id,
        )

        assert manager.recover_abandoned_retry_claims(batch_id) == 2
        manager.check_batch_completion(batch_id)

        factory = get_session_factory()
        session = factory()
        recovered = session.get(Job, "job_retryrecover00")
        abandoned = session.get(Job, "job_retryrecover01")
        assert recovered.batch_id is None
        assert recovered.retry_claimed == 0
        assert abandoned.batch_id == batch_id
        assert abandoned.retry_claimed == 0
        assert session.get(Batch, batch_id).status == "partial_failure"
        session.close()

    def test_reconciler_recounts_after_detaching_admitted_original(self, manager):
        """Crash recovery keeps total_jobs aligned after removing the old row."""
        batch_id = "batch_retryrecount01"
        _insert_batch(batch_id, status="processing", total_jobs=3)
        _insert_job(
            "job_retryrecount00",
            status="failed",
            batch_id=batch_id,
            retry_claimed=1,
            retried_to="job_newrecount00",
        )
        _insert_job(
            "job_newrecount00",
            status="completed",
            batch_id=batch_id,
        )
        _insert_job(
            "job_newrecount01",
            status="completed",
            batch_id=batch_id,
        )

        assert manager.recover_abandoned_retry_claims(batch_id) == 1
        manager.check_batch_completion(batch_id)

        factory = get_session_factory()
        session = factory()
        batch = session.get(Batch, batch_id)
        assert batch.total_jobs == 2
        assert session.query(Job).filter(Job.batch_id == batch_id).count() == 2
        assert batch.status == "completed"
        session.close()

    def test_reconciler_adopts_unbatched_replacement_before_finalizing(self, manager):
        """Crash recovery must not strand an individual retry outside its batch."""
        batch_id = "batch_retryadopt01"
        _insert_batch(batch_id, status="processing", total_jobs=1)
        _insert_job(
            "job_retryadoptold",
            status="failed",
            batch_id=batch_id,
            retry_claimed=1,
            retried_to="job_retryadoptnew",
        )
        _insert_job(
            "job_retryadoptnew",
            status="submitted",
            batch_id=None,
        )

        assert manager.recover_abandoned_retry_claims(batch_id) == 1

        session = get_session_factory()()
        original = session.get(Job, "job_retryadoptold")
        replacement = session.get(Job, "job_retryadoptnew")
        batch = session.get(Batch, batch_id)
        assert original.batch_id is None
        assert original.retry_claimed == 0
        assert replacement.batch_id == batch_id
        assert batch.total_jobs == 1
        session.close()

        manager.check_batch_completion(batch_id)
        assert manager.get_batch(batch_id).status == "processing"
        with get_session_factory()() as session:
            session.get(Job, "job_retryadoptnew").status = "completed"
            session.commit()
        manager.check_batch_completion(batch_id)
        assert manager.get_batch(batch_id).status == "completed"

    def test_reconciler_keeps_claim_when_replacement_tenant_mismatches(self, manager):
        batch_id = "batch_retrytenantguard"
        _insert_batch(batch_id, status="processing", total_jobs=1, tenant_id="tenant_a")
        _insert_job(
            "job_retrytenantold",
            status="failed",
            batch_id=batch_id,
            tenant_id="tenant_a",
            retry_claimed=1,
            retried_to="job_retrytenantnew",
        )
        _insert_job(
            "job_retrytenantnew",
            status="submitted",
            batch_id=None,
            tenant_id="tenant_b",
        )

        assert manager.recover_abandoned_retry_claims(batch_id) == 0
        manager.check_batch_completion(batch_id)

        with get_session_factory()() as session:
            original = session.get(Job, "job_retrytenantold")
            replacement = session.get(Job, "job_retrytenantnew")
            assert original.retry_claimed == 1
            assert original.batch_id == batch_id
            assert replacement.batch_id is None
            assert session.get(Batch, batch_id).status == "processing"

    @pytest.mark.parametrize("stale_pointer", [None, "job_missing_foreign_retry"])
    def test_reconciler_keeps_foreign_original_claim(self, manager, stale_pointer):
        batch_id = f"batch_retryforeignparent_{'missing' if stale_pointer else 'none'}"
        old_id = f"job_retryforeignparent_{'missing' if stale_pointer else 'none'}"
        _insert_batch(batch_id, status="processing", total_jobs=1, tenant_id="tenant_a")
        _insert_job(
            old_id,
            status="failed",
            batch_id=batch_id,
            tenant_id="tenant_b",
            retry_claimed=1,
            retried_to=stale_pointer,
            settings={"retried_to": stale_pointer, "superseded": bool(stale_pointer)},
        )

        assert manager.recover_abandoned_retry_claims(batch_id) == 0

        with get_session_factory()() as session:
            original = session.get(Job, old_id)
            batch = session.get(Batch, batch_id)
            assert original.retry_claimed == 1
            assert original.batch_id == batch_id
            assert original.retried_to == stale_pointer
            assert batch.total_jobs == 1
            assert batch.status == "processing"

    @pytest.mark.parametrize("request_tenant", [None, "tenant_a"])
    def test_batch_retry_refuses_mixed_tenant_membership(
        self, manager, request_tenant
    ):
        batch_id = f"batch_retrymixed_{'platform' if request_tenant is None else 'scoped'}"
        old_id = f"job_retrymixed_{'platform' if request_tenant is None else 'scoped'}"
        _insert_batch(batch_id, status="failed", total_jobs=1, tenant_id="tenant_a")
        _insert_job(
            old_id,
            status="failed",
            batch_id=batch_id,
            tenant_id="tenant_b",
        )

        with pytest.raises(ValueError, match="outside its tenant membership"):
            manager.retry_batch(batch_id, tenant_id=request_tenant)

        manager._job_manager.retry_job.assert_not_called()
        with get_session_factory()() as session:
            original = session.get(Job, old_id)
            batch = session.get(Batch, batch_id)
            assert original.batch_id == batch_id
            assert original.retry_claimed in (None, 0)
            assert batch.total_jobs == 1
            assert batch.status == "failed"

    @pytest.mark.parametrize("link_shape", ["self", "cycle"])
    def test_reconciler_keeps_claim_for_self_or_cyclic_retry_link(
        self, manager, link_shape
    ):
        batch_id = f"batch_retrycycle_{link_shape}"
        original_id = f"job_retrycycle_old_{link_shape}"
        replacement_id = original_id if link_shape == "self" else f"job_retrycycle_new_{link_shape}"
        _insert_batch(batch_id, status="processing", total_jobs=1)
        _insert_job(
            original_id,
            status="failed",
            batch_id=batch_id,
            retry_claimed=1,
            retried_to=replacement_id,
        )
        if link_shape == "cycle":
            _insert_job(
                replacement_id,
                status="submitted",
                batch_id=None,
                retried_to=original_id,
            )

        assert manager.recover_abandoned_retry_claims(batch_id) == 0
        manager.check_batch_completion(batch_id)

        with get_session_factory()() as session:
            original = session.get(Job, original_id)
            batch = session.get(Batch, batch_id)
            assert original.batch_id == batch_id
            assert original.retry_claimed == 1
            assert batch.total_jobs == 1
            assert batch.status == "processing"

    @pytest.mark.parametrize("conflict", ["tenant", "batch"])
    def test_retry_batch_keeps_claim_for_foreign_replacement(self, manager, conflict):
        batch_id = f"batch_retryidentity_{conflict}"
        _insert_batch(batch_id, status="failed", total_jobs=1, tenant_id="tenant_a")
        _insert_job(
            f"job_retryidentity_old_{conflict}",
            status="failed",
            batch_id=batch_id,
            tenant_id="tenant_a",
        )
        replacement_batch = None
        replacement_tenant = "tenant_a"
        if conflict == "batch":
            replacement_batch = f"batch_retryidentity_other_{conflict}"
            _insert_batch(
                replacement_batch,
                status="processing",
                total_jobs=1,
                tenant_id="tenant_a",
            )
        else:
            replacement_tenant = "tenant_b"
        replacement_id = f"job_retryidentity_new_{conflict}"
        _insert_job(
            replacement_id,
            status="submitted",
            batch_id=replacement_batch,
            tenant_id=replacement_tenant,
        )
        with get_session_factory()() as session:
            replacement = session.get(Job, replacement_id)

        manager._job_manager.retry_job.side_effect = lambda *_a, **_kw: replacement
        with pytest.raises(ValueError, match="No jobs could be retried"):
            manager.retry_batch(batch_id)

        with get_session_factory()() as session:
            original = session.get(Job, f"job_retryidentity_old_{conflict}")
            stored_replacement = session.get(Job, replacement_id)
            batch = session.get(Batch, batch_id)
            assert original.retry_claimed == 1
            assert original.batch_id == batch_id
            assert stored_replacement.batch_id == replacement_batch
            assert batch.status == "processing"

    @pytest.mark.parametrize("conflict", ["tenant", "batch"])
    def test_batch_id_assignment_rejects_foreign_identity(self, manager, conflict):
        batch_id = f"batch_assignidentity_{conflict}"
        _insert_batch(batch_id, tenant_id="tenant_a")
        job_batch = None
        job_tenant = "tenant_a"
        if conflict == "batch":
            job_batch = f"batch_assignidentity_other_{conflict}"
            _insert_batch(job_batch, tenant_id="tenant_a")
        else:
            job_tenant = "tenant_b"
        job_id = f"job_assignidentity_{conflict}"
        _insert_job(job_id, batch_id=job_batch, tenant_id=job_tenant)

        with pytest.raises(ValueError, match="another batch|tenant does not match"):
            manager._set_job_batch_id(job_id, batch_id)

        with get_session_factory()() as session:
            assert session.get(Job, job_id).batch_id == job_batch

    def test_retry_batch_skips_column_only_superseded_link(self, manager):
        batch_id = "batch_retrycolumnlink"
        old_id = "job_retrycolumnlink"
        _insert_batch(batch_id, status="failed", total_jobs=1)
        _insert_job(
            old_id,
            status="failed",
            batch_id=batch_id,
            retried_to=old_id,
        )

        with pytest.raises(ValueError, match="No failed or cancelled jobs"):
            manager.retry_batch(batch_id)

        manager._job_manager.retry_job.assert_not_called()
        with get_session_factory()() as session:
            original = session.get(Job, old_id)
            batch = session.get(Batch, batch_id)
            assert original.batch_id == batch_id
            assert original.retried_to == old_id
            assert batch.total_jobs == 1
            assert batch.status == "failed"

    def test_retry_batch_cleans_settings_only_stale_claim_before_retry(self, manager):
        batch_id = "batch_retrystalesettings"
        old_id = "job_retrystalesettings"
        _insert_batch(batch_id, status="failed", total_jobs=1)
        _insert_job(
            old_id,
            status="failed",
            batch_id=batch_id,
            retry_claimed=1,
            retried_to=None,
            settings={
                "retried_to": "job_missing_stale_replacement",
                "superseded": True,
                "previous_batch_id": batch_id,
            },
        )

        with patch.object(manager, "_publish_batch_event"):
            batch, replacements = manager.retry_batch(batch_id)

        assert len(replacements) == 1
        assert replacements[0].batch_id == batch_id
        assert batch.status == "processing"
        with get_session_factory()() as session:
            original = session.get(Job, old_id)
            assert original.retry_claimed == 0
            assert original.batch_id is None
            assert original.retried_to == replacements[0].job_id
            assert session.get(Batch, batch_id).total_jobs == 1

    def test_retry_batch_retries_failed_jobs(self, manager):
        """Retrying a batch creates new jobs for failed children."""
        _insert_batch("batch_aaaaaaaaaaaa", status="partial_failure", total_jobs=2)
        _insert_job(
            "job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa"
        )
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
        _insert_job(
            "job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa"
        )

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

    def test_retry_passes_tenant_id_to_retry_job(self, manager):
        """R4-RC-03: retry_job must be tenant-pinned under scoped calls."""
        _insert_batch(
            "batch_retrytenant1",
            status="partial_failure",
            total_jobs=1,
        )
        # Stamp batch tenant via settings
        factory = get_session_factory()
        session = factory()
        b = session.get(Batch, "batch_retrytenant1")
        b.settings = {"tenant_id": "tenant_aaaaaaaaaaaa"}
        session.commit()
        session.close()
        _insert_job(
            "job_retrytenant1",
            status="failed",
            batch_id="batch_retrytenant1",
            tenant_id="tenant_aaaaaaaaaaaa",
        )

        with patch.object(manager, "_publish_batch_event"):
            manager.retry_batch("batch_retrytenant1", tenant_id="tenant_aaaaaaaaaaaa")

        call = manager._job_manager.retry_job.call_args
        assert call.args[0] == "job_retrytenant1"
        assert call.kwargs.get("tenant_id") == "tenant_aaaaaaaaaaaa"

    def test_retry_does_not_reclone_superseded_originals(self, manager):
        """Second retry must not re-select originals already retried (R4-RC-03)."""
        _insert_batch("batch_noreclone01", status="partial_failure", total_jobs=1)
        _insert_job(
            "job_noreclone01",
            status="failed",
            batch_id="batch_noreclone01",
        )

        with patch.object(manager, "_publish_batch_event"):
            first = manager.retry_batch("batch_noreclone01")
            assert first is not None
            assert len(first[1]) == 1

            with pytest.raises(ValueError, match="No failed or cancelled"):
                manager.retry_batch("batch_noreclone01")

    def test_cancel_passes_tenant_id_to_cancel_job(self, manager):
        """R4-RC-03: cancel_job must be tenant-pinned under scoped calls."""
        _insert_batch("batch_canceltid01", total_jobs=1)
        factory = get_session_factory()
        session = factory()
        b = session.get(Batch, "batch_canceltid01")
        b.settings = {"tenant_id": "tenant_aaaaaaaaaaaa"}
        session.commit()
        session.close()
        _insert_job(
            "job_canceltid01",
            status="processing",
            batch_id="batch_canceltid01",
            tenant_id="tenant_aaaaaaaaaaaa",
        )

        with patch.object(manager, "_publish_batch_event"):
            manager.cancel_batch("batch_canceltid01", tenant_id="tenant_aaaaaaaaaaaa")

        call = manager._job_manager.cancel_job.call_args
        assert call.args[0] == "job_canceltid01"
        assert call.kwargs.get("tenant_id") == "tenant_aaaaaaaaaaaa"


# ---------------------------------------------------------------------------
# Batch completion detection
# ---------------------------------------------------------------------------


class TestBatchCompletion:
    def test_batch_completes_when_all_jobs_terminal(self, manager):
        """Batch status updates to completed when all jobs are terminal."""
        _insert_batch("batch_aaaaaaaaaaaa", status="processing", total_jobs=2)
        _insert_job(
            "job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa"
        )
        _insert_job(
            "job_bbbbbbbbbbbb", status="completed", batch_id="batch_aaaaaaaaaaaa"
        )

        with patch.object(manager, "_publish_batch_event"):
            manager.check_batch_completion("batch_aaaaaaaaaaaa")

        batch = manager.get_batch("batch_aaaaaaaaaaaa")
        assert batch.status == "completed"
        assert batch.completed_at is not None
        assert batch.jobs_completed == 2

    def test_batch_partial_failure(self, manager):
        """Batch is partial_failure when some jobs fail."""
        _insert_batch("batch_aaaaaaaaaaaa", status="processing", total_jobs=2)
        _insert_job(
            "job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa"
        )
        _insert_job("job_bbbbbbbbbbbb", status="failed", batch_id="batch_aaaaaaaaaaaa")

        with patch.object(manager, "_publish_batch_event"):
            manager.check_batch_completion("batch_aaaaaaaaaaaa")

        batch = manager.get_batch("batch_aaaaaaaaaaaa")
        assert batch.status == "partial_failure"
        assert batch.jobs_completed == 1
        assert batch.jobs_failed == 1

    def test_mixed_tenant_batch_is_not_aggregated(self, manager):
        """A foreign child cannot change another tenant's batch counters/status."""
        _insert_batch(
            "batch_completionmixed01",
            status="processing",
            total_jobs=2,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        _insert_job(
            "job_completionmixed01",
            status="completed",
            batch_id="batch_completionmixed01",
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        _insert_job(
            "job_completionmixed02",
            status="failed",
            batch_id="batch_completionmixed01",
            tenant_id="tenant_bbbbbbbbbbbb",
        )

        with patch.object(manager, "_publish_batch_event") as publish:
            manager.check_batch_completion("batch_completionmixed01")

        batch = manager.get_batch("batch_completionmixed01")
        assert batch.status == "processing"
        assert batch.jobs_completed == 0
        assert batch.jobs_failed == 0
        assert batch.completed_at is None
        publish.assert_not_called()

    def test_batch_does_not_complete_while_processing(self, manager):
        """Batch stays processing when some jobs are not terminal."""
        _insert_batch("batch_aaaaaaaaaaaa", status="processing", total_jobs=2)
        _insert_job(
            "job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa"
        )
        _insert_job(
            "job_bbbbbbbbbbbb", status="processing", batch_id="batch_aaaaaaaaaaaa"
        )

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
        _insert_job(
            "job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa"
        )

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
        _insert_job(
            "job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa"
        )

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

    def test_get_batch_jobs_tenant_scope_excludes_mixed_child(self, manager):
        """A scoped batch response must not include a foreign child row."""
        _insert_batch(
            "batch_mixedtenant01",
            total_jobs=2,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        _insert_job(
            "job_mixedtenant01",
            batch_id="batch_mixedtenant01",
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        _insert_job(
            "job_mixedtenant02",
            batch_id="batch_mixedtenant01",
            tenant_id="tenant_bbbbbbbbbbbb",
        )

        jobs = manager.get_batch_jobs(
            "batch_mixedtenant01", tenant_id="tenant_aaaaaaaaaaaa"
        )

        assert [job.job_id for job in jobs] == ["job_mixedtenant01"]

    def test_scoped_cancel_does_not_cancel_with_foreign_active_child(self, manager):
        """A tenant cannot finalize a mixed batch around another tenant's job."""
        _insert_batch(
            "batch_cancelmixed01",
            status="processing",
            total_jobs=2,
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        _insert_job(
            "job_cancelmixed01",
            status="submitted",
            batch_id="batch_cancelmixed01",
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        _insert_job(
            "job_cancelmixed02",
            status="processing",
            batch_id="batch_cancelmixed01",
            tenant_id="tenant_bbbbbbbbbbbb",
        )

        with patch.object(manager, "_publish_batch_event"):
            batch = manager.cancel_batch(
                "batch_cancelmixed01", tenant_id="tenant_aaaaaaaaaaaa"
            )

        assert batch is not None
        assert batch.status == "processing"

    def test_submit_persists_durable_tenant_id(self, manager):
        """RC-03: submit stamps batches.tenant_id column, not only settings."""
        from api.tenant_manager import create_tenant

        tenant = create_tenant(name="batch-admission-test")
        with patch.object(manager, "_publish_batch_event"):
            batch, _jobs = manager.submit_batch(
                files=[("a.pdf", b"x")],
                tenant_id=tenant.tenant_id,
            )
        assert batch.tenant_id == tenant.tenant_id
        assert batch.settings.get("tenant_id") == tenant.tenant_id
        # Reload from DB
        reloaded = manager.get_batch(batch.batch_id)
        assert reloaded is not None
        assert reloaded.tenant_id == tenant.tenant_id

    def test_get_batch_refuses_any_child_admit_for_unstamped(self, manager):
        """RC-03 residual: unstamped batch is not visible via child job tenant."""
        _insert_batch("batch_unstamped01")  # no tenant_id column / settings
        _insert_job(
            "job_childtenant01",
            batch_id="batch_unstamped01",
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        # Scoped call must NOT admit via any-child ownership
        assert (
            manager.get_batch("batch_unstamped01", tenant_id="tenant_aaaaaaaaaaaa")
            is None
        )
        # Unscoped (platform) still sees it
        assert manager.get_batch("batch_unstamped01") is not None

    def test_get_batch_column_tenant_match(self, manager):
        _insert_batch("batch_coltenant01", tenant_id="tenant_aaaaaaaaaaaa")
        assert (
            manager.get_batch("batch_coltenant01", tenant_id="tenant_aaaaaaaaaaaa")
            is not None
        )
        assert (
            manager.get_batch("batch_coltenant01", tenant_id="tenant_bbbbbbbbbbbb")
            is None
        )

    def test_list_batches_no_any_child_admit(self, manager):
        _insert_batch("batch_listchild01")  # unstamped
        _insert_job(
            "job_listchild01",
            batch_id="batch_listchild01",
            tenant_id="tenant_aaaaaaaaaaaa",
        )
        batches, total = manager.list_batches(tenant_id="tenant_aaaaaaaaaaaa")
        ids = {b.batch_id for b in batches}
        assert "batch_listchild01" not in ids
