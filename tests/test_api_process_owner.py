"""Owned process fixtures for the standalone API's startup/recovery boundary."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Timer
from unittest.mock import Mock, patch

import pytest

from api.process_owner import ApiProcessOwner, undrained_owners


def test_process_owner_rejects_overlap_and_releases_after_crash(tmp_path):
    database = str(tmp_path / "jobs.db")
    script = (
        "from api.process_owner import ApiProcessOwner; import sys,time; "
        "owner=ApiProcessOwner(sys.argv[1]); owner.acquire(); "
        "print('owned',flush=True); time.sleep(30)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, database], stdout=subprocess.PIPE,
        text=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    try:
        assert child.stdout.readline().strip() == "owned"
        with pytest.raises(RuntimeError, match="already owned"):
            ApiProcessOwner(database).acquire()
    finally:
        child.kill()
        child.wait(timeout=5)
        child.stdout.close()
    after = ApiProcessOwner(database)
    after.acquire()
    after.release()
    assert after.path.exists()  # unlinking would permit two independent locks


def test_owner_refuses_linked_lock_file(tmp_path):
    owner = ApiProcessOwner(str(tmp_path / "jobs.db"))
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    owner.path.symlink_to(target)
    with pytest.raises((OSError, RuntimeError)):
        owner.acquire()
    assert target.read_text() == "preserve"


def test_app_construction_cannot_recover_active_jobs():
    from api.main import create_app

    with patch("api.job_manager.recover_interrupted_jobs") as recover:
        create_app()
    recover.assert_not_called()


@pytest.mark.parametrize("drained", [True, False])
def test_lifespan_owns_store_before_recovery_and_through_shutdown(drained):
    from api import database
    from api.main import _lifespan, create_app

    manager = Mock()
    manager.shutdown.return_value = drained
    app = create_app()

    async def exercise():
        with (
            patch("api.job_manager.get_manager", return_value=manager),
            patch("api.job_manager.recover_interrupted_jobs") as recover,
            patch("api.webhooks.shutdown_webhook_pool"),
        ):
            async with _lifespan(app):
                recover.assert_called_once_with()
                async with _expect_blocked_lifespan(app):
                    pass
                assert recover.call_count == 1
            candidate = ApiProcessOwner(database._engine_db_path)
            if drained:
                candidate.acquire()
                candidate.release()
            else:
                with pytest.raises(RuntimeError, match="already owned"):
                    candidate.acquire()

    try:
        asyncio.run(exercise())
    finally:
        # The test mock owns no real writers; release only its retained fixture.
        for owner in list(undrained_owners):
            owner.release()
            undrained_owners.remove(owner)


@asynccontextmanager
async def _expect_blocked_lifespan(app):
    from api.main import _lifespan

    with pytest.raises(RuntimeError, match="already owned"):
        async with _lifespan(app):
            pytest.fail("second API lifespan entered")
    yield


def test_recovery_failure_releases_ownership_without_starting_workers():
    from api import database
    from api.main import _lifespan, create_app

    app = create_app()

    async def exercise():
        with (
            patch("api.job_manager.get_manager") as manager,
            patch("api.job_manager.recover_interrupted_jobs", side_effect=PermissionError),
            patch("api.webhooks.shutdown_webhook_pool"),
        ):
            with pytest.raises(RuntimeError, match="PID fences"):
                async with _lifespan(app):
                    pytest.fail("unknown recovery state must prevent serving")
            manager.assert_not_called()
        candidate = ApiProcessOwner(database._engine_db_path)
        candidate.acquire()
        candidate.release()

    asyncio.run(exercise())


def test_repeated_shutdown_does_not_report_live_writer_as_drained():
    from api.database import get_session_factory
    from api.job_manager import JobManager

    manager = JobManager(get_session_factory())
    worker = Mock()
    worker.is_alive.return_value = True
    manager._worker_threads = [worker]
    assert manager.shutdown(timeout=0) is False
    assert manager.shutdown(timeout=0) is False
    worker.is_alive.return_value = False
    assert manager.shutdown(timeout=0) is True


def test_background_writer_can_finish_within_configured_shutdown_budget(monkeypatch):
    from api import database
    from api.background_work import start_writer
    from api.main import _lifespan, create_app

    monkeypatch.setenv("JOB_MANAGER_SHUTDOWN_TIMEOUT", "3")
    release = Event()
    manager = Mock()
    manager.shutdown.return_value = True

    async def exercise():
        with (
            patch("api.job_manager.get_manager", return_value=manager),
            patch("api.job_manager.recover_interrupted_jobs"),
            patch("api.webhooks.shutdown_webhook_pool"),
        ):
            async with _lifespan(create_app()):
                writer = start_writer(lambda: release.wait(5))
                timer = Timer(1.2, release.set)
                timer.start()
            timer.join(timeout=5)
            writer.join(timeout=5)
        candidate = ApiProcessOwner(database._engine_db_path)
        candidate.acquire()
        candidate.release()

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        for owner in list(undrained_owners):
            owner.release()
            undrained_owners.remove(owner)


def test_blocked_webhook_respects_lifespan_budget_and_retains_owner(monkeypatch):
    from api import database
    from api.main import _lifespan, create_app
    from api.webhooks import _submit_bounded, shutdown_webhook_pool

    monkeypatch.setenv("JOB_MANAGER_SHUTDOWN_TIMEOUT", "0.05")
    entered, release = Event(), Event()
    manager = Mock()
    manager.shutdown.return_value = True

    def callback():
        entered.set()
        release.wait(5)

    async def exercise():
        with (
            patch("api.job_manager.get_manager", return_value=manager),
            patch("api.job_manager.recover_interrupted_jobs"),
        ):
            start = time.monotonic()
            async with _lifespan(create_app()):
                _submit_bounded(callback, "owned-shutdown-fixture")
                assert entered.wait(1)
            assert time.monotonic() - start < 2
            assert not release.is_set()
            with pytest.raises(RuntimeError, match="already owned"):
                ApiProcessOwner(database._engine_db_path).acquire()

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        shutdown_webhook_pool()
        for owner in list(undrained_owners):
            owner.release()
            undrained_owners.remove(owner)
