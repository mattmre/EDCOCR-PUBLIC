"""Tests for api/job_manager.py -- direct unit tests for JobManager methods."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from subprocess import TimeoutExpired
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from api.database import Job, UsageRecord, get_session_factory
from api.job_manager import (
    JobManager,
    _write_pipeline_heartbeat,
    recover_interrupted_jobs,
)

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
        mock_config.MAX_QUEUED_JOBS = 64
        mock_config.MAX_UPLOAD_SIZE_MB = 1
        mock_config.WEBHOOK_TIMEOUT = 30
        mock_config.WEBHOOK_MAX_RETRIES = 0
        mock_config.WEBHOOK_SECRET = ""

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


def _write_completed_artifacts(output_dir: Path) -> None:
    """Create the minimum durable artifact pair required for API success."""
    pdf_dir = output_dir / "EXPORT" / "PDF"
    text_dir = output_dir / "EXPORT" / "TEXT"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / "result.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (text_dir / "result.txt").write_text("recognized text", encoding="utf-8")


@pytest.mark.skipif(sys.platform != "linux", reason="restart recovery scans Linux /proc")
class TestRestartPidRecovery:
    @staticmethod
    def _proc_scope(*pids: int):
        """Expose real /proc entries for test children, excluding host services."""
        real_scandir = os.scandir

        def scoped_scandir(path):
            if str(path) != "/proc":
                return real_scandir(path)
            with real_scandir(path) as entries:
                selected = [entry for entry in entries if entry.name in {str(p) for p in pids}]
            return nullcontext(selected)

        return patch("api.job_manager.os.scandir", side_effect=scoped_scandir)

    @staticmethod
    def _child(job_id: str | None = None) -> subprocess.Popen:
        environment = os.environ.copy()
        environment.pop("OCR_JOB_ID", None)
        if job_id is not None:
            environment["OCR_JOB_ID"] = job_id
        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=environment,
            start_new_session=True,
        )

    @staticmethod
    def _cleanup(child: subprocess.Popen) -> None:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)

    def test_cancelled_spawning_fence_without_child_is_released(self, manager):
        job_id = _insert_job("job_restart_nochild", status="cancelled", pid=-1)
        with self._proc_scope():
            assert recover_interrupted_jobs(get_session_factory()) == 1
        job = manager.get_job(job_id)
        assert job.status == "cancelled"
        assert job.pid is None

    def test_unrelated_live_pid_is_never_signalled(self, manager):
        child = self._child()
        try:
            job_id = _insert_job("job_restart_reused", status="cancelled", pid=child.pid)
            with self._proc_scope(child.pid):
                assert recover_interrupted_jobs(get_session_factory()) == 1
            assert manager.get_job(job_id).pid is None
            assert child.poll() is None
        finally:
            self._cleanup(child)

    @pytest.mark.parametrize("persisted_pid", ["actual", -1])
    def test_owned_child_stops_before_releasing_fence(self, manager, persisted_pid):
        job_id = f"job_restart_owned_{persisted_pid}"
        child = self._child(job_id)
        try:
            _insert_job(
                job_id,
                status="cancelled",
                pid=child.pid if persisted_pid == "actual" else -1,
            )
            with self._proc_scope(child.pid):
                assert recover_interrupted_jobs(get_session_factory()) == 1
            assert child.wait(timeout=5) != 0
            assert manager.get_job(job_id).pid is None
        finally:
            self._cleanup(child)

    def test_unknown_process_state_retains_fence(self, manager):
        job_id = _insert_job("job_restart_unknown", status="cancelled", pid=-1)
        with patch(
            "api.job_manager._orphaned_pipeline_pids",
            side_effect=PermissionError("/proc unavailable"),
        ):
            with pytest.raises(RuntimeError, match="refusing to serve stale job state"):
                recover_interrupted_jobs(get_session_factory())
        assert manager.get_job(job_id).pid == -1

    def test_cancel_releases_stale_unrelated_pid(self, manager):
        child = self._child()
        try:
            job_id = _insert_job("job_cancel_stalepid", status="processing", pid=child.pid)
            with (
                self._proc_scope(child.pid),
                patch.object(manager, "_fire_webhook"),
                patch.object(manager, "_publish_job_event"),
            ):
                result = manager.cancel_job(job_id)
            assert result.status == "cancelled"
            assert manager.get_job(job_id).pid is None
            assert child.poll() is None
        finally:
            self._cleanup(child)

    def test_real_runner_cancel_stops_child_before_retry(self, manager, tmp_path):
        job_id = _insert_job("job_cancel_real_child")
        source_dir = tmp_path / "source" / job_id
        output_dir = tmp_path / "output" / job_id
        source_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        (source_dir / "test.pdf").write_bytes(
            (Path(__file__).parent / "fixtures" / "sample.pdf").read_bytes()
        )
        pipeline = tmp_path / "sleep_pipeline.py"
        pipeline.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

        with (
            patch.dict(os.environ, {"HEALTHCHECK_FILE": str(tmp_path / "heartbeat")}),
            patch("api.job_manager.config.PIPELINE_SCRIPT", str(pipeline)),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
            patch.object(manager, "_ensure_workers"),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            runner = executor.submit(
                manager._run_pipeline,
                job_id,
                str(source_dir),
                str(output_dir),
                {},
            )
            child = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    with manager._pipeline_condition:
                        child = manager._active_pipeline_processes.get(job_id)
                    if child is not None and child.poll() is None:
                        break
                    time.sleep(0.01)
                else:
                    pytest.fail("real pipeline child did not start")

                assert manager.cancel_job(job_id).status == "cancelled"
                runner.result(timeout=5)
                assert child.poll() is not None
                assert manager.get_job(job_id).pid is None
                assert manager.retry_job(job_id) is not None
            finally:
                if child is not None and child.poll() is None:
                    self._cleanup(child)


# ---------------------------------------------------------------------------
# _run_pipeline Tests
# ---------------------------------------------------------------------------


def test_write_pipeline_heartbeat_refreshes_configured_file(tmp_path):
    heartbeat = tmp_path / "health" / "pipeline-heartbeat"
    with patch.dict(os.environ, {"HEALTHCHECK_FILE": str(heartbeat)}):
        before = time.time()
        _write_pipeline_heartbeat()

    assert heartbeat.is_file()
    assert float(heartbeat.read_text(encoding="utf-8")) >= before - 1


class TestRunPipeline:
    """Tests for JobManager._run_pipeline()."""

    def test_verbose_real_child_can_complete_without_pipe_backpressure(
        self, manager, tmp_path, capfd
    ):
        job_id = _insert_job("job_abcdefabcdef")
        output_dir = tmp_path / "output" / job_id
        output_dir.mkdir(parents=True)
        pipeline = tmp_path / "verbose_pipeline.py"
        pipeline.write_text(
            "import pathlib, sys\n"
            "sys.stdout.buffer.write(b'x' * (1024 * 1024))\n"
            "sys.stdout.buffer.flush()\n"
            "root = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "for kind, name, data in [('PDF', 'result.pdf', b'%PDF-1.4\\n%%EOF\\n'), "
            "('TEXT', 'result.txt', b'recognized text')]:\n"
            "    target = root / 'EXPORT' / kind / name\n"
            "    target.parent.mkdir(parents=True, exist_ok=True)\n"
            "    target.write_bytes(data)\n",
            encoding="utf-8",
        )

        with (
            patch("api.job_manager.config.PIPELINE_SCRIPT", str(pipeline)),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            runner = pool.submit(
                manager._run_pipeline,
                job_id,
                str(tmp_path / "source"),
                str(output_dir),
                {},
            )
            try:
                runner.result(timeout=5)
            finally:
                if not runner.done():
                    manager.cancel_job(job_id)
                    runner.result(timeout=5)

        assert manager.get_job(job_id).status == "completed"
        assert (output_dir / "EXPORT" / "PDF" / "result.pdf").is_file()
        assert len(capfd.readouterr().out) >= 1024 * 1024

    def test_pipeline_uses_the_running_python_interpreter(self, manager, tmp_path):
        """The API image may expose python3 but not a bare ``python`` binary."""
        job_id = _insert_job("job_pythoninterp")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with (
            patch(
                "api.job_manager.subprocess.Popen", return_value=mock_proc
            ) as mock_popen,
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        assert mock_popen.call_args.args[0][0] == sys.executable

    def test_pipeline_spawn_refreshes_heartbeat_before_first_poll(
        self, manager, tmp_path
    ):
        job_id = _insert_job("job_spawn_heartbeat")
        mock_proc = MagicMock(pid=12345, returncode=0)
        mock_proc.wait.return_value = 0

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch("api.job_manager._write_pipeline_heartbeat") as heartbeat,
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        heartbeat.assert_called_once_with()

    def test_progress_monitor_refreshes_heartbeat_only_while_child_is_live(
        self, manager, tmp_path
    ):
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]

        with (
            patch("api.job_manager._write_pipeline_heartbeat") as heartbeat,
            patch("api.job_manager.time.sleep"),
        ):
            manager._monitor_progress("missing-job", str(tmp_path), mock_proc)

        heartbeat.assert_called_once_with()

    def test_pipeline_success_sets_completed(self, manager, tmp_path):
        """Successful pipeline sets status to completed."""
        job_id = _insert_job("job_aaaaaaaaaaaa")
        source_dir = str(tmp_path / "source")
        output_dir = str(tmp_path / "output")
        _write_completed_artifacts(Path(output_dir))

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, source_dir, output_dir, {})

        job = manager.get_job(job_id)
        assert job.status == "completed"
        assert job.pid is None
        assert job.completed_at is not None
        assert job.processing_time is not None

    def test_cancel_waits_for_spawn_and_stops_child_before_return(
        self, manager, tmp_path
    ):
        job_id = _insert_job("job_spawn_cancel_race")
        spawn_entered, release_spawn, killed = Event(), Event(), Event()

        class Child:
            pid = 84201
            returncode = None

            def wait(self, timeout=None):
                assert killed.wait(timeout=5)
                self.returncode = -9
                return self.returncode

            def poll(self):
                return self.returncode

            def kill(self):
                killed.set()

        child = Child()

        def spawn(*args, **kwargs):
            spawn_entered.set()
            assert release_spawn.wait(timeout=5)
            return child

        with (
            patch("api.job_manager.subprocess.Popen", side_effect=spawn),
            patch("api.job_manager._write_pipeline_heartbeat"),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_publish_job_event"),
            patch.object(manager, "_fire_webhook"),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            runner = pool.submit(
                manager._run_pipeline, job_id, str(tmp_path), str(tmp_path), {}
            )
            assert spawn_entered.wait(timeout=5)
            canceller = pool.submit(manager.cancel_job, job_id)
            assert not canceller.done()
            release_spawn.set()
            assert canceller.result(timeout=5).status == "cancelled"
            assert killed.is_set()
            runner.result(timeout=5)

        assert manager.get_job(job_id).status == "cancelled"

    def test_other_manager_cancel_waits_for_persisted_spawn_fence(
        self, manager, tmp_path
    ):
        job_id = _insert_job("job_remote_spawn_race")
        other_manager = JobManager(get_session_factory())
        spawn_entered, release_spawn, killed = Event(), Event(), Event()

        class Child:
            pid = 84203

            def kill(self):
                killed.set()

            def wait(self, timeout=None):
                assert killed.wait(timeout=5)
                return -9

        def spawn(*args, **kwargs):
            spawn_entered.set()
            assert release_spawn.wait(timeout=5)
            return Child()

        with (
            patch("api.job_manager.subprocess.Popen", side_effect=spawn),
            patch.object(manager, "_publish_job_event"),
            patch.object(other_manager, "_publish_job_event"),
            patch.object(other_manager, "_fire_webhook"),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            runner = pool.submit(
                manager._run_pipeline, job_id, str(tmp_path), str(tmp_path), {}
            )
            assert spawn_entered.wait(timeout=5)
            canceller = pool.submit(other_manager.cancel_job, job_id)
            deadline = time.monotonic() + 5
            while manager.get_job(job_id).status != "cancelled":
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert not canceller.done()
            release_spawn.set()
            assert canceller.result(timeout=5).status == "cancelled"
            runner.result(timeout=5)

        assert killed.is_set()
        assert manager.get_job(job_id).pid is None

    def test_terminal_cancel_wins_over_stale_success_without_side_effects(
        self, manager, tmp_path
    ):
        job_id = _insert_job(
            "job_terminal_cancel_race", tenant_id="race-tenant", total_pages=2
        )
        entered, release = Event(), Event()
        child = MagicMock(pid=84202, returncode=0)
        child.wait.return_value = 0

        def artifacts(_path):
            entered.set()
            assert release.wait(timeout=5)
            return True

        with (
            patch("api.job_manager.subprocess.Popen", return_value=child),
            patch("api.job_manager._has_completed_artifacts", side_effect=artifacts),
            patch("api.job_manager._write_pipeline_heartbeat"),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_publish_job_event") as publish,
            patch.object(manager, "_fire_webhook") as webhook,
            patch("api.quota.consume_page_quota") as quota,
            patch("api.usage.record_processing_seconds") as usage,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            runner = pool.submit(
                manager._run_pipeline, job_id, str(tmp_path), str(tmp_path), {}
            )
            assert entered.wait(timeout=5)
            assert manager.cancel_job(job_id).status == "cancelled"
            release.set()
            runner.result(timeout=5)

        assert manager.get_job(job_id).status == "cancelled"
        assert all(call.args[1] != "job.completed" for call in publish.call_args_list)
        assert webhook.call_count == 1
        quota.assert_not_called()
        usage.assert_not_called()

    def test_monitor_cannot_publish_progress_after_cancel(self, manager, tmp_path):
        job_id = _insert_job(
            "job_progress_cancel_race",
            status="processing",
            total_pages=3,
            pages_completed=0,
        )
        entered, release = Event(), Event()
        child = MagicMock()
        child.poll.return_value = None

        def pause(_seconds):
            entered.set()
            assert release.wait(timeout=5)

        with (
            patch("api.job_manager.time.sleep", side_effect=pause),
            patch("api.job_manager._has_completed_artifacts", return_value=True),
            patch("api.job_manager._write_pipeline_heartbeat"),
            patch.object(manager, "_publish_job_event") as publish,
            patch.object(manager, "_fire_webhook"),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            monitor = pool.submit(manager._monitor_progress, job_id, str(tmp_path), child)
            assert entered.wait(timeout=5)
            assert manager.cancel_job(job_id).status == "cancelled"
            release.set()
            monitor.result(timeout=5)

        job = manager.get_job(job_id)
        assert job.status == "cancelled"
        assert job.pages_completed == 0
        assert all(call.args[1] != "job.progress" for call in publish.call_args_list)

    def test_zero_exit_without_artifacts_sets_failed(self, manager, tmp_path):
        """A zero subprocess exit cannot create a false-success API job."""
        job_id = _insert_job("job_noartifacts1")
        output_dir = tmp_path / "empty-output"
        output_dir.mkdir()
        mock_proc = MagicMock(pid=12345, returncode=0)
        mock_proc.wait.return_value = 0

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, str(tmp_path), str(output_dir), {})

        job = manager.get_job(job_id)
        assert job.status == "failed"
        assert (
            "without complete pdf and text artifacts"
            in (job.error_message or "").lower()
        )

    def test_pipeline_failure_sets_failed(self, manager, tmp_path):
        """Non-zero exit code sets status to failed."""
        job_id = _insert_job("job_bbbbbbbbbbbb")
        source_dir = str(tmp_path / "source")
        output_dir = str(tmp_path / "output")

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, source_dir, output_dir, {})

        job = manager.get_job(job_id)
        assert job.status == "failed"
        assert "exited with code 1" in job.error_message.lower()

    def test_pipeline_exception_sets_failed(self, manager, tmp_path):
        """Exception during subprocess launch sets status to failed."""
        job_id = _insert_job("job_cccccccccccc")
        source_dir = str(tmp_path / "source")
        output_dir = str(tmp_path / "output")

        with (
            patch(
                "api.job_manager.subprocess.Popen", side_effect=OSError("spawn failed")
            ),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
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

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        assert 42 in pid_captured

    def test_pipeline_docintel_adds_flags(self, manager, tmp_path):
        """DocIntel settings add --enable-docintel and --docintel-mode flags."""
        job_id = _insert_job("job_eeeeeeeeeeee")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with (
            patch(
                "api.job_manager.subprocess.Popen", return_value=mock_proc
            ) as mock_popen,
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(
                job_id,
                str(tmp_path),
                str(tmp_path),
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

        with (
            patch(
                "api.job_manager.subprocess.Popen", return_value=mock_proc
            ) as mock_popen,
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(
                job_id,
                str(tmp_path),
                str(tmp_path),
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

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
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

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
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

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
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

    def test_pipeline_starts_a_new_session_on_posix(self, manager, tmp_path):
        """A real pipeline must have a killable process-group boundary."""
        job_id = _insert_job("job_newsession1")

        mock_proc = MagicMock()
        mock_proc.pid = 12346
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc) as popen,
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        if os.name != "nt":
            assert popen.call_args.kwargs["start_new_session"] is True
        assert popen.call_args.kwargs["env"]["OCR_JOB_ID"] == job_id

    def test_pipeline_clears_pid_after_completion(self, manager, tmp_path):
        """PID is cleared to None after pipeline finishes (success or failure)."""
        job_id = _insert_job("job_pidcleartest")

        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
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

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook") as mock_webhook,
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        mock_webhook.assert_called_once()

    def test_pipeline_fires_webhook_on_failure(self, manager, tmp_path):
        """Pipeline calls _fire_webhook after failed completion."""
        job_id = _insert_job("job_webhookfail")

        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.returncode = 2
        mock_proc.wait.return_value = 2

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook") as mock_webhook,
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        mock_webhook.assert_called_once()

    def test_pipeline_fires_webhook_on_exception(self, manager, tmp_path):
        """Pipeline calls _fire_webhook even when an exception occurs."""
        job_id = _insert_job("job_webhookexc1")

        with (
            patch("api.job_manager.subprocess.Popen", side_effect=RuntimeError("boom")),
            patch.object(manager, "_fire_webhook") as mock_webhook,
            patch.object(manager, "_publish_job_event"),
        ):
            manager._run_pipeline(job_id, str(tmp_path), str(tmp_path), {})

        mock_webhook.assert_called_once()

    def test_pipeline_publishes_processing_and_completed_events(
        self, manager, tmp_path
    ):
        """Pipeline emits durable events at processing start and successful completion."""
        job_id = _insert_job("job_eventsuccess")
        _write_completed_artifacts(tmp_path)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event") as mock_publish,
        ):
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

        with (
            patch("api.job_manager.subprocess.Popen", return_value=mock_proc),
            patch.object(manager, "_monitor_progress"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
            patch("api.job_manager.datetime") as mock_datetime,
        ):
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

        with (
            patch("api.job_manager.os.kill") as mock_kill,
            patch("api.job_manager._pid_belongs_to_job", return_value=True),
            patch.object(manager, "_await_cancelled_writer"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            result = manager.cancel_job(job_id)

        assert result.status == "cancelled"
        # A remote runner clears the PID only after it has reaped its child.
        assert result.pid == 99999
        mock_kill.assert_called_once()
        # Verify signal argument -- SIGKILL on Unix, 9 fallback on Windows
        call_args = mock_kill.call_args[0]
        assert call_args[0] == 99999

    def test_cancel_with_pid_swallows_oserror(self, manager):
        """Cancel handles OSError from os.kill gracefully."""
        job_id = _insert_job("job_111111111111", status="processing", pid=99999)

        with (
            patch("api.job_manager.os.kill", side_effect=OSError("No such process")),
            patch("api.job_manager._pid_belongs_to_job", return_value=True),
            patch.object(manager, "_await_cancelled_writer"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
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

    def test_repeat_cancel_stops_writer_left_by_failed_first_attempt(self, manager):
        job_id = _insert_job("job_cancel_retry_stop", status="cancelled")
        child = MagicMock()
        child.poll.return_value = None
        with manager._pipeline_condition:
            manager._active_pipeline_processes[job_id] = child
        try:
            with (
                patch.object(manager, "_publish_job_event") as publish,
                patch.object(manager, "_fire_webhook") as webhook,
            ):
                result = manager.cancel_job(job_id)
        finally:
            with manager._pipeline_condition:
                manager._active_pipeline_processes.pop(job_id)
        assert result.status == "cancelled"
        child.kill.assert_called_once()
        child.wait.assert_called_once()
        publish.assert_not_called()
        webhook.assert_not_called()

    def test_cancel_nonexistent_returns_none(self, manager):
        """Cancel of nonexistent job returns None."""
        result = manager.cancel_job("job_000000000000")
        assert result is None

    def test_cancel_foreign_tenant_does_not_sigkill(self, manager):
        job_id = _insert_job(
            "job_tenantkill1",
            status="processing",
            pid=424242,
            tenant_id="tenant-a",
        )
        with (
            patch("api.job_manager._kill_pipeline_pid") as mock_kill,
            patch("api.job_manager._pid_belongs_to_job", return_value=True),
            patch.object(manager, "_await_cancelled_writer"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            missing = manager.cancel_job(job_id, tenant_id="tenant-b")
            owned = manager.cancel_job(job_id, tenant_id="tenant-a")
        assert missing is None
        assert owned.status == "cancelled"
        mock_kill.assert_called_once_with(424242)

    def test_cancel_submitted_job_without_pid(self, manager):
        """Cancel of submitted job (no PID yet) transitions to cancelled."""
        job_id = _insert_job("job_nopidcancel", status="submitted")

        with (
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event"),
        ):
            result = manager.cancel_job(job_id)

        assert result.status == "cancelled"
        assert result.completed_at is not None

    def test_retry_rejects_cancelled_job_while_old_writer_is_live(
        self, manager, tmp_path
    ):
        job_id = _insert_job("job_retry_live_writer", status="cancelled")
        source = Path(manager.get_job(job_id).source_file)
        assert source.name == "test.pdf"
        source_dir = tmp_path / "source" / job_id
        source_dir.mkdir()
        (source_dir / "test.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        child = MagicMock()
        child.poll.return_value = None
        with manager._pipeline_condition:
            manager._active_pipeline_processes[job_id] = child
        try:
            with pytest.raises(ValueError, match="writer has not stopped"):
                manager.retry_job(job_id)
        finally:
            with manager._pipeline_condition:
                manager._active_pipeline_processes.pop(job_id)

    def test_cancel_fires_webhook(self, manager):
        """Cancel fires webhook after status transition."""
        job_id = _insert_job("job_cancelwebhk", status="processing", pid=99999)

        with (
            patch("api.job_manager.os.kill"),
            patch.object(manager, "_fire_webhook") as mock_webhook,
            patch.object(manager, "_publish_job_event"),
        ):
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

        with (
            patch("api.job_manager.os.kill"),
            patch.object(manager, "_fire_webhook"),
            patch.object(manager, "_publish_job_event") as mock_publish,
        ):
            manager.cancel_job(job_id)

        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[1] == "job.cancelled"


class TestSubmitJob:
    """Tests for submitted-job event publication."""

    def test_submit_publishes_submitted_event(self, manager, tmp_path):
        """Submitting a job emits a durable submitted event after commit."""
        fake_thread = MagicMock()

        with (
            patch("api.job_manager.threading.Thread", return_value=fake_thread),
            patch.object(manager, "_publish_job_event") as mock_publish,
        ):
            # S3-11: page estimate fail-closed needs a real PDF parse
            sample = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
            job = manager.submit(
                upload_filename="input.pdf",
                upload_content=sample.read_bytes(),
            )

        assert job.status == "submitted"
        assert job.source_hash == hashlib.sha256(sample.read_bytes()).hexdigest()
        assert fake_thread.start.call_count == (
            manager._priority_queue._max_concurrent + 1
        )  # workers plus the durable batch reconciler
        mock_publish.assert_called_once()
        assert mock_publish.call_args.args[1] == "job.submitted"

    def test_stream_upload_is_copied_in_bounded_reads(self, manager, tmp_path):
        """Upload streams never call read() without a chunk size (R20-A10)."""
        from api.job_manager import _copy_upload_stream

        class ChunkOnlyBytesIO(io.BytesIO):
            def read(self, size=-1):
                assert size > 0
                return super().read(size)

        destination = tmp_path / "streamed.bin"
        _copy_upload_stream(ChunkOnlyBytesIO(b"x" * 2048), destination)
        assert destination.read_bytes() == b"x" * 2048

    def test_stream_upload_enforces_size_before_admit(self, manager, tmp_path):
        """Oversized streams are rejected and their partial file is removed."""
        from api.job_manager import _copy_upload_stream

        destination = tmp_path / "too-large.bin"
        with pytest.raises(ValueError, match="Upload exceeds"):
            _copy_upload_stream(io.BytesIO(b"x" * (1024 * 1024 + 1)), destination)
        assert not destination.exists()

    def test_rejected_upload_cleans_staging_directories(self, manager):
        """A pre-admit rejection must not accumulate empty job directories."""
        with pytest.raises(ValueError, match="Upload exceeds"):
            manager.submit(
                upload_filename="oversized.pdf",
                upload_content=b"x" * (1024 * 1024 + 1),
            )

        from api import job_manager as job_manager_module

        assert not list(Path(job_manager_module.config.SOURCE_FOLDER).iterdir())
        # The admission lock is control state, never uploaded tenant bytes.
        assert {
            p.name for p in Path(job_manager_module.config.OUTPUT_FOLDER).iterdir()
        } <= {".admit_capacity.lock"}


# ---------------------------------------------------------------------------
# get_result_artifacts Tests
# ---------------------------------------------------------------------------


class TestGetResultArtifacts:
    """Tests for get_result_artifacts()."""

    def test_returns_all_artifact_types(self, manager, tmp_path):
        """Discovers PDF, TEXT, and STRUCTURE artifacts."""
        job_id = "job_333333333333"
        result_dir = tmp_path / "output" / job_id
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
            job_id,
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
        job_id = "job_555555555555"
        result_dir = tmp_path / "output" / job_id
        pdf_dir = result_dir / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"%PDF-1.0")

        job_id = _insert_job(
            job_id,
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

    def test_symlink_job_root_is_not_inventoried(self, manager, tmp_path):
        victim = tmp_path / "victim"
        (victim / "EXPORT" / "PDF").mkdir(parents=True)
        (victim / "EXPORT" / "PDF" / "secret.pdf").write_bytes(b"%PDF-1.0")
        lexical = tmp_path / "job_a"
        lexical.symlink_to(victim)
        job_id = _insert_job(
            "job_symlinkroot",
            status="completed",
            result_path=str(lexical),
        )
        assert manager.get_result_artifacts(job_id) == {}


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
        """Queue at MAX_QUEUED_JOBS returns False."""
        # Partial mock config falls back to MAX_CONCURRENT_JOBS=64.
        # Insert 64 active jobs.
        for i in range(64):
            _insert_job(f"job_full_{i:04d}", status="submitted")

        assert manager.check_queue_capacity() is False

    def test_queue_capacity_can_exceed_pipeline_concurrency(self, manager, monkeypatch):
        """Bounded pending work does not increase inference concurrency."""
        from api import job_manager

        monkeypatch.setattr(job_manager.config, "MAX_CONCURRENT_JOBS", 1)
        monkeypatch.setattr(job_manager.config, "MAX_QUEUED_JOBS", 3)
        _insert_job("job_pending_1", status="processing")
        _insert_job("job_pending_2", status="submitted")

        assert manager.check_queue_capacity() is True

        _insert_job("job_pending_3", status="submitted")
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
# Graceful Shutdown Tests (maintenance review)
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Tests for JobManager.shutdown() and non-daemon worker threads."""

    def test_worker_threads_are_non_daemon(self, manager):
        """Worker threads must be non-daemon so in-flight jobs are not killed
        when the interpreter exits (maintenance review)."""
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


def test_staging_holds_tenant_lifecycle_fence_until_rejection_cleanup(manager):
    """A purge snapshot cannot pass an upload that has not got a Job row yet."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from api import job_manager as module
    from api.tenant_manager import _admission_serialization

    staged = Event()
    release = Event()
    attempting = Event()
    fenced = Event()

    def blocked_hash(path):
        assert path.exists()
        staged.set()
        assert release.wait(timeout=5)
        raise OSError("injected staging rejection")

    def snapshot():
        attempting.set()
        with _admission_serialization():
            fenced.set()
            assert not list(Path(module.config.SOURCE_FOLDER).iterdir())

    with patch("api.job_manager._compute_source_hash", side_effect=blocked_hash):
        with ThreadPoolExecutor(max_workers=2) as pool:
            upload = pool.submit(
                manager.submit, upload_filename="a.pdf", upload_content=b"fixture"
            )
            assert staged.wait(timeout=5)
            purge = pool.submit(snapshot)
            assert attempting.wait(timeout=5)
            try:
                assert not fenced.wait(timeout=0.2)
            finally:
                release.set()
            with pytest.raises(ValueError, match="hashed"):
                upload.result(timeout=5)
            purge.result(timeout=5)
    assert fenced.is_set()


def test_retry_admission_atomically_links_original_and_is_replay_safe(manager):
    """A request lost after INSERT cannot create a second retry on replay."""
    from api import job_manager as module

    old_id = "job_retry_commit"
    _insert_job(old_id, status="failed", batch_id="batch_retry_commit")
    sample = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
    old_source = Path(module.config.SOURCE_FOLDER) / old_id
    old_source.mkdir()
    (old_source / "input.pdf").write_bytes(sample.read_bytes())
    with (
        patch.object(manager, "_ensure_workers"),
        patch.object(manager, "_publish_job_event"),
    ):
        first = manager.retry_job(old_id, batch_id="batch_retry_commit")
        second = manager.retry_job(old_id, batch_id="batch_retry_commit")
    assert first.job_id == second.job_id
    with manager._session_factory() as session:
        original = session.get(Job, old_id)
        assert original.retried_to == first.job_id
        assert original.batch_id is None
        assert original.retry_claimed == 0
        assert session.query(Job).count() == 2


def test_batch_retry_admission_keeps_claim_until_batch_manager_releases_it(manager):
    """A fast completion cannot clear the whole-request batch retry fence."""
    from api import job_manager as module

    old_id = "job_retry_claimed"
    batch_id = "batch_retry_claimed"
    _insert_job(
        old_id,
        status="failed",
        batch_id=batch_id,
        retry_claimed=1,
    )
    sample = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
    old_source = Path(module.config.SOURCE_FOLDER) / old_id
    old_source.mkdir()
    (old_source / "input.pdf").write_bytes(sample.read_bytes())

    with (
        patch.object(manager, "_ensure_workers"),
        patch.object(manager, "_publish_job_event"),
    ):
        replacement = manager.retry_job(old_id, batch_id=batch_id)

    with manager._session_factory() as session:
        original = session.get(Job, old_id)
        admitted = session.get(Job, replacement.job_id)
        assert original.retried_to == replacement.job_id
        assert original.batch_id == batch_id
        assert original.retry_claimed == 1
        assert admitted.batch_id == batch_id


def test_individual_retry_cannot_overwrite_a_concurrent_batch_claim(manager):
    """A stale individual retry read must not create an orphan replacement."""
    from api import job_manager as module

    old_id = "job_retry_claim_race"
    batch_id = "batch_retry_claim_race"
    _insert_job(old_id, status="failed", batch_id=batch_id)
    sample = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
    old_source = Path(module.config.SOURCE_FOLDER) / old_id
    old_source.mkdir()
    (old_source / "input.pdf").write_bytes(sample.read_bytes())

    original_submit = manager.submit

    def claim_then_submit(**kwargs):
        with manager._session_factory() as session:
            original = session.get(Job, old_id)
            original.retry_claimed = 1
            session.commit()
        return original_submit(**kwargs)

    manager.submit = claim_then_submit
    with (
        patch.object(manager, "_ensure_workers"),
        patch.object(manager, "_publish_job_event"),
        pytest.raises(ValueError, match="batch retry is already in progress"),
    ):
        manager.retry_job(old_id)

    with manager._session_factory() as session:
        original = session.get(Job, old_id)
        assert original.batch_id == batch_id
        assert original.retry_claimed == 1
        assert original.retried_to is None
        assert session.query(Job).count() == 1


def test_individual_retry_holds_shared_fence_through_reconciliation(manager, monkeypatch):
    """Recovery and finalization cannot pass an individual retry's first read."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from api import job_manager as module
    from api.batch_manager import BatchManager
    from api.database import Batch

    old_id = "job_retry_reconcile_fence"
    batch_id = "batch_retry_reconcile_fence"
    factory = manager._session_factory
    _insert_job(
        old_id,
        status="failed",
        batch_id=batch_id,
        retry_claimed=1,
    )
    with factory() as session:
        session.add(Batch(batch_id=batch_id, status="processing", total_jobs=1))
        session.commit()

    source_dir = Path(module.config.SOURCE_FOLDER) / old_id
    source_dir.mkdir()
    sample = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
    (source_dir / "input.pdf").write_bytes(sample.read_bytes())

    real_thread_lock = module._BATCH_RETRY_THREAD_LOCK
    attempting_lock = Event()
    first_retry_read = Event()
    retry_started = Event()

    class ObservedLock:
        def __enter__(self):
            attempting_lock.set()
            real_thread_lock.acquire()

        def __exit__(self, *_exc):
            real_thread_lock.release()

    monkeypatch.setattr(module, "_BATCH_RETRY_THREAD_LOCK", ObservedLock())

    def tracked_factory():
        first_retry_read.set()
        return factory()

    manager._session_factory = tracked_factory
    batch_manager = BatchManager(factory, job_manager=manager)

    def retry_request():
        retry_started.set()
        return manager.retry_job(old_id)

    with (
        patch.object(manager, "_ensure_workers"),
        patch.object(manager, "_publish_job_event"),
        patch.object(batch_manager, "_publish_batch_event"),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        future = None
        with module._batch_retry_serialization_lock():
            # Ignore this main-thread acquire; observe the waiting request.
            attempting_lock.clear()
            future = pool.submit(retry_request)
            assert retry_started.wait(timeout=5)
            assert attempting_lock.wait(timeout=5)

            # The reconciler holds the same fence in production. While it
            # recovers the abandoned claim and finalizes, the retry must not
            # reach its first ORM read or submit a child into that interval.
            assert not first_retry_read.wait(timeout=0.1)
            assert batch_manager.recover_abandoned_retry_claims(batch_id) == 1
            batch_manager.check_batch_completion(batch_id)
            with factory() as session:
                assert session.get(Batch, batch_id).status == "failed"
            assert not first_retry_read.is_set()

        replacement = future.result(timeout=10)

    assert first_retry_read.is_set()
    assert replacement.batch_id is None
    with factory() as session:
        assert session.get(Batch, batch_id).status == "failed"


def test_retry_replay_rejects_linked_replacement_for_another_tenant(manager):
    from api import job_manager as module
    from api.tenant_manager import create_tenant

    old_id = "job_retry_tenant_link"
    replacement_id = "job_retry_foreign_tenant_link"
    source_tenant = create_tenant(name="retry-source-tenant")
    replacement_tenant = create_tenant(name="retry-replacement-tenant")
    _insert_job(
        old_id,
        status="failed",
        tenant_id=source_tenant.tenant_id,
        retried_to=replacement_id,
    )
    _insert_job(
        replacement_id,
        status="submitted",
        tenant_id=replacement_tenant.tenant_id,
    )
    source_dir = Path(module.config.SOURCE_FOLDER) / old_id
    source_dir.mkdir()
    sample = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
    (source_dir / "input.pdf").write_bytes(sample.read_bytes())

    with pytest.raises(ValueError, match="tenant does not match"):
        manager.retry_job(old_id, tenant_id=source_tenant.tenant_id)


@pytest.mark.parametrize("link_shape", ["self", "cycle"])
def test_retry_replay_rejects_self_or_superseded_replacement_link(
    manager, link_shape
):
    from api import job_manager as module

    old_id = f"job_retry_bad_link_{link_shape}"
    replacement_id = old_id if link_shape == "self" else f"job_retry_child_{link_shape}"
    _insert_job(old_id, status="failed", retried_to=replacement_id)
    if link_shape == "cycle":
        _insert_job(
            replacement_id,
            status="submitted",
            retried_to=old_id,
        )
    source_dir = Path(module.config.SOURCE_FOLDER) / old_id
    source_dir.mkdir()
    sample = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
    (source_dir / "input.pdf").write_bytes(sample.read_bytes())

    with pytest.raises(ValueError, match="self-referential or superseded"):
        manager.retry_job(old_id)
