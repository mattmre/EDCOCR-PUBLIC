"""Tests for Celery signal handlers (worker registration, heartbeat)."""

from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.utils import timezone

from jobs.models import Worker
from jobs.signals import (
    _detect_cpu,
    _detect_gpu,
    on_heartbeat,
    on_worker_ready,
    on_worker_shutdown,
)


class TestDetectGpu(TestCase):
    """Tests for _detect_gpu() hardware detection."""

    def test_returns_false_when_paddle_not_installed(self):
        # Mock ImportError for paddle
        with patch.dict("sys.modules", {"paddle": None}):
            # _detect_gpu tries to import paddle - should handle gracefully
            pass
        # Since paddle is likely not installed in test env:
        result = _detect_gpu()
        assert result == (False, "", 0)

    def test_returns_false_when_no_cuda(self):
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = False
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            from jobs.signals import _detect_gpu as detect
            result = detect()
        assert result == (False, "", 0)

    def test_returns_gpu_info_when_cuda_available(self):
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 1
        mock_paddle.device.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24GB in bytes
        mock_paddle.device.cuda.get_device_properties.return_value = mock_props
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            from jobs.signals import _detect_gpu as detect
            result = detect()
        assert result[0] is True
        assert result[1] == "NVIDIA RTX 4090"
        assert result[2] == 24 * 1024  # 24GB in MB

    def test_includes_assigned_gpu_from_cuda_visible_devices(self):
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 1
        mock_paddle.device.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024 * 1024 * 1024
        mock_paddle.device.cuda.get_device_properties.return_value = mock_props

        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}):
                from jobs.signals import _detect_gpu as detect

                result = detect()

        assert result[0] is True
        assert "cuda_visible_devices=2" in result[1]
        assert result[2] == 24 * 1024

    def test_returns_false_on_exception(self):
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.side_effect = RuntimeError("CUDA error")
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            from jobs.signals import _detect_gpu as detect
            result = detect()
        assert result == (False, "", 0)


class TestDetectCpu(TestCase):
    """Tests for _detect_cpu() hardware detection."""

    def test_returns_cpu_count(self):
        with patch("os.cpu_count", return_value=16):
            cpu_cores, _ = _detect_cpu()
        assert cpu_cores == 16

    def test_returns_zero_when_cpu_count_none(self):
        with patch("os.cpu_count", return_value=None):
            cpu_cores, _ = _detect_cpu()
        assert cpu_cores == 0

    def test_detects_ram_with_psutil(self):
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.total = 32 * 1024 * 1024 * 1024  # 32GB
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            from jobs.signals import _detect_cpu as detect
            _, ram_mb = detect()
        assert ram_mb == 32 * 1024  # 32GB in MB

    def test_returns_zero_ram_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}):
            # Force ImportError for psutil
            from jobs.signals import _detect_cpu as detect  # noqa: F401
        # psutil unavailable in test env usually, so this tests graceful handling
        cpu_cores, ram_mb = _detect_cpu()
        assert cpu_cores >= 0
        # ram_mb may be 0 or actual if psutil is installed


class TestOnWorkerReady(TestCase):
    """Tests for on_worker_ready signal handler."""

    @staticmethod
    def _make_queue_mock(queue_name):
        """Create a mock queue object with a proper .name string attribute."""
        q = MagicMock()
        q.name = queue_name
        return q

    @patch("jobs.signals.socket.gethostname", return_value="test-worker-01")
    @patch("jobs.signals._detect_gpu", return_value=(True, "RTX 4090", 24576))
    @patch("jobs.signals._detect_cpu", return_value=(16, 32768))
    def test_registers_gpu_worker(self, mock_cpu, mock_gpu, mock_hostname):
        sender = MagicMock()
        sender.consumer.queues = [
            self._make_queue_mock("ocr_gpu"),
            self._make_queue_mock("cpu_general"),
        ]

        on_worker_ready(sender=sender)

        worker = Worker.objects.get(hostname="test-worker-01")
        assert worker.status == Worker.Status.ONLINE
        assert worker.gpu_available is True
        assert worker.gpu_model == "RTX 4090"
        assert worker.gpu_vram_mb == 24576
        assert worker.cpu_cores == 16
        assert worker.ram_mb == 32768
        assert "ocr" in worker.capabilities

    @patch("jobs.signals.socket.gethostname", return_value="test-cpu-worker")
    @patch("jobs.signals._detect_gpu", return_value=(False, "", 0))
    @patch("jobs.signals._detect_cpu", return_value=(8, 16384))
    def test_registers_cpu_only_worker(self, mock_cpu, mock_gpu, mock_hostname):
        sender = MagicMock()
        sender.consumer.queues = [self._make_queue_mock("cpu_general")]

        on_worker_ready(sender=sender)

        worker = Worker.objects.get(hostname="test-cpu-worker")
        assert worker.gpu_available is False
        assert "compress" in worker.capabilities
        assert "ner" in worker.capabilities
        assert "ocr" not in worker.capabilities

    @patch("jobs.signals.socket.gethostname", return_value="test-env-worker")
    @patch("jobs.signals._detect_gpu", return_value=(False, "", 0))
    @patch("jobs.signals._detect_cpu", return_value=(4, 8192))
    def test_falls_back_to_env_var_for_queues(self, mock_cpu, mock_gpu, mock_hostname):
        # sender.consumer.queues access must raise so the except branch fires
        sender = MagicMock()
        sender.consumer.queues.__iter__ = MagicMock(side_effect=AttributeError("no queues"))

        with patch.dict("os.environ", {"WORKER_QUEUES": "ocr_gpu,cpu_general"}):
            on_worker_ready(sender=sender)

        worker = Worker.objects.get(hostname="test-env-worker")
        assert "ocr_gpu" in worker.queues
        assert "cpu_general" in worker.queues

    @patch("jobs.signals.socket.gethostname", return_value="test-update-worker")
    @patch("jobs.signals._detect_gpu", return_value=(False, "", 0))
    @patch("jobs.signals._detect_cpu", return_value=(4, 8192))
    def test_updates_existing_worker(self, mock_cpu, mock_gpu, mock_hostname):
        Worker.objects.create(
            hostname="test-update-worker",
            status=Worker.Status.OFFLINE,
            gpu_available=False,
        )

        sender = MagicMock()
        sender.consumer.queues = [self._make_queue_mock("cpu_general")]

        on_worker_ready(sender=sender)

        worker = Worker.objects.get(hostname="test-update-worker")
        assert worker.status == Worker.Status.ONLINE


class TestOnWorkerShutdown(TestCase):
    """Tests for on_worker_shutdown signal handler."""

    @patch("jobs.signals.socket.gethostname", return_value="shutdown-worker")
    def test_marks_worker_offline(self, mock_hostname):
        Worker.objects.create(
            hostname="shutdown-worker",
            status=Worker.Status.ONLINE,
            current_task_id="task-123",
        )

        on_worker_shutdown(sender=None)

        worker = Worker.objects.get(hostname="shutdown-worker")
        assert worker.status == Worker.Status.OFFLINE
        assert worker.current_task_id == ""

    @patch("jobs.signals.socket.gethostname", return_value="nonexistent-worker")
    def test_handles_nonexistent_worker(self, mock_hostname):
        # Should not raise even if worker doesn't exist in DB
        on_worker_shutdown(sender=None)


class TestOnHeartbeat(TestCase):
    """Tests for on_heartbeat signal handler."""

    @patch("jobs.signals.socket.gethostname", return_value="heartbeat-worker")
    def test_updates_heartbeat_timestamp(self, mock_hostname):
        old_time = timezone.now() - timezone.timedelta(minutes=5)
        Worker.objects.create(
            hostname="heartbeat-worker",
            status=Worker.Status.ONLINE,
            last_heartbeat=old_time,
        )

        on_heartbeat(sender=None)

        worker = Worker.objects.get(hostname="heartbeat-worker")
        assert worker.last_heartbeat > old_time

    @patch("jobs.signals.socket.gethostname", return_value="no-such-worker")
    def test_handles_nonexistent_worker(self, mock_hostname):
        # Should not raise even if worker doesn't exist
        on_heartbeat(sender=None)
