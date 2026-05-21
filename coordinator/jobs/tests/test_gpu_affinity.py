"""Tests for GPU queue affinity feature (Phase 7A).

Tests cover:
- _extract_gpu_index() — CUDA_VISIBLE_DEVICES env var parsing
- _route_task() — Celery task routing with per-GPU queues
- Worker.gpu_index field — model field and registration integration
- Capability inference — per-GPU queue names map to OCR capability

Run with:
    cd coordinator && python -m pytest jobs/tests/test_gpu_affinity.py -v
"""

import itertools
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from jobs.models import Worker
from jobs.signals import _extract_gpu_index
from version import __version__

# ===========================================================================
# _extract_gpu_index() tests
# ===========================================================================


class TestExtractGpuIndex(TestCase):
    """Tests for _extract_gpu_index() CUDA_VISIBLE_DEVICES parsing."""

    def test_single_gpu_index(self):
        """CUDA_VISIBLE_DEVICES='2' should return 2."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}):
            result = _extract_gpu_index()
        assert result == 2

    def test_first_of_multiple(self):
        """CUDA_VISIBLE_DEVICES='1,3' should return the first index (1)."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1,3"}):
            result = _extract_gpu_index()
        assert result == 1

    def test_zero_index(self):
        """CUDA_VISIBLE_DEVICES='0' should return 0 (not None)."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            result = _extract_gpu_index()
        assert result == 0

    def test_empty_returns_none(self):
        """CUDA_VISIBLE_DEVICES='' should return None."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            result = _extract_gpu_index()
        assert result is None

    def test_all_returns_none(self):
        """CUDA_VISIBLE_DEVICES='all' should return None."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "all"}):
            result = _extract_gpu_index()
        assert result is None

    def test_none_returns_none(self):
        """CUDA_VISIBLE_DEVICES='none' should return None."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "none"}):
            result = _extract_gpu_index()
        assert result is None

    def test_unset_returns_none(self):
        """When CUDA_VISIBLE_DEVICES is not set, should return None."""
        env = {"CUDA_VISIBLE_DEVICES": "SENTINEL"}
        with patch.dict("os.environ", env):
            del env["CUDA_VISIBLE_DEVICES"]
            # Ensure the env var is truly absent
            import os
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            result = _extract_gpu_index()
        assert result is None

    def test_non_numeric_returns_none(self):
        """CUDA_VISIBLE_DEVICES='GPU-UUID' should return None."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-UUID"}):
            result = _extract_gpu_index()
        assert result is None


# ===========================================================================
# _route_task() tests
# ===========================================================================


class TestRouteTask(TestCase):
    """Tests for _route_task() Celery task routing with per-GPU queues."""

    def test_gpu_task_default_shared_queue(self):
        """When per-GPU queues are disabled, process_page routes to 'ocr_gpu'."""
        import coordinator.celery as celery_module

        with patch.object(celery_module, '_PER_GPU_QUEUES_ENABLED', False):
            result = celery_module._route_task(
                'jobs.tasks.process_page', None, None, None,
            )
        assert result == {'queue': 'ocr_gpu'}

    def test_gpu_task_per_gpu_round_robin(self):
        """When per-GPU queues are enabled, GPU tasks distribute round-robin."""
        import coordinator.celery as celery_module

        with patch.object(celery_module, '_PER_GPU_QUEUES_ENABLED', True), \
             patch.object(celery_module, '_GPU_COUNT', 4), \
             patch.object(celery_module, '_gpu_round_robin_counter', itertools.count()):

            result0 = celery_module._route_task(
                'jobs.tasks.process_page', None, None, None,
            )
            result1 = celery_module._route_task(
                'jobs.tasks.process_page', None, None, None,
            )
            result2 = celery_module._route_task(
                'jobs.tasks.process_document', None, None, None,
            )
            result3 = celery_module._route_task(
                'jobs.tasks.process_page', None, None, None,
            )
            # After 4 tasks, wraps around
            result4 = celery_module._route_task(
                'jobs.tasks.process_page', None, None, None,
            )

        assert result0 == {'queue': 'ocr_gpu_0'}
        assert result1 == {'queue': 'ocr_gpu_1'}
        assert result2 == {'queue': 'ocr_gpu_2'}
        assert result3 == {'queue': 'ocr_gpu_3'}
        # Wraps around to 0
        assert result4 == {'queue': 'ocr_gpu_0'}

    def test_cpu_task_routes_unchanged(self):
        """compress_pdf always routes to 'cpu_general' regardless of per-GPU setting."""
        import coordinator.celery as celery_module

        # Test with per-GPU enabled -- should not affect CPU tasks
        with patch.object(celery_module, '_PER_GPU_QUEUES_ENABLED', True), \
             patch.object(celery_module, '_GPU_COUNT', 4):
            result = celery_module._route_task(
                'jobs.tasks.compress_pdf', None, None, None,
            )
        assert result == {'queue': 'cpu_general'}

    def test_coordinator_task_routes_unchanged(self):
        """ingest_document always routes to 'coordinator' regardless of per-GPU setting."""
        import coordinator.celery as celery_module

        with patch.object(celery_module, '_PER_GPU_QUEUES_ENABLED', True), \
             patch.object(celery_module, '_GPU_COUNT', 4):
            result = celery_module._route_task(
                'jobs.tasks.ingest_document', None, None, None,
            )
        assert result == {'queue': 'coordinator'}

    def test_unknown_task_returns_none(self):
        """Unknown task names should return None (default routing)."""
        import coordinator.celery as celery_module

        result = celery_module._route_task(
            'some.unknown.task', None, None, None,
        )
        assert result is None

    def test_process_document_also_routes_to_gpu(self):
        """process_document task should also route to GPU queues."""
        import coordinator.celery as celery_module

        with patch.object(celery_module, '_PER_GPU_QUEUES_ENABLED', False):
            result = celery_module._route_task(
                'jobs.tasks.process_document', None, None, None,
            )
        assert result == {'queue': 'ocr_gpu'}

    def test_extract_entities_routes_to_cpu(self):
        """extract_entities should always route to cpu_general."""
        import coordinator.celery as celery_module

        result = celery_module._route_task(
            'jobs.tasks.extract_entities', None, None, None,
        )
        assert result == {'queue': 'cpu_general'}


# ===========================================================================
# Worker.gpu_index model field tests
# ===========================================================================


class TestWorkerGpuIndex(TestCase):
    """Tests for the Worker model gpu_index field and registration."""

    def test_worker_model_gpu_index_default_none(self):
        """New Worker should have gpu_index=None by default."""
        worker = Worker.objects.create(hostname="test-default-gpuidx")
        assert worker.gpu_index is None

    def test_worker_model_gpu_index_stored(self):
        """Worker.objects.create(gpu_index=2) should persist correctly."""
        Worker.objects.create(hostname="test-gpuidx-store", gpu_index=2)
        worker = Worker.objects.get(hostname="test-gpuidx-store")
        assert worker.gpu_index == 2

    def test_worker_model_gpu_index_zero(self):
        """gpu_index=0 should be stored as 0, not confused with None."""
        Worker.objects.create(hostname="test-gpuidx-zero", gpu_index=0)
        worker = Worker.objects.get(hostname="test-gpuidx-zero")
        assert worker.gpu_index == 0
        assert worker.gpu_index is not None

    def test_register_worker_saves_gpu_index(self):
        """register_worker(gpu_index=1) should persist gpu_index on the Worker row."""
        from jobs.tasks import register_worker

        register_worker(
            hostname="test-register-gpuidx",
            queues=["ocr_gpu_1"],
            capabilities=["ocr"],
            gpu_available=True,
            gpu_model="RTX 4090",
            gpu_vram_mb=24576,
            gpu_index=1,
            cpu_cores=16,
            ram_mb=32768,
            pipeline_version=__version__,
        )

        worker = Worker.objects.get(hostname="test-register-gpuidx")
        assert worker.gpu_index == 1
        assert worker.gpu_available is True
        assert worker.status == Worker.Status.ONLINE

    def test_register_worker_gpu_index_none(self):
        """register_worker without gpu_index should leave it as None."""
        from jobs.tasks import register_worker

        register_worker(
            hostname="test-register-no-gpuidx",
            queues=["cpu_general"],
            capabilities=["compress", "ner"],
        )

        worker = Worker.objects.get(hostname="test-register-no-gpuidx")
        assert worker.gpu_index is None

    def test_register_worker_updates_gpu_index(self):
        """Calling register_worker twice should update gpu_index."""
        from jobs.tasks import register_worker

        register_worker(
            hostname="test-update-gpuidx",
            queues=["ocr_gpu_0"],
            capabilities=["ocr"],
            gpu_index=0,
        )
        worker = Worker.objects.get(hostname="test-update-gpuidx")
        assert worker.gpu_index == 0

        # Re-register with different gpu_index
        register_worker(
            hostname="test-update-gpuidx",
            queues=["ocr_gpu_2"],
            capabilities=["ocr"],
            gpu_index=2,
        )
        worker.refresh_from_db()
        assert worker.gpu_index == 2


# ===========================================================================
# Capability inference from per-GPU queue names
# ===========================================================================


class TestCapabilityInference(TestCase):
    """Tests for capability inference from per-GPU queue names in signals."""

    @staticmethod
    def _make_queue_mock(queue_name):
        """Create a mock queue object with a .name attribute."""
        q = MagicMock()
        q.name = queue_name
        return q

    @patch("jobs.signals.socket.gethostname", return_value="test-pergpu-cap")
    @patch("jobs.signals._detect_gpu", return_value=(True, "RTX 4090", 24576))
    @patch("jobs.signals._detect_cpu", return_value=(16, 32768))
    @patch("jobs.signals._extract_gpu_index", return_value=0)
    def test_per_gpu_queue_infers_ocr_capability(
        self, mock_gpuidx, mock_cpu, mock_gpu, mock_hostname,
    ):
        """queues=['ocr_gpu_0'] should produce capabilities=['ocr']."""
        from jobs.signals import on_worker_ready

        sender = MagicMock()
        sender.consumer.queues = [self._make_queue_mock("ocr_gpu_0")]

        on_worker_ready(sender=sender)

        worker = Worker.objects.get(hostname="test-pergpu-cap")
        assert "ocr" in worker.capabilities

    @patch("jobs.signals.socket.gethostname", return_value="test-shared-cap")
    @patch("jobs.signals._detect_gpu", return_value=(True, "RTX 4090", 24576))
    @patch("jobs.signals._detect_cpu", return_value=(16, 32768))
    @patch("jobs.signals._extract_gpu_index", return_value=None)
    def test_shared_queue_infers_ocr_capability(
        self, mock_gpuidx, mock_cpu, mock_gpu, mock_hostname,
    ):
        """queues=['ocr_gpu'] should produce capabilities=['ocr']."""
        from jobs.signals import on_worker_ready

        sender = MagicMock()
        sender.consumer.queues = [self._make_queue_mock("ocr_gpu")]

        on_worker_ready(sender=sender)

        worker = Worker.objects.get(hostname="test-shared-cap")
        assert "ocr" in worker.capabilities

    @patch("jobs.signals.socket.gethostname", return_value="test-multi-cap")
    @patch("jobs.signals._detect_gpu", return_value=(True, "RTX 4090", 24576))
    @patch("jobs.signals._detect_cpu", return_value=(16, 32768))
    @patch("jobs.signals._extract_gpu_index", return_value=1)
    def test_per_gpu_queue_with_cpu_infers_both_capabilities(
        self, mock_gpuidx, mock_cpu, mock_gpu, mock_hostname,
    ):
        """queues=['ocr_gpu_1', 'cpu_general'] should produce both ocr and compress/ner."""
        from jobs.signals import on_worker_ready

        sender = MagicMock()
        sender.consumer.queues = [
            self._make_queue_mock("ocr_gpu_1"),
            self._make_queue_mock("cpu_general"),
        ]

        on_worker_ready(sender=sender)

        worker = Worker.objects.get(hostname="test-multi-cap")
        assert "ocr" in worker.capabilities
        assert "compress" in worker.capabilities
        assert "ner" in worker.capabilities

    @patch("jobs.signals.socket.gethostname", return_value="test-nocap")
    @patch("jobs.signals._detect_gpu", return_value=(False, "", 0))
    @patch("jobs.signals._detect_cpu", return_value=(4, 8192))
    @patch("jobs.signals._extract_gpu_index", return_value=None)
    def test_unrelated_queue_no_ocr_capability(
        self, mock_gpuidx, mock_cpu, mock_gpu, mock_hostname,
    ):
        """queues=['cpu_general'] should not produce 'ocr' capability."""
        from jobs.signals import on_worker_ready

        sender = MagicMock()
        sender.consumer.queues = [self._make_queue_mock("cpu_general")]

        on_worker_ready(sender=sender)

        worker = Worker.objects.get(hostname="test-nocap")
        assert "ocr" not in worker.capabilities
        assert "compress" in worker.capabilities
