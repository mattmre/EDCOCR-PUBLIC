"""Tests for GPU scaling support in coordinator signals.

Validates the VRAM-based concurrency heuristic, GPU detection fallback,
and worker-ready logging for multi-GPU deployments.
"""

import logging
from unittest.mock import MagicMock, patch

from django.test import TestCase

from jobs.signals import _detect_gpu, recommended_concurrency


class TestRecommendedConcurrencyFormula(TestCase):
    """Validate the recommended_concurrency() heuristic across VRAM values."""

    def test_recommended_concurrency_formula(self):
        """Test the formula for several representative VRAM values.

        Heuristic: vram_mb // 4096, clamped to [1, 24].
        """
        test_cases = [
            (8192, 2),    # 8 GB
            (16384, 4),   # 16 GB
            (24576, 6),   # 24 GB
            (49152, 12),  # 48 GB
            (81920, 20),  # 80 GB (A100)
        ]
        for vram_mb, expected in test_cases:
            with self.subTest(vram_mb=vram_mb):
                result = recommended_concurrency(vram_mb)
                self.assertEqual(
                    result, expected,
                    f"recommended_concurrency({vram_mb}) = {result}, expected {expected}",
                )

    def test_recommended_concurrency_cap_at_24(self):
        """Very large VRAM (200000 MB) should be capped at 24."""
        result = recommended_concurrency(200000)
        self.assertEqual(result, 24)

    def test_recommended_concurrency_min_1(self):
        """Small VRAM (4096 MB) should yield minimum concurrency of 1."""
        result = recommended_concurrency(4096)
        self.assertEqual(result, 1)


class TestGpuDetectionWithoutPaddle(TestCase):
    """Validate GPU detection gracefully handles missing paddle."""

    def test_gpu_detection_without_paddle(self):
        """When paddle import fails, _detect_gpu should return (False, '', 0)."""
        # Force paddle import to raise ImportError inside _detect_gpu
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "paddle":
                raise ImportError("No module named 'paddle'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _detect_gpu()

        self.assertEqual(result, (False, "", 0))


class TestWorkerReadyLogsConcurrency(TestCase):
    """Validate that on_worker_ready logs recommended concurrency for GPU workers."""

    @staticmethod
    def _make_queue_mock(queue_name):
        """Create a mock queue object with a .name attribute."""
        q = MagicMock()
        q.name = queue_name
        return q

    @patch("jobs.signals.socket.gethostname", return_value="test-gpu-scaling-worker")
    @patch("jobs.signals._detect_cpu", return_value=(16, 32768))
    @patch("jobs.signals._detect_gpu", return_value=(True, "NVIDIA RTX 4090", 24576))
    def test_worker_ready_logs_concurrency(self, mock_gpu, mock_cpu, mock_hostname):
        """When GPU is detected with VRAM, on_worker_ready should log
        a message containing 'recommended concurrency'.
        """
        from jobs.signals import on_worker_ready

        sender = MagicMock()
        sender.consumer.queues = [
            self._make_queue_mock("ocr_gpu"),
            self._make_queue_mock("cpu_general"),
        ]

        with self.assertLogs("jobs.signals", level=logging.INFO) as log_ctx:
            on_worker_ready(sender=sender)

        # Find the log message that contains the recommended concurrency info
        concurrency_logged = any(
            "recommended concurrency" in msg.lower()
            for msg in log_ctx.output
        )
        self.assertTrue(
            concurrency_logged,
            f"Expected 'recommended concurrency' in log output, got: {log_ctx.output}",
        )
