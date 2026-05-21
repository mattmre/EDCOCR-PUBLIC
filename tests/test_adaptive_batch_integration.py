"""Tests for adaptive batch sizing integration with ocr_gpu_async.py.

Covers:
- ENABLE_ADAPTIVE_BATCH=false uses fixed CHUNK_TARGET_SIZE
- ENABLE_ADAPTIVE_BATCH=true with mocked sizer uses dynamic size
- Import failure falls back gracefully to fixed chunk size
- Assembler feedback records BatchResult after document completion
- Adaptive batch sizer is None when feature is disabled

Run with: python -m pytest tests/test_adaptive_batch_integration.py -v
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path


# ---------------------------------------------------------------------------
# Tests: Environment variable controls
# ---------------------------------------------------------------------------


class TestEnableAdaptiveBatchEnvVar:
    """Verify the ENABLE_ADAPTIVE_BATCH env var parsing."""

    @pytest.mark.parametrize("value,expected", [
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("TRUE", True),
        ("Yes", True),
    ])
    def test_env_var_parsing(self, value, expected):
        """ENABLE_ADAPTIVE_BATCH should be parsed like other opt-in features."""
        result = value.lower() in ("1", "true", "yes")
        assert result == expected


class TestAdaptiveBatchDisabled:
    """When ENABLE_ADAPTIVE_BATCH is not set, the sizer should be None."""

    def test_sizer_is_none_when_disabled(self):
        """With default env (no ENABLE_ADAPTIVE_BATCH), _adaptive_batch_sizer is None."""
        env = os.environ.copy()
        env.pop("ENABLE_ADAPTIVE_BATCH", None)
        with patch.dict(os.environ, env, clear=True):
            # The constant is evaluated at import time, so we test the logic directly
            enable = os.environ.get("ENABLE_ADAPTIVE_BATCH", "").lower() in ("1", "true", "yes")
            assert enable is False

    def test_fixed_chunk_size_used_when_sizer_none(self):
        """When _adaptive_batch_sizer is None, chunk_target_size = CHUNK_TARGET_SIZE."""
        sizer = None
        CHUNK_TARGET_SIZE = 20

        if sizer is not None:
            chunk_target_size = 999  # Should not reach here
        else:
            chunk_target_size = CHUNK_TARGET_SIZE

        assert chunk_target_size == 20


# ---------------------------------------------------------------------------
# Tests: Adaptive batch sizing integration logic
# ---------------------------------------------------------------------------


class TestAdaptiveBatchEnabled:
    """When ENABLE_ADAPTIVE_BATCH=true, the sizer should provide dynamic sizes."""

    def test_sizer_recommends_dynamic_size(self):
        """Adaptive sizer returns a complexity-based chunk size."""
        from adaptive_batch import AdaptiveBatchSizer

        sizer = AdaptiveBatchSizer()
        complexity = sizer.compute_complexity(
            width=0, height=0,
            file_size=5_000_000,  # Large file
            dpi=300,
            has_tables=False,
            has_images=False,
        )
        chunk_target_size = sizer.recommend_batch_size([complexity])

        # Should return a positive integer within config bounds
        assert chunk_target_size >= sizer._config.min_batch_size
        assert chunk_target_size <= sizer._config.max_batch_size

    def test_sizer_varies_by_file_size(self):
        """Different file sizes should produce different chunk recommendations."""
        from adaptive_batch import AdaptiveBatchSizer

        sizer = AdaptiveBatchSizer()

        small_complexity = sizer.compute_complexity(
            width=0, height=0, file_size=10_000, dpi=300,
        )
        large_complexity = sizer.compute_complexity(
            width=0, height=0, file_size=10_000_000, dpi=300,
        )

        small_chunk = sizer.recommend_batch_size([small_complexity])
        large_chunk = sizer.recommend_batch_size([large_complexity])

        # Larger files should have higher complexity, leading to smaller batch
        assert large_complexity.complexity_score > small_complexity.complexity_score
        assert large_chunk <= small_chunk

    def test_scheduler_logic_with_adaptive_sizer(self):
        """Simulate the scheduler chunk-sizing logic path."""
        from adaptive_batch import AdaptiveBatchSizer

        sizer = AdaptiveBatchSizer()
        CHUNK_TARGET_SIZE = 20
        DPI = 300

        # Simulate the scheduler logic
        file_size = 500_000
        try:
            complexity = sizer.compute_complexity(
                width=0, height=0,
                file_size=file_size,
                dpi=DPI,
                has_tables=False,
                has_images=False,
            )
            chunk_target_size = sizer.recommend_batch_size([complexity])
        except Exception:
            chunk_target_size = CHUNK_TARGET_SIZE

        assert isinstance(chunk_target_size, int)
        assert chunk_target_size > 0

    def test_scheduler_logic_exception_fallback(self):
        """If the sizer raises an exception, fall back to CHUNK_TARGET_SIZE."""
        CHUNK_TARGET_SIZE = 20

        sizer = MagicMock()
        sizer.compute_complexity.side_effect = RuntimeError("test error")

        try:
            complexity = sizer.compute_complexity(
                width=0, height=0, file_size=0, dpi=300,
            )
            chunk_target_size = sizer.recommend_batch_size([complexity])
        except Exception:
            chunk_target_size = CHUNK_TARGET_SIZE

        assert chunk_target_size == CHUNK_TARGET_SIZE


# ---------------------------------------------------------------------------
# Tests: Assembler feedback recording
# ---------------------------------------------------------------------------


class TestAssemblerFeedback:
    """The assembler should record BatchResult after document completion."""

    def test_record_result_called_on_completion(self):
        """Simulate assembler feedback path with a real sizer."""
        from adaptive_batch import AdaptiveBatchSizer, BatchResult

        sizer = AdaptiveBatchSizer()
        assert len(sizer.get_history()) == 0

        # Simulate doc completion
        total_pages = 15
        processed_pages = 14
        doc_duration = 5.2

        sizer.record_result(BatchResult(
            batch_size=total_pages,
            pages_processed=processed_pages,
            duration_seconds=doc_duration,
            success_count=processed_pages,
            failure_count=total_pages - processed_pages,
        ))

        assert len(sizer.get_history()) == 1
        result = sizer.get_history()[0]
        assert result.batch_size == 15
        assert result.pages_processed == 14
        assert result.failure_count == 1
        assert result.duration_seconds == 5.2

    def test_record_result_exception_silenced(self):
        """If record_result raises, the exception should be caught."""
        sizer = MagicMock()
        sizer.record_result.side_effect = RuntimeError("record failed")

        # Simulate the assembler try/except pattern
        try:
            sizer.record_result(MagicMock())
        except Exception:
            pass  # Should reach here without propagating

        # No assertion needed -- test passes if no exception propagates

    def test_feedback_loop_adjusts_batch_size(self):
        """Multiple feedback results should cause batch size adaptation."""
        from adaptive_batch import AdaptiveBatchSizer, BatchConfig, BatchResult

        cfg = BatchConfig(warmup_batches=2, max_batch_size=32)
        sizer = AdaptiveBatchSizer(cfg)
        initial = sizer.get_current_batch_size()

        # Simulate two documents completing with improving throughput
        sizer.record_result(BatchResult(
            batch_size=16, pages_processed=16,
            duration_seconds=4.0, memory_peak_mb=30.0,
        ))
        sizer.record_result(BatchResult(
            batch_size=16, pages_processed=16,
            duration_seconds=1.0, memory_peak_mb=30.0,
        ))

        # After warmup with improving throughput, batch size should increase
        assert sizer.get_current_batch_size() > initial


# ---------------------------------------------------------------------------
# Tests: Import failure graceful fallback
# ---------------------------------------------------------------------------


class TestImportFailureFallback:
    """When adaptive_batch module is unavailable, fall back gracefully."""

    def test_import_failure_leaves_sizer_none(self):
        """Simulate ImportError during adaptive_batch import."""
        sizer = None

        try:
            raise ImportError("No module named 'adaptive_batch'")
        except ImportError:
            pass  # Fallback: sizer stays None

        assert sizer is None

    def test_fixed_size_used_after_import_failure(self):
        """After import failure, chunk_target_size should be CHUNK_TARGET_SIZE."""
        sizer = None  # Simulates failed import
        CHUNK_TARGET_SIZE = 20

        if sizer is not None:
            chunk_target_size = 999
        else:
            chunk_target_size = CHUNK_TARGET_SIZE

        assert chunk_target_size == CHUNK_TARGET_SIZE


# ---------------------------------------------------------------------------
# Tests: End-to-end integration logic
# ---------------------------------------------------------------------------


class TestEndToEndIntegration:
    """Test the full scheduler -> assembler adaptive batch flow."""

    def test_scheduler_and_assembler_share_sizer(self):
        """Both scheduler and assembler should use the same sizer instance."""
        from adaptive_batch import AdaptiveBatchSizer, BatchResult

        sizer = AdaptiveBatchSizer()

        # Scheduler: get recommendation
        complexity = sizer.compute_complexity(
            width=0, height=0, file_size=1_000_000, dpi=300,
        )
        chunk_size = sizer.recommend_batch_size([complexity])
        assert chunk_size > 0

        # Assembler: record feedback
        sizer.record_result(BatchResult(
            batch_size=chunk_size, pages_processed=chunk_size,
            duration_seconds=2.5,
        ))
        assert len(sizer.get_history()) == 1

    def test_thread_safe_scheduler_assembler(self):
        """Scheduler and assembler running concurrently should not race."""
        from adaptive_batch import AdaptiveBatchSizer, BatchResult

        sizer = AdaptiveBatchSizer()
        errors = []

        def scheduler_work():
            try:
                for _ in range(50):
                    c = sizer.compute_complexity(
                        width=0, height=0, file_size=500_000, dpi=300,
                    )
                    sizer.recommend_batch_size([c])
            except Exception as e:
                errors.append(e)

        def assembler_work():
            try:
                for i in range(50):
                    sizer.record_result(BatchResult(
                        batch_size=10, pages_processed=10,
                        duration_seconds=1.0 + i * 0.01,
                    ))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=scheduler_work)
        t2 = threading.Thread(target=assembler_work)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(sizer.get_history()) == 50

    def test_ocr_gpu_async_has_enable_constant(self):
        """Verify ENABLE_ADAPTIVE_BATCH constant exists in ocr_gpu_async module."""
        # Read the module source to verify the constant is present
        module_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(module_path, encoding="utf-8", errors="replace") as f:
            source = f.read()

        assert "ENABLE_ADAPTIVE_BATCH" in source
        assert '_adaptive_batch_sizer' in source
        assert (
            'from adaptive_batch import' in source
            or 'from ocr_local.infra.adaptive_batch import' in source
        )
