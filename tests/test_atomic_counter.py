"""Tests for thread-safe page counter.

Verifies that global_pages_processed is protected by _pages_processed_lock,
matching the established pattern used for global_docs_processed.
"""

import inspect
import threading


def test_pages_processed_lock_exists():
    """_pages_processed_lock must exist and be a threading.Lock."""
    import ocr_gpu_async

    assert hasattr(ocr_gpu_async, "_pages_processed_lock"), (
        "ocr_gpu_async._pages_processed_lock not found"
    )
    lock = ocr_gpu_async._pages_processed_lock
    assert isinstance(lock, type(threading.Lock())), (
        f"Expected threading.Lock, got {type(lock)}"
    )


def test_docs_processed_lock_exists():
    """_docs_processed_lock must also still exist (regression guard)."""
    import ocr_gpu_async

    assert hasattr(ocr_gpu_async, "_docs_processed_lock"), (
        "ocr_gpu_async._docs_processed_lock not found"
    )


def test_worker_thread_uses_pages_lock():
    """worker_thread source must contain 'with _pages_processed_lock:' around the increment."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    assert "_pages_processed_lock" in source, (
        "worker_thread does not reference _pages_processed_lock"
    )
    assert "with _pages_processed_lock:" in source, (
        "worker_thread does not use 'with _pages_processed_lock:' context manager"
    )


def test_monitor_thread_uses_pages_lock():
    """monitor_thread source must snapshot global_pages_processed under the lock."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.monitor_thread)
    assert "_pages_processed_lock" in source, (
        "monitor_thread does not reference _pages_processed_lock"
    )
    assert "with _pages_processed_lock:" in source, (
        "monitor_thread does not use 'with _pages_processed_lock:' context manager"
    )


def test_lock_pattern_symmetry():
    """Both counters must follow the same lock pattern."""
    import ocr_gpu_async

    # Both locks exist
    assert hasattr(ocr_gpu_async, "_pages_processed_lock")
    assert hasattr(ocr_gpu_async, "_docs_processed_lock")

    # Both counters exist
    assert hasattr(ocr_gpu_async, "global_pages_processed")
    assert hasattr(ocr_gpu_async, "global_docs_processed")

    # Counters are integers (value depends on test ordering -- they are
    # module-level globals that other tests may have incremented)
    assert isinstance(ocr_gpu_async.global_pages_processed, int)
    assert isinstance(ocr_gpu_async.global_docs_processed, int)
