"""Regression tests for DPI escalation task_done() guard.

The expert panel finding  stated that DPI escalation re-queues a task
but still calls task_done(), causing image_queue.join() to unblock prematurely.

Verification: the current code uses a _dpi_requeued flag to guard task_done():

    _dpi_requeued = False          # reset each iteration
    try:
        ...
        image_queue.put(task)
        _dpi_requeued = True       # skip task_done() in finally
        continue
    finally:
        if not _dpi_requeued:
            image_queue.task_done()

These tests verify that guard is in place and structurally correct.
"""

from __future__ import annotations

import inspect


def test_dpi_requeued_flag_initialised_each_iteration():
    """_dpi_requeued = False must appear inside the worker_thread while-loop body."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    assert "_dpi_requeued = False" in source, (
        "worker_thread must reset _dpi_requeued to False at the top of each loop iteration"
    )


def test_dpi_requeued_set_true_on_requeue():
    """_dpi_requeued = True must appear in the DPI escalation branch."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    assert "_dpi_requeued = True" in source, (
        "worker_thread must set _dpi_requeued = True when re-queuing for DPI escalation"
    )


def test_task_done_guarded_by_dpi_requeued():
    """image_queue.task_done() must be guarded by 'if not _dpi_requeued'."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    assert "if not _dpi_requeued:" in source, (
        "worker_thread must guard task_done() with 'if not _dpi_requeued:'"
    )
    # Verify task_done() call appears inside the guard, not unconditionally
    lines = source.splitlines()
    guard_idx = next(
        (i for i, ln in enumerate(lines) if "if not _dpi_requeued:" in ln), None
    )
    assert guard_idx is not None, "guard line not found"
    # The task_done() call should appear on the line(s) immediately after the guard
    guarded_block = "\n".join(lines[guard_idx : guard_idx + 4])
    assert "task_done()" in guarded_block, (
        "image_queue.task_done() must appear inside the 'if not _dpi_requeued:' block"
    )


def test_task_done_not_called_unconditionally():
    """task_done() must NOT appear outside the _dpi_requeued guard."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    lines = source.splitlines()

    task_done_lines = [i for i, ln in enumerate(lines) if "task_done()" in ln]
    assert len(task_done_lines) >= 1, "task_done() must be called somewhere in worker_thread"

    for idx in task_done_lines:
        # Find the nearest preceding 'if not _dpi_requeued:' within 5 lines
        preceding = "\n".join(lines[max(0, idx - 5) : idx + 1])
        assert "_dpi_requeued" in preceding, (
            f"task_done() at source line ~{idx} is not guarded by _dpi_requeued check"
        )


def test_dpi_requeued_flag_count():
    """_dpi_requeued must appear at least 3 times: init, set-true, and guard."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    count = source.count("_dpi_requeued")
    assert count >= 3, (
        f"Expected _dpi_requeued to appear ≥3 times (init, set, guard), found {count}"
    )
