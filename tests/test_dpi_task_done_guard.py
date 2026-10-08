"""Regression tests for DPI escalation task_done() guard (maintenance review).

The expert panel finding maintenance review stated that DPI escalation re-queues a task
but still calls task_done(), causing image_queue.join() to unblock prematurely.

Verification: every consumed queue item calls ``task_done`` even when it
enqueues a replacement attempt.  The flag guards terminal-only accounting and
image disposal so the requeued object remains usable:

    _dpi_requeued = False          # reset each iteration
    try:
        ...
        image_queue.put(task)      # unfinished +1
        _dpi_requeued = True
        continue
    finally:
        image_queue.task_done()    # consumed attempt unfinished -1
        if not _dpi_requeued:
            ...                    # count/dispose only terminal attempt

These tests verify that guard is in place and structurally correct.
"""

from __future__ import annotations

import ast
import inspect


def test_dpi_requeued_flag_initialised_each_iteration():
    """_dpi_requeued = False must appear inside the worker_thread while-loop body."""
    import ocr_gpu_async

    tree = ast.parse(inspect.getsource(ocr_gpu_async.worker_thread))
    initialized = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_dpi_requeued"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in ast.walk(tree)
    )
    assert initialized, "worker_thread must reset _dpi_requeued to False at the top of each loop iteration"


def test_dpi_requeued_set_true_on_requeue():
    """_dpi_requeued = True must appear in the DPI escalation branch."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    assert (
        "_dpi_requeued = True" in source
    ), "worker_thread must set _dpi_requeued = True when re-queuing for DPI escalation"
    assert "image_queue.put_nowait(task)" in source
    assert "image_queue.put(task)" not in source
    assert "except queue.Full:" in source


def test_task_done_precedes_terminal_only_guard():
    """Every consumed attempt completes before terminal-only accounting."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    task_done_idx = source.index("image_queue.task_done()")
    terminal_guard_idx = source.index("if not _dpi_requeued:", task_done_idx)
    assert task_done_idx < terminal_guard_idx


def test_requeued_image_is_not_deleted():
    """Image disposal must remain inside the terminal-only guard."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    guard_idx = source.index("if not _dpi_requeued:")
    delete_idx = source.index("del task.image", guard_idx)
    assert guard_idx < delete_idx


def test_dpi_requeued_flag_count():
    """_dpi_requeued must appear at least 3 times: init, set-true, and guard."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    count = source.count("_dpi_requeued")
    assert (
        count >= 3
    ), f"Expected _dpi_requeued to appear ≥3 times (init, set, guard), found {count}"


def test_checkpoint_and_page_cache_commit_only_after_escalation_decision():
    """A nonterminal low-DPI attempt must not survive as a resume/cache hit."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.worker_thread)
    escalation_commit = source.index("_dpi_requeued = True")
    checkpoint_commit = source.index("_embed_page_checkpoint_metadata")
    page_cache_commit = source.index("_page_cache.put", checkpoint_commit)

    assert escalation_commit < checkpoint_commit < page_cache_commit
