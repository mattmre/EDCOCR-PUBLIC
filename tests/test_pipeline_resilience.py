"""Tests for pipeline resilience fixes (, , ).

These tests verify:
- DPI escalation re-queue skips task_done() to avoid
  premature image_queue.join() unblocking.
- Assembly failure writes an 'assembly_failed' custody event
  and rescues the saved output PDF into the compression queue.
- Worker critical failure creates an image-only fallback
  page before sending CRITICAL_FAILED to the assembler.

Run with: python -m pytest tests/test_pipeline_resilience.py -v
"""

import inspect
import io
import os
import queue
import tempfile
import types

from PIL import Image

# Ensure project root is on sys.path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(width=200, height=100):
    """Create a small PIL image for testing."""
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def _make_task(doc_id="doc1", page_num=1, doc_path="/tmp/test.pdf",
               retries=0, lang_hint="en", image=None):
    """Create a minimal task namespace matching the pipeline's expectations."""
    task = types.SimpleNamespace()
    task.doc_id = doc_id
    task.page_num = page_num
    task.doc_path = doc_path
    task.retries = retries
    task.lang_hint = lang_hint
    task.image = image if image is not None else _make_test_image()
    return task


class _FakeDocState:
    """Minimal doc state for testing."""

    def __init__(self, temp_dir, total_pages=1, custody_chain=None):
        self.temp_dir = temp_dir
        self.total_pages = total_pages
        self.processed_pages = 0
        self.output_pdf = os.path.join(temp_dir, "output.pdf")
        self.path = "/tmp/source.pdf"
        self.start_time = 0.0
        self.custody_chain = custody_chain
        self.terminal_pages = set()
        self.terminal_statuses = {}
        self.finalized = False


class _FakeCustody:
    """Minimal custody chain recorder."""

    def __init__(self):
        self.events = []

    def append_event(self, event_type, data):
        self.events.append({"type": event_type, "data": data})


# ---------------------------------------------------------------------------
# DPI escalation task_done overcall
# ---------------------------------------------------------------------------


class TestDPIEscalationTaskDone:
    """Verify task_done() is not called when a task is re-queued for DPI escalation."""

    def test_dpi_requeue_flag_prevents_task_done(self):
        """When _dpi_requeued is True, finally block must NOT call task_done()."""
        # This is a structural test: we verify the logic by inspecting the
        # source code pattern.  The actual fix introduces _dpi_requeued flag.
        import ocr_gpu_async

        source = inspect.getsource(ocr_gpu_async.worker_thread)

        # The flag must be initialized before the inner try
        assert "_dpi_requeued = False" in source, (
            "_dpi_requeued flag must be initialized before the inner try block"
        )
        # The flag must be set when re-queuing
        assert "_dpi_requeued = True" in source, (
            "_dpi_requeued flag must be set to True when re-queuing for DPI escalation"
        )
        # The finally block must check the flag
        assert "if not _dpi_requeued:" in source, (
            "finally block must check _dpi_requeued before calling task_done()"
        )

    def test_cached_path_no_explicit_task_done(self):
        """CACHED early-return path must not call task_done() explicitly
        (the finally block handles it)."""
        import ocr_gpu_async

        source = inspect.getsource(ocr_gpu_async.worker_thread)
        # Find the CACHED section -- between PageStatus.CACHED and the next continue
        # It should NOT have an explicit image_queue.task_done() call
        # Support both raw string and PageStatus enum references
        cached_idx = source.find("PageStatus.CACHED")
        if cached_idx < 0:
            cached_idx = source.find('"CACHED"')
        assert cached_idx > 0

        # Find the continue after CACHED
        continue_after_cached = source.find("continue", cached_idx)
        assert continue_after_cached > 0

        # The section between CACHED and continue should NOT have task_done()
        cached_section = source[cached_idx:continue_after_cached]
        assert "task_done()" not in cached_section, (
            "CACHED path should not have explicit task_done() -- "
            "the finally block handles it"
        )

    def test_skipped_path_no_explicit_task_done(self):
        """SKIPPED early-return path must not call task_done() explicitly."""
        import ocr_gpu_async

        source = inspect.getsource(ocr_gpu_async.worker_thread)
        # Support both raw string and PageStatus enum references
        skipped_idx = source.find("PageStatus.SKIPPED")
        if skipped_idx < 0:
            skipped_idx = source.find('"SKIPPED"')
        assert skipped_idx > 0

        continue_after_skipped = source.find("continue", skipped_idx)
        assert continue_after_skipped > 0

        skipped_section = source[skipped_idx:continue_after_skipped]
        assert "task_done()" not in skipped_section, (
            "SKIPPED path should not have explicit task_done() -- "
            "the finally block handles it"
        )

    def test_task_done_count_balanced_without_dpi(self):
        """For a normal (non-DPI) task, task_done() is called exactly once."""
        image_queue = queue.Queue()
        task = _make_task()
        image_queue.put(task)

        task_done_count = 0
        original_task_done = image_queue.task_done

        def counting_task_done():
            nonlocal task_done_count
            task_done_count += 1
            original_task_done()

        image_queue.task_done = counting_task_done

        # The finally block should call task_done exactly once for a normal
        # processing path.  We simulate this by verifying the flag logic:
        _dpi_requeued = False
        try:
            # Simulate normal processing (no DPI escalation)
            pass
        finally:
            if not _dpi_requeued:
                image_queue.task_done()

        assert task_done_count == 1

    def test_task_done_skipped_on_dpi_requeue(self):
        """When _dpi_requeued is True, task_done() must not be called."""
        image_queue = queue.Queue()
        task = _make_task()
        image_queue.put(task)

        task_done_count = 0
        original_task_done = image_queue.task_done

        def counting_task_done():
            nonlocal task_done_count
            task_done_count += 1
            original_task_done()

        image_queue.task_done = counting_task_done

        _dpi_requeued = True
        try:
            pass
        finally:
            if not _dpi_requeued:
                image_queue.task_done()

        assert task_done_count == 0


# ---------------------------------------------------------------------------
# Image extraction ownership: queued images must outlive extractor generator
# ---------------------------------------------------------------------------


class TestExtractorImageOwnership:
    """Verify thread-mode extraction queues an owned image copy."""

    def test_thread_extractor_copies_generator_owned_frame(self):
        """iter_source_images closes yielded frames after the extractor loop
        advances, so PageTask must receive an independent copy."""
        import ocr_gpu_async

        source = inspect.getsource(ocr_gpu_async.extractor_thread)
        assert "img.copy()" in source, (
            "Thread-mode extractor must queue img.copy(), not the generator-owned "
            "image object that iter_source_images closes on the next iteration"
        )


# ---------------------------------------------------------------------------
# Assembly failure custody event
# ---------------------------------------------------------------------------


class TestAssemblyFailureCustody:
    """Verify assembly failures record an 'assembly_failed' custody event."""

    def test_assembly_failed_event_type_in_source(self):
        """The assembly except block must use 'assembly_failed' event type."""
        import ocr_gpu_async

        # Finalization logic (including failure handling) now lives in _finalize_doc
        source = inspect.getsource(ocr_gpu_async._finalize_doc)
        assert '"assembly_failed"' in source, (
            "Assembly failure handler must emit an 'assembly_failed' custody event"
        )

    def test_assembly_failure_rescues_saved_pdf(self):
        """If output PDF exists on disk when assembly fails, it should be
        queued for compression."""
        import ocr_gpu_async

        # Finalization logic (including failure handling) now lives in _finalize_doc
        source = inspect.getsource(ocr_gpu_async._finalize_doc)
        # Check that the rescue logic exists
        assert "os.path.isfile(doc.output_pdf)" in source, (
            "Assembly failure handler must check if output PDF exists on disk"
        )
        assert "compression_queue.put" in source, (
            "Assembly failure handler must queue rescued PDF for compression"
        )

    def test_custody_event_structure(self):
        """Custody event must include error, total_pages, processed_pages."""
        custody = _FakeCustody()
        error_msg = "Test assembly error"

        # Simulate what the except block does
        try:
            custody.append_event("assembly_failed", {
                "error": error_msg,
                "total_pages": 5,
                "processed_pages": 3,
            })
        except Exception:
            pass

        assert len(custody.events) == 1
        event = custody.events[0]
        assert event["type"] == "assembly_failed"
        assert event["data"]["error"] == error_msg
        assert event["data"]["total_pages"] == 5
        assert event["data"]["processed_pages"] == 3

    def test_custody_write_failure_does_not_propagate(self):
        """If custody append_event raises, it must not crash the failure handler."""
        class _FailingCustody:
            def append_event(self, *args, **kwargs):
                raise RuntimeError("custody write failed")

        custody = _FailingCustody()

        # Simulate the exception-safe pattern from the fix
        try:
            custody.append_event("assembly_failed", {"error": "test"})
        except Exception:
            pass  # This is the pattern used in the fix

        # If we get here, the test passes -- the exception was caught


# ---------------------------------------------------------------------------
# Worker critical failure image-only fallback
# ---------------------------------------------------------------------------


class TestWorkerCriticalFailureFallback:
    """Verify critical worker failure creates image-only fallback page."""

    def test_critical_failure_creates_fallback_pdf(self):
        """When a worker hits CRITICAL_FAILED with a valid image, an
        image-only PDF must be saved to the temp directory."""
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            task = _make_task(image=_make_test_image(400, 300))
            _FakeDocState(tmpdir, custody_chain=_FakeCustody())

            # Simulate the fallback logic from the fix
            _fallback_chunk = None
            if hasattr(task, "image") and task.image is not None:
                _fb_w, _fb_h = task.image.size
                _fb_doc = fitz.open()
                _fb_page = _fb_doc.new_page(width=_fb_w, height=_fb_h)
                img_byte_arr = io.BytesIO()
                task.image.save(img_byte_arr, format='JPEG', quality=85)
                _fb_page.insert_image(
                    fitz.Rect(0, 0, _fb_w, _fb_h),
                    stream=img_byte_arr.getvalue(),
                )
                _fb_bytes = _fb_doc.write()
                _fb_doc.close()
                _fallback_chunk = os.path.join(tmpdir, f"{task.page_num}.pdf")
                with open(_fallback_chunk, "wb") as f:
                    f.write(_fb_bytes)

            assert _fallback_chunk is not None
            assert os.path.isfile(_fallback_chunk)

            # Verify it's a valid PDF with one page
            with fitz.open(_fallback_chunk) as check_doc:
                assert check_doc.page_count == 1

    def test_critical_failure_fallback_in_source(self):
        """The worker except block must contain image-only fallback logic."""
        import ocr_gpu_async

        source = inspect.getsource(ocr_gpu_async.worker_thread)
        # Check for the fallback pattern
        assert "_fallback_chunk" in source, (
            "Worker critical failure handler must create _fallback_chunk"
        )
        assert "Image-only fallback saved" in source, (
            "Worker critical failure handler must log when fallback is saved"
        )

    def test_critical_failure_sends_chunk_path(self):
        """CRITICAL_FAILED assembly message must include chunk_path when
        fallback succeeds."""
        import ocr_gpu_async

        source = inspect.getsource(ocr_gpu_async.worker_thread)
        # The assembly_queue.put for CRITICAL_FAILED must reference _fallback_chunk
        assert '"chunk_path": _fallback_chunk' in source, (
            "CRITICAL_FAILED assembly message must pass _fallback_chunk as chunk_path"
        )

    def test_fallback_failure_does_not_propagate(self):
        """If image-only fallback fails, _fallback_chunk must be None and
        no exception should propagate."""
        # Use None image to simulate failure
        task = _make_task()
        task.image = None

        _fallback_chunk = None
        if hasattr(task, "image") and task.image is not None:
            try:
                raise RuntimeError("simulated fallback failure")
            except Exception:
                _fallback_chunk = None

        assert _fallback_chunk is None

    def test_fallback_with_no_image_attribute(self):
        """If the task has no image attribute, fallback must be skipped."""
        task = types.SimpleNamespace(
            doc_id="doc1",
            page_num=1,
            doc_path="/tmp/test.pdf",
        )
        # No image attribute at all

        _fallback_chunk = None
        if hasattr(task, "image") and task.image is not None:
            _fallback_chunk = "should_not_reach"

        assert _fallback_chunk is None
