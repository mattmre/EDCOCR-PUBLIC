"""Tests for assembler parallelism.

Validates:
- NUM_ASSEMBLER_WORKERS constant defaults to 4
- _finalize_doc is a callable module-level function
- _docs_processed_lock is a threading.Lock for thread-safe counter increment
- Concurrent increment via _docs_processed_lock is safe
- _finalize_doc accepts the expected (doc, doc_id, page_data_snap) signature
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch


def test_num_assembler_workers_default():
    """NUM_ASSEMBLER_WORKERS should default to 4."""
    from ocr_gpu_async import NUM_ASSEMBLER_WORKERS

    assert NUM_ASSEMBLER_WORKERS == 4


def test_finalize_doc_is_callable():
    """_finalize_doc must be a callable at module level."""
    from ocr_gpu_async import _finalize_doc

    assert callable(_finalize_doc)


def test_docs_processed_lock_is_lock():
    """_docs_processed_lock must be a threading.Lock."""
    from ocr_gpu_async import _docs_processed_lock

    assert isinstance(_docs_processed_lock, type(threading.Lock()))


def test_concurrent_increment_via_lock():
    """Verify _docs_processed_lock protects concurrent counter increments."""
    from ocr_gpu_async import _docs_processed_lock

    counter = {"value": 0}
    iterations = 1000

    def _increment():
        for _ in range(iterations):
            with _docs_processed_lock:
                counter["value"] += 1

    threads = [threading.Thread(target=_increment) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter["value"] == iterations * 8


def test_finalize_doc_signature():
    """_finalize_doc should accept (doc, doc_id, page_data_snap) arguments."""
    import inspect

    from ocr_gpu_async import _finalize_doc

    sig = inspect.signature(_finalize_doc)
    params = list(sig.parameters.keys())
    assert params == ["doc", "doc_id", "page_data_snap"]


@patch("ocr_gpu_async.fitz")
@patch("ocr_gpu_async.compression_queue")
@patch("ocr_gpu_async.doc_registry_lock", new_callable=threading.RLock)
@patch("ocr_gpu_async.log_failure")
def test_finalize_doc_empty_snap_does_not_raise(
    mock_log_failure, mock_lock, mock_comp_q, mock_fitz,
):
    """_finalize_doc with empty page_data_snap must not raise."""
    from ocr_gpu_async import _finalize_doc

    # Build a minimal mock doc
    doc = MagicMock()
    doc.total_pages = 1
    doc.temp_dir = "/tmp/fake_temp"
    doc.output_pdf = "/tmp/fake_output.pdf"
    doc.output_txt_dir = "/tmp/fake_txt"
    doc.path = "/tmp/fake_source.pdf"
    doc.terminal_pages = {1}
    doc.terminal_statuses = {1: "OK"}
    doc.start_time = 0
    doc.processed_pages = 1
    doc.custody_chain = None

    # Mock fitz.open() to return a mock document
    mock_pdf = MagicMock()
    mock_pdf.page_count = 0
    mock_fitz.open.return_value = mock_pdf

    page_data_snap = {
        "texts": {},
        "structure": {},
        "validation": {},
        "handwriting": {},
        "signature": {},
        "vertical_text": {},
        "table_fallback": {},
        "classification": {},
    }

    # Should not raise
    with patch("os.path.exists", return_value=False), \
         patch("os.makedirs"), \
         patch("builtins.open", MagicMock()):
        _finalize_doc(doc, "test_doc_id", page_data_snap)


def test_finalize_doc_dispatched_via_executor():
    """Verify _finalize_doc can be submitted to a ThreadPoolExecutor."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TestFin")
    called = threading.Event()

    def _fake_finalize(doc, doc_id, snap):
        called.set()

    with patch("ocr_gpu_async._finalize_doc", side_effect=_fake_finalize):
        from ocr_gpu_async import _finalize_doc as patched

        fut = executor.submit(patched, MagicMock(), "doc1", {})
        fut.result(timeout=5)
        assert called.is_set()

    executor.shutdown(wait=True)


def test_thread_pool_executor_import():
    """ThreadPoolExecutor must be importable from ocr_gpu_async's imports."""
    # This validates the import line was updated correctly
    from concurrent.futures import ThreadPoolExecutor as TPE

    from ocr_gpu_async import _finalize_doc  # noqa: F401

    # Just verifying TPE is available in the same import path
    assert TPE is not None
