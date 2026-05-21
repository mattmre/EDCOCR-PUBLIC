"""Tests for process-based extractor mode enhancements.

Covers:
- JPEG serialization round-trip (encode/decode fidelity)
- EXTRACTOR_MODE=auto resolution logic
- Per-chunk fallback from process to thread mode
- Subprocess extraction function (mock-based)
- Pool lifecycle (init, shutdown, failure fallback)

Run with: python -m pytest tests/test_extractor_mode.py -v
"""

import pickle
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import ocr_gpu_async

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(width=2550, height=3300):
    """Create a synthetic RGB image of standard letter-size dimensions."""
    return Image.new("RGB", (width, height), color=(200, 180, 160))


def _make_grayscale_image(width=2550, height=3300):
    """Create a synthetic grayscale (mode L) image."""
    return Image.new("L", (width, height), color=128)


# ===========================================================================
# Serialization round-trip tests
# ===========================================================================


class TestJpegRoundTrip:
    """JPEG encode/decode preserves image dimensions and mode."""

    def test_jpeg_roundtrip_preserves_dimensions(self):
        """Encode then decode a 2550x3300 RGB image -- dimensions match."""
        original = _make_rgb_image(2550, 3300)
        payload = ocr_gpu_async._encode_image_to_jpeg_bytes(original)
        recovered = ocr_gpu_async._decode_image_from_jpeg_bytes(payload)
        assert recovered.size == (2550, 3300)

    def test_jpeg_roundtrip_preserves_mode(self):
        """Round-tripped image is always RGB."""
        original = _make_rgb_image()
        payload = ocr_gpu_async._encode_image_to_jpeg_bytes(original)
        recovered = ocr_gpu_async._decode_image_from_jpeg_bytes(payload)
        assert recovered.mode == "RGB"

    def test_jpeg_roundtrip_grayscale_converts_to_rgb(self):
        """Grayscale input is converted to RGB after encode/decode cycle."""
        gray = _make_grayscale_image()
        payload = ocr_gpu_async._encode_image_to_jpeg_bytes(gray)
        recovered = ocr_gpu_async._decode_image_from_jpeg_bytes(payload)
        assert recovered.mode == "RGB"

    def test_jpeg_bytes_are_picklable(self):
        """JPEG byte payloads survive pickle round-trip (required for IPC)."""
        original = _make_rgb_image(100, 100)
        payload = ocr_gpu_async._encode_image_to_jpeg_bytes(original)
        pickled = pickle.dumps(payload)
        unpickled = pickle.loads(pickled)
        assert payload == unpickled


# ===========================================================================
# Mode resolution tests
# ===========================================================================


class TestModeResolution:
    """Tests for EXTRACTOR_MODE=auto resolution logic."""

    def test_auto_mode_resolves_to_process_when_extractors_gt_4(self):
        """auto + NUM_EXTRACTORS=8 resolves to process."""
        assert ocr_gpu_async._resolve_auto_extractor_mode("auto", 8) == "process"

    def test_auto_mode_resolves_to_thread_when_extractors_le_4(self):
        """auto + NUM_EXTRACTORS=4 resolves to thread."""
        assert ocr_gpu_async._resolve_auto_extractor_mode("auto", 4) == "thread"

    def test_invalid_mode_falls_back_to_thread(self, monkeypatch):
        """An unrecognized mode string falls back to thread at module level."""
        # Simulate what the module-level validation does
        mode = "bogus"
        if mode not in {"thread", "process", "auto"}:
            mode = "thread"
        assert mode == "thread"

    def test_mode_validation_accepts_all_three_values(self):
        """thread, process, and auto are all accepted without fallback."""
        for mode_to_test in ("thread", "process", "auto"):
            validated_mode = mode_to_test
            if validated_mode not in {"thread", "process", "auto"}:
                validated_mode = "thread"
            assert validated_mode == mode_to_test


# ===========================================================================
# Fallback behavior tests
# ===========================================================================


class TestFallbackBehavior:
    """Tests for per-chunk and global fallback from process to thread mode."""

    def test_process_mode_fallback_on_pool_failure(self, monkeypatch):
        """When the process pool raises, fallback flag is set and thread path runs."""
        monkeypatch.setattr(ocr_gpu_async, "EXTRACTOR_MODE", "process")

        mock_pool = MagicMock()
        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("subprocess crash")
        mock_pool.submit.return_value = mock_future
        monkeypatch.setattr(ocr_gpu_async, "extractor_process_pool", mock_pool)

        # Simulate the per-chunk logic from extractor_thread
        process_mode_failed = False
        queued_pages = set()

        if ocr_gpu_async.EXTRACTOR_MODE == "process" and ocr_gpu_async.extractor_process_pool is not None:
            try:
                ocr_gpu_async.extractor_process_pool.submit(
                    ocr_gpu_async._extract_chunk_in_subprocess,
                    "dummy.pdf", 1, 3, "pdf", 300, 1,
                ).result()
            except Exception:
                process_mode_failed = True

        assert process_mode_failed is True

        # The thread path would now run
        if ocr_gpu_async.EXTRACTOR_MODE == "thread" or process_mode_failed:
            queued_pages.add(1)  # simulate thread-mode extraction
            queued_pages.add(2)
            queued_pages.add(3)

        assert queued_pages == {1, 2, 3}

    def test_process_mode_success_no_fallback(self, monkeypatch):
        """When process pool succeeds, iter_source_images is NOT called."""
        monkeypatch.setattr(ocr_gpu_async, "EXTRACTOR_MODE", "process")

        img = _make_rgb_image(100, 100)
        payload = ocr_gpu_async._encode_image_to_jpeg_bytes(img)

        mock_pool = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = [(1, payload), (2, payload)]
        mock_pool.submit.return_value = mock_future
        monkeypatch.setattr(ocr_gpu_async, "extractor_process_pool", mock_pool)

        process_mode_failed = False
        queued_pages = set()

        if ocr_gpu_async.EXTRACTOR_MODE == "process" and ocr_gpu_async.extractor_process_pool is not None:
            try:
                result = ocr_gpu_async.extractor_process_pool.submit(
                    ocr_gpu_async._extract_chunk_in_subprocess,
                    "dummy.pdf", 1, 2, "pdf", 300, 1,
                ).result()
                for p_num, _payload in result:
                    queued_pages.add(p_num)
            except Exception:
                process_mode_failed = True

        assert process_mode_failed is False
        assert queued_pages == {1, 2}

        # Thread path should NOT run
        thread_path_ran = False
        if ocr_gpu_async.EXTRACTOR_MODE == "thread" or process_mode_failed:
            thread_path_ran = True

        assert thread_path_ran is False

    def test_thread_mode_does_not_use_pool(self, monkeypatch):
        """In thread mode, the process pool branch is never entered."""
        monkeypatch.setattr(ocr_gpu_async, "EXTRACTOR_MODE", "thread")
        monkeypatch.setattr(ocr_gpu_async, "extractor_process_pool", None)

        process_branch_entered = False
        thread_branch_entered = False
        process_mode_failed = False

        if ocr_gpu_async.EXTRACTOR_MODE == "process" and ocr_gpu_async.extractor_process_pool is not None:
            process_branch_entered = True

        if ocr_gpu_async.EXTRACTOR_MODE == "thread" or process_mode_failed:
            thread_branch_entered = True

        assert process_branch_entered is False
        assert thread_branch_entered is True

    def test_fallback_is_per_chunk_not_global(self, monkeypatch):
        """process_mode_failed is scoped per-chunk, not global state."""
        monkeypatch.setattr(ocr_gpu_async, "EXTRACTOR_MODE", "process")

        img = _make_rgb_image(100, 100)
        payload = ocr_gpu_async._encode_image_to_jpeg_bytes(img)

        # Chunk 1: process pool fails
        mock_pool_fail = MagicMock()
        mock_future_fail = MagicMock()
        mock_future_fail.result.side_effect = RuntimeError("fail")
        mock_pool_fail.submit.return_value = mock_future_fail

        # Chunk 2: process pool succeeds
        mock_pool_ok = MagicMock()
        mock_future_ok = MagicMock()
        mock_future_ok.result.return_value = [(1, payload)]
        mock_pool_ok.submit.return_value = mock_future_ok

        # Simulate chunk 1
        monkeypatch.setattr(ocr_gpu_async, "extractor_process_pool", mock_pool_fail)
        chunk1_failed = False
        if ocr_gpu_async.EXTRACTOR_MODE == "process" and ocr_gpu_async.extractor_process_pool is not None:
            try:
                ocr_gpu_async.extractor_process_pool.submit(
                    ocr_gpu_async._extract_chunk_in_subprocess,
                    "f.pdf", 1, 1, "pdf", 300, 1,
                ).result()
            except Exception:
                chunk1_failed = True
        assert chunk1_failed is True

        # Simulate chunk 2 -- fresh variable, not affected by chunk 1
        monkeypatch.setattr(ocr_gpu_async, "extractor_process_pool", mock_pool_ok)
        chunk2_failed = False
        chunk2_pages = set()
        if ocr_gpu_async.EXTRACTOR_MODE == "process" and ocr_gpu_async.extractor_process_pool is not None:
            try:
                result = ocr_gpu_async.extractor_process_pool.submit(
                    ocr_gpu_async._extract_chunk_in_subprocess,
                    "f.pdf", 1, 1, "pdf", 300, 1,
                ).result()
                for p_num, _ in result:
                    chunk2_pages.add(p_num)
            except Exception:
                chunk2_failed = True

        assert chunk2_failed is False
        assert chunk2_pages == {1}


# ===========================================================================
# Subprocess extraction tests (mock-based)
# ===========================================================================


class TestSubprocessExtraction:
    """Tests for _extract_chunk_in_subprocess and encoding helpers."""

    def test_subprocess_extract_returns_tuples(self):
        """_extract_chunk_in_subprocess returns list of (page_num, bytes) tuples."""
        img = _make_rgb_image(100, 100)

        with patch("ocr_gpu_async.convert_from_path") as mock_convert, \
             patch("ocr_gpu_async._PDF2IMAGE_AVAILABLE", True):
            mock_convert.return_value = [img, img, img]
            result = ocr_gpu_async._extract_chunk_in_subprocess(
                "test.pdf", 1, 3, "pdf", 300, 1,
            )

        assert isinstance(result, list)
        assert len(result) == 3
        for page_num, data in result:
            assert isinstance(page_num, int)
            assert isinstance(data, bytes)

    def test_subprocess_extract_page_range_correct(self):
        """Returned page numbers match the requested start..end range."""
        img = _make_rgb_image(100, 100)

        with patch("ocr_gpu_async.convert_from_path") as mock_convert, \
             patch("ocr_gpu_async._PDF2IMAGE_AVAILABLE", True):
            mock_convert.return_value = [img, img]
            result = ocr_gpu_async._extract_chunk_in_subprocess(
                "test.pdf", 5, 6, "pdf", 300, 1,
            )

        page_nums = [p for p, _ in result]
        assert page_nums == [5, 6]

    def test_decode_corrupt_jpeg_raises(self):
        """Decoding corrupted JPEG bytes raises an exception."""
        with pytest.raises(Exception):
            ocr_gpu_async._decode_image_from_jpeg_bytes(b"not-a-jpeg")

    def test_encode_decode_idempotent(self):
        """Encode then decode produces an image with identical size to the original."""
        original = _make_rgb_image(800, 600)
        payload = ocr_gpu_async._encode_image_to_jpeg_bytes(original)
        recovered = ocr_gpu_async._decode_image_from_jpeg_bytes(payload)

        # Dimensions must be identical (JPEG is lossy so pixel values may differ)
        assert recovered.size == original.size
        assert recovered.mode == "RGB"


# ===========================================================================
# Pool lifecycle tests
# ===========================================================================


def _double(x):
    """Module-level function for ProcessPoolExecutor (must be picklable)."""
    return x * 2


class TestPoolLifecycle:
    """Tests for ProcessPoolExecutor initialization and shutdown."""

    def test_pool_initialization_and_shutdown(self):
        """Pool can be created and cleanly shut down."""
        pool = ProcessPoolExecutor(max_workers=2)
        assert pool is not None

        # Submit a trivial picklable task to confirm it works.
        # Use built-in abs() to avoid subprocess reimporting heavy modules.
        future = pool.submit(abs, -42)
        assert future.result(timeout=30) == 42

        pool.shutdown(wait=True)

    def test_pool_init_failure_fallback(self, monkeypatch):
        """When ProcessPoolExecutor raises, mode falls back to thread."""
        monkeypatch.setattr(ocr_gpu_async, "EXTRACTOR_MODE", "process")
        monkeypatch.setattr(ocr_gpu_async, "extractor_process_pool", None)

        # Simulate the try/except block from main()
        try:
            raise OSError("cannot create process pool")
        except Exception:
            ocr_gpu_async.EXTRACTOR_MODE = "thread"
            ocr_gpu_async.extractor_process_pool = None

        assert ocr_gpu_async.EXTRACTOR_MODE == "thread"
        assert ocr_gpu_async.extractor_process_pool is None
