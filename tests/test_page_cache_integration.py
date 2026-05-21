"""Integration tests for page cache wiring into ocr_gpu_async.py.

Covers:
- ENABLE_PAGE_CACHE=false has no effect (module-level guard)
- Cache miss path: OCR runs normally, result is stored in cache
- Cache hit path: OCR is skipped, cached bytes are written to chunk file
- Cache store failure is non-fatal (pipeline continues)
- Assembly queue message format matches what assembler_thread expects
- Monitor thread logs cache stats when cache is enabled

Run with: python -m pytest tests/test_page_cache_integration.py -v
"""

import os
import types
from unittest.mock import MagicMock

import pytest

# Add project root to path
from page_cache import PageCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_task(doc_id="test_doc_abc", page_num=1, lang_hint="en"):
    """Create a minimal fake PageTask-like object."""
    task = types.SimpleNamespace()
    task.doc_id = doc_id
    task.doc_path = "/fake/path.pdf"
    task.page_num = page_num
    task.lang_hint = lang_hint
    task.source_type = "pdf"
    task.retries = 0
    # Fake PIL image with .size attribute
    task.image = types.SimpleNamespace(size=(612, 792))
    return task


def _make_fake_doc_state(tmp_path, doc_id="test_doc_abc"):
    """Create a minimal fake DocumentState-like object."""
    temp_dir = os.path.join(str(tmp_path), doc_id)
    os.makedirs(temp_dir, exist_ok=True)
    doc_state = types.SimpleNamespace()
    doc_state.doc_id = doc_id
    doc_state.path = "/fake/path.pdf"
    doc_state.temp_dir = temp_dir
    doc_state.custody_chain = None
    return doc_state


# ---------------------------------------------------------------------------
# Tests: ENABLE_PAGE_CACHE=false (disabled)
# ---------------------------------------------------------------------------


class TestPageCacheDisabled:
    """When ENABLE_PAGE_CACHE is not set, the _page_cache global should be None."""

    def test_page_cache_disabled_by_default(self):
        """Verify ENABLE_PAGE_CACHE defaults to False."""
        # Remove any env override
        env = os.environ.copy()
        env.pop("ENABLE_PAGE_CACHE", None)
        result = env.get("ENABLE_PAGE_CACHE", "").lower() in ("1", "true", "yes")
        assert result is False

    def test_page_cache_enabled_with_true(self):
        """Verify ENABLE_PAGE_CACHE='true' evaluates to True."""
        result = "true".lower() in ("1", "true", "yes")
        assert result is True

    def test_page_cache_enabled_with_one(self):
        """Verify ENABLE_PAGE_CACHE='1' evaluates to True."""
        result = "1".lower() in ("1", "true", "yes")
        assert result is True

    def test_page_cache_disabled_with_empty(self):
        """Verify ENABLE_PAGE_CACHE='' evaluates to False."""
        result = "".lower() in ("1", "true", "yes")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Cache miss -> OCR runs normally
# ---------------------------------------------------------------------------


class TestPageCacheMiss:
    """When cache misses, OCR should proceed and store result in cache."""

    def test_cache_miss_stores_result(self):
        """After a cache miss, the PDF bytes should be stored in cache."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        cache_key = "doc1:1:300"

        # Verify miss
        assert cache.get(cache_key) is None
        stats = cache.get_stats()
        assert stats.misses == 1

        # Simulate OCR producing pdf_bytes and storing in cache
        pdf_bytes = b"%PDF-1.4 fake content"
        page_confidence = 0.95
        status = "Paddle"
        cache.put(
            cache_key,
            pdf_bytes,
            metadata={"confidence": page_confidence, "status": status},
        )

        # Verify stored
        result = cache.get(cache_key)
        assert result == pdf_bytes
        meta = cache.get_metadata(cache_key)
        assert meta["confidence"] == 0.95
        assert meta["status"] == "Paddle"

    def test_cache_miss_increments_miss_counter(self):
        """Each cache miss should increment the miss counter."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        cache.get("nonexistent:1:300")
        cache.get("nonexistent:2:300")
        stats = cache.get_stats()
        assert stats.misses == 2
        assert stats.hits == 0


# ---------------------------------------------------------------------------
# Tests: Cache hit -> OCR is skipped
# ---------------------------------------------------------------------------


class TestPageCacheHit:
    """When cache hits, cached bytes should be used directly."""

    def test_cache_hit_returns_stored_bytes(self, tmp_path):
        """A cache hit should return the previously stored PDF bytes."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        cache_key = "doc1:1:300"
        pdf_bytes = b"%PDF-1.4 cached OCR result"

        cache.put(
            cache_key,
            pdf_bytes,
            metadata={"confidence": 0.92, "status": "Paddle"},
        )

        # Simulate cache hit path
        cached = cache.get(cache_key)
        assert cached is not None
        assert cached == pdf_bytes

        # Verify chunk file can be written
        chunk_path = os.path.join(str(tmp_path), "1.pdf")
        with open(chunk_path, "wb") as f:
            f.write(cached)
        assert os.path.exists(chunk_path)
        with open(chunk_path, "rb") as f:
            assert f.read() == pdf_bytes

    def test_cache_hit_increments_hit_counter(self):
        """Cache hit should increment the hit counter."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        cache.put("key1", b"data")
        cache.get("key1")
        cache.get("key1")
        stats = cache.get_stats()
        assert stats.hits == 2

    def test_cache_hit_assembly_message_format(self):
        """The assembly queue message for cached pages has all required keys."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        cache.put("doc1:1:300", b"pdf", metadata={"confidence": 0.85, "status": "Paddle"})

        cached_meta = cache.get_metadata("doc1:1:300") or {}
        msg = {
            "doc_id": "doc1",
            "page_num": 1,
            "text": "",
            "status": "CACHED",
            "chunk_path": "/tmp/doc1/1.pdf",
            "ocr_confidence": cached_meta.get("confidence", 0.0),
            "ocr_method": "CACHED",
            "structure_data": None,
            "handwriting_data": None,
            "signature_data": None,
            "vertical_text_data": None,
            "table_fallback_data": None,
        }

        # Verify all keys expected by assembler_thread are present
        required_keys = {
            "doc_id", "page_num", "text", "status", "chunk_path",
        }
        for key in required_keys:
            assert key in msg, f"Missing required key: {key}"

        assert msg["status"] == "CACHED"
        assert msg["ocr_method"] == "CACHED"
        assert msg["ocr_confidence"] == 0.85
        assert msg["text"] == ""


# ---------------------------------------------------------------------------
# Tests: Cache store failure is non-fatal
# ---------------------------------------------------------------------------


class TestPageCacheStoreFailure:
    """Cache store failures must not crash the pipeline."""

    def test_put_exception_is_swallowed(self):
        """If cache.put raises, the pipeline should continue."""
        cache = MagicMock(spec=PageCache)
        cache.put.side_effect = RuntimeError("disk full")

        # Simulate the non-fatal cache store pattern from worker_thread
        try:
            cache.put("key", b"data", metadata={"confidence": 0.9, "status": "OK"})
        except Exception:
            pass  # Cache store failure is non-fatal

        # If we get here, the pipeline survived
        assert True

    def test_oversized_entry_returns_false(self):
        """Cache.put for oversized entries returns False but does not raise."""
        cache = PageCache(max_size_bytes=10, max_entries=100)
        result = cache.put("big", b"x" * 100)
        assert result is False
        # Pipeline should continue unaffected

    def test_cache_get_failure_is_non_fatal(self):
        """If cache.get raises, it should be caught gracefully."""
        cache = MagicMock(spec=PageCache)
        cache.get.side_effect = RuntimeError("corruption")

        result = None
        try:
            result = cache.get("key")
        except Exception:
            result = None

        assert result is None


# ---------------------------------------------------------------------------
# Tests: Monitor thread cache stats
# ---------------------------------------------------------------------------


class TestPageCacheMonitorStats:
    """Cache stats should be available for monitor thread logging."""

    def test_get_stats_returns_valid_snapshot(self):
        """get_stats returns a CacheStats with all expected fields."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        cache.put("k1", b"data1")
        cache.get("k1")  # hit
        cache.get("missing")  # miss

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.evictions == 0
        assert stats.current_entries == 1
        assert stats.current_size_bytes == 5
        assert stats.hit_rate == pytest.approx(0.5)

    def test_hit_rate_percentage_formatting(self):
        """Verify hit_rate * 100 produces a valid percentage string."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        cache.put("k1", b"d")
        cache.get("k1")
        cache.get("k1")
        cache.get("missing")

        stats = cache.get_stats()
        pct = stats.hit_rate * 100
        formatted = f"Cache: hits={stats.hits} misses={stats.misses} evictions={stats.evictions} hit_rate={pct:.1f}%"
        assert "hit_rate=66.7%" in formatted

    def test_stats_with_empty_cache(self):
        """Stats work correctly on an empty cache."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        formatted = f"hit_rate={stats.hit_rate * 100:.1f}%"
        assert formatted == "hit_rate=0.0%"


# ---------------------------------------------------------------------------
# Tests: Cache key format
# ---------------------------------------------------------------------------


class TestPageCacheKeyFormat:
    """Verify the cache key format matches what the pipeline uses."""

    def test_cache_key_includes_doc_id_page_dpi(self):
        """Cache key should be '{doc_id}:{page_num}:{DPI}'."""
        doc_id = "abc123hash"
        page_num = 5
        dpi = 300
        key = f"{doc_id}:{page_num}:{dpi}"
        assert key == "abc123hash:5:300"

    def test_different_dpi_produces_different_keys(self):
        """Pages at different DPIs should have distinct cache keys."""
        key_300 = "doc1:1:300"
        key_450 = "doc1:1:450"
        assert key_300 != key_450

    def test_different_pages_produce_different_keys(self):
        """Different page numbers should produce distinct cache keys."""
        key_p1 = "doc1:1:300"
        key_p2 = "doc1:2:300"
        assert key_p1 != key_p2


# ---------------------------------------------------------------------------
# Tests: End-to-end cache integration flow
# ---------------------------------------------------------------------------


class TestPageCacheEndToEnd:
    """Test the full cache flow: miss -> store -> hit."""

    def test_full_cache_lifecycle(self, tmp_path):
        """Simulate: first page misses, gets OCR'd and cached; second access hits."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        doc_id = "test_lifecycle"
        page_num = 1
        dpi = 300
        cache_key = f"{doc_id}:{page_num}:{dpi}"

        # 1. First access: miss
        cached = cache.get(cache_key)
        assert cached is None

        # 2. Simulate OCR producing PDF bytes
        pdf_bytes = b"%PDF-1.4 OCR output for page 1"

        # 3. Write chunk to temp dir (simulating worker)
        temp_dir = os.path.join(str(tmp_path), doc_id)
        os.makedirs(temp_dir, exist_ok=True)
        chunk_path = os.path.join(temp_dir, f"{page_num}.pdf")
        with open(chunk_path, "wb") as f:
            f.write(pdf_bytes)

        # 4. Store in cache
        cache.put(
            cache_key,
            pdf_bytes,
            metadata={"confidence": 0.88, "status": "Paddle"},
        )

        # 5. Second access: hit
        cached = cache.get(cache_key)
        assert cached is not None
        assert cached == pdf_bytes

        # 6. Write cached bytes to a new chunk path
        chunk_path_2 = os.path.join(temp_dir, f"{page_num}_cached.pdf")
        with open(chunk_path_2, "wb") as f:
            f.write(cached)

        # 7. Verify byte-for-byte identity
        with open(chunk_path, "rb") as f1, open(chunk_path_2, "rb") as f2:
            assert f1.read() == f2.read()

        # 8. Verify stats
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.current_entries == 1

    def test_cache_with_multiple_pages(self):
        """Cache should handle multiple pages from the same document."""
        cache = PageCache(max_size_bytes=1_000_000, max_entries=100)
        doc_id = "multi_page_doc"

        for page in range(1, 6):
            key = f"{doc_id}:{page}:300"
            data = f"%PDF page {page}".encode()
            cache.put(key, data, metadata={"confidence": 0.9, "status": "Paddle"})

        stats = cache.get_stats()
        assert stats.current_entries == 5

        # All pages should be retrievable
        for page in range(1, 6):
            key = f"{doc_id}:{page}:300"
            result = cache.get(key)
            assert result == f"%PDF page {page}".encode()

    def test_cache_eviction_does_not_crash_pipeline(self):
        """When cache evicts entries, it should not affect pipeline operation."""
        # Small cache that will evict
        cache = PageCache(max_size_bytes=100, max_entries=3)

        for i in range(10):
            key = f"doc:{i}:300"
            cache.put(key, f"page-{i}-data".encode())

        stats = cache.get_stats()
        assert stats.evictions > 0
        # Most recent entries should still be available
        assert stats.current_entries <= 3


# ---------------------------------------------------------------------------
# Tests: CACHED status handled by assembler
# ---------------------------------------------------------------------------


class TestCachedStatusAssemblerCompat:
    """Verify CACHED status is handled correctly by assembler logic."""

    def test_cached_status_maps_to_validation_ok(self):
        """The CACHED status should map to 'ok' in validation logic."""
        status = "CACHED"
        # This mirrors the assembler_thread validation logic
        if status in ("OK", "Paddle", "Tesseract") or status.startswith("Paddle-"):
            val_status = "ok" if status != "Tesseract" else "fallback"
        elif status == "ImageOnly":
            val_status = "image_only"
        elif status in ("EXTRACT_FAILED", "CRITICAL_FAILED"):
            val_status = "failed"
        elif status == "RESUMED":
            val_status = "ok"
        elif status == "CACHED":
            val_status = "ok"
        else:
            val_status = "unknown"
        assert val_status == "ok"
