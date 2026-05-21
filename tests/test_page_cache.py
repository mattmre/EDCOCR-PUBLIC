"""Tests for memory-mapped page caching module (page_cache.py).

Covers:
- CacheStrategy enum values and count
- CacheEntry defaults and is_expired property
- CacheStats defaults, hit_rate, and to_dict
- PageCache construction with defaults and custom values
- put/get basic operations
- put evicts when max_entries reached
- put evicts when max_size_bytes reached
- put rejects oversized single entry
- put updates existing key
- get updates last_accessed and access_count
- get expired entry returns None and increments misses
- contains for present, absent, and expired entries
- remove existing and nonexistent keys
- get_metadata for present, absent, and expired entries
- get_stats tracks hits, misses, and evictions
- clear resets all entries and counters
- keys returns non-expired keys and prunes expired ones
- Thread safety under concurrent put/get

Run with: python -m pytest tests/test_page_cache.py -v
"""

import threading
import time

import pytest

# Add project root to path
from page_cache import CacheEntry, CacheStats, CacheStrategy, PageCache

# ---------------------------------------------------------------------------
# Tests: CacheStrategy
# ---------------------------------------------------------------------------


class TestCacheStrategy:
    def test_enum_has_three_members(self):
        assert len(CacheStrategy) == 3

    def test_lru_value(self):
        assert CacheStrategy.LRU.value == "lru"

    def test_lfu_value(self):
        assert CacheStrategy.LFU.value == "lfu"

    def test_ttl_value(self):
        assert CacheStrategy.TTL.value == "ttl"

    def test_enum_from_value(self):
        assert CacheStrategy("lru") is CacheStrategy.LRU


# ---------------------------------------------------------------------------
# Tests: CacheEntry
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_default_fields(self):
        now = time.time()
        entry = CacheEntry(
            key="page1",
            data=b"hello",
            size_bytes=5,
            created_at=now,
            last_accessed=now,
        )
        assert entry.key == "page1"
        assert entry.data == b"hello"
        assert entry.size_bytes == 5
        assert entry.access_count == 0
        assert entry.ttl_seconds == 0.0
        assert entry.metadata == {}

    def test_is_expired_no_ttl(self):
        now = time.time()
        entry = CacheEntry(
            key="k",
            data=b"",
            size_bytes=0,
            created_at=now - 9999,
            last_accessed=now,
            ttl_seconds=0,
        )
        assert entry.is_expired is False

    def test_is_expired_within_ttl(self):
        now = time.time()
        entry = CacheEntry(
            key="k",
            data=b"",
            size_bytes=0,
            created_at=now,
            last_accessed=now,
            ttl_seconds=60,
        )
        assert entry.is_expired is False

    def test_is_expired_past_ttl(self):
        entry = CacheEntry(
            key="k",
            data=b"",
            size_bytes=0,
            created_at=time.time() - 10,
            last_accessed=time.time(),
            ttl_seconds=1,
        )
        assert entry.is_expired is True

    def test_custom_metadata(self):
        now = time.time()
        entry = CacheEntry(
            key="k",
            data=b"x",
            size_bytes=1,
            created_at=now,
            last_accessed=now,
            metadata={"page": 3},
        )
        assert entry.metadata == {"page": 3}


# ---------------------------------------------------------------------------
# Tests: CacheStats
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_default_values(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.current_size_bytes == 0
        assert stats.current_entries == 0
        assert stats.max_size_bytes == 0
        assert stats.max_entries == 0

    def test_hit_rate_zero_lookups(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_mixed(self):
        stats = CacheStats(hits=3, misses=7)
        assert stats.hit_rate == pytest.approx(0.3)

    def test_to_dict_includes_hit_rate(self):
        stats = CacheStats(hits=1, misses=1, max_size_bytes=100, max_entries=10)
        d = stats.to_dict()
        assert d["hit_rate"] == pytest.approx(0.5)
        assert d["hits"] == 1
        assert d["misses"] == 1
        assert d["max_size_bytes"] == 100
        assert d["max_entries"] == 10

    def test_to_dict_keys(self):
        d = CacheStats().to_dict()
        expected_keys = {
            "hits", "misses", "evictions",
            "current_size_bytes", "current_entries",
            "max_size_bytes", "max_entries", "hit_rate",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: PageCache — construction
# ---------------------------------------------------------------------------


class TestPageCacheConstruction:
    def test_default_construction(self):
        cache = PageCache()
        stats = cache.get_stats()
        assert stats.max_size_bytes == 536_870_912
        assert stats.max_entries == 1024
        assert stats.current_entries == 0

    def test_custom_construction(self):
        cache = PageCache(max_size_bytes=1024, max_entries=4, default_ttl=30.0)
        stats = cache.get_stats()
        assert stats.max_size_bytes == 1024
        assert stats.max_entries == 4


# ---------------------------------------------------------------------------
# Tests: PageCache — put / get basics
# ---------------------------------------------------------------------------


class TestPageCachePutGet:
    def test_put_and_get(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        assert cache.put("p1", b"page-one-data") is True
        result = cache.get("p1")
        assert result == b"page-one-data"

    def test_get_missing_returns_none(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        assert cache.get("nonexistent") is None

    def test_put_updates_existing_key(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"v1")
        cache.put("p1", b"version-two")
        assert cache.get("p1") == b"version-two"
        stats = cache.get_stats()
        assert stats.current_entries == 1

    def test_put_rejects_oversized_entry(self):
        cache = PageCache(max_size_bytes=10, max_entries=10)
        result = cache.put("big", b"x" * 20)
        assert result is False
        assert cache.get("big") is None

    def test_put_with_metadata(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data", metadata={"dpi": 300})
        assert cache.get_metadata("p1") == {"dpi": 300}

    def test_put_with_custom_ttl(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10, default_ttl=0)
        cache.put("p1", b"data", ttl=0.01)
        # Entry is alive initially
        assert cache.get("p1") == b"data"
        time.sleep(0.02)
        # Now expired
        assert cache.get("p1") is None


# ---------------------------------------------------------------------------
# Tests: PageCache — eviction
# ---------------------------------------------------------------------------


class TestPageCacheEviction:
    def test_evicts_when_max_entries_reached(self):
        cache = PageCache(max_size_bytes=10_000, max_entries=2)
        cache.put("a", b"aaa")
        cache.put("b", b"bbb")
        # This should evict the LRU entry ('a')
        cache.put("c", b"ccc")
        assert cache.get("a") is None
        assert cache.get("b") == b"bbb"
        assert cache.get("c") == b"ccc"
        stats = cache.get_stats()
        assert stats.evictions >= 1

    def test_evicts_when_max_size_bytes_reached(self):
        cache = PageCache(max_size_bytes=10, max_entries=100)
        cache.put("a", b"12345")  # 5 bytes
        cache.put("b", b"12345")  # 5 bytes — now at 10
        # Adding 6 bytes forces eviction of at least one entry
        cache.put("c", b"123456")
        assert cache.contains("c") is True
        stats = cache.get_stats()
        assert stats.evictions >= 1
        assert stats.current_size_bytes <= 10

    def test_lru_evicts_least_recently_used(self):
        cache = PageCache(max_size_bytes=10_000, max_entries=3)
        # Use time.sleep to ensure distinct timestamps for LRU ordering
        cache.put("a", b"1")
        import time as _time
        _time.sleep(0.01)
        cache.put("b", b"2")
        _time.sleep(0.01)
        cache.put("c", b"3")
        _time.sleep(0.01)
        # Access 'a' so it is no longer LRU
        cache.get("a")
        _time.sleep(0.01)
        # Evict should remove 'b' (oldest last_accessed)
        cache.put("d", b"4")
        assert cache.contains("a") is True
        assert cache.contains("b") is False
        assert cache.contains("c") is True
        assert cache.contains("d") is True


# ---------------------------------------------------------------------------
# Tests: PageCache — get behaviour
# ---------------------------------------------------------------------------


class TestPageCacheGetBehaviour:
    def test_get_updates_last_accessed(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data")
        before = time.time()
        time.sleep(0.01)
        cache.get("p1")
        # Entry's last_accessed should be updated beyond 'before'
        with cache._lock:
            entry = cache._entries["p1"]
            assert entry.last_accessed >= before

    def test_get_increments_access_count(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data")
        cache.get("p1")
        cache.get("p1")
        cache.get("p1")
        with cache._lock:
            assert cache._entries["p1"].access_count == 3

    def test_get_expired_returns_none_and_removes(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("p1") is None
        assert cache.contains("p1") is False
        stats = cache.get_stats()
        assert stats.current_entries == 0


# ---------------------------------------------------------------------------
# Tests: PageCache — contains
# ---------------------------------------------------------------------------


class TestPageCacheContains:
    def test_contains_present(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data")
        assert cache.contains("p1") is True

    def test_contains_absent(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        assert cache.contains("nope") is False

    def test_contains_expired(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data", ttl=0.01)
        time.sleep(0.02)
        assert cache.contains("p1") is False


# ---------------------------------------------------------------------------
# Tests: PageCache — remove
# ---------------------------------------------------------------------------


class TestPageCacheRemove:
    def test_remove_existing(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data")
        assert cache.remove("p1") is True
        assert cache.get("p1") is None
        stats = cache.get_stats()
        assert stats.current_entries == 0
        assert stats.current_size_bytes == 0

    def test_remove_nonexistent(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        assert cache.remove("nope") is False


# ---------------------------------------------------------------------------
# Tests: PageCache — get_metadata
# ---------------------------------------------------------------------------


class TestPageCacheGetMetadata:
    def test_get_metadata_present(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data", metadata={"page": 1, "dpi": 300})
        meta = cache.get_metadata("p1")
        assert meta == {"page": 1, "dpi": 300}

    def test_get_metadata_absent(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        assert cache.get_metadata("nope") is None

    def test_get_metadata_expired(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data", metadata={"x": 1}, ttl=0.01)
        time.sleep(0.02)
        assert cache.get_metadata("p1") is None

    def test_get_metadata_returns_copy(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data", metadata={"page": 1})
        meta = cache.get_metadata("p1")
        meta["page"] = 99
        assert cache.get_metadata("p1") == {"page": 1}


# ---------------------------------------------------------------------------
# Tests: PageCache — get_stats
# ---------------------------------------------------------------------------


class TestPageCacheStats:
    def test_stats_initial(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.current_entries == 0
        assert stats.current_size_bytes == 0

    def test_stats_tracks_hits(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"data")
        cache.get("p1")
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 0

    def test_stats_tracks_misses(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.get("nope")
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 1

    def test_stats_tracks_evictions(self):
        cache = PageCache(max_size_bytes=10_000, max_entries=1)
        cache.put("a", b"1")
        cache.put("b", b"2")
        stats = cache.get_stats()
        assert stats.evictions == 1

    def test_stats_current_size(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("p1", b"12345")
        cache.put("p2", b"abc")
        stats = cache.get_stats()
        assert stats.current_size_bytes == 8
        assert stats.current_entries == 2


# ---------------------------------------------------------------------------
# Tests: PageCache — clear
# ---------------------------------------------------------------------------


class TestPageCacheClear:
    def test_clear_removes_all(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("a", b"1")
        cache.put("b", b"2")
        cache.get("a")
        cache.get("missing")
        cache.clear()
        stats = cache.get_stats()
        assert stats.current_entries == 0
        assert stats.current_size_bytes == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_clear_allows_reuse(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("a", b"1")
        cache.clear()
        cache.put("a", b"new")
        assert cache.get("a") == b"new"


# ---------------------------------------------------------------------------
# Tests: PageCache — keys
# ---------------------------------------------------------------------------


class TestPageCacheKeys:
    def test_keys_empty(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        assert cache.keys() == []

    def test_keys_returns_all(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("a", b"1")
        cache.put("b", b"2")
        cache.put("c", b"3")
        assert sorted(cache.keys()) == ["a", "b", "c"]

    def test_keys_prunes_expired(self):
        cache = PageCache(max_size_bytes=1024, max_entries=10)
        cache.put("live", b"data", ttl=0)
        cache.put("dead", b"data", ttl=0.01)
        time.sleep(0.02)
        result = cache.keys()
        assert "live" in result
        assert "dead" not in result


# ---------------------------------------------------------------------------
# Tests: PageCache — thread safety
# ---------------------------------------------------------------------------


class TestPageCacheThreadSafety:
    def test_concurrent_put_get(self):
        cache = PageCache(max_size_bytes=100_000, max_entries=200)
        errors: list[str] = []
        barrier = threading.Barrier(10)

        def writer(thread_id: int) -> None:
            barrier.wait()
            for i in range(20):
                key = f"t{thread_id}-{i}"
                cache.put(key, f"data-{thread_id}-{i}".encode())

        def reader(thread_id: int) -> None:
            barrier.wait()
            for i in range(20):
                key = f"t{thread_id}-{i}"
                # May or may not be present yet — should never raise
                try:
                    cache.get(key)
                except Exception as exc:
                    errors.append(str(exc))

        threads = []
        for tid in range(5):
            threads.append(threading.Thread(target=writer, args=(tid,)))
            threads.append(threading.Thread(target=reader, args=(tid,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        # Verify cache is internally consistent
        stats = cache.get_stats()
        assert stats.current_entries >= 0
        assert stats.current_size_bytes >= 0
