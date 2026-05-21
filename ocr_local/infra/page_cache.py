"""Memory-mapped page caching for the OCR pipeline.

Provides an LRU page cache that avoids re-processing pages already seen
during a pipeline run.  The :class:`PageCache` supports configurable
maximum size (both byte-count and entry-count), optional per-entry TTL,
and pluggable eviction strategies.

Thread safety is guaranteed by a :class:`threading.Lock` around every
mutating operation.

Environment Variables:
    PAGE_CACHE_MAX_SIZE_BYTES (int):
        Maximum aggregate size of cached data in bytes.
        Default: ``536870912`` (512 MiB).
    PAGE_CACHE_MAX_ENTRIES (int):
        Maximum number of entries in the cache.
        Default: ``1024``.
    PAGE_CACHE_DEFAULT_TTL (float):
        Default time-to-live for cache entries in seconds.
        ``0`` means no expiry.  Default: ``0``.
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CacheStrategy(Enum):
    """Supported cache eviction strategies."""

    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single cached page.

    Attributes:
        key: Unique identifier for the cached page.
        data: Raw page bytes.
        size_bytes: Length of *data* in bytes.
        created_at: Unix timestamp when the entry was created.
        last_accessed: Unix timestamp of the most recent access.
        access_count: Number of times the entry has been read.
        ttl_seconds: Time-to-live in seconds; ``0`` disables expiry.
        metadata: Arbitrary metadata attached to the entry.
    """

    key: str
    data: bytes
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Return ``True`` if the entry has exceeded its TTL."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) >= self.ttl_seconds


@dataclass
class CacheStats:
    """Snapshot of cache performance counters.

    Attributes:
        hits: Number of successful cache lookups.
        misses: Number of cache misses.
        evictions: Number of entries evicted to make room.
        current_size_bytes: Total bytes currently stored.
        current_entries: Number of entries currently stored.
        max_size_bytes: Configured maximum byte capacity.
        max_entries: Configured maximum entry count.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size_bytes: int = 0
    current_entries: int = 0
    max_size_bytes: int = 0
    max_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a float in [0.0, 1.0].

        Returns ``0.0`` when no lookups have been performed.
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> dict:
        """Serialize the stats to a plain dictionary."""
        d = asdict(self)
        d["hit_rate"] = self.hit_rate
        return d


# ---------------------------------------------------------------------------
# PageCache
# ---------------------------------------------------------------------------


class PageCache:
    """Thread-safe LRU page cache with byte- and entry-count limits.

    Parameters:
        max_size_bytes: Maximum aggregate data size in bytes.
        max_entries: Maximum number of entries.
        default_ttl: Default TTL for entries (seconds).  ``0`` = no expiry.
        strategy: Eviction strategy (currently only :attr:`CacheStrategy.LRU`
            is implemented; the parameter is accepted for forward-compatibility).
    """

    def __init__(
        self,
        max_size_bytes: int = 536_870_912,
        max_entries: int = 1024,
        default_ttl: float = 0.0,
        strategy: CacheStrategy = CacheStrategy.LRU,
    ) -> None:
        self._max_size_bytes = max_size_bytes
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._strategy = strategy

        self._entries: dict[str, CacheEntry] = {}
        self._current_size_bytes: int = 0

        # Counters
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        self._lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def put(
        self,
        key: str,
        data: bytes,
        metadata: Optional[dict] = None,
        ttl: Optional[float] = None,
    ) -> bool:
        """Insert or update an entry in the cache.

        Returns ``True`` on success, ``False`` if the single entry exceeds
        the maximum cache size.
        """
        size = len(data)
        if size > self._max_size_bytes:
            logger.warning(
                "Cache entry %s (%d bytes) exceeds max cache size (%d bytes); skipping",
                key,
                size,
                self._max_size_bytes,
            )
            return False

        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()

        with self._lock:
            # If updating an existing key, remove old size first
            if key in self._entries:
                self._current_size_bytes -= self._entries[key].size_bytes

            # Evict until there is room
            while (
                self._entries
                and key not in self._entries
                and len(self._entries) >= self._max_entries
            ):
                self._evict()

            while (
                self._entries
                and (self._current_size_bytes + size) > self._max_size_bytes
            ):
                self._evict()

            entry = CacheEntry(
                key=key,
                data=data,
                size_bytes=size,
                created_at=now,
                last_accessed=now,
                access_count=0,
                ttl_seconds=effective_ttl,
                metadata=metadata if metadata is not None else {},
            )
            self._entries[key] = entry
            self._current_size_bytes += size

            logger.debug("Cache PUT %s (%d bytes)", key, size)
            return True

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve cached data by key.

        Returns ``None`` on a miss or if the entry has expired.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                self._remove_entry(key)
                self._misses += 1
                return None

            entry.last_accessed = time.time()
            entry.access_count += 1
            self._hits += 1
            return entry.data

    def contains(self, key: str) -> bool:
        """Return ``True`` if *key* is present and not expired."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                self._remove_entry(key)
                return False
            return True

    def remove(self, key: str) -> bool:
        """Remove an entry.  Returns ``True`` if the key existed."""
        with self._lock:
            if key not in self._entries:
                return False
            self._remove_entry(key)
            return True

    def get_metadata(self, key: str) -> Optional[dict]:
        """Return metadata for *key*, or ``None`` if not present."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                self._remove_entry(key)
                return None
            return dict(entry.metadata)

    def get_stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                current_size_bytes=self._current_size_bytes,
                current_entries=len(self._entries),
                max_size_bytes=self._max_size_bytes,
                max_entries=self._max_entries,
            )

    def clear(self) -> None:
        """Remove all entries and reset counters."""
        with self._lock:
            self._entries.clear()
            self._current_size_bytes = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.debug("Cache cleared")

    def keys(self) -> list[str]:
        """Return a list of all non-expired keys in the cache."""
        with self._lock:
            expired = [k for k, v in self._entries.items() if v.is_expired]
            for k in expired:
                self._remove_entry(k)
            return list(self._entries.keys())

    # -- internal helpers ---------------------------------------------------

    def _evict(self) -> None:
        """Evict the least-recently-used entry (by *last_accessed*)."""
        if not self._entries:
            return
        lru_key = min(self._entries, key=lambda k: self._entries[k].last_accessed)
        self._remove_entry(lru_key)
        self._evictions += 1
        logger.debug("Cache EVICT %s", lru_key)

    def _remove_entry(self, key: str) -> None:
        """Remove *key* from the cache and adjust the byte counter."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._current_size_bytes -= entry.size_bytes
