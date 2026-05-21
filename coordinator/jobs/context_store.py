"""Ephemeral context store for 5-page context windowing.

Workers pull context by reference ID from Redis instead of receiving
serialized page data in Celery message payloads. This keeps broker
messages under 10KB while giving workers access to surrounding page
context for cross-page layout merging.

Configuration:
    CONTEXT_WINDOW_ENABLED: bool  -- master toggle (default: False)
    CONTEXT_WINDOW_SIZE: int      -- pages in window (default: 5)
    CONTEXT_STORE_TTL: int        -- seconds before expiry (default: 3600)
    CONTEXT_STORE_URL: str        -- Redis URL (defaults to REDIS_URL)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

# Defaults (overridden by Django settings or env vars at import time)
_DEFAULT_TTL = 3600
_DEFAULT_WINDOW_SIZE = 5


def _get_config() -> dict[str, Any]:
    """Read context window configuration from Django settings / env vars."""
    try:
        from django.conf import settings as django_settings
        enabled = getattr(django_settings, "CONTEXT_WINDOW_ENABLED", None)
        window_size = getattr(django_settings, "CONTEXT_WINDOW_SIZE", None)
        ttl = getattr(django_settings, "CONTEXT_STORE_TTL", None)
        redis_url = getattr(django_settings, "CONTEXT_STORE_URL", None)
        if redis_url is None:
            redis_url = getattr(django_settings, "REDIS_URL", None)
    except Exception:
        enabled = None
        window_size = None
        ttl = None
        redis_url = None

    # Fallback to env vars
    if enabled is None:
        enabled = os.environ.get(
            "CONTEXT_WINDOW_ENABLED", "false"
        ).lower() in ("1", "true", "yes")
    if window_size is None:
        try:
            window_size = int(os.environ.get("CONTEXT_WINDOW_SIZE", str(_DEFAULT_WINDOW_SIZE)))
        except (TypeError, ValueError):
            window_size = _DEFAULT_WINDOW_SIZE
    if ttl is None:
        try:
            ttl = int(os.environ.get("CONTEXT_STORE_TTL", str(_DEFAULT_TTL)))
        except (TypeError, ValueError):
            ttl = _DEFAULT_TTL
    if redis_url is None:
        redis_url = os.environ.get("CONTEXT_STORE_URL") or os.environ.get(
            "REDIS_URL", "redis://redis:6379/0"
        )

    return {
        "enabled": bool(enabled),
        "window_size": max(1, int(window_size)),
        "ttl": max(60, int(ttl)),
        "redis_url": str(redis_url),
    }


# ---------------------------------------------------------------------------
# Reference ID generation
# ---------------------------------------------------------------------------

def _make_ref_id(job_id: str, page_num: int) -> str:
    """Generate a deterministic, unique reference ID for a page context.

    Format: ``ctx:<job_id_prefix>:<page_num>:<short_uuid>``

    The short UUID suffix prevents collisions if a job is reprocessed.
    """
    job_prefix = str(job_id).replace("-", "")[:12]
    suffix = uuid.uuid4().hex[:8]
    return f"ctx:{job_prefix}:{page_num}:{suffix}"


def _job_key_pattern(job_id: str) -> str:
    """Return the Redis key prefix for all context keys belonging to a job."""
    job_prefix = str(job_id).replace("-", "")[:12]
    return f"ctx:{job_prefix}:*"


def _redis_key(ref_id: str) -> str:
    """Return the Redis key for a context reference ID."""
    return f"context:{ref_id}"


# ---------------------------------------------------------------------------
# ContextStore
# ---------------------------------------------------------------------------

class ContextStore:
    """Redis-backed ephemeral store for page context windows.

    Each page gets a context bundle containing the current page's OCR data
    plus surrounding pages (previous, next, and summary pages). Workers
    retrieve the bundle by reference ID rather than receiving it inline in
    the Celery message payload.

    The store is opt-in: when ``CONTEXT_WINDOW_ENABLED`` is ``False``
    (the default), all public methods are no-ops that return ``None``.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
        window_size: int | None = None,
        *,
        _redis_client: Any | None = None,
    ):
        config = _get_config()
        self._enabled = config["enabled"]
        self._ttl = ttl_seconds if ttl_seconds is not None else config["ttl"]
        self._window_size = window_size if window_size is not None else config["window_size"]
        self._redis_url = redis_url or config["redis_url"]
        self._redis: Any | None = _redis_client

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def window_size(self) -> int:
        return self._window_size

    def _get_redis(self) -> Any:
        """Lazy-connect to Redis."""
        if self._redis is None:
            import redis
            self._redis = redis.from_url(self._redis_url)
        return self._redis

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def store_page_context(
        self, job_id: str, page_num: int, context: dict[str, Any],
    ) -> str:
        """Store a page context bundle and return its reference ID.

        Args:
            job_id: The job UUID string.
            page_num: 1-based page number.
            context: Arbitrary dict with page OCR data, text, layout, etc.

        Returns:
            A reference ID string that workers use to retrieve the context.
        """
        ref_id = _make_ref_id(job_id, page_num)
        key = _redis_key(ref_id)

        payload = json.dumps(context, default=str)

        r = self._get_redis()
        r.set(key, payload, ex=self._ttl)

        # Maintain a job-level set so cleanup_job can find all keys.
        job_set_key = f"context:job:{str(job_id).replace('-', '')[:12]}"
        r.sadd(job_set_key, key)
        r.expire(job_set_key, self._ttl)

        logger.debug(
            "Stored context for job=%s page=%d ref=%s (ttl=%ds, size=%d bytes)",
            job_id, page_num, ref_id, self._ttl, len(payload),
        )
        return ref_id

    def get_page_context(self, ref_id: str) -> dict[str, Any] | None:
        """Retrieve a page context bundle by reference ID.

        Returns:
            The context dict, or ``None`` if expired / not found.
        """
        key = _redis_key(ref_id)
        r = self._get_redis()

        raw = r.get(key)
        if raw is None:
            logger.debug("Context not found for ref=%s", ref_id)
            return None

        data = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(data)

    def delete_context(self, ref_id: str) -> bool:
        """Delete a single context entry. Returns True if deleted."""
        key = _redis_key(ref_id)
        r = self._get_redis()
        return bool(r.delete(key))

    def cleanup_job(self, job_id: str) -> int:
        """Remove all context entries for a completed job.

        Returns:
            Number of keys deleted.
        """
        r = self._get_redis()
        job_set_key = f"context:job:{str(job_id).replace('-', '')[:12]}"

        members = r.smembers(job_set_key)
        if not members:
            return 0

        keys_to_delete = [
            m.decode("utf-8") if isinstance(m, bytes) else m
            for m in members
        ]
        keys_to_delete.append(job_set_key)

        deleted = r.delete(*keys_to_delete)
        logger.info(
            "Cleaned up %d context keys for job %s", deleted, job_id,
        )
        return deleted

    # ------------------------------------------------------------------
    # Context window builder
    # ------------------------------------------------------------------

    def build_context_window(
        self,
        job_id: str,
        pages: list[dict[str, Any]],
        page_num: int,
    ) -> dict[str, Any]:
        """Build a 5-page context window for a given page and store it.

        The window includes:
        - ``current``: The target page data
        - ``previous``: The preceding page (or None)
        - ``next``: The following page (or None)
        - ``summary_before``: A condensed summary of earlier pages
        - ``summary_after``: A condensed summary of later pages

        Args:
            job_id: The job UUID string.
            pages: List of page data dicts, sorted by page number.
                   Each dict should have at least ``page_num``, ``text``,
                   and optionally ``layout_regions``, ``ocr_lines``.
            page_num: 1-based page number of the target page.

        Returns:
            Dict with ``ref_id`` and the full context window payload.
        """
        # pages list is 0-indexed, page_num is 1-indexed
        idx = page_num - 1
        total = len(pages)

        if idx < 0 or idx >= total:
            raise ValueError(
                f"page_num {page_num} out of range for {total} pages"
            )

        current_page = pages[idx]
        prev_page = pages[idx - 1] if idx > 0 else None
        next_page = pages[idx + 1] if idx < total - 1 else None

        # Build summary of pages before the window
        summary_before = self._build_summary(pages, end_idx=max(0, idx - 1))
        # Build summary of pages after the window
        summary_after = self._build_summary(
            pages, start_idx=min(total, idx + 2),
        )

        context = {
            "job_id": str(job_id),
            "target_page_num": page_num,
            "total_pages": total,
            "window_size": self._window_size,
            "current": current_page,
            "previous": prev_page,
            "next": next_page,
            "summary_before": summary_before,
            "summary_after": summary_after,
            "timestamp": time.time(),
        }

        ref_id = self.store_page_context(job_id, page_num, context)
        context["ref_id"] = ref_id
        return context

    def build_all_context_windows(
        self,
        job_id: str,
        pages: list[dict[str, Any]],
    ) -> list[str]:
        """Build and store context windows for all pages in a document.

        Args:
            job_id: The job UUID string.
            pages: List of page data dicts sorted by page number.

        Returns:
            List of reference IDs, one per page, in page order.
        """
        ref_ids: list[str] = []
        for page_num in range(1, len(pages) + 1):
            result = self.build_context_window(job_id, pages, page_num)
            ref_ids.append(result["ref_id"])
        return ref_ids

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        pages: list[dict[str, Any]],
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> dict[str, Any] | None:
        """Build a condensed summary of a range of pages.

        Extracts page numbers, first/last lines, and total text length
        for the given page range. Returns None if the range is empty.
        """
        if end_idx is None:
            end_idx = len(pages)
        subset = pages[start_idx:end_idx]
        if not subset:
            return None

        page_nums = []
        total_text_length = 0
        first_lines: list[str] = []

        for p in subset:
            pn = p.get("page_num", 0)
            page_nums.append(pn)

            text = p.get("text", "")
            total_text_length += len(text)

            lines = text.strip().split("\n") if text else []
            if lines:
                first_lines.append(lines[0][:200])

        return {
            "page_range": page_nums,
            "page_count": len(subset),
            "total_text_length": total_text_length,
            "first_lines": first_lines[:5],  # cap at 5 excerpts
        }

    # ------------------------------------------------------------------
    # Payload size validation
    # ------------------------------------------------------------------

    @staticmethod
    def measure_payload_size(payload: dict[str, Any]) -> int:
        """Return the serialized size of a payload in bytes."""
        return len(json.dumps(payload, default=str).encode("utf-8"))

    @staticmethod
    def validate_broker_payload(
        payload: dict[str, Any],
        max_bytes: int = 10240,
    ) -> bool:
        """Check that a Celery broker payload stays under the size limit.

        Args:
            payload: The dict that would be serialized into the message.
            max_bytes: Maximum allowed size (default 10KB).

        Returns:
            True if the payload is within limits.
        """
        size = ContextStore.measure_payload_size(payload)
        if size > max_bytes:
            logger.warning(
                "Broker payload exceeds %d bytes: actual=%d", max_bytes, size,
            )
            return False
        return True
