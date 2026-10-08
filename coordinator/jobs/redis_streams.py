import json
import logging
import re
from typing import Any, Dict, List, Optional

from django.conf import settings

from .redis_client import create_redis_client

logger = logging.getLogger(__name__)

# PH2-16: free-form stream names rejected — only ocr:* product streams.
_STREAM_NAME_RE = re.compile(r"^ocr:[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
# LT-03 / S2-06: tenant-bound streams are ocr:{tenant_id}:... (tenant segment
# required under multi-tenant / production unless STREAM_ALLOW_GLOBAL=true).
_TENANT_STREAM_RE = re.compile(
    r"^ocr:(?P<tenant>[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}):"
    r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$"
)


def validate_stream_name(
    stream_name: str,
    *,
    tenant_id: str | None = None,
    require_tenant: bool | None = None,
) -> str:
    """Return *stream_name* if allowed; raise ValueError otherwise (PH2-16 / LT-03).

    When *require_tenant* is True (or auto under DEPLOYMENT_ENV=production /
    ENABLE_MULTITENANCY without STREAM_ALLOW_GLOBAL), the name must be
    ``ocr:{tenant_id}:...`` and *tenant_id* must match the embedded segment.
    """
    import os

    name = (stream_name or "").strip()
    if not name or not _STREAM_NAME_RE.match(name):
        raise ValueError(
            f"Invalid Redis stream name {stream_name!r}; must match "
            r"'ocr:[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}' (PH2-16)"
        )
    if require_tenant is None:
        allow_global = (
            os.environ.get("STREAM_ALLOW_GLOBAL") or ""
        ).strip().lower() in (
            "1",
            "true",
            "yes",
        )
        dep = (os.environ.get("DEPLOYMENT_ENV") or "").strip().lower()
        mt = (os.environ.get("ENABLE_MULTITENANCY") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        require_tenant = (dep in ("production", "prod") or mt) and not allow_global
    if require_tenant:
        m = _TENANT_STREAM_RE.match(name)
        if not m:
            raise ValueError(
                f"Invalid Redis stream name {stream_name!r}; tenant-bound form "
                r"required: 'ocr:{{tenant_id}}:...' (LT-03/S2-06)"
            )
        if tenant_id is not None and str(tenant_id).strip():
            if m.group("tenant") != str(tenant_id).strip():
                raise ValueError(
                    f"Redis stream tenant segment {m.group('tenant')!r} does not "
                    f"match tenant_id={tenant_id!r} (LT-03/S2-06)"
                )
    return name


class RedisStreamClient:
    """
    Redis Streams client for temporary storage/queue layer.
    Allows workers to publish/subscribe or get/set page chunks using Redis.
    """

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize connection to Redis. Uses REDIS_URL from settings if not provided."""
        url = redis_url or getattr(settings, "REDIS_URL", "redis://redis:6379/0")
        self.redis = create_redis_client(url)
        self.last_cache_error = None

    def publish_chunk(
        self,
        stream_name: str,
        chunk_id: str,
        data: Dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> str:
        """
        Publish a page chunk to a Redis Stream.

        Args:
            stream_name: The name of the Redis stream.
            chunk_id: The identifier for this chunk (e.g. document_id:page_num).
            data: The payload data to publish.
            tenant_id: Optional tenant bind for LT-03 stream ACL (also read
                from ``data['tenant_id']`` when present).

        Returns:
            The Redis stream message ID.
        """
        tid = tenant_id
        if tid is None and isinstance(data, dict):
            tid = data.get("tenant_id")  # type: ignore[assignment]
        stream_name = validate_stream_name(
            stream_name, tenant_id=str(tid) if tid is not None else None
        )
        # Redis streams accept flat dictionaries with string/byte values.
        payload = {}
        for k, v in data.items():
            if isinstance(v, (str, bytes, int, float)):
                payload[k] = v
            else:
                payload[k] = json.dumps(v)

        payload["chunk_id"] = chunk_id

        try:
            return self.redis.xadd(stream_name, payload)
        except Exception as exc:
            logger.error(
                "Failed to publish chunk %s to stream %s: %s",
                chunk_id,
                stream_name,
                exc,
            )
            raise

    def read_chunk(
        self, stream_name: str, last_id: str = "0-0", count: int = 1, block: int = 0
    ) -> List[Any]:
        """
        Read page chunks from a Redis Stream.

        Args:
            stream_name: The name of the Redis stream.
            last_id: The ID to read after (default "0-0" for beginning).
            count: Number of messages to read.
            block: Milliseconds to block. 0 means block indefinitely if not found. None means no block.

        Returns:
            The raw stream data from Redis.
        """
        stream_name = validate_stream_name(stream_name)
        try:
            return self.redis.xread({stream_name: last_id}, count=count, block=block)
        except Exception as exc:
            logger.error("Failed to read from stream %s: %s", stream_name, exc)
            raise

    def set_chunk_cache(self, chunk_id: str, data: str, expire: int = 3600) -> bool:
        """
        Set an ephemeral page chunk in Redis cache (standard K/V).
        Useful for retrieving a 5-page context quickly.

        Args:
            chunk_id: The unique chunk identifier.
            data: The chunk data as a string (often JSON).
            expire: TTL in seconds.

        Returns:
            True if set successfully.
        """
        try:
            return self.redis.set(f"chunk:{chunk_id}", data, ex=expire)
        except Exception as exc:
            logger.error("Failed to cache chunk %s: %s", chunk_id, exc)
            raise

    def get_chunk_cache(self, chunk_id: str) -> Optional[str]:
        """
        Get an ephemeral page chunk from Redis cache.

        Args:
            chunk_id: The unique chunk identifier.

        Returns:
            The cached data string or None if not found.
        """
        try:
            val = self.redis.get(f"chunk:{chunk_id}")
            return val.decode("utf-8") if val else None
        except Exception as exc:
            logger.error("Failed to get cached chunk %s: %s", chunk_id, exc)
            return None

    def delete_chunk_cache(self, chunk_id: str) -> int:
        """
        Delete an ephemeral page chunk from Redis cache.

        Args:
            chunk_id: The unique chunk identifier.

        Returns:
            Number of keys deleted (1 or 0).
        """
        self.last_cache_error = None
        try:
            return self.redis.delete(f"chunk:{chunk_id}")
        except Exception as exc:
            self.last_cache_error = exc
            logger.error("Failed to delete cached chunk %s: %s", chunk_id, exc)
            return 0
