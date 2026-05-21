import json
import logging
from typing import Any, Dict, List, Optional

import redis
from django.conf import settings

logger = logging.getLogger(__name__)


class RedisStreamClient:
    """
    Redis Streams client for temporary storage/queue layer.
    Allows workers to publish/subscribe or get/set page chunks using Redis.
    """

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize connection to Redis. Uses REDIS_URL from settings if not provided."""
        url = redis_url or getattr(settings, "REDIS_URL", "redis://redis:6379/0")
        self.redis = redis.from_url(url)

    def publish_chunk(self, stream_name: str, chunk_id: str, data: Dict[str, Any]) -> str:
        """
        Publish a page chunk to a Redis Stream.
        
        Args:
            stream_name: The name of the Redis stream.
            chunk_id: The identifier for this chunk (e.g. document_id:page_num).
            data: The payload data to publish.
            
        Returns:
            The Redis stream message ID.
        """
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
            logger.error("Failed to publish chunk %s to stream %s: %s", chunk_id, stream_name, exc)
            raise

    def read_chunk(self, stream_name: str, last_id: str = "0-0", count: int = 1, block: int = 0) -> List[Any]:
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
        try:
            return self.redis.delete(f"chunk:{chunk_id}")
        except Exception as exc:
            logger.error("Failed to delete cached chunk %s: %s", chunk_id, exc)
            return 0
