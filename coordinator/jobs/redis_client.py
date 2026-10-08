"""Redis client construction with production-qualified Sentinel discovery.

PostgreSQL, RabbitMQ, and object storage own durable OCR state. Redis remains a
replaceable cache/result/context dependency, but every consumer must discover
the same writable primary after failover. This module is the single client
factory for Django cache, streams, and context storage.
"""

from __future__ import annotations

import os
from functools import cached_property
from urllib.parse import unquote, urlparse

import redis
from django.core.cache.backends.redis import RedisCache, RedisCacheClient


class RedisSentinelConfigurationError(ValueError):
    """Raised when Sentinel mode is requested without usable discovery data."""


def sentinel_nodes(raw: str | None = None) -> list[tuple[str, int]]:
    """Parse the bounded comma-separated Sentinel host list."""
    value = raw if raw is not None else os.environ.get("REDIS_SENTINEL_NODES", "")
    nodes: list[tuple[str, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        host, separator, port_text = item.rpartition(":")
        if not separator or not host or not port_text.isdigit():
            raise RedisSentinelConfigurationError(
                "REDIS_SENTINEL_NODES entries must use host:port syntax"
            )
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise RedisSentinelConfigurationError(
                "REDIS_SENTINEL_NODES contains an invalid port"
            )
        nodes.append((host, port))
    if len(nodes) > 32:
        raise RedisSentinelConfigurationError(
            "REDIS_SENTINEL_NODES exceeds the 32-node safety bound"
        )
    return nodes


def create_redis_client(redis_url: str, **options):
    """Return a direct Redis client or Sentinel-discovered writable primary."""
    master_name = os.environ.get("REDIS_SENTINEL_MASTER_NAME", "").strip()
    if not master_name:
        return redis.from_url(redis_url, **options)

    nodes = sentinel_nodes()
    if not nodes:
        raise RedisSentinelConfigurationError(
            "REDIS_SENTINEL_MASTER_NAME requires REDIS_SENTINEL_NODES"
        )

    parsed = urlparse(redis_url)
    if parsed.scheme not in {"redis", "rediss", "sentinel"}:
        raise RedisSentinelConfigurationError(
            "Sentinel Redis URL must use redis, rediss, or sentinel scheme"
        )
    try:
        db = int((parsed.path or "/0").lstrip("/") or "0")
    except ValueError as exc:
        raise RedisSentinelConfigurationError("Redis database must be numeric") from exc

    sentinel_password = os.environ.get("REDIS_SENTINEL_PASSWORD", "").strip()
    sentinel_kwargs = {
        "socket_connect_timeout": float(
            os.environ.get("REDIS_SENTINEL_CONNECT_TIMEOUT_SECONDS", "2")
        ),
        "socket_timeout": float(
            os.environ.get("REDIS_SENTINEL_SOCKET_TIMEOUT_SECONDS", "5")
        ),
    }
    if sentinel_password:
        sentinel_kwargs["password"] = sentinel_password

    sentinel = redis.sentinel.Sentinel(nodes, sentinel_kwargs=sentinel_kwargs)
    client_options = {
        "db": db,
        "username": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password) if parsed.password else None,
        "socket_connect_timeout": sentinel_kwargs["socket_connect_timeout"],
        "socket_timeout": sentinel_kwargs["socket_timeout"],
        **options,
    }
    if parsed.scheme == "rediss":
        client_options["ssl"] = True
    return sentinel.master_for(master_name, **client_options)


class SentinelRedisCacheClient(RedisCacheClient):
    """Django RedisCache adapter that always resolves Sentinel's primary."""

    @cached_property
    def _sentinel_client(self):
        return create_redis_client(self._servers[0], **self._pool_options)

    def get_client(self, key=None, *, write=False):
        del key, write
        return self._sentinel_client


class SentinelRedisCache(RedisCache):
    """Django cache backend for the shared Sentinel client factory."""

    def __init__(self, server, params):
        super().__init__(server, params)
        self._class = SentinelRedisCacheClient
