import json

import pytest

from jobs.redis_streams import RedisStreamClient, validate_stream_name


class FakeRedis:
    def __init__(self):
        self.stream_entries: list[tuple[str, dict[str, object]]] = []
        self.cache: dict[str, bytes] = {}

    def xadd(self, stream_name, payload):
        self.stream_entries.append((stream_name, payload))
        return "1710000000000-0"

    def xread(self, streams, count=1, block=0):
        return [
            ("stream", [("1710000000000-0", {"chunk_id": "doc:1"})]),
            streams,
            count,
            block,
        ]

    def set(self, key, value, ex=None):
        if isinstance(value, str):
            value = value.encode("utf-8")
        self.cache[key] = value
        self.cache[f"{key}:ttl"] = str(ex).encode("utf-8")
        return True

    def get(self, key):
        return self.cache.get(key)

    def delete(self, key):
        return 1 if self.cache.pop(key, None) is not None else 0


def test_publish_chunk_serializes_non_scalar_fields(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("jobs.redis_streams.create_redis_client", lambda url: fake)

    client = RedisStreamClient("redis://redis:6379/0")
    msg_id = client.publish_chunk(
        "ocr:ctx-stream",
        "job-1:3",
        {"page": 3, "layout": {"type": "table"}, "score": 0.91},
    )

    assert msg_id == "1710000000000-0"
    assert fake.stream_entries == [
        (
            "ocr:ctx-stream",
            {
                "page": 3,
                "layout": json.dumps({"type": "table"}),
                "score": 0.91,
                "chunk_id": "job-1:3",
            },
        )
    ]


def test_read_chunk_passes_stream_arguments(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("jobs.redis_streams.create_redis_client", lambda url: fake)

    client = RedisStreamClient("redis://redis:6379/0")
    result = client.read_chunk("ocr:ctx-stream", last_id="1-0", count=2, block=50)

    assert result[1] == {"ocr:ctx-stream": "1-0"}
    assert result[2] == 2
    assert result[3] == 50


def test_chunk_cache_round_trip(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("jobs.redis_streams.create_redis_client", lambda url: fake)

    client = RedisStreamClient("redis://redis:6379/0")

    assert client.set_chunk_cache("job-1:3", '{"text":"hello"}', expire=120) is True
    assert client.get_chunk_cache("job-1:3") == '{"text":"hello"}'
    assert client.delete_chunk_cache("job-1:3") == 1
    assert client.get_chunk_cache("job-1:3") is None


def test_validate_stream_name_refuses_freeform():
    with pytest.raises(ValueError, match="PH2-16"):
        validate_stream_name("ctx-stream")
    with pytest.raises(ValueError, match="PH2-16"):
        validate_stream_name("../evil")
    assert validate_stream_name("ocr:job-context:abc") == "ocr:job-context:abc"


def test_publish_refuses_invalid_stream(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("jobs.redis_streams.create_redis_client", lambda url: fake)
    client = RedisStreamClient("redis://redis:6379/0")
    with pytest.raises(ValueError, match="PH2-16"):
        client.publish_chunk("free-form", "c1", {"x": 1})
