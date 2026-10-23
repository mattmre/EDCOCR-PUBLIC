"""Immutable source lease safety and concurrency tests."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlencode

import pytest

from coordinator.jobs.source_cache import (
    ImmutableSourceCache,
    SourceIdentity,
    _assert_presigned_unexpired,
    lease_presigned_source,
)
from coordinator.jobs.storage import CachedS3Backend


def _identity(data: bytes, *, revision: str = "v1", job: str = "j1") -> SourceIdentity:
    return SourceIdentity("tenant-a", job, 1, f"jobs/{job}/source/input.pdf", hashlib.sha256(data).hexdigest(), revision, len(data))


def test_concurrent_same_source_fetches_once_and_shares_path(tmp_path: Path) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"))
    data = b"immutable pdf bytes"
    identity = _identity(data)
    starts = threading.Barrier(6)
    fetch_count = 0
    lock = threading.Lock()
    paths: list[str] = []

    def fetch(path: str) -> None:
        nonlocal fetch_count
        with lock:
            fetch_count += 1
        time.sleep(0.03)
        Path(path).write_bytes(data)

    def read() -> bytes:
        starts.wait(timeout=3)
        with cache.lease(identity, fetch, lambda: None) as path:
            paths.append(path)
            return Path(path).read_bytes()

    with ThreadPoolExecutor(max_workers=6) as executor:
        assert list(executor.map(lambda _: read(), range(6))) == [data] * 6
    assert fetch_count == 1
    assert len(set(paths)) == 1


def test_diagnostic_counters_measure_one_successful_fetch_and_warm_reuses(tmp_path: Path) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"), diagnostics_enabled=True)
    data = b"measured source payload"
    identity = _identity(data)
    with ThreadPoolExecutor(max_workers=4) as executor:
        def read(_index: int) -> bytes:
            with cache.lease(identity, lambda path: Path(path).write_bytes(data), lambda: None) as path:
                return Path(path).read_bytes()

        assert list(executor.map(read, range(4))) == [data] * 4
    assert cache.diagnostic_counters() == {
        "leases": 4,
        "cache_hits": 3,
        "successful_fetches": 1,
        "successful_fetch_bytes": len(data),
        "cache_layer_copy_bytes": 0,
        "source_http_get_attempts": 0,
        "source_http_get_success_bytes": 0,
    }


def test_authority_checked_on_warm_hit_and_after_wait(tmp_path: Path) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"))
    data = b"source"
    identity = _identity(data)
    calls = 0

    def authorize() -> None:
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise PermissionError("expired capability")

    with cache.lease(identity, lambda path: Path(path).write_bytes(data), authorize):
        pass
    with pytest.raises(PermissionError, match="expired"):
        with cache.lease(identity, lambda _: pytest.fail("warm hit fetched"), authorize):
            pass
    assert calls == 3


def test_revision_change_refetches_and_digest_conflict_fails(tmp_path: Path) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"))
    old = b"old"
    new = b"new"
    with cache.lease(_identity(old, revision="v1"), lambda path: Path(path).write_bytes(old), lambda: None):
        pass
    with cache.lease(_identity(new, revision="v2"), lambda path: Path(path).write_bytes(new), lambda: None) as path:
        assert Path(path).read_bytes() == new
    with pytest.raises(RuntimeError, match="digest conflicts"):
        with cache.lease(_identity(old, revision="v3"), lambda path: Path(path).write_bytes(new), lambda: None):
            pass


def test_active_reader_prevents_overcommit_then_frees_capacity(tmp_path: Path) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"), max_size_bytes=8, max_capacity_wait_seconds=0.1)
    first = b"aaaaaa"
    second = b"bbbbbb"
    with cache.lease(_identity(first, job="j1"), lambda path: Path(path).write_bytes(first), lambda: None) as held:
        with pytest.raises(RuntimeError, match="capacity exhausted"):
            with cache.lease(_identity(second, job="j2"), lambda path: Path(path).write_bytes(second), lambda: None):
                pass
        assert Path(held).read_bytes() == first
    with cache.lease(_identity(second, job="j2"), lambda path: Path(path).write_bytes(second), lambda: None) as path:
        assert Path(path).read_bytes() == second


def test_unrelated_source_fetches_overlap(tmp_path: Path) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"), max_fetches=2)
    barrier = threading.Barrier(2, timeout=3)

    def read(job: str) -> bytes:
        data = job.encode()

        def fetch(path: str) -> None:
            barrier.wait()
            Path(path).write_bytes(data)

        with cache.lease(_identity(data, job=job), fetch, lambda: None) as path:
            return Path(path).read_bytes()

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert set(executor.map(read, ["j1", "j2"])) == {b"j1", b"j2"}


def test_cache_refuses_linked_payload_and_unsafe_key(tmp_path: Path) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"))
    data = b"source"
    identity = _identity(data)
    blob, _, _ = cache._paths(identity)
    outside = tmp_path / "outside"
    outside.write_bytes(data)
    os.symlink(outside, blob)
    with pytest.raises(RuntimeError, match="unsafe"):
        with cache.lease(identity, lambda _: None, lambda: None):
            pass
    with pytest.raises(ValueError, match="canonical"):
        SourceIdentity("tenant", "job", 1, "jobs/../other/file", identity.sha256, "v1", len(data))


@pytest.mark.timeout(2)
@pytest.mark.parametrize("which", ["blob", "manifest", "lock"])
def test_cache_refuses_fifo_without_blocking(tmp_path: Path, which: str) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"))
    data = b"source"
    identity = _identity(data)
    path = dict(zip(("blob", "manifest", "lock"), cache._paths(identity)))[which]
    os.mkfifo(path)
    with pytest.raises(RuntimeError, match="unsafe"):
        with cache.lease(identity, lambda _: pytest.fail("unsafe path fetched"), lambda: None):
            pass


def test_presigned_expiry_guard() -> None:
    now = datetime.now(timezone.utc)
    fresh = urlencode({"X-Amz-Date": now.strftime("%Y%m%dT%H%M%SZ"), "X-Amz-Expires": "60"})
    expired = urlencode({"X-Amz-Date": (now - timedelta(minutes=2)).strftime("%Y%m%dT%H%M%SZ"), "X-Amz-Expires": "60"})
    _assert_presigned_unexpired(f"https://s3.example/jobs/j1/source/input.pdf?{fresh}")
    with pytest.raises(ValueError, match="expired"):
        _assert_presigned_unexpired(f"https://s3.example/jobs/j1/source/input.pdf?{expired}")
    with pytest.raises(ValueError, match="unavailable"):
        _assert_presigned_unexpired("https://s3.example/jobs/j1/source/input.pdf")


def test_sdk_adapter_reuses_bytes_and_checks_revision_on_warm_hit(tmp_path: Path) -> None:
    data = b"first revision"
    inner = Mock()
    inner.endpoint = "https://s3.example"
    inner.bucket = "bucket"
    inner.region = "us-east-1"
    inner.object_revision.return_value = ("v1", '"etag-1"', len(data))
    inner.download_file.side_effect = lambda _key, path, **_kwargs: Path(path).write_bytes(data)
    backend = CachedS3Backend(inner, str(tmp_path / "cache"))
    identity = _identity(data, revision=backend.source_revision("jobs/j1/source/input.pdf"))
    checks = 0

    def authorize() -> None:
        nonlocal checks
        checks += 1

    with backend.lease_source(identity, authorize=authorize) as first:
        assert Path(first).read_bytes() == data
    with backend.lease_source(identity, authorize=authorize) as second:
        assert second == first
    assert inner.download_file.call_count == 1
    assert checks == 4  # Entry and post-wait checks on both leases.
    inner.object_revision.return_value = ("v2", '"etag-2"', len(data))
    with pytest.raises(RuntimeError, match="revision changed"):
        with backend.lease_source(identity, authorize=authorize):
            pass


def test_presigned_adapter_reuses_bytes_but_rechecks_url_and_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from coordinator.jobs import presigned_io

    data = b"presigned source"
    cache = ImmutableSourceCache(str(tmp_path / "cache"), diagnostics_enabled=True)
    identity = _identity(data, revision="coordinator-head-v1")
    monkeypatch.setenv("S3_PRESIGNED_ALLOWED_HOSTS", "s3.example.com")
    monkeypatch.setenv("S3_BUCKET", "owned-bucket")
    now = datetime.now(timezone.utc)

    def url(*, age_minutes: int = 0, job: str = "j1") -> str:
        query = urlencode({
            "X-Amz-Date": (now - timedelta(minutes=age_minutes)).strftime("%Y%m%dT%H%M%SZ"),
            "X-Amz-Expires": "60",
            "X-Amz-Signature": "test-signature",
        })
        return f"https://s3.example.com/owned-bucket/jobs/{job}/source/input.pdf?{query}"

    fetches = 0

    def download(_url: str, path: str, **kwargs: object) -> str:
        nonlocal fetches
        fetches += 1
        kwargs["transfer_observer"]("get_attempt", 0)
        Path(path).write_bytes(data)
        kwargs["transfer_observer"]("get_success_bytes", len(data))
        return path

    monkeypatch.setattr(presigned_io, "download_presigned", download)
    checks = 0

    def authorize() -> None:
        nonlocal checks
        checks += 1

    with lease_presigned_source(cache, identity, url(), authorize=authorize) as path:
        assert Path(path).read_bytes() == data
    with lease_presigned_source(cache, identity, url(), authorize=authorize) as path:
        assert Path(path).read_bytes() == data
    assert fetches == 1
    assert checks == 4
    counters = cache.diagnostic_counters()
    assert counters["source_http_get_attempts"] == 1
    assert counters["source_http_get_success_bytes"] == len(data)
    with pytest.raises(ValueError, match="expired"):
        with lease_presigned_source(cache, identity, url(age_minutes=2), authorize=authorize):
            pass
    with pytest.raises(ValueError, match="not bound"):
        with lease_presigned_source(cache, identity, url(job="foreign"), authorize=authorize):
            pass
    assert fetches == 1


def test_lease_body_exception_releases_reader_and_budget(tmp_path: Path) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"), max_size_bytes=8)
    first = b"aaaaaa"
    second = b"bbbbbb"
    with pytest.raises(ValueError, match="page failed"):
        with cache.lease(_identity(first, job="j1"), lambda path: Path(path).write_bytes(first), lambda: None):
            raise ValueError("page failed")
    assert not list((tmp_path / "cache").glob(".reserve-*"))
    with cache.lease(_identity(second, job="j2"), lambda path: Path(path).write_bytes(second), lambda: None) as path:
        assert Path(path).read_bytes() == second


def test_mutation_during_lease_fails_closed(tmp_path: Path) -> None:
    cache = ImmutableSourceCache(str(tmp_path / "cache"))
    data = b"source"
    with pytest.raises(RuntimeError, match="changed during read"):
        with cache.lease(_identity(data), lambda path: Path(path).write_bytes(data), lambda: None) as path:
            Path(path).write_bytes(b"mutate")
            with pytest.raises(RuntimeError, match="changed during read"):
                path.check()


def _forked_lease_worker(cache_dir: str, identity: SourceIdentity, data: bytes, start, output, fetch_log: str) -> None:
    cache = ImmutableSourceCache(cache_dir, max_size_bytes=8, max_capacity_wait_seconds=0.25)
    start.wait(timeout=3)

    def fetch(path: str) -> None:
        with open(fetch_log, "ab") as handle:
            handle.write(identity.job_id.encode() + b"\n")
        Path(path).write_bytes(data)

    try:
        with cache.lease(identity, fetch, lambda: None) as path:
            time.sleep(0.1)
            output.put(("ok", Path(path).read_bytes()))
    except Exception as exc:
        output.put(("error", str(exc)))


def _forked_slow_fetch(cache_dir: str, identity: SourceIdentity, data: bytes, entered, release, output) -> None:
    cache = ImmutableSourceCache(cache_dir, max_size_bytes=8, max_capacity_wait_seconds=1)

    def fetch(path: str) -> None:
        entered.set()
        if not release.wait(timeout=4):
            raise TimeoutError("slow fetch release timed out")
        Path(path).write_bytes(data)

    try:
        with cache.lease(identity, fetch, lambda: None):
            output.put("ok")
    except Exception as exc:
        output.put(f"error:{exc}")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX flock/fork")
def test_cross_process_singleflight_and_shared_budget(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    cache_dir = str(tmp_path / "cache")
    fetch_log = str(tmp_path / "fetch.log")
    start = context.Event()
    output = context.Queue()
    data = b"aaaaaa"
    identity = _identity(data)
    processes = [
        context.Process(target=_forked_lease_worker, args=(cache_dir, identity, data, start, output, fetch_log))
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        assert [output.get(timeout=5) for _ in processes] == [("ok", data)] * 2
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
    assert Path(fetch_log).read_bytes().splitlines() == [b"j1"]

    # Different objects cannot both reserve six bytes in an eight-byte cache.
    busy = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_forked_lease_worker,
        args=(cache_dir, _identity(b"cccccc", job="j3"), b"cccccc", busy, output, fetch_log),
    )
    cache = ImmutableSourceCache(cache_dir, max_size_bytes=8, max_capacity_wait_seconds=0.25)
    try:
        with cache.lease(identity, lambda _: pytest.fail("warm source fetched"), lambda: None):
            process.start()
            busy.set()
            assert output.get(timeout=5)[0] == "error"
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX flock/fork")
def test_cross_process_inflight_reservation_prevents_overcommit(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    cache_dir = str(tmp_path / "cache")
    entered = context.Event()
    release = context.Event()
    output = context.Queue()
    first_data = b"aaaaaa"
    slow = context.Process(
        target=_forked_slow_fetch,
        args=(cache_dir, _identity(first_data, job="j1"), first_data, entered, release, output),
    )
    try:
        slow.start()
        assert entered.wait(timeout=3)
        cache = ImmutableSourceCache(cache_dir, max_size_bytes=8, max_capacity_wait_seconds=0.15)
        second_data = b"bbbbbb"
        with pytest.raises(RuntimeError, match="capacity exhausted"):
            with cache.lease(_identity(second_data, job="j2"), lambda path: Path(path).write_bytes(second_data), lambda: None):
                pass
        assert list(Path(cache_dir).glob(".reserve-*"))
    finally:
        release.set()
        slow.join(timeout=5)
        if slow.is_alive():
            slow.terminate()
            slow.join(timeout=2)
    assert output.get(timeout=2) == "ok"
