"""Tests for ``ocr_local.translation.cache`` -- Plan B Wave M2.

Coverage:
    - Fresh load with downloads disallowed -> ModelNotCachedError
    - Fresh load with allow_download=True -> MODEL_DOWNLOADED custody
    - Cached load updates access_time without re-downloading
    - Corrupt files trigger MODEL_INTEGRITY_FAILED
    - LRU eviction frees enough bytes; pinned models exempt
    - NC-license block raises PolicyDenied + emits custody event
    - Air-gapped mode forces ``allow_download=False``
    - 4-thread concurrent get -- no race on cache_state.json
    - register_model writes manifest + integrity-verifies on next get
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ocr_local.translation.cache import (
    CachedModelInfo,
    ModelIntegrityError,
    ModelNotCachedError,
    evict_lru,
    get_translation_model_path,
    list_cached_models,
    pin_model,
    register_model,
    set_custody_chain,
    set_download_fn,
    unpin_model,
    verify_model_integrity,
)
from ocr_local.translation.custody_adapter import ReasonCode
from ocr_local.translation.policy import PolicyDenied, TenantPolicy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Isolated cache root under ``tmp_path``."""
    root = tmp_path / "cache"
    root.mkdir()
    return root


@pytest.fixture
def custody() -> MagicMock:
    """Mock custody chain with assert helpers."""
    chain = MagicMock()
    set_custody_chain(chain)
    yield chain
    set_custody_chain(None)


@pytest.fixture(autouse=True)
def _reset_download_fn():
    """Make sure tests start with no download fn registered."""
    yield
    set_download_fn(None)
    set_custody_chain(None)


def _custody_reason_codes(chain: MagicMock) -> list[str]:
    """Return list of reason_code strings emitted to the chain."""
    return [
        call.args[1]["reason_code"]
        for call in chain.log_event.call_args_list
        if call.args and len(call.args) >= 2
    ]


def _make_fake_model(
    src_dir: Path,
    files: dict[str, bytes] | None = None,
) -> tuple[dict[str, str], dict[str, bytes]]:
    """Create a fake model directory with deterministic file contents.

    Returns ``(sha256_manifest, file_bytes)``.
    """
    if files is None:
        files = {
            "model.bin": b"\x00\x01weights\x00\x01" * 16,
            "vocab.txt": b"hello\nworld\n",
        }
    src_dir.mkdir(parents=True, exist_ok=True)
    sha_map: dict[str, str] = {}
    import hashlib

    for fname, content in files.items():
        (src_dir / fname).write_bytes(content)
        sha_map[fname] = hashlib.sha256(content).hexdigest()
    return sha_map, files


def _make_download_fn(
    files: dict[str, bytes] | None = None,
    license: str = "Apache-2.0",
    nc_licensed: bool = False,
):
    """Build a download fn that drops ``files`` into dest_dir."""
    import hashlib

    if files is None:
        files = {
            "model.bin": b"weights-payload",
            "vocab.txt": b"a\nb\nc\n",
        }

    def _fn(engine: str, model_id: str, dest_dir: Path) -> dict:
        dest_dir.mkdir(parents=True, exist_ok=True)
        sha_map: dict[str, str] = {}
        total = 0
        for fname, content in files.items():
            (dest_dir / fname).write_bytes(content)
            sha_map[fname] = hashlib.sha256(content).hexdigest()
            total += len(content)
        return {
            "sha256": sha_map,
            "license": license,
            "nc_licensed": nc_licensed,
            "size_bytes": total,
        }

    return _fn


# ---------------------------------------------------------------------------
# 1. Fresh load -- downloads disallowed
# ---------------------------------------------------------------------------


def test_missing_model_no_allow_download_raises(cache_dir: Path):
    with pytest.raises(ModelNotCachedError) as exc:
        get_translation_model_path(
            "local_ct2_opus",
            "opus-en-fr",
            allow_download=False,
            cache_dir=cache_dir,
        )
    assert exc.value.engine == "local_ct2_opus"
    assert exc.value.model_id == "opus-en-fr"


# ---------------------------------------------------------------------------
# 2. Fresh load -- download succeeds and emits MODEL_DOWNLOADED
# ---------------------------------------------------------------------------


def test_download_emits_model_downloaded_custody(cache_dir: Path, custody: MagicMock):
    set_download_fn(_make_download_fn())

    path = get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )
    assert path.exists()
    assert (path / "model.bin").exists()
    assert (path / "vocab.txt").exists()
    assert (path / "manifest.json").exists()

    codes = _custody_reason_codes(custody)
    assert ReasonCode.MODEL_DOWNLOADED.value in codes


def test_downloaded_manifest_round_trips(cache_dir: Path):
    set_download_fn(_make_download_fn(license="CC-BY-4.0"))
    path = get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["nc_licensed"] is False
    assert "sha256" in manifest
    assert "model.bin" in manifest["sha256"]


# ---------------------------------------------------------------------------
# 3. Cached load -- access_time updated, no re-download
# ---------------------------------------------------------------------------


def test_cached_load_does_not_redownload(cache_dir: Path, custody: MagicMock):
    fn = MagicMock(side_effect=_make_download_fn())
    set_download_fn(fn)

    # First call downloads.
    get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )
    assert fn.call_count == 1

    # Second call should reuse the cached model.
    get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )
    assert fn.call_count == 1  # NOT re-called


def test_cached_load_updates_access_time(cache_dir: Path):
    set_download_fn(_make_download_fn())
    get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )
    state_path = cache_dir / "_state" / "cache_state.json"
    initial = json.loads(state_path.read_text())["models"][
        "local_ct2_opus/opus-en-fr"
    ]["access_time"]
    time.sleep(0.02)
    get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=False,
        cache_dir=cache_dir,
    )
    later = json.loads(state_path.read_text())["models"][
        "local_ct2_opus/opus-en-fr"
    ]["access_time"]
    assert later > initial


# ---------------------------------------------------------------------------
# 4. Integrity failure
# ---------------------------------------------------------------------------


def test_corrupt_file_raises_and_emits_integrity_failed(
    cache_dir: Path, custody: MagicMock
):
    set_download_fn(_make_download_fn())
    path = get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )
    custody.reset_mock()
    # Corrupt one of the files.
    (path / "model.bin").write_bytes(b"corrupted-junk")

    with pytest.raises(ModelIntegrityError) as exc:
        get_translation_model_path(
            "local_ct2_opus",
            "opus-en-fr",
            allow_download=False,
            cache_dir=cache_dir,
        )
    assert "model.bin" in exc.value.bad_files

    codes = _custody_reason_codes(custody)
    assert ReasonCode.MODEL_INTEGRITY_FAILED.value in codes


def test_verify_model_integrity_returns_false_after_corruption(cache_dir: Path):
    set_download_fn(_make_download_fn())
    path = get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )
    assert verify_model_integrity(
        "local_ct2_opus", "opus-en-fr", cache_dir=cache_dir
    )
    (path / "vocab.txt").write_bytes(b"different")
    assert not verify_model_integrity(
        "local_ct2_opus", "opus-en-fr", cache_dir=cache_dir
    )


# ---------------------------------------------------------------------------
# 5. LRU eviction
# ---------------------------------------------------------------------------


def test_evict_lru_frees_bytes_pinned_exempt(cache_dir: Path, custody: MagicMock):
    # Stage three models with deterministic access_time ordering.
    set_download_fn(_make_download_fn(files={"model.bin": b"a" * 1024}))
    get_translation_model_path(
        "local_ct2_opus",
        "model-a",
        allow_download=True,
        cache_dir=cache_dir,
    )
    time.sleep(0.01)
    get_translation_model_path(
        "local_ct2_opus",
        "model-b",
        allow_download=True,
        cache_dir=cache_dir,
    )
    time.sleep(0.01)
    get_translation_model_path(
        "local_ct2_opus",
        "model-c",
        allow_download=True,
        cache_dir=cache_dir,
    )

    # Pin model-a so even though it's the oldest it must be kept.
    pin_model("local_ct2_opus", "model-a", cache_dir=cache_dir)
    custody.reset_mock()

    evicted = evict_lru(2048, cache_dir=cache_dir)
    # Should evict at least model-b (oldest unpinned) before stopping.
    assert "local_ct2_opus/model-a" not in evicted
    assert "local_ct2_opus/model-b" in evicted

    # Pinned model still on disk.
    assert (cache_dir / "local_ct2_opus" / "model-a").exists()
    # Evicted model gone.
    assert not (cache_dir / "local_ct2_opus" / "model-b").exists()

    codes = _custody_reason_codes(custody)
    assert ReasonCode.MODEL_EVICTED.value in codes


def test_evict_lru_returns_empty_for_zero_target(cache_dir: Path):
    assert evict_lru(0, cache_dir=cache_dir) == []
    assert evict_lru(-100, cache_dir=cache_dir) == []


def test_pin_then_unpin_round_trip(cache_dir: Path, custody: MagicMock):
    set_download_fn(_make_download_fn())
    get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )
    pin_model("local_ct2_opus", "opus-en-fr", cache_dir=cache_dir)
    listing = list_cached_models(cache_dir=cache_dir)
    assert any(c.pinned for c in listing if c.model_id == "opus-en-fr")

    unpin_model("local_ct2_opus", "opus-en-fr", cache_dir=cache_dir)
    listing = list_cached_models(cache_dir=cache_dir)
    assert all(not c.pinned for c in listing if c.model_id == "opus-en-fr")

    codes = _custody_reason_codes(custody)
    assert ReasonCode.MODEL_PINNED.value in codes
    assert ReasonCode.MODEL_UNPINNED.value in codes


# ---------------------------------------------------------------------------
# 6. NC-license block
# ---------------------------------------------------------------------------


def test_nc_licensed_blocked_when_policy_disallows(
    cache_dir: Path, custody: MagicMock
):
    set_download_fn(
        _make_download_fn(license="CC-BY-NC-4.0", nc_licensed=True)
    )
    policy = TenantPolicy(tenant_id="commercial-tenant", allow_nllb_commercial=False)

    with pytest.raises(PolicyDenied) as exc:
        get_translation_model_path(
            "local_ct2_nllb",
            "nllb-200-1.3b",
            allow_download=True,
            tenant_policy=policy,
            cache_dir=cache_dir,
        )
    assert exc.value.reason_code == ReasonCode.MODEL_LOAD_BLOCKED_NC_LICENSE

    codes = _custody_reason_codes(custody)
    assert ReasonCode.MODEL_LOAD_BLOCKED_NC_LICENSE.value in codes


def test_nc_licensed_allowed_when_policy_permits(cache_dir: Path):
    set_download_fn(
        _make_download_fn(license="CC-BY-NC-4.0", nc_licensed=True)
    )
    policy = TenantPolicy(tenant_id="research", allow_nllb_commercial=True)

    path = get_translation_model_path(
        "local_ct2_nllb",
        "nllb-200-1.3b",
        allow_download=True,
        tenant_policy=policy,
        cache_dir=cache_dir,
    )
    assert path.exists()


def test_nc_block_uses_attribute_alias():
    """Cache must honour either ``allow_nc_licensed`` or ``allow_nllb_commercial``."""
    from ocr_local.translation.cache import _tenant_allows_nc

    class _Stub:
        allow_nc_licensed = False

    assert not _tenant_allows_nc(_Stub())

    class _Legacy:
        allow_nllb_commercial = False

    assert not _tenant_allows_nc(_Legacy())

    class _Permissive:
        allow_nc_licensed = True

    assert _tenant_allows_nc(_Permissive())

    assert _tenant_allows_nc(None)


# ---------------------------------------------------------------------------
# 7. Air-gapped mode forces allow_download=False
# ---------------------------------------------------------------------------


def test_airgapped_forces_no_download(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TRANSLATION_AIRGAPPED", "true")
    set_download_fn(_make_download_fn())

    with pytest.raises(ModelNotCachedError) as exc:
        get_translation_model_path(
            "local_ct2_opus",
            "opus-en-fr",
            allow_download=True,  # caller asks; airgap overrides
            cache_dir=cache_dir,
        )
    assert "airgapped" in exc.value.reason


def test_airgapped_serves_pre_baked_models(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """register_model should make a model usable in air-gapped mode."""
    src = cache_dir.parent / "src"
    sha_map, _files = _make_fake_model(src)
    register_model(
        "local_ct2_opus",
        "pre-baked",
        src,
        license="Apache-2.0",
        nc_licensed=False,
        sha256_manifest=sha_map,
        cache_dir=cache_dir,
    )

    monkeypatch.setenv("TRANSLATION_AIRGAPPED", "true")
    path = get_translation_model_path(
        "local_ct2_opus",
        "pre-baked",
        allow_download=True,
        cache_dir=cache_dir,
    )
    assert path.exists()


# ---------------------------------------------------------------------------
# 8. Concurrency -- 4 threads racing on cache_state.json
# ---------------------------------------------------------------------------


def test_concurrent_get_no_race_on_state(cache_dir: Path):
    """Four threads call get_translation_model_path simultaneously.

    With the file lock + thread lock, the cache_state.json must remain
    valid JSON containing exactly the expected entry, and no thread
    should crash with a transient parse error.
    """
    set_download_fn(_make_download_fn())
    # Pre-populate so all four threads hit the cache-hit path.
    get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )

    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def _worker():
        try:
            barrier.wait(timeout=5)
            for _ in range(20):
                get_translation_model_path(
                    "local_ct2_opus",
                    "opus-en-fr",
                    allow_download=False,
                    cache_dir=cache_dir,
                )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"thread errors: {errors!r}"

    # State file must still parse and contain the entry.
    state_path = cache_dir / "_state" / "cache_state.json"
    state = json.loads(state_path.read_text())
    assert "local_ct2_opus/opus-en-fr" in state["models"]


# ---------------------------------------------------------------------------
# 9. register_model -- pre-baked model integrity-verifies on next get
# ---------------------------------------------------------------------------


def test_register_model_writes_manifest(cache_dir: Path):
    src = cache_dir.parent / "src"
    sha_map, files = _make_fake_model(src)

    target = register_model(
        "local_ct2_opus",
        "pre-baked",
        src,
        license="Apache-2.0",
        nc_licensed=False,
        sha256_manifest=sha_map,
        cache_dir=cache_dir,
    )

    assert (target / "manifest.json").exists()
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["license"] == "Apache-2.0"
    assert manifest["nc_licensed"] is False
    assert manifest["sha256"] == sha_map
    assert manifest["size_bytes"] == sum(len(b) for b in files.values())


def test_register_model_then_get_succeeds(cache_dir: Path, custody: MagicMock):
    src = cache_dir.parent / "src"
    sha_map, _files = _make_fake_model(src)
    register_model(
        "local_ct2_opus",
        "pre-baked",
        src,
        license="Apache-2.0",
        nc_licensed=False,
        sha256_manifest=sha_map,
        cache_dir=cache_dir,
    )
    custody.reset_mock()

    path = get_translation_model_path(
        "local_ct2_opus",
        "pre-baked",
        allow_download=False,
        cache_dir=cache_dir,
    )
    assert path.exists()
    codes = _custody_reason_codes(custody)
    assert ReasonCode.MODEL_INTEGRITY_VERIFIED.value in codes


def test_register_model_rejects_corrupt_source(cache_dir: Path):
    src = cache_dir.parent / "src"
    src.mkdir()
    (src / "model.bin").write_bytes(b"actual-content")
    bad_sha = {"model.bin": "deadbeef" * 8}  # wrong digest

    with pytest.raises(ModelIntegrityError):
        register_model(
            "local_ct2_opus",
            "bad",
            src,
            license="Apache-2.0",
            nc_licensed=False,
            sha256_manifest=bad_sha,
            cache_dir=cache_dir,
        )


def test_register_model_requires_existing_source(cache_dir: Path):
    with pytest.raises(FileNotFoundError):
        register_model(
            "local_ct2_opus",
            "ghost",
            cache_dir.parent / "does-not-exist",
            license="Apache-2.0",
            nc_licensed=False,
            sha256_manifest={"x.bin": "0" * 64},
            cache_dir=cache_dir,
        )


def test_register_model_requires_non_empty_manifest(cache_dir: Path):
    src = cache_dir.parent / "src"
    src.mkdir()
    with pytest.raises(ValueError):
        register_model(
            "local_ct2_opus",
            "empty",
            src,
            license="Apache-2.0",
            nc_licensed=False,
            sha256_manifest={},
            cache_dir=cache_dir,
        )


# ---------------------------------------------------------------------------
# 10. list_cached_models reflects state correctly
# ---------------------------------------------------------------------------


def test_list_cached_models_returns_metadata(cache_dir: Path):
    set_download_fn(_make_download_fn(license="Apache-2.0"))
    get_translation_model_path(
        "local_ct2_opus",
        "opus-en-fr",
        allow_download=True,
        cache_dir=cache_dir,
    )

    listing = list_cached_models(cache_dir=cache_dir)
    assert len(listing) == 1
    info = listing[0]
    assert isinstance(info, CachedModelInfo)
    assert info.engine == "local_ct2_opus"
    assert info.model_id == "opus-en-fr"
    assert info.license == "Apache-2.0"
    assert info.size_bytes > 0
    assert info.access_time > 0


def test_default_download_fn_raises(cache_dir: Path):
    """Without set_download_fn, missing model + allow_download must raise."""
    with pytest.raises(NotImplementedError):
        get_translation_model_path(
            "local_ct2_opus",
            "opus-en-fr",
            allow_download=True,
            cache_dir=cache_dir,
        )
