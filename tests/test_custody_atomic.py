"""Tests for custody chain atomic write behavior (maintenance review).

Verifies that append_event() uses fsync to ensure bytes reach stable storage
before updating in-memory state, preventing hash chain divergence on power loss.
"""

import json
import os
from unittest.mock import patch

import pytest

from custody import CustodyChain


class _PostCommitCleanupSignal(BaseException):
    """Non-Exception signal injected only after a real cleanup side effect."""


def _fail_custody_data_appends(message):
    """Fail descriptor-bound JSONL writes while leaving the lock usable."""
    real_open = os.open

    def failing_open(path, flags, *args, **kwargs):
        path = os.fspath(path)
        if path.endswith(".custody.jsonl") and flags & (os.O_WRONLY | os.O_RDWR):
            raise OSError(message)
        return real_open(path, flags, *args, **kwargs)

    return failing_open


class TestFsyncCalledOnWrite:
    """Verify os.fsync is called on every successful disk write."""

    def test_fsync_called_on_single_write(self, tmp_path):
        """os.fsync must be called once per successful append_event."""
        chain = CustodyChain("doc_fsync", "/test/path", str(tmp_path))

        with patch("custody.os.fsync", wraps=os.fsync) as mock_fsync:
            chain.append_event("file_ingested", {"test": True})

        # One sync for the evidence inode and one for the canonical parent
        # directory (first-create/name durability).
        assert mock_fsync.call_count == 2
        # Argument should be a valid file descriptor (integer)
        fd_arg = mock_fsync.call_args[0][0]
        assert isinstance(fd_arg, int)

    def test_fsync_called_on_each_event(self, tmp_path):
        """os.fsync must be called for every event written to disk."""
        chain = CustodyChain("doc_multi", "/test/path", str(tmp_path))

        with patch("custody.os.fsync", wraps=os.fsync) as mock_fsync:
            chain.append_event("file_ingested", {"step": 1})
            chain.append_event("page_extracted", {"step": 2})
            chain.append_event("ocr_primary", {"step": 3})

        assert mock_fsync.call_count == 6

    def test_fsync_not_called_without_filepath(self):
        """Chains without a custody_dir should not call fsync."""
        chain = CustodyChain("doc_nofs", "/test/path")

        with patch("custody.os.fsync") as mock_fsync:
            chain.append_event("file_ingested", {"test": True})

        mock_fsync.assert_not_called()

    def test_descriptor_write_called_before_fsync(self, tmp_path):
        """The exact descriptor append must complete before its durability sync."""
        chain = CustodyChain("doc_order", "/test/path", str(tmp_path))
        call_order = []
        real_write = os.write
        real_fsync = os.fsync

        def tracking_write(fd, payload):
            call_order.append("write")
            return real_write(fd, payload)

        def tracking_fsync(fd):
            call_order.append("fsync")
            return real_fsync(fd)

        with (
            patch("custody.os.write", side_effect=tracking_write),
            patch("custody.os.fsync", side_effect=tracking_fsync),
        ):
            chain.append_event("file_ingested", {"order": "test"})

        assert call_order == ["write", "fsync", "fsync"]

    def test_parent_directory_fsync_failure_rolls_back_before_retry(self, tmp_path):
        """A failed canonical-name sync cannot duplicate the logical event."""
        import custody as custody_module

        chain = CustodyChain("doc_parent_sync", "/test/path", str(tmp_path))
        real_fsync = os.fsync
        failed_directory_sync = False

        def fail_first_directory_sync(fd):
            nonlocal failed_directory_sync
            if not failed_directory_sync and os.path.isdir(f"/proc/self/fd/{fd}"):
                failed_directory_sync = True
                raise OSError("injected parent directory fsync failure")
            return real_fsync(fd)

        with (
            patch.object(
                custody_module.os,
                "fsync",
                side_effect=fail_first_directory_sync,
            ),
            patch.object(custody_module.time, "sleep") as mock_sleep,
        ):
            event = chain.append_event("file_ingested", {"test": True})

        assert failed_directory_sync
        mock_sleep.assert_called_once()
        lines = (
            (tmp_path / "doc_parent_sync.custody.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 1
        assert json.loads(lines[0])["hash"] == event["hash"]


@pytest.mark.parametrize(
    "failure_point",
    ["unlock", "custody_fd_close", "parent_fd_close"],
)
@pytest.mark.parametrize(
    "failure_type",
    [OSError, _PostCommitCleanupSignal],
    ids=["exception", "baseexception"],
)
def test_post_commit_cleanup_error_never_blindly_retries(
    tmp_path,
    failure_point,
    failure_type,
):
    """Cleanup debt after the commit boundary cannot append twice."""
    import stat

    import custody as custody_module

    chain = CustodyChain(f"doc_cleanup_{failure_point}", "/test/path", str(tmp_path))
    injected = False

    with patch.object(custody_module.time, "sleep") as mock_sleep:
        if failure_point == "unlock":
            real_unlock = custody_module._unlock_file_strict

            def finish_unlock_then_raise(lock_file):
                nonlocal injected
                real_unlock(lock_file)
                if not injected:
                    injected = True
                    raise failure_type("injected post-unlock failure")

            cleanup_patch = patch.object(
                custody_module,
                "_unlock_file_strict",
                side_effect=finish_unlock_then_raise,
            )
        else:
            real_close = os.close
            expected_directory = failure_point == "parent_fd_close"

            def finish_close_then_raise(fd):
                nonlocal injected
                metadata = os.fstat(fd)
                should_fail = (
                    stat.S_ISDIR(metadata.st_mode)
                    if expected_directory
                    else stat.S_ISREG(metadata.st_mode)
                )
                real_close(fd)
                if should_fail and not injected:
                    injected = True
                    raise failure_type(f"injected {failure_point}")

            cleanup_patch = patch.object(
                custody_module.os,
                "close",
                side_effect=finish_close_then_raise,
            )

        with cleanup_patch:
            event = chain.append_event("file_ingested", {"cleanup": failure_point})

    assert injected
    mock_sleep.assert_not_called()
    assert chain.integrity_compromised is False
    path = tmp_path / f"doc_cleanup_{failure_point}.custody.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["hash"] == event["hash"]


class TestMemoryNotUpdatedOnDiskFailure:
    """Verify in-memory state is unchanged when disk write fails."""

    def test_events_list_unchanged_on_first_event_failure(self, tmp_path):
        """If the very first disk write fails, events list should still
        contain the event (existing behavior) but hash should NOT advance."""
        chain = CustodyChain("doc_fail1", "/test/path", str(tmp_path))

        with (
            patch(
                "custody.os.open",
                side_effect=_fail_custody_data_appends("disk full"),
            ),
            patch("custody.time.sleep"),
        ):
            with pytest.raises(OSError):
                chain.append_event("file_ingested", {"test": True})

        # A failed durable append is not reported as an in-memory event.
        assert chain.events == []
        # But hash did NOT advance (chain integrity preserved for next event)
        assert chain._prev_hash is None
        assert chain.integrity_compromised is True

    def test_events_list_preserved_on_subsequent_failure(self, tmp_path):
        """After a successful first event, a failed second write should NOT
        corrupt the in-memory chain state."""
        chain = CustodyChain("doc_fail2", "/test/path", str(tmp_path))

        # First event succeeds
        event1 = chain.append_event("file_ingested", {"step": 1})
        assert len(chain.events) == 1
        assert chain._prev_hash == event1["hash"]
        prev_hash_before_failure = chain._prev_hash

        # Second event fails
        with (
            patch(
                "custody.os.open",
                side_effect=_fail_custody_data_appends("disk full"),
            ),
            patch("custody.time.sleep"),
        ):
            with pytest.raises(OSError):
                chain.append_event("page_extracted", {"step": 2})

        # The failed event is not added to the in-memory chain.
        assert len(chain.events) == 1
        # But _prev_hash did NOT advance past the first event
        assert chain._prev_hash == prev_hash_before_failure

    def test_fsync_failure_treated_as_write_failure(self, tmp_path):
        """If os.fsync raises, the event should be treated as a write failure."""
        chain = CustodyChain("doc_fsyncfail", "/test/path", str(tmp_path))

        with (
            patch("custody.os.fsync", side_effect=OSError("fsync failed")),
            patch("custody.time.sleep"),
        ):
            with pytest.raises(OSError):
                chain.append_event("file_ingested", {"test": True})

        # fsync failure is caught by the OSError handler in the retry loop
        assert chain.integrity_compromised is True
        # Event is not in memory and hash is not advanced.
        assert chain.events == []
        assert chain._prev_hash is None

    def test_transient_fsync_failure_rolls_back_before_retry(self, tmp_path):
        """A post-write fsync failure must not duplicate the retried JSONL line."""
        chain = CustodyChain("doc_fsync_retry", "/test/path", str(tmp_path))
        real_fsync = os.fsync
        fsync_calls = 0

        def fail_first_fsync(fd):
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 1:
                raise OSError("transient fsync failure after append")
            return real_fsync(fd)

        with (
            patch("custody.os.fsync", side_effect=fail_first_fsync),
            patch("custody.time.sleep"),
        ):
            event = chain.append_event("file_ingested", {"test": True})

        # First append fsync failed, rollback was fsynced, then the retry was
        # fsynced.  The original bytes must not remain as a duplicate line.
        assert fsync_calls == 4
        assert chain.integrity_compromised is False
        assert chain._prev_hash == event["hash"]
        filepath = tmp_path / "doc_fsync_retry.custody.jsonl"
        lines = filepath.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["hash"] == event["hash"]

        loaded = CustodyChain.load_from_file(str(filepath))
        is_valid, message = loaded.verify_chain()
        assert is_valid is True, message

    def test_rollback_fsync_failure_stops_retries_fail_closed(self, tmp_path):
        """An uncertain rollback must stop instead of risking another append."""
        chain = CustodyChain("doc_rollback_fail", "/test/path", str(tmp_path))

        with (
            patch(
                "custody.os.fsync",
                side_effect=OSError("append and rollback fsync failed"),
            ) as mock_fsync,
            patch("custody.time.sleep") as mock_sleep,
        ):
            with pytest.raises(OSError):
                chain.append_event("file_ingested", {"test": True})

        # One fsync for the append and one for its rollback.  There must be no
        # second append attempt once rollback durability is uncertain.
        assert mock_fsync.call_count == 2
        mock_sleep.assert_not_called()
        assert chain.integrity_compromised is True
        assert chain._prev_hash is None
        assert chain.events == []
        assert (tmp_path / "doc_rollback_fail.custody.jsonl").read_bytes() == b""


class TestDiskBeforeMemoryOrdering:
    """Verify that disk write (with fsync) completes before in-memory update."""

    def test_memory_state_at_fsync_time(self, tmp_path):
        """At the moment fsync is called, in-memory events list should NOT
        yet contain the new event."""
        chain = CustodyChain("doc_ordering", "/test/path", str(tmp_path))

        events_len_at_fsync = []
        prev_hash_at_fsync = []

        original_fsync = os.fsync

        def spy_fsync(fd):
            events_len_at_fsync.append(len(chain.events))
            prev_hash_at_fsync.append(chain._prev_hash)
            return original_fsync(fd)

        with patch("custody.os.fsync", side_effect=spy_fsync):
            event = chain.append_event("file_ingested", {"order": "test"})

        # At fsync time: events list was empty, _prev_hash was None
        assert events_len_at_fsync == [0, 0]
        assert prev_hash_at_fsync == [None, None]

        # After append_event returns: event is in list, hash is set
        assert len(chain.events) == 1
        assert chain._prev_hash == event["hash"]

    def test_memory_state_at_fsync_time_second_event(self, tmp_path):
        """For the second event, verify in-memory state at fsync time
        reflects only the first event."""
        chain = CustodyChain("doc_order2", "/test/path", str(tmp_path))

        event1 = chain.append_event("file_ingested", {"step": 1})

        events_len_at_fsync = []
        prev_hash_at_fsync = []

        original_fsync = os.fsync

        def spy_fsync(fd):
            events_len_at_fsync.append(len(chain.events))
            prev_hash_at_fsync.append(chain._prev_hash)
            return original_fsync(fd)

        with patch("custody.os.fsync", side_effect=spy_fsync):
            event2 = chain.append_event("page_extracted", {"step": 2})

        # At fsync time: only 1 event in list, prev_hash is event1's hash
        assert events_len_at_fsync == [1, 1]
        assert prev_hash_at_fsync == [event1["hash"], event1["hash"]]

        # After return: 2 events, prev_hash is event2's hash
        assert len(chain.events) == 2
        assert chain._prev_hash == event2["hash"]

    def test_chain_verifies_after_fsync_write(self, tmp_path):
        """Full chain should verify after multiple fsync-backed writes."""
        chain = CustodyChain("doc_verify", "/test/path", str(tmp_path))

        chain.append_event("file_ingested", {"hash": "abc123"})
        chain.append_event("page_extracted", {"page": 1})
        chain.append_event("ocr_primary", {"confidence": 0.95})
        chain.append_event("assembly_complete", {"pages": 1})

        is_valid, message = chain.verify_chain()
        assert is_valid is True
        assert "4 events" in message

        # Verify the JSONL file also has all events
        filepath = tmp_path / "doc_verify.custody.jsonl"
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 4

    def test_fsync_on_retry_success(self, tmp_path):
        """When a write succeeds on retry, fsync must still be called."""
        chain = CustodyChain("doc_retry_fsync", "/test/path", str(tmp_path))

        real_open = os.open
        call_count = 0

        def flaky_open(path, flags, *args, **kwargs):
            nonlocal call_count
            if (
                os.fspath(path).endswith(".custody.jsonl")
                and flags & os.O_WRONLY
                and flags & os.O_CREAT
            ):
                call_count += 1
                if call_count <= 1:
                    raise OSError("transient failure")
            return real_open(path, flags, *args, **kwargs)

        with (
            patch("custody.os.open", side_effect=flaky_open),
            patch("custody.time.sleep"),
            patch("custody.os.fsync", wraps=os.fsync) as mock_fsync,
        ):
            chain.append_event("file_ingested", {"retry": True})

        # Successful commit syncs both JSONL bytes and the canonical parent.
        assert mock_fsync.call_count == 2
        assert chain.integrity_compromised is False
        assert chain._prev_hash is not None


class TestTrailingNewlineNormalization:
    """Missing final LF must be repaired durably before the next append."""

    def test_missing_lf_is_one_encoded_append_before_fsync(self, tmp_path):
        """The repair prefix and event are one rollback-verifiable append."""
        chain = CustodyChain("doc_norm_fsync", "/test/path", str(tmp_path))
        first = chain.append_event("file_ingested", {"step": 1})
        path = tmp_path / "doc_norm_fsync.custody.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        assert not path.read_bytes().endswith(b"\n")

        call_order = []
        original_fsync = os.fsync
        original_write = os.write

        def spy_write(fd, payload):
            call_order.append(("write", bytes(payload)))
            return original_write(fd, payload)

        def spy_fsync(fd):
            call_order.append("fsync")
            return original_fsync(fd)

        with (
            patch("custody.os.write", side_effect=spy_write),
            patch("custody.os.fsync", side_effect=spy_fsync),
        ):
            second = chain.append_event("page_extracted", {"step": 2})

        assert len(call_order) == 3
        assert call_order[0][0] == "write"
        assert call_order[0][1].startswith(b"\n{")
        assert call_order[1:] == ["fsync", "fsync"]
        assert second["prev_hash"] == first["hash"]
        assert b"}{" not in path.read_bytes()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_append_rollback_restores_exact_pre_append_bytes(self, tmp_path):
        """A failed prefix+event append restores the original unterminated tip."""
        chain = CustodyChain("doc_norm_rollback", "/test/path", str(tmp_path))
        first = chain.append_event("file_ingested", {"step": 1})
        path = tmp_path / "doc_norm_rollback.custody.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        original = path.read_bytes()
        real_fsync = os.fsync
        fsync_calls = 0

        def fail_each_append_fsync(fd):
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls % 2 == 1:
                raise OSError("append fsync failed")
            return real_fsync(fd)

        with (
            patch("custody.os.fsync", side_effect=fail_each_append_fsync),
            patch("custody.time.sleep"),
        ):
            with pytest.raises(OSError):
                chain.append_event("page_extracted", {"step": 2})

        assert fsync_calls == 6
        assert chain.integrity_compromised is True
        assert chain._prev_hash == first["hash"]
        raw = path.read_bytes()
        assert raw == original
        assert not raw.endswith(b"\n")
        assert b"}{" not in raw
        lines = raw.decode("utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["hash"] == first["hash"]
