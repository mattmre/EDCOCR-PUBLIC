"""Tests for custody chain atomic write behavior.

Verifies that append_event() uses fsync to ensure bytes reach stable storage
before updating in-memory state, preventing hash chain divergence on power loss.
"""

import os
from unittest.mock import patch

from custody import CustodyChain


class TestFsyncCalledOnWrite:
    """Verify os.fsync is called on every successful disk write."""

    def test_fsync_called_on_single_write(self, tmp_path):
        """os.fsync must be called once per successful append_event."""
        chain = CustodyChain("doc_fsync", "/test/path", str(tmp_path))

        with patch("custody.os.fsync", wraps=os.fsync) as mock_fsync:
            chain.append_event("file_ingested", {"test": True})

        assert mock_fsync.call_count == 1
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

        assert mock_fsync.call_count == 3

    def test_fsync_not_called_without_filepath(self):
        """Chains without a custody_dir should not call fsync."""
        chain = CustodyChain("doc_nofs", "/test/path")

        with patch("custody.os.fsync") as mock_fsync:
            chain.append_event("file_ingested", {"test": True})

        mock_fsync.assert_not_called()

    def test_flush_called_before_fsync(self, tmp_path):
        """f.flush() must be called before os.fsync(f.fileno())."""
        chain = CustodyChain("doc_order", "/test/path", str(tmp_path))
        call_order = []

        real_open = open

        class OrderTrackingFile:
            """Wraps the real file to track flush/fileno call order."""

            def __init__(self, real_file):
                self._real = real_file

            def write(self, data):
                return self._real.write(data)

            def flush(self):
                call_order.append("flush")
                return self._real.flush()

            def fileno(self):
                call_order.append("fileno")
                return self._real.fileno()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self._real.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self._real, name)

        def tracking_open(*args, **kwargs):
            f = real_open(*args, **kwargs)
            if len(args) >= 2 and args[1] == "a":
                return OrderTrackingFile(f)
            return f

        with patch("builtins.open", side_effect=tracking_open), \
             patch("custody.os.fsync"):
            chain.append_event("file_ingested", {"order": "test"})

        # flush must appear before fileno (which is called for fsync arg)
        assert "flush" in call_order
        assert "fileno" in call_order
        flush_idx = call_order.index("flush")
        fileno_idx = call_order.index("fileno")
        assert flush_idx < fileno_idx, "flush() must be called before fileno()/fsync()"


class TestMemoryNotUpdatedOnDiskFailure:
    """Verify in-memory state is unchanged when disk write fails."""

    def test_events_list_unchanged_on_first_event_failure(self, tmp_path):
        """If the very first disk write fails, events list should still
        contain the event (existing behavior) but hash should NOT advance."""
        chain = CustodyChain("doc_fail1", "/test/path", str(tmp_path))

        with patch("builtins.open", side_effect=OSError("disk full")), \
             patch("custody.time.sleep"):
            chain.append_event("file_ingested", {"test": True})

        # Event is in memory (existing behavior: event added for audit trail)
        assert len(chain.events) == 1
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
        with patch("builtins.open", side_effect=OSError("disk full")), \
             patch("custody.time.sleep"):
            chain.append_event("page_extracted", {"step": 2})

        # In-memory list has the failed event (for audit)
        assert len(chain.events) == 2
        # But _prev_hash did NOT advance past the first event
        assert chain._prev_hash == prev_hash_before_failure

    def test_fsync_failure_treated_as_write_failure(self, tmp_path):
        """If os.fsync raises, the event should be treated as a write failure."""
        chain = CustodyChain("doc_fsyncfail", "/test/path", str(tmp_path))

        with patch("custody.os.fsync", side_effect=OSError("fsync failed")), \
             patch("custody.time.sleep"):
            chain.append_event("file_ingested", {"test": True})

        # fsync failure is caught by the OSError handler in the retry loop
        assert chain.integrity_compromised is True
        # Event is in memory but hash not advanced
        assert len(chain.events) == 1
        assert chain._prev_hash is None


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
        assert events_len_at_fsync == [0]
        assert prev_hash_at_fsync == [None]

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
        assert events_len_at_fsync == [1]
        assert prev_hash_at_fsync == [event1["hash"]]

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

        real_open = open
        call_count = 0

        def flaky_open(*args, **kwargs):
            nonlocal call_count
            if len(args) >= 2 and args[1] == "a":
                call_count += 1
                if call_count <= 1:
                    raise OSError("transient failure")
            return real_open(*args, **kwargs)

        with patch("builtins.open", side_effect=flaky_open), \
             patch("custody.time.sleep"), \
             patch("custody.os.fsync", wraps=os.fsync) as mock_fsync:
            chain.append_event("file_ingested", {"retry": True})

        # fsync should be called once (on the successful attempt)
        assert mock_fsync.call_count == 1
        assert chain.integrity_compromised is False
        assert chain._prev_hash is not None
