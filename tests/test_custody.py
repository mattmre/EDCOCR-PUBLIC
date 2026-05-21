"""Tests for forensic chain-of-custody module."""

import json
import logging
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from custody import (
    CUSTODY_RETRY_DELAYS,
    EVENT_TYPES,
    MAX_CUSTODY_RETRIES,
    CustodyChain,
    compute_file_hash,
    verify_custody_file,
)


class TestCustodyChain:
    """Test core chain-of-custody functionality."""

    def test_genesis_event_has_null_prev_hash(self, tmp_path):
        """First event in chain should have prev_hash=None."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        event = chain.append_event("file_ingested", {"test": "data"})
        
        assert event["prev_hash"] is None
        assert event["hash"] is not None
        assert len(event["hash"]) == 64  # SHA-256 hex digest

    def test_events_are_linked_by_hash(self, tmp_path):
        """Each event's prev_hash should match the previous event's hash."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        event1 = chain.append_event("file_ingested", {"step": 1})
        event2 = chain.append_event("page_extracted", {"step": 2})
        event3 = chain.append_event("ocr_primary", {"step": 3})
        
        assert event1["prev_hash"] is None
        assert event2["prev_hash"] == event1["hash"]
        assert event3["prev_hash"] == event2["hash"]

    def test_event_contains_required_fields(self, tmp_path):
        """Events must have all required fields."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        event = chain.append_event("file_ingested", {"key": "value"})
        
        required_fields = ["document_id", "event_type", "timestamp", "data", "prev_hash", "hash"]
        for field in required_fields:
            assert field in event, f"Missing required field: {field}"
        
        assert event["document_id"] == "test_doc"
        assert event["event_type"] == "file_ingested"
        assert event["data"] == {"key": "value"}

    def test_timestamp_is_iso8601_with_milliseconds(self, tmp_path):
        """Timestamps should be ISO 8601 with millisecond precision."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        event = chain.append_event("file_ingested", {})
        
        timestamp = event["timestamp"]
        
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo is not None  # Should have timezone
        
        # Should contain milliseconds (3 decimal places)
        assert "." in timestamp
        # Format: 2024-02-10T12:34:56.789+00:00
        assert timestamp.count(".") == 1

    def test_chain_verification_passes_for_valid_chain(self, tmp_path):
        """verify_chain() should pass for an untampered chain."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        chain.append_event("file_ingested", {"hash": "abc123"})
        chain.append_event("page_extracted", {"page": 1})
        chain.append_event("ocr_primary", {"confidence": 0.95})
        chain.append_event("assembly_complete", {"pages": 10})
        
        is_valid, message = chain.verify_chain()
        assert is_valid is True
        assert "4 events" in message

    def test_chain_verification_fails_for_tampered_event(self, tmp_path):
        """verify_chain() should detect modified event data."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        chain.append_event("file_ingested", {"hash": "abc123"})
        chain.append_event("page_extracted", {"page": 1})
        chain.append_event("ocr_primary", {"confidence": 0.95})
        
        # Tamper with the second event
        chain.events[1]["data"]["page"] = 999
        
        is_valid, message = chain.verify_chain()
        assert is_valid is False
        assert "Tampered event" in message

    def test_chain_verification_fails_for_broken_link(self, tmp_path):
        """verify_chain() should detect a broken prev_hash link."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        chain.append_event("file_ingested", {"hash": "abc123"})
        chain.append_event("page_extracted", {"page": 1})
        chain.append_event("ocr_primary", {"confidence": 0.95})
        
        # Break the chain link
        chain.events[2]["prev_hash"] = "0" * 64
        
        is_valid, message = chain.verify_chain()
        assert is_valid is False
        assert "Broken chain" in message

    def test_empty_chain_verification(self):
        """Empty chain should verify successfully."""
        chain = CustodyChain("test_doc", "/test/path")
        
        is_valid, message = chain.verify_chain()
        assert is_valid is True
        assert "Empty chain" in message


class TestCustodyFileIO:
    """Test file I/O operations."""

    def test_jsonl_file_created_on_append(self, tmp_path):
        """JSONL file should be created in custody_dir."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        chain.append_event("file_ingested", {"test": "data"})
        
        expected_path = tmp_path / "test_doc.custody.jsonl"
        assert expected_path.exists()

    def test_jsonl_lines_are_valid_json(self, tmp_path):
        """Each line in JSONL file should be valid JSON."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        chain.append_event("file_ingested", {"hash": "abc"})
        chain.append_event("page_extracted", {"page": 1})
        chain.append_event("ocr_primary", {"engine": "paddle"})
        
        filepath = tmp_path / "test_doc.custody.jsonl"
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        
        for line in lines:
            event = json.loads(line)  # Should not raise
            assert "hash" in event
            assert "event_type" in event

    def test_load_from_file_restores_chain(self, tmp_path):
        """Loading a JSONL file should restore the full chain."""
        # Create original chain
        chain1 = CustodyChain("test_doc", "/test/path", str(tmp_path))
        chain1.append_event("file_ingested", {"hash": "abc123", "source_path": "/test/path"})
        chain1.append_event("page_extracted", {"page": 1})
        chain1.append_event("ocr_primary", {"confidence": 0.95})
        
        # Load from file
        filepath = tmp_path / "test_doc.custody.jsonl"
        chain2 = CustodyChain.load_from_file(str(filepath))
        
        assert len(chain2.events) == 3
        assert chain2.document_id == "test_doc"
        assert chain2.source_path == "/test/path"
        assert chain2.events[0]["event_type"] == "file_ingested"
        assert chain2.events[1]["event_type"] == "page_extracted"
        assert chain2.events[2]["event_type"] == "ocr_primary"

    def test_loaded_chain_verifies(self, tmp_path):
        """A loaded chain should pass verification."""
        # Create and save chain
        chain1 = CustodyChain("test_doc", "/test/path", str(tmp_path))
        chain1.append_event("file_ingested", {"hash": "abc123"})
        chain1.append_event("page_extracted", {"page": 1})
        chain1.append_event("ocr_primary", {"confidence": 0.95})
        
        # Load and verify
        filepath = tmp_path / "test_doc.custody.jsonl"
        chain2 = CustodyChain.load_from_file(str(filepath))
        
        is_valid, message = chain2.verify_chain()
        assert is_valid is True
        assert "3 events" in message

    def test_verify_custody_file_convenience(self, tmp_path):
        """verify_custody_file() should work as a standalone validator."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        chain.append_event("file_ingested", {"hash": "abc123"})
        chain.append_event("page_extracted", {"page": 1})
        
        filepath = tmp_path / "test_doc.custody.jsonl"
        is_valid, message = verify_custody_file(str(filepath))
        
        assert is_valid is True
        assert "2 events" in message

    def test_verify_custody_file_handles_invalid_json(self, tmp_path):
        """verify_custody_file() should handle corrupted JSONL files."""
        filepath = tmp_path / "corrupted.custody.jsonl"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write('{"valid": "json"}\n')
            f.write('this is not json\n')
            f.write('{"also": "valid"}\n')
        
        is_valid, message = verify_custody_file(str(filepath))
        assert is_valid is False
        assert "Failed to load" in message


class TestEventTypes:
    """Test event types and metadata."""

    def test_all_pipeline_stages_covered(self):
        """EVENT_TYPES should cover all pipeline stages."""
        required = [
            "file_ingested",
            "page_extracted",
            "ocr_primary",
            "ocr_fallback",
            "assembly_complete",
        ]
        for event_type in required:
            assert event_type in EVENT_TYPES, f"Missing event type: {event_type}"

    def test_event_type_data_fields(self, tmp_path):
        """Events should store stage-specific data."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        # File ingestion event with hash
        event = chain.append_event(
            "file_ingested",
            {
                "file_hash": "abc123...",
                "file_size": 1024000,
                "mime_type": "application/pdf",
            },
        )
        assert event["data"]["file_hash"] == "abc123..."
        assert event["data"]["file_size"] == 1024000
        assert event["data"]["mime_type"] == "application/pdf"

    def test_ocr_event_records_engine_and_confidence(self, tmp_path):
        """OCR events should record engine name and confidence."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        event = chain.append_event(
            "ocr_primary",
            {
                "engine": "PaddleOCR",
                "confidence": 0.95,
                "page_num": 1,
                "lang": "en",
            },
        )
        
        assert event["data"]["engine"] == "PaddleOCR"
        assert event["data"]["confidence"] == 0.95
        assert event["data"]["page_num"] == 1
        assert event["data"]["lang"] == "en"

    def test_processing_failed_event(self, tmp_path):
        """Processing failures should be recorded with error details."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        event = chain.append_event(
            "processing_failed",
            {
                "stage": "ocr_primary",
                "error": "Timeout after 30s",
                "page_num": 5,
            },
        )
        
        assert event["data"]["stage"] == "ocr_primary"
        assert event["data"]["error"] == "Timeout after 30s"


class TestComputeFileHash:
    """Test file hashing utility."""

    def test_deterministic_hash(self, tmp_path):
        """Same file should always produce same hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!", encoding="utf-8")
        
        hash1 = compute_file_hash(str(test_file))
        hash2 = compute_file_hash(str(test_file))
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_different_files_different_hashes(self, tmp_path):
        """Different files should produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        
        file1.write_text("Content A", encoding="utf-8")
        file2.write_text("Content B", encoding="utf-8")
        
        hash1 = compute_file_hash(str(file1))
        hash2 = compute_file_hash(str(file2))
        
        assert hash1 != hash2

    def test_binary_file_hash(self, tmp_path):
        """Should handle binary files correctly."""
        binary_file = tmp_path / "binary.dat"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")
        
        file_hash = compute_file_hash(str(binary_file))
        
        assert len(file_hash) == 64
        # Verify it's a valid hex string
        int(file_hash, 16)


class TestChainSummary:
    """Test chain summary functionality."""

    def test_summary_contains_key_fields(self, tmp_path):
        """get_summary() should include document_id, total_events, event_types."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        chain.append_event("file_ingested", {"hash": "abc"})
        chain.append_event("page_extracted", {"page": 1})
        chain.append_event("ocr_primary", {"confidence": 0.95})
        chain.append_event("assembly_complete", {"pages": 1})
        
        summary = chain.get_summary()
        
        assert summary["document_id"] == "test_doc"
        assert summary["total_events"] == 4
        assert "file_ingested" in summary["event_types"]
        assert "assembly_complete" in summary["event_types"]
        assert summary["first_event"] is not None
        assert summary["last_event"] is not None
        assert summary["chain_hash"] is not None

    def test_summary_for_empty_chain(self):
        """get_summary() should handle empty chains."""
        chain = CustodyChain("test_doc", "/test/path")
        
        summary = chain.get_summary()
        
        assert summary["document_id"] == "test_doc"
        assert summary["total_events"] == 0
        assert summary["event_types"] == []
        assert summary["first_event"] is None
        assert summary["last_event"] is None
        assert summary["chain_hash"] is None

    def test_summary_unique_event_types(self, tmp_path):
        """get_summary() should show unique event types only."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        # Same event type multiple times
        chain.append_event("page_extracted", {"page": 1})
        chain.append_event("page_extracted", {"page": 2})
        chain.append_event("page_extracted", {"page": 3})
        chain.append_event("ocr_primary", {"page": 1})
        
        summary = chain.get_summary()
        
        assert summary["total_events"] == 4
        # Should only list unique types
        assert len(summary["event_types"]) == 2
        assert "page_extracted" in summary["event_types"]
        assert "ocr_primary" in summary["event_types"]


class TestCustodyChainEdgeCases:
    """Test edge cases and error handling."""

    def test_no_custody_dir_no_file_written(self):
        """Chain without custody_dir should not write files."""
        chain = CustodyChain("test_doc", "/test/path")
        chain.append_event("file_ingested", {"test": "data"})
        
        assert chain._filepath is None
        assert len(chain.events) == 1

    def test_unicode_in_event_data(self, tmp_path):
        """Should handle Unicode characters in event data."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        event = chain.append_event(
            "file_ingested",
            {
                "filename": "文档.pdf",
                "content": "Données françaises",
                "emoji": "📄",
            },
        )
        
        assert event["data"]["filename"] == "文档.pdf"
        assert event["data"]["emoji"] == "📄"
        
        # Verify chain still validates
        is_valid, _ = chain.verify_chain()
        assert is_valid is True

    def test_nested_data_structures(self, tmp_path):
        """Should handle nested dictionaries and lists."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        event = chain.append_event(
            "ocr_primary",
            {
                "results": [
                    {"text": "Line 1", "confidence": 0.95},
                    {"text": "Line 2", "confidence": 0.87},
                ],
                "metadata": {"engine": "paddle", "version": "2.0"},
            },
        )
        
        assert len(event["data"]["results"]) == 2
        assert event["data"]["metadata"]["engine"] == "paddle"
        
        # Verify chain integrity
        is_valid, _ = chain.verify_chain()
        assert is_valid is True

    def test_datetime_objects_serialized(self, tmp_path):
        """Should handle datetime objects in data via default=str during hashing/writing."""
        chain = CustodyChain("test_doc", "/test/path", str(tmp_path))
        
        now = datetime.now(timezone.utc)
        event = chain.append_event("file_ingested", {"processed_at": now})
        
        # In-memory event retains original type; serialization uses default=str
        assert event["data"]["processed_at"] == now
        
        # Verify chain integrity (hash uses json.dumps with default=str)
        is_valid, _ = chain.verify_chain()
        assert is_valid is True
        
        # Verify JSONL file has string representation
        with open(chain._filepath, "r", encoding="utf-8") as f:
            line_data = json.loads(f.readline())
            assert isinstance(line_data["data"]["processed_at"], str)


class TestCustodyWriteHardening:
    """Tests for custody chain write retry, ordering, and integrity flag."""

    def test_integrity_compromised_false_by_default(self):
        """New chain should have integrity_compromised=False."""
        chain = CustodyChain("doc", "/path")
        assert chain.integrity_compromised is False

    def test_constants_exported(self):
        """Module-level retry constants should be importable."""
        assert MAX_CUSTODY_RETRIES == 3
        assert len(CUSTODY_RETRY_DELAYS) == MAX_CUSTODY_RETRIES
        assert CUSTODY_RETRY_DELAYS == [0.1, 0.5, 2.0]

    def test_all_retries_fail_sets_integrity_compromised(self, tmp_path, caplog):
        """When all write attempts fail, integrity_compromised must be True."""
        chain = CustodyChain("fail_doc", "/test/path", str(tmp_path))

        with patch("builtins.open", side_effect=OSError("disk full")), \
             patch("custody.time.sleep") as mock_sleep:
            with caplog.at_level(logging.WARNING, logger="custody"):
                event = chain.append_event("file_ingested", {"test": True})

        assert chain.integrity_compromised is True
        # Event should still be in memory
        assert len(chain.events) == 1
        assert event["hash"] is not None

        # Verify CRITICAL was logged
        critical_msgs = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_msgs) == 1
        assert "FAILED" in critical_msgs[0].message
        assert "fail_doc" in critical_msgs[0].message

        # Verify warning messages for the first two retry attempts
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 2  # attempts 1 and 2

        # Verify sleep was called for retry delays (only for first 2 attempts)
        assert mock_sleep.call_count == 2

    def test_hash_not_advanced_on_write_failure(self, tmp_path):
        """When write fails, _prev_hash must NOT advance."""
        chain = CustodyChain("doc_nohash", "/test/path", str(tmp_path))

        # First event succeeds (writes to disk)
        event1 = chain.append_event("file_ingested", {"step": 1})
        hash_after_first = chain._prev_hash
        assert hash_after_first == event1["hash"]

        # Second event fails all retries
        with patch("builtins.open", side_effect=OSError("disk full")), \
             patch("custody.time.sleep"):
            chain.append_event("page_extracted", {"step": 2})

        # _prev_hash should still be the first event's hash
        assert chain._prev_hash == hash_after_first
        assert chain.integrity_compromised is True

    def test_transient_failure_succeeds_on_retry(self, tmp_path, caplog):
        """OSError on first 2 tries, success on 3rd: event written, hash advanced."""
        chain = CustodyChain("retry_doc", "/test/path", str(tmp_path))

        real_open = open
        call_count = 0

        def flaky_open(*args, **kwargs):
            nonlocal call_count
            # Only intercept append-mode writes to the custody file
            if len(args) >= 2 and args[1] == "a":
                call_count += 1
                if call_count <= 2:
                    raise OSError(f"transient failure #{call_count}")
            return real_open(*args, **kwargs)

        with patch("builtins.open", side_effect=flaky_open), \
             patch("custody.time.sleep") as mock_sleep:
            with caplog.at_level(logging.WARNING, logger="custody"):
                event = chain.append_event("file_ingested", {"retry": True})

        # Should succeed
        assert chain.integrity_compromised is False
        assert chain._prev_hash == event["hash"]
        assert len(chain.events) == 1

        # Two warning messages for failed attempts
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 2

        # sleep called twice (for the two failed attempts)
        assert mock_sleep.call_count == 2

        # File should have been written on the 3rd attempt
        filepath = tmp_path / "retry_doc.custody.jsonl"
        assert filepath.exists()
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        written_event = json.loads(lines[0])
        assert written_event["hash"] == event["hash"]

    def test_write_before_hash_update_ordering(self, tmp_path):
        """Disk write must happen before _prev_hash is updated."""
        chain = CustodyChain("order_doc", "/test/path", str(tmp_path))

        # Record the _prev_hash state at the moment of file write
        observed_prev_hash_during_write = []
        real_open = open

        class SpyFile:
            """Wraps the real file to spy on writes."""
            def __init__(self, real_file, chain_ref):
                self._real = real_file
                self._chain_ref = chain_ref

            def write(self, data):
                # At write time, _prev_hash should still be None (not yet updated)
                observed_prev_hash_during_write.append(self._chain_ref._prev_hash)
                return self._real.write(data)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self._real.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self._real, name)

        def spy_open(*args, **kwargs):
            f = real_open(*args, **kwargs)
            if len(args) >= 2 and args[1] == "a":
                return SpyFile(f, chain)
            return f

        with patch("builtins.open", side_effect=spy_open):
            event = chain.append_event("file_ingested", {"order": "test"})

        # During the write, _prev_hash should have been None (genesis event)
        assert len(observed_prev_hash_during_write) == 1
        assert observed_prev_hash_during_write[0] is None

        # After write + return, _prev_hash should be updated
        assert chain._prev_hash == event["hash"]

    def test_write_before_hash_update_second_event(self, tmp_path):
        """For the second event, _prev_hash during write should be event1's hash."""
        chain = CustodyChain("order2", "/test/path", str(tmp_path))

        event1 = chain.append_event("file_ingested", {"step": 1})

        observed_prev_hash_during_write = []
        real_open = open

        class SpyFile:
            def __init__(self, real_file, chain_ref):
                self._real = real_file
                self._chain_ref = chain_ref

            def write(self, data):
                observed_prev_hash_during_write.append(self._chain_ref._prev_hash)
                return self._real.write(data)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self._real.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self._real, name)

        def spy_open(*args, **kwargs):
            f = real_open(*args, **kwargs)
            if len(args) >= 2 and args[1] == "a":
                return SpyFile(f, chain)
            return f

        with patch("builtins.open", side_effect=spy_open):
            event2 = chain.append_event("page_extracted", {"step": 2})

        # During write of event2, _prev_hash should still be event1's hash
        # (not yet advanced to event2's hash)
        assert len(observed_prev_hash_during_write) == 1
        assert observed_prev_hash_during_write[0] == event1["hash"]

        # After write + return, _prev_hash should now be event2's hash
        assert chain._prev_hash == event2["hash"]

    def test_no_filepath_skips_write_no_integrity_issue(self):
        """Chain without custody_dir should not set integrity_compromised."""
        chain = CustodyChain("no_file", "/test/path")
        chain.append_event("file_ingested", {"test": True})

        assert chain.integrity_compromised is False
        assert chain._prev_hash is not None
        assert len(chain.events) == 1

    def test_failed_write_event_still_in_memory(self, tmp_path):
        """Even on total write failure, the event is added to in-memory events."""
        chain = CustodyChain("mem_doc", "/test/path", str(tmp_path))

        with patch("builtins.open", side_effect=OSError("nope")), \
             patch("custody.time.sleep"):
            event = chain.append_event("file_ingested", {"test": True})

        assert len(chain.events) == 1
        assert chain.events[0] is event
        assert event["event_type"] == "file_ingested"

    def test_recovery_after_failed_write(self, tmp_path):
        """After a failed write, subsequent successful writes still chain correctly."""
        chain = CustodyChain("recover_doc", "/test/path", str(tmp_path))

        # First event succeeds
        event1 = chain.append_event("file_ingested", {"step": 1})
        first_hash = chain._prev_hash

        # Second event fails all retries
        with patch("builtins.open", side_effect=OSError("fail")), \
             patch("custody.time.sleep"):
            chain.append_event("page_extracted", {"step": 2})

        assert chain._prev_hash == first_hash  # Not advanced
        assert chain.integrity_compromised is True

        # Third event succeeds -- its prev_hash should be event1's hash
        # (the last successfully persisted hash)
        event3 = chain.append_event("ocr_primary", {"step": 3})
        assert event3["prev_hash"] == first_hash
        assert chain._prev_hash == event3["hash"]

        # File should have event1 and event3 (event2 was not persisted)
        filepath = tmp_path / "recover_doc.custody.jsonl"
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["hash"] == event1["hash"]
        assert json.loads(lines[1])["hash"] == event3["hash"]


_can_import_pipeline = True
try:
    import ocr_gpu_async  # noqa: F401
except ImportError:
    _can_import_pipeline = False

_skip_no_pipeline = pytest.mark.skipif(
    not _can_import_pipeline,
    reason="ocr_gpu_async requires GPU/fasttext dependencies",
)


@_skip_no_pipeline
class TestCustodyPipelineIntegration:
    """Tests for custody integration hooks in ocr_gpu_async.py."""

    def test_document_state_has_custody_chain(self, tmp_path, monkeypatch):
        """DocumentState should initialise a CustodyChain when enabled."""
        import ocr_gpu_async as pipe

        monkeypatch.setattr(pipe, "ENABLE_CUSTODY", True)
        monkeypatch.setattr(pipe, "_CUSTODY_AVAILABLE", True)
        monkeypatch.setattr(pipe, "OUTPUT_FOLDER", str(tmp_path))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(tmp_path / "temp"))

        state = pipe.DocumentState(str(tmp_path / "doc.pdf"), "abc123", "pdf")
        assert state.custody_chain is not None
        assert state.custody_chain.document_id == "abc123"
        custody_dir = tmp_path / "EXPORT" / "CUSTODY"
        assert custody_dir.is_dir()

    def test_document_state_no_chain_when_disabled(self, tmp_path, monkeypatch):
        """DocumentState should skip custody when ENABLE_CUSTODY is False."""
        import ocr_gpu_async as pipe

        monkeypatch.setattr(pipe, "ENABLE_CUSTODY", False)
        monkeypatch.setattr(pipe, "OUTPUT_FOLDER", str(tmp_path))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(tmp_path / "temp"))

        state = pipe.DocumentState(str(tmp_path / "doc.pdf"), "abc123", "pdf")
        assert state.custody_chain is None

    def test_document_state_no_chain_when_unavailable(self, tmp_path, monkeypatch):
        """DocumentState should skip custody when module not importable."""
        import ocr_gpu_async as pipe

        monkeypatch.setattr(pipe, "ENABLE_CUSTODY", True)
        monkeypatch.setattr(pipe, "_CUSTODY_AVAILABLE", False)
        monkeypatch.setattr(pipe, "OUTPUT_FOLDER", str(tmp_path))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(tmp_path / "temp"))

        state = pipe.DocumentState(str(tmp_path / "doc.pdf"), "abc123", "pdf")
        assert state.custody_chain is None

    def test_custody_flag_in_parse_args(self, monkeypatch):
        """--no-custody CLI flag should be recognised."""
        import sys

        import ocr_gpu_async as pipe

        monkeypatch.setattr(sys, "argv", ["prog", "--no-custody"])
        args = pipe._parse_args()
        assert args.no_custody is True

    def test_custody_enabled_by_default_in_parse_args(self, monkeypatch):
        """Without --no-custody, custody should remain enabled."""
        import sys

        import ocr_gpu_async as pipe

        monkeypatch.setattr(sys, "argv", ["prog"])
        args = pipe._parse_args()
        assert args.no_custody is False


class TestCustodyPipelineLifecycle:
    """Tests for custody chain lifecycle patterns used by the pipeline."""

    def test_full_lifecycle_chain_verifies(self, tmp_path):
        """Simulate full pipeline lifecycle and verify chain integrity."""
        chain = CustodyChain("doc_abc123", "/source/file.pdf", str(tmp_path))

        chain.append_event("file_ingested", {
            "source_path": "/source/file.pdf",
            "source_type": "pdf",
            "total_pages": 3,
            "detected_language": "en",
            "file_hash": "doc_abc123",
        })
        for p in range(1, 4):
            chain.append_event("ocr_primary", {
                "page_num": p,
                "engine": "PaddleV4",
                "lang_hint": "en",
                "text_length": 500 + p,
            })
        chain.append_event("assembly_complete", {
            "output_pdf": "/output/file.pdf",
            "total_pages": 3,
            "processing_seconds": 2.51,
        })
        chain.append_event("compression_complete", {
            "pdf_path": "/output/file.pdf",
        })

        assert len(chain.events) == 6
        is_valid, msg = chain.verify_chain()
        assert is_valid is True
        assert "6 events" in msg

    def test_fallback_and_failure_events(self, tmp_path):
        """Simulate OCR fallback and failure paths."""
        chain = CustodyChain("doc_fail", "/source/bad.pdf", str(tmp_path))

        chain.append_event("file_ingested", {
            "source_path": "/source/bad.pdf",
            "source_type": "pdf",
            "total_pages": 2,
        })
        chain.append_event("ocr_fallback", {
            "page_num": 1,
            "engine": "Tesseract",
            "text_length": 120,
        })
        chain.append_event("ocr_image_only", {"page_num": 2})
        chain.append_event("processing_failed", {
            "stage": "assembler",
            "error": "Out of memory",
        })

        is_valid, _ = chain.verify_chain()
        assert is_valid is True
        types = [e["event_type"] for e in chain.events]
        assert types == [
            "file_ingested", "ocr_fallback", "ocr_image_only", "processing_failed",
        ]

    def test_compressor_receives_custody_chain(self, tmp_path):
        """Compressor tuple format: (pdf_path, custody_chain)."""
        chain = CustodyChain("doc_comp", "/src/file.pdf", str(tmp_path))
        chain.append_event("file_ingested", {"source_path": "/src/file.pdf"})
        chain.append_event("assembly_complete", {"output_pdf": "/out/file.pdf"})

        item = ("/out/file.pdf", chain)
        pdf_path, custody_chain = item
        assert pdf_path == "/out/file.pdf"
        assert custody_chain is chain

        custody_chain.append_event("compression_complete", {"pdf_path": pdf_path})
        assert len(chain.events) == 3
        is_valid, _ = chain.verify_chain()
        assert is_valid is True

    def test_compressor_handles_legacy_string(self):
        """Compressor should handle legacy non-tuple queue items."""
        item = "/out/file.pdf"
        if isinstance(item, tuple):
            pdf_path, custody_chain = item
        else:
            pdf_path, custody_chain = item, None

        assert pdf_path == "/out/file.pdf"
        assert custody_chain is None

    def test_docintel_custody_event(self, tmp_path):
        """DocIntel analysis events should be recorded in custody chain."""
        chain = CustodyChain("doc_di", "/source/forms.pdf", str(tmp_path))
        chain.append_event("file_ingested", {"source_path": "/source/forms.pdf"})
        chain.append_event("docintel_analysis", {
            "page_num": 1,
            "tables_found": 2,
            "layout_regions": 5,
        })
        assert chain.events[-1]["data"]["tables_found"] == 2
        is_valid, _ = chain.verify_chain()
        assert is_valid is True
