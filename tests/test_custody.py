"""Tests for forensic chain-of-custody module."""

import ast
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pytest

from custody import (
    CANONICAL_TIMESTAMP_PATTERN,
    CUSTODY_RETRY_DELAYS,
    EVENT_TYPES,
    MAX_CUSTODY_RETRIES,
    CustodyChain,
    compute_file_hash,
    validate_event_envelope,
    verify_custody_file,
)


def _rehash_event(event: dict) -> dict:
    """Return a copy with a correctly recomputed hash for the current fields."""
    payload = {key: value for key, value in event.items() if key != "hash"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    rebuilt = dict(payload)
    rebuilt["hash"] = digest
    return rebuilt


def _fail_custody_data_appends(message):
    """Fail descriptor-bound JSONL writes without intercepting lock reads."""
    real_open = os.open

    def failing_open(path, flags, *args, **kwargs):
        path = os.fspath(path)
        if path.endswith(".custody.jsonl") and flags & (os.O_WRONLY | os.O_RDWR):
            raise OSError(message)
        return real_open(path, flags, *args, **kwargs)

    return failing_open


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

        required_fields = [
            "document_id",
            "event_type",
            "timestamp",
            "data",
            "prev_hash",
            "hash",
        ]
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
        chain1.append_event(
            "file_ingested", {"hash": "abc123", "source_path": "/test/path"}
        )
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

    def test_load_from_bytes_restores_and_verifies_chain(self, tmp_path):
        chain = CustodyChain("bytes_doc", "/test/bytes.pdf", str(tmp_path))
        chain.append_event(
            "file_ingested",
            {"source_path": "/test/bytes.pdf", "file_hash": "a" * 64},
        )
        chain.append_event("assembly_complete", {"pages": 1})

        raw = (tmp_path / "bytes_doc.custody.jsonl").read_bytes()
        loaded = CustodyChain.load_from_bytes(
            raw,
            max_bytes=len(raw),
            max_events=2,
        )

        assert loaded.document_id == "bytes_doc"
        assert loaded.source_path == "/test/bytes.pdf"
        assert loaded.events == chain.events
        assert loaded.verify_chain()[0] is True

    def test_load_from_bytes_rejects_duplicate_json_keys(self, tmp_path):
        chain = CustodyChain("duplicate_key_doc", "/test/duplicate.pdf", str(tmp_path))
        chain.append_event("file_ingested", {"source_path": "/test/duplicate.pdf"})
        path = tmp_path / "duplicate_key_doc.custody.jsonl"
        line = path.read_text(encoding="utf-8").splitlines()[0]
        duplicated = line.replace(
            '"document_id":',
            '"document_id":"forged", "document_id":',
            1,
        )
        assert duplicated != line

        with pytest.raises(ValueError, match="duplicate JSON key in custody event"):
            CustodyChain.load_from_bytes((duplicated + "\n").encode("utf-8"))

        duplicated_nested = line.replace(
            '"source_path":',
            '"source_path":"forged", "source_path":',
            1,
        )
        assert duplicated_nested != line
        with pytest.raises(ValueError, match="duplicate JSON key in custody event"):
            CustodyChain.load_from_bytes((duplicated_nested + "\n").encode("utf-8"))

        path.write_text(duplicated + "\n", encoding="utf-8")
        is_valid, message = verify_custody_file(str(path))
        assert is_valid is False
        assert "duplicate JSON key in custody event" in message

        path.write_text(duplicated_nested + "\n", encoding="utf-8")
        is_valid, message = verify_custody_file(str(path))
        assert is_valid is False
        assert "duplicate JSON key in custody event" in message

    @pytest.mark.parametrize("max_events", [0, -1, "1", True])
    def test_load_from_bytes_rejects_invalid_event_limit(self, max_events):
        with pytest.raises(ValueError, match="max_events must be a positive integer"):
            CustodyChain.load_from_bytes(b"", max_events=max_events)

    @pytest.mark.parametrize("raw", ["", bytearray(), memoryview(b"")])
    def test_load_from_bytes_requires_immutable_bytes(self, raw):
        with pytest.raises(TypeError, match="raw must be bytes"):
            CustodyChain.load_from_bytes(raw)

    @pytest.mark.parametrize("document_id", ["../evil", "nested/evil", r"nested\evil"])
    def test_durable_chain_rejects_path_like_document_id(
        self,
        tmp_path,
        document_id,
    ):
        with pytest.raises(ValueError, match="filename-safe ID"):
            CustodyChain(document_id, "/source.pdf", str(tmp_path))

    def test_load_from_bytes_enforces_byte_limit_before_parsing(self):
        with pytest.raises(ValueError, match="4-byte safety limit"):
            CustodyChain.load_from_bytes(b"{}\n{}\n", max_bytes=4)

    def test_constructor_continues_existing_verified_chain(self, tmp_path):
        """Crash recovery must append to, not restart, an existing chain."""
        chain1 = CustodyChain("resume_doc", "/test/path", str(tmp_path))
        first = chain1.append_event(
            "file_ingested",
            {"source_path": "/test/path"},
        )

        chain2 = CustodyChain("resume_doc", "/test/path", str(tmp_path))
        resumed = chain2.append_event(
            "processing_resumed",
            {"resumed_pages": [1]},
        )

        assert len(chain2.events) == 2
        assert resumed["prev_hash"] == first["hash"]
        is_valid, message = chain2.verify_chain()
        assert is_valid is True, message
        loaded = CustodyChain.load_from_file(str(tmp_path / "resume_doc.custody.jsonl"))
        is_valid, message = loaded.verify_chain()
        assert is_valid is True, message
        assert not (tmp_path / "resume_doc.custody.lock").exists()

    def test_stale_instances_reload_persisted_tip_before_append(self, tmp_path):
        """Instances created at the same tip must not fork the disk chain."""
        first_writer = CustodyChain("stale_doc", "/test/path", str(tmp_path))
        stale_writer = CustodyChain("stale_doc", "/test/path", str(tmp_path))

        first = first_writer.append_event("file_ingested", {"writer": "first"})
        second = stale_writer.append_event(
            "processing_resumed",
            {"writer": "stale"},
        )

        assert second["prev_hash"] == first["hash"]
        loaded = CustodyChain.load_from_file(str(tmp_path / "stale_doc.custody.jsonl"))
        assert len(loaded.events) == 2
        is_valid, message = loaded.verify_chain()
        assert is_valid is True, message

    def test_spawned_processes_serialize_and_reload_the_shared_tip(self, tmp_path):
        """Independent processes must produce one linear custody history."""
        chain = CustodyChain("process_race", "/test/path", str(tmp_path))
        chain.append_event("file_ingested", {"source_path": "/test/path"})

        worker_code = """
import sys
import time
from pathlib import Path
from custody import CustodyChain

custody_dir, writer = sys.argv[1:3]
chain = CustodyChain("process_race", "/test/path", custody_dir)
Path(custody_dir, f"ready-{writer}").write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 15
while not Path(custody_dir, "start-race").exists():
    if time.monotonic() >= deadline:
        raise RuntimeError("Timed out waiting to start custody race")
    time.sleep(0.01)
chain.append_event("page_extracted", {"writer": writer})
"""
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", worker_code, str(tmp_path), writer],
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for writer in ("a", "b")
        ]

        try:
            deadline = time.monotonic() + 15
            while len(list(tmp_path.glob("ready-*"))) < len(processes):
                assert (
                    time.monotonic() < deadline
                ), "custody workers did not become ready"
                time.sleep(0.01)
            (tmp_path / "start-race").write_text("start", encoding="utf-8")
            for process in processes:
                output, _ = process.communicate(timeout=20)
                assert process.returncode == 0, output
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

        loaded = CustodyChain.load_from_file(
            str(tmp_path / "process_race.custody.jsonl")
        )
        assert len(loaded.events) == 3
        is_valid, message = loaded.verify_chain()
        assert is_valid is True, message

    def test_constructor_rejects_tampered_existing_chain(self, tmp_path):
        chain = CustodyChain("tampered_resume", "/test/path", str(tmp_path))
        chain.append_event("file_ingested", {"source_path": "/test/path"})
        path = tmp_path / "tampered_resume.custody.jsonl"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["data"]["source_path"] = "/changed/path"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="failed verification"):
            CustodyChain("tampered_resume", "/test/path", str(tmp_path))

    def test_append_rejects_chain_corrupted_after_construction(self, tmp_path):
        chain = CustodyChain("late_corruption", "/test/path", str(tmp_path))
        chain.append_event("file_ingested", {"source_path": "/test/path"})
        stale = CustodyChain("late_corruption", "/test/path", str(tmp_path))
        path = tmp_path / "late_corruption.custody.jsonl"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["data"]["source_path"] = "/tampered/path"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        corrupt_bytes = path.read_bytes()

        with pytest.raises(ValueError, match="failed verification"):
            stale.append_event("processing_resumed", {"resumed_pages": [1]})

        assert path.read_bytes() == corrupt_bytes
        assert len(stale.events) == 1

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(b" \n\t", id="whitespace"),
            pytest.param(b'{"document_id": "truncated"', id="truncated-json"),
            pytest.param(b"[]\n", id="non-object-event"),
            pytest.param(b'{"document_id": "malformed"}\n', id="missing-fields"),
        ],
    )
    def test_constructor_normalizes_unreadable_existing_chain(
        self,
        tmp_path,
        payload,
    ):
        path = tmp_path / "unreadable.custody.jsonl"
        path.write_bytes(payload)

        with pytest.raises(ValueError, match="Existing custody chain unreadable"):
            CustodyChain("unreadable", "/test/path", str(tmp_path))

        assert path.read_bytes() == payload

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
            f.write("this is not json\n")
            f.write('{"also": "valid"}\n')

        is_valid, message = verify_custody_file(str(filepath))
        assert is_valid is False
        assert "Failed to load" in message

    def test_load_and_verify_accept_final_event_without_trailing_lf(self, tmp_path):
        """A valid final event missing its trailing LF still loads/validates."""
        chain = CustodyChain("no_lf_load", "/test/path", str(tmp_path))
        first = chain.append_event("file_ingested", {"source_path": "/test/path"})
        path = tmp_path / "no_lf_load.custody.jsonl"
        stripped = path.read_bytes().rstrip(b"\n")
        path.write_bytes(stripped)
        assert not path.read_bytes().endswith(b"\n")

        loaded = CustodyChain.load_from_file(str(path))
        is_valid, message = loaded.verify_chain()
        assert is_valid is True, message
        assert len(loaded.events) == 1
        assert loaded.events[0]["hash"] == first["hash"]

    def test_append_normalizes_missing_final_lf_before_next_event(self, tmp_path):
        """Next append must not concatenate JSON objects after a missing LF."""
        chain = CustodyChain("normalize_lf", "/test/path", str(tmp_path))
        first = chain.append_event("file_ingested", {"source_path": "/test/path"})
        path = tmp_path / "normalize_lf.custody.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        assert not path.read_bytes().endswith(b"\n")

        resumed = CustodyChain("normalize_lf", "/test/path", str(tmp_path))
        second = resumed.append_event("page_extracted", {"page": 1})

        raw = path.read_bytes()
        assert b"}{" not in raw
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["hash"] == first["hash"]
        assert json.loads(lines[1])["hash"] == second["hash"]
        assert raw.endswith(b"\n")
        assert second["prev_hash"] == first["hash"]

        loaded = CustodyChain.load_from_file(str(path))
        is_valid, message = loaded.verify_chain()
        assert is_valid is True, message
        assert len(loaded.events) == 2

    def test_normalize_failure_fails_closed_without_adjacent_objects(self, tmp_path):
        """If durable LF repair fails, refuse the append instead of gluing JSON."""
        chain = CustodyChain("normalize_fail", "/test/path", str(tmp_path))
        first = chain.append_event("file_ingested", {"source_path": "/test/path"})
        path = tmp_path / "normalize_fail.custody.jsonl"
        stripped = path.read_bytes().rstrip(b"\n")
        path.write_bytes(stripped)
        original = path.read_bytes()
        assert not original.endswith(b"\n")

        stale = CustodyChain("normalize_fail", "/test/path", str(tmp_path))
        with (
            patch("custody.os.write", side_effect=OSError("cannot append")),
            patch("custody.time.sleep"),
        ):
            with pytest.raises(OSError, match="write failed"):
                stale.append_event("page_extracted", {"page": 1})

        assert stale.integrity_compromised is True
        assert stale._prev_hash == first["hash"]
        assert path.read_bytes() == original
        assert b"}{" not in path.read_bytes()
        assert stale.events == [first]

        loaded = CustodyChain.load_from_file(str(path))
        is_valid, message = loaded.verify_chain()
        assert is_valid is True, message
        assert len(loaded.events) == 1


class TestEventTypeAndDataEnforcement:
    """Fail closed when event_type/data violate the registry and schema."""

    def test_append_rejects_unknown_event_type(self, tmp_path):
        chain = CustodyChain("unknown_type", "/test/path", str(tmp_path))
        with pytest.raises(ValueError, match="unknown event_type"):
            chain.append_event("not_a_registered_event", {"x": 1})
        assert chain.events == []
        assert not (tmp_path / "unknown_type.custody.jsonl").exists()

    def test_append_rejects_non_mapping_data(self, tmp_path):
        chain = CustodyChain("bad_data_append", "/test/path", str(tmp_path))
        with pytest.raises(ValueError, match="data must be a JSON object"):
            chain.append_event("file_ingested", ["not", "a", "mapping"])
        assert chain.events == []
        assert not (tmp_path / "bad_data_append.custody.jsonl").exists()

    def test_append_accepts_none_data_as_empty_object(self, tmp_path):
        chain = CustodyChain("none_data", "/test/path", str(tmp_path))
        event = chain.append_event("file_ingested", None)
        assert event["data"] == {}
        is_valid, message = chain.verify_chain()
        assert is_valid is True, message

    def test_load_rejects_unknown_event_type_on_disk(self, tmp_path):
        path = tmp_path / "tampered_type.custody.jsonl"
        payload = {
            "document_id": "tampered_type",
            "event_type": "NOT_A_REGISTERED_EVENT",
            "timestamp": "2026-01-01T00:00:00.000+00:00",
            "data": {"source_path": "/test/path"},
            "prev_hash": None,
            "hash": "a" * 64,
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown event_type"):
            CustodyChain.load_from_file(str(path))

    def test_load_rejects_non_mapping_data_on_disk(self, tmp_path):
        path = tmp_path / "tampered_data.custody.jsonl"
        payload = {
            "document_id": "tampered_data",
            "event_type": "file_ingested",
            "timestamp": "2026-01-01T00:00:00.000+00:00",
            "data": ["not", "a", "mapping"],
            "prev_hash": None,
            "hash": "b" * 64,
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="data must be a JSON object"):
            CustodyChain.load_from_file(str(path))

    def test_verify_rejects_unknown_event_type_in_memory(self):
        chain = CustodyChain("verify_unknown", "/test/path")
        chain.events.append(
            {
                "document_id": "verify_unknown",
                "event_type": "NOT_REGISTERED",
                "timestamp": "2026-01-01T00:00:00.000+00:00",
                "data": {},
                "prev_hash": None,
                "hash": "c" * 64,
            }
        )
        is_valid, message = chain.verify_chain()
        assert is_valid is False
        assert "unknown event_type" in message

    def test_verify_rejects_non_mapping_data_in_memory(self):
        chain = CustodyChain("verify_bad_data", "/test/path")
        chain.events.append(
            {
                "document_id": "verify_bad_data",
                "event_type": "file_ingested",
                "timestamp": "2026-01-01T00:00:00.000+00:00",
                "data": "not-a-mapping",
                "prev_hash": None,
                "hash": "d" * 64,
            }
        )
        is_valid, message = chain.verify_chain()
        assert is_valid is False
        assert "data must be a JSON object" in message

    def test_verify_custody_file_rejects_unknown_event_type(self, tmp_path):
        path = tmp_path / "verify_file_type.custody.jsonl"
        payload = {
            "document_id": "verify_file_type",
            "event_type": "STILL_NOT_REGISTERED",
            "timestamp": "2026-01-01T00:00:00.000+00:00",
            "data": {},
            "prev_hash": None,
            "hash": "e" * 64,
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        is_valid, message = verify_custody_file(str(path))
        assert is_valid is False
        assert "unknown event_type" in message

    def test_constructor_rejects_existing_chain_with_unknown_event_type(self, tmp_path):
        path = tmp_path / "ctor_unknown.custody.jsonl"
        payload = {
            "document_id": "ctor_unknown",
            "event_type": "BOGUS_EVENT",
            "timestamp": "2026-01-01T00:00:00.000+00:00",
            "data": {},
            "prev_hash": None,
            "hash": "f" * 64,
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Existing custody chain unreadable"):
            CustodyChain("ctor_unknown", "/test/path", str(tmp_path))
        assert path.read_text(encoding="utf-8") == json.dumps(payload) + "\n"


class TestEventEnvelopeValidation:
    """Full custody event envelope validation (schema-aligned, fail-closed)."""

    def _base_event(self, **overrides):
        event = {
            "document_id": "envelope_doc",
            "event_type": "file_ingested",
            "timestamp": "2026-01-01T00:00:00.000+00:00",
            "data": {"source_path": "/test/path"},
            "prev_hash": None,
            "hash": "a" * 64,
        }
        event.update(overrides)
        return event

    @pytest.mark.parametrize(
        "mutator, match",
        [
            pytest.param(
                lambda e: {**e, "document_id": ""},
                "document_id must be a non-empty string",
                id="empty-document-id",
            ),
            pytest.param(
                lambda e: {**e, "document_id": 123},
                "document_id must be a non-empty string",
                id="non-string-document-id",
            ),
            pytest.param(
                lambda e: {**e, "event_type": "NOT_REGISTERED"},
                "unknown event_type",
                id="unknown-event-type",
            ),
            pytest.param(
                lambda e: {**e, "timestamp": 1719792000},
                "timezone-aware RFC3339",
                id="integer-timestamp",
            ),
            pytest.param(
                lambda e: {**e, "timestamp": "2026-01-01T00:00:00.000"},
                "timezone-aware RFC3339",
                id="naive-timestamp",
            ),
            pytest.param(
                lambda e: {**e, "timestamp": "not-a-timestamp"},
                "timezone-aware RFC3339",
                id="malformed-timestamp",
            ),
            pytest.param(
                lambda e: {**e, "data": ["not", "mapping"]},
                "data must be a JSON object",
                id="non-mapping-data",
            ),
            pytest.param(
                lambda e: {**e, "prev_hash": "A" * 64},
                "prev_hash must be lowercase 64-char SHA-256 hex",
                id="uppercase-prev-hash",
            ),
            pytest.param(
                lambda e: {**e, "prev_hash": "abc"},
                "prev_hash must be lowercase 64-char SHA-256 hex",
                id="short-prev-hash",
            ),
            pytest.param(
                lambda e: {**e, "hash": "B" * 64},
                "hash must be lowercase 64-char SHA-256 hex",
                id="uppercase-hash",
            ),
            pytest.param(
                lambda e: {**e, "hash": 12},
                "hash must be lowercase 64-char SHA-256 hex",
                id="integer-hash",
            ),
            pytest.param(
                lambda e: {k: v for k, v in e.items() if k != "timestamp"},
                "missing fields",
                id="missing-timestamp",
            ),
        ],
    )
    def test_validate_event_envelope_rejects_malformed_fields(self, mutator, match):
        event = mutator(self._base_event())
        with pytest.raises(ValueError, match=match):
            validate_event_envelope(event)

    @pytest.mark.parametrize(
        "mutator, match",
        [
            pytest.param(
                lambda e: {**e, "timestamp": "2026-01-01T00:00:00.000"},
                "timezone-aware RFC3339",
                id="rehashed-naive-timestamp",
            ),
            pytest.param(
                lambda e: {**e, "document_id": ""},
                "document_id must be a non-empty string",
                id="rehashed-empty-document-id",
            ),
            pytest.param(
                lambda e: {**e, "prev_hash": "AB" * 32},
                "prev_hash must be lowercase 64-char SHA-256 hex",
                id="rehashed-uppercase-prev-hash",
            ),
            pytest.param(
                lambda e: {**e, "event_type": "STILL_BOGUS"},
                "unknown event_type",
                id="rehashed-unknown-event-type",
            ),
        ],
    )
    def test_load_and_verify_reject_correctly_rehashed_invalid_envelopes(
        self, tmp_path, mutator, match
    ):
        """Even when the hash is recomputed over bad fields, envelope fails closed."""
        invalid = _rehash_event(mutator(self._base_event()))
        # Hash is self-consistent but envelope is still invalid.
        payload_bytes = json.dumps(invalid, sort_keys=True, default=str).encode("utf-8")
        assert (
            hashlib.sha256(
                json.dumps(
                    {k: v for k, v in invalid.items() if k != "hash"},
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            == invalid["hash"]
        )
        del payload_bytes

        path = tmp_path / "bad_envelope.custody.jsonl"
        path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match=match):
            CustodyChain.load_from_file(str(path))

        is_valid, message = verify_custody_file(str(path))
        assert is_valid is False
        assert match.split()[0] in message or match in message

        memory = CustodyChain("envelope_doc", "/test/path")
        memory.events.append(invalid)
        is_valid, message = memory.verify_chain()
        assert is_valid is False
        assert match in message or match.split()[0] in message

    @pytest.mark.parametrize(
        "timestamp",
        [
            "2026-01-01T00:00:00.000Z",
            "2026-01-01T00:00:00.000+00:00",
            "2026-07-11T12:34:56.789+00:00",
            "2026-07-11T12:34:56Z",
            "2026-07-11T12:34:56-05:00",
            "2026-12-31T23:59:59.999999+00:00",
        ],
    )
    def test_accepts_z_and_offset_timestamps(self, timestamp):
        event = _rehash_event(self._base_event(timestamp=timestamp))
        validated = validate_event_envelope(event)
        assert validated["timestamp"] == timestamp

    @pytest.mark.parametrize(
        "timestamp",
        [
            pytest.param("2026-01-01 00:00:00.000Z", id="space-separator"),
            pytest.param("2026-W01-1T00:00:00Z", id="week-date"),
            pytest.param("2026-01-01T00:00:00.000z", id="lowercase-z"),
            pytest.param("2026-01-01T00:00:00.000", id="naive"),
            pytest.param("2026-01-01T00:00:60Z", id="leap-second"),
            pytest.param("2026-02-30T00:00:00Z", id="bad-calendar-date"),
            pytest.param("2026-04-31T00:00:00Z", id="april-31"),
            pytest.param("2026-01-01T24:00:00Z", id="hour-24"),
            pytest.param("2026-01-01T00:00:00+24:00", id="bad-offset-hour"),
            pytest.param("2026-01-01T00:00:00+00:60", id="bad-offset-minute"),
            pytest.param("not-a-timestamp", id="malformed"),
        ],
    )
    def test_rejects_noncanonical_timestamps(self, timestamp):
        event = _rehash_event(self._base_event(timestamp=timestamp))
        with pytest.raises(ValueError, match="timezone-aware RFC3339"):
            validate_event_envelope(event)

    def test_runtime_and_schema_agree_on_timestamp_matrix(self):
        """Runtime validator and public schema agree on accept/reject set.

        Lexical rejects use the schema pattern. Calendar-invalid but
        lexically shaped values (e.g. Feb 30) must fail *both* runtime and
        Draft7 FormatChecker date-time (requires declared rfc3339-validator).
        """
        import re

        try:
            import rfc3339_validator  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment gap
            pytest.fail(
                "rfc3339-validator is a declared requirements.txt dependency "
                "and must be installed for public schema calendar enforcement; "
                f"import failed: {exc}"
            )

        schema_path = (
            Path(__file__).resolve().parents[1] / "schemas" / "custody.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        pattern = schema["properties"]["timestamp"]["pattern"]
        assert pattern == CANONICAL_TIMESTAMP_PATTERN
        assert schema["properties"]["timestamp"]["format"] == "date-time"
        compiled = re.compile(pattern)
        format_checker = jsonschema.Draft7Validator.FORMAT_CHECKER
        assert "date-time" in format_checker.checkers

        accepted = [
            "2026-01-01T00:00:00.000Z",
            "2026-01-01T00:00:00.000+00:00",
            "2026-07-11T12:34:56.789+00:00",
            "2026-07-11T12:34:56Z",
            "2026-07-11T12:34:56-05:00",
            # Historical application-generated form must remain valid.
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        ]
        rejected_lexical = [
            "2026-01-01 00:00:00.000Z",
            "2026-W01-1T00:00:00Z",
            "2026-01-01T00:00:00.000z",
            "2026-01-01T00:00:00.000",
            "2026-01-01T00:00:60Z",
            "2026-01-01T24:00:00Z",
            "2026-01-01T00:00:00+24:00",
            "2026-01-01T00:00:00+00:60",
            "not-a-timestamp",
        ]
        # Calendar-invalid: must fail runtime AND schema format checker.
        rejected_calendar = [
            "2026-02-30T00:00:00Z",
            "2026-04-31T12:00:00+00:00",
        ]

        for timestamp in accepted:
            assert compiled.fullmatch(timestamp), timestamp
            event = _rehash_event(self._base_event(timestamp=timestamp))
            assert validate_event_envelope(event)["timestamp"] == timestamp
            jsonschema.validate(event, schema, format_checker=format_checker)

        for timestamp in rejected_lexical:
            assert compiled.fullmatch(timestamp) is None, timestamp
            event = _rehash_event(self._base_event(timestamp=timestamp))
            with pytest.raises(ValueError):
                validate_event_envelope(event)
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(
                    _rehash_event(self._base_event(timestamp=timestamp)),
                    schema,
                    format_checker=format_checker,
                )

        for timestamp in rejected_calendar:
            # Pattern alone may still match; format + runtime must not.
            assert compiled.fullmatch(timestamp), timestamp
            event = _rehash_event(self._base_event(timestamp=timestamp))
            with pytest.raises(ValueError):
                validate_event_envelope(event)
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(
                    event,
                    schema,
                    format_checker=format_checker,
                )

    def test_appended_events_validate_against_public_schema_with_format(self, tmp_path):
        schema_path = (
            Path(__file__).resolve().parents[1] / "schemas" / "custody.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        chain = CustodyChain("schema_doc", "/test/path", str(tmp_path))
        event = chain.append_event("file_ingested", {"source_path": "/test/path"})
        jsonschema.validate(
            event,
            schema,
            format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
        )
        # Second event exercises non-null prev_hash pattern.
        second = chain.append_event("page_extracted", {"page": 1})
        jsonschema.validate(
            second,
            schema,
            format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
        )
        # Generated timestamps remain the historical +00:00 form.
        assert event["timestamp"].endswith("+00:00")
        assert "T" in event["timestamp"]

    def test_schema_patterns_match_runtime_envelope_rules(self):
        schema_path = (
            Path(__file__).resolve().parents[1] / "schemas" / "custody.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["properties"]["document_id"]["minLength"] == 1
        assert schema["properties"]["timestamp"]["format"] == "date-time"
        assert (
            schema["properties"]["timestamp"]["pattern"] == CANONICAL_TIMESTAMP_PATTERN
        )
        assert schema["properties"]["hash"]["pattern"] == "^[0-9a-f]{64}$"
        prev_any = schema["properties"]["prev_hash"]["anyOf"]
        assert {"type": "null"} in prev_any
        assert any(branch.get("pattern") == "^[0-9a-f]{64}$" for branch in prev_any)
        # Public schema intentionally allows additional properties.
        assert "additionalProperties" not in schema

    def test_forensic_panel_schema_is_intentionally_incompatible(self):
        """docs/forensic/custody-schema-v1.json is a different panel contract.

        It uses JOB_STARTED-style event types and a distinct envelope
        (event_id/job_id/chain_hash).  Runtime validation targets
        schemas/custody.schema.json and must not be weakened to match it.
        """
        root = Path(__file__).resolve().parents[1]
        forensic = root / "docs" / "forensic" / "custody-schema-v1.json"
        public = root / "schemas" / "custody.schema.json"
        assert forensic.is_file()
        forensic_schema = json.loads(forensic.read_text(encoding="utf-8"))
        public_schema = json.loads(public.read_text(encoding="utf-8"))
        public_required = set(public_schema["required"])
        forensic_required = set(forensic_schema["required"])
        assert public_required != forensic_required
        assert "document_id" in public_required
        assert "document_id" not in forensic_required
        assert "event_id" in forensic_required


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
            "processing_resumed",
            "compression_failed",
        ]
        for event_type in required:
            assert event_type in EVENT_TYPES, f"Missing event type: {event_type}"

    def test_registry_matches_schema_and_covers_runtime_emitters(self):
        """Public schema and registry must admit production custody literals."""
        root = Path(__file__).resolve().parents[1]
        schema_path = root / "schemas" / "custody.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        admitted_values = schema["properties"]["event_type"]["enum"]
        admitted = set(admitted_values)
        assert len(admitted_values) == len(admitted)

        def strings_in(node):
            return {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }

        def emitted_literals(relative_path, call_names):
            tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))

            def assigned_strings(value):
                # ``dict.pop("event", default)`` names a payload key in its
                # first argument; only the default is itself an event type.
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "pop"
                    and len(value.args) >= 2
                ):
                    return strings_in(value.args[1])
                return strings_in(value)

            assignments = {
                target.id: assigned_strings(node.value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            emitted = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                if not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in call_names:
                    continue
                event_arg = node.args[0]
                if isinstance(event_arg, ast.Constant) and isinstance(
                    event_arg.value, str
                ):
                    emitted.add(event_arg.value)
                elif isinstance(event_arg, ast.Name):
                    emitted.update(assignments.get(event_arg.id, set()))
            return emitted

        runtime_emitted = set()
        for relative_path, call_names in (
            ("ocr_gpu_async.py", {"append_event"}),
            ("custody_hooks.py", {"append_event"}),
            ("ocr_local/features/language_detection.py", {"append_event"}),
            ("ocr_local/translation/custody_adapter.py", {"log_event"}),
            ("ocr_local/translation/cache.py", {"log_event"}),
            ("ocr_local/translation/quality_custody.py", {"log_event"}),
            ("ocr_local/enrichment_bus.py", {"_emit"}),
        ):
            runtime_emitted.update(emitted_literals(relative_path, call_names))

        provenance_tree = ast.parse(
            (root / "ocr_local/engines/provenance.py").read_text(encoding="utf-8")
        )
        provenance_names = {
            "ENGINE_PROVENANCE",
            "LICENSE_BLOCKED",
            "LANE_BOUNDARY_ENFORCED",
        }
        runtime_emitted.update(
            node.value.value
            for node in provenance_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in provenance_names
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )

        required_runtime = {
            "assembly_failed",
            "ENGINE_PROVENANCE",
            "LICENSE_BLOCKED",
            "LANE_BOUNDARY_ENFORCED",
            "LANGUAGE_DETECTED",
            "LANGUAGE_MIXED_SCRIPT",
            "TRANSLATION_APPLIED",
            "TRANSLATION_MODEL_CACHE",
            "PLUGIN_REGISTERED",
            "PLUGIN_REJECTED",
        }
        assert required_runtime <= runtime_emitted
        assert runtime_emitted <= set(EVENT_TYPES)
        assert admitted == set(EVENT_TYPES)

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

        with (
            patch(
                "custody.os.open",
                side_effect=_fail_custody_data_appends("disk full"),
            ),
            patch("custody.time.sleep") as mock_sleep,
        ):
            with caplog.at_level(logging.WARNING, logger="custody"):
                with pytest.raises(OSError):
                    chain.append_event("file_ingested", {"test": True})

        assert chain.integrity_compromised is True
        # A failed durable append must not be reported as an in-memory event.
        assert chain.events == []

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
        with (
            patch(
                "custody.os.open",
                side_effect=_fail_custody_data_appends("disk full"),
            ),
            patch("custody.time.sleep"),
        ):
            with pytest.raises(OSError):
                chain.append_event("page_extracted", {"step": 2})

        # _prev_hash should still be the first event's hash
        assert chain._prev_hash == hash_after_first
        assert chain.integrity_compromised is True

    def test_transient_failure_succeeds_on_retry(self, tmp_path, caplog):
        """OSError on first 2 tries, success on 3rd: event written, hash advanced."""
        chain = CustodyChain("retry_doc", "/test/path", str(tmp_path))

        real_open = os.open
        call_count = 0

        def flaky_open(path, flags, *args, **kwargs):
            nonlocal call_count
            # Only intercept create/append writes to the custody file.
            if (
                os.fspath(path).endswith(".custody.jsonl")
                and flags & os.O_RDWR
                and flags & os.O_CREAT
            ):
                call_count += 1
                if call_count <= 2:
                    raise OSError(f"transient failure #{call_count}")
            return real_open(path, flags, *args, **kwargs)

        with (
            patch("custody.os.open", side_effect=flaky_open),
            patch("custody.time.sleep") as mock_sleep,
        ):
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
        real_write = os.write

        def spy_write(fd, data):
            observed_prev_hash_during_write.append(chain._prev_hash)
            return real_write(fd, data)

        with patch("custody.os.write", side_effect=spy_write):
            event = chain.append_event("file_ingested", {"order": "test"})

        # During the write, _prev_hash should have been None (genesis event)
        assert observed_prev_hash_during_write
        assert all(value is None for value in observed_prev_hash_during_write)

        # After write + return, _prev_hash should be updated
        assert chain._prev_hash == event["hash"]

    def test_write_before_hash_update_second_event(self, tmp_path):
        """For the second event, _prev_hash during write should be event1's hash."""
        chain = CustodyChain("order2", "/test/path", str(tmp_path))

        event1 = chain.append_event("file_ingested", {"step": 1})

        observed_prev_hash_during_write = []
        real_write = os.write

        def spy_write(fd, data):
            observed_prev_hash_during_write.append(chain._prev_hash)
            return real_write(fd, data)

        with patch("custody.os.write", side_effect=spy_write):
            event2 = chain.append_event("page_extracted", {"step": 2})

        # During write of event2, _prev_hash should still be event1's hash
        # (not yet advanced to event2's hash)
        assert observed_prev_hash_during_write
        assert all(value == event1["hash"] for value in observed_prev_hash_during_write)

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

        with (
            patch(
                "custody.os.open",
                side_effect=_fail_custody_data_appends("nope"),
            ),
            patch("custody.time.sleep"),
        ):
            with pytest.raises(OSError):
                chain.append_event("file_ingested", {"test": True})

        assert chain.events == []

    def test_recovery_after_failed_write(self, tmp_path):
        """After a failed write, subsequent successful writes still chain correctly."""
        chain = CustodyChain("recover_doc", "/test/path", str(tmp_path))

        # First event succeeds
        event1 = chain.append_event("file_ingested", {"step": 1})
        first_hash = chain._prev_hash

        # Second event fails all retries
        with (
            patch(
                "custody.os.open",
                side_effect=_fail_custody_data_appends("fail"),
            ),
            patch("custody.time.sleep"),
        ):
            with pytest.raises(OSError):
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

        chain.append_event(
            "file_ingested",
            {
                "source_path": "/source/file.pdf",
                "source_type": "pdf",
                "total_pages": 3,
                "detected_language": "en",
                "file_hash": "doc_abc123",
            },
        )
        for p in range(1, 4):
            chain.append_event(
                "ocr_primary",
                {
                    "page_num": p,
                    "engine": "PaddleV4",
                    "lang_hint": "en",
                    "text_length": 500 + p,
                },
            )
        chain.append_event(
            "assembly_complete",
            {
                "output_pdf": "/output/file.pdf",
                "total_pages": 3,
                "processing_seconds": 2.51,
            },
        )
        chain.append_event(
            "compression_complete",
            {
                "pdf_path": "/output/file.pdf",
            },
        )

        assert len(chain.events) == 6
        is_valid, msg = chain.verify_chain()
        assert is_valid is True
        assert "6 events" in msg

    def test_fallback_and_failure_events(self, tmp_path):
        """Simulate OCR fallback and failure paths."""
        chain = CustodyChain("doc_fail", "/source/bad.pdf", str(tmp_path))

        chain.append_event(
            "file_ingested",
            {
                "source_path": "/source/bad.pdf",
                "source_type": "pdf",
                "total_pages": 2,
            },
        )
        chain.append_event(
            "ocr_fallback",
            {
                "page_num": 1,
                "engine": "Tesseract",
                "text_length": 120,
            },
        )
        chain.append_event("ocr_image_only", {"page_num": 2})
        chain.append_event(
            "processing_failed",
            {
                "stage": "assembler",
                "error": "Out of memory",
            },
        )

        is_valid, _ = chain.verify_chain()
        assert is_valid is True
        types = [e["event_type"] for e in chain.events]
        assert types == [
            "file_ingested",
            "ocr_fallback",
            "ocr_image_only",
            "processing_failed",
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
        chain.append_event(
            "docintel_analysis",
            {
                "page_num": 1,
                "tables_found": 2,
                "layout_regions": 5,
            },
        )
        assert chain.events[-1]["data"]["tables_found"] == 2
        is_valid, _ = chain.verify_chain()
        assert is_valid is True


def test_descriptor_bound_append_extends_selected_inode_and_tip(tmp_path: Path):
    from ocr_local.features.custody import append_event_to_bound_chain

    chain = CustodyChain("bound_doc", "/source.pdf", str(tmp_path))
    chain.append_event("file_ingested", {"file_hash": "a" * 64})
    metadata = Path(chain._filepath).stat()
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        event = append_event_to_bound_chain(
            custody_dir_fd=directory_fd,
            document_id="bound_doc",
            expected_device=metadata.st_dev,
            expected_inode=metadata.st_ino,
            expected_size=metadata.st_size,
            expected_chain_head=chain.events[-1]["hash"],
            event_type="LANGUAGE_REDETECTED",
            data={"primary_language": "en"},
        )
    finally:
        os.close(directory_fd)

    loaded = CustodyChain.load_from_file(chain._filepath)
    assert len(loaded.events) == 2
    assert loaded.events[-1] == event
    assert loaded.verify_chain()[0] is True


def test_descriptor_bound_append_refuses_same_name_replacement(tmp_path: Path):
    from ocr_local.features.custody import append_event_to_bound_chain

    chain = CustodyChain("bound_doc", "/source.pdf", str(tmp_path))
    chain.append_event("file_ingested", {"file_hash": "a" * 64})
    selected_path = Path(chain._filepath)
    metadata = selected_path.stat()
    selected_tip = chain.events[-1]["hash"]
    authorized_path = tmp_path / "authorized.custody.jsonl"
    selected_path.rename(authorized_path)
    replacement = CustodyChain("bound_doc", "/foreign.pdf", str(tmp_path))
    replacement.append_event("file_ingested", {"file_hash": "b" * 64})

    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError, match="inode changed"):
            append_event_to_bound_chain(
                custody_dir_fd=directory_fd,
                document_id="bound_doc",
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
                expected_size=metadata.st_size,
                expected_chain_head=selected_tip,
                event_type="LANGUAGE_REDETECTED",
                data={"primary_language": "en"},
            )
    finally:
        os.close(directory_fd)

    assert len(CustodyChain.load_from_file(authorized_path).events) == 1
    assert len(CustodyChain.load_from_file(replacement._filepath).events) == 1


def test_descriptor_bound_append_rolls_back_outer_name_check_failure(
    tmp_path: Path,
    monkeypatch,
):
    """A final context check failure must not leave a reported-failed event."""
    import custody as custody_module

    chain = CustodyChain("outer_swap", "/source.pdf", str(tmp_path))
    chain.append_event("file_ingested", {"file_hash": "a" * 64})
    selected_path = Path(chain._filepath)
    original = selected_path.read_bytes()
    metadata = selected_path.stat()
    selected_tip = chain.events[-1]["hash"]
    renamed_path = tmp_path / "renamed-authority.custody.jsonl"
    real_assert = custody_module._assert_chain_name_inode
    post_write_checks = 0

    def rename_on_outer_check(directory_fd, filename, expected_metadata):
        nonlocal post_write_checks
        current = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if current.st_size > metadata.st_size:
            post_write_checks += 1
            if post_write_checks == 4:
                os.rename(
                    filename,
                    renamed_path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
        return real_assert(directory_fd, filename, expected_metadata)

    monkeypatch.setattr(
        custody_module,
        "_assert_chain_name_inode",
        rename_on_outer_check,
    )
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError, match="canonical custody name disappeared"):
            custody_module.append_event_to_bound_chain(
                custody_dir_fd=directory_fd,
                document_id="outer_swap",
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
                expected_size=metadata.st_size,
                expected_chain_head=selected_tip,
                event_type="LANGUAGE_REDETECTED",
                data={"primary_language": "en"},
            )
    finally:
        os.close(directory_fd)

    assert post_write_checks == 4
    assert not selected_path.exists()
    assert renamed_path.read_bytes() == original
    rolled_back = CustodyChain.load_from_file(str(renamed_path))
    assert len(rolled_back.events) == 1
    assert rolled_back.events[-1]["hash"] == selected_tip
    assert rolled_back.verify_chain()[0] is True


def test_bound_transaction_rolls_back_non_oserror_final_authority_failure(
    tmp_path: Path,
):
    """Every final authority callback failure uses the exact append journal."""
    from ocr_local.features.custody import open_bound_chain_transaction

    chain = CustodyChain("callback_failure", "/source.pdf", str(tmp_path))
    first = chain.append_event("file_ingested", {"file_hash": "a" * 64})
    path = Path(chain._filepath)
    original = path.read_bytes()
    metadata = path.stat()
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)

    def fail_final_authority() -> None:
        raise ValueError("injected final authority change")

    try:
        with pytest.raises(ValueError, match="final authority change"):
            with open_bound_chain_transaction(
                custody_dir_fd=directory_fd,
                document_id="callback_failure",
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
                expected_size=metadata.st_size,
                expected_chain_head=first["hash"],
                final_authority_check=fail_final_authority,
            ) as transaction:
                transaction.append_event(
                    "LANGUAGE_REDETECT_PREPARED",
                    {"operation_id": "redetect_test", "phase": "prepared"},
                )
    finally:
        os.close(directory_fd)

    assert path.read_bytes() == original
    loaded = CustodyChain.load_from_file(str(path))
    assert loaded.events == [first]
    assert loaded.verify_chain()[0] is True


def test_bound_transaction_exposes_committed_outcome_after_exact_boundary(
    tmp_path: Path,
):
    """The transaction publishes committed before its cleanup phase begins."""
    from ocr_local.features.custody import open_bound_chain_transaction

    chain = CustodyChain("outcome_commit", "/source.pdf", str(tmp_path))
    first = chain.append_event("file_ingested", {"file_hash": "a" * 64})
    path = Path(chain._filepath)
    metadata = path.stat()
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    transaction = None
    try:
        with open_bound_chain_transaction(
            custody_dir_fd=directory_fd,
            document_id="outcome_commit",
            expected_device=metadata.st_dev,
            expected_inode=metadata.st_ino,
            expected_size=metadata.st_size,
            expected_chain_head=first["hash"],
        ) as transaction:
            assert transaction.outcome == "safe_uncommitted"
            transaction.append_event("processing_resumed", {"reason": "test"})
        assert transaction.outcome == "committed"
    finally:
        os.close(directory_fd)

    loaded = CustodyChain.load_from_file(str(path))
    assert [event["event_type"] for event in loaded.events] == [
        "file_ingested",
        "processing_resumed",
    ]
    assert loaded.verify_chain()[0] is True


def test_bound_transaction_savepoint_exposes_safe_uncommitted_outcome(
    tmp_path: Path,
):
    """An exact final-append rollback retains PREPARED and remains restorable."""
    from ocr_local.features.custody import open_bound_chain_transaction

    class FinalAuthoritySignal(BaseException):
        pass

    signal = FinalAuthoritySignal("injected final validation signal")
    chain = CustodyChain("outcome_safe", "/source.pdf", str(tmp_path))
    first = chain.append_event("file_ingested", {"file_hash": "a" * 64})
    path = Path(chain._filepath)
    metadata = path.stat()
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    transaction = None

    def fail_final_authority() -> None:
        raise signal

    try:
        with pytest.raises(FinalAuthoritySignal) as raised:
            with open_bound_chain_transaction(
                custody_dir_fd=directory_fd,
                document_id="outcome_safe",
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
                expected_size=metadata.st_size,
                expected_chain_head=first["hash"],
                final_authority_check=fail_final_authority,
            ) as transaction:
                transaction.append_event(
                    "LANGUAGE_REDETECT_PREPARED",
                    {"operation_id": "redetect_test", "phase": "prepared"},
                )
                transaction.mark_rollback_savepoint()
                transaction.append_event(
                    "LANGUAGE_REDETECTED",
                    {"operation_id": "redetect_test", "phase": "committed"},
                )
        assert raised.value is signal
        assert transaction.outcome == "safe_uncommitted"
    finally:
        os.close(directory_fd)

    loaded = CustodyChain.load_from_file(str(path))
    assert [event["event_type"] for event in loaded.events] == [
        "file_ingested",
        "LANGUAGE_REDETECT_PREPARED",
    ]
    assert loaded.verify_chain()[0] is True


def test_bound_transaction_foreign_tail_exposes_rollback_uncertain(
    tmp_path: Path,
):
    """A valid unjournaled tail is preserved and makes rollback uncertain."""
    import custody as custody_module

    class ForeignTailSignal(BaseException):
        pass

    signal = ForeignTailSignal("injected foreign tail")
    chain = CustodyChain("outcome_uncertain", "/source.pdf", str(tmp_path))
    first = chain.append_event("file_ingested", {"file_hash": "a" * 64})
    path = Path(chain._filepath)
    metadata = path.stat()
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    transaction = None
    try:
        with pytest.raises(ForeignTailSignal) as raised:
            with custody_module.open_bound_chain_transaction(
                custody_dir_fd=directory_fd,
                document_id="outcome_uncertain",
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
                expected_size=metadata.st_size,
                expected_chain_head=first["hash"],
            ) as transaction:
                transaction.append_event(
                    "LANGUAGE_REDETECT_PREPARED",
                    {"operation_id": "redetect_test", "phase": "prepared"},
                )
                transaction.mark_rollback_savepoint()
                committed = transaction.append_event(
                    "LANGUAGE_REDETECTED",
                    {"operation_id": "redetect_test", "phase": "committed"},
                )
                _foreign_event, encoded = custody_module._generated_event(
                    document_id=transaction.document_id,
                    prev_hash=committed["hash"],
                    event_type="processing_resumed",
                    data={"reason": "foreign-tail-control"},
                )
                view = memoryview(encoded)
                while view:
                    written = os.write(transaction.custody_fd, view)
                    assert written > 0
                    view = view[written:]
                os.fsync(transaction.custody_fd)
                raise signal
        assert raised.value is signal
        assert transaction.outcome == "rollback_uncertain"
        if hasattr(signal, "add_note"):
            assert any(
                "outcome=rollback_uncertain" in note for note in signal.__notes__
            )
    finally:
        os.close(directory_fd)

    loaded = CustodyChain.load_from_file(str(path))
    assert [event["event_type"] for event in loaded.events] == [
        "file_ingested",
        "LANGUAGE_REDETECT_PREPARED",
        "LANGUAGE_REDETECTED",
        "processing_resumed",
    ]
    assert loaded.verify_chain()[0] is True


def test_unjournaled_bytes_after_savepoint_are_rollback_uncertain(
    tmp_path: Path,
):
    """Durable bytes that never entered the journal cannot be a no-op rollback."""
    from ocr_local.features.custody import open_bound_chain_transaction

    class UnjournaledSignal(BaseException):
        pass

    signal = UnjournaledSignal("injected after unjournaled durable bytes")
    chain = CustodyChain("outcome_unjournaled", "/source.pdf", str(tmp_path))
    first = chain.append_event("file_ingested", {"file_hash": "a" * 64})
    path = Path(chain._filepath)
    metadata = path.stat()
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    transaction = None
    try:
        with pytest.raises(UnjournaledSignal) as raised:
            with open_bound_chain_transaction(
                custody_dir_fd=directory_fd,
                document_id="outcome_unjournaled",
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
                expected_size=metadata.st_size,
                expected_chain_head=first["hash"],
            ) as transaction:
                prepared = transaction.append_event(
                    "LANGUAGE_REDETECT_PREPARED",
                    {"operation_id": "redetect_test", "phase": "prepared"},
                )
                transaction.mark_rollback_savepoint()
                _foreign_event, encoded = __import__(
                    "ocr_local.features.custody", fromlist=["_generated_event"]
                )._generated_event(
                    document_id=transaction.document_id,
                    prev_hash=prepared["hash"],
                    event_type="LANGUAGE_REDETECTED",
                    data={"operation_id": "redetect_test", "phase": "committed"},
                )
                view = memoryview(encoded)
                while view:
                    written = os.write(transaction.custody_fd, view)
                    assert written > 0
                    view = view[written:]
                os.fsync(transaction.custody_fd)
                raise signal
        assert raised.value is signal
        assert transaction.outcome == "rollback_uncertain"
    finally:
        os.close(directory_fd)

    loaded = CustodyChain.load_from_file(str(path))
    assert [event["event_type"] for event in loaded.events] == [
        "file_ingested",
        "LANGUAGE_REDETECT_PREPARED",
        "LANGUAGE_REDETECTED",
    ]
    assert loaded.verify_chain()[0] is True


def test_append_rolls_back_when_canonical_inode_gains_hardlink(
    tmp_path: Path,
    monkeypatch,
):
    """A link-count drift during fsync cannot produce a reported-success event."""
    import custody as custody_module

    chain = CustodyChain("link_drift", "/source.pdf", str(tmp_path))
    first = chain.append_event("file_ingested", {"file_hash": "a" * 64})
    path = Path(chain._filepath)
    alias = tmp_path / "link-drift-alias.jsonl"
    original = path.read_bytes()
    original_size = len(original)
    real_fsync = os.fsync
    linked = False

    def link_during_append_fsync(fd):
        nonlocal linked
        metadata = os.fstat(fd)
        if not linked and metadata.st_size > original_size:
            os.link(path, alias)
            linked = True
        return real_fsync(fd)

    monkeypatch.setattr(custody_module.os, "fsync", link_during_append_fsync)
    with pytest.raises(OSError, match="canonical custody name changed"):
        chain.append_event("page_extracted", {"page": 1})

    assert linked is True
    assert chain.integrity_compromised is True
    assert chain.events == [first]
    assert chain._prev_hash == first["hash"]
    assert path.read_bytes() == original
    assert alias.read_bytes() == original
    alias.unlink()
    loaded = CustodyChain.load_from_file(str(path))
    assert loaded.events == [first]
    assert loaded.verify_chain()[0] is True


def test_append_rolls_back_when_custody_parent_directory_is_replaced(
    tmp_path: Path,
    monkeypatch,
):
    """A renamed custody directory cannot become an orphan commit target."""
    import custody as custody_module

    custody_dir = tmp_path / "custody"
    custody_dir.mkdir()
    chain = CustodyChain("parent_swap", "/source.pdf", str(custody_dir))
    orphan = tmp_path / "authorized-custody"
    real_append = custody_module._append_event_to_locked_fd
    swapped = False

    def swap_parent_after_durable_append(**kwargs):
        nonlocal swapped
        event = real_append(**kwargs)
        if not swapped:
            swapped = True
            custody_dir.rename(orphan)
            custody_dir.mkdir()
        return event

    monkeypatch.setattr(
        custody_module,
        "_append_event_to_locked_fd",
        swap_parent_after_durable_append,
    )

    with pytest.raises(OSError, match="custody directory changed"):
        chain.append_event("file_ingested", {"source_hash": "a" * 64})

    assert swapped
    assert not (custody_dir / "parent_swap.custody.jsonl").exists()
    orphan_chain = orphan / "parent_swap.custody.jsonl"
    assert orphan_chain.exists()
    assert orphan_chain.read_bytes() == b""
    assert chain.events == []
    assert chain._prev_hash is None
    assert chain.integrity_compromised is True


def test_load_from_file_rejects_fifo_without_blocking(tmp_path: Path):
    """A special-file custody name must fail before a blocking read."""
    fifo = tmp_path / "fifo.custody.jsonl"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError):
        pytest.skip("FIFOs are unavailable")

    started = time.monotonic()
    with pytest.raises(OSError, match="single-link regular file"):
        CustodyChain.load_from_file(str(fifo))
    assert time.monotonic() - started < 1.0


def test_replacing_legacy_lock_name_cannot_split_custody_writers(tmp_path: Path):
    """All writers serialize on the evidence inode, never a replaceable sidecar."""
    chain = CustodyChain("inode_lock", "/source.pdf", str(tmp_path))
    chain.append_event("file_ingested", {"file_hash": "a" * 64})
    legacy_lock = tmp_path / "inode_lock.custody.lock"
    legacy_lock.write_bytes(b"legacy lock name")

    paused_worker = r"""
import sys
import time
from pathlib import Path
import custody

custody_dir = Path(sys.argv[1])
real_lock = custody._lock_file_strict
paused = False

def pause_first_lock(lock_file):
    global paused
    real_lock(lock_file)
    if paused:
        return
    paused = True
    (custody_dir / "a-locked").write_text("locked", encoding="utf-8")
    deadline = time.monotonic() + 15
    while not (custody_dir / "release-a").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for release")
        time.sleep(0.01)

custody._lock_file_strict = pause_first_lock
chain = custody.CustodyChain("inode_lock", "/source.pdf", str(custody_dir))
chain.append_event("page_extracted", {"writer": "a"})
(custody_dir / "a-done").write_text("done", encoding="utf-8")
"""
    second_worker = r"""
import sys
from pathlib import Path
from custody import CustodyChain

custody_dir = Path(sys.argv[1])
(custody_dir / "b-started").write_text("started", encoding="utf-8")
chain = CustodyChain("inode_lock", "/source.pdf", str(custody_dir))
(custody_dir / "b-constructed").write_text("constructed", encoding="utf-8")
chain.append_event("page_extracted", {"writer": "b"})
(custody_dir / "b-done").write_text("done", encoding="utf-8")
"""
    repo_root = Path(__file__).resolve().parents[1]
    first = subprocess.Popen(
        [sys.executable, "-c", paused_worker, str(tmp_path)],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    second = None
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "a-locked").exists():
            assert time.monotonic() < deadline, "first writer did not acquire lock"
            time.sleep(0.01)

        legacy_lock.rename(tmp_path / "retired.custody.lock")
        legacy_lock.write_bytes(b"replacement lock name")
        second = subprocess.Popen(
            [sys.executable, "-c", second_worker, str(tmp_path)],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while not (tmp_path / "b-started").exists():
            assert time.monotonic() < deadline, "second writer did not start"
            time.sleep(0.01)
        time.sleep(0.5)
        assert not (tmp_path / "b-constructed").exists()

        (tmp_path / "release-a").write_text("release", encoding="utf-8")
        first_output, _ = first.communicate(timeout=20)
        second_output, _ = second.communicate(timeout=20)
        assert first.returncode == 0, first_output
        assert second.returncode == 0, second_output
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    loaded = CustodyChain.load_from_file(str(tmp_path / "inode_lock.custody.jsonl"))
    assert len(loaded.events) == 3
    assert {event["data"].get("writer") for event in loaded.events[1:]} == {
        "a",
        "b",
    }
    assert loaded.verify_chain()[0] is True
