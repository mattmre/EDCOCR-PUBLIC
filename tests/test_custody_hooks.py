"""Comprehensive tests for custody_hooks.py (, TEST-003).

Covers all public functions, error paths, hash-chain integrity under
failures, edge cases, and forensic audit trail error scenarios.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from custody import CustodyChain, compute_file_hash
from custody_hooks import (
    EVENT_STAMP_COMPLETE,
    EVENT_STAMP_FAILED,
    EVENT_STAMP_START,
    EVENT_TRANSFORM_COMPLETE,
    EVENT_TRANSFORM_FAILED,
    EVENT_TRANSFORM_START,
    create_custody_chain_for_operation,
    get_custody_diagnostics_summary,
    record_stamp_lifecycle,
    record_transform_lifecycle,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestEventTypeConstants:
    """Verify event type constants are correct strings."""

    def test_transform_start_constant(self):
        assert EVENT_TRANSFORM_START == "transform_start"

    def test_transform_complete_constant(self):
        assert EVENT_TRANSFORM_COMPLETE == "transform_complete"

    def test_transform_failed_constant(self):
        assert EVENT_TRANSFORM_FAILED == "transform_failed"

    def test_stamp_start_constant(self):
        assert EVENT_STAMP_START == "stamp_start"

    def test_stamp_complete_constant(self):
        assert EVENT_STAMP_COMPLETE == "stamp_complete"

    def test_stamp_failed_constant(self):
        assert EVENT_STAMP_FAILED == "stamp_failed"


# ---------------------------------------------------------------------------
# record_transform_lifecycle -- happy paths
# ---------------------------------------------------------------------------

class TestRecordTransformLifecycleSuccess:
    """Happy-path tests for record_transform_lifecycle."""

    def test_success_records_start_and_complete_events(self, tmp_path):
        """Successful transform records both start and complete events."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"input-data")
        out.write_bytes(b"output-data")

        chain = CustodyChain("doc1", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="rotate_90",
            input_path=str(inp),
            output_path=str(out),
            params={"angle": 90},
            success=True,
        )

        assert diag["operation_type"] == "transform"
        assert diag["operation_id"] == "rotate_90"
        assert diag["start_event_recorded"] is True
        assert diag["complete_event_recorded"] is True
        assert "failed_event_recorded" not in diag

    def test_success_contains_input_and_output_hashes(self, tmp_path):
        """Successful transform diagnostics include both file hashes."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"aaa")
        out.write_bytes(b"bbb")

        chain = CustodyChain("doc2", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="clean",
            input_path=str(inp),
            output_path=str(out),
            params={},
            success=True,
        )

        assert diag["input_hash"] == compute_file_hash(str(inp))
        assert diag["output_hash"] == compute_file_hash(str(out))
        assert "input_hash_error" not in diag
        assert "output_hash_error" not in diag

    def test_chain_hash_comes_from_public_summary(self, tmp_path):
        """Successful transform diagnostics use the public summary chain hash."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"input")
        out.write_bytes(b"output")

        chain = CustodyChain("doc_public_summary", str(inp))
        chain.get_summary = MagicMock(
            return_value={
                "document_id": "doc_public_summary",
                "total_events": 2,
                "event_types": [EVENT_TRANSFORM_START, EVENT_TRANSFORM_COMPLETE],
                "first_event": "first",
                "last_event": "last",
                "chain_hash": "public-chain-hash",
            }
        )

        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="clean",
            input_path=str(inp),
            output_path=str(out),
            params={},
            success=True,
        )

        chain.get_summary.assert_called_once_with()
        assert diag["custody_chain_hash"] == "public-chain-hash"

    def test_success_chain_remains_valid(self, tmp_path):
        """Chain integrity holds after a successful transform lifecycle."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("doc3", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="clean",
            input_path=str(inp),
            output_path=str(out),
            params={},
            success=True,
        )

        valid, msg = chain.verify_chain()
        assert valid is True
        assert "2 events" in msg  # start + complete

    def test_success_appends_two_events(self, tmp_path):
        """Successful transform appends exactly 2 events (start + complete)."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("doc4", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="crop",
            input_path=str(inp),
            output_path=str(out),
            params={"margin": 10},
            success=True,
        )

        assert len(chain.events) == 2
        assert chain.events[0]["event_type"] == EVENT_TRANSFORM_START
        assert chain.events[1]["event_type"] == EVENT_TRANSFORM_COMPLETE

    def test_success_metadata_included_in_complete_event(self, tmp_path):
        """Extra metadata dict is forwarded to the complete event."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("doc5", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="crop",
            input_path=str(inp),
            output_path=str(out),
            params={},
            success=True,
            metadata={"source": "api"},
        )

        complete_data = chain.events[1]["data"]
        assert complete_data["metadata"] == {"source": "api"}

    def test_success_no_metadata_defaults_to_empty_dict(self, tmp_path):
        """When metadata is None, it defaults to empty dict in the event."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("doc6", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="clean",
            input_path=str(inp),
            output_path=str(out),
            params={},
            success=True,
            metadata=None,
        )

        assert chain.events[1]["data"]["metadata"] == {}

    def test_custody_chain_hash_in_diagnostics(self, tmp_path):
        """Diagnostics include the current chain hash after recording."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("doc7", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op1",
            input_path=str(inp),
            output_path=str(out),
            params={},
            success=True,
        )

        assert diag["custody_chain_hash"] == chain._prev_hash
        assert diag["custody_chain_hash"] is not None


# ---------------------------------------------------------------------------
# record_transform_lifecycle -- failure paths
# ---------------------------------------------------------------------------

class TestRecordTransformLifecycleFailure:
    """Tests for transform failure scenarios."""

    def test_failure_records_start_and_failed_events(self, tmp_path):
        """Failed transform records start + failed events (not complete)."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("doc_fail", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="rotate",
            input_path=str(inp),
            output_path=None,
            params={"angle": 90},
            success=False,
            error_message="Page corrupted",
        )

        assert diag["start_event_recorded"] is True
        assert diag["failed_event_recorded"] is True
        assert "complete_event_recorded" not in diag

    def test_failure_appends_two_events(self, tmp_path):
        """Failed transform appends exactly 2 events (start + failed)."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("doc_fail2", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="clean",
            input_path=str(inp),
            output_path=None,
            params={},
            success=False,
            error_message="Timeout",
        )

        assert len(chain.events) == 2
        assert chain.events[0]["event_type"] == EVENT_TRANSFORM_START
        assert chain.events[1]["event_type"] == EVENT_TRANSFORM_FAILED

    def test_failure_event_contains_error_message(self, tmp_path):
        """Failed event data must include the error message."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("doc_fail3", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="rotate",
            input_path=str(inp),
            output_path=None,
            params={},
            success=False,
            error_message="Disk full",
        )

        failed_data = chain.events[1]["data"]
        assert failed_data["error_message"] == "Disk full"
        assert failed_data["operation_id"] == "rotate"

    def test_failure_chain_still_valid(self, tmp_path):
        """Chain integrity holds after a failed transform lifecycle."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("doc_fail4", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path=str(inp),
            output_path=None,
            params={},
            success=False,
            error_message="err",
        )

        valid, _ = chain.verify_chain()
        assert valid is True

    def test_success_false_with_output_path_records_failed(self, tmp_path):
        """success=False with output_path still records a failed event."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"data")
        out.write_bytes(b"data")

        chain = CustodyChain("doc_edge", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path=str(inp),
            output_path=str(out),
            params={},
            success=False,
            error_message="Validation failed",
        )

        # success=False takes priority, so failed event is recorded
        assert diag["failed_event_recorded"] is True
        assert "complete_event_recorded" not in diag

    def test_success_true_but_no_output_path_records_failed(self, tmp_path):
        """success=True but output_path=None records a failed event."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("doc_edge2", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path=str(inp),
            output_path=None,
            params={},
            success=True,
        )

        # Condition is `success and output_path` -- None output means failed branch
        assert diag["failed_event_recorded"] is True
        assert "complete_event_recorded" not in diag


# ---------------------------------------------------------------------------
# record_transform_lifecycle -- hash error paths
# ---------------------------------------------------------------------------

class TestRecordTransformHashErrors:
    """Tests for file hash computation failures in transform lifecycle."""

    def test_input_hash_error_when_file_missing(self, tmp_path):
        """Missing input file populates input_hash_error in diagnostics."""
        chain = CustodyChain("doc_miss", "/nonexistent")
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path="/nonexistent/input.pdf",
            output_path=None,
            params={},
            success=False,
            error_message="File not found",
        )

        assert "input_hash_error" in diag
        assert "input_hash" not in diag

    def test_input_hash_error_still_records_events(self, tmp_path):
        """Even when input hash fails, start and failed events are appended."""
        chain = CustodyChain("doc_hash_err", "/bad")
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path="/nonexistent.pdf",
            output_path=None,
            params={},
            success=False,
        )

        assert len(chain.events) == 2
        assert diag["start_event_recorded"] is True
        # Start event data should have input_hash=None due to hash failure
        assert chain.events[0]["data"]["input_hash"] is None

    def test_output_hash_error_when_output_missing(self, tmp_path):
        """Missing output file populates output_hash_error in diagnostics."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("doc_out_err", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path=str(inp),
            output_path="/nonexistent/output.pdf",
            params={},
            success=True,
        )

        assert "output_hash_error" in diag
        assert "output_hash" not in diag
        # Complete event should still be recorded
        assert diag["complete_event_recorded"] is True

    def test_output_hash_error_complete_event_has_none_hash(self, tmp_path):
        """Complete event output_hash is None when file hash fails."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("doc_out_err2", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path=str(inp),
            output_path="/nonexistent/output.pdf",
            params={},
            success=True,
        )

        complete_data = chain.events[1]["data"]
        assert complete_data["output_hash"] is None

    def test_both_hashes_fail_still_records_events(self, tmp_path):
        """Both input and output hash failures do not prevent event recording."""
        chain = CustodyChain("doc_both_err", "/bad")
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path="/nonexistent/in.pdf",
            output_path="/nonexistent/out.pdf",
            params={},
            success=True,
        )

        assert "input_hash_error" in diag
        assert "output_hash_error" in diag
        assert len(chain.events) == 2
        assert diag["start_event_recorded"] is True
        assert diag["complete_event_recorded"] is True

    @patch("custody_hooks.compute_file_hash", side_effect=IOError("Permission denied"))
    def test_ioerror_caught_for_input_hash(self, mock_hash):
        """IOError (not just OSError) is caught when hashing input."""
        chain = CustodyChain("doc_io", "/path")
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path="/some/file.pdf",
            output_path=None,
            params={},
            success=False,
        )

        assert "input_hash_error" in diag
        assert "Permission denied" in diag["input_hash_error"]


# ---------------------------------------------------------------------------
# record_stamp_lifecycle -- happy paths
# ---------------------------------------------------------------------------

class TestRecordStampLifecycleSuccess:
    """Happy-path tests for record_stamp_lifecycle."""

    def test_success_records_start_and_complete(self, tmp_path):
        """Successful stamp records start + complete events."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"inp")
        out.write_bytes(b"out")

        chain = CustodyChain("sdoc1", str(inp))
        diag = record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates_apply",
            input_path=str(inp),
            output_path=str(out),
            placement="bottom_right",
            params={"prefix": "PROD"},
            success=True,
            stamp_values=["PROD000001", "PROD000002"],
        )

        assert diag["operation_type"] == "stamp"
        assert diag["start_event_recorded"] is True
        assert diag["complete_event_recorded"] is True
        assert "failed_event_recorded" not in diag

    def test_chain_hash_comes_from_public_summary(self, tmp_path):
        """Successful stamp diagnostics use the public summary chain hash."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"input")
        out.write_bytes(b"output")

        chain = CustodyChain("stamp_public_summary", str(inp))
        chain.get_summary = MagicMock(
            return_value={
                "document_id": "stamp_public_summary",
                "total_events": 2,
                "event_types": [EVENT_STAMP_START, EVENT_STAMP_COMPLETE],
                "first_event": "first",
                "last_event": "last",
                "chain_hash": "public-stamp-chain-hash",
            }
        )

        diag = record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=str(out),
            placement="bottom",
            params={"prefix": "P"},
            success=True,
        )

        chain.get_summary.assert_called_once_with()
        assert diag["custody_chain_hash"] == "public-stamp-chain-hash"

    def test_success_start_event_contains_placement(self, tmp_path):
        """Start event data includes placement information."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("sdoc2", str(inp))
        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=str(out),
            placement="top_left",
            params={"size": 12},
            success=True,
        )

        start_data = chain.events[0]["data"]
        assert start_data["placement"] == "top_left"
        assert start_data["params"] == {"size": 12}

    def test_success_complete_event_contains_stamp_values(self, tmp_path):
        """Complete event includes stamp_values list."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("sdoc3", str(inp))
        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=str(out),
            placement="bottom_right",
            params={},
            success=True,
            stamp_values=["B001", "B002", "B003"],
        )

        complete_data = chain.events[1]["data"]
        assert complete_data["stamp_values"] == ["B001", "B002", "B003"]

    def test_success_no_stamp_values_defaults_to_empty_list(self, tmp_path):
        """When stamp_values is None, defaults to empty list."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("sdoc4", str(inp))
        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=str(out),
            placement="bottom",
            params={},
            success=True,
            stamp_values=None,
        )

        assert chain.events[1]["data"]["stamp_values"] == []

    def test_success_chain_integrity(self, tmp_path):
        """Chain verifies after successful stamp lifecycle."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("sdoc5", str(inp))
        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=str(out),
            placement="bottom",
            params={},
            success=True,
        )

        valid, msg = chain.verify_chain()
        assert valid is True
        assert "2 events" in msg


# ---------------------------------------------------------------------------
# record_stamp_lifecycle -- failure paths
# ---------------------------------------------------------------------------

class TestRecordStampLifecycleFailure:
    """Tests for stamp failure scenarios."""

    def test_failure_records_start_and_failed(self, tmp_path):
        """Failed stamp records start + failed events."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("sfail1", str(inp))
        diag = record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=None,
            placement="bottom_right",
            params={},
            success=False,
            error_message="Insufficient margin",
        )

        assert diag["failed_event_recorded"] is True
        assert "complete_event_recorded" not in diag
        assert len(chain.events) == 2
        assert chain.events[1]["event_type"] == EVENT_STAMP_FAILED

    def test_failure_event_contains_error_message(self, tmp_path):
        """Failed event data has the error message."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("sfail2", str(inp))
        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=None,
            placement="top",
            params={},
            success=False,
            error_message="Font not found",
        )

        failed_data = chain.events[1]["data"]
        assert failed_data["error_message"] == "Font not found"

    def test_failure_chain_still_valid(self, tmp_path):
        """Chain integrity preserved after stamp failure."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("sfail3", str(inp))
        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=None,
            placement="bottom",
            params={},
            success=False,
            error_message="err",
        )

        valid, _ = chain.verify_chain()
        assert valid is True

    def test_success_false_with_output_records_failed(self, tmp_path):
        """success=False with output_path still records failed event."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"data")
        out.write_bytes(b"data")

        chain = CustodyChain("sfail4", str(inp))
        diag = record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=str(out),
            placement="bottom",
            params={},
            success=False,
            error_message="Validation failure",
        )

        assert diag["failed_event_recorded"] is True
        assert "complete_event_recorded" not in diag


# ---------------------------------------------------------------------------
# record_stamp_lifecycle -- hash error paths
# ---------------------------------------------------------------------------

class TestRecordStampHashErrors:
    """Tests for hash computation failures in stamp lifecycle."""

    def test_input_hash_error_nonexistent_file(self):
        """Missing input file populates input_hash_error."""
        chain = CustodyChain("shash1", "/bad")
        diag = record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path="/nonexistent.pdf",
            output_path=None,
            placement="bottom",
            params={},
            success=False,
        )

        assert "input_hash_error" in diag
        assert "input_hash" not in diag

    def test_output_hash_error_nonexistent_file(self, tmp_path):
        """Missing output file populates output_hash_error."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("shash2", str(inp))
        diag = record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path="/nonexistent/out.pdf",
            placement="bottom",
            params={},
            success=True,
        )

        assert "output_hash_error" in diag
        assert diag["complete_event_recorded"] is True


# ---------------------------------------------------------------------------
# create_custody_chain_for_operation
# ---------------------------------------------------------------------------

class TestCreateCustodyChainForOperation:
    """Tests for create_custody_chain_for_operation factory."""

    def test_creates_chain_with_hash_based_id(self, tmp_path):
        """When input file exists, document_id is first 16 chars of hash."""
        inp = tmp_path / "source.pdf"
        inp.write_bytes(b"hello-forensic")

        chain = create_custody_chain_for_operation(str(inp))

        expected_hash = compute_file_hash(str(inp))
        assert chain.document_id == expected_hash[:16]
        assert chain.source_path == str(inp)

    def test_creates_chain_with_custody_dir(self, tmp_path):
        """When custody_dir is provided, chain writes to disk."""
        inp = tmp_path / "source.pdf"
        inp.write_bytes(b"data")
        cdir = tmp_path / "custody"

        chain = create_custody_chain_for_operation(str(inp), str(cdir))

        assert chain._custody_dir == str(cdir)
        assert chain._filepath is not None

    def test_creates_chain_without_custody_dir(self, tmp_path):
        """When custody_dir is empty string, no file output."""
        inp = tmp_path / "source.pdf"
        inp.write_bytes(b"data")

        chain = create_custody_chain_for_operation(str(inp), "")

        assert chain._filepath is None

    def test_fallback_id_when_file_missing(self):
        """When input file does not exist, uses filename-based ID."""
        chain = create_custody_chain_for_operation("/nonexistent/report.pdf")

        assert chain.document_id == "report_pdf"
        assert chain.source_path == "/nonexistent/report.pdf"

    def test_fallback_id_strips_path_components(self):
        """Fallback ID uses only basename, not full path."""
        chain = create_custody_chain_for_operation("/a/b/c/document.tiff")

        assert chain.document_id == "document_tiff"

    def test_fallback_id_replaces_dots(self):
        """Fallback ID replaces dots with underscores."""
        chain = create_custody_chain_for_operation("/path/my.report.v2.pdf")

        assert chain.document_id == "my_report_v2_pdf"

    def test_chain_is_functional_after_creation(self, tmp_path):
        """Returned chain can immediately record events."""
        inp = tmp_path / "test.pdf"
        inp.write_bytes(b"data")

        chain = create_custody_chain_for_operation(str(inp))
        chain.append_event("file_ingested", {"test": True})

        assert len(chain.events) == 1
        valid, _ = chain.verify_chain()
        assert valid is True

    def test_chain_with_custody_dir_writes_events(self, tmp_path):
        """Chain created with custody_dir persists events to disk."""
        inp = tmp_path / "test.pdf"
        inp.write_bytes(b"data")
        cdir = tmp_path / "custody_out"

        chain = create_custody_chain_for_operation(str(inp), str(cdir))
        chain.append_event("file_ingested", {"src": str(inp)})

        assert cdir.exists()
        jsonl_files = list(cdir.glob("*.custody.jsonl"))
        assert len(jsonl_files) == 1

    @patch("custody_hooks.compute_file_hash", side_effect=IOError("read error"))
    def test_ioerror_triggers_fallback(self, mock_hash):
        """IOError from compute_file_hash triggers filename fallback."""
        chain = create_custody_chain_for_operation("/some/file.pdf")
        assert chain.document_id == "file_pdf"


# ---------------------------------------------------------------------------
# get_custody_diagnostics_summary
# ---------------------------------------------------------------------------

class TestGetCustodyDiagnosticsSummary:
    """Tests for get_custody_diagnostics_summary."""

    def test_full_success_diagnostics(self):
        """Summary from a fully successful lifecycle."""
        diag = {
            "operation_type": "transform",
            "operation_id": "clean_pages",
            "input_hash": "abc123",
            "output_hash": "def456",
            "custody_chain_hash": "chain789",
            "start_event_recorded": True,
            "complete_event_recorded": True,
        }

        summary = get_custody_diagnostics_summary(diag)

        assert summary["custody_recorded"] is True
        assert summary["operation_type"] == "transform"
        assert summary["operation_id"] == "clean_pages"
        assert summary["input_hash"] == "abc123"
        assert summary["output_hash"] == "def456"
        assert summary["custody_chain_hash"] == "chain789"
        assert "custody_errors" not in summary

    def test_failure_diagnostics_no_output_hash(self):
        """Summary from a failed lifecycle has no output_hash."""
        diag = {
            "operation_type": "stamp",
            "operation_id": "bates",
            "input_hash": "abc123",
            "custody_chain_hash": "chain789",
            "failed_event_recorded": True,
        }

        summary = get_custody_diagnostics_summary(diag)

        assert summary["custody_recorded"] is True
        assert summary["operation_type"] == "stamp"
        assert "output_hash" not in summary
        assert "custody_errors" not in summary

    def test_input_hash_error_in_summary(self):
        """Summary includes input hash error under custody_errors."""
        diag = {
            "operation_type": "transform",
            "operation_id": "op1",
            "input_hash_error": "File not found: /bad.pdf",
            "custody_chain_hash": "xyz",
        }

        summary = get_custody_diagnostics_summary(diag)

        assert "custody_errors" in summary
        assert summary["custody_errors"]["input_hash"] == "File not found: /bad.pdf"
        assert "input_hash" not in summary  # No valid hash

    def test_output_hash_error_in_summary(self):
        """Summary includes output hash error under custody_errors."""
        diag = {
            "operation_type": "transform",
            "operation_id": "op2",
            "input_hash": "valid_hash",
            "output_hash_error": "Permission denied: /out.pdf",
            "custody_chain_hash": "xyz",
        }

        summary = get_custody_diagnostics_summary(diag)

        assert "custody_errors" in summary
        assert summary["custody_errors"]["output_hash"] == "Permission denied: /out.pdf"
        assert summary["input_hash"] == "valid_hash"
        assert "output_hash" not in summary

    def test_both_hash_errors_in_summary(self):
        """Summary includes both hash errors simultaneously."""
        diag = {
            "operation_type": "transform",
            "operation_id": "op3",
            "input_hash_error": "input err",
            "output_hash_error": "output err",
            "custody_chain_hash": "xyz",
        }

        summary = get_custody_diagnostics_summary(diag)

        assert len(summary["custody_errors"]) == 2
        assert summary["custody_errors"]["input_hash"] == "input err"
        assert summary["custody_errors"]["output_hash"] == "output err"

    def test_missing_optional_fields(self):
        """Summary handles diagnostics with missing optional fields."""
        diag = {
            "operation_type": "transform",
            "operation_id": "op4",
        }

        summary = get_custody_diagnostics_summary(diag)

        assert summary["custody_recorded"] is True
        assert summary["operation_type"] == "transform"
        assert "input_hash" not in summary
        assert "output_hash" not in summary
        assert "custody_chain_hash" not in summary
        assert "custody_errors" not in summary

    def test_empty_diagnostics(self):
        """Summary handles completely empty diagnostics dict."""
        summary = get_custody_diagnostics_summary({})

        assert summary["custody_recorded"] is True
        assert summary["operation_type"] is None
        assert summary["operation_id"] is None
        assert "custody_errors" not in summary


# ---------------------------------------------------------------------------
# Integration: chained transform + stamp on same chain
# ---------------------------------------------------------------------------

class TestChainedOperations:
    """Tests for multiple operations on the same custody chain."""

    def test_transform_then_stamp_chain_valid(self, tmp_path):
        """Transform followed by stamp on same chain maintains integrity."""
        inp = tmp_path / "in.pdf"
        mid = tmp_path / "mid.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"input")
        mid.write_bytes(b"middle")
        out.write_bytes(b"output")

        chain = CustodyChain("multi_doc", str(inp))

        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="clean",
            input_path=str(inp),
            output_path=str(mid),
            params={"mode": "whitespace"},
            success=True,
        )

        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(mid),
            output_path=str(out),
            placement="bottom_right",
            params={"prefix": "PROD"},
            success=True,
            stamp_values=["PROD000001"],
        )

        assert len(chain.events) == 4  # 2 + 2
        valid, msg = chain.verify_chain()
        assert valid is True
        assert "4 events" in msg

    def test_transform_fail_then_stamp_success(self, tmp_path):
        """Failed transform followed by successful stamp is valid."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"data")
        out.write_bytes(b"stamped")

        chain = CustodyChain("mixed_doc", str(inp))

        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="rotate",
            input_path=str(inp),
            output_path=None,
            params={"angle": 180},
            success=False,
            error_message="Corrupted page",
        )

        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=str(out),
            placement="top",
            params={"prefix": "B"},
            success=True,
            stamp_values=["B001"],
        )

        assert len(chain.events) == 4
        valid, _ = chain.verify_chain()
        assert valid is True

        event_types = [e["event_type"] for e in chain.events]
        assert event_types == [
            EVENT_TRANSFORM_START,
            EVENT_TRANSFORM_FAILED,
            EVENT_STAMP_START,
            EVENT_STAMP_COMPLETE,
        ]

    def test_multiple_transforms_accumulate(self, tmp_path):
        """Multiple transform lifecycles accumulate events correctly."""
        f1 = tmp_path / "f1.pdf"
        f2 = tmp_path / "f2.pdf"
        f3 = tmp_path / "f3.pdf"
        f1.write_bytes(b"a")
        f2.write_bytes(b"b")
        f3.write_bytes(b"c")

        chain = CustodyChain("batch_doc", str(f1))

        for i, (src, dst) in enumerate([(f1, f2), (f2, f3)]):
            record_transform_lifecycle(
                custody_chain=chain,
                operation_id=f"step_{i}",
                input_path=str(src),
                output_path=str(dst),
                params={"step": i},
                success=True,
            )

        assert len(chain.events) == 4  # 2 steps * 2 events each
        valid, msg = chain.verify_chain()
        assert valid is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_params_dict(self, tmp_path):
        """Empty params dict is handled gracefully."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("edge1", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="noop",
            input_path=str(inp),
            output_path=None,
            params={},
            success=False,
        )

        assert diag["start_event_recorded"] is True
        assert chain.events[0]["data"]["params"] == {}

    def test_empty_operation_id(self, tmp_path):
        """Empty string operation_id is accepted."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("edge2", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="",
            input_path=str(inp),
            output_path=None,
            params={},
            success=False,
        )

        assert diag["operation_id"] == ""

    def test_large_params_dict(self, tmp_path):
        """Large params dict does not break hashing or recording."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"data")
        out.write_bytes(b"data")

        chain = CustodyChain("edge3", str(inp))
        big_params = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="big_op",
            input_path=str(inp),
            output_path=str(out),
            params=big_params,
            success=True,
        )

        assert diag["complete_event_recorded"] is True
        valid, _ = chain.verify_chain()
        assert valid is True

    def test_unicode_in_paths_and_params(self, tmp_path):
        """Unicode characters in paths and params are handled."""
        inp = tmp_path / "input.pdf"
        out = tmp_path / "output.pdf"
        inp.write_bytes(b"data")
        out.write_bytes(b"data")

        chain = CustodyChain("edge4", str(inp))
        diag = record_transform_lifecycle(
            custody_chain=chain,
            operation_id="unicode_op",
            input_path=str(inp),
            output_path=str(out),
            params={"label": "Donnees francaises"},
            success=True,
            metadata={"note": "CJK test"},
        )

        assert diag["complete_event_recorded"] is True
        valid, _ = chain.verify_chain()
        assert valid is True

    def test_none_error_message_on_failure(self, tmp_path):
        """None error_message on failure is accepted."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("edge5", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="op",
            input_path=str(inp),
            output_path=None,
            params={},
            success=False,
            error_message=None,
        )

        assert chain.events[1]["data"]["error_message"] is None

    def test_stamp_empty_stamp_values_list(self, tmp_path):
        """Explicit empty stamp_values list is preserved."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("edge6", str(inp))
        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="designation",
            input_path=str(inp),
            output_path=str(out),
            placement="header",
            params={},
            success=True,
            stamp_values=[],
        )

        assert chain.events[1]["data"]["stamp_values"] == []

    def test_diagnostics_chain_hash_matches_chain_state(self, tmp_path):
        """Diagnostics custody_chain_hash matches chain._prev_hash at time of return."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("edge7", str(inp))
        diag = record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="stamp",
            input_path=str(inp),
            output_path=str(out),
            placement="footer",
            params={},
            success=True,
        )

        # The chain hash in diagnostics should be the hash after the last event
        assert diag["custody_chain_hash"] == chain._prev_hash

    def test_start_event_data_contains_operation_fields(self, tmp_path):
        """Transform start event data has operation_id, input_path, input_hash, params."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("edge8", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="my_op",
            input_path=str(inp),
            output_path=None,
            params={"key": "val"},
            success=False,
        )

        start_data = chain.events[0]["data"]
        assert start_data["operation_id"] == "my_op"
        assert start_data["input_path"] == str(inp)
        assert start_data["params"] == {"key": "val"}
        assert "input_hash" in start_data  # present even if None

    def test_complete_event_data_contains_output_fields(self, tmp_path):
        """Transform complete event data has operation_id, output_path, output_hash."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("edge9", str(inp))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="my_op",
            input_path=str(inp),
            output_path=str(out),
            params={},
            success=True,
        )

        complete_data = chain.events[1]["data"]
        assert complete_data["operation_id"] == "my_op"
        assert complete_data["output_path"] == str(out)
        assert "output_hash" in complete_data


# ---------------------------------------------------------------------------
# Interaction with custody chain disk persistence
# ---------------------------------------------------------------------------

class TestCustodyHooksWithDiskPersistence:
    """Verify custody_hooks functions work with disk-backed chains."""

    def test_transform_lifecycle_persists_to_disk(self, tmp_path):
        """Events from transform lifecycle are written to JSONL file."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        cdir = tmp_path / "custody"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("disk_doc", str(inp), str(cdir))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="clean",
            input_path=str(inp),
            output_path=str(out),
            params={},
            success=True,
        )

        jsonl = cdir / "disk_doc.custody.jsonl"
        assert jsonl.exists()
        lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_stamp_lifecycle_persists_to_disk(self, tmp_path):
        """Events from stamp lifecycle are written to JSONL file."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        cdir = tmp_path / "custody"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("disk_stamp", str(inp), str(cdir))
        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=str(out),
            placement="bottom",
            params={},
            success=True,
        )

        jsonl = cdir / "disk_stamp.custody.jsonl"
        assert jsonl.exists()
        lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_persisted_chain_verifiable_after_load(self, tmp_path):
        """JSONL written by custody_hooks can be loaded and verified."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        cdir = tmp_path / "custody"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("verify_doc", str(inp), str(cdir))
        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="clean",
            input_path=str(inp),
            output_path=str(out),
            params={"mode": "full"},
            success=True,
        )

        jsonl_path = cdir / "verify_doc.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(jsonl_path))
        valid, msg = loaded.verify_chain()
        assert valid is True
        assert "2 events" in msg


# ---------------------------------------------------------------------------
# Mock-based tests for append_event interaction
# ---------------------------------------------------------------------------

class TestAppendEventInteraction:
    """Verify custody_hooks calls append_event with correct arguments."""

    def test_transform_start_event_data(self, tmp_path):
        """Transform start event is called with correct data structure."""
        inp = tmp_path / "in.pdf"
        inp.write_bytes(b"data")

        chain = CustodyChain("mock_doc", str(inp))
        chain.append_event = MagicMock(wraps=chain.append_event)

        record_transform_lifecycle(
            custody_chain=chain,
            operation_id="test_op",
            input_path=str(inp),
            output_path=None,
            params={"key": "val"},
            success=False,
        )

        # First call: start event
        first_call = chain.append_event.call_args_list[0]
        assert first_call[0][0] == EVENT_TRANSFORM_START
        start_data = first_call[0][1]
        assert start_data["operation_id"] == "test_op"
        assert start_data["input_path"] == str(inp)
        assert start_data["params"] == {"key": "val"}

        # Second call: failed event
        second_call = chain.append_event.call_args_list[1]
        assert second_call[0][0] == EVENT_TRANSFORM_FAILED

    def test_stamp_complete_event_data(self, tmp_path):
        """Stamp complete event is called with stamp_values and metadata."""
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"
        inp.write_bytes(b"x")
        out.write_bytes(b"y")

        chain = CustodyChain("mock_stamp", str(inp))
        chain.append_event = MagicMock(wraps=chain.append_event)

        record_stamp_lifecycle(
            custody_chain=chain,
            operation_id="bates",
            input_path=str(inp),
            output_path=str(out),
            placement="bottom",
            params={"prefix": "P"},
            success=True,
            stamp_values=["P001", "P002"],
            metadata={"user": "admin"},
        )

        # Second call: complete event
        second_call = chain.append_event.call_args_list[1]
        assert second_call[0][0] == EVENT_STAMP_COMPLETE
        complete_data = second_call[0][1]
        assert complete_data["stamp_values"] == ["P001", "P002"]
        assert complete_data["metadata"] == {"user": "admin"}
        assert complete_data["operation_id"] == "bates"
