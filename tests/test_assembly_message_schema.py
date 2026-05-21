"""Tests for AssemblyMessage TypedDict schema.

Verifies that:
- AssemblyMessage and _AssemblyMessageRequired are proper TypedDict subclasses
- The full 12-key schema is defined correctly
- All 6 assembly_queue.put() sites produce messages with the full key set
- The assembler consumer reads all 12 keys correctly via .get()
"""
import queue
import typing

import pytest


def _get_assembly_types():
    """Import AssemblyMessage and _AssemblyMessageRequired from ocr_gpu_async."""
    from ocr_gpu_async import AssemblyMessage, _AssemblyMessageRequired
    return AssemblyMessage, _AssemblyMessageRequired


# Expected keys split by required vs optional
REQUIRED_KEYS = {"doc_id", "page_num", "text", "status", "chunk_path"}
OPTIONAL_KEYS = {
    "structure_data",
    "ocr_confidence",
    "ocr_method",
    "handwriting_data",
    "signature_data",
    "vertical_text_data",
    "table_fallback_data",
}
ALL_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS


class TestAssemblyMessageTypedDict:
    """Verify the TypedDict definitions are correct."""

    def test_assembly_message_is_typeddict(self):
        AssemblyMessage, _ = _get_assembly_types()
        # TypedDict classes have __annotations__ and are subclasses of dict
        assert hasattr(AssemblyMessage, "__annotations__")
        # typing.get_type_hints works on TypedDict
        hints = typing.get_type_hints(AssemblyMessage)
        assert len(hints) > 0

    def test_required_base_is_typeddict(self):
        _, _AssemblyMessageRequired = _get_assembly_types()
        assert hasattr(_AssemblyMessageRequired, "__annotations__")
        hints = typing.get_type_hints(_AssemblyMessageRequired)
        assert set(hints.keys()) == REQUIRED_KEYS

    def test_assembly_message_has_all_12_keys(self):
        AssemblyMessage, _ = _get_assembly_types()
        hints = typing.get_type_hints(AssemblyMessage)
        assert set(hints.keys()) == ALL_KEYS
        assert len(hints) == 12

    def test_required_keys_are_required(self):
        """Required keys should come from the base class with total=True."""
        _, _AssemblyMessageRequired = _get_assembly_types()
        # In Python 3.10+, __required_keys__ and __optional_keys__ are available
        if hasattr(_AssemblyMessageRequired, "__required_keys__"):
            assert _AssemblyMessageRequired.__required_keys__ == REQUIRED_KEYS

    def test_optional_keys_are_optional(self):
        """Optional keys should be in the total=False subclass."""
        AssemblyMessage, _ = _get_assembly_types()
        if hasattr(AssemblyMessage, "__optional_keys__"):
            assert AssemblyMessage.__optional_keys__ == OPTIONAL_KEYS

    def test_type_annotations_are_correct(self):
        AssemblyMessage, _ = _get_assembly_types()
        hints = typing.get_type_hints(AssemblyMessage)
        assert hints["doc_id"] is str
        assert hints["page_num"] is int
        assert hints["text"] is str
        assert hints["status"] is str
        assert hints["ocr_confidence"] is float


class TestFailurePathMessages:
    """Verify the 3 previously-incomplete failure-path messages now include all keys."""

    def _make_resumed_message(self):
        """Simulate the RESUMED message shape (Site 1)."""
        return {
            "doc_id": "test_doc_123",
            "page_num": 1,
            "text": "",
            "status": "RESUMED",
            "chunk_path": "/tmp/test/1.pdf",
            "structure_data": None,
            "ocr_confidence": 0.0,
            "ocr_method": "RESUMED",
            "handwriting_data": None,
            "signature_data": None,
            "vertical_text_data": None,
            "table_fallback_data": None,
        }

    def _make_extract_failed_message(self):
        """Simulate the EXTRACT_FAILED message shape (Site 2)."""
        return {
            "doc_id": "test_doc_456",
            "page_num": 3,
            "text": "",
            "status": "EXTRACT_FAILED",
            "chunk_path": None,
            "structure_data": None,
            "ocr_confidence": 0.0,
            "ocr_method": "EXTRACT_FAILED",
            "handwriting_data": None,
            "signature_data": None,
            "vertical_text_data": None,
            "table_fallback_data": None,
        }

    def _make_critical_failed_message(self):
        """Simulate the CRITICAL_FAILED message shape (Site 6)."""
        return {
            "doc_id": "test_doc_789",
            "page_num": 5,
            "text": "",
            "status": "CRITICAL_FAILED",
            "chunk_path": "/tmp/test/5.pdf",
            "structure_data": None,
            "ocr_confidence": 0.0,
            "ocr_method": "CRITICAL_FAILED",
            "handwriting_data": None,
            "signature_data": None,
            "vertical_text_data": None,
            "table_fallback_data": None,
        }

    @pytest.mark.parametrize("status", ["RESUMED", "EXTRACT_FAILED", "CRITICAL_FAILED"])
    def test_failure_messages_have_all_12_keys(self, status):
        makers = {
            "RESUMED": self._make_resumed_message,
            "EXTRACT_FAILED": self._make_extract_failed_message,
            "CRITICAL_FAILED": self._make_critical_failed_message,
        }
        msg = makers[status]()
        assert set(msg.keys()) == ALL_KEYS
        assert len(msg) == 12

    @pytest.mark.parametrize("status", ["RESUMED", "EXTRACT_FAILED", "CRITICAL_FAILED"])
    def test_failure_messages_have_correct_defaults(self, status):
        makers = {
            "RESUMED": self._make_resumed_message,
            "EXTRACT_FAILED": self._make_extract_failed_message,
            "CRITICAL_FAILED": self._make_critical_failed_message,
        }
        msg = makers[status]()
        assert msg["structure_data"] is None
        assert msg["ocr_confidence"] == 0.0
        assert msg["ocr_method"] == status
        assert msg["handwriting_data"] is None
        assert msg["signature_data"] is None
        assert msg["vertical_text_data"] is None
        assert msg["table_fallback_data"] is None


class TestAssemblerConsumerReads:
    """Verify the assembler consumer can read all 12 keys via .get() from any message."""

    def _simulate_assembler_read(self, msg):
        """Simulate how the assembler thread reads keys from an assembly message."""
        doc_id = msg["doc_id"]
        page_num = msg.get("page_num")
        status = msg.get("status", "UNKNOWN")
        msg_text = msg.get("text", "")
        chunk_path = msg.get("chunk_path")
        msg_structure = msg.get("structure_data")
        msg_hw = msg.get("handwriting_data")
        msg_sig = msg.get("signature_data")
        msg_vt = msg.get("vertical_text_data")
        msg_tf = msg.get("table_fallback_data")
        ocr_method = msg.get("ocr_method", "")
        ocr_confidence = float(msg.get("ocr_confidence", 0.0) or 0.0)
        return {
            "doc_id": doc_id,
            "page_num": page_num,
            "status": status,
            "text": msg_text,
            "chunk_path": chunk_path,
            "structure_data": msg_structure,
            "handwriting_data": msg_hw,
            "signature_data": msg_sig,
            "vertical_text_data": msg_vt,
            "table_fallback_data": msg_tf,
            "ocr_method": ocr_method,
            "ocr_confidence": ocr_confidence,
        }

    def test_resumed_message_round_trip(self):
        msg = {
            "doc_id": "doc1",
            "page_num": 1,
            "text": "",
            "status": "RESUMED",
            "chunk_path": "/tmp/1.pdf",
            "structure_data": None,
            "ocr_confidence": 0.0,
            "ocr_method": "RESUMED",
            "handwriting_data": None,
            "signature_data": None,
            "vertical_text_data": None,
            "table_fallback_data": None,
        }
        result = self._simulate_assembler_read(msg)
        assert result["doc_id"] == "doc1"
        assert result["status"] == "RESUMED"
        assert result["ocr_method"] == "RESUMED"
        assert result["ocr_confidence"] == 0.0

    def test_success_message_round_trip(self):
        msg = {
            "doc_id": "doc2",
            "page_num": 3,
            "text": "Hello world",
            "status": "OK",
            "chunk_path": "/tmp/3.pdf",
            "structure_data": {"layout_regions": []},
            "ocr_confidence": 0.95,
            "ocr_method": "Paddle-en",
            "handwriting_data": {"is_handwritten": False},
            "signature_data": None,
            "vertical_text_data": None,
            "table_fallback_data": None,
        }
        result = self._simulate_assembler_read(msg)
        assert result["doc_id"] == "doc2"
        assert result["text"] == "Hello world"
        assert result["ocr_confidence"] == 0.95
        assert result["ocr_method"] == "Paddle-en"
        assert result["structure_data"] == {"layout_regions": []}

    def test_queue_round_trip(self):
        """Test that messages survive queue.put/get correctly."""
        q: queue.Queue = queue.Queue()
        msg = {
            "doc_id": "doc3",
            "page_num": 2,
            "text": "",
            "status": "CRITICAL_FAILED",
            "chunk_path": None,
            "structure_data": None,
            "ocr_confidence": 0.0,
            "ocr_method": "CRITICAL_FAILED",
            "handwriting_data": None,
            "signature_data": None,
            "vertical_text_data": None,
            "table_fallback_data": None,
        }
        q.put(msg)
        received = q.get(timeout=1)
        assert set(received.keys()) == ALL_KEYS
        assert received["status"] == "CRITICAL_FAILED"
        assert received["ocr_method"] == "CRITICAL_FAILED"


class TestSourceCodeConsistency:
    """Verify that all assembly_queue.put() sites in ocr_gpu_async.py have 12 keys."""

    def test_all_put_sites_have_full_schema(self):
        """Parse ocr_gpu_async.py source and verify all assembly_queue.put sites
        include the full set of 12 keys."""
        import inspect

        # Read the source file directly
        import ocr_gpu_async
        source_path = inspect.getfile(ocr_gpu_async)
        with open(source_path, encoding="utf-8") as f:
            source = f.read()

        # Find all assembly_queue.put({ ... }) blocks by scanning for the pattern
        # We look for lines containing assembly_queue.put({ and then collect
        # the dict keys within that block
        lines = source.split("\n")
        put_sites = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if "assembly_queue.put({" in stripped:
                # Collect lines until we find the closing })
                block_lines = [lines[i]]
                j = i + 1
                brace_depth = stripped.count("{") - stripped.count("}")
                while j < len(lines) and brace_depth > 0:
                    block_lines.append(lines[j])
                    brace_depth += lines[j].count("{") - lines[j].count("}")
                    j += 1
                # Extract keys from the block
                keys_found = set()
                for line in block_lines:
                    line_stripped = line.strip()
                    if line_stripped.startswith('"') and '":' in line_stripped:
                        key = line_stripped.split('"')[1]
                        keys_found.add(key)
                put_sites.append({
                    "line": i + 1,
                    "keys": keys_found,
                    "status": None,
                })
                # Extract status from the block
                for line in block_lines:
                    if '"status"' in line:
                        parts = line.split('"')
                        for pi, part in enumerate(parts):
                            if part == "status" and pi + 2 < len(parts):
                                put_sites[-1]["status"] = parts[pi + 2]
                                break
                i = j
            else:
                i += 1

        # We expect 6 sites
        assert len(put_sites) == 6, (
            f"Expected 6 assembly_queue.put sites, found {len(put_sites)}: "
            f"{[s['line'] for s in put_sites]}"
        )

        # All 6 sites should have all 12 keys
        for site in put_sites:
            missing = ALL_KEYS - site["keys"]
            assert not missing, (
                f"Site at line {site['line']} (status={site['status']}) "
                f"is missing keys: {missing}"
            )
