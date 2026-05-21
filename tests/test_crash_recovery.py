"""
Crash recovery automation tests for the page-level resume pipeline.

Validates that the async OCR pipeline correctly handles mid-processing
crashes, worker failures, coordinator restarts, temp dir corruption,
and concurrent document processing where one document crashes.

Scenarios:
  1. Mid-page crash simulation (partial temp dir with existing page PDFs)
  2. Worker OOM/error simulation (MemoryError during processing)
  3. Coordinator restart simulation (doc_registry partial state recovery)
  4. Temp dir corruption (zero-byte and malformed page PDFs)
  5. Concurrent crash/resume (two documents, one crashes)
  6. Resume gap logic (non-contiguous existing pages)
  7. Assembler corrupt chunk handling (fitz.open fails on bad PDF)

Run with: python -m pytest tests/test_crash_recovery.py -v
"""

import json
import os
import queue
import threading
import time

import pytest

_can_import_pipeline = True
try:
    import ocr_gpu_async as pipe
except ImportError:
    _can_import_pipeline = False

_skip_no_pipeline = pytest.mark.skipif(
    not _can_import_pipeline,
    reason="ocr_gpu_async requires optional OCR runtime dependencies",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc_state(pipe_mod, source_file, doc_id, source_type="pdf"):
    """Create a DocumentState and register it in the pipeline registry."""
    doc = pipe_mod.DocumentState(str(source_file), doc_id, source_type)
    pipe_mod.doc_registry[doc_id] = doc
    return doc


def _create_chunk_pdf(pipe_mod, path, text="Page content"):
    """Create a minimal valid PDF file at path using fitz."""
    with pipe_mod.fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), text)
        pdf.save(str(path))


def _monkeypatch_feature_flags_off(monkeypatch, pipe_mod):
    """Disable all optional feature flags so assembler runs lean."""
    monkeypatch.setattr(pipe_mod, "ENABLE_DOCUMENT_INTELLIGENCE", False)
    monkeypatch.setattr(pipe_mod, "ENABLE_VALIDATION", False)
    monkeypatch.setattr(pipe_mod, "ENABLE_NER", False)
    monkeypatch.setattr(pipe_mod, "ENABLE_HANDWRITING", False)
    monkeypatch.setattr(pipe_mod, "ENABLE_CLASSIFICATION", False)
    monkeypatch.setattr(pipe_mod, "ENABLE_EXTRACTION", False)
    monkeypatch.setattr(pipe_mod, "ENABLE_SIGNATURE_VERIFICATION", False)
    monkeypatch.setattr(pipe_mod, "ENABLE_VERTICAL_TEXT", False)
    monkeypatch.setattr(pipe_mod, "ENABLE_TABLE_FALLBACK", False)
    monkeypatch.setattr(pipe_mod, "_VALIDATION_AVAILABLE", False)
    monkeypatch.setattr(pipe_mod, "_NER_AVAILABLE", False)
    monkeypatch.setattr(pipe_mod, "_HANDWRITING_AVAILABLE", False)
    monkeypatch.setattr(pipe_mod, "_CLASSIFICATION_AVAILABLE", False)
    monkeypatch.setattr(pipe_mod, "_EXTRACTION_AVAILABLE", False)
    monkeypatch.setattr(pipe_mod, "_SIGNATURE_VERIFICATION_AVAILABLE", False)
    monkeypatch.setattr(pipe_mod, "_VERTICAL_TEXT_AVAILABLE", False)
    monkeypatch.setattr(pipe_mod, "_TABLE_FALLBACK_AVAILABLE", False)


def _monkeypatch_pipeline_dirs(monkeypatch, pipe_mod, tmp_path):
    """Set up source/output/temp dirs and global state for a test."""
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipe_mod, "SOURCE_FOLDER", str(source_dir))
    monkeypatch.setattr(pipe_mod, "OUTPUT_FOLDER", str(output_dir))
    monkeypatch.setattr(pipe_mod, "TEMP_FOLDER", str(temp_dir))
    monkeypatch.setattr(pipe_mod, "stop_event", threading.Event())
    monkeypatch.setattr(pipe_mod, "doc_registry", {})
    monkeypatch.setattr(pipe_mod, "assembly_queue", queue.Queue())
    monkeypatch.setattr(pipe_mod, "image_queue", queue.Queue())
    monkeypatch.setattr(pipe_mod, "compression_queue", queue.Queue())

    return source_dir, output_dir, temp_dir


def _run_assembler_until_done(pipe_mod, doc_id, timeout=10):
    """Run assembler thread until doc_id is finalized or timeout."""
    thread = threading.Thread(target=pipe_mod.assembler_thread, daemon=True)
    thread.start()

    deadline = time.time() + timeout
    while time.time() < deadline and doc_id in pipe_mod.doc_registry:
        time.sleep(0.05)

    pipe_mod.stop_event.set()
    thread.join(timeout=5)


def _run_assembler_until_all_done(pipe_mod, doc_ids, timeout=10):
    """Run assembler thread until all doc_ids are finalized or timeout."""
    thread = threading.Thread(target=pipe_mod.assembler_thread, daemon=True)
    thread.start()

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(d_id in pipe_mod.doc_registry for d_id in doc_ids):
            break
        time.sleep(0.05)

    pipe_mod.stop_event.set()
    thread.join(timeout=5)


# ===========================================================================
# Scenario 1: Mid-page crash simulation
# ===========================================================================

@_skip_no_pipeline
class TestMidPageCrashResume:
    """Simulate a crash after processing pages 1-5 of a 10-page document.

    When the pipeline restarts, the scheduler should:
    - Detect existing per-page PDFs (1.pdf through 5.pdf) in temp dir
    - Send RESUMED messages for pages 1-5 to the assembler
    - Queue chunk tasks only for missing pages 6-10
    """

    def test_scheduler_detects_existing_pages_and_queues_gaps(
        self, tmp_path, monkeypatch,
    ):
        source_dir = tmp_path / "source"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        source_file = source_dir / "report.pdf"
        source_file.write_bytes(b"%PDF-1.4\nTest document\n%%EOF\n")

        doc_id = "doc-midcrash"
        doc_temp = temp_dir / doc_id
        doc_temp.mkdir(parents=True, exist_ok=True)

        # Pre-create pages 1-5 as if they were processed before crash
        for p in range(1, 6):
            _create_chunk_pdf(pipe, doc_temp / f"{p}.pdf", f"Page {p}")

        # Write matching manifest so resume is valid
        fp = pipe.compute_source_fingerprint(str(source_file))
        pipe._write_resume_manifest(str(doc_temp), str(source_file), fp)

        class DummyDocState:
            def __init__(self, path, did, source_type):
                self.path = path
                self.doc_id = did
                self.source_type = source_type
                self.total_pages = 0
                self.custody_chain = None
                self.temp_dir = os.path.join(str(temp_dir), did)
                os.makedirs(self.temp_dir, exist_ok=True)

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "chunk_queue", queue.Queue())
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "DocumentState", DummyDocState)
        monkeypatch.setattr(
            pipe, "build_resume_doc_id", lambda _p, _f: doc_id,
        )
        monkeypatch.setattr(
            pipe, "compute_source_fingerprint_fast",
            lambda _p: fp,
        )
        monkeypatch.setattr(pipe, "detect_language", lambda _p: "en")
        monkeypatch.setattr(
            pipe, "classify_source_file", lambda _p: ("pdf", None),
        )
        # 10-page document
        monkeypatch.setattr(pipe, "get_source_page_count", lambda _p, _t: 10)
        monkeypatch.setattr(
            pipe.os, "walk",
            lambda _p: [(str(source_dir), [], ["report.pdf"])],
        )
        monkeypatch.setattr(pipe, "log_failure", lambda *_a, **_k: None)

        pipe.scheduler_thread()

        # Collect all RESUMED messages from assembly_queue
        resumed_pages = []
        while not pipe.assembly_queue.empty():
            msg = pipe.assembly_queue.get_nowait()
            if msg["status"] == "RESUMED":
                resumed_pages.append(msg["page_num"])

        # Pages 1-5 should be RESUMED
        assert sorted(resumed_pages) == [1, 2, 3, 4, 5]

        # Pages 6-10 should be queued for extraction
        queued_pages = set()
        while not pipe.chunk_queue.empty():
            _, _, start, end, _, _ = pipe.chunk_queue.get_nowait()
            for p in range(start, end + 1):
                queued_pages.add(p)

        assert queued_pages == {6, 7, 8, 9, 10}

    def test_assembler_merges_resumed_and_new_pages(
        self, tmp_path, monkeypatch,
    ):
        """Assembler correctly merges RESUMED pages (from temp) with newly
        processed pages into a final PDF."""
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)

        failures = []
        monkeypatch.setattr(
            pipe, "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        source_file = source_dir / "merged.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-merge-test"
        doc = _make_doc_state(pipe, source_file, doc_id)
        doc.total_pages = 4

        # Create chunk PDFs for all 4 pages in temp dir
        os.makedirs(doc.temp_dir, exist_ok=True)
        for p in range(1, 5):
            _create_chunk_pdf(pipe, os.path.join(doc.temp_dir, f"{p}.pdf"), f"Page {p}")

        # Pages 1-2 are RESUMED, pages 3-4 are freshly processed
        for p in [1, 2]:
            pipe.assembly_queue.put({
                "doc_id": doc_id,
                "page_num": p,
                "text": "",
                "status": "RESUMED",
                "chunk_path": os.path.join(doc.temp_dir, f"{p}.pdf"),
            })
        for p in [3, 4]:
            pipe.assembly_queue.put({
                "doc_id": doc_id,
                "page_num": p,
                "text": f"Page {p} text",
                "status": "Paddle",
                "chunk_path": os.path.join(doc.temp_dir, f"{p}.pdf"),
            })

        _run_assembler_until_done(pipe, doc_id)

        # Document should be finalized and removed from registry
        assert doc_id not in pipe.doc_registry

        # Output PDF should exist with all 4 pages
        assert os.path.exists(doc.output_pdf)
        with pipe.fitz.open(doc.output_pdf) as final_pdf:
            assert final_pdf.page_count == 4

        # No failures should have been logged for this merge
        merge_failures = [
            f for f in failures if "doc-merge-test" in str(f)
        ]
        assert len(merge_failures) == 0


# ===========================================================================
# Scenario 2: Worker OOM/error simulation
# ===========================================================================

@_skip_no_pipeline
class TestWorkerErrorSimulation:
    """Verify that when a worker encounters a critical error (e.g. MemoryError),
    the page is marked as CRITICAL_FAILED, not silently lost, and other pages
    continue processing normally."""

    def test_critical_failed_page_does_not_block_document(
        self, tmp_path, monkeypatch,
    ):
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)

        failures = []
        monkeypatch.setattr(
            pipe, "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        source_file = source_dir / "oom.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-oom"
        doc = _make_doc_state(pipe, source_file, doc_id)
        doc.total_pages = 3

        # Create chunks for pages 1 and 3 (page 2 "crashed" -- no chunk file)
        os.makedirs(doc.temp_dir, exist_ok=True)
        _create_chunk_pdf(pipe, os.path.join(doc.temp_dir, "1.pdf"), "OK Page 1")
        _create_chunk_pdf(pipe, os.path.join(doc.temp_dir, "3.pdf"), "OK Page 3")

        # Page 1: OK
        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 1,
            "text": "Page 1 text",
            "status": "Paddle",
            "chunk_path": os.path.join(doc.temp_dir, "1.pdf"),
        })
        # Page 2: CRITICAL_FAILED (simulates MemoryError in worker)
        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 2,
            "text": "",
            "status": "CRITICAL_FAILED",
            "chunk_path": None,
        })
        # Page 3: OK
        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 3,
            "text": "Page 3 text",
            "status": "Paddle",
            "chunk_path": os.path.join(doc.temp_dir, "3.pdf"),
        })

        _run_assembler_until_done(pipe, doc_id)

        assert doc_id not in pipe.doc_registry

        # Output PDF should exist (with 2 real pages; missing page 2 logged)
        assert os.path.exists(doc.output_pdf)
        with pipe.fitz.open(doc.output_pdf) as final_pdf:
            # Page 2 has no chunk file so only pages 1 and 3 are inserted
            assert final_pdf.page_count == 2

    def test_extract_failed_page_continues_processing(
        self, tmp_path, monkeypatch,
    ):
        """EXTRACT_FAILED status from extractor should be terminal and allow
        the rest of the document to proceed."""
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)

        failures = []
        monkeypatch.setattr(
            pipe, "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        source_file = source_dir / "extract_fail.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-extfail"
        doc = _make_doc_state(pipe, source_file, doc_id)
        doc.total_pages = 2

        os.makedirs(doc.temp_dir, exist_ok=True)
        _create_chunk_pdf(pipe, os.path.join(doc.temp_dir, "2.pdf"), "Good Page 2")

        # Page 1: EXTRACT_FAILED
        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 1,
            "text": "",
            "status": "EXTRACT_FAILED",
            "chunk_path": None,
        })
        # Page 2: OK
        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 2,
            "text": "Page 2 content",
            "status": "Paddle",
            "chunk_path": os.path.join(doc.temp_dir, "2.pdf"),
        })

        _run_assembler_until_done(pipe, doc_id)

        assert doc_id not in pipe.doc_registry
        assert os.path.exists(doc.output_pdf)

        with pipe.fitz.open(doc.output_pdf) as final_pdf:
            # Only page 2 has a chunk, page 1 is missing (EXTRACT_FAILED)
            assert final_pdf.page_count == 1


# ===========================================================================
# Scenario 3: Coordinator restart simulation
# ===========================================================================

@_skip_no_pipeline
class TestCoordinatorRestartSimulation:
    """Simulate a coordinator restart by pre-populating doc_registry
    with partial state, then re-running the scheduler to verify it
    picks up processing from where it left off."""

    def test_scheduler_resumes_from_existing_temp_state(
        self, tmp_path, monkeypatch,
    ):
        """After restart, scheduler finds existing pages and correctly
        identifies only missing pages for re-processing."""
        source_dir = tmp_path / "source"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        source_file = source_dir / "restart.pdf"
        source_file.write_bytes(b"%PDF-1.4\nRestart test\n%%EOF\n")

        doc_id = "doc-restart"
        doc_temp = temp_dir / doc_id
        doc_temp.mkdir(parents=True, exist_ok=True)

        # Simulate 3 of 5 pages already processed
        for p in [1, 2, 3]:
            _create_chunk_pdf(pipe, doc_temp / f"{p}.pdf", f"Recovered page {p}")

        fp = pipe.compute_source_fingerprint(str(source_file))
        pipe._write_resume_manifest(str(doc_temp), str(source_file), fp)

        class DummyDocState:
            def __init__(self, path, did, source_type):
                self.path = path
                self.doc_id = did
                self.source_type = source_type
                self.total_pages = 0
                self.custody_chain = None
                self.temp_dir = os.path.join(str(temp_dir), did)
                os.makedirs(self.temp_dir, exist_ok=True)

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "chunk_queue", queue.Queue())
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "DocumentState", DummyDocState)
        monkeypatch.setattr(pipe, "build_resume_doc_id", lambda _p, _f: doc_id)
        monkeypatch.setattr(pipe, "compute_source_fingerprint_fast", lambda _p: fp)
        monkeypatch.setattr(pipe, "detect_language", lambda _p: "en")
        monkeypatch.setattr(
            pipe, "classify_source_file", lambda _p: ("pdf", None),
        )
        monkeypatch.setattr(pipe, "get_source_page_count", lambda _p, _t: 5)
        monkeypatch.setattr(
            pipe.os, "walk",
            lambda _p: [(str(source_dir), [], ["restart.pdf"])],
        )
        monkeypatch.setattr(pipe, "log_failure", lambda *_a, **_k: None)

        pipe.scheduler_thread()

        # Verify RESUMED messages for pages 1-3
        resumed_pages = []
        while not pipe.assembly_queue.empty():
            msg = pipe.assembly_queue.get_nowait()
            if msg["status"] == "RESUMED":
                resumed_pages.append(msg["page_num"])
        assert sorted(resumed_pages) == [1, 2, 3]

        # Verify extraction tasks queued for pages 4-5
        queued_pages = set()
        while not pipe.chunk_queue.empty():
            _, _, start, end, _, _ = pipe.chunk_queue.get_nowait()
            for p in range(start, end + 1):
                queued_pages.add(p)
        assert queued_pages == {4, 5}

    def test_full_resume_all_pages_exist(self, tmp_path, monkeypatch):
        """When all pages already exist in temp dir, scheduler sends all
        RESUMED messages and queues no extraction tasks."""
        source_dir = tmp_path / "source"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        source_file = source_dir / "complete.pdf"
        source_file.write_bytes(b"%PDF-1.4\nComplete doc\n%%EOF\n")

        doc_id = "doc-all-exist"
        doc_temp = temp_dir / doc_id
        doc_temp.mkdir(parents=True, exist_ok=True)

        total_pages = 3
        for p in range(1, total_pages + 1):
            _create_chunk_pdf(pipe, doc_temp / f"{p}.pdf", f"Page {p}")

        fp = pipe.compute_source_fingerprint(str(source_file))
        pipe._write_resume_manifest(str(doc_temp), str(source_file), fp)

        class DummyDocState:
            def __init__(self, path, did, source_type):
                self.path = path
                self.doc_id = did
                self.source_type = source_type
                self.total_pages = 0
                self.custody_chain = None
                self.temp_dir = os.path.join(str(temp_dir), did)
                os.makedirs(self.temp_dir, exist_ok=True)

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "chunk_queue", queue.Queue())
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "DocumentState", DummyDocState)
        monkeypatch.setattr(pipe, "build_resume_doc_id", lambda _p, _f: doc_id)
        monkeypatch.setattr(pipe, "compute_source_fingerprint_fast", lambda _p: fp)
        monkeypatch.setattr(pipe, "detect_language", lambda _p: "en")
        monkeypatch.setattr(
            pipe, "classify_source_file", lambda _p: ("pdf", None),
        )
        monkeypatch.setattr(
            pipe, "get_source_page_count", lambda _p, _t: total_pages,
        )
        monkeypatch.setattr(
            pipe.os, "walk",
            lambda _p: [(str(source_dir), [], ["complete.pdf"])],
        )
        monkeypatch.setattr(pipe, "log_failure", lambda *_a, **_k: None)

        pipe.scheduler_thread()

        # All pages RESUMED
        resumed_pages = []
        while not pipe.assembly_queue.empty():
            msg = pipe.assembly_queue.get_nowait()
            if msg["status"] == "RESUMED":
                resumed_pages.append(msg["page_num"])
        assert sorted(resumed_pages) == [1, 2, 3]

        # No extraction tasks queued
        assert pipe.chunk_queue.empty()


# ===========================================================================
# Scenario 4: Temp dir corruption
# ===========================================================================

@_skip_no_pipeline
class TestTempDirCorruption:
    """Verify the pipeline handles corrupt/zero-byte page PDFs in temp dir."""

    def test_assembler_handles_zero_byte_chunk(self, tmp_path, monkeypatch):
        """A zero-byte chunk file should be detected as corrupt by the
        assembler, logged as a failure, and the document still finalizes."""
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)

        failures = []
        monkeypatch.setattr(
            pipe, "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        source_file = source_dir / "corrupt.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-corrupt"
        doc = _make_doc_state(pipe, source_file, doc_id)
        doc.total_pages = 2

        os.makedirs(doc.temp_dir, exist_ok=True)
        # Page 1: zero-byte (corrupt) chunk
        corrupt_path = os.path.join(doc.temp_dir, "1.pdf")
        with open(corrupt_path, "wb") as f:
            f.write(b"")

        # Page 2: valid chunk
        _create_chunk_pdf(pipe, os.path.join(doc.temp_dir, "2.pdf"), "Good page")

        # Both pages are RESUMED
        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 1,
            "text": "",
            "status": "RESUMED",
            "chunk_path": corrupt_path,
        })
        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 2,
            "text": "Page 2",
            "status": "RESUMED",
            "chunk_path": os.path.join(doc.temp_dir, "2.pdf"),
        })

        _run_assembler_until_done(pipe, doc_id)

        assert doc_id not in pipe.doc_registry
        assert os.path.exists(doc.output_pdf)

        # Page 1 is corrupt, so only page 2 should be in the final PDF
        with pipe.fitz.open(doc.output_pdf) as final_pdf:
            assert final_pdf.page_count == 1

        # Corrupt chunk should generate a failure log entry
        corrupt_failures = [
            f for f in failures if f[1] == 1 and "Corrupt" in str(f[2])
        ]
        assert len(corrupt_failures) > 0

    def test_assembler_handles_invalid_pdf_content(self, tmp_path, monkeypatch):
        """A chunk file with garbage content (not a valid PDF) should be
        detected as corrupt and not crash the assembler."""
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)

        failures = []
        monkeypatch.setattr(
            pipe, "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        source_file = source_dir / "garbage.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-garbage"
        doc = _make_doc_state(pipe, source_file, doc_id)
        doc.total_pages = 2

        os.makedirs(doc.temp_dir, exist_ok=True)
        # Page 1: garbage content
        garbage_path = os.path.join(doc.temp_dir, "1.pdf")
        with open(garbage_path, "wb") as f:
            f.write(b"THIS IS NOT A PDF FILE AT ALL")

        # Page 2: valid chunk
        _create_chunk_pdf(pipe, os.path.join(doc.temp_dir, "2.pdf"), "Valid")

        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 1,
            "text": "",
            "status": "RESUMED",
            "chunk_path": garbage_path,
        })
        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 2,
            "text": "Valid text",
            "status": "Paddle",
            "chunk_path": os.path.join(doc.temp_dir, "2.pdf"),
        })

        _run_assembler_until_done(pipe, doc_id)

        assert doc_id not in pipe.doc_registry
        assert os.path.exists(doc.output_pdf)

        # Document should still finalize (with page 2)
        with pipe.fitz.open(doc.output_pdf) as final_pdf:
            assert final_pdf.page_count == 1

    def test_resume_invalidation_clears_corrupt_chunks(self, tmp_path):
        """When source fingerprint changes, even corrupt chunks are removed."""
        temp_dir = tmp_path / "resume_corrupt"
        temp_dir.mkdir()

        source_path = "/app/ocr_source/test.pdf"
        fp_old = {"content_sha256": "a" * 64, "size_bytes": 100, "mtime_ns": 1}
        fp_new = {"content_sha256": "b" * 64, "size_bytes": 200, "mtime_ns": 2}

        # Create a mix of valid and corrupt chunks
        (temp_dir / "1.pdf").write_bytes(b"")  # zero-byte
        (temp_dir / "2.pdf").write_bytes(b"garbage")  # not a PDF
        (temp_dir / "3.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")  # valid-ish

        pipe._write_resume_manifest(str(temp_dir), source_path, fp_old)

        result = pipe.prepare_resume_state(str(temp_dir), source_path, fp_new)

        assert result["status"] == "invalidated"
        assert result["removed_entries"] == 3
        # All chunks should be gone
        remaining = [
            f for f in os.listdir(str(temp_dir))
            if f != pipe.RESUME_MANIFEST_FILENAME
        ]
        assert len(remaining) == 0


# ===========================================================================
# Scenario 5: Concurrent crash/resume
# ===========================================================================

@_skip_no_pipeline
class TestConcurrentCrashResume:
    """When two documents are being processed and one crashes, the other
    should continue unaffected through the assembler."""

    def test_one_doc_crashes_other_completes(self, tmp_path, monkeypatch):
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)

        failures = []
        monkeypatch.setattr(
            pipe, "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        # Document A: 2 pages, processes normally
        source_a = source_dir / "doc_a.pdf"
        source_a.write_bytes(b"%PDF-1.4\n%%EOF\n")
        doc_a = _make_doc_state(pipe, source_a, "doc-a")
        doc_a.total_pages = 2
        os.makedirs(doc_a.temp_dir, exist_ok=True)
        for p in [1, 2]:
            _create_chunk_pdf(
                pipe, os.path.join(doc_a.temp_dir, f"{p}.pdf"), f"A page {p}",
            )

        # Document B: 2 pages, page 2 crashes (CRITICAL_FAILED)
        source_b = source_dir / "doc_b.pdf"
        source_b.write_bytes(b"%PDF-1.4\n%%EOF\n")
        doc_b = _make_doc_state(pipe, source_b, "doc-b")
        doc_b.total_pages = 2
        os.makedirs(doc_b.temp_dir, exist_ok=True)
        _create_chunk_pdf(
            pipe, os.path.join(doc_b.temp_dir, "1.pdf"), "B page 1",
        )

        # Interleave messages from both documents
        pipe.assembly_queue.put({
            "doc_id": "doc-a", "page_num": 1, "text": "A1",
            "status": "Paddle",
            "chunk_path": os.path.join(doc_a.temp_dir, "1.pdf"),
        })
        pipe.assembly_queue.put({
            "doc_id": "doc-b", "page_num": 1, "text": "B1",
            "status": "Paddle",
            "chunk_path": os.path.join(doc_b.temp_dir, "1.pdf"),
        })
        pipe.assembly_queue.put({
            "doc_id": "doc-a", "page_num": 2, "text": "A2",
            "status": "Paddle",
            "chunk_path": os.path.join(doc_a.temp_dir, "2.pdf"),
        })
        pipe.assembly_queue.put({
            "doc_id": "doc-b", "page_num": 2, "text": "",
            "status": "CRITICAL_FAILED", "chunk_path": None,
        })

        _run_assembler_until_all_done(pipe, ["doc-a", "doc-b"])

        # Both should be finalized
        assert "doc-a" not in pipe.doc_registry
        assert "doc-b" not in pipe.doc_registry

        # Doc A should have 2 pages
        assert os.path.exists(doc_a.output_pdf)
        with pipe.fitz.open(doc_a.output_pdf) as pdf_a:
            assert pdf_a.page_count == 2

        # Doc B should have 1 page (page 2 crashed)
        assert os.path.exists(doc_b.output_pdf)
        with pipe.fitz.open(doc_b.output_pdf) as pdf_b:
            assert pdf_b.page_count == 1


# ===========================================================================
# Scenario 6: Resume gap logic (non-contiguous existing pages)
# ===========================================================================

@_skip_no_pipeline
class TestResumeGapLogic:
    """Verify the scheduler correctly handles non-contiguous gaps in
    existing pages (e.g., pages 1, 3, 5 exist but 2, 4 do not)."""

    def test_non_contiguous_gaps_queued_correctly(self, tmp_path, monkeypatch):
        source_dir = tmp_path / "source"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        source_file = source_dir / "gaps.pdf"
        source_file.write_bytes(b"%PDF-1.4\nGap test\n%%EOF\n")

        doc_id = "doc-gaps"
        doc_temp = temp_dir / doc_id
        doc_temp.mkdir(parents=True, exist_ok=True)

        # Pages 1, 3, 5 exist (gaps at 2 and 4)
        for p in [1, 3, 5]:
            _create_chunk_pdf(pipe, doc_temp / f"{p}.pdf", f"Page {p}")

        fp = pipe.compute_source_fingerprint(str(source_file))
        pipe._write_resume_manifest(str(doc_temp), str(source_file), fp)

        class DummyDocState:
            def __init__(self, path, did, source_type):
                self.path = path
                self.doc_id = did
                self.source_type = source_type
                self.total_pages = 0
                self.custody_chain = None
                self.temp_dir = os.path.join(str(temp_dir), did)
                os.makedirs(self.temp_dir, exist_ok=True)

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "chunk_queue", queue.Queue())
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "DocumentState", DummyDocState)
        monkeypatch.setattr(pipe, "build_resume_doc_id", lambda _p, _f: doc_id)
        monkeypatch.setattr(pipe, "compute_source_fingerprint_fast", lambda _p: fp)
        monkeypatch.setattr(pipe, "detect_language", lambda _p: "en")
        monkeypatch.setattr(
            pipe, "classify_source_file", lambda _p: ("pdf", None),
        )
        monkeypatch.setattr(pipe, "get_source_page_count", lambda _p, _t: 5)
        monkeypatch.setattr(
            pipe.os, "walk",
            lambda _p: [(str(source_dir), [], ["gaps.pdf"])],
        )
        monkeypatch.setattr(pipe, "log_failure", lambda *_a, **_k: None)

        pipe.scheduler_thread()

        # Pages 1, 3, 5 should be RESUMED
        resumed_pages = []
        while not pipe.assembly_queue.empty():
            msg = pipe.assembly_queue.get_nowait()
            if msg["status"] == "RESUMED":
                resumed_pages.append(msg["page_num"])
        assert sorted(resumed_pages) == [1, 3, 5]

        # Pages 2 and 4 should be queued as separate gap chunks
        queued_pages = set()
        chunk_count = 0
        while not pipe.chunk_queue.empty():
            _, _, start, end, _, _ = pipe.chunk_queue.get_nowait()
            chunk_count += 1
            for p in range(start, end + 1):
                queued_pages.add(p)

        assert queued_pages == {2, 4}
        # With non-contiguous gaps, each gap page is flushed separately
        # because a RESUMED page breaks continuity
        assert chunk_count == 2

    def test_single_missing_page_in_middle(self, tmp_path, monkeypatch):
        """Single missing page in the middle of a document."""
        source_dir = tmp_path / "source"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        source_file = source_dir / "single_gap.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-single-gap"
        doc_temp = temp_dir / doc_id
        doc_temp.mkdir(parents=True, exist_ok=True)

        # Pages 1, 2, 4, 5 exist (only page 3 missing)
        for p in [1, 2, 4, 5]:
            _create_chunk_pdf(pipe, doc_temp / f"{p}.pdf", f"Page {p}")

        fp = pipe.compute_source_fingerprint(str(source_file))
        pipe._write_resume_manifest(str(doc_temp), str(source_file), fp)

        class DummyDocState:
            def __init__(self, path, did, source_type):
                self.path = path
                self.doc_id = did
                self.source_type = source_type
                self.total_pages = 0
                self.custody_chain = None
                self.temp_dir = os.path.join(str(temp_dir), did)
                os.makedirs(self.temp_dir, exist_ok=True)

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "chunk_queue", queue.Queue())
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "DocumentState", DummyDocState)
        monkeypatch.setattr(pipe, "build_resume_doc_id", lambda _p, _f: doc_id)
        monkeypatch.setattr(pipe, "compute_source_fingerprint_fast", lambda _p: fp)
        monkeypatch.setattr(pipe, "detect_language", lambda _p: "en")
        monkeypatch.setattr(
            pipe, "classify_source_file", lambda _p: ("pdf", None),
        )
        monkeypatch.setattr(pipe, "get_source_page_count", lambda _p, _t: 5)
        monkeypatch.setattr(
            pipe.os, "walk",
            lambda _p: [(str(source_dir), [], ["single_gap.pdf"])],
        )
        monkeypatch.setattr(pipe, "log_failure", lambda *_a, **_k: None)

        pipe.scheduler_thread()

        resumed_pages = []
        while not pipe.assembly_queue.empty():
            msg = pipe.assembly_queue.get_nowait()
            if msg["status"] == "RESUMED":
                resumed_pages.append(msg["page_num"])
        assert sorted(resumed_pages) == [1, 2, 4, 5]

        queued_pages = set()
        while not pipe.chunk_queue.empty():
            _, _, start, end, _, _ = pipe.chunk_queue.get_nowait()
            for p in range(start, end + 1):
                queued_pages.add(p)
        assert queued_pages == {3}


# ===========================================================================
# Scenario 7: Assembler missing chunk detection
# ===========================================================================

@_skip_no_pipeline
class TestAssemblerMissingChunkDetection:
    """Verify that the assembler correctly identifies and logs missing
    chunks in various scenarios."""

    def test_missing_chunk_for_non_terminal_page_logs_crash(
        self, tmp_path, monkeypatch,
    ):
        """When a page is terminal-acknowledged but its chunk file is missing
        (and status is not EXTRACT/CRITICAL_FAILED), log it."""
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)

        failures = []
        monkeypatch.setattr(
            pipe, "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        source_file = source_dir / "missing.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-missing-chunk"
        doc = _make_doc_state(pipe, source_file, doc_id)
        doc.total_pages = 1

        os.makedirs(doc.temp_dir, exist_ok=True)
        # Do NOT create the chunk file -- simulate a lost write

        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 1,
            "text": "Some text",
            "status": "RESUMED",
            "chunk_path": os.path.join(doc.temp_dir, "1.pdf"),
        })

        _run_assembler_until_done(pipe, doc_id)

        assert doc_id not in pipe.doc_registry

        # Verify failure was logged for the missing chunk
        missing_failures = [
            f for f in failures
            if f[1] == 1 and "Missing chunk" in str(f[2])
        ]
        assert len(missing_failures) > 0

    def test_placeholder_page_when_all_chunks_missing(
        self, tmp_path, monkeypatch,
    ):
        """When all chunk files are missing, the assembler creates a
        placeholder page instead of an empty PDF."""
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)

        failures = []
        monkeypatch.setattr(
            pipe, "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        source_file = source_dir / "all_missing.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-all-missing"
        doc = _make_doc_state(pipe, source_file, doc_id)
        doc.total_pages = 2

        os.makedirs(doc.temp_dir, exist_ok=True)
        # No chunk files created

        for p in [1, 2]:
            pipe.assembly_queue.put({
                "doc_id": doc_id,
                "page_num": p,
                "text": "",
                "status": "CRITICAL_FAILED",
                "chunk_path": None,
            })

        _run_assembler_until_done(pipe, doc_id)

        assert doc_id not in pipe.doc_registry
        assert os.path.exists(doc.output_pdf)

        # Should have a placeholder page since no real pages could be merged
        with pipe.fitz.open(doc.output_pdf) as final_pdf:
            assert final_pdf.page_count == 1
            placeholder_text = final_pdf[0].get_text("text")
            assert "No renderable" in placeholder_text or "failures" in placeholder_text.lower()


# ===========================================================================
# Scenario 8: Resume text recovery from chunk PDF
# ===========================================================================

@_skip_no_pipeline
class TestResumedTextRecovery:
    """Verify that RESUMED pages correctly recover embedded text from
    their chunk PDF files."""

    def test_resumed_page_recovers_text_from_chunk(
        self, tmp_path, monkeypatch,
    ):
        """When a RESUMED message has empty text, the assembler should
        extract text from the chunk PDF."""
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)
        monkeypatch.setattr(pipe, "log_failure", lambda *_a, **_k: None)

        source_file = source_dir / "recover.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-recover-text"
        doc = _make_doc_state(pipe, source_file, doc_id)
        doc.total_pages = 1

        os.makedirs(doc.temp_dir, exist_ok=True)
        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        _create_chunk_pdf(pipe, chunk_path, "Recovered OCR text content")

        pipe.assembly_queue.put({
            "doc_id": doc_id,
            "page_num": 1,
            "text": "",  # Empty -- assembler should recover from chunk
            "status": "RESUMED",
            "chunk_path": chunk_path,
        })

        _run_assembler_until_done(pipe, doc_id)

        assert doc_id not in pipe.doc_registry
        assert os.path.exists(doc.output_pdf)

        # Verify that the output text file contains recovered content
        txt_dir = doc.output_txt_dir
        txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
        assert len(txt_files) == 1
        txt_content = open(
            os.path.join(txt_dir, txt_files[0]), "r", encoding="utf-8",
        ).read()
        assert "Recovered OCR text content" in txt_content


# ===========================================================================
# Scenario 9: Duplicate terminal messages
# ===========================================================================

@_skip_no_pipeline
class TestDuplicateTerminalMessages:
    """Verify that duplicate terminal messages for the same page are
    handled gracefully without double-counting."""

    def test_duplicate_resumed_does_not_double_count(
        self, tmp_path, monkeypatch,
    ):
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch, pipe, tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)
        monkeypatch.setattr(pipe, "log_failure", lambda *_a, **_k: None)

        source_file = source_dir / "dup.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        doc_id = "doc-dup"
        doc = _make_doc_state(pipe, source_file, doc_id)
        doc.total_pages = 1

        os.makedirs(doc.temp_dir, exist_ok=True)
        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        _create_chunk_pdf(pipe, chunk_path, "Dup test")

        # Send duplicate RESUMED messages
        for _ in range(3):
            pipe.assembly_queue.put({
                "doc_id": doc_id,
                "page_num": 1,
                "text": "Dup text",
                "status": "RESUMED",
                "chunk_path": chunk_path,
            })

        _run_assembler_until_done(pipe, doc_id)

        assert doc_id not in pipe.doc_registry
        assert os.path.exists(doc.output_pdf)

        with pipe.fitz.open(doc.output_pdf) as final_pdf:
            # Only one page should be in the final PDF despite 3 messages
            assert final_pdf.page_count == 1


# ===========================================================================
# Scenario 10: Source fingerprint change invalidates resume
# ===========================================================================

@_skip_no_pipeline
class TestSourceChangeInvalidation:
    """Verify that when the source document changes between runs, all
    stale temp state is properly invalidated and re-processed."""

    def test_changed_source_invalidates_all_pages(self, tmp_path, monkeypatch):
        source_dir = tmp_path / "source"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        source_file = source_dir / "changed.pdf"
        source_file.write_bytes(b"%PDF-1.4\nOriginal content\n%%EOF\n")

        fp_original = pipe.compute_source_fingerprint(str(source_file))
        doc_id = "doc-changed"
        doc_temp = temp_dir / doc_id
        doc_temp.mkdir(parents=True, exist_ok=True)

        # Create pages from "previous run"
        for p in range(1, 4):
            _create_chunk_pdf(pipe, doc_temp / f"{p}.pdf", f"Old page {p}")
        pipe._write_resume_manifest(str(doc_temp), str(source_file), fp_original)

        # Now modify the source file
        source_file.write_bytes(b"%PDF-1.4\nModified content\n%%EOF\n")
        fp_modified = pipe.compute_source_fingerprint(str(source_file))

        # Verify fingerprints differ
        assert fp_original["content_sha256"] != fp_modified["content_sha256"]

        # prepare_resume_state should invalidate
        result = pipe.prepare_resume_state(
            str(doc_temp), str(source_file), fp_modified,
        )

        assert result["status"] == "invalidated"
        assert result["removed_entries"] == 3

        # No chunk files should remain
        remaining = [
            f for f in os.listdir(str(doc_temp))
            if f != pipe.RESUME_MANIFEST_FILENAME
        ]
        assert len(remaining) == 0

        # Manifest should be updated with new fingerprint
        manifest = json.loads(
            (doc_temp / pipe.RESUME_MANIFEST_FILENAME).read_text(encoding="utf-8"),
        )
        assert manifest["source_fingerprint"]["content_sha256"] == fp_modified["content_sha256"]
