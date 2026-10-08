"""Regression tests for async pipeline failure handling paths."""

import json
import os
import queue
import threading
import time
from pathlib import Path

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


@_skip_no_pipeline
class TestAsyncFailurePaths:
    def test_fatal_declaration_linearizes_after_locked_publish_commit(
        self,
        monkeypatch,
    ):
        """A concurrent fatal request cannot split publish/custody/handoff."""
        timeline = []

        class RecordingEvent(threading.Event):
            def __init__(self, label):
                super().__init__()
                self.label = label

            def set(self):
                timeline.append(self.label)
                super().set()

        handoff_lock = threading.RLock()
        abort = RecordingEvent("abort")
        fatal = RecordingEvent("fatal")
        attempting = threading.Event()
        monkeypatch.setattr(pipe, "_finalization_handoff_lock", handoff_lock)
        monkeypatch.setattr(pipe, "finalization_abort_event", abort)
        monkeypatch.setattr(pipe, "fatal_pipeline_error_event", fatal)

        def request_fatal():
            attempting.set()
            pipe._request_finalization_abort()

        with handoff_lock:
            thread = threading.Thread(target=request_fatal, daemon=True)
            thread.start()
            assert attempting.wait(1)
            time.sleep(0.02)
            assert thread.is_alive()
            assert not abort.is_set()
            assert not fatal.is_set()
            timeline.extend(("publish", "custody", "handoff"))

        thread.join(timeout=1)
        assert not thread.is_alive()
        assert timeline == ["publish", "custody", "handoff", "abort", "fatal"]

    @pytest.mark.parametrize("role", ["OCR worker", "assembler/finalizer"])
    def test_surviving_pipeline_thread_marks_pipeline_fatal(
        self,
        monkeypatch,
        role,
    ):
        release = threading.Event()
        late_worker = threading.Thread(
            target=release.wait,
            name="late-ocr-worker",
            daemon=True,
        )
        late_worker.start()
        fatal = threading.Event()
        monkeypatch.setattr(pipe, "fatal_pipeline_error_event", fatal)

        alive = pipe._join_threads_or_mark_fatal(
            [late_worker],
            role,
            timeout=0.01,
        )

        assert alive == ["late-ocr-worker"]
        assert fatal.is_set()
        release.set()
        late_worker.join(timeout=1)

    def test_shutdown_abort_prevents_late_finalization_publish(
        self,
        tmp_path,
        monkeypatch,
    ):
        output = tmp_path / "output"
        stage = output / ".finalize-doc"
        staged_pdf = stage / "EXPORT" / "PDF" / "doc.pdf"
        staged_pdf.parent.mkdir(parents=True)
        staged_pdf.write_bytes(b"%PDF-staged")
        canonical = output / "EXPORT" / "PDF" / "doc.pdf"
        abort = threading.Event()
        abort.set()
        monkeypatch.setattr(pipe, "OUTPUT_FOLDER", str(output))
        monkeypatch.setattr(pipe, "finalization_abort_event", abort)
        monkeypatch.setattr(pipe, "fatal_pipeline_error_event", threading.Event())

        with pytest.raises(RuntimeError, match="finalization abort barrier"):
            pipe._publish_finalization_stage(str(stage), str(canonical))

        assert not canonical.exists()
        assert staged_pdf.exists()

    def test_scheduler_contains_document_state_init_failure_and_continues(
        self,
        tmp_path,
        monkeypatch,
    ):
        source_dir = tmp_path / "source"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True)
        temp_dir.mkdir(parents=True)
        (source_dir / "bad.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        (source_dir / "good.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        failures = []

        class DummyDocState:
            def __init__(self, path, doc_id, source_type):
                if path.endswith("bad.pdf"):
                    raise ValueError("tampered custody chain")
                self.path = path
                self.doc_id = doc_id
                self.source_type = source_type
                self.total_pages = 0
                self.custody_chain = None
                self.completion_fence = pipe.DocumentCompletionFence(doc_id)
                self.temp_dir = os.path.join(str(temp_dir), doc_id)
                os.makedirs(self.temp_dir, exist_ok=True)

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(
            pipe,
            "assembly_producers_done_event",
            threading.Event(),
        )
        monkeypatch.setattr(pipe, "finalization_abort_event", threading.Event())
        monkeypatch.setattr(pipe, "fatal_pipeline_error_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "chunk_queue", queue.Queue())
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "DocumentState", DummyDocState)
        monkeypatch.setattr(
            pipe,
            "build_resume_doc_id",
            lambda path, _fingerprint: "doc-" + os.path.basename(path).split(".")[0],
        )
        monkeypatch.setattr(
            pipe,
            "compute_source_fingerprint",
            lambda _path: {"content_sha256": "a" * 64, "size_bytes": 1, "mtime_ns": 1},
        )
        monkeypatch.setattr(pipe, "detect_language", lambda _path: "en")
        monkeypatch.setattr(pipe, "classify_source_file", lambda _path: ("pdf", None))
        monkeypatch.setattr(pipe, "get_source_page_count", lambda _path, _type: 1)
        monkeypatch.setattr(
            pipe.os,
            "walk",
            lambda _path: [(str(source_dir), [], ["bad.pdf", "good.pdf"])],
        )
        monkeypatch.setattr(
            pipe,
            "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        pipe.scheduler_thread()

        assert "doc-bad" not in pipe.doc_registry
        assert "doc-good" in pipe.doc_registry
        assert pipe.fatal_pipeline_error_event.is_set()
        queued = pipe.chunk_queue.get_nowait()
        assert queued[0] == "doc-good"
        assert any("DOCUMENT_STATE_INIT_FAILED" in item[2] for item in failures)

    def test_scheduler_rejects_zero_page_source(self, tmp_path, monkeypatch):
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "zero.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")

        failures = []

        class DummyDocState:
            def __init__(self, path, doc_id, source_type):  # noqa: ARG002
                self.path = path
                self.doc_id = doc_id
                self.source_type = source_type
                self.total_pages = 0
                self.custody_chain = None

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        fatal = threading.Event()
        monkeypatch.setattr(pipe, "fatal_pipeline_error_event", fatal)
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "DocumentState", DummyDocState)
        monkeypatch.setattr(pipe, "get_path_based_doc_id", lambda _p: "doc-zero")
        monkeypatch.setattr(pipe, "detect_language", lambda _p: "en")
        monkeypatch.setattr(pipe, "classify_source_file", lambda _p: ("pdf", None))
        monkeypatch.setattr(pipe, "get_source_page_count", lambda _p, _t: 0)
        monkeypatch.setattr(
            pipe.os, "walk", lambda _p: [(str(source_dir), [], ["zero.pdf"])]
        )
        monkeypatch.setattr(
            pipe,
            "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        pipe.scheduler_thread()

        assert "doc-zero" not in pipe.doc_registry
        assert failures
        assert fatal.is_set()
        assert failures[0][1] == 0
        assert "SOURCE_PAGE_COUNT_ZERO" in failures[0][2]

    def test_assembler_logs_missing_chunk_after_terminal_status(
        self,
        tmp_path,
        monkeypatch,
    ):
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        source_file = source_dir / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        failures = []

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "OUTPUT_FOLDER", str(output_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(
            pipe,
            "assembly_producers_done_event",
            threading.Event(),
        )
        monkeypatch.setattr(pipe, "finalization_abort_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "image_queue", queue.Queue())
        monkeypatch.setattr(pipe, "compression_queue", queue.Queue())
        monkeypatch.setattr(pipe, "ENABLE_DOCUMENT_INTELLIGENCE", False)
        monkeypatch.setattr(pipe, "ENABLE_VALIDATION", False)
        monkeypatch.setattr(pipe, "ENABLE_NER", False)
        monkeypatch.setattr(pipe, "ENABLE_HANDWRITING", False)
        monkeypatch.setattr(pipe, "ENABLE_CLASSIFICATION", False)
        monkeypatch.setattr(pipe, "ENABLE_EXTRACTION", False)
        monkeypatch.setattr(pipe, "_VALIDATION_AVAILABLE", False)
        monkeypatch.setattr(pipe, "_NER_AVAILABLE", False)
        monkeypatch.setattr(pipe, "_HANDWRITING_AVAILABLE", False)
        monkeypatch.setattr(pipe, "_CLASSIFICATION_AVAILABLE", False)
        monkeypatch.setattr(pipe, "_EXTRACTION_AVAILABLE", False)
        monkeypatch.setattr(
            pipe,
            "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        doc = pipe.DocumentState(str(source_file), "doc-missing", "pdf")
        doc.total_pages = 1
        pipe.doc_registry["doc-missing"] = doc
        fatal = threading.Event()
        monkeypatch.setattr(pipe, "fatal_pipeline_error_event", fatal)
        # RESUMED without resume_metadata/digest fails admission: no terminal
        # page update and no finalization. Queue accounting still completes.
        pipe.assembly_queue.put(
            {
                "doc_id": "doc-missing",
                "page_num": 1,
                "status": "RESUMED",
                "text": "",
                "chunk_path": None,
                "checkpoint_sha256": None,
            },
        )

        pipe.assembly_producers_done_event.set()
        thread = threading.Thread(target=pipe.assembler_thread, daemon=True)
        thread.start()

        deadline = time.time() + 5
        while time.time() < deadline and not fatal.is_set():
            time.sleep(0.05)

        pipe.stop_event.set()
        thread.join(timeout=5)

        assert fatal.is_set()
        assert "doc-missing" in pipe.doc_registry
        assert any(
            "REUSE_ADMISSION_FAILED" in error for _path, _page, error in failures
        )
        assert pipe.compression_queue.empty()

    def test_scheduler_invalidates_stale_resume_cache_for_changed_source(
        self,
        tmp_path,
        monkeypatch,
    ):
        source_dir = tmp_path / "source"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        source_file = source_dir / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        stale_doc_id = "doc-stale"
        stale_doc_dir = temp_dir / stale_doc_id
        stale_doc_dir.mkdir(parents=True, exist_ok=True)
        (stale_doc_dir / "1.pdf").write_bytes(b"%PDF-1.4 stale chunk\n%%EOF\n")

        old_fingerprint = {
            "content_sha256": "a" * 64,
            "size_bytes": 12,
            "mtime_ns": 100,
        }
        (stale_doc_dir / pipe.RESUME_MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": pipe.RESUME_MANIFEST_SCHEMA_VERSION,
                    "source_path": str(source_file),
                    "source_fingerprint": old_fingerprint,
                },
            ),
            encoding="utf-8",
        )

        new_fingerprint = pipe.compute_source_fingerprint(str(source_file))

        class DummyDocState:
            def __init__(self, path, doc_id, source_type):
                self.path = path
                self.doc_id = doc_id
                self.source_type = source_type
                self.total_pages = 0
                self.custody_chain = None
                self.temp_dir = os.path.join(pipe.TEMP_FOLDER, doc_id)
                os.makedirs(self.temp_dir, exist_ok=True)

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "chunk_queue", queue.Queue())
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "DocumentState", DummyDocState)
        monkeypatch.setattr(pipe, "build_resume_doc_id", lambda _p, _f: stale_doc_id)
        monkeypatch.setattr(pipe, "detect_language", lambda _p: "en")
        monkeypatch.setattr(pipe, "classify_source_file", lambda _p: ("pdf", None))
        monkeypatch.setattr(pipe, "get_source_page_count", lambda _p, _t: 1)
        monkeypatch.setattr(
            pipe.os, "walk", lambda _p: [(str(source_dir), [], ["doc.pdf"])]
        )
        monkeypatch.setattr(pipe, "log_failure", lambda *_args, **_kwargs: None)

        pipe.scheduler_thread()

        assert pipe.assembly_queue.empty()
        queued = pipe.chunk_queue.get_nowait()
        assert queued[2] == 1
        assert queued[3] == 1
        assert not (stale_doc_dir / "1.pdf").exists()

        manifest = json.loads(
            (stale_doc_dir / pipe.RESUME_MANIFEST_FILENAME).read_text(encoding="utf-8"),
        )
        assert (
            manifest["source_fingerprint"]["content_sha256"]
            == new_fingerprint["content_sha256"]
        )

    def test_assembler_resumed_page_uses_exact_checkpoint_not_stale_sidecar(
        self,
        tmp_path,
        monkeypatch,
    ):
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        source_file = source_dir / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")

        validation_dir = output_dir / "EXPORT" / "VALIDATION"
        validation_dir.mkdir(parents=True, exist_ok=True)
        prior_val_path = validation_dir / "doc.validation.json"
        prior_val_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "pages": [
                        {
                            "page_num": 1,
                            "ocr_method": "PaddleOCR",
                            "ocr_language": "en",
                            "ocr_confidence": 0.997,
                            "text_length": 22,
                            "has_text": True,
                            "status": "ok",
                        },
                    ],
                },
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "OUTPUT_FOLDER", str(output_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(
            pipe,
            "assembly_producers_done_event",
            threading.Event(),
        )
        monkeypatch.setattr(pipe, "finalization_abort_event", threading.Event())
        monkeypatch.setattr(pipe, "fatal_pipeline_error_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "image_queue", queue.Queue())
        monkeypatch.setattr(pipe, "compression_queue", queue.Queue())
        monkeypatch.setattr(pipe, "ENABLE_DOCUMENT_INTELLIGENCE", False)
        monkeypatch.setattr(pipe, "ENABLE_VALIDATION", True)
        monkeypatch.setattr(pipe, "_VALIDATION_AVAILABLE", True)
        monkeypatch.setattr(pipe, "ENABLE_NER", False)
        monkeypatch.setattr(pipe, "ENABLE_HANDWRITING", False)
        monkeypatch.setattr(pipe, "ENABLE_CLASSIFICATION", False)
        monkeypatch.setattr(pipe, "ENABLE_EXTRACTION", False)
        monkeypatch.setattr(pipe, "_NER_AVAILABLE", False)
        monkeypatch.setattr(pipe, "_HANDWRITING_AVAILABLE", False)
        monkeypatch.setattr(pipe, "_CLASSIFICATION_AVAILABLE", False)
        monkeypatch.setattr(pipe, "_EXTRACTION_AVAILABLE", False)

        doc = pipe.DocumentState(str(source_file), "doc-resumed-quality", "pdf")
        doc.total_pages = 1
        os.makedirs(doc.temp_dir, exist_ok=True)
        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        with pipe.fitz.open() as chunk_pdf:
            page = chunk_pdf.new_page()
            page.insert_text((72, 72), "Recovered resume text")
            chunk_pdf.save(chunk_path)
        extracted = pipe._extract_text_from_chunk_pdf(chunk_path, strict=True)
        profile = pipe._checkpoint_processing_profile(
            actual_render_dpi=int(pipe.DPI),
            page_routing_target="default",
            execution_device="cpu",
            execution_engine="PaddleOCR",
            gpu_optimization_applied=False,
            preprocessing_applied=False,
        )
        raw = Path(chunk_path).read_bytes()
        with pipe.fitz.open(stream=raw, filetype="pdf") as checkpoint_doc:
            geometry = pipe.PageGeometry.from_pdf_page(checkpoint_doc[0])
        Path(chunk_path).write_bytes(
            pipe._embed_page_checkpoint_metadata(
                raw,
                document_id=doc.doc_id,
                page_num=1,
                status=pipe.PageStatus.PADDLE,
                ocr_method="PaddleOCR",
                ocr_language="fr",
                ocr_confidence=0.876,
                text_length=len(extracted),
                has_text=bool(extracted),
                processing_profile=profile,
                canonical_text=extracted,
                page_geometry=geometry,
            )
        )
        resume_metadata = pipe._load_page_checkpoint_metadata(
            chunk_path,
            document_id=doc.doc_id,
            page_num=1,
        )

        pipe.doc_registry[doc.doc_id] = doc
        import hashlib

        checkpoint_sha256 = hashlib.sha256(Path(chunk_path).read_bytes()).hexdigest()
        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "status": "RESUMED",
                "text": extracted,
                "chunk_path": chunk_path,
                "checkpoint_sha256": checkpoint_sha256,
                "resume_metadata": resume_metadata,
                "cache_hit": False,
            },
        )

        pipe.assembly_producers_done_event.set()
        thread = threading.Thread(target=pipe.assembler_thread, daemon=True)
        thread.start()

        deadline = time.time() + 5
        while time.time() < deadline and doc.doc_id in pipe.doc_registry:
            time.sleep(0.05)

        pipe.stop_event.set()
        thread.join(timeout=5)

        assert doc.doc_id not in pipe.doc_registry
        output_val = validation_dir / "doc.validation.json"
        assert output_val.exists()

        report = json.loads(output_val.read_text(encoding="utf-8"))
        # Exact checkpoint metadata overrides the stale sidecar trap.
        assert report["pages"][0]["ocr_language"] == "fr"
        assert report["pages"][0]["ocr_confidence"] == pytest.approx(0.876, rel=1e-3)
        assert report["quality"]["classification"] == "high_quality"
        assert report["quality"]["overall_confidence"] == pytest.approx(
            0.876,
            rel=1e-3,
        )
        assert report["pages"][0]["status"] == "ok"
        assert report["pages"][0]["ocr_method"] == "PaddleOCR"
        assert report["pages"][0]["ocr_confidence"] != pytest.approx(0.997)
        assert report["pages"][0]["ocr_language"] != "en"

    def test_assembler_resumed_without_metadata_fails_admission(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Provenance-free RESUMED must not inherit stale sidecars or ship."""
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        temp_dir = tmp_path / "temp"
        source_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        source_file = source_dir / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4\n%%EOF\n")
        validation_dir = output_dir / "EXPORT" / "VALIDATION"
        validation_dir.mkdir(parents=True, exist_ok=True)
        (validation_dir / "doc.validation.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "pages": [
                        {
                            "page_num": 1,
                            "ocr_method": "PaddleOCR",
                            "ocr_language": "en",
                            "ocr_confidence": 0.997,
                            "text_length": 22,
                            "has_text": True,
                            "status": "ok",
                        },
                    ],
                },
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(pipe, "SOURCE_FOLDER", str(source_dir))
        monkeypatch.setattr(pipe, "OUTPUT_FOLDER", str(output_dir))
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_dir))
        monkeypatch.setattr(pipe, "stop_event", threading.Event())
        monkeypatch.setattr(
            pipe,
            "assembly_producers_done_event",
            threading.Event(),
        )
        monkeypatch.setattr(pipe, "finalization_abort_event", threading.Event())
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "assembly_queue", queue.Queue())
        monkeypatch.setattr(pipe, "image_queue", queue.Queue())
        monkeypatch.setattr(pipe, "compression_queue", queue.Queue())
        monkeypatch.setattr(pipe, "ENABLE_DOCUMENT_INTELLIGENCE", False)
        monkeypatch.setattr(pipe, "ENABLE_VALIDATION", True)
        monkeypatch.setattr(pipe, "_VALIDATION_AVAILABLE", True)
        monkeypatch.setattr(pipe, "ENABLE_NER", False)
        monkeypatch.setattr(pipe, "ENABLE_HANDWRITING", False)
        monkeypatch.setattr(pipe, "ENABLE_CLASSIFICATION", False)
        monkeypatch.setattr(pipe, "ENABLE_EXTRACTION", False)
        fatal = threading.Event()
        failures = []
        monkeypatch.setattr(pipe, "fatal_pipeline_error_event", fatal)
        monkeypatch.setattr(
            pipe,
            "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )
        doc = pipe.DocumentState(str(source_file), "doc-resumed-no-meta", "pdf")
        doc.total_pages = 1
        os.makedirs(doc.temp_dir, exist_ok=True)
        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        with pipe.fitz.open() as chunk_pdf:
            page = chunk_pdf.new_page()
            page.insert_text((72, 72), "Recovered resume text")
            chunk_pdf.save(chunk_path)
        pipe.doc_registry[doc.doc_id] = doc
        import hashlib

        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "status": "RESUMED",
                "text": "",
                "chunk_path": chunk_path,
                "checkpoint_sha256": hashlib.sha256(
                    Path(chunk_path).read_bytes()
                ).hexdigest(),
                "resume_metadata": None,
                "cache_hit": False,
            },
        )
        pipe.assembly_producers_done_event.set()
        thread = threading.Thread(target=pipe.assembler_thread, daemon=True)
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline and doc.doc_id in pipe.doc_registry:
            time.sleep(0.05)
        pipe.stop_event.set()
        thread.join(timeout=5)
        assert fatal.is_set()
        assert any("REUSE_ADMISSION_FAILED" in err for _p, _pg, err in failures)
        assert pipe.compression_queue.empty()
        # Must not overwrite with a successful validation report from recovery.
        report = json.loads(
            (validation_dir / "doc.validation.json").read_text(encoding="utf-8"),
        )
        # Prior trap file may remain only if finalization never rewrote it.
        if report.get("quality"):
            assert report["pages"][0]["ocr_confidence"] != pytest.approx(0.876)
