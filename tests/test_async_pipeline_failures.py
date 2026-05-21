"""Regression tests for async pipeline failure handling paths."""

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


@_skip_no_pipeline
class TestAsyncFailurePaths:
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
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "DocumentState", DummyDocState)
        monkeypatch.setattr(pipe, "get_path_based_doc_id", lambda _p: "doc-zero")
        monkeypatch.setattr(pipe, "detect_language", lambda _p: "en")
        monkeypatch.setattr(pipe, "classify_source_file", lambda _p: ("pdf", None))
        monkeypatch.setattr(pipe, "get_source_page_count", lambda _p, _t: 0)
        monkeypatch.setattr(pipe.os, "walk", lambda _p: [(str(source_dir), [], ["zero.pdf"])])
        monkeypatch.setattr(
            pipe,
            "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        pipe.scheduler_thread()

        assert "doc-zero" not in pipe.doc_registry
        assert failures
        assert failures[0][1] == 0
        assert "SOURCE_PAGE_COUNT_ZERO" in failures[0][2]

    def test_assembler_logs_missing_chunk_after_terminal_status(
        self, tmp_path, monkeypatch,
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
        pipe.assembly_queue.put(
            {"doc_id": "doc-missing", "page_num": 1, "status": "RESUMED", "text": ""},
        )

        thread = threading.Thread(target=pipe.assembler_thread, daemon=True)
        thread.start()

        deadline = time.time() + 5
        while time.time() < deadline and "doc-missing" in pipe.doc_registry:
            time.sleep(0.05)

        pipe.stop_event.set()
        thread.join(timeout=5)

        assert "doc-missing" not in pipe.doc_registry
        assert any(
            "Missing chunk after terminal status RESUMED" in error
            for _path, _page, error in failures
        )

    def test_scheduler_invalidates_stale_resume_cache_for_changed_source(
        self, tmp_path, monkeypatch,
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

        new_fingerprint = {
            "content_sha256": "b" * 64,
            "size_bytes": 34,
            "mtime_ns": 200,
        }

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
        monkeypatch.setattr(pipe, "compute_source_fingerprint_fast", lambda _p: new_fingerprint)
        monkeypatch.setattr(pipe, "detect_language", lambda _p: "en")
        monkeypatch.setattr(pipe, "classify_source_file", lambda _p: ("pdf", None))
        monkeypatch.setattr(pipe, "get_source_page_count", lambda _p, _t: 1)
        monkeypatch.setattr(pipe.os, "walk", lambda _p: [(str(source_dir), [], ["doc.pdf"])])
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
        assert manifest["source_fingerprint"]["content_sha256"] == new_fingerprint["content_sha256"]

    def test_assembler_resumed_page_preserves_prior_validation_confidence(
        self, tmp_path, monkeypatch,
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

        pipe.doc_registry[doc.doc_id] = doc
        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "status": "RESUMED",
                "text": "",
                "chunk_path": chunk_path,
            },
        )

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
        assert report["quality"]["classification"] == "high_quality"
        assert report["quality"]["overall_confidence"] == pytest.approx(0.997, rel=1e-3)
        assert report["pages"][0]["status"] == "ok"
        assert report["pages"][0]["ocr_method"] == "PaddleOCR"
        assert report["pages"][0]["ocr_confidence"] == pytest.approx(0.997, rel=1e-3)
