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

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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

_PDF_META_KEYS = (
    "title",
    "author",
    "subject",
    "keywords",
    "creator",
    "producer",
    "creationDate",
    "modDate",
    "trapped",
)


def _make_doc_state(pipe_mod, source_file, doc_id, source_type="pdf"):
    """Create a DocumentState and register it in the pipeline registry."""
    doc = pipe_mod.DocumentState(str(source_file), doc_id, source_type)
    pipe_mod.doc_registry[doc_id] = doc
    return doc


def _create_chunk_pdf(pipe_mod, path, text="Page content"):
    """Create a minimal valid PDF file at path using fitz."""
    with pipe_mod.fitz.open() as pdf:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
        pdf.save(str(path))


def _default_processing_profile(
    pipe_mod,
    *,
    actual_render_dpi=None,
    page_routing_target="default",
    execution_device="cpu",
    execution_engine="PaddleOCR",
    gpu_optimization_applied=False,
    preprocessing_applied=False,
):
    """Build a standard page-local processing profile for checkpoints."""
    return pipe_mod._checkpoint_processing_profile(
        actual_render_dpi=(
            int(pipe_mod.DPI) if actual_render_dpi is None else int(actual_render_dpi)
        ),
        page_routing_target=page_routing_target,
        execution_device=execution_device,
        execution_engine=execution_engine,
        gpu_optimization_applied=gpu_optimization_applied,
        preprocessing_applied=preprocessing_applied,
    )


def _create_checkpoint_pdf(
    pipe_mod,
    path,
    *,
    document_id,
    page_num,
    text="Page content",
    ocr_language="en",
    ocr_confidence=0.91,
    execution_device="cpu",
    execution_engine="PaddleOCR",
    gpu_optimization_applied=False,
    preprocessing_applied=False,
    actual_render_dpi=None,
    page_routing_target="default",
    status=None,
    ocr_method=None,
    language_data=None,
):
    """Create a page checkpoint with embedded crash-recovery provenance."""
    from pathlib import Path

    path = Path(path)
    _create_chunk_pdf(pipe_mod, path, text)
    extracted = pipe_mod._extract_text_from_chunk_pdf(str(path), strict=True)
    profile = _default_processing_profile(
        pipe_mod,
        actual_render_dpi=actual_render_dpi,
        page_routing_target=page_routing_target,
        execution_device=execution_device,
        execution_engine=execution_engine,
        gpu_optimization_applied=gpu_optimization_applied,
        preprocessing_applied=preprocessing_applied,
    )
    status_value = pipe_mod.PageStatus.PADDLE if status is None else status
    method_value = "PaddleOCR" if ocr_method is None else ocr_method
    with pipe_mod.fitz.open(path) as synthetic_source:
        source_geometry = pipe_mod.PageGeometry.from_pdf_page(synthetic_source[0])
    payload = pipe_mod._embed_page_checkpoint_metadata(
        path.read_bytes(),
        document_id=document_id,
        page_num=page_num,
        status=status_value,
        ocr_method=method_value,
        ocr_language=ocr_language,
        ocr_confidence=ocr_confidence,
        text_length=len(extracted),
        has_text=bool(extracted),
        language_data=language_data,
        processing_profile=profile,
        canonical_text=extracted,
        page_geometry=source_geometry,
    )
    path.write_bytes(payload)
    return extracted


def _create_empty_image_only_checkpoint(
    pipe_mod,
    path,
    *,
    document_id,
    page_num,
    profile_overrides=None,
):
    """Create an ImageOnly-style empty-text checkpoint (no text layer)."""
    from pathlib import Path

    path = Path(path)
    with pipe_mod.fitz.open() as pdf:
        pdf.new_page()
        path.write_bytes(pdf.tobytes())
    profile = _default_processing_profile(
        pipe_mod,
        execution_device="none",
        execution_engine="ImageOnly",
    )
    if profile_overrides:
        profile = dict(profile)
        profile.update(profile_overrides)
    path.write_bytes(
        pipe_mod._embed_page_checkpoint_metadata(
            path.read_bytes(),
            document_id=document_id,
            page_num=page_num,
            status=pipe_mod.PageStatus.IMAGE_ONLY,
            ocr_method="ImageOnly",
            ocr_language="",
            ocr_confidence=0.0,
            text_length=0,
            has_text=False,
            processing_profile=profile,
            canonical_text="",
            page_geometry=pipe_mod.PageGeometry(
                595, 842, "pdf",
                source_cropbox=(0, 0, 595, 842),
                source_mediabox=(0, 0, 595, 842),
            ),
        ),
    )


def _checkpoint_sha256_for_path(path):
    """Lowercase SHA-256 of durable checkpoint bytes (test helper)."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _content_fingerprint(data: bytes):
    return {
        "content_sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _linux_sealed_salvage_fixture(pipe_mod, monkeypatch, tmp_path, *, doc_id, original):
    """Sealed memfd + durable snapshot + resume manifest under patched TEMP."""
    temp_root = tmp_path / "temp"
    monkeypatch.setattr(pipe_mod, "TEMP_FOLDER", str(temp_root))
    doc_temp = temp_root / doc_id
    snapshot_dir = doc_temp / pipe_mod.SOURCE_SNAPSHOT_DIRNAME
    snapshot_dir.mkdir(parents=True)
    canonical = snapshot_dir / "source.pdf"
    canonical.write_bytes(original)
    fingerprint = _content_fingerprint(original)
    pipe_mod._write_resume_manifest(str(doc_temp), "matter.pdf", fingerprint)
    authority = pipe_mod._open_immutable_source_authority(str(canonical), fingerprint)
    original_input = tmp_path / "original-input.pdf"
    original_input.write_bytes(original)
    doc_state = SimpleNamespace(
        path=str(original_input),
        source_authority=authority,
        source_snapshot_path=str(canonical),
        source_fingerprint=fingerprint,
        temp_dir=str(doc_temp),
    )
    return SimpleNamespace(
        temp_root=temp_root,
        doc_temp=doc_temp,
        canonical=canonical,
        authority=authority,
        fingerprint=fingerprint,
        doc_state=doc_state,
        original_input=original_input,
    )


def _run_extractor_chunk(
    pipe_mod,
    monkeypatch,
    *,
    doc_id,
    queued_path,
    registry,
    iter_source_images,
    start=1,
    end=1,
    lang_hint="en",
    source_type="pdf",
    iter_fitz_image_frames=None,
):
    """Drive one extractor chunk to a terminal assembly message."""
    # Custody/race tests deliberately use non-PDF bytes and stub the renderer;
    # supply an explicit synthetic page-space contract for this isolated lane.
    monkeypatch.setattr(
        pipe_mod, "_capture_source_geometries",
        lambda _path, first, last, _kind: {
            number: pipe_mod.PageGeometry(8, 8, "checkpoint-local")
            for number in range(first, last + 1)
        },
    )
    hard_stop = threading.Event()
    chunks = queue.Queue()
    images = queue.Queue()
    assembly = queue.Queue()
    failures = []
    chunks.put((doc_id, queued_path, start, end, lang_hint, source_type))
    monkeypatch.setattr(pipe_mod, "hard_stop_event", hard_stop)
    monkeypatch.setattr(pipe_mod, "chunk_queue", chunks)
    monkeypatch.setattr(pipe_mod, "image_queue", images)
    monkeypatch.setattr(pipe_mod, "assembly_queue", assembly)
    monkeypatch.setattr(pipe_mod, "EXTRACTOR_MODE", "thread")
    monkeypatch.setattr(pipe_mod, "extractor_process_pool", None)
    monkeypatch.setattr(pipe_mod, "doc_registry", registry)
    monkeypatch.setattr(pipe_mod, "doc_registry_lock", threading.RLock())
    monkeypatch.setattr(pipe_mod, "iter_source_images", iter_source_images)
    if iter_fitz_image_frames is not None:
        monkeypatch.setattr(pipe_mod, "iter_fitz_image_frames", iter_fitz_image_frames)
    monkeypatch.setattr(
        pipe_mod,
        "log_failure",
        lambda input_path, page_num, reason: failures.append(
            (input_path, page_num, reason)
        ),
    )
    thread = threading.Thread(target=pipe_mod.extractor_thread, args=(0,), daemon=True)
    thread.start()
    assert pipe_mod._join_queue_with_timeout(chunks, 5)
    hard_stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    messages = []
    while not assembly.empty():
        messages.append(assembly.get_nowait())
    queued_images = []
    while not images.empty():
        queued_images.append(images.get_nowait())
    return queued_images, messages, failures


def _rewrite_checkpoint_subject(pipe_mod, path, payload):
    """Overwrite PDF subject with a checkpoint metadata JSON payload."""
    path = str(path)
    with pipe_mod.fitz.open(path) as doc:
        subject = pipe_mod.CHECKPOINT_METADATA_SUBJECT_PREFIX + json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
        current = doc.metadata or {}
        md = {k: str(current.get(k, "") or "") for k in _PDF_META_KEYS}
        md["subject"] = subject
        doc.set_metadata(md)
        data = doc.tobytes()
    with open(path, "wb") as handle:
        handle.write(data)


def _apply_metadata_envelope(meta, envelope):
    """Merge top-level and nested processing_profile envelope overrides."""
    broken = dict(meta)
    profile = dict(meta.get("processing_profile") or {})
    for key, value in envelope.items():
        if key == "processing_profile" and isinstance(value, dict):
            profile.update(value)
        else:
            broken[key] = value
    broken["processing_profile"] = profile
    return broken


def _put_live_assembly_page(
    pipe_mod,
    doc,
    page_num,
    text,
    *,
    status=None,
    ocr_method="PaddleOCR",
    ocr_confidence=0.9,
    ocr_language="en",
):
    """Write a V3 checkpoint and queue a live (non-reuse) assembly message."""
    path = os.path.join(doc.temp_dir, f"{page_num}.pdf")
    status_value = pipe_mod.PageStatus.PADDLE if status is None else status
    extracted = _create_checkpoint_pdf(
        pipe_mod,
        path,
        document_id=doc.doc_id,
        page_num=page_num,
        text=text,
        ocr_language=ocr_language,
        ocr_confidence=ocr_confidence,
        status=status_value,
        ocr_method=ocr_method,
        execution_engine=(
            "PaddleOCR"
            if ocr_method == "PaddleOCR"
            else str(ocr_method.value if hasattr(ocr_method, "value") else ocr_method)
        ),
    )
    pipe_mod.assembly_queue.put(
        {
            "doc_id": doc.doc_id,
            "page_num": page_num,
            "text": extracted,
            "status": status_value,
            "chunk_path": path,
            "checkpoint_sha256": _checkpoint_sha256_for_path(path),
            "ocr_confidence": ocr_confidence,
            "ocr_method": ocr_method,
            "ocr_language": ocr_language,
            "resume_metadata": None,
            "cache_hit": False,
        }
    )
    return extracted


def _put_resumed_assembly_page(
    pipe_mod,
    doc,
    page_num,
    text,
    *,
    ocr_confidence=0.9,
    ocr_language="en",
):
    """Write a V3 checkpoint and queue a RESUMED message with admission fields."""
    path = os.path.join(doc.temp_dir, f"{page_num}.pdf")
    extracted = _create_checkpoint_pdf(
        pipe_mod,
        path,
        document_id=doc.doc_id,
        page_num=page_num,
        text=text,
        ocr_language=ocr_language,
        ocr_confidence=ocr_confidence,
    )
    meta = pipe_mod._load_page_checkpoint_metadata(
        path,
        document_id=doc.doc_id,
        page_num=page_num,
    )
    pipe_mod.assembly_queue.put(
        {
            "doc_id": doc.doc_id,
            "page_num": page_num,
            "text": extracted,
            "status": "RESUMED",
            "chunk_path": path,
            "checkpoint_sha256": _checkpoint_sha256_for_path(path),
            "ocr_confidence": meta["ocr_confidence"],
            "ocr_method": meta["ocr_method"],
            "ocr_language": meta["ocr_language"],
            "resume_metadata": meta,
            "cache_hit": False,
        }
    )
    return extracted, meta


def _put_terminal_status(pipe_mod, doc_id, page_num, status):
    """Queue a terminal failure status with no chunk (EXTRACT/CRITICAL_FAILED)."""
    pipe_mod.assembly_queue.put(
        {
            "doc_id": doc_id,
            "page_num": page_num,
            "text": "",
            "status": status,
            "chunk_path": None,
            "checkpoint_sha256": None,
            "cache_hit": False,
        }
    )


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
    monkeypatch.setattr(
        pipe_mod,
        "assembly_producers_done_event",
        threading.Event(),
    )
    monkeypatch.setattr(pipe_mod, "finalization_abort_event", threading.Event())
    monkeypatch.setattr(pipe_mod, "fatal_pipeline_error_event", threading.Event())
    monkeypatch.setattr(pipe_mod, "doc_registry", {})
    monkeypatch.setattr(pipe_mod, "assembly_queue", queue.Queue())
    monkeypatch.setattr(pipe_mod, "image_queue", queue.Queue())
    monkeypatch.setattr(pipe_mod, "compression_queue", queue.Queue())

    return source_dir, output_dir, temp_dir


def _setup_assembler_case(
    monkeypatch,
    pipe_mod,
    tmp_path,
    *,
    source_name,
    doc_id,
    total_pages,
    capture_fatal=False,
    source_bytes=b"%PDF-1.4\n%%EOF\n",
    feature_flags_off=True,
):
    """Dirs, feature flags, failure capture, and a registered DocumentState."""
    source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
        monkeypatch,
        pipe_mod,
        tmp_path,
    )
    if feature_flags_off:
        _monkeypatch_feature_flags_off(monkeypatch, pipe_mod)

    failures = []
    monkeypatch.setattr(
        pipe_mod,
        "log_failure",
        lambda path, page, error: failures.append((path, page, error)),
    )
    fatal = None
    if capture_fatal:
        fatal = threading.Event()
        monkeypatch.setattr(pipe_mod, "fatal_pipeline_error_event", fatal)

    source_file = source_dir / source_name
    source_file.write_bytes(source_bytes)
    doc = _make_doc_state(pipe_mod, source_file, doc_id)
    doc.total_pages = total_pages
    os.makedirs(doc.temp_dir, exist_ok=True)
    return source_dir, output_dir, temp_dir, source_file, doc, failures, fatal


def _setup_scheduler_resume(
    monkeypatch,
    pipe_mod,
    tmp_path,
    *,
    source_name,
    source_bytes,
    doc_id,
    total_pages,
    existing_pages,
    page_text=None,
    conf_fn=None,
):
    """Source + checkpoint pages + resume manifest + scheduler monkeypatches.

    Returns ``(source_file, doc_temp, fingerprint)``.
    """
    source_dir = tmp_path / "source"
    temp_dir = tmp_path / "temp"
    source_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    source_file = source_dir / source_name
    source_file.write_bytes(source_bytes)
    doc_temp = temp_dir / doc_id
    doc_temp.mkdir(parents=True, exist_ok=True)

    for p in existing_pages:
        text = page_text(p) if callable(page_text) else (page_text or f"Page {p}")
        kwargs = {
            "document_id": doc_id,
            "page_num": p,
            "text": text,
        }
        if conf_fn is not None:
            kwargs["ocr_confidence"] = conf_fn(p)
        _create_checkpoint_pdf(pipe_mod, doc_temp / f"{p}.pdf", **kwargs)

    fp = pipe_mod.compute_source_fingerprint(str(source_file))
    pipe_mod._write_resume_manifest(str(doc_temp), str(source_file), fp)

    class DummyDocState:
        def __init__(self, path, did, source_type):
            self.path = path
            self.doc_id = did
            self.source_type = source_type
            self.total_pages = 0
            self.custody_chain = None
            self.completion_fence = pipe_mod.DocumentCompletionFence(did)
            self.temp_dir = os.path.join(str(temp_dir), did)
            os.makedirs(self.temp_dir, exist_ok=True)

    monkeypatch.setattr(pipe_mod, "SOURCE_FOLDER", str(source_dir))
    monkeypatch.setattr(pipe_mod, "TEMP_FOLDER", str(temp_dir))
    monkeypatch.setattr(pipe_mod, "stop_event", threading.Event())
    monkeypatch.setattr(pipe_mod, "doc_registry", {})
    monkeypatch.setattr(pipe_mod, "chunk_queue", queue.Queue())
    monkeypatch.setattr(pipe_mod, "assembly_queue", queue.Queue())
    monkeypatch.setattr(pipe_mod, "DocumentState", DummyDocState)
    monkeypatch.setattr(pipe_mod, "build_resume_doc_id", lambda _p, _f: doc_id)
    monkeypatch.setattr(pipe_mod, "compute_source_fingerprint", lambda _p: fp)
    monkeypatch.setattr(pipe_mod, "detect_language", lambda _p: "en")
    monkeypatch.setattr(pipe_mod, "classify_source_file", lambda _p: ("pdf", None))
    monkeypatch.setattr(pipe_mod, "get_source_page_count", lambda _p, _t: total_pages)
    monkeypatch.setattr(
        pipe_mod.os,
        "walk",
        lambda _p: [(str(source_dir), [], [source_name])],
    )
    monkeypatch.setattr(pipe_mod, "log_failure", lambda *_a, **_k: None)
    return source_file, doc_temp, fp


def _drain_resumed_messages(pipe_mod):
    """Drain assembly_queue RESUMED messages (order preserved)."""
    resumed = []
    while not pipe_mod.assembly_queue.empty():
        msg = pipe_mod.assembly_queue.get_nowait()
        if msg["status"] == "RESUMED":
            resumed.append(msg)
    return resumed


def _drain_queued_pages(pipe_mod):
    """Drain chunk_queue into a page set and chunk count."""
    queued_pages = set()
    chunk_count = 0
    while not pipe_mod.chunk_queue.empty():
        _, _, start, end, _, _ = pipe_mod.chunk_queue.get_nowait()
        chunk_count += 1
        for p in range(start, end + 1):
            queued_pages.add(p)
    return queued_pages, chunk_count


def _run_assembler_until_done(pipe_mod, doc_id, timeout=10):
    """Run assembler thread until doc_id is finalized or timeout."""
    pipe_mod.assembly_producers_done_event.set()
    thread = threading.Thread(target=pipe_mod.assembler_thread, daemon=True)
    thread.start()

    deadline = time.time() + timeout
    while time.time() < deadline and doc_id in pipe_mod.doc_registry:
        time.sleep(0.05)

    pipe_mod.stop_event.set()
    thread.join(timeout=5)


def _run_assembler_until_queue_drained(pipe_mod, timeout=3):
    """Process preloaded messages without waiting for a rejected doc to vanish."""
    pipe_mod.assembly_producers_done_event.set()
    thread = threading.Thread(target=pipe_mod.assembler_thread, daemon=True)
    thread.start()
    assert pipe_mod._join_queue_with_timeout(pipe_mod.assembly_queue, timeout)
    pipe_mod.stop_event.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _run_assembler_until_all_done(pipe_mod, doc_ids, timeout=10):
    """Run assembler thread until all doc_ids are finalized or timeout."""
    pipe_mod.assembly_producers_done_event.set()
    thread = threading.Thread(target=pipe_mod.assembler_thread, daemon=True)
    thread.start()

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(d_id in pipe_mod.doc_registry for d_id in doc_ids):
            break
        time.sleep(0.05)

    pipe_mod.stop_event.set()
    thread.join(timeout=5)


def _assert_assembly_fatal_no_output(pipe_mod, doc, failures, fatal, *, err_token):
    """Common fatal-finalization assertions (failure signal, no compression)."""
    assert fatal.is_set()
    assert any(err_token in err for _p, _pg, err in failures)
    assert pipe_mod.compression_queue.empty()


def _language_share_payload(share_map, counts_map=None):
    """Minimal language_data dict for share-type gate tests."""
    counts = counts_map or {lang: 10 for lang in share_map}
    return {
        "page_num": 1,
        "primary_language": "en",
        "primary_confidence": 0.9,
        "languages_detected": list(share_map.keys()),
        "language_char_shares": share_map,
        "language_char_counts": counts,
        "scripts_detected": ["latin"],
        "mixed_script": False,
        "span_count": 1,
        "spans_labeled": 1,
    }


def _make_page_language(
    pipe_mod,
    *,
    page_num,
    language,
    confidence,
    char_count,
):
    """Compact PageLanguage fixture used by recovery weighting tests."""
    return pipe_mod.PageLanguage(
        page_num=page_num,
        primary_language=language,
        primary_confidence=confidence,
        languages_detected=[language],
        language_char_shares={language: 1.0},
        scripts_detected=["latin"],
        mixed_script=False,
        span_count=1,
        spans_labeled=1,
        spans=[
            pipe_mod.SpanLanguage(
                bbox=[],
                text_sample="",
                language=language,
                confidence=confidence,
                script="latin",
                detection_method="fasttext",
                char_count=char_count,
            ),
        ],
    )


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
        self,
        tmp_path,
        monkeypatch,
    ):
        _setup_scheduler_resume(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="report.pdf",
            source_bytes=b"%PDF-1.4\nTest document\n%%EOF\n",
            doc_id="doc-midcrash",
            total_pages=10,
            existing_pages=range(1, 6),
            conf_fn=lambda p: 0.90 + (p / 100),
        )
        pipe.scheduler_thread()

        resumed_messages = _drain_resumed_messages(pipe)
        assert sorted(msg["page_num"] for msg in resumed_messages) == [1, 2, 3, 4, 5]
        for msg in resumed_messages:
            assert msg["ocr_method"] == "PaddleOCR"
            assert msg["ocr_language"] == "en"
            assert msg["ocr_confidence"] == pytest.approx(
                0.90 + (msg["page_num"] / 100),
            )
            assert msg["resume_metadata"]["validation_status"] == "ok"
            assert msg["resume_metadata"]["has_text"] is True
            expected_text = f"Page {msg['page_num']}"
            # Scheduler must queue the exact validated checkpoint text.
            assert msg["text"] == expected_text
            assert len(msg["text"]) == msg["resume_metadata"]["text_length"]
            assert bool(msg["text"]) is msg["resume_metadata"]["has_text"]

        queued_pages, _ = _drain_queued_pages(pipe)
        assert queued_pages == {6, 7, 8, 9, 10}

    def test_scheduler_content_reads_and_queue_use_immutable_authority(
        self,
        tmp_path,
        monkeypatch,
    ):
        original = b"%PDF-1.4\nimmutable-generation\n%%EOF\n"
        source_file, _doc_temp, _fp = _setup_scheduler_resume(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="spooled.pdf",
            source_bytes=original,
            doc_id="doc-spooled",
            total_pages=1,
            existing_pages=[],
        )
        observed = []

        def observe_spool(path, *_args):
            observed.append(path)
            assert path != str(source_file)
            assert Path(path).read_bytes() == original
            # A source-path swap/restore cannot influence later reads because
            # every callback and queued extraction path owns the spool.
            source_file.write_bytes(b"swapped")
            source_file.write_bytes(original)
            return "en" if len(observed) == 1 else 1

        monkeypatch.setattr(pipe, "detect_language", observe_spool)
        monkeypatch.setattr(pipe, "get_source_page_count", observe_spool)

        pipe.scheduler_thread()

        queued = pipe.chunk_queue.get_nowait()
        spool_path = queued[1]
        assert observed == [spool_path, spool_path]
        authority = pipe.doc_registry["doc-spooled"].source_authority
        assert spool_path == authority.runtime_path
        assert Path(spool_path).parts[:2] == ("/", "proc")
        assert pipe.SOURCE_SNAPSHOT_DIRNAME in Path(authority.canonical_path).parts
        assert Path(spool_path).read_bytes() == original

    def test_scheduler_classifies_only_captured_generation_header(
        self,
        tmp_path,
        monkeypatch,
    ):
        source_file, _doc_temp, _fp = _setup_scheduler_resume(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="classified.pdf",
            source_bytes=b"%PDF-1.4\nclassified generation\n%%EOF\n",
            doc_id="doc-classified-spool",
            total_pages=1,
            existing_pages=[],
        )
        observed = []

        def classify_capture(path, header):
            observed.append((path, header))
            assert path == str(source_file)
            assert header == b"%PDF-1.4\nclassified generation\n%%EOF\n"
            return "pdf", None

        monkeypatch.setattr(pipe, "_classify_captured_source", classify_capture)
        pipe.scheduler_thread()

        assert len(observed) == 1
        assert observed[0][0] == str(source_file)

    def test_snapshot_rejects_source_generation_change_during_copy(
        self,
        tmp_path,
        monkeypatch,
    ):
        source = tmp_path / "source.pdf"
        source.write_bytes(b"stable bytes")
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(tmp_path / "temp"))
        real_fstat = pipe.os.fstat
        calls = 0

        def drifting_fstat(fd):
            nonlocal calls
            calls += 1
            stat_info = real_fstat(fd)
            if calls != 2:
                return stat_info
            return SimpleNamespace(
                st_dev=stat_info.st_dev,
                st_ino=stat_info.st_ino,
                st_size=stat_info.st_size,
                st_mtime=stat_info.st_mtime,
                st_ctime=stat_info.st_ctime,
                st_mtime_ns=stat_info.st_mtime_ns + 1,
                st_ctime_ns=stat_info.st_ctime_ns,
            )

        monkeypatch.setattr(pipe.os, "fstat", drifting_fstat)
        with pytest.raises(RuntimeError, match="source generation changed"):
            pipe._stage_source_snapshot(str(source))
        assert list((tmp_path / "temp").glob(".source-snapshot-*")) == []

    def test_failed_spool_replace_preserves_prior_recovery_snapshot(
        self,
        tmp_path,
        monkeypatch,
    ):
        doc_temp = tmp_path / "doc"
        snapshot_dir = doc_temp / pipe.SOURCE_SNAPSHOT_DIRNAME
        snapshot_dir.mkdir(parents=True)
        prior = snapshot_dir / "source.pdf"
        prior.write_bytes(b"prior-valid-generation")
        staged = tmp_path / "new.pdf"
        staged.write_bytes(b"new-generation")

        monkeypatch.setattr(
            pipe.os,
            "replace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace denied")),
        )
        with pytest.raises(OSError, match="replace denied"):
            pipe._publish_source_snapshot(
                str(staged),
                str(doc_temp),
                "matter.pdf",
                {"content_sha256": "0" * 64, "size_bytes": 14},
            )

        assert prior.read_bytes() == b"prior-valid-generation"

    @pytest.mark.skipif(os.name != "nt", reason="Windows mmap sharing semantics")
    def test_windows_open_authority_blocks_swap_and_in_place_overwrite(
        self,
        tmp_path,
    ):
        canonical = tmp_path / "source.pdf"
        original = b"%PDF-1.4\nopened generation\n%%EOF\n"
        canonical.write_bytes(original)
        fingerprint = {
            "content_sha256": hashlib.sha256(original).hexdigest(),
            "size_bytes": len(original),
        }
        authority = pipe._open_immutable_source_authority(
            str(canonical),
            fingerprint,
        )
        replacement = tmp_path / "replacement.pdf"
        replacement.write_bytes(b"%PDF-1.4\nreplacement\n%%EOF\n")
        try:
            with pytest.raises(OSError):
                os.replace(replacement, canonical)
            with pytest.raises(OSError):
                canonical.write_bytes(b"in-place attacker")
            assert Path(authority.runtime_path).read_bytes() == original
            assert (
                authority.verify(fingerprint)["content_sha256"]
                == (fingerprint["content_sha256"])
            )
        finally:
            authority.close()

    def test_open_authority_rejects_resume_bytes_not_matching_manifest(
        self,
        tmp_path,
    ):
        canonical = tmp_path / "source.pdf"
        original = b"%PDF-1.4\ncurrent bytes\n%%EOF\n"
        canonical.write_bytes(original)
        wrong = {
            "content_sha256": hashlib.sha256(b"different original").hexdigest(),
            "size_bytes": len(original),
        }
        with pytest.raises(RuntimeError, match="diverg"):
            pipe._open_immutable_source_authority(str(canonical), wrong)

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="Linux sealed-memfd authority",
    )
    def test_linux_sealed_authority_is_child_readable_and_not_writable(
        self,
        tmp_path,
    ):
        canonical = tmp_path / "source.pdf"
        original = b"%PDF-1.4\nsealed generation\n%%EOF\n"
        canonical.write_bytes(original)
        fingerprint = {
            "content_sha256": hashlib.sha256(original).hexdigest(),
            "size_bytes": len(original),
        }
        authority = pipe._open_immutable_source_authority(
            str(canonical),
            fingerprint,
        )
        try:
            reader = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import hashlib,sys;"
                        "print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())"
                    ),
                    authority.runtime_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            assert reader.stdout.strip() == fingerprint["content_sha256"]
            writer = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,sys;"
                        "f=os.open(sys.argv[1],os.O_WRONLY);"
                        "os.write(f,b'X')"
                    ),
                    authority.runtime_path,
                ],
                capture_output=True,
                text=True,
            )
            assert writer.returncode != 0
            assert (
                authority.verify(fingerprint)["content_sha256"]
                == (fingerprint["content_sha256"])
            )
        finally:
            authority.close()

    def test_process_partial_failure_thread_fallback_skips_queued_pages(
        self,
        tmp_path,
        monkeypatch,
    ):
        from PIL import Image

        hard_stop = threading.Event()
        chunks = queue.Queue()
        images = queue.Queue()
        assembly = queue.Queue()
        source = tmp_path / "source.pdf"
        source.write_bytes(b"%PDF-1.4\n%%EOF\n")
        chunks.put(("doc-process", str(source), 1, 3, "en", "pdf"))

        class ImmediateFuture:
            def result(self):
                # Page 1 is queued successfully; decoding page 2 then fails.
                return [(1, b"one"), (2, b"bad"), (3, b"three")]

        class Pool:
            def submit(self, *_args, **_kwargs):
                return ImmediateFuture()

        def decode(payload):
            if payload == b"bad":
                raise RuntimeError("decode failure after first queued page")
            return Image.new("RGB", (8, 8), "white")

        def thread_images(*_args, **_kwargs):
            for page_num in range(1, 4):
                yield page_num, Image.new("RGB", (8, 8), "white")

        monkeypatch.setattr(pipe, "hard_stop_event", hard_stop)
        monkeypatch.setattr(pipe, "chunk_queue", chunks)
        monkeypatch.setattr(pipe, "image_queue", images)
        monkeypatch.setattr(pipe, "assembly_queue", assembly)
        monkeypatch.setattr(pipe, "EXTRACTOR_MODE", "process")
        monkeypatch.setattr(pipe, "extractor_process_pool", Pool())
        monkeypatch.setattr(pipe, "_decode_image_from_jpeg_bytes", decode)
        monkeypatch.setattr(pipe, "iter_source_images", thread_images)
        monkeypatch.setattr(
            pipe, "_capture_source_geometries",
            lambda _path, first, last, _kind: {
                number: pipe.PageGeometry(8, 8, "checkpoint-local")
                for number in range(first, last + 1)
            },
        )
        monkeypatch.setattr(
            pipe,
            "doc_registry",
            {"doc-process": SimpleNamespace(path=str(source))},
        )

        thread = threading.Thread(target=pipe.extractor_thread, args=(0,), daemon=True)
        thread.start()
        assert pipe._join_queue_with_timeout(chunks, 3)
        hard_stop.set()
        thread.join(timeout=3)
        assert not thread.is_alive()

        queued_pages = []
        while not images.empty():
            queued_pages.append(images.get_nowait().page_num)
        assert queued_pages == [1, 2, 3]
        assert assembly.empty()

    def test_missing_registry_emits_every_page_failure_without_killing_extractor(
        self,
        tmp_path,
        monkeypatch,
    ):
        hard_stop = threading.Event()
        chunks = queue.Queue()
        images = queue.Queue()
        assembly = queue.Queue()
        source = tmp_path / "missing-registry.pdf"
        source.write_bytes(b"%PDF-1.4\n%%EOF\n")
        chunks.put(("doc-missing-registry", str(source), 1, 2, "en", "pdf"))
        failures = []

        monkeypatch.setattr(pipe, "hard_stop_event", hard_stop)
        monkeypatch.setattr(pipe, "chunk_queue", chunks)
        monkeypatch.setattr(pipe, "image_queue", images)
        monkeypatch.setattr(pipe, "assembly_queue", assembly)
        monkeypatch.setattr(pipe, "doc_registry", {})
        monkeypatch.setattr(pipe, "doc_registry_lock", threading.RLock())
        monkeypatch.setattr(
            pipe,
            "log_failure",
            lambda input_path, page_num, reason: failures.append(
                (input_path, page_num, reason)
            ),
        )

        thread = threading.Thread(target=pipe.extractor_thread, args=(0,), daemon=True)
        thread.start()
        assert pipe._join_queue_with_timeout(chunks, 3)
        hard_stop.set()
        thread.join(timeout=3)

        assert not thread.is_alive()
        messages = [assembly.get_nowait(), assembly.get_nowait()]
        assert [message["page_num"] for message in messages] == [1, 2]
        assert all(
            message["status"] == pipe.PageStatus.EXTRACT_FAILED for message in messages
        )
        assert [entry[1] for entry in failures] == [1, 2]
        assert all(entry[0] == str(source) for entry in failures)

    def test_thread_extractor_preserves_renderer_page_numbers(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A missing rendered page must not shift later source page identity."""
        from PIL import Image

        hard_stop = threading.Event()
        chunks = queue.Queue()
        images = queue.Queue()
        assembly = queue.Queue()
        source = tmp_path / "source.pdf"
        source.write_bytes(b"%PDF-1.4\n%%EOF\n")
        chunks.put(("doc-thread", str(source), 1, 3, "en", "pdf"))

        def thread_images(*_args, **_kwargs):
            yield 1, Image.new("RGB", (8, 8), "white")
            yield 3, Image.new("RGB", (8, 8), "white")

        monkeypatch.setattr(pipe, "hard_stop_event", hard_stop)
        monkeypatch.setattr(pipe, "chunk_queue", chunks)
        monkeypatch.setattr(pipe, "image_queue", images)
        monkeypatch.setattr(pipe, "assembly_queue", assembly)
        monkeypatch.setattr(pipe, "EXTRACTOR_MODE", "thread")
        monkeypatch.setattr(pipe, "iter_source_images", thread_images)
        monkeypatch.setattr(
            pipe, "_capture_source_geometries",
            lambda _path, first, last, _kind: {
                number: pipe.PageGeometry(8, 8, "checkpoint-local")
                for number in range(first, last + 1)
            },
        )
        monkeypatch.setattr(
            pipe,
            "doc_registry",
            {"doc-thread": SimpleNamespace(path=str(source))},
        )

        thread = threading.Thread(target=pipe.extractor_thread, args=(0,), daemon=True)
        thread.start()
        assert pipe._join_queue_with_timeout(chunks, 3)
        hard_stop.set()
        thread.join(timeout=3)
        assert not thread.is_alive()

        queued_pages = []
        while not images.empty():
            queued_pages.append(images.get_nowait().page_num)
        assert queued_pages == [1, 3]
        failed_pages = []
        while not assembly.empty():
            failed_pages.append(assembly.get_nowait()["page_num"])
        assert failed_pages == [2]

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="Linux sealed-memfd live salvage",
    )
    def test_linux_live_authority_salvage_survives_snapshot_replace(
        self,
        tmp_path,
        monkeypatch,
    ):
        from PIL import Image

        original = b"%PDF-1.4\nsealed salvage generation\n%%EOF\n"
        doc_id = "doc-live-salvage"
        fixture = _linux_sealed_salvage_fixture(
            pipe, monkeypatch, tmp_path, doc_id=doc_id, original=original
        )
        rastered = []

        def fake_iter(path, start, end, source_type):
            del start, end, source_type
            if not rastered:
                replacement = tmp_path / "attacker.pdf"
                replacement.write_bytes(b"%PDF-1.4\nattacker\n%%EOF\n")
                os.replace(str(replacement), str(fixture.canonical))
                fixture.canonical.write_bytes(b"truncated")
                rastered.append(path)
                raise RuntimeError("forced extract failure")
            rastered.append(path)
            assert Path(path).read_bytes() == original
            yield 1, Image.new("RGB", (8, 8), "white")

        try:
            _images, messages, failures = _run_extractor_chunk(
                pipe,
                monkeypatch,
                doc_id=doc_id,
                queued_path=fixture.authority.runtime_path,
                registry={doc_id: fixture.doc_state},
                iter_source_images=fake_iter,
            )
            assert rastered == [
                fixture.authority.runtime_path,
                fixture.authority.runtime_path,
            ]
            assert Path(fixture.authority.runtime_path).read_bytes() == original
            assert (
                fixture.authority.verify(fixture.fingerprint)["content_sha256"]
                == fixture.fingerprint["content_sha256"]
            )
            assert fixture.canonical.read_bytes() == b"truncated"
            assert [msg["status"] for msg in messages] == [pipe.PageStatus.IMAGE_ONLY]
            assert messages[0]["chunk_path"]
            assert failures[0][2].startswith("EXTRACT_SALVAGED_IMAGE_ONLY:")
        finally:
            fixture.authority.close()

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="Linux sealed-memfd restart salvage",
    )
    def test_linux_closed_fd_matching_manifest_may_salvage_verified_snapshot(
        self,
        tmp_path,
        monkeypatch,
    ):
        from PIL import Image

        original = b"%PDF-1.4\nrestart salvage generation\n%%EOF\n"
        doc_id = "doc-restart-salvage"
        fixture = _linux_sealed_salvage_fixture(
            pipe, monkeypatch, tmp_path, doc_id=doc_id, original=original
        )
        dead_runtime = fixture.authority.runtime_path
        fixture.authority.close()
        rastered = []
        salvage_bytes = []
        origins = []
        real_select = pipe._select_salvage_source

        def wrapped_select(doc_id_arg, source_runtime_path):
            selected = real_select(doc_id_arg, source_runtime_path)
            origins.append(None if selected is None else selected.origin)
            return selected

        monkeypatch.setattr(pipe, "_select_salvage_source", wrapped_select)

        def fake_iter(path, start, end, source_type):
            del start, end, source_type
            rastered.append(path)
            if len(rastered) == 1:
                raise RuntimeError("live fd gone")
            salvage_bytes.append(Path(path).read_bytes())
            yield 1, Image.new("RGB", (8, 8), "white")

        _images, messages, failures = _run_extractor_chunk(
            pipe,
            monkeypatch,
            doc_id=doc_id,
            queued_path=dead_runtime,
            registry={doc_id: fixture.doc_state},
            iter_source_images=fake_iter,
        )
        assert rastered[0] == dead_runtime
        assert len(rastered) == 2
        salvage_path = rastered[1]
        assert salvage_path.startswith(f"/proc/{os.getpid()}/fd/")
        assert salvage_path != str(fixture.canonical)
        assert salvage_bytes == [original]
        assert origins == ["restart_snapshot"]
        assert [msg["status"] for msg in messages] == [pipe.PageStatus.IMAGE_ONLY]
        assert failures[0][2].startswith("EXTRACT_SALVAGED_IMAGE_ONLY:")

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="Linux sealed-memfd restart salvage",
    )
    def test_linux_closed_fd_diverged_durable_is_extract_failed(
        self,
        tmp_path,
        monkeypatch,
    ):
        from PIL import Image

        original = b"%PDF-1.4\nrestart mismatch generation\n%%EOF\n"
        doc_id = "doc-restart-mismatch"
        fixture = _linux_sealed_salvage_fixture(
            pipe, monkeypatch, tmp_path, doc_id=doc_id, original=original
        )
        dead_runtime = fixture.authority.runtime_path
        fixture.authority.close()
        fixture.canonical.write_bytes(b"%PDF-1.4\ntampered durable bytes\n%%EOF\n")
        rastered = []
        fitz_paths = []
        salvage_kwargs = []
        origins = []
        real_select = pipe._select_salvage_source

        def wrapped_select(doc_id_arg, source_runtime_path):
            selected = real_select(doc_id_arg, source_runtime_path)
            origins.append(None if selected is None else selected.origin)
            return selected

        monkeypatch.setattr(pipe, "_select_salvage_source", wrapped_select)

        def fake_iter(path, start, end, source_type):
            del start, end, source_type
            rastered.append(path)
            raise RuntimeError("live fd gone")

        def fake_fitz(path, start, end):
            del start, end
            fitz_paths.append(path)
            yield 1, Image.new("RGB", (8, 8), "white")

        real_salvage = pipe._salvage_extractor_image_only

        def wrapped_salvage(**kwargs):
            salvage_kwargs.append(kwargs)
            return real_salvage(**kwargs)

        monkeypatch.setattr(pipe, "_salvage_extractor_image_only", wrapped_salvage)
        _images, messages, failures = _run_extractor_chunk(
            pipe,
            monkeypatch,
            doc_id=doc_id,
            queued_path=dead_runtime,
            registry={doc_id: fixture.doc_state},
            iter_source_images=fake_iter,
            iter_fitz_image_frames=fake_fitz,
        )
        assert salvage_kwargs == []
        assert origins == [None]
        assert fitz_paths == []
        assert [msg["status"] for msg in messages] == [pipe.PageStatus.EXTRACT_FAILED]
        assert messages[0]["chunk_path"] is None
        assert failures[0][2].startswith("EXTRACT_FAILED:")
        assert pipe.PageStatus.IMAGE_ONLY not in [msg["status"] for msg in messages]

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="Linux sealed-memfd live salvage",
    )
    def test_linux_registry_present_does_not_prefer_source_snapshot_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        from PIL import Image

        original = b"%PDF-1.4\nlive preferred generation\n%%EOF\n"
        trap = b"%PDF-1.4\ntrap snapshot bytes\n%%EOF\n"
        doc_id = "doc-registry-prefer-runtime"
        fixture = _linux_sealed_salvage_fixture(
            pipe, monkeypatch, tmp_path, doc_id=doc_id, original=original
        )
        fixture.canonical.write_bytes(trap)
        rastered = []
        registry = {doc_id: fixture.doc_state}
        monkeypatch.setattr(pipe, "doc_registry", registry)
        monkeypatch.setattr(pipe, "doc_registry_lock", threading.RLock())

        def fake_iter(path, start, end, source_type):
            del start, end, source_type
            data = Path(path).read_bytes()
            rastered.append((path, data))
            if data != original:
                raise RuntimeError("refusing durable snapshot / trap bytes")
            yield 1, Image.new("RGB", (8, 8), "white")

        try:
            selected = pipe._select_salvage_source(doc_id, str(fixture.canonical))
            assert selected is not None
            assert selected.origin == "live_authority"
            assert selected.runtime_path == fixture.authority.runtime_path
            assert selected.runtime_path != str(fixture.canonical)
            assert Path(selected.runtime_path).read_bytes() == original

            _images, messages, failures = _run_extractor_chunk(
                pipe,
                monkeypatch,
                doc_id=doc_id,
                queued_path=str(fixture.canonical),
                registry=registry,
                iter_source_images=fake_iter,
            )
            assert rastered[0][0] == str(fixture.canonical)
            assert rastered[0][1] == trap
            assert rastered[1][0] == fixture.authority.runtime_path
            assert rastered[1][1] == original
            assert str(fixture.canonical) not in [path for path, _data in rastered[1:]]
            assert [msg["status"] for msg in messages] == [pipe.PageStatus.IMAGE_ONLY]
            assert failures[0][2].startswith("EXTRACT_SALVAGED_IMAGE_ONLY:")
            assert (
                fixture.authority.verify(fixture.fingerprint)["content_sha256"]
                == fixture.fingerprint["content_sha256"]
            )
        finally:
            fixture.authority.close()

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="Linux proc-fd vs durable snapshot anti-pattern",
    )
    def test_linux_other_pid_proc_fd_without_fingerprint_does_not_salvage_disk(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Rewrite of overlay test_missing_registry_salvages_from_durable_snapshot.

        Overlay treated ``/proc/1/fd/999`` as dead and admitted the disk spool
        with no digest. Missing fingerprint must stay EXTRACT_FAILED.
        """
        from PIL import Image

        doc_id = "doc-durable-snapshot"
        temp_root = tmp_path
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_root))
        snapshot_dir = temp_root / doc_id / pipe.SOURCE_SNAPSHOT_DIRNAME
        snapshot_dir.mkdir(parents=True)
        snapshot = snapshot_dir / "source.pdf"
        snapshot.write_bytes(b"%PDF-1.4\n%%EOF\n")
        rastered = []
        salvage_kwargs = []

        def fake_iter(path, start, end, source_type):
            del start, end, source_type
            rastered.append(path)
            raise RuntimeError("dead other-pid fd")

        def fake_fitz(path, start, end):
            del start, end
            yield 1, Image.new("RGB", (8, 8), "white")

        real_salvage = pipe._salvage_extractor_image_only

        def wrapped_salvage(**kwargs):
            salvage_kwargs.append(kwargs)
            return real_salvage(**kwargs)

        monkeypatch.setattr(pipe, "_salvage_extractor_image_only", wrapped_salvage)
        _images, messages, failures = _run_extractor_chunk(
            pipe,
            monkeypatch,
            doc_id=doc_id,
            queued_path="/proc/1/fd/999",
            registry={},
            iter_source_images=fake_iter,
            iter_fitz_image_frames=fake_fitz,
        )
        assert salvage_kwargs == []
        assert [msg["status"] for msg in messages] == [pipe.PageStatus.EXTRACT_FAILED]
        assert pipe.PageStatus.IMAGE_ONLY not in [msg["status"] for msg in messages]
        assert failures[0][2].startswith("EXTRACT_FAILED:")
        assert str(snapshot) not in rastered

    @pytest.mark.skipif(
        os.name != "nt",
        reason="Windows mapped-snapshot salvage preference",
    )
    def test_windows_queued_mapped_snapshot_preferred_over_original_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Architecture Class 2: Windows salvage prefers queued mapped snapshot.

        Live authority wins when open. With authority closed, a queued durable
        snapshot is ``queued_runtime``; ``DocumentState.path`` is never rastered
        and falls through to verified restart snapshot.
        """
        original = b"%PDF-1.4\nwindows salvage generation\n%%EOF\n"
        doc_id = "doc-win-salvage-pref"
        temp_root = tmp_path / "temp"
        monkeypatch.setattr(pipe, "TEMP_FOLDER", str(temp_root))
        doc_temp = temp_root / doc_id
        snapshot_dir = doc_temp / pipe.SOURCE_SNAPSHOT_DIRNAME
        snapshot_dir.mkdir(parents=True)
        canonical = snapshot_dir / "source.pdf"
        canonical.write_bytes(original)
        fingerprint = _content_fingerprint(original)
        pipe._write_resume_manifest(str(doc_temp), "matter.pdf", fingerprint)
        original_input = tmp_path / "original-input.pdf"
        original_input.write_bytes(original)
        authority = pipe._open_immutable_source_authority(str(canonical), fingerprint)
        doc_state = SimpleNamespace(
            path=str(original_input),
            source_authority=authority,
            source_snapshot_path=str(canonical),
            source_fingerprint=fingerprint,
            temp_dir=str(doc_temp),
        )
        monkeypatch.setattr(pipe, "doc_registry", {doc_id: doc_state})
        monkeypatch.setattr(pipe, "doc_registry_lock", threading.RLock())
        selected = None
        try:
            selected = pipe._select_salvage_source(doc_id, str(canonical))
            assert selected is not None
            assert selected.origin == "live_authority"
            assert os.path.normcase(selected.runtime_path) == os.path.normcase(
                str(canonical)
            )
            assert selected.runtime_path != str(original_input)
        finally:
            authority.close()
            if selected is not None:
                pipe._close_fd_quietly(selected.owned_fd)

        queued = pipe._select_salvage_source(doc_id, str(canonical))
        try:
            assert queued is not None
            assert queued.origin == "queued_runtime"
            assert os.path.normcase(queued.runtime_path) == os.path.normcase(
                str(canonical)
            )
            assert os.path.normcase(queued.runtime_path) != os.path.normcase(
                str(original_input)
            )
        finally:
            if queued is not None:
                pipe._close_fd_quietly(queued.owned_fd)

        from_original = pipe._select_salvage_source(doc_id, str(original_input))
        try:
            assert from_original is not None
            assert from_original.origin == "restart_snapshot"
            assert os.path.normcase(
                os.path.realpath(from_original.runtime_path)
            ) == os.path.normcase(os.path.realpath(str(canonical)))
            assert os.path.normcase(
                os.path.realpath(from_original.runtime_path)
            ) != os.path.normcase(os.path.realpath(str(original_input)))
        finally:
            if from_original is not None:
                pipe._close_fd_quietly(from_original.owned_fd)

    def test_assembler_merges_resumed_and_new_pages(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Assembler correctly merges RESUMED pages (from temp) with newly
        processed pages into a final PDF."""
        _source_dir, _output_dir, _temp_dir, _sf, doc, failures, _fatal = (
            _setup_assembler_case(
                monkeypatch,
                pipe,
                tmp_path,
                source_name="merged.pdf",
                doc_id="doc-merge-test",
                total_pages=4,
            )
        )

        for p in [1, 2]:
            _put_resumed_assembly_page(pipe, doc, p, f"Page {p}")
        for p in [3, 4]:
            _put_live_assembly_page(pipe, doc, p, f"Page {p} text")

        _run_assembler_until_queue_drained(pipe)

        assert doc.doc_id not in pipe.doc_registry
        assert os.path.exists(doc.output_pdf)
        with pipe.fitz.open(doc.output_pdf) as final_pdf:
            assert final_pdf.page_count == 4

        merge_failures = [f for f in failures if "doc-merge-test" in str(f)]
        assert len(merge_failures) == 0


@_skip_no_pipeline
class TestExecutableSignatureRefuse:
    """PE/ELF must not be admitted by OCR classify or captured-header spool."""

    _REFUSE = "Executable binary signature refused (PE/ELF; B01)"

    def test_classify_source_file_refuses_pe_as_pdf(self, tmp_path):
        payload = tmp_path / "evil.pdf"
        payload.write_bytes(b"MZ" + b"\x00" * 64)
        source_type, reason = pipe.classify_source_file(str(payload))
        assert source_type is None
        assert reason == self._REFUSE
        assert "extension fallback" not in reason.lower()

    def test_classify_source_file_refuses_elf_as_pdf(self, tmp_path):
        payload = tmp_path / "evil.pdf"
        payload.write_bytes(b"\x7fELF" + b"\x00" * 64)
        source_type, reason = pipe.classify_source_file(str(payload))
        assert source_type is None
        assert reason == self._REFUSE

    def test_classify_captured_source_refuses_pe_and_elf(self):
        for header in (b"MZ" + b"\x00" * 32, b"\x7fELF" + b"\x00" * 32):
            source_type, reason = pipe._classify_captured_source(
                "payload.pdf",
                header,
            )
            assert source_type is None
            assert reason == self._REFUSE

    @pytest.mark.parametrize(
        ("magic", "doc_id"),
        [
            (b"MZ" + b"\x00" * 64, "doc-pe-refuse"),
            (b"\x7fELF" + b"\x00" * 64, "doc-elf-refuse"),
        ],
    )
    def test_scheduler_refuses_executable_from_captured_header(
        self,
        tmp_path,
        monkeypatch,
        magic,
        doc_id,
    ):
        source_file, _doc_temp, _fp = _setup_scheduler_resume(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="payload.pdf",
            source_bytes=magic,
            doc_id=doc_id,
            total_pages=1,
            existing_pages=[],
        )
        fatal = threading.Event()
        monkeypatch.setattr(pipe, "fatal_pipeline_error_event", fatal)
        observed = []
        real_classify = pipe._classify_captured_source

        def classify_capture(path, header, *args, **kwargs):
            observed.append((path, bytes(header)))
            return real_classify(path, header, *args, **kwargs)

        monkeypatch.setattr(pipe, "_classify_captured_source", classify_capture)
        pipe.scheduler_thread()

        assert len(observed) == 1
        assert observed[0][0] == str(source_file)
        assert observed[0][1].startswith(magic[:4])
        assert doc_id not in pipe.doc_registry
        assert pipe.chunk_queue.empty()
        assert fatal.is_set()
        leftovers = list(Path(pipe.TEMP_FOLDER).glob(".source-snapshot-*"))
        assert leftovers == []


# ===========================================================================
# Scenario 2: Worker OOM/error simulation
# ===========================================================================


@_skip_no_pipeline
class TestWorkerErrorSimulation:
    """A worker failure without a durable page checkpoint vetoes publication."""

    def test_critical_failed_page_without_checkpoint_blocks_document(
        self,
        tmp_path,
        monkeypatch,
    ):
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="oom.pdf",
            doc_id="doc-oom",
            total_pages=3,
            capture_fatal=True,
        )

        _put_live_assembly_page(pipe, doc, 1, "OK Page 1")
        _put_terminal_status(pipe, doc.doc_id, 2, "CRITICAL_FAILED")
        _put_live_assembly_page(pipe, doc, 3, "OK Page 3")

        _run_assembler_until_queue_drained(pipe)

        assert not doc.finalized
        assert not os.path.exists(doc.output_pdf)
        _assert_assembly_fatal_no_output(
            pipe,
            doc,
            failures,
            fatal,
            err_token="DOCUMENT_CHECKPOINTS_UNAVAILABLE",
        )

    def test_extract_failed_page_without_checkpoint_blocks_document(
        self,
        tmp_path,
        monkeypatch,
    ):
        """EXTRACT_FAILED without evidence cannot produce a partial PDF."""
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="extract_fail.pdf",
            doc_id="doc-extfail",
            total_pages=2,
            capture_fatal=True,
        )

        _put_terminal_status(pipe, doc.doc_id, 1, "EXTRACT_FAILED")
        _put_live_assembly_page(pipe, doc, 2, "Good Page 2")

        _run_assembler_until_queue_drained(pipe)

        assert not doc.finalized
        assert not os.path.exists(doc.output_pdf)
        _assert_assembly_fatal_no_output(
            pipe,
            doc,
            failures,
            fatal,
            err_token="DOCUMENT_CHECKPOINTS_UNAVAILABLE",
        )


# ===========================================================================
# Scenario 3: Coordinator restart simulation
# ===========================================================================


@_skip_no_pipeline
class TestCoordinatorRestartSimulation:
    """Simulate a coordinator restart by pre-populating doc_registry
    with partial state, then re-running the scheduler to verify it
    picks up processing from where it left off."""

    def test_scheduler_resumes_from_existing_temp_state(
        self,
        tmp_path,
        monkeypatch,
    ):
        """After restart, scheduler finds existing pages and correctly
        identifies only missing pages for re-processing."""
        _setup_scheduler_resume(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="restart.pdf",
            source_bytes=b"%PDF-1.4\nRestart test\n%%EOF\n",
            doc_id="doc-restart",
            total_pages=5,
            existing_pages=[1, 2, 3],
            page_text=lambda p: f"Recovered page {p}",
        )
        pipe.scheduler_thread()

        resumed = _drain_resumed_messages(pipe)
        assert sorted(msg["page_num"] for msg in resumed) == [1, 2, 3]
        queued_pages, _ = _drain_queued_pages(pipe)
        assert queued_pages == {4, 5}

    def test_full_resume_all_pages_exist(self, tmp_path, monkeypatch):
        """When all pages already exist in temp dir, scheduler sends all
        RESUMED messages and queues no extraction tasks."""
        total_pages = 3
        _setup_scheduler_resume(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="complete.pdf",
            source_bytes=b"%PDF-1.4\nComplete doc\n%%EOF\n",
            doc_id="doc-all-exist",
            total_pages=total_pages,
            existing_pages=range(1, total_pages + 1),
        )
        pipe.scheduler_thread()

        resumed = _drain_resumed_messages(pipe)
        assert sorted(msg["page_num"] for msg in resumed) == [1, 2, 3]
        assert pipe.chunk_queue.empty()


# ===========================================================================
# Scenario 4: Temp dir corruption
# ===========================================================================


@_skip_no_pipeline
class TestTempDirCorruption:
    """Verify the pipeline handles corrupt/zero-byte page PDFs in temp dir."""

    def test_page_checkpoint_validator_rejects_partial_and_accepts_complete(
        self,
        tmp_path,
    ):
        valid = tmp_path / "valid.pdf"
        partial = tmp_path / "partial.pdf"
        _create_chunk_pdf(pipe, valid, "Complete checkpoint")
        partial.write_bytes(valid.read_bytes()[:64])

        assert pipe._is_valid_page_checkpoint(str(valid)) is True
        assert pipe._is_valid_page_checkpoint(str(partial)) is False

    def test_checkpoint_metadata_round_trip_and_mismatch_rejection(
        self,
        tmp_path,
        monkeypatch,
    ):
        checkpoint = tmp_path / "1.pdf"
        _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id="doc-metadata",
            page_num=1,
            text="Durable OCR text",
            ocr_language="fr",
            ocr_confidence=0.97,
        )

        metadata = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-metadata",
            page_num=1,
        )
        extracted = pipe._extract_text_from_chunk_pdf(str(checkpoint), strict=True)
        assert metadata == {
            "schema_version": pipe.CHECKPOINT_METADATA_SCHEMA_VERSION,
            "document_id": "doc-metadata",
            "page_num": 1,
            "status": "Paddle",
            "ocr_method": "PaddleOCR",
            "ocr_language": "fr",
            "detected_language": "",
            "detected_language_confidence": None,
            "source_prior_label": "",
            "model_language": "fr",
            "ocr_confidence": pytest.approx(0.97),
            "text_length": len(extracted),
            "has_text": True,
            "text_sha256": pipe._canonical_text_sha256(extracted),
            "validation_status": "ok",
            "language_analysis_complete": True,
            "per_span_language_enabled": bool(
                pipe.ENABLE_PER_SPAN_LANGUAGE and pipe._LANGUAGE_DETECTION_AVAILABLE
            ),
            "language_data": None,
            "processing_profile": _default_processing_profile(pipe),
            "page_geometry": metadata["page_geometry"],
            "geometry_issues": [],
            "review_reasons": [],
            "candidate_reviews": [],
            "coverage_assessment": {},
            "needs_review": False,
        }
        assert metadata["page_geometry"]["source_kind"] == "pdf"

        with pytest.raises(ValueError, match="document mismatch"):
            pipe._load_page_checkpoint_metadata(
                str(checkpoint),
                document_id="different-document",
                page_num=1,
            )
        with pytest.raises(ValueError, match="page mismatch"):
            pipe._load_page_checkpoint_metadata(
                str(checkpoint),
                document_id="doc-metadata",
                page_num=2,
            )

        fallback = tmp_path / "2.pdf"
        _create_checkpoint_pdf(
            pipe,
            fallback,
            document_id="doc-metadata",
            page_num=2,
            text="Fallback OCR text",
            status=pipe.PageStatus.TESSERACT,
            ocr_method=pipe.PageStatus.TESSERACT,
            ocr_language="en",
            ocr_confidence=0.72,
            execution_engine="Tesseract",
        )
        fallback_metadata = pipe._load_page_checkpoint_metadata(
            str(fallback),
            document_id="doc-metadata",
            page_num=2,
        )
        assert fallback_metadata["ocr_method"] == "Tesseract"
        assert fallback_metadata["validation_status"] == "fallback"

        monkeypatch.setattr(pipe, "ENABLE_PER_SPAN_LANGUAGE", True)
        monkeypatch.setattr(pipe, "_LANGUAGE_DETECTION_AVAILABLE", True)
        monkeypatch.setattr(pipe, "LANGUAGE_INCLUDE_SPANS", False)
        assert "without per-span" in pipe._checkpoint_resume_incompatibility(metadata)

    def test_legacy_provenance_free_checkpoint_always_rejected(
        self,
        tmp_path,
        monkeypatch,
    ):
        legacy = tmp_path / "legacy.pdf"
        _create_chunk_pdf(pipe, legacy, "Legacy unmarked checkpoint")
        meta = pipe._load_page_checkpoint_metadata(
            str(legacy),
            document_id="doc-legacy",
            page_num=1,
        )
        assert meta is None
        # Rejection is independent of common optional feature flags.
        monkeypatch.setattr(pipe, "ENABLE_PER_SPAN_LANGUAGE", False)
        monkeypatch.setattr(pipe, "ENABLE_DOCUMENT_INTELLIGENCE", False)
        monkeypatch.setattr(pipe, "ENABLE_HANDWRITING", False)
        monkeypatch.setattr(pipe, "ENABLE_PAGE_ROUTING", False)
        monkeypatch.setattr(pipe, "ENABLE_GPU_OPTIMIZATION", False)
        reason = pipe._checkpoint_resume_incompatibility(meta)
        assert reason is not None
        assert "legacy" in reason.lower() or "provenance" in reason.lower()

    def test_checkpoint_profile_material_field_mismatches_and_match(
        self,
        tmp_path,
        monkeypatch,
    ):
        checkpoint = tmp_path / "1.pdf"
        _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id="doc-profile",
            page_num=1,
            text="Profile gate text",
        )
        metadata = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-profile",
            page_num=1,
        )
        assert pipe._checkpoint_resume_incompatibility(metadata) is None

        # Each new material field must fail closed when missing or wrong.
        material_globals = [
            "page_routing_enabled",
            "gpu_optimization_enabled",
            "page_cache_enabled",
            "adaptive_batch_enabled",
            "paddle_device",
            "inference_backend",
            "ocr_engine_selection",
        ]
        for field in material_globals:
            broken = dict(metadata)
            broken_profile = dict(metadata["processing_profile"])
            if field.endswith("_enabled"):
                broken_profile[field] = not broken_profile[field]
            else:
                broken_profile[field] = f"mismatch-{field}"
            broken["processing_profile"] = broken_profile
            reason = pipe._checkpoint_resume_incompatibility(broken)
            assert reason is not None, field
            assert field in reason or "differs" in reason

        for field in ("page_cache_enabled", "adaptive_batch_enabled"):
            broken = dict(metadata)
            broken_profile = dict(metadata["processing_profile"])
            broken_profile.pop(field)
            broken["processing_profile"] = broken_profile
            reason = pipe._checkpoint_resume_incompatibility(broken)
            assert reason is not None
            assert field in reason

        page_local = [
            "actual_render_dpi",
            "page_routing_target",
            "execution_device",
            "execution_engine",
            "gpu_optimization_applied",
            "preprocessing_applied",
        ]
        for field in page_local:
            missing = dict(metadata)
            missing_profile = dict(metadata["processing_profile"])
            missing_profile.pop(field, None)
            missing["processing_profile"] = missing_profile
            reason = pipe._checkpoint_resume_incompatibility(missing)
            assert reason is not None, field
            assert field in reason

        for field, bad_value, match in (
            ("actual_render_dpi", 0, "actual_render_dpi"),
            ("execution_device", "tpu", "execution_device"),
            ("execution_engine", "UnknownEngine", "execution_engine"),
            ("gpu_optimization_applied", "yes", "gpu_optimization_applied"),
        ):
            bad = dict(metadata)
            bad_profile = dict(metadata["processing_profile"])
            bad_profile[field] = bad_value
            bad["processing_profile"] = bad_profile
            assert match in pipe._checkpoint_resume_incompatibility(bad)

        # Matching full profile (including page-local actuals) remains reusable.
        matching = dict(metadata)
        mp = metadata["processing_profile"]
        matching["processing_profile"] = _default_processing_profile(
            pipe,
            actual_render_dpi=mp["actual_render_dpi"],
            page_routing_target=mp["page_routing_target"],
            execution_device=mp["execution_device"],
            execution_engine=mp["execution_engine"],
            gpu_optimization_applied=mp["gpu_optimization_applied"],
            preprocessing_applied=mp["preprocessing_applied"],
        )
        assert pipe._checkpoint_resume_incompatibility(matching) is None

    def test_canonical_text_exact_equality_live_vs_resume_source(
        self,
        tmp_path,
    ):
        """Live authority must equal text re-extracted from the same checkpoint."""
        checkpoint = tmp_path / "1.pdf"
        ocr_accumulator = "Exact authority text\nwith trailing space "
        _create_chunk_pdf(pipe, checkpoint, ocr_accumulator.strip())
        raw_bytes = checkpoint.read_bytes()
        canonical = pipe._canonical_text_after_checkpoint_pdf(
            raw_bytes,
            ocr_text=ocr_accumulator,
        )
        checkpoint.write_bytes(
            pipe._embed_page_checkpoint_metadata(
                raw_bytes,
                document_id="doc-text",
                page_num=1,
                status=pipe.PageStatus.PADDLE,
                ocr_method="PaddleOCR",
                ocr_language="en",
                ocr_confidence=0.99,
                text_length=len(canonical),
                has_text=bool(canonical),
                processing_profile=_default_processing_profile(pipe),
                canonical_text=canonical,
            ),
        )
        resumed = pipe._extract_text_from_chunk_pdf(str(checkpoint), strict=True)
        meta = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-text",
            page_num=1,
        )
        assert resumed == canonical
        assert meta["text_length"] == len(canonical)
        assert meta["text_length"] == len(resumed)
        # Exact equality, not substring-only.
        assert resumed == "Exact authority text\nwith trailing space".strip() or (
            resumed == canonical
        )
        assert "Exact authority text" in resumed  # sanity
        assert resumed == canonical  # hard equality gate

    def test_canonical_text_fails_closed_when_layer_missing(self, tmp_path):
        # Image-only style PDF with no text layer + non-empty OCR accumulator.
        with pipe.fitz.open() as pdf:
            pdf.new_page()
            raw = pdf.tobytes()
        try:
            pipe._canonical_text_after_checkpoint_pdf(
                raw,
                ocr_text="accumulator still has text",
            )
        except RuntimeError as exc:
            assert "empty" in str(exc).lower() or "failed" in str(exc).lower()
        else:
            raise AssertionError("expected fail-closed canonical extraction")

    def test_compact_language_recovery_preserves_document_weighting(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(pipe, "ENABLE_PER_SPAN_LANGUAGE", True)
        monkeypatch.setattr(pipe, "_LANGUAGE_DETECTION_AVAILABLE", True)
        monkeypatch.setattr(pipe, "LANGUAGE_INCLUDE_SPANS", False)
        pages = [
            _make_page_language(
                pipe,
                page_num=1,
                language="en",
                confidence=0.95,
                char_count=10,
            ),
            _make_page_language(
                pipe,
                page_num=2,
                language="fr",
                confidence=0.88,
                char_count=1000,
            ),
        ]
        recovered = []
        for page_language in pages:
            checkpoint = tmp_path / f"{page_language.page_num}.pdf"
            _create_checkpoint_pdf(
                pipe,
                checkpoint,
                document_id="doc-language",
                page_num=page_language.page_num,
                text=f"Page {page_language.page_num}",
                ocr_language=page_language.primary_language,
                ocr_confidence=page_language.primary_confidence,
                language_data=page_language,
            )
            metadata = pipe._load_page_checkpoint_metadata(
                str(checkpoint),
                document_id="doc-language",
                page_num=page_language.page_num,
            )
            recovered.append(pipe._restore_checkpoint_page_language(metadata))

        original_doc = pipe.finalize_document_language(
            "doc-language", "doc.pdf", pages, "", "", "1.0"
        )
        recovered_doc = pipe.finalize_document_language(
            "doc-language", "doc.pdf", recovered, "", "", "1.0"
        )
        assert recovered_doc.primary_language == original_doc.primary_language == "fr"
        assert recovered_doc.language_char_shares == original_doc.language_char_shares
        assert recovered_doc.primary_confidence == original_doc.primary_confidence

    def test_checkpoint_with_incomplete_language_metadata_fails_closed(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(pipe, "ENABLE_PER_SPAN_LANGUAGE", True)
        checkpoint = tmp_path / "1.pdf"
        page_language = _make_page_language(
            pipe,
            page_num=1,
            language="en",
            confidence=0.95,
            char_count=12,
        )
        _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id="doc-malformed-language",
            page_num=1,
            text="Language checkpoint",
            ocr_language="en",
            ocr_confidence=0.95,
            language_data=page_language,
        )

        with pipe.fitz.open(str(checkpoint)) as checkpoint_doc:
            metadata = checkpoint_doc.metadata
            subject = metadata["subject"]
            payload = json.loads(
                subject[len(pipe.CHECKPOINT_METADATA_SUBJECT_PREFIX) :],
            )
            del payload["language_data"]["scripts_detected"]
            metadata["subject"] = pipe.CHECKPOINT_METADATA_SUBJECT_PREFIX + json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            )
            checkpoint_doc.set_metadata(metadata)
            malformed_bytes = checkpoint_doc.tobytes()
        checkpoint.write_bytes(malformed_bytes)

        with pytest.raises(ValueError, match="detected scripts"):
            pipe._load_page_checkpoint_metadata(
                str(checkpoint),
                document_id="doc-malformed-language",
                page_num=1,
            )

    @pytest.mark.parametrize(
        ("source_name", "chunk_bytes", "doc_id"),
        [
            ("corrupt.pdf", b"", "doc-corrupt"),
            ("garbage.pdf", b"THIS IS NOT A PDF FILE AT ALL", "doc-garbage"),
        ],
        ids=[
            "assembler_handles_zero_byte_chunk",
            "assembler_handles_invalid_pdf_content",
        ],
    )
    def test_assembler_handles_corrupt_present_chunk(
        self,
        tmp_path,
        monkeypatch,
        source_name,
        chunk_bytes,
        doc_id,
    ):
        """A present zero-byte or garbage chunk makes finalization fatal."""
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name=source_name,
            doc_id=doc_id,
            total_pages=2,
            capture_fatal=True,
        )

        corrupt_path = os.path.join(doc.temp_dir, "1.pdf")
        with open(corrupt_path, "wb") as f:
            f.write(chunk_bytes)

        _put_live_assembly_page(pipe, doc, 2, "Good page")
        pipe.assembly_queue.put(
            {
                "doc_id": doc_id,
                "page_num": 1,
                "text": "stale",
                "status": "Paddle",
                "chunk_path": corrupt_path,
                "checkpoint_sha256": _checkpoint_sha256_for_path(corrupt_path),
                "cache_hit": False,
            }
        )

        _run_assembler_until_queue_drained(pipe)

        # Strict admission rejects unreadable bytes before terminal state is
        # mutated, leaving the document eligible for a later valid retry.
        assert doc_id in pipe.doc_registry
        assert doc.processed_pages == 1
        assert 1 not in doc.terminal_pages
        _assert_assembly_fatal_no_output(
            pipe,
            doc,
            failures,
            fatal,
            err_token="CHECKPOINT_ADMISSION_FAILED",
        )

    def test_resume_invalidation_clears_corrupt_chunks(self, tmp_path):
        """When source fingerprint changes, even corrupt chunks are removed."""
        temp_dir = tmp_path / "resume_corrupt"
        temp_dir.mkdir()

        source_path = "/app/ocr_source/test.pdf"
        fp_old = {"content_sha256": "a" * 64, "size_bytes": 100, "mtime_ns": 1}
        fp_new = {"content_sha256": "b" * 64, "size_bytes": 200, "mtime_ns": 2}

        (temp_dir / "1.pdf").write_bytes(b"")  # zero-byte
        (temp_dir / "2.pdf").write_bytes(b"garbage")  # not a PDF
        (temp_dir / "3.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")  # valid-ish

        pipe._write_resume_manifest(str(temp_dir), source_path, fp_old)
        result = pipe.prepare_resume_state(str(temp_dir), source_path, fp_new)

        assert result["status"] == "invalidated"
        assert result["removed_entries"] == 3
        remaining = [
            f for f in os.listdir(str(temp_dir)) if f != pipe.RESUME_MANIFEST_FILENAME
        ]
        assert len(remaining) == 0


# ===========================================================================
# Scenario 5: Concurrent crash/resume
# ===========================================================================


@_skip_no_pipeline
class TestConcurrentCrashResume:
    """A fatal incomplete page vetoes every pending publication in the run."""

    def test_one_doc_missing_checkpoint_vetoes_all_publication(
        self,
        tmp_path,
        monkeypatch,
    ):
        source_dir, output_dir, temp_dir = _monkeypatch_pipeline_dirs(
            monkeypatch,
            pipe,
            tmp_path,
        )
        _monkeypatch_feature_flags_off(monkeypatch, pipe)
        failures = []
        monkeypatch.setattr(
            pipe,
            "log_failure",
            lambda path, page, error: failures.append((path, page, error)),
        )

        source_a = source_dir / "doc_a.pdf"
        source_a.write_bytes(b"%PDF-1.4\n%%EOF\n")
        doc_a = _make_doc_state(pipe, source_a, "doc-a")
        doc_a.total_pages = 2
        os.makedirs(doc_a.temp_dir, exist_ok=True)

        source_b = source_dir / "doc_b.pdf"
        source_b.write_bytes(b"%PDF-1.4\n%%EOF\n")
        doc_b = _make_doc_state(pipe, source_b, "doc-b")
        doc_b.total_pages = 2
        os.makedirs(doc_b.temp_dir, exist_ok=True)

        # Interleave messages from both documents
        _put_live_assembly_page(pipe, doc_a, 1, "A page 1")
        _put_live_assembly_page(pipe, doc_b, 1, "B page 1")
        _put_live_assembly_page(pipe, doc_a, 2, "A page 2")
        _put_terminal_status(pipe, "doc-b", 2, "CRITICAL_FAILED")

        _run_assembler_until_queue_drained(pipe)

        assert doc_a.finalized
        assert not doc_b.finalized
        assert pipe.fatal_pipeline_error_event.is_set()
        assert not pipe.compression_queue.empty()
        assert os.path.exists(doc_a.output_pdf)
        assert not os.path.exists(doc_b.output_pdf)
        assert any(
            "DOCUMENT_CHECKPOINTS_UNAVAILABLE" in error
            for _path, _page, error in failures
        )


# ===========================================================================
# Scenario 6: Resume gap logic (non-contiguous existing pages)
# ===========================================================================


@_skip_no_pipeline
class TestResumeGapLogic:
    """Verify the scheduler correctly handles non-contiguous gaps in
    existing pages (e.g., pages 1, 3, 5 exist but 2, 4 do not)."""

    def test_non_contiguous_gaps_queued_correctly(self, tmp_path, monkeypatch):
        _setup_scheduler_resume(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="gaps.pdf",
            source_bytes=b"%PDF-1.4\nGap test\n%%EOF\n",
            doc_id="doc-gaps",
            total_pages=5,
            existing_pages=[1, 3, 5],
        )
        pipe.scheduler_thread()

        resumed = _drain_resumed_messages(pipe)
        assert sorted(msg["page_num"] for msg in resumed) == [1, 3, 5]

        queued_pages, chunk_count = _drain_queued_pages(pipe)
        assert queued_pages == {2, 4}
        # With non-contiguous gaps, each gap page is flushed separately
        # because a RESUMED page breaks continuity
        assert chunk_count == 2

    def test_single_missing_page_in_middle(self, tmp_path, monkeypatch):
        """Single missing page in the middle of a document."""
        _setup_scheduler_resume(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="single_gap.pdf",
            source_bytes=b"%PDF-1.4\n%%EOF\n",
            doc_id="doc-single-gap",
            total_pages=5,
            existing_pages=[1, 2, 4, 5],
        )
        pipe.scheduler_thread()

        resumed = _drain_resumed_messages(pipe)
        assert sorted(msg["page_num"] for msg in resumed) == [1, 2, 4, 5]
        queued_pages, _ = _drain_queued_pages(pipe)
        assert queued_pages == {3}


# ===========================================================================
# Scenario 7: Assembler missing chunk detection
# ===========================================================================


@_skip_no_pipeline
class TestAssemblerMissingChunkDetection:
    """Verify that the assembler correctly identifies and logs missing
    chunks in various scenarios."""

    def test_missing_chunk_for_non_terminal_page_logs_crash(
        self,
        tmp_path,
        monkeypatch,
    ):
        """When a page is terminal-acknowledged but its chunk file is missing
        (and status is not EXTRACT/CRITICAL_FAILED), log it."""
        _sd, _od, _td, _sf, doc, failures, _fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="missing.pdf",
            doc_id="doc-missing-chunk",
            total_pages=1,
        )

        # Create a valid live checkpoint, admit its digest, then lose the file
        # before finalization (simulates a post-queue lost write).
        extracted = _create_checkpoint_pdf(
            pipe,
            os.path.join(doc.temp_dir, "1.pdf"),
            document_id=doc.doc_id,
            page_num=1,
            text="Some text",
        )
        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        digest = _checkpoint_sha256_for_path(chunk_path)
        os.remove(chunk_path)

        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "text": extracted,
                "status": "Paddle",
                "chunk_path": chunk_path,
                "checkpoint_sha256": digest,
                "ocr_method": "PaddleOCR",
                "ocr_language": "en",
                "ocr_confidence": 0.9,
                "cache_hit": False,
            }
        )

        _run_assembler_until_queue_drained(pipe)

        assert doc.doc_id in pipe.doc_registry
        assert doc.processed_pages == 0
        assert any(
            f[1] == 1 and "CHECKPOINT_ADMISSION_FAILED" in str(f[2]) for f in failures
        )

    def test_all_chunks_missing_blocks_placeholder_publication(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Lossless assembly never substitutes a placeholder for source pages."""
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="all_missing.pdf",
            doc_id="doc-all-missing",
            total_pages=2,
            capture_fatal=True,
        )

        for p in [1, 2]:
            _put_terminal_status(pipe, doc.doc_id, p, "CRITICAL_FAILED")

        _run_assembler_until_queue_drained(pipe)

        assert not doc.finalized
        assert not os.path.exists(doc.output_pdf)
        _assert_assembly_fatal_no_output(
            pipe,
            doc,
            failures,
            fatal,
            err_token="DOCUMENT_CHECKPOINTS_UNAVAILABLE",
        )


# ===========================================================================
# Scenario 8: Resume text recovery from chunk PDF
# ===========================================================================


@_skip_no_pipeline
class TestResumedTextRecovery:
    """Reuse messages require content-bound provenance; no non-strict recovery."""

    def test_resumed_without_provenance_fails_admission_and_finalization(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Provenance-free RESUMED must not recover text or ship outputs."""
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="recover.pdf",
            doc_id="doc-recover-text",
            total_pages=1,
            capture_fatal=True,
        )

        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        _create_chunk_pdf(pipe, chunk_path, "Recovered OCR text content")

        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "text": "",  # Empty without resume_metadata — admission fails
                "status": "RESUMED",
                "chunk_path": chunk_path,
                "checkpoint_sha256": _checkpoint_sha256_for_path(chunk_path),
            }
        )

        _run_assembler_until_queue_drained(pipe)

        assert fatal.is_set()
        assert any("REUSE_ADMISSION_FAILED" in err for _p, _pg, err in failures)
        # Admission failure must not mark the page terminal or finalize.
        assert doc.doc_id in pipe.doc_registry
        assert 1 not in doc.terminal_pages
        assert doc.processed_pages == 0
        assert not doc.finalized
        assert (
            not os.path.exists(doc.output_pdf)
            or open(
                doc.output_pdf,
                "rb",
            ).read()
        )  # may exist as stale; must not be successful deliverable
        txt_dir = doc.output_txt_dir
        if os.path.isdir(txt_dir):
            txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
            assert txt_files == []
        assert pipe.compression_queue.empty()

    def test_cache_hit_without_provenance_fails_admission(self, tmp_path, monkeypatch):
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="cached.pdf",
            doc_id="doc-cached-text",
            total_pages=1,
            capture_fatal=True,
        )

        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        _create_chunk_pdf(pipe, chunk_path, "Cached OCR text content")

        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "text": "",
                "status": "Paddle",
                "chunk_path": chunk_path,
                "checkpoint_sha256": _checkpoint_sha256_for_path(chunk_path),
                "ocr_method": "PaddleOCR",
                "ocr_language": "en",
                "ocr_confidence": 0.93,
                "cache_hit": True,
            }
        )
        _run_assembler_until_queue_drained(pipe)

        assert fatal.is_set()
        assert any("REUSE_ADMISSION_FAILED" in err for _p, _pg, err in failures)
        assert doc.doc_id in pipe.doc_registry
        assert 1 not in doc.terminal_pages
        assert doc.processed_pages == 0
        assert not doc.finalized
        assert pipe.compression_queue.empty()
        if os.path.isdir(doc.output_txt_dir):
            assert [
                f for f in os.listdir(doc.output_txt_dir) if f.endswith(".txt")
            ] == []


# ===========================================================================
# Scenario 9: Duplicate terminal messages
# ===========================================================================


@_skip_no_pipeline
class TestDuplicateTerminalMessages:
    """Verify that duplicate terminal messages for the same page are
    handled gracefully without double-counting."""

    def test_duplicate_resumed_does_not_double_count(
        self,
        tmp_path,
        monkeypatch,
    ):
        _sd, _od, _td, _sf, doc, _failures, _fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="dup.pdf",
            doc_id="doc-dup",
            total_pages=1,
        )
        # Override log_failure to no-op (setup already captures; ignore noise).
        monkeypatch.setattr(pipe, "log_failure", lambda *_a, **_k: None)

        extracted, meta = _put_resumed_assembly_page(pipe, doc, 1, "Dup test")
        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        digest = _checkpoint_sha256_for_path(chunk_path)
        for _ in range(2):
            pipe.assembly_queue.put(
                {
                    "doc_id": doc.doc_id,
                    "page_num": 1,
                    "text": extracted,
                    "status": "RESUMED",
                    "chunk_path": chunk_path,
                    "checkpoint_sha256": digest,
                    "ocr_confidence": meta["ocr_confidence"],
                    "ocr_method": meta["ocr_method"],
                    "ocr_language": meta["ocr_language"],
                    "resume_metadata": meta,
                    "cache_hit": False,
                }
            )

        _run_assembler_until_done(pipe, doc.doc_id)

        assert doc.doc_id not in pipe.doc_registry
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

        for p in range(1, 4):
            _create_chunk_pdf(pipe, doc_temp / f"{p}.pdf", f"Old page {p}")
        pipe._write_resume_manifest(str(doc_temp), str(source_file), fp_original)

        source_file.write_bytes(b"%PDF-1.4\nModified content\n%%EOF\n")
        fp_modified = pipe.compute_source_fingerprint(str(source_file))
        assert fp_original["content_sha256"] != fp_modified["content_sha256"]

        result = pipe.prepare_resume_state(
            str(doc_temp),
            str(source_file),
            fp_modified,
        )

        assert result["status"] == "invalidated"
        assert result["removed_entries"] == 3
        remaining = [
            f for f in os.listdir(str(doc_temp)) if f != pipe.RESUME_MANIFEST_FILENAME
        ]
        assert len(remaining) == 0

        manifest = json.loads(
            (doc_temp / pipe.RESUME_MANIFEST_FILENAME).read_text(encoding="utf-8"),
        )
        assert (
            manifest["source_fingerprint"]["content_sha256"]
            == fp_modified["content_sha256"]
        )

    def test_source_change_during_processing_refuses_final_assembly(
        self,
        tmp_path,
        monkeypatch,
    ):
        _sd, _od, _td, source_file, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="mutable.pdf",
            doc_id="doc-source-mutated",
            total_pages=1,
            capture_fatal=True,
            source_bytes=b"%PDF-original-%%EOF",
        )
        doc.source_fingerprint = pipe.compute_source_fingerprint(str(source_file))
        _put_live_assembly_page(pipe, doc, 1, "Original page")
        source_file.write_bytes(b"%PDF-mutated!-%%EOF")
        _run_assembler_until_done(pipe, doc.doc_id)

        assert fatal.is_set()
        assert not os.path.exists(doc.output_pdf)
        assert pipe.compression_queue.empty()
        assert failures and "SOURCE_INTEGRITY_FAILED" in failures[0][2]

    def test_unreadable_source_never_rescues_stale_output(
        self,
        tmp_path,
        monkeypatch,
    ):
        _sd, _od, _td, source_file, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="deleted.pdf",
            doc_id="doc-source-deleted",
            total_pages=1,
            capture_fatal=True,
            source_bytes=b"%PDF-original-%%EOF",
        )
        doc.source_fingerprint = pipe.compute_source_fingerprint(str(source_file))

        class BrokenCustody:
            def append_event(self, *_args, **_kwargs):
                raise OSError("custody store unavailable")

        # A custody failure inside the source-integrity guard must not fall
        # through to the assembly rescue path and enqueue an older output.
        doc.custody_chain = BrokenCustody()
        stale_bytes = b"%PDF-stale-output"
        with open(doc.output_pdf, "wb") as stale_output:
            stale_output.write(stale_bytes)
        _put_live_assembly_page(pipe, doc, 1, "Original page")
        source_file.unlink()
        _run_assembler_until_done(pipe, doc.doc_id)

        assert fatal.is_set()
        assert open(doc.output_pdf, "rb").read() == stale_bytes
        assert pipe.compression_queue.empty()
        assert failures and "SOURCE_INTEGRITY_FAILED" in failures[0][2]

    def test_final_pdf_save_failure_sets_fatal_and_skips_stale_output(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Save failure must fail the pipeline and never enqueue older bytes."""
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="save-fail.pdf",
            doc_id="doc-save-fail",
            total_pages=1,
            capture_fatal=True,
            source_bytes=b"%PDF-original-%%EOF",
        )
        stale_bytes = b"%PDF-stale-preexisting-output"
        with open(doc.output_pdf, "wb") as stale_output:
            stale_output.write(stale_bytes)

        # Build the V3 checkpoint before patching Document.save (chunk creation
        # also uses save).
        _put_live_assembly_page(pipe, doc, 1, "Save fail page")

        def _fail_save(self, *args, **kwargs):
            raise OSError("injected final_pdf.save failure")

        monkeypatch.setattr(pipe.fitz.Document, "save", _fail_save)
        _run_assembler_until_done(pipe, doc.doc_id)

        assert fatal.is_set()
        assert doc.doc_id not in pipe.doc_registry
        assert open(doc.output_pdf, "rb").read() == stale_bytes
        assert pipe.compression_queue.empty()
        assert failures and "ASSEMBLY_FAILED" in failures[0][2]
        assert "injected final_pdf.save failure" in failures[0][2]


# ===========================================================================
# Scenario 11: Strict text authority + truthful profile closed sets
# ===========================================================================


@_skip_no_pipeline
class TestStrictCheckpointTextAuthority:
    """Resume/cache accept only exact metadata-bound checkpoint text."""

    def test_validated_checkpoint_text_accepts_matching_length_and_presence(
        self,
        tmp_path,
    ):
        checkpoint = tmp_path / "1.pdf"
        extracted = _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id="doc-auth",
            page_num=1,
            text="Exact bound text",
        )
        meta = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-auth",
            page_num=1,
        )
        text = pipe._validated_checkpoint_text(str(checkpoint), meta)
        assert text == extracted
        assert len(text) == meta["text_length"]
        assert bool(text) is meta["has_text"]
        assert meta["text_sha256"] == pipe._canonical_text_sha256(text)
        assert meta["schema_version"] == pipe.CHECKPOINT_METADATA_SCHEMA_VERSION

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("text_length", "plus_one", "length mismatch"),
            ("has_text", False, "presence mismatch"),
        ],
        ids=[
            "validated_checkpoint_text_rejects_length_mismatch",
            "validated_checkpoint_text_rejects_presence_mismatch",
        ],
    )
    def test_validated_checkpoint_text_rejects_field_mismatch(
        self,
        tmp_path,
        field,
        value,
        match,
    ):
        checkpoint = tmp_path / "1.pdf"
        _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id="doc-auth",
            page_num=1,
            text="Exact bound text",
        )
        meta = dict(
            pipe._load_page_checkpoint_metadata(
                str(checkpoint),
                document_id="doc-auth",
                page_num=1,
            )
        )
        if field == "text_length" and value == "plus_one":
            meta["text_length"] = meta["text_length"] + 1
        else:
            meta[field] = value
        with pytest.raises(ValueError, match=match):
            pipe._validated_checkpoint_text(str(checkpoint), meta)

    def test_validated_checkpoint_text_rejects_same_length_digest_mismatch(
        self,
        tmp_path,
    ):
        """AAAA -> BBBB same length must fail the content-bind digest check."""
        checkpoint = tmp_path / "1.pdf"
        _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id="doc-digest",
            page_num=1,
            text="AAAA",
        )
        meta = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-digest",
            page_num=1,
        )
        # Rewrite PDF text layer to BBBB (same length) without updating digest.
        _create_chunk_pdf(pipe, checkpoint, "BBBB")
        _rewrite_checkpoint_subject(pipe, checkpoint, meta)
        with pytest.raises(ValueError, match="digest mismatch"):
            pipe._validated_checkpoint_text(str(checkpoint), meta)

    def test_empty_text_sha256_is_stored_and_validated(self, tmp_path):
        checkpoint = tmp_path / "empty.pdf"
        _create_empty_image_only_checkpoint(
            pipe,
            checkpoint,
            document_id="doc-empty",
            page_num=1,
        )
        meta = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-empty",
            page_num=1,
        )
        assert meta["text_sha256"] == pipe._EMPTY_TEXT_SHA256
        assert pipe._validated_checkpoint_text(str(checkpoint), meta) == ""

    def test_scheduler_discards_length_mismatched_resume_checkpoint(
        self,
        tmp_path,
        monkeypatch,
    ):
        _source_file, doc_temp, _fp = _setup_scheduler_resume(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="report.pdf",
            source_bytes=b"%PDF-1.4\n%%EOF\n",
            doc_id="doc-mismatch-resume",
            total_pages=1,
            existing_pages=[1],
            page_text=lambda _p: "Good page",
        )
        # Corrupt metadata length while leaving PDF text intact.
        meta = pipe._load_page_checkpoint_metadata(
            str(doc_temp / "1.pdf"),
            document_id="doc-mismatch-resume",
            page_num=1,
        )
        broken = dict(meta)
        broken["text_length"] = int(meta["text_length"]) + 5
        _rewrite_checkpoint_subject(pipe, doc_temp / "1.pdf", broken)

        pipe.scheduler_thread()

        assert _drain_resumed_messages(pipe) == []
        assert not (doc_temp / "1.pdf").exists()
        # Page requeued for extraction.
        assert not pipe.chunk_queue.empty()


@_skip_no_pipeline
class TestTruthfulPageProfiles:
    """Closed-set and realized execution provenance for page checkpoints."""

    @pytest.mark.parametrize(
        ("global_name", "profile_name"),
        [
            ("ENABLE_PAGE_CACHE", "page_cache_enabled"),
            ("ENABLE_ADAPTIVE_BATCH", "adaptive_batch_enabled"),
        ],
    )
    def test_cache_and_batch_profile_flags_track_runtime(
        self,
        monkeypatch,
        global_name,
        profile_name,
    ):
        monkeypatch.setattr(pipe, global_name, False)
        assert _default_processing_profile(pipe)[profile_name] is False
        monkeypatch.setattr(pipe, global_name, True)
        assert _default_processing_profile(pipe)[profile_name] is True

    def test_language_provenance_is_actual_digest_or_explicit_unavailable(
        self,
        monkeypatch,
    ):
        digest = "A" * 64
        monkeypatch.setattr(pipe, "lang_model", SimpleNamespace(_sha256=digest))
        actual = pipe._language_model_provenance()
        assert actual["detector_model_sha256"] == digest.lower()
        assert actual["fasttext_model_sha256"] == digest.lower()
        assert actual["tokenizer_sha256"] == "not_applicable:fasttext_internal"

        monkeypatch.setattr(pipe, "lang_model", object())
        unavailable = pipe._language_model_provenance()
        assert unavailable["detector_model_sha256"].startswith("unavailable:")
        assert unavailable["fasttext_model_sha256"].startswith("unavailable:")
        assert all(unavailable.values())

    def test_gpu_optimization_enabled_tracks_batch_preprocessor(self, monkeypatch):
        monkeypatch.setattr(pipe, "ENABLE_GPU_OPTIMIZATION", True)
        monkeypatch.setattr(pipe, "_batch_preprocessor", object())
        profile = _default_processing_profile(
            pipe,
            execution_device="gpu",
            gpu_optimization_applied=True,
        )
        assert profile["gpu_optimization_enabled"] is True
        assert profile["gpu_optimization_applied"] is True
        monkeypatch.setattr(pipe, "_batch_preprocessor", None)
        profile2 = _default_processing_profile(pipe, execution_device="gpu")
        assert profile2["gpu_optimization_enabled"] is False

    def test_page_local_closed_sets_for_realized_engines(self, monkeypatch):
        # applied=True is only coherent when the availability flags are true.
        monkeypatch.setattr(pipe, "ENABLE_GPU_OPTIMIZATION", True)
        monkeypatch.setattr(pipe, "_batch_preprocessor", object())
        for device, engine, applied in (
            ("gpu", "PaddleOCR", True),
            ("cpu", "PaddleOCR", False),
            ("cpu", "Tesseract", False),
            ("none", "ImageOnly", False),
        ):
            profile = _default_processing_profile(
                pipe,
                execution_device=device,
                execution_engine=engine,
                gpu_optimization_applied=applied,
            )
            assert pipe._checkpoint_page_local_profile_issue(profile) is None
            assert pipe._checkpoint_profile_incompatibility(profile) is None

    def test_task_render_dpi_tracks_successful_escalation(self):
        from PIL import Image

        task = pipe.PageTask("d", "x.pdf", 1, Image.new("RGB", (8, 8), "white"))
        assert task.render_dpi == int(pipe.DPI)
        assert pipe._task_render_dpi(task) == int(pipe.DPI)
        task.render_dpi = 450
        assert pipe._task_render_dpi(task) == 450

    def test_embed_refuses_missing_processing_profile(self, tmp_path):
        checkpoint = tmp_path / "1.pdf"
        _create_chunk_pdf(pipe, checkpoint, "No profile")
        with pytest.raises(ValueError, match="processing_profile is required"):
            pipe._embed_page_checkpoint_metadata(
                checkpoint.read_bytes(),
                document_id="doc-no-profile",
                page_num=1,
                status=pipe.PageStatus.PADDLE,
                ocr_method="PaddleOCR",
                ocr_language="en",
                ocr_confidence=0.9,
                text_length=10,
                has_text=True,
                processing_profile=None,
                canonical_text="No profile",
            )

    @pytest.mark.parametrize(
        ("envelope", "match"),
        [
            ({"ocr_confidence": True}, "ocr_confidence"),
            ({"ocr_confidence": "0.5"}, "ocr_confidence"),
            ({"ocr_confidence": float("nan")}, "ocr_confidence"),
            ({"ocr_confidence": 1.5}, "ocr_confidence"),
            ({"text_length": True}, "text_length"),
            ({"text_length": 1.5}, "text_length"),
            ({"text_length": -1}, "text_length"),
            ({"text_sha256": "ABC"}, "text_sha256"),
            ({"text_sha256": "A" * 64}, "text_sha256"),  # uppercase rejected
            ({"has_text": True, "text_length": 0}, "has_text true"),
            ({"has_text": False, "text_length": 3}, "has_text false"),
            (
                {
                    "status": "Paddle",
                    "validation_status": "fallback",
                    "ocr_method": "PaddleOCR",
                },
                "validation_status ok",
            ),
            (
                {
                    "status": "Tesseract",
                    "validation_status": "fallback",
                    "ocr_method": "Tesseract",
                    "processing_profile": {
                        "execution_device": "gpu",
                        "execution_engine": "Tesseract",
                    },
                },
                "execution_device cpu",
            ),
            (
                {
                    "status": "ImageOnly",
                    "validation_status": "image_only",
                    "ocr_method": "ImageOnly",
                    "has_text": True,
                    "text_length": 4,
                    "text_sha256": "a" * 64,
                    "processing_profile": {
                        "execution_device": "none",
                        "execution_engine": "ImageOnly",
                    },
                },
                "no text",
            ),
            ({"status": "RESUMED"}, "Unsupported checkpoint status"),
            (
                {
                    "processing_profile": {
                        "actual_render_dpi": True,
                    },
                },
                "actual_render_dpi",
            ),
        ],
    )
    def test_invalid_checkpoint_envelope_table(self, tmp_path, envelope, match):
        checkpoint = tmp_path / "env.pdf"
        _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id="doc-env",
            page_num=1,
            text="Env text",
        )
        meta = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-env",
            page_num=1,
        )
        broken = _apply_metadata_envelope(meta, envelope)
        _rewrite_checkpoint_subject(pipe, checkpoint, broken)
        with pytest.raises(ValueError, match=match):
            pipe._load_page_checkpoint_metadata(
                str(checkpoint),
                document_id="doc-env",
                page_num=1,
            )

    def test_finalization_rejects_mutated_chunk_after_queue_snapshot(
        self,
        tmp_path,
        monkeypatch,
    ):
        """TOCTOU: same-length text mutation after snap must not ship PDF/sidecars."""
        _sd, output_dir, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="toctou.pdf",
            doc_id="doc-toctou",
            total_pages=1,
            capture_fatal=True,
        )
        monkeypatch.setattr(pipe, "ENABLE_VALIDATION", True)
        monkeypatch.setattr(pipe, "_VALIDATION_AVAILABLE", True)

        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        original = _create_checkpoint_pdf(
            pipe,
            chunk_path,
            document_id=doc.doc_id,
            page_num=1,
            text="AAAA",
        )
        meta = pipe._load_page_checkpoint_metadata(
            chunk_path,
            document_id=doc.doc_id,
            page_num=1,
        )
        original_digest = _checkpoint_sha256_for_path(chunk_path)
        snap_text = original

        # Mutate on-disk chunk to same-length BBBB while keeping original metadata.
        _create_chunk_pdf(pipe, chunk_path, "BBBB")
        _rewrite_checkpoint_subject(pipe, chunk_path, meta)
        assert _checkpoint_sha256_for_path(chunk_path) != original_digest

        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "text": snap_text,
                "status": "Paddle",
                "chunk_path": chunk_path,
                "checkpoint_sha256": original_digest,
                "ocr_confidence": 0.9,
                "ocr_method": "PaddleOCR",
                "ocr_language": "en",
                "cache_hit": False,
            }
        )
        _run_assembler_until_queue_drained(pipe)

        assert fatal.is_set()
        assert any("CHECKPOINT_ADMISSION_FAILED" in err for _p, _pg, err in failures)
        assert pipe.compression_queue.empty()
        val_dir = output_dir / "EXPORT" / "VALIDATION"
        if val_dir.is_dir():
            assert list(val_dir.rglob("*.validation.json")) == []
        if os.path.exists(doc.output_pdf):
            with pipe.fitz.open(doc.output_pdf) as out_pdf:
                if out_pdf.page_count > 0:
                    assert "BBBB" not in out_pdf[0].get_text("text")

    def test_finalization_rejects_visual_only_mutation_after_queue(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Whole-byte binding: visual/image mutation with same text must fail."""
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="visual-toctou.pdf",
            doc_id="doc-visual-toctou",
            total_pages=1,
            capture_fatal=True,
        )

        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        snap_text = _create_checkpoint_pdf(
            pipe,
            chunk_path,
            document_id=doc.doc_id,
            page_num=1,
            text="VisualBind",
        )
        meta = pipe._load_page_checkpoint_metadata(
            chunk_path,
            document_id=doc.doc_id,
            page_num=1,
        )
        original_digest = _checkpoint_sha256_for_path(chunk_path)

        # Mutate only visual content: draw filled rect, keep text + metadata.
        with pipe.fitz.open(chunk_path) as mutated:
            page = mutated[0]
            page.draw_rect(
                pipe.fitz.Rect(10, 10, 200, 200),
                color=(1, 0, 0),
                fill=(1, 0, 0),
            )
            subject = pipe.CHECKPOINT_METADATA_SUBJECT_PREFIX + json.dumps(
                meta,
                sort_keys=True,
                separators=(",", ":"),
            )
            current = mutated.metadata or {}
            md = {k: str(current.get(k, "") or "") for k in _PDF_META_KEYS}
            md["subject"] = subject
            mutated.set_metadata(md)
            mutated_bytes = mutated.tobytes()
        with open(chunk_path, "wb") as handle:
            handle.write(mutated_bytes)

        # Text and embedded metadata still match; only PDF bytes changed.
        assert pipe._extract_text_from_chunk_pdf(chunk_path, strict=True) == snap_text
        reloaded = pipe._load_page_checkpoint_metadata(
            chunk_path,
            document_id=doc.doc_id,
            page_num=1,
        )
        assert reloaded["text_sha256"] == meta["text_sha256"]
        assert _checkpoint_sha256_for_path(chunk_path) != original_digest

        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "text": snap_text,
                "status": "Paddle",
                "chunk_path": chunk_path,
                "checkpoint_sha256": original_digest,
                "ocr_confidence": 0.9,
                "ocr_method": "PaddleOCR",
                "ocr_language": "en",
                "cache_hit": False,
            }
        )
        _run_assembler_until_queue_drained(pipe)

        assert fatal.is_set()
        assert any("CHECKPOINT_ADMISSION_FAILED" in err for _p, _pg, err in failures)
        assert any(
            "checkpoint_sha256 does not match" in err for _p, _pg, err in failures
        )
        assert pipe.compression_queue.empty()
        assert not os.path.exists(doc.output_pdf) or (
            os.path.getsize(doc.output_pdf) == 0
            if os.path.exists(doc.output_pdf)
            else True
        )
        # No successful compression hand-off / deliverable.
        if os.path.exists(doc.output_pdf) and os.path.getsize(doc.output_pdf) > 0:
            with pipe.fitz.open(doc.output_pdf) as out_pdf:
                if out_pdf.page_count == 1:
                    drawings = out_pdf[0].get_drawings()
                    assert drawings == [] or "No renderable" in out_pdf[0].get_text(
                        "text"
                    )

    @pytest.mark.parametrize(
        ("share_value", "match"),
        [
            (True, "language share type"),
            ("0.5", "language share type"),
            ("1", "language share type"),
            (1.5, "out of range"),
            (float("nan"), "out of range"),
        ],
    )
    def test_language_char_shares_reject_bool_and_numeric_strings(
        self,
        tmp_path,
        share_value,
        match,
    ):
        checkpoint = tmp_path / "shares.pdf"
        _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id="doc-shares",
            page_num=1,
            text="Share gate",
        )
        meta = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-shares",
            page_num=1,
        )
        broken = dict(meta)
        broken["per_span_language_enabled"] = True
        broken["language_data"] = _language_share_payload({"en": share_value})
        _rewrite_checkpoint_subject(pipe, checkpoint, broken)
        with pytest.raises(ValueError, match=match):
            pipe._load_page_checkpoint_metadata(
                str(checkpoint),
                document_id="doc-shares",
                page_num=1,
            )

    def test_language_char_shares_accept_int_float_in_unit_interval(self, tmp_path):
        checkpoint = tmp_path / "shares-ok.pdf"
        _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id="doc-shares-ok",
            page_num=1,
            text="Share ok",
        )
        meta = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-shares-ok",
            page_num=1,
        )
        good = dict(meta)
        good["per_span_language_enabled"] = True
        good["language_data"] = _language_share_payload(
            {"en": 1, "fr": 0.0},
            {"en": 8, "fr": 0},
        )
        _rewrite_checkpoint_subject(pipe, checkpoint, good)
        loaded = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-shares-ok",
            page_num=1,
        )
        assert loaded["language_data"]["language_char_shares"]["en"] == 1.0
        assert loaded["language_data"]["language_char_shares"]["fr"] == 0.0

    @pytest.mark.parametrize(
        ("envelope", "match"),
        [
            (
                {
                    "status": "Paddle",
                    "validation_status": "ok",
                    "ocr_method": "PaddleOCR",
                    "ocr_language": "",
                    "model_language": "",
                    "processing_profile": {
                        "execution_device": "cpu",
                        "execution_engine": "PaddleOCR",
                    },
                },
                "non-empty ocr_language",
            ),
            (
                {
                    "status": "ImageOnly",
                    "validation_status": "image_only",
                    "ocr_method": "ImageOnly",
                    "has_text": False,
                    "text_length": 0,
                    "text_sha256": hashlib.sha256(b"").hexdigest(),
                    "processing_profile": {
                        "execution_device": "none",
                        "execution_engine": "ImageOnly",
                        # enabled=True so page-local applied/enabled check passes;
                        # ImageOnly status still rejects applied=True.
                        "gpu_optimization_enabled": True,
                        "gpu_optimization_applied": True,
                        "preprocessing_applied": False,
                    },
                },
                "gpu_optimization_applied false",
            ),
            (
                {
                    "status": "CRITICAL_FAILED",
                    "validation_status": "failed",
                    "ocr_method": "ImageOnly",
                    "has_text": False,
                    "text_length": 0,
                    "text_sha256": hashlib.sha256(b"").hexdigest(),
                    "processing_profile": {
                        "execution_device": "none",
                        "execution_engine": "ImageOnly",
                        "preprocessing_enabled": True,
                        "gpu_optimization_applied": False,
                        "preprocessing_applied": True,
                    },
                },
                "preprocessing_applied false",
            ),
        ],
    )
    def test_status_profile_invariants_table(self, tmp_path, envelope, match):
        checkpoint = tmp_path / "status.pdf"
        # ImageOnly/CRITICAL empty-text envelopes need an empty PDF text layer.
        if envelope.get("status") in ("ImageOnly", "CRITICAL_FAILED"):
            status = (
                pipe.PageStatus.IMAGE_ONLY
                if envelope["status"] == "ImageOnly"
                else pipe.PageStatus.CRITICAL_FAILED
            )
            with pipe.fitz.open() as pdf:
                pdf.new_page()
                checkpoint.write_bytes(pdf.tobytes())
            profile = _default_processing_profile(
                pipe,
                execution_device="none",
                execution_engine="ImageOnly",
            )
            checkpoint.write_bytes(
                pipe._embed_page_checkpoint_metadata(
                    checkpoint.read_bytes(),
                    document_id="doc-status",
                    page_num=1,
                    status=status,
                    ocr_method="ImageOnly",
                    ocr_language="",
                    ocr_confidence=0.0,
                    text_length=0,
                    has_text=False,
                    processing_profile=profile,
                    canonical_text="",
                )
            )
        else:
            _create_checkpoint_pdf(
                pipe,
                checkpoint,
                document_id="doc-status",
                page_num=1,
                text="Status text",
            )
        meta = pipe._load_page_checkpoint_metadata(
            str(checkpoint),
            document_id="doc-status",
            page_num=1,
        )
        broken = _apply_metadata_envelope(meta, envelope)
        _rewrite_checkpoint_subject(pipe, checkpoint, broken)
        with pytest.raises(ValueError, match=match):
            pipe._load_page_checkpoint_metadata(
                str(checkpoint),
                document_id="doc-status",
                page_num=1,
            )

    @pytest.mark.parametrize(
        ("status", "ocr_language", "expect_issue"),
        [
            ("Paddle", "en", None),
            ("Paddle-fr", "fr", None),
            ("Paddle-fr", "en", "suffix must match ocr_language"),
            ("Paddle-", "en", "non-empty language suffix"),
            ("Paddle", "", "non-empty ocr_language"),
            ("Paddle-en", "", "non-empty ocr_language"),
        ],
    )
    def test_paddle_dynamic_provenance_table(self, status, ocr_language, expect_issue):
        """Compact pass/fail table for Paddle / Paddle-<lang> language binding."""
        text = "Paddle provenance"
        payload = {
            "status": status,
            "validation_status": "ok",
            "ocr_method": "PaddleOCR",
            "ocr_language": ocr_language,
            "has_text": True,
            "text_length": len(text),
            "text_sha256": pipe._canonical_text_sha256(text),
            "processing_profile": _default_processing_profile(pipe),
        }
        issue = pipe._checkpoint_status_consistency_issue(payload)
        if expect_issue is None:
            assert issue is None
        else:
            assert issue is not None
            assert expect_issue in issue

    def test_page_local_applied_enabled_consistency_table(self, monkeypatch):
        """applied=True requires enabled=True; enabled=True/applied=False is ok."""
        monkeypatch.setattr(pipe, "ENABLE_GPU_OPTIMIZATION", False)
        monkeypatch.setattr(pipe, "_batch_preprocessor", None)
        monkeypatch.setattr(pipe, "ENABLE_PREPROCESSING", False)
        base = _default_processing_profile(pipe)
        assert base["gpu_optimization_enabled"] is False
        assert base["preprocessing_enabled"] is False
        assert pipe._checkpoint_page_local_profile_issue(base) is None

        cases = (
            ({"gpu_optimization_applied": True}, "gpu_optimization_applied requires"),
            ({"preprocessing_applied": True}, "preprocessing_applied requires"),
            (
                {
                    "gpu_optimization_enabled": True,
                    "gpu_optimization_applied": False,
                    "preprocessing_enabled": True,
                    "preprocessing_applied": False,
                },
                None,
            ),
            (
                {
                    "gpu_optimization_enabled": True,
                    "gpu_optimization_applied": True,
                    "preprocessing_enabled": True,
                    "preprocessing_applied": True,
                },
                None,
            ),
        )
        for overrides, expect in cases:
            profile = dict(base)
            profile.update(overrides)
            issue = pipe._checkpoint_page_local_profile_issue(profile)
            if expect is None:
                assert issue is None, overrides
            else:
                assert issue is not None and expect in issue, overrides

    def test_post_write_durable_digest_divergence_rejected(self, tmp_path):
        """Live/fallback authority rejects durable bytes that diverge from intended."""
        path = tmp_path / "1.pdf"
        intended = b"%PDF-1.4 intended checkpoint\n%%EOF\n"
        path.write_bytes(b"%PDF-1.4 mutated checkpoint\n%%EOF\n")
        with pytest.raises(RuntimeError, match="durable digest diverged"):
            pipe._durable_checkpoint_matching_intended_bytes(str(path), intended)

    def test_reuse_admission_failure_does_not_update_accumulators(
        self,
        tmp_path,
        monkeypatch,
    ):
        """RESUMED missing metadata: fatal + no terminal/finalize side effects."""
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="reuse-reject.pdf",
            doc_id="doc-reuse-reject",
            total_pages=2,
            capture_fatal=True,
        )

        # Valid page 2 so a partial admission bug would still not complete
        # if page 1 is wrongly counted... we require page 1 NOT counted.
        _put_live_assembly_page(pipe, doc, 2, "Keep page 2")

        chunk_path = os.path.join(doc.temp_dir, "1.pdf")
        _create_checkpoint_pdf(
            pipe,
            chunk_path,
            document_id=doc.doc_id,
            page_num=1,
            text="Rejected resume",
        )
        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "text": "Rejected resume",
                "status": "RESUMED",
                "chunk_path": chunk_path,
                "checkpoint_sha256": _checkpoint_sha256_for_path(chunk_path),
                "resume_metadata": None,
                "cache_hit": False,
            }
        )

        _run_assembler_until_queue_drained(pipe)

        assert fatal.is_set()
        assert any("REUSE_ADMISSION_FAILED" in err for _p, _pg, err in failures)
        assert 1 not in doc.terminal_pages
        assert 1 not in doc.terminal_statuses
        # Page 2 may have been processed; page 1 must not count toward completion.
        assert doc.processed_pages == len(doc.terminal_pages)
        assert 1 not in doc.terminal_pages
        assert not doc.finalized
        assert doc.doc_id in pipe.doc_registry
        assert pipe.compression_queue.empty()
        # A lossy replacement message cannot rehabilitate this globally fatal
        # run; restart must re-extract page 1 into a valid checkpoint.
        assert doc.terminal_pages == {2}


def _minimal_finalization_snapshot(text, digest):
    """Small complete finalizer input for one admitted page."""
    return {
        "texts": {1: text},
        "structure": {},
        "validation": {},
        "handwriting": {},
        "signature": {},
        "vertical_text": {},
        "table_fallback": {},
        "classification": {},
        "language": {},
        "chunk_digests": {1: digest},
        "no_chunk_failures": set(),
    }


@_skip_no_pipeline
class TestFinalCheckpointAdmissionHardening:
    @pytest.mark.parametrize(
        "override",
        [
            {"page_num": 0},
            {"page_num": True},
            {"status": "UNKNOWN"},
            {"status": "OK"},
            {"text": "forged failure text"},
            {"cache_hit": "false"},
            {"resume_metadata": {}},
        ],
        ids=(
            "page-zero",
            "page-bool",
            "unknown-status",
            "success-without-checkpoint",
            "failure-with-text",
            "non-bool-cache-hit",
            "live-resume-metadata",
        ),
    )
    def test_invalid_terminal_messages_fail_before_state_mutation(
        self,
        tmp_path,
        monkeypatch,
        override,
    ):
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="invalid-message.pdf",
            doc_id="doc-invalid-message",
            total_pages=1,
            capture_fatal=True,
        )
        message = {
            "doc_id": doc.doc_id,
            "page_num": 1,
            "text": "",
            "status": "CRITICAL_FAILED",
            "chunk_path": None,
            "checkpoint_sha256": None,
            "cache_hit": False,
            "resume_metadata": None,
        }
        message.update(override)
        pipe.assembly_queue.put(message)

        _run_assembler_until_queue_drained(pipe)

        assert fatal.is_set()
        assert failures
        assert doc.processed_pages == 0
        assert doc.terminal_pages == set()
        assert not doc.finalized
        assert pipe.compression_queue.empty()

    def test_conflicting_duplicate_is_fatal_and_first_admission_wins(
        self,
        tmp_path,
        monkeypatch,
    ):
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="duplicate.pdf",
            doc_id="doc-conflicting-duplicate",
            total_pages=1,
            capture_fatal=True,
        )
        _put_live_assembly_page(pipe, doc, 1, "first authority")
        first = pipe.assembly_queue.get_nowait()
        pipe.assembly_queue.task_done()
        conflict = dict(first)
        conflict["ocr_language"] = "fr"
        pipe.assembly_queue.put(first)
        pipe.assembly_queue.put(conflict)

        _run_assembler_until_queue_drained(pipe)

        assert fatal.is_set()
        assert any(
            "CONFLICTING_DUPLICATE_TERMINAL_MESSAGE" in error
            for _path, _page, error in failures
        )
        assert doc.processed_pages == 1
        assert doc.terminal_pages == {1}
        assert doc.terminal_statuses[1] == pipe.PageStatus.PADDLE.value
        assert not doc.finalized
        assert pipe.compression_queue.empty()

    def test_producer_drain_lost_terminal_blocks_all_finalizers(
        self,
        tmp_path,
        monkeypatch,
    ):
        source_dir, _od, _td, _sf, complete, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="complete.pdf",
            doc_id="doc-complete",
            total_pages=1,
            capture_fatal=True,
        )
        incomplete_source = source_dir / "incomplete.pdf"
        incomplete_source.write_bytes(b"%PDF-1.4\n%%EOF\n")
        incomplete = _make_doc_state(
            pipe,
            incomplete_source,
            "doc-incomplete",
        )
        incomplete.total_pages = 2
        submitted = []
        monkeypatch.setattr(
            pipe,
            "_finalize_doc",
            lambda *args: submitted.append(args),
        )
        _put_live_assembly_page(pipe, complete, 1, "complete page")
        _put_live_assembly_page(pipe, incomplete, 1, "only first page")

        _run_assembler_until_queue_drained(pipe)

        assert fatal.is_set()
        assert len(submitted) == 1
        assert submitted[0][0] is complete
        assert complete.finalized
        assert not incomplete.finalized
        assert any("missing=[2]" in error for _path, _page, error in failures)
        assert pipe.compression_queue.empty()

    def test_admitted_chunk_lost_before_finalizer_is_fatal(
        self,
        tmp_path,
        monkeypatch,
    ):
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="lost-after-admission.pdf",
            doc_id="doc-lost-after-admission",
            total_pages=1,
            capture_fatal=True,
        )
        checkpoint = os.path.join(doc.temp_dir, "1.pdf")
        text = _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id=doc.doc_id,
            page_num=1,
            text="admitted bytes",
        )
        digest = _checkpoint_sha256_for_path(checkpoint)
        os.remove(checkpoint)

        pipe._finalize_doc(
            doc,
            doc.doc_id,
            _minimal_finalization_snapshot(text, digest),
        )

        assert fatal.is_set()
        assert any("lost an admitted" in error for _p, _pg, error in failures)
        assert not os.path.exists(doc.output_pdf)
        assert pipe.compression_queue.empty()

    @pytest.mark.parametrize("failure_mode", ["sidecar", "custody"])
    def test_late_finalization_failure_preserves_prior_canonical_pdf(
        self,
        tmp_path,
        monkeypatch,
        failure_mode,
    ):
        _sd, output_dir, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="staged-rollback.pdf",
            doc_id="doc-staged-rollback",
            total_pages=1,
            capture_fatal=True,
        )
        os.makedirs(os.path.dirname(doc.output_pdf), exist_ok=True)
        prior = b"%PDF-prior-canonical"
        Path(doc.output_pdf).write_bytes(prior)
        checkpoint = os.path.join(doc.temp_dir, "1.pdf")
        text = _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id=doc.doc_id,
            page_num=1,
            text="new staged bytes",
        )
        digest = _checkpoint_sha256_for_path(checkpoint)
        expected_error = f"late {failure_mode} failure"
        if failure_mode == "sidecar":
            monkeypatch.setattr(
                pipe,
                "write_text_atomic",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    OSError(expected_error)
                ),
            )
        else:
            doc.custody_chain = type(
                "FailingCustody",
                (),
                {
                    "append_event": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        OSError(expected_error)
                    )
                },
            )()

        pipe._finalize_doc(
            doc,
            doc.doc_id,
            _minimal_finalization_snapshot(text, digest),
        )

        assert fatal.is_set()
        assert any(expected_error in error for _p, _pg, error in failures)
        assert Path(doc.output_pdf).read_bytes() == prior
        assert pipe.compression_queue.empty()
        assert list(output_dir.glob(".finalize-*")) == []

    @pytest.mark.parametrize("surface", ["validation", "retrieval"])
    def test_enabled_required_surface_failure_blocks_publication(
        self,
        tmp_path,
        monkeypatch,
        surface,
    ):
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name=f"{surface}-required.pdf",
            doc_id=f"doc-{surface}-required",
            total_pages=1,
            capture_fatal=True,
        )
        checkpoint = os.path.join(doc.temp_dir, "1.pdf")
        text = _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id=doc.doc_id,
            page_num=1,
            text="required surface",
        )
        digest = _checkpoint_sha256_for_path(checkpoint)
        if surface == "validation":
            monkeypatch.setattr(pipe, "ENABLE_VALIDATION", True)
            monkeypatch.setattr(pipe, "_VALIDATION_AVAILABLE", True)
            monkeypatch.setattr(pipe, "write_validation_json", lambda *_a, **_k: None)
        else:
            monkeypatch.setattr(pipe, "ENABLE_RETRIEVAL_OUTPUT", True)
            monkeypatch.setattr(pipe, "_OUTPUT_ASSEMBLER_AVAILABLE", True)
            monkeypatch.setattr(
                pipe,
                "assemble_retrieval_output",
                lambda **_kwargs: SimpleNamespace(
                    classification={},
                    entities=[],
                    key_value_pairs=[],
                ),
            )
            monkeypatch.setattr(
                pipe,
                "write_retrieval_json",
                lambda *_a, **_k: "retrieval.json",
            )
            monkeypatch.setattr(
                pipe,
                "write_retrieval_markdown",
                lambda *_a, **_k: None,
            )

        pipe._finalize_doc(
            doc,
            doc.doc_id,
            _minimal_finalization_snapshot(text, digest),
        )

        assert fatal.is_set()
        assert not os.path.exists(doc.output_pdf)
        assert pipe.compression_queue.empty()
        assert any(surface in error.lower() for _p, _pg, error in failures)

    def test_full_compression_queue_fails_before_publish_or_custody(
        self,
        tmp_path,
        monkeypatch,
    ):
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="handoff-full.pdf",
            doc_id="doc-handoff-full",
            total_pages=1,
            capture_fatal=True,
        )
        os.makedirs(os.path.dirname(doc.output_pdf), exist_ok=True)
        prior = b"%PDF-prior-canonical"
        Path(doc.output_pdf).write_bytes(prior)
        checkpoint = os.path.join(doc.temp_dir, "1.pdf")
        text = _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id=doc.doc_id,
            page_num=1,
            text="new staged bytes",
        )
        digest = _checkpoint_sha256_for_path(checkpoint)

        class RecordingCustody:
            integrity_compromised = False

            def __init__(self):
                self.events = []

            def append_event(self, event_type, data):
                self.events.append((event_type, data))

        custody = RecordingCustody()
        doc.custody_chain = custody
        full_queue = queue.Queue(maxsize=1)
        full_queue.put("occupied")
        monkeypatch.setattr(pipe, "compression_queue", full_queue)
        monkeypatch.setattr(
            pipe,
            "COMPRESSION_HANDOFF_RESERVATION_TIMEOUT_SECONDS",
            0.0,
        )

        pipe._finalize_doc(
            doc,
            doc.doc_id,
            _minimal_finalization_snapshot(text, digest),
        )

        assert fatal.is_set()
        assert Path(doc.output_pdf).read_bytes() == prior
        assert full_queue.qsize() == 1
        assert all(event != "assembly_complete" for event, _data in custody.events)
        assert any("capacity unavailable" in error for _p, _pg, error in failures)

    def test_enabled_language_checkpoint_refuses_incomplete_analysis(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(pipe, "ENABLE_PER_SPAN_LANGUAGE", True)
        raw_path = tmp_path / "language.pdf"
        _create_chunk_pdf(pipe, raw_path, "text requiring language analysis")
        canonical = pipe._extract_text_from_chunk_pdf(str(raw_path), strict=True)
        with pytest.raises(ValueError, match="analysis is incomplete"):
            pipe._embed_page_checkpoint_metadata(
                raw_path.read_bytes(),
                document_id="doc-language-incomplete",
                page_num=1,
                status=pipe.PageStatus.PADDLE,
                ocr_method="PaddleOCR",
                ocr_language="en",
                ocr_confidence=0.9,
                text_length=len(canonical),
                has_text=True,
                processing_profile=_default_processing_profile(pipe),
                canonical_text=canonical,
            )

    def test_text_checkpoint_refuses_type_shaped_empty_language_summary(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(pipe, "ENABLE_PER_SPAN_LANGUAGE", True)
        raw_path = tmp_path / "empty-language.pdf"
        _create_chunk_pdf(pipe, raw_path, "text requiring real analysis")
        canonical = pipe._extract_text_from_chunk_pdf(str(raw_path), strict=True)
        empty_summary = {
            "page_num": 1,
            "primary_language": "und",
            "primary_confidence": 0.0,
            "languages_detected": [],
            "language_char_shares": {},
            "language_char_counts": {},
            "scripts_detected": [],
            "mixed_script": False,
            "span_count": 0,
            "spans_labeled": 0,
        }
        with pytest.raises(ValueError, match="no analyzed language spans"):
            pipe._embed_page_checkpoint_metadata(
                raw_path.read_bytes(),
                document_id="doc-empty-language",
                page_num=1,
                status=pipe.PageStatus.PADDLE,
                ocr_method="PaddleOCR",
                ocr_language="en",
                ocr_confidence=0.9,
                text_length=len(canonical),
                has_text=True,
                language_data=empty_summary,
                processing_profile=_default_processing_profile(pipe),
                canonical_text=canonical,
            )

    def test_queued_language_must_equal_checkpoint_summary(
        self,
        tmp_path,
        monkeypatch,
    ):
        _sd, _od, _td, _sf, doc, failures, fatal = _setup_assembler_case(
            monkeypatch,
            pipe,
            tmp_path,
            source_name="language-mismatch.pdf",
            doc_id="doc-language-mismatch",
            total_pages=1,
            capture_fatal=True,
        )
        monkeypatch.setattr(pipe, "ENABLE_PER_SPAN_LANGUAGE", True)
        checkpoint = os.path.join(doc.temp_dir, "1.pdf")
        embedded = _language_share_payload({"en": 1.0})
        text = _create_checkpoint_pdf(
            pipe,
            checkpoint,
            document_id=doc.doc_id,
            page_num=1,
            text="language-bound text",
            language_data=embedded,
        )
        queued = _language_share_payload({"fr": 1.0})
        queued["primary_language"] = "fr"
        pipe.assembly_queue.put(
            {
                "doc_id": doc.doc_id,
                "page_num": 1,
                "text": text,
                "status": pipe.PageStatus.PADDLE,
                "chunk_path": checkpoint,
                "checkpoint_sha256": _checkpoint_sha256_for_path(checkpoint),
                "ocr_method": "PaddleOCR",
                "ocr_language": "en",
                "ocr_confidence": 0.9,
                "language_data": queued,
                "cache_hit": False,
            }
        )

        _run_assembler_until_queue_drained(pipe)

        assert fatal.is_set()
        assert not doc.finalized
        assert doc.terminal_pages == set()
        assert any(
            "queued language analysis diverges" in error
            for _path, _page, error in failures
        )
        assert pipe.compression_queue.empty()

    def test_invalid_cache_candidate_never_overwrites_canonical_checkpoint(
        self,
        tmp_path,
    ):
        canonical = tmp_path / "1.pdf"
        _create_checkpoint_pdf(
            pipe,
            canonical,
            document_id="doc-cache-authority",
            page_num=1,
            text="canonical authority",
        )
        prior = canonical.read_bytes()

        with pytest.raises((RuntimeError, ValueError)):
            pipe._publish_or_select_cached_checkpoint(
                str(canonical),
                b"not a checkpoint",
                document_id="doc-cache-authority",
                page_num=1,
            )

        assert canonical.read_bytes() == prior
