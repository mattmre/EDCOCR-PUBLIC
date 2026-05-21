"""
Resume regression test suite for ocr_gpu_async.py crash-resume infrastructure.

Covers two tiers:
  Tier 1 — Pure function determinism (fingerprint, doc-id, manifest I/O)
  Tier 2 — Strict-PDF-signature filtering (classify_source_file edge cases)

These tests form a regression firewall ensuring that any refactor of the resume
subsystem preserves the core invariants: deterministic fingerprints, correct
cache invalidation, and reliable manifest round-trip.

Run with: python -m pytest tests/test_resume_regression.py -v
"""

import ast
import hashlib
import io
import json
import os
import shutil
import types

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants (must match ocr_gpu_async.py)
# ---------------------------------------------------------------------------
RESUME_MANIFEST_FILENAME = "resume_manifest.json"
RESUME_MANIFEST_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Import strategy — mirrors tests/test_utilities.py _load_utility_functions()
# ---------------------------------------------------------------------------
def _load_utility_functions():
    """Extract resume-related utility functions from ocr_gpu_async.py
    without triggering PaddleOCR/Tesseract imports."""
    import datetime

    src_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py"
    )

    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    mod = types.ModuleType("ocr_resume_utils")
    mod.__file__ = src_path

    # Inject stdlib dependencies used by the target functions
    mod.os = os
    mod.io = io
    mod.hashlib = hashlib
    mod.json = json
    mod.datetime = datetime
    mod.shutil = shutil
    mod.np = np
    mod.Image = Image

    # Inject constants the functions reference
    mod.SOURCE_FOLDER = "/app/ocr_source"
    mod.RESUME_MANIFEST_FILENAME = RESUME_MANIFEST_FILENAME
    mod.RESUME_MANIFEST_SCHEMA_VERSION = RESUME_MANIFEST_SCHEMA_VERSION
    mod.PDF_EXTENSIONS = {".pdf"}
    mod.VIDEO_EXTENSIONS = {
        ".mp4", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".mkv", ".webm",
    }
    mod.PHASE1_IMAGE_EXTENSIONS = {
        ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif",
        ".webp", ".jp2", ".jpx", ".pnm", ".pbm", ".pgm", ".ppm",
        ".pcx", ".ico", ".svg", ".svgz", ".wmf", ".emf",
    }
    mod.PHASE2_IMAGE_EXTENSIONS = {
        ".heic", ".heif", ".avif", ".jxl", ".jxr", ".dcx", ".xps",
    }
    mod.STRICT_PDF_SIGNATURE = False

    import tempfile as _tempfile
    mod.tempfile = _tempfile

    function_names = [
        "read_file_header",
        "detect_magic_family",
        "classify_source_file",
        "compute_source_fingerprint",
        "build_resume_doc_id",
        "_resume_manifest_path",
        "_load_resume_manifest",
        "_write_manifest_atomic",
        "_write_resume_manifest",
        "_reset_resume_temp_dir",
        "prepare_resume_state",
    ]

    source_tree = ast.parse(source)
    function_nodes = {
        node.name: node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef)
    }

    for func_name in function_names:
        node = function_nodes.get(func_name)
        if node is None:
            continue

        func_source = ast.get_source_segment(source, node)
        if not func_source:
            continue

        try:
            exec(compile(func_source, src_path, "exec"), mod.__dict__)
        except Exception as e:
            print(f"Warning: Could not load {func_name}: {e}")

    return mod


# Try direct import first, fall back to source extraction
try:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from ocr_gpu_async import (
        _load_resume_manifest,
        _reset_resume_temp_dir,
        _resume_manifest_path,
        _write_manifest_atomic,
        _write_resume_manifest,
        build_resume_doc_id,
        classify_source_file,
        compute_source_fingerprint,
        prepare_resume_state,
    )
    _DIRECT_IMPORT = True
except ImportError:
    _utils = _load_utility_functions()
    compute_source_fingerprint = _utils.compute_source_fingerprint
    build_resume_doc_id = _utils.build_resume_doc_id
    _resume_manifest_path = _utils._resume_manifest_path
    _load_resume_manifest = _utils._load_resume_manifest
    _write_manifest_atomic = _utils._write_manifest_atomic
    _write_resume_manifest = _utils._write_resume_manifest
    _reset_resume_temp_dir = _utils._reset_resume_temp_dir
    prepare_resume_state = _utils.prepare_resume_state
    classify_source_file = _utils.classify_source_file
    _DIRECT_IMPORT = False


# ===========================================================================
# Tier 1, Class 1: TestSourceFingerprintDeterminism
# ===========================================================================

class TestSourceFingerprintDeterminism:
    """Verify compute_source_fingerprint returns stable, complete results."""

    def test_fingerprint_same_content_same_result(self, tmp_path):
        """Identical file content must produce identical fingerprints."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4\nSome content\n%%EOF\n")

        fp1 = compute_source_fingerprint(str(f))
        fp2 = compute_source_fingerprint(str(f))
        assert fp1 == fp2

    def test_fingerprint_changes_when_content_changes(self, tmp_path):
        """Rewriting the file changes the content_sha256."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4\nOriginal\n%%EOF\n")
        fp1 = compute_source_fingerprint(str(f))

        f.write_bytes(b"%PDF-1.4\nModified\n%%EOF\n")
        fp2 = compute_source_fingerprint(str(f))

        assert fp1["content_sha256"] != fp2["content_sha256"]

    def test_fingerprint_contains_required_keys(self, tmp_path):
        """Fingerprint dict must contain all three canonical keys."""
        f = tmp_path / "sample.pdf"
        f.write_bytes(b"%PDF-1.4\n%%EOF\n")

        fp = compute_source_fingerprint(str(f))
        assert "content_sha256" in fp
        assert "size_bytes" in fp
        assert "mtime_ns" in fp

    def test_fingerprint_sha256_is_64_hex_chars(self, tmp_path):
        """The content_sha256 field must be a 64-character lowercase hex string."""
        f = tmp_path / "sample.pdf"
        f.write_bytes(b"test data for hashing")

        fp = compute_source_fingerprint(str(f))
        sha = fp["content_sha256"]
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_fingerprint_size_bytes_matches_stat(self, tmp_path):
        """size_bytes must agree with os.stat()."""
        content = b"%PDF-1.7\nSome arbitrary bytes here\n%%EOF\n"
        f = tmp_path / "sized.pdf"
        f.write_bytes(content)

        fp = compute_source_fingerprint(str(f))
        assert fp["size_bytes"] == len(content)
        assert fp["size_bytes"] == os.stat(str(f)).st_size


# ===========================================================================
# Tier 1, Class 2: TestResumeDocIdDeterminism
# ===========================================================================

class TestResumeDocIdDeterminism:
    """Verify build_resume_doc_id returns stable, distinguishing results."""

    _BASE_PATH = "/app/ocr_source/folder/document.pdf"
    _BASE_FP = {"content_sha256": "a" * 64, "size_bytes": 1024, "mtime_ns": 1000000000}

    def test_same_path_same_fingerprint_same_id(self):
        """Deterministic: same inputs always yield same doc id."""
        id1 = build_resume_doc_id(self._BASE_PATH, self._BASE_FP)
        id2 = build_resume_doc_id(self._BASE_PATH, self._BASE_FP)
        assert id1 == id2

    def test_different_content_hash_different_id(self):
        """Changing content_sha256 must change the doc id."""
        fp_alt = {**self._BASE_FP, "content_sha256": "b" * 64}
        id_base = build_resume_doc_id(self._BASE_PATH, self._BASE_FP)
        id_alt = build_resume_doc_id(self._BASE_PATH, fp_alt)
        assert id_base != id_alt

    def test_different_size_different_id(self):
        """Changing size_bytes must change the doc id."""
        fp_alt = {**self._BASE_FP, "size_bytes": 9999}
        id_base = build_resume_doc_id(self._BASE_PATH, self._BASE_FP)
        id_alt = build_resume_doc_id(self._BASE_PATH, fp_alt)
        assert id_base != id_alt

    def test_different_mtime_different_id(self):
        """Changing mtime_ns must change the doc id."""
        fp_alt = {**self._BASE_FP, "mtime_ns": 2000000000}
        id_base = build_resume_doc_id(self._BASE_PATH, self._BASE_FP)
        id_alt = build_resume_doc_id(self._BASE_PATH, fp_alt)
        assert id_base != id_alt

    def test_different_path_different_id(self):
        """Different source path must produce different doc id."""
        id_base = build_resume_doc_id(self._BASE_PATH, self._BASE_FP)
        id_alt = build_resume_doc_id("/app/ocr_source/other/file.pdf", self._BASE_FP)
        assert id_base != id_alt

    def test_doc_id_is_hex_string(self):
        """Doc id must be a 16-character lowercase hex string (sha256 prefix)."""
        doc_id = build_resume_doc_id(self._BASE_PATH, self._BASE_FP)
        assert len(doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_doc_id_stable_across_invocations(self):
        """Recomputing with identical inputs across separate calls yields same value."""
        ids = [build_resume_doc_id(self._BASE_PATH, self._BASE_FP) for _ in range(10)]
        assert len(set(ids)) == 1


# ===========================================================================
# Tier 1, Class 3: TestPrepareResumeStateInvalidation
# ===========================================================================

class TestPrepareResumeStateInvalidation:
    """Verify prepare_resume_state correctly invalidates stale state."""

    _SOURCE_PATH = "/app/ocr_source/doc.pdf"
    _FP_V1 = {"content_sha256": "a" * 64, "size_bytes": 100, "mtime_ns": 1}
    _FP_V2 = {"content_sha256": "b" * 64, "size_bytes": 200, "mtime_ns": 2}

    def _write_manifest(self, temp_dir, source_path, fingerprint, schema_version=RESUME_MANIFEST_SCHEMA_VERSION):
        """Helper to write a manifest file directly."""
        manifest_path = os.path.join(str(temp_dir), RESUME_MANIFEST_FILENAME)
        payload = {
            "schema_version": schema_version,
            "source_path": source_path,
            "source_fingerprint": fingerprint,
        }
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def test_invalidates_when_content_changes(self, tmp_path):
        """When content_sha256 changes, stale chunks deleted, manifest updated."""
        temp_dir = tmp_path / "resume"
        temp_dir.mkdir()
        (temp_dir / "1.pdf").write_bytes(b"stale chunk")
        self._write_manifest(temp_dir, self._SOURCE_PATH, self._FP_V1)

        result = prepare_resume_state(str(temp_dir), self._SOURCE_PATH, self._FP_V2)

        assert result["status"] == "invalidated"
        assert not (temp_dir / "1.pdf").exists()
        manifest = json.loads(
            (temp_dir / RESUME_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert manifest["source_fingerprint"]["content_sha256"] == self._FP_V2["content_sha256"]

    def test_invalidates_when_size_changes(self, tmp_path):
        """When size_bytes differs, state is invalidated."""
        temp_dir = tmp_path / "resume"
        temp_dir.mkdir()
        (temp_dir / "1.pdf").write_bytes(b"chunk data")
        fp_size_changed = {**self._FP_V1, "size_bytes": 9999}
        self._write_manifest(temp_dir, self._SOURCE_PATH, self._FP_V1)

        result = prepare_resume_state(
            str(temp_dir), self._SOURCE_PATH, fp_size_changed
        )
        assert result["status"] == "invalidated"

    def test_valid_when_fingerprint_matches(self, tmp_path):
        """When fingerprint matches, existing chunks are preserved."""
        temp_dir = tmp_path / "resume"
        temp_dir.mkdir()
        chunk = temp_dir / "1.pdf"
        chunk.write_bytes(b"good chunk")
        self._write_manifest(temp_dir, self._SOURCE_PATH, self._FP_V1)

        result = prepare_resume_state(str(temp_dir), self._SOURCE_PATH, self._FP_V1)

        assert result["status"] == "valid"
        assert result["removed_entries"] == 0
        assert chunk.exists(), "Valid chunks must be preserved"

    def test_initialized_when_empty_temp_dir(self, tmp_path):
        """Empty temp dir with no manifest yields 'initialized'."""
        temp_dir = tmp_path / "resume-empty"
        temp_dir.mkdir()

        result = prepare_resume_state(str(temp_dir), self._SOURCE_PATH, self._FP_V1)

        assert result["status"] == "initialized"
        assert result["removed_entries"] == 0

    def test_invalidated_when_schema_version_mismatch(self, tmp_path):
        """Schema version mismatch causes invalidation."""
        temp_dir = tmp_path / "resume"
        temp_dir.mkdir()
        (temp_dir / "1.pdf").write_bytes(b"chunk")
        # Write manifest with wrong schema version
        self._write_manifest(
            temp_dir, self._SOURCE_PATH, self._FP_V1, schema_version=999
        )

        result = prepare_resume_state(str(temp_dir), self._SOURCE_PATH, self._FP_V1)

        assert result["status"] == "invalidated"

    def test_invalidated_when_manifest_corrupt_json(self, tmp_path):
        """Corrupt JSON manifest triggers invalidation."""
        temp_dir = tmp_path / "resume"
        temp_dir.mkdir()
        (temp_dir / "2.pdf").write_bytes(b"chunk")
        manifest_path = temp_dir / RESUME_MANIFEST_FILENAME
        manifest_path.write_text("{this is not valid json!!!", encoding="utf-8")

        result = prepare_resume_state(str(temp_dir), self._SOURCE_PATH, self._FP_V1)

        assert result["status"] == "invalidated"

    def test_stale_chunks_removed_on_invalidation(self, tmp_path):
        """All PDF chunks must be removed during invalidation."""
        temp_dir = tmp_path / "resume"
        temp_dir.mkdir()
        chunks = []
        for i in range(1, 6):
            p = temp_dir / f"{i}.pdf"
            p.write_bytes(f"chunk {i}".encode())
            chunks.append(p)
        self._write_manifest(temp_dir, self._SOURCE_PATH, self._FP_V1)

        prepare_resume_state(str(temp_dir), self._SOURCE_PATH, self._FP_V2)

        for chunk in chunks:
            assert not chunk.exists(), f"{chunk.name} should have been removed"

    def test_manifest_rewritten_after_invalidation(self, tmp_path):
        """After invalidation the manifest reflects the new fingerprint."""
        temp_dir = tmp_path / "resume"
        temp_dir.mkdir()
        self._write_manifest(temp_dir, self._SOURCE_PATH, self._FP_V1)
        (temp_dir / "1.pdf").write_bytes(b"stale")

        prepare_resume_state(str(temp_dir), self._SOURCE_PATH, self._FP_V2)

        manifest = json.loads(
            (temp_dir / RESUME_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert manifest["source_fingerprint"] == self._FP_V2
        assert manifest["source_path"] == self._SOURCE_PATH
        assert manifest["schema_version"] == RESUME_MANIFEST_SCHEMA_VERSION


# ===========================================================================
# Tier 1, Class 4: TestManifestIO
# ===========================================================================

class TestManifestIO:
    """Verify manifest write/load round-trip and error handling."""

    _SOURCE_PATH = "/app/ocr_source/test/doc.pdf"
    _FP = {"content_sha256": "c" * 64, "size_bytes": 512, "mtime_ns": 3000}

    def test_write_then_load_roundtrip(self, tmp_path):
        """Write + load must preserve all data fields."""
        temp_dir = str(tmp_path / "manifest-test")
        os.makedirs(temp_dir, exist_ok=True)

        _write_resume_manifest(temp_dir, self._SOURCE_PATH, self._FP)
        loaded = _load_resume_manifest(temp_dir)

        assert loaded is not None
        assert loaded["schema_version"] == RESUME_MANIFEST_SCHEMA_VERSION
        assert loaded["source_path"] == self._SOURCE_PATH
        assert loaded["source_fingerprint"] == self._FP
        assert "updated_at" in loaded

    def test_load_returns_none_for_missing_file(self, tmp_path):
        """Missing manifest returns None (not an error)."""
        result = _load_resume_manifest(str(tmp_path))
        assert result is None

    def test_load_returns_none_for_corrupt_json(self, tmp_path):
        """Corrupt JSON returns None."""
        manifest_path = tmp_path / RESUME_MANIFEST_FILENAME
        manifest_path.write_text("{{not json at all", encoding="utf-8")

        result = _load_resume_manifest(str(tmp_path))
        assert result is None

    def test_load_returns_none_for_wrong_schema_version(self, tmp_path):
        """Manifest with unknown schema version returns None."""
        manifest_path = tmp_path / RESUME_MANIFEST_FILENAME
        payload = {
            "schema_version": 999,
            "source_path": self._SOURCE_PATH,
            "source_fingerprint": self._FP,
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        result = _load_resume_manifest(str(tmp_path))
        assert result is None

    def test_load_returns_none_when_fingerprint_not_dict(self, tmp_path):
        """Manifest where source_fingerprint is not a dict returns None."""
        manifest_path = tmp_path / RESUME_MANIFEST_FILENAME
        payload = {
            "schema_version": RESUME_MANIFEST_SCHEMA_VERSION,
            "source_path": self._SOURCE_PATH,
            "source_fingerprint": "not-a-dict",
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        result = _load_resume_manifest(str(tmp_path))
        assert result is None


# ===========================================================================
# Tier 1, Class 5: TestAtomicManifestCrashSafety
# ===========================================================================

class TestAtomicManifestCrashSafety:
    """Verify _write_manifest_atomic crash-safety guarantees.

    The resume manifest must be written via tmp->fsync->rename so that a
    power failure mid-write cannot leave a corrupt/partial manifest on disk.
    """

    def test_atomic_write_creates_target(self, tmp_path):
        """Happy path: atomic write produces a readable JSON file at target."""
        target = str(tmp_path / "manifest.json")
        data = {"a": 1, "b": [2, 3], "c": "text"}
        _write_manifest_atomic(target, data)

        assert os.path.exists(target)
        with open(target, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_write_leaves_no_tmp_files_on_success(self, tmp_path):
        """After a successful write, no .tmp sidecar files should remain."""
        target = str(tmp_path / "manifest.json")
        _write_manifest_atomic(target, {"x": 1})

        residual = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
        assert residual == []

    def test_atomic_write_overwrites_existing_manifest(self, tmp_path):
        """Subsequent writes atomically replace earlier content."""
        target = str(tmp_path / "manifest.json")
        _write_manifest_atomic(target, {"v": 1})
        _write_manifest_atomic(target, {"v": 2})

        with open(target, "r", encoding="utf-8") as f:
            assert json.load(f) == {"v": 2}

    def test_atomic_write_cleans_tmp_on_serialization_failure(self, tmp_path):
        """If json.dump raises (unserializable data), no orphan tmp remains."""
        target = str(tmp_path / "manifest.json")

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            _write_manifest_atomic(target, {"bad": Unserializable()})

        # Neither the target nor a leftover tmp file should exist.
        assert not os.path.exists(target)
        residual = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
        assert residual == []

    def test_atomic_write_creates_parent_directory(self, tmp_path):
        """Atomic write creates the parent directory if it is missing."""
        nested = tmp_path / "nested" / "subdir"
        target = str(nested / "manifest.json")

        _write_manifest_atomic(target, {"ok": True})

        assert os.path.exists(target)

    def test_resume_manifest_write_is_fsynced(self, tmp_path, monkeypatch):
        """_write_resume_manifest must call os.fsync for crash safety."""
        calls = {"fsync": 0, "replace": 0}

        real_fsync = os.fsync
        real_replace = os.replace

        def tracking_fsync(fd):
            calls["fsync"] += 1
            return real_fsync(fd)

        def tracking_replace(src, dst):
            # fsync must be called before os.replace
            assert calls["fsync"] >= 1, (
                "os.fsync must be called before os.replace to guarantee "
                "crash-safety"
            )
            calls["replace"] += 1
            return real_replace(src, dst)

        # Patch on the ocr_gpu_async module when directly imported, or on
        # the dynamically-loaded utility module when source-extracted.
        if _DIRECT_IMPORT:
            import ocr_gpu_async as _target_mod
        else:
            _target_mod = _utils

        monkeypatch.setattr(_target_mod.os, "fsync", tracking_fsync)
        monkeypatch.setattr(_target_mod.os, "replace", tracking_replace)

        temp_dir = str(tmp_path / "doc-temp")
        fp = {"content_sha256": "a" * 64, "size_bytes": 1, "mtime_ns": 2}
        _write_resume_manifest(temp_dir, "/app/ocr_source/x.pdf", fp)

        assert calls["fsync"] >= 1, "fsync must be invoked at least once"
        assert calls["replace"] >= 1, "os.replace must finalize the manifest"

    def test_resume_manifest_no_partial_on_replace_failure(self, tmp_path, monkeypatch):
        """If os.replace fails, the prior manifest is not corrupted."""
        temp_dir = str(tmp_path / "doc-temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Write a valid initial manifest
        fp1 = {"content_sha256": "a" * 64, "size_bytes": 1, "mtime_ns": 1}
        _write_resume_manifest(temp_dir, "/src/a.pdf", fp1)
        initial = _load_resume_manifest(temp_dir)
        assert initial is not None

        # Force os.replace to fail on the next write
        if _DIRECT_IMPORT:
            import ocr_gpu_async as _target_mod
        else:
            _target_mod = _utils

        def failing_replace(src, dst):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(_target_mod.os, "replace", failing_replace)

        fp2 = {"content_sha256": "b" * 64, "size_bytes": 2, "mtime_ns": 2}
        with pytest.raises(OSError, match="simulated disk failure"):
            _write_resume_manifest(temp_dir, "/src/b.pdf", fp2)

        # Prior manifest must still be readable and uncorrupted
        after = _load_resume_manifest(temp_dir)
        assert after is not None
        assert after["source_fingerprint"] == fp1

        # No orphan tmp files left behind
        residual = [p for p in os.listdir(temp_dir) if p.endswith(".tmp")]
        assert residual == []


# ===========================================================================
# Tier 2, Class 6: TestStrictPdfSignatureRegression
# ===========================================================================

class TestStrictPdfSignatureRegression:
    """Verify strict-PDF-signature mode correctly filters malformed PDFs."""

    def test_valid_pdf_accepted_in_strict_mode(self, tmp_path):
        """A file with a real %PDF- header passes strict mode."""
        pdf = tmp_path / "valid.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

        result, warning = classify_source_file(str(pdf), strict_pdf_signature=True)
        assert result == "pdf"

    def test_garbage_pdf_rejected_in_strict_mode(self, tmp_path):
        """A .pdf file without %PDF- header is rejected in strict mode."""
        garbage = tmp_path / "garbage.pdf"
        garbage.write_bytes(b"not a pdf at all")

        result, reason = classify_source_file(str(garbage), strict_pdf_signature=True)
        assert result is None
        assert "missing pdf signature" in reason.lower()

    def test_image_files_unaffected_by_strict_mode(self, tmp_path):
        """Strict PDF mode does not affect image file classification."""
        jpg = tmp_path / "photo.jpg"
        # Write minimal JPEG header (SOI marker + JFIF-like bytes)
        jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        result, warning = classify_source_file(str(jpg), strict_pdf_signature=True)
        assert result == "image"

    def test_empty_pdf_accepted_in_permissive_mode(self, tmp_path):
        """In permissive mode (strict=False), empty .pdf accepted by extension fallback."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")

        result, warning = classify_source_file(
            str(empty_pdf), strict_pdf_signature=False
        )
        assert result == "pdf"
        assert "extension fallback" in warning.lower()

    @pytest.mark.parametrize(
        ("file_name", "file_content", "expected_result", "description"),
        [
            (
                "valid.pdf",
                b"%PDF-1.4\n%%EOF\n",
                "pdf",
                "Valid PDF should be accepted",
            ),
            (
                "bad.pdf",
                b"not a pdf at all",
                None,
                "Garbage PDF should be rejected",
            ),
            (
                "scan.png",
                b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
                "image",
                "Valid image should be accepted",
            ),
        ],
    )
    def test_mixed_batch_strict_mode(
        self, tmp_path, file_name, file_content, expected_result, description
    ):
        """Mix of valid/invalid files in strict mode: correct per-file filtering."""
        test_file = tmp_path / file_name
        test_file.write_bytes(file_content)

        result, _ = classify_source_file(
            str(test_file), strict_pdf_signature=True
        )
        assert result == expected_result, description
