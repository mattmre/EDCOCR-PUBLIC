"""
Unit tests for ocr_gpu_async.py utility functions.

These tests cover pure utility functions that do NOT require:
- GPU access
- Docker environment
- PaddleOCR/Tesseract engines
- Running pipeline threads

Run with: python -m pytest tests/ -v
"""
import datetime
import hashlib
import io
import json
import os
import shutil
import tempfile

import numpy as np
import pytest
from PIL import Image

# Add project root to path so we can import individual functions
# without triggering full PaddleOCR initialization
from tests.create_fixtures import FIXTURES_DIR, create_fixtures

# ---------------------------------------------------------------------------
# Fixtures (pytest)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def setup_fixtures():
    """Create test fixture files once per session."""
    create_fixtures()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Import utilities from ocr_gpu_async — need to handle import side effects
# ---------------------------------------------------------------------------

# We import the module-level functions by loading the source as a module.
# This avoids triggering PaddleOCR/Tesseract imports that would fail
# without GPU/system deps.

def _load_utility_functions():
    """Extract utility functions from ocr_gpu_async.py without running imports."""
    import types

    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py")

    # Read source and extract only the utility functions we need
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Create a minimal module with just the functions we need
    mod = types.ModuleType("ocr_utils")
    mod.__file__ = src_path

    # These are needed by the utility functions
    mod.os = os
    mod.io = io
    mod.hashlib = hashlib
    mod.json = json
    mod.datetime = datetime
    mod.shutil = shutil
    mod.np = np
    mod.Image = Image

    # Set constants needed by utility functions
    mod.SOURCE_FOLDER = "/app/ocr_source"
    mod.RESUME_MANIFEST_FILENAME = "resume_manifest.json"
    mod.RESUME_MANIFEST_SCHEMA_VERSION = 2
    mod.PDF_EXTENSIONS = {".pdf"}
    mod.VIDEO_EXTENSIONS = {
        ".mp4", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".mkv", ".webm",
    }
    mod.PHASE1_IMAGE_EXTENSIONS = {
        ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif",
        ".webp", ".jp2", ".jpx", ".pnm", ".pbm", ".pgm", ".ppm",
        ".pcx", ".ico", ".svg", ".svgz", ".wmf", ".emf",
    }
    mod.PHASE2_IMAGE_EXTENSIONS = {".heic", ".heif", ".avif", ".jxl", ".jxr", ".dcx", ".xps"}

    # Extract and compile individual functions
    import re

    function_names = [
        "read_file_header", "detect_magic_family", "classify_source_file",
        "_sanitize_path_segment", "build_output_rel_stem", "get_path_based_doc_id",
        "compute_source_fingerprint", "build_resume_doc_id", "_resume_manifest_path",
        "_load_resume_manifest", "_write_resume_manifest", "_reset_resume_temp_dir",
        "prepare_resume_state",
        "img_to_bytes", "to_plain_list", "extract_paddle_lines",
        "parse_structure_result", "_build_document_summary",
    ]

    for func_name in function_names:
        # Find the function definition in source
        pattern = rf'^(def {func_name}\(.*?\n(?:(?:    .*\n|[ \t]*\n)*))'
        match = re.search(pattern, source, re.MULTILINE)
        if match:
            func_source = match.group(1)
            try:
                exec(compile(func_source, src_path, "exec"), mod.__dict__)
            except Exception as e:
                print(f"Warning: Could not load {func_name}: {e}")

    return mod


# Try direct import first (works if all deps installed), fall back to extraction
try:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from ocr_gpu_async import (
        build_output_rel_stem,
        build_resume_doc_id,
        classify_source_file,
        compute_source_fingerprint,
        detect_magic_family,
        extract_paddle_lines,
        get_path_based_doc_id,
        img_to_bytes,
        prepare_resume_state,
        read_file_header,
        to_plain_list,
    )
    _DIRECT_IMPORT = True
except ImportError:
    _utils = _load_utility_functions()
    read_file_header = _utils.read_file_header
    detect_magic_family = _utils.detect_magic_family
    classify_source_file = _utils.classify_source_file
    build_output_rel_stem = _utils.build_output_rel_stem
    compute_source_fingerprint = _utils.compute_source_fingerprint
    build_resume_doc_id = _utils.build_resume_doc_id
    prepare_resume_state = _utils.prepare_resume_state
    get_path_based_doc_id = _utils.get_path_based_doc_id
    img_to_bytes = _utils.img_to_bytes
    to_plain_list = _utils.to_plain_list
    extract_paddle_lines = _utils.extract_paddle_lines
    _DIRECT_IMPORT = False


# ===========================================================================
# Tests: read_file_header
# ===========================================================================

class TestReadFileHeader:
    def test_reads_pdf_header(self):
        header = read_file_header(os.path.join(FIXTURES_DIR, "sample.pdf"))
        assert header.startswith(b"%PDF-")

    def test_reads_png_header(self):
        header = read_file_header(os.path.join(FIXTURES_DIR, "sample.png"))
        assert header.startswith(b"\x89PNG")

    def test_max_bytes_limit(self):
        header = read_file_header(os.path.join(FIXTURES_DIR, "sample.pdf"), max_bytes=5)
        assert len(header) == 5

    def test_nonexistent_file_returns_empty(self):
        header = read_file_header("/nonexistent/path/file.xyz")
        assert header == b""

    def test_empty_file(self):
        header = read_file_header(os.path.join(FIXTURES_DIR, "empty.pdf"))
        assert header == b""


# ===========================================================================
# Tests: detect_magic_family
# ===========================================================================

class TestDetectMagicFamily:
    @pytest.mark.parametrize("filename,expected", [
        ("sample.pdf", "pdf"),
        ("sample.png", "image"),
        ("sample.jpg", "image"),
        ("sample.tif", "image"),
        ("sample.bmp", "image"),
        ("sample.gif", "image"),
        ("sample.webp", "image"),
        ("sample.ico", "image"),
        ("sample.ppm", "image"),
        ("sample.svg", "image"),
        ("sample.jp2", "image"),
    ])
    def test_known_formats(self, filename, expected):
        path = os.path.join(FIXTURES_DIR, filename)
        assert detect_magic_family(path) == expected

    def test_unknown_format(self):
        assert detect_magic_family(os.path.join(FIXTURES_DIR, "data.csv")) is None

    def test_empty_file(self):
        assert detect_magic_family(os.path.join(FIXTURES_DIR, "empty.pdf")) is None

    def test_nonexistent(self):
        assert detect_magic_family("/no/such/file") is None


# ===========================================================================
# Tests: classify_source_file
# ===========================================================================

class TestClassifySourceFile:
    def test_valid_pdf(self):
        result, warning = classify_source_file(os.path.join(FIXTURES_DIR, "sample.pdf"))
        assert result == "pdf"
        assert warning is None

    def test_valid_png(self):
        result, warning = classify_source_file(os.path.join(FIXTURES_DIR, "sample.png"))
        assert result == "image"

    def test_valid_jpg(self):
        result, warning = classify_source_file(os.path.join(FIXTURES_DIR, "sample.jpg"))
        assert result == "image"

    def test_valid_tif(self):
        result, warning = classify_source_file(os.path.join(FIXTURES_DIR, "sample.tif"))
        assert result == "image"

    def test_valid_mp4(self, tmp_path):
        video = tmp_path / "sample.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
        result, warning = classify_source_file(str(video))
        assert result == "video"
        assert warning is None

    def test_signature_mismatch_rejected(self):
        """PDF extension but PNG magic bytes should be rejected."""
        result, reason = classify_source_file(os.path.join(FIXTURES_DIR, "mismatch.pdf"))
        assert result is None
        assert "mismatch" in reason.lower()

    def test_unsupported_extension(self):
        result, reason = classify_source_file(os.path.join(FIXTURES_DIR, "data.csv"))
        assert result is None
        assert "unsupported" in reason.lower()

    def test_phase2_extension_rejected(self):
        result, reason = classify_source_file(os.path.join(FIXTURES_DIR, "photo.heic"))
        assert result is None
        assert "phase-2" in reason.lower()

    def test_empty_pdf_accepted_by_extension_fallback(self):
        """Empty file with .pdf extension — no magic match, accepted by extension."""
        result, warning = classify_source_file(os.path.join(FIXTURES_DIR, "empty.pdf"))
        assert result == "pdf"
        assert "extension fallback" in warning.lower()

    def test_empty_pdf_rejected_in_strict_mode(self):
        result, reason = classify_source_file(
            os.path.join(FIXTURES_DIR, "empty.pdf"),
            strict_pdf_signature=True,
        )
        assert result is None
        assert "missing pdf signature" in reason.lower()

    def test_image_fallback_allowed_in_strict_mode(self):
        result, warning = classify_source_file(
            os.path.join(FIXTURES_DIR, "sample.jpg"),
            strict_pdf_signature=True,
        )
        assert result == "image"
        assert warning is None


# ===========================================================================
# Tests: build_output_rel_stem
# ===========================================================================

class TestBuildOutputRelStem:
    def test_pdf_returns_stem_only(self):
        # Note: this function uses SOURCE_FOLDER="/app/ocr_source"
        path = "/app/ocr_source/subfolder/document.pdf"
        result = build_output_rel_stem(path, "pdf")
        expected = os.path.join("subfolder", "document")
        assert result == expected

    def test_image_appends_ext_token(self):
        path = "/app/ocr_source/scans/page.tiff"
        result = build_output_rel_stem(path, "image")
        expected = os.path.join("scans", "page__tiff")
        assert result == expected

    def test_image_different_extension(self):
        path = "/app/ocr_source/photo.jpg"
        result = build_output_rel_stem(path, "image")
        assert result == "photo__jpg"

    def test_nested_path(self):
        path = "/app/ocr_source/a/b/c/deep.pdf"
        result = build_output_rel_stem(path, "pdf")
        expected = os.path.join("a", "b", "c", "deep")
        assert result == expected


# ===========================================================================
# Tests: get_path_based_doc_id (renamed from get_file_hash)
# ===========================================================================

class TestGetPathBasedDocId:
    def test_deterministic(self):
        h1 = get_path_based_doc_id("/some/path/file.pdf")
        h2 = get_path_based_doc_id("/some/path/file.pdf")
        assert h1 == h2

    def test_different_paths_different_hashes(self):
        h1 = get_path_based_doc_id("/path/a.pdf")
        h2 = get_path_based_doc_id("/path/b.pdf")
        assert h1 != h2

    def test_returns_16_char_hex(self):
        h = get_path_based_doc_id("/any/path.pdf")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_is_sha256_prefix(self):
        path = "/test/path.pdf"
        expected = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
        assert get_path_based_doc_id(path) == expected


# ===========================================================================
# Tests: resume integrity helpers
# ===========================================================================

class TestResumeIntegrityHelpers:
    def test_compute_source_fingerprint_includes_content_and_metadata(self, tmp_path):
        source = tmp_path / "source.pdf"
        source.write_bytes(b"%PDF-1.4\\nresume-test\\n%%EOF\\n")

        fp = compute_source_fingerprint(str(source))
        assert len(fp["content_sha256"]) == 64
        assert fp["size_bytes"] == source.stat().st_size
        assert fp["mtime_ns"] > 0

    def test_build_resume_doc_id_changes_when_fingerprint_changes(self):
        path = "/app/ocr_source/a/doc.pdf"
        fp_a = {"content_sha256": "a" * 64, "size_bytes": 100, "mtime_ns": 1}
        fp_b = {"content_sha256": "b" * 64, "size_bytes": 100, "mtime_ns": 1}
        id_a = build_resume_doc_id(path, fp_a)
        id_b = build_resume_doc_id(path, fp_b)
        assert id_a != id_b

    def test_prepare_resume_state_invalidates_stale_chunks(self, tmp_path):
        temp_dir = tmp_path / "resume-temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        stale_chunk = temp_dir / "1.pdf"
        stale_chunk.write_bytes(b"stale")

        old_fp = {"content_sha256": "a" * 64, "size_bytes": 100, "mtime_ns": 1}
        (temp_dir / "resume_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_path": "/app/ocr_source/doc.pdf",
                    "source_fingerprint": old_fp,
                },
            ),
            encoding="utf-8",
        )

        new_fp = {"content_sha256": "b" * 64, "size_bytes": 200, "mtime_ns": 2}
        result = prepare_resume_state(
            str(temp_dir),
            "/app/ocr_source/doc.pdf",
            new_fp,
        )

        assert result["status"] == "invalidated"
        assert not stale_chunk.exists()
        manifest = json.loads((temp_dir / "resume_manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_fingerprint"]["content_sha256"] == new_fp["content_sha256"]


# ===========================================================================
# Tests: img_to_bytes
# ===========================================================================

class TestImgToBytes:
    def test_converts_pil_to_jpeg_bytes(self):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        data = img_to_bytes(img)
        assert isinstance(data, bytes)
        assert data[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_output_is_valid_jpeg(self):
        img = Image.new("RGB", (50, 50), color=(0, 255, 0))
        data = img_to_bytes(img)
        reloaded = Image.open(io.BytesIO(data))
        assert reloaded.size == (50, 50)
        assert reloaded.format == "JPEG"


# ===========================================================================
# Tests: to_plain_list
# ===========================================================================

class TestToPlainList:
    def test_none_returns_empty(self):
        assert to_plain_list(None) == []

    def test_list_passthrough(self):
        assert to_plain_list([1, 2, 3]) == [1, 2, 3]

    def test_tuple_to_list(self):
        assert to_plain_list((1, 2, 3)) == [1, 2, 3]

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_plain_list(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_empty_list(self):
        assert to_plain_list([]) == []

    def test_other_type_returns_empty(self):
        assert to_plain_list("string") == []
        assert to_plain_list(42) == []


# ===========================================================================
# Tests: extract_paddle_lines
# ===========================================================================

class TestExtractPaddleLines:
    def test_empty_result(self):
        assert extract_paddle_lines(None) == []
        assert extract_paddle_lines([]) == []

    def test_v2_format(self):
        """PaddleOCR v2 returns list of [box, (text, confidence)]."""
        result = [[
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ("Hello World", 0.95)],
            [[[10, 40], [100, 40], [100, 60], [10, 60]], ("Second Line", 0.90)],
        ]]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2
        assert len(lines[0]) == 3  # (text, box, confidence) triple
        assert lines[0][0] == "Hello World"
        assert lines[1][0] == "Second Line"
        assert abs(lines[0][2] - 0.95) < 0.01
        assert abs(lines[1][2] - 0.90) < 0.01

    def test_dict_format(self):
        """PaddleOCR v3/PP-StructureV3 returns dict with rec_texts + dt_polys."""
        result = [{
            "rec_texts": ["Line One", "Line Two"],
            "dt_polys": [
                [[10, 10], [100, 10], [100, 30], [10, 30]],
                [[10, 40], [100, 40], [100, 60], [10, 60]],
            ],
        }]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2
        assert len(lines[0]) == 3  # (text, box, confidence) triple
        assert lines[0][0] == "Line One"
        assert lines[1][0] == "Line Two"
        # No rec_scores provided, confidence defaults to 0.0
        assert lines[0][2] == 0.0

    def test_dict_format_with_scores(self):
        """PaddleOCR v3 dict with rec_scores returns confidence values."""
        result = [{
            "rec_texts": ["Score A", "Score B"],
            "rec_scores": [0.92, 0.78],
            "dt_polys": [
                [[0, 0], [100, 0], [100, 30], [0, 30]],
                [[0, 40], [100, 40], [100, 70], [0, 70]],
            ],
        }]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2
        assert abs(lines[0][2] - 0.92) < 0.01
        assert abs(lines[1][2] - 0.78) < 0.01

    def test_dict_skips_empty_text(self):
        result = [{"rec_texts": ["Hello", "", "World"], "dt_polys": []}]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2

    def test_v2_skips_empty_text(self):
        result = [[
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("", 0.1)],
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("Real text", 0.9)],
        ]]
        lines = extract_paddle_lines(result)
        assert len(lines) == 1
        assert lines[0][0] == "Real text"
        assert abs(lines[0][2] - 0.9) < 0.01


# ===========================================================================
# Tests: optimize_pdfs.py
# ===========================================================================

class TestOptimizePdfs:
    """Tests for optimize_pdfs module (no Ghostscript required for import tests)."""

    def test_module_imports(self):
        import optimize_pdfs
        assert hasattr(optimize_pdfs, "optimize_pdf")
        assert hasattr(optimize_pdfs, "DEFAULT_QUALITY")

    def test_default_quality(self):
        import optimize_pdfs
        assert optimize_pdfs.DEFAULT_QUALITY == "/prepress"


# ===========================================================================
# Tests: download_models.py
# ===========================================================================

class TestDownloadModels:
    """Tests for download_models module constants."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_paddleocr(self):
        pytest.importorskip("paddleocr", reason="PaddleOCR not installed")

    def test_target_langs_count(self):
        """Verify the low-resource tranche expands the baked model baseline."""
        import download_models
        assert len(download_models.TARGET_LANGS) == 34

    def test_target_langs_includes_english(self):
        import download_models
        assert "en" in download_models.TARGET_LANGS

    def test_target_langs_includes_key_languages(self):
        import download_models
        for lang in ["ch", "japan", "korean", "fr", "german", "es", "ar", "fa", "ta", "ka"]:
            assert lang in download_models.TARGET_LANGS, f"Missing: {lang}"
