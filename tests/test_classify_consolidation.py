"""Tests for classify_source_file consolidation.

Verifies that the canonical ``classify_source_file`` in
``ocr_distributed.ocr_utils`` correctly handles both worker and coordinator
semantics via the ``include_coordinator_types`` parameter.
"""

import os

import pytest

from ocr_distributed.ocr_utils import (
    COORDINATOR_TEXT_EXTENSIONS,
    classify_source_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(directory, name, content=b""):
    """Create a file in *directory* with the given *name* and return its path."""
    path = os.path.join(directory, name)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# Worker mode (include_coordinator_types=False, the default)
# ---------------------------------------------------------------------------


class TestWorkerMode:
    """Default mode used by OCR workers -- text types are rejected."""

    def test_pdf_accepted(self, tmp_path):
        p = _touch(str(tmp_path), "doc.pdf", b"%PDF-1.4 fake")
        result, warning = classify_source_file(p)
        assert result == "pdf"

    def test_image_accepted(self, tmp_path):
        # PNG magic bytes
        p = _touch(str(tmp_path), "scan.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result, warning = classify_source_file(p)
        assert result == "image"

    def test_tiff_accepted(self, tmp_path):
        p = _touch(str(tmp_path), "scan.tiff", b"II\x2a\x00" + b"\x00" * 100)
        result, warning = classify_source_file(p)
        assert result == "image"

    def test_video_accepted(self, tmp_path):
        # ftyp magic at offset 4
        p = _touch(str(tmp_path), "clip.mp4", b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)
        result, warning = classify_source_file(p)
        assert result == "video"

    @pytest.mark.parametrize("ext", sorted(COORDINATOR_TEXT_EXTENSIONS))
    def test_text_extensions_rejected_by_default(self, tmp_path, ext):
        """Workers must NOT accept coordinator-only text types."""
        p = _touch(str(tmp_path), f"file{ext}", b"some content")
        result, reason = classify_source_file(p)
        assert result is None, f"Expected rejection for {ext}, got {result!r}"
        assert "Unsupported extension" in reason

    def test_unsupported_extension_rejected(self, tmp_path):
        p = _touch(str(tmp_path), "file.xyz", b"data")
        result, reason = classify_source_file(p)
        assert result is None
        assert "Unsupported extension" in reason

    def test_heic_phase2_rejected(self, tmp_path):
        p = _touch(str(tmp_path), "photo.heic", b"\x00" * 100)
        result, reason = classify_source_file(p)
        assert result is None
        assert "Phase-2" in reason


# ---------------------------------------------------------------------------
# Coordinator mode (include_coordinator_types=True)
# ---------------------------------------------------------------------------


class TestCoordinatorMode:
    """Coordinator-level classification -- text types are accepted."""

    @pytest.mark.parametrize("ext", sorted(COORDINATOR_TEXT_EXTENSIONS))
    def test_text_extensions_accepted(self, tmp_path, ext):
        """Coordinator should accept .txt, .md, .csv, .json as 'text'."""
        p = _touch(str(tmp_path), f"file{ext}", b"some content")
        result, warning = classify_source_file(p, include_coordinator_types=True)
        assert result == "text", f"Expected 'text' for {ext}, got {result!r}"
        assert warning is None

    def test_pdf_still_accepted(self, tmp_path):
        p = _touch(str(tmp_path), "doc.pdf", b"%PDF-1.4 fake")
        result, warning = classify_source_file(p, include_coordinator_types=True)
        assert result == "pdf"

    def test_image_still_accepted(self, tmp_path):
        p = _touch(str(tmp_path), "scan.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result, warning = classify_source_file(p, include_coordinator_types=True)
        assert result == "image"

    def test_video_still_accepted(self, tmp_path):
        p = _touch(str(tmp_path), "clip.mp4", b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)
        result, warning = classify_source_file(p, include_coordinator_types=True)
        assert result == "video"

    def test_unsupported_still_rejected(self, tmp_path):
        p = _touch(str(tmp_path), "file.xyz", b"data")
        result, reason = classify_source_file(p, include_coordinator_types=True)
        assert result is None
        assert "Unsupported extension" in reason

    def test_heic_phase2_still_rejected(self, tmp_path):
        p = _touch(str(tmp_path), "photo.heic", b"\x00" * 100)
        result, reason = classify_source_file(p, include_coordinator_types=True)
        assert result is None
        assert "Phase-2" in reason


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure the new parameter does not break existing call sites."""

    def test_default_is_false(self, tmp_path):
        """Calling without the param should behave like workers (no text)."""
        p = _touch(str(tmp_path), "data.csv", b"a,b,c")
        result, _ = classify_source_file(p)
        assert result is None

    def test_strict_pdf_still_works(self, tmp_path):
        """strict_pdf_signature param coexists with include_coordinator_types."""
        # A .pdf file with NO PDF magic bytes
        p = _touch(str(tmp_path), "fake.pdf", b"NOT A PDF")
        result, reason = classify_source_file(
            p, strict_pdf_signature=True, include_coordinator_types=True,
        )
        assert result is None
        assert "strict mode" in reason

    def test_signature_mismatch_still_detected(self, tmp_path):
        """Magic-byte mismatch detection is not broken by the new param."""
        # File named .png but containing PDF magic bytes
        p = _touch(str(tmp_path), "tricky.png", b"%PDF-1.4 fake")
        result, reason = classify_source_file(p, include_coordinator_types=True)
        assert result is None
        assert "mismatch" in reason.lower()


# ---------------------------------------------------------------------------
# COORDINATOR_TEXT_EXTENSIONS constant
# ---------------------------------------------------------------------------


class TestConstant:
    """Verify the exported constant matches expectations."""

    def test_expected_extensions(self):
        assert COORDINATOR_TEXT_EXTENSIONS == {".txt", ".md", ".csv", ".json"}
