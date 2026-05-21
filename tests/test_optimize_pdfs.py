"""Tests for optimize_pdfs.py.

Covers Ghostscript compression, timeout handling, integrity validation,
batch processing, and skip logic.
"""

import os
import subprocess
import tempfile
from unittest.mock import MagicMock, mock_open, patch

from optimize_pdfs import (
    DEFAULT_QUALITY,
    GHOSTSCRIPT_TIMEOUT,
    VALID_QUALITIES,
    main,
    optimize_pdf,
)


class TestOptimizePdfGhostscriptArgs:
    """Verify Ghostscript subprocess invocation arguments."""

    @patch("optimize_pdfs.os.remove")
    @patch("optimize_pdfs.os.rename")
    @patch("builtins.open", mock_open(read_data=b"%PDF-1.4"))
    @patch("optimize_pdfs.os.path.getsize")
    @patch("optimize_pdfs.os.path.exists", return_value=True)
    @patch("optimize_pdfs.subprocess.run")
    def test_ghostscript_args_include_safety_flags(
        self, mock_run, mock_exists, mock_getsize, mock_rename, mock_remove
    ):
        """Ghostscript command includes expected device, compat, and quality flags."""
        mock_getsize.side_effect = [10000, 5000]  # original, new

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            optimize_pdf(pdf_path, "/prepress")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "gs"
            assert "-dSAFER" in cmd, "Ghostscript must run with -dSAFER sandbox"
            assert "-sDEVICE=pdfwrite" in cmd
            assert "-dCompatibilityLevel=1.4" in cmd
            assert "-dPDFSETTINGS=/prepress" in cmd
            assert "-dNOPAUSE" in cmd
            assert "-dBATCH" in cmd
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    @patch("optimize_pdfs.os.remove")
    @patch("optimize_pdfs.os.rename")
    @patch("builtins.open", mock_open(read_data=b"%PDF-1.4"))
    @patch("optimize_pdfs.os.path.getsize")
    @patch("optimize_pdfs.os.path.exists", return_value=True)
    @patch("optimize_pdfs.subprocess.run")
    def test_dsafer_precedes_sdevice(
        self, mock_run, mock_exists, mock_getsize, mock_rename, mock_remove
    ):
        """-dSAFER must appear before -sDEVICE to sandbox Ghostscript."""
        mock_getsize.side_effect = [10000, 5000]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            optimize_pdf(pdf_path)
            cmd = mock_run.call_args[0][0]
            safer_idx = cmd.index("-dSAFER")
            device_idx = cmd.index("-sDEVICE=pdfwrite")
            assert safer_idx < device_idx, "-dSAFER must come before -sDEVICE"
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    @patch("optimize_pdfs.subprocess.run")
    def test_timeout_passed_to_subprocess(self, mock_run):
        """Ghostscript subprocess receives the configured timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gs", timeout=GHOSTSCRIPT_TIMEOUT)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            optimize_pdf(pdf_path)
            _, kwargs = mock_run.call_args
            assert kwargs["timeout"] == GHOSTSCRIPT_TIMEOUT
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)


class TestOptimizePdfTimeoutHandling:
    """Verify graceful handling of Ghostscript timeout."""

    @patch("optimize_pdfs.os.path.exists", return_value=False)
    @patch("optimize_pdfs.subprocess.run")
    def test_timeout_does_not_crash(self, mock_run, mock_exists):
        """TimeoutExpired is caught -- function returns without raising."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gs", timeout=300)
        # Should not raise
        optimize_pdf("/fake/test.pdf")

    @patch("optimize_pdfs.os.remove")
    @patch("optimize_pdfs.os.path.exists", return_value=True)
    @patch("optimize_pdfs.subprocess.run")
    def test_timeout_cleans_temp_file(self, mock_run, mock_exists, mock_remove):
        """Temp file is removed after timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gs", timeout=300)
        optimize_pdf("/fake/test.pdf")
        mock_remove.assert_called_once_with("/fake/test.pdf.tmp.pdf")


class TestOptimizePdfOriginalPreservation:
    """Verify original file is preserved on failure conditions."""

    @patch("optimize_pdfs.os.path.exists", return_value=False)
    @patch("optimize_pdfs.subprocess.run")
    def test_original_preserved_on_gs_failure(self, mock_run, mock_exists):
        """On CalledProcessError, original file is not touched."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="gs", stderr=b"error"
        )
        # Should not raise
        optimize_pdf("/fake/doc.pdf")

    @patch("optimize_pdfs.os.remove")
    @patch("optimize_pdfs.os.path.exists", return_value=True)
    @patch("optimize_pdfs.os.path.getsize")
    @patch("optimize_pdfs.subprocess.run")
    def test_original_preserved_on_tiny_output(
        self, mock_run, mock_getsize, mock_exists, mock_remove
    ):
        """Output smaller than 100 bytes is rejected -- original kept."""
        mock_getsize.side_effect = [10000, 50]  # original=10k, output=50 bytes

        optimize_pdf("/fake/doc.pdf")

        # Temp file should be removed, but no rename (original kept)
        mock_remove.assert_called_once_with("/fake/doc.pdf.tmp.pdf")

    @patch("optimize_pdfs.os.remove")
    @patch("builtins.open", mock_open(read_data=b"NOT-A-PDF"))
    @patch("optimize_pdfs.os.path.exists", return_value=True)
    @patch("optimize_pdfs.os.path.getsize")
    @patch("optimize_pdfs.subprocess.run")
    def test_original_preserved_on_invalid_pdf_header(
        self, mock_run, mock_getsize, mock_exists, mock_remove
    ):
        """Output that is not a valid PDF (wrong header) is rejected."""
        mock_getsize.side_effect = [10000, 5000]  # sizes look fine

        optimize_pdf("/fake/doc.pdf")

        # Temp file removed, original kept
        mock_remove.assert_called_once_with("/fake/doc.pdf.tmp.pdf")


class TestOptimizePdfBatch:
    """Test batch processing via main()."""

    @patch("optimize_pdfs.optimize_pdf")
    @patch("optimize_pdfs.os.walk")
    def test_batch_processes_multiple_files(self, mock_walk, mock_optimize):
        """main() calls optimize_pdf for each .pdf file found."""
        mock_walk.return_value = [
            ("/app/ocr_output", [], ["doc1.pdf", "doc2.pdf", "readme.txt"]),
        ]

        with patch("optimize_pdfs.argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(quality="/prepress")
            main()

        assert mock_optimize.call_count == 2
        # Use os.path.join for platform-independent path matching
        mock_optimize.assert_any_call(
            os.path.join("/app/ocr_output", "doc1.pdf"), "/prepress"
        )
        mock_optimize.assert_any_call(
            os.path.join("/app/ocr_output", "doc2.pdf"), "/prepress"
        )


class TestOptimizePdfSkipLogic:
    """Test conditions that cause files to be skipped."""

    def test_skip_already_optimized_files(self):
        """Files ending with _optimized.pdf are skipped."""
        with patch("optimize_pdfs.subprocess.run") as mock_run:
            optimize_pdf("/fake/doc_optimized.pdf")
            mock_run.assert_not_called()

    def test_invalid_quality_rejected(self):
        """Invalid quality setting causes early return."""
        with patch("optimize_pdfs.subprocess.run") as mock_run:
            optimize_pdf("/fake/doc.pdf", "/invalid_quality")
            mock_run.assert_not_called()

    @patch("optimize_pdfs.os.remove")
    @patch("optimize_pdfs.os.path.exists")
    @patch("optimize_pdfs.os.path.getsize", return_value=0)
    @patch("optimize_pdfs.subprocess.run")
    def test_skip_zero_byte_file(self, mock_run, mock_getsize, mock_exists, mock_remove):
        """Zero-byte original file is skipped after Ghostscript runs."""
        mock_exists.return_value = True
        optimize_pdf("/fake/empty.pdf")
        # Temp file cleaned up
        mock_remove.assert_called_once_with("/fake/empty.pdf.tmp.pdf")


class TestOptimizePdfConstants:
    """Verify module-level constants are sensible."""

    def test_default_quality_is_valid(self):
        assert DEFAULT_QUALITY in VALID_QUALITIES

    def test_timeout_is_positive(self):
        assert GHOSTSCRIPT_TIMEOUT > 0

    def test_valid_qualities_set(self):
        expected = {"/screen", "/ebook", "/printer", "/prepress"}
        assert VALID_QUALITIES == expected
