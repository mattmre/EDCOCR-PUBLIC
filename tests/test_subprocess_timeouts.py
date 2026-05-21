"""Tests for subprocess timeout safety (, , ).

Validates that Ghostscript, Tesseract, and Poppler calls have timeouts
and handle subprocess.TimeoutExpired gracefully without crashing threads.

Run with: python -m pytest tests/test_subprocess_timeouts.py -v
"""

import logging
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Add project root to path


# ---------------------------------------------------------------------------
# Ghostscript timeout in optimize_pdfs.py
# ---------------------------------------------------------------------------


class TestGhostscriptTimeout:
    """Verify optimize_pdf handles Ghostscript subprocess timeout."""

    def test_timeout_logs_warning_and_preserves_original(self, tmp_path, caplog):
        """TimeoutExpired should log WARNING and keep original file intact."""
        pdf = tmp_path / "doc.pdf"
        original_content = b"%PDF-1.0 " + b"x" * 10000
        pdf.write_bytes(original_content)

        from optimize_pdfs import GHOSTSCRIPT_TIMEOUT, optimize_pdf

        with patch(
            "optimize_pdfs.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="gs", timeout=GHOSTSCRIPT_TIMEOUT),
        ):
            with caplog.at_level(logging.WARNING):
                optimize_pdf(str(pdf))

        # Original file must be preserved
        assert pdf.exists()
        assert pdf.read_bytes() == original_content

        # Warning must be logged
        assert any("timed out" in msg.lower() for msg in caplog.messages)

    def test_timeout_cleans_up_temp_file(self, tmp_path):
        """TimeoutExpired should remove any partial temp file."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.0 " + b"x" * 1000)
        temp_path = str(pdf) + ".tmp.pdf"

        from optimize_pdfs import GHOSTSCRIPT_TIMEOUT, optimize_pdf

        def fake_gs_with_temp(cmd, **kwargs):
            # Simulate partial temp file creation before timeout
            for arg in cmd:
                if str(arg).startswith("-sOutputFile="):
                    out = arg.split("=", 1)[1]
                    with open(out, "wb") as f:
                        f.write(b"partial")
            raise subprocess.TimeoutExpired(cmd="gs", timeout=GHOSTSCRIPT_TIMEOUT)

        with patch("optimize_pdfs.subprocess.run", side_effect=fake_gs_with_temp):
            optimize_pdf(str(pdf))

        # Temp file must be cleaned up
        assert not os.path.exists(temp_path)
        # Original preserved
        assert pdf.exists()

    def test_timeout_constant_is_300(self):
        """GHOSTSCRIPT_TIMEOUT should be 300 seconds."""
        from optimize_pdfs import GHOSTSCRIPT_TIMEOUT

        assert GHOSTSCRIPT_TIMEOUT == 300

    def test_subprocess_run_called_with_timeout(self, tmp_path):
        """subprocess.run must receive timeout=GHOSTSCRIPT_TIMEOUT."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.0 " + b"x" * 1000)

        from optimize_pdfs import GHOSTSCRIPT_TIMEOUT, optimize_pdf

        with patch("optimize_pdfs.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            optimize_pdf(str(pdf))

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == GHOSTSCRIPT_TIMEOUT


# ---------------------------------------------------------------------------
# Tesseract timeout in ocr_gpu_async.py
# ---------------------------------------------------------------------------


class TestTesseractTimeout:
    """Verify TESSERACT_TIMEOUT constant exists and is configured."""

    def test_tesseract_timeout_constant_exists(self):
        """TESSERACT_TIMEOUT should be importable and set to 120."""
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py"), encoding="utf-8") as f:
            source = f.read()

        # Verify the constant definition exists in source
        assert "TESSERACT_TIMEOUT" in source

        # Verify the timeout kwarg is passed to image_to_data
        assert "timeout=TESSERACT_TIMEOUT" in source

    def test_tesseract_timeout_default_value(self):
        """TESSERACT_TIMEOUT default should be 120 seconds."""
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py"), encoding="utf-8") as f:
            source = f.read()

        # Find the line defining TESSERACT_TIMEOUT
        for line in source.splitlines():
            if line.startswith("TESSERACT_TIMEOUT"):
                # Should contain 120 as default
                assert "120" in line
                break
        else:
            pytest.fail("TESSERACT_TIMEOUT definition not found at module level")


# ---------------------------------------------------------------------------
# Poppler timeout in ocr_gpu_async.py and dpi_escalation.py
# ---------------------------------------------------------------------------


class TestPopplerTimeoutOcrGpuAsync:
    """Verify Poppler timeout is wired in ocr_gpu_async.py."""

    def test_poppler_timeout_constant_exists(self):
        """POPPLER_TIMEOUT should be defined in ocr_gpu_async.py."""
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py"), encoding="utf-8") as f:
            source = f.read()

        assert "POPPLER_TIMEOUT" in source

    def test_convert_from_path_has_timeout(self):
        """All convert_from_path calls must include timeout=POPPLER_TIMEOUT."""
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py"), encoding="utf-8") as f:
            source = f.read()

        # Count occurrences of the actual function invocations vs timeout usage.
        invocation_count = source.count("images = convert_from_path(")
        timeout_count = source.count("timeout=POPPLER_TIMEOUT")

        assert invocation_count > 0, "No convert_from_path invocations found"
        assert timeout_count >= invocation_count, (
            f"Found {invocation_count} convert_from_path invocations but only "
            f"{timeout_count} timeout=POPPLER_TIMEOUT usages"
        )

    def test_poppler_timeout_default_value(self):
        """POPPLER_TIMEOUT default should be 300 seconds."""
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py"), encoding="utf-8") as f:
            source = f.read()

        for line in source.splitlines():
            if line.startswith("POPPLER_TIMEOUT"):
                assert "300" in line
                break
        else:
            pytest.fail("POPPLER_TIMEOUT definition not found at module level")


class TestPopplerTimeoutDpiEscalation:
    """Verify Poppler timeout is wired in dpi_escalation.py."""

    def test_poppler_timeout_constant_exists(self):
        """POPPLER_TIMEOUT should be defined in dpi_escalation.py."""
        from dpi_escalation import POPPLER_TIMEOUT

        assert POPPLER_TIMEOUT == 300

    @patch("dpi_escalation._convert_from_path")
    def test_timeout_kwarg_passed(self, mock_convert):
        """convert_from_path should receive timeout=POPPLER_TIMEOUT."""
        mock_img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [mock_img]

        from dpi_escalation import POPPLER_TIMEOUT, re_extract_page_at_dpi

        re_extract_page_at_dpi("/fake/doc.pdf", 1, 450)

        mock_convert.assert_called_once_with(
            "/fake/doc.pdf", first_page=1, last_page=1, dpi=450,
            timeout=POPPLER_TIMEOUT,
        )

    @patch("dpi_escalation._convert_from_path")
    def test_poppler_timeout_returns_none(self, mock_convert, caplog):
        """Poppler timeout should return None and log warning (not crash)."""
        mock_convert.side_effect = Exception("Timeout expired")

        from dpi_escalation import re_extract_page_at_dpi

        with caplog.at_level(logging.WARNING):
            result = re_extract_page_at_dpi("/fake/doc.pdf", 1, 450)

        assert result is None
        assert any("failed" in msg.lower() for msg in caplog.messages)
