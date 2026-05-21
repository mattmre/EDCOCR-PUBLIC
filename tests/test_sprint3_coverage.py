"""Sprint 3 coverage tests: T3 (_sanitize_table_html XSS) and T4 (optimize_pdfs).

T3: Validates that _sanitize_table_html strips dangerous HTML tags and attributes
    while preserving safe table-related elements.

T4: Validates optimize_pdf() behavior with mocked Ghostscript subprocess calls,
    including integrity checks and error handling.

Run with: python -m pytest tests/test_sprint3_coverage.py -v
"""
import os
import re
import subprocess
import types
from unittest.mock import MagicMock, patch

# Add project root to path


# ---------------------------------------------------------------------------
# Import _sanitize_table_html — same pattern as test_utilities.py
# ---------------------------------------------------------------------------

def _load_sanitize_function():
    """Extract _sanitize_table_html from ocr_gpu_async.py without running imports."""
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py"
    )
    with open(src_path, encoding="utf-8") as f:
        source = f.read()

    mod = types.ModuleType("sanitize_utils")
    mod.__file__ = src_path
    mod.re = re

    # Extract constants and function using line-based approach
    # This avoids regex issues with raw strings containing special characters
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("_SAFE_TABLE_TAGS"):
            exec(compile(line, src_path, "exec"), mod.__dict__)
        elif line.startswith("_HTML_TAG_RE"):
            exec(compile(line, src_path, "exec"), mod.__dict__)

    # Extract the function
    func_pattern = r'^(def _sanitize_table_html\(.*?\n(?:(?:    .*\n|[ \t]*\n)*))'
    func_match = re.search(func_pattern, source, re.MULTILINE)
    if func_match:
        exec(compile(func_match.group(1), src_path, "exec"), mod.__dict__)

    return mod


try:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from ocr_gpu_async import _sanitize_table_html
    _DIRECT_IMPORT = True
except ImportError:
    _mod = _load_sanitize_function()
    _sanitize_table_html = _mod._sanitize_table_html
    _DIRECT_IMPORT = False


# ===========================================================================
# T3: _sanitize_table_html XSS Tests
# ===========================================================================

class TestSanitizeTableHtml:
    """Tests for T3: _sanitize_table_html XSS prevention."""

    def test_preserves_table_tags(self):
        html = "<table><tr><td>data</td></tr></table>"
        result = _sanitize_table_html(html)
        assert "<table>" in result
        assert "<tr>" in result
        assert "<td>" in result
        assert "</td>" in result

    def test_strips_script_tags(self):
        html = '<table><tr><td><script>alert("xss")</script></td></tr></table>'
        result = _sanitize_table_html(html)
        assert "<script>" not in result
        assert "</script>" not in result
        assert "alert" in result  # text content survives

    def test_strips_img_tag(self):
        html = '<td><img src=x onerror=alert(1)></td>'
        result = _sanitize_table_html(html)
        assert "<img" not in result
        assert "onerror" not in result

    def test_strips_iframe(self):
        html = '<td><iframe src="https://evil.com"></iframe></td>'
        result = _sanitize_table_html(html)
        assert "<iframe" not in result

    def test_strips_event_handler_attributes(self):
        html = '<td onmouseover="alert(1)">data</td>'
        result = _sanitize_table_html(html)
        assert "onmouseover" not in result
        assert "<td>" in result  # tag preserved, attrs stripped

    def test_strips_style_tag(self):
        html = '<table><style>body{display:none}</style><tr><td>x</td></tr></table>'
        result = _sanitize_table_html(html)
        assert "<style>" not in result

    def test_strips_a_tag(self):
        html = '<td><a href="javascript:alert(1)">link</a></td>'
        result = _sanitize_table_html(html)
        assert "<a " not in result
        assert "javascript:" not in result
        assert "link" in result

    def test_strips_div_span(self):
        html = '<table><div><span>text</span></div></table>'
        result = _sanitize_table_html(html)
        assert "<div>" not in result
        assert "<span>" not in result
        assert "text" in result

    def test_preserves_all_safe_tags(self):
        html = (
            "<table><thead><tr><th>H</th></tr></thead>"
            "<tbody><tr><td>D</td></tr></tbody>"
            "<tfoot><tr><td>F</td></tr></tfoot></table>"
        )
        result = _sanitize_table_html(html)
        for tag in ["table", "thead", "tbody", "tfoot", "tr", "th", "td"]:
            assert f"<{tag}>" in result

    def test_strips_class_attribute(self):
        html = '<td class="highlight">data</td>'
        result = _sanitize_table_html(html)
        assert result == "<td>data</td>"

    def test_case_insensitive_xss(self):
        html = '<td><SCRIPT>alert(1)</SCRIPT></td>'
        result = _sanitize_table_html(html)
        # Script tags should be stripped regardless of case
        assert "<SCRIPT>" not in result
        assert "<script>" not in result

    def test_empty_input(self):
        assert _sanitize_table_html("") == ""

    def test_plain_text_unchanged(self):
        result = _sanitize_table_html("just plain text")
        assert result == "just plain text"

    def test_preserves_caption_tag(self):
        html = "<table><caption>Title</caption><tr><td>1</td></tr></table>"
        result = _sanitize_table_html(html)
        assert "<caption>" in result
        assert "Title" in result

    def test_preserves_colgroup_col_tags(self):
        html = '<table><colgroup><col span="2"></colgroup><tr><td>1</td></tr></table>'
        result = _sanitize_table_html(html)
        assert "<colgroup>" in result
        assert "<col>" in result
        # span attribute should be stripped
        assert 'span=' not in result

    def test_nested_dangerous_tags(self):
        html = '<td><div><script><b>evil</b></script></div></td>'
        result = _sanitize_table_html(html)
        assert "<script>" not in result
        assert "<div>" not in result
        assert "<b>" not in result
        assert "evil" in result

    def test_svg_stripped(self):
        html = '<td><svg onload="alert(1)"><circle r="50"/></svg></td>'
        result = _sanitize_table_html(html)
        assert "<svg" not in result
        assert "onload" not in result

    def test_form_elements_stripped(self):
        html = '<td><input type="text" value="xss"><button>Click</button></td>'
        result = _sanitize_table_html(html)
        assert "<input" not in result
        assert "<button>" not in result


# ===========================================================================
# T4: optimize_pdfs Tests
# ===========================================================================

class TestOptimizePdf:
    """Tests for T4: optimize_pdfs.optimize_pdf()."""

    def test_skips_already_optimized(self, tmp_path):
        pdf = tmp_path / "doc_optimized.pdf"
        pdf.write_bytes(b"%PDF-1.0")
        from optimize_pdfs import optimize_pdf
        with patch("optimize_pdfs.subprocess.run") as mock_run:
            optimize_pdf(str(pdf))
        mock_run.assert_not_called()

    def test_successful_optimization(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.0 " + b"x" * 10000)
        from optimize_pdfs import optimize_pdf

        def fake_gs(cmd, **kwargs):
            # Create a valid smaller output file
            for arg in cmd:
                if arg.startswith("-sOutputFile="):
                    out = arg.split("=", 1)[1]
                    with open(out, "wb") as f:
                        f.write(b"%PDF-1.0 " + b"y" * 5000)
            return MagicMock(returncode=0)

        with patch("optimize_pdfs.subprocess.run", side_effect=fake_gs):
            optimize_pdf(str(pdf))

        # Original should be replaced with optimized version
        assert pdf.exists()
        content = pdf.read_bytes()
        assert b"y" * 100 in content  # Has the optimized content

    def test_rejects_output_under_100_bytes(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        original = b"%PDF-1.0 " + b"x" * 10000
        pdf.write_bytes(original)
        from optimize_pdfs import optimize_pdf

        def fake_gs(cmd, **kwargs):
            for arg in cmd:
                if arg.startswith("-sOutputFile="):
                    out = arg.split("=", 1)[1]
                    with open(out, "wb") as f:
                        f.write(b"tiny")
            return MagicMock(returncode=0)

        with patch("optimize_pdfs.subprocess.run", side_effect=fake_gs):
            optimize_pdf(str(pdf))

        # Original should be untouched
        assert pdf.read_bytes() == original

    def test_rejects_invalid_pdf_header(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        original = b"%PDF-1.0 " + b"x" * 10000
        pdf.write_bytes(original)
        from optimize_pdfs import optimize_pdf

        def fake_gs(cmd, **kwargs):
            for arg in cmd:
                if arg.startswith("-sOutputFile="):
                    out = arg.split("=", 1)[1]
                    with open(out, "wb") as f:
                        f.write(b"NOT-A-PDF " + b"x" * 200)
            return MagicMock(returncode=0)

        with patch("optimize_pdfs.subprocess.run", side_effect=fake_gs):
            optimize_pdf(str(pdf))

        assert pdf.read_bytes() == original

    def test_handles_ghostscript_failure(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.0 " + b"x" * 1000)
        from optimize_pdfs import optimize_pdf

        with patch(
            "optimize_pdfs.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "gs", stderr=b"error"),
        ):
            optimize_pdf(str(pdf))  # Should not raise

        assert pdf.exists()  # Original preserved

    def test_handles_no_output_file(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.0 " + b"x" * 1000)
        from optimize_pdfs import optimize_pdf

        with patch(
            "optimize_pdfs.subprocess.run",
            return_value=MagicMock(returncode=0),
        ):
            optimize_pdf(str(pdf))  # No output file created by mock

        assert pdf.exists()  # Original preserved

    def test_temp_file_cleaned_on_rejection(self, tmp_path):
        """When output is rejected (too small), temp file should be removed."""
        pdf = tmp_path / "doc.pdf"
        original = b"%PDF-1.0 " + b"x" * 10000
        pdf.write_bytes(original)
        temp_path = str(pdf) + ".tmp.pdf"
        from optimize_pdfs import optimize_pdf

        def fake_gs(cmd, **kwargs):
            for arg in cmd:
                if arg.startswith("-sOutputFile="):
                    out = arg.split("=", 1)[1]
                    with open(out, "wb") as f:
                        f.write(b"tiny")
            return MagicMock(returncode=0)

        with patch("optimize_pdfs.subprocess.run", side_effect=fake_gs):
            optimize_pdf(str(pdf))

        # Temp file should have been cleaned up
        assert not os.path.exists(temp_path)

    def test_temp_file_cleaned_on_gs_error(self, tmp_path):
        """When Ghostscript fails, temp file should be removed if it exists."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.0 " + b"x" * 1000)
        temp_path = str(pdf) + ".tmp.pdf"
        from optimize_pdfs import optimize_pdf

        def fake_gs_with_temp(cmd, **kwargs):
            # Create temp file then raise error
            for arg in cmd:
                if arg.startswith("-sOutputFile="):
                    out = arg.split("=", 1)[1]
                    with open(out, "wb") as f:
                        f.write(b"partial")
            raise subprocess.CalledProcessError(1, "gs", stderr=b"error")

        with patch("optimize_pdfs.subprocess.run", side_effect=fake_gs_with_temp):
            optimize_pdf(str(pdf))

        # Temp file should have been cleaned up
        assert not os.path.exists(temp_path)

    def test_custom_quality_parameter(self, tmp_path):
        """Custom quality parameter is passed to Ghostscript command."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.0 " + b"x" * 1000)
        from optimize_pdfs import optimize_pdf

        captured_cmd = []

        def capture_gs(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return MagicMock(returncode=0)

        with patch("optimize_pdfs.subprocess.run", side_effect=capture_gs):
            optimize_pdf(str(pdf), quality="/ebook")

        assert any("/ebook" in arg for arg in captured_cmd)

    def test_default_quality_is_prepress(self):
        """Default quality setting should be /prepress."""
        from optimize_pdfs import DEFAULT_QUALITY
        assert DEFAULT_QUALITY == "/prepress"

    def test_non_optimized_suffix_processed(self, tmp_path):
        """Files without _optimized suffix should be processed."""
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.0 " + b"x" * 1000)
        from optimize_pdfs import optimize_pdf

        with patch("optimize_pdfs.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            optimize_pdf(str(pdf))

        mock_run.assert_called_once()
