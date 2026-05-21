"""Tests for Phase 3B Table Export functionality.

Run with: python -m pytest tests/test_table_export.py -v
"""
import csv
import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _load_table_functions():
    """Extract table export functions from ocr_gpu_async.py without GPU imports."""
    import re
    import types

    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py")
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    mod = types.ModuleType("table_export_utils")
    mod.__file__ = src_path
    mod.os = os
    mod.csv = __import__("csv")
    import logging
    mod.logger = logging.getLogger("test_table_export")
    mod.SOURCE_FOLDER = "/app/ocr_source"

    # Extract HTMLTableParser class
    cls_pattern = r'^(class HTMLTableParser\(.*?\n(?:(?:    .*\n|[ \t]*\n)*))'
    cls_match = re.search(cls_pattern, source, re.MULTILINE)
    if cls_match:
        from html.parser import HTMLParser
        mod.HTMLParser = HTMLParser
        exec(compile(cls_match.group(1), src_path, "exec"), mod.__dict__)

    # Constants needed by _sanitize_table_html
    mod.re = re
    mod._SAFE_TABLE_TAGS = frozenset({"table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup", "col"})
    mod._HTML_TAG_RE = re.compile(r"<(/?)(\w+)([^>]*)>", re.DOTALL)

    # Extract functions (order matters — dependencies first)
    for func_name in ["_sanitize_path_segment", "_sanitize_table_html", "html_table_to_csv_rows", "write_extracted_tables"]:
        pattern = rf'^(def {func_name}\(.*?\n(?:(?:    .*\n|[ \t]*\n)*))'
        match = re.search(pattern, source, re.MULTILINE)
        if match:
            try:
                exec(compile(match.group(1), src_path, "exec"), mod.__dict__)
            except Exception as e:
                print(f"Warning: Could not load {func_name}: {e}")
    return mod


try:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from ocr_gpu_async import (
        HTMLTableParser,
        html_table_to_csv_rows,
        write_extracted_tables,
    )
    _DIRECT_IMPORT = True
except ImportError:
    _utils = _load_table_functions()
    HTMLTableParser = _utils.HTMLTableParser
    html_table_to_csv_rows = _utils.html_table_to_csv_rows
    write_extracted_tables = _utils.write_extracted_tables
    _DIRECT_IMPORT = False


# ===========================================================================
# Tests: HTMLTableParser / html_table_to_csv_rows
# ===========================================================================

class TestHTMLTableToCsvRows:

    def test_simple_table(self):
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        rows = html_table_to_csv_rows(html)
        assert rows == [["A", "B"]]

    def test_multi_row(self):
        html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>"
        rows = html_table_to_csv_rows(html)
        assert len(rows) == 2
        assert rows[0] == ["Name", "Age"]
        assert rows[1] == ["Alice", "30"]

    def test_whitespace_stripped(self):
        html = "<table><tr><td>  X  </td><td>\n Y \n</td></tr></table>"
        rows = html_table_to_csv_rows(html)
        assert rows == [["X", "Y"]]

    def test_empty_cells(self):
        html = "<table><tr><td></td><td>Z</td></tr></table>"
        rows = html_table_to_csv_rows(html)
        assert rows == [["", "Z"]]

    def test_none_input(self):
        rows = html_table_to_csv_rows(None)
        assert rows == []

    def test_empty_string(self):
        rows = html_table_to_csv_rows("")
        assert rows == []

    def test_malformed_html(self):
        rows = html_table_to_csv_rows("<table><tr><td>Incomplete")
        assert isinstance(rows, list)


# ===========================================================================
# Tests: write_extracted_tables
# ===========================================================================

class TestWriteExtractedTables:

    @pytest.fixture
    def source_folder_patch(self):
        """Patch SOURCE_FOLDER for write_extracted_tables."""
        try:
            import ocr_gpu_async
            original = ocr_gpu_async.SOURCE_FOLDER
            ocr_gpu_async.SOURCE_FOLDER = "/app/ocr_source"
            yield
            ocr_gpu_async.SOURCE_FOLDER = original
        except ImportError:
            yield  # Non-direct import uses module stub

    def test_no_tables_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = [{"page_num": 1, "tables": [], "layout_regions": []}]
            count = write_extracted_tables("doc1", "/app/ocr_source/test.pdf", pages, tmpdir)
            assert count == 0

    def test_single_table_creates_files(self, source_folder_patch):
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = [{
                "page_num": 1,
                "tables": [{"html": "<table><tr><td>A</td><td>B</td></tr></table>", "cell_bbox": []}],
                "layout_regions": [],
            }]
            count = write_extracted_tables("doc1", "/app/ocr_source/report.pdf", pages, tmpdir)
            assert count == 1
            tables_dir = os.path.join(tmpdir, "EXPORT", "TABLES")
            assert os.path.exists(os.path.join(tables_dir, "report_p1_t0.html"))
            assert os.path.exists(os.path.join(tables_dir, "report_p1_t0.csv"))

    def test_csv_content_correct(self, source_folder_patch):
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = [{
                "page_num": 1,
                "tables": [{"html": "<table><tr><td>X</td><td>Y</td></tr><tr><td>1</td><td>2</td></tr></table>", "cell_bbox": []}],
                "layout_regions": [],
            }]
            write_extracted_tables("doc1", "/app/ocr_source/test.pdf", pages, tmpdir)
            csv_path = os.path.join(tmpdir, "EXPORT", "TABLES", "test_p1_t0.csv")
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            assert rows == [["X", "Y"], ["1", "2"]]

    def test_extracted_files_added_to_structure(self, source_folder_patch):
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = [{
                "page_num": 1,
                "tables": [{"html": "<table><tr><td>A</td></tr></table>", "cell_bbox": []}],
                "layout_regions": [],
            }]
            write_extracted_tables("doc1", "/app/ocr_source/test.pdf", pages, tmpdir)
            ef = pages[0]["tables"][0].get("extracted_files")
            assert ef is not None
            assert "html" in ef
            assert "csv" in ef

    def test_multi_page_multi_table(self, source_folder_patch):
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = [
                {"page_num": 1, "tables": [
                    {"html": "<table><tr><td>T1</td></tr></table>", "cell_bbox": []},
                    {"html": "<table><tr><td>T2</td></tr></table>", "cell_bbox": []},
                ], "layout_regions": []},
                {"page_num": 3, "tables": [
                    {"html": "<table><tr><td>T3</td></tr></table>", "cell_bbox": []},
                ], "layout_regions": []},
            ]
            count = write_extracted_tables("doc1", "/app/ocr_source/doc.pdf", pages, tmpdir)
            assert count == 3
            tables_dir = os.path.join(tmpdir, "EXPORT", "TABLES")
            assert os.path.exists(os.path.join(tables_dir, "doc_p1_t0.html"))
            assert os.path.exists(os.path.join(tables_dir, "doc_p1_t1.html"))
            assert os.path.exists(os.path.join(tables_dir, "doc_p3_t0.html"))

    def test_skips_empty_html(self, source_folder_patch):
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = [{
                "page_num": 1,
                "tables": [{"html": "", "cell_bbox": []}, {"html": None, "cell_bbox": []}],
                "layout_regions": [],
            }]
            count = write_extracted_tables("doc1", "/app/ocr_source/test.pdf", pages, tmpdir)
            assert count == 0
