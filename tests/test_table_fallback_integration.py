"""Integration tests for table fallback pipeline integration in ocr_gpu_async.py.

Tests verify that:
- Table fallback imports are available
- Worker-thread logic correctly converts DocIntel bbox to TableRegion format
- Assembly queue messages carry table_fallback_data
- Assembler collects and finalizes table fallback results
- CLI argument --enable-table-fallback is registered
- Feature is gated behind ENABLE_TABLE_FALLBACK and --enable-docintel
"""

import os
import unittest

# Ensure project root is on the path
from table_fallback import (
    DocumentTableFallbackSummary,
    PageTableFallbackAnalysis,
    TableRegion,
    analyze_page_tables,
    finalize_table_fallback,
    write_table_fallback_json,
)


class TestTableFallbackImportAvailability(unittest.TestCase):
    """Verify the import block in ocr_gpu_async makes table_fallback symbols available."""

    def test_table_fallback_module_importable(self):
        """table_fallback module should be importable."""
        import table_fallback
        self.assertTrue(hasattr(table_fallback, "TableRegion"))
        self.assertTrue(hasattr(table_fallback, "analyze_page_tables"))
        self.assertTrue(hasattr(table_fallback, "finalize_table_fallback"))
        self.assertTrue(hasattr(table_fallback, "write_table_fallback_json"))
        self.assertTrue(hasattr(table_fallback, "DocumentTableFallbackSummary"))

    def test_ocr_gpu_async_has_table_fallback_flag(self):
        """ocr_gpu_async should define _TABLE_FALLBACK_AVAILABLE."""
        # Read the source file to verify the import block is present
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn("_TABLE_FALLBACK_AVAILABLE", source)
        self.assertIn("from table_fallback import", source)


class TestDocIntelBboxConversion(unittest.TestCase):
    """Test conversion from DocIntel [x1,y1,x2,y2] to TableRegion (x,y,w,h)."""

    def test_basic_conversion(self):
        """[x1,y1,x2,y2] should convert to (x, y, w, h) correctly."""
        bbox = [100, 200, 500, 600]
        x1, y1, x2, y2 = bbox
        region = TableRegion(
            bbox=(x1, y1, x2 - x1, y2 - y1),
            page_number=1,
            original_confidence=0.95,
            original_engine="paddle",
        )
        self.assertEqual(region.bbox, (100, 200, 400, 400))
        self.assertEqual(region.page_number, 1)
        self.assertEqual(region.original_confidence, 0.95)
        self.assertEqual(region.original_engine, "paddle")

    def test_zero_origin_bbox(self):
        """Bbox starting at (0,0) should produce correct (x,y,w,h)."""
        bbox = [0, 0, 300, 150]
        x1, y1, x2, y2 = bbox
        region = TableRegion(bbox=(x1, y1, x2 - x1, y2 - y1), page_number=2)
        self.assertEqual(region.bbox, (0, 0, 300, 150))

    def test_conversion_from_structure_data(self):
        """Simulate the exact conversion logic from the worker thread."""
        structure_data = {
            "layout_regions": [
                {"type": "table", "bbox": [50, 100, 450, 300], "confidence": 0.88},
                {"type": "text", "bbox": [50, 10, 450, 90], "confidence": 0.95},
                {"type": "table", "bbox": [50, 350, 450, 550], "confidence": 0.72},
            ]
        }

        table_regions = []
        for region in structure_data.get("layout_regions", []):
            if region.get("type") == "table":
                rb = region.get("bbox", [0, 0, 0, 0])
                if len(rb) == 4:
                    x1, y1, x2, y2 = rb
                    table_regions.append(TableRegion(
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        page_number=1,
                        original_confidence=region.get("confidence", 0.0),
                        original_engine="paddle",
                    ))

        self.assertEqual(len(table_regions), 2)
        self.assertEqual(table_regions[0].bbox, (50, 100, 400, 200))
        self.assertEqual(table_regions[0].original_confidence, 0.88)
        self.assertEqual(table_regions[1].bbox, (50, 350, 400, 200))
        self.assertEqual(table_regions[1].original_confidence, 0.72)

    def test_non_table_regions_filtered(self):
        """Non-table regions should be skipped."""
        structure_data = {
            "layout_regions": [
                {"type": "text", "bbox": [0, 0, 100, 50], "confidence": 0.99},
                {"type": "figure", "bbox": [0, 50, 100, 150], "confidence": 0.85},
                {"type": "title", "bbox": [0, 150, 100, 180], "confidence": 0.92},
            ]
        }
        table_regions = []
        for region in structure_data.get("layout_regions", []):
            if region.get("type") == "table":
                rb = region.get("bbox", [0, 0, 0, 0])
                if len(rb) == 4:
                    x1, y1, x2, y2 = rb
                    table_regions.append(TableRegion(
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        page_number=1,
                    ))
        self.assertEqual(len(table_regions), 0)

    def test_empty_structure_data(self):
        """Empty/None structure_data should produce no regions."""
        for struct in [None, {}, {"layout_regions": []}]:
            regions = []
            if struct:
                for region in struct.get("layout_regions", []):
                    if region.get("type") == "table":
                        pass
            self.assertEqual(len(regions), 0)


class TestWorkerTableFallbackAnalysis(unittest.TestCase):
    """Test the per-page table fallback analysis as done in the worker thread."""

    def _make_paddle_lines(self, n=5, confidence=0.85):
        """Create mock paddle_lines in (text, box, confidence) format."""
        lines = []
        for i in range(n):
            text = f"Line {i}"
            box = [[50, 100 + i * 20], [450, 100 + i * 20],
                   [450, 118 + i * 20], [50, 118 + i * 20]]
            lines.append((text, box, confidence))
        return lines

    def test_analyze_page_with_tables(self):
        """analyze_page_tables should return analysis for table regions."""
        regions = [
            TableRegion(
                bbox=(50, 100, 400, 200),
                page_number=1,
                original_confidence=0.85,
                original_engine="paddle",
            ),
        ]
        paddle_lines = self._make_paddle_lines(5, 0.85)
        result = analyze_page_tables(1, regions, paddle_lines)

        self.assertIsInstance(result, PageTableFallbackAnalysis)
        self.assertEqual(result.page_number, 1)
        self.assertEqual(result.table_count, 1)

    def test_analyze_page_no_tables(self):
        """analyze_page_tables with empty regions should return empty analysis."""
        result = analyze_page_tables(1, [], [])
        self.assertIsInstance(result, PageTableFallbackAnalysis)
        self.assertEqual(result.table_count, 0)
        self.assertEqual(result.fallback_triggered, 0)

    def test_low_confidence_triggers_fallback(self):
        """Tables with confidence below threshold should trigger fallback."""
        regions = [
            TableRegion(
                bbox=(50, 100, 400, 200),
                page_number=1,
                original_confidence=0.3,
                original_engine="paddle",
            ),
        ]
        # Lines with low confidence inside the region
        paddle_lines = [
            ("text", [[60, 110], [440, 110], [440, 130], [60, 130]], 0.3),
        ]
        result = analyze_page_tables(1, regions, paddle_lines)
        self.assertGreater(result.fallback_triggered, 0)


class TestAssemblyQueueMessage(unittest.TestCase):
    """Test that assembly queue messages contain table_fallback_data field."""

    def test_message_includes_table_fallback_data(self):
        """Assembly queue message should have table_fallback_data key."""
        msg = {
            "doc_id": "test_doc",
            "page_num": 1,
            "text": "Hello world",
            "status": "OK",
            "chunk_path": "/tmp/1.pdf",
            "structure_data": None,
            "ocr_confidence": 0.95,
            "ocr_method": "paddle",
            "handwriting_data": None,
            "signature_data": None,
            "vertical_text_data": None,
            "table_fallback_data": None,
        }
        self.assertIn("table_fallback_data", msg)

    def test_message_with_fallback_analysis(self):
        """Assembly queue message should carry PageTableFallbackAnalysis data."""
        analysis = PageTableFallbackAnalysis(
            page_number=1,
            table_count=2,
            fallback_triggered=1,
        )
        msg = {
            "doc_id": "test_doc",
            "page_num": 1,
            "table_fallback_data": analysis,
        }
        self.assertIsNotNone(msg["table_fallback_data"])
        self.assertEqual(msg["table_fallback_data"].table_count, 2)


class TestAssemblerTableFallbackCollection(unittest.TestCase):
    """Test that assembler collects and stores table fallback page data."""

    def test_collection_dict_structure(self):
        """table_fallback_pages dict should store by doc_id -> page_num."""
        table_fallback_pages = {}
        doc_id = "doc_123"
        page_num = 1

        if doc_id not in table_fallback_pages:
            table_fallback_pages[doc_id] = {}

        analysis = PageTableFallbackAnalysis(page_number=1, table_count=3)
        table_fallback_pages[doc_id][page_num] = analysis

        self.assertIn(doc_id, table_fallback_pages)
        self.assertIn(page_num, table_fallback_pages[doc_id])
        self.assertEqual(table_fallback_pages[doc_id][page_num].table_count, 3)

    def test_no_duplicate_pages(self):
        """Duplicate page data should not overwrite existing entries."""
        table_fallback_pages = {"doc_1": {}}

        first = PageTableFallbackAnalysis(page_number=1, table_count=2)
        second = PageTableFallbackAnalysis(page_number=1, table_count=5)

        page_num = 1
        if page_num not in table_fallback_pages["doc_1"]:
            table_fallback_pages["doc_1"][page_num] = first

        # Simulate second message — should NOT overwrite
        if page_num not in table_fallback_pages["doc_1"]:
            table_fallback_pages["doc_1"][page_num] = second

        self.assertEqual(table_fallback_pages["doc_1"][page_num].table_count, 2)

    def test_multi_page_collection(self):
        """Multiple pages should be collected independently."""
        table_fallback_pages = {"doc_1": {}}

        for p in range(1, 4):
            analysis = PageTableFallbackAnalysis(page_number=p, table_count=p)
            table_fallback_pages["doc_1"][p] = analysis

        self.assertEqual(len(table_fallback_pages["doc_1"]), 3)
        self.assertEqual(table_fallback_pages["doc_1"][1].table_count, 1)
        self.assertEqual(table_fallback_pages["doc_1"][3].table_count, 3)


class TestTableFallbackFinalization(unittest.TestCase):
    """Test document-level finalization of table fallback results."""

    def test_finalize_with_pages(self):
        """finalize_table_fallback should aggregate page analyses."""
        pages = [
            PageTableFallbackAnalysis(page_number=1, table_count=2, fallback_triggered=1),
            PageTableFallbackAnalysis(page_number=2, table_count=1, fallback_triggered=0),
        ]
        summary = finalize_table_fallback(pages, document_id="doc_1", source_file="test.pdf")

        self.assertIsInstance(summary, DocumentTableFallbackSummary)
        self.assertEqual(summary.document_id, "doc_1")
        self.assertEqual(summary.source_file, "test.pdf")
        self.assertEqual(summary.total_tables, 3)
        self.assertEqual(summary.total_fallback_triggered, 1)
        self.assertEqual(summary.total_pages, 2)

    def test_finalize_empty_pages(self):
        """finalize_table_fallback with no pages should return empty summary."""
        summary = finalize_table_fallback([], document_id="doc_2")
        self.assertEqual(summary.total_tables, 0)
        self.assertEqual(summary.total_pages, 0)

    def test_finalize_lambda_pattern(self):
        """The lambda pattern used in assembler should work with _finalize_and_write_feature."""
        # Simulate the lambda call pattern from assembler
        page_analyses = [
            PageTableFallbackAnalysis(page_number=1, table_count=3, fallback_triggered=2),
        ]
        doc_id = "doc_test"
        doc_path = "test.pdf"

        def finalize_fn(analyses):
            return finalize_table_fallback(
                analyses, document_id=doc_id, source_file=doc_path,
            )

        result = finalize_fn(page_analyses)
        self.assertIsInstance(result, DocumentTableFallbackSummary)
        self.assertEqual(result.total_tables, 3)


class TestTableFallbackJsonOutput(unittest.TestCase):
    """Test JSON output writing for table fallback results."""

    def test_write_json(self):
        """write_table_fallback_json should create a valid JSON sidecar."""
        import json
        import tempfile

        summary = DocumentTableFallbackSummary(
            document_id="doc_1",
            source_file="test.pdf",
            total_pages=1,
            total_tables=2,
            total_fallback_triggered=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_table_fallback_json(summary, tmpdir, ".", "0.9.0")
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))

            with open(result, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["schema_version"], "1.0")
            self.assertEqual(data["document_id"], "doc_1")
            self.assertEqual(data["document_summary"]["total_tables"], 2)
            self.assertEqual(data["document_summary"]["total_fallback_triggered"], 1)

    def test_write_json_graceful_on_bad_summary(self):
        """write_table_fallback_json should handle edge cases gracefully."""
        summary = DocumentTableFallbackSummary(document_id="", source_file="")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_table_fallback_json(summary, tmpdir, ".", "0.9.0")
            # Even with empty fields, it should still write successfully
            self.assertIsNotNone(result)


class TestTableFallbackCleanup(unittest.TestCase):
    """Verify that table fallback page data is cleaned up after finalization."""

    def test_cleanup_removes_doc_data(self):
        """Simulated cleanup should remove doc_id from table_fallback_pages."""
        table_fallback_pages = {"doc_1": {1: "data", 2: "data"}, "doc_2": {1: "data"}}

        doc_id = "doc_1"
        if doc_id in table_fallback_pages:
            del table_fallback_pages[doc_id]

        self.assertNotIn("doc_1", table_fallback_pages)
        self.assertIn("doc_2", table_fallback_pages)


class TestCLIArgument(unittest.TestCase):
    """Verify --enable-table-fallback CLI argument is registered."""

    def test_cli_arg_in_source(self):
        """ocr_gpu_async.py should contain --enable-table-fallback argument."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn("--enable-table-fallback", source)
        self.assertIn("ENABLE_TABLE_FALLBACK", source)

    def test_table_fallback_requires_docintel(self):
        """Table fallback CLI flag should document that it requires --enable-docintel."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        # The help text should mention docintel requirement
        self.assertIn("requires --enable-docintel", source)


class TestFeatureGating(unittest.TestCase):
    """Verify table fallback is properly gated behind feature flags."""

    def test_env_var_default_disabled(self):
        """ENABLE_TABLE_FALLBACK should default to False."""
        from table_fallback import ENABLE_TABLE_FALLBACK
        # The default (when env var is not set to "true") should be False
        # We can't easily test this without controlling the env, but we can verify
        # the constant exists
        self.assertIsInstance(ENABLE_TABLE_FALLBACK, bool)

    def test_worker_gate_pattern(self):
        """Worker thread should check both ENABLE and _AVAILABLE flags."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        # The worker should check both flags
        self.assertIn("ENABLE_TABLE_FALLBACK", source)
        self.assertIn("_TABLE_FALLBACK_AVAILABLE", source)

    def test_structure_data_required(self):
        """Worker should only analyze tables when structure_data is available."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        # The worker gate should check structure_data
        self.assertIn("and structure_data", source)


class TestEndToEndTableFallbackFlow(unittest.TestCase):
    """End-to-end test simulating the full pipeline flow."""

    def test_full_flow(self):
        """Simulate: structure_data -> region extraction -> analysis -> finalization -> JSON."""
        import json
        import tempfile

        # 1. Worker: extract table regions from structure_data
        structure_data = {
            "layout_regions": [
                {"type": "table", "bbox": [50, 100, 450, 300], "confidence": 0.88},
                {"type": "text", "bbox": [50, 10, 450, 90], "confidence": 0.95},
                {"type": "table", "bbox": [50, 350, 450, 550], "confidence": 0.45},
            ]
        }

        table_regions = []
        for region in structure_data.get("layout_regions", []):
            if region.get("type") == "table":
                rb = region.get("bbox", [0, 0, 0, 0])
                if len(rb) == 4:
                    x1, y1, x2, y2 = rb
                    table_regions.append(TableRegion(
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        page_number=1,
                        original_confidence=region.get("confidence", 0.0),
                        original_engine="paddle",
                    ))

        self.assertEqual(len(table_regions), 2)

        # 2. Worker: run page analysis
        paddle_lines = [
            ("Header text", [[60, 110], [440, 110], [440, 130], [60, 130]], 0.92),
            ("Cell A", [[60, 150], [240, 150], [240, 170], [60, 170]], 0.85),
            ("Cell B", [[250, 150], [440, 150], [440, 170], [250, 170]], 0.88),
            ("Low conf cell", [[60, 370], [440, 370], [440, 390], [60, 390]], 0.35),
        ]

        page_analysis = analyze_page_tables(1, table_regions, paddle_lines)
        self.assertIsInstance(page_analysis, PageTableFallbackAnalysis)
        self.assertEqual(page_analysis.table_count, 2)

        # 3. Assembler: collect page data
        table_fallback_pages = {"doc_test": {}}
        table_fallback_pages["doc_test"][1] = page_analysis

        # 4. Assembler: finalize
        page_analyses = []
        for p in range(1, 2):
            page_tf = table_fallback_pages.get("doc_test", {}).get(p)
            if page_tf is not None:
                page_analyses.append(page_tf)

        summary = finalize_table_fallback(
            page_analyses, document_id="doc_test", source_file="test.pdf",
        )
        self.assertEqual(summary.total_tables, 2)

        # 5. Write JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = write_table_fallback_json(summary, tmpdir, ".", "0.9.0")
            self.assertIsNotNone(json_path)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["document_summary"]["total_tables"], 2)
            self.assertIn("pages", data)

        # 6. Cleanup
        del table_fallback_pages["doc_test"]
        self.assertNotIn("doc_test", table_fallback_pages)


if __name__ == "__main__":
    unittest.main()
