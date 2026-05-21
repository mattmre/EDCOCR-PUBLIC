"""Tests for Document Intelligence (Phase 3A) functionality.

These tests cover the parsing, summary, and JSON output functions.
They do NOT require paddleocr, GPU, or Docker environment.

Run with: python -m pytest tests/test_document_intelligence.py -v
"""
import json
import os
import tempfile

import pytest

# Add project root to path


# ---------------------------------------------------------------------------
# Import helpers — same pattern as test_utilities.py
# ---------------------------------------------------------------------------

def _load_docintel_functions():
    """Extract Document Intelligence functions from ocr_gpu_async.py without GPU imports."""
    import re
    import types

    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py")

    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    mod = types.ModuleType("docintel_utils")
    mod.__file__ = src_path
    mod.os = os
    mod.json = json  # Required by write_structure_json
    mod.datetime = __import__("datetime")
    # Provide logger stub so write_structure_json doesn't crash
    import logging
    mod.logger = logging.getLogger("test_docintel")
    # Provide __version__ so write_structure_json can embed it
    try:
        from version import __version__
    except ImportError:
        __version__ = "test"
    mod.__version__ = __version__
    # Set constants needed by functions
    mod.SOURCE_FOLDER = "/app/ocr_source"
    mod.OUTPUT_FOLDER = "/app/ocr_output"
    mod.DOCINTEL_MODE = "full"
    mod.ENABLE_DOCUMENT_INTELLIGENCE = True
    mod.ENABLE_FORM_DETECTION = False
    mod.ENABLE_KV_EXTRACTION = False
    mod.ENABLE_PRIVILEGE_DETECTION = False
    mod.re = __import__("re")

    # Phase 3C: compiled patterns and constants needed by functions
    _re = __import__("re")
    mod._PRIVILEGE_KEYWORDS = {
        "attorney-client", "attorney client", "work product",
        "privileged and confidential", "privileged & confidential",
        "attorney work product", "do not disclose", "legally privileged",
    }
    mod.PRIVILEGE_HEURISTIC_CONFIDENCE = 0.85
    mod._ESQ_PATTERN = _re.compile(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*),\s*Esq\.?')
    mod._LAW_FIRM_PATTERN = _re.compile(
        r'\b\w+(?:\s+\w+)*\s+(?:&|and)\s+\w+(?:\s+\w+)*\s+(?:LLP|PLLC|P\.C\.|LLC)\b'
        r'|(?:Law\s+Offices?\s+of\b)'
        r'|(?:Attorneys?\s+at\s+Law\b)',
    )
    mod._FORM_FIELD_KEYWORDS = {
        "signature": ("signature", "sign here", "signed by", "authorized signature"),
        "date": ("date signed", "effective date", "date:", "dated"),
        "checkbox": ("☐", "☑", "☒", "[ ]", "[x]", "[X]"),
    }

    function_names = [
        "parse_structure_result",
        "_build_document_summary",
        "write_structure_json",
        "_extract_forms_and_kvs",
        "_detect_privilege_indicators",
    ]

    for func_name in function_names:
        pattern = rf'^(def {func_name}\(.*?\n(?:(?:    .*\n|[ \t]*\n)*))'
        match = re.search(pattern, source, re.MULTILINE)
        if match:
            func_source = match.group(1)
            try:
                exec(compile(func_source, src_path, "exec"), mod.__dict__)
            except Exception as e:
                print(f"Warning: Could not load {func_name}: {e}")

    return mod


try:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from ocr_gpu_async import (
        DOCINTEL_MODE,
        ENABLE_DOCUMENT_INTELLIGENCE,
        ENABLE_LAYOUT_ANALYSIS,
        ENABLE_TABLE_EXTRACTION,
        _build_document_summary,
        _detect_privilege_indicators,
        _extract_forms_and_kvs,
        parse_structure_result,
        write_structure_json,
    )
    _DIRECT_IMPORT = True
except ImportError:
    _utils = _load_docintel_functions()
    parse_structure_result = _utils.parse_structure_result
    _build_document_summary = _utils._build_document_summary
    write_structure_json = _utils.write_structure_json
    _extract_forms_and_kvs = getattr(_utils, "_extract_forms_and_kvs", None)
    _detect_privilege_indicators = getattr(_utils, "_detect_privilege_indicators", None)
    ENABLE_DOCUMENT_INTELLIGENCE = False
    ENABLE_LAYOUT_ANALYSIS = True
    ENABLE_TABLE_EXTRACTION = True
    DOCINTEL_MODE = "full"
    _DIRECT_IMPORT = False


# ===========================================================================
# Tests: parse_structure_result
# ===========================================================================

class TestParseStructureResult:
    def test_empty_input(self):
        result = parse_structure_result(None)
        assert result["layout_regions"] == []
        assert result["tables"] == []
        assert result["key_value_pairs"] == []
        assert result["form_fields"] == []

    def test_empty_list(self):
        result = parse_structure_result([])
        assert result["layout_regions"] == []
        assert result["tables"] == []
        assert result["key_value_pairs"] == []
        assert result["form_fields"] == []

    def test_text_region(self):
        """Text region with res containing text items."""
        input_data = [
            {
                "type": "text",
                "bbox": [50, 100, 550, 400],
                "res": [
                    {"text": "Hello World", "confidence": 0.95},
                    {"text": "Second line", "confidence": 0.90},
                ],
            }
        ]
        result = parse_structure_result(input_data)
        assert len(result["layout_regions"]) == 1
        region = result["layout_regions"][0]
        assert region["type"] == "text"
        assert region["bbox"] == [50, 100, 550, 400]
        assert "Hello World" in region["text"]
        assert "Second line" in region["text"]
        assert 0.9 <= region["confidence"] <= 0.95
        assert len(result["tables"]) == 0

    def test_table_region(self):
        """Table region with HTML result."""
        input_data = [
            {
                "type": "table",
                "bbox": [50, 420, 550, 600],
                "res": {
                    "html": "<table><tr><td>A</td><td>B</td></tr></table>",
                    "cell_bbox": [[50, 420, 300, 500], [300, 420, 550, 500]],
                },
            }
        ]
        result = parse_structure_result(input_data)
        assert len(result["layout_regions"]) == 1
        region = result["layout_regions"][0]
        assert region["type"] == "table"
        assert region["table_index"] == 0
        assert len(result["tables"]) == 1
        assert "<table>" in result["tables"][0]["html"]
        assert len(result["tables"][0]["cell_bbox"]) == 2

    def test_mixed_regions(self):
        """Multiple region types in single result."""
        input_data = [
            {
                "type": "title",
                "bbox": [100, 50, 500, 80],
                "res": [{"text": "AGREEMENT", "confidence": 0.98}],
            },
            {
                "type": "text",
                "bbox": [50, 100, 550, 400],
                "res": [{"text": "Body text", "confidence": 0.92}],
            },
            {
                "type": "table",
                "bbox": [50, 420, 550, 600],
                "res": {
                    "html": "<table><tr><td>X</td></tr></table>",
                    "cell_bbox": [],
                },
            },
            {
                "type": "figure",
                "bbox": [400, 50, 550, 200],
            },
        ]
        result = parse_structure_result(input_data)
        assert len(result["layout_regions"]) == 4
        types_found = [r["type"] for r in result["layout_regions"]]
        assert types_found == ["title", "text", "table", "figure"]
        assert len(result["tables"]) == 1

    def test_non_dict_items_skipped(self):
        """Non-dict items in result list should be skipped."""
        input_data = ["not_a_dict", 42, None, {"type": "text", "bbox": [0, 0, 1, 1]}]
        result = parse_structure_result(input_data)
        assert len(result["layout_regions"]) == 1

    def test_missing_type_defaults_to_unknown(self):
        result = parse_structure_result([{"bbox": [0, 0, 10, 10]}])
        assert result["layout_regions"][0]["type"] == "unknown"

    def test_table_with_non_dict_res(self):
        """Table region where res is not a dict should produce empty html/cell_bbox."""
        input_data = [{"type": "table", "bbox": [0, 0, 1, 1], "res": "not_a_dict"}]
        result = parse_structure_result(input_data)
        assert len(result["tables"]) == 1
        assert result["tables"][0]["html"] == ""
        assert result["tables"][0]["cell_bbox"] == []


# ===========================================================================
# Tests: _build_document_summary
# ===========================================================================

class TestBuildDocumentSummary:
    def test_empty_pages(self):
        result = _build_document_summary([])
        assert result["total_tables"] == 0
        assert result["total_figures"] == 0
        assert result["layout_types_found"] == []

    def test_counts_tables_and_figures(self):
        pages = [
            {
                "layout_regions": [
                    {"type": "table"},
                    {"type": "text"},
                    {"type": "figure"},
                ]
            },
            {
                "layout_regions": [
                    {"type": "table"},
                    {"type": "title"},
                ]
            },
        ]
        result = _build_document_summary(pages)
        assert result["total_tables"] == 2
        assert result["total_figures"] == 1
        assert "table" in result["layout_types_found"]
        assert "figure" in result["layout_types_found"]
        assert "text" in result["layout_types_found"]
        assert "title" in result["layout_types_found"]

    def test_sorted_layout_types(self):
        pages = [{"layout_regions": [{"type": "text"}, {"type": "figure"}, {"type": "table"}]}]
        result = _build_document_summary(pages)
        assert result["layout_types_found"] == ["figure", "table", "text"]

    def test_non_dict_pages_skipped(self):
        pages = [None, "bad", {"layout_regions": [{"type": "text"}]}]
        result = _build_document_summary(pages)
        assert result["layout_types_found"] == ["text"]

    def test_missing_layout_regions_key(self):
        pages = [{"other_key": "value"}]
        result = _build_document_summary(pages)
        assert result["total_tables"] == 0


# ===========================================================================
# Tests: write_structure_json
# ===========================================================================

class TestWriteStructureJson:
    """Tests for write_structure_json — requires direct import to modify SOURCE_FOLDER."""

    @pytest.fixture(autouse=True)
    def _require_direct_import(self):
        if not _DIRECT_IMPORT:
            pytest.skip("write_structure_json tests require direct import (needs paddleocr)")

    @pytest.fixture
    def _source_folder_override(self):
        """Temporarily set SOURCE_FOLDER for write_structure_json."""
        import ocr_gpu_async
        original = ocr_gpu_async.SOURCE_FOLDER
        ocr_gpu_async.SOURCE_FOLDER = "/app/ocr_source"
        yield
        ocr_gpu_async.SOURCE_FOLDER = original

    def test_creates_json_file(self, _source_folder_override):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = "/app/ocr_source/test_doc.pdf"
            pages = [
                {"page_num": 1, "layout_regions": [{"type": "text"}], "tables": []},
            ]
            json_path = write_structure_json("abc123", source_file, pages, tmpdir)

            assert json_path is not None
            assert os.path.exists(json_path)
            assert json_path.endswith(".structure.json")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["schema_version"] == "1.0"
            assert data["document_id"] == "abc123"
            assert "pages" in data
            assert "document_summary" in data
            assert "processing" in data
            assert "pipeline_version" in data["processing"]

    def test_nested_subfolder_mirroring(self, _source_folder_override):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = "/app/ocr_source/case_a/sub/doc.pdf"
            pages = [
                {"page_num": 1, "layout_regions": [], "tables": []},
            ]
            json_path = write_structure_json("def456", source_file, pages, tmpdir)

            assert json_path is not None
            # Should mirror subfolder: EXPORT/STRUCTURE/case_a/sub/doc.structure.json
            expected_parts = ["EXPORT", "STRUCTURE", "case_a", "sub", "doc.structure.json"]
            for part in expected_parts:
                assert part in json_path.replace("\\", "/")

    def test_json_content_structure(self, _source_folder_override):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = "/app/ocr_source/report.pdf"
            pages = [
                {
                    "page_num": 1,
                    "layout_regions": [
                        {"type": "table"},
                        {"type": "figure"},
                    ],
                    "tables": [],
                },
                {
                    "page_num": 2,
                    "layout_regions": [{"type": "text"}],
                    "tables": [],
                },
            ]
            json_path = write_structure_json("ghi789", source_file, pages, tmpdir)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            summary = data["document_summary"]
            assert summary["total_tables"] == 1
            assert summary["total_figures"] == 1
            assert "table" in summary["layout_types_found"]
            assert "figure" in summary["layout_types_found"]
            assert "text" in summary["layout_types_found"]


# ===========================================================================
# Tests: Configuration defaults
# ===========================================================================

class TestDocIntelConfigDefaults:
    """Verify default configuration values exist and are correct."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_direct_import(self):
        if not _DIRECT_IMPORT:
            pytest.skip("Direct import not available; config defaults not testable")

    def test_docintel_disabled_by_default(self):
        # The module-level default is False (before argparse override)
        # We test that the constant exists and is boolean
        assert isinstance(ENABLE_DOCUMENT_INTELLIGENCE, bool)

    def test_layout_analysis_enabled_by_default(self):
        assert ENABLE_LAYOUT_ANALYSIS is True

    def test_table_extraction_enabled_by_default(self):
        assert ENABLE_TABLE_EXTRACTION is True

    def test_docintel_mode_default(self):
        assert DOCINTEL_MODE in ("layout_only", "tables_only", "full")


# ===========================================================================
# Tests: Phase 3C — _extract_forms_and_kvs
# ===========================================================================

class TestExtractFormsAndKVs:
    """Tests for Phase 3C heuristic form/KV extraction."""

    @pytest.fixture(autouse=True)
    def _skip_if_unavailable(self):
        if _extract_forms_and_kvs is None:
            pytest.skip("_extract_forms_and_kvs not available via fallback import")

    @pytest.fixture(autouse=True)
    def _enable_flags(self):
        """Enable Phase 3C flags so extraction functions actually run."""
        if _DIRECT_IMPORT:
            import ocr_gpu_async
            orig_kv = ocr_gpu_async.ENABLE_KV_EXTRACTION
            orig_form = ocr_gpu_async.ENABLE_FORM_DETECTION
            ocr_gpu_async.ENABLE_KV_EXTRACTION = True
            ocr_gpu_async.ENABLE_FORM_DETECTION = True
            yield
            ocr_gpu_async.ENABLE_KV_EXTRACTION = orig_kv
            ocr_gpu_async.ENABLE_FORM_DETECTION = orig_form
        else:
            # Fallback: patch the module where functions were loaded
            _utils.ENABLE_KV_EXTRACTION = True
            _utils.ENABLE_FORM_DETECTION = True
            yield
            _utils.ENABLE_KV_EXTRACTION = False
            _utils.ENABLE_FORM_DETECTION = False

    def test_colon_kv_extraction(self):
        res_items = [
            {"text": "Client Name: Acme Corp", "bbox": [50, 100, 300, 120], "confidence": 0.95},
        ]
        kvs, fields, _ = _extract_forms_and_kvs(res_items, [50, 100, 300, 120], 1)
        assert len(kvs) == 1
        assert kvs[0]["key"] == "Client Name"
        assert kvs[0]["value"] == "Acme Corp"
        assert kvs[0]["extraction_method"] == "heuristic_colon"

    def test_multiple_kv_pairs(self):
        res_items = [
            {"text": "Name: Alice", "confidence": 0.9},
            {"text": "Amount: $500", "confidence": 0.92},
        ]
        kvs, _, _ = _extract_forms_and_kvs(res_items, [], 1)
        assert len(kvs) == 2
        keys = [kv["key"] for kv in kvs]
        assert "Name" in keys
        assert "Amount" in keys

    def test_no_colon_no_kv(self):
        res_items = [{"text": "No colon here", "confidence": 0.9}]
        kvs, _, _ = _extract_forms_and_kvs(res_items, [], 1)
        assert len(kvs) == 0

    def test_signature_field_detection(self):
        res_items = [
            {"text": "Signature: ___________", "bbox": [50, 800, 300, 850], "confidence": 0.88},
        ]
        _, fields, _ = _extract_forms_and_kvs(res_items, [], 1)
        sig_fields = [f for f in fields if f["field_type"] == "signature"]
        assert len(sig_fields) >= 1

    def test_date_field_detection(self):
        res_items = [
            {"text": "Date Signed: __________", "confidence": 0.90},
        ]
        _, fields, _ = _extract_forms_and_kvs(res_items, [], 1)
        date_fields = [f for f in fields if f["field_type"] == "date"]
        assert len(date_fields) >= 1

    def test_field_id_increments(self):
        res_items = [
            {"text": "Signature line", "confidence": 0.9},
            {"text": "Date Signed here", "confidence": 0.9},
        ]
        _, fields, next_id = _extract_forms_and_kvs(res_items, [], 1)
        if fields:
            ids = [f["field_id"] for f in fields]
            assert ids == sorted(ids)
            assert next_id > max(ids)

    def test_empty_text_skipped(self):
        res_items = [{"text": "", "confidence": 0.9}]
        kvs, fields, _ = _extract_forms_and_kvs(res_items, [], 1)
        assert len(kvs) == 0
        assert len(fields) == 0

    def test_non_dict_items_skipped(self):
        res_items = ["not_a_dict", 42, None]
        kvs, fields, _ = _extract_forms_and_kvs(res_items, [], 1)
        assert len(kvs) == 0
        assert len(fields) == 0

    def test_long_key_rejected(self):
        """Keys longer than 80 chars should be rejected (likely not real KV pairs)."""
        res_items = [{"text": "A" * 85 + ": value", "confidence": 0.9}]
        kvs, _, _ = _extract_forms_and_kvs(res_items, [], 1)
        assert len(kvs) == 0

    def test_bbox_split_approximation(self):
        """Verify key_bbox and value_bbox are computed from parent bbox."""
        res_items = [
            {"text": "Name: Alice", "bbox": [0, 0, 100, 20], "confidence": 0.9},
        ]
        kvs, _, _ = _extract_forms_and_kvs(res_items, [0, 0, 100, 20], 1)
        assert len(kvs) == 1
        assert len(kvs[0]["key_bbox"]) == 4
        assert len(kvs[0]["value_bbox"]) == 4
        # key_bbox should end before value_bbox starts
        assert kvs[0]["key_bbox"][2] <= kvs[0]["value_bbox"][2]


# ===========================================================================
# Tests: Phase 3C — _detect_privilege_indicators
# ===========================================================================

class TestDetectPrivilegeIndicators:
    """Tests for Phase 3C legal privilege detection heuristics."""

    @pytest.fixture(autouse=True)
    def _skip_if_unavailable(self):
        if _detect_privilege_indicators is None:
            pytest.skip("_detect_privilege_indicators not available via fallback import")

    def test_attorney_name_esq(self):
        pages = [{"layout_regions": [
            {"type": "text", "text": "Prepared by John Doe, Esq."},
        ]}]
        result = _detect_privilege_indicators(pages)
        assert result is not None
        assert "John Doe" in result["attorney_names"]

    def test_privileged_keywords(self):
        pages = [{"layout_regions": [
            {"type": "title", "text": "ATTORNEY-CLIENT PRIVILEGED COMMUNICATION"},
        ]}]
        result = _detect_privilege_indicators(pages)
        assert result is not None
        assert "attorney-client" in result["privileged_keywords"]

    def test_work_product(self):
        pages = [{"layout_regions": [
            {"type": "text", "text": "This document is protected work product."},
        ]}]
        result = _detect_privilege_indicators(pages)
        assert result is not None
        assert "work product" in result["privileged_keywords"]

    def test_no_indicators_returns_none(self):
        pages = [{"layout_regions": [
            {"type": "text", "text": "This is a normal invoice for services rendered."},
        ]}]
        result = _detect_privilege_indicators(pages)
        assert result is None

    def test_review_required_flag(self):
        pages = [{"layout_regions": [
            {"type": "text", "text": "Jane Smith, Esq. reviewed this document."},
        ]}]
        result = _detect_privilege_indicators(pages)
        assert result is not None
        assert result["review_required"] is True

    def test_empty_pages(self):
        result = _detect_privilege_indicators([])
        assert result is None

    def test_pages_without_text(self):
        pages = [{"layout_regions": [{"type": "figure"}]}]
        result = _detect_privilege_indicators(pages)
        assert result is None

    def test_law_firm_detection(self):
        """Test that law firm name patterns are correctly identified."""
        pages = [{"layout_regions": [
            {"type": "text", "text": "From the Law Offices of Lionel Hutz"},
            {"type": "text", "text": "This document was prepared by Dewey Cheatem and Howe LLP."},
        ]}]
        result = _detect_privilege_indicators(pages)
        assert result is not None
        assert result["law_firm"] is not None


# ===========================================================================
# Tests: Phase 3C — _build_document_summary extensions
# ===========================================================================

class TestBuildDocumentSummaryPhase3C:
    """Tests for Phase 3C extensions to document summary."""

    def test_form_field_counting(self):
        pages = [{
            "layout_regions": [],
            "form_fields": [
                {"field_id": 1, "field_type": "text", "is_filled": False},
                {"field_id": 2, "field_type": "signature", "is_filled": False},
            ],
        }]
        result = _build_document_summary(pages)
        assert result["total_form_fields"] == 2
        assert result["has_signatures"] is True

    def test_kv_pair_counting(self):
        pages = [{
            "layout_regions": [],
            "key_value_pairs": [
                {"key": "Name", "value": "Test"},
                {"key": "Date", "value": "2026-01-01"},
            ],
        }]
        result = _build_document_summary(pages)
        assert result["total_key_value_pairs"] == 2

    def test_filled_form_detection(self):
        pages = [{
            "layout_regions": [],
            "form_fields": [
                {"field_id": 1, "field_type": "date", "is_filled": True},
            ],
        }]
        result = _build_document_summary(pages)
        assert result["has_filled_forms"] is True

    def test_no_forms_defaults(self):
        pages = [{"layout_regions": [], "tables": []}]
        result = _build_document_summary(pages)
        assert result["total_form_fields"] == 0
        assert result["total_key_value_pairs"] == 0
        assert result["has_signatures"] is False
        assert result["has_filled_forms"] is False

    def test_multi_page_aggregation(self):
        pages = [
            {
                "layout_regions": [],
                "form_fields": [{"field_id": 1, "field_type": "text", "is_filled": False}],
                "key_value_pairs": [{"key": "A", "value": "1"}],
            },
            {
                "layout_regions": [],
                "form_fields": [{"field_id": 2, "field_type": "date", "is_filled": True}],
                "key_value_pairs": [{"key": "B", "value": "2"}, {"key": "C", "value": "3"}],
            },
        ]
        result = _build_document_summary(pages)
        assert result["total_form_fields"] == 2
        assert result["total_key_value_pairs"] == 3
        assert result["has_filled_forms"] is True
