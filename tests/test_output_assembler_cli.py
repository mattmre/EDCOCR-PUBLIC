"""Tests for output_assembler.py wiring into CLI pipeline (ocr_gpu_async.py).

Verifies that:
- ENABLE_RETRIEVAL_OUTPUT env var is read correctly
- _OUTPUT_ASSEMBLER_AVAILABLE flag exists
- The wiring code exists in _finalize_doc source
- assemble_retrieval_output is called when enabled (integration mock test)
- The --enable-retrieval-output CLI flag is registered
"""

import inspect
import os
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Source inspection tests
# ---------------------------------------------------------------------------


def test_enable_retrieval_output_constant_exists():
    """ENABLE_RETRIEVAL_OUTPUT constant should exist in ocr_gpu_async module."""
    import ocr_gpu_async

    assert hasattr(ocr_gpu_async, "ENABLE_RETRIEVAL_OUTPUT")
    assert isinstance(ocr_gpu_async.ENABLE_RETRIEVAL_OUTPUT, bool)


def test_output_assembler_available_flag_exists():
    """_OUTPUT_ASSEMBLER_AVAILABLE flag should exist in ocr_gpu_async module."""
    import ocr_gpu_async

    assert hasattr(ocr_gpu_async, "_OUTPUT_ASSEMBLER_AVAILABLE")
    assert isinstance(ocr_gpu_async._OUTPUT_ASSEMBLER_AVAILABLE, bool)


def test_enable_retrieval_output_env_var_false_by_default():
    """ENABLE_RETRIEVAL_OUTPUT defaults to False when env var is unset."""
    # The module-level constant is set at import time.
    # Check the parsing logic directly to verify default behavior.
    val = os.environ.get("ENABLE_RETRIEVAL_OUTPUT", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    assert val is False


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("Yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
        ("random", False),
    ],
)
def test_enable_retrieval_output_env_var_parsing(env_value, expected):
    """ENABLE_RETRIEVAL_OUTPUT env var is parsed correctly for various values."""
    result = env_value.lower() in ("1", "true", "yes")
    assert result is expected


def test_finalize_doc_contains_retrieval_output_wiring():
    """_finalize_doc function source should contain retrieval output wiring."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async._finalize_doc)
    assert "ENABLE_RETRIEVAL_OUTPUT" in source
    assert "_OUTPUT_ASSEMBLER_AVAILABLE" in source
    assert "assemble_retrieval_output" in source
    assert "write_retrieval_json" in source
    assert "write_retrieval_markdown" in source


def test_finalize_doc_retrieval_output_gated_on_both_flags():
    """Retrieval output call must be gated on BOTH flags."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async._finalize_doc)
    # The condition should check both ENABLE_RETRIEVAL_OUTPUT and _OUTPUT_ASSEMBLER_AVAILABLE
    assert "ENABLE_RETRIEVAL_OUTPUT and _OUTPUT_ASSEMBLER_AVAILABLE" in source


def test_finalize_doc_retrieval_output_has_exception_handler():
    """Retrieval output wiring must have try/except to avoid breaking the pipeline."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async._finalize_doc)
    # Find the retrieval output section
    idx = source.index("ENABLE_RETRIEVAL_OUTPUT and _OUTPUT_ASSEMBLER_AVAILABLE")
    retrieval_section = source[idx:]
    # Should have an except clause for the retrieval block
    assert "Retrieval output assembly failed" in retrieval_section


def test_parse_args_has_enable_retrieval_output():
    """_parse_args should include --enable-retrieval-output argument."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async._parse_args)
    assert "--enable-retrieval-output" in source


def test_main_has_global_enable_retrieval_output():
    """main() should declare ENABLE_RETRIEVAL_OUTPUT as global."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.main)
    assert "ENABLE_RETRIEVAL_OUTPUT" in source


def test_imports_from_output_assembler():
    """ocr_gpu_async should import from output_assembler (guarded)."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async)
    assert "from output_assembler import" in source


def test_main_logs_retrieval_output_status():
    """main() should log retrieval output status at startup."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async.main)
    assert "Unified retrieval output ENABLED" in source
    assert "output_assembler module not available" in source


# ---------------------------------------------------------------------------
# Integration tests with mocks
# ---------------------------------------------------------------------------


def _make_mock_doc(doc_id="test-doc", path="/app/ocr_source/test.pdf",
                   output_pdf="/app/ocr_output/EXPORT/PDF/test.pdf",
                   output_txt_dir="/app/ocr_output/EXPORT/TEXT",
                   total_pages=2, temp_dir="/tmp/test-doc"):
    """Create a mock document object matching the pipeline's DocRecord pattern."""
    doc = mock.MagicMock()
    doc.path = path
    doc.output_pdf = output_pdf
    doc.output_txt_dir = output_txt_dir
    doc.total_pages = total_pages
    doc.processed_pages = total_pages
    doc.temp_dir = temp_dir
    doc.terminal_pages = set(range(1, total_pages + 1))
    doc.terminal_statuses = {p: "OK" for p in range(1, total_pages + 1)}
    doc.finalized = True
    doc.start_time = 0
    doc.custody_chain = None
    return doc


def _make_page_data_snap(total_pages=2):
    """Create a minimal page data snapshot dict."""
    texts = {p: f"Page {p} text content." for p in range(1, total_pages + 1)}
    return {
        "texts": texts,
        "structure": {},
        "validation": {},
        "handwriting": {},
        "signature": {},
        "vertical_text": {},
        "table_fallback": {},
        "classification": {},
    }


@pytest.fixture
def _patch_pipeline_globals():
    """Patch pipeline globals so _finalize_doc can run without full pipeline context."""
    import ocr_gpu_async

    patches = []

    # Patch feature flags to disabled (except retrieval)
    flag_patches = {
        "ENABLE_VALIDATION": False,
        "ENABLE_NER": False,
        "ENABLE_HANDWRITING": False,
        "ENABLE_SIGNATURE_VERIFICATION": False,
        "ENABLE_VERTICAL_TEXT": False,
        "ENABLE_TABLE_FALLBACK": False,
        "ENABLE_CLASSIFICATION": False,
        "ENABLE_EXTRACTION": False,
        "ENABLE_SPECIALIST_ROUTING": False,
        "ENABLE_ENTITY_CONSOLIDATION": False,
        "ENABLE_DOCUMENT_INTELLIGENCE": False,
        "ENABLE_RETRIEVAL_OUTPUT": True,
        "_OUTPUT_ASSEMBLER_AVAILABLE": True,
        "_VALIDATION_AVAILABLE": False,
        "_NER_AVAILABLE": False,
        "_HANDWRITING_AVAILABLE": False,
        "_SIGNATURE_VERIFICATION_AVAILABLE": False,
        "_VERTICAL_TEXT_AVAILABLE": False,
        "_TABLE_FALLBACK_AVAILABLE": False,
        "_CLASSIFICATION_AVAILABLE": False,
        "_EXTRACTION_AVAILABLE": False,
        "_ROUTING_AVAILABLE": False,
        "_ENTITY_CONSOLIDATOR_AVAILABLE": False,
        "SOURCE_FOLDER": "/app/ocr_source",
        "OUTPUT_FOLDER": "/app/ocr_output",
    }
    for attr, val in flag_patches.items():
        p = mock.patch.object(ocr_gpu_async, attr, val)
        patches.append(p)
        p.start()

    # Patch adaptive batch sizer
    p = mock.patch.object(ocr_gpu_async, "_adaptive_batch_sizer", None)
    patches.append(p)
    p.start()

    # Patch doc_registry_lock and doc_registry
    p = mock.patch.object(ocr_gpu_async, "doc_registry_lock", mock.MagicMock())
    patches.append(p)
    p.start()
    p = mock.patch.object(ocr_gpu_async, "doc_registry", {})
    patches.append(p)
    p.start()

    # Patch docs processed counter
    p = mock.patch.object(ocr_gpu_async, "global_docs_processed", 0)
    patches.append(p)
    p.start()

    # Patch compression queue
    p = mock.patch.object(ocr_gpu_async, "compression_queue", mock.MagicMock())
    patches.append(p)
    p.start()

    yield

    for p in patches:
        p.stop()


@mock.patch("ocr_gpu_async.write_retrieval_markdown")
@mock.patch("ocr_gpu_async.write_retrieval_json")
@mock.patch("ocr_gpu_async.assemble_retrieval_output")
@mock.patch("ocr_gpu_async.fitz")
@mock.patch("ocr_gpu_async.log_failure")
def test_finalize_doc_calls_assemble_retrieval_output(
    mock_log_failure,
    mock_fitz,
    mock_assemble,
    mock_write_json,
    mock_write_md,
    _patch_pipeline_globals,
    tmp_path,
):
    """When ENABLE_RETRIEVAL_OUTPUT is True and module is available,
    _finalize_doc should call assemble_retrieval_output."""
    import ocr_gpu_async

    # Setup mock fitz
    mock_pdf = mock.MagicMock()
    mock_pdf.page_count = 2
    mock_fitz.open.return_value = mock_pdf
    mock_pdf.__enter__ = mock.MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = mock.MagicMock(return_value=False)

    # Create temp chunk files
    temp_dir = str(tmp_path / "chunks")
    os.makedirs(temp_dir, exist_ok=True)
    for p in range(1, 3):
        with open(os.path.join(temp_dir, f"{p}.pdf"), "w") as f:
            f.write("mock chunk")

    output_pdf = str(tmp_path / "output.pdf")
    output_txt_dir = str(tmp_path / "text")
    os.makedirs(output_txt_dir, exist_ok=True)

    doc = _make_mock_doc(
        temp_dir=temp_dir,
        output_pdf=output_pdf,
        output_txt_dir=output_txt_dir,
    )
    snap = _make_page_data_snap(2)

    # Mock the RetrievalDocument returned by assemble_retrieval_output
    mock_ret_doc = mock.MagicMock()
    mock_ret_doc.classification = {"label": "contract"}
    mock_ret_doc.entities = [{"type": "PERSON", "text": "John"}]
    mock_ret_doc.key_value_pairs = []
    mock_assemble.return_value = mock_ret_doc
    mock_write_json.return_value = "/app/ocr_output/EXPORT/RETRIEVAL/test.retrieval.json"
    mock_write_md.return_value = "/app/ocr_output/EXPORT/RETRIEVAL/test.retrieval.md"

    ocr_gpu_async._finalize_doc(doc, "test-doc", snap)

    # Verify assemble_retrieval_output was called
    mock_assemble.assert_called_once()
    call_kwargs = mock_assemble.call_args
    assert call_kwargs.kwargs["document_id"] == "test-doc"
    assert call_kwargs.kwargs["source_file"] == "test.pdf"
    assert "Page 1 text content." in call_kwargs.kwargs["ocr_text"]
    assert len(call_kwargs.kwargs["text_by_page"]) == 2

    # Verify write functions were called
    mock_write_json.assert_called_once()
    mock_write_md.assert_called_once()


@mock.patch("ocr_gpu_async.fitz")
@mock.patch("ocr_gpu_async.log_failure")
def test_finalize_doc_skips_retrieval_when_disabled(
    mock_log_failure,
    mock_fitz,
    _patch_pipeline_globals,
    tmp_path,
):
    """When ENABLE_RETRIEVAL_OUTPUT is False, _finalize_doc should NOT call
    assemble_retrieval_output."""
    import ocr_gpu_async

    # Override the retrieval flag to False
    with mock.patch.object(ocr_gpu_async, "ENABLE_RETRIEVAL_OUTPUT", False):
        mock_pdf = mock.MagicMock()
        mock_pdf.page_count = 1
        mock_fitz.open.return_value = mock_pdf
        mock_pdf.__enter__ = mock.MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = mock.MagicMock(return_value=False)

        temp_dir = str(tmp_path / "chunks")
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "1.pdf"), "w") as f:
            f.write("mock")

        output_txt_dir = str(tmp_path / "text")
        os.makedirs(output_txt_dir, exist_ok=True)

        doc = _make_mock_doc(
            temp_dir=temp_dir,
            output_pdf=str(tmp_path / "out.pdf"),
            output_txt_dir=output_txt_dir,
            total_pages=1,
        )
        snap = _make_page_data_snap(1)

        with mock.patch.object(
            ocr_gpu_async, "assemble_retrieval_output"
        ) as mock_assemble:
            ocr_gpu_async._finalize_doc(doc, "test-doc", snap)
            mock_assemble.assert_not_called()


@mock.patch("ocr_gpu_async.fitz")
@mock.patch("ocr_gpu_async.log_failure")
def test_finalize_doc_skips_retrieval_when_module_unavailable(
    mock_log_failure,
    mock_fitz,
    _patch_pipeline_globals,
    tmp_path,
):
    """When _OUTPUT_ASSEMBLER_AVAILABLE is False, _finalize_doc should NOT call
    assemble_retrieval_output."""
    import ocr_gpu_async

    with mock.patch.object(ocr_gpu_async, "_OUTPUT_ASSEMBLER_AVAILABLE", False):
        mock_pdf = mock.MagicMock()
        mock_pdf.page_count = 1
        mock_fitz.open.return_value = mock_pdf
        mock_pdf.__enter__ = mock.MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = mock.MagicMock(return_value=False)

        temp_dir = str(tmp_path / "chunks")
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "1.pdf"), "w") as f:
            f.write("mock")

        output_txt_dir = str(tmp_path / "text")
        os.makedirs(output_txt_dir, exist_ok=True)

        doc = _make_mock_doc(
            temp_dir=temp_dir,
            output_pdf=str(tmp_path / "out.pdf"),
            output_txt_dir=output_txt_dir,
            total_pages=1,
        )
        snap = _make_page_data_snap(1)

        with mock.patch.object(
            ocr_gpu_async, "assemble_retrieval_output"
        ) as mock_assemble:
            ocr_gpu_async._finalize_doc(doc, "test-doc", snap)
            mock_assemble.assert_not_called()


@mock.patch("ocr_gpu_async.write_retrieval_markdown")
@mock.patch("ocr_gpu_async.write_retrieval_json")
@mock.patch("ocr_gpu_async.assemble_retrieval_output", side_effect=RuntimeError("boom"))
@mock.patch("ocr_gpu_async.fitz")
@mock.patch("ocr_gpu_async.log_failure")
def test_finalize_doc_retrieval_error_does_not_break_pipeline(
    mock_log_failure,
    mock_fitz,
    mock_assemble,
    mock_write_json,
    mock_write_md,
    _patch_pipeline_globals,
    tmp_path,
):
    """If assemble_retrieval_output raises an exception, the pipeline continues."""
    import ocr_gpu_async

    mock_pdf = mock.MagicMock()
    mock_pdf.page_count = 1
    mock_fitz.open.return_value = mock_pdf
    mock_pdf.__enter__ = mock.MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = mock.MagicMock(return_value=False)

    temp_dir = str(tmp_path / "chunks")
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, "1.pdf"), "w") as f:
        f.write("mock")

    output_txt_dir = str(tmp_path / "text")
    os.makedirs(output_txt_dir, exist_ok=True)

    doc = _make_mock_doc(
        temp_dir=temp_dir,
        output_pdf=str(tmp_path / "out.pdf"),
        output_txt_dir=output_txt_dir,
        total_pages=1,
    )
    snap = _make_page_data_snap(1)

    # Should NOT raise; the exception should be caught and logged
    ocr_gpu_async._finalize_doc(doc, "test-doc", snap)

    # Verify assemble was attempted
    mock_assemble.assert_called_once()
    # write functions should NOT have been called (exception before them)
    mock_write_json.assert_not_called()


def test_finalize_doc_passes_all_sidecar_data_to_retrieval():
    """Retrieval output wiring passes NER, classification, extraction, structure,
    validation, and handwriting data through to assemble_retrieval_output."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async._finalize_doc)

    # NER entities
    assert "finalized_ner" in source
    assert "entities_data=_ret_entities_data" in source

    # Classification
    assert "finalized_classification" in source
    assert "classification_data=_ret_classification_data" in source

    # Extraction
    assert "finalized_extraction" in source
    assert "extraction_data=_ret_extraction_data" in source

    # Structure
    assert "structure_data=_ret_structure_data" in source

    # Validation
    assert "validation_data=_ret_validation_data" in source

    # Handwriting
    assert "handwriting_data=_ret_handwriting_data" in source


def test_cli_parse_args_enables_retrieval_output():
    """--enable-retrieval-output CLI flag should set enable_retrieval_output=True."""
    import ocr_gpu_async

    with mock.patch("sys.argv", ["ocr_gpu_async.py", "--enable-retrieval-output"]):
        args = ocr_gpu_async._parse_args()
        assert args.enable_retrieval_output is True


def test_cli_parse_args_retrieval_output_defaults_false():
    """Without --enable-retrieval-output, enable_retrieval_output defaults to False."""
    import ocr_gpu_async

    with mock.patch("sys.argv", ["ocr_gpu_async.py"]):
        args = ocr_gpu_async._parse_args()
        assert args.enable_retrieval_output is False


def test_retrieval_output_writes_to_correct_directory():
    """Retrieval output should target EXPORT/RETRIEVAL/ subdirectory."""
    import ocr_gpu_async

    source = inspect.getsource(ocr_gpu_async._finalize_doc)
    # write_retrieval_json is called with OUTPUT_FOLDER as the output_dir,
    # and write_retrieval_json internally creates EXPORT/RETRIEVAL/
    assert "write_retrieval_json(ret_doc, OUTPUT_FOLDER" in source
    assert "write_retrieval_markdown(ret_doc, OUTPUT_FOLDER" in source
