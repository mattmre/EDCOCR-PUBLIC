"""Tests for layoutlm_data — Training data pipeline for LayoutLMv3.

Covers AnnotatedWord, AnnotatedPage, JSONL loading, label validation,
and HuggingFace dataset creation (mocked). All tests run WITHOUT
torch/transformers/datasets installed.

Run with: python -m pytest tests/test_layoutlm_data.py -v
"""

import json
import warnings
from unittest.mock import MagicMock, patch

import pytest

from layoutlm_data import AnnotatedPage, AnnotatedWord, load_custom_jsonl
from layoutlm_labels import build_label_set

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_label_set():
    """Default label set with 9 entity types."""
    return build_label_set("default", [
        "INVOICE_NUMBER", "DATE", "AMOUNT", "PERSON_NAME",
        "ORGANIZATION", "ADDRESS", "REFERENCE_NUMBER",
        "PHONE_NUMBER", "EMAIL",
    ])


@pytest.fixture
def small_label_set():
    """Minimal label set for focused tests."""
    return build_label_set("small", ["DATE", "AMOUNT"])


@pytest.fixture
def sample_jsonl(tmp_path, default_label_set):
    """Create a sample JSONL file with two annotated pages."""
    pages = [
        {
            "doc_id": "doc_001",
            "page_num": 1,
            "words": [
                {"text": "Invoice", "bbox": [10, 20, 100, 40], "label": "O"},
                {"text": "#12345", "bbox": [105, 20, 200, 40], "label": "B-INVOICE_NUMBER"},
                {"text": "Date:", "bbox": [10, 50, 60, 70], "label": "O"},
                {"text": "2024-01-15", "bbox": [65, 50, 180, 70], "label": "B-DATE"},
            ],
            "image_path": "images/doc_001_p1.png",
        },
        {
            "doc_id": "doc_001",
            "page_num": 2,
            "words": [
                {"text": "Total:", "bbox": [10, 10, 80, 30], "label": "O"},
                {"text": "$5,000", "bbox": [85, 10, 180, 30], "label": "B-AMOUNT"},
            ],
        },
    ]
    filepath = tmp_path / "train.jsonl"
    with open(filepath, "w", encoding="utf-8") as fh:
        for page in pages:
            fh.write(json.dumps(page) + "\n")
    return filepath


# ---------------------------------------------------------------------------
# AnnotatedWord tests
# ---------------------------------------------------------------------------


class TestAnnotatedWord:
    """Tests for the AnnotatedWord dataclass."""

    def test_default_values(self):
        """Default bbox is empty and label is 'O'."""
        w = AnnotatedWord(text="hello")
        assert w.text == "hello"
        assert w.bbox == []
        assert w.label == "O"

    def test_custom_values(self):
        """All fields are set correctly."""
        w = AnnotatedWord(text="Invoice", bbox=[10, 20, 100, 40], label="B-DATE")
        assert w.text == "Invoice"
        assert w.bbox == [10, 20, 100, 40]
        assert w.label == "B-DATE"


# ---------------------------------------------------------------------------
# AnnotatedPage tests
# ---------------------------------------------------------------------------


class TestAnnotatedPage:
    """Tests for the AnnotatedPage dataclass."""

    def test_default_values(self):
        """Default page has empty words, no image, empty doc_id."""
        p = AnnotatedPage()
        assert p.words == []
        assert p.image_path is None
        assert p.doc_id == ""
        assert p.page_num == 0

    def test_with_words(self):
        """Page with words stores them correctly."""
        words = [
            AnnotatedWord(text="hello", bbox=[0, 0, 10, 10], label="O"),
            AnnotatedWord(text="world", bbox=[15, 0, 30, 10], label="B-DATE"),
        ]
        p = AnnotatedPage(words=words, doc_id="d1", page_num=1)
        assert len(p.words) == 2
        assert p.words[0].text == "hello"
        assert p.doc_id == "d1"


# ---------------------------------------------------------------------------
# load_custom_jsonl tests
# ---------------------------------------------------------------------------


class TestLoadCustomJsonl:
    """Tests for the JSONL loader."""

    def test_load_basic(self, sample_jsonl, default_label_set):
        """Loading a valid JSONL produces the correct number of pages."""
        pages = load_custom_jsonl(str(sample_jsonl), default_label_set)
        assert len(pages) == 2

    def test_page_content(self, sample_jsonl, default_label_set):
        """Page content is correctly parsed."""
        pages = load_custom_jsonl(str(sample_jsonl), default_label_set)
        assert pages[0].doc_id == "doc_001"
        assert pages[0].page_num == 1
        assert pages[0].image_path == "images/doc_001_p1.png"
        assert len(pages[0].words) == 4
        assert pages[0].words[1].label == "B-INVOICE_NUMBER"

    def test_page_without_image(self, sample_jsonl, default_label_set):
        """Pages without image_path have None."""
        pages = load_custom_jsonl(str(sample_jsonl), default_label_set)
        assert pages[1].image_path is None

    def test_file_not_found(self, default_label_set):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_custom_jsonl("/nonexistent/path.jsonl", default_label_set)

    def test_malformed_json(self, tmp_path, default_label_set):
        """Malformed JSON line raises ValueError."""
        filepath = tmp_path / "bad.jsonl"
        filepath.write_text("not valid json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_custom_jsonl(str(filepath), default_label_set)

    def test_unknown_labels_warn(self, tmp_path, small_label_set):
        """Unknown labels trigger a warning and are replaced with 'O'."""
        page = {
            "doc_id": "d1",
            "page_num": 1,
            "words": [
                {"text": "x", "bbox": [0, 0, 10, 10], "label": "B-UNKNOWN_ENTITY"},
            ],
        }
        filepath = tmp_path / "warn.jsonl"
        filepath.write_text(json.dumps(page) + "\n", encoding="utf-8")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pages = load_custom_jsonl(str(filepath), small_label_set)

        # Label should have been replaced
        assert pages[0].words[0].label == "O"
        # Warning should have been emitted
        assert any("Unknown label" in str(w.message) for w in caught)

    def test_empty_file(self, tmp_path, default_label_set):
        """An empty JSONL file returns an empty list."""
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("", encoding="utf-8")
        pages = load_custom_jsonl(str(filepath), default_label_set)
        assert pages == []

    def test_blank_lines_skipped(self, tmp_path, default_label_set):
        """Blank lines in the JSONL file are silently skipped."""
        page = {
            "doc_id": "d1",
            "page_num": 1,
            "words": [{"text": "hi", "bbox": [0, 0, 10, 10], "label": "O"}],
        }
        filepath = tmp_path / "blanks.jsonl"
        content = "\n" + json.dumps(page) + "\n\n"
        filepath.write_text(content, encoding="utf-8")
        pages = load_custom_jsonl(str(filepath), default_label_set)
        assert len(pages) == 1


# ---------------------------------------------------------------------------
# create_hf_dataset tests (mocked)
# ---------------------------------------------------------------------------


class TestCreateHfDataset:
    """Tests for create_hf_dataset with mocked transformers/datasets."""

    def test_import_error_without_transformers(self, default_label_set):
        """ImportError is raised when transformers is not installed."""
        pages = [AnnotatedPage(words=[], doc_id="d1", page_num=1)]
        with patch.dict("sys.modules", {"transformers": None}):
            from layoutlm_data import create_hf_dataset
            with pytest.raises(ImportError, match="transformers"):
                create_hf_dataset(pages, default_label_set)

    def test_import_error_without_datasets(self, default_label_set):
        """ImportError is raised when datasets library is not installed."""
        # Mock transformers but not datasets
        mock_transformers = MagicMock()
        pages = [AnnotatedPage(words=[], doc_id="d1", page_num=1)]
        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "datasets": None,
        }):
            from layoutlm_data import create_hf_dataset
            with pytest.raises(ImportError, match="datasets"):
                create_hf_dataset(pages, default_label_set)
