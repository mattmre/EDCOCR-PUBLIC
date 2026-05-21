"""Tests for ocr_local.translation.segmentation."""

from __future__ import annotations

import sys
from unittest import mock

from ocr_local.translation.segmentation import (
    _pysbd_lang,
    _regex_segment,
    segment_to_sentences,
)


def test_segment_english_basic():
    result = segment_to_sentences("Hello world. How are you?", "en")
    assert len(result) == 2
    assert result[0]["text"] == "Hello world."
    assert result[1]["text"] == "How are you?"


def test_segment_returns_list_of_dicts():
    result = segment_to_sentences("First. Second.", "en")
    for item in result:
        assert isinstance(item, dict)
        assert "text" in item
        assert "bbox" in item
        assert "span_id" in item


def test_segment_span_id_format():
    result = segment_to_sentences("First. Second. Third.", "en")
    for item in result:
        assert item["span_id"].startswith("seg_")


def test_segment_bbox_is_list():
    result = segment_to_sentences("Hello.", "en")
    assert len(result) == 1
    bbox = result[0]["bbox"]
    assert isinstance(bbox, list)
    assert len(bbox) == 4
    for v in bbox:
        assert isinstance(v, float)


def test_segment_empty_string():
    assert segment_to_sentences("", "en") == []


def test_segment_single_sentence():
    result = segment_to_sentences("This has no punctuation", "en")
    assert len(result) == 1
    assert result[0]["text"] == "This has no punctuation"


def test_regex_fallback_used_when_pysbd_missing():
    # Force ImportError by setting pysbd to None in sys.modules.
    with mock.patch.dict(sys.modules, {"pysbd": None}):
        result = segment_to_sentences("First. Second. Third.", "en")
    assert len(result) == 3
    assert result[0]["text"] == "First."
    assert result[1]["text"] == "Second."
    assert result[2]["text"] == "Third."


def test_pysbd_lang_mapping_english():
    assert _pysbd_lang("en") == "en"
    assert _pysbd_lang("en-US") == "en"
    assert _pysbd_lang("EN") == "en"


def test_pysbd_lang_mapping_unknown():
    assert _pysbd_lang("xx-unknown") == "en"
    assert _pysbd_lang("kli") == "en"


def test_segment_filters_empty_strings():
    # Pure whitespace between separators should be filtered out.
    result = segment_to_sentences("First.    Second.   ", "en")
    assert len(result) == 2
    for item in result:
        assert item["text"].strip() != ""


def test_regex_segment_basic():
    """_regex_segment exposed for fallback testing."""
    parts = _regex_segment("One. Two! Three?")
    assert parts == ["One.", "Two!", "Three?"]
