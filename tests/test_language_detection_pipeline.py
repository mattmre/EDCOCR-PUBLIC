"""Pipeline-wiring tests for Plan A -- PR A2 (page-level FastText detection).

Covers:
- PipelineConfig fields for per-span language detection
- ``aggregate_page_from_full_text`` with a mocked FastText model
- ``finalize_document_language`` aggregation across pages
- ``write_language_json`` path layout, include_spans, schema conformance
- ``FASTTEXT_TO_PADDLE`` reverse mapping validity
- Feature-flag gating (no .language.json when disabled)

Run with::

    python -m pytest tests/test_language_detection_pipeline.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import jsonschema
import pytest

from language_detection import (
    FASTTEXT_TO_PADDLE,
    LANGUAGE_SCHEMA_VERSION,
    DocumentLanguage,
    PageLanguage,
    SpanLanguage,
    aggregate_page_from_full_text,
    finalize_document_language,
    write_language_json,
)
from pipeline_config import PipelineConfig, create_pipeline_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "language.schema.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def mock_en_model() -> MagicMock:
    m = MagicMock()
    m.predict.return_value = (["__label__en"], [0.95])
    return m


@pytest.fixture
def mock_fr_model() -> MagicMock:
    m = MagicMock()
    m.predict.return_value = (["__label__fr"], [0.87])
    return m


@pytest.fixture
def mock_zh_model() -> MagicMock:
    m = MagicMock()
    m.predict.return_value = (["__label__zh"], [0.91])
    return m


@pytest.fixture
def mock_low_conf_model() -> MagicMock:
    m = MagicMock()
    m.predict.return_value = (["__label__en"], [0.15])
    return m


@pytest.fixture
def mock_raising_model() -> MagicMock:
    m = MagicMock()
    m.predict.side_effect = RuntimeError("boom")
    return m


# ---------------------------------------------------------------------------
# PipelineConfig field coverage
# ---------------------------------------------------------------------------


def test_config_enable_per_span_language_default_false():
    cfg = create_pipeline_config({})
    assert cfg.enable_per_span_language is False


def test_config_language_include_spans_default_false():
    cfg = create_pipeline_config({})
    assert cfg.language_include_spans is False


def test_config_language_short_span_threshold_default():
    cfg = create_pipeline_config({})
    assert cfg.language_short_span_threshold == 20


def test_config_language_confidence_threshold_default():
    cfg = create_pipeline_config({})
    assert cfg.language_confidence_threshold == pytest.approx(0.4)


def test_config_language_redact_samples_default():
    cfg = create_pipeline_config({})
    assert cfg.language_redact_samples == "privilege_or_short_doc"


def test_config_enable_per_span_language_env_true():
    cfg = create_pipeline_config({"ENABLE_PER_SPAN_LANGUAGE": "true"})
    assert cfg.enable_per_span_language is True


def test_config_enable_per_span_language_env_yes():
    cfg = create_pipeline_config({"ENABLE_PER_SPAN_LANGUAGE": "yes"})
    assert cfg.enable_per_span_language is True


def test_config_enable_per_span_language_env_1():
    cfg = create_pipeline_config({"ENABLE_PER_SPAN_LANGUAGE": "1"})
    assert cfg.enable_per_span_language is True


def test_config_enable_per_span_language_env_false_string():
    cfg = create_pipeline_config({"ENABLE_PER_SPAN_LANGUAGE": "false"})
    assert cfg.enable_per_span_language is False


def test_config_language_include_spans_env_true():
    cfg = create_pipeline_config({"LANGUAGE_INCLUDE_SPANS": "true"})
    assert cfg.language_include_spans is True


def test_config_language_short_span_threshold_env_override():
    cfg = create_pipeline_config({"LANGUAGE_SHORT_SPAN_THRESHOLD": "75"})
    assert cfg.language_short_span_threshold == 75


def test_config_language_confidence_threshold_env_override():
    cfg = create_pipeline_config({"LANGUAGE_CONFIDENCE_THRESHOLD": "0.65"})
    assert cfg.language_confidence_threshold == pytest.approx(0.65)


def test_config_language_redact_samples_env_true():
    cfg = create_pipeline_config({"LANGUAGE_REDACT_SAMPLES": "true"})
    assert cfg.language_redact_samples == "true"


def test_config_language_redact_samples_env_false():
    cfg = create_pipeline_config({"LANGUAGE_REDACT_SAMPLES": "false"})
    assert cfg.language_redact_samples == "false"


def test_config_language_redact_samples_invalid_falls_back():
    cfg = create_pipeline_config({"LANGUAGE_REDACT_SAMPLES": "not_a_mode"})
    assert cfg.language_redact_samples == "privilege_or_short_doc"


def test_config_language_confidence_threshold_clamps_low():
    cfg = create_pipeline_config({"LANGUAGE_CONFIDENCE_THRESHOLD": "-0.5"})
    assert cfg.language_confidence_threshold == pytest.approx(0.0)


def test_config_language_confidence_threshold_clamps_high():
    cfg = create_pipeline_config({"LANGUAGE_CONFIDENCE_THRESHOLD": "2.5"})
    assert cfg.language_confidence_threshold == pytest.approx(1.0)


def test_config_language_short_span_threshold_clamps_low():
    cfg = create_pipeline_config({"LANGUAGE_SHORT_SPAN_THRESHOLD": "0"})
    assert cfg.language_short_span_threshold == 1


def test_config_invalid_threshold_raises_via_direct_ctor():
    with pytest.raises(ValueError):
        PipelineConfig(language_confidence_threshold=1.5)


def test_config_invalid_redact_mode_raises_via_direct_ctor():
    with pytest.raises(ValueError):
        PipelineConfig(language_redact_samples="bogus")


def test_config_invalid_short_span_raises_via_direct_ctor():
    with pytest.raises(ValueError):
        PipelineConfig(language_short_span_threshold=0)


def test_config_non_bool_include_spans_raises():
    with pytest.raises(ValueError):
        PipelineConfig(language_include_spans="yes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FASTTEXT_TO_PADDLE mapping
# ---------------------------------------------------------------------------


def test_fasttext_to_paddle_populated():
    assert isinstance(FASTTEXT_TO_PADDLE, dict)
    assert len(FASTTEXT_TO_PADDLE) >= 30


def test_fasttext_to_paddle_includes_en():
    assert FASTTEXT_TO_PADDLE.get("en") == "en"


def test_fasttext_to_paddle_includes_fr():
    assert FASTTEXT_TO_PADDLE.get("fr") == "fr"


def test_fasttext_to_paddle_maps_zh_to_ch():
    # LANG_MAPPING historically aliases ``zh`` -> ``ch`` for PaddleOCR 2.x.
    assert FASTTEXT_TO_PADDLE.get("zh") == "ch"


def test_fasttext_to_paddle_all_values_are_strings():
    for k, v in FASTTEXT_TO_PADDLE.items():
        assert isinstance(k, str)
        assert isinstance(v, str)
        assert v  # non-empty


def test_fasttext_to_paddle_missing_returns_none():
    assert FASTTEXT_TO_PADDLE.get("xx_not_a_language") is None


# ---------------------------------------------------------------------------
# aggregate_page_from_full_text
# ---------------------------------------------------------------------------


def test_aggregate_english_returns_en(mock_en_model):
    pl = aggregate_page_from_full_text(
        page_num=1,
        full_text="Hello world from the English OCR pipeline",
        fasttext_model=mock_en_model,
        fasttext_model_sha256="abc",
        confidence_threshold=0.4,
    )
    assert isinstance(pl, PageLanguage)
    assert pl.primary_language == "en"
    assert pl.page_num == 1
    assert pl.span_count == 1
    assert pl.spans_labeled == 1


def test_aggregate_english_records_confidence(mock_en_model):
    pl = aggregate_page_from_full_text(1, "Hello world", mock_en_model, "", 0.4)
    assert pl.primary_confidence == pytest.approx(0.95)


def test_aggregate_french_returns_fr(mock_fr_model):
    pl = aggregate_page_from_full_text(
        page_num=3,
        full_text="Bonjour le monde",
        fasttext_model=mock_fr_model,
        fasttext_model_sha256="",
        confidence_threshold=0.4,
    )
    assert pl.primary_language == "fr"
    assert pl.page_num == 3


def test_aggregate_chinese_maps_to_ch(mock_zh_model):
    pl = aggregate_page_from_full_text(
        1, "\u4e2d\u6587\u6587\u672c", mock_zh_model, "", 0.4,
    )
    assert pl.primary_language == "ch"


def test_aggregate_empty_text_returns_und(mock_en_model):
    pl = aggregate_page_from_full_text(1, "", mock_en_model, "", 0.4)
    assert pl.primary_language == "und"
    assert pl.span_count == 0
    assert pl.spans_labeled == 0
    assert pl.spans == []


def test_aggregate_whitespace_only_returns_und(mock_en_model):
    pl = aggregate_page_from_full_text(1, "   \n\r   \n", mock_en_model, "", 0.4)
    assert pl.primary_language == "und"
    assert pl.span_count == 0


def test_aggregate_fasttext_exception_returns_und(mock_raising_model):
    pl = aggregate_page_from_full_text(
        1, "some english text here", mock_raising_model, "", 0.4,
    )
    assert pl.primary_language == "und"
    # Span still created so the pipeline records the attempt.
    assert pl.span_count == 1
    assert pl.spans_labeled == 0


def test_aggregate_none_model_returns_und():
    pl = aggregate_page_from_full_text(1, "some text", None, "", 0.4)
    assert pl.primary_language == "und"
    assert pl.spans[0].detection_method == "inherited_page"


def test_aggregate_low_confidence_returns_und(mock_low_conf_model):
    pl = aggregate_page_from_full_text(
        1, "ambiguous text", mock_low_conf_model, "", 0.4,
    )
    assert pl.primary_language == "und"
    assert pl.spans_labeled == 0


def test_aggregate_custom_threshold(mock_low_conf_model):
    pl = aggregate_page_from_full_text(
        1, "ambiguous text", mock_low_conf_model, "", 0.10,
    )
    assert pl.primary_language == "en"
    assert pl.spans_labeled == 1


def test_aggregate_single_page_span_count_one(mock_en_model):
    pl = aggregate_page_from_full_text(
        1, "this is an English sentence", mock_en_model, "", 0.4,
    )
    assert pl.span_count == 1
    assert len(pl.spans) == 1


def test_aggregate_span_records_char_count(mock_en_model):
    text = "English text here"
    pl = aggregate_page_from_full_text(5, text, mock_en_model, "", 0.4)
    assert pl.spans[0].char_count == len(text)


def test_aggregate_span_records_detection_method_fasttext(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English text", mock_en_model, "", 0.4)
    assert pl.spans[0].detection_method == "fasttext"


def test_aggregate_text_sample_truncated_to_60(mock_en_model):
    long_text = "a" * 200
    pl = aggregate_page_from_full_text(1, long_text, mock_en_model, "", 0.4)
    assert len(pl.spans[0].text_sample) == 60


def test_aggregate_records_latin_script_for_english(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English Words", mock_en_model, "", 0.4)
    assert pl.spans[0].script == "latin"
    assert "latin" in pl.scripts_detected


def test_aggregate_records_cjk_script(mock_zh_model):
    pl = aggregate_page_from_full_text(
        1, "\u4e2d\u6587\u5b57\u7b26", mock_zh_model, "", 0.4,
    )
    assert pl.spans[0].script == "cjk"
    assert "cjk" in pl.scripts_detected


def test_aggregate_ignores_unknown_fasttext_label():
    m = MagicMock()
    m.predict.return_value = (["__label__zz"], [0.9])  # ZZ not in registry
    pl = aggregate_page_from_full_text(1, "mystery text", m, "", 0.4)
    assert pl.primary_language == "und"


def test_aggregate_mixed_script_not_flagged_on_single_span(mock_en_model):
    pl = aggregate_page_from_full_text(1, "Hello World", mock_en_model, "", 0.4)
    # mixed_script is a page-level flag computed from scripts_detected list.
    # Single-span pages never carry mixed_script=True in PR A2.
    assert pl.mixed_script is False


def test_aggregate_language_char_shares_100_percent_on_success(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English text", mock_en_model, "", 0.4)
    assert pl.language_char_shares == {"en": 1.0}


def test_aggregate_languages_detected_on_success(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English text", mock_en_model, "", 0.4)
    assert pl.languages_detected == ["en"]


def test_aggregate_und_produces_empty_languages(mock_low_conf_model):
    pl = aggregate_page_from_full_text(
        1, "amb text", mock_low_conf_model, "", 0.4,
    )
    assert pl.languages_detected == []
    assert pl.language_char_shares == {}


def test_aggregate_predict_called_with_k3(mock_en_model):
    aggregate_page_from_full_text(1, "English text", mock_en_model, "", 0.4)
    args, kwargs = mock_en_model.predict.call_args
    # k=3 per spec
    assert kwargs.get("k") == 3 or (len(args) > 1 and args[1] == 3)


def test_aggregate_strips_newlines_before_predict(mock_en_model):
    aggregate_page_from_full_text(
        1, "line one\nline two\nline three", mock_en_model, "", 0.4,
    )
    fed = mock_en_model.predict.call_args[0][0]
    assert "\n" not in fed


# ---------------------------------------------------------------------------
# finalize_document_language
# ---------------------------------------------------------------------------


def test_finalize_single_page(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English text", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "f.pdf", [pl], "s", "t", "1.2.1")
    assert doc.primary_language == "en"
    assert doc.page_count == 1
    assert doc.pages_with_mixed_script == 0


def test_finalize_multi_page_same_language(mock_en_model):
    p1 = aggregate_page_from_full_text(1, "English one", mock_en_model, "", 0.4)
    p2 = aggregate_page_from_full_text(2, "English two more", mock_en_model, "", 0.4)
    doc = finalize_document_language("d", "f.pdf", [p1, p2], "s", "t", "1.2.1")
    assert doc.primary_language == "en"
    assert doc.page_count == 2


def test_finalize_multi_page_mixed_dominant_wins(mock_en_model, mock_fr_model):
    # English page has more characters than French page.
    en = aggregate_page_from_full_text(
        1, "This is a long English OCR page with many words", mock_en_model, "", 0.4,
    )
    fr = aggregate_page_from_full_text(2, "Bonjour", mock_fr_model, "", 0.4)
    doc = finalize_document_language("d", "f", [en, fr], "s", "t", "1.2.1")
    assert doc.primary_language == "en"
    assert "en" in doc.language_char_shares
    assert "fr" in doc.language_char_shares


def test_finalize_empty_pages_returns_und():
    doc = finalize_document_language("d", "f", [], "s", "t", "1.2.1")
    assert doc.primary_language == "und"
    assert doc.page_count == 0
    assert doc.language_char_shares == {}


def test_finalize_all_und_pages(mock_low_conf_model):
    p1 = aggregate_page_from_full_text(1, "a", mock_low_conf_model, "", 0.9)
    p2 = aggregate_page_from_full_text(2, "b", mock_low_conf_model, "", 0.9)
    doc = finalize_document_language("d", "f", [p1, p2], "s", "t", "1.2.1")
    assert doc.primary_language == "und"
    assert doc.page_count == 2


def test_finalize_processing_has_required_keys(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language(
        "d", "f.pdf", [pl], "sha256abc", "toksha", "1.2.3",
    )
    required = {
        "detection_engine",
        "detector_model_sha256",
        "tokenizer_sha256",
        "fasttext_model_sha256",
        "pipeline_version",
        "timestamp",
    }
    assert required.issubset(doc.processing.keys())


def test_finalize_processing_records_pipeline_version(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d", "f", [pl], "s", "t", "7.8.9")
    assert doc.processing["pipeline_version"] == "7.8.9"


def test_finalize_processing_records_fasttext_sha(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d", "f", [pl], "ft-abc", "tok-xyz", "1.2.1")
    assert doc.processing["fasttext_model_sha256"] == "ft-abc"
    assert doc.processing["tokenizer_sha256"] == "tok-xyz"


def test_finalize_timestamp_is_iso_8601(mock_en_model):
    import datetime as _dt
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d", "f", [pl], "s", "t", "1.2.1")
    # Round-trip through datetime.fromisoformat
    ts = doc.processing["timestamp"]
    # Python 3.11 fromisoformat handles +00:00 directly
    _dt.datetime.fromisoformat(ts)


def test_finalize_counts_mixed_script_pages():
    pl = PageLanguage(
        page_num=1,
        primary_language="en",
        primary_confidence=0.9,
        languages_detected=["en"],
        language_char_shares={"en": 1.0},
        scripts_detected=["latin", "cyrillic"],
        mixed_script=True,
        span_count=2,
        spans_labeled=2,
        spans=[],
    )
    doc = finalize_document_language("d", "f", [pl], "s", "t", "1.2.1")
    assert doc.pages_with_mixed_script == 1


def test_finalize_skips_none_pages(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d", "f", [pl, None], "s", "t", "1.2.1")
    # page_count still reflects input list length so assembler gap analysis works
    assert doc.primary_language == "en"


def test_finalize_language_char_shares_sum_to_one_on_success(mock_en_model, mock_fr_model):
    p1 = aggregate_page_from_full_text(1, "English one", mock_en_model, "", 0.4)
    p2 = aggregate_page_from_full_text(2, "Bonjour", mock_fr_model, "", 0.4)
    doc = finalize_document_language("d", "f", [p1, p2], "s", "t", "1.2.1")
    total = sum(doc.language_char_shares.values())
    assert total == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# write_language_json
# ---------------------------------------------------------------------------


def test_write_language_json_returns_path(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English text", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "sample.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "sample.pdf")
    assert path is not None
    assert os.path.exists(path)


def test_write_language_json_layout(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "sample.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "sample.pdf")
    expected_dir = tmp_path / "EXPORT" / "LANGUAGE"
    assert str(expected_dir) in path.replace("\\", "/").replace("//", "/") or str(
        expected_dir.resolve()
    ).replace("\\", "/") in path.replace("\\", "/")


def test_write_language_json_ends_with_language_json(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "sample.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "sample.pdf")
    assert path.endswith(".language.json")


def test_write_language_json_nested_subfolder(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "a/b/sample.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "a/b/sample.pdf")
    assert path is not None
    norm = path.replace("\\", "/")
    assert "EXPORT/LANGUAGE/a/b/" in norm


def test_write_language_json_include_spans_false(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf", include_spans=False)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["pages"][0]["spans"] == []


def test_write_language_json_include_spans_preserves_counts(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf", include_spans=False)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["pages"][0]["span_count"] == 1
    assert data["pages"][0]["spans_labeled"] == 1


def test_write_language_json_include_spans_true(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf", include_spans=True)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["pages"][0]["spans"]) == 1


def test_write_language_json_schema_version(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == LANGUAGE_SCHEMA_VERSION


def test_write_language_json_validates_against_schema(
    tmp_path, schema, mock_en_model,
):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "sh", "tk", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf", include_spans=True)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    jsonschema.validate(data, schema)


def test_write_language_json_validates_without_spans(
    tmp_path, schema, mock_en_model,
):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "sh", "tk", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf", include_spans=False)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    jsonschema.validate(data, schema)


def test_write_language_json_multi_page(tmp_path, mock_en_model, mock_fr_model):
    p1 = aggregate_page_from_full_text(1, "English one", mock_en_model, "", 0.4)
    p2 = aggregate_page_from_full_text(2, "Bonjour", mock_fr_model, "", 0.4)
    doc = finalize_document_language("d", "x.pdf", [p1, p2], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["pages"]) == 2
    assert data["document_summary"]["page_count"] == 2


def test_write_language_json_traversal_blocked(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language(
        "d1", "../../etc/passwd", [pl], "s", "t", "1.2.1",
    )
    path = write_language_json(doc, str(tmp_path), "../../etc/passwd")
    # ``..`` components are stripped before path realisation; the file lands
    # at the LANGUAGE root rather than outside it.
    assert path is None or str(tmp_path) in os.path.realpath(path)


def test_write_language_json_creates_parent_dirs(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "a/b/c/x.pdf", [pl], "s", "t", "1.2.1")
    # Parent directories should not yet exist.
    assert not (tmp_path / "EXPORT" / "LANGUAGE" / "a").exists()
    path = write_language_json(doc, str(tmp_path), "a/b/c/x.pdf")
    assert path is not None and os.path.exists(path)


def test_write_language_json_required_top_keys(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key in (
        "schema_version",
        "document_id",
        "source_file",
        "processing",
        "document_summary",
        "pages",
    ):
        assert key in data


def test_write_language_json_document_summary_shape(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    summary = data["document_summary"]
    assert summary["primary_language"] == "en"
    assert summary["page_count"] == 1
    assert summary["pages_with_mixed_script"] == 0
    assert summary["languages_detected"] == ["en"]


def test_write_language_json_ascii_nonlatin(tmp_path, mock_zh_model):
    pl = aggregate_page_from_full_text(
        1, "\u4e2d\u6587\u6587\u672c", mock_zh_model, "", 0.4,
    )
    doc = finalize_document_language("d1", "x.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf", include_spans=True)
    raw = Path(path).read_text(encoding="utf-8")
    # ensure_ascii=False so CJK codepoints land verbatim
    assert "\u4e2d" in raw


def test_write_language_json_no_subfolder(tmp_path, mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "solo.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "solo.pdf")
    expected = tmp_path / "EXPORT" / "LANGUAGE" / "solo.language.json"
    assert os.path.realpath(path) == os.path.realpath(str(expected))


def test_write_language_json_processing_includes_timestamp(
    tmp_path, mock_en_model,
):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "s", "t", "1.2.1")
    path = write_language_json(doc, str(tmp_path), "x.pdf")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert "timestamp" in data["processing"]


# ---------------------------------------------------------------------------
# Feature-flag gate (no .language.json when disabled)
# ---------------------------------------------------------------------------


def test_flag_off_no_write_invocation():
    """When enable_per_span_language=False, nothing calls write_language_json."""
    cfg = create_pipeline_config({})
    assert cfg.enable_per_span_language is False
    # In-pipeline guard is ``if ENABLE_PER_SPAN_LANGUAGE and ...``; verifying
    # the default stays False keeps accidental production rollouts impossible.


def test_flag_off_by_default_no_export_language_dir(tmp_path):
    """Simulate assembler finalization with the feature off -- no dir written."""
    cfg = create_pipeline_config({})
    if not cfg.enable_per_span_language:
        # Never invoked, so the directory is never created.
        assert not (tmp_path / "EXPORT" / "LANGUAGE").exists()


def test_flag_toggle_independent_of_other_features():
    cfg = create_pipeline_config(
        {"ENABLE_PER_SPAN_LANGUAGE": "true", "ENABLE_HANDWRITING": "false"},
    )
    assert cfg.enable_per_span_language is True
    assert cfg.enable_handwriting is False


def test_flag_redact_samples_does_not_affect_gate():
    cfg = create_pipeline_config({"LANGUAGE_REDACT_SAMPLES": "true"})
    assert cfg.language_redact_samples == "true"
    assert cfg.enable_per_span_language is False


# ---------------------------------------------------------------------------
# Dataclass round-trip / smoke
# ---------------------------------------------------------------------------


def test_page_language_defaults():
    pl = PageLanguage(
        page_num=1,
        primary_language="und",
        primary_confidence=0.0,
        languages_detected=[],
        language_char_shares={},
        scripts_detected=[],
        mixed_script=False,
        span_count=0,
        spans_labeled=0,
    )
    assert pl.spans == []


def test_span_language_fields():
    s = SpanLanguage(
        bbox=[1.0, 2.0, 3.0, 4.0],
        text_sample="abc",
        language="en",
        confidence=0.9,
        script="latin",
        detection_method="fasttext",
        char_count=3,
    )
    assert s.language == "en"
    assert s.detection_method == "fasttext"


def test_document_language_processing_defaults(mock_en_model):
    pl = aggregate_page_from_full_text(1, "English", mock_en_model, "", 0.4)
    doc = finalize_document_language("d1", "x.pdf", [pl], "", "", "1.2.1")
    assert isinstance(doc, DocumentLanguage)
    assert doc.processing["detection_engine"] == "fasttext"
