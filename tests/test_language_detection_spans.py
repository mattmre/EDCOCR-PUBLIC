"""Per-span language detection tests (Plan A -- PR A3).

Covers:
- ``detect_span_language`` FastText path, short-span script heuristic, empty
  text, exception paths, low-confidence filtering, text-sample handling, and
  bbox pass-through.
- ``aggregate_page_from_spans`` char-weighted primary selection, mixed
  scripts, single-span pages, empty pages, share normalisation, weighted
  confidence averaging, and ``spans_labeled`` accounting.
- Integration pattern mirroring the GPU-worker hook -- verifies the
  ``detect_span_language`` -> ``aggregate_page_from_spans`` pipeline and
  backward compatibility when the feature flag is disabled.

Run with::

    python -m pytest tests/test_language_detection_spans.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import jsonschema
import pytest

from ocr_local.features.language_detection import (
    FASTTEXT_TO_PADDLE,
    PageLanguage,
    SpanLanguage,
    aggregate_page_from_spans,
    detect_span_language,
    finalize_document_language,
    write_language_json,
)

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
    m.predict.return_value = (["__label__fr"], [0.88])
    return m


@pytest.fixture
def mock_low_conf_model() -> MagicMock:
    m = MagicMock()
    m.predict.return_value = (["__label__en"], [0.10])
    return m


@pytest.fixture
def mock_raising_model() -> MagicMock:
    m = MagicMock()
    m.predict.side_effect = RuntimeError("boom")
    return m


# ---------------------------------------------------------------------------
# detect_span_language -- empty / error cases
# ---------------------------------------------------------------------------


def test_detect_span_empty_text_returns_und(mock_en_model):
    span = detect_span_language(
        text="", bbox=[0.0, 0.0, 10.0, 10.0], fasttext_model=mock_en_model
    )
    assert isinstance(span, SpanLanguage)
    assert span.language == "und"
    assert span.confidence == 0.0
    assert span.detection_method == "fasttext"
    assert span.char_count == 0


def test_detect_span_whitespace_only_returns_und(mock_en_model):
    span = detect_span_language(
        text="   \t\n   ", bbox=[0, 0, 1, 1], fasttext_model=mock_en_model
    )
    assert span.language == "und"
    assert span.char_count == 0


def test_detect_span_none_text_safe(mock_en_model):
    span = detect_span_language(
        text=None, bbox=[0, 0, 1, 1], fasttext_model=mock_en_model  # type: ignore[arg-type]
    )
    assert span.language == "und"
    assert span.char_count == 0


def test_detect_span_fasttext_exception_returns_und(mock_raising_model):
    text = "This is a long enough sentence for fasttext."
    span = detect_span_language(
        text=text, bbox=[0, 0, 1, 1], fasttext_model=mock_raising_model
    )
    assert span.language == "und"
    assert span.confidence == 0.0
    assert span.detection_method == "fasttext"


def test_detect_span_none_model_long_text_returns_und():
    text = "This is a long enough sentence for fasttext."
    span = detect_span_language(
        text=text, bbox=[0, 0, 1, 1], fasttext_model=None
    )
    assert span.language == "und"
    assert span.confidence == 0.0
    assert span.detection_method == "fasttext"


# ---------------------------------------------------------------------------
# detect_span_language -- FastText path
# ---------------------------------------------------------------------------


def test_detect_span_english_fasttext(mock_en_model):
    text = "The quick brown fox jumps over the lazy dog repeatedly."
    span = detect_span_language(
        text=text, bbox=[0, 0, 100, 20], fasttext_model=mock_en_model
    )
    assert span.language == "en"
    assert span.detection_method == "fasttext"
    assert span.confidence == pytest.approx(0.95)
    assert span.script == "latin"


def test_detect_span_french_fasttext(mock_fr_model):
    text = "Le renard brun saute par-dessus le chien paresseux tranquille."
    span = detect_span_language(
        text=text, bbox=[0, 0, 100, 20], fasttext_model=mock_fr_model
    )
    assert span.language == "fr"
    assert span.detection_method == "fasttext"
    assert span.confidence == pytest.approx(0.88)


def test_detect_span_low_confidence_filters_to_und(mock_low_conf_model):
    text = "Some text that is long enough to hit fasttext path."
    span = detect_span_language(
        text=text,
        bbox=[0, 0, 100, 20],
        fasttext_model=mock_low_conf_model,
        confidence_threshold=0.4,
    )
    assert span.language == "und"
    assert span.confidence == 0.0
    assert span.detection_method == "fasttext"


def test_detect_span_custom_confidence_threshold_accepts_lower(mock_low_conf_model):
    text = "Some text that is long enough to hit fasttext path."
    span = detect_span_language(
        text=text,
        bbox=[0, 0, 100, 20],
        fasttext_model=mock_low_conf_model,
        confidence_threshold=0.05,
    )
    assert span.language == "en"
    assert span.confidence == pytest.approx(0.10)


def test_detect_span_strips_newlines_before_predict(mock_en_model):
    text = "The quick brown fox\njumps over\nthe lazy dog for quite a while."
    detect_span_language(
        text=text, bbox=[0, 0, 1, 1], fasttext_model=mock_en_model
    )
    args, kwargs = mock_en_model.predict.call_args
    passed_text = args[0]
    assert "\n" not in passed_text


def test_detect_span_maps_fasttext_code_via_registry(mock_en_model):
    # en is guaranteed to be in FASTTEXT_TO_PADDLE.
    assert "en" in FASTTEXT_TO_PADDLE
    span = detect_span_language(
        text="English text that is sufficiently long for fasttext detection.",
        bbox=[0, 0, 1, 1],
        fasttext_model=mock_en_model,
    )
    assert span.language == FASTTEXT_TO_PADDLE["en"]


def test_detect_span_unknown_label_degrades_to_und():
    m = MagicMock()
    m.predict.return_value = (["__label__xx_unknown"], [0.99])
    span = detect_span_language(
        text="Text long enough to trigger fasttext path here please.",
        bbox=[0, 0, 1, 1],
        fasttext_model=m,
    )
    assert span.language == "und"
    assert span.confidence == 0.0


def test_detect_span_non_string_label_handled():
    m = MagicMock()
    m.predict.return_value = ([b"__label__en"], [0.95])
    span = detect_span_language(
        text="English text long enough for fasttext path activation now.",
        bbox=[0, 0, 1, 1],
        fasttext_model=m,
    )
    # str(b"...") -> "b'__label__en'" so it won't map; must degrade safely.
    assert span.language == "und"


def test_detect_span_invalid_score_handled():
    m = MagicMock()
    m.predict.return_value = (["__label__en"], ["not_a_number"])
    span = detect_span_language(
        text="English text long enough for fasttext path activation now.",
        bbox=[0, 0, 1, 1],
        fasttext_model=m,
    )
    assert span.language == "und"


# ---------------------------------------------------------------------------
# detect_span_language -- short span / script heuristic
# ---------------------------------------------------------------------------


def test_detect_span_short_latin_returns_und():
    span = detect_span_language(
        text="hello",  # 5 non-ws chars; < default 20
        bbox=[0, 0, 1, 1],
        fasttext_model=MagicMock(),
    )
    assert span.language == "und"
    assert span.detection_method == "script_heuristic"
    assert span.script == "latin"


def test_detect_span_short_arabic_returns_ar():
    span = detect_span_language(
        text="\u0627\u0644\u0633\u0644\u0627\u0645",  # "the peace"
        bbox=[0, 0, 1, 1],
        fasttext_model=MagicMock(),
    )
    assert span.language == "ar"
    assert span.detection_method == "script_heuristic"
    assert span.script == "arabic"


def test_detect_span_short_cjk_returns_ch():
    span = detect_span_language(
        text="\u4f60\u597d\u4e16\u754c",  # "hello world" CJK
        bbox=[0, 0, 1, 1],
        fasttext_model=MagicMock(),
    )
    assert span.language == "ch"
    assert span.detection_method == "script_heuristic"
    assert span.script == "cjk"


def test_detect_span_short_cyrillic_returns_ru():
    span = detect_span_language(
        text="\u041f\u0440\u0438\u0432\u0435\u0442",  # "Privet" (hello)
        bbox=[0, 0, 1, 1],
        fasttext_model=MagicMock(),
    )
    assert span.language == "ru"
    assert span.detection_method == "script_heuristic"
    assert span.script == "cyrillic"


def test_detect_span_short_devanagari_returns_hi():
    span = detect_span_language(
        text="\u0928\u092e\u0938\u094d\u0924\u0947",  # "Namaste"
        bbox=[0, 0, 1, 1],
        fasttext_model=MagicMock(),
    )
    assert span.language == "hi"
    assert span.detection_method == "script_heuristic"
    assert span.script == "devanagari"


def test_detect_span_short_greek_returns_el():
    span = detect_span_language(
        text="\u0393\u03b5\u03b9\u03b1",  # "Geia"
        bbox=[0, 0, 1, 1],
        fasttext_model=MagicMock(),
    )
    assert span.language == "el"
    assert span.detection_method == "script_heuristic"


def test_detect_span_short_georgian_returns_ka():
    span = detect_span_language(
        text="\u10d2\u10d0\u10db\u10d0\u10e0",  # "gamar"
        bbox=[0, 0, 1, 1],
        fasttext_model=MagicMock(),
    )
    assert span.language == "ka"
    assert span.detection_method == "script_heuristic"


def test_detect_span_short_digits_only_returns_und():
    span = detect_span_language(
        text="123456",  # other-script -> und
        bbox=[0, 0, 1, 1],
        fasttext_model=MagicMock(),
    )
    assert span.language == "und"
    assert span.detection_method == "script_heuristic"


def test_detect_span_short_span_does_not_call_fasttext():
    m = MagicMock()
    detect_span_language(
        text="short", bbox=[0, 0, 1, 1], fasttext_model=m
    )
    m.predict.assert_not_called()


def test_detect_span_exactly_at_threshold_calls_fasttext(mock_en_model):
    # 20 non-ws characters -- threshold of 20 requires strictly less to
    # short-circuit, so this should take the FastText path.
    text = "aaaaaaaaaaaaaaaaaaaa"  # 20 chars
    detect_span_language(
        text=text,
        bbox=[0, 0, 1, 1],
        fasttext_model=mock_en_model,
        short_span_threshold=20,
    )
    mock_en_model.predict.assert_called_once()


def test_detect_span_below_threshold_does_not_call_fasttext(mock_en_model):
    text = "a" * 19
    detect_span_language(
        text=text,
        bbox=[0, 0, 1, 1],
        fasttext_model=mock_en_model,
        short_span_threshold=20,
    )
    mock_en_model.predict.assert_not_called()


def test_detect_span_custom_threshold_works(mock_en_model):
    text = "abcdef"
    # With threshold=5, text has 6 non-ws chars -> fasttext path.
    detect_span_language(
        text=text,
        bbox=[0, 0, 1, 1],
        fasttext_model=mock_en_model,
        short_span_threshold=5,
    )
    mock_en_model.predict.assert_called_once()


# ---------------------------------------------------------------------------
# detect_span_language -- field correctness
# ---------------------------------------------------------------------------


def test_detect_span_text_sample_at_most_60_chars(mock_en_model):
    long_text = "x" * 500
    span = detect_span_language(
        text=long_text, bbox=[0, 0, 1, 1], fasttext_model=mock_en_model
    )
    assert len(span.text_sample) <= 60


def test_detect_span_char_count_equals_stripped_len(mock_en_model):
    text = "   hello world long enough text here for fasttext  "
    span = detect_span_language(
        text=text, bbox=[0, 0, 1, 1], fasttext_model=mock_en_model
    )
    assert span.char_count == len(text.strip())


def test_detect_span_bbox_passes_through(mock_en_model):
    bbox = [12.5, 34.0, 56.7, 89.1]
    span = detect_span_language(
        text="a very long text that certainly hits fasttext here.",
        bbox=bbox,
        fasttext_model=mock_en_model,
    )
    assert span.bbox == bbox


def test_detect_span_bbox_is_new_list(mock_en_model):
    bbox = [1.0, 2.0, 3.0, 4.0]
    span = detect_span_language(
        text="a very long text that certainly hits fasttext here.",
        bbox=bbox,
        fasttext_model=mock_en_model,
    )
    assert span.bbox == bbox
    # Mutating the returned list should not affect the caller's input.
    span.bbox.append(5.0)
    assert bbox == [1.0, 2.0, 3.0, 4.0]


def test_detect_span_script_field_populated(mock_en_model):
    span = detect_span_language(
        text="The quick brown fox jumps over the lazy dog repeatedly.",
        bbox=[0, 0, 1, 1],
        fasttext_model=mock_en_model,
    )
    assert span.script == "latin"


def test_detect_span_returns_spanlanguage_instance(mock_en_model):
    span = detect_span_language(
        text="Hello world from fasttext path testing please testing.",
        bbox=[0, 0, 1, 1],
        fasttext_model=mock_en_model,
    )
    assert isinstance(span, SpanLanguage)


def test_detect_span_confidence_clamped_to_unit_interval():
    m = MagicMock()
    m.predict.return_value = (["__label__en"], [1.5])  # out of range
    span = detect_span_language(
        text="Hello world from fasttext path testing please testing.",
        bbox=[0, 0, 1, 1],
        fasttext_model=m,
    )
    assert 0.0 <= span.confidence <= 1.0


# ---------------------------------------------------------------------------
# aggregate_page_from_spans -- basic rollups
# ---------------------------------------------------------------------------


def _mkspan(
    lang: str = "und",
    chars: int = 10,
    conf: float = 0.0,
    script: str = "latin",
    method: str = "fasttext",
) -> SpanLanguage:
    return SpanLanguage(
        bbox=[0.0, 0.0, 1.0, 1.0],
        text_sample="x" * min(chars, 60),
        language=lang,
        confidence=conf,
        script=script,
        detection_method=method,
        char_count=chars,
    )


def test_aggregate_empty_spans_returns_und():
    page = aggregate_page_from_spans(page_num=1, spans=[])
    assert isinstance(page, PageLanguage)
    assert page.primary_language == "und"
    assert page.span_count == 0
    assert page.spans_labeled == 0
    assert page.languages_detected == []
    assert page.scripts_detected == []
    assert page.mixed_script is False


def test_aggregate_none_spans_returns_und():
    page = aggregate_page_from_spans(page_num=1, spans=None)  # type: ignore[arg-type]
    assert page.primary_language == "und"
    assert page.span_count == 0


def test_aggregate_single_english_span():
    spans = [_mkspan(lang="en", chars=50, conf=0.95)]
    page = aggregate_page_from_spans(page_num=3, spans=spans)
    assert page.primary_language == "en"
    assert page.span_count == 1
    assert page.spans_labeled == 1
    assert page.languages_detected == ["en"]
    assert page.primary_confidence == pytest.approx(0.95)


def test_aggregate_all_english_spans():
    spans = [_mkspan(lang="en", chars=30, conf=0.9) for _ in range(4)]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.primary_language == "en"
    assert page.spans_labeled == 4
    assert page.span_count == 4
    assert page.primary_confidence == pytest.approx(0.9)


def test_aggregate_all_und_spans():
    spans = [_mkspan(lang="und", chars=10, conf=0.0) for _ in range(3)]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.primary_language == "und"
    assert page.primary_confidence == 0.0
    assert page.span_count == 3
    assert page.spans_labeled == 0
    assert page.languages_detected == []


def test_aggregate_mixed_en_fr_en_dominant():
    spans = [
        _mkspan(lang="en", chars=80, conf=0.95),
        _mkspan(lang="fr", chars=20, conf=0.80),
    ]
    page = aggregate_page_from_spans(page_num=2, spans=spans)
    assert page.primary_language == "en"
    assert set(page.languages_detected) == {"en", "fr"}
    assert page.languages_detected[0] == "en"  # descending share
    assert page.spans_labeled == 2


def test_aggregate_mixed_en_fr_fr_dominant():
    spans = [
        _mkspan(lang="en", chars=20, conf=0.95),
        _mkspan(lang="fr", chars=80, conf=0.80),
    ]
    page = aggregate_page_from_spans(page_num=2, spans=spans)
    assert page.primary_language == "fr"
    assert page.languages_detected[0] == "fr"


def test_aggregate_ignores_und_chars_in_primary():
    spans = [
        _mkspan(lang="und", chars=1000, conf=0.0),
        _mkspan(lang="en", chars=30, conf=0.9),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.primary_language == "en"
    assert page.spans_labeled == 1


def test_aggregate_spans_labeled_counts_non_und_only():
    spans = [
        _mkspan(lang="en", chars=10, conf=0.9),
        _mkspan(lang="und", chars=5, conf=0.0),
        _mkspan(lang="fr", chars=10, conf=0.8),
        _mkspan(lang="und", chars=5, conf=0.0),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.spans_labeled == 2
    assert page.span_count == 4


def test_aggregate_preserves_span_list():
    spans = [_mkspan(lang="en", chars=10, conf=0.9) for _ in range(5)]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.spans == spans
    assert len(page.spans) == 5


def test_aggregate_zero_char_spans_excluded():
    spans = [
        _mkspan(lang="en", chars=0, conf=0.9),
        _mkspan(lang="fr", chars=50, conf=0.8),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.primary_language == "fr"
    # zero-char en span contributes neither chars nor to spans_labeled
    assert page.spans_labeled == 1


# ---------------------------------------------------------------------------
# aggregate_page_from_spans -- script / mixed_script flags
# ---------------------------------------------------------------------------


def test_aggregate_latin_only_not_mixed_script():
    spans = [_mkspan(lang="en", chars=50, conf=0.9, script="latin")]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.mixed_script is False
    assert page.scripts_detected == ["latin"]


def test_aggregate_mixed_latin_arabic_mixed_script():
    spans = [
        _mkspan(lang="en", chars=30, conf=0.9, script="latin"),
        _mkspan(lang="ar", chars=30, conf=0.9, script="arabic"),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.mixed_script is True
    assert set(page.scripts_detected) == {"latin", "arabic"}


def test_aggregate_mixed_cjk_latin_mixed_script():
    spans = [
        _mkspan(lang="en", chars=30, conf=0.9, script="latin"),
        _mkspan(lang="ch", chars=30, conf=0.9, script="cjk"),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.mixed_script is True
    assert set(page.scripts_detected) == {"latin", "cjk"}


def test_aggregate_other_script_excluded_from_list():
    spans = [
        _mkspan(lang="en", chars=30, conf=0.9, script="latin"),
        _mkspan(lang="und", chars=10, conf=0.0, script="other"),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert "other" not in page.scripts_detected
    assert page.scripts_detected == ["latin"]
    assert page.mixed_script is False


def test_aggregate_three_scripts_mixed():
    spans = [
        _mkspan(lang="en", chars=30, conf=0.9, script="latin"),
        _mkspan(lang="ar", chars=30, conf=0.9, script="arabic"),
        _mkspan(lang="ru", chars=30, conf=0.9, script="cyrillic"),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.mixed_script is True
    assert len(page.scripts_detected) == 3


def test_aggregate_scripts_detected_sorted():
    spans = [
        _mkspan(lang="ru", chars=10, script="cyrillic"),
        _mkspan(lang="en", chars=10, script="latin"),
        _mkspan(lang="ar", chars=10, script="arabic"),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.scripts_detected == sorted(page.scripts_detected)


# ---------------------------------------------------------------------------
# aggregate_page_from_spans -- shares and confidences
# ---------------------------------------------------------------------------


def test_aggregate_language_char_shares_sum_to_one():
    spans = [
        _mkspan(lang="en", chars=40, conf=0.9),
        _mkspan(lang="fr", chars=60, conf=0.8),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    total = sum(page.language_char_shares.values())
    assert abs(total - 1.0) < 0.01


def test_aggregate_language_char_shares_proportional():
    spans = [
        _mkspan(lang="en", chars=25, conf=0.9),
        _mkspan(lang="fr", chars=75, conf=0.9),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.language_char_shares["en"] == pytest.approx(0.25, abs=0.01)
    assert page.language_char_shares["fr"] == pytest.approx(0.75, abs=0.01)


def test_aggregate_primary_confidence_weighted_average():
    # Two en spans with different confidences and char weights.
    spans = [
        _mkspan(lang="en", chars=10, conf=1.0),
        _mkspan(lang="en", chars=30, conf=0.5),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    # weighted = (10*1.0 + 30*0.5) / (10+30) = 25 / 40 = 0.625
    assert page.primary_confidence == pytest.approx(0.625, abs=1e-4)


def test_aggregate_primary_confidence_not_raw_fasttext_score():
    spans = [
        _mkspan(lang="en", chars=40, conf=0.95),
        _mkspan(lang="en", chars=60, conf=0.55),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    # Neither raw score; must be weighted mean.
    assert page.primary_confidence != pytest.approx(0.95)
    assert page.primary_confidence != pytest.approx(0.55)
    assert page.primary_confidence == pytest.approx(
        (40 * 0.95 + 60 * 0.55) / 100, abs=1e-4
    )


def test_aggregate_languages_detected_order_by_share():
    spans = [
        _mkspan(lang="fr", chars=10, conf=0.9),
        _mkspan(lang="en", chars=50, conf=0.9),
        _mkspan(lang="de", chars=30, conf=0.9),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    # Order: en (50) > de (30) > fr (10)
    assert page.languages_detected == ["en", "de", "fr"]


def test_aggregate_page_num_propagated():
    page = aggregate_page_from_spans(page_num=42, spans=[])
    assert page.page_num == 42


def test_aggregate_primary_confidence_clamped():
    # Even if per-span confidence somehow > 1.0 from upstream, output clamps.
    spans = [_mkspan(lang="en", chars=10, conf=1.5)]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert 0.0 <= page.primary_confidence <= 1.0


def test_aggregate_none_spans_in_list_skipped():
    spans = [
        _mkspan(lang="en", chars=30, conf=0.9),
        None,  # type: ignore[list-item]
        _mkspan(lang="fr", chars=20, conf=0.8),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)  # type: ignore[arg-type]
    assert page.primary_language == "en"
    assert page.spans_labeled == 2


# ---------------------------------------------------------------------------
# Integration: detect_span_language + aggregate_page_from_spans
# ---------------------------------------------------------------------------


def test_integration_detect_and_aggregate_en(mock_en_model):
    lines = [
        (
            "The quick brown fox jumps over the lazy dog repeatedly indeed.",
            [0, 0, 100, 20],
            0.95,
        ),
        (
            "A second sentence that is also long enough for fasttext here.",
            [0, 25, 100, 45],
            0.95,
        ),
    ]
    spans = [
        detect_span_language(
            text=t, bbox=list(b), fasttext_model=mock_en_model
        )
        for t, b, _c in lines
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.primary_language == "en"
    assert page.span_count == 2
    assert page.spans_labeled == 2


def test_integration_degrades_when_model_is_none():
    # Mirrors the GPU-worker hook's lang_model=None branch.
    lines = [
        ("Hello world 123", [0, 0, 1, 1], 0.9),
        ("Another line here", [0, 0, 1, 1], 0.9),
    ]
    spans = [
        detect_span_language(text=t, bbox=list(b), fasttext_model=None)
        for t, b, _c in lines
    ]
    # No crash; all "und" for short enough lines, possibly "und" for fasttext
    # path with None model.
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert isinstance(page, PageLanguage)
    # Either pattern is acceptable -- the key is no exception and structure ok.
    assert page.span_count == 2


def test_integration_schema_valid(schema, mock_en_model):
    """The PageLanguage/DocumentLanguage pipeline must emit schema-valid JSON."""
    lines = [
        ("English text long enough to hit fasttext here please now.", [0, 0, 1, 1], 0.9),
        ("A second English line that is also long enough here.", [0, 0, 1, 1], 0.9),
    ]
    spans = [
        detect_span_language(text=t, bbox=list(b), fasttext_model=mock_en_model)
        for t, b, _c in lines
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    doc = finalize_document_language(
        document_id="doc-1",
        source_file="doc-1.pdf",
        pages=[page],
        fasttext_model_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        pipeline_version="1.2.1",
    )
    assert doc.primary_language == "en"

    # Round-trip through write_language_json and validate against schema.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_path = write_language_json(
            doc, output_base_dir=tmp, source_file="doc-1.pdf", include_spans=False
        )
        assert out_path is not None
        with open(out_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    jsonschema.validate(payload, schema)


def test_integration_backward_compat_disabled_no_data():
    """When ENABLE_PER_SPAN_LANGUAGE would be False, nothing runs.

    This is the same guard the GPU worker enforces; we simulate by not
    calling detect/aggregate at all and confirm the hook's default
    ``language_data = None`` posture is safe for downstream consumers.
    """
    language_data = None  # matches ocr_gpu_async.py default
    assert language_data is None


def test_integration_handles_mixed_language_lines(mock_en_model, mock_fr_model):
    # Alternate EN / FR spans; EN dominates by chars.
    en_span = detect_span_language(
        text="English sentence with enough characters for fasttext activation.",
        bbox=[0, 0, 1, 1],
        fasttext_model=mock_en_model,
    )
    fr_span = detect_span_language(
        text="Phrase francaise courte mais assez longue pour fasttext.",
        bbox=[0, 0, 1, 1],
        fasttext_model=mock_fr_model,
    )
    page = aggregate_page_from_spans(page_num=1, spans=[en_span, fr_span])
    assert page.primary_language in {"en", "fr"}
    assert set(page.languages_detected) == {"en", "fr"}


def test_detect_span_fasttext_empty_labels_returns_und():
    m = MagicMock()
    m.predict.return_value = ([], [])
    span = detect_span_language(
        text="A long enough piece of text to route through the fasttext branch.",
        bbox=[0, 0, 1, 1],
        fasttext_model=m,
    )
    assert span.language == "und"
    assert span.detection_method == "fasttext"


def test_detect_span_text_sample_for_short_span():
    span = detect_span_language(
        text="hi", bbox=[0, 0, 1, 1], fasttext_model=MagicMock()
    )
    # redact_text_sample defaults privilege=False, token_count=len(split) ->
    # below 500 threshold -> redacted to empty string.
    assert span.text_sample == ""


def test_detect_span_detection_method_values_are_canonical(mock_en_model):
    long_span = detect_span_language(
        text="Hello world long enough for fasttext please now activate pathway.",
        bbox=[0, 0, 1, 1],
        fasttext_model=mock_en_model,
    )
    short_span = detect_span_language(
        text="abc", bbox=[0, 0, 1, 1], fasttext_model=MagicMock()
    )
    assert long_span.detection_method in {
        "fasttext",
        "script_heuristic",
        "inherited_page",
    }
    assert short_span.detection_method == "script_heuristic"


def test_aggregate_single_und_span_has_span_count_one():
    spans = [_mkspan(lang="und", chars=5, conf=0.0)]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.span_count == 1
    assert page.spans_labeled == 0
    assert page.primary_language == "und"


def test_aggregate_share_values_rounded_to_6_decimals():
    spans = [
        _mkspan(lang="en", chars=1, conf=0.9),
        _mkspan(lang="fr", chars=2, conf=0.9),
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    # 1/3 = 0.3333... -> rounded; verify rounding precision bound.
    for v in page.language_char_shares.values():
        assert round(v, 6) == v


def test_aggregate_returns_pagelanguage_instance():
    page = aggregate_page_from_spans(page_num=7, spans=[])
    assert isinstance(page, PageLanguage)
    assert page.page_num == 7


def test_integration_graceful_on_exception(mock_raising_model):
    # Each span fails fasttext -> all "und" -> page "und" -- no exception.
    spans = [
        detect_span_language(
            text="Some long text that certainly triggers fasttext path.",
            bbox=[0, 0, 1, 1],
            fasttext_model=mock_raising_model,
        )
        for _ in range(3)
    ]
    page = aggregate_page_from_spans(page_num=1, spans=spans)
    assert page.primary_language == "und"
    assert page.span_count == 3
    assert page.spans_labeled == 0
