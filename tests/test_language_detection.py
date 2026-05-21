"""Unit tests for language_detection module (Plan A -- PR A1).

Covers:
- SpanLanguage/PageLanguage/DocumentLanguage dataclass round-trip
- get_script_family() against all recognized Unicode blocks
- redact_text_sample() across privilege/token-count matrix
- schemas/language.schema.json structure and validation
- BCP-47 completeness + regex shape across the 45-language registry
- RTL subtag correctness for ar/fa/ur/ug
- Default collections on SpanLanguage.spans / PageLanguage
- Mixed-script aggregation invariants

Run with: python -m pytest tests/test_language_detection.py -v
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pytest

from language_detection import (
    SCRIPT_FAMILIES,
    DocumentLanguage,
    PageLanguage,
    SpanLanguage,
    get_script_family,
    redact_text_sample,
)
from ocr_local.config.language_config import LANGUAGE_REGISTRY, LanguageEntry

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "language.schema.json"

# BCP-47 shape: primary subtag (2-3 lowercase letters), optional Script (Title-case 4 letters),
# optional Region (2-3 uppercase letters OR 3 digits).
_BCP47_RE = re.compile(r"^[a-z]{2,3}(-[A-Z][a-zA-Z]{3})?(-[A-Z]{2,3}|-[0-9]{3})?$")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def language_schema() -> dict:
    """Load the .language.json JSON schema once per test."""
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def minimal_valid_payload() -> dict:
    """A minimum-valid payload that satisfies schemas/language.schema.json."""
    return {
        "schema_version": "1.0",
        "document_id": "doc-001",
        "source_file": "sample.pdf",
        "processing": {
            "detection_engine": "fasttext+script_heuristic",
            "detector_model_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "fasttext_model_sha256": "c" * 64,
            "pipeline_version": "1.2.1",
            "timestamp": "2026-04-24T12:00:00.000Z",
        },
        "document_summary": {
            "primary_language": "en",
            "primary_confidence": 0.93,
            "languages_detected": ["en"],
            "language_char_shares": {"en": 1.0},
            "page_count": 1,
            "pages_with_mixed_script": 0,
        },
        "pages": [
            {
                "page_num": 1,
                "primary_language": "en",
                "primary_confidence": 0.93,
                "languages_detected": ["en"],
                "language_char_shares": {"en": 1.0},
                "scripts_detected": ["latin"],
                "mixed_script": False,
                "span_count": 1,
                "spans_labeled": 1,
                "spans": [
                    {
                        "bbox": [10.0, 20.0, 100.0, 40.0],
                        "text_sample": "Hello world",
                        "language": "en",
                        "confidence": 0.93,
                        "script": "latin",
                        "detection_method": "fasttext",
                        "char_count": 11,
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# SpanLanguage round-trip
# ---------------------------------------------------------------------------


def test_span_language_construction():
    span = SpanLanguage(
        bbox=[0.0, 0.0, 10.0, 10.0],
        text_sample="hi",
        language="en",
        confidence=0.9,
        script="latin",
        detection_method="fasttext",
        char_count=2,
    )
    assert span.language == "en"
    assert span.char_count == 2
    assert span.detection_method == "fasttext"


def test_span_language_bbox_is_list():
    span = SpanLanguage(
        bbox=[1.0, 2.0, 3.0, 4.0],
        text_sample="x",
        language="en",
        confidence=0.5,
        script="latin",
        detection_method="fasttext",
        char_count=1,
    )
    assert isinstance(span.bbox, list)
    assert span.bbox == [1.0, 2.0, 3.0, 4.0]


def test_span_language_asdict_round_trip():
    span = SpanLanguage(
        bbox=[0.0, 0.0, 1.0, 1.0],
        text_sample="hello",
        language="en",
        confidence=0.88,
        script="latin",
        detection_method="fasttext",
        char_count=5,
    )
    d = asdict(span)
    assert d["language"] == "en"
    assert d["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert d["confidence"] == 0.88
    assert d["script"] == "latin"
    assert d["char_count"] == 5


def test_span_language_und_language():
    span = SpanLanguage(
        bbox=[0.0, 0.0, 1.0, 1.0],
        text_sample="",
        language="und",
        confidence=0.0,
        script="other",
        detection_method="script_heuristic",
        char_count=0,
    )
    assert span.language == "und"
    assert span.script == "other"


def test_span_language_detection_methods():
    for method in ("fasttext", "script_heuristic", "inherited_page"):
        span = SpanLanguage(
            bbox=[0.0, 0.0, 1.0, 1.0],
            text_sample="x",
            language="en",
            confidence=0.5,
            script="latin",
            detection_method=method,
            char_count=1,
        )
        assert span.detection_method == method


# ---------------------------------------------------------------------------
# PageLanguage round-trip
# ---------------------------------------------------------------------------


def test_page_language_construction():
    page = PageLanguage(
        page_num=1,
        primary_language="en",
        primary_confidence=0.9,
        languages_detected=["en"],
        language_char_shares={"en": 1.0},
        scripts_detected=["latin"],
        mixed_script=False,
        span_count=1,
        spans_labeled=1,
    )
    assert page.page_num == 1
    assert page.primary_language == "en"
    assert page.spans == []


def test_page_language_default_spans_is_empty_list():
    page = PageLanguage(
        page_num=2,
        primary_language="und",
        primary_confidence=0.0,
        languages_detected=[],
        language_char_shares={},
        scripts_detected=[],
        mixed_script=False,
        span_count=0,
        spans_labeled=0,
    )
    assert page.spans == []


def test_page_language_default_spans_is_independent_instance():
    # default_factory must return a fresh list per instance
    p1 = PageLanguage(
        page_num=1,
        primary_language="en",
        primary_confidence=1.0,
        languages_detected=["en"],
        language_char_shares={"en": 1.0},
        scripts_detected=["latin"],
        mixed_script=False,
        span_count=0,
        spans_labeled=0,
    )
    p2 = PageLanguage(
        page_num=2,
        primary_language="en",
        primary_confidence=1.0,
        languages_detected=["en"],
        language_char_shares={"en": 1.0},
        scripts_detected=["latin"],
        mixed_script=False,
        span_count=0,
        spans_labeled=0,
    )
    p1.spans.append(
        SpanLanguage(
            bbox=[0.0, 0.0, 1.0, 1.0],
            text_sample="x",
            language="en",
            confidence=0.5,
            script="latin",
            detection_method="fasttext",
            char_count=1,
        )
    )
    assert p2.spans == []


def test_page_language_mixed_script_single_script():
    page = PageLanguage(
        page_num=1,
        primary_language="en",
        primary_confidence=0.9,
        languages_detected=["en"],
        language_char_shares={"en": 1.0},
        scripts_detected=["latin"],
        mixed_script=False,
        span_count=3,
        spans_labeled=3,
    )
    assert page.mixed_script is False


def test_page_language_mixed_script_multi_script():
    page = PageLanguage(
        page_num=1,
        primary_language="ch",
        primary_confidence=0.8,
        languages_detected=["ch", "en"],
        language_char_shares={"ch": 0.7, "en": 0.3},
        scripts_detected=["cjk", "latin"],
        mixed_script=True,
        span_count=5,
        spans_labeled=5,
    )
    assert page.mixed_script is True
    assert len(page.scripts_detected) == 2


def test_page_language_language_char_shares_sum_plausible():
    shares = {"en": 0.6, "fr": 0.4}
    page = PageLanguage(
        page_num=1,
        primary_language="en",
        primary_confidence=0.9,
        languages_detected=["en", "fr"],
        language_char_shares=shares,
        scripts_detected=["latin"],
        mixed_script=False,
        span_count=10,
        spans_labeled=10,
    )
    assert abs(sum(page.language_char_shares.values()) - 1.0) < 1e-9


def test_page_language_asdict_round_trip():
    page = PageLanguage(
        page_num=7,
        primary_language="fr",
        primary_confidence=0.77,
        languages_detected=["fr"],
        language_char_shares={"fr": 1.0},
        scripts_detected=["latin"],
        mixed_script=False,
        span_count=2,
        spans_labeled=2,
    )
    d = asdict(page)
    assert d["page_num"] == 7
    assert d["primary_language"] == "fr"
    assert d["spans"] == []


# ---------------------------------------------------------------------------
# DocumentLanguage round-trip
# ---------------------------------------------------------------------------


def test_document_language_construction():
    doc = DocumentLanguage(
        document_id="doc-xyz",
        source_file="evidence.pdf",
        primary_language="en",
        primary_confidence=0.95,
        languages_detected=["en"],
        language_char_shares={"en": 1.0},
        page_count=1,
        pages_with_mixed_script=0,
        pages=[],
        processing={
            "engine": "fasttext+script_heuristic",
            "pipeline_version": "1.2.1",
            "detector_model_sha256": "x" * 64,
            "tokenizer_sha256": "y" * 64,
            "fasttext_model_sha256": "z" * 64,
            "timestamp": "2026-04-24T00:00:00Z",
        },
    )
    assert doc.document_id == "doc-xyz"
    assert doc.page_count == 1
    assert doc.pages == []


def test_document_language_pages_with_mixed_script_counts():
    p1 = PageLanguage(
        page_num=1,
        primary_language="en",
        primary_confidence=0.9,
        languages_detected=["en"],
        language_char_shares={"en": 1.0},
        scripts_detected=["latin"],
        mixed_script=False,
        span_count=5,
        spans_labeled=5,
    )
    p2 = PageLanguage(
        page_num=2,
        primary_language="ru",
        primary_confidence=0.85,
        languages_detected=["ru", "en"],
        language_char_shares={"ru": 0.6, "en": 0.4},
        scripts_detected=["cyrillic", "latin"],
        mixed_script=True,
        span_count=6,
        spans_labeled=6,
    )
    p3 = PageLanguage(
        page_num=3,
        primary_language="ch",
        primary_confidence=0.88,
        languages_detected=["ch", "en"],
        language_char_shares={"ch": 0.8, "en": 0.2},
        scripts_detected=["cjk", "latin"],
        mixed_script=True,
        span_count=4,
        spans_labeled=4,
    )
    pages = [p1, p2, p3]
    mixed_count = sum(1 for p in pages if p.mixed_script)
    doc = DocumentLanguage(
        document_id="doc-1",
        source_file="multilang.pdf",
        primary_language="ru",
        primary_confidence=0.85,
        languages_detected=["ru", "ch", "en"],
        language_char_shares={"ru": 0.4, "ch": 0.4, "en": 0.2},
        page_count=3,
        pages_with_mixed_script=mixed_count,
        pages=pages,
        processing={},
    )
    assert doc.pages_with_mixed_script == 2


def test_document_language_asdict_round_trip():
    doc = DocumentLanguage(
        document_id="d",
        source_file="f.pdf",
        primary_language="en",
        primary_confidence=1.0,
        languages_detected=["en"],
        language_char_shares={"en": 1.0},
        page_count=0,
        pages_with_mixed_script=0,
        pages=[],
        processing={"pipeline_version": "1.2.1"},
    )
    d = asdict(doc)
    assert d["document_id"] == "d"
    assert d["processing"]["pipeline_version"] == "1.2.1"


# ---------------------------------------------------------------------------
# get_script_family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ch", ["A", "Z", "a", "z", "M", "q"])
def test_script_family_latin_basic(ch):
    assert get_script_family(ch) == "latin"


@pytest.mark.parametrize("ch", ["\u00c0", "\u00e9", "\u0100", "\u024f"])
def test_script_family_latin_extended(ch):
    assert get_script_family(ch) == "latin"


@pytest.mark.parametrize("ch", ["\u0400", "\u042f", "\u044f", "\u04ff", "я", "ю"])
def test_script_family_cyrillic(ch):
    assert get_script_family(ch) == "cyrillic"


@pytest.mark.parametrize("ch", ["中", "日", "本", "\u4e00", "\u9fff"])
def test_script_family_cjk_ideographs(ch):
    assert get_script_family(ch) == "cjk"


@pytest.mark.parametrize("ch", ["\u3042", "\u30a2", "\u3040", "\u30ff"])
def test_script_family_cjk_kana(ch):
    assert get_script_family(ch) == "cjk"


@pytest.mark.parametrize("ch", ["\uac00", "\ud7a3", "한", "글"])
def test_script_family_cjk_hangul(ch):
    assert get_script_family(ch) == "cjk"


@pytest.mark.parametrize("ch", ["ع", "ب", "\u0600", "\u06ff", "\u0750", "\u077f"])
def test_script_family_arabic(ch):
    assert get_script_family(ch) == "arabic"


@pytest.mark.parametrize("ch", ["अ", "क", "\u0900", "\u097f"])
def test_script_family_devanagari(ch):
    assert get_script_family(ch) == "devanagari"


@pytest.mark.parametrize("ch", ["ა", "ბ", "\u10a0", "\u10ff"])
def test_script_family_georgian(ch):
    assert get_script_family(ch) == "georgian"


@pytest.mark.parametrize("ch", ["α", "β", "\u0370", "\u03ff"])
def test_script_family_greek(ch):
    assert get_script_family(ch) == "greek"


@pytest.mark.parametrize("ch", ["0", "9", "!", "?", " ", "$", "\t"])
def test_script_family_other_digits_and_punct(ch):
    assert get_script_family(ch) == "other"


def test_script_family_empty_string_returns_other():
    assert get_script_family("") == "other"


def test_script_family_multichar_returns_other():
    assert get_script_family("ab") == "other"


def test_script_family_unclassified_block_returns_other():
    # Runic block (U+16A0-U+16FF) is not in our classifier
    assert get_script_family("\u16a0") == "other"


def test_script_family_values_are_in_registry():
    for ch in ("A", "я", "中", "ع", "अ", "α", "ა", "0"):
        assert get_script_family(ch) in SCRIPT_FAMILIES


# ---------------------------------------------------------------------------
# redact_text_sample
# ---------------------------------------------------------------------------


def test_redact_privilege_flagged_returns_empty():
    assert redact_text_sample(
        "attorney client content", privilege_flagged=True, token_count=10_000
    ) == ""


def test_redact_below_token_threshold_returns_empty():
    assert redact_text_sample(
        "short doc content", privilege_flagged=False, token_count=499
    ) == ""


def test_redact_at_threshold_returns_sample():
    assert redact_text_sample(
        "threshold doc content",
        privilege_flagged=False,
        token_count=500,
    ) == "threshold doc content"


def test_redact_above_threshold_returns_first_60_chars():
    text = "A" * 200
    result = redact_text_sample(text, privilege_flagged=False, token_count=1_000)
    assert len(result) == 60
    assert result == "A" * 60


def test_redact_short_text_returned_as_is():
    result = redact_text_sample(
        "Hello", privilege_flagged=False, token_count=1_000
    )
    assert result == "Hello"


def test_redact_custom_threshold():
    result = redact_text_sample(
        "content", privilege_flagged=False, token_count=100, threshold=50
    )
    assert result == "content"


def test_redact_custom_threshold_below():
    assert redact_text_sample(
        "content", privilege_flagged=False, token_count=10, threshold=50
    ) == ""


def test_redact_privilege_wins_over_high_token_count():
    assert redact_text_sample(
        "sensitive", privilege_flagged=True, token_count=10_000_000
    ) == ""


def test_redact_none_text_returns_empty():
    result = redact_text_sample(
        None, privilege_flagged=False, token_count=1_000  # type: ignore[arg-type]
    )
    assert result == ""


def test_redact_exactly_60_chars_returns_all():
    text = "x" * 60
    assert redact_text_sample(
        text, privilege_flagged=False, token_count=1_000
    ) == text


# ---------------------------------------------------------------------------
# JSON schema validation
# ---------------------------------------------------------------------------


def test_schema_file_exists():
    assert SCHEMA_PATH.exists()


def test_schema_is_valid_draft07(language_schema):
    jsonschema.Draft7Validator.check_schema(language_schema)


def test_minimal_payload_validates(language_schema, minimal_valid_payload):
    jsonschema.validate(minimal_valid_payload, language_schema)


def test_missing_schema_version_fails(language_schema, minimal_valid_payload):
    del minimal_valid_payload["schema_version"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(minimal_valid_payload, language_schema)


def test_missing_processing_block_fails(language_schema, minimal_valid_payload):
    del minimal_valid_payload["processing"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(minimal_valid_payload, language_schema)


def test_missing_detector_sha_fails(language_schema, minimal_valid_payload):
    del minimal_valid_payload["processing"]["detector_model_sha256"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(minimal_valid_payload, language_schema)


def test_missing_page_primary_language_fails(language_schema, minimal_valid_payload):
    del minimal_valid_payload["pages"][0]["primary_language"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(minimal_valid_payload, language_schema)


def test_missing_span_bbox_fails(language_schema, minimal_valid_payload):
    del minimal_valid_payload["pages"][0]["spans"][0]["bbox"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(minimal_valid_payload, language_schema)


def test_bbox_wrong_length_fails(language_schema, minimal_valid_payload):
    minimal_valid_payload["pages"][0]["spans"][0]["bbox"] = [1.0, 2.0, 3.0]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(minimal_valid_payload, language_schema)


def test_document_summary_missing_primary_language_fails(
    language_schema, minimal_valid_payload
):
    del minimal_valid_payload["document_summary"]["primary_language"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(minimal_valid_payload, language_schema)


# ---------------------------------------------------------------------------
# BCP-47 registry validation
# ---------------------------------------------------------------------------


def test_registry_has_at_least_45_entries():
    assert len(LANGUAGE_REGISTRY) >= 45


def test_all_entries_have_nonempty_bcp47():
    missing = [code for code, entry in LANGUAGE_REGISTRY.items() if not entry.bcp47]
    assert missing == [], f"Missing bcp47 on entries: {missing}"


def test_all_bcp47_values_match_format():
    bad = []
    for code, entry in LANGUAGE_REGISTRY.items():
        if not _BCP47_RE.match(entry.bcp47):
            bad.append((code, entry.bcp47))
    assert bad == [], f"Bad BCP-47 tags: {bad}"


@pytest.mark.parametrize(
    "paddle_code,expected_bcp47",
    [
        ("en", "en"),
        ("fr", "fr"),
        ("german", "de"),
        ("es", "es"),
        ("it", "it"),
        ("pt", "pt"),
        ("nl", "nl"),
        ("ru", "ru"),
        ("uk", "uk"),
        ("be", "be"),
        ("bg", "bg"),
        ("ch", "zh-Hans"),
        ("chinese_cht", "zh-Hant"),
        ("japan", "ja"),
        ("korean", "ko"),
        ("ar", "ar"),
        ("fa", "fa"),
        ("ur", "ur"),
        ("ug", "ug"),
        ("hi", "hi"),
        ("ta", "ta"),
        ("te", "te"),
        ("kn", "kn"),
        ("el", "el"),
        ("ka", "ka"),
        ("rs_latin", "sr-Latn"),
        ("th", "th"),
        ("bn", "bn"),
        ("mr", "mr"),
        ("ne", "ne"),
    ],
)
def test_bcp47_mapping_exact(paddle_code, expected_bcp47):
    entry = LANGUAGE_REGISTRY[paddle_code]
    assert entry.bcp47 == expected_bcp47


@pytest.mark.parametrize("paddle_code", ["ar", "fa", "ur", "ug"])
def test_rtl_languages_flagged_and_bcp47_simple(paddle_code):
    entry = LANGUAGE_REGISTRY[paddle_code]
    assert entry.rtl is True
    # These Arabic-script languages use simple 2-letter BCP-47 tags.
    assert entry.bcp47 == paddle_code


def test_bcp47_zh_tags_use_script_subtag():
    assert LANGUAGE_REGISTRY["ch"].bcp47 == "zh-Hans"
    assert LANGUAGE_REGISTRY["chinese_cht"].bcp47 == "zh-Hant"


def test_bcp47_rs_latin_uses_script_subtag():
    assert LANGUAGE_REGISTRY["rs_latin"].bcp47 == "sr-Latn"


def test_language_entry_has_bcp47_field():
    # Validate the dataclass itself has the bcp47 attribute (not just via instances).
    assert "bcp47" in LanguageEntry.__dataclass_fields__


def test_bcp47_default_is_empty_string():
    assert LanguageEntry.__dataclass_fields__["bcp47"].default == ""


# ---------------------------------------------------------------------------
# SCRIPT_FAMILIES invariant
# ---------------------------------------------------------------------------


def test_script_families_tuple_contains_other():
    assert "other" in SCRIPT_FAMILIES


def test_script_families_tuple_deduped():
    assert len(set(SCRIPT_FAMILIES)) == len(SCRIPT_FAMILIES)


def test_script_families_is_tuple():
    assert isinstance(SCRIPT_FAMILIES, tuple)


# ---------------------------------------------------------------------------
# Shim pathway: importing via root module preserves identity
# ---------------------------------------------------------------------------


def test_root_shim_module_identity():
    import language_detection as root_mod
    import ocr_local.features.language_detection as pkg_mod

    assert root_mod is pkg_mod


def test_root_shim_preserves_private_symbol_access():
    # _BCP47_RE equivalent: ensure private-style imports from root still work.
    # We re-import to prove sys.modules replacement landed cleanly.
    import language_detection as lang_mod

    assert hasattr(lang_mod, "SCRIPT_FAMILIES")
    assert hasattr(lang_mod, "get_script_family")
    assert hasattr(lang_mod, "redact_text_sample")
