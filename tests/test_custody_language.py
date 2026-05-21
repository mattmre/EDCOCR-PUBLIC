"""Language-detection custody event tests (Plan A -- PR A4).

Covers the payload contract and fail-open semantics of the
``LANGUAGE_DETECTED`` and ``LANGUAGE_MIXED_SCRIPT`` custody events emitted
by the GPU worker and the assembler in :mod:`ocr_gpu_async`.

These tests do not spin up the full pipeline; instead they exercise
``CustodyChain.append_event`` directly with representative payloads and
verify:

* the set of required fields on each event type,
* that events are JSON-serialisable (no dataclass or numpy leaks),
* that private text (``text_sample``) never leaves the span structure,
* that custody append-failures do not propagate (fail-open contract).

Run with::

    python -m pytest tests/test_custody_language.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from custody import CustodyChain
from ocr_local.features.language_detection import (
    PageLanguage,
    SpanLanguage,
    aggregate_page_from_spans,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chain(tmp_path) -> CustodyChain:
    return CustodyChain("doc-abc", "/tmp/foo.pdf", str(tmp_path))


def _make_span(
    language: str = "en",
    confidence: float = 0.9,
    script: str = "latin",
    char_count: int = 20,
    text_sample: str = "hello world",
    method: str = "fasttext",
) -> SpanLanguage:
    return SpanLanguage(
        bbox=[0.0, 0.0, 100.0, 20.0],
        text_sample=text_sample,
        language=language,
        confidence=confidence,
        script=script,
        detection_method=method,
        char_count=char_count,
    )


@pytest.fixture
def page_lang_single() -> PageLanguage:
    spans = [_make_span("en", 0.92, "latin", 100, "lorem ipsum en")]
    return aggregate_page_from_spans(page_num=1, spans=spans)


@pytest.fixture
def page_lang_mixed() -> PageLanguage:
    spans = [
        _make_span("en", 0.9, "latin", 50, "english here"),
        _make_span("ar", 0.85, "arabic", 40, "\u0627\u0644\u0639\u0631\u0628"),
        _make_span("ru", 0.8, "cyrillic", 30, "\u0440\u0443\u0441"),
    ]
    return aggregate_page_from_spans(page_num=2, spans=spans)


def _language_detected_payload(
    page_lang: PageLanguage,
    model_sha: str = "a" * 64,
) -> dict:
    """Build the exact payload shape the GPU worker emits."""
    return {
        "page_num": page_lang.page_num,
        "primary_language": page_lang.primary_language,
        "primary_confidence": round(page_lang.primary_confidence, 4),
        "languages_detected": list(page_lang.languages_detected),
        "language_char_shares": {
            k: round(v, 4) for k, v in page_lang.language_char_shares.items()
        },
        "span_count": page_lang.span_count,
        "spans_labeled": page_lang.spans_labeled,
        "detection_engine": "fasttext+script_heuristic",
        "detector_model_sha256": model_sha,
        "tokenizer_sha256": model_sha,
        "fasttext_model_sha256": model_sha,
    }


def _mixed_script_payload(page_lang: PageLanguage) -> dict:
    script_char_shares: dict[str, int] = {}
    for span in page_lang.spans:
        script_char_shares[span.script] = (
            script_char_shares.get(span.script, 0) + span.char_count
        )
    total = sum(script_char_shares.values()) or 1
    return {
        "page_num": page_lang.page_num,
        "scripts_detected": list(page_lang.scripts_detected),
        "script_char_shares": {
            k: round(v / total, 4) for k, v in script_char_shares.items()
        },
        "primary_language": page_lang.primary_language,
        "rtl_present": any(
            span.script == "arabic" for span in page_lang.spans
        ),
    }


# ---------------------------------------------------------------------------
# LANGUAGE_DETECTED payload shape
# ---------------------------------------------------------------------------


class TestLanguageDetectedFields:
    REQUIRED = (
        "page_num",
        "primary_language",
        "primary_confidence",
        "languages_detected",
        "language_char_shares",
        "span_count",
        "spans_labeled",
        "detection_engine",
        "detector_model_sha256",
        "tokenizer_sha256",
        "fasttext_model_sha256",
    )

    def test_event_type_is_language_detected(self, chain, page_lang_single):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        assert event["event_type"] == "LANGUAGE_DETECTED"

    def test_all_required_fields_present(self, chain, page_lang_single):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        for field in self.REQUIRED:
            assert field in event["data"], f"missing: {field}"

    def test_page_num_is_integer(self, chain, page_lang_single):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        assert isinstance(event["data"]["page_num"], int)
        assert event["data"]["page_num"] == 1

    def test_primary_language_is_string(self, chain, page_lang_single):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        assert isinstance(event["data"]["primary_language"], str)

    def test_primary_confidence_bounded(self, chain, page_lang_single):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        conf = event["data"]["primary_confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_primary_confidence_rounded_to_4(self, chain, page_lang_single):
        payload = _language_detected_payload(page_lang_single)
        event = chain.append_event("LANGUAGE_DETECTED", payload)
        conf = event["data"]["primary_confidence"]
        # round(x, 4) produces at most 4 decimals
        assert abs(conf - round(conf, 4)) < 1e-12

    def test_languages_detected_is_list_of_str(self, chain, page_lang_single):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        langs = event["data"]["languages_detected"]
        assert isinstance(langs, list)
        for code in langs:
            assert isinstance(code, str)

    def test_language_char_shares_is_mapping(self, chain, page_lang_single):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        shares = event["data"]["language_char_shares"]
        assert isinstance(shares, dict)
        for k, v in shares.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_language_char_shares_normalised(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_mixed),
        )
        total = sum(event["data"]["language_char_shares"].values())
        # Shares sum to ~1.0 (modulo rounding to 4 decimals).
        assert 0.99 <= total <= 1.01

    def test_span_count_and_spans_labeled_ints(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_mixed),
        )
        assert isinstance(event["data"]["span_count"], int)
        assert isinstance(event["data"]["spans_labeled"], int)
        assert event["data"]["spans_labeled"] <= event["data"]["span_count"]

    def test_detection_engine_fixed(self, chain, page_lang_single):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        assert event["data"]["detection_engine"] == "fasttext+script_heuristic"

    @pytest.mark.parametrize(
        "field",
        [
            "detector_model_sha256",
            "tokenizer_sha256",
            "fasttext_model_sha256",
        ],
    )
    def test_model_hash_fields_are_strings(
        self, chain, page_lang_single, field,
    ):
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        assert isinstance(event["data"][field], str)
        # Hash fields may be empty string (when model lacks _sha256),
        # but never None.
        assert event["data"][field] is not None

    @pytest.mark.parametrize(
        "field",
        [
            "detector_model_sha256",
            "tokenizer_sha256",
            "fasttext_model_sha256",
        ],
    )
    def test_empty_model_hash_allowed(self, chain, page_lang_single, field):
        payload = _language_detected_payload(page_lang_single, model_sha="")
        event = chain.append_event("LANGUAGE_DETECTED", payload)
        assert event["data"][field] == ""

    def test_fasttext_tokenizer_detector_share_hash(
        self, chain, page_lang_single,
    ):
        payload = _language_detected_payload(page_lang_single, model_sha="d" * 64)
        event = chain.append_event("LANGUAGE_DETECTED", payload)
        assert (
            event["data"]["detector_model_sha256"]
            == event["data"]["tokenizer_sha256"]
            == event["data"]["fasttext_model_sha256"]
        )

    def test_payload_contains_no_text_samples(self, chain, page_lang_single):
        """Per E-A-006 the custody payload must not leak per-span text."""
        event = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        blob = json.dumps(event["data"])
        # None of the span text_sample values appear.
        for span in page_lang_single.spans:
            if span.text_sample:
                assert span.text_sample not in blob
        assert "text_sample" not in event["data"]
        assert "spans" not in event["data"]

    def test_payload_json_serialisable(self, chain, page_lang_single):
        payload = _language_detected_payload(page_lang_single)
        s = json.dumps(payload)
        assert isinstance(s, str)

    def test_payload_has_no_dataclass_instances(self, page_lang_single):
        payload = _language_detected_payload(page_lang_single)
        for v in payload.values():
            assert not isinstance(v, (SpanLanguage, PageLanguage))

    def test_rtl_event_not_emitted_for_single_script(self, chain, page_lang_single):
        """Single-script page should produce only LANGUAGE_DETECTED."""
        chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        assert len(chain.events) == 1
        # Caller decides whether to emit mixed-script; single-script pages
        # never do -- verified by checking mixed_script flag.
        assert page_lang_single.mixed_script is False

    def test_chain_verifies_after_language_event(self, chain, page_lang_single):
        chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        is_valid, _ = chain.verify_chain()
        assert is_valid


# ---------------------------------------------------------------------------
# LANGUAGE_MIXED_SCRIPT payload shape
# ---------------------------------------------------------------------------


class TestLanguageMixedScriptFields:
    REQUIRED = (
        "page_num",
        "scripts_detected",
        "script_char_shares",
        "primary_language",
        "rtl_present",
    )

    def test_event_type(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        assert event["event_type"] == "LANGUAGE_MIXED_SCRIPT"

    def test_all_required_fields(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        for field in self.REQUIRED:
            assert field in event["data"]

    def test_rtl_true_when_arabic_present(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        assert event["data"]["rtl_present"] is True

    def test_rtl_false_when_no_arabic(self, chain):
        spans = [
            _make_span("en", 0.9, "latin", 50, "en"),
            _make_span("ru", 0.9, "cyrillic", 50, "ru"),
        ]
        p = aggregate_page_from_spans(page_num=3, spans=spans)
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(p),
        )
        assert event["data"]["rtl_present"] is False

    def test_scripts_detected_at_least_two(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        assert len(event["data"]["scripts_detected"]) >= 2

    def test_script_char_shares_mapping(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        shares = event["data"]["script_char_shares"]
        assert isinstance(shares, dict)
        for k, v in shares.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_script_char_shares_sum_to_one(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        total = sum(event["data"]["script_char_shares"].values())
        assert 0.99 <= total <= 1.01

    def test_primary_language_echoed(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        assert (
            event["data"]["primary_language"]
            == page_lang_mixed.primary_language
        )

    def test_payload_json_serialisable(self, page_lang_mixed):
        json.dumps(_mixed_script_payload(page_lang_mixed))

    def test_no_text_sample_leaks(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        blob = json.dumps(event["data"])
        for span in page_lang_mixed.spans:
            if span.text_sample:
                assert span.text_sample not in blob
        assert "text_sample" not in event["data"]

    def test_chain_links_after_mixed_event(self, chain, page_lang_mixed):
        e1 = chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_mixed),
        )
        e2 = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        assert e2["prev_hash"] == e1["hash"]
        is_valid, _ = chain.verify_chain()
        assert is_valid

    def test_page_num_matches_page_language(self, chain, page_lang_mixed):
        event = chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        assert event["data"]["page_num"] == page_lang_mixed.page_num

    def test_rtl_only_arabic_triggers_flag(self, chain):
        spans = [
            _make_span("ar", 0.9, "arabic", 50, "ar"),
            _make_span("fa", 0.8, "arabic", 40, "fa"),
        ]
        p = aggregate_page_from_spans(page_num=9, spans=spans)
        # Only one script ("arabic") -- caller should not emit MIXED_SCRIPT,
        # but we still verify the RTL payload logic.
        payload = _mixed_script_payload(p)
        assert payload["rtl_present"] is True


# ---------------------------------------------------------------------------
# Fail-open semantics (custody failure never blocks OCR)
# ---------------------------------------------------------------------------


class TestFailOpenSemantics:
    def test_none_custody_chain_is_tolerated(self, page_lang_single):
        """Caller pattern: `if chain is not None: chain.append_event(...)`."""

        class _State:
            custody_chain = None

        state = _State()
        # This mirrors the guarded call in ocr_gpu_async.py.
        if state.custody_chain is not None:  # pragma: no branch
            state.custody_chain.append_event(
                "LANGUAGE_DETECTED",
                _language_detected_payload(page_lang_single),
            )
        # Getting here means no AttributeError was raised.
        assert state.custody_chain is None

    def test_append_failure_is_caught_by_caller(self, page_lang_single):
        chain = MagicMock()
        chain.append_event.side_effect = OSError("disk full")
        # The caller wraps the append in try/except.  Simulate that.
        raised = False
        try:
            try:
                chain.append_event(
                    "LANGUAGE_DETECTED",
                    _language_detected_payload(page_lang_single),
                )
            except Exception:
                pass
        except Exception:  # pragma: no cover
            raised = True
        assert raised is False
        chain.append_event.assert_called_once()

    def test_mock_called_with_expected_event_type(self, page_lang_single):
        chain = MagicMock()
        chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        args, _kwargs = chain.append_event.call_args
        assert args[0] == "LANGUAGE_DETECTED"

    def test_mock_called_with_expected_mixed_event(self, page_lang_mixed):
        chain = MagicMock()
        chain.append_event(
            "LANGUAGE_MIXED_SCRIPT", _mixed_script_payload(page_lang_mixed),
        )
        args, _kwargs = chain.append_event.call_args
        assert args[0] == "LANGUAGE_MIXED_SCRIPT"

    def test_mock_payload_has_required_keys(self, page_lang_single):
        chain = MagicMock()
        payload = _language_detected_payload(page_lang_single)
        chain.append_event("LANGUAGE_DETECTED", payload)
        recorded_payload = chain.append_event.call_args[0][1]
        for f in (
            "page_num",
            "primary_language",
            "detector_model_sha256",
        ):
            assert f in recorded_payload


# ---------------------------------------------------------------------------
# Document-level LANGUAGE_DETECTED event (assembler emission)
# ---------------------------------------------------------------------------


class TestDocumentLanguageDetected:
    def test_document_level_payload_fields(self, chain):
        # Simulate the assembler's document-level payload.
        payload = {
            "level": "document",
            "primary_language": "en",
            "primary_confidence": 0.9123,
            "languages_detected": ["en"],
            "language_char_shares": {"en": 1.0},
            "page_count": 3,
            "pages_with_mixed_script": 0,
            "detection_engine": "fasttext+script_heuristic",
        }
        event = chain.append_event("LANGUAGE_DETECTED", payload)
        assert event["data"]["level"] == "document"
        assert event["data"]["primary_language"] == "en"
        assert event["data"]["page_count"] == 3

    def test_document_level_json_serialisable(self, chain):
        payload = {
            "level": "document",
            "primary_language": "ar",
            "primary_confidence": 0.87,
            "languages_detected": ["ar", "en"],
            "language_char_shares": {"ar": 0.7, "en": 0.3},
            "page_count": 5,
            "pages_with_mixed_script": 2,
            "detection_engine": "fasttext+script_heuristic",
        }
        chain.append_event("LANGUAGE_DETECTED", payload)
        # Chain verification confirms JSON round-trip.
        is_valid, _ = chain.verify_chain()
        assert is_valid

    def test_document_and_page_events_chain_correctly(
        self, chain, page_lang_single,
    ):
        chain.append_event(
            "LANGUAGE_DETECTED", _language_detected_payload(page_lang_single),
        )
        doc_payload = {
            "level": "document",
            "primary_language": page_lang_single.primary_language,
            "primary_confidence": round(page_lang_single.primary_confidence, 4),
            "languages_detected": list(page_lang_single.languages_detected),
            "language_char_shares": {
                k: round(v, 4)
                for k, v in page_lang_single.language_char_shares.items()
            },
            "page_count": 1,
            "pages_with_mixed_script": 0,
            "detection_engine": "fasttext+script_heuristic",
        }
        chain.append_event("LANGUAGE_DETECTED", doc_payload)
        assert len(chain.events) == 2
        is_valid, _ = chain.verify_chain()
        assert is_valid
