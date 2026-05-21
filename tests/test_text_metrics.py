"""Tests for ocr_distributed.text_metrics — shared OCR quality metrics."""

from __future__ import annotations

import unicodedata

import pytest

from ocr_distributed.text_metrics import (
    accuracy_summary,
    character_error_rate,
    edit_distance,
    line_accuracy,
    word_error_rate,
)

# ---------------------------------------------------------------------------
# edit_distance
# ---------------------------------------------------------------------------


class TestEditDistance:
    """Tests for the Levenshtein edit distance function."""

    def test_identical_strings(self):
        assert edit_distance("hello", "hello") == 0

    def test_completely_different(self):
        assert edit_distance("abc", "xyz") == 3

    def test_empty_both(self):
        assert edit_distance("", "") == 0

    def test_empty_first(self):
        assert edit_distance("", "abc") == 3

    def test_empty_second(self):
        assert edit_distance("abc", "") == 3

    def test_single_insertion(self):
        assert edit_distance("abc", "abcd") == 1

    def test_single_deletion(self):
        assert edit_distance("abcd", "abc") == 1

    def test_single_substitution(self):
        assert edit_distance("abc", "axc") == 1

    def test_symmetry(self):
        """Edit distance is symmetric: d(a,b) == d(b,a)."""
        assert edit_distance("kitten", "sitting") == edit_distance("sitting", "kitten")

    def test_known_value_kitten_sitting(self):
        """Classic Levenshtein example: kitten -> sitting = 3."""
        assert edit_distance("kitten", "sitting") == 3

    def test_unicode_characters(self):
        assert edit_distance("cafe\u0301", "cafe\u0301") == 0

    def test_different_lengths(self):
        """Longer string produces distance equal to extra characters if prefix matches."""
        assert edit_distance("a", "abcdef") == 5

    def test_space_optimisation(self):
        """Verify result is correct for strings where s1 < s2 (triggers swap)."""
        assert edit_distance("ab", "abcde") == 3


# ---------------------------------------------------------------------------
# character_error_rate
# ---------------------------------------------------------------------------


class TestCharacterErrorRate:
    """Tests for CER computation."""

    def test_perfect_match(self):
        assert character_error_rate("hello", "hello") == 0.0

    def test_some_errors(self):
        # "helo" vs "hello": 1 deletion, len(ref)=5
        cer = character_error_rate("hello", "helo")
        assert cer == pytest.approx(0.2)

    def test_all_errors(self):
        # Completely different, same length: 3 substitutions / 3 chars
        cer = character_error_rate("abc", "xyz")
        assert cer == pytest.approx(1.0)

    def test_both_empty(self):
        assert character_error_rate("", "") == 0.0

    def test_ref_empty_hyp_not(self):
        assert character_error_rate("", "abc") == 1.0

    def test_hyp_empty_ref_not(self):
        # 5 deletions / 5 chars = 1.0
        assert character_error_rate("hello", "") == 1.0

    def test_nfc_normalization_effect(self):
        """NFC normalization merges combining sequences."""
        # e + combining acute accent => e-acute (1 char)
        ref_decomposed = "caf\u0065\u0301"  # "cafe" + combining accent
        ref_composed = unicodedata.normalize("NFC", ref_decomposed)
        # Without normalization, they might differ; with normalization, equal
        assert character_error_rate(ref_decomposed, ref_composed, normalize=True) == 0.0

    def test_no_normalization(self):
        """Without NFC, combining characters may produce a non-zero CER."""
        ref = "caf\u00e9"  # precomposed e-acute
        hyp = "caf\u0065\u0301"  # decomposed e + combining accent
        cer = character_error_rate(ref, hyp, normalize=False)
        # Decomposed has one extra character, so CER > 0
        assert cer > 0.0

    def test_returns_float(self):
        result = character_error_rate("test", "text")
        assert isinstance(result, float)

    def test_can_exceed_one(self):
        """CER can exceed 1.0 when hypothesis has many insertions."""
        cer = character_error_rate("a", "abcdef")
        assert cer > 1.0


# ---------------------------------------------------------------------------
# word_error_rate
# ---------------------------------------------------------------------------


class TestWordErrorRate:
    """Tests for WER computation."""

    def test_perfect_match(self):
        assert word_error_rate("the quick fox", "the quick fox") == 0.0

    def test_one_word_error(self):
        # 3 words, 1 substitution
        wer = word_error_rate("the quick fox", "the slow fox")
        assert wer == pytest.approx(1.0 / 3.0)

    def test_both_empty(self):
        assert word_error_rate("", "") == 0.0

    def test_ref_empty_hyp_not(self):
        assert word_error_rate("", "some words") == 1.0

    def test_extra_words(self):
        """Hypothesis has more words than reference."""
        wer = word_error_rate("hello world", "hello big wide world")
        assert wer > 0.0

    def test_missing_words(self):
        """Hypothesis is missing words from reference."""
        wer = word_error_rate("one two three", "one three")
        assert wer == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# line_accuracy
# ---------------------------------------------------------------------------


class TestLineAccuracy:
    """Tests for line accuracy computation."""

    def test_perfect_match(self):
        text = "line one\nline two\nline three"
        assert line_accuracy(text, text) == 1.0

    def test_all_different(self):
        assert line_accuracy("a\nb\nc", "x\ny\nz") == 0.0

    def test_both_empty(self):
        assert line_accuracy("", "") == 1.0

    def test_partial_match(self):
        ref = "a\nb\nc\nd"
        hyp = "a\nx\nc\ny"
        # 4 lines, 2 match (a, c)
        assert line_accuracy(ref, hyp) == pytest.approx(0.5)

    def test_extra_lines_in_hypothesis(self):
        ref = "a"
        hyp = "a\nb\nc"
        # 3 total (max), 1 match
        assert line_accuracy(ref, hyp) == pytest.approx(1.0 / 3.0)

    def test_fewer_lines_in_hypothesis(self):
        ref = "a\nb\nc"
        hyp = "a"
        # 3 total (max), 1 match
        assert line_accuracy(ref, hyp) == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# accuracy_summary
# ---------------------------------------------------------------------------


class TestAccuracySummary:
    """Tests for the combined accuracy_summary function."""

    def test_returns_dict_with_expected_keys(self):
        result = accuracy_summary("hello world", "hello world")
        assert set(result.keys()) == {"cer", "wer", "line_accuracy", "char_count", "word_count"}

    def test_perfect_match_values(self):
        result = accuracy_summary("hello world", "hello world")
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert result["line_accuracy"] == 1.0
        assert result["char_count"] == 11
        assert result["word_count"] == 2

    def test_char_count_matches_reference(self):
        ref = "some reference text"
        result = accuracy_summary(ref, "some modified text")
        assert result["char_count"] == len(ref)

    def test_word_count_matches_reference(self):
        ref = "one two three four"
        result = accuracy_summary(ref, "one two four")
        assert result["word_count"] == 4
