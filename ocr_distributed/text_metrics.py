"""Shared text comparison metrics for OCR quality evaluation.

Consolidates edit distance, CER, WER, and line accuracy computations
into a single reusable module.  All functions are pure, stdlib-only, and
safe for use in both the monolithic pipeline and distributed workers.

Four independent implementations previously existed across the codebase:
  - scripts/corpus_validator.py   (TextComparator._levenshtein)
  - scripts/benchmark_handwriting.py (_edit_distance)
  - scripts/benchmark_comparison.py  (compute_edit_distance)
  - scripts/benchmark_accuracy.py    (levenshtein_distance)

This module replaces all four with a single canonical implementation.
"""

from __future__ import annotations

import unicodedata


def edit_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Uses Wagner-Fischer dynamic programming with O(min(m, n)) space by
    keeping only two rows at a time.  Always iterates over the shorter
    string in the inner loop for optimal memory usage.

    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.

    Returns
    -------
    int
        Minimum number of single-character insertions, deletions, or
        substitutions to transform *s1* into *s2*.
    """
    # Ensure s1 is the longer string so that the inner loop (s2) is
    # shorter, giving us O(min(m, n)) space.
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m = len(s1)
    n = len(s2)

    if n == 0:
        return m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]


def _sequence_edit_distance(seq1: list, seq2: list) -> int:
    """Levenshtein edit distance on arbitrary sequences (e.g. word lists).

    Same algorithm as :func:`edit_distance` but operates on list elements
    instead of individual characters.
    """
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    m = len(seq1)
    n = len(seq2)

    if n == 0:
        return m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]


def character_error_rate(
    reference: str, hypothesis: str, normalize: bool = True
) -> float:
    """Compute Character Error Rate (CER).

    CER = edit_distance(reference, hypothesis) / len(reference)

    Parameters
    ----------
    reference : str
        Ground truth text.
    hypothesis : str
        OCR output text.
    normalize : bool
        Apply Unicode NFC normalization before comparison (default True).

    Returns
    -------
    float
        CER as a float.  Returns 0.0 when both strings are empty.
        Returns 1.0 when *reference* is empty but *hypothesis* is not.
        Can exceed 1.0 when the hypothesis has many insertions.
    """
    if normalize:
        reference = unicodedata.normalize("NFC", reference)
        hypothesis = unicodedata.normalize("NFC", hypothesis)

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    dist = edit_distance(reference, hypothesis)
    return dist / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER).

    WER = word_edit_distance(ref_words, hyp_words) / len(ref_words)

    Splits on whitespace and uses word-level edit distance.

    Parameters
    ----------
    reference : str
        Ground truth text.
    hypothesis : str
        OCR output text.

    Returns
    -------
    float
        WER as a float.  Returns 0.0 when both are empty, 1.0 when
        *reference* is empty but *hypothesis* is not.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = _sequence_edit_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def line_accuracy(reference: str, hypothesis: str) -> float:
    """Compute the fraction of lines that match exactly.

    Lines are split on newlines and compared positionally.  Extra or
    missing lines count as mismatches.

    Parameters
    ----------
    reference : str
        Ground truth text.
    hypothesis : str
        OCR output text.

    Returns
    -------
    float
        Fraction of matching lines (0.0 to 1.0).  Returns 1.0 when both
        strings are empty.
    """
    ref_lines = reference.splitlines()
    hyp_lines = hypothesis.splitlines()

    if len(ref_lines) == 0 and len(hyp_lines) == 0:
        return 1.0

    total = max(len(ref_lines), len(hyp_lines))
    matches = 0
    for i in range(min(len(ref_lines), len(hyp_lines))):
        if ref_lines[i] == hyp_lines[i]:
            matches += 1

    return matches / total


def accuracy_summary(reference: str, hypothesis: str) -> dict:
    """Compute a full accuracy summary between reference and hypothesis.

    Returns a dictionary containing CER, WER, line accuracy, and basic
    character/word counts of the reference.

    Parameters
    ----------
    reference : str
        Ground truth text.
    hypothesis : str
        OCR output text.

    Returns
    -------
    dict
        Keys: ``cer``, ``wer``, ``line_accuracy``, ``char_count``,
        ``word_count``.
    """
    return {
        "cer": character_error_rate(reference, hypothesis),
        "wer": word_error_rate(reference, hypothesis),
        "line_accuracy": line_accuracy(reference, hypothesis),
        "char_count": len(reference),
        "word_count": len(reference.split()),
    }
