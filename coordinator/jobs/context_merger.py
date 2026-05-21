"""Cross-page data merger for context-aware OCR post-processing.

Uses the 5-page context window to detect and merge content that spans
page breaks:
- Tables that continue across pages
- Paragraphs split at page boundaries
- Sentence fragments that belong together

This module is opt-in and only activated when CONTEXT_WINDOW_ENABLED=true.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

# Minimum text overlap ratio to consider two fragments as the same sentence
_SENTENCE_CONTINUATION_MIN_WORDS = 3

# Patterns that indicate a table header row
_TABLE_HEADER_PATTERN = re.compile(
    r"^[\s|+\-=]+$"  # separator rows like "---+---" or "===|==="
)

# Characters that suggest a sentence was cut mid-flow
_SENTENCE_BREAK_CHARS = frozenset(",:;-")


# ---------------------------------------------------------------------------
# Content continuation detection
# ---------------------------------------------------------------------------

def detect_continued_paragraph(
    page_end_text: str,
    next_page_start_text: str,
) -> bool:
    """Detect whether a paragraph continues from one page to the next.

    Heuristics:
    - Previous page ends without terminal punctuation (.!?)
    - Previous page ends with a comma, semicolon, or hyphen
    - Next page starts with a lowercase letter (continuation)
    - Previous page's last line is a short fragment

    Args:
        page_end_text: The last ~500 characters of the current page.
        next_page_start_text: The first ~500 characters of the next page.

    Returns:
        True if the text likely continues across the page break.
    """
    if not page_end_text or not next_page_start_text:
        return False

    # Strip trailing whitespace
    end_stripped = page_end_text.rstrip()
    start_stripped = next_page_start_text.lstrip()

    if not end_stripped or not start_stripped:
        return False

    last_char = end_stripped[-1]

    # Strong signal: ends with continuation characters
    if last_char in _SENTENCE_BREAK_CHARS:
        return True

    # Strong signal: ends with hyphen (word break)
    if last_char == "-" and len(end_stripped) > 1 and end_stripped[-2].isalpha():
        return True

    # Next page starts with lowercase letter (continuation)
    if start_stripped[0].islower():
        # Also check that previous page doesn't end with sentence-terminal
        if last_char not in ".!?":
            return True

    # Previous page ends without terminal punctuation
    if last_char.isalpha() and last_char not in ".!?":
        # Check if the last line is short (< 60 chars), suggesting mid-sentence break
        last_line = end_stripped.split("\n")[-1].strip()
        if len(last_line) < 60:
            return True

    return False


def detect_continued_table(
    page_end_text: str,
    next_page_start_text: str,
) -> bool:
    """Detect whether a table continues from one page to the next.

    Heuristics:
    - Previous page ends with table-like structure (pipes, tabs, aligned columns)
    - Next page starts with table-like structure
    - Both pages have consistent column separators

    Args:
        page_end_text: The last ~500 characters of the current page.
        next_page_start_text: The first ~500 characters of the next page.

    Returns:
        True if a table likely spans the page break.
    """
    if not page_end_text or not next_page_start_text:
        return False

    end_lines = page_end_text.strip().split("\n")
    start_lines = next_page_start_text.strip().split("\n")

    if not end_lines or not start_lines:
        return False

    # Check for pipe-delimited table markers
    end_has_pipes = any("|" in line for line in end_lines[-3:])
    start_has_pipes = any("|" in line for line in start_lines[:3])

    if end_has_pipes and start_has_pipes:
        return True

    # Check for tab-separated columns
    end_has_tabs = any("\t" in line for line in end_lines[-3:])
    start_has_tabs = any("\t" in line for line in start_lines[:3])

    if end_has_tabs and start_has_tabs:
        return True

    # Check for consistent whitespace-aligned columns
    end_spaces = _count_consistent_spaces(end_lines[-3:])
    start_spaces = _count_consistent_spaces(start_lines[:3])

    if end_spaces >= 2 and start_spaces >= 2:
        return True

    return False


def _count_consistent_spaces(lines: list[str]) -> int:
    """Count positions where multiple lines have whitespace alignment.

    Returns the number of column-like positions found.
    """
    if len(lines) < 2:
        return 0

    # Find positions where spaces appear in multiple lines
    max_len = max(len(line) for line in lines)
    aligned_positions = 0

    for pos in range(1, min(max_len, 200)):
        spaces_at_pos = sum(
            1 for line in lines
            if pos < len(line) and line[pos] == " " and pos > 0 and line[pos - 1] != " "
        )
        if spaces_at_pos >= 2:
            aligned_positions += 1

    return aligned_positions


# ---------------------------------------------------------------------------
# Merging operations
# ---------------------------------------------------------------------------

def merge_paragraphs(
    current_text: str,
    next_text: str,
) -> str:
    """Merge two text fragments that span a page break.

    Handles:
    - Hyphenated word breaks (re-joins the word)
    - Simple concatenation with proper spacing

    Args:
        current_text: Text from the end of the current page.
        next_text: Text from the start of the next page.

    Returns:
        The merged text string.
    """
    if not current_text:
        return next_text
    if not next_text:
        return current_text

    end = current_text.rstrip()
    start = next_text.lstrip()

    # Handle hyphenated word break
    if end.endswith("-"):
        # Remove the hyphen and join directly
        return end[:-1] + start

    # Standard join with single space
    return end + " " + start


def merge_table_rows(
    current_rows: list[list[str]],
    next_rows: list[list[str]],
    *,
    skip_repeated_header: bool = True,
) -> list[list[str]]:
    """Merge table rows that span a page break.

    When ``skip_repeated_header`` is True, checks if the first row(s) of
    the next page duplicate a header row from the current page and skips
    them.

    Args:
        current_rows: Table rows (list of cell strings) from current page.
        next_rows: Table rows from the next page.
        skip_repeated_header: Whether to detect and skip repeated headers.

    Returns:
        Combined list of table rows.
    """
    if not current_rows:
        return list(next_rows)
    if not next_rows:
        return list(current_rows)

    result = list(current_rows)

    start_idx = 0
    if skip_repeated_header and len(current_rows) >= 1 and len(next_rows) >= 1:
        # Check if the first row of next page matches first row of current page
        # (repeated header)
        if _rows_match(current_rows[0], next_rows[0]):
            start_idx = 1
            logger.debug("Skipping repeated table header on page continuation")

            # Also skip separator rows following the repeated header
            if start_idx < len(next_rows):
                next_row_text = " ".join(next_rows[start_idx])
                if _TABLE_HEADER_PATTERN.match(next_row_text):
                    start_idx += 1

    result.extend(next_rows[start_idx:])
    return result


def _rows_match(row_a: list[str], row_b: list[str]) -> bool:
    """Check if two table rows have matching content (case-insensitive)."""
    if len(row_a) != len(row_b):
        return False
    return all(
        a.strip().lower() == b.strip().lower()
        for a, b in zip(row_a, row_b)
    )


# ---------------------------------------------------------------------------
# High-level merger
# ---------------------------------------------------------------------------

class ContextMerger:
    """Merge cross-page content using context window data.

    Takes a context window (from ContextStore) and identifies content
    that should be merged across page boundaries.
    """

    def merge_from_context(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze a context window and produce merge annotations.

        Args:
            context: A context window dict with ``current``, ``previous``,
                     and ``next`` page data.

        Returns:
            Dict with merge annotations:
            - ``paragraph_continuation``: bool
            - ``table_continuation``: bool
            - ``merged_text``: str or None (merged paragraph if applicable)
            - ``merged_table_rows``: list or None (merged rows if applicable)
            - ``page_num``: int
        """
        current = context.get("current") or {}
        previous = context.get("previous")
        next_page = context.get("next")
        page_num = context.get("target_page_num", 0)

        current_text = current.get("text", "")

        result: dict[str, Any] = {
            "page_num": page_num,
            "paragraph_continuation_from_previous": False,
            "paragraph_continuation_to_next": False,
            "table_continuation_from_previous": False,
            "table_continuation_to_next": False,
            "merged_text_prefix": None,
            "merged_text_suffix": None,
            "merged_table_rows": None,
        }

        # Check continuation from previous page
        if previous:
            prev_text = previous.get("text", "")
            if prev_text and current_text:
                # Use last 500 chars of prev, first 500 chars of current
                end_snippet = prev_text[-500:]
                start_snippet = current_text[:500]

                if detect_continued_paragraph(end_snippet, start_snippet):
                    result["paragraph_continuation_from_previous"] = True
                    # Merge the trailing text of previous with leading text of current
                    prev_last_para = _extract_last_paragraph(prev_text)
                    curr_first_para = _extract_first_paragraph(current_text)
                    result["merged_text_prefix"] = merge_paragraphs(
                        prev_last_para, curr_first_para,
                    )

                if detect_continued_table(end_snippet, start_snippet):
                    result["table_continuation_from_previous"] = True

                    # Merge table rows if structured data is available
                    prev_table = previous.get("table_rows")
                    curr_table = current.get("table_rows")
                    if prev_table and curr_table:
                        result["merged_table_rows"] = merge_table_rows(
                            prev_table, curr_table,
                        )

        # Check continuation to next page
        if next_page:
            next_text = next_page.get("text", "")
            if current_text and next_text:
                end_snippet = current_text[-500:]
                start_snippet = next_text[:500]

                if detect_continued_paragraph(end_snippet, start_snippet):
                    result["paragraph_continuation_to_next"] = True
                    curr_last_para = _extract_last_paragraph(current_text)
                    next_first_para = _extract_first_paragraph(next_text)
                    result["merged_text_suffix"] = merge_paragraphs(
                        curr_last_para, next_first_para,
                    )

                if detect_continued_table(end_snippet, start_snippet):
                    result["table_continuation_to_next"] = True

        return result


def _extract_last_paragraph(text: str) -> str:
    """Extract the last paragraph from text (split by double newlines)."""
    paragraphs = re.split(r"\n\s*\n", text.rstrip())
    return paragraphs[-1].strip() if paragraphs else ""


def _extract_first_paragraph(text: str) -> str:
    """Extract the first paragraph from text (split by double newlines)."""
    paragraphs = re.split(r"\n\s*\n", text.lstrip())
    return paragraphs[0].strip() if paragraphs else ""
