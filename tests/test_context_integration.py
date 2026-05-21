"""Integration tests for context_store and context_merger modules.

Tests the full lifecycle of the 5-page context windowing system:
- ContextStore CRUD with ephemeral page contexts
- 5-page sliding window construction (prev/current/next + 2 summaries)
- ContextMerger cross-page paragraph and table continuation detection
- Store-to-merger end-to-end workflows
- TTL/expiry simulation
- Concurrent access patterns
- Error handling for missing pages, expired context, edge cases

These tests use an in-memory FakeRedis to avoid requiring a real Redis server.

Run with: python -m pytest tests/test_context_integration.py -v
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

import pytest

from coordinator.jobs.context_merger import (
    ContextMerger,
    _extract_first_paragraph,
    _extract_last_paragraph,
    detect_continued_paragraph,
    detect_continued_table,
    merge_paragraphs,
    merge_table_rows,
)
from coordinator.jobs.context_store import (
    ContextStore,
    _redis_key,
)

# ---------------------------------------------------------------------------
# FakeRedis -- in-memory Redis substitute for testing
# ---------------------------------------------------------------------------


class FakeRedis:
    """Thread-safe in-memory Redis stand-in for integration testing.

    Supports set/get/delete/sadd/smembers/expire operations with
    TTL-based expiry simulation.
    """

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._expiry: dict[str, float] = {}
        self._sets: dict[str, set[bytes]] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        if isinstance(value, str):
            value = value.encode("utf-8")
        with self._lock:
            self._store[key] = value
            if ex is not None:
                self._expiry[key] = time.time() + ex
        return True

    def get(self, key: str) -> bytes | None:
        with self._lock:
            if key in self._expiry and time.time() > self._expiry[key]:
                self._store.pop(key, None)
                self._expiry.pop(key, None)
                return None
            return self._store.get(key)

    def delete(self, *keys: str) -> int:
        count = 0
        with self._lock:
            for key in keys:
                if key in self._store:
                    del self._store[key]
                    count += 1
                if key in self._sets:
                    del self._sets[key]
                    count += 1
                self._expiry.pop(key, None)
        return count

    def sadd(self, key: str, *values: Any) -> int:
        added = 0
        with self._lock:
            if key not in self._sets:
                self._sets[key] = set()
            for v in values:
                if isinstance(v, str):
                    v = v.encode("utf-8")
                if v not in self._sets[key]:
                    self._sets[key].add(v)
                    added += 1
        return added

    def smembers(self, key: str) -> set[bytes]:
        with self._lock:
            return set(self._sets.get(key, set()))

    def expire(self, key: str, seconds: int) -> bool:
        with self._lock:
            if key in self._store or key in self._sets:
                self._expiry[key] = time.time() + seconds
                return True
        return False

    def keys_count(self) -> int:
        """Return total number of stored keys (data + sets)."""
        with self._lock:
            return len(self._store) + len(self._sets)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis():
    """Provide a fresh in-memory FakeRedis instance."""
    return FakeRedis()


@pytest.fixture
def store(fake_redis):
    """ContextStore wired to FakeRedis with explicit config."""
    s = ContextStore(
        redis_url="redis://localhost:6379/0",
        ttl_seconds=3600,
        window_size=5,
        _redis_client=fake_redis,
    )
    s._enabled = True
    return s


@pytest.fixture
def merger():
    """ContextMerger instance."""
    return ContextMerger()


def _make_pages(n: int) -> list[dict[str, Any]]:
    """Generate n pages of synthetic document data."""
    pages = []
    for i in range(1, n + 1):
        pages.append({
            "page_num": i,
            "text": (
                f"Content of page {i}. This page discusses topic {i} "
                f"in detail with supporting evidence and references."
            ),
            "ocr_confidence": 0.90 + (i % 5) * 0.02,
            "layout_regions": [
                {"type": "text", "bbox": [10, 10, 500, 700]},
            ],
        })
    return pages


@pytest.fixture
def pages_3():
    """3-page document."""
    return _make_pages(3)


@pytest.fixture
def pages_7():
    """7-page document."""
    return _make_pages(7)


@pytest.fixture
def pages_20():
    """20-page document for stress testing."""
    return _make_pages(20)


@pytest.fixture
def continuation_pages():
    """Pages specifically designed to test cross-page continuations."""
    return [
        {
            "page_num": 1,
            "text": (
                "The forensic analysis of document batch 2024-A began "
                "on January 15. The examiner noted several irregularities "
                "in the handwriting samples,\n\n"
                "A second paragraph discusses the ink composition "
                "analysis which revealed that the documents were pro-"
            ),
        },
        {
            "page_num": 2,
            "text": (
                "duced using a standard ballpoint pen manufactured "
                "after 2019. The chemical signature matched reference "
                "samples from the defendant's office.\n\n"
                "| Document ID | Date | Examiner | Status |\n"
                "| DOC-001 | 2024-01-15 | Smith | Complete |"
            ),
        },
        {
            "page_num": 3,
            "text": (
                "| DOC-002 | 2024-01-16 | Jones | Pending |\n"
                "| DOC-003 | 2024-01-17 | Smith | Complete |\n\n"
                "The final results are summarized in the appendix."
            ),
        },
    ]


@pytest.fixture
def table_pages():
    """Pages with structured table data for merge testing."""
    return [
        {
            "page_num": 1,
            "text": "| Name | Amount | Date |\n| Smith | $5000 | 2024-01-01 |",
            "table_rows": [
                ["Name", "Amount", "Date"],
                ["Smith", "$5000", "2024-01-01"],
            ],
        },
        {
            "page_num": 2,
            "text": "| Name | Amount | Date |\n| Jones | $3000 | 2024-01-02 |",
            "table_rows": [
                ["Name", "Amount", "Date"],
                ["Jones", "$3000", "2024-01-02"],
            ],
        },
        {
            "page_num": 3,
            "text": "Summary of all payments processed in Q1 2024.",
            "table_rows": None,
        },
    ]


# ===========================================================================
# Context Store Lifecycle Tests
# ===========================================================================


class TestContextStoreLifecycle:
    """Full lifecycle: create, store, retrieve, expire, cleanup."""

    def test_store_retrieve_delete_cycle(self, store):
        """Store a page context, retrieve it, then delete it."""
        ctx = {"text": "Page one content", "page_num": 1, "confidence": 0.95}
        ref_id = store.store_page_context("job-lifecycle", 1, ctx)

        # Retrieve
        result = store.get_page_context(ref_id)
        assert result is not None
        assert result["text"] == "Page one content"
        assert result["page_num"] == 1
        assert result["confidence"] == 0.95

        # Delete
        assert store.delete_context(ref_id) is True
        assert store.get_page_context(ref_id) is None

    def test_store_multiple_pages_cleanup_all(self, store):
        """Store context for multiple pages, then clean up by job ID."""
        job_id = str(uuid.uuid4())
        refs = []
        for page in range(1, 11):
            ref = store.store_page_context(
                job_id, page, {"text": f"page {page}", "page_num": page}
            )
            refs.append(ref)

        # All should be retrievable
        for ref in refs:
            assert store.get_page_context(ref) is not None

        # Cleanup by job ID
        deleted = store.cleanup_job(job_id)
        assert deleted >= 10

        # All should be gone
        for ref in refs:
            assert store.get_page_context(ref) is None

    def test_cleanup_does_not_affect_other_jobs(self, store):
        """Cleaning up one job should not touch another job's context."""
        job_a = str(uuid.uuid4())
        job_b = str(uuid.uuid4())

        ref_a = store.store_page_context(job_a, 1, {"text": "job A page 1"})
        ref_b = store.store_page_context(job_b, 1, {"text": "job B page 1"})

        store.cleanup_job(job_a)

        assert store.get_page_context(ref_a) is None
        assert store.get_page_context(ref_b) is not None
        assert store.get_page_context(ref_b)["text"] == "job B page 1"

    def test_expired_context_returns_none(self, fake_redis):
        """Context with very short TTL should expire."""
        store = ContextStore(
            redis_url="redis://localhost:6379/0",
            ttl_seconds=1,
            window_size=5,
            _redis_client=fake_redis,
        )
        store._enabled = True

        ref_id = store.store_page_context("job-exp", 1, {"text": "ephemeral"})
        assert store.get_page_context(ref_id) is not None

        # Simulate expiry by manipulating FakeRedis internal state
        key = _redis_key(ref_id)
        with fake_redis._lock:
            fake_redis._expiry[key] = time.time() - 1

        assert store.get_page_context(ref_id) is None

    def test_ref_id_uniqueness_across_pages_and_jobs(self, store):
        """Reference IDs must be unique for each store call."""
        seen = set()
        for job_num in range(5):
            job_id = f"job-{job_num}"
            for page_num in range(1, 6):
                ref = store.store_page_context(
                    job_id, page_num, {"page_num": page_num}
                )
                assert ref not in seen, f"Duplicate ref_id: {ref}"
                seen.add(ref)
        assert len(seen) == 25

    def test_overwrite_same_page_creates_new_ref(self, store):
        """Re-storing context for the same page creates a new ref ID."""
        ref1 = store.store_page_context("job-1", 1, {"version": 1})
        ref2 = store.store_page_context("job-1", 1, {"version": 2})
        assert ref1 != ref2

        # Both should be retrievable (they have different keys)
        ctx1 = store.get_page_context(ref1)
        ctx2 = store.get_page_context(ref2)
        assert ctx1["version"] == 1
        assert ctx2["version"] == 2


# ===========================================================================
# 5-Page Sliding Window Tests
# ===========================================================================


class TestSlidingWindow:
    """Verify the 5-page context window structure."""

    def test_window_structure_first_page(self, store, pages_7):
        """Page 1: no previous, no summary_before."""
        ctx = store.build_context_window("job-w", pages_7, 1)
        assert ctx["target_page_num"] == 1
        assert ctx["current"]["page_num"] == 1
        assert ctx["previous"] is None
        assert ctx["next"]["page_num"] == 2
        assert ctx["summary_before"] is None
        assert ctx["summary_after"] is not None

    def test_window_structure_second_page(self, store, pages_7):
        """Page 2: previous = page 1, no summary_before (no pages before page 1)."""
        ctx = store.build_context_window("job-w", pages_7, 2)
        assert ctx["target_page_num"] == 2
        assert ctx["previous"]["page_num"] == 1
        assert ctx["current"]["page_num"] == 2
        assert ctx["next"]["page_num"] == 3

    def test_window_structure_middle_page(self, store, pages_7):
        """Page 4 of 7: full window with summaries on both sides."""
        ctx = store.build_context_window("job-w", pages_7, 4)
        assert ctx["target_page_num"] == 4
        assert ctx["previous"]["page_num"] == 3
        assert ctx["current"]["page_num"] == 4
        assert ctx["next"]["page_num"] == 5
        assert ctx["summary_before"] is not None
        assert ctx["summary_before"]["page_count"] >= 1
        assert ctx["summary_after"] is not None
        assert ctx["summary_after"]["page_count"] >= 1

    def test_window_structure_last_page(self, store, pages_7):
        """Page 7: no next, no summary_after."""
        ctx = store.build_context_window("job-w", pages_7, 7)
        assert ctx["target_page_num"] == 7
        assert ctx["previous"]["page_num"] == 6
        assert ctx["current"]["page_num"] == 7
        assert ctx["next"] is None

    def test_window_single_page_doc(self, store):
        """Single-page document: no prev, no next, no summaries."""
        pages = [{"page_num": 1, "text": "Only page in the document."}]
        ctx = store.build_context_window("job-single", pages, 1)

        assert ctx["previous"] is None
        assert ctx["next"] is None
        assert ctx["current"]["text"] == "Only page in the document."
        assert ctx["total_pages"] == 1

    def test_window_two_page_doc(self, store):
        """Two-page document: page 1 has next only, page 2 has prev only."""
        pages = _make_pages(2)

        ctx1 = store.build_context_window("job-2p", pages, 1)
        assert ctx1["previous"] is None
        assert ctx1["next"]["page_num"] == 2

        ctx2 = store.build_context_window("job-2p", pages, 2)
        assert ctx2["previous"]["page_num"] == 1
        assert ctx2["next"] is None

    def test_build_all_windows(self, store, pages_7):
        """build_all_context_windows returns one ref_id per page."""
        ref_ids = store.build_all_context_windows("job-all", pages_7)
        assert len(ref_ids) == 7

        for i, ref_id in enumerate(ref_ids):
            ctx = store.get_page_context(ref_id)
            assert ctx is not None
            assert ctx["target_page_num"] == i + 1

    def test_build_all_windows_large_doc(self, store, pages_20):
        """20-page document: all windows should be valid."""
        ref_ids = store.build_all_context_windows("job-20p", pages_20)
        assert len(ref_ids) == 20

        # Spot-check first, middle, and last
        ctx_first = store.get_page_context(ref_ids[0])
        assert ctx_first["previous"] is None
        assert ctx_first["next"]["page_num"] == 2

        ctx_mid = store.get_page_context(ref_ids[9])
        assert ctx_mid["target_page_num"] == 10
        assert ctx_mid["previous"]["page_num"] == 9
        assert ctx_mid["next"]["page_num"] == 11
        assert ctx_mid["summary_before"] is not None
        assert ctx_mid["summary_after"] is not None

        ctx_last = store.get_page_context(ref_ids[19])
        assert ctx_last["previous"]["page_num"] == 19
        assert ctx_last["next"] is None

    def test_window_invalid_page_number(self, store, pages_7):
        """Out-of-range page numbers should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            store.build_context_window("job-err", pages_7, 0)

        with pytest.raises(ValueError, match="out of range"):
            store.build_context_window("job-err", pages_7, 8)

        with pytest.raises(ValueError, match="out of range"):
            store.build_context_window("job-err", pages_7, -1)

    def test_window_contains_timestamp(self, store, pages_3):
        """Each context window should contain a float timestamp."""
        ctx = store.build_context_window("job-ts", pages_3, 2)
        assert "timestamp" in ctx
        assert isinstance(ctx["timestamp"], float)
        assert ctx["timestamp"] > 0

    def test_window_contains_metadata(self, store, pages_7):
        """Context window includes job_id, total_pages, and window_size."""
        ctx = store.build_context_window("job-meta", pages_7, 3)
        assert ctx["job_id"] == "job-meta"
        assert ctx["total_pages"] == 7
        assert ctx["window_size"] == 5

    def test_summary_structure(self, store, pages_7):
        """Summaries should contain page_range, page_count, total_text_length, first_lines."""
        ctx = store.build_context_window("job-sum", pages_7, 4)
        for summary_key in ("summary_before", "summary_after"):
            summary = ctx[summary_key]
            if summary is not None:
                assert "page_range" in summary
                assert "page_count" in summary
                assert "total_text_length" in summary
                assert "first_lines" in summary
                assert isinstance(summary["page_range"], list)
                assert isinstance(summary["first_lines"], list)
                assert summary["page_count"] > 0
                assert summary["total_text_length"] > 0


# ===========================================================================
# Context Merger Integration Tests
# ===========================================================================


class TestContextMergerIntegration:
    """Test ContextMerger with realistic page data."""

    def test_paragraph_continuation_detected(self, store, merger, continuation_pages):
        """Hyphenated word break across pages should be detected as continuation."""
        ref_ids = store.build_all_context_windows("job-cont", continuation_pages)
        ctx = store.get_page_context(ref_ids[1])  # page 2

        result = merger.merge_from_context(ctx)
        assert result["page_num"] == 2
        assert result["paragraph_continuation_from_previous"] is True
        assert result["merged_text_prefix"] is not None

    def test_table_continuation_detected(self, store, merger, continuation_pages):
        """Table spanning pages 2-3 should be detected."""
        ref_ids = store.build_all_context_windows("job-tbl", continuation_pages)
        ctx = store.get_page_context(ref_ids[2])  # page 3

        result = merger.merge_from_context(ctx)
        assert result["page_num"] == 3
        assert result["table_continuation_from_previous"] is True

    def test_no_continuation_clean_boundaries(self, store, merger, pages_7):
        """Standard pages with clean sentence endings should not trigger continuation."""
        ref_ids = store.build_all_context_windows("job-clean", pages_7)
        # Pages in pages_7 end with periods, so no continuation expected
        for i, ref_id in enumerate(ref_ids):
            ctx = store.get_page_context(ref_id)
            result = merger.merge_from_context(ctx)
            assert result["page_num"] == i + 1

    def test_table_row_merge_with_header_dedup(self, store, merger, table_pages):
        """Table rows across pages should merge with header deduplication."""
        ref_ids = store.build_all_context_windows("job-tblmerge", table_pages)
        ctx = store.get_page_context(ref_ids[1])  # page 2

        result = merger.merge_from_context(ctx)
        assert result["table_continuation_from_previous"] is True
        assert result["merged_table_rows"] is not None

        # Header dedup: page 1 has ["Name", "Amount", "Date"] + 1 data row
        # page 2 has same header + 1 data row -> merged should have 3 rows
        merged = result["merged_table_rows"]
        assert len(merged) == 3
        assert merged[0] == ["Name", "Amount", "Date"]
        assert merged[1] == ["Smith", "$5000", "2024-01-01"]
        assert merged[2] == ["Jones", "$3000", "2024-01-02"]

    def test_merger_first_page_no_previous(self, store, merger, pages_3):
        """First page should have no continuation from previous."""
        ref_ids = store.build_all_context_windows("job-fp", pages_3)
        ctx = store.get_page_context(ref_ids[0])

        result = merger.merge_from_context(ctx)
        assert result["paragraph_continuation_from_previous"] is False
        assert result["table_continuation_from_previous"] is False
        assert result["merged_text_prefix"] is None

    def test_merger_last_page_no_next(self, store, merger, pages_3):
        """Last page should have no continuation to next."""
        ref_ids = store.build_all_context_windows("job-lp", pages_3)
        ctx = store.get_page_context(ref_ids[-1])

        result = merger.merge_from_context(ctx)
        assert result["paragraph_continuation_to_next"] is False
        assert result["table_continuation_to_next"] is False
        assert result["merged_text_suffix"] is None

    def test_merger_result_keys(self, store, merger, pages_3):
        """Merger result should contain all expected annotation keys."""
        ref_ids = store.build_all_context_windows("job-keys", pages_3)
        ctx = store.get_page_context(ref_ids[1])

        result = merger.merge_from_context(ctx)
        expected_keys = {
            "page_num",
            "paragraph_continuation_from_previous",
            "paragraph_continuation_to_next",
            "table_continuation_from_previous",
            "table_continuation_to_next",
            "merged_text_prefix",
            "merged_text_suffix",
            "merged_table_rows",
        }
        assert set(result.keys()) == expected_keys

    def test_merger_with_empty_context(self, merger):
        """Merger should handle context with no page data gracefully."""
        ctx = {
            "target_page_num": 1,
            "current": {},
            "previous": None,
            "next": None,
        }
        result = merger.merge_from_context(ctx)
        assert result["page_num"] == 1
        assert result["paragraph_continuation_from_previous"] is False
        assert result["paragraph_continuation_to_next"] is False

    def test_merger_with_missing_current(self, merger):
        """Merger handles missing 'current' key gracefully."""
        ctx = {
            "target_page_num": 1,
            "previous": None,
            "next": None,
        }
        result = merger.merge_from_context(ctx)
        assert result["page_num"] == 1


# ===========================================================================
# Cross-Page Paragraph Merging Tests
# ===========================================================================


class TestCrossPageParagraphMerging:
    """Detailed tests for paragraph merge scenarios."""

    def test_hyphenated_word_merge(self):
        """Hyphenated word at page break should be rejoined."""
        result = merge_paragraphs("The docu-", "ment was examined.")
        assert result == "The document was examined."

    def test_comma_continuation_merge(self):
        """Text ending with comma merges with next page."""
        result = merge_paragraphs(
            "The evidence included photographs,",
            "witness statements, and physical evidence."
        )
        assert "photographs," in result
        assert "witness statements" in result

    def test_merge_preserves_all_content(self):
        """Merged text should contain all words from both inputs."""
        current = "The forensic examiner concluded that"
        next_text = "the document was authentic."
        result = merge_paragraphs(current, next_text)
        for word in current.split():
            assert word in result
        for word in next_text.split():
            assert word in result

    def test_merge_empty_current(self):
        """Empty current text returns next text unchanged."""
        assert merge_paragraphs("", "Next page text.") == "Next page text."

    def test_merge_empty_next(self):
        """Empty next text returns current text unchanged."""
        assert merge_paragraphs("Current page text.", "") == "Current page text."

    def test_merge_both_empty(self):
        """Both empty returns empty."""
        assert merge_paragraphs("", "") == ""


# ===========================================================================
# Cross-Page Table Merging Tests
# ===========================================================================


class TestCrossPageTableMerging:
    """Table row merging across page breaks."""

    def test_simple_table_merge(self):
        """Two table fragments with no shared header merge directly."""
        current = [["A", "B"], ["1", "2"]]
        next_rows = [["3", "4"], ["5", "6"]]
        result = merge_table_rows(current, next_rows)
        assert len(result) == 4

    def test_repeated_header_skipped(self):
        """Repeated header row in next page should be skipped."""
        current = [["Name", "Value"], ["Alpha", "100"]]
        next_rows = [["Name", "Value"], ["Beta", "200"]]
        result = merge_table_rows(current, next_rows, skip_repeated_header=True)
        assert len(result) == 3
        assert result[2] == ["Beta", "200"]

    def test_case_insensitive_header_match(self):
        """Header comparison should be case-insensitive."""
        current = [["Name", "VALUE"], ["A", "1"]]
        next_rows = [["name", "value"], ["B", "2"]]
        result = merge_table_rows(current, next_rows, skip_repeated_header=True)
        assert len(result) == 3

    def test_different_headers_not_skipped(self):
        """Different headers should not be treated as duplicates."""
        current = [["Name", "Age"], ["John", "30"]]
        next_rows = [["City", "State"], ["NYC", "NY"]]
        result = merge_table_rows(current, next_rows, skip_repeated_header=True)
        assert len(result) == 4

    def test_skip_header_disabled(self):
        """When skip_repeated_header is False, all rows are kept."""
        current = [["H1", "H2"], ["A", "B"]]
        next_rows = [["H1", "H2"], ["C", "D"]]
        result = merge_table_rows(current, next_rows, skip_repeated_header=False)
        assert len(result) == 4

    def test_empty_current_rows(self):
        """Empty current rows return next rows."""
        result = merge_table_rows([], [["A", "B"]])
        assert result == [["A", "B"]]

    def test_empty_next_rows(self):
        """Empty next rows return current rows."""
        result = merge_table_rows([["A", "B"]], [])
        assert result == [["A", "B"]]


# ===========================================================================
# Continuation Detection Tests
# ===========================================================================


class TestContinuationDetection:
    """Paragraph and table continuation heuristics."""

    def test_period_ending_no_continuation(self):
        """Text ending with period should not be detected as continuation."""
        assert detect_continued_paragraph(
            "This sentence ends properly.",
            "A new sentence starts here."
        ) is False

    def test_comma_ending_continuation(self):
        """Text ending with comma is a strong continuation signal."""
        assert detect_continued_paragraph(
            "The items include apples,",
            "oranges, and bananas."
        ) is True

    def test_semicolon_ending_continuation(self):
        assert detect_continued_paragraph(
            "First clause;",
            "second clause follows."
        ) is True

    def test_hyphen_word_break_continuation(self):
        assert detect_continued_paragraph(
            "The investiga-",
            "tion concluded that..."
        ) is True

    def test_lowercase_start_continuation(self):
        """Next page starting with lowercase after non-terminal ending."""
        assert detect_continued_paragraph(
            "The analysis showed that",
            "results were consistent."
        ) is True

    def test_question_mark_no_continuation(self):
        assert detect_continued_paragraph(
            "Was the document authentic?",
            "The examiner confirmed it."
        ) is False

    def test_exclamation_no_continuation(self):
        assert detect_continued_paragraph(
            "The evidence is clear!",
            "Next section discusses."
        ) is False

    def test_empty_page_end_no_continuation(self):
        assert detect_continued_paragraph("", "Next page.") is False

    def test_empty_next_page_no_continuation(self):
        assert detect_continued_paragraph("Current page.", "") is False

    def test_pipe_table_continuation(self):
        """Pipe-delimited tables on both sides should trigger continuation."""
        assert detect_continued_table(
            "| Col1 | Col2 |\n| A | B |",
            "| C | D |\n| E | F |"
        ) is True

    def test_tab_table_continuation(self):
        """Tab-separated tables on both sides should trigger continuation."""
        assert detect_continued_table(
            "Col1\tCol2\nA\tB",
            "C\tD\nE\tF"
        ) is True

    def test_no_table_markers_no_continuation(self):
        assert detect_continued_table(
            "Regular paragraph text here.",
            "More paragraph text here."
        ) is False

    def test_table_only_one_side_no_continuation(self):
        """Table markers on only one side should not trigger."""
        assert detect_continued_table(
            "Just plain text.",
            "| Col1 | Col2 |"
        ) is False


# ===========================================================================
# End-to-End Workflow Tests
# ===========================================================================


class TestEndToEndWorkflow:
    """Full store -> window -> merge -> cleanup workflows."""

    def test_full_document_processing_flow(self, store, merger, continuation_pages):
        """Simulate processing a 3-page document with continuations."""
        job_id = str(uuid.uuid4())

        # Step 1: Build all context windows
        ref_ids = store.build_all_context_windows(job_id, continuation_pages)
        assert len(ref_ids) == 3

        # Step 2: Process each page through the merger
        merge_results = []
        for ref_id in ref_ids:
            ctx = store.get_page_context(ref_id)
            assert ctx is not None
            result = merger.merge_from_context(ctx)
            merge_results.append(result)

        # Step 3: Verify merge annotations
        assert merge_results[0]["page_num"] == 1
        assert merge_results[1]["page_num"] == 2
        assert merge_results[2]["page_num"] == 3

        # Page 2 should detect paragraph continuation from page 1
        assert merge_results[1]["paragraph_continuation_from_previous"] is True

        # Page 3 should detect table continuation from page 2
        assert merge_results[2]["table_continuation_from_previous"] is True

        # Step 4: Cleanup
        deleted = store.cleanup_job(job_id)
        assert deleted >= 3

        # Step 5: Verify cleanup
        for ref_id in ref_ids:
            assert store.get_page_context(ref_id) is None

    def test_celery_task_payload_size(self, store, pages_7):
        """Verify ref-ID-based task payloads stay under 10KB broker limit."""
        job_id = str(uuid.uuid4())
        ref_ids = store.build_all_context_windows(job_id, pages_7)

        for i, ref_id in enumerate(ref_ids):
            payload = {
                "job_id": job_id,
                "page_num": i + 1,
                "context_ref_id": ref_id,
                "presigned_urls": {
                    "source_get": "https://s3.example.com/bucket/source?sig=abc",
                    "page_pdf_put": "https://s3.example.com/bucket/page.pdf?sig=def",
                    "page_text_put": "https://s3.example.com/bucket/page.txt?sig=ghi",
                },
            }
            assert ContextStore.validate_broker_payload(payload) is True
            size = ContextStore.measure_payload_size(payload)
            assert size < 10240, f"Page {i+1} payload is {size} bytes"

    def test_worker_retrieves_and_processes_context(self, store, merger, pages_7):
        """Simulate a worker receiving a ref_id and processing the context."""
        job_id = str(uuid.uuid4())
        ref_ids = store.build_all_context_windows(job_id, pages_7)

        # Simulate worker receiving task with ref_id
        task_ref_id = ref_ids[3]  # page 4
        ctx = store.get_page_context(task_ref_id)
        assert ctx is not None
        assert ctx["target_page_num"] == 4

        # Worker runs merger
        merge_result = merger.merge_from_context(ctx)
        assert merge_result["page_num"] == 4

    def test_multiple_jobs_isolation(self, store, merger, pages_3, pages_7):
        """Multiple jobs should have independent context stores."""
        job_a = str(uuid.uuid4())
        job_b = str(uuid.uuid4())

        refs_a = store.build_all_context_windows(job_a, pages_3)
        refs_b = store.build_all_context_windows(job_b, pages_7)

        # Both should be accessible
        assert len(refs_a) == 3
        assert len(refs_b) == 7

        ctx_a = store.get_page_context(refs_a[0])
        ctx_b = store.get_page_context(refs_b[0])
        assert ctx_a["job_id"] == job_a
        assert ctx_b["job_id"] == job_b

        # Cleanup job A should not affect job B
        store.cleanup_job(job_a)
        for ref in refs_a:
            assert store.get_page_context(ref) is None
        for ref in refs_b:
            assert store.get_page_context(ref) is not None

        store.cleanup_job(job_b)

    def test_reprocessed_job_creates_new_refs(self, store, pages_3):
        """If a job is reprocessed, new ref IDs are generated."""
        job_id = str(uuid.uuid4())
        refs_first = store.build_all_context_windows(job_id, pages_3)
        refs_second = store.build_all_context_windows(job_id, pages_3)

        # All refs should be different
        assert set(refs_first).isdisjoint(set(refs_second))

        # Both sets should be retrievable
        for ref in refs_first + refs_second:
            assert store.get_page_context(ref) is not None


# ===========================================================================
# Concurrent Access Tests
# ===========================================================================


class TestConcurrentAccess:
    """Thread safety tests for the context store."""

    def test_concurrent_store_operations(self, store):
        """Multiple threads storing context concurrently should not lose data."""
        errors = []
        refs_by_thread: dict[int, list[str]] = {}

        def store_pages(thread_id: int):
            try:
                job_id = f"job-thread-{thread_id}"
                refs = []
                for page in range(1, 6):
                    ref = store.store_page_context(
                        job_id, page, {"thread": thread_id, "page": page}
                    )
                    refs.append(ref)
                refs_by_thread[thread_id] = refs
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_pages, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(refs_by_thread) == 5

        # Verify all stored contexts are retrievable
        for thread_id, refs in refs_by_thread.items():
            assert len(refs) == 5
            for ref in refs:
                ctx = store.get_page_context(ref)
                assert ctx is not None
                assert ctx["thread"] == thread_id

    def test_concurrent_store_and_cleanup(self, store):
        """Storing and cleaning up different jobs concurrently is safe."""
        job_to_clean = str(uuid.uuid4())
        job_to_keep = str(uuid.uuid4())

        # Pre-populate the job to clean
        for page in range(1, 6):
            store.store_page_context(
                job_to_clean, page, {"text": f"clean page {page}"}
            )

        errors = []
        keep_refs = []

        def store_keep_job():
            try:
                for page in range(1, 6):
                    ref = store.store_page_context(
                        job_to_keep, page, {"text": f"keep page {page}"}
                    )
                    keep_refs.append(ref)
            except Exception as e:
                errors.append(e)

        def cleanup_old_job():
            try:
                store.cleanup_job(job_to_clean)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=store_keep_job)
        t2 = threading.Thread(target=cleanup_old_job)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors
        # The kept job should still be fully accessible
        for ref in keep_refs:
            assert store.get_page_context(ref) is not None


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Error resilience and edge cases."""

    def test_missing_page_returns_none(self, store):
        """Getting a non-existent ref_id returns None, not an error."""
        result = store.get_page_context("ctx:nonexistent:1:abcdef12")
        assert result is None

    def test_delete_nonexistent_context(self, store):
        """Deleting a non-existent ref_id returns False."""
        assert store.delete_context("ctx:nonexistent:1:abcdef12") is False

    def test_cleanup_nonexistent_job(self, store):
        """Cleaning up a non-existent job returns 0."""
        assert store.cleanup_job("nonexistent-job-id") == 0

    def test_store_with_special_characters(self, store):
        """Context with unicode and special characters should round-trip."""
        ctx = {
            "text": "Forensic report: \u00e9\u00e8\u00ea \u00fc\u00f6\u00e4 \u4e2d\u6587 \u65e5\u672c\u8a9e",
            "page_num": 1,
            "notes": 'Contains "quotes" and \'apostrophes\'',
        }
        ref_id = store.store_page_context("job-unicode", 1, ctx)
        result = store.get_page_context(ref_id)
        assert result["text"] == ctx["text"]
        assert result["notes"] == ctx["notes"]

    def test_store_large_page_context(self, store):
        """Large page context (typical OCR output) should store and retrieve."""
        large_text = "Line of OCR text. " * 500  # ~9500 characters
        ctx = {
            "text": large_text,
            "page_num": 1,
            "ocr_lines": [
                (f"Line {i}", [10, i * 15, 500, (i + 1) * 15], 0.90)
                for i in range(100)
            ],
        }
        ref_id = store.store_page_context("job-large", 1, ctx)
        result = store.get_page_context(ref_id)
        assert result is not None
        assert len(result["text"]) == len(large_text)
        assert len(result["ocr_lines"]) == 100

    def test_merger_handles_none_text_fields(self, merger):
        """Merger should not crash when text fields are None."""
        ctx = {
            "target_page_num": 1,
            "current": {"text": None},
            "previous": {"text": None},
            "next": {"text": None},
        }
        result = merger.merge_from_context(ctx)
        assert result["paragraph_continuation_from_previous"] is False

    def test_merger_handles_empty_text_fields(self, merger):
        """Merger with empty strings should produce no continuations."""
        ctx = {
            "target_page_num": 1,
            "current": {"text": ""},
            "previous": {"text": ""},
            "next": {"text": ""},
        }
        result = merger.merge_from_context(ctx)
        assert result["paragraph_continuation_from_previous"] is False
        assert result["paragraph_continuation_to_next"] is False

    def test_detect_continued_paragraph_whitespace_only(self):
        """Whitespace-only text should not trigger continuation."""
        assert detect_continued_paragraph("   \n  ", "Next page.") is False
        assert detect_continued_paragraph("Current page.", "   \n  ") is False


# ===========================================================================
# Payload Size Validation Tests
# ===========================================================================


class TestPayloadSizeValidation:
    """Broker payload size constraints."""

    def test_ref_id_payload_under_limit(self):
        """A task payload with context ref_id stays under 10KB."""
        payload = {
            "job_id": str(uuid.uuid4()),
            "page_num": 42,
            "context_ref_id": "ctx:550e8400e29b:42:abcdef12",
            "presigned_urls": {
                "source_get": "https://s3.example.com/signed?token=abc123",
                "page_pdf_put": "https://s3.example.com/signed?token=def456",
                "page_text_put": "https://s3.example.com/signed?token=ghi789",
            },
        }
        assert ContextStore.validate_broker_payload(payload) is True

    def test_inline_data_payload_exceeds_limit(self):
        """A payload with inline page data should exceed the 10KB limit."""
        payload = {
            "job_id": str(uuid.uuid4()),
            "page_data": {
                "text": "x" * 15000,
                "ocr_lines": [("line", [0, 0, 100, 10], 0.9)] * 100,
            },
        }
        assert ContextStore.validate_broker_payload(payload, max_bytes=10240) is False

    def test_measure_payload_size_consistency(self):
        """measure_payload_size should match json.dumps byte length."""
        payload = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}
        expected = len(json.dumps(payload, default=str).encode("utf-8"))
        assert ContextStore.measure_payload_size(payload) == expected


# ===========================================================================
# Paragraph/Table Helper Tests
# ===========================================================================


class TestHelperFunctions:
    """Tests for paragraph and table extraction helpers."""

    def test_extract_last_paragraph_multi(self):
        """Extract last paragraph from multi-paragraph text."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        assert _extract_last_paragraph(text) == "Third paragraph."

    def test_extract_first_paragraph_multi(self):
        """Extract first paragraph from multi-paragraph text."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        assert _extract_first_paragraph(text) == "First paragraph."

    def test_extract_last_paragraph_single(self):
        assert _extract_last_paragraph("Only paragraph.") == "Only paragraph."

    def test_extract_first_paragraph_single(self):
        assert _extract_first_paragraph("Only paragraph.") == "Only paragraph."

    def test_extract_from_empty_string(self):
        assert _extract_last_paragraph("") == ""
        assert _extract_first_paragraph("") == ""

    def test_extract_with_whitespace_paragraphs(self):
        """Whitespace between paragraphs should be handled."""
        text = "Para one.\n\n   \n\nPara two."
        assert _extract_first_paragraph(text) == "Para one."
